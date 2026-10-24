"""Ablation runner: one-factor-at-a-time around a baseline, plus optional factorial blocks.

Indexes are cached per chunking configuration (chunking is the only factor that changes the
index), so a grid of ~20 experiments costs a handful of ingestion passes. Every comparison with
the baseline is paired (same questions) and corrected for multiple testing.
"""

from __future__ import annotations

import itertools
import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ragqa import __version__
from ragqa.config import PipelineConfig, Settings
from ragqa.embeddings import Embedder, build_embedder
from ragqa.eval.dataset import EvalItem, dataset_fingerprint, filter_split, load_dataset, locate_evidence
from ragqa.eval.judge import LLMJudge
from ragqa.eval.runner import DEFAULT_KS, QuestionRecord, aggregate, run_items
from ragqa.eval.stats import holm_bonferroni, paired_bootstrap_diff, paired_permutation_test
from ragqa.generation.llm import LLM
from ragqa.ingest.loaders import load_path
from ragqa.pipeline import RAGPipeline, build_generator, build_llm
from ragqa.store import MemoryStore
from ragqa.types import Document


class VariabilitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperatures: list[float] = Field(default_factory=lambda: [0.0, 0.3, 0.7, 1.0])
    runs: int = 5
    limit: int | None = None


class AblationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "ablation"
    dataset: Path = Path("data/eval/qa.jsonl")
    corpus: Path = Path("data/corpus")
    split: str = "test"
    embedder: Literal["wordllama", "hashing", "openai"] = "wordllama"
    generator: Literal["extractive", "llm"] = "extractive"
    seed: int = 13
    judge: bool = Field(False, description="Add LLM-as-judge metrics (needs an LLM)")
    baseline: dict[str, Any] = Field(default_factory=dict)
    factors: dict[str, list[Any]] = Field(default_factory=dict)
    interactions: list[list[str]] = Field(default_factory=list)
    primary_metrics: list[str] = Field(default_factory=lambda: ["mrr", "overall_correct"])
    report_metrics: list[str] = Field(
        default_factory=lambda: [
            "recall@5",
            "mrr",
            "ndcg@5",
            "accuracy",
            "f1",
            "abstention_correct",
            "overall_correct",
            "citation_precision",
            "grounded",
            "injection_followed",
        ]
    )
    ks: list[int] = Field(default_factory=lambda: list(DEFAULT_KS))
    n_boot: int = 2000
    n_perm: int = 10000
    variability: VariabilitySpec | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> AblationSpec:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class Experiment:
    name: str
    overrides: dict[str, Any]
    config: PipelineConfig
    factor: str | None = None
    level: Any = None


@dataclass
class AblationResult:
    spec: AblationSpec
    experiments: list[Experiment]
    records: dict[str, list[QuestionRecord]]
    summaries: dict[str, dict[str, dict[str, float]]]
    comparisons: dict[str, dict[str, dict[str, float]]]
    factor_levels: dict[str, list[tuple[Any, str]]]
    interaction_tables: list[dict[str, Any]]
    manifest: dict[str, Any]
    variability: dict[str, Any] | None = None
    category_scores: dict[str, dict[str, float]] = field(default_factory=dict)


def _get(config: PipelineConfig, dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


def _label(value: Any) -> str:
    if isinstance(value, bool):
        return "on" if value else "off"
    return str(value)


def expand_experiments(
    spec: AblationSpec,
) -> tuple[list[Experiment], dict[str, list[tuple[Any, str]]], list[dict[str, Any]]]:
    base = PipelineConfig.model_validate(spec.baseline)
    experiments = [Experiment("baseline", {}, base)]
    by_fingerprint = {base.config_id: "baseline"}
    factor_levels: dict[str, list[tuple[Any, str]]] = {}

    def add(name: str, overrides: dict[str, Any], factor: str | None = None, level: Any = None) -> str:
        config = base.with_overrides(overrides)
        existing = by_fingerprint.get(config.config_id)
        if existing:
            return existing
        experiments.append(Experiment(name, overrides, config, factor, level))
        by_fingerprint[config.config_id] = name
        return name

    for factor, levels in spec.factors.items():
        factor_levels[factor] = []
        for level in levels:
            name = add(f"{factor}={_label(level)}", {factor: level}, factor, level)
            factor_levels[factor].append((level, name))

    tables: list[dict[str, Any]] = []
    for block in spec.interactions:
        level_lists = [spec.factors[f] for f in block]
        cells: dict[str, str] = {}
        for combo in itertools.product(*level_lists):
            overrides = dict(zip(block, combo))
            name = add(", ".join(f"{f}={_label(v)}" for f, v in overrides.items()), overrides)
            cells["|".join(_label(v) for v in combo)] = name
        tables.append({"factors": block, "levels": [[_label(v) for v in lv] for lv in level_lists], "cells": cells})
    return experiments, factor_levels, tables


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class AblationRunner:
    def __init__(
        self,
        spec: AblationSpec,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.spec = spec
        self.settings = settings or Settings()
        self.log = log
        cache = self.settings.data_dir / "embeddings.sqlite" if self.settings.embedding_cache else None
        key = self.settings.openai_api_key.get_secret_value() if self.settings.openai_api_key else None
        self.embedder = embedder or build_embedder(
            spec.embedder,
            model=self.settings.embedding_model,
            api_key=key,
            dimensions=self.settings.embedding_dimensions,
            cache_path=cache,
        )
        self.llm = llm if llm is not None else build_llm(self.settings)
        if spec.generator == "llm" and self.llm is None:
            raise ValueError("generator=llm requires OPENAI_API_KEY (or RAGQA_LLM_BASE_URL)")
        self.store = MemoryStore()
        self.documents: list[Document] = load_path(spec.corpus)
        items = load_dataset(spec.dataset)
        report = locate_evidence(items, {d.doc_id: d for d in self.documents})
        if not report.ok:
            raise ValueError("Invalid evaluation set:\n" + "\n".join(report.errors))
        self.items: list[EvalItem] = filter_split(items, spec.split)
        self._indexed: set[str] = set()

    def pipeline_for(self, config: PipelineConfig, seed: int | None = None, llm: LLM | None = None) -> RAGPipeline:
        collection = f"abl_{config.fingerprint('chunking')}"
        llm = llm or self.llm
        generator = build_generator(self.spec.generator, config, self.embedder, llm, seed=seed)
        pipeline = RAGPipeline(
            config, store=self.store, embedder=self.embedder, generator=generator, collection=collection, llm=llm
        )
        if collection not in self._indexed:
            report = pipeline.ingest(self.documents)
            self._indexed.add(collection)
            self.log(
                f"  indexed {collection}: {report.chunks} chunks ({config.chunking.strategy}, "
                f"size={config.chunking.size}, headers={config.chunking.contextual_headers})"
            )
        return pipeline

    def run(self) -> AblationResult:
        spec = self.spec
        experiments, factor_levels, tables = expand_experiments(spec)
        self.log(
            f"{len(experiments)} experiments x {len(self.items)} questions "
            f"(split={spec.split}, embedder={self.embedder.name}, generator={spec.generator})"
        )
        records: dict[str, list[QuestionRecord]] = {}
        summaries: dict[str, dict[str, dict[str, float]]] = {}
        judge = LLMJudge(self.llm) if spec.judge and self.llm is not None else None
        started = time.perf_counter()
        for i, exp in enumerate(experiments, start=1):
            t0 = time.perf_counter()
            pipeline = self.pipeline_for(exp.config, seed=spec.seed)
            records[exp.name] = run_items(pipeline, self.items, exp.name, spec.ks, judge=judge)
            summaries[exp.name] = aggregate(records[exp.name], n_boot=spec.n_boot, seed=spec.seed)
            headline = "  ".join(
                f"{m}={summaries[exp.name][m]['mean']:.3f}" for m in spec.primary_metrics if m in summaries[exp.name]
            )
            self.log(f"[{i}/{len(experiments)}] {exp.name:<45} {headline}  ({time.perf_counter() - t0:.1f}s)")

        comparisons = self.compare(records)
        manifest = {
            "name": spec.name,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ragqa_version": __version__,
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "dataset": str(spec.dataset),
            "dataset_sha256_12": dataset_fingerprint(spec.dataset),
            "split": spec.split,
            "n_questions": len(self.items),
            "n_documents": len(self.documents),
            "embedder": self.embedder.name,
            "generator": spec.generator
            if spec.generator == "extractive"
            else f"llm:{self.llm.model if self.llm else '?'}",
            "baseline_config_id": experiments[0].config.config_id,
            "baseline_config": experiments[0].config.model_dump(),
            "seed": spec.seed,
            "runtime_s": round(time.perf_counter() - started, 1),
        }
        result = AblationResult(
            spec=spec,
            experiments=experiments,
            records=records,
            summaries=summaries,
            comparisons=comparisons,
            factor_levels=factor_levels,
            interaction_tables=tables,
            manifest=manifest,
        )
        if spec.variability is not None:
            from ragqa.eval.variability import run_variability

            self.log("Variability study...")
            result.variability = run_variability(self, experiments[0].config, spec.variability)
        return result

    def compare(self, records: dict[str, list[QuestionRecord]]) -> dict[str, dict[str, dict[str, float]]]:
        """Paired difference vs baseline for each experiment and primary metric (Holm-adjusted)."""
        base = {r.item_id: r.metrics for r in records["baseline"]}
        out: dict[str, dict[str, dict[str, float]]] = {}
        raw_p: dict[str, dict[str, float]] = {m: {} for m in self.spec.primary_metrics}
        for name, recs in records.items():
            if name == "baseline":
                continue
            out[name] = {}
            for metric in self.spec.primary_metrics:
                pairs = [
                    (r.metrics[metric], base[r.item_id][metric])
                    for r in recs
                    if metric in r.metrics and metric in base.get(r.item_id, {})
                ]
                if not pairs:
                    continue
                a, b = [p[0] for p in pairs], [p[1] for p in pairs]
                diff, lo, hi = paired_bootstrap_diff(a, b, n_boot=self.spec.n_boot, seed=self.spec.seed)
                p = paired_permutation_test(a, b, n_perm=self.spec.n_perm, seed=self.spec.seed)
                out[name][metric] = {"diff": diff, "lo": lo, "hi": hi, "p": p}
                raw_p[metric][name] = p
        for metric, pvals in raw_p.items():
            for name, adj in holm_bonferroni(pvals).items():
                out[name][metric]["p_holm"] = adj
        return out


def save_result(result: AblationResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "records.jsonl").open("w", encoding="utf-8") as fh:
        for recs in result.records.values():
            for rec in recs:
                fh.write(json.dumps(rec.to_dict()) + "\n")
    payload = {
        "manifest": result.manifest,
        "experiments": [
            {
                "name": e.name,
                "factor": e.factor,
                "level": e.level,
                "overrides": e.overrides,
                "config_id": e.config.config_id,
            }
            for e in result.experiments
        ],
        "summaries": result.summaries,
        "comparisons": result.comparisons,
        "factor_levels": {f: [[_label(lv), name] for lv, name in levels] for f, levels in result.factor_levels.items()},
        "interactions": result.interaction_tables,
        "variability": result.variability,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")

    metrics = result.spec.report_metrics
    lines = ["experiment,factor,level," + ",".join(f"{m},{m}_lo,{m}_hi" for m in metrics) + ",latency_p50_ms,cost_usd"]
    for exp in result.experiments:
        summary = result.summaries[exp.name]
        cells = [exp.name.replace(",", ";"), exp.factor or "", _label(exp.level) if exp.factor else ""]
        for m in metrics:
            s = summary.get(m)
            cells += [f"{s['mean']:.4f}", f"{s['lo']:.4f}", f"{s['hi']:.4f}"] if s else ["", "", ""]
        cells += [f"{summary['latency_ms']['p50']:.2f}", f"{summary['cost_usd']['total']:.6f}"]
        lines.append(",".join(cells))
    (out / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
