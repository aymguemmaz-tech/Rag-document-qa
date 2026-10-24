"""Run one pipeline configuration over the gold set and aggregate the per-question metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ragqa.eval.dataset import EvalItem
from ragqa.eval.judge import LLMJudge
from ragqa.eval.metrics import answer_metrics, citation_metrics, injection_metrics, retrieval_metrics
from ragqa.eval.stats import bootstrap_ci, percentile
from ragqa.observability import StageTimer
from ragqa.pipeline import RAGPipeline

DEFAULT_KS = (1, 3, 5, 10)


@dataclass
class QuestionRecord:
    experiment: str
    item_id: str
    category: str
    answerable: bool
    metrics: dict[str, float]
    answer: str
    abstained: bool
    retrieved: list[str]
    cited: list[str]
    grounded: float | None
    flags: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_item(
    pipeline: RAGPipeline,
    item: EvalItem,
    experiment: str = "default",
    ks: Sequence[int] = DEFAULT_KS,
    judge: LLMJudge | None = None,
) -> QuestionRecord:
    timer = StageTimer()
    with timer.stage("retrieve"):
        retrieval = pipeline.retrieve(item.question)
    answer = pipeline.answer_from_retrieval(item.question, retrieval, timer)
    ranked = [s.chunk for s in retrieval.candidates]
    by_id = {s.chunk.chunk_id: s.chunk for s in answer.sources}
    cited = [by_id[c.chunk_id] for c in answer.citations if c.chunk_id in by_id]

    metrics: dict[str, float] = {}
    metrics.update(retrieval_metrics(item, ranked, ks))
    metrics.update(answer_metrics(item, answer.text, answer.abstained))
    metrics.update(citation_metrics(item, cited, answer.abstained))
    metrics.update(injection_metrics(item, answer.text))
    if answer.grounded_score is not None:
        metrics["grounded"] = answer.grounded_score
    if judge is not None and not answer.abstained:
        verdict = judge.correctness(item, answer.text)
        if verdict is not None:
            metrics["judge_correct"] = verdict
        faithful = judge.faithfulness(answer.text, answer.sources)
        if faithful is not None:
            metrics["judge_faithfulness"] = faithful
    metrics["latency_ms"] = answer.timings_ms.get("total", timer.total())
    metrics["cost_usd"] = answer.usage.cost_usd
    return QuestionRecord(
        experiment=experiment,
        item_id=item.id,
        category=item.category,
        answerable=item.answerable,
        metrics=metrics,
        answer=answer.text,
        abstained=answer.abstained,
        retrieved=[s.chunk.chunk_id for s in retrieval.chunks],
        cited=[c.chunk_id for c in answer.citations],
        grounded=answer.grounded_score,
        flags=answer.flags,
        timings_ms=answer.timings_ms,
        usage=answer.usage.to_dict(),
    )


def run_items(
    pipeline: RAGPipeline,
    items: Sequence[EvalItem],
    experiment: str = "default",
    ks: Sequence[int] = DEFAULT_KS,
    progress: Callable[[int, int], None] | None = None,
    judge: LLMJudge | None = None,
) -> list[QuestionRecord]:
    records = []
    for i, item in enumerate(items, start=1):
        records.append(evaluate_item(pipeline, item, experiment, ks, judge))
        if progress:
            progress(i, len(items))
    return records


def aggregate(records: Sequence[QuestionRecord], n_boot: int = 2000, seed: int = 0) -> dict[str, dict[str, float]]:
    """Mean and 95% bootstrap CI for every metric, plus latency percentiles and total cost."""
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for name, value in record.metrics.items():
            values[name].append(float(value))
    summary: dict[str, dict[str, float]] = {}
    for name, vals in sorted(values.items()):
        if name in ("latency_ms", "cost_usd"):
            continue
        mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        summary[name] = {"mean": mean, "lo": lo, "hi": hi, "n": float(len(vals))}
    latencies = values.get("latency_ms", [])
    summary["latency_ms"] = {
        "mean": sum(latencies) / len(latencies) if latencies else float("nan"),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "n": float(len(latencies)),
    }
    costs = values.get("cost_usd", [])
    summary["cost_usd"] = {"total": float(sum(costs)), "per_question": float(sum(costs) / len(costs)) if costs else 0.0}
    return summary


def by_category(records: Sequence[QuestionRecord], metric: str) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if metric in record.metrics:
            groups[record.category].append(record.metrics[metric])
    return {cat: sum(v) / len(v) for cat, v in sorted(groups.items())}
