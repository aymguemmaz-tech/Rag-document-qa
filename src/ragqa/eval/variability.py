"""Output-variability study: ask every question N times per temperature and measure stability.

LLM answers are stochastic; a single evaluation run hides how often the *same* question flips
between right and wrong. Reported per temperature:

* accuracy mean ± standard deviation across runs
* flip rate - share of questions whose correctness is not constant across runs
* answer agreement - mean pairwise exact agreement of normalised answers
* pairwise F1 - mean token F1 between answers to the same question
* citation Jaccard - mean pairwise overlap of the cited source sets
"""

from __future__ import annotations

import itertools
import statistics
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from ragqa.config import PipelineConfig
from ragqa.eval.metrics import token_f1
from ragqa.generation.citations import strip_citations
from ragqa.generation.llm import LLM, LLMResponse, Message
from ragqa.text import normalize_answer

if TYPE_CHECKING:
    from ragqa.eval.ablation import AblationRunner, VariabilitySpec


class TemperatureLLM(LLM):
    """Wraps an LLM and forces a sampling temperature on every call."""

    def __init__(self, inner: LLM, temperature: float) -> None:
        self.inner = inner
        self.model = inner.model
        self.temperature = temperature

    def complete(self, messages: list[Message], **overrides: Any) -> LLMResponse:
        return self.inner.complete(messages, **{"temperature": self.temperature, **overrides})

    def stream(self, messages: list[Message], **overrides: Any) -> Iterator[str | LLMResponse]:
        return self.inner.stream(messages, **{"temperature": self.temperature, **overrides})


def _pairwise_mean(values: list[Any], fn: Any) -> float:
    pairs = list(itertools.combinations(values, 2))
    return sum(fn(a, b) for a, b in pairs) / len(pairs) if pairs else 1.0


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def run_variability(runner: AblationRunner, config: PipelineConfig, spec: VariabilitySpec) -> dict[str, Any]:
    from ragqa.eval.runner import run_items

    items = runner.items[: spec.limit] if spec.limit else runner.items
    rows: list[dict[str, Any]] = []
    for temperature in spec.temperatures:
        per_run_accuracy: list[float] = []
        answers: dict[str, list[tuple[str, float, frozenset[str]]]] = {i.id: [] for i in items}
        for run in range(spec.runs):
            if runner.spec.generator == "extractive":
                cfg = config.with_overrides({"generation.temperature": temperature})
                pipeline = runner.pipeline_for(cfg, seed=runner.spec.seed * 1000 + run)
            else:
                assert runner.llm is not None
                pipeline = runner.pipeline_for(config, llm=TemperatureLLM(runner.llm, temperature))
            records = run_items(pipeline, items, f"T={temperature}/run{run}", runner.spec.ks)
            per_run_accuracy.append(sum(r.metrics["overall_correct"] for r in records) / len(records))
            for r in records:
                answers[r.item_id].append(
                    (normalize_answer(strip_citations(r.answer)), r.metrics["overall_correct"], frozenset(r.cited))
                )
        flips = sum(1 for runs in answers.values() if len({c for _, c, _ in runs}) > 1)
        rows.append(
            {
                "temperature": temperature,
                "accuracy_mean": statistics.fmean(per_run_accuracy),
                "accuracy_sd": statistics.pstdev(per_run_accuracy) if len(per_run_accuracy) > 1 else 0.0,
                "flip_rate": flips / len(items),
                "answer_agreement": statistics.fmean(
                    _pairwise_mean([a for a, _, _ in runs], lambda x, y: float(x == y)) for runs in answers.values()
                ),
                "pairwise_f1": statistics.fmean(
                    _pairwise_mean([a for a, _, _ in runs], token_f1) for runs in answers.values()
                ),
                "citation_jaccard": statistics.fmean(
                    _pairwise_mean([c for _, _, c in runs], _jaccard) for runs in answers.values()
                ),
            }
        )
        runner.log(
            f"  T={temperature}: accuracy {rows[-1]['accuracy_mean']:.3f}±{rows[-1]['accuracy_sd']:.3f}, "
            f"flip rate {rows[-1]['flip_rate']:.3f}"
        )
    return {"runs": spec.runs, "n_questions": len(items), "generator": runner.spec.generator, "rows": rows}
