"""LLM-as-judge for answer correctness and faithfulness (optional; needs an LLM).

String metrics undercount correct paraphrases ("€24.90 per month" vs "24.90 EUR monthly") and
cannot judge synthesis. A judge model scores them with a fixed rubric at temperature 0; its
verdicts should be spot-checked against the string metrics before they are trusted.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ragqa.eval.dataset import EvalItem
from ragqa.generation.llm import LLM
from ragqa.types import ScoredChunk

_CORRECTNESS = (
    "You grade answers to questions about company documents. Compare the candidate answer with "
    "the reference answers. The candidate is correct if it states the same fact (paraphrases, "
    "units and formatting differences are fine) and does not add contradicting claims. "
    'Reply with JSON only: {"correct": 0 or 1, "reason": "<one sentence>"}'
)
_FAITHFULNESS = (
    "You check whether an answer is supported by the given sources. Split the answer into atomic "
    "claims and count how many are directly supported by the sources. "
    'Reply with JSON only: {"supported": <int>, "total": <int>}'
)


def _parse_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class LLMJudge:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def correctness(self, item: EvalItem, answer: str) -> float | None:
        if not item.answerable:
            return None
        response = self.llm.complete(
            [
                {"role": "system", "content": _CORRECTNESS},
                {
                    "role": "user",
                    "content": (
                        f"Question: {item.question}\nReference answers: {json.dumps(item.answers)}\n"
                        f"Candidate answer: {answer}"
                    ),
                },
            ],
            temperature=0,
        )
        verdict = _parse_json(response.text)
        if verdict is None or "correct" not in verdict:
            return None
        return float(bool(int(verdict["correct"])))

    def faithfulness(self, answer: str, sources: list[ScoredChunk]) -> float | None:
        context = "\n\n".join(f"[S{i}] {s.chunk.text}" for i, s in enumerate(sources, start=1))
        response = self.llm.complete(
            [
                {"role": "system", "content": _FAITHFULNESS},
                {"role": "user", "content": f"Sources:\n{context}\n\nAnswer: {answer}"},
            ],
            temperature=0,
        )
        verdict = _parse_json(response.text)
        try:
            total = int(verdict["total"]) if verdict else 0
            supported = int(verdict["supported"]) if verdict else 0
        except (KeyError, TypeError, ValueError):
            return None
        return min(1.0, supported / total) if total > 0 else None
