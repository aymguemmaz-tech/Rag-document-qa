"""Prompt templates and context assembly (token budget, ordering, source labels)."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

from ragqa.text import count_tokens
from ragqa.types import ScoredChunk


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: int
    description: str
    system: str
    user: str

    def render(self, question: str, context: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system.strip()},
            {"role": "user", "content": self.user.format(question=question.strip(), context=context).strip()},
        ]


def load_templates(path: str | Path | None = None) -> dict[str, PromptTemplate]:
    if path is not None:
        raw = Path(path).read_text(encoding="utf-8")
    else:
        raw = resources.files("ragqa.prompts").joinpath("templates.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    version = int(data.get("version", 1))
    return {
        name: PromptTemplate(
            name=name,
            version=version,
            description=spec.get("description", ""),
            system=spec["system"],
            user=spec["user"],
        )
        for name, spec in data["templates"].items()
    }


def sandwich(items: list[ScoredChunk]) -> list[ScoredChunk]:
    """Place the strongest sources at both ends ("lost in the middle", Liu et al., 2023)."""
    front, back = items[0::2], items[1::2]
    return front + back[::-1]


def build_context(
    sources: list[ScoredChunk], max_tokens: int, order: str = "relevance"
) -> tuple[str, list[ScoredChunk]]:
    """Select sources within the token budget and render them with ``[S#]`` labels.

    Returns the context string and the sources in label order (``S1`` is index 0).
    """
    used: list[ScoredChunk] = []
    seen_text: set[str] = set()
    budget = max_tokens
    for item in sources:
        key = " ".join(item.chunk.text.split())
        if key in seen_text:
            continue
        cost = count_tokens(item.chunk.text) + 12
        if used and cost > budget:
            continue
        used.append(item)
        seen_text.add(key)
        budget -= cost
    if order == "sandwich":
        used = sandwich(used)
    blocks = [f"[S{i}] {s.chunk.location}\n{s.chunk.text.strip()}" for i, s in enumerate(used, start=1)]
    return "\n\n".join(blocks), used
