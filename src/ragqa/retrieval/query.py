"""LLM-based query transformations.

* ``multi_query`` - paraphrase the question N ways and retrieve for each (recall on vague or
  oddly-worded questions)
* ``hyde``        - Hypothetical Document Embeddings (Gao et al., 2022): embed a short passage
  the LLM imagines would answer the question; used for dense retrieval only, because the
  invented facts must never reach lexical matching or the answer
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ragqa.generation.llm import LLM
from ragqa.types import Usage


@dataclass(frozen=True)
class QueryVariant:
    text: str
    kind: str = "original"
    dense: bool = True
    lexical: bool = True


class QueryExpander(ABC):
    @abstractmethod
    def expand(self, query: str) -> tuple[list[QueryVariant], Usage]: ...


class MultiQueryExpander(QueryExpander):
    def __init__(self, llm: LLM, n: int = 3) -> None:
        self.llm = llm
        self.n = n

    def expand(self, query: str) -> tuple[list[QueryVariant], Usage]:
        response = self.llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        f"Rewrite the user's question in {self.n} different ways that could match how "
                        "the answer is phrased in company documents. One rewrite per line, no numbering."
                    ),
                },
                {"role": "user", "content": query},
            ]
        )
        rewrites: list[str] = []
        for line in response.text.splitlines():
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if line and line.lower() != query.lower() and line not in rewrites:
                rewrites.append(line)
        variants = [QueryVariant(query)] + [QueryVariant(r, "paraphrase") for r in rewrites[: self.n]]
        return variants, response.usage


class HyDEExpander(QueryExpander):
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def expand(self, query: str) -> tuple[list[QueryVariant], Usage]:
        response = self.llm.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Write a short, factual-sounding paragraph (3 sentences) from an internal "
                        "company document that would answer the question. Do not mention the question."
                    ),
                },
                {"role": "user", "content": query},
            ]
        )
        variants = [QueryVariant(query)]
        if response.text.strip():
            variants.append(QueryVariant(response.text.strip(), "hyde", dense=True, lexical=False))
        return variants, response.usage
