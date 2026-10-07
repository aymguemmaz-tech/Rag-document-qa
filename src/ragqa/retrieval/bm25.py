"""Okapi BM25 over analyzed terms (used by the in-memory store and as a reference implementation)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from ragqa.text import analyze


class BM25Index:
    def __init__(self, documents: Sequence[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for doc_idx, text in enumerate(documents):
            terms = analyze(text)
            self.lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                self.postings[term].append((doc_idx, tf))
        self.n = len(self.lengths)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        if not self.n:
            return scores
        for term in dict.fromkeys(analyze(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_idx, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_idx] / (self.avgdl or 1.0))
                scores[doc_idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
