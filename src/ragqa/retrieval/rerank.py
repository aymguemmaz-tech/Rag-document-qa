"""Second-stage rerankers applied to the fused candidate list.

* ``mmr``           - Maximal Marginal Relevance: trades relevance for diversity (fewer
                      near-duplicate chunks wasting context budget)
* ``maxsim``        - late interaction (ColBERT-style MaxSim) over static word vectors: each query
                      term is matched to its most similar chunk term; fused with the first-stage
                      rank by RRF. Captures soft term matches (price ~ cost ~ fee) cheaply
* ``llm``           - listwise reranking by a chat model (most accurate, adds latency and cost)
* ``cross_encoder`` - a sentence-transformers cross-encoder (optional dependency)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ragqa.embeddings import Embedder
from ragqa.generation.llm import LLM
from ragqa.retrieval.fusion import reciprocal_rank_fusion
from ragqa.text import content_words
from ragqa.types import ScoredChunk, Usage


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> tuple[list[ScoredChunk], Usage]:
        """Return candidates in the new order (at least ``k`` when available) and usage."""


def _reranked(items: list[ScoredChunk], scores: list[float], source: str) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=c.chunk, score=float(s), rank=i, source=source, embedding=c.embedding, flags=list(c.flags))
        for i, (c, s) in enumerate(zip(items, scores), start=1)
    ]


class MMRReranker(Reranker):
    name = "mmr"

    def __init__(self, embedder: Embedder, lambda_: float = 0.7) -> None:
        self.embedder = embedder
        self.lambda_ = lambda_

    def _vectors(self, candidates: list[ScoredChunk]) -> np.ndarray:
        dim = self.embedder.dim
        missing = [i for i, c in enumerate(candidates) if c.embedding is None or len(c.embedding) != dim]
        vectors = [c.embedding for c in candidates]
        if missing:
            fresh = self.embedder.embed([candidates[i].chunk.embed_text for i in missing])
            for i, vec in zip(missing, fresh):
                vectors[i] = vec
        return np.stack([np.asarray(v, dtype=np.float32) for v in vectors])

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> tuple[list[ScoredChunk], Usage]:
        if len(candidates) <= 1:
            return candidates, Usage()
        vectors = self._vectors(candidates)
        relevance = vectors @ self.embedder.embed_query(query)
        selected: list[int] = []
        mmr_scores: list[float] = []
        remaining = list(range(len(candidates)))
        while remaining and len(selected) < k:
            if selected:
                redundancy = np.max(vectors[remaining] @ vectors[selected].T, axis=1)
            else:
                redundancy = np.zeros(len(remaining))
            scores = self.lambda_ * relevance[remaining] - (1 - self.lambda_) * redundancy
            best = int(np.argmax(scores))
            selected.append(remaining.pop(best))
            mmr_scores.append(float(scores[best]))
        rest = sorted(remaining, key=lambda i: -relevance[i])
        order = selected + rest
        scores_out = mmr_scores + [float(relevance[i]) for i in rest]
        return _reranked([candidates[i] for i in order], scores_out, "mmr"), Usage()


class MaxSimReranker(Reranker):
    name = "maxsim"

    def __init__(self, embedder: Embedder, fusion_k: int = 60) -> None:
        if not hasattr(embedder, "word_vectors"):
            raise ValueError("MaxSim reranking needs an embedder with word vectors (wordllama)")
        self.embedder = embedder
        self.fusion_k = fusion_k

    def score(self, query: str, text: str) -> float:
        q_terms = list(dict.fromkeys(content_words(query)))
        d_terms = list(dict.fromkeys(content_words(text)))
        if not q_terms or not d_terms:
            return 0.0
        word_vectors: Any = self.embedder.word_vectors
        q_vecs, d_vecs = word_vectors(q_terms), word_vectors(d_terms)
        return float(np.mean(np.max(q_vecs @ d_vecs.T, axis=1)))

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> tuple[list[ScoredChunk], Usage]:
        if len(candidates) <= 1:
            return candidates, Usage()
        scores = [self.score(query, c.chunk.embed_text) for c in candidates]
        by_maxsim = [candidates[i] for i in np.argsort(-np.asarray(scores), kind="stable")]
        fused = reciprocal_rank_fusion([candidates, by_maxsim], k=self.fusion_k)
        for item in fused:
            item.source = "maxsim"
        return fused, Usage()


class LLMReranker(Reranker):
    name = "llm"
    SYSTEM = (
        "You rank passages by how useful they are for answering a question. "
        "Reply with passage numbers only, most useful first, separated by commas."
    )

    def __init__(self, llm: LLM, max_passage_chars: int = 700) -> None:
        self.llm = llm
        self.max_passage_chars = max_passage_chars

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> tuple[list[ScoredChunk], Usage]:
        if len(candidates) <= 1:
            return candidates, Usage()
        passages = "\n\n".join(
            f"[{i}] {c.chunk.location}\n{c.chunk.text[: self.max_passage_chars]}"
            for i, c in enumerate(candidates, start=1)
        )
        response = self.llm.complete(
            [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": f"Question: {query}\n\nPassages:\n{passages}"},
            ]
        )
        order: list[int] = []
        for token in re.findall(r"\d+", response.text):
            idx = int(token) - 1
            if 0 <= idx < len(candidates) and idx not in order:
                order.append(idx)
        order += [i for i in range(len(candidates)) if i not in order]
        n = len(order)
        return (
            _reranked([candidates[i] for i in order], [float(n - r) for r in range(n)], "llm"),
            response.usage,
        )


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", model: Any = None) -> None:
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("pip install 'ragqa[rerank]' to use the cross-encoder") from exc
            model = CrossEncoder(model_name)
        self.model = model

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> tuple[list[ScoredChunk], Usage]:
        if len(candidates) <= 1:
            return candidates, Usage()
        scores = np.asarray(self.model.predict([(query, c.chunk.embed_text) for c in candidates]))
        order = list(np.argsort(-scores, kind="stable"))
        return _reranked([candidates[i] for i in order], [float(scores[i]) for i in order], "cross_encoder"), Usage()
