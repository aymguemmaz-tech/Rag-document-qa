"""Retrieval orchestration: query expansion -> dense/lexical search -> RRF fusion -> guard -> rerank."""

from __future__ import annotations

from dataclasses import dataclass, field

from ragqa.config import RetrievalConfig
from ragqa.embeddings import Embedder
from ragqa.generation.grounding import InjectionGuard
from ragqa.observability import StageTimer
from ragqa.pricing import cost_usd
from ragqa.retrieval.fusion import reciprocal_rank_fusion
from ragqa.retrieval.query import QueryExpander, QueryVariant
from ragqa.retrieval.rerank import Reranker
from ragqa.store.base import VectorStore
from ragqa.text import count_tokens
from ragqa.types import ScoredChunk, Usage


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    candidates: list[ScoredChunk]
    flagged: list[ScoredChunk] = field(default_factory=list)
    variants: list[QueryVariant] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    timings_ms: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        collection: str,
        cfg: RetrievalConfig,
        *,
        reranker: Reranker | None = None,
        expander: QueryExpander | None = None,
        guard: InjectionGuard | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.collection = collection
        self.cfg = cfg
        self.reranker = reranker
        self.expander = expander
        self.guard = guard

    def _embedding_usage(self, texts: list[str]) -> Usage:
        tokens = sum(count_tokens(t) for t in texts)
        model = self.embedder.name.split(":", 1)[1] if self.embedder.name.startswith("openai:") else ""
        return Usage(embedding_tokens=tokens, cost_usd=cost_usd(model, tokens) if model else 0.0)

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        k = k or self.cfg.k
        timer = StageTimer()
        usage = Usage()

        variants = [QueryVariant(query)]
        if self.expander is not None:
            with timer.stage("expand"):
                variants, expand_usage = self.expander.expand(query)
            usage += expand_usage

        lists: list[list[ScoredChunk]] = []
        weights: list[float] = []
        fetch_k = max(self.cfg.fetch_k, k)
        with timer.stage("search"):
            if self.cfg.mode in ("dense", "hybrid"):
                dense_texts = [v.text for v in variants if v.dense]
                usage += self._embedding_usage(dense_texts)
                for vector in self.embedder.embed(dense_texts):
                    lists.append(self.store.dense_search(self.collection, vector, fetch_k))
                    weights.append(self.cfg.dense_weight)
            if self.cfg.mode in ("lexical", "hybrid"):
                for variant in variants:
                    if variant.lexical:
                        lists.append(self.store.lexical_search(self.collection, variant.text, fetch_k))
                        weights.append(self.cfg.lexical_weight)

        if len(lists) == 1:
            fused = lists[0]
        else:
            with timer.stage("fuse"):
                fused = reciprocal_rank_fusion(lists, weights, k=self.cfg.rrf_k)

        if self.cfg.exclude_superseded:  # document lifecycle: never answer from retired versions
            fused = [c for c in fused if c.chunk.metadata.get("status") != "superseded"]

        flagged: list[ScoredChunk] = []
        if self.guard is not None:
            fused, flagged = self.guard.apply(fused)

        candidates = fused
        if self.reranker is not None and fused:
            with timer.stage("rerank"):
                head, rerank_usage = self.reranker.rerank(query, fused[: self.cfg.rerank_candidates], k)
            usage += rerank_usage
            candidates = head + fused[self.cfg.rerank_candidates :]

        for rank, item in enumerate(candidates, start=1):
            item.rank = rank
        return RetrievalResult(
            chunks=candidates[:k],
            candidates=candidates,
            flagged=flagged,
            variants=variants,
            usage=usage,
            timings_ms=timer.timings,
        )
