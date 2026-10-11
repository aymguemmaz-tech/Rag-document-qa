"""Rank fusion: Reciprocal Rank Fusion (Cormack et al., 2009).

RRF scores a candidate by ``sum_i w_i / (k + rank_i)``. It needs no score calibration between
retrievers whose raw scores live on different scales (cosine vs BM25), which is why it is the
default way to combine dense and lexical results.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragqa.types import ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]],
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> list[ScoredChunk]:
    weights = list(weights) if weights is not None else [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights and ranked_lists must have the same length")
    scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    first_seen: dict[str, int] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, item in enumerate(ranked, start=1):
            cid = item.chunk_id
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            first_seen.setdefault(cid, len(first_seen))
            if cid not in best or (best[cid].embedding is None and item.embedding is not None):
                best[cid] = item
    order = sorted(scores, key=lambda cid: (-scores[cid], first_seen[cid]))
    return [
        ScoredChunk(
            chunk=best[cid].chunk,
            score=scores[cid],
            rank=i,
            source="rrf",
            embedding=best[cid].embedding,
            flags=list(best[cid].flags),
        )
        for i, cid in enumerate(order, start=1)
    ]
