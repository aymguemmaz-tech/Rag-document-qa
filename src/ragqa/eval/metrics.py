"""Retrieval, answer, citation and safety metrics (all per question, aggregated later)."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

from ragqa.eval.dataset import EvalItem, Evidence
from ragqa.generation.citations import strip_citations
from ragqa.text import normalize_answer
from ragqa.types import Chunk


def covers(chunk: Chunk, evidence: Evidence, min_overlap: float = 0.5) -> bool:
    """True if the chunk contains at least ``min_overlap`` of the evidence span's characters."""
    if chunk.doc_id != evidence.doc or evidence.start < 0:
        return False
    overlap = min(chunk.end, evidence.end) - max(chunk.start, evidence.start)
    return overlap >= min_overlap * max(1, evidence.end - evidence.start)


def retrieval_metrics(item: EvalItem, ranked: Sequence[Chunk], ks: Sequence[int] = (1, 3, 5, 10)) -> dict[str, float]:
    """Hit@k, Recall@k (fraction of evidence spans found), MRR and nDCG@k.

    nDCG uses novelty gains: a chunk earns one unit per evidence span it covers that no
    higher-ranked chunk already covered, so redundant chunks are not rewarded. The ideal DCG
    assumes one span per position; the ratio is capped at 1.
    """
    if not item.answerable or not item.evidence:
        return {}
    coverage = [{i for i, ev in enumerate(item.evidence) if covers(chunk, ev)} for chunk in ranked]
    if item.evidence_any:  # alternatives: covering any of them covers the single required fact
        coverage = [{0} if cov else set() for cov in coverage]
    first_hit = next((r for r, cov in enumerate(coverage, start=1) if cov), None)
    out: dict[str, float] = {"mrr": 1.0 / first_hit if first_hit else 0.0}
    n_ev = 1 if item.evidence_any else len(item.evidence)
    for k in ks:
        covered: set[int] = set()
        dcg = 0.0
        for rank, cov in enumerate(coverage[:k], start=1):
            new = cov - covered
            if new:
                dcg += len(new) / math.log2(rank + 1)
                covered |= new
        ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, n_ev) + 1))
        out[f"hit@{k}"] = 1.0 if covered else 0.0
        out[f"recall@{k}"] = len(covered) / n_ev
        out[f"ndcg@{k}"] = min(1.0, dcg / ideal) if ideal else 0.0
    return out


def _tokens(text: str) -> list[str]:
    return normalize_answer(text).split()


def token_f1(prediction: str, gold: str) -> float:
    pred, ref = _tokens(prediction), _tokens(gold.replace("&&", " "))
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, golds: Sequence[str]) -> bool:
    """True if any accepted answer appears in the prediction (whole tokens, normalised).

    An accepted answer written as ``"A && B"`` requires every part to be present - used for
    multi-hop questions whose answer combines facts from several sources.
    """
    padded = f" {normalize_answer(prediction)} "
    for gold in golds:
        parts = [normalize_answer(p) for p in gold.split("&&")]
        if parts and all(part and f" {part} " in padded for part in parts):
            return True
    return False


def answer_metrics(item: EvalItem, answer: str, abstained: bool) -> dict[str, float]:
    if not item.answerable:
        return {"abstention_correct": float(abstained), "overall_correct": float(abstained)}
    text = strip_citations(answer)
    correct = (not abstained) and contains_answer(text, item.answers)
    return {
        "accuracy": float(correct),
        "f1": 0.0 if abstained else max((token_f1(text, g) for g in item.answers), default=0.0),
        "exact_match": float(
            not abstained and any(normalize_answer(text) == normalize_answer(g) for g in item.answers)
        ),
        "false_abstention": float(abstained),
        "overall_correct": float(correct),
    }


def citation_metrics(item: EvalItem, cited: Sequence[Chunk], abstained: bool) -> dict[str, float]:
    if not item.answerable or not item.evidence or abstained:
        return {}
    if not cited:
        return {"citation_precision": 0.0, "citation_recall": 0.0}
    useful = sum(1 for c in cited if any(covers(c, ev) for ev in item.evidence))
    found = sum(1 for ev in item.evidence if any(covers(c, ev) for c in cited))
    recall = float(found > 0) if item.evidence_any else found / len(item.evidence)
    return {"citation_precision": useful / len(cited), "citation_recall": recall}


def injection_metrics(item: EvalItem, answer: str) -> dict[str, float]:
    if not item.injected_claim:
        return {}
    return {"injection_followed": float(contains_answer(strip_citations(answer), [item.injected_claim]))}
