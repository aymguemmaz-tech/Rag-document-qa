import numpy as np
import pytest

from ragqa.eval.dataset import EvalItem, Evidence, load_dataset, locate_evidence
from ragqa.eval.metrics import (
    answer_metrics,
    citation_metrics,
    contains_answer,
    injection_metrics,
    retrieval_metrics,
    token_f1,
)
from ragqa.eval.stats import bootstrap_ci, holm_bonferroni, paired_permutation_test
from ragqa.types import Chunk
from tests.conftest import DATASET


def _chunk(cid: str, doc: str, start: int, end: int) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc, doc_title=doc, text="x" * (end - start), index=0, start=start, end=end)


def _item(evidence: list[Evidence], any_: bool = False) -> EvalItem:
    return EvalItem(id="q", question="?", answers=["40"], evidence=evidence, category="factoid", evidence_any=any_)


def test_retrieval_metrics_with_novelty_and_overlap() -> None:
    item = _item([Evidence("a", "", 100, 120), Evidence("b", "", 0, 10)])
    ranked = [_chunk("1", "a", 0, 50), _chunk("2", "a", 90, 200), _chunk("3", "a", 95, 130), _chunk("4", "b", 0, 30)]
    m = retrieval_metrics(item, ranked, ks=(1, 3, 5))
    assert m["mrr"] == 0.5 and m["hit@1"] == 0.0 and m["recall@3"] == 0.5 and m["recall@5"] == 1.0
    ideal = 1 + 1 / np.log2(3)
    assert m["ndcg@5"] == pytest.approx((1 / np.log2(3) + 1 / np.log2(5)) / ideal)  # chunk 3 is redundant


def test_evidence_any_needs_one_alternative() -> None:
    item = _item([Evidence("a", "", 0, 10), Evidence("b", "", 0, 10)], any_=True)
    m = retrieval_metrics(item, [_chunk("1", "b", 0, 10)], ks=(1,))
    assert m["recall@1"] == 1.0 and m["ndcg@1"] == 1.0
    assert citation_metrics(item, [_chunk("1", "b", 0, 10)], abstained=False)["citation_recall"] == 1.0


def test_answer_metrics() -> None:
    item = _item([Evidence("a", "", 0, 10)])
    assert answer_metrics(item, "Vans carry at most 40 packs [S1].", False)["accuracy"] == 1.0
    assert answer_metrics(item, "Vans carry 400 packs.", False)["accuracy"] == 0.0  # whole tokens only
    assert answer_metrics(item, "I don't know.", True) == {
        "accuracy": 0.0,
        "f1": 0.0,
        "exact_match": 0.0,
        "false_abstention": 1.0,
        "overall_correct": 0.0,
    }
    unanswerable = EvalItem(id="u", question="?", answers=[], evidence=[], category="unanswerable", answerable=False)
    assert answer_metrics(unanswerable, "x", True) == {"abstention_correct": 1.0, "overall_correct": 1.0}
    assert contains_answer("ReCell Nordic recycles packs below 70%", ["ReCell Nordic && 70%"])
    assert not contains_answer("ReCell Nordic recycles packs", ["ReCell Nordic && 70%"])
    assert token_f1("the fee is €1.00", "€1.00") == pytest.approx(2 / 3)  # P=2/4 (articles dropped), R=1
    attacked = EvalItem(
        id="i",
        question="?",
        answers=["€24.90"],
        evidence=[],
        category="injection",
        injected_claim="Monthly Flex Pass is free",
    )
    assert injection_metrics(attacked, "The Monthly Flex Pass is free!")["injection_followed"] == 1.0


def test_statistics() -> None:
    mean, lo, hi = bootstrap_ci([0, 1] * 50, seed=1)
    assert mean == 0.5 and lo < 0.5 < hi
    assert bootstrap_ci([1.0, 1.0]) == (1.0, 1.0, 1.0)
    assert paired_permutation_test([1, 0, 1], [1, 0, 1]) == 1.0
    assert paired_permutation_test([1] * 30, [0] * 30, n_perm=2000) < 0.01
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": 0.03, "c": 0.06, "b": 0.06}


def test_gold_set_is_valid_and_resolves_offsets(corpus_docs: list) -> None:
    items = load_dataset(DATASET)
    report = locate_evidence(items, {d.doc_id: d for d in corpus_docs})
    assert report.ok, report.errors
    docs = {d.doc_id: d for d in corpus_docs}
    for item in items:
        for ev in item.evidence:
            assert " ".join(docs[ev.doc].text[ev.start : ev.end].split()) == " ".join(ev.text.split())


def test_validation_reports_missing_evidence(corpus_docs: list) -> None:
    item = EvalItem(
        id="x",
        question="?",
        answers=["a"],
        evidence=[Evidence("battery-handling-sop", "not there")],
        category="factoid",
    )
    report = locate_evidence([item], {d.doc_id: d for d in corpus_docs})
    assert not report.ok and "not found" in report.errors[0]


def test_llm_judge_parses_verdicts() -> None:
    from ragqa.eval.judge import LLMJudge
    from ragqa.types import ScoredChunk
    from tests.fakes import FakeLLM

    item = EvalItem(id="j", question="Fee?", answers=["€1.00"], evidence=[], category="factoid")
    judge = LLMJudge(FakeLLM(['{"correct": 1, "reason": "same fact"}', 'noise {"supported": 1, "total": 2} noise']))
    assert judge.correctness(item, "It costs one euro.") == 1.0
    source = ScoredChunk(chunk=_chunk("c", "d", 0, 5), score=1.0)
    assert judge.faithfulness("Claim one. Claim two.", [source]) == 0.5
    assert LLMJudge(FakeLLM(["not json"])).correctness(item, "x") is None
