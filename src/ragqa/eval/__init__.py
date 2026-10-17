"""Evaluation layer: gold set, metrics, statistics, ablations, variability and reports."""

from ragqa.eval.dataset import EvalItem, Evidence, load_dataset, locate_evidence
from ragqa.eval.metrics import answer_metrics, citation_metrics, retrieval_metrics
from ragqa.eval.stats import bootstrap_ci, holm_bonferroni, paired_permutation_test

__all__ = [
    "EvalItem",
    "Evidence",
    "answer_metrics",
    "bootstrap_ci",
    "citation_metrics",
    "holm_bonferroni",
    "load_dataset",
    "locate_evidence",
    "paired_permutation_test",
    "retrieval_metrics",
]
