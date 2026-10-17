"""Statistics for small evaluation sets: bootstrap intervals and paired significance tests.

With ~50 questions, a 4-point difference can easily be noise. Every reported mean gets a 95%
percentile-bootstrap interval, and every comparison against the baseline gets a paired
sign-flip permutation test (the same questions are answered by both configurations) with a
Holm-Bonferroni correction across all comparisons in the ablation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1 or np.all(arr == arr[0]):
        return float(arr.mean()), float(arr.mean()), float(arr.mean())
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(n_boot, arr.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(lo), float(hi)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) with a percentile-bootstrap interval over paired questions."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(diff.tolist(), n_boot=n_boot, alpha=alpha, seed=seed)


def paired_permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test on paired differences. Returns a p-value."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size == 0 or np.all(diff == 0):
        return 1.0
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, diff.size))
    permuted = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(permuted >= observed - 1e-12) + 1) / (n_perm + 1))


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else float("nan")
