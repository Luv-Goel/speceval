"""Delta computation and statistical comparison between evaluation runs."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as _sp_stats

from speceval.exceptions import CompareError
from speceval.store.base import ResultStore


@dataclass
class ComparisonResult:
    """Result of comparing two evaluation runs."""

    run_a: str
    """Identifier of the first run (baseline)."""

    run_b: str
    """Identifier of the second run (comparison)."""

    metric_deltas: dict[str, float]
    """Per-metric difference ``b - a``."""

    significance: dict[str, float]
    """Per-metric p-value from bootstrap significance test."""

    effect_sizes: dict[str, float]
    """Per-metric Cohen's *d* effect size."""

    details: dict[str, Any] = field(default_factory=dict)
    """Additional diagnostic information."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_runs(
    run_a_id: str,
    run_b_id: str,
    store: ResultStore,
    n_resamples: int = 1000,
) -> ComparisonResult:
    """Compare two runs and return a ``ComparisonResult``.

    Args:
        run_a_id: Baseline run identifier.
        run_b_id: Comparison run identifier.
        store: Initialised result store.
        n_resamples: Number of bootstrap resamples for significance testing.

    Returns:
        A populated ``ComparisonResult``.

    Raises:
        CompareError: If runs cannot be loaded or have no matching metrics.
    """
    results_a = store.get_results(run_a_id)
    results_b = store.get_results(run_b_id)

    if not results_a:
        raise CompareError(f"Run {run_a_id!r} has no results")
    if not results_b:
        raise CompareError(f"Run {run_b_id!r} has no results")

    # Aggregate per-metric scores per item
    metrics_a = _aggregate_metrics(results_a)
    metrics_b = _aggregate_metrics(results_b)

    common_metrics = set(metrics_a) & set(metrics_b)
    if not common_metrics:
        raise CompareError(
            f"No common metrics found between runs. "
            f"Run A has {set(metrics_a)}, Run B has {set(metrics_b)}"
        )

    deltas: dict[str, float] = {}
    significance: dict[str, float] = {}
    effect_sizes: dict[str, float] = {}

    for metric in sorted(common_metrics):
        samples_a = np.array(metrics_a[metric], dtype=np.float64)
        samples_b = np.array(metrics_b[metric], dtype=np.float64)

        deltas[metric] = compute_deltas(samples_a, samples_b)
        significance[metric] = bootstrap_significance(
            samples_a, samples_b, n_resamples=n_resamples
        )
        effect_sizes[metric] = cohens_d(samples_a, samples_b)

    return ComparisonResult(
        run_a=run_a_id,
        run_b=run_b_id,
        metric_deltas=deltas,
        significance=significance,
        effect_sizes=effect_sizes,
        details={
            "n_items_a": len(results_a),
            "n_items_b": len(results_b),
            "n_metrics": len(common_metrics),
        },
    )


def compute_deltas(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> float:
    """Compute the mean difference ``b - a``.

    Args:
        samples_a: Per-item scores from run A.
        samples_b: Per-item scores from run B.

    Returns:
        ``mean(samples_b) - mean(samples_a)``.
    """
    return float(np.mean(samples_b) - np.mean(samples_a))


def bootstrap_significance(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    n_resamples: int = 1000,
    random_seed: int | None = 42,
) -> float:
    """Two-sided bootstrap significance test for paired differences.

    Shifts *samples_b* to have the same mean as *samples_a* under the null
    hypothesis, then resamples the paired differences with replacement.

    Args:
        samples_a: Baseline per-item scores.
        samples_b: Comparison per-item scores.
        n_resamples: Number of bootstrap resamples.
        random_seed: Seed for reproducibility. Pass ``None`` for non-deterministic.

    Returns:
        p-value in ``[0, 1]``.
    """
    if len(samples_a) != len(samples_b):
        raise CompareError(
            f"Cannot bootstrap: length mismatch ({len(samples_a)} vs {len(samples_b)}). "
            f"Consider using paired or unpaired tests accordingly."
        )
    n = len(samples_a)
    if n == 0:
        return 1.0

    observed_diff = float(np.mean(samples_b) - np.mean(samples_a))

    # Shift B to have same mean as A (null hypothesis)
    shifted_b = samples_b - np.mean(samples_b) + np.mean(samples_a)

    rng = random.Random(random_seed)
    count_extreme = 0
    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        boot_a = samples_a[indices]
        boot_b = shifted_b[indices]
        boot_diff = float(np.mean(boot_b) - np.mean(boot_a))
        if abs(boot_diff) >= abs(observed_diff):
            count_extreme += 1

    return (count_extreme + 1) / (n_resamples + 1)


def cohens_d(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
) -> float:
    """Compute Cohen's *d* effect size (pooled standard deviation).

    .. math::

        d = \\frac{\\mu_b - \\mu_a}{s_p}

    where :math:`s_p` is the pooled standard deviation.

    Args:
        samples_a: Baseline scores.
        samples_b: Comparison scores.

    Returns:
        Cohen's *d* (positive means B > A).
    """
    n_a, n_b = len(samples_a), len(samples_b)
    if n_a < 2 or n_b < 2:
        return 0.0

    var_a = float(np.var(samples_a, ddof=1))
    var_b = float(np.var(samples_b, ddof=1))

    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return 0.0

    return float((np.mean(samples_b) - np.mean(samples_a)) / pooled)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _aggregate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, list[float]]:
    """Collect per-item metric values across all results.

    Each result dict is expected to have a ``metrics_json`` key containing a
    JSON string of ``{metric_name: score, ...}``.
    """
    aggregated: dict[str, list[float]] = {}
    for item in results:
        try:
            metrics = json.loads(item["metrics_json"])
        except (json.JSONDecodeError, KeyError):
            continue
        for metric_name, score in metrics.items():
            try:
                fscore = float(score)
            except (TypeError, ValueError):
                continue
            aggregated.setdefault(metric_name, []).append(fscore)
    return aggregated


__all__ = [
    "ComparisonResult",
    "compare_runs",
    "compute_deltas",
    "bootstrap_significance",
    "cohens_d",
]
