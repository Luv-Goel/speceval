"""Run comparison utilities."""

from __future__ import annotations

from .delta import (
    ComparisonResult,
    compare_runs,
    compute_deltas,
    bootstrap_significance,
    cohens_d,
)

__all__ = [
    "ComparisonResult",
    "compare_runs",
    "compute_deltas",
    "bootstrap_significance",
    "cohens_d",
]
