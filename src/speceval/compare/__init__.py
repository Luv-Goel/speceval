"""Run comparison utilities."""

from __future__ import annotations

from .delta import (
    ComparisonResult,
    bootstrap_significance,
    cohens_d,
    compare_runs,
    compute_deltas,
)

__all__ = [
    "ComparisonResult",
    "compare_runs",
    "compute_deltas",
    "bootstrap_significance",
    "cohens_d",
]
