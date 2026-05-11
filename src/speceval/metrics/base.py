"""Base types for metric definitions."""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol


class MetricFn(Protocol):
    """Protocol for a metric function.

    A metric function takes two parallel lists of strings (predictions and
    references) plus optional keyword arguments and returns a single ``float``
    score.
    """

    def __call__(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> float:
        ...


class MetricResult(NamedTuple):
    """Result of a single metric computation."""

    name: str
    """Short metric identifier (e.g. ``"accuracy"``)."""

    value: float
    """The scalar score."""

    details: dict[str, Any]
    """Optional per-metric detailed breakdown."""


__all__ = ["MetricFn", "MetricResult"]
