"""Metrics registry and computation for speceval."""

from __future__ import annotations

import logging
from typing import Any

from speceval.exceptions import MetricError

from . import classification, generation
from .base import MetricFn, MetricResult

logger = logging.getLogger(__name__)

_registry: dict[str, MetricFn] = {}
_registry_initialized: bool = False


def register(name: str, fn: MetricFn) -> None:
    """Register a metric function under *name*.

    Args:
        name: Metric name (e.g. ``"accuracy"``).
        fn: Callable with signature ``(predictions, references, **kwargs) -> float``.

    Raises:
        MetricError: If *name* is already registered.
    """
    if name in _registry:
        raise MetricError(f"Metric {name!r} is already registered")
    _registry[name] = fn
    logger.debug("Registered metric %r", name)


def get(name: str) -> MetricFn:
    """Look up a registered metric by name.

    Args:
        name: Registered metric name.

    Returns:
        The metric callable.

    Raises:
        MetricError: If *name* is not registered.
    """
    if name not in _registry:
        raise MetricError(f"Metric {name!r} is not registered. Available: {list_metrics()}")
    return _registry[name]


def list_metrics() -> list[str]:
    """Return sorted list of registered metric names."""
    return sorted(_registry)


def compute_metric(
    name: str,
    predictions: list[str],
    references: list[str],
    **kwargs: Any,
) -> MetricResult:
    """Compute a named metric on the given predictions and references.

    Args:
        name: Metric name (must be registered).
        predictions: Model outputs.
        references: Ground-truth answers.
        **kwargs: Extra keyword arguments forwarded to the metric function.

    Returns:
        A ``MetricResult`` namedtuple with ``name``, ``value``, and ``details``.

    Raises:
        MetricError: On failure.
    """
    fn = get(name)
    try:
        value = fn(predictions, references, **kwargs)
    except Exception as exc:
        raise MetricError(f"Metric {name!r} failed: {exc}") from exc
    return MetricResult(name=name, value=value, details={})


def register_all(force: bool = False) -> None:
    """Register all built-in metrics.

    Args:
        force: If True, clear existing registry before re-registering.
    """
    global _registry_initialized
    if not force and _registry_initialized:
        return

    _registry.clear()
    # Classification metrics
    register("accuracy", classification.accuracy)
    register("f1", classification.f1_score)
    register("precision", classification.precision)
    register("recall", classification.recall)
    # Generation metrics
    register("exact_match", generation.exact_match)
    register("bleu", generation.bleu)
    register("rouge_l", generation.rouge_l)
    register("perplexity", generation.perplexity)

    _registry_initialized = True
    logger.info("All built-in metrics registered (%d total)", len(_registry))


__all__ = [
    "register",
    "get",
    "list_metrics",
    "compute_metric",
    "register_all",
    "MetricFn",
    "MetricResult",
]
