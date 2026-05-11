"""Data types for individual evaluation items and their results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalTask:
    """A single evaluation item — one (model_input, expected_output, metadata) triplet.

    Attributes
    ----------
    model_input : dict
        The input sent to the model (e.g. {"messages": [...]} or {"prompt": "..."}).
    expected_output : Any
        The reference / ground-truth output (type depends on the metric).
    metadata : dict
        Arbitrary metadata attached to this item (e.g. source row, index, split).
    """

    model_input: dict
    expected_output: Any
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """The outcome of running a single EvalTask through a model.

    Attributes
    ----------
    input : dict
        The original model input (copied from EvalTask).
    expected : Any
        The original expected output (copied from EvalTask).
    prediction : Any
        The output produced by the model (or the error message string if failed).
    metrics : dict
        Dictionary of computed metric names to numeric values.
    metadata : dict
        The metadata dict from the original EvalTask, enriched with run context.
    duration_ms : float
        Wall-clock duration of the prediction call in milliseconds.
    error : str | None
        If the item failed, a description of the error; otherwise None.
    """

    input: dict
    expected: Any
    prediction: Any
    metrics: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None
