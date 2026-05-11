"""Classification metrics — accuracy, precision, recall, F1."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.metrics import accuracy_score as _sk_accuracy
    from sklearn.metrics import f1_score as _sk_f1
    from sklearn.metrics import precision_score as _sk_precision
    from sklearn.metrics import recall_score as _sk_recall

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


def _to_labels(predictions: list[str], references: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Convert string predictions/references to integer label arrays.

    All unique strings across both lists are collected and mapped to integers.
    """
    all_labels = sorted(set(predictions) | set(references))
    mapping = {lab: i for i, lab in enumerate(all_labels)}
    y_pred = np.array([mapping[p] for p in predictions], dtype=np.int32)
    y_true = np.array([mapping[r] for r in references], dtype=np.int32)
    return y_pred, y_true


def accuracy(predictions: list[str], references: list[str], **kwargs: Any) -> float:
    """Compute accuracy (fraction of exact matches).

    Args:
        predictions: Predicted labels.
        references: Ground-truth labels.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references length mismatch: "
            f"{len(predictions)} vs {len(references)}"
        )
    if len(predictions) == 0:
        return 0.0

    if _HAS_SKLEARN:
        return float(_sk_accuracy(references, predictions))

    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(predictions)


def precision(predictions: list[str], references: list[str], **kwargs: Any) -> float:
    """Compute macro-averaged precision.

    Args:
        predictions: Predicted labels.
        references: Ground-truth labels.

    Returns:
        Macro precision in ``[0, 1]``.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references length mismatch: "
            f"{len(predictions)} vs {len(references)}"
        )
    if len(predictions) == 0:
        return 0.0
    if _HAS_SKLEARN:
        return float(_sk_precision(references, predictions, average="macro", zero_division=0.0))

    return _macro_binary_metric(predictions, references, metric="precision")


def recall(predictions: list[str], references: list[str], **kwargs: Any) -> float:
    """Compute macro-averaged recall.

    Args:
        predictions: Predicted labels.
        references: Ground-truth labels.

    Returns:
        Macro recall in ``[0, 1]``.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references length mismatch: "
            f"{len(predictions)} vs {len(references)}"
        )
    if len(predictions) == 0:
        return 0.0
    if _HAS_SKLEARN:
        return float(_sk_recall(references, predictions, average="macro", zero_division=0.0))

    return _macro_binary_metric(predictions, references, metric="recall")


def f1_score(predictions: list[str], references: list[str], **kwargs: Any) -> float:
    r"""Compute macro-averaged F\ :sub:`1` score.

    Args:
        predictions: Predicted labels.
        references: Ground-truth labels.

    Returns:
        Macro F1 in ``[0, 1]``.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references length mismatch: "
            f"{len(predictions)} vs {len(references)}"
        )
    if len(predictions) == 0:
        return 0.0
    if _HAS_SKLEARN:
        return float(_sk_f1(references, predictions, average="macro", zero_division=0.0))

    return _macro_binary_metric(predictions, references, metric="f1")


# ---------------------------------------------------------------------------
# Manual numpy fallback helpers
# ---------------------------------------------------------------------------


def _macro_binary_metric(
    predictions: list[str],
    references: list[str],
    metric: str = "f1",
) -> float:
    """Compute macro-averaged binary metrics without sklearn."""
    y_pred, y_true = _to_labels(predictions, references)
    classes = sorted(set(y_true) | set(y_pred))
    scores = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))

        if metric == "precision":
            denom = tp + fp
            scores.append(float(tp / denom) if denom > 0 else 0.0)
        elif metric == "recall":
            denom = tp + fn
            scores.append(float(tp / denom) if denom > 0 else 0.0)
        else:  # f1
            denom_p = tp + fp
            denom_r = tp + fn
            p = tp / denom_p if denom_p > 0 else 0.0
            r = tp / denom_r if denom_r > 0 else 0.0
            scores.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


__all__ = ["accuracy", "precision", "recall", "f1_score"]
