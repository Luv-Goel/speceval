"""Abstract result store interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ResultStore(ABC):
    """Abstract base for persisting evaluation results.

    Implementations must be thread-safe where documented; all raise
    ``StoreError`` (from ``speceval.exceptions``) on failure.
    """

    @abstractmethod
    def init_store(self) -> None:
        """Create (or connect to) the underlying storage.

        Called once before any other operations.
        """

    @abstractmethod
    def save_result(
        self,
        *,
        run_id: str,
        item_index: int,
        input_json: str,
        expected: str,
        prediction: str,
        metrics_json: str,
        duration_ms: float,
    ) -> None:
        """Persist a single evaluation item result.

        Args:
            run_id: Unique run identifier.
            item_index: Zero-based index of the item in the dataset.
            input_json: Serialised input prompt / data.
            expected: Ground-truth reference string.
            prediction: Model output string.
            metrics_json: JSON-serialised dict of metric values.
            duration_ms: Inference latency in milliseconds.
        """

    @abstractmethod
    def get_results(self, run_id: str) -> list[dict[str, Any]]:
        """Return all item-level results for a run.

        Args:
            run_id: Run identifier.

        Returns:
            List of dicts with keys matching the ``save_result`` parameters.
        """

    @abstractmethod
    def get_runs(self) -> list[dict[str, Any]]:
        """Return metadata for every recorded run.

        Returns:
            List of dicts with keys ``id``, ``spec_hash``, ``model_name``,
            ``dataset_name``, ``timestamp``, ``provenance_json``, ``status``.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the store (e.g. database connections)."""


__all__ = ["ResultStore"]
