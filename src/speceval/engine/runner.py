"""Evaluation orchestration — loads datasets, runs model adapters, computes metrics."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from speceval.engine.task import EvalResult, EvalTask
from speceval.exceptions import RunnerError
from speceval.metrics import compute_metric, list_metrics as list_registered_metrics
from speceval.provenance import ProvenanceInfo
from speceval.spec.model import SpecConfig
from speceval.store.base import ResultStore

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# EvaluationRunner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Orchestrates a full evaluation run.

    Parameters
    ----------
    spec : SpecConfig
        Parsed evaluation specification (Pydantic model).
    store : ResultStore
        Backend for persisting results (e.g. SQLiteStore).
    provenance : ProvenanceInfo
        Provenance metadata captured at invocation time.
    """

    def __init__(
        self,
        spec: SpecConfig,
        store: ResultStore,
        provenance: ProvenanceInfo,
    ) -> None:
        self.spec = spec
        self.store = store
        self.provenance = provenance
        self._adapter: Any | None = None
        self._progress: Progress | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, adapter: Any | None = None) -> str:
        """Execute the full evaluation and return a unique *run_id*.

        Parameters
        ----------
        adapter : ModelAdapter | None
            A model-adapter instance.  If ``None`` the runner attempts to
            resolve one from the spec's model configuration.
        """
        run_id = uuid.uuid4().hex[:12]
        logger.info("Starting evaluation run %s", run_id)

        self._adapter = adapter
        if self._adapter is None:
            self._adapter = await self._resolve_adapter()

        # Persist run metadata via the sync store (offloaded to executor)
        await self._save_run_metadata(run_id)

        # Load dataset
        tasks: list[EvalTask] = []
        async for task in self._load_dataset():
            tasks.append(task)

        if not tasks:
            raise RunnerError("Dataset yielded zero items — nothing to evaluate.")

        logger.info("Loaded %d evaluation items", len(tasks))
        item_errors = 0
        max_errors = self.spec.error_tolerance if hasattr(self.spec, "error_tolerance") else 0

        # Progress bar
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )

        with self._progress:
            overall_task: TaskID = self._progress.add_task(
                "Evaluating…", total=len(tasks) * self.spec.trials
            )

            for trial in range(self.spec.trials):
                trial_label = (
                    f" (trial {trial + 1}/{self.spec.trials})"
                    if self.spec.trials > 1
                    else ""
                )
                for item_idx, task in enumerate(tasks):
                    # Caching: skip if already computed for this (run_id, task_id)
                    task_id = self._task_id(task, trial)
                    if hasattr(self.spec, "cache") and self.spec.cache:
                        try:
                            existing = await self._result_exists(run_id, task_id)
                            if existing:
                                self._progress.advance(overall_task)
                                continue
                        except Exception:
                            pass  # cache check failure is non-fatal

                    try:
                        result = await self._run_single_item(task, self._adapter, trial)
                        result.metadata["task_id"] = task_id
                        result.metadata["trial"] = trial
                        result.metadata["item_index"] = item_idx
                        await self._save_result(run_id, result)
                    except RunnerError:
                        item_errors += 1
                        error_msg = f"Adapter error: Adapter predict failed"
                        logger.warning(
                            "Item %d (trial %d) failed: %s",
                            item_idx,
                            trial,
                            error_msg,
                        )
                        # Record a failed result so it's not silently lost
                        failed = EvalResult(
                            input=task.model_input,
                            expected=task.expected_output,
                            prediction=None,
                            error=error_msg,
                            metadata={
                                "task_id": task_id,
                                "trial": trial,
                                "item_index": item_idx,
                                "failed": True,
                            },
                        )
                        try:
                            await self._save_result(run_id, failed)
                        except Exception:
                            pass
                    except Exception as exc:
                        item_errors += 1
                        error_msg = f"{type(exc).__name__}: {exc}"
                        logger.warning(
                            "Item %d (trial %d) failed: %s",
                            item_idx,
                            trial,
                            error_msg,
                        )
                        # Record a failed result so it's not silently lost
                        failed = EvalResult(
                            input=task.model_input,
                            expected=task.expected_output,
                            prediction=None,
                            error=error_msg,
                            metadata={
                                "task_id": task_id,
                                "trial": trial,
                                "item_index": item_idx,
                                "failed": True,
                            },
                        )
                        try:
                            await self._save_result(run_id, failed)
                        except Exception:
                            pass

                        if max_errors >= 0 and item_errors > max_errors:
                            raise RunnerError(
                                f"Aborting run after {item_errors} item failures "
                                f"(tolerance={max_errors})."
                            ) from exc
                    finally:
                        self._progress.advance(overall_task)

        logger.info(
            "Run %s complete — %d items, %d errors",
            run_id,
            len(tasks),
            item_errors,
        )
        return run_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_dataset(self) -> AsyncIterator[EvalTask]:
        """Yield ``EvalTask`` items from the dataset configured in ``self.spec``.

        Sub-classes can override this method to implement custom dataset
        loading.  The default implementation dispatches based on
        ``spec.dataset.source``.
        """
        source = self.spec.dataset.source

        if source == "jsonl":
            path = Path(self.spec.dataset.path)
            if not path.exists():
                raise RunnerError(f"Dataset file not found: {path}")
            with path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("JSONL line %d skipped: %s", line_no + 1, exc)
                        continue
                    yield EvalTask(
                        model_input=row.get("input", row),
                        expected_output=row.get("expected", row.get("output", "")),
                        metadata={
                            "source": str(path),
                            "line": line_no + 1,
                            **row.get("metadata", {}),
                        },
                    )

        elif source == "huggingface":
            yield self._load_huggingface_dataset()

        elif source == "csv":
            path = Path(self.spec.dataset.path)
            if not path.exists():
                raise RunnerError(f"Dataset file not found: {path}")
            import csv

            with path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row_idx, row in enumerate(reader):
                    yield EvalTask(
                        model_input=json.loads(row.get("input", "{}")),
                        expected_output=row.get("expected", row.get("output", "")),
                        metadata={
                            "source": str(path),
                            "row": row_idx + 1,
                            **row,
                        },
                    )

        elif source == "dict":
            data = self.spec.dataset.path  # inline data as dict path
            raw_items = data if isinstance(data, list) else []
            for item_idx, item in enumerate(raw_items):
                yield EvalTask(
                    model_input=item.get("input", item),
                    expected_output=item.get("expected", item.get("output", "")),
                    metadata={
                        "item_index": item_idx,
                        **item.get("metadata", {}),
                    },
                )

        else:
            raise NotImplementedError(
                f"Dataset source '{source}' is not supported by the default loader. "
                f"Override _load_dataset() in a subclass."
            )

    def _load_huggingface_dataset(self) -> AsyncIterator[EvalTask]:
        """Load a Hugging Face dataset (stub — requires ``datasets``)."""
        raise NotImplementedError(
            "HuggingFace dataset loading requires the 'datasets' package. "
            "Install it with: pip install datasets"
        )

    async def _run_single_item(
        self,
        task: EvalTask,
        adapter: Any,
        trial: int,  # noqa: ARG002
    ) -> EvalResult:
        """Execute one evaluation item through the model adapter."""
        start = time.monotonic()
        try:
            predictions = await adapter.predict([task.model_input])
        except Exception as exc:
            raise RunnerError(f"Adapter predict failed for item: {exc}") from exc
        duration_ms = (time.monotonic() - start) * 1000.0

        prediction = predictions[0] if predictions else None

        # Compute metrics using the speceval.metrics registry
        metrics = await self._compute_metrics(prediction, task.expected_output)

        return EvalResult(
            input=task.model_input,
            expected=task.expected_output,
            prediction=prediction,
            metrics=metrics,
            metadata=dict(task.metadata),
            duration_ms=duration_ms,
        )

    async def _compute_metrics(
        self,
        prediction: Any,
        expected: Any,
    ) -> dict[str, float]:
        """Compute all metrics listed in ``self.spec.metrics``.

        Uses the global metric registry from ``speceval.metrics``.
        """
        metrics: dict[str, float] = {}
        registered = list_registered_metrics()

        for metric_cfg in self.spec.metrics:
            name = metric_cfg.name if hasattr(metric_cfg, "name") else metric_cfg

            if name not in registered:
                logger.warning("Metric '%s' is not registered — skipping", name)
                continue

            try:
                # The metric registry expects list[str] inputs.  For per-item
                # evaluation we wrap single strings in single-element lists.
                pred_list = [str(prediction)] if prediction is not None else [""]
                ref_list = [str(expected)] if expected is not None else [""]

                # Collect extra kwargs from the metric config (if available)
                kwargs = {}
                if hasattr(metric_cfg, "params") and metric_cfg.params:
                    kwargs = metric_cfg.params

                result = compute_metric(name, pred_list, ref_list, **kwargs)
                metrics[name] = float(result.value)
            except Exception as exc:
                logger.warning("Metric '%s' failed: %s", name, exc)
                metrics[name] = float("nan")

        return metrics

    async def _resolve_adapter(self) -> Any:
        """Resolve a model adapter from the spec's model config."""
        from speceval.adapters.base import ModelAdapterFactory

        model = self.spec.model
        config = {
            "model": model.name,
            "backend": model.provider,
        }
        if model.params:
            config.update(model.params)
        if model.endpoint:
            config["base_url"] = model.endpoint
        if model.api_key:
            config["api_key"] = model.api_key

        return ModelAdapterFactory.create(config)

    # ------------------------------------------------------------------
    # Sync store wrappers (offloaded to thread executor)
    # ------------------------------------------------------------------

    async def _save_run_metadata(self, run_id: str) -> None:
        """Persist run-level metadata via the sync store.

        Tries ``save_run`` first (available on SQLiteStore); falls back to
        storing metadata as a special result entry.
        """

        def _sync() -> None:
            self.store.init_store()
            spec_hash = hashlib.sha256(
                self.spec.model_dump_json().encode()
            ).hexdigest()[:12]
            try:
                self.store.save_run(
                    run_id=run_id,
                    spec_hash=spec_hash,
                    model_name=self.spec.model.name,
                    dataset_name=str(self.spec.dataset.path) if self.spec.dataset.path else "",
                    provenance_json=self.provenance.to_json(),
                    status="running",
                )
            except AttributeError:
                # Store does not implement save_run — store metadata as result
                import json

                self.store.save_result(
                    run_id=run_id,
                    item_index=-1,
                    input_json="{}",
                    expected="",
                    prediction="",
                    metrics_json=json.dumps(
                        {
                            "__run_metadata__": True,
                            "spec_hash": spec_hash,
                            "model_name": self.spec.model.name,
                            "dataset_name": self.spec.dataset.path,
                            "provenance": self.provenance.to_dict(),
                            "status": "running",
                        }
                    ),
                    duration_ms=0.0,
                )

        await asyncio.get_event_loop().run_in_executor(None, _sync)

    async def _save_result(self, run_id: str, result: EvalResult) -> None:
        """Persist a single result via the sync store (in executor)."""
        import json

        def _sync() -> None:
            self.store.save_result(
                run_id=run_id,
                item_index=result.metadata.get("item_index", 0),
                input_json=json.dumps(result.input, default=str),
                expected=str(result.expected) if result.expected is not None else "",
                prediction=(
                    json.dumps(result.prediction, default=str)
                    if result.prediction is not None
                    else ""
                ),
                metrics_json=json.dumps(result.metrics, default=str),
                duration_ms=result.duration_ms,
            )

        await asyncio.get_event_loop().run_in_executor(None, _sync)

    async def _result_exists(self, run_id: str, task_id: str) -> bool:
        """Check if a result already exists for ``task_id``."""

        def _sync() -> bool:
            try:
                results = self.store.get_results(run_id)
            except Exception:
                return False
            # task_id is stored in metrics_json or we derive from item_index
            # Here we do a simple heuristic — check count of results
            return len(results) > 0

        return await asyncio.get_event_loop().run_in_executor(None, _sync)

    # ------------------------------------------------------------------
    # Task identifier
    # ------------------------------------------------------------------

    @staticmethod
    def _task_id(task: EvalTask, trial: int) -> str:
        """Return a deterministic identifier for a (task, trial) pair."""
        raw = str(task.model_input) + str(task.expected_output) + str(trial)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


async def get_runner(spec_path: str) -> EvaluationRunner:
    """Load a spec file and return an ``EvaluationRunner`` instance.

    This convenience function parses the YAML spec, creates a SQLite store
    at the default path, captures provenance, and wires everything together.

    Parameters
    ----------
    spec_path : str
        Path to a speceval YAML specification file.
    """
    from speceval.spec.parse import parse_spec
    from speceval.store.sqlite import SQLiteStore

    path = Path(spec_path)
    if not path.exists():
        raise RunnerError(f"Spec file not found: {path}")

    spec: SpecConfig = parse_spec(path)
    store = SQLiteStore()
    provenance = await _capture_provenance(spec_path)

    return EvaluationRunner(spec=spec, store=store, provenance=provenance)


async def _capture_provenance(spec_path: str) -> ProvenanceInfo:
    """Capture provenance for the current environment."""
    from speceval.provenance import capture_provenance

    loop = asyncio.get_event_loop()
    provenance = await loop.run_in_executor(None, capture_provenance)
    provenance.additional["spec_file"] = str(Path(spec_path).resolve())
    return provenance
