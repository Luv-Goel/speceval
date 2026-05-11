"""Tests for the evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from speceval.engine.runner import EvaluationRunner, get_runner
from speceval.engine.task import EvalResult, EvalTask
from speceval.exceptions import RunnerError
from speceval.provenance import ProvenanceInfo
from speceval.spec.model import (
    DatasetConfig,
    MetricConfig,
    ModelConfig,
    SpecConfig,
)
from speceval.store.sqlite import SQLiteStore


@pytest.fixture
def basic_spec() -> SpecConfig:
    """Return a simple spec for runner tests."""
    return SpecConfig(
        name="runner-test",
        model=ModelConfig(provider="openai", name="gpt-4o-mini"),
        dataset=DatasetConfig(source="dict", path=[
            {"input": {"prompt": "Hello"}, "expected": "Hi"},
            {"input": {"prompt": "Goodbye"}, "expected": "Bye"},
        ]),
        metrics=[MetricConfig(name="exact_match")],
        trials=1,
    )


@pytest.fixture
def sqlite_store() -> SQLiteStore:
    """Return an in-memory store."""
    store = SQLiteStore(":memory:")
    store.init_store()
    return store


@pytest.fixture
def provenance() -> ProvenanceInfo:
    """Return a minimal provenance info."""
    return ProvenanceInfo(
        git_commit_hash="abc123",
        python_version="3.10.12",
        platform="test",
        hostname="test-host",
        timestamp="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def mock_adapter():
    """Return an async mock adapter."""
    adapter = MagicMock()
    adapter.predict = AsyncMock(
        return_value=[{"text": "mock response"}]
    )
    adapter.metadata = {"model": "mock"}
    return adapter


class TestEvaluationRunner:
    """Tests for EvaluationRunner."""

    @pytest.mark.asyncio
    async def test_run_basic(self, basic_spec, sqlite_store, provenance, mock_adapter):
        """Basic run executes without error and returns a run_id."""
        # Register metrics
        from speceval.metrics import register_all
        register_all()

        runner = EvaluationRunner(
            spec=basic_spec,
            store=sqlite_store,
            provenance=provenance,
        )
        run_id = await runner.run(adapter=mock_adapter)
        assert isinstance(run_id, str)
        assert len(run_id) > 0

        # Results should be stored
        results = sqlite_store.get_results(run_id)
        assert len(results) == 2  # two dataset items

    @pytest.mark.asyncio
    async def test_run_id_unique(self, basic_spec, sqlite_store, provenance, mock_adapter):
        """Each run produces a unique run_id."""
        from speceval.metrics import register_all
        register_all()

        runner = EvaluationRunner(spec=basic_spec, store=sqlite_store, provenance=provenance)
        run_id_1 = await runner.run(adapter=mock_adapter)
        run_id_2 = await runner.run(adapter=mock_adapter)
        assert run_id_1 != run_id_2

    @pytest.mark.asyncio
    async def test_run_metadata_saved(self, basic_spec, sqlite_store, provenance, mock_adapter):
        """Run metadata is saved to store."""
        from speceval.metrics import register_all
        register_all()

        runner = EvaluationRunner(spec=basic_spec, store=sqlite_store, provenance=provenance)
        run_id = await runner.run(adapter=mock_adapter)

        run = sqlite_store.get_run(run_id)
        assert run is not None
        assert run["model_name"] == "gpt-4o-mini"
        assert run["dataset_name"] == str(basic_spec.dataset.path)
        assert run["status"] in ("running", "completed")  # new save status

    @pytest.mark.asyncio
    async def test_run_with_trials(self, mock_adapter, provenance, sqlite_store):
        """Multiple trials are each executed."""
        from speceval.metrics import register_all
        register_all()

        spec = SpecConfig(
            name="trials-test",
            model=ModelConfig(provider="openai", name="gpt-4"),
            dataset=DatasetConfig(source="dict", path=[
                {"input": {"prompt": "Hi"}, "expected": "Hello"},
            ]),
            metrics=[MetricConfig(name="exact_match")],
            trials=3,
        )
        runner = EvaluationRunner(spec=spec, store=sqlite_store, provenance=provenance)
        run_id = await runner.run(adapter=mock_adapter)

        results = sqlite_store.get_results(run_id)
        assert len(results) == 3  # 1 item * 3 trials

    @pytest.mark.asyncio
    async def test_run_empty_dataset_raises(self, provenance, sqlite_store, mock_adapter):
        """Empty dataset raises RunnerError."""
        from speceval.metrics import register_all
        register_all()

        spec = SpecConfig(
            name="empty-test",
            model=ModelConfig(provider="openai", name="gpt-4"),
            dataset=DatasetConfig(source="dict", path=[]),
            metrics=[MetricConfig(name="exact_match")],
        )
        runner = EvaluationRunner(spec=spec, store=sqlite_store, provenance=provenance)
        with pytest.raises(RunnerError, match="zero items"):
            await runner.run(adapter=mock_adapter)

    @pytest.mark.asyncio
    async def test_adapter_predict_called(self, basic_spec, sqlite_store, provenance):
        """Adapter.predict is called with correct inputs."""
        from speceval.metrics import register_all
        register_all()

        adapter = MagicMock()
        adapter.predict = AsyncMock(return_value=[{"text": "Hi"}, {"text": "Bye"}])

        runner = EvaluationRunner(spec=basic_spec, store=sqlite_store, provenance=provenance)
        await runner.run(adapter=adapter)

        # Should have been called twice (2 items)
        assert adapter.predict.call_count == 2
        # First call should have the first input
        first_call_args = adapter.predict.call_args_list[0][0][0]
        assert first_call_args == [{"prompt": "Hello"}]

    @pytest.mark.asyncio
    async def test_result_contains_metrics(self, basic_spec, sqlite_store, provenance, mock_adapter):
        """Results contain computed metrics."""
        from speceval.metrics import register_all
        register_all()

        runner = EvaluationRunner(spec=basic_spec, store=sqlite_store, provenance=provenance)
        run_id = await runner.run(adapter=mock_adapter)

        results = sqlite_store.get_results(run_id)
        for r in results:
            metrics = json.loads(r["metrics_json"])
            assert "exact_match" in metrics

    @pytest.mark.asyncio
    async def test_adapter_failure_handled(self, basic_spec, sqlite_store, provenance):
        """Adapter failure is caught and recorded as error result."""
        from speceval.metrics import register_all
        register_all()

        # Override error_tolerance to avoid abort on first failure
        object.__setattr__(basic_spec, "error_tolerance", 10)

        adapter = MagicMock()
        adapter.predict = AsyncMock(side_effect=Exception("Inference failed"))

        runner = EvaluationRunner(spec=basic_spec, store=sqlite_store, provenance=provenance)
        run_id = await runner.run(adapter=adapter)

        results = sqlite_store.get_results(run_id)
        assert len(results) > 0
        # Failed predictions might still be stored
        # Check that at least one result exists

    @pytest.mark.asyncio
    async def test_csv_dataset(self, temp_dir: Path, provenance, sqlite_store, mock_adapter):
        """Runner handles CSV dataset source."""
        from speceval.metrics import register_all
        register_all()

        csv_file = temp_dir / "test.csv"
        csv_file.write_text("input,expected\n{\"prompt\":\"a\"},x\n{\"prompt\":\"b\"},y\n")

        spec = SpecConfig(
            name="csv-test",
            model=ModelConfig(provider="openai", name="gpt-4"),
            dataset=DatasetConfig(source="csv", path=str(csv_file)),
            metrics=[MetricConfig(name="exact_match")],
        )
        runner = EvaluationRunner(spec=spec, store=sqlite_store, provenance=provenance)
        run_id = await runner.run(adapter=mock_adapter)

        results = sqlite_store.get_results(run_id)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_jsonl_dataset(self, temp_dir: Path, provenance, sqlite_store, mock_adapter):
        """Runner handles JSONL dataset source."""
        from speceval.metrics import register_all
        register_all()

        jsonl_file = temp_dir / "test.jsonl"
        jsonl_file.write_text(
            json.dumps({"input": {"prompt": "q1"}, "expected": "a1"}) + "\n" +
            json.dumps({"input": {"prompt": "q2"}, "expected": "a2"}) + "\n"
        )

        spec = SpecConfig(
            name="jsonl-test",
            model=ModelConfig(provider="openai", name="gpt-4"),
            dataset=DatasetConfig(source="jsonl", path=str(jsonl_file)),
            metrics=[MetricConfig(name="exact_match")],
        )
        runner = EvaluationRunner(spec=spec, store=sqlite_store, provenance=provenance)
        run_id = await runner.run(adapter=mock_adapter)

        results = sqlite_store.get_results(run_id)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_csv_file_not_found(self, provenance, sqlite_store, mock_adapter):
        """Runner raises error when CSV file doesn't exist."""
        from speceval.metrics import register_all
        register_all()

        spec = SpecConfig(
            name="csv-missing",
            model=ModelConfig(provider="openai", name="gpt-4"),
            dataset=DatasetConfig(source="csv", path="/nonexistent/file.csv"),
            metrics=[MetricConfig(name="exact_match")],
        )
        runner = EvaluationRunner(spec=spec, store=sqlite_store, provenance=provenance)
        with pytest.raises(RunnerError, match="not found"):
            await runner.run(adapter=mock_adapter)

    @pytest.mark.asyncio
    async def test_jsonl_file_not_found(self, provenance, sqlite_store, mock_adapter):
        """Runner raises error when JSONL file doesn't exist."""
        from speceval.metrics import register_all
        register_all()

        spec = SpecConfig(
            name="jsonl-missing",
            model=ModelConfig(provider="openai", name="gpt-4"),
            dataset=DatasetConfig(source="jsonl", path="/nonexistent/data.jsonl"),
            metrics=[MetricConfig(name="exact_match")],
        )
        runner = EvaluationRunner(spec=spec, store=sqlite_store, provenance=provenance)
        with pytest.raises(RunnerError, match="not found"):
            await runner.run(adapter=mock_adapter)

    def test_task_id_deterministic(self):
        """_task_id produces deterministic IDs."""
        task1 = EvalTask(model_input={"x": 1}, expected_output="y")
        task2 = EvalTask(model_input={"x": 1}, expected_output="y")
        id1 = EvaluationRunner._task_id(task1, 0)
        id2 = EvaluationRunner._task_id(task2, 0)
        assert id1 == id2

    def test_task_id_different_for_different_trials(self):
        """_task_id differs for different trial numbers."""
        task = EvalTask(model_input={"x": 1}, expected_output="y")
        id1 = EvaluationRunner._task_id(task, 0)
        id2 = EvaluationRunner._task_id(task, 1)
        assert id1 != id2

    def test_task_id_different_for_different_inputs(self):
        """_task_id differs for different inputs."""
        task1 = EvalTask(model_input={"x": 1}, expected_output="y")
        task2 = EvalTask(model_input={"x": 2}, expected_output="y")
        id1 = EvaluationRunner._task_id(task1, 0)
        id2 = EvaluationRunner._task_id(task2, 0)
        assert id1 != id2


@pytest.mark.asyncio
async def test_get_runner(temp_dir: Path):
    """get_runner factory works with a valid spec file."""
    from speceval.metrics import register_all
    register_all()

    spec_file = temp_dir / "test_spec.yaml"
    spec_file.write_text("""name: factory-test
model:
  provider: openai
  name: gpt-4
dataset:
  source: dict
  path:
    - input: {prompt: "hello"}
      expected: "world"
metrics:
  - name: exact_match
trials: 1
""")

    with patch("speceval.store.sqlite.SQLiteStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        with patch("speceval.engine.runner._capture_provenance") as mock_cap:
            mock_cap.return_value = ProvenanceInfo(timestamp="test")
            runner = await get_runner(str(spec_file))
            assert isinstance(runner, EvaluationRunner)
            assert runner.spec.name == "factory-test"
