"""Tests for SQLite result store."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from speceval.exceptions import StoreError
from speceval.store.sqlite import SQLiteStore


class TestSQLiteStoreInit:
    """Tests for store initialization."""

    def test_init_memory(self):
        """In-memory store initializes without error."""
        store = SQLiteStore(":memory:")
        store.init_store()
        assert store._conn is not None
        store.close()

    def test_init_file(self, temp_dir: Path):
        """File-based store creates database file."""
        db_path = temp_dir / "test_results.db"
        store = SQLiteStore(str(db_path))
        store.init_store()
        assert db_path.exists()
        store.close()

    def test_init_twice(self):
        """Calling init_store twice is safe."""
        store = SQLiteStore(":memory:")
        store.init_store()
        store.init_store()  # second call should not raise
        store.close()

    def test_close_twice(self):
        """Calling close twice is safe."""
        store = SQLiteStore(":memory:")
        store.init_store()
        store.close()
        store.close()  # should not raise


class TestSQLiteStoreSaveGet:
    """Tests for saving and retrieving results."""

    def test_save_and_get_results(self, sqlite_store: SQLiteStore):
        """Saved results can be retrieved."""
        store = sqlite_store
        store.save_run(
            run_id="test_run",
            spec_hash="abc",
            model_name="gpt-4",
            dataset_name="test_data",
            provenance_json="{}",
            status="completed",
        )
        store.save_result(
            run_id="test_run",
            item_index=0,
            input_json='{"prompt": "test"}',
            expected="expected_output",
            prediction="model_output",
            metrics_json='{"accuracy": 0.95}',
            duration_ms=100.0,
        )
        results = store.get_results("test_run")
        assert len(results) == 1
        assert results[0]["item_index"] == 0
        assert results[0]["expected"] == "expected_output"
        assert results[0]["prediction"] == "model_output"
        assert json.loads(results[0]["metrics_json"]) == {"accuracy": 0.95}

    def test_get_results_empty(self, sqlite_store: SQLiteStore):
        """Getting results for a run with no results returns empty list."""
        results = sqlite_store.get_results("nonexistent")
        assert results == []

    def test_multiple_results(self, sqlite_store: SQLiteStore):
        """Multiple results for same run are returned in order."""
        store = sqlite_store
        store.save_run(
            run_id="multi_run",
            spec_hash="abc",
            model_name="gpt-4",
            dataset_name="test_data",
            provenance_json="{}",
            status="completed",
        )
        for i in range(5):
            store.save_result(
                run_id="multi_run",
                item_index=i,
                input_json=json.dumps({"idx": i}),
                expected=f"exp_{i}",
                prediction=f"pred_{i}",
                metrics_json=json.dumps({"accuracy": 0.8 + i * 0.05}),
                duration_ms=50.0 + i * 10.0,
            )
        results = store.get_results("multi_run")
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r["item_index"] == i

    def test_results_order_by_item_index(self, sqlite_store: SQLiteStore):
        """Results are ordered by item_index."""
        store = sqlite_store
        store.save_run(
            run_id="ordered_run",
            spec_hash="abc",
            model_name="gpt-4",
            dataset_name="test_data",
            provenance_json="{}",
            status="completed",
        )
        indices = [3, 1, 4, 0, 2]
        for i in indices:
            store.save_result(
                run_id="ordered_run",
                item_index=i,
                input_json="{}",
                expected="",
                prediction="",
                metrics_json="{}",
                duration_ms=0.0,
            )
        results = store.get_results("ordered_run")
        item_indices = [r["item_index"] for r in results]
        assert item_indices == sorted(item_indices)

    def test_save_without_init_raises(self):
        """Saving without init_store raises StoreError."""
        store = SQLiteStore(":memory:")
        with pytest.raises(StoreError, match="not initialised"):
            store.save_result(
                run_id="test",
                item_index=0,
                input_json="{}",
                expected="",
                prediction="",
                metrics_json="{}",
                duration_ms=0.0,
            )

    def test_get_without_init_raises(self):
        """Getting without init_store raises StoreError."""
        store = SQLiteStore(":memory:")
        with pytest.raises(StoreError, match="not initialised"):
            store.get_results("test")


class TestSQLiteStoreRuns:
    """Tests for run metadata management."""

    def test_save_and_get_run(self, sqlite_store: SQLiteStore):
        """Saved run metadata can be retrieved."""
        store = sqlite_store
        store.save_run(
            run_id="run_1",
            spec_hash="abc123",
            model_name="gpt-4",
            dataset_name="test_data",
            provenance_json=json.dumps({"user": "test"}),
            status="completed",
        )
        run = store.get_run("run_1")
        assert run is not None
        assert run["id"] == "run_1"
        assert run["spec_hash"] == "abc123"
        assert run["model_name"] == "gpt-4"
        assert run["dataset_name"] == "test_data"
        assert run["status"] == "completed"

    def test_get_run_not_found(self, sqlite_store: SQLiteStore):
        """Getting a non-existent run returns None."""
        run = sqlite_store.get_run("nonexistent")
        assert run is None

    def test_get_runs_empty(self, sqlite_store: SQLiteStore):
        """get_runs returns empty list when no runs exist."""
        runs = sqlite_store.get_runs()
        assert runs == []

    def test_get_runs_multiple(self, sqlite_store: SQLiteStore):
        """get_runs returns all saved runs."""
        store = sqlite_store
        store.save_run(run_id="run_a", spec_hash="a", model_name="m1", dataset_name="d1",
                       provenance_json="{}", status="completed")
        store.save_run(run_id="run_b", spec_hash="b", model_name="m2", dataset_name="d2",
                       provenance_json="{}", status="running")

        runs = store.get_runs()
        assert len(runs) == 2
        run_ids = {r["id"] for r in runs}
        assert run_ids == {"run_a", "run_b"}

    def test_get_runs_ordered_by_timestamp_desc(self, sqlite_store: SQLiteStore):
        """get_runs orders by timestamp descending."""
        store = sqlite_store
        store.save_run(run_id="first", spec_hash="a", model_name="m", dataset_name="d",
                       provenance_json="{}", status="completed")
        store.save_run(run_id="second", spec_hash="b", model_name="m", dataset_name="d",
                       provenance_json="{}", status="completed")

        runs = store.get_runs()
        assert len(runs) >= 2
        # The most recent should be first
        assert runs[0]["timestamp"] >= runs[-1]["timestamp"]

    def test_save_run_updates(self, sqlite_store: SQLiteStore):
        """Save run with same ID updates existing record."""
        store = sqlite_store
        store.save_run(run_id="run_1", spec_hash="abc", model_name="m", dataset_name="d",
                       provenance_json="{}", status="running")
        store.save_run(run_id="run_1", spec_hash="abc", model_name="m", dataset_name="d",
                       provenance_json="{}", status="completed")

        run = store.get_run("run_1")
        assert run["status"] == "completed"

    def test_save_run_without_init_raises(self):
        """save_run without init_store raises StoreError."""
        store = SQLiteStore(":memory:")
        with pytest.raises(StoreError, match="not initialised"):
            store.save_run(run_id="test", spec_hash="a", model_name="m",
                           dataset_name="d", provenance_json="{}")


class TestSQLiteStorePersistence:
    """Tests for persistence across close/reopen."""

    @pytest.mark.skipif(sys.platform.startswith("win"), reason="Windows temp file locking")
    def test_persistence_to_file(self):
        """Data persists when store is closed and reopened."""
        import os
        import tempfile
        tmp = tempfile.mktemp(suffix=".db", prefix="speceval_persist_")
        db_path = tmp

        # Write
        store = SQLiteStore(db_path)
        store.init_store()
        store.save_run(
            run_id="persisted_run",
            spec_hash="abc",
            model_name="m",
            dataset_name="d",
            provenance_json="{}",
        )
        store.save_result(
            run_id="persisted_run",
            item_index=0,
            input_json="{}",
            expected="exp",
            prediction="pred",
            metrics_json='{"score": 1.0}',
            duration_ms=50.0,
        )
        store.save_run(
            run_id="persisted_run",
            spec_hash="abc",
            model_name="m",
            dataset_name="d",
            provenance_json="{}",
        )
        store.close()

        # Read back
        store2 = SQLiteStore(db_path)
        store2.init_store()
        results = store2.get_results("persisted_run")
        assert len(results) == 1
        run = store2.get_run("persisted_run")
        assert run is not None
        store2.close()
        # Cleanup temp db files
        for f in [db_path, db_path + "-wal", db_path + "-shm"]:
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass


class TestSQLiteStoreErrorHandling:
    """Tests for error handling."""

    def test_get_run_without_init_raises(self):
        """get_run without init raises StoreError."""
        store = SQLiteStore(":memory:")
        with pytest.raises(StoreError, match="not initialised"):
            store.get_run("test")

    def test_get_runs_without_init_raises(self):
        """get_runs without init raises StoreError."""
        store = SQLiteStore(":memory:")
        with pytest.raises(StoreError, match="not initialised"):
            store.get_runs()
