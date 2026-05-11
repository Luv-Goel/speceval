"""Shared fixtures for speceval tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from speceval.spec.model import (
    AssertionConfig,
    DatasetConfig,
    EnvConfig,
    MetricConfig,
    ModelConfig,
    SpecConfig,
)
from speceval.store.sqlite import SQLiteStore


@pytest.fixture
def sample_spec() -> SpecConfig:
    """Return a fully-populated SpecConfig for testing."""
    return SpecConfig(
        name="test-evaluation",
        description="A sample evaluation spec for testing",
        model=ModelConfig(
            provider="openai",
            name="gpt-4o-mini",
            params={"temperature": 0.0, "max_tokens": 256},
        ),
        dataset=DatasetConfig(
            source="jsonl",
            path="./test_data.jsonl",
            limit=10,
        ),
        metrics=[
            MetricConfig(name="exact_match"),
            MetricConfig(name="bleu", params={"max_n": 4}),
        ],
        trials=1,
        seeds={},
        env=EnvConfig(
            seeds={"global": 42},
            env_vars={"TEST_VAR": "hello"},
        ),
        assertions=[
            AssertionConfig(
                metric_name="exact_match",
                operator="gte",
                value=0.5,
                description="Exact match must be at least 0.5",
            ),
        ],
    )


@pytest.fixture
def sample_spec_dict() -> dict[str, Any]:
    """Return a raw dict matching sample_spec, as if parsed from YAML."""
    return {
        "name": "test-evaluation",
        "description": "A sample evaluation spec for testing",
        "model": {
            "provider": "openai",
            "name": "gpt-4o-mini",
            "params": {"temperature": 0.0, "max_tokens": 256},
        },
        "dataset": {
            "source": "jsonl",
            "path": "./test_data.jsonl",
            "limit": 10,
        },
        "metrics": [
            {"name": "exact_match"},
            {"name": "bleu", "params": {"max_n": 4}},
        ],
        "trials": 1,
        "env": {
            "seeds": {"global": 42},
            "env_vars": {"TEST_VAR": "hello"},
        },
        "assertions": [
            {
                "metric_name": "exact_match",
                "operator": "gte",
                "value": 0.5,
                "description": "Exact match must be at least 0.5",
            },
        ],
    }


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for file-based tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sqlite_store() -> SQLiteStore:
    """Return an in-memory SQLiteStore, already initialised."""
    store = SQLiteStore(":memory:")
    store.init_store()
    return store


@pytest.fixture
def populated_store(sqlite_store: SQLiteStore) -> SQLiteStore:
    """Return a store with sample runs and results pre-populated."""
    store = sqlite_store

    # Run A
    store.save_run(
        run_id="run_a",
        spec_hash="abc123",
        model_name="gpt-4o-mini",
        dataset_name="test_data",
        provenance_json=json.dumps({"source": "test"}),
        status="completed",
    )
    for i in range(5):
        store.save_result(
            run_id="run_a",
            item_index=i,
            input_json=json.dumps({"prompt": f"input_{i}"}),
            expected=f"expected_{i}",
            prediction=f"prediction_{i}",
            metrics_json=json.dumps({"accuracy": 0.8 + i * 0.05, "f1": 0.7 + i * 0.06}),
            duration_ms=100.0,
        )

    # Run B
    store.save_run(
        run_id="run_b",
        spec_hash="def456",
        model_name="gpt-4o",
        dataset_name="test_data",
        provenance_json=json.dumps({"source": "test"}),
        status="completed",
    )
    for i in range(5):
        store.save_result(
            run_id="run_b",
            item_index=i,
            input_json=json.dumps({"prompt": f"input_{i}"}),
            expected=f"expected_{i}",
            prediction=f"prediction_B_{i}",
            metrics_json=json.dumps({"accuracy": 0.85 + i * 0.05, "f1": 0.75 + i * 0.06}),
            duration_ms=90.0,
        )

    return store


@pytest.fixture
def sample_spec_yaml() -> str:
    """Return a valid YAML string for a minimal spec."""
    return """
name: minimal-test
model:
  provider: openai
  name: gpt-4o-mini
dataset:
  source: jsonl
  path: ./data.jsonl
metrics:
  - name: exact_match
trials: 1
"""


@pytest.fixture
def mock_adapter():
    """Return a mock model adapter for runner tests."""
    from unittest.mock import AsyncMock, MagicMock

    adapter = MagicMock()
    adapter.predict = AsyncMock(
        return_value=[{"text": "mock prediction"}]
    )
    return adapter
