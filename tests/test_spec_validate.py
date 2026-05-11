"""Tests for spec validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from speceval.exceptions import SpecValidationError
from speceval.spec.model import (
    AssertionConfig,
    DatasetConfig,
    EnvConfig,
    MetricConfig,
    ModelConfig,
    SpecConfig,
)
from speceval.spec.validate import validate_spec, validate_spec_strict


def _make_valid_spec(**overrides) -> SpecConfig:
    """Create a minimally valid spec, then apply overrides via __setattr__."""
    spec = SpecConfig(
        name="test",
        model=ModelConfig(provider="openai", name="gpt-4"),
        dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
        metrics=[MetricConfig(name="exact_match")],
    )
    for k, v in overrides.items():
        object.__setattr__(spec, k, v)
    return spec


class TestValidateSpec:
    """Tests for validate_spec (returns warning list)."""

    def test_valid_spec_returns_empty(self, sample_spec):
        """A fully valid spec returns an empty warning list."""
        warnings = validate_spec(sample_spec)
        assert warnings == []

    def test_minimal_valid_spec(self):
        """Minimal valid spec returns no warnings."""
        spec = SpecConfig(
            name="minimal",
            model=ModelConfig(provider="openai", name="gpt-4"),
            dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
            metrics=[MetricConfig(name="exact_match")],
        )
        warnings = validate_spec(spec)
        assert warnings == []

    def test_unknown_provider(self):
        """Unknown provider is caught by Pydantic at construction time."""
        with pytest.raises(ValidationError, match="provider"):
            ModelConfig(provider="unknown_provider", name="m")

    def test_custom_provider_no_endpoint(self):
        """Custom provider without endpoint is caught by Pydantic."""
        with pytest.raises(ValidationError, match="endpoint"):
            ModelConfig(provider="custom", name="m")

    def test_custom_provider_with_endpoint_ok(self):
        """Custom provider with endpoint is valid."""
        spec = SpecConfig(
            name="test",
            model=ModelConfig(provider="custom", name="m", endpoint="http://localhost:8000"),
            dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
            metrics=[MetricConfig(name="exact_match")],
        )
        warnings = validate_spec(spec)
        provider_warnings = [w for w in warnings if "provider" in w.lower()]
        assert not provider_warnings

    def test_file_source_missing_path(self):
        """File-based source without path generates warning."""
        spec = _make_valid_spec(dataset=DatasetConfig(source="csv", path=""))
        warnings = validate_spec(spec)
        assert any("dataset.path is required" in w for w in warnings)

    def test_huggingface_source_missing_path(self):
        """Huggingface source without path generates warning."""
        spec = _make_valid_spec(dataset=DatasetConfig(source="huggingface", path=""))
        warnings = validate_spec(spec)
        assert any("dataset.path is required" in w for w in warnings)

    def test_dataset_limit_negative(self):
        """Negative dataset limit is caught by Pydantic."""
        with pytest.raises(ValidationError, match="positive"):
            DatasetConfig(source="jsonl", path="./data.jsonl", limit=-1)

    def test_dataset_limit_zero(self):
        """Zero dataset limit is caught by Pydantic."""
        with pytest.raises(ValidationError, match="positive"):
            DatasetConfig(source="jsonl", path="./data.jsonl", limit=0)

    def test_no_metrics(self):
        """Empty metrics list generates warning."""
        spec = _make_valid_spec()
        # Bypass Pydantic (SpecConfig requires min_length=1)
        object.__setattr__(spec, "metrics", [])
        warnings = validate_spec(spec)
        assert any("At least one metric" in w for w in warnings)

    def test_empty_metric_name(self):
        """Empty metric name generates warning."""
        spec = _make_valid_spec(metrics=[MetricConfig(name="")])
        warnings = validate_spec(spec)
        assert any("metrics[0].name is empty" in w for w in warnings)

    def test_trials_zero(self):
        """Trials < 1 generates warning."""
        spec = _make_valid_spec()
        # Bypass Pydantic (SpecConfig requires ge=1)
        object.__setattr__(spec, "trials", 0)
        warnings = validate_spec(spec)
        assert any("trials" in w for w in warnings)

    def test_duplicate_seed_keys(self):
        """Case-insensitive duplicate seeds are caught by Pydantic."""
        with pytest.raises(ValidationError, match="duplicate"):
            SpecConfig(
                name="test",
                model=ModelConfig(provider="openai", name="gpt-4"),
                dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
                metrics=[MetricConfig(name="exact_match")],
                seeds={"seed1": 42, "SEED1": 43},
            )

    def test_duplicate_env_seed_keys(self):
        """Case-insensitive duplicate env.seeds are caught by Pydantic."""
        with pytest.raises(ValidationError, match="duplicate"):
            SpecConfig(
                name="test",
                model=ModelConfig(provider="openai", name="gpt-4"),
                dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
                metrics=[MetricConfig(name="exact_match")],
                env=EnvConfig(seeds={"A": 1, "a": 2}),
            )

    def test_assertion_unknown_metric(self):
        """Assertion referencing unknown metric is caught by Pydantic."""
        with pytest.raises(ValidationError, match="unknown metric"):
            SpecConfig(
                name="test",
                model=ModelConfig(provider="openai", name="gpt-4"),
                dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
                metrics=[MetricConfig(name="exact_match")],
                assertions=[
                    AssertionConfig(
                        metric_name="nonexistent_metric",
                        operator="gte",
                        value=0.5,
                    ),
                ],
            )

    def test_all_known_providers(self):
        """All known providers are accepted without warning."""
        for provider in ["openai", "huggingface", "anthropic", "vllm", "custom"]:
            endpoint = "http://localhost:8000" if provider == "custom" else None
            spec = SpecConfig(
                name="test",
                model=ModelConfig(provider=provider, name="m", endpoint=endpoint),
                dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
                metrics=[MetricConfig(name="exact_match")],
            )
            warnings = validate_spec(spec)
            prov_warnings = [w for w in warnings if "provider" in w]
            assert not prov_warnings, f"Provider {provider} generated warnings: {prov_warnings}"


class TestValidateSpecStrict:
    """Tests for validate_spec_strict (raises on issues)."""

    def test_valid_spec_no_error(self, sample_spec):
        """A valid spec does not raise."""
        validate_spec_strict(sample_spec)  # should not raise

    def test_invalid_spec_raises(self):
        """Invalid spec raises SpecValidationError."""
        spec = _make_valid_spec()
        object.__setattr__(spec, "dataset", DatasetConfig(source="csv", path=""))
        with pytest.raises(SpecValidationError, match="validation failed"):
            validate_spec_strict(spec)

    def test_error_message_includes_warnings(self):
        """Error message contains all warning details."""
        spec = _make_valid_spec()
        object.__setattr__(spec, "dataset", DatasetConfig(source="csv", path=""))
        object.__setattr__(spec, "metrics", [MetricConfig(name="")])
        with pytest.raises(SpecValidationError) as excinfo:
            validate_spec_strict(spec)
        msg = str(excinfo.value)
        assert "dataset.path is required" in msg
        assert "metrics[0].name is empty" in msg
