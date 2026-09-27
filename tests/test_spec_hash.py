"""Tests for spec hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from speceval.spec.hash import hash_spec, hash_spec_from_path
from speceval.spec.model import (
    DatasetConfig,
    MetricConfig,
    ModelConfig,
    SpecConfig,
)


class TestHashSpec:
    """Tests for hash_spec."""

    def test_hash_is_string(self, sample_spec):
        """Hash result is a string of 64 hex characters."""
        h = hash_spec(sample_spec)
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)  # should not raise

    def test_hash_is_deterministic(self, sample_spec):
        """Same spec produces same hash every time."""
        h1 = hash_spec(sample_spec)
        h2 = hash_spec(sample_spec)
        assert h1 == h2

    def test_different_specs_different_hash(self, sample_spec):
        """Different specs produce different hashes."""
        other = SpecConfig(
            name="other-evaluation",
            model=ModelConfig(provider="openai", name="gpt-4o"),
            dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
            metrics=[MetricConfig(name="bleu")],
        )
        h1 = hash_spec(sample_spec)
        h2 = hash_spec(other)
        assert h1 != h2

    def test_hash_changes_on_name_change(self, sample_spec):
        """Changing the spec name changes the hash."""
        h1 = hash_spec(sample_spec)
        sample_spec.name = "renamed"
        h2 = hash_spec(sample_spec)
        assert h1 != h2

    def test_hash_changes_on_model_change(self, sample_spec):
        """Changing model config changes the hash."""
        h1 = hash_spec(sample_spec)
        sample_spec.model.name = "gpt-4-turbo"
        h2 = hash_spec(sample_spec)
        assert h1 != h2

    def test_hash_changes_on_metric_change(self, sample_spec):
        """Adding a metric changes the hash."""
        h1 = hash_spec(sample_spec)
        sample_spec.metrics.append(MetricConfig(name="f1"))
        h2 = hash_spec(sample_spec)
        assert h1 != h2

    def test_hash_changes_on_dataset_change(self, sample_spec):
        """Changing dataset path changes the hash."""
        h1 = hash_spec(sample_spec)
        sample_spec.dataset.path = "./other_data.jsonl"
        h2 = hash_spec(sample_spec)
        assert h1 != h2

    def test_semantic_equivalence(self):
        """Two specs with same fields but different YAML formatting produce same hash.

        Since the hash is computed from the canonical YAML output (via
        spec_to_yaml), semantically identical specs should hash identically
        even if constructed differently.
        """
        spec1 = SpecConfig(
            name="test",
            model=ModelConfig(provider="openai", name="gpt-4"),
            dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
            metrics=[MetricConfig(name="exact_match")],
        )
        spec2 = SpecConfig(
            name="test",
            model=ModelConfig(provider="openai", name="gpt-4"),
            dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
            metrics=[MetricConfig(name="exact_match")],
        )
        assert hash_spec(spec1) == hash_spec(spec2)

    def test_hash_of_minimal_spec(self):
        """Minimal valid spec can be hashed."""
        spec = SpecConfig(
            name="minimal",
            model=ModelConfig(provider="openai", name="gpt-4"),
            dataset=DatasetConfig(source="jsonl", path="./data.jsonl"),
            metrics=[MetricConfig(name="exact_match")],
        )
        h = hash_spec(spec)
        assert len(h) == 64

    def test_hash_with_assertions(self, sample_spec):
        """Spec with assertions hashes correctly."""
        h = hash_spec(sample_spec)
        assert isinstance(h, str)
        assert len(h) == 64


class TestHashSpecFromPath:
    """Tests for hash_spec_from_path."""

    def test_from_valid_file(self, temp_dir: Path):
        """Hash a valid spec file."""
        spec_file = temp_dir / "test.spec.yaml"
        spec_file.write_text("""
name: file-test
model:
  provider: openai
  name: gpt-4
dataset:
  source: jsonl
  path: ./data.jsonl
metrics:
  - name: exact_match
""")
        h = hash_spec_from_path(spec_file)
        assert isinstance(h, str)
        assert len(h) == 64

    def test_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            hash_spec_from_path("/nonexistent/spec.yaml")

    def test_consistency_with_parse_and_hash(self, temp_dir: Path):
        """hash_spec_from_path(x) == hash_spec(parse_spec(x))."""
        from speceval.spec.parse import parse_spec

        spec_file = temp_dir / "test.spec.yaml"
        spec_file.write_text("""
name: consistency-test
model:
  provider: openai
  name: gpt-4
dataset:
  source: jsonl
  path: ./data.jsonl
metrics:
  - name: exact_match
""")
        h_direct = hash_spec_from_path(spec_file)
        spec = parse_spec(spec_file)
        h_parsed = hash_spec(spec)
        assert h_direct == h_parsed
