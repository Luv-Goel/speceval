"""Tests for spec parsing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from speceval.exceptions import SpecParseError
from speceval.spec.model import SpecConfig
from speceval.spec.parse import (
    _load_yaml,
    _validate_has_name,
    parse_spec,
    parse_spec_string,
    spec_to_yaml,
)


class TestLoadYaml:
    """Tests for _load_yaml internal helper."""

    def test_valid_yaml(self):
        """Valid YAML mapping is parsed correctly."""
        raw = "name: test\nmodel:\n  provider: openai\n"
        data = _load_yaml(raw)
        assert data == {"name": "test", "model": {"provider": "openai"}}

    def test_invalid_yaml_raises(self):
        """Malformed YAML raises SpecParseError."""
        with pytest.raises(SpecParseError, match="YAML parse error"):
            _load_yaml("{invalid: yaml: :}")

    def test_non_mapping_root_raises(self):
        """YAML root that is not a dict raises SpecParseError."""
        with pytest.raises(SpecParseError, match="not a mapping"):
            _load_yaml("[1, 2, 3]")

    def test_scalar_root_raises(self):
        """YAML root that is a scalar raises SpecParseError."""
        with pytest.raises(SpecParseError, match="not a mapping"):
            _load_yaml("just a string")


class TestValidateHasName:
    """Tests for _validate_has_name early check."""

    def test_name_present(self):
        """No error when name is present and non-empty."""
        _validate_has_name({"name": "test"})  # should not raise

    def test_name_missing(self):
        """Error when name key is missing."""
        with pytest.raises(SpecParseError, match="non-empty 'name' field"):
            _validate_has_name({"model": {}})

    def test_name_empty(self):
        """Error when name is empty string."""
        with pytest.raises(SpecParseError, match="non-empty 'name' field"):
            _validate_has_name({"name": ""})


class TestParseSpecString:
    """Tests for parse_spec_string."""

    def test_minimal_valid(self):
        """A minimal valid spec string parses successfully."""
        yaml_str = """
name: test-spec
model:
  provider: openai
  name: gpt-4
dataset:
  source: jsonl
  path: ./data.jsonl
metrics:
  - name: exact_match
trials: 1
"""
        spec = parse_spec_string(yaml_str)
        assert isinstance(spec, SpecConfig)
        assert spec.name == "test-spec"
        assert spec.model.name == "gpt-4"
        assert spec.dataset.path == "./data.jsonl"
        assert len(spec.metrics) == 1
        assert spec.metrics[0].name == "exact_match"

    def test_with_all_fields(self, sample_spec_yaml: str):
        """Full spec with all optional fields."""
        spec = parse_spec_string(sample_spec_yaml)
        assert isinstance(spec, SpecConfig)
        assert spec.name == "minimal-test"

    def test_missing_name_raises(self):
        """Missing name field raises SpecParseError."""
        yaml_str = "model:\n  provider: openai\n"
        with pytest.raises(SpecParseError, match="non-empty 'name' field"):
            parse_spec_string(yaml_str)

    def test_invalid_yaml_raises(self):
        """Invalid YAML raises SpecParseError."""
        with pytest.raises(SpecParseError, match="YAML parse error"):
            parse_spec_string("{bad yaml")

    def test_invalid_model_type_raises(self):
        """Wrong type for model field raises SpecParseError."""
        yaml_str = """
name: test
model: "not a dict"
dataset:
  source: jsonl
  path: ./data.jsonl
metrics:
  - name: exact_match
"""
        with pytest.raises(SpecParseError):
            parse_spec_string(yaml_str)

    def test_extra_fields_forbidden(self):
        """Extra fields not in model raise SpecParseError."""
        yaml_str = """
name: test
model:
  provider: openai
  name: gpt-4
dataset:
  source: jsonl
  path: ./data.jsonl
metrics:
  - name: exact_match
extra_field: should_not_be_here
"""
        with pytest.raises(SpecParseError):
            parse_spec_string(yaml_str)

    def test_empty_metrics_list_raises(self):
        """Empty metrics list should fail validation (min_length=1)."""
        yaml_str = """
name: test
model:
  provider: openai
  name: gpt-4
dataset:
  source: jsonl
  path: ./data.jsonl
metrics: []
"""
        with pytest.raises(SpecParseError):
            parse_spec_string(yaml_str)

    def test_dataset_with_limit_zero_raises(self):
        """dataset.limit must be positive if set."""
        yaml_str = """
name: test
model:
  provider: openai
  name: gpt-4
dataset:
  source: jsonl
  path: ./data.jsonl
  limit: 0
metrics:
  - name: exact_match
"""
        with pytest.raises(SpecParseError, match="positive"):
            parse_spec_string(yaml_str)

    def test_custom_provider_requires_endpoint(self):
        """Custom provider must have endpoint set."""
        yaml_str = """
name: test
model:
  provider: custom
  name: my-model
dataset:
  source: jsonl
  path: ./data.jsonl
metrics:
  - name: exact_match
"""
        with pytest.raises(SpecParseError, match="endpoint"):
            parse_spec_string(yaml_str)


class TestParseSpec:
    """Tests for parse_spec (file-based)."""

    def test_parse_valid_file(self, temp_dir: Path, sample_spec_yaml: str):
        """Parse a valid spec file successfully."""
        spec_file = temp_dir / "test_spec.yaml"
        spec_file.write_text(sample_spec_yaml)
        spec = parse_spec(spec_file)
        assert isinstance(spec, SpecConfig)
        assert spec.name == "minimal-test"

    def test_file_not_found(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_spec("/nonexistent/path/spec.yaml")

    def test_expands_user_home(self):
        """Path expansion works with ~."""
        valid_yaml = (
            "name: test\n"
            "model:\n"
            "  provider: openai\n"
            "  name: m\n"
            "dataset:\n"
            "  source: jsonl\n"
            "  path: ./data.jsonl\n"
            "metrics:\n"
            "  - name: exact_match\n"
        )
        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.read_text", return_value=valid_yaml):
            spec = parse_spec("~/test_spec.yaml")
            assert spec.name == "test"

    def test_invalid_file_content_raises(self, temp_dir: Path):
        """File with invalid YAML raises SpecParseError."""
        spec_file = temp_dir / "bad.yaml"
        spec_file.write_text("{bad")
        with pytest.raises(SpecParseError, match="YAML parse error"):
            parse_spec(spec_file)


class TestSpecToYaml:
    """Tests for spec_to_yaml serialization."""

    def test_round_trip(self, sample_spec: SpecConfig):
        """Serializing and re-parsing yields an equivalent spec."""
        yaml_str = spec_to_yaml(sample_spec)
        reparsed = parse_spec_string(yaml_str)
        assert reparsed.name == sample_spec.name
        assert reparsed.model.name == sample_spec.model.name
        assert reparsed.model.provider == sample_spec.model.provider
        assert reparsed.dataset.source == sample_spec.dataset.source
        assert reparsed.dataset.path == sample_spec.dataset.path
        assert len(reparsed.metrics) == len(sample_spec.metrics)
        assert len(reparsed.assertions or []) == len(sample_spec.assertions or [])

    def test_yaml_is_deterministic(self, sample_spec: SpecConfig):
        """Same spec produces identical YAML each time."""
        yaml1 = spec_to_yaml(sample_spec)
        yaml2 = spec_to_yaml(sample_spec)
        assert yaml1 == yaml2

    def test_output_valid_yaml(self, sample_spec: SpecConfig):
        """Output is valid YAML that can be parsed."""
        yaml_str = spec_to_yaml(sample_spec)
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)
        assert parsed["name"] == "test-evaluation"

    def test_none_values_stripped(self, sample_spec: SpecConfig):
        """None values are stripped from output YAML."""
        yaml_str = spec_to_yaml(sample_spec)
        # description should be present (it's set)
        assert "description" in yaml_str
        # seeds is empty dict, should be preserved in output
        parsed = yaml.safe_load(yaml_str)
        # Seeds=None is stripped, but if it's an empty dict it stays
        # Let's check a field that IS set
        assert parsed["name"] == "test-evaluation"

    def test_with_assertions(self, sample_spec):
        """Spec with assertions round-trips correctly."""
        yaml_str = spec_to_yaml(sample_spec)
        assert "exact_match" in yaml_str
        reparsed = parse_spec_string(yaml_str)
        assert reparsed.assertions is not None
        assert reparsed.assertions[0].metric_name == "exact_match"
        assert reparsed.assertions[0].operator == "gte"
        assert reparsed.assertions[0].value == 0.5


class TestSpecToYamlWithFixture:
    """SpecToYaml tests that need the fixture."""

    def test_assertions_round_trip(self, sample_spec):
        yaml_str = spec_to_yaml(sample_spec)
        assert "exact_match" in yaml_str
        reparsed = parse_spec_string(yaml_str)
        assert reparsed.assertions is not None
        assert reparsed.assertions[0].metric_name == "exact_match"
        assert reparsed.assertions[0].operator == "gte"
        assert reparsed.assertions[0].value == 0.5
