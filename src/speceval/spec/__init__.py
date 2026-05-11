"""Specification models — parse, validate, hash, and resolve evaluation specs."""

from speceval.spec.hash import hash_spec, hash_spec_from_path
from speceval.spec.model import (
    AssertionConfig,
    DatasetConfig,
    EnvConfig,
    MetricConfig,
    ModelConfig,
    SpecConfig,
)
from speceval.spec.parse import parse_spec, parse_spec_string, spec_to_yaml
from speceval.spec.resolve import resolve_spec
from speceval.spec.validate import validate_spec, validate_spec_strict

__all__ = [
    # Models
    "SpecConfig",
    "ModelConfig",
    "DatasetConfig",
    "MetricConfig",
    "AssertionConfig",
    "EnvConfig",
    # Parse
    "parse_spec",
    "parse_spec_string",
    "spec_to_yaml",
    # Validate
    "validate_spec",
    "validate_spec_strict",
    # Hash
    "hash_spec",
    "hash_spec_from_path",
    # Resolve
    "resolve_spec",
]
