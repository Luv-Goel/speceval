"""Validation utilities for evaluation specifications.

Provides two entry points:

* ``validate_spec`` — returns a list of warning strings (empty on success).
* ``validate_spec_strict`` — raises ``SpecValidationError`` on any issue.
"""

from __future__ import annotations

from speceval.exceptions import SpecValidationError
from speceval.spec.model import SpecConfig

# ---------------------------------------------------------------------------
# Known providers
# ---------------------------------------------------------------------------

KNOWN_PROVIDERS = {"openai", "huggingface", "anthropic", "vllm", "custom"}

# Sources that require a file-system path
FILE_SOURCES = {"csv", "jsonl"}


def validate_spec(spec: SpecConfig) -> list[str]:
    """Validate *spec* and return a list of warning / error messages.

    An empty list means the specification is valid.

    Checks performed:
        - Model provider is a known value.
        - Dataset source has required fields present.
        - At least one metric is defined.
        - Seed keys are unique (case-insensitive) — already enforced by the
          model, double-checked here for runtime warnings.
        - ``dataset.limit`` is a positive integer when set.
        - ``trials`` is a positive integer.
    """
    warnings: list[str] = []

    # -- model checks -------------------------------------------------------
    if spec.model.provider not in KNOWN_PROVIDERS:
        warnings.append(
            f"Unknown model provider {spec.model.provider!r}. "
            f"Known providers: {', '.join(sorted(KNOWN_PROVIDERS))}."
        )

    if spec.model.provider == "custom" and not spec.model.endpoint:
        warnings.append(
            "model.provider is 'custom' but no model.endpoint is set."
        )

    # -- dataset checks -----------------------------------------------------
    ds = spec.dataset

    if ds.source in FILE_SOURCES:
        if not ds.path:
            warnings.append(
                f"dataset.path is required when source is {ds.source!r}."
            )

    if ds.source == "huggingface":
        if not ds.path:
            warnings.append(
                "dataset.path is required when source is 'huggingface'."
            )

    if ds.limit is not None and ds.limit < 1:
        warnings.append(
            f"dataset.limit must be a positive integer, got {ds.limit}."
        )

    # -- metrics checks -----------------------------------------------------
    if not spec.metrics:
        warnings.append("At least one metric must be defined in 'metrics'.")

    for i, m in enumerate(spec.metrics):
        if not m.name:
            warnings.append(f"metrics[{i}].name is empty.")

    # -- trials checks ------------------------------------------------------
    if spec.trials < 1:
        warnings.append(f"trials must be >= 1, got {spec.trials}.")

    # -- seeds checks -------------------------------------------------------
    seeds = spec.seeds or {}
    if seeds:
        seen: set[str] = set()
        for key in seeds:
            lower = key.lower()
            if lower in seen:
                warnings.append(
                    f"Duplicate seed key (case-insensitive): {key!r}."
                )
            seen.add(lower)

    if spec.env and spec.env.seeds:
        seen_env: set[str] = set()
        for key in spec.env.seeds:
            lower = key.lower()
            if lower in seen_env:
                warnings.append(
                    f"Duplicate env.seeds key (case-insensitive): {key!r}."
                )
            seen_env.add(lower)

    # -- assertions checks --------------------------------------------------
    if spec.assertions:
        metric_names = {m.name for m in spec.metrics}
        for i, a in enumerate(spec.assertions):
            if a.metric_name not in metric_names:
                warnings.append(
                    f"assertions[{i}] references unknown metric "
                    f"{a.metric_name!r}. Available: {sorted(metric_names)}."
                )

    return warnings


def validate_spec_strict(spec: SpecConfig) -> None:
    """Validate *spec* and raise ``SpecValidationError`` on any issue.

    This is a convenience wrapper around ``validate_spec`` that raises an
    exception when problems are found, rather than returning a list.

    Raises:
        SpecValidationError: If any validation issue is detected.
    """
    warnings = validate_spec(spec)
    if warnings:
        msg = "Specification validation failed:\n" + "\n".join(
            f"  - {w}" for w in warnings
        )
        raise SpecValidationError(msg)
