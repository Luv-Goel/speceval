"""Core Pydantic models for evaluation specifications."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel as _BaseModel
from pydantic import Field, model_validator


class BaseModel(_BaseModel):
    """Base model with strict extra-field handling for spec integrity."""

    model_config = {"extra": "forbid", "populate_by_name": True, "validate_default": True}


# ---------------------------------------------------------------------------
# Provider literals
# ---------------------------------------------------------------------------

KnownProvider = Literal["openai", "huggingface", "anthropic", "vllm", "custom"]


# ---------------------------------------------------------------------------
# MetricConfig
# ---------------------------------------------------------------------------


class MetricConfig(BaseModel):
    """A single metric to compute during evaluation.

    Attributes:
        name: Metric identifier (e.g. ``"exact_match"``, ``"bleu"``, ``"toxicity"``).
        params: Optional keyword arguments forwarded to the metric implementation.
        threshold: Optional pass/fail threshold; if set the result is compared
            against this value to determine success.
    """

    name: str
    params: dict[str, Any] | None = Field(default_factory=dict)
    threshold: float | None = None


# ---------------------------------------------------------------------------
# AssertionConfig
# ---------------------------------------------------------------------------


class AssertionConfig(BaseModel):
    """An assertion that must hold for a metric result.

    Attributes:
        metric_name: The metric this assertion applies to.
        operator: Comparison operator.
        value: The expected value to compare against.
        description: Human-readable description of the assertion.
    """

    metric_name: str
    operator: Literal["gte", "lte", "eq", "approx"]
    value: float | int | str | bool
    description: str | None = None


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------


class DatasetConfig(BaseModel):
    """Description of an evaluation dataset.

    Attributes:
        source: Where the dataset comes from — one of
            ``"huggingface"``, ``"csv"``, ``"jsonl"``, or ``"dict"``.
        path: Path or Hugging Face dataset identifier.
        split: Dataset split (e.g. ``"train"``, ``"test"``). Ignored for
            non-Hugging Face sources.
        subset: Optional subset / configuration name for Hugging Face datasets.
        limit: Optional maximum number of rows to load.
        preprocess: Optional reference to a preprocessing script or
            dotted Python path (``module.function``).
    """

    source: Literal["huggingface", "csv", "jsonl", "dict"]
    path: str | list[dict[str, Any]] | None = None
    split: str | None = None
    subset: str | None = None
    limit: int | None = None
    preprocess: str | None = None

    @model_validator(mode="after")
    def _check_limit_positive(self) -> DatasetConfig:
        if self.limit is not None and self.limit < 1:
            raise ValueError("dataset.limit must be a positive integer when set")
        return self


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for the model under evaluation.

    Attributes:
        provider: Supported providers: ``openai``, ``huggingface``,
            ``anthropic``, ``vllm``, or ``custom``.
        name: Model name / identifier (e.g. ``"gpt-4o"``, ``"meta-llama/Llama-2-7b"``).
        params: Optional generation parameters (temperature, max_tokens, …).
        endpoint: Optional custom endpoint URL (required for ``custom`` provider).
        api_key: Optional API key; may be a literal value or an env-var
            reference like ``$MY_API_KEY``.
    """

    provider: KnownProvider
    name: str
    params: dict[str, Any] | None = Field(default_factory=dict)
    endpoint: str | None = None
    api_key: str | None = None

    @model_validator(mode="after")
    def _check_endpoint_for_custom(self) -> ModelConfig:
        if self.provider == "custom" and not self.endpoint:
            raise ValueError("model.endpoint is required when provider='custom'")
        return self


# ---------------------------------------------------------------------------
# EnvConfig
# ---------------------------------------------------------------------------


class EnvConfig(BaseModel):
    """Environment / execution configuration.

    Attributes:
        seeds: Per-dataset or global random seeds.
        dependencies: List of pip-style dependency strings.
        docker: Optional Docker image or compose file reference.
        env_vars: Extra environment variables to set during evaluation.
    """

    seeds: dict[str, int] | None = Field(default_factory=dict)
    dependencies: list[str] | None = Field(default_factory=list)
    docker: str | None = None
    env_vars: dict[str, str] | None = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_seed_keys(self) -> EnvConfig:
        seeds = self.seeds
        if seeds is not None:
            seen: set[str] = set()
            for key in seeds:
                lower = key.lower()
                if lower in seen:
                    raise ValueError(f"duplicate seed key (case-insensitive): {key!r}")
                seen.add(lower)
        return self


# ---------------------------------------------------------------------------
# SpecConfig  —  top-level specification
# ---------------------------------------------------------------------------


class SpecConfig(BaseModel):
    """Complete evaluation specification.

    This is the root model deserialised from a ``.spec.yaml`` file.

    Attributes:
        name: Human-readable name for this evaluation.
        description: Optional longer description.
        model: The model configuration under test.
        dataset: The dataset to evaluate against.
        metrics: One or more metrics to compute.
        trials: Number of repeated evaluation trials (default 1).
        seeds: Optional global seeds dictionary (use **EnvConfig** for
            per-dataset seeds).
        env: Optional environment configuration.
        assertions: Optional list of assertions that must pass.
    """

    name: str
    description: str | None = None
    model: ModelConfig
    dataset: DatasetConfig
    metrics: list[MetricConfig] = Field(min_length=1)
    trials: int = Field(default=1, ge=1)
    seeds: dict[str, int] | None = Field(default_factory=dict)
    env: EnvConfig | None = None
    assertions: list[AssertionConfig] | None = None

    @model_validator(mode="after")
    def _check_seeds_duplicates(self) -> SpecConfig:
        seeds = self.seeds
        if seeds is not None:
            seen: set[str] = set()
            for key in seeds:
                lower = key.lower()
                if lower in seen:
                    raise ValueError(
                        f"duplicate seed key (case-insensitive): {key!r}"
                    )
                seen.add(lower)
        return self

    @model_validator(mode="after")
    def _check_assertion_metric_names(self) -> SpecConfig:
        assertions = self.assertions
        if assertions is None:
            return self
        metric_names = {m.name for m in self.metrics}
        for a in assertions:
            if a.metric_name not in metric_names:
                raise ValueError(
                    f"assertion references unknown metric {a.metric_name!r}; "
                    f"available metrics: {sorted(metric_names)}"
                )
        return self
