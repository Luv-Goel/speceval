"""Parse YAML specification files into ``SpecConfig`` instances."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from speceval.exceptions import SpecParseError
from speceval.spec.model import SpecConfig


def _load_yaml(content: str) -> Dict[str, Any]:
    """Parse *content* as YAML and return a raw dict."""
    try:
        data: Dict[str, Any] = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise SpecParseError(f"YAML parse error: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecParseError("YAML root is not a mapping (expected a spec dictionary)")
    return data


def _validate_has_name(data: Dict[str, Any]) -> None:
    """Early check that a ``name`` key exists for better error messages."""
    if "name" not in data or not data["name"]:
        raise SpecParseError("spec must contain a non-empty 'name' field")


def parse_spec(path: Path) -> SpecConfig:
    """Load a ``.spec.yaml`` file and parse it into a ``SpecConfig``.

    Args:
        path: Filesystem path to the YAML specification file.

    Returns:
        A validated ``SpecConfig`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        SpecParseError: If the YAML is malformed or validation fails.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Spec file not found: {path}")

    content = path.read_text(encoding="utf-8")
    return parse_spec_string(content)


def parse_spec_string(content: str) -> SpecConfig:
    """Parse a YAML string directly into a ``SpecConfig``.

    Args:
        content: Raw YAML string.

    Returns:
        A validated ``SpecConfig`` instance.

    Raises:
        SpecParseError: If the YAML is malformed or validation fails.
    """
    data = _load_yaml(content)
    _validate_has_name(data)

    try:
        spec = SpecConfig.model_validate(data)
    except Exception as exc:
        raise SpecParseError(str(exc)) from exc

    return spec


def spec_to_yaml(spec: SpecConfig) -> str:
    """Serialize a ``SpecConfig`` back into a canonical YAML string.

    Args:
        spec: The specification to serialize.

    Returns:
        YAML-formatted string.

    Raises:
        SpecParseError: If serialization fails unexpectedly.
    """
    try:
        raw: Dict[str, Any] = spec.model_dump(
            mode="python",
            exclude_unset=False,
            exclude_defaults=False,
        )
        # Strip out None values for a cleaner output (keep empty containers).
        cleaned: Dict[str, Any] = _strip_none(raw)
        return yaml.dump(
            cleaned,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        ).strip() + "\n"
    except Exception as exc:
        raise SpecParseError(f"Failed to serialize spec to YAML: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_none(d: Any) -> Any:
    """Recursively remove keys with ``None`` values from dicts."""
    if isinstance(d, dict):
        return {k: _strip_none(v) for k, v in d.items() if v is not None}
    if isinstance(d, list):
        return [_strip_none(item) for item in d]
    return d
