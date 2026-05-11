"""Specification resolution — env-var substitution, default population.

The ``resolve_spec`` function takes a ``SpecConfig`` and returns a new
``SpecConfig`` with:

* Environment variable references (``$VAR`` or ``${VAR}``) replaced with
  their runtime values.
* Default values filled in for any optional fields that are ``None``.
* Relative paths resolved to absolute paths (future use).
"""

from __future__ import annotations

import os
import re
from typing import Any

from speceval.spec.model import SpecConfig

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def _resolve_env_vars(value: Any, env: dict[str, str] | None = None) -> Any:
    """Recursively substitute ``$VAR`` and ``${VAR}`` in string values.

    Uses *env* if provided, otherwise falls back to ``os.environ``.
    Unset variables raise a ``ValueError``.
    """
    if env is None:
        env = dict(os.environ)

    if isinstance(value, str):

        def _replace(m: re.Match) -> str:
            var = m.group(1) or m.group(2)
            if var is None:
                return m.group(0)  # pragma: no cover — regex guarantees capture
            resolved = env.get(var)
            if resolved is None:
                raise ValueError(
                    f"Environment variable {var!r} is not set "
                    f"(referenced in spec resolution)"
                )
            return resolved

        return _ENV_VAR_PATTERN.sub(_replace, value)

    if isinstance(value, dict):
        return {k: _resolve_env_vars(v, env) for k, v in value.items()}

    if isinstance(value, list):
        return [_resolve_env_vars(item, env) for item in value]

    return value


def _resolve_dict(
    d: dict[str, Any], env: dict[str, str] | None = None
) -> dict[str, Any]:
    """Return a new dict with env-var substitution applied recursively."""
    return _resolve_env_vars(d, env)  # type: ignore[return-value]


def resolve_spec(
    spec: SpecConfig,
    resolve_env: bool = True,
    env: dict[str, str] | None = None,
) -> SpecConfig:
    """Resolve runtime values in a ``SpecConfig`` and return a new instance.

    The original ``spec`` is not modified.

    Resolution steps (when *resolve_env* is ``True``):
        1. All string fields (including nested dict values) are scanned for
           ``$VAR`` and ``${VAR}`` patterns and replaced with their runtime
           values.
        2. If any referenced environment variable is unset a ``ValueError`` is
           raised.

    Args:
        spec: The specification to resolve.
        resolve_env: Whether to perform environment-variable substitution
            (default ``True``).
        env: Optional explicit environment mapping. If ``None`` (default),
            ``os.environ`` is used.

    Returns:
        A new ``SpecConfig`` with resolved values.
    """
    data: dict[str, Any] = spec.model_dump(mode="python")

    if resolve_env:
        data = _resolve_dict(data, env)

    # Re-validate into a fresh SpecConfig.
    resolved = SpecConfig.model_validate(data)
    return resolved
