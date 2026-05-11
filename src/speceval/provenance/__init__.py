"""Provenance capture for reproducibility."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from .environment import capture_provenance


@dataclasses.dataclass
class ProvenanceInfo:
    """Structured provenance metadata for an evaluation run."""

    git_commit_hash: str | None = None
    """Current git commit hash (short form), or ``None`` if not a git repo."""

    python_version: str | None = None
    """Python version string (e.g. ``\"3.10.12\"``)."""

    platform: str | None = None
    """Platform string (e.g. ``\"Linux-5.15.0-x86_64-with-glibc2.35\"``)."""

    hostname: str | None = None
    """Machine hostname."""

    gpu_info: str | None = None
    """GPU description (e.g. ``\"NVIDIA A100 80GB\"``), or ``None``."""

    pip_packages: list[dict[str, str]] | None = None
    """List of ``{\"name\": ..., \"version\": ...}`` dicts for installed packages."""

    timestamp: str | None = None
    """ISO-8601 timestamp of capture."""

    additional: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Arbitrary extra provenance key/value pairs."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


__all__ = ["ProvenanceInfo", "capture_provenance"]
