"""Canonical hashing of evaluation specifications.

Uses SHA-256 over the deterministic YAML representation produced by
``spec_to_yaml`` so that two semantically identical specs (regardless of
whitespace, key ordering, or comments in the original file) produce the same
hash.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from speceval.exceptions import SpecParseError
from speceval.spec.model import SpecConfig
from speceval.spec.parse import parse_spec, spec_to_yaml


def hash_spec(spec: SpecConfig) -> str:
    """Return the SHA-256 hex digest of the canonical YAML for *spec*.

    The canonical form is produced by :func:`~speceval.spec.parse.spec_to_yaml`,
    which serialises the ``SpecConfig`` deterministically.

    Args:
        spec: The specification to hash.

    Returns:
        A 64-character hexadecimal SHA-256 digest.
    """
    canonical = spec_to_yaml(spec)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_spec_from_path(path: Path) -> str:
    """Load a spec from *path* and return its canonical hash.

    This is a convenience function equivalent to::

        hash_spec(parse_spec(path))

    Args:
        path: Filesystem path to a ``.spec.yaml`` file.

    Returns:
        A 64-character hexadecimal SHA-256 digest.

    Raises:
        FileNotFoundError: If *path* does not exist.
        SpecParseError: If the file cannot be parsed.
    """
    spec = parse_spec(path)
    return hash_spec(spec)
