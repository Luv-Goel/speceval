"""Result store for speceval evaluation runs."""

from __future__ import annotations

from .base import ResultStore
from .sqlite import SQLiteStore

__all__ = ["ResultStore", "SQLiteStore"]
