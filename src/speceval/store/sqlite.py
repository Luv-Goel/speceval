"""SQLite-backed result store."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from speceval.exceptions import StoreError
from speceval.store.base import ResultStore


class SQLiteStore(ResultStore):
    """SQLite-backed :class:`ResultStore` implementation.

    Uses a single database file with two tables:

    * ``runs`` — run-level metadata
    * ``results`` — per-item evaluation results

    WAL journal mode is enabled for concurrent read access.

    Thread-safe via a reentrant lock.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = Path(db_path) if db_path != ":memory:" else ":memory:"
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_store(self) -> None:
        """Open connection and create tables if they don't exist."""
        with self._lock:
            try:
                self._conn = sqlite3.connect(
                    str(self._db_path) if self._db_path != ":memory:" else ":memory:",
                    check_same_thread=False,
                )
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA foreign_keys=ON;")
                self._create_tables()
            except sqlite3.Error as exc:
                raise StoreError(f"Failed to initialise SQLite store: {exc}") from exc

    def save_result(
        self,
        *,
        run_id: str,
        item_index: int,
        input_json: str,
        expected: str,
        prediction: str,
        metrics_json: str,
        duration_ms: float,
    ) -> None:
        with self._lock:
            self._assert_open()
            try:
                self._conn.execute(
                    """INSERT INTO results
                       (run_id, item_index, input_json, expected, prediction,
                        metrics_json, duration_ms)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, item_index, input_json, expected, prediction,
                     metrics_json, duration_ms),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                raise StoreError(f"Failed to save result: {exc}") from exc

    def get_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            self._assert_open()
            try:
                cursor = self._conn.execute(
                    """SELECT run_id, item_index, input_json, expected, prediction,
                              metrics_json, duration_ms
                       FROM results
                       WHERE run_id = ?
                       ORDER BY item_index""",
                    (run_id,),
                )
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.Error as exc:
                raise StoreError(f"Failed to get results: {exc}") from exc

    def get_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            self._assert_open()
            try:
                cursor = self._conn.execute(
                    """SELECT id, spec_hash, model_name, dataset_name, timestamp,
                              provenance_json, status
                       FROM runs
                       ORDER BY timestamp DESC""",
                )
                return [dict(row) for row in cursor.fetchall()]
            except sqlite3.Error as exc:
                raise StoreError(f"Failed to get runs: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                finally:
                    self._conn = None

    # ------------------------------------------------------------------
    # Run metadata helpers
    # ------------------------------------------------------------------

    def save_run(
        self,
        *,
        run_id: str,
        spec_hash: str,
        model_name: str,
        dataset_name: str,
        provenance_json: str,
        status: str = "completed",
    ) -> None:
        """Insert or update a run record.

        This is a convenience method not strictly required by the abstract
        ``ResultStore`` interface but used by the CLI and engine.
        """
        with self._lock:
            self._assert_open()
            try:
                self._conn.execute(
                    """INSERT OR REPLACE INTO runs
                       (id, spec_hash, model_name, dataset_name, timestamp,
                        provenance_json, status)
                       VALUES (?, ?, ?, ?, datetime('now'), ?, ?)""",
                    (run_id, spec_hash, model_name, dataset_name,
                     provenance_json, status),
                )
                self._conn.commit()
            except sqlite3.Error as exc:
                raise StoreError(f"Failed to save run: {exc}") from exc

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return a single run record, or *None* if not found."""
        with self._lock:
            self._assert_open()
            try:
                cursor = self._conn.execute(
                    "SELECT * FROM runs WHERE id = ?", (run_id,)
                )
                row = cursor.fetchone()
                return dict(row) if row else None
            except sqlite3.Error as exc:
                raise StoreError(f"Failed to get run: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_open(self) -> None:
        if self._conn is None:
            raise StoreError("Store is not initialised — call init_store() first")

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id              TEXT PRIMARY KEY,
                spec_hash       TEXT NOT NULL,
                model_name      TEXT NOT NULL,
                dataset_name    TEXT NOT NULL,
                timestamp       TEXT NOT NULL DEFAULT (datetime('now')),
                provenance_json TEXT NOT NULL DEFAULT '{}',
                status          TEXT NOT NULL DEFAULT 'completed'
            );

            CREATE TABLE IF NOT EXISTS results (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id       TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                item_index   INTEGER NOT NULL,
                input_json   TEXT NOT NULL,
                expected     TEXT NOT NULL,
                prediction   TEXT NOT NULL,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                duration_ms  REAL NOT NULL DEFAULT 0.0
            );

            CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
            CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON runs(timestamp);
            """
        )


__all__ = ["SQLiteStore"]
