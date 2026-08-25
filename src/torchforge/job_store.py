"""SQLite-backed storage for pipeline jobs.

Jobs survive backend restarts instead of living only in process memory.
The database lives under the configured project root so that test and
production roots stay isolated from each other.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ACTIVE_STATUSES = frozenset({"queued", "running", "cancelling"})
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
MAX_LOG_ENTRIES = 200
RETENTION_DAYS = 7

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    stages TEXT NOT NULL,
    stage TEXT,
    status TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    logs TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    duration_ms INTEGER,
    options TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_created_at_idx ON jobs (created_at);
"""

_COLUMN_FOR_FIELD = {
    "stage": "stage",
    "status": "status",
    "progress": "progress",
    "error": "error",
    "createdAt": "created_at",
    "startedAt": "started_at",
    "finishedAt": "finished_at",
    "durationMs": "duration_ms",
}

_JOB_FIELDS = (
    "id",
    "paperId",
    "stages",
    "stage",
    "status",
    "progress",
    "logs",
    "error",
    "createdAt",
    "startedAt",
    "finishedAt",
    "durationMs",
    "options",
)


_ROW_COLUMNS = (
    "id",
    "paper_id",
    "stages",
    "stage",
    "status",
    "progress",
    "logs",
    "error",
    "created_at",
    "started_at",
    "finished_at",
    "duration_ms",
    "options",
)

_FIELD_FOR_COLUMN = {
    "id": "id",
    "paper_id": "paperId",
    "stages": "stages",
    "stage": "stage",
    "status": "status",
    "progress": "progress",
    "logs": "logs",
    "error": "error",
    "created_at": "createdAt",
    "started_at": "startedAt",
    "finished_at": "finishedAt",
    "duration_ms": "durationMs",
    "options": "options",
}

_JSON_FIELDS = frozenset({"stages", "logs", "options"})


class JobStore:
    """Thread-safe SQLite persistence for job records."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self._purge_expired()

    def _purge_expired(self) -> None:
        cutoff = (datetime.now(UTC) - timedelta(days=RETENTION_DAYS)).isoformat()
        with self._lock:
            self._connection.execute(
                "DELETE FROM jobs WHERE status IN (?, ?, ?) AND created_at < ?",
                (*TERMINAL_STATUSES, cutoff),
            )
            self._connection.commit()

    def create(self, record: dict[str, Any]) -> dict[str, Any]:
        row = {
            "id": record["id"],
            "paper_id": record["paperId"],
            "stages": json.dumps(record["stages"]),
            "stage": record["stage"],
            "status": record["status"],
            "progress": record["progress"],
            "logs": json.dumps(record["logs"]),
            "error": record["error"],
            "created_at": record["createdAt"],
            "started_at": record["startedAt"],
            "finished_at": record["finishedAt"],
            "duration_ms": record["durationMs"],
            "options": json.dumps(record["options"]),
        }
        with self._lock:
            self._connection.execute(
                (
                    "INSERT INTO jobs "
                    "(id, paper_id, stages, stage, status, progress, logs, error, "
                    "created_at, started_at, finished_at, duration_ms, options) "
                    "VALUES (:id, :paper_id, :stages, :stage, :status, :progress, "
                    ":logs, :error, :created_at, :started_at, :finished_at, "
                    ":duration_ms, :options)"
                ),
                row,
            )
            self._connection.commit()
        return {field: record[field] for field in _JOB_FIELDS}

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._row_to_job(row) if row is not None else None

    def list(self, paper_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if paper_id is None:
                rows = self._connection.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT * FROM jobs WHERE paper_id = ? ORDER BY created_at DESC",
                    (paper_id,),
                ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def update(self, job_id: str, **values: Any) -> bool:
        assignments = []
        parameters: list[Any] = []
        for field, value in values.items():
            column = _COLUMN_FOR_FIELD.get(field)
            if column is None:
                raise ValueError(f"Unsupported job field for update: {field}")
            assignments.append(f"{column} = ?")
            parameters.append(value)
        if not assignments:
            return True
        parameters.append(job_id)
        with self._lock:
            cursor = self._connection.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?",
                parameters,
            )
            self._connection.commit()
        return cursor.rowcount > 0

    def append_log(self, job_id: str, message: str, timestamp: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT logs FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return False
            entries = json.loads(row[0])
            entries.append({"time": timestamp, "message": message})
            entries = entries[-MAX_LOG_ENTRIES:]
            self._connection.execute(
                "UPDATE jobs SET logs = ? WHERE id = ?",
                (json.dumps(entries), job_id),
            )
            self._connection.commit()
        return True

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> dict[str, Any]:
        job: dict[str, Any] = {}
        for column, value in zip(_ROW_COLUMNS, row):
            field = _FIELD_FOR_COLUMN[column]
            job[field] = json.loads(value) if field in _JSON_FIELDS else value
        return job
