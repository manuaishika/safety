"""SQLite grade cache keyed by a hash of everything the grader sees."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from indicsafe.data.hashing import canonical_json, sha256_hex
from indicsafe.grading.models import GraderOutcome


def cache_key(
    grader_model: str, rubric_version: str, system: str, prompt: str, max_tokens: int
) -> str:
    return sha256_hex(canonical_json([grader_model, rubric_version, system, prompt, max_tokens]))


class GradeCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS grades (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self._db.commit()

    def get(self, key: str) -> GraderOutcome | None:
        row = self._db.execute("SELECT value FROM grades WHERE key=?", (key,)).fetchone()
        return GraderOutcome.model_validate_json(row[0]) if row else None

    def put(self, key: str, outcome: GraderOutcome) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO grades (key, value) VALUES (?, ?)",
            (key, outcome.model_copy(update={"from_cache": False}).model_dump_json()),
        )
        self._db.commit()

    def __len__(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM grades").fetchone()[0])

    def close(self) -> None:
        self._db.close()
