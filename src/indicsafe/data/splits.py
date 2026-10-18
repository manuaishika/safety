"""dev/test assignment and the frozen-test lock.

Assignment is a pure function of item_id (stable under adding items). It is *not* exactly
stratified by category: exact stratification would reassign existing items whenever a
category grows, which a frozen test set cannot tolerate. `stats` reports the realised
per-category split so imbalance is visible. Once frozen, membership comes from the lock.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from indicsafe.data.hashing import item_hash, sha256_hex
from indicsafe.schema import Item, Split

SALT = "indicsafe-split-v1"
TEST_FRACTION = 0.3


def hash_split(item_id: str, test_fraction: float = TEST_FRACTION) -> Split:
    u = int(sha256_hex(f"{SALT}:{item_id}")[:15], 16) / 16**15
    return "test" if u < test_fraction else "dev"


class FreezeLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] | None = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        )

    @property
    def frozen(self) -> bool:
        return self.data is not None

    def split_of(self, item_id: str) -> Split:
        if self.data is None:
            return hash_split(item_id)
        return "test" if item_id in self.data["items"] else "dev"

    def freeze(self, items: list[Item], reason: str) -> None:
        if self.frozen:
            raise RuntimeError("already frozen; use unfreeze with a reason to re-baseline")
        test = {i.item_id: item_hash(i) for i in items if hash_split(i.item_id) == "test"}
        self._write(test, [], "freeze", reason)

    def unfreeze(self, items: list[Item], reason: str) -> None:
        if self.data is None:
            raise RuntimeError("not frozen")
        current = {i.item_id: item_hash(i) for i in items}
        kept = {k: current.get(k, "") for k in self.data["items"]}
        self._write(kept, self.data["log"], "unfreeze", reason)

    def check(self, items: list[Item]) -> list[str]:
        if self.data is None:
            return []
        current = {i.item_id: item_hash(i) for i in items}
        problems = []
        for item_id, h in self.data["items"].items():
            if item_id not in current:
                problems.append(f"frozen test item {item_id!r} was removed")
            elif current[item_id] != h:
                problems.append(f"frozen test item {item_id!r} was modified (use unfreeze)")
        return problems

    def _write(
        self, test_items: dict[str, str], log: list[dict[str, str]], action: str, reason: str
    ) -> None:
        if not reason.strip():
            raise ValueError("a non-empty reason is required")
        at = datetime.now(UTC).isoformat(timespec="seconds")
        entry = {"action": action, "reason": reason.strip(), "at": at}
        self.data = {"items": test_items, "log": [*log, entry]}
        self.path.write_text(
            json.dumps(self.data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
