"""Gated public export of the item set: dev only unless the frozen test split is released."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from indicsafe.data.hashing import dataset_hash
from indicsafe.data.splits import FreezeLock
from indicsafe.schema import Item


class ReleaseError(Exception):
    pass


def export_public(
    items: list[Item],
    lock: FreezeLock,
    out: Path,
    *,
    include_test: bool,
    reason: str,
    release_log: Path,
) -> dict[str, object]:
    """Write items.jsonl + MANIFEST.json. Test-split text needs the freeze, an explicit flag and a
    logged reason: until then only dev items leave the repo."""
    if include_test:
        if not lock.frozen:
            raise ReleaseError(
                "the test split is not frozen yet; run `indicsafe data freeze` first"
            )
        if not reason.strip():
            raise ReleaseError("releasing the test split needs --reason (it is logged)")
    chosen = [i for i in items if include_test or lock.split_of(i.item_id) == "dev"]
    withheld = len(items) - len(chosen)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.jsonl").open("w", encoding="utf-8") as f:
        for i in sorted(chosen, key=lambda x: x.item_id):
            row = i.model_dump(mode="json") | {"split": lock.split_of(i.item_id)}
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    review = Counter(t.review_status for i in chosen for t in i.translations)
    manifest: dict[str, object] = {
        "dataset_hash": dataset_hash(items),
        "n_items_exported": len(chosen),
        "n_items_withheld": withheld,
        "includes_test_split": include_test,
        "split_is_frozen": lock.frozen,
        "translation_review_status": dict(sorted(review.items())),
        "warning": "Translations are AI-drafted unless review_status says reviewed. "
        "The harmful requests are low-severity by design but are still sensitive text.",
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if include_test:
        entry = {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "reason": reason.strip(),
            "dataset_hash": manifest["dataset_hash"],
            "n_items": len(chosen),
        }
        release_log.parent.mkdir(parents=True, exist_ok=True)
        with release_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return manifest
