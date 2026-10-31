"""Ingest filled annotation sheets (one or more annotators)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from indicsafe.validation.models import HUMAN_LABELS, Annotation, ExportKey

_TRUE = {"yes", "y", "true", "1"}
_FALSE = {"no", "n", "false", "0"}


class ImportError_(Exception):
    pass


def read_sheet(
    path: Path, key: ExportKey, annotator: str | None = None, allow_incomplete: bool = False
) -> list[Annotation]:
    valid_ids = {r.row_id for r in key.rows}
    errors: list[str] = []
    out: list[Annotation] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"row_id", "human_label"} - set(reader.fieldnames or [])
        if missing:
            raise ImportError_(f"{path.name}: missing column(s) {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            rid = (row.get("row_id") or "").strip()
            if rid not in valid_ids:
                errors.append(f"{path.name}:{line}: unknown row_id {rid!r}")
                continue
            label = (row.get("human_label") or "").strip().upper()
            who = annotator or (row.get("annotator_id") or "").strip()
            if not label:
                if not allow_incomplete:
                    errors.append(f"{path.name}:{line}: {rid} has no human_label")
                continue
            if label not in HUMAN_LABELS:
                errors.append(f"{path.name}:{line}: {rid} label {label!r} not in {HUMAN_LABELS}")
                continue
            if not who:
                errors.append(
                    f"{path.name}:{line}: {rid} has no annotator_id (or pass --annotator)"
                )
                continue
            raw = (row.get("human_harmful_content_provided") or "").strip().lower()
            if raw in _TRUE:
                harm: bool | None = True
            elif raw in _FALSE:
                harm = False
            elif raw == "":
                harm = None
            else:
                errors.append(f"{path.name}:{line}: {rid} harmful flag {raw!r} not yes/no")
                continue
            if (rid, who) in seen:
                errors.append(f"{path.name}:{line}: duplicate row {rid} for annotator {who}")
                continue
            seen.add((rid, who))
            out.append(
                Annotation(
                    row_id=rid, annotator_id=who, label=label,
                    harmful_content_provided=harm, notes=(row.get("notes") or "").strip(),
                )
            )  # fmt: skip
    if errors:
        raise ImportError_(
            "\n".join(errors[:25]) + (f"\n... {len(errors) - 25} more" if len(errors) > 25 else "")
        )
    if not out:
        raise ImportError_(f"{path.name}: no labelled rows")
    return out


def load_annotations(directory: Path) -> list[Annotation]:
    p = directory / "annotations.jsonl"
    if not p.exists():
        return []
    return [
        Annotation.model_validate_json(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line
    ]


def merge_and_save(
    directory: Path, new: list[Annotation], overwrite: bool = False
) -> tuple[list[Annotation], int, int]:
    """Merge into annotations.jsonl; returns (all, added, changed).

    Re-importing identical labels is a no-op. A different label for the same (row, annotator)
    is an error unless ``overwrite``.
    """
    cur = {(a.row_id, a.annotator_id): a for a in load_annotations(directory)}
    added = changed = 0
    for a in new:
        k = (a.row_id, a.annotator_id)
        if k in cur and cur[k] != a:
            if not overwrite:
                raise ImportError_(
                    f"{a.row_id}/{a.annotator_id}: already labelled differently "
                    f"({cur[k].label} vs {a.label}); pass --overwrite to replace"
                )
            changed += 1
        elif k not in cur:
            added += 1
        cur[k] = a
    merged = sorted(cur.values(), key=lambda a: (a.row_id, a.annotator_id))
    with (directory / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for a in merged:
            f.write(
                json.dumps(a.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
            )
    return merged, added, changed
