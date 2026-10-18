"""Load canonical item files (one YAML file per item) from disk."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from indicsafe.schema import Item


class LoadError(Exception):
    pass


def load_items(root: Path) -> list[Item]:
    files = sorted(root.rglob("*.yaml"))
    if not files:
        raise LoadError(f"no *.yaml item files under {root}")
    items: list[Item] = []
    errors: list[str] = []
    for f in files:
        try:
            raw = yaml.safe_load(f.read_text(encoding="utf-8"))
            item = Item.model_validate(raw)
        except (yaml.YAMLError, ValidationError) as e:
            errors.append(f"{f}: {e}")
            continue
        if f.stem != item.item_id:
            errors.append(f"{f}: file name must equal item_id ({item.item_id!r})")
        items.append(item)
    if errors:
        raise LoadError("\n".join(errors))
    return items
