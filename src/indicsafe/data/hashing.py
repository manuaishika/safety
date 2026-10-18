"""Stable content hashes and sample ids."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from indicsafe.schema import Item


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def item_hash(item: Item) -> str:
    return sha256_hex(canonical_json(item.model_dump(mode="json")))


def dataset_hash(items: Iterable[Item]) -> str:
    """Order-independent hash over all item content, translations and review state."""
    parts = sorted(item_hash(i) for i in items)
    return sha256_hex("\n".join(parts))


def sample_id(item_id: str, language: str, variant: str, condition: str) -> str:
    """Stable id of item x language x variant x condition.

    Only names enter the hash (not variant versions or text), so adding a variant or
    editing a translation never changes an existing id; the dataset hash covers content.
    """
    return sha256_hex(canonical_json([item_id, language, variant, condition]))[:16]
