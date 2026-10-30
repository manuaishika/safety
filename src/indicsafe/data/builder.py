"""Items -> samples (deterministic), plus the dataset manifest."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from indicsafe.conditions import CONDITION_PROMPTS
from indicsafe.data.hashing import canonical_json, dataset_hash, sample_id, sha256_hex
from indicsafe.data.splits import FreezeLock
from indicsafe.data.validate import validate_items
from indicsafe.languages import LANGUAGES
from indicsafe.schema import SCHEMA_VERSION, Item, Sample
from indicsafe.variants import REGISTRY, UnsupportedLanguage, make_rng, protected_spans


class BuildError(Exception):
    pass


def build_samples(
    items: list[Item],
    languages: Sequence[str],
    lock: FreezeLock,
    variants: Sequence[str] = ("identity",),
    conditions: Sequence[str] = ("none",),
    seed: int = 0,
    skip_unsupported: bool = False,
) -> list[Sample]:
    validate_items(items)
    for lang in languages:
        if lang not in LANGUAGES:
            raise BuildError(f"unknown language {lang!r}; known: {sorted(LANGUAGES)}")
    for v in variants:
        if v not in REGISTRY:
            raise BuildError(f"unknown variant {v!r}; known: {sorted(REGISTRY)}")
    for c in conditions:
        if c not in CONDITION_PROMPTS:
            raise BuildError(f"unknown condition {c!r}; known: {list(CONDITION_PROMPTS)}")

    samples: list[Sample] = []
    missing: list[str] = []
    for item in sorted(items, key=lambda i: i.item_id):
        by_lang = {t.language: t for t in item.translations}
        english = by_lang["en"].text
        for lang in languages:
            t = by_lang.get(lang)
            if t is None:
                missing.append(f"{item.item_id}: no {lang} translation")
                continue
            for v in variants:
                variant = REGISTRY[v]
                rng = make_rng(seed, item.item_id, lang, v)
                try:
                    text, info = variant.apply_with_info(t.text, lang, rng)
                except UnsupportedLanguage as e:
                    if skip_unsupported:
                        continue
                    raise BuildError(f"{item.item_id}: {e}") from e
                lost = [p for p in protected_spans(t.text) if p not in text]
                if lost or not text.strip():
                    raise BuildError(
                        f"{item.item_id}/{lang}/{v}: variant corrupted protected spans {lost}"
                        if lost
                        else f"{item.item_id}/{lang}/{v}: variant produced empty text"
                    )
                for c in conditions:
                    samples.append(
                        Sample(
                            sample_id=sample_id(item.item_id, lang, v, c),
                            item_id=item.item_id,
                            category=item.category,
                            expected_behaviour=item.expected_behaviour,
                            severity_tier=item.severity_tier,
                            india_specific=item.india_specific,
                            split=lock.split_of(item.item_id),
                            language=lang,
                            script=t.script,
                            variant=v,
                            variant_version=variant.version,
                            variant_info=info,
                            condition=c,
                            text=text,
                            source_text=t.text,
                            english_reference=english,
                            provenance=t.provenance,
                            review_status=t.review_status,
                        )
                    )
    if missing:
        raise BuildError("missing translations:\n" + "\n".join(missing))
    ids = Counter(s.sample_id for s in samples)
    dup = [k for k, n in ids.items() if n > 1]
    if dup:
        raise BuildError(f"sample_id collision: {dup}")  # pragma: no cover
    return samples


def git_state(repo: Path) -> tuple[str | None, bool | None]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, None
    return sha, dirty


def compute_stats(items: list[Item], lock: FreezeLock) -> dict[str, Any]:
    lang = Counter(t.language for i in items for t in i.translations)
    review = Counter(t.review_status for i in items for t in i.translations)
    lang_review = Counter(f"{t.language}/{t.review_status}" for i in items for t in i.translations)
    split_cat: Counter[str] = Counter(f"{lock.split_of(i.item_id)}/{i.category}" for i in items)
    return {
        "items": len(items),
        "harmful": sum(i.expected_behaviour == "refuse" for i in items),
        "benign": sum(i.expected_behaviour == "comply" for i in items),
        "india_specific": sum(i.india_specific for i in items),
        "translations": sum(lang.values()),
        "by_language": dict(sorted(lang.items())),
        "by_review_status": dict(sorted(review.items())),
        "by_language_review_status": dict(sorted(lang_review.items())),
        "by_split": dict(Counter(lock.split_of(i.item_id) for i in items)),
        "by_split_category": dict(sorted(split_cat.items())),
        "frozen": lock.frozen,
    }


def make_manifest(
    items: list[Item], samples: list[Sample], lock: FreezeLock, repo: Path, seed: int = 0
) -> dict[str, Any]:
    sha, dirty = git_state(repo)
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_hash": dataset_hash(items),
        "samples_hash": sha256_hex(canonical_json([s.model_dump(mode="json") for s in samples])),
        "n_samples": len(samples),
        "seed": seed,
        "variants": {v: REGISTRY[v].version for v in sorted({s.variant for s in samples})},
        "git_sha": sha,
        "git_dirty": dirty,
        **compute_stats(items, lock),
    }


def write_built(samples: list[Sample], manifest: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
            f.write("\n")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
