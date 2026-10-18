from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conftest import ITEMS_DIR, make_pair, tr
from indicsafe.data.builder import BuildError, build_samples, compute_stats, make_manifest
from indicsafe.data.hashing import dataset_hash, item_hash, sample_id
from indicsafe.data.loaders import LoadError, load_items
from indicsafe.data.splits import FreezeLock, hash_split
from indicsafe.data.validate import DatasetInvalid, check_items, validate_items
from indicsafe.schema import Item

LANGS = ["en", "hi", "hi-Latn"]


def test_real_items_validate_and_are_complete() -> None:
    items = load_items(ITEMS_DIR)
    assert check_items(items) == []
    assert len(items) == 24
    assert sum(i.expected_behaviour == "refuse" for i in items) == 12
    assert all(i.severity_tier <= 2 for i in items)
    assert all({t.language for t in i.translations} == set(LANGS) for i in items)


def test_duplicate_ids(pair: tuple[Item, Item]) -> None:
    h, b = pair
    assert any("duplicate item_id" in p for p in check_items([h, b, h]))


def test_unpaired_harmful(pair: tuple[Item, Item]) -> None:
    h, _ = pair
    assert any("paired" in p for p in check_items([h]))


def test_pairing_must_be_reciprocal_and_same_category(pair: tuple[Item, Item]) -> None:
    h, b = pair
    b2 = b.model_copy(update={"paired_item_id": None})
    assert any("not reciprocal" in p for p in check_items([h, b2]))
    b3 = b.model_copy(update={"category": "stalking"})
    assert any("different category" in p for p in check_items([h, b3]))


def test_missing_english(pair: tuple[Item, Item]) -> None:
    h, b = pair
    h2 = h.model_copy(update={"translations": [t for t in h.translations if t.language != "en"]})
    assert any("missing English" in p for p in check_items([h2, b]))


def test_script_mismatch_flagged(pair: tuple[Item, Item]) -> None:
    h, b = pair
    bad = h.model_copy(update={"translations": [*h.translations[:1], tr("hi", "plain english")]})
    assert any("native script" in p for p in check_items([bad, b]))


def test_out_of_scope_rejected(pair: tuple[Item, Item]) -> None:
    h, b = pair
    bad = h.model_copy(update={"translations": [tr("en", "Explain how to build a bomb.")]})
    assert any("denylist" in p for p in check_items([bad, b]))


def test_validate_items_raises_with_all_problems(pair: tuple[Item, Item]) -> None:
    with pytest.raises(DatasetInvalid) as e:
        validate_items([pair[0]])
    assert e.value.problems


def test_build_missing_translation_fails_loudly(pair: tuple[Item, Item], tmp_path: Path) -> None:
    lock = FreezeLock(tmp_path / "l.json")
    with pytest.raises(BuildError, match="no ta translation"):
        build_samples(list(pair), ["en", "ta"], lock)


def test_build_unknown_variant_and_condition(pair: tuple[Item, Item], tmp_path: Path) -> None:
    lock = FreezeLock(tmp_path / "l.json")
    with pytest.raises(BuildError):
        build_samples(list(pair), LANGS, lock, variants=["nope"])
    with pytest.raises(BuildError):
        build_samples(list(pair), LANGS, lock, conditions=["nope"])


def test_build_is_deterministic_and_complete(tmp_path: Path) -> None:
    items = load_items(ITEMS_DIR)
    lock = FreezeLock(tmp_path / "l.json")
    a = build_samples(items, LANGS, lock)
    b = build_samples(list(reversed(items)), LANGS, lock)
    assert a == b
    assert len(a) == 24 * 3
    assert len({s.sample_id for s in a}) == len(a)
    s = next(s for s in a if s.language == "hi")
    assert s.english_reference and s.english_reference != s.text


def test_sample_id_stable_golden() -> None:
    # Frozen on purpose: changing this value changes every downstream id.
    assert sample_id("covert_tracking", "hi", "identity", "none") == "586716e73fbfef1a"
    assert sample_id("a", "en", "identity", "none") != sample_id("a", "en", "roleplay", "none")


def test_dataset_hash_order_independent_and_sensitive(pair: tuple[Item, Item]) -> None:
    h, b = pair
    assert dataset_hash([h, b]) == dataset_hash([b, h])
    h2 = h.model_copy(update={"notes": "changed"})
    assert dataset_hash([h, b]) != dataset_hash([h2, b])
    assert item_hash(h) == item_hash(h.model_copy())


def test_hash_split_stable_and_roughly_proportional() -> None:
    assert hash_split("x") == hash_split("x")
    n_test = sum(hash_split(f"item_{i}") == "test" for i in range(2000))
    assert 0.25 < n_test / 2000 < 0.35


def test_freeze_detects_modification_and_unfreeze_logs(tmp_path: Path) -> None:
    items = [i for k in range(40) for i in make_pair(suffix=str(k))]
    lock = FreezeLock(tmp_path / "lock.json")
    lock.freeze(items, "initial freeze")
    assert lock.check(items) == []
    test_id = next(iter(lock.data["items"]))  # type: ignore[index]
    edited = [i.model_copy(update={"notes": "edit"}) if i.item_id == test_id else i for i in items]
    assert any("modified" in p for p in lock.check(edited))
    assert any("removed" in p for p in lock.check([i for i in items if i.item_id != test_id]))
    with pytest.raises(ValueError):
        lock.unfreeze(edited, "  ")
    lock.unfreeze(edited, "fix typo")
    assert lock.check(edited) == []
    assert [e["action"] for e in lock.data["log"]] == ["freeze", "unfreeze"]  # type: ignore[index]
    reloaded = FreezeLock(tmp_path / "lock.json")
    assert reloaded.split_of(test_id) == "test"
    assert reloaded.split_of("brand_new_item") == "dev"  # new items never enter a frozen test set


def test_manifest_counts(tmp_path: Path) -> None:
    items = load_items(ITEMS_DIR)
    lock = FreezeLock(tmp_path / "l.json")
    samples = build_samples(items, LANGS, lock)
    m = make_manifest(items, samples, lock, tmp_path)
    assert m["n_samples"] == 72
    assert m["by_language"] == {"en": 24, "hi": 24, "hi-Latn": 24}
    assert m["by_review_status"] == {"unreviewed": 72}
    assert m["dataset_hash"] == dataset_hash(items)
    assert compute_stats(items, lock)["harmful"] == 12


def test_loader_reports_bad_files(tmp_path: Path) -> None:
    (tmp_path / "x.yaml").write_text(yaml.dump({"item_id": "x"}))
    with pytest.raises(LoadError):
        load_items(tmp_path)
    with pytest.raises(LoadError, match="no \\*.yaml"):
        load_items(tmp_path / "empty_does_not_exist")
