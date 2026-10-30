from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

import indicsafe.mock as mock
from conftest import ITEMS_DIR
from indicsafe.compute import (
    PlanError,
    _units,
    detect_hardware,
    load_bench,
    plan_subset,
    recommend,
    save_bench,
)
from indicsafe.config import RunConfig, run_id
from indicsafe.data.builder import build_samples
from indicsafe.data.loaders import load_items
from indicsafe.data.splits import FreezeLock
from indicsafe.variants import REGISTRY

MODELS = ["scripted/strict", "scripted/leaky"]
BASE = {"name": "t", "models": MODELS, "languages": ["en", "hi"]}


def test_config_rejects_bad_values() -> None:
    for bad in (
        {"languages": ["xx"]},
        {"variants": ["nope"]},
        {"conditions": ["nope"]},
        {"epochs": 3},  # temperature 0
        {"models": ["openai/gpt-4o"]},  # hosted without allow_hosted
        {"surprise": 1},
    ):
        with pytest.raises(ValidationError):
            RunConfig.model_validate({**BASE, **bad})
    ok = RunConfig.model_validate({**BASE, "epochs": 3, "generation": {"temperature": 0.7}})
    assert ok.epochs == 3
    RunConfig.model_validate({**BASE, "models": ["openai/x"], "allow_hosted": True})


def test_run_id_deterministic_and_sensitive() -> None:
    a = RunConfig.model_validate(BASE)
    assert run_id(a, "h1") == run_id(RunConfig.model_validate(BASE), "h1")
    assert run_id(a, "h1") != run_id(a, "h2")
    assert run_id(a, "h1") != run_id(a.model_copy(update={"seed": 1}), "h1")


def test_recommend_tiers() -> None:
    small = {r.tier: r.fits for r in recommend(8.0, None, 50)}
    assert small == {"1-2B": True, "3-4B": True, "7-9B": False, "12-14B": False}
    gpu = {r.tier: r.fits for r in recommend(16.0, 16.0, 50)}
    assert all(gpu.values())
    assert not any(r.fits for r in recommend(None, None, 50))
    assert not recommend(64.0, None, 1)[1].fits  # disk too small


def test_detect_hardware_runs_anywhere(tmp_path: Path) -> None:
    hw = detect_hardware(tmp_path)
    assert hw.disk_free_gb > 0 and hw.os


def test_bench_roundtrip_and_measurement(tmp_path: Path) -> None:
    from indicsafe.compute import bench_model

    items = load_items(ITEMS_DIR)
    samples = build_samples(items, ["en", "hi"], FreezeLock(tmp_path / "l.json"))
    mock.register_samples(samples)
    r = asyncio.run(bench_model("scripted/strict", [s.text for s in samples[:6]], 32))
    assert r.n_prompts == 6 and r.seconds_per_sample >= 0 and r.token_count_source == "estimate"
    save_bench(tmp_path / "b.json", {r.model: r})
    assert load_bench(tmp_path / "b.json")[r.model] == r
    assert load_bench(tmp_path / "missing.json") == {}


# ----------------------------------------------------------------------------- planner


def _plan(hours: float, sps: float = 100.0, min_units: int = 8):  # type: ignore[no-untyped-def]
    items = load_items(ITEMS_DIR)
    return items, plan_subset(
        items,
        MODELS,
        ["identity", "translit_noise_light"],
        ["en", "hi", "hi-Latn"],
        lambda v, lang: lang in REGISTRY[v].languages,
        {m: sps for m in MODELS},
        hours,
        min_units=min_units,
    )


def test_units_keep_pairs_together_and_are_stratified() -> None:
    items = load_items(ITEMS_DIR)
    units = _units(items)
    by_id = {i.item_id: i for i in items}
    assert sorted(i for u in units for i in u) == sorted(by_id)
    assert all(len(u) == 2 and by_id[u[0]].category == by_id[u[1]].category for u in units)
    first = [by_id[u[0]].category for u in units[:12]]
    assert len(set(first)) == 12  # first 12 units cover all 12 categories


def test_plan_full_grid_when_budget_is_large() -> None:
    items, p = _plan(1000)
    assert not p.dropped_cells and not p.dropped_item_ids
    assert len(p.item_ids) == len(items)
    assert p.est_seconds == pytest.approx(p.full_est_seconds)


def test_plan_drops_items_before_cells_and_respects_budget() -> None:
    # full grid = 5.33 h; 8 pair-units across all 8 cells = 3.56 h, so 4 h drops items only
    items, p = _plan(4.0)
    assert 16 <= len(p.item_ids) < len(items) and not p.dropped_cells
    assert p.est_seconds <= p.budget_seconds
    assert len(p.item_ids) >= 16  # 8 pair-units
    assert set(p.item_ids).isdisjoint(p.dropped_item_ids)


def test_plan_drops_whole_groups_before_languages() -> None:
    _, p = _plan(1.5)  # fits 8 units for one (variant, model) group with all 3 languages only
    assert {(c.variant, c.model) for c in p.cells} == {("identity", MODELS[0])}
    assert [c.language for c in p.cells] == ["en", "hi", "hi-Latn"]
    assert p.dropped_cells and p.est_seconds <= p.budget_seconds


def test_plan_trims_languages_only_as_last_resort_keeping_baseline() -> None:
    _, p = _plan(1.0)
    assert [c.language for c in p.cells] == ["en", "hi"]
    assert p.est_seconds <= p.budget_seconds


def test_plan_impossible_budget_raises() -> None:
    with pytest.raises(PlanError, match="cannot cover"):
        _plan(0.0001, sps=1000.0)
    with pytest.raises(PlanError, match="no bench data"):
        plan_subset([], MODELS, ["identity"], ["en"], lambda v, lang: True, {}, 1.0)


def test_plan_is_deterministic() -> None:
    assert _plan(4.0)[1].item_ids == _plan(4.0)[1].item_ids
