"""Tiny real-model smoke test. Skipped unless Ollama is running with at least one model."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from conftest import ITEMS_DIR
from indicsafe.compute import bench_model, detect_hardware
from indicsafe.config import RunConfig
from indicsafe.runner import bench_prompts

_hw = detect_hardware()

pytestmark = [
    pytest.mark.local_model,
    pytest.mark.skipif(not _hw.ollama_models, reason="Ollama not running / no models pulled"),
]


def test_bench_first_local_model(tmp_path: Path) -> None:
    model = f"ollama/{_hw.ollama_models[0]}"
    cfg = RunConfig.model_validate(
        {"name": "local", "models": [model], "languages": ["en", "hi"], "split": "all"}
    )
    prompts = bench_prompts(cfg, ITEMS_DIR, tmp_path / "lock.json", 2)
    r = asyncio.run(bench_model(model, prompts, 32))
    assert r.seconds_per_sample > 0 and r.tokens_per_sec > 0
