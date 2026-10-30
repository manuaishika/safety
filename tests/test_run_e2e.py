"""End-to-end generation runs with scripted in-process models (no network, no downloads)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest
from inspect_ai.log import list_eval_logs, read_eval_log
from typer.testing import CliRunner

import indicsafe.mock as mock
from conftest import ITEMS_DIR, REPO
from indicsafe.cli import app
from indicsafe.config import RunConfig
from indicsafe.runner import execute, prepare_run
from indicsafe.tasks import indic_refusal

runner = CliRunner()


def _cfg(**kw: object) -> RunConfig:
    base: dict[str, object] = {
        "name": "e2e",
        "models": ["scripted/leaky"],
        "languages": ["en", "hi", "hi-Latn"],
        "split": "all",
        "generation": {"max_tokens": 32},
    }
    return RunConfig.model_validate({**base, **kw})


def _prepare(cfg: RunConfig, tmp_path: Path):  # type: ignore[no-untyped-def]
    return prepare_run(cfg, ITEMS_DIR, tmp_path / "lock.json", REPO)


def _logs(out_dir: Path):  # type: ignore[no-untyped-def]
    return [read_eval_log(i) for i in list_eval_logs(str(out_dir / "logs"))]


def test_end_to_end_table_matches_the_scripted_policy(tmp_path: Path) -> None:
    prepared = _prepare(_cfg(models=["scripted/leaky", "scripted/strict"]), tmp_path)
    ok, out_dir = execute(prepared, tmp_path / "runs")
    assert ok
    logs = {log.eval.model: log for log in _logs(out_dir)}
    assert set(logs) == {"scripted/leaky", "scripted/strict"}
    for model, log in logs.items():
        assert log.status == "success" and len(log.samples or []) == 72
        policy = model.split("/")[1]
        seen: Counter[tuple[str, str, bool]] = Counter()
        for s in log.samples or []:
            md = s.metadata
            expected_refusal = mock.scripted_refuses(
                policy, s.input if isinstance(s.input, str) else ""
            )
            refused = s.output.completion in mock.REFUSALS.values()
            assert refused == expected_refusal
            assert md["source_text"] and md["english_reference"]
            seen[(md["language"], md["expected_behaviour"], refused)] += 1
        if policy == "strict":
            assert all(refused == (exp == "refuse") for (_, exp, refused) in seen)
        else:  # leaky: English is perfect, other languages leak
            assert seen[("en", "refuse", False)] == 0 and seen[("en", "refuse", True)] == 12
            leaks = seen[("hi", "refuse", False)] + seen[("hi-Latn", "refuse", False)]
            assert leaks > 0
    meta = json.loads((out_dir / "run_meta.json").read_text())
    assert meta["dataset_hash"] == prepared.meta.dataset_hash and meta["finished_at"]
    assert meta["config"]["providers"] == ["scripted"]
    assert logs["scripted/leaky"].eval.metadata["dataset_hash"] == meta["dataset_hash"]


def test_system_prompt_condition_is_applied(tmp_path: Path) -> None:
    cfg = _cfg(languages=["en"], conditions=["none", "india_persona"])
    ok, out_dir = execute(_prepare(cfg, tmp_path), tmp_path / "runs")
    assert ok
    roles = Counter()
    for s in _logs(out_dir)[0].samples or []:
        first = s.messages[0]
        roles[(s.metadata["condition"], first.role)] += 1
    assert roles[("none", "user")] == 24 and roles[("india_persona", "system")] == 24


def test_task_function_builds_dataset() -> None:
    t = indic_refusal(languages="en,hi", split="all", items_dir=str(ITEMS_DIR))
    assert len(t.dataset) == 48
    t = indic_refusal(languages="en", categories="stalking", split="all", items_dir=str(ITEMS_DIR))
    assert len(t.dataset) == 2


def test_resume_after_failure_redoes_only_unfinished_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(languages=["en", "hi"], generation={"max_tokens": 32, "max_connections": 1})
    calls = tmp_path / "calls.log"
    monkeypatch.setenv("INDICSAFE_SCRIPTED_LOG", str(calls))
    monkeypatch.setattr(mock, "_calls", 0)
    monkeypatch.setenv("INDICSAFE_SCRIPTED_FAIL_AFTER", "15")
    ok, out_dir = execute(_prepare(cfg, tmp_path), tmp_path / "runs")
    assert not ok
    assert len(calls.read_text().split()) == 15

    monkeypatch.delenv("INDICSAFE_SCRIPTED_FAIL_AFTER")
    monkeypatch.setattr(mock, "_calls", 0)
    ok, out_dir2 = execute(_prepare(cfg, tmp_path), tmp_path / "runs")
    assert ok and out_dir2 == out_dir  # same config + dataset => same run folder
    lines = calls.read_text().split()
    assert len(lines) == 48 and len(set(lines)) == 48  # nothing repeated
    assert len(_logs(out_dir)[-1].samples or []) == 48


def test_resume_after_hard_kill(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "name: killme\nmodels: [scripted/leaky]\nlanguages: [en, hi]\nsplit: all\n"
        "generation: {max_tokens: 32, max_connections: 1}\n"
    )
    calls = tmp_path / "calls.log"
    env = {
        **__import__("os").environ,
        "INDICSAFE_SCRIPTED_LOG": str(calls),
        "INDICSAFE_SCRIPTED_DELAY": "0.1",
    }
    cmd = [
        sys.executable, "-m", "indicsafe.cli", "run", "--config", str(cfg),
        "--out", str(tmp_path / "runs"), "--bench-file", str(tmp_path / "b.json"),
        "--lock", str(tmp_path / "lock.json"), "--items", str(ITEMS_DIR),
    ]  # fmt: skip
    proc = subprocess.Popen(
        cmd, env=env, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        if calls.exists() and len(calls.read_text().split()) >= 12:
            break
        time.sleep(0.05)
    proc.kill()
    proc.wait()
    done_before = len(calls.read_text().split())
    assert 12 <= done_before < 48

    env.pop("INDICSAFE_SCRIPTED_DELAY")
    r = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    lines = calls.read_text().split()
    assert len(set(lines)) == 48
    assert len(lines) - 48 <= 2  # at most the in-flight sample(s) at kill time are repeated
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    final = [x for x in _logs(run_dirs[0]) if x.status == "success"]
    assert final and len(final[-1].samples or []) == 48


def test_cli_dry_run_and_errors(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("name: d\nmodels: [scripted/strict]\nlanguages: [en]\nsplit: all\n")
    base = ["--config", str(cfg), "--bench-file", str(tmp_path / "none.json"),
            "--lock", str(tmp_path / "lock.json"), "--items", str(ITEMS_DIR)]  # fmt: skip
    r = runner.invoke(app, ["run", *base, "--dry-run"])
    assert r.exit_code == 0 and "24 generations" in r.output and "no model calls" in r.output
    assert not (tmp_path / "runs").exists()
    r = runner.invoke(app, ["run", *base, "--dry-run", "--max-hours", "1"])
    assert r.exit_code == 1  # budget planning needs bench data
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0 and "Model sizes" in r.output
