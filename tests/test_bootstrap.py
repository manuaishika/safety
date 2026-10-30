from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import REPO
from indicsafe.cli import app

SCRIPT = REPO / "scripts" / "bootstrap_free_gpu.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _env(**kw: str) -> dict[str, str]:
    return {**os.environ, **kw}


def test_models_command_lists_only_ollama_tags(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "name: m\nmodels: [ollama/qwen2.5:1.5b, scripted/strict, ollama/llama3.2:3b]\n"
        "languages: [en]\n"
    )
    r = CliRunner().invoke(app, ["models", "--config", str(cfg)])
    assert r.exit_code == 0
    assert r.stdout.split() == ["qwen2.5:1.5b", "llama3.2:3b"]


def test_script_syntax_and_dry_run_changes_nothing(tmp_path: Path) -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    out = tmp_path / "art"
    r = subprocess.run(
        ["bash", str(SCRIPT)],
        env=_env(DRY_RUN="1", OUT=str(out), MAX_HOURS="3"),
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    assert "ollama.com/install.sh" in r.stdout and "--max-hours 3" in r.stdout
    assert "ollama pull" in r.stdout
    assert not out.exists()  # nothing was created


def test_script_runs_end_to_end_and_is_rerunnable(tmp_path: Path) -> None:
    out = tmp_path / "art"
    env = _env(
        SKIP_SYNC="1",
        SKIP_OLLAMA="1",
        CONFIG="configs/smoke_scripted.yaml",
        OUT=str(out),
        INDICSAFE_SCRIPTED_LOG=str(tmp_path / "calls.log"),
    )
    for _ in range(2):  # second pass models a restarted session: must not redo any work
        r = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, cwd=REPO)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    names = zipfile.ZipFile(out / "indicsafe_artifacts.zip").namelist()
    assert "bench.json" in names
    assert any(n.endswith("run_meta.json") for n in names)
    assert sum(n.endswith(".eval") for n in names) == 2  # one log per model
    calls = (tmp_path / "calls.log").read_text().split()
    # bench makes a few calls; the run itself must not repeat any generation on re-run
    assert len(calls) < 2 * 336 + 40


def test_notebook_is_valid_and_points_at_the_script() -> None:
    nb = json.loads((REPO / "notebooks" / "indicsafe_free_gpu.ipynb").read_text())
    assert nb["nbformat"] == 4 and nb["cells"]
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "scripts/bootstrap_free_gpu.sh" in code and "github.com/manuaishika/safety" in code
    assert SCRIPT.exists() and os.access(SCRIPT, os.X_OK)
