from __future__ import annotations

import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from conftest import ITEMS_DIR
from indicsafe.cli import app

runner = CliRunner()


def test_validate_build_stats(tmp_path: Path) -> None:
    lock = str(tmp_path / "lock.json")
    r = runner.invoke(app, ["data", "validate", "--items", str(ITEMS_DIR), "--lock", lock])
    assert r.exit_code == 0 and "24 items valid" in r.output
    out = tmp_path / "built"
    r = runner.invoke(
        app, ["data", "build", "--items", str(ITEMS_DIR), "--out", str(out), "--lock", lock]
    )
    assert r.exit_code == 0, r.output
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["n_samples"] == 72
    assert len((out / "samples.jsonl").read_text().splitlines()) == 72
    r = runner.invoke(app, ["data", "stats", "--items", str(ITEMS_DIR), "--lock", lock])
    assert json.loads(r.output)["items"] == 24


def test_validate_fails_on_broken_pairing(tmp_path: Path) -> None:
    items = tmp_path / "items"
    shutil.copytree(ITEMS_DIR, items)
    (items / "fraud" / "spot_fake_reviews.yaml").unlink()
    r = runner.invoke(
        app, ["data", "validate", "--items", str(items), "--lock", str(tmp_path / "l.json")]
    )
    assert r.exit_code == 1


def test_freeze_roundtrip(tmp_path: Path) -> None:
    items = tmp_path / "items"
    shutil.copytree(ITEMS_DIR, items)
    lock = str(tmp_path / "lock.json")
    base = ["--items", str(items), "--lock", lock]
    assert runner.invoke(app, ["data", "freeze", "--reason", "t", *base]).exit_code == 0
    assert runner.invoke(app, ["data", "freeze", "--reason", "t", *base]).exit_code == 1
    assert runner.invoke(app, ["data", "validate", *base]).exit_code == 0
    frozen = json.loads(Path(lock).read_text())["items"]
    victim = next(iter(frozen))
    f = next(items.rglob(f"{victim}.yaml"))
    f.write_text(f.read_text().replace("notes:", "notes: changed;", 1))
    assert runner.invoke(app, ["data", "validate", *base]).exit_code == 1
    assert runner.invoke(app, ["data", "unfreeze", "--reason", "ok", *base]).exit_code == 0
    assert runner.invoke(app, ["data", "validate", *base]).exit_code == 0
