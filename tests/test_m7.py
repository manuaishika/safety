from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import indicsafe.repro as repro
from conftest import ITEMS_DIR, REPO
from indicsafe.cli import app
from indicsafe.data.loaders import load_items
from indicsafe.data.release import ReleaseError, export_public
from indicsafe.data.splits import FreezeLock
from indicsafe.licenses import dataset_license, lookup, report_rows
from indicsafe.repro import (
    check_reproducible,
    diff_digests,
    digest_tree,
    verify_report,
)
from test_analysis_metrics import _grade, _write

runner = CliRunner()


# --------------------------------------------------------------------- repro helpers


def test_digest_and_diff(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    d1 = digest_tree(tmp_path)
    (tmp_path / "b.txt").write_text("3")
    (tmp_path / "c.txt").write_text("4")
    d2 = digest_tree(tmp_path)
    assert diff_digests(d1, d1) == []
    assert diff_digests(d1, d2) == ["only in second run: c.txt", "differs: b.txt"]
    assert diff_digests(d2, d1) == ["only in first run: c.txt", "differs: b.txt"]


def _tiny_grades(tmp_path: Path) -> Path:
    grades = []
    for i in range(12):
        for lang in ("en", "hi"):
            ok = lang == "en" or i % 2 == 0
            grades.append(
                _grade(
                    f"{lang}{i}", item_id=f"i{i}", language=lang, label="REFUSE" if ok else "COMPLY"
                )  # fmt: skip
            )
    return _write(tmp_path, "g", grades)


def test_check_reproducible_passes_on_the_real_pipeline(tmp_path: Path) -> None:
    n, problems = check_reproducible(
        [_tiny_grades(tmp_path)], include_unreviewed=True, n_boot=30, bayes=False
    )
    assert problems == [] and n > 20


def test_reproducibility_does_not_depend_on_which_commit_runs_the_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a commit between two runs once changed analysis_manifest.json."""
    import indicsafe.analysis.pipeline as pipeline

    states = iter([("a" * 40, True), ("b" * 40, False)])
    monkeypatch.setattr(pipeline, "git_state", lambda _repo: next(states))
    n, problems = check_reproducible(
        [_tiny_grades(tmp_path)], include_unreviewed=True, n_boot=20, bayes=False
    )
    assert problems == [] and n > 20


def test_check_reproducible_catches_nondeterminism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grades = _tiny_grades(tmp_path)
    real = repro.run_analysis
    calls = {"n": 0}

    def flaky(g: list[Path], out: Path, **kw: object):  # type: ignore[no-untyped-def]
        res = real(g, out, **kw)  # type: ignore[arg-type]
        calls["n"] += 1
        (out / "tables" / "rates_extra.csv").write_text(f"run {calls['n']}\n")
        return res

    monkeypatch.setattr(repro, "run_analysis", flaky)
    _, problems = check_reproducible([grades], include_unreviewed=True, n_boot=20, bayes=False)
    assert problems == ["differs: tables/rates_extra.csv"]


def test_verify_report_detects_edits_missing_and_extra_files(tmp_path: Path) -> None:
    rep = tmp_path / "rep"
    rep.mkdir()
    (rep / "report.md").write_text("hello")
    (rep / "t.csv").write_text("a,b")
    (rep / "report_manifest.json").write_text(json.dumps(digest_tree(rep) | {}))
    # the manifest above was computed before it existed, so it lists exactly the two files
    assert verify_report(rep) == []
    (rep / "report.md").write_text("tampered")
    (rep / "t.csv").unlink()
    (rep / "new.csv").write_text("x")
    assert sorted(verify_report(rep)) == [
        "missing: t.csv",
        "modified: report.md",
        "not in manifest: new.csv",
    ]
    assert verify_report(tmp_path / "nope") == [
        f"{tmp_path / 'nope' / 'report_manifest.json'} not found"
    ]


# --------------------------------------------------------------------- licences


def test_license_registry_flags_custom_and_noncommercial_licences() -> None:
    assert lookup("scripted/leaky") is not None and lookup("scripted/leaky").flag is None  # type: ignore[union-attr]
    for model in ("ollama/llama3.2:3b", "ollama/gemma2:2b", "ollama/aya-expanse:8b"):
        info = lookup(model)
        assert info is not None and info.flag, model
    assert "Apache" in lookup("ollama/qwen2.5:1.5b").license  # type: ignore[union-attr]
    assert lookup("ollama/totally-unknown-model") is None  # "ollama" is not the Llama family
    assert lookup("ollama/mystery:1b") is None and lookup("hf/some-org/other-model") is None
    assert lookup("hf/meta-llama/Llama-3.2-3B-Instruct").family == "Llama 3.x"  # type: ignore[union-attr]
    assert lookup("ollama/llama3.2:3b").family == "Llama 3.x"  # type: ignore[union-attr]


def test_dataset_licence_is_flagged_until_a_licence_file_exists(tmp_path: Path) -> None:
    assert dataset_license(tmp_path).flag and "none chosen" in dataset_license(tmp_path).license
    (tmp_path / "LICENSE").write_text("MIT")
    assert dataset_license(tmp_path).flag is None
    assert report_rows(["scripted/x"], tmp_path)[-1][0].startswith("dataset")


def test_licenses_command(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "name: l\nmodels: [ollama/llama3.2:3b, ollama/mystery:1b, scripted/strict]\n"
        "languages: [en]\n"
    )
    r = runner.invoke(app, ["licenses", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    assert "from memory, not fetched" in r.output
    assert "! ollama/llama3.2:3b" in r.output and "? ollama/mystery:1b" in r.output
    assert "  scripted/strict" in r.output and "dataset: data/items" in r.output


# --------------------------------------------------------------------- gated public export


def test_export_public_withholds_the_test_split_by_default(tmp_path: Path) -> None:
    items = load_items(ITEMS_DIR)
    lock = FreezeLock(tmp_path / "lock.json")
    m = export_public(items, lock, tmp_path / "pub", include_test=False, reason="",
                      release_log=tmp_path / "log.jsonl")  # fmt: skip
    rows = [json.loads(x) for x in (tmp_path / "pub" / "items.jsonl").read_text().splitlines()]
    assert rows and all(r["split"] == "dev" for r in rows)
    assert m["n_items_exported"] == len(rows) and m["n_items_withheld"] == 24 - len(rows) > 0
    assert m["includes_test_split"] is False and not (tmp_path / "log.jsonl").exists()
    assert "review_status" in json.dumps(rows[0]) and "warning" in m


def test_releasing_the_test_split_needs_a_freeze_a_flag_and_a_logged_reason(tmp_path: Path) -> None:
    items = load_items(ITEMS_DIR)
    lock = FreezeLock(tmp_path / "lock.json")
    log = tmp_path / "log.jsonl"
    with pytest.raises(ReleaseError, match="not frozen"):
        export_public(
            items, lock, tmp_path / "p", include_test=True, reason="paper", release_log=log
        )
    lock.freeze(items, "initial freeze")
    with pytest.raises(ReleaseError, match="--reason"):
        export_public(items, lock, tmp_path / "p", include_test=True, reason="  ", release_log=log)
    assert not log.exists()
    m = export_public(
        items, lock, tmp_path / "p", include_test=True, reason="camera-ready", release_log=log
    )
    assert m["n_items_exported"] == 24 and m["includes_test_split"] is True
    entry = json.loads(log.read_text().splitlines()[0])
    assert entry["reason"] == "camera-ready" and entry["n_items"] == 24 and entry["dataset_hash"]


def test_export_public_cli(tmp_path: Path) -> None:
    base = ["data", "export-public", "--out", str(tmp_path / "o"), "--items", str(ITEMS_DIR),
            "--lock", str(tmp_path / "lock.json"),
            "--release-log", str(tmp_path / "log.jsonl")]  # fmt: skip
    r = runner.invoke(app, base)
    assert r.exit_code == 0 and "withheld" in r.output
    r = runner.invoke(app, [*base, "--include-test", "--reason", "x"])
    assert r.exit_code == 1 and "not frozen" in r.output


# --------------------------------------------------------------------- hygiene


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_no_logs_results_keys_or_built_data_are_tracked() -> None:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    files = out.stdout.splitlines()
    banned = re.compile(r"^(results|logs)/|^data/built/|\.eval$|\.sqlite$|(^|/)\.env$|__pycache__")
    assert [f for f in files if banned.search(f)] == []
    secret = re.compile(
        r"sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}"
    )
    text_suffixes = {".py", ".yaml", ".yml", ".md", ".json", ".sh", ".toml", ".ipynb"}
    leaks = [
        f
        for f in files
        if (REPO / f).is_file()
        and (REPO / f).suffix in text_suffixes
        and secret.search((REPO / f).read_text(encoding="utf-8", errors="ignore"))
    ]
    assert leaks == []


# ------------------------------------------------------------- the whole pipeline via the CLI


def test_full_pipeline_through_the_cli(tmp_path: Path) -> None:
    def ok(args: list[str]) -> str:
        r = runner.invoke(app, args)
        assert r.exit_code == 0, f"{args[0]} failed:\n{r.output[-1500:]}"
        return r.output

    runs = tmp_path / "runs"
    ok(["run", "--config", str(REPO / "configs/smoke_scripted.yaml"), "--out", str(runs),
        "--bench-file", str(tmp_path / "b.json"), "--lock", str(tmp_path / "lock.json"),
        "--items", str(ITEMS_DIR)])  # fmt: skip
    (run_dir,) = list(runs.iterdir())
    grader_cfg = str(REPO / "configs/graders/scripted.yaml")
    ok(["grade", "--logs", str(run_dir), "--grader-config", grader_cfg,
        "--cache", str(tmp_path / "c.sqlite")])  # fmt: skip
    gdir = run_dir / "grades" / "1.0.0" / "scripted"
    an, rep = tmp_path / "an", tmp_path / "rep"
    ok(["analyze", "--grades", str(gdir), "--out", str(an), "--include-unreviewed",
        "--n-boot", "30", "--no-bayes"])  # fmt: skip
    ok(["report", "--analysis", str(an), "--out", str(rep)])
    assert "report matches its manifest" in ok(["verify-report", "--report", str(rep)])
    out = ok(
        [
            "repro-check",
            "--grades",
            str(gdir),
            "--include-unreviewed",
            "--n-boot",
            "30",
            "--no-bayes",
        ]
    )
    assert "byte-identical" in out
    md = (rep / "report.md").read_text()
    assert "SYNTHETIC DATA" in md and "UNCLEAR share" in md  # flawed second grader -> disagreements
    (rep / "report.md").write_text(md + "\nedited by hand")
    r = runner.invoke(app, ["verify-report", "--report", str(rep)])
    assert r.exit_code == 1 and "modified: report.md" in r.output
