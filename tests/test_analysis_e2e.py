"""Scripted run -> grade -> analyze -> report, checked against the known scripted truth."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import indicsafe.mock as mock
from conftest import ITEMS_DIR, REPO
from indicsafe.analysis.pipeline import POLICIES, PRIMARY, run_analysis
from indicsafe.analysis.tidy import AnalysisError
from indicsafe.cli import app
from indicsafe.config import RunConfig
from indicsafe.grading.cache import GradeCache
from indicsafe.grading.models import GraderConfig
from indicsafe.grading.pipeline import grade_samples, read_samples, write_outputs
from indicsafe.grading.rubric import load_rubric
from indicsafe.runner import execute, prepare_run

runner = CliRunner()
N_BOOT = 60
SHARED_BOOT = 100


@pytest.fixture(scope="module")
def graded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two scripted models x 3 languages x 2 variants, graded by an accurate grader."""
    tmp = tmp_path_factory.mktemp("an")
    cfg = RunConfig.model_validate(
        {
            "name": "ane2e",
            "models": ["scripted/leaky", "scripted/strict"],
            "languages": ["en", "hi", "hi-Latn"],
            "variants": ["identity", "roleplay"],
            "split": "all",
            "generation": {"max_tokens": 32},
        }
    )
    ok, run_dir = execute(prepare_run(cfg, ITEMS_DIR, tmp / "lock.json", REPO), tmp / "runs")
    assert ok
    samples, skipped = read_samples(run_dir)
    gcfg = GraderConfig.model_validate({"name": "acc", "graders": ["scriptedgrader/accurate"]})
    cache = GradeCache(tmp / "c.sqlite")
    grades, stats = asyncio.run(grade_samples(samples, gcfg, cache))
    cache.close()
    return write_outputs(grades, stats, gcfg, load_rubric(), samples, skipped, run_dir, REPO, None)


@pytest.fixture(scope="module")
def analysed(graded: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One full analysis (all policies, Bayesian fit included) shared by the read-only tests."""
    out = tmp_path_factory.mktemp("analysis") / "a"
    _analyse(graded, out, bayes=True, n_boot=SHARED_BOOT)
    return out


def _analyse(graded: Path, out: Path, **kw: object) -> dict[str, object]:
    args: dict[str, object] = {"include_unreviewed": True, "n_boot": N_BOOT, "bayes": False}
    args.update(kw)
    return run_analysis([graded], out, **args).manifest  # type: ignore[arg-type]


def test_tidy_table_has_the_required_columns_and_no_text(analysed: Path) -> None:
    t = pd.read_parquet(analysed / "tidy.parquet")
    required = {"model", "language", "script", "variant", "condition", "category", "item_id",
                "expected", "label", "correct", "refused", "answered_in_prompt_language",
                "grader_version", "dataset_hash", "run_id"}  # fmt: skip
    assert required <= set(t.columns)
    assert len(t) == 288 and set(t.script) == {"native", "romanized"}
    assert not any(c in t.columns for c in ("response", "request", "text"))  # labels only
    assert t.grader_version.eq("1.0.0").all() and t.dataset_hash.nunique() == 1


def test_results_match_the_scripted_truth(analysed: Path) -> None:
    tdir = analysed / "tables" / PRIMARY.name
    rates = pd.read_csv(tdir / "rates.csv")
    gaps = pd.read_csv(tdir / "gaps.csv")
    mc = pd.read_csv(tdir / "mcnemar.csv")
    strict = rates[rates.model == "scripted/strict"]
    assert (strict[strict.expected == "refuse"].rate == 1.0).all()  # strict refuses everywhere
    leaky = rates[(rates.model == "scripted/leaky") & (rates.expected == "refuse")]
    en = leaky[leaky.language == "en"]
    assert (en.rate == 1.0).all()
    assert (leaky[leaky.language != "en"].rate < 1.0).all()  # leaky leaks in hi and hi-Latn
    g = gaps[(gaps.model == "scripted/leaky") & (gaps.expected == "refuse")]
    assert (g.gap < -0.2).all() and (g.ci_high < 0).all() and (g.p_holm < 0.05).all()
    assert (gaps[gaps.model == "scripted/strict"].gap == 0).all()
    assert (gaps[gaps.expected == "comply"].gap == 0).all()  # no over-refusal anywhere
    leak = mc[(mc.model == "scripted/leaky") & (mc.expected == "refuse")]
    assert (leak.language_only == 0).all() and (leak.baseline_only > 0).all()
    assert (leak.n_pairs == 12).all() and (leak.p_holm < 0.05).all()
    # numbers agree with an independent computation from the scripted policy itself
    t = pd.read_parquet(analysed / "tidy.parquet")
    for lang in ("hi", "hi-Latn"):
        sub = t[(t.model == "scripted/leaky") & (t.language == lang) & (t.expected == "refuse")
                & (t.variant == "identity")]  # fmt: skip
        want = float(sub.correct.astype(bool).mean())
        got = leak[(leak.language == lang) & (leak.variant == "identity")]
        assert got.paired_gap.iloc[0] == pytest.approx(want - 1.0)


def test_all_policies_are_written_and_regression_status_is_explicit(
    graded: Path, tmp_path: Path
) -> None:
    _analyse(graded, tmp_path / "a", bayes=True, n_boot=50)
    for pol in POLICIES:
        for name in ("rates", "rates_by_category", "gaps", "mcnemar", "regression"):
            assert (tmp_path / "a" / "tables" / pol.name / f"{name}.csv").exists()
    reg = pd.read_csv(tmp_path / "a" / "tables" / PRIMARY.name / "regression.csv")
    assert reg.status.notna().all()  # every row says what happened
    assert reg.method.str.startswith("Bayesian").any() and reg.method.str.startswith("GEE").any()
    # English is refused 100% of the time here, so the GEE odds ratio is not estimable and
    # must be flagged rather than reported as a number
    gee = reg[reg.method.str.startswith("GEE") & reg.term.fillna("").str.contains("language")]
    assert gee.status.str.contains("separation").all() and gee.odds_ratio.isna().all()
    assert gee.p_value.isna().all()


def test_figures_exist_in_both_formats(analysed: Path) -> None:
    man = json.loads((analysed / "analysis_manifest.json").read_text())
    figs = analysed / "figures"
    names = {p.name for p in figs.iterdir()}
    assert {"rates_harmful_refusal_rate.png", "rates_harmful_refusal_rate.pdf",
            "gaps_harmful_refusal_rate.png", "gaps_harmful_refusal_rate.pdf"} <= names  # fmt: skip
    assert all((figs / n).stat().st_size > 1000 for n in names)
    assert set(man["figures"]) >= {"harmful_refusal_rate", "gaps_harmful_refusal_rate"}


def _digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}  # fmt: skip


def test_reanalysis_is_byte_identical(graded: Path, analysed: Path, tmp_path: Path) -> None:
    _analyse(graded, tmp_path / "b", bayes=True, n_boot=SHARED_BOOT)
    a, b = _digest(analysed), _digest(tmp_path / "b")
    assert a.keys() == b.keys() and len(a) > 20
    assert a == b, [k for k in a if a[k] != b[k]]


def test_seed_is_recorded_and_never_moves_point_estimates(
    graded: Path, analysed: Path, tmp_path: Path
) -> None:
    mb = _analyse(graded, tmp_path / "b", seed=2, n_boot=30)
    assert mb["seed"] == 2
    ga = pd.read_csv(analysed / "tables" / PRIMARY.name / "gaps.csv")
    gb = pd.read_csv(tmp_path / "b" / "tables" / PRIMARY.name / "gaps.csv")
    assert (ga.gap == gb.gap).all()  # point estimates never depend on the seed
    # (interval endpoints sit on a coarse 1/12 lattice with 12 items, so they may coincide;
    # seed sensitivity of the bootstrap itself is tested in test_analysis_metrics)


def test_review_filter_and_pooling_guards(graded: Path, tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="--include-unreviewed"):
        run_analysis([graded], tmp_path / "x", include_unreviewed=False, n_boot=10, bayes=False)
    with pytest.raises(AnalysisError, match="baseline"):
        run_analysis(
            [graded], tmp_path / "y", include_unreviewed=True, baseline="ta", n_boot=10, bayes=False
        )
    other = tmp_path / "other"
    other.mkdir()
    rows = [json.loads(x) for x in (graded / "grades.jsonl").read_text().splitlines()]
    for r in rows:
        r["rubric_version"] = "9.9.9"
    (other / "grades.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(AnalysisError, match="refusing to pool"):
        run_analysis(
            [graded, other], tmp_path / "z", include_unreviewed=True, n_boot=10, bayes=False
        )


def test_cli_analyze_and_report_roundtrip(graded: Path, tmp_path: Path) -> None:
    out, rep = tmp_path / "an", tmp_path / "report"
    base = ["analyze", "--grades", str(graded), "--out", str(out), "--n-boot", "30", "--no-bayes"]
    r = runner.invoke(app, base)
    assert r.exit_code == 1 and "--include-unreviewed" in r.output
    r = runner.invoke(app, [*base, "--include-unreviewed"])
    assert r.exit_code == 0, r.output
    assert runner.invoke(app, ["report", "--analysis", str(tmp_path / "none")]).exit_code == 1
    r = runner.invoke(app, ["report", "--analysis", str(out), "--out", str(rep)])
    assert r.exit_code == 0, r.output
    md = (rep / "report.md").read_text()
    needles = [
        "SYNTHETIC DATA", "Unreviewed translations are included", "McNemar",
        "Limits of these numbers", "No human validation supplied", "min detectable gap",
    ]  # fmt: skip
    for needle in needles:
        assert needle in md
    assert "7.000" not in md and "0.000 |" not in md  # counts are integers; p-values are bounded
    manifest = json.loads((rep / "report_manifest.json").read_text())
    assert "report.md" in manifest and "figures/gaps_harmful_refusal_rate.png" in manifest
    assert not any("response" in c for c in pd.read_csv(rep / "tidy.csv").columns)


def test_report_is_reproducible(analysed: Path, tmp_path: Path) -> None:
    for name in ("r1", "r2"):
        r = runner.invoke(
            app, ["report", "--analysis", str(analysed), "--out", str(tmp_path / name)]
        )
        assert r.exit_code == 0, r.output
    assert _digest(tmp_path / "r1") == _digest(tmp_path / "r2")


def test_report_includes_validation_numbers_when_supplied(
    graded: Path, analysed: Path, tmp_path: Path
) -> None:
    from indicsafe.grading.models import FinalGrade
    from indicsafe.validation.export import build_export, write_export
    from indicsafe.validation.models import Annotation
    from indicsafe.validation.report import build_report, render_markdown

    run_dir = graded.parent.parent.parent
    samples, _ = read_samples(run_dir)
    texts = {(s.subject_model, s.sample_id, s.epoch): s for s in samples}
    rows = [
        FinalGrade.model_validate_json(x)
        for x in (graded / "grades.jsonl").read_text().splitlines()
    ]
    key, sheet = build_export(rows, texts, n=60, seed=0, include_test=True, grades_dir=str(graded),
                              grades_sha256="x", rubric_version="1.0.0",
                              grader_config="acc")  # fmt: skip
    anns = [Annotation(row_id=s["row_id"], annotator_id="A",
                       label="REFUSE" if s["response"] in mock.REFUSALS.values() else "COMPLY")
            for s in sheet]  # fmt: skip
    vrep = build_report(key, anns, rows, n_boot=40)
    vdir = tmp_path / "val"
    write_export(vdir, key, sheet)
    (vdir / "report.json").write_text(json.dumps(vrep))
    (vdir / "report.md").write_text(render_markdown(vrep))
    args = ["report", "--analysis", str(analysed), "--out", str(tmp_path / "rep")]
    r = runner.invoke(app, [*args, "--validation", str(vdir)])
    assert r.exit_code == 0, r.output
    md = (tmp_path / "rep" / "report.md").read_text()
    assert (
        "Grader error measured against human labels" in md
        and "No human validation supplied" not in md
    )
    assert (tmp_path / "rep" / "validation_report.json").exists()
