from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

import indicsafe.mock as mock
from conftest import ITEMS_DIR, REPO
from indicsafe.cli import app
from indicsafe.config import RunConfig
from indicsafe.data.hashing import sha256_hex
from indicsafe.grading.cache import GradeCache
from indicsafe.grading.models import FinalGrade, GraderConfig
from indicsafe.grading.pipeline import ToGrade, grade_samples, read_samples, write_outputs
from indicsafe.grading.rubric import load_rubric
from indicsafe.runner import execute, prepare_run
from indicsafe.validation.export import (
    SHEET_COLUMNS,
    allocate_equal,
    allocate_proportional,
    build_export,
    write_export,
)
from indicsafe.validation.importer import (
    ImportError_,
    load_annotations,
    merge_and_save,
    read_sheet,
)
from indicsafe.validation.models import Annotation, ExportKey
from indicsafe.validation.report import build_report, render_markdown

runner = CliRunner()
REFUSALS = set(mock.REFUSALS.values())


# ------------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def graded(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """A scripted run graded by a deliberately flawed grader (flips ~30% of hi/hi-Latn)."""
    import asyncio

    tmp = tmp_path_factory.mktemp("val")
    cfg = RunConfig.model_validate(
        {
            "name": "valtest",
            "models": ["scripted/leaky"],
            "languages": ["en", "hi", "hi-Latn"],
            "variants": ["identity", "roleplay", "code_switch_50"],
            "split": "all",
            "generation": {"max_tokens": 32},
        }
    )
    ok, run_dir = execute(prepare_run(cfg, ITEMS_DIR, tmp / "lock.json", REPO), tmp / "runs")
    assert ok
    samples, skipped = read_samples(run_dir)
    gcfg = GraderConfig.model_validate({"name": "flawed", "graders": ["scriptedgrader/flip_hi"]})
    cache = GradeCache(tmp / "c.sqlite")
    grades, stats = asyncio.run(grade_samples(samples, gcfg, cache))
    cache.close()
    out = write_outputs(grades, stats, gcfg, load_rubric(), samples, skipped, run_dir, REPO, None)
    return {"grades": grades, "samples": samples, "gdir": out, "run_dir": run_dir}


def _export(g: dict[str, object], **kw: object):  # type: ignore[no-untyped-def]
    samples: list[ToGrade] = g["samples"]  # type: ignore[assignment]
    texts = {(s.subject_model, s.sample_id, s.epoch): s for s in samples}
    args: dict[str, object] = {
        "n": 90, "seed": 1, "include_test": True, "grades_dir": "x", "grades_sha256": "abc",
        "rubric_version": "1.0.0", "grader_config": "flawed",
    }  # fmt: skip
    args.update(kw)
    return build_export(g["grades"], texts, **args)  # type: ignore[arg-type]


def _truth(response: str) -> str:
    return "REFUSE" if response in REFUSALS else "COMPLY"


def _annotate(key: ExportKey, sheet: list[dict[str, str]], who: str, flip_every: int = 0):  # type: ignore[no-untyped-def]
    anns = []
    for i, row in enumerate(sheet):
        label = _truth(row["response"])
        if flip_every and i % flip_every == 0:
            label = "COMPLY" if label == "REFUSE" else "REFUSE"
        anns.append(Annotation(row_id=row["row_id"], annotator_id=who, label=label))  # type: ignore[arg-type]
    return anns


# ------------------------------------------------------------------------------- export


def test_sheet_hides_the_grader_and_everything_that_identifies_it(
    graded: dict[str, object], tmp_path: Path
) -> None:
    key, sheet = _export(graded)
    write_export(tmp_path / "e", key, sheet)
    text = (tmp_path / "e" / "annotator_sheet.csv").read_text(encoding="utf-8-sig")
    header = text.splitlines()[0].split(",")
    assert tuple(header) == SHEET_COLUMNS
    for secret in ("flip_hi", "scriptedgrader", "leaky", "grader", "expected", "disagreement"):
        assert secret not in text
    assert (tmp_path / "e" / "key.json").exists() and "REFUSE" in (
        tmp_path / "e" / "key.json"
    ).read_text()


def test_equal_rows_per_language_and_weights_rebuild_the_population(
    graded: dict[str, object],
) -> None:
    key, _ = _export(graded, n=90)
    langs = {r.language for r in key.rows}
    assert langs == {"en", "hi", "hi-Latn"}
    for lang in langs:
        assert sum(r.language == lang for r in key.rows) == 30
    # design weights sum back to each (language, label, category) cell's population
    grades: list[FinalGrade] = graded["grades"]  # type: ignore[assignment]
    pop: dict[tuple[str, str, str], int] = {}
    for g in grades:
        pop[(g.language, g.label, g.category)] = pop.get((g.language, g.label, g.category), 0) + 1
    got: dict[tuple[str, str, str], float] = {}
    for r in key.rows:
        k = (r.language, r.grader_label, r.category)
        got[k] = got.get(k, 0.0) + r.weight
    assert got and all(got[k] == pytest.approx(pop[k]) for k in got)
    assert key.unsampled_population_share == pytest.approx(
        sum(v for k, v in pop.items() if k not in got) / len(grades)
    )
    assert key.mode == "sample" and key.n_population == 216


def test_export_is_deterministic_and_seed_changes_it(graded: dict[str, object]) -> None:
    a, sheet_a = _export(graded, seed=3)
    b, sheet_b = _export(graded, seed=3)
    c, _ = _export(graded, seed=4)
    assert a == b and sheet_a == sheet_b
    assert [r.sample_id for r in a.rows] != [r.sample_id for r in c.rows]


def test_rows_are_shuffled_not_grouped(graded: dict[str, object]) -> None:
    key, _ = _export(graded)
    langs = [r.language for r in key.rows]
    assert langs != sorted(langs)
    assert [r.row_id for r in key.rows] == [f"r{i:04d}" for i in range(1, 91)]


def test_test_split_excluded_unless_requested_and_all_mode(graded: dict[str, object]) -> None:
    key, _ = _export(graded, include_test=False, n=None)
    assert key.mode == "all" and all(r.split != "test" for r in key.rows)
    assert key.n_excluded_test > 0 and any("test-split" in w for w in key.warnings)
    assert all(r.weight == 1.0 for r in key.rows)
    full, _ = _export(graded, include_test=True, n=None)
    assert len(full.rows) == 216
    big, _ = _export(graded, include_test=True, n=10_000)
    assert big.mode == "all" and any("exporting everything" in w for w in big.warnings)


def test_empty_responses_are_never_exported(graded: dict[str, object]) -> None:
    grades: list[FinalGrade] = graded["grades"]  # type: ignore[assignment]
    g0 = grades[0].model_copy(update={"label_source": "rule:empty", "label": "UNCLEAR"})
    mutated = dict(graded, grades=[g0, *grades[1:]])
    key, _ = _export(mutated, n=None)
    assert key.n_excluded_empty == 1 and len(key.rows) == 215


def test_allocation_helpers() -> None:
    assert allocate_equal(10, {"a": 100, "b": 100}) == {"a": 5, "b": 5}
    assert allocate_equal(10, {"a": 3, "b": 100}) == {"a": 3, "b": 7}  # capped, rest spills over
    assert allocate_equal(10, {"a": 2, "b": 3}) == {"a": 2, "b": 3}
    got = allocate_proportional(10, {"x": 50, "y": 30, "z": 20})
    assert got == {"x": 5, "y": 3, "z": 2} and sum(got.values()) == 10
    with pytest.raises(ValueError):
        allocate_proportional(5, {"x": 2, "y": 2})
    for n in range(0, 21):
        t = allocate_proportional(n, {"a": 7, "b": 5, "c": 11, "d": 1})
        assert sum(t.values()) == n and all(
            0 <= t[k] <= s for k, s in {"a": 7, "b": 5, "c": 11, "d": 1}.items()
        )


# ------------------------------------------------------------------------------- import


def _write_filled(
    path: Path, sheet: list[dict[str, str]], anns: list[Annotation], harm: str = ""
) -> None:
    by = {a.row_id: a for a in anns}
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SHEET_COLUMNS))
        w.writeheader()
        for row in sheet:
            a = by.get(row["row_id"])
            w.writerow({**row, "human_label": a.label.lower() if a else "",
                        "annotator_id": a.annotator_id if a else "",
                        "human_harmful_content_provided": harm})  # fmt: skip


def test_import_roundtrip_multi_annotator_and_idempotence(
    graded: dict[str, object], tmp_path: Path
) -> None:
    key, sheet = _export(graded)
    ann_a, ann_b = _annotate(key, sheet, "A"), _annotate(key, sheet, "B", flip_every=10)
    _write_filled(tmp_path / "a.csv", sheet, ann_a, harm="Yes")
    _write_filled(tmp_path / "b.csv", sheet, ann_b)
    got = read_sheet(tmp_path / "a.csv", key) + read_sheet(tmp_path / "b.csv", key)
    assert got[0].label == ann_a[0].label  # case-insensitive labels
    assert got[0].harmful_content_provided is True and got[-1].harmful_content_provided is None
    merged, added, changed = merge_and_save(tmp_path, got)
    assert (added, changed, len(merged)) == (180, 0, 180)
    _, added2, changed2 = merge_and_save(tmp_path, got)
    assert (added2, changed2) == (0, 0)
    assert load_annotations(tmp_path) == merged
    flipped = [a.model_copy(update={"label": "UNCLEAR"}) for a in got[:1]]
    with pytest.raises(ImportError_, match="already labelled differently"):
        merge_and_save(tmp_path, flipped)
    _, _, ch = merge_and_save(tmp_path, flipped, overwrite=True)
    assert ch == 1


def test_import_rejects_bad_sheets(graded: dict[str, object], tmp_path: Path) -> None:
    key, sheet = _export(graded)
    anns = _annotate(key, sheet, "A")
    p = tmp_path / "s.csv"

    def write(rows: list[dict[str, str]]) -> None:
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(SHEET_COLUMNS))
            w.writeheader()
            w.writerows(rows)

    good = {**sheet[0], "human_label": "REFUSE", "annotator_id": "A"}
    write([{**good, "row_id": "r9999"}])
    with pytest.raises(ImportError_, match="unknown row_id"):
        read_sheet(p, key)
    write([{**good, "human_label": "MAYBE"}])
    with pytest.raises(ImportError_, match="not in"):
        read_sheet(p, key)
    write([{**good, "annotator_id": ""}])
    with pytest.raises(ImportError_, match="annotator_id"):
        read_sheet(p, key)
    assert read_sheet(p, key, annotator="Z")[0].annotator_id == "Z"
    write([{**good, "human_harmful_content_provided": "perhaps"}])
    with pytest.raises(ImportError_, match="yes/no"):
        read_sheet(p, key)
    write([good, good])
    with pytest.raises(ImportError_, match="duplicate"):
        read_sheet(p, key)
    write([{**good, "human_label": ""}])
    with pytest.raises(ImportError_, match="no human_label"):
        read_sheet(p, key)
    with pytest.raises(ImportError_, match="no labelled rows"):
        read_sheet(p, key, allow_incomplete=True)
    p.write_text("row_id,notes\nr0001,x\n")
    with pytest.raises(ImportError_, match="missing column"):
        read_sheet(p, key)
    assert anns  # fixture sanity


# ------------------------------------------------------------------------------- report


def test_report_finds_the_planted_differential_error(graded: dict[str, object]) -> None:
    key, sheet = _export(graded, n=150)
    rep = build_report(key, _annotate(key, sheet, "A"), graded["grades"], n_boot=300)  # type: ignore[arg-type]
    pl = rep["per_language"]
    assert pl["en"]["accuracy_weighted"] == 1.0 and pl["en"]["cohen_kappa"] == 1.0
    for lang in ("hi", "hi-Latn"):
        assert 0.5 < pl[lang]["accuracy_weighted"] < 0.9
        d = rep["differential_grader_error_vs_baseline"][lang]
        assert d["error_difference"] > 0.15 and d["ci95"][0] > 0 and d["p_holm"] < 0.05
        assert (
            sum(sum(r.values()) for r in pl[lang]["confusion_human_by_grader"].values())
            == pl[lang]["n"]
        )
    assert any("one annotator" in w for w in rep["warnings"])
    md = render_markdown(rep)
    assert "Differential grader error" in md and "Assumptions" in md


def test_report_is_deterministic(graded: dict[str, object]) -> None:
    key, sheet = _export(graded)
    anns = _annotate(key, sheet, "A")
    a = build_report(key, anns, graded["grades"], n_boot=100)  # type: ignore[arg-type]
    b = build_report(key, anns, graded["grades"], n_boot=100)  # type: ignore[arg-type]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_report_multi_annotator_agreement_and_disputes(graded: dict[str, object]) -> None:
    key, sheet = _export(graded)
    a, b = _annotate(key, sheet, "A"), _annotate(key, sheet, "B", flip_every=9)
    rep = build_report(key, a + b, graded["grades"], n_boot=50)  # type: ignore[arg-type]
    ha = rep["human_agreement"]
    assert ha["annotators"] == ["A", "B"] and ha["n_rows_with_2plus_annotators"] == 90
    assert 0.5 < ha["krippendorff_alpha"] < 1.0 and ha["fleiss_n_rows"] == 90
    assert ha["pairwise"]["A~B"]["n"] == 90
    n_flipped = len(range(0, 90, 9))
    assert ha["n_disputed_excluded"] == n_flipped  # two raters disagreeing = no majority
    assert rep["n_labelled_used"] == 90 - n_flipped


def test_report_needs_annotations(graded: dict[str, object]) -> None:
    key, _ = _export(graded)
    with pytest.raises(ValueError, match="no human labels"):
        build_report(key, [], graded["grades"])  # type: ignore[arg-type]


# ----------------------------------------------------- estimator correctness (synthetic)


def _synthetic(
    seed: int,
) -> tuple[
    list[FinalGrade],
    dict[tuple[str, str, int], ToGrade],
    dict[str, bool],
    dict[str, dict[str, float]],
]:
    """Large population with known true refusal prevalence and known grader Se/Sp per language."""
    rng = random.Random(seed)
    truth_rate = {"en": {"refuse": 0.8, "comply": 0.02}, "hi": {"refuse": 0.5, "comply": 0.02},
                  "hi-Latn": {"refuse": 0.3, "comply": 0.02}}  # fmt: skip
    se_sp = {"en": (0.95, 0.97), "hi": (0.7, 0.85), "hi-Latn": (0.65, 0.8)}
    grades, texts, is_refusal = [], {}, {}
    for lang in truth_rate:
        for item in range(60):
            expected = "refuse" if item % 2 == 0 else "comply"
            for rep in range(10):
                sid = f"{lang}-{item}-{rep}"
                refused = rng.random() < truth_rate[lang][expected]
                se, sp = se_sp[lang]
                said_refuse = rng.random() < (se if refused else 1 - sp)
                grades.append(
                    FinalGrade(
                        run_id="r", subject_model="m", sample_id=sid, epoch=1,
                        item_id=f"i{item}", category=f"c{item % 6}", language=lang,
                        variant="identity", condition="none", split="dev",
                        expected_behaviour=expected,
                        label="REFUSE" if said_refuse else "COMPLY", label_source="ensemble",
                        grader_labels={"g": "REFUSE" if said_refuse else "COMPLY"},
                        harmful_content_provided=False, disagreement=False, parse_failures=0,
                        lexicon_refusal=False, lexicon_matches=[], response_language_heuristic=lang,
                        response_language_graders={"g": lang}, answered_in_prompt_language=True,
                        translated=False, rubric_version="1.0.0", grader_config="g",
                        dataset_hash="d", review_status="unreviewed",
                        provenance="llm:claude",
                    )
                )  # fmt: skip
                texts[("m", sid, 1)] = ToGrade(
                    run_id="r", dataset_hash="d", subject_model="m", sample_id=sid, epoch=1,
                    item_id=f"i{item}", category=f"c{item % 6}", language=lang,
                    variant="identity", condition="none", split="dev", expected=expected,
                    request="q", request_english="q", response="R" if refused else "C",
                    review_status="unreviewed", provenance="llm:claude",
                )  # fmt: skip
                is_refusal[sid] = refused
    true_prev = {
        lang: {
            e: sum(
                is_refusal[g.sample_id]
                for g in grades
                if g.language == lang and g.expected_behaviour == e
            )
            / sum(1 for g in grades if g.language == lang and g.expected_behaviour == e)
            for e in ("refuse", "comply")
        }
        for lang in truth_rate
    }
    return grades, texts, is_refusal, true_prev


def test_corrected_estimates_are_less_biased_than_raw_grader_rates() -> None:
    corrected: dict[str, list[float]] = {"hi": [], "hi-Latn": []}
    observed: dict[str, list[float]] = {"hi": [], "hi-Latn": []}
    truth: dict[str, float] = {}
    for seed in range(12):
        grades, texts, is_ref, true_prev = _synthetic(100)  # fixed population
        key, sheet = build_export(
            grades, texts, n=720, seed=seed, include_test=True, grades_dir="x", grades_sha256="s",
            rubric_version="1.0.0", grader_config="g",
        )  # fmt: skip
        sid_by_row = {r.row_id: r.sample_id for r in key.rows}
        anns = [
            Annotation(row_id=rid, annotator_id="A", label="REFUSE" if is_ref[sid] else "COMPLY")
            for rid, sid in sid_by_row.items()
        ]
        rep = build_report(key, anns, grades, n_boot=10, seed=seed)
        for lang in corrected:
            est = rep["corrected_estimates"][lang]["harmful_refusal_rate"]
            assert est["se_sp_basis"] == "class-specific" and est["corrected"] is not None
            corrected[lang].append(est["corrected"])
            observed[lang].append(est["grader_observed"])
            truth[lang] = true_prev[lang]["refuse"]
    for lang in corrected:
        mean_c = sum(corrected[lang]) / len(corrected[lang])
        assert abs(mean_c - truth[lang]) < 0.04, (lang, mean_c, truth[lang])
    # Where the raw grader rate really is biased (Hindi: observed ~0.43 vs true ~0.52) the
    # correction removes most of it. (In hi-Latn the grader's false positives and false
    # negatives happen to offset in this population, so the raw rate is accidentally close.)
    mean_o = sum(observed["hi"]) / len(observed["hi"])
    assert abs(mean_o - truth["hi"]) > 0.05
    mean_c = sum(corrected["hi"]) / len(corrected["hi"])
    assert abs(mean_c - truth["hi"]) < abs(mean_o - truth["hi"]) / 2


# ------------------------------------------------------------------------------- CLI


def test_cli_roundtrip_and_guards(graded: dict[str, object], tmp_path: Path) -> None:
    gdir = Path(graded["gdir"])  # type: ignore[arg-type]
    out = tmp_path / "exp"
    base = [
        "validate",
        "export",
        "--grades",
        str(gdir),
        "--seed",
        "1",
        "--include-test",
        "--out",
        str(out),
    ]
    assert runner.invoke(app, [*base]).exit_code == 1  # neither --n nor --all
    assert runner.invoke(app, [*base, "--n", "60", "--all"]).exit_code == 1  # both
    r = runner.invoke(app, [*base, "--n", "60"])
    assert r.exit_code == 0, r.output
    key = ExportKey.model_validate_json((out / "key.json").read_text())
    rows = list(csv.DictReader((out / "annotator_sheet.csv").open(encoding="utf-8-sig")))
    anns = _annotate(key, rows, "ann1")
    _write_filled(tmp_path / "f.csv", rows, anns)
    r = runner.invoke(
        app, ["validate", "import", "--export", str(out), "--sheet", str(tmp_path / "f.csv")]
    )
    assert r.exit_code == 0 and "60 new" in r.output
    r = runner.invoke(app, ["validate", "report", "--export", str(out), "--n-boot", "50"])
    assert r.exit_code == 0, r.output
    assert (out / "report.json").exists() and (out / "report.md").exists()
    # the report refuses to run against grades that changed after the export
    gfile = gdir / "grades.jsonl"
    original = gfile.read_text()
    try:
        gfile.write_text(original + " ")
        r = runner.invoke(app, ["validate", "report", "--export", str(out)])
        assert r.exit_code == 1 and "changed since the export" in r.output
    finally:
        gfile.write_text(original)
    assert sha256_hex(gfile.read_text()) == key.grades_sha256


def test_cli_import_errors_are_clean(graded: dict[str, object], tmp_path: Path) -> None:
    gdir = Path(graded["gdir"])  # type: ignore[arg-type]
    out = tmp_path / "exp"
    runner.invoke(
        app,
        [
            "validate",
            "export",
            "--grades",
            str(gdir),
            "--n",
            "30",
            "--include-test",
            "--out",
            str(out),
        ],
    )
    r = runner.invoke(
        app,
        ["validate", "import", "--export", str(out), "--sheet", str(out / "annotator_sheet.csv")],
    )
    assert r.exit_code == 1 and "no human_label" in r.output  # blank sheet, strict by default
    r = runner.invoke(app, ["validate", "report", "--export", str(out)])
    assert r.exit_code == 1 and "no human labels" in r.output
