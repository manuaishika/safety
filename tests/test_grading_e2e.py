"""Generate with the scripted subject, grade with scripted graders, check against the truth."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import indicsafe.mock as mock
from conftest import ITEMS_DIR, REPO
from indicsafe.cli import app
from indicsafe.config import RunConfig
from indicsafe.grading.cache import GradeCache
from indicsafe.grading.models import FinalGrade, GraderConfig
from indicsafe.grading.pipeline import (
    GradingError,
    ToGrade,
    grade_samples,
    read_samples,
    resolve_run_dir,
)
from indicsafe.runner import execute, prepare_run

runner = CliRunner()
LANGS = ["en", "hi", "hi-Latn"]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("gen")
    cfg = RunConfig.model_validate(
        {
            "name": "gradee2e",
            "models": ["scripted/leaky"],
            "languages": LANGS,
            "split": "all",
            "generation": {"max_tokens": 32},
        }
    )
    prepared = prepare_run(cfg, ITEMS_DIR, tmp / "lock.json", REPO)
    ok, out = execute(prepared, tmp / "runs")
    assert ok
    return out


def _cfg(**kw: object) -> GraderConfig:
    base: dict[str, object] = {
        "name": "t",
        "graders": ["scriptedgrader/accurate", "scriptedgrader/flip_hi"],
    }
    return GraderConfig.model_validate({**base, **kw})


def _grade(samples: list[ToGrade], cfg: GraderConfig, cache: Path, translator=None):  # type: ignore[no-untyped-def]
    c = GradeCache(cache)
    try:
        return asyncio.run(grade_samples(samples, cfg, c, translator))
    finally:
        c.close()


def test_grades_match_the_scripted_truth_and_disagreements_are_unclear(
    run_dir: Path, tmp_path: Path
) -> None:
    samples, skipped = read_samples(run_dir)
    assert len(samples) == 72 and skipped == 0
    grades, stats = _grade(samples, _cfg(), tmp_path / "c.sqlite")
    by_key = {(g.sample_id, g.epoch): g for g in grades}
    for s in samples:
        g = by_key[(s.sample_id, s.epoch)]
        truth = "REFUSE" if s.response in mock.REFUSALS.values() else "COMPLY"
        if g.disagreement:
            assert g.label == "UNCLEAR" and g.language != "en"  # only the Hindi-ish replies flip
        else:
            assert g.label == truth
        assert g.lexicon_refusal == (truth == "REFUSE")
        assert g.response_language_heuristic == s.language
        assert g.answered_in_prompt_language and not g.translated
        assert g.harmful_content_provided == (truth == "COMPLY" and s.expected == "refuse") or (
            g.disagreement
        )
    assert stats["disagreement_rate"] and 0 < float(stats["disagreement_rate"]) < 0.5  # type: ignore[arg-type]
    assert stats["parse_failure_rate"] == 0.0
    assert all(g.rubric_version == "1.0.0" for g in grades)


def test_accurate_ensemble_is_exact_and_rerun_uses_the_cache(
    run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "grader.log"
    monkeypatch.setenv("INDICSAFE_GRADER_LOG", str(log))
    samples, _ = read_samples(run_dir)
    cfg = _cfg(graders=["scriptedgrader/accurate", "scriptedgrader/bad_json_once"])
    grades, stats = _grade(samples, cfg, tmp_path / "c.sqlite")
    assert stats["disagreement_rate"] == 0.0 and stats["cache_hits"] == 0
    assert all(g.label in ("REFUSE", "COMPLY") for g in grades)
    first_calls = len(log.read_text().split())
    assert first_calls == 72 + 2 * 72  # accurate once; bad_json_once twice (retry)
    grades2, stats2 = _grade(samples, cfg, tmp_path / "c.sqlite")
    assert grades2 == grades and stats2["cache_hits"] == 144
    assert len(log.read_text().split()) == first_calls  # nothing re-graded


def test_all_graders_failing_is_explicit_not_guessed(run_dir: Path, tmp_path: Path) -> None:
    samples, _ = read_samples(run_dir)
    cfg = _cfg(graders=["scriptedgrader/bad_json_always"], max_retries=1)
    grades, stats = _grade(samples[:6], cfg, tmp_path / "c.sqlite")
    assert all(g.label == "UNCLEAR" and g.label_source == "all_graders_failed" for g in grades)
    assert all(g.parse_failures == 1 for g in grades)
    assert stats["n_all_graders_failed"] == 6 and stats["parse_failure_rate"] == 1.0


def test_self_grading_is_blocked_by_default(run_dir: Path, tmp_path: Path) -> None:
    samples, _ = read_samples(run_dir)
    with pytest.raises(GradingError, match="also being evaluated"):
        _grade(samples, _cfg(graders=["scripted/leaky"]), tmp_path / "c.sqlite")


def test_empty_responses_are_rule_labelled_without_calling_a_grader(
    run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = tmp_path / "g.log"
    monkeypatch.setenv("INDICSAFE_GRADER_LOG", str(log))
    s = read_samples(run_dir)[0][0]
    empty = ToGrade(**{**s.__dict__, "response": "  "})
    grades, _ = _grade([empty], _cfg(), tmp_path / "c.sqlite")
    assert grades[0].label == "UNCLEAR" and grades[0].label_source == "rule:empty"
    assert not log.exists()


class FakeTranslator:
    id = "fake:translator"

    def can_translate(self, language: str) -> bool:
        return language == "hi"

    def translate(self, text: str, language: str) -> str:
        return "I can't help with that request."


def test_translation_is_recorded_as_noise_source(run_dir: Path, tmp_path: Path) -> None:
    samples, _ = read_samples(run_dir)
    grades, stats = _grade(
        samples, _cfg(graders=["scriptedgrader/accurate"]), tmp_path / "c.sqlite", FakeTranslator()
    )
    hi = [g for g, s in zip(grades, samples, strict=True) if s.language == "hi"]
    assert hi and all(g.translated for g in hi)
    assert not any(g.translated for g, s in zip(grades, samples, strict=True) if s.language != "hi")
    assert stats["translated"] == len(hi)
    assert all(g.label in ("REFUSE", "COMPLY") for g in grades)  # original stays authoritative


def test_cli_grade_writes_reproducible_outputs(run_dir: Path, tmp_path: Path) -> None:
    args = [
        "grade",
        "--logs", str(run_dir),
        "--grader-config", str(REPO / "configs" / "graders" / "scripted.yaml"),
        "--cache", str(tmp_path / "cache.sqlite"),
    ]  # fmt: skip
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output
    out = resolve_run_dir(run_dir) / "grades" / "1.0.0" / "scripted"
    first = (out / "grades.jsonl").read_bytes()
    rows = [FinalGrade.model_validate_json(line) for line in first.decode().splitlines()]
    assert len(rows) == 72
    meta = json.loads((out / "grading_meta.json").read_text())
    assert meta["rubric_version"] == "1.0.0" and len(meta["rubric_sha256"]) == 64
    assert meta["subject_models"] == ["scripted/leaky"] and meta["stats"]["cache_hits"] == 0
    r2 = runner.invoke(app, args)
    assert r2.exit_code == 0 and "cache hits: 144" in r2.output
    assert (out / "grades.jsonl").read_bytes() == first  # byte-identical on re-grade


def test_cli_grade_errors_are_clean(tmp_path: Path) -> None:
    cfg = str(REPO / "configs" / "graders" / "scripted.yaml")
    r = runner.invoke(app, ["grade", "--logs", str(tmp_path), "--grader-config", cfg])
    assert r.exit_code == 1 and "no Inspect logs" in r.output
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ngraders: [openai/gpt-4o]\n")
    r = runner.invoke(app, ["grade", "--logs", str(tmp_path), "--grader-config", str(bad)])
    assert r.exit_code == 1 and "hosted" in r.output
