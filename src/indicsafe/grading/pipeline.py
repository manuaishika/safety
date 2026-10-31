"""Read Inspect logs, grade every response, write grades.jsonl + grading_meta.json."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import get_model

import indicsafe.mock as mock
from indicsafe.data.builder import git_state
from indicsafe.grading.cache import GradeCache
from indicsafe.grading.ensemble import vote
from indicsafe.grading.grader import grade_one
from indicsafe.grading.language import answered_in_prompt_language, detect_language
from indicsafe.grading.lexicon import lexicon_refusal
from indicsafe.grading.models import FinalGrade, GraderConfig, GraderOutcome
from indicsafe.grading.rubric import Rubric, load_rubric
from indicsafe.grading.translate import Translator


class GradingError(Exception):
    pass


@dataclass(frozen=True)
class ToGrade:
    run_id: str
    dataset_hash: str
    subject_model: str
    sample_id: str
    epoch: int
    item_id: str
    category: str
    language: str
    variant: str
    condition: str
    split: str
    expected: str
    request: str
    request_english: str
    response: str
    review_status: str
    provenance: str


def resolve_run_dir(logs: Path) -> Path:
    return logs.parent if logs.name == "logs" else logs


def read_samples(logs: Path) -> tuple[list[ToGrade], int]:
    """All completed samples in the logs, plus the count of errored samples skipped."""
    infos = list_eval_logs(str(logs if logs.name == "logs" else logs / "logs"))
    if not infos:
        raise GradingError(f"no Inspect logs found under {logs}")
    out: list[ToGrade] = []
    skipped = 0
    for info in infos:
        log = read_eval_log(info)
        if log.status != "success":
            raise GradingError(
                f"{info.name}: run is {log.status}; finish it (re-run the same command) "
                "before grading"
            )
        meta = log.eval.metadata or {}
        for s in log.samples or []:
            if s.error is not None or s.output is None:
                skipped += 1
                continue
            m = s.metadata
            out.append(
                ToGrade(
                    run_id=str(meta.get("run_id", "")),
                    dataset_hash=str(meta.get("dataset_hash", "")),
                    subject_model=log.eval.model,
                    sample_id=str(s.id),
                    epoch=s.epoch,
                    item_id=m["item_id"],
                    category=m["category"],
                    language=m["language"],
                    variant=m["variant"],
                    condition=m["condition"],
                    split=m["split"],
                    expected=m["expected_behaviour"],
                    request=s.input if isinstance(s.input, str) else str(s.input),
                    request_english=m["english_reference"],
                    response=s.output.completion,
                    review_status=m["review_status"],
                    provenance=m["provenance"],
                )
            )
    out.sort(key=lambda x: (x.subject_model, x.sample_id, x.epoch))
    # Lets scripted test doubles look prompts up; harmless for real models.
    mock.register_logged_samples((x.request, x.language, x.expected) for x in out)
    return out, skipped


def check_graders(subjects: set[str], cfg: GraderConfig) -> None:
    clash = sorted(subjects & set(cfg.graders))
    if clash and not cfg.allow_self_grading:
        raise GradingError(
            f"grader model(s) {clash} are also being evaluated; use different graders "
            "(or set allow_self_grading: true and accept the bias)"
        )


async def grade_samples(
    samples: list[ToGrade],
    cfg: GraderConfig,
    cache: GradeCache,
    translator: Translator | None = None,
) -> tuple[list[FinalGrade], dict[str, object]]:
    rubric: Rubric = load_rubric(cfg.rubric_version)
    check_graders({s.subject_model for s in samples}, cfg)

    translations: dict[int, str] = {}
    langs = [detect_language(s.response) for s in samples]
    if translator is not None:
        for i, (s, lang) in enumerate(zip(samples, langs, strict=True)):
            if s.response.strip() and lang != "en" and translator.can_translate(lang):
                translations[i] = translator.translate(s.response, lang)

    todo = [i for i, s in enumerate(samples) if s.response.strip()]
    prompts = {
        i: rubric.user_prompt(
            samples[i].request, samples[i].request_english, samples[i].response, translations.get(i)
        )
        for i in todo
    }
    outcomes: dict[int, list[GraderOutcome]] = {i: [] for i in todo}
    # One grader at a time: local machines hold one model in memory.
    for gid in cfg.graders:
        model = get_model(gid)
        sem = asyncio.Semaphore(cfg.max_connections)
        results = await asyncio.gather(
            *(grade_one(model, gid, rubric, prompts[i], cfg, cache, sem) for i in todo)
        )
        for i, r in zip(todo, results, strict=True):
            outcomes[i].append(r)

    grades: list[FinalGrade] = []
    for i, s in enumerate(samples):
        lex, matches = lexicon_refusal(s.response)
        common = dict(
            run_id=s.run_id, subject_model=s.subject_model, sample_id=s.sample_id, epoch=s.epoch,
            item_id=s.item_id, category=s.category, language=s.language, variant=s.variant,
            condition=s.condition,
            split=s.split, expected_behaviour=s.expected, lexicon_refusal=lex,
            lexicon_matches=matches, response_language_heuristic=langs[i],
            answered_in_prompt_language=answered_in_prompt_language(s.language, langs[i]),
            translated=i in translations, rubric_version=rubric.version, grader_config=cfg.name,
            dataset_hash=s.dataset_hash, review_status=s.review_status, provenance=s.provenance,
        )  # fmt: skip
        if i not in outcomes:  # empty response: no grader call, rule-based and explicit
            grades.append(
                FinalGrade(
                    **common,
                    label="UNCLEAR",
                    label_source="rule:empty",
                    grader_labels={},
                    harmful_content_provided=False,
                    disagreement=False,
                    parse_failures=0,
                    response_language_graders={},
                )
            )
            continue
        outs = outcomes[i]
        label, source, disagreement, harmful = vote(outs)
        grades.append(
            FinalGrade(
                **common, label=label, label_source=source,
                grader_labels={o.grader_model: o.parsed.label if o.parsed else None for o in outs},
                harmful_content_provided=harmful, disagreement=disagreement,
                parse_failures=sum(o.parsed is None for o in outs),
                response_language_graders={
                    o.grader_model: o.parsed.response_language if o.parsed else None for o in outs
                },
            )
        )  # fmt: skip

    flat = [o for outs in outcomes.values() for o in outs]
    multi = [outs for outs in outcomes.values() if sum(o.parsed is not None for o in outs) >= 2]
    stats: dict[str, object] = {
        "n_samples": len(samples),
        "n_graded_by_model": len(todo),
        "n_empty_rule": len(samples) - len(todo),
        "grader_calls": len(flat),
        "cache_hits": sum(o.from_cache for o in flat),
        "parse_failure_rate": (sum(o.parsed is None for o in flat) / len(flat)) if flat else 0.0,
        "disagreement_rate": (sum(vote(outs)[2] for outs in multi) / len(multi) if multi else None),
        "n_all_graders_failed": sum(g.label_source == "all_graders_failed" for g in grades),
        "label_counts": dict(Counter(g.label for g in grades)),
        "translated": len(translations),
    }
    return grades, stats


def write_outputs(
    grades: list[FinalGrade],
    stats: dict[str, object],
    cfg: GraderConfig,
    rubric: Rubric,
    samples: list[ToGrade],
    skipped_errors: int,
    run_dir: Path,
    repo: Path,
    translator: Translator | None,
) -> Path:
    out = run_dir / "grades" / rubric.version / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "grades.jsonl").open("w", encoding="utf-8") as f:
        for g in grades:
            f.write(
                json.dumps(g.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
            )
    sha, dirty = git_state(repo)
    meta = {
        "grader_config": cfg.model_dump(mode="json"),
        "rubric_version": rubric.version,
        "rubric_sha256": rubric.sha256,
        "translator": translator.id if translator else None,
        "dataset_hashes": sorted({s.dataset_hash for s in samples}),
        "run_ids": sorted({s.run_id for s in samples}),
        "subject_models": sorted({s.subject_model for s in samples}),
        "skipped_errored_samples": skipped_errors,
        "stats": stats,
        "git_sha": sha,
        "git_dirty": dirty,
        "graded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "grading_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out
