"""Orchestrates a run: samples -> optional time-budget plan -> Inspect eval_set (resumable)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from indicsafe.compute import BenchResult, Cell, Plan, plan_subset
from indicsafe.config import RunConfig
from indicsafe.data.builder import build_samples
from indicsafe.data.hashing import dataset_hash
from indicsafe.data.loaders import load_items
from indicsafe.data.splits import FreezeLock
from indicsafe.data.validate import validate_items
from indicsafe.runmeta import build_run_meta, write_run_meta
from indicsafe.schema import RunMeta, Sample
from indicsafe.variants import REGISTRY


@dataclass
class PreparedRun:
    config: RunConfig
    meta: RunMeta
    samples_by_model: dict[str, list[Sample]]
    skipped_cells: list[tuple[str, str]]  # (variant, language) the variant does not support
    plan: Plan | None = None
    notes: list[str] = field(default_factory=list)


def prepare_run(
    config: RunConfig,
    items_dir: Path,
    lock_path: Path,
    repo: Path,
    bench: dict[str, BenchResult] | None = None,
    max_hours: float | None = None,
) -> PreparedRun:
    items = load_items(items_dir)
    validate_items(items)
    lock = FreezeLock(lock_path)
    problems = lock.check(items)
    if problems:
        raise RuntimeError("frozen test items changed:\n" + "\n".join(problems))
    skipped = [
        (v, lang)
        for v in config.variants
        for lang in config.languages
        if lang not in REGISTRY[v].languages
    ]
    if skipped and not config.skip_unsupported:
        raise RuntimeError(f"unsupported (variant, language) cells: {skipped}")
    samples = build_samples(
        items,
        config.languages,
        lock,
        variants=config.variants,
        conditions=config.conditions,
        seed=config.seed,
        skip_unsupported=True,
    )
    cats = set(config.categories) if config.categories else None
    samples = [
        s
        for s in samples
        if (cats is None or s.category in cats) and config.split in ("all", s.split)
    ]
    if not samples:
        raise RuntimeError("no samples match the config filters (categories / split)")

    plan: Plan | None = None
    by_model: dict[str, list[Sample]] = {m: list(samples) for m in config.models}
    if max_hours is not None:
        if bench is None:
            raise RuntimeError("--max-hours needs bench data")
        kept_items = {s.item_id for s in samples}
        sel_items = [i for i in items if i.item_id in kept_items]
        plan = plan_subset(
            sel_items,
            config.models,
            config.variants,
            config.languages,
            lambda v, lang: lang in REGISTRY[v].languages,
            {m: bench[m].seconds_per_sample for m in config.models if m in bench},
            max_hours,
            multiplier=len(config.conditions) * config.epochs,
        )
        keep_ids = set(plan.item_ids)
        by_model = {
            m: [
                s
                for s in samples
                if s.item_id in keep_ids and Cell(m, s.variant, s.language) in set(plan.cells)
            ]
            for m in config.models
        }

    extra: dict[str, object] = {
        "skipped_cells": [list(c) for c in skipped],
        "samples_per_model": {m: len(v) for m, v in by_model.items()},
    }
    if plan:
        extra["plan"] = {
            "budget_hours": max_hours,
            "est_hours": plan.est_seconds / 3600,
            "full_grid_hours": plan.full_est_seconds / 3600,
            "dropped_cells": [[c.model, c.variant, c.language] for c in plan.dropped_cells],
            "n_items_kept": len(plan.item_ids),
            "n_items_dropped": len(plan.dropped_item_ids),
        }
    meta = build_run_meta(config, dataset_hash(items), repo, extra)
    return PreparedRun(config, meta, by_model, skipped, plan)


def describe(run: PreparedRun, bench: dict[str, BenchResult]) -> str:
    c = run.config
    lines = [f"run_id: {run.meta.run_id}", f"dataset_hash: {run.meta.dataset_hash[:12]}"]
    total = 0
    for m, ss in run.samples_by_model.items():
        gens = len(ss) * c.epochs
        total += gens
        est = (
            f"~{gens * bench[m].seconds_per_sample / 3600:.2f} h"
            if m in bench
            else "no bench data (run `indicsafe bench`)"
        )
        lines.append(f"  {m}: {len(ss)} samples x {c.epochs} epoch(s) = {gens} generations, {est}")
    lines.append(f"total generations: {total} (no model calls made in dry run)")
    if run.skipped_cells:
        lines.append(f"skipped unsupported (variant, language): {run.skipped_cells}")
    if run.plan:
        p = run.plan
        lines.append(
            f"plan: {len(p.item_ids)} items kept, {len(p.dropped_item_ids)} dropped; "
            f"est {p.est_seconds / 3600:.2f} h of {p.budget_seconds / 3600:.2f} h budget "
            f"(full grid {p.full_est_seconds / 3600:.2f} h)"
        )
        if p.dropped_cells:
            lines.append(
                f"  dropped cells: {[(x.model, x.variant, x.language) for x in p.dropped_cells]}"
            )
    return "\n".join(lines)


def execute(run: PreparedRun, out_root: Path) -> tuple[bool, Path]:
    """Run the generations. Re-running the same config resumes: finished samples are reused."""
    from inspect_ai import eval_set

    from indicsafe.tasks import make_task

    c = run.config
    out_dir = out_root / run.meta.run_id
    log_dir = out_dir / "logs"
    write_run_meta(run.meta, out_dir)
    tasks = [
        make_task(
            ss,
            name=f"indic_refusal_{i}",
            model=m,
            generation=c.generation,
            epochs=c.epochs,
            metadata={
                "run_id": run.meta.run_id,
                "dataset_hash": run.meta.dataset_hash,
                "seed": c.seed,
            },
        )
        for i, (m, ss) in enumerate(run.samples_by_model.items())
        if ss
    ]
    ok, _ = eval_set(
        tasks,
        log_dir=str(log_dir),
        score=False,
        log_buffer=1,  # flush after every sample so a killed run loses at most in-flight ones
        retry_attempts=2,
        fail_on_error=True,  # an errored sample fails the task so the retry/resume redoes it
        max_tasks=1,  # one model at a time: local machines hold one model in memory
        max_samples=c.generation.max_connections,
        display="plain",
    )
    if ok:
        done = run.meta.model_copy(update={"finished_at": datetime.now(UTC)})
        write_run_meta(done, out_dir)
    return ok, out_dir


def bench_prompts(config: RunConfig, items_dir: Path, lock_path: Path, n: int) -> list[str]:
    """n identity-variant prompts spread evenly across the configured languages."""
    import indicsafe.mock as mock

    items = load_items(items_dir)
    samples = build_samples(items, config.languages, FreezeLock(lock_path))
    mock.register_samples(samples)
    by_lang: dict[str, list[str]] = {lang: [] for lang in config.languages}
    for s in samples:
        by_lang[s.language].append(s.text)
    out: list[str] = []
    i = 0
    while len(out) < n and any(by_lang.values()):
        for lang in config.languages:
            if by_lang[lang] and len(out) < n:
                out.append(by_lang[lang][i % len(by_lang[lang])])
        i += 1
    return out
