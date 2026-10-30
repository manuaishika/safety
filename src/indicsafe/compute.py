"""Hardware detection, model recommendations, benchmarking and the time-budget planner."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from indicsafe.data.hashing import sha256_hex
from indicsafe.schema import Item


class PlanError(Exception):
    pass


# --------------------------------------------------------------------------- doctor


class Hardware(BaseModel):
    os: str
    cpu_count: int | None
    ram_total_gb: float | None
    ram_available_gb: float | None
    disk_free_gb: float
    gpu_name: str | None
    vram_gb: float | None
    ollama_binary: bool
    ollama_running: bool
    ollama_models: list[str]


def _meminfo() -> tuple[float | None, float | None]:
    try:
        kv = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            kv[k] = float(v.split()[0]) / 1024 / 1024  # kB -> GB
        return kv["MemTotal"], kv.get("MemAvailable")
    except (OSError, KeyError, ValueError):
        pass
    try:  # macOS
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        ).stdout
        return float(out) / 1024**3, None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None, None


def _gpu() -> tuple[str | None, float | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        name, mem = (x.strip() for x in out.split(","))
        return name, float(mem) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None, None


def _ollama_models() -> tuple[bool, list[str]]:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.5) as r:
            data = json.load(r)
        return True, sorted(m["name"] for m in data.get("models", []))
    except (OSError, ValueError, KeyError):
        return False, []


def detect_hardware(path: Path | None = None) -> Hardware:
    total, avail = _meminfo()
    gpu, vram = _gpu()
    running, models = _ollama_models()
    return Hardware(
        os=f"{platform.system()} {platform.release()}",
        cpu_count=os.cpu_count(),
        ram_total_gb=round(total, 1) if total else None,
        ram_available_gb=round(avail, 1) if avail else None,
        disk_free_gb=round(shutil.disk_usage(path or Path.cwd()).free / 1024**3, 1),
        gpu_name=gpu,
        vram_gb=round(vram, 1) if vram else None,
        ollama_binary=shutil.which("ollama") is not None,
        ollama_running=running,
        ollama_models=models,
    )


# Approximate memory for 4-bit GGUF weights plus runtime/context. Rough guidance only: check
# real usage with `ollama ps`. Disk need is roughly the weights size.
TIERS: tuple[tuple[str, float], ...] = (
    ("1-2B", 2.0),
    ("3-4B", 3.5),
    ("7-9B", 7.0),
    ("12-14B", 11.0),
)


class Recommendation(BaseModel):
    tier: str
    needs_gb: float
    fits: bool
    note: str


def recommend(
    ram_gb: float | None, vram_gb: float | None, disk_free_gb: float
) -> list[Recommendation]:
    """Which Q4 model sizes fit. With a GPU the budget is VRAM; otherwise RAM minus OS headroom."""
    if vram_gb:
        budget, where = vram_gb - 0.5, "VRAM"
    elif ram_gb:
        budget, where = ram_gb - 3.0, "RAM (CPU only)"
    else:
        budget, where = 0.0, "unknown memory"
    out = []
    for tier, need in TIERS:
        fits = need <= budget and need <= disk_free_gb
        note = (
            f"fits in {where}" if fits else f"needs ~{need:.1f} GB; budget {budget:.1f} GB {where}"
        )
        if fits and where.startswith("RAM") and need > 5:
            note += "; expect slow generation on CPU"
        out.append(Recommendation(tier=tier, needs_gb=need, fits=fits, note=note))
    return out


# --------------------------------------------------------------------------- bench


class BenchResult(BaseModel):
    model: str
    n_prompts: int
    seconds_per_sample: float
    tokens_per_sec: float
    mean_output_tokens: float
    max_tokens: int
    token_count_source: str  # "usage" or "estimate"
    measured_at: str
    host: str


def load_bench(path: Path) -> dict[str, BenchResult]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: BenchResult.model_validate(v) for k, v in raw.items()}


def save_bench(path: Path, results: dict[str, BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({k: v.model_dump() for k, v in sorted(results.items())}, indent=2) + "\n",
        encoding="utf-8",
    )


async def bench_model(model_id: str, prompts: Sequence[str], max_tokens: int) -> BenchResult:
    """Time real generations on this machine (needs the model to be available)."""
    from inspect_ai.model import GenerateConfig, get_model

    model = get_model(model_id)
    cfg = GenerateConfig(max_tokens=max_tokens, temperature=0.0)
    secs, toks, from_usage = 0.0, 0, True
    for p in prompts:
        t0 = time.perf_counter()
        out = await model.generate(p, config=cfg)
        secs += time.perf_counter() - t0
        if out.usage and out.usage.output_tokens:
            toks += out.usage.output_tokens
        else:
            from_usage = False
            toks += max(
                1, math.ceil(len(out.completion) / 3)
            )  # conservative; Indic scripts tokenise heavily
    n = len(prompts)
    return BenchResult(
        model=model_id,
        n_prompts=n,
        seconds_per_sample=secs / n,
        tokens_per_sec=toks / secs if secs > 0 else float("inf"),
        mean_output_tokens=toks / n,
        max_tokens=max_tokens,
        token_count_source="usage" if from_usage else "estimate",
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        host=platform.node() or "unknown",
    )


# --------------------------------------------------------------------------- planner


@dataclass(frozen=True)
class Cell:
    model: str
    variant: str
    language: str


@dataclass
class Plan:
    cells: list[Cell]
    item_ids: list[str]
    dropped_cells: list[Cell]
    dropped_item_ids: list[str]
    est_seconds: float
    budget_seconds: float
    full_est_seconds: float


def _units(items: Sequence[Item]) -> list[list[str]]:
    """Harmful item + its benign control stay together; units are stratified by category."""
    by_id = {i.item_id: i for i in items}
    seen: set[str] = set()
    units: dict[str, list[tuple[str, list[str]]]] = {}
    for it in sorted(items, key=lambda i: i.item_id):
        if it.item_id in seen:
            continue
        group = [it.item_id]
        if it.paired_item_id and it.paired_item_id in by_id:
            group.append(it.paired_item_id)
        seen.update(group)
        key = sha256_hex(group[0])
        units.setdefault(it.category, []).append((key, group))
    ordered = {c: [g for _, g in sorted(v)] for c, v in sorted(units.items())}
    out: list[list[str]] = []
    depth = max((len(v) for v in ordered.values()), default=0)
    for d in range(depth):  # round-robin across categories => any prefix is stratified
        for c in ordered:
            if d < len(ordered[c]):
                out.append(ordered[c][d])
    return out


def plan_subset(
    items: Sequence[Item],
    models: Sequence[str],
    variants: Sequence[str],
    languages: Sequence[str],
    supported: Callable[[str, str], bool],
    seconds_per_sample: dict[str, float],
    budget_hours: float,
    multiplier: int = 1,
    min_units: int = 8,
) -> Plan:
    """Largest stratified subset of models x variants x languages x items within a time budget.

    ``multiplier`` = conditions x epochs. Cells are kept in priority order (first variant,
    then model), and the plan keeps as many whole (variant, model) groups as still allow
    ``min_units`` paired item units; items are then added as the budget allows. Everything
    dropped is reported.
    """
    missing = [m for m in models if m not in seconds_per_sample]
    if missing:
        raise PlanError(f"no bench data for {missing}; run `indicsafe bench` first")
    cells = [
        Cell(m, v, lang)
        for v in variants
        for m in models
        for lang in languages
        if supported(v, lang)
    ]
    units = _units(items)
    budget = budget_hours * 3600
    all_ids = [i for u in units for i in u]

    def unit_cost(unit: list[str], cs: Sequence[Cell]) -> float:
        return len(unit) * multiplier * sum(seconds_per_sample[c.model] for c in cs)

    full = unit_cost(all_ids, cells)
    need = min(min_units, len(units))

    # Candidate cell sets, best first: whole (variant, model) groups so that every kept group
    # still compares all its languages; only as a last resort trim languages of the first
    # group (keeping the first-listed language, the baseline).
    groups: list[list[Cell]] = []
    for c in cells:
        if groups and (groups[-1][0].variant, groups[-1][0].model) == (c.variant, c.model):
            groups[-1].append(c)
        else:
            groups.append([c])
    candidates = [[c for g in groups[:k] for c in g] for k in range(len(groups), 0, -1)]
    if groups:
        candidates += [groups[0][:k] for k in range(len(groups[0]) - 1, 0, -1)]

    for kept in candidates:
        total, chosen = 0.0, []
        for u in units:
            uc = unit_cost(u, kept)
            if total + uc > budget:
                break
            total += uc
            chosen.append(u)
        if len(chosen) >= need and chosen:
            ids = [i for u in chosen for i in u]
            kept_set = set(kept)
            return Plan(
                cells=list(kept),
                item_ids=ids,
                dropped_cells=[c for c in cells if c not in kept_set],
                dropped_item_ids=[i for i in all_ids if i not in set(ids)],
                est_seconds=total,
                budget_seconds=budget,
                full_est_seconds=full,
            )
    raise PlanError(
        f"a {budget_hours}h budget cannot cover even one cell with {need} item units "
        f"(full grid needs {full / 3600:.2f}h)"
    )
