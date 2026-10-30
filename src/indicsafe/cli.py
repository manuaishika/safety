"""indicsafe command line."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer

from indicsafe.compute import (
    PlanError,
    bench_model,
    detect_hardware,
    load_bench,
    recommend,
    save_bench,
)
from indicsafe.config import load_config, provider_of
from indicsafe.data.builder import (
    BuildError,
    build_samples,
    compute_stats,
    make_manifest,
    write_built,
)
from indicsafe.data.loaders import LoadError, load_items
from indicsafe.data.splits import FreezeLock
from indicsafe.data.validate import DatasetInvalid, check_items
from indicsafe.runner import bench_prompts, describe, execute, prepare_run
from indicsafe.schema import Item
from indicsafe.variants import get_variant, variant_names

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Dataset commands.")
app.add_typer(data_app, name="data")

ConfigOpt = Annotated[Path, typer.Option("--config", help="Run config YAML.")]
BenchPath = Annotated[Path, typer.Option("--bench-file", help="Bench results JSON.")]

ItemsDir = Annotated[Path, typer.Option("--items", help="Directory of item YAML files.")]
LockPath = Annotated[Path, typer.Option("--lock", help="Frozen-test lock file.")]


def _fail(msg: str) -> typer.Exit:
    typer.echo(msg, err=True)
    return typer.Exit(code=1)


def _load(items_dir: Path) -> list[Item]:
    try:
        return load_items(items_dir)
    except LoadError as e:
        raise _fail(str(e)) from e


@data_app.command("validate")
def validate(
    items: ItemsDir = Path("data/items"), lock: LockPath = Path("data/test_freeze.json")
) -> None:
    """Validate items; also fails if frozen test items changed."""
    loaded = _load(items)
    problems = check_items(loaded) + FreezeLock(lock).check(loaded)
    if problems:
        raise _fail(f"{len(problems)} problem(s):\n" + "\n".join(f"  - {p}" for p in problems))
    typer.echo(f"OK: {len(loaded)} items valid")


@data_app.command("build")
def build(
    languages: Annotated[str, typer.Option(help="Comma-separated BCP-47 tags.")] = "en,hi,hi-Latn",
    items: ItemsDir = Path("data/items"),
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("data/built"),
    lock: LockPath = Path("data/test_freeze.json"),
    variants: Annotated[str, typer.Option(help="Comma-separated variant names.")] = "identity",
    seed: Annotated[int, typer.Option(help="Seed for variant randomness.")] = 0,
    skip_unsupported: Annotated[
        bool, typer.Option(help="Skip (variant, language) pairs a variant does not support.")
    ] = False,
) -> None:
    """Expand items into samples and write samples.jsonl + manifest.json."""
    loaded = _load(items)
    flock = FreezeLock(lock)
    problems = flock.check(loaded)
    if problems:
        raise _fail("\n".join(problems))
    try:
        samples = build_samples(
            loaded,
            [x.strip() for x in languages.split(",")],
            flock,
            variants=[x.strip() for x in variants.split(",")],
            seed=seed,
            skip_unsupported=skip_unsupported,
        )
    except (BuildError, DatasetInvalid) as e:
        raise _fail(str(e)) from e
    manifest = make_manifest(loaded, samples, flock, Path.cwd(), seed)
    write_built(samples, manifest, out)
    typer.echo(
        f"wrote {len(samples)} samples to {out} (dataset_hash {manifest['dataset_hash'][:12]})"
    )


@app.command("variants")
def list_variants() -> None:
    """List registered prompt variants."""
    for name in variant_names():
        v = get_variant(name)
        typer.echo(f"{name}  v{v.version}  [{', '.join(sorted(v.languages))}]  {v.description}")


@data_app.command("stats")
def stats(
    items: ItemsDir = Path("data/items"), lock: LockPath = Path("data/test_freeze.json")
) -> None:
    """Print counts per language, review status, split and category."""
    typer.echo(json.dumps(compute_stats(_load(items), FreezeLock(lock)), indent=2))


@data_app.command("freeze")
def freeze(
    reason: Annotated[str, typer.Option(help="Why the test split is being frozen.")],
    items: ItemsDir = Path("data/items"),
    lock: LockPath = Path("data/test_freeze.json"),
) -> None:
    """Freeze the current test split (item content hashes)."""
    loaded = _load(items)
    flock = FreezeLock(lock)
    try:
        flock.freeze(loaded, reason)
    except (RuntimeError, ValueError) as e:
        raise _fail(str(e)) from e
    n = len(flock.data["items"]) if flock.data else 0
    typer.echo(f"frozen {n} test items -> {lock}")


@data_app.command("unfreeze")
def unfreeze(
    reason: Annotated[str, typer.Option(help="Logged reason for touching the test split.")],
    items: ItemsDir = Path("data/items"),
    lock: LockPath = Path("data/test_freeze.json"),
) -> None:
    """Re-baseline the freeze lock after a deliberate, logged change to test items."""
    loaded = _load(items)
    try:
        FreezeLock(lock).unfreeze(loaded, reason)
    except (RuntimeError, ValueError) as e:
        raise _fail(str(e)) from e
    typer.echo("lock re-baselined and logged")


@app.command("doctor")
def doctor() -> None:
    """Detect RAM / VRAM / CPU / Ollama / disk and recommend model sizes that fit."""
    hw = detect_hardware()
    typer.echo(json.dumps(hw.model_dump(), indent=2))
    typer.echo("\nModel sizes (4-bit):")
    for r in recommend(hw.ram_total_gb, hw.vram_gb, hw.disk_free_gb):
        typer.echo(f"  {'OK ' if r.fits else 'no '} {r.tier:7s} {r.note}")
    if hw.ram_total_gb is None:
        typer.echo("RAM could not be detected on this OS: tell me your RAM/GPU and I'll size runs.")
    if not hw.ollama_running:
        typer.echo("Ollama is not running here; local runs need it (or HF/vLLM).")


@app.command("models")
def list_models(config: ConfigOpt) -> None:
    """Print the Ollama tags a config needs, one per line (used by the bootstrap script)."""
    cfg = load_config(config)
    for m in cfg.models:
        if provider_of(m) == "ollama":
            typer.echo(m.split("/", 1)[1])
        else:
            typer.echo(f"note: {m} is not an Ollama model; nothing to pull", err=True)


@app.command("bench")
def bench(
    config: ConfigOpt,
    n: Annotated[int, typer.Option(help="Prompts to time per model.")] = 12,
    bench_file: BenchPath = Path("results/bench.json"),
    items: ItemsDir = Path("data/items"),
    lock: LockPath = Path("data/test_freeze.json"),
) -> None:
    """Measure seconds/sample and tokens/sec for each configured model on this machine."""
    cfg = load_config(config)
    prompts = bench_prompts(cfg, items, lock, n)
    results = load_bench(bench_file)
    for m in cfg.models:
        typer.echo(f"benchmarking {m} on {len(prompts)} prompts ...")
        r = asyncio.run(bench_model(m, prompts, cfg.generation.max_tokens))
        results[m] = r
        rate = f"{r.seconds_per_sample:.2f} s/sample, {r.tokens_per_sec:.1f} tok/s"
        typer.echo(f"  {rate} ({r.token_count_source})")
    save_bench(bench_file, results)
    typer.echo(f"saved {bench_file}")


@app.command("run")
def run(
    config: ConfigOpt,
    dry_run: Annotated[bool, typer.Option(help="Count samples/time; make no model calls.")] = False,
    max_hours: Annotated[
        float | None, typer.Option(help="Keep the largest stratified subset that fits.")
    ] = None,
    out: Annotated[Path, typer.Option(help="Root for run folders.")] = Path("results/runs"),
    bench_file: BenchPath = Path("results/bench.json"),
    items: ItemsDir = Path("data/items"),
    lock: LockPath = Path("data/test_freeze.json"),
) -> None:
    """Generate model responses. Re-running the same config resumes where it stopped."""
    cfg = load_config(config)
    bench_data = load_bench(bench_file)
    try:
        prepared = prepare_run(cfg, items, lock, Path.cwd(), bench_data or None, max_hours)
    except (RuntimeError, PlanError, BuildError, DatasetInvalid) as e:
        raise _fail(str(e)) from e
    typer.echo(describe(prepared, bench_data))
    if dry_run:
        return
    ok, out_dir = execute(prepared, out)
    typer.echo(
        f"{'finished' if ok else 'INCOMPLETE (re-run the same command to resume)'}: {out_dir}"
    )
    if not ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
