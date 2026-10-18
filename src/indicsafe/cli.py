"""indicsafe command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

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
from indicsafe.schema import Item

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Dataset commands.")
app.add_typer(data_app, name="data")

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
) -> None:
    """Expand items into samples and write samples.jsonl + manifest.json."""
    loaded = _load(items)
    flock = FreezeLock(lock)
    problems = flock.check(loaded)
    if problems:
        raise _fail("\n".join(problems))
    try:
        samples = build_samples(loaded, [x.strip() for x in languages.split(",")], flock)
    except (BuildError, DatasetInvalid) as e:
        raise _fail(str(e)) from e
    manifest = make_manifest(loaded, samples, flock, Path.cwd())
    write_built(samples, manifest, out)
    typer.echo(
        f"wrote {len(samples)} samples to {out} (dataset_hash {manifest['dataset_hash'][:12]})"
    )


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


if __name__ == "__main__":
    app()
