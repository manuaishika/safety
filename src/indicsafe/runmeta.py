"""RunMeta construction and persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from indicsafe.config import RunConfig, provider_of, run_id
from indicsafe.data.builder import git_state
from indicsafe.schema import RunMeta

PACKAGES = ("indicsafe", "inspect-ai", "pydantic", "pyyaml", "typer")


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "unknown"
    return out


def build_run_meta(
    config: RunConfig, dataset_hash: str, repo: Path, extra: dict[str, object] | None = None
) -> RunMeta:
    sha, dirty = git_state(repo)
    cfg: dict[str, object] = {**config.model_dump(mode="json"), **(extra or {})}
    cfg["providers"] = sorted({provider_of(m) for m in config.models})
    return RunMeta(
        run_id=run_id(config, dataset_hash),
        config=cfg,
        dataset_hash=dataset_hash,
        git_sha=sha,
        git_dirty=dirty,
        package_versions=package_versions(),
        model_ids=list(config.models),
        seeds={"variants": config.seed, "generation": config.generation.seed or 0},
        started_at=datetime.now(UTC),
    )


def write_run_meta(meta: RunMeta, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_meta.json"
    path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
