"""`indicsafe analyze`: grades -> tidy table, tables per scoring policy, figures, manifest.

Everything is deterministic given the inputs, seed and package versions: floats are rounded,
rows are sorted, no timestamps are written, and figure metadata is stripped, so re-running on
the same grades gives byte-identical files.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import pandas as pd

from indicsafe.analysis.figures import plot_gaps, plot_rates
from indicsafe.analysis.metrics import gaps, mcnemar_tests, rates, round_frame
from indicsafe.analysis.models import fit_models
from indicsafe.analysis.tidy import (
    TIDY_COLUMNS,
    AnalysisError,
    Policy,
    apply_policy,
    filter_review,
    load_grades,
)
from indicsafe.data.builder import git_state
from indicsafe.data.hashing import sha256_hex

PRIMARY = Policy("strict", "exclude")
POLICIES = [
    PRIMARY,
    Policy("lenient", "exclude"),
    Policy("strict", "incorrect"),
    Policy("lenient", "incorrect"),
]
PKGS = ("indicsafe", "pandas", "numpy", "scipy", "statsmodels", "matplotlib", "pyarrow")


@dataclass
class AnalysisResult:
    out_dir: Path
    manifest: dict[str, object]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    round_frame(df).to_csv(path, index=False, lineterminator="\n")


def md_table(df: pd.DataFrame, cols: list[str], pct: tuple[str, ...] = ()) -> str:
    """Small deterministic markdown table (no tabulate dependency)."""

    def cell(col: str, v: object) -> str:
        if v is None or (isinstance(v, float) and v != v):
            return ""
        if isinstance(v, float):
            return f"{v:.0%}" if col in pct else f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(cell(c, r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _file_hash(p: Path) -> str:
    return sha256_hex(p.read_text(encoding="utf-8")) if p.exists() else ""


def _versions() -> dict[str, str]:
    out = {}
    for p in PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = "unknown"
    out["python"] = platform.python_version()
    return out


def run_analysis(
    grades_dirs: list[Path],
    out_dir: Path,
    *,
    include_unreviewed: bool = False,
    baseline: str = "en",
    n_boot: int = 2000,
    seed: int = 0,
    interaction: bool = False,
    bayes: bool = True,
    repo: Path | None = None,
) -> AnalysisResult:
    raw = load_grades(grades_dirs)
    df, review_counts = filter_review(raw, include_unreviewed)
    if baseline not in set(df["language"]):
        raise AnalysisError(f"baseline language {baseline!r} has no rows after filtering")
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy = apply_policy(df, PRIMARY)
    tidy.to_parquet(out_dir / "tidy.parquet", index=False)
    write_csv(tidy, out_dir / "tidy.csv")
    cells = (
        tidy.groupby(["model", "variant", "condition", "expected", "language"], sort=True)
        .agg(
            n_rows=("label", "size"), n_items=("item_id", "nunique"), n_unclear=("unclear", "sum")
        )  # fmt: skip
        .reset_index()
    )
    cells["n_used"] = cells["n_rows"] - cells["n_unclear"]
    write_csv(cells, out_dir / "cells.csv")

    figures: dict[str, list[str]] = {}
    for pol in POLICIES:
        d = apply_policy(df, pol)
        tdir = out_dir / "tables" / pol.name
        r = rates(d, ("language",), n_boot, seed)
        rc = rates(d, ("language", "category"), n_boot, seed)
        g = gaps(d, baseline, n_boot, seed)
        m = mcnemar_tests(d, baseline)
        reg = fit_models(d, baseline, interaction, bayes, seed)
        for name, t in (("rates", r), ("rates_by_category", rc), ("gaps", g),
                        ("mcnemar", m), ("regression", reg)):  # fmt: skip
            write_csv(t, tdir / f"{name}.csv")
        if pol == PRIMARY:
            for k, paths in {**plot_rates(r, out_dir / "figures"),
                             **plot_gaps(g, out_dir / "figures")}.items():  # fmt: skip
                figures[k] = [p.name for p in paths]

    sha, dirty = git_state(repo or Path.cwd())
    manifest: dict[str, object] = {
        "inputs": [
            {
                "grades_dir": str(d),
                "grades_sha256": _file_hash(d / "grades.jsonl"),
                "grading_meta_sha256": _file_hash(d / "grading_meta.json"),
            }
            for d in grades_dirs
        ],  # fmt: skip
        "dataset_hash": str(raw["dataset_hash"].iloc[0]),
        "rubric_version": str(raw["grader_version"].iloc[0]),
        "grader_config": str(raw["grader_config"].iloc[0]),
        "models": sorted(df["model"].unique()),
        "languages": sorted(df["language"].unique()),
        "n_rows_graded": len(raw),
        "n_rows_analysed": len(df),
        "review_status_counts": review_counts,
        "include_unreviewed": include_unreviewed,
        "baseline": baseline,
        "seed": seed,
        "n_boot": n_boot,
        "interaction": interaction,
        "bayes": bayes,
        "policies": [p.name for p in POLICIES],
        "primary_policy": PRIMARY.name,
        "figures": figures,
        "versions": _versions(),
        "git_sha": sha,
        "git_dirty": dirty,
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return AnalysisResult(out_dir, manifest)


__all__ = ["POLICIES", "PRIMARY", "TIDY_COLUMNS", "AnalysisResult", "md_table", "run_analysis"]
