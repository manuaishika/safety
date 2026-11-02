"""Reproducibility checks: byte-compare two analysis trees, verify a report against its manifest."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from indicsafe.analysis.pipeline import run_analysis


def digest_tree(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root``, keyed by relative path."""
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def diff_digests(a: dict[str, str], b: dict[str, str]) -> list[str]:
    problems = [f"only in first run: {k}" for k in sorted(a.keys() - b.keys())]
    problems += [f"only in second run: {k}" for k in sorted(b.keys() - a.keys())]
    problems += [f"differs: {k}" for k in sorted(a.keys() & b.keys()) if a[k] != b[k]]
    return problems


def check_reproducible(grades: list[Path], **analysis_kwargs: object) -> tuple[int, list[str]]:
    """Run the analysis twice from scratch and compare every output byte for byte.

    Returns (number of files compared, problems). Empty problems means reproducible.
    """
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "first", Path(tmp) / "second"
        run_analysis(grades, a, **analysis_kwargs)  # type: ignore[arg-type]
        run_analysis(grades, b, **analysis_kwargs)  # type: ignore[arg-type]
        da, db = digest_tree(a), digest_tree(b)
    return len(da), diff_digests(da, db)


def verify_report(report: Path) -> list[str]:
    """Check a report folder against its own manifest: nothing edited, missing or added."""
    mf = report / "report_manifest.json"
    if not mf.exists():
        return [f"{mf} not found"]
    want: dict[str, str] = json.loads(mf.read_text(encoding="utf-8"))
    have = {k: v for k, v in digest_tree(report).items() if k != "report_manifest.json"}
    problems = [f"missing: {k}" for k in sorted(want.keys() - have.keys())]
    problems += [f"not in manifest: {k}" for k in sorted(have.keys() - want.keys())]
    problems += [f"modified: {k}" for k in sorted(want.keys() & have.keys()) if want[k] != have[k]]
    return problems
