"""`indicsafe report`: assemble tables, figures and honest limits into one folder."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from indicsafe.analysis.pipeline import PRIMARY, md_table

KEYS = ["model", "variant", "condition", "language"]


def _read(analysis: Path, policy: str, name: str) -> pd.DataFrame:
    return pd.read_csv(analysis / "tables" / policy / f"{name}.csv")


def _headline(analysis: Path, policy: str, expected: str) -> pd.DataFrame:
    r = _read(analysis, policy, "rates")
    g = _read(analysis, policy, "gaps")
    m = _read(analysis, policy, "mcnemar")
    r, g, m = (t[t.expected == expected] for t in (r, g, m))
    out = r.merge(
        g[[*KEYS, "gap", "ci_low", "ci_high", "p_holm"]].rename(
            columns={"ci_low": "gap_low", "ci_high": "gap_high", "p_holm": "gap_p_holm"}),
        on=KEYS, how="left",
    )  # fmt: skip
    out = out.merge(
        m[[*KEYS, "n_pairs", "baseline_only", "language_only", "p_exact", "p_holm", "mde_gap"]]
        .rename(columns={"baseline_only": "en_only", "language_only": "lang_only",
                         "p_holm": "mcnemar_p_holm"}),
        on=KEYS, how="left",
    )  # fmt: skip
    out["rate_ci"] = [f"{a:.0%} [{b:.0%}, {c:.0%}]" if pd.notna(a) else "" for a, b, c in
                      zip(out.rate, out.ci_low, out.ci_high, strict=True)]  # fmt: skip
    out["gap_ci"] = [f"{a:+.0%} [{b:+.0%}, {c:+.0%}]" if pd.notna(a) else "" for a, b, c in
                     zip(out.gap, out.gap_low, out.gap_high, strict=True)]  # fmt: skip
    return out


def _p(v: object) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    x = float(v)  # type: ignore[arg-type]
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def _n(v: object) -> str:
    return "" if v is None or (isinstance(v, float) and v != v) else str(int(v))  # type: ignore[call-overload]


def _section(df: pd.DataFrame, rate_name: str) -> list[str]:
    lines: list[str] = []
    for (model, variant, cond), g in df.groupby(["model", "variant", "condition"], sort=True):
        lines += [f"**{model}** · variant `{variant}` · condition `{cond}`", ""]
        t = pd.DataFrame(
            {
                "language": g["language"],
                "n used": g["n_used"].map(_n),
                "unclear": g["n_unclear"].map(_n),
                rate_name: g["rate_ci"],
                "gap vs en [95% CI]": g["gap_ci"],
                "en right only": g["en_only"].map(_n),
                "lang right only": g["lang_only"].map(_n),
                "McNemar p (Holm)": g["mcnemar_p_holm"].map(_p),
                "min detectable gap": ["" if pd.isna(x) else f"{x:.0%}" for x in g["mde_gap"]],
            }
        )
        lines += [md_table(t, list(t.columns)), ""]
    return lines


def build_report(analysis: Path, out: Path, validation: Path | None = None) -> dict[str, str]:
    man: dict[str, Any] = json.loads((analysis / "analysis_manifest.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("tables", "figures"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        shutil.copytree(analysis / sub, out / sub)
    for f in ("tidy.parquet", "tidy.csv", "cells.csv", "analysis_manifest.json"):
        shutil.copy2(analysis / f, out / f)

    cells = pd.read_csv(analysis / "cells.csv")
    pol = PRIMARY.name
    L: list[str] = ["# IndicSafe results", ""]
    synthetic = [m for m in man["models"] if str(m).startswith("scripted")]
    if synthetic:
        L += [f"> **SYNTHETIC DATA.** Models {synthetic} are scripted test doubles, not real "
              "language models. These numbers only demonstrate the pipeline.", ""]  # fmt: skip
    rc = man["review_status_counts"]
    if man["include_unreviewed"]:
        L += [f"> **Unreviewed translations are included** (review status counts: {rc}). "
              "Prompts were AI-drafted and not checked by native speakers; treat results as "
              "provisional.", ""]  # fmt: skip
    L += [f"Models: {', '.join(man['models'])}. Languages: {', '.join(man['languages'])}. "
          f"Dataset `{man['dataset_hash'][:12]}`, rubric {man['rubric_version']}, grader config "
          f"`{man['grader_config']}`. {man['n_rows_analysed']} of {man['n_rows_graded']} graded "
          f"responses analysed. Scoring policy: `{pol}` (PARTIAL counts as failure; UNCLEAR rows "
          "are excluded and counted).", ""]  # fmt: skip

    L += ["## Do refusals hold in Indian languages?", "",
          "Harmful requests: share the model refused, per language, with the gap to English.", ""]  # fmt: skip
    L += _section(_headline(analysis, pol, "refuse"), "refused (95% CI)")
    L += [
        "## Does the model over-refuse harmless requests?",
        "",
        "Benign controls: share the model answered (over-refusal = 100% minus this).",
        "",
    ]
    L += _section(_headline(analysis, pol, "comply"), "answered (95% CI)")
    L += ["Reading the tables: *gap* is language minus English (negative = weaker); its interval "
          "resamples whole items. The test is exact McNemar on item pairs, Holm-adjusted across "
          "languages, variants and conditions within a model. *en right only / lang right only* are the discordant item "
          "pairs behind the exact McNemar test; *min detectable gap* is the smallest paired "
          "difference this many discordant pairs could detect with 80% power (blank = none).", ""]  # fmt: skip

    L += ["## Sensitivity to how PARTIAL and UNCLEAR are scored", ""]
    rows = []
    for p in ("strict-unclear_exclude", "lenient-unclear_exclude", "strict-unclear_incorrect"):
        g = _read(analysis, p, "gaps")
        g = g[g.expected == "refuse"]
        for _, x in g.iterrows():
            rows.append({"policy": p, "model": x.model, "variant": x.variant,
                         "condition": x.condition, "language": x.language,
                         "harmful gap vs en": f"{x.gap:+.0%} [{x.ci_low:+.0%}, {x.ci_high:+.0%}]"})  # fmt: skip
    if rows:
        L += [md_table(pd.DataFrame(rows), list(rows[0])), ""]

    L += ["## Regression (odds of refusal by language)", "",
          "Two models per class, same covariates (language, plus category/model/variant/"
          "condition where they vary). **GEE** odds ratios are population-averaged and use "
          "item-clustered robust errors; the **Bayesian mixed model** gives item-conditional "
          "odds ratios (typically further from 1) with approximate intervals. Do not compare "
          "the two numerically; they answer different questions.", ""]  # fmt: skip
    reg = _read(analysis, pol, "regression")
    reg = reg[
        reg.term.fillna("").str.contains("language|^-$|random", regex=True) | (reg.method == "-")
    ]
    reg = reg.assign(
        term=reg.term.fillna("").str.replace(
            r"C\(language.*?\)\[T\.(.*?)\]", r"\1 vs en", regex=True
        )
    )
    if not reg.empty:
        L += [
            md_table(
                reg,
                [
                    "expected",
                    "method",
                    "term",
                    "odds_ratio",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "status",
                ],
            ),
            "",
        ]

    # ---------------------------------------------------------------- limits
    L += ["## Limits of these numbers", ""]
    L += [f"- Smallest cell: {int(cells.n_used.min())} responses used (median {int(cells.n_used.median())}); "
          f"cells under 20 responses: {int((cells.n_used < 20).sum())} of {len(cells)}."]  # fmt: skip
    un = cells.groupby("language")[["n_unclear", "n_rows"]].sum()
    un["share"] = un.n_unclear / un.n_rows
    L += ["- UNCLEAR share by language (excluded from rates; high values bias them): "
          + ", ".join(f"{lang} {s:.0%}" for lang, s in un.share.items()) + "."]  # fmt: skip
    L += [f"- Translation review status of analysed prompts: {rc}."]
    if validation is not None and (validation / "report.json").exists():
        v = json.loads((validation / "report.json").read_text())
        L += ["", "### Grader error measured against human labels", "",
              f"Validation export `{v['export_id']}`: {v['n_labelled_used']} human-labelled rows, "
              f"annotators {', '.join(v['human_agreement']['annotators'])}.", ""]  # fmt: skip
        vt = pd.DataFrame([{"language": k, "n": d["n"], "accuracy": d["accuracy_weighted"],
                            "kappa": d["cohen_kappa"], "refusal Se": d["refusal_sensitivity"],
                            "refusal Sp": d["refusal_specificity"]} for k, d in v["per_language"].items()])  # fmt: skip
        L += [md_table(vt, list(vt.columns)), ""]
        diff = v["differential_grader_error_vs_baseline"]
        L += ["Differential grader error vs English: " + "; ".join(
            f"{k} {d['error_difference']:+.1%} (95% CI {d['ci95'][0]:+.1%} to {d['ci95'][1]:+.1%}, "
            f"Holm p {d['p_holm']})" if d['ci95'] else f"{k} {d['error_difference']}" for k, d in diff.items()) + ".", ""]  # fmt: skip
        L += [
            "Refusal rates corrected for this grader error (see the validation report for "
            "assumptions) are in `validation_report.md`.",
            "",
        ]
        shutil.copy2(validation / "report.json", out / "validation_report.json")
        if (validation / "report.md").exists():
            shutil.copy2(validation / "report.md", out / "validation_report.md")
    else:
        L += ["- **No human validation supplied.** Grader error per language is unmeasured, and "
              "it can differ across languages, which would confound every gap above. Run "
              "`indicsafe validate` and pass `--validation`.", ""]  # fmt: skip
    L += ["", "## Reproduce", "",
          f"Inputs: {', '.join(i['grades_dir'] for i in man['inputs'])}. Seed {man['seed']}, "
          f"{man['n_boot']} bootstrap resamples, package versions in `analysis_manifest.json`. "
          "Re-running `indicsafe analyze` on the same grades gives byte-identical tables.", ""]  # fmt: skip
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")

    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "report_manifest.json")
    digest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out / "report_manifest.json").write_text(json.dumps(digest, indent=2, sort_keys=True) + "\n")
    return digest
