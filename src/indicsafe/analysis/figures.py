"""Publication figures (matplotlib), styled per the dataviz method and byte-reproducible.

* Language is the entity: it keeps one colour from the validated 7-slot categorical order in
  every figure (colour follows the entity, never its rank) and is also named on the axis, so
  identity is never colour alone.
* Three palette slots are below 3:1 contrast on the light surface, so every mark carries a
  direct value label in text ink and each figure ships next to its data table.
* Light surface only; the validated dark steps are in the dataviz reference if needed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
# Validated (scripts/validate_palette.js, light + dark) fixed order; language -> slot.
LANG_COLORS = {
    "en": "#2a78d6", "hi": "#eb6834", "hi-Latn": "#1baf7a", "mr": "#eda100",
    "bn": "#e87ba4", "ta": "#008300", "te": "#6250d6",
}  # fmt: skip
LANG_ORDER = list(LANG_COLORS)
MARKER = 7  # points, ~9.3 px: above the 8 px minimum
MAX_ROWS = 6

MEASURE_TITLE = {
    "harmful_refusal_rate": "Harmful requests refused",
    "benign_compliance_rate": "Benign requests answered",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
            "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
            "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10,
            "axes.titleweight": "regular", "axes.titlelocation": "left",
            "font.family": "DejaVu Sans",
        }
    )  # fmt: skip


def _save(fig: Figure, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, meta in (
        ("png", {"Software": None}),
        ("pdf", {"CreationDate": None, "ModDate": None, "Creator": "indicsafe"}),
    ):  # fmt: skip
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", metadata=meta)
        paths.append(p)
    plt.close(fig)
    return paths


def _legend(fig: Figure, langs: list[str]) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LANG_COLORS[lang],
               markeredgecolor=SURFACE, markersize=MARKER, label=lang)
        for lang in langs
    ]  # fmt: skip
    fig.legend(
        handles=handles, frameon=False, ncol=len(langs), loc="upper right",
        bbox_to_anchor=(1.0, 1.04), fontsize=8, labelcolor=INK_2, handlelength=1,
    )  # fmt: skip


def _langs(df: pd.DataFrame) -> list[str]:
    present = set(df["language"])
    return [lang for lang in LANG_ORDER if lang in present] + sorted(present - set(LANG_ORDER))


def plot_rates(rates: pd.DataFrame, out_dir: Path) -> dict[str, list[Path]]:
    """One figure per measure: columns = models, rows = (variant, condition); dot + 95% CI."""
    _style()
    made: dict[str, list[Path]] = {}
    for measure, title in MEASURE_TITLE.items():
        d = rates[(rates["measure"] == measure) & rates["rate"].notna()]
        if d.empty:
            continue
        models = sorted(d["model"].unique())
        cells: list[tuple[str, str]] = [
            (str(v), str(c))
            for v, c in d[["variant", "condition"]]
            .drop_duplicates()
            .sort_values(["variant", "condition"])
            .itertuples(index=False, name=None)
        ][:MAX_ROWS]
        langs = _langs(d)
        fig, axes = plt.subplots(
            len(cells), len(models), figsize=(3.4 * len(models) + 0.6, 2.3 * len(cells) + 0.8),
            squeeze=False, sharey=True, layout="constrained",
        )  # fmt: skip
        for r, (variant, cond) in enumerate(cells):
            for c, model in enumerate(models):
                ax = axes[r][c]
                g = d[(d.model == model) & (d.variant == variant) & (d.condition == cond)]
                for x, lang in enumerate(langs):
                    row = g[g.language == lang]
                    if row.empty:
                        continue
                    v = row.iloc[0]
                    ax.vlines(x, v.ci_low, v.ci_high, color=LANG_COLORS.get(lang, INK_2), lw=1.5)
                    ax.plot(x, v.rate, "o", ms=MARKER, color=LANG_COLORS.get(lang, INK_2),
                            mec=SURFACE, mew=1.5)  # fmt: skip
                    ax.annotate(f"{v.rate:.0%}", (x, v.rate), textcoords="offset points",
                                xytext=(9, 0), va="center", fontsize=8, color=INK)  # fmt: skip
                ax.set_xticks(range(len(langs)), langs)
                ax.set_xlim(-0.6, len(langs) - 0.4)
                ax.set_ylim(-0.03, 1.05)
                ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
                ax.grid(axis="x", visible=False)
                head = model if r == 0 else ""
                ax.set_title(
                    (head + ("\n" if head else "") + f"{variant} / {cond}").strip(), fontsize=8
                )
        _legend(fig, langs)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11)
        made[measure] = _save(fig, out_dir, f"rates_{measure}")
    return made


def plot_gaps(gaps: pd.DataFrame, out_dir: Path) -> dict[str, list[Path]]:
    """Gap vs the baseline language with 95% CI. Rows: variant/condition/language."""
    _style()
    made: dict[str, list[Path]] = {}
    for expected, measure in (
        ("refuse", "harmful_refusal_rate"),
        ("comply", "benign_compliance_rate"),
    ):
        d = gaps[(gaps["expected"] == expected) & gaps["gap"].notna()]
        if d.empty:
            continue
        models = sorted(d["model"].unique())
        langs = _langs(d)
        n_rows = (
            d[["model", "variant", "condition", "language"]]
            .drop_duplicates()
            .groupby("model")
            .size()
            .max()
        )
        fig, axes = plt.subplots(
            1,
            len(models),
            figsize=(3.8 * len(models) + 0.6, 0.4 * float(n_rows) + 1.8),
            squeeze=False,
            sharex=True,
            layout="constrained",
        )
        for c, model in enumerate(models):
            ax = axes[0][c]
            g = d[d.model == model].copy()
            g["lang_i"] = g["language"].map({lang: i for i, lang in enumerate(langs)})
            g = g.sort_values(["variant", "condition", "lang_i"]).reset_index(drop=True)
            for y, v in g.iterrows():
                col = LANG_COLORS.get(v.language, INK_2)
                ax.hlines(y, v.ci_low, v.ci_high, color=col, lw=1.5)
                ax.plot(v.gap, y, "o", ms=MARKER, color=col, mec=SURFACE, mew=1.5)
                ax.annotate(f"{v.gap:+.0%}", (v.ci_high, y), textcoords="offset points",
                            xytext=(6, 0), va="center", fontsize=8, color=INK)  # fmt: skip
            ax.axvline(0, color=INK_2, lw=1)
            labels = [f"{v.variant}/{v.condition} {v.language}" for _, v in g.iterrows()]
            ax.set_yticks(range(len(g)), labels, fontsize=7)
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(lambda x, _: f"{x:+.0%}")
            ax.grid(axis="y", visible=False)
            ax.set_title(model, fontsize=8)
        _legend(fig, langs)
        fig.supxlabel(
            "gap vs baseline language, 95% CI (whole items resampled)", fontsize=8, color=INK_2
        )
        fig.suptitle(
            f"{MEASURE_TITLE[measure]}: gap vs English", x=0.01, ha="left", fontsize=11, y=1.02
        )
        made[f"gaps_{measure}"] = _save(fig, out_dir, f"gaps_{measure}")
    return made


__all__ = ["LANG_COLORS", "plot_gaps", "plot_rates"]
