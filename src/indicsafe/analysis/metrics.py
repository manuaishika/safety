"""Rates, gaps vs a baseline language, exact McNemar tests, and the power helper.

Units: items are the unit of replication (translations of one item are not independent).
Intervals for gaps resample whole items; Wilson intervals are used only for marginal rates
where every item appears once. All randomness is seeded from the group key, so results do not
depend on the order groups are processed in.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from indicsafe.data.hashing import sha256_hex
from indicsafe.stats import bootstrap_p_two_sided, holm, wilson_interval

GROUP = ["model", "variant", "condition", "expected"]
MEASURE = {"refuse": "harmful_refusal_rate", "comply": "benign_compliance_rate"}


def _seed(base: int, *key: object) -> int:
    return int(sha256_hex(f"{base}|{'|'.join(map(str, key))}")[:8], 16)


def _used(d: pd.DataFrame) -> pd.DataFrame:
    return d[d["correct"].notna()]


# ------------------------------------------------------------------------------ rates


def _boot_rate_ci(k: np.ndarray, n: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    m = len(k)
    idx = np.random.default_rng(seed).integers(0, m, size=(n_boot, m))
    den = n[idx].sum(axis=1)
    ok = den > 0
    reps = k[idx].sum(axis=1)[ok] / den[ok]
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def rates(
    d: pd.DataFrame, by: Sequence[str] = ("language",), n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Success rate per group (harmful: refusal rate; benign: compliance rate).

    Columns: n_rows (incl. unclear), n_unclear, n_used, n_items, k, rate, ci_low, ci_high,
    ci_method ('wilson' when each item appears once, else item-cluster bootstrap), measure.
    """
    keys = [*GROUP, *by]
    out: list[dict[str, object]] = []
    for key, g in d.groupby(keys, sort=True, dropna=False):
        key_t = key if isinstance(key, tuple) else (key,)
        used = _used(g)
        n_used = len(used)
        row: dict[str, object] = dict(zip(keys, key_t, strict=True))
        row.update(
            n_rows=len(g), n_unclear=int(g["unclear"].sum()), n_used=n_used,
            n_items=int(g["item_id"].nunique()),
        )  # fmt: skip
        if n_used == 0:
            row.update(k=0, rate=np.nan, ci_low=np.nan, ci_high=np.nan, ci_method="none")
        else:
            k = int(used["correct"].astype(bool).sum())
            row.update(k=k, rate=k / n_used)
            per_item = used.groupby("item_id")["correct"].agg(
                k=lambda s: int(s.astype(bool).sum()), n="count"
            )
            if (per_item["n"] == 1).all():
                lo, hi = wilson_interval(k, n_used)
                row.update(ci_low=lo, ci_high=hi, ci_method="wilson")
            else:
                lo, hi = _boot_rate_ci(
                    per_item["k"].to_numpy(float), per_item["n"].to_numpy(float), n_boot,
                    _seed(seed, *key_t),
                )  # fmt: skip
                row.update(ci_low=lo, ci_high=hi, ci_method="cluster_bootstrap")
        row["measure"] = MEASURE[str(row["expected"])]
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------ gaps


def _per_item(g: pd.DataFrame, lang: str) -> pd.DataFrame:
    sub = _used(g[g["language"] == lang])
    agg = sub.groupby("item_id")["correct"].agg(k=lambda s: int(s.astype(bool).sum()), n="count")
    return pd.DataFrame(agg)


def gaps(d: pd.DataFrame, baseline: str = "en", n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Rate(language) - rate(baseline), item-cluster bootstrap CI and two-sided p, per group.

    A negative harmful-refusal gap means refusals weaken in that language.
    """
    out: list[dict[str, object]] = []
    for key, g in d.groupby(GROUP, sort=True, dropna=False):
        base = _per_item(g, baseline)
        for lang in sorted(set(g["language"]) - {baseline}):
            other = _per_item(g, lang)
            items = sorted(set(base.index) | set(other.index))
            kb = base["k"].reindex(items, fill_value=0).to_numpy(float)
            nb = base["n"].reindex(items, fill_value=0).to_numpy(float)
            kl = other["k"].reindex(items, fill_value=0).to_numpy(float)
            nl = other["n"].reindex(items, fill_value=0).to_numpy(float)
            row: dict[str, object] = dict(zip(GROUP, key, strict=True))
            row.update(language=lang, baseline=baseline, n_items=len(items),
                       n_used_language=int(nl.sum()), n_used_baseline=int(nb.sum()))  # fmt: skip
            if nb.sum() == 0 or nl.sum() == 0:
                row.update(rate_language=np.nan, rate_baseline=np.nan, gap=np.nan,
                           ci_low=np.nan, ci_high=np.nan, p_bootstrap=np.nan)  # fmt: skip
            else:
                rate_l, rate_b = kl.sum() / nl.sum(), kb.sum() / nb.sum()
                m = len(items)
                idx = np.random.default_rng(_seed(seed, *key, lang)).integers(
                    0, m, size=(n_boot, m)
                )
                dl, db = nl[idx].sum(axis=1), nb[idx].sum(axis=1)
                ok = (dl > 0) & (db > 0)
                reps = kl[idx].sum(axis=1)[ok] / dl[ok] - kb[idx].sum(axis=1)[ok] / db[ok]
                row.update(
                    rate_language=rate_l, rate_baseline=rate_b, gap=rate_l - rate_b,
                    ci_low=float(np.percentile(reps, 2.5)),
                    ci_high=float(np.percentile(reps, 97.5)),
                    p_bootstrap=bootstrap_p_two_sided(reps.tolist()),
                )  # fmt: skip
            out.append(row)
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["p_holm"] = np.nan
    for _, idx_ in res.groupby(["model", "expected"]).groups.items():
        ps = res.loc[idx_, "p_bootstrap"]
        valid = ps.notna()
        if valid.any():
            res.loc[ps[valid].index, "p_holm"] = holm(ps[valid].tolist())
    return res


# ------------------------------------------------------------------------------ McNemar


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts (binomial test, p0 = 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(sps.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def mcnemar_mde(
    n_pairs: int, discordant_share: float, alpha: float = 0.05, power: float = 0.8
) -> float | None:
    """Smallest detectable difference in success rates for an exact McNemar test.

    Conditional on the discordant count (n_pairs * discordant_share). Returns the minimum
    |difference| such that power >= `power`, or None when no difference can reach it (for
    example with fewer than 6 discordant pairs the exact test can never reach alpha = 0.05).
    """
    nd = round(n_pairs * discordant_share)
    if nd <= 0:
        return None
    reject = [x for x in range(nd + 1) if exact_mcnemar_p(x, nd - x) <= alpha]
    if not reject:
        return None

    def pw(p: float) -> float:
        return float(sum(sps.binom.pmf(x, nd, p) for x in reject))

    if pw(1.0) < power:
        return None
    lo, hi = 0.5, 1.0  # power rises monotonically with p above 0.5
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if pw(mid) >= power else (mid, hi)
    return float((2 * hi - 1) * nd / n_pairs)


def mcnemar_tests(
    d: pd.DataFrame, baseline: str = "en", alpha: float = 0.05, power: float = 0.8
) -> pd.DataFrame:
    """Exact McNemar per (model, variant, condition, expected, language) vs the baseline.

    Pairs are items. An item's outcome in a cell is its success when the majority of its
    (non-unclear) epochs succeed. Reports the discordant counts, not just p; Holm adjusts
    within each (model, expected) family across languages, variants and conditions.
    """
    out: list[dict[str, object]] = []
    for key, g in d.groupby(GROUP, sort=True, dropna=False):
        used = _used(g)
        item = used.groupby(["language", "item_id"])["correct"].agg(
            lambda s: float(s.astype(bool).mean()) > 0.5
        )
        base = (
            item.xs(baseline, level="language")
            if baseline in item.index.get_level_values(0)
            else None
        )
        for lang in sorted(set(g["language"]) - {baseline}):
            row: dict[str, object] = dict(zip(GROUP, key, strict=True))
            row.update(language=lang, baseline=baseline)
            if base is None or lang not in item.index.get_level_values(0):
                row.update(n_pairs=0, both_success=0, baseline_only=0, language_only=0,
                           both_fail=0, p_exact=np.nan, mde_gap=np.nan)  # fmt: skip
                out.append(row)
                continue
            other = item.xs(lang, level="language")
            both = pd.concat([base.rename("b"), other.rename("l")], axis=1, join="inner").astype(
                bool
            )
            a = int((both.b & both.l).sum())
            b = int((both.b & ~both.l).sum())
            c = int((~both.b & both.l).sum())
            dd = int((~both.b & ~both.l).sum())
            n = len(both)
            mde = mcnemar_mde(n, (b + c) / n, alpha, power) if n else None
            row.update(
                n_pairs=n, both_success=a, baseline_only=b, language_only=c, both_fail=dd,
                discordant=b + c, paired_gap=(c - b) / n if n else np.nan,
                p_exact=exact_mcnemar_p(b, c), mde_gap=np.nan if mde is None else mde,
            )  # fmt: skip
            out.append(row)
    res = pd.DataFrame(out)
    if res.empty:
        return res
    res["p_holm"] = np.nan
    for _, idx_ in res.groupby(["model", "expected"]).groups.items():
        ps = res.loc[idx_, "p_exact"]
        valid = ps.notna()
        if valid.any():
            res.loc[ps[valid].index, "p_holm"] = holm(ps[valid].tolist())
    return res


def round_frame(df: pd.DataFrame, nd: int = 6) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(nd)
    return out


__all__ = ["exact_mcnemar_p", "gaps", "mcnemar_mde", "mcnemar_tests", "rates", "round_frame"]
