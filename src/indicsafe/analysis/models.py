"""Regression of refusal on language (and covariates), with item as the unit of replication.

Primary: logistic GEE with an exchangeable working correlation clustered on item_id; the
odds ratios are population-averaged and use robust (sandwich) standard errors. Sensitivity:
a Bayesian logistic mixed model with an item random intercept (variational Bayes; approximate
posterior, no p-values). statsmodels has no frequentist logistic GLMM, which is why GEE is
primary (see docs/DECISIONS.md).

Convergence handling: every fit is wrapped; warnings are recorded and surfaced in `status`,
exceptions become an explicit failed row, a constant outcome is skipped with a reason, and a
coefficient with |log OR| > 8 is flagged as probable separation. Nothing fails silently.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from indicsafe.analysis.metrics import MEASURE

SEPARATION_LOG_OR = 8.0
COLUMNS = [
    "expected", "method", "term", "odds_ratio", "ci_low", "ci_high", "p_value", "status",
    "n_obs", "n_clusters", "formula",
]  # fmt: skip


def _or(x: float) -> float:
    """exp(x) as an odds ratio; NaN when astronomically large (a symptom of separation)."""
    return float(np.exp(x)) if abs(x) <= 50 else float("nan")


def build_formula(d: pd.DataFrame, baseline: str = "en", interaction: bool = False) -> str:
    """refused ~ language (+ category + model + variant + condition, each only if >1 level)."""
    terms = [f"C(language, Treatment(reference='{baseline}'))"]
    for col in ("category", "model", "variant", "condition"):
        if d[col].nunique() > 1:
            terms.append(f"C({col})")
    if interaction and d["category"].nunique() > 1:
        terms.append(f"C(language, Treatment(reference='{baseline}')):C(category)")
    return "refused ~ " + " + ".join(terms)


def _row(expected: str, method: str, status: str, n_obs: int, n_cl: int, formula: str,
         term: str = "", **vals: float) -> dict[str, object]:  # fmt: skip
    r: dict[str, object] = dict.fromkeys(COLUMNS, np.nan)
    r.update(expected=expected, method=method, term=term, status=status, n_obs=n_obs,
             n_clusters=n_cl, formula=formula)  # fmt: skip
    r.update(vals)
    return r


def _fit_gee(sub: pd.DataFrame, formula: str, expected: str) -> list[dict[str, object]]:
    n_obs, n_cl = len(sub), sub["item_id"].nunique()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = smf.gee(
                formula, groups="item_id", data=sub, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            ).fit()  # fmt: skip
            params, ci = res.params, res.conf_int()
            pv, bse = res.pvalues, res.bse
        except Exception as e:  # noqa: BLE001 - reported, never swallowed
            return [_row(expected, "GEE", f"failed: {type(e).__name__}: {e}", n_obs, n_cl, formula)]
    notes = sorted({str(w.message).split("\n")[0][:120] for w in caught})
    rows = []
    for term in params.index:
        sep = (
            abs(params[term]) > SEPARATION_LOG_OR
            or not np.isfinite(bse[term])
            or max(abs(ci.loc[term, 0]), abs(ci.loc[term, 1])) > 50
        )
        status = "ok"
        if notes:
            status = "warnings: " + "; ".join(notes)
        if sep:
            status = (status + "; " if status != "ok" else "") + "possible separation"
        rows.append(
            _row(
                expected,
                "GEE (exchangeable, clustered on item)",
                status,
                n_obs,
                n_cl,
                formula,
                term=str(term),
                odds_ratio=_or(params[term]),
                ci_low=_or(ci.loc[term, 0]),
                ci_high=_or(ci.loc[term, 1]),
                p_value=float("nan") if sep else float(pv[term]),
            )  # fmt: skip
        )
    return rows


def _fit_bayes(
    sub: pd.DataFrame, formula: str, expected: str, seed: int
) -> list[dict[str, object]]:
    n_obs, n_cl = len(sub), sub["item_id"].nunique()
    method = "Bayesian mixed logit, item random intercept (variational Bayes)"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = BinomialBayesMixedGLM.from_formula(
                formula, {"item": "0 + C(item_id)"}, sub
            )  # fmt: skip
            res = model.fit_vb(rng=seed)
        except Exception as e:  # noqa: BLE001
            return [
                _row(expected, method, f"failed: {type(e).__name__}: {e}", n_obs, n_cl, formula)
            ]
    notes = sorted({str(w.message).split("\n")[0][:120] for w in caught})
    status = "ok (approximate posterior; interval = exp(mean +/- 1.96 sd))"
    if notes:
        status = "warnings: " + "; ".join(notes)
    rows = []
    for name, mean, sd in zip(model.exog_names, res.fe_mean, res.fe_sd, strict=True):
        s = status + ("; possible separation" if abs(mean) > SEPARATION_LOG_OR else "")
        rows.append(
            _row(
                expected,
                method,
                s,
                n_obs,
                n_cl,
                formula,
                term=str(name),
                odds_ratio=_or(mean),
                ci_low=_or(mean - 1.96 * sd),
                ci_high=_or(mean + 1.96 * sd),
            )  # fmt: skip
        )
    rows.append(
        _row(
            expected,
            method,
            status,
            n_obs,
            n_cl,
            formula,
            term="item random-effect SD",
            odds_ratio=float(np.exp(res.vcp_mean[0])),
        )  # fmt: skip
    )
    return rows


def fit_models(
    d: pd.DataFrame,
    baseline: str = "en",
    interaction: bool = False,
    bayes: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Fit refused ~ language + covariates separately for harmful and benign items."""
    out: list[dict[str, object]] = []
    for expected in ("refuse", "comply"):
        sub = d[(d["expected"] == expected) & d["refused"].notna()].copy()
        sub["refused"] = sub["refused"].astype(int)
        name = MEASURE[expected]
        if sub.empty:
            out.append(_row(expected, "-", "skipped: no usable rows", 0, 0, ""))
            continue
        formula = build_formula(sub, baseline, interaction)
        if sub["language"].nunique() < 2:
            out.append(_row(expected, "-", "skipped: only one language", len(sub), 0, formula))
            continue
        if sub["refused"].nunique() < 2:
            only = "refused" if sub["refused"].iloc[0] else "complied"
            out.append(_row(expected, "-", f"skipped: every {name} row {only} (no variation)",
                            len(sub), sub["item_id"].nunique(), formula))  # fmt: skip
            continue
        out += _fit_gee(sub, formula, expected)
        if bayes:
            out += _fit_bayes(sub, formula, expected, seed)
    return pd.DataFrame(out, columns=COLUMNS)
