from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicsafe.analysis.models import build_formula, fit_models
from indicsafe.analysis.tidy import Policy, apply_policy

LANG_TERM = "C(language, Treatment(reference='en'))[T.hi]"


def simulate(n_items: int = 300, b: float = 1.5, sd: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Harmful-class rows from a logistic model with a per-item random intercept."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        u = rng.normal(0, sd)
        for lang, eff in (("en", 0.0), ("hi", b)):
            p = 1 / (1 + np.exp(-(-1 + eff + u)))
            rows.append(
                {"model": "m", "variant": "identity", "condition": "none", "category": f"c{i % 3}",
                 "review_status": "unreviewed", "epoch": 1, "language": lang, "item_id": f"i{i}",
                 "expected": "refuse", "label": "REFUSE" if rng.random() < p else "COMPLY"}
            )  # fmt: skip
    return apply_policy(pd.DataFrame(rows), Policy())


def marginal_or(b: float, sd: float) -> float:
    """Population-averaged OR implied by the item-level model (Gauss-Hermite integration)."""
    x, w = np.polynomial.hermite_e.hermegauss(60)
    w = w / w.sum()

    def p(eff: float) -> float:
        return float((w / (1 + np.exp(-(-1 + eff + sd * x)))).sum())

    return (p(b) / (1 - p(b))) / (p(0) / (1 - p(0)))


def test_gee_recovers_the_population_averaged_odds_ratio() -> None:
    res = fit_models(simulate(), bayes=False)
    r = res[(res.expected == "refuse") & (res.term == LANG_TERM)].iloc[0]
    target = marginal_or(1.5, 1.0)
    assert r.ci_low < target < r.ci_high and r.p_value < 1e-6 and r.status == "ok"
    assert target < np.exp(1.5)  # population-averaged OR is attenuated vs item-conditional OR


def test_mixed_model_recovers_the_item_conditional_odds_ratio() -> None:
    res = fit_models(simulate(), bayes=True)
    r = res[(res.method.str.startswith("Bayesian")) & (res.term == LANG_TERM)].iloc[0]
    assert r.ci_low < np.exp(1.5) < r.ci_high
    sd = res[res.term == "item random-effect SD"].iloc[0].odds_ratio
    assert 0.7 < sd < 1.3  # true item SD is 1.0


def test_null_effect_is_not_flagged() -> None:
    r = fit_models(simulate(b=0.0, seed=3), bayes=False)
    r = r[r.term == LANG_TERM].iloc[0]
    assert r.ci_low < 1 < r.ci_high and r.p_value > 0.05


def test_no_outcome_variation_and_single_language_are_skipped_with_reasons() -> None:
    d = simulate(40)
    allref = d.assign(refused=True, label="REFUSE")
    res = fit_models(allref, bayes=False)
    assert any("no variation" in s for s in res.status) and (res.method == "-").any()
    one = d[d.language == "en"]
    assert any("only one language" in s for s in fit_models(one, bayes=False).status)
    assert (fit_models(d.iloc[0:0], bayes=False).status == "skipped: no usable rows").all()


def test_perfect_separation_is_flagged_not_silent() -> None:
    d = simulate(60)
    d.loc[d.language == "hi", "refused"] = True
    d.loc[d.language == "en", "refused"] = False
    res = fit_models(d, bayes=False)
    gee = res[res.method.str.startswith("GEE")]
    assert not (gee.status == "ok").all()  # failed, warned or flagged as separation


def test_failed_fit_becomes_an_explicit_row(monkeypatch: pytest.MonkeyPatch) -> None:
    import statsmodels.formula.api as smf

    def boom(*args: object, **kwargs: object) -> object:
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(smf, "gee", boom)
    res = fit_models(simulate(30), bayes=False)
    row = res[res.expected == "refuse"].iloc[0]
    assert row.status == "failed: LinAlgError: singular matrix" and np.isnan(row.odds_ratio)


def test_formula_only_includes_factors_with_levels() -> None:
    d = simulate(10)
    assert build_formula(d) == "refused ~ C(language, Treatment(reference='en')) + C(category)"
    d2 = d.assign(model=d["model"].where(d.index % 2 == 0, "m2"))
    assert "C(model)" in build_formula(d2)
    assert ":C(category)" in build_formula(d, interaction=True)
    assert ":C(category)" not in build_formula(d.assign(category="c"), interaction=True)


def test_covariates_and_all_columns_present() -> None:
    d = simulate(60)
    d = pd.concat([d.assign(model="a"), d.assign(model="b")], ignore_index=True)
    res = fit_models(d, bayes=False)
    assert any("C(model)" in t for t in res.term)
    assert {"odds_ratio", "ci_low", "ci_high", "p_value", "status", "n_obs", "n_clusters"} <= set(
        res.columns
    )
    assert (res[res.expected == "refuse"].n_clusters == 60).all()


def test_fits_are_deterministic_given_the_seed() -> None:
    d = simulate(80)
    a, b = fit_models(d, bayes=True, seed=5), fit_models(d, bayes=True, seed=5)
    pd.testing.assert_frame_equal(a, b)
