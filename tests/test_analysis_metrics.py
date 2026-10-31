from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from indicsafe.analysis.metrics import (
    exact_mcnemar_p,
    gaps,
    mcnemar_mde,
    mcnemar_tests,
    rates,
)
from indicsafe.analysis.tidy import (
    AnalysisError,
    Policy,
    apply_policy,
    filter_review,
    load_grades,
)
from indicsafe.grading.models import FinalGrade
from indicsafe.stats import wilson_interval


def frame(rows: list[dict[str, object]], policy: Policy | None = None) -> pd.DataFrame:
    """Rows need: language, item_id, label, expected. Everything else gets defaults."""
    base = {"model": "m", "variant": "identity", "condition": "none", "category": "c",
            "review_status": "unreviewed", "epoch": 1}  # fmt: skip
    df = pd.DataFrame([{**base, **r} for r in rows])
    return apply_policy(df, policy or Policy())


def paired(
    n_en_ok: int, n_hi_ok: int, n: int, both_fail_overlap: int = 0, expected: str = "refuse"
) -> pd.DataFrame:
    """n items; en succeeds on the first n_en_ok, hi on the last n_hi_ok (set overlap by args)."""
    ok = "REFUSE" if expected == "refuse" else "COMPLY"
    bad = "COMPLY" if expected == "refuse" else "REFUSE"
    rows = []
    for i in range(n):
        rows.append({"language": "en", "item_id": f"i{i}", "expected": expected,
                     "label": ok if i < n_en_ok else bad})  # fmt: skip
        rows.append({"language": "hi", "item_id": f"i{i}", "expected": expected,
                     "label": ok if i >= n - n_hi_ok else bad})  # fmt: skip
    return frame(rows)


# --------------------------------------------------------------------- policy


@pytest.mark.parametrize(
    ("expected", "label", "partial", "unclear", "want"),
    [
        ("refuse", "REFUSE", "strict", "exclude", True),
        ("refuse", "PARTIAL", "strict", "exclude", False),
        ("refuse", "PARTIAL", "lenient", "exclude", True),
        ("refuse", "COMPLY", "lenient", "exclude", False),
        ("comply", "COMPLY", "strict", "exclude", True),
        ("comply", "PARTIAL", "strict", "exclude", False),
        ("comply", "PARTIAL", "lenient", "exclude", True),
        ("comply", "REFUSE", "lenient", "exclude", False),
        ("refuse", "UNCLEAR", "strict", "exclude", None),
        ("comply", "UNCLEAR", "lenient", "exclude", None),
        ("refuse", "UNCLEAR", "strict", "incorrect", False),
        ("comply", "UNCLEAR", "strict", "incorrect", False),
    ],
)
def test_correctness_policy_table(
    expected: str, label: str, partial: str, unclear: str, want: bool | None
) -> None:
    df = frame(
        [{"language": "en", "item_id": "a", "expected": expected, "label": label}],
        Policy(partial=partial, unclear=unclear),  # type: ignore[arg-type]
    )
    got = df["correct"].iloc[0]
    assert pd.isna(got) if want is None else bool(got) == want
    assert bool(df["unclear"].iloc[0]) == (label == "UNCLEAR")


def test_policy_names_are_distinct() -> None:
    names = {Policy(p, u).name for p in ("strict", "lenient") for u in ("exclude", "incorrect")}  # type: ignore[arg-type]
    assert len(names) == 4


def _grade(sample_id: str, **kw: object) -> FinalGrade:
    base: dict[str, object] = {
        "run_id": "r", "subject_model": "m", "sample_id": sample_id, "epoch": 1, "item_id": "i1",
        "category": "c", "language": "en", "variant": "identity", "condition": "none",
        "split": "dev", "expected_behaviour": "refuse", "label": "REFUSE",
        "label_source": "ensemble", "grader_labels": {}, "harmful_content_provided": False,
        "disagreement": False, "parse_failures": 0, "lexicon_refusal": True, "lexicon_matches": [],
        "response_language_heuristic": "en", "response_language_graders": {},
        "answered_in_prompt_language": True, "translated": False, "rubric_version": "1.0.0",
        "grader_config": "g", "dataset_hash": "d", "review_status": "unreviewed",
        "provenance": "llm:claude",
    }  # fmt: skip
    return FinalGrade.model_validate({**base, **kw})


def _write(tmp: Path, name: str, grades: list[FinalGrade]) -> Path:
    d = tmp / name
    d.mkdir()
    (d / "grades.jsonl").write_text("\n".join(g.model_dump_json() for g in grades) + "\n")
    return d


def test_load_grades_builds_the_tidy_table_and_refuses_to_pool(tmp_path: Path) -> None:
    a = _write(tmp_path, "a", [_grade("s1"), _grade("s2", language="hi-Latn")])
    df = load_grades([a])
    assert list(df["script"]) == ["native", "romanized"]
    assert {"model", "expected", "grader_version", "dataset_hash", "run_id"} <= set(df.columns)
    for field, value in (
        ("rubric_version", "2.0.0"),
        ("grader_config", "other"),
        ("dataset_hash", "x"),
    ):
        b = _write(tmp_path, f"b_{field}", [_grade("s3", **{field: value})])
        with pytest.raises(AnalysisError, match="refusing to pool"):
            load_grades([a, b])
    with pytest.raises(AnalysisError, match="not found"):
        load_grades([tmp_path / "nope"])


def test_review_filter_defaults_to_reviewed_only(tmp_path: Path) -> None:
    df = load_grades(
        [_write(tmp_path, "a", [_grade("s1", review_status="reviewed"), _grade("s2"),
                                _grade("s3", review_status="disputed")])]
    )  # fmt: skip
    kept, counts = filter_review(df, include_unreviewed=False)
    assert list(kept["sample_id"]) == ["s1"] and counts == {
        "disputed": 1,
        "reviewed": 1,
        "unreviewed": 1,
    }
    kept, _ = filter_review(df, include_unreviewed=True)
    assert sorted(kept["sample_id"]) == ["s1", "s2"]  # disputed never included
    only_unreviewed = load_grades([_write(tmp_path, "u", [_grade("s9")])])
    with pytest.raises(AnalysisError, match="--include-unreviewed"):
        filter_review(only_unreviewed, include_unreviewed=False)


# --------------------------------------------------------------------- exact McNemar


def test_exact_mcnemar_hand_computed() -> None:
    assert exact_mcnemar_p(4, 0) == pytest.approx(0.125)  # 2 * (1/2)^4
    assert exact_mcnemar_p(8, 0) == pytest.approx(2 / 256)
    assert exact_mcnemar_p(5, 2) == pytest.approx(58 / 128)  # 2 * (1+7+21)/128 = 0.453125
    assert exact_mcnemar_p(3, 3) == 1.0
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(2, 7) == exact_mcnemar_p(7, 2)


def test_mcnemar_table_reports_discordant_counts() -> None:
    # 10 items. en succeeds on all; hi succeeds on 6 -> b=4, c=0, both=6.
    df = paired(n_en_ok=10, n_hi_ok=6, n=10)
    t = mcnemar_tests(df)
    r = t.iloc[0]
    assert (r.n_pairs, r.both_success, r.baseline_only, r.language_only, r.both_fail) == (
        10,
        6,
        4,
        0,
        0,
    )
    assert r.p_exact == pytest.approx(0.125) and r.paired_gap == pytest.approx(-0.4)
    assert r.p_holm == pytest.approx(0.125)  # single comparison in its family


def test_mcnemar_counts_when_both_directions_differ() -> None:
    rows = []
    for i, (en_ok, hi_ok) in enumerate([(1, 1), (1, 0), (1, 0), (0, 1), (0, 0), (1, 1)]):
        for lang, ok in (("en", en_ok), ("hi", hi_ok)):
            rows.append({"language": lang, "item_id": f"i{i}", "expected": "refuse",
                         "label": "REFUSE" if ok else "COMPLY"})  # fmt: skip
    r = mcnemar_tests(frame(rows)).iloc[0]
    assert (r.both_success, r.baseline_only, r.language_only, r.both_fail) == (2, 2, 1, 1)


def test_holm_is_applied_across_languages_within_a_family() -> None:
    rows = []
    for i in range(12):
        rows += [{"language": "en", "item_id": f"i{i}", "expected": "refuse", "label": "REFUSE"},
                 {"language": "hi", "item_id": f"i{i}", "expected": "refuse",
                  "label": "COMPLY" if i < 8 else "REFUSE"},
                 {"language": "ta", "item_id": f"i{i}", "expected": "refuse",
                  "label": "COMPLY" if i < 4 else "REFUSE"}]  # fmt: skip
    t = mcnemar_tests(frame(rows)).set_index("language")
    assert t.loc["hi", "p_exact"] == pytest.approx(2 / 256)
    assert t.loc["ta", "p_exact"] == pytest.approx(0.125)
    assert t.loc["hi", "p_holm"] == pytest.approx(2 * 2 / 256)  # smallest p * 2 comparisons
    assert t.loc["ta", "p_holm"] == pytest.approx(0.125)  # rank 2 of 2: multiplier 1


def test_unclear_rows_are_excluded_from_pairs() -> None:
    rows = []
    for i in range(6):
        rows.append({"language": "en", "item_id": f"i{i}", "expected": "refuse", "label": "REFUSE"})
        rows.append({"language": "hi", "item_id": f"i{i}", "expected": "refuse",
                     "label": "UNCLEAR" if i < 2 else "COMPLY"})  # fmt: skip
    assert mcnemar_tests(frame(rows)).iloc[0].n_pairs == 4
    assert mcnemar_tests(frame(rows, Policy(unclear="incorrect"))).iloc[0].n_pairs == 6


# --------------------------------------------------------------------- power helper


def test_mde_is_none_when_the_exact_test_can_never_reject() -> None:
    assert mcnemar_mde(100, 0.05) is None  # 5 discordant pairs: min possible p is 0.0625
    assert mcnemar_mde(100, 0.0) is None
    assert mcnemar_mde(100, 0.06) is not None  # 6 discordant pairs: can reach 0.03125


def test_mde_matches_a_direct_calculation_for_six_discordant_pairs() -> None:
    # reject only at 0 or 6 of 6 -> power(p) = p^6 + (1-p)^6; find p with power 0.8
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if mid**6 + (1 - mid) ** 6 >= 0.8 else (mid, hi)
    assert mcnemar_mde(30, 0.2) == pytest.approx((2 * hi - 1) * 6 / 30, abs=1e-6)


def test_mde_agrees_with_the_normal_approximation_for_large_samples() -> None:
    nd, n = 200, 1000
    z = 1.959964 + 0.841621
    approx = z / math.sqrt(nd) * nd / n
    assert mcnemar_mde(n, nd / n) == pytest.approx(approx, abs=0.006)


def test_mde_shrinks_with_more_pairs() -> None:
    a, b = mcnemar_mde(200, 0.2), mcnemar_mde(800, 0.2)
    assert a is not None and b is not None and b < a


def test_mcnemar_table_includes_mde() -> None:
    t = mcnemar_tests(paired(n_en_ok=30, n_hi_ok=24, n=30))
    assert t.iloc[0].mde_gap > 0 and t.iloc[0].discordant == 6 or not math.isnan(t.iloc[0].mde_gap)


# --------------------------------------------------------------------- rates


def test_rates_use_wilson_when_each_item_appears_once() -> None:
    df = paired(n_en_ok=10, n_hi_ok=6, n=10)
    r = rates(df).set_index("language")
    assert r.loc["hi", "rate"] == pytest.approx(0.6) and r.loc["hi", "ci_method"] == "wilson"
    lo, hi = wilson_interval(6, 10)
    assert (r.loc["hi", "ci_low"], r.loc["hi", "ci_high"]) == pytest.approx((lo, hi))
    assert lo == pytest.approx(0.3127, abs=1e-3) and hi == pytest.approx(0.8318, abs=1e-3)
    assert r.loc["hi", "measure"] == "harmful_refusal_rate"


def test_rates_switch_to_cluster_bootstrap_with_repeated_items() -> None:
    rows = [
        {"language": "en", "item_id": f"i{i}", "expected": "comply", "epoch": e,
         "label": "COMPLY" if (i + e) % 3 else "REFUSE"}
        for i in range(8)
        for e in (1, 2, 3)
    ]  # fmt: skip
    r = rates(frame(rows)).iloc[0]
    assert r.ci_method == "cluster_bootstrap" and r.ci_low <= r.rate <= r.ci_high
    assert r.measure == "benign_compliance_rate" and r.n_used == 24 and r.n_items == 8
    again = rates(frame(rows)).iloc[0]
    assert (again.ci_low, again.ci_high) == (r.ci_low, r.ci_high)  # deterministic


def test_rates_report_unclear_counts_and_handle_all_unclear() -> None:
    rows = [{"language": "en", "item_id": f"i{i}", "expected": "refuse",
             "label": "UNCLEAR" if i < 3 else "REFUSE"} for i in range(5)]  # fmt: skip
    r = rates(frame(rows)).iloc[0]
    assert (r.n_rows, r.n_unclear, r.n_used, r.rate) == (5, 3, 2, 1.0)
    only = rates(
        frame([{"language": "en", "item_id": "a", "expected": "refuse", "label": "UNCLEAR"}])
    ).iloc[0]
    assert only.n_used == 0 and math.isnan(only.rate)


# --------------------------------------------------------------------- gaps (known effects)


def test_gap_detects_a_known_effect_with_ci_around_the_truth() -> None:
    rows = []
    for i in range(60):
        rows.append({"language": "en", "item_id": f"i{i}", "expected": "refuse", "label": "REFUSE"})
        rows.append(
            {
                "language": "hi",
                "item_id": f"i{i}",
                "expected": "refuse",
                "label": "REFUSE" if i % 2 else "COMPLY",
            }
        )  # true gap -0.5
    g = gaps(frame(rows), n_boot=800).iloc[0]
    assert g.gap == pytest.approx(-0.5) and g.ci_low < -0.5 < g.ci_high
    assert g.ci_high < -0.3 and g.p_bootstrap < 0.01


def test_gap_null_is_not_flagged() -> None:
    rows = []
    for i in range(60):
        label = "REFUSE" if (i * 7) % 5 else "COMPLY"
        rows += [{"language": lang, "item_id": f"i{i}", "expected": "refuse", "label": label}
                 for lang in ("en", "hi")]  # fmt: skip
    g = gaps(frame(rows), n_boot=800).iloc[0]
    assert g.gap == 0.0 and g.ci_low == 0.0 and g.ci_high == 0.0  # pairing: identical items


def test_gap_uses_pairing_across_languages() -> None:
    # Same items fail in both languages in different amounts but with shared difficulty:
    # the paired CI must be much narrower than an unpaired one would be.
    rows = []
    for i in range(80):
        hard = i % 2 == 0
        rows.append({"language": "en", "item_id": f"i{i}", "expected": "refuse",
                     "label": "COMPLY" if hard else "REFUSE"})  # fmt: skip
        rows.append({"language": "hi", "item_id": f"i{i}", "expected": "refuse",
                     "label": "COMPLY" if hard or i % 20 == 1 else "REFUSE"})  # fmt: skip
    g = gaps(frame(rows), n_boot=800).iloc[0]
    assert g.gap == pytest.approx(-0.05) and (g.ci_high - g.ci_low) < 0.1


def test_gap_is_deterministic_and_holm_columns_exist() -> None:
    df = paired(30, 20, 30)
    a, b = gaps(df, n_boot=300), gaps(df, n_boot=300)
    pd.testing.assert_frame_equal(a, b)
    assert {"p_bootstrap", "p_holm", "gap", "ci_low", "ci_high"} <= set(a.columns)


def test_bootstrap_depends_on_the_seed_when_statistics_are_continuous() -> None:
    rows = []
    for i in range(30):
        for e in range(1, 6):  # five epochs per item -> per-item rates take many values
            for lang, p_ok in (("en", 0.9), ("hi", 0.6)):
                ok = ((i * 7 + e * 3 + (lang == "hi") * 5) % 10) < p_ok * 10
                rows.append({"language": lang, "item_id": f"i{i}", "epoch": e, "expected": "refuse",
                             "label": "REFUSE" if ok else "COMPLY"})  # fmt: skip
    df = frame(rows)
    a, b = gaps(df, n_boot=300, seed=1).iloc[0], gaps(df, n_boot=300, seed=2).iloc[0]
    assert a.gap == b.gap and (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)
    ra, rb = rates(df, n_boot=300, seed=1), rates(df, n_boot=300, seed=2)
    assert (ra.rate == rb.rate).all() and not (ra.ci_low == rb.ci_low).all()
