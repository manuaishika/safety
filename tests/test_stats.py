from __future__ import annotations

import itertools
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from indicsafe.stats import (
    bootstrap_p_two_sided,
    cluster_bootstrap,
    cohen_kappa,
    fleiss_kappa,
    holm,
    krippendorff_alpha_nominal,
    percentile,
    rogan_gladen,
    wilson_interval,
)


def test_wilson_known_values() -> None:
    lo, hi = wilson_interval(8, 10)  # textbook: (0.4902, 0.9433)
    assert lo == pytest.approx(0.4902, abs=1e-3) and hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2


def test_cohen_kappa_textbook_2x2() -> None:
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa = 0.4
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_cohen_kappa_edges() -> None:
    assert cohen_kappa([], []) is None
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0  # constant agreement
    assert cohen_kappa(["a", "a"], ["a", "b"]) == pytest.approx(0.0)
    assert cohen_kappa(["a", "b", "a", "b"], ["b", "a", "b", "a"]) == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])


def test_fleiss_kappa_wikipedia_example() -> None:
    m = [
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ]  # fmt: skip
    assert fleiss_kappa(m) == pytest.approx(0.210, abs=5e-4)
    with pytest.raises(ValueError):
        fleiss_kappa([[2, 0], [1, 0]])


def _alpha_bruteforce(units: list[list[str]]) -> float | None:
    """Independent definition: 1 - D_o / D_e over pairable values, pair by pair."""
    pair_units = [u for u in units if len(u) >= 2]
    values = [v for u in pair_units for v in u]
    n = len(values)
    if n < 2:
        return None
    d_o = (
        sum(sum(a != b for a, b in itertools.permutations(u, 2)) / (len(u) - 1) for u in pair_units)
        / n
    )
    d_e = sum(a != b for a, b in itertools.permutations(values, 2)) / (n * (n - 1))
    return 1.0 if d_e == 0 else 1.0 - d_o / d_e


def test_krippendorff_known_cases() -> None:
    assert krippendorff_alpha_nominal([["a", "a"], ["b", "b"], ["a", "a"]]) == pytest.approx(1.0)
    assert krippendorff_alpha_nominal([["a", "b"], ["b", "a"]]) == pytest.approx(-0.5)
    assert krippendorff_alpha_nominal([["a"], ["b"]]) is None
    # Varying numbers of coders per unit, checked against the independent pairwise definition.
    data = [
        ["1", "1"], ["2", "2", "2"], ["3", "3", "3"], ["3", "3", "3"], ["2", "2", "2"],
        ["1", "2", "3"], ["4", "4", "4"], ["1", "1", "2"], ["2", "2", "2"],
        ["5", "5"], ["1", "1"], ["3", "3"],
    ]  # fmt: skip
    got = krippendorff_alpha_nominal(data)
    assert got == pytest.approx(_alpha_bruteforce(data))


@given(
    st.lists(
        st.lists(st.sampled_from(list("abc")), min_size=1, max_size=5), min_size=1, max_size=12
    )
)
def test_krippendorff_matches_bruteforce(units: list[list[str]]) -> None:
    got, want = krippendorff_alpha_nominal(units), _alpha_bruteforce(units)
    assert (got is None) == (want is None)
    if got is not None and want is not None:
        assert got == pytest.approx(want, abs=1e-9)


def test_holm() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
    assert holm([0.2, 0.9]) == pytest.approx([0.4, 0.9])
    assert all(a <= 1 for a in holm([0.6, 0.7, 0.9]))


def test_percentile_and_bootstrap_are_deterministic() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == 3.0
    assert percentile([0.0, 10.0], 0.25) == 2.5
    clusters = {f"c{i}": [float(i % 2)] for i in range(20)}

    def mean(rows: list[float]) -> float:
        return sum(rows) / len(rows)

    a = cluster_bootstrap(clusters, mean, n_boot=300, seed=1)
    assert a == cluster_bootstrap(clusters, mean, n_boot=300, seed=1)
    lo, hi, reps = a
    assert lo is not None and hi is not None and lo < 0.5 < hi and len(reps) == 300
    assert cluster_bootstrap({}, mean)[0] is None


def test_bootstrap_known_effect_is_detected_and_null_is_not() -> None:
    # Known effect: every cluster has value 1 -> CI around 1, null 0 rejected.
    ones = {f"c{i}": [1.0, 1.0] for i in range(15)}

    def mean(rows: list[float]) -> float:
        return sum(rows) / len(rows)

    _, _, reps = cluster_bootstrap(ones, mean, n_boot=500)
    assert bootstrap_p_two_sided(reps, 0.0) == 0.0
    mixed = {f"c{i}": [float(i % 2)] for i in range(40)}
    _, _, reps = cluster_bootstrap(mixed, mean, n_boot=500)
    assert bootstrap_p_two_sided(reps, 0.5) > 0.5


def test_rogan_gladen() -> None:
    # true prevalence 0.6, Se 0.9, Sp 0.8 -> observed 0.9*0.6 + 0.2*0.4 = 0.62
    assert rogan_gladen(0.62, 0.9, 0.8) == pytest.approx(0.6)
    assert rogan_gladen(0.5, 0.5, 0.5) is None  # uninformative grader
    assert rogan_gladen(0.99, 0.9, 0.9) == 1.0 and rogan_gladen(0.0, 0.9, 0.9) == 0.0
    assert not math.isnan(rogan_gladen(0.3, 0.95, 0.95) or 0.0)
