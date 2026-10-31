"""Small, dependency-free statistics used by validation (and reused by the analysis in M6).

Everything here is deterministic given its inputs and seed.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from typing import TypeVar

T = TypeVar("T")


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Unweighted Cohen's kappa; None when undefined (no rows, or chance agreement is 1)."""
    if len(a) != len(b):
        raise ValueError("rating lists differ in length")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb.get(c, 0) for c in ca) / (n * n)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1 - pe)


def fleiss_kappa(counts: Sequence[Sequence[int]]) -> float | None:
    """Fleiss' kappa from an items x categories count matrix (same raters per item)."""
    if not counts:
        return None
    n = sum(counts[0])
    if n < 2 or any(sum(row) != n for row in counts):
        raise ValueError("Fleiss' kappa needs the same number (>=2) of raters for every item")
    big_n = len(counts)
    k = len(counts[0])
    p_j = [sum(row[j] for row in counts) / (big_n * n) for j in range(k)]
    p_i = [(sum(c * c for c in row) - n) / (n * (n - 1)) for row in counts]
    pbar = sum(p_i) / big_n
    pe = sum(p * p for p in p_j)
    if pe >= 1.0:
        return 1.0 if pbar == 1.0 else None
    return (pbar - pe) / (1 - pe)


def krippendorff_alpha_nominal(units: Sequence[Sequence[Hashable]]) -> float | None:
    """Krippendorff's alpha for nominal data; units with fewer than 2 ratings are ignored.

    Handles a varying number of raters per unit.
    """
    o: dict[tuple[Hashable, Hashable], float] = {}
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        for i, c in enumerate(vals):
            for j, k in enumerate(vals):
                if i != j:
                    o[(c, k)] = o.get((c, k), 0.0) + 1 / (m - 1)
    n_c: dict[Hashable, float] = {}
    for (c, _k), v in o.items():
        n_c[c] = n_c.get(c, 0.0) + v
    n = sum(n_c.values())
    if n < 2:
        return None
    observed = sum(v for (c, k), v in o.items() if c != k)
    expected = sum(n_c[c] * n_c[k] for c in n_c for k in n_c if c != k)
    if expected == 0:
        return 1.0
    return 1.0 - (n - 1) * observed / expected


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def percentile(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("no values")
    pos = q * (len(sorted_vals) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def cluster_bootstrap(
    clusters: dict[str, list[T]],
    stat: Callable[[list[T]], float | None],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float | None, float | None, list[float]]:
    """Percentile 95% CI of ``stat`` resampling whole clusters (e.g. items) with replacement.

    Returns (lo, hi, replicates). Replicates where ``stat`` is undefined are dropped; if more
    than half are undefined the interval is None.
    """
    keys = sorted(clusters)
    if not keys:
        return None, None, []
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(n_boot):
        rows: list[T] = []
        for k in rng.choices(keys, k=len(keys)):
            rows.extend(clusters[k])
        v = stat(rows)
        if v is not None and not math.isnan(v):
            reps.append(v)
    if len(reps) < n_boot / 2:
        return None, None, reps
    reps.sort()
    return percentile(reps, 0.025), percentile(reps, 0.975), reps


def bootstrap_p_two_sided(reps: Sequence[float], null: float = 0.0) -> float:
    """Two-sided bootstrap p-value for 'statistic == null' (percentile-inversion style)."""
    if not reps:
        return 1.0
    below = sum(r <= null for r in reps) / len(reps)
    above = sum(r >= null for r in reps) / len(reps)
    return min(1.0, 2 * min(below, above))


def rogan_gladen(p_obs: float, sensitivity: float, specificity: float) -> float | None:
    """Misclassification-corrected prevalence, clipped to [0, 1].

    None when the classifier is uninformative (sensitivity + specificity - 1 <= 0.05).
    """
    denom = sensitivity + specificity - 1
    if denom <= 0.05:
        return None
    return min(1.0, max(0.0, (p_obs + specificity - 1) / denom))
