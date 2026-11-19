"""Distribution measures: pure functions over lists of numbers or mappings of counts. No state, no dependencies."""
from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence

EPS = 1e-4  # floor on a bin's share so the log term in PSI stays finite


# ---------------------------------------------------------------------------------------------------------------- binning

def quantile_edges(values: Sequence[float], bins: int) -> list[float]:
    """Bin edges at the baseline's quantiles. Tied values collapse edges, so a skewed or discrete column gets fewer bins."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    ordered = sorted(values)
    return sorted({ordered[min(len(ordered) - 1, int(i * len(ordered) / bins))] for i in range(1, bins)})


def bin_counts(values: Sequence[float], edges: Sequence[float]) -> list[int]:
    counts = [0] * (len(edges) + 1)
    for value in values:
        counts[bisect_right(edges, value)] += 1
    return counts


def proportions(values: Sequence[float], edges: Sequence[float], floor: float = EPS) -> list[float]:
    counts = bin_counts(values, edges)
    total = len(values)
    return [max(c / total, floor) for c in counts]


def psi_from(expected: Sequence[float], actual: Sequence[float]) -> float:
    """PSI of two aligned lists of proportions: the sum of (a - e) * ln(a / e)."""
    return sum((a - e) * math.log(a / e) for e, a in zip(expected, actual, strict=True))


def psi(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """Population Stability Index of ``actual`` against ``expected``, using the quantile bins of ``expected``."""
    if not expected or not actual:
        raise ValueError("both samples must be non-empty")
    edges = quantile_edges(expected, bins)
    return psi_from(proportions(expected, edges), proportions(actual, edges))


def categorical_psi(expected: Mapping[str, float], actual: Mapping[str, float]) -> float:
    keys = sorted(set(expected) | set(actual))
    e_total, a_total = sum(expected.values()), sum(actual.values())
    if e_total <= 0 or a_total <= 0:
        raise ValueError("both count tables must be non-empty")
    e = [max(expected.get(k, 0) / e_total, EPS) for k in keys]
    a = [max(actual.get(k, 0) / a_total, EPS) for k in keys]
    return psi_from(e, a)


# -------------------------------------------------------------------------------------------------- special functions

def _gamma_q(a: float, x: float) -> float:
    """Regularised upper incomplete gamma function Q(a, x) (series below a + 1, continued fraction above)."""
    if x <= 0:
        return 1.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(2000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(log_front))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = d if abs(d) > tiny else tiny
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return min(1.0, math.exp(log_front) * h)


def chi2_sf(x: float, df: int) -> float:
    """Survival function of the chi-square distribution: P(X >= x)."""
    if df < 1:
        raise ValueError("df must be at least 1")
    return _gamma_q(df / 2, x / 2)


def kolmogorov_sf(lam: float) -> float:
    """Survival function of the Kolmogorov distribution: 2 * sum (-1)^(j-1) exp(-2 j^2 lam^2)."""
    if lam < 0.2:
        return 1.0
    total = 0.0
    for j in range(1, 200):
        term = 2 * (-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam)
        total += term
        if abs(term) < 1e-16:
            break
    return min(1.0, max(0.0, total))


# ------------------------------------------------------------------------------------------------- numeric comparisons

def ks_stat(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: the largest gap between the empirical CDFs."""
    left, right = sorted(a), sorted(b)
    if not left or not right:
        raise ValueError("both samples must be non-empty")
    worst = 0.0
    for point in sorted(set(left) | set(right)):
        gap = abs(bisect_right(left, point) / len(left) - bisect_right(right, point) / len(right))
        worst = max(worst, gap)
    return worst


def ks_test(a: Sequence[float], b: Sequence[float], n_a: int | None = None, n_b: int | None = None) -> tuple[float, float]:
    """(statistic, p-value) for the two-sample KS test. The p-value is asymptotic with the usual small-sample correction.
    ``n_a`` and ``n_b`` override the sample sizes used for the p-value when one side is a summary of a larger sample."""
    d = ks_stat(a, b)
    n, m = n_a or len(a), n_b or len(b)
    ne = n * m / (n + m)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    return d, kolmogorov_sf(lam)


def wasserstein(a: Sequence[float], b: Sequence[float]) -> float:
    """First Wasserstein (earth mover's) distance between two one-dimensional samples, in the units of the data."""
    left, right = sorted(a), sorted(b)
    if not left or not right:
        raise ValueError("both samples must be non-empty")
    points = sorted(set(left) | set(right))
    area = 0.0
    for lo, hi in zip(points, points[1:], strict=False):
        gap = abs(bisect_right(left, lo) / len(left) - bisect_right(right, lo) / len(right))
        area += gap * (hi - lo)
    return area


# --------------------------------------------------------------------------------------------- categorical comparisons

def jensen_shannon(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence in bits: 0 for identical distributions, 1 for disjoint ones. Symmetric and always finite."""
    sp, sq = sum(p), sum(q)
    if sp <= 0 or sq <= 0 or len(p) != len(q):
        raise ValueError("need two aligned, non-empty distributions")
    p = [x / sp for x in p]
    q = [x / sq for x in q]
    m = [(x + y) / 2 for x, y in zip(p, q, strict=True)]

    def kl(u: Sequence[float], v: Sequence[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(u, v, strict=True) if x > 0)

    return max(0.0, (kl(p, m) + kl(q, m)) / 2)


def chi_square(expected: Mapping[str, float], actual: Mapping[str, float]) -> tuple[float, int, float]:
    """Two-sample chi-square test that two count tables share a distribution. Returns (statistic, df, p-value).
    Reliable when most expected cell counts are at least 5."""
    keys = [k for k in sorted(set(expected) | set(actual)) if expected.get(k, 0) + actual.get(k, 0) > 0]
    n1, n2 = sum(expected.values()), sum(actual.values())
    if n1 <= 0 or n2 <= 0:
        raise ValueError("both count tables must be non-empty")
    if len(keys) < 2:
        return 0.0, 1, 1.0
    stat = 0.0
    for k in keys:
        o1, o2 = expected.get(k, 0), actual.get(k, 0)
        total = o1 + o2
        e1, e2 = total * n1 / (n1 + n2), total * n2 / (n1 + n2)
        stat += (o1 - e1) ** 2 / e1 + (o2 - e2) ** 2 / e2
    df = len(keys) - 1
    return stat, df, chi2_sf(stat, df)


# ------------------------------------------------------------------------------------------- sampling-noise calibration

def psi_noise_floor(n_expected: int, n_actual: int, bins: int = 10) -> float:
    """The PSI you should expect from two samples of the *same* distribution, from sampling alone: (bins - 1)(1/n + 1/m).
    A PSI near this number is not drift. It is why 0.1 means little on 50 rows and a lot on 50,000."""
    return (bins - 1) * (1 / n_expected + 1 / n_actual)


def psi_pvalue(value: float, n_expected: int, n_actual: int, bins: int = 10) -> float:
    """Probability of a PSI at least this large if nothing changed. Uses PSI ~ (1/n + 1/m) * chi-square(bins - 1), which
    holds when bins hold enough rows (about 5 each)."""
    if bins < 2:
        return 1.0
    return chi2_sf(value / (1 / n_expected + 1 / n_actual), bins - 1)


def binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact. Used to ask whether a rate is higher than the baseline predicts."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    log_c = math.lgamma(n + 1)
    total = 0.0
    mode = n * p
    for i in range(k, n + 1):
        term = math.exp(log_c - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
        total += term
        if i > mode and term < 1e-17 * max(total, 1e-300):
            break
    return min(1.0, total)


def range_exceed_p(k: int, n_baseline: int, n_new: int) -> float:
    """Upper bound on the probability that at least ``k`` of ``n_new`` values land outside the min-max range of ``n_baseline``
    values drawn from the same distribution. Exact for one tail (the k largest of the pooled values would all have to be new),
    doubled for two tails. Expect a few points outside the range even with no drift, and more on a small baseline."""
    if k <= 0:
        return 1.0
    half = (k + 1) // 2
    if half > n_new:
        return 0.0
    p = 1.0
    for i in range(half):
        p *= (n_new - i) / (n_baseline + n_new - i)
    return min(1.0, 2 * p)


def two_proportion_p(x1: int, n1: int, x2: int, n2: int) -> float:
    """Two-sided p-value that two rates (x1 of n1, x2 of n2) are equal, by the pooled z-test."""
    if n1 <= 0 or n2 <= 0:
        return 1.0
    pooled = (x1 + x2) / (n1 + n2)
    var = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if var <= 0:
        return 1.0
    z = (x1 / n1 - x2 / n2) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def bh_adjust(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, so watching many features at once does not raise a false alarm per feature."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, pvalues[i] * n / rank)
        adjusted[i] = running
    return adjusted


# ---------------------------------------------------------------------------------------------------------------- bands

def status(value: float, watch: float = 0.1, alert: float = 0.25) -> str:
    """The conventional PSI bands: under 0.1 stable, under 0.25 moderate, above that major."""
    if value < watch:
        return "stable"
    if value < alert:
        return "moderate"
    return "major"


def report(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> dict[str, object]:
    p = psi(expected, actual, bins)
    return {"psi": round(p, 4), "ks": round(ks_stat(expected, actual), 4), "status": status(p)}
