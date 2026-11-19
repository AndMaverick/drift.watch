"""Checking new data against a baseline: per-feature drift, with the sampling noise accounted for."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import measures
from .baseline import (
    OTHER,
    Baseline,
    CategoricalBaseline,
    ColumnBaseline,
    NumericBaseline,
    Row,
    is_null,
    label,
    to_number,
)

ORDER = {"ok": 0, "watch": 1, "alert": 2}


@dataclass(frozen=True)
class Policy:
    """When to raise a flag. The PSI bands are the conventional ones. Everything else makes the verdict robust to small samples."""

    psi_watch: float = 0.1
    psi_alert: float = 0.25
    alpha: float = 0.05  # a PSI above its band only counts when it is also statistically distinguishable from noise
    min_samples: int = 30  # fewer non-null values than this and a feature is reported as insufficient, never alerted
    null_watch: float = 0.05  # absolute change in the null rate
    null_alert: float = 0.20
    rate_watch: float = 0.02  # share of values outside the baseline's range, in unseen categories, or of the wrong type
    rate_alert: float = 0.10
    adjust: bool = True  # Benjamini-Hochberg across features, so 50 features do not mean 2 or 3 false alarms


@dataclass
class FeatureDrift:
    name: str
    kind: str
    n: int
    n_rows: int = 0
    severity: str = "ok"
    insufficient: bool = False
    psi: float = 0.0
    noise_floor: float = 0.0  # the PSI expected from sampling alone at these sample sizes
    p_value: float = 1.0  # that the binned distributions are the same (two-sample chi-square)
    p_adjusted: float = 1.0
    ks: float | None = None
    ks_p: float | None = None
    shift: float | None = None  # Wasserstein distance as a multiple of the baseline's IQR
    js: float = 0.0
    chi2_p: float | None = None
    null_rate: float = 0.0
    null_rate_baseline: float = 0.0
    out_of_range: float = 0.0
    new_category_rate: float = 0.0
    type_mismatch: float = 0.0
    top_shifts: list[tuple[str, float, float]] = field(default_factory=list)  # (where, baseline share, now share)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["top_shifts"] = [list(t) for t in self.top_shifts]
        return d


@dataclass
class DriftReport:
    features: list[FeatureDrift]
    rows: int
    policy: Policy

    @property
    def verdict(self) -> str:
        return max((f.severity for f in self.features), key=ORDER.__getitem__, default="ok")

    def ranked(self) -> list[FeatureDrift]:
        return sorted(self.features, key=lambda f: (-ORDER[f.severity], -f.psi, f.name))

    def drifted(self, at: str = "watch") -> list[FeatureDrift]:
        return [f for f in self.ranked() if ORDER[f.severity] >= ORDER[at]]

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "rows": self.rows, "features": [f.to_dict() for f in self.ranked()]}

    def render(self, limit: int = 20) -> str:
        lines = [f"verdict: {self.verdict.upper()}  ({self.rows} rows, {len(self.features)} features, "
                 f"{len(self.drifted('watch'))} drifted)"]
        lines.append(f"  {'feature':<28}{'status':<8}{'psi':>8}{'noise':>8}{'p(adj)':>9}{'ks':>7}{'shift':>7}  notes")
        for f in self.ranked()[:limit]:
            status = "n/a" if f.insufficient else f.severity
            ks = f"{f.ks:.2f}" if f.ks is not None else "-"
            shift = f"{f.shift:.2f}" if f.shift is not None else "-"
            notes = "; ".join(f.reasons[:2])
            lines.append(f"  {f.name[:27]:<28}{status:<8}{f.psi:>8.3f}{f.noise_floor:>8.3f}{f.p_adjusted:>9.3f}{ks:>7}{shift:>7}  {notes}")
        if len(self.features) > limit:
            lines.append(f"  ... {len(self.features) - limit} more")
        return "\n".join(lines)

    def to_markdown(self, limit: int = 20) -> str:
        out = [f"**Drift verdict: {self.verdict.upper()}** ({self.rows} rows, {len(self.drifted('watch'))} of {len(self.features)} features drifted)",
               "", "| feature | status | PSI | noise floor | adj. p | notes |", "|---|---|---|---|---|---|"]
        for f in self.ranked()[:limit]:
            out.append(f"| {f.name} | {'n/a' if f.insufficient else f.severity} | {f.psi:.3f} | {f.noise_floor:.3f} | {f.p_adjusted:.3f} | {'; '.join(f.reasons[:2])} |")
        return "\n".join(out)


def _bin_label(edges: Sequence[float], i: int) -> str:
    lo = "-inf" if i == 0 else f"{edges[i - 1]:g}"
    hi = "+inf" if i >= len(edges) else f"{edges[i]:g}"
    return f"[{lo}, {hi})"


def _band(value: float, watch: float, alert: float) -> str:
    return "alert" if value >= alert else "watch" if value >= watch else "ok"


def _raise(f: FeatureDrift, severity: str, reason: str) -> None:
    f.reasons.append(reason)
    if ORDER[severity] > ORDER[f.severity]:
        f.severity = severity


def _measure_numeric(base: NumericBaseline, raw: Sequence[Any]) -> FeatureDrift:
    f = FeatureDrift(base.name, "numeric", 0)
    present = [v for v in raw if not is_null(v)]
    numbers = [x for x in (to_number(v) for v in present) if x is not None]
    f.n = len(numbers)
    f.null_rate = 1 - len(present) / len(raw) if raw else 0.0
    f.null_rate_baseline = base.null_rate
    f.type_mismatch = 1 - len(numbers) / len(present) if present else 0.0
    if not numbers:
        return f
    f.out_of_range = sum(x < base.minimum or x > base.maximum for x in numbers) / len(numbers)
    expected = [max(p, measures.EPS) for p in base.proportions]
    counts = measures.bin_counts(numbers, base.edges)
    actual = [max(c / len(numbers), measures.EPS) for c in counts]
    f.psi = measures.psi_from(expected, actual)
    bins = len(expected)
    f.noise_floor = measures.psi_noise_floor(base.n, len(numbers), bins)
    _, _, f.chi2_p = measures.chi_square({str(i): p * base.n for i, p in enumerate(base.proportions)}, {str(i): float(c) for i, c in enumerate(counts)})
    f.p_value = f.chi2_p  # a two-sample chi-square on the bin counts: PSI measures how much, this asks whether it could be chance
    f.ks, f.ks_p = measures.ks_test(base.summary, numbers, base.n, len(numbers))
    f.shift = measures.wasserstein(base.summary, numbers) / base.scale
    f.js = measures.jensen_shannon(expected, actual)
    movers = sorted(range(bins), key=lambda i: -abs(actual[i] - base.proportions[i]))[:3]
    f.top_shifts = [(_bin_label(base.edges, i), base.proportions[i], counts[i] / len(numbers)) for i in movers
                    if abs(actual[i] - base.proportions[i]) > 0.01]
    return f


def _measure_categorical(base: CategoricalBaseline, raw: Sequence[Any]) -> FeatureDrift:
    f = FeatureDrift(base.name, "categorical", 0)
    values = [label(v) for v in raw if not is_null(v)]
    f.n = len(values)
    f.null_rate = 1 - len(values) / len(raw) if raw else 0.0
    f.null_rate_baseline = base.null_rate
    if not values:
        return f
    known = set(base.counts)
    counts: dict[str, int] = {}
    unseen = 0
    new_labels: set[str] = set()
    for v in values:
        key = v
        if v not in known:
            if base.capped:
                key = OTHER  # the baseline cannot say whether this is new or one of the rare categories it folded together
            else:
                unseen += 1
                new_labels.add(v)
                key = f"{v} (new)" if len(new_labels) <= 50 else "(other new)"
        counts[key] = counts.get(key, 0) + 1
    f.new_category_rate = unseen / len(values)
    keys = sorted(set(base.counts) | set(counts))
    e_counts = {k: float(base.counts.get(k, 0)) for k in keys}
    a_counts = {k: float(counts.get(k, 0)) for k in keys}
    e = [max(e_counts[k] / base.n, measures.EPS) for k in keys]
    a = [max(a_counts[k] / len(values), measures.EPS) for k in keys]
    f.psi = measures.psi_from(e, a)
    f.noise_floor = measures.psi_noise_floor(base.n, len(values), len(keys))
    _, _, f.chi2_p = measures.chi_square(e_counts, a_counts)
    f.p_value = f.chi2_p
    f.js = measures.jensen_shannon([e_counts[k] for k in keys], [a_counts[k] for k in keys])
    movers = sorted(keys, key=lambda k: -abs(a_counts[k] / len(values) - e_counts[k] / base.n))[:3]
    f.top_shifts = [(k, e_counts[k] / base.n, a_counts[k] / len(values)) for k in movers
                    if abs(a_counts[k] / len(values) - e_counts[k] / base.n) > 0.01]
    return f


def _judge(f: FeatureDrift, base: ColumnBaseline, policy: Policy, rate_alpha: float) -> None:
    f.insufficient = f.n < policy.min_samples
    if f.insufficient:
        f.reasons.append(f"only {f.n} values (need {policy.min_samples}), not enough to judge")
    else:
        band = _band(f.psi, policy.psi_watch, policy.psi_alert)
        if band != "ok" and f.p_adjusted >= policy.alpha:
            f.reasons.append(f"PSI {f.psi:.3f} is in the {band} band but within sampling noise (p={f.p_adjusted:.2f})")
        elif band != "ok":
            where = f", mostly {f.top_shifts[0][0]}" if f.top_shifts else ""
            _raise(f, band, f"PSI {f.psi:.3f} (noise floor {f.noise_floor:.3f}, p={f.p_adjusted:.3g}){where}")
    # Rate signals are about specific changes, so they apply even when the main comparison is underpowered. Each is tested
    # against the rate the baseline predicts for unchanged data, with the level split across features (Bonferroni).
    delta = abs(f.null_rate - f.null_rate_baseline)
    if delta >= policy.null_watch and f.n_rows:
        x1, x2 = round(f.null_rate_baseline * base.rows), round(f.null_rate * f.n_rows)
        if measures.two_proportion_p(x1, base.rows, x2, f.n_rows) < rate_alpha:
            _raise(f, _band(delta, policy.null_watch, policy.null_alert), f"null rate {f.null_rate_baseline:.1%} -> {f.null_rate:.1%}")
    expected_new = max(sum(1 for c in base.counts.values() if c == 1) / base.n, 1 / (base.n + 1)) if isinstance(base, CategoricalBaseline) else 0.0
    k_out = round(f.out_of_range * f.n)
    if f.out_of_range >= policy.rate_watch and measures.range_exceed_p(k_out, base.n, f.n) < rate_alpha:
        _raise(f, _band(f.out_of_range, policy.rate_watch, policy.rate_alert), f"{f.out_of_range:.1%} outside the baseline range")
    for value, what, p0 in ((f.new_category_rate, "in unseen categories", expected_new),
                            (f.type_mismatch, "not numbers", 1 / (base.n + 1))):
        if value >= policy.rate_watch and measures.binomial_sf(round(value * f.n), f.n, p0) < rate_alpha:
            _raise(f, _band(value, policy.rate_watch, policy.rate_alert), f"{value:.1%} {what}")

def assess(base: ColumnBaseline, raw: Sequence[Any], policy: Policy | None = None, p_adjusted: float | None = None) -> FeatureDrift:
    """Judge one column's new values against its baseline. ``p_adjusted`` lets a caller supply a multiplicity-adjusted p-value."""
    policy = policy or Policy()
    f = _measure_numeric(base, raw) if isinstance(base, NumericBaseline) else _measure_categorical(base, raw)
    f.n_rows = len(raw)
    f.p_adjusted = f.p_value if p_adjusted is None else p_adjusted
    _judge(f, base, policy, policy.alpha)
    return f


def check(baseline: Baseline, rows: Iterable[Row], policy: Policy | None = None) -> DriftReport:
    """Compare ``rows`` with the baseline, feature by feature. A column the baseline watches but the rows lack entirely is
    an alert: an upstream field that disappeared is the most common real drift."""
    policy = policy or Policy()
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to check")
    data = baseline.extract(rows)
    feats: list[FeatureDrift] = []
    for name, col in baseline.columns.items():
        values = data[name]
        f = _measure_numeric(col, values) if isinstance(col, NumericBaseline) else _measure_categorical(col, values)
        f.n_rows = len(values)
        feats.append(f)
    judged = [f for f in feats if not all(is_null(v) for v in data[f.name])]
    adjusted = measures.bh_adjust([f.p_value for f in judged]) if policy.adjust and judged else [f.p_value for f in judged]
    adj = {f.name: p for f, p in zip(judged, adjusted, strict=True)}
    for f in feats:
        col = baseline.columns[f.name]
        f.p_adjusted = adj.get(f.name, 1.0)
        if f.name not in adj and col.null_rate < 0.5:
            _raise(f, "alert", f"column is missing or empty in the new data (null rate {col.null_rate:.1%} in the baseline)")
            f.insufficient = False
            continue
        _judge(f, col, policy, policy.alpha / max(len(feats), 1))
    return DriftReport(feats, len(rows), policy)
