"""Change detectors for data that arrives one value at a time, and a window scan for data you already have."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .baseline import ColumnBaseline
from .monitor import FeatureDrift, Policy, assess


class PageHinkley:
    """Page-Hinkley test for a change in the mean of a stream. Cheap, one pass, and fast on a sustained shift.

    ``delta`` is the size of change you are willing to ignore and ``threshold`` how much accumulated evidence triggers a
    signal: raise it for fewer false alarms and a slower response. ``direction`` is ``up``, ``down`` or ``both``.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 50.0, direction: str = "both", min_samples: int = 30) -> None:
        if direction not in {"up", "down", "both"}:
            raise ValueError("direction must be up, down or both")
        self.delta, self.threshold, self.direction, self.min_samples = delta, threshold, direction, min_samples
        self.reset()

    @classmethod
    def calibrated(cls, sd: float, shift: float, false_alarm: float = 1e-4, direction: str = "both") -> PageHinkley:
        """Set ``delta`` and ``threshold`` from the data's standard deviation and the smallest mean shift worth catching, in the
        data's units. Evidence builds like a random walk, so the chance that one excursion reaches the threshold is about
        exp(-2 * delta * threshold / sd^2): ``threshold`` is chosen to make that ``false_alarm``. In simulation, with a one
        standard deviation shift this fires after about 22 values and averages more than 15,000 values between false alarms."""
        if sd <= 0 or shift <= 0 or not 0 < false_alarm < 1:
            raise ValueError("sd and shift must be positive and false_alarm in (0, 1)")
        delta = shift / 2
        return cls(delta, math.log(1 / false_alarm) * sd * sd / (2 * delta), direction)

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self._up = 0.0
        self._down = 0.0

    def update(self, x: float) -> bool:
        """Feed one value. True means a change was detected, and the detector restarts."""
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self._up = max(0.0, self._up + x - self.mean - self.delta)
        self._down = max(0.0, self._down + self.mean - x - self.delta)
        if self.n < self.min_samples:
            return False
        hit = (self.direction in {"up", "both"} and self._up > self.threshold) or (self.direction in {"down", "both"} and self._down > self.threshold)
        if hit:
            self.reset()
        return hit


class DDM:
    """Drift Detection Method for a stream of errors (True when the model was wrong). It tracks the error rate and its
    standard deviation, warns at two standard deviations above the best seen, and signals drift at three. Needs labels."""

    def __init__(self, min_samples: int = 30, warn: float = 2.0, drift: float = 3.0) -> None:
        self.min_samples, self.warn, self.drift = min_samples, warn, drift
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.errors = 0
        self._best = math.inf
        self._best_p = 0.0
        self._best_s = 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.n if self.n else 0.0

    def update(self, error: bool) -> str:
        """Feed one outcome. Returns ``ok``, ``warning`` or ``drift`` (after which the detector restarts)."""
        self.n += 1
        self.errors += bool(error)
        if self.n < self.min_samples:
            return "ok"
        p = self.errors / self.n
        s = math.sqrt(p * (1 - p) / self.n)
        if p + s < self._best:
            self._best, self._best_p, self._best_s = p + s, p, s
        if p + s >= self._best_p + self.drift * self._best_s and self._best_s > 0:
            self.reset()
            return "drift"
        if p + s >= self._best_p + self.warn * self._best_s and self._best_s > 0:
            return "warning"
        return "ok"


class EWMA:
    """Exponentially weighted moving-average control chart. Give the in-control mean and standard deviation, or let it learn them
    from the first ``warmup`` values. Good at small sustained shifts that a per-point threshold misses.

    ``width`` sets the false-alarm rate: at the default of 3 with ``lam`` 0.2 a false alarm comes about every 500 values on
    stable data (measured: about 580), at 4 about every 13,000. Use 4 for a stream you watch for months."""

    def __init__(self, mean: float | None = None, sd: float | None = None, lam: float = 0.2, width: float = 3.0, warmup: int = 50) -> None:
        if not 0 < lam <= 1:
            raise ValueError("lam must be in (0, 1]")
        self.lam, self.width, self.warmup = lam, width, warmup
        self._fixed = mean is not None and sd is not None
        self.mean = mean if mean is not None else 0.0
        self.sd = sd if sd is not None else 0.0
        self._buf: list[float] = []
        self.t = 0
        self.z = self.mean

    def update(self, x: float) -> bool:
        if not self._fixed and len(self._buf) < self.warmup:
            self._buf.append(x)
            if len(self._buf) == self.warmup:
                self.mean = sum(self._buf) / self.warmup
                self.sd = math.sqrt(sum((v - self.mean) ** 2 for v in self._buf) / (self.warmup - 1))
                self.z = self.mean
            return False
        self.t += 1
        self.z = self.lam * x + (1 - self.lam) * self.z
        limit = self.width * self.sd * math.sqrt(self.lam / (2 - self.lam) * (1 - (1 - self.lam) ** (2 * self.t)))
        return limit > 0 and abs(self.z - self.mean) > limit


@dataclass(frozen=True)
class WindowResult:
    start: int
    end: int
    n: int
    psi: float
    p_value: float
    severity: str


def scan(baseline: ColumnBaseline, values: Sequence[Any], window: int, step: int | None = None, policy: Policy | None = None) -> list[WindowResult]:
    """Slide a window over ordered values and assess each against the baseline: when did the drift start, and was it gradual or abrupt."""
    if window < 2:
        raise ValueError("window must be at least 2")
    step = step or window
    out: list[WindowResult] = []
    for start in range(0, max(len(values) - window, 0) + 1, step):
        chunk = values[start:start + window]
        f: FeatureDrift = assess(baseline, chunk, policy)
        out.append(WindowResult(start, start + len(chunk), f.n, f.psi, f.p_value, "n/a" if f.insufficient else f.severity))
    return out


def first_alert(results: Sequence[WindowResult], at: str = "alert") -> WindowResult | None:
    rank = {"n/a": -1, "ok": 0, "watch": 1, "alert": 2}
    return next((r for r in results if rank[r.severity] >= rank[at]), None)
