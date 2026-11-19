"""Fitting a baseline: what the data looked like when the model was built, stored compactly and exactly enough to test against."""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import measures, text

SCHEMA = 1
OTHER = "__other__"
Row = Mapping[str, Any]


def to_number(value: Any) -> float | None:
    """A finite float from a number or numeric string, else None. Booleans are not numbers here."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    elif isinstance(value, str):
        try:
            out = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return out if math.isfinite(out) else None


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"nan", "null", "none", "na", "n/a"}
    return isinstance(value, float) and math.isnan(value)


def label(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


@dataclass
class NumericBaseline:
    name: str
    n: int  # non-null values the baseline was fitted on
    rows: int  # rows, including nulls
    null_rate: float
    edges: list[float]
    proportions: list[float]  # share of the baseline in each bin, unfloored
    summary: list[float]  # up to 1000 evenly spaced quantiles: an exact copy when n <= 1000
    minimum: float
    maximum: float
    scale: float  # interquartile range (or standard deviation when that is zero), used to express shifts in natural units
    kind: str = "numeric"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class CategoricalBaseline:
    name: str
    n: int
    rows: int
    null_rate: float
    counts: dict[str, int]
    capped: bool = False  # rarer categories were folded into "__other__"
    kind: str = "categorical"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


ColumnBaseline = NumericBaseline | CategoricalBaseline


def _summary(ordered: Sequence[float], size: int = 1000) -> list[float]:
    n = len(ordered)
    if n <= size:
        return list(ordered)
    return [ordered[round(i * (n - 1) / (size - 1))] for i in range(size)]


def _quantile(ordered: Sequence[float], q: float) -> float:
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def fit_numeric(name: str, raw: Sequence[Any], bins: int = 10) -> NumericBaseline:
    numbers = sorted(x for x in (to_number(v) for v in raw if not is_null(v)) if x is not None)
    if len(numbers) < 2:
        raise ValueError(f"column {name!r}: need at least two numeric values to fit a baseline")
    edges = measures.quantile_edges(numbers, bins)
    counts = measures.bin_counts(numbers, edges)
    mean = sum(numbers) / len(numbers)
    sd = math.sqrt(sum((x - mean) ** 2 for x in numbers) / (len(numbers) - 1))
    iqr = _quantile(numbers, 0.75) - _quantile(numbers, 0.25)
    scale = iqr if iqr > 0 else (sd if sd > 0 else 1.0)
    return NumericBaseline(name, len(numbers), len(raw), 1 - len(numbers) / len(raw) if raw else 0.0, edges,
                           [c / len(numbers) for c in counts], _summary(numbers), numbers[0], numbers[-1], scale)


def fit_categorical(name: str, raw: Sequence[Any], max_categories: int = 50) -> CategoricalBaseline:
    values = [label(v) for v in raw if not is_null(v)]
    if not values:
        raise ValueError(f"column {name!r}: no values to fit a baseline")
    counts = Counter(values)
    capped = len(counts) > max_categories
    if capped:
        top = dict(counts.most_common(max_categories))
        top[OTHER] = len(values) - sum(top.values())
        counts = Counter(top)
    return CategoricalBaseline(name, len(values), len(raw), 1 - len(values) / len(raw), dict(counts), capped)


def infer_kind(raw: Sequence[Any]) -> str:
    """``numeric`` when nearly every value parses as a number and there is real spread, otherwise ``categorical``.
    A column of a few whole numbers (a 0/1 label, a rating from 1 to 5) is categorical: its distribution is a mix of classes."""
    present = [v for v in raw if not is_null(v)]
    if not present:
        return "categorical"
    nums = [x for x in (to_number(v) for v in present) if x is not None]
    if len(nums) / len(present) < 0.95:
        return "categorical"
    if all(x.is_integer() for x in nums) and len(set(nums)) <= 10:
        return "categorical"
    return "numeric"


@dataclass
class Baseline:
    columns: dict[str, ColumnBaseline]
    vocab: dict[str, dict[str, int]] = field(default_factory=dict)  # text column -> document frequency of words
    skipped: dict[str, str] = field(default_factory=dict)  # column -> why it is not monitored
    rows: int = 0
    bins: int = 10
    version: int = SCHEMA

    # ----------------------------------------------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, rows: Sequence[Row], columns: Sequence[str] | None = None, text_columns: Sequence[str] = (),
            kinds: Mapping[str, str] | None = None, bins: int = 10, max_categories: int = 50) -> Baseline:
        """Fit a baseline from rows (dicts). Columns default to every key that is not a text column. Identifier-like columns
        (almost every value distinct) are skipped and reported in ``skipped``, since their drift is meaningless."""
        if not rows:
            raise ValueError("no rows to fit")
        kinds = kinds or {}
        names = list(columns) if columns is not None else sorted({k for r in rows for k in r} - set(text_columns))
        base = cls({}, rows=len(rows), bins=bins)
        for name in names:
            raw = [r.get(name) for r in rows]
            if all(is_null(v) for v in raw):
                base.skipped[name] = "no values"
                continue
            kind = kinds.get(name) or infer_kind(raw)
            if kind == "categorical":
                distinct = len({label(v) for v in raw if not is_null(v)})
                if columns is None and name not in kinds and len(raw) >= 30 and distinct > 0.9 * len([v for v in raw if not is_null(v)]):
                    base.skipped[name] = "identifier-like (almost every value is distinct)"
                    continue
                base.columns[name] = fit_categorical(name, raw, max_categories)
            else:
                nums = [x for x in (to_number(v) for v in raw) if x is not None]
                if (columns is None and name not in kinds and len(nums) >= 30 and all(x.is_integer() for x in nums)
                        and len(set(nums)) == len(nums) and max(nums) - min(nums) + 1 == len(nums)):
                    base.skipped[name] = "identifier-like (a dense run of distinct whole numbers)"
                    continue
                base.columns[name] = fit_numeric(name, raw, bins)
        for col in text_columns:
            docs = [str(r.get(col)) for r in rows if not is_null(r.get(col))]
            if not docs:
                base.skipped[col] = "no text"
                continue
            vocab = text.vocabulary(docs)
            base.vocab[col] = vocab
            feats = [text.features(d, vocab, leave_one_out=True) for d in docs]
            for key in feats[0]:
                values = [f[key] for f in feats]
                base.columns[f"{col}.{key}"] = (fit_categorical(f"{col}.{key}", values, max_categories) if key == "script"
                                                else fit_numeric(f"{col}.{key}", values, bins))
        if not base.columns:
            raise ValueError("no columns could be fitted")
        return base

    # --------------------------------------------------------------------------------------------------- extracting
    def extract(self, rows: Iterable[Row]) -> dict[str, list[Any]]:
        """The values of every monitored column in ``rows``, computing text features where the baseline has a vocabulary."""
        rows = list(rows)
        out: dict[str, list[Any]] = {name: [] for name in self.columns}
        text_cols = set(self.vocab)
        for row in rows:
            derived: dict[str, dict[str, float | str]] = {}
            for col in text_cols:
                value = row.get(col)
                derived[col] = {} if is_null(value) else dict(text.features(str(value), self.vocab[col]))
            for name in self.columns:
                col, _, feature = name.partition(".")
                if col in text_cols and feature:
                    out[name].append(derived[col].get(feature))
                else:
                    out[name].append(row.get(name))
        return out

    # ---------------------------------------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "rows": self.rows, "bins": self.bins, "skipped": self.skipped, "vocab": self.vocab,
                "columns": {k: v.to_dict() for k, v in self.columns.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Baseline:
        if data.get("version") != SCHEMA:
            raise ValueError(f"unsupported baseline version {data.get('version')!r}")
        cols: dict[str, ColumnBaseline] = {}
        for name, d in data["columns"].items():
            d = dict(d)
            kind = d.pop("kind")
            cols[name] = NumericBaseline(**d) if kind == "numeric" else CategoricalBaseline(**d)
        return cls(cols, {k: dict(v) for k, v in data.get("vocab", {}).items()}, dict(data.get("skipped", {})),
                   int(data.get("rows", 0)), int(data.get("bins", 10)))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> Baseline:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
