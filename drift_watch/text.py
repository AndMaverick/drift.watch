"""Turn free text (prompts, tickets, model outputs) into numeric and categorical features the drift machinery can watch."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

_WORD = re.compile(r"\w+", re.UNICODE)

_SCRIPTS: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("cjk", ((0x4E00, 0x9FFF), (0x3040, 0x30FF), (0xAC00, 0xD7AF), (0x3400, 0x4DBF))),
    ("cyrillic", ((0x0400, 0x04FF),)),
    ("devanagari", ((0x0900, 0x097F),)),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("greek", ((0x0370, 0x03FF),)),
)


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def script_of(char: str) -> str | None:
    """The writing system of a letter, or None for anything that is not a letter."""
    if not char.isalpha():
        return None
    code = ord(char)
    for name, ranges in _SCRIPTS:
        if any(lo <= code <= hi for lo, hi in ranges):
            return name
    return "latin" if code < 0x0250 or 0x1E00 <= code <= 0x1EFF else "other"


def dominant_script(text: str, mixed_below: float = 0.7) -> str:
    """The script most of the letters belong to, ``mixed`` when no script holds ``mixed_below`` of them, ``none`` with no letters."""
    counts = Counter(s for s in map(script_of, text) if s)
    total = sum(counts.values())
    if not total:
        return "none"
    name, count = counts.most_common(1)[0]
    return name if count / total >= mixed_below else "mixed"


def vocabulary(texts: Iterable[str], limit: int = 20000) -> dict[str, int]:
    """Document frequency of each word across ``texts``, keeping the ``limit`` most common."""
    df: Counter[str] = Counter()
    for text in texts:
        df.update(set(tokens(text)))
    return dict(df.most_common(limit))


def features(text: str, vocab: Mapping[str, int] | None = None, *, leave_one_out: bool = False) -> dict[str, float | str]:
    """Describe one text: size, composition, script and, given a vocabulary, the share of words never seen before.

    ``leave_one_out`` is for the baseline itself: a word only counts as known if some *other* baseline text also has it,
    so the baseline's out-of-vocabulary rate is honest instead of zero by construction.
    """
    toks = tokens(text)
    chars = len(text)
    letters = sum(c.isalpha() for c in text)
    out: dict[str, float | str] = {
        "chars": float(chars),
        "words": float(len(toks)),
        "digit_share": (sum(c.isdigit() for c in text) / chars) if chars else 0.0,
        "upper_share": (sum(c.isupper() for c in text) / letters) if letters else 0.0,
        "script": dominant_script(text),
    }
    if vocab is not None:
        need = 2 if leave_one_out else 1
        out["oov_rate"] = (sum(vocab.get(t, 0) < need for t in toks) / len(toks)) if toks else 0.0
    return out


def profile(texts: Sequence[str]) -> dict[str, list[float | str]]:
    """Column-wise features for a list of texts (no vocabulary)."""
    cols: dict[str, list[float | str]] = {}
    for text in texts:
        for key, value in features(text).items():
            cols.setdefault(key, []).append(value)
    return cols
