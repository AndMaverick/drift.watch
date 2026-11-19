"""Reading rows from JSON Lines, a JSON array, or CSV."""
from __future__ import annotations

import csv
import json
from typing import Any


def read_rows(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as fh:
        if path.lower().endswith(".csv"):
            return [{k: (None if v == "" else v) for k, v in row.items()} for row in csv.DictReader(fh)]
        raw = fh.read()
    if path.lower().endswith(".json"):
        data = json.loads(raw)
        if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
            raise ValueError(f"{path}: expected a JSON array of objects")
        return data
    rows = []
    for n, line in enumerate(raw.splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError(f"{path}: line {n} is not valid JSON") from None
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {n} is not an object")
            rows.append(row)
    return rows
