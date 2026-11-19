"""``python -m drift_watch``: fit a baseline, check new data against it, scan a column over time. Local files only."""
from __future__ import annotations

import argparse
import json
import sys

from .baseline import Baseline, NumericBaseline
from .io import read_rows
from .monitor import ORDER, Policy, check
from .stream import first_alert, scan


def _policy(args: argparse.Namespace) -> Policy:
    return Policy(psi_watch=args.psi_watch, psi_alert=args.psi_alert, alpha=args.alpha, min_samples=args.min_samples, adjust=not args.no_adjust)


def _add_policy(p: argparse.ArgumentParser) -> None:
    p.add_argument("--psi-watch", type=float, default=0.1)
    p.add_argument("--psi-alert", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.05, help="a PSI above its band only counts if it beats sampling noise at this level")
    p.add_argument("--min-samples", type=int, default=30)
    p.add_argument("--no-adjust", action="store_true", help="skip the Benjamini-Hochberg correction across features")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="drift_watch", description="Drift detection that separates drift from sampling noise.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fit = sub.add_parser("fit", help="fit a baseline from a reference dataset")
    fit.add_argument("data")
    fit.add_argument("--out", required=True)
    fit.add_argument("--columns", help="comma-separated columns to monitor (default: all but text columns)")
    fit.add_argument("--text", help="comma-separated free-text columns to profile (length, script, unseen-word rate)")
    fit.add_argument("--bins", type=int, default=10)
    ck = sub.add_parser("check", help="compare new data with a baseline; exit 1 when the gate is reached")
    ck.add_argument("baseline")
    ck.add_argument("data")
    ck.add_argument("--fail-on", choices=["watch", "alert", "never"], default="alert")
    ck.add_argument("--json", action="store_true")
    ck.add_argument("--markdown", action="store_true")
    _add_policy(ck)
    sc = sub.add_parser("scan", help="assess a column over sliding windows of ordered data")
    sc.add_argument("baseline")
    sc.add_argument("data")
    sc.add_argument("--column", required=True)
    sc.add_argument("--window", type=int, default=200)
    sc.add_argument("--step", type=int)
    _add_policy(sc)
    sh = sub.add_parser("show", help="describe a saved baseline")
    sh.add_argument("baseline")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "fit":
            cols = [c for c in args.columns.split(",") if c] if args.columns else None
            texts = [c for c in (args.text or "").split(",") if c]
            base = Baseline.fit(read_rows(args.data), cols, texts, bins=args.bins)
            base.save(args.out)
            print(f"fitted {len(base.columns)} features on {base.rows} rows -> {args.out}")
            for name, why in base.skipped.items():
                print(f"  skipped {name}: {why}")
            return 0
        base = Baseline.load(args.baseline)
        if args.cmd == "show":
            print(f"baseline: {base.rows} rows, {len(base.columns)} features, {len(base.vocab)} text vocabularies")
            for name, col in base.columns.items():
                extra = (f"range [{col.minimum:g}, {col.maximum:g}], {len(col.edges) + 1} bins" if isinstance(col, NumericBaseline)
                         else f"{len(col.counts)} categories{' (capped)' if col.capped else ''}")
                print(f"  {name:<28}{col.kind:<12}n={col.n:<8}null {col.null_rate:.1%}  {extra}")
            for name, why in base.skipped.items():
                print(f"  skipped {name}: {why}")
            return 0
        rows = read_rows(args.data)
        policy = _policy(args)
        if args.cmd == "scan":
            if args.column not in base.columns:
                print(f"error: {args.column!r} is not in the baseline", file=sys.stderr)
                return 2
            values = base.extract(rows)[args.column]
            results = scan(base.columns[args.column], values, args.window, args.step, policy)
            print(f"{'window':<16}{'n':>6}{'psi':>9}{'p':>9}  status")
            for r in results:
                print(f"{r.start}-{r.end:<10}{r.n:>6}{r.psi:>9.3f}{r.p_value:>9.3f}  {r.severity}")
            hit = first_alert(results)
            print(f"first alert: window starting at row {hit.start}" if hit else "no window reached alert")
            return 0
        report = check(base, rows, policy)
        print(json.dumps(report.to_dict(), indent=1, default=str) if args.json else report.to_markdown() if args.markdown else report.render())
        return 1 if args.fail_on != "never" and ORDER[report.verdict] >= ORDER[args.fail_on] else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
