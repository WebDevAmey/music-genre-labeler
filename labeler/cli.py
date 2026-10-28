"""Command line: python -m labeler {import,suggest,stats,export}"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import metrics
from .export import to_csv
from .labelers import LLMLabeler, KeywordLabeler
from .pipeline import suggest_all
from .store import Store
from .task import LABELS

DEFAULT_DB = "labels.db"


def build_report(store: Store) -> metrics.Report:
    items = [it for it in store.reviewed() if it.ai_label is not None]
    return metrics.report(
        human=[it.human_label for it in items],
        ai=[it.ai_label for it in items],
        confidence=[it.ai_confidence for it in items],
        labels=LABELS,
    )


def format_report(r: metrics.Report) -> str:
    if r.n == 0:
        return "No reviewed items yet."
    pct = lambda x: "  n/a" if x is None else f"{x:5.1%}"  # noqa: E731
    lines = [
        f"Reviewed items: {r.n}",
        f"AI accuracy:    {r.accuracy:.1%}",
        f"Cohen's kappa:  {r.kappa:.3f}",
        "",
        f"{'label':<14}{'support':>8}{'precision':>11}{'recall':>9}{'f1':>8}",
    ]
    for s in r.per_label:
        lines.append(f"{s.label:<14}{s.support:>8}{pct(s.precision):>11}{pct(s.recall):>9}{pct(s.f1):>8}")
    lines += ["", f"{'confidence':<14}{'items':>8}{'accuracy':>11}"]
    for b in r.bands:
        lines.append(f"{b.band:<14}{b.n:>8}{pct(b.accuracy):>11}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="labeler", description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite file (default {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import", help="load items from a CSV with id,text columns")
    p.add_argument("csv", type=Path)

    p = sub.add_parser("suggest", help="AI-label every item that has no suggestion yet")
    p.add_argument("--labeler", choices=("llm", "keyword"), default="llm")
    p.add_argument("--limit", type=int, help="label at most N items")
    p.add_argument("--workers", type=int, default=4, help="parallel API calls (default 4)")

    sub.add_parser("stats", help="AI accuracy against reviewed labels")

    p = sub.add_parser("export", help="write labels to CSV")
    p.add_argument("-o", "--out", type=Path, default=Path("labels.csv"))
    p.add_argument("--all", action="store_true", help="include unreviewed items")

    args = parser.parse_args(argv)
    store = Store(args.db)
    try:
        if args.command == "import":
            added = store.import_csv(args.csv)
            print(f"Imported {added} new items ({store.count()} total).")

        elif args.command == "suggest":
            labeler = LLMLabeler() if args.labeler == "llm" else KeywordLabeler()
            progress = lambda done, total: print(f"\r{done}/{total}", end="", file=sys.stderr)  # noqa: E731
            result = suggest_all(store, labeler, args.limit, args.workers, progress)
            print(f"\nLabeled {result.labeled} items with {labeler.name}.", file=sys.stderr)
            for item_id, reason in result.failed:
                print(f"  failed {item_id}: {reason}", file=sys.stderr)
            return 1 if result.failed else 0

        elif args.command == "stats":
            print(store.counts_by_status())
            print(format_report(build_report(store)))

        elif args.command == "export":
            items = store.all_items() if args.all else store.reviewed()
            args.out.write_text(to_csv(items), encoding="utf-8")
            print(f"Wrote {len(items)} rows to {args.out}.")
    finally:
        store.close()
    return 0
