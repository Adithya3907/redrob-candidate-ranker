#!/usr/bin/env python3
"""Single-command entry point for the online ranking phase.

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Assumes the offline pre-build has already produced a LanceDB database at
artifacts/lancedb (see scripts/build_index.py). This script only performs
the online phase: it does not re-embed the candidate pool, since that work
is fixed-cost and must happen once, ahead of time, not inside the 5-minute
window this script is measured against.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from src.ranker import config
from src.ranker.index import connect, open_table
from src.ranker.pipeline import run_pipeline
from src.ranker.validate import assert_valid_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank candidates against the Redrob JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl (unused directly; present for interface compatibility with the offline-built index).")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument("--lancedb-path", default=str(config.LANCEDB_PATH), help="Path to the pre-built LanceDB database.")
    return parser.parse_args()


def write_csv(rows, out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(config.OUTPUT_COLUMNS)
        for row in rows:
            writer.writerow([row.candidate_id, row.rank, row.score, row.reasoning])


def main() -> int:
    args = parse_args()
    start = time.monotonic()

    db_path = Path(args.lancedb_path)
    if not db_path.exists():
        print(
            f"No pre-built index found at {db_path}. Run scripts/build_index.py "
            "first -- the offline pre-build is not part of this script.",
            file=sys.stderr,
        )
        return 1

    db = connect(str(db_path))
    table = open_table(db)

    ranked = run_pipeline(table)
    assert_valid_submission(ranked)

    out_path = Path(args.out)
    write_csv(ranked, out_path)

    elapsed = time.monotonic() - start
    print(f"Wrote {len(ranked)} ranked rows to {out_path} in {elapsed:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
