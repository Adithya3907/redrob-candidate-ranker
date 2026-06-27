#!/usr/bin/env python3
"""Offline pre-build (Phase A): run once, ahead of submission, with no time
limit. Streams candidates.jsonl, runs the honeypot/disqualifier gates and
feature extraction, computes embeddings, writes everything to LanceDB, and
builds the FTS and vector indexes.

    python scripts/build_index.py --candidates ./candidates.jsonl

The resulting database is what rank.py reads at submission time. This script
is declared separately in submission_metadata.yaml under pre_computation_required
and pre_computation_time_minutes -- it is not part of the 5-minute online budget.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ranker import config
from src.ranker.embed import encode_passages, resolve_device
from src.ranker.features import extract_features
from src.ranker.gates import evaluate_gates
from src.ranker.index import build_indexes, connect, create_table, write_batches
from src.ranker.ingest import batched, stream_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the offline LanceDB index.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--lancedb-path", default=str(config.LANCEDB_PATH))
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for embedding computation. Pre-computation is not "
        "subject to the hackathon's no-GPU-during-ranking constraint, so "
        "'cuda' or 'auto' is safe and substantially faster on a machine "
        "with a GPU (e.g. a Colab/Kaggle runtime). The online path "
        "(rank.py) always uses CPU regardless of this flag.",
    )
    return parser.parse_args()


def _build_row(candidate: dict, today: date) -> dict:
    feature_row = extract_features(candidate, today)
    gate_result = evaluate_gates(candidate, feature_row.career_text, today)

    record = feature_row.to_dict()
    record["is_excluded"] = 1.0 if not gate_result.is_eligible else 0.0
    record["exclusion_reason"] = ",".join(gate_result.honeypot_reasons + gate_result.disqualifier_reasons)
    record["soft_penalty"] = gate_result.soft_penalty
    record["soft_flags"] = ",".join(gate_result.soft_flags)
    return record


def main() -> int:
    args = parse_args()
    today = date.today()
    start = time.monotonic()

    resolved_device = resolve_device(args.device)
    print(f"Embedding device: {resolved_device}")

    db = connect(args.lancedb_path)
    table = create_table(db, overwrite=True)

    total_written = 0
    total_excluded = 0

    for batch in batched(stream_candidates(args.candidates), args.batch_size):
        rows = [_build_row(candidate, today) for candidate in batch]
        embeddings = encode_passages([row["career_text"] for row in rows], device=resolved_device)
        for row, embedding in zip(rows, embeddings):
            row["embedding"] = embedding.tolist()
            if row["is_excluded"] >= 1.0:
                total_excluded += 1

        total_written += write_batches(table, rows, batch_size=len(rows))
        print(f"  ... {total_written} candidates written", end="\r")

    print()
    print(f"Wrote {total_written} candidates ({total_excluded} excluded by hard gates).")

    print("Building FTS and vector indexes...")
    build_indexes(table)

    elapsed = time.monotonic() - start
    print(f"Offline pre-build complete in {elapsed / 60:.1f} minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
