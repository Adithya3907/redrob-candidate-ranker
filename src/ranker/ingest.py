"""Streaming reader for the candidate pool.

Reads candidates.jsonl, or a gzip-compressed candidates.jsonl.gz, as an
iterator of decoded records. Records are yielded one at a time and grouped
into batches on demand, so peak memory usage stays proportional to batch
size rather than file size.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar

T = TypeVar("T")


def _open_candidates_file(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def stream_candidates(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one decoded candidate record at a time."""
    path = Path(path)
    with _open_candidates_file(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def batched(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    """Group an iterable into fixed-size lists. The final batch may be shorter."""
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def count_candidates(path: str | Path) -> int:
    """Count records without materializing the full collection in memory."""
    return sum(1 for _ in stream_candidates(path))


if __name__ == "__main__":
    import sys

    candidates_path = sys.argv[1] if len(sys.argv) > 1 else "candidates.jsonl"
    total = 0
    for batch in batched(stream_candidates(candidates_path), batch_size=1000):
        total += len(batch)
    print(f"Streamed {total} records from {candidates_path}.")
