"""Stage B-11: in-pipeline self-check.

Runs immediately after ranking, before rank.py writes the CSV or declares
success. The hackathon's own validate_submission.py catches the same format
issues -- but only if someone remembers to run it separately, after the
fact. Running the equivalent checks inside the pipeline itself means a
malformed run fails loudly, here, the moment it happens, rather than
shipping a bad CSV that looks fine until someone else's validator catches
it later.
"""

from __future__ import annotations

from . import config
from .pipeline import RankedCandidate


def assert_valid_submission(ranked: list[RankedCandidate]) -> None:
    n = config.OUTPUT_ROW_COUNT
    if len(ranked) != n:
        raise AssertionError(f"Expected exactly {n} ranked rows, got {len(ranked)}")

    ranks = [row.rank for row in ranked]
    if ranks != list(range(1, n + 1)):
        raise AssertionError("Ranks must be 1..N exactly, in ascending order")

    ids = [row.candidate_id for row in ranked]
    if len(set(ids)) != n:
        raise AssertionError("Duplicate candidate_id found in ranked output")

    scores = [row.score for row in ranked]
    if any(scores[i] < scores[i + 1] for i in range(n - 1)):
        raise AssertionError("Scores must be non-increasing by rank")

    reasonings = [row.reasoning.strip() for row in ranked]
    if not all(reasonings):
        raise AssertionError("Empty reasoning string found")
    if len(set(reasonings)) != n:
        raise AssertionError("Duplicate reasoning string found across two or more rows")

    print(
        f"Stage 11 self-check passed: {n} rows, ranks 1-{n} in order, "
        f"scores non-increasing, reasoning non-empty and unique across all rows."
    )
