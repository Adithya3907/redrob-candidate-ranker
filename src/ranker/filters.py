"""Stage B-5: deterministic shortlist cap.

The hard eligibility filters (consulting-only, non-technical title, below
experience floor) are pre-computed offline in gates.py and enforced as a
zero-cost WHERE clause inside every recall query in recall.py -- there is no
additional filtering work to do online beyond bounding the shortlist size
before the expensive reranking stage.
"""

from __future__ import annotations

from . import config
from .recall import RecallRow


def apply_shortlist_cap(
    rows: list[RecallRow], cap: int = config.SHORTLIST_HARD_CAP
) -> list[RecallRow]:
    """Bounds the number of candidates entering Stage 6, selected by highest
    recall_score if the cap is triggered. A ceiling, not a floor: fewer
    survivors than the cap pass through unchanged."""
    if len(rows) <= cap:
        return rows
    return sorted(rows, key=lambda row: row.recall_score, reverse=True)[:cap]
