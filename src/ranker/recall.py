"""Stage B-2 through B-4: dual-channel full-text recall, vector ANN recall,
and the union/dedup step that combines both into one shortlist with a single
normalized recall_score.

fts_columns is the correct, current LanceDB kwarg for targeting a specific
FTS-indexed column (confirmed against LanceDB's own docs). Every .where()
call also passes prefilter=True explicitly -- LanceDB's own FTS
documentation demonstrates this as the reliable way to combine a filter
with full-text search; relying on default post-filter behavior is the
condition under which lancedb/lancedb#1656 reported a filter being
silently ignored. _drop_excluded() below re-applies the is_excluded check
in Python as a second, independent layer on top of that, since the
downside of either failing silently (a honeypot leaking into the final
ranking) is a hard disqualification, not just a quality regression.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .embed import encode_query
from .jd_queries import FTS_CAREER_TEXT_QUERY, FTS_FULL_TEXT_QUERY, VECTOR_QUERY_TEXT

ELIGIBLE_FILTER = "is_excluded < 1.0"


@dataclass
class RecallRow:
    candidate_id: str
    bm25_score: float | None
    cosine_distance: float | None
    recall_score: float = 0.0
    found_via: str = "unknown"


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {key: 1.0 for key in values}
    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def _drop_excluded(results: list[dict]) -> list[dict]:
    """Defense in depth: re-checks is_excluded in Python after the query,
    rather than trusting .where() alone. LanceDB has a documented issue
    (lancedb/lancedb#1656) where a .where() filter can be silently ignored
    when combined with FTS search under certain index conditions in this
    version range. A honeypot leaking past a silently-ignored filter is
    exactly the failure mode that triggers automatic disqualification
    (>10% honeypot rate in top 100) -- this is cheap insurance against a
    single upstream library quirk causing that outcome undetected.
    """
    return [row for row in results if row.get("is_excluded", 0.0) < 1.0]


def fts_recall(table, query: str, top_k: int, text_column: str) -> dict[str, float]:
    results = (
        table.search(query, query_type="fts", fts_columns=text_column)
        .where(ELIGIBLE_FILTER, prefilter=True)
        .select(["candidate_id", "is_excluded"])
        .limit(top_k)
        .to_list()
    )
    results = _drop_excluded(results)
    return {row["candidate_id"]: row["_score"] for row in results}


def vector_recall(table, query_text: str, top_k: int) -> dict[str, float]:
    query_vector = encode_query(query_text)
    results = (
        table.search(query_vector, vector_column_name="embedding")
        .where(ELIGIBLE_FILTER, prefilter=True)
        .select(["candidate_id", "is_excluded"])
        .limit(top_k)
        .to_list()
    )
    results = _drop_excluded(results)
    return {row["candidate_id"]: row["_distance"] for row in results}


def build_recall_shortlist(table) -> list[RecallRow]:
    fts_a = fts_recall(table, FTS_CAREER_TEXT_QUERY, config.FTS_CAREER_TEXT_TOP_K, "career_text")
    fts_b = fts_recall(table, FTS_FULL_TEXT_QUERY, config.FTS_FULL_TEXT_TOP_K, "full_text")
    bm25_raw = {**fts_b, **fts_a}  # career_text channel takes precedence on overlap

    vector_distance_raw = vector_recall(table, VECTOR_QUERY_TEXT, config.VECTOR_RECALL_TOP_K)
    cosine_similarity_raw = {cid: 1.0 - dist for cid, dist in vector_distance_raw.items()}

    bm25_norm = _normalize(bm25_raw)
    cosine_norm = _normalize(cosine_similarity_raw)

    all_ids = set(bm25_norm) | set(cosine_norm)
    rows = []
    for cid in all_ids:
        has_keyword_match = cid in bm25_raw
        has_semantic_match = cid in vector_distance_raw
        if has_keyword_match and has_semantic_match:
            found_via = "keyword + semantic"
        elif has_semantic_match:
            found_via = "semantic only"
        else:
            found_via = "keyword only"

        rows.append(
            RecallRow(
                candidate_id=cid,
                bm25_score=bm25_raw.get(cid),
                cosine_distance=vector_distance_raw.get(cid),
                recall_score=0.5 * bm25_norm.get(cid, 0.0) + 0.5 * cosine_norm.get(cid, 0.0),
                found_via=found_via,
            )
        )
    return rows
