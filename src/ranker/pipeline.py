"""
Orchestrates recall, the shortlist cap, dual-pass reranking, behavioral
scoring, the final composite, top-100 extraction with deterministic
tie-breaking, and reasoning generation, against a pre-built LanceDB table.
"""

from __future__ import annotations
from dataclasses import dataclass
from . import config
from .behavioral import score_behavioral
from .compose import compose_score
from .features import FeatureRow
from .filters import apply_shortlist_cap
from .reasoning import ReasoningContext, generate_reasoning
from .recall import build_recall_shortlist
from .rerank import rerank


@dataclass
class RankedCandidate:
    candidate_id: str
    rank: int
    score: float
    reasoning: str


def _rows_from_table(table, candidate_ids: list[str]) -> dict[str, dict]:
    quoted_ids = ", ".join(f"'{cid}'" for cid in candidate_ids)
    
    matches = (
        table.search()
        .where(f"candidate_id IN ({quoted_ids})")
        .limit(len(candidate_ids)) 
        .to_list()
    )
    
    rows_by_id = {row["candidate_id"]: row for row in matches}
    
    missing = set(candidate_ids) - set(rows_by_id)
    if missing:
        raise KeyError(f"candidate_ids not found in table: {sorted(missing)[:5]}...")
        
    return rows_by_id


def _feature_row_from_table_row(table_row: dict) -> FeatureRow:
    field_names = FeatureRow.__dataclass_fields__.keys()
    return FeatureRow(**{name: table_row[name] for name in field_names})


def _deterministic_tie_break(rows: list[tuple[FeatureRow, float]]) -> list[tuple[FeatureRow, float]]:
    return sorted(rows, key=lambda pair: (-pair[1], pair[0].candidate_id))


def run_pipeline(table) -> list[RankedCandidate]:
    recall_rows = build_recall_shortlist(table)
    shortlisted = apply_shortlist_cap(recall_rows, cap=config.SHORTLIST_HARD_CAP)
    found_via_by_id = {row.candidate_id: row.found_via for row in shortlisted}

    table_rows = _rows_from_table(table, [row.candidate_id for row in shortlisted])
    feature_rows = {
        cid: _feature_row_from_table_row(table_row) for cid, table_row in table_rows.items()
    }

    candidate_ids = list(feature_rows.keys())
    career_texts = [feature_rows[cid].career_text for cid in candidate_ids]
    rerank_results = {result.candidate_id: result for result in rerank(candidate_ids, career_texts)}

    scored: list[tuple[FeatureRow, float, object]] = []
    for cid in candidate_ids:
        feature_row = feature_rows[cid]
        rerank_result = rerank_results[cid]
        behavioral = score_behavioral(feature_row)
        soft_penalty = table_rows[cid]["soft_penalty"]
        composed = compose_score(feature_row, rerank_result, behavioral, soft_penalty)
        scored.append((feature_row, composed.final_score, composed))

    ranked_pairs = _deterministic_tie_break([(row, round(score, 4)) for row, score, _ in scored])
    composed_by_id = {row.candidate_id: composed for row, _, composed in scored}

    top_100 = ranked_pairs[: config.OUTPUT_ROW_COUNT]

    results = []
    previous_score = None 

    for rank, (feature_row, score) in enumerate(top_100, start=1):
        # Ensure the score never exceeds 0.985 and strictly strictly flows downward
        ABSOLUTE_MAX = 0.985
        safe_score = min(ABSOLUTE_MAX, score)
        clamped_score = safe_score if previous_score is None else min(safe_score, previous_score)

        composed = composed_by_id[feature_row.candidate_id]
        reasoning_text = generate_reasoning(
            ReasoningContext(
                feature_row=feature_row,
                composed=composed,
                rank=rank,
                found_via=found_via_by_id.get(feature_row.candidate_id, "unknown"),
            )
        )
        results.append(
            RankedCandidate(
                candidate_id=feature_row.candidate_id,
                rank=rank,
                score=clamped_score,  
                reasoning=reasoning_text,
            )
        )
        
        previous_score = clamped_score

    return results
