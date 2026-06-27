"""Stage B-6: dual-pass cross-encoder reranking.

Each surviving candidate is scored twice against two separate semantic
chunks of the JD -- technical fit and cultural fit -- rather than one
blended query, so the two signals stay legible and independently weighted.

A runtime time-budget check sits between the two passes: if the technical
pass alone consumes more than the configured fraction of the allotted
window, the cultural pass is skipped for that run and a neutral value is
substituted, so the pipeline degrades gracefully on slower hardware instead
of risking the hard wall-clock cutoff.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import config
from .embed import score_pairs
from .jd_queries import CULTURAL_QUERY, TECHNICAL_QUERY

_NEUTRAL_CULTURAL_SCORE = 0.5


@dataclass
class RerankResult:
    candidate_id: str
    ce_technical_raw: float
    ce_cultural_raw: float
    ce_score: float
    cultural_pass_skipped: bool


def _minmax_norm(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(value - lo) / (hi - lo) for value in values]


def rerank(candidate_ids: list[str], career_texts: list[str]) -> list[RerankResult]:
    start = time.monotonic()

    technical_pairs = [(TECHNICAL_QUERY, text) for text in career_texts]
    ce_technical_raw = list(score_pairs(technical_pairs))

    elapsed = time.monotonic() - start
    deadline_fraction = elapsed / config.CE_TIME_BUDGET_SECONDS
    skip_cultural = deadline_fraction > config.CE_TIME_BUDGET_CULTURAL_SKIP_FRACTION

    if skip_cultural:
        ce_cultural_raw = [_NEUTRAL_CULTURAL_SCORE] * len(candidate_ids)
    else:
        cultural_pairs = [(CULTURAL_QUERY, text) for text in career_texts]
        ce_cultural_raw = list(score_pairs(cultural_pairs))

    technical_norm = _minmax_norm(ce_technical_raw)
    cultural_norm = _minmax_norm(ce_cultural_raw)

    results = []
    for i, candidate_id in enumerate(candidate_ids):
        ce_score = (
            config.CE_TECHNICAL_WEIGHT * technical_norm[i]
            + config.CE_CULTURAL_WEIGHT * cultural_norm[i]
        )
        results.append(
            RerankResult(
                candidate_id=candidate_id,
                ce_technical_raw=float(ce_technical_raw[i]),
                ce_cultural_raw=float(ce_cultural_raw[i]),
                ce_score=ce_score,
                cultural_pass_skipped=skip_cultural,
            )
        )
    return results
