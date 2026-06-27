"""Stage B-7: behavioral signal scoring.

The 23 raw behavioral signals are grouped into 4 super-features --
availability, reliability, market demand, platform trust -- which reduces
the Optuna search space from 23 independent weights to 4, keeping the
tuning problem well-posed relative to the size of the gold set. Within-group
sub-weights are hand-set design decisions, not tuned, to keep the search
space small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import config
from .features import FeatureRow


def _recency_score(days_ago: int) -> float:
    return math.exp(-days_ago / config.AVAILABILITY_RECENCY_DECAY_DAYS)


def _notice_score(notice_days: int) -> float:
    overage = max(0, notice_days - config.NOTICE_GRACE_PERIOD_DAYS)
    return max(0.0, 1.0 - overage / config.NOTICE_PENALTY_SCALE_DAYS)


def _location_multiplier(row: FeatureRow) -> float:
    if row.preferred_city_match:
        return 1.00
    if row.is_india:
        return 0.92
    if row.willing_to_relocate:
        return 0.85
    return 0.75


def availability_score(row: FeatureRow) -> float:
    open_to_work_score = (
        config.OPEN_TO_WORK_TRUE_SCORE if row.open_to_work else config.OPEN_TO_WORK_FALSE_SCORE
    )
    base = (
        config.AVAILABILITY_RECENCY_WEIGHT * _recency_score(row.last_active_days_ago)
        + config.AVAILABILITY_NOTICE_WEIGHT * _notice_score(row.notice_days)
        + config.AVAILABILITY_OPEN_TO_WORK_WEIGHT * open_to_work_score
    )
    availability = base * _location_multiplier(row)
    
    # Apply hard cliffs strictly to availability, not the overall raw score
    if row.notice_days > config.NOTICE_HARD_CLIFF_DAYS:
        availability *= config.NOTICE_HARD_CLIFF_MULTIPLIER
    if row.last_active_days_ago > config.RECENCY_HARD_CLIFF_DAYS:
        availability *= config.RECENCY_HARD_CLIFF_MULTIPLIER
        
    return availability


def reliability_score(row: FeatureRow) -> float:
    speed_score = max(0.0, 1.0 - row.avg_response_hours / config.RESPONSE_SPEED_SCALE_HOURS)
    return (
        config.RELIABILITY_RESPONSE_RATE_WEIGHT * (row.recruiter_response_rate ** 2)
        + config.RELIABILITY_RESPONSE_SPEED_WEIGHT * speed_score
        + config.RELIABILITY_INTERVIEW_RATE_WEIGHT * row.interview_completion_rate
        + config.RELIABILITY_OFFER_RATE_WEIGHT * row.offer_acceptance_rate_adj
    )


def market_demand_score(row: FeatureRow) -> float:
    return (
        config.MARKET_DEMAND_VIEWS_WEIGHT * row.profile_views_norm
        + config.MARKET_DEMAND_SAVED_WEIGHT * row.saved_by_recruiters_norm
        + config.MARKET_DEMAND_SEARCH_WEIGHT * row.search_appearance_norm
    )


def platform_trust_score(row: FeatureRow) -> float:
    verification_score = (
        int(row.verified_email) + int(row.verified_phone) + int(row.linkedin_connected)
    ) / 3.0
    social_score = 0.5 * row.connections_log + 0.5 * row.endorsements_log
    return (
        config.PLATFORM_TRUST_COMPLETENESS_WEIGHT * row.profile_completeness
        + config.PLATFORM_TRUST_GITHUB_WEIGHT * row.github_score_adj
        + config.PLATFORM_TRUST_VERIFICATION_WEIGHT * verification_score
        + config.PLATFORM_TRUST_SOCIAL_WEIGHT * social_score
    )


@dataclass
class BehavioralScore:
    candidate_id: str
    availability: float
    reliability: float
    market_demand: float
    platform_trust: float
    behavioral_score: float


def score_behavioral(
    row: FeatureRow,
    weight_availability: float = config.BEHAVIORAL_WEIGHT_AVAILABILITY,
    weight_reliability: float = config.BEHAVIORAL_WEIGHT_RELIABILITY,
    weight_market_demand: float = config.BEHAVIORAL_WEIGHT_MARKET_DEMAND,
    weight_platform_trust: float = config.BEHAVIORAL_WEIGHT_PLATFORM_TRUST,
) -> BehavioralScore:
    availability = availability_score(row)
    reliability = reliability_score(row)
    market_demand = market_demand_score(row)
    platform_trust = platform_trust_score(row)

    raw = (
        weight_availability * availability
        + weight_reliability * reliability
        + weight_market_demand * market_demand
        + weight_platform_trust * platform_trust
    )

    sigmoid = 1.0 / (
        1.0
        + math.exp(
            -config.BEHAVIORAL_SIGMOID_STEEPNESS * (raw - config.BEHAVIORAL_SIGMOID_MIDPOINT)
        )
    )

    return BehavioralScore(
        candidate_id=row.candidate_id,
        availability=availability,
        reliability=reliability,
        market_demand=market_demand,
        platform_trust=platform_trust,
        behavioral_score=sigmoid,
    )
