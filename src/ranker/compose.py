"""Stage B-8: final composite score.

Fuses the cross-encoder relevance score (Stage 6, itself already a
technical/cultural blend) with the behavioral score (Stage 7), then applies
role-fit multipliers as a proportional scale rather than a flat additive
penalty -- a strong relevance score paired with a clear band violation
(experience, consulting-dominant career, low education tier, short tenure)
should be scaled down proportionally, not just docked a fixed amount.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .behavioral import BehavioralScore
from .features import FeatureRow
from .rerank import RerankResult


def experience_multiplier(years_exp: float) -> float:
    for lo, hi, mult in config.EXPERIENCE_MULTIPLIER_BANDS:
        if lo <= years_exp <= hi:
            return mult
    return config.EXPERIENCE_MULTIPLIER_DEFAULT


def product_company_multiplier(ratio: float) -> float:
    for minimum, mult in config.PRODUCT_COMPANY_MULTIPLIER_BANDS:
        if ratio >= minimum:
            return mult
    return config.PRODUCT_COMPANY_MULTIPLIER_DEFAULT


def education_multiplier(tier: str) -> float:
    return config.EDUCATION_TIER_MULTIPLIER.get(tier, config.EDUCATION_TIER_MULTIPLIER["unknown"])


def tenure_multiplier(longest_tenure_years: float) -> float:
    for minimum, mult in config.TENURE_MULTIPLIER_BANDS:
        if longest_tenure_years >= minimum:
            return mult
    return config.TENURE_MULTIPLIER_DEFAULT


def impact_verb_multiplier(impact_verb_count: int) -> float:
    """Small positive multiplier for career_text containing high-ownership
    action verbs (shipped, deployed, architected, scaled, owned, built),
    linear up to a capped count. Bounded deliberately small (max +5%) since
    bare verb presence is a weaker signal than verb-near-a-technical-object
    would be."""
    capped_count = min(impact_verb_count, config.IMPACT_VERB_MAX_COUNT)
    return 1.0 + capped_count * config.IMPACT_VERB_STEP


@dataclass
class ComposedScore:
    candidate_id: str
    final_score: float
    ce_score: float
    behavioral_score: float
    exp_mult: float
    pc_mult: float
    edu_mult: float
    tenure_mult: float
    impact_mult: float


def compose_score(
    feature_row: FeatureRow,
    rerank_result: RerankResult,
    behavioral: BehavioralScore,
    soft_penalty: float,
    final_relevance_weight: float = config.FINAL_RELEVANCE_WEIGHT,
    final_behavioral_weight: float = config.FINAL_BEHAVIORAL_WEIGHT,
) -> ComposedScore:
    exp_mult = experience_multiplier(feature_row.years_exp)
    pc_mult = product_company_multiplier(feature_row.product_company_ratio)
    edu_mult = education_multiplier(feature_row.education_tier)
    t_mult = tenure_multiplier(feature_row.longest_tenure_years)
    impact_mult = impact_verb_multiplier(feature_row.impact_verb_count)

    base = (
        final_relevance_weight * rerank_result.ce_score
        + final_behavioral_weight * behavioral.behavioral_score
    )
    final_score = base * exp_mult * pc_mult * edu_mult * t_mult * impact_mult
    final_score = max(0.0, final_score - soft_penalty * 0.1)

    return ComposedScore(
        candidate_id=feature_row.candidate_id,
        final_score=final_score,
        ce_score=rerank_result.ce_score,
        behavioral_score=behavioral.behavioral_score,
        exp_mult=exp_mult,
        pc_mult=pc_mult,
        edu_mult=edu_mult,
        tenure_mult=t_mult,
        impact_mult=impact_mult,
    )
