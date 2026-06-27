"""Feature extraction and text column construction.

Builds the two text columns used for retrieval (career_text, full_text) and
the structured numeric feature set consumed by every later stage, so no
downstream stage needs to re-parse a raw candidate record or rescan free text.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from flashtext import KeywordProcessor

from . import config
from .gates import compute_product_company_ratio

_IMPACT_PROCESSOR = KeywordProcessor()
_IMPACT_PROCESSOR.add_keywords_from_list(list(config.IMPACT_VERBS))


def build_career_text(candidate: dict[str, Any]) -> str:
    """High-precision evidence channel: headline, summary, and every role's
    title/company/description. This is where genuine production experience
    is described, as opposed to a bare skill-list keyword."""
    profile = candidate["profile"]
    parts = [profile["headline"], profile["summary"]]
    for role in candidate["career_history"]:
        parts.append(f"{role['title']} at {role['company']}: {role['description']}")
    return " | ".join(parts)


def build_full_text(candidate: dict[str, Any], career_text: str) -> str:
    """Broader recall channel: career_text plus skills and certifications,
    for candidates whose career descriptions are thin but whose skill list
    is genuinely relevant."""
    skills = ", ".join(skill["name"] for skill in candidate["skills"])
    certs = ", ".join(cert["name"] for cert in candidate.get("certifications", []))
    return f"{career_text} | Skills: {skills} | Certs: {certs}"


def _longest_tenure_years(candidate: dict[str, Any]) -> float:
    months = [role["duration_months"] for role in candidate["career_history"]]
    return max(months, default=0) / 12.0


def _is_india(country: str) -> bool:
    return country.strip().lower() == "india"


def _preferred_city_match(location: str) -> bool:
    location_lower = location.lower()
    return any(city in location_lower for city in config.PREFERRED_CITIES)


def _days_since(date_string: str, today: date) -> int:
    return (today - date.fromisoformat(date_string)).days


def _log_normalize(value: float, cap: float) -> float:
    clipped = min(max(value, 0), cap)
    return math.log1p(clipped) / math.log1p(cap)


@dataclass
class FeatureRow:
    candidate_id: str
    current_title: str
    current_company: str
    career_text: str
    full_text: str

    years_exp: float
    ai_ml_title_match: bool
    longest_tenure_years: float
    product_company_ratio: float
    education_tier: str

    location: str
    is_india: bool
    preferred_city_match: bool
    willing_to_relocate: bool

    open_to_work: bool
    notice_days: int
    last_active_days_ago: int

    recruiter_response_rate: float
    avg_response_hours: float
    interview_completion_rate: float
    offer_acceptance_rate_adj: float
    github_score_adj: float

    profile_completeness: float
    connections_log: float
    endorsements_log: float
    profile_views_norm: float
    saved_by_recruiters_norm: float
    search_appearance_norm: float

    github_is_linked: bool

    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    impact_verb_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _highest_education_tier(education: list[dict[str, Any]]) -> str:
    if not education:
        return "unknown"
    tier_rank = {"tier_1": 0, "tier_2": 1, "tier_3": 2, "tier_4": 3, "unknown": 4}
    tiers = [entry.get("tier", "unknown") or "unknown" for entry in education]
    return min(tiers, key=lambda tier: tier_rank.get(tier, 4))


def extract_features(candidate: dict[str, Any], today: date | None = None) -> FeatureRow:
    today = today or date.today()
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]

    career_text = build_career_text(candidate)
    full_text = build_full_text(candidate, career_text)

    return FeatureRow(
        candidate_id=candidate["candidate_id"],
        current_title=profile["current_title"],
        current_company=profile["current_company"],
        career_text=career_text,
        full_text=full_text,
        years_exp=profile["years_of_experience"],
        ai_ml_title_match=profile["current_title"] in config.AI_ML_TRACK_TITLES,
        longest_tenure_years=_longest_tenure_years(candidate),
        product_company_ratio=compute_product_company_ratio(candidate),
        education_tier=_highest_education_tier(candidate["education"]),
        location=profile["location"],
        is_india=_is_india(profile["country"]),
        preferred_city_match=_preferred_city_match(profile["location"]),
        willing_to_relocate=signals["willing_to_relocate"],
        open_to_work=signals["open_to_work_flag"],
        notice_days=signals["notice_period_days"],
        last_active_days_ago=_days_since(signals["last_active_date"], today),
        recruiter_response_rate=signals["recruiter_response_rate"],
        avg_response_hours=signals["avg_response_time_hours"],
        interview_completion_rate=signals["interview_completion_rate"],
        offer_acceptance_rate_adj=(
            config.OFFER_ACCEPTANCE_NEUTRAL_IMPUTED
            if signals["offer_acceptance_rate"] == config.OFFER_ACCEPTANCE_SENTINEL
            else signals["offer_acceptance_rate"]
        ),
        github_score_adj=(
            config.GITHUB_NEUTRAL_IMPUTED
            if signals["github_activity_score"] == config.GITHUB_SENTINEL
            else signals["github_activity_score"] / 100.0
        ),
        profile_completeness=signals["profile_completeness_score"] / 100.0,
        connections_log=_log_normalize(signals["connection_count"], config.CONNECTIONS_NORM_CAP),
        endorsements_log=_log_normalize(
            signals["endorsements_received"], config.ENDORSEMENTS_NORM_CAP
        ),
        profile_views_norm=_log_normalize(
            signals["profile_views_received_30d"], config.PROFILE_VIEWS_NORM_CAP
        ),
        saved_by_recruiters_norm=_log_normalize(
            signals["saved_by_recruiters_30d"], config.SAVED_BY_RECRUITERS_NORM_CAP
        ),
        search_appearance_norm=_log_normalize(
            signals["search_appearance_30d"], config.SEARCH_APPEARANCE_NORM_CAP
        ),
        github_is_linked=signals["github_activity_score"] != config.GITHUB_SENTINEL,
        verified_email=signals["verified_email"],
        verified_phone=signals["verified_phone"],
        linkedin_connected=signals["linkedin_connected"],
        impact_verb_count=len(_IMPACT_PROCESSOR.extract_keywords(career_text)),
    )


if __name__ == "__main__":
    import json
    import sys

    candidates_path = sys.argv[1] if len(sys.argv) > 1 else "candidates.jsonl"
    sample_size = 5
    with open(candidates_path) as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            candidate = json.loads(line)
            row = extract_features(candidate)
            print(row.candidate_id, row.current_title, "| years_exp:", row.years_exp,
                  "| ai_ml_title_match:", row.ai_ml_title_match,
                  "| product_company_ratio:", round(row.product_company_ratio, 2),
                  "| github_score_adj:", round(row.github_score_adj, 3))
