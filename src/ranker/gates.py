"""
gates.py -- Stage A-1 (structural honeypot pre-flagging) and Stage 5a (JD hard
eligibility filters), kept in one file because they're conceptually the same
kind of operation: a deterministic, directly-testable rule that decides whether
a candidate is even allowed to compete, run once per candidate and cached as a
column rather than recomputed per query.

Two categories, kept distinct on purpose (see ARCHITECTURE.md Section 1):

  - structural_honeypot   : the profile is internally impossible. score = -99.0.
                             "~80 honeypots with subtly impossible profiles" per
                             the hackathon README.
  - jd_hard_disqualifier   : the profile is real, but the JD explicitly excludes
                             this category of candidate. score = -50.0.

Both are HARD exclusions -- once flagged, a candidate never reaches Stage 6/7/8
scoring. Soft flags (salary_inversion, ghost_coder, too_good_to_be_true,
cv_speech_robotics_without_ir) return a penalty float instead, and are combined
into the final composite score in compose.py, never used to exclude outright.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from dateutil.relativedelta import relativedelta

from . import config


@dataclass
class GateResult:
    """Output of running every gate against one candidate."""

    candidate_id: str
    is_structural_honeypot: bool = False
    honeypot_reasons: list[str] = field(default_factory=list)
    is_jd_hard_disqualified: bool = False
    disqualifier_reasons: list[str] = field(default_factory=list)
    soft_penalty: float = 0.0
    soft_flags: list[str] = field(default_factory=list)

    @property
    def is_eligible(self) -> bool:
        return not (self.is_structural_honeypot or self.is_jd_hard_disqualified)

    @property
    def exclusion_score(self) -> float | None:
        """Returns the hard sentinel score if excluded, else None."""
        if self.is_structural_honeypot:
            return config.STRUCTURAL_HONEYPOT_SCORE
        if self.is_jd_hard_disqualified:
            return config.JD_HARD_DISQUALIFIER_SCORE
        return None


def _months_between(start: date, end: date) -> int:
    r = relativedelta(end, start)
    return r.years * 12 + r.months


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


# ---------------------------------------------------------------------------
# Structural honeypot checks (Stage A-1)
# ---------------------------------------------------------------------------

def _check_duration_math_mismatch(candidate: dict[str, Any], today: date) -> bool:
    for role in candidate["career_history"]:
        start = _parse_date(role["start_date"])
        end = _parse_date(role["end_date"]) or today
        if start is None:
            continue
        actual_months = _months_between(start, end)
        if abs(actual_months - role["duration_months"]) > config.DURATION_MATH_TOLERANCE_MONTHS:
            return True
    return False


def _check_expert_zero_duration(candidate: dict[str, Any]) -> bool:
    count = sum(
        1
        for skill in candidate["skills"]
        if skill["proficiency"] == "expert"
        and skill.get("duration_months", 999) <= config.EXPERT_ZERO_DURATION_MAX_MONTHS
    )
    return count >= config.EXPERT_ZERO_DURATION_MIN_COUNT


def _check_application_inversion(candidate: dict[str, Any]) -> bool:
    sig = candidate["redrob_signals"]
    return (
        sig["applications_submitted_30d"] == 0
        and sig["profile_views_received_30d"] >= config.APPLICATION_INVERSION_VIEWS_THRESHOLD
    )


_SENIORITY_WORDS = ("Senior", "Staff", "Lead", "Principal")


def _check_stagnant_title(candidate: dict[str, Any]) -> bool:
    """Literal title-string match, AND the title must carry a seniority
    qualifier (Senior/Staff/Lead/Principal).

    The seniority-qualifier requirement is not optional: without it, this
    rule also flags candidates in generic, non-ladder titles (e.g. "QA
    Engineer" at three different employers) who simply never had a
    title-progression structure to begin with -- that's a normal career,
    not a structural anomaly. Verified against the real 100K pool: with the
    qualifier required, this rule fires on 6 candidates; without it, 202 --
    almost all of which would be ordinary, unremarkable careers wrongly
    flagged.

    Title normalization (stripping the seniority prefix before comparing) is
    deliberately NOT used here either: it conflates a genuine promotion
    (Senior -> Staff) with stagnation, since both reduce to the same base
    title once stripped.
    """
    profile = candidate["profile"]
    history = candidate["career_history"]
    if profile["years_of_experience"] < config.STAGNANT_TITLE_MIN_YEARS_EXPERIENCE:
        return False
    if len(history) < config.STAGNANT_TITLE_MIN_JOBS:
        return False
    titles = {role["title"] for role in history}
    if len(titles) != 1:
        return False
    only_title = next(iter(titles))
    return any(word in only_title for word in _SENIORITY_WORDS)


def run_structural_honeypot_checks(
    candidate: dict[str, Any], today: date | None = None
) -> tuple[bool, list[str]]:
    today = today or date.today()
    reasons = []

    if _check_duration_math_mismatch(candidate, today):
        reasons.append("duration_math_mismatch")
    if _check_expert_zero_duration(candidate):
        reasons.append("expert_zero_duration")
    if _check_application_inversion(candidate):
        reasons.append("application_inversion")
    if _check_stagnant_title(candidate):
        reasons.append("stagnant_title_3plus")

    return (len(reasons) > 0, reasons)


# ---------------------------------------------------------------------------
# JD hard disqualifier checks (Stage 5a)
# ---------------------------------------------------------------------------

def compute_product_company_ratio(candidate: dict[str, Any]) -> float:
    """Fraction of career_history months spent at non-consulting employers."""
    history = candidate["career_history"]
    total_months = sum(role["duration_months"] for role in history)
    if total_months == 0:
        return 0.0
    product_months = sum(
        role["duration_months"]
        for role in history
        if role["company"] not in config.CONSULTING_COMPANIES
    )
    return product_months / total_months


def _check_consulting_only(candidate: dict[str, Any]) -> bool:
    return compute_product_company_ratio(candidate) < config.CONSULTING_ONLY_MAX_PRODUCT_RATIO


def _check_nontechnical_title(candidate: dict[str, Any]) -> bool:
    return candidate["profile"]["current_title"] in config.NONTECHNICAL_TITLES


def _check_below_experience_floor(candidate: dict[str, Any]) -> bool:
    return candidate["profile"]["years_of_experience"] < config.MIN_YEARS_EXPERIENCE_HARD_FLOOR

def _check_pure_research_only(candidate: dict[str, Any], career_text: str) -> bool:
    if "Research" not in candidate["profile"]["current_title"]:
        return False
    text = career_text.lower()
    return not any(verb in text for verb in config.IMPACT_VERBS)

def run_jd_hard_disqualifier_checks(candidate: dict[str, Any], career_text: str) -> tuple[bool, list[str]]:
    reasons = []
    if _check_pure_research_only(candidate, career_text):
        reasons.append("pure_research_only")
    if _check_consulting_only(candidate):
        reasons.append("consulting_only")
    if _check_nontechnical_title(candidate):
        reasons.append("nontechnical_title")
    if _check_below_experience_floor(candidate):
        reasons.append("below_experience_floor")

    return (len(reasons) > 0, reasons)

    # cv_speech_robotics_without_ir is deliberately NOT a hard gate -- see
    # run_soft_flag_checks below. It stays soft until independently validated
    # against a larger sample; see config.py and ARCHITECTURE.md Section 4.

    return (len(reasons) > 0, reasons)


# ---------------------------------------------------------------------------
# Soft flags -- penalties, never exclusions
# ---------------------------------------------------------------------------

_PROGRAMMING_ADJACENT_SKILLS = {
    "Python", "Java", "C++", "Go", "JavaScript", "TypeScript", "SQL", "Scala",
    "Rust", "Kotlin", "Swift", "C#", ".NET",
}


def _check_salary_inversion(candidate: dict[str, Any]) -> bool:
    salary = candidate["redrob_signals"]["expected_salary_range_inr_lpa"]
    return salary["min"] > salary["max"]


def _check_ghost_coder(candidate: dict[str, Any]) -> bool:
    sig = candidate["redrob_signals"]
    if sig["github_activity_score"] <= config.GHOST_CODER_GITHUB_THRESHOLD:
        return False
    skill_names = {skill["name"] for skill in candidate["skills"]}
    return skill_names.isdisjoint(_PROGRAMMING_ADJACENT_SKILLS)


def _check_too_good_to_be_true(candidate: dict[str, Any]) -> bool:
    sig = candidate["redrob_signals"]
    offer_rate = sig["offer_acceptance_rate"]
    offer_ok = offer_rate == config.OFFER_ACCEPTANCE_SENTINEL or offer_rate >= 0.85
    return (
        sig["profile_completeness_score"] >= config.TOO_GOOD_PROFILE_COMPLETENESS_MIN
        and sig["recruiter_response_rate"] >= config.TOO_GOOD_RESPONSE_RATE_MIN
        and sig["interview_completion_rate"] >= config.TOO_GOOD_INTERVIEW_RATE_MIN
        and offer_ok
    )


def _check_cv_speech_without_ir(career_text: str) -> bool:
    """Experimental, soft-only signal -- see config.py and ARCHITECTURE.md
    Section 4. Uses multi-word phrase matching rather than single generic
    words to reduce the boilerplate false-positive/false-negative problem
    found during validation."""
    text = career_text.lower()
    has_cv_evidence = any(phrase in text for phrase in config.CV_SPECIFIC_PHRASES)
    has_ir_evidence = any(phrase in text for phrase in config.IR_NLP_PHRASES)
    return has_cv_evidence and not has_ir_evidence


def run_soft_flag_checks(
    candidate: dict[str, Any], career_text: str
) -> tuple[float, list[str]]:
    penalty = 0.0
    flags = []

    if _check_salary_inversion(candidate):
        penalty += config.SALARY_INVERSION_PENALTY
        flags.append("salary_inversion")
    if _check_ghost_coder(candidate):
        penalty += config.GHOST_CODER_PENALTY
        flags.append("ghost_coder")
    if _check_too_good_to_be_true(candidate):
        penalty += config.TOO_GOOD_PENALTY
        flags.append("too_good_to_be_true")
    if _check_cv_speech_without_ir(career_text):
        penalty += config.CV_WITHOUT_IR_PENALTY
        flags.append("cv_speech_robotics_without_ir")

    return (penalty, flags)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_gates(
    candidate: dict[str, Any], career_text: str, today: date | None = None
) -> GateResult:
    is_honeypot, honeypot_reasons = run_structural_honeypot_checks(candidate, today)
    is_disqualified, disqualifier_reasons = run_jd_hard_disqualifier_checks(candidate, career_text)
    soft_penalty, soft_flags = run_soft_flag_checks(candidate, career_text)

    return GateResult(
        candidate_id=candidate["candidate_id"],
        is_structural_honeypot=is_honeypot,
        honeypot_reasons=honeypot_reasons,
        is_jd_hard_disqualified=is_disqualified,
        disqualifier_reasons=disqualifier_reasons,
        soft_penalty=soft_penalty,
        soft_flags=soft_flags,
    )


# ---------------------------------------------------------------------------
# Self-check -- run this file directly: `python -m src.ranker.gates
# path/to/candidates.jsonl` from the repo root. Acts as the regression test
# for this module: a small set of known-eligible and known-excluded
# candidate_ids must always classify the same way.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    CANDIDATES_PATH = sys.argv[1] if len(sys.argv) > 1 else "candidates.jsonl"

    MUST_STAY_ELIGIBLE = {"CAND_0018499", "CAND_0080766", "CAND_0041517", "CAND_0091192"}
    KNOWN_STRUCTURAL_HONEYPOT_SAMPLE = {
        "CAND_0006567", "CAND_0049896", "CAND_0061265",  # application_inversion
        "CAND_0007353", "CAND_0008960",                   # duration_math_mismatch
    }

    found_eligible_failures = []
    found_honeypot_misses = []
    total = 0

    with open(CANDIDATES_PATH) as f:
        for line in f:
            if not line.strip():
                continue
            candidate = json.loads(line)
            total += 1
            cid = candidate["candidate_id"]

            if cid not in MUST_STAY_ELIGIBLE and cid not in KNOWN_STRUCTURAL_HONEYPOT_SAMPLE:
                continue

            career_text = candidate["profile"]["summary"] + " " + " ".join(
                role["description"] for role in candidate["career_history"]
            )
            result = evaluate_gates(candidate, career_text)

            if cid in MUST_STAY_ELIGIBLE and not result.is_eligible:
                found_eligible_failures.append((cid, result.honeypot_reasons, result.disqualifier_reasons))
            if cid in KNOWN_STRUCTURAL_HONEYPOT_SAMPLE and result.is_eligible:
                found_honeypot_misses.append(cid)

    print(f"Scanned {total} candidates.")
    print(f"Should-be-eligible candidates wrongly excluded: {len(found_eligible_failures)}")
    for failure in found_eligible_failures:
        print("  FAIL:", failure)
    print(f"Known honeypots that slipped through: {len(found_honeypot_misses)}")
    for miss in found_honeypot_misses:
        print("  MISS:", miss)
    if not found_eligible_failures and not found_honeypot_misses:
        print("All checks passed.")
