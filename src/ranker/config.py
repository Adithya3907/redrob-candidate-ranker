"""
config.py -- every tuned constant for the ranking pipeline lives here, and nowhere
else. If you find yourself typing a bare number into gates.py, features.py,
behavioral.py, or compose.py, stop and add it here instead. This is the single
file you point to in the Stage 5 interview when asked "why this number and not
another" -- every constant below has a one-line reason attached.

Values marked [OPTUNA-TUNED] are placeholders until research/tune_weights.py has
actually been run against gold/gold_set_optuna_v3.jsonl. Replace them, don't trust
the defaults shipped here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LANCEDB_PATH = ARTIFACTS_DIR / "lancedb"
GOLD_SET_OPTUNA_PATH = PROJECT_ROOT / "gold" / "gold_set_optuna_v3.jsonl"

# ---------------------------------------------------------------------------
# Models (Stage A-3, Stage B-6)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Reference "today" -- NEVER hardcode a literal date here. All staleness /
# tenure-since calculations must call datetime.date.today() at runtime, not a
# fixed constant, or the pipeline silently produces wrong answers after the
# date this file was written.
# ---------------------------------------------------------------------------
# (Intentionally no TODAY constant. See src/ranker/features.py -- it calls
#  date.today() directly. This comment exists to stop a future edit from
#  "helpfully" adding one back.)

# ---------------------------------------------------------------------------
# Taxonomy -- derived directly from querying the full 100K candidate pool.
# These are dataset facts, not assumptions. Re-verify if the dataset changes.
# ---------------------------------------------------------------------------

# The 20 titles confirmed, by direct query, to be the AI/ML career track in
# this dataset. Used as a soft boost signal in features.py, NEVER as a hard
# include/exclude gate -- the JD explicitly warns that a real fit can carry a
# plain-language, non-AI-sounding title.
AI_ML_TRACK_TITLES = {
    "ML Engineer", "AI Research Engineer", "Data Scientist",
    "Senior Software Engineer (ML)", "Computer Vision Engineer",
    "Junior ML Engineer", "AI Specialist", "Recommendation Systems Engineer",
    "Machine Learning Engineer", "Applied ML Engineer", "Search Engineer",
    "AI Engineer", "Senior Data Scientist", "NLP Engineer",
    "Senior NLP Engineer", "Senior Machine Learning Engineer",
    "Staff Machine Learning Engineer", "Senior AI Engineer",
    "Senior Applied Scientist", "Lead AI Engineer",
}

# The 12 titles confirmed, by direct query, to be the irrelevant-function noise
# pool in this dataset (the keyword-stuffing trap population). Hard filter
# input -- Stage 5a #2.
NONTECHNICAL_TITLES = {
    "Business Analyst", "HR Manager", "Mechanical Engineer", "Accountant",
    "Project Manager", "Customer Support", "Operations Manager",
    "Content Writer", "Sales Executive", "Civil Engineer",
    "Graphic Designer", "Marketing Manager",
}

# Consulting/IT-services firms for the consulting_only hard filter (Stage 5a #1).
# JD names TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini explicitly with
# "etc." -- the rest are well-known peers in the same category, confirmed
# present as real company names in this dataset.
CONSULTING_COMPANIES = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "HCL", "Mphasis", "Tech Mahindra", "Hexaware", "LTIMindtree",
}

# JD-preferred locations (soft signal, never a hard gate -- JD explicitly says
# Hyderabad/Pune/Mumbai/Delhi NCR are all welcome beyond the headline
# Noida/Pune framing).
PREFERRED_CITIES = {
    "noida", "pune", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "new delhi", "navi mumbai",
}

# ---------------------------------------------------------------------------
# Hard gate thresholds -- Stage A-1 (structural honeypots) and Stage 5a (JD
# disqualifiers). These exclude a candidate outright. Be conservative here;
# a wrong hard exclusion is unrecoverable (see ARCHITECTURE.md Section 0) --
# a wrong soft penalty is not.
# ---------------------------------------------------------------------------

# duration_math_mismatch: |actual_months - stated duration_months| > this value
DURATION_MATH_TOLERANCE_MONTHS = 3

# expert_zero_duration: skill.proficiency == "expert" AND skill.duration_months
# <= this value. Verified clean binary signature in the real data: candidates
# have either zero or 3+ such skills, never 1-2.
EXPERT_ZERO_DURATION_MAX_MONTHS = 2
EXPERT_ZERO_DURATION_MIN_COUNT = 3

# application_inversion: applications_submitted_30d == 0 AND
# profile_views_received_30d >= this value.
APPLICATION_INVERSION_VIEWS_THRESHOLD = 150

# stagnant_title_3plus_jobs: literal title-string match (NOT seniority-prefix
# stripped -- that version produced false positives on real promoted
# candidates, see ARCHITECTURE.md Section 2). Requires >= this many distinct
# employers, all with the identical title string, over >= this many years.
STAGNANT_TITLE_MIN_JOBS = 3
STAGNANT_TITLE_MIN_YEARS_EXPERIENCE = 5.0

# consulting_only: (months at non-consulting employers / total career months)
# below this ratio.
CONSULTING_ONLY_MAX_PRODUCT_RATIO = 0.05

# Experience floor -- deliberately set BELOW the JD's stated 5-9y band. This
# only catches clearly-under-experienced profiles; it is not meant to enforce
# the soft 5-9y preference (that's exp_mult in compose.py).
MIN_YEARS_EXPERIENCE_HARD_FLOOR = 3.5

# ---------------------------------------------------------------------------
# Soft flag thresholds -- contribute a penalty, never an outright exclusion.
# See ARCHITECTURE.md Sections 3-4 for why these are soft, not hard.
# ---------------------------------------------------------------------------

SALARY_INVERSION_PENALTY = 0.4          # additive penalty to honeypot_score
GHOST_CODER_GITHUB_THRESHOLD = 70.0
GHOST_CODER_PENALTY = 0.4
SOFT_PENALTY_SCALE = 0.5
# too_good_to_be_true: all of profile_completeness_score, recruiter_response_rate,
# interview_completion_rate simultaneously above these. Set to literally match
# "above the 95th percentile" as originally specified, NOT independently
# re-derived from the real data in this session -- recompute precisely by
# adding percentile output to scripts/build_index.py once it has run end to
# end, and adjust these three constants directly if the real distribution
# disagrees with this assumption.
TOO_GOOD_PROFILE_COMPLETENESS_MIN = 95.0
TOO_GOOD_RESPONSE_RATE_MIN = 0.95
TOO_GOOD_INTERVIEW_RATE_MIN = 0.95
TOO_GOOD_PENALTY = 0.4

# cv_speech_robotics_without_ir -- INTENTIONALLY SOFT, not a hard gate, per
# ARCHITECTURE.md Section 4. The naive single-word keyword check was proven
# unreliable both directions in this exact dataset. This multi-word-phrase
# version is an improvement, not a verified-correct replacement -- treat its
# output with real skepticism until tested against a larger sample.
CV_SPECIFIC_PHRASES = (
    "object detection", "image segmentation", "image classification",
    "convolutional neural network", "ocr pipeline", "video analytics",
    "speech recognition", "speech synthesis", "audio classification",
    "robotic", "sensor fusion", "slam", "lidar",
)
IR_NLP_PHRASES = (
    "vector search", "semantic search", "hybrid retrieval", "hybrid search",
    "learning-to-rank", "learning to rank", "bm25", "dense retrieval",
    "retrieval-augmented", "ranking pipeline", "ranking system",
    "recommendation system", "recommender system", "embedding",
    "information retrieval",
)
CV_WITHOUT_IR_PENALTY = 0.3

# ---------------------------------------------------------------------------
# Sentinel value imputation (Stage A-2). 64.6% of the pool has no GitHub
# linked (-1); 59.6% has no offer history (-1). Treat as "missing," not "bad."
# ---------------------------------------------------------------------------
GITHUB_SENTINEL = -1
GITHUB_NEUTRAL_IMPUTED = 0.30
OFFER_ACCEPTANCE_SENTINEL = -1
OFFER_ACCEPTANCE_NEUTRAL_IMPUTED = 0.45

# ---------------------------------------------------------------------------
# Log-scale normalization caps for right-skewed count fields (Stage A-2).
# Computed from the real distribution of the full candidate pool rather than
# the literal max, since these fields have a long tail and a min-max scale
# against the true max would compress the bulk of the population near zero.
# normalized = log1p(min(x, cap)) / log1p(cap)
# ---------------------------------------------------------------------------
PROFILE_VIEWS_NORM_CAP = 150
SAVED_BY_RECRUITERS_NORM_CAP = 20
SEARCH_APPEARANCE_NORM_CAP = 250
CONNECTIONS_NORM_CAP = 600
ENDORSEMENTS_NORM_CAP = 60

# ---------------------------------------------------------------------------
# Recall funnel sizes (Stage 2-4)
# ---------------------------------------------------------------------------
FTS_CAREER_TEXT_TOP_K = 1000
FTS_FULL_TEXT_TOP_K = 700
VECTOR_RECALL_TOP_K = 900
SHORTLIST_HARD_CAP = 1200          # Stage 5b deterministic ceiling into Stage 6

# ---------------------------------------------------------------------------
# Stage 6: cross-encoder dual-pass fusion
# ---------------------------------------------------------------------------
CE_TECHNICAL_WEIGHT = 0.70
CE_CULTURAL_WEIGHT = 0.30
CE_TIME_BUDGET_SECONDS = 270        # hard internal deadline, leaves margin to 300s
CE_TIME_BUDGET_CULTURAL_SKIP_FRACTION = 0.60   # skip pass 2 if pass 1 alone used this much

# ---------------------------------------------------------------------------
# Stage 7: behavioral scoring -- 4 super-groups. [OPTUNA-TUNED] placeholders.
# ---------------------------------------------------------------------------
BEHAVIORAL_WEIGHT_AVAILABILITY = 0.5644   # [OPTUNA-TUNED] placeholder
BEHAVIORAL_WEIGHT_RELIABILITY = 0.5996    # [OPTUNA-TUNED] placeholder
BEHAVIORAL_WEIGHT_MARKET_DEMAND = 0.0598  # [OPTUNA-TUNED] placeholder
BEHAVIORAL_WEIGHT_PLATFORM_TRUST = 0.2250 # [OPTUNA-TUNED] placeholder

# Within-group sub-weights (hand-set design choices -- defend these as design
# decisions, not as something Optuna chose; only the 4 weights above are
# in scope for tuning, to keep the search space small relative to the 51-row
# gold set).
AVAILABILITY_RECENCY_WEIGHT = 0.40
AVAILABILITY_NOTICE_WEIGHT = 0.35
AVAILABILITY_OPEN_TO_WORK_WEIGHT = 0.25
AVAILABILITY_RECENCY_DECAY_DAYS = 90.0
NOTICE_GRACE_PERIOD_DAYS = 30
NOTICE_PENALTY_SCALE_DAYS = 150.0
OPEN_TO_WORK_TRUE_SCORE = 1.0
OPEN_TO_WORK_FALSE_SCORE = 0.6

RELIABILITY_RESPONSE_RATE_WEIGHT = 0.35
RELIABILITY_RESPONSE_SPEED_WEIGHT = 0.25
RELIABILITY_INTERVIEW_RATE_WEIGHT = 0.25
RELIABILITY_OFFER_RATE_WEIGHT = 0.15
RESPONSE_SPEED_SCALE_HOURS = 280.0

MARKET_DEMAND_VIEWS_WEIGHT = 0.35
MARKET_DEMAND_SAVED_WEIGHT = 0.35
MARKET_DEMAND_SEARCH_WEIGHT = 0.30

PLATFORM_TRUST_COMPLETENESS_WEIGHT = 0.30
PLATFORM_TRUST_GITHUB_WEIGHT = 0.30
PLATFORM_TRUST_VERIFICATION_WEIGHT = 0.25
PLATFORM_TRUST_SOCIAL_WEIGHT = 0.15

BEHAVIORAL_SIGMOID_STEEPNESS = 5.0
BEHAVIORAL_SIGMOID_MIDPOINT = 0.5

# ---------------------------------------------------------------------------
# Stage 8: final composite + role-fit multipliers
# ---------------------------------------------------------------------------
FINAL_RELEVANCE_WEIGHT = 0.4001       # [OPTUNA-TUNED] placeholder
FINAL_BEHAVIORAL_WEIGHT = 0.5999      # [OPTUNA-TUNED] placeholder

EXPERIENCE_MULTIPLIER_BANDS = (
    # (min_years_inclusive, max_years_inclusive, multiplier)
    (5.0, 9.0, 1.00),
    (4.0, 5.0, 0.88),
    (9.0, 12.0, 0.92),
)
EXPERIENCE_MULTIPLIER_DEFAULT = 0.72

PRODUCT_COMPANY_MULTIPLIER_BANDS = (
    # (minimum_ratio_inclusive, multiplier), checked in order, first match wins
    (0.85, 1.00),
    (0.50, 0.90),
    (0.20, 0.78),
)
PRODUCT_COMPANY_MULTIPLIER_DEFAULT = 0.60

EDUCATION_TIER_MULTIPLIER = {
    "tier_1": 1.00, "tier_2": 0.97, "tier_3": 0.94, "tier_4": 0.91,
    "unknown": 0.95,
}

TENURE_MULTIPLIER_BANDS = (
    # (minimum_years_inclusive, multiplier), checked in order, first match wins
    (2.5, 1.00),
    (1.5, 0.93),
)
TENURE_MULTIPLIER_DEFAULT = 0.80

# Hard-gate / disqualification sentinel scores (sort to the bottom, never
# appear in a real top-100 if the funnel upstream is working correctly).
STRUCTURAL_HONEYPOT_SCORE = -99.0
JD_HARD_DISQUALIFIER_SCORE = -50.0

# ---------------------------------------------------------------------------
# Stage 10: reasoning generation -- shipper-evidence multiplier
# ---------------------------------------------------------------------------
# Multi-word phrases, not bare verbs -- "built" or "owned" alone occur too
# incidentally in ordinary prose to be reliable evidence; a multi-word phrase
# is materially harder to land on by accident, which is the entire premise
# of using this as a signal at all.
IMPACT_VERBS = (
    "shipped", "deployed", "architected", "launched",
    "built and shipped", "owned and shipped",
    "released to production", "took to production",
    "drove to launch", "pushed to prod",
)
IMPACT_VERB_STEP = 0.0125
IMPACT_VERB_MAX_COUNT = 4            # caps the multiplier at 1.05

# Output contract (Stage 11 / validate_submission.py)
OUTPUT_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
OUTPUT_ROW_COUNT = 100
