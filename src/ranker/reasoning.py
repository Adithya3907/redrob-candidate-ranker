"""Stage B-10: reasoning generation.

Builds the 1-2 sentence reasoning string for each of the final 100
candidates from structured fields and FlashText-detected controlled
keywords only. Free text from profile.summary is never copied directly --
career-history summaries are candidate-authored, sometimes boilerplate, and
sometimes reused verbatim across unrelated profiles in this dataset, so
every clause in the output traces back to a field that was already
computed earlier in the pipeline rather than being generated independently.
"""

from __future__ import annotations

from dataclasses import dataclass

from flashtext import KeywordProcessor

from .compose import ComposedScore
from .features import FeatureRow

_TECHNICAL_PROCESSOR = KeywordProcessor()
_TECHNICAL_PROCESSOR.add_keywords_from_list([
    "learning-to-rank", "learning to rank", "LTR", "RAG", "retrieval-augmented",
    "vector search", "vector database", "embedding retrieval", "semantic search",
    "dense retrieval", "hybrid search", "recommendation system", "recommender system",
    "ranking system", "ranking pipeline", "NDCG", "MRR", "MAP", "evaluation framework",
    "fine-tuning", "LoRA", "QLoRA", "PEFT", "XGBoost", "LightGBM", "gradient boosting",
    "BM25", "inverted index", "full-text search", "sentence-transformers", "BGE", "E5",
    "Pinecone", "Weaviate", "Qdrant", "Milvus", "FAISS", "Elasticsearch", "OpenSearch",
    "MLflow", "Kubeflow", "Airflow", "production ML pipeline", "A/B test",
])

_CULTURAL_PROCESSOR = KeywordProcessor()
_CULTURAL_PROCESSOR.add_keywords_from_list([
    "shipped", "launched", "0 to 1", "scrappy", "owned end-to-end",
    "cross-functional", "iterated quickly", "rapid iteration",
])

_TECHNICAL_EVIDENCE_PRIORITY = (
    (("RAG", "retrieval-augmented"), "RAG/retrieval pipeline work in career history"),
    (
        ("vector search", "vector database", "embedding retrieval", "dense retrieval",
         "hybrid search", "sentence-transformers", "BGE", "E5",
         "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus"),
        "embedding-based retrieval system evidence",
    ),
    (
        ("BM25", "inverted index", "full-text search", "Elasticsearch", "OpenSearch"),
        "lexical/full-text search infrastructure evidence",
    ),
    (
        ("learning-to-rank", "learning to rank", "LTR", "ranking pipeline", "ranking system",
         "XGBoost", "LightGBM", "gradient boosting"),
        "learning-to-rank production experience",
    ),
    (("recommendation system", "recommender system"), "recommendation system production evidence"),
    (("semantic search",), "semantic search deployment evidence"),
    (("NDCG", "MRR", "MAP", "evaluation framework"), "ranking evaluation framework experience"),
    (("fine-tuning", "LoRA", "QLoRA", "PEFT"), "LLM fine-tuning experience"),
    (
        ("MLflow", "Kubeflow", "Airflow", "production ML pipeline", "A/B test"),
        "production ML pipeline/ops experience",
    ),
)
_DEFAULT_EVIDENCE = "applied ML/NLP career background"


def _strongest_technical_evidence(found_keywords: list[str]) -> str:
    found_set = set(found_keywords)
    for trigger_words, label in _TECHNICAL_EVIDENCE_PRIORITY:
        if found_set & set(trigger_words):
            return label
    return _DEFAULT_EVIDENCE


def _notice_clause(notice_days: int) -> str:
    if notice_days <= 30:
        return f"notice {notice_days}d"
    if notice_days <= 60:
        return f"notice {notice_days}d (manageable)"
    if notice_days <= 90:
        return f"notice {notice_days}d (flagged)"
    return f"notice {notice_days}d (long)"


def _github_clause(github_score_adj: float, github_is_linked: bool) -> str:
    if not github_is_linked:
        return ""
    return f"; GitHub activity {github_score_adj * 100:.0f}/100"


def _provenance_clause(found_via: str) -> str:
    """Surfaces the one genuinely interesting provenance case: a candidate
    a keyword search would have missed entirely. Silent for the other two
    cases (keyword-only, or found by both) to avoid adding a clause that's
    true for nearly every row, which would read as templated filler rather
    than information."""
    if found_via == "semantic only":
        return " Surfaced by semantic match alone -- a keyword search would have missed this profile."
    return ""


def _dominant_penalty_factor(row: FeatureRow, composed: ComposedScore) -> str | None:
    """The candidate's own single largest contributing weakness, used for
    the micro-rank justification clause. Not a literal comparison against
    the candidate one rank above -- a per-row honest accounting of what's
    actually dragging this specific candidate's score down."""
    candidates = []
    if row.notice_days > 30:
        candidates.append(((row.notice_days - 30) / 150, f"{row.notice_days}d notice period"))
    if row.last_active_days_ago > 60:
        candidates.append((row.last_active_days_ago / 365, f"inactive {row.last_active_days_ago}d"))
    if row.recruiter_response_rate < 0.5:
        candidates.append((0.5 - row.recruiter_response_rate, f"response rate {row.recruiter_response_rate:.2f}"))
    if composed.exp_mult < 1.0:
        candidates.append((1.0 - composed.exp_mult, "experience outside the ideal band"))
    if composed.pc_mult < 1.0:
        candidates.append((1.0 - composed.pc_mult, "limited product-company tenure"))
    if row.impact_verb_count == 0:
        candidates.append((0.35, "no shipped/owned/built evidence detected"))

    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


@dataclass
class ReasoningContext:
    feature_row: FeatureRow
    composed: ComposedScore
    rank: int
    found_via: str = "unknown"

_BORDERLINE_TITLE_MARKERS = ("Research", "Computer Vision")


def _borderline_defense_clause(row: FeatureRow) -> str | None:
    """Generates an explicit defense for candidates whose current title
    superficially resembles a disqualified pattern but who passed the gates
    legitimately. Only fires when the generic evidence fallback would
    otherwise be used -- a Stage-4 reviewer can't tell 'the AI missed this'
    apart from 'the AI correctly let this through without explaining why,'
    so this exists to make the second case say so explicitly."""
    if not any(marker in row.current_title for marker in _BORDERLINE_TITLE_MARKERS):
        return None
    if row.impact_verb_count > 0:
        return f"currently {row.current_title}, but career history shows shipped production evidence"
    if row.product_company_ratio >= 0.85:
        return f"currently {row.current_title}, but career history shows strong prior product-company tenure"
    return None

def generate_reasoning(context: ReasoningContext) -> str:
    row = context.feature_row
    found_technical = _TECHNICAL_PROCESSOR.extract_keywords(row.career_text)
    found_cultural = _CULTURAL_PROCESSOR.extract_keywords(row.career_text)

    evidence = _strongest_technical_evidence(found_technical)
    if evidence == _DEFAULT_EVIDENCE:
        defense_clause = _borderline_defense_clause(row)
        if defense_clause:
            evidence = defense_clause

    if row.longest_tenure_years >= 3.0:
        cultural_note = f"; {row.longest_tenure_years:.0f}y+ tenure at one employer"
    elif found_cultural:
        cultural_note = f"; career history emphasizes {found_cultural[0]}-style delivery"
    else:
        cultural_note = ""

    avail = _notice_clause(row.notice_days)
    github_clause = _github_clause(row.github_score_adj, row.github_is_linked)

    base_sentence = (
        f"{row.current_title} at {row.current_company} ({row.years_exp:.1f}y); "
        f"{evidence}{cultural_note}. {avail}, response rate {row.recruiter_response_rate:.2f}"
        f"{github_clause}."
    )

    dominant_weakness = _dominant_penalty_factor(row, context.composed)
    if dominant_weakness and context.rank > 10:
        base_sentence += f" Largest drag on rank: {dominant_weakness}."

    return base_sentence
