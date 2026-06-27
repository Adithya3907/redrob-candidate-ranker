"""Centralized JD-derived query strings.

Every semantic query the pipeline issues against career_text or full_text is
defined here once, so the interpretation of the job description encoded into
the system lives in a single, reviewable place rather than scattered across
the recall and reranking stages.
"""

FTS_CAREER_TEXT_QUERY = (
    "embedding retrieval vector search hybrid search learning-to-rank "
    "ranking pipeline semantic search recommendation system production "
    "machine learning pipeline evaluation framework NDCG MRR offline "
    "online correlation A/B test deployed inference scale"
)

FTS_FULL_TEXT_QUERY = (
    "sentence-transformers BGE E5 FAISS Pinecone Weaviate Qdrant "
    "OpenSearch Elasticsearch Milvus PEFT LoRA QLoRA fine-tuning "
    "XGBoost LightGBM production ML AI engineer NLP retrieval"
)

VECTOR_QUERY_TEXT = (
    "Senior AI engineer with production experience building embedding-based "
    "retrieval systems, vector databases, hybrid search, learning-to-rank, "
    "and evaluation frameworks. Has shipped ranking or recommendation "
    "systems to real users at scale. Strong Python, product company "
    "background, not pure research."
)

TECHNICAL_QUERY = (
    "Senior AI engineer role requiring production experience with "
    "embeddings-based retrieval systems (sentence-transformers, OpenAI "
    "embeddings, BGE, E5) deployed to real users, including embedding "
    "drift, index refresh, and retrieval-quality regression. Production "
    "experience with vector databases or hybrid search infrastructure "
    "(Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch). "
    "Strong Python with real code-quality emphasis. Hands-on evaluation "
    "framework experience for ranking systems -- NDCG, MRR, MAP, "
    "offline-to-online correlation, A/B interpretation. Learning-to-rank "
    "experience, XGBoost-based or neural. Has shipped at least one "
    "end-to-end ranking, search, or recommendation system to real users "
    "at meaningful scale."
)

CULTURAL_QUERY = (
    "Comfortable with ambiguity and shifting product priorities, suited to "
    "an early-stage startup rather than a fixed-checklist role. A "
    "generalist who cares about the end product and user outcomes, not "
    "just writing algorithms in isolation. A shipper who values rapid "
    "iteration and getting things to market over polishing research. "
    "Async-first communicator who writes clearly and at length. Has stayed "
    "multiple years at past employers rather than switching every "
    "eighteen months. Thinks in systems rather than chasing the newest "
    "framework. Comfortable disagreeing openly and deciding quickly "
    "without a rigid corporate process."
)
