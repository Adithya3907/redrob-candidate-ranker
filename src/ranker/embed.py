"""Embedding computation using BGE-small.

BGE models are trained for asymmetric retrieval: the query side carries an
instruction prefix, the passage side does not. Mixing the two up measurably
degrades retrieval quality, so the two encoding paths are kept as separate
functions rather than one shared one with a flag.

Device handling: the online ranking path (rank.py, recall.py, rerank.py)
never passes a device argument, so it always defaults to CPU, matching the
hackathon's no-GPU-during-ranking constraint. The offline pre-build
(scripts/build_index.py) is NOT subject to that constraint -- pre-computation
is explicitly allowed to use a GPU -- and can pass device="cuda" or
device="auto" to take advantage of one when available.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from . import config

_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_embedding_models: dict[str, SentenceTransformer] = {}
_cross_encoder_models: dict[str, CrossEncoder] = {}


def resolve_device(device: str = "cpu") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def get_embedding_model(device: str = "cpu") -> SentenceTransformer:
    resolved = resolve_device(device)
    if resolved not in _embedding_models:
        _embedding_models[resolved] = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=resolved)
    return _embedding_models[resolved]


def get_cross_encoder_model(device: str = "cpu") -> CrossEncoder:
    resolved = resolve_device(device)
    if resolved not in _cross_encoder_models:
        _cross_encoder_models[resolved] = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, device=resolved)
    return _cross_encoder_models[resolved]


def encode_passages(texts: Sequence[str], batch_size: int = 256, device: str = "cpu") -> np.ndarray:
    """Encode candidate career_text. No instruction prefix -- passage side."""
    model = get_embedding_model(device)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def encode_query(text: str, device: str = "cpu") -> np.ndarray:
    """Encode a JD query string. Instruction-prefixed -- query side."""
    model = get_embedding_model(device)
    return model.encode(
        _QUERY_INSTRUCTION + text,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def score_pairs(pairs: Sequence[tuple[str, str]], batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """Cross-encoder relevance score for a list of (query, passage) pairs."""
    model = get_cross_encoder_model(device)
    return model.predict(list(pairs), batch_size=batch_size, show_progress_bar=False)


if __name__ == "__main__":
    sample_texts = [
        "Built a hybrid BM25 and dense retrieval ranking pipeline for search.",
        "Managed accounts payable and quarterly financial reporting.",
    ]
    vectors = encode_passages(sample_texts)
    print("Passage embedding shape:", vectors.shape)

    query_vector = encode_query("Senior AI engineer with retrieval and ranking experience")
    print("Query embedding shape:", query_vector.shape)

    similarity = float(np.dot(query_vector, vectors[0]))
    print("Cosine similarity to relevant passage:", round(similarity, 4))

    pair_scores = score_pairs([("Senior AI engineer", sample_texts[0]), ("Senior AI engineer", sample_texts[1])])
    print("Cross-encoder scores:", [round(s, 4) for s in pair_scores])
