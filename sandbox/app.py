"""Minimal hosted demo for the required sandbox link.

Runs the full pipeline against a small uploaded sample (<=100 candidates),
within the same compute constraints as the real submission, so a reviewer
can verify the ranker actually runs without needing the full 100K pool or
a pre-built LanceDB database -- this builds a tiny in-memory index on the
fly for the uploaded sample.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ranker.embed import encode_passages
from src.ranker.features import extract_features
from src.ranker.gates import evaluate_gates
from src.ranker.index import build_indexes, connect, create_table, write_batches
from src.ranker.ingest import stream_candidates
from src.ranker.pipeline import run_pipeline

st.set_page_config(page_title="Redrob Ranker -- Sandbox", layout="wide")
st.title("Redrob Candidate Ranker -- Sandbox Demo")
st.caption(
    "Upload a small candidates.jsonl sample (<=100 rows) to verify the "
    "pipeline runs end to end. The real submission ranks the full 100K "
    "pool against a pre-built index; this sandbox builds a small one "
    "on the fly for demonstration."
)

uploaded = st.file_uploader("candidates.jsonl sample", type=["jsonl"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp.write(uploaded.getvalue().decode("utf-8"))
        tmp_path = tmp.name

    candidates = list(stream_candidates(tmp_path))
    st.write(f"Loaded {len(candidates)} candidates.")

    if len(candidates) > 100:
        st.warning("Sandbox is intended for samples of 100 or fewer candidates.")

    with st.spinner("Building a temporary index and ranking..."):
        db = connect(tempfile.mkdtemp())
        table = create_table(db, overwrite=True)

        rows = []
        for candidate in candidates:
            feature_row = extract_features(candidate)
            gate_result = evaluate_gates(candidate, feature_row.career_text)
            record = feature_row.to_dict()
            record["is_excluded"] = 1.0 if not gate_result.is_eligible else 0.0
            record["exclusion_reason"] = ",".join(
                gate_result.honeypot_reasons + gate_result.disqualifier_reasons
            )
            record["soft_penalty"] = gate_result.soft_penalty
            record["soft_flags"] = ",".join(gate_result.soft_flags)
            rows.append(record)

        embeddings = encode_passages([row["career_text"] for row in rows])
        for row, embedding in zip(rows, embeddings):
            row["embedding"] = embedding.tolist()

        write_batches(table, rows, batch_size=len(rows))
        build_indexes(table)  # falls back to a flat vector scan below 256 rows

        ranked = run_pipeline(table)

    st.success(f"Ranked {len(ranked)} candidates.")
    st.dataframe(
        [{"rank": r.rank, "candidate_id": r.candidate_id, "score": r.score, "reasoning": r.reasoning} for r in ranked],
        use_container_width=True,
    )
