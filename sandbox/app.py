"""Sandbox demo for the required hackathon link.

Runs the full pipeline against an uploaded sample,
within the same compute constraints as the real submission.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ranker.embed import encode_passages
from src.ranker.features import extract_features
from src.ranker.gates import evaluate_gates
from src.ranker.index import build_indexes, connect, create_table, write_batches
from src.ranker.pipeline import run_pipeline

st.set_page_config(page_title="Redrob Ranker -- Sandbox", layout="wide")

# Reverted Title and added GitHub Link
st.title("Redrob Candidate Ranker -- Sandbox Demo")
st.markdown("[🔗 View Source on GitHub: Adithya3907/redrob-candidate-ranker](https://github.com/Adithya3907/redrob-candidate-ranker)")
st.caption(
    "Upload a small candidate sample to verify the pipeline runs end to end. "
    "The real submission ranks the full 100K pool against a pre-built index; "
    "this sandbox builds a small one on the fly for demonstration."
)

# Added CSV and XLSX to the allowed types
uploaded = st.file_uploader(
    "Upload candidates sample (Max 100)", 
    type=["jsonl", "json"]
)

if uploaded is not None:
    file_ext = uploaded.name.split('.')[-1].lower()
    candidates = []
    
    # -------------------------------------------------------------------------
    # Parse File Based on Extension
    # -------------------------------------------------------------------------
    try:
        if file_ext == "jsonl":
            lines = uploaded.getvalue().decode("utf-8").splitlines()
            candidates = [json.loads(line) for line in lines if line.strip()]
        elif file_ext == "json":
            candidates = json.loads(uploaded.getvalue().decode("utf-8"))
            if not isinstance(candidates, list):
                st.error("JSON file must be a list of candidate objects.")
                st.stop()
        elif file_ext == "csv":
            df = pd.read_csv(uploaded)
            candidates = df.to_dict(orient="records")
        elif file_ext == "xlsx":
            df = pd.read_excel(uploaded)
            candidates = df.to_dict(orient="records")
    except Exception as e:
        st.error(f"Error reading {file_ext} file: {e}")
        st.stop()
        
    total = len(candidates)
    if total == 0:
        st.error("No valid candidates found in the file.")
        st.stop()
        
    st.write(f"Loaded {total} candidates.")
    if total > 101:
        st.error("⚠️ Spec violation: Sandbox is strictly limited to 101 candidates maximum.")
        st.stop()

    # -------------------------------------------------------------------------
    # Live Progress Bar & Pipeline Execution
    # -------------------------------------------------------------------------
    progress_bar = st.progress(0, text="Initializing pipeline...")
    
    try:
        db = connect(tempfile.mkdtemp())
        table = create_table(db, overwrite=True)
        
        rows = []
        
        # Phase 1: Feature Extraction & Gating (0% to 30%)
        for i, candidate in enumerate(candidates):
            percent = int((i / total) * 30)
            progress_bar.progress(percent, text=f"Phase 1/4: Extracting features & evaluating gates ({i+1}/{total})...")
            
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
            
        # Phase 2: Embedding (30% to 70%)
        progress_bar.progress(30, text="Phase 2/4: Computing BGE-small dense embeddings (this takes time)...")
        texts = [row["career_text"] for row in rows]
        embeddings = encode_passages(texts)
        
        for row, embedding in zip(rows, embeddings):
            row["embedding"] = embedding.tolist()
            
        # Phase 3: Indexing (70% to 85%)
        progress_bar.progress(70, text="Phase 3/4: Building ephemeral LanceDB Index...")
        write_batches(table, rows, batch_size=len(rows))
        build_indexes(table)
        
        # Phase 4: Ranking & Behavioral Scoring (85% to 100%)
        progress_bar.progress(85, text="Phase 4/4: Reranking via cross-encoder and fusing behavioral signals...")
        ranked = run_pipeline(table)
        
        progress_bar.progress(100, text="Pipeline complete!")
        st.success(f"Successfully ranked {len(ranked)} candidates.")
        
    except KeyError as e:
        st.error(f"Data Schema Error: Missing expected field {e}. If using CSV/XLSX, ensure it exactly matches the JSON structure expected by the pipeline.")
        st.stop()
    except Exception as e:
        st.error(f"Pipeline Error: {e}")
        st.stop()

    # -------------------------------------------------------------------------
    # Display Results
    # -------------------------------------------------------------------------
    results_data = [
        {
            "Rank": r.rank, 
            "Candidate ID": r.candidate_id, 
            "Score": round(r.score, 4), 
            "Reasoning": r.reasoning
        } 
        for r in ranked
    ]
    
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    # Download Button
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Output CSV",
        data=csv,
        file_name='WhiteNoise.csv',
        mime='text/csv',
    )