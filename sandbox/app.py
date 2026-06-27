"""Premium hosted demo for the required sandbox link.

Runs the full pipeline against a small uploaded sample,
within the same compute constraints as the real submission, so a reviewer
can verify the ranker actually runs without needing the full 100K pool or
a pre-built LanceDB database.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ranker.embed import encode_passages
from src.ranker.features import extract_features
from src.ranker.gates import evaluate_gates
from src.ranker.index import build_indexes, connect, create_table, write_batches
from src.ranker.ingest import stream_candidates
from src.ranker.pipeline import run_pipeline

# -----------------------------------------------------------------------------
# UI Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WhiteNoise | Candidate Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for a cleaner, ultra-modern look
st.markdown(
    """
    <style>
    /* Main container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Premium Header */
    h1 {
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    /* Subtitle text */
    .subtitle {
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Status metrics */
    .metric-container {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Header Section
# -----------------------------------------------------------------------------
st.title("⚡ WhiteNoise Ranker")
st.markdown(
    '<p class="subtitle">Stage 3 Verification Sandbox • Built for the Redrob 2026 AI Hackathon</p>',
    unsafe_allow_html=True,
)

with st.expander("ℹ️ How this Sandbox works", expanded=False):
    st.markdown(
        """
        This environment demonstrates the **Phase B (Online Ranking)** logic of the WhiteNoise pipeline. 
        
        Because the full 100,000-candidate LanceDB index exceeds Streamlit Cloud's memory limits, 
        this sandbox dynamically constructs an ephemeral vector/FTS index from your uploaded sample, 
        and then executes the exact same cross-encoder reranking and behavioral fusion logic used in our final submission.
        
        **Supported Formats:** `.jsonl`, `.json`
        """
    )

st.divider()

# -----------------------------------------------------------------------------
# File Upload Section
# -----------------------------------------------------------------------------
# Note: Streamlit file size limits are controlled via config.toml, not directly in code.
# To allow 1GB uploads, you MUST create a `.streamlit/config.toml` file in your repo.
uploaded = st.file_uploader(
    "Upload Candidate Data", 
    type=["jsonl", "json"],
    help="Upload a subset of candidates to verify pipeline execution."
)

if uploaded is not None:
    # Read the file to determine if we need to convert JSON to JSONL
    file_bytes = uploaded.getvalue()
    file_extension = uploaded.name.split('.')[-1].lower()
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        if file_extension == "json":
            # Handle standard JSON arrays
            try:
                data = json.loads(file_bytes.decode("utf-8"))
                if isinstance(data, list):
                    for item in data:
                        tmp.write(json.dumps(item) + "\n")
                else:
                    st.error("JSON file must contain a list of candidate objects.")
                    st.stop()
            except json.JSONDecodeError:
                st.error("Failed to parse JSON file. Please ensure it is valid.")
                st.stop()
        else:
            # Handle JSONL directly
            tmp.write(file_bytes.decode("utf-8"))
            
        tmp_path = tmp.name

    # -------------------------------------------------------------------------
    # Pipeline Execution
    # -------------------------------------------------------------------------
    candidates = list(stream_candidates(tmp_path))
    
    if not candidates:
        st.error("No valid candidates found in the uploaded file.")
        st.stop()
        
    if len(candidates) > 500:
        st.warning("⚠️ Large sample detected. The in-memory embedding step may exceed Streamlit's CPU/RAM limits. We recommend testing with <100 candidates.")

    # Status UI
    status_container = st.container()
    col1, col2, col3 = status_container.columns(3)
    
    col1.metric("Candidates Loaded", len(candidates))
    
    start_time = time.time()
    
    with st.status("Executing WhiteNoise Pipeline...", expanded=True) as status:
        st.write("🔧 Initializing ephemeral LanceDB workspace...")
        db = connect(tempfile.mkdtemp())
        table = create_table(db, overwrite=True)

        st.write("🛡️ Running deterministic hard gates and extracting features...")
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

        st.write("🧠 Computing BGE-small dense embeddings...")
        embeddings = encode_passages([row["career_text"] for row in rows])
        for row, embedding in zip(rows, embeddings):
            row["embedding"] = embedding.tolist()

        st.write("💾 Committing to index and executing multi-stage retrieval...")
        write_batches(table, rows, batch_size=len(rows))
        build_indexes(table)  

        st.write("⚖️ Reranking via Cross-Encoder and fusing behavioral signals...")
        ranked = run_pipeline(table)
        
        status.update(label="Pipeline Execution Complete", state="complete", expanded=False)

    end_time = time.time()
    
    # Update metrics post-run
    col2.metric("Candidates Ranked", len(ranked))
    col3.metric("Execution Time", f"{end_time - start_time:.2f}s")

    # -------------------------------------------------------------------------
    # Results Display
    # -------------------------------------------------------------------------
    st.subheader("🏆 Final Ranking Output")
    
    # Convert to DataFrame for a polished display
    results_data = [
        {
            "Rank": r.rank, 
            "Candidate ID": r.candidate_id, 
            "Composite Score": round(r.score, 4), 
            "Audit Trail (Reasoning)": r.reasoning
        } 
        for r in ranked
    ]
    
    df_results = pd.DataFrame(results_data)
    
    # Apply styling
    st.dataframe(
        df_results,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Composite Score": st.column_config.NumberColumn(format="%.4f"),
        }
    )
    
    # CSV Download Button
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download WhiteNoise.csv",
        data=csv,
        file_name='WhiteNoise.csv',
        mime='text/csv',
    )