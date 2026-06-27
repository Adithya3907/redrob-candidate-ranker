# Redrob Candidate Ranker

A CPU-only ranking pipeline that scores 100,000 candidate profiles against a
job description and returns the top 100, each with a rank, a score, and a
grounded explanation. Built for the Redrob Data \& AI Challenge.

Runs in under 5 minutes, on CPU, with no GPU and no network access during
ranking.

## Approach

The dataset contains a deliberate trap: candidates whose skills list is
dense with AI buzzwords, but whose actual career history (e.g., Marketing
Manager) has nothing to do with the role. An architecture built on keyword
density (BM25) fails this immediately.

Our system does not just match text; it reverse-engineers the Job
Description's explicit business constraints. We built a multi-stage funnel
that thinks like a Senior Technical Recruiter:

* **Semantic Intent Over Keyword Matching.** We pre-compute dense vectors
offline (`BAAI/bge-small`) and use a Cross-Encoder (`ms-marco-MiniLM`) at
runtime. Instead of playing Ctrl+F with the JD, our pipeline
mathematically understands context — identifying that a candidate who
"shipped a learning-to-rank pipeline" is a perfect fit, even if they never
typed the word "recommendation."
* **Strict Business Constraints (Hard Gates).** The JD lists explicit
negative constraints. Before the AI even looks at a profile, our
deterministic pipeline drops candidates with pure Computer Vision titles,
academic-only researchers without production deployments, and IT
Services/Consulting backgrounds lacking product-company validation.
* **Behavioral Reality-Check.** A perfect-on-paper engineer with a 120-day
notice period who hasn't logged in for 3 months is, for hiring purposes, a
ghost. We apply aggressive, Optuna-tuned mathematical penalties to
Redrob's behavioral signals (recruiter response rate, recency, notice
period). Availability dictates rank just as much as capability.
* **Deterministic Honeypot Assassination.** The dataset contains \~80 subtly
impossible honeypot profiles. Rather than relying on semantic AI to spot
these fakes, we use cross-field arithmetic (e.g., claimed tenure > elapsed
calendar time, or "expert" skills with zero months of usage). Honeypots
are structurally isolated and zeroed out before scoring. See
[ARCHITECTURE.md §5.1](ARCHITECTURE.md#51-honeypot-detection-is-structural-not-keyword-based).
* **Defensible, Grounded Reasoning.** Every Top 100 pick generates a
reasoning string that acts as its defense attorney. If the AI surfaces an
unconventional pick (e.g., a current CV Engineer), the reasoning
explicitly highlights their past NLP production experience to prove
compliance with JD constraints. See
[ARCHITECTURE.md §5.2](ARCHITECTURE.md#52-reasoning-generation-cannot-hallucinate-by-construction).

Full design rationale, stage-by-stage data flow, and known limitations are in
[**ARCHITECTURE.md**](ARCHITECTURE.md).

## Reproduce

> **Prerequisite:** Local reproduction of this pipeline (both Phase A index building and Phase B ranking) requires **Python 3.12**. For the official Stage 3 automated evaluation, please use the Docker instructions at the bottom of this document.

```bash
git clone https://github.com/Adithya3907/redrob-candidate-ranker
cd redrob-candidate-ranker
```

Download `candidates.zip` and place it in the repository root.

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Expand-Archive -Path candidates.zip -DestinationPath . -Force
```

### Linux / macOS

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
unzip candidates.zip
```

### 🚀 Execute the Pipeline

```bash
# 1. Online ranking (the timed, official reproduction command)
python rank.py --candidates ./candidates.jsonl --out ./WhiteNoise.csv

# [Verification Steps]
# 2. Offline pre-build (takes ~25 mins on GPU if rebuilding artifacts):
#    python scripts/build_index.py --candidates ./candidates.jsonl
```

## Project structure

|Path|Contents|
|-|-|
|`rank.py`|Online ranking entry point — the official reproduction command.|
|`src/ranker/`|Pipeline implementation. One module per stage; see [ARCHITECTURE.md §3](ARCHITECTURE.md#3-building-blocks) for the full stage map.|
|`scripts/build\_index.py`|Offline pre-build entry point (Phase A).|
|`research/tune\_weights.py`|Optuna weight tuning against the gold set. Not imported by `rank.py`.|
|`gold/`|Hand-verified gold set used only by `tune\_weights.py`.|
|`sandbox/app.py`|Streamlit demo for the required hosted sandbox.|
|`artifacts/`|Generated LanceDB database. Built by `build\_index.py`, baked into the Docker image.|

Module-level detail for everything under `src/ranker/` — responsibility,
stage number, and why each one is designed the way it is — is in
[ARCHITECTURE.md §3](ARCHITECTURE.md#3-building-blocks).

## Weight tuning

```bash
python research/tune\_weights.py --candidates ./candidates.jsonl --trials 5000
```

Optimizes directly against NDCG@k — the metric family the hackathon's own
evaluation uses — and reports a baseline-versus-tuned comparison rather than
a single final number. Copy the printed values into `src/ranker/config.py`,
replacing the constants marked `\[OPTUNA-TUNED]`. Rationale in
[ARCHITECTURE.md §6.7](ARCHITECTURE.md#67-weight-tuning-optimizes-ndcg-directly-against-real-production-code).

## 🖥️ Sandbox (Live Demo)

**🌐 Try the Live UI:** https://redrob-candidate-ranker-adithya3907.streamlit.app/

The hosted sandbox accepts a sample file (JSON/JSONL/CSV/XLSX) of up to 100 candidates. Because of cloud memory constraints, it builds a temporary ephemeral LanceDB index from your uploaded sample alone, and runs the full Phase A + Phase B pipeline live in the browser. It is a working demonstration of the pipeline's logic independent of the full 100K build.

### Run the Sandbox Locally
If you wish to run the UI on your own machine:
```bash
streamlit run sandbox/app.py
```

## 🚀 Reproduction (Docker)

This pipeline uses an **Offline-Build, Online-Rank** architecture to strictly adhere to the 5-minute, CPU-only, and no-network constraints. 

Because embedding 100,000 candidates takes ~25 minutes on GPU, the heavy dense embeddings and LanceDB index generation (Phase A) are baked directly into the Docker build step. The timed ranking execution (Phase B) runs entirely offline.

### Prerequisites
Please ensure the official `candidates.jsonl` (465MB) file is placed in the root directory of this repository before building. It is excluded from version control but required for the build context.

### Step 1: Build the Image (Phase A - Pre-computation)
*Note: Network access is temporarily enabled during this step to download the BGE-small weights and build the LanceDB index.*
```bash
docker build -t whitenoise-ranker .
```

### Step 2: Run the Ranker (Phase B - Timed Execution)
*Note: This command strictly enforces the Stage 3 constraints (no network, 16GB RAM, only CPU). It will execute the cross-encoder and output the top 100 CSV within the 5-minute window.*
```bash
docker run --rm --network none --memory="16g" -v $(pwd):/app/output whitenoise-ranker
```
*The final output will be generated in your local directory as `WhiteNoise.csv`.*

## AI tooling disclosure

See [submission\_metadata.yaml](submission_metadata.yaml).

