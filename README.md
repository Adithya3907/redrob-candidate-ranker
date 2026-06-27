# Redrob Candidate Ranker

A CPU-only ranking pipeline that scores 100,000 candidate profiles against a
job description and returns the top 100, each with a rank, a score, and a
grounded explanation. Built for the Redrob Data & AI Challenge.

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

- **Semantic Intent Over Keyword Matching.** We pre-compute dense vectors
  offline (`BAAI/bge-small`) and use a Cross-Encoder (`ms-marco-MiniLM`) at
  runtime. Instead of playing Ctrl+F with the JD, our pipeline
  mathematically understands context — identifying that a candidate who
  "shipped a learning-to-rank pipeline" is a perfect fit, even if they never
  typed the word "recommendation."
- **Strict Business Constraints (Hard Gates).** The JD lists explicit
  negative constraints. Before the AI even looks at a profile, our
  deterministic pipeline drops candidates with pure Computer Vision titles,
  academic-only researchers without production deployments, and IT
  Services/Consulting backgrounds lacking product-company validation.
- **Behavioral Reality-Check.** A perfect-on-paper engineer with a 120-day
  notice period who hasn't logged in for 3 months is, for hiring purposes, a
  ghost. We apply aggressive, Optuna-tuned mathematical penalties to
  Redrob's behavioral signals (recruiter response rate, recency, notice
  period). Availability dictates rank just as much as capability.
- **Deterministic Honeypot Assassination.** The dataset contains ~80 subtly
  impossible honeypot profiles. Rather than relying on semantic AI to spot
  these fakes, we use cross-field arithmetic (e.g., claimed tenure > elapsed
  calendar time, or "expert" skills with zero months of usage). Honeypots
  are structurally isolated and zeroed out before scoring. See
  [ARCHITECTURE.md §5.1](ARCHITECTURE.md#51-honeypot-detection-is-structural-not-keyword-based).
- **Defensible, Grounded Reasoning.** Every Top 100 pick generates a
  reasoning string that acts as its defense attorney. If the AI surfaces an
  unconventional pick (e.g., a current CV Engineer), the reasoning
  explicitly highlights their past NLP production experience to prove
  compliance with JD constraints. See
  [ARCHITECTURE.md §5.2](ARCHITECTURE.md#52-reasoning-generation-cannot-hallucinate-by-construction).

Full design rationale, stage-by-stage data flow, and known limitations are in
[**ARCHITECTURE.md**](ARCHITECTURE.md).

## Quick start

```bash
git clone https://github.com/Adithya3907/redrob-candidate-ranker
cd redrob-candidate-ranker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Download `candidates.zip` and place it in the repository root.

### Windows (PowerShell)

```powershell
Expand-Archive -Path candidates.zip -DestinationPath . -Force
```

### Linux / macOS

```bash
unzip candidates.zip
```

```bash
# 1. Online ranking (the timed, official reproduction command)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# [Verification Steps]
# 2. Offline pre-build (takes ~25 mins on GPU if rebuilding artifacts):
#    python scripts/build_index.py --candidates ./candidates.jsonl
```

## Project structure

| Path | Contents |
|---|---|
| `rank.py` | Online ranking entry point — the official reproduction command. |
| `src/ranker/` | Pipeline implementation. One module per stage; see [ARCHITECTURE.md §3](ARCHITECTURE.md#3-building-blocks) for the full stage map. |
| `scripts/build_index.py` | Offline pre-build entry point (Phase A). |
| `research/tune_weights.py` | Optuna weight tuning against the gold set. Not imported by `rank.py`. |
| `gold/` | Hand-verified gold set used only by `tune_weights.py`. |
| `sandbox/app.py` | Streamlit demo for the required hosted sandbox. |
| `artifacts/` | Generated LanceDB database. Built by `build_index.py`, baked into the Docker image. |

Module-level detail for everything under `src/ranker/` — responsibility,
stage number, and why each one is designed the way it is — is in
[ARCHITECTURE.md §3](ARCHITECTURE.md#3-building-blocks).

## Weight tuning

```bash
python research/tune_weights.py --candidates ./candidates.jsonl --trials 5000
```

Optimizes directly against NDCG@k — the metric family the hackathon's own
evaluation uses — and reports a baseline-versus-tuned comparison rather than
a single final number. Copy the printed values into `src/ranker/config.py`,
replacing the constants marked `[OPTUNA-TUNED]`. Rationale in
[ARCHITECTURE.md §6.7](ARCHITECTURE.md#67-weight-tuning-optimizes-ndcg-directly-against-real-production-code).

## Sandbox

```bash
streamlit run sandbox/app.py
```

Accepts a sample of up to 100 candidates, builds a temporary index from that
sample alone, and runs the full pipeline against it — a working demonstration
of the pipeline independent of the full 100K build.

## Docker

`candidates.jsonl` is excluded from version control but is required in the
Docker build context: the Phase A build runs as part of `docker build`
itself, not as a separate manual step.

```bash
docker build -t redrob-ranker .     # runs the full Phase A build; ~25 min on GPU
docker run --rm -v $(pwd)/out:/app/out redrob-ranker \
  --candidates ./candidates.jsonl --out ./out/submission.csv
```

A clone of this repository plus a local copy of the dataset is sufficient to
reproduce both the build and the ranking from these two commands.

## AI tooling disclosure

See [submission_metadata.yaml](submission_metadata.yaml).
