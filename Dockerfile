FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# candidates.jsonl is NOT committed to git (465MB, not original work to
# redistribute -- see .gitignore) but IS required in the Docker build
# context: place a local copy at the repo root before running `docker build`.
# The dataset is identical for every participant and fixed ahead of
# submission, so baking the embedding/index step in at build time, rather
# than inside the timed container run, is what makes the online phase fit
# the 5-minute budget at all -- see ARCHITECTURE.md Section 0 and the
# Phase A / Phase B split.
RUN python scripts/build_index.py --candidates ./candidates.jsonl

ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "./candidates.jsonl", "--out", "./submission.csv"]
