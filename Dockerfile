FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

# Build the LanceDB index offline (Phase A)
RUN python scripts/build_index.py --candidates ./candidates.jsonl

# CRITICAL: Force HuggingFace offline so it doesn't crash during Phase B
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["python", "rank.py"]
# Output MUST be the team name per hackathon rules
CMD ["--candidates", "./candidates.jsonl", "--out", "./WhiteNoise.csv"]