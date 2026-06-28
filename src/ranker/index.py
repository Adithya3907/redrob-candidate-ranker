"""LanceDB schema definition, table construction, and index building.

Owns the offline write path (Stage A-4): defines the on-disk schema, writes
candidate rows in batches, and builds the FTS and vector indexes once all
rows are present. The online query path (Stage B) opens the table this
module produces but performs its own searches in recall.py.
"""

from __future__ import annotations

from typing import Any, Iterable

import lancedb
from lancedb.pydantic import LanceModel, Vector

from . import config

TABLE_NAME = "candidates"

# Below this row count, IVF_PQ cannot train (it requires a minimum number of
# vectors per partition by default). The small-sample sandbox path runs on
# far fewer rows than this, so vector search falls back to an unindexed flat
# scan there -- correct results, just without the index, which is the right
# tradeoff at that scale.
MIN_ROWS_FOR_VECTOR_INDEX = 256


class CandidateRecord(LanceModel):
    candidate_id: str
    current_title: str
    current_company: str
    career_text: str
    full_text: str
    embedding: Vector(config.EMBEDDING_DIM)

    is_excluded: float
    exclusion_reason: str
    soft_penalty: float
    soft_flags: str

    years_exp: float
    ai_ml_title_match: bool
    longest_tenure_years: float
    product_company_ratio: float
    education_tier: str

    location: str
    is_india: bool
    preferred_city_match: bool
    willing_to_relocate: bool

    open_to_work: bool
    notice_days: int
    last_active_days_ago: int

    recruiter_response_rate: float
    avg_response_hours: float
    interview_completion_rate: float
    offer_acceptance_rate_adj: float
    github_score_adj: float

    profile_completeness: float
    connections_log: float
    endorsements_log: float
    profile_views_norm: float
    saved_by_recruiters_norm: float
    search_appearance_norm: float

    github_is_linked: bool

    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    impact_verb_count: int


def connect(db_path: str | None = None) -> lancedb.DBConnection:
    return lancedb.connect(str(db_path or config.LANCEDB_PATH))


def create_table(db: lancedb.DBConnection, overwrite: bool = True):
    mode = "overwrite" if overwrite else "create"
    return db.create_table(TABLE_NAME, schema=CandidateRecord, mode=mode)


def open_table(db: lancedb.DBConnection):
    return db.open_table(TABLE_NAME)


def write_batches(table, rows: Iterable[dict[str, Any]], batch_size: int = 1000) -> int:
    batch: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            table.add(batch)
            total += len(batch)
            batch = []
    if batch:
        table.add(batch)
        total += len(batch)
    return total


def build_indexes(table) -> None:
    print("  building FTS index on career_text and full_text...")
    
    # Pass both columns as a list in a single, unified call
    table.create_fts_index(["career_text", "full_text"], replace=True)

    row_count = table.count_rows()
    if row_count >= MIN_ROWS_FOR_VECTOR_INDEX:
        print(f"  building IVF_PQ vector index over {row_count} rows "
              "(this is the slow step -- CPU-bound regardless of the "
              "device used for embedding, no GPU acceleration here)...")
        table.create_index(
            metric="cosine",
            vector_column_name="embedding",
            num_partitions=256,
            num_sub_vectors=16,
        )
        print("  vector index complete.")
    else:
        print(
            f"Skipping vector index: {row_count} rows < {MIN_ROWS_FOR_VECTOR_INDEX} "
            "required to train IVF_PQ. Vector search will use an unindexed flat scan."
        )


if __name__ == "__main__":
    db = connect()
    table = create_table(db)
    print(f"Created table '{TABLE_NAME}' with schema:")
    for field_name, field in CandidateRecord.model_fields.items():
        print(f"  {field_name}: {field.annotation}")
