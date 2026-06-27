# Architecture

This document covers how the Redrob Candidate Ranker works end to end: the
constraints it is designed against, how data moves through it, the rationale
behind each non-obvious decision, and what is still open. Setup and usage
commands are in [README.md](README.md).

## Table of contents

1. [Context and constraints](#1-context-and-constraints)
2. [High-level architecture](#2-high-level-architecture)
3. [Building blocks](#3-building-blocks)
4. [Runtime view](#4-runtime-view)
5. [Cross-cutting concerns](#5-cross-cutting-concerns)
6. [Architecture decisions](#6-architecture-decisions)
7. [Known limitations and open risks](#7-known-limitations-and-open-risks)
8. [Glossary](#8-glossary)

---

## 1. Context and constraints

The system ranks a fixed pool of 100,000 candidate profiles against one job
description (Senior AI Engineer, founding team) and produces a CSV of the
top 100, each row carrying a rank, a score, and a one-to-two-sentence
reasoning string.

Hard constraints, imposed by the hackathon specification:

| Constraint | Limit |
|---|---|
| Wall time for the online ranking step | ≤ 5 minutes |
| Memory | ≤ 16 GB |
| Compute | CPU-only, no GPU |
| Network access during ranking | None |
| Honeypot rate in the top 100 | ≤ 10%, or the submission is disqualified |
| Output shape | Exactly 100 rows, ranks 1–100, scores non-increasing, unique reasoning per row |
| Reproducibility | A single command must regenerate the submission from the repository |

Embedding and indexing 100,000 candidates does not fit inside a 5-minute
budget on its own — at roughly 150 sentences per second on CPU, that step
alone takes over 11 minutes. The two-phase split described in
[§2](#2-high-level-architecture) exists specifically to move that cost
outside the timed window.

---

## 2. High-level architecture

```
                    ┌──────────────────────────────────────────┐
                    │   PHASE A — Offline pre-build            │
                    │   No time limit · run once before submit │
                    │                                          │
  candidates.jsonl  │   stream → gate-check → feature-extract  │
       ──────────►  │        → embed → write to LanceDB        │
                    │                                          │
                    └────────────────────┬─────────────────────┘
                                         │
                                  artifacts/lancedb/
                                  (baked into the Docker image)
                                         │
                    ┌────────────────────▼───────────────────────┐
                    │   PHASE B — Online ranking                 │
                    │   ≤ 5 min · ≤ 16 GB · CPU-only · no network│
                    │                                            │
                    │   recall → rerank → behavioral score       │
                    │        → composite score → reasoning       │
                    │                                            │
                    └────────────────────┬───────────────────────┘
                                         │
                                  WhiteNoise.csv
                                  (top 100, ranked)
```

Phase A pays the one fixed, expensive cost — embedding the full pool — with
no clock running. Phase B reads the result of that build and performs only
bounded, cheap work, which is what makes the 5-minute budget achievable.

---

## 3. Building blocks

Each module owns one stage. The table below follows the actual data flow,
top to bottom.

| Module | Stage | Responsibility |
|---|---|---|
| `ingest.py` | A-0 | Streams `candidates.jsonl[.gz]` as an iterator; the full file is never loaded into memory at once. |
| `features.py` | A-1 | Builds the two retrieval text columns (`career_text`, `full_text`) and 30+ structured columns per candidate. |
| `gates.py` | A-1 / A-2 | Flags structural honeypots and JD hard disqualifiers as hard exclusions, plus four soft signals scored as penalties. See [§6.1](#61-literal-title-matching-for-stagnant-title-detection-not-normalized). |
| `embed.py` | A-3 | Loads BGE-small and the cross-encoder once each; exposes passage/query encoding and pairwise scoring. |
| `index.py` | A-4 | LanceDB schema, table writes, and FTS + vector index construction. |
| `jd_queries.py` | — | Job-description query text used across three stages, kept separate from `config.py` because it is prose, not a tunable constant. |
| `recall.py` | B-2–B-4 | Dual-channel full-text recall plus vector ANN recall, unioned into one shortlist with a combined `recall_score`. |
| `filters.py` | B-5 | Deterministic shortlist cap bounding how many candidates reach the cross-encoder. |
| `rerank.py` | B-6 | Dual-pass cross-encoder scoring — technical fit, cultural fit — with a runtime circuit breaker. |
| `behavioral.py` | B-7 | Combines the 23 `redrob_signals` fields into 4 interpretable super-features. |
| `compose.py` | B-8 | Fuses relevance and behavioral scores, applies role-fit multipliers, computes the per-candidate dominant penalty. |
| `reasoning.py` | B-10 | Generates the reasoning string from structured fields and detected keywords only. |
| `pipeline.py` | B-1–B-11 | Orchestrates every stage above into `run_pipeline(table) -> list[RankedCandidate]`. |
| `validate.py` | B-11 | In-pipeline format check, run before the CSV is written. |
| `config.py` | — | Every tuned constant, in one place. |

Entry points (scripts, not library code):

| Script | When to run it |
|---|---|
| `rank.py` | The official, timed reproduction command. |
| `scripts/build_index.py` | Phase A — once, before submitting. |
| `research/tune_weights.py` | On demand, to re-tune weights against the gold set. Not imported by `rank.py`. |
| `sandbox/app.py` | The required hosted demo. Builds a small temporary index from an uploaded sample. |

---

## 4. Runtime view

### 4.1 Phase A — offline pre-build

```
for each batch of ~500 candidates, streamed (never all 100K in memory):
    extract_features()    → career_text, full_text, 30+ structured columns
    evaluate_gates()      → is_excluded (0.0 / -50.0 / -99.0), soft_penalty, soft_flags
    encode_passages()     → 384-dim embedding per candidate
    write_batches()       → append to LanceDB

once all batches are written:
    build_indexes()       → FTS index on career_text, FTS index on full_text,
                             vector index (IVF_PQ) on embedding
```

Runtime is dominated by embedding (~150 sentences/sec on CPU, ~11 minutes for
100K candidates). This step has no time limit and runs once.

### 4.2 Phase B — online ranking

```
[recall]      FTS search × 2 channels + vector ANN search, all filtered to
              is_excluded == eligible at query time → ~1,000-2,200 candidates,
              each with a recall_score and a found_via provenance tag
[cap]         keep the top SHORTLIST_HARD_CAP (1,200) by recall_score
[rerank]      cross-encoder pass 1 (technical) → pass 2 (cultural), unless the
              circuit breaker has already fired (§6.4)
[behavioral]  23 redrob_signals → 4 super-features → one behavioral_score
[compose]     fuse relevance + behavioral, apply role-fit multipliers
[rank]        sort, deterministic tie-break, take the top 100
[reasoning]   one sentence per candidate, structured fields and detected
              keywords only
[validate]    in-pipeline self-check — fails immediately if anything is
              wrong, before any CSV is written
```

Wall time is dominated by the cross-encoder stage. Every other stage is
sub-second at this scale, since FTS and vector queries against a pre-built
index do not get meaningfully slower as the result size grows.

---

## 5. Cross-cutting concerns

### 5.1 Honeypot detection is structural, not keyword-based

Every exclusion rule is a cross-field arithmetic or structural check —
duration math, skill-claim-versus-usage-time, application/view-count
inversion, literal title repetition — never a keyword match. The dataset's
keyword-stuffing trap (an HR Manager profile listing nine AI-sounding
skills) is specifically designed to defeat keyword-based filters; structural
checks are not affected by vocabulary at all.

### 5.2 Reasoning generation cannot hallucinate, by construction

`reasoning.py` never reads `profile.summary` or any other free-text field
directly. Every clause traces back to a value computed earlier in the
pipeline — a structured column, or a FlashText-detected keyword from a fixed
vocabulary. This is a structural property of how the templates are built,
not a guideline the code happens to follow: there is no code path through
which uncontrolled text could be injected.

### 5.3 Fail fast inside the pipeline itself

`validate.py`'s checks run automatically inside `rank.py`, immediately
before the CSV is written: exact row count, ranks 1–100 with no gaps or
duplicates, non-increasing scores, no empty or duplicate reasoning strings.
A failure raises immediately rather than producing a malformed file that
would only be caught later by `validate_submission.py`, or by a reviewer.
Ad hoc workflow checks — running against a small slice first, eyeballing the
top 10, timing a real run — are a development practice, not part of this
repository; the codebase stays limited to what actually produces the result.

### 5.4 Defense in depth on the one filter that cannot be wrong

Every `.where()` call in `recall.py` passes `prefilter=True` explicitly,
matching LanceDB's documented approach for combining a filter with
full-text search. `recall.py` separately re-checks `is_excluded` in Python
after every query, as an independent second layer. This is motivated by a
documented LanceDB issue ([lancedb/lancedb#1656](https://github.com/lancedb/lancedb/issues/1656))
in which a `.where()` filter was silently ignored when combined with FTS.
To be precise about scope: the reported trigger condition is a scalar index
on the filtered column, which this schema does not build — only FTS and
vector indexes exist here, so the cited issue is not a confirmed match to
this exact configuration. The redundant check is kept regardless, because
the cost of being wrong (a honeypot reaching the final ranking, which is an
automatic disqualification) is high enough to justify a second layer even
against a risk that is plausible rather than confirmed here.

---

## 6. Architecture decisions

Each entry states the decision, the reasoning behind it, and what was
considered instead.

### 6.1 Literal title matching for stagnant-title detection, not normalized

**Decision:** flag `stagnant_title_3plus` only on an exact, unmodified title
string repeated across 3+ employers, without stripping seniority prefixes
such as "Senior" or "Staff" first.

**Reasoning:** a normalized version — treating "Senior MLE" and "Staff MLE"
as equivalent — flags a candidate who was genuinely promoted and then took a
same-level role elsewhere, which is an ordinary career, not a structural
anomaly. Verified against the real data: the normalized rule fired on 202
candidates; the literal rule fires on 6, all of which show zero title
movement across 3+ employers over 5+ years.

**Alternative considered:** the normalized version was implemented first and
rejected after producing a confirmed false positive on what independent
verification identified as the strongest candidate in the entire pool.

### 6.2 Salary-range inversion is a soft signal, not a hard exclusion

**Decision:** `min > max` on expected salary range contributes a penalty and
never excludes a candidate outright.

**Reasoning:** this condition holds for roughly one in five candidates in
the full dataset — far too common to be a deliberately placed trap, and
consistent with the two values simply not being sorted at the
data-generation step. A hard exclusion here would discard real, otherwise
strong candidates for a reason unrelated to profile authenticity.

### 6.3 The cross-encoder runs twice, against two separate query chunks

**Decision:** Stage 6 scores every shortlisted candidate against a technical
query and a cultural query independently, then fuses the two scores with an
explicit, named weight, rather than scoring once against a single combined
query.

**Reasoning:** a single blended query forces the model to average two
different kinds of fit into one number that cannot be decomposed afterward —
there is no way to distinguish a candidate who is technically strong but
culturally unclear from one who is moderate on both axes. Two separate
scores remain auditable, independently weighted, and independently citable
in the reasoning text. The cost — twice the inference calls — is covered by
the runtime circuit breaker in [§6.4](#64-a-runtime-circuit-breaker-not-a-static-timing-budget)
rather than a static time estimate.

### 6.4 A runtime circuit breaker, not a static timing budget

**Decision:** Stage 6 checks elapsed time after the first cross-encoder pass
and skips the second pass, substituting a neutral score, if too much of the
budget is already spent, rather than relying on a fixed throughput estimate.

**Reasoning:** a static estimate ("cross-encoders run at ~9 pairs/sec, so N
candidates take T seconds") is a guess about hardware that has not been
measured. A runtime check degrades gracefully under real conditions instead
of risking a hard timeout when the guess is wrong.

### 6.5 Recall is wide; filtering happens later, on explicit criteria

**Decision:** the recall stage requests 1,000–2,200 candidates across two
FTS channels and one vector channel, and lets explicit downstream rules
decide who survives, rather than narrowing the recall window itself.

**Reasoning:** FTS and vector queries against a pre-built index cost almost
the same regardless of result size. Narrowing recall to save time buys
nothing measurable while risking the silent loss of a strong candidate whose
best evidence sentence ranked outside a too-narrow cutoff in one channel.
The saved budget is spent where it is actually expensive — Stage 6.

### 6.6 Behavioral signals combine additively, then get sigmoid-squashed

**Decision:** the four behavioral super-features — availability,
reliability, market demand, platform trust — are weighted, summed, and
passed through a sigmoid, rather than multiplied together.

**Reasoning:** a multiplicative combination lets a single weak signal
collapse an otherwise strong score disproportionately. An additive
combination with a sigmoid lets a real weakness — a six-month-stale profile,
for example — pull a score down substantially without zeroing out an
otherwise excellent candidate, matching how the gold set's labeled examples
treat that case.

### 6.7 Weight tuning optimizes NDCG directly, against real production code

**Decision:** `research/tune_weights.py` calls the actual `behavioral.py`
and `compose.py` scoring functions, with weight overrides passed as
parameters, and optimizes NDCG@k — the same metric family the hackathon's
own evaluation uses — rather than reimplementing the scoring formula
separately or optimizing a proxy metric.

**Reasoning:** a separate reimplementation can silently drift out of sync
with the production scoring code the moment either one changes. Optimizing
the real evaluation metric against the real production formula means the
reported result answers "how much did tuning help" directly, rather than
answering a related but different question.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| Structural honeypot | A candidate profile that is internally, arithmetically impossible — for example, 14 years of tenure in a role that started 33 months ago. Detected by cross-field math, never by keyword. Hard exclusion. |
| JD hard disqualifier | A real, internally consistent candidate who is explicitly out of scope per the job description — for example, a consulting-only career history. Hard exclusion, distinct from a honeypot. |
| Soft flag | A signal that contributes a penalty but never excludes a candidate outright — for example, salary-range inversion. |
| Recall | The stage that casts a wide net over the eligible pool using inexpensive signals (keyword and vector search) before any expensive scoring runs. |
| Shortlist cap | The deterministic ceiling on how many candidates from recall proceed to the cross-encoder. |
| Cross-encoder | A model that scores a (query, document) pair by reading both together, as opposed to comparing two independently computed embeddings. More accurate and substantially slower per comparison, which is why it only runs on the capped shortlist. |
| `found_via` | Per-candidate provenance: whether a candidate was recalled by keyword search, vector search, or both. Surfaced in the reasoning text only when a candidate was found by vector search alone — direct evidence that the system found something a keyword search would have missed. |
