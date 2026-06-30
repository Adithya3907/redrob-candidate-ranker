#!/usr/bin/env python3
"""Offline weight tuning via Optuna against gold/gold_set_optuna_v3.jsonl.

Runs only here, on demand, on a developer machine. Never imported by rank.py
or any part of the online pipeline -- the output of this script is a small
set of numbers, copied by hand into config.py once tuning is satisfactory,
replacing the placeholders marked [OPTUNA-TUNED].

    python research/tune_weights.py --candidates ./candidates.jsonl --trials 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ranker import config
from src.ranker.behavioral import score_behavioral
from src.ranker.compose import (
    education_multiplier,
    experience_multiplier,
    impact_verb_multiplier,
    product_company_multiplier,
    tenure_multiplier,
)
from src.ranker.embed import score_pairs
from src.ranker.features import extract_features
from src.ranker.gates import evaluate_gates
from src.ranker.jd_queries import CULTURAL_QUERY, TECHNICAL_QUERY


def load_gold_set(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_candidate_lookup(candidates_path: Path, candidate_ids: set[str]) -> dict[str, dict]:
    lookup = {}
    with open(candidates_path) as f:
        for line in f:
            record = json.loads(line)
            if record["candidate_id"] in candidate_ids:
                lookup[record["candidate_id"]] = record
                if len(lookup) == len(candidate_ids):
                    break
    return lookup


def precompute_static_components(gold_rows: list[dict], candidates_path: Path) -> list[dict]:
    """Computes every part of the score that does not depend on a trial's
    weights, once, up front -- so each Optuna trial is cheap arithmetic
    instead of re-running the cross-encoder and gate checks per trial."""
    candidate_ids = {row["candidate_id"] for row in gold_rows}
    lookup = load_candidate_lookup(candidates_path, candidate_ids)

    technical_pairs = []
    cultural_pairs = []
    precomputed = []

    for gold_row in gold_rows:
        cid = gold_row["candidate_id"]
        candidate = lookup[cid]
        feature_row = extract_features(candidate)
        gate_result = evaluate_gates(candidate, feature_row.career_text)

        technical_pairs.append((TECHNICAL_QUERY, feature_row.career_text))
        cultural_pairs.append((CULTURAL_QUERY, feature_row.career_text))

        precomputed.append({
            "candidate_id": cid,
            "true_relevance": gold_row["true_relevance"],
            "feature_row": feature_row,
            "soft_penalty": gate_result.soft_penalty,
            "exp_mult": experience_multiplier(feature_row.years_exp),
            "pc_mult": product_company_multiplier(feature_row.product_company_ratio),
            "edu_mult": education_multiplier(feature_row.education_tier),
            "tenure_mult": tenure_multiplier(feature_row.longest_tenure_years),
            "impact_mult": impact_verb_multiplier(feature_row.impact_verb_count),
        })

    ce_technical_raw = score_pairs(technical_pairs)
    ce_cultural_raw = score_pairs(cultural_pairs)
    for row, technical, cultural in zip(precomputed, ce_technical_raw, ce_cultural_raw):
        row["ce_technical_raw"] = float(technical)
        row["ce_cultural_raw"] = float(cultural)

    return precomputed


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(value - lo) / (hi - lo) for value in values]


def ndcg_at_k(predicted_scores: list[float], true_relevance: list[float], k: int) -> float:
    pairs = sorted(zip(predicted_scores, true_relevance), key=lambda pair: -pair[0])
    dcg = sum(rel / np.log2(i + 2) for i, (_, rel) in enumerate(pairs[:k]))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(sorted(true_relevance, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def _normalized_quad(trial: optuna.Trial, names: tuple[str, str, str, str], bounds: tuple[float, float]) -> tuple[float, float, float, float]:
    raw = [trial.suggest_float(name, *bounds) for name in names]
    total = sum(raw)
    return tuple(value / total for value in raw)


def _default_objective_params() -> dict:
    """The current config.py constants, expressed as the same parameter
    dict shape objective() expects -- used once, before tuning starts, to
    report a real baseline rather than only ever showing the post-tuning
    number. "We measured X, then improved it to Y" is a materially stronger
    claim than "we ran Optuna and got some weights."
    """
    return {
        "ce_technical_weight": config.CE_TECHNICAL_WEIGHT,
        "w_availability": config.BEHAVIORAL_WEIGHT_AVAILABILITY,
        "w_reliability": config.BEHAVIORAL_WEIGHT_RELIABILITY,
        "w_market_demand": config.BEHAVIORAL_WEIGHT_MARKET_DEMAND,
        "w_platform_trust": config.BEHAVIORAL_WEIGHT_PLATFORM_TRUST,
        "final_relevance_weight": config.FINAL_RELEVANCE_WEIGHT,
    }



def score_with_fixed_params(params: dict, precomputed: list[dict]) -> float:
    """Same scoring logic as objective(), but against a fixed parameter
    dict instead of an Optuna trial -- used to compute the baseline."""
    ce_technical_weight = params["ce_technical_weight"]
    ce_cultural_weight = 1.0 - ce_technical_weight

    raw = [params["w_availability"], params["w_reliability"], params["w_market_demand"], params["w_platform_trust"]]
    total = sum(raw)
    w_availability, w_reliability, w_market, w_trust = (value / total for value in raw)

    final_relevance_weight = params["final_relevance_weight"]
    final_behavioral_weight = 1.0 - final_relevance_weight

    technical_norm = _minmax([row["ce_technical_raw"] for row in precomputed])
    cultural_norm = _minmax([row["ce_cultural_raw"] for row in precomputed])

    predicted_scores = []
    true_relevance = []
    for i, row in enumerate(precomputed):
        ce_score = ce_technical_weight * technical_norm[i] + ce_cultural_weight * cultural_norm[i]
        behavioral = score_behavioral(
            row["feature_row"],
            weight_availability=w_availability,
            weight_reliability=w_reliability,
            weight_market_demand=w_market,
            weight_platform_trust=w_trust,
        )
        
        base = final_relevance_weight * ce_score + final_behavioral_weight * behavioral.behavioral_score
        raw_penalized = base * row["exp_mult"] * row["pc_mult"] * row["edu_mult"] * row["tenure_mult"]
        impact_bonus = (1.0 - raw_penalized) * (row["impact_mult"] - 1.0)
        final_score = raw_penalized + impact_bonus
        final_score = max(0.0, final_score - row["soft_penalty"] * config.SOFT_PENALTY_SCALE)

        predicted_scores.append(final_score)
        true_relevance.append(row["true_relevance"])

    return ndcg_at_k(predicted_scores, true_relevance, k=len(precomputed))


def objective(trial: optuna.Trial, precomputed: list[dict]) -> float:
    ce_technical_weight = trial.suggest_float("ce_technical_weight", 0.5, 0.9)
    ce_cultural_weight = 1.0 - ce_technical_weight

    w_availability, w_reliability, w_market, w_trust = _normalized_quad(
        trial,
        ("w_availability", "w_reliability", "w_market_demand", "w_platform_trust"),
        (0.05, 0.6),
    )

    final_relevance_weight = trial.suggest_float("final_relevance_weight", 0.4, 0.8)
    final_behavioral_weight = 1.0 - final_relevance_weight

    technical_norm = _minmax([row["ce_technical_raw"] for row in precomputed])
    cultural_norm = _minmax([row["ce_cultural_raw"] for row in precomputed])

    predicted_scores = []
    true_relevance = []

    for i, row in enumerate(precomputed):
        ce_score = ce_technical_weight * technical_norm[i] + ce_cultural_weight * cultural_norm[i]

        behavioral = score_behavioral(
            row["feature_row"],
            weight_availability=w_availability,
            weight_reliability=w_reliability,
            weight_market_demand=w_market,
            weight_platform_trust=w_trust,
        )

        base = final_relevance_weight * ce_score + final_behavioral_weight * behavioral.behavioral_score
        raw_penalized = base * row["exp_mult"] * row["pc_mult"] * row["edu_mult"] * row["tenure_mult"]
        impact_bonus = (1.0 - raw_penalized) * (row["impact_mult"] - 1.0)
        final_score = raw_penalized + impact_bonus
        final_score = max(0.0, final_score - row["soft_penalty"] * config.SOFT_PENALTY_SCALE)

        predicted_scores.append(final_score)
        true_relevance.append(row["true_relevance"])

    return ndcg_at_k(predicted_scores, true_relevance, k=len(precomputed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--trials", type=int, default=200)
    args = parser.parse_args()

    gold_rows = load_gold_set(config.GOLD_SET_OPTUNA_PATH)
    print(f"Loaded {len(gold_rows)} gold-set rows. Precomputing static components...")
    precomputed = precompute_static_components(gold_rows, Path(args.candidates))

    baseline_ndcg = score_with_fixed_params(_default_objective_params(), precomputed)
    print(f"\nBaseline NDCG (current config.py defaults, untuned): {baseline_ndcg:.4f}")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, precomputed), n_trials=args.trials)

    print(f"\nBaseline NDCG (untuned):  {baseline_ndcg:.4f}")
    print(f"Best NDCG ({args.trials} Optuna trials): {study.best_value:.4f}")
    print(f"Improvement: {study.best_value - baseline_ndcg:+.4f}")
    print("\nBest params -- copy these into config.py, replacing the [OPTUNA-TUNED] placeholders:")
    #for key, value in study.best_params.items():
    #    print(f"  {key} = {value:.4f}")
    best = study.best_params
    
    # 1. Normalize the behavioral weights (so they sum to 1.0)
    raw_beh = [
        best["w_availability"], 
        best["w_reliability"], 
        best["w_market_demand"], 
        best["w_platform_trust"]
    ]
    total_beh = sum(raw_beh)
    w_avail, w_rel, w_market, w_trust = [v / total_beh for v in raw_beh]
    
    # 2. Calculate the complementary weights
    ce_tech = best["ce_technical_weight"]
    ce_cult = 1.0 - ce_tech
    final_rel = best["final_relevance_weight"]
    final_beh = 1.0 - final_rel
    
    # 3. Print the ACTUAL values for config.py
    print(f"  CE_TECHNICAL_WEIGHT = {ce_tech:.4f}")
    print(f"  CE_CULTURAL_WEIGHT = {ce_cult:.4f}")
    print(f"  BEHAVIORAL_WEIGHT_AVAILABILITY = {w_avail:.4f}")
    print(f"  BEHAVIORAL_WEIGHT_RELIABILITY = {w_rel:.4f}")
    print(f"  BEHAVIORAL_WEIGHT_MARKET_DEMAND = {w_market:.4f}")
    print(f"  BEHAVIORAL_WEIGHT_PLATFORM_TRUST = {w_trust:.4f}")
    print(f"  FINAL_RELEVANCE_WEIGHT = {final_rel:.4f}")
    print(f"  FINAL_BEHAVIORAL_WEIGHT = {final_beh:.4f}")


if __name__ == "__main__":
    main()
