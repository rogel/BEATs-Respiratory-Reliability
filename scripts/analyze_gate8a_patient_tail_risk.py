#!/usr/bin/env python3
"""Test whether patient tail risk is concentrated and reproducible."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from respiratory_sound.patient_risk import (
    jaccard_index,
    patient_risk_table,
    top_risk_patients,
)

DOMAINS = ("icbhi2017", "sprsound2022")
FAMILIES = ("domain_class_event", "event_random")
SEEDS = (20_260_729, 20_260_730, 20_260_731)
TAIL_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("artifacts/gate6a_predictions/calibration"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/gate8a_patient_risk"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/gate8a_patient_tail_risk_final.json"),
    )
    return parser.parse_args()


def _load_predictions(
    prediction_root: Path,
    family: str,
    domain: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    probability_columns = []
    for seed in SEEDS:
        path = prediction_root / f"{family}_seed{seed}_{domain}.csv"
        frame = pd.read_csv(path, dtype={"patient_id": str})
        required = {"sample_id", "patient_id", "target", "probability_1"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate sample IDs")
        probability_column = f"probability_1_seed{seed}"
        seed_frame = frame[
            ["sample_id", "patient_id", "target", "probability_1"]
        ].rename(columns={"probability_1": probability_column})
        if merged is None:
            merged = seed_frame
        else:
            merged = merged.merge(
                seed_frame,
                on=["sample_id", "patient_id", "target"],
                how="inner",
                validate="one_to_one",
            )
        probability_columns.append(probability_column)
    if merged is None:
        raise RuntimeError("no prediction files loaded")
    merged["probability_1"] = merged[probability_columns].mean(axis=1)
    return merged


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    correlation = float(spearmanr(first, second).statistic)
    return correlation if np.isfinite(correlation) else 0.0


def _analyze_domain(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, set[str]]:
    ensemble = patient_risk_table(frame)
    combined = ensemble.rename(columns={"patient_risk": "ensemble_patient_risk"})
    seed_tables: dict[int, pd.DataFrame] = {}
    seed_tail_sets: dict[int, set[str]] = {}
    for seed in SEEDS:
        probability_column = f"probability_1_seed{seed}"
        seed_table = patient_risk_table(
            frame,
            probability_column=probability_column,
        )
        seed_tables[seed] = seed_table
        seed_tail_sets[seed] = set(top_risk_patients(seed_table, TAIL_FRACTION))
        combined = combined.merge(
            seed_table[["patient_id", "patient_risk"]].rename(
                columns={"patient_risk": f"patient_risk_seed{seed}"}
            ),
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )

    rank_correlations = []
    tail_jaccards = []
    for first_seed, second_seed in itertools.combinations(SEEDS, 2):
        first_column = f"patient_risk_seed{first_seed}"
        second_column = f"patient_risk_seed{second_seed}"
        rank_correlations.append(
            {
                "seeds": [first_seed, second_seed],
                "spearman": _safe_spearman(
                    combined[first_column].to_numpy(dtype=np.float64),
                    combined[second_column].to_numpy(dtype=np.float64),
                ),
            }
        )
        tail_jaccards.append(
            {
                "seeds": [first_seed, second_seed],
                "jaccard": jaccard_index(
                    seed_tail_sets[first_seed],
                    seed_tail_sets[second_seed],
                ),
            }
        )

    ensemble_tail = set(top_risk_patients(ensemble, TAIL_FRACTION))
    tail_mask = ensemble["patient_id"].isin(ensemble_tail)
    tail_risk_ratio = float(
        ensemble.loc[tail_mask, "patient_risk"].mean()
        / ensemble["patient_risk"].mean()
    )
    total_errors = int(ensemble["error_count"].sum())
    tail_errors = int(ensemble.loc[tail_mask, "error_count"].sum())
    tail_error_share = float(tail_errors / total_errors) if total_errors else 0.0
    event_count_risk_spearman = _safe_spearman(
        ensemble["event_count"].to_numpy(dtype=np.float64),
        ensemble["patient_risk"].to_numpy(dtype=np.float64),
    )
    tail_event_mask = frame["patient_id"].astype(str).isin(ensemble_tail)
    tail_targets = sorted(
        frame.loc[tail_event_mask, "target"].astype(int).unique().tolist()
    )
    payload = {
        "patients": int(len(ensemble)),
        "tail_patients": sorted(ensemble_tail),
        "tail_patient_count": int(len(ensemble_tail)),
        "tail_risk_ratio": tail_risk_ratio,
        "tail_error_share": tail_error_share,
        "tail_errors": tail_errors,
        "total_errors": total_errors,
        "median_pairwise_seed_risk_spearman": float(
            np.median([item["spearman"] for item in rank_correlations])
        ),
        "pairwise_seed_risk_spearman": rank_correlations,
        "mean_pairwise_seed_tail_jaccard": float(
            np.mean([item["jaccard"] for item in tail_jaccards])
        ),
        "pairwise_seed_tail_jaccard": tail_jaccards,
        "event_count_risk_spearman": event_count_risk_spearman,
        "absolute_event_count_risk_spearman": abs(event_count_risk_spearman),
        "tail_targets_present": tail_targets,
        "tail_contains_both_classes": tail_targets == [0, 1],
    }
    return payload, combined, ensemble_tail


def main() -> None:
    args = parse_args()
    prediction_root = args.prediction_root.resolve()
    output_root = args.output_root.resolve()
    output_json = args.output_json.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    family_results: dict[str, Any] = {}
    tail_sets: dict[str, dict[str, set[str]]] = {
        family: {} for family in FAMILIES
    }
    for family in FAMILIES:
        family_results[family] = {}
        for domain in DOMAINS:
            frame = _load_predictions(prediction_root, family, domain)
            result, patient_table, tail_set = _analyze_domain(frame)
            family_results[family][domain] = result
            tail_sets[family][domain] = tail_set
            patient_table.to_csv(
                output_root / f"{family}_{domain}_patient_risk.csv",
                index=False,
            )

    cross_family_jaccards = {
        domain: jaccard_index(
            tail_sets["domain_class_event"][domain],
            tail_sets["event_random"][domain],
        )
        for domain in DOMAINS
    }
    primary = family_results["domain_class_event"]
    gate = {
        "tail_risk_ratio_each_domain_at_least_1_5": bool(
            all(primary[domain]["tail_risk_ratio"] >= 1.5 for domain in DOMAINS)
        ),
        "tail_error_share_each_domain_at_least_0_40": bool(
            all(primary[domain]["tail_error_share"] >= 0.4 for domain in DOMAINS)
        ),
        "median_seed_spearman_each_domain_at_least_0_60": bool(
            all(
                primary[domain]["median_pairwise_seed_risk_spearman"] >= 0.6
                for domain in DOMAINS
            )
        ),
        "mean_seed_tail_jaccard_each_domain_at_least_0_40": bool(
            all(
                primary[domain]["mean_pairwise_seed_tail_jaccard"] >= 0.4
                for domain in DOMAINS
            )
        ),
        "cross_family_tail_jaccard_each_domain_at_least_0_50": bool(
            all(cross_family_jaccards[domain] >= 0.5 for domain in DOMAINS)
        ),
        "absolute_event_count_risk_spearman_each_domain_at_most_0_70": bool(
            all(
                primary[domain]["absolute_event_count_risk_spearman"] <= 0.7
                for domain in DOMAINS
            )
        ),
        "tail_contains_both_classes_each_domain": bool(
            all(
                primary[domain]["tail_contains_both_classes"]
                for domain in DOMAINS
            )
        ),
    }
    gate["passed"] = bool(all(gate.values()))
    payload = {
        "status": "gate_8a_passed" if gate["passed"] else "gate_8a_rejected",
        "data_role": "calibration",
        "additional_training": False,
        "validation_select_accessed": False,
        "locked_tests_accessed": False,
        "families": family_results,
        "cross_family_tail_jaccard": cross_family_jaccards,
        "gate": gate,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
