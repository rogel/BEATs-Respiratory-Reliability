#!/usr/bin/env python3
"""Analyze one or three frozen Gate 11A full-finetuning runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from respiratory_sound.gate11a import (
    ALLOWED_SEEDS,
    CLASS_NAMES,
    DOMAINS,
    VALIDATION_ROLE,
    assert_selected_rows,
    sha256_file,
    single_seed_decision,
    three_seed_decision,
)
from respiratory_sound.metrics import (
    paired_patient_bootstrap_difference,
    patient_bootstrap_intervals,
    respiratory_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/cross_domain_binary.csv"),
    )
    parser.add_argument(
        "--freeze-file",
        type=Path,
        default=Path("artifacts/gate11a_exactmask_fullft_freeze.json"),
    )
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--lora-reference-artifact",
        type=Path,
        default=Path("artifacts/gate9d_multiseed_final.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_run(
    run_dir: Path,
    *,
    freeze_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    configuration = json.loads(
        (run_dir / "configuration.json").read_text(encoding="utf-8")
    )
    if (
        summary["gate"] != "11A"
        or configuration["gate"] != "11A"
        or summary["freeze_sha256"] != freeze_sha256
        or configuration["freeze_sha256"] != freeze_sha256
    ):
        raise ValueError(f"Run is not tied to the frozen Gate 11A: {run_dir}")
    if (
        summary["calibration_accessed"]
        or summary["locked_tests_accessed"]
        or configuration["calibration_accessed"]
        or configuration["locked_tests_accessed"]
    ):
        raise ValueError(f"Run crossed the Gate 11A data boundary: {run_dir}")
    if int(summary["seed"]) != int(configuration["seed"]):
        raise ValueError(f"Run seed mismatch: {run_dir}")
    return summary, configuration


def _attach_validation_metadata(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    domain: str,
) -> pd.DataFrame:
    required_predictions = {"sample_id", "target", "probability_1"}
    missing = required_predictions.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if predictions["sample_id"].duplicated().any():
        raise ValueError("Prediction table contains duplicate sample IDs")
    metadata = manifest.loc[
        manifest["protocol_role"].eq(VALIDATION_ROLE)
        & manifest["dataset"].astype(str).eq(domain)
    ][
        [
            "sample_id",
            "dataset",
            "patient_id",
            "protocol_role",
            "locked",
            "binary_label_id",
        ]
    ].copy()
    assert_selected_rows(
        metadata,
        expected_role=VALIDATION_ROLE,
        expected_domain=domain,
    )
    merged = metadata.merge(
        predictions[["sample_id", "target", "probability_1"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(metadata) or len(merged) != len(predictions):
        raise ValueError("Prediction table does not exactly match validation_select")
    if not np.array_equal(
        merged["binary_label_id"].to_numpy(dtype=int),
        merged["target"].to_numpy(dtype=int),
    ):
        raise ValueError("Prediction targets do not match the frozen manifest")
    probabilities = merged["probability_1"].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or not ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    ):
        raise ValueError("Prediction probabilities must be finite and in [0, 1]")
    return merged.sort_values("sample_id").reset_index(drop=True)


def _ensemble_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    if not tables:
        raise ValueError("At least one prediction table is required")
    reference = tables[0][
        [
            "sample_id",
            "dataset",
            "patient_id",
            "binary_label_id",
        ]
    ].copy()
    probabilities = []
    for index, table in enumerate(tables):
        aligned = reference[["sample_id"]].merge(
            table[["sample_id", "probability_1"]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
        if len(aligned) != len(reference):
            raise ValueError("Seed prediction tables are not fully paired")
        probabilities.append(
            aligned["probability_1"].to_numpy(dtype=float)
        )
        if index:
            metadata = table[
                [
                    "sample_id",
                    "dataset",
                    "patient_id",
                    "binary_label_id",
                ]
            ]
            if not reference.equals(metadata.reset_index(drop=True)):
                raise ValueError("Seed prediction metadata differs")
    reference["probability_1"] = np.mean(
        np.stack(probabilities),
        axis=0,
    )
    reference["prediction"] = (
        reference["probability_1"].to_numpy() >= 0.5
    ).astype(int)
    return reference


def _table_metrics(table: pd.DataFrame) -> dict[str, Any]:
    targets = table["binary_label_id"].to_numpy(dtype=int)
    predictions = table["prediction"].to_numpy(dtype=int)
    probabilities = table["probability_1"].to_numpy(dtype=float)
    metrics = respiratory_metrics(targets, predictions, CLASS_NAMES)
    metrics["auroc"] = float(roc_auc_score(targets, probabilities))
    return metrics


def _paired_mean_domain_bootstrap(
    full_ft: dict[str, pd.DataFrame],
    reference: dict[str, pd.DataFrame],
    *,
    iterations: int,
    seed: int,
    confidence: float,
) -> dict[str, float]:
    random_generator = np.random.default_rng(seed)
    prepared: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        left = full_ft[domain].rename(
            columns={"probability_1": "full_probability_1"}
        )
        right = reference[domain].rename(
            columns={"probability_1": "reference_probability_1"}
        )
        paired = left.merge(
            right[["sample_id", "reference_probability_1"]],
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(left) or len(paired) != len(right):
            raise ValueError("Full-FT and LoRA predictions are not fully paired")
        patients = paired["patient_id"].astype(str).to_numpy()
        unique_patients = np.unique(patients)
        prepared[domain] = {
            "table": paired,
            "patients": unique_patients,
            "indices": {
                patient: np.flatnonzero(patients == patient)
                for patient in unique_patients
            },
        }

    def domain_gain(domain: str, indices: np.ndarray) -> float:
        table = prepared[domain]["table"]
        targets = table["binary_label_id"].to_numpy(dtype=int)[indices]
        full_predictions = (
            table["full_probability_1"].to_numpy(dtype=float)[indices] >= 0.5
        ).astype(int)
        reference_predictions = (
            table["reference_probability_1"].to_numpy(dtype=float)[indices] >= 0.5
        ).astype(int)
        full_score = respiratory_metrics(
            targets,
            full_predictions,
            CLASS_NAMES,
        )["average_score"]
        reference_score = respiratory_metrics(
            targets,
            reference_predictions,
            CLASS_NAMES,
        )["average_score"]
        return float(full_score - reference_score)

    samples: list[float] = []
    for _ in range(iterations):
        gains = []
        for domain in DOMAINS:
            patients = prepared[domain]["patients"]
            sampled = random_generator.choice(
                patients,
                size=len(patients),
                replace=True,
            )
            indices = np.concatenate(
                [prepared[domain]["indices"][patient] for patient in sampled]
            )
            gains.append(domain_gain(domain, indices))
        samples.append(float(np.mean(gains)))
    point = float(
        np.mean(
            [
                domain_gain(
                    domain,
                    np.arange(len(prepared[domain]["table"])),
                )
                for domain in DOMAINS
            ]
        )
    )
    alpha = (1.0 - confidence) / 2.0
    values = np.asarray(samples)
    return {
        "estimate": point,
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1.0 - alpha)),
        "probability_full_ft_greater_than_lora": float(
            np.mean(values > 0)
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = _resolve(root, args.manifest).resolve()
    freeze_path = _resolve(root, args.freeze_file).resolve()
    output_path = _resolve(root, args.output).resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze["gate"] != "11A"
        or freeze["status"] != "sealed_before_formal_validation"
    ):
        raise ValueError("Gate 11A freeze is not sealed")
    freeze_sha256 = sha256_file(freeze_path)
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})

    run_dirs = [_resolve(root, path).resolve() for path in args.run_dirs]
    if len(run_dirs) not in {1, 3}:
        raise ValueError("Gate 11A analysis requires one or three runs")
    loaded = [
        _load_run(run_dir, freeze_sha256=freeze_sha256)
        for run_dir in run_dirs
    ]
    summaries = [item[0] for item in loaded]
    seeds = [int(summary["seed"]) for summary in summaries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Gate 11A analysis received duplicate seeds")

    if len(run_dirs) == 1:
        decision = single_seed_decision(summaries[0])
        payload = {
            "gate": "11A",
            "stage": "single_seed",
            "freeze": str(freeze_path.relative_to(root)),
            "freeze_sha256": freeze_sha256,
            "run": str(run_dirs[0].relative_to(root)),
            "decision": decision,
            "recommendation": (
                "continue_remaining_seeds"
                if decision["passed"]
                else "stop_route_unless_reproducible_implementation_error"
            ),
            "calibration_accessed": False,
            "locked_tests_accessed": False,
        }
        _atomic_json(payload, output_path)
        print(json.dumps(payload, indent=2))
        return

    if set(seeds) != set(ALLOWED_SEEDS):
        raise ValueError(f"Gate 11A requires seeds {list(ALLOWED_SEEDS)}")
    run_by_seed = {
        int(summary["seed"]): run_dir
        for summary, run_dir in zip(summaries, run_dirs, strict=True)
    }
    summary_by_seed = {
        int(summary["seed"]): summary
        for summary in summaries
    }
    ordered_summaries = [
        summary_by_seed[seed]
        for seed in ALLOWED_SEEDS
    ]
    gate_decision = three_seed_decision(ordered_summaries)

    full_seed_tables: dict[str, list[pd.DataFrame]] = {
        domain: []
        for domain in DOMAINS
    }
    for seed in ALLOWED_SEEDS:
        for domain in DOMAINS:
            predictions = pd.read_csv(
                run_by_seed[seed]
                / f"best_validation_predictions_{domain}.csv"
            )
            full_seed_tables[domain].append(
                _attach_validation_metadata(
                    predictions,
                    manifest,
                    domain=domain,
                )
            )
    full_ensemble = {
        domain: _ensemble_tables(full_seed_tables[domain])
        for domain in DOMAINS
    }

    reference_artifact_path = _resolve(
        root,
        args.lora_reference_artifact,
    ).resolve()
    reference_artifact = json.loads(
        reference_artifact_path.read_text(encoding="utf-8")
    )
    reference_runs = [
        (root / relative).resolve()
        for relative in reference_artifact["balanced_runs"]
    ]
    if len(reference_runs) != 3:
        raise ValueError("Matched LoRA reference must contain three runs")
    reference_seed_tables: dict[str, list[pd.DataFrame]] = {
        domain: []
        for domain in DOMAINS
    }
    reference_seeds = []
    for run_dir in reference_runs:
        summary = json.loads(
            (run_dir / "summary.json").read_text(encoding="utf-8")
        )
        reference_seeds.append(int(summary["seed"]))
        for domain in DOMAINS:
            predictions = pd.read_csv(
                run_dir / f"best_validation_predictions_{domain}.csv"
            )
            reference_seed_tables[domain].append(
                _attach_validation_metadata(
                    predictions,
                    manifest,
                    domain=domain,
                )
            )
    if set(reference_seeds) != set(ALLOWED_SEEDS):
        raise ValueError("Matched LoRA reference uses different seeds")
    lora_ensemble = {
        domain: _ensemble_tables(reference_seed_tables[domain])
        for domain in DOMAINS
    }

    bootstrap = freeze["bootstrap"]
    iterations = int(bootstrap["iterations"])
    bootstrap_seed = int(bootstrap["seed"])
    confidence = float(bootstrap["confidence"])
    absolute_intervals = {}
    paired_differences = {}
    ensemble_metrics = {}
    reference_metrics = {}
    for domain in DOMAINS:
        full = full_ensemble[domain]
        lora = lora_ensemble[domain]
        ensemble_metrics[domain] = _table_metrics(full)
        reference_metrics[domain] = _table_metrics(lora)
        targets = full["binary_label_id"].to_numpy(dtype=int)
        full_predictions = full["prediction"].to_numpy(dtype=int)
        lora_predictions = lora["prediction"].to_numpy(dtype=int)
        patients = full["patient_id"].astype(str).to_numpy()
        absolute_intervals[domain] = patient_bootstrap_intervals(
            targets,
            full_predictions,
            patients,
            iterations=iterations,
            seed=bootstrap_seed,
            confidence=confidence,
            class_names=CLASS_NAMES,
        )
        paired_differences[domain] = paired_patient_bootstrap_difference(
            targets,
            full_predictions,
            lora_predictions,
            patients,
            metric="average_score",
            iterations=iterations,
            seed=bootstrap_seed,
            confidence=confidence,
            class_names=CLASS_NAMES,
        )
    paired_differences["two_domain_mean"] = _paired_mean_domain_bootstrap(
        full_ensemble,
        lora_ensemble,
        iterations=iterations,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    superiority_supported = all(
        paired_differences[key]["estimate"] > 0
        and paired_differences[key]["lower"] > 0
        for key in (*DOMAINS, "two_domain_mean")
    )

    payload = {
        "gate": "11A",
        "stage": "three_seed_confirmation",
        "status": (
            "gate11a_passed"
            if gate_decision["passed"]
            else "gate11a_failed"
        ),
        "freeze": str(freeze_path.relative_to(root)),
        "freeze_sha256": freeze_sha256,
        "runs": [
            str(run_by_seed[seed].relative_to(root))
            for seed in ALLOWED_SEEDS
        ],
        "gate_decision": gate_decision,
        "primary_full_ft_probability_ensemble": {
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "metrics": ensemble_metrics,
            "patient_cluster_absolute_intervals": absolute_intervals,
        },
        "matched_lora_probability_ensemble": {
            "artifact": str(reference_artifact_path.relative_to(root)),
            "artifact_sha256": sha256_file(reference_artifact_path),
            "metrics": reference_metrics,
        },
        "paired_full_ft_minus_lora_bootstrap": {
            "iterations": iterations,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "results": paired_differences,
            "full_ft_superiority_supported_in_each_domain_and_mean": (
                superiority_supported
            ),
            "claim_rule": (
                "If false, report a performance-efficiency trade-off and "
                "do not claim full-FT superiority."
            ),
        },
        "recommendation": (
            "freeze_classifier_and_design_calibration_selective_evaluation"
            if gate_decision["passed"]
            else "stop_route_without_calibration_or_locked_test"
        ),
        "calibration_accessed": False,
        "locked_tests_accessed": False,
    }
    _atomic_json(payload, output_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
