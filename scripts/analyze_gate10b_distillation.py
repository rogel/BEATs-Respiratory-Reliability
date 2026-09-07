#!/usr/bin/env python3
"""Analyze the frozen Gate 10B distilled-student experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from respiratory_sound.gate10b import distillation_decision

DOMAINS = ("icbhi2017", "sprsound2022")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--student-run", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--teacher-runs", type=Path, nargs=6, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=4_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_731)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_run(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, pd.DataFrame]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    configuration = json.loads(
        (run_dir / "configuration.json").read_text(encoding="utf-8")
    )
    if bool(configuration["locked_test_accessed"]):
        raise ValueError(f"Locked test was accessed in {run_dir}")
    frames = {
        domain: pd.read_csv(
            run_dir / f"best_validation_predictions_{domain}.csv"
        )
        for domain in DOMAINS
    }
    return summary, configuration, frames


def _teacher_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    base: pd.DataFrame | None = None
    columns = []
    for index, frame in enumerate(frames):
        column = f"probability_1_teacher_{index}"
        selected = frame[["sample_id", "target", "probability_1"]].rename(
            columns={"probability_1": column}
        )
        if base is None:
            base = selected
        else:
            base = base.merge(
                selected,
                on=["sample_id", "target"],
                how="inner",
                validate="one_to_one",
            )
        columns.append(column)
    if base is None or len(base) != len(frames[0]):
        raise ValueError("Teacher frames do not contain identical samples")
    base["probability_1"] = base[columns].mean(axis=1)
    base["prediction"] = (base["probability_1"] >= 0.5).astype(int)
    return base[["sample_id", "target", "probability_1", "prediction"]]


def _average_score(target: np.ndarray, prediction: np.ndarray) -> float:
    normal = target == 0
    adventitious = target == 1
    if not normal.any() or not adventitious.any():
        raise ValueError("Both classes are required")
    return float(
        (
            (prediction[normal] == 0).mean()
            + (prediction[adventitious] == 1).mean()
        )
        / 2.0
    )


def _pair_with_patients(
    student: pd.DataFrame,
    control: pd.DataFrame,
    patient_lookup: pd.DataFrame,
) -> pd.DataFrame:
    required = {"sample_id", "target", "prediction", "probability_1"}
    for frame in (student, control):
        if not required.issubset(frame):
            raise ValueError("Prediction file is missing required columns")
        expected = (frame["probability_1"].to_numpy() >= 0.5).astype(int)
        if not np.array_equal(expected, frame["prediction"].to_numpy(dtype=int)):
            raise ValueError("Stored predictions do not use threshold 0.5")
    paired = student[list(required)].merge(
        control[list(required)],
        on=["sample_id", "target"],
        suffixes=("_student", "_control"),
        validate="one_to_one",
    )
    if len(paired) != len(student) or len(paired) != len(control):
        raise ValueError("Student and control predictions are not fully paired")
    paired = paired.merge(
        patient_lookup,
        on="sample_id",
        how="left",
        validate="many_to_one",
    )
    if paired["patient_id"].isna().any():
        raise ValueError("Patient lookup is incomplete")
    return paired


def _bootstrap_gain(
    paired: pd.DataFrame,
    *,
    iterations: int,
    random_generator: np.random.Generator,
) -> np.ndarray:
    patient_values = paired["patient_id"].astype(str).to_numpy()
    patients = np.unique(patient_values)
    rows_by_patient = {
        patient: np.flatnonzero(patient_values == patient)
        for patient in patients
    }
    target = paired["target"].to_numpy(dtype=int)
    student = paired["prediction_student"].to_numpy(dtype=int)
    control = paired["prediction_control"].to_numpy(dtype=int)
    gains = np.empty(iterations, dtype=np.float64)
    completed = 0
    while completed < iterations:
        sampled_patients = random_generator.choice(
            patients,
            size=len(patients),
            replace=True,
        )
        indices = np.concatenate(
            [rows_by_patient[str(patient)] for patient in sampled_patients]
        )
        sampled_target = target[indices]
        if not (
            np.any(sampled_target == 0)
            and np.any(sampled_target == 1)
        ):
            continue
        gains[completed] = (
            _average_score(sampled_target, student[indices])
            - _average_score(sampled_target, control[indices])
        )
        completed += 1
    return gains


def _bootstrap_summary(gains: np.ndarray) -> dict[str, Any]:
    return {
        "mean_gain": float(gains.mean()),
        "confidence_interval_95": [
            float(np.quantile(gains, 0.025)),
            float(np.quantile(gains, 0.975)),
        ],
        "probability_gain_above_zero": float(np.mean(gains > 0.0)),
    }


def _binary_kl(teacher: np.ndarray, student: np.ndarray) -> float:
    epsilon = 1.0e-8
    teacher = np.clip(teacher, epsilon, 1.0 - epsilon)
    student = np.clip(student, epsilon, 1.0 - epsilon)
    return float(
        np.mean(
            teacher * np.log(teacher / student)
            + (1.0 - teacher)
            * np.log((1.0 - teacher) / (1.0 - student))
        )
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations < 100:
        raise ValueError("At least 100 bootstrap iterations are required")
    student_summary, student_config, student_frames = _load_run(
        args.student_run
    )
    control_summary, control_config, control_frames = _load_run(
        args.control_run
    )
    if int(student_config["seed"]) != int(control_config["seed"]):
        raise ValueError("Student and control seed mismatch")
    if not bool(student_config["distillation"]["enabled"]):
        raise ValueError("Student configuration did not enable distillation")
    if bool(control_config.get("distillation", {}).get("enabled", False)):
        raise ValueError("Matched control unexpectedly enabled distillation")
    if student_config["sampling_mode"] != control_config["sampling_mode"]:
        raise ValueError("Student and control sampling-mode mismatch")

    teacher_by_domain = {domain: [] for domain in DOMAINS}
    for teacher_run in args.teacher_runs:
        _, _, frames = _load_run(teacher_run)
        for domain in DOMAINS:
            teacher_by_domain[domain].append(frames[domain])
    teacher_frames = {
        domain: _teacher_frame(teacher_by_domain[domain])
        for domain in DOMAINS
    }
    teacher_scores = {}
    alignment_by_domain = {}
    for domain in DOMAINS:
        teacher = teacher_frames[domain]
        target = teacher["target"].to_numpy(dtype=int)
        teacher_scores[domain] = _average_score(
            target,
            teacher["prediction"].to_numpy(dtype=int),
        )
        student_teacher = teacher.merge(
            student_frames[domain][["sample_id", "target", "probability_1"]],
            on=["sample_id", "target"],
            suffixes=("_teacher", "_student"),
            validate="one_to_one",
        )
        control_teacher = teacher.merge(
            control_frames[domain][["sample_id", "target", "probability_1"]],
            on=["sample_id", "target"],
            suffixes=("_teacher", "_control"),
            validate="one_to_one",
        )
        alignment_by_domain[domain] = {
            "student_binary_kl": _binary_kl(
                student_teacher["probability_1_teacher"].to_numpy(dtype=float),
                student_teacher["probability_1_student"].to_numpy(dtype=float),
            ),
            "control_binary_kl": _binary_kl(
                control_teacher["probability_1_teacher"].to_numpy(dtype=float),
                control_teacher["probability_1_control"].to_numpy(dtype=float),
            ),
        }
    teacher_scores["worst_database"] = min(
        teacher_scores[domain] for domain in DOMAINS
    )
    teacher_scores["mean_database"] = float(
        np.mean([teacher_scores[domain] for domain in DOMAINS])
    )
    student_kl = float(
        np.mean(
            [
                alignment_by_domain[domain]["student_binary_kl"]
                for domain in DOMAINS
            ]
        )
    )
    control_kl = float(
        np.mean(
            [
                alignment_by_domain[domain]["control_binary_kl"]
                for domain in DOMAINS
            ]
        )
    )
    teacher_alignment = {
        "per_domain": alignment_by_domain,
        "student_equal_database_mean_binary_kl": student_kl,
        "control_equal_database_mean_binary_kl": control_kl,
        "relative_reduction": (control_kl - student_kl) / control_kl,
    }

    manifest = pd.read_csv(args.manifest, dtype={"patient_id": str})
    validation = manifest.loc[
        manifest["protocol_role"].eq("validation_select"),
        ["sample_id", "patient_id", "dataset"],
    ]
    random_generator = np.random.default_rng(args.bootstrap_seed)
    bootstrap: dict[str, Any] = {
        "per_domain": {},
        "iterations": args.bootstrap_iterations,
        "seed": args.bootstrap_seed,
    }
    gain_samples = []
    pairing_audit = {}
    for domain in DOMAINS:
        lookup = validation.loc[
            validation["dataset"].astype(str).eq(domain),
            ["sample_id", "patient_id"],
        ]
        paired = _pair_with_patients(
            student_frames[domain],
            control_frames[domain],
            lookup,
        )
        gains = _bootstrap_gain(
            paired,
            iterations=args.bootstrap_iterations,
            random_generator=random_generator,
        )
        gain_samples.append(gains)
        bootstrap["per_domain"][domain] = _bootstrap_summary(gains)
        pairing_audit[domain] = {
            "events": len(paired),
            "patients": int(paired["patient_id"].nunique()),
            "targets_identical": True,
            "fixed_threshold_predictions_verified": True,
        }
    bootstrap["mean_domain"] = _bootstrap_summary(
        np.mean(np.stack(gain_samples), axis=0)
    )
    decision = distillation_decision(
        student=student_summary,
        control=control_summary,
        teacher_scores=teacher_scores,
        bootstrap=bootstrap,
        teacher_alignment=teacher_alignment,
    )
    result = {
        "gate": "10B-single-seed",
        "status": (
            "gate_10b_single_seed_passed"
            if decision["passed"]
            else "gate_10b_single_seed_failed"
        ),
        "development_only": True,
        "calibration_accessed": False,
        "locked_tests_accessed": False,
        "student_run": str(args.student_run),
        "control_run": str(args.control_run),
        "teacher_runs": [str(path) for path in args.teacher_runs],
        "pairing_audit": pairing_audit,
        "patient_cluster_bootstrap": bootstrap,
        "teacher_alignment": teacher_alignment,
        "decision": decision,
        "freeze": "artifacts/gate10b_sampling_diverse_distillation_freeze.json",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
