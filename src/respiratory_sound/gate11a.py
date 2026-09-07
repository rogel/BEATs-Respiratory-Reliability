"""Frozen protocol and decision helpers for Gate 11A."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DOMAINS = ("icbhi2017", "sprsound2022")
ALLOWED_SEEDS = (20_260_729, 20_260_730, 20_260_731)
TRAIN_ROLE = "train_fit"
VALIDATION_ROLE = "validation_select"
CLASS_NAMES = ("normal", "adventitious")

EXPECTED_ROLE_COUNTS = {
    TRAIN_ROLE: 7_515,
    VALIDATION_ROLE: 1_567,
}
EXPECTED_DOMAIN_ROLE_COUNTS = {
    ("icbhi2017", TRAIN_ROLE): 2_910,
    ("icbhi2017", VALIDATION_ROLE): 554,
    ("sprsound2022", TRAIN_ROLE): 4_605,
    ("sprsound2022", VALIDATION_ROLE): 1_013,
}

SINGLE_SEED_THRESHOLDS = {
    "icbhi_average_score": 0.69,
    "sprsound_average_score": 0.86,
    "two_domain_mean_average_score": 0.79,
    "sensitivity_floor": 0.50,
    "specificity_floor": 0.50,
}
THREE_SEED_THRESHOLDS = {
    "mean_icbhi_average_score": 0.69,
    "mean_sprsound_average_score": 0.86,
    "mean_two_domain_average_score": 0.79,
    "minimum_individual_seed_passes": 2,
    "sensitivity_floor_every_seed_domain": 0.50,
    "specificity_floor_every_seed_domain": 0.50,
}

CRITICAL_CODE_FILES = (
    "scripts/freeze_gate11a_exactmask_fullft.py",
    "scripts/smoke_gate11a_exactmask_fullft.py",
    "scripts/train_gate11a_exactmask_fullft.py",
    "scripts/analyze_gate11a_exactmask_fullft.py",
    "src/respiratory_sound/gate11a.py",
    "src/respiratory_sound/training_state.py",
    "src/respiratory_sound/models/pretrained_audio.py",
    "src/respiratory_sound/data/audio.py",
    "src/respiratory_sound/data/sampling.py",
    "src/respiratory_sound/metrics.py",
    "src/respiratory_sound/training.py",
    "src/respiratory_sound/runtime.py",
)
CRITICAL_EVIDENCE_FILES = (
    "artifacts/gate11a_exactmask_fullft_quality_smoke.json",
    "artifacts/gate9b_exact_token_mask_amendment.json",
    "artifacts/gate9d_multiseed_final.json",
    (
        "runs/gate9b_exactmask_lora_qv_r8_seed20260729/"
        "summary.json"
    ),
    (
        "runs/gate9b_exactmask_lora_qv_r8_seed20260729/"
        "best_validation_predictions_icbhi2017.csv"
    ),
    (
        "runs/gate9b_exactmask_lora_qv_r8_seed20260729/"
        "best_validation_predictions_sprsound2022.csv"
    ),
    (
        "runs/gate9d_domain_class_event_lora_qv_r8_seed20260730/"
        "summary.json"
    ),
    (
        "runs/gate9d_domain_class_event_lora_qv_r8_seed20260730/"
        "best_validation_predictions_icbhi2017.csv"
    ),
    (
        "runs/gate9d_domain_class_event_lora_qv_r8_seed20260730/"
        "best_validation_predictions_sprsound2022.csv"
    ),
    (
        "runs/gate9d_domain_class_event_lora_qv_r8_seed20260731/"
        "summary.json"
    ),
    (
        "runs/gate9d_domain_class_event_lora_qv_r8_seed20260731/"
        "best_validation_predictions_icbhi2017.csv"
    ),
    (
        "runs/gate9d_domain_class_event_lora_qv_r8_seed20260731/"
        "best_validation_predictions_sprsound2022.csv"
    ),
)


def sha256_file(path: Path) -> str:
    """Hash one file without loading large checkpoints fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_python_tree(root: Path) -> str:
    """Hash the path and contents of every Python source file in a tree."""
    files = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not files:
        raise ValueError(f"No Python source files found under {root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def gate11a_hashes(
    root: Path,
    *,
    manifest_path: Path,
    data_config_path: Path,
    model_config_path: Path,
    experiment_config_path: Path,
    checkpoint_path: Path,
    beats_source_dir: Path,
) -> dict[str, str]:
    """Compute every data, configuration, checkpoint, and source hash to freeze."""
    targets = {
        "manifest_sha256": manifest_path,
        "data_config_sha256": data_config_path,
        "model_config_sha256": model_config_path,
        "experiment_config_sha256": experiment_config_path,
        "upstream_checkpoint_sha256": checkpoint_path,
        **{
            f"code:{relative}": root / relative
            for relative in CRITICAL_CODE_FILES
        },
        **{
            f"evidence:{relative}": root / relative
            for relative in CRITICAL_EVIDENCE_FILES
        },
    }
    missing = [str(path) for path in targets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Gate 11A hash targets: {missing}")
    hashes = {name: sha256_file(path) for name, path in targets.items()}
    hashes["beats_source_tree_sha256"] = sha256_python_tree(beats_source_dir)
    return hashes


def _boolean_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    valid = normalized.isin({"true", "false", "1", "0"})
    if not bool(valid.all()):
        raise ValueError("Manifest locked column contains invalid boolean values")
    return normalized.isin({"true", "1"})


def assert_gate11a_protocol(
    manifest: pd.DataFrame,
    *,
    train_role: str,
    validation_role: str,
) -> dict[str, Any]:
    """Enforce the exact Gate 11A data boundary before constructing datasets."""
    if train_role != TRAIN_ROLE or validation_role != VALIDATION_ROLE:
        raise ValueError(
            "Gate 11A is frozen to train_fit and validation_select only"
        )
    required = {
        "sample_id",
        "dataset",
        "patient_id",
        "protocol_role",
        "locked",
        "binary_label_id",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate sample IDs")

    selected = manifest.loc[
        manifest["protocol_role"].isin({TRAIN_ROLE, VALIDATION_ROLE})
    ].copy()
    if bool(_boolean_series(selected["locked"]).any()):
        raise ValueError("Gate 11A selected a locked row")
    if set(selected["dataset"].astype(str)) != set(DOMAINS):
        raise ValueError("Gate 11A requires exactly ICBHI 2017 and SPRSound")
    if set(selected["binary_label_id"].astype(int)) != {0, 1}:
        raise ValueError("Gate 11A requires binary labels 0 and 1")

    role_counts = {
        role: int((selected["protocol_role"] == role).sum())
        for role in (TRAIN_ROLE, VALIDATION_ROLE)
    }
    if role_counts != EXPECTED_ROLE_COUNTS:
        raise ValueError(f"Unexpected Gate 11A role counts: {role_counts}")
    domain_role_counts = {
        (domain, role): int(
            (
                selected["dataset"].astype(str).eq(domain)
                & selected["protocol_role"].eq(role)
            ).sum()
        )
        for domain in DOMAINS
        for role in (TRAIN_ROLE, VALIDATION_ROLE)
    }
    if domain_role_counts != EXPECTED_DOMAIN_ROLE_COUNTS:
        raise ValueError(
            f"Unexpected Gate 11A domain/role counts: {domain_role_counts}"
        )

    train = selected.loc[selected["protocol_role"].eq(TRAIN_ROLE)]
    validation = selected.loc[selected["protocol_role"].eq(VALIDATION_ROLE)]
    train_patients = set(
        zip(train["dataset"], train["patient_id"].astype(str), strict=True)
    )
    validation_patients = set(
        zip(validation["dataset"], validation["patient_id"].astype(str), strict=True)
    )
    overlap = train_patients.intersection(validation_patients)
    if overlap:
        raise ValueError(
            f"Patient leakage between Gate 11A train and validation: {overlap}"
        )
    return {
        "selected_rows": len(selected),
        "role_counts": role_counts,
        "domain_role_counts": {
            f"{domain}:{role}": count
            for (domain, role), count in domain_role_counts.items()
        },
        "train_validation_patient_overlap": 0,
        "calibration_selected": False,
        "locked_selected": False,
    }


def assert_selected_rows(
    rows: pd.DataFrame,
    *,
    expected_role: str,
    expected_domain: str | None = None,
) -> None:
    """Recheck the rows held by a constructed Dataset or output table."""
    required = {
        "sample_id",
        "dataset",
        "protocol_role",
        "locked",
        "binary_label_id",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Selected rows are missing columns: {sorted(missing)}")
    if rows.empty:
        raise ValueError("Gate 11A selected an empty row set")
    if set(rows["protocol_role"].astype(str)) != {expected_role}:
        raise ValueError(
            f"Gate 11A rows crossed the frozen {expected_role} boundary"
        )
    if bool(_boolean_series(rows["locked"]).any()):
        raise ValueError("Gate 11A rows contain locked samples")
    if rows["sample_id"].duplicated().any():
        raise ValueError("Gate 11A selected duplicate sample IDs")
    if not set(rows["binary_label_id"].astype(int)).issubset({0, 1}):
        raise ValueError("Gate 11A rows contain invalid binary labels")
    domains = set(rows["dataset"].astype(str))
    expected_domains = {expected_domain} if expected_domain is not None else set(DOMAINS)
    if domains != expected_domains:
        raise ValueError(
            f"Unexpected Gate 11A domains: found={domains}, expected={expected_domains}"
        )


def _summary_scores(summary: dict[str, Any]) -> dict[str, float]:
    metrics = summary["best_validation_metrics"]
    values = {
        "icbhi_average_score": float(metrics["icbhi2017"]["average_score"]),
        "sprsound_average_score": float(metrics["sprsound2022"]["average_score"]),
        "icbhi_sensitivity": float(metrics["icbhi2017"]["sensitivity"]),
        "icbhi_specificity": float(metrics["icbhi2017"]["specificity"]),
        "sprsound_sensitivity": float(metrics["sprsound2022"]["sensitivity"]),
        "sprsound_specificity": float(metrics["sprsound2022"]["specificity"]),
    }
    values["two_domain_mean_average_score"] = float(
        np.mean(
            [
                values["icbhi_average_score"],
                values["sprsound_average_score"],
            ]
        )
    )
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("Gate 11A summary contains non-finite decision metrics")
    return values


def single_seed_decision(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply every frozen numerical and safety condition to one seed."""
    seed = int(summary["seed"])
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Unexpected Gate 11A seed: {seed}")
    scores = _summary_scores(summary)
    checks = {
        "icbhi_average_score_at_least_0_69": (
            scores["icbhi_average_score"]
            >= SINGLE_SEED_THRESHOLDS["icbhi_average_score"]
        ),
        "sprsound_average_score_at_least_0_86": (
            scores["sprsound_average_score"]
            >= SINGLE_SEED_THRESHOLDS["sprsound_average_score"]
        ),
        "two_domain_mean_average_score_at_least_0_79": (
            scores["two_domain_mean_average_score"]
            >= SINGLE_SEED_THRESHOLDS["two_domain_mean_average_score"]
        ),
        "sensitivity_each_domain_at_least_0_50": all(
            scores[f"{domain}_sensitivity"]
            >= SINGLE_SEED_THRESHOLDS["sensitivity_floor"]
            for domain in ("icbhi", "sprsound")
        ),
        "specificity_each_domain_at_least_0_50": all(
            scores[f"{domain}_specificity"]
            >= SINGLE_SEED_THRESHOLDS["specificity_floor"]
            for domain in ("icbhi", "sprsound")
        ),
        "finite_audit_passed": bool(summary.get("finite_audit_passed", False)),
        "data_boundary_audit_passed": bool(
            summary.get("data_boundary_audit_passed", False)
        ),
        "exact_mask_audit_passed": bool(
            summary.get("exact_mask_audit_passed", False)
        ),
    }
    return {
        "gate": "11A",
        "seed": seed,
        "scores": scores,
        "checks": checks,
        "passed": all(checks.values()),
    }


def three_seed_decision(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen three-seed Gate without using ensemble metrics."""
    if len(summaries) != len(ALLOWED_SEEDS):
        raise ValueError("Gate 11A confirmation requires exactly three summaries")
    by_seed = {int(summary["seed"]): summary for summary in summaries}
    if set(by_seed) != set(ALLOWED_SEEDS):
        raise ValueError(
            f"Gate 11A requires the frozen seeds {list(ALLOWED_SEEDS)}"
        )
    single = [
        single_seed_decision(by_seed[seed])
        for seed in ALLOWED_SEEDS
    ]
    per_seed_scores = {
        seed: _summary_scores(by_seed[seed])
        for seed in ALLOWED_SEEDS
    }
    mean_scores = {
        "icbhi_average_score": float(
            np.mean(
                [
                    per_seed_scores[seed]["icbhi_average_score"]
                    for seed in ALLOWED_SEEDS
                ]
            )
        ),
        "sprsound_average_score": float(
            np.mean(
                [
                    per_seed_scores[seed]["sprsound_average_score"]
                    for seed in ALLOWED_SEEDS
                ]
            )
        ),
    }
    mean_scores["two_domain_mean_average_score"] = float(
        np.mean(
            [
                mean_scores["icbhi_average_score"],
                mean_scores["sprsound_average_score"],
            ]
        )
    )
    standard_deviations = {
        domain: float(
            np.std(
                [
                    per_seed_scores[seed][f"{domain}_average_score"]
                    for seed in ALLOWED_SEEDS
                ],
                ddof=1,
            )
        )
        for domain in ("icbhi", "sprsound")
    }
    standard_deviations["two_domain_mean"] = float(
        np.std(
            [
                per_seed_scores[seed]["two_domain_mean_average_score"]
                for seed in ALLOWED_SEEDS
            ],
            ddof=1,
        )
    )
    metrics = {
        seed: by_seed[seed]["best_validation_metrics"]
        for seed in ALLOWED_SEEDS
    }
    checks = {
        "mean_icbhi_average_score_at_least_0_69": (
            mean_scores["icbhi_average_score"]
            >= THREE_SEED_THRESHOLDS["mean_icbhi_average_score"]
        ),
        "mean_sprsound_average_score_at_least_0_86": (
            mean_scores["sprsound_average_score"]
            >= THREE_SEED_THRESHOLDS["mean_sprsound_average_score"]
        ),
        "mean_two_domain_average_score_at_least_0_79": (
            mean_scores["two_domain_mean_average_score"]
            >= THREE_SEED_THRESHOLDS["mean_two_domain_average_score"]
        ),
        "single_seed_gate_passes_at_least_2": (
            sum(item["passed"] for item in single)
            >= THREE_SEED_THRESHOLDS["minimum_individual_seed_passes"]
        ),
        "all_seed_domain_sensitivity_at_least_0_50": all(
            float(metrics[seed][domain]["sensitivity"])
            >= THREE_SEED_THRESHOLDS[
                "sensitivity_floor_every_seed_domain"
            ]
            for seed in ALLOWED_SEEDS
            for domain in DOMAINS
        ),
        "all_seed_domain_specificity_at_least_0_50": all(
            float(metrics[seed][domain]["specificity"])
            >= THREE_SEED_THRESHOLDS[
                "specificity_floor_every_seed_domain"
            ]
            for seed in ALLOWED_SEEDS
            for domain in DOMAINS
        ),
    }
    return {
        "gate": "11A",
        "seeds": list(ALLOWED_SEEDS),
        "single_seed_decisions": single,
        "mean_scores": mean_scores,
        "sample_standard_deviations": standard_deviations,
        "checks": checks,
        "passed": all(checks.values()),
    }
