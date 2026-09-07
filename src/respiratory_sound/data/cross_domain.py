"""Unified binary protocol for cross-dataset respiratory-sound experiments."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

DEVELOPMENT_ROLES = ("train_fit", "validation_select", "calibration")
LOCKED_ROLES = ("locked_test", "locked_intra_test", "locked_inter_test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(frame: pd.DataFrame, column: str, values: Iterable[object]) -> np.ndarray:
    return (
        frame[column]
        .value_counts(normalize=True)
        .reindex(list(values), fill_value=0.0)
        .to_numpy(dtype=np.float64)
    )


def patient_partition(
    frame: pd.DataFrame,
    *,
    patient_column: str = "patient_id",
    label_column: str = "binary_label_id",
    domain_column: str | None = None,
    selection_fraction: float = 0.15,
    calibration_fraction: float = 0.15,
    seed: int = 20_260_728,
    attempts: int = 8_192,
) -> tuple[dict[str, str], dict[str, object]]:
    """Create a deterministic patient-exclusive train/selection/calibration split.

    Random search is used only to find a balanced fixed partition. The chosen
    partition is deterministic for a given manifest and seed.
    """
    if frame.empty:
        raise ValueError("Cannot partition an empty frame")
    required = {patient_column, label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required partition columns: {sorted(missing)}")
    if not 0 < selection_fraction < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("Selection and calibration fractions must be between zero and one")
    if selection_fraction + calibration_fraction >= 1:
        raise ValueError("Selection and calibration fractions must sum to less than one")
    if attempts < 1:
        raise ValueError("attempts must be positive")

    patients = np.asarray(sorted(frame[patient_column].astype(str).unique()))
    if len(patients) < 6:
        raise ValueError("At least six patients are required for a three-way split")
    selection_count = max(2, round(len(patients) * selection_fraction))
    calibration_count = max(2, round(len(patients) * calibration_fraction))
    fit_count = len(patients) - selection_count - calibration_count
    if fit_count < 2:
        raise ValueError("The requested fractions leave too few fitting patients")

    labels = sorted(frame[label_column].unique())
    if len(labels) < 2:
        raise ValueError("At least two labels are required")
    overall_prevalence = _distribution(frame, label_column, labels)
    domain_values: list[object] = []
    usable_domain_column = None
    if domain_column is not None:
        if domain_column not in frame.columns:
            raise ValueError(f"Missing domain column: {domain_column}")
        domain_patient_support = (
            frame[[patient_column, domain_column]]
            .drop_duplicates()
            .groupby(domain_column)[patient_column]
            .nunique()
        )
        domain_values = sorted(domain_patient_support[domain_patient_support >= 3].index)
        if len(domain_values) >= 2:
            usable_domain_column = domain_column
            overall_domain = _distribution(frame, domain_column, domain_values)
        else:
            overall_domain = np.asarray([], dtype=np.float64)
    else:
        overall_domain = np.asarray([], dtype=np.float64)

    target_sample_fractions = {
        "train_fit": 1.0 - selection_fraction - calibration_fraction,
        "validation_select": selection_fraction,
        "calibration": calibration_fraction,
    }
    random_generator = np.random.default_rng(seed)
    best: tuple[float, dict[str, str], dict[str, object]] | None = None

    for _ in range(attempts):
        shuffled = random_generator.permutation(patients)
        role_patients = {
            "validation_select": set(shuffled[:selection_count]),
            "calibration": set(
                shuffled[selection_count : selection_count + calibration_count]
            ),
            "train_fit": set(shuffled[selection_count + calibration_count :]),
        }
        role_frames = {
            role: frame.loc[frame[patient_column].astype(str).isin(patient_set)]
            for role, patient_set in role_patients.items()
        }
        if any(
            set(role_frame[label_column].unique()) != set(labels)
            for role_frame in role_frames.values()
        ):
            continue

        score = 0.0
        diagnostics: dict[str, object] = {}
        for role, role_frame in role_frames.items():
            prevalence = _distribution(role_frame, label_column, labels)
            sample_fraction = len(role_frame) / len(frame)
            score += 2.0 * float(np.abs(prevalence - overall_prevalence).sum())
            score += abs(sample_fraction - target_sample_fractions[role])
            domain_distance = 0.0
            if usable_domain_column is not None:
                domain_distribution = _distribution(
                    role_frame,
                    usable_domain_column,
                    domain_values,
                )
                domain_distance = float(
                    np.abs(domain_distribution - overall_domain).sum()
                )
                score += 0.25 * domain_distance
            diagnostics[role] = {
                "patients": len(role_patients[role]),
                "samples": len(role_frame),
                "sample_fraction": sample_fraction,
                "positive_prevalence": float(
                    role_frame[label_column].astype(float).mean()
                ),
                "domain_l1_distance": domain_distance,
            }

        mapping = {
            patient: role
            for role, patient_set in role_patients.items()
            for patient in patient_set
        }
        metadata = {
            "seed": seed,
            "attempts": attempts,
            "objective": score,
            "patient_counts": {
                "train_fit": fit_count,
                "validation_select": selection_count,
                "calibration": calibration_count,
            },
            "overall_positive_prevalence": float(
                frame[label_column].astype(float).mean()
            ),
            "balanced_domains": [str(value) for value in domain_values],
            "roles": diagnostics,
        }
        if best is None or score < best[0]:
            best = (score, mapping, metadata)

    if best is None:
        raise ValueError("Could not find a three-way patient split containing both labels")
    return best[1], best[2]


def _icbhi_rows(
    cycles: pd.DataFrame,
    recordings: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cycles = cycles.copy()
    cycles["source_patient_id"] = cycles["patient_id"].astype(str)
    cycles["binary_label_id"] = (cycles["label_name"] != "normal").astype(int)
    cycles["binary_label_name"] = np.where(
        cycles["binary_label_id"].eq(0),
        "normal",
        "adventitious",
    )
    development = cycles.loc[cycles["strict_split"] == "train"].copy()
    role_mapping, split_metadata = patient_partition(
        development,
        patient_column="source_patient_id",
        label_column="binary_label_id",
        domain_column="device",
        seed=seed,
    )
    cycles["protocol_role"] = cycles["source_patient_id"].map(role_mapping)
    cycles.loc[cycles["strict_split"] == "test", "protocol_role"] = "locked_test"

    sample_rates = recordings.set_index("recording_id")["sample_rate"]
    rows = pd.DataFrame(
        {
            "dataset": "icbhi2017",
            "source_sample_id": cycles["sample_id"],
            "source_patient_id": cycles["source_patient_id"],
            "source_recording_id": cycles["recording_id"],
            "wav_path": cycles["wav_path"],
            "start_seconds": cycles["start_seconds"],
            "end_seconds": cycles["end_seconds"],
            "event_duration_seconds": cycles["cycle_duration_seconds"],
            "fine_label_name": cycles["label_name"],
            "binary_label_name": cycles["binary_label_name"],
            "binary_label_id": cycles["binary_label_id"],
            "source_sample_rate": cycles["recording_id"].map(sample_rates),
            "device": cycles["device"],
            "acquisition_domain": "icbhi2017::" + cycles["device"].astype(str),
            "recording_position": cycles["chest_location"],
            "age": np.nan,
            "sex_code": np.nan,
            "protocol_role": cycles["protocol_role"],
            "source_benchmark_split": cycles["strict_split"],
        }
    )
    return rows, split_metadata


def _sprsound_rows(
    events: pd.DataFrame,
    recordings: pd.DataFrame,
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    events = events.copy()
    events["source_patient_id"] = events["patient_id"].astype(str)
    development = events.loc[events["benchmark_split"] == "train"].copy()
    role_mapping, split_metadata = patient_partition(
        development,
        patient_column="source_patient_id",
        label_column="binary_label_id",
        domain_column="recording_position",
        seed=seed,
    )
    events["protocol_role"] = events["source_patient_id"].map(role_mapping)
    events.loc[
        events["benchmark_split"] == "testing_1",
        "protocol_role",
    ] = "locked_intra_test"
    events.loc[
        events["benchmark_split"] == "testing_2",
        "protocol_role",
    ] = "locked_inter_test"

    sample_rates = recordings.set_index("recording_id")["sample_rate"]
    rows = pd.DataFrame(
        {
            "dataset": "sprsound2022",
            "source_sample_id": events["sample_id"],
            "source_patient_id": events["source_patient_id"],
            "source_recording_id": events["recording_id"],
            "wav_path": events["wav_path"],
            "start_seconds": events["start_seconds"],
            "end_seconds": events["end_seconds"],
            "event_duration_seconds": events["event_duration_seconds"],
            "fine_label_name": events["label_name"],
            "binary_label_name": events["binary_label_name"],
            "binary_label_id": events["binary_label_id"],
            "source_sample_rate": events["recording_id"].map(sample_rates),
            "device": "unreported",
            "acquisition_domain": "sprsound2022::published_collection",
            "recording_position": events["recording_position"],
            "age": events["age"],
            "sex_code": events["sex_code"],
            "protocol_role": events["protocol_role"],
            "source_benchmark_split": events["benchmark_split"],
        }
    )
    return rows, split_metadata


def build_cross_domain_manifest(
    *,
    icbhi_cycles: pd.DataFrame,
    icbhi_recordings: pd.DataFrame,
    sprsound_events: pd.DataFrame,
    sprsound_recordings: pd.DataFrame,
    seed: int = 20_260_728,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Harmonize ICBHI and SPRSound without opening any model test predictions."""
    icbhi, icbhi_split = _icbhi_rows(
        icbhi_cycles,
        icbhi_recordings,
        seed=seed,
    )
    sprsound, sprsound_split = _sprsound_rows(
        sprsound_events,
        sprsound_recordings,
        seed=seed + 1,
    )
    unified = pd.concat([icbhi, sprsound], ignore_index=True)
    unified["sample_id"] = (
        unified["dataset"] + "::" + unified["source_sample_id"].astype(str)
    )
    unified["patient_id"] = (
        unified["dataset"] + "::" + unified["source_patient_id"].astype(str)
    )
    unified["recording_id"] = (
        unified["dataset"] + "::" + unified["source_recording_id"].astype(str)
    )
    unified["locked"] = unified["protocol_role"].isin(LOCKED_ROLES)

    ordered_columns = [
        "sample_id",
        "dataset",
        "source_sample_id",
        "patient_id",
        "source_patient_id",
        "recording_id",
        "source_recording_id",
        "wav_path",
        "start_seconds",
        "end_seconds",
        "event_duration_seconds",
        "fine_label_name",
        "binary_label_name",
        "binary_label_id",
        "source_sample_rate",
        "acquisition_domain",
        "device",
        "recording_position",
        "age",
        "sex_code",
        "protocol_role",
        "locked",
        "source_benchmark_split",
    ]
    unified = unified[ordered_columns].sort_values(
        ["dataset", "source_sample_id"]
    ).reset_index(drop=True)

    development_sets = {
        (dataset, role): set(
            unified.loc[
                (unified["dataset"] == dataset)
                & (unified["protocol_role"] == role),
                "patient_id",
            ]
        )
        for dataset in ("icbhi2017", "sprsound2022")
        for role in DEVELOPMENT_ROLES
    }
    inter_test_patients = set(
        unified.loc[
            unified["protocol_role"].isin({"locked_test", "locked_inter_test"}),
            "patient_id",
        ]
    )
    development_patients = set(
        unified.loc[unified["protocol_role"].isin(DEVELOPMENT_ROLES), "patient_id"]
    )
    root = Path.cwd() if project_root is None else project_root
    checks = {
        "unique_sample_ids": not unified["sample_id"].duplicated().any(),
        "all_roles_known": set(unified["protocol_role"])
        <= set(DEVELOPMENT_ROLES + LOCKED_ROLES),
        "development_roles_have_both_labels": all(
            set(
                unified.loc[
                    (unified["dataset"] == dataset)
                    & (unified["protocol_role"] == role),
                    "binary_label_id",
                ]
            )
            == {0, 1}
            for dataset in ("icbhi2017", "sprsound2022")
            for role in DEVELOPMENT_ROLES
        ),
        "development_patient_disjoint": all(
            not (
                development_sets[(dataset, role_a)]
                & development_sets[(dataset, role_b)]
            )
            for dataset in ("icbhi2017", "sprsound2022")
            for role_index, role_a in enumerate(DEVELOPMENT_ROLES)
            for role_b in DEVELOPMENT_ROLES[role_index + 1 :]
        ),
        "independent_test_patient_disjoint": not (
            development_patients & inter_test_patients
        ),
        "no_locked_rows_in_development": not unified.loc[
            unified["protocol_role"].isin(DEVELOPMENT_ROLES), "locked"
        ].any(),
        "all_source_paths_present": all(
            (root / str(path)).is_file() for path in unified["wav_path"]
        ),
        "labels_harmonized": set(unified["binary_label_name"])
        == {"normal", "adventitious"},
        "calibration_has_at_least_ten_patients_per_dataset": all(
            unified.loc[
                (unified["dataset"] == dataset)
                & (unified["protocol_role"] == "calibration"),
                "patient_id",
            ].nunique()
            >= 10
            for dataset in ("icbhi2017", "sprsound2022")
        ),
        "calibration_has_at_least_100_samples_per_class": all(
            (
                unified.loc[
                    (unified["dataset"] == dataset)
                    & (unified["protocol_role"] == "calibration"),
                    "binary_label_id",
                ].value_counts()
                >= 100
            ).all()
            for dataset in ("icbhi2017", "sprsound2022")
        ),
    }
    role_summary: dict[str, object] = {}
    for (dataset, role), group in unified.groupby(["dataset", "protocol_role"]):
        role_summary[f"{dataset}:{role}"] = {
            "samples": len(group),
            "patients": group["patient_id"].nunique(),
            "recordings": group["recording_id"].nunique(),
            "normal": int((group["binary_label_id"] == 0).sum()),
            "adventitious": int((group["binary_label_id"] == 1).sum()),
            "positive_prevalence": float(group["binary_label_id"].mean()),
        }
    icbhi_development = unified.loc[
        (unified["dataset"] == "icbhi2017")
        & unified["protocol_role"].isin(DEVELOPMENT_ROLES)
    ]
    icbhi_device_summary = {
        str(device): {
            "samples": len(group),
            "patients": group["patient_id"].nunique(),
            "positive_prevalence": float(group["binary_label_id"].mean()),
        }
        for device, group in icbhi_development.groupby("device")
    }
    icbhi_patient_device_counts = (
        icbhi_development[["patient_id", "device"]]
        .drop_duplicates()
        .groupby("patient_id")["device"]
        .nunique()
        .value_counts()
        .sort_index()
    )
    sprsound_development = unified.loc[
        (unified["dataset"] == "sprsound2022")
        & unified["protocol_role"].isin(DEVELOPMENT_ROLES)
    ]
    sprsound_position_summary = {
        str(position): {
            "samples": len(group),
            "patients": group["patient_id"].nunique(),
            "positive_prevalence": float(group["binary_label_id"].mean()),
        }
        for position, group in sprsound_development.groupby("recording_position")
    }
    development_shift_summary = {}
    for dataset, group in unified.loc[
        unified["protocol_role"].isin(DEVELOPMENT_ROLES)
    ].groupby("dataset"):
        development_shift_summary[str(dataset)] = {
            "samples": len(group),
            "patients": group["patient_id"].nunique(),
            "positive_prevalence": float(group["binary_label_id"].mean()),
            "duration_seconds": {
                "median": float(group["event_duration_seconds"].median()),
                "p10": float(group["event_duration_seconds"].quantile(0.10)),
                "p90": float(group["event_duration_seconds"].quantile(0.90)),
            },
            "source_sample_rates": {
                str(int(rate)): int(count)
                for rate, count in group["source_sample_rate"].value_counts().items()
            },
            "fine_label_counts": {
                str(label): int(count)
                for label, count in group["fine_label_name"].value_counts().items()
            },
        }

    audit = {
        "data_gate_passed": all(checks.values()),
        "checks": checks,
        "split_seed": seed,
        "label_mapping": {
            "icbhi2017": "normal -> normal; crackle/wheeze/both -> adventitious",
            "sprsound2022": "normal -> normal; all six non-normal event labels -> adventitious",
        },
        "device_claim_limit": (
            "SPRSound has no per-record device field; ICBHI device is strongly "
            "patient-confounded. Device invariance is not an admissible primary claim."
        ),
        "confounding_summary": {
            "icbhi_devices": icbhi_device_summary,
            "icbhi_devices_per_patient": {
                str(int(count)): int(patients)
                for count, patients in icbhi_patient_device_counts.items()
            },
            "sprsound_device_field": "unreported",
            "sprsound_recording_positions": sprsound_position_summary,
        },
        "development_shift_summary": development_shift_summary,
        "split_metadata": {
            "icbhi2017": icbhi_split,
            "sprsound2022": sprsound_split,
        },
        "role_summary": role_summary,
    }
    return unified, audit
