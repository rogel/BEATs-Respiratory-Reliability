"""SPRSound 2022 parsing, leakage checks, and event-manifest construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedGroupKFold

EXPECTED_RECORDING_COUNTS = {"train": 1_949, "testing_1": 379, "testing_2": 355}
EXPECTED_EVENT_COUNTS = {"train": 6_656, "testing_1": 1_004, "testing_2": 1_429}
EXPECTED_LABEL_COUNTS = {
    "normal": 6_887,
    "rhonchi": 53,
    "wheeze": 865,
    "stridor": 17,
    "coarse_crackle": 66,
    "fine_crackle": 1_167,
    "wheeze_crackle": 34,
}
EXPECTED_PATIENT_COUNTS = {"train": 251, "testing_1": 162, "testing_2": 41}
EXPECTED_SAMPLE_RATE = 8_000
EXPECTED_CHANNELS = 1
EXPECTED_SUBTYPE = "PCM_16"

RAW_TO_CANONICAL_LABEL = {
    "Normal": "normal",
    "Rhonchi": "rhonchi",
    "Wheeze": "wheeze",
    "Stridor": "stridor",
    "Coarse Crackle": "coarse_crackle",
    "Fine Crackle": "fine_crackle",
    "Wheeze+Crackle": "wheeze_crackle",
}
LABEL_TO_ID = {label: index for index, label in enumerate(EXPECTED_LABEL_COUNTS)}
MORPHOLOGY_NAMES = ("transient", "continuous")
MORPHOLOGY_TARGETS = {
    "normal": (0.0, 0.0),
    "coarse_crackle": (1.0, 0.0),
    "fine_crackle": (1.0, 0.0),
    "rhonchi": (0.0, 1.0),
    "wheeze": (0.0, 1.0),
    "stridor": (0.0, 1.0),
    "wheeze_crackle": (1.0, 1.0),
}


@dataclass(frozen=True)
class RecordingMetadata:
    recording_id: str
    patient_id: str
    age: str
    sex_code: str
    recording_position: str
    recording_number: str


def parse_recording_stem(stem: str) -> RecordingMetadata:
    """Parse the five documented fields in an SPRSound recording stem."""
    fields = stem.split("_")
    if len(fields) != 5:
        raise ValueError(f"Unexpected SPRSound recording name: {stem}")
    return RecordingMetadata(
        recording_id=stem,
        patient_id=fields[0],
        age=fields[1],
        sex_code=fields[2],
        recording_position=fields[3],
        recording_number=fields[4],
    )


def canonical_label(raw_label: str) -> str:
    """Map the official display label to a stable machine-readable name."""
    try:
        return RAW_TO_CANONICAL_LABEL[raw_label]
    except KeyError as error:
        raise ValueError(f"Unknown SPRSound event label: {raw_label}") from error


def morphology_target(label_name: str) -> tuple[float, float]:
    """Map a fine-grained event label to transient and continuous attributes."""
    try:
        return MORPHOLOGY_TARGETS[label_name]
    except KeyError as error:
        raise ValueError(f"Unknown morphology source label: {label_name}") from error


def _patient_holdout(
    train_events: pd.DataFrame,
    seed: int,
    n_splits: int = 5,
    attempts: int = 64,
) -> tuple[set[str], dict[str, object]]:
    """Choose a deterministic patient-exclusive validation fold with all classes."""
    labels = train_events["label_id"].to_numpy()
    groups = train_events["patient_id"].to_numpy()
    overall = train_events["label_id"].value_counts(normalize=True).reindex(
        LABEL_TO_ID.values(),
        fill_value=0,
    )
    target_fraction = 1.0 / n_splits
    best: tuple[float, set[str], dict[str, object]] | None = None

    for offset in range(attempts):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + offset,
        )
        for fold_index, (_, validation_indices) in enumerate(
            splitter.split(np.zeros(len(train_events)), labels, groups)
        ):
            validation = train_events.iloc[validation_indices]
            validation_patients = set(validation["patient_id"])
            training = train_events.loc[~train_events["patient_id"].isin(validation_patients)]
            if set(validation["label_id"]) != set(LABEL_TO_ID.values()):
                continue
            if set(training["label_id"]) != set(LABEL_TO_ID.values()):
                continue
            validation_distribution = (
                validation["label_id"]
                .value_counts(normalize=True)
                .reindex(LABEL_TO_ID.values(), fill_value=0)
            )
            fraction = len(validation_patients) / train_events["patient_id"].nunique()
            score = float((validation_distribution - overall).abs().sum())
            score += abs(fraction - target_fraction)
            metadata = {
                "search_seed": seed + offset,
                "fold_index": fold_index,
                "score": score,
                "validation_patient_fraction": fraction,
            }
            if best is None or score < best[0]:
                best = (score, validation_patients, metadata)

    if best is None:
        raise ValueError("Could not construct a patient-exclusive validation fold with all classes")
    return best[1], best[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sprsound_manifests(
    dataset_root: Path,
    project_root: Path,
    seed: int = 20_260_727,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build recording/event manifests and a strict source-integrity report."""
    split_sources = {
        "train": (
            dataset_root / "train2022_wav",
            dataset_root / "train2022_json",
        ),
        "testing_1": (
            dataset_root / "test2022_wav",
            dataset_root / "test2022_json" / "intra_test_json",
        ),
        "testing_2": (
            dataset_root / "test2022_wav",
            dataset_root / "test2022_json" / "inter_test_json",
        ),
    }
    recording_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    missing_wav: list[str] = []
    malformed_json: list[str] = []
    annotation_bound_errors: list[str] = []
    invalid_audio: list[str] = []

    for benchmark_split, (wav_dir, annotation_dir) in split_sources.items():
        annotation_paths = sorted(annotation_dir.glob("*.json"))
        for annotation_path in annotation_paths:
            wav_path = wav_dir / f"{annotation_path.stem}.wav"
            if not wav_path.exists():
                missing_wav.append(f"{benchmark_split}:{annotation_path.stem}")
                continue
            metadata = parse_recording_stem(annotation_path.stem)
            try:
                payload = json.loads(annotation_path.read_text(encoding="utf-8"))
                raw_events = payload["event_annotation"]
                record_annotation = str(payload["record_annotation"])
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                malformed_json.append(f"{benchmark_split}:{annotation_path.stem}:{error}")
                continue

            audio_info = sf.info(wav_path)
            duration_seconds = audio_info.frames / audio_info.samplerate
            if (
                audio_info.samplerate != EXPECTED_SAMPLE_RATE
                or audio_info.channels != EXPECTED_CHANNELS
                or audio_info.subtype != EXPECTED_SUBTYPE
            ):
                invalid_audio.append(annotation_path.stem)
            relative_wav = wav_path.resolve().relative_to(project_root.resolve()).as_posix()
            relative_annotation = (
                annotation_path.resolve().relative_to(project_root.resolve()).as_posix()
            )

            for event_index, raw_event in enumerate(raw_events):
                start_seconds = float(raw_event["start"]) / 1_000.0
                end_seconds = float(raw_event["end"]) / 1_000.0
                if (
                    start_seconds < 0
                    or end_seconds <= start_seconds
                    or end_seconds > duration_seconds + 0.05
                ):
                    annotation_bound_errors.append(
                        f"{benchmark_split}:{annotation_path.stem}:{event_index}"
                    )
                label_name = canonical_label(str(raw_event["type"]))
                event_rows.append(
                    {
                        "sample_id": f"{metadata.recording_id}__event_{event_index:03d}",
                        **metadata.__dict__,
                        "wav_path": relative_wav,
                        "annotation_path": relative_annotation,
                        "benchmark_split": benchmark_split,
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "event_duration_seconds": end_seconds - start_seconds,
                        "label_name": label_name,
                        "label_id": LABEL_TO_ID[label_name],
                        "binary_label_name": (
                            "normal" if label_name == "normal" else "adventitious"
                        ),
                        "binary_label_id": 0 if label_name == "normal" else 1,
                    }
                )

            recording_rows.append(
                {
                    **metadata.__dict__,
                    "wav_path": relative_wav,
                    "annotation_path": relative_annotation,
                    "benchmark_split": benchmark_split,
                    "record_annotation": record_annotation,
                    "sample_rate": audio_info.samplerate,
                    "channels": audio_info.channels,
                    "audio_subtype": audio_info.subtype,
                    "frames": audio_info.frames,
                    "audio_duration_seconds": duration_seconds,
                    "num_events": len(raw_events),
                }
            )

    recordings = pd.DataFrame(recording_rows)
    events = pd.DataFrame(event_rows)
    validation_patients, holdout_metadata = _patient_holdout(
        events.loc[events["benchmark_split"] == "train"].reset_index(drop=True),
        seed=seed,
    )
    recordings["development_split"] = recordings.apply(
        lambda row: (
            "validation"
            if row["benchmark_split"] == "train" and row["patient_id"] in validation_patients
            else row["benchmark_split"]
        ),
        axis=1,
    )
    events["development_split"] = events.apply(
        lambda row: (
            "validation"
            if row["benchmark_split"] == "train" and row["patient_id"] in validation_patients
            else row["benchmark_split"]
        ),
        axis=1,
    )

    json_stems = {
        split: {path.stem for path in annotation_dir.glob("*.json")}
        for split, (_, annotation_dir) in split_sources.items()
    }
    train_wav_stems = {path.stem for path in split_sources["train"][0].glob("*.wav")}
    test_wav_stems = {path.stem for path in split_sources["testing_1"][0].glob("*.wav")}
    expected_test_stems = json_stems["testing_1"] | json_stems["testing_2"]
    unmatched_wav = sorted(
        {f"train:{stem}" for stem in train_wav_stems - json_stems["train"]}
        | {f"test:{stem}" for stem in test_wav_stems - expected_test_stems}
    )
    patient_sets = {
        split: set(recordings.loc[recordings["benchmark_split"] == split, "patient_id"])
        for split in split_sources
    }
    development_patient_sets = {
        split: set(recordings.loc[recordings["development_split"] == split, "patient_id"])
        for split in ("train", "validation", "testing_2")
    }
    recording_counts = recordings["benchmark_split"].value_counts().sort_index().to_dict()
    event_counts = events["benchmark_split"].value_counts().sort_index().to_dict()
    label_counts = events["label_name"].value_counts().sort_index().to_dict()
    patient_counts = {split: len(patients) for split, patients in patient_sets.items()}
    development_label_counts = {
        split: {
            label: int(count)
            for label, count in group["label_name"].value_counts().sort_index().items()
        }
        for split, group in events.groupby("development_split")
    }
    checks = {
        "recording_counts_match": recording_counts == EXPECTED_RECORDING_COUNTS,
        "event_counts_match": event_counts == EXPECTED_EVENT_COUNTS,
        "label_counts_match": label_counts == EXPECTED_LABEL_COUNTS,
        "patient_counts_match": patient_counts == EXPECTED_PATIENT_COUNTS,
        "all_json_have_wav": not missing_wav,
        "all_wav_have_json": not unmatched_wav,
        "all_json_valid": not malformed_json,
        "all_audio_format_valid": not invalid_audio,
        "annotation_bounds_valid": not annotation_bound_errors,
        "testing_1_is_intra_subject": patient_sets["testing_1"] <= patient_sets["train"],
        "testing_2_is_inter_subject": not (
            patient_sets["testing_2"] & (patient_sets["train"] | patient_sets["testing_1"])
        ),
        "development_train_validation_disjoint": not (
            development_patient_sets["train"] & development_patient_sets["validation"]
        ),
        "development_testing_2_disjoint": not (
            development_patient_sets["testing_2"]
            & (
                development_patient_sets["train"]
                | development_patient_sets["validation"]
            )
        ),
        "all_classes_in_development_train_and_validation": (
            set(
                events.loc[events["development_split"] == "train", "label_name"]
            )
            == set(EXPECTED_LABEL_COUNTS)
            and set(
                events.loc[events["development_split"] == "validation", "label_name"]
            )
            == set(EXPECTED_LABEL_COUNTS)
        ),
    }
    report: dict[str, object] = {
        "recording_count": len(recordings),
        "event_count": len(events),
        "recording_counts": recording_counts,
        "event_counts": event_counts,
        "label_counts": label_counts,
        "patient_counts": patient_counts,
        "patient_overlaps": {
            "train_testing_1": len(patient_sets["train"] & patient_sets["testing_1"]),
            "train_testing_2": len(patient_sets["train"] & patient_sets["testing_2"]),
            "testing_1_testing_2": len(
                patient_sets["testing_1"] & patient_sets["testing_2"]
            ),
        },
        "development_recording_counts": (
            recordings["development_split"].value_counts().sort_index().to_dict()
        ),
        "development_event_counts": (
            events["development_split"].value_counts().sort_index().to_dict()
        ),
        "development_patient_counts": {
            split: len(patients) for split, patients in development_patient_sets.items()
        },
        "development_label_counts": development_label_counts,
        "validation_selection": holdout_metadata,
        "sample_rate_counts": recordings["sample_rate"].value_counts().sort_index().to_dict(),
        "channel_counts": recordings["channels"].value_counts().sort_index().to_dict(),
        "audio_subtype_counts": (
            recordings["audio_subtype"].value_counts().sort_index().to_dict()
        ),
        "record_annotation_counts": (
            recordings["record_annotation"].value_counts().sort_index().to_dict()
        ),
        "missing_wav": missing_wav,
        "unmatched_wav": unmatched_wav,
        "malformed_json": malformed_json,
        "invalid_audio": invalid_audio,
        "annotation_bound_errors": annotation_bound_errors,
        "checks": checks,
    }
    report["data_gate_passed"] = all(checks.values())
    return recordings, events, report


def save_manifests(
    recordings: pd.DataFrame,
    events: pd.DataFrame,
    report: dict[str, object],
    output_dir: Path,
) -> dict[str, str]:
    """Persist SPRSound manifests, audit report, and content hashes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings_path = output_dir / "sprsound2022_recordings.csv"
    events_path = output_dir / "sprsound2022_events.csv"
    report_path = output_dir / "sprsound2022_audit.json"
    recordings.to_csv(recordings_path, index=False)
    events.to_csv(events_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    hashes = {
        path.name: sha256_file(path) for path in (recordings_path, events_path, report_path)
    }
    (output_dir / "sprsound2022_manifest_sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return hashes
