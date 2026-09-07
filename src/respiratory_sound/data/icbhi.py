"""ICBHI 2017 parsing, split auditing, and manifest construction."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.model_selection import StratifiedGroupKFold

EXPECTED_RECORDINGS = 920
EXPECTED_CYCLES = 6_898
EXPECTED_LABEL_COUNTS = {
    "normal": 3_642,
    "crackle": 1_864,
    "wheeze": 886,
    "both": 506,
}
EXPECTED_OFFICIAL_RECORDING_COUNTS = {"train": 539, "test": 381}
EXPECTED_OFFICIAL_OVERLAPPING_PATIENTS = {"156", "218"}
EXPECTED_SPLIT_ALIASES = {
    "226_1b1_Pl_sc_Meditron": "226_1b1_Pl_sc_LittC2SE",
}
LABEL_TO_ID = {"normal": 0, "crackle": 1, "wheeze": 2, "both": 3}


@dataclass(frozen=True)
class RecordingMetadata:
    recording_id: str
    patient_id: str
    recording_index: str
    chest_location: str
    acquisition_mode: str
    device: str


def parse_recording_stem(stem: str) -> RecordingMetadata:
    """Parse an ICBHI recording stem into its documented fields."""
    fields = stem.split("_")
    if len(fields) != 5:
        raise ValueError(f"Unexpected ICBHI recording name: {stem}")
    return RecordingMetadata(
        recording_id=stem,
        patient_id=fields[0],
        recording_index=fields[1],
        chest_location=fields[2],
        acquisition_mode=fields[3],
        device=fields[4],
    )


def load_official_split(path: Path) -> dict[str, str]:
    """Load the official recording-level train/test mapping."""
    mapping: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 2 or fields[1] not in {"train", "test"}:
                raise ValueError(f"Invalid split row at line {line_number}: {line.rstrip()}")
            recording_id, split = fields
            if recording_id in mapping:
                raise ValueError(f"Duplicate recording in split file: {recording_id}")
            mapping[recording_id] = split
    return mapping


def recording_core(recording_id: str) -> str:
    """Return the four stable filename fields, excluding corrected device names."""
    fields = recording_id.split("_")
    if len(fields) != 5:
        raise ValueError(f"Unexpected ICBHI recording name: {recording_id}")
    return "_".join(fields[:4])


def label_from_flags(crackle: int, wheeze: int) -> str:
    if (crackle, wheeze) == (0, 0):
        return "normal"
    if (crackle, wheeze) == (1, 0):
        return "crackle"
    if (crackle, wheeze) == (0, 1):
        return "wheeze"
    if (crackle, wheeze) == (1, 1):
        return "both"
    raise ValueError(f"Invalid crackle/wheeze flags: {(crackle, wheeze)}")


def parse_cycle_annotations(path: Path) -> list[dict[str, float | int | str]]:
    """Parse the four-column cycle annotation file associated with one WAV."""
    cycles: list[dict[str, float | int | str]] = []
    with path.open(encoding="utf-8") as handle:
        for cycle_index, line in enumerate(handle):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"Invalid annotation row in {path}: {line.rstrip()}")
            start, end = float(fields[0]), float(fields[1])
            crackle, wheeze = int(fields[2]), int(fields[3])
            if start < 0 or end <= start:
                raise ValueError(f"Invalid cycle bounds in {path}: {start}, {end}")
            label_name = label_from_flags(crackle, wheeze)
            cycles.append(
                {
                    "cycle_index": cycle_index,
                    "start_seconds": start,
                    "end_seconds": end,
                    "cycle_duration_seconds": end - start,
                    "crackle": crackle,
                    "wheeze": wheeze,
                    "label_name": label_name,
                    "label_id": LABEL_TO_ID[label_name],
                }
            )
    return cycles


def _strict_patient_splits(
    official_split: dict[str, str],
) -> tuple[dict[str, str], set[str], set[str], set[str]]:
    patient_to_official_splits: dict[str, set[str]] = {}
    for recording_id, split in official_split.items():
        patient_id = parse_recording_stem(recording_id).patient_id
        patient_to_official_splits.setdefault(patient_id, set()).add(split)

    official_train_patients = {
        patient for patient, splits in patient_to_official_splits.items() if "train" in splits
    }
    official_test_patients = {
        patient for patient, splits in patient_to_official_splits.items() if "test" in splits
    }
    overlapping_patients = official_train_patients & official_test_patients
    strict_mapping = {
        patient: ("test" if "test" in splits else "train")
        for patient, splits in patient_to_official_splits.items()
    }
    return (
        strict_mapping,
        official_train_patients,
        official_test_patients,
        overlapping_patients,
    )


def _development_patient_splits(
    cycles: pd.DataFrame,
    seed: int,
    n_splits: int = 5,
) -> dict[str, str]:
    strict_train = cycles[cycles["strict_split"] == "train"].reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, validation_indices = next(
        splitter.split(
            np.zeros(len(strict_train)),
            strict_train["label_id"].to_numpy(),
            groups=strict_train["patient_id"].to_numpy(),
        )
    )
    validation_patients = set(strict_train.iloc[validation_indices]["patient_id"])
    all_patients = set(cycles["patient_id"])
    strict_test_patients = set(cycles.loc[cycles["strict_split"] == "test", "patient_id"])
    return {
        patient: (
            "test"
            if patient in strict_test_patients
            else "validation"
            if patient in validation_patients
            else "train"
        )
        for patient in all_patients
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_icbhi_manifests(
    audio_dir: Path,
    official_split_file: Path,
    project_root: Path,
    seed: int = 20_260_727,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build recording and cycle manifests and return a validation report."""
    official_split = load_official_split(official_split_file)
    (
        strict_patient_split,
        official_train_patients,
        official_test_patients,
        official_overlap,
    ) = _strict_patient_splits(official_split)

    wav_paths = sorted(audio_dir.rglob("*.wav"))
    stems = [path.stem for path in wav_paths]
    duplicate_stems = sorted(stem for stem, count in Counter(stems).items() if count > 1)
    if duplicate_stems:
        raise ValueError(f"Duplicate WAV recording stems: {duplicate_stems[:5]}")

    recording_rows: list[dict[str, object]] = []
    cycle_rows: list[dict[str, object]] = []
    annotation_bound_errors: list[str] = []
    missing_annotations: list[str] = []
    missing_split_entries: list[str] = []
    matched_split_entries: set[str] = set()
    split_aliases: dict[str, str] = {}
    split_entries_by_core: dict[str, list[str]] = {}
    for official_recording_id in official_split:
        split_entries_by_core.setdefault(
            recording_core(official_recording_id),
            [],
        ).append(official_recording_id)

    for wav_path in wav_paths:
        metadata = parse_recording_stem(wav_path.stem)
        annotation_path = wav_path.with_suffix(".txt")
        if not annotation_path.exists():
            missing_annotations.append(metadata.recording_id)
            continue
        official_recording_id = metadata.recording_id
        if official_recording_id not in official_split:
            candidates = split_entries_by_core.get(
                recording_core(metadata.recording_id),
                [],
            )
            if len(candidates) != 1:
                missing_split_entries.append(metadata.recording_id)
                continue
            official_recording_id = candidates[0]
            split_aliases[official_recording_id] = metadata.recording_id

        audio_info = sf.info(wav_path)
        audio_duration = audio_info.frames / audio_info.samplerate
        official = official_split[official_recording_id]
        matched_split_entries.add(official_recording_id)
        strict = strict_patient_split[metadata.patient_id]
        relative_wav_path = wav_path.resolve().relative_to(project_root.resolve())
        cycles = parse_cycle_annotations(annotation_path)

        for cycle in cycles:
            if float(cycle["end_seconds"]) > audio_duration + 0.05:
                annotation_bound_errors.append(f"{metadata.recording_id}:{cycle['cycle_index']}")
            cycle_rows.append(
                {
                    "sample_id": (
                        f"{metadata.recording_id}__cycle_{int(cycle['cycle_index']):03d}"
                    ),
                    **metadata.__dict__,
                    "wav_path": relative_wav_path.as_posix(),
                    "official_split": official,
                    "strict_split": strict,
                    **cycle,
                }
            )

        recording_rows.append(
            {
                **metadata.__dict__,
                "wav_path": relative_wav_path.as_posix(),
                "annotation_path": annotation_path.resolve()
                .relative_to(project_root.resolve())
                .as_posix(),
                "official_split": official,
                "strict_split": strict,
                "sample_rate": audio_info.samplerate,
                "frames": audio_info.frames,
                "audio_duration_seconds": audio_duration,
                "num_cycles": len(cycles),
            }
        )

    recordings = pd.DataFrame(recording_rows)
    cycles = pd.DataFrame(cycle_rows)
    if not cycles.empty:
        development_split = _development_patient_splits(cycles, seed=seed)
        cycles["development_split"] = cycles["patient_id"].map(development_split)
        recordings["development_split"] = recordings["patient_id"].map(development_split)
        development_patient_sets = {
            split: set(recordings.loc[recordings["development_split"] == split, "patient_id"])
            for split in ("train", "validation", "test")
        }
        development_patient_overlap = (
            (development_patient_sets["train"] & development_patient_sets["validation"])
            | (development_patient_sets["train"] & development_patient_sets["test"])
            | (development_patient_sets["validation"] & development_patient_sets["test"])
        )
    else:
        development_patient_sets = {split: set() for split in ("train", "validation", "test")}
        development_patient_overlap = set()
    strict_train_set = set(recordings.loc[recordings["strict_split"] == "train", "patient_id"])
    strict_test_set = set(recordings.loc[recordings["strict_split"] == "test", "patient_id"])
    strict_patient_overlap = strict_train_set & strict_test_set

    report: dict[str, object] = {
        "recording_count": len(recordings),
        "cycle_count": len(cycles),
        "label_counts": (
            cycles["label_name"].value_counts().sort_index().to_dict() if not cycles.empty else {}
        ),
        "official_recording_counts": (
            recordings["official_split"].value_counts().sort_index().to_dict()
            if not recordings.empty
            else {}
        ),
        "official_train_patient_count": len(official_train_patients),
        "official_test_patient_count": len(official_test_patients),
        "official_overlapping_patients": sorted(official_overlap),
        "strict_train_patient_count": sum(
            split == "train" for split in strict_patient_split.values()
        ),
        "strict_test_patient_count": sum(
            split == "test" for split in strict_patient_split.values()
        ),
        "strict_patient_overlap_count": len(strict_patient_overlap),
        "development_recording_counts": (
            recordings["development_split"].value_counts().sort_index().to_dict()
            if not recordings.empty
            else {}
        ),
        "development_cycle_counts": (
            cycles["development_split"].value_counts().sort_index().to_dict()
            if not cycles.empty
            else {}
        ),
        "development_patient_counts": {
            split: len(patients) for split, patients in development_patient_sets.items()
        },
        "development_label_counts": (
            {
                split: {
                    label: int(count) for label, count in group["label_name"].value_counts().items()
                }
                for split, group in cycles.groupby("development_split")
            }
            if not cycles.empty
            else {}
        ),
        "development_overlapping_patients": sorted(development_patient_overlap),
        "sample_rate_counts": (
            recordings["sample_rate"].value_counts().sort_index().to_dict()
            if not recordings.empty
            else {}
        ),
        "missing_annotations": missing_annotations,
        "missing_split_entries": missing_split_entries,
        "split_entries_without_wav": sorted(set(official_split) - matched_split_entries),
        "split_aliases": split_aliases,
        "annotation_bound_errors": annotation_bound_errors,
        "checks": {
            "recording_count_matches": len(recordings) == EXPECTED_RECORDINGS,
            "cycle_count_matches": len(cycles) == EXPECTED_CYCLES,
            "label_counts_match": (
                cycles["label_name"].value_counts().to_dict() == EXPECTED_LABEL_COUNTS
                if not cycles.empty
                else False
            ),
            "official_recording_counts_match": (
                recordings["official_split"].value_counts().to_dict()
                == EXPECTED_OFFICIAL_RECORDING_COUNTS
                if not recordings.empty
                else False
            ),
            "official_overlap_matches_documented": (
                official_overlap == EXPECTED_OFFICIAL_OVERLAPPING_PATIENTS
            ),
            "split_aliases_match_documented": split_aliases == EXPECTED_SPLIT_ALIASES,
            "strict_patient_disjoint": not strict_patient_overlap,
            "development_patient_disjoint": not development_patient_overlap,
            "all_annotations_present": not missing_annotations,
            "all_split_entries_present": not missing_split_entries
            and not (set(official_split) - matched_split_entries),
            "annotation_bounds_valid": not annotation_bound_errors,
        },
    }
    report["data_gate_passed"] = all(report["checks"].values())
    return recordings, cycles, report


def save_manifests(
    recordings: pd.DataFrame,
    cycles: pd.DataFrame,
    report: dict[str, object],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings_path = output_dir / "icbhi2017_recordings.csv"
    cycles_path = output_dir / "icbhi2017_cycles.csv"
    report_path = output_dir / "icbhi2017_audit.json"

    recordings.to_csv(recordings_path, index=False)
    cycles.to_csv(cycles_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    hashes = {
        recordings_path.name: sha256_file(recordings_path),
        cycles_path.name: sha256_file(cycles_path),
        report_path.name: sha256_file(report_path),
    }
    (output_dir / "icbhi2017_manifest_sha256.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return hashes
