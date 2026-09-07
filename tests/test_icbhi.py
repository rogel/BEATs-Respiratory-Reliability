from pathlib import Path

import pytest

from respiratory_sound.data.icbhi import (
    _strict_patient_splits,
    label_from_flags,
    load_official_split,
    parse_cycle_annotations,
    parse_recording_stem,
    recording_core,
)


def test_parse_recording_stem() -> None:
    metadata = parse_recording_stem("156_2b3_Al_mc_AKGC417L")
    assert metadata.patient_id == "156"
    assert metadata.chest_location == "Al"
    assert metadata.device == "AKGC417L"


def test_recording_core_ignores_only_device_field() -> None:
    assert (
        recording_core("226_1b1_Pl_sc_LittC2SE")
        == recording_core("226_1b1_Pl_sc_Meditron")
        == "226_1b1_Pl_sc"
    )


@pytest.mark.parametrize(
    ("crackle", "wheeze", "expected"),
    [(0, 0, "normal"), (1, 0, "crackle"), (0, 1, "wheeze"), (1, 1, "both")],
)
def test_label_from_flags(crackle: int, wheeze: int, expected: str) -> None:
    assert label_from_flags(crackle, wheeze) == expected


def test_parse_annotations(tmp_path: Path) -> None:
    annotation = tmp_path / "sample.txt"
    annotation.write_text("0.0\t1.5\t0\t0\n1.5\t2.0\t1\t1\n", encoding="utf-8")
    cycles = parse_cycle_annotations(annotation)
    assert [cycle["label_name"] for cycle in cycles] == ["normal", "both"]


def test_strict_split_moves_overlapping_patient_to_test(tmp_path: Path) -> None:
    split_file = tmp_path / "split.txt"
    split_file.write_text(
        "156_2b3_Al_mc_AKGC417L\ttrain\n"
        "156_2b3_Ar_mc_AKGC417L\ttest\n"
        "157_1b1_Al_sc_Meditron\ttrain\n",
        encoding="utf-8",
    )
    official = load_official_split(split_file)
    strict, train_patients, test_patients, overlap = _strict_patient_splits(official)
    assert train_patients == {"156", "157"}
    assert test_patients == {"156"}
    assert overlap == {"156"}
    assert strict == {"156": "test", "157": "train"}
