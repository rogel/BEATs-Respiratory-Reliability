from pathlib import Path

import pandas as pd
import pytest

from respiratory_sound.data.sprsound import (
    LABEL_TO_ID,
    _patient_holdout,
    canonical_label,
    morphology_target,
    parse_recording_stem,
)


def test_parse_recording_stem() -> None:
    metadata = parse_recording_stem("41283394_2.8_1_p2_2790")
    assert metadata.patient_id == "41283394"
    assert metadata.age == "2.8"
    assert metadata.recording_position == "p2"


def test_canonical_label() -> None:
    assert canonical_label("Fine Crackle") == "fine_crackle"
    assert canonical_label("Wheeze+Crackle") == "wheeze_crackle"


def test_morphology_target_mapping() -> None:
    assert morphology_target("normal") == (0.0, 0.0)
    assert morphology_target("fine_crackle") == (1.0, 0.0)
    assert morphology_target("coarse_crackle") == (1.0, 0.0)
    assert morphology_target("wheeze") == (0.0, 1.0)
    assert morphology_target("rhonchi") == (0.0, 1.0)
    assert morphology_target("stridor") == (0.0, 1.0)
    assert morphology_target("wheeze_crackle") == (1.0, 1.0)
    for label_name in LABEL_TO_ID:
        transient, continuous = morphology_target(label_name)
        assert int(transient or continuous) == int(label_name != "normal")


def test_unknown_morphology_label_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown morphology source label"):
        morphology_target("unknown")


def test_patient_holdout_is_disjoint_and_contains_all_classes(tmp_path: Path) -> None:
    del tmp_path
    rows = []
    for patient_index in range(20):
        for label_id in LABEL_TO_ID.values():
            rows.append({"patient_id": str(patient_index), "label_id": label_id})
    events = pd.DataFrame(rows)
    validation_patients, metadata = _patient_holdout(events, seed=7, attempts=2)
    assert validation_patients
    assert len(validation_patients) < events["patient_id"].nunique()
    assert metadata["validation_patient_fraction"] == 0.2
    validation = events.loc[events["patient_id"].isin(validation_patients)]
    training = events.loc[~events["patient_id"].isin(validation_patients)]
    assert set(validation["label_id"]) == set(LABEL_TO_ID.values())
    assert set(training["label_id"]) == set(LABEL_TO_ID.values())
