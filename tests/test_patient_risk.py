import numpy as np
import pandas as pd

from respiratory_sound.patient_risk import (
    jaccard_index,
    patient_risk_table,
    top_risk_patients,
)


def test_patient_risk_balances_classes_before_averaging() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["a"] * 11 + ["b", "b"],
            "target": [0] * 10 + [1, 0, 1],
            "probability_1": [0.1] * 10 + [0.1, 0.1, 0.9],
        }
    )

    table = patient_risk_table(frame).set_index("patient_id")

    expected_a = (-np.log(0.9) - np.log(0.1)) / 2.0
    expected_b = -np.log(0.9)
    assert np.isclose(table.loc["a", "patient_risk"], expected_a)
    assert np.isclose(table.loc["b", "patient_risk"], expected_b)


def test_top_risk_patients_uses_ceiling_and_patient_tie_break() -> None:
    table = pd.DataFrame(
        {
            "patient_id": ["c", "b", "a", "d", "e"],
            "patient_risk": [0.2, 0.9, 0.9, 0.1, 0.3],
        }
    )

    selected = top_risk_patients(table, fraction=0.25)

    assert selected == ("a", "b")


def test_jaccard_index_handles_overlap_and_empty_sets() -> None:
    assert jaccard_index({"a", "b"}, {"b", "c"}) == 1 / 3
    assert jaccard_index(set(), set()) == 1.0
