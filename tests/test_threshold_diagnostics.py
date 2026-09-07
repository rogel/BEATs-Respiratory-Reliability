import numpy as np
import pandas as pd

from respiratory_sound.threshold_diagnostics import (
    leave_one_patient_out_thresholds,
    patient_class_balanced_accuracy,
    select_patient_balanced_threshold,
    threshold_summary,
)


def test_patient_class_balance_prevents_long_patient_domination() -> None:
    targets = np.asarray([0] * 10 + [0] + [1] * 10 + [1])
    predictions = np.asarray([0] * 10 + [1] + [1] * 10 + [0])
    patients = np.asarray(["long0"] * 10 + ["short0"] + ["long1"] * 10 + ["short1"])

    score = patient_class_balanced_accuracy(targets, predictions, patients)

    assert score == 0.5


def test_patient_balanced_threshold_uses_domain_equal_objective() -> None:
    targets = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.2, 0.4, 0.45, 0.55, 0.6, 0.8, 0.9])
    patients = np.asarray(["a0", "a1", "a2", "a3", "b0", "b1", "b2", "b3"])
    domains = np.asarray(["a"] * 4 + ["b"] * 4)

    threshold, score = select_patient_balanced_threshold(
        targets,
        probabilities,
        patients,
        domains,
    )

    assert 0.2 < threshold < 0.8
    assert score >= 0.75


def test_leave_one_patient_out_thresholds_are_finite_and_patient_constant() -> None:
    frame = pd.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "probability_1": [0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.4, 0.85],
            "patient_id": ["a0", "a0", "a1", "a1", "b0", "b0", "b1", "b1"],
            "dataset": ["a", "a", "a", "a", "b", "b", "b", "b"],
        }
    )

    predictions, thresholds, by_patient = leave_one_patient_out_thresholds(
        frame,
        "domain_conditioned",
    )
    summary = threshold_summary(by_patient)

    assert np.isfinite(thresholds).all()
    assert np.isin(predictions, (0, 1)).all()
    assert summary["patients"] == 4
    for patient_id in frame["patient_id"].unique():
        patient_thresholds = thresholds[frame["patient_id"].eq(patient_id)]
        assert np.unique(patient_thresholds).size == 1
