import numpy as np

from respiratory_sound.metrics import (
    icbhi_metrics,
    paired_patient_bootstrap_difference,
    paired_patient_bootstrap_mean_seed_difference,
    patient_bootstrap_intervals,
    respiratory_metrics,
)


def test_icbhi_metrics_uses_normal_specificity_and_aggregated_abnormal_sensitivity() -> None:
    targets = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    predictions = np.array([0, 1, 1, 1, 2, 0, 3, 2])

    metrics = icbhi_metrics(targets, predictions)

    assert metrics["specificity"] == 0.5
    assert metrics["sensitivity"] == 4 / 6
    assert metrics["icbhi_score"] == (0.5 + 4 / 6) / 2
    assert metrics["confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [1, 0, 1, 0],
        [0, 0, 1, 1],
    ]


def test_patient_bootstrap_is_degenerate_for_perfect_predictions() -> None:
    targets = np.tile(np.arange(4), 4)
    patient_ids = np.repeat(np.array(["a", "b", "c", "d"]), 4)

    intervals = patient_bootstrap_intervals(
        targets,
        targets,
        patient_ids,
        iterations=50,
        seed=7,
    )

    for interval in intervals.values():
        assert interval == {"estimate": 1.0, "lower": 1.0, "upper": 1.0}


def test_sprsound_score_matches_official_definition() -> None:
    targets = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 1, 1, 0, 2, 0])
    metrics = respiratory_metrics(targets, predictions, ("normal", "rhonchi", "wheeze"))
    assert metrics["specificity"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["average_score"] == 0.5
    assert metrics["harmonic_score"] == 0.5
    assert metrics["score"] == 0.5


def test_paired_bootstrap_is_zero_for_identical_models() -> None:
    targets = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    predictions = np.array([0, 1, 0, 3, 0, 2, 2, 3])
    patient_ids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])

    difference = paired_patient_bootstrap_difference(
        targets,
        predictions,
        predictions,
        patient_ids,
        iterations=50,
        seed=7,
    )

    assert difference == {
        "estimate": 0.0,
        "lower": 0.0,
        "upper": 0.0,
        "probability_a_greater_than_b": 0.0,
    }


def test_mean_seed_bootstrap_detects_consistent_improvement() -> None:
    targets = np.array([0, 1, 0, 1])
    patient_ids = np.array(["1", "1", "2", "2"])
    predictions_a = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
    predictions_b = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

    difference = paired_patient_bootstrap_mean_seed_difference(
        targets,
        predictions_a,
        predictions_b,
        patient_ids,
        iterations=100,
        seed=7,
    )

    assert difference["estimate"] > 0
    assert difference["probability_a_greater_than_b"] > 0.5
