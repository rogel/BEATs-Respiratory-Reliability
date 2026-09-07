import numpy as np

from respiratory_sound.selective_prediction import (
    equal_frequency_reliability_table,
    fit_uncertainty_cutoff,
    normalized_binary_entropy,
    risk_coverage_curve,
    selective_operating_point,
)


def test_entropy_is_low_at_extremes_and_high_at_half() -> None:
    entropy = normalized_binary_entropy(np.asarray((0.01, 0.5, 0.99)))
    assert entropy[1] > entropy[0]
    assert np.isclose(entropy[0], entropy[2])


def test_fixed_cutoff_reports_class_coverage_and_lower_risk() -> None:
    targets = np.asarray((0, 0, 0, 1, 1, 1))
    probabilities = np.asarray((0.01, 0.2, 0.49, 0.51, 0.8, 0.99))
    uncertainty = normalized_binary_entropy(probabilities)
    cutoff = fit_uncertainty_cutoff(uncertainty, 4 / 6)
    point = selective_operating_point(targets, probabilities, cutoff)

    assert point["accepted"] == 4
    assert point["normal_coverage"] == 2 / 3
    assert point["adventitious_coverage"] == 2 / 3
    assert point["selective_error_risk"] == 0.0


def test_risk_curve_and_reliability_table_are_complete() -> None:
    targets = np.asarray((0, 1, 0, 1))
    probabilities = np.asarray((0.1, 0.8, 0.7, 0.6))
    summary, curve = risk_coverage_curve(targets, probabilities)
    table = equal_frequency_reliability_table(targets, probabilities, bins=2)

    assert curve.shape == (4, 3)
    assert curve[-1, 0] == 1.0
    assert summary["full_coverage_error_risk"] == 0.25
    assert sum(row["count"] for row in table) == 4
