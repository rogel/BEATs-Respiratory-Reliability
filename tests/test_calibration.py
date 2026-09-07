import numpy as np

from respiratory_sound.calibration import (
    PositivePlattScaler,
    equal_frequency_ece,
    fit_positive_platt,
    fit_temperature,
    probability_metrics,
    select_balanced_threshold,
)


def test_positive_platt_preserves_probability_ranking() -> None:
    probabilities = np.asarray((0.05, 0.2, 0.6, 0.95))
    calibrated = PositivePlattScaler(slope=0.7, intercept=-1.2).transform(
        probabilities
    )

    assert np.all(np.diff(calibrated) > 0)


def test_target_calibration_corrects_shifted_synthetic_probabilities() -> None:
    targets = np.asarray([0] * 50 + [1] * 50)
    latent = np.concatenate(
        (np.linspace(-2.0, -0.2, 50), np.linspace(0.2, 2.0, 50))
    )
    shifted_probabilities = 1.0 / (1.0 + np.exp(-(latent + 3.0)))
    raw_metrics = probability_metrics(targets, shifted_probabilities)

    temperature = fit_temperature(targets, shifted_probabilities)
    platt = fit_positive_platt(targets, shifted_probabilities)
    calibrated = platt.transform(shifted_probabilities)
    calibrated_metrics = probability_metrics(targets, calibrated)
    threshold, score = select_balanced_threshold(targets, calibrated)

    assert temperature.temperature > 1.0
    assert platt.slope > 0.0
    assert calibrated_metrics["negative_log_likelihood"] < raw_metrics[
        "negative_log_likelihood"
    ]
    assert calibrated_metrics["brier_score"] < raw_metrics["brier_score"]
    assert 0.0 < threshold < 1.0
    assert score == 1.0


def test_equal_frequency_ece_is_zero_for_binwise_perfect_probabilities() -> None:
    targets = np.asarray([0, 1, 0, 1])
    probabilities = np.asarray([0.0, 1.0, 0.0, 1.0])

    assert equal_frequency_ece(targets, probabilities, bins=2) < 1.0e-5
