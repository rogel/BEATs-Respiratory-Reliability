import numpy as np
import pytest
from scipy.special import expit

from respiratory_sound.revision_calibration_regression import (
    CalibrationRegressionError, fit_calibration_regression,
)


def known_data():
    y = np.concatenate([np.r_[np.zeros(100 - n), np.ones(n)] for n in (25, 50, 75)])
    p = np.repeat([0.25, 0.5, 0.75], 100)
    return y, p


def test_known_finite_mle_and_independent_solver():
    y, p = known_data()
    result = fit_calibration_regression(y, p, independent_check=True)
    assert result["calibration_intercept"] == pytest.approx(0, abs=1e-7)
    assert result["calibration_slope"] == pytest.approx(1, abs=1e-7)
    assert result["independent_BFGS_check_passed"]


def test_large_miscalibration_is_not_separation():
    y, p = known_data()
    x = np.log(p / (1 - p))
    result = fit_calibration_regression(y, expit((x + 8) / 2), independent_check=True)
    assert result["calibration_intercept"] == pytest.approx(-8, abs=1e-6)
    assert result["calibration_slope"] == pytest.approx(2, abs=1e-6)


def test_temperature_reparameterization():
    y, p = known_data()
    raw = fit_calibration_regression(y, p)
    scaled = fit_calibration_regression(y, expit(np.log(p / (1 - p)) / 1.8))
    assert scaled["calibration_intercept"] == pytest.approx(raw["calibration_intercept"], abs=1e-7)
    assert scaled["calibration_slope"] == pytest.approx(1.8 * raw["calibration_slope"], abs=1e-7)


@pytest.mark.parametrize("y,p,reason", [
    ([0, 0, 1, 1], [.1, .2, .8, .9], "complete_separation"),
    ([0, 0, 1, 1], [.9, .8, .2, .1], "complete_separation"),
    ([0, 0, 1, 1], [.1, .5, .5, .9], "quasi_complete_separation"),
    ([0, 0, 1, 1], [.5, .5, .5, .5], "rank_deficient_predictor"),
    ([0, 0, 0, 0], [.1, .3, .5, .7], "single_class"),
])
def test_explicit_nonestimability(y, p, reason):
    with pytest.raises(CalibrationRegressionError) as exc:
        fit_calibration_regression(y, p)
    assert exc.value.reason == reason


@pytest.mark.parametrize("p", [[.1, np.nan, .5, .9], [.1, 1.1, .5, .9]])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError):
        fit_calibration_regression([0, 1, 0, 1], p)


def test_finite_wrong_optimizer_result_is_rejected(monkeypatch):
    from types import SimpleNamespace
    import respiratory_sound.revision_calibration_regression as module
    monkeypatch.setattr(module, "minimize", lambda *a, **kw: SimpleNamespace(
        x=np.array([10.0, -10.0]), success=True, message="false success", nit=1))
    y, p = known_data()
    with pytest.raises(CalibrationRegressionError, match="numerical_nonconvergence"):
        fit_calibration_regression(y, p)


def test_permutation_invariance():
    y, p = known_data()
    index = np.random.default_rng(27).permutation(len(y))
    original, permuted = fit_calibration_regression(y, p), fit_calibration_regression(y[index], p[index])
    for key in ("calibration_intercept", "calibration_slope"):
        assert original[key] == pytest.approx(permuted[key], abs=1e-7)
