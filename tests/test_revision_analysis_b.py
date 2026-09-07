"""Synthetic pre-inference QA; never loads study predictions or locked outcomes."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("revision_b", SCRIPTS / "run_revision_analysis_b.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def data():
    return pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "dataset": ["icbhi2017"]*4,
        "patient_id": ["p1", "p1", "p2", "p2"], "protocol_role": ["validation_select"]*4,
        "locked": [False]*4, "binary_label_id": [0, 1, 0, 1], "fine_label_name": ["x"]*4,
        "event_duration_seconds": [1.0]*4, "target": [0, 1, 0, 1],
        "probability_1": [.1, .4, .6, .9], "probability_0": [.9, .6, .4, .1], "prediction": [0, 0, 1, 1]})


def test_valid_metadata_and_order():
    frame = data()
    result = b.checked_frame(frame.iloc[::-1], frame, "icbhi2017", "validation_select")
    assert result.sample_id.tolist() == ["a", "b", "c", "d"]


@pytest.mark.parametrize("column,value", [("patient_id", "bad"), ("target", 1),
    ("probability_1", np.nan), ("probability_1", 1.2), ("prediction", 1), ("locked", True)])
def test_corrupt_metadata_or_probability_rejected(column, value):
    manifest = data()
    bad = manifest.copy()
    bad.loc[0, column] = value
    with pytest.raises(AssertionError):
        b.checked_frame(bad, manifest, "icbhi2017", "validation_select")


def test_incomplete_coverage_rejected():
    with pytest.raises(AssertionError):
        b.checked_frame(data().iloc[:3], data(), "icbhi2017", "validation_select")


def test_interval_does_not_drop_invalid_replicate():
    result = b.interval([.1, np.nan, .2], .15)
    assert result["lower"] is None and result["valid_replicates"] == 2 and result["invalid_replicates"] == 1


def test_known_interval_and_zero_paired_difference():
    assert b.interval([0]*4000, 0)["upper"] == 0
    y = np.array([0, 1, 0, 1])
    p = np.array([.1, .9, .8, .3])
    assert b._as(y, p) == .5
    resamples = b._patient_resamples(np.array(["p1", "p1", "p2", "p2"]))
    assert len(resamples) == 4000
    assert all(b._as(y[ix], p[ix]) - b._as(y[ix], p[ix]) == 0 for ix in resamples)


def test_positive_temperature_and_regression_identity():
    p = np.linspace(.1, .9, 20)
    y = np.tile([0, 1], 10)
    scaled = b.TemperatureScaler(1.7).transform(p)
    assert np.array_equal(p >= .5, scaled >= .5)
    raw_fit, scaled_fit = b.regression(y, p, independent=True), b.regression(y, scaled, independent=True)
    np.testing.assert_allclose(scaled_fit["calibration_slope"], raw_fit["calibration_slope"] * 1.7, atol=1e-6)


def test_nonestimability_explicit_not_infinite():
    fit = b.regression(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
    assert fit["status"] == "nonestimable" and fit["calibration_slope"] is None
