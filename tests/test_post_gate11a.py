import numpy as np
import pandas as pd
import pytest

from respiratory_sound.post_gate11a import (
    SEEDS,
    assert_role_rows,
    probability_ensemble,
)


def _prediction_frame(seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "dataset": ["icbhi2017", "icbhi2017"],
            "patient_id": ["p1", "p2"],
            "protocol_role": ["calibration", "calibration"],
            "locked": [False, False],
            "binary_label_id": [0, 1],
            "fine_label_name": ["normal", "crackle"],
            "event_duration_seconds": [1.0, 2.0],
            "target": [0, 1],
            "probability_1": [0.1 + seed * 0.0, 0.8],
        }
    )


def test_three_seed_probability_ensemble_is_aligned() -> None:
    frames = [_prediction_frame(seed) for seed in SEEDS]
    ensemble = probability_ensemble(frames)

    assert np.allclose(ensemble["probability_1"], [0.1, 0.8])
    assert ensemble["prediction"].tolist() == [0, 1]


def test_role_assertion_distinguishes_locked_rows() -> None:
    frame = _prediction_frame(SEEDS[0])
    assert_role_rows(
        frame,
        expected_role="calibration",
        expected_domain="icbhi2017",
        expect_locked=False,
    )
    with pytest.raises(ValueError, match="locked flag"):
        assert_role_rows(
            frame,
            expected_role="calibration",
            expected_domain="icbhi2017",
            expect_locked=True,
        )
