"""Synthetic metadata, support and summary tests; no study-model outcome reads."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_revision_stability_panel_20260906 as panel
from respiratory_sound.models.pretrained_audio import _beats_token_valid_mask


def test_selection_is_invariant_to_order_and_predictions_and_excludes_heldout():
    frame = pd.DataFrame({"sample_id": ["a", "b", "c"], "dataset": ["icbhi2017"] * 3,
        "binary_label_id": [0, 0, 0], "protocol_role": ["validation_select", "validation_select", "locked_test"],
        "event_duration_seconds": [1., 1., 1.], "probability_1": [.1, .9, .4]})
    first = panel.select_ids(frame, ["known_edge"])[0]
    frame.probability_1 = [1., 0., .9]
    assert panel.select_ids(frame.iloc[::-1], ["known_edge"])[0] == first
    assert "c" not in first and "known_edge" in first and len(first) == 2


@pytest.mark.parametrize("length", [1, 2016, 2192, 127999, 128000])
def test_offsets_are_legal_unique_and_have_centre(length):
    values = panel.offsets("synthetic", length)
    starts = [v["start_sample"] for v in values]
    assert starts[0] == (128000 - length) // 2
    assert len(set(starts)) == len(starts) and all(0 <= s <= 128000 - length for s in starts)


def test_documented_mapping_cases_and_full_support():
    assert panel.geometry(2016, 114458)["legacy_valid_tokens"] == 7
    assert panel.geometry(2016, 125554)["legacy_valid_tokens"] == 0
    assert panel.geometry(2016, 114458)["exact_valid_tokens"] == 16
    assert panel.geometry(2016, 125554)["exact_valid_tokens"] == 16
    full = panel.geometry(128000, 0)
    assert full["fully_valid_tokens"] == 392 and full["partially_valid_tokens"] == 8
    assert full["fully_padding_tokens"] == 0


def test_empty_support_fails_closed():
    with pytest.raises(ValueError):
        _beats_token_valid_mask(torch.zeros(1, 128000, dtype=torch.bool),
                               fbank_frames=798, patch_size=16, frequency_patches=8)


def test_missing_position_is_not_silently_dropped_from_summary():
    rows = [{"placement": "centre", "probability_1": .4}, {"placement": "right", "probability_1": None}]
    value = panel.placement_summary(rows)
    assert value["planned_positions"] == 2 and value["nonfinite_positions"] == 1
    assert value["probability_range"] is None and value["class_flip_rate_vs_centre"] is None


def test_finite_placement_summary_and_half_threshold():
    rows = [{"placement": "centre", "probability_1": .5}, {"placement": "right", "probability_1": .25}]
    value = panel.placement_summary(rows)
    assert value["probability_range"] == .25 and value["class_flip_rate_vs_centre"] == .5
    assert value["probability_variance"] == pytest.approx(np.var([.5, .25]))


def test_nonfinite_hook_counts_are_preserved():
    value = panel.finite_stats((torch.tensor([1., float("nan")]), None, torch.tensor([float("inf")])))
    assert value == {"elements": 3, "nonfinite": 2}
