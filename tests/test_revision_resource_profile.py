"""Synthetic-only cost-measurement tests, no model checkpoint or study event reads."""
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_revision_resource_profile_20260906 as profile


def test_latency_units_and_percentile():
    value = profile.summarize_times(np.arange(1, 21), 8)
    assert value["batch_latency_median_seconds"] == 10.5
    assert value["batch_latency_p95_seconds"] == pytest.approx(19.05)
    assert value["per_event_latency_median_seconds"] == 10.5 / 8
    assert value["throughput_events_per_second"] == 8 / 10.5


@pytest.mark.parametrize("values", [[1.] * 19, [1.] * 19 + [np.nan], [1.] * 19 + [0.], [1.] * 19 + [-1.]])
def test_no_dropping_bad_iterations(values):
    with pytest.raises(AssertionError): profile.summarize_times(values, 1)


def test_member_count_and_failed_legacy_not_substituted():
    assert profile.MEMBERS["legacy-mask"] == (20260729, 20260731)
    for family in profile.FAMILIES[:-1]: assert profile.MEMBERS[family] == (20260729, 20260730, 20260731)


def test_synthetic_input_is_reproducible_and_all_valid():
    a, ma, ha = profile.synthetic_input(1, torch.device("cpu"))
    b, mb, hb = profile.synthetic_input(1, torch.device("cpu"))
    assert ha == hb and torch.equal(a, b) and torch.equal(ma, mb)
    assert a.shape == (1, 128000) and ma.all() and a.min() >= -1 and a.max() <= 1


class Tiny(torch.nn.Module):
    def forward(self, waveform, support):
        mean = (waveform * support).mean(1)
        return torch.stack([mean, -mean], -1)


def test_timing_smoke_with_synthetic_tiny_model():
    result = profile.time_models([Tiny(), Tiny()], 1, torch.device("cpu"))
    assert result["finite_passes"] == 25 and len(result["timings_seconds"]) == 20
    assert result["maximum_sampled_mps_current_bytes"] is None


def test_nonfinite_model_fails_warmup():
    class Bad(Tiny):
        def forward(self, waveform, support):
            return super().forward(waveform, support) * float("nan")
    with pytest.raises(FloatingPointError):
        profile.time_models([Bad()], 1, torch.device("cpu"))
