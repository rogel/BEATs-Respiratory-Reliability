import math

import numpy as np
import pandas as pd

from respiratory_sound.data.sampling import (
    DomainClassEventBatchSampler,
    DomainClassPatientBatchSampler,
    EventRandomBatchSampler,
)


def _sampling_rows() -> pd.DataFrame:
    rows = []
    for domain in ("a", "b"):
        for label in (0, 1):
            for patient, events in (("short", 1), ("long", 20)):
                rows.extend(
                    {
                        "dataset": domain,
                        "binary_label_id": label,
                        "patient_id": f"{domain}_{label}_{patient}",
                    }
                    for _ in range(events)
                )
    return pd.DataFrame(rows)


def test_hierarchical_sampler_balances_every_domain_class_stratum() -> None:
    rows = _sampling_rows()
    sampler = DomainClassPatientBatchSampler(
        rows,
        batch_size=16,
        samples_per_epoch=160,
        seed=17,
    )

    for batch in sampler:
        selected = rows.loc[batch]
        counts = selected.groupby(["dataset", "binary_label_id"]).size()
        assert set(counts.index) == {("a", 0), ("a", 1), ("b", 0), ("b", 1)}
        assert np.array_equal(counts.to_numpy(), np.full(4, 4))


def test_hierarchical_sampler_prevents_long_patient_event_domination() -> None:
    rows = _sampling_rows()
    sampler = DomainClassPatientBatchSampler(
        rows,
        batch_size=40,
        samples_per_epoch=4_000,
        seed=19,
    )
    selected_indices = [index for batch in sampler for index in batch]
    patient_counts = rows.loc[selected_indices, "patient_id"].value_counts()
    ratios = []
    for domain in ("a", "b"):
        for label in (0, 1):
            ratios.append(
                patient_counts[f"{domain}_{label}_long"]
                / patient_counts[f"{domain}_{label}_short"]
            )

    assert all(0.8 < ratio < 1.25 for ratio in ratios)


def test_hierarchical_sampler_is_epoch_deterministic() -> None:
    rows = _sampling_rows()
    sampler = DomainClassPatientBatchSampler(
        rows,
        batch_size=16,
        samples_per_epoch=64,
        seed=23,
    )
    first = list(sampler)
    sampler.set_epoch(0)
    repeated = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)

    assert first == repeated
    assert first != second_epoch


def test_domain_class_event_sampler_balances_strata_but_preserves_event_bias() -> None:
    rows = _sampling_rows()
    sampler = DomainClassEventBatchSampler(
        rows,
        batch_size=40,
        samples_per_epoch=4_000,
        seed=29,
    )
    selected_indices = [index for batch in sampler for index in batch]
    selected = rows.loc[selected_indices]
    strata_counts = selected.groupby(["dataset", "binary_label_id"]).size()
    assert np.array_equal(strata_counts.to_numpy(), np.full(4, 1_000))
    patient_counts = selected["patient_id"].value_counts()
    ratios = []
    for domain in ("a", "b"):
        for label in (0, 1):
            ratios.append(
                patient_counts[f"{domain}_{label}_long"]
                / patient_counts[f"{domain}_{label}_short"]
            )
    assert all(ratio > 10 for ratio in ratios)


def test_event_random_sampler_matches_steps_and_is_epoch_deterministic() -> None:
    rows = _sampling_rows()
    sampler = EventRandomBatchSampler(
        rows,
        batch_size=16,
        samples_per_epoch=len(rows),
        seed=31,
    )
    first = list(sampler)
    sampler.set_epoch(0)
    repeated = list(sampler)
    sampler.set_epoch(1)
    second_epoch = list(sampler)

    assert len(first) == math.ceil(len(rows) / 16)
    assert all(len(batch) == 16 for batch in first)
    assert first == repeated
    assert first != second_epoch
