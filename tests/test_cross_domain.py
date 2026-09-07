from __future__ import annotations

import pandas as pd

from respiratory_sound.data.cross_domain import DEVELOPMENT_ROLES, patient_partition


def _toy_frame() -> pd.DataFrame:
    rows = []
    for patient_index in range(30):
        for label in (0, 1):
            rows.append(
                {
                    "patient_id": f"p{patient_index:02d}",
                    "binary_label_id": label,
                    "domain": f"d{patient_index % 3}",
                }
            )
    return pd.DataFrame(rows)


def test_patient_partition_is_deterministic_and_disjoint() -> None:
    frame = _toy_frame()
    mapping_a, metadata_a = patient_partition(
        frame,
        domain_column="domain",
        seed=42,
        attempts=128,
    )
    mapping_b, metadata_b = patient_partition(
        frame,
        domain_column="domain",
        seed=42,
        attempts=128,
    )
    assert mapping_a == mapping_b
    assert metadata_a == metadata_b
    assert set(mapping_a.values()) == set(DEVELOPMENT_ROLES)
    assert set(mapping_a) == set(frame["patient_id"])


def test_patient_partition_preserves_both_labels() -> None:
    frame = _toy_frame()
    mapping, _ = patient_partition(
        frame,
        domain_column="domain",
        seed=7,
        attempts=64,
    )
    assigned = frame.assign(protocol_role=frame["patient_id"].map(mapping))
    for role in DEVELOPMENT_ROLES:
        assert set(
            assigned.loc[assigned["protocol_role"] == role, "binary_label_id"]
        ) == {0, 1}
