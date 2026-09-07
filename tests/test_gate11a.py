from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from respiratory_sound.gate11a import (
    ALLOWED_SEEDS,
    EXPECTED_DOMAIN_ROLE_COUNTS,
    TRAIN_ROLE,
    VALIDATION_ROLE,
    assert_gate11a_protocol,
    assert_selected_rows,
    sha256_python_tree,
    single_seed_decision,
    three_seed_decision,
)


def _protocol_manifest() -> pd.DataFrame:
    rows = []
    sample_index = 0
    for (domain, role), count in EXPECTED_DOMAIN_ROLE_COUNTS.items():
        patient_prefix = "train" if role == TRAIN_ROLE else "validation"
        for index in range(count):
            rows.append(
                {
                    "sample_id": f"sample-{sample_index}",
                    "dataset": domain,
                    "patient_id": f"{domain}-{patient_prefix}-{index % 7}",
                    "protocol_role": role,
                    "locked": False,
                    "binary_label_id": index % 2,
                }
            )
            sample_index += 1
    return pd.DataFrame(rows)


def _summary(
    seed: int,
    *,
    icbhi: tuple[float, float, float] = (0.70, 0.70, 0.70),
    sprsound: tuple[float, float, float] = (0.88, 0.86, 0.90),
    audits: bool = True,
) -> dict:
    return {
        "seed": seed,
        "best_validation_metrics": {
            "icbhi2017": {
                "average_score": icbhi[0],
                "sensitivity": icbhi[1],
                "specificity": icbhi[2],
            },
            "sprsound2022": {
                "average_score": sprsound[0],
                "sensitivity": sprsound[1],
                "specificity": sprsound[2],
            },
        },
        "finite_audit_passed": audits,
        "data_boundary_audit_passed": audits,
        "exact_mask_audit_passed": audits,
    }


def test_gate11a_protocol_accepts_only_frozen_roles() -> None:
    audit = assert_gate11a_protocol(
        _protocol_manifest(),
        train_role=TRAIN_ROLE,
        validation_role=VALIDATION_ROLE,
    )

    assert audit["selected_rows"] == 9_082
    assert audit["train_validation_patient_overlap"] == 0
    assert not audit["calibration_selected"]
    assert not audit["locked_selected"]


def test_gate11a_protocol_rejects_role_substitution() -> None:
    with pytest.raises(ValueError, match="train_fit and validation_select"):
        assert_gate11a_protocol(
            _protocol_manifest(),
            train_role=TRAIN_ROLE,
            validation_role="calibration",
        )


def test_gate11a_selected_rows_reject_locked_sample() -> None:
    rows = _protocol_manifest()
    selected = rows.loc[rows["protocol_role"].eq(TRAIN_ROLE)].copy()
    selected.loc[selected.index[0], "locked"] = True

    with pytest.raises(ValueError, match="locked"):
        assert_selected_rows(selected, expected_role=TRAIN_ROLE)


def test_gate11a_single_seed_requires_numerical_and_audit_checks() -> None:
    passing = single_seed_decision(_summary(ALLOWED_SEEDS[0]))
    unsafe = single_seed_decision(
        _summary(
            ALLOWED_SEEDS[0],
            icbhi=(0.70, 0.49, 0.91),
        )
    )
    missing_audit = single_seed_decision(
        _summary(ALLOWED_SEEDS[0], audits=False)
    )

    assert passing["passed"]
    assert not unsafe["passed"]
    assert not missing_audit["passed"]


def test_gate11a_three_seed_requires_two_individual_passes() -> None:
    summaries = [
        _summary(
            ALLOWED_SEEDS[0],
            icbhi=(0.72, 0.72, 0.72),
            sprsound=(0.90, 0.88, 0.92),
        ),
        _summary(
            ALLOWED_SEEDS[1],
            icbhi=(0.72, 0.72, 0.72),
            sprsound=(0.90, 0.88, 0.92),
        ),
        _summary(
            ALLOWED_SEEDS[2],
            icbhi=(0.68, 0.68, 0.68),
            sprsound=(0.88, 0.86, 0.90),
        ),
    ]

    decision = three_seed_decision(summaries)

    assert decision["passed"]
    assert (
        sum(item["passed"] for item in decision["single_seed_decisions"])
        == 2
    )


def test_gate11a_python_tree_hash_includes_relative_paths(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    nested = tmp_path / "nested"
    nested.mkdir()
    second = nested / "b.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second.write_text("value = 2\n", encoding="utf-8")
    initial = sha256_python_tree(tmp_path)

    second.write_text("value = 3\n", encoding="utf-8")

    assert sha256_python_tree(tmp_path) != initial
