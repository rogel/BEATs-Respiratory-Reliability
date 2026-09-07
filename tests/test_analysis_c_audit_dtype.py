"""Synthetic precision-domain tests; never reads study predictions or fits models."""
import ast
from io import StringIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_analysis_c_final_dtype_corrected as audit


def csv_roundtrip(values):
    return pd.read_csv(StringIO(pd.DataFrame({"p": values}).to_csv(index=False))).p.to_numpy()


def test_mixed_dtype_decimal_roundtrip_reconstructs_not_silently_zeroed():
    producer = np.array([.12345679, .8765432, .50123453], dtype=np.float32)
    saved = csv_roundtrip(producer)
    recorded = audit.numerical_comparison(producer, saved)
    assert recorded["maximum_absolute_probability_difference"] > 1e-10
    result = audit.reconstruct_comparison(saved, saved, recorded, right_was_float32=False, same_path=True)
    assert result["csv_float64_recomputation"]["maximum_absolute_probability_difference"] == 0
    assert result["original_producer_dtype_reconstruction"] == recorded


def test_double_float32_subtraction_keeps_original_mean_precision():
    left = np.array([.12345679, .8765432, .50123453], dtype=np.float32)
    right = np.array([.12145679, .8765431, .50123465], dtype=np.float32)
    recorded = audit.numerical_comparison(left, right)
    result = audit.reconstruct_comparison(csv_roundtrip(left), csv_roundtrip(right), recorded,
                                         right_was_float32=True, same_path=False)
    assert result["original_producer_dtype_reconstruction"] == recorded
    assert result["csv_float64_recomputation"]["maximum_absolute_probability_difference"] > 1e-6
    assert result["producer_dtypes"] == ["float32", "float32"]


def test_wrong_precision_domain_is_rejected():
    producer = np.array([.12345679, .8765432], dtype=np.float32)
    saved = csv_roundtrip(producer)
    recorded = audit.numerical_comparison(producer, saved)
    with pytest.raises(AssertionError):
        audit.reconstruct_comparison(saved, saved, recorded, right_was_float32=True, same_path=True)


@pytest.mark.parametrize("field", ["maximum_absolute_probability_difference", "mean_absolute_probability_difference",
                                  "events_above_1e_6", "changed_0_5_decisions"])
def test_tampered_diagnostic_is_rejected(field):
    values = np.array([.2, .8], dtype=np.float32)
    saved = csv_roundtrip(values)
    recorded = audit.numerical_comparison(values, saved)
    recorded[field] += 1
    with pytest.raises(AssertionError):
        audit.reconstruct_comparison(saved, saved, recorded, right_was_float32=False, same_path=True)


@pytest.mark.parametrize("bad", [[np.nan, .2], [np.inf, .2], [-.01, .2], [1.01, .2], []])
def test_invalid_probability_is_rejected(bad):
    with pytest.raises(AssertionError):
        audit.numerical_comparison(bad, bad)


def test_same_path_gate_is_not_relaxed():
    left = np.array([.12345979, .8765432], dtype=np.float32)
    right = csv_roundtrip(np.array([.12345679, .8765432], dtype=np.float32))
    recorded = audit.numerical_comparison(left, right)
    with pytest.raises(AssertionError):
        audit.reconstruct_comparison(csv_roundtrip(left), right, recorded,
                                     right_was_float32=False, same_path=True)


def test_threshold_boundary_uses_greater_equal_and_preserves_flips():
    below = np.nextafter(np.float32(.5), np.float32(0))
    left = np.array([below, .5], dtype=np.float32)
    right = np.array([.5, below], dtype=np.float32)
    recorded = audit.numerical_comparison(left, right)
    assert recorded["changed_0_5_decisions"] == 2
    result = audit.reconstruct_comparison(csv_roundtrip(left), csv_roundtrip(right), recorded,
                                         right_was_float32=True, same_path=False)
    assert result["csv_float64_recomputation"]["changed_0_5_decisions"] == 2
    with pytest.raises(AssertionError):
        audit.reconstruct_comparison(csv_roundtrip(left), csv_roundtrip(right), recorded,
                                     right_was_float32=True, same_path=True)


def test_statistical_recomputation_body_and_metadata_validation_unchanged():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    original = (scripts / "audit_analysis_c_final.py").read_text()
    revised = (scripts / "audit_analysis_c_final_dtype_corrected.py").read_text()
    start = '    table=pd.read_csv(c.OUTPUT/"classification_metrics.csv")'
    stop = '    c.verify()\n'
    old_body = original[original.index(start):original.rindex(stop)]
    new_body = revised[revised.index(start):revised.rindex(stop)]
    assert old_body == new_body
    def checked_node(source):
        return next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "checked_input")
    assert ast.dump(checked_node(original)) == ast.dump(checked_node(revised))
