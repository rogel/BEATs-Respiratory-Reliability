"""Synthetic gate tests; no access to study predictions or scientific outcomes."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_revision_analysis_b_path_corrected import require_validation_audit


def audit():
    record = {"modes": {m: {"maximum_absolute_probability_difference": 3e-8, "changed_0_5_decisions": 0}
        for m in ("training_evaluate_workers4", "infer_workers0_threads1")},
        "threads1_vs_training_path": {"maximum_absolute_probability_difference": 0.0}}
    return {"status": "complete_all_same_path_checks_passed", "calibration_accessed": False,
        "heldout_accessed": False, "no_training_or_selection": True,
        "records": {f"{f}/{s}/{d}": deepcopy(record) for f in ("exact", "no-js")
            for s in (20260729, 20260730, 20260731) for d in ("icbhi2017", "sprsound2022")}}


def test_complete_audit_passes():
    assert len(require_validation_audit(audit())["records"]) == 12


@pytest.mark.parametrize("field", ["missing_case", "large_same_path_error", "decision_flip", "heldout_access"])
def test_bad_audit_blocks_formal_inference(field):
    value = audit()
    key = "no-js/20260730/sprsound2022"
    if field == "missing_case":
        del value["records"][key]
    elif field == "large_same_path_error":
        value["records"][key]["modes"]["training_evaluate_workers4"]["maximum_absolute_probability_difference"] = .002
    elif field == "decision_flip":
        value["records"][key]["modes"]["infer_workers0_threads1"]["changed_0_5_decisions"] = 1
    else:
        value["heldout_accessed"] = True
    with pytest.raises(AssertionError):
        require_validation_audit(value)
