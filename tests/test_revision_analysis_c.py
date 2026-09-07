"""Synthetic Analysis C tests; no study file reads, inference or training."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_revision_analysis_c as c
from audit_analysis_c_final import manual_classification, manual_probability, assert_interval


def records():
    return {key:{"modes":{mode:{"maximum_absolute_probability_difference":2e-8,"changed_0_5_decisions":0} for mode in c.MODES},
                 "threads1_vs_training_path":{"maximum_absolute_probability_difference":0.,"changed_0_5_decisions":0}}
            for key in c.expected_cases()}


def test_complete_replay_passes():
    c.require_replay(records())


def test_missing_case_rejected():
    value=records(); value.pop(next(iter(value)))
    with pytest.raises(AssertionError): c.require_replay(value)


@pytest.mark.parametrize("mode",[c.MODES[0],c.MODES[2]])
def test_same_path_discrepancy_rejected(mode):
    value=records(); value[next(iter(value))]["modes"][mode]["maximum_absolute_probability_difference"]=2e-6
    with pytest.raises(AssertionError): c.require_replay(value)


def test_cross_path_flip_requires_review():
    value=records(); value[next(iter(value))]["modes"][c.MODES[1]]["changed_0_5_decisions"]=1
    c.require_replay(value,require_cross_decisions=False)
    with pytest.raises(AssertionError): c.require_replay(value)


@pytest.mark.parametrize("mode",[c.MODES[0],c.MODES[2]])
def test_same_path_classification_flip_rejected(mode):
    value=records(); value[next(iter(value))]["modes"][mode]["changed_0_5_decisions"]=1
    with pytest.raises(AssertionError): c.require_replay(value)


def test_missing_or_misaligned_patient_pair_rejected():
    frame=pd.DataFrame({"sample_id":["a","b"],"patient_id":["p","p"],"target":[0,1],"probability_1":[.2,.8]})
    data={(family,domain,role):frame.copy() for family in c.FAMILIES for domain,role in c.b.PRIMARY.items()}
    data[c.FAMILY,"icbhi2017","locked_test"].loc[0,"patient_id"]="wrong"
    with pytest.raises(AssertionError): c.paired_contrasts(data)


def test_paired_contrast_direction_and_all_draws():
    frame=pd.DataFrame({"sample_id":["a","b","c","d"],"patient_id":["p1","p1","p2","p2"],
                        "target":[0,1,0,1],"probability_1":[.2,.8,.3,.9]})
    data={(family,domain,role):frame.copy() for family in c.FAMILIES for domain,role in c.b.PRIMARY.items()}
    for domain,role in c.b.PRIMARY.items(): data[c.FAMILY,domain,role]["probability_1"]=[.8,.2,.7,.1]
    result,draws=c.paired_contrasts(data)
    for domain in c.b.PRIMARY:
        assert result[domain]["fullft_average_score"]==1 and result[domain]["matched_lora_average_score"]==0
        np.testing.assert_array_equal(draws[domain],np.ones(4000))
    assert_interval(result["equal_database_mean"],np.ones(4000),1.)


def test_invalid_replicates_never_silently_dropped():
    record=c.b.interval([0.,np.nan,1.],.5)
    assert_interval(record,[0.,np.nan,1.],.5)
    assert record["lower"] is None and record["invalid_replicates"]==1


def test_independent_metrics_match_formal_helpers():
    y=np.array([0,1,0,1,0,1]); p=np.array([.1,.3,.6,.9,.2,.8])
    frame=pd.DataFrame({"target":y})
    for k,v in manual_classification(y,p).items():
        assert c.b._classification(frame,p)[k]==pytest.approx(v,abs=1e-12)
    assert manual_probability(y,p)==pytest.approx(c.probability_metrics(y,p,bins=15),abs=1e-12)


def test_frozen_scope():
    assert c.FAMILIES==("exact","matched-lora")
    assert c.PREFIXES=={20260729:7,20260730:9,20260731:6}
    assert c.b.ITERATIONS==4000 and c.b.BOOTSTRAP_SEED==20260729
    assert c.PRED!=c.b.PRED and c.OUTPUT!=c.b.OUTPUT and c.FREEZE!=c.b.FREEZE
