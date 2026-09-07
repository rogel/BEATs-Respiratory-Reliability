"""Analytical QA identities on synthetic data; no study-outcome access."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_analysis_b_final import manual_classification, manual_probability, patient_draws, assert_interval
from respiratory_sound.calibration import probability_metrics


def test_known_classification_counts():
    got = manual_classification(np.array([0,0,1,1]),np.array([.2,.6,.6,.7]))
    assert got["specificity"]==.5 and got["sensitivity"]==1 and got["average_score"]==.75


def test_probability_small_bins_and_extremes():
    y,p = np.array([0,1,0,1]),np.array([0.,1.,.5,.6])
    a,b = manual_probability(y,p),probability_metrics(y,p,bins=15)
    for name in a:
        np.testing.assert_allclose(a[name],b[name],atol=1e-15,rtol=0)


def test_cluster_draws_keep_events_together():
    frame = pd.DataFrame({"patient_id":["a","a","b","b"]})
    draws = list(patient_draws(frame,np.random.default_rng(20260729)))
    assert len(draws)==4000
    for ix in draws:
        assert np.sum(ix==0)==np.sum(ix==1) and np.sum(ix==2)==np.sum(ix==3)


def test_interval_preserves_nonestimability():
    record = {"estimate":None,"valid_replicates":2,"invalid_replicates":1,"lower":None,"upper":None}
    assert_interval(record,[.1,np.nan,.2],None)
