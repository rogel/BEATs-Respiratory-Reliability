#!/usr/bin/env python3
"""Independent result QA for the author-approved corrected Analysis B entry.

Recompute from frozen probabilities using count-based classification and explicit
patient multiplicities, not the formal metrics/bootstrap helpers. No new inference
or model/threshold/calibrator selection. Only a separate audit JSON is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from run_revision_analysis_b_path_corrected import ROOT, PRED, OUTPUT, JOB, AMENDMENT, verify
from run_revision_analysis_b import now, write_json
from respiratory_sound.gate11a import sha256_file

ROLES = {"icbhi2017": ("validation_select", "calibration", "locked_test"),
         "sprsound2022": ("validation_select", "calibration", "locked_inter_test", "locked_intra_test")}
PRIMARY = {"icbhi2017": "locked_test", "sprsound2022": "locked_inter_test"}
SEEDS = (20260729, 20260730, 20260731)
NAMES = ("negative_log_likelihood", "brier_score", "equal_frequency_ece")


def manual_classification(y, p):
    prediction = p >= .5
    tp, tn = np.sum(prediction & (y == 1)), np.sum(~prediction & (y == 0))
    fp, fn = np.sum(prediction & (y == 0)), np.sum(~prediction & (y == 1))
    se, sp = tp/(tp+fn), tn/(tn+fp)
    f1_pos = 2*tp/(2*tp+fp+fn)
    f1_neg = 2*tn/(2*tn+fp+fn)
    return {"sensitivity": float(se), "specificity": float(sp), "average_score": float((se+sp)/2),
            "macro_f1": float((f1_pos+f1_neg)/2), "auroc": float(roc_auc_score(y, p)),
            "accuracy": float((tp+tn)/len(y))}


def manual_probability(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    order = np.argsort(p, kind="stable")
    ece = sum(len(ix)*abs(p[ix].mean()-y[ix].mean())/len(y) for ix in np.array_split(order, min(15,len(y))))
    return dict(zip(NAMES, (float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
                            float(np.mean((p-y)**2)), float(ece)), strict=True))


def manual_fit(y, p):
    p = np.clip(p, 1e-6, 1-1e-6)
    x = np.log(p)-np.log1p(-p)
    design = np.column_stack([np.ones(len(x)), x])
    def objective(b):
        z = design @ b
        return np.mean(np.logaddexp(0, z)-y*z)
    def jac(b):
        return design.T @ (expit(design @ b)-y)/len(y)
    fit = minimize(objective, [0., 0.], jac=jac, method="BFGS", options={"gtol": 1e-11, "maxiter": 1000})
    assert np.max(np.abs(jac(fit.x))) < 1e-7
    return fit.x


def patient_draws(frame, generator):
    patients = frame.patient_id.to_numpy(dtype=str)
    unique = np.unique(patients)
    indices = [np.flatnonzero(patients == patient) for patient in unique]
    for _ in range(4000):
        chosen = generator.choice(len(unique), len(unique), replace=True)
        yield np.concatenate([indices[i] for i in chosen])


def assert_interval(record, values, point):
    values = np.asarray(values)
    valid = np.isfinite(values)
    assert record["valid_replicates"] == int(valid.sum())
    assert record["invalid_replicates"] == int((~valid).sum())
    if point is None:
        assert record["estimate"] is None
    else:
        np.testing.assert_allclose(record["estimate"], point, rtol=0, atol=1e-12)
    if valid.all():
        np.testing.assert_allclose([record["lower"], record["upper"]], np.quantile(values, [.025, .975]), rtol=0, atol=1e-12)
    else:
        assert record["lower"] is None and record["upper"] is None


def main():
    verify()
    job = json.loads((JOB / "status.json").read_text())
    assert job["status"] == "completed_pending_independent_audit"
    for stage in job["stages"]:
        assert stage["exit_code"] == 0
        assert sha256_file(JOB / f"{stage['stage']}.log") == stage["log_sha256"]
    result = json.loads((OUTPUT / "analysis_b_final.json").read_text())
    assert sha256_file(OUTPUT / "analysis_b_final.json") == job["result_sha256"]
    receipt = json.loads((OUTPUT / "path_correction_execution_receipt.json").read_text())
    assert receipt["path_amendment_sha256"] == sha256_file(AMENDMENT)
    assert receipt["analysis_b_final_sha256"] == job["result_sha256"]
    for name, digest in result["output_hashes"].items():
        assert sha256_file(OUTPUT / name) == digest, name
    prediction_manifest = json.loads((PRED / "prediction_manifest.json").read_text())
    for name, digest in prediction_manifest["files"].items():
        assert sha256_file(PRED / name) == digest, name
    manifest = pd.read_csv(ROOT / "data/manifests/cross_domain_binary.csv", dtype={"patient_id": str})
    table = pd.read_csv(OUTPUT / "classification_metrics.csv")
    assert len(table) == 56
    ensembles, verified_cells = {}, 0
    for family in ("exact", "no-js"):
        for domain, roles in ROLES.items():
            for role in roles:
                expected = manifest.loc[(manifest.dataset == domain) & (manifest.protocol_role == role)].sort_values("sample_id").reset_index(drop=True)
                arrays = []
                for seed in SEEDS:
                    if family == "no-js":
                        path = PRED / f"seed{seed}_{domain}_{role}.csv"
                    elif role == "validation_select":
                        path = ROOT / f"runs/gate11a_exactmask_fullft_seed{seed}/best_validation_predictions_{domain}.csv"
                    else:
                        subdir = "calibration_predictions" if role == "calibration" else "locked_predictions"
                        path = ROOT / f"artifacts/post_gate11a/{subdir}/fullft_seed{seed}_{domain}_{role}.csv"
                    frame = pd.read_csv(path, dtype={"patient_id": str}).sort_values("sample_id").reset_index(drop=True)
                    assert not frame.sample_id.duplicated().any() and len(frame) == len(expected)
                    for column in ("sample_id", "patient_id", "dataset", "protocol_role", "locked", "binary_label_id"):
                        assert frame[column].astype(str).equals(expected[column].astype(str)), (family, seed, domain, role, column)
                    assert np.array_equal(frame.target, expected.binary_label_id)
                    y, p = frame.target.to_numpy(), frame.probability_1.to_numpy()
                    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
                    assert np.array_equal(frame.prediction, (p>=.5).astype(int))
                    arrays.append(p)
                    selected = table.loc[(table.family == family)&(table.dataset == domain)&(table.role == role)&(table.member == f"seed{seed}")]
                    assert len(selected) == 1
                    for k, v in manual_classification(y, p).items():
                        np.testing.assert_allclose(selected.iloc[0][k], v, rtol=0, atol=1e-12)
                        verified_cells += 1
                ensemble = np.mean(np.stack(arrays, axis=1), axis=1)
                ensemble_frame = expected.copy()
                ensemble_frame["target"], ensemble_frame["probability_1"] = y, ensemble
                ensembles[family, domain, role] = ensemble_frame
                selected = table.loc[(table.family==family)&(table.dataset==domain)&(table.role==role)&(table.member=="ensemble")]
                assert len(selected) == 1
                for k, v in manual_classification(y, ensemble).items():
                    np.testing.assert_allclose(selected.iloc[0][k], v, rtol=0, atol=1e-12)
                    verified_cells += 1
    draws, points = {}, []
    saved_draws = np.load(OUTPUT / "classification_bootstrap_draws.npz")
    for (domain, role), sequence in zip(PRIMARY.items(), np.random.SeedSequence(20260729).spawn(2), strict=True):
        left, right = ensembles["exact", domain, role], ensembles["no-js", domain, role]
        y, p, q = left.target.to_numpy(), left.probability_1.to_numpy(), right.probability_1.to_numpy()
        def score(y, p):
            return .5*(np.mean(p[y==1]>=.5)+np.mean(p[y==0]<.5))
        point = float(score(y,p)-score(y,q))
        values = [score(y[ix],p[ix])-score(y[ix],q[ix]) if len(np.unique(y[ix]))==2 else np.nan
                  for ix in patient_draws(left, np.random.default_rng(sequence))]
        np.testing.assert_allclose(values, saved_draws[domain], rtol=0, atol=1e-14)
        assert_interval(result["primary_contrast"][domain]["js_on_minus_no_js"], values, point)
        draws[domain] = values
        points.append(point)
    equal = np.mean(np.stack(list(draws.values())), axis=0)
    np.testing.assert_allclose(equal, saved_draws["equal_database_mean"], rtol=0, atol=1e-14)
    assert_interval(result["primary_contrast"]["equal_database_mean"], equal, np.mean(points))

    calibration_table = pd.read_csv(OUTPUT / "calibration_metrics.csv")
    assert len(calibration_table) == 28
    cal_cells, regression_checks, interval_checks = 0, 0, 0
    for (family, domain, role), frame in ensembles.items():
        y, p = frame.target.to_numpy(), frame.probability_1.to_numpy()
        temperature = result["temperatures"][f"{family}/{domain}"]
        clipped = np.clip(p, 1e-6, 1-1e-6)
        scaled = expit((np.log(clipped)-np.log1p(-clipped))/temperature)
        assert np.array_equal(p>=.5,scaled>=.5)
        point_metrics = {}
        for label, probability in (("raw",p),("temperature",scaled)):
            row = calibration_table.loc[(calibration_table.family==family)&(calibration_table.dataset==domain)&
                (calibration_table.role==role)&(calibration_table.scale==label)].iloc[0]
            point_metrics[label] = manual_probability(y, probability)
            for k,v in point_metrics[label].items():
                np.testing.assert_allclose(row[k],v,rtol=0,atol=1e-12)
                cal_cells += 1
            if row.regression_status == "finite_independently_verified":
                coefficient = manual_fit(y, probability)
                np.testing.assert_allclose(coefficient,[row.calibration_intercept,row.calibration_slope],atol=1e-6,rtol=1e-6)
                regression_checks += 1
        metric_draws = {name: [] for name in NAMES}
        for ix in patient_draws(frame,np.random.default_rng(20260729)):
            if len(np.unique(y[ix]))==2:
                a,c = manual_probability(y[ix],p[ix]),manual_probability(y[ix],scaled[ix])
                for name in NAMES:
                    metric_draws[name].append(a[name]-c[name])
            else:
                for name in NAMES:
                    metric_draws[name].append(np.nan)
        key = f"{family}/{domain}/{role}"
        for name in NAMES:
            assert_interval(result["raw_minus_temperature_probability_intervals"][key][name],metric_draws[name],
                            point_metrics["raw"][name]-point_metrics["temperature"][name])
            interval_checks += 1
        print(json.dumps({"audit_role":key,"calibration_checks_passed":True}),flush=True)
    diagnostics = json.loads((OUTPUT / "calibration_regression_diagnostics.json").read_text())
    coefficient_intervals_checked, coefficient_probes = 0, 0
    for domain, roles in ROLES.items():
        for role in roles:
            key = f"exact/{domain}/{role}"
            info = diagnostics["primary_exact_coefficient_intervals"][key]
            coefficients = np.load(OUTPUT / f"exact_coefficients_{domain}_{role}.npz")["raw"]
            assert coefficients.shape == (4000,2)
            assert set(np.flatnonzero(~np.isfinite(coefficients).all(axis=1))) == {x["replicate"] for x in info["nonestimable_replicates"]}
            point = diagnostics["point_fits"][f"{key}/raw"]
            temperature = result["temperatures"][f"exact/{domain}"]
            for j,name in enumerate(("calibration_intercept","calibration_slope")):
                multiplier = 1 if j==0 else temperature
                for label,factor in (("raw",1),("temperature",multiplier),("raw_minus_temperature",1-multiplier)):
                    assert_interval(info[label][name],coefficients[:,j]*factor,None if point[name] is None else point[name]*factor)
                    coefficient_intervals_checked += 1
            frame = ensembles["exact",domain,role]
            for iteration,ix in enumerate(patient_draws(frame,np.random.default_rng(20260729))):
                if iteration in (0,1999,3999) and np.isfinite(coefficients[iteration]).all():
                    fitted = manual_fit(frame.target.to_numpy()[ix],frame.probability_1.to_numpy()[ix])
                    np.testing.assert_allclose(fitted,coefficients[iteration],atol=1e-6,rtol=1e-6)
                    coefficient_probes += 1
    threshold_table = pd.read_csv(OUTPUT / "exact_validation_threshold_sensitivity.csv")
    assert len(threshold_table)==34 and set(threshold_table.role)=={"validation_select"}
    for _,row in threshold_table.iterrows():
        frame = ensembles["exact",row.dataset,"validation_select"]
        y,p = frame.target.to_numpy(),frame.probability_1.to_numpy()
        decision = p>=row.threshold
        # Use the same labels at an explicitly supplied decision boundary, not a selected operating point.
        expected = manual_classification(y,decision.astype(float))
        for k in ("sensitivity","specificity","average_score","macro_f1"):
            np.testing.assert_allclose(row[k],expected[k],rtol=0,atol=1e-12)
        np.testing.assert_allclose(row.prediction_prevalence,decision.mean(),atol=1e-12,rtol=0)
        np.testing.assert_allclose(row.normal_error,decision[y==0].mean(),atol=1e-12,rtol=0)
        np.testing.assert_allclose(row.adventitious_error,(~decision[y==1]).mean(),atol=1e-12,rtol=0)
    verify()
    audit = {"status":"independent_analysis_b_QA_passed","audited_at":now(),
        "classification_rows":56,"classification_cells_recomputed":verified_cells,
        "primary_bootstrap_draws_independently_reproduced":12000,"primary_intervals_reproduced":3,
        "calibration_point_cells_recomputed":cal_cells,"independent_joint_regression_point_checks":regression_checks,
        "raw_minus_temperature_intervals_independently_reproduced":interval_checks,
        "all_prediction_hashes_and_patient_metadata_verified":True,"original_freezes_verified":True,
        "result_sha256":job["result_sha256"],"path_amendment_sha256":sha256_file(AMENDMENT),
        "audit_source_sha256":sha256_file(Path(__file__).resolve()),
        "primary_coefficient_intervals_recomputed_from_all_stored_draws":coefficient_intervals_checked,
        "primary_coefficient_draws_independently_refitted":coefficient_probes,
        "validation_threshold_rows_independently_recomputed":34,
        "scientific_conclusion_review":"required before proceeding to matched-budget LoRA"}
    write_json(OUTPUT / "independent_result_audit.json",audit)
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == "__main__":
    main()
