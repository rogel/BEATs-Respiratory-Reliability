#!/usr/bin/env python3
"""Independent count-based Analysis C QA; only writes a separate audit artifact."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

import run_revision_analysis_c as c
from audit_analysis_b_final import manual_classification, manual_probability, manual_fit, patient_draws, assert_interval, NAMES
from respiratory_sound.gate11a import sha256_file


def checked_input(path, manifest, domain, role):
    frame = pd.read_csv(path,dtype={"patient_id":str}).sort_values("sample_id").reset_index(drop=True)
    expected = manifest.loc[(manifest.dataset==domain)&(manifest.protocol_role==role)].sort_values("sample_id").reset_index(drop=True)
    assert len(frame)==len(expected) and not frame.sample_id.duplicated().any()
    for name in ("sample_id","dataset","patient_id","protocol_role","locked","binary_label_id","fine_label_name"):
        assert frame[name].astype(str).equals(expected[name].astype(str)),(domain,role,name)
    np.testing.assert_allclose(frame.event_duration_seconds,expected.event_duration_seconds,atol=1e-12,rtol=1e-12)
    assert np.array_equal(frame.target,expected.binary_label_id)
    p=frame.probability_1.to_numpy()
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    assert np.array_equal(frame.prediction,p>=.5)
    np.testing.assert_allclose(frame.probability_0+p,1,atol=2e-7,rtol=0)
    return frame


def audit_replay(manifest):
    audit=c.verify_replay()
    comparisons=0
    for seed in c.ALLOWED_SEEDS:
        for domain in c.b.ROLES:
            old=checked_input(c._run_dir(c.ROOT,c.FAMILY,seed)/f"best_validation_predictions_{domain}.csv",manifest,domain,"validation_select")
            frames={mode:checked_input(c.VALIDATION/f"seed{seed}_{domain}_{mode}.csv",manifest,domain,"validation_select") for mode in c.MODES}
            record=audit["records"][f"{seed}/{domain}"]
            pairs=[(frames[mode],old,record["modes"][mode]) for mode in c.MODES]
            pairs.append((frames[c.MODES[2]],frames[c.MODES[0]],record["threads1_vs_training_path"]))
            for left,right,stored in pairs:
                delta=np.abs(left.probability_1.to_numpy()-right.probability_1.to_numpy())
                np.testing.assert_allclose(stored["maximum_absolute_probability_difference"],delta.max(),rtol=0,atol=1e-14)
                np.testing.assert_allclose(stored["mean_absolute_probability_difference"],delta.mean(),rtol=0,atol=1e-14)
                assert stored["changed_0_5_decisions"]==int(np.sum(left.prediction!=right.prediction))
                assert stored["events_above_1e_6"]==int(np.sum(delta>1e-6))
                comparisons+=1
    return comparisons


def main():
    c.verify()
    job=c.read_json(c.JOB/"status.json")
    assert job["status"]=="completed_pending_independent_audit"
    assert [s["stage"] for s in job["stages"]]==["validate","predict","analyze"]
    for stage in job["stages"]:
        assert stage["exit_code"]==0
        assert sha256_file(c.JOB/f"{stage['stage']}.log")==stage["log_sha256"]
    result=c.read_json(c.OUTPUT/"analysis_c_final.json")
    assert sha256_file(c.OUTPUT/"analysis_c_final.json")==job["result_sha256"]
    assert result["freeze_sha256"]==job["freeze_sha256"]==sha256_file(c.FREEZE)
    assert result["families"]==list(c.FAMILIES) and result["seeds"]==list(c.ALLOWED_SEEDS)
    assert not result["selection_after_locked_access"] and result["threshold"]==.5
    for name,digest in result["output_hashes"].items():
        assert sha256_file(c.OUTPUT/name)==digest,name
    sources=c.read_json(c.PRED/"prediction_manifest.json")
    assert sha256_file(c.PRED/"prediction_manifest.json")==result["prediction_manifest_sha256"]
    assert len(sources["files"])==28
    for name,digest in sources["files"].items():
        assert sha256_file(c.PRED/name)==digest,name
    for name,digest in result["input_prediction_hashes"].items():
        assert sha256_file(Path(name))==digest,name
    manifest=pd.read_csv(c.ROOT/"data/manifests/cross_domain_binary.csv",dtype={"patient_id":str})
    replay_comparisons=audit_replay(manifest)
    table=pd.read_csv(c.OUTPUT/"classification_metrics.csv")
    assert len(table)==56 and set(table.family)==set(c.FAMILIES)
    assert not table.duplicated(["family","member","dataset","role"]).any()
    ensembles,cells={},0
    for family in c.FAMILIES:
        for domain,roles in c.b.ROLES.items():
            for role in roles:
                frames=[]
                for seed in c.ALLOWED_SEEDS:
                    if family==c.FAMILY:
                        path=c.PRED/f"seed{seed}_{domain}_{role}.csv"
                    elif role=="validation_select":
                        path=c.ROOT/f"runs/gate11a_exactmask_fullft_seed{seed}/best_validation_predictions_{domain}.csv"
                    else:
                        sub="calibration_predictions" if role=="calibration" else "locked_predictions"
                        path=c.ROOT/f"artifacts/post_gate11a/{sub}/fullft_seed{seed}_{domain}_{role}.csv"
                    frame=checked_input(path,manifest,domain,role)
                    frames.append(frame)
                    row=table.loc[(table.family==family)&(table.member==f"seed{seed}")&(table.dataset==domain)&(table.role==role)]
                    assert len(row)==1
                    assert row.iloc[0]["samples"]==len(frame) and row.iloc[0]["patients"]==frame.patient_id.nunique()
                    for name,value in manual_classification(frame.target.to_numpy(),frame.probability_1.to_numpy()).items():
                        np.testing.assert_allclose(row.iloc[0][name],value,rtol=0,atol=1e-12); cells+=1
                ensemble=frames[0].copy()
                ensemble["probability_1"]=np.mean(np.column_stack([f.probability_1 for f in frames]),axis=1)
                ensembles[family,domain,role]=ensemble
                row=table.loc[(table.family==family)&(table.member=="ensemble")&(table.dataset==domain)&(table.role==role)]
                assert len(row)==1
                assert row.iloc[0]["samples"]==len(ensemble) and row.iloc[0]["patients"]==ensemble.patient_id.nunique()
                for name,value in manual_classification(ensemble.target.to_numpy(),ensemble.probability_1.to_numpy()).items():
                    np.testing.assert_allclose(row.iloc[0][name],value,rtol=0,atol=1e-12); cells+=1
                if family==c.FAMILY:
                    saved=checked_input(c.PRED/f"ensemble_{domain}_{role}.csv",manifest,domain,role)
                    np.testing.assert_allclose(ensemble.probability_1,saved.probability_1,rtol=0,atol=1e-14)
    draws,points={},[]
    saved=np.load(c.OUTPUT/"classification_bootstrap_draws.npz")
    for (domain,role),sequence in zip(c.b.PRIMARY.items(),np.random.SeedSequence(20260729).spawn(2),strict=True):
        left,right=ensembles["exact",domain,role],ensembles[c.FAMILY,domain,role]
        y,p,q=left.target.to_numpy(),left.probability_1.to_numpy(),right.probability_1.to_numpy()
        def score(y,p):
            return .5*(np.mean(p[y==1]>=.5)+np.mean(p[y==0]<.5))
        point=float(score(y,p)-score(y,q))
        values=np.asarray([score(y[ix],p[ix])-score(y[ix],q[ix]) if len(np.unique(y[ix]))==2 else np.nan
                           for ix in patient_draws(left,np.random.default_rng(sequence))])
        np.testing.assert_allclose(values,saved[domain],rtol=0,atol=1e-14)
        record=result["primary_contrast"][domain]
        np.testing.assert_allclose([record["fullft_average_score"],record["matched_lora_average_score"]],[score(y,p),score(y,q)],rtol=0,atol=1e-12)
        assert_interval(record["fullft_minus_matched_lora"],values,point)
        draws[domain]=values; points.append(point)
    equal=np.mean(np.stack(list(draws.values())),axis=0)
    np.testing.assert_allclose(equal,saved["equal_database_mean"],rtol=0,atol=1e-14)
    assert_interval(result["primary_contrast"]["equal_database_mean"],equal,np.mean(points))
    cal_table=pd.read_csv(c.OUTPUT/"calibration_metrics.csv")
    assert len(cal_table)==28 and not cal_table.duplicated(["family","dataset","role","scale"]).any()
    cal_cells,regression_checks,intervals=0,0,0
    for domain in c.b.ROLES:
        assert result["temperatures"][f"exact/{domain}"]==c.b.PROTECTED_T[domain]
        cal=ensembles[c.FAMILY,domain,"calibration"]
        x=np.log(np.clip(cal.probability_1,1e-6,1-1e-6))-np.log1p(-np.clip(cal.probability_1,1e-6,1-1e-6))
        fit=minimize_scalar(lambda z:manual_probability(cal.target.to_numpy(),expit(x/np.exp(z)))["negative_log_likelihood"],
                            bounds=(-5,5),method="bounded",options={"xatol":1e-10})
        assert fit.success
        np.testing.assert_allclose(result["temperatures"][f"{c.FAMILY}/{domain}"],np.exp(fit.x),atol=1e-6,rtol=1e-6)
    for (family,domain,role),frame in ensembles.items():
        y,p=frame.target.to_numpy(),frame.probability_1.to_numpy()
        temperature=result["temperatures"][f"{family}/{domain}"]
        clipped=np.clip(p,1e-6,1-1e-6)
        scaled=expit((np.log(clipped)-np.log1p(-clipped))/temperature)
        assert np.array_equal(p>=.5,scaled>=.5)
        points={}
        for label,values in (("raw",p),("temperature",scaled)):
            selected=cal_table.loc[(cal_table.family==family)&(cal_table.dataset==domain)&(cal_table.role==role)&(cal_table.scale==label)]
            assert len(selected)==1
            row=selected.iloc[0]; points[label]=manual_probability(y,values)
            for name,value in points[label].items():
                np.testing.assert_allclose(row[name],value,rtol=0,atol=1e-12); cal_cells+=1
            if row.regression_status=="finite_independently_verified":
                np.testing.assert_allclose(manual_fit(y,values),[row.calibration_intercept,row.calibration_slope],rtol=1e-6,atol=1e-6)
                regression_checks+=1
            else:
                assert row.regression_status=="nonestimable" and pd.isna(row.calibration_slope) and pd.isna(row.calibration_intercept)
        values={name:[] for name in NAMES}
        for ix in patient_draws(frame,np.random.default_rng(20260729)):
            if len(np.unique(y[ix]))==2:
                a,z=manual_probability(y[ix],p[ix]),manual_probability(y[ix],scaled[ix])
                for name in NAMES: values[name].append(a[name]-z[name])
            else:
                for name in NAMES: values[name].append(np.nan)
        key=f"{family}/{domain}/{role}"
        for name in NAMES:
            assert_interval(result["raw_minus_temperature_probability_intervals"][key][name],values[name],points["raw"][name]-points["temperature"][name]); intervals+=1
        print(f"Independent calibration QA passed: {key}",flush=True)
    c.verify()
    c.write_json(c.OUTPUT/"independent_result_audit.json",{"status":"independent_analysis_c_QA_passed","audited_at":c.now(),
        "classification_rows":56,"classification_cells_recomputed":cells,"primary_draws_independently_reproduced":12000,
        "primary_intervals_reproduced":3,"calibration_point_cells_recomputed":cal_cells,"regression_point_checks":regression_checks,
        "raw_minus_temperature_intervals_reproduced":intervals,"temperatures_independently_recomputed_from_calibration_only":2,
        "validation_path_comparisons_recomputed":replay_comparisons,"all_prediction_metadata_and_hashes_verified":True,
        "original_freezes_verified":True,"result_sha256":job["result_sha256"],"freeze_sha256":sha256_file(c.FREEZE),
        "audit_source_sha256":sha256_file(Path(__file__).resolve()),"scientific_conclusion_review":"required before using these results"})
    print(c.read_json(c.OUTPUT/"independent_result_audit.json"),flush=True)


if __name__=="__main__":
    main()
