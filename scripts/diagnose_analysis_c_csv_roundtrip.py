#!/usr/bin/env python3
"""Read-only validation-artifact diagnosis; no model or formal analysis rerun."""
from pathlib import Path
import numpy as np
import pandas as pd
import run_revision_analysis_c as c
from respiratory_sound.gate11a import sha256_file


def comparison(left, right):
    delta=np.abs(left-right)
    return {"maximum_absolute_probability_difference":float(delta.max()),
            "mean_absolute_probability_difference":float(delta.mean()),
            "events_above_1e_6":int(np.sum(delta>1e-6)),
            "changed_0_5_decisions":int(np.sum((left>=.5)!=(right>=.5)))}


def main():
    c.verify()
    validation=c.verify_replay()
    output=c.BASE/"analysis_c_csv_roundtrip_diagnostic_2026_09_06"
    assert not output.exists()
    result_path=c.OUTPUT/"analysis_c_final.json"
    protected_result=sha256_file(result_path)
    inputs={str(c.VALIDATION/"audit_final.json"):sha256_file(c.VALIDATION/"audit_final.json")}
    cases=[]
    for seed in c.ALLOWED_SEEDS:
        for domain in c.b.ROLES:
            source=c._run_dir(c.ROOT,c.FAMILY,seed)/f"best_validation_predictions_{domain}.csv"
            paths={mode:c.VALIDATION/f"seed{seed}_{domain}_{mode}.csv" for mode in c.MODES}
            old=pd.read_csv(source).sort_values("sample_id").reset_index(drop=True)
            frames={mode:pd.read_csv(path).sort_values("sample_id").reset_index(drop=True) for mode,path in paths.items()}
            inputs[str(source)]=sha256_file(source)
            inputs.update({str(path):sha256_file(path) for path in paths.values()})
            stored=validation["records"][f"{seed}/{domain}"]
            pairs=[(mode,frames[mode],old,stored["modes"][mode],False) for mode in c.MODES]
            pairs.append(("threads1_vs_training_path",frames[c.MODES[2]],frames[c.MODES[0]],stored["threads1_vs_training_path"],True))
            for mode,left,right,record,right_was_float32 in pairs:
                assert left.sample_id.equals(right.sample_id) and left.target.equals(right.target)
                p,q=left.probability_1.to_numpy(),right.probability_1.to_numpy()
                csv=comparison(p,q)
                # Both producers use softmax CPU tensors -> numpy -> dataframe (float32).
                # The frozen selection CSV was already read as float64 at formal comparison.
                # For mode-to-mode comparison both operands were still float32 in memory.
                restored=comparison(p.astype(np.float32),q.astype(np.float32) if right_was_float32 else q)
                error={name:abs(restored[name]-record[name]) for name in restored}
                assert error["maximum_absolute_probability_difference"]<=1e-14
                assert error["mean_absolute_probability_difference"]<=1e-14
                assert error["events_above_1e_6"]==0 and error["changed_0_5_decisions"]==0
                cases.append({"seed":seed,"domain":domain,"comparison":mode,
                    "recorded_in_memory":{name:record[name] for name in restored},
                    "csv_read_as_float64":csv,"producer_dtype_reconstruction":restored,
                    "maximum_absolute_record_reconstruction_error":max(error.values())})
    c.verify()
    assert sha256_file(result_path)==protected_result
    output.mkdir()
    c.write_json(output/"diagnostic.json",{"status":"all_24_recorded_path_comparisons_reconstructed_from_source_dtypes",
        "diagnosed_at":c.now(),"comparisons":cases,"source_script_sha256":sha256_file(Path(__file__).resolve()),
        "inputs_sha256":inputs,"analysis_c_result_unchanged_sha256":protected_result,
        "frozen_audit_script_unchanged_sha256":sha256_file(c.ROOT/"scripts/audit_analysis_c_final.py"),
        "no_training_or_inference":True,"validation_only_diagnostic":True,
        "no_formal_metric_or_interval_change":True,"formal_independent_audit_status":"failed_pending_prospective_implementation_correction",
        "observed_failed_check":{"location":"audit_analysis_c_final.py:audit_replay, first maximum-difference check",
            "recorded_value":2.9423522951432806e-8,"CSV_recomputed_value":0.,"tolerance":1e-14},
        "conclusion":"audit compared float32 in-memory diagnostics with float64 CSV round-trip values; precision domains differ"})
    print({"diagnostic":str(output/"diagnostic.json"),"sha256":sha256_file(output/"diagnostic.json"),
           "comparisons":len(cases),"maximum_reconstruction_error":max(x["maximum_absolute_record_reconstruction_error"] for x in cases)},flush=True)


if __name__=="__main__":
    main()
