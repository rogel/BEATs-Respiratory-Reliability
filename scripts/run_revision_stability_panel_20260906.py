#!/usr/bin/env python3
"""Prespecified development-only diagnostic panel. Never updates model parameters."""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_revision_analysis_c as c
import run_revision_resource_profile_20260906 as resource
from audit_revision_mask_mapping import _stable_offset
from compute_feature_stats import feature_config_from_yaml
from respiratory_sound.gate11a import sha256_file
from respiratory_sound.models.pretrained_audio import (
    _beats_token_support_fraction, _beats_token_valid_mask, _beats_legacy_token_valid_mask,
)
from respiratory_sound.post_gate11a import build_waveform_dataset
from respiratory_sound.training import jensen_shannon_consistency

FREEZE = c.BASE / "stability_panel_freeze_2026_09_06.json"
OUTPUT = c.BASE / "stability_panel_2026_09_06"
SPEC = c.REPORTS / "25_STABILITY_PLACEMENT_EXECUTION_SPEC_2026-09-06.md"
MAPPING = c.BASE / "mask_mapping/exact_legacy_mapping_event_placements.csv"
FAILURE = c.BASE / "failures/legacy_mask_seed20260730_epoch6_mask_replay.json"
BINS = (-1, .5, 1, 2, 4, 8, float("inf"))
BIN_NAMES = ("0_to_0.5", "0.5_to_1", "1_to_2", "2_to_4", "4_to_8", "over_8")
MODEL_MEMBERS = {"exact": tuple(c.ALLOWED_SEEDS), "legacy-mask": (20260729, 20260731)}
N = 128000


def select_ids(manifest, zero_ids):
    # This selector reads only prespecified metadata, never a model output column.
    rows = manifest.loc[manifest.protocol_role == "validation_select",
                        ["sample_id", "dataset", "binary_label_id", "event_duration_seconds"]].copy()
    rows["duration_bin"] = pd.cut(rows.event_duration_seconds, BINS, labels=BIN_NAMES)
    rows["selection_key"] = rows.sample_id.map(lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    chosen = rows.sort_values(["selection_key", "sample_id"]).groupby(
        ["dataset", "binary_label_id", "duration_bin"], observed=True, sort=True).head(1)
    selected = sorted(set(chosen.sample_id.astype(str)) | set(zero_ids))
    occupied = {(str(r.dataset), int(r.binary_label_id), str(r.duration_bin)) for r in chosen.itertuples()}
    empty = [list(key) for domain in c.b.ROLES for target in (0, 1) for bin_name in BIN_NAMES
             if (key := (domain, target, bin_name)) not in occupied]
    return selected, chosen.drop(columns="selection_key").to_dict("records"), empty


def offsets(sample_id, length, extra=()):
    assert 0 < length <= N
    maximum = N - length
    candidates = [("centre", maximum // 2), ("left", 0), ("right", maximum),
                  ("stable_random", _stable_offset(sample_id, maximum))]
    candidates.extend((f"documented_{value}", int(value)) for value in extra)
    seen, result = set(), []
    for label, value in candidates:
        assert 0 <= value <= maximum
        if value not in seen:
            result.append({"label": label, "start_sample": value}); seen.add(value)
    return result


def geometry(length, start):
    assert 0 < length <= N and 0 <= start <= N - length
    support = torch.zeros(1, N, dtype=torch.bool)
    support[0, start:start + length] = True
    q = _beats_token_support_fraction(support, fbank_frames=798, patch_size=16, frequency_patches=8)
    valid = _beats_token_valid_mask(support, fbank_frames=798, patch_size=16, frequency_patches=8)
    legacy = _beats_legacy_token_valid_mask(support, fbank_frames=798, tokens=392)
    # Independent receptive-field count: last real frame ends at sample127920.
    independent = []
    for patch in range(50):
        left = patch * 16 * 160
        right = min(left + 2800, 797 * 160 + 400)
        count = max(0, min(start + length, right) - max(start, left))
        independent.extend([count / 2800] * 8)
    np.testing.assert_allclose(q.numpy()[0], independent, atol=1e-14, rtol=0)
    assert torch.equal(valid, q > 0)
    return {"exact_valid_tokens": int(valid.sum()), "legacy_valid_tokens": int(legacy.sum()),
            "fully_padding_tokens": int((q == 0).sum()),
            "partially_valid_tokens": int(((q > 0) & (q < 1)).sum()),
            "fully_valid_tokens": int((q == 1).sum()), "independent_q_verified": True}


def datasets():
    feature = feature_config_from_yaml(c.ROOT / "configs/data/gate9a_beats.yaml")
    return {(domain, role): build_waveform_dataset(manifest_path=c.ROOT / "data/manifests/cross_domain_binary.csv",
            project_root=c.ROOT, feature_config=feature, role=role, domain=domain)
            for domain in c.b.ROLES for role in ("train_fit", "validation_select")}


def wave_for(case, data):
    if case["dataset"] == "synthetic":
        generator = torch.Generator(device="cpu").manual_seed(20260729)
        return torch.randn(case["length"], generator=generator).clamp(-1, 1)
    dataset = data[case["dataset"], case["protocol_role"]]
    rows = dataset.rows.loc[dataset.rows.sample_id == case["sample_id"]]
    assert len(rows) == 1
    wave = dataset._load_waveform(rows.iloc[0])
    if len(wave) > N:
        first = (len(wave) - N) // 2
        wave = wave[first:first + N]
    assert torch.isfinite(wave).all() and len(wave) > 0
    return wave


def wave_hash(wave):
    return hashlib.sha256(wave.contiguous().numpy().tobytes()).hexdigest()


def placed(wave, start):
    assert 0 <= start <= N - len(wave)
    waveform = torch.zeros(1, N); support = torch.zeros(1, N, dtype=torch.bool)
    waveform[0, start:start + len(wave)] = wave
    support[0, start:start + len(wave)] = True
    return waveform, support


def freeze():
    assert not FREEZE.exists() and not OUTPUT.exists()
    resource.verify()
    assert c.read_json(resource.OUTPUT / "independent_resource_audit.json")["status"] == "resource_QA_passed"
    torch.set_num_threads(12)
    manifest = pd.read_csv(c.ROOT / "data/manifests/cross_domain_binary.csv", dtype={"patient_id": str})
    overlap = c.b._patient_overlap_audit(manifest)
    mapping = pd.read_csv(MAPPING, dtype={"patient_id": str})
    zeros = mapping.loc[mapping.legacy_zero_valid == 1]
    assert len(zeros) == 6 and set(zeros.protocol_role) <= {"train_fit", "validation_select"}
    failed = c.read_json(FAILURE)["first_batch_with_zero_legacy_view"][0]
    selected, strata, empty = select_ids(manifest, zeros.sample_id.astype(str))
    assert failed["sample_id"] in selected
    paths = {Path(__file__).resolve(), c.ROOT / "tests/test_revision_stability_panel.py", SPEC, MAPPING, FAILURE,
             resource.FREEZE, resource.OUTPUT / "independent_resource_audit.json",
             c.ROOT / "scripts/audit_revision_mask_mapping.py"}
    data = datasets(); cases = []
    for sample_id in selected:
        row = manifest.loc[manifest.sample_id == sample_id].iloc[0]
        assert row.protocol_role in ("train_fit", "validation_select") and not bool(row.locked)
        case = {"sample_id": sample_id, "dataset": str(row.dataset), "patient_id": str(row.patient_id),
                "protocol_role": str(row.protocol_role), "target": int(row.binary_label_id),
                "event_duration_seconds": float(row.event_duration_seconds),
                "known_zero_token_event": sample_id in set(zeros.sample_id)}
        wave = wave_for(case, data)
        extra = [failed["first_valid_sample"]] if sample_id == failed["sample_id"] else []
        case.update(length=len(wave), waveform_sha256=wave_hash(wave), placements=offsets(sample_id, len(wave), extra))
        for item in case["placements"]: item.update(geometry(len(wave), item["start_sample"]))
        cases.append(case); paths.add(c.ROOT / str(row.wav_path))
    synthetic = {"sample_id": "synthetic::documented_2016", "dataset": "synthetic", "patient_id": None,
                 "protocol_role": "synthetic", "target": 0, "length": 2016, "known_zero_token_event": True,
                 "placements": offsets("synthetic::documented_2016", 2016, (114458, 125554))}
    synthetic["waveform_sha256"] = wave_hash(wave_for(synthetic, data))
    for item in synthetic["placements"]: item.update(geometry(2016, item["start_sample"]))
    cases.append(synthetic)
    for family, seeds in MODEL_MEMBERS.items():
        for seed in seeds: paths.add(resource.directory(family, seed) / "best.pt")
    c.write_json(FREEZE, {"status": "frozen_before_panel_model_diagnostics", "frozen_at": c.now(),
        "selection_strata": strata, "empty_strata": empty, "cases": cases, "models": MODEL_MEMBERS,
        "real_events": len(cases) - 1, "synthetic_events": 1,
        "event_placements": sum(len(case["placements"]) for case in cases),
        "known_zero_legacy_mapping_rows": zeros.to_dict("records"), "patient_overlap_audit": overlap,
        "no_calibration_or_heldout_panel_events": True, "no_training_or_model_selection": True,
        "cpu_threads": 12, "files": {str(path): sha256_file(path) for path in sorted(paths)}})
    print({"freeze_sha256": sha256_file(FREEZE), "real_events": len(cases) - 1,
           "placements": sum(len(case["placements"]) for case in cases), "empty_strata": len(empty)}, flush=True)


def verify():
    frozen = c.read_json(FREEZE)
    assert frozen["status"] == "frozen_before_panel_model_diagnostics"
    assert frozen["models"] == {key: list(value) for key, value in MODEL_MEMBERS.items()}
    for name, digest in frozen["files"].items(): assert sha256_file(Path(name)) == digest, name
    resource.verify()
    return frozen


def finite_stats(value):
    if isinstance(value, torch.Tensor):
        finite = torch.isfinite(value)
        return {"elements": value.numel(), "nonfinite": int((~finite).sum().detach().cpu())}
    if isinstance(value, (tuple, list)):
        parts = [finite_stats(item) for item in value]
        return {key: sum(part[key] for part in parts) for key in ("elements", "nonfinite")}
    return {"elements": 0, "nonfinite": 0}


def backward_diagnostic(model, wave, target, first, second, device):
    # Eval plus autograd: deterministic 12-layer coverage, no optimiser or state update.
    model.eval(); model.zero_grad(set_to_none=True)
    observed, handles = {}, []
    def record(key, value):
        observed.setdefault(key, []).append(finite_stats(value))
    handles.append(model.backbone.patch_embedding.register_forward_pre_hook(lambda _m, inp: record("frontend", inp)))
    handles.append(model.backbone.encoder.register_forward_hook(lambda _m, _inp, out: record("encoder", out)))
    handles.append(model.classifier.register_forward_pre_hook(lambda _m, inp: record("pooled", inp)))
    for index, layer in enumerate(model.backbone.encoder.layers):
        handles.append(layer.register_forward_hook(lambda _m, _inp, out, i=index: record(f"layer_{i}", out)))
        handles.append(layer.self_attn.register_forward_hook(lambda _m, _inp, out, i=index: record(f"attention_{i}", out)))
        handles.append(layer.self_attn.dropout_module.register_forward_hook(
            lambda _m, _inp, out, i=index: record(f"attention_probabilities_{i}", out)))
    result = {"first_start": first, "second_start": second, "observed": observed, "exception": None}
    try:
        x, mx = placed(wave, first); z, mz = placed(wave, second)
        a = model(x.to(device), mx.to(device)); b = model(z.to(device), mz.to(device))
        logits = torch.stack((a, b), 1)
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), torch.tensor([target, target], device=device))
        js = jensen_shannon_consistency(logits)
        loss = ce + .02 * js
        result["logits"] = finite_stats(logits)
        result["ce"] = finite_stats(ce); result["js"] = finite_stats(js); result["loss"] = finite_stats(loss)
        loss.backward()
    except Exception as error:
        result["exception"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        for handle in handles: handle.remove()
    gradients = {name: finite_stats(p.grad) for name, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    result["gradient_tensors_observed"] = len(gradients)
    result["nonfinite_gradients"] = {name: value for name, value in gradients.items() if value["nonfinite"]}
    result["gradient_elements_observed"] = sum(x["elements"] for x in gradients.values())
    required = {"frontend", "encoder", "pooled"} | {f"{name}_{index}" for index in range(12)
                for name in ("layer", "attention", "attention_probabilities")}
    result["all_required_hooks_observed_twice"] = set(observed) == required and all(len(observed[key]) == 2 for key in required)
    result["all_finite"] = (result["exception"] is None and result["all_required_hooks_observed_twice"] and
        len(gradients) > 0 and not result["nonfinite_gradients"] and
        all(item["nonfinite"] == 0 for values in observed.values() for item in values) and
        all(result.get(name, {"nonfinite": 1})["nonfinite"] == 0 for name in ("logits", "ce", "js", "loss")))
    model.zero_grad(set_to_none=True)
    return result


def placement_summary(rows):
    assert rows
    centre = next(row for row in rows if row["placement"] == "centre")
    finite = [row for row in rows if row["probability_1"] is not None]
    out = {"planned_positions": len(rows), "finite_positions": len(finite), "nonfinite_positions": len(rows) - len(finite),
           "probability_range": None, "probability_variance": None, "class_flip_rate_vs_centre": None}
    if len(finite) == len(rows):
        values = np.array([row["probability_1"] for row in rows])
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        out.update(probability_range=float(np.ptp(values)), probability_variance=float(np.var(values)),
                   class_flip_rate_vs_centre=float(np.mean((values >= .5) != (centre["probability_1"] >= .5))))
    return out


def run():
    frozen = verify()
    assert torch.backends.mps.is_available()
    torch.set_num_threads(12)
    OUTPUT.mkdir(exist_ok=False)
    state = {"status": "running", "started_at": c.now(), "pid": os.getpid(), "completed_models": 0,
             "freeze_sha256": sha256_file(FREEZE), "placement_records": [], "backward_records": [], "model_audits": []}
    c.write_json(OUTPUT / "status.json", {k: v for k, v in state.items() if not isinstance(v, list)}, replace=True)
    data = datasets()
    waves = {case["sample_id"]: wave_for(case, data) for case in frozen["cases"]}
    for case in frozen["cases"]:
        assert wave_hash(waves[case["sample_id"]]) == case["waveform_sha256"]
        assert len(waves[case["sample_id"]]) == case["length"]
    device = torch.device("mps")
    try:
        for family, seeds in MODEL_MEMBERS.items():
            for seed in seeds:
                model, identity = resource.load(family, seed, device)
                state["model_audits"].append({"family": family, **identity})
                if seed == 20260729:
                    control = torch.linspace(-.5, .5, N)
                    smoke = backward_diagnostic(model, control, 0, 0, 0, device)
                    state["backward_records"].append({"family": family, "seed": seed,
                        "sample_id": "synthetic::all_valid_control", "legacy_zero_in_pair": False, **smoke})
                    c.write_json(OUTPUT / f"{family}_smoke.json", smoke)
                    assert smoke["all_finite"], (family, "all-valid control failed")
                for case in frozen["cases"]:
                    wave = waves[case["sample_id"]]
                    for item in case["placements"]:
                        x, mask = placed(wave, item["start_sample"])
                        expected_zero = family == "legacy-mask" and item["legacy_valid_tokens"] == 0
                        record = {"family": family, "seed": seed, "sample_id": case["sample_id"],
                            "placement": item["label"], "start_sample": item["start_sample"],
                            "probability_1": None, "prediction": None, "exception": None,
                            "exact_valid_tokens": item["exact_valid_tokens"], "legacy_valid_tokens": item["legacy_valid_tokens"]}
                        try:
                            with torch.inference_mode():
                                logits = model(x.to(device), mask.to(device))
                                probs = torch.softmax(logits, -1).cpu()
                            record["logits_finite"] = bool(torch.isfinite(logits).all().cpu())
                            record["probabilities_finite"] = bool(torch.isfinite(probs).all())
                            if record["logits_finite"] and record["probabilities_finite"]:
                                p = float(probs[0, 1]); assert 0 <= p <= 1
                                record.update(probability_1=p, prediction=int(p >= .5))
                        except Exception as error:
                            record["exception"] = {"type": type(error).__name__, "message": str(error)}
                        state["placement_records"].append(record)
                        if record["probability_1"] is None and not expected_zero:
                            raise FloatingPointError(f"Unexpected finite failure: {family}/{seed}/{case['sample_id']}/{item['label']}")
                    if seed == 20260729:
                        centre = (N - len(wave)) // 2
                        stress = [N - len(wave)] + [item["start_sample"] for item in case["placements"] if item["label"].startswith("documented_")]
                        for start in dict.fromkeys(stress):
                            first_geom, second_geom = geometry(len(wave), centre), geometry(len(wave), start)
                            zero = first_geom["legacy_valid_tokens"] == 0 or second_geom["legacy_valid_tokens"] == 0
                            result = backward_diagnostic(model, wave, case["target"], centre, start, device)
                            state["backward_records"].append({"family": family, "seed": seed,
                                "sample_id": case["sample_id"], "legacy_zero_in_pair": zero, **result})
                            if not result["all_finite"] and not (family == "legacy-mask" and zero):
                                raise FloatingPointError(f"Unexpected backward failure: {family}/{case['sample_id']}/{start}")
                del model
                gc.collect(); torch.mps.empty_cache()
                state["completed_models"] += 1
                c.write_json(OUTPUT / f"completed_model_{state['completed_models']}.json", state)
                c.write_json(OUTPUT / "status.json", {k: v for k, v in state.items() if not isinstance(v, list)}, replace=True)
                print({"family": family, "seed": seed, "completed_models": state["completed_models"],
                       "placement_records": len(state["placement_records"]), "backward_records": len(state["backward_records"])}, flush=True)
        summaries = []
        for family, seeds in MODEL_MEMBERS.items():
            for seed in seeds:
                for case in frozen["cases"]:
                    selected = [r for r in state["placement_records"] if (r["family"], r["seed"], r["sample_id"]) == (family, seed, case["sample_id"])]
                    summaries.append({"family": family, "seed": seed, "sample_id": case["sample_id"], **placement_summary(selected)})
        verify()
        state.update(status="complete_pending_independent_QA", completed_at=c.now(), placement_summaries=summaries,
                     no_parameter_updates=True, no_calibration_or_heldout_events=True)
        c.write_json(OUTPUT / "stability_panel.json", state)
        c.write_json(OUTPUT / "status.json", {"status": state["status"], "completed_at": state["completed_at"],
            "completed_models": 5, "result_sha256": sha256_file(OUTPUT / "stability_panel.json")}, replace=True)
    except Exception as error:
        state.update(status="unexpected_failure_no_automatic_retry", ended_at=c.now(), error_type=type(error).__name__, error=str(error))
        c.write_json(OUTPUT / "unexpected_failure.json", state)
        c.write_json(OUTPUT / "status.json", {k: v for k, v in state.items() if not isinstance(v, list)}, replace=True)
        raise


def audit():
    frozen = verify()
    result = c.read_json(OUTPUT / "stability_panel.json")
    assert result["status"] == "complete_pending_independent_QA" and result["completed_models"] == 5
    assert sha256_file(OUTPUT / "stability_panel.json") == c.read_json(OUTPUT / "status.json")["result_sha256"]
    expected = {(family, seed, case["sample_id"], item["start_sample"])
                for family, seeds in MODEL_MEMBERS.items() for seed in seeds
                for case in frozen["cases"] for item in case["placements"]}
    actual = [(r["family"], r["seed"], r["sample_id"], r["start_sample"]) for r in result["placement_records"]]
    assert set(actual) == expected and len(actual) == len(expected)
    for case in frozen["cases"]:
        for item in case["placements"]:
            recalculated = geometry(case["length"], item["start_sample"])
            assert all(item[key] == value for key, value in recalculated.items())
            assert item["fully_padding_tokens"] + item["partially_valid_tokens"] + item["fully_valid_tokens"] == 400
    for row in result["placement_records"]:
        p = row["probability_1"]
        if p is None: assert row["family"] == "legacy-mask" and row["legacy_valid_tokens"] == 0
        else: assert math.isfinite(p) and 0 <= p <= 1 and row["prediction"] == int(p >= .5)
    for row in result["placement_summaries"]:
        group = [r for r in result["placement_records"] if all(r[k] == row[k] for k in ("family", "seed", "sample_id"))]
        assert row["planned_positions"] == len(group)
        probabilities = [r["probability_1"] for r in group]
        finite = sum(p is not None for p in probabilities)
        assert row["finite_positions"] == finite and row["nonfinite_positions"] == len(group) - finite
        if finite != len(group):
            assert row["probability_range"] is None and row["probability_variance"] is None and row["class_flip_rate_vs_centre"] is None
        else:
            mean = sum(probabilities) / len(probabilities)
            variance = sum((p - mean) ** 2 for p in probabilities) / len(probabilities)
            centre = next(r["probability_1"] for r in group if r["placement"] == "centre")
            flips = sum((p >= .5) != (centre >= .5) for p in probabilities) / len(probabilities)
            np.testing.assert_allclose([max(probabilities) - min(probabilities), variance, flips],
                [row["probability_range"], row["probability_variance"], row["class_flip_rate_vs_centre"]], atol=1e-14, rtol=0)
    pair_count = 1 + sum(1 + sum(item["label"].startswith("documented_") for item in case["placements"]) for case in frozen["cases"])
    assert len(result["backward_records"]) == 2 * pair_count
    expected_pairs = {(family, "synthetic::all_valid_control", 0, 0) for family in MODEL_MEMBERS}
    for family in MODEL_MEMBERS:
        for case in frozen["cases"]:
            centre = (N - case["length"]) // 2
            stress = [N - case["length"]] + [item["start_sample"] for item in case["placements"] if item["label"].startswith("documented_")]
            expected_pairs.update((family, case["sample_id"], centre, start) for start in stress)
    actual_pairs = [(row["family"], row["sample_id"], row["first_start"], row["second_start"]) for row in result["backward_records"]]
    assert set(actual_pairs) == expected_pairs and len(actual_pairs) == len(expected_pairs)
    for row in result["backward_records"]:
        assert row["seed"] == 20260729
        if not row["all_finite"]: assert row["family"] == "legacy-mask" and row["legacy_zero_in_pair"]
        else:
            assert row["all_required_hooks_observed_twice"] and row["gradient_tensors_observed"] > 0
            assert not row["nonfinite_gradients"]
            assert all(item["nonfinite"] == 0 for values in row["observed"].values() for item in values)
            assert all(row[name]["nonfinite"] == 0 for name in ("logits", "ce", "js", "loss"))
    c.write_json(OUTPUT / "independent_stability_audit.json", {"status": "stability_placement_QA_passed", "audited_at": c.now(),
        "event_placements_per_model": frozen["event_placements"], "models": 5,
        "prediction_records_verified": len(actual), "backward_records_verified": len(result["backward_records"]),
        "placement_summary_rows_verified": len(result["placement_summaries"]),
        "expected_legacy_nonfinite_prediction_records": sum(r["probability_1"] is None for r in result["placement_records"]),
        "expected_legacy_nonfinite_backward_records": sum(not r["all_finite"] for r in result["backward_records"]),
        "exact_all_finite": all(r["probability_1"] is not None for r in result["placement_records"] if r["family"] == "exact"),
        "source_and_checkpoint_hashes_verified": True, "no_patient_role_changes": True,
        "result_sha256": sha256_file(OUTPUT / "stability_panel.json"), "freeze_sha256": sha256_file(FREEZE)})
    print(c.read_json(OUTPUT / "independent_stability_audit.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "audit"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "audit": audit}[args.stage]()
