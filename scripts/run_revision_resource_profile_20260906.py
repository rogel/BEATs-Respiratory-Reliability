#!/usr/bin/env python3
"""Frozen inference timing plus source-backed training telemetry; no model fitting."""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

import run_revision_analysis_c as c
from predict_revision_ablation import _load_model, _run_dir
from respiratory_sound.gate11a import sha256_file
from respiratory_sound.post_gate11a import load_frozen_model, run_map

FAMILIES = ("exact", "practical-lora", "no-js", "matched-lora", "legacy-mask")
SEEDS = tuple(c.ALLOWED_SEEDS)
MEMBERS = {family: (SEEDS[0], SEEDS[2]) if family == "legacy-mask" else SEEDS for family in FAMILIES}
FREEZE = c.BASE / "resource_profile_freeze_2026_09_06.json"
OUTPUT = c.BASE / "resource_profile_2026_09_06"
SPEC = c.REPORTS / "23_RESOURCE_PROFILE_EXECUTION_SPEC_2026-09-06.md"
UPSTREAM = c.ROOT / "checkpoints/pretrained/beats_as2m_cpt2/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
WARMUPS, ITERATIONS, BATCHES = 5, 20, (1, 8)


def directory(family, seed):
    if family in ("exact", "practical-lora"):
        return c.ROOT / run_map("fullft" if family == "exact" else "lora")[seed]
    return _run_dir(c.ROOT, family, seed)


def freeze():
    assert not FREEZE.exists() and not OUTPUT.exists()
    c.verify()
    qa = c.OUTPUT / "independent_result_audit.json"
    assert c.read_json(qa)["status"] == "independent_analysis_c_QA_passed"
    paths = {Path(__file__).resolve(), c.ROOT / "tests/test_revision_resource_profile.py", SPEC,
             c.FREEZE, qa, UPSTREAM, c.ROOT / "scripts/profile_revision_models.py",
             c.ROOT / "src/respiratory_sound/post_gate11a.py",
             c.BASE / "failures/legacy_mask_seed20260730_nonfinite.json"}
    for family in FAMILIES:
        for seed in SEEDS:
            run = directory(family, seed)
            for name in ("configuration.json", "history.csv", "summary.json", "best.pt"):
                path = run / name
                if name == "summary.json" and family == "legacy-mask" and seed == SEEDS[1]:
                    assert not path.exists()
                else:
                    assert path.exists(), path
                    paths.add(path)
            job = c.BASE / f"jobs_2026_09_05/{family.replace('-', '_')}_seed{seed}/status.json"
            if job.exists(): paths.add(job)
    c.write_json(FREEZE, {"status": "frozen_before_resource_timing", "frozen_at": c.now(),
        "members": MEMBERS, "warmups": WARMUPS, "iterations": ITERATIONS, "batch_sizes": BATCHES,
        "cpu_threads": 12, "synthetic_seed": SEEDS[0], "study_outcome_inference": False,
        "files": {str(path): sha256_file(path) for path in sorted(paths)}})
    print({"freeze_sha256": sha256_file(FREEZE), "files": len(paths)}, flush=True)


def verify():
    frozen = c.read_json(FREEZE)
    assert frozen["status"] == "frozen_before_resource_timing"
    assert frozen["members"] == {k: list(v) for k, v in MEMBERS.items()}
    assert frozen["iterations"] == 20 and frozen["warmups"] == 5
    assert frozen["batch_sizes"] == [1, 8] and frozen["cpu_threads"] == 12
    for name, digest in frozen["files"].items():
        assert sha256_file(Path(name)) == digest, name
    c.verify()
    return frozen


def summarize_times(times, batch_size):
    values = np.asarray(times, dtype=float)
    assert values.shape == (ITERATIONS,) and np.isfinite(values).all() and (values > 0).all()
    assert batch_size in BATCHES
    median = float(np.median(values))
    return {"batch_latency_median_seconds": median,
            "batch_latency_p95_seconds": float(np.quantile(values, .95)),
            "per_event_latency_median_seconds": median / batch_size,
            "throughput_events_per_second": batch_size / median}


def synthetic_input(batch_size, device):
    generator = torch.Generator(device="cpu").manual_seed(SEEDS[0] + batch_size)
    waveform = torch.randn(batch_size, 128000, generator=generator).clamp_(-1, 1)
    digest = hashlib.sha256(waveform.numpy().tobytes()).hexdigest()
    return waveform.to(device), torch.ones_like(waveform, dtype=torch.bool, device=device), digest


def sync(device):
    if device.type == "mps": torch.mps.synchronize()


@torch.inference_mode()
def time_models(models, batch_size, device):
    waveform, support, digest = synthetic_input(batch_size, device)
    process = psutil.Process()
    rss_start = process.memory_info().rss
    observed_rss, current, driver = [], [], []
    def forward():
        return torch.stack([torch.softmax(model(waveform, support), -1) for model in models]).mean(0)
    def check(output):
        assert output.shape == (batch_size, 2)
        if not bool(torch.isfinite(output).all().cpu()):
            raise FloatingPointError("Nonfinite synthetic profiling probability")
        np.testing.assert_allclose(output.sum(1).cpu().numpy(), 1., atol=2e-7, rtol=0)
    started = c.now()
    for _ in range(WARMUPS):
        output = forward(); sync(device); check(output)
    timings = []
    for _ in range(ITERATIONS):
        sync(device)
        begin = time.perf_counter()
        output = forward()
        sync(device)
        timings.append(time.perf_counter() - begin)
        check(output)
        observed_rss.append(process.memory_info().rss)
        if device.type == "mps":
            current.append(torch.mps.current_allocated_memory())
            driver.append(torch.mps.driver_allocated_memory())
    return {"started_at": started, "input_sha256": digest, "timings_seconds": timings,
            **summarize_times(timings, batch_size), "cpu_threads": torch.get_num_threads(),
            "finite_passes": WARMUPS + ITERATIONS,
            "rss_before_bytes": rss_start, "maximum_sampled_rss_bytes": max(observed_rss),
            "maximum_sampled_mps_current_bytes": max(current) if current else None,
            "maximum_sampled_mps_driver_bytes": max(driver) if driver else None}


def load(family, seed, device):
    if family in ("exact", "practical-lora"):
        model = load_frozen_model(c.ROOT, family="fullft" if family == "exact" else "lora", seed=seed, device=device)
    else:
        model, _ = _load_model(c.ROOT, family=family, seed=seed, device=device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    lora = family in ("practical-lora", "matched-lora")
    assert n == (99842 if lora else 90313330)
    assert total == (90411634 if lora else 90313330)
    assert all(bool(torch.isfinite(p).all().cpu()) for p in model.parameters())
    path = directory(family, seed) / "best.pt"
    return model, {"seed": seed, "checkpoint_sha256": sha256_file(path), "checkpoint_bytes": path.stat().st_size,
                   "trainable_parameters": n, "total_parameters": total,
                   "storage_scope": "adapter_and_head_only_requires_shared_backbone" if lora else "full_model_weights"}


def training_resources():
    records = []
    for family in FAMILIES:
        for seed in SEEDS:
            run = directory(family, seed)
            history = pd.read_csv(run / "history.csv")
            assert np.isfinite(history.select_dtypes(include=np.number).to_numpy()).all()
            assert np.array_equal(history.epoch, np.arange(1, len(history) + 1))
            summary = c.read_json(run / "summary.json") if (run / "summary.json").exists() else None
            complete = summary is not None
            if complete: assert summary["epochs_completed"] == len(history)
            seconds = float(history.epoch_seconds.sum()) if "epoch_seconds" in history else None
            if summary and "training_seconds_total" in summary:
                np.testing.assert_allclose(seconds, summary["training_seconds_total"], rtol=0, atol=1e-7)
            wall = None if not summary else summary.get("wall_seconds_this_invocation", summary.get("elapsed_seconds_this_invocation"))
            job = c.BASE / f"jobs_2026_09_05/{family.replace('-', '_')}_seed{seed}/status.json"
            external_wall = None
            if job.exists():
                state = c.read_json(job)
                if state.get("ended_at"):
                    external_wall = (datetime.fromisoformat(state["ended_at"]) - datetime.fromisoformat(state["started_at"])).total_seconds()
            lora = family in ("practical-lora", "matched-lora")
            records.append({"family": family, "seed": seed, "completed_formal_run": complete,
                "epochs_recorded": len(history), "selected_epoch": summary["best_epoch"] if summary else None,
                "recorded_epoch_seconds_total": seconds, "training_loop_wall_seconds_this_invocation": wall,
                "supervisor_wall_seconds": external_wall, "memory": summary.get("memory") if summary else None,
                "trainable_parameters": 99842 if lora else 90313330,
                "total_parameters": 90411634 if lora else 90313330,
                "checkpoint_bytes": (run / "best.pt").stat().st_size,
                "checkpoint_sha256": sha256_file(run / "best.pt"),
                "history_sha256": sha256_file(run / "history.csv"),
                "summary_sha256": sha256_file(run / "summary.json") if summary else None,
                "telemetry_note": "unavailable fields remain null; no imputation" if complete else
                    "incomplete Legacy730: five completed epochs only; failed epoch/replay cost not fully recorded; checkpoint diagnostic only"})
    return records


def run():
    verify()
    assert torch.backends.mps.is_available()
    torch.set_num_threads(12)
    OUTPUT.mkdir(exist_ok=False)
    progress = {"status": "running", "started_at": c.now(), "pid": os.getpid(), "completed_settings": 0,
                "freeze_sha256": sha256_file(FREEZE), "inference": []}
    c.write_json(OUTPUT / "status.json", progress, replace=True)
    device = torch.device("mps")
    try:
        for family in FAMILIES:
            for label, seeds in (("single_seed20260729", (SEEDS[0],)),
                                 ("ensemble_2_completers" if family == "legacy-mask" else "ensemble_3", MEMBERS[family])):
                models, identities = [], []
                for seed in seeds:
                    model, identity = load(family, seed, device)
                    models.append(model); identities.append(identity)
                del model
                for batch in BATCHES:
                    item = {"family": family, "model_set": label, "seeds": seeds, "batch_size": batch,
                            "models": identities, **time_models(models, batch, device)}
                    progress["inference"].append(item)
                    progress["completed_settings"] += 1
                    c.write_json(OUTPUT / "status.json", progress, replace=True)
                    print({"family": family, "model_set": label, "batch_size": batch,
                           "median_seconds": item["batch_latency_median_seconds"],
                           "completed_settings": progress["completed_settings"]}, flush=True)
                del models
                gc.collect(); torch.mps.empty_cache()
        verify()
        resources = training_resources()
        c.write_json(OUTPUT / "resource_profile.json", {"status": "complete_pending_independent_QA", "completed_at": c.now(),
            "hardware": "Apple M4 Max, 128 GB unified memory", "device": "mps", "cpu_threads": 12,
            "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                         "platform": platform.platform()},
            "warmups": WARMUPS, "timed_iterations": ITERATIONS, "disk_io_in_timed_region": False,
            "frontend_and_cpu_mps_transfer_in_timed_region": True, "inference": progress["inference"],
            "training": resources, "shared_upstream_checkpoint_bytes": UPSTREAM.stat().st_size,
            "shared_upstream_sha256": sha256_file(UPSTREAM), "freeze_sha256": sha256_file(FREEZE),
            "scientific_scope": "hardware-specific descriptive timing; Legacy ensemble has two completers; no causal cost claim"})
        progress.update(status="completed_pending_independent_QA", ended_at=c.now(),
                        result_sha256=sha256_file(OUTPUT / "resource_profile.json"))
        c.write_json(OUTPUT / "status.json", progress, replace=True)
    except Exception as error:
        progress.update(status="failed_no_automatic_retry", ended_at=c.now(), exception_type=type(error).__name__, exception=str(error))
        c.write_json(OUTPUT / "status.json", progress, replace=True)
        raise


def audit():
    verify()
    result = c.read_json(OUTPUT / "resource_profile.json")
    status = c.read_json(OUTPUT / "status.json")
    assert sha256_file(OUTPUT / "resource_profile.json") == status["result_sha256"]
    assert result["freeze_sha256"] == sha256_file(FREEZE)
    assert len(result["inference"]) == 20 and len(result["training"]) == 15
    identities = set()
    for row in result["inference"]:
        identities.add((row["family"], row["model_set"], row["batch_size"]))
        values = sorted(row["timings_seconds"])
        assert len(values) == 20 and all(math.isfinite(x) and x > 0 for x in values)
        median = (values[9] + values[10]) / 2
        p95 = values[18] + .05 * (values[19] - values[18])
        batch = row["batch_size"]
        np.testing.assert_allclose([median, p95, median / batch, batch / median],
            [row[k] for k in ("batch_latency_median_seconds", "batch_latency_p95_seconds",
                              "per_event_latency_median_seconds", "throughput_events_per_second")], atol=1e-10, rtol=1e-12)
        assert row["finite_passes"] == 25 and row["cpu_threads"] == 12
        assert len(row["models"]) == len(row["seeds"])
        assert row["seeds"] == ([SEEDS[0]] if row["model_set"].startswith("single") else list(MEMBERS[row["family"]]))
        for model in row["models"]:
            checkpoint = directory(row["family"], model["seed"]) / "best.pt"
            assert sha256_file(checkpoint) == model["checkpoint_sha256"]
            assert checkpoint.stat().st_size == model["checkpoint_bytes"]
    assert len(identities) == 20
    for row in result["training"]:
        history = pd.read_csv(directory(row["family"], row["seed"]) / "history.csv")
        assert row["epochs_recorded"] == len(history)
        if "epoch_seconds" in history:
            np.testing.assert_allclose(sum(history.epoch_seconds.tolist()), row["recorded_epoch_seconds_total"], atol=1e-7, rtol=0)
        else: assert row["recorded_epoch_seconds_total"] is None
    assert sum(not row["completed_formal_run"] for row in result["training"]) == 1
    c.write_json(OUTPUT / "independent_resource_audit.json", {"status": "resource_QA_passed", "audited_at": c.now(),
        "settings": 20, "timed_iterations_verified": 400, "timing_cells_recomputed": 80, "training_records": 15,
        "no_patient_data_in_timing": True, "all_inputs_and_outputs_hashed": True,
        "result_sha256": sha256_file(OUTPUT / "resource_profile.json"), "freeze_sha256": sha256_file(FREEZE)})
    print(c.read_json(OUTPUT / "independent_resource_audit.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "run", "audit"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "audit": audit}[args.stage]()
