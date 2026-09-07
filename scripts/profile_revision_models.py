#!/usr/bin/env python3
"""Profile frozen revision models on fixed 8-second synthetic inputs."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from predict_revision_ablation import _load_model as load_revision_model

from respiratory_sound.gate11a import ALLOWED_SEEDS, sha256_file
from respiratory_sound.post_gate11a import load_frozen_model
from respiratory_sound.runtime import select_device

FAMILIES = ("exact", "legacy-mask", "no-js", "matched-lora")
BATCH_SIZES = (1, 8)
WARMUPS = 5
ITERATIONS = 20
PROFILE_SEED = 20_260_729


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _load_family_model(
    root: Path,
    family: str,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if family == "exact":
        model = load_frozen_model(root, family="fullft", seed=seed, device=device)
        run = root / f"runs/gate11a_exactmask_fullft_seed{seed}"
        return model, {
            "run": str(run.relative_to(root)),
            "checkpoint_sha256": sha256_file(run / "best.pt"),
        }
    return load_revision_model(
        root,
        family=family,
        seed=seed,
        device=device,
    )


@torch.inference_mode()
def _time_models(
    models: list[torch.nn.Module],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(PROFILE_SEED + batch_size)
    waveforms = torch.randn(
        (batch_size, 128_000), generator=generator, dtype=torch.float32
    ).clamp_(-1.0, 1.0).to(device)
    support = torch.ones((batch_size, 128_000), dtype=torch.bool, device=device)

    def forward() -> torch.Tensor:
        probabilities = [torch.softmax(model(waveforms, support), -1) for model in models]
        return torch.stack(probabilities).mean(0)

    for _ in range(WARMUPS):
        output = forward()
        if not bool(torch.isfinite(output).all().cpu()):
            raise FloatingPointError("Profiling warm-up became non-finite")
        _sync(device)
    timings = []
    for _ in range(ITERATIONS):
        _sync(device)
        started = time.perf_counter()
        output = forward()
        _sync(device)
        timings.append(time.perf_counter() - started)
        if not bool(torch.isfinite(output).all().cpu()):
            raise FloatingPointError("Profiling pass became non-finite")
    values = np.asarray(timings, dtype=float)
    median = float(np.median(values))
    return {
        "batch_latency_median_seconds": median,
        "batch_latency_p95_seconds": float(np.quantile(values, 0.95)),
        "per_event_latency_median_seconds": median / batch_size,
        "throughput_events_per_second": batch_size / median,
    }


def main() -> None:
    root = Path.cwd().resolve()
    device = select_device("mps")
    if device.type != "mps":
        raise SystemExit("Formal revision profiling requires MPS")
    results: dict[str, Any] = {}
    for family in FAMILIES:
        results[family] = {"single_seed20260729": {}, "three_model_ensemble": {}}
        model, audit = _load_family_model(root, family, ALLOWED_SEEDS[0], device)
        results[family]["single_seed20260729"]["model_audit"] = audit
        for batch_size in BATCH_SIZES:
            results[family]["single_seed20260729"][f"batch_{batch_size}"] = (
                _time_models([model], batch_size=batch_size, device=device)
            )
        del model
        torch.mps.empty_cache()

        models = []
        audits = []
        for seed in ALLOWED_SEEDS:
            model, audit = _load_family_model(root, family, seed, device)
            models.append(model)
            audits.append(audit)
        results[family]["three_model_ensemble"]["model_audits"] = audits
        for batch_size in BATCH_SIZES:
            results[family]["three_model_ensemble"][f"batch_{batch_size}"] = (
                _time_models(models, batch_size=batch_size, device=device)
            )
        del models
        torch.mps.empty_cache()

    training_resources = {}
    for family in FAMILIES[1:]:
        summaries = []
        for seed in ALLOWED_SEEDS:
            path = (
                root
                / "runs/revision_2026_09_03"
                / f"revision_{family.replace('-', '_')}_seed{seed}"
                / "summary.json"
            )
            summary = json.loads(path.read_text(encoding="utf-8"))
            summaries.append(
                {
                    "seed": seed,
                    "summary_sha256": sha256_file(path),
                    "epochs_completed": summary["epochs_completed"],
                    "training_seconds_total": summary["training_seconds_total"],
                    "wall_seconds_this_invocation": summary["wall_seconds_this_invocation"],
                    "memory": summary["memory"],
                    "trainable_parameters": summary["trainable_parameters"],
                    "total_parameters": summary["total_parameters"],
                    "best_checkpoint_bytes": summary["best_checkpoint_bytes"],
                }
            )
        training_resources[family] = summaries

    payload = {
        "schema_version": 1,
        "stage": "BEATS_revision_resource_profile",
        "hardware": "Apple M4 Max, 128 GB unified memory",
        "device": str(device),
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
        },
        "protocol": {
            "input": "fixed-seed synthetic 8-second 16-kHz all-valid waveform",
            "batch_sizes": BATCH_SIZES,
            "warmups": WARMUPS,
            "timed_iterations": ITERATIONS,
            "device_synchronization": "before and after every timed forward pass",
            "disk_io_in_timed_region": False,
        },
        "inference": results,
        "training": training_resources,
        "scope_note": "Measurements describe this hardware/software setup only.",
        "finite_value_audit_passed": True,
    }
    output_root = root / "artifacts/revision_2026_09_03/resource_profile"
    output_root.mkdir(parents=True, exist_ok=False)
    _atomic_json(payload, output_root / "revision_resource_profile.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
