#!/usr/bin/env python3
"""Profile frozen full-FT and LoRA single/ensemble inference efficiency on MPS."""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from respiratory_sound.gate11a import sha256_file
from respiratory_sound.post_gate11a import SEEDS, load_frozen_model, run_map
from respiratory_sound.runtime import select_device

BATCH_SIZES = (1, 8)
WARMUP_ITERATIONS = 5
TIMED_ITERATIONS = 20
SAMPLES = 8 * 16_000


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _memory(device: torch.device) -> dict[str, int]:
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss = raw_rss if sys.platform == "darwin" else raw_rss * 1024
    return {
        "process_peak_rss_bytes": rss,
        "mps_current_allocated_bytes": (
            int(torch.mps.current_allocated_memory()) if device.type == "mps" else 0
        ),
        "mps_driver_allocated_bytes": (
            int(torch.mps.driver_allocated_memory()) if device.type == "mps" else 0
        ),
    }


@torch.inference_mode()
def _profile(
    models: list[torch.nn.Module],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20_260_729 + batch_size)
    waveforms = (0.01 * torch.randn(batch_size, SAMPLES, generator=generator)).to(device)
    masks = torch.ones((batch_size, SAMPLES), dtype=torch.bool, device=device)

    def forward() -> torch.Tensor:
        return torch.stack(
            [torch.softmax(model(waveforms, masks), dim=-1) for model in models]
        ).mean(dim=0)

    for _ in range(WARMUP_ITERATIONS):
        output = forward()
        if not bool(torch.isfinite(output).all().detach().cpu()):
            raise FloatingPointError("Efficiency warmup produced non-finite output")
    _synchronize(device)
    durations = []
    peak = _memory(device)
    for _ in range(TIMED_ITERATIONS):
        _synchronize(device)
        started = time.perf_counter()
        output = forward()
        _synchronize(device)
        durations.append(time.perf_counter() - started)
        current = _memory(device)
        peak = {key: max(peak[key], current[key]) for key in peak}
    values = np.asarray(durations, dtype=np.float64)
    median_batch = float(np.median(values))
    return {
        "batch_size": batch_size,
        "warmup_iterations": WARMUP_ITERATIONS,
        "timed_iterations": TIMED_ITERATIONS,
        "median_batch_latency_seconds": median_batch,
        "p95_batch_latency_seconds": float(np.quantile(values, 0.95)),
        "median_per_event_latency_seconds": median_batch / batch_size,
        "events_per_second_at_median": batch_size / median_batch,
        "peak_memory": peak,
    }


def _training_evidence(root: Path, family: str) -> list[dict[str, Any]]:
    results = []
    for seed, run in run_map(family).items():
        summary = json.loads((root / run / "summary.json").read_text(encoding="utf-8"))
        results.append(
            {
                "seed": seed,
                "best_epoch": int(summary["best_epoch"]),
                "epochs_completed": int(summary["epochs_completed"]),
                "training_parameters": int(summary["training_parameters"]),
                "total_parameters": int(summary["total_parameters"]),
                "training_seconds_total": summary.get("training_seconds_total"),
                "memory": summary.get("memory"),
            }
        )
    return results


def main() -> None:
    root = Path.cwd().resolve()
    output = root / "artifacts/post_gate11a/efficiency_final.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Efficiency result already exists: {output}")
    development_freeze = root / "artifacts/post_gate11a_development_freeze.json"
    calibration = (
        root
        / "artifacts/post_gate11a/calibration_analysis/post_gate11a_calibration_final.json"
    )
    if json.loads(development_freeze.read_text())["status"] != "sealed_before_calibration":
        raise ValueError("Invalid development freeze")
    if json.loads(calibration.read_text())["locked_tests_accessed"]:
        raise ValueError("Calibration artifact unexpectedly accessed locked data")
    device = select_device("mps")
    if device.type != "mps":
        raise SystemExit("Formal efficiency profiling requires MPS")

    families: dict[str, Any] = {}
    for family in ("fullft", "lora"):
        models = [
            load_frozen_model(root, family=family, seed=seed, device=device)
            for seed in SEEDS
        ]
        single = models[:1]
        total_parameters = [sum(p.numel() for p in model.parameters()) for model in models]
        trainable_parameters = [
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            for model in models
        ]
        checkpoint_bytes = [
            (root / run_map(family)[seed] / "best.pt").stat().st_size
            for seed in SEEDS
        ]
        families[family] = {
            "single": {
                "logical_total_parameters": total_parameters[0],
                "training_parameters": trainable_parameters[0],
                "checkpoint_bytes": checkpoint_bytes[0],
                "profiles": {
                    str(batch): _profile(single, device=device, batch_size=batch)
                    for batch in BATCH_SIZES
                },
            },
            "three_seed_ensemble": {
                "logical_total_parameters": int(sum(total_parameters)),
                "training_parameters_across_members": int(sum(trainable_parameters)),
                "checkpoint_bytes": int(sum(checkpoint_bytes)),
                "weights": [1 / 3, 1 / 3, 1 / 3],
                "profiles": {
                    str(batch): _profile(models, device=device, batch_size=batch)
                    for batch in BATCH_SIZES
                },
            },
            "training_evidence": _training_evidence(root, family),
        }
        del models
        if device.type == "mps":
            torch.mps.empty_cache()

    payload = {
        "stage": "post_gate11a_efficiency",
        "device": str(device),
        "input": {
            "sample_rate": 16_000,
            "seconds": 8.0,
            "samples": SAMPLES,
            "mask": "all valid",
        },
        "families": families,
        "development_freeze_sha256": sha256_file(development_freeze),
        "calibration_artifact_sha256": sha256_file(calibration),
        "calibration_accessed": True,
        "locked_tests_accessed": False,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
