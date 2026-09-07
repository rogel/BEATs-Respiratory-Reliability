#!/usr/bin/env python3
"""Generate frozen calibration or one-time locked predictions after Gate 11A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from compute_feature_stats import feature_config_from_yaml

from respiratory_sound.gate11a import sha256_file
from respiratory_sound.post_gate11a import (
    SEEDS,
    build_waveform_dataset,
    infer_probabilities,
    load_frozen_model,
    probability_ensemble,
)
from respiratory_sound.runtime import select_device

ROLE_BY_STAGE = {
    "smoke": {"icbhi2017": ("validation_select",)},
    "calibration": {
        "icbhi2017": ("calibration",),
        "sprsound2022": ("calibration",),
    },
    "locked": {
        "icbhi2017": ("locked_test",),
        "sprsound2022": ("locked_inter_test", "locked_intra_test"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", choices=tuple(ROLE_BY_STAGE), required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-locked-access", action="store_true")
    return parser.parse_args()


def _load_freeze(root: Path, stage: str) -> tuple[dict, Path, str] | None:
    if stage == "smoke":
        return None
    name = (
        "post_gate11a_development_freeze.json"
        if stage == "calibration"
        else "post_gate11a_locked_freeze.json"
    )
    path = root / "artifacts" / name
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = (
        "sealed_before_calibration"
        if stage == "calibration"
        else "sealed_before_locked_access"
    )
    if payload.get("status") != expected:
        raise ValueError(f"Invalid {stage} freeze status")
    return payload, path, sha256_file(path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_root = (root / args.output_root).resolve()
    freeze = _load_freeze(root, args.stage)
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")
    if args.batch_size != 8 and args.stage != "smoke":
        raise ValueError("Formal post-Gate-11A inference is frozen to batch size 8")
    families = ("fullft",) if args.stage in {"smoke", "calibration"} else ("fullft", "lora")
    seeds = (SEEDS[0],) if args.stage == "smoke" else SEEDS

    access_path = root / "artifacts" / "post_gate11a_locked_access.json"
    access: dict | None = None
    if args.stage == "locked":
        assert freeze is not None
        _, freeze_path, freeze_sha = freeze
        if access_path.exists():
            access = json.loads(access_path.read_text(encoding="utf-8"))
            if not args.resume_locked_access or access.get("status") != "started":
                raise RuntimeError("Locked performance has already been accessed")
            if access.get("freeze_sha256") != freeze_sha:
                raise RuntimeError("Locked access cannot resume under a changed freeze")
        else:
            access = {
                "status": "started",
                "freeze": str(freeze_path.relative_to(root)),
                "freeze_sha256": freeze_sha,
                "completed_prediction_files": {},
                "rule": "resume may reuse completed files but may not delete or regenerate them",
            }
            _atomic_json(access, access_path)

    output_root.mkdir(parents=True, exist_ok=args.stage == "smoke")
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    feature_config = feature_config_from_yaml(root / "configs/data/gate9a_beats.yaml")
    datasets = {
        (domain, role): build_waveform_dataset(
            manifest_path=manifest_path,
            project_root=root,
            feature_config=feature_config,
            role=role,
            domain=domain,
        )
        for domain, roles in ROLE_BY_STAGE[args.stage].items()
        for role in roles
    }
    generated: dict[str, str] = {}
    for family in families:
        for seed in seeds:
            model = load_frozen_model(root, family=family, seed=seed, device=device)
            for (domain, role), dataset in datasets.items():
                name = f"{family}_seed{seed}_{domain}_{role}.csv"
                path = output_root / name
                if args.stage == "locked" and path.exists():
                    assert access is not None
                    recorded = access["completed_prediction_files"].get(name)
                    if recorded != sha256_file(path):
                        raise RuntimeError(f"Unverified locked resume file: {name}")
                else:
                    frame = infer_probabilities(
                        model,
                        dataset,
                        device=device,
                        batch_size=args.batch_size,
                    )
                    frame.to_csv(path, index=False)
                    if args.stage == "locked":
                        assert access is not None
                        access["completed_prediction_files"][name] = sha256_file(path)
                        _atomic_json(access, access_path)
                generated[name] = sha256_file(path)
            del model
            if device.type == "mps":
                torch.mps.empty_cache()

    for family in families:
        for domain, roles in ROLE_BY_STAGE[args.stage].items():
            for role in roles:
                frames = [
                    pd.read_csv(
                        output_root / f"{family}_seed{seed}_{domain}_{role}.csv",
                        dtype={"patient_id": str},
                    )
                    for seed in seeds
                ]
                if len(frames) == 3:
                    ensemble = probability_ensemble(frames)
                    name = f"{family}_ensemble_{domain}_{role}.csv"
                    path = output_root / name
                    ensemble.to_csv(path, index=False)
                    generated[name] = sha256_file(path)

    manifest = {
        "stage": args.stage,
        "device": str(device),
        "batch_size": args.batch_size,
        "families": list(families),
        "seeds": list(seeds),
        "roles": ROLE_BY_STAGE[args.stage],
        "files": generated,
        "calibration_accessed": args.stage == "calibration",
        "locked_tests_accessed": args.stage == "locked",
    }
    _atomic_json(manifest, output_root / "prediction_manifest.json")
    if args.stage == "locked":
        assert access is not None
        access["status"] = "predictions_complete"
        access["prediction_manifest_sha256"] = sha256_file(output_root / "prediction_manifest.json")
        _atomic_json(access, access_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
