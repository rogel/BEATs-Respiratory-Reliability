#!/usr/bin/env python3
"""Generate frozen revision-stage predictions after a family is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from compute_feature_stats import feature_config_from_yaml

from respiratory_sound.gate11a import ALLOWED_SEEDS, sha256_file
from respiratory_sound.models.beats_adaptation import configure_beats_adaptation
from respiratory_sound.models.pretrained_audio import load_beats_transfer
from respiratory_sound.post_gate11a import (
    build_waveform_dataset,
    infer_probabilities,
)
from respiratory_sound.runtime import select_device
from respiratory_sound.training_state import load_trainable_parameter_state

PLAN_SHA256 = "18b6e39bba441e5be1ec870591b9dbf2846b7e6ab6293be9d211aab12809be39"
AMENDMENT_SHA256 = "9b2b7e19ae0f14fc8e3107b579d45987e9a729753507f858a2ca12ce94aa91e4"
MANIFEST_SHA256 = "2dd168129fc0fc159db201f2990c4d7074cb746fc9bc2c13bd5031e47dbd1419"
FAMILIES = ("legacy-mask", "no-js", "matched-lora")
ROLES = {
    "icbhi2017": ("validation_select", "calibration", "locked_test"),
    "sprsound2022": (
        "validation_select",
        "calibration",
        "locked_inter_test",
        "locked_intra_test",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", choices=ALLOWED_SEEDS)
    return parser.parse_args()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_dir(root: Path, family: str, seed: int) -> Path:
    return (
        root
        / "runs/revision_2026_09_03"
        / f"revision_{family.replace('-', '_')}_seed{seed}"
    )


def _ensemble(frames: list[pd.DataFrame], seeds: tuple[int, ...]) -> pd.DataFrame:
    if len(frames) != len(seeds) or not frames:
        raise ValueError("Ensemble frames and seeds must be non-empty and aligned")
    keys = [
        "sample_id",
        "dataset",
        "patient_id",
        "protocol_role",
        "locked",
        "binary_label_id",
        "fine_label_name",
        "event_duration_seconds",
        "target",
    ]
    merged = frames[0][keys + ["probability_1"]].rename(
        columns={"probability_1": f"probability_1_seed{seeds[0]}"}
    )
    for seed, frame in zip(seeds[1:], frames[1:], strict=True):
        merged = merged.merge(
            frame[keys + ["probability_1"]].rename(
                columns={"probability_1": f"probability_1_seed{seed}"}
            ),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    columns = [f"probability_1_seed{seed}" for seed in seeds]
    merged["probability_1"] = merged[columns].mean(axis=1)
    merged["probability_0"] = 1.0 - merged["probability_1"]
    merged["prediction"] = (merged["probability_1"] >= 0.5).astype(int)
    if len(merged) != len(frames[0]):
        raise ValueError("Revision ensemble lost samples")
    return merged


def _load_model(
    root: Path,
    *,
    family: str,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    run_dir = _run_dir(root, family, seed)
    summary_path = run_dir / "summary.json"
    configuration_path = run_dir / "configuration.json"
    checkpoint_path = run_dir / "best.pt"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        summary["family"] != family
        or configuration["family"] != family
        or checkpoint["family"] != family
        or int(summary["seed"]) != seed
        or int(configuration["seed"]) != seed
        or int(checkpoint["seed"]) != seed
        or checkpoint["analysis_plan_sha256"] != PLAN_SHA256
        or summary["best_checkpoint_sha256"] != sha256_file(checkpoint_path)
        or checkpoint["configuration_sha256"] != sha256_file(configuration_path)
        or summary["configuration_sha256"] != sha256_file(configuration_path)
    ):
        raise ValueError(f"Revision checkpoint identity mismatch: {run_dir}")
    if summary["calibration_accessed"] or summary["locked_tests_accessed"]:
        raise ValueError(f"Training run crossed its data boundary: {run_dir}")
    model_config = yaml.safe_load(
        (root / "configs/model/gate9a_beats_as2m_cpt2.yaml").read_text(
            encoding="utf-8"
        )
    )
    specification = checkpoint["specification"]
    model = load_beats_transfer(
        root / str(model_config["checkpoint"]),
        root / str(model_config["source_dir"]),
        mask_mode=str(specification["mask_mode"]),
    )
    if specification["adaptation"] == "lora":
        audit = configure_beats_adaptation(
            model,
            strategy="lora_qv",
            lora_last_n_layers=4,
            lora_rank=8,
            lora_alpha=16.0,
            lora_dropout=0.05,
        )
        if audit.trainable_parameters != 99_842:
            raise ValueError("Matched-LoRA parameter count changed at inference")
    load_trainable_parameter_state(
        model,
        checkpoint["trainable_state_dict"],
        list(checkpoint["trainable_parameter_names"]),
    )
    model.eval()
    return model.to(device), {
        "run": str(run_dir.relative_to(root)),
        "summary_sha256": sha256_file(summary_path),
        "configuration_sha256": sha256_file(configuration_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_epoch": int(summary["best_epoch"]),
        "epochs_completed": int(summary["epochs_completed"]),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Revision manifest hash changed")
    if args.batch_size != 8:
        raise ValueError("Formal revision inference is frozen to batch size 8")
    seeds = tuple(args.seeds or ALLOWED_SEEDS)
    expected_seeds = (
        (20_260_729, 20_260_731) if args.family == "legacy-mask" else ALLOWED_SEEDS
    )
    if seeds != expected_seeds:
        raise ValueError(
            f"Frozen seed set for {args.family} is {expected_seeds}, not {seeds}"
        )
    if args.family == "legacy-mask":
        amendment_path = (
            root
            / "../experiment_plans/2026-09-03/"
            "03_ANALYSIS_PLAN_AMENDMENT_2026-09-04_LEGACY_FAILURE.md"
        ).resolve()
        if sha256_file(amendment_path) != AMENDMENT_SHA256:
            raise ValueError("Legacy failure amendment hash changed")
    inference_freeze_path = (
        root / "artifacts/revision_2026_09_03/analysis_a_inference_freeze.json"
    )
    inference_freeze = json.loads(inference_freeze_path.read_text(encoding="utf-8"))
    if inference_freeze.get("status") != (
        "frozen_before_analysis_a_calibration_and_heldout_inference"
    ):
        raise ValueError("Analysis A inference freeze is invalid")
    if sha256_file(Path(__file__).resolve()) != inference_freeze.get(
        "inference_script_sha256"
    ):
        raise ValueError("Frozen Analysis A inference script hash changed")
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")
    output_root = (
        root
        / "artifacts/revision_2026_09_03/predictions"
        / args.family.replace("-", "_")
    )
    output_root.mkdir(parents=True, exist_ok=False)
    feature_config = feature_config_from_yaml(root / "configs/data/gate9a_beats.yaml")
    datasets = {
        (domain, role): build_waveform_dataset(
            manifest_path=manifest_path,
            project_root=root,
            feature_config=feature_config,
            role=role,
            domain=domain,
        )
        for domain, roles in ROLES.items()
        for role in roles
    }
    files: dict[str, str] = {}
    runs: dict[str, Any] = {}
    for seed in seeds:
        model, run_audit = _load_model(
            root,
            family=args.family,
            seed=seed,
            device=device,
        )
        runs[str(seed)] = run_audit
        for (domain, role), dataset in datasets.items():
            frame = infer_probabilities(
                model,
                dataset,
                device=device,
                batch_size=args.batch_size,
            )
            name = f"seed{seed}_{domain}_{role}.csv"
            path = output_root / name
            frame.to_csv(path, index=False)
            files[name] = sha256_file(path)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    for domain, roles in ROLES.items():
        for role in roles:
            frames = [
                pd.read_csv(
                    output_root / f"seed{seed}_{domain}_{role}.csv",
                    dtype={"patient_id": str},
                )
                for seed in seeds
            ]
            ensemble = _ensemble(frames, seeds)
            name = f"ensemble_{domain}_{role}.csv"
            path = output_root / name
            ensemble.to_csv(path, index=False)
            files[name] = sha256_file(path)

    manifest = {
        "schema_version": 1,
        "stage": "BEATS_revision_predictions",
        "family": args.family,
        "device": str(device),
        "batch_size": args.batch_size,
        "analysis_plan_sha256": PLAN_SHA256,
        "legacy_failure_amendment_sha256": (
            AMENDMENT_SHA256 if args.family == "legacy-mask" else None
        ),
        "manifest_sha256": MANIFEST_SHA256,
        "analysis_a_inference_freeze_sha256": sha256_file(inference_freeze_path),
        "runs": runs,
        "seeds": seeds,
        "roles": ROLES,
        "files": files,
        "calibration_accessed": True,
        "locked_tests_accessed": True,
        "selection_after_locked_access": False,
    }
    _atomic_json(manifest, output_root / "prediction_manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
