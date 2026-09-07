#!/usr/bin/env python3
"""Train one frozen BEATS revision-stage ablation seed."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from train_gate11a_exactmask_fullft import (
    MemoryTracker,
    _atomic_csv,
    _atomic_json,
    _augmentation,
    _cpu_clone,
    _evaluate,
    _prediction_frame,
    _train_one_epoch,
)

from respiratory_sound.data.audio import ICBHICycleDataset
from respiratory_sound.data.sampling import DomainClassEventBatchSampler
from respiratory_sound.gate11a import (
    ALLOWED_SEEDS,
    DOMAINS,
    TRAIN_ROLE,
    VALIDATION_ROLE,
    assert_gate11a_protocol,
    assert_selected_rows,
    sha256_file,
)
from respiratory_sound.models.beats_adaptation import configure_beats_adaptation
from respiratory_sound.models.pretrained_audio import load_beats_transfer
from respiratory_sound.runtime import select_device
from respiratory_sound.training import consistency_weight, seed_everything, warmup_cosine_scheduler
from respiratory_sound.training_state import (
    atomic_torch_save,
    capture_rng_state,
    load_trainable_parameter_state,
    restore_rng_state,
)

PLAN_RELATIVE = Path(
    "../experiment_plans/2026-09-03/"
    "01_REVISION_ANALYSIS_PLAN_FROZEN_2026-09-03.md"
)
PLAN_SHA256 = "18b6e39bba441e5be1ec870591b9dbf2846b7e6ab6293be9d211aab12809be39"
MANIFEST_SHA256 = "2dd168129fc0fc159db201f2990c4d7074cb746fc9bc2c13bd5031e47dbd1419"
UPSTREAM_SHA256 = "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c"
FIXED_PREFIX_EPOCHS = {20_260_729: 7, 20_260_730: 9, 20_260_731: 6}
FAMILIES = ("legacy-mask", "no-js", "matched-lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _family_spec(config: dict[str, Any], family: str, seed: int) -> dict[str, Any]:
    common = dict(config["common"])
    selected = dict(config["families"][family])
    spec = {**common, **selected}
    spec["family"] = family
    spec["seed"] = seed
    if bool(spec["fixed_prefix"]):
        spec["epochs_to_run"] = FIXED_PREFIX_EPOCHS[seed]
    else:
        spec["epochs_to_run"] = int(spec["max_epochs"])
    return spec


def _trainable_state(model: nn.Module) -> tuple[dict[str, Tensor], list[str]]:
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return state, list(state)


def _parameter_groups(
    model: nn.Module,
    *,
    adaptation: str,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Module):
        raise ValueError("Revision model must expose a classifier")
    head = [parameter for parameter in classifier.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in head}
    backbone_or_adapter = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    if not head or not backbone_or_adapter:
        raise ValueError("Revision optimiser groups must both be non-empty")
    if adaptation == "fullft" and not all(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("Full-FT did not expose every parameter")
    return backbone_or_adapter, head


def _software() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
    }


def _save_best(
    model: nn.Module,
    path: Path,
    *,
    family: str,
    seed: int,
    spec: dict[str, Any],
    selection: dict[str, float | int],
    configuration_sha256: str,
) -> None:
    state, names = _trainable_state(model)
    atomic_torch_save(
        {
            "schema_version": 1,
            "stage": "BEATS_revision",
            "family": family,
            "seed": seed,
            "state_mode": "trainable_parameters",
            "trainable_state_dict": state,
            "trainable_parameter_names": names,
            "specification": spec,
            "selection": selection,
            "configuration_sha256": configuration_sha256,
            "analysis_plan_sha256": PLAN_SHA256,
            "calibration_accessed": False,
            "locked_tests_accessed": False,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    seed = int(args.seed)
    config_path = root / "configs/revision/beats_ablation.yaml"
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    data_config_path = root / "configs/data/gate9a_beats.yaml"
    model_config_path = root / "configs/model/gate9a_beats_as2m_cpt2.yaml"
    freeze_path = root / "artifacts/revision_2026_09_03/implementation_freeze.json"
    plan_path = (root / PLAN_RELATIVE).resolve()

    required = (
        config_path,
        manifest_path,
        data_config_path,
        model_config_path,
        plan_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing revision inputs: {missing}")
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("Frozen revision analysis plan hash changed")
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Frozen analysis manifest hash changed")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    spec = _family_spec(config, args.family, seed)
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    checkpoint_path = root / str(model_config["checkpoint"])
    beats_source = root / str(model_config["source_dir"])
    if sha256_file(checkpoint_path) != UPSTREAM_SHA256:
        raise ValueError("Frozen upstream BEATs checkpoint hash changed")

    formal = args.smoke_batches == 0
    if formal:
        if args.device != "mps":
            raise ValueError("Formal revision training is frozen to MPS")
        if not freeze_path.is_file():
            raise FileNotFoundError("Implementation freeze must exist before formal training")
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        for relative, expected in freeze["files"].items():
            observed = sha256_file(root / relative)
            if observed != expected:
                raise ValueError(
                    f"Frozen implementation changed: {relative}: {observed} != {expected}"
                )

    seed_everything(seed)
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    protocol_audit = assert_gate11a_protocol(
        manifest,
        train_role=TRAIN_ROLE,
        validation_role=VALIDATION_ROLE,
    )
    feature_config = feature_config_from_yaml(data_config_path)
    augmentation = _augmentation(config["common"])
    train_dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column="protocol_role",
        split_value=TRAIN_ROLE,
        feature_config=feature_config,
        training=True,
        num_views=2,
        augmentation=augmentation,
        label_column="binary_label_id",
        return_waveform=True,
        waveform_only=True,
    )
    assert_selected_rows(train_dataset.rows, expected_role=TRAIN_ROLE)
    label_lookup = dict(
        zip(
            train_dataset.rows["sample_id"].astype(str),
            train_dataset.rows["binary_label_id"].astype(int),
            strict=True,
        )
    )
    validation_datasets = {
        domain: ICBHICycleDataset(
            manifest_path=manifest_path,
            project_root=root,
            split_column="protocol_role",
            split_value=VALIDATION_ROLE,
            feature_config=feature_config,
            training=False,
            num_views=1,
            label_column="binary_label_id",
            return_waveform=True,
            waveform_only=True,
            row_filters={"dataset": domain},
        )
        for domain in DOMAINS
    }
    for domain, dataset in validation_datasets.items():
        assert_selected_rows(
            dataset.rows,
            expected_role=VALIDATION_ROLE,
            expected_domain=domain,
        )

    sampler = DomainClassEventBatchSampler(
        train_dataset.rows,
        batch_size=int(spec["batch_size"]),
        samples_per_epoch=int(spec["samples_per_epoch"]),
        seed=seed,
        class_column="binary_label_id",
    )
    generator = torch.Generator()
    generator.manual_seed(seed + 11_000_000)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=int(spec["num_workers"]),
        pin_memory=False,
        generator=generator,
    )
    validation_loaders = {
        domain: DataLoader(
            dataset,
            batch_size=int(spec["batch_size"]),
            shuffle=False,
            num_workers=int(spec["num_workers"]),
            pin_memory=False,
        )
        for domain, dataset in validation_datasets.items()
    }

    model = load_beats_transfer(
        checkpoint_path,
        beats_source,
        mask_mode=str(spec["mask_mode"]),
    )
    adaptation_audit: dict[str, Any] | None = None
    if spec["adaptation"] == "lora":
        adaptation_audit = configure_beats_adaptation(
            model,
            strategy="lora_qv",
            lora_last_n_layers=4,
            lora_rank=8,
            lora_alpha=16.0,
            lora_dropout=0.05,
        ).to_dict()
        if int(adaptation_audit["trainable_parameters"]) != 99_842:
            raise ValueError("Matched LoRA trainable-parameter count changed")
    backbone_or_adapter, head = _parameter_groups(
        model,
        adaptation=str(spec["adaptation"]),
    )
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    model = model.to(device)

    first_group_name = "backbone" if spec["adaptation"] == "fullft" else "adaptation"
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_or_adapter,
                "lr": float(spec["backbone_or_adapter_learning_rate"]),
                "group_name": first_group_name,
            },
            {
                "params": head,
                "lr": float(spec["head_learning_rate"]),
                "group_name": "head",
            },
        ],
        weight_decay=float(spec["weight_decay"]),
    )
    scheduler = warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=int(spec["warmup_epochs"]),
        max_epochs=int(spec["schedule_horizon_epochs"]),
    )

    memory = MemoryTracker()
    memory.update(device)
    if args.smoke_batches:
        sampler.set_epoch(0)
        limited = []
        for index, batch in enumerate(train_loader):
            limited.append(batch)
            if index + 1 >= args.smoke_batches:
                break
        smoke_metrics = _train_one_epoch(
            model,
            limited,
            optimizer,
            device,
            consistency_strength=float(spec["js_max_weight"]) / 20.0,
            memory_tracker=memory,
            label_lookup=label_lookup,
        )
        smoke_result = {
            "status": "smoke_passed",
            "family": args.family,
            "seed": seed,
            "batches": len(limited),
            "mask_mode": spec["mask_mode"],
            "adaptation": spec["adaptation"],
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "trainable_parameter_names": trainable_names,
            "metrics": smoke_metrics,
            "memory": memory.to_dict(),
            "analysis_plan_sha256": PLAN_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
        }
        smoke_path = (
            root
            / "artifacts/revision_2026_09_03/smoke"
            / f"{args.family.replace('-', '_')}_seed{seed}.json"
        )
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(smoke_result, smoke_path)
        print(
            json.dumps(
                {
                    **{
                        key: value
                        for key, value in smoke_result.items()
                        if key != "trainable_parameter_names"
                    },
                    "trainable_parameter_name_count": len(trainable_names),
                    "output": str(smoke_path.relative_to(root)),
                    "output_sha256": sha256_file(smoke_path),
                },
                indent=2,
            )
        )
        return

    run_name = f"revision_{args.family.replace('-', '_')}_seed{seed}"
    run_dir = root / "runs/revision_2026_09_03" / run_name
    resume_path = _resolve(root, args.resume_state).resolve() if args.resume_state else None
    if resume_path is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif resume_path.parent != run_dir.resolve() or not resume_path.is_file():
        raise ValueError("Resume state must belong to this revision run")

    configuration = {
        "schema_version": 1,
        "stage": "BEATS_revision",
        "run_name": run_name,
        "family": args.family,
        "seed": seed,
        "device": str(device),
        "analysis_plan": str(plan_path),
        "analysis_plan_sha256": PLAN_SHA256,
        "implementation_freeze": str(freeze_path.relative_to(root)),
        "implementation_freeze_sha256": sha256_file(freeze_path),
        "manifest_sha256": MANIFEST_SHA256,
        "upstream_checkpoint_sha256": UPSTREAM_SHA256,
        "config_sha256": sha256_file(config_path),
        "specification": spec,
        "protocol_audit": protocol_audit,
        "pretrained_load_audit": model.pretrained_load_audit,
        "adaptation_audit": adaptation_audit,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_names": trainable_names,
        "batches_per_epoch": len(sampler),
        "software": _software(),
        "calibration_accessed": False,
        "locked_tests_accessed": False,
    }
    configuration_path = run_dir / "configuration.json"
    if resume_path is None:
        _atomic_json(configuration, configuration_path)
    elif json.loads(configuration_path.read_text(encoding="utf-8")) != configuration:
        raise ValueError("Revision recovery configuration changed")
    configuration_sha256 = sha256_file(configuration_path)

    history: list[dict[str, float]] = []
    best_worst = -1.0
    best_mean = -1.0
    best_epoch = 0
    best_metrics: dict[str, dict[str, Any]] | None = None
    stale_epochs = 0
    start_epoch = 0
    wall_started = time.monotonic()
    if resume_path is not None:
        recovery = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            recovery["configuration_sha256"] != configuration_sha256
            or recovery["analysis_plan_sha256"] != PLAN_SHA256
            or recovery["family"] != args.family
            or int(recovery["seed"]) != seed
        ):
            raise ValueError("Revision recovery identity mismatch")
        load_trainable_parameter_state(
            model,
            recovery["trainable_state_dict"],
            recovery["trainable_parameter_names"],
        )
        optimizer.load_state_dict(recovery["optimizer_state_dict"])
        scheduler.load_state_dict(recovery["scheduler_state_dict"])
        history = list(recovery["history"])
        best_worst = float(recovery["best_worst"])
        best_mean = float(recovery["best_mean"])
        best_epoch = int(recovery["best_epoch"])
        best_metrics = recovery["best_metrics"]
        stale_epochs = int(recovery["stale_epochs"])
        start_epoch = int(recovery["next_epoch"])
        generator.set_state(recovery["train_loader_generator_state"])
        restore_rng_state(recovery["rng_state"], device)
        saved_memory = recovery["memory"]
        memory = MemoryTracker(
            peak_process_rss_bytes=int(saved_memory["peak_process_rss_bytes"]),
            peak_mps_current_allocated_bytes=int(
                saved_memory["peak_mps_current_allocated_bytes"]
            ),
            peak_mps_driver_allocated_bytes=int(
                saved_memory["peak_mps_driver_allocated_bytes"]
            ),
        )

    maximum_epochs = int(spec["epochs_to_run"])
    patience = int(spec["early_stopping_patience"])
    for epoch in range(start_epoch, maximum_epochs):
        epoch_started = time.monotonic()
        sampler.set_epoch(epoch)
        js_weight = consistency_weight(
            epoch,
            maximum=float(spec["js_max_weight"]),
            warmup_epochs=20,
        )
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            consistency_strength=js_weight,
            memory_tracker=memory,
            label_lookup=label_lookup,
        )
        validation_results = {
            domain: _evaluate(model, loader, device, memory_tracker=memory)
            for domain, loader in validation_loaders.items()
        }
        scores = {
            domain: float(result.metrics["average_score"])
            for domain, result in validation_results.items()
        }
        worst = min(scores.values())
        mean = float(np.mean(tuple(scores.values())))
        learning_rates = {
            str(group["group_name"]): float(group["lr"])
            for group in optimizer.param_groups
        }
        row = {
            "epoch": float(epoch + 1),
            "epoch_seconds": time.monotonic() - epoch_started,
            f"{first_group_name}_learning_rate": learning_rates[first_group_name],
            "head_learning_rate": learning_rates["head"],
            "consistency_weight": js_weight,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            "validation_worst_average_score": worst,
            "validation_mean_average_score": mean,
            **{
                f"validation_{domain}_{metric}": float(value)
                for domain, result in validation_results.items()
                for metric, value in result.metrics.items()
                if isinstance(value, float)
            },
        }
        if not all(np.isfinite(float(value)) for value in row.values()):
            raise FloatingPointError("Revision epoch history became non-finite")
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = worst > best_worst + 1.0e-12 or (
            abs(worst - best_worst) <= 1.0e-12 and mean > best_mean + 1.0e-12
        )
        if improved:
            best_worst = worst
            best_mean = mean
            best_epoch = epoch + 1
            best_metrics = {
                domain: result.metrics for domain, result in validation_results.items()
            }
            stale_epochs = 0
            _save_best(
                model,
                run_dir / "best.pt",
                family=args.family,
                seed=seed,
                spec=spec,
                selection={
                    "epoch": best_epoch,
                    "minimum_domain_average_score": best_worst,
                    "mean_domain_average_score": best_mean,
                },
                configuration_sha256=configuration_sha256,
            )
            for domain, result in validation_results.items():
                _atomic_csv(
                    _prediction_frame(
                        result,
                        validation_datasets[domain].rows,
                        domain=domain,
                    ),
                    run_dir / f"best_validation_predictions_{domain}.csv",
                )
        else:
            stale_epochs += 1
        _atomic_csv(pd.DataFrame(history), run_dir / "history.csv")
        scheduler.step()
        state, names = _trainable_state(model)
        training_complete = epoch + 1 >= maximum_epochs or (
            not bool(spec["fixed_prefix"]) and stale_epochs >= patience
        )
        atomic_torch_save(
            {
                "schema_version": 1,
                "family": args.family,
                "seed": seed,
                "configuration_sha256": configuration_sha256,
                "analysis_plan_sha256": PLAN_SHA256,
                "next_epoch": epoch + 1,
                "history": history,
                "best_worst": best_worst,
                "best_mean": best_mean,
                "best_epoch": best_epoch,
                "best_metrics": best_metrics,
                "stale_epochs": stale_epochs,
                "training_complete": training_complete,
                "trainable_state_dict": state,
                "trainable_parameter_names": names,
                "optimizer_state_dict": _cpu_clone(optimizer.state_dict()),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": capture_rng_state(device),
                "train_loader_generator_state": generator.get_state(),
                "memory": memory.to_dict(),
                "calibration_accessed": False,
                "locked_tests_accessed": False,
            },
            run_dir / "resume_state.pt",
        )
        if training_complete:
            break

    if best_metrics is None:
        raise RuntimeError("Revision training produced no checkpoint")
    best_path = run_dir / "best.pt"
    summary = {
        "schema_version": 1,
        "stage": "BEATS_revision",
        "family": args.family,
        "seed": seed,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_worst_domain_average_score": best_worst,
        "best_mean_domain_average_score": best_mean,
        "best_validation_metrics": best_metrics,
        "fixed_prefix": bool(spec["fixed_prefix"]),
        "training_seconds_total": float(sum(row["epoch_seconds"] for row in history)),
        "wall_seconds_this_invocation": time.monotonic() - wall_started,
        "memory": memory.to_dict(),
        "trainable_parameters": configuration["trainable_parameters"],
        "total_parameters": configuration["total_parameters"],
        "best_checkpoint_bytes": best_path.stat().st_size,
        "best_checkpoint_sha256": sha256_file(best_path),
        "configuration_sha256": configuration_sha256,
        "finite_audit_passed": True,
        "data_boundary_audit_passed": True,
        "calibration_accessed": False,
        "locked_tests_accessed": False,
    }
    _atomic_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
