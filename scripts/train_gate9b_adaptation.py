#!/usr/bin/env python3
"""Train one frozen Gate 9B BEATs adaptation candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from train_gate9a_pretrained import (
    CLASS_NAMES,
    _assert_development_protocol,
    _augmentation,
    _evaluate,
    _prediction_frame,
    _sha256,
)

from respiratory_sound.data.audio import ICBHICycleDataset
from respiratory_sound.data.sampling import (
    DomainClassEventBatchSampler,
    EventRandomBatchSampler,
)
from respiratory_sound.distillation import binary_distillation_kl
from respiratory_sound.models.beats_adaptation import (
    configure_beats_adaptation,
    projection_drift_regularization,
    reset_projection_drift,
)
from respiratory_sound.models.pretrained_audio import load_beats_transfer
from respiratory_sound.runtime import select_device
from respiratory_sound.training import (
    consistency_weight,
    jensen_shannon_consistency,
    seed_everything,
    warmup_cosine_scheduler,
)
from respiratory_sound.training_state import (
    atomic_torch_save,
    capture_rng_state,
    load_trainable_parameter_state,
    restore_rng_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--mps-sync-debug", action="store_true")
    parser.add_argument("--diagnostic-output", type=Path)
    return parser.parse_args()


def _trainable_parameter_groups(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Module):
        raise ValueError("Gate 9B models must expose a classifier")
    classifier_parameters = [
        parameter for parameter in classifier.parameters()
        if parameter.requires_grad
    ]
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    adaptation_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in classifier_ids
    ]
    if not classifier_parameters:
        raise ValueError("Gate 9B classifier must be trainable")
    return adaptation_parameters, classifier_parameters


def _batch_sampler(
    rows: pd.DataFrame,
    experiment: dict[str, Any],
    *,
    batch_size: int,
    seed: int,
    label_column: str,
) -> tuple[Any, str]:
    mode = str(experiment.get("sampling", {}).get("mode", "domain_class_event"))
    common = {
        "rows": rows,
        "batch_size": batch_size,
        "samples_per_epoch": int(experiment["samples_per_epoch"]),
        "seed": seed,
    }
    if mode == "event_random":
        return EventRandomBatchSampler(**common), mode
    if mode == "domain_class_event":
        return (
            DomainClassEventBatchSampler(
                **common,
                class_column=label_column,
            ),
            mode,
        )
    raise ValueError(f"Unsupported Gate 9B/9D sampling mode: {mode}")


def _forward_batch(
    model: nn.Module,
    waveforms: Tensor,
    sample_masks: Tensor,
    device: torch.device,
) -> Tensor:
    reset_projection_drift(model)
    batch_size, num_views, channels, samples = waveforms.shape
    if channels != 1:
        raise ValueError("Gate 9B requires mono waveform views")
    flat_waveforms = waveforms.reshape(batch_size * num_views, samples).to(device)
    flat_masks = sample_masks.reshape(batch_size * num_views, samples).to(device)
    return model(flat_waveforms, flat_masks).reshape(batch_size, num_views, -1)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    consistency_strength: float,
    drift_anchor_strength: float,
    teacher_probability_lookup: dict[str, float] | None = None,
    distillation_strength: float = 0.0,
    distillation_temperature: float = 1.0,
    mps_sync_debug: bool = False,
    diagnostic_output: Path | None = None,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "consistency_loss": 0.0,
        "projection_drift": 0.0,
        "weighted_projection_drift_loss": 0.0,
        "distillation_loss": 0.0,
        "weighted_distillation_loss": 0.0,
    }
    total_samples = 0
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    def synchronize(
        stage: str,
        batch_index: int,
        sample_ids: list[str],
        sample_masks: Tensor,
    ) -> None:
        if not mps_sync_debug or device.type != "mps":
            return
        try:
            torch.mps.synchronize()
        except Exception:
            diagnostic = {
                "mps_sync_failure_stage": stage,
                "batch_index": batch_index,
                "sample_ids": list(sample_ids),
                "valid_samples_per_view": (
                    sample_masks.sum(dim=-1).tolist()
                ),
            }
            print(json.dumps(diagnostic), flush=True)
            raise

    for batch_index, (waveforms, sample_masks, targets, sample_ids) in enumerate(loader):
        pre_forward_rng = (
            capture_rng_state(device)
            if mps_sync_debug and diagnostic_output is not None
            else None
        )
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_batch(model, waveforms, sample_masks, device)
        synchronize("forward", batch_index, sample_ids, sample_masks)
        repeated_targets = targets[:, None].expand(-1, logits.shape[1]).reshape(-1)
        classification_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            repeated_targets,
        )
        consistency_loss = (
            jensen_shannon_consistency(logits)
            if logits.shape[1] > 1
            else logits.new_zeros(())
        )
        projection_drift = projection_drift_regularization(model)
        weighted_drift = drift_anchor_strength * projection_drift
        if teacher_probability_lookup is None:
            distillation_loss = logits.new_zeros(())
        else:
            teacher_probability = logits.new_tensor(
                [
                    teacher_probability_lookup[str(sample_id)]
                    for sample_id in sample_ids
                ]
            )
            distillation_loss = binary_distillation_kl(
                logits,
                teacher_probability,
                temperature=distillation_temperature,
            )
        weighted_distillation = distillation_strength * distillation_loss
        loss = (
            classification_loss
            + consistency_strength * consistency_loss
            + weighted_drift
            + weighted_distillation
        )
        if mps_sync_debug:
            finite_checks = {
                "logits": bool(torch.isfinite(logits).all().detach().cpu()),
                "classification_loss": bool(
                    torch.isfinite(classification_loss).detach().cpu()
                ),
                "consistency_loss": bool(
                    torch.isfinite(consistency_loss).detach().cpu()
                ),
                "projection_drift": bool(
                    torch.isfinite(projection_drift).detach().cpu()
                ),
                "distillation_loss": bool(
                    torch.isfinite(distillation_loss).detach().cpu()
                ),
                "total_loss": bool(torch.isfinite(loss).detach().cpu()),
            }
            if not all(finite_checks.values()):
                if diagnostic_output is not None:
                    trainable_state, trainable_names = _trainable_state(model)
                    atomic_torch_save(
                        {
                            "waveforms": waveforms,
                            "sample_masks": sample_masks,
                            "targets": targets.detach().cpu(),
                            "sample_ids": list(sample_ids),
                            "trainable_state_dict": trainable_state,
                            "trainable_parameter_names": trainable_names,
                            "pre_forward_rng_state": pre_forward_rng,
                            "consistency_strength": consistency_strength,
                            "drift_anchor_strength": drift_anchor_strength,
                            "batch_index": batch_index,
                        },
                        diagnostic_output,
                    )
                diagnostic = {
                    "nonfinite_training_batch": batch_index,
                    "sample_ids": list(sample_ids),
                    "finite_checks": finite_checks,
                    "logits": logits.detach().cpu().tolist(),
                    "valid_samples_per_view": sample_masks.sum(dim=-1).tolist(),
                }
                print(json.dumps(diagnostic), flush=True)
                raise FloatingPointError("Non-finite Gate 9B training batch")
        loss.backward()
        synchronize("backward", batch_index, sample_ids, sample_masks)
        nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        optimizer.step()
        synchronize("optimizer_step", batch_index, sample_ids, sample_masks)
        batch_size = targets.shape[0]
        values = {
            "loss": loss,
            "classification_loss": classification_loss,
            "consistency_loss": consistency_loss,
            "projection_drift": projection_drift,
            "weighted_projection_drift_loss": weighted_drift,
            "distillation_loss": distillation_loss,
            "weighted_distillation_loss": weighted_distillation,
        }
        for name, value in values.items():
            totals[name] += float(value.detach().cpu()) * batch_size
        total_samples += batch_size
    return {name: value / total_samples for name, value in totals.items()}


def _trainable_state(model: nn.Module) -> tuple[dict[str, Tensor], list[str]]:
    names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in names
    }
    if set(state) != names:
        raise RuntimeError("Could not serialize every trainable Gate 9B parameter")
    return state, sorted(names)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    data_config_path = (root / args.data_config).resolve()
    model_config_path = (root / args.model_config).resolve()
    experiment_config_path = (root / args.experiment_config).resolve()
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else experiment["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    train_role = str(experiment.get("train_value", "train_fit"))
    validation_role = str(experiment.get("validation_value", "validation_select"))
    _assert_development_protocol(manifest, train_role, validation_role)
    feature_config = feature_config_from_yaml(data_config_path)
    label_column = str(experiment.get("label_column", "binary_label_id"))
    consistency = experiment["consistency"]
    num_views = 2 if bool(consistency["enabled"]) else 1
    train_dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column="protocol_role",
        split_value=train_role,
        feature_config=feature_config,
        training=True,
        num_views=num_views,
        augmentation=_augmentation(experiment),
        label_column=label_column,
        return_waveform=True,
        waveform_only=True,
    )
    domains = sorted(str(value) for value in train_dataset.rows["dataset"].unique())
    if domains != ["icbhi2017", "sprsound2022"]:
        raise ValueError(f"Unexpected Gate 9B domains: {domains}")
    validation_datasets = {
        domain: ICBHICycleDataset(
            manifest_path=manifest_path,
            project_root=root,
            split_column="protocol_role",
            split_value=validation_role,
            feature_config=feature_config,
            training=False,
            num_views=1,
            label_column=label_column,
            return_waveform=True,
            waveform_only=True,
            row_filters={"dataset": domain},
        )
        for domain in domains
    }
    batch_size = int(experiment["batch_size"])
    sampler, sampling_mode = _batch_sampler(
        train_dataset.rows,
        experiment,
        batch_size=batch_size,
        seed=seed,
        label_column=label_column,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=int(experiment.get("num_workers", 0)),
        pin_memory=False,
    )
    validation_loaders = {
        domain: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(experiment.get("num_workers", 0)),
            pin_memory=False,
        )
        for domain, dataset in validation_datasets.items()
    }

    checkpoint = (root / str(model_config["checkpoint"])).resolve()
    source_dir = (root / str(model_config["source_dir"])).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing BEATs checkpoint: {checkpoint}")
    model = load_beats_transfer(checkpoint, source_dir)
    adaptation = model_config["adaptation"]
    audit = configure_beats_adaptation(
        model,
        strategy=str(adaptation["strategy"]),
        lora_last_n_layers=int(adaptation.get("last_n_layers", 4)),
        lora_rank=int(adaptation.get("rank", 8)),
        lora_alpha=float(adaptation.get("alpha", 16.0)),
        lora_dropout=float(adaptation.get("dropout", 0.05)),
    )
    adaptation_parameters, classifier_parameters = _trainable_parameter_groups(model)
    model = model.to(device)
    optimizer_config = experiment["optimizer"]
    parameter_groups: list[dict[str, Any]] = []
    if adaptation_parameters:
        parameter_groups.append(
            {
                "params": adaptation_parameters,
                "lr": float(optimizer_config["adaptation_learning_rate"]),
                "group_name": "adaptation",
            }
        )
    parameter_groups.append(
        {
            "params": classifier_parameters,
            "lr": float(optimizer_config["head_learning_rate"]),
            "group_name": "head",
        }
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    schedule = experiment["schedule"]
    max_epochs = int(schedule["max_epochs"])
    scheduler = warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=int(schedule["warmup_epochs"]),
        max_epochs=max_epochs,
    )
    patience = int(schedule["early_stopping_patience"])
    drift_anchor = experiment["projection_drift_anchor"]
    anchor_enabled = bool(drift_anchor["enabled"])
    distillation = experiment.get("distillation", {"enabled": False})
    distillation_enabled = bool(distillation.get("enabled", False))
    teacher_probability_lookup: dict[str, float] | None = None
    teacher_targets_path: Path | None = None
    teacher_targets_sha256: str | None = None
    if distillation_enabled:
        teacher_targets_path = (
            root / str(distillation["teacher_targets"])
        ).resolve()
        teacher_targets = pd.read_csv(teacher_targets_path)
        required_columns = {
            "sample_id",
            "target",
            "teacher_probability_1",
        }
        if not required_columns.issubset(teacher_targets):
            raise ValueError("Teacher target file is missing required columns")
        if teacher_targets["sample_id"].duplicated().any():
            raise ValueError("Teacher target file contains duplicate sample IDs")
        expected = train_dataset.rows[
            ["sample_id", label_column]
        ].rename(columns={label_column: "target"})
        audited = expected.merge(
            teacher_targets[list(required_columns)],
            on=["sample_id", "target"],
            how="inner",
            validate="one_to_one",
        )
        if len(audited) != len(expected) or len(audited) != len(teacher_targets):
            raise ValueError("Teacher targets do not exactly match train_fit")
        probabilities = audited["teacher_probability_1"].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or not (
            (probabilities >= 0.0) & (probabilities <= 1.0)
        ).all():
            raise ValueError("Teacher probabilities must be finite and in [0, 1]")
        teacher_probability_lookup = dict(
            zip(
                audited["sample_id"].astype(str),
                probabilities,
                strict=True,
            )
        )
        teacher_targets_sha256 = _sha256(teacher_targets_path)

    run_dir = root / "runs" / args.run_name
    resume_path = (
        (root / args.resume_state).resolve()
        if args.resume_state is not None
        else None
    )
    if resume_path is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        if resume_path.parent != run_dir.resolve():
            raise ValueError("Recovery state must belong to the requested run directory")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing recovery run directory: {run_dir}")
    gate_name = str(experiment.get("gate", "9B"))
    configuration = {
        "gate": gate_name,
        "candidate": str(model_config["candidate"]),
        "seed": seed,
        "device": str(device),
        "model": model_config,
        "experiment": experiment,
        "manifest_sha256": _sha256(manifest_path),
        "upstream_checkpoint_sha256": _sha256(checkpoint),
        "beats_backbone_source_sha256": _sha256(source_dir / "backbone.py"),
        "beats_transfer_wrapper_sha256": _sha256(
            root / "src/respiratory_sound/models/pretrained_audio.py"
        ),
        "formal_status": "exact_token_mask_corrected_candidate",
        "exact_token_mask_amendment": (
            "artifacts/gate9b_exact_token_mask_amendment.json"
        ),
        "token_mask_mapping": {
            "sample_frame_length": 400,
            "sample_frame_shift": 160,
            "fbank_time_frames_before_tail_padding": 798,
            "fbank_time_frames_after_tail_padding": 800,
            "patch_size": 16,
            "frequency_patches": 8,
            "tokens": 400,
            "supports_arbitrary_valid_audio_position": True,
        },
        "pretrained_load_audit": model.pretrained_load_audit,
        "adaptation_audit": audit.to_dict(),
        "train_samples": len(train_dataset),
        "validation_samples_by_domain": {
            domain: len(dataset)
            for domain, dataset in validation_datasets.items()
        },
        "batches_per_epoch": len(sampler),
        "sampling_mode": sampling_mode,
        "samples_per_batch_stratum": getattr(sampler, "per_stratum", None),
        "locked_test_accessed": False,
        "distillation": {
            "enabled": distillation_enabled,
            "teacher_targets": (
                str(teacher_targets_path.relative_to(root))
                if teacher_targets_path is not None
                else None
            ),
            "teacher_targets_sha256": teacher_targets_sha256,
            "temperature": (
                float(distillation["temperature"])
                if distillation_enabled
                else None
            ),
            "max_weight": (
                float(distillation["max_weight"])
                if distillation_enabled
                else 0.0
            ),
        },
    }
    configuration_path = run_dir / "configuration.json"
    if resume_path is None:
        configuration_path.write_text(
            json.dumps(configuration, indent=2),
            encoding="utf-8",
        )
    else:
        saved_configuration = json.loads(
            configuration_path.read_text(encoding="utf-8")
        )
        checks = {
            "candidate": saved_configuration["candidate"] == configuration["candidate"],
            "seed": saved_configuration["seed"] == configuration["seed"],
            "manifest": (
                saved_configuration["manifest_sha256"]
                == configuration["manifest_sha256"]
            ),
            "checkpoint": (
                saved_configuration["upstream_checkpoint_sha256"]
                == configuration["upstream_checkpoint_sha256"]
            ),
            "beats_backbone_source": (
                saved_configuration["beats_backbone_source_sha256"]
                == configuration["beats_backbone_source_sha256"]
            ),
            "beats_transfer_wrapper": (
                saved_configuration.get("beats_transfer_wrapper_sha256")
                == configuration["beats_transfer_wrapper_sha256"]
            ),
            "formal_status": (
                saved_configuration.get("formal_status")
                == configuration["formal_status"]
            ),
            "model": saved_configuration["model"] == configuration["model"],
            "experiment": (
                saved_configuration["experiment"] == configuration["experiment"]
            ),
        }
        if not all(checks.values()):
            raise ValueError(f"Recovery configuration mismatch: {checks}")

    history: list[dict[str, float]] = []
    best_worst = -1.0
    best_mean = -1.0
    best_epoch = 0
    best_row: dict[str, float] | None = None
    best_validation_metrics: dict[str, dict[str, Any]] | None = None
    stale_epochs = 0
    start_epoch = 0
    if resume_path is not None:
        recovery = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if int(recovery["schema_version"]) != 1:
            raise ValueError("Unsupported Gate 9B recovery schema")
        if (
            str(recovery["candidate"]) != str(model_config["candidate"])
            or int(recovery["seed"]) != seed
        ):
            raise ValueError("Recovery candidate or seed mismatch")
        load_trainable_parameter_state(
            model,
            recovery["trainable_state_dict"],
            list(recovery["trainable_parameter_names"]),
        )
        optimizer.load_state_dict(recovery["optimizer_state_dict"])
        scheduler.load_state_dict(recovery["scheduler_state_dict"])
        history = list(recovery["history"])
        best_worst = float(recovery["best_worst"])
        best_mean = float(recovery["best_mean"])
        best_epoch = int(recovery["best_epoch"])
        best_row = recovery["best_row"]
        best_validation_metrics = recovery["best_validation_metrics"]
        stale_epochs = int(recovery["stale_epochs"])
        start_epoch = int(recovery["next_epoch"])
        if start_epoch != len(history) or start_epoch >= max_epochs:
            raise ValueError("Invalid next epoch in Gate 9B recovery state")
        restore_rng_state(recovery["rng_state"], device)
        print(
            json.dumps({
                "recovered_from": str(resume_path),
                "next_epoch": start_epoch + 1,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
            }),
            flush=True,
        )

    for epoch in range(start_epoch, max_epochs):
        sampler.set_epoch(epoch)
        js_weight = consistency_weight(
            epoch,
            maximum=float(consistency["max_weight"]),
            warmup_epochs=int(consistency["warmup_epochs"]),
        )
        anchor_weight = (
            consistency_weight(
                epoch,
                maximum=float(drift_anchor["max_weight"]),
                warmup_epochs=int(drift_anchor["warmup_epochs"]),
            )
            if anchor_enabled
            else 0.0
        )
        distillation_weight = (
            consistency_weight(
                epoch,
                maximum=float(distillation["max_weight"]),
                warmup_epochs=int(distillation["warmup_epochs"]),
            )
            if distillation_enabled
            else 0.0
        )
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            consistency_strength=js_weight,
            drift_anchor_strength=anchor_weight,
            teacher_probability_lookup=teacher_probability_lookup,
            distillation_strength=distillation_weight,
            distillation_temperature=float(
                distillation.get("temperature", 1.0)
            ),
            mps_sync_debug=bool(args.mps_sync_debug),
            diagnostic_output=(
                (root / args.diagnostic_output).resolve()
                if args.diagnostic_output is not None
                else None
            ),
        )
        validation_results = {
            domain: _evaluate(model, loader, device)
            for domain, loader in validation_loaders.items()
        }
        domain_scores = {
            domain: float(result.metrics["average_score"])
            for domain, result in validation_results.items()
        }
        worst = min(domain_scores.values())
        mean = float(np.mean(tuple(domain_scores.values())))
        learning_rates = {
            str(group["group_name"]): float(group["lr"])
            for group in optimizer.param_groups
        }
        row = {
            "epoch": float(epoch + 1),
            "adaptation_learning_rate": learning_rates.get("adaptation", 0.0),
            "head_learning_rate": learning_rates["head"],
            "consistency_weight": js_weight,
            "projection_drift_anchor_weight": anchor_weight,
            "distillation_weight": distillation_weight,
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
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = worst > best_worst + 1.0e-12 or (
            abs(worst - best_worst) <= 1.0e-12
            and mean > best_mean + 1.0e-12
        )
        if improved:
            best_worst = worst
            best_mean = mean
            best_epoch = epoch + 1
            best_row = row
            best_validation_metrics = {
                domain: result.metrics
                for domain, result in validation_results.items()
            }
            stale_epochs = 0
            trainable_state, trainable_names = _trainable_state(model)
            torch.save(
                {
                    "trainable_state_dict": trainable_state,
                    "trainable_parameter_names": trainable_names,
                    "model_config": model_config,
                    "data_config": yaml.safe_load(
                        data_config_path.read_text(encoding="utf-8")
                    ),
                    "class_names": CLASS_NAMES,
                    "seed": seed,
                    "gate": gate_name,
                },
                run_dir / "best.pt",
            )
            for domain, result in validation_results.items():
                _prediction_frame(result).to_csv(
                    run_dir / f"best_validation_predictions_{domain}.csv",
                    index=False,
                )
        else:
            stale_epochs += 1
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        scheduler.step()
        current_state, current_names = _trainable_state(model)
        atomic_torch_save(
            {
                "schema_version": 1,
                "gate": gate_name,
                "candidate": str(model_config["candidate"]),
                "seed": seed,
                "next_epoch": epoch + 1,
                "history": history,
                "best_worst": best_worst,
                "best_mean": best_mean,
                "best_epoch": best_epoch,
                "best_row": best_row,
                "best_validation_metrics": best_validation_metrics,
                "stale_epochs": stale_epochs,
                "trainable_state_dict": current_state,
                "trainable_parameter_names": current_names,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": capture_rng_state(device),
                "locked_test_accessed": False,
            },
            run_dir / "resume_state.pt",
        )
        if stale_epochs >= patience:
            break

    if best_validation_metrics is None or best_row is None:
        raise RuntimeError("Gate 9B training did not produce a checkpoint")
    summary = {
        "gate": gate_name,
        "candidate": str(model_config["candidate"]),
        "strategy": str(adaptation["strategy"]),
        "seed": seed,
        "selection_metric": "minimum_domain_average_score",
        "best_epoch": best_epoch,
        "best_worst_domain_average_score": best_worst,
        "best_mean_domain_average_score": best_mean,
        "epochs_completed": len(history),
        "training_parameters": audit.trainable_parameters,
        "total_parameters": audit.total_parameters,
        "trainable_fraction": audit.trainable_fraction,
        "best_train_projection_drift": best_row["train_projection_drift"],
        "best_projection_drift_anchor_weight": best_row[
            "projection_drift_anchor_weight"
        ],
        "best_validation_metrics": best_validation_metrics,
        "locked_test_accessed": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
