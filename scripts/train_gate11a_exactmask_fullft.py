#!/usr/bin/env python3
"""Run one frozen Gate 11A exact-mask BEATs full-finetuning seed."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import AugmentationConfig, ICBHICycleDataset
from respiratory_sound.data.sampling import DomainClassEventBatchSampler
from respiratory_sound.gate11a import (
    ALLOWED_SEEDS,
    CLASS_NAMES,
    DOMAINS,
    SINGLE_SEED_THRESHOLDS,
    THREE_SEED_THRESHOLDS,
    TRAIN_ROLE,
    VALIDATION_ROLE,
    assert_gate11a_protocol,
    assert_selected_rows,
    gate11a_hashes,
    sha256_file,
    single_seed_decision,
)
from respiratory_sound.metrics import respiratory_metrics
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
    restore_rng_state,
)


@dataclass(frozen=True)
class PredictionResult:
    loss: float
    metrics: dict[str, Any]
    targets: np.ndarray
    probabilities: np.ndarray
    sample_ids: list[str]


@dataclass
class MemoryTracker:
    """Track explicit MPS allocations and process high-water memory."""

    peak_process_rss_bytes: int = 0
    peak_mps_current_allocated_bytes: int = 0
    peak_mps_driver_allocated_bytes: int = 0

    def update(self, device: torch.device) -> None:
        raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        rss_bytes = raw_rss if sys.platform == "darwin" else raw_rss * 1024
        self.peak_process_rss_bytes = max(
            self.peak_process_rss_bytes,
            rss_bytes,
        )
        if device.type == "mps":
            self.peak_mps_current_allocated_bytes = max(
                self.peak_mps_current_allocated_bytes,
                int(torch.mps.current_allocated_memory()),
            )
            self.peak_mps_driver_allocated_bytes = max(
                self.peak_mps_driver_allocated_bytes,
                int(torch.mps.driver_allocated_memory()),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_process_rss_bytes": self.peak_process_rss_bytes,
            "peak_mps_current_allocated_bytes": (
                self.peak_mps_current_allocated_bytes
            ),
            "peak_mps_driver_allocated_bytes": (
                self.peak_mps_driver_allocated_bytes
            ),
            "measurement": (
                "process ru_maxrss plus MPS allocation samples after every batch"
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/cross_domain_binary.csv"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/gate9a_beats.yaml"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/gate9a_beats_as2m_cpt2.yaml"),
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path(
            "configs/experiment/"
            "gate11a_beats_exactmask_fullft_seed20260729.yaml"
        ),
    )
    parser.add_argument(
        "--freeze-file",
        type=Path,
        default=Path("artifacts/gate11a_exactmask_fullft_freeze.json"),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--resume-state", type=Path)
    return parser.parse_args()


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _model_state_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }


def _augmentation(experiment: dict[str, Any]) -> AugmentationConfig:
    values = experiment["augmentation"]
    return AugmentationConfig(
        gain_min=float(values["gain_min"]),
        gain_max=float(values["gain_max"]),
        noise_probability=float(values["noise_probability"]),
        noise_snr_min_db=float(values["noise_snr_min_db"]),
        noise_snr_max_db=float(values["noise_snr_max_db"]),
        shift_probability=float(values["shift_probability"]),
        max_shift_fraction=float(values["max_shift_fraction"]),
        frequency_mask_bins=0,
        time_mask_frames=0,
    )


def _classifier_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Module):
        raise ValueError("Gate 11A BEATs must expose a classifier")
    classifier_parameters = list(classifier.parameters())
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in classifier_ids
    ]
    if not classifier_parameters or not backbone_parameters:
        raise ValueError("Gate 11A parameter groups must both be non-empty")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Gate 11A requires every model parameter to be trainable")
    return backbone_parameters, classifier_parameters


def _assert_waveform_batch(
    waveforms: Tensor,
    sample_masks: Tensor,
) -> None:
    if waveforms.ndim != 4:
        raise ValueError("Gate 11A waveforms must be [batch, views, channels, samples]")
    if sample_masks.shape != (
        waveforms.shape[0],
        waveforms.shape[1],
        waveforms.shape[3],
    ):
        raise ValueError("Gate 11A sample masks do not match waveforms")
    if waveforms.shape[2] != 1:
        raise ValueError("Gate 11A requires mono waveform views")
    if sample_masks.dtype != torch.bool:
        raise ValueError("Gate 11A sample masks must be boolean")
    if bool((~sample_masks.any(dim=-1)).any()):
        raise ValueError("Gate 11A received an all-padding waveform view")
    if not bool(torch.isfinite(waveforms).all()):
        raise FloatingPointError("Gate 11A received a non-finite waveform")


def _forward_batch(
    model: nn.Module,
    waveforms: Tensor,
    sample_masks: Tensor,
    device: torch.device,
) -> Tensor:
    _assert_waveform_batch(waveforms, sample_masks)
    batch_size, num_views, _, samples = waveforms.shape
    flat_waveforms = waveforms.reshape(batch_size * num_views, samples).to(device)
    flat_masks = sample_masks.reshape(batch_size * num_views, samples).to(device)
    logits = model(flat_waveforms, flat_masks).reshape(batch_size, num_views, -1)
    if not bool(torch.isfinite(logits).all().detach().cpu()):
        raise FloatingPointError("Gate 11A produced non-finite logits")
    return logits


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    consistency_strength: float,
    memory_tracker: MemoryTracker,
    label_lookup: dict[str, int],
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "consistency_loss": 0.0,
        "gradient_norm_before_clip": 0.0,
    }
    total_samples = 0
    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    for waveforms, sample_masks, targets, sample_ids in loader:
        expected_targets = torch.tensor(
            [label_lookup[str(sample_id)] for sample_id in sample_ids],
            dtype=targets.dtype,
        )
        if not torch.equal(targets, expected_targets):
            raise ValueError("Gate 11A sample/label mismatch during training")
        targets = targets.to(device)
        if not bool(((targets == 0) | (targets == 1)).all().detach().cpu()):
            raise ValueError("Gate 11A received a non-binary target")
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_batch(model, waveforms, sample_masks, device)
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
        loss = classification_loss + consistency_strength * consistency_loss
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise FloatingPointError("Gate 11A produced a non-finite training loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if (
                parameter.requires_grad
                and parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all().detach().cpu())
            ):
                raise FloatingPointError(
                    f"Gate 11A produced a non-finite gradient: {name}"
                )
        gradient_norm = nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
        if not bool(torch.isfinite(gradient_norm).detach().cpu()):
            raise FloatingPointError("Gate 11A gradient norm is non-finite")
        optimizer.step()
        memory_tracker.update(device)

        batch_size = targets.shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["classification_loss"] += (
            float(classification_loss.detach().cpu()) * batch_size
        )
        totals["consistency_loss"] += (
            float(consistency_loss.detach().cpu()) * batch_size
        )
        totals["gradient_norm_before_clip"] += (
            float(gradient_norm.detach().cpu()) * batch_size
        )
        total_samples += batch_size
    if total_samples == 0:
        raise RuntimeError("Gate 11A training loader was empty")
    return {
        name: value / total_samples
        for name, value in totals.items()
    }


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    memory_tracker: MemoryTracker,
) -> PredictionResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[Tensor] = []
    all_probabilities: list[Tensor] = []
    all_sample_ids: list[str] = []
    for waveforms, sample_masks, targets, sample_ids in loader:
        if waveforms.shape[1] != 1:
            raise ValueError("Gate 11A validation must contain one waveform view")
        targets_device = targets.to(device)
        logits = _forward_batch(model, waveforms, sample_masks, device)[:, 0]
        loss = nn.functional.cross_entropy(logits, targets_device)
        probabilities = torch.softmax(logits, dim=-1)
        if not bool(torch.isfinite(loss).detach().cpu()) or not bool(
            torch.isfinite(probabilities).all().detach().cpu()
        ):
            raise FloatingPointError("Gate 11A validation became non-finite")
        total_loss += float(loss.cpu()) * targets.shape[0]
        total_samples += targets.shape[0]
        all_targets.append(targets)
        all_probabilities.append(probabilities.cpu())
        all_sample_ids.extend(str(value) for value in sample_ids)
        memory_tracker.update(device)
    if total_samples == 0:
        raise RuntimeError("Gate 11A validation loader was empty")
    targets_array = torch.cat(all_targets).numpy()
    probability_array = torch.cat(all_probabilities).numpy()
    predictions = (probability_array[:, 1] >= 0.5).astype(np.int64)
    metrics = respiratory_metrics(targets_array, predictions, CLASS_NAMES)
    metrics["auroc"] = float(
        roc_auc_score(targets_array, probability_array[:, 1])
    )
    metrics["nll"] = total_loss / total_samples
    return PredictionResult(
        loss=total_loss / total_samples,
        metrics=metrics,
        targets=targets_array,
        probabilities=probability_array,
        sample_ids=all_sample_ids,
    )


def _prediction_frame(
    result: PredictionResult,
    rows: pd.DataFrame,
    *,
    domain: str,
) -> pd.DataFrame:
    assert_selected_rows(
        rows,
        expected_role=VALIDATION_ROLE,
        expected_domain=domain,
    )
    metadata_columns = [
        "sample_id",
        "dataset",
        "patient_id",
        "protocol_role",
        "locked",
        "binary_label_id",
        "fine_label_name",
        "event_duration_seconds",
    ]
    metadata = rows[metadata_columns].copy()
    predictions = pd.DataFrame(
        {
            "sample_id": result.sample_ids,
            "target": result.targets,
            "prediction": (result.probabilities[:, 1] >= 0.5).astype(int),
            "probability_0": result.probabilities[:, 0],
            "probability_1": result.probabilities[:, 1],
        }
    )
    if predictions["sample_id"].duplicated().any():
        raise ValueError("Gate 11A validation predictions contain duplicate IDs")
    merged = metadata.merge(
        predictions,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(metadata) or len(merged) != len(predictions):
        raise ValueError("Gate 11A validation predictions do not match selected rows")
    if not np.array_equal(
        merged["binary_label_id"].to_numpy(dtype=int),
        merged["target"].to_numpy(dtype=int),
    ):
        raise ValueError("Gate 11A sample/label mismatch in prediction output")
    assert_selected_rows(
        merged,
        expected_role=VALIDATION_ROLE,
        expected_domain=domain,
    )
    return merged


def _validate_freeze(
    root: Path,
    freeze_path: Path,
    *,
    seed: int,
    manifest_path: Path,
    data_config_path: Path,
    model_config_path: Path,
    experiment_config_path: Path,
    checkpoint_path: Path,
    beats_source_dir: Path,
) -> tuple[dict[str, Any], str, dict[str, str]]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        int(freeze["schema_version"]) != 1
        or str(freeze["gate"]) != "11A"
        or str(freeze["status"]) != "sealed_before_formal_validation"
    ):
        raise ValueError("Gate 11A freeze file is not sealed")
    if list(freeze["allowed_seeds"]) != list(ALLOWED_SEEDS):
        raise ValueError("Gate 11A freeze has unexpected seeds")
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Gate 11A seed is not frozen: {seed}")
    boundary = freeze["development_data_boundary"]
    if (
        boundary["train_role"] != TRAIN_ROLE
        or boundary["validation_role"] != VALIDATION_ROLE
        or boundary["calibration_allowed"]
        or boundary["locked_allowed"]
    ):
        raise ValueError("Gate 11A freeze has an unsafe data boundary")
    if freeze["single_seed_thresholds"] != SINGLE_SEED_THRESHOLDS:
        raise ValueError("Gate 11A single-seed thresholds changed")
    if freeze["three_seed_thresholds"] != THREE_SEED_THRESHOLDS:
        raise ValueError("Gate 11A three-seed thresholds changed")
    current_hashes = gate11a_hashes(
        root,
        manifest_path=manifest_path,
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        experiment_config_path=experiment_config_path,
        checkpoint_path=checkpoint_path,
        beats_source_dir=beats_source_dir,
    )
    if freeze["hashes"] != current_hashes:
        differing = {
            name: {
                "frozen": freeze["hashes"].get(name),
                "current": current_hashes.get(name),
            }
            for name in set(freeze["hashes"]).union(current_hashes)
            if freeze["hashes"].get(name) != current_hashes.get(name)
        }
        raise ValueError(f"Gate 11A frozen hashes changed: {differing}")
    return freeze, sha256_file(freeze_path), current_hashes


def _sampler_state(
    sampler: DomainClassEventBatchSampler,
    *,
    completed_epoch: int,
) -> dict[str, Any]:
    return {
        "class": type(sampler).__name__,
        "seed": sampler.seed,
        "completed_epoch": completed_epoch,
        "batch_size": sampler.batch_size,
        "samples_per_epoch": sampler.samples_per_epoch,
        "num_batches": sampler.num_batches,
        "per_stratum": sampler.per_stratum,
    }


def _validate_sampler_state(
    saved: dict[str, Any],
    sampler: DomainClassEventBatchSampler,
    *,
    next_epoch: int,
) -> None:
    expected = _sampler_state(
        sampler,
        completed_epoch=next_epoch - 1,
    )
    if saved != expected:
        raise ValueError(
            f"Gate 11A sampler recovery state mismatch: "
            f"saved={saved}, expected={expected}"
        )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = _resolve(root, args.manifest).resolve()
    data_config_path = _resolve(root, args.data_config).resolve()
    model_config_path = _resolve(root, args.model_config).resolve()
    experiment_config_path = _resolve(root, args.experiment_config).resolve()
    freeze_path = _resolve(root, args.freeze_file).resolve()

    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(
        experiment_config_path.read_text(encoding="utf-8")
    )
    seed = int(args.seed if args.seed is not None else experiment["seed"])
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Unexpected Gate 11A seed: {seed}")
    expected_run_name = f"gate11a_exactmask_fullft_seed{seed}"
    if args.run_name != expected_run_name:
        raise ValueError(
            f"Gate 11A run name must be {expected_run_name}"
        )
    if list(experiment["allowed_seeds"]) != list(ALLOWED_SEEDS):
        raise ValueError("Experiment config does not contain the frozen seeds")
    if str(experiment["sampling"]["mode"]) != "domain_class_event":
        raise ValueError("Gate 11A is frozen to domain_class_event sampling")
    if args.device != "mps":
        raise ValueError("Formal Gate 11A training is frozen to MPS")
    seed_everything(seed)
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")

    checkpoint_path = (root / str(model_config["checkpoint"])).resolve()
    beats_source_dir = (root / str(model_config["source_dir"])).resolve()
    freeze, freeze_sha256, frozen_hashes = _validate_freeze(
        root,
        freeze_path,
        seed=seed,
        manifest_path=manifest_path,
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        experiment_config_path=experiment_config_path,
        checkpoint_path=checkpoint_path,
        beats_source_dir=beats_source_dir,
    )

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    protocol_audit = assert_gate11a_protocol(
        manifest,
        train_role=str(experiment["train_value"]),
        validation_role=str(experiment["validation_value"]),
    )
    feature_config = feature_config_from_yaml(data_config_path)
    label_column = str(experiment["label_column"])
    consistency = experiment["consistency"]
    num_views = 2 if bool(consistency["enabled"]) else 1
    train_dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column="protocol_role",
        split_value=TRAIN_ROLE,
        feature_config=feature_config,
        training=True,
        num_views=num_views,
        augmentation=_augmentation(experiment),
        label_column=label_column,
        return_waveform=True,
        waveform_only=True,
    )
    assert_selected_rows(train_dataset.rows, expected_role=TRAIN_ROLE)
    train_label_lookup = dict(
        zip(
            train_dataset.rows["sample_id"].astype(str),
            train_dataset.rows[label_column].astype(int),
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
            label_column=label_column,
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

    batch_size = int(experiment["batch_size"])
    sampler = DomainClassEventBatchSampler(
        train_dataset.rows,
        batch_size=batch_size,
        samples_per_epoch=int(experiment["samples_per_epoch"]),
        seed=seed,
        class_column=label_column,
    )
    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(seed + 11_000_000)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=int(experiment["num_workers"]),
        pin_memory=False,
        generator=train_loader_generator,
    )
    validation_loaders = {
        domain: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(experiment["num_workers"]),
            pin_memory=False,
        )
        for domain, dataset in validation_datasets.items()
    }

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing BEATs checkpoint: {checkpoint_path}")
    model = load_beats_transfer(checkpoint_path, beats_source_dir)
    backbone_parameters, classifier_parameters = _classifier_parameters(model)
    model = model.to(device)
    optimizer_config = experiment["optimizer"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(optimizer_config["backbone_learning_rate"]),
                "group_name": "backbone",
            },
            {
                "params": classifier_parameters,
                "lr": float(optimizer_config["head_learning_rate"]),
                "group_name": "head",
            },
        ],
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

    run_dir = root / "runs" / args.run_name
    resume_path = (
        _resolve(root, args.resume_state).resolve()
        if args.resume_state is not None
        else None
    )
    if resume_path is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        if resume_path.parent != run_dir.resolve():
            raise ValueError("Recovery state must belong to the requested run")
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Missing recovery run directory: {run_dir}")

    training_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if training_parameters != total_parameters:
        raise ValueError("Gate 11A full finetuning did not expose every parameter")
    configuration = {
        "gate": "11A",
        "formal_status": "sealed_exactmask_fullft",
        "seed": seed,
        "device": str(device),
        "run_name": args.run_name,
        "model": model_config,
        "experiment": experiment,
        "freeze_file": str(freeze_path.relative_to(root)),
        "freeze_sha256": freeze_sha256,
        "frozen_hashes": frozen_hashes,
        "protocol_audit": protocol_audit,
        "pretrained_load_audit": model.pretrained_load_audit,
        "train_samples": len(train_dataset),
        "validation_samples_by_domain": {
            domain: len(dataset)
            for domain, dataset in validation_datasets.items()
        },
        "batches_per_epoch": len(sampler),
        "sampling_mode": "domain_class_event",
        "samples_per_batch_stratum": sampler.per_stratum,
        "training_parameters": training_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": training_parameters / total_parameters,
        "calibration_accessed": False,
        "locked_tests_accessed": False,
    }
    configuration_path = run_dir / "configuration.json"
    if resume_path is None:
        _atomic_json(configuration, configuration_path)
    else:
        saved_configuration = json.loads(
            configuration_path.read_text(encoding="utf-8")
        )
        if saved_configuration != configuration:
            raise ValueError("Gate 11A recovery configuration changed")
    configuration_sha256 = sha256_file(configuration_path)

    history: list[dict[str, float]] = []
    best_worst = -1.0
    best_mean = -1.0
    best_epoch = 0
    best_validation_metrics: dict[str, dict[str, Any]] | None = None
    stale_epochs = 0
    start_epoch = 0
    finite_audit_passed = True
    resume_training_complete = False
    memory_tracker = MemoryTracker()
    memory_tracker.update(device)
    start_time = time.monotonic()

    if resume_path is not None:
        recovery = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        if int(recovery["schema_version"]) != 1:
            raise ValueError("Unsupported Gate 11A recovery schema")
        if (
            str(recovery["gate"]) != "11A"
            or int(recovery["seed"]) != seed
            or recovery["freeze_sha256"] != freeze_sha256
            or recovery["configuration_sha256"] != configuration_sha256
            or recovery["frozen_hashes"] != frozen_hashes
        ):
            raise ValueError("Gate 11A recovery identity mismatch")
        model.load_state_dict(recovery["model_state_dict"], strict=True)
        optimizer.load_state_dict(recovery["optimizer_state_dict"])
        scheduler.load_state_dict(recovery["scheduler_state_dict"])
        history = list(recovery["history"])
        best_worst = float(recovery["best_worst"])
        best_mean = float(recovery["best_mean"])
        best_epoch = int(recovery["best_epoch"])
        best_validation_metrics = recovery["best_validation_metrics"]
        stale_epochs = int(recovery["stale_epochs"])
        start_epoch = int(recovery["next_epoch"])
        finite_audit_passed = bool(recovery["finite_audit_passed"])
        resume_training_complete = bool(recovery["training_complete"])
        if start_epoch != len(history) or not 0 <= start_epoch <= max_epochs:
            raise ValueError("Invalid Gate 11A next epoch in recovery state")
        expected_complete = stale_epochs >= patience or start_epoch >= max_epochs
        if resume_training_complete != expected_complete:
            raise ValueError("Gate 11A recovery completion state is inconsistent")
        _validate_sampler_state(
            recovery["sampler_state"],
            sampler,
            next_epoch=start_epoch,
        )
        train_loader_generator.set_state(
            recovery["train_loader_generator_state"]
        )
        restore_rng_state(recovery["rng_state"], device)
        saved_memory = recovery["memory_tracker"]
        memory_tracker = MemoryTracker(
            peak_process_rss_bytes=int(saved_memory["peak_process_rss_bytes"]),
            peak_mps_current_allocated_bytes=int(
                saved_memory["peak_mps_current_allocated_bytes"]
            ),
            peak_mps_driver_allocated_bytes=int(
                saved_memory["peak_mps_driver_allocated_bytes"]
            ),
        )
        print(
            json.dumps(
                {
                    "recovered_from": str(resume_path),
                    "next_epoch": start_epoch + 1,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                }
            ),
            flush=True,
        )

    selection_tolerance = float(
        experiment["checkpoint_selection"]["tolerance"]
    )
    epoch_limit = start_epoch if resume_training_complete else max_epochs
    for epoch in range(start_epoch, epoch_limit):
        epoch_started = time.monotonic()
        sampler.set_epoch(epoch)
        js_weight = consistency_weight(
            epoch,
            maximum=float(consistency["max_weight"]),
            warmup_epochs=int(consistency["warmup_epochs"]),
        )
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            consistency_strength=js_weight,
            memory_tracker=memory_tracker,
            label_lookup=train_label_lookup,
        )
        validation_results = {
            domain: _evaluate(
                model,
                loader,
                device,
                memory_tracker=memory_tracker,
            )
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
            "epoch_seconds": time.monotonic() - epoch_started,
            "backbone_learning_rate": learning_rates["backbone"],
            "head_learning_rate": learning_rates["head"],
            "consistency_weight": js_weight,
            **{
                f"train_{name}": value
                for name, value in train_metrics.items()
            },
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
            finite_audit_passed = False
            raise FloatingPointError("Gate 11A epoch history became non-finite")
        history.append(row)
        print(json.dumps(row), flush=True)

        improved = worst > best_worst + selection_tolerance or (
            abs(worst - best_worst) <= selection_tolerance
            and mean > best_mean + selection_tolerance
        )
        if improved:
            best_worst = worst
            best_mean = mean
            best_epoch = epoch + 1
            best_validation_metrics = {
                domain: result.metrics
                for domain, result in validation_results.items()
            }
            stale_epochs = 0
            atomic_torch_save(
                {
                    "schema_version": 1,
                    "gate": "11A",
                    "seed": seed,
                    "model_state_dict": _model_state_cpu(model),
                    "model_config": model_config,
                    "data_config": yaml.safe_load(
                        data_config_path.read_text(encoding="utf-8")
                    ),
                    "experiment_config": experiment,
                    "class_names": CLASS_NAMES,
                    "freeze_sha256": freeze_sha256,
                    "configuration_sha256": configuration_sha256,
                    "selection": {
                        "epoch": best_epoch,
                        "minimum_domain_average_score": best_worst,
                        "mean_domain_average_score": best_mean,
                    },
                    "calibration_accessed": False,
                    "locked_tests_accessed": False,
                },
                run_dir / "best.pt",
            )
            for domain, result in validation_results.items():
                prediction_frame = _prediction_frame(
                    result,
                    validation_datasets[domain].rows,
                    domain=domain,
                )
                _atomic_csv(
                    prediction_frame,
                    run_dir
                    / f"best_validation_predictions_{domain}.csv",
                )
        else:
            stale_epochs += 1

        _atomic_csv(pd.DataFrame(history), run_dir / "history.csv")
        scheduler.step()
        memory_tracker.update(device)
        training_complete = (
            stale_epochs >= patience or epoch + 1 >= max_epochs
        )
        atomic_torch_save(
            {
                "schema_version": 1,
                "gate": "11A",
                "seed": seed,
                "freeze_sha256": freeze_sha256,
                "configuration_sha256": configuration_sha256,
                "frozen_hashes": frozen_hashes,
                "next_epoch": epoch + 1,
                "history": history,
                "best_worst": best_worst,
                "best_mean": best_mean,
                "best_epoch": best_epoch,
                "best_validation_metrics": best_validation_metrics,
                "stale_epochs": stale_epochs,
                "finite_audit_passed": finite_audit_passed,
                "training_complete": training_complete,
                "model_state_dict": _model_state_cpu(model),
                "optimizer_state_dict": _cpu_clone(optimizer.state_dict()),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": capture_rng_state(device),
                "sampler_state": _sampler_state(
                    sampler,
                    completed_epoch=epoch,
                ),
                "train_loader_generator_state": (
                    train_loader_generator.get_state()
                ),
                "memory_tracker": memory_tracker.to_dict(),
                "calibration_accessed": False,
                "locked_tests_accessed": False,
            },
            run_dir / "resume_state.pt",
        )
        if training_complete:
            break

    if best_validation_metrics is None:
        raise RuntimeError("Gate 11A training did not produce a checkpoint")
    elapsed_seconds = time.monotonic() - start_time
    memory_tracker.update(device)
    best_checkpoint_path = run_dir / "best.pt"
    resume_state_path = run_dir / "resume_state.pt"
    summary = {
        "gate": "11A",
        "candidate": "beats_exactmask_fullft",
        "seed": seed,
        "selection_metric": "minimum_domain_average_score_then_mean",
        "best_epoch": best_epoch,
        "best_worst_domain_average_score": best_worst,
        "best_mean_domain_average_score": best_mean,
        "epochs_completed": len(history),
        "training_parameters": training_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": training_parameters / total_parameters,
        "best_validation_metrics": best_validation_metrics,
        "finite_audit_passed": finite_audit_passed,
        "data_boundary_audit_passed": True,
        "exact_mask_audit_passed": True,
        "protocol_audit": protocol_audit,
        "elapsed_seconds_this_invocation": elapsed_seconds,
        "training_seconds_total": float(
            sum(row["epoch_seconds"] for row in history)
        ),
        "memory": memory_tracker.to_dict(),
        "disk": {
            "best_checkpoint_bytes": best_checkpoint_path.stat().st_size,
            "resume_state_bytes": resume_state_path.stat().st_size,
        },
        "freeze_sha256": freeze_sha256,
        "configuration_sha256": configuration_sha256,
        "calibration_accessed": False,
        "locked_tests_accessed": False,
    }
    decision = single_seed_decision(summary)
    _atomic_json(summary, run_dir / "summary.json")
    _atomic_json(
        {
            "gate": "11A",
            "stage": "single_seed",
            "freeze_file": str(freeze_path.relative_to(root)),
            "freeze_sha256": freeze_sha256,
            "decision": decision,
            "recommendation": (
                "continue_remaining_seeds"
                if decision["passed"]
                else "stop_route_unless_reproducible_implementation_error"
            ),
            "calibration_accessed": False,
            "locked_tests_accessed": False,
        },
        run_dir / "decision.json",
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "decision": decision,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
