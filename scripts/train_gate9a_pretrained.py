#!/usr/bin/env python3
"""Fine-tune one frozen Gate 9A AudioSet-pretrained candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import AugmentationConfig, ICBHICycleDataset
from respiratory_sound.data.sampling import DomainClassEventBatchSampler
from respiratory_sound.metrics import respiratory_metrics
from respiratory_sound.models.pretrained_audio import (
    load_beats_transfer,
    load_panns_cnn6_transfer,
)
from respiratory_sound.runtime import select_device
from respiratory_sound.training import (
    consistency_weight,
    jensen_shannon_consistency,
    seed_everything,
    warmup_cosine_scheduler,
)

CLASS_NAMES = ("normal", "adventitious")
DEVELOPMENT_ROLES = {"train_fit", "validation_select", "calibration"}


@dataclass(frozen=True)
class PredictionResult:
    loss: float
    metrics: dict[str, Any]
    targets: np.ndarray
    probabilities: np.ndarray
    sample_ids: list[str]


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
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _augmentation(experiment: dict[str, Any]) -> AugmentationConfig:
    values = experiment.get("augmentation", {})
    return AugmentationConfig(
        gain_min=float(values.get("gain_min", 0.8)),
        gain_max=float(values.get("gain_max", 1.2)),
        noise_probability=float(values.get("noise_probability", 0.5)),
        noise_snr_min_db=float(values.get("noise_snr_min_db", 12.0)),
        noise_snr_max_db=float(values.get("noise_snr_max_db", 30.0)),
        shift_probability=float(values.get("shift_probability", 0.5)),
        max_shift_fraction=float(values.get("max_shift_fraction", 0.1)),
        frequency_mask_bins=0,
        time_mask_frames=0,
    )


def _load_model(
    root: Path,
    model_config: dict[str, Any],
) -> tuple[nn.Module, Path]:
    model_type = str(model_config["type"])
    checkpoint = (root / str(model_config["checkpoint"])).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {checkpoint}")
    if model_type == "panns_cnn6":
        return load_panns_cnn6_transfer(checkpoint), checkpoint
    if model_type == "beats":
        source_dir = (root / str(model_config["source_dir"])).resolve()
        return load_beats_transfer(checkpoint, source_dir), checkpoint
    raise ValueError(f"Unsupported Gate 9A model type: {model_type}")


def _classifier_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Module):
        raise ValueError("Gate 9A models must expose a classifier module")
    classifier_parameters = list(classifier.parameters())
    classifier_ids = {id(parameter) for parameter in classifier_parameters}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in classifier_ids
    ]
    if not backbone_parameters or not classifier_parameters:
        raise ValueError("Backbone and classifier parameter groups must both be non-empty")
    return backbone_parameters, classifier_parameters


def _forward_batch(
    model: nn.Module,
    waveforms: Tensor,
    sample_masks: Tensor,
    device: torch.device,
) -> Tensor:
    batch_size, num_views, channels, samples = waveforms.shape
    if channels != 1:
        raise ValueError("Gate 9A requires mono waveform views")
    flat_waveforms = waveforms.reshape(batch_size * num_views, samples).to(device)
    flat_masks = sample_masks.reshape(batch_size * num_views, samples).to(device)
    return model(flat_waveforms, flat_masks).reshape(batch_size, num_views, -1)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    consistency_strength: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "classification_loss": 0.0, "consistency_loss": 0.0}
    total_samples = 0
    for waveforms, sample_masks, targets, _ in loader:
        targets = targets.to(device)
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
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = targets.shape[0]
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["classification_loss"] += (
            float(classification_loss.detach().cpu()) * batch_size
        )
        totals["consistency_loss"] += (
            float(consistency_loss.detach().cpu()) * batch_size
        )
        total_samples += batch_size
    return {name: value / total_samples for name, value in totals.items()}


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> PredictionResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[Tensor] = []
    all_probabilities: list[Tensor] = []
    all_sample_ids: list[str] = []
    for waveforms, sample_masks, targets, sample_ids in loader:
        if waveforms.shape[1] != 1:
            raise ValueError("Validation must contain exactly one waveform view")
        targets_device = targets.to(device)
        logits = _forward_batch(model, waveforms, sample_masks, device)[:, 0]
        loss = nn.functional.cross_entropy(logits, targets_device)
        probabilities = torch.softmax(logits, dim=-1)
        total_loss += float(loss.cpu()) * targets.shape[0]
        total_samples += targets.shape[0]
        all_targets.append(targets)
        all_probabilities.append(probabilities.cpu())
        all_sample_ids.extend(str(value) for value in sample_ids)
    target_array = torch.cat(all_targets).numpy()
    probability_array = torch.cat(all_probabilities).numpy()
    predictions = probability_array.argmax(axis=1)
    return PredictionResult(
        loss=total_loss / total_samples,
        metrics=respiratory_metrics(target_array, predictions, CLASS_NAMES),
        targets=target_array,
        probabilities=probability_array,
        sample_ids=all_sample_ids,
    )


def _prediction_frame(result: PredictionResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": result.sample_ids,
            "target": result.targets,
            "prediction": result.probabilities.argmax(axis=1),
            "probability_0": result.probabilities[:, 0],
            "probability_1": result.probabilities[:, 1],
        }
    )


def _assert_development_protocol(
    manifest: pd.DataFrame,
    train_role: str,
    validation_role: str,
) -> None:
    if train_role not in DEVELOPMENT_ROLES or validation_role not in DEVELOPMENT_ROLES:
        raise ValueError("Gate 9A may only use named development roles")
    if train_role == validation_role:
        raise ValueError("Training and validation roles must differ")
    used = manifest["protocol_role"].isin({train_role, validation_role})
    if manifest.loc[used, "locked"].astype(bool).any():
        raise ValueError("A locked row was selected by the Gate 9A configuration")
    train = manifest.loc[manifest["protocol_role"].eq(train_role)]
    validation = manifest.loc[manifest["protocol_role"].eq(validation_role)]
    train_patients = set(zip(train["dataset"], train["patient_id"], strict=True))
    validation_patients = set(
        zip(validation["dataset"], validation["patient_id"], strict=True)
    )
    overlap = train_patients.intersection(validation_patients)
    if overlap:
        raise ValueError(f"Patient leakage between training and validation: {overlap}")


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
        raise ValueError(f"Unexpected Gate 9A domains: {domains}")
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
    sampler = DomainClassEventBatchSampler(
        train_dataset.rows,
        batch_size=batch_size,
        samples_per_epoch=int(experiment["samples_per_epoch"]),
        seed=seed,
        class_column=label_column,
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

    model, upstream_checkpoint = _load_model(root, model_config)
    backbone_parameters, classifier_parameters = _classifier_parameters(model)
    model = model.to(device)
    optimizer_config = experiment["optimizer"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(optimizer_config["backbone_learning_rate"]),
            },
            {
                "params": classifier_parameters,
                "lr": float(optimizer_config["head_learning_rate"]),
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
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        "gate": "9A",
        "seed": seed,
        "device": str(device),
        "model": model_config,
        "experiment": experiment,
        "manifest_sha256": _sha256(manifest_path),
        "upstream_checkpoint_sha256": _sha256(upstream_checkpoint),
        "pretrained_load_audit": getattr(model, "pretrained_load_audit", {}),
        "train_samples": len(train_dataset),
        "train_patients_by_domain": {
            domain: int(
                train_dataset.rows.loc[
                    train_dataset.rows["dataset"].eq(domain),
                    "patient_id",
                ].nunique()
            )
            for domain in domains
        },
        "validation_samples_by_domain": {
            domain: len(dataset)
            for domain, dataset in validation_datasets.items()
        },
        "batches_per_epoch": len(sampler),
        "samples_per_batch_stratum": sampler.per_stratum,
        "training_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "locked_test_accessed": False,
    }
    (run_dir / "configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8",
    )

    history: list[dict[str, float]] = []
    best_worst = -1.0
    best_mean = -1.0
    best_results: dict[str, PredictionResult] | None = None
    stale_epochs = 0
    for epoch in range(max_epochs):
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
        row = {
            "epoch": float(epoch + 1),
            "backbone_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
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
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = worst > best_worst + 1.0e-12 or (
            abs(worst - best_worst) <= 1.0e-12
            and mean > best_mean + 1.0e-12
        )
        if improved:
            best_worst = worst
            best_mean = mean
            best_results = validation_results
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "model_config": model_config,
                    "data_config": yaml.safe_load(
                        data_config_path.read_text(encoding="utf-8")
                    ),
                    "class_names": CLASS_NAMES,
                    "seed": seed,
                    "gate": "9A",
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
        if stale_epochs >= patience:
            break

    if best_results is None:
        raise RuntimeError("Gate 9A training did not produce a checkpoint")
    summary = {
        "gate": "9A",
        "candidate": str(model_config["type"]),
        "seed": seed,
        "selection_metric": "minimum_domain_average_score",
        "best_worst_domain_average_score": best_worst,
        "best_mean_domain_average_score": best_mean,
        "epochs_completed": len(history),
        "best_validation_metrics": {
            domain: result.metrics
            for domain, result in best_results.items()
        },
        "locked_test_accessed": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
