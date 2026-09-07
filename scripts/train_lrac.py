#!/usr/bin/env python3
"""Train LRAC-Net or its same-content temporally reversed control."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import (
    AugmentationConfig,
    FeatureNormalization,
    ICBHICycleDataset,
)
from respiratory_sound.metrics import respiratory_metrics
from respiratory_sound.models.lrac import LRACNet
from respiratory_sound.runtime import select_device
from respiratory_sound.training import (
    class_weights,
    consistency_weight,
    jensen_shannon_consistency,
    seed_everything,
    warmup_cosine_scheduler,
)


@dataclass(frozen=True)
class LRACResult:
    loss: float
    metrics: dict[str, object]
    spectrum_metrics: dict[str, object]
    waveform_metrics: dict[str, object]
    alignment_metrics: dict[str, float]
    targets: np.ndarray
    probabilities: np.ndarray
    spectrum_probabilities: np.ndarray
    waveform_probabilities: np.ndarray
    residual_margins: np.ndarray
    aligned_reliability_means: np.ndarray
    reversed_reliability_means: np.ndarray
    classification_reliability_means: np.ndarray
    sample_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/sprsound2022_events.csv"),
    )
    parser.add_argument(
        "--feature-stats",
        type=Path,
        default=Path(
            "data/manifests/sprsound2022_binary_zero_pad_feature_stats.json"
        ),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/sprsound2022_binary_zero_pad.yaml"),
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def model_from_config(config: dict[str, object]) -> LRACNet:
    spectrum = config["spectrum"]
    waveform = config["waveform"]
    if not isinstance(spectrum, dict) or not isinstance(waveform, dict):
        raise ValueError("spectrum and waveform model sections must be mappings")
    return LRACNet(
        alignment_mode=str(config["alignment_mode"]),
        spectrum_channels=tuple(int(value) for value in spectrum["channels"]),
        spectrum_depths=tuple(int(value) for value in spectrum["depths"]),
        spectrum_temporal_dilations=tuple(
            tuple(int(value) for value in stage)
            for stage in spectrum["temporal_dilations"]
        ),
        spectrum_expansion_ratio=int(spectrum["expansion_ratio"]),
        waveform_channels=tuple(int(value) for value in waveform["channels"]),
        projection_dimension=int(config["projection_dimension"]),
        interaction_hidden=int(config["interaction_hidden"]),
        max_residual_margin=float(config["max_residual_margin"]),
        num_classes=int(config["num_classes"]),
    )


def _disabled_augmentation() -> AugmentationConfig:
    return AugmentationConfig(
        gain_min=1.0,
        gain_max=1.0,
        noise_probability=0.0,
        shift_probability=0.0,
        frequency_mask_bins=0,
        time_mask_frames=0,
    )


def local_alignment_loss(
    aligned_logits: Tensor,
    reversed_logits: Tensor,
    valid_masks: Tensor,
) -> Tensor:
    aligned_valid = aligned_logits[valid_masks]
    reversed_valid = reversed_logits[valid_masks]
    logits = torch.cat((aligned_valid, reversed_valid))
    targets = torch.cat(
        (
            torch.ones_like(aligned_valid),
            torch.zeros_like(reversed_valid),
        )
    )
    return nn.functional.binary_cross_entropy_with_logits(logits, targets)


def train_one_epoch(
    model: LRACNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_function: nn.Module,
    loss_weights: dict[str, float],
    consistency_strength: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "final": 0.0,
        "spectrum": 0.0,
        "waveform": 0.0,
        "alignment": 0.0,
        "consistency": 0.0,
    }
    total_samples = 0
    for (
        views,
        frame_masks,
        waveforms,
        sample_masks,
        targets,
        _,
    ) in loader:
        batch_size, num_views, channels, mels, frames = views.shape
        samples = waveforms.shape[-1]
        inputs = views.reshape(batch_size * num_views, channels, mels, frames).to(
            device
        )
        input_masks = frame_masks.reshape(batch_size * num_views, frames).to(device)
        waveform_inputs = waveforms.reshape(
            batch_size * num_views,
            1,
            samples,
        ).to(device)
        waveform_masks = sample_masks.reshape(
            batch_size * num_views,
            samples,
        ).to(device)
        targets = targets.to(device)
        repeated_targets = targets.unsqueeze(1).expand(-1, num_views).reshape(-1)

        optimizer.zero_grad(set_to_none=True)
        (
            logits,
            spectrum_logits,
            waveform_logits,
            _,
            aligned_logits,
            reversed_logits,
            local_masks,
            _,
        ) = model(
            inputs,
            waveform_inputs,
            frame_masks=input_masks,
            sample_masks=waveform_masks,
            return_aux=True,
        )
        viewed_logits = logits.reshape(batch_size, num_views, -1)
        final_loss = loss_function(logits, repeated_targets)
        spectrum_loss = loss_function(spectrum_logits, repeated_targets)
        waveform_loss = loss_function(waveform_logits, repeated_targets)
        alignment_loss = local_alignment_loss(
            aligned_logits,
            reversed_logits,
            local_masks,
        )
        consistency_loss = (
            jensen_shannon_consistency(viewed_logits)
            if num_views > 1
            else logits.new_zeros(())
        )
        loss = (
            loss_weights["final"] * final_loss
            + loss_weights["spectrum"] * spectrum_loss
            + loss_weights["waveform"] * waveform_loss
            + loss_weights["alignment"] * alignment_loss
            + consistency_strength * consistency_loss
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        for name, value in (
            ("loss", loss),
            ("final", final_loss),
            ("spectrum", spectrum_loss),
            ("waveform", waveform_loss),
            ("alignment", alignment_loss),
            ("consistency", consistency_loss),
        ):
            totals[name] += float(value.detach().cpu()) * batch_size
        total_samples += batch_size
    return {name: value / total_samples for name, value in totals.items()}


def alignment_metrics(
    aligned_logits: np.ndarray,
    reversed_logits: np.ndarray,
) -> dict[str, float]:
    logits = np.concatenate((aligned_logits, reversed_logits))
    targets = np.concatenate(
        (
            np.ones_like(aligned_logits, dtype=int),
            np.zeros_like(reversed_logits, dtype=int),
        )
    )
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return {
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "accuracy": float(np.mean((probabilities >= 0.5) == targets)),
        "aligned_reliability_mean": float(
            np.mean(probabilities[: len(aligned_logits)])
        ),
        "reversed_reliability_mean": float(
            np.mean(probabilities[len(aligned_logits) :])
        ),
    }


@torch.inference_mode()
def evaluate(
    model: LRACNet,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
    class_names: tuple[str, ...],
) -> LRACResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[Tensor] = []
    all_probabilities: list[Tensor] = []
    all_spectrum_probabilities: list[Tensor] = []
    all_waveform_probabilities: list[Tensor] = []
    all_residuals: list[Tensor] = []
    all_aligned_logits: list[Tensor] = []
    all_reversed_logits: list[Tensor] = []
    all_aligned_means: list[Tensor] = []
    all_reversed_means: list[Tensor] = []
    all_classification_means: list[Tensor] = []
    all_sample_ids: list[str] = []
    for (
        views,
        frame_masks,
        waveforms,
        sample_masks,
        targets,
        sample_ids,
    ) in loader:
        if views.shape[1] != 1 or waveforms.shape[1] != 1:
            raise ValueError("Evaluation requires exactly one aligned view")
        (
            logits,
            spectrum_logits,
            waveform_logits,
            residual,
            aligned_logits,
            reversed_logits,
            local_masks,
            classification_gates,
        ) = model(
            views[:, 0].to(device),
            waveforms[:, 0].to(device),
            frame_masks=frame_masks[:, 0].to(device),
            sample_masks=sample_masks[:, 0].to(device),
            return_aux=True,
        )
        targets_device = targets.to(device)
        loss = loss_function(logits, targets_device)
        total_loss += float(loss.cpu()) * targets.shape[0]
        total_samples += targets.shape[0]
        valid_weights = local_masks.to(dtype=logits.dtype)
        valid_counts = valid_weights.sum(dim=1).clamp_min(1.0)
        all_targets.append(targets.cpu())
        all_probabilities.append(torch.softmax(logits, dim=-1).cpu())
        all_spectrum_probabilities.append(
            torch.softmax(spectrum_logits, dim=-1).cpu()
        )
        all_waveform_probabilities.append(
            torch.softmax(waveform_logits, dim=-1).cpu()
        )
        all_residuals.append(residual.cpu())
        all_aligned_logits.append(aligned_logits[local_masks].cpu())
        all_reversed_logits.append(reversed_logits[local_masks].cpu())
        all_aligned_means.append(
            (
                torch.sigmoid(aligned_logits) * valid_weights
            ).sum(dim=1).div(valid_counts).cpu()
        )
        all_reversed_means.append(
            (
                torch.sigmoid(reversed_logits) * valid_weights
            ).sum(dim=1).div(valid_counts).cpu()
        )
        all_classification_means.append(
            (classification_gates * valid_weights)
            .sum(dim=1)
            .div(valid_counts)
            .cpu()
        )
        all_sample_ids.extend(sample_ids)

    targets_array = torch.cat(all_targets).numpy()
    probabilities = torch.cat(all_probabilities).numpy()
    spectrum_probabilities = torch.cat(all_spectrum_probabilities).numpy()
    waveform_probabilities = torch.cat(all_waveform_probabilities).numpy()
    aligned_array = torch.cat(all_aligned_logits).numpy()
    reversed_array = torch.cat(all_reversed_logits).numpy()
    return LRACResult(
        loss=total_loss / total_samples,
        metrics=respiratory_metrics(
            targets_array,
            probabilities.argmax(axis=1),
            class_names,
        ),
        spectrum_metrics=respiratory_metrics(
            targets_array,
            spectrum_probabilities.argmax(axis=1),
            class_names,
        ),
        waveform_metrics=respiratory_metrics(
            targets_array,
            waveform_probabilities.argmax(axis=1),
            class_names,
        ),
        alignment_metrics=alignment_metrics(aligned_array, reversed_array),
        targets=targets_array,
        probabilities=probabilities,
        spectrum_probabilities=spectrum_probabilities,
        waveform_probabilities=waveform_probabilities,
        residual_margins=torch.cat(all_residuals).numpy(),
        aligned_reliability_means=torch.cat(all_aligned_means).numpy(),
        reversed_reliability_means=torch.cat(all_reversed_means).numpy(),
        classification_reliability_means=torch.cat(
            all_classification_means
        ).numpy(),
        sample_ids=all_sample_ids,
    )


def _scalar_metrics(prefix: str, metrics: dict[str, object]) -> dict[str, float]:
    return {
        f"{prefix}_{name}": float(value)
        for name, value in metrics.items()
        if isinstance(value, float)
    }


def residual_summary(result: LRACResult, maximum: float) -> dict[str, float]:
    absolute = np.abs(result.residual_margins)
    return {
        "mean": float(np.mean(result.residual_margins)),
        "mean_absolute": float(np.mean(absolute)),
        "standard_deviation": float(np.std(result.residual_margins)),
        "maximum_absolute": float(np.max(absolute)),
        "saturation_fraction": float(np.mean(absolute >= 0.95 * maximum)),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    stats_path = (root / args.feature_stats).resolve()
    data_config_path = (root / args.data_config).resolve()
    model_config_path = (root / args.model_config).resolve()
    experiment_config_path = (root / args.experiment_config).resolve()
    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    data_config = yaml.safe_load(data_config_path.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else experiment["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")

    feature_config = feature_config_from_yaml(data_config_path)
    normalization = FeatureNormalization.from_json(stats_path)
    split_column = str(experiment.get("split_column", "development_split"))
    label_column = str(experiment.get("label_column", "binary_label_id"))
    class_names = tuple(str(name) for name in data_config["classes"])
    selection_metric = str(experiment.get("selection_metric", "score"))
    consistency = experiment.get("consistency", {})
    num_views = 2 if consistency.get("enabled", False) else 1
    augmentation_enabled = bool(experiment.get("augmentation", {}).get("enabled", True))
    augmentation = None if augmentation_enabled else _disabled_augmentation()
    common_dataset = {
        "manifest_path": manifest_path,
        "project_root": root,
        "split_column": split_column,
        "feature_config": feature_config,
        "normalization": normalization,
        "label_column": label_column,
        "return_waveform": True,
    }
    train_dataset = ICBHICycleDataset(
        **common_dataset,
        split_value=str(experiment.get("train_value", "train")),
        training=augmentation_enabled,
        num_views=num_views,
        augmentation=augmentation,
        sample_limit=experiment.get("sample_limit_train"),
    )
    validation_dataset = ICBHICycleDataset(
        **common_dataset,
        split_value=str(experiment.get("validation_value", "validation")),
        training=False,
        num_views=1,
        sample_limit=experiment.get("sample_limit_validation"),
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": int(experiment["batch_size"]),
        "num_workers": int(experiment.get("num_workers", 0)),
        "pin_memory": False,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )

    model = model_from_config(model_config).to(device)
    weights = class_weights(
        train_dataset.rows[label_column],
        num_classes=int(model_config["num_classes"]),
        power=float(experiment["loss"]["class_weight_power"]),
    ).to(device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    loss_weights = {
        name: float(experiment["loss"][f"{name}_weight"])
        for name in ("final", "spectrum", "waveform", "alignment")
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(experiment["optimizer"]["learning_rate"]),
        weight_decay=float(experiment["optimizer"]["weight_decay"]),
    )
    max_epochs = int(experiment["schedule"]["max_epochs"])
    scheduler = warmup_cosine_scheduler(
        optimizer,
        warmup_epochs=int(experiment["schedule"]["warmup_epochs"]),
        max_epochs=max_epochs,
    )
    patience = int(experiment["schedule"]["early_stopping_patience"])

    run_dir = root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    parameter_counts = {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "spectrum": sum(parameter.numel() for parameter in model.spectrum.parameters()),
        "waveform": sum(parameter.numel() for parameter in model.waveform.parameters()),
        "local_alignment": sum(
            parameter.numel()
            for module in (
                model.spectrum_projection,
                model.waveform_projection,
                model.reliability,
                model.correction,
            )
            for parameter in module.parameters()
        ),
    }
    snapshot = {
        "seed": seed,
        "device": str(device),
        "model": model_config,
        "experiment": experiment,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "class_weights": weights.detach().cpu().tolist(),
        "class_names": class_names,
        "label_column": label_column,
        "selection_metric": selection_metric,
        "parameter_counts": parameter_counts,
    }
    (run_dir / "configuration.json").write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8",
    )

    history: list[dict[str, float | int]] = []
    best_score = -1.0
    best_result: LRACResult | None = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(max_epochs):
        strength = consistency_weight(
            epoch,
            maximum=float(consistency.get("max_weight", 0.0)),
            warmup_epochs=int(consistency.get("warmup_epochs", 0)),
        )
        train_losses = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_function,
            loss_weights,
            consistency_strength=strength,
        )
        validation = evaluate(
            model,
            validation_loader,
            device,
            loss_function,
            class_names,
        )
        residual = residual_summary(
            validation,
            maximum=float(model_config["max_residual_margin"]),
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "consistency_weight": strength,
            **{f"train_{name}_loss": value for name, value in train_losses.items()},
            "validation_loss": validation.loss,
            **_scalar_metrics("validation", validation.metrics),
            **_scalar_metrics("validation_spectrum", validation.spectrum_metrics),
            **_scalar_metrics("validation_waveform", validation.waveform_metrics),
            **{
                f"validation_alignment_{name}": value
                for name, value in validation.alignment_metrics.items()
            },
            **{
                f"validation_residual_{name}": value
                for name, value in residual.items()
            },
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(json.dumps(row))
        score = float(validation.metrics[selection_metric])
        if score > best_score:
            best_score = score
            best_result = validation
            best_epoch = epoch + 1
            stale_epochs = 0
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "feature_config": feature_config.__dict__,
                    "normalization_path": str(stats_path.relative_to(root)),
                    "class_names": class_names,
                    "label_column": label_column,
                    "selection_metric": selection_metric,
                    "validation_metrics": validation.metrics,
                    "validation_spectrum_metrics": validation.spectrum_metrics,
                    "validation_waveform_metrics": validation.waveform_metrics,
                    "validation_alignment_metrics": validation.alignment_metrics,
                },
                run_dir / "best.pt",
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= patience:
            break

    if best_result is None:
        raise RuntimeError("Training did not produce a validation result")
    prediction_table = pd.DataFrame(
        {
            "sample_id": best_result.sample_ids,
            "patient_id": [
                sample_id.split("_", maxsplit=1)[0]
                for sample_id in best_result.sample_ids
            ],
            "target": best_result.targets,
            "prediction": best_result.probabilities.argmax(axis=1),
            "spectrum_prediction": best_result.spectrum_probabilities.argmax(axis=1),
            "waveform_prediction": best_result.waveform_probabilities.argmax(axis=1),
            "residual_margin": best_result.residual_margins,
            "aligned_reliability_mean": best_result.aligned_reliability_means,
            "reversed_reliability_mean": best_result.reversed_reliability_means,
            "classification_reliability_mean": (
                best_result.classification_reliability_means
            ),
            **{
                f"probability_{index}": best_result.probabilities[:, index]
                for index in range(best_result.probabilities.shape[1])
            },
            **{
                f"spectrum_probability_{index}": (
                    best_result.spectrum_probabilities[:, index]
                )
                for index in range(best_result.spectrum_probabilities.shape[1])
            },
            **{
                f"waveform_probability_{index}": (
                    best_result.waveform_probabilities[:, index]
                )
                for index in range(best_result.waveform_probabilities.shape[1])
            },
        }
    )
    prediction_table.to_csv(run_dir / "best_validation_predictions.csv", index=False)
    summary = {
        "selection_metric": selection_metric,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_validation_metrics": best_result.metrics,
        "best_spectrum_metrics": best_result.spectrum_metrics,
        "best_waveform_metrics": best_result.waveform_metrics,
        "best_alignment_metrics": best_result.alignment_metrics,
        "best_residual_summary": residual_summary(
            best_result,
            maximum=float(model_config["max_residual_margin"]),
        ),
        "epochs_completed": len(history),
        "parameter_counts": parameter_counts,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
