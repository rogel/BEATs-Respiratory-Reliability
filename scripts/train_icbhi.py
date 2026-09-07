#!/usr/bin/env python3
"""Train one model configuration on the leakage-controlled ICBHI development split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from torch import nn
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import (
    AugmentationConfig,
    FeatureNormalization,
    ICBHICycleDataset,
)
from respiratory_sound.data.sprsound import MORPHOLOGY_NAMES, morphology_target
from respiratory_sound.models.network import TFDCRNet
from respiratory_sound.models.reparameterization import switch_model_to_deploy
from respiratory_sound.runtime import select_device
from respiratory_sound.training import (
    attribute_pos_weights,
    branch_weight_summary,
    class_weights,
    consistency_weight,
    evaluate,
    evaluate_morphology,
    seed_everything,
    train_one_epoch,
    warmup_cosine_scheduler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/icbhi2017_cycles.csv"),
    )
    parser.add_argument(
        "--feature-stats",
        type=Path,
        default=Path("data/manifests/icbhi2017_feature_stats.json"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/icbhi2017.yaml"),
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--filter-column")
    parser.add_argument("--filter-value")
    return parser.parse_args()


def _model_from_config(config: dict[str, object]) -> TFDCRNet:
    return TFDCRNet(
        mode=str(config["mode"]),
        input_channels=int(config["input_channels"]),
        channels=tuple(int(value) for value in config["channels"]),
        depths=tuple(int(value) for value in config["depths"]),
        temporal_dilations=tuple(
            tuple(int(value) for value in stage) for stage in config["temporal_dilations"]
        ),
        expansion_ratio=int(config["expansion_ratio"]),
        num_classes=int(config["num_classes"]),
        include_joint=bool(config.get("include_joint", False)),
        branch_types=(
            tuple(str(branch) for branch in config["branch_types"])
            if "branch_types" in config
            else None
        ),
        pooling=str(config.get("pooling", "gap")),
        lse_temperature=float(config.get("lse_temperature", 1.0)),
        gate_conditioning=str(config.get("gate_conditioning", "none")),
        gate_stages=tuple(int(stage) for stage in config.get("gate_stages", ())),
        gate_reduction=int(config.get("gate_reduction", 4)),
        gate_max_delta=float(config.get("gate_max_delta", 0.5)),
        morphology_supervision=str(config.get("morphology_supervision", "none")),
        morphology_stages=tuple(
            int(stage) for stage in config.get("morphology_stages", ())
        ),
        morphology_branch_indices=tuple(
            int(index) for index in config.get("morphology_branch_indices", (1, 2))
        ),
        morphology_source=str(config.get("morphology_source", "branch_response")),
        morphology_routing=str(config.get("morphology_routing", "mean")),
        hierarchy_fusion=str(config.get("hierarchy_fusion", "none")),
        hierarchy_max_weight=float(config.get("hierarchy_max_weight", 1.0)),
        input_normalization=str(config.get("input_normalization", "none")),
        relaxed_frequency_weight=float(
            config.get("relaxed_frequency_weight", 0.5)
        ),
        normalization_eps=float(config.get("normalization_eps", 1.0e-5)),
    )


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
    label_column = str(experiment.get("label_column", "label_id"))
    class_names = tuple(str(name) for name in data_config["classes"])
    selection_metric = str(experiment.get("selection_metric", "average_score"))
    train_value = str(experiment.get("train_value", "train"))
    validation_value = str(experiment.get("validation_value", "validation"))
    sample_limit_train = experiment.get("sample_limit_train")
    sample_limit_validation = experiment.get("sample_limit_validation")
    consistency = experiment.get("consistency", {})
    num_views = 2 if consistency.get("enabled", False) else 1
    augmentation_config = experiment.get("augmentation", {})
    augmentation_enabled = bool(augmentation_config.get("enabled", True))
    augmentation = AugmentationConfig(
        **{
            key: value
            for key, value in augmentation_config.items()
            if key != "enabled"
        }
    )
    if not augmentation_enabled:
        augmentation = AugmentationConfig(
            gain_min=1.0,
            gain_max=1.0,
            noise_probability=0.0,
            shift_probability=0.0,
            frequency_mask_bins=0,
            time_mask_frames=0,
            frequency_response_probability=0.0,
            frequency_response_max_db=0.0,
        )
    row_filters = (
        {args.filter_column: args.filter_value}
        if args.filter_column is not None
        else None
    )

    train_dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column=split_column,
        split_value=train_value,
        feature_config=feature_config,
        normalization=normalization,
        training=augmentation_enabled,
        num_views=num_views,
        augmentation=augmentation,
        sample_limit=sample_limit_train,
        label_column=label_column,
        row_filters=row_filters,
    )
    validation_dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column=split_column,
        split_value=validation_value,
        feature_config=feature_config,
        normalization=normalization,
        training=False,
        num_views=1,
        sample_limit=sample_limit_validation,
        label_column=label_column,
        row_filters=row_filters,
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

    model = _model_from_config(model_config).to(device)
    weights = class_weights(
        train_dataset.rows[label_column],
        num_classes=int(model_config["num_classes"]),
        power=float(experiment["loss"]["class_weight_power"]),
    ).to(device)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    morphology_config = experiment.get("morphology", {})
    morphology_enabled = bool(morphology_config.get("enabled", False))
    morphology_train_targets: dict[str, tuple[float, float]] | None = None
    morphology_validation_targets: dict[str, tuple[float, float]] | None = None
    morphology_weights: torch.Tensor | None = None
    morphology_loss_function: nn.Module | None = None
    morphology_strength = 0.0
    if morphology_enabled:
        if str(model_config.get("morphology_supervision", "none")) == "none":
            raise ValueError("Enabled morphology loss requires model auxiliary heads")
        for dataset in (train_dataset, validation_dataset):
            if "label_name" not in dataset.rows.columns:
                raise ValueError("Morphology supervision requires label_name in the manifest")
        morphology_train_targets = {
            str(row.sample_id): morphology_target(str(row.label_name))
            for row in train_dataset.rows.itertuples(index=False)
        }
        morphology_validation_targets = {
            str(row.sample_id): morphology_target(str(row.label_name))
            for row in validation_dataset.rows.itertuples(index=False)
        }
        morphology_weights = attribute_pos_weights(
            morphology_train_targets.values(),
            power=float(morphology_config.get("class_weight_power", -0.5)),
        ).to(device)
        morphology_loss_function = nn.BCEWithLogitsLoss(
            pos_weight=morphology_weights
        )
        morphology_strength = float(morphology_config.get("loss_weight", 0.4))
        if morphology_strength < 0:
            raise ValueError("Morphology loss_weight must be non-negative")
    elif str(model_config.get("morphology_supervision", "none")) != "none":
        raise ValueError("Configured morphology heads require an enabled morphology loss")
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
        "row_filters": row_filters,
        "morphology_attribute_names": (
            MORPHOLOGY_NAMES if morphology_enabled else None
        ),
        "morphology_pos_weights": (
            morphology_weights.detach().cpu().tolist()
            if morphology_weights is not None
            else None
        ),
        "training_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "deployed_parameters": sum(
            parameter.numel()
            for parameter in switch_model_to_deploy(model).parameters()
        ),
    }
    (run_dir / "configuration.json").write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8",
    )

    history: list[dict[str, float | int]] = []
    best_score = -1.0
    stale_epochs = 0
    best_result = None
    best_morphology_result = None
    best_branch_weights = None
    best_hierarchy_weight = None
    for epoch in range(max_epochs):
        strength = consistency_weight(
            epoch,
            maximum=float(consistency.get("max_weight", 0.0)),
            warmup_epochs=int(consistency.get("warmup_epochs", 0)),
        )
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_function,
            consistency_strength=strength,
            morphology_targets=morphology_train_targets,
            morphology_loss_function=morphology_loss_function,
            morphology_strength=morphology_strength,
        )
        validation_result = evaluate(
            model,
            validation_loader,
            device,
            loss_function,
            class_names=class_names,
        )
        validation_morphology = (
            evaluate_morphology(
                model,
                validation_loader,
                device,
                morphology_loss_function,
                morphology_validation_targets,
                attribute_names=MORPHOLOGY_NAMES,
            )
            if morphology_loss_function is not None
            and morphology_validation_targets is not None
            else None
        )
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "consistency_weight": strength,
            "train_loss": train_loss,
            "validation_loss": validation_result.loss,
            **{
                f"validation_{name}": float(value)
                for name, value in validation_result.metrics.items()
                if isinstance(value, float)
            },
            **(
                {
                    "validation_morphology_loss": validation_morphology.loss,
                    **{
                        f"validation_{attribute}_{metric}": value
                        for attribute, attribute_metrics in (
                            validation_morphology.metrics.items()
                        )
                        for metric, value in attribute_metrics.items()
                    },
                }
                if validation_morphology is not None
                else {}
            ),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        score = float(validation_result.metrics[selection_metric])
        print(json.dumps(row))
        if score > best_score:
            best_score = score
            stale_epochs = 0
            best_result = validation_result
            best_morphology_result = validation_morphology
            best_branch_weights = branch_weight_summary(model)
            best_hierarchy_weight = (
                float(model.hierarchy_weight().detach().cpu())
                if model.hierarchy_fusion != "none"
                else None
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "feature_config": feature_config.__dict__,
                    "normalization_path": str(stats_path.relative_to(root)),
                    "class_names": class_names,
                    "label_column": label_column,
                    "selection_metric": selection_metric,
                    "row_filters": row_filters,
                    "validation_metrics": validation_result.metrics,
                    "validation_morphology_metrics": (
                        validation_morphology.metrics
                        if validation_morphology is not None
                        else None
                    ),
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
                sample_id.split("_", maxsplit=1)[0] for sample_id in best_result.sample_ids
            ],
            "target": best_result.targets,
            "prediction": best_result.predictions,
            **{
                f"probability_{class_index}": best_result.probabilities[:, class_index]
                for class_index in range(best_result.probabilities.shape[1])
            },
        }
    )
    prediction_table.to_csv(run_dir / "best_validation_predictions.csv", index=False)
    if best_morphology_result is not None:
        morphology_prediction_table = pd.DataFrame(
            {
                "sample_id": best_morphology_result.sample_ids,
                **{
                    f"target_{attribute}": best_morphology_result.targets[:, index]
                    for index, attribute in enumerate(MORPHOLOGY_NAMES)
                },
                **{
                    f"probability_{attribute}": (
                        best_morphology_result.probabilities[:, index]
                    )
                    for index, attribute in enumerate(MORPHOLOGY_NAMES)
                },
            }
        )
        morphology_prediction_table.to_csv(
            run_dir / "best_validation_morphology_predictions.csv",
            index=False,
        )
    summary = {
        "selection_metric": selection_metric,
        "best_selection_score": best_score,
        "best_validation_metrics": best_result.metrics,
        "best_validation_morphology_metrics": (
            best_morphology_result.metrics
            if best_morphology_result is not None
            else None
        ),
        "epochs_completed": len(history),
        "branch_weights_best_checkpoint": best_branch_weights,
        "branch_weights_final": branch_weight_summary(model),
        "hierarchy_weight_best_checkpoint": best_hierarchy_weight,
        "hierarchy_weight_final": (
            float(model.hierarchy_weight().detach().cpu())
            if model.hierarchy_fusion != "none"
            else None
        ),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
