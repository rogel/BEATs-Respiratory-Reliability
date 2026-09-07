#!/usr/bin/env python3
"""Train one shared model with balanced patients, classes, and databases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
from respiratory_sound.data.sampling import (
    DomainClassEventBatchSampler,
    DomainClassPatientBatchSampler,
    EventRandomBatchSampler,
)
from respiratory_sound.models.factory import tfdcr_from_config
from respiratory_sound.models.reparameterization import switch_model_to_deploy
from respiratory_sound.multidomain import domain_alignment_loss
from respiratory_sound.runtime import select_device
from respiratory_sound.training import (
    consistency_weight,
    evaluate,
    jensen_shannon_consistency,
    seed_everything,
    warmup_cosine_scheduler,
)

CLASS_NAMES = ("normal", "adventitious")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def _augmentation(experiment: dict[str, Any]) -> tuple[bool, AugmentationConfig]:
    config = experiment.get("augmentation", {})
    enabled = bool(config.get("enabled", True))
    if enabled:
        return enabled, AugmentationConfig(
            **{key: value for key, value in config.items() if key != "enabled"}
        )
    return enabled, AugmentationConfig(
        gain_min=1.0,
        gain_max=1.0,
        noise_probability=0.0,
        shift_probability=0.0,
        frequency_mask_bins=0,
        time_mask_frames=0,
        frequency_response_probability=0.0,
        frequency_response_max_db=0.0,
    )


def _domain_lookup(rows: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    domains = sorted(str(value) for value in rows["dataset"].unique())
    if len(domains) != 2:
        raise ValueError("Joint training requires exactly two databases")
    domain_to_id = {domain: index for index, domain in enumerate(domains)}
    sample_to_domain = {
        str(row.sample_id): domain_to_id[str(row.dataset)]
        for row in rows.itertuples(index=False)
    }
    return domain_to_id, sample_to_domain


def _batch_sampler(
    rows: pd.DataFrame,
    experiment: dict[str, Any],
    batch_size: int,
    seed: int,
    label_column: str,
) -> tuple[Any, str]:
    sampling = experiment.get("sampling", {})
    mode = str(sampling.get("mode", "domain_class_patient"))
    common = {
        "rows": rows,
        "batch_size": batch_size,
        "samples_per_epoch": experiment.get("samples_per_epoch"),
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
    if mode == "domain_class_patient":
        return (
            DomainClassPatientBatchSampler(
                **common,
                class_column=label_column,
            ),
            mode,
        )
    raise ValueError(f"Unsupported sampling mode: {mode}")


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sample_to_domain: dict[str, int],
    consistency_strength: float,
    alignment_mode: str,
    alignment_strength: float,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "consistency_loss": 0.0,
        "alignment_loss": 0.0,
    }
    total_samples = 0
    for views, frame_masks, targets, sample_ids in loader:
        batch_size, num_views, channels, mels, frames = views.shape
        flat_views = views.reshape(
            batch_size * num_views,
            channels,
            mels,
            frames,
        ).to(device)
        flat_masks = frame_masks.reshape(batch_size * num_views, frames).to(device)
        targets = targets.to(device)
        domain_ids = torch.tensor(
            [sample_to_domain[sample_id] for sample_id in sample_ids],
            dtype=torch.long,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        feature_maps = model.forward_features(flat_views, frame_masks=flat_masks)
        flat_embeddings = model.pool(feature_maps, frame_masks=flat_masks)
        flat_logits = model.classifier(flat_embeddings)
        logits = flat_logits.reshape(batch_size, num_views, -1)
        embeddings = flat_embeddings.reshape(batch_size, num_views, -1).mean(dim=1)
        repeated_targets = targets[:, None].expand(-1, num_views).reshape(-1)
        classification_loss = nn.functional.cross_entropy(
            flat_logits,
            repeated_targets,
        )
        consistency_loss = (
            jensen_shannon_consistency(logits)
            if num_views > 1
            else flat_logits.new_zeros(())
        )
        alignment_loss = domain_alignment_loss(
            embeddings,
            domain_ids,
            targets,
            mode=alignment_mode,
        )
        loss = (
            classification_loss
            + consistency_strength * consistency_loss
            + alignment_strength * alignment_loss
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        for name, value in (
            ("loss", loss),
            ("classification_loss", classification_loss),
            ("consistency_loss", consistency_loss),
            ("alignment_loss", alignment_loss),
        ):
            totals[name] += float(value.detach().cpu()) * batch_size
        total_samples += batch_size
    return {name: value / total_samples for name, value in totals.items()}


def _prediction_frame(result: Any) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": result.sample_ids,
            "target": result.targets,
            "prediction": result.predictions,
            "probability_0": result.probabilities[:, 0],
            "probability_1": result.probabilities[:, 1],
        }
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
    experiment = yaml.safe_load(experiment_config_path.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else experiment["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")

    feature_config = feature_config_from_yaml(data_config_path)
    normalization = FeatureNormalization.from_json(stats_path)
    split_column = str(experiment.get("split_column", "protocol_role"))
    train_value = str(experiment.get("train_value", "train_fit"))
    validation_value = str(
        experiment.get("validation_value", "validation_select")
    )
    label_column = str(experiment.get("label_column", "binary_label_id"))
    augmentation_enabled, augmentation = _augmentation(experiment)
    consistency = experiment.get("consistency", {})
    num_views = 2 if consistency.get("enabled", False) else 1
    sample_limit_validation = experiment.get("sample_limit_validation")
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
        label_column=label_column,
    )
    domains = sorted(str(value) for value in train_dataset.rows["dataset"].unique())
    validation_datasets = {
        domain: ICBHICycleDataset(
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
    domain_to_id, sample_to_domain = _domain_lookup(train_dataset.rows)
    model = tfdcr_from_config(model_config).to(device)
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
    alignment = experiment.get("alignment", {})
    alignment_mode = str(alignment.get("mode", "none"))
    alignment_strength = float(alignment.get("weight", 0.0))
    if alignment_mode == "none" and alignment_strength != 0.0:
        raise ValueError("Alignment weight must be zero when alignment is disabled")

    run_dir = root / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = {
        "seed": seed,
        "device": str(device),
        "model": model_config,
        "experiment": experiment,
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
        "domain_to_id": domain_to_id,
        "batches_per_epoch": len(sampler),
        "sampling_mode": sampling_mode,
        "samples_per_batch_stratum": getattr(sampler, "per_stratum", None),
        "training_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "deployed_parameters": sum(
            parameter.numel()
            for parameter in switch_model_to_deploy(model).parameters()
        ),
        "locked_test_accessed": False,
    }
    (run_dir / "configuration.json").write_text(
        json.dumps(snapshot, indent=2),
        encoding="utf-8",
    )

    history: list[dict[str, float]] = []
    best_worst_score = -1.0
    best_mean_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_results: dict[str, Any] | None = None
    stale_epochs = 0
    for epoch in range(max_epochs):
        sampler.set_epoch(epoch)
        js_strength = consistency_weight(
            epoch,
            maximum=float(consistency.get("max_weight", 0.0)),
            warmup_epochs=int(consistency.get("warmup_epochs", 0)),
        )
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            sample_to_domain,
            consistency_strength=js_strength,
            alignment_mode=alignment_mode,
            alignment_strength=alignment_strength,
        )
        validation_results = {
            domain: evaluate(
                model,
                loader,
                device,
                nn.CrossEntropyLoss(),
                class_names=CLASS_NAMES,
            )
            for domain, loader in validation_loaders.items()
        }
        domain_scores = {
            domain: float(result.metrics["average_score"])
            for domain, result in validation_results.items()
        }
        worst_score = min(domain_scores.values())
        mean_score = sum(domain_scores.values()) / len(domain_scores)
        row = {
            "epoch": float(epoch + 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "consistency_weight": js_strength,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            "validation_worst_average_score": worst_score,
            "validation_mean_average_score": mean_score,
            **{
                f"validation_{domain}_{metric}": float(value)
                for domain, result in validation_results.items()
                for metric, value in result.metrics.items()
                if isinstance(value, float)
            },
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        improved = worst_score > best_worst_score + 1.0e-12 or (
            abs(worst_score - best_worst_score) <= 1.0e-12
            and mean_score > best_mean_score + 1.0e-12
        )
        if improved:
            best_worst_score = worst_score
            best_mean_score = mean_score
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_results = validation_results
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": best_state,
                    "model_config": model_config,
                    "feature_config": vars(feature_config),
                    "normalization_path": str(stats_path.relative_to(root)),
                    "label_column": label_column,
                    "class_names": CLASS_NAMES,
                    "seed": seed,
                    "joint_multidomain": True,
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

    if best_state is None or best_results is None:
        raise RuntimeError("Joint training did not produce a best checkpoint")
    summary = {
        "selection_metric": "minimum_domain_average_score",
        "best_worst_domain_average_score": best_worst_score,
        "best_mean_domain_average_score": best_mean_score,
        "epochs_completed": len(history),
        "best_validation_metrics": {
            domain: result.metrics for domain, result in best_results.items()
        },
        "alignment_mode": alignment_mode,
        "alignment_strength": alignment_strength,
        "locked_test_accessed": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
