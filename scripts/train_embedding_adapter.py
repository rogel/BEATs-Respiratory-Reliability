#!/usr/bin/env python3
"""Train a source-anchored low-rank adapter with patient-internal selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import Tensor, nn
from torch.nn import functional as F

from respiratory_sound.adaptation import (
    SourceAnchoredPrototypeAdapter,
    class_prototypes,
)
from respiratory_sound.metrics import (
    paired_patient_bootstrap_difference,
    patient_bootstrap_intervals,
    respiratory_metrics,
)

CLASS_NAMES = ("normal", "adventitious")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-embeddings", type=Path, required=True)
    parser.add_argument("--target-calibration-embeddings", type=Path, required=True)
    parser.add_argument("--target-evaluation-embeddings", type=Path, required=True)
    parser.add_argument("--gate4a-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("prototype", "ce_only"), default="prototype")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--prototype-temperature", type=float, default=0.1)
    parser.add_argument("--prototype-loss-weight", type=float, default=0.3)
    parser.add_argument("--identity-loss-weight", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--maximum-epochs", type=int, default=200)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20_260_729)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--backbone-parameters", type=int, default=540_243)
    return parser.parse_args()


def _load_embeddings(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve()) as payload:
        required = {
            "embeddings",
            "targets",
            "sample_ids",
            "patient_ids",
            "classifier_weight",
            "classifier_bias",
        }
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        result = {name: payload[name].copy() for name in required}
    embeddings = result["embeddings"]
    targets = result["targets"]
    if embeddings.ndim != 2 or targets.shape != (embeddings.shape[0],):
        raise ValueError(f"{path} has incompatible embedding/target shapes")
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError(f"{path} must contain both binary classes")
    return result


def _class_weights(targets: Tensor) -> Tensor:
    counts = torch.bincount(targets, minlength=2).to(dtype=torch.float32)
    weights = counts.pow(-0.5)
    return weights / weights.mean()


def _new_model(
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    prototypes: Tensor,
    args: argparse.Namespace,
) -> SourceAnchoredPrototypeAdapter:
    return SourceAnchoredPrototypeAdapter(
        classifier_weight=classifier_weight,
        classifier_bias=classifier_bias,
        source_prototypes=prototypes,
        rank=args.rank,
        residual_scale=args.residual_scale,
        prototype_temperature=args.prototype_temperature,
    )


def _train_epoch(
    model: SourceAnchoredPrototypeAdapter,
    embeddings: Tensor,
    targets: Tensor,
    optimizer: torch.optim.Optimizer,
    prototype_loss_weight: float,
    identity_loss_weight: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    class_logits, prototype_logits, adapted = model(embeddings)
    weights = _class_weights(targets)
    classification_loss = F.cross_entropy(class_logits, targets, weight=weights)
    prototype_loss = F.cross_entropy(prototype_logits, targets, weight=weights)
    identity_loss = (adapted - embeddings).square().mean()
    total_loss = (
        classification_loss
        + prototype_loss_weight * prototype_loss
        + identity_loss_weight * identity_loss
    )
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return {
        "loss": float(total_loss.detach()),
        "classification_loss": float(classification_loss.detach()),
        "prototype_loss": float(prototype_loss.detach()),
        "identity_loss": float(identity_loss.detach()),
    }


@torch.inference_mode()
def _evaluate(
    model: SourceAnchoredPrototypeAdapter,
    embeddings: Tensor,
    targets: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    classifier_logits, _, _ = model(embeddings)
    probabilities = torch.softmax(classifier_logits, dim=1)[:, 1].cpu().numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = respiratory_metrics(targets, predictions, CLASS_NAMES)
    metrics.update(
        {
            "auroc": float(roc_auc_score(targets, probabilities)),
            "auprc": float(average_precision_score(targets, probabilities)),
        }
    )
    return metrics, probabilities, predictions


def _fit_model(
    train_embeddings: Tensor,
    train_targets: Tensor,
    validation_embeddings: Tensor,
    validation_targets: np.ndarray,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    prototypes: Tensor,
    args: argparse.Namespace,
    epochs: int,
    seed: int,
) -> tuple[SourceAnchoredPrototypeAdapter, list[dict[str, float]], int]:
    torch.manual_seed(seed)
    model = _new_model(classifier_weight, classifier_bias, prototypes, args)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    prototype_weight = args.prototype_loss_weight if args.mode == "prototype" else 0.0
    history: list[dict[str, float]] = []
    best_epoch = 1
    best_selection = -np.inf
    best_state: dict[str, Tensor] | None = None
    for epoch in range(1, epochs + 1):
        losses = _train_epoch(
            model,
            train_embeddings,
            train_targets,
            optimizer,
            prototype_loss_weight=prototype_weight,
            identity_loss_weight=args.identity_loss_weight,
        )
        validation_metrics, _, _ = _evaluate(
            model,
            validation_embeddings,
            validation_targets,
        )
        selection_score = (
            float(validation_metrics["average_score"])
            + float(validation_metrics["auroc"])
        ) / 2.0
        history.append(
            {
                "epoch": float(epoch),
                **losses,
                "validation_average_score": float(
                    validation_metrics["average_score"]
                ),
                "validation_auroc": float(validation_metrics["auroc"]),
                "selection_score": selection_score,
            }
        )
        if selection_score > best_selection + 1.0e-12:
            best_selection = selection_score
            best_epoch = epoch
            best_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Adapter training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def main() -> None:
    args = parse_args()
    if args.maximum_epochs < 1 or args.folds < 2:
        raise ValueError("maximum_epochs and folds must be positive")
    if args.mode == "ce_only" and args.prototype_loss_weight <= 0:
        raise ValueError("Keep the frozen positive prototype weight for matched controls")
    source = _load_embeddings(args.source_embeddings)
    calibration = _load_embeddings(args.target_calibration_embeddings)
    evaluation = _load_embeddings(args.target_evaluation_embeddings)
    embedding_dims = {
        source["embeddings"].shape[1],
        calibration["embeddings"].shape[1],
        evaluation["embeddings"].shape[1],
    }
    if len(embedding_dims) != 1:
        raise ValueError("Source, calibration, and evaluation dimensions do not match")
    if set(calibration["patient_ids"]).intersection(evaluation["patient_ids"]):
        raise ValueError("Calibration and evaluation patients overlap")
    for name in ("classifier_weight", "classifier_bias"):
        if not np.array_equal(source[name], calibration[name]) or not np.array_equal(
            source[name],
            evaluation[name],
        ):
            raise ValueError(f"Frozen {name} differs across embedding files")

    source_embeddings = torch.from_numpy(source["embeddings"]).float()
    source_targets = torch.from_numpy(source["targets"]).long()
    calibration_embeddings = torch.from_numpy(calibration["embeddings"]).float()
    calibration_targets = torch.from_numpy(calibration["targets"]).long()
    evaluation_embeddings = torch.from_numpy(evaluation["embeddings"]).float()
    classifier_weight = torch.from_numpy(source["classifier_weight"]).float()
    classifier_bias = torch.from_numpy(source["classifier_bias"]).float()
    prototypes = class_prototypes(source_embeddings, source_targets)

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    fold_payloads: list[dict[str, Any]] = []
    fold_best_epochs: list[int] = []
    for fold_index, (train_indices, stop_indices) in enumerate(
        splitter.split(
            calibration["embeddings"],
            calibration["targets"],
            groups=calibration["patient_ids"],
        ),
        start=1,
    ):
        if set(np.unique(calibration["targets"][stop_indices])) != {0, 1}:
            raise ValueError(f"Fold {fold_index} stop set does not contain both classes")
        _, history, best_epoch = _fit_model(
            calibration_embeddings[train_indices],
            calibration_targets[train_indices],
            calibration_embeddings[stop_indices],
            calibration["targets"][stop_indices],
            classifier_weight,
            classifier_bias,
            prototypes,
            args,
            epochs=args.maximum_epochs,
            seed=args.seed + fold_index,
        )
        fold_best_epochs.append(best_epoch)
        fold_payloads.append(
            {
                "fold": fold_index,
                "fit_patients": int(
                    len(np.unique(calibration["patient_ids"][train_indices]))
                ),
                "stop_patients": int(
                    len(np.unique(calibration["patient_ids"][stop_indices]))
                ),
                "best_epoch": best_epoch,
                "best_row": history[best_epoch - 1],
            }
        )

    final_epochs = max(1, int(np.floor(np.median(fold_best_epochs) + 0.5)))
    torch.manual_seed(args.seed)
    final_model = _new_model(
        classifier_weight,
        classifier_bias,
        prototypes,
        args,
    )
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    prototype_weight = args.prototype_loss_weight if args.mode == "prototype" else 0.0
    final_history: list[dict[str, float]] = []
    for epoch in range(1, final_epochs + 1):
        losses = _train_epoch(
            final_model,
            calibration_embeddings,
            calibration_targets,
            final_optimizer,
            prototype_loss_weight=prototype_weight,
            identity_loss_weight=args.identity_loss_weight,
        )
        final_history.append({"epoch": float(epoch), **losses})

    evaluation_metrics, probabilities, predictions = _evaluate(
        final_model,
        evaluation_embeddings,
        evaluation["targets"],
    )
    patient_ids = evaluation["patient_ids"].astype(str)
    intervals = patient_bootstrap_intervals(
        evaluation["targets"],
        predictions,
        patient_ids,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
        class_names=CLASS_NAMES,
    )
    gate4a = pd.read_csv(args.gate4a_predictions, dtype={"patient_id": str})
    gate4a = gate4a.set_index("sample_id").loc[evaluation["sample_ids"]]
    if not np.array_equal(gate4a["target"].to_numpy(), evaluation["targets"]):
        raise ValueError("Gate 4A targets do not align with evaluation embeddings")
    gate4a_predictions = gate4a["calibrated_prediction"].to_numpy(dtype=np.int64)
    gate4a_metrics = respiratory_metrics(
        evaluation["targets"],
        gate4a_predictions,
        CLASS_NAMES,
    )
    paired_gate4a = paired_patient_bootstrap_difference(
        evaluation["targets"],
        predictions,
        gate4a_predictions,
        patient_ids,
        metric="average_score",
        iterations=args.bootstrap_iterations,
        seed=args.seed,
        class_names=CLASS_NAMES,
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in final_model.parameters()
        if parameter.requires_grad
    )
    parameter_fraction = trainable_parameters / args.backbone_parameters
    with torch.inference_mode():
        _, _, adapted_calibration = final_model(calibration_embeddings)
        delta = adapted_calibration - calibration_embeddings
        delta_mean_absolute = float(delta.abs().mean())
        embedding_mean_absolute = float(calibration_embeddings.abs().mean())
    bottleneck_gate = {
        "average_score_at_least_0_60": bool(
            evaluation_metrics["average_score"] >= 0.6
        ),
        "auroc_at_least_0_65": bool(evaluation_metrics["auroc"] >= 0.65),
        "sensitivity_at_least_0_50": bool(
            evaluation_metrics["sensitivity"] >= 0.5
        ),
        "specificity_at_least_0_50": bool(
            evaluation_metrics["specificity"] >= 0.5
        ),
        "exceeds_gate4a_average_score": bool(
            evaluation_metrics["average_score"] > gate4a_metrics["average_score"]
        ),
        "parameter_fraction_at_most_0_02": bool(parameter_fraction <= 0.02),
    }
    bottleneck_gate["passed"] = bool(all(bottleneck_gate.values()))
    payload = {
        "mode": args.mode,
        "seed": args.seed,
        "source_embeddings": str(args.source_embeddings),
        "target_calibration_embeddings": str(args.target_calibration_embeddings),
        "target_evaluation_embeddings": str(args.target_evaluation_embeddings),
        "calibration_patients": int(len(np.unique(calibration["patient_ids"]))),
        "evaluation_patients": int(len(np.unique(evaluation["patient_ids"]))),
        "patient_overlap": False,
        "source_prototype_cosine_similarity": float(
            F.cosine_similarity(prototypes[0], prototypes[1], dim=0)
        ),
        "folds": fold_payloads,
        "fold_best_epochs": fold_best_epochs,
        "final_epochs": final_epochs,
        "final_training_last_row": final_history[-1],
        "trainable_parameters": trainable_parameters,
        "backbone_parameters": args.backbone_parameters,
        "trainable_parameter_fraction": parameter_fraction,
        "delta_mean_absolute": delta_mean_absolute,
        "embedding_mean_absolute": embedding_mean_absolute,
        "evaluation_metrics": evaluation_metrics,
        "patient_bootstrap_intervals": intervals,
        "gate4a_metrics": gate4a_metrics,
        "paired_average_score_difference_vs_gate4a": paired_gate4a,
        "bottleneck_gate": bottleneck_gate,
        "locked_test_accessed": False,
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "configuration": vars(args),
            "final_epochs": final_epochs,
            "source_prototypes": prototypes,
        },
        output_dir / "adapter.pt",
    )
    pd.DataFrame(
        {
            "sample_id": evaluation["sample_ids"],
            "patient_id": patient_ids,
            "target": evaluation["targets"],
            "probability_1": probabilities,
            "prediction": predictions,
        }
    ).to_csv(output_dir / "evaluation_predictions.csv", index=False)
    pd.DataFrame(final_history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
