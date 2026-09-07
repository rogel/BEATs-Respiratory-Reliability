"""Reproducible training and evaluation utilities."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from respiratory_sound.metrics import CLASS_NAMES, respiratory_metrics


@dataclass(frozen=True)
class EpochResult:
    loss: float
    metrics: dict[str, object]
    targets: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray
    sample_ids: list[str]


@dataclass(frozen=True)
class MorphologyResult:
    loss: float
    metrics: dict[str, dict[str, float]]
    targets: np.ndarray
    probabilities: np.ndarray
    sample_ids: list[str]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(
    labels: Iterable[int],
    num_classes: int = 4,
    power: float = -0.5,
) -> Tensor:
    counts = torch.bincount(torch.tensor(list(labels)), minlength=num_classes).float()
    if torch.any(counts == 0):
        raise ValueError(f"At least one class is absent from training data: {counts.tolist()}")
    weights = counts.pow(power)
    return weights / weights.mean()


def attribute_pos_weights(targets: Iterable[Iterable[float]], power: float = -0.5) -> Tensor:
    """Return BCE positive weights using the same count-power convention as class weights."""
    target_tensor = torch.tensor(list(targets), dtype=torch.float32)
    if target_tensor.ndim != 2:
        raise ValueError("Attribute targets must have shape [samples, attributes]")
    positives = target_tensor.sum(dim=0)
    negatives = target_tensor.shape[0] - positives
    if torch.any(positives == 0) or torch.any(negatives == 0):
        raise ValueError(
            "Each attribute must contain positive and negative training samples"
        )
    return (positives / negatives).pow(power)


def consistency_weight(epoch: int, maximum: float, warmup_epochs: int) -> float:
    if maximum <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return maximum
    return maximum * min(1.0, (epoch + 1) / warmup_epochs)


def jensen_shannon_consistency(logits: Tensor) -> Tensor:
    """Jensen-Shannon divergence across augmented views: [batch, views, classes]."""
    if logits.ndim != 3 or logits.shape[1] < 2:
        raise ValueError("Expected at least two prediction views")
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    mean_probability = probabilities.mean(dim=1)
    mean_log_probability = mean_probability.clamp_min(1e-8).log()
    divergences = (probabilities * (log_probabilities - mean_log_probability.unsqueeze(1))).sum(
        dim=-1
    )
    return divergences.mean()


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    max_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_function: nn.Module,
    consistency_strength: float,
    morphology_targets: Mapping[str, tuple[float, float]] | None = None,
    morphology_loss_function: nn.Module | None = None,
    morphology_strength: float = 0.0,
) -> float:
    if (morphology_targets is None) != (morphology_loss_function is None):
        raise ValueError(
            "morphology_targets and morphology_loss_function must be provided together"
        )
    if morphology_targets is None and morphology_strength != 0:
        raise ValueError("morphology_strength requires morphology supervision")
    model.train()
    total_loss = 0.0
    total_samples = 0
    for views, frame_masks, targets, sample_ids in loader:
        batch_size, num_views, channels, mels, frames = views.shape
        views = views.reshape(batch_size * num_views, channels, mels, frames).to(device)
        frame_masks = frame_masks.reshape(batch_size * num_views, frames).to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if morphology_targets is None:
            logits = model(views, frame_masks=frame_masks)
            morphology_logits = None
        else:
            logits, morphology_logits = model(
                views,
                frame_masks=frame_masks,
                return_aux=True,
            )
        logits = logits.reshape(batch_size, num_views, -1)
        repeated_targets = targets.unsqueeze(1).expand(-1, num_views).reshape(-1)
        supervised_loss = loss_function(logits.reshape(-1, logits.shape[-1]), repeated_targets)
        consistency_loss = (
            jensen_shannon_consistency(logits) if num_views > 1 else logits.new_zeros(())
        )
        loss = supervised_loss + consistency_strength * consistency_loss
        if morphology_logits is not None:
            batch_morphology_targets = torch.tensor(
                [morphology_targets[sample_id] for sample_id in sample_ids],
                dtype=morphology_logits.dtype,
                device=device,
            )
            repeated_morphology_targets = (
                batch_morphology_targets.unsqueeze(1)
                .expand(-1, num_views, -1)
                .reshape(batch_size * num_views, -1)
            )
            morphology_loss = morphology_loss_function(
                morphology_logits,
                repeated_morphology_targets,
            )
            loss = loss + morphology_strength * morphology_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return total_loss / total_samples


@torch.inference_mode()
def evaluate_morphology(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
    target_lookup: Mapping[str, tuple[float, float]],
    attribute_names: tuple[str, str] = ("transient", "continuous"),
) -> MorphologyResult:
    """Evaluate training-only morphology heads without affecting model selection."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[Tensor] = []
    all_probabilities: list[Tensor] = []
    all_sample_ids: list[str] = []
    for views, frame_masks, _, sample_ids in loader:
        if views.shape[1] != 1:
            raise ValueError("Morphology evaluation requires exactly one view")
        inputs = views[:, 0].to(device)
        input_masks = frame_masks[:, 0].to(device)
        _, morphology_logits = model(
            inputs,
            frame_masks=input_masks,
            return_aux=True,
        )
        targets = torch.tensor(
            [target_lookup[sample_id] for sample_id in sample_ids],
            dtype=morphology_logits.dtype,
            device=device,
        )
        loss = loss_function(morphology_logits, targets)
        total_loss += float(loss.cpu()) * targets.shape[0]
        total_samples += targets.shape[0]
        all_targets.append(targets.cpu())
        all_probabilities.append(torch.sigmoid(morphology_logits).cpu())
        all_sample_ids.extend(sample_ids)

    target_array = torch.cat(all_targets).numpy()
    probability_array = torch.cat(all_probabilities).numpy()
    metrics: dict[str, dict[str, float]] = {}
    for attribute_index, attribute_name in enumerate(attribute_names):
        attribute_targets = target_array[:, attribute_index].astype(int)
        attribute_probabilities = probability_array[:, attribute_index]
        attribute_predictions = (attribute_probabilities >= 0.5).astype(int)
        metrics[attribute_name] = {
            "f1": float(
                f1_score(
                    attribute_targets,
                    attribute_predictions,
                    zero_division=0,
                )
            ),
            "roc_auc": float(
                roc_auc_score(attribute_targets, attribute_probabilities)
            ),
            "average_precision": float(
                average_precision_score(
                    attribute_targets,
                    attribute_probabilities,
                )
            ),
        }
    return MorphologyResult(
        loss=total_loss / total_samples,
        metrics=metrics,
        targets=target_array,
        probabilities=probability_array,
        sample_ids=all_sample_ids,
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> EpochResult:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[Tensor] = []
    all_probabilities: list[Tensor] = []
    all_sample_ids: list[str] = []
    for views, frame_masks, targets, sample_ids in loader:
        if views.shape[1] != 1:
            raise ValueError("Evaluation dataset must return exactly one view")
        inputs = views[:, 0].to(device)
        input_masks = frame_masks[:, 0].to(device)
        targets_device = targets.to(device)
        logits = model(inputs, frame_masks=input_masks)
        loss = loss_function(logits, targets_device)
        probabilities = torch.softmax(logits, dim=-1)
        total_loss += float(loss.cpu()) * targets.shape[0]
        total_samples += targets.shape[0]
        all_targets.append(targets.cpu())
        all_probabilities.append(probabilities.cpu())
        all_sample_ids.extend(sample_ids)

    targets_array = torch.cat(all_targets).numpy()
    probabilities_array = torch.cat(all_probabilities).numpy()
    predictions = probabilities_array.argmax(axis=1)
    return EpochResult(
        loss=total_loss / total_samples,
        metrics=respiratory_metrics(targets_array, predictions, class_names),
        targets=targets_array,
        predictions=predictions,
        probabilities=probabilities_array,
        sample_ids=all_sample_ids,
    )


def branch_weight_summary(model: nn.Module) -> dict[str, list[float]]:
    from respiratory_sound.models.reparameterization import ReparamDepthwiseConv2d

    return {
        name: module.branch_weights().detach().cpu().tolist()
        for name, module in model.named_modules()
        if isinstance(module, ReparamDepthwiseConv2d) and not module.deploy
    }
