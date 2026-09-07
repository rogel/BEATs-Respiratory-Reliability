"""Parameter-efficient BEATs adaptation strategies for Gate 9B."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from respiratory_sound.models.pretrained_audio import BEATsTransfer

AdaptationStrategy = Literal[
    "head_only",
    "last_block",
    "lora_qv",
    "drift_anchored_lora_qv",
]


@dataclass(frozen=True)
class AdaptationAudit:
    strategy: str
    total_parameters: int
    trainable_parameters: int
    trainable_fraction: float
    trainable_backbone_parameters: int
    trainable_classifier_parameters: int
    lora_modules: int
    lora_rank: int | None
    lora_alpha: float | None
    lora_dropout: float | None
    adapted_layer_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LoRALinear(nn.Module):
    """Frozen linear projection plus a trainable low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.last_relative_rms_drift: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        base_outputs = self.base(inputs)
        residual = self.lora_b(self.lora_a(self.dropout(inputs))) * self.scaling
        base_energy = base_outputs.detach().float().square().mean().clamp_min(1.0e-12)
        residual_energy = residual.float().square().mean()
        smoothing = 1.0e-8
        self.last_relative_rms_drift = (
            torch.sqrt(residual_energy / base_energy + smoothing)
            - smoothing**0.5
        )
        return base_outputs + residual


def _freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _encoder_layers(model: BEATsTransfer) -> nn.ModuleList:
    layers = getattr(model.backbone.encoder, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("BEATs encoder does not expose a ModuleList of layers")
    return layers


def _install_lora(
    model: BEATsTransfer,
    *,
    layer_indices: tuple[int, ...],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[LoRALinear]:
    layers = _encoder_layers(model)
    installed: list[LoRALinear] = []
    for index in layer_indices:
        if index < 0 or index >= len(layers):
            raise ValueError(f"Invalid BEATs layer index: {index}")
        attention = layers[index].self_attn
        for name in ("q_proj", "v_proj"):
            projection = getattr(attention, name)
            if not isinstance(projection, nn.Linear):
                raise TypeError(f"Expected Linear at encoder layer {index} {name}")
            adapted = LoRALinear(
                projection,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            setattr(attention, name, adapted)
            installed.append(adapted)
    return installed


def configure_beats_adaptation(
    model: BEATsTransfer,
    *,
    strategy: AdaptationStrategy,
    lora_last_n_layers: int = 4,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
) -> AdaptationAudit:
    """Freeze a loaded BEATs model and enable only the requested adaptation."""
    _freeze_all(model)
    model.classifier.requires_grad_(True)
    layers = _encoder_layers(model)
    lora_modules: list[LoRALinear] = []
    adapted_indices: tuple[int, ...] = ()

    if strategy == "head_only":
        pass
    elif strategy == "last_block":
        adapted_indices = (len(layers) - 1,)
        layers[-1].requires_grad_(True)
        model.backbone.encoder.layer_norm.requires_grad_(True)
    elif strategy in {"lora_qv", "drift_anchored_lora_qv"}:
        if lora_last_n_layers < 1 or lora_last_n_layers > len(layers):
            raise ValueError("lora_last_n_layers must fit the BEATs encoder")
        adapted_indices = tuple(range(len(layers) - lora_last_n_layers, len(layers)))
        lora_modules = _install_lora(
            model,
            layer_indices=adapted_indices,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
    else:
        raise ValueError(f"Unsupported BEATs adaptation strategy: {strategy}")

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    classifier_trainable = sum(
        parameter.numel() for parameter in model.classifier.parameters()
        if parameter.requires_grad
    )
    backbone_trainable = trainable - classifier_trainable
    model.gate9b_strategy = strategy
    model.gate9b_lora_modules = tuple(lora_modules)
    audit = AdaptationAudit(
        strategy=strategy,
        total_parameters=total,
        trainable_parameters=trainable,
        trainable_fraction=trainable / total,
        trainable_backbone_parameters=backbone_trainable,
        trainable_classifier_parameters=classifier_trainable,
        lora_modules=len(lora_modules),
        lora_rank=lora_rank if lora_modules else None,
        lora_alpha=lora_alpha if lora_modules else None,
        lora_dropout=lora_dropout if lora_modules else None,
        adapted_layer_indices=adapted_indices,
    )
    model.gate9b_adaptation_audit = audit.to_dict()
    return audit


def projection_drift_regularization(model: nn.Module) -> Tensor:
    """Return mean drift across adapters used by the most recent forward pass."""
    reference = next(model.parameters())
    modules = [
        module
        for module in model.modules()
        if isinstance(module, LoRALinear)
    ]
    if not modules:
        return reference.new_zeros(())
    values = [
        module.last_relative_rms_drift
        for module in modules
        if module.last_relative_rms_drift is not None
    ]
    if not values:
        return reference.new_zeros(())
    return torch.stack(values).mean()


def reset_projection_drift(model: nn.Module) -> None:
    """Clear cached adapter drift before a new stochastic-depth forward pass."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.last_relative_rms_drift = None
