"""Parameter-efficient class-conditional adaptation on frozen audio embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LowRankResidualAdapter(nn.Module):
    """Zero-initialized low-rank residual mapping for a frozen embedding."""

    def __init__(
        self,
        embedding_dim: int,
        rank: int = 8,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if embedding_dim < 1 or rank < 1:
            raise ValueError("embedding_dim and rank must be positive")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        self.embedding_dim = int(embedding_dim)
        self.residual_scale = float(residual_scale)
        self.down = nn.Linear(embedding_dim, rank)
        self.up = nn.Linear(rank, embedding_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.last_delta: Tensor | None = None

    def forward(self, embeddings: Tensor) -> Tensor:
        normalized = F.layer_norm(embeddings, (self.embedding_dim,))
        delta = self.up(F.gelu(self.down(normalized)))
        self.last_delta = delta
        return embeddings + self.residual_scale * delta


def class_prototypes(embeddings: Tensor, targets: Tensor) -> Tensor:
    if embeddings.ndim != 2 or targets.ndim != 1:
        raise ValueError("embeddings and targets must have shapes [N, D] and [N]")
    if embeddings.shape[0] != targets.shape[0]:
        raise ValueError("embeddings and targets must contain the same samples")
    if set(targets.detach().cpu().tolist()) != {0, 1}:
        raise ValueError("targets must contain both binary classes")
    normalized = F.normalize(embeddings, dim=1)
    prototypes = torch.stack(
        [normalized[targets == class_index].mean(dim=0) for class_index in (0, 1)]
    )
    return F.normalize(prototypes, dim=1)


class SourceAnchoredPrototypeAdapter(nn.Module):
    """Adapt target embeddings while retaining a frozen source decision head."""

    def __init__(
        self,
        classifier_weight: Tensor,
        classifier_bias: Tensor,
        source_prototypes: Tensor,
        rank: int = 8,
        residual_scale: float = 0.5,
        prototype_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if classifier_weight.ndim != 2 or classifier_weight.shape[0] != 2:
            raise ValueError("classifier_weight must have shape [2, embedding_dim]")
        if classifier_bias.shape != (2,):
            raise ValueError("classifier_bias must have shape [2]")
        if source_prototypes.shape != classifier_weight.shape:
            raise ValueError("source_prototypes must match classifier_weight")
        if prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        self.adapter = LowRankResidualAdapter(
            embedding_dim=classifier_weight.shape[1],
            rank=rank,
            residual_scale=residual_scale,
        )
        self.prototype_temperature = float(prototype_temperature)
        self.register_buffer("classifier_weight", classifier_weight.detach().clone())
        self.register_buffer("classifier_bias", classifier_bias.detach().clone())
        self.register_buffer(
            "source_prototypes",
            F.normalize(source_prototypes.detach().clone(), dim=1),
        )

    def forward(self, embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        adapted = self.adapter(embeddings)
        classifier_logits = F.linear(
            adapted,
            self.classifier_weight,
            self.classifier_bias,
        )
        prototype_logits = (
            F.normalize(adapted, dim=1) @ self.source_prototypes.T
        ) / self.prototype_temperature
        return classifier_logits, prototype_logits, adapted
