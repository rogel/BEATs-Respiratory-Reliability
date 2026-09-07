"""Losses for supervised multi-database respiratory-sound training."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def domain_alignment_loss(
    embeddings: Tensor,
    domain_ids: Tensor,
    targets: Tensor,
    mode: str,
) -> Tensor:
    """Align database means globally or within each ground-truth class."""
    if mode == "none":
        return embeddings.new_zeros(())
    if mode not in {"unconditional", "class_conditional"}:
        raise ValueError(f"Unsupported alignment mode: {mode}")
    if embeddings.ndim != 2 or domain_ids.ndim != 1 or targets.ndim != 1:
        raise ValueError("Expected [N,D] embeddings and [N] domain/target arrays")
    if not (
        embeddings.shape[0] == domain_ids.shape[0] == targets.shape[0]
    ):
        raise ValueError("Embedding, domain, and target sample counts must match")
    if set(domain_ids.detach().cpu().tolist()) != {0, 1}:
        raise ValueError("Every balanced batch must contain both databases")

    def cosine_distance(first: Tensor, second: Tensor) -> Tensor:
        return 1.0 - F.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0))[0]

    if mode == "unconditional":
        means = [embeddings[domain_ids == domain].mean(dim=0) for domain in (0, 1)]
        return cosine_distance(means[0], means[1])

    if set(targets.detach().cpu().tolist()) != {0, 1}:
        raise ValueError("Every balanced batch must contain both classes")
    losses = []
    for class_index in (0, 1):
        class_means = [
            embeddings[
                (domain_ids == domain) & (targets == class_index)
            ].mean(dim=0)
            for domain in (0, 1)
        ]
        if not all(torch.isfinite(mean).all() for mean in class_means):
            raise ValueError("Every database-class stratum must be non-empty")
        losses.append(cosine_distance(class_means[0], class_means[1]))
    return torch.stack(losses).mean()
