"""Small, testable utilities for binary probability distillation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def binary_teacher_distribution(
    probability_1: Tensor,
    *,
    temperature: float,
    epsilon: float = 1.0e-6,
) -> Tensor:
    """Convert class-1 probabilities to a temperature-softened two-class target."""
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    probability_1 = probability_1.clamp(epsilon, 1.0 - epsilon)
    teacher_logit = torch.logit(probability_1)
    binary_logits = torch.stack(
        (torch.zeros_like(teacher_logit), teacher_logit),
        dim=-1,
    )
    return torch.softmax(binary_logits / temperature, dim=-1)


def binary_distillation_kl(
    logits: Tensor,
    teacher_probability_1: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Return temperature-scaled KL for logits shaped [batch, views, 2]."""
    if logits.ndim != 3 or logits.shape[-1] != 2:
        raise ValueError("Binary distillation expects logits shaped [batch, views, 2]")
    if teacher_probability_1.shape != (logits.shape[0],):
        raise ValueError("Teacher probabilities must contain one value per event")
    target = binary_teacher_distribution(
        teacher_probability_1,
        temperature=temperature,
    )
    target = target[:, None, :].expand(-1, logits.shape[1], -1)
    student_log_probability = torch.log_softmax(logits / temperature, dim=-1)
    return (
        nn.functional.kl_div(
            student_log_probability.reshape(-1, 2),
            target.reshape(-1, 2),
            reduction="batchmean",
        )
        * temperature**2
    )
