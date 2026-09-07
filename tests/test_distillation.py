from __future__ import annotations

import pytest
import torch

from respiratory_sound.distillation import (
    binary_distillation_kl,
    binary_teacher_distribution,
)


def test_teacher_distribution_preserves_binary_probability_at_temperature_one() -> None:
    probabilities = torch.tensor([0.2, 0.7])
    distribution = binary_teacher_distribution(
        probabilities,
        temperature=1.0,
    )
    assert torch.allclose(distribution[:, 1], probabilities, atol=1.0e-6)
    assert torch.allclose(distribution.sum(dim=1), torch.ones(2))


def test_distillation_kl_is_zero_for_matching_student() -> None:
    probabilities = torch.tensor([0.2, 0.7])
    binary_logits = torch.stack(
        (torch.zeros_like(probabilities), torch.logit(probabilities)),
        dim=-1,
    )
    logits = binary_logits[:, None, :].repeat(1, 2, 1)
    loss = binary_distillation_kl(
        logits,
        probabilities,
        temperature=2.0,
    )
    assert float(loss) == pytest.approx(0.0, abs=1.0e-6)


def test_distillation_rejects_nonbinary_logits() -> None:
    with pytest.raises(ValueError, match="\\[batch, views, 2\\]"):
        binary_distillation_kl(
            torch.zeros(2, 1, 3),
            torch.tensor([0.2, 0.7]),
            temperature=2.0,
        )
