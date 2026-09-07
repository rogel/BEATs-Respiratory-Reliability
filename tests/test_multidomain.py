import pytest
import torch

from respiratory_sound.multidomain import domain_alignment_loss


def test_class_conditional_alignment_is_zero_for_matching_class_means() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    domains = torch.tensor([0, 0, 1, 1])
    targets = torch.tensor([0, 1, 0, 1])

    assert torch.allclose(
        domain_alignment_loss(embeddings, domains, targets, "class_conditional"),
        torch.zeros(()),
    )


def test_class_conditioning_detects_swapped_classes_hidden_by_global_mean() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    )
    domains = torch.tensor([0, 0, 1, 1])
    targets = torch.tensor([0, 1, 0, 1])

    unconditional = domain_alignment_loss(
        embeddings,
        domains,
        targets,
        "unconditional",
    )
    conditional = domain_alignment_loss(
        embeddings,
        domains,
        targets,
        "class_conditional",
    )

    assert torch.allclose(unconditional, torch.zeros(()), atol=1.0e-6)
    assert conditional > 0.9


def test_alignment_requires_complete_balanced_batch() -> None:
    with pytest.raises(ValueError, match="both databases"):
        domain_alignment_loss(
            torch.randn(4, 3),
            torch.zeros(4, dtype=torch.long),
            torch.tensor([0, 1, 0, 1]),
            "class_conditional",
        )
