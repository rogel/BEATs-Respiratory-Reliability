import torch

from respiratory_sound.adaptation import (
    LowRankResidualAdapter,
    SourceAnchoredPrototypeAdapter,
    class_prototypes,
)


def test_low_rank_adapter_starts_as_exact_identity() -> None:
    adapter = LowRankResidualAdapter(embedding_dim=16, rank=4)
    embeddings = torch.randn(7, 16)

    assert torch.equal(adapter(embeddings), embeddings)


def test_source_anchored_adapter_only_trains_low_rank_parameters() -> None:
    model = SourceAnchoredPrototypeAdapter(
        classifier_weight=torch.randn(2, 16),
        classifier_bias=torch.randn(2),
        source_prototypes=torch.randn(2, 16),
        rank=4,
    )

    assert set(dict(model.named_parameters())) == {
        "adapter.down.weight",
        "adapter.down.bias",
        "adapter.up.weight",
        "adapter.up.bias",
    }


def test_prototype_loss_reaches_adapter_after_identity_anchor() -> None:
    torch.manual_seed(31)
    model = SourceAnchoredPrototypeAdapter(
        classifier_weight=torch.randn(2, 12),
        classifier_bias=torch.randn(2),
        source_prototypes=torch.randn(2, 12),
        rank=3,
    )
    embeddings = torch.randn(8, 12)
    _, prototype_logits, _ = model(embeddings)
    prototype_logits.square().mean().backward()

    assert model.adapter.up.weight.grad is not None
    assert torch.isfinite(model.adapter.up.weight.grad).all()


def test_class_prototypes_are_unit_normalized() -> None:
    embeddings = torch.randn(10, 6)
    targets = torch.tensor([0] * 5 + [1] * 5)
    prototypes = class_prototypes(embeddings, targets)

    assert prototypes.shape == (2, 6)
    assert torch.allclose(
        prototypes.norm(dim=1),
        torch.ones(2),
        atol=1.0e-6,
    )
