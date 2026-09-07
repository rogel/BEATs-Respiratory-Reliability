import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from respiratory_sound.models.network import TFDCRNet
from respiratory_sound.training import (
    attribute_pos_weights,
    class_weights,
    consistency_weight,
    jensen_shannon_consistency,
)


class _MorphologyTensorDataset(Dataset):
    def __init__(self) -> None:
        self.sample_ids = [f"sample_{index}" for index in range(4)]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        generator = torch.Generator().manual_seed(index)
        views = torch.randn(2, 1, 16, 24, generator=generator)
        masks = torch.ones(2, 24, dtype=torch.bool)
        return views, masks, index % 2, self.sample_ids[index]


def test_class_weights_are_normalized_and_upweight_rare_classes() -> None:
    weights = class_weights([0, 0, 0, 0, 1, 1, 2, 3], power=-0.5)

    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[0] < weights[1] < weights[2]
    assert weights[2] == weights[3]


def test_attribute_pos_weights_use_training_prevalence() -> None:
    weights = attribute_pos_weights(
        [(0.0, 0.0)] * 6 + [(1.0, 0.0)] * 2 + [(0.0, 1.0)] * 2,
        power=-0.5,
    )

    assert weights.tolist() == pytest.approx([2.0, 2.0])


def test_consistency_loss_is_zero_for_identical_views() -> None:
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [2.0, 0.0, -1.0]],
            [[-1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
        ]
    )

    assert jensen_shannon_consistency(logits).item() == pytest.approx(0.0, abs=1e-7)


def test_consistency_weight_warms_up_and_saturates() -> None:
    assert consistency_weight(0, maximum=0.4, warmup_epochs=4) == pytest.approx(0.1)
    assert consistency_weight(3, maximum=0.4, warmup_epochs=4) == pytest.approx(0.4)
    assert consistency_weight(20, maximum=0.4, warmup_epochs=4) == pytest.approx(0.4)


def test_two_view_training_repeats_morphology_targets() -> None:
    from respiratory_sound.training import train_one_epoch

    dataset = _MorphologyTensorDataset()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    model = TFDCRNet(
        mode="tfdcr",
        channels=(4, 8),
        depths=(1, 1),
        temporal_dilations=((1,), (1,)),
        expansion_ratio=2,
        num_classes=2,
        morphology_supervision="targeted",
        morphology_stages=(1,),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    target_lookup = {
        "sample_0": (0.0, 0.0),
        "sample_1": (1.0, 0.0),
        "sample_2": (0.0, 1.0),
        "sample_3": (1.0, 1.0),
    }
    loss = train_one_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        nn.CrossEntropyLoss(),
        consistency_strength=0.1,
        morphology_targets=target_lookup,
        morphology_loss_function=nn.BCEWithLogitsLoss(),
        morphology_strength=0.4,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert model.transient_heads[0].weight.grad is not None
    assert model.continuous_heads[0].weight.grad is not None
