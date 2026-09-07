import torch
from torch import nn

from respiratory_sound.models.beats_adaptation import (
    LoRALinear,
    configure_beats_adaptation,
    projection_drift_regularization,
    reset_projection_drift,
)
from respiratory_sound.models.pretrained_audio import BEATsTransfer


class _ToyAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)


class _ToyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = _ToyAttention(width)
        self.feed_forward = nn.Linear(width, width)


class _ToyEncoder(nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer(width) for _ in range(depth)])
        self.layer_norm = nn.LayerNorm(width)


class _ToyBackbone(nn.Module):
    def __init__(self, width: int = 4, depth: int = 4) -> None:
        super().__init__()
        self.encoder = _ToyEncoder(width, depth)


def _toy_model() -> BEATsTransfer:
    return BEATsTransfer(_ToyBackbone(), embedding_dim=4, load_audit={})


def test_lora_linear_is_exact_at_initialization() -> None:
    base = nn.Linear(4, 4)
    adapted = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
    inputs = torch.randn(3, 4)

    outputs = adapted(inputs)

    assert torch.equal(outputs, base(inputs))
    assert adapted.last_relative_rms_drift is not None
    assert float(adapted.last_relative_rms_drift.detach()) == 0.0
    adapted.last_relative_rms_drift.backward()
    assert adapted.lora_b.weight.grad is not None
    assert torch.isfinite(adapted.lora_b.weight.grad).all()


def test_gate9b_lora_freezes_base_and_reports_drift() -> None:
    model = _toy_model()
    audit = configure_beats_adaptation(
        model,
        strategy="drift_anchored_lora_qv",
        lora_last_n_layers=2,
        lora_rank=2,
        lora_alpha=4.0,
        lora_dropout=0.0,
    )
    adapters = [
        module for module in model.modules()
        if isinstance(module, LoRALinear)
    ]
    inputs = torch.randn(3, 4)
    for adapter in adapters:
        adapter(inputs)

    assert audit.lora_modules == 4
    assert audit.adapted_layer_indices == (2, 3)
    assert all(
        not parameter.requires_grad
        for adapter in adapters
        for parameter in adapter.base.parameters()
    )
    assert float(projection_drift_regularization(model).detach()) == 0.0

    for adapter in adapters:
        nn.init.constant_(adapter.lora_b.weight, 0.1)
        adapter(inputs)

    assert float(projection_drift_regularization(model).detach()) > 0.0

    reset_projection_drift(model)
    assert float(projection_drift_regularization(model).detach()) == 0.0
    adapters[0](inputs)
    assert float(projection_drift_regularization(model).detach()) > 0.0
    assert adapters[1].last_relative_rms_drift is None


def test_head_only_and_last_block_have_expected_trainable_scope() -> None:
    head_only = _toy_model()
    head_audit = configure_beats_adaptation(head_only, strategy="head_only")
    assert head_audit.trainable_parameters == 10
    assert all(
        not parameter.requires_grad
        for parameter in head_only.backbone.parameters()
    )

    last_block = _toy_model()
    last_audit = configure_beats_adaptation(last_block, strategy="last_block")
    assert last_audit.trainable_backbone_parameters > 0
    assert all(
        not parameter.requires_grad
        for parameter in last_block.backbone.encoder.layers[0].parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in last_block.backbone.encoder.layers[-1].parameters()
    )
