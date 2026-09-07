import pytest
import torch

from respiratory_sound.models import (
    AnchoredChannelGate,
    DifferenceDepthwiseConv2d,
    EventPool2d,
    MaskAwareRelaxedFrequencyNorm,
    ReparamDepthwiseConv2d,
    TFDCRNet,
    switch_model_to_deploy,
)


def test_mask_aware_frequency_norm_ignores_padding_and_normalizes_valid_frames() -> None:
    layer = MaskAwareRelaxedFrequencyNorm(relaxation_weight=1.0)
    inputs = torch.randn(2, 1, 5, 12)
    masks = torch.zeros(2, 12, dtype=torch.bool)
    masks[0, 2:10] = True
    masks[1, 4:9] = True
    modified = inputs.clone()
    modified[0, :, :, ~masks[0]] = 10_000.0
    modified[1, :, :, ~masks[1]] = -10_000.0

    expected = layer(inputs, masks)
    actual = layer(modified, masks)

    assert torch.allclose(expected, actual)
    assert torch.count_nonzero(actual.masked_select(~masks[:, None, None, :])) == 0
    for sample_index in range(2):
        valid = actual[sample_index, :, :, masks[sample_index]]
        assert torch.allclose(
            valid.mean(dim=-1),
            torch.zeros_like(valid.mean(dim=-1)),
            atol=1.0e-6,
        )
        assert torch.allclose(
            valid.var(dim=-1, correction=0),
            torch.ones_like(valid.var(dim=-1, correction=0)),
            atol=1.0e-4,
        )


def test_relaxed_frequency_norm_adds_no_trainable_parameters() -> None:
    options = {
        "mode": "plain",
        "channels": (8, 12),
        "depths": (1, 1),
        "temporal_dilations": ((1,), (1,)),
        "expansion_ratio": 2,
        "num_classes": 2,
        "pooling": "masked_mean",
    }
    plain = TFDCRNet(**options)
    candidate = TFDCRNet(
        input_normalization="relaxed_frequency",
        relaxed_frequency_weight=0.5,
        **options,
    )

    assert sum(parameter.numel() for parameter in plain.parameters()) == sum(
        parameter.numel() for parameter in candidate.parameters()
    )


def test_anchored_gate_starts_as_exact_identity_and_matches_control_size() -> None:
    torch.manual_seed(5)
    event_gate = AnchoredChannelGate(channels=12, conditioning="event")
    control_gate = AnchoredChannelGate(channels=12, conditioning="constant")
    response = torch.randn(3, 12, 7, 11)
    conditioning = torch.randn_like(response)
    masks = torch.ones(3, 11, dtype=torch.bool)

    assert torch.equal(event_gate(response, conditioning, masks), response)
    assert sum(parameter.numel() for parameter in event_gate.parameters()) == sum(
        parameter.numel() for parameter in control_gate.parameters()
    )


def test_event_gate_varies_by_sample_but_constant_control_does_not() -> None:
    torch.manual_seed(6)
    event_gate = AnchoredChannelGate(channels=8, conditioning="event")
    control_gate = AnchoredChannelGate(channels=8, conditioning="constant")
    with torch.no_grad():
        event_gate.expand.weight.normal_(std=0.2)
        control_gate.load_state_dict(event_gate.state_dict())
    response = torch.ones(2, 8, 3, 5)
    conditioning = torch.stack(
        (torch.zeros(8, 3, 5), torch.ones(8, 3, 5)),
    )

    event_gate(response, conditioning)
    control_gate(response, conditioning)

    assert not torch.allclose(event_gate.last_scale[0], event_gate.last_scale[1])
    assert torch.allclose(control_gate.last_scale[0], control_gate.last_scale[1])


def test_masked_pooling_ignores_invalid_time_frames() -> None:
    pool = EventPool2d(mode="masked_mean_lse", lse_temperature=0.7)
    inputs = torch.randn(2, 5, 4, 8)
    masks = torch.tensor(
        [
            [False, False, True, True, True, True, False, False],
            [True, True, True, False, False, False, False, False],
        ]
    )
    modified = inputs.clone()
    modified[..., ~masks[0]] = modified[..., ~masks[0]]
    modified[0, :, :, ~masks[0]] = 10_000.0
    modified[1, :, :, ~masks[1]] = -10_000.0

    expected = pool(inputs, masks)
    actual = pool(modified, masks)

    assert expected.shape == (2, 10)
    assert torch.allclose(expected, actual)


@pytest.mark.parametrize("branch_type", ["temporal", "frequency", "joint"])
def test_difference_kernels_have_zero_dc_response(branch_type: str) -> None:
    layer = DifferenceDepthwiseConv2d(channels=4, branch_type=branch_type)
    kernel_sum = layer.equivalent_kernel().sum(dim=(1, 2, 3))
    assert torch.allclose(kernel_sum, torch.zeros_like(kernel_sum), atol=1.0e-6)


@pytest.mark.parametrize(
    "branch_types",
    [
        ("standard",),
        ("standard", "standard", "standard"),
        ("standard", "temporal", "frequency"),
        ("standard", "temporal", "frequency", "joint"),
    ],
)
def test_layer_deployment_equivalence(branch_types: tuple[str, ...]) -> None:
    torch.manual_seed(7)
    layer = ReparamDepthwiseConv2d(
        channels=8,
        branch_types=branch_types,
        dilation=(1, 2),
    )
    layer.train()
    for _ in range(4):
        layer(torch.randn(3, 8, 17, 29))
    layer.eval()
    inputs = torch.randn(2, 8, 17, 29)
    expected = layer(inputs)
    converted = switch_model_to_deploy(layer)
    actual = converted(inputs)
    max_error = (expected - actual).abs().max().item()
    assert max_error < 1.0e-5


@pytest.mark.parametrize("mode", ["plain", "repconv", "tfdcr"])
def test_network_output_and_deployment_equivalence(mode: str) -> None:
    torch.manual_seed(11)
    model = TFDCRNet(
        mode=mode,
        channels=(8, 12),
        depths=(1, 1),
        temporal_dilations=((1,), (2,)),
        expansion_ratio=2,
        num_classes=4,
    )
    model.train()
    for _ in range(3):
        model(torch.randn(2, 1, 32, 64))
    model.eval()
    inputs = torch.randn(2, 1, 32, 64)
    expected = model(inputs)
    converted = switch_model_to_deploy(model)
    actual = converted(inputs)
    assert actual.shape == (2, 4)
    assert (expected - actual).abs().max().item() < 1.0e-5


@pytest.mark.parametrize("pooling", ["masked_mean", "masked_mean_lse"])
def test_masked_network_deployment_equivalence(pooling: str) -> None:
    torch.manual_seed(12)
    model = TFDCRNet(
        mode="tfdcr",
        channels=(8, 12),
        depths=(1, 1),
        temporal_dilations=((1,), (2,)),
        expansion_ratio=2,
        num_classes=4,
        pooling=pooling,
    )
    model.train()
    inputs = torch.randn(2, 1, 32, 64)
    masks = torch.zeros(2, 64, dtype=torch.bool)
    masks[:, 10:50] = True
    for _ in range(3):
        model(inputs, frame_masks=masks)
    model.eval()
    expected = model(inputs, frame_masks=masks)
    actual = switch_model_to_deploy(model)(inputs, frame_masks=masks)

    assert actual.shape == (2, 4)
    assert (expected - actual).abs().max().item() < 1.0e-5


def test_event_gated_network_deployment_equivalence() -> None:
    torch.manual_seed(14)
    model = TFDCRNet(
        mode="tfdcr",
        channels=(8, 12),
        depths=(1, 1),
        temporal_dilations=((1,), (1,)),
        expansion_ratio=2,
        num_classes=2,
        pooling="masked_mean",
        gate_conditioning="event",
        gate_stages=(1,),
    )
    model.train()
    inputs = torch.randn(2, 1, 32, 64)
    masks = torch.zeros(2, 64, dtype=torch.bool)
    masks[:, 8:52] = True
    for _ in range(3):
        model(inputs, frame_masks=masks)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, AnchoredChannelGate):
                module.expand.weight.normal_(std=0.1)
    model.eval()
    expected = model(inputs, frame_masks=masks)
    actual = switch_model_to_deploy(model)(inputs, frame_masks=masks)

    assert (expected - actual).abs().max().item() < 1.0e-5


@pytest.mark.parametrize("supervision", ["shared", "targeted"])
def test_morphology_heads_return_two_attributes_and_deploy_cleanly(
    supervision: str,
) -> None:
    torch.manual_seed(15)
    model = TFDCRNet(
        mode="tfdcr",
        channels=(8, 12),
        depths=(1, 1),
        temporal_dilations=((1,), (1,)),
        expansion_ratio=2,
        num_classes=2,
        pooling="masked_mean",
        morphology_supervision=supervision,
        morphology_stages=(1,),
        morphology_branch_indices=(1, 2),
    )
    model.train()
    inputs = torch.randn(3, 1, 32, 64)
    masks = torch.zeros(3, 64, dtype=torch.bool)
    masks[:, 8:52] = True
    for _ in range(3):
        model(inputs, frame_masks=masks)
    model.eval()

    class_logits, morphology_logits = model(
        inputs,
        frame_masks=masks,
        return_aux=True,
    )
    converted = switch_model_to_deploy(model)
    deployed_logits = converted(inputs, frame_masks=masks)

    assert class_logits.shape == (3, 2)
    assert morphology_logits.shape == (3, 2)
    assert converted.transient_heads is None
    assert converted.continuous_heads is None
    assert (class_logits - deployed_logits).abs().max().item() < 1.0e-5


def test_shared_and_targeted_morphology_controls_are_parameter_matched() -> None:
    options = {
        "mode": "tfdcr",
        "channels": (8, 12, 16),
        "depths": (1, 1, 1),
        "temporal_dilations": ((1,), (1,), (1,)),
        "expansion_ratio": 2,
        "num_classes": 2,
        "morphology_stages": (1, 2),
    }
    shared = TFDCRNet(morphology_supervision="shared", **options)
    targeted = TFDCRNet(morphology_supervision="targeted", **options)

    assert sum(parameter.numel() for parameter in shared.parameters()) == sum(
        parameter.numel() for parameter in targeted.parameters()
    )
    assert sum(
        parameter.numel() for parameter in switch_model_to_deploy(shared).parameters()
    ) == sum(
        parameter.numel() for parameter in switch_model_to_deploy(targeted).parameters()
    )


def test_targeted_morphology_loss_reaches_both_difference_branches() -> None:
    torch.manual_seed(16)
    model = TFDCRNet(
        mode="tfdcr",
        channels=(8, 12),
        depths=(1, 1),
        temporal_dilations=((1,), (1,)),
        expansion_ratio=2,
        num_classes=2,
        morphology_supervision="targeted",
        morphology_stages=(1,),
    )
    inputs = torch.randn(4, 1, 32, 64)
    _, morphology_logits = model(inputs, return_aux=True)
    morphology_logits.square().mean().backward()
    selected_block = model.features[2][-1]
    temporal_weights = selected_block.depthwise.branches[1].conv.edge_weights
    frequency_weights = selected_block.depthwise.branches[2].conv.edge_weights

    assert temporal_weights.grad is not None
    assert frequency_weights.grad is not None
    assert torch.isfinite(temporal_weights.grad).all()
    assert torch.isfinite(frequency_weights.grad).all()


def test_multiscale_hierarchy_is_anchored_routed_and_deploys_exactly() -> None:
    torch.manual_seed(17)
    model = TFDCRNet(
        mode="tfdcr",
        channels=(8, 12),
        depths=(1, 1),
        temporal_dilations=((1,), (1,)),
        expansion_ratio=2,
        num_classes=2,
        pooling="masked_mean",
        morphology_supervision="shared",
        morphology_stages=(0, 1),
        morphology_source="stage_output",
        morphology_routing="softmax",
        hierarchy_fusion="noisy_or",
    )
    inputs = torch.randn(3, 1, 32, 64)
    masks = torch.zeros(3, 64, dtype=torch.bool)
    masks[:, 5:57] = True
    model.train()
    for _ in range(3):
        model(inputs, frame_masks=masks)
    model.eval()

    anchored_logits = model(inputs, frame_masks=masks)
    routing = model.last_morphology_routing
    model.hierarchy_fusion = "none"
    base_logits = model(inputs, frame_masks=masks)
    model.hierarchy_fusion = "noisy_or"

    assert torch.equal(anchored_logits, base_logits)
    assert routing is not None
    assert torch.allclose(
        routing["transient"],
        torch.full_like(routing["transient"], 0.5),
    )
    assert torch.allclose(
        routing["continuous"],
        torch.full_like(routing["continuous"], 0.5),
    )

    with torch.no_grad():
        assert model.hierarchy_weight_logit is not None
        model.hierarchy_weight_logit.fill_(0.3)
    expected, morphology_logits = model(
        inputs,
        frame_masks=masks,
        return_aux=True,
    )
    converted = switch_model_to_deploy(model)
    actual, deployed_morphology_logits = converted(
        inputs,
        frame_masks=masks,
        return_aux=True,
    )

    assert morphology_logits.shape == (3, 2)
    assert converted.transient_heads is not None
    assert converted.continuous_heads is not None
    assert converted.transient_routers is not None
    assert converted.continuous_routers is not None
    assert (expected - actual).abs().max().item() < 1.0e-5
    assert (
        morphology_logits - deployed_morphology_logits
    ).abs().max().item() < 1.0e-5


def test_multiscale_morphology_loss_reaches_evidence_heads_and_routers() -> None:
    torch.manual_seed(18)
    model = TFDCRNet(
        mode="tfdcr",
        channels=(8, 12),
        depths=(1, 1),
        temporal_dilations=((1,), (1,)),
        expansion_ratio=2,
        num_classes=2,
        morphology_supervision="shared",
        morphology_stages=(0, 1),
        morphology_source="stage_output",
        morphology_routing="softmax",
        hierarchy_fusion="noisy_or",
    )
    inputs = torch.randn(4, 1, 32, 64)
    targets = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    _, morphology_logits = model(inputs, return_aux=True)
    torch.nn.functional.binary_cross_entropy_with_logits(
        morphology_logits,
        targets,
    ).backward()

    assert model.transient_heads is not None
    assert model.continuous_heads is not None
    assert model.transient_routers is not None
    assert model.continuous_routers is not None
    modules = (
        *model.transient_heads,
        *model.continuous_heads,
        *model.transient_routers,
        *model.continuous_routers,
    )
    for module in modules:
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()


def test_targeted_morphology_rejects_stage_output_source() -> None:
    with pytest.raises(
        ValueError,
        match="Targeted morphology supervision requires branch_response",
    ):
        TFDCRNet(
            mode="tfdcr",
            channels=(8, 12),
            depths=(1, 1),
            temporal_dilations=((1,), (1,)),
            expansion_ratio=2,
            num_classes=2,
            morphology_supervision="targeted",
            morphology_stages=(1,),
            morphology_source="stage_output",
        )


@pytest.mark.parametrize(
    "branch_types",
    [
        ("standard", "temporal"),
        ("standard", "frequency"),
    ],
)
def test_custom_ablation_branches_deploy_exactly(branch_types: tuple[str, ...]) -> None:
    torch.manual_seed(13)
    model = TFDCRNet(
        mode="tfdcr",
        channels=(8, 12),
        depths=(1, 1),
        temporal_dilations=((1,), (2,)),
        expansion_ratio=2,
        num_classes=4,
        branch_types=branch_types,
    )
    model.train()
    for _ in range(3):
        model(torch.randn(2, 1, 32, 64))
    model.eval()
    inputs = torch.randn(2, 1, 32, 64)
    expected = model(inputs)
    actual = switch_model_to_deploy(model)(inputs)
    assert (expected - actual).abs().max().item() < 1.0e-5
