"""Lightweight spectrogram backbone shared by Plain, RepConv, and TFDCR controls."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from respiratory_sound.models.reparameterization import ReparamDepthwiseConv2d


def branch_types_for_mode(mode: str, include_joint: bool = False) -> tuple[str, ...]:
    if mode == "plain":
        return ("standard",)
    if mode == "repconv":
        return ("standard", "standard", "standard")
    if mode == "tfdcr":
        branches = ("standard", "temporal", "frequency")
        return branches + (("joint",) if include_joint else ())
    raise ValueError(f"Unsupported model mode: {mode}")


class SpectrogramBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion_ratio: int,
        branch_types: tuple[str, ...],
        temporal_dilation: int,
        gate: AnchoredChannelGate | None = None,
    ) -> None:
        super().__init__()
        hidden_channels = channels * expansion_ratio
        self.depthwise = ReparamDepthwiseConv2d(
            channels=channels,
            branch_types=branch_types,
            dilation=(1, temporal_dilation),
        )
        self.gate = gate
        self.activation1 = nn.GELU()
        self.expand = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True)
        self.activation2 = nn.GELU()
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)

    def forward(
        self,
        inputs: Tensor,
        frame_masks: Tensor | None = None,
    ) -> Tensor:
        outputs = self.depthwise(inputs)
        return self._finish(inputs, outputs, frame_masks)

    def forward_with_branch_responses(
        self,
        inputs: Tensor,
        frame_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, ...]]:
        """Expose pre-fusion branch responses for training-only morphology supervision."""
        fused_response, branch_responses = self.depthwise.forward_branches(inputs)
        outputs = self._finish(inputs, fused_response, frame_masks)
        return outputs, fused_response, branch_responses

    def _finish(
        self,
        inputs: Tensor,
        outputs: Tensor,
        frame_masks: Tensor | None,
    ) -> Tensor:
        residual = inputs
        if self.gate is not None:
            outputs = self.gate(
                response=outputs,
                conditioning_features=inputs,
                frame_masks=frame_masks,
            )
        outputs = self.activation1(outputs)
        outputs = self.activation2(self.expand(outputs))
        outputs = self.project(outputs)
        return outputs + residual


class Downsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=(2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class EventPool2d(nn.Module):
    """Pool only time positions that originate from real (non-padding) audio."""

    def __init__(self, mode: str = "gap", lse_temperature: float = 1.0) -> None:
        super().__init__()
        if mode not in {"gap", "masked_mean", "masked_mean_lse"}:
            raise ValueError(f"Unsupported pooling mode: {mode}")
        if lse_temperature <= 0:
            raise ValueError("lse_temperature must be positive")
        self.mode = mode
        self.lse_temperature = float(lse_temperature)

    @property
    def output_multiplier(self) -> int:
        return 2 if self.mode == "masked_mean_lse" else 1

    def forward(self, inputs: Tensor, frame_masks: Tensor | None = None) -> Tensor:
        if self.mode == "gap":
            return F.adaptive_avg_pool2d(inputs, 1).flatten(1)

        batch_size, _, frequency_bins, time_bins = inputs.shape
        if frame_masks is None:
            frame_masks = torch.ones(
                batch_size,
                time_bins,
                dtype=torch.bool,
                device=inputs.device,
            )
        if frame_masks.ndim != 2 or frame_masks.shape[0] != batch_size:
            raise ValueError("frame_masks must have shape [batch, input_time_frames]")

        pooled_mask = F.adaptive_max_pool1d(
            frame_masks.to(dtype=inputs.dtype).unsqueeze(1),
            output_size=time_bins,
        )
        weights = pooled_mask.unsqueeze(2).expand(-1, 1, frequency_bins, -1)
        valid_count = weights.sum(dim=(2, 3)).clamp_min(1.0)
        masked_mean = (inputs * weights).sum(dim=(2, 3)) / valid_count
        if self.mode == "masked_mean":
            return masked_mean

        temperature = self.lse_temperature
        scaled = inputs / temperature
        valid = weights.expand(-1, inputs.shape[1], -1, -1).bool()
        masked_scaled = scaled.masked_fill(~valid, -torch.inf)
        normalized_lse = temperature * (
            torch.logsumexp(masked_scaled.flatten(2), dim=2) - valid_count.log()
        )
        return torch.cat((masked_mean, normalized_lse), dim=1)


class MaskAwareRelaxedFrequencyNorm(nn.Module):
    """Reduce frequency-response style while preserving event-level evidence."""

    def __init__(self, relaxation_weight: float = 0.5, eps: float = 1.0e-5) -> None:
        super().__init__()
        if not 0.0 <= relaxation_weight <= 1.0:
            raise ValueError("relaxation_weight must be in [0, 1]")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.relaxation_weight = float(relaxation_weight)
        self.eps = float(eps)

    def forward(
        self,
        inputs: Tensor,
        frame_masks: Tensor | None = None,
    ) -> Tensor:
        if inputs.ndim != 4:
            raise ValueError("inputs must have shape [batch, channels, frequency, time]")
        if frame_masks is None:
            frame_masks = torch.ones(
                inputs.shape[0],
                inputs.shape[-1],
                dtype=torch.bool,
                device=inputs.device,
            )
        if frame_masks.ndim != 2 or frame_masks.shape != (
            inputs.shape[0],
            inputs.shape[-1],
        ):
            raise ValueError("frame_masks must have shape [batch, input_time_frames]")

        weights = frame_masks.to(dtype=inputs.dtype).unsqueeze(1).unsqueeze(1)
        valid_count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (inputs * weights).sum(dim=-1, keepdim=True) / valid_count
        variance = (
            (inputs - mean).square() * weights
        ).sum(dim=-1, keepdim=True) / valid_count
        normalized = (inputs - mean) * torch.rsqrt(variance + self.eps)
        outputs = torch.lerp(inputs, normalized, self.relaxation_weight)
        return outputs.masked_fill(~frame_masks[:, None, None, :], 0.0)


def masked_spatial_mean(inputs: Tensor, frame_masks: Tensor | None) -> Tensor:
    """Return a channel descriptor without averaging padded time positions."""
    if frame_masks is None:
        return inputs.mean(dim=(2, 3))
    if frame_masks.ndim != 2 or frame_masks.shape[0] != inputs.shape[0]:
        raise ValueError("frame_masks must have shape [batch, input_time_frames]")
    pooled_mask = F.adaptive_max_pool1d(
        frame_masks.to(dtype=inputs.dtype).unsqueeze(1),
        output_size=inputs.shape[-1],
    )
    weights = pooled_mask.unsqueeze(2).expand(-1, 1, inputs.shape[2], -1)
    valid_count = weights.sum(dim=(2, 3)).clamp_min(1.0)
    return (inputs * weights).sum(dim=(2, 3)) / valid_count


class AnchoredChannelGate(nn.Module):
    """Calibrate a folded response while starting exactly from the identity map."""

    def __init__(
        self,
        channels: int,
        conditioning: str = "event",
        reduction: int = 4,
        max_delta: float = 0.5,
    ) -> None:
        super().__init__()
        if conditioning not in {"event", "constant"}:
            raise ValueError(f"Unsupported gate conditioning: {conditioning}")
        if reduction < 1:
            raise ValueError("reduction must be positive")
        if not 0 < max_delta <= 1:
            raise ValueError("max_delta must be in (0, 1]")
        hidden_channels = max(8, channels // reduction)
        self.conditioning = conditioning
        self.max_delta = float(max_delta)
        self.reduce = nn.Linear(channels, hidden_channels)
        self.activation = nn.SiLU()
        self.expand = nn.Linear(hidden_channels, channels)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        self.last_scale: Tensor | None = None

    def forward(
        self,
        response: Tensor,
        conditioning_features: Tensor,
        frame_masks: Tensor | None = None,
    ) -> Tensor:
        descriptor = masked_spatial_mean(conditioning_features, frame_masks)
        if self.conditioning == "constant":
            descriptor = torch.ones_like(descriptor)
        logits = self.expand(self.activation(self.reduce(descriptor)))
        scale = 1.0 + self.max_delta * torch.tanh(logits)
        self.last_scale = scale.detach()
        return response * scale.unsqueeze(-1).unsqueeze(-1)


class TFDCRNet(nn.Module):
    """Four-stage network with identical deployment topology across control modes."""

    def __init__(
        self,
        mode: str = "tfdcr",
        input_channels: int = 1,
        channels: Sequence[int] = (24, 48, 96, 160),
        depths: Sequence[int] = (2, 2, 3, 2),
        temporal_dilations: Sequence[Sequence[int]] = (
            (1, 1),
            (1, 2),
            (1, 2, 4),
            (1, 2),
        ),
        expansion_ratio: int = 2,
        num_classes: int = 4,
        include_joint: bool = False,
        branch_types: Sequence[str] | None = None,
        pooling: str = "gap",
        lse_temperature: float = 1.0,
        gate_conditioning: str = "none",
        gate_stages: Sequence[int] = (),
        gate_reduction: int = 4,
        gate_max_delta: float = 0.5,
        morphology_supervision: str = "none",
        morphology_stages: Sequence[int] = (),
        morphology_branch_indices: Sequence[int] = (1, 2),
        morphology_source: str = "branch_response",
        morphology_routing: str = "mean",
        hierarchy_fusion: str = "none",
        hierarchy_max_weight: float = 1.0,
        input_normalization: str = "none",
        relaxed_frequency_weight: float = 0.5,
        normalization_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        if not (len(channels) == len(depths) == len(temporal_dilations)):
            raise ValueError("channels, depths, and temporal_dilations must align")
        for depth, dilations in zip(depths, temporal_dilations, strict=True):
            if depth != len(dilations):
                raise ValueError("Each stage needs one temporal dilation per block")

        selected_branch_types = (
            tuple(branch_types)
            if branch_types is not None
            else branch_types_for_mode(mode, include_joint=include_joint)
        )
        if not selected_branch_types:
            raise ValueError("At least one branch type is required")
        if gate_conditioning not in {"none", "event", "constant"}:
            raise ValueError(f"Unsupported gate_conditioning: {gate_conditioning}")
        selected_gate_stages = {int(stage) for stage in gate_stages}
        if any(stage < 0 or stage >= len(channels) for stage in selected_gate_stages):
            raise ValueError("gate_stages contains an invalid zero-based stage index")
        if gate_conditioning == "none" and selected_gate_stages:
            raise ValueError("gate_stages requires event or constant conditioning")
        if morphology_supervision not in {"none", "shared", "targeted"}:
            raise ValueError(
                f"Unsupported morphology_supervision: {morphology_supervision}"
            )
        if morphology_source not in {"branch_response", "stage_output"}:
            raise ValueError(f"Unsupported morphology_source: {morphology_source}")
        if morphology_routing not in {"mean", "softmax"}:
            raise ValueError(f"Unsupported morphology_routing: {morphology_routing}")
        if hierarchy_fusion not in {"none", "noisy_or"}:
            raise ValueError(f"Unsupported hierarchy_fusion: {hierarchy_fusion}")
        if input_normalization not in {"none", "relaxed_frequency"}:
            raise ValueError(f"Unsupported input_normalization: {input_normalization}")
        if not 0 < hierarchy_max_weight <= 1:
            raise ValueError("hierarchy_max_weight must be in (0, 1]")
        selected_morphology_stages = tuple(int(stage) for stage in morphology_stages)
        if len(set(selected_morphology_stages)) != len(selected_morphology_stages):
            raise ValueError("morphology_stages must not contain duplicates")
        if any(
            stage < 0 or stage >= len(channels)
            for stage in selected_morphology_stages
        ):
            raise ValueError("morphology_stages contains an invalid zero-based stage index")
        if morphology_supervision == "none" and selected_morphology_stages:
            raise ValueError("morphology_stages requires shared or targeted supervision")
        if morphology_supervision != "none" and not selected_morphology_stages:
            raise ValueError("Morphology supervision requires at least one stage")
        if morphology_supervision == "targeted" and morphology_source != "branch_response":
            raise ValueError("Targeted morphology supervision requires branch_response")
        if morphology_routing == "softmax" and len(selected_morphology_stages) < 2:
            raise ValueError("Softmax morphology routing requires at least two stages")
        if hierarchy_fusion != "none" and morphology_supervision == "none":
            raise ValueError("Hierarchy fusion requires morphology supervision")
        if hierarchy_fusion != "none" and num_classes != 2:
            raise ValueError("Hierarchy fusion currently requires binary classification")
        selected_branch_indices = tuple(int(index) for index in morphology_branch_indices)
        if len(selected_branch_indices) != 2:
            raise ValueError(
                "morphology_branch_indices must contain transient and continuous indices"
            )
        if morphology_supervision == "targeted" and any(
            index < 0 or index >= len(selected_branch_types)
            for index in selected_branch_indices
        ):
            raise ValueError("morphology_branch_indices contains an invalid branch index")
        self.mode = mode
        self.morphology_supervision = morphology_supervision
        self.morphology_stages = selected_morphology_stages
        self.morphology_branch_indices = selected_branch_indices
        self.morphology_source = morphology_source
        self.morphology_routing = morphology_routing
        self.hierarchy_fusion = hierarchy_fusion
        self.hierarchy_max_weight = float(hierarchy_max_weight)
        self.last_morphology_routing: dict[str, Tensor] | None = None
        self.last_hierarchy_weight: Tensor | None = None
        self.input_normalizer: MaskAwareRelaxedFrequencyNorm | None = (
            MaskAwareRelaxedFrequencyNorm(
                relaxation_weight=relaxed_frequency_weight,
                eps=normalization_eps,
            )
            if input_normalization == "relaxed_frequency"
            else None
        )
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                channels[0],
                kernel_size=3,
                stride=(2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels[0]),
            nn.GELU(),
        )

        stages: list[nn.Module] = []
        for stage_index, (stage_channels, stage_dilations) in enumerate(
            zip(channels, temporal_dilations, strict=True)
        ):
            if stage_index > 0:
                stages.append(Downsample(channels[stage_index - 1], stage_channels))
            stages.append(
                nn.Sequential(
                    *[
                        SpectrogramBlock(
                            channels=stage_channels,
                            expansion_ratio=expansion_ratio,
                            branch_types=selected_branch_types,
                            temporal_dilation=temporal_dilation,
                            gate=(
                                AnchoredChannelGate(
                                    channels=stage_channels,
                                    conditioning=gate_conditioning,
                                    reduction=gate_reduction,
                                    max_delta=gate_max_delta,
                                )
                                if stage_index in selected_gate_stages
                                else None
                            ),
                        )
                        for temporal_dilation in stage_dilations
                    ]
                )
            )
        self.features = nn.Sequential(*stages)
        self.pool = EventPool2d(mode=pooling, lse_temperature=lse_temperature)
        self.classifier = nn.Linear(
            channels[-1] * self.pool.output_multiplier,
            num_classes,
        )
        if morphology_supervision == "none":
            self.transient_heads: nn.ModuleList | None = None
            self.continuous_heads: nn.ModuleList | None = None
            self.transient_routers: nn.ModuleList | None = None
            self.continuous_routers: nn.ModuleList | None = None
        else:
            self.transient_heads = nn.ModuleList(
                nn.Linear(channels[stage], 1) for stage in selected_morphology_stages
            )
            self.continuous_heads = nn.ModuleList(
                nn.Linear(channels[stage], 1) for stage in selected_morphology_stages
            )
            if morphology_routing == "softmax":
                self.transient_routers = nn.ModuleList(
                    nn.Linear(channels[stage], 1)
                    for stage in selected_morphology_stages
                )
                self.continuous_routers = nn.ModuleList(
                    nn.Linear(channels[stage], 1)
                    for stage in selected_morphology_stages
                )
                for router in (*self.transient_routers, *self.continuous_routers):
                    nn.init.zeros_(router.weight)
                    nn.init.zeros_(router.bias)
            else:
                self.transient_routers = None
                self.continuous_routers = None
        self.hierarchy_weight_logit = (
            nn.Parameter(torch.zeros(())) if hierarchy_fusion == "noisy_or" else None
        )

    def forward_features(
        self,
        inputs: Tensor,
        frame_masks: Tensor | None = None,
    ) -> Tensor:
        """Return the final feature map without event pooling or classification."""
        if self.input_normalizer is not None:
            inputs = self.input_normalizer(inputs, frame_masks=frame_masks)
        outputs = self.stem(inputs)
        for module in self.features:
            if isinstance(module, Downsample):
                outputs = module(outputs)
            else:
                for block in module:
                    outputs = block(outputs, frame_masks=frame_masks)
        return outputs

    def forward(
        self,
        inputs: Tensor,
        frame_masks: Tensor | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        needs_morphology = return_aux or self.hierarchy_fusion == "noisy_or"
        if not needs_morphology:
            outputs = self.forward_features(inputs, frame_masks=frame_masks)
            pooled = self.pool(outputs, frame_masks=frame_masks)
            return self.classifier(pooled)

        if self.input_normalizer is not None:
            inputs = self.input_normalizer(inputs, frame_masks=frame_masks)
        outputs = self.stem(inputs)
        morphology_responses: dict[
            int, tuple[Tensor, Tensor | None, tuple[Tensor, ...]]
        ] = {}
        stage_index = -1
        for module in self.features:
            if isinstance(module, Downsample):
                outputs = module(outputs)
            else:
                stage_index += 1
                for block_index, block in enumerate(module):
                    is_selected_output = (
                        needs_morphology
                        and stage_index in self.morphology_stages
                        and block_index == len(module) - 1
                    )
                    if is_selected_output and self.morphology_source == "branch_response":
                        outputs, fused, branches = block.forward_with_branch_responses(
                            outputs,
                            frame_masks=frame_masks,
                        )
                        morphology_responses[stage_index] = (
                            outputs,
                            fused,
                            branches,
                        )
                    else:
                        outputs = block(outputs, frame_masks=frame_masks)
                        if is_selected_output:
                            morphology_responses[stage_index] = (
                                outputs,
                                None,
                                (),
                            )
        pooled = self.pool(outputs, frame_masks=frame_masks)
        logits = self.classifier(pooled)
        morphology_logits = (
            self._morphology_logits(morphology_responses, frame_masks)
            if needs_morphology
            else None
        )
        if self.hierarchy_fusion == "noisy_or":
            if morphology_logits is None:
                raise RuntimeError("Hierarchy fusion is missing morphology evidence")
            logits = self._apply_hierarchy_fusion(logits, morphology_logits)
        if not return_aux:
            return logits
        if morphology_logits is None:
            raise RuntimeError("Auxiliary output is missing morphology evidence")
        return logits, morphology_logits

    def _morphology_logits(
        self,
        responses: dict[int, tuple[Tensor, Tensor | None, tuple[Tensor, ...]]],
        frame_masks: Tensor | None,
    ) -> Tensor:
        if (
            self.morphology_supervision == "none"
            or self.transient_heads is None
            or self.continuous_heads is None
        ):
            raise RuntimeError("Morphology auxiliary heads are not configured")
        if set(responses) != set(self.morphology_stages):
            raise RuntimeError("Missing morphology response from a selected stage")

        transient_logits: list[Tensor] = []
        continuous_logits: list[Tensor] = []
        transient_router_logits: list[Tensor] = []
        continuous_router_logits: list[Tensor] = []
        transient_index, continuous_index = self.morphology_branch_indices
        for head_index, stage in enumerate(self.morphology_stages):
            stage_output, fused, branches = responses[stage]
            if self.morphology_source == "stage_output":
                transient_source = stage_output
                continuous_source = stage_output
            elif self.morphology_supervision == "shared":
                if fused is None:
                    raise RuntimeError("Shared morphology response is missing")
                transient_source = fused
                continuous_source = fused
            else:
                transient_source = branches[transient_index]
                continuous_source = branches[continuous_index]
            # The signed finite-difference response would cancel under a raw
            # global mean. Apply the block's parameter-free nonlinearity to
            # shared and targeted sources identically before masked pooling.
            transient_descriptor = masked_spatial_mean(
                F.gelu(transient_source),
                frame_masks,
            )
            continuous_descriptor = masked_spatial_mean(
                F.gelu(continuous_source),
                frame_masks,
            )
            transient_logits.append(
                self.transient_heads[head_index](transient_descriptor).squeeze(-1)
            )
            continuous_logits.append(
                self.continuous_heads[head_index](continuous_descriptor).squeeze(-1)
            )
            if self.morphology_routing == "softmax":
                if self.transient_routers is None or self.continuous_routers is None:
                    raise RuntimeError("Softmax morphology routers are not configured")
                transient_router_logits.append(
                    self.transient_routers[head_index](transient_descriptor).squeeze(-1)
                )
                continuous_router_logits.append(
                    self.continuous_routers[head_index](continuous_descriptor).squeeze(-1)
                )

        stacked_transient = torch.stack(transient_logits, dim=1)
        stacked_continuous = torch.stack(continuous_logits, dim=1)
        if self.morphology_routing == "mean":
            transient_output = stacked_transient.mean(dim=1)
            continuous_output = stacked_continuous.mean(dim=1)
            self.last_morphology_routing = None
        else:
            transient_weights = torch.softmax(
                torch.stack(transient_router_logits, dim=1),
                dim=1,
            )
            continuous_weights = torch.softmax(
                torch.stack(continuous_router_logits, dim=1),
                dim=1,
            )
            transient_output = (stacked_transient * transient_weights).sum(dim=1)
            continuous_output = (stacked_continuous * continuous_weights).sum(dim=1)
            self.last_morphology_routing = {
                "transient": transient_weights.detach(),
                "continuous": continuous_weights.detach(),
            }
        return torch.stack((transient_output, continuous_output), dim=1)

    def hierarchy_weight(self) -> Tensor:
        """Return the bounded reliability assigned to fine-label hierarchy evidence."""
        if self.hierarchy_weight_logit is None:
            raise RuntimeError("Hierarchy fusion is not configured")
        return self.hierarchy_max_weight * torch.tanh(self.hierarchy_weight_logit)

    def _apply_hierarchy_fusion(
        self,
        class_logits: Tensor,
        morphology_logits: Tensor,
    ) -> Tensor:
        probabilities = torch.sigmoid(morphology_logits)
        probability_or = 1.0 - (1.0 - probabilities[:, 0]) * (
            1.0 - probabilities[:, 1]
        )
        hierarchy_logit = torch.logit(probability_or.clamp(1.0e-5, 1.0 - 1.0e-5))
        weight = self.hierarchy_weight()
        self.last_hierarchy_weight = weight.detach()
        residual_margin = weight * hierarchy_logit
        residual = torch.stack(
            (-0.5 * residual_margin, 0.5 * residual_margin),
            dim=1,
        )
        return class_logits + residual

    def remove_training_heads(self) -> None:
        """Discard morphology heads after training without changing class predictions."""
        if self.hierarchy_fusion != "none":
            return
        self.transient_heads = None
        self.continuous_heads = None
        self.transient_routers = None
        self.continuous_routers = None
        self.morphology_supervision = "none"
        self.morphology_stages = ()
