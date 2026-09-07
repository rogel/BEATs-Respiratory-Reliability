"""Locally reliable waveform-spectrogram alignment with a temporal control."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from respiratory_sound.models.network import TFDCRNet
from respiratory_sound.models.wavespec import LightweightWaveformEncoder


def downsample_frame_mask(
    frame_masks: Tensor | None,
    batch_size: int,
    output_steps: int,
    device: torch.device,
) -> Tensor:
    if frame_masks is None:
        return torch.ones(
            batch_size,
            output_steps,
            dtype=torch.bool,
            device=device,
        )
    if frame_masks.ndim != 2 or frame_masks.shape[0] != batch_size:
        raise ValueError("frame_masks must have shape [batch, input_frames]")
    return (
        F.adaptive_max_pool1d(
            frame_masks.to(dtype=torch.float32).unsqueeze(1),
            output_size=output_steps,
        )
        .squeeze(1)
        .bool()
    )


def masked_resample_tokens(
    features: Tensor,
    masks: Tensor,
    output_steps: int,
) -> tuple[Tensor, Tensor]:
    """Resample [B,C,T] features without averaging invalid padded positions."""
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, channels, time]")
    if masks.shape != (features.shape[0], features.shape[-1]):
        raise ValueError("masks must align with feature time steps")
    weights = masks.to(dtype=features.dtype).unsqueeze(1)
    input_steps = features.shape[-1]
    boundaries = torch.div(
        torch.arange(
            output_steps + 1,
            device=features.device,
            dtype=torch.long,
        )
        * input_steps,
        output_steps,
        rounding_mode="floor",
    )
    starts = boundaries[:-1]
    ends = boundaries[1:]
    feature_prefix = F.pad(
        torch.cumsum(features * weights, dim=-1),
        (1, 0),
    )
    weight_prefix = F.pad(torch.cumsum(weights, dim=-1), (1, 0))
    numerator = feature_prefix.index_select(-1, ends) - feature_prefix.index_select(
        -1,
        starts,
    )
    denominator = weight_prefix.index_select(-1, ends) - weight_prefix.index_select(
        -1,
        starts,
    )
    outputs = numerator / denominator.clamp_min(1.0)
    output_masks = denominator.squeeze(1) > 0.0
    return outputs.transpose(1, 2), output_masks


def reverse_valid_tokens(tokens: Tensor, masks: Tensor) -> Tensor:
    """Reverse token order only within valid positions, preserving mask and multiset."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [batch, time, channels]")
    if masks.shape != tokens.shape[:2]:
        raise ValueError("masks must have shape [batch, time]")
    outputs = tokens.clone()
    for batch_index in range(tokens.shape[0]):
        valid_indices = masks[batch_index].nonzero(as_tuple=False).flatten()
        if valid_indices.numel() > 1:
            outputs[batch_index, valid_indices] = tokens[
                batch_index,
                valid_indices.flip(0),
            ]
    return outputs


class LocalInteractionHead(nn.Module):
    def __init__(self, input_features: int, hidden_features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs).squeeze(-1)


class LRACNet(nn.Module):
    """Fuse synchronized local evidence and expose a same-content reversed control."""

    def __init__(
        self,
        alignment_mode: str,
        spectrum_channels: Sequence[int] = (24, 48, 96, 160),
        spectrum_depths: Sequence[int] = (2, 2, 3, 2),
        spectrum_temporal_dilations: Sequence[Sequence[int]] = (
            (1, 1),
            (1, 1),
            (1, 1, 1),
            (1, 1),
        ),
        spectrum_expansion_ratio: int = 2,
        waveform_channels: Sequence[int] = (24, 48, 64, 96, 128),
        projection_dimension: int = 32,
        interaction_hidden: int = 32,
        max_residual_margin: float = 2.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if alignment_mode not in {"aligned", "reverse_control"}:
            raise ValueError(f"Unsupported alignment_mode: {alignment_mode}")
        if num_classes != 2:
            raise ValueError("LRAC evidence correction currently requires two classes")
        if projection_dimension < 1 or interaction_hidden < 1:
            raise ValueError("Projection and interaction widths must be positive")
        if max_residual_margin <= 0:
            raise ValueError("max_residual_margin must be positive")
        self.alignment_mode = alignment_mode
        self.max_residual_margin = float(max_residual_margin)
        self.spectrum = TFDCRNet(
            mode="plain",
            input_channels=1,
            channels=spectrum_channels,
            depths=spectrum_depths,
            temporal_dilations=spectrum_temporal_dilations,
            expansion_ratio=spectrum_expansion_ratio,
            num_classes=num_classes,
            pooling="masked_mean",
        )
        self.waveform = LightweightWaveformEncoder(
            channels=waveform_channels,
            num_classes=num_classes,
        )
        self.spectrum_projection = nn.Sequential(
            nn.Linear(spectrum_channels[-1], projection_dimension),
            nn.LayerNorm(projection_dimension),
        )
        self.waveform_projection = nn.Sequential(
            nn.Linear(waveform_channels[-1], projection_dimension),
            nn.LayerNorm(projection_dimension),
        )
        interaction_features = 4 * projection_dimension
        self.reliability = LocalInteractionHead(
            interaction_features,
            interaction_hidden,
        )
        self.correction = LocalInteractionHead(
            interaction_features,
            interaction_hidden,
        )
        correction_output = self.correction.network[-1]
        if not isinstance(correction_output, nn.Linear):
            raise RuntimeError("Correction output layer is not linear")
        nn.init.zeros_(correction_output.weight)
        nn.init.zeros_(correction_output.bias)

    @staticmethod
    def _interaction(spectrum_tokens: Tensor, waveform_tokens: Tensor) -> Tensor:
        return torch.cat(
            (
                spectrum_tokens,
                waveform_tokens,
                (spectrum_tokens - waveform_tokens).abs(),
                spectrum_tokens * waveform_tokens,
            ),
            dim=-1,
        )

    def _local_tokens(
        self,
        spectrograms: Tensor,
        waveforms: Tensor,
        frame_masks: Tensor | None,
        sample_masks: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        spectrum_features = self.spectrum.forward_features(
            spectrograms,
            frame_masks=frame_masks,
        )
        spectrum_logits = self.spectrum.classifier(
            self.spectrum.pool(spectrum_features, frame_masks=frame_masks)
        )
        spectrum_tokens = spectrum_features.mean(dim=2).transpose(1, 2)
        spectrum_masks = downsample_frame_mask(
            frame_masks,
            batch_size=spectrum_tokens.shape[0],
            output_steps=spectrum_tokens.shape[1],
            device=spectrum_tokens.device,
        )

        waveform_features, waveform_feature_masks = self.waveform.forward_features(
            waveforms,
            sample_masks=sample_masks,
        )
        waveform_descriptor = torch.cat(
            (
                (
                    waveform_features
                    * waveform_feature_masks.to(waveform_features.dtype).unsqueeze(1)
                ).sum(dim=-1)
                / waveform_feature_masks.sum(dim=-1, keepdim=True)
                .to(waveform_features.dtype)
                .clamp_min(1.0),
                waveform_features.masked_fill(
                    ~waveform_feature_masks.unsqueeze(1),
                    -torch.inf,
                ).amax(dim=-1),
            ),
            dim=1,
        )
        waveform_logits = self.waveform.classifier(waveform_descriptor)
        waveform_tokens, waveform_masks = masked_resample_tokens(
            waveform_features,
            waveform_feature_masks,
            output_steps=spectrum_tokens.shape[1],
        )
        common_masks = spectrum_masks & waveform_masks
        if not bool(common_masks.any(dim=1).all()):
            raise RuntimeError("At least one sample has no common valid local positions")
        return (
            spectrum_logits,
            waveform_logits,
            spectrum_tokens,
            waveform_tokens,
            common_masks,
        )

    def forward(
        self,
        spectrograms: Tensor,
        waveforms: Tensor,
        frame_masks: Tensor | None = None,
        sample_masks: Tensor | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        (
            spectrum_logits,
            waveform_logits,
            raw_spectrum_tokens,
            raw_waveform_tokens,
            common_masks,
        ) = self._local_tokens(
            spectrograms,
            waveforms,
            frame_masks,
            sample_masks,
        )
        spectrum_tokens = self.spectrum_projection(raw_spectrum_tokens)
        waveform_tokens = self.waveform_projection(raw_waveform_tokens)
        reversed_waveform_tokens = reverse_valid_tokens(
            waveform_tokens,
            common_masks,
        )
        aligned_interaction = self._interaction(
            spectrum_tokens,
            waveform_tokens,
        )
        reversed_interaction = self._interaction(
            spectrum_tokens,
            reversed_waveform_tokens,
        )
        classification_interaction = (
            aligned_interaction
            if self.alignment_mode == "aligned"
            else reversed_interaction
        )
        reliability_logits = self.reliability(classification_interaction)
        reliability_gates = torch.sigmoid(reliability_logits)
        local_corrections = torch.tanh(self.correction(classification_interaction))
        valid_weights = common_masks.to(dtype=local_corrections.dtype)
        residual_margin = self.max_residual_margin * (
            reliability_gates * local_corrections * valid_weights
        ).sum(dim=1) / valid_weights.sum(dim=1).clamp_min(1.0)
        final_logits = spectrum_logits + torch.stack(
            (-0.5 * residual_margin, 0.5 * residual_margin),
            dim=1,
        )

        # Re-project detached backbone features so the auxiliary alignment loss
        # trains the shared projections/reliability head without directly moving
        # either backbone.
        detached_spectrum = self.spectrum_projection(raw_spectrum_tokens.detach())
        detached_waveform = self.waveform_projection(raw_waveform_tokens.detach())
        detached_reversed = reverse_valid_tokens(
            detached_waveform,
            common_masks,
        )
        aligned_reliability_logits = self.reliability(
            self._interaction(detached_spectrum, detached_waveform)
        )
        reversed_reliability_logits = self.reliability(
            self._interaction(detached_spectrum, detached_reversed)
        )
        if return_aux:
            return (
                final_logits,
                spectrum_logits,
                waveform_logits,
                residual_margin,
                aligned_reliability_logits,
                reversed_reliability_logits,
                common_masks,
                reliability_gates,
            )
        return final_logits
