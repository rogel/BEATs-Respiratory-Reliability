"""Waveform-spectrogram evidence fusion with a parameter-matched control."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from respiratory_sound.models.network import TFDCRNet


class WaveformResidualBlock(nn.Module):
    """Depthwise-separable temporal downsampling with a learned residual path."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        padding = kernel_size // 2
        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.main(inputs) + self.skip(inputs))


def masked_temporal_statistics(inputs: Tensor, sample_masks: Tensor | None) -> Tensor:
    """Concatenate mean and max over positions descended from real audio."""
    batch_size, channels, time_steps = inputs.shape
    if sample_masks is None:
        return torch.cat((inputs.mean(dim=-1), inputs.amax(dim=-1)), dim=1)
    if sample_masks.ndim != 2 or sample_masks.shape[0] != batch_size:
        raise ValueError("sample_masks must have shape [batch, input_samples]")
    pooled_mask = F.adaptive_max_pool1d(
        sample_masks.to(dtype=inputs.dtype).unsqueeze(1),
        output_size=time_steps,
    ).bool()
    weights = pooled_mask.to(dtype=inputs.dtype)
    valid_count = weights.sum(dim=-1).clamp_min(1.0)
    mean = (inputs * weights).sum(dim=-1) / valid_count
    maximum = inputs.masked_fill(~pooled_mask.expand(-1, channels, -1), -torch.inf).amax(
        dim=-1
    )
    return torch.cat((mean, maximum), dim=1)


class LightweightWaveformEncoder(nn.Module):
    """Aggressively strided waveform encoder sized for local MPS experiments."""

    def __init__(
        self,
        channels: Sequence[int] = (24, 48, 64, 96, 128),
        stem_kernel: int = 80,
        stem_stride: int = 40,
        block_kernel: int = 9,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("At least two waveform channel widths are required")
        self.stem = nn.Sequential(
            nn.Conv1d(
                1,
                channels[0],
                kernel_size=stem_kernel,
                stride=stem_stride,
                padding=(stem_kernel - stem_stride) // 2,
                bias=False,
            ),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                WaveformResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=block_kernel,
                )
                for in_channels, out_channels in zip(
                    channels[:-1],
                    channels[1:],
                    strict=True,
                )
            ]
        )
        self.classifier = nn.Linear(2 * channels[-1], num_classes)

    def forward_features(
        self,
        waveforms: Tensor,
        sample_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return the final temporal features and their downsampled validity mask."""
        if waveforms.ndim != 3 or waveforms.shape[1] != 1:
            raise ValueError("waveforms must have shape [batch, 1, samples]")
        outputs = self.blocks(self.stem(waveforms))
        if sample_masks is None:
            output_masks = torch.ones(
                outputs.shape[0],
                outputs.shape[-1],
                dtype=torch.bool,
                device=outputs.device,
            )
        else:
            if sample_masks.ndim != 2 or sample_masks.shape[0] != outputs.shape[0]:
                raise ValueError("sample_masks must have shape [batch, input_samples]")
            output_masks = F.adaptive_max_pool1d(
                sample_masks.to(dtype=outputs.dtype).unsqueeze(1),
                output_size=outputs.shape[-1],
            ).squeeze(1).bool()
        return outputs, output_masks

    def forward(self, waveforms: Tensor, sample_masks: Tensor | None = None) -> Tensor:
        outputs, output_masks = self.forward_features(waveforms, sample_masks)
        descriptor = masked_temporal_statistics(outputs, output_masks)
        return self.classifier(descriptor)


class WaveSpecEvidenceFusion(nn.Module):
    """Anchor predictions to Log-Mel evidence and learn a bounded waveform residual."""

    def __init__(
        self,
        fusion_mode: str,
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
        fusion_hidden: int = 16,
        max_residual_margin: float = 2.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"concat_control", "interaction"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        if num_classes != 2:
            raise ValueError("Evidence-margin fusion currently requires two classes")
        if max_residual_margin <= 0:
            raise ValueError("max_residual_margin must be positive")
        self.fusion_mode = fusion_mode
        self.max_residual_margin = float(max_residual_margin)
        # Construct the anchor first so identical seeds produce identical anchor weights
        # across the parameter-matched control and candidate.
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
        self.fusion = nn.Sequential(
            nn.Linear(4, fusion_hidden),
            nn.SiLU(),
            nn.Linear(fusion_hidden, 1),
        )
        output_layer = self.fusion[-1]
        if not isinstance(output_layer, nn.Linear):
            raise RuntimeError("Fusion output layer is not linear")
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        self.last_residual_margin: Tensor | None = None
        self.last_evidence_features: Tensor | None = None

    def _evidence_features(self, spectrum_logits: Tensor, waveform_logits: Tensor) -> Tensor:
        spectrum_margin = spectrum_logits[:, 1] - spectrum_logits[:, 0]
        waveform_margin = waveform_logits[:, 1] - waveform_logits[:, 0]
        spectrum_evidence = torch.tanh(0.5 * spectrum_margin)
        waveform_evidence = torch.tanh(0.5 * waveform_margin)
        if self.fusion_mode == "interaction":
            disagreement = (spectrum_evidence - waveform_evidence).abs()
            agreement = spectrum_evidence * waveform_evidence
        else:
            disagreement = torch.zeros_like(spectrum_evidence)
            agreement = torch.zeros_like(spectrum_evidence)
        return torch.stack(
            (spectrum_evidence, waveform_evidence, disagreement, agreement),
            dim=1,
        )

    def forward(
        self,
        spectrograms: Tensor,
        waveforms: Tensor,
        frame_masks: Tensor | None = None,
        sample_masks: Tensor | None = None,
        return_aux: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        spectrum_logits = self.spectrum(spectrograms, frame_masks=frame_masks)
        waveform_logits = self.waveform(waveforms, sample_masks=sample_masks)
        evidence_features = self._evidence_features(spectrum_logits, waveform_logits)
        residual_margin = self.max_residual_margin * torch.tanh(
            self.fusion(evidence_features).squeeze(1)
        )
        final_logits = spectrum_logits + torch.stack(
            (-0.5 * residual_margin, 0.5 * residual_margin),
            dim=1,
        )
        self.last_residual_margin = residual_margin.detach()
        self.last_evidence_features = evidence_features.detach()
        if return_aux:
            return (
                final_logits,
                spectrum_logits,
                waveform_logits,
                residual_margin,
                evidence_features,
            )
        return final_logits
