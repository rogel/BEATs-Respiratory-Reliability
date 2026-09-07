"""Gate 9A adapters for published AudioSet-pretrained audio encoders."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import load_file
from torch import Tensor, nn
from torch.nn import functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


def _init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    bias = getattr(layer, "bias", None)
    if bias is not None:
        bias.data.zero_()


def _init_bn(layer: nn.modules.batchnorm._BatchNorm) -> None:
    layer.bias.data.zero_()
    layer.weight.data.fill_(1.0)


class ConvBlock5x5(nn.Module):
    """Weight-compatible copy of the convolution block used by PANNs CNN6."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        _init_layer(self.conv1)
        _init_bn(self.bn1)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = F.relu_(self.bn1(self.conv1(inputs)))
        return F.avg_pool2d(outputs, kernel_size=(2, 2))


def _pool_mask(mask: Tensor) -> Tensor:
    return F.max_pool1d(mask[:, None].float(), kernel_size=2, stride=2)[:, 0].bool()


def _masked_max_mean(inputs: Tensor, mask: Tensor) -> Tensor:
    """Pool [batch, channels, time] without allowing zero padding to vote."""
    if mask.shape != (inputs.shape[0], inputs.shape[2]):
        raise ValueError("Temporal mask does not match CNN feature length")
    weights = mask[:, None].to(dtype=inputs.dtype)
    mean = (inputs * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    maximum = inputs.masked_fill(~mask[:, None], torch.finfo(inputs.dtype).min).max(dim=2).values
    return maximum + mean


def _beats_token_valid_mask(
    sample_masks: Tensor,
    *,
    fbank_frames: int,
    patch_size: int,
    frequency_patches: int,
) -> Tensor:
    """Map arbitrary-position valid samples to flattened BEATs patch tokens."""
    frame_valid = F.max_pool1d(
        sample_masks[:, None].to(dtype=torch.float32),
        kernel_size=400,
        stride=160,
    )[:, 0].to(torch.bool)
    if frame_valid.shape[1] != fbank_frames:
        raise RuntimeError("Exact BEATs frame mask does not match fbank length")
    extra_frames = (-fbank_frames) % patch_size
    if extra_frames:
        frame_valid = F.pad(frame_valid, (0, extra_frames), value=False)
    patch_valid = F.max_pool1d(
        frame_valid[:, None].to(dtype=torch.float32),
        kernel_size=patch_size,
        stride=patch_size,
    )[:, 0].to(torch.bool)
    token_valid = (
        patch_valid[:, :, None]
        .expand(-1, -1, frequency_patches)
        .reshape(patch_valid.shape[0], -1)
    )
    if bool((~token_valid.any(dim=1)).any()):
        raise ValueError("Non-empty waveform became all-padding BEATs tokens")
    return token_valid


def _beats_token_support_fraction(
    sample_masks: Tensor,
    *,
    fbank_frames: int,
    patch_size: int,
    frequency_patches: int,
) -> Tensor:
    """Return q_j for the composed sample-to-token receptive fields.

    The denominator is the complete 16-frame receptive-field span. Frame slots
    appended to complete the final temporal patch contribute zero support, as do
    waveform samples beyond the last real filterbank frame.
    """
    if sample_masks.ndim != 2 or sample_masks.dtype != torch.bool:
        raise ValueError("sample_masks must be a boolean [batch, samples] tensor")
    if patch_size < 1 or fbank_frames < 1 or frequency_patches < 1:
        raise ValueError("BEATs frontend dimensions must be positive")
    frame_length = 400
    frame_shift = 160
    complete_receptive_field = frame_length + (patch_size - 1) * frame_shift
    time_patches = (fbank_frames + patch_size - 1) // patch_size
    support = sample_masks.to(dtype=torch.float64)
    cumulative = F.pad(torch.cumsum(support, dim=1), (1, 0), value=0.0)
    fractions: list[Tensor] = []
    for patch_index in range(time_patches):
        first_frame = patch_index * patch_size
        real_frames = min(patch_size, fbank_frames - first_frame)
        start = first_frame * frame_shift
        end = start + (real_frames - 1) * frame_shift + frame_length
        end = min(end, sample_masks.shape[1])
        supported = cumulative[:, end] - cumulative[:, start]
        fractions.append(supported / float(complete_receptive_field))
    patch_fraction = torch.stack(fractions, dim=1)
    token_fraction = (
        patch_fraction[:, :, None]
        .expand(-1, -1, frequency_patches)
        .reshape(patch_fraction.shape[0], -1)
    )
    if bool(((token_fraction < 0.0) | (token_fraction > 1.0)).any()):
        raise RuntimeError("BEATs token support fractions left [0, 1]")
    return token_fraction


def _beats_legacy_token_valid_mask(
    sample_masks: Tensor,
    *,
    fbank_frames: int,
    tokens: int,
) -> Tensor:
    """Reproduce the official BEATs right-padding downsampling semantics."""
    if sample_masks.ndim != 2 or sample_masks.dtype != torch.bool:
        raise ValueError("sample_masks must be a boolean [batch, samples] tensor")
    padding = ~sample_masks
    for output_length in (fbank_frames, tokens):
        extra = padding.shape[1] % output_length
        if extra:
            padding = padding[:, :-extra]
        padding = padding.reshape(padding.shape[0], output_length, -1).all(dim=-1)
    return ~padding


class PANNsCnn6Transfer(nn.Module):
    """Published CNN6 with a mask-correct two-class transfer head."""

    sample_rate = 32_000
    window_size = 1_024
    hop_size = 320
    mel_bins = 64
    embedding_dim = 512

    def __init__(self, classes_num: int = 527) -> None:
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=self.window_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=self.sample_rate,
            n_fft=self.window_size,
            n_mels=self.mel_bins,
            fmin=50,
            fmax=14_000,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )
        self.bn0 = nn.BatchNorm2d(self.mel_bins)
        self.conv_block1 = ConvBlock5x5(1, 64)
        self.conv_block2 = ConvBlock5x5(64, 128)
        self.conv_block3 = ConvBlock5x5(128, 256)
        self.conv_block4 = ConvBlock5x5(256, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc_audioset = nn.Linear(512, classes_num)
        _init_bn(self.bn0)
        _init_layer(self.fc1)
        _init_layer(self.fc_audioset)
        self.pretrained_load_audit: dict[str, Any] = {}

    @property
    def classifier(self) -> nn.Module:
        return self.fc_audioset

    def replace_classifier(self, num_classes: int = 2) -> None:
        self.fc_audioset = nn.Linear(self.embedding_dim, num_classes)
        _init_layer(self.fc_audioset)

    def forward(self, waveforms: Tensor, sample_masks: Tensor) -> Tensor:
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples]")
        if sample_masks.shape != waveforms.shape:
            raise ValueError("sample_masks must match waveforms")
        features = self.spectrogram_extractor(waveforms)
        features = self.logmel_extractor(features)
        frame_count = features.shape[2]
        centers = (
            torch.arange(frame_count, device=sample_masks.device) * self.hop_size
        ).clamp_max(sample_masks.shape[1] - 1)
        mask = sample_masks[:, centers]
        features = features.transpose(1, 3)
        features = self.bn0(features)
        features = features.transpose(1, 3)
        features = features.masked_fill(~mask[:, None, :, None], 0.0)
        if self.training:
            features = self.spec_augmenter(features)
        for block in (
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
        ):
            features = block(features)
            mask = _pool_mask(mask)
            features = features.masked_fill(~mask[:, None, :, None], 0.0)
            features = F.dropout(features, p=0.2, training=self.training)
        features = features.mean(dim=3)
        pooled = _masked_max_mean(features, mask)
        pooled = F.dropout(pooled, p=0.5, training=self.training)
        embeddings = F.relu_(self.fc1(pooled))
        return self.fc_audioset(embeddings)


def _checkpoint_state(path: Path) -> dict[str, Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Unsupported PANNs checkpoint payload")
    state = payload.get("model", payload.get("model_state_dict", payload))
    if not isinstance(state, dict):
        raise ValueError("PANNs checkpoint does not contain a state dictionary")
    return state


def load_panns_cnn6_transfer(path: Path, num_classes: int = 2) -> PANNsCnn6Transfer:
    model = PANNsCnn6Transfer(classes_num=527)
    state = _checkpoint_state(path)
    normalized_state = {
        key.removeprefix("model.").removeprefix("backbone."): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(normalized_state, strict=False)
    allowed_missing = {
        "spectrogram_extractor.stft.conv_real.weight",
        "spectrogram_extractor.stft.conv_imag.weight",
        "logmel_extractor.melW",
    }
    unexpected = list(incompatible.unexpected_keys)
    missing = [
        key for key in incompatible.missing_keys
        if key not in allowed_missing
    ]
    if missing or unexpected:
        raise ValueError(
            f"Incompatible CNN6 checkpoint; missing={missing}, unexpected={unexpected}"
        )
    loaded_backbone_keys = [
        key for key in normalized_state
        if not key.startswith("fc_audioset.")
    ]
    if not loaded_backbone_keys:
        raise ValueError("CNN6 checkpoint did not load backbone tensors")
    model.pretrained_load_audit = {
        "checkpoint": str(path),
        "source_tensor_count": len(state),
        "loaded_backbone_tensor_count": len(loaded_backbone_keys),
        "ignored_frontend_missing_keys": sorted(
            set(incompatible.missing_keys).intersection(allowed_missing)
        ),
        "unexpected_keys": unexpected,
    }
    model.replace_classifier(num_classes)
    return model


class BEATsTransfer(nn.Module):
    """Official BEATs encoder with CPU fbank preprocessing and a binary head."""

    def __init__(
        self,
        backbone: nn.Module,
        embedding_dim: int,
        load_audit: dict[str, Any],
        *,
        mask_mode: Literal["exact", "legacy"] = "exact",
    ) -> None:
        super().__init__()
        if mask_mode not in {"exact", "legacy"}:
            raise ValueError(f"Unsupported BEATs mask mode: {mask_mode}")
        self.backbone = backbone
        self.classifier = nn.Linear(embedding_dim, 2)
        _init_layer(self.classifier)
        self.pretrained_load_audit = load_audit
        self.mask_mode = mask_mode

    def forward(self, waveforms: Tensor, sample_masks: Tensor) -> Tensor:
        if waveforms.ndim != 2 or sample_masks.shape != waveforms.shape:
            raise ValueError("BEATs expects matching [batch, samples] waveform and mask")
        cpu_waveforms = waveforms.detach().to("cpu")
        cpu_valid_samples = sample_masks.detach().to("cpu")
        fbanks = self.backbone.preprocess(cpu_waveforms)
        patch_size = int(self.backbone.input_patch_size)
        if self.mask_mode == "legacy":
            fbank_frames = fbanks.shape[1]
            legacy_padding = self.backbone.forward_padding_mask(
                fbanks,
                ~cpu_valid_samples,
            )
            fbanks = fbanks.to(waveforms.device)
            patch_features = self.backbone.patch_embedding(fbanks.unsqueeze(1))
            features = patch_features.reshape(
                patch_features.shape[0],
                patch_features.shape[1],
                -1,
            ).transpose(1, 2)
            padding = self.backbone.forward_padding_mask(
                features,
                legacy_padding.to(features.device),
            )
            expected = _beats_legacy_token_valid_mask(
                cpu_valid_samples,
                fbank_frames=fbank_frames,
                tokens=features.shape[1],
            )
            if not torch.equal(~padding.detach().to("cpu"), expected):
                raise RuntimeError("Legacy BEATs padding-mask replay disagrees")
            features = self.backbone.layer_norm(features)
            if self.backbone.post_extract_proj is not None:
                features = self.backbone.post_extract_proj(features)
            features = self.backbone.dropout_input(features)
            encoded, _ = self.backbone.encoder(features, padding_mask=padding)
            weights = (~padding).to(dtype=encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return self.classifier(pooled)

        extra_frames = (-fbanks.shape[1]) % patch_size
        if extra_frames:
            fbanks = F.pad(fbanks, (0, 0, 0, extra_frames))
        fbanks = fbanks.to(waveforms.device)
        patch_features = self.backbone.patch_embedding(fbanks.unsqueeze(1))
        time_patches = patch_features.shape[2]
        frequency_patches = patch_features.shape[3]
        token_valid = _beats_token_valid_mask(
            cpu_valid_samples,
            fbank_frames=fbanks.shape[1] - extra_frames,
            patch_size=patch_size,
            frequency_patches=frequency_patches,
        )
        if token_valid.shape[1] != time_patches * frequency_patches:
            raise RuntimeError("Exact BEATs patch mask does not match patch embedding")
        padding = (~token_valid).to(waveforms.device)
        features = patch_features
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.backbone.layer_norm(features)
        if self.backbone.post_extract_proj is not None:
            features = self.backbone.post_extract_proj(features)
        features = self.backbone.dropout_input(features)
        encoded, _ = self.backbone.encoder(features, padding_mask=padding)
        weights = (~padding).to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def load_beats_transfer(
    checkpoint_path: Path,
    source_dir: Path,
    num_classes: int = 2,
    *,
    mask_mode: Literal["exact", "legacy"] = "exact",
) -> BEATsTransfer:
    if num_classes != 2:
        raise ValueError("Gate 9A BEATs adapter is frozen to two classes")
    source = str(source_dir.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    beats_module = importlib.import_module("BEATs")
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    config = beats_module.BEATsConfig(payload["cfg"])
    backbone = beats_module.BEATs(config)
    backbone.load_state_dict(payload["model"], strict=True)
    predictor_parameters = 0
    if backbone.predictor is not None:
        predictor_parameters = sum(
            parameter.numel() for parameter in backbone.predictor.parameters()
        )
        backbone.predictor = None
    return BEATsTransfer(
        backbone=backbone,
        embedding_dim=int(config.encoder_embed_dim),
        mask_mode=mask_mode,
        load_audit={
            "checkpoint": str(checkpoint_path),
            "source_tensor_count": len(payload["model"]),
            "discarded_audioset_predictor_parameters": predictor_parameters,
            "source_finetuned_model": bool(config.finetuned_model),
            "mask_mode": mask_mode,
        },
    )
