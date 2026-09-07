"""Online audio loading, augmentation, and Log-Mel feature extraction."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    clip_seconds: float = 8.0
    duration_fit: str = "repeat"
    random_pad_position: bool = False
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 64
    f_min: float = 50.0
    f_max: float = 2_000.0


@dataclass(frozen=True)
class AugmentationConfig:
    gain_min: float = 0.8
    gain_max: float = 1.2
    noise_probability: float = 0.5
    noise_snr_min_db: float = 12.0
    noise_snr_max_db: float = 30.0
    shift_probability: float = 0.5
    max_shift_fraction: float = 0.1
    frequency_mask_bins: int = 8
    time_mask_frames: int = 40
    frequency_response_probability: float = 0.0
    frequency_response_max_db: float = 0.0
    frequency_response_knots: int = 8


@dataclass(frozen=True)
class FeatureNormalization:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def from_json(cls, path: Path) -> FeatureNormalization:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            std=tuple(float(value) for value in payload["std"]),
        )


class LogMelFeature(nn.Module):
    """Convert a mono waveform to a numerically stored Log-Mel tensor."""

    def __init__(
        self,
        config: FeatureConfig,
        normalization: FeatureNormalization | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels,
            power=2.0,
        )
        if normalization is None:
            self.register_buffer("mean", None)
            self.register_buffer("std", None)
        else:
            if len(normalization.mean) != config.n_mels:
                raise ValueError("Normalization mean length does not match n_mels")
            if len(normalization.std) != config.n_mels:
                raise ValueError("Normalization std length does not match n_mels")
            self.register_buffer(
                "mean",
                torch.tensor(normalization.mean, dtype=torch.float32).reshape(1, -1, 1),
            )
            self.register_buffer(
                "std",
                torch.tensor(normalization.std, dtype=torch.float32).reshape(1, -1, 1),
            )

    def forward(self, waveform: Tensor) -> Tensor:
        power = self.mel(waveform)
        features = 10.0 * torch.log10(power.clamp_min(1e-10))
        if self.mean is not None and self.std is not None:
            features = (features - self.mean) / self.std.clamp_min(1e-5)
        return features


class ICBHICycleDataset(Dataset):
    """Load annotated respiratory cycles directly from WAV files."""

    def __init__(
        self,
        manifest_path: Path,
        project_root: Path,
        split_column: str,
        split_value: str,
        feature_config: FeatureConfig,
        normalization: FeatureNormalization | None = None,
        training: bool = False,
        num_views: int = 1,
        augmentation: AugmentationConfig | None = None,
        sample_limit: int | None = None,
        label_column: str = "label_id",
        return_waveform: bool = False,
        waveform_only: bool = False,
        row_filters: Mapping[str, object] | None = None,
    ) -> None:
        if num_views < 1:
            raise ValueError("num_views must be at least one")
        if waveform_only and not return_waveform:
            raise ValueError("waveform_only requires return_waveform=True")
        manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
        for column, value in (row_filters or {}).items():
            if column not in manifest.columns:
                raise ValueError(f"Missing row-filter column: {column}")
            manifest = manifest.loc[manifest[column] == value]
        if split_column not in manifest.columns:
            raise ValueError(f"Missing split column: {split_column}")
        self.rows = manifest.loc[manifest[split_column] == split_value].reset_index(drop=True)
        if label_column not in self.rows.columns:
            raise ValueError(f"Missing label column: {label_column}")
        self.label_column = label_column
        if sample_limit is not None:
            if sample_limit < self.rows[label_column].nunique():
                raise ValueError("sample_limit must allow at least one sample per class")
            labels = sorted(self.rows[label_column].unique())
            per_class, remainder = divmod(sample_limit, len(labels))
            selected: list[pd.DataFrame] = []
            for label_index, label in enumerate(labels):
                class_limit = per_class + int(label_index < remainder)
                selected.append(
                    self.rows.loc[self.rows[label_column] == label].iloc[:class_limit]
                )
            self.rows = (
                pd.concat(selected)
                .sort_values("sample_id")
                .iloc[:sample_limit]
                .reset_index(drop=True)
            )
        if self.rows.empty:
            raise ValueError(f"No samples for {split_column}={split_value}")
        self.project_root = project_root
        self.config = feature_config
        self.training = training
        self.num_views = num_views
        self.augmentation = augmentation or AugmentationConfig()
        if not 0.0 <= self.augmentation.frequency_response_probability <= 1.0:
            raise ValueError("frequency_response_probability must be in [0, 1]")
        if self.augmentation.frequency_response_max_db < 0.0:
            raise ValueError("frequency_response_max_db must be non-negative")
        if self.augmentation.frequency_response_knots < 2:
            raise ValueError("frequency_response_knots must be at least two")
        self.return_waveform = return_waveform
        self.waveform_only = waveform_only
        self.feature = (
            None
            if waveform_only
            else LogMelFeature(feature_config, normalization=normalization)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def _load_waveform(self, row: pd.Series) -> Tensor:
        wav_path = self.project_root / str(row["wav_path"])
        with sf.SoundFile(wav_path) as audio_file:
            source_rate = int(audio_file.samplerate)
            start_frame = max(0, math.floor(float(row["start_seconds"]) * source_rate))
            end_frame = min(
                len(audio_file),
                math.ceil(float(row["end_seconds"]) * source_rate),
            )
            audio_file.seek(start_frame)
            samples = audio_file.read(
                frames=end_frame - start_frame,
                dtype="float32",
                always_2d=True,
            )
        waveform = torch.from_numpy(samples).mean(dim=1)
        if not waveform.numel():
            raise ValueError(f"Empty annotated cycle in {wav_path}")
        if source_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=source_rate,
                new_freq=self.config.sample_rate,
            )
        waveform = waveform - waveform.mean()
        peak = waveform.abs().max().clamp_min(1e-6)
        return waveform / peak

    def _fit_duration(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        target = round(self.config.sample_rate * self.config.clip_seconds)
        if waveform.numel() >= target:
            if self.training:
                start = int(torch.randint(0, waveform.numel() - target + 1, ()).item())
            else:
                start = (waveform.numel() - target) // 2
            fitted = waveform[start : start + target]
            return fitted, torch.ones(target, dtype=torch.bool)
        if self.config.duration_fit == "zero_pad":
            maximum_start = target - waveform.numel()
            if self.training and self.config.random_pad_position and maximum_start > 0:
                start = int(torch.randint(0, maximum_start + 1, ()).item())
            else:
                start = maximum_start // 2
            fitted = torch.zeros(target, dtype=waveform.dtype)
            fitted[start : start + waveform.numel()] = waveform
            sample_mask = torch.zeros(target, dtype=torch.bool)
            sample_mask[start : start + waveform.numel()] = True
            return fitted, sample_mask
        if self.config.duration_fit != "repeat":
            raise ValueError(f"Unsupported duration_fit mode: {self.config.duration_fit}")
        repeats = math.ceil(target / waveform.numel())
        tiled = waveform.repeat(repeats)
        if self.training and tiled.numel() > target:
            start = int(torch.randint(0, tiled.numel() - target + 1, ()).item())
            return tiled[start : start + target], torch.ones(target, dtype=torch.bool)
        return tiled[:target], torch.ones(target, dtype=torch.bool)

    def _augment_waveform(self, waveform: Tensor) -> Tensor:
        config = self.augmentation
        gain = torch.empty(()).uniform_(config.gain_min, config.gain_max)
        waveform = waveform * gain
        if torch.rand(()) < config.shift_probability:
            max_shift = round(waveform.numel() * config.max_shift_fraction)
            shift = int(torch.randint(-max_shift, max_shift + 1, ()).item())
            waveform = torch.roll(waveform, shifts=shift)
        if torch.rand(()) < config.noise_probability:
            signal_rms = waveform.square().mean().sqrt().clamp_min(1e-6)
            snr_db = torch.empty(()).uniform_(
                config.noise_snr_min_db,
                config.noise_snr_max_db,
            )
            noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
            waveform = waveform + torch.randn_like(waveform) * noise_rms
        return waveform.clamp(-1.0, 1.0)

    def _mask_features(self, features: Tensor) -> Tensor:
        config = self.augmentation
        if config.frequency_mask_bins > 0:
            features = torchaudio.functional.mask_along_axis(
                features,
                mask_param=config.frequency_mask_bins,
                mask_value=0.0,
                axis=1,
            )
        if config.time_mask_frames > 0:
            features = torchaudio.functional.mask_along_axis(
                features,
                mask_param=config.time_mask_frames,
                mask_value=0.0,
                axis=2,
            )
        return features

    def _randomize_frequency_response(self, features: Tensor) -> Tensor:
        config = self.augmentation
        if (
            config.frequency_response_max_db == 0.0
            or torch.rand(()) >= config.frequency_response_probability
        ):
            return features
        control_points = torch.empty(
            1,
            1,
            config.frequency_response_knots,
            dtype=features.dtype,
            device=features.device,
        ).uniform_(
            -config.frequency_response_max_db,
            config.frequency_response_max_db,
        )
        response_db = F.interpolate(
            control_points,
            size=features.shape[1],
            mode="linear",
            align_corners=True,
        ).reshape(1, features.shape[1], 1)
        response_db = response_db - response_db.mean()
        if self.feature is None:
            raise RuntimeError("Feature randomization is unavailable in waveform-only mode")
        if self.feature.std is not None:
            response_db = response_db / self.feature.std.clamp_min(1.0e-5)
        return features + response_db

    def _frame_mask(self, sample_mask: Tensor, frame_count: int) -> Tensor:
        if bool(sample_mask.all()):
            return torch.ones(frame_count, dtype=torch.bool)
        valid_samples = sample_mask.nonzero(as_tuple=False).flatten()
        if not valid_samples.numel():
            raise ValueError("Duration fitting produced an empty valid region")
        first_sample = int(valid_samples[0])
        last_sample_exclusive = int(valid_samples[-1]) + 1
        frame_centers = torch.arange(frame_count) * self.config.hop_length
        return (frame_centers >= first_sample) & (frame_centers < last_sample_exclusive)

    def _make_view(self, waveform: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.feature is None:
            raise RuntimeError("Log-Mel view requested from a waveform-only dataset")
        if self.config.duration_fit == "zero_pad":
            source = self._augment_waveform(waveform) if self.training else waveform
            fitted, sample_mask = self._fit_duration(source)
        else:
            fitted, sample_mask = self._fit_duration(waveform)
            if self.training:
                fitted = self._augment_waveform(fitted)
        features = self.feature(fitted.unsqueeze(0))
        frame_mask = self._frame_mask(sample_mask, features.shape[-1])
        if self.training:
            features = self._randomize_frequency_response(features)
            features = self._mask_features(features)
        if self.config.duration_fit == "zero_pad":
            features = features.masked_fill(~frame_mask.reshape(1, 1, -1), 0.0)
        return features, frame_mask, fitted.unsqueeze(0), sample_mask

    def _make_waveform_view(self, waveform: Tensor) -> tuple[Tensor, Tensor]:
        """Create one fitted waveform view without computing unused Log-Mel features."""
        if self.config.duration_fit == "zero_pad":
            source = self._augment_waveform(waveform) if self.training else waveform
            fitted, sample_mask = self._fit_duration(source)
        else:
            fitted, sample_mask = self._fit_duration(waveform)
            if self.training:
                fitted = self._augment_waveform(fitted)
        return fitted.unsqueeze(0), sample_mask

    def __getitem__(self, index: int) -> tuple:
        row = self.rows.iloc[index]
        waveform = self._load_waveform(row)
        if self.waveform_only:
            waveform_pairs = [
                self._make_waveform_view(waveform) for _ in range(self.num_views)
            ]
            waveforms = torch.stack([pair[0] for pair in waveform_pairs])
            sample_masks = torch.stack([pair[1] for pair in waveform_pairs])
            return (
                waveforms,
                sample_masks,
                int(row[self.label_column]),
                str(row["sample_id"]),
            )
        view_pairs = [self._make_view(waveform) for _ in range(self.num_views)]
        views = torch.stack([pair[0] for pair in view_pairs])
        frame_masks = torch.stack([pair[1] for pair in view_pairs])
        if self.return_waveform:
            waveforms = torch.stack([pair[2] for pair in view_pairs])
            sample_masks = torch.stack([pair[3] for pair in view_pairs])
            return (
                views,
                frame_masks,
                waveforms,
                sample_masks,
                int(row[self.label_column]),
                str(row["sample_id"]),
            )
        return views, frame_masks, int(row[self.label_column]), str(row["sample_id"])
