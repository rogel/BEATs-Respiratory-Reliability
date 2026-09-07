from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from respiratory_sound.data.audio import (
    AugmentationConfig,
    FeatureConfig,
    FeatureNormalization,
    ICBHICycleDataset,
)


def _write_synthetic_manifest(tmp_path: Path) -> Path:
    sample_rate = 8_000
    time = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate
    waveform = 0.5 * np.sin(2 * np.pi * 300 * time)
    wav_path = tmp_path / "synthetic.wav"
    sf.write(wav_path, waveform, sample_rate)
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "synthetic__cycle_000",
                "patient_id": "001",
                "wav_path": wav_path.relative_to(tmp_path).as_posix(),
                "start_seconds": 0.0,
                "end_seconds": 0.25,
                "label_id": 2,
                "development_split": "train",
                "dataset": "source_a",
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path


def test_cycle_dataset_applies_row_filters(tmp_path: Path) -> None:
    manifest_path = _write_synthetic_manifest(tmp_path)
    manifest = pd.read_csv(manifest_path)
    second = manifest.iloc[0].copy()
    second["sample_id"] = "other__cycle_000"
    second["dataset"] = "source_b"
    pd.concat([manifest, second.to_frame().T], ignore_index=True).to_csv(
        manifest_path,
        index=False,
    )
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
        split_column="development_split",
        split_value="train",
        feature_config=FeatureConfig(
            sample_rate=16_000,
            clip_seconds=1.0,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=16,
            f_max=4_000,
        ),
        row_filters={"dataset": "source_b"},
    )

    assert len(dataset) == 1
    assert dataset.rows.iloc[0]["sample_id"] == "other__cycle_000"


def test_cycle_dataset_loads_resamples_and_repeats(tmp_path: Path) -> None:
    manifest_path = _write_synthetic_manifest(tmp_path)
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
        split_column="development_split",
        split_value="train",
        feature_config=FeatureConfig(
            sample_rate=16_000,
            clip_seconds=1.0,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=16,
            f_max=4_000,
        ),
        training=False,
        num_views=2,
    )
    views, frame_masks, label, sample_id = dataset[0]

    assert views.shape == (2, 1, 16, 201)
    assert frame_masks.shape == (2, 201)
    assert frame_masks.all()
    assert torch.isfinite(views).all()
    assert torch.equal(views[0], views[1])
    assert label == 2
    assert sample_id == "synthetic__cycle_000"


def test_cycle_dataset_training_views_are_finite(tmp_path: Path) -> None:
    manifest_path = _write_synthetic_manifest(tmp_path)
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
        split_column="development_split",
        split_value="train",
        feature_config=FeatureConfig(
            sample_rate=16_000,
            clip_seconds=1.0,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=16,
            f_max=4_000,
        ),
        training=True,
        num_views=2,
    )
    views, frame_masks, _, _ = dataset[0]

    assert views.shape == (2, 1, 16, 201)
    assert frame_masks.shape == (2, 201)
    assert frame_masks.all()
    assert torch.isfinite(views).all()


def test_frequency_response_randomization_is_smooth_and_view_specific(
    tmp_path: Path,
) -> None:
    manifest_path = _write_synthetic_manifest(tmp_path)
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
        split_column="development_split",
        split_value="train",
        feature_config=FeatureConfig(
            sample_rate=16_000,
            clip_seconds=1.0,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=16,
            f_max=4_000,
        ),
        normalization=FeatureNormalization(
            mean=tuple(0.0 for _ in range(16)),
            std=tuple(2.0 for _ in range(16)),
        ),
        training=True,
        num_views=2,
        augmentation=AugmentationConfig(
            gain_min=1.0,
            gain_max=1.0,
            noise_probability=0.0,
            shift_probability=0.0,
            frequency_mask_bins=0,
            time_mask_frames=0,
            frequency_response_probability=1.0,
            frequency_response_max_db=6.0,
            frequency_response_knots=4,
        ),
    )
    torch.manual_seed(21)
    views, frame_masks, _, _ = dataset[0]
    view_difference = views[0] - views[1]

    assert frame_masks.all()
    assert not torch.allclose(views[0], views[1])
    assert torch.allclose(
        view_difference.std(dim=-1),
        torch.zeros_like(view_difference.std(dim=-1)),
        atol=1.0e-5,
    )
    frequency_curve = view_difference.mean(dim=-1).flatten()
    assert torch.max(torch.abs(torch.diff(frequency_curve))) < 3.0


def test_cycle_dataset_zero_padding_marks_only_real_audio(tmp_path: Path) -> None:
    manifest_path = _write_synthetic_manifest(tmp_path)
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
        split_column="development_split",
        split_value="train",
        feature_config=FeatureConfig(
            sample_rate=16_000,
            clip_seconds=1.0,
            duration_fit="zero_pad",
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=16,
            f_max=4_000,
        ),
        training=False,
        num_views=1,
    )
    views, frame_masks, _, _ = dataset[0]
    valid = frame_masks[0]
    valid_indices = valid.nonzero(as_tuple=False).flatten()

    assert 0 < int(valid.sum()) < valid.numel()
    assert torch.equal(
        valid_indices,
        torch.arange(valid_indices[0], valid_indices[-1] + 1),
    )
    assert torch.count_nonzero(views[0, :, :, ~valid]) == 0


def test_cycle_dataset_can_return_aligned_waveform_and_masks(tmp_path: Path) -> None:
    manifest_path = _write_synthetic_manifest(tmp_path)
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
        split_column="development_split",
        split_value="train",
        feature_config=FeatureConfig(
            sample_rate=16_000,
            clip_seconds=1.0,
            duration_fit="zero_pad",
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=16,
            f_max=4_000,
        ),
        training=False,
        num_views=2,
        return_waveform=True,
    )
    views, frame_masks, waveforms, sample_masks, label, sample_id = dataset[0]

    assert views.shape == (2, 1, 16, 201)
    assert frame_masks.shape == (2, 201)
    assert waveforms.shape == (2, 1, 16_000)
    assert sample_masks.shape == (2, 16_000)
    assert torch.equal(waveforms[0], waveforms[1])
    assert torch.equal(sample_masks[0], sample_masks[1])
    assert torch.count_nonzero(waveforms[0, 0, ~sample_masks[0]]) == 0
    assert label == 2
    assert sample_id == "synthetic__cycle_000"


def test_cycle_dataset_waveform_only_skips_spectrogram_output(tmp_path: Path) -> None:
    manifest_path = _write_synthetic_manifest(tmp_path)
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=tmp_path,
        split_column="development_split",
        split_value="train",
        feature_config=FeatureConfig(
            sample_rate=16_000,
            clip_seconds=1.0,
            duration_fit="zero_pad",
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=16,
            f_max=4_000,
        ),
        training=False,
        num_views=2,
        return_waveform=True,
        waveform_only=True,
    )
    waveforms, sample_masks, label, sample_id = dataset[0]

    assert waveforms.shape == (2, 1, 16_000)
    assert sample_masks.shape == (2, 16_000)
    assert torch.equal(waveforms[0], waveforms[1])
    assert torch.equal(sample_masks[0], sample_masks[1])
    assert torch.count_nonzero(waveforms[0, 0, ~sample_masks[0]]) == 0
    assert label == 2
    assert sample_id == "synthetic__cycle_000"
