#!/usr/bin/env python3
"""Run the complete training pipeline on ephemeral synthetic audio."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import (
    FeatureConfig,
    FeatureNormalization,
    ICBHICycleDataset,
)
from respiratory_sound.models.network import TFDCRNet
from respiratory_sound.training import (
    class_weights,
    evaluate,
    seed_everything,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def build_synthetic_data(root: Path) -> Path:
    sample_rate = 8_000
    duration = 0.5
    time = np.arange(round(sample_rate * duration), dtype=np.float32) / sample_rate
    rows: list[dict[str, object]] = []
    for label in range(4):
        for sample_index in range(6):
            frequency = 180 + label * 140 + sample_index * 3
            waveform = 0.5 * np.sin(2 * np.pi * frequency * time)
            waveform += 0.05 * np.sin(2 * np.pi * (frequency * 2.1) * time)
            wav_path = root / f"class_{label}_{sample_index}.wav"
            sf.write(wav_path, waveform, sample_rate)
            rows.append(
                {
                    "sample_id": wav_path.stem,
                    "patient_id": f"{label}{sample_index:02d}",
                    "wav_path": wav_path.name,
                    "start_seconds": 0.0,
                    "end_seconds": duration,
                    "label_id": label,
                    "development_split": "train" if sample_index < 4 else "validation",
                }
            )
    manifest_path = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def main() -> None:
    args = parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")
    seed_everything(20_260_727)
    device = torch.device(args.device)
    with tempfile.TemporaryDirectory(prefix="respiratory_pipeline_") as temp_directory:
        root = Path(temp_directory)
        manifest_path = build_synthetic_data(root)
        config = FeatureConfig(
            sample_rate=16_000,
            clip_seconds=1.0,
            n_fft=256,
            win_length=200,
            hop_length=80,
            n_mels=16,
            f_max=4_000,
        )
        normalization = FeatureNormalization(
            mean=(0.0,) * config.n_mels,
            std=(1.0,) * config.n_mels,
        )
        train_dataset = ICBHICycleDataset(
            manifest_path=manifest_path,
            project_root=root,
            split_column="development_split",
            split_value="train",
            feature_config=config,
            normalization=normalization,
            training=True,
            num_views=2,
        )
        validation_dataset = ICBHICycleDataset(
            manifest_path=manifest_path,
            project_root=root,
            split_column="development_split",
            split_value="validation",
            feature_config=config,
            normalization=normalization,
            training=False,
            num_views=1,
        )
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)
        model = TFDCRNet(
            mode="tfdcr",
            channels=(8, 16),
            depths=(1, 1),
            temporal_dilations=((1,), (2,)),
            expansion_ratio=2,
        ).to(device)
        weights = class_weights(train_dataset.rows["label_id"]).to(device)
        loss_function = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = [
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                loss_function,
                consistency_strength=0.1,
            )
            for _ in range(2)
        ]
        validation = evaluate(model, validation_loader, device, loss_function)
        parameters_with_gradients = sum(
            parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
            for parameter in model.parameters()
        )
        result = {
            "device": str(device),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "losses": losses,
            "validation_metrics": validation.metrics,
            "parameters_with_finite_gradients": parameters_with_gradients,
            "gate_passed": all(np.isfinite(losses)) and parameters_with_gradients > 0,
        }
        print(json.dumps(result, indent=2))
        if not result["gate_passed"]:
            raise SystemExit("Synthetic end-to-end training gate failed")


if __name__ == "__main__":
    main()
