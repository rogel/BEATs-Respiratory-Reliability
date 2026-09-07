#!/usr/bin/env python3
"""Compute per-Mel-bin normalization statistics from training patients only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import FeatureConfig, ICBHICycleDataset
from respiratory_sound.data.icbhi import sha256_file


def feature_config_from_yaml(path: Path) -> FeatureConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    feature = payload["feature"]
    return FeatureConfig(
        sample_rate=int(payload["sample_rate"]),
        clip_seconds=float(payload["clip_seconds"]),
        duration_fit=str(payload.get("duration_fit", "repeat")),
        random_pad_position=bool(payload.get("random_pad_position", False)),
        n_fft=int(feature["n_fft"]),
        win_length=int(feature["win_length"]),
        hop_length=int(feature["hop_length"]),
        n_mels=int(feature["n_mels"]),
        f_min=float(feature["f_min"]),
        f_max=float(feature["f_max"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/icbhi2017_cycles.csv"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data/icbhi2017.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/icbhi2017_feature_stats.json"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-column", default="development_split")
    parser.add_argument("--split-value", default="train")
    parser.add_argument("--label-column", default="label_id")
    parser.add_argument("--filter-column")
    parser.add_argument("--filter-value")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    manifest_path = (project_root / args.manifest).resolve()
    config_path = (project_root / args.data_config).resolve()
    output_path = (project_root / args.output).resolve()
    feature_config = feature_config_from_yaml(config_path)
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        split_column=args.split_column,
        split_value=args.split_value,
        feature_config=feature_config,
        training=False,
        num_views=1,
        row_filters=(
            {args.filter_column: args.filter_value}
            if args.filter_column is not None
            else None
        ),
        label_column=args.label_column,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    feature_sum = torch.zeros(feature_config.n_mels, dtype=torch.float64)
    feature_square_sum = torch.zeros_like(feature_sum)
    count = 0
    for views, frame_masks, _, _ in loader:
        features = views[:, 0, 0].to(torch.float64)
        valid = frame_masks[:, 0].to(torch.float64)
        feature_sum += (features * valid.unsqueeze(1)).sum(dim=(0, 2))
        feature_square_sum += (features.square() * valid.unsqueeze(1)).sum(dim=(0, 2))
        count += int(valid.sum())

    mean = feature_sum / count
    variance = feature_square_sum / count - mean.square()
    std = variance.clamp_min(1e-10).sqrt()
    payload = {
        "split_column": args.split_column,
        "split_value": args.split_value,
        "label_column": args.label_column,
        "row_filter": (
            {args.filter_column: args.filter_value}
            if args.filter_column is not None
            else None
        ),
        "sample_count": len(dataset),
        "element_count_per_mel_bin": count,
        "manifest_sha256": sha256_file(manifest_path),
        "data_config_sha256": sha256_file(config_path),
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
