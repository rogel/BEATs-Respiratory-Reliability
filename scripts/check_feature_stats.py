#!/usr/bin/env python3
"""Verify saved training-only normalization statistics and normalized features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from compute_feature_stats import feature_config_from_yaml
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import FeatureNormalization, ICBHICycleDataset
from respiratory_sound.data.icbhi import sha256_file


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
        "--feature-stats",
        type=Path,
        default=Path("data/manifests/icbhi2017_feature_stats.json"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    data_config_path = (root / args.data_config).resolve()
    stats_path = (root / args.feature_stats).resolve()
    stats_payload = json.loads(stats_path.read_text(encoding="utf-8"))
    hashes_match = stats_payload["manifest_sha256"] == sha256_file(manifest_path) and stats_payload[
        "data_config_sha256"
    ] == sha256_file(data_config_path)
    feature_config = feature_config_from_yaml(data_config_path)
    normalization = FeatureNormalization.from_json(stats_path)
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column="development_split",
        split_value="train",
        feature_config=feature_config,
        normalization=normalization,
        training=False,
        num_views=1,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    feature_sum = torch.zeros(feature_config.n_mels, dtype=torch.float64)
    feature_square_sum = torch.zeros_like(feature_sum)
    count = 0
    all_finite = True
    for views, frame_masks, _, _ in loader:
        features = views[:, 0, 0].to(torch.float64)
        all_finite = all_finite and bool(torch.isfinite(features).all())
        valid = frame_masks[:, 0].to(torch.float64)
        feature_sum += (features * valid.unsqueeze(1)).sum(dim=(0, 2))
        feature_square_sum += (features.square() * valid.unsqueeze(1)).sum(dim=(0, 2))
        count += int(valid.sum())
    mean = feature_sum / count
    std = (feature_square_sum / count - mean.square()).clamp_min(0).sqrt()
    maximum_mean_error = float(mean.abs().max())
    maximum_std_error = float((std - 1.0).abs().max())
    payload = {
        "sample_count": len(dataset),
        "all_features_finite": all_finite,
        "source_hashes_match": hashes_match,
        "maximum_absolute_normalized_mean": maximum_mean_error,
        "maximum_absolute_normalized_std_error": maximum_std_error,
        "gate_passed": (
            all_finite and hashes_match and maximum_mean_error < 1e-5 and maximum_std_error < 1e-5
        ),
    }
    print(json.dumps(payload, indent=2))
    if not payload["gate_passed"]:
        raise SystemExit("Feature normalization gate failed")


if __name__ == "__main__":
    main()
