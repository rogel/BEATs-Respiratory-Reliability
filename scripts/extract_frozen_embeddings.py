#!/usr/bin/env python3
"""Extract frozen pooled embeddings without touching locked test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import (
    FeatureConfig,
    FeatureNormalization,
    ICBHICycleDataset,
)
from respiratory_sound.models.factory import tfdcr_from_config
from respiratory_sound.runtime import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-stats", type=Path, required=True)
    parser.add_argument("--split-column", required=True)
    parser.add_argument("--split-value", required=True)
    parser.add_argument("--filter-column", required=True)
    parser.add_argument("--filter-value", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.split_value.lower().startswith("locked_") or args.split_value.lower() == "test":
        raise SystemExit("Locked splits cannot be used for adaptation embeddings")
    root = args.project_root.resolve()
    checkpoint_path = (root / args.checkpoint).resolve()
    manifest_path = (root / args.manifest).resolve()
    stats_path = (root / args.feature_stats).resolve()
    output_path = (root / args.output).resolve()
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = tfdcr_from_config(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    feature_config = FeatureConfig(**checkpoint["feature_config"])
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column=args.split_column,
        split_value=args.split_value,
        feature_config=feature_config,
        normalization=FeatureNormalization.from_json(stats_path),
        training=False,
        num_views=1,
        label_column=str(checkpoint.get("label_column", "label_id")),
        row_filters={args.filter_column: args.filter_value},
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    all_embeddings: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_sample_ids: list[str] = []
    for views, frame_masks, targets, sample_ids in loader:
        inputs = views[:, 0].to(device)
        masks = frame_masks[:, 0].to(device)
        feature_maps = model.forward_features(inputs, frame_masks=masks)
        embeddings = model.pool(feature_maps, frame_masks=masks)
        all_embeddings.append(embeddings.cpu())
        all_targets.append(targets.cpu())
        all_sample_ids.extend(sample_ids)

    embeddings_array = torch.cat(all_embeddings).numpy().astype(np.float32)
    targets_array = torch.cat(all_targets).numpy().astype(np.int64)
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    patient_lookup = manifest.set_index("sample_id")["patient_id"].astype(str)
    patient_ids = np.asarray([patient_lookup[sample_id] for sample_id in all_sample_ids])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings_array,
        targets=targets_array,
        sample_ids=np.asarray(all_sample_ids),
        patient_ids=patient_ids,
        classifier_weight=model.classifier.weight.detach().cpu().numpy(),
        classifier_bias=model.classifier.bias.detach().cpu().numpy(),
    )
    metadata = {
        "checkpoint": str(checkpoint_path.relative_to(root)),
        "manifest": str(manifest_path.relative_to(root)),
        "feature_stats": str(stats_path.relative_to(root)),
        "split_column": args.split_column,
        "split_value": args.split_value,
        "row_filter": {args.filter_column: args.filter_value},
        "samples": len(dataset),
        "patients": int(len(np.unique(patient_ids))),
        "embedding_dim": int(embeddings_array.shape[1]),
        "class_counts": {
            str(class_index): int((targets_array == class_index).sum())
            for class_index in (0, 1)
        },
        "locked_test_accessed": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
