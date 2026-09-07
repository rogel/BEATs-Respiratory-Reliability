#!/usr/bin/env python3
"""Replay seed-20260730 epoch-6 masks without altering the formal run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from torch.utils.data import DataLoader
from train_gate11a_exactmask_fullft import _augmentation

from respiratory_sound.data.audio import ICBHICycleDataset
from respiratory_sound.data.sampling import DomainClassEventBatchSampler
from respiratory_sound.gate11a import sha256_file
from respiratory_sound.models.pretrained_audio import (
    _beats_legacy_token_valid_mask,
    _beats_token_valid_mask,
)

SEED = 20_260_730
EPOCH_ZERO_BASED = 5
PLAN_SHA256 = "18b6e39bba441e5be1ec870591b9dbf2846b7e6ab6293be9d211aab12809be39"
MANIFEST_SHA256 = "2dd168129fc0fc159db201f2990c4d7074cb746fc9bc2c13bd5031e47dbd1419"


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    root = Path.cwd().resolve()
    run = root / "runs/revision_2026_09_03/revision_legacy_mask_seed20260730"
    configuration_path = run / "configuration.json"
    recovery_path = run / "resume_state.pt"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    recovery = torch.load(
        recovery_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if (
        int(configuration["seed"]) != SEED
        or configuration["family"] != "legacy-mask"
        or configuration["analysis_plan_sha256"] != PLAN_SHA256
        or recovery["analysis_plan_sha256"] != PLAN_SHA256
        or int(recovery["next_epoch"]) != EPOCH_ZERO_BASED
        or len(recovery["history"]) != EPOCH_ZERO_BASED
        or recovery["configuration_sha256"] != sha256_file(configuration_path)
    ):
        raise ValueError("Seed-20260730 recovery identity changed")
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Manifest hash changed")
    config_path = root / "configs/revision/beats_ablation.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    feature_config = feature_config_from_yaml(root / "configs/data/gate9a_beats.yaml")
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column="protocol_role",
        split_value="train_fit",
        feature_config=feature_config,
        training=True,
        num_views=2,
        augmentation=_augmentation(config["common"]),
        label_column="binary_label_id",
        return_waveform=True,
        waveform_only=True,
    )
    sampler = DomainClassEventBatchSampler(
        dataset.rows,
        batch_size=8,
        samples_per_epoch=7_515,
        seed=SEED,
        class_column="binary_label_id",
    )
    sampler.set_epoch(EPOCH_ZERO_BASED)
    generator = torch.Generator()
    generator.manual_seed(SEED + 11_000_000)
    generator.set_state(recovery["train_loader_generator_state"])
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        pin_memory=False,
        generator=generator,
    )
    zero_views: list[dict[str, Any]] = []
    batches_scanned = 0
    views_scanned = 0
    for batch_index, (_, masks, targets, sample_ids) in enumerate(loader):
        flat = masks.reshape(-1, masks.shape[-1])
        legacy = _beats_legacy_token_valid_mask(
            flat,
            fbank_frames=798,
            tokens=392,
        )
        exact = _beats_token_valid_mask(
            flat,
            fbank_frames=798,
            patch_size=16,
            frequency_patches=8,
        )
        batches_scanned += 1
        views_scanned += len(flat)
        for flat_index in torch.nonzero(~legacy.any(dim=1), as_tuple=False).flatten():
            batch_row = int(flat_index) // masks.shape[1]
            view = int(flat_index) % masks.shape[1]
            mask = flat[int(flat_index)]
            positions = torch.nonzero(mask, as_tuple=False).flatten()
            zero_views.append(
                {
                    "batch_index_zero_based": batch_index,
                    "sample_id": str(sample_ids[batch_row]),
                    "target": int(targets[batch_row]),
                    "view_index_zero_based": view,
                    "valid_waveform_samples": int(mask.sum()),
                    "first_valid_sample": int(positions[0]),
                    "last_valid_sample_inclusive": int(positions[-1]),
                    "legacy_valid_tokens": int(legacy[int(flat_index)].sum()),
                    "exact_valid_tokens": int(exact[int(flat_index)].sum()),
                    "other_view_legacy_valid_tokens": int(
                        legacy[batch_row * masks.shape[1] + (1 - view)].sum()
                    ),
                }
            )
        if zero_views:
            break
    payload = {
        "schema_version": 1,
        "stage": "BEATS_seed20260730_epoch6_mask_replay",
        "status": "zero_legacy_token_view_found" if zero_views else "no_zero_view_found",
        "formal_run_modified": False,
        "seed": SEED,
        "epoch_one_based": EPOCH_ZERO_BASED + 1,
        "batches_scanned": batches_scanned,
        "views_scanned": views_scanned,
        "first_batch_with_zero_legacy_view": zero_views,
        "configuration_sha256": sha256_file(configuration_path),
        "recovery_state_sha256": sha256_file(recovery_path),
        "manifest_sha256": sha256_file(manifest_path),
        "revision_config_sha256": sha256_file(config_path),
        "analysis_plan_sha256": PLAN_SHA256,
        "calibration_accessed": False,
        "locked_tests_accessed": False,
    }
    output = (
        root
        / "artifacts/revision_2026_09_03/failures/"
        "legacy_mask_seed20260730_epoch6_mask_replay.json"
    )
    _atomic_json(payload, output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
