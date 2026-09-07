"""Inference helpers for the frozen post-Gate-11A evidence chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from respiratory_sound.data.audio import ICBHICycleDataset
from respiratory_sound.models.beats_adaptation import configure_beats_adaptation
from respiratory_sound.models.pretrained_audio import load_beats_transfer
from respiratory_sound.training_state import load_trainable_parameter_state

DOMAINS = ("icbhi2017", "sprsound2022")
SEEDS = (20_260_729, 20_260_730, 20_260_731)
FULLFT_RUNS = {
    20_260_729: "runs/gate11a_exactmask_fullft_seed20260729",
    20_260_730: "runs/gate11a_exactmask_fullft_seed20260730",
    20_260_731: "runs/gate11a_exactmask_fullft_seed20260731",
}
LORA_RUNS = {
    20_260_729: "runs/gate9b_exactmask_lora_qv_r8_seed20260729",
    20_260_730: "runs/gate9d_domain_class_event_lora_qv_r8_seed20260730",
    20_260_731: "runs/gate9d_domain_class_event_lora_qv_r8_seed20260731",
}


def run_map(family: str) -> dict[int, str]:
    if family == "fullft":
        return FULLFT_RUNS
    if family == "lora":
        return LORA_RUNS
    raise ValueError(f"Unsupported post-Gate-11A family: {family}")


def load_frozen_model(
    root: Path,
    *,
    family: str,
    seed: int,
    device: torch.device,
) -> nn.Module:
    if seed not in SEEDS:
        raise ValueError(f"Unexpected seed: {seed}")
    checkpoint_path = root / run_map(family)[seed] / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    upstream = (root / str(model_config["checkpoint"])).resolve()
    source_dir = (root / str(model_config["source_dir"])).resolve()
    model = load_beats_transfer(upstream, source_dir)
    if family == "fullft":
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        adaptation = model_config["adaptation"]
        configure_beats_adaptation(
            model,
            strategy=str(adaptation["strategy"]),
            lora_last_n_layers=int(adaptation.get("last_n_layers", 4)),
            lora_rank=int(adaptation.get("rank", 8)),
            lora_alpha=float(adaptation.get("alpha", 16.0)),
            lora_dropout=float(adaptation.get("dropout", 0.05)),
        )
        load_trainable_parameter_state(
            model,
            checkpoint["trainable_state_dict"],
            list(checkpoint["trainable_parameter_names"]),
        )
    model.eval()
    return model.to(device)


def build_waveform_dataset(
    *,
    manifest_path: Path,
    project_root: Path,
    feature_config: Any,
    role: str,
    domain: str,
) -> ICBHICycleDataset:
    if domain not in DOMAINS:
        raise ValueError(f"Unexpected domain: {domain}")
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        split_column="protocol_role",
        split_value=role,
        feature_config=feature_config,
        training=False,
        num_views=1,
        label_column="binary_label_id",
        return_waveform=True,
        waveform_only=True,
        row_filters={"dataset": domain},
    )
    assert_role_rows(
        dataset.rows,
        expected_role=role,
        expected_domain=domain,
        expect_locked=role.startswith("locked"),
    )
    return dataset


def assert_role_rows(
    rows: pd.DataFrame,
    *,
    expected_role: str,
    expected_domain: str,
    expect_locked: bool,
) -> None:
    required = {
        "sample_id",
        "dataset",
        "patient_id",
        "protocol_role",
        "locked",
        "binary_label_id",
    }
    missing = required.difference(rows.columns)
    if missing or rows.empty:
        raise ValueError(f"Invalid post-Gate-11A row set; missing={sorted(missing)}")
    if rows["sample_id"].duplicated().any():
        raise ValueError("Post-Gate-11A row set contains duplicate samples")
    if set(rows["protocol_role"].astype(str)) != {expected_role}:
        raise ValueError("Post-Gate-11A role boundary mismatch")
    if set(rows["dataset"].astype(str)) != {expected_domain}:
        raise ValueError("Post-Gate-11A domain boundary mismatch")
    locked = rows["locked"]
    locked_bool = (
        locked
        if locked.dtype == bool
        else locked.astype(str).str.lower().isin({"true", "1"})
    )
    if bool(locked_bool.all()) != expect_locked or bool(locked_bool.any()) != expect_locked:
        raise ValueError("Post-Gate-11A locked flag mismatch")
    if not set(rows["binary_label_id"].astype(int)).issubset({0, 1}):
        raise ValueError("Post-Gate-11A row set contains invalid labels")


@torch.inference_mode()
def infer_probabilities(
    model: nn.Module,
    dataset: ICBHICycleDataset,
    *,
    device: torch.device,
    batch_size: int = 8,
) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sample_ids: list[str] = []
    targets: list[torch.Tensor] = []
    probabilities: list[torch.Tensor] = []
    for waveforms, masks, batch_targets, batch_ids in loader:
        if waveforms.ndim != 4 or waveforms.shape[1:3] != (1, 1):
            raise ValueError("Post-Gate-11A inference expects one mono view")
        flat = waveforms[:, 0, 0].to(device)
        flat_masks = masks[:, 0].to(device)
        logits = model(flat, flat_masks)
        batch_probabilities = torch.softmax(logits, dim=-1)
        if not bool(torch.isfinite(batch_probabilities).all().detach().cpu()):
            raise FloatingPointError("Post-Gate-11A inference became non-finite")
        sample_ids.extend(str(value) for value in batch_ids)
        targets.append(batch_targets.cpu())
        probabilities.append(batch_probabilities.cpu())
    probability_array = torch.cat(probabilities).numpy()
    target_array = torch.cat(targets).numpy()
    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "target": target_array,
            "probability_0": probability_array[:, 0],
            "probability_1": probability_array[:, 1],
        }
    )
    metadata = dataset.rows[
        [
            "sample_id",
            "dataset",
            "patient_id",
            "protocol_role",
            "locked",
            "binary_label_id",
            "fine_label_name",
            "event_duration_seconds",
        ]
    ].copy()
    merged = metadata.merge(predictions, on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(dataset) or not np.array_equal(
        merged["binary_label_id"].to_numpy(dtype=int),
        merged["target"].to_numpy(dtype=int),
    ):
        raise ValueError("Post-Gate-11A predictions do not match the selected samples")
    merged["prediction"] = (merged["probability_1"] >= 0.5).astype(int)
    return merged


def probability_ensemble(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) != len(SEEDS):
        raise ValueError("Post-Gate-11A ensemble requires all three seeds")
    keys = [
        "sample_id",
        "dataset",
        "patient_id",
        "protocol_role",
        "locked",
        "binary_label_id",
        "fine_label_name",
        "event_duration_seconds",
        "target",
    ]
    merged = frames[0][keys + ["probability_1"]].rename(
        columns={"probability_1": f"probability_1_seed{SEEDS[0]}"}
    )
    for seed, frame in zip(SEEDS[1:], frames[1:], strict=True):
        merged = merged.merge(
            frame[keys + ["probability_1"]].rename(
                columns={"probability_1": f"probability_1_seed{seed}"}
            ),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    probability_columns = [f"probability_1_seed{seed}" for seed in SEEDS]
    merged["probability_1"] = merged[probability_columns].mean(axis=1)
    merged["probability_0"] = 1.0 - merged["probability_1"]
    merged["prediction"] = (merged["probability_1"] >= 0.5).astype(int)
    if len(merged) != len(frames[0]):
        raise ValueError("Post-Gate-11A ensemble lost samples")
    return merged
