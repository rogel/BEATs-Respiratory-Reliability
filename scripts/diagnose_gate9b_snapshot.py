#!/usr/bin/env python3
"""Locate the first non-finite module in a captured Gate 9B batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor, nn
from train_gate9b_adaptation import _forward_batch

from respiratory_sound.models.beats_adaptation import (
    configure_beats_adaptation,
    projection_drift_regularization,
)
from respiratory_sound.models.pretrained_audio import load_beats_transfer
from respiratory_sound.training import jensen_shannon_consistency
from respiratory_sound.training_state import (
    load_trainable_parameter_state,
    restore_rng_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--backward", action="store_true")
    return parser.parse_args()


def _tensors(value: Any) -> list[Tensor]:
    if isinstance(value, Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [
            tensor
            for item in value
            for tensor in _tensors(item)
        ]
    if isinstance(value, dict):
        return [
            tensor
            for item in value.values()
            for tensor in _tensors(item)
        ]
    return []


def _audited_module(name: str) -> bool:
    return (
        name == "classifier"
        or name.startswith("backbone.patch_embedding")
        or name.startswith("backbone.layer_norm")
        or name.startswith("backbone.post_extract_proj")
        or name.startswith("backbone.dropout_input")
        or name.startswith("backbone.encoder.pos_conv")
        or name.startswith("backbone.encoder.layer_norm")
        or name.startswith("backbone.encoder.layers.")
    )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    snapshot = torch.load(
        (root / args.snapshot).resolve(),
        map_location="cpu",
        weights_only=False,
    )
    model_config = yaml.safe_load(
        (root / args.model_config).read_text(encoding="utf-8")
    )
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")

    model = load_beats_transfer(
        (root / str(model_config["checkpoint"])).resolve(),
        (root / str(model_config["source_dir"])).resolve(),
    )
    adaptation = model_config["adaptation"]
    configure_beats_adaptation(
        model,
        strategy=str(adaptation["strategy"]),
        lora_last_n_layers=int(adaptation.get("last_n_layers", 4)),
        lora_rank=int(adaptation.get("rank", 8)),
        lora_alpha=float(adaptation.get("alpha", 16.0)),
        lora_dropout=float(adaptation.get("dropout", 0.05)),
    )
    model = model.to(device)
    load_trainable_parameter_state(
        model,
        snapshot["trainable_state_dict"],
        list(snapshot["trainable_parameter_names"]),
    )
    model.train()
    restore_rng_state(snapshot["pre_forward_rng_state"], device)

    records: list[dict[str, Any]] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], outputs: Any) -> None:
            for output_index, tensor in enumerate(_tensors(outputs)):
                finite = torch.isfinite(tensor)
                if bool(finite.all().detach().cpu()):
                    continue
                record = {
                    "order": len(records),
                    "module": name,
                    "output_index": output_index,
                    "shape": list(tensor.shape),
                    "nonfinite_count": int((~finite).sum().detach().cpu()),
                }
                records.append(record)
                print(json.dumps(record), flush=True)

        return hook

    for name, module in model.named_modules():
        if name and _audited_module(name):
            handles.append(module.register_forward_hook(make_hook(name)))

    waveforms = snapshot["waveforms"]
    sample_masks = snapshot["sample_masks"]
    if args.backward:
        logits = _forward_batch(model, waveforms, sample_masks, device)
        targets = snapshot["targets"].to(device)
        repeated_targets = targets[:, None].expand(-1, logits.shape[1]).reshape(-1)
        classification_loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            repeated_targets,
        )
        consistency_loss = jensen_shannon_consistency(logits)
        projection_drift = projection_drift_regularization(model)
        loss = (
            classification_loss
            + float(snapshot["consistency_strength"]) * consistency_loss
            + float(snapshot["drift_anchor_strength"]) * projection_drift
        )
        loss.backward()
        if device.type == "mps":
            torch.mps.synchronize()
        gradients_finite = all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all().detach().cpu())
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        loss_value = float(loss.detach().cpu())
    else:
        with torch.no_grad():
            logits = _forward_batch(model, waveforms, sample_masks, device)
        gradients_finite = None
        loss_value = None
    for handle in handles:
        handle.remove()
    result = {
        "snapshot": str(args.snapshot),
        "batch_index": int(snapshot["batch_index"]),
        "sample_ids": list(snapshot["sample_ids"]),
        "waveforms_finite": bool(torch.isfinite(waveforms).all()),
        "first_nonfinite_module": records[0] if records else None,
        "nonfinite_module_records": records,
        "logits": logits.detach().cpu().tolist(),
        "backward_executed": bool(args.backward),
        "loss": loss_value,
        "all_trainable_gradients_finite": gradients_finite,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
