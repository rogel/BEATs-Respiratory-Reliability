#!/usr/bin/env python3
"""Instrument the documented exact/legacy zero-token stability replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from respiratory_sound.gate11a import sha256_file
from respiratory_sound.models.beats_adaptation import configure_beats_adaptation
from respiratory_sound.models.pretrained_audio import load_beats_transfer
from respiratory_sound.runtime import select_device
from respiratory_sound.training import jensen_shannon_consistency, seed_everything

PLAN_SHA256 = "18b6e39bba441e5be1ec870591b9dbf2846b7e6ab6293be9d211aab12809be39"
UPSTREAM_SHA256 = "e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    return parser.parse_args()


def _all_finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all().detach().cpu())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value if item is not None)
    return True


def _run_mode(
    mode: str,
    *,
    checkpoint: Path,
    source: Path,
    waveforms: Tensor,
    masks: Tensor,
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(20_260_729)
    model = load_beats_transfer(checkpoint, source, mask_mode=mode)
    adaptation = configure_beats_adaptation(
        model,
        strategy="lora_qv",
        lora_last_n_layers=4,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
    )
    model = model.to(device).train()
    layer_finite: dict[str, bool] = {}
    attention_finite: dict[str, bool] = {}
    hooks: list[Any] = []
    for index, layer in enumerate(model.backbone.encoder.layers):
        hooks.append(
            layer.register_forward_hook(
                lambda _module, _inputs, output, index=index: layer_finite.__setitem__(
                    str(index), _all_finite(output)
                )
            )
        )
        hooks.append(
            layer.self_attn.register_forward_hook(
                lambda _module, _inputs, output, index=index: attention_finite.__setitem__(
                    str(index), _all_finite(output)
                )
            )
        )
    result: dict[str, Any] = {
        "mask_mode": mode,
        "adaptation": adaptation.to_dict(),
        "layer_output_finite": layer_finite,
        "attention_output_finite": attention_finite,
        "exception": None,
    }
    try:
        model.zero_grad(set_to_none=True)
        first = model(waveforms.to(device), masks.to(device))
        second = model(waveforms.to(device), masks.to(device))
        logits = torch.stack((first, second), dim=1)
        targets = torch.tensor([0, 1], device=device)
        classification = nn.functional.cross_entropy(
            logits.reshape(-1, 2), targets[:, None].expand(-1, 2).reshape(-1)
        )
        consistency = jensen_shannon_consistency(logits)
        loss = classification + 0.02 * consistency
        result.update(
            {
                "logits_finite": _all_finite(logits),
                "classification_loss_finite": _all_finite(classification),
                "consistency_loss_finite": _all_finite(consistency),
                "total_loss_finite": _all_finite(loss),
            }
        )
        loss.backward()
        trainable_gradients = {
            name: _all_finite(parameter.grad)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        }
        result["trainable_gradients_present"] = len(trainable_gradients)
        result["trainable_gradients_all_finite"] = bool(
            trainable_gradients and all(trainable_gradients.values())
        )
        result["nonfinite_gradient_names"] = [
            name for name, finite in trainable_gradients.items() if not finite
        ]
    except Exception as error:  # the legacy failure is an audited outcome
        result["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        result.setdefault("logits_finite", False)
        result.setdefault("classification_loss_finite", False)
        result.setdefault("consistency_loss_finite", False)
        result.setdefault("total_loss_finite", False)
        result.setdefault("trainable_gradients_present", 0)
        result.setdefault("trainable_gradients_all_finite", False)
        result.setdefault("nonfinite_gradient_names", [])
    finally:
        for hook in hooks:
            hook.remove()
    result["encoder_layers_observed"] = len(layer_finite)
    result["encoder_layers_all_finite"] = bool(
        len(layer_finite) == 12 and all(layer_finite.values())
    )
    result["attention_modules_observed"] = len(attention_finite)
    result["attention_outputs_all_finite"] = bool(
        len(attention_finite) == 12 and all(attention_finite.values())
    )
    return result


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    plan = (
        root
        / "../experiment_plans/2026-09-03/"
        "01_REVISION_ANALYSIS_PLAN_FROZEN_2026-09-03.md"
    ).resolve()
    checkpoint = (
        root
        / "checkpoints/pretrained/beats_as2m_cpt2/"
        "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    )
    source = root / "third_party/beats"
    if sha256_file(plan) != PLAN_SHA256:
        raise ValueError("Frozen analysis plan changed")
    if sha256_file(checkpoint) != UPSTREAM_SHA256:
        raise ValueError("Frozen BEATs checkpoint changed")
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")
    masks = torch.zeros(2, 128_000, dtype=torch.bool)
    masks[0, 114_458:116_474] = True
    masks[1, 125_554:127_570] = True
    generator = torch.Generator().manual_seed(20_260_729)
    waveforms = torch.randn((2, 128_000), generator=generator) * 0.01
    waveforms = waveforms.masked_fill(~masks, 0.0)
    result = {
        "schema_version": 1,
        "stage": "BEATS_revision",
        "analysis": "documented_failure_numerical_stability_replay",
        "analysis_plan_sha256": PLAN_SHA256,
        "device": str(device),
        "valid_samples_per_view": masks.sum(dim=1).tolist(),
        "exact": _run_mode(
            "exact",
            checkpoint=checkpoint,
            source=source,
            waveforms=waveforms,
            masks=masks,
            device=device,
        ),
        "legacy": _run_mode(
            "legacy",
            checkpoint=checkpoint,
            source=source,
            waveforms=waveforms,
            masks=masks,
            device=device,
        ),
    }
    output = root / "artifacts/revision_2026_09_03/mask_mapping/numerical_stability_replay.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
