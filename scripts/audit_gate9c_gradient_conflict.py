#!/usr/bin/env python3
"""Run the frozen Gate 9C-A training-only gradient-conflict audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from compute_feature_stats import feature_config_from_yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader
from train_gate9a_pretrained import _assert_development_protocol, _augmentation
from train_gate9b_adaptation import (
    _forward_batch,
    _sha256,
    _trainable_parameter_groups,
)

from respiratory_sound.data.audio import ICBHICycleDataset
from respiratory_sound.data.sampling import DomainClassEventBatchSampler
from respiratory_sound.gradient_conflict import (
    domain_class_masks,
    gradient_cosine,
    gradient_dot,
    gradient_norm,
    mean_gradients,
    named_lora_parameters,
    summarize_conflict_rows,
    summarize_norm_balance_rows,
)
from respiratory_sound.models.beats_adaptation import (
    configure_beats_adaptation,
    projection_drift_regularization,
)
from respiratory_sound.models.pretrained_audio import load_beats_transfer
from respiratory_sound.runtime import select_device
from respiratory_sound.training import (
    jensen_shannon_consistency,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-checkpoint", type=Path)
    return parser.parse_args()


def _loader(
    dataset: ICBHICycleDataset,
    *,
    batch_size: int,
    batches: int,
    seed: int,
    label_column: str,
) -> DataLoader:
    sampler = DomainClassEventBatchSampler(
        dataset.rows,
        batch_size=batch_size,
        samples_per_epoch=batch_size * batches,
        seed=seed,
        class_column=label_column,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=False,
    )


def _per_event_classification_loss(logits: Tensor, targets: Tensor) -> Tensor:
    batch_size, views, classes = logits.shape
    repeated = targets[:, None].expand(-1, views).reshape(-1)
    return nn.functional.cross_entropy(
        logits.reshape(-1, classes),
        repeated,
        reduction="none",
    ).reshape(batch_size, views).mean(dim=1)


def _audit_batches(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    domains: tuple[str, str],
    samples_per_stratum: int,
    lora_parameters: list[nn.Parameter],
    layer_groups: dict[int, list[int]],
    stage: str,
) -> list[dict[str, Any]]:
    model.train()
    rows = []
    for batch_index, (waveforms, sample_masks, targets, sample_ids) in enumerate(loader):
        targets = targets.to(device)
        logits = _forward_batch(model, waveforms, sample_masks, device)
        event_losses = _per_event_classification_loss(logits, targets)
        masks = domain_class_masks(
            sample_ids,
            targets,
            expected_domains=domains,
            samples_per_stratum=samples_per_stratum,
        )
        stratum_order = [
            (domains[0], 0),
            (domains[1], 0),
            (domains[0], 1),
            (domains[1], 1),
        ]
        gradients: dict[tuple[str, int], tuple[Tensor | None, ...]] = {}
        for loss_index, stratum in enumerate(stratum_order):
            stratum_loss = event_losses[masks[stratum]].mean()
            gradients[stratum] = torch.autograd.grad(
                stratum_loss,
                lora_parameters,
                retain_graph=loss_index < len(stratum_order) - 1,
                allow_unused=True,
            )
        normal_cosine = gradient_cosine(
            gradients[(domains[0], 0)],
            gradients[(domains[1], 0)],
        )
        adventitious_cosine = gradient_cosine(
            gradients[(domains[0], 1)],
            gradients[(domains[1], 1)],
        )
        if normal_cosine is None or adventitious_cosine is None:
            raise RuntimeError("Full LoRA gradient cosine is undefined")
        first_domain = mean_gradients(
            gradients[(domains[0], 0)],
            gradients[(domains[0], 1)],
        )
        second_domain = mean_gradients(
            gradients[(domains[1], 0)],
            gradients[(domains[1], 1)],
        )
        aggregate_cosine = gradient_cosine(first_domain, second_domain)
        if aggregate_cosine is None:
            raise RuntimeError("Aggregate domain gradient cosine is undefined")
        gradient_norms = {
            domain: {
                "normal": gradient_norm(gradients[(domain, 0)]),
                "adventitious": gradient_norm(gradients[(domain, 1)]),
            }
            for domain in domains
        }
        class_domain_metrics = {}
        for class_index, class_name in enumerate(("normal", "adventitious")):
            first_gradient = gradients[(domains[0], class_index)]
            second_gradient = gradients[(domains[1], class_index)]
            first_norm = gradient_norms[domains[0]][class_name]
            second_norm = gradient_norms[domains[1]][class_name]
            same_class_dot = gradient_dot(first_gradient, second_gradient)
            combined_square = (
                first_norm**2
                + second_norm**2
                + 2.0 * same_class_dot
            )
            if combined_square <= 0.0:
                raise RuntimeError("Degenerate same-class combined gradient")
            norm_sum = first_norm + second_norm
            class_domain_metrics[class_name] = {
                "norm_shares": {
                    domains[0]: first_norm / norm_sum,
                    domains[1]: second_norm / norm_sum,
                },
                "directional_shares": {
                    domains[0]: (
                        first_norm**2 + same_class_dot
                    ) / combined_square,
                    domains[1]: (
                        second_norm**2 + same_class_dot
                    ) / combined_square,
                },
                f"signed_log_{domains[0]}_over_{domains[1]}_norm_ratio": (
                    math.log(first_norm / second_norm)
                ),
                "max_to_min_norm_ratio": (
                    max(first_norm, second_norm)
                    / min(first_norm, second_norm)
                ),
            }
        same_class_sum = (
            gradient_dot(
                gradients[(domains[0], 0)],
                gradients[(domains[1], 0)],
            )
            + gradient_dot(
                gradients[(domains[0], 1)],
                gradients[(domains[1], 1)],
            )
        )
        cross_class_sum = (
            gradient_dot(
                gradients[(domains[0], 0)],
                gradients[(domains[1], 1)],
            )
            + gradient_dot(
                gradients[(domains[0], 1)],
                gradients[(domains[1], 0)],
            )
        )
        aggregate_dot = gradient_dot(first_domain, second_domain)
        reconstructed_dot = 0.25 * (same_class_sum + cross_class_sum)
        layer_cosines = {}
        for layer, indices in sorted(layer_groups.items()):
            layer_cosines[str(layer)] = {
                "normal": gradient_cosine(
                    gradients[(domains[0], 0)],
                    gradients[(domains[1], 0)],
                    indices=indices,
                ),
                "adventitious": gradient_cosine(
                    gradients[(domains[0], 1)],
                    gradients[(domains[1], 1)],
                    indices=indices,
                ),
                "aggregate": gradient_cosine(
                    first_domain,
                    second_domain,
                    indices=indices,
                ),
            }
        row = {
            "stage": stage,
            "batch_index": batch_index,
            "normal_cosine": normal_cosine,
            "adventitious_cosine": adventitious_cosine,
            "aggregate_domain_cosine": aggregate_cosine,
            "gradient_norms": gradient_norms,
            "class_domain_metrics": class_domain_metrics,
            "cross_class_cosines": {
                f"{domains[0]}_normal__{domains[1]}_adventitious": (
                    gradient_cosine(
                        gradients[(domains[0], 0)],
                        gradients[(domains[1], 1)],
                    )
                ),
                f"{domains[0]}_adventitious__{domains[1]}_normal": (
                    gradient_cosine(
                        gradients[(domains[0], 1)],
                        gradients[(domains[1], 0)],
                    )
                ),
            },
            "dot_decomposition": {
                "same_class_sum": same_class_sum,
                "cross_class_sum": cross_class_sum,
                "aggregate_dot": aggregate_dot,
                "reconstructed_aggregate_dot": reconstructed_dot,
                "reconstruction_error": aggregate_dot - reconstructed_dot,
            },
            "projection_drift": float(
                projection_drift_regularization(model).detach().cpu()
            ),
            "finite": all(
                bool(torch.isfinite(value).all().detach().cpu())
                for value in (logits, event_losses)
            ),
            "layer_cosines": layer_cosines,
        }
        rows.append(row)
        if (batch_index + 1) % 16 == 0 or batch_index + 1 == len(loader):
            print(
                json.dumps({
                    "stage": stage,
                    "completed_batches": batch_index + 1,
                    "total_batches": len(loader),
                }),
                flush=True,
            )
    return rows


def _training_prefix(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    consistency_weight: float,
    clip_norm: float,
) -> dict[str, float]:
    model.train()
    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    totals = {
        "classification_loss": 0.0,
        "consistency_loss": 0.0,
        "projection_drift": 0.0,
    }
    for update, (waveforms, sample_masks, targets, _) in enumerate(loader, start=1):
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_batch(model, waveforms, sample_masks, device)
        event_losses = _per_event_classification_loss(logits, targets)
        classification_loss = event_losses.mean()
        consistency_loss = jensen_shannon_consistency(logits)
        drift = projection_drift_regularization(model)
        loss = classification_loss + consistency_weight * consistency_loss
        if not bool(torch.isfinite(loss).detach().cpu()):
            raise FloatingPointError(f"Non-finite Gate 9C prefix update {update}")
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, max_norm=clip_norm)
        optimizer.step()
        totals["classification_loss"] += float(classification_loss.detach().cpu())
        totals["consistency_loss"] += float(consistency_loss.detach().cpu())
        totals["projection_drift"] += float(drift.detach().cpu())
        if update % 64 == 0 or update == len(loader):
            print(
                json.dumps({
                    "stage": "ordinary_lora_prefix",
                    "completed_updates": update,
                    "total_updates": len(loader),
                }),
                flush=True,
            )
    return {
        key: value / len(loader)
        for key, value in totals.items()
    }


def _gate_decision(
    summary: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "complete_finite_batches": (
            int(summary["batches"]) == int(gate["complete_finite_batches"])
            and bool(summary["all_finite"])
        ),
        "any_class_conflict_fraction": (
            float(summary["any_class_conflict_fraction"])
            >= float(gate["any_class_conflict_fraction_at_least"])
        ),
        "maximum_single_class_conflict_fraction": (
            max(float(value) for value in summary["class_conflict_fraction"].values())
            >= float(gate["maximum_single_class_conflict_fraction_at_least"])
        ),
        "hidden_class_conflict_fraction": (
            float(summary["hidden_class_conflict_fraction"])
            >= float(gate["hidden_class_conflict_fraction_at_least"])
        ),
        "median_conflict_severity": (
            float(summary["median_conflict_severity"])
            >= float(gate["median_conflict_severity_at_least"])
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
    }


def _norm_balance_gate_decision(
    summary: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    class_checks = {}
    for class_name, metrics in summary["classes"].items():
        checks = {
            "median_imbalance_factor": (
                float(metrics["median_imbalance_factor"])
                >= float(gate["median_imbalance_factor_at_least"])
            ),
            "dominant_domain_fraction": (
                float(metrics["dominant_domain_fraction"])
                >= float(gate["dominant_domain_fraction_at_least"])
            ),
            "first_half_dominant_fraction": (
                float(metrics["first_half_dominant_fraction"])
                >= float(gate["each_half_dominant_fraction_at_least"])
            ),
            "second_half_dominant_fraction": (
                float(metrics["second_half_dominant_fraction"])
                >= float(gate["each_half_dominant_fraction_at_least"])
            ),
            "same_sign_blocks": (
                int(metrics["same_sign_block_count"])
                >= int(gate["same_sign_blocks_at_least"])
            ),
            "dominant_norm_share": (
                float(
                    metrics["median_norm_share"][
                        metrics["dominant_domain"]
                    ]
                )
                >= float(gate["dominant_norm_share_at_least"])
            ),
            "same_class_median_cosine": (
                float(metrics["median_same_class_cosine"])
                >= float(gate["same_class_median_cosine_at_least"])
            ),
        }
        class_checks[class_name] = {
            "checks": checks,
            "passed": all(checks.values()),
        }
    heterogeneity = summary["class_conditioning_heterogeneity"]
    amplitude_heterogeneity = (
        float(heterogeneity["median"])
        >= float(gate["heterogeneity_median_at_least"])
        and sum(
            float(value) >= float(gate["heterogeneity_block_median_at_least"])
            for value in heterogeneity["block_medians"]
        )
        >= int(gate["heterogeneity_blocks_at_least"])
    )
    normal = class_checks["normal"]
    adventitious = class_checks["adventitious"]
    dominant_flip = (
        normal["passed"]
        and adventitious["passed"]
        and (
            summary["classes"]["normal"]["dominant_domain"]
            != summary["classes"]["adventitious"]["dominant_domain"]
        )
    )
    class_conditioning = {
        "amplitude_heterogeneity": amplitude_heterogeneity,
        "dominant_domain_flip": dominant_flip,
        "passed": amplitude_heterogeneity or dominant_flip,
    }
    any_class_imbalance = any(
        value["passed"] for value in class_checks.values()
    )
    weak_domain = str(gate["weak_domain"])
    strong_domain = str(gate["known_strong_domain"])
    weak_domain_protection = any(
        value["passed"]
        and summary["classes"][class_name]["dominant_domain"] == strong_domain
        and float(
            summary["classes"][class_name]["median_norm_share"][strong_domain]
        )
        >= float(gate["dominant_norm_share_at_least"])
        and float(
            summary["classes"][class_name]["median_directional_share"][
                strong_domain
            ]
        )
        >= float(gate["dominant_directional_share_at_least"])
        for class_name, value in class_checks.items()
    )
    complete = int(summary["batches"]) == int(gate["complete_batches"])
    return {
        "complete_batches": complete,
        "class_checks": class_checks,
        "any_class_imbalance_passed": any_class_imbalance,
        "class_conditioning": class_conditioning,
        "weak_domain": weak_domain,
        "known_strong_domain": strong_domain,
        "weak_domain_protection_passed": (
            complete
            and any_class_imbalance
            and class_conditioning["passed"]
            and weak_domain_protection
        ),
        "passed": (
            complete
            and any_class_imbalance
            and class_conditioning["passed"]
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    data_config_path = (root / args.data_config).resolve()
    model_config = yaml.safe_load(
        (root / args.model_config).read_text(encoding="utf-8")
    )
    experiment = yaml.safe_load(
        (root / args.experiment_config).read_text(encoding="utf-8")
    )
    seed = int(experiment["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    if args.device == "mps" and device.type != "mps":
        raise SystemExit("MPS was requested but is unavailable")

    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    train_role = str(experiment["train_value"])
    _assert_development_protocol(manifest, train_role, "validation_select")
    feature_config = feature_config_from_yaml(data_config_path)
    label_column = str(experiment["label_column"])
    dataset = ICBHICycleDataset(
        manifest_path=manifest_path,
        project_root=root,
        split_column="protocol_role",
        split_value=train_role,
        feature_config=feature_config,
        training=True,
        num_views=int(experiment["num_views"]),
        augmentation=_augmentation(experiment),
        label_column=label_column,
        return_waveform=True,
        waveform_only=True,
    )
    domains = tuple(sorted(str(value) for value in dataset.rows["dataset"].unique()))
    if domains != ("icbhi2017", "sprsound2022"):
        raise ValueError(f"Unexpected Gate 9C domains: {domains}")
    batch_size = int(experiment["batch_size"])
    samples_per_stratum = batch_size // 4

    checkpoint = (root / str(model_config["checkpoint"])).resolve()
    source_dir = (root / str(model_config["source_dir"])).resolve()
    model = load_beats_transfer(checkpoint, source_dir)
    adaptation = model_config["adaptation"]
    audit = configure_beats_adaptation(
        model,
        strategy="lora_qv",
        lora_last_n_layers=int(adaptation["last_n_layers"]),
        lora_rank=int(adaptation["rank"]),
        lora_alpha=float(adaptation["alpha"]),
        lora_dropout=float(adaptation["dropout"]),
    )
    adaptation_parameters, classifier_parameters = _trainable_parameter_groups(model)
    lora_names, lora_parameters, layer_groups = named_lora_parameters(model)
    if sum(parameter.numel() for parameter in lora_parameters) != 98_304:
        raise ValueError("Unexpected Gate 9C LoRA parameter count")
    model = model.to(device)

    audit_config = experiment["audit"]
    initial_loader = _loader(
        dataset,
        batch_size=batch_size,
        batches=int(audit_config["initialization_batches"]),
        seed=seed + int(audit_config["initialization_sampler_seed_offset"]),
        label_column=label_column,
    )
    initialization_rows = _audit_batches(
        model,
        initial_loader,
        device=device,
        domains=domains,
        samples_per_stratum=samples_per_stratum,
        lora_parameters=lora_parameters,
        layer_groups=layer_groups,
        stage="initialization",
    )

    prefix_config = experiment["prefix"]
    prefix_loader = _loader(
        dataset,
        batch_size=batch_size,
        batches=int(prefix_config["updates"]),
        seed=seed,
        label_column=label_column,
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": adaptation_parameters,
                "lr": float(prefix_config["adaptation_learning_rate"]),
            },
            {
                "params": classifier_parameters,
                "lr": float(prefix_config["head_learning_rate"]),
            },
        ],
        weight_decay=float(prefix_config["weight_decay"]),
    )
    prefix_summary = _training_prefix(
        model,
        prefix_loader,
        optimizer,
        device=device,
        consistency_weight=float(prefix_config["consistency_weight"]),
        clip_norm=float(prefix_config["gradient_clip_norm"]),
    )
    if args.prefix_checkpoint is not None:
        prefix_checkpoint = (root / args.prefix_checkpoint).resolve()
        prefix_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "gate": "9C-A2",
                "seed": seed,
                "updates": int(prefix_config["updates"]),
                "trainable_model_state": {
                    name: parameter.detach().cpu()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                },
                "optimizer_state": optimizer.state_dict(),
            },
            prefix_checkpoint,
        )

    post_loader = _loader(
        dataset,
        batch_size=batch_size,
        batches=int(audit_config["post_prefix_batches"]),
        seed=seed + int(audit_config["post_prefix_sampler_seed_offset"]),
        label_column=label_column,
    )
    post_rows = _audit_batches(
        model,
        post_loader,
        device=device,
        domains=domains,
        samples_per_stratum=samples_per_stratum,
        lora_parameters=lora_parameters,
        layer_groups=layer_groups,
        stage="post_prefix",
    )
    initialization_summary = summarize_conflict_rows(initialization_rows)
    post_summary = summarize_conflict_rows(post_rows)
    decision = _gate_decision(post_summary, experiment["gate"])
    initialization_norm_balance = summarize_norm_balance_rows(
        initialization_rows,
        domains=domains,
    )
    post_norm_balance = summarize_norm_balance_rows(
        post_rows,
        domains=domains,
    )
    norm_balance_decision = (
        _norm_balance_gate_decision(
            post_norm_balance,
            experiment["norm_balance_gate"],
        )
        if "norm_balance_gate" in experiment
        else None
    )
    result = {
        "gate": "9C-A",
        "purpose": "training_only_class_conditional_gradient_conflict_audit",
        "seed": seed,
        "device": str(device),
        "validation_select_accessed": False,
        "calibration_accessed": False,
        "locked_test_accessed": False,
        "configuration": experiment,
        "model_audit": audit.to_dict(),
        "lora_parameter_names": lora_names,
        "lora_parameter_count": sum(
            parameter.numel() for parameter in lora_parameters
        ),
        "layer_parameter_counts": {
            str(layer): sum(lora_parameters[index].numel() for index in indices)
            for layer, indices in layer_groups.items()
        },
        "prefix_summary": prefix_summary,
        "initialization": {
            "summary": initialization_summary,
            "norm_balance": initialization_norm_balance,
            "rows": initialization_rows,
        },
        "post_prefix": {
            "summary": post_summary,
            "norm_balance": post_norm_balance,
            "rows": post_rows,
        },
        "decision": decision,
        "norm_balance_decision": norm_balance_decision,
        "code_audit": {
            "manifest_sha256": _sha256(manifest_path),
            "upstream_checkpoint_sha256": _sha256(checkpoint),
            "beats_backbone_source_sha256": _sha256(source_dir / "backbone.py"),
            "beats_transfer_wrapper_sha256": _sha256(
                root / "src/respiratory_sound/models/pretrained_audio.py"
            ),
            "gradient_conflict_module_sha256": _sha256(
                root / "src/respiratory_sound/gradient_conflict.py"
            ),
            "freeze_artifact": "artifacts/gate9c_gradient_conflict_freeze.json",
        },
    }
    if device.type == "mps":
        result["resource"] = {
            "mps_allocated_gib": (
                float(torch.mps.current_allocated_memory()) / 1024**3
            ),
            "mps_driver_allocated_gib": (
                float(torch.mps.driver_allocated_memory()) / 1024**3
            ),
        }
    output_path = (root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "post_prefix_summary": post_summary,
        "post_prefix_norm_balance": post_norm_balance,
        "decision": decision,
        "norm_balance_decision": norm_balance_decision,
    }, indent=2))


if __name__ == "__main__":
    main()
