"""Utilities for class-conditional cross-domain gradient-conflict audits."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

_LAYER_PATTERN = re.compile(r"\.encoder\.layers\.(\d+)\.")


def sample_domains(sample_ids: Sequence[str]) -> list[str]:
    """Extract and validate database prefixes from unified sample identifiers."""
    domains = []
    for sample_id in sample_ids:
        pieces = str(sample_id).split("::", maxsplit=1)
        if len(pieces) != 2 or not all(pieces):
            raise ValueError(f"Invalid unified sample identifier: {sample_id}")
        domains.append(pieces[0])
    return domains


def domain_class_masks(
    sample_ids: Sequence[str],
    targets: Tensor,
    *,
    expected_domains: Sequence[str],
    samples_per_stratum: int,
) -> dict[tuple[str, int], Tensor]:
    """Build exact database-by-class masks for one balanced training batch."""
    domains = sample_domains(sample_ids)
    target_values = targets.detach().cpu().tolist()
    if len(domains) != len(target_values):
        raise ValueError("Sample identifiers and targets have different lengths")
    masks = {}
    for domain in expected_domains:
        for class_index in (0, 1):
            mask = torch.tensor(
                [
                    sample_domain == domain and int(target) == class_index
                    for sample_domain, target in zip(
                        domains,
                        target_values,
                        strict=True,
                    )
                ],
                dtype=torch.bool,
                device=targets.device,
            )
            if int(mask.sum().detach().cpu()) != samples_per_stratum:
                raise ValueError(
                    "Gate 9C batch is not exactly balanced for "
                    f"{domain}, class {class_index}"
                )
            masks[(str(domain), class_index)] = mask
    return masks


def named_lora_parameters(
    model: nn.Module,
) -> tuple[list[str], list[nn.Parameter], dict[int, list[int]]]:
    """Return LoRA parameters and their encoder-layer index groups."""
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (".lora_a." in name or ".lora_b." in name)
    ]
    if not named:
        raise ValueError("No trainable LoRA parameters were found")
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    groups: dict[int, list[int]] = {}
    for parameter_index, name in enumerate(names):
        match = _LAYER_PATTERN.search(name)
        if match is None:
            raise ValueError(f"Cannot resolve encoder layer from LoRA parameter: {name}")
        groups.setdefault(int(match.group(1)), []).append(parameter_index)
    return names, parameters, groups


def gradient_cosine(
    left: Sequence[Tensor | None],
    right: Sequence[Tensor | None],
    *,
    indices: Sequence[int] | None = None,
) -> float | None:
    """Compute a finite cosine while treating unused parameters as zero."""
    if len(left) != len(right):
        raise ValueError("Gradient tuples have different lengths")
    selected = range(len(left)) if indices is None else indices
    dot = 0.0
    left_square = 0.0
    right_square = 0.0
    for index in selected:
        left_value = left[index]
        right_value = right[index]
        if left_value is not None:
            if not bool(torch.isfinite(left_value).all().detach().cpu()):
                raise FloatingPointError("Non-finite left gradient")
            left_square += float(left_value.detach().float().square().sum().cpu())
        if right_value is not None:
            if not bool(torch.isfinite(right_value).all().detach().cpu()):
                raise FloatingPointError("Non-finite right gradient")
            right_square += float(right_value.detach().float().square().sum().cpu())
        if left_value is not None and right_value is not None:
            dot += float(
                (left_value.detach().float() * right_value.detach().float()).sum().cpu()
            )
    if left_square <= 0.0 or right_square <= 0.0:
        return None
    cosine = dot / math.sqrt(left_square * right_square)
    return float(max(-1.0, min(1.0, cosine)))


def gradient_dot(
    left: Sequence[Tensor | None],
    right: Sequence[Tensor | None],
    *,
    indices: Sequence[int] | None = None,
) -> float:
    """Return the inner product of two gradient tuples."""
    if len(left) != len(right):
        raise ValueError("Gradient tuples have different lengths")
    selected = range(len(left)) if indices is None else indices
    dot = 0.0
    for index in selected:
        left_value = left[index]
        right_value = right[index]
        if left_value is None or right_value is None:
            continue
        if not bool(torch.isfinite(left_value).all().detach().cpu()):
            raise FloatingPointError("Non-finite left gradient")
        if not bool(torch.isfinite(right_value).all().detach().cpu()):
            raise FloatingPointError("Non-finite right gradient")
        dot += float(
            (left_value.detach().float() * right_value.detach().float()).sum().cpu()
        )
    return dot


def gradient_norm(
    gradients: Sequence[Tensor | None],
    *,
    indices: Sequence[int] | None = None,
) -> float:
    """Return the Euclidean norm of a gradient tuple."""
    selected = range(len(gradients)) if indices is None else indices
    square = 0.0
    for index in selected:
        value = gradients[index]
        if value is None:
            continue
        if not bool(torch.isfinite(value).all().detach().cpu()):
            raise FloatingPointError("Non-finite gradient")
        square += float(value.detach().float().square().sum().cpu())
    return math.sqrt(square)


def mean_gradients(
    first: Sequence[Tensor | None],
    second: Sequence[Tensor | None],
) -> tuple[Tensor | None, ...]:
    """Average two gradient tuples, preserving unused entries."""
    if len(first) != len(second):
        raise ValueError("Gradient tuples have different lengths")
    averaged = []
    for first_value, second_value in zip(first, second, strict=True):
        if first_value is None and second_value is None:
            averaged.append(None)
        elif first_value is None:
            averaged.append(second_value * 0.5)
        elif second_value is None:
            averaged.append(first_value * 0.5)
        else:
            averaged.append((first_value + second_value) * 0.5)
    return tuple(averaged)


def summarize_norm_balance_rows(
    rows: Sequence[dict[str, Any]],
    *,
    domains: Sequence[str],
) -> dict[str, Any]:
    """Summarize persistent class-by-domain LoRA gradient-energy imbalance."""
    if not rows:
        raise ValueError("Norm-balance audit rows must not be empty")
    if len(domains) != 2:
        raise ValueError("Norm-balance audit requires exactly two domains")
    first_domain, second_domain = (str(domain) for domain in domains)
    class_names = ("normal", "adventitious")
    class_summary = {}
    signed_log_ratios = {}
    for class_name in class_names:
        first_norms = np.asarray(
            [
                float(row["gradient_norms"][first_domain][class_name])
                for row in rows
            ],
            dtype=np.float64,
        )
        second_norms = np.asarray(
            [
                float(row["gradient_norms"][second_domain][class_name])
                for row in rows
            ],
            dtype=np.float64,
        )
        if np.any(first_norms <= 0.0) or np.any(second_norms <= 0.0):
            raise ValueError("Gradient norms must be strictly positive")
        ratios = first_norms / second_norms
        signed_log_ratios[class_name] = np.log(ratios)
        median_log_ratio = float(np.median(np.log(ratios)))
        dominant_domain = (
            first_domain if median_log_ratio >= 0.0 else second_domain
        )
        dominance = ratios >= 1.0 if dominant_domain == first_domain else ratios < 1.0
        midpoint = len(rows) // 2
        median_first = float(np.median(first_norms))
        median_second = float(np.median(second_norms))
        inverse_first = 1.0 / median_first
        inverse_second = 1.0 / median_second
        inverse_sum = inverse_first + inverse_second
        norm_shares = {
            first_domain: first_norms / (first_norms + second_norms),
            second_domain: second_norms / (first_norms + second_norms),
        }
        block_log_ratio_medians = [
            float(np.median(block))
            for block in np.array_split(np.log(ratios), 4)
        ]
        class_summary[class_name] = {
            "median_norm": {
                first_domain: median_first,
                second_domain: median_second,
            },
            f"median_{first_domain}_over_{second_domain}_ratio": float(
                np.median(ratios)
            ),
            "median_imbalance_factor": float(
                np.median(np.maximum(ratios, 1.0 / ratios))
            ),
            "dominant_domain": dominant_domain,
            "dominant_domain_fraction": float(np.mean(dominance)),
            "first_half_dominant_fraction": float(np.mean(dominance[:midpoint])),
            "second_half_dominant_fraction": float(np.mean(dominance[midpoint:])),
            "same_sign_block_count": int(
                sum(
                    (
                        block_median >= 0.0
                        if dominant_domain == first_domain
                        else block_median < 0.0
                    )
                    for block_median in block_log_ratio_medians
                )
            ),
            "block_log_ratio_medians": block_log_ratio_medians,
            "median_norm_share": {
                domain: float(np.median(values))
                for domain, values in norm_shares.items()
            },
            "median_directional_share": {
                domain: float(
                    np.median(
                        [
                            float(
                                row["class_domain_metrics"][class_name][
                                    "directional_shares"
                                ][domain]
                            )
                            for row in rows
                        ]
                    )
                )
                for domain in (first_domain, second_domain)
            },
            "median_same_class_cosine": float(
                np.median(
                    [
                        float(row[f"{class_name}_cosine"])
                        for row in rows
                    ]
                )
            ),
            "inverse_median_norm_weights_sum_to_two": {
                first_domain: float(2.0 * inverse_first / inverse_sum),
                second_domain: float(2.0 * inverse_second / inverse_sum),
            },
        }
    heterogeneity = np.abs(
        signed_log_ratios["adventitious"] - signed_log_ratios["normal"]
    )
    aggregate_negative_rows = [
        row
        for row in rows
        if float(row["aggregate_domain_cosine"]) < 0.0
    ]
    cross_class_explanations = [
        (
            float(row["dot_decomposition"]["cross_class_sum"])
            < -float(row["dot_decomposition"]["same_class_sum"])
        )
        for row in aggregate_negative_rows
    ]
    return {
        "batches": len(rows),
        "domains": [first_domain, second_domain],
        "classes": class_summary,
        "class_conditioning_heterogeneity": {
            "median": float(np.median(heterogeneity)),
            "block_medians": [
                float(np.median(block))
                for block in np.array_split(heterogeneity, 4)
            ],
            "fraction_at_least_log_1_10": float(
                np.mean(heterogeneity >= math.log(1.10))
            ),
        },
        "aggregate_negative_batches": len(aggregate_negative_rows),
        "aggregate_negative_explained_by_cross_class_fraction": (
            float(np.mean(cross_class_explanations))
            if cross_class_explanations
            else None
        ),
        "maximum_dot_reconstruction_error": float(
            max(
                abs(float(row["dot_decomposition"]["reconstruction_error"]))
                for row in rows
            )
        ),
    }


def summarize_conflict_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate batch-level conflict diagnostics."""
    if not rows:
        raise ValueError("Conflict audit rows must not be empty")
    class_names = ("normal", "adventitious")
    class_conflicts = {
        class_name: [
            float(row[f"{class_name}_cosine"]) < 0.0
            for row in rows
        ]
        for class_name in class_names
    }
    any_class_conflict = [
        any(class_conflicts[class_name][index] for class_name in class_names)
        for index in range(len(rows))
    ]
    aggregate_conflicts = [
        float(row["aggregate_domain_cosine"]) < 0.0
        for row in rows
    ]
    hidden_conflicts = [
        not aggregate_conflicts[index] and any_class_conflict[index]
        for index in range(len(rows))
    ]
    severities = [
        -float(row[f"{class_name}_cosine"])
        for row in rows
        for class_name in class_names
        if float(row[f"{class_name}_cosine"]) < 0.0
    ]
    layer_indices = sorted(
        {
            int(layer)
            for row in rows
            for layer in row.get("layer_cosines", {})
        }
    )
    layer_summary = {}
    for layer in layer_indices:
        values = [
            float(cosine)
            for row in rows
            for cosine in row.get("layer_cosines", {}).get(str(layer), {}).values()
            if cosine is not None
        ]
        layer_summary[str(layer)] = {
            "active_cosines": len(values),
            "conflict_fraction": (
                float(np.mean(np.asarray(values) < 0.0)) if values else None
            ),
            "median_cosine": float(np.median(values)) if values else None,
        }
    return {
        "batches": len(rows),
        "all_finite": all(bool(row["finite"]) for row in rows),
        "class_conflict_fraction": {
            class_name: float(np.mean(class_conflicts[class_name]))
            for class_name in class_names
        },
        "any_class_conflict_fraction": float(np.mean(any_class_conflict)),
        "aggregate_domain_conflict_fraction": float(np.mean(aggregate_conflicts)),
        "hidden_class_conflict_fraction": float(np.mean(hidden_conflicts)),
        "median_conflict_severity": (
            float(np.median(severities)) if severities else 0.0
        ),
        "class_median_cosine": {
            class_name: float(
                np.median(
                    [float(row[f"{class_name}_cosine"]) for row in rows]
                )
            )
            for class_name in class_names
        },
        "aggregate_domain_median_cosine": float(
            np.median(
                [float(row["aggregate_domain_cosine"]) for row in rows]
            )
        ),
        "layer_summary": layer_summary,
    }
