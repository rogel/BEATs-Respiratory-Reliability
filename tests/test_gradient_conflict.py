from __future__ import annotations

import math

import torch

from respiratory_sound.gradient_conflict import (
    domain_class_masks,
    gradient_cosine,
    gradient_dot,
    gradient_norm,
    mean_gradients,
    sample_domains,
    summarize_conflict_rows,
    summarize_norm_balance_rows,
)


def test_domain_class_masks_require_exact_balance() -> None:
    sample_ids = [
        "a::0",
        "a::1",
        "a::2",
        "a::3",
        "b::0",
        "b::1",
        "b::2",
        "b::3",
    ]
    targets = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    masks = domain_class_masks(
        sample_ids,
        targets,
        expected_domains=("a", "b"),
        samples_per_stratum=2,
    )
    assert sample_domains(sample_ids) == ["a"] * 4 + ["b"] * 4
    assert all(int(mask.sum()) == 2 for mask in masks.values())


def test_gradient_cosine_handles_conflict_and_unused_parameters() -> None:
    left = (torch.tensor([1.0, 0.0]), None)
    right = (torch.tensor([-1.0, 0.0]), torch.tensor([2.0]))
    assert math.isclose(
        gradient_cosine(left, right),
        -1.0 / math.sqrt(5.0),
    )
    averaged = mean_gradients(left, right)
    assert torch.equal(averaged[0], torch.zeros(2))
    assert torch.equal(averaged[1], torch.tensor([1.0]))


def test_conflict_summary_detects_hidden_class_conflict() -> None:
    rows = [
        {
            "normal_cosine": -0.2,
            "adventitious_cosine": 0.4,
            "aggregate_domain_cosine": 0.1,
            "finite": True,
            "layer_cosines": {},
        },
        {
            "normal_cosine": 0.3,
            "adventitious_cosine": 0.2,
            "aggregate_domain_cosine": 0.2,
            "finite": True,
            "layer_cosines": {},
        },
    ]
    summary = summarize_conflict_rows(rows)
    assert summary["any_class_conflict_fraction"] == 0.5
    assert summary["hidden_class_conflict_fraction"] == 0.5
    assert summary["aggregate_domain_conflict_fraction"] == 0.0
    assert summary["median_conflict_severity"] == 0.2


def test_gradient_dot_and_norm_reconstruct_mean_gradient_dot() -> None:
    first_normal = (torch.tensor([2.0, 0.0]),)
    first_adventitious = (torch.tensor([0.0, 2.0]),)
    second_normal = (torch.tensor([1.0, 0.0]),)
    second_adventitious = (torch.tensor([0.0, -1.0]),)
    first_domain = mean_gradients(first_normal, first_adventitious)
    second_domain = mean_gradients(second_normal, second_adventitious)
    same_class = (
        gradient_dot(first_normal, second_normal)
        + gradient_dot(first_adventitious, second_adventitious)
    )
    cross_class = (
        gradient_dot(first_normal, second_adventitious)
        + gradient_dot(first_adventitious, second_normal)
    )
    assert gradient_norm(first_normal) == 2.0
    assert math.isclose(
        gradient_dot(first_domain, second_domain),
        0.25 * (same_class + cross_class),
    )


def test_norm_balance_summary_requires_persistent_imbalance() -> None:
    rows = []
    for index in range(4):
        rows.append({
            "normal_cosine": 0.8,
            "adventitious_cosine": 0.7,
            "aggregate_domain_cosine": -0.1 if index < 2 else 0.1,
            "gradient_norms": {
                "a": {"normal": 1.0, "adventitious": 4.0},
                "b": {"normal": 2.0, "adventitious": 2.0},
            },
            "class_domain_metrics": {
                "normal": {
                    "directional_shares": {"a": 0.35, "b": 0.65},
                },
                "adventitious": {
                    "directional_shares": {"a": 0.65, "b": 0.35},
                },
            },
            "dot_decomposition": {
                "same_class_sum": 1.0,
                "cross_class_sum": -2.0,
                "reconstruction_error": 0.0,
            },
        })
    summary = summarize_norm_balance_rows(rows, domains=("a", "b"))
    assert summary["classes"]["normal"]["dominant_domain"] == "b"
    assert summary["classes"]["normal"]["median_imbalance_factor"] == 2.0
    assert summary["classes"]["adventitious"]["dominant_domain"] == "a"
    assert summary["classes"]["adventitious"]["median_imbalance_factor"] == 2.0
    assert math.isclose(
        summary["class_conditioning_heterogeneity"]["median"],
        math.log(4.0),
    )
    assert (
        summary["aggregate_negative_explained_by_cross_class_fraction"]
        == 1.0
    )
