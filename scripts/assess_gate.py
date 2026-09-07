#!/usr/bin/env python3
"""Apply explicit continuation gates to completed experiment summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="gate", required=True)

    overfit = subparsers.add_parser("overfit")
    overfit.add_argument("--summary", type=Path, required=True)
    overfit.add_argument("--minimum-accuracy", type=float, default=0.95)

    method = subparsers.add_parser("method")
    method.add_argument("--plain", type=Path, required=True)
    method.add_argument("--repconv", type=Path, required=True)
    method.add_argument("--tfdcr", type=Path, required=True)
    method.add_argument("--minimum-plain-gain", type=float, default=0.005)
    method.add_argument("--require-macro-f1-gain", action="store_true")
    method.add_argument("--output", type=Path)

    confirmation = subparsers.add_parser("confirmation")
    confirmation.add_argument("--plain", type=Path, required=True)
    confirmation.add_argument("--repconv", type=Path, required=True)
    confirmation.add_argument("--tfdcr", type=Path, required=True)
    confirmation.add_argument("--minimum-plain-gain", type=float, default=0.01)
    confirmation.add_argument("--minimum-repconv-wins", type=int, default=2)
    confirmation.add_argument("--output", type=Path)
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def icbhi_score(metrics: dict[str, object]) -> float:
    if "icbhi_score" in metrics:
        return float(metrics["icbhi_score"])
    return float(metrics["average_score"])


def assess_overfit(args: argparse.Namespace) -> dict[str, object]:
    summary = load_summary(args.summary)
    accuracy = float(summary["best_validation_metrics"]["accuracy"])
    return {
        "gate": "tiny_subset_overfit",
        "accuracy": accuracy,
        "minimum_accuracy": args.minimum_accuracy,
        "passed": accuracy >= args.minimum_accuracy,
        "next_action": (
            "run_pipeline_smoke"
            if accuracy >= args.minimum_accuracy
            else "debug_data_labels_or_optimization"
        ),
    }


def assess_method(args: argparse.Namespace) -> dict[str, object]:
    summaries = {
        "plain": load_summary(args.plain),
        "repconv": load_summary(args.repconv),
        "tfdcr": load_summary(args.tfdcr),
    }
    metrics = {model: summary["best_validation_metrics"] for model, summary in summaries.items()}
    score_gain_plain = icbhi_score(metrics["tfdcr"]) - icbhi_score(metrics["plain"])
    score_gain_repconv = icbhi_score(metrics["tfdcr"]) - icbhi_score(metrics["repconv"])
    macro_f1_gain = float(metrics["tfdcr"]["macro_f1"]) - float(metrics["plain"]["macro_f1"])
    checks = {
        "minimum_plain_score_gain": score_gain_plain >= args.minimum_plain_gain,
        "better_than_repconv": score_gain_repconv > 0,
        "macro_f1_not_worse": (macro_f1_gain >= 0 if args.require_macro_f1_gain else True),
    }
    passed = all(checks.values())
    return {
        "gate": "single_seed_method",
        "metrics": metrics,
        "score_gain_over_plain": score_gain_plain,
        "score_gain_over_repconv": score_gain_repconv,
        "macro_f1_gain_over_plain": macro_f1_gain,
        "checks": checks,
        "passed": passed,
        "next_action": (
            "run_three_seed_confirmation"
            if passed
            else "inspect_failure_and_do_not_expand_experiments"
        ),
    }


def assess_confirmation(args: argparse.Namespace) -> dict[str, object]:
    aggregates = {
        "plain": load_summary(args.plain),
        "repconv": load_summary(args.repconv),
        "tfdcr": load_summary(args.tfdcr),
    }
    values = {
        model: aggregate["metrics"]["icbhi_score"]["values"]
        for model, aggregate in aggregates.items()
    }
    macro_f1_means = {
        model: float(aggregate["metrics"]["macro_f1"]["mean"])
        for model, aggregate in aggregates.items()
    }
    means = {
        model: float(aggregate["metrics"]["icbhi_score"]["mean"])
        for model, aggregate in aggregates.items()
    }
    if len({len(model_values) for model_values in values.values()}) != 1:
        raise ValueError("All models must have the same number of paired seeds")
    gains_plain = [
        float(tfdcr) - float(plain)
        for tfdcr, plain in zip(values["tfdcr"], values["plain"], strict=True)
    ]
    gains_repconv = [
        float(tfdcr) - float(repconv)
        for tfdcr, repconv in zip(values["tfdcr"], values["repconv"], strict=True)
    ]
    repconv_wins = sum(gain > 0 for gain in gains_repconv)
    mean_gain_plain = means["tfdcr"] - means["plain"]
    mean_gain_repconv = means["tfdcr"] - means["repconv"]
    checks = {
        "minimum_mean_plain_score_gain": mean_gain_plain >= args.minimum_plain_gain,
        "mean_better_than_repconv": mean_gain_repconv > 0,
        "repconv_pairwise_wins": repconv_wins >= args.minimum_repconv_wins,
        "macro_f1_mean_not_worse_than_plain": (macro_f1_means["tfdcr"] >= macro_f1_means["plain"]),
    }
    passed = all(checks.values())
    return {
        "gate": "three_seed_confirmation",
        "icbhi_score_means": means,
        "macro_f1_means": macro_f1_means,
        "paired_score_gains_over_plain": gains_plain,
        "paired_score_gains_over_repconv": gains_repconv,
        "mean_score_gain_over_plain": mean_gain_plain,
        "mean_score_gain_over_repconv": mean_gain_repconv,
        "repconv_pairwise_wins": repconv_wins,
        "required_repconv_pairwise_wins": args.minimum_repconv_wins,
        "checks": checks,
        "passed": passed,
        "next_action": (
            "freeze_core_method_then_run_ablations_and_second_dataset"
            if passed
            else "downgrade_claim_and_inspect_instability_before_expanding"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.gate == "overfit":
        payload = assess_overfit(args)
    elif args.gate == "method":
        payload = assess_method(args)
    else:
        payload = assess_confirmation(args)
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if getattr(args, "output", None) is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if not payload["passed"]:
        raise SystemExit(f"{args.gate} gate failed")


if __name__ == "__main__":
    main()
