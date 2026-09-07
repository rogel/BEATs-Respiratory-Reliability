#!/usr/bin/env python3
"""Apply the frozen Gate 9A rules to completed development summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from respiratory_sound.gate9a import (
    resolve_gate_status,
    single_seed_decision,
    three_seed_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--checkpoint-audit",
        type=Path,
        default=Path("artifacts/gate9a_pretrained_checkpoint_audit.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    summaries = [
        json.loads((root / run_dir / "summary.json").read_text(encoding="utf-8"))
        for run_dir in args.run_dir
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for summary in summaries:
        if summary.get("locked_test_accessed") is not False:
            raise ValueError("Gate 9A summary does not certify locked-test isolation")
        grouped[str(summary["candidate"])].append(summary)
    audit = json.loads(
        (root / args.checkpoint_audit).read_text(encoding="utf-8")
    )
    source_ready = {
        "panns_cnn6": audit["panns_cnn6"]["official_tensor_equality"] == "passed",
        "beats": audit["beats"]["strict_architecture_key_check"] == "passed",
    }
    candidate_results = {}
    for candidate, candidate_summaries in grouped.items():
        ordered = sorted(candidate_summaries, key=lambda value: int(value["seed"]))
        single = [single_seed_decision(summary) for summary in ordered]
        result = {
            "checkpoint_provenance_ready": source_ready[candidate],
            "single_seed_decisions": single,
            "single_seed_screen_passed": (
                len(single) >= 1 and single[0]["passed"] and source_ready[candidate]
            ),
            "three_seed_decision": (
                three_seed_decision(ordered)
                if len(ordered) == 3
                else None
            ),
        }
        candidate_results[candidate] = result
    single_passers = [
        candidate
        for candidate, result in candidate_results.items()
        if result["single_seed_screen_passed"]
    ]
    completed_candidates = set(grouped)
    both_screens_complete = {"panns_cnn6", "beats"}.issubset(completed_candidates)
    three_seed_passers = [
        candidate
        for candidate, result in candidate_results.items()
        if result["three_seed_decision"] is not None
        and result["three_seed_decision"]["passed"]
    ]
    payload = {
        "gate": "9A",
        "development_only": True,
        "locked_tests_accessed": False,
        "candidates": candidate_results,
        "single_seed_passers": single_passers,
        "three_seed_passers": three_seed_passers,
        "both_single_seed_screens_complete": both_screens_complete,
        "decision": resolve_gate_status(
            candidate_results,
            both_single_seed_screens_complete=both_screens_complete,
        ),
    }
    output = (root / args.output).resolve()
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
