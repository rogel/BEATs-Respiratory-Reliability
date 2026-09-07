#!/usr/bin/env python3
"""Create audited SPRSound 2022 recording and event manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from respiratory_sound.data.sprsound import build_sprsound_manifests, save_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/sprsound2022/official_repo/BioCAS2022"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--seed", type=int, default=20_260_727)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    recordings, events, report = build_sprsound_manifests(
        dataset_root=(project_root / args.dataset_root).resolve(),
        project_root=project_root,
        seed=args.seed,
    )
    hashes = save_manifests(
        recordings=recordings,
        events=events,
        report=report,
        output_dir=(project_root / args.output_dir).resolve(),
    )
    print(json.dumps({"audit": report, "sha256": hashes}, indent=2, sort_keys=True))
    if not report["data_gate_passed"]:
        raise SystemExit(
            "SPRSound data gate failed; inspect data/manifests/sprsound2022_audit.json"
        )


if __name__ == "__main__":
    main()
