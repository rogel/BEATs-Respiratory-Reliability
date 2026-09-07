"""Create audited ICBHI recording and respiratory-cycle manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from respiratory_sound.data.icbhi import build_icbhi_manifests, save_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/raw/icbhi2017/ICBHI_final_database"),
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=Path("data/raw/icbhi2017/ICBHI_challenge_train_test.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--seed", type=int, default=20_260_727)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    recordings, cycles, report = build_icbhi_manifests(
        audio_dir=(project_root / args.audio_dir).resolve(),
        official_split_file=(project_root / args.split_file).resolve(),
        project_root=project_root,
        seed=args.seed,
    )
    hashes = save_manifests(
        recordings=recordings,
        cycles=cycles,
        report=report,
        output_dir=(project_root / args.output_dir).resolve(),
    )
    print(json.dumps({"audit": report, "sha256": hashes}, indent=2, sort_keys=True))
    if not report["data_gate_passed"]:
        raise SystemExit("ICBHI data gate failed; inspect data/manifests/icbhi2017_audit.json")


if __name__ == "__main__":
    main()
