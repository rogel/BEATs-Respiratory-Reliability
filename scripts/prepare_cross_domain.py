#!/usr/bin/env python3
"""Build the frozen unified binary manifest and its leakage audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from respiratory_sound.data.cross_domain import build_cross_domain_manifest, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=20_260_728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_dir = root / "data" / "manifests"
    sources = {
        "icbhi_cycles": manifest_dir / "icbhi2017_cycles.csv",
        "icbhi_recordings": manifest_dir / "icbhi2017_recordings.csv",
        "sprsound_events": manifest_dir / "sprsound2022_events.csv",
        "sprsound_recordings": manifest_dir / "sprsound2022_recordings.csv",
    }
    frames = {
        name: pd.read_csv(path, dtype={"patient_id": str})
        for name, path in sources.items()
    }
    unified, audit = build_cross_domain_manifest(
        icbhi_cycles=frames["icbhi_cycles"],
        icbhi_recordings=frames["icbhi_recordings"],
        sprsound_events=frames["sprsound_events"],
        sprsound_recordings=frames["sprsound_recordings"],
        seed=args.seed,
        project_root=root,
    )
    output_manifest = manifest_dir / "cross_domain_binary.csv"
    output_audit = manifest_dir / "cross_domain_binary_audit.json"
    output_hashes = manifest_dir / "cross_domain_binary_sha256.json"
    unified.to_csv(output_manifest, index=False)
    output_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    hashes = {
        "source_manifests": {
            name: sha256_file(path) for name, path in sources.items()
        },
        "cross_domain_binary": sha256_file(output_manifest),
        "cross_domain_binary_audit": sha256_file(output_audit),
    }
    output_hashes.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    if not audit["data_gate_passed"]:
        raise SystemExit("Cross-domain data gate failed")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
