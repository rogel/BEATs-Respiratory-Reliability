#!/usr/bin/env python3
"""Audit exact and official-legacy BEATs token masks without model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from respiratory_sound.gate11a import sha256_file
from respiratory_sound.models.pretrained_audio import (
    _beats_legacy_token_valid_mask,
    _beats_token_support_fraction,
    _beats_token_valid_mask,
)

PLAN_SHA256 = "18b6e39bba441e5be1ec870591b9dbf2846b7e6ab6293be9d211aab12809be39"
MANIFEST_SHA256 = "2dd168129fc0fc159db201f2990c4d7074cb746fc9bc2c13bd5031e47dbd1419"
TARGET_SAMPLES = 128_000
TARGET_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _resampled_length(row: pd.Series) -> int:
    source_rate = int(row["source_sample_rate"])
    start = max(0, math.floor(float(row["start_seconds"]) * source_rate))
    end = math.ceil(float(row["end_seconds"]) * source_rate)
    source_length = max(1, end - start)
    if source_rate == TARGET_RATE:
        return min(TARGET_SAMPLES, source_length)
    return min(TARGET_SAMPLES, math.ceil(source_length * TARGET_RATE / source_rate))


def _stable_offset(sample_id: str, maximum_start: int) -> int:
    if maximum_start <= 0:
        return 0
    value = int.from_bytes(
        hashlib.sha256(sample_id.encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    return value % (maximum_start + 1)


def _placement_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        values = pd.Series(row._asdict())
        length = _resampled_length(values)
        maximum = TARGET_SAMPLES - length
        centre = maximum // 2
        placements: list[tuple[str, int]] = [("centre", centre)]
        seen = {centre}
        for label, offset in (
            ("left", 0),
            ("right", maximum),
            ("stable_random", _stable_offset(str(values["sample_id"]), maximum)),
        ):
            if offset not in seen:
                placements.append((label, offset))
                seen.add(offset)
        if maximum <= 512:
            for offset in range(maximum + 1):
                if offset not in seen:
                    placements.append((f"all_{offset}", offset))
                    seen.add(offset)
        for label, start in placements:
            records.append(
                {
                    "sample_id": str(values["sample_id"]),
                    "dataset": str(values["dataset"]),
                    "patient_id": str(values["patient_id"]),
                    "protocol_role": str(values["protocol_role"]),
                    "event_samples_16khz": length,
                    "placement": label,
                    "start_sample": int(start),
                }
            )
    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = root / "data/manifests/cross_domain_binary.csv"
    plan_path = (
        root
        / "../experiment_plans/2026-09-03/"
        "01_REVISION_ANALYSIS_PLAN_FROZEN_2026-09-03.md"
    ).resolve()
    if sha256_file(plan_path) != PLAN_SHA256:
        raise ValueError("Frozen analysis plan changed")
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("Frozen manifest changed")
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    rows = _placement_rows(manifest)
    output_records: list[dict[str, object]] = []
    batch_size = 256
    for first in range(0, len(rows), batch_size):
        batch = rows.iloc[first : first + batch_size]
        masks = torch.zeros(len(batch), TARGET_SAMPLES, dtype=torch.bool)
        for index, (_, row) in enumerate(batch.iterrows()):
            start = int(row["start_sample"])
            length = int(row["event_samples_16khz"])
            masks[index, start : start + length] = True
        exact = _beats_token_valid_mask(
            masks,
            fbank_frames=798,
            patch_size=16,
            frequency_patches=8,
        )
        fraction = _beats_token_support_fraction(
            masks,
            fbank_frames=798,
            patch_size=16,
            frequency_patches=8,
        )
        if not torch.equal(exact, fraction > 0):
            raise RuntimeError("q_j and exact binary mask disagree")
        legacy = _beats_legacy_token_valid_mask(
            masks,
            fbank_frames=798,
            tokens=392,
        )
        exact_common = exact[:, :392]
        excluded = (exact_common & ~legacy).sum(dim=1) + exact[:, 392:].sum(dim=1)
        included_padding = (legacy & ~exact_common).sum(dim=1)
        for index, (_, row) in enumerate(batch.iterrows()):
            exact_valid = int(exact[index].sum())
            exact_padding = int((~exact[index]).sum())
            record = row.to_dict()
            record.update(
                {
                    "exact_valid_tokens": exact_valid,
                    "legacy_valid_tokens": int(legacy[index].sum()),
                    "fully_padding_tokens": int((fraction[index] == 0).sum()),
                    "partially_valid_tokens": int(
                        ((fraction[index] > 0) & (fraction[index] < 1)).sum()
                    ),
                    "fully_valid_tokens": int((fraction[index] == 1).sum()),
                    "exact_valid_excluded_by_legacy": int(excluded[index]),
                    "padding_only_included_by_legacy": int(included_padding[index]),
                    "excluded_fraction_of_exact_valid": (
                        float(excluded[index]) / exact_valid if exact_valid else np.nan
                    ),
                    "included_fraction_of_exact_padding": (
                        float(included_padding[index]) / exact_padding
                        if exact_padding
                        else 0.0
                    ),
                    "legacy_zero_valid": int(not bool(legacy[index].any())),
                }
            )
            output_records.append(record)

    detailed = pd.DataFrame.from_records(output_records)
    numeric = detailed.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.drop(columns=["excluded_fraction_of_exact_valid"]).to_numpy()).all():
        raise FloatingPointError("Mapping audit produced non-finite values")
    replay = torch.zeros(2, TARGET_SAMPLES, dtype=torch.bool)
    replay[0, 114_458:116_474] = True
    replay[1, 125_554:127_570] = True
    replay_exact = _beats_token_valid_mask(
        replay, fbank_frames=798, patch_size=16, frequency_patches=8
    ).sum(dim=1).tolist()
    replay_legacy = _beats_legacy_token_valid_mask(
        replay, fbank_frames=798, tokens=392
    ).sum(dim=1).tolist()
    if replay_exact != [16, 16] or replay_legacy != [7, 0]:
        raise RuntimeError("Documented failure replay changed")

    aggregate = {
        "schema_version": 1,
        "stage": "BEATS_revision",
        "analysis": "exact_versus_official_legacy_token_mapping",
        "analysis_plan_sha256": PLAN_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "events": int(manifest.shape[0]),
        "event_placement_rows": int(detailed.shape[0]),
        "unique_resampled_lengths": int(detailed["event_samples_16khz"].nunique()),
        "documented_failure_replay": {
            "exact_valid_tokens": replay_exact,
            "legacy_valid_tokens": replay_legacy,
        },
        "centre_placement": {},
        "stress_placements": {},
        "q_j_definition_audit_passed": True,
        "finite_audit_passed": True,
    }
    for label, frame in (
        ("centre_placement", detailed.loc[detailed["placement"] == "centre"]),
        ("stress_placements", detailed),
    ):
        aggregate[label] = {
            "rows": int(len(frame)),
            "events_with_excluded_exact_valid_tokens": int(
                frame.loc[frame["exact_valid_excluded_by_legacy"] > 0, "sample_id"].nunique()
            ),
            "events_with_included_padding_only_tokens": int(
                frame.loc[frame["padding_only_included_by_legacy"] > 0, "sample_id"].nunique()
            ),
            "zero_valid_legacy_rows": int(frame["legacy_zero_valid"].sum()),
            "events_with_any_zero_valid_legacy_placement": int(
                frame.loc[frame["legacy_zero_valid"] > 0, "sample_id"].nunique()
            ),
            "exact_valid_excluded_total": int(
                frame["exact_valid_excluded_by_legacy"].sum()
            ),
            "padding_only_included_total": int(
                frame["padding_only_included_by_legacy"].sum()
            ),
            "fully_padding_tokens_total": int(frame["fully_padding_tokens"].sum()),
            "partially_valid_tokens_total": int(frame["partially_valid_tokens"].sum()),
            "fully_valid_tokens_total": int(frame["fully_valid_tokens"].sum()),
        }

    output_dir = root / "artifacts/revision_2026_09_03/mask_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "exact_legacy_mapping_event_placements.csv"
    result_path = output_dir / "exact_legacy_mapping_audit.json"
    detailed.to_csv(details_path, index=False)
    aggregate["details_sha256"] = sha256_file(details_path)
    result_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
