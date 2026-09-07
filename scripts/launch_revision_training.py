#!/usr/bin/env python3
"""Launch one frozen seed with durable logs and a process-exit record.

No training logic is changed; no retry, seed substitution or automatic model
selection is performed here. The assistant audits/reports each completed seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from respiratory_sound.gate11a import sha256_file


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("no-js", "matched-lora"), required=True)
    parser.add_argument("--seed", choices=(20260729, 20260730, 20260731), type=int, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    base = root / "artifacts/revision_2026_09_03"
    freeze = json.loads((base / "implementation_freeze.json").read_text())
    for relative, digest in freeze["files"].items():
        assert sha256_file(root / relative) == digest, relative
    corrected = base / "analysis_a_corrected_2026_09_05/analysis_a_final.json"
    assert sha256_file(corrected) == "6d9cadf5a92553f9d306bdcc037be8638afe9c084d9bbaf150fe2e55cb72d32d"
    name = f"{args.family.replace('-', '_')}_seed{args.seed}"
    run = root / "runs/revision_2026_09_03" / f"revision_{name}"
    if run.exists():
        raise FileExistsError(f"Existing run must be inspected, never silently restarted: {run}")
    job_dir = base / "jobs_2026_09_05" / name
    if args.check_only:
        assert not job_dir.exists(), job_dir
        print(json.dumps({"status": "preflight_passed", "family": args.family, "seed": args.seed,
                          "training_inputs_hash_verified": len(freeze["files"]),
                          "calibration_correction_verified": True}))
        return
    if not args.worker:
        job_dir.mkdir(parents=True, exist_ok=False)
        with (job_dir / "supervisor.log").open("xb") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--family", args.family,
                 "--seed", str(args.seed), "--worker"], cwd=root,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps({"status": "supervisor_dispatched", "supervisor_pid": process.pid,
                          "job_dir": str(job_dir)}, indent=2))
        return
    status_path = job_dir / "status.json"
    state = {"family": args.family, "seed": args.seed, "started_at": now(),
             "supervisor_pid": os.getpid(), "launcher_sha256": sha256_file(Path(__file__).resolve()),
             "run_dir": str(run), "status": "starting"}
    try:
        with (base / "revision_training.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            command = ["/usr/bin/caffeinate", "-i", sys.executable, "-u",
                       str(root / "scripts/train_revision_ablation.py"),
                       "--family", args.family, "--seed", str(args.seed), "--device", "mps"]
            with (job_dir / "training.log").open("xb") as log:
                process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                state.update(status="running", training_process_pid=process.pid, command=command)
                save(status_path, state)
                code = process.wait()
            state.update(status="completed_pending_audit" if code == 0 else "failed_needs_review",
                         exit_code=code, ended_at=now(), training_log_sha256=sha256_file(job_dir / "training.log"))
            if (run / "summary.json").is_file():
                state["summary_sha256"] = sha256_file(run / "summary.json")
            save(status_path, state)
    except Exception as exc:
        state.update(status="supervisor_failed_needs_review", ended_at=now(),
                     error_type=type(exc).__name__, error=str(exc))
        save(status_path, state)
        raise


if __name__ == "__main__":
    main()
