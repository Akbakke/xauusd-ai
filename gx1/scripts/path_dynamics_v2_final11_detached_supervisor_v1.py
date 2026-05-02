#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

V2_FIELDS = [
    "last_peak_ts_utc",
    "last_mfe_ts_utc",
    "last_peak_mfe_bps",
    "max_mfe_without_mae_bps",
    "mfe_mae_sequence_order",
]
RID_RE = re.compile(r"--run-id\s+(E2E_SANITY_ORDERFIX_\d{8}_\d{8})")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_dates(run_id: str) -> tuple[str, str]:
    start_raw, end_raw = run_id.rsplit("_", 2)[-2:]
    return (
        f"{start_raw[:4]}-{start_raw[4:6]}-{start_raw[6:]}",
        f"{end_raw[:4]}-{end_raw[4:6]}-{end_raw[6:]}",
    )


def _read_run_ids(root: Path) -> list[str]:
    manifest = root / "path_dynamics_v2_full_replay_launcher_manifest_v1.json"
    return list(json.loads(manifest.read_text(encoding="utf-8"))["run_ids"])


def _trade_rows(root: Path, run_id: str) -> int | None:
    path = root / "runs" / run_id / f"trade_outcomes_{run_id}_MERGED.parquet"
    if not path.exists():
        return None
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def _expected_trade_rows(expected_runs_root: Path | None, run_id: str) -> int | None:
    if expected_runs_root is None:
        return None
    path = expected_runs_root / run_id / f"trade_outcomes_{run_id}_MERGED.parquet"
    if not path.exists():
        return None
    try:
        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def _trace_ok(root: Path, run_id: str) -> tuple[bool, int]:
    paths = sorted((root / "runs" / run_id / "replay").glob("**/EXIT_EVAL_TRACE.csv"))
    if not paths:
        return False, 0
    ok = True
    for path in paths:
        header = path.open("r", encoding="utf-8", errors="replace").readline().strip().split(",")
        ok = ok and all(field in header for field in V2_FIELDS)
    return ok, len(paths)


def _classify(root: Path, run_id: str, expected_runs_root: Path | None = None) -> dict[str, Any]:
    run_dir = root / "runs" / run_id
    completed = (run_dir / "RUN_COMPLETED.json").exists()
    rows = _trade_rows(root, run_id)
    expected_rows = _expected_trade_rows(expected_runs_root, run_id)
    trace_ok, trace_count = _trace_ok(root, run_id)
    if completed and expected_rows is not None and rows != expected_rows:
        status = "needs_clean_restart"
        reason = f"COUNT_MISMATCH_EXPECTED_{expected_rows}_OBSERVED_{rows}"
    elif completed and rows == 0:
        status = "accepted_zero_trade_no_trace_expected"
        reason = "ZERO_TRADE_NO_TRACE_EXPECTED"
    elif completed and rows and rows > 0 and trace_ok:
        status = "accepted_nonzero_v2_trace"
        reason = "NONZERO_COMPLETED_WITH_V2_TRACE"
    elif completed and rows and rows > 0:
        status = "needs_clean_restart"
        reason = "NONZERO_COMPLETED_MISSING_V2_TRACE"
    elif completed:
        status = "needs_clean_restart"
        reason = "RUN_COMPLETED_BUT_OUTCOME_NOT_ACCEPTED"
    else:
        status = "pending"
        reason = "NOT_COMPLETED"
    return {
        "run_id": run_id,
        "status": status,
        "reason": reason,
        "run_completed": bool(completed),
        "trade_rows": rows,
        "expected_trade_rows": expected_rows,
        "trace_count": int(trace_count),
        "trace_v2": bool(trace_ok),
    }


def _discover_running(root: Path) -> dict[str, dict[str, Any]]:
    pattern = f"run_truth_e2e_sanity.*{root}"
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    running: dict[str, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        if "pgrep -af" in line:
            continue
        match = RID_RE.search(line)
        if not match:
            continue
        pid = int(line.split(maxsplit=1)[0])
        run_id = match.group(1)
        running[run_id] = {
            "pid": pid,
            "source": "DISCOVERED_EXISTING_PROCESS",
            "run_dir": str(root / "runs" / run_id),
            "log_path": "",
            "started_at_utc": "UNKNOWN_PREEXISTING",
        }
    return running


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _archive_run(root: Path, run_id: str) -> str | None:
    run_dir = root / "runs" / run_id
    if not run_dir.exists():
        return None
    archive_dir = root / "final11_detached_clean_restart_archive_v1"
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / f"{run_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    shutil.move(str(run_dir), str(target))
    return str(target)


def _launch(root: Path, run_id: str, logs_dir: Path, python_exe: str) -> dict[str, Any]:
    start_ts, end_ts = _parse_dates(run_id)
    run_dir = root / "runs" / run_id
    log_path = logs_dir / f"{run_id}.log"
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "GX1_DATA": "/home/andre2/GX1_DATA",
            "GX1_CANONICAL_TRUTH_FILE": "/home/andre2/src/GX1_ENGINE/gx1/configs/canonical_truth_signal_only.json",
            "GX1_STRICT_MASK": "1",
            "GX1_REPLAY_MODEL_PATH_DETERMINISTIC_BLOCK": "0",
            "GX1_REPLAY_THRESHOLD_REJECT_HOLD_FASTPATH": "1",
            "GX1_REPLAY_EXIT_IO_FULL_RECORDS": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TORCH_NUM_THREADS": "1",
        }
    )
    command = [
        python_exe,
        "-m",
        "gx1.scripts.run_truth_e2e_sanity",
        "--run-id",
        run_id,
        "--run-dir",
        str(run_dir),
        "--start-ts",
        start_ts,
        "--end-ts",
        end_ts,
    ]
    handle = log_path.open("wb")
    process = subprocess.Popen(
        command,
        cwd="/home/andre2/src/GX1_ENGINE",
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return {
        "pid": process.pid,
        "process": process,
        "handle": handle,
        "source": "DETACHED_SUPERVISOR_LAUNCHED",
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "started_at_utc": _now(),
    }


def _status_payload(
    root: Path,
    run_ids: list[str],
    running: dict[str, dict[str, Any]],
    attempts: dict[str, int],
    expected_runs_root: Path | None = None,
) -> dict[str, Any]:
    scan = [_classify(root, run_id, expected_runs_root) for run_id in run_ids]
    return {
        "layer_name": "PATH_DYNAMICS_LOGGING_V2_FINAL11_DETACHED_SUPERVISOR_STATUS_V1",
        "updated_at_utc": _now(),
        "root": str(root),
        "accepted_count": sum(row["status"].startswith("accepted_") for row in scan),
        "zero_trade_count": sum(row["status"] == "accepted_zero_trade_no_trace_expected" for row in scan),
        "nonzero_v2_count": sum(row["status"] == "accepted_nonzero_v2_trace" for row in scan),
        "pending_or_bad": [row for row in scan if not row["status"].startswith("accepted_")],
        "running": {
            run_id: {key: value for key, value in meta.items() if key not in {"process", "handle"}}
            for run_id, meta in running.items()
        },
        "attempts": dict(attempts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/home/andre2/GX1_DATA/reports/truth_e2e_sanity/PATH_DYNAMICS_LOGGING_V2_REPLAY_20260422_1227",
    )
    parser.add_argument("--max-parallel", type=int, default=11)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--poll-sec", type=int, default=20)
    parser.add_argument("--python-exe", default="/home/andre2/venvs/gx1/bin/python")
    parser.add_argument(
        "--expected-runs-root",
        default="/home/andre2/GX1_DATA/reports/truth_e2e_sanity/MANAGEMENT_PATH_DYNAMICS_UPSTREAM_REPLAY_V2_20260419_142449/runs",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    expected_runs_root = Path(args.expected_runs_root).expanduser().resolve() if args.expected_runs_root else None
    status_path = root / "path_dynamics_v2_final11_detached_supervisor_status_v1.json"
    logs_dir = root / "logs_final11_detached_supervisor_v1"
    run_ids = _read_run_ids(root)
    attempts: dict[str, int] = {}
    running = _discover_running(root)
    launched_or_restarted: list[dict[str, Any]] = []

    while True:
        # Retire discovered processes that have exited, then classify their outputs.
        for run_id, meta in list(running.items()):
            process = meta.get("process")
            if process is not None:
                return_code = process.poll()
                if return_code is None:
                    continue
                meta["handle"].close()
                running.pop(run_id, None)
                continue
            if not _pid_alive(int(meta["pid"])):
                running.pop(run_id, None)

        scan = [_classify(root, run_id, expected_runs_root) for run_id in run_ids]
        remaining = [row for row in scan if not row["status"].startswith("accepted_")]
        if not remaining:
            payload = _status_payload(root, run_ids, running, attempts, expected_runs_root)
            payload["finished_at_utc"] = _now()
            payload["final_status"] = "ALL_ACCEPTED"
            payload["launched_or_restarted"] = launched_or_restarted
            status_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"event": "detached_done", "accepted": payload["accepted_count"]}, ensure_ascii=True), flush=True)
            return

        for row in remaining:
            run_id = row["run_id"]
            if run_id in running or len(running) >= int(args.max_parallel):
                continue
            attempts[run_id] = attempts.get(run_id, 0) + 1
            if attempts[run_id] > int(args.max_attempts):
                continue
            archive_path = _archive_run(root, run_id)
            meta = _launch(root, run_id, logs_dir, args.python_exe)
            running[run_id] = meta
            launched_or_restarted.append(
                {"run_id": run_id, "attempt": attempts[run_id], "archive_path": archive_path, "started_at_utc": meta["started_at_utc"]}
            )

        payload = _status_payload(root, run_ids, running, attempts, expected_runs_root)
        payload["final_status"] = "RUNNING"
        payload["launched_or_restarted"] = launched_or_restarted
        status_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "event": "detached_progress",
                    "accepted": payload["accepted_count"],
                    "running": len(running),
                    "remaining": len(payload["pending_or_bad"]),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        time.sleep(max(5, int(args.poll_sec)))


if __name__ == "__main__":
    main()
