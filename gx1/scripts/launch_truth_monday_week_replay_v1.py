#!/usr/bin/env python3
"""Launch Monday-week TRUTH replays in bounded parallel batches.

This launcher is intentionally small and boring:
- reads the Monday-week reorg contract,
- skips quarantined weeks,
- skips completed run dirs,
- optionally archives stale partial run dirs,
- runs gx1.scripts.run_truth_e2e_sanity with a bounded worker count.

It does not promote any artifact to live and does not modify policy behavior.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CALENDAR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/TRUTH_CALENDAR_REORG_MONDAY_WEEK_V1.json"
)
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_VENV_PYTHON = Path("/home/andre2/venvs/gx1/bin/python")
DEFAULT_REPO_ROOT = Path("/home/andre2/src/GX1_ENGINE")


@dataclass(frozen=True)
class WeekSpec:
    run_id: str
    start_ts: str
    end_ts: str
    quarantine_status: str
    quarantine_reason: str | None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso_z(value: str) -> str:
    return str(value).replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _load_weeks(calendar_path: Path, *, include_quarantine: bool) -> list[WeekSpec]:
    obj = _read_json(calendar_path)
    weeks_raw = obj.get("full_monday_weeks")
    if not isinstance(weeks_raw, list):
        raise RuntimeError(f"{calendar_path} missing list full_monday_weeks")

    weeks: list[WeekSpec] = []
    for raw in weeks_raw:
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid week entry in {calendar_path}: {raw!r}")
        status = str(raw.get("quarantine_status", ""))
        if status != "ACTIVE_CANDIDATE" and not include_quarantine:
            continue
        run_id = str(raw.get("run_id", "")).strip()
        start_ts = str(raw.get("trading_start_utc") or raw.get("calendar_start_utc") or "").strip()
        end_ts = str(raw.get("calendar_end_exclusive_utc") or "").strip()
        if not run_id or not start_ts or not end_ts:
            raise RuntimeError(f"Invalid week spec: {raw!r}")
        weeks.append(
            WeekSpec(
                run_id=run_id,
                start_ts=_iso_z(start_ts),
                end_ts=_iso_z(end_ts),
                quarantine_status=status,
                quarantine_reason=raw.get("quarantine_reason"),
            )
        )
    return weeks


def _is_completed(run_dir: Path) -> bool:
    completed = run_dir / "RUN_COMPLETED.json"
    postrun = run_dir / "POSTRUN_E2E.json"
    if not completed.exists() or not postrun.exists():
        return False
    try:
        completed_obj = _read_json(completed)
        postrun_obj = _read_json(postrun)
    except Exception:
        return False
    return str(completed_obj.get("status", "")).upper() == "COMPLETED" and bool(postrun_obj.get("passed", False))


def _archive_stale_run_dir(run_dir: Path, archive_root: Path) -> Path:
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / run_dir.name
    if target.exists():
        suffix = 1
        while (archive_root / f"{run_dir.name}.{suffix:03d}").exists():
            suffix += 1
        target = archive_root / f"{run_dir.name}.{suffix:03d}"
    shutil.move(str(run_dir), str(target))
    return target


def _build_command(python_bin: Path, week: WeekSpec) -> list[str]:
    return [
        str(python_bin),
        "-m",
        "gx1.scripts.run_truth_e2e_sanity",
        "--run-id",
        week.run_id,
        "--start-ts",
        week.start_ts,
        "--end-ts",
        week.end_ts,
    ]


def _postrun_status(reports_root: Path, run_id: str, returncode: int | None) -> dict[str, Any]:
    run_dir = reports_root / run_id
    completed = run_dir / "RUN_COMPLETED.json"
    postrun = run_dir / "POSTRUN_E2E.json"
    footer = run_dir / "replay" / "chunk_0" / "chunk_footer.json"
    trade_outcomes = run_dir / f"trade_outcomes_{run_id}_MERGED.parquet"
    out: dict[str, Any] = {
        "run_id": run_id,
        "returncode": returncode,
        "run_dir": str(run_dir),
        "run_completed_exists": completed.exists(),
        "postrun_exists": postrun.exists(),
        "trade_outcomes_exists": trade_outcomes.exists(),
        "status": "UNKNOWN",
        "n_trades_closed": None,
    }
    try:
        if completed.exists():
            out["run_completed_status"] = _read_json(completed).get("status")
        if postrun.exists():
            postrun_obj = _read_json(postrun)
            out["postrun_passed"] = bool(postrun_obj.get("passed", False))
            out["postrun_gates_failed"] = postrun_obj.get("gates_failed")
        if footer.exists():
            footer_obj = _read_json(footer)
            out["chunk_status"] = footer_obj.get("status")
            out["n_trades_closed"] = footer_obj.get("n_trades_closed")
            out["bars_processed"] = footer_obj.get("bars_processed")
            out["wall_clock_sec"] = footer_obj.get("wall_clock_sec")
    except Exception as exc:
        out["artifact_read_error"] = f"{type(exc).__name__}: {exc}"
    completed_artifacts_pass = bool(out.get("run_completed_exists")) and bool(out.get("postrun_passed"))
    if completed_artifacts_pass:
        out["status"] = "COMPLETED"
        if returncode not in (0, None):
            out["returncode_anomaly"] = True
            out["returncode_anomaly_reason"] = "PROCESS_NONZERO_AFTER_COMPLETED_ARTIFACTS"
    elif returncode is None:
        out["status"] = "RUNNING"
    else:
        out["status"] = "FAILED_OR_INCOMPLETE"
    return out


def _write_status(
    status_path: Path,
    *,
    launch_id: str,
    selected: list[WeekSpec],
    skipped_completed: list[str],
    skipped_quarantine: list[dict[str, Any]],
    archived_stale: list[dict[str, str]],
    running: dict[str, subprocess.Popen[Any]],
    completed_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    running_rows = [{"run_id": run_id, "pid": proc.pid} for run_id, proc in running.items()]
    payload = {
        "launch_id": launch_id,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "selected_count": len(selected),
        "skipped_completed_count": len(skipped_completed),
        "skipped_quarantine_count": len(skipped_quarantine),
        "archived_stale_count": len(archived_stale),
        "running_count": len(running_rows),
        "completed_count": len(completed_rows),
        "failed_count": len(failed_rows),
        "selected_run_ids": [w.run_id for w in selected],
        "skipped_completed_run_ids": skipped_completed,
        "skipped_quarantine": skipped_quarantine,
        "archived_stale": archived_stale,
        "running": running_rows,
        "completed": completed_rows,
        "failed": failed_rows,
    }
    _write_json(status_path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Monday-week TRUTH replays in bounded batches.")
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_VENV_PYTHON)
    parser.add_argument("--max-workers", type=int, default=15)
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no cap")
    parser.add_argument("--include-quarantine", action="store_true")
    parser.add_argument("--archive-stale", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=20.0)
    args = parser.parse_args()

    if args.max_workers < 1:
        raise RuntimeError("--max-workers must be >= 1")
    if not args.python_bin.exists():
        raise RuntimeError(f"python bin not found: {args.python_bin}")
    if not args.repo_root.exists():
        raise RuntimeError(f"repo root not found: {args.repo_root}")

    launch_id = f"TRUTH_MONDAY_WEEK_REPLAY_LAUNCH_V1_{_utc_stamp()}"
    launch_root = args.reports_root / launch_id
    logs_root = launch_root / "logs"
    archive_root = launch_root / "stale_archives"
    status_path = launch_root / "status.json"
    launch_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    all_weeks = _load_weeks(args.calendar, include_quarantine=args.include_quarantine)
    calendar_obj = _read_json(args.calendar)
    skipped_quarantine = [
        {
            "run_id": str(w.get("run_id")),
            "reason": w.get("quarantine_reason"),
        }
        for w in calendar_obj.get("full_monday_weeks", [])
        if isinstance(w, dict) and str(w.get("quarantine_status")) != "ACTIVE_CANDIDATE"
    ]

    selected: list[WeekSpec] = []
    skipped_completed: list[str] = []
    archived_stale: list[dict[str, str]] = []
    for week in all_weeks:
        run_dir = args.reports_root / week.run_id
        if _is_completed(run_dir):
            skipped_completed.append(week.run_id)
            continue
        if run_dir.exists():
            if not args.archive_stale:
                raise RuntimeError(f"stale/incomplete run dir exists; use --archive-stale: {run_dir}")
            archived = _archive_stale_run_dir(run_dir, archive_root)
            archived_stale.append({"run_id": week.run_id, "archived_to": str(archived)})
        selected.append(week)

    if args.max_runs > 0:
        selected = selected[: args.max_runs]

    manifest = {
        "launch_id": launch_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "calendar": str(args.calendar),
        "reports_root": str(args.reports_root),
        "repo_root": str(args.repo_root),
        "python_bin": str(args.python_bin),
        "max_workers": args.max_workers,
        "max_runs": args.max_runs,
        "include_quarantine": bool(args.include_quarantine),
        "archive_stale": bool(args.archive_stale),
        "dry_run": bool(args.dry_run),
        "selected_count": len(selected),
        "skipped_completed_count": len(skipped_completed),
        "skipped_quarantine_count": len(skipped_quarantine),
        "archived_stale_count": len(archived_stale),
        "selected": [week.__dict__ for week in selected],
        "skipped_completed": skipped_completed,
        "skipped_quarantine": skipped_quarantine,
        "archived_stale": archived_stale,
        "status_path": str(status_path),
    }
    _write_json(launch_root / "launch_manifest.json", manifest)

    completed_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    running: dict[str, subprocess.Popen[Any]] = {}
    log_handles: dict[str, Any] = {}

    _write_status(
        status_path,
        launch_id=launch_id,
        selected=selected,
        skipped_completed=skipped_completed,
        skipped_quarantine=skipped_quarantine,
        archived_stale=archived_stale,
        running=running,
        completed_rows=completed_rows,
        failed_rows=failed_rows,
        dry_run=args.dry_run,
    )

    print(f"[LAUNCH] launch_root={launch_root}")
    print(f"[LAUNCH] selected={len(selected)} skipped_completed={len(skipped_completed)} skipped_quarantine={len(skipped_quarantine)}")
    if args.dry_run:
        for week in selected:
            print(" ".join(_build_command(args.python_bin, week)))
        return 0

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.repo_root)
    env.setdefault("GX1_DATA", "/home/andre2/GX1_DATA")

    try:
        queue = list(selected)
        while queue or running:
            while queue and len(running) < args.max_workers:
                week = queue.pop(0)
                cmd = _build_command(args.python_bin, week)
                log_path = logs_root / f"{week.run_id}.log"
                handle = log_path.open("w", encoding="utf-8")
                handle.write(f"# cmd: {' '.join(cmd)}\n")
                handle.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(args.repo_root),
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                running[week.run_id] = proc
                log_handles[week.run_id] = handle
                print(f"[LAUNCH] started run_id={week.run_id} pid={proc.pid} log={log_path}", flush=True)

            finished: list[tuple[str, int]] = []
            for run_id, proc in list(running.items()):
                rc = proc.poll()
                if rc is not None:
                    finished.append((run_id, int(rc)))

            for run_id, rc in finished:
                proc = running.pop(run_id)
                handle = log_handles.pop(run_id, None)
                if handle is not None:
                    handle.flush()
                    handle.close()
                row = _postrun_status(args.reports_root, run_id, rc)
                if row.get("status") == "COMPLETED":
                    completed_rows.append(row)
                else:
                    failed_rows.append(row)
                print(
                    f"[LAUNCH] finished run_id={run_id} rc={rc} status={row.get('status')} trades={row.get('n_trades_closed')}",
                    flush=True,
                )
                if args.stop_on_failure and row.get("status") != "COMPLETED":
                    queue.clear()

            _write_status(
                status_path,
                launch_id=launch_id,
                selected=selected,
                skipped_completed=skipped_completed,
                skipped_quarantine=skipped_quarantine,
                archived_stale=archived_stale,
                running=running,
                completed_rows=completed_rows,
                failed_rows=failed_rows,
                dry_run=args.dry_run,
            )

            if queue or running:
                time.sleep(max(1.0, float(args.poll_sec)))
    finally:
        for handle in log_handles.values():
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass

    final = {
        "launch_id": launch_id,
        "completed_count": len(completed_rows),
        "failed_count": len(failed_rows),
        "skipped_completed_count": len(skipped_completed),
        "skipped_quarantine_count": len(skipped_quarantine),
        "status_path": str(status_path),
        "manifest_path": str(launch_root / "launch_manifest.json"),
        "completed": completed_rows,
        "failed": failed_rows,
    }
    _write_json(launch_root / "final_summary.json", final)
    print(f"[LAUNCH] final_summary={launch_root / 'final_summary.json'}")
    return 0 if not failed_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
