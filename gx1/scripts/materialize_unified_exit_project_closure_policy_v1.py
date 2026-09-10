#!/usr/bin/env python3
"""Fit a TRAIN-clock closure policy and apply it unchanged to TRAIN or VAL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    build_market_closure_authority,
    build_project_inferred_closure_policy,
    exact_schedule_from_project_policy,
    file_sha256,
    require_project_inferred_closure_policy,
)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"UNIFIED_EXIT_PROJECT_CLOSURE_{label}_PATH_INVALID")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"UNIFIED_EXIT_PROJECT_CLOSURE_{label}_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"UNIFIED_EXIT_PROJECT_CLOSURE_{label}_INVALID")
    return value


def _load_clock(
    source_path: Path,
    manifest_path: Path,
    *,
    split: str,
    fit_only: bool,
) -> tuple[pd.Series, str, str]:
    source = source_path.expanduser().resolve()
    manifest_file = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_file, "M1_MANIFEST")
    source_sha = file_sha256(source)
    manifest_sha = file_sha256(manifest_file)
    if (
        split not in {"train", "val"}
        or source.is_symlink()
        or not source.is_file()
        or manifest.get("schema_version")
        != "gx1_unified_exit_pilot_m1_child_view_v1"
        or manifest.get("instrument") != "XAU_USD"
        or manifest.get("timeframe") != "M1"
        or manifest.get("timestamp_semantics") != "bar_start_utc"
        or manifest.get("split") != split
        or manifest.get("test_accessed") is not False
        or manifest.get("output_parquet") != str(source)
        or manifest.get("output_parquet_sha256") != source_sha
        or manifest.get("context_rows_excluded_from_policy_fit") is not True
        or manifest.get("required_local_history_rows") != 480
    ):
        raise RuntimeError("UNIFIED_EXIT_PROJECT_CLOSURE_M1_MANIFEST_INVALID")
    try:
        times = pd.read_parquet(source, columns=["time"])["time"]
    except (OSError, ValueError) as exc:
        raise RuntimeError("UNIFIED_EXIT_PROJECT_CLOSURE_M1_SOURCE_INVALID") from exc
    observed_clock_sha = hashlib.sha256(
        np.asarray(pd.DatetimeIndex(times).asi8, dtype="<i8").tobytes()
    ).hexdigest()
    if len(times) != manifest.get("row_count") or observed_clock_sha != manifest.get(
        "clock_sha256"
    ):
        raise RuntimeError("UNIFIED_EXIT_PROJECT_CLOSURE_M1_CLOCK_INVALID")
    if not fit_only:
        return times, source_sha, manifest_sha
    start = pd.Timestamp(manifest.get("fit_window_start_utc"))
    end = pd.Timestamp(manifest.get("fit_window_end_utc_exclusive"))
    clock = pd.DatetimeIndex(times).as_unit("ns")
    mask = (clock >= start) & (clock < end)
    fit_times = pd.Series(clock[mask])
    fit_clock_sha = hashlib.sha256(
        np.asarray(clock.asi8[mask], dtype="<i8").tobytes()
    ).hexdigest()
    if (
        start.tz is None
        or end.tz is None
        or end <= start
        or len(fit_times) != manifest.get("fit_row_count")
        or int(np.count_nonzero(clock < start)) != manifest.get("context_row_count")
        or fit_clock_sha != manifest.get("fit_clock_sha256")
    ):
        raise RuntimeError("UNIFIED_EXIT_PROJECT_CLOSURE_FIT_WINDOW_INVALID")
    return fit_times, source_sha, manifest_sha


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def _publish_file(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_PROJECT_CLOSURE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, staging_name = tempfile.mkstemp(prefix=f".{path.name}.staging.", dir=path.parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(staging, path)
    except Exception:
        staging.unlink(missing_ok=True)
        raise


def fit_project_closure_policy(
    *,
    train_m1_source_path: Path,
    train_m1_manifest_path: Path,
    minimum_daily_support: int,
    minimum_weekend_support: int,
    output_path: Path,
    publish: bool,
) -> dict[str, Any]:
    times, source_sha, _manifest_sha = _load_clock(
        train_m1_source_path,
        train_m1_manifest_path,
        split="train",
        fit_only=True,
    )
    policy = build_project_inferred_closure_policy(
        train_m1_times=times,
        train_m1_source_sha256=source_sha,
        minimum_daily_support=minimum_daily_support,
        minimum_weekend_support=minimum_weekend_support,
    )
    if policy["decision"] != "PASS":
        raise RuntimeError("UNIFIED_EXIT_PROJECT_CLOSURE_NO_QUALIFIED_RULES")
    output = output_path.expanduser().resolve()
    if publish:
        _publish_file(output, _json_bytes(policy))
    return {
        "mode": "publish" if publish else "validate_no_publish",
        "decision": "PASS",
        "published": publish,
        "output_path": str(output),
        "policy_sha256": policy["policy_sha256"],
        "qualified_rule_count": len(policy["rules"]),
        "policy": policy if not publish else None,
        "test_data_used": False,
    }


def apply_project_closure_policy(
    *,
    policy_path: Path,
    target_split: str,
    target_m1_source_path: Path,
    target_m1_manifest_path: Path,
    output_dir: Path,
    publish: bool,
) -> dict[str, Any]:
    policy_file = policy_path.expanduser().resolve()
    raw_policy = _read_json(policy_file, "POLICY")
    policy = require_project_inferred_closure_policy(
        raw_policy,
        expected_train_m1_source_sha256=raw_policy.get("train_m1_source_sha256"),
    )
    times, source_sha, manifest_sha = _load_clock(
        target_m1_source_path,
        target_m1_manifest_path,
        split=target_split,
        fit_only=False,
    )
    schedule = exact_schedule_from_project_policy(
        policy=policy,
        expected_train_m1_source_sha256=policy["train_m1_source_sha256"],
        target_split=target_split,
        target_m1_times=times,
    )
    output = output_dir.expanduser().resolve()
    schedule_path = output / "exact_schedule.json"
    authority_path = output / "market_closure_authority.json"
    schedule_bytes = _json_bytes(schedule)
    schedule_file_sha = hashlib.sha256(schedule_bytes).hexdigest()
    authority = build_market_closure_authority(
        m1_times=times,
        m1_source_path=target_m1_source_path.expanduser().resolve(),
        m1_source_sha256=source_sha,
        m1_source_manifest_path=target_m1_manifest_path.expanduser().resolve(),
        m1_source_manifest_sha256=manifest_sha,
        exact_schedule=schedule,
        exact_schedule_path=schedule_path,
        exact_schedule_file_sha256=schedule_file_sha,
    )
    authority_bytes = _json_bytes(authority)
    if publish:
        if output.exists() or output.is_symlink():
            raise RuntimeError("UNIFIED_EXIT_PROJECT_CLOSURE_OUTPUT_EXISTS")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
        try:
            (staging / schedule_path.name).write_bytes(schedule_bytes)
            (staging / authority_path.name).write_bytes(authority_bytes)
            os.rename(staging, output)
        except Exception:
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
            raise
    return {
        "mode": "publish" if publish else "validate_no_publish",
        "decision": "PASS",
        "published": publish,
        "output_dir": str(output),
        "target_split": target_split,
        "project_policy_sha256": policy["policy_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "authority_sha256": authority["artifact_sha256"],
        "known_market_closure_count": authority["known_market_closure_count"],
        "unknown_source_gap_count": authority["unknown_source_gap_count"],
        "authority": authority if not publish else None,
        "test_data_used": False,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("support count must be at least two")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--train-m1-source", type=Path, required=True)
    fit.add_argument("--train-m1-manifest", type=Path, required=True)
    fit.add_argument("--minimum-daily-support", type=_positive_int, required=True)
    fit.add_argument("--minimum-weekend-support", type=_positive_int, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--publish", action="store_true")
    apply = subparsers.add_parser("apply")
    apply.add_argument("--policy", type=Path, required=True)
    apply.add_argument("--target-split", choices=("train", "val"), required=True)
    apply.add_argument("--target-m1-source", type=Path, required=True)
    apply.add_argument("--target-m1-manifest", type=Path, required=True)
    apply.add_argument("--output-dir", type=Path, required=True)
    apply.add_argument("--publish", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fit":
        result = fit_project_closure_policy(
            train_m1_source_path=args.train_m1_source,
            train_m1_manifest_path=args.train_m1_manifest,
            minimum_daily_support=args.minimum_daily_support,
            minimum_weekend_support=args.minimum_weekend_support,
            output_path=args.output,
            publish=args.publish,
        )
    else:
        result = apply_project_closure_policy(
            policy_path=args.policy,
            target_split=args.target_split,
            target_m1_source_path=args.target_m1_source,
            target_m1_manifest_path=args.target_m1_manifest,
            output_dir=args.output_dir,
            publish=args.publish,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "apply_project_closure_policy",
    "fit_project_closure_policy",
)
