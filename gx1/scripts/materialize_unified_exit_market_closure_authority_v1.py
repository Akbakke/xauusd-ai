#!/usr/bin/env python3
"""Materialize an immutable PRETEST XAU market-closure authority."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from gx1.contracts.unified_exit_market_closure_authority_v1 import (
    build_market_closure_authority,
    file_sha256,
    require_exact_market_schedule,
)


def _read_json(path: Path, label: str) -> dict:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"UNIFIED_EXIT_MARKET_CLOSURE_{label}_PATH_INVALID")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"UNIFIED_EXIT_MARKET_CLOSURE_{label}_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"UNIFIED_EXIT_MARKET_CLOSURE_{label}_INVALID")
    return value


def materialize_market_closure_authority(
    *,
    m1_source_path: Path,
    m1_source_manifest_path: Path,
    exact_schedule_path: Path,
    output_path: Path,
    publish: bool,
) -> dict:
    """Validate inputs and optionally publish the no-replace authority."""

    if type(publish) is not bool:
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_INVOCATION_INVALID")
    source_path = m1_source_path.expanduser().resolve()
    source_manifest_path = m1_source_manifest_path.expanduser().resolve()
    schedule_path = exact_schedule_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    for path in (source_path, source_manifest_path, schedule_path):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_INPUT_PATH_INVALID")
    source_manifest = _read_json(source_manifest_path, "M1_MANIFEST")
    source_sha = file_sha256(source_path)
    source_manifest_sha = file_sha256(source_manifest_path)
    if (
        source_manifest.get("instrument") != "XAU_USD"
        or source_manifest.get("timeframe") != "M1"
        or source_manifest.get("timestamp_semantics") != "bar_start_utc"
        or source_manifest.get("test_accessed") is not False
        or source_manifest.get("output_parquet") != str(source_path)
        or source_manifest.get("output_parquet_sha256") != source_sha
    ):
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_M1_MANIFEST_INVALID")
    schedule = require_exact_market_schedule(
        _read_json(schedule_path, "SCHEDULE")
    )
    try:
        times = pd.read_parquet(source_path, columns=["time"])["time"]
    except (OSError, ValueError) as exc:
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_M1_SOURCE_INVALID") from exc
    authority = build_market_closure_authority(
        m1_times=times,
        m1_source_path=source_path,
        m1_source_sha256=source_sha,
        m1_source_manifest_path=source_manifest_path,
        m1_source_manifest_sha256=source_manifest_sha,
        exact_schedule=schedule,
        exact_schedule_path=schedule_path,
        exact_schedule_file_sha256=file_sha256(schedule_path),
    )
    report = {
        "mode": "publish" if publish else "validate_no_publish",
        "decision": "PASS",
        "published": publish,
        "output_path": str(output),
        "artifact_sha256": authority["artifact_sha256"],
        "observed_gap_count": authority["observed_gap_count"],
        "known_market_closure_count": authority["known_market_closure_count"],
        "unknown_source_gap_count": authority["unknown_source_gap_count"],
        "test_data_used": False,
    }
    if not publish:
        return {**report, "authority": authority}
    if output.exists() or output.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_MARKET_CLOSURE_OUTPUT_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, staging_name = tempfile.mkstemp(
        prefix=f".{output.name}.staging.", dir=output.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(
                json.dumps(authority, indent=2, sort_keys=True, allow_nan=False).encode()
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(staging, output)
    except Exception:
        staging.unlink(missing_ok=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-source-parquet", type=Path, required=True)
    parser.add_argument("--m1-source-manifest", type=Path, required=True)
    parser.add_argument("--exact-xau-market-schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    result = materialize_market_closure_authority(
        m1_source_path=args.m1_source_parquet,
        m1_source_manifest_path=args.m1_source_manifest,
        exact_schedule_path=args.exact_xau_market_schedule,
        output_path=args.output,
        publish=args.publish,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("materialize_market_closure_authority",)
