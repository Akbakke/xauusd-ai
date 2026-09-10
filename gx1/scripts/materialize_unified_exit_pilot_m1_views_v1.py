#!/usr/bin/env python3
"""Materialize immutable TRAIN/VAL M1 child views for the lifecycle pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


VIEW_SCHEMA_VERSION = "gx1_unified_exit_pilot_m1_child_view_v1"
ROOT_SCHEMA_VERSION = "gx1_unified_exit_pilot_m1_child_view_root_v1"
LOCAL_HISTORY_ROWS = 480
SPLIT_WINDOWS = {
    "train": ("2025-06-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00", 65_295),
    "val": ("2026-06-01T00:00:00+00:00", "2026-07-01T00:00:00+00:00", 5_508),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("PILOT_M1_VIEW_JSON_INVALID")
    return value


def build_views(
    *,
    child_admission_path: Path,
    parent_m1_path: Path,
    parent_m1_manifest_path: Path,
    output_root: Path,
    publish: bool,
) -> dict[str, Any]:
    admission = _json(child_admission_path)
    parent_manifest = _json(parent_m1_manifest_path)
    parent_sha = _sha(parent_m1_path)
    if (
        admission.get("decision") != "PASS"
        or admission.get("test_accessed") is not False
        or parent_manifest.get("output_parquet") != str(parent_m1_path)
        or parent_manifest.get("output_parquet_sha256") != parent_sha
        or parent_manifest.get("quote_complete_m1") is not True
        or parent_manifest.get("test_accessed") is not False
    ):
        raise RuntimeError("PILOT_M1_VIEW_PARENT_INVALID")
    parent = pq.read_table(parent_m1_path)
    times = pd.DatetimeIndex(parent["time"].to_pandas()).as_unit("ns")
    if times.hasnans or not times.is_unique or not times.is_monotonic_increasing:
        raise RuntimeError("PILOT_M1_VIEW_CLOCK_INVALID")
    manifests: dict[str, Any] = {}
    tables: dict[str, pa.Table] = {}
    for split, (start_raw, end_raw, expected_entries) in SPLIT_WINDOWS.items():
        child = admission["splits"][split]
        if child["rows"] != expected_entries or _sha(Path(child["parquet_path"])) != child["parquet_sha256"]:
            raise RuntimeError("PILOT_M1_VIEW_CHILD_INVALID")
        entry_times = pd.DatetimeIndex(
            pq.read_table(child["parquet_path"], columns=["time"])["time"].to_pandas()
        ).as_unit("ns")
        first_state_ns = int(entry_times.asi8[0] + 300_000_000_000)
        first_state_pos = int(np.searchsorted(times.asi8, first_state_ns))
        if first_state_pos >= len(times) or int(times.asi8[first_state_pos]) != first_state_ns or first_state_pos < LOCAL_HISTORY_ROWS - 1:
            raise RuntimeError("PILOT_M1_VIEW_FIRST_STATE_INVALID")
        context_start_pos = first_state_pos - (LOCAL_HISTORY_ROWS - 1)
        end_ns = pd.Timestamp(end_raw).value
        stop = int(np.searchsorted(times.asi8, end_ns))
        if stop <= first_state_pos:
            raise RuntimeError("PILOT_M1_VIEW_END_INVALID")
        table = parent.slice(context_start_pos, stop - context_start_pos)
        tables[split] = table
        view_times = times[context_start_pos:stop]
        fit_start_ns = pd.Timestamp(start_raw).value
        fit_mask = (view_times.asi8 >= fit_start_ns) & (view_times.asi8 < end_ns)
        manifest = {
            "schema_version": VIEW_SCHEMA_VERSION,
            "decision": "PASS",
            "split": split,
            "instrument": "XAU_USD",
            "timeframe": "M1",
            "timestamp_semantics": "bar_start_utc",
            "output_parquet": str(output_root / f"{split}.m1.parquet"),
            "output_parquet_sha256": None,
            "row_count": len(view_times),
            "context_row_count": int(np.count_nonzero(view_times.asi8 < fit_start_ns)),
            "fit_row_count": int(np.count_nonzero(fit_mask)),
            "fit_window_start_utc": start_raw,
            "fit_window_end_utc_exclusive": end_raw,
            "first_entry_time_utc": entry_times[0].isoformat(),
            "first_state_time_utc": pd.Timestamp(first_state_ns, tz="UTC").isoformat(),
            "required_local_history_rows": LOCAL_HISTORY_ROWS,
            "first_state_local_history_rows_available": LOCAL_HISTORY_ROWS,
            "time_min_utc": view_times[0].isoformat(),
            "time_max_utc": view_times[-1].isoformat(),
            "clock_sha256": hashlib.sha256(np.asarray(view_times.asi8, dtype="<i8").tobytes()).hexdigest(),
            "fit_clock_sha256": hashlib.sha256(np.asarray(view_times.asi8[fit_mask], dtype="<i8").tobytes()).hexdigest(),
            "parent_m1_path": str(parent_m1_path),
            "parent_m1_sha256": parent_sha,
            "parent_m1_manifest_path": str(parent_m1_manifest_path),
            "parent_m1_manifest_sha256": _sha(parent_m1_manifest_path),
            "child_admission_path": str(child_admission_path),
            "child_admission_sha256": _sha(child_admission_path),
            "child_parquet_sha256": child["parquet_sha256"],
            "right_censor_time_utc_exclusive": end_raw,
            "context_rows_excluded_from_policy_fit": True,
            "test_accessed": False,
        }
        manifests[split] = manifest
    if not publish:
        return {"mode": "validate_no_publish", "decision": "PASS", "published": False, "manifests": manifests}
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("PILOT_M1_VIEW_OUTPUT_EXISTS")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging.", dir=output_root.parent))
    try:
        for split, table in tables.items():
            parquet_path = staging / f"{split}.m1.parquet"
            pq.write_table(table, parquet_path, compression="zstd", use_dictionary=False, write_statistics=True)
            manifests[split]["output_parquet_sha256"] = _sha(parquet_path)
            manifests[split]["manifest_payload_sha256"] = _canonical(manifests[split])
            (staging / f"{split}.manifest.json").write_text(json.dumps(manifests[split], sort_keys=True, indent=2, allow_nan=False) + "\n")
        root = {
            "schema_version": ROOT_SCHEMA_VERSION,
            "decision": "PASS",
            "splits": {
                split: {
                    "parquet_path": str(output_root / f"{split}.m1.parquet"),
                    "parquet_sha256": manifest["output_parquet_sha256"],
                    "manifest_path": str(output_root / f"{split}.manifest.json"),
                    "manifest_sha256": _sha(staging / f"{split}.manifest.json"),
                    "clock_sha256": manifest["clock_sha256"],
                    "fit_clock_sha256": manifest["fit_clock_sha256"],
                    "rows": manifest["row_count"],
                    "fit_rows": manifest["fit_row_count"],
                    "context_rows": manifest["context_row_count"],
                }
                for split, manifest in manifests.items()
            },
            "parent_m1_sha256": parent_sha,
            "child_admission_sha256": _sha(child_admission_path),
            "test_accessed": False,
        }
        root["root_sha256"] = _canonical(root)
        (staging / "M1_CHILD_VIEW_ROOT.json").write_text(json.dumps(root, sort_keys=True, indent=2, allow_nan=False) + "\n")
        os.rename(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"mode": "publish", "decision": "PASS", "published": True, "output_root": str(output_root), "root": root}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-admission", type=Path, required=True)
    parser.add_argument("--parent-m1", type=Path, required=True)
    parser.add_argument("--parent-m1-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build_views(child_admission_path=args.child_admission.resolve(), parent_m1_path=args.parent_m1.resolve(), parent_m1_manifest_path=args.parent_m1_manifest.resolve(), output_root=args.output_root.resolve(), publish=args.publish), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
