#!/usr/bin/env python3
"""Materialize the smart520 live-state rank reference for a specific dataset frame.

The live smart520 state builder pins frame-global ctx_cat buckets and source
``atr`` values from the training/evidence frame. A new XAU direction-repair
dataset must therefore carry its own reference artifact instead of reusing the
old July promotion artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "time",
    "atr",
    "atr_bps",
    "spread_bps",
    "vol_regime_id",
    "spread_bucket",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_ts(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _finite_sorted(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return np.sort(arr.astype(np.float64, copy=False))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    return str(obj)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source_parquet).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    model_start = _parse_ts(args.model_range_start)
    reference_end = _parse_ts(args.reference_end)
    if reference_end < model_start:
        raise RuntimeError("reference_end must be >= model_range_start")
    if not source.is_file():
        raise RuntimeError(f"source parquet missing: {source}")

    frame = pd.read_parquet(source, columns=list(REQUIRED_COLUMNS))
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise RuntimeError(f"source parquet lacks required rank-reference columns: {missing}")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if frame["time"].isna().any():
        raise RuntimeError("source parquet contains unparsable time values")
    frame = frame.sort_values("time").reset_index(drop=True)
    ref = frame[(frame["time"] >= model_start) & (frame["time"] <= reference_end)].copy()
    if ref.empty:
        raise RuntimeError(
            f"no reference rows in [{model_start}, {reference_end}] from {source}"
        )
    if int(args.min_rows) > 0 and len(ref) < int(args.min_rows):
        raise RuntimeError(f"rank reference has {len(ref)} rows, below min_rows={args.min_rows}")

    for col in ("atr", "atr_bps", "spread_bps"):
        vals = pd.to_numeric(ref[col], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(vals).all():
            raise RuntimeError(f"rank reference column {col!r} contains non-finite values")
    vol = pd.to_numeric(ref["vol_regime_id"], errors="coerce").to_numpy(dtype=np.int64)
    spread = pd.to_numeric(ref["spread_bucket"], errors="coerce").to_numpy(dtype=np.int64)
    if not np.isin(vol, [0, 1, 2, 3, 4]).all():
        raise RuntimeError("vol_regime_id must be in 0..4")
    if not np.isin(spread, [0, 1, 2, 3, 4]).all():
        raise RuntimeError("spread_bucket must be in 0..4")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        time_ns=pd.DatetimeIndex(ref["time"]).asi8.astype(np.int64),
        vol_regime_id=vol.astype(np.int64),
        spread_bucket=spread.astype(np.int64),
        atr_pinned=pd.to_numeric(ref["atr"], errors="coerce").to_numpy(dtype=np.float64),
        atr_bps_sorted=_finite_sorted(ref["atr_bps"].to_numpy(dtype=np.float64)),
        spread_bps_sorted=_finite_sorted(ref["spread_bps"].to_numpy(dtype=np.float64)),
    )

    sidecar = out.with_suffix(out.suffix + ".json")
    report = {
        "schema_version": "smart520_rank_reference_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_parquet": str(source),
        "source_parquet_sha256": _sha256_file(source),
        "out_npz": str(out),
        "out_npz_sha256": _sha256_file(out),
        "model_range_start_utc": str(model_start),
        "reference_end_utc": str(reference_end),
        "row_count": int(len(ref)),
        "time_min": str(ref["time"].iloc[0]),
        "time_max": str(ref["time"].iloc[-1]),
        "vol_regime_id_counts": {str(k): int(v) for k, v in pd.Series(vol).value_counts().sort_index().items()},
        "spread_bucket_counts": {str(k): int(v) for k, v in pd.Series(spread).value_counts().sort_index().items()},
    }
    sidecar.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-range-start", default="2020-11-09T00:00:00Z")
    parser.add_argument("--reference-end", required=True)
    parser.add_argument("--min-rows", type=int, default=1000)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
