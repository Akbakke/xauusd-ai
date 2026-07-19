#!/usr/bin/env python3
"""Materialize one immutable TRAIN-only model-native rank reference.

Only the declared TRAIN interval contributes values to the two frozen ECDF
distributions.  A preceding common-history interval is read solely so the
causal ATR at TRAIN_START has the same warmup as dataset build and serving.
No timestamps, per-row buckets, validation rows, test rows, or pinned ATR state
are written to the artifact.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
    compute_causal_market_rank_inputs,
    parse_utc,
    sha256_file,
)
from gx1_guards.gates import require_retrain_vedtak


REQUIRED_COLUMNS = ("time", "high", "low", "close", "bid_close", "ask_close")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return str(value)


def run(args: argparse.Namespace) -> dict[str, Any]:
    explicit_vedtak_id = require_retrain_vedtak(getattr(args, "vedtak", None))
    source = Path(args.source_parquet).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    history_start = parse_utc(args.history_start, field="history_start")
    fit_start = parse_utc(args.fit_start, field="fit_start")
    fit_end = parse_utc(args.fit_end, field="fit_end")
    if not history_start <= fit_start <= fit_end:
        raise RuntimeError(
            "MODEL_NATIVE_TRAIN_RANK_WINDOW_INVALID: expected history_start <= fit_start <= fit_end"
        )
    if not source.is_file():
        raise RuntimeError(f"MODEL_NATIVE_TRAIN_RANK_SOURCE_MISSING: {source}")
    if out.suffix != ".npz":
        raise RuntimeError(f"MODEL_NATIVE_TRAIN_RANK_OUTPUT_SUFFIX_INVALID: {out}")
    sidecar = out.with_suffix(out.suffix + ".json")
    if out.exists() or sidecar.exists():
        raise RuntimeError(f"MODEL_NATIVE_TRAIN_RANK_OUTPUT_NOT_FRESH: {out}")

    frame = pd.read_parquet(source, columns=list(REQUIRED_COLUMNS))
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise RuntimeError(f"MODEL_NATIVE_TRAIN_RANK_SOURCE_FIELDS_MISSING: {missing}")
    try:
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    except Exception as exc:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_SOURCE_TIME_INVALID") from exc
    if frame["time"].isna().any() or frame["time"].duplicated().any():
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_SOURCE_TIME_INVALID")
    frame = frame.sort_values("time", kind="mergesort").reset_index(drop=True)
    history = frame[(frame["time"] >= history_start) & (frame["time"] <= fit_end)].copy()
    if history.empty:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_HISTORY_EMPTY")
    if history["time"].iloc[0] > fit_start:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_HISTORY_DOES_NOT_REACH_FIT_START")

    derived = compute_causal_market_rank_inputs(history)
    fit_mask = (history["time"] >= fit_start) & (history["time"] <= fit_end)
    fit = derived.loc[fit_mask]
    fit_times = history.loc[fit_mask, "time"].reset_index(drop=True)
    min_rows = int(args.min_rows)
    if fit.empty or (min_rows > 0 and len(fit) < min_rows):
        raise RuntimeError(
            f"MODEL_NATIVE_TRAIN_RANK_FIT_ROWS_INSUFFICIENT: rows={len(fit)} min_rows={min_rows}"
        )
    atr_sorted = np.sort(fit["atr_bps"].to_numpy(dtype=np.float64))
    spread_sorted = np.sort(fit["spread_bps"].to_numpy(dtype=np.float64))
    if not np.isfinite(atr_sorted).all() or not np.isfinite(spread_sorted).all():
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_FIT_NONFINITE")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        schema_version=np.asarray([MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION]),
        fit_start_ns=np.asarray([fit_start.value], dtype=np.int64),
        fit_end_ns=np.asarray([fit_end.value], dtype=np.int64),
        fit_row_count=np.asarray([len(fit)], dtype=np.int64),
        explicit_vedtak_id=np.asarray([explicit_vedtak_id]),
        atr_bps_sorted=atr_sorted,
        spread_bps_sorted=spread_sorted,
    )
    pre_fit_history_rows = int((history["time"] < fit_start).sum())
    report = {
        "schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "explicit_vedtak_id": explicit_vedtak_id,
        "fit_scope": "train_only",
        "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
        "row_level_state_present": False,
        "forbidden_row_level_keys_absent": True,
        "source_parquet": str(source),
        "source_parquet_sha256": sha256_file(source),
        "out_npz": str(out),
        "out_npz_sha256": sha256_file(out),
        "history_start_utc": str(history_start),
        "fit_start_utc": str(fit_start),
        "fit_end_utc": str(fit_end),
        "fit_row_count": int(len(fit)),
        "fit_time_min": str(fit_times.iloc[0]),
        "fit_time_max": str(fit_times.iloc[-1]),
        "pre_fit_history_row_count": pre_fit_history_rows,
        "history_time_min": str(history["time"].iloc[0]),
        "history_time_max": str(history["time"].iloc[-1]),
        "stored_distributions": ["atr_bps_sorted", "spread_bps_sorted"],
        "validation_or_test_rows_stored": False,
    }
    sidecar.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--fit-start", required=True)
    parser.add_argument("--fit-end", required=True)
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument(
        "--vedtak",
        required=True,
        help="Explicit user decision ID bound into the immutable rank-reference sidecar.",
    )
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
