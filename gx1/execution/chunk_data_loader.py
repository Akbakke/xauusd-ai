"""
TRUTH-grade Chunk Data Loader

Responsibilities:
- Load RAW + PREBUILT parquet deterministically
- Enforce TRUTH invariants (1W1C, schema, tz, join ratios)
- Align on eval window (SSoT) with optional left-padding
- Write chunk_{idx}_data.parquet atomically
- Emit RAW_PREBUILT_JOIN.json with eval-level diagnostics

NO FALLBACKS. Any ambiguity = hard fail.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from gx1.execution.chunk_bootstrap import BootstrapContext
from gx1.runtime.column_collision_guard import assert_no_case_collisions
from gx1.utils.atomic_json import atomic_write_json
from gx1.utils.ts_utils import ensure_ts_column

log = logging.getLogger(__name__)


def _read_parquet_schema_columns(path: Path) -> Set[str]:
    """Return set of column names in parquet file (for strict schema validation)."""
    import pyarrow.parquet as pq  # type: ignore

    return set(pq.read_schema(path).names)


# -----------------------------
# Data contracts
# -----------------------------


@dataclass
class DataContext:
    # SSoT dataframes
    chunk_df: pd.DataFrame  # indexed by ts (UTC), includes prebuilt
    chunk_df_save: pd.DataFrame  # flat, with `time` column

    # Paths
    chunk_data_path_abs: Path

    # Bar counts (explicit, footer-ready)
    bars_total_input_all: int  # incl padding (after join)
    bars_total_eval: int  # eval window only (after join)

    # Timestamps
    actual_chunk_start: pd.Timestamp
    eval_start_ts: pd.Timestamp
    eval_end_ts: pd.Timestamp

    # Prebuilt
    prebuilt_parquet_path_resolved: Optional[str]

    # Join diagnostics
    join_metrics_path: Path

    # Collision resolution (if any)
    case_collision_resolution: Optional[Dict[str, Any]]

    # Timings
    t_load_raw_s: float
    t_load_prebuilt_s: float
    t_join_s: float
    t_write_s: float

    # Prebuilt carried to runner (runner.prebuilt_used / prebuilt_features_df)
    prebuilt_features_df: Optional[pd.DataFrame] = None
    prebuilt_used: bool = False


# -----------------------------
# Helpers (TRUTH strict)
# -----------------------------


def _require_utc_ts(ts: pd.Timestamp, label: str) -> pd.Timestamp:
    if not isinstance(ts, pd.Timestamp):
        raise RuntimeError(f"[TS_FAIL] {label} is not a pandas Timestamp")
    if ts.tz is None:
        raise RuntimeError(f"[TS_FAIL] {label} must be tz-aware UTC, got tz=None")
    return ts.tz_convert("UTC")


def _require_utc_dtindex(df: pd.DataFrame, label: str) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError(f"[TS_FAIL] {label}: index is not DatetimeIndex")
    if df.index.tz is None:
        raise RuntimeError(f"[TS_FAIL] {label}: index tz is None (must be tz-aware UTC)")
    # Normalize to UTC (no-op if already UTC)
    df.index = df.index.tz_convert("UTC")


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    tmp = out_path.with_suffix(f".parquet.tmp.{os.getpid()}")
    df.to_parquet(tmp, index=False, compression="snappy")

    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for blk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(blk)

    os.replace(tmp, out_path)
    log.info(f"[DATA_WRITE] parquet={out_path} bytes={out_path.stat().st_size} sha256={h.hexdigest()[:16]}...")


def _select_prebuilt_columns(bootstrap_ctx: BootstrapContext) -> List[str]:
    """
    TRUTH: prebuilt columns are explicit SSoT (bootstrap_ctx.prebuilt_required_columns),
    sourced from canonical truth config. No fallback.

    Loader contract: we ALWAYS read 'time' in addition to required columns (for parquet filters + ensure_ts_column),
    but 'time' is not treated as a feature column and is dropped before join.
    """
    required = getattr(bootstrap_ctx, "prebuilt_required_columns", None)
    if required is None:
        raise RuntimeError(
            "BootstrapContext missing prebuilt_required_columns. "
            "Set it in chunk_bootstrap from canonical truth config."
        )
    cols = list(required)
    if "time" not in cols:
        cols = ["time"] + cols
    return cols


def _drop_time_column_if_present(df: pd.DataFrame) -> pd.DataFrame:
    # TRUTH: avoid raw/prebuilt column overlap on "time" (we join on ts-index).
    if "time" in df.columns:
        return df.drop(columns=["time"])
    return df


# -----------------------------
# Main loader
# -----------------------------


def load_chunk_data(
    bootstrap_ctx: BootstrapContext,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
) -> DataContext:
    """
    TRUTH-grade loader. Any ambiguity or data loss = hard fail.
    """

    # -----------------------------
    # TRUTH hard gates (1W1C)
    # -----------------------------
    workers = getattr(bootstrap_ctx, "workers", 1)
    chunks = getattr(bootstrap_ctx, "chunks", 1)
    if workers != 1 or chunks != 1:
        raise RuntimeError("[TRUTH_1W1C_ONLY] chunk_data_loader requires workers=1, chunks=1")

    chunk_idx = int(bootstrap_ctx.chunk_idx)

    eval_start = _require_utc_ts(chunk_start, "chunk_start")
    eval_end = _require_utc_ts(chunk_end, "chunk_end")
    if eval_start >= eval_end:
        raise RuntimeError("[TS_FAIL] chunk_start >= chunk_end")

    padding_days = int(getattr(bootstrap_ctx, "chunk_local_padding_days", 0) or 0)
    actual_chunk_start = eval_start - pd.Timedelta(days=padding_days) if padding_days > 0 else eval_start

    # -----------------------------
    # STEP 1: Load RAW (OHLC + bid/ask if present)
    # -----------------------------
    RAW_TIME_COL = "time"
    RAW_COLS = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]

    t0 = time.time()

    raw_schema = _read_parquet_schema_columns(bootstrap_ctx.data_path)
    raw_load_cols = [c for c in RAW_COLS if c in raw_schema]

    # TRUTH minimum requirement
    needed = {"time", "open", "high", "low", "close"}
    if not needed.issubset(set(raw_load_cols)):
        raise RuntimeError(
            f"[CHUNK {chunk_idx}] RAW parquet must have time + OHLC; missing={sorted(list(needed - set(raw_load_cols)))} "
            f"schema_head={sorted(list(raw_schema))[:30]}..."
        )

    raw_df = pd.read_parquet(
        bootstrap_ctx.data_path,
        columns=raw_load_cols,
        filters=[
            (RAW_TIME_COL, ">=", actual_chunk_start),
            (RAW_TIME_COL, "<=", eval_end),
        ],
    )
    if raw_df.empty:
        raise RuntimeError(f"[CHUNK {chunk_idx}] RAW empty in range [{actual_chunk_start}, {eval_end}]")

    raw_df = ensure_ts_column(raw_df, context=f"RAW chunk_{chunk_idx}")
    raw_df = raw_df.sort_values("ts").drop_duplicates(subset=["ts"], keep="first")
    raw_df = raw_df.set_index("ts")
    _require_utc_dtindex(raw_df, label=f"RAW chunk_{chunk_idx}")

    # Drop "time" column to avoid overlap on join (we re-materialize later)
    raw_df = _drop_time_column_if_present(raw_df)

    t_load_raw_s = time.time() - t0
    log.info(f"[CHUNK {chunk_idx}] RAW rows={len(raw_df)} load={t_load_raw_s:.2f}s")

    # Eval slice (SSoT) from RAW
    raw_eval = raw_df.loc[(raw_df.index >= eval_start) & (raw_df.index <= eval_end)]
    if raw_eval.empty:
        raise RuntimeError(f"[CHUNK {chunk_idx}] RAW eval window empty [{eval_start}, {eval_end}]")

    # -----------------------------
    # STEP 2: Load PREBUILT (SSoT resolved path + required columns)
    # -----------------------------
    if not getattr(bootstrap_ctx, "prebuilt_enabled", False):
        raise RuntimeError("[TRUTH_NO_FALLBACK] PREBUILT must be enabled in TRUTH")

    prebuilt_resolved = getattr(bootstrap_ctx, "prebuilt_parquet_path_resolved", None)
    if not prebuilt_resolved:
        raise RuntimeError("[TRUTH_NO_FALLBACK] prebuilt_enabled but no prebuilt path resolved (bootstrap)")

    prebuilt_path = Path(str(prebuilt_resolved)).expanduser().resolve()
    if not prebuilt_path.is_file():
        raise RuntimeError(f"[TRUTH_NO_FALLBACK] prebuilt parquet not found: {prebuilt_path}")

    prebuilt_cols = _select_prebuilt_columns(bootstrap_ctx)

    # TRUTH: enforce prebuilt schema contains requested columns
    prebuilt_schema_cols = _read_parquet_schema_columns(prebuilt_path)
    missing_prebuilt_cols = [c for c in prebuilt_cols if c not in prebuilt_schema_cols]
    if missing_prebuilt_cols:
        raise RuntimeError(
            f"[CHUNK {chunk_idx}] PREBUILT missing required columns: {missing_prebuilt_cols} (prebuilt={prebuilt_path})"
        )

    t1 = time.time()
    prebuilt_df = pd.read_parquet(
        prebuilt_path,
        columns=prebuilt_cols,
        filters=[
            ("time", ">=", actual_chunk_start),
            ("time", "<=", eval_end),
        ],
    )
    t_load_prebuilt_s = time.time() - t1

    if prebuilt_df.empty:
        raise RuntimeError(f"[CHUNK {chunk_idx}] PREBUILT empty in required range [{actual_chunk_start}, {eval_end}]")

    prebuilt_df = ensure_ts_column(prebuilt_df, context=f"PREBUILT chunk_{chunk_idx}")
    prebuilt_df = prebuilt_df.sort_values("ts").drop_duplicates(subset=["ts"], keep="first")
    prebuilt_df = prebuilt_df.set_index("ts")
    _require_utc_dtindex(prebuilt_df, label=f"PREBUILT chunk_{chunk_idx}")

    # Drop "time" column to avoid overlap on join (we join on index)
    prebuilt_df = _drop_time_column_if_present(prebuilt_df)

    # Collision guard BEFORE join
    case_collision_resolution = assert_no_case_collisions(
        prebuilt_df,
        context=f"PREBUILT before join chunk_{chunk_idx}",
        allow_close_alias_compat=False,
    )

    # Hard overlap check (TRUTH): after dropping "time", there should be no overlap
    overlap = set(raw_df.columns) & set(prebuilt_df.columns)
    if overlap:
        raise RuntimeError(f"[CHUNK {chunk_idx}] COLUMN_OVERLAP raw vs prebuilt not allowed: {sorted(list(overlap))}")

    # -----------------------------
    # STEP 2b: Join (inner = TRUTH strict)
    # -----------------------------
    t2 = time.time()
    chunk_df = raw_df.join(prebuilt_df, how="inner")
    t_join_s = time.time() - t2

    # Eval join diagnostics (SSoT)
    join_eval = chunk_df.loc[(chunk_df.index >= eval_start) & (chunk_df.index <= eval_end)]
    join_ratio_eval = len(join_eval) / len(raw_eval) if len(raw_eval) > 0 else 0.0

    JOIN_RATIO_TRUTH = 0.995
    join_metrics_path = Path(bootstrap_ctx.chunk_output_dir) / "RAW_PREBUILT_JOIN.json"

    join_metrics: Dict[str, Any] = {
        "chunk_id": chunk_idx,
        "run_id": getattr(bootstrap_ctx, "run_id", None),
        "raw_rows_all": int(len(raw_df)),
        "raw_rows_eval": int(len(raw_eval)),
        "prebuilt_rows_all": int(len(prebuilt_df)),
        "join_rows_all": int(len(chunk_df)),
        "join_rows_eval": int(len(join_eval)),
        # Stable key name(s)
        "join_ratio": float(join_ratio_eval),
        "join_ratio_eval": float(join_ratio_eval),
        "join_ratio_threshold_truth": float(JOIN_RATIO_TRUTH),
        "ts_min_eval": str(raw_eval.index.min()),
        "ts_max_eval": str(raw_eval.index.max()),
        "missing_eval_bars": int(len(raw_eval) - len(join_eval)),
        "t_load_raw_s": float(t_load_raw_s),
        "t_load_prebuilt_s": float(t_load_prebuilt_s),
        "t_join_s": float(t_join_s),
        "prebuilt_parquet_path_resolved": str(prebuilt_path),
        "prebuilt_required_columns_len": int(len(getattr(bootstrap_ctx, "prebuilt_required_columns") or [])),
    }
    atomic_write_json(join_metrics_path, join_metrics)

    if join_ratio_eval < JOIN_RATIO_TRUTH:
        raise RuntimeError(f"[JOIN_RATIO_FAIL] eval join_ratio {join_ratio_eval:.4f} < {JOIN_RATIO_TRUTH}")

    log.info(
        f"[CHUNK {chunk_idx}] PREBUILT join eval_ratio={join_ratio_eval:.4f} "
        f"load={t_load_prebuilt_s:.2f}s join={t_join_s:.2f}s"
    )

    # -----------------------------
    # STEP 3: Prepare save DF
    # -----------------------------
    bars_total_input_all = int(len(chunk_df))
    bars_total_eval = int(len(join_eval))

    # Flatten for downstream: "time" column required
    chunk_df_save = chunk_df.reset_index().rename(columns={"ts": "time"})

    # TRUTH sanity: OHLC must exist after join
    for col in ("open", "high", "low", "close"):
        if col not in chunk_df_save.columns:
            raise RuntimeError(f"[DATA_FAIL] Missing OHLC column after join: {col}")

    # Collision guard before write (no case collisions)
    case_collision_resolution_save = assert_no_case_collisions(
        chunk_df_save,
        context=f"chunk_df_save before write chunk_{chunk_idx}",
        allow_close_alias_compat=False,
    )
    if case_collision_resolution_save:
        case_collision_resolution = case_collision_resolution_save

    # -----------------------------
    # STEP 4: Write parquet atomically
    # -----------------------------
    out_path = Path(bootstrap_ctx.chunk_output_dir) / f"chunk_{chunk_idx}_data.parquet"
    t3 = time.time()
    _atomic_write_parquet(chunk_df_save, out_path)
    t_write_s = time.time() - t3

    # -----------------------------
    # Done
    # -----------------------------
    return DataContext(
        chunk_df=chunk_df,
        chunk_df_save=chunk_df_save,
        chunk_data_path_abs=out_path,
        bars_total_input_all=bars_total_input_all,
        bars_total_eval=bars_total_eval,
        actual_chunk_start=actual_chunk_start,
        eval_start_ts=eval_start,
        eval_end_ts=eval_end,
        prebuilt_parquet_path_resolved=str(prebuilt_path),
        join_metrics_path=join_metrics_path,
        case_collision_resolution=case_collision_resolution,
        t_load_raw_s=float(t_load_raw_s),
        t_load_prebuilt_s=float(t_load_prebuilt_s),
        t_join_s=float(t_join_s),
        t_write_s=float(t_write_s),
        prebuilt_features_df=prebuilt_df,
        prebuilt_used=True,
    )