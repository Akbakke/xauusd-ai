#!/usr/bin/env python3
"""
Build ENTRY_V10 dataset (ONE UNIVERSE) from XGB-annotated entry data.

ONE UNIVERSE contract:
- Transformers consume:
  - seq: [N, lookback, 7]   (XGB_SIGNAL_BRIDGE_V1 over time)
  - snap: [N, 7]            (XGB_SIGNAL_BRIDGE_V1 "now")
  - ctx_cont: [N, 6]        (ORDERED_CTX_CONT_NAMES_EXTENDED[:6])
  - ctx_cat:  [N, 6]        (ORDERED_CTX_CAT_NAMES_EXTENDED[:6])
  - y_direction: int labels

Input:
  - XGB-annotated parquet: data/entry_v10/xgb_annotated.parquet
    Must contain:
      - all XGB_SIGNAL_BRIDGE_V1 fields (7 signals)
      - all ctx contract columns for 6/6

Output:
  - data/entry_v10/entry_v10_dataset.parquet
    Contains:
      - seq, snap, ctx_cont, ctx_cat, y_direction (+ ts/time if present)

Usage:
  python -m gx1.rl.entry_v10.build_entry_v10_dataset \
    --input-parquet data/entry_v10/xgb_annotated.parquet \
    --output-parquet data/entry_v10/entry_v10_dataset.parquet \
    --lookback 30

Note:
  - NO padding. Rows without sufficient history (< lookback bars) are dropped.
  - NO legacy: no 16/88, no session_id/vol_regime_id/trend_regime_id columns.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gx1.contracts.signal_bridge_v1 import (
    ORDERED_CTX_CAT_NAMES_EXTENDED,
    ORDERED_CTX_CONT_NAMES_EXTENDED,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEQ_DIM = 7
SNAP_DIM = 7
CTX_CONT_DIM = 6
CTX_CAT_DIM = 6


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _infer_signal_fields(df: pd.DataFrame) -> List[str]:
    """
    Resolve the 7 signal_bridge fields from the gx1.contracts.signal_bridge_v1 module if available,
    else fall back to common names.

    Hard fail if we cannot resolve exactly 7 present columns.
    """
    # Try to import canonical ordered signal field list (preferred).
    fields: Optional[List[str]] = None
    try:
        from gx1.contracts.signal_bridge_v1 import ORDERED_FIELDS  # type: ignore

        if isinstance(ORDERED_FIELDS, (list, tuple)) and all(isinstance(x, str) for x in ORDERED_FIELDS):
            fields = list(ORDERED_FIELDS)
    except Exception:
        fields = None

    # Known fallback names (still requires presence in df).
    if not fields:
        fields = [
            "p_long",
            "p_short",
            "p_flat",
            "p_hat",
            "uncertainty_score",
            "margin_top1_top2",
            "entropy",
        ]

    present = [c for c in fields if c in df.columns]
    _require(
        len(present) == 7,
        f"[ONE_UNIVERSE_FAIL] Missing signal bridge columns. "
        f"Need 7 fields={fields}. Present={present}. Missing={[c for c in fields if c not in df.columns]}",
    )
    return present


def _get_time_col(df: pd.DataFrame) -> Optional[str]:
    if "ts" in df.columns:
        return "ts"
    if "time" in df.columns:
        return "time"
    return None


def _coerce_numeric_series(s: pd.Series) -> np.ndarray:
    """
    Convert series to float32 array, coercing non-numeric to NaN then to 0.0.
    """
    if s.dtype == object:
        s = pd.to_numeric(s, errors="coerce")
    arr = s.to_numpy(copy=False)
    # Convert to float32
    arr = arr.astype(np.float32, copy=False)
    # Replace NaN/Inf with 0.0 (dataset build should be strict-ish, but keep it deterministic)
    bad = ~np.isfinite(arr)
    if np.any(bad):
        arr = arr.copy()
        arr[bad] = 0.0
    return arr


def _extract_ctx_cont_row(row: pd.Series, ctx_cols: List[str]) -> np.ndarray:
    vals = []
    for c in ctx_cols:
        v = row.get(c, None)
        if v is None:
            raise RuntimeError(f"[CTX_CONT_MISSING] Missing ctx_cont column: {c}")
        try:
            fv = float(v)
        except Exception:
            fv = float(pd.to_numeric(v, errors="coerce") or 0.0)
        if not np.isfinite(fv):
            fv = 0.0
        vals.append(fv)
    arr = np.array(vals, dtype=np.float32)
    _require(arr.shape == (CTX_CONT_DIM,), f"[CTX_CONT_DIM_MISMATCH] got={arr.shape} expected={(CTX_CONT_DIM,)}")
    return arr


def _extract_ctx_cat_row(row: pd.Series, ctx_cols: List[str]) -> np.ndarray:
    vals = []
    for c in ctx_cols:
        v = row.get(c, None)
        if v is None:
            raise RuntimeError(f"[CTX_CAT_MISSING] Missing ctx_cat column: {c}")
        try:
            iv = int(v)
        except Exception:
            iv = int(pd.to_numeric(v, errors="coerce") or 0)
        vals.append(iv)
    arr = np.array(vals, dtype=np.int64)
    _require(arr.shape == (CTX_CAT_DIM,), f"[CTX_CAT_DIM_MISMATCH] got={arr.shape} expected={(CTX_CAT_DIM,)}")
    return arr


def build_entry_v10_dataset_one_universe(
    input_parquet: Path,
    output_parquet: Path,
    lookback: int = 30,
) -> None:
    log.info(f"Loading input dataset: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    log.info(f"Loaded {len(df):,} rows")

    _require(len(df) > 0, "[DATASET_EMPTY] input parquet has 0 rows")

    # Hard-ban legacy columns if present (ONE UNIVERSE)
    banned_cols = {"session_id", "vol_regime_id", "trend_regime_id", "atr_regime_id"}
    present_banned = sorted(banned_cols & set(df.columns))
    _require(
        not present_banned,
        f"[ONE_UNIVERSE_FAIL] Legacy regime cols present in input (forbidden): {present_banned}",
    )

    # Resolve signal fields (7)
    signal_cols = _infer_signal_fields(df)
    log.info(f"Signal bridge fields (7): {signal_cols}")

    # ctx 6/6 columns (contract prefix order)
    ctx_cont_cols = list(ORDERED_CTX_CONT_NAMES_EXTENDED[:CTX_CONT_DIM])
    ctx_cat_cols = list(ORDERED_CTX_CAT_NAMES_EXTENDED[:CTX_CAT_DIM])

    missing_ctx = [c for c in (ctx_cont_cols + ctx_cat_cols) if c not in df.columns]
    _require(
        not missing_ctx,
        f"[ONE_UNIVERSE_FAIL] Missing ctx contract columns for 6/6. Missing={missing_ctx}",
    )
    log.info(f"ctx_cont(6) cols: {ctx_cont_cols}")
    log.info(f"ctx_cat(6) cols: {ctx_cat_cols}")

    # Labels
    if "y_direction" not in df.columns:
        if "y" in df.columns:
            df["y_direction"] = (pd.to_numeric(df["y"], errors="coerce").fillna(0.0) > 0).astype(int)
            log.warning("[LABELS] y_direction missing; derived from y>0.")
        else:
            raise RuntimeError("[LABELS_MISSING] Missing y_direction (and no y to derive)")

    # Sort by time if possible (keeps sequences sane)
    tcol = _get_time_col(df)
    if tcol:
        df = df.sort_values(tcol).reset_index(drop=True)
        log.info(f"Sorted by {tcol}")
    else:
        log.warning("No ts/time column found; using current row order")

    _require(lookback >= 2, f"[LOOKBACK_FAIL] lookback must be >=2, got {lookback}")

    # Pre-coerce signal columns to numeric arrays (fast)
    sig_arrays: Dict[str, np.ndarray] = {}
    for c in signal_cols:
        sig_arrays[c] = _coerce_numeric_series(df[c])

    y_arr = pd.to_numeric(df["y_direction"], errors="coerce").fillna(0).astype(int).to_numpy()

    sequences: List[List[List[float]]] = []
    snapshots: List[List[float]] = []
    ctx_conts: List[List[float]] = []
    ctx_cats: List[List[int]] = []
    y_directions: List[int] = []
    kept_idx: List[int] = []

    n_valid = 0
    n_skipped = 0

    log.info(f"Building sequences with lookback={lookback} (NO padding; drop <lookback)")
    for i in range(lookback - 1, len(df)):
        start = i - lookback + 1
        if start < 0:
            n_skipped += 1
            continue

        # seq: [lookback, 7]
        seq_mat = np.zeros((lookback, SEQ_DIM), dtype=np.float32)
        for j, c in enumerate(signal_cols):
            seq_mat[:, j] = sig_arrays[c][start : i + 1]

        # snap: [7] (current row)
        snap_vec = seq_mat[-1, :].copy()

        # ctx: per-row (current row)
        row = df.iloc[i]
        ctx_cont = _extract_ctx_cont_row(row, ctx_cont_cols)
        ctx_cat = _extract_ctx_cat_row(row, ctx_cat_cols)

        y = int(y_arr[i])

        sequences.append(seq_mat.tolist())
        snapshots.append(snap_vec.astype(np.float32).tolist())
        ctx_conts.append(ctx_cont.astype(np.float32).tolist())
        ctx_cats.append(ctx_cat.astype(np.int64).tolist())
        y_directions.append(y)
        kept_idx.append(i)
        n_valid += 1

        if (i + 1) % 20000 == 0:
            log.info(f"Processed {i + 1:,}/{len(df):,} rows; valid={n_valid:,} skipped={n_skipped:,}")

    log.info(f"Built {n_valid:,} valid rows (skipped {n_skipped:,})")
    _require(n_valid > 0, "[NO_OUTPUT] No valid sequences produced (check lookback vs dataset length)")

    out: Dict[str, Any] = {
        "seq": sequences,
        "snap": snapshots,
        "ctx_cont": ctx_conts,
        "ctx_cat": ctx_cats,
        "y_direction": y_directions,
    }

    # Preserve timestamp column aligned to kept indices
    if tcol:
        out[tcol] = df.loc[kept_idx, tcol].to_list()

    output_df = pd.DataFrame(out)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving ONE UNIVERSE dataset to {output_parquet}")
    output_df.to_parquet(output_parquet, index=False)
    log.info(f"Saved {len(output_df):,} rows")

    # Summary
    log.info("Dataset summary (ONE UNIVERSE):")
    log.info(f"  - Rows: {len(output_df):,}")
    log.info(f"  - seq shape: ({lookback}, {SEQ_DIM})")
    log.info(f"  - snap shape: ({SNAP_DIM},)")
    log.info(f"  - ctx_cont shape: ({CTX_CONT_DIM},)")
    log.info(f"  - ctx_cat shape: ({CTX_CAT_DIM},)")
    log.info(f"  - y_direction distribution: {pd.Series(y_directions).value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ENTRY_V10 dataset (ONE UNIVERSE) from XGB-annotated data")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Path to XGB-annotated dataset (must include signal_bridge_v1 + ctx 6/6 cols)",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        required=True,
        help="Path to save ONE UNIVERSE V10 dataset",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Sequence length (default: 30). No padding; drops rows with insufficient history.",
    )

    args = parser.parse_args()

    build_entry_v10_dataset_one_universe(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        lookback=args.lookback,
    )


if __name__ == "__main__":
    main()