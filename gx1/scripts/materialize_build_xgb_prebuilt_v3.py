#!/usr/bin/env python3
"""Build XGB v3 prebuilt parquet by merging canonical_v2 + BASE28's V10 ctx features.

Mission
-------
canonical_features_v2.parquet has 49 _v1_* features + 11 new D1/M15 indicators,
but lacks the 22 V10 runtime ctx features (atr_bps, session_id, swing-state, etc.)
that XGB v3 retraining needs. Those are pre-computed in the existing BASE28
prebuilt parquet.

This script does an inner-join on time to produce a single prebuilt that has:
  - All canonical_v2 columns (49 _v1_* + 11 D1/M15 + high-level + m5_phase + OHLC)
  - All BASE28 ctx columns NOT in canonical_v2 (22 V10 ctx + 5 session-conditioned _v1_int_*)

Output schema (all feature universes in one place):
  - time + OHLC carry-over
  - 49 _v1_* base features (from canonical_v2)
  - 5 _v1_int_* session-conditioned features (from BASE28)
  - 19 high-level features (atr, ret, ema_slope, rvol, etc — from canonical_v2)
  - 5 m5_phase one-hot (from canonical_v2)
  - 11 D1/M15 new features (from canonical_v2)
  - 22 V10 ctx + categorical (from BASE28: session_id, regime, swing, micro_momentum, HTF)
  → ~120 cols total, ~96 are usable features for XGB/V10/V3 universe

XGB v3 selects ~91 of these via xgb_input_features_canonical_v2_v1.json contract.

Output
------
  /home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V2_PREBUILT/
      xauusd_m5_CANONICAL_V2_2020_2026.parquet

Research-only. No runtime promotion until 2026-Q2 rebuild deploys.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)


ACTION = "BUILD_XGB_PREBUILT_V3_FROM_CANONICAL_V2"
DEFAULT_CANONICAL_V2_PATH = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANONICAL_FEATURES_V2/canonical_features_v2.parquet"
)
DEFAULT_BASE28_PREBUILT = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/ctx16cat6_20260315_210518/"
    "xauusd_m5_BASE28_2020_2026__CTX_CONT16_CAT6_20260315_210518.parquet"
)
DEFAULT_OUT_PATH = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V2_PREBUILT/"
    "xauusd_m5_CANONICAL_V2_2020_2026.parquet"
)

_stamp = contract_gate._stamp
_utc_now = contract_gate._utc_now
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json


def merge_canonical_v2_with_base28_ctx(
    canonical_v2_path: Path,
    base28_path: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Inner-join on time. Returns (merged_df, column_inventory)."""
    print(f"[{ACTION}] reading canonical_v2 from {canonical_v2_path}", flush=True)
    cv2 = pd.read_parquet(canonical_v2_path)
    cv2["time"] = pd.to_datetime(cv2["time"], utc=True)
    # Drop OHLC carry-over — XGB trainer joins the M5 tape later for OHLC.
    # Keeping these columns here would cause merge collisions (close_x / close_y).
    drop_ohlc = [c for c in ("open", "high", "low", "close", "volume",
                              "bid_open", "bid_high", "bid_low", "bid_close",
                              "ask_open", "ask_high", "ask_low", "ask_close") if c in cv2.columns]
    if drop_ohlc:
        cv2 = cv2.drop(columns=drop_ohlc)
        print(f"[{ACTION}] dropped OHLC carry-over from canonical_v2: {drop_ohlc}", flush=True)
    print(f"[{ACTION}] canonical_v2: {len(cv2):,} rows × {len(cv2.columns)} cols", flush=True)

    print(f"[{ACTION}] reading BASE28 prebuilt from {base28_path}", flush=True)
    base28 = pd.read_parquet(base28_path)
    base28["time"] = pd.to_datetime(base28["time"], utc=True)
    print(f"[{ACTION}] BASE28: {len(base28):,} rows × {len(base28.columns)} cols", flush=True)

    # Identify unique columns to take from BASE28 (those NOT already in canonical_v2)
    cv2_cols = set(cv2.columns)
    base28_extras = [c for c in base28.columns if c not in cv2_cols and c != "time"]
    print(f"[{ACTION}] BASE28 columns to merge in (not in canonical_v2): {len(base28_extras)}", flush=True)
    for c in base28_extras:
        print(f"    + {c}", flush=True)

    # Inner-join on time — both should have the same M5 grid 2020-2026
    base28_keep = base28[["time"] + base28_extras].copy()
    print(f"[{ACTION}] inner-joining on time...", flush=True)
    t0 = _time.time()
    merged = cv2.merge(base28_keep, on="time", how="inner", validate="one_to_one")
    elapsed = _time.time() - t0
    print(f"[{ACTION}] merge done in {elapsed:.1f}s, "
          f"result: {len(merged):,} rows × {len(merged.columns)} cols", flush=True)

    if len(merged) == 0:
        raise RuntimeError(f"[{ACTION}] inner join produced 0 rows — time grids don't overlap?")
    if len(merged) < min(len(cv2), len(base28)) * 0.95:
        print(f"[{ACTION}] WARNING: merged rows ({len(merged):,}) << either input "
              f"(cv2={len(cv2):,}, base28={len(base28):,}) — possible time-grid mismatch", flush=True)

    inventory = {
        "_v1_canonical_v2": [c for c in merged.columns if c.startswith("_v1_") and not c.startswith("_v1h") and c not in base28_extras],
        "_v1_int_base28": [c for c in base28_extras if c.startswith("_v1_int_") or c.startswith("_v1_is_")],
        "_v1h1_canonical_v2": [c for c in merged.columns if c.startswith("_v1h1_")],
        "_v1h4_canonical_v2": [c for c in merged.columns if c.startswith("_v1h4_")],
        "high_level_canonical_v2": [c for c in merged.columns if c in (
            "mid", "range", "body_pct", "wick_asym", "atr", "atr50", "atr_z",
            "std50", "ret_1", "ret_5", "ret_20", "roc20", "roc100", "rvol_20",
            "rvol_60", "ema20_slope", "ema100_slope", "pos_vs_ema200", "vol_ratio",
        )],
        "m5_phase": [c for c in merged.columns if c.startswith("m5_phase_")],
        "d1_canon_v2": [c for c in merged.columns if c.startswith("d1_") and c.endswith("_canon_v2")],
        "m15_canon_v2": [c for c in merged.columns if c.startswith("m15_") and c.endswith("_canon_v2")],
        "v10_ctx_cont_base28": [c for c in base28_extras if c in (
            "atr_bps", "spread_bps", "D1_dist_from_ema200_atr", "H1_range_compression_ratio",
            "D1_atr_percentile_252", "M15_range_compression_ratio",
            "micro_momentum_3", "micro_momentum_5", "micro_acceleration",
            "wick_ratio", "distance_ema_fast",
            "dist_last_swing_high_atr", "dist_last_swing_low_atr",
            "bars_since_swing_high", "bars_since_swing_low",
            "retracement_from_last_impulse",
        )],
        "v10_ctx_cat_base28": [c for c in base28_extras if c in (
            "session_id", "trend_regime_id", "vol_regime_id",
            "atr_bucket", "spread_bucket", "H4_trend_sign_cat",
        )],
        "session_timing_base28": [c for c in base28_extras if c in (
            "is_ASIA", "minutes_since_session_open", "minutes_to_next_session_boundary",
            "session_change_flag", "session_tradable",
        )],
    }
    return merged, inventory


def write_artifacts(
    out_path: Path = DEFAULT_OUT_PATH,
    *,
    canonical_v2_path: Path = DEFAULT_CANONICAL_V2_PATH,
    base28_path: Path = DEFAULT_BASE28_PREBUILT,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged, inventory = merge_canonical_v2_with_base28_ctx(canonical_v2_path, base28_path)

    print(f"[{ACTION}] feature inventory:", flush=True)
    for grp, cols in inventory.items():
        print(f"    {grp}: {len(cols)} features", flush=True)
    total_in_inventory = sum(len(v) for v in inventory.values())
    print(f"    INVENTORY TOTAL: {total_in_inventory} features (+ time, OHLC carryover)", flush=True)
    print(f"    PARQUET TOTAL COLS: {len(merged.columns)}", flush=True)

    merged.to_parquet(out_path, index=False)
    print(f"[{ACTION}] wrote {out_path} ({out_path.stat().st_size / 1024**2:.1f}MB)", flush=True)

    # Also write a CURRENT_MANIFEST.json so train_xgb can resolve via manifest
    manifest_path = out_path.parent / "CURRENT_MANIFEST.json"
    manifest = {
        "version": "v1",
        "created_at": _utc_now(),
        "canonical_prebuilt_parquet": str(out_path),
        "prebuilt_parquet": str(out_path),
        "schema_hash": str(int(hash(tuple(merged.columns)) & 0xFFFFFFFFFFFFFFFF)),
        "n_rows": int(len(merged)),
        "n_cols": int(len(merged.columns)),
        "source_canonical_v2": str(canonical_v2_path),
        "source_base28_prebuilt": str(base28_path),
        "build_action": ACTION,
        "feature_inventory": {grp: cols for grp, cols in inventory.items()},
    }
    _write_json(manifest_path, manifest)
    print(f"[{ACTION}] wrote manifest {manifest_path}", flush=True)

    summary = {
        "layer_name": "XGB_PREBUILT_V3_SUMMARY",
        "action_v1": ACTION,
        "out_path_v1": str(out_path),
        "manifest_path_v1": str(manifest_path),
        "built_at_utc_v1": _utc_now(),
        "canonical_v2_source_v1": str(canonical_v2_path),
        "base28_source_v1": str(base28_path),
        "n_rows_v1": int(len(merged)),
        "n_cols_v1": int(len(merged.columns)),
        "feature_inventory_v1": {grp: cols for grp, cols in inventory.items()},
        "feature_counts_per_group_v1": {grp: len(cols) for grp, cols in inventory.items()},
        "feature_total_excluding_time_ohlc_v1": total_in_inventory,
        "research_only_v1": True,
    }
    summary_path = out_path.parent / "xgb_prebuilt_v3_summary.json"
    _write_json(summary_path, summary)
    return {"out_path": str(out_path), "manifest_path": str(manifest_path), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION}.")
    parser.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--canonical-v2", type=str, default=str(DEFAULT_CANONICAL_V2_PATH))
    parser.add_argument("--base28-prebuilt", type=str, default=str(DEFAULT_BASE28_PREBUILT))
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()

    result = write_artifacts(
        out_path=Path(args.out_path).expanduser().resolve(),
        canonical_v2_path=Path(args.canonical_v2).expanduser().resolve(),
        base28_path=Path(args.base28_prebuilt).expanduser().resolve(),
        built_at_utc=args.built_at_utc,
    )
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
