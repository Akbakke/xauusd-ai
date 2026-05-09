#!/usr/bin/env python3
"""Build canonical M5 feature parquet V2 — extends V1 with explicit D1 + M15 features.

Mission
-------
canonical_features_v2 = canonical_features_v1 (84 features)
                      + 6 D1 indicators (forward-aligned to M5 timestamps)
                      + 5 M15 indicators (forward-aligned to M5 timestamps)
                      = 95 features per M5 bar

Why v2
------
Audit (2026-05-02) revealed the GX1 stack's downstream consumers (V10 transformer,
V3 transformer, entry-IQL, exit-IQL) all use canonical_features_v1 as their multi-TF
context source, but v1 is missing **explicit D1 and M15 indicators**:
  - V1 has 84 features but multi-TF coverage is dominated by H1 (6) + H4 (5).
  - D1 is represented only by 2 single-value runtime features inside V10's ctx_cont.
  - M15 is represented only by 1 (M15_range_compression_ratio).

For the 2026-Q2 full-stack rebuild (XGB + V10 + V3 + IQLs all retrained on 2020-2026
bidirectional XAUUSD), the four model layers must consume from a unified feature
universe. canonical_v2 is the foundation: every layer downstream joins from this
single parquet.

New features added (11 total)
-----------------------------
D1 (6):
  d1_atr14_canon_v2                 — 14-day ATR (raw vol-regime indicator)
  d1_rsi14_canon_v2                 — 14-day RSI (D1 momentum)
  d1_ema_slope_20_canon_v2          — slope of D1 EMA(20), 5-day lookback
  d1_range_z_20_canon_v2            — D1 daily-range z-score over 20 days
  d1_close_pct_in_20day_range_canon_v2 — (close - low20) / (high20 - low20), 0..1
  d1_pct_change_5_canon_v2          — 5-day pct change

M15 (5):
  m15_atr14_canon_v2                — M15 14-bar ATR
  m15_rsi14_canon_v2                — M15 14-bar RSI
  m15_ema_slope_5_canon_v2          — M15 EMA(5) slope
  m15_range_z_20_canon_v2           — M15 range z-score
  m15_trend_sign_canon_v2           — sign of M15 EMA(5)-EMA(20) (-1/0/+1)

NOTE: features the v1 V10 ctx already has are NOT duplicated:
  - D1_dist_from_ema200_atr (V10 ctx)
  - D1_atr_percentile_252 (V10 ctx)
  - M15_range_compression_ratio (V10 ctx)
  - H1_range_compression_ratio (V10 ctx)
  - H4_trend_sign_cat (V10 ctx)

Alignment
---------
D1/M15 features are computed on resampled OHLC, then forward-aligned to each M5
timestamp via pd.merge_asof(direction='backward'). This means each M5 bar gets
the most recent D1/M15 value at-or-before its timestamp — same semantics as live
runtime.

Output
------
  /home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANONICAL_FEATURES_V2/canonical_features_v2.parquet

Research-only. No runtime promotion until 2026-Q2 rebuild deploys the full stack.
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
    materialize_build_canonical_features_v1 as v1_builder,
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.features.smc_v1 import compute_smc_features, SMC_FEATURE_NAMES


ACTION = "BUILD_CANONICAL_FEATURES_V2"
DEFAULT_M5_TAPE_ROOT = v1_builder.DEFAULT_M5_TAPE_ROOT
DEFAULT_OUT_PATH = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/CANONICAL_FEATURES_V2/canonical_features_v2.parquet"
)
SUFFIX = "_canon_v2"

D1_FEATURE_NAMES = [
    "d1_atr14",
    "d1_rsi14",
    "d1_ema_slope_20",
    "d1_range_z_20",
    "d1_close_pct_in_20day_range",
    "d1_pct_change_5",
]
M15_FEATURE_NAMES = [
    "m15_atr14",
    "m15_rsi14",
    "m15_ema_slope_5",
    "m15_range_z_20",
    "m15_trend_sign",
]

_stamp = contract_gate._stamp
_utc_now = contract_gate._utc_now
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json


# ---------------------------------------------------------------------------
# Indicator math (offline batch — pandas is fine, runtime hot-path uses numpy)
# ---------------------------------------------------------------------------


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    return _true_range(high, low, close).rolling(n, min_periods=max(2, n // 4)).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI over close-to-close changes."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.clip(0.0, 100.0)


# ---------------------------------------------------------------------------
# D1 features
# ---------------------------------------------------------------------------


def compute_d1_features(m5: pd.DataFrame) -> pd.DataFrame:
    """Resample M5 → D1, compute D1 indicators, return DataFrame indexed by D1 close-time.

    Returns columns: [time, d1_atr14_canon_v2, d1_rsi14_canon_v2, ...]
    Time = D1 bucket start (UTC midnight).
    """
    work = m5[["time", "open", "high", "low", "close"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work = work.set_index("time").sort_index()

    d1 = work.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna(how="all")

    out = pd.DataFrame(index=d1.index)
    out[f"d1_atr14{SUFFIX}"] = _atr(d1["high"], d1["low"], d1["close"], 14)
    out[f"d1_rsi14{SUFFIX}"] = _rsi(d1["close"], 14)
    ema20 = d1["close"].ewm(span=20, adjust=False).mean()
    out[f"d1_ema_slope_20{SUFFIX}"] = ema20 - ema20.shift(5)
    d1_range = d1["high"] - d1["low"]
    range_mean_20 = d1_range.rolling(20, min_periods=5).mean()
    range_std_20 = d1_range.rolling(20, min_periods=5).std().replace(0.0, np.nan)
    out[f"d1_range_z_20{SUFFIX}"] = (d1_range - range_mean_20) / range_std_20
    high20 = d1["high"].rolling(20, min_periods=5).max()
    low20 = d1["low"].rolling(20, min_periods=5).min()
    range20 = (high20 - low20).replace(0.0, np.nan)
    out[f"d1_close_pct_in_20day_range{SUFFIX}"] = (d1["close"] - low20) / range20
    out[f"d1_pct_change_5{SUFFIX}"] = d1["close"].pct_change(5) * 10000.0  # bps

    out = out.reset_index()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["_time_ns"] = out["time"].astype("int64")
    return out.sort_values("_time_ns", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# M15 features
# ---------------------------------------------------------------------------


def compute_m15_features(m5: pd.DataFrame) -> pd.DataFrame:
    """Resample M5 → M15, compute M15 indicators, return DataFrame indexed by M15 close-time."""
    work = m5[["time", "open", "high", "low", "close"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True)
    work = work.set_index("time").sort_index()

    m15 = work.resample("15min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }).dropna(how="all")

    out = pd.DataFrame(index=m15.index)
    out[f"m15_atr14{SUFFIX}"] = _atr(m15["high"], m15["low"], m15["close"], 14)
    out[f"m15_rsi14{SUFFIX}"] = _rsi(m15["close"], 14)
    ema5 = m15["close"].ewm(span=5, adjust=False).mean()
    ema20 = m15["close"].ewm(span=20, adjust=False).mean()
    out[f"m15_ema_slope_5{SUFFIX}"] = ema5 - ema5.shift(3)
    m15_range = m15["high"] - m15["low"]
    range_mean_20 = m15_range.rolling(20, min_periods=5).mean()
    range_std_20 = m15_range.rolling(20, min_periods=5).std().replace(0.0, np.nan)
    out[f"m15_range_z_20{SUFFIX}"] = (m15_range - range_mean_20) / range_std_20
    trend_diff = ema5 - ema20
    out[f"m15_trend_sign{SUFFIX}"] = np.sign(trend_diff).fillna(0.0)

    out = out.reset_index()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out["_time_ns"] = out["time"].astype("int64")
    return out.sort_values("_time_ns", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Alignment + assembly
# ---------------------------------------------------------------------------


def merge_asof_features(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    *,
    base_time_col: str = "time",
) -> pd.DataFrame:
    """merge_asof base.time ← extra._time_ns, direction='backward'.

    Returns base + extra's feature columns. extra must have _time_ns and the feature columns.
    """
    if extra is None or len(extra) == 0:
        return base
    base_sorted = base.sort_values(base_time_col, kind="mergesort").reset_index(drop=True)
    base_sorted["_base_time_ns"] = pd.to_datetime(base_sorted[base_time_col], utc=True).astype("int64")
    extra_sorted = extra.sort_values("_time_ns", kind="mergesort").reset_index(drop=True)
    feat_cols = [c for c in extra_sorted.columns if c not in ("time", "_time_ns")]

    merged = pd.merge_asof(
        base_sorted,
        extra_sorted[["_time_ns"] + feat_cols],
        left_on="_base_time_ns",
        right_on="_time_ns",
        direction="backward",
    )
    merged = merged.drop(columns=["_base_time_ns", "_time_ns"], errors="ignore")
    return merged


def build_canonical_v2(m5: pd.DataFrame) -> pd.DataFrame:
    print(f"[{ACTION}] step 1/4 — building v1 canonical features...", flush=True)
    t0 = _time.time()
    v1_feats = v1_builder.build_canonical_features(m5)
    print(f"[{ACTION}] v1 done in {_time.time() - t0:.1f}s, shape={v1_feats.shape}", flush=True)

    print(f"[{ACTION}] step 2/4 — computing D1 features...", flush=True)
    t0 = _time.time()
    d1_feats = compute_d1_features(m5)
    print(f"[{ACTION}] D1 done in {_time.time() - t0:.1f}s, "
          f"d1_rows={len(d1_feats):,}, cols={len(d1_feats.columns) - 2}", flush=True)

    print(f"[{ACTION}] step 3/4 — computing M15 features...", flush=True)
    t0 = _time.time()
    m15_feats = compute_m15_features(m5)
    print(f"[{ACTION}] M15 done in {_time.time() - t0:.1f}s, "
          f"m15_rows={len(m15_feats):,}, cols={len(m15_feats.columns) - 2}", flush=True)

    print(f"[{ACTION}] step 4/5 — merge_asof D1 + M15 onto v1...", flush=True)
    t0 = _time.time()
    out = merge_asof_features(v1_feats, d1_feats)
    out = merge_asof_features(out, m15_feats)
    print(f"[{ACTION}] merge done in {_time.time() - t0:.1f}s, shape after merge={out.shape}", flush=True)

    print(f"[{ACTION}] step 5/5 — computing SMC features (HH/HL state, BOS, CHOCH, sweep, premium/discount)...", flush=True)
    t0 = _time.time()
    smc_df = compute_smc_features(out)
    out = pd.concat([out.reset_index(drop=True), smc_df.reset_index(drop=True)], axis=1)
    print(f"[{ACTION}] SMC done in {_time.time() - t0:.1f}s, "
          f"added {len(SMC_FEATURE_NAMES)} features, final shape={out.shape}", flush=True)

    return out


def write_artifacts(
    out_path: Path = DEFAULT_OUT_PATH,
    *,
    m5_tape_root: Path = DEFAULT_M5_TAPE_ROOT,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{ACTION}] loading M5 tape from {m5_tape_root}", flush=True)
    m5 = v1_builder.load_m5_tape(m5_tape_root)
    print(f"[{ACTION}] M5 bars loaded: {len(m5):,}  range={m5['time'].min()} -> {m5['time'].max()}", flush=True)

    feats = build_canonical_v2(m5)

    # Inventory
    n_v1_basic = sum(1 for c in feats.columns if c.startswith("_v1_") and not c.startswith("_v1h"))
    n_v1h1 = sum(1 for c in feats.columns if c.startswith("_v1h1_"))
    n_v1h4 = sum(1 for c in feats.columns if c.startswith("_v1h4_"))
    n_phase = sum(1 for c in feats.columns if c.startswith("m5_phase_"))
    n_d1_v2 = sum(1 for c in feats.columns if c.startswith("d1_") and c.endswith(SUFFIX))
    n_m15_v2 = sum(1 for c in feats.columns if c.startswith("m15_") and c.endswith(SUFFIX))
    n_smc = sum(1 for c in feats.columns if c.startswith("smc_"))
    high_level_names = (
        "mid", "range", "body_pct", "wick_asym", "atr", "atr50", "atr_z",
        "std50", "ret_1", "ret_5", "ret_20", "roc20", "roc100", "rvol_20",
        "rvol_60", "ema20_slope", "ema100_slope", "pos_vs_ema200", "vol_ratio",
    )
    n_high_level = sum(1 for c in feats.columns if c in high_level_names)
    print(f"[{ACTION}] feature counts: _v1_*={n_v1_basic}, _v1h1_*={n_v1h1}, _v1h4_*={n_v1h4}, "
          f"m5_phase_*={n_phase}, high_level={n_high_level}, "
          f"d1_*{SUFFIX}={n_d1_v2}, m15_*{SUFFIX}={n_m15_v2}, smc_*={n_smc}", flush=True)
    print(f"[{ACTION}] total cols: {len(feats.columns)}", flush=True)

    feats.to_parquet(out_path, index=False)
    print(f"[{ACTION}] wrote {out_path} ({out_path.stat().st_size / 1024**2:.1f}MB)", flush=True)

    summary = {
        "layer_name": "CANONICAL_FEATURES_V2_SUMMARY",
        "action_v1": ACTION,
        "out_path_v1": str(out_path),
        "built_at_utc_v1": _utc_now(),
        "m5_tape_root_v1": str(m5_tape_root),
        "m5_bars_loaded_v1": int(len(m5)),
        "m5_date_range_v1": [str(m5["time"].min()), str(m5["time"].max())],
        "n_features_v1_basic_v1": n_v1_basic,
        "n_features_v1h1_v1": n_v1h1,
        "n_features_v1h4_v1": n_v1h4,
        "n_features_m5_phase_v1": n_phase,
        "n_features_high_level_v1": n_high_level,
        "n_features_d1_v2_v1": n_d1_v2,
        "n_features_m15_v2_v1": n_m15_v2,
        "total_columns_v1": int(len(feats.columns)),
        "v2_d1_feature_names_v1": [f"{n}{SUFFIX}" for n in D1_FEATURE_NAMES],
        "v2_m15_feature_names_v1": [f"{n}{SUFFIX}" for n in M15_FEATURE_NAMES],
        "research_only_v1": True,
    }
    summary_path = out_path.parent / "canonical_features_v2_summary.json"
    _write_json(summary_path, summary)
    print(f"[{ACTION}] summary at {summary_path}", flush=True)
    return {"out_path": str(out_path), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION}.")
    parser.add_argument("--out-path", type=str, default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--m5-root", type=str, default=str(DEFAULT_M5_TAPE_ROOT))
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()

    out_path = Path(args.out_path).expanduser().resolve()
    m5_tape_root = Path(args.m5_root).expanduser().resolve()
    result = write_artifacts(out_path=out_path, m5_tape_root=m5_tape_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
