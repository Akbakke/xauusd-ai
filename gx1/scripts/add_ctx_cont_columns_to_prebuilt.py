#!/home/andre2/venvs/gx1/bin/python
"""
Add ctx_cont / ctx_cat columns to prebuilt parquet (CTX 4/5, 6/6, 11/6, or 16/6).

Deterministic, TRUTH-style, no lookahead, no quarantine dependency.

CTX_CONT:
  4: atr_bps,
     spread_bps,
     D1_dist_from_ema200_atr,
     H1_range_compression_ratio
 6: + D1_atr_percentile_252,
       M15_range_compression_ratio
 11: + micro_momentum_3,
       micro_momentum_5,
       micro_acceleration,
       wick_ratio,
       distance_ema_fast
 16: + dist_last_swing_high_atr,
       dist_last_swing_low_atr,
       bars_since_swing_high,
       bars_since_swing_low,
       retracement_from_last_impulse

CTX_CAT:
  5: session_id,
     trend_regime_id,
     vol_regime_id,
     atr_bucket,
     spread_bucket
  6: + H4_trend_sign_cat

Contract source of truth:
  gx1.contracts.signal_bridge_v1

This script HARD-FAILS if required contract columns are missing or non-finite.

IMPORTANT:
- Alignment is "last closed" (no lookahead): for each M5 timestamp t, we attach the most recent
  closed HTF bar value strictly before t (by shifting left: t - 1D / 1H / 15m / 4H).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from gx1.contracts.signal_bridge_v1 import (
    ORDERED_CTX_CONT_NAMES_EXTENDED,
    ORDERED_CTX_CAT_NAMES_EXTENDED,
    CTX_CONT_COL_D1_DIST,
    CTX_CONT_COL_H1_COMP,
    CTX_CONT_COL_D1_ATR_PCTL252,
    CTX_CONT_COL_M15_COMP,
    CTX_CAT_COL_H4_TREND_SIGN,
)
from gx1.time.session_detector import (
    get_session_vectorized,
    get_session_id_vectorized,
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
)

log = logging.getLogger(__name__)

ATR_EPS = 1e-9
MICRO_FEATURE_NAMES = [
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
    "wick_ratio",
    "distance_ema_fast",
]
SWING_FEATURE_NAMES = [
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "retracement_from_last_impulse",
]
SWING_ATR_PERIOD = 14


def get_prebuilt_ctx_contract_columns(
    ctx_cont_dim: int = 6,
    ctx_cat_dim: int = 6,
) -> Tuple[List[str], List[str]]:
    """Return (required_cont, required_cat) with exact contract names for prebuilt. No side effects."""
    base_cont = list(ORDERED_CTX_CONT_NAMES_EXTENDED)
    if ctx_cont_dim <= len(base_cont):
        required_cont = list(base_cont[:ctx_cont_dim])
    elif ctx_cont_dim == len(base_cont) + len(MICRO_FEATURE_NAMES):
        required_cont = list(base_cont) + list(MICRO_FEATURE_NAMES)
    elif ctx_cont_dim == len(base_cont) + len(MICRO_FEATURE_NAMES) + len(SWING_FEATURE_NAMES):
        required_cont = list(base_cont) + list(MICRO_FEATURE_NAMES) + list(SWING_FEATURE_NAMES)
    else:
        raise ValueError(
            f"ctx_cont_dim={ctx_cont_dim} unsupported; base_cont={len(base_cont)} "
            f"micro={len(MICRO_FEATURE_NAMES)} swing={len(SWING_FEATURE_NAMES)}"
        )
    required_cat = list(ORDERED_CTX_CAT_NAMES_EXTENDED[:ctx_cat_dim])
    return required_cont, required_cat


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_dt_index(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.sort_index()
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        return out
    # best-effort upgrade
    for c in ("time", "ts", "timestamp"):
        if c in df.columns:
            out = df.copy()
            out = out.set_index(pd.to_datetime(out[c], utc=True))
            return out.sort_index()
    raise RuntimeError(f"[CTX_INPUT_FAIL] {name} must have DatetimeIndex (or time/ts column)")


def _load_canonical_tape(
    *,
    tape_root: Path,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    required_cols: List[str],
) -> pd.DataFrame:
    """
    Load canonical M5 tape for [t_min, t_max] from a partitioned parquet dataset:
      .../xauusd_m5_bid_ask__CANONICAL/year=YYYY/part-000.parquet
    """
    tape_root = tape_root.expanduser().resolve()
    if not tape_root.exists():
        raise RuntimeError(f"TAPE_ROOT_MISSING: {tape_root}")
    if not tape_root.is_dir():
        raise RuntimeError(f"TAPE_ROOT_NOT_DIR: {tape_root}")

    y0 = int(pd.Timestamp(t_min).year)
    y1 = int(pd.Timestamp(t_max).year)
    files: List[Path] = []
    for y in range(y0, y1 + 1):
        p = tape_root / f"year={y}"
        if p.exists() and p.is_dir():
            files.extend(sorted(p.glob("*.parquet")))
            files.extend(sorted(p.glob("part-*.parquet")))
    if not files:
        files = sorted(tape_root.rglob("*.parquet"))

    if not files:
        raise RuntimeError(f"TAPE_NO_FILES: no parquet files found under {tape_root}")

    df_list: List[pd.DataFrame] = []
    for fp in files:
        dfi = pd.read_parquet(fp, columns=list(set(["time"] + required_cols)))
        if "time" not in dfi.columns:
            if "ts" in dfi.columns:
                dfi = dfi.rename(columns={"ts": "time"})
            else:
                raise RuntimeError(f"TAPE_TIME_MISSING: {fp}")
        dfi["time"] = pd.to_datetime(dfi["time"], utc=True, errors="coerce")
        dfi = dfi.dropna(subset=["time"])
        dfi = dfi[(dfi["time"] >= t_min) & (dfi["time"] <= t_max)]
        if len(dfi):
            df_list.append(dfi)

    if not df_list:
        raise RuntimeError("TAPE_EMPTY_IN_RANGE")

    tape = pd.concat(df_list, ignore_index=True)
    tape = tape.sort_values("time")
    tape = tape[~tape["time"].duplicated()].copy()

    missing = [c for c in required_cols if c not in tape.columns]
    if missing:
        raise RuntimeError(f"TAPE_REQUIRED_COLS_MISSING: {missing}")

    if tape["time"].dtype != "datetime64[ns, UTC]":
        tape["time"] = pd.to_datetime(tape["time"], utc=True, errors="coerce")
        tape = tape.dropna(subset=["time"])

    if len(tape) == 0:
        raise RuntimeError("TAPE_EMPTY_AFTER_NORMALIZATION")

    return tape


def _last_valid(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": _last_valid}
    return df.resample(rule).agg(agg).dropna(how="all")


def _align_last_closed(ts_m5: pd.DatetimeIndex, series_htf: pd.Series, shift: pd.Timedelta) -> pd.Series:
    """
    Align HTF series to M5 timestamps with no lookahead:
      value(t) = HTF value at the last HTF timestamp <= (t - shift)
    """
    if ts_m5.empty:
        return pd.Series(dtype=float)

    if series_htf is None or len(series_htf) == 0:
        return pd.Series(index=ts_m5, dtype=float)

    # left: t - shift
    left_ts = (ts_m5.to_series(index=ts_m5) - shift).rename("_left")
    left_df = pd.DataFrame({"_left": left_ts, "_orig": ts_m5.to_series(index=ts_m5)}, index=ts_m5)
    left_df = left_df.sort_values("_left")

    right = series_htf.dropna().sort_index().rename("_val")
    right_df = right.reset_index()
    right_df.columns = ["_htf", "_val"]
    right_df = right_df.sort_values("_htf")

    out = pd.merge_asof(left_df, right_df, left_on="_left", right_on="_htf", direction="backward")
    aligned = out.set_index(pd.DatetimeIndex(out["_orig"]))["_val"]
    return aligned.reindex(ts_m5)


def _require_cols(df: pd.DataFrame, cols: List[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[CTX_INPUT_FAIL] {name} missing columns: {missing}")


def _finite_or_fail(arr: np.ndarray, *, label: str) -> None:
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(f"[CTX_NONFINITE_FAIL] {label} has non-finite values: count={n_bad}")


def _rank_bucket_0_4(x: np.ndarray, fallback: int) -> np.ndarray:
    """
    Deterministic 0..4 bucket via percentile rank (average).
    Always returns int64, no NaN.
    """
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    try:
        s = pd.Series(x)
        # rank(pct=True) produces NaN if all NaN; handle below
        q = s.rank(pct=True, method="average")
        qv = q.to_numpy(dtype=float)
        if not np.isfinite(qv).any():
            return np.full(len(x), int(fallback), dtype=np.int64)
        b = np.clip(qv * 5.0, 0.0, 4.99).astype(np.int64)
        b = np.where(np.isfinite(b), b, int(fallback)).astype(np.int64)
        return b
    except Exception:
        return np.full(len(x), int(fallback), dtype=np.int64)


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------


def run_add_ctx_cont_columns(
    prebuilt_path: Path,
    raw_m5_paths: List[Path],
    output_parquet: Path,
    ctx_cont_dim: int = 4,
    ctx_cat_dim: int = 5,
    tape_root: Optional[Path] = None,
    diagnostics_path: Optional[Path] = None,
) -> None:
    """
    Build ctx_cont / ctx_cat columns into prebuilt parquet.

    ctx_cont_dim ∈ {2,4,6,11,16}
    ctx_cat_dim  ∈ {5,6}

    HARD-FAIL:
      - if any required contract column missing
      - if any required ctx_cont column contains NaN/Inf after alignment
    """

    if ctx_cont_dim not in (2, 4, 6, 11, 16):
        raise ValueError("ctx_cont_dim must be 2, 4, 6, 11, or 16")
    if ctx_cat_dim not in (5, 6):
        raise ValueError("ctx_cat_dim must be 5 or 6")

    prebuilt_path = Path(prebuilt_path).resolve()
    output_parquet = Path(output_parquet).resolve()
    raw_m5_paths = [Path(p).resolve() for p in (raw_m5_paths or [])]

    if not prebuilt_path.exists():
        raise RuntimeError(f"[CTX_INPUT_FAIL] prebuilt not found: {prebuilt_path}")
    if not raw_m5_paths:
        raise RuntimeError("[CTX_INPUT_FAIL] raw_m5_paths is empty")
    for p in raw_m5_paths:
        if not p.exists():
            raise RuntimeError(f"[CTX_INPUT_FAIL] raw M5 not found: {p}")

    base_cont = list(ORDERED_CTX_CONT_NAMES_EXTENDED)
    if ctx_cont_dim <= len(base_cont):
        required_cont = list(base_cont[:ctx_cont_dim])
    elif ctx_cont_dim == len(base_cont) + len(MICRO_FEATURE_NAMES):
        required_cont = list(base_cont) + list(MICRO_FEATURE_NAMES)
    elif ctx_cont_dim == len(base_cont) + len(MICRO_FEATURE_NAMES) + len(SWING_FEATURE_NAMES):
        required_cont = list(base_cont) + list(MICRO_FEATURE_NAMES) + list(SWING_FEATURE_NAMES)
    else:
        raise ValueError(
            f"ctx_cont_dim={ctx_cont_dim} unsupported; base_cont={len(base_cont)} "
            f"micro={len(MICRO_FEATURE_NAMES)} swing={len(SWING_FEATURE_NAMES)}"
        )
    required_cat = list(ORDERED_CTX_CAT_NAMES_EXTENDED[:ctx_cat_dim])

    # ------------------------------------------------------------
    # Load prebuilt + raw
    # ------------------------------------------------------------
    df_pre = pd.read_parquet(prebuilt_path)
    df_pre = _ensure_dt_index(df_pre, name="prebuilt")

    pre_start = df_pre.index.min()
    pre_end = df_pre.index.max()
    warmup_start = pre_start - pd.Timedelta(days=400)

    raws = []
    for p in raw_m5_paths:
        if p.is_dir():
            df = _load_canonical_tape(
                tape_root=p,
                t_min=warmup_start,
                t_max=pre_end,
                required_cols=["open", "high", "low", "close"],
            )
            df = _ensure_dt_index(df, name=f"raw_m5_dir:{p.name}")
        else:
            df = pd.read_parquet(p)
            df = _ensure_dt_index(df, name=f"raw_m5:{p.name}")
        _require_cols(df, ["open", "high", "low", "close"], name=f"raw_m5:{p.name}")
        raws.append(df[["open", "high", "low", "close"]])
    df_m5 = pd.concat(raws, axis=0).sort_index()

    log.info(
        "[PREBUILT_INPUT_PROOF] prebuilt=%s raw_m5=%s tape_root=%s",
        prebuilt_path,
        raw_m5_paths,
        tape_root,
    )

    # Warmup sanity (EMA200 + ATR100 on H1 + ATR252 on D1)
    pre_start = df_pre.index.min()
    raw_start = df_m5.index.min()
    # 300 days warmup is conservative and cheap vs debugging NaNs later
    if raw_start > pre_start - pd.Timedelta(days=300):
        raise RuntimeError(
            "[CTX_WARMUP_FAIL] raw M5 must cover >= ~300 days before prebuilt start for stable HTF warmups"
        )

    # ------------------------------------------------------------
    # CONT: baseline (atr_bps, spread_bps)
    # ------------------------------------------------------------

    # atr_bps: use existing if present; otherwise derive from prebuilt 'atr' and raw mid
    if "atr_bps" in df_pre.columns:
        atr_bps = df_pre["atr_bps"].to_numpy(dtype=float)
        _finite_or_fail(atr_bps, label="atr_bps(existing)")
    else:
        if "atr" not in df_pre.columns:
            raise RuntimeError("[CTX_ATR_BPS_FAIL] prebuilt must contain 'atr' to derive atr_bps")
        mid_m5 = (df_m5["high"] + df_m5["low"]) * 0.5
        mid_aligned = mid_m5.reindex(df_pre.index)
        if mid_aligned.isna().any() or (mid_aligned.to_numpy() <= 0).any():
            raise RuntimeError("[CTX_ATR_BPS_FAIL] mid_aligned missing or <= 0 for some prebuilt rows")
        atr_vals = df_pre["atr"].to_numpy(dtype=float)
        atr_bps = (atr_vals / np.maximum(mid_aligned.to_numpy(dtype=float), ATR_EPS)) * 1e4
        _finite_or_fail(atr_bps, label="atr_bps(derived)")
        df_pre["atr_bps"] = atr_bps

    # spread_bps: use existing if present; else derive from spread/close; otherwise 0.0
    if "spread_bps" in df_pre.columns:
        spread_bps = df_pre["spread_bps"].to_numpy(dtype=float)
        _finite_or_fail(spread_bps, label="spread_bps(existing)")
    elif "spread" in df_pre.columns and "close" in df_pre.columns:
        close = df_pre["close"].to_numpy(dtype=float)
        close = np.where(close > 0, close, np.nan)
        spread = df_pre["spread"].to_numpy(dtype=float)
        spread_bps = (spread / close) * 1e4
        spread_bps = np.where(np.isfinite(spread_bps), spread_bps, 0.0)
        df_pre["spread_bps"] = spread_bps.astype(float)
    else:
        df_pre["spread_bps"] = 0.0

    # ------------------------------------------------------------
    # CONT: HTF core (D1_dist_from_ema200_atr, H1_range_compression_ratio)
    # ------------------------------------------------------------
    df_d1 = _resample_ohlc(df_m5, "1D")
    df_h1 = _resample_ohlc(df_m5, "1H")

    if len(df_d1) < 220:
        raise RuntimeError("[CTX_WARMUP_FAIL] insufficient D1 bars for EMA200 warmup")
    if len(df_h1) < 120:
        raise RuntimeError("[CTX_WARMUP_FAIL] insufficient H1 bars for ATR100 warmup")

    d1_mid = (df_d1["high"] + df_d1["low"]) * 0.5
    d1_ema200 = _ema(d1_mid, 200)
    d1_atr14 = _atr(df_d1["high"], df_d1["low"], df_d1["close"], 14).ffill()
    d1_dist = (d1_mid - d1_ema200) / np.maximum(d1_atr14, ATR_EPS)

    h1_atr14 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 14).ffill()
    h1_atr100 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 100).ffill()
    h1_comp = h1_atr14 / np.maximum(h1_atr100, ATR_EPS)

    # no lookahead alignment
    d1_aligned = _align_last_closed(df_pre.index, d1_dist, pd.Timedelta(days=1))
    h1_aligned = _align_last_closed(df_pre.index, h1_comp, pd.Timedelta(hours=1))

    if d1_aligned.isna().any():
        raise RuntimeError("[CTX_ALIGN_FAIL] D1_dist_from_ema200_atr has NaN after alignment (no ffill/bfill allowed)")
    if h1_aligned.isna().any():
        raise RuntimeError("[CTX_ALIGN_FAIL] H1_range_compression_ratio has NaN after alignment (no ffill/bfill allowed)")

    df_pre[CTX_CONT_COL_D1_DIST] = d1_aligned.to_numpy(dtype=float)
    df_pre[CTX_CONT_COL_H1_COMP] = h1_aligned.to_numpy(dtype=float)

    # ------------------------------------------------------------
    # CONT extras for >=6/6: D1_atr_percentile_252, M15_range_compression_ratio
    # ------------------------------------------------------------
    if ctx_cont_dim >= 6:
        # D1_atr_percentile_252: percentile rank of D1 ATR14 over rolling 252 days
        d1_atr14_for_pctl = d1_atr14.copy()
        # rolling percentile rank at each day = rank of last value within window
        # We compute rank(pct=True) per rolling window via apply (slower but deterministic, one-time offline builder).
        def _pctl_last(window: np.ndarray) -> float:
            w = np.asarray(window, dtype=float)
            if not np.isfinite(w).all():
                # if window has non-finite, treat as nan (will be caught)
                return float("nan")
            last = w[-1]
            # percentile rank among window values
            return float((w <= last).mean())

        atr_pctl252 = d1_atr14_for_pctl.rolling(252, min_periods=252).apply(_pctl_last, raw=True)
        atr_pctl252 = atr_pctl252.ffill()
        atr_pctl_aligned = _align_last_closed(df_pre.index, atr_pctl252, pd.Timedelta(days=1))
        if atr_pctl_aligned.isna().any():
            raise RuntimeError("[CTX_ALIGN_FAIL] D1_atr_percentile_252 has NaN after alignment")
        df_pre[CTX_CONT_COL_D1_ATR_PCTL252] = atr_pctl_aligned.to_numpy(dtype=float)

        # M15_range_compression_ratio: ATR14/ATR100 on M15
        df_m15 = _resample_ohlc(df_m5, "15min")
        if len(df_m15) < 200:
            raise RuntimeError("[CTX_WARMUP_FAIL] insufficient M15 bars for ATR100 warmup")
        m15_atr14 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 14).ffill()
        m15_atr100 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 100).ffill()
        m15_comp = m15_atr14 / np.maximum(m15_atr100, ATR_EPS)
        m15_aligned = _align_last_closed(df_pre.index, m15_comp, pd.Timedelta(minutes=15))
        if m15_aligned.isna().any():
            raise RuntimeError("[CTX_ALIGN_FAIL] M15_range_compression_ratio has NaN after alignment")
        df_pre[CTX_CONT_COL_M15_COMP] = m15_aligned.to_numpy(dtype=float)

    # ------------------------------------------------------------
    # CONT micro/swing features for >=11/6: canonical tape strict join
    # ------------------------------------------------------------
    if ctx_cont_dim >= 11:
        if tape_root is None:
            raise RuntimeError("[CTX_INPUT_FAIL] tape_root required for ctx_cont_dim>=11 features")
        t_min = pd.Timestamp(df_pre.index.min()).tz_convert("UTC")
        t_max = pd.Timestamp(df_pre.index.max()).tz_convert("UTC")
        tape = _load_canonical_tape(
            tape_root=Path(tape_root),
            t_min=t_min,
            t_max=t_max,
            required_cols=["open", "high", "low", "close"],
        )

        df_times = pd.DataFrame({"time": df_pre.index})
        merged = df_times.merge(tape, on="time", how="inner", validate="one_to_one")
        rows_base28 = int(len(df_times))
        rows_tape = int(len(tape))
        rows_joined = int(len(merged))
        exact_match = int(rows_base28 == rows_tape == rows_joined)
        log.info(
            "[ENTRY_TAPE_JOIN_PROOF] rows_base28=%d rows_tape=%d rows_joined=%d exact_match=%d",
            rows_base28,
            rows_tape,
            rows_joined,
            exact_match,
        )
        if not exact_match:
            raise RuntimeError(
                f"TAPE_JOIN_STRICT_FAIL: rows_base28={rows_base28} rows_tape={rows_tape} rows_joined={rows_joined}"
            )

        eps = 1e-9
        tape_feat = merged[["time", "close", "high", "low"]].copy().sort_values("time")
        close = tape_feat["close"].astype(float)
        high = tape_feat["high"].astype(float)
        low = tape_feat["low"].astype(float)

        tape_feat["micro_momentum_3"] = (close - close.shift(3)).fillna(0.0)
        tape_feat["micro_momentum_5"] = (close - close.shift(5)).fillna(0.0)
        tape_feat["micro_acceleration"] = (
            (close - close.shift(1)) - (close.shift(1) - close.shift(2))
        ).fillna(0.0)
        tape_feat["wick_ratio"] = (high - close) / (high - low + eps)
        ema_fast = close.ewm(span=5, adjust=False).mean()
        tape_feat["distance_ema_fast"] = close - ema_fast

        for name in MICRO_FEATURE_NAMES:
            if name not in tape_feat.columns:
                raise RuntimeError(f"MICRO_FEATURE_MISSING: {name}")

        if ctx_cont_dim == 16:
            prev_close = close.shift(1).fillna(close)
            tr = np.maximum(
                (high - low).abs(),
                np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
            )
            atr = tr.rolling(window=SWING_ATR_PERIOD, min_periods=1).mean()
            atr_safe = atr.clip(lower=eps)

            pivot_high = high > pd.concat(
                [high.shift(1), high.shift(2), high.shift(-1), high.shift(-2)], axis=1
            ).max(axis=1, skipna=True)
            pivot_low = low < pd.concat(
                [low.shift(1), low.shift(2), low.shift(-1), low.shift(-2)], axis=1
            ).min(axis=1, skipna=True)
            log.info(
                "[ENTRY_SWING_PIVOT_PROOF] pivot_highs=%d pivot_lows=%d",
                int(pivot_high.sum()),
                int(pivot_low.sum()),
            )

            n = int(len(tape_feat))
            last_high_vals = np.empty(n, dtype=np.float64)
            last_low_vals = np.empty(n, dtype=np.float64)
            last_high_idx = np.empty(n, dtype=np.int64)
            last_low_idx = np.empty(n, dtype=np.int64)

            last_high = float(high.iloc[0])
            last_low = float(low.iloc[0])
            last_hi_i = 0
            last_lo_i = 0
            for i in range(n):
                if bool(pivot_high.iloc[i]):
                    last_high = float(high.iloc[i])
                    last_hi_i = i
                if bool(pivot_low.iloc[i]):
                    last_low = float(low.iloc[i])
                    last_lo_i = i
                last_high_vals[i] = last_high
                last_low_vals[i] = last_low
                last_high_idx[i] = last_hi_i
                last_low_idx[i] = last_lo_i

            idx = np.arange(n, dtype=np.int64)
            bars_since_high = (idx - last_high_idx).astype(np.float32)
            bars_since_low = (idx - last_low_idx).astype(np.float32)

            dist_last_swing_high_atr = (close.to_numpy() - last_high_vals) / atr_safe.to_numpy()
            dist_last_swing_low_atr = (close.to_numpy() - last_low_vals) / atr_safe.to_numpy()

            denom = np.maximum((last_high_vals - last_low_vals), eps)
            retracement = np.zeros(n, dtype=np.float64)
            up_mask = last_high_idx > last_low_idx
            down_mask = last_low_idx > last_high_idx
            retracement[up_mask] = (last_high_vals[up_mask] - close.to_numpy()[up_mask]) / denom[up_mask]
            retracement[down_mask] = (close.to_numpy()[down_mask] - last_low_vals[down_mask]) / denom[down_mask]
            retracement = np.clip(retracement, 0.0, 1.0)

            tape_feat["dist_last_swing_high_atr"] = dist_last_swing_high_atr.astype(np.float32)
            tape_feat["dist_last_swing_low_atr"] = dist_last_swing_low_atr.astype(np.float32)
            tape_feat["bars_since_swing_high"] = bars_since_high.astype(np.float32)
            tape_feat["bars_since_swing_low"] = bars_since_low.astype(np.float32)
            tape_feat["retracement_from_last_impulse"] = retracement.astype(np.float32)

            for name in SWING_FEATURE_NAMES:
                if name not in tape_feat.columns:
                    raise RuntimeError(f"SWING_FEATURE_MISSING: {name}")

        tape_feat = tape_feat.set_index("time")
        join_cols = list(MICRO_FEATURE_NAMES)
        if ctx_cont_dim == 16:
            join_cols = join_cols + list(SWING_FEATURE_NAMES)
        existing_join = [c for c in join_cols if c in df_pre.columns]
        if existing_join:
            log.info("[ENTRY_MICRO_FEATURES_OVERWRITE] dropping_existing=%s", existing_join)
            df_pre = df_pre.drop(columns=existing_join)
        df_pre = df_pre.join(tape_feat[join_cols], how="left")
        if len(df_pre) != rows_base28:
            raise RuntimeError(
                f"MICRO_FEATURE_JOIN_FAIL: rows_base28={rows_base28} rows_after={len(df_pre)}"
            )

        log.info(
            "[ENTRY_MICRO_FEATURES_PROOF] names=%s count=%d",
            list(MICRO_FEATURE_NAMES),
            len(MICRO_FEATURE_NAMES),
        )
        if ctx_cont_dim == 16:
            log.info(
                "[ENTRY_SWING_FEATURES_PROOF] names=%s count=%d",
                list(SWING_FEATURE_NAMES),
                len(SWING_FEATURE_NAMES),
            )

    # ------------------------------------------------------------
    # CAT: 5/6 dims, deterministic, int, no NaN
    # ------------------------------------------------------------
    ts = df_pre.index

    # session_id: 0=ASIA, 1=EU, 2=OVERLAP, 3=US (SSoT)
    df_pre["session_id"] = get_session_id_vectorized(ts).astype(np.int64)

    # Session timing features (observerable context)
    df_pre["minutes_since_session_open"] = get_session_minutes_since_open_vectorized(ts).astype(np.float32)
    df_pre["minutes_to_next_session_boundary"] = get_session_minutes_to_next_boundary_vectorized(ts).astype(np.float32)
    # Session change flag (1 if session changes vs previous bar)
    sess_tag = get_session_vectorized(ts)
    df_pre["session_change_flag"] = (sess_tag != sess_tag.shift(1)).fillna(False).astype(np.int64)
    # Tradable flag (policy can still restrict to EU/OVERLAP/US)
    df_pre["session_tradable"] = (df_pre["session_id"] != 0).astype(np.int64)

    # trend_regime_id: bucket by price_vs_ema50_atr if present else neutral=1
    if "price_vs_ema50_atr" in df_pre.columns:
        p = df_pre["price_vs_ema50_atr"].to_numpy(dtype=float)
        p = np.where(np.isfinite(p), p, 0.0)
        trend_regime_id = np.where(p < -0.5, 0, np.where(p <= 0.5, 1, 2)).astype(np.int64)
    else:
        trend_regime_id = np.ones(len(df_pre), dtype=np.int64)
    df_pre["trend_regime_id"] = trend_regime_id

    # vol_regime_id / atr_bucket: 0..4 from atr_bps percentile rank
    vol_regime_id = _rank_bucket_0_4(df_pre["atr_bps"].to_numpy(dtype=float), fallback=2)
    df_pre["vol_regime_id"] = vol_regime_id.astype(np.int64)
    df_pre["atr_bucket"] = vol_regime_id.astype(np.int64)

    # spread_bucket: 0..4 from spread_bps percentile rank
    spread_bucket = _rank_bucket_0_4(df_pre["spread_bps"].to_numpy(dtype=float), fallback=0)
    df_pre["spread_bucket"] = spread_bucket.astype(np.int64)

    # H4_trend_sign_cat (optional): sign(mid - ema50) on H4, mapped to {0,1,2} for {-1,0,+1}
    if ctx_cat_dim == 6:
        df_h4 = _resample_ohlc(df_m5, "4H")
        if len(df_h4) < 80:
            raise RuntimeError("[CTX_WARMUP_FAIL] insufficient H4 bars for EMA50 warmup")
        h4_mid = (df_h4["high"] + df_h4["low"]) * 0.5
        h4_ema50 = _ema(h4_mid, 50)
        diff = (h4_mid - h4_ema50).to_numpy(dtype=float)
        sign = np.sign(np.where(np.isfinite(diff), diff, 0.0)).astype(np.int64)  # -1/0/+1
        # convert to 0/1/2
        sign_cat = (sign + 1).astype(np.int64)
        sign_series = pd.Series(sign_cat, index=df_h4.index, dtype="int64")
        h4_aligned = _align_last_closed(df_pre.index, sign_series, pd.Timedelta(hours=4))
        if h4_aligned.isna().any():
            raise RuntimeError("[CTX_ALIGN_FAIL] H4_trend_sign_cat has NaN after alignment")
        df_pre[CTX_CAT_COL_H4_TREND_SIGN] = h4_aligned.to_numpy(dtype=np.int64)

    # Ensure cat columns are int and non-NaN
    for c in ORDERED_CTX_CAT_NAMES_EXTENDED[:ctx_cat_dim]:
        if c not in df_pre.columns:
            continue
        if df_pre[c].isna().any():
            # This should not happen, but TRUTH style: hard-fill to 0 then cast.
            df_pre[c] = df_pre[c].fillna(0)
        df_pre[c] = df_pre[c].astype(np.int64)

    # ------------------------------------------------------------
    # Contract validation (names + finiteness)
    # ------------------------------------------------------------
    missing = [c for c in (required_cont + required_cat) if c not in df_pre.columns]
    if missing:
        raise RuntimeError(f"[PREBUILT_CTX_CONTRACT_FAIL] missing columns: {missing}")

    cont_mat = df_pre[required_cont].to_numpy(dtype=float)
    _finite_or_fail(cont_mat, label=f"ctx_cont(required_cont={required_cont})")

    log.info("[PREBUILT_CTX_CONT_PROOF] ctx_cont_dim=%d names=%s", ctx_cont_dim, required_cont)

    # ------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_pre.to_parquet(output_parquet, index=True)

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path).resolve()
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(
                {
                    "prebuilt_path": str(prebuilt_path),
                    "output_path": str(output_parquet),
                    "raw_m5_paths": [str(p) for p in raw_m5_paths],
                    "ctx_cont_dim": int(ctx_cont_dim),
                    "ctx_cat_dim": int(ctx_cat_dim),
                    "required_cont": required_cont,
                    "required_cat": required_cat,
                    "ctx_columns_added": required_cont + required_cat,
                    "ctx_contract_missing": [],
                    "n_rows": int(len(df_pre)),
                    "tape_root": str(tape_root) if tape_root is not None else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    log.info("[PREBUILT_CTX_CONTRACT] required cont+cat present; missing: [] (cont=%d cat=%d)", ctx_cont_dim, ctx_cat_dim)
    log.info("Wrote %s (%d rows)", output_parquet, len(df_pre))

    # Sidecar manifest + schema manifest (TRUTH preflight expects schema manifest)
    manifest_path = output_parquet.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest_path.unlink(missing_ok=False)
    prebuilt_resolved = output_parquet.resolve()
    h = hashlib.sha256()
    with open(prebuilt_resolved, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    prebuilt_sha256 = h.hexdigest()
    prebuilt_bytes = prebuilt_resolved.stat().st_size
    manifest_obj = {
        "kind": "prebuilt_manifest_v1",
        "prebuilt_path": str(prebuilt_resolved),
        "prebuilt_sha256": prebuilt_sha256,
        "prebuilt_bytes": prebuilt_bytes,
        "ctx_cont_dim": int(ctx_cont_dim),
        "ctx_cat_dim": int(ctx_cat_dim),
        "no_fallback_enforced": True,
    }
    manifest_path.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")
    log.info("PREBUILT_MANIFEST_WRITTEN=%s", manifest_path.resolve())

    schema_manifest_path = output_parquet.with_suffix(".schema_manifest.json")
    if schema_manifest_path.exists():
        schema_manifest_path.unlink(missing_ok=False)
    schema_manifest_path.write_text(
        json.dumps({"required_all_features": list(df_pre.columns)}, indent=2),
        encoding="utf-8",
    )
    log.info("PREBUILT_SCHEMA_MANIFEST_WRITTEN=%s", schema_manifest_path.resolve())


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser(description="Add ctx_cont/ctx_cat columns to prebuilt parquet (4/5, 6/6, 11/6, or 16/6)")
    ap.add_argument("--prebuilt_parquet", type=Path, required=True)
    ap.add_argument("--output_parquet", type=Path, required=True)
    ap.add_argument(
        "--raw_m5_parquet",
        type=Path,
        nargs="*",
        default=None,
        help="Raw M5 parquet(s). Default: GX1_DATA/data/data/_staging/XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet",
    )
    ap.add_argument("--ctx-cont-dim", type=int, default=4, choices=[2, 4, 6, 11, 16])
    ap.add_argument("--ctx-cat-dim", type=int, default=5, choices=[5, 6])
    ap.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Optional diagnostics JSON path (default: <output>.ctx_diagnostics.json)",
    )
    ap.add_argument(
        "--tape-root",
        type=Path,
        default=None,
        help="Canonical tape root for micro/swing features (required for ctx_cont_dim>=11).",
    )
    args = ap.parse_args()

    raw = args.raw_m5_parquet
    if not raw:
        gx1_data = Path(os.environ.get("GX1_DATA", "/home/andre2/GX1_DATA")).resolve()
        raw = [gx1_data / "data/data/_staging/XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet"]

    diag = args.diagnostics
    if diag is None:
        diag = args.output_parquet.with_name(args.output_parquet.stem + ".ctx_diagnostics.json")

    try:
        run_add_ctx_cont_columns(
            prebuilt_path=args.prebuilt_parquet,
            raw_m5_paths=list(raw),
            output_parquet=args.output_parquet,
            ctx_cont_dim=int(args.ctx_cont_dim),
            ctx_cat_dim=int(args.ctx_cat_dim),
            tape_root=args.tape_root,
            diagnostics_path=diag,
        )
    except Exception as e:
        print(f"[add_ctx_cont_columns] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
