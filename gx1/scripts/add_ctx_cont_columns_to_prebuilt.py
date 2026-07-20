#!/home/andre2/venvs/gx1/bin/python
"""
Add the causal ctx_cont source prefix and exact model-native ctx_cat fields.

Deterministic, TRUTH-style, no lookahead, no quarantine dependency.

CTX_CONT has one exact prebuilt surface: six causal source-prefix fields,
five canonical micro fields and five canonical swing fields. Five session
fields are recomputed from UTC time in the same output. Alternate dimensions
are retired and fail closed.

CTX_CAT (exact active five-field order): session_id, vol_regime_id,
atr_bucket, spread_bucket, H4_trend_sign_cat.  The retired categorical
trend_regime_id bucket is not an alternate output mode.

Contract source of truth:
  gx1.contracts.entry_model_native_signal_v1

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

from gx1.features.model_native_market_context_v1 import (
    derive_observed_spread_bps,
)

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_MICRO_FIELDS,
    MODEL_NATIVE_CTX_CONT_SESSION_FIELDS,
    MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS,
    MODEL_NATIVE_CTX_CONT_SWING_FIELDS,
    MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS,
    MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS,
)
from gx1.features.micro_structure_v1 import compute_micro_structure_features
from gx1.features.swing_structure_v1 import (
    SWING_ATR_PERIOD_V1,
    SWING_LOOKBACK_V1,
    compute_swing_structure_features,
)
from gx1.time.session_detector import (
    get_session_id_vectorized,
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
)

log = logging.getLogger(__name__)

ATR_EPS = 1e-9
CTX_CONT_COL_D1_DIST = MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS[2]
CTX_CONT_COL_H1_COMP = MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS[3]
CTX_CONT_COL_D1_ATR_PCTL252 = MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS[4]
CTX_CONT_COL_M15_COMP = MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS[5]
CTX_CAT_COL_H4_TREND_SIGN = MODEL_NATIVE_CTX_CAT_FIELDS[-1]


def get_prebuilt_ctx_contract_columns() -> Tuple[List[str], List[str]]:
    """Return (required_cont, required_cat) with exact contract names for prebuilt. No side effects."""
    return list(MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS), list(
        MODEL_NATIVE_CTX_CAT_FIELDS
    )


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
    raise RuntimeError(
        f"[CTX_INPUT_FAIL] {name} must have DatetimeIndex (or time/ts column)"
    )


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


def _align_last_closed(
    ts_m5: pd.DatetimeIndex, series_htf: pd.Series, shift: pd.Timedelta
) -> pd.Series:
    """
    Align HTF series to M5 timestamps with no lookahead:
      value(t) = HTF value at the last HTF timestamp <= (t + 5min - shift)
    """
    if ts_m5.empty:
        return pd.Series(dtype=float)

    if series_htf is None or len(series_htf) == 0:
        return pd.Series(index=ts_m5, dtype=float)

    # M5 timestamps label the bar start; the decision observes its close at t+5min.
    left_ts = (ts_m5.to_series(index=ts_m5) + pd.Timedelta(minutes=5) - shift).rename(
        "_left"
    )
    left_df = pd.DataFrame(
        {"_left": left_ts, "_orig": ts_m5.to_series(index=ts_m5)}, index=ts_m5
    )
    left_df = left_df.sort_values("_left")

    right = series_htf.dropna().sort_index().rename("_val")
    right_df = right.reset_index()
    right_df.columns = ["_htf", "_val"]
    right_df = right_df.sort_values("_htf")

    out = pd.merge_asof(
        left_df, right_df, left_on="_left", right_on="_htf", direction="backward"
    )
    aligned = out.set_index(pd.DatetimeIndex(out["_orig"]))["_val"]
    return aligned.reindex(ts_m5)


def _require_cols(df: pd.DataFrame, cols: List[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[CTX_INPUT_FAIL] {name} missing columns: {missing}")


def _finite_or_fail(arr: np.ndarray, *, label: str) -> None:
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(
            f"[CTX_NONFINITE_FAIL] {label} has non-finite values: count={n_bad}"
        )


def _derive_spread_bps_from_available(df: pd.DataFrame) -> np.ndarray:
    """Backward import name for the strict shared model-native primitive."""

    return derive_observed_spread_bps(df)


def _rank_bucket_0_4(x: np.ndarray) -> np.ndarray:
    """Return deterministic finite 0..4 percentile-rank buckets or fail."""

    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise RuntimeError(f"CTX_RANK_BUCKET_SHAPE_INVALID: {x.shape}")
    if not np.isfinite(x).all():
        raise RuntimeError("CTX_RANK_BUCKET_SOURCE_NONFINITE")
    qv = pd.Series(x).rank(pct=True, method="average").to_numpy(dtype=float)
    if not np.isfinite(qv).all():
        raise RuntimeError("CTX_RANK_BUCKET_OUTPUT_NONFINITE")
    return np.clip(qv * 5.0, 0.0, 4.99).astype(np.int64)


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------


def run_add_ctx_cont_columns(
    prebuilt_path: Path,
    raw_m5_paths: List[Path],
    output_parquet: Path,
    tape_root: Optional[Path] = None,
    diagnostics_path: Optional[Path] = None,
) -> None:
    """
    Build ctx_cont / ctx_cat columns into prebuilt parquet.

    The output has one exact 16-field prebuilt surface plus the five session
    fields, and one exact five-field categorical surface.

    HARD-FAIL:
      - if any required contract column missing
      - if any required ctx_cont column contains NaN/Inf after alignment
    """

    required_cont, required_cat = get_prebuilt_ctx_contract_columns()
    prebuilt_ctx_cont_dim = len(required_cont)
    ctx_cat_dim = len(required_cat)

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

    # ------------------------------------------------------------
    # Load prebuilt + raw
    # ------------------------------------------------------------
    df_pre = pd.read_parquet(prebuilt_path)
    df_pre = _ensure_dt_index(df_pre, name="prebuilt")

    pre_start = df_pre.index.min()
    pre_end = df_pre.index.max()
    warmup_start = pre_start - pd.Timedelta(days=400)

    # volume is carried through because the REGIME_V4 block requires the exact
    # raw M5 OHLCV surface; _resample_ohlc aggregates an explicit OHLC dict, so
    # the extra column is inert for the HTF paths.
    _raw_m5_required = ["open", "high", "low", "close", "volume"]
    raws = []
    for p in raw_m5_paths:
        if p.is_dir():
            df = _load_canonical_tape(
                tape_root=p,
                t_min=warmup_start,
                t_max=pre_end,
                required_cols=_raw_m5_required,
            )
            df = _ensure_dt_index(df, name=f"raw_m5_dir:{p.name}")
        else:
            df = pd.read_parquet(p)
            df = _ensure_dt_index(df, name=f"raw_m5:{p.name}")
        _require_cols(df, _raw_m5_required, name=f"raw_m5:{p.name}")
        raws.append(df[_raw_m5_required])
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
            raise RuntimeError(
                "[CTX_ATR_BPS_FAIL] prebuilt must contain 'atr' to derive atr_bps"
            )
        mid_m5 = (df_m5["high"] + df_m5["low"]) * 0.5
        mid_aligned = mid_m5.reindex(df_pre.index)
        if mid_aligned.isna().any() or (mid_aligned.to_numpy() <= 0).any():
            raise RuntimeError(
                "[CTX_ATR_BPS_FAIL] mid_aligned missing or <= 0 for some prebuilt rows"
            )
        atr_vals = df_pre["atr"].to_numpy(dtype=float)
        atr_bps = (
            atr_vals / np.maximum(mid_aligned.to_numpy(dtype=float), ATR_EPS)
        ) * 1e4
        _finite_or_fail(atr_bps, label="atr_bps(derived)")
        df_pre["atr_bps"] = atr_bps

    # spread_bps: use existing if present; else derive from bid/ask close exactly like
    # live ctx augmentation; else derive from spread/close; otherwise 0.0.
    df_pre["spread_bps"] = _derive_spread_bps_from_available(df_pre)

    # ------------------------------------------------------------
    # CONT: HTF core (D1_dist_from_ema200_atr, H1_range_compression_ratio)
    # ------------------------------------------------------------
    df_d1 = _resample_ohlc(df_m5, "1D")
    df_h1 = _resample_ohlc(df_m5, "1h")

    if len(df_d1) < 220:
        raise RuntimeError("[CTX_WARMUP_FAIL] insufficient D1 bars for EMA200 warmup")
    if len(df_h1) < 120:
        raise RuntimeError("[CTX_WARMUP_FAIL] insufficient H1 bars for ATR100 warmup")

    d1_mid = (df_d1["high"] + df_d1["low"]) * 0.5
    d1_ema200 = _ema(d1_mid, 200)
    d1_atr14 = _atr(df_d1["high"], df_d1["low"], df_d1["close"], 14).ffill()
    d1_dist = (d1_mid - d1_ema200) / np.maximum(d1_atr14, ATR_EPS)
    d1_dist.iloc[:219] = np.nan

    h1_atr14 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 14).ffill()
    h1_atr100 = _atr(df_h1["high"], df_h1["low"], df_h1["close"], 100).ffill()
    h1_comp = h1_atr14 / np.maximum(h1_atr100, ATR_EPS)
    h1_comp.iloc[:119] = np.nan

    # no lookahead alignment
    d1_aligned = _align_last_closed(df_pre.index, d1_dist, pd.Timedelta(days=1))
    h1_aligned = _align_last_closed(df_pre.index, h1_comp, pd.Timedelta(hours=1))

    if d1_aligned.isna().any():
        raise RuntimeError(
            "[CTX_ALIGN_FAIL] D1_dist_from_ema200_atr has NaN after alignment (no ffill/bfill allowed)"
        )
    if h1_aligned.isna().any():
        raise RuntimeError(
            "[CTX_ALIGN_FAIL] H1_range_compression_ratio has NaN after alignment (no ffill/bfill allowed)"
        )

    df_pre[CTX_CONT_COL_D1_DIST] = d1_aligned.to_numpy(dtype=float)
    df_pre[CTX_CONT_COL_H1_COMP] = h1_aligned.to_numpy(dtype=float)

    # ------------------------------------------------------------
    # Exact D1/M15 context: D1 ATR percentile and M15 compression.
    # ------------------------------------------------------------
    d1_atr14_for_pctl = d1_atr14.copy()

    def _pctl_last(window: np.ndarray) -> float:
        w = np.asarray(window, dtype=float)
        if not np.isfinite(w).all():
            return float("nan")
        return float((w <= w[-1]).mean())

    atr_pctl252 = d1_atr14_for_pctl.rolling(252, min_periods=252).apply(
        _pctl_last,
        raw=True,
    )
    atr_pctl252 = atr_pctl252.ffill()
    atr_pctl252.iloc[:269] = np.nan
    atr_pctl_aligned = _align_last_closed(
        df_pre.index,
        atr_pctl252,
        pd.Timedelta(days=1),
    )
    if atr_pctl_aligned.isna().any():
        raise RuntimeError(
            "[CTX_ALIGN_FAIL] D1_atr_percentile_252 has NaN after alignment"
        )
    df_pre[CTX_CONT_COL_D1_ATR_PCTL252] = atr_pctl_aligned.to_numpy(dtype=float)

    df_m15 = _resample_ohlc(df_m5, "15min")
    if len(df_m15) < 200:
        raise RuntimeError("[CTX_WARMUP_FAIL] insufficient M15 bars for ATR100 warmup")
    m15_atr14 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 14).ffill()
    m15_atr100 = _atr(df_m15["high"], df_m15["low"], df_m15["close"], 100).ffill()
    m15_comp = m15_atr14 / np.maximum(m15_atr100, ATR_EPS)
    m15_comp.iloc[:199] = np.nan
    m15_aligned = _align_last_closed(
        df_pre.index,
        m15_comp,
        pd.Timedelta(minutes=15),
    )
    if m15_aligned.isna().any():
        raise RuntimeError(
            "[CTX_ALIGN_FAIL] M15_range_compression_ratio has NaN after alignment"
        )
    df_pre[CTX_CONT_COL_M15_COMP] = m15_aligned.to_numpy(dtype=float)

    # ------------------------------------------------------------
    # Exact canonical micro/swing context from the strict tape join.
    # ------------------------------------------------------------
    if tape_root is None:
        raise RuntimeError("[CTX_INPUT_FAIL] exact canonical tape_root is required")
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
            "TAPE_JOIN_STRICT_FAIL: "
            f"rows_base28={rows_base28} rows_tape={rows_tape} "
            f"rows_joined={rows_joined}"
        )

    tape_feat = merged[["time", "close", "high", "low"]].copy().sort_values("time")
    high = tape_feat["high"].to_numpy(dtype=np.float64)
    low = tape_feat["low"].to_numpy(dtype=np.float64)
    close = tape_feat["close"].to_numpy(dtype=np.float64)
    for name, values in compute_micro_structure_features(high, low, close).items():
        tape_feat[name] = values
    swing = compute_swing_structure_features(
        high,
        low,
        close,
        lookback=SWING_LOOKBACK_V1,
        atr_period=SWING_ATR_PERIOD_V1,
    )
    for name, values in swing.items():
        tape_feat[name] = values
    log.info(
        "[ENTRY_SWING_PIVOT_PROOF] swing_resets_high=%d swing_resets_low=%d",
        int((np.diff(swing["bars_since_swing_high"]) < 0).sum()),
        int((np.diff(swing["bars_since_swing_low"]) < 0).sum()),
    )

    tape_feat = tape_feat.set_index("time")
    join_cols = list(MODEL_NATIVE_CTX_CONT_MICRO_FIELDS) + list(
        MODEL_NATIVE_CTX_CONT_SWING_FIELDS
    )
    existing_join = [c for c in join_cols if c in df_pre.columns]
    if existing_join:
        log.info(
            "[ENTRY_MICRO_FEATURES_OVERWRITE] dropping_existing=%s",
            existing_join,
        )
        df_pre = df_pre.drop(columns=existing_join)
    df_pre = df_pre.join(tape_feat[join_cols], how="left")
    if len(df_pre) != rows_base28:
        raise RuntimeError(
            f"MICRO_FEATURE_JOIN_FAIL: rows_base28={rows_base28} rows_after={len(df_pre)}"
        )

    log.info(
        "[ENTRY_MICRO_FEATURES_PROOF] names=%s count=%d",
        list(MODEL_NATIVE_CTX_CONT_MICRO_FIELDS),
        len(MODEL_NATIVE_CTX_CONT_MICRO_FIELDS),
    )
    log.info(
        "[ENTRY_SWING_FEATURES_PROOF] names=%s count=%d",
        list(MODEL_NATIVE_CTX_CONT_SWING_FIELDS),
        len(MODEL_NATIVE_CTX_CONT_SWING_FIELDS),
    )

    # ------------------------------------------------------------
    # CAT: 5/6 dims, deterministic, int, no NaN
    # ------------------------------------------------------------
    ts = df_pre.index

    # session_id: 0=ASIA, 1=EU, 2=OVERLAP, 3=US (SSoT)
    df_pre["session_id"] = get_session_id_vectorized(ts).astype(np.int64)
    df_pre["is_ASIA"] = (df_pre["session_id"] == 0).astype(np.int64)

    # Session timing features (observerable context)
    df_pre["minutes_since_session_open"] = get_session_minutes_since_open_vectorized(
        ts
    ).astype(np.float32)
    df_pre["minutes_to_next_session_boundary"] = (
        get_session_minutes_to_next_boundary_vectorized(ts).astype(np.float32)
    )
    # Session change flag (1 if session changes vs previous bar)
    session_id = df_pre["session_id"].to_numpy(dtype=np.int64)
    session_change = np.zeros(len(session_id), dtype=np.int64)
    if len(session_id) > 1:
        session_change[1:] = (session_id[1:] != session_id[:-1]).astype(np.int64)
    df_pre["session_change_flag"] = session_change
    # Tradable flag (policy can still restrict to EU/OVERLAP/US)
    df_pre["session_tradable"] = (df_pre["session_id"] != 0).astype(np.int64)

    # Legacy trend_regime_id has one immutable source: exact D1 distance.
    # price_vs_ema50_atr remains separate continuous learned evidence.
    if "D1_dist_from_ema200_atr" not in df_pre.columns:
        raise RuntimeError(
            "[CTX_TREND_REGIME] exact D1_dist_from_ema200_atr source missing"
        )
    d = df_pre["D1_dist_from_ema200_atr"].to_numpy(dtype=float)
    _finite_or_fail(d, label="trend_regime_id.D1_dist_from_ema200_atr")
    trend_regime_id = np.where(d < -1.0, 0, np.where(d <= 1.0, 1, 2)).astype(np.int64)
    df_pre["trend_regime_id"] = trend_regime_id

    # vol_regime_id / atr_bucket: 0..4 from atr_bps percentile rank
    vol_regime_id = _rank_bucket_0_4(df_pre["atr_bps"].to_numpy(dtype=float))
    df_pre["vol_regime_id"] = vol_regime_id.astype(np.int64)
    df_pre["atr_bucket"] = vol_regime_id.astype(np.int64)

    # spread_bucket: 0..4 from spread_bps percentile rank
    spread_bucket = _rank_bucket_0_4(df_pre["spread_bps"].to_numpy(dtype=float))
    df_pre["spread_bucket"] = spread_bucket.astype(np.int64)

    # H4_trend_sign_cat (optional): sign(mid - ema50) on H4, mapped to {0,1,2} for {-1,0,+1}
    # R4 (2026-06-04): compute H4_trend_sign_cat whenever it's in the (possibly trend_regime_id-
    # dropped) ctx_cat contract — NOT a hardcoded `ctx_cat_dim == 6`. The active
    # contract is 5 names (trend_regime_id dropped) but H4 is STILL required (it's the 5th), so a
    # 5-dim build must still compute it. Contract-driven, robust to the dim change.
    if CTX_CAT_COL_H4_TREND_SIGN in required_cat:
        df_h4 = _resample_ohlc(df_m5, "4h")
        if len(df_h4) < 80:
            raise RuntimeError(
                "[CTX_WARMUP_FAIL] insufficient H4 bars for EMA50 warmup"
            )
        h4_mid = (df_h4["high"] + df_h4["low"]) * 0.5
        h4_ema50 = _ema(h4_mid, 50)
        diff = h4_mid - h4_ema50
        sign_series = (np.sign(diff) + 1.0).astype(np.float64)
        sign_series.iloc[:79] = np.nan
        h4_aligned = _align_last_closed(
            df_pre.index, sign_series, pd.Timedelta(hours=4)
        )
        if h4_aligned.isna().any():
            raise RuntimeError(
                "[CTX_ALIGN_FAIL] H4_trend_sign_cat has NaN after alignment"
            )
        df_pre[CTX_CAT_COL_H4_TREND_SIGN] = h4_aligned.to_numpy(dtype=np.int64)

    # Ensure the exact active categorical contract is observed and integral.
    for c in required_cat:
        if c not in df_pre.columns or df_pre[c].isna().any():
            raise RuntimeError(
                f"[PREBUILT_CTX_CAT] exact finite categorical source missing: {c}"
            )
        df_pre[c] = df_pre[c].astype(np.int64)

    # ------------------------------------------------------------
    # Contract validation (names + finiteness)
    # ------------------------------------------------------------
    required_output_cont = list(MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS)
    if required_output_cont != (
        required_cont + list(MODEL_NATIVE_CTX_CONT_SESSION_FIELDS)
    ):
        raise RuntimeError("PREBUILT_CTX_OWNER_ORDER_MISMATCH")
    missing = [
        c for c in (required_output_cont + required_cat) if c not in df_pre.columns
    ]
    if missing:
        raise RuntimeError(f"[PREBUILT_CTX_CONTRACT_FAIL] missing columns: {missing}")

    cont_mat = df_pre[required_output_cont].to_numpy(dtype=float)
    _finite_or_fail(
        cont_mat,
        label=f"ctx_cont(required_output_cont={required_output_cont})",
    )

    log.info(
        "[PREBUILT_CTX_CONT_PROOF] core_dim=%d v1_output_dim=%d names=%s",
        prebuilt_ctx_cont_dim,
        len(required_output_cont),
        required_output_cont,
    )

    # REGIME_V4 is an immutable active transform, never an environment-selected surface.
    from gx1.features.regime_v4_features import (
        REGIME_V4_DERIVED_COLS,
        REGIME_V4_SOURCE_COLS,
        add_regime_v4_features,
    )

    df_pre = df_pre.sort_index()
    from gx1.features.htf_features import attach_v2_mtf_per_bar_scalars as _attach_v2

    _rv4_required = ["open", "high", "low", "close", "volume"]
    _rv4_missing = [name for name in _rv4_required if name not in df_m5.columns]
    if _rv4_missing:
        raise RuntimeError(
            f"[REGIME_V4] exact raw M5 OHLCV source missing: {_rv4_missing}"
        )
    _rv4_m5 = df_m5[_rv4_required].astype(np.float64).copy()
    _rv4_src_map = [
        ("ema20_slope_atr", "ema20_slope_atr"),
        ("ema_stack_aligned", "ema_stack_aligned_v2"),
        ("regime_class_id", "regime_class_id"),
        ("trend_age_bars_norm", "trend_age_bars_norm"),
        ("mom_5_atr", "mom_5_atr"),
        ("mom_20_atr", "mom_20_atr"),
        ("rsi14_centered", "rsi14_centered"),
        ("atr_bps_14", "atr_bps_14"),
        ("lower_wick_pct", "lower_wick_pct"),
    ]
    _attached = _attach_v2(
        _rv4_m5,
        df_pre.index.asi8,
        _rv4_src_map,
        ("m15", "h1", "h4", "d1", "m5"),
        frozenset({("d1", "lower_wick_pct")}),
    )
    for _c, _v in _attached.items():
        df_pre[_c] = _v
    add_regime_v4_features(df_pre)
    _regime_trim_required = list(REGIME_V4_SOURCE_COLS) + list(REGIME_V4_DERIVED_COLS)
    _regime_v4_emitted = True
    log.info("[REGIME_V4] emitted immutable causal regime features")

    # Cross/cyclic sources must already belong to this exact prebuilt. A
    # side-loaded parquet would create a second owner and can silently inject
    # stale rows, so missing fields fail closed.
    _need_cross = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "smc_premium_state",
        "m5h1_momentum",
    ]
    _missing_cross = [name for name in _need_cross if name not in df_pre.columns]
    if _missing_cross:
        raise RuntimeError(
            "[CTX_CROSS_SOURCE_MISSING] exact prebuilt must own all cross sources: "
            f"{_missing_cross}"
        )
    _finite_or_fail(
        df_pre[_need_cross].to_numpy(dtype=np.float64), label="exact cross sources"
    )

    # B2 train==serve (2026-06-05, XGB-audit): bake the SHIFTED _v1_is_EU/_v1_is_US (np.roll by 1, [0]=0).
    # The V10 build (exact source parquet) + candidate batch run XGB INFERENCE on THIS parquet, and the
    # new XGB is trained on SHIFTED is_EU/is_US (base80 builder B2) matching live serve
    # (v12_ctx_augment_live.py:165-172). Those builders otherwise derive them UNSHIFTED
    # (build_entry_v10_ctx_training_dataset_v3.py:1234-1237) -> train≠serve on ~0.7% session-boundary bars.
    # Outside the REGIME_V4 block: this is an XGB-bridge input, always needed. df_pre is time-sorted.
    if "_v1_is_EU" not in df_pre.columns or "_v1_is_US" not in df_pre.columns:
        from gx1.time.session_detector import get_session_vectorized as _get_sess

        _sess_b2 = _get_sess(pd.to_datetime(df_pre.index, utc=True))
        for _b2name, _b2tag in (("_v1_is_EU", "EU"), ("_v1_is_US", "US")):
            if _b2name not in df_pre.columns:
                _b2raw = (_sess_b2 == _b2tag).to_numpy(dtype=np.float64)
                _b2sh = np.roll(_b2raw, 1)
                _b2sh[0] = 0.0
                df_pre[_b2name] = _b2sh
        log.info(
            "[B2] baked shifted _v1_is_EU/_v1_is_US (train==serve XGB-inference parity)"
        )
    # *_us session-interactions on the SHIFTED is_US basis (same formulas as the trainer :876-881 and live
    # serve _add_session_interactions). The candidate batch's run_xgb_inference REQUIRES these present
    # (raises on missing); the V10 builder derives them too — baking here keeps ONE basis for every consumer.
    _is_us_b2 = df_pre["_v1_is_US"].to_numpy(dtype=np.float64)
    for _usname, _ussrc in (
        ("_v1_int_ema_us", "_v1_ema_diff"),
        ("_v1_int_range_us", "_v1_range_z"),
        ("_v1_int_slope_h1_us", "_v1h1_slope3"),
    ):
        if _usname not in df_pre.columns and _ussrc in df_pre.columns:
            df_pre[_usname] = df_pre[_ussrc].to_numpy(dtype=np.float64) * _is_us_b2

    from gx1.scripts.augment_forward_outcome_v2 import trim_causal_context_warmup_prefix

    df_pre = trim_causal_context_warmup_prefix(df_pre, _regime_trim_required)
    _finite_or_fail(
        df_pre[_regime_trim_required].to_numpy(dtype=np.float64),
        label="immutable REGIME_V4 output",
    )

    # ------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    # ONE-TRUTH SHAPE (2026-06-05): emit `time` as a plain column with a clean RangeIndex.
    # Saving index=True while a `time` column already exists produced a parquet whose
    # DatetimeIndex was *named* "time" AND carried a `time` column — the downstream V10
    # builder's `df.reset_index(drop=False)` then collides ("cannot insert time, already
    # exists"). Normalize here so every consumer (V10 build, Entry-IQL build, candidate batch)
    # loads a consistent time-as-column frame.
    _out = df_pre
    if "time" in _out.columns:
        _out = _out.reset_index(
            drop=True
        )  # time already a column → drop redundant index
    elif _out.index.name == "time" or isinstance(_out.index, pd.DatetimeIndex):
        _out = _out.reset_index(drop=False).rename(
            columns={"index": "time"}
        )  # surface time index
    _out.to_parquet(output_parquet, index=False)

    if diagnostics_path is not None:
        diagnostics_path = Path(diagnostics_path).resolve()
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(
                {
                    "prebuilt_path": str(prebuilt_path),
                    "output_path": str(output_parquet),
                    "raw_m5_paths": [str(p) for p in raw_m5_paths],
                    "prebuilt_ctx_cont_dim": int(prebuilt_ctx_cont_dim),
                    "entry_v1_ctx_cont_dim": len(required_output_cont),
                    "ctx_cat_dim": int(ctx_cat_dim),
                    "regime_v4_emitted": bool(_regime_v4_emitted),
                    "required_prebuilt_cont": required_cont,
                    "required_entry_v1_cont": required_output_cont,
                    "required_cat": required_cat,
                    "ctx_columns_added": required_output_cont + required_cat,
                    "ctx_contract_missing": [],
                    "n_rows": int(len(df_pre)),
                    "tape_root": str(tape_root) if tape_root is not None else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    log.info(
        "[PREBUILT_CTX_CONTRACT] exact cont+cat present; missing: [] (cont=%d cat=%d)",
        len(required_output_cont),
        ctx_cat_dim,
    )
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
        "kind": "entry_model_native_prebuilt_manifest_v2",
        "prebuilt_path": str(prebuilt_resolved),
        "prebuilt_sha256": prebuilt_sha256,
        "prebuilt_bytes": prebuilt_bytes,
        "prebuilt_ctx_cont_dim": int(prebuilt_ctx_cont_dim),
        "prebuilt_ctx_cont_names": required_cont,
        "entry_v1_ctx_cont_dim": len(required_output_cont),
        "entry_v1_ctx_cont_names": required_output_cont,
        "ctx_cat_dim": int(ctx_cat_dim),
        "ctx_cat_names": required_cat,
        "regime_v4_emitted": bool(
            _regime_v4_emitted
        ),  # parquet carries the 16 REGIME_V4 cols (see diagnostics note)
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

    ap = argparse.ArgumentParser(
        description="Add the exact model-native Entry prebuilt context"
    )
    ap.add_argument("--prebuilt_parquet", type=Path, required=True)
    ap.add_argument("--output_parquet", type=Path, required=True)
    ap.add_argument(
        "--raw_m5_parquet",
        type=Path,
        nargs="*",
        default=None,
        help="Raw M5 parquet(s). Default: GX1_DATA/data/data/_staging/XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet",
    )
    ap.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Optional diagnostics JSON path (default: <output>.ctx_diagnostics.json)",
    )
    ap.add_argument(
        "--tape-root",
        type=Path,
        required=True,
        help="Mandatory canonical tape root for exact micro/swing features.",
    )
    args = ap.parse_args()

    raw = args.raw_m5_parquet
    if not raw:
        gx1_data = Path(os.environ.get("GX1_DATA", "/home/andre2/GX1_DATA")).resolve()
        raw = [
            gx1_data
            / "data/data/_staging/XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet"
        ]

    diag = args.diagnostics
    if diag is None:
        diag = args.output_parquet.with_name(
            args.output_parquet.stem + ".ctx_diagnostics.json"
        )

    try:
        run_add_ctx_cont_columns(
            prebuilt_path=args.prebuilt_parquet,
            raw_m5_paths=list(raw),
            output_parquet=args.output_parquet,
            tape_root=args.tape_root,
            diagnostics_path=diag,
        )
    except Exception as e:
        print(f"[add_ctx_cont_columns] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
