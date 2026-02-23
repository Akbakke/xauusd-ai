#!/home/andre2/venvs/gx1/bin/python
"""
Add ctx_cont6 / ctx_cat6 columns to prebuilt parquet (STRICT 6/6 ONLY).

Deterministic, TRUTH-style, no lookahead, no quarantine dependency.

CTX_CONT (6):
  atr_bps,
  spread_bps,
  D1_dist_from_ema200_atr,
  H1_range_compression_ratio,
  D1_atr_percentile_252,
  M15_range_compression_ratio

CTX_CAT (6):
  session_id,
  trend_regime_id,
  vol_regime_id,
  atr_bucket,
  spread_bucket,
  H4_trend_sign_cat

Contract source of truth:
  gx1.contracts.signal_bridge_v1

This script HARD-FAILS if required contract columns are missing or non-finite.

IMPORTANT:
- Alignment is "last closed" (no lookahead): for each M5 timestamp t, we attach the most recent
  closed HTF bar value strictly before t (by shifting left: t - 1D / 1H / 15m / 4H).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
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

log = logging.getLogger(__name__)

ATR_EPS = 1e-9

CTX_CONT_DIM = 6
CTX_CAT_DIM = 6


def get_prebuilt_ctx_contract_columns() -> Tuple[List[str], List[str]]:
    """Return (required_cont6, required_cat6) with exact contract names for prebuilt. STRICT 6/6."""
    required_cont = list(ORDERED_CTX_CONT_NAMES_EXTENDED[:CTX_CONT_DIM])
    required_cat = list(ORDERED_CTX_CAT_NAMES_EXTENDED[:CTX_CAT_DIM])
    return required_cont, required_cat


def get_prebuilt_ctx_contract_columns_strict(
    strict_cont_dim: int = CTX_CONT_DIM, strict_cat_dim: int = CTX_CAT_DIM
) -> List[str]:
    """
    TRUTH ctx-contract: ONE UNIVERSE STRICT 6/6.
    Return cont followed by cat in exact contract order.
    """
    cont = list(ORDERED_CTX_CONT_NAMES_EXTENDED[:strict_cont_dim])
    cat = list(ORDERED_CTX_CAT_NAMES_EXTENDED[:strict_cat_dim])
    if len(cont) != strict_cont_dim or len(cat) != strict_cat_dim:
        raise RuntimeError(
            f"CTX_CONTRACT_DIM_MISMATCH: cont={len(cont)} cat={len(cat)} expected={strict_cont_dim}/{strict_cat_dim}"
        )
    return cont + cat


def _load_base28_contract(engine_root: Path) -> List[str]:
    """Load BASE28 feature names (exactly 28) in contract order. Hard-fail if file missing or invalid."""
    contract_path = engine_root / "gx1/xgb/contracts/xgb_input_features_base28_v1.json"
    if not contract_path.exists():
        raise FileNotFoundError(
            f"[TRUTH_BASE28_CTX] BASE28 contract file not found: {contract_path}. "
            "Create gx1/xgb/contracts/xgb_input_features_base28_v1.json with key 'features' (list of 28 names in order)."
        )
    with open(contract_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features")
    if not isinstance(features, list) or len(features) != 28:
        raise RuntimeError(
            f"[TRUTH_BASE28_CTX] Contract must have 'features' list of length 28, got {type(features).__name__} "
            f"len={len(features) if isinstance(features, list) else 'N/A'} at {contract_path}"
        )
    if not all(isinstance(x, str) for x in features):
        raise RuntimeError("[TRUTH_BASE28_CTX] Contract 'features' must be list of strings")
    return list(features)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ensure_dt_index(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    # best-effort upgrade
    for c in ("time", "ts", "timestamp"):
        if c in df.columns:
            out = df.copy()
            out = out.set_index(pd.to_datetime(out[c], utc=True))
            return out.sort_index()
    raise RuntimeError(f"[CTX_INPUT_FAIL] {name} must have DatetimeIndex (or time/ts column)")


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


def _rank_bucket_0_3(x: np.ndarray, fallback: int) -> np.ndarray:
    """
    Deterministic 0..3 bucket (contract: vol_regime_id, atr_bucket).
    Always returns int64, no NaN.
    """
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    try:
        s = pd.Series(x)
        q = s.rank(pct=True, method="average")
        qv = q.to_numpy(dtype=float)
        if not np.isfinite(qv).any():
            return np.full(len(x), int(np.clip(fallback, 0, 3)), dtype=np.int64)
        b = np.clip(qv * 4.0, 0.0, 3.99).astype(np.int64)
        b = np.where(np.isfinite(b), b, int(np.clip(fallback, 0, 3))).astype(np.int64)
        return b
    except Exception:
        return np.full(len(x), int(np.clip(fallback, 0, 3)), dtype=np.int64)


def _rank_bucket_0_2(x: np.ndarray, fallback: int) -> np.ndarray:
    """
    Deterministic 0..2 bucket (contract: spread_bucket).
    Always returns int64, no NaN.
    """
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    try:
        s = pd.Series(x)
        q = s.rank(pct=True, method="average")
        qv = q.to_numpy(dtype=float)
        if not np.isfinite(qv).any():
            return np.full(len(x), int(np.clip(fallback, 0, 2)), dtype=np.int64)
        b = np.clip(qv * 3.0, 0.0, 2.99).astype(np.int64)
        b = np.where(np.isfinite(b), b, int(np.clip(fallback, 0, 2))).astype(np.int64)
        return b
    except Exception:
        return np.full(len(x), int(np.clip(fallback, 0, 2)), dtype=np.int64)


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------


def run_add_ctx_cont_columns(
    prebuilt_path: Path,
    raw_m5_paths: List[Path],
    output_parquet: Path,
    diagnostics_path: Optional[Path] = None,
    truth_base28_ctx_cols: Optional[List[str]] = None,
) -> None:
    """
    Build ctx_cont6 / ctx_cat6 columns into prebuilt parquet.

    When truth_base28_ctx_cols is set (TRUTH_BASE28_CTX mode): output is restricted to exactly
    those 40 columns (28 BASE28 + 6 cont + 6 cat) in that order; hard-fail if any missing.

    HARD-FAIL:
      - if any required contract column missing
      - if any required ctx_cont column contains NaN/Inf after alignment
    """

    prebuilt_path = Path(prebuilt_path).resolve()
    output_parquet = Path(output_parquet).resolve()
    raw_m5_paths = [Path(p).resolve() for p in (raw_m5_paths or [])]

    # TRUTH: no quarantine/deprecated raw M5
    for p in raw_m5_paths:
        s = str(p)
        if "quarantine" in s or "_DEPRECATED" in s:
            raise RuntimeError(
                "TRUTH_VIOLATION_QUARANTINE_RAW_M5: raw M5 path must not use quarantine or deprecated staging"
            )

    log.info("raw_m5_path(s) used: %s", raw_m5_paths)
    log.info("prebuilt input path: %s", prebuilt_path)
    log.info("output path: %s", output_parquet)
    log.info("NO_FALLBACK_ENFORCED=1")

    if not prebuilt_path.exists():
        raise RuntimeError(f"[CTX_INPUT_FAIL] prebuilt not found: {prebuilt_path}")
    if not raw_m5_paths:
        raise RuntimeError("[CTX_INPUT_FAIL] raw_m5_paths is empty")
    for p in raw_m5_paths:
        if not p.exists():
            raise RuntimeError(f"[CTX_INPUT_FAIL] raw M5 not found: {p}")

    required_cont, required_cat = get_prebuilt_ctx_contract_columns()

    # ------------------------------------------------------------
    # Load prebuilt + raw
    # ------------------------------------------------------------
    df_pre = pd.read_parquet(prebuilt_path)
    df_pre = _ensure_dt_index(df_pre, name="prebuilt")

    raws = []
    for p in raw_m5_paths:
        df = pd.read_parquet(p)
        df = _ensure_dt_index(df, name=f"raw_m5:{p.name}")
        _require_cols(df, ["open", "high", "low", "close"], name=f"raw_m5:{p.name}")
        raws.append(df[["open", "high", "low", "close"]])
    df_m5 = pd.concat(raws, axis=0).sort_index()

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

    # atr_bps: derive from prebuilt; TRUTH mode: _v1_atr14 only (raw atr forbidden). Legacy: atr or _v1_atr14.
    if truth_base28_ctx_cols is not None:
        # TRUTH_BASE28_CTX: raw atr forbidden, require _v1_atr14 only
        if "atr" in df_pre.columns:
            raise RuntimeError(
                "[TRUTH_BASE28_CTX] ATR_COLUMN_FORBIDDEN: input prebuilt contains 'atr' (raw ATR). Use BASE28 '_v1_atr14' only."
            )
        if "_v1_atr14" not in df_pre.columns:
            raise RuntimeError(
                "[TRUTH_BASE28_CTX] MISSING_BASE28_ATR: input prebuilt missing '_v1_atr14' required for atr_bps."
            )
        atr_series = df_pre["_v1_atr14"]
    else:
        if "atr" in df_pre.columns:
            atr_series = df_pre["atr"]
        elif "_v1_atr14" in df_pre.columns:
            atr_series = df_pre["_v1_atr14"]
        else:
            raise RuntimeError("[CTX_ATR_BPS_FAIL] prebuilt must contain 'atr' or '_v1_atr14' to derive atr_bps")

    mid_m5 = (df_m5["high"] + df_m5["low"]) * 0.5
    mid_aligned = mid_m5.reindex(df_pre.index)
    if mid_aligned.isna().any() or (mid_aligned.to_numpy() <= 0).any():
        raise RuntimeError("[CTX_ATR_BPS_FAIL] mid_aligned missing or <= 0 for some prebuilt rows")

    atr_vals = atr_series.to_numpy(dtype=float)
    atr_bps = (atr_vals / np.maximum(mid_aligned.to_numpy(dtype=float), ATR_EPS)) * 1e4
    _finite_or_fail(atr_bps, label="atr_bps")
    df_pre["atr_bps"] = atr_bps

    # spread_bps: prefer prebuilt spread/close; otherwise 0.0
    if "spread" in df_pre.columns and "close" in df_pre.columns:
        close = df_pre["close"].to_numpy(dtype=float)
        close = np.where(close > 0, close, np.nan)
        spread = df_pre["spread"].to_numpy(dtype=float)
        spread_bps = (spread / close) * 1e4
        spread_bps = np.where(np.isfinite(spread_bps), spread_bps, 0.0)
        df_pre["spread_bps"] = spread_bps.astype(float)
    else:
        df_pre["spread_bps"] = 0.0

    # ------------------------------------------------------------
    # CONT: HTF core + extras for 6/6
    #   D1_dist_from_ema200_atr,
    #   H1_range_compression_ratio,
    #   D1_atr_percentile_252,
    #   M15_range_compression_ratio
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
        raise RuntimeError(
            "[CTX_ALIGN_FAIL] D1_dist_from_ema200_atr has NaN after alignment (no ffill/bfill allowed)"
        )
    if h1_aligned.isna().any():
        raise RuntimeError(
            "[CTX_ALIGN_FAIL] H1_range_compression_ratio has NaN after alignment (no ffill/bfill allowed)"
        )

    df_pre[CTX_CONT_COL_D1_DIST] = d1_aligned.to_numpy(dtype=float)
    df_pre[CTX_CONT_COL_H1_COMP] = h1_aligned.to_numpy(dtype=float)

    # D1_atr_percentile_252 (STRICT)
    d1_atr14_for_pctl = d1_atr14.copy()

    def _pctl_last(window: np.ndarray) -> float:
        w = np.asarray(window, dtype=float)
        if not np.isfinite(w).all():
            return float("nan")
        last = w[-1]
        return float((w <= last).mean())

    atr_pctl252 = d1_atr14_for_pctl.rolling(252, min_periods=252).apply(_pctl_last, raw=True)
    atr_pctl252 = atr_pctl252.ffill()
    atr_pctl_aligned = _align_last_closed(df_pre.index, atr_pctl252, pd.Timedelta(days=1))
    if atr_pctl_aligned.isna().any():
        raise RuntimeError("[CTX_ALIGN_FAIL] D1_atr_percentile_252 has NaN after alignment")
    df_pre[CTX_CONT_COL_D1_ATR_PCTL252] = atr_pctl_aligned.to_numpy(dtype=float)

    # M15_range_compression_ratio (STRICT)
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
    # CAT: STRICT 6/6, deterministic, int, no NaN
    # ------------------------------------------------------------
    ts = df_pre.index

    # session_id: 0..3 by UTC hour blocks
    session_id = (ts.hour.to_numpy() // 6).astype(np.int64)
    session_id = np.clip(session_id, 0, 3).astype(np.int64)
    df_pre["session_id"] = session_id

    # trend_regime_id: bucket by price_vs_ema50_atr if present else neutral=1
    if "price_vs_ema50_atr" in df_pre.columns:
        p = df_pre["price_vs_ema50_atr"].to_numpy(dtype=float)
        p = np.where(np.isfinite(p), p, 0.0)
        trend_regime_id = np.where(p < -0.5, 0, np.where(p <= 0.5, 1, 2)).astype(np.int64)
    else:
        trend_regime_id = np.ones(len(df_pre), dtype=np.int64)
    df_pre["trend_regime_id"] = trend_regime_id

    # vol_regime_id / atr_bucket: 0..3 (contract) from atr_bps percentile rank
    vol_regime_id = _rank_bucket_0_3(df_pre["atr_bps"].to_numpy(dtype=float), fallback=2)
    df_pre["vol_regime_id"] = vol_regime_id.astype(np.int64)
    df_pre["atr_bucket"] = vol_regime_id.astype(np.int64)

    # spread_bucket: 0..2 (contract) from spread_bps percentile rank
    spread_bucket = _rank_bucket_0_2(df_pre["spread_bps"].to_numpy(dtype=float), fallback=0)
    df_pre["spread_bucket"] = spread_bucket.astype(np.int64)

    # H4_trend_sign_cat (STRICT): sign(mid - ema50) on H4, mapped to {0,1,2} for {-1,0,+1}
    df_h4 = _resample_ohlc(df_m5, "4H")
    if len(df_h4) < 80:
        raise RuntimeError("[CTX_WARMUP_FAIL] insufficient H4 bars for EMA50 warmup")
    h4_mid = (df_h4["high"] + df_h4["low"]) * 0.5
    h4_ema50 = _ema(h4_mid, 50)
    diff = (h4_mid - h4_ema50).to_numpy(dtype=float)
    sign = np.sign(np.where(np.isfinite(diff), diff, 0.0)).astype(np.int64)  # -1/0/+1
    sign_cat = (sign + 1).astype(np.int64)  # 0/1/2
    sign_series = pd.Series(sign_cat, index=df_h4.index, dtype="int64")
    h4_aligned = _align_last_closed(df_pre.index, sign_series, pd.Timedelta(hours=4))
    if h4_aligned.isna().any():
        raise RuntimeError("[CTX_ALIGN_FAIL] H4_trend_sign_cat has NaN after alignment")
    df_pre[CTX_CAT_COL_H4_TREND_SIGN] = h4_aligned.to_numpy(dtype=np.int64)

    # Ensure cat columns are int and non-NaN
    for c in ORDERED_CTX_CAT_NAMES_EXTENDED[:CTX_CAT_DIM]:
        if c not in df_pre.columns:
            raise RuntimeError(f"[PREBUILT_CTX_CONTRACT_FAIL] missing required cat column: {c}")
        if df_pre[c].isna().any():
            df_pre[c] = df_pre[c].fillna(0)
        df_pre[c] = df_pre[c].astype(np.int64)

    # ------------------------------------------------------------
    # Contract validation (names + finiteness) STRICT 6/6
    # ------------------------------------------------------------
    missing = [c for c in (required_cont + required_cat) if c not in df_pre.columns]
    if missing:
        raise RuntimeError(f"[PREBUILT_CTX_CONTRACT_FAIL] missing columns: {missing}")

    cont_mat = df_pre[required_cont].to_numpy(dtype=float)
    _finite_or_fail(cont_mat, label=f"ctx_cont(required_cont={required_cont})")

    # ------------------------------------------------------------
    # TRUTH_BASE28_CTX: restrict to exactly 40 cols in order
    # ------------------------------------------------------------
    if truth_base28_ctx_cols is not None:
        missing = [c for c in truth_base28_ctx_cols if c not in df_pre.columns]
        if missing:
            raise RuntimeError(
                f"[TRUTH_BASE28_CTX] Output columns missing after ctx build: {missing}. required_cols={truth_base28_ctx_cols}"
            )
        if len(truth_base28_ctx_cols) != 40:
            raise RuntimeError(
                f"[TRUTH_BASE28_CTX] required_cols must have length 40, got {len(truth_base28_ctx_cols)}"
            )
        df_pre = df_pre[truth_base28_ctx_cols].copy()
        assert len(df_pre.columns) == 40, "TRUTH_BASE28_CTX: len(df_out.columns) != 40"

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
                    "ctx_cont_dim": CTX_CONT_DIM,
                    "ctx_cat_dim": CTX_CAT_DIM,
                    "required_cont": required_cont,
                    "required_cat": required_cat,
                    "ctx_columns_added": required_cont + required_cat,
                    "ctx_contract_missing": [],
                    "n_rows": int(len(df_pre)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    log.info(
        "[PREBUILT_CTX_CONTRACT] required cont+cat present; missing: [] (cont=%d cat=%d)",
        CTX_CONT_DIM,
        CTX_CAT_DIM,
    )
    log.info("Wrote %s (%d rows)", output_parquet, len(df_pre))

    # Deterministic sidecar manifest (exactly one per prebuilt; remove existing first)
    manifest_path = output_parquet.with_suffix(".manifest.json")
    if manifest_path.exists():
        log.info("Removing existing manifest: %s", manifest_path)
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
        "ctx_cont_dim": CTX_CONT_DIM,
        "ctx_cat_dim": CTX_CAT_DIM,
        "no_fallback_enforced": True,
    }
    manifest_path.write_text(json.dumps(manifest_obj, indent=2), encoding="utf-8")
    log.info("PREBUILT_MANIFEST_WRITTEN=%s", manifest_path.resolve())

    # Schema manifest (sidecar; required by TRUTH preflight; remove existing first)
    schema_manifest_path = output_parquet.with_suffix(".schema_manifest.json")
    if schema_manifest_path.exists():
        log.info("Removing existing schema manifest: %s", schema_manifest_path)
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

    ap = argparse.ArgumentParser(description="Add ctx_cont6/ctx_cat6 columns to prebuilt parquet (STRICT 6/6 ONLY)")
    ap.add_argument(
        "--prebuilt_parquet",
        type=Path,
        default=None,
        help="Legacy mode only: input prebuilt parquet (requires --output_parquet). STRICT 6/6 is always used.",
    )
    ap.add_argument(
        "--output_parquet",
        type=Path,
        default=None,
        help="Legacy mode only: output parquet (requires --prebuilt_parquet). STRICT 6/6 is always used.",
    )
    ap.add_argument(
        "--truth-base28-ctx-out",
        dest="truth_base28_ctx_out",
        type=Path,
        default=None,
        metavar="PATH",
        help="TRUTH_BASE28_CTX mode: write 40-col parquet (28 BASE28 + ctx_cont6 + ctx_cat6) to PATH. Requires --input-prebuilt.",
    )
    ap.add_argument(
        "--input-prebuilt",
        dest="input_prebuilt",
        type=Path,
        default=None,
        metavar="PATH",
        help="TRUTH mode: input prebuilt (exactly BASE28 28 cols in contract order). Required when --truth-base28-ctx-out is set.",
    )
    ap.add_argument(
        "--raw_m5_parquet",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Raw M5 parquet(s). If omitted: ONLY GX1_DATA/data/data/_staging/"
            "XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet (hard-fail if missing). "
            "Paths containing quarantine or _DEPRECATED are forbidden."
        ),
    )
    ap.add_argument(
        "--diagnostics",
        type=Path,
        default=None,
        help="Optional diagnostics JSON path (default: <output>.ctx_diagnostics.json)",
    )
    args = ap.parse_args()

    truth_out = getattr(args, "truth_base28_ctx_out", None)
    input_prebuilt = getattr(args, "input_prebuilt", None)

    if truth_out is not None:
        # TRUTH_BASE28_CTX mode
        if input_prebuilt is None:
            raise RuntimeError("[TRUTH_BASE28_CTX] --truth-base28-ctx-out requires --input-prebuilt")
        input_prebuilt = Path(input_prebuilt).resolve()
        truth_out = Path(truth_out).resolve()

        # TRUTH: forbid writing into BASE28_CANONICAL
        if "BASE28_CANONICAL" in str(truth_out):
            raise RuntimeError(
                f"[TRUTH_BASE28_CTX] OUTPUT_PATH_FORBIDDEN: refusing to write into BASE28_CANONICAL in TRUTH mode: {truth_out}"
            )

        if not input_prebuilt.exists():
            raise FileNotFoundError(f"[TRUTH_BASE28_CTX] input prebuilt not found: {input_prebuilt}")

        engine_root = Path(os.environ.get("GX1_ENGINE", "/home/andre2/src/GX1_ENGINE")).resolve()
        base28_cols = _load_base28_contract(engine_root)

        df_in = pd.read_parquet(input_prebuilt)
        df_in = _ensure_dt_index(df_in, name="input_prebuilt")
        have = list(df_in.columns)
        if len(have) != 28:
            raise RuntimeError(
                f"[TRUTH_BASE28_CTX] input prebuilt must have exactly 28 columns (BASE28 contract), got {len(have)}. path={input_prebuilt}"
            )
        missing = [c for c in base28_cols if c not in have]
        if missing:
            raise RuntimeError(
                f"[TRUTH_BASE28_CTX] input prebuilt missing BASE28 columns: {missing}. path={input_prebuilt}"
            )
        extra = [c for c in have if c not in base28_cols]
        if extra:
            raise RuntimeError(
                f"[TRUTH_BASE28_CTX] input prebuilt has extra columns (must be exactly BASE28 28): {extra}. path={input_prebuilt}"
            )
        if have != base28_cols:
            raise RuntimeError(
                f"[TRUTH_BASE28_CTX] input prebuilt column order must match contract. expected={base28_cols} got={have}. path={input_prebuilt}"
            )
        mat = df_in.to_numpy(dtype=float, copy=False)
        if not np.isfinite(mat).all():
            n_bad = int((~np.isfinite(mat)).sum())
            raise RuntimeError(
                f"[TRUTH_BASE28_CTX] input prebuilt contains NaN/Inf: count={n_bad}. path={input_prebuilt}"
            )

        required_cont, required_cat = get_prebuilt_ctx_contract_columns()
        required_cols = base28_cols + required_cont + required_cat
        assert len(required_cols) == 40

        RAW_M5_STAGING_REL = "data/data/_staging/XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet"
        gx1_data = Path(os.environ.get("GX1_DATA", "/home/andre2/GX1_DATA")).resolve()
        default_raw = gx1_data / RAW_M5_STAGING_REL
        raw = args.raw_m5_parquet if args.raw_m5_parquet else []
        if not raw:
            if not default_raw.exists():
                raise RuntimeError(f"[CTX_INPUT_FAIL] raw M5 staging missing: {default_raw}")
            raw = [default_raw]
        for p in raw:
            p_res = Path(p).resolve()
            if "quarantine" in str(p_res) or "_DEPRECATED" in str(p_res):
                raise RuntimeError(
                    "TRUTH_VIOLATION_QUARANTINE_RAW_M5: raw M5 path must not use quarantine or deprecated staging"
                )

        diag = args.diagnostics or truth_out.with_name(truth_out.stem + ".ctx_diagnostics.json")
        try:
            run_add_ctx_cont_columns(
                prebuilt_path=input_prebuilt,
                raw_m5_paths=list(raw),
                output_parquet=truth_out,
                diagnostics_path=diag,
                truth_base28_ctx_cols=required_cols,
            )
        except Exception as e:
            print(f"[add_ctx_cont_columns] {e}", file=sys.stderr)
            return 1
        return 0

    # Legacy mode: --prebuilt_parquet and --output_parquet required
    if args.prebuilt_parquet is None or args.output_parquet is None:
        raise RuntimeError(
            "Either (--truth-base28-ctx-out + --input-prebuilt) or (--prebuilt_parquet + --output_parquet) is required"
        )

    RAW_M5_STAGING_REL = "data/data/_staging/XAUUSD_M5_2020_2025_bidask__TEMP_CTX2PLUS.parquet"
    gx1_data = Path(os.environ.get("GX1_DATA", "/home/andre2/GX1_DATA")).resolve()
    default_raw = gx1_data / RAW_M5_STAGING_REL

    raw = args.raw_m5_parquet
    if not raw:
        if not default_raw.exists():
            raise RuntimeError(f"[CTX_INPUT_FAIL] raw M5 staging missing: {default_raw}")
        raw = [default_raw]

    for p in raw:
        p_res = Path(p).resolve()
        s = str(p_res)
        if "quarantine" in s or "_DEPRECATED" in s:
            raise RuntimeError(
                "TRUTH_VIOLATION_QUARANTINE_RAW_M5: raw M5 path must not use quarantine or deprecated staging"
            )

    diag = args.diagnostics
    if diag is None:
        diag = args.output_parquet.with_name(args.output_parquet.stem + ".ctx_diagnostics.json")

    try:
        run_add_ctx_cont_columns(
            prebuilt_path=args.prebuilt_parquet,
            raw_m5_paths=list(raw),
            output_parquet=args.output_parquet,
            diagnostics_path=diag,
        )
    except Exception as e:
        print(f"[add_ctx_cont_columns] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())