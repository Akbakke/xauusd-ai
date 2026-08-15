#!/usr/bin/env python3
# ruff: noqa: E402
"""V2 Phase B5 — augment forward-outcome parquets with 125 per-TF + 28 group-A V2 scalars.

C+prune strategy: extract the exact historical 25-field group-A subset per TF
from the verified V4 cache (5 TFs × 25 = 125), plus 28 group-A features. This
derived context surface is not the model's separate 5-per-TF V4 input grid
(width = htf_features.MULTI_TF_FEATURE_COUNT_V4, derived from the declared tuples).

OPTIMIZED (2026-07-21): builds all caches once, resolves each TF snapshot once,
and keeps row lookup zero-copy. Measured full-contract throughput is >2,000
candidates/s on the six-year source (vs single-digit throughput per worker
with full-matrix casts at every lookup).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/andre2/src/GX1_ENGINE")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gx1.features.htf_features import (
    build_multi_tf_per_bar_features_v4,
    HTF_V4_MATRIX_CONTRACT,
    multi_tf_bar_label,
    multi_tf_resample,
    MULTI_TF_PER_BAR_FEATURES_V4,
    MULTI_TF_V4_GROUP_A_BASE_FEATURES,
    MULTI_TF_TIMEFRAMES,
    validate_causal_feature_matrix,
)

TF_NAMES = MULTI_TF_TIMEFRAMES
_GROUP_A_MTF_SOURCE_FEATURES = MULTI_TF_V4_GROUP_A_BASE_FEATURES
_GROUP_A_MTF_SOURCE_INDICES = tuple(
    MULTI_TF_PER_BAR_FEATURES_V4.index(name)
    for name in _GROUP_A_MTF_SOURCE_FEATURES
)


class CausalContextWarmupError(RuntimeError):
    """Required causal history does not yet exist for this source prefix."""

def _pertf_name(tf: str, feat: str) -> str:
    base = f"{tf.lower()}_{feat}"
    return base if feat.endswith("_v2") else base + "_v2"

# ── Immutable closed-bar contract for the per-TF V2-cache asof ────────────────────────────────
# The MULTI_TF_V2 cache is START-stamped (resample label='left'); a bare searchsorted(side='right') at an
# intraday decision picks the SAME-period still-FORMING D1/H4/H1 bar (features use the full bar's FUTURE
# close) = forward leak. Proven: the V2 per-TF adds gave +0.18 naive AUC that COLLAPSES to ~0 causal. These
# helpers feed the V2 per-TF scalars and model-native context. The leaky mode is
# retired: every caller selects the last HTF bar closed by the M5 decision moment
# (M5 closes 5min after label ts): a bar started at S is closed by D iff S <= D - bar_duration; M5 → unchanged.
_DECISION_M5_NS = 300_000_000_000  # M5 bar closes 5min after its label ts
from gx1.features.htf_features import MULTI_TF_SHIFT as _MTF_SHIFT
_TF_SHIFT_NS = {tf: int(_MTF_SHIFT[tf].value) for tf in TF_NAMES}

def _cache_cutoff_ns(
    ts_ns: int,
    tf: str,
    *,
    decision_bar_duration_ns: int = _DECISION_M5_NS,
) -> int:
    """Return the latest start-stamped ``tf`` row closed at decision time."""
    if tf not in _TF_SHIFT_NS:
        raise RuntimeError(f"[CTX_CAUSALITY] unsupported timeframe: {tf!r}")
    return ts_ns + int(decision_bar_duration_ns) - _TF_SHIFT_NS[tf]


def trim_causal_context_warmup_prefix(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...] | list[str],
) -> pd.DataFrame:
    """Remove only a contiguous non-finite prefix from causal feature output.

    Warmup absence is represented as unavailable data, never as a numeric
    sentinel. A non-finite value after the first complete row is corruption and
    fails closed. Callers must still enforce their own sequence-length minimum.
    """
    required = list(required_columns)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"[CTX_WARMUP_TRIM] required columns missing: {missing}")
    values = frame[required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(values).all(axis=1)
    if not invalid.any():
        return frame
    first_valid = int(np.argmax(~invalid)) if (~invalid).any() else len(frame)
    if first_valid == len(frame):
        raise RuntimeError("[CTX_WARMUP_TRIM] no complete causal context row exists")
    if not invalid[:first_valid].all() or invalid[first_valid:].any():
        raise RuntimeError("[CTX_WARMUP_TRIM] non-finite context is not a contiguous warmup prefix")
    trimmed = frame.iloc[first_valid:].copy(deep=False)
    trimmed.attrs.update(frame.attrs)
    trimmed.attrs["causal_context_warmup_rows_trimmed"] = first_valid
    return trimmed

PER_TF_FEATURE_NAMES = tuple(
    _pertf_name(tf, feat)
    for tf in TF_NAMES
    for feat in MULTI_TF_V4_GROUP_A_BASE_FEATURES
)
# 5 × 25 = 125

GROUP_A_FEATURE_NAMES = (
    "dist_to_R1_atr", "dist_to_R2_atr", "dist_to_S1_atr", "dist_to_S2_atr",
    # 2026-05-24 Bug 2 fix: added M15+D1 (was only M5/H1/H4)
    "dist_to_m5_hi_atr",  "dist_to_m5_lo_atr",
    "dist_to_m15_hi_atr", "dist_to_m15_lo_atr",
    "dist_to_h1_hi_atr",  "dist_to_h1_lo_atr",
    "dist_to_h4_hi_atr",  "dist_to_h4_lo_atr",
    "dist_to_d1_hi_atr",  "dist_to_d1_lo_atr",
    "long_win_rate_last10",  "long_mean_pnl_last10",
    "long_n_consec_losses",  "long_time_since_last_close_min",
    "short_win_rate_last10", "short_mean_pnl_last10",
    "short_n_consec_losses", "short_time_since_last_close_min",
)
PORTFOLIO_FEATURE_NAMES = (
    "long_win_rate_last10", "long_mean_pnl_last10",
    "long_n_consec_losses", "long_time_since_last_close_min",
    "short_win_rate_last10", "short_mean_pnl_last10",
    "short_n_consec_losses", "short_time_since_last_close_min",
)
# 2026-05-24 GROUP_S: SMC features from canonical_v3 (joined by decision_ts).
# These ARE in canonical_v3 prebuilt with real signal (smc_choch 421 non-zero,
# smc_bos_down 76K, smc_sweep_up/down 25K each) but were NEVER joined into
# forward_outcome → all zero downstream. Fix: load canonical_v3 once, asof-join
# by time to add SMC cols (with _canon_v1 suffix to match downstream expectations).
GROUP_S_SMC_FEATURE_NAMES = (
    "smc_swing_state_canon_v1",
    "smc_bos_up_canon_v1",
    "smc_bos_down_canon_v1",
    "smc_choch_canon_v1",
    "smc_sweep_up_canon_v1",
    "smc_sweep_down_canon_v1",
    "smc_sweep_event_age_bars_canon_v2",
    "smc_pivot_envelope_position_canon_v1",
)
# --forward-outcome-dir is REQUIRED (no silent stale default; rule 8). The old hardcoded literal
# (CANDIDATE_FORWARD_OUTCOME_V3PLUS_..._20260521 LOCK) is superseded 2x (v3+ -> COSTFIX -> fase2b)
# and quarantined — pass the current forward-outcome dir explicitly so an augment can't read a stale set.
M5_PREBUILT = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")


@dataclass
class AugmentContext:
    """Pre-built caches for fast per-candidate compute."""
    decision_close: np.ndarray      # native local-clock close prices
    decision_ts_ns: np.ndarray      # native local-clock bar-start timestamps
    m5_close: np.ndarray            # close prices
    m5_high: np.ndarray
    m5_low: np.ndarray
    m5_ts_ns: np.ndarray            # int64 ns
    # Per-TF cache (from V2 multi-TF builder): {tf: feats_df_with_attrs}
    multi_tf: dict
    # Resampled H1/H4 OHLC for liquidity zones
    h1_ts_ns: np.ndarray
    h1_high: np.ndarray
    h1_low: np.ndarray
    h4_ts_ns: np.ndarray
    h4_high: np.ndarray
    h4_low: np.ndarray
    # 2026-05-24 Bug 2 fix: M15 + D1 for liquidity zones
    m15_ts_ns: np.ndarray
    m15_high: np.ndarray
    m15_low: np.ndarray
    d1_ts_ns: np.ndarray
    d1_high: np.ndarray
    d1_low: np.ndarray
    # D1 pivot levels per M5 (R1/R2/S1/S2 from prior D1 OHLC)
    # store as per-day arrays: lookup via date
    daily_pivot_by_date: dict       # date_str → {R1, R2, S1, S2}
    # Trade history from journal — DataFrame with (close_ts, side, pnl_bps)
    trade_history: pd.DataFrame
    decision_bar_duration_ns: int


def _require_utc_timestamp(value: object, *, context: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts) or ts.tzinfo is None or ts.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"[{context}] timestamp must be finite UTC: {value!r}")
    return ts.tz_convert("UTC")


def _validate_m5_frame(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"[{context}] M5 source must be a non-empty DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise RuntimeError(f"[{context}] M5 source needs a timezone-aware UTC DatetimeIndex")
    if any(pd.Timestamp(ts).utcoffset() != pd.Timedelta(0) for ts in df.index[:1]):
        raise RuntimeError(f"[{context}] M5 source index must be UTC")
    if df.index.hasnans or not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise RuntimeError(f"[{context}] M5 timestamps must be finite, unique and chronological")
    missing = [name for name in ("high", "low", "close") if name not in df.columns]
    if missing:
        raise RuntimeError(f"[{context}] M5 source missing exact columns: {missing}")
    numeric = df.loc[:, ["high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"[{context}] M5 OHLC sources must be finite")
    high = numeric["high"].to_numpy(dtype=np.float64)
    low = numeric["low"].to_numpy(dtype=np.float64)
    close = numeric["close"].to_numpy(dtype=np.float64)
    if np.any(low <= 0.0) or np.any(high < low) or np.any(close < low) or np.any(close > high):
        raise RuntimeError(f"[{context}] M5 OHLC geometry is invalid")
    return numeric


def _indexed_m5_ohlc_frame(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Return exact chronological M5 high/low/close with a UTC index."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError(f"[{context}] M5 source must be a non-empty DataFrame")
    if "time" in frame.columns:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame["time"], utc=True, errors="coerce")
        )
    elif isinstance(frame.index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(
            pd.to_datetime(frame.index, utc=True, errors="coerce")
        )
    else:
        raise RuntimeError(f"[{context}] M5 source needs a time column or DatetimeIndex")
    missing = [name for name in ("high", "low", "close") if name not in frame.columns]
    if missing:
        raise RuntimeError(f"[{context}] M5 source missing exact columns: {missing}")
    out = frame.loc[:, ["high", "low", "close"]].copy()
    out.index = index
    _validate_m5_frame(out, context=context)
    return out


def _build_resampled_ohlc_array(df: pd.DataFrame, timeframe: str) -> tuple:
    """Return (ts_int64, high, low) for one declared timeframe's bars.

    V30 package 3 (2026-08-13): keyed on the declared TIMEFRAME through the one
    cadence+origin owner (``htf_features.multi_tf_resample``) instead of a bare
    rule string, so the D1 liquidity-zone highs/lows sit on the same
    trading-day bars as the rest of the surface.
    """
    resamp = (
        multi_tf_resample(df, timeframe)
        .agg({"high": "max", "low": "min"})
        .dropna()
    )
    ts_ns = resamp.index.values.astype("datetime64[ns]").astype(np.int64)
    return ts_ns, resamp["high"].to_numpy(np.float64), resamp["low"].to_numpy(np.float64)


def _build_daily_pivots(df: pd.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    """Compute classic pivots per trading day for later prior-day lookup.

    V30 package 3 (2026-08-13): the "trading day" this docstring always claimed
    is now the actual bin.  The literal ``resample("1D")`` used pandas' default
    midnight-UTC origin, so a Sunday 22:00-24:00 reopen stub became its own
    "day" and its 2-hour high/low/close produced a pivot set that the next
    session read as the previous day's.  Routed through the one cadence+origin
    owner (``htf_features.multi_tf_resample``) with the D1 trading-day origin.
    """
    daily = (
        multi_tf_resample(df, "D1")
        .agg({"high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    out = {}
    for ts, row in daily.iterrows():
        high, low, close = float(row["high"]), float(row["low"]), float(row["close"])
        pp = (high + low + close) / 3.0
        out[pd.Timestamp(ts).tz_convert("UTC")] = {
            "R1": 2 * pp - low,  "R2": pp + (high - low),
            "S1": 2 * pp - high,  "S2": pp - (high - low),
        }
    return out


def _build_trade_history(journal_dir: Path, suffix: str = "live_v12_4") -> pd.DataFrame:
    """One-shot parse of all live journals → DataFrame (close_ts, side, pnl_bps)."""
    if not journal_dir.exists():
        return pd.DataFrame(columns=["close_ts", "side", "pnl_bps"])
    files = sorted(journal_dir.glob(f"v12_paper_journal_*_{suffix}.jsonl"))
    rows = []
    last_snap: dict[str, dict] = {}
    prev_ids: set[str] = set()
    for fp in files:
        try:
            with fp.open() as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    otrs = e.get("open_trade_records") or []
                    cur_ids = {str(r["trade_id"]) for r in otrs if r.get("trade_id")}
                    for r in otrs:
                        tid = str(r.get("trade_id") or "")
                        if tid:
                            last_snap[tid] = {
                                "side": r.get("side"),
                                "pnl_bps": r.get("pnl_bps"),
                                "close_ts": e.get("logged_at_utc") or e.get("ts_utc"),
                            }
                    for tid in (prev_ids - cur_ids):
                        snap = last_snap.pop(tid, None)
                        if snap and snap.get("pnl_bps") is not None and snap.get("close_ts"):
                            rows.append(snap)
                    prev_ids = cur_ids
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["close_ts", "side", "pnl_bps"])
    df = pd.DataFrame(rows)
    df["close_ts"] = pd.to_datetime(df["close_ts"], utc=True)
    df["pnl_bps"] = pd.to_numeric(df["pnl_bps"], errors="coerce")
    return df.dropna(subset=["close_ts", "pnl_bps"]).sort_values("close_ts").reset_index(drop=True)


def _assert_multi_tf_cache_fresh(
    m5_df: pd.DataFrame,
    multi_tf: dict,
    *,
    final_decision_ts_ns: int,
    decision_bar_duration_ns: int = _DECISION_M5_NS,
) -> None:
    """Require the exact causal cache and finite evidence at the final cutoff."""
    if not isinstance(multi_tf, dict):
        raise RuntimeError("[MTF_CACHE_CONTRACT] multi_tf must be an explicit dictionary")
    if set(multi_tf) != set(TF_NAMES):
        raise RuntimeError(
            f"[MTF_CACHE_CONTRACT] exact TF keys required: expected={list(TF_NAMES)} "
            f"observed={sorted(map(str, multi_tf))}"
        )
    if (
        isinstance(final_decision_ts_ns, bool)
        or not isinstance(final_decision_ts_ns, (int, np.integer))
    ):
        raise RuntimeError("[MTF_CACHE_CONTRACT] final decision timestamp is invalid")
    final_decision_ts_ns = int(final_decision_ts_ns)
    expected_width = len(MULTI_TF_PER_BAR_FEATURES_V4)
    for tf in TF_NAMES:
        feats = multi_tf[tf]
        if not isinstance(feats, pd.DataFrame) or feats.empty:
            raise RuntimeError(f"[MTF_CACHE_CONTRACT] {tf} cache must be a non-empty DataFrame")
        ts_arr = np.asarray(feats.attrs.get("ts_int64"), dtype=np.int64)
        values = np.asarray(feats.attrs.get("feats_np"))
        if ts_arr.ndim != 1 or len(ts_arr) != len(feats) or np.any(np.diff(ts_arr) <= 0):
            raise RuntimeError(f"[MTF_CACHE_CONTRACT] {tf} timestamps are missing or invalid")
        if (
            feats.attrs.get("htf_feature_contract")
            != HTF_V4_MATRIX_CONTRACT
            or tuple(feats.columns) != MULTI_TF_PER_BAR_FEATURES_V4
        ):
            raise RuntimeError(f"[MTF_CACHE_CONTRACT] {tf} exact causal matrix contract missing")
        warmup_rows = validate_causal_feature_matrix(
            values,
            expected_width=expected_width,
            context=f"MTF_CACHE_CONTRACT_{tf}",
        )
        if feats.attrs.get("causal_warmup_rows") != warmup_rows:
            raise RuntimeError(f"[MTF_CACHE_CONTRACT] {tf} warmup metadata mismatch")
        closed_cutoff = _cache_cutoff_ns(
            final_decision_ts_ns,
            tf,
            decision_bar_duration_ns=decision_bar_duration_ns,
        )
        expected = (
            multi_tf_resample(m5_df, tf)
            .agg({"high": "max", "low": "min", "close": "last"})
            .dropna()
            .index.view("int64")
        )
        expected_right = int(np.searchsorted(expected, closed_cutoff, side="right"))
        cache_right = int(np.searchsorted(ts_arr, closed_cutoff, side="right"))
        if expected_right == 0 or cache_right == 0:
            raise RuntimeError(f"[MTF_CACHE_CONTRACT] {tf} has no closed history at final cutoff")
        expected_latest = int(expected[expected_right - 1])
        cache_latest = int(ts_arr[cache_right - 1])
        if cache_latest != expected_latest:
            raise RuntimeError(
                f"[MTF_CACHE_STALE] {tf} cache cannot cover the final closed bar: "
                f"cache_latest={pd.Timestamp(cache_latest, tz='UTC')} "
                f"expected_latest={pd.Timestamp(expected_latest, tz='UTC')}"
            )


def build_context(
    m5_df: pd.DataFrame,
    multi_tf: dict,
    journal_dir: Path,
    *,
    decision_frame: pd.DataFrame | None = None,
    decision_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> AugmentContext:
    """Pre-compute all caches ONCE. Heavy upfront cost, fast per-candidate after."""
    if not isinstance(decision_bar_duration, pd.Timedelta) or decision_bar_duration <= pd.Timedelta(0):
        raise RuntimeError("[CTX_BUILD] decision bar duration must be positive")
    decision_bar_duration_ns = int(decision_bar_duration.value)
    _validate_m5_frame(m5_df, context="CTX_BUILD_M5")
    if np.any(m5_df.index.asi8 % _DECISION_M5_NS != 0):
        raise RuntimeError("[CTX_BUILD] M5 context timestamps are off the five-minute grid")
    decision = m5_df if decision_frame is None else decision_frame
    _validate_m5_frame(decision, context="CTX_BUILD_DECISION")
    if np.any(decision.index.asi8 % decision_bar_duration_ns != 0):
        raise RuntimeError("[CTX_BUILD] decision timestamps are off the declared local grid")
    _assert_multi_tf_cache_fresh(
        m5_df,
        multi_tf,
        final_decision_ts_ns=int(decision.index[-1].value),
        decision_bar_duration_ns=decision_bar_duration_ns,
    )  # fail-closed on stale multi-TF cache (rule 4)
    ts_ns = m5_df.index.values.astype("datetime64[ns]").astype(np.int64)
    h1_ts, h1_hi, h1_lo = _build_resampled_ohlc_array(m5_df, "H1")
    h4_ts, h4_hi, h4_lo = _build_resampled_ohlc_array(m5_df, "H4")
    # 2026-05-24 Bug 2 fix: M15 + D1 resampled OHLC for liquidity zones
    m15_ts, m15_hi, m15_lo = _build_resampled_ohlc_array(m5_df, "M15")
    d1_ts, d1_hi, d1_lo = _build_resampled_ohlc_array(m5_df, "D1")
    # Daily pivots
    daily_pivots = _build_daily_pivots(m5_df)
    # Trade history (journal)
    trade_hist = _build_trade_history(journal_dir)

    return AugmentContext(
        decision_close=decision["close"].to_numpy(np.float64),
        decision_ts_ns=decision.index.values.astype("datetime64[ns]").astype(np.int64),
        m5_close=m5_df["close"].to_numpy(np.float64),
        m5_high=m5_df["high"].to_numpy(np.float64),
        m5_low=m5_df["low"].to_numpy(np.float64),
        m5_ts_ns=ts_ns,
        multi_tf=multi_tf,
        h1_ts_ns=h1_ts, h1_high=h1_hi, h1_low=h1_lo,
        h4_ts_ns=h4_ts, h4_high=h4_hi, h4_low=h4_lo,
        m15_ts_ns=m15_ts, m15_high=m15_hi, m15_low=m15_lo,
        d1_ts_ns=d1_ts, d1_high=d1_hi, d1_low=d1_lo,
        daily_pivot_by_date=daily_pivots,
        trade_history=trade_hist,
        decision_bar_duration_ns=decision_bar_duration_ns,
    )


def _tf_cache_row(ctx: AugmentContext, tf: str, ts_ns: int) -> np.ndarray:
    if tf not in ctx.multi_tf:
        raise RuntimeError(f"[CTX_MTF_SOURCE] missing exact timeframe {tf}")
    feats = ctx.multi_tf[tf]
    # The cache contract is already int64/float32.  Never request a dtype here:
    # doing so used to copy the complete (up to 461k x 25) float32 matrix to
    # float64 for every row lookup.  A single candidate performs five TF
    # lookups, so that hidden coercion dominated the full-history ranker.
    ts_arr = np.asarray(feats.attrs.get("ts_int64"))
    values = np.asarray(feats.attrs.get("feats_np"))
    if (
        ts_arr.dtype != np.dtype(np.int64)
        or values.dtype != np.dtype(np.float32)
        or ts_arr.ndim != 1
        or values.shape != (len(ts_arr), len(MULTI_TF_PER_BAR_FEATURES_V4))
    ):
        raise RuntimeError(f"[CTX_MTF_SOURCE] malformed {tf} cache arrays")
    right = int(
        np.searchsorted(
            ts_arr,
            _cache_cutoff_ns(
                ts_ns,
                tf,
                decision_bar_duration_ns=ctx.decision_bar_duration_ns,
            ),
            side="right",
        )
    )
    if right == 0:
        raise CausalContextWarmupError(
            f"[CTX_MTF_WARMUP] no closed {tf} row at {pd.Timestamp(ts_ns, tz='UTC')}"
        )
    warmup_rows = feats.attrs.get("causal_warmup_rows")
    if (
        isinstance(warmup_rows, bool)
        or not isinstance(warmup_rows, (int, np.integer))
        or not 0 <= int(warmup_rows) <= len(ts_arr)
    ):
        raise RuntimeError(f"[CTX_MTF_SOURCE] malformed {tf} warmup metadata")
    if right - 1 < int(warmup_rows):
        raise CausalContextWarmupError(
            f"[CTX_MTF_WARMUP] {tf} indicator row is not fully warmed at decision cutoff"
        )
    row = values[right - 1]
    if not np.isfinite(row).all():
        raise RuntimeError(f"[CTX_MTF_SOURCE] non-finite {tf} row at decision cutoff")
    return row


def _per_tf_all(ctx: AugmentContext, ts_ns: int) -> dict[str, float]:
    """O(log N) lookup of all 25 features per TF using V2 cache's int64 arrays."""
    out: dict[str, float] = {}
    for tf in TF_NAMES:
        row = _tf_cache_row(ctx, tf, ts_ns)
        for output_name, source_index in zip(
            MULTI_TF_V4_GROUP_A_BASE_FEATURES,
            _GROUP_A_MTF_SOURCE_INDICES,
            strict=True,
        ):
            out[_pertf_name(tf, output_name)] = float(row[source_index])
    return out


def _closed_m5_index(ctx: AugmentContext, ts_ns: int) -> int:
    """Return the latest M5 row closed at the native decision availability."""

    cutoff_ns = _cache_cutoff_ns(
        ts_ns,
        "M5",
        decision_bar_duration_ns=ctx.decision_bar_duration_ns,
    )
    right = int(np.searchsorted(ctx.m5_ts_ns, cutoff_ns, side="right"))
    if right == 0:
        raise CausalContextWarmupError(
            "[CTX_M5_WARMUP] no closed M5 row at decision availability"
        )
    return right - 1


def _pivots(ctx: AugmentContext, ts: pd.Timestamp, current_atr: float, current_price: float) -> dict[str, float]:
    """Lookup prior-day pivots — O(1) dict lookup."""
    ts = _require_utc_timestamp(ts, context="CTX_PIVOT")
    if not np.isfinite(current_atr) or current_atr <= 0.0 or not np.isfinite(current_price):
        raise RuntimeError("[CTX_PIVOT] price and ATR must be finite; ATR must be positive")
    # V30 package 3 (2026-08-13): the target day must be floored on the SAME
    # grid the pivot keys are labelled on (``_build_daily_pivots``).  With the
    # trading-day origin, ``ts.normalize()`` (midnight) would have selected the
    # bin opened at 22:00 the previous calendar day — which is still OPEN for
    # any decision before 22:00 — i.e. a lookahead.  ``multi_tf_bar_label``
    # returns the bin containing ``ts``; a strictly-earlier key therefore
    # closed at or before that bin's open, hence at or before ``ts``.
    target_day = multi_tf_bar_label(ts, "D1")
    eligible_days = [day for day in ctx.daily_pivot_by_date if day < target_day]
    if not eligible_days:
        raise CausalContextWarmupError("[CTX_PIVOT_WARMUP] no completed prior trading day")
    p = ctx.daily_pivot_by_date[max(eligible_days)]
    return {
        "dist_to_R1_atr": (current_price - p["R1"]) / current_atr,
        "dist_to_R2_atr": (current_price - p["R2"]) / current_atr,
        "dist_to_S1_atr": (current_price - p["S1"]) / current_atr,
        "dist_to_S2_atr": (current_price - p["S2"]) / current_atr,
    }


def _liquidity_zones(ctx: AugmentContext, ts_ns: int, current_price: float, current_atr: float) -> dict[str, float]:
    """Distance to nearest unswept high/low per TF — uses pre-resampled arrays."""
    if not np.isfinite(current_atr) or current_atr <= 0.0 or not np.isfinite(current_price):
        raise RuntimeError("[CTX_LIQUIDITY] price and ATR must be finite; ATR must be positive")
    out: dict[str, float] = {}
    for tf_name, ts_arr, hi_arr, lo_arr, lookback in (
        ("m5",  ctx.m5_ts_ns,  ctx.m5_high,  ctx.m5_low,  240),
        ("m15", ctx.m15_ts_ns, ctx.m15_high, ctx.m15_low, 192),
        ("h1",  ctx.h1_ts_ns,  ctx.h1_high,  ctx.h1_low,  168),
        ("h4",  ctx.h4_ts_ns,  ctx.h4_high,  ctx.h4_low,  168),
        ("d1",  ctx.d1_ts_ns,  ctx.d1_high,  ctx.d1_low,  60),
    ):
        cutoff_ns = _cache_cutoff_ns(
            ts_ns,
            tf_name.upper(),
            decision_bar_duration_ns=ctx.decision_bar_duration_ns,
        )
        right = int(np.searchsorted(ts_arr, cutoff_ns, side="right"))
        if right < lookback:
            raise CausalContextWarmupError(
                f"[CTX_LIQUIDITY_WARMUP] {tf_name.upper()} requires {lookback} closed rows; "
                f"observed={right}"
            )
        left = right - lookback
        window_hi = hi_arr[left:right]
        window_lo = lo_arr[left:right]
        # Positive means an unswept level still exists beyond price. If none
        # exists, the nearest already-swept level is retained with a negative
        # sign instead of a fabricated zero-distance level.
        highs_above = window_hi[window_hi > current_price]
        nearest_hi = float(highs_above.min()) if len(highs_above) else float(window_hi.max())
        out[f"dist_to_{tf_name}_hi_atr"] = float((nearest_hi - current_price) / current_atr)
        lows_below = window_lo[window_lo < current_price]
        nearest_lo = float(lows_below.max()) if len(lows_below) else float(window_lo.min())
        out[f"dist_to_{tf_name}_lo_atr"] = float((current_price - nearest_lo) / current_atr)
    return out


def _per_side_perf(ctx: AugmentContext, ts: pd.Timestamp, lookback_n: int = 10) -> dict[str, float]:
    """Lookup last-N closures before ts from pre-built trade history."""
    if isinstance(lookback_n, bool) or not isinstance(lookback_n, int) or lookback_n <= 0:
        raise RuntimeError("[CTX_PORTFOLIO] lookback_n must be a positive integer")
    if len(ctx.trade_history) == 0:
        raise CausalContextWarmupError("[CTX_PORTFOLIO_WARMUP] no closed-trade history")
    th = ctx.trade_history
    eligible = th[th["close_ts"] <= ts]
    if eligible.empty:
        raise CausalContextWarmupError("[CTX_PORTFOLIO_WARMUP] no trade closed by decision time")
    out: dict[str, float] = {}
    for side in ("long", "short"):
        side_df = eligible[eligible["side"] == side].tail(lookback_n)
        if side_df.empty:
            raise CausalContextWarmupError(
                f"[CTX_PORTFOLIO_WARMUP] no closed {side} trade by decision time"
            )
        pnls = side_df["pnl_bps"].to_numpy(dtype=np.float64)
        if not np.isfinite(pnls).all():
            raise RuntimeError(f"[CTX_PORTFOLIO] non-finite {side} pnl evidence")
        out[f"{side}_win_rate_last10"] = float((pnls > 0).mean())
        out[f"{side}_mean_pnl_last10"] = float(pnls.mean())
        # consec losses from end backwards
        n_consec = 0
        for p in reversed(pnls.tolist()):
            if p > 0:
                break
            n_consec += 1
        out[f"{side}_n_consec_losses"] = float(n_consec)
        mins = max(0.0, min(1440.0, (ts - side_df["close_ts"].iloc[-1]).total_seconds() / 60.0))
        out[f"{side}_time_since_last_close_min"] = float(mins)
    return out


def augment_candidate(
    ctx: AugmentContext,
    ts: pd.Timestamp,
    *,
    include_portfolio: bool = True,
) -> dict[str, float]:
    """Fast per-candidate compute for per-TF source and Group-A primitives."""
    if not isinstance(include_portfolio, bool):
        raise TypeError("[CTX_CANDIDATE] include_portfolio must be bool")
    ts = _require_utc_timestamp(ts, context="CTX_CANDIDATE")
    ts_ns = int(ts.value)
    decision_idx = int(np.searchsorted(ctx.decision_ts_ns, ts_ns, side="left"))
    if (
        decision_idx >= len(ctx.decision_ts_ns)
        or int(ctx.decision_ts_ns[decision_idx]) != ts_ns
    ):
        raise RuntimeError(
            f"[CTX_CANDIDATE] timestamp is not an exact native decision row: {ts}"
        )
    current_price = float(ctx.decision_close[decision_idx])
    m5_idx = _closed_m5_index(ctx, ts_ns)
    # Resolve exactly one row from each TF and reuse that common market
    # snapshot for the raw level-distance primitives.
    per_tf = _per_tf_all(ctx, ts_ns)
    atr_bps = float(per_tf[_pertf_name("M5", "atr_bps_14")])
    if not np.isfinite(current_price) or current_price <= 0.0:
        raise RuntimeError("[CTX_CANDIDATE] current local close must be finite and positive")
    if not np.isfinite(atr_bps):
        raise RuntimeError("[CTX_CANDIDATE] current M5 atr_bps_14 must be finite and positive")
    if atr_bps <= 0.0:
        raise CausalContextWarmupError("[CTX_CANDIDATE_WARMUP] current M5 ATR is not warmed")
    current_m5_close = float(ctx.m5_close[m5_idx])
    if not np.isfinite(current_m5_close) or current_m5_close <= 0.0:
        raise RuntimeError("[CTX_CANDIDATE] current closed M5 close is invalid")
    current_atr = atr_bps / 1e4 * current_m5_close

    out: dict[str, float] = {}
    liq = _liquidity_zones(ctx, ts_ns, current_price, current_atr)
    out.update(per_tf)                                         # 125
    out.update(_pivots(ctx, ts, current_atr, current_price))   # 4
    out.update(liq)                                            # 10 (5 TFs × hi/lo)
    if include_portfolio:
        out.update(_per_side_perf(ctx, ts))                    # 8
    required_group_a = set(GROUP_A_FEATURE_NAMES)
    if not include_portfolio:
        required_group_a.difference_update(PORTFOLIO_FEATURE_NAMES)
    required = set(PER_TF_FEATURE_NAMES) | required_group_a
    missing = sorted(required - set(out))
    if missing:
        raise RuntimeError(f"[CTX_CANDIDATE] derived feature contract incomplete: {missing}")
    values = np.asarray([out[name] for name in sorted(required)], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("[CTX_CANDIDATE] derived feature contract contains non-finite values")
    return out


def compute_attach_rows(
    ctx,
    ts_index: pd.DatetimeIndex,
    lo: int,
    hi: int,
    *,
    extract: list,
) -> dict:
    """Per-row augment_candidate loop for rows [lo, hi) against a FULL context.

    Extracted from attach_group_a_ctx_columns so callers can
    parallelize the row loop across workers while every worker indexes the
    SAME full-series context arrays (exact parity by construction; the
    trailing-1yr percentile arrays are precomputed in build_context from the
    complete frame). Rows outside [lo, hi) stay NaN in the returned arrays.
    """
    n = len(ts_index)
    cols = {k: np.full(n, np.nan, dtype=np.float32) for k in extract}
    complete_started = False
    for i in range(lo, hi):
        ts = ts_index[i]
        try:
            feat = augment_candidate(ctx, ts, include_portfolio=False)
        except CausalContextWarmupError:
            if complete_started:
                raise RuntimeError(
                    f"[CTX_CONT_PARITY] causal source gap after warmup at {ts}"
                )
            continue
        complete_started = True
        for k in extract:
            if k not in feat:
                raise RuntimeError(f"[CTX_CONT_PARITY] derived feature missing: {k}")
            value = float(feat[k])
            if not np.isfinite(value):
                raise RuntimeError(f"[CTX_CONT_PARITY] non-finite derived feature: {k}")
            cols[k][i] = value
    return cols


_PARALLEL_ATTACH_SHARED: tuple = ()

_GROUP_A_CHECKPOINT_SCHEMA_VERSION = "group_a_attach_checkpoint_v4"
_GROUP_A_CHECKPOINT_CHUNK_ROWS = 4096


def _sha256_bytes_iter(parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
    return digest.hexdigest()


def _group_a_array_bytes(value: np.ndarray) -> bytes:
    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return header + b"\0" + array.tobytes()


def _group_a_checkpoint_manifest(
    *,
    checkpoint_key: str,
    df: pd.DataFrame,
    ts_index: pd.DatetimeIndex,
    multi_tf: dict,
    extract: list[str],
    chunk_rows: int,
    context_m5: pd.DataFrame | None,
    decision_bar_duration_ns: int = _DECISION_M5_NS,
) -> dict:
    if not isinstance(checkpoint_key, str) or len(checkpoint_key) != 64 or any(
        ch not in "0123456789abcdef" for ch in checkpoint_key
    ):
        raise RuntimeError("[CTX_CONT_CHECKPOINT] checkpoint_key must be lowercase SHA-256")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise RuntimeError("[CTX_CONT_CHECKPOINT] chunk_rows must be a positive integer")
    frame_digest = hashlib.sha256()
    frame_digest.update(b"group_a_attach_frame_v1\0")
    for name, values in (
        ("time_ns", ts_index.asi8.astype(np.int64, copy=False)),
        ("high", pd.to_numeric(df["high"], errors="coerce").to_numpy(np.float64)),
        ("low", pd.to_numeric(df["low"], errors="coerce").to_numpy(np.float64)),
        ("close", pd.to_numeric(df["close"], errors="coerce").to_numpy(np.float64)),
    ):
        frame_digest.update(name.encode("utf-8") + b"\0")
        frame_digest.update(_group_a_array_bytes(values))

    history = _indexed_m5_ohlc_frame(
        df if context_m5 is None else context_m5,
        context="CTX_CONT_CHECKPOINT_HISTORY",
    )
    history_digest = hashlib.sha256()
    history_digest.update(b"group_a_context_m5_v1\0")
    for name, values in (
        ("time_ns", history.index.asi8.astype(np.int64, copy=False)),
        ("high", history["high"].to_numpy(np.float64)),
        ("low", history["low"].to_numpy(np.float64)),
        ("close", history["close"].to_numpy(np.float64)),
    ):
        history_digest.update(name.encode("utf-8") + b"\0")
        history_digest.update(_group_a_array_bytes(values))

    mtf_digest = hashlib.sha256()
    mtf_digest.update(b"group_a_attach_multi_tf_v1\0")
    if set(multi_tf) != set(TF_NAMES):
        raise RuntimeError("[CTX_CONT_CHECKPOINT] exact five-timeframe cache required")
    for tf_name in TF_NAMES:
        frame = multi_tf[tf_name]
        ts_values = np.asarray(frame.attrs.get("ts_int64"))
        feat_values = np.asarray(frame.attrs.get("feats_np"))
        mtf_digest.update(tf_name.encode("ascii") + b"\0")
        mtf_digest.update(_group_a_array_bytes(ts_values))
        mtf_digest.update(_group_a_array_bytes(feat_values))

    bounds = [
        [lo, min(lo + chunk_rows, len(df))]
        for lo in range(0, len(df), chunk_rows)
    ]
    return {
        "schema_version": _GROUP_A_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_key": checkpoint_key,
        "row_count": int(len(df)),
        "frame_sha256": frame_digest.hexdigest(),
        "context_m5_sha256": history_digest.hexdigest(),
        "context_m5_rows": int(len(history)),
        "context_m5_time_min_utc": history.index[0].isoformat(),
        "context_m5_time_max_utc": history.index[-1].isoformat(),
        "multi_tf_sha256": mtf_digest.hexdigest(),
        "extract": list(extract),
        "extract_sha256": _sha256_bytes_iter(
            [b"group_a_extract_v1\0", "\n".join(extract).encode("utf-8")]
        ),
        "decision_bar_duration_ns": int(decision_bar_duration_ns),
        "chunk_rows": int(chunk_rows),
        "bounds": bounds,
    }


def _canonical_json_bytes(payload: dict) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_exclusive_bytes(path: Path, encoded: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _initialize_group_a_checkpoint(
    checkpoint_dir: Path,
    manifest: dict,
) -> tuple[Path, str]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if checkpoint_dir.is_symlink():
        raise RuntimeError("[CTX_CONT_CHECKPOINT] checkpoint directory may not be a symlink")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint_dir.is_dir():
        raise RuntimeError("[CTX_CONT_CHECKPOINT] checkpoint path is not a directory")
    manifest_path = checkpoint_dir / "CHECKPOINT_MANIFEST.json"
    encoded = _canonical_json_bytes(manifest)
    if manifest_path.exists():
        if manifest_path.is_symlink() or manifest_path.read_bytes() != encoded:
            raise RuntimeError("[CTX_CONT_CHECKPOINT] manifest identity mismatch")
    else:
        unexpected = list(checkpoint_dir.iterdir())
        if unexpected:
            raise RuntimeError(
                f"[CTX_CONT_CHECKPOINT] files exist before manifest: {unexpected}"
            )
        _write_exclusive_bytes(manifest_path, encoded)
    return manifest_path, hashlib.sha256(encoded).hexdigest()


def _group_a_chunk_path(checkpoint_dir: Path, lo: int, hi: int) -> Path:
    return checkpoint_dir / f"chunk_{lo:09d}_{hi:09d}.npz"


def _write_group_a_chunk(
    path: Path,
    *,
    manifest_sha256: str,
    checkpoint_key: str,
    lo: int,
    hi: int,
    times_ns: np.ndarray,
    values: np.ndarray,
) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"[CTX_CONT_CHECKPOINT] stale partial file: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            np.savez(
                handle,
                schema_version=np.array(_GROUP_A_CHECKPOINT_SCHEMA_VERSION),
                manifest_sha256=np.array(manifest_sha256),
                checkpoint_key=np.array(checkpoint_key),
                lo=np.int64(lo),
                hi=np.int64(hi),
                times_ns=np.asarray(times_ns, dtype=np.int64),
                values=np.asarray(values, dtype=np.float32),
            )
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"[CTX_CONT_CHECKPOINT] chunk path already exists: {path}")
    os.replace(temporary, path)


def _load_group_a_chunk(
    path: Path,
    *,
    manifest_sha256: str,
    checkpoint_key: str,
    lo: int,
    hi: int,
    expected_times_ns: np.ndarray,
    width: int,
) -> np.ndarray:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"[CTX_CONT_CHECKPOINT] chunk is missing/non-regular: {path}")
    try:
        with np.load(path, allow_pickle=False) as payload:
            expected_keys = {
                "schema_version",
                "manifest_sha256",
                "checkpoint_key",
                "lo",
                "hi",
                "times_ns",
                "values",
            }
            if set(payload.files) != expected_keys:
                raise RuntimeError("chunk key set mismatch")
            values = np.asarray(payload["values"], dtype=np.float32)
            if (
                str(payload["schema_version"]) != _GROUP_A_CHECKPOINT_SCHEMA_VERSION
                or str(payload["manifest_sha256"]) != manifest_sha256
                or str(payload["checkpoint_key"]) != checkpoint_key
                or int(payload["lo"]) != lo
                or int(payload["hi"]) != hi
                or not np.array_equal(
                    np.asarray(payload["times_ns"], dtype=np.int64),
                    np.asarray(expected_times_ns, dtype=np.int64),
                )
                or values.shape != (hi - lo, width)
            ):
                raise RuntimeError("chunk identity/shape mismatch")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        raise RuntimeError(f"[CTX_CONT_CHECKPOINT] invalid chunk {path}: {exc}") from exc
    return values


def _compute_attach_rows_compact(
    ctx,
    ts_index: pd.DatetimeIndex,
    lo: int,
    hi: int,
    *,
    extract: list[str],
) -> np.ndarray:
    """Exact row loop returning only [lo, hi), avoiding full-N worker arrays."""
    values = np.full((hi - lo, len(extract)), np.nan, dtype=np.float32)
    complete_started = False
    for row, i in enumerate(range(lo, hi)):
        ts = ts_index[i]
        try:
            feat = augment_candidate(ctx, ts, include_portfolio=False)
        except CausalContextWarmupError:
            if complete_started:
                raise RuntimeError(
                    f"[CTX_CONT_PARITY] causal source gap after warmup at {ts}"
                )
            continue
        complete_started = True
        for column, name in enumerate(extract):
            if name not in feat:
                raise RuntimeError(f"[CTX_CONT_PARITY] derived feature missing: {name}")
            value = float(feat[name])
            if not np.isfinite(value):
                raise RuntimeError(f"[CTX_CONT_PARITY] non-finite derived feature: {name}")
            values[row, column] = value
    return values


def _parallel_attach_chunk_worker(args: tuple) -> tuple:
    """Row-loop worker over [lo, hi) against the fork-shared full context."""
    chunk_index, lo, hi = args
    ctx, ts_index, extract = _PARALLEL_ATTACH_SHARED
    values = _compute_attach_rows_compact(ctx, ts_index, lo, hi, extract=extract)
    return chunk_index, lo, hi, values


def attach_group_a_ctx_columns_parallel(
    df: pd.DataFrame,
    *,
    multi_tf: dict,
    journal_label: str = "parity",
    workers: int = 12,
    spot_check_rows: int = 40,
    checkpoint_dir: Path | None = None,
    checkpoint_key: str | None = None,
    checkpoint_chunk_rows: int = _GROUP_A_CHECKPOINT_CHUNK_ROWS,
    context_m5: pd.DataFrame | None = None,
    base_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """Exact parallel variant of :func:`attach_group_a_ctx_columns`.

    ONE full-series context is built in the parent (long-memory arrays such as
    the trailing-1yr percentile arrays are therefore identical for every
    worker) and only the zero-copy augment_candidate loop is fanned out over
    fork()ed workers — exact by construction, no chunk overlap. After the
    merge, spot_check_rows evenly spaced finite rows are recomputed serially
    and asserted bit-exact against the merged result. ``context_m5`` supplies
    causal prehistory when ``df`` is only an emission/history slice; its OHLC
    and identity are validated and checkpoint-bound.
    """
    import multiprocessing as mp

    global _PARALLEL_ATTACH_SHARED
    ctx, ts_index, extract = build_attach_context(
        df,
        multi_tf=multi_tf,
        journal_label=journal_label,
        context_m5=context_m5,
        base_bar_duration=base_bar_duration,
    )
    if (checkpoint_dir is None) != (checkpoint_key is None):
        raise RuntimeError(
            "[CTX_CONT_CHECKPOINT] checkpoint_dir and checkpoint_key are jointly required"
        )
    n = len(df)
    workers = max(1, min(int(workers), n))
    chunk_rows = int(checkpoint_chunk_rows)
    bounds = [
        (index, lo, min(lo + chunk_rows, n))
        for index, lo in enumerate(range(0, n, chunk_rows))
    ]
    checkpoint_path = None
    manifest_sha256 = None
    manifest = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
        manifest = _group_a_checkpoint_manifest(
            checkpoint_key=str(checkpoint_key),
            df=df,
            ts_index=ts_index,
            multi_tf=multi_tf,
            extract=extract,
            chunk_rows=chunk_rows,
            context_m5=context_m5,
            decision_bar_duration_ns=int(base_bar_duration.value),
        )
        _, manifest_sha256 = _initialize_group_a_checkpoint(
            checkpoint_path, manifest
        )
        expected_names = {
            "CHECKPOINT_MANIFEST.json",
            "CHECKPOINT_COMPLETE.json",
            *{
                _group_a_chunk_path(checkpoint_path, lo, hi).name
                for _, lo, hi in bounds
            },
        }
        entries = list(checkpoint_path.iterdir())
        partials = sorted(
            path for path in entries if path.name.startswith(".") and ".partial-" in path.name
        )
        if partials:
            raise RuntimeError(
                f"[CTX_CONT_CHECKPOINT] interrupted partial chunks require a fresh event: {partials}"
            )
        unexpected = sorted(path for path in entries if path.name not in expected_names)
        if unexpected:
            raise RuntimeError(
                f"[CTX_CONT_CHECKPOINT] unexpected checkpoint entries: {unexpected}"
            )

    cols = {k: np.full(n, np.nan, dtype=np.float32) for k in extract}
    pending = []
    completed_paths: list[Path] = []
    for chunk_index, lo, hi in bounds:
        chunk_path = (
            _group_a_chunk_path(checkpoint_path, lo, hi)
            if checkpoint_path is not None
            else None
        )
        if chunk_path is not None and chunk_path.exists():
            values = _load_group_a_chunk(
                chunk_path,
                manifest_sha256=str(manifest_sha256),
                checkpoint_key=str(checkpoint_key),
                lo=lo,
                hi=hi,
                expected_times_ns=ts_index.asi8[lo:hi],
                width=len(extract),
            )
            for column, name in enumerate(extract):
                cols[name][lo:hi] = values[:, column]
            completed_paths.append(chunk_path)
        else:
            pending.append((chunk_index, lo, hi))

    _PARALLEL_ATTACH_SHARED = (ctx, ts_index, extract)
    try:
        if workers == 1:
            results = map(_parallel_attach_chunk_worker, pending)
            pool = None
        else:
            mp_ctx = mp.get_context("fork")
            pool = mp_ctx.Pool(processes=workers)
            results = pool.imap_unordered(
                _parallel_attach_chunk_worker, pending, chunksize=1
            )
        try:
            for _, lo, hi, values in results:
                if checkpoint_path is not None:
                    chunk_path = _group_a_chunk_path(checkpoint_path, lo, hi)
                    _write_group_a_chunk(
                        chunk_path,
                        manifest_sha256=str(manifest_sha256),
                        checkpoint_key=str(checkpoint_key),
                        lo=lo,
                        hi=hi,
                        times_ns=ts_index.asi8[lo:hi],
                        values=values,
                    )
                    completed_paths.append(chunk_path)
                for column, name in enumerate(extract):
                    cols[name][lo:hi] = values[:, column]
        except BaseException:
            if pool is not None:
                pool.terminate()
                pool.join()
            raise
        else:
            if pool is not None:
                pool.close()
                pool.join()
    finally:
        _PARALLEL_ATTACH_SHARED = ()

    complete_path = None
    complete_sha256 = None
    if checkpoint_path is not None:
        expected_chunk_paths = [
            _group_a_chunk_path(checkpoint_path, lo, hi) for _, lo, hi in bounds
        ]
        if sorted(set(completed_paths)) != sorted(expected_chunk_paths):
            raise RuntimeError("[CTX_CONT_CHECKPOINT] incomplete exact chunk set")
        chunk_hashes = []
        for path in expected_chunk_paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            chunk_hashes.append(
                {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
            )
        complete_payload = {
            "schema_version": _GROUP_A_CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_key": checkpoint_key,
            "checkpoint_manifest_path": str(checkpoint_path / "CHECKPOINT_MANIFEST.json"),
            "checkpoint_manifest_sha256": manifest_sha256,
            "row_count": n,
            "chunk_count": len(expected_chunk_paths),
            "chunks": chunk_hashes,
        }
        complete_path = checkpoint_path / "CHECKPOINT_COMPLETE.json"
        encoded = _canonical_json_bytes(complete_payload)
        if complete_path.exists():
            if complete_path.is_symlink() or complete_path.read_bytes() != encoded:
                raise RuntimeError("[CTX_CONT_CHECKPOINT] completion identity mismatch")
        else:
            _write_exclusive_bytes(complete_path, encoded)
        complete_sha256 = hashlib.sha256(encoded).hexdigest()

    finite = np.flatnonzero(np.isfinite(cols[extract[0]]))
    if len(finite) == 0:
        raise RuntimeError(
            "[CTX_CONT_PARITY] parallel attach produced no finite rows"
        )
    picks = finite[
        np.linspace(0, len(finite) - 1, min(int(spot_check_rows), len(finite)))
        .astype(int)
    ]
    for i in picks:
        serial = compute_attach_rows(
            ctx, ts_index, int(i), int(i) + 1, extract=extract
        )
        for k in extract:
            a = np.float32(serial[k][int(i)])
            b = cols[k][int(i)]
            if not (a == b or (np.isnan(a) and np.isnan(b))):
                raise RuntimeError(
                    "[CTX_CONT_PARITY] parallel attach spot-check mismatch: "
                    f"row={int(i)} col={k} serial={a} parallel={b}"
                )
    result = finalize_attach_columns(df, cols)
    if complete_path is not None:
        result.attrs["group_a_checkpoint_complete_path"] = str(complete_path)
        result.attrs["group_a_checkpoint_complete_sha256"] = complete_sha256
        result.attrs["group_a_checkpoint_key"] = checkpoint_key
    return result


def attach_group_a_ctx_columns(
    df: pd.DataFrame,
    *,
    multi_tf: dict,
    journal_label: str = "parity",
    context_m5: pd.DataFrame | None = None,
    base_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """Add the 24 raw GROUP-A ctx_cont primitives to ``df`` in place.

    ONE TRUTH for the V10 entry builder, the V3 exit builder, and the
    inference-batch candidate generator (serving). All three compute these
    ctx_cont features identically from :func:`augment_candidate`, so the model
    sees the same values at train and inference time (no train/serve skew).

    Requirements:
      - ``df`` has lowercase ``high``/``low``/``close`` columns.
      - ``df`` has a ``time`` column OR a tz-aware ``DatetimeIndex``.

    Existing derived columns are never trusted or passed through: they are
    overwritten from the exact sources on every call. ``multi_tf`` is explicit
    and mandatory, so serving cannot silently fall back to a stale disk cache.
    ``context_m5`` is mandatory for a non-M5 local clock. It supplies the one
    real M5/M15/H1/H4/D1 context well while local prices remain native to the
    decision lane.
    """
    ctx, ts_index, extract = build_attach_context(
        df,
        multi_tf=multi_tf,
        journal_label=journal_label,
        context_m5=context_m5,
        base_bar_duration=base_bar_duration,
    )
    cols = compute_attach_rows(ctx, ts_index, 0, len(df), extract=extract)
    return finalize_attach_columns(df, cols)


def build_attach_context(
    df: pd.DataFrame,
    *,
    multi_tf: dict,
    journal_label: str = "parity",
    context_m5: pd.DataFrame | None = None,
    base_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
):
    """Validate ``df`` and build the FULL-series augment context.

    Factored from attach_group_a_ctx_columns so a parallel caller
    can build the context once (full series -> exact trailing-1yr arrays) and
    fan the row loop out over workers. A separate ``context_m5`` prevents a
    decision slice from resetting 60-D1-bar liquidity and pivot state, and
    prevents native M1 rows from ever being mislabeled as M5 context.
    """
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS as _GROUP_A,
    )
    if not isinstance(multi_tf, dict):
        raise RuntimeError("[CTX_CONT_PARITY] explicit multi_tf source is required")

    decision_local = _indexed_m5_ohlc_frame(
        df, context="CTX_CONT_PARITY_LOCAL"
    )
    ts_index = decision_local.index
    if base_bar_duration != pd.Timedelta(minutes=5) and context_m5 is None:
        raise RuntimeError(
            "[CTX_CONT_PARITY] explicit true-M5 context is required for a non-M5 lane"
        )
    m5 = (
        decision_local
        if context_m5 is None
        else _indexed_m5_ohlc_frame(context_m5, context="CTX_CONT_FULL_HISTORY")
    )
    if base_bar_duration == pd.Timedelta(minutes=5) and context_m5 is not None:
        decision_positions = m5.index.get_indexer(ts_index)
        if np.any(decision_positions < 0):
            missing_time = ts_index[int(np.flatnonzero(decision_positions < 0)[0])]
            raise RuntimeError(
                f"[CTX_CONT_FULL_HISTORY] decision timestamp absent from context: {missing_time}"
            )
        if not np.array_equal(
            m5.iloc[decision_positions].to_numpy(dtype=np.float64),
            decision_local.to_numpy(dtype=np.float64),
        ):
            raise RuntimeError(
                "[CTX_CONT_FULL_HISTORY] decision OHLC differs from the context source"
            )
    ctx = build_context(
        m5, multi_tf,
        journal_dir=Path(f"/nonexistent_{journal_label}_journal"),
        decision_frame=decision_local,
        decision_bar_duration=base_bar_duration,
    )

    return ctx, ts_index, list(_GROUP_A)


def finalize_attach_columns(
    df: pd.DataFrame,
    cols: dict,
) -> pd.DataFrame:
    """Assemble the attach output columns onto ``df`` (post-row-loop steps)."""
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS as _GROUP_A,
    )
    out_cols = {k: cols[k] for k in _GROUP_A}
    values = np.column_stack([out_cols[name] for name in _GROUP_A])
    valid = np.isfinite(values).all(axis=1)
    if not valid.any():
        raise RuntimeError("[CTX_CONT_PARITY] no complete causal context row exists")
    result = df.copy(deep=False)
    for column, values in out_cols.items():
        result[column] = values
    result.attrs.update(df.attrs)
    result.attrs["causal_context_warmup_rows"] = int(np.count_nonzero(~valid))
    return result


def augment_week(week_pq: Path, out_pq: Path, ctx: AugmentContext,
                 smc_cache: pd.DataFrame | None = None) -> dict:
    df = pd.read_parquet(week_pq)
    n = len(df)
    if n == 0:
        out_pq.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_pq, index=False)
        return {"week": week_pq.stem, "n": 0, "skipped": True}
    if "decision_ts_utc" not in df.columns:
        raise RuntimeError(f"{week_pq} missing decision_ts_utc")
    # 2026-05-24: drop existing V2/group-A/SMC cols if input was pre-augmented
    # (prevents duplicate cols when re-running on V10V2_AUGMENTED with bug-fixes).
    cols_to_overwrite = (
        set(PER_TF_FEATURE_NAMES)
        | set(GROUP_A_FEATURE_NAMES)
        | set(GROUP_S_SMC_FEATURE_NAMES)
    )
    existing_to_drop = [c for c in df.columns if c in cols_to_overwrite]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)
    ts_arr = pd.to_datetime(df["decision_ts_utc"], utc=True)
    new_cols: dict[str, list[float]] = {
        name: [] for name in (PER_TF_FEATURE_NAMES + GROUP_A_FEATURE_NAMES)
    }
    t0 = time.time()
    for ts in ts_arr:
        d = augment_candidate(ctx, ts)
        for k, v in d.items():
            new_cols[k].append(v)
    # Build new cols as a DataFrame, then concat once (avoids fragmentation warning).
    new_df = pd.DataFrame({k: np.asarray(vals, dtype=np.float32) for k, vals in new_cols.items()},
                          index=df.index)
    df = pd.concat([df, new_df], axis=1)

    # 2026-05-24 BUG-3 FIX: join SMC features from canonical_v3 by decision_ts_utc.
    # smc_cache is the canonical_v3 SMC slice (time + smc_* cols), loaded once
    # in main() and passed in. We asof-join by time (canonical M5 bars).
    if smc_cache is not None and len(smc_cache) > 0:
        # Build asof-join: each candidate ts → nearest preceding M5 bar in canonical
        pd.to_datetime(df["decision_ts_utc"], utc=True).sort_values()
        cand_idx = pd.to_datetime(df["decision_ts_utc"], utc=True).reset_index(drop=True)
        smc_sorted = smc_cache.sort_values("time").reset_index(drop=True)
        merged = pd.merge_asof(
            cand_idx.to_frame("decision_ts_utc").sort_values("decision_ts_utc"),
            smc_sorted, left_on="decision_ts_utc", right_on="time",
            direction="backward", tolerance=pd.Timedelta("5min"),
        )
        # Reorder back to original df order
        merged = merged.set_index("decision_ts_utc").reindex(cand_idx).reset_index(drop=True)
        smc_sources = [
            "smc_swing_state", "smc_bos_up", "smc_bos_down", "smc_choch",
            "smc_sweep_up", "smc_sweep_down",
            "smc_sweep_event_age_bars",
            "smc_pivot_envelope_position",
        ]
        missing_smc = [name for name in smc_sources if name not in merged.columns]
        if missing_smc:
            raise RuntimeError(f"[AUG_V2_SMC] canonical SMC sources missing: {missing_smc}")
        for src_col, dst_col in zip(smc_sources, GROUP_S_SMC_FEATURE_NAMES):
            values = pd.to_numeric(merged[src_col], errors="coerce").to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise RuntimeError(f"[AUG_V2_SMC] missing/non-finite asof values for {src_col}")
            df[dst_col] = values.astype(np.float32)
    else:
        raise RuntimeError("[AUG_V2_SMC] explicit non-empty canonical SMC cache is required")

    out_pq.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_pq, index=False)
    elapsed = time.time() - t0
    return {"week": week_pq.stem, "n": n, "elapsed_sec": elapsed,
            "rate_per_sec": n / elapsed if elapsed > 0 else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-outcome-dir", type=Path, required=True,
                    help="explicit forward-outcome dir (no silent stale default; pass the current set)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--m5-prebuilt", type=Path, default=M5_PREBUILT)
    ap.add_argument("--n-weeks-test", type=int, default=0)
    ap.add_argument(
        "--v29-registry-constants-manifest",
        type=Path,
        required=True,
        help=(
            "Explicit JSON artifact carrying the frozen TRAIN-fitted V29 "
            "registry constants (the authoritative V4 cache manifest.json; "
            "bare payloads are forbidden); no default exists"
        ),
    )
    ap.add_argument("--volatility-squeeze-manifest", type=Path, required=True)
    ap.add_argument(
        "--expected-volatility-squeeze-manifest-sha256",
        required=True,
    )
    args = ap.parse_args()

    out_dir = args.out_dir or args.forward_outcome_dir.with_name(
        args.forward_outcome_dir.name + "_AUGMENTED_V2"
    )
    out_per_week = out_dir / "per_week"
    out_per_week.mkdir(parents=True, exist_ok=True)
    existing_outputs = sorted(out_per_week.glob("*.parquet"))
    existing_manifests = sorted(out_dir.glob("manifest*.json"))
    if existing_outputs or existing_manifests:
        raise RuntimeError(
            "[AUG_V2_OUTPUT_NOT_CLEAN] refusing stale/pass-through artifacts; use a new empty "
            f"--out-dir (weekly={len(existing_outputs)} manifests={len(existing_manifests)})"
        )
    print(f"[AUG_V2] source: {args.forward_outcome_dir}")
    print(f"[AUG_V2] output: {out_dir}")
    print(f"[AUG_V2] new cols per row: {len(PER_TF_FEATURE_NAMES) + len(GROUP_A_FEATURE_NAMES)} "
          f"(125 per-TF + 28 group-A)")

    print("[AUG_V2] loading M5 prebuilt + building V2 multi-TF cache...")
    t0 = time.time()
    m5_df = pd.read_parquet(args.m5_prebuilt, columns=["time", "open", "high", "low", "close", "volume"])
    m5_df["time"] = pd.to_datetime(m5_df["time"], utc=True)
    m5_df = m5_df.set_index("time").sort_index()
    from gx1.features.htf_features import load_v29_registry_constants_manifest
    from gx1.features.volatility_squeeze_state_v1 import (
        load_volatility_squeeze_artifact_manifest,
    )

    multi_tf = build_multi_tf_per_bar_features_v4(
        m5_df,
        v29_registry_constants=load_v29_registry_constants_manifest(
            args.v29_registry_constants_manifest
        ),
        volatility_squeeze_artifacts=load_volatility_squeeze_artifact_manifest(
            args.volatility_squeeze_manifest.expanduser().resolve(strict=True),
            expected_sha256=args.expected_volatility_squeeze_manifest_sha256,
        ),
    )
    print(f"[AUG_V2]   M5={len(m5_df):,} bars  multi-TF in {time.time()-t0:.1f}s")

    print("[AUG_V2] building augment context (one-shot pre-compute)...")
    t1 = time.time()
    ctx = build_context(m5_df, multi_tf, JOURNAL_DIR)
    print(f"[AUG_V2]   context built in {time.time()-t1:.1f}s "
          f"(trade_history rows: {len(ctx.trade_history)})")

    # 2026-05-24 BUG-3 FIX: load SMC features from canonical_v3 prebuilt
    print("[AUG_V2] loading SMC features from canonical_v3 prebuilt...")
    t_smc = time.time()
    smc_cols = ["time"] + [
        "smc_swing_state","smc_bos_up","smc_bos_down","smc_choch",
        "smc_sweep_up","smc_sweep_down",
        "smc_sweep_event_age_bars",
        "smc_pivot_envelope_position",
    ]
    smc_cache = pd.read_parquet(args.m5_prebuilt, columns=smc_cols)
    smc_cache["time"] = pd.to_datetime(smc_cache["time"], utc=True, errors="coerce")
    smc_times = pd.DatetimeIndex(smc_cache["time"])
    if (
        smc_cache.empty
        or smc_times.hasnans
        or not smc_times.is_unique
        or not smc_times.is_monotonic_increasing
        or not smc_times.equals(m5_df.index)
    ):
        raise RuntimeError("[AUG_V2_SMC] SMC timestamps must exactly equal canonical M5 source")
    smc_values = smc_cache[smc_cols[1:]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    if not np.isfinite(smc_values).all():
        raise RuntimeError("[AUG_V2_SMC] canonical SMC sources must be finite")
    swing = smc_values[:, 0]
    if not np.equal(swing, np.rint(swing)).all() or np.any((swing < 0.0) | (swing > 4.0)):
        raise RuntimeError("[AUG_V2_SMC] smc_swing_state must use exact enum 0..4")
    print(f"[AUG_V2]   SMC cache loaded: {len(smc_cache):,} rows × {len(smc_cols)-1} cols "
          f"in {time.time()-t_smc:.1f}s")

    week_files = sorted((args.forward_outcome_dir / "per_week").glob("forward_outcomes_*.parquet"))
    if args.n_weeks_test > 0:
        week_files = week_files[:args.n_weeks_test]
        print(f"[AUG_V2] SMOKE TEST: first {args.n_weeks_test} weeks")
    print(f"[AUG_V2] processing {len(week_files)} weekly parquets...")

    total_n = 0
    total_t = 0.0
    week_rows: dict[str, int] = {}
    errors: list[str] = []
    for i, wp in enumerate(week_files):
        out_pq = out_per_week / wp.name
        try:
            s = augment_week(wp, out_pq, ctx, smc_cache=smc_cache)
            week_rows[wp.name] = int(s["n"])
            total_n += s["n"]
            total_t += s.get("elapsed_sec", 0)
            if (i+1) % 25 == 0 or i+1 == len(week_files):
                rate = total_n / max(total_t, 1e-6)
                print(f"  [{i+1}/{len(week_files)}] {wp.stem}  n={s['n']:>4}  "
                      f"({rate:.0f} cand/s, {total_n:,} done)", flush=True)
        except Exception as exc:
            errors.append(f"{wp.name}: {exc}")
            print(f"  [{i+1}/{len(week_files)}] {wp.stem}  ERROR: {exc}", flush=True)
        gc.collect()

    print(f"\n[AUG_V2] DONE — {total_n:,} candidates in {total_t/60:.1f} min "
          f"({total_n/max(total_t,1e-6):.0f} cand/s)")
    print(f"[AUG_V2] output → {out_dir}")
    # 2026-06-11: run-manifest (rule 4 — the step1feats build had NO manifest and had to be
    # verified forensically after a reboot wiped the only log). Records source, env-gates,
    # commit, per-week rowcounts and errors; status != DONE means PARTIAL — do not consume.
    import json as _json
    import subprocess as _sp
    try:
        _commit = _sp.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                          capture_output=True, text=True).stdout.strip() or "unknown"
    except Exception:
        _commit = "unknown"
    manifest = {
        "schema": "AUG_V2_MANIFEST_V1",
        "built_utc": pd.Timestamp.utcnow().isoformat(),
        "commit": _commit,
        "source_forward_outcome_dir": str(args.forward_outcome_dir),
        "m5_prebuilt": str(args.m5_prebuilt),
        "feature_contract": "fixed_group_a_32_no_ambient_gates_v2",
        "n_week_files_seen": len(week_files),
        "n_built": len(week_rows),
        "n_skipped_existing": 0,
        "rows_built_total": total_n,
        "per_week_rows": week_rows,
        "errors": errors,
        "status": "DONE" if not errors else "FAILED_PARTIAL",
    }
    (out_dir / "manifest_v1.json").write_text(_json.dumps(manifest, indent=2))
    print(f"[AUG_V2] manifest → {out_dir / 'manifest_v1.json'}  status={manifest['status']}")
    if errors:
        print(f"[AUG_V2] FAIL-LOUD: {len(errors)} week(s) errored — output is PARTIAL, do not consume.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
