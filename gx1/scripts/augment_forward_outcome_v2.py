#!/usr/bin/env python3
"""V2 Phase B5 — augment forward-outcome parquets with 125 per-TF + 28 group-A V2 scalars.

C+prune strategy: embed ALL 25 V2-features per TF (5 TFs × 25 = 125), plus 28
group-A features. Permutation importance later (Phase C4b/C5b) prunes weak ones.

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
import math
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
    build_multi_tf_per_bar_features_v2,
    HTF_V2_MATRIX_CONTRACT,
    MULTI_TF_PER_BAR_FEATURES_V2,
    MULTI_TF_RESAMPLE_RULES,
    validate_causal_feature_matrix,
)
# Round-number proximity: import the ONE-TRUTH compute (group_a_features) — no duplicated math.
from gx1.features.group_a_features import (
    round_number_levels as _round_number_levels, ROUND_FEATURE_NAMES as _ROUND_FEATURE_NAMES,
)
_AUG_ROUND_ON = os.environ.get("GX1_ROUND_NUMBER", "0") == "1"

TF_NAMES = ("M5", "M15", "H1", "H4", "D1")


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

def _cache_cutoff_ns(ts_ns: int, tf: str) -> int:
    """Return the latest start-stamped ``tf`` row closed at decision time."""
    if tf not in _TF_SHIFT_NS:
        raise RuntimeError(f"[CTX_CAUSALITY] unsupported timeframe: {tf!r}")
    return ts_ns + _DECISION_M5_NS - _TF_SHIFT_NS[tf]


def compute_smc_swing_dip_interaction(
    smc_swing_state: np.ndarray | pd.Series,
    dip_proximity_m5: np.ndarray | pd.Series,
) -> np.ndarray:
    """Compute the categorical SMC-state×dip interaction without frame fit.

    ``smc_swing_state`` is the canonical fixed enum 0..4. The retired
    implementation divided it by ``max(abs(state))`` over the supplied frame;
    appending a future state therefore rewrote historical model inputs. A fixed
    denominator of four preserves the established full-enum encoding while
    making every row prefix-invariant.
    """
    state = np.asarray(smc_swing_state, dtype=np.float64)
    dip = np.asarray(dip_proximity_m5, dtype=np.float64)
    if state.ndim != 1 or dip.ndim != 1 or state.shape != dip.shape or state.size == 0:
        raise RuntimeError(
            "[CTX_CAUSALITY] smc_swing_state and dip_proximity_m5 must be "
            "non-empty equal-length 1D arrays"
        )
    if not np.isfinite(state).all() or not np.isfinite(dip).all():
        raise RuntimeError("[CTX_CAUSALITY] SMC×dip sources must be finite")
    if not np.equal(state, np.rint(state)).all() or np.any((state < 0.0) | (state > 4.0)):
        raise RuntimeError("[CTX_CAUSALITY] smc_swing_state must use exact enum values 0..4")
    if np.any((dip < 0.0) | (dip > 1.0)):
        raise RuntimeError("[CTX_CAUSALITY] dip_proximity_m5 must be within [0, 1]")
    return ((state / 4.0) * dip).astype(np.float32)


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
    trimmed = frame.iloc[first_valid:].copy()
    trimmed.attrs.update(frame.attrs)
    trimmed.attrs["causal_context_warmup_rows_trimmed"] = first_valid
    return trimmed

PER_TF_FEATURE_NAMES = tuple(
    _pertf_name(tf, feat) for tf in TF_NAMES for feat in MULTI_TF_PER_BAR_FEATURES_V2
)
# 5 × 25 = 125

GROUP_A_FEATURE_NAMES = (
    "is_asia_eu_overlap", "is_eu_us_overlap", "is_eu_only", "is_us_only",
    "atr_ratio_m5_h4", "atr_ratio_m15_d1", "atr_ratio_h1_d1", "atr_ratio_m5_m15",
    "vol_pct_m5_1yr", "vol_pct_h1_1yr",
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
if _AUG_ROUND_ON:  # +5 ONLY under GX1_ROUND_NUMBER → default-OFF keeps the Group-A name set byte-identical;
    # the zero-fill (augment_candidate m5_idx<0), new_cols dict (augment_week), and attach_group_a copy-set all
    # iterate GROUP_A_FEATURE_NAMES, so appending here wires every code path in one place.
    GROUP_A_FEATURE_NAMES = GROUP_A_FEATURE_NAMES + _ROUND_FEATURE_NAMES

# FVG (fair-value-gap) 3-bar imbalance proximity per-TF M5/M15/H1 (2026-06-11, env-gated GX1_FVG_FEATURES,
# default OFF). EMPTY tuple when OFF → the zero-fill / new_cols / cols_to_overwrite paths add nothing →
# byte-identical to cement. (H4/D1 gaps too sparse for shuffled-liveness → M5/M15/H1 only.)
_FVG_ON = os.environ.get("GX1_FVG_FEATURES", "0") == "1"
FVG_FEATURE_NAMES = tuple(
    f"{tf}_{c}" for tf in ("m5", "m15", "h1")
    for c in ("dist_to_unfilled_fvg_atr", "fvg_active")
) if _FVG_ON else ()

# 2026-05-24 GROUP_S: SMC features from canonical_v3 (joined by decision_ts).
# These ARE in canonical_v3 prebuilt with real signal (smc_choch 421 non-zero,
# smc_bos_down 76K, smc_sweep_up/down 25K each) but were NEVER joined into
# forward_outcome → all zero downstream. Fix: load canonical_v3 once, asof-join
# by time to add SMC cols (with _canon_v1 suffix to match downstream expectations).
# 2026-05-24 PM: 5-TF dip + structure features (computed in augment_candidate).
# Persisted into fwd parquet so both Entry-IQL and Exit-IQL state builders can
# read them via column lookup (no need to recompute at training time).
DIP_STRUCT_FEATURE_NAMES = tuple(
    [f"dip_proximity_{tf}_v3"  for tf in ("m5","m15","h1","h4","d1")]
  + [f"dip_confirmed_{tf}_v3"  for tf in ("m5","m15","h1","h4","d1")]
  + [f"struct_continuation_up_{tf}_v3"      for tf in ("m5","m15","h1","h4","d1")]
  + [f"struct_pullback_in_uptrend_{tf}_v3"  for tf in ("m5","m15","h1","h4","d1")]
  + [f"struct_continuation_down_{tf}_v3"    for tf in ("m5","m15","h1","h4","d1")]
  + [f"struct_bounce_in_downtrend_{tf}_v3"  for tf in ("m5","m15","h1","h4","d1")]
  + [f"struct_pullback_depth_{tf}_v3"       for tf in ("m5","m15","h1","h4","d1")]
  + ["struct_all_tf_pullback_v3", "struct_tf_agree_count_v3", "struct_dip_x_uptrend_v3"]
)  # 5×7 + 3 = 38 cols (smc_swing_x_dip computed downstream after SMC join)

GROUP_S_SMC_FEATURE_NAMES = (
    "smc_swing_state_canon_v1",
    "smc_bos_up_canon_v1",
    "smc_bos_down_canon_v1",
    "smc_choch_canon_v1",
    "smc_sweep_up_canon_v1",
    "smc_sweep_down_canon_v1",
    "smc_sweep_size_atr_canon_v1",
    "smc_bars_since_sweep_canon_v1",
    "smc_premium_discount_canon_v1",
    "smc_premium_state_canon_v1",
)
ASIA_HOURS = set(list(range(22, 24)) + list(range(0, 9)))
EU_HOURS   = set(range(7, 17))
US_HOURS   = set(range(13, 22))

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
    # Pre-computed ATR percentile arrays — at each M5 bar, current ATR vs trailing-1yr
    m5_atr_pct_1yr: np.ndarray
    h1_atr_pct_1yr: np.ndarray      # per-H1-bar
    h1_atr_pct_ts_ns: np.ndarray
    # D1 pivot levels per M5 (R1/R2/S1/S2 from prior D1 OHLC)
    # store as per-day arrays: lookup via date
    daily_pivot_by_date: dict       # date_str → {R1, R2, S1, S2}
    # Trade history from journal — DataFrame with (close_ts, side, pnl_bps)
    trade_history: pd.DataFrame


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


def _build_resampled_ohlc_array(df: pd.DataFrame, rule: str) -> tuple:
    """Return (ts_int64, high, low) for resampled bars."""
    resamp = df.resample(rule).agg({"high": "max", "low": "min"}).dropna()
    ts_ns = resamp.index.values.astype("datetime64[ns]").astype(np.int64)
    return ts_ns, resamp["high"].to_numpy(np.float64), resamp["low"].to_numpy(np.float64)


def _build_atr_percentile_array(df: pd.DataFrame, ts_ns: np.ndarray, window_days: int = 365) -> np.ndarray:
    """For each row, percentile-rank of current ATR vs last `window_days` ATRs.

    Uses Wilder ATR-14. Returns array of percentiles in [0, 1].
    """
    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days <= 0:
        raise RuntimeError("[CTX_VOL_PERCENTILE] window_days must be a positive integer")
    numeric = _validate_m5_frame(df, context="CTX_VOL_PERCENTILE")
    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    if ts_ns.ndim != 1 or len(ts_ns) != len(numeric):
        raise RuntimeError("[CTX_VOL_PERCENTILE] timestamp/source length mismatch")
    if np.any(np.diff(ts_ns) <= 0):
        raise RuntimeError("[CTX_VOL_PERCENTILE] timestamps must be strictly chronological")
    high, low, close = numeric["high"], numeric["low"], numeric["close"]
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean().to_numpy(np.float64)
    if not np.isfinite(atr).all() or np.any(atr <= 0.0):
        raise RuntimeError("[CTX_VOL_PERCENTILE] causal ATR must be finite and positive")

    # Exact O(N log N) trailing rank.  The retired strided approximation left
    # the entire first window at a fabricated 0.5 and carried old ranks between
    # strides.  Coordinate compression + a Fenwick tree provides the strict
    # ``count(previous/current ATR < current ATR) / window_count`` definition
    # for every row without consulting any future value.
    ranks = np.searchsorted(np.unique(atr), atr).astype(np.int64, copy=False)
    tree = np.zeros(int(ranks.max()) + 2, dtype=np.int64)

    def _add(rank: int, delta: int) -> None:
        node = rank + 1
        while node < len(tree):
            tree[node] += delta
            node += node & -node

    def _count_less(rank: int) -> int:
        total = 0
        node = rank
        while node > 0:
            total += int(tree[node])
            node -= node & -node
        return total

    pct = np.empty(len(atr), dtype=np.float32)
    left = 0
    lookback_ns = int(pd.Timedelta(days=window_days).value)
    for i, rank in enumerate(ranks):
        cutoff = int(ts_ns[i]) - lookback_ns
        while left < i and int(ts_ns[left]) < cutoff:
            _add(int(ranks[left]), -1)
            left += 1
        _add(int(rank), 1)
        count = i - left + 1
        pct[i] = np.float32(_count_less(int(rank)) / count)
    if not np.isfinite(pct).all() or np.any((pct < 0.0) | (pct > 1.0)):
        raise RuntimeError("[CTX_VOL_PERCENTILE] invalid causal percentile output")
    return pct


def _build_daily_pivots(df: pd.DataFrame) -> dict[pd.Timestamp, dict[str, float]]:
    """Compute classic pivots per trading day for later prior-day lookup."""
    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
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


def _assert_multi_tf_cache_fresh(m5_df: pd.DataFrame, multi_tf: dict) -> None:
    """Require the exact causal cache and finite evidence at the final cutoff."""
    if not isinstance(multi_tf, dict):
        raise RuntimeError("[MTF_CACHE_CONTRACT] multi_tf must be an explicit dictionary")
    if set(multi_tf) != set(TF_NAMES):
        raise RuntimeError(
            f"[MTF_CACHE_CONTRACT] exact TF keys required: expected={list(TF_NAMES)} "
            f"observed={sorted(map(str, multi_tf))}"
        )
    m5_last = int(pd.Timestamp(m5_df.index[-1]).value)
    expected_width = len(MULTI_TF_PER_BAR_FEATURES_V2)
    for tf in TF_NAMES:
        feats = multi_tf[tf]
        if not isinstance(feats, pd.DataFrame) or feats.empty:
            raise RuntimeError(f"[MTF_CACHE_CONTRACT] {tf} cache must be a non-empty DataFrame")
        ts_arr = np.asarray(feats.attrs.get("ts_int64"), dtype=np.int64)
        values = np.asarray(feats.attrs.get("feats_np"))
        if ts_arr.ndim != 1 or len(ts_arr) != len(feats) or np.any(np.diff(ts_arr) <= 0):
            raise RuntimeError(f"[MTF_CACHE_CONTRACT] {tf} timestamps are missing or invalid")
        if feats.attrs.get("htf_feature_contract") != HTF_V2_MATRIX_CONTRACT:
            raise RuntimeError(f"[MTF_CACHE_CONTRACT] {tf} exact causal matrix contract missing")
        warmup_rows = validate_causal_feature_matrix(
            values,
            expected_width=expected_width,
            context=f"MTF_CACHE_CONTRACT_{tf}",
        )
        if feats.attrs.get("causal_warmup_rows") != warmup_rows:
            raise RuntimeError(f"[MTF_CACHE_CONTRACT] {tf} warmup metadata mismatch")
        closed_cutoff = _cache_cutoff_ns(m5_last, tf)
        expected = (
            m5_df.resample(MULTI_TF_RESAMPLE_RULES[tf])
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


def build_context(m5_df: pd.DataFrame, multi_tf: dict, journal_dir: Path) -> AugmentContext:
    """Pre-compute all caches ONCE. Heavy upfront cost, fast per-candidate after."""
    _validate_m5_frame(m5_df, context="CTX_BUILD")
    _assert_multi_tf_cache_fresh(m5_df, multi_tf)  # fail-closed on stale multi-TF cache (rule 4)
    ts_ns = m5_df.index.values.astype("datetime64[ns]").astype(np.int64)
    h1_ts, h1_hi, h1_lo = _build_resampled_ohlc_array(m5_df, "1h")
    h4_ts, h4_hi, h4_lo = _build_resampled_ohlc_array(m5_df, "4h")
    # 2026-05-24 Bug 2 fix: M15 + D1 resampled OHLC for liquidity zones
    m15_ts, m15_hi, m15_lo = _build_resampled_ohlc_array(m5_df, "15min")
    d1_ts, d1_hi, d1_lo = _build_resampled_ohlc_array(m5_df, "1D")
    # ATR percentile arrays
    m5_atr_pct = _build_atr_percentile_array(m5_df, ts_ns)
    # H1 ATR percentile
    h1_df = m5_df.resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    h1_ts_pct = h1_df.index.values.astype("datetime64[ns]").astype(np.int64)
    h1_atr_pct = _build_atr_percentile_array(h1_df, h1_ts_pct)
    # Daily pivots
    daily_pivots = _build_daily_pivots(m5_df)
    # Trade history (journal)
    trade_hist = _build_trade_history(journal_dir)

    return AugmentContext(
        m5_close=m5_df["close"].to_numpy(np.float64),
        m5_high=m5_df["high"].to_numpy(np.float64),
        m5_low=m5_df["low"].to_numpy(np.float64),
        m5_ts_ns=ts_ns,
        multi_tf=multi_tf,
        h1_ts_ns=h1_ts, h1_high=h1_hi, h1_low=h1_lo,
        h4_ts_ns=h4_ts, h4_high=h4_hi, h4_low=h4_lo,
        m15_ts_ns=m15_ts, m15_high=m15_hi, m15_low=m15_lo,
        d1_ts_ns=d1_ts, d1_high=d1_hi, d1_low=d1_lo,
        m5_atr_pct_1yr=m5_atr_pct,
        h1_atr_pct_1yr=h1_atr_pct,
        h1_atr_pct_ts_ns=h1_ts_pct,
        daily_pivot_by_date=daily_pivots,
        trade_history=trade_hist,
    )


def _session_overlap(ts: pd.Timestamp) -> dict[str, float]:
    ts = _require_utc_timestamp(ts, context="CTX_SESSION")
    h = ts.hour
    asia, eu, us = h in ASIA_HOURS, h in EU_HOURS, h in US_HOURS
    return {
        "is_asia_eu_overlap": float(asia and eu),
        "is_eu_us_overlap":   float(eu and us),
        "is_eu_only":         float(eu and not us and not asia),
        "is_us_only":         float(us and not eu and not asia),
    }


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
        or values.shape != (len(ts_arr), len(MULTI_TF_PER_BAR_FEATURES_V2))
    ):
        raise RuntimeError(f"[CTX_MTF_SOURCE] malformed {tf} cache arrays")
    right = int(np.searchsorted(ts_arr, _cache_cutoff_ns(ts_ns, tf), side="right"))
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
        for j, feat in enumerate(MULTI_TF_PER_BAR_FEATURES_V2):
            out[_pertf_name(tf, feat)] = float(row[j])
    return out


def _vol_term(per_tf: dict[str, float]) -> dict[str, float]:
    """ATR ratios from the already-resolved five-TF feature snapshot."""
    def _last_atr(tf: str) -> float:
        name = _pertf_name(tf, "atr_bps_14")
        if name not in per_tf:
            raise RuntimeError(f"[CTX_VOL_TERM] missing exact source {name}")
        value = float(per_tf[name])
        if not np.isfinite(value):
            raise RuntimeError(f"[CTX_VOL_TERM] {tf} atr_bps_14 must be finite and positive")
        if value <= 0.0:
            raise CausalContextWarmupError(f"[CTX_VOL_TERM_WARMUP] {tf} ATR is not warmed")
        return value
    a_m5 = _last_atr("M5")
    a_m15 = _last_atr("M15")
    a_h1 = _last_atr("H1")
    a_h4 = _last_atr("H4")
    a_d1 = _last_atr("D1")
    return {
        "atr_ratio_m5_h4":  min(50.0, a_m5 / a_h4),
        "atr_ratio_m15_d1": min(50.0, a_m15 / a_d1),
        "atr_ratio_h1_d1":  min(50.0, a_h1 / a_d1),
        "atr_ratio_m5_m15": min(50.0, a_m5 / a_m15),
    }


def _vol_pct(ctx: AugmentContext, ts_ns: int) -> dict[str, float]:
    """Lookup pre-computed M5 / H1 ATR percentile at ts_ns."""
    m5_idx = int(np.searchsorted(ctx.m5_ts_ns, ts_ns, side="left"))
    if m5_idx >= len(ctx.m5_ts_ns) or int(ctx.m5_ts_ns[m5_idx]) != int(ts_ns):
        raise RuntimeError("[CTX_VOL_PERCENTILE] decision timestamp is not an exact M5 source row")
    h1_right = int(
        np.searchsorted(ctx.h1_atr_pct_ts_ns, _cache_cutoff_ns(ts_ns, "H1"), side="right")
    )
    if h1_right == 0:
        raise CausalContextWarmupError(
            "[CTX_VOL_PERCENTILE_WARMUP] no closed H1 percentile row at decision time"
        )
    m5_pct = float(ctx.m5_atr_pct_1yr[m5_idx])
    h1_pct = float(ctx.h1_atr_pct_1yr[h1_right - 1])
    if not np.isfinite([m5_pct, h1_pct]).all():
        raise RuntimeError("[CTX_VOL_PERCENTILE] percentile sources must be finite")
    return {"vol_pct_m5_1yr": m5_pct, "vol_pct_h1_1yr": h1_pct}


def _pivots(ctx: AugmentContext, ts: pd.Timestamp, current_atr: float, current_price: float) -> dict[str, float]:
    """Lookup prior-day pivots — O(1) dict lookup."""
    ts = _require_utc_timestamp(ts, context="CTX_PIVOT")
    if not np.isfinite(current_atr) or current_atr <= 0.0 or not np.isfinite(current_price):
        raise RuntimeError("[CTX_PIVOT] price and ATR must be finite; ATR must be positive")
    target_day = ts.normalize()
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
        right = int(np.searchsorted(ts_arr, _cache_cutoff_ns(ts_ns, tf_name.upper()), side="right"))
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


def _dip_struct_5tf(per_tf_out: dict[str, float], liq_out: dict[str, float]) -> dict[str, float]:
    """5-TF dip + structure features computed from per-TF V2 + liquidity zones.
    Persisted into fwd dataset so both Entry-IQL and Exit-IQL state builders can
    read them without recomputing.
    """
    out: dict[str, float] = {}
    # DIP per TF: proximity-to-low × sigmoid(ema20-slope)
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        dist_name = f"dist_to_{tf}_lo_atr"
        slope_name = f"{tf}_ema20_slope_atr_v2"
        if dist_name not in liq_out or slope_name not in per_tf_out:
            raise RuntimeError(
                f"[CTX_DIP_STRUCT] exact sources required: {dist_name}, {slope_name}"
            )
        dist_lo = float(liq_out[dist_name])
        slope = float(per_tf_out[slope_name])
        if not np.isfinite([dist_lo, slope]).all():
            raise RuntimeError(f"[CTX_DIP_STRUCT] non-finite source for {tf}")
        dip_prox = max(0.0, min(1.0, 1.0 - dist_lo / 2.0))
        recovery = 1.0 / (1.0 + math.exp(-slope * 5.0))
        out[f"dip_proximity_{tf}_v3"] = dip_prox
        out[f"dip_confirmed_{tf}_v3"] = dip_prox * recovery
    # STRUCTURE per TF: HH/HL/LH/LL via mom_5 + mom_20 signs
    pback = {}
    for tf in ("m5", "m15", "h1", "h4", "d1"):
        mom5_name = f"{tf}_mom_5_atr_v2"
        mom20_name = f"{tf}_mom_20_atr_v2"
        if mom5_name not in per_tf_out or mom20_name not in per_tf_out:
            raise RuntimeError(f"[CTX_DIP_STRUCT] exact momentum sources required for {tf}")
        m5_mom = float(per_tf_out[mom5_name])
        m20_mom = float(per_tf_out[mom20_name])
        if not np.isfinite([m5_mom, m20_mom]).all():
            raise RuntimeError(f"[CTX_DIP_STRUCT] non-finite momentum source for {tf}")
        out[f"struct_continuation_up_{tf}_v3"] = float((m20_mom > 0) and (m5_mom > 0))
        out[f"struct_pullback_in_uptrend_{tf}_v3"] = float((m20_mom > 0) and (m5_mom < 0))
        out[f"struct_continuation_down_{tf}_v3"] = float((m20_mom < 0) and (m5_mom < 0))
        out[f"struct_bounce_in_downtrend_{tf}_v3"] = float((m20_mom < 0) and (m5_mom > 0))
        depth = -m5_mom / max(abs(m20_mom), 1e-6) if m20_mom > 1e-6 else 0.0
        out[f"struct_pullback_depth_{tf}_v3"] = max(-2.0, min(2.0, depth))
        pback[tf] = out[f"struct_pullback_in_uptrend_{tf}_v3"]
    # Multi-TF combo: strict AND across all 5 TFs
    out["struct_all_tf_pullback_v3"] = pback["m5"] * pback["m15"] * pback["h1"] * pback["h4"] * pback["d1"]
    out["struct_tf_agree_count_v3"] = (pback["m5"] + pback["m15"] + pback["h1"] + pback["h4"] + pback["d1"]) / 5.0
    avg_dip = (out["dip_confirmed_m5_v3"] + out["dip_confirmed_m15_v3"] + out["dip_confirmed_h1_v3"]
               + out["dip_confirmed_h4_v3"] + out["dip_confirmed_d1_v3"]) / 5.0
    out["struct_dip_x_uptrend_v3"] = avg_dip * pback["m5"]
    # struct_smc_swing_x_dip_v3 computed downstream in augment_week (needs SMC join)
    return out


def _fvg_5tf(ctx: "AugmentContext", ts_ns: int, current_price: float, current_atr: float) -> dict[str, float]:
    """Nearest UNFILLED 3-bar fair-value-gap per TF — signed ATR-distance + decayed activity.
    Bullish FVG: high[k-2] < low[k] (gap band); bearish: low[k-2] > high[k]. A gap is FILLED once a later
    bar trades back through it. Mirrors _liquidity_zones' per-TF resampled hi/lo arrays. M5/M15/H1 only."""
    import math as _math
    atr_safe = max(current_atr, 1e-3)
    CLIP, TAU = 3.0, 1.0
    M5_NS = 300_000_000_000  # decision moment = close of the M5 bar labeled ts (= ts + 5min)
    out: dict[str, float] = {}
    for tf, ts_arr, hi_arr, lo_arr, lb, period_ns in (
        ("m5", ctx.m5_ts_ns, ctx.m5_high, ctx.m5_low, 240, M5_NS),
        ("m15", ctx.m15_ts_ns, ctx.m15_high, ctx.m15_low, 192, 900_000_000_000),
        ("h1", ctx.h1_ts_ns, ctx.h1_high, ctx.h1_low, 168, 3_600_000_000_000),
    ):
        # 2026-06-11 LOOK-AHEAD FIX: only bars whose period has COMPLETED by the decision moment
        # (close of the M5 bar labeled ts, i.e. ts+5min) may enter the window. The old side="right"
        # cut at ts included the FORMING M15/H1 bar, whose full-period high/low (resampled from the
        # complete tape) leaks intra-period FUTURE data → build≠serve at the same ts. The serve-time
        # contract for FVG is therefore COMPLETED TF bars only. M5 is unchanged (label<=ts ⇔
        # completed-by-ts+5min on the native M5 grid).
        right = int(np.searchsorted(ts_arr, ts_ns + M5_NS - period_ns, side="right"))
        if right < 3:
            out[f"{tf}_dist_to_unfilled_fvg_atr"] = CLIP
            out[f"{tf}_fvg_active"] = float(_math.exp(-CLIP / TAU))
            continue
        left = max(0, right - lb)
        hi = hi_arr[left:right]
        lo = lo_arr[left:right]
        n = len(hi)
        bull = hi[:-2] < lo[2:]            # gap band (hi[k-2], lo[k])
        bear = lo[:-2] > hi[2:]            # gap band (hi[k], lo[k-2])
        suf_minlow = np.minimum.accumulate(lo[::-1])[::-1]   # suf_minlow[j] = min(lo[j:])
        suf_maxhi = np.maximum.accumulate(hi[::-1])[::-1]
        best_signed = None
        best_abs = CLIP * atr_safe + 1.0
        for j in np.flatnonzero(bull):        # local formation idx = j+2
            f = j + 2
            gap_lo, gap_hi = hi[j], lo[f]     # gap_lo < gap_hi by construction
            if f + 1 < n and suf_minlow[f + 1] <= gap_lo:
                continue                      # filled (later low traded back down through the gap)
            if current_price >= gap_hi:
                signed = current_price - gap_hi
            elif current_price <= gap_lo:
                signed = -(gap_lo - current_price)
            else:
                signed = 0.0                  # inside the gap
            if abs(signed) < best_abs:
                best_abs = abs(signed)
                best_signed = signed
        for j in np.flatnonzero(bear):
            f = j + 2
            gap_lo, gap_hi = hi[f], lo[j]
            if f + 1 < n and suf_maxhi[f + 1] >= gap_hi:
                continue
            if current_price <= gap_lo:
                signed = -(gap_lo - current_price)
            elif current_price >= gap_hi:
                signed = current_price - gap_hi
            else:
                signed = 0.0
            if abs(signed) < best_abs:
                best_abs = abs(signed)
                best_signed = signed
        if best_signed is None:
            out[f"{tf}_dist_to_unfilled_fvg_atr"] = CLIP
            out[f"{tf}_fvg_active"] = float(_math.exp(-CLIP / TAU))
        else:
            d = max(-CLIP, min(CLIP, float(best_signed) / atr_safe))
            out[f"{tf}_dist_to_unfilled_fvg_atr"] = float(d)
            out[f"{tf}_fvg_active"] = float(_math.exp(-abs(d) / TAU))
    return out


def augment_candidate(
    ctx: AugmentContext,
    ts: pd.Timestamp,
    *,
    include_portfolio: bool = True,
) -> dict[str, float]:
    """Fast per-candidate compute: O(log N) lookups for all V2 + dip/struct features."""
    if not isinstance(include_portfolio, bool):
        raise TypeError("[CTX_CANDIDATE] include_portfolio must be bool")
    ts = _require_utc_timestamp(ts, context="CTX_CANDIDATE")
    ts_ns = int(ts.value)
    m5_idx = int(np.searchsorted(ctx.m5_ts_ns, ts_ns, side="left"))
    if m5_idx >= len(ctx.m5_ts_ns) or int(ctx.m5_ts_ns[m5_idx]) != ts_ns:
        raise RuntimeError(f"[CTX_CANDIDATE] decision timestamp is not an exact M5 row: {ts}")
    current_price = float(ctx.m5_close[m5_idx])
    # Resolve exactly one row from each TF and reuse that common market
    # snapshot for ATR, volatility-term and dip/structure cooperation.
    per_tf = _per_tf_all(ctx, ts_ns)
    atr_bps = float(per_tf[_pertf_name("M5", "atr_bps_14")])
    if not np.isfinite(current_price) or current_price <= 0.0:
        raise RuntimeError("[CTX_CANDIDATE] current M5 close must be finite and positive")
    if not np.isfinite(atr_bps):
        raise RuntimeError("[CTX_CANDIDATE] current M5 atr_bps_14 must be finite and positive")
    if atr_bps <= 0.0:
        raise CausalContextWarmupError("[CTX_CANDIDATE_WARMUP] current M5 ATR is not warmed")
    current_atr = atr_bps / 1e4 * current_price

    out: dict[str, float] = {}
    liq = _liquidity_zones(ctx, ts_ns, current_price, current_atr)
    out.update(per_tf)                                         # 125
    out.update(_session_overlap(ts))                           # 4
    out.update(_vol_term(per_tf))                              # 4
    out.update(_vol_pct(ctx, ts_ns))                           # 2
    out.update(_pivots(ctx, ts, current_atr, current_price))   # 4
    if _AUG_ROUND_ON:
        out.update(_round_number_levels(current_price, current_atr))   # 5 (env-gated)
    out.update(liq)                                            # 10 (5 TFs × hi/lo)
    if include_portfolio:
        out.update(_per_side_perf(ctx, ts))                    # 8
    out.update(_dip_struct_5tf(per_tf, liq))                   # 38 (dip + struct)
    if _FVG_ON:
        out.update(_fvg_5tf(ctx, ts_ns, current_price, current_atr))   # 6 (3 TFs × {dist, active})
    required_group_a = set(GROUP_A_FEATURE_NAMES)
    if not include_portfolio:
        required_group_a.difference_update(PORTFOLIO_FEATURE_NAMES)
    required = set(PER_TF_FEATURE_NAMES) | required_group_a | set(DIP_STRUCT_FEATURE_NAMES)
    if _FVG_ON:
        required.update(FVG_FEATURE_NAMES)
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

    Extracted from attach_group_a_dip_struct_ctx_columns so callers can
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

_GROUP_A_CHECKPOINT_SCHEMA_VERSION = "group_a_attach_checkpoint_v1"
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
    smc_col: str,
    chunk_rows: int,
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
        (smc_col, pd.to_numeric(df[smc_col], errors="coerce").to_numpy(np.float64)),
    ):
        frame_digest.update(name.encode("utf-8") + b"\0")
        frame_digest.update(_group_a_array_bytes(values))

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
        "multi_tf_sha256": mtf_digest.hexdigest(),
        "extract": list(extract),
        "extract_sha256": _sha256_bytes_iter(
            [b"group_a_extract_v1\0", "\n".join(extract).encode("utf-8")]
        ),
        "smc_col": smc_col,
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


def attach_group_a_dip_struct_ctx_columns_parallel(
    df: pd.DataFrame,
    *,
    multi_tf: dict,
    journal_label: str = "parity",
    smc_col: str = "smc_swing_state",
    workers: int = 12,
    spot_check_rows: int = 40,
    checkpoint_dir: Path | None = None,
    checkpoint_key: str | None = None,
    checkpoint_chunk_rows: int = _GROUP_A_CHECKPOINT_CHUNK_ROWS,
) -> pd.DataFrame:
    """Exact parallel variant of attach_group_a_dip_struct_ctx_columns.

    ONE full-series context is built in the parent (long-memory arrays such as
    the trailing-1yr percentile arrays are therefore identical for every
    worker) and only the zero-copy augment_candidate loop is fanned out over
    fork()ed workers — exact by construction, no chunk overlap. After the
    merge, spot_check_rows evenly spaced finite rows are recomputed serially
    and asserted bit-exact against the merged result.
    """
    import multiprocessing as mp

    global _PARALLEL_ATTACH_SHARED
    ctx, ts_index, extract, dip_from_aug = build_attach_context(
        df, multi_tf=multi_tf, journal_label=journal_label, smc_col=smc_col
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
            smc_col=smc_col,
            chunk_rows=chunk_rows,
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
    result = finalize_attach_columns(
        df, cols, smc_col=smc_col, dip_from_aug=dip_from_aug
    )
    if complete_path is not None:
        result.attrs["group_a_checkpoint_complete_path"] = str(complete_path)
        result.attrs["group_a_checkpoint_complete_sha256"] = complete_sha256
        result.attrs["group_a_checkpoint_key"] = checkpoint_key
    return result


def attach_group_a_dip_struct_ctx_columns(
    df: pd.DataFrame,
    *,
    multi_tf: dict,
    journal_label: str = "parity",
    smc_col: str = "smc_swing_state",
) -> pd.DataFrame:
    """Add the 24 GROUP-A parity + 36 dip/struct ctx_cont columns to ``df`` in place.

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
    """
    ctx, ts_index, extract, dip_from_aug = build_attach_context(
        df, multi_tf=multi_tf, journal_label=journal_label, smc_col=smc_col
    )
    cols = compute_attach_rows(ctx, ts_index, 0, len(df), extract=extract)
    return finalize_attach_columns(
        df, cols, smc_col=smc_col, dip_from_aug=dip_from_aug
    )


def build_attach_context(
    df: pd.DataFrame,
    *,
    multi_tf: dict,
    journal_label: str = "parity",
    smc_col: str = "smc_swing_state",
):
    """Validate ``df`` and build the FULL-series augment context.

    Factored from attach_group_a_dip_struct_ctx_columns so a parallel caller
    can build the context once (full series -> exact trailing-1yr arrays) and
    fan the row loop out over workers.
    """
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS as _GROUP_A,
        MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS as _DIP_STRUCT,
    )
    if not isinstance(multi_tf, dict):
        raise RuntimeError("[CTX_CONT_PARITY] explicit multi_tf source is required")
    if smc_col not in df.columns:
        raise RuntimeError(f"[CTX_CONT_PARITY] exact SMC source missing: {smc_col}")

    for c in ("high", "low", "close"):
        if c not in df.columns:
            raise RuntimeError(
                f"[CTX_CONT_PARITY] df missing required OHLC column '{c}'"
            )
    if "time" in df.columns:
        raw_time = pd.to_datetime(df["time"], utc=True, errors="coerce")
        ts_index = pd.DatetimeIndex(raw_time)
    elif isinstance(df.index, pd.DatetimeIndex):
        ts_index = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True, errors="coerce"))
    else:
        raise RuntimeError("[CTX_CONT_PARITY] df needs a 'time' column or DatetimeIndex")
    if ts_index.hasnans or not ts_index.is_monotonic_increasing or not ts_index.is_unique:
        raise RuntimeError("[CTX_CONT_PARITY] timestamps must be finite, unique and chronological")

    m5 = df[["high", "low", "close"]].copy()
    m5.index = ts_index
    _validate_m5_frame(m5, context="CTX_CONT_PARITY")
    ctx = build_context(
        m5, multi_tf,
        journal_dir=Path(f"/nonexistent_{journal_label}_journal"),
    )

    dip_from_aug = [f for f in _DIP_STRUCT if f != "struct_smc_swing_x_dip_v3"]
    extract = list(_GROUP_A) + dip_from_aug + ["dip_proximity_m5_v3"]
    return ctx, ts_index, extract, dip_from_aug


def finalize_attach_columns(
    df: pd.DataFrame,
    cols: dict,
    *,
    smc_col: str,
    dip_from_aug: list,
) -> pd.DataFrame:
    """Assemble the attach output columns onto ``df`` (post-row-loop steps)."""
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS as _GROUP_A,
    )
    out_cols = {k: cols[k] for k in (list(_GROUP_A) + dip_from_aug)}

    sw = pd.to_numeric(df[smc_col], errors="coerce").to_numpy(np.float64)
    valid = np.isfinite(cols["dip_proximity_m5_v3"])
    interaction = np.full(len(df), np.nan, dtype=np.float32)
    if not valid.any():
        raise RuntimeError("[CTX_CONT_PARITY] no complete causal context row exists")
    interaction[valid] = compute_smc_swing_dip_interaction(
        sw[valid],
        cols["dip_proximity_m5_v3"][valid],
    )
    out_cols["struct_smc_swing_x_dip_v3"] = interaction
    new_cols = pd.DataFrame(out_cols, index=df.index)
    existing = [c for c in new_cols.columns if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    result = pd.concat([df, new_cols], axis=1)
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
    cols_to_overwrite = (set(PER_TF_FEATURE_NAMES) | set(GROUP_A_FEATURE_NAMES)
                         | set(GROUP_S_SMC_FEATURE_NAMES) | set(DIP_STRUCT_FEATURE_NAMES)
                         | set(FVG_FEATURE_NAMES))   # FVG_FEATURE_NAMES = () when GX1_FVG_FEATURES off
    existing_to_drop = [c for c in df.columns if c in cols_to_overwrite]
    if existing_to_drop:
        df = df.drop(columns=existing_to_drop)
    ts_arr = pd.to_datetime(df["decision_ts_utc"], utc=True)
    new_cols: dict[str, list[float]] = {
        name: [] for name in (PER_TF_FEATURE_NAMES + GROUP_A_FEATURE_NAMES + DIP_STRUCT_FEATURE_NAMES
                              + FVG_FEATURE_NAMES)
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
            "smc_sweep_up", "smc_sweep_down", "smc_sweep_size_atr",
            "smc_bars_since_sweep", "smc_premium_discount", "smc_premium_state",
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

    # 2026-05-24 PM: struct_smc_swing_x_dip_v3 = SMC swing state × M5 dip proximity.
    # Computed here (after SMC join) since needs both signals.
    if "smc_swing_state_canon_v1" not in df.columns or "dip_proximity_m5_v3" not in df.columns:
        raise RuntimeError("[AUG_V2_SMC] exact SMC swing and dip sources are required")
    sw = pd.to_numeric(df["smc_swing_state_canon_v1"], errors="coerce").to_numpy(dtype=np.float64)
    dp = pd.to_numeric(df["dip_proximity_m5_v3"], errors="coerce").to_numpy(dtype=np.float64)
    df["struct_smc_swing_x_dip_v3"] = compute_smc_swing_dip_interaction(sw, dp)

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
    multi_tf = build_multi_tf_per_bar_features_v2(m5_df)
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
        "smc_sweep_up","smc_sweep_down","smc_sweep_size_atr",
        "smc_bars_since_sweep","smc_premium_discount","smc_premium_state",
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
        "env_gates": {k: os.environ.get(k, "0") for k in
                      ("GX1_ROUND_NUMBER", "GX1_FVG_FEATURES", "GX1_SMC_SWEEP_RECLAIM")},
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
