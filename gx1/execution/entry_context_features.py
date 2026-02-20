"""
Entry Context Features - Model Input (ONE UNIVERSE 6/6, not gates).

STEP 2: Context features are INPUT to the model, allowing the model to learn
optimal entries per regime/session/spread context.

ONE UNIVERSE doctrine (locked):
- ctx_cont_dim == 6
- ctx_cat_dim  == 6
- No fallback. No optional dims. No silent defaults.
- Tensor order and naming follow gx1.contracts.signal_bridge_v1 (ORDERED_CTX_*_EXTENDED prefix).

If any required ctx feature is missing or non-finite → hard fail.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from gx1.contracts.signal_bridge_v1 import (
    CTX_CONT_COL_D1_ATR_PCTL252,
    CTX_CONT_COL_D1_DIST,
    CTX_CONT_COL_H1_COMP,
    CTX_CONT_COL_M15_COMP,
    ORDERED_CTX_CAT_NAMES_EXTENDED,
    ORDERED_CTX_CONT_NAMES_EXTENDED,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ONE UNIVERSE: ONLY 6/6
# ---------------------------------------------------------------------
CTX_CONT_DIM = 6
CTX_CAT_DIM = 6


def _is_truth_or_smoke() -> bool:
    mode = os.getenv("GX1_RUN_MODE", "").upper()
    return os.getenv("GX1_TRUTH_MODE", "0") == "1" or mode in {"TRUTH", "SMOKE"}


def _finite(x: float) -> bool:
    return bool(np.isfinite(x))


def _require(condition: bool, msg: str) -> None:
    if not condition:
        raise RuntimeError(msg)


@dataclass(frozen=True)
class EntryContextFeatures:
    """
    Context features for ENTRY_V10 model input (ONE UNIVERSE 6/6).

    Categorical (6):
      0 session_id          : 0=ASIA, 1=EU, 2=US, 3=OVERLAP
      1 trend_regime_id     : 0=TREND_DOWN, 1=TREND_NEUTRAL, 2=TREND_UP
      2 vol_regime_id       : 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME
      3 atr_bucket          : 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME
      4 spread_bucket       : 0=LOW, 1=MEDIUM, 2=HIGH
      5 h4_trend_sign_cat   : 0/1/2 (contract column exists in ORDERED_CTX_CAT_NAMES_EXTENDED[5])

    Continuous (6):
      0 atr_bps                         : clipped [0, 1000]
      1 spread_bps                      : clipped [0, 500]
      2 d1_dist_from_ema200_atr          : finite (contract col CTX_CONT_COL_D1_DIST)
      3 h1_range_compression_ratio       : finite (contract col CTX_CONT_COL_H1_COMP)
      4 d1_atr_percentile_252            : finite (contract col CTX_CONT_COL_D1_ATR_PCTL252)
      5 m15_range_compression_ratio      : finite (contract col CTX_CONT_COL_M15_COMP)

    No optional fields. No defaults. No silent “0.0” padding.
    """

    # Categorical features (integer IDs)
    session_id: int
    trend_regime_id: int
    vol_regime_id: int
    atr_bucket: int
    spread_bucket: int
    h4_trend_sign_cat: int

    # Continuous features (normalized)
    atr_bps: float
    spread_bps: float
    d1_dist_from_ema200_atr: float
    h1_range_compression_ratio: float
    d1_atr_percentile_252: float
    m15_range_compression_ratio: float

    # Metadata (debug only; not part of tensors)
    _atr_bps_raw: float | None = None
    _spread_bps_raw: float | None = None
    _source: str = "computed"

    def to_tensor_categorical(self) -> np.ndarray:
        """Return int64 tensor of length 6 (order = ORDERED_CTX_CAT_NAMES_EXTENDED[:6])."""
        arr = np.array(
            [
                int(self.session_id),
                int(self.trend_regime_id),
                int(self.vol_regime_id),
                int(self.atr_bucket),
                int(self.spread_bucket),
                int(self.h4_trend_sign_cat),
            ],
            dtype=np.int64,
        )
        _require(arr.shape[0] == CTX_CAT_DIM, f"[CTX__DIM_MISMATCH] cat tensor len={arr.shape[0]} expected={CTX_CAT_DIM}")
        return arr

    def to_tensor_continuous(self) -> np.ndarray:
        """Return float32 tensor of length 6 (order = ORDERED_CTX_CONT_NAMES_EXTENDED[:6])."""
        arr = np.array(
            [
                float(self.atr_bps),
                float(self.spread_bps),
                float(self.d1_dist_from_ema200_atr),
                float(self.h1_range_compression_ratio),
                float(self.d1_atr_percentile_252),
                float(self.m15_range_compression_ratio),
            ],
            dtype=np.float32,
        )
        _require(arr.shape[0] == CTX_CONT_DIM, f"[CTX__DIM_MISMATCH] cont tensor len={arr.shape[0]} expected={CTX_CONT_DIM}")
        return arr

    def as_contract_dict(self) -> Dict[str, Any]:
        """
        Export using contract column names for debugging/auditing (not required by model).
        Keys align with ORDERED_CTX_*_EXTENDED prefix columns.
        """
        cont_cols = list(ORDERED_CTX_CONT_NAMES_EXTENDED[:CTX_CONT_DIM])
        cat_cols = list(ORDERED_CTX_CAT_NAMES_EXTENDED[:CTX_CAT_DIM])
        cont_vals = [
            float(self.atr_bps),
            float(self.spread_bps),
            float(self.d1_dist_from_ema200_atr),
            float(self.h1_range_compression_ratio),
            float(self.d1_atr_percentile_252),
            float(self.m15_range_compression_ratio),
        ]
        cat_vals = [
            int(self.session_id),
            int(self.trend_regime_id),
            int(self.vol_regime_id),
            int(self.atr_bucket),
            int(self.spread_bucket),
            int(self.h4_trend_sign_cat),
        ]
        return {
            "ctx_cont": dict(zip(cont_cols, cont_vals)),
            "ctx_cat": dict(zip(cat_cols, cat_vals)),
            "_meta": {"source": self._source, "atr_bps_raw": self._atr_bps_raw, "spread_bps_raw": self._spread_bps_raw},
        }

    def validate(self) -> None:
        """Strict validation for ONE UNIVERSE 6/6. Raises RuntimeError on any invalid state."""
        # Categorical ranges
        _require(0 <= int(self.session_id) <= 3, f"[CTX_CAT_FAIL] session_id out of range: {self.session_id} (expected 0-3)")
        _require(0 <= int(self.trend_regime_id) <= 2, f"[CTX_CAT_FAIL] trend_regime_id out of range: {self.trend_regime_id} (expected 0-2)")
        _require(0 <= int(self.vol_regime_id) <= 3, f"[CTX_CAT_FAIL] vol_regime_id out of range: {self.vol_regime_id} (expected 0-3)")
        _require(0 <= int(self.atr_bucket) <= 3, f"[CTX_CAT_FAIL] atr_bucket out of range: {self.atr_bucket} (expected 0-3)")
        _require(0 <= int(self.spread_bucket) <= 2, f"[CTX_CAT_FAIL] spread_bucket out of range: {self.spread_bucket} (expected 0-2)")
        _require(0 <= int(self.h4_trend_sign_cat) <= 2, f"[CTX_CAT_FAIL] h4_trend_sign_cat out of range: {self.h4_trend_sign_cat} (expected 0-2)")

        # Continuous must be finite
        _require(_finite(float(self.atr_bps)), f"[CTX_CONT_FAIL] atr_bps not finite: {self.atr_bps}")
        _require(_finite(float(self.spread_bps)), f"[CTX_CONT_FAIL] spread_bps not finite: {self.spread_bps}")
        _require(_finite(float(self.d1_dist_from_ema200_atr)), f"[CTX_CONT_FAIL] {CTX_CONT_COL_D1_DIST} not finite: {self.d1_dist_from_ema200_atr}")
        _require(_finite(float(self.h1_range_compression_ratio)), f"[CTX_CONT_FAIL] {CTX_CONT_COL_H1_COMP} not finite: {self.h1_range_compression_ratio}")
        _require(_finite(float(self.d1_atr_percentile_252)), f"[CTX_CONT_FAIL] {CTX_CONT_COL_D1_ATR_PCTL252} not finite: {self.d1_atr_percentile_252}")
        _require(_finite(float(self.m15_range_compression_ratio)), f"[CTX_CONT_FAIL] {CTX_CONT_COL_M15_COMP} not finite: {self.m15_range_compression_ratio}")

        # Range sanity for clipped ones
        _require(0.0 <= float(self.atr_bps) <= 1000.0, f"[CTX_CONT_FAIL] atr_bps out of range: {self.atr_bps} (expected [0, 1000])")
        _require(0.0 <= float(self.spread_bps) <= 500.0, f"[CTX_CONT_FAIL] spread_bps out of range: {self.spread_bps} (expected [0, 500])")


def build_entry_context_features(
    candles: pd.DataFrame,
    policy_state: dict,
    *,
    # Baseline
    atr_proxy: float | None = None,
    spread_bps: float | None = None,
    # Required ONE UNIVERSE 6/6 extensions (must be provided; typically from prebuilt)
    d1_dist_from_ema200_atr: float | None = None,
    h1_range_compression_ratio: float | None = None,
    d1_atr_percentile_252: float | None = None,
    m15_range_compression_ratio: float | None = None,
    h4_trend_sign_cat: int | None = None,
    **kwargs: object,
) -> EntryContextFeatures:
    """
    Build entry context features (ONE UNIVERSE 6/6).

    Policy:
    - Baseline (atr_bps/spread_bps + buckets) can be computed.
    - Extended 6/6 fields MUST be provided (typically from prebuilt join). No fallback.
    """
    from gx1.execution.live_features import infer_session_tag

    current_ts = candles.index[-1] if len(candles) > 0 else pd.Timestamp.now(tz="UTC")

    # session_id (time-based)
    current_session = policy_state.get("session")
    if not current_session:
        current_session = infer_session_tag(current_ts).upper()
        policy_state["session"] = current_session

    valid_sessions = {"ASIA", "EU", "US", "OVERLAP"}
    tag = (current_session or "").strip().upper()
    _require(tag in valid_sessions, f"[CTX_CAT_FAIL] unknown session tag: {current_session!r} (expected one of {sorted(valid_sessions)})")
    session_map = {"ASIA": 0, "EU": 1, "US": 2, "OVERLAP": 3}
    session_id = session_map[tag]

    # spread_bps
    if spread_bps is None:
        _require(
            ("bid_close" in candles.columns and "ask_close" in candles.columns),
            "[CTX_CONT_FAIL] spread_bps missing (no bid_close/ask_close in candles)",
        )
        bid = candles["bid_close"].iloc[-1]
        ask = candles["ask_close"].iloc[-1]
        _require(pd.notna(bid) and pd.notna(ask) and float(bid) > 0.0, "[CTX_CONT_FAIL] spread_bps missing (bid/ask invalid)")
        spread_price = float(ask) - float(bid)
        spread_bps_raw = (spread_price / float(bid)) * 10000.0
        spread_bps = float(spread_bps_raw)
    else:
        spread_bps_raw = float(spread_bps)

    spread_bps_clipped = max(0.0, min(500.0, float(spread_bps_raw)))

    # spread_bucket (simple)
    if spread_bps_clipped < 10.0:
        spread_bucket = 0
    elif spread_bps_clipped < 30.0:
        spread_bucket = 1
    else:
        spread_bucket = 2

    # atr_proxy (cheap compute if missing)
    if atr_proxy is None:
        atr_proxy = _compute_cheap_atr_proxy(candles, window=14)
    _require(atr_proxy is not None, f"[CTX_CONT_FAIL] atr_proxy unavailable (candles_len={len(candles)})")

    close = candles.get("close", None)
    _require(close is not None and len(close) > 0, "[CTX_CONT_FAIL] atr_bps: close missing")
    current_price = float(close.iloc[-1])
    _require(current_price > 0.0, "[CTX_CONT_FAIL] atr_bps: close price <= 0")
    atr_bps_raw = (float(atr_proxy) / current_price) * 10000.0
    atr_bps_clipped = max(0.0, min(1000.0, float(atr_bps_raw)))

    # atr_bucket / vol_regime_id (simple)
    if atr_bps_clipped < 30.0:
        atr_bucket = 0
    elif atr_bps_clipped < 100.0:
        atr_bucket = 1
    elif atr_bps_clipped < 200.0:
        atr_bucket = 2
    else:
        atr_bucket = 3
    vol_regime_id = atr_bucket

    # trend_regime_id
    trend_regime = policy_state.get("brain_trend_regime", "TREND_NEUTRAL")
    trend_map = {"TREND_DOWN": 0, "TREND_NEUTRAL": 1, "TREND_UP": 2}
    _require(trend_regime in trend_map, f"[CTX_CAT_FAIL] brain_trend_regime unknown: {trend_regime!r}")
    trend_regime_id = trend_map[trend_regime]

    # -----------------------------------------------------------------
    # ONE UNIVERSE required extensions (no fallback)
    # -----------------------------------------------------------------
    _require(d1_dist_from_ema200_atr is not None, f"[CTX_CONT_FAIL] missing required {CTX_CONT_COL_D1_DIST} (d1_dist_from_ema200_atr)")
    _require(h1_range_compression_ratio is not None, f"[CTX_CONT_FAIL] missing required {CTX_CONT_COL_H1_COMP} (h1_range_compression_ratio)")
    _require(d1_atr_percentile_252 is not None, f"[CTX_CONT_FAIL] missing required {CTX_CONT_COL_D1_ATR_PCTL252} (d1_atr_percentile_252)")
    _require(m15_range_compression_ratio is not None, f"[CTX_CONT_FAIL] missing required {CTX_CONT_COL_M15_COMP} (m15_range_compression_ratio)")
    _require(h4_trend_sign_cat is not None, "[CTX_CAT_FAIL] missing required h4_trend_sign_cat (ctx_cat dim 6)")

    ctx = EntryContextFeatures(
        session_id=int(session_id),
        trend_regime_id=int(trend_regime_id),
        vol_regime_id=int(vol_regime_id),
        atr_bucket=int(atr_bucket),
        spread_bucket=int(spread_bucket),
        h4_trend_sign_cat=int(h4_trend_sign_cat),
        atr_bps=float(atr_bps_clipped),
        spread_bps=float(spread_bps_clipped),
        d1_dist_from_ema200_atr=float(d1_dist_from_ema200_atr),
        h1_range_compression_ratio=float(h1_range_compression_ratio),
        d1_atr_percentile_252=float(d1_atr_percentile_252),
        m15_range_compression_ratio=float(m15_range_compression_ratio),
        _atr_bps_raw=float(atr_bps_raw),
        _spread_bps_raw=float(spread_bps_raw),
        _source="computed",
    )

    # Validate (always strict)
    ctx.validate()

    # Extra paranoia in TRUTH/SMOKE: assert tensor lengths now (fast)
    if _is_truth_or_smoke():
        _ = ctx.to_tensor_categorical()
        _ = ctx.to_tensor_continuous()

    return ctx


def _compute_cheap_atr_proxy(candles: pd.DataFrame, window: int = 14) -> float | None:
    """
    Compute ultra-cheap ATR proxy from raw candles (no feature build required).

    Same style as soft eligibility: fixed-window mean true range, no randomness.
    """
    if candles.empty or len(candles) < window:
        return None

    try:
        high = candles.get("high", None)
        low = candles.get("low", None)
        close = candles.get("close", None)
        if high is None or low is None or close is None:
            return None

        high_arr = high.iloc[-window:].to_numpy(dtype=np.float64, copy=False)
        low_arr = low.iloc[-window:].to_numpy(dtype=np.float64, copy=False)
        close_arr = close.iloc[-window:].to_numpy(dtype=np.float64, copy=False)

        # prev_close: first element has no previous -> NaN, no roll wrap
        prev_close = np.concatenate([[np.nan], close_arr[:-1]])
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - prev_close)
        tr3 = np.abs(low_arr - prev_close)

        tr = tr1.copy()
        mask = ~np.isnan(prev_close)
        tr[mask] = np.maximum(tr1[mask], np.maximum(tr2[mask], tr3[mask]))

        atr = float(np.mean(tr))
        if not np.isfinite(atr):
            return None
        return atr
    except Exception:
        return None