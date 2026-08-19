"""Strict observed-market primitives shared by model-native build and serve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.features.technical_indicators_v1 import wilder_atr

# One ATR owner (rule 19).  ``ctx_cont.atr_bps`` previously ran its own
# ``true_range.rolling(14, min_periods=1).mean()`` — a simple moving average
# over a partial window — while every other ATR on the surface is the classic
# Wilder RMA from ``technical_indicators_v1.wilder_atr``: ``_v1_atr14`` (index 0
# of the same signal vector, same native M5 clock), the per-TF ``atr_bps_14``
# block, the V29 level/trendline registries and every ``*_atr``-normalized
# field.  Two estimators of one named quantity fed the same vector and
# disagreed bar to bar; no gate can see that (rule 25).  This owner now routes
# through the single Wilder owner.
#
# Period 14 is not a new magnitude: it is the exact period this owner already
# used (``rolling(14)``) and the same period ``basic_v1`` uses for ``_v1_atr14``
# (``wilder_atr(high, low, close, 14)``), which is the field ``atr_bps`` is now
# consistent with.
MODEL_NATIVE_ATR_PERIOD_V1 = 14
# ``wilder_atr`` documents its own warmup: "the first defined ATR is at row
# ``period-1``".  The preceding rows are honest NaN, never a partial-window
# mean that reads as a converged ATR (the retired ``min_periods=1``).  Derived
# from the owner's contract, not chosen.
MODEL_NATIVE_ATR_CAUSAL_WARMUP_ROWS_V1 = MODEL_NATIVE_ATR_PERIOD_V1 - 1
# Exactly the ctx-contract field this owner emits with that NaN prefix, for the
# lanes' ``trim_causal_context_warmup_prefix`` lists.  ``spread_bps`` is defined
# on every row and must NOT be listed.  The bare ``atr`` column this function
# also returns is not a ctx contract field; where a lane writes it from this
# owner it is ``atr_bps`` multiplied by a strictly positive price level, so it
# is non-finite on exactly the same rows and the same trim removes it.
MODEL_NATIVE_ATR_WARMUP_PREFIX_FIELDS_V1 = ("atr_bps",)


def _finite_or_fail(values: np.ndarray, *, label: str) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"[MODEL_NATIVE_CONTEXT_NONFINITE] {label}: "
            f"count={int(np.count_nonzero(~np.isfinite(values)))}"
        )


def derive_observed_spread_bps(frame: pd.DataFrame) -> np.ndarray:
    """Return causal spread bps without defaults, clipping or zero fill."""

    if "spread_bps" in frame.columns:
        spread_bps = frame["spread_bps"].to_numpy(dtype=np.float64)
        _finite_or_fail(spread_bps, label="spread_bps")
        if np.any(spread_bps < 0.0):
            raise RuntimeError(
                "[MODEL_NATIVE_CONTEXT_INVALID] spread_bps contains negative values: "
                f"count={int(np.count_nonzero(spread_bps < 0.0))}"
            )
        return spread_bps

    if {"bid_close", "ask_close"}.issubset(frame.columns):
        bid = frame["bid_close"].to_numpy(dtype=np.float64)
        ask = frame["ask_close"].to_numpy(dtype=np.float64)
        _finite_or_fail(bid, label="bid_close")
        _finite_or_fail(ask, label="ask_close")
        invalid = (bid <= 0.0) | (ask < bid)
        if np.any(invalid):
            raise RuntimeError(
                "[MODEL_NATIVE_CONTEXT_INVALID] invalid bid/ask spread rows: "
                f"count={int(np.count_nonzero(invalid))}"
            )
        spread_bps = (ask - bid) / bid * 1e4
        _finite_or_fail(spread_bps, label="spread_bps_from_bid_ask")
        return spread_bps

    if {"spread", "close"}.issubset(frame.columns):
        spread = frame["spread"].to_numpy(dtype=np.float64)
        close = frame["close"].to_numpy(dtype=np.float64)
        _finite_or_fail(spread, label="spread")
        _finite_or_fail(close, label="close")
        invalid = (spread < 0.0) | (close <= 0.0)
        if np.any(invalid):
            raise RuntimeError(
                "[MODEL_NATIVE_CONTEXT_INVALID] invalid spread/close rows: "
                f"count={int(np.count_nonzero(invalid))}"
            )
        spread_bps = spread / close * 1e4
        _finite_or_fail(spread_bps, label="spread_bps_from_spread_close")
        return spread_bps

    raise RuntimeError(
        "[MODEL_NATIVE_CONTEXT_MISSING] observed spread requires spread_bps, "
        "bid_close+ask_close, or spread+close"
    )


def derive_model_native_atr_spread_bps(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact raw ATR/spread context without any rank or bucket."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError("[MODEL_NATIVE_CONTEXT_EMPTY] ATR/spread source")
    required = ("high", "low", "close")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(
            f"[MODEL_NATIVE_CONTEXT_MISSING] ATR source fields: {missing}"
        )
    numeric = {
        name: pd.to_numeric(frame[name], errors="raise").to_numpy(
            dtype=np.float64
        )
        for name in required
    }
    for name, values in numeric.items():
        _finite_or_fail(values, label=name)
    high = numeric["high"]
    low = numeric["low"]
    close = numeric["close"]
    invalid = (
        (high <= 0.0)
        | (low <= 0.0)
        | (close <= 0.0)
        | (high < low)
        | (high < close)
        | (low > close)
    )
    if invalid.any():
        raise RuntimeError(
            "[MODEL_NATIVE_CONTEXT_INVALID] OHLC rows: "
            f"count={int(np.count_nonzero(invalid))}"
        )
    high_s = pd.Series(high, index=frame.index, dtype=np.float64)
    low_s = pd.Series(low, index=frame.index, dtype=np.float64)
    close_s = pd.Series(close, index=frame.index, dtype=np.float64)
    # One Wilder owner, no partial window: rows before the seed stay NaN for
    # the caller's causal-history trim (MODEL_NATIVE_ATR_WARMUP_PREFIX_FIELDS_V1).
    atr = wilder_atr(
        high_s, low_s, close_s, MODEL_NATIVE_ATR_PERIOD_V1
    ).to_numpy(dtype=np.float64)
    # One denominator for one concept (rule 19).  This owner used to divide by
    # the bar midpoint ``(high + low) / 2`` while the per-timeframe sibling
    # ``atr_bps_14`` -- in the SAME signal vector, read by the SAME encoder
    # -- divides by ``close``.  Two conventions for one named quantity, which
    # no gate can see (rule 25).  ``close`` wins, and it is not a chosen
    # magnitude but the convention already emitted by every other ``*_bps``
    # owner in this repository (rule 2a/2b), measured on 2026-08-19 from real
    # emitted bytes rather than from a restated literal (rule 13):
    #
    #   * the per-TF ``atr_bps_14`` column of the V31 MULTI_TF_V4_CACHE
    #     (cache_identity_sha256 f986e8ac2b19..., 477,229 native M5 rows)
    #     reproduces as ``wilder_atr(...)/close*1e4`` to float32 storage
    #     resolution (max |rel| 5.9e-08) and misses ``/mid`` on 96.28% of rows;
    #   * ``micro_structure_v1``'s ``close_return_{3,5}_bps``,
    #     ``spread_extremes_sum_bps`` and ``quote_range_asymmetry_bps``,
    #     executed on the full declared M5 tape, match ``/close`` and miss
    #     ``/mid`` by up to 127 bps;
    #   * this file's own ``derive_observed_spread_bps`` already divides by a
    #     quoted CLOSE price (``bid_close``, or ``close``), never by a midrange.
    #
    # ``close`` is also the price the decision is taken at: the model acts on
    # the closed bar, so a magnitude "in bps" is in bps of the price that is
    # live when the order would be sent.
    #
    # The midrange denominator was not merely a different scale, it was a
    # direction leak.  Algebraically ``atr/mid == (atr/close) * (close/mid)``
    # exactly, and ``close/mid - 1`` is a monotone re-expression of where the
    # close sits inside the bar: on the full tape it correlates 0.7263
    # (Pearson) / 0.9170 (Spearman) with the intrabar close position, and
    # -0.7263 with ``close_distance_below_high_range_fraction`` -- a field this
    # same ctx vector ALREADY carries from its own owner.  The midrange form
    # multiplied a volatility magnitude by a second owner's direction field.
    #
    # Measured size of the split on XAU_M5_NATIVE_2019_20260804_V4 (537,861
    # rows, 2019-01-01..2026-08-04, 537,848 rows after the Wilder warmup):
    # |rel diff| mean 2.21e-04, p99.9 2.14e-03, max 1.27e-02; RMS(diff) is
    # 0.17% of the field's own standard deviation; Spearman between the two
    # conventions 0.99999988.  The inconsistency was real and numerically
    # small; it is removed because it is a second owner, not because it was
    # large.
    atr_bps = atr / close * 1e4
    spread_bps = derive_observed_spread_bps(frame)
    warmup = MODEL_NATIVE_ATR_CAUSAL_WARMUP_ROWS_V1
    if len(atr_bps) <= warmup:
        raise RuntimeError(
            "[MODEL_NATIVE_CONTEXT_SHORT] ATR source needs more than "
            f"{warmup} rows for a defined Wilder-{MODEL_NATIVE_ATR_PERIOD_V1} "
            f"ATR; got {len(atr_bps)}"
        )
    if np.isfinite(atr_bps[:warmup]).any():
        raise RuntimeError(
            "[MODEL_NATIVE_CONTEXT_INVALID] atr_bps warmup prefix is not "
            "entirely unavailable"
        )
    _finite_or_fail(atr_bps[warmup:], label="atr_bps")
    if np.any(atr_bps[warmup:] <= 0.0):
        raise RuntimeError("[MODEL_NATIVE_CONTEXT_INVALID] nonpositive atr_bps")
    return pd.DataFrame(
        {"atr": atr, "atr_bps": atr_bps, "spread_bps": spread_bps},
        index=frame.index,
    )
