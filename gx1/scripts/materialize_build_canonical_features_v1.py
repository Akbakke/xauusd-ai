#!/usr/bin/env python3
"""Build the canonical model-agnostic M5 feature parquet.

Mission
-------
Produce one canonical feature parquet covering the declared source interval.
Output is keyed by `time` and feeds the current model-native dataset/runtime
owners; it contains no decision policy.

Feature families
----------------
1. basic_v1 (~60 features): all `_v1_*` features, `_v1h1_*` H1 multi-TF,
   `_v1h4_*` H4 multi-TF — the canonical V10 input.
2. m5_phase (5 features): one-hot of (minute_of_hour // 12) ∈ {0..4}.
3. High-level basics: mid, range, atr50, atr_z, ret_1/20, roc100,
   rvol_20/60, vol_ratio, ema20_slope_atr, ema100_slope_atr, plus the
   basic_v1 plus5 block (atr, std50, roc20, _v1_vwap_drift48).
   ``ret_5``, ``body_pct``, ``wick_asym`` and ``pos_vs_ema200`` were retired
   on 2026-08-19 as exact duplicates of live fields owned elsewhere; see
   add_high_level_basics.

Output
------
  canonical_features_v1.parquet  — single file, 449k rows × ~80 cols, indexed by `time`

Joinability
-----------
Downstream causal builders load this file and use `np.searchsorted` on the
time-array for O(log N) lookup at any
candidate or held-trade-bar timestamp.

This is RESEARCH-ONLY. No runtime promotion. Production V10/V3 features
remain canonical-runtime; this is a derivative offline view to feed IQL.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.features.basic_v1 import (  # noqa: E402
    PLUS5_FEATURES,
    build_basic_v1,
    compute_plus5_features,
)
from gx1.features.technical_indicators_v1 import (  # noqa: E402
    classic_ema,
    wilder_atr14_positive,
)
ACTION = "BUILD_CANONICAL_FEATURES_V1"
# The module-level z-score clip bound is RETIRED (2026-08-19).  Its only
# consumer was ``atr_z``, and its only stated justification was the 1e-9
# standard-deviation floor that the same repair removed; see the comment at
# the atr_z assignment.
DEFAULT_M5_TAPE_ROOT = Path(
    "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL"
)


def load_m5_tape(m5_root: Path = DEFAULT_M5_TAPE_ROOT) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for year_dir in sorted(m5_root.glob("year=*")):
        for parquet in sorted(year_dir.glob("*.parquet")):
            parts.append(pd.read_parquet(parquet))
    if not parts:
        raise RuntimeError(f"[{ACTION}] no M5 parquets under {m5_root}")
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)
    return df


def add_m5_phase(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 one-hot phase features = (minute_of_hour // 12)."""
    minute = df["time"].dt.minute
    phase = (minute // 12).clip(0, 4).astype(int)
    for i in range(5):
        df[f"m5_phase_{i}"] = (phase == i).astype(np.float32)
    return df


def add_high_level_basics(df: pd.DataFrame) -> pd.DataFrame:
    """Add the local-clock high-level feature columns.

    The quote-spanned bar range uses bid/ask extremes over the bid+ask mid;
    every price-derived column uses the mid OHLC frame's own ``close``.
    """
    # The OHLCV geometry owner runs FIRST so an invalid bar fails closed before
    # any derived column is written.  Its emitted columns are still assigned at
    # the end of this function, so the output column order is unchanged.
    plus5 = compute_plus5_features(df)

    mid = (df["bid_close"] + df["ask_close"]) / 2.0
    df["mid"] = mid.astype(np.float32)
    # 2026-08-19 epsilon repair.  ``np.maximum(mid, 1e-9)`` fabricated a
    # 1e13-bps range out of a non-positive mid instead of reporting that the
    # two-sided quote is invalid.  Convention adopted verbatim from this
    # repository's one divide-by-denominator owner,
    # basic_v1._divide_where_positive / technical_indicators_v1
    # .wilder_atr14_positive: divide only where the denominator is defined and
    # strictly positive, otherwise the quotient is unavailable (NaN).  No
    # magnitude is introduced or removed on any row with a valid quote.
    df["range"] = (
        (df["ask_high"] - df["bid_low"]).div(mid.where(mid > 0.0)) * 10000.0
    ).astype(np.float32)

    # ``body_pct`` and ``wick_asym`` are RETIRED here (2026-08-19).  Both were
    # exact functions of columns that remain model inputs in the mandatory
    # causal candle family
    # (gx1.features.entry_candle_primitives_v1), so no market evidence leaves
    # the learned path (rule 4).  PROVEN FROM SOURCE AND ALGEBRA, independent
    # of data (rule 2c), with ``R = high - low`` and the candle owner's three
    # emitted range shares ``b = candle.raw_body_signed_range``,
    # ``u = candle.raw_upper_wick_share``, ``l = candle.raw_lower_wick_share``:
    #
    # 1. ``body_pct == abs(b)``.  On ``R > 0`` the candle owner emits
    #    ``b = (close - open) / R`` and this block emitted
    #    ``abs(close - open) / R``; because ``low <= open, close <= high`` the
    #    identity ``abs(close - open) <= R`` holds on every bar, so the outer
    #    ``clip(0, 1)`` was inert and the two agree exactly.  On ``R == 0`` the
    #    bar has ``open == close``, so the retired epsilon denominator produced
    #    ``0 / 1e-9 == 0.0`` and the candle owner emits the same storage
    #    ``0.0``.  ``b`` additionally carries the SIGN that ``body_pct``
    #    discarded, so the retirement strictly adds information.
    # 2. ``wick_asym == (u - l) / (u + l)`` wherever ``u + l > 0`` (divide the
    #    retired formula's numerator and denominator by ``R``).  Where
    #    ``u + l == 0`` — a marubozu, or a zero-range bar — the retired field
    #    emitted 0.0 for an undefined ``0 / 0``, a placeholder that reads as
    #    "symmetric wicks" and carries no observation (rule 2e); ``u`` and
    #    ``l`` are both retained, so that state is still fully described.
    #    ``wick_asym`` was already declared retired by the signal contract's
    #    own v14 note; this emission was the leftover.

    # ATR families.  ``atr14``/``atr14_positive`` come from the one Wilder
    # owner (technical_indicators_v1.wilder_atr14_positive, the same
    # ``wilder_atr(high, low, close, 14)`` that produces ``_v1_atr14`` beside
    # these columns), never from a second in-file ATR.
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    _atr14, atr14_positive = wilder_atr14_positive(high, low, close)

    high_low = high - low
    high_pc = (high - close.shift(1)).abs()
    low_pc = (low - close.shift(1)).abs()
    # Row 0 has no previous close, so the two gap terms are NaN and the
    # row-wise max is ``high - low`` — the same first-row true range
    # wilder_atr documents and uses.
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    # 2026-08-19 warmup repair.  ``min_periods=1`` made row 0's "50-bar ATR" a
    # single true range and rows 1..48 partial-window means that read as
    # converged values -- the neutral-fill-inside-warmup class basic_v1 and
    # technical_indicators_v1 already repaired for themselves
    # ("warmup=causal_nan_prefix_no_partial_window_no_neutral_fill").  The
    # window is the field's own declared 50; the compared min_periods=10 on the
    # z-score moments had no named origin at all (rule 2a).  Every first finite
    # row below is DERIVED, not chosen: atr50 at index 49, its 50-bar moments
    # at 98.  The deepest prefix this whole block emits is ema100_slope_atr at
    # index 119, and every one of them is inside the surface's declared
    # entry_model_native_feature_layers_v1.PRICE_DERIVED_CAUSAL_WARMUP_ROWS
    # floor, so no emitted sample row loses a value.  Read that floor from its
    # owner; the literal is deliberately not restated here (it moved once on
    # 2026-08-19 while this comment was being written).
    atr50 = tr.rolling(50, min_periods=50).mean()
    df["atr50"] = atr50.astype(np.float32)
    atr50_mean = atr50.rolling(50, min_periods=50).mean()
    atr50_std = atr50.rolling(50, min_periods=50).std()
    # 2026-08-19 epsilon + clip repair.  The retired form divided by
    # ``atr50_std.clip(lower=1e-9)`` and then clipped the quotient to +/-6.0.
    # The clip's own stated premise was the epsilon floor ("without an output
    # clip, atr_z divided by a 1e-9-floored std blows up unbounded"); removing
    # the floor removes the premise, and an undefined z-score must be reported
    # as unavailable rather than as a bounded number (rule 2e).  The surviving
    # construct is exactly basic_v1's own z-score owner,
    # ``_divide_where_positive(arr - mean, standard_deviation)`` -- no epsilon,
    # no clip.  Removing an exception is allowed; no magnitude is invented.
    # The moments are taken on the float64 series, not on the float32 storage
    # column, per the technical_indicators_v1 block convention ("callers
    # consume this exact float64 block and cast to float32 afterwards").
    df["atr_z"] = (
        (atr50 - atr50_mean).div(atr50_std.where(atr50_std > 0.0))
    ).astype(np.float32)

    # Returns.  ``ret_5`` is RETIRED here (2026-08-19): it was
    # ``close.pct_change(5) * 1e4``, character-identical to the live
    # ctx_cont field ``close_return_5_bps``, whose own declared formula is
    # ``close_return_5_bps[t] = (close[t]/close[t-5] - 1) * 10000``
    # (gx1.features.micro_structure_v1) on the same closed bars -- one field,
    # two owners (rule 19).  ``ret_1`` and ``ret_20`` are NOT duplicated: the
    # micro family carries only the 3-bar and 5-bar returns and the 1-bar
    # return ACCELERATION, never the 1-bar or 20-bar return itself.
    df["ret_1"] = (close.pct_change(1) * 10000.0).astype(np.float32)
    df["ret_20"] = (close.pct_change(20) * 10000.0).astype(np.float32)
    df["roc100"] = (close.pct_change(100) * 10000.0).astype(np.float32)

    # Realized vol (bps).  2026-08-19 warmup repair: ``max(2, window // 4)``
    # had no named origin (rule 2a) and let a 20-bar realized volatility be
    # estimated from 5 returns.  The window is the field's own declared length;
    # first finite row is the window itself (the first pct_change is NaN), 20
    # and 60, both inside the declared surface floor.  Every EMITTED row is
    # bit-identical either way -- past index 20 and 60 the window is already
    # full under the retired min_periods too -- so the TRAIN-fitted
    # entry_volatility_semantics_v1 scale that reads rvol_20 needs no refit.
    close_returns = close.pct_change()

    def rvol_window(window: int) -> pd.Series:
        return (
            close_returns.rolling(window, min_periods=window).std()
            * 10000.0
            * np.sqrt(window)
        )

    rvol_20 = rvol_window(20)
    rvol_60 = rvol_window(60)
    df["rvol_20"] = rvol_20.astype(np.float32)
    df["rvol_60"] = rvol_60.astype(np.float32)

    # EMA trend block.
    # 2026-08-19 EMA-convention repair.  This block used
    # ``close.ewm(span=..., adjust=False)`` -- a seedless recursion that starts
    # from close[0] and emits a value on every row, including rows where the
    # span has not been observed -- while ``basic_v1._ema`` in the SAME base
    # signal block, ``technical_indicators_v1.classic_ema`` and the local
    # price-derived layer all use the classic SMA-seeded EMA.  Two disagreeing
    # implementations of one concept inside one block is the defect class of
    # 2026-08-13 (rule 25), and the seedless variant additionally fabricates
    # values inside its own warmup.  The classic owner wins because it is the
    # one every other EMA consumer on this surface already imports; no third
    # convention is introduced.  First finite rows are DERIVED from the spans:
    # ema20 at 19, ema100 at 99, ema200 at 199.
    #
    # 2026-08-19 volatility-coupling repair (same class as basic_v1's
    # ``_v1_ema3_ema6_spread_atr``).  The three fields were price-relative
    # (delta / price * 1e4), so their magnitude carried the volatility regime
    # rather than the trend.  MEASURED on the complete declared native M5 tape
    # XAU_M5_NATIVE_2019_20260804_V4 (537,861 rows, 2019-01-01..2026-08-04),
    # on the 537,657 rows left after a 201-row causal trim (the declared floor
    # at the hour of measurement), as the IQR width of the last third
    # against the first third
    # (179,219 rows per third, complete population, not a sample):
    #   retired ema20_slope 1.36, ema100_slope 1.33, pos_vs_ema200 1.32
    #   -- against 1.30 for the raw ``ret_1`` they should NOT be tracking,
    #   and 1.23 for the raw ATR/price level itself;
    #   repaired ema20_slope_atr 1.00 and ema100_slope_atr 1.00.
    # "45 bps above the slow EMA" cannot be told apart from "a loud market";
    # "1.2 ATR" can.
    # No magnitude is invented: the denominator becomes the exact ATR-multiple
    # convention already used by the per-TF sibling field of the SAME NAME --
    # htf_features emits ``ema20_slope_atr = (ema20 - ema20.shift(5)) /
    # atr_positive`` on every lane -- so this repair also ends a local/MTF
    # convention split for one named concept.  The k-bar lookbacks 5 and 20 are
    # the fields' own pre-existing constants and are unchanged.
    # Renamed with the repair, per this repository's standing rule that a
    # value/unit change must not keep a name a stale artifact could match
    # (``_v1h1_atr_bps``, ``vwap_rolling5_slope_atr``,
    # ``_v1_ema3_ema6_spread_atr``).
    # Rule 4 -- no market evidence is removed.  The numerators are preserved
    # exactly; only the unit changes, from price-fraction to ATR multiples.
    # The price-relative view of the same displacement is still carried by the
    # mandatory causal layer's local EMA block, and the volatility level itself
    # by ``_v1_atr14`` / ``atr50`` / ``rvol_20``, so the model can still form
    # (delta / close) as (ATR-multiple) x (_v1_atr14 / close) if that is what
    # it wants.
    #
    # ``pos_vs_ema200`` is RETIRED here (2026-08-19) rather than repaired.  Its
    # repaired form would have been ``(close - ema200) / atr14_positive`` over
    # the classic SMA-seeded EMA200 -- which is, term for term, the field the
    # mandatory causal layer already owns on this same local clock
    # (``chart.local_price_vs_ema200_*`` in
    # gx1.features.entry_model_native_feature_layers_v1, computed from
    # technical_indicators_v1.ema50_200_spread_atr_block's ``ema200`` and
    # ``atr14_positive``, i.e. the same classic_ema and the same
    # wilder_atr14_positive this block imports).  Keeping both would create a
    # second owner for one field (rule 19).  Rule 4 holds without the repair
    # too: with ``p = (close - ema200)/close * 1e4`` the retired quantity is the
    # exact function ``(close - ema200)/ema200 * 1e4 == p / (1 - p/1e4)``, so
    # the retained field determines it on every row where close > 0.
    ema20 = classic_ema(close, 20)
    ema100 = classic_ema(close, 100)
    df["ema20_slope_atr"] = (
        (ema20 - ema20.shift(5)).div(atr14_positive)
    ).astype(np.float32)
    df["ema100_slope_atr"] = (
        (ema100 - ema100.shift(20)).div(atr14_positive)
    ).astype(np.float32)

    # vol_ratio: rvol_20 / rvol_60.  2026-08-19 epsilon repair: the retired
    # ``clip(lower=1e-6)`` turned a zero 60-bar realized volatility -- a
    # completely flat window, an availability state -- into a ratio of up to
    # 1e6.  Same divide-where-positive convention as above.
    df["vol_ratio"] = (rvol_20.div(rvol_60.where(rvol_60 > 0.0))).astype(
        np.float32
    )

    for feature in PLUS5_FEATURES:
        df[feature] = plus5[feature]
    return df


def build_basic_v1_chunked(
    df: pd.DataFrame,
    chunk_size: int = 100_000,
    *,
    decision_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """Run the local-only basic_v1 owner over the full tape."""

    # basic_v1 expects DataFrame with DatetimeIndex
    work = df.set_index("time").copy()

    print(f"[{ACTION}] running basic_v1 on {len(work):,} rows...", flush=True)
    t0 = _time.time()
    if not isinstance(decision_bar_duration, pd.Timedelta) or decision_bar_duration <= pd.Timedelta(0):
        raise RuntimeError("[BUILD_CANONICAL_FEATURES_V1] decision bar duration must be positive")
    result = build_basic_v1(
        work,
        decision_delay_seconds=int(decision_bar_duration.total_seconds()),
    )
    if isinstance(result, tuple):
        result = result[0]
    elapsed = _time.time() - t0
    print(f"[{ACTION}] basic_v1 done in {elapsed:.1f}s, output shape: {result.shape}", flush=True)
    result = result.reset_index().rename(columns={"index": "time"})
    if "time" not in result.columns:
        result["time"] = work.index.values
    return result


def build_canonical_features(
    m5: pd.DataFrame,
    *,
    decision_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """Apply basic_v1 + high-level basics + m5_phase. Returns full feature parquet."""
    feats = build_basic_v1_chunked(
        m5,
        decision_bar_duration=decision_bar_duration,
    )

    # Add high-level basics from raw bars
    feats = add_high_level_basics(feats)

    # Add m5_phase one-hot
    feats = add_m5_phase(feats)

    return feats
