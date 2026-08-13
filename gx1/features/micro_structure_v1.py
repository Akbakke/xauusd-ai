"""One-truth causal M5 micro-structure features for model-native Entry."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.features.model_native_market_context_v1 import derive_observed_spread_bps


MICRO_FEATURE_NAMES_V1 = (
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
    "wick_ratio",
    "distance_ema_fast",
)
MICRO_EMA_SPAN_V1 = 5

# V30 package 4 (2026-08-13) — quote/spread dynamics.
#
# PURPOSE: abstention and execution-regime evidence.  These three fields say
# how expensive and how disorderly the quote is at the decision bar; they are
# NOT a direction signal and must never be read as one: orthogonal
# microstructure sources were OOS-refuted for DIRECTION on the retired chain,
# and that fence stands.  The model may learn to abstain or to size down when
# execution conditions are hostile; no post-model rule may consume them
# (rule 3: argmax stays the only decision authority).
#
# Why they exist at all: the ctx surface carried exactly ONE spread field
# (``spread_bps``, a level) and no quote-dynamics evidence, while the legacy
# ``_v1_spread_z`` died CONSTANT on the retired canonical route.  Measured on
# the complete declared native M5 tape 2026-08-13 (N=537,861 rows): spread_bps
# mean 2.23 / std 2.61 / p1 1.11 / p99 14.66 with ~3261 distinct rounded
# values; the 1-bar change is nonzero on 99.93% of rows with std 1.161; the
# intrabar quote envelope (ask_high - bid_low) is 10.21 bps mean / 8.64 std;
# the quote-range asymmetry is nonzero on 87.85% of rows with std 0.969 bps.
# Three fields, no thresholds, no clips, no new magnitudes.
SPREAD_DYNAMICS_FEATURE_NAMES_V1 = (
    "spread_bps_delta_1",
    "spread_intrabar_range_bps",
    "quote_range_asymmetry_bps",
)
# Exact source columns.  All seven are members of
# ``xau_tape_provenance_v1.CANONICAL_NATIVE_REQUIRED_COLUMNS`` (plus the mid
# ``close``), so every offline ctx producer's frame carries them by contract.
SPREAD_DYNAMICS_SOURCE_COLUMNS_V1 = (
    "close",
    "bid_close",
    "ask_close",
    "bid_high",
    "bid_low",
    "ask_high",
    "ask_low",
)
# ``spread_bps_delta_1`` needs one preceding row and has no value on the first
# row of any frame.  That row is an honest NaN, never a parked zero that would
# read as "the spread did not move" (rule 2e); the ctx warmup-trim contract
# removes it, exactly as it removes the swing pivot-delta prefixes.
SPREAD_DYNAMICS_CAUSAL_WARMUP_ROWS_V1 = 1
# Exactly the emitted fields that carry that NaN prefix.  The ctx producers
# join this tuple into their causal warmup-trim lists; the other two fields are
# defined on every row and must NOT be listed (a field with no prefix in a trim
# list is a silent no-op today and a wrong trim after any later change).
SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1 = ("spread_bps_delta_1",)
if set(SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1) - set(
    SPREAD_DYNAMICS_FEATURE_NAMES_V1
):
    raise RuntimeError("SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_UNKNOWN")


def compute_micro_structure_features(
    high,
    low,
    close,
    *,
    ema_span: int = MICRO_EMA_SPAN_V1,
    eps: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Return the exact causal five-field micro surface.

    Momentum values without enough preceding bars have the contract-defined
    boundary value zero. No future row or external feature copy participates.
    """

    h = np.asarray(high, dtype=np.float64)
    low_values = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    if h.ndim != 1 or low_values.ndim != 1 or c.ndim != 1:
        raise RuntimeError(
            "MICRO_STRUCTURE_SOURCE_NOT_1D: "
            f"high={h.shape} low={low_values.shape} close={c.shape}"
        )
    if not (len(h) == len(low_values) == len(c)) or len(c) == 0:
        raise RuntimeError(
            "MICRO_STRUCTURE_SOURCE_LENGTH_INVALID: "
            f"high={len(h)} low={len(low_values)} close={len(c)}"
        )
    if (
        not np.isfinite(h).all()
        or not np.isfinite(low_values).all()
        or not np.isfinite(c).all()
    ):
        raise RuntimeError("MICRO_STRUCTURE_SOURCE_NONFINITE")
    if np.any(h <= 0.0) or np.any(low_values <= 0.0) or np.any(c <= 0.0):
        raise RuntimeError("MICRO_STRUCTURE_SOURCE_NONPOSITIVE")
    if np.any(h < low_values) or np.any(h < c) or np.any(low_values > c):
        raise RuntimeError("MICRO_STRUCTURE_SOURCE_GEOMETRY_INVALID")
    if isinstance(ema_span, bool) or not isinstance(ema_span, int) or ema_span < 1:
        raise RuntimeError(f"MICRO_STRUCTURE_EMA_SPAN_INVALID: {ema_span!r}")
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise RuntimeError(f"MICRO_STRUCTURE_EPS_INVALID: {eps!r}")

    close_series = pd.Series(c)
    # Momentum/acceleration/EMA distance are emitted in bps of the current
    # close (repo ret_* convention: materialize_build_canonical_features_v1
    # pct_change * 10000). Raw USD diffs are era proxies on gold's multi-year
    # price drift and are forbidden as decision inputs. The fillna(0.0)
    # boundary rows keep the contract-defined warmup value; they are masked by
    # the global warmup downstream.
    momentum_3 = (
        close_series.diff(3).fillna(0.0).to_numpy(dtype=np.float64) / c * 1e4
    )
    momentum_5 = (
        close_series.diff(5).fillna(0.0).to_numpy(dtype=np.float64) / c * 1e4
    )
    acceleration = (
        close_series.diff().diff().fillna(0.0).to_numpy(dtype=np.float64) / c * 1e4
    )
    wick_ratio = (h - c) / (h - low_values + float(eps))
    # A zero-range bar carries no close-location evidence; emit the neutral
    # 0.5 (basic_v1._v1_clv's documented convention for the identical
    # degenerate case) instead of the fabricated 0.0 ("closed at high").
    wick_ratio = np.where(h == low_values, 0.5, wick_ratio)
    ema_fast = close_series.ewm(span=ema_span, adjust=False).mean().to_numpy()

    result = {
        "micro_momentum_3": momentum_3.astype(np.float32),
        "micro_momentum_5": momentum_5.astype(np.float32),
        "micro_acceleration": acceleration.astype(np.float32),
        "wick_ratio": wick_ratio.astype(np.float32),
        "distance_ema_fast": ((c - ema_fast) / c * 1e4).astype(np.float32),
    }
    if tuple(result) != MICRO_FEATURE_NAMES_V1 or any(
        values.shape != c.shape or not np.isfinite(values).all()
        for values in result.values()
    ):
        raise RuntimeError("MICRO_STRUCTURE_OUTPUT_INVALID")
    return result


def compute_spread_dynamics_features(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the exact causal three-field quote/spread-dynamics surface.

    Causality: every input is an aggregate of the decision bar's OWN closed
    quotes.  ``bid_close``/``ask_close`` are the same two columns the ctx
    ``spread_bps`` owner reads (``entry_model_native_state_v2
    .compute_causal_market_rank_inputs`` builds ``frame[["bid_close",
    "ask_close"]]`` and hands it to ``derive_observed_spread_bps``); the four
    extremes are that bar's own high/low of the bid and ask series.  This is
    the identical closed-bar convention every other current-bar ctx field uses,
    so the block exposes nothing at ``t`` that the surface does not already
    expose at ``t``.  ``spread_bps_delta_1`` additionally reads row ``t-1``
    only.  No forward row participates anywhere.

    Units: bps of the current mid close, this file's own ``/ close * 1e4``
    convention (``micro_momentum_*``/``micro_acceleration``/
    ``distance_ema_fast``) and ``derive_observed_spread_bps``'s ``* 1e4``.
    No threshold, no clip and no new magnitude is introduced: this owner clips
    nothing, and neither does this block.  ``quote_range_asymmetry_bps`` stays
    SIGNED (an ask range wider than the bid range is different evidence from
    the reverse).

    One spread owner: the level is NOT recomputed here.  It is produced by
    ``derive_observed_spread_bps`` from the same two columns and the same
    ``(ask - bid) / bid * 1e4`` formula that puts ``spread_bps`` on the ctx
    surface, so ``spread_bps_delta_1[t] == spread_bps[t] - spread_bps[t-1]``
    holds exactly on the emitted ctx column.
    """

    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError(
            f"SPREAD_DYNAMICS_SOURCE_FRAME_INVALID: {type(frame).__name__}"
        )
    if len(frame) == 0:
        raise RuntimeError("SPREAD_DYNAMICS_SOURCE_ROWS_EMPTY")
    missing = [
        name
        for name in SPREAD_DYNAMICS_SOURCE_COLUMNS_V1
        if name not in frame.columns
    ]
    if missing:
        raise RuntimeError(f"SPREAD_DYNAMICS_SOURCE_FIELDS_MISSING: {missing}")

    values: dict[str, np.ndarray] = {}
    for name in SPREAD_DYNAMICS_SOURCE_COLUMNS_V1:
        try:
            column = pd.to_numeric(frame[name], errors="raise").to_numpy(
                dtype=np.float64
            )
        except Exception as exc:
            raise RuntimeError(
                f"SPREAD_DYNAMICS_SOURCE_COLUMN_INVALID: {name}"
            ) from exc
        if column.ndim != 1:
            raise RuntimeError(
                f"SPREAD_DYNAMICS_SOURCE_NOT_1D: {name} shape={column.shape}"
            )
        if not np.isfinite(column).all():
            raise RuntimeError(f"SPREAD_DYNAMICS_SOURCE_NONFINITE: {name}")
        if np.any(column <= 0.0):
            raise RuntimeError(f"SPREAD_DYNAMICS_SOURCE_NONPOSITIVE: {name}")
        values[name] = column

    close = values["close"]
    bid_high = values["bid_high"]
    bid_low = values["bid_low"]
    ask_high = values["ask_high"]
    ask_low = values["ask_low"]
    # A quote bar is valid only when each side's own high dominates its low and
    # the ask series dominates the bid series.  Both are data-integrity facts of
    # a two-sided quote, not chosen tolerances; a violation is a broken tape and
    # fails closed.  Together they also prove ask_high >= bid_low, so the
    # intrabar envelope below is non-negative by construction.
    if (
        np.any(bid_high < bid_low)
        or np.any(ask_high < ask_low)
        or np.any(ask_high < bid_high)
        or np.any(ask_low < bid_low)
    ):
        raise RuntimeError("SPREAD_DYNAMICS_SOURCE_QUOTE_GEOMETRY_INVALID")

    spread_bps = np.asarray(
        derive_observed_spread_bps(frame[["bid_close", "ask_close"]].copy()),
        dtype=np.float64,
    )
    if spread_bps.shape != close.shape:
        raise RuntimeError(
            f"SPREAD_DYNAMICS_SPREAD_SHAPE_INVALID: {spread_bps.shape}"
        )
    spread_delta_1 = np.empty_like(spread_bps)
    spread_delta_1[0] = np.nan
    spread_delta_1[1:] = spread_bps[1:] - spread_bps[:-1]

    intrabar_range_bps = (ask_high - bid_low) / close * 1e4
    quote_range_asymmetry_bps = (
        (ask_high - ask_low) - (bid_high - bid_low)
    ) / close * 1e4

    result = {
        "spread_bps_delta_1": spread_delta_1.astype(np.float32),
        "spread_intrabar_range_bps": intrabar_range_bps.astype(np.float32),
        "quote_range_asymmetry_bps": quote_range_asymmetry_bps.astype(np.float32),
    }
    if tuple(result) != SPREAD_DYNAMICS_FEATURE_NAMES_V1 or any(
        array.shape != close.shape for array in result.values()
    ):
        raise RuntimeError("SPREAD_DYNAMICS_OUTPUT_INVALID")
    warmup = SPREAD_DYNAMICS_CAUSAL_WARMUP_ROWS_V1
    delta = result["spread_bps_delta_1"]
    if not np.isnan(delta[:warmup]).all() or not np.isfinite(delta[warmup:]).all():
        raise RuntimeError("SPREAD_DYNAMICS_WARMUP_PREFIX_INVALID")
    for name in ("spread_intrabar_range_bps", "quote_range_asymmetry_bps"):
        if not np.isfinite(result[name]).all():
            raise RuntimeError(f"SPREAD_DYNAMICS_OUTPUT_NONFINITE: {name}")
    if np.any(result["spread_intrabar_range_bps"] < 0.0):
        raise RuntimeError("SPREAD_DYNAMICS_INTRABAR_RANGE_NEGATIVE")
    return result
