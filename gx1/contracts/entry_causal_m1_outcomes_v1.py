"""Exact M5-decision to M1-fill outcome primitives for Entry auxiliaries.

The Entry model observes a completed M5 bar.  A historical auxiliary label may
therefore not enter at that bar's stored close: the first represented
executable quote is the authoritative M1 bid/ask open at the M5 decision
availability time.  This module materializes that relationship and computes
forward outcomes without inventing quotes across missing minutes or closures.

It is intentionally independent of model training and of a fitted policy.  It
supplies the causal price/path evidence that the direction, sizing and
representation-auxiliary owners must consume in a successor rebuild.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    EXIT_DECISION_BAR_SECONDS,
)


ENTRY_CAUSAL_M1_OUTCOME_SCHEMA_VERSION = "entry_causal_m1_outcomes_v1"
ENTRY_CAUSAL_M1_DECISION_TIME_SEMANTICS = (
    "authoritative_m5_bar_close_available_at"
)
ENTRY_CAUSAL_M1_LONG_ENTRY_PRICE = (
    "ask_open_first_authoritative_m1_at_or_after_entry_decision"
)
ENTRY_CAUSAL_M1_SHORT_ENTRY_PRICE = (
    "bid_open_first_authoritative_m1_at_or_after_entry_decision"
)
ENTRY_CAUSAL_M1_LONG_EXIT_PRICE = (
    "bid_open_first_authoritative_m1_at_or_after_fitted_exit_decision"
)
ENTRY_CAUSAL_M1_SHORT_EXIT_PRICE = (
    "ask_open_first_authoritative_m1_at_or_after_fitted_exit_decision"
)
ENTRY_CAUSAL_M1_FILL_BINDING = "exact_m1_quote_time_and_bid_ask"

_M1_REQUIRED_COLUMNS = (
    "time",
    "bid_open",
    "bid_high",
    "bid_low",
    "ask_open",
    "ask_high",
    "ask_low",
)
_SURFACE_COLUMNS = (
    "time",
    "entry_decision_at",
    "entry_m1_row",
    "entry_bid",
    "entry_ask",
    "entry_fill_bound",
)


@dataclass(frozen=True)
class CausalM1QuoteSource:
    """Validated immutable-in-practice M1 arrays reusable across horizons."""

    times: pd.DatetimeIndex
    values: dict[str, np.ndarray]


def causal_m1_target_contract() -> dict[str, Any]:
    """Return the exact quote semantics required by the causality launch gate."""

    return {
        "schema_version": ENTRY_CAUSAL_M1_OUTCOME_SCHEMA_VERSION,
        "entry_decision_time": ENTRY_CAUSAL_M1_DECISION_TIME_SEMANTICS,
        "long_entry_price": ENTRY_CAUSAL_M1_LONG_ENTRY_PRICE,
        "short_entry_price": ENTRY_CAUSAL_M1_SHORT_ENTRY_PRICE,
        "long_exit_price": ENTRY_CAUSAL_M1_LONG_EXIT_PRICE,
        "short_exit_price": ENTRY_CAUSAL_M1_SHORT_EXIT_PRICE,
        "entry_fill_binding": ENTRY_CAUSAL_M1_FILL_BINDING,
        "path_extrema": "authoritative_m1_bid_ask_ohlc_between_fill_and_exit",
        "missing_or_gapped_m1_path": "label_invalid_not_price_substituted",
        "target_affects_feature_availability": False,
    }


def _utc_index(values: Any, *, name: str, seconds: int) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="coerce"))
    except Exception as exc:
        raise RuntimeError(f"ENTRY_CAUSAL_M1_{name}_TIME_INVALID") from exc
    index = index.as_unit("ns")
    if (
        len(index) == 0
        or index.hasnans
        or not index.is_unique
        or not index.is_monotonic_increasing
        or not index.floor(f"{seconds}s").equals(index)
    ):
        raise RuntimeError(f"ENTRY_CAUSAL_M1_{name}_TIME_INVALID")
    return index


def _m1_arrays(
    m1: pd.DataFrame | CausalM1QuoteSource,
) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    if isinstance(m1, CausalM1QuoteSource):
        if (
            not isinstance(m1.times, pd.DatetimeIndex)
            or set(m1.values) != set(_M1_REQUIRED_COLUMNS[1:])
        ):
            raise RuntimeError("ENTRY_CAUSAL_M1_PREPARED_SOURCE_INVALID")
        return m1.times, m1.values
    if not isinstance(m1, pd.DataFrame) or any(
        column not in m1.columns for column in _M1_REQUIRED_COLUMNS
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_SOURCE_COLUMNS_INVALID")
    times = _utc_index(m1["time"], name="SOURCE_M1", seconds=EXIT_DECISION_BAR_SECONDS)
    numeric: dict[str, np.ndarray] = {}
    for column in _M1_REQUIRED_COLUMNS[1:]:
        values = pd.to_numeric(m1[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise RuntimeError(f"ENTRY_CAUSAL_M1_SOURCE_VALUES_INVALID:{column}")
        numeric[column] = values
    if (
        np.any(numeric["bid_high"] < numeric["bid_open"])
        or np.any(numeric["bid_low"] > numeric["bid_open"])
        or np.any(numeric["ask_high"] < numeric["ask_open"])
        or np.any(numeric["ask_low"] > numeric["ask_open"])
        or np.any(numeric["ask_open"] <= numeric["bid_open"])
        or np.any(numeric["ask_high"] <= numeric["bid_high"])
        or np.any(numeric["ask_low"] <= numeric["bid_low"])
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_SOURCE_QUOTE_GEOMETRY_INVALID")
    return times, numeric


def prepare_causal_m1_quote_source(closed_m1: pd.DataFrame) -> CausalM1QuoteSource:
    """Validate M1 once and retain its exact arrays for a multi-horizon fit.

    Prepared input is an in-memory optimisation only: it does not weaken the
    source checks and callers still bind the parquet's SHA-256 in their
    artifact lineage.  Arrays are marked read-only to prevent accidental
    mutation between the 96 candidate-horizon evaluations.
    """

    times, values = _m1_arrays(closed_m1)
    prepared = {name: np.asarray(value, dtype=np.float64).copy() for name, value in values.items()}
    for value in prepared.values():
        value.setflags(write=False)
    return CausalM1QuoteSource(times=times.copy(), values=prepared)


def build_entry_m1_fill_surface(
    *,
    m5_decision_times: Sequence[Any] | pd.DatetimeIndex,
    closed_m1: pd.DataFrame | CausalM1QuoteSource,
) -> pd.DataFrame:
    """Bind every M5 decision time to its exact first represented M1 fill.

    The result retains M5 rows whose contemporaneous M1 quote is absent but
    marks them unbound. Consumers must exclude these rows; they must not
    forward-fill, substitute a M5 close or silently replace the quote with zero.
    """

    m5_times = _utc_index(
        m5_decision_times,
        name="DECISION_M5",
        seconds=ENTRY_DECISION_BAR_SECONDS,
    )
    m1_times, values = _m1_arrays(closed_m1)
    decision_at = m5_times + pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS)
    positions = np.searchsorted(m1_times.asi8, decision_at.asi8, side="left")
    bound = positions < len(m1_times)
    candidates = np.flatnonzero(bound)
    bound[candidates] &= (
        m1_times.asi8[positions[candidates]] == decision_at.asi8[candidates]
    )
    entry_rows = np.full(len(m5_times), -1, dtype=np.int64)
    entry_rows[bound] = positions[bound]
    entry_bid = np.full(len(m5_times), np.nan, dtype=np.float64)
    entry_ask = np.full(len(m5_times), np.nan, dtype=np.float64)
    entry_bid[bound] = values["bid_open"][positions[bound]]
    entry_ask[bound] = values["ask_open"][positions[bound]]
    return pd.DataFrame(
        {
            "time": m5_times,
            "entry_decision_at": decision_at,
            "entry_m1_row": entry_rows,
            "entry_bid": entry_bid,
            "entry_ask": entry_ask,
            "entry_fill_bound": bound,
        },
        columns=_SURFACE_COLUMNS,
    )


def _require_surface(surface: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(surface, pd.DataFrame) or tuple(surface.columns) != _SURFACE_COLUMNS:
        raise RuntimeError("ENTRY_CAUSAL_M1_FILL_SURFACE_SCHEMA_INVALID")
    m5_times = _utc_index(
        surface["time"], name="FILL_SURFACE_M5", seconds=ENTRY_DECISION_BAR_SECONDS
    )
    decision_at = _utc_index(
        surface["entry_decision_at"],
        name="FILL_SURFACE_DECISION",
        seconds=EXIT_DECISION_BAR_SECONDS,
    )
    expected_decision = m5_times + pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS)
    if not decision_at.equals(expected_decision):
        raise RuntimeError("ENTRY_CAUSAL_M1_FILL_SURFACE_DECISION_CLOCK_INVALID")
    rows = pd.to_numeric(surface["entry_m1_row"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    bound = surface["entry_fill_bound"].to_numpy()
    if (
        bound.dtype != np.bool_
        or not np.isfinite(rows).all()
        or not np.equal(rows, np.floor(rows)).all()
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_FILL_SURFACE_BINDING_INVALID")
    result = surface.copy()
    result["time"] = m5_times
    result["entry_decision_at"] = decision_at
    result["entry_m1_row"] = rows.astype(np.int64)
    for column in ("entry_bid", "entry_ask"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    valid_quotes = (
        np.isfinite(result["entry_bid"].to_numpy(dtype=np.float64))
        & np.isfinite(result["entry_ask"].to_numpy(dtype=np.float64))
        & (result["entry_bid"].to_numpy(dtype=np.float64) > 0.0)
        & (result["entry_ask"].to_numpy(dtype=np.float64)
           > result["entry_bid"].to_numpy(dtype=np.float64))
    )
    if (
        np.any(bound & ~valid_quotes)
        or np.any(~bound & (rows.astype(np.int64) != -1))
        or np.any(~bound & valid_quotes)
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_FILL_SURFACE_QUOTE_INVALID")
    return result


def _forward_extreme(values: np.ndarray, *, width: int, maximum: bool) -> np.ndarray:
    """Extrema over [row, row + width), without a giant 2-D window matrix."""

    if width < 1 or width > len(values):
        raise RuntimeError("ENTRY_CAUSAL_M1_PATH_WIDTH_INVALID")
    series = pd.Series(np.asarray(values, dtype=np.float64)[::-1])
    rolling = (
        series.rolling(width, min_periods=width).max()
        if maximum
        else series.rolling(width, min_periods=width).min()
    )
    return rolling.to_numpy(dtype=np.float64)[::-1]


def causal_m1_outcomes_at_horizon(
    *,
    fill_surface: pd.DataFrame,
    closed_m1: pd.DataFrame | CausalM1QuoteSource,
    horizon_m5_bars: int,
) -> pd.DataFrame:
    """Compute executable PnL and MFE/MAE from the exact M1 fill timeline.

    An invalid row has no forward-filled quote and no guessed path: all numeric
    outcome fields are NaN and ``outcome_valid`` is false. The path contains
    the M1 bars from the fill open up to, but not including, the later M1-open
    exit decision. Thus every high/low in the MFE/MAE window occurs after fill
    and before the executable exit quote.
    """

    if type(horizon_m5_bars) is not int or horizon_m5_bars < 1:
        raise RuntimeError("ENTRY_CAUSAL_M1_HORIZON_INVALID")
    surface = _require_surface(fill_surface)
    m1_times, values = _m1_arrays(closed_m1)
    m1_bars = horizon_m5_bars * (
        ENTRY_DECISION_BAR_SECONDS // EXIT_DECISION_BAR_SECONDS
    )
    start = surface["entry_m1_row"].to_numpy(dtype=np.int64)
    fill_bound = surface["entry_fill_bound"].to_numpy(dtype=np.bool_)
    end = start + m1_bars
    valid = fill_bound & (start >= 0) & (end < len(m1_times))
    positions = np.flatnonzero(valid)
    expected_start = pd.DatetimeIndex(surface["entry_decision_at"])
    expected_end = expected_start + pd.Timedelta(minutes=5 * horizon_m5_bars)
    valid[positions] &= (
        m1_times.asi8[start[positions]] == expected_start.asi8[positions]
    )
    positions = np.flatnonzero(valid)
    valid[positions] &= (
        m1_times.asi8[end[positions]] == expected_end.asi8[positions]
    )

    size = len(surface)
    exit_bid = np.full(size, np.nan, dtype=np.float64)
    exit_ask = np.full(size, np.nan, dtype=np.float64)
    max_bid = _forward_extreme(values["bid_high"], width=m1_bars, maximum=True)
    min_bid = _forward_extreme(values["bid_low"], width=m1_bars, maximum=False)
    max_ask = _forward_extreme(values["ask_high"], width=m1_bars, maximum=True)
    min_ask = _forward_extreme(values["ask_low"], width=m1_bars, maximum=False)
    entry_bid = surface["entry_bid"].to_numpy(dtype=np.float64)
    entry_ask = surface["entry_ask"].to_numpy(dtype=np.float64)
    exit_bid[valid] = values["bid_open"][end[valid]]
    exit_ask[valid] = values["ask_open"][end[valid]]
    long_pnl = np.full(size, np.nan, dtype=np.float64)
    short_pnl = np.full(size, np.nan, dtype=np.float64)
    long_mfe = np.full(size, np.nan, dtype=np.float64)
    long_mae = np.full(size, np.nan, dtype=np.float64)
    short_mfe = np.full(size, np.nan, dtype=np.float64)
    short_mae = np.full(size, np.nan, dtype=np.float64)
    long_pnl[valid] = (exit_bid[valid] / entry_ask[valid] - 1.0) * 1e4
    short_pnl[valid] = (entry_bid[valid] / exit_ask[valid] - 1.0) * 1e4
    long_mfe[valid] = (max_bid[start[valid]] / entry_ask[valid] - 1.0) * 1e4
    long_mae[valid] = (1.0 - min_bid[start[valid]] / entry_ask[valid]) * 1e4
    short_mfe[valid] = (1.0 - min_ask[start[valid]] / entry_bid[valid]) * 1e4
    short_mae[valid] = (max_ask[start[valid]] / entry_bid[valid] - 1.0) * 1e4
    if (
        not np.isfinite(long_pnl[valid]).all()
        or not np.isfinite(short_pnl[valid]).all()
        or np.any(long_mae[valid] < 0.0)
        or np.any(short_mae[valid] < 0.0)
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_OUTCOME_VALUES_INVALID")
    exit_time = pd.Series(pd.NaT, index=np.arange(size), dtype="datetime64[ns, UTC]")
    exit_time.iloc[valid] = m1_times[end[valid]]
    return pd.DataFrame(
        {
            "time": surface["time"].to_numpy(),
            "entry_decision_at": surface["entry_decision_at"].to_numpy(),
            "exit_decision_at": exit_time.to_numpy(),
            "horizon_m5_bars": np.full(size, horizon_m5_bars, dtype=np.int32),
            "horizon_m1_bars": np.full(size, m1_bars, dtype=np.int32),
            "entry_bid": entry_bid,
            "entry_ask": entry_ask,
            "exit_bid": exit_bid,
            "exit_ask": exit_ask,
            "long_executable_pnl_bps": long_pnl,
            "short_executable_pnl_bps": short_pnl,
            "long_mfe_bps": long_mfe,
            "long_mae_bps": long_mae,
            "short_mfe_bps": short_mfe,
            "short_mae_bps": short_mae,
            "outcome_valid": valid,
        }
    )


def causal_m1_terminal_outcomes_at_horizon(
    *,
    fill_surface: pd.DataFrame,
    closed_m1: pd.DataFrame | CausalM1QuoteSource,
    horizon_m5_bars: int,
) -> pd.DataFrame:
    """Return exact terminal executable PnL without allocating path extrema.

    Policy fitting probes all candidate horizons.  It needs the later
    executable bid/ask quote, but not MFE/MAE, so this preserves the exact
    causal clock without repeatedly creating four rolling M1 extrema arrays.
    """

    if type(horizon_m5_bars) is not int or horizon_m5_bars < 1:
        raise RuntimeError("ENTRY_CAUSAL_M1_HORIZON_INVALID")
    surface = _require_surface(fill_surface)
    m1_times, values = _m1_arrays(closed_m1)
    m1_bars = horizon_m5_bars * (
        ENTRY_DECISION_BAR_SECONDS // EXIT_DECISION_BAR_SECONDS
    )
    start = surface["entry_m1_row"].to_numpy(dtype=np.int64)
    fill_bound = surface["entry_fill_bound"].to_numpy(dtype=np.bool_)
    end = start + m1_bars
    valid = fill_bound & (start >= 0) & (end < len(m1_times))
    positions = np.flatnonzero(valid)
    expected_start = pd.DatetimeIndex(surface["entry_decision_at"])
    expected_end = expected_start + pd.Timedelta(minutes=5 * horizon_m5_bars)
    valid[positions] &= (
        m1_times.asi8[start[positions]] == expected_start.asi8[positions]
    )
    positions = np.flatnonzero(valid)
    valid[positions] &= (
        m1_times.asi8[end[positions]] == expected_end.asi8[positions]
    )
    size = len(surface)
    exit_bid = np.full(size, np.nan, dtype=np.float64)
    exit_ask = np.full(size, np.nan, dtype=np.float64)
    exit_bid[valid] = values["bid_open"][end[valid]]
    exit_ask[valid] = values["ask_open"][end[valid]]
    entry_bid = surface["entry_bid"].to_numpy(dtype=np.float64)
    entry_ask = surface["entry_ask"].to_numpy(dtype=np.float64)
    long_pnl = np.full(size, np.nan, dtype=np.float64)
    short_pnl = np.full(size, np.nan, dtype=np.float64)
    long_pnl[valid] = (exit_bid[valid] / entry_ask[valid] - 1.0) * 1e4
    short_pnl[valid] = (entry_bid[valid] / exit_ask[valid] - 1.0) * 1e4
    if not np.isfinite(long_pnl[valid]).all() or not np.isfinite(short_pnl[valid]).all():
        raise RuntimeError("ENTRY_CAUSAL_M1_TERMINAL_OUTCOME_VALUES_INVALID")
    exit_time = pd.Series(pd.NaT, index=np.arange(size), dtype="datetime64[ns, UTC]")
    exit_time.iloc[valid] = m1_times[end[valid]]
    return pd.DataFrame(
        {
            "time": surface["time"].to_numpy(),
            "entry_decision_at": surface["entry_decision_at"].to_numpy(),
            "exit_decision_at": exit_time.to_numpy(),
            "horizon_m5_bars": np.full(size, horizon_m5_bars, dtype=np.int32),
            "horizon_m1_bars": np.full(size, m1_bars, dtype=np.int32),
            "entry_bid": entry_bid,
            "entry_ask": entry_ask,
            "exit_bid": exit_bid,
            "exit_ask": exit_ask,
            "long_executable_pnl_bps": long_pnl,
            "short_executable_pnl_bps": short_pnl,
            "outcome_valid": valid,
        }
    )
