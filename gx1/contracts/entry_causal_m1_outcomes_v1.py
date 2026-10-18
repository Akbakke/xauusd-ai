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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    EXIT_DECISION_BAR_SECONDS,
)


ENTRY_CAUSAL_M1_OUTCOME_SCHEMA_VERSION = "entry_causal_m1_outcomes_v2_entry_notional_pnl"
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
        "long_pnl_bps_formula": "(exit_bid-entry_ask)/entry_ask*10000",
        "short_pnl_bps_formula": "(entry_bid-exit_ask)/entry_bid*10000",
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
    # Match canonical native geometry: positive prices and ask >= bid.
    # Locked quotes are retained exactly; crossed quotes remain invalid.
    if (
        np.any(numeric["bid_high"] < numeric["bid_open"])
        or np.any(numeric["bid_low"] > numeric["bid_open"])
        or np.any(numeric["ask_high"] < numeric["ask_open"])
        or np.any(numeric["ask_low"] > numeric["ask_open"])
        or np.any(numeric["ask_open"] < numeric["bid_open"])
        or np.any(numeric["ask_high"] < numeric["bid_high"])
        or np.any(numeric["ask_low"] < numeric["bid_low"])
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
           >= result["entry_bid"].to_numpy(dtype=np.float64))
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


def _entry_notional_pnl_bps(
    *,
    entry_bid: np.ndarray,
    entry_ask: np.ndarray,
    exit_bid: np.ndarray,
    exit_ask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the Entry-notional return used by the unified Exit rewards.

    A short's notional is fixed at entry. Dividing by its eventual exit quote
    instead would overstate gains, understate losses and disagree with MFE/MAE.
    Both materializers share this calculation so policy fitting and emitted
    auxiliary outcomes cannot use different definitions.
    """

    return (
        (exit_bid - entry_ask) / entry_ask * 1e4,
        (entry_bid - exit_ask) / entry_bid * 1e4,
    )


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
    long_pnl[valid], short_pnl[valid] = _entry_notional_pnl_bps(
        entry_bid=entry_bid[valid], entry_ask=entry_ask[valid],
        exit_bid=exit_bid[valid], exit_ask=exit_ask[valid],
    )
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
    long_pnl[valid], short_pnl[valid] = _entry_notional_pnl_bps(
        entry_bid=entry_bid[valid], entry_ask=entry_ask[valid],
        exit_bid=exit_bid[valid], exit_ask=exit_ask[valid],
    )
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


# Operator decision18Oct2026: first path observation window reuses the exact
# already TRAIN-selected19M5 bars; this is not a new search or holding limit.
M5_PATH_OBSERVATION_HORIZON_BARS = 19
M5_PATH_HORIZON_ORIGIN_POLICY_SHA256 = 'ae0070802493ee21f7232ed61d588362e6a1a8d428bd0b185f6d7a2e56f7c6d3'
M5_PATH_TARGET_FIELDS = ('mfe_bps', 'mae_before_first_mfe_lower_bps',
    'mae_before_first_mfe_upper_bps', 'path_utility_bps', 'first_mfe_m1_offset')
M5_PATH_TARGET_COLUMNS = tuple('y_' + side + '_m5_path_' + field
    for side in ('long', 'short') for field in M5_PATH_TARGET_FIELDS)
M5_PATH_VALID_COLUMN = 'm5_path_target_valid'


def causal_m1_path_quality_outcomes(
    *, fill_surface: pd.DataFrame, closed_m1: pd.DataFrame | CausalM1QuoteSource,
    horizon_m5_bars: int,
) -> pd.DataFrame:
    """Signed MFE minus conservative pre-peak MAE; no hindsight cash return.

    The first favorable extreme is selected on executable BID (LONG) / ASK
    (SHORT). OHLC cannot order high/low inside its M1 bucket. Lower MAE includes
    every earlier bucket and the known peak-bucket opening quote; upper MAE
    additionally includes the entire peak bucket. Primary utility uses upper.
    No extrema, observation horizon or timing is an executable Exit policy.
    """
    terminal = causal_m1_terminal_outcomes_at_horizon(fill_surface=fill_surface,
        closed_m1=closed_m1, horizon_m5_bars=horizon_m5_bars)
    surface = _require_surface(fill_surface)
    _, quotes = _m1_arrays(closed_m1)
    valid = terminal['outcome_valid'].to_numpy(dtype=bool)
    rows = np.flatnonzero(valid)
    starts = surface['entry_m1_row'].to_numpy(dtype=np.int64)[rows]
    bid = surface['entry_bid'].to_numpy(dtype=np.float64)[rows]
    ask = surface['entry_ask'].to_numpy(dtype=np.float64)[rows]
    initial_long, initial_short = _entry_notional_pnl_bps(
        entry_bid=bid, entry_ask=ask, exit_bid=bid, exit_ask=ask)
    output = {'time': surface['time'].to_numpy(),
        'entry_decision_at': surface['entry_decision_at'].to_numpy(),
        'exit_decision_at': terminal['exit_decision_at'].to_numpy(),
        M5_PATH_VALID_COLUMN: valid}
    width = horizon_m5_bars * (ENTRY_DECISION_BAR_SECONDS // EXIT_DECISION_BAR_SECONDS)
    for side, initial in (('long', initial_long), ('short', initial_short)):
        peak = np.full(len(rows), -np.inf)
        running = initial.copy()
        lower, upper = np.zeros(len(rows)), np.zeros(len(rows))
        first_peak = np.zeros(len(rows), dtype=np.int64)
        for offset in range(width):
            positions = starts + offset
            if side == 'long':
                favorable = (quotes['bid_high'][positions] - ask) / ask * 1e4
                adverse = (quotes['bid_low'][positions] - ask) / ask * 1e4
                opened = (quotes['bid_open'][positions] - ask) / ask * 1e4
            else:
                favorable = (bid - quotes['ask_low'][positions]) / bid * 1e4
                adverse = (bid - quotes['ask_high'][positions]) / bid * 1e4
                opened = (bid - quotes['ask_open'][positions]) / bid * 1e4
            new_peak = favorable > peak  # strict: retain first equal maximum
            lower = np.where(new_peak, -np.minimum(running, opened), lower)
            upper = np.where(new_peak, -np.minimum(running, adverse), upper)
            first_peak = np.where(new_peak, offset, first_peak)
            peak = np.maximum(peak, favorable)
            running = np.minimum(running, adverse)
        if (not np.isfinite(peak).all() or not np.isfinite(lower).all()
                or not np.isfinite(upper).all() or np.any(lower < 0) or np.any(upper < lower)):
            raise RuntimeError('ENTRY_M5_PATH_EXTREMA_INVALID')
        values = (peak, lower, upper, peak - upper, first_peak)
        for field, values in zip(M5_PATH_TARGET_FIELDS, values):
            full = np.full(len(surface), np.nan)
            full[rows] = values
            output['y_' + side + '_m5_path_' + field] = full
    return pd.DataFrame(output)


# Explicit successor: resource chunking is not an observation/trading horizon.
DYNAMIC_M1_PREFIX_SCHEMA_VERSION = "gx1_entry_dynamic_m1_prefix_v1"
DYNAMIC_M1_PREFIX_HORIZON_MINUTES = tuple(range(1, 481))
DYNAMIC_M1_PREFIX_SIDE_ORDER = ("long", "short")
DYNAMIC_M1_PREFIX_CLOCK_KIND = "elapsed_wall_clock_minutes"
DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT = "terminal_open_inclusive"
DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT = "closed_m1_prefix"
DYNAMIC_M1_PREFIX_ENDPOINT_KINDS = (
    DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT, DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT,
)
DYNAMIC_M1_PREFIX_QUERY_CHUNK_ROWS = 4096


def dynamic_m1_prefix_contract() -> dict[str, Any]:
    return {
        "schema_version": DYNAMIC_M1_PREFIX_SCHEMA_VERSION,
        "horizon_minutes": list(DYNAMIC_M1_PREFIX_HORIZON_MINUTES),
        "clock_kind": DYNAMIC_M1_PREFIX_CLOCK_KIND,
        "side_order": list(DYNAMIC_M1_PREFIX_SIDE_ORDER),
        "endpoint_kinds": list(DYNAMIC_M1_PREFIX_ENDPOINT_KINDS),
        "endpoint_kind_codes": {kind: code for code, kind in enumerate(DYNAMIC_M1_PREFIX_ENDPOINT_KINDS)},
        "fill_binding": ENTRY_CAUSAL_M1_FILL_BINDING,
        "terminal_markout": "exact_bid_ask_endpoints_entry_notional_gross_bps",
        "terminal_mask_independent_of_path": True,
        "path_fields": ["mfe_bps", "mae_before_first_mfe_lower_bps",
                        "mae_before_first_mfe_upper_bps"],
        "terminal_endpoint_path": "closed_m1_ohlc_before_T_plus_exact_T_open_only",
        "closed_endpoint_path": "closed_m1_ohlc_before_T_without_T_quote",
        "first_equal_peak": "first_exact_equal_extreme_retained",
        "intraminute_order": "lower_upper_bounds_not_invented_tick_order",
        "unknown_gap": "path_censored_terminal_endpoints_remain_independent",
        "known_closure": "only_source_clock_bound_external_exact_schedule",
        "cost_at_peak_materialized": False,
        "holding_time_rule": False,
        "future_horizon_max_target": False,
        "masked_float_storage": "nan",
        "masked_source_row": -1,
        "physical_source_provenance_owner": "caller_verified_quote_source_binding",
    }


@dataclass(frozen=True)
class _DynamicRangeArgIndex:
    values: np.ndarray
    tree: np.ndarray
    leaf_count: int
    maximum: bool

    def _choose(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lv = self.values[np.maximum(left, 0)]
        rv = self.values[np.maximum(right, 0)]
        better = lv > rv if self.maximum else lv < rv
        use_left = (right < 0) | ((left >= 0) &
            (better | ((lv == rv) & (left <= right))))
        return np.where(use_left, left, right)

    def query(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """First arg-extreme on each half-open source interval; empty -> -1."""
        left = np.asarray(starts, dtype=np.int64) + self.leaf_count
        right = np.asarray(ends, dtype=np.int64) + self.leaf_count
        result = np.full(len(left), -1, dtype=np.int64)
        while np.any(left < right):
            active = left < right
            take = active & ((left & 1) != 0)
            result[take] = self._choose(result[take], self.tree[left[take]])
            left[take] += 1
            take = active & ((right & 1) != 0)
            right[take] -= 1
            result[take] = self._choose(result[take], self.tree[right[take]])
            left //= 2
            right //= 2
        return result


def _dynamic_range_arg_index(values: np.ndarray, *, maximum: bool) -> _DynamicRangeArgIndex:
    size = 1 << (len(values) - 1).bit_length()
    tree = np.full(2 * size, -1, dtype=np.int64)
    tree[size:size + len(values)] = np.arange(len(values), dtype=np.int64)
    index = _DynamicRangeArgIndex(values, tree, size, maximum)
    width = size // 2
    while width:
        rows = np.arange(width, 2 * width, dtype=np.int64)
        tree[rows] = index._choose(tree[rows * 2], tree[rows * 2 + 1])
        width //= 2
    tree.setflags(write=False)
    return index


@dataclass(frozen=True)
class DynamicM1PrefixIndex:
    source: CausalM1QuoteSource
    split_end_exclusive: pd.Timestamp
    source_parquet_sha256: str
    source_row_offset: int
    source_clock_sha256: str
    source_rows: int
    indexed_clock_sha256: str
    unknown_gap_prefix: np.ndarray
    known_closure_count: int
    closure_authority_sha256: str | None
    _ranges: Mapping[str, _DynamicRangeArgIndex]

    def binding(self) -> dict[str, Any]:
        """Computed clock/window identity; physical file verification stays with caller."""
        return {
            "schema_version": DYNAMIC_M1_PREFIX_SCHEMA_VERSION,
            "source_parquet_sha256": self.source_parquet_sha256,
            "source_clock_sha256": self.source_clock_sha256,
            "source_rows": self.source_rows,
            "indexed_clock_sha256": self.indexed_clock_sha256,
            "indexed_rows": len(self.source.times),
            "source_row_offset": self.source_row_offset,
            "split_end_exclusive": self.split_end_exclusive.isoformat(),
            "clock_kind": DYNAMIC_M1_PREFIX_CLOCK_KIND,
            "unknown_gap_count": int(self.unknown_gap_prefix[-1]),
            "known_closure_count": self.known_closure_count,
            "closure_authority_sha256": self.closure_authority_sha256,
        }


def prepare_dynamic_m1_prefix_index(
    closed_m1: pd.DataFrame | CausalM1QuoteSource, *,
    split_end_exclusive: Any, source_parquet_sha256: str,
    source_row_offset: int = 0,
    closure_authority: Mapping[str, Any] | None = None,
    exact_schedule: Mapping[str, Any] | None = None,
) -> DynamicM1PrefixIndex:
    """Build one O(N) range index after clock-only split projection.

    The caller verifies the physical quote file and its retained contiguous row
    mapping. This API does not invent a file path or claim to have hashed a file.
    source_clock_sha256 covers all supplied clock metadata; indexed_clock_sha256
    covers only the rows whose prices may be read before the exclusive cutoff.
    CausalM1QuoteSource may carry the full clock with prices for only that
    retained prefix; cutoff/later prices need never be loaded or padded.
    """
    from gx1.contracts.unified_exit_market_closure_authority_v1 import (
        m1_clock_sha256, require_exact_market_schedule,
        require_market_closure_authority,
    )
    if (not isinstance(source_parquet_sha256, str) or len(source_parquet_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_parquet_sha256)):
        raise RuntimeError("DYNAMIC_M1_PREFIX_SOURCE_SHA_INVALID")
    if type(source_row_offset) is not int or source_row_offset < 0:
        raise RuntimeError("DYNAMIC_M1_PREFIX_SOURCE_OFFSET_INVALID")
    if isinstance(closed_m1, CausalM1QuoteSource):
        raw_times = closed_m1.times
    elif isinstance(closed_m1, pd.DataFrame) and "time" in closed_m1:
        raw_times = closed_m1["time"]
    else:
        raise RuntimeError("DYNAMIC_M1_PREFIX_SOURCE_INVALID")
    clock = _utc_index(raw_times, name="DYNAMIC_SOURCE", seconds=EXIT_DECISION_BAR_SECONDS)
    cutoff = pd.Timestamp(split_end_exclusive)
    if pd.isna(cutoff) or cutoff.tzinfo is None or cutoff.floor("min") != cutoff:
        raise RuntimeError("DYNAMIC_M1_PREFIX_SPLIT_CUTOFF_INVALID")
    cutoff = cutoff.tz_convert("UTC")
    retained = int(clock.searchsorted(cutoff, side="left"))
    if not retained:
        raise RuntimeError("DYNAMIC_M1_PREFIX_SOURCE_EMPTY_BEFORE_SPLIT")
    # No value conversion, geometry scan or range-tree read of cutoff/later rows.
    if isinstance(closed_m1, CausalM1QuoteSource):
        if set(closed_m1.values) != set(_M1_REQUIRED_COLUMNS[1:]):
            raise RuntimeError("DYNAMIC_M1_PREFIX_PREPARED_SOURCE_INVALID")
        scoped = pd.DataFrame({"time": clock[:retained], **{
            name: closed_m1.values[name][:retained] for name in _M1_REQUIRED_COLUMNS[1:]
        }})
    else:
        scoped = closed_m1.iloc[:retained]
    source = prepare_causal_m1_quote_source(scoped)
    source = CausalM1QuoteSource(source.times.copy(deep=True), source.values)
    source_clock_sha = m1_clock_sha256(clock)
    gap_rows = np.flatnonzero(np.diff(clock.asi8) != 60 * 10**9)
    known: set[int] = set()
    authority_sha = None
    if (closure_authority is None) != (exact_schedule is None):
        raise RuntimeError("DYNAMIC_M1_PREFIX_CLOSURE_BINDING_INCOMPLETE")
    if closure_authority is not None:
        schedule = require_exact_market_schedule(exact_schedule)
        if schedule["source_method"] not in (
                "externally_sourced_exact_xau_utc_closure_intervals_v1",
                "observed_m1_clock_all_gaps_unknown_v1"):
            raise RuntimeError("DYNAMIC_M1_PREFIX_INFERRED_CLOSURE_FORBIDDEN")
        authority = require_market_closure_authority(
            closure_authority, expected_m1_source_sha256=source_parquet_sha256,
            expected_m1_clock_sha256=source_clock_sha)
        if (authority["schedule_sha256"] != schedule["schedule_sha256"]
                or authority["m1_row_count"] != len(clock)
                or len(authority["intervals"]) != len(gap_rows)
                or authority["coverage_start_utc"] != schedule["coverage_start_utc"]
                or authority["coverage_end_utc_exclusive"] != schedule["coverage_end_utc_exclusive"]
                or pd.Timestamp(schedule["coverage_start_utc"]) > clock[0]
                or pd.Timestamp(schedule["coverage_end_utc_exclusive"]) < clock[-1] + pd.Timedelta(minutes=1)):
            raise RuntimeError("DYNAMIC_M1_PREFIX_CLOSURE_CLOCK_MISMATCH")
        if (schedule["source_method"] == "observed_m1_clock_all_gaps_unknown_v1"
                and schedule["source_reference_sha256"] != source_clock_sha):
            raise RuntimeError("DYNAMIC_M1_PREFIX_CLOSURE_CLOCK_MISMATCH")
        scheduled = {(pd.Timestamp(item["start_utc"]).value,
                      pd.Timestamp(item["end_utc_exclusive"]).value)
                     for item in schedule["intervals"]}
        for row, record in zip(gap_rows, authority["intervals"]):
            interval = (int(clock.asi8[row]) + 60 * 10**9, int(clock.asi8[row + 1]))
            if (record["gap_after_m1_row"] != int(row)
                    or pd.Timestamp(record["previous_bar_start_utc"]).value != int(clock.asi8[row])
                    or pd.Timestamp(record["next_bar_start_utc"]).value != int(clock.asi8[row + 1])
                    or pd.Timestamp(record["closure_start_utc"]).value != interval[0]
                    or pd.Timestamp(record["closure_end_utc_exclusive"]).value != interval[1]
                    or bool(record["successor_across_gap_allowed"]) != (interval in scheduled)):
                raise RuntimeError("DYNAMIC_M1_PREFIX_CLOSURE_INTERVAL_MISMATCH")
            if record["successor_across_gap_allowed"]:
                known.add(int(row))
        authority_sha = authority["artifact_sha256"]
    unknown = np.zeros(retained, dtype=np.int64)
    for row in gap_rows:
        if row < retained and int(row) not in known:
            unknown[row] = 1
    prefix = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(unknown)))
    prefix.setflags(write=False)
    ranges = {name: _dynamic_range_arg_index(source.values[name], maximum=maximum)
              for name, maximum in (("bid_high", True), ("bid_low", False),
                                     ("ask_low", False), ("ask_high", True))}
    return DynamicM1PrefixIndex(
        source, cutoff, source_parquet_sha256, source_row_offset,
        source_clock_sha, len(clock), m1_clock_sha256(source.times), prefix,
        sum(row < retained for row in known), authority_sha, ranges)


def build_dynamic_m1_fill_surface(
    index: DynamicM1PrefixIndex, *, m5_decision_times: Sequence[Any] | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Reuse the exact fill owner, preserving the caller's original source-row map."""
    result = build_entry_m1_fill_surface(m5_decision_times=m5_decision_times,
                                       closed_m1=index.source)
    bound = result["entry_fill_bound"].to_numpy(bool)
    result.loc[bound, "entry_m1_row"] += index.source_row_offset
    return result


def query_dynamic_m1_prefix(
    index: DynamicM1PrefixIndex, *, fill_surface: pd.DataFrame,
    query_entry_rows: Sequence[int] | np.ndarray,
    horizon_minutes: Sequence[int] | np.ndarray,
    endpoint_kind: str = DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT,
    query_chunk_rows: int = DYNAMIC_M1_PREFIX_QUERY_CHUNK_ROWS,
) -> dict[str, Any]:
    """Query O(Q) individual prefixes in resource-bounded internal chunks.

    The source index is shared. No entry-by-all-horizons matrix is created.
    Terminal gross markout remains observable across an unknown interior gap;
    path extrema do not. Net costs and masked batch-zeroing belong to caller.
    """
    if not isinstance(index, DynamicM1PrefixIndex):
        raise RuntimeError("DYNAMIC_M1_PREFIX_INDEX_INVALID")
    entries, horizons = np.asarray(query_entry_rows), np.asarray(horizon_minutes)
    if (entries.ndim != 1 or horizons.shape != entries.shape or not len(entries)
            or not np.issubdtype(entries.dtype, np.integer)
            or not np.issubdtype(horizons.dtype, np.integer)
            or np.any(entries < 0) or np.any(horizons < 1)
            or np.any(horizons > DYNAMIC_M1_PREFIX_HORIZON_MINUTES[-1])
            or type(query_chunk_rows) is not int or query_chunk_rows < 1
            or endpoint_kind not in DYNAMIC_M1_PREFIX_ENDPOINT_KINDS):
        raise RuntimeError("DYNAMIC_M1_PREFIX_QUERY_INVALID")
    if (not isinstance(fill_surface, pd.DataFrame)
            or tuple(fill_surface.columns) != _SURFACE_COLUMNS
            or np.any(entries >= len(fill_surface))):
        raise RuntimeError("DYNAMIC_M1_PREFIX_FILL_SURFACE_INVALID")
    entries = entries.astype(np.int64, copy=False)
    horizons = horizons.astype(np.int64, copy=False)
    unique, inverse = np.unique(entries, return_inverse=True)
    selected_clock = fill_surface.loc[:, ["time", "entry_decision_at"]].iloc[unique]
    m5_times = _utc_index(selected_clock["time"], name="DYNAMIC_FILL_M5", seconds=ENTRY_DECISION_BAR_SECONDS)
    decision = _utc_index(selected_clock["entry_decision_at"], name="DYNAMIC_FILL_DECISION", seconds=EXIT_DECISION_BAR_SECONDS)
    if not decision.equals(m5_times + pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS)):
        raise RuntimeError("DYNAMIC_M1_PREFIX_FILL_CLOCK_INVALID")
    # Validate/read fill prices only for decisions before this split's cutoff.
    before_cutoff = decision.asi8 < index.split_end_exclusive.value
    local = np.full(len(unique), -1, dtype=np.int64)
    bound = np.zeros(len(unique), dtype=bool)
    if before_cutoff.any():
        checked = _require_surface(fill_surface.iloc[unique[before_cutoff]])
        pos = checked["entry_m1_row"].to_numpy(np.int64) - index.source_row_offset
        declared = checked["entry_fill_bound"].to_numpy(bool)
        safe = np.clip(pos, 0, len(index.source.times) - 1)
        exact = (declared & (pos >= 0) & (pos < len(index.source.times))
                 & (index.source.times.asi8[safe] == decision.asi8[before_cutoff]))
        if np.any(declared & ~exact):
            raise RuntimeError("DYNAMIC_M1_PREFIX_FILL_SOURCE_ROW_MISMATCH")
        if (np.any(exact & (checked["entry_bid"].to_numpy(np.float64)
                            != index.source.values["bid_open"][safe]))
                or np.any(exact & (checked["entry_ask"].to_numpy(np.float64)
                                  != index.source.values["ask_open"][safe]))):
            raise RuntimeError("DYNAMIC_M1_PREFIX_FILL_SOURCE_QUOTE_MISMATCH")
        local[before_cutoff] = np.where(exact, pos, -1)
        bound[before_cutoff] = exact
    q = len(entries)
    decision_ns = decision.asi8[inverse]
    terminal_ns = decision_ns + horizons * 60 * 10**9
    result: dict[str, Any] = {
        "schema_version": DYNAMIC_M1_PREFIX_SCHEMA_VERSION,
        "clock_kind": DYNAMIC_M1_PREFIX_CLOCK_KIND,
        "endpoint_kind": endpoint_kind,
        "side_order": DYNAMIC_M1_PREFIX_SIDE_ORDER,
        "query_entry_rows": entries.copy(), "horizon_minutes": horizons.copy(),
        "terminal_time_ns": terminal_ns,
        "elapsed_wall_seconds": horizons * 60,
        "observed_m1_count": np.zeros(q, dtype=np.int64),
        "terminal_source_row": np.full(q, -1, dtype=np.int64),
        "terminal_censor_reason": np.full(q, "fill_unavailable", dtype="U64"),
        "path_censor_reason": np.full(q, "fill_unavailable", dtype="U64"),
        "terminal_gross_observed": np.zeros((q, 2), dtype=bool),
        "path_observed": np.zeros((q, 2), dtype=bool),
        "first_mfe_source_row": np.full((q, 2), -1, dtype=np.int64),
        "first_mfe_offset_observed_m1": np.full((q, 2), -1, dtype=np.int64),
        "first_mfe_at_terminal_open": np.zeros((q, 2), dtype=bool),
    }
    for name in ("terminal_gross_return_bps", "mfe_bps",
                 "mae_before_first_mfe_lower_bps", "mae_before_first_mfe_upper_bps"):
        result[name] = np.full((q, 2), np.nan, dtype=np.float64)
    clock = index.source.times.asi8
    quotes = index.source.values
    after = ~before_cutoff[inverse]
    result["terminal_censor_reason"][after] = "split_right_censor"
    result["path_censor_reason"][after] = "split_right_censor"
    for begin in range(0, q, query_chunk_rows):
        stop = min(q, begin + query_chunk_rows)
        batch = np.arange(begin, stop, dtype=np.int64)
        rows = batch[bound[inverse[batch]]]
        if not len(rows):
            continue
        starts = local[inverse[rows]]
        target = terminal_ns[rows]
        terminal_allowed = target < index.split_end_exclusive.value
        path_allowed = (target < index.split_end_exclusive.value
                        if endpoint_kind == DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT
                        else target <= index.split_end_exclusive.value)
        end = np.searchsorted(clock, target, side="left")
        exact = np.zeros(len(rows), dtype=bool)
        present = end < len(clock)
        exact[present] = clock[end[present]] == target[present]
        terminal_rows = rows[terminal_allowed & exact]
        if endpoint_kind == DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT:
            result["terminal_censor_reason"][rows] = "terminal_quote_missing"
            result["terminal_censor_reason"][rows[~terminal_allowed]] = "split_right_censor"
            if len(terminal_rows):
                s = starts[terminal_allowed & exact]
                e = end[terminal_allowed & exact]
                long, short = _entry_notional_pnl_bps(
                    entry_bid=quotes["bid_open"][s], entry_ask=quotes["ask_open"][s],
                    exit_bid=quotes["bid_open"][e], exit_ask=quotes["ask_open"][e])
                result["terminal_gross_return_bps"][terminal_rows] = np.column_stack((long, short))
                result["terminal_gross_observed"][terminal_rows] = True
                result["terminal_source_row"][terminal_rows] = e + index.source_row_offset
                result["terminal_censor_reason"][terminal_rows] = "observed"
        else:
            result["terminal_censor_reason"][rows] = "closed_m1_prefix_only"
            result["terminal_censor_reason"][rows[~path_allowed]] = "split_right_censor"
        # Count unknown gap starts strictly before T. A gap starting at the
        # last complete bar's close T does not erase that closed-only prefix.
        gap_stop = np.searchsorted(clock, target - 60 * 10**9, side="left")
        unknown = (index.unknown_gap_prefix[gap_stop]
                   - index.unknown_gap_prefix[starts]) > 0
        count = np.maximum(end - starts, 0)
        result["observed_m1_count"][rows[path_allowed]] = count[path_allowed]
        path_ok = path_allowed & ~unknown & (count > 0)
        result["path_censor_reason"][rows] = "prefix_close_missing"
        if endpoint_kind == DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT:
            path_ok &= exact
            result["path_censor_reason"][rows[~exact]] = "terminal_quote_missing"
        else:
            last = np.clip(end - 1, 0, len(clock) - 1)
            closes_at_target = clock[last] + 60 * 10**9 == target
            path_ok &= closes_at_target
        result["path_censor_reason"][rows[unknown]] = "unknown_source_absence"
        beyond = target > clock[-1] + 60 * 10**9
        result["path_censor_reason"][rows[beyond & ~unknown]] = "source_window_unavailable"
        result["path_censor_reason"][rows[~path_allowed]] = "split_right_censor"
        good = rows[path_ok]
        if not len(good):
            continue
        starts, ends = starts[path_ok], end[path_ok]
        result["path_observed"][good] = True
        result["path_censor_reason"][good] = "observed"
        for side, favorable_name, adverse_name, maximum in (
            (0, "bid_high", "bid_low", True), (1, "ask_low", "ask_high", False)
        ):
            peaks = index._ranges[favorable_name].query(starts, ends)
            terminal_peak = np.zeros(len(good), dtype=bool)
            favorable = quotes[favorable_name][peaks].copy()
            if endpoint_kind == DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT:
                opening = quotes["bid_open" if side == 0 else "ask_open"][ends]
                terminal_peak = opening > favorable if maximum else opening < favorable
                peaks = np.where(terminal_peak, ends, peaks)
                favorable = np.where(terminal_peak, opening, favorable)
            previous = index._ranges[adverse_name].query(starts, peaks)
            entry = quotes["ask_open" if side == 0 else "bid_open"][starts]
            running = quotes["bid_open" if side == 0 else "ask_open"][starts].copy()
            has_previous = previous >= 0
            if side == 0:
                running[has_previous] = np.minimum(running[has_previous], quotes[adverse_name][previous[has_previous]])
                lower_price = np.minimum(running, quotes["bid_open"][peaks])
            else:
                running[has_previous] = np.maximum(running[has_previous], quotes[adverse_name][previous[has_previous]])
                lower_price = np.maximum(running, quotes["ask_open"][peaks])
            upper_price = lower_price.copy()
            interior = ~terminal_peak
            # Never read terminal-bar high/low: at an instantaneous terminal
            # opening peak, both pre-peak bounds equal the already-known prefix.
            if side == 0:
                upper_price[interior] = np.minimum(running[interior], quotes[adverse_name][peaks[interior]])
                mfe = (favorable - entry) / entry * 1e4
                lower, upper = (entry - lower_price) / entry * 1e4, (entry - upper_price) / entry * 1e4
            else:
                upper_price[interior] = np.maximum(running[interior], quotes[adverse_name][peaks[interior]])
                mfe = (entry - favorable) / entry * 1e4
                lower, upper = (lower_price - entry) / entry * 1e4, (upper_price - entry) / entry * 1e4
            if (not np.isfinite(mfe).all() or not np.isfinite(lower).all()
                    or not np.isfinite(upper).all() or np.any(lower < 0) or np.any(upper < lower)):
                raise RuntimeError("DYNAMIC_M1_PREFIX_PATH_VALUES_INVALID")
            result["mfe_bps"][good, side] = mfe
            result["mae_before_first_mfe_lower_bps"][good, side] = lower
            result["mae_before_first_mfe_upper_bps"][good, side] = upper
            result["first_mfe_source_row"][good, side] = peaks + index.source_row_offset
            result["first_mfe_offset_observed_m1"][good, side] = peaks - starts
            result["first_mfe_at_terminal_open"][good, side] = terminal_peak
    return result
