"""Observed-market Entry targets on trading-day horizons (operator decision 11.10.2026).

Explicitly bound successor to ``entry_observed_market_v1`` for the native D1
wave: the same observed, executable, teacher-free cash outcome, measured at
1, 5 and 20 trading days instead of a fixed count of M5 bars. The decision
authority is the 20-day horizon's unique argmax over LONG/SHORT/FLAT values in
bps; the shorter horizons are retained observation targets, never holding
limits.

Trading days are sessions on the one UTC session clock (boundary 22:00 UTC).
A horizon of ``h`` trading days ends in the session that is ``h`` observed
sessions after the entry session; sessions without any M1 bar (weekends,
holidays, source absence) are skipped, never counted. The exit is the first
closed M1 bar in that session at or after the entry's offset into its own
session; a row whose target session or offset has no such bar is censored
and excluded by the label owner, never filled.

Costs reuse the v1 economics owner: the round trip is charged once and
financing accrues over the actual wall-clock seconds between the entry
decision and the exit bar. Nothing here queries a model, clips an outcome or
grants serving authority.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    CausalM1QuoteSource, _entry_notional_pnl_bps, _m1_arrays, _require_surface, causal_m1_target_contract,
)
from gx1.contracts.entry_fitted_q_v1 import ENTRY_FITTED_Q_ACTION_ORDER
from gx1.contracts.entry_observed_market_v1 import (
    SCHEMA_VERSION as PREDECESSOR_SCHEMA_VERSION, build_entry_observed_market_targets,
)
from gx1.time.session_detector import (
    TRADING_SESSION_BOUNDARY_OFFSET, TRADING_SESSION_CLOCK_SCHEMA_VERSION, TRADING_SESSION_DURATION,
)

SCHEMA_VERSION = "gx1_entry_observed_market_horizons_v1"
OBSERVATION_HORIZONS_TRADING_DAYS = (1, 5, 20)
DECISION_HORIZON_TRADING_DAYS = 20
GROSS_TARGET_COLUMNS_BY_HORIZON = {
    h: (f"y_long_gross_at_{h}d_bps", f"y_short_gross_at_{h}d_bps") for h in OBSERVATION_HORIZONS_TRADING_DAYS
}
ELAPSED_COLUMN_BY_HORIZON = {h: f"elapsed_seconds_at_{h}d" for h in OBSERVATION_HORIZONS_TRADING_DAYS}
BATCH_TARGET_KEY_BY_HORIZON = {h: f"entry_observed_market_target_bps_h{h}d" for h in OBSERVATION_HORIZONS_TRADING_DAYS}
DECISION_BATCH_TARGET_KEY = BATCH_TARGET_KEY_BY_HORIZON[DECISION_HORIZON_TRADING_DAYS]


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def entry_observed_market_horizons_contract() -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "predecessor_schema_version": PREDECESSOR_SCHEMA_VERSION,
        "action_order": list(ENTRY_FITTED_Q_ACTION_ORDER),
        "quote_contract": causal_m1_target_contract(),
        "session_clock": TRADING_SESSION_CLOCK_SCHEMA_VERSION,
        "session_boundary_utc_offset_seconds": int(TRADING_SESSION_BOUNDARY_OFFSET.total_seconds()),
        "observation_horizons_trading_days": list(OBSERVATION_HORIZONS_TRADING_DAYS),
        "decision_horizon_trading_days": DECISION_HORIZON_TRADING_DAYS,
        "horizon_rule": "h observed sessions after the entry session; sessions without M1 bars are skipped, never counted",
        "exit_rule": "first closed M1 bar in the target session at or after the entry's offset into its own session; otherwise censored",
        "elapsed_source": "actual wall-clock seconds from the entry decision to the exit bar (financing)",
        "gross_target_columns": {str(h): list(c) for h, c in GROSS_TARGET_COLUMNS_BY_HORIZON.items()},
        "elapsed_columns": {str(h): c for h, c in ELAPSED_COLUMN_BY_HORIZON.items()},
        "batch_target_keys": {str(h): k for h, k in BATCH_TARGET_KEY_BY_HORIZON.items()},
        "decision_batch_target_key": DECISION_BATCH_TARGET_KEY,
        "decision_authority": "unique argmax over the decision horizon's LONG/SHORT/FLAT values in bps; exact ties fail closed",
        "target": "observed_executable_terminal_cash_return_net_of_declared_costs",
        "target_unit": "raw_bps",
        "flat_target_bps": 0.0,
        "exit_model_used": False,
        "future_outcomes_used_as_model_inputs": False,
        "hindsight_best_exit_used": False,
        "negative_outcomes_clipped": False,
        "censored_rows_filled": False,
        "holding_time_limit": False,
        "horizon_search_allowed": False,
        "score_is_calibrated_probability": False,
        "serving_authority_granted": False,
    }
    return {**value, "contract_sha256": _sha(value)}


def session_day_index(times: pd.DatetimeIndex) -> np.ndarray:
    """Integer index of the UTC session (boundary 22:00) that contains each time."""
    index = pd.DatetimeIndex(times)
    if index.tz is None:
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_CLOCK_NAIVE")
    shifted = index.tz_convert("UTC") - TRADING_SESSION_BOUNDARY_OFFSET
    return (shifted.as_unit("ns").asi8 // int(TRADING_SESSION_DURATION.value)).astype(np.int64)


def causal_m1_terminal_outcomes_at_trading_days(
    *, fill_surface: pd.DataFrame, closed_m1: pd.DataFrame | CausalM1QuoteSource, horizon_trading_days: int,
) -> pd.DataFrame:
    """Exact executable terminal PnL ``horizon_trading_days`` observed sessions after entry."""
    if type(horizon_trading_days) is not int or horizon_trading_days < 1:
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_HORIZON_INVALID")
    surface = _require_surface(fill_surface)
    m1_times, values = _m1_arrays(closed_m1)
    m1_session = session_day_index(m1_times)
    sessions, first_row = np.unique(m1_session, return_index=True)  # observed sessions and their first M1 row
    size = len(surface)
    start = surface["entry_m1_row"].to_numpy(dtype=np.int64)
    fill_bound = surface["entry_fill_bound"].to_numpy(dtype=np.bool_)
    decision_at = pd.DatetimeIndex(surface["entry_decision_at"])
    valid = fill_bound & (start >= 0) & (start < len(m1_times))
    positions = np.flatnonzero(valid)
    valid[positions] &= m1_times.asi8[start[positions]] == decision_at.asi8[positions]
    exit_row = np.full(size, -1, dtype=np.int64)
    positions = np.flatnonzero(valid)
    if len(positions):
        entry_session = session_day_index(decision_at[positions])
        entry_pos = np.searchsorted(sessions, entry_session)
        in_range = (entry_pos < len(sessions)) & (entry_pos + horizon_trading_days < len(sessions))
        in_range &= sessions[np.minimum(entry_pos, len(sessions) - 1)] == entry_session
        target_pos = np.minimum(entry_pos + horizon_trading_days, len(sessions) - 1)
        session_start_ns = sessions * int(TRADING_SESSION_DURATION.value) + int(TRADING_SESSION_BOUNDARY_OFFSET.value)
        offset_ns = decision_at.asi8[positions] - session_start_ns[np.minimum(entry_pos, len(sessions) - 1)]
        candidate_ns = session_start_ns[target_pos] + offset_ns
        exit_candidate = np.searchsorted(m1_times.asi8, candidate_ns, side="left")
        ok = in_range & (exit_candidate < len(m1_times))
        same_session = np.zeros(len(positions), dtype=bool)
        same_session[ok] = m1_session[exit_candidate[ok]] == sessions[target_pos[ok]]
        ok &= same_session
        exit_row[positions[ok]] = exit_candidate[ok]
        valid[positions[~ok]] = False
    exit_bid = np.full(size, np.nan)
    exit_ask = np.full(size, np.nan)
    exit_bid[valid] = values["bid_open"][exit_row[valid]]
    exit_ask[valid] = values["ask_open"][exit_row[valid]]
    entry_bid = surface["entry_bid"].to_numpy(dtype=np.float64)
    entry_ask = surface["entry_ask"].to_numpy(dtype=np.float64)
    long_pnl = np.full(size, np.nan)
    short_pnl = np.full(size, np.nan)
    if valid.any():
        long_pnl[valid], short_pnl[valid] = _entry_notional_pnl_bps(
            entry_bid=entry_bid[valid], entry_ask=entry_ask[valid], exit_bid=exit_bid[valid], exit_ask=exit_ask[valid])
        if not np.isfinite(long_pnl[valid]).all() or not np.isfinite(short_pnl[valid]).all():
            raise RuntimeError("ENTRY_OBSERVED_HORIZONS_OUTCOME_VALUES_INVALID")
    exit_time = pd.Series(pd.NaT, index=np.arange(size), dtype="datetime64[ns, UTC]")
    exit_time.iloc[valid] = m1_times[exit_row[valid]]
    elapsed = np.full(size, -1, dtype=np.int64)
    elapsed[valid] = (m1_times.asi8[exit_row[valid]] - decision_at.asi8[valid]) // 10**9
    if valid.any() and not bool((elapsed[valid] > 0).all()):
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_ELAPSED_INVALID")
    long_col, short_col = GROSS_TARGET_COLUMNS_BY_HORIZON.get(horizon_trading_days, (f"y_long_gross_at_{horizon_trading_days}d_bps", f"y_short_gross_at_{horizon_trading_days}d_bps"))
    return pd.DataFrame({
        "time": surface["time"].to_numpy(),
        "entry_decision_at": surface["entry_decision_at"].to_numpy(),
        "exit_decision_at": exit_time.to_numpy(),
        "horizon_trading_days": np.full(size, horizon_trading_days, dtype=np.int32),
        ELAPSED_COLUMN_BY_HORIZON.get(horizon_trading_days, f"elapsed_seconds_at_{horizon_trading_days}d"): elapsed,
        "entry_bid": entry_bid, "entry_ask": entry_ask, "exit_bid": exit_bid, "exit_ask": exit_ask,
        long_col: long_pnl, short_col: short_pnl, "outcome_valid": valid,
    })


@torch.no_grad()
def build_entry_observed_market_horizon_targets(
    *, gross_by_horizon: Mapping[int, torch.Tensor], elapsed_by_horizon: Mapping[int, torch.Tensor],
    economics: Mapping[str, Any],
) -> dict[int, dict[str, torch.Tensor]]:
    """Net LONG/SHORT/FLAT=0 per horizon through the predecessor's cost arithmetic; all horizons required."""
    if set(gross_by_horizon) != set(OBSERVATION_HORIZONS_TRADING_DAYS) or set(elapsed_by_horizon) != set(OBSERVATION_HORIZONS_TRADING_DAYS):
        raise RuntimeError("ENTRY_OBSERVED_HORIZONS_TARGET_INPUT_INVALID")
    rows = None
    out = {}
    for h in OBSERVATION_HORIZONS_TRADING_DAYS:
        built = build_entry_observed_market_targets(
            gross_return_bps=gross_by_horizon[h], elapsed_wall_clock_seconds=elapsed_by_horizon[h], economics=economics)
        if rows is None:
            rows = built["target_bps"].shape[0]
        elif built["target_bps"].shape[0] != rows:
            raise RuntimeError("ENTRY_OBSERVED_HORIZONS_TARGET_ROWS_DIFFER")
        out[h] = built
    return out


def require_entry_observed_market_horizon_batch(
    batch: Mapping[str, Any], *, device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Exact per-horizon batch keys (rule 17): every horizon present, finite, FLAT exactly zero."""
    out = {}
    rows = None
    for h, key in BATCH_TARGET_KEY_BY_HORIZON.items():
        targets = batch.get(key)
        if (
            not isinstance(targets, torch.Tensor) or targets.dtype != torch.float32 or targets.ndim != 2
            or targets.shape[1] != 3 or targets.shape[0] < 1 or not bool(torch.isfinite(targets).all())
            or not bool((targets[:, 2] == 0.0).all()) or (rows is not None and targets.shape[0] != rows)
        ):
            raise RuntimeError(f"ENTRY_OBSERVED_HORIZONS_BATCH_TARGET_INVALID:{key}")
        rows = targets.shape[0]
        targets = targets.detach().to(device)
        out[h] = (targets, torch.ones_like(targets, dtype=torch.bool))
    return out
