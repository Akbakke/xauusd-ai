"""Research-only trade-path statistics for an exact unified Exit replay.

This module is deliberately downstream of the canonical full-TEST Entry/Exit
replay producer.  It neither generates a trade, chooses a side, nor supplies
an execution cost.  Instead it turns that producer's immutable M1 bid/ask
traces into audit-friendly path metrics.  In particular, it keeps the
distinction between observed quote-delta facts and unbound net economics:
commission, realised slippage, financing and a broker account ledger remain
outside this contract and therefore cannot be claimed as net PnL.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    recompute_joint_exit_replay_coverage,
    unified_replay_net_cost_policy_metadata,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SHORT_INDEX,
)
from gx1.replay.source_tape_v1 import SourceTape


TRADE_PATH_METRICS_SCHEMA_VERSION = "gx1_unified_exit_trade_path_metrics_v1"
TRADE_PATH_METRICS_DECISION = "RESEARCH_ONLY_BLOCKED_NET_ECONOMICS"
_REQUIRED_REPLAY_COLUMNS = frozenset(
    {
        "reference_row_id",
        "time",
        "model_direction_index",
        "authorized_order",
        "entry_bid",
        "entry_ask",
        "entry_fill_time",
        "model_exit_fill_bid",
        "model_exit_fill_ask",
        "model_exit_fill_time",
        "account_equity",
        "units",
        "session",
        "vol_regime",
    }
)
_REQUIRED_TRACE_COLUMNS = frozenset(
    {
        "reference_row_id",
        "closed_bar_time",
        "step",
        "closed_m1_source_path",
        "closed_m1_source_sha256",
    }
)


class TradePathMetricsError(RuntimeError):
    """Replay rows cannot establish one exact research path report."""


def _fail(detail: str) -> None:
    raise TradePathMetricsError(f"[UNIFIED_EXIT_TRADE_PATH_METRICS_INVALID] {detail}")


def _utc_series(frame: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if values.isna().any():
        _fail(f"{context}.{column} must contain finite UTC timestamps")
    return values


def _finite_column(frame: pd.DataFrame, column: str, *, context: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        _fail(f"{context}.{column} must be finite numeric")
    return values


def _strict_boolean(frame: pd.DataFrame, column: str, *, context: str) -> np.ndarray:
    values = frame[column].tolist()
    if not all(isinstance(value, (bool, np.bool_)) for value in values):
        _fail(f"{context}.{column} must contain exact booleans")
    return np.asarray(values, dtype=bool)


def _safe_quantile(values: Iterable[float], quantile: float) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return None
    if not np.isfinite(array).all():
        _fail("quantile received a non-finite numeric value")
    return float(np.quantile(array, quantile))


def _mfe_mae_summary(trades: pd.DataFrame) -> dict[str, Any]:
    count = int(len(trades))
    if count == 0:
        return {
            "trade_rows": 0,
            "win_rate": None,
            "mean_executable_quote_delta_bps": None,
            "total_executable_quote_delta_usd": None,
            "mean_mfe_bps": None,
            "mean_mae_bps": None,
            "mfe_to_mae_ratio": None,
            "p05_executable_quote_delta_bps": None,
            "mae_before_mfe_rows": 0,
            "mfe_before_mae_rows": 0,
            "same_m1_bar_order_unknown_rows": 0,
            "mae_before_mfe_rate_determined": None,
            "mean_holding_minutes": None,
        }
    quote_bps = trades["executable_quote_delta_bps"].to_numpy(dtype=np.float64)
    quote_usd = trades["executable_quote_delta_usd"].to_numpy(dtype=np.float64)
    mfe = trades["intrabar_mfe_bps"].to_numpy(dtype=np.float64)
    mae = trades["intrabar_mae_bps"].to_numpy(dtype=np.float64)
    holding = trades["holding_minutes"].to_numpy(dtype=np.float64)
    for label, values in (
        ("quote_bps", quote_bps),
        ("quote_usd", quote_usd),
        ("mfe", mfe),
        ("mae", mae),
        ("holding", holding),
    ):
        if not np.isfinite(values).all():
            _fail(f"trade summary {label} is non-finite")
    if np.any(mae < 0.0) or np.any(holding <= 0.0):
        _fail("trade summary has invalid adverse excursion or holding time")
    mae_before = trades["mae_before_mfe"].to_numpy(dtype=bool)
    mfe_before = trades["mfe_before_mae"].to_numpy(dtype=bool)
    unknown = trades["same_m1_bar_order_unknown"].to_numpy(dtype=bool)
    if np.any((mae_before & mfe_before) | ((mae_before | mfe_before) & unknown)):
        _fail("MAE/MFE order flags are mutually inconsistent")
    determined = int(np.count_nonzero(mae_before | mfe_before))
    mean_mae = float(np.mean(mae))
    return {
        "trade_rows": count,
        "win_rate": float(np.mean(quote_bps > 0.0)),
        "mean_executable_quote_delta_bps": float(np.mean(quote_bps)),
        "total_executable_quote_delta_usd": float(np.sum(quote_usd)),
        "mean_mfe_bps": float(np.mean(mfe)),
        "mean_mae_bps": mean_mae,
        "mfe_to_mae_ratio": (
            None if mean_mae == 0.0 else float(np.mean(mfe) / mean_mae)
        ),
        "p05_executable_quote_delta_bps": _safe_quantile(quote_bps, 0.05),
        "mae_before_mfe_rows": int(np.count_nonzero(mae_before)),
        "mfe_before_mae_rows": int(np.count_nonzero(mfe_before)),
        "same_m1_bar_order_unknown_rows": int(np.count_nonzero(unknown)),
        "mae_before_mfe_rate_determined": (
            None
            if determined == 0
            else float(np.count_nonzero(mae_before) / determined)
        ),
        "mean_holding_minutes": float(np.mean(holding)),
    }


def _source_tapes_for_authorized_traces(
    trace_rows: pd.DataFrame,
) -> dict[tuple[str, str], SourceTape]:
    tapes: dict[tuple[str, str], SourceTape] = {}
    for (raw_path, raw_sha), scoped in trace_rows.groupby(
        ["closed_m1_source_path", "closed_m1_source_sha256"], sort=True
    ):
        path = str(raw_path)
        digest = str(raw_sha).strip().lower()
        if not path or not digest or len(digest) != 64:
            _fail("trace source binding is empty or malformed")
        tape = SourceTape.load(Path(path))
        if tape.source_sha256 != digest:
            _fail(f"trace source hash differs from bound source: {path}")
        if not len(scoped):
            _fail("trace source grouping unexpectedly empty")
        tapes[(str(tape.source_path), digest)] = tape
    return tapes


def _trace_intrabar_excursion(
    *,
    trace: pd.DataFrame,
    replay_row: pd.Series,
    tape: SourceTape,
) -> dict[str, Any]:
    ordered = trace.sort_values("step", kind="mergesort").reset_index(drop=True)
    close_times = _utc_series(ordered, "closed_bar_time", context="trace")
    positions = tape.indices_for_times(close_times)
    direction = int(replay_row["model_direction_index"])
    entry_bid = float(replay_row["entry_bid"])
    entry_ask = float(replay_row["entry_ask"])
    if direction == MODEL_DIRECTION_LONG_INDEX:
        favorable = np.maximum(
            0.0, (tape.bid_high[positions] / entry_ask - 1.0) * 10_000.0
        )
        adverse = np.maximum(
            0.0, (1.0 - tape.bid_low[positions] / entry_ask) * 10_000.0
        )
    elif direction == MODEL_DIRECTION_SHORT_INDEX:
        favorable = np.maximum(
            0.0, (1.0 - tape.ask_low[positions] / entry_bid) * 10_000.0
        )
        adverse = np.maximum(
            0.0, (tape.ask_high[positions] / entry_bid - 1.0) * 10_000.0
        )
    else:
        _fail("intrabar path requested for a FLAT replay row")
    if (
        not np.isfinite(favorable).all()
        or not np.isfinite(adverse).all()
        or np.any(adverse < 0.0)
    ):
        _fail("intrabar MFE/MAE is invalid")
    mfe_index = int(np.argmax(favorable))
    mae_index = int(np.argmax(adverse))
    mfe_time = close_times.iloc[mfe_index]
    mae_time = close_times.iloc[mae_index]
    same_bar = bool(mfe_time == mae_time)
    return {
        "intrabar_mfe_bps": float(favorable[mfe_index]),
        "intrabar_mae_bps": float(adverse[mae_index]),
        "mfe_time": mfe_time,
        "mae_time": mae_time,
        # An OHLC candle has no tick-order proof inside one minute.  Preserve
        # that uncertainty rather than inventing MAE-before-MFE chronology.
        "mae_before_mfe": bool(not same_bar and mae_time < mfe_time),
        "mfe_before_mae": bool(not same_bar and mfe_time < mae_time),
        "same_m1_bar_order_unknown": same_bar,
    }


def derive_unified_exit_trade_path_metrics(
    *,
    replay_rows: pd.DataFrame,
    exit_trace_rows: pd.DataFrame,
    candidate_bundle_sha256: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Derive immutable-path research metrics from an exact unified replay.

    The returned report is deliberately blocked for net-PnL, candidate
    admission and production use.  ``serial_one_position_ledger`` is a
    deterministic diagnostic that admits only one completed position at a
    time; it makes overlap handling and drawdown assumptions visible instead
    of treating a dense per-bar signal table as an equity curve.
    """

    missing_replay = sorted(_REQUIRED_REPLAY_COLUMNS - set(replay_rows.columns))
    missing_trace = sorted(_REQUIRED_TRACE_COLUMNS - set(exit_trace_rows.columns))
    if missing_replay or missing_trace:
        _fail(
            f"required columns missing: replay={missing_replay} trace={missing_trace}"
        )
    coverage = recompute_joint_exit_replay_coverage(
        replay_rows,
        exit_trace_rows=exit_trace_rows,
        candidate_bundle_sha256=candidate_bundle_sha256,
        context="TRADE_PATH_METRICS_REPLAY",
    )
    replay = replay_rows.copy().reset_index(drop=True)
    replay["time"] = _utc_series(replay, "time", context="replay")
    replay["entry_fill_time"] = _utc_series(
        replay, "entry_fill_time", context="replay"
    )
    replay["model_exit_fill_time"] = pd.to_datetime(
        replay["model_exit_fill_time"], utc=True, errors="coerce"
    )
    authorized = _strict_boolean(replay, "authorized_order", context="replay")
    directions = pd.to_numeric(
        replay["model_direction_index"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(directions).all() or not np.array_equal(
        directions, directions.astype(np.int64)
    ):
        _fail("replay directions are not exact integer values")
    directions = directions.astype(np.int64)
    trade_mask = authorized & np.isin(
        directions, [MODEL_DIRECTION_LONG_INDEX, MODEL_DIRECTION_SHORT_INDEX]
    )
    if not int(np.count_nonzero(trade_mask)):
        _fail("no authorized non-FLAT trade rows")
    if replay.loc[trade_mask, "model_exit_fill_time"].isna().any():
        _fail("authorized trade rows lack model Exit fill times")
    if replay.loc[trade_mask, "reference_row_id"].astype(str).duplicated().any():
        _fail("authorized replay reference ids are not unique")

    traces = exit_trace_rows.copy()
    traces["closed_bar_time"] = _utc_series(
        traces, "closed_bar_time", context="trace"
    )
    trace_by_id = {
        str(reference_id): group.copy()
        for reference_id, group in traces.groupby("reference_row_id", sort=False)
    }
    authorized_replay = replay.loc[trade_mask].copy()
    authorized_ids = set(authorized_replay["reference_row_id"].astype(str))
    missing_traces = sorted(authorized_ids - set(trace_by_id))
    if missing_traces:
        _fail(f"authorized rows missing unified Exit traces: count={len(missing_traces)}")
    authorized_traces = traces.loc[
        traces["reference_row_id"].astype(str).isin(authorized_ids)
    ].copy()
    raw_tapes = _source_tapes_for_authorized_traces(authorized_traces)

    trade_records: list[dict[str, Any]] = []
    for _, row in authorized_replay.sort_values("time", kind="mergesort").iterrows():
        reference_id = str(row["reference_row_id"])
        trace = trace_by_id[reference_id]
        raw_paths = set(trace["closed_m1_source_path"].astype(str))
        raw_shas = {str(value).strip().lower() for value in trace["closed_m1_source_sha256"]}
        if len(raw_paths) != 1 or len(raw_shas) != 1:
            _fail(f"{reference_id} trace source binding is not singular")
        raw_path = next(iter(raw_paths))
        raw_sha = next(iter(raw_shas))
        expected_path = str(Path(raw_path).expanduser().resolve(strict=True))
        matching = [
            tape
            for (path, digest), tape in raw_tapes.items()
            if path == expected_path and digest == raw_sha
        ]
        if len(matching) != 1:
            _fail(f"{reference_id} trace source tape is unavailable")
        tape = matching[0]
        entry_fill = pd.Timestamp(row["entry_fill_time"])
        exit_fill = pd.Timestamp(row["model_exit_fill_time"])
        if exit_fill <= entry_fill:
            _fail(f"{reference_id} Exit fill does not follow Entry fill")
        direction = int(row["model_direction_index"])
        entry_bid = float(row["entry_bid"])
        entry_ask = float(row["entry_ask"])
        exit_bid = float(row["model_exit_fill_bid"])
        exit_ask = float(row["model_exit_fill_ask"])
        units = float(row["units"])
        if not np.isfinite([entry_bid, entry_ask, exit_bid, exit_ask, units]).all() or units <= 0.0:
            _fail(f"{reference_id} quote/units are invalid")
        if direction == MODEL_DIRECTION_LONG_INDEX:
            quote_delta = exit_bid - entry_ask
            reference_price = entry_ask
            side = "LONG"
        else:
            quote_delta = entry_bid - exit_ask
            reference_price = entry_bid
            side = "SHORT"
        excursion = _trace_intrabar_excursion(
            trace=trace,
            replay_row=row,
            tape=tape,
        )
        trade_records.append(
            {
                "reference_row_id": reference_id,
                "decision_time": pd.Timestamp(row["time"]),
                "entry_fill_time": entry_fill,
                "exit_fill_time": exit_fill,
                "side": side,
                "session": str(row["session"]),
                "vol_regime": str(row["vol_regime"]),
                "units": units,
                "executable_quote_delta_bps": float(quote_delta / reference_price * 10_000.0),
                "executable_quote_delta_usd": float(quote_delta * units),
                "holding_minutes": float((exit_fill - entry_fill).total_seconds() / 60.0),
                "source_tape_sha256": tape.source_sha256,
                **excursion,
            }
        )
    trades = pd.DataFrame(trade_records).sort_values(
        ["entry_fill_time", "decision_time", "reference_row_id"], kind="mergesort"
    ).reset_index(drop=True)
    if trades.empty:
        _fail("authorized replay rows produced no trade metrics")

    serial_rows: list[int] = []
    next_free_time: pd.Timestamp | None = None
    for index, row in trades.iterrows():
        entry_fill = pd.Timestamp(row["entry_fill_time"])
        if next_free_time is None or entry_fill >= next_free_time:
            serial_rows.append(int(index))
            next_free_time = pd.Timestamp(row["exit_fill_time"])
    serial = trades.iloc[serial_rows].copy().reset_index(drop=True)
    base_equity_values = _finite_column(
        authorized_replay.sort_values("time", kind="mergesort"),
        "account_equity",
        context="replay",
    )
    if np.any(base_equity_values <= 0.0):
        _fail("reference account equity must be finite-positive")
    base_equity = float(base_equity_values[0])
    serial["equity_after_quote_delta_usd"] = base_equity + serial[
        "executable_quote_delta_usd"
    ].cumsum()
    running_peak = np.maximum.accumulate(
        np.concatenate(
            (
                [base_equity],
                serial["equity_after_quote_delta_usd"].to_numpy(dtype=np.float64),
            )
        )
    )
    serial_drawdown = (
        serial["equity_after_quote_delta_usd"].to_numpy(dtype=np.float64)
        - running_peak[1:]
    )
    serial["drawdown_from_peak_usd"] = serial_drawdown
    max_drawdown_usd = float(max(0.0, -float(np.min(serial_drawdown))))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for column in ("side", "session", "vol_regime"):
        grouped[column] = [
            {column: str(key), **_mfe_mae_summary(group)}
            for key, group in trades.groupby(column, sort=True)
        ]
    cost_policy = unified_replay_net_cost_policy_metadata()
    report = {
        "schema_version": TRADE_PATH_METRICS_SCHEMA_VERSION,
        "decision": TRADE_PATH_METRICS_DECISION,
        "failures": [
            "commission_slippage_financing_and_broker_ledger_unbound",
            "serial_portfolio_ledger_is_research_diagnostic_only",
        ],
        "coverage": coverage,
        "cost_policy": cost_policy,
        "trade_path_definition": {
            "mfe_mae": "exact_bid_ask_intrabar_extremes_over_model_held_closed_M1_bars",
            "mae_before_mfe": "strict_first_extreme_M1_timestamp_only; same_bar_order_unknown",
            "turnover": "two_one_way_events_per_completed_authorized_trade",
            "serial_ledger": (
                "chronological_one_position_no_overlap_unitized_quote_delta_"
                "diagnostic"
            ),
        },
        "authorized_independent_trade_metrics": {
            **_mfe_mae_summary(trades),
            "one_way_turnover_events": int(2 * len(trades)),
            "completed_round_trips": int(len(trades)),
        },
        "serial_one_position_ledger": {
            "selected_trade_rows": int(len(serial)),
            "skipped_overlapping_authorized_rows": int(len(trades) - len(serial)),
            "reference_start_equity_usd": base_equity,
            "final_quote_delta_equity_usd": float(serial["equity_after_quote_delta_usd"].iloc[-1]),
            "max_drawdown_quote_delta_usd": max_drawdown_usd,
            "max_drawdown_quote_delta_bps_of_reference_equity": float(
                max_drawdown_usd / base_equity * 10_000.0
            ),
        },
        "regime_breakdown": grouped,
        "production_authority_ready": False,
        "edge_claim_allowed": False,
    }
    return report, trades


__all__ = (
    "TRADE_PATH_METRICS_DECISION",
    "TRADE_PATH_METRICS_SCHEMA_VERSION",
    "TradePathMetricsError",
    "derive_unified_exit_trade_path_metrics",
)
