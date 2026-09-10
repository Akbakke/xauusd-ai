"""Canonical all-time Exit summaries shared by offline and live state."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


LIFETIME_SUMMARY_SCHEMA_VERSION = "gx1_unified_exit_lifetime_summary_v1"
LIFETIME_SUMMARY_FIELD_ORDER = (
    "bars_in_trade_log1p",
    "elapsed_wall_clock_seconds_log1p",
    "cum_mfe_bps_log1p",
    "cum_mae_bps_signed_log1p",
    "drawdown_from_mfe_bps_log1p",
    "bars_since_mfe_peak_log1p",
    "current_executable_pnl_bps_signed_log1p",
)
LIFETIME_SUMMARY_DIM = len(LIFETIME_SUMMARY_FIELD_ORDER)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def lifetime_summary_registry() -> dict[str, Any]:
    value = {
        "schema_version": LIFETIME_SUMMARY_SCHEMA_VERSION,
        "field_order": list(LIFETIME_SUMMARY_FIELD_ORDER),
        "field_order_sha256": canonical_sha256(list(LIFETIME_SUMMARY_FIELD_ORDER)),
        "dimension": LIFETIME_SUMMARY_DIM,
        "side_order": ["long", "short"],
        "pnl_semantics": {
            "long": "entry_ask_to_current_bid",
            "short": "entry_bid_to_current_ask",
            "mfe_mae": "intrabar_executable_side_relative_to_entry",
        },
        "bars_since_peak_semantics": (
            "TradeState strict-new-positive-cumulative-mfe reset; otherwise increment"
        ),
        "elapsed_semantics": "entry_fill_time_to_current_bar_close_wall_clock",
        "transforms": {
            "nonnegative": "log1p(x)",
            "signed": "sign(x)*log1p(abs(x))",
        },
    }
    value["registry_sha256"] = canonical_sha256(value)
    return value


def build_lifetime_summary(
    *,
    side: str,
    bars_in_trade: int,
    elapsed_wall_clock_seconds: int,
    current_executable_pnl_bps: float,
    cum_mfe_bps: float,
    cum_mae_bps: float,
    bars_since_mfe_peak: int,
) -> dict[str, Any]:
    if (
        side not in {"long", "short"}
        or isinstance(bars_in_trade, bool)
        or not isinstance(bars_in_trade, int)
        or bars_in_trade < 1
        or isinstance(elapsed_wall_clock_seconds, bool)
        or not isinstance(elapsed_wall_clock_seconds, int)
        or elapsed_wall_clock_seconds < 1
        or isinstance(bars_since_mfe_peak, bool)
        or not isinstance(bars_since_mfe_peak, int)
        or not 0 <= bars_since_mfe_peak <= bars_in_trade
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFETIME_SUMMARY_IDENTITY_INVALID")
    numeric = tuple(
        float(value)
        for value in (current_executable_pnl_bps, cum_mfe_bps, cum_mae_bps)
    )
    if (
        not all(math.isfinite(value) for value in numeric)
        or cum_mfe_bps < 0.0
        or cum_mae_bps > 0.0
        or cum_mfe_bps + 1e-9 < current_executable_pnl_bps
        or cum_mae_bps - 1e-9 > current_executable_pnl_bps
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFETIME_SUMMARY_VALUE_INVALID")
    drawdown = max(0.0, cum_mfe_bps - current_executable_pnl_bps)
    raw = {
        "bars_in_trade": bars_in_trade,
        "elapsed_wall_clock_seconds": elapsed_wall_clock_seconds,
        "cum_mfe_bps": float(cum_mfe_bps),
        "cum_mae_bps": float(cum_mae_bps),
        "drawdown_from_mfe_bps": drawdown,
        "bars_since_mfe_peak": bars_since_mfe_peak,
        "current_executable_pnl_bps": float(current_executable_pnl_bps),
    }
    values = np.asarray(
        [
            math.log1p(bars_in_trade),
            math.log1p(elapsed_wall_clock_seconds),
            math.log1p(cum_mfe_bps),
            _signed_log1p(cum_mae_bps),
            math.log1p(drawdown),
            math.log1p(bars_since_mfe_peak),
            _signed_log1p(current_executable_pnl_bps),
        ],
        dtype="<f8",
    )
    values.setflags(write=False)
    value = {
        "schema_version": LIFETIME_SUMMARY_SCHEMA_VERSION,
        "registry_sha256": lifetime_summary_registry()["registry_sha256"],
        "side": side,
        "raw": raw,
        "values": values,
    }
    value["summary_sha256"] = lifetime_summary_sha256(value)
    return value


def lifetime_summary_sha256(value: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in ("schema_version", "registry_sha256", "side", "raw"):
        digest.update(
            json.dumps(
                value[key], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        )
        digest.update(b"\0")
    values = np.ascontiguousarray(value["values"], dtype="<f8")
    digest.update(values.tobytes())
    return digest.hexdigest()


def require_lifetime_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "registry_sha256",
        "side",
        "raw",
        "values",
        "summary_sha256",
    }:
        raise RuntimeError("UNIFIED_EXIT_LIFETIME_SUMMARY_INVALID")
    observed = dict(value)
    raw = observed["raw"]
    if not isinstance(raw, Mapping):
        raise RuntimeError("UNIFIED_EXIT_LIFETIME_SUMMARY_INVALID")
    rebuilt = build_lifetime_summary(
        side=observed["side"],
        bars_in_trade=raw.get("bars_in_trade"),
        elapsed_wall_clock_seconds=raw.get("elapsed_wall_clock_seconds"),
        current_executable_pnl_bps=raw.get("current_executable_pnl_bps"),
        cum_mfe_bps=raw.get("cum_mfe_bps"),
        cum_mae_bps=raw.get("cum_mae_bps"),
        bars_since_mfe_peak=raw.get("bars_since_mfe_peak"),
    )
    if (
        observed["schema_version"] != LIFETIME_SUMMARY_SCHEMA_VERSION
        or observed["registry_sha256"]
        != lifetime_summary_registry()["registry_sha256"]
        or not isinstance(observed["values"], np.ndarray)
        or observed["values"].dtype != np.dtype("<f8")
        or observed["values"].shape != (LIFETIME_SUMMARY_DIM,)
        or observed["values"].flags.writeable
        or not np.array_equal(observed["values"], rebuilt["values"])
        or observed["summary_sha256"] != rebuilt["summary_sha256"]
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFETIME_SUMMARY_INVALID")
    return observed


def lifetime_summary_from_path(
    *,
    side: str,
    entry_fill_time: Any,
    entry_bid: float,
    entry_ask: float,
    bar_times: Sequence[Any],
    bid_high: Sequence[float],
    bid_low: Sequence[float],
    bid_close: Sequence[float],
    ask_high: Sequence[float],
    ask_low: Sequence[float],
    ask_close: Sequence[float],
) -> dict[str, Any]:
    """Reference owner matching TradeState.update_bar exactly."""

    times = pd.DatetimeIndex(pd.to_datetime(bar_times, utc=True, errors="coerce"))
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (bid_high, bid_low, bid_close, ask_high, ask_low, ask_close)
    ]
    if (
        side not in {"long", "short"}
        or times.empty
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or any(values.shape != (len(times),) for values in arrays)
        or any(not np.isfinite(values).all() for values in arrays)
        or not math.isfinite(float(entry_bid))
        or not math.isfinite(float(entry_ask))
        or float(entry_bid) <= 0.0
        or float(entry_ask) <= float(entry_bid)
    ):
        raise RuntimeError("UNIFIED_EXIT_LIFETIME_PATH_INVALID")
    bh, bl, bc, ah, al, ac = arrays
    if side == "long":
        current = (bc[-1] - entry_ask) / entry_ask * 10_000.0
        peaks = (bh - entry_ask) / entry_ask * 10_000.0
        troughs = (bl - entry_ask) / entry_ask * 10_000.0
    else:
        current = (entry_bid - ac[-1]) / entry_bid * 10_000.0
        peaks = (entry_bid - al) / entry_bid * 10_000.0
        troughs = (entry_bid - ah) / entry_bid * 10_000.0
    mfe = max(0.0, float(np.max(peaks)))
    mae = min(0.0, float(np.min(troughs)))
    positive_peak_rows = np.flatnonzero(peaks > 0.0)
    if positive_peak_rows.size:
        peak_value = float(np.max(peaks))
        first_peak = int(np.flatnonzero(peaks == peak_value)[0])
        bars_since_peak = len(times) - 1 - first_peak
    else:
        bars_since_peak = len(times)
    fill = pd.Timestamp(entry_fill_time)
    if pd.isna(fill) or fill.tz is None or fill.utcoffset() != pd.Timedelta(0):
        raise RuntimeError("UNIFIED_EXIT_LIFETIME_PATH_INVALID")
    decision = times[-1] + pd.Timedelta(minutes=1)
    elapsed = int((decision - fill).total_seconds())
    return build_lifetime_summary(
        side=side,
        bars_in_trade=len(times),
        elapsed_wall_clock_seconds=elapsed,
        current_executable_pnl_bps=float(current),
        cum_mfe_bps=mfe,
        cum_mae_bps=mae,
        bars_since_mfe_peak=bars_since_peak,
    )


def lifetime_summary_from_trade_state(trade_state: Any) -> dict[str, Any]:
    """Project the persisted live TradeState fields through the same transform."""

    last = getattr(trade_state, "last_processed_m1_ts", None)
    entry = getattr(trade_state, "entry_ts", None)
    if last is None or entry is None:
        raise RuntimeError("UNIFIED_EXIT_LIFETIME_TRADE_STATE_EMPTY")
    elapsed = int(
        (pd.Timestamp(last) + pd.Timedelta(minutes=1) - pd.Timestamp(entry)).total_seconds()
    )
    return build_lifetime_summary(
        side=str(trade_state.side),
        bars_in_trade=int(trade_state.bars_in_trade),
        elapsed_wall_clock_seconds=elapsed,
        current_executable_pnl_bps=float(trade_state.current_pnl_bps),
        cum_mfe_bps=float(trade_state.cum_mfe_bps),
        cum_mae_bps=float(trade_state.cum_mae_bps),
        bars_since_mfe_peak=int(trade_state.bars_since_mfe_peak),
    )


__all__ = (
    "LIFETIME_SUMMARY_DIM",
    "LIFETIME_SUMMARY_FIELD_ORDER",
    "LIFETIME_SUMMARY_SCHEMA_VERSION",
    "build_lifetime_summary",
    "lifetime_summary_from_path",
    "lifetime_summary_from_trade_state",
    "lifetime_summary_registry",
    "require_lifetime_summary",
)
