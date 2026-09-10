from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.unified_exit_lifetime_summary_v1 import (
    LIFETIME_SUMMARY_FIELD_ORDER,
    lifetime_summary_from_path,
    lifetime_summary_registry,
    require_lifetime_summary,
)
from gx1.execution.v12_trade_state import TradeState
from gx1.models.entry_v10.direction_decision_contract import (
    CLOSED_M1_PATH_SCHEMA_VERSION,
)


def _state(side: str) -> TradeState:
    return TradeState(
        entry_ts=pd.Timestamp("2026-07-03T21:54:30Z"),
        side=side,
        entry_bid=100.0,
        entry_ask=100.1,
        entry_spread_bps=10.0,
        v10_snapshot={},
        entry_decision_token_snapshot={},
        units=1,
        sizing_execution_evidence={},
        model_bundle_binding=None,
        entry_source_pair_binding=None,
        broker_account_binding=None,
        trade_id=f"fixture-{side}",
        current_bid=100.0,
        current_ask=100.1,
    )


def _bars() -> list[dict]:
    times = [
        "2026-07-03T21:55:00Z",
        "2026-07-03T21:56:00Z",
        "2026-07-05T22:05:00Z",
    ]
    mids = [100.2, 99.8, 100.4]
    bars = []
    for timestamp, mid in zip(times, mids, strict=True):
        bars.append(
            {
                "schema_version": CLOSED_M1_PATH_SCHEMA_VERSION,
                "time": timestamp,
                "complete": True,
                "source_path": "/immutable/pretest.parquet",
                "source_sha256": "a" * 64,
                "bid_open": mid - 0.05,
                "bid_high": mid + 0.15,
                "bid_low": mid - 0.25,
                "bid_close": mid - 0.05,
                "ask_open": mid + 0.05,
                "ask_high": mid + 0.25,
                "ask_low": mid - 0.15,
                "ask_close": mid + 0.05,
                "mid_open": mid,
                "mid_high": mid + 0.2,
                "mid_low": mid - 0.2,
                "mid_close": mid,
                "volume": 10,
            }
        )
    return bars


@pytest.mark.parametrize("side", ["long", "short"])
def test_offline_path_and_live_trade_state_have_byte_exact_summary(side: str) -> None:
    state = _state(side)
    bars = _bars()
    for bar in bars:
        state.update_bar(**bar)
    offline = lifetime_summary_from_path(
        side=side,
        entry_fill_time=state.entry_ts,
        entry_bid=state.entry_bid,
        entry_ask=state.entry_ask,
        bar_times=[bar["time"] for bar in bars],
        bid_high=[bar["bid_high"] for bar in bars],
        bid_low=[bar["bid_low"] for bar in bars],
        bid_close=[bar["bid_close"] for bar in bars],
        ask_high=[bar["ask_high"] for bar in bars],
        ask_low=[bar["ask_low"] for bar in bars],
        ask_close=[bar["ask_close"] for bar in bars],
    )
    live = state.lifetime_summary_v1()
    assert offline["raw"] == live["raw"]
    assert offline["summary_sha256"] == live["summary_sha256"]
    assert np.array_equal(offline["values"], live["values"])
    assert live["raw"]["elapsed_wall_clock_seconds"] > 48 * 60 * 60


def test_registry_is_ordered_and_summary_tamper_fails_closed() -> None:
    registry = lifetime_summary_registry()
    assert tuple(registry["field_order"]) == LIFETIME_SUMMARY_FIELD_ORDER
    state = _state("long")
    state.update_bar(**_bars()[0])
    summary = state.lifetime_summary_v1()
    require_lifetime_summary(summary)
    tampered = dict(summary)
    values = summary["values"].copy()
    values[0] += 1.0
    values.setflags(write=False)
    tampered["values"] = values
    with pytest.raises(RuntimeError, match="SUMMARY_INVALID"):
        require_lifetime_summary(tampered)
