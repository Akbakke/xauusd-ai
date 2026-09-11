from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.unified_exit_economic_step_provider_v1 import (
    LazyUnifiedExitEconomicStepProviderV1,
)


def _provider(*, known: bool) -> LazyUnifiedExitEconomicStepProviderV1:
    provider = LazyUnifiedExitEconomicStepProviderV1.__new__(
        LazyUnifiedExitEconomicStepProviderV1
    )
    times = pd.DatetimeIndex(["2026-07-03T21:59Z", "2026-07-05T22:00Z"])
    provider._times_ns = np.asarray(times.asi8, dtype=np.int64)
    provider._prices = {
        "bid_close": np.asarray([100.0, 101.0]),
        "ask_close": np.asarray([100.1, 101.1]),
    }
    provider._component_hashes = {
        "executable_bid_ask": "1" * 64,
        "commission": "2" * 64,
        "execution_slippage": "3" * 64,
        "financing_or_swap": "4" * 64,
        "guaranteed_execution_fee": "5" * 64,
    }
    provider._commission_total = (0.0, 0.0)
    provider._slippage_total = (0.0, 0.0)
    provider._financing_cost_annual_bps = (500.0, 250.0)
    provider._risk_penalty = (100.0, 100.0)
    provider._authority_sha = "6" * 64
    provider._capital_hurdle_sha = "7" * 64
    provider._gap_source_sha = "8" * 64
    provider._closure_by_row = (
        {
            0: {
                "previous_bar_start_utc": times[0].isoformat(),
                "next_bar_start_utc": times[1].isoformat(),
                "classification": "declared_weekend_market_closure",
                "successor_across_gap_allowed": True,
                "interval_sha256": "9" * 64,
            }
        }
        if known
        else {}
    )
    return provider


def test_hold_step_uses_full_closure_wall_clock_for_financing_and_risk() -> None:
    provider = _provider(known=True)
    elapsed = int(
        (provider._times_ns[1] - provider._times_ns[0]) // 1_000_000_000
    )
    step = provider._step(
        entry_price=100.1,
        state_row=0,
        side_index=0,
        action="hold",
        event_kind="HOLD",
    )
    assert step["interval_end_time_ns"] - step["interval_start_time_ns"] == (
        elapsed * 1_000_000_000
    )
    assert step["gap"]["classification"] == "declared_market_closure"
    assert step["gap"]["classification_artifact_sha256"] == "9" * 64
    assert step["financing_or_swap"]["value_bps"] < 0.0
    assert step["risk_utility_penalty"]["value_bps"] > 0.0


def test_unknown_gap_still_fails_closed() -> None:
    provider = _provider(known=False)
    with pytest.raises(RuntimeError, match="GAP_UNVERIFIED"):
        provider._hold_elapsed_seconds(np.asarray([0], dtype=np.int64))
