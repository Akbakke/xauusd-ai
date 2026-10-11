from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts import entry_observed_market_horizons_v1 as hz
from gx1.contracts.entry_causal_m1_outcomes_v1 import build_entry_m1_fill_surface
from gx1.contracts.entry_observed_market_v1 import entry_observed_market_economics
from tests.test_unified_exit_prospective_cost_policy_v1 import END, START, _build, _load


@pytest.fixture
def economics(tmp_path):
    authority = _load(_build(tmp_path), "parameter_authority")
    return entry_observed_market_economics(authority, coverage_start_utc=START, coverage_end_utc=END, verify_local_sources=False)


def _m1_sessions(*, weeks: int = 3, holiday: pd.Timestamp | None = None) -> pd.DataFrame:
    """Five-minute bars on the M1 grid for Monday..Friday sessions (22:00 UTC boundary); weekends are absent."""
    times = []
    start = pd.Timestamp("2026-01-04T22:00:00Z")  # Sunday 22:00 = Monday session open
    for day in range(7 * weeks):
        session_open = start + pd.Timedelta(days=day)
        weekday = (session_open + pd.Timedelta(hours=2)).weekday()  # weekday of the session's main day
        if weekday >= 5 or (holiday is not None and session_open == holiday):
            continue
        times.extend(session_open + pd.Timedelta(minutes=5 * k) for k in range(24 * 12))
    index = pd.DatetimeIndex(times)
    n = len(index)
    bid_open = 2000.0 + np.arange(n, dtype=np.float64) * 0.5
    return pd.DataFrame({"time": index, "bid_open": bid_open, "bid_high": bid_open + 1.0, "bid_low": bid_open - 1.0,
                         "ask_open": bid_open + 0.3, "ask_high": bid_open + 1.3, "ask_low": bid_open - 0.7})


def test_contract_names_horizons_decision_and_keys_and_is_stable():
    a, b = hz.entry_observed_market_horizons_contract(), hz.entry_observed_market_horizons_contract()
    assert a == b and a["observation_horizons_trading_days"] == [1, 5, 20] and a["decision_horizon_trading_days"] == 20
    assert a["decision_batch_target_key"] == "entry_observed_market_target_bps_h20d"
    assert a["holding_time_limit"] is False and a["serving_authority_granted"] is False and a["censored_rows_filled"] is False
    assert hz.session_day_index(pd.DatetimeIndex(["2026-01-05T21:59Z", "2026-01-05T22:00Z"]))[1] - \
        hz.session_day_index(pd.DatetimeIndex(["2026-01-05T21:59Z", "2026-01-05T22:00Z"]))[0] == 1


def test_one_and_five_trading_days_skip_the_weekend_and_keep_the_session_offset():
    m1 = _m1_sessions()
    entry = pd.DatetimeIndex(["2026-01-08T10:00:00Z"])  # Thursday 10:00, session opened Wed 22:00
    surface = build_entry_m1_fill_surface(m5_decision_times=entry, closed_m1=m1)
    one = hz.causal_m1_terminal_outcomes_at_trading_days(fill_surface=surface, closed_m1=m1, horizon_trading_days=1)
    five = hz.causal_m1_terminal_outcomes_at_trading_days(fill_surface=surface, closed_m1=m1, horizon_trading_days=5)
    # the M5 decision at bar open 10:00 is taken at 10:05; the exit keeps that offset in the target session
    assert bool(one["outcome_valid"].iloc[0]) and pd.Timestamp(one["exit_decision_at"].iloc[0]) == pd.Timestamp("2026-01-09T10:05:00Z")
    assert bool(five["outcome_valid"].iloc[0]) and pd.Timestamp(five["exit_decision_at"].iloc[0]) == pd.Timestamp("2026-01-15T10:05:00Z")
    assert five["elapsed_seconds_at_5d"].iloc[0] == 7 * 86400 and one["elapsed_seconds_at_1d"].iloc[0] == 86400
    row = m1.set_index("time")
    exit_bid, entry_ask = row.loc["2026-01-15T10:05:00Z", "bid_open"], row.loc["2026-01-08T10:05:00Z", "ask_open"]
    assert abs(five["y_long_gross_at_5d_bps"].iloc[0] - (exit_bid - entry_ask) / entry_ask * 1e4) < 1e-9
    exit_ask, entry_bid = row.loc["2026-01-15T10:05:00Z", "ask_open"], row.loc["2026-01-08T10:05:00Z", "bid_open"]
    assert abs(five["y_short_gross_at_5d_bps"].iloc[0] - (entry_bid - exit_ask) / entry_bid * 1e4) < 1e-9


def test_a_missing_session_is_skipped_not_counted_and_censoring_fails_closed():
    holiday = pd.Timestamp("2026-01-13T22:00:00Z")  # Wednesday 14 January session absent
    m1 = _m1_sessions(holiday=holiday)
    entry = pd.DatetimeIndex(["2026-01-08T10:00:00Z", "2026-01-20T10:00:00Z"])
    surface = build_entry_m1_fill_surface(m5_decision_times=entry, closed_m1=m1)
    five = hz.causal_m1_terminal_outcomes_at_trading_days(fill_surface=surface, closed_m1=m1, horizon_trading_days=5)
    assert pd.Timestamp(five["exit_decision_at"].iloc[0]) == pd.Timestamp("2026-01-16T10:05:00Z")  # one session later than without the holiday
    assert not bool(five["outcome_valid"].iloc[1]) and np.isnan(five["y_long_gross_at_5d_bps"].iloc[1]) and five["elapsed_seconds_at_5d"].iloc[1] == -1
    # entry at the last hour of a session whose target session has no bar at that offset: censored, never rolled
    short = m1.loc[~((m1["time"] >= "2026-01-09T20:00:00Z") & (m1["time"] < "2026-01-09T22:00:00Z"))]
    surface = build_entry_m1_fill_surface(m5_decision_times=pd.DatetimeIndex(["2026-01-08T21:00:00Z"]), closed_m1=short)
    one = hz.causal_m1_terminal_outcomes_at_trading_days(fill_surface=surface, closed_m1=short, horizon_trading_days=1)
    assert not bool(one["outcome_valid"].iloc[0])
    with pytest.raises(RuntimeError, match="HORIZON_INVALID"):
        hz.causal_m1_terminal_outcomes_at_trading_days(fill_surface=surface, closed_m1=short, horizon_trading_days=0)


def test_horizon_targets_charge_financing_over_actual_elapsed_seconds(economics):
    gross = {h: torch.tensor([[10.0, -10.0], [0.0, 0.0]]) for h in (1, 5, 20)}
    elapsed = {1: torch.tensor([86400, 86400]), 5: torch.tensor([7 * 86400, 7 * 86400]), 20: torch.tensor([28 * 86400, 28 * 86400])}
    out = hz.build_entry_observed_market_horizon_targets(gross_by_horizon=gross, elapsed_by_horizon=elapsed, economics=economics)
    assert set(out) == {1, 5, 20} and all(out[h]["target_bps"].shape == (2, 3) for h in out)
    assert bool((out[1]["target_bps"][:, 2] == 0).all())
    # longer horizons pay more financing on the same gross outcome
    assert out[20]["target_bps"][0, 0] <= out[5]["target_bps"][0, 0] <= out[1]["target_bps"][0, 0]
    with pytest.raises(RuntimeError, match="TARGET_INPUT_INVALID"):
        hz.build_entry_observed_market_horizon_targets(gross_by_horizon={1: gross[1]}, elapsed_by_horizon=elapsed, economics=economics)


def test_batch_requires_every_horizon_key_exactly():
    good = {hz.BATCH_TARGET_KEY_BY_HORIZON[h]: torch.tensor([[1.0, -1.0, 0.0]], dtype=torch.float32) for h in (1, 5, 20)}
    out = hz.require_entry_observed_market_horizon_batch(good, device=torch.device("cpu"))
    assert set(out) == {1, 5, 20} and out[20][0].shape == (1, 3)
    with pytest.raises(RuntimeError, match="h5d"):
        hz.require_entry_observed_market_horizon_batch({k: v for k, v in good.items() if "h5d" not in k}, device=torch.device("cpu"))
    bad = dict(good)
    bad[hz.DECISION_BATCH_TARGET_KEY] = torch.tensor([[1.0, -1.0, 0.5]], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="h20d"):
        hz.require_entry_observed_market_horizon_batch(bad, device=torch.device("cpu"))
