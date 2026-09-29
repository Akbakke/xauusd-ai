"""Mechanics of the preregistered model-free baselines (synthetic tapes prove only that code runs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts.research_entry_direction_walkforward_v1 import Tape
from gx1.scripts import research_model_free_baselines_v1 as baselines

POLICY = {
    "slippage_bps_per_execution": 2.0,
    "commission_bps_per_execution": 0.0,
    "long_annual_cost_rate": 0.054,
    "short_annual_cost_rate": 0.0,
    "seconds_per_year": 31_557_600.0,
}


def _tape(times, mid, spread=0.2) -> Tape:
    mid = np.asarray(mid, dtype=np.float64)
    return Tape(
        time=pd.DatetimeIndex(times),
        mid=mid,
        bid=mid - spread / 2,
        ask=mid + spread / 2,
        manifest_sha256="0" * 64,
        root="synthetic",
    )


def _grid(start: str, bars: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=bars, freq="5min", tz="UTC")


def test_trade_outcomes_fill_cost_and_financing_scenarios():
    tape = _tape(_grid("2012-01-02 10:00", 13), np.linspace(100.0, 101.2, 13))
    entry, exit_ = np.array([0, 0, 0]), np.array([12, 12, 12])
    gross, net_a = baselines.trade_outcomes(tape, entry, exit_, np.array([1, -1, 0]), POLICY, financing=True)
    _, net_b = baselines.trade_outcomes(tape, entry, exit_, np.array([1, -1, 0]), POLICY, financing=False)
    assert gross[0] == pytest.approx((tape.bid[12] / tape.ask[0] - 1) * 1e4)
    assert gross[1] == pytest.approx((1 - tape.ask[12] / tape.bid[0]) * 1e4)
    assert gross[2] == 0.0 and net_a[2] == 0.0
    financing = 0.054 * 3600 / 31_557_600.0 * 1e4
    assert net_b[0] == pytest.approx(gross[0] - 4.0)
    assert net_a[0] == pytest.approx(gross[0] - 4.0 - financing)
    assert net_a[1] == pytest.approx(net_b[1]) == pytest.approx(gross[1] - 4.0)


def test_momentum_trades_never_span_a_gap_and_never_overlap():
    times = _grid("2012-01-02 10:00", 20).append(_grid("2012-01-02 13:00", 20))
    mid = np.concatenate([np.linspace(100, 102, 20), np.linspace(102, 100, 20)])
    entries, exits, sides = baselines.momentum_trades(
        _tape(times, mid), lookback=2, hold=3, reverse=False, start=pd.Timestamp("2012-01-01", tz="UTC")
    )
    breaks = baselines.contiguity_breaks(pd.DatetimeIndex(times))
    assert len(entries) > 0
    assert np.all(breaks[entries - 2] == breaks[exits])
    assert np.all(entries[1:] >= exits[:-1])
    assert set(sides[entries < 20]) == {1} and set(sides[entries >= 22]) == {-1}
    _, _, reversed_sides = baselines.momentum_trades(
        _tape(times, mid), lookback=2, hold=3, reverse=True, start=pd.Timestamp("2012-01-01", tz="UTC")
    )
    assert np.array_equal(reversed_sides, -sides)


def test_d1_trading_day_starts_at_22_utc():
    times = pd.DatetimeIndex(["2012-01-02 21:50", "2012-01-02 21:55", "2012-01-02 22:00", "2012-01-03 21:55"], tz="UTC")
    assert baselines.d1_closes(_tape(times, [1, 2, 3, 4])).tolist() == [1, 3]


def test_orb_enters_on_first_close_outside_the_first_hour_and_exits_before_session_end():
    times = _grid("2012-01-02 07:00", 108)  # 07:00 .. 15:55
    mid = np.full(len(times), 100.0)
    mid[:12] = np.linspace(99.5, 100.5, 12)
    mid[14] = 100.8  # 08:10 closes above the range
    entries, exits, sides = baselines.orb_trades(
        _tape(times, mid), open_hour=7, exit_hour=16, start=pd.Timestamp("2012-01-01", tz="UTC")
    )
    assert entries.tolist() == [14] and sides.tolist() == [1]
    assert times[exits[0]] == pd.Timestamp("2012-01-02 15:55", tz="UTC")


def test_session_window_crossing_midnight():
    times = _grid("2012-01-02 22:00", 108)  # 22:00 .. 06:55 next day
    entries, exits, sides = baselines.session_trades(
        _tape(times, np.linspace(100, 101, 108)), start_hour=22, end_hour=7, side=-1,
        start=pd.Timestamp("2012-01-01", tz="UTC"),
    )
    assert entries.tolist() == [0] and exits.tolist() == [107] and sides.tolist() == [-1]


def test_swing_signals_tsmom_and_sma():
    close = np.concatenate([np.linspace(100, 200, 300), np.linspace(200, 150, 60)])
    signals = baselines.swing_signals(close)
    assert np.isnan(signals["tsmom_252"][:252]).all()
    assert signals["tsmom_21"][250] == 1 and signals["tsmom_21"][-1] == -1
    assert signals["sma200"][299] == 1
    assert signals["combo"][299] == 1


@pytest.mark.parametrize(
    "stats,scenario_b,expected",
    [
        ({"t": 3.2, "positive_full_year_share": 0.7}, 1.0, "GO"),
        ({"t": 3.2, "positive_full_year_share": 0.7}, -1.0, "NO_GO"),
        ({"t": 3.2, "positive_full_year_share": 0.5}, None, "NO_GO"),
        ({"t": 2.4, "positive_full_year_share": 0.62}, None, "LOVENDE"),
        ({"t": 1.9, "positive_full_year_share": 0.9}, None, "NO_GO"),
        ({"n": 1}, None, "NO_GO"),
    ],
)
def test_verdict_rule_is_the_registration(stats, scenario_b, expected):
    assert baselines.verdict(stats, scenario_b_mean=scenario_b) == expected

def _curve(times, rates, end, markup=0.0129):
    return baselines.ResearchFinancingCurve(
        effective_at=pd.DatetimeIndex(times), benchmark_annual_rate=np.asarray(rates),
        coverage_end=pd.Timestamp(end), broker_markup=markup,
        seconds_per_year=POLICY["seconds_per_year"],
    )


def test_historical_financing_integrates_rate_changes_and_closed_market_days():
    curve = _curve(["2020-01-03T00:00Z", "2020-01-05T00:00Z"],
                   [0.04, 0.06], "2020-01-07T00:00Z")
    start = pd.DatetimeIndex(["2020-01-03T12:00Z"])
    end = pd.DatetimeIndex(["2020-01-06T12:00Z"])
    long_cost, short_cost = curve.integrated_cost_rates(start, end)
    year_days = POLICY["seconds_per_year"] / 86400
    assert long_cost[0] == pytest.approx((1.5 * 0.04 + 1.5 * 0.06 + 3 * 0.0129) / year_days)
    assert short_cost[0] == pytest.approx((3 * 0.0129 - 1.5 * 0.04 - 1.5 * 0.06) / year_days)
    assert short_cost[0] < 0  # credit is preserved
    # Integration is additive even when a rate change falls exactly on a boundary.
    split = pd.DatetimeIndex(["2020-01-05T00:00Z"])
    a = curve.integrated_cost_rates(start, split)
    b = curve.integrated_cost_rates(split, end)
    np.testing.assert_allclose(long_cost, a[0] + b[0])
    np.testing.assert_allclose(short_cost, a[1] + b[1])


def test_financing_coverage_and_mutability_fail_closed():
    with pytest.raises(RuntimeError, match="CURVE_INVALID"):
        _curve(["2020-01-01"], [0.04], "2020-01-03T00:00Z")
    rates = np.array([0.04])
    curve = _curve(["2020-01-01T00:00Z"], rates, "2020-01-03T00:00Z")
    rates[0] = 99
    assert curve.benchmark_annual_rate[0] == 0.04
    with pytest.raises(ValueError):
        curve.benchmark_annual_rate[0] = 99
    for start, end in [("2019-12-31T23:59Z", "2020-01-02T00:00Z"),
                       ("2020-01-02T00:00Z", "2020-01-03T00:01Z"),
                       ("2020-01-02T00:00Z", "2020-01-01T00:00Z")]:
        with pytest.raises(RuntimeError, match="COVERAGE_INVALID"):
            curve.integrated_cost_rates(pd.DatetimeIndex([start]), pd.DatetimeIndex([end]))


def test_historical_trade_financing_preserves_short_credit_and_zero_scenario():
    time = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    tape = _tape(time, [100, 110, 90, 120], spread=2)
    curve = _curve([time[0]], [0.0411], time[-1] + pd.Timedelta(days=1))
    _, financed = baselines.trade_outcomes(tape, np.array([0, 0]), np.array([3, 3]),
                                          np.array([1, -1]), POLICY, financing=True, financing_curve=curve)
    _, zero = baselines.trade_outcomes(tape, np.array([0, 0]), np.array([3, 3]),
                                      np.array([1, -1]), POLICY, financing=False, financing_curve=curve)
    years = 3 * 86400 / POLICY["seconds_per_year"]
    assert zero[0] - financed[0] == pytest.approx(0.054 * years * 1e4)
    assert zero[1] - financed[1] == pytest.approx(-0.0282 * years * 1e4)


@pytest.mark.parametrize("liquidate", [True, False])
def test_continuous_buy_and_hold_matches_one_cash_round_trip_and_marks_open_units(liquidate):
    time = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    tape = _tape(time, [100, 110, 90, 120], spread=2)
    path = baselines.portfolio_path(
        tape, np.ones(3), initial_equity=100, slippage_bps_per_execution=2,
        commission_bps_per_execution=1, financing_curve=None, liquidate_at_end=liquidate,
    )
    # Independent two-fill cash ledger. Interior observations create no executions.
    expected = 100 + tape.bid[-1] - tape.ask[0] - (tape.ask[0] + tape.bid[-1]) * 3 / 1e4
    assert path.equity_liquidation.iloc[-1] == pytest.approx(expected)
    assert path.traded_units.iloc[1:3].tolist() == [0, 0]
    summary = baselines.portfolio_summary(path, periods_per_year=252)
    assert summary["execution_count"] == (2 if liquidate else 1)
    assert summary["open_units"] == (0 if liquidate else 1)
    assert summary["total_net_bps"] == pytest.approx((expected / 100 - 1) * 1e4)
    assert summary["total_mid_pnl_bps"] == pytest.approx(2000)
    eq = np.r_[100, path.equity_liquidation.to_numpy()]
    assert summary["max_drawdown"] == pytest.approx(np.max(1 - eq / np.maximum.accumulate(eq)))
    prev = np.r_[100, path.equity_liquidation.to_numpy()[1:-1]]
    returns = path.equity_liquidation.to_numpy()[1:] / prev - 1
    assert summary["sharpe_zero_cash"] == pytest.approx(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    assert path.attrs["historical_broker_cost_truth"] is False


def test_portfolio_financing_keeps_opening_basis_through_price_changes_and_resizing():
    time = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    tape = _tape(time, [100, 110, 90, 120], spread=2)
    curve = _curve([time[0]], [0.0411], time[-1] + pd.Timedelta(days=1))
    kwargs = dict(initial_equity=1000, slippage_bps_per_execution=0,
                  commission_bps_per_execution=0, financing_curve=curve, liquidate_at_end=True)
    path = baselines.portfolio_path(tape, np.array([1, 2, 1]), **kwargs)
    np.testing.assert_allclose(path.opening_notional_after, [101, 212, 106, 0])
    day = 86400 / POLICY["seconds_per_year"]
    np.testing.assert_allclose(path.financing_cost, np.array([0, 101, 212, 106]) * 0.054 * day)
    short = baselines.portfolio_path(tape, -np.ones(3), **kwargs)
    np.testing.assert_allclose(short.financing_cost, np.array([0, 99, 99, 99]) * -0.0282 * day)
    flip = baselines.portfolio_path(tape, np.array([1, -1, -1]), **kwargs)
    np.testing.assert_allclose(flip.traded_units, [1, -2, 0, 1])
    np.testing.assert_allclose(flip.financing_cost, np.array([0, 101 * 0.054, -109 * 0.0282, -109 * 0.0282]) * day)
    flat = baselines.portfolio_path(tape, np.zeros(3), **kwargs)
    np.testing.assert_array_equal(flat.equity_liquidation, np.full(4, 1000))


def test_portfolio_rejects_missing_positions_and_preserves_insolvency():
    time = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    tape = _tape(time, [100, 20, 10], spread=0)
    kwargs = dict(initial_equity=100, slippage_bps_per_execution=0,
                  commission_bps_per_execution=0, financing_curve=None, liquidate_at_end=False)
    with pytest.raises(RuntimeError, match="INPUT_INVALID"):
        baselines.portfolio_path(tape, np.array([np.nan, 1]), **kwargs)
    summary = baselines.portfolio_summary(baselines.portfolio_path(tape, np.array([2, 2]), **kwargs),
                                          periods_per_year=252)
    assert summary["insolvent"] and summary["max_drawdown"] > 1
    assert "sharpe_zero_cash" not in summary


def test_causal_risk_sizing_uses_same_scale_for_long_and_model_and_ignores_future():
    mid = np.array([100, 102, 101, 104, 103, 108, 109, 107.0])
    side = np.array([1, -1, 0, 1, -1, 1, -1, 1.0])
    kwargs = dict(lookback=3, periods_per_year=252, target_annual_vol=0.10,
                  max_gross_leverage=1.0, initial_equity=100)
    long = baselines.causal_risk_units(mid, np.ones(len(mid)), **kwargs)
    model = baselines.causal_risk_units(mid, side, **kwargs)
    np.testing.assert_allclose(model, long * side, equal_nan=True)
    assert np.isnan(model[:3]).all()
    changed = mid.copy()
    changed[5:] *= 10
    future_changed = baselines.causal_risk_units(changed, side, **kwargs)
    np.testing.assert_allclose(model[:5], future_changed[:5], equal_nan=True)
    assert np.all(np.abs(model[3:]) * mid[3:] <= 100)

def test_stationary_bootstrap_preserves_blocks_and_paired_columns():
    n = 5000
    generator = baselines.stationary_bootstrap_indices(n, draws=2, mean_block_length=20, seed=19)
    first, second = list(generator)
    assert first.shape == (n,) and first.min() >= 0 and first.max() < n
    break_share = np.mean(first[1:] != (first[:-1] + 1) % n)
    assert abs(break_share - 1 / 20) < 0.02
    x = np.arange(n)
    paired = np.column_stack([x, -x])
    np.testing.assert_array_equal(paired[first].sum(axis=1), np.zeros(n))
    replay = list(baselines.stationary_bootstrap_indices(n, draws=2, mean_block_length=20, seed=19))
    np.testing.assert_array_equal(first, replay[0])
    assert not np.array_equal(first, second)
    iid = next(baselines.stationary_bootstrap_indices(n, draws=2, mean_block_length=1, seed=19))
    assert np.mean(iid[1:] != (iid[:-1] + 1) % n) > 0.95


def _infer(point, errors, names, scale=1.0):
    point = np.asarray(point) * scale
    errors = np.asarray(errors) * scale
    k = len(point)
    return baselines.max_t_inference(
        point, point + errors, names=names, alpha=0.05, desired_power=0.8,
        effect_sizes=np.tile([0.1, 0.5, 1.0], (k, 1)) * scale,
        minimum_relevant_effect=np.full(k, 0.1 * scale),
    )


def test_max_t_shared_duplicates_do_not_inflate_correction_and_units_are_invariant():
    rng = np.random.default_rng(10)
    errors = rng.normal(0, 0.1, size=(1999, 1))
    one = _infer([0.5], errors, ["edge"])
    duplicate = _infer([0.5, 0.5], np.repeat(errors, 2, axis=1), ["edge", "same_edge"])
    assert one["simultaneous_critical_value"] == pytest.approx(duplicate["simultaneous_critical_value"], rel=1e-12)
    for endpoint in duplicate["endpoints"]:
        assert endpoint["effect_verdict"] == "GO"
        assert endpoint["max_t_adjusted_two_sided_p"] == one["endpoints"][0]["max_t_adjusted_two_sided_p"]
    scaled = _infer([0.5], errors, ["edge"], scale=1e4)
    original, converted = one["endpoints"][0], scaled["endpoints"][0]
    assert converted["max_t_adjusted_two_sided_p"] == original["max_t_adjusted_two_sided_p"]
    assert converted["mde_against_zero"] == pytest.approx(original["mde_against_zero"] * 1e4)
    assert 0.8 - 1 / 1999 <= original["plug_in_power_at_mde"] <= 0.8 + 1 / 1999
    assert [r["power"] for r in original["power_at_declared_effects"]] == sorted(
        r["power"] for r in original["power_at_declared_effects"])


def test_max_t_distinguishes_evidence_of_small_effect_from_imprecision():
    rng = np.random.default_rng(12)
    errors = rng.normal(size=(1999, 3)) * np.array([0.01, 0.5, 0.01])
    report = _infer([0.0, 0.0, 0.5], errors, ["precise_small", "imprecise", "positive"])
    assert [r["effect_verdict"] for r in report["endpoints"]] == ["NO_GO", "INKONKLUSIV", "GO"]
    # Include the entire family: an independently varying endpoint raises the joint cutoff.
    alone = _infer([0.0], errors[:, :1], ["precise_small"])
    assert report["simultaneous_critical_value"] >= alone["simultaneous_critical_value"]
    e = report["endpoints"][0]
    maxima = np.max(np.abs(errors / errors.std(axis=0, ddof=1)), axis=1)
    assert np.mean(maxima <= report["simultaneous_critical_value"]) >= 0.95
    assert e["simultaneous_lower"] < 0 < e["simultaneous_upper"] < 0.1


def test_max_t_rejects_missing_or_degenerate_family_members():
    with pytest.raises(RuntimeError, match="DEGENERATE_ENDPOINT"):
        _infer([1.0], np.zeros((20, 1)), ["constant"])
    errors = np.arange(20.0)[:, None]
    errors[3] = np.nan
    with pytest.raises(RuntimeError, match="INPUT_INVALID"):
        _infer([1.0], errors, ["missing"])
    with pytest.raises(RuntimeError, match="INPUT_INVALID"):
        _infer([1.0, 1.0], np.ones((20, 2)), ["duplicate_name", "duplicate_name"])

def test_max_t_finite_draw_cutoff_agrees_with_adjusted_p_resolution():
    errors = np.linspace(-2, 2, 50)[:, None]
    point = np.array([1.96])
    report = baselines.max_t_inference(
        point, point + errors, names=["finite_draws"], alpha=0.05, desired_power=0.8,
        effect_sizes=np.array([[0.01, 0.1, 0.5]]), minimum_relevant_effect=np.array([0.01]),
    )
    endpoint = report["endpoints"][0]
    assert endpoint["max_t_adjusted_two_sided_p"] > 0.05
    assert endpoint["simultaneous_lower"] < 0
    assert endpoint["effect_verdict"] == "INKONKLUSIV"
    with pytest.raises(RuntimeError, match="DRAWS_TOO_FEW"):
        _infer([1.0], np.array([[-0.1], [0.1]]), ["too_few"])
