from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    build_entry_m1_fill_surface,
    causal_m1_outcomes_at_horizon,
    causal_m1_terminal_outcomes_at_horizon,
    causal_m1_target_contract,
    prepare_causal_m1_quote_source,
)
from gx1.contracts.entry_execution_causality_v1 import (
    legacy_same_close_target_contract_failures,
)


def _m1_frame(*, periods: int = 24) -> pd.DataFrame:
    times = pd.date_range("2026-01-05T00:00:00Z", periods=periods, freq="min")
    bid_open = 100.0 + np.arange(periods, dtype=np.float64)
    bid_high = bid_open + 0.4
    bid_low = bid_open - 0.2
    ask_open = bid_open + 0.1
    return pd.DataFrame(
        {
            "time": times,
            "bid_open": bid_open,
            "bid_high": bid_high,
            "bid_low": bid_low,
            "ask_open": ask_open,
            "ask_high": bid_high + 0.1,
            "ask_low": bid_low + 0.1,
        }
    )


def test_causal_target_contract_has_no_same_close_price() -> None:
    assert legacy_same_close_target_contract_failures(causal_m1_target_contract()) == []


def test_m5_decision_binds_to_next_m1_open_and_path_excludes_exit_bar() -> None:
    m1 = _m1_frame()
    m5_times = pd.DatetimeIndex(
        pd.to_datetime(
            ["2026-01-05T00:00:00Z", "2026-01-05T00:05:00Z"], utc=True
        )
    )
    surface = build_entry_m1_fill_surface(
        m5_decision_times=m5_times,
        closed_m1=m1,
    )
    outcomes = causal_m1_outcomes_at_horizon(
        fill_surface=surface,
        closed_m1=m1,
        horizon_m5_bars=1,
    )

    first = outcomes.iloc[0]
    assert surface["entry_decision_at"].iloc[0] == pd.Timestamp(
        "2026-01-05T00:05:00Z"
    )
    assert surface["entry_m1_row"].iloc[0] == 5
    assert surface["entry_ask"].iloc[0] == pytest.approx(105.1)
    assert first["exit_decision_at"] == pd.Timestamp("2026-01-05T00:10:00Z")
    assert first["exit_bid"] == pytest.approx(110.0)
    assert first["long_executable_pnl_bps"] == pytest.approx(
        (110.0 / 105.1 - 1.0) * 1e4
    )
    # The exit's M1 bar begins only after the path ends. Its 110.4 high cannot
    # become an earlier favourable excursion.
    assert first["long_mfe_bps"] == pytest.approx(
        (109.4 / 105.1 - 1.0) * 1e4
    )
    assert first["long_mae_bps"] == pytest.approx(
        (1.0 - 104.8 / 105.1) * 1e4
    )
    assert bool(first["outcome_valid"]) is True


def test_missing_m1_path_never_substitutes_an_exit_quote() -> None:
    m1 = _m1_frame().drop(index=8).reset_index(drop=True)
    surface = build_entry_m1_fill_surface(
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(["2026-01-05T00:00:00Z"], utc=True)
        ),
        closed_m1=m1,
    )
    outcomes = causal_m1_outcomes_at_horizon(
        fill_surface=surface,
        closed_m1=m1,
        horizon_m5_bars=1,
    )

    row = outcomes.iloc[0]
    assert bool(row["outcome_valid"]) is False
    assert pd.isna(row["exit_decision_at"])
    assert np.isnan(row["exit_bid"])
    assert np.isnan(row["long_executable_pnl_bps"])
    assert np.isnan(row["long_mfe_bps"])


def test_missing_fill_minute_is_marked_unbound_not_replaced_by_m5_close() -> None:
    m1 = _m1_frame().drop(index=5).reset_index(drop=True)
    surface = build_entry_m1_fill_surface(
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(
                ["2026-01-05T00:00:00Z", "2026-01-05T00:05:00Z"], utc=True
            )
        ),
        closed_m1=m1,
    )

    assert surface["entry_fill_bound"].tolist() == [False, True]
    assert surface["entry_m1_row"].tolist()[0] == -1
    assert np.isnan(surface["entry_bid"].iloc[0])


def test_terminal_outcome_has_same_clock_and_pnl_without_path_windows() -> None:
    m1 = _m1_frame()
    surface = build_entry_m1_fill_surface(
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(["2026-01-05T00:00:00Z"], utc=True)
        ),
        closed_m1=m1,
    )
    out = causal_m1_terminal_outcomes_at_horizon(
        fill_surface=surface, closed_m1=m1, horizon_m5_bars=1
    )
    assert bool(out.loc[0, "outcome_valid"])
    assert out.loc[0, "exit_decision_at"] == pd.Timestamp("2026-01-05T00:10:00Z")
    assert out.loc[0, "long_executable_pnl_bps"] == pytest.approx(
        (110.0 / 105.1 - 1.0) * 1e4
    )


@pytest.mark.parametrize("price_step", [-1.0, 1.0])
def test_both_materializers_use_entry_notional_for_long_and_short_pnl(
    price_step: float,
) -> None:
    m1 = _m1_frame()
    bid = 100.0 + price_step * np.arange(len(m1), dtype=np.float64)
    for prefix, spread in (("bid", 0.0), ("ask", 0.1)):
        m1[f"{prefix}_open"] = bid + spread
        m1[f"{prefix}_high"] = bid + spread + 0.4
        m1[f"{prefix}_low"] = bid + spread - 0.2
    surface = build_entry_m1_fill_surface(
        m5_decision_times=m1["time"].iloc[[0]], closed_m1=m1,
    )
    full = causal_m1_outcomes_at_horizon(
        fill_surface=surface, closed_m1=m1, horizon_m5_bars=1,
    )
    terminal = causal_m1_terminal_outcomes_at_horizon(
        fill_surface=surface, closed_m1=m1, horizon_m5_bars=1,
    )
    entry_bid = float(m1.loc[5, "bid_open"])
    entry_ask = float(m1.loc[5, "ask_open"])
    exit_bid = float(m1.loc[10, "bid_open"])
    exit_ask = float(m1.loc[10, "ask_open"])
    expected_long = (exit_bid - entry_ask) / entry_ask * 1e4
    expected_short = (entry_bid - exit_ask) / entry_bid * 1e4
    for out in (full, terminal):
        assert bool(out.loc[0, "outcome_valid"])
        assert out.loc[0, "long_executable_pnl_bps"] == pytest.approx(expected_long)
        assert out.loc[0, "short_executable_pnl_bps"] == pytest.approx(expected_short)
        assert out.loc[0, "short_executable_pnl_bps"] != pytest.approx(
            (entry_bid / exit_ask - 1.0) * 1e4
        )
    np.testing.assert_array_equal(
        full[["long_executable_pnl_bps", "short_executable_pnl_bps"]],
        terminal[["long_executable_pnl_bps", "short_executable_pnl_bps"]],
    )


def test_causality_gate_rejects_reciprocal_return_contract_v1() -> None:
    old_contract = causal_m1_target_contract()
    old_contract["schema_version"] = "entry_causal_m1_outcomes_v1"
    old_contract.pop("long_pnl_bps_formula")
    old_contract.pop("short_pnl_bps_formula")
    assert legacy_same_close_target_contract_failures(old_contract)


def test_prepared_m1_source_reuses_the_exact_validated_quotes() -> None:
    m1 = _m1_frame()
    prepared = prepare_causal_m1_quote_source(m1)
    assert all(not values.flags.writeable for values in prepared.values.values())
    surface = build_entry_m1_fill_surface(
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(["2026-01-05T00:00:00Z"], utc=True)
        ),
        closed_m1=prepared,
    )
    out = causal_m1_terminal_outcomes_at_horizon(
        fill_surface=surface, closed_m1=prepared, horizon_m5_bars=1
    )
    assert bool(out.loc[0, "outcome_valid"])


def test_unbound_surface_row_cannot_hide_an_arbitrary_m1_pointer() -> None:
    m1 = _m1_frame().drop(index=5).reset_index(drop=True)
    surface = build_entry_m1_fill_surface(
        m5_decision_times=pd.DatetimeIndex(
            pd.to_datetime(["2026-01-05T00:00:00Z"], utc=True)
        ),
        closed_m1=m1,
    )
    surface.loc[0, "entry_m1_row"] = -2

    with pytest.raises(
        RuntimeError, match="ENTRY_CAUSAL_M1_FILL_SURFACE_QUOTE_INVALID"
    ):
        causal_m1_outcomes_at_horizon(
            fill_surface=surface,
            closed_m1=m1,
            horizon_m5_bars=1,
        )


@pytest.mark.parametrize("field", ("open", "high", "low"))
def test_native_locked_quotes_are_exact_and_crossed_quotes_still_fail(field):
    m1 = _m1_frame()
    m1.loc[5, f"ask_{field}"] = m1.loc[5, f"bid_{field}"]
    prepared = prepare_causal_m1_quote_source(m1)
    assert prepared.values[f"ask_{field}"][5] == m1.loc[5, f"bid_{field}"]
    surface = build_entry_m1_fill_surface(m5_decision_times=m1.time.iloc[[0]], closed_m1=prepared)
    for owner in (causal_m1_outcomes_at_horizon, causal_m1_terminal_outcomes_at_horizon):
        result = owner(fill_surface=surface, closed_m1=prepared, horizon_m5_bars=1)
        assert bool(result.loc[0, "outcome_valid"])
        assert surface.loc[0, "entry_ask"] == m1.loc[5, "ask_open"]
    m1.loc[5, f"ask_{field}"] = m1.loc[5, f"bid_{field}"] - 0.01
    with pytest.raises(RuntimeError, match="SOURCE_QUOTE_GEOMETRY_INVALID"):
        prepare_causal_m1_quote_source(m1)



def test_m5_path_uses_first_executable_peak_and_conservative_peak_minute_bounds():
    from gx1.contracts.entry_causal_m1_outcomes_v1 import causal_m1_path_quality_outcomes
    m1 = _m1_frame(periods=12)
    for prefix, spread in (('bid', 0.), ('ask', 1.)):
        m1[prefix + '_open'] = 100. + spread
        m1[prefix + '_high'] = 100. + spread
        m1[prefix + '_low'] = 100. + spread
    # Fill at minute5: bid100/ask101. Peak's low may occur before OR after
    # its high; later adverse excursions must not leak into pre-first-peak MAE.
    m1.loc[6, ['bid_high', 'ask_high', 'bid_low', 'ask_low']] = [105., 106., 90., 91.]
    m1.loc[7, ['bid_high', 'ask_high', 'bid_low', 'ask_low']] = [104., 105., 80., 81.]
    m1.loc[8, ['bid_high', 'ask_high', 'bid_low', 'ask_low']] = [105., 106., 70., 71.]
    m1.loc[10, ['bid_high', 'ask_high']] = [200., 201.]  # terminal bucket excluded
    fill = build_entry_m1_fill_surface(m5_decision_times=m1.time.iloc[[0]], closed_m1=m1)
    out = causal_m1_path_quality_outcomes(fill_surface=fill, closed_m1=m1, horizon_m5_bars=1).iloc[0]
    assert bool(out['m5_path_target_valid'])
    assert out['y_long_m5_path_first_mfe_m1_offset'] == 1
    assert out['y_long_m5_path_mfe_bps'] == pytest.approx(4 / 101 * 1e4)
    assert out['y_long_m5_path_mae_before_first_mfe_lower_bps'] == pytest.approx(1 / 101 * 1e4)
    assert out['y_long_m5_path_mae_before_first_mfe_upper_bps'] == pytest.approx(11 / 101 * 1e4)
    assert out['y_long_m5_path_path_utility_bps'] == pytest.approx((4 - 11) / 101 * 1e4)


def test_m5_path_preserves_negative_mfe_and_short_entry_notional():
    from gx1.contracts.entry_causal_m1_outcomes_v1 import causal_m1_path_quality_outcomes
    m1 = _m1_frame(periods=12)
    for prefix, spread in (('bid', 0.), ('ask', 1.)):
        m1[prefix + '_open'] = 100. + spread
        m1[prefix + '_high'] = 100. + spread
        m1[prefix + '_low'] = 100. + spread
    m1.loc[6, ['bid_high', 'ask_high', 'bid_low', 'ask_low']] = [111., 112., 95., 96.]
    fill = build_entry_m1_fill_surface(m5_decision_times=m1.time.iloc[[0]], closed_m1=m1)
    out = causal_m1_path_quality_outcomes(fill_surface=fill, closed_m1=m1, horizon_m5_bars=1).iloc[0]
    assert out['y_short_m5_path_mfe_bps'] == pytest.approx(4 / 100 * 1e4)
    assert out['y_short_m5_path_mae_before_first_mfe_lower_bps'] == pytest.approx(1 / 100 * 1e4)
    assert out['y_short_m5_path_mae_before_first_mfe_upper_bps'] == pytest.approx(12 / 100 * 1e4)
    m1.loc[6, ['bid_high', 'ask_high', 'bid_low', 'ask_low']] = [100., 101., 100., 101.]
    out = causal_m1_path_quality_outcomes(fill_surface=fill, closed_m1=m1, horizon_m5_bars=1).iloc[0]
    assert out['y_long_m5_path_mfe_bps'] < 0
    assert out['y_long_m5_path_path_utility_bps'] < out['y_long_m5_path_mfe_bps']


@pytest.mark.parametrize('missing', ['fill', 'inside'])
def test_m5_path_censors_missing_fill_or_internal_gap_without_later_fill(missing):
    from gx1.contracts.entry_causal_m1_outcomes_v1 import causal_m1_path_quality_outcomes, M5_PATH_TARGET_COLUMNS
    m1 = _m1_frame(periods=12).drop(index=5 if missing == 'fill' else 7).reset_index(drop=True)
    fill = build_entry_m1_fill_surface(m5_decision_times=pd.DatetimeIndex(['2026-01-05T00:00Z']), closed_m1=m1)
    out = causal_m1_path_quality_outcomes(fill_surface=fill, closed_m1=m1, horizon_m5_bars=1)
    assert not bool(out.loc[0, 'm5_path_target_valid'])
    assert out[list(M5_PATH_TARGET_COLUMNS)].isna().all().all()


def test_m5_path_friday_signal_is_not_silently_filled_on_monday():
    from gx1.contracts.entry_causal_m1_outcomes_v1 import causal_m1_path_quality_outcomes
    m1 = _m1_frame(periods=12)
    m1['time'] = pd.date_range('2026-01-12T00:00Z', periods=len(m1), freq='min')
    fill = build_entry_m1_fill_surface(m5_decision_times=pd.DatetimeIndex(['2026-01-09T21:55Z']), closed_m1=m1)
    assert not bool(fill.loc[0, 'entry_fill_bound']) and fill.loc[0, 'entry_m1_row'] == -1
    out = causal_m1_path_quality_outcomes(fill_surface=fill, closed_m1=m1, horizon_m5_bars=1)
    assert not bool(out.loc[0, 'm5_path_target_valid'])
