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


# Dynamic successor regressions. Existing fixed-horizon fixtures above stay intact.
from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    DYNAMIC_M1_PREFIX_HORIZON_MINUTES, DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT,
    build_dynamic_m1_fill_surface, dynamic_m1_prefix_contract,
    prepare_dynamic_m1_prefix_index, query_dynamic_m1_prefix,
)


def _dynamic_index(frame, *, cutoff=None, offset=0, **kwargs):
    return prepare_dynamic_m1_prefix_index(
        frame, split_end_exclusive=(frame.time.iloc[-1] + pd.Timedelta(minutes=1)
                                    if cutoff is None else cutoff),
        source_parquet_sha256="a" * 64, source_row_offset=offset, **kwargs)


def _dynamic_query(index, *, horizons=(1,), entries=None, endpoint_kind="terminal_open_inclusive", chunk=4096):
    fill = build_dynamic_m1_fill_surface(index, m5_decision_times=[pd.Timestamp("2026-01-05T00:00Z")])
    return query_dynamic_m1_prefix(index, fill_surface=fill,
        query_entry_rows=np.zeros(len(horizons), dtype=np.int64) if entries is None else entries,
        horizon_minutes=np.asarray(horizons), endpoint_kind=endpoint_kind, query_chunk_rows=chunk)


def _exhaustive_dynamic_prefix(frame, start, end, side, *, terminal):
    """Tiny direct source scan, independent of the range index/chunk algorithm."""
    entry = float(frame.iloc[start]["ask_open" if side == 0 else "bid_open"])
    opening = frame["bid_open" if side == 0 else "ask_open"].to_numpy()
    favorable = frame["bid_high" if side == 0 else "ask_low"].to_numpy()
    adverse = frame["bid_low" if side == 0 else "ask_high"].to_numpy()
    candidates = list(favorable[start:end])
    if terminal:
        candidates.append(opening[end])
    position = int(np.argmax(candidates) if side == 0 else np.argmin(candidates))
    peak = start + position
    at_terminal = peak == end
    previous = [opening[start], *adverse[start:peak]]
    lo_price = min([*previous, opening[peak]]) if side == 0 else max([*previous, opening[peak]])
    hi_price = lo_price if at_terminal else (
        min([*previous, adverse[peak]]) if side == 0 else max([*previous, adverse[peak]]))
    if side == 0:
        values = ((candidates[position] - entry) / entry * 1e4,
                  (entry - lo_price) / entry * 1e4, (entry - hi_price) / entry * 1e4)
    else:
        values = ((entry - candidates[position]) / entry * 1e4,
                  (lo_price - entry) / entry * 1e4, (hi_price - entry) / entry * 1e4)
    return values, peak, at_terminal


@pytest.mark.parametrize("endpoint", ["terminal_open_inclusive", "closed_m1_prefix"])
def test_dynamic_queries_match_exhaustive_source_prefixes_both_sides(endpoint):
    frame = _m1_frame(periods=45)
    rng = np.random.default_rng(41)
    opened = 100 + rng.normal(0, .3, len(frame)).cumsum()
    for prefix, spread in (("bid", 0.), ("ask", .1)):
        frame[prefix + "_open"] = opened + spread
    frame["bid_high"] = opened + rng.uniform(0, .8, len(frame))
    frame["bid_low"] = opened - rng.uniform(0, .8, len(frame))
    frame["ask_high"] = frame["bid_high"] + .1
    frame["ask_low"] = frame["bid_low"] + .1
    index = _dynamic_index(frame, offset=1234)
    fill = build_dynamic_m1_fill_surface(index, m5_decision_times=frame.time.iloc[[0, 5, 10]])
    entries = np.repeat(np.arange(3, dtype=np.int64), 20)
    horizons = np.tile(np.arange(1, 21, dtype=np.int64), 3)
    actual = query_dynamic_m1_prefix(index, fill_surface=fill, query_entry_rows=entries,
        horizon_minutes=horizons, endpoint_kind=endpoint, query_chunk_rows=7)
    assert actual["path_observed"].all()
    for row, (entry_row, horizon) in enumerate(zip(entries, horizons)):
        start = int(fill.iloc[entry_row].entry_m1_row) - 1234
        end = start + int(horizon)
        for side in (0, 1):
            values, peak, at_terminal = _exhaustive_dynamic_prefix(
                frame, start, end, side, terminal=endpoint == "terminal_open_inclusive")
            observed = [actual[name][row, side] for name in
                ("mfe_bps", "mae_before_first_mfe_lower_bps", "mae_before_first_mfe_upper_bps")]
            np.testing.assert_allclose(observed, values, rtol=0, atol=1e-10)
            assert actual["first_mfe_source_row"][row, side] == peak + 1234
            assert actual["first_mfe_at_terminal_open"][row, side] == at_terminal
    assert index.binding()["source_row_offset"] == 1234
    assert index.binding()["indexed_rows"] == len(frame)
    assert all(not values.flags.writeable for values in index.source.values.values())


def test_dynamic_unknown_gap_keeps_endpoint_return_and_shorter_prefixes():
    frame = _m1_frame(periods=20).drop(index=8).reset_index(drop=True)
    index = _dynamic_index(frame)
    out = _dynamic_query(index, horizons=(1, 3, 5))
    assert out["terminal_gross_observed"].tolist() == [[True, True], [False, False], [True, True]]
    assert out["path_observed"].tolist() == [[True, True], [False, False], [False, False]]
    assert out["path_censor_reason"].tolist() == ["observed", "terminal_quote_missing", "unknown_source_absence"]
    assert np.isfinite(out["terminal_gross_return_bps"][2]).all()
    assert np.isnan(out["mfe_bps"][2]).all()
    closed = _dynamic_query(index, horizons=(3,), endpoint_kind=DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT)
    assert closed["path_observed"].all() and not closed["terminal_gross_observed"].any()
    assert closed["observed_m1_count"].tolist() == [3]


def test_dynamic_terminal_open_is_peak_without_reading_terminal_high_low():
    frame = _m1_frame(periods=15)
    frame.loc[10, ["bid_open", "ask_open", "bid_high", "ask_high"]] = [200., 200.1, 210., 210.1]
    first = _dynamic_query(_dynamic_index(frame), horizons=(5,))
    frame.loc[10, ["bid_high", "ask_high", "bid_low", "ask_low"]] = [10000., 10000.1, .01, .11]
    changed = _dynamic_query(_dynamic_index(frame), horizons=(5,))
    for name in ("terminal_gross_return_bps", "mfe_bps", "mae_before_first_mfe_lower_bps",
                 "mae_before_first_mfe_upper_bps", "first_mfe_source_row"):
        np.testing.assert_array_equal(first[name], changed[name])
    assert first["first_mfe_at_terminal_open"][0, 0]
    assert first["mae_before_first_mfe_lower_bps"][0, 0] == first["mae_before_first_mfe_upper_bps"][0, 0]
    assert np.all(first["mfe_bps"] >= first["terminal_gross_return_bps"])


@pytest.mark.parametrize("side", [0, 1])
def test_dynamic_first_equal_peak_preserves_earlier_intraminute_bounds(side):
    frame = _m1_frame(periods=15)
    for prefix, spread in (("bid", 0.), ("ask", 1.)):
        for field in ("open", "high", "low"):
            frame[prefix + "_" + field] = 100. + spread
    if side == 0:
        frame.loc[6, ["bid_high", "ask_high", "bid_low", "ask_low"]] = [105., 106., 90., 91.]
        frame.loc[8, ["bid_high", "ask_high", "bid_low", "ask_low"]] = [105., 106., 70., 71.]
        frame.loc[10, ["bid_open", "ask_open", "bid_high", "ask_high"]] = [105., 106., 105., 106.]
        expected_lower, expected_upper = 1/101*1e4, 11/101*1e4
    else:
        frame.loc[6, ["bid_low", "ask_low", "bid_high", "ask_high"]] = [94., 95., 109., 110.]
        frame.loc[8, ["bid_low", "ask_low", "bid_high", "ask_high"]] = [94., 95., 139., 140.]
        frame.loc[10, ["bid_open", "ask_open", "bid_low", "ask_low"]] = [94., 95., 94., 95.]
        expected_lower, expected_upper = 1/100*1e4, 10/100*1e4
    out = _dynamic_query(_dynamic_index(frame), horizons=(5,))
    assert out["first_mfe_source_row"][0, side] == 6
    assert not out["first_mfe_at_terminal_open"][0, side]
    assert out["mae_before_first_mfe_lower_bps"][0, side] == pytest.approx(expected_lower)
    assert out["mae_before_first_mfe_upper_bps"][0, side] == pytest.approx(expected_upper)


def test_dynamic_cutoff_precedes_price_validation_and_preserves_closed_only_boundary():
    frame = _m1_frame(periods=20)
    cutoff = frame.time.iloc[12]
    fields = [name for name in frame if name != "time"]
    frame.loc[12:, fields] = np.nan  # Reading/validating these prices is forbidden.
    index = _dynamic_index(frame, cutoff=cutoff)
    assert index.binding()["source_rows"] == 20 and index.binding()["indexed_rows"] == 12
    out = _dynamic_query(index, horizons=(6, 7))
    assert out["terminal_gross_observed"].tolist() == [[True, True], [False, False]]
    assert out["path_censor_reason"][1] == "split_right_censor"
    closed = _dynamic_query(index, horizons=(7,), endpoint_kind="closed_m1_prefix")
    assert closed["path_observed"].all()
    frame.loc[10, "bid_open"] = np.nan
    with pytest.raises(RuntimeError, match="SOURCE_VALUES_INVALID"):
        _dynamic_index(frame, cutoff=cutoff)


def test_dynamic_missing_exact_fill_is_not_replaced_and_fill_binding_cannot_be_forged():
    frame = _m1_frame(periods=20)
    index = _dynamic_index(frame.drop(index=5).reset_index(drop=True))
    out = _dynamic_query(index)
    assert not out["terminal_gross_observed"].any() and not out["path_observed"].any()
    index = _dynamic_index(frame, offset=99)
    fill = build_dynamic_m1_fill_surface(index, m5_decision_times=frame.time.iloc[[0]])
    fill.loc[0, "entry_m1_row"] += 1
    with pytest.raises(RuntimeError, match="FILL_SOURCE_ROW_MISMATCH"):
        query_dynamic_m1_prefix(index, fill_surface=fill, query_entry_rows=[0], horizon_minutes=[1])


def test_dynamic_all_480_horizons_use_resource_chunks_without_a_global_query_limit(monkeypatch):
    import gx1.contracts.entry_causal_m1_outcomes_v1 as owner
    index = _dynamic_index(_m1_frame(periods=500))
    original = owner._DynamicRangeArgIndex.query
    widths = []
    def recorded(self, starts, ends):
        widths.append(len(starts))
        return original(self, starts, ends)
    monkeypatch.setattr(owner._DynamicRangeArgIndex, "query", recorded)
    horizons = np.tile(np.asarray(DYNAMIC_M1_PREFIX_HORIZON_MINUTES), 16)
    out = _dynamic_query(index, horizons=horizons)
    assert out["terminal_gross_return_bps"].shape == (16 * 480, 2)
    assert out["terminal_gross_observed"].all() and out["path_observed"].all()
    assert max(widths) <= 4096
    assert dynamic_m1_prefix_contract()["horizon_minutes"] == list(range(1, 481))


@pytest.mark.parametrize("horizons", [[0], [481], [1.0], [True], [1, 2]])
def test_dynamic_query_rejects_invalid_or_unmatched_horizon_requests(horizons):
    index = _dynamic_index(_m1_frame())
    fill = build_dynamic_m1_fill_surface(index, m5_decision_times=[pd.Timestamp("2026-01-05T00:00Z")])
    with pytest.raises(RuntimeError, match="QUERY_INVALID"):
        query_dynamic_m1_prefix(index, fill_surface=fill, query_entry_rows=[0], horizon_minutes=horizons)


@pytest.mark.parametrize("endpoint", ["terminal_open_inclusive", "closed_m1_prefix"])
def test_dynamic_signed_negative_mfe_retains_initial_spread(endpoint):
    frame = _m1_frame(periods=15)
    for prefix, value in (("bid", 100.), ("ask", 101.)):
        for field in ("open", "high", "low"):
            frame[prefix + "_" + field] = value
    out = _dynamic_query(_dynamic_index(frame), horizons=(5,), endpoint_kind=endpoint)
    np.testing.assert_allclose(out["mfe_bps"], [[-1/101*1e4, -1/100*1e4]])
    np.testing.assert_allclose(out["mae_before_first_mfe_lower_bps"], -out["mfe_bps"])
    np.testing.assert_array_equal(out["mae_before_first_mfe_lower_bps"],
                                  out["mae_before_first_mfe_upper_bps"])
    np.testing.assert_array_equal(out["first_mfe_source_row"], [[5, 5]])
    assert not out["first_mfe_at_terminal_open"].any()


def _dynamic_closure_binding(frame, *, method="externally_sourced_exact_xau_utc_closure_intervals_v1"):
    """Synthetic declared schedule; not evidence about actual broker sessions."""
    from pathlib import Path
    from gx1.contracts.unified_exit_market_closure_authority_v1 import (
        MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION, build_market_closure_authority,
        build_unknown_gap_only_schedule, seal_exact_market_schedule,
    )
    if method == "observed_m1_clock_all_gaps_unknown_v1":
        schedule = build_unknown_gap_only_schedule(frame.time)
    else:
        schedule = seal_exact_market_schedule({
            "schema_version": MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
            "decision": "PASS", "instrument": "XAU_USD", "timeframe": "M1",
            "coverage_start_utc": frame.time.iloc[0].isoformat(),
            "coverage_end_utc_exclusive": (frame.time.iloc[-1] + pd.Timedelta(minutes=1)).isoformat(),
            "interval_semantics": "left_closed_right_open_utc",
            "source_method": method, "source_reference_sha256": "1" * 64,
            "intervals": [{"kind": "daily_maintenance",
                "start_utc": (frame.time.iloc[9] + pd.Timedelta(minutes=1)).isoformat(),
                "end_utc_exclusive": frame.time.iloc[10].isoformat(),
                "source_event_id": "synthetic-fixture-exact-gap"}],
            "test_data_used": False,
        })
    authority = build_market_closure_authority(
        m1_times=frame.time, m1_source_path=Path("/immutable/quotes.parquet"),
        m1_source_sha256="a" * 64,
        m1_source_manifest_path=Path("/immutable/quotes.manifest.json"),
        m1_source_manifest_sha256="b" * 64, exact_schedule=schedule,
        exact_schedule_path=Path("/immutable/closure.json"),
        exact_schedule_file_sha256="c" * 64)
    return {"exact_schedule": schedule, "closure_authority": authority}


def test_dynamic_only_exact_source_bound_closure_admits_cross_gap_path():
    frame = _m1_frame(periods=20)
    frame.loc[10:, "time"] += pd.Timedelta(minutes=60)
    unknown = _dynamic_query(_dynamic_index(frame), horizons=(65,))
    assert unknown["terminal_gross_observed"].all()
    assert not unknown["path_observed"].any()
    assert unknown["path_censor_reason"].tolist() == ["unknown_source_absence"]
    binding = _dynamic_closure_binding(frame)
    index = _dynamic_index(frame, **binding)
    exact = _dynamic_query(index, horizons=(65,))
    assert exact["terminal_gross_observed"].all() and exact["path_observed"].all()
    assert exact["observed_m1_count"].tolist() == [5]
    assert exact["elapsed_wall_seconds"].tolist() == [65 * 60]
    assert index.binding()["known_closure_count"] == 1
    assert index.binding()["unknown_gap_count"] == 0
    assert index.binding()["closure_authority_sha256"] == binding["closure_authority"]["artifact_sha256"]
    # Equal physical values with a different actual clock must not reuse a receipt.
    changed = frame.copy()
    changed.loc[10:, "time"] += pd.Timedelta(minutes=1)
    with pytest.raises(RuntimeError, match="CLOSURE_AUTHORITY_INVALID"):
        _dynamic_index(changed, **binding)
    with pytest.raises(RuntimeError, match="CLOSURE_AUTHORITY_INVALID"):
        prepare_dynamic_m1_prefix_index(frame,
            split_end_exclusive=frame.time.iloc[-1] + pd.Timedelta(minutes=1),
            source_parquet_sha256="d" * 64, **binding)


def test_dynamic_inferred_closures_are_rejected_and_unknown_receipts_remain_censored():
    frame = _m1_frame(periods=20)
    frame.loc[10:, "time"] += pd.Timedelta(minutes=60)
    inferred = _dynamic_closure_binding(frame, method="project_inferred_pretest_closure_policy_v1")
    with pytest.raises(RuntimeError, match="INFERRED_CLOSURE_FORBIDDEN"):
        _dynamic_index(frame, **inferred)
    unknown = _dynamic_closure_binding(frame, method="observed_m1_clock_all_gaps_unknown_v1")
    out = _dynamic_query(_dynamic_index(frame, **unknown), horizons=(65,))
    assert out["terminal_gross_observed"].all() and not out["path_observed"].any()
    assert out["path_censor_reason"].tolist() == ["unknown_source_absence"]


def test_dynamic_full_clock_with_only_loaded_prefix_prices_matches_projected_frame():
    from gx1.contracts.entry_causal_m1_outcomes_v1 import CausalM1QuoteSource
    frame = _m1_frame(periods=20)
    cutoff = frame.time.iloc[12]
    prefix = CausalM1QuoteSource(pd.DatetimeIndex(frame.time), {
        name: frame[name].iloc[:12].to_numpy(copy=True) for name in frame if name != "time"})
    actual = _dynamic_index(prefix, cutoff=cutoff)
    expected = _dynamic_index(frame, cutoff=cutoff)
    assert actual.binding() == expected.binding()
    a = _dynamic_query(actual, horizons=(1, 6, 7))
    b = _dynamic_query(expected, horizons=(1, 6, 7))
    for name in ("terminal_gross_return_bps", "mfe_bps", "path_observed", "terminal_gross_observed"):
        np.testing.assert_array_equal(a[name], b[name])
    assert actual.binding()["source_rows"] == 20 and actual.binding()["indexed_rows"] == 12
    # Retained prices and clock remain independent of subsequent caller mutation.
    old_clock = actual.source.times.asi8.copy()
    old_price = actual.source.values["bid_open"].copy()
    prefix.values["bid_open"][:] = 1.
    frame.loc[:, "time"] += pd.Timedelta(days=1)
    np.testing.assert_array_equal(actual.source.times.asi8, old_clock)
    np.testing.assert_array_equal(actual.source.values["bid_open"], old_price)


def test_dynamic_endpoint_codes_are_declared_known_query_semantics():
    from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_ENDPOINT_KINDS
    assert DYNAMIC_M1_PREFIX_ENDPOINT_KINDS == ("closed_m1_prefix", "terminal_open_inclusive")
    contract = dynamic_m1_prefix_contract()
    assert contract["endpoint_kinds"] == list(DYNAMIC_M1_PREFIX_ENDPOINT_KINDS)
    assert contract["endpoint_kind_codes"] == {"closed_m1_prefix": 0, "terminal_open_inclusive": 1}
    assert contract["future_horizon_max_target"] is False


@pytest.mark.parametrize("forbidden", ["future_elapsed_seconds", "first_mfe_source_row", "path_observed"])
def test_dynamic_query_cannot_accept_caller_supplied_future_observations(forbidden):
    index = _dynamic_index(_m1_frame())
    fill = build_dynamic_m1_fill_surface(index, m5_decision_times=[pd.Timestamp("2026-01-05T00:00Z")])
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        query_dynamic_m1_prefix(index, fill_surface=fill,
            query_entry_rows=[0], horizon_minutes=[1], **{forbidden: [0]})
