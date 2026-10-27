"""Focused synthetic qualification of first-positive quote target semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_causal_m1_outcomes_v1 import (
    DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT,
    DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT,
    _dynamic_range_arg_index,
    build_dynamic_m1_fill_surface,
    prepare_dynamic_m1_prefix_index,
    query_dynamic_m1_prefix,
)
from gx1.contracts.entry_causal_m1_first_positive_quote_v1 import (
    _first_strict_crossing,
    first_positive_quote_target_contract,
    query_first_positive_quote,
)


def _frame(periods=20):
    times = pd.date_range("2024-05-06T00:00:00Z", periods=periods, freq="min")
    data = {"time": times}
    for side in ("bid", "ask"):
        for kind in ("open", "high", "low"):
            data[side + "_" + kind] = np.full(len(times), 100., dtype=np.float64)
    return pd.DataFrame(data)


def _bar(frame, row, opening, high, low):
    for side in ("bid", "ask"):
        frame.loc[row, side + "_open"] = opening
        frame.loc[row, side + "_high"] = high
        frame.loc[row, side + "_low"] = low


def _index_and_fill(frame, *, cutoff=None, offset=0):
    if cutoff is None:
        cutoff = frame["time"].iloc[-1] + pd.Timedelta(minutes=1)
    index = prepare_dynamic_m1_prefix_index(
        frame, split_end_exclusive=cutoff, source_parquet_sha256="a" * 64,
        source_row_offset=offset,
    )
    fill = build_dynamic_m1_fill_surface(
        index, m5_decision_times=pd.DatetimeIndex([pd.Timestamp("2024-05-06T00:00:00Z")]),
    )
    return index, fill


def _query(index, fill, horizon, endpoint=DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT, basis="gross_quote_markout"):
    return query_first_positive_quote(
        index, fill_surface=fill, query_entry_rows=np.array([0], np.int64),
        horizon_minutes=np.array([horizon], np.int64), event_basis=basis, endpoint_kind=endpoint,
    )


def test_first_positive_is_not_first_horizon_maximum():
    frame = _frame()
    _bar(frame, 5, 100., 100.05, 100.)
    _bar(frame, 6, 100.05, 100.05, 99.80)
    _bar(frame, 7, 99.80, 100.10, 99.80)
    index, fill = _index_and_fill(frame, offset=17)
    result = _query(index, fill, 3)
    original = result["original_dynamic_prefix"]
    assert result["first_event_path_observed"][0, 0]
    assert result["first_event_nominal_happened"][0, 0]
    assert result["first_event_source_row"][0, 0] == 5 + 17
    assert result["first_event_entry_bar_order_unknown"][0, 0]
    assert result["nominal_pre_event_adverse_lower_bps"][0, 0] == pytest.approx(0.)
    assert result["nominal_pre_event_adverse_upper_bps"][0, 0] == pytest.approx(0.)
    assert original["first_mfe_source_row"][0, 0] == 7 + 17
    assert original["mfe_bps"][0, 0] == pytest.approx(10.)
    assert original["mae_before_first_mfe_lower_bps"][0, 0] == pytest.approx(20.)
    assert original["mae_before_first_mfe_upper_bps"][0, 0] == pytest.approx(20.)
    assert not result["first_event_nominal_happened"].flags.writeable


def test_same_bar_crossings_keep_side_specific_ambiguity_bounds():
    frame = _frame()
    _bar(frame, 5, 100., 100.1, 99.8)
    index, fill = _index_and_fill(frame)
    result = _query(index, fill, 1)
    assert result["first_event_path_observed"].tolist() == [[True, True]]
    assert result["first_event_nominal_happened"].tolist() == [[True, True]]
    assert result["first_event_at_open"].tolist() == [[False, False]]
    assert result["first_event_source_row"].tolist() == [[5, 5]]
    assert result["first_event_entry_bar_order_unknown"].tolist() == [[True, True]]
    assert result["first_event_occurrence_lower"].tolist() == [[False, False]]
    assert result["first_event_occurrence_upper"].tolist() == [[True, True]]
    assert result["nominal_pre_event_adverse_lower_bps"].tolist() == [[0., 0.]]
    assert result["nominal_pre_event_adverse_upper_bps"][0, 0] == pytest.approx(20.)
    assert result["nominal_pre_event_adverse_upper_bps"][0, 1] == pytest.approx(10.)
    assert result["first_event_nominal_bar_start_ns"][0, 0] == frame.loc[5, "time"].value
    assert result["first_event_nominal_bar_start_ns"][0, 0] == result["first_event_nominal_bar_start_ns"][0, 1]


def test_entry_bucket_candidate_can_be_confirmed_by_a_later_closed_bar():
    frame = _frame()
    _bar(frame, 5, 100., 100.05, 100.)
    _bar(frame, 6, 100., 100.10, 100.)
    index, fill = _index_and_fill(frame)
    result = _query(index, fill, 2)
    assert result["first_event_entry_bar_order_unknown"][0, 0]
    assert result["first_event_occurrence_lower"][0, 0]
    assert result["first_event_occurrence_upper"][0, 0]
    assert result["first_event_source_row"][0, 0] == 5


def test_crossing_at_later_open_does_not_include_later_bar_low():
    frame = _frame()
    _bar(frame, 6, 100.05, 100.05, 99.90)
    index, fill = _index_and_fill(frame)
    result = _query(index, fill, 2)
    assert result["first_event_nominal_happened"][0, 0]
    assert result["first_event_at_open"][0, 0]
    assert result["first_event_source_row"][0, 0] == 6
    assert not result["first_event_entry_bar_order_unknown"][0, 0]
    assert result["first_event_occurrence_lower"][0, 0]
    assert result["first_event_occurrence_upper"][0, 0]
    assert result["nominal_pre_event_adverse_lower_bps"][0, 0] == pytest.approx(0.)
    assert result["nominal_pre_event_adverse_upper_bps"][0, 0] == pytest.approx(0.)
    assert result["first_event_nominal_bar_start_ns"][0, 0] == frame.loc[6, "time"].value
    assert "subminute_quote_time_unavailable" in first_positive_quote_target_contract()["quote_time"]


def test_terminal_open_is_only_included_for_its_explicit_endpoint():
    frame = _frame()
    _bar(frame, 6, 100.05, 100.20, 99.80)
    index, fill = _index_and_fill(frame)
    closed = _query(index, fill, 1, DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT)
    inclusive = _query(index, fill, 1, DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT)
    assert closed["first_event_path_observed"][0, 0]
    assert not closed["first_event_nominal_happened"][0, 0]
    assert inclusive["first_event_nominal_happened"][0, 0]
    assert inclusive["first_event_at_open"][0, 0]
    assert inclusive["first_event_source_row"][0, 0] == 6
    assert inclusive["first_event_at_nominal_terminal_open"][0, 0]
    assert not inclusive["first_event_occurrence_lower"][0, 0]
    assert inclusive["first_event_occurrence_upper"][0, 0]
    assert inclusive["nominal_pre_event_adverse_upper_bps"][0, 0] == pytest.approx(0.)


def test_exact_zero_is_known_no_event_not_missing():
    index, fill = _index_and_fill(_frame())
    result = _query(index, fill, 2)
    assert result["first_event_path_observed"].tolist() == [[True, True]]
    assert result["first_event_nominal_happened"].tolist() == [[False, False]]
    assert result["first_event_source_row"].tolist() == [[-1, -1]]
    assert np.isnan(result["nominal_pre_event_adverse_lower_bps"]).all()
    assert np.isnan(result["nominal_pre_event_adverse_upper_bps"]).all()


def test_observed_spread_must_be_recovered_before_gross_positive():
    frame = _frame()
    for kind in ("open", "high", "low"):
        frame["ask_" + kind] += 0.02
    frame.loc[5, "bid_high"] = 100.01
    frame.loc[5, "ask_high"] = 100.03
    frame.loc[6, "bid_high"] = 100.03
    frame.loc[6, "ask_high"] = 100.05
    index, fill = _index_and_fill(frame)
    result = _query(index, fill, 2)
    assert result["first_event_source_row"][0, 0] == 6
    assert result["nominal_pre_event_adverse_lower_bps"][0, 0] == pytest.approx(0.02 / 100.02 * 10000.)
    assert "neither_includes_commission" in first_positive_quote_target_contract()["costs"]
    price_move = _query(index, fill, 2, basis="same_side_quote_move")
    assert price_move["first_event_source_row"][0, 0] == 5
    assert price_move["first_event_entry_bar_order_unknown"][0, 0]


def test_short_spread_must_also_be_recovered_before_gross_positive():
    frame = _frame()
    for kind in ("open", "high", "low"):
        frame["ask_" + kind] += 0.02
    frame.loc[5, "bid_low"] = 99.99
    frame.loc[5, "ask_low"] = 100.01
    frame.loc[6, "bid_low"] = 99.97
    frame.loc[6, "ask_low"] = 99.99
    index, fill = _index_and_fill(frame)
    gross = _query(index, fill, 2)
    assert gross["first_event_source_row"][0, 1] == 6
    assert gross["nominal_pre_event_adverse_lower_bps"][0, 1] == pytest.approx(2.)
    price_move = _query(index, fill, 2, basis="same_side_quote_move")
    assert price_move["first_event_source_row"][0, 1] == 5
    assert price_move["first_event_entry_bar_order_unknown"][0, 1]


def test_unknown_gap_stays_unknown_not_a_negative_event():
    frame = _frame().drop(index=6).reset_index(drop=True)
    index, fill = _index_and_fill(frame)
    result = _query(index, fill, 3)
    assert result["first_event_path_observed"].tolist() == [[False, False]]
    assert result["first_event_nominal_happened"].tolist() == [[False, False]]
    assert result["original_dynamic_prefix"]["path_censor_reason"][0] == "unknown_source_absence"
    assert np.isnan(result["nominal_pre_event_adverse_lower_bps"]).all()


def test_closed_cutoff_ignores_poisoned_later_prices_and_inclusive_is_censored():
    frame = _frame()
    cutoff = frame.loc[8, "time"]
    frame.loc[8:, ["bid_open", "bid_high", "bid_low"]] = -999.
    index, fill = _index_and_fill(frame, cutoff=cutoff)
    closed = _query(index, fill, 3, DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT)
    inclusive = _query(index, fill, 3, DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT)
    assert closed["first_event_path_observed"].tolist() == [[True, True]]
    assert not closed["first_event_nominal_happened"].any()
    assert inclusive["first_event_path_observed"].tolist() == [[False, False]]
    assert inclusive["original_dynamic_prefix"]["path_censor_reason"][0] == "split_right_censor"


@pytest.mark.parametrize("maximum", [True, False])
def test_first_crossing_range_search_matches_independent_linear_scan(maximum):
    rng = np.random.RandomState(1968)
    values = rng.randint(-8, 9, size=128).astype(np.float64)
    values.setflags(write=False)
    index = _dynamic_range_arg_index(values, maximum=maximum)
    starts = rng.randint(0, 120, size=300).astype(np.int64)
    ends = np.array([rng.randint(int(s) + 1, 129) for s in starts], np.int64)
    barrier = rng.randint(-10, 11, size=300).astype(np.float64)
    observed = _first_strict_crossing(index, starts, ends, barrier, maximum=maximum)
    expected = []
    for start, end, limit in zip(starts, ends, barrier):
        found = [i for i in range(int(start), int(end))
                 if (values[i] > limit if maximum else values[i] < limit)]
        expected.append(found[0] if found else -1)
    np.testing.assert_array_equal(observed, np.asarray(expected, np.int64))


def test_original_dynamic_labels_and_masks_are_unchanged():
    frame = _frame()
    _bar(frame, 5, 100., 100.10, 99.90)
    index, fill = _index_and_fill(frame)
    inputs = dict(fill_surface=fill, query_entry_rows=np.array([0], np.int64),
                  horizon_minutes=np.array([3], np.int64),
                  endpoint_kind=DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT)
    original = query_dynamic_m1_prefix(index, **inputs)
    returned = query_first_positive_quote(index, event_basis="gross_quote_markout", **inputs)["original_dynamic_prefix"]
    assert set(original) == set(returned)
    for key in original:
        if isinstance(original[key], np.ndarray):
            np.testing.assert_array_equal(original[key], returned[key])
        else:
            assert original[key] == returned[key]


def test_ten_and_sixty_minutes_use_the_same_entry_and_elapsed_clock():
    frame = _frame(periods=75)
    _bar(frame, 14, 100., 100.10, 100.)
    index, fill = _index_and_fill(frame)
    horizons = np.arange(1, 61, dtype=np.int64)
    result = query_first_positive_quote(
        index, fill_surface=fill, query_entry_rows=np.zeros(60, np.int64),
        horizon_minutes=horizons, event_basis="gross_quote_markout", endpoint_kind=DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT,
        query_chunk_rows=8,
    )
    assert result["first_event_path_observed"][:, 0].all()
    assert result["first_event_nominal_happened"][:, 0].tolist() == [False] * 9 + [True] * 51
    assert (result["first_event_source_row"][9:, 0] == 14).all()
    decision = frame.loc[5, "time"].value
    np.testing.assert_array_equal(
        result["original_dynamic_prefix"]["terminal_time_ns"],
        decision + horizons * 60_000_000_000,
    )


def test_one_minute_candidate_hit_is_flagged_and_no_hit_is_not_missing():
    frame = _frame()
    _bar(frame, 5, 100., 100.10, 100.)
    index, _ = _index_and_fill(frame)
    fill = build_dynamic_m1_fill_surface(
        index,
        m5_decision_times=pd.DatetimeIndex([frame.loc[0, "time"], frame.loc[5, "time"]]),
    )
    result = query_first_positive_quote(
        index, fill_surface=fill, query_entry_rows=np.array([0, 1], np.int64),
        horizon_minutes=np.array([1, 1], np.int64), event_basis="gross_quote_markout",
        endpoint_kind=DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT,
    )
    assert result["first_event_path_observed"][:, 0].tolist() == [True, True]
    assert result["first_event_nominal_happened"][:, 0].tolist() == [True, False]
    assert result["first_event_entry_bar_order_unknown"][:, 0].tolist() == [True, False]
    assert result["first_event_occurrence_lower"][:, 0].tolist() == [False, False]
    assert result["first_event_occurrence_upper"][:, 0].tolist() == [True, False]
    assert "outcome_dependent" in first_positive_quote_target_contract()["entry_bar_censoring"]


def test_event_basis_must_be_explicit_and_known():
    index, fill = _index_and_fill(_frame())
    with pytest.raises(RuntimeError, match="EVENT_BASIS"):
        _query(index, fill, 1, basis="not_an_event_basis")
    with pytest.raises(TypeError, match="event_basis"):
        query_first_positive_quote(
            index, fill_surface=fill, query_entry_rows=np.array([0], np.int64),
            horizon_minutes=np.array([1], np.int64),
        )


def test_bad_entry_pointer_fails_closed():
    index, fill = _index_and_fill(_frame())
    fill.loc[0, "entry_m1_row"] += 1
    with pytest.raises(RuntimeError, match="FILL_SOURCE_ROW_MISMATCH"):
        _query(index, fill, 1)


def test_chunked_queries_match_individual_queries():
    frame = _frame()
    _bar(frame, 6, 100., 100.10, 99.90)
    index, fill = _index_and_fill(frame)
    many = query_first_positive_quote(
        index, fill_surface=fill, query_entry_rows=np.array([0, 0, 0], np.int64),
        horizon_minutes=np.array([1, 2, 3], np.int64), event_basis="gross_quote_markout",
        endpoint_kind=DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT, query_chunk_rows=1,
    )
    for i, horizon in enumerate((1, 2, 3)):
        one = _query(index, fill, horizon)
        for key in ("first_event_path_observed", "first_event_nominal_happened", "first_event_occurrence_lower",
                    "first_event_occurrence_upper", "first_event_entry_bar_order_unknown", "first_event_at_open",
                    "first_event_source_row", "first_event_nominal_bar_start_ns",
                    "first_event_at_nominal_terminal_open", "nominal_pre_event_adverse_lower_bps",
                    "nominal_pre_event_adverse_upper_bps"):
            np.testing.assert_array_equal(many[key][i:i + 1], one[key])
