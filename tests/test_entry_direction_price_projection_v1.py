"""Small synthetic endpoint/price-label tests; no genuine input or model."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import entry_causal_m1_outcomes_v1 as quote
from gx1.contracts import entry_direction_price_projection_v1 as price
from gx1.contracts.entry_direction_snapshot_data_v1 import direction_targets_from_observations
from tests.test_entry_causal_m1_outcomes import _m1_frame, _dynamic_index
from tests.test_entry_direction_snapshot_data_v1 import _economics


def _case(frame=None, *, cutoff=None, offset=0, bars=3):
    frame = _m1_frame(periods=45) if frame is None else frame
    index = _dynamic_index(frame, cutoff=cutoff, offset=offset)
    clocks = pd.date_range(frame.time.iloc[0].floor("5min"), periods=bars, freq="5min")
    fill = quote.build_dynamic_m1_fill_surface(index, m5_decision_times=clocks)
    return index, fill


def _project(index, fill, parents=(0,)):
    rows = np.asarray(parents, np.int64)
    return price.project_direction_price_endpoints(index, fill_surface=fill, parent_rows=rows,
        bar_start_ns=pd.DatetimeIndex(fill.iloc[rows].time).as_unit("ns").asi8.copy(),
        dynamic_entry_m1_row=fill.iloc[rows].entry_m1_row.to_numpy(np.int64, copy=True))


def _targets(result, index, economics):
    return direction_targets_from_observations(result["terminal_observations"],
        parent_rows=result["parent_rows"], decision_times_ns=result["decision_ns"],
        economics=economics, split_end_exclusive_ns=int(index.split_end_exclusive.value))


def _funding(rates):
    value = _economics()
    value.pop("economics_sha256")
    value["annual_financing_cost_bps"] = list(rates)
    value["economics_sha256"] = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return value


def test_flat_mid_is_known_zero_despite_spread_and_unequal_funding(monkeypatch):
    frame = _m1_frame(periods=25)
    for side, value in (("bid", 99.0), ("ask", 101.0)):
        for field in ("open", "high", "low"):
            frame[side + "_" + field] = value
    index, fill = _case(frame)
    def forbidden(*args, **kwargs):
        raise AssertionError("endpoint helper queried paths or rebuilt the index")
    monkeypatch.setattr(quote, "query_dynamic_m1_prefix", forbidden)
    monkeypatch.setattr(quote, "prepare_dynamic_m1_prefix_index", forbidden)
    result = _project(index, fill)
    assert result["observed"].all()
    np.testing.assert_array_equal(result["mid_move"], np.zeros((1, 3), np.float64))
    np.testing.assert_array_equal(result["mid_return_bps"], np.zeros((1, 3), np.float64))
    assert result["sign"].dtype == np.int8 and not result["sign"].any()
    unfunded = _targets(result, index, _funding((0.0, 0.0)))
    funded = _targets(result, index, _funding((532.0, 0.0)))
    assert np.all(unfunded["contrast_bps"] != 0)
    assert np.all(funded["contrast_bps"] != unfunded["contrast_bps"])
    assert result["observed"].all() and not result["sign"].any()


@pytest.mark.parametrize("offset", [0, 37])
def test_fixed_horizons_parent_order_and_existing_terminal_adapter_are_exact(offset):
    index, fill = _case(offset=offset)
    before = {name: value.tobytes() for name, value in index.source.values.items()}
    result = _project(index, fill, parents=(0, 2))
    np.testing.assert_array_equal(result["parent_rows"], [0, 2])
    np.testing.assert_array_equal(result["terminal_time_ns"],
        result["decision_ns"][:, None] + np.array([1, 5, 15], np.int64)[None, :] * 60 * 10**9)
    assert result["observed"].all() and np.all(result["sign"] == 1)
    assert result["mid_move"].dtype == result["mid_return_bps"].dtype == np.float64
    original = quote.query_dynamic_m1_prefix(index, fill_surface=fill,
        query_entry_rows=np.repeat(np.array([0, 2], np.int64), 3),
        horizon_minutes=np.tile(np.array([1, 5, 15], np.int64), 2))
    projected = result["terminal_observations"]
    for key in ("query_entry_rows", "horizon_minutes", "terminal_time_ns", "elapsed_wall_seconds",
                "terminal_gross_return_bps", "terminal_gross_observed"):
        assert projected[key].dtype == original[key].dtype
        assert projected[key].tobytes() == original[key].tobytes()
    np.testing.assert_array_equal(result["terminal_source_row"],
        original["terminal_source_row"].reshape(2, 3))
    left = _targets(result, index, _economics())
    right = direction_targets_from_observations(original, parent_rows=result["parent_rows"],
        decision_times_ns=result["decision_ns"], economics=_economics(),
        split_end_exclusive_ns=int(index.split_end_exclusive.value))
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].dtype == right[key].dtype and left[key].tobytes() == right[key].tobytes()
    assert all(value.tobytes() == before[name] for name, value in index.source.values.items())
    assert all(not value.flags.writeable for value in result.values() if isinstance(value, np.ndarray))


def test_unknown_interior_gap_keeps_exact_endpoint_price_observed():
    frame = _m1_frame(periods=25).drop(index=8).reset_index(drop=True)
    index, fill = _case(frame)
    result = _project(index, fill)
    original = quote.query_dynamic_m1_prefix(index, fill_surface=fill,
        query_entry_rows=np.zeros(3, np.int64), horizon_minutes=np.array([1, 5, 15], np.int64))
    assert result["observed"].tolist() == [[True, True, True]]
    assert original["path_observed"].tolist() == [[True, True], [False, False], [False, False]]
    assert result["terminal_source_row"][0, 1] != result["dynamic_entry_m1_row"][0] + 5
    assert result["terminal_observations"]["terminal_gross_return_bps"].tobytes() == original["terminal_gross_return_bps"].tobytes()


@pytest.mark.parametrize("fault", ["missing_endpoint", "exclusive_cutoff"])
def test_missing_exact_quote_and_cutoff_remain_masked_not_known_zero(fault):
    frame = _m1_frame(periods=25)
    cutoff = None
    if fault == "missing_endpoint":
        frame = frame.drop(index=6).reset_index(drop=True)
        expected = [[False, True, True]]
    else:
        cutoff = frame.time.iloc[20]
        frame.loc[20:, [column for column in frame if column != "time"]] = np.nan
        expected = [[True, True, False]]
    index, fill = _case(frame, cutoff=cutoff)
    result = _project(index, fill)
    assert result["observed"].tolist() == expected
    missing = ~result["observed"]
    assert np.all(result["sign"][missing] == 0)
    assert np.all(result["mid_move"][missing] == 0) and np.all(result["mid_return_bps"][missing] == 0)
    assert np.all(result["terminal_source_row"][missing] == -1)
    assert np.isnan(result["terminal_observations"]["terminal_gross_return_bps"][missing.reshape(-1)]).all()
    assert np.all(result["sign"][result["observed"]] == 1)


def test_missing_fill_is_not_replaced_by_a_later_opening_quote():
    index, fill = _case(_m1_frame(periods=25).drop(index=5).reset_index(drop=True))
    result = _project(index, fill)
    assert result["dynamic_entry_m1_row"].tolist() == [-1]
    assert not result["observed"].any() and not result["sign"].any()
    assert np.all(result["terminal_source_row"] == -1)


def test_decision_at_cutoff_is_censored_before_inspecting_its_fill_prices():
    frame = _m1_frame(periods=30)
    cutoff = frame.time.iloc[20]
    frame.loc[20:, [column for column in frame if column != "time"]] = np.nan
    index, fill = _case(frame, cutoff=cutoff, bars=4)
    fill = fill.copy()
    fill["entry_bid"] = fill["entry_bid"].astype(object)
    fill["entry_ask"] = fill["entry_ask"].astype(object)
    fill.loc[3, ["entry_bid", "entry_ask"]] = "future price must not be inspected"
    result = _project(index, fill, parents=(0, 3))
    assert result["observed"][0].tolist() == [True, True, False]
    assert not result["observed"][1].any() and not result["sign"][1].any()


@pytest.mark.parametrize("fault,error", [
    ("pointer", "POINTER_MISMATCH"), ("bar_clock", "BAR_CLOCK"),
    ("fill_quote", "FILL_QUOTE_MISMATCH"), ("parent_dtype", "PARENT_ROWS"),
])
def test_clock_pointer_and_original_quote_tampering_fail(fault, error):
    index, fill = _case()
    rows = np.array([0], np.int64)
    clocks = pd.DatetimeIndex(fill.iloc[rows].time).as_unit("ns").asi8.copy()
    pointers = fill.iloc[rows].entry_m1_row.to_numpy(np.int64, copy=True)
    if fault == "pointer":
        pointers += 1
    elif fault == "bar_clock":
        clocks += 60 * 10**9
    elif fault == "fill_quote":
        fill = fill.copy()
        fill.loc[0, "entry_bid"] += .01
    else:
        rows = rows.astype(np.int32)
    with pytest.raises(RuntimeError, match=error):
        price.project_direction_price_endpoints(index, fill_surface=fill, parent_rows=rows,
            bar_start_ns=clocks, dynamic_entry_m1_row=pointers)
