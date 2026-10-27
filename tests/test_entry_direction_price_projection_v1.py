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


def _project(index, fill, parents=(0,), **options):
    rows = np.asarray(parents, np.int64)
    return price.project_direction_price_endpoints(index, fill_surface=fill, parent_rows=rows,
        bar_start_ns=pd.DatetimeIndex(fill.iloc[rows].time).as_unit("ns").asi8.copy(),
        dynamic_entry_m1_row=fill.iloc[rows].entry_m1_row.to_numpy(np.int64, copy=True), **options)


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


@pytest.mark.parametrize("offset", [0, 37])
def test_dense_minutes_preserve_legacy_columns_and_quote_identity(offset):
    index, fill = _case(_m1_frame(periods=90), offset=offset)
    parents = (0, 2)
    old = _project(index, fill, parents)
    dense = _project(index, fill, parents, horizons_minutes=price.DENSE_HORIZONS_MINUTES)
    assert dense["schema_version"] == price.CURVE_SCHEMA_VERSION
    assert old["schema_version"] == price.SCHEMA_VERSION
    assert dense["horizons_minutes"] == tuple(range(1, 61))
    assert dense["observed"].shape == (2, 60) and dense["observed"].all()
    columns = [0, 4, 14]
    for key, value in old.items():
        if isinstance(value, np.ndarray):
            candidate = dense[key][:, columns] if value.ndim == 2 else dense[key]
            assert candidate.dtype == value.dtype and candidate.tobytes() == value.tobytes(), key
    original = quote.query_dynamic_m1_prefix(index, fill_surface=fill,
        query_entry_rows=np.repeat(np.array(parents, np.int64), 60),
        horizon_minutes=np.tile(np.arange(1, 61, dtype=np.int64), 2))
    obs = dense["terminal_observations"]
    for key in ("query_entry_rows", "horizon_minutes", "terminal_time_ns", "elapsed_wall_seconds",
                "terminal_gross_return_bps", "terminal_gross_observed"):
        assert obs[key].dtype == original[key].dtype
        assert obs[key].tobytes() == original[key].tobytes(), key
    select = (np.arange(2)[:, None] * 60 + np.array(columns)[None, :]).reshape(-1)
    for key, value in old["terminal_observations"].items():
        if isinstance(value, np.ndarray):
            assert obs[key][select].tobytes() == value.tobytes(), key
    np.testing.assert_array_equal(dense["terminal_source_row"],
        original["terminal_source_row"].reshape(2, 60))
    assert all(not x.flags.writeable for group in (dense, obs)
        for x in group.values() if isinstance(x, np.ndarray))


def _with_midpoints(frame, midpoints):
    frame = frame.copy()
    for side, spread in (("bid", -.01), ("ask", .01)):
        frame[side + "_open"] = midpoints + spread
        frame[side + "_high"] = midpoints + spread + .001
        frame[side + "_low"] = midpoints + spread - .001
    return frame


@pytest.mark.parametrize("direction", [1, -1])
def test_dense_signed_bps_rise_return_from_same_entry(direction):
    mid = np.full(80, 100.0)
    mid[10], mid[15], mid[20] = 100 + direction * .2, 100 + direction * .5, 100 + direction * .3
    index, fill = _case(_with_midpoints(_m1_frame(periods=80), mid))
    result = _project(index, fill, parents=(0, 1),
        horizons_minutes=price.DENSE_HORIZONS_MINUTES)
    np.testing.assert_allclose(result["mid_return_bps"][0, [4, 9, 14, 59]],
        direction * np.array([20., 50., 30., 0.]), rtol=0, atol=1e-10)
    np.testing.assert_array_equal(result["sign"][0, [4, 9, 14, 59]],
        direction * np.array([1, 1, 1, 0], np.int8))
    # A later M5 origin has its own denominator; it cannot inherit the old entry.
    assert result["mid_return_bps"][1, 4] == pytest.approx(
        10000 * (mid[15] - mid[10]) / mid[10], abs=1e-10)
    assert result["mid_return_bps"][1, 4] != pytest.approx(result["mid_return_bps"][0, 9])
    assert result["observed"][0, 59]  # Known return to entry, not missing/FLAT.


@pytest.mark.parametrize("horizons", [
    (), (0,), (61,), (1, 1), (5, 1), (True,), (1.0,),
    [1, 5, 15], None, (np.int64(10),), (1, "10"),
])
def test_horizon_domain_rejects_implicit_or_invalid_axes_before_source_access(horizons):
    with pytest.raises(RuntimeError, match="HORIZON_DOMAIN"):
        price.project_direction_price_endpoints(None, fill_surface=None,
            parent_rows=None, bar_start_ns=None, dynamic_entry_m1_row=None,
            horizons_minutes=horizons)


def test_ten_minute_target_ignores_intraminute_extremes():
    frame = _m1_frame(periods=80)
    index, fill = _case(frame)
    original = _project(index, fill, horizons_minutes=(10,))
    frame.loc[9, ["bid_high", "ask_high"]] += 1000
    index, fill = _case(frame)
    changed = _project(index, fill, horizons_minutes=(10,))
    for key in ("terminal_time_ns", "terminal_source_row", "observed", "mid_move", "mid_return_bps", "sign"):
        assert changed[key].tobytes() == original[key].tobytes()
    assert changed["terminal_time_ns"][0, 0] == changed["decision_ns"][0] + 10 * 60 * 10**9


def test_dense_observed_zero_missing_minute_and_cutoff_are_distinct():
    frame = _with_midpoints(_m1_frame(periods=80), np.full(80, 100.0))
    cutoff = frame.time.iloc[35]  # Exactly D+30 is censored.
    frame.loc[35:, [x for x in frame if x != "time"]] = np.nan
    frame = frame.drop(index=15).reset_index(drop=True)  # D+10 quote is absent.
    index, fill = _case(frame, cutoff=cutoff)
    result = _project(index, fill, horizons_minutes=price.DENSE_HORIZONS_MINUTES)
    assert result["observed"][0, 8] and result["sign"][0, 8] == 0
    assert not result["observed"][0, 9] and result["terminal_source_row"][0, 9] == -1
    assert result["terminal_time_ns"][0, 9] < cutoff.value
    assert result["observed"][0, 10]  # A missing minute does not shift later targets.
    assert not result["observed"][0, 29:].any()
    assert result["terminal_time_ns"][0, 29] == cutoff.value
    assert np.all(result["mid_return_bps"][~result["observed"]] == 0)
    assert np.all(result["terminal_source_row"][~result["observed"]] == -1)
    assert np.isnan(result["terminal_observations"]["terminal_gross_return_bps"]
        [~result["observed"].reshape(-1)]).all()


def test_nonlegacy_three_column_axis_cannot_enter_old_cost_adapter():
    index, fill = _case(_m1_frame(periods=80))
    result = _project(index, fill, horizons_minutes=(1, 10, 60))
    assert result["schema_version"] == price.CURVE_SCHEMA_VERSION
    assert result["mid_return_bps"].shape == (1, 3)
    with pytest.raises(RuntimeError, match="TARGET_QUERY"):
        _targets(result, index, _economics())


def test_dense_missing_entry_is_not_replaced_by_a_later_quote():
    index, fill = _case(_m1_frame(periods=80).drop(index=5).reset_index(drop=True))
    result = _project(index, fill, horizons_minutes=price.DENSE_HORIZONS_MINUTES)
    assert result["dynamic_entry_m1_row"].tolist() == [-1]
    assert not result["observed"].any()
    assert np.all(result["terminal_source_row"] == -1)
    assert np.all(result["mid_return_bps"] == 0) and np.all(result["sign"] == 0)


def test_dense_cutoff_entry_never_inspects_future_fill_prices():
    frame = _m1_frame(periods=80)
    cutoff = frame.time.iloc[20]
    frame.loc[20:, [x for x in frame if x != "time"]] = np.nan
    index, fill = _case(frame, cutoff=cutoff, bars=4)
    fill = fill.copy()
    for key in ("entry_bid", "entry_ask"):
        fill[key] = fill[key].astype(object)
        fill.loc[3, key] = "future quote must not be inspected"
    result = _project(index, fill, parents=(0, 3),
        horizons_minutes=price.DENSE_HORIZONS_MINUTES)
    assert result["observed"][0, :14].all()
    assert not result["observed"][0, 14:].any()
    assert not result["observed"][1].any()


def test_clock_overflow_guard_uses_the_requested_maximum_horizon():
    index, fill = _case()
    bar_ns = 300 * 10**9
    near_end = ((np.iinfo(np.int64).max - bar_ns - 30 * 60 * 10**9) // bar_ns) * bar_ns
    kwargs = dict(fill_surface=fill, parent_rows=np.array([0], np.int64),
        bar_start_ns=np.array([near_end], np.int64),
        dynamic_entry_m1_row=fill.iloc[[0]].entry_m1_row.to_numpy(np.int64, copy=True))
    with pytest.raises(RuntimeError, match="BAR_CLOCK"):
        price.project_direction_price_endpoints(index, **kwargs, horizons_minutes=(60,))
    with pytest.raises(RuntimeError, match="FILL_CLOCK_MISMATCH"):
        price.project_direction_price_endpoints(index, **kwargs)


def test_explicit_legacy_horizons_preserve_default_schema_and_values():
    index, fill = _case()
    left, right = _project(index, fill), _project(index, fill, horizons_minutes=(1, 5, 15))
    assert left.keys() == right.keys() and left["schema_version"] == price.SCHEMA_VERSION
    for group_left, group_right in ((left, right), (left["terminal_observations"], right["terminal_observations"])):
        for key, value in group_left.items():
            if isinstance(value, np.ndarray):
                assert value.dtype == group_right[key].dtype and value.tobytes() == group_right[key].tobytes()
            elif not isinstance(value, dict):
                assert value == group_right[key]


@pytest.mark.parametrize("horizons", [price.REPORT_HORIZONS_MINUTES, (10,)])
def test_report_anchors_and_ten_only_use_elapsed_minutes_and_price_bps(horizons):
    assert price.REPORT_HORIZONS_MINUTES == (1, 5, 10, 15, 30, 60)
    index, fill = _case(_m1_frame(periods=80))
    result = _project(index, fill, horizons_minutes=horizons)
    assert result["horizons_minutes"] == horizons and result["observed"].all()
    np.testing.assert_array_equal(result["elapsed_wall_seconds"], [np.array(horizons) * 60])
    np.testing.assert_array_equal(result["terminal_time_ns"],
        result["decision_ns"][:, None] + np.array(horizons, np.int64)[None, :] * 60 * 10**9)
    # This synthetic fixture increases its midpoint by exactly 1 price unit/minute.
    np.testing.assert_allclose(result["mid_return_bps"][0],
        10000 * np.array(horizons, np.float64) / 105.05, rtol=0, atol=1e-10)


def _midpoint_audit_case(frame=None, *, cutoff=None):
    index, fill = _case(frame, cutoff=cutoff, bars=2)
    projected = _project(index, fill, parents=(0, 1))
    names = ("parent_rows", "bar_start_ns", "dynamic_entry_m1_row", "decision_ns",
        "terminal_time_ns", "elapsed_wall_seconds", "terminal_source_row",
        "observed", "mid_move", "mid_return_bps", "sign")
    saved = {name: projected[name].copy() for name in names}

    def quotes(rows, observed):
        result = {"time_ns": np.zeros(rows.shape, np.int64), "source_row": rows.copy(),
            "observed": observed.copy()}
        local = rows[observed] - index.source_row_offset
        result["time_ns"][observed] = index.source.times.asi8[local]
        for name in ("bid_open", "ask_open"):
            result[name] = np.zeros(rows.shape, np.float64)
            result[name][observed] = index.source.values[name][local]
        result["open"] = (result["bid_open"] + result["ask_open"]) / 2.0
        return result

    entry = quotes(saved["dynamic_entry_m1_row"], np.ones(2, np.bool_))
    terminal = quotes(saved["terminal_source_row"], saved["observed"])
    return saved, entry, terminal, int(index.split_end_exclusive.value)


def _compare_midpoint_case(case):
    saved, entry, terminal, cutoff = case
    return price.compare_direction_native_midpoint(saved, entry_quotes=entry,
        terminal_quotes=terminal, cutoff_ns=cutoff)


def test_native_midpoint_audit_reproduces_legacy_bits_and_leaves_inputs_unchanged():
    case = _midpoint_audit_case()
    before = [{name: value.tobytes() for name, value in group.items()} for group in case[:3]]
    result = _compare_midpoint_case(case)
    assert all(result["legacy_bit_parity"].values())
    for output, original in (("native_mid_move", "mid_move"),
            ("native_mid_return_bps", "mid_return_bps"), ("native_sign", "sign")):
        assert result["arrays"][output].tobytes() == case[0][original].tobytes()
    assert all(not value.flags.writeable for value in result["arrays"].values())
    for report in result["per_horizon"]:
        assert report["observed"] == 2
        assert report["sign_confusion_legacy_by_native"] == [[0, 0, 0], [0, 0, 0], [0, 0, 2]]
        assert report["return_bps_difference"]["maximum_absolute_difference"] == 0
    assert before == [{name: value.tobytes() for name, value in group.items()} for group in case[:3]]


def test_native_midpoint_audit_reports_true_direction_disagreement_without_rewriting_legacy():
    case = _midpoint_audit_case()
    case[2]["open"][0, 0] = case[1]["open"][0] - 1.0
    result = _compare_midpoint_case(case)
    first = result["per_horizon"][0]
    assert first["sign_confusion_legacy_by_native"] == [[0, 0, 0], [0, 0, 0], [1, 0, 1]]
    assert first["nonzero_sign_flips"] == 1
    assert first["terminal_price_difference"]["numerically_equal"] == 1
    assert first["return_bps_difference"]["maximum_absolute_difference"] > 0
    assert case[0]["sign"][0, 0] == 1 and result["arrays"]["native_sign"][0, 0] == -1


def test_native_midpoint_audit_distinguishes_zero_missing_and_cutoff():
    frame = _with_midpoints(_m1_frame(periods=45), np.full(45, 100.0))
    cutoff = frame.time.iloc[20]
    frame = frame.drop(index=6).reset_index(drop=True)
    case = _midpoint_audit_case(frame, cutoff=cutoff)
    case[2]["open"][0, 1] += 1.0
    result = _compare_midpoint_case(case)
    first, fifth, fifteenth = result["per_horizon"]
    assert first["observed"] == 1 and first["mature_missing"] == 1
    assert first["sign_confusion_legacy_by_native"][1][1] == 1
    assert fifth["legacy_zero_to_native_nonzero"] == 1
    assert fifteenth["observed"] == 0 and fifteenth["cutoff_censored"] == 2
    assert fifteenth["return_bps_difference"]["count"] == 0
    assert fifteenth["return_bps_difference"]["mean_signed_difference"] is None
    assert result["arrays"]["native_mid_return_bps"][0, 0] == 0
    assert not result["arrays"]["observed"][0, 0]
    assert case[0]["observed"][1, 0] and result["arrays"]["native_sign"][1, 0] == 0


@pytest.mark.parametrize("name", ["mid_move", "mid_return_bps", "sign"])
def test_native_midpoint_audit_blocks_legacy_mismatch_before_alternative_reporting(name):
    case = _midpoint_audit_case()
    case[0][name][0, 0] = -1 if name == "sign" else case[0][name][0, 0] + 1
    with pytest.raises(RuntimeError, match="MIDPOINT_LEGACY_BIT_PARITY"):
        _compare_midpoint_case(case)


@pytest.mark.parametrize("fault,error", [
    ("entry_clock", "MIDPOINT_ENTRY_JOIN"),
    ("terminal_clock", "MIDPOINT_TERMINAL_JOIN"),
    ("terminal_source", "MIDPOINT_TERMINAL_JOIN"),
    ("saved_clock", "MIDPOINT_TERMINAL_CLOCK"),
    ("entry_nan", "MIDPOINT_ENTRY_PRICE_OR_STORAGE"),
    ("terminal_nan", "MIDPOINT_TERMINAL_PRICE_OR_STORAGE"),
    ("price_dtype", "MIDPOINT_TERMINAL_HEADER:open"),
    ("price_shape", "MIDPOINT_TERMINAL_HEADER:open"),
    ("saved_shape", "MIDPOINT_SAVED_HEADER:observed"),
    ("cutoff", "MIDPOINT_SAVED_SOURCE_OR_CUTOFF"),
])
def test_native_midpoint_audit_rejects_shifted_malformed_or_unobserved_source(fault, error):
    saved, entry, terminal, cutoff = _midpoint_audit_case()
    if fault == "entry_clock":
        entry["time_ns"][0] += 60 * 10**9
    elif fault == "terminal_clock":
        terminal["time_ns"][0, 0] += 60 * 10**9
    elif fault == "terminal_source":
        terminal["source_row"][0, 0] += 1
    elif fault == "saved_clock":
        saved["terminal_time_ns"][0, 0] += 60 * 10**9
    elif fault == "entry_nan":
        entry["open"][0] = np.nan
    elif fault == "terminal_nan":
        terminal["open"][0, 0] = np.nan
    elif fault == "price_dtype":
        terminal["open"] = terminal["open"].astype(np.float32)
    elif fault == "price_shape":
        terminal["open"] = terminal["open"][:, :2]
    elif fault == "saved_shape":
        saved["observed"] = saved["observed"][:, 0]
    else:
        cutoff = int(saved["decision_ns"][-1] + 60 * 10**9)
    with pytest.raises(RuntimeError, match=error):
        price.compare_direction_native_midpoint(saved, entry_quotes=entry,
            terminal_quotes=terminal, cutoff_ns=cutoff)


@pytest.mark.parametrize("fault", ["nan", "nonzero", "observed_mask"])
def test_native_midpoint_audit_rejects_fabricated_missing_storage(fault):
    case = _midpoint_audit_case(_m1_frame(periods=45).drop(index=6).reset_index(drop=True))
    if fault == "observed_mask":
        case[2]["observed"][0, 0] = True
    else:
        case[2]["open"][0, 0] = np.nan if fault == "nan" else 1.0
    with pytest.raises(RuntimeError, match="MIDPOINT_TERMINAL_PRICE_OR_STORAGE"):
        _compare_midpoint_case(case)
