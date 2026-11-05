"""Pure candidate-target algebra/clock/artifact checks; no native model."""
from itertools import product
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target

START = pd.Timestamp("2024-05-01T00:00:00Z").value
SOURCE = {"path": "/synthetic/m1.parquet", "sha256": "1" * 64}
ATR_SOURCE = {"path": "/synthetic/m5.parquet", "sha256": "2" * 64}


def frame_from_closes(prices, *, minutes=None):
    prices = np.asarray(prices, np.float64)
    opens = np.r_[prices[0], prices[:-1]]
    minute = np.arange(len(prices)) if minutes is None else np.asarray(minutes)
    return pd.DataFrame({
        "time": pd.to_datetime(START + minute * target.MINUTE_NS, utc=True),
        "open": opens, "high": np.maximum(opens, prices),
        "low": np.minimum(opens, prices), "close": prices,
    })


def atr_fixture(cutoff, *, values=None, available=None):
    available = np.asarray([START] if available is None else available, np.int64)
    values = np.asarray([2.0] if values is None else values, np.float64)
    return {"schema_version": target.SCHEMA_VERSION + "_causal_atr", "cutoff_ns": cutoff,
            "definition_sha256": target.dynamic_mtf_state_target_contract()["contract_sha256"],
            "surfaces": {tf: {"available_ns": available.copy(), "atr": values.copy()} for tf in target.TIMEFRAMES}}


def build(frame, *, cutoff=None, atr=None):
    cutoff = int(START + 120 * target.MINUTE_NS) if cutoff is None else cutoff
    return target.build_dynamic_mtf_state_series(frame, tf_atr=atr_fixture(cutoff) if atr is None else atr,
        source_ref=SOURCE, atr_source_ref=ATR_SOURCE, cutoff_ns=cutoff)


def build_with_known_pause(frame):
    from pathlib import Path
    from gx1.contracts import unified_exit_market_closure_authority_v1 as closure

    times = pd.DatetimeIndex(frame.time)
    gap_rows = np.flatnonzero(np.diff(times.asi8) != target.MINUTE_NS)
    assert len(gap_rows) == 1
    row = int(gap_rows[0])
    schedule = closure.seal_exact_market_schedule({
        "schema_version": closure.MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION,
        "decision": "PASS", "instrument": "XAU_USD", "timeframe": "M1",
        "coverage_start_utc": times[0].isoformat(),
        "coverage_end_utc_exclusive": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
        "interval_semantics": "left_closed_right_open_utc",
        "source_method": "externally_sourced_exact_xau_utc_closure_intervals_v1",
        "source_reference_sha256": "3" * 64, "test_data_used": False,
        "intervals": [{"kind": "holiday",
            "start_utc": (times[row] + pd.Timedelta(minutes=1)).isoformat(),
            "end_utc_exclusive": times[row + 1].isoformat(),
            "source_event_id": "synthetic_documented_pause"}],
    })
    authority = closure.build_market_closure_authority(
        m1_times=times, m1_source_path=Path(SOURCE["path"]), m1_source_sha256=SOURCE["sha256"],
        m1_source_manifest_path=Path("/synthetic/m1_manifest.json"), m1_source_manifest_sha256="4" * 64,
        exact_schedule=schedule, exact_schedule_path=Path("/synthetic/schedule.json"),
        exact_schedule_file_sha256="5" * 64)
    cutoff = int(times[-1].value + 2 * target.MINUTE_NS)
    return target.build_dynamic_mtf_state_series(frame, tf_atr=atr_fixture(cutoff),
        source_ref=SOURCE, atr_source_ref=ATR_SOURCE, cutoff_ns=cutoff,
        closure_authority=authority, exact_schedule=schedule)


def exact_point_path(path, *, direction, anchor, threshold=1.0):
    low = high = anchor
    for price in path:
        if direction == target.UP:
            high = max(high, price)
            if high - price >= threshold:
                direction, low = target.DOWN, price
        elif direction == target.DOWN:
            low = min(low, price)
            if price - low >= threshold:
                direction, high = target.UP, price
        else:
            if price - low >= threshold:
                direction, high = target.UP, price
            elif high - price >= threshold:
                direction, low = target.DOWN, price
            else:
                low, high = min(low, price), max(high, price)
    return direction, high if direction == target.UP else low


def test_contract_owns_approved_factor_clocks_codes_and_trial_domain():
    contract = target.dynamic_mtf_state_target_contract()
    assert target.require_dynamic_mtf_state_target_contract(contract) == contract
    assert target.STATE_NAMES == ("DOWN", "FLAT", "UP")
    assert (target.DOWN, target.FLAT, target.UP) == (0, 1, 2)
    assert contract["minimum_move_atr"] == 0.5 and contract["atr_period"] == 14
    assert target.default_trial_lead_minutes() == tuple(range(1, 1441))
    assert contract["local_window_m1_bars"] == dict(zip(target.TIMEFRAMES, (5, 15, 60, 240, 1440)))
    assert "wall_minute" in contract["query"]
    bad = dict(contract, minimum_move_atr=0.6)
    with pytest.raises(RuntimeError, match="TARGET_CONTRACT"):
        target.require_dynamic_mtf_state_target_contract(bad)


@pytest.mark.parametrize("initial", [target.UNKNOWN, target.DOWN, target.UP])
def test_ending_state_and_anchor_intervals_hold_for_enumerated_compatible_paths(initial):
    # Includes repeated extrema and extra reversals, not only two OHLC orders.
    grid = (0.0, 0.5, 1.0, 1.5, 2.0)
    checked = 0
    for close in grid:
        direction, lower, upper, ambiguous = target._step_direction(initial, 1.0, 1.0, 2.0, 0.0, close, 1.0)
        assert ambiguous
        for middle in product(grid, repeat=4):
            path = (1.0, *middle, close)
            if min(path) != 0.0 or max(path) != 2.0:
                continue
            true_direction, true_anchor = exact_point_path(path, direction=initial, anchor=1.0)
            if direction != target.UNKNOWN:
                assert true_direction == direction
                assert lower <= true_anchor <= upper
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("close, paths", [
    (0.5, ((1.0, 2.0, 0.0, 1.0, 0.5), (1.0, 0.0, 2.0, 0.5))),
    (1.5, ((1.0, 0.0, 2.0, 1.0, 1.5), (1.0, 2.0, 0.0, 1.5))),
])
def test_wide_interior_endpoint_is_unknown_despite_nearness_to_extreme(close, paths):
    # Same OHLC, opposite ending directions: extra internal reversals matter.
    actual = {exact_point_path(path, direction=target.UNKNOWN, anchor=1.0)[0] for path in paths}
    assert actual == {target.UP, target.DOWN}
    assert target._step_direction(target.UNKNOWN, 1.0, 1.0, 2.0, 0.0, close, 1.0) == (
        target.UNKNOWN, close, close, True)


def test_uncertain_prior_extreme_bounds_are_not_forced_to_direction():
    assert target._step_direction(target.UP, 1.5, 2.0, 1.4, 0.8, 1.3, 1.0)[0] == target.UNKNOWN
    assert target._step_direction(target.DOWN, 0.0, 0.5, 1.2, 0.6, 0.7, 1.0)[0] == target.UNKNOWN


def test_reversal_equality_and_ambiguous_ties_are_explicit():
    assert target._step_direction(target.UP, 2.0, 2.0, 1.5, 1.0, 1.2, 1.0)[0] == target.DOWN
    assert target._step_direction(target.UNKNOWN, 1.0, 1.0, 2.0, 0.0, 1.0, 1.0)[0] == target.UNKNOWN
    assert target._step_direction(target.UNKNOWN, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0) == (target.UP, 1.0, 1.0, True)


def test_flat_requires_full_local_span_and_uses_total_range():
    f = frame_from_closes([100.0] * 10)
    f["high"], f["low"] = 100.1, 99.9
    s = build(f).arrays
    assert not s["observed"][:4, 0].any()
    assert np.all(s["state"][4:, 0] == target.FLAT)
    assert not s["observed"][:, -1].any()  # no full D1 span, no invented FLAT
    metadata = build(f).metadata
    assert metadata["supplied_source_rows"] == len(f)
    assert "supplied_clock_sha256" in metadata and "full_source_rows" not in metadata
    f.loc[6, ["high", "low", "close"]] = [102.0, 98.0, 100.0]
    changed = build(f).arrays
    assert changed["state"][6, 0] == target.UNKNOWN
    assert not changed["observed"][6, 0]
    assert changed["reason"][6, 0] & target.INTRAMINUTE_ORDER_UNKNOWN


def test_ordered_reversals_do_not_need_crossing_the_initial_price():
    f = frame_from_closes([100.0, 100.4, 100.8, 101.2, 101.6, 101.2, 100.8, 100.4, 100.8, 101.2, 101.6])
    s = build(f).arrays
    observed = s["state"][:, 0][s["observed"][:, 0]]
    compressed = observed[np.r_[True, observed[1:] != observed[:-1]]]
    assert compressed.tolist() == [target.UP, target.DOWN, target.UP]
    assert f.loc[7, "close"] > f.loc[0, "open"]
    assert s["state"][7, 0] == target.DOWN


def test_gap_blocks_flat_but_not_forever_confirmed_new_direction():
    f = frame_from_closes([100.0] * 14, minutes=[*range(6), *range(7, 15)])
    s = build(f).arrays
    assert s["state"][4, 0] == target.FLAT
    assert not s["observed"][6:10, 0].any()
    assert s["state"][10, 0] == target.FLAT
    assert np.all((s["reason"][6:10, 0] & target.UNKNOWN_SOURCE_GAP) != 0)
    f.loc[7:, "close"] = np.arange(len(f) - 7) * 0.4 + 100.4
    f["open"] = np.r_[f.close.iloc[0], f.close.to_numpy()[:-1]]
    f["high"], f["low"] = np.maximum(f.open, f.close), np.minimum(f.open, f.close)
    moved = build(f).arrays
    assert moved["state"][9, 0] == target.UP
    assert not moved["path_observed"][9, 0]


@pytest.mark.parametrize("tf", target.TIMEFRAMES)
def test_active_window_requires_exactly_N_mature_observations(tf):
    window = target.LOCAL_WINDOW_M1_BARS[tf]
    col = target.TIMEFRAMES.index(tf)
    frame = frame_from_closes([100.0] * (window + 1))
    cutoff = int(START + window * target.MINUTE_NS)
    immature_last = build(frame, cutoff=cutoff).arrays
    assert len(immature_last["validity_ns"]) == window - 1
    assert np.all((immature_last["reason"][:, col] & target.LOCAL_SPAN_INCOMPLETE) != 0)
    assert not immature_last["observed"][:, col].any()
    mature_last = build(frame, cutoff=cutoff + target.MINUTE_NS).arrays
    assert len(mature_last["validity_ns"]) == window
    assert mature_last["validity_ns"][-1] == cutoff
    assert mature_last["maturity_ns"][-1, col] == cutoff
    assert mature_last["state"][-1, col] == target.FLAT
    assert mature_last["path_observed"][-1, col]
    assert not (mature_last["reason"][-1, col] & target.LOCAL_SPAN_INCOMPLETE)


def test_known_long_pause_matches_compressed_genuine_tape_and_does_not_age_activity():
    prices = np.full(1446, 100.0)
    prices[1438] = 102.0
    compact = frame_from_closes(prices)
    paused = frame_from_closes(prices, minutes=np.r_[np.arange(1440), np.arange(1440, 1446) + 3 * 1440])
    plain = build(compact, cutoff=int(START + 1448 * target.MINUTE_NS)).arrays
    resumed = build_with_known_pause(paused).arrays
    for name in ("state", "observed", "path_observed", "local_range", "reason", "threshold", "atr_available_ns"):
        assert np.array_equal(plain[name], resumed[name]), name
    # Two real bars after a long closure cannot erase the prior D1 movement.
    assert resumed["local_range"][1441, -1] == 2.0
    assert resumed["state"][1441, -1] != target.FLAT
    assert not resumed["path_observed"][1441, -1]
    # Last ambiguous M5 bar was row1439: it stays in the N-bar window through1443.
    assert not resumed["path_observed"][1443, 0]
    assert resumed["path_observed"][1444, 0]
    assert np.all(resumed["validity_ns"][1440:] - plain["validity_ns"][1440:] == 3 * 1440 * target.MINUTE_NS)


def test_known_long_pause_with_only_two_new_bars_cannot_supply_missing_D1_history():
    paused = frame_from_closes([100.0] * 3, minutes=[0, 3 * 1440, 3 * 1440 + 1])
    result = build_with_known_pause(paused).arrays
    assert not result["observed"][:, -1].any()
    assert not result["path_observed"][:, -1].any()
    assert np.all((result["reason"][:, -1] & target.LOCAL_SPAN_INCOMPLETE) != 0)


@pytest.mark.parametrize("tf", target.TIMEFRAMES)
def test_unknown_pause_requires_N_new_observations_for_flat_and_path(tf):
    window = target.LOCAL_WINDOW_M1_BARS[tf]
    col = target.TIMEFRAMES.index(tf)
    minutes = np.r_[np.arange(window), np.arange(window, 2 * window) + 3 * 1440]
    frame = frame_from_closes([100.0] * (2 * window), minutes=minutes)
    cutoff = int(START + (int(minutes[-1]) + 2) * target.MINUTE_NS)
    result = build(frame, cutoff=cutoff).arrays
    assert result["state"][window - 1, col] == target.FLAT
    assert not result["observed"][window:2 * window - 1, col].any()
    assert not result["path_observed"][window:2 * window - 1, col].any()
    assert np.all((result["reason"][window:2 * window - 1, col] & target.UNKNOWN_SOURCE_GAP) != 0)
    assert result["state"][2 * window - 1, col] == target.FLAT
    assert result["path_observed"][2 * window - 1, col]
    assert not (result["reason"][2 * window - 1, col] & target.UNKNOWN_SOURCE_GAP)


def test_known_pause_preserves_reopen_jump_in_N_observation_range():
    window = target.LOCAL_WINDOW_M1_BARS["M5"]
    prices = [100.0] * window + [103.0] * (window + 1)
    minutes = np.r_[np.arange(window), np.arange(window, 2 * window + 1) + 3 * 1440]
    frame = frame_from_closes(prices, minutes=minutes)
    # Real reopening OHLC is entirely103; the previous real close still matters.
    frame.loc[window, ["open", "high", "low", "close"]] = 103.0
    result = build_with_known_pause(frame).arrays
    assert result["local_range"][window, 0] == 3.0
    assert result["state"][window, 0] == target.UP
    assert result["local_range"][2 * window - 1, 0] == 3.0
    assert result["state"][2 * window - 1, 0] != target.FLAT
    assert result["local_range"][2 * window, 0] == 0.0
    assert result["state"][2 * window, 0] == target.FLAT


def test_future_edits_do_not_change_past_states_or_thresholds():
    f = frame_from_closes(np.linspace(100.0, 106.0, 20))
    cutoff = int(START + 30 * target.MINUTE_NS)
    atr = atr_fixture(cutoff, values=[2.0, 4.0], available=[START, START + 10 * target.MINUTE_NS])
    before = build(f, cutoff=cutoff, atr=atr)
    changed = f.copy()
    changed.loc[12:, ["open", "high", "low", "close"]] += 50.0
    after = build(changed, cutoff=cutoff, atr=atr)
    for name in ("state", "observed", "threshold", "local_range", "reason"):
        assert np.array_equal(before.arrays[name][:12], after.arrays[name][:12])
    assert np.all(before.arrays["threshold"][:9] == 1.0)
    assert np.all(before.arrays["threshold"][9:] == 2.0)
    assert np.all(before.arrays["atr_available_ns"] <= before.arrays["maturity_ns"])


def test_unavailable_atr_is_masked_not_filled_with_flat():
    f = frame_from_closes([100.0] * 6)
    cutoff = int(START + 10 * target.MINUTE_NS)
    atr = atr_fixture(cutoff, values=[np.nan])
    s = build(f, cutoff=cutoff, atr=atr).arrays
    assert not s["observed"].any()
    assert np.all(s["state"] == target.UNKNOWN)
    assert np.all((s["reason"] & target.ATR_UNAVAILABLE) != 0)


def test_query_exact_parent_clock_maturity_and_long_leads():
    f = frame_from_closes([100.0] * 100)
    series = build(f)
    parents = np.array([10, 20], np.int64)
    issues = START + np.array([5, 10], np.int64) * target.MINUTE_NS
    series = target.bind_dynamic_mtf_state_parents(series, parent_rows=parents, issue_ns=issues)
    q = target.query_dynamic_mtf_states(series, parent_rows=parents, issue_ns=issues,
        cutoff_ns=int(START + 90 * target.MINUTE_NS), lead_minutes=(1, 60, 80, 90))
    assert q["state"].shape == (2, 5, 4)
    assert q["observed"][0, 0, 2]  # lead80 is supported beyond the old60min cap
    assert not q["observed"][1, :, 2].any()  # equality at cutoff is immature
    assert np.all((q["reason"][1, :, 2] & target.IMMATURE) != 0)
    assert np.all(q["current_state"][:, 0] == target.FLAT)
    with pytest.raises(RuntimeError, match="QUERY_PARENT_IDENTITY"):
        target.query_dynamic_mtf_states(series, parent_rows=parents, issue_ns=issues + target.MINUTE_NS,
            cutoff_ns=int(START + 90 * target.MINUTE_NS), lead_minutes=(1,))


def test_immature_queries_hide_post_cutoff_source_reason_changes():
    parents = np.array([0], np.int64)
    issues = np.array([START + 5 * target.MINUTE_NS], np.int64)
    series = target.bind_dynamic_mtf_state_parents(build(frame_from_closes([100.0] * 20)),
        parent_rows=parents, issue_ns=issues)
    cutoff = int(START + 10 * target.MINUTE_NS)
    kwargs = dict(parent_rows=parents, issue_ns=issues, cutoff_ns=cutoff,
                  lead_minutes=(1, 4, 5, 6, 15, 30))
    before = target.query_dynamic_mtf_states(series, **kwargs)
    future = series.arrays["validity_ns"] >= cutoff
    for reason in (0, int(target.ATR_UNAVAILABLE | target.UNKNOWN_SOURCE_GAP
                          | target.INTRAMINUTE_ORDER_UNKNOWN | target.DIRECTION_UNRESOLVED)):
        changed_reason = series.arrays["reason"].copy()
        changed_reason[future] = reason
        changed_reason.setflags(write=False)
        changed_arrays = dict(series.arrays, reason=changed_reason)
        changed = target.DynamicMtfStateSeries(MappingProxyType(changed_arrays), series.metadata)
        after = target.query_dynamic_mtf_states(changed, **kwargs)
        for name in before:
            assert np.array_equal(before[name], after[name]), name
        # Includes cutoff equality, present future rows and absent future rows.
        assert np.all(after["reason"][:, :, 2:] == target.IMMATURE)
        assert np.all(after["state"][:, :, 2:] == target.UNKNOWN)
        assert not after["observed"][:, :, 2:].any()
        assert not after["path_observed"][:, :, 2:].any()


def test_missing_source_validity_stays_unknown():
    f = frame_from_closes([100.0] * 12, minutes=[*range(7), *range(8, 13)])
    series = target.bind_dynamic_mtf_state_parents(build(f), parent_rows=np.array([0], np.int64),
        issue_ns=np.array([START + 5 * target.MINUTE_NS], np.int64))
    q = target.query_dynamic_mtf_states(series, parent_rows=np.array([0], np.int64),
        issue_ns=np.array([START + 5 * target.MINUTE_NS], np.int64),
        cutoff_ns=int(START + 100 * target.MINUTE_NS), lead_minutes=(3,))
    assert not q["observed"].any()
    assert np.all(q["state"] == target.UNKNOWN)
    assert np.all(q["reason"] == target.QUERY_UNAVAILABLE)


def test_canonical_wilder_and_session_clock_are_reused_without_gap_reset():
    n = 20 * 24 * 12
    times = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC").delete(slice(100, 120))
    frame = pd.DataFrame({"time": times, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0})
    cutoff = pd.Timestamp("2024-01-22", tz="UTC").value
    result = target.build_causal_tf_atr(frame, cutoff_ns=int(cutoff))
    for tf, values in result["surfaces"].items():
        assert np.isnan(values["atr"][:13]).all()
        assert np.all(values["atr"][13:] == 2.0)
        offset = int(pd.Timedelta(target.MULTI_TF_RESAMPLE_ORIGIN_CONTRACT[tf]).value)
        span = int(target.MULTI_TF_SHIFT[tf].value)
        assert np.all((values["available_ns"] - offset) % span == 0)
    # The same supplied bars are closed under both cutoffs. Input-only math
    # stays identical beyond TEST without extending historical target access.
    for input_cutoff in (int(cutoff), int(target.TEST_BOUNDARY_NS) + target.MINUTE_NS):
        input_atr = target._build_input_tf_atr(frame, cutoff_ns=input_cutoff)
        assert set(input_atr) == set(result)
        assert input_atr["cutoff_ns"] == input_cutoff
        for name in ("schema_version", "definition_sha256"):
            assert input_atr[name] == result[name]
        assert tuple(input_atr["surfaces"]) == target.TIMEFRAMES
        for tf in target.TIMEFRAMES:
            for name in ("available_ns", "atr"):
                np.testing.assert_array_equal(input_atr["surfaces"][tf][name], result["surfaces"][tf][name])


def test_historical_atr_cutoff_rejects_test_before_input_math(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("historical TEST cutoff reached input ATR math")

    monkeypatch.setattr(target, "_build_input_tf_atr", forbidden)
    with pytest.raises(RuntimeError, match="DYNAMIC_MTF_STATE_CUTOFF"):
        target.build_causal_tf_atr(None, cutoff_ns=int(target.TEST_BOUNDARY_NS) + target.MINUTE_NS)


@pytest.mark.parametrize("cutoff", [
    None, True, False, 0, -target.MINUTE_NS, target.MINUTE_NS - 1,
    float(target.MINUTE_NS), np.int64(target.MINUTE_NS), str(target.MINUTE_NS),
])
def test_input_atr_cutoff_is_strict_before_frame_access(cutoff):
    with pytest.raises(RuntimeError, match="DYNAMIC_MTF_STATE_CUTOFF"):
        target._build_input_tf_atr(None, cutoff_ns=cutoff)


def test_immutable_artifact_roundtrip_and_source_cutoff_binding(tmp_path):
    series = target.bind_dynamic_mtf_state_parents(build(frame_from_closes([100.0] * 10)),
        parent_rows=np.array([0], np.int64), issue_ns=np.array([START + 5 * target.MINUTE_NS], np.int64))
    binding = target.save_dynamic_mtf_state_series(series, tmp_path / "series")
    expected = {"m1": SOURCE, "m5_atr": ATR_SOURCE}
    loaded = target.load_dynamic_mtf_state_series(binding, expected_source_refs=expected,
        expected_cutoff_ns=series.metadata["cutoff_ns"])
    assert all(not a.flags.writeable for a in loaded.arrays.values())
    assert all(np.array_equal(a, loaded.arrays[name]) for name, a in series.arrays.items())
    with pytest.raises(RuntimeError, match="ARTIFACT_BINDING"):
        target.read_dynamic_mtf_state_series_metadata(binding,
            expected_source_refs={**expected, "m1": {**SOURCE, "sha256": "3" * 64}},
            expected_cutoff_ns=series.metadata["cutoff_ns"])
    with pytest.raises(RuntimeError, match="OUTPUT_DIRECTORY"):
        target.save_dynamic_mtf_state_series(series, tmp_path / "series")


def test_historical_current_query_separates_state_path_and_missing_source():
    frame = frame_from_closes([100.0] * 19, minutes=[*range(14), *range(15, 20)])
    # The final OHLC extreme proves UP, while intraminute path remains unknown.
    frame.loc[9, ["high", "low", "close"]] = [102.0, 98.0, 102.0]
    parents = np.array([10, 20, 30], np.int64)
    issues = START + np.array([5, 10, 15], np.int64) * target.MINUTE_NS
    series = target.bind_dynamic_mtf_state_parents(build(frame), parent_rows=parents, issue_ns=issues)
    cutoff = int(START + 100 * target.MINUTE_NS)
    current = target.query_historical_dynamic_mtf_current_state(
        series, parent_rows=parents, issue_ns=issues, cutoff_ns=cutoff)
    assert set(current) == {"parent_rows", "issue_ns", "current_state", "current_observed",
                            "current_path_observed", "clock_basis"}
    assert current["clock_basis"] == "historical_event_time_only_not_online_delivery"
    assert current["current_state"][:, 0].tolist() == [target.FLAT, target.UP, target.UNKNOWN]
    assert current["current_observed"][:, 0].tolist() == [True, True, False]
    assert current["current_path_observed"][:, 0].tolist() == [True, False, False]
    assert not current["current_observed"][2].any()
    assert not current["current_path_observed"][2].any()
    full = target.query_dynamic_mtf_states(series, parent_rows=parents, issue_ns=issues,
                                            cutoff_ns=cutoff, lead_minutes=(1, 5))
    for key in ("current_state", "current_observed", "current_path_observed"):
        assert np.array_equal(current[key], full[key])
    # A consumer cannot mutate the source-bound tape by editing the result.
    current["current_state"][0, 0] = target.DOWN
    assert series.arrays["state"][series.arrays["issue_source_index"][0], 0] == target.FLAT


def test_historical_current_query_is_invariant_to_future_prices_and_atr():
    frame = frame_from_closes(np.linspace(100.0, 102.0, 20))
    cutoff = int(START + 40 * target.MINUTE_NS)
    issue = np.array([START + 10 * target.MINUTE_NS], np.int64)
    parents = np.array([7], np.int64)
    available = [START, START + 11 * target.MINUTE_NS]
    before = target.bind_dynamic_mtf_state_parents(
        build(frame, cutoff=cutoff, atr=atr_fixture(cutoff, values=[2.0, 4.0], available=available)),
        parent_rows=parents, issue_ns=issue)
    changed = frame.copy()
    changed.loc[10:, ["open", "high", "low", "close"]] += 50.0
    after = target.bind_dynamic_mtf_state_parents(
        build(changed, cutoff=cutoff, atr=atr_fixture(cutoff, values=[2.0, 9.0], available=available)),
        parent_rows=parents, issue_ns=issue)
    kwargs = {"parent_rows": parents, "issue_ns": issue, "cutoff_ns": cutoff}
    a = target.query_historical_dynamic_mtf_current_state(before, **kwargs)
    b = target.query_historical_dynamic_mtf_current_state(after, **kwargs)
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    assert not np.array_equal(before.arrays["threshold"][10:], after.arrays["threshold"][10:])


def test_historical_current_query_rejects_parent_and_future_clock_mismatch():
    parents = np.array([7], np.int64)
    issue = np.array([START + 5 * target.MINUTE_NS], np.int64)
    cutoff = int(START + 100 * target.MINUTE_NS)
    series = target.bind_dynamic_mtf_state_parents(build(frame_from_closes([100.0] * 12)),
                                                     parent_rows=parents, issue_ns=issue)
    kwargs = {"parent_rows": parents, "issue_ns": issue, "cutoff_ns": cutoff}
    with pytest.raises(RuntimeError, match="QUERY_PARENT_IDENTITY"):
        target.query_historical_dynamic_mtf_current_state(series, **{**kwargs, "issue_ns": issue + target.MINUTE_NS})
    arrays = dict(series.arrays)
    index = arrays["issue_source_index"].copy()
    index[0] += 1
    arrays["issue_source_index"] = index
    bad = target.DynamicMtfStateSeries(MappingProxyType(arrays), series.metadata)
    with pytest.raises(RuntimeError, match="CURRENT_SOURCE_VALIDITY"):
        target.query_historical_dynamic_mtf_current_state(bad, **kwargs)
    arrays = dict(series.arrays)
    atr_clock = arrays["atr_available_ns"].copy()
    atr_clock[series.arrays["issue_source_index"][0], 0] = issue[0] + target.MINUTE_NS
    arrays["atr_available_ns"] = atr_clock
    bad = target.DynamicMtfStateSeries(MappingProxyType(arrays), series.metadata)
    with pytest.raises(RuntimeError, match="CURRENT_ATR_CLOCK"):
        target.query_historical_dynamic_mtf_current_state(bad, **kwargs)


def _confirmed_age_fixture(states, *, minutes=None, path_observed=None):
    states = np.asarray(states, np.int8)
    if states.ndim == 1:
        states = np.repeat(states[:, None], len(target.TIMEFRAMES), axis=1)
    minute = np.arange(len(states)) if minutes is None else np.asarray(minutes)
    cutoff = int(START + (int(minute[-1]) + 2) * target.MINUTE_NS)
    series = build(frame_from_closes([100.] * len(states), minutes=minute), cutoff=cutoff)
    arrays = dict(series.arrays)
    arrays["state"] = states.copy()
    arrays["observed"] = states != target.UNKNOWN
    arrays["reason"] = np.where(arrays["observed"], 0, target.DIRECTION_UNRESOLVED).astype(np.uint8)
    arrays["path_observed"] = (arrays["observed"].copy() if path_observed is None
                               else np.asarray(path_observed, bool).copy())
    # Synthetic class/path evidence isolates the age contract from price algebra.
    return target.DynamicMtfStateSeries(MappingProxyType(arrays), series.metadata)


def _query_age(series, issue_minutes):
    parents = np.arange(len(issue_minutes), dtype=np.int64)
    issues = START + np.asarray(issue_minutes, np.int64) * target.MINUTE_NS
    bound = target.bind_dynamic_mtf_state_parents(series, parent_rows=parents, issue_ns=issues)
    return target.query_historical_dynamic_mtf_confirmed_run_age(bound,
        parent_rows=parents, issue_ns=issues, cutoff_ns=series.metadata["cutoff_ns"])


def test_confirmed_age_requires_observed_change_and_separate_tf_path_masks():
    states = np.repeat(np.array([1, 1, 2, 2, 2, 2, -1, 2, 2, 2,
                                 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], np.int8)[:, None], 5, axis=1)
    states[:, 2] = target.UP  # A left-truncated run has no confirmed beginning.
    states[:14, 3] = target.UP
    path = states != target.UNKNOWN
    path[12, 1] = False
    path[:13, 4] = False
    result = _query_age(_confirmed_age_fixture(states, path_observed=path), [5, 10, 15, 20])
    expected = np.array([[3, 3, 0, 0, 0], [0, 0, 0, 0, 0],
                         [5, 0, 0, 1, 0], [5, 5, 0, 5, 5]], np.int64)
    np.testing.assert_array_equal(result["current_confirmed_run_age_minutes"], expected)
    np.testing.assert_array_equal(result["current_confirmed_run_age_observed"], expected > 0)
    assert result["current_observed"][1].all()  # Known class need not mean known age.
    assert result["clock_basis"] == "historical_event_time_only_not_online_delivery"


@pytest.mark.parametrize("class_change_over_gap", (False, True))
def test_confirmed_age_resets_at_missing_m1_and_missing_m5_issue(class_change_over_gap):
    minutes = [*range(9), *range(10, 20)]
    states = [target.FLAT] * 2 + [target.UP] * (len(minutes) - 4) + [target.DOWN] * 2
    if class_change_over_gap:
        states[9:-2] = [target.FLAT] * len(states[9:-2])
    result = _query_age(_confirmed_age_fixture(states, minutes=minutes), [5, 10, 15, 20])
    np.testing.assert_array_equal(result["current_confirmed_run_age_minutes"][:, 0], [3, 0, 0, 2])
    assert not result["current_observed"][1].any()
    assert result["current_observed"][2].all()
    assert not result["current_confirmed_run_age_observed"][2].any()


def test_confirmed_age_reads_no_post_issue_state_or_clock_channels():
    states = [target.FLAT, target.UP] + [target.UP] * 18
    series = _confirmed_age_fixture(states)
    parents = np.array([7], np.int64)
    issue = START + np.array([10], np.int64) * target.MINUTE_NS
    bound = target.bind_dynamic_mtf_state_parents(series, parent_rows=parents, issue_ns=issue)
    kwargs = dict(parent_rows=parents, issue_ns=issue, cutoff_ns=series.metadata["cutoff_ns"])
    expected = target.query_historical_dynamic_mtf_confirmed_run_age(bound, **kwargs)
    # Any access to a later state/path/ATR/maturity row now raises or mis-shapes.
    limited = {key: (value[:10] if value.ndim == 2 else value) for key, value in bound.arrays.items()}
    limited = target.DynamicMtfStateSeries(MappingProxyType(limited), bound.metadata)
    actual = target.query_historical_dynamic_mtf_confirmed_run_age(limited, **kwargs)
    for key in expected:
        assert np.array_equal(actual[key], expected[key]), key
    assert actual["current_confirmed_run_age_minutes"][0, 0] == 9
    # Adding later issues to one vectorized query cannot change an earlier age.
    together = _query_age(series, [10, 20])
    np.testing.assert_array_equal(together["current_confirmed_run_age_minutes"][0],
                                  actual["current_confirmed_run_age_minutes"][0])


def test_confirmed_age_has_no_d1_duration_cap_and_rejects_early_atr_lookahead():
    series = _confirmed_age_fixture([target.FLAT] + [target.UP] * 1504)
    result = _query_age(series, [1505])
    assert (result["current_confirmed_run_age_minutes"] == 1504).all()
    parents = np.array([7], np.int64)
    issues = START + np.array([1505], np.int64) * target.MINUTE_NS
    bound = target.bind_dynamic_mtf_state_parents(series, parent_rows=parents, issue_ns=issues)
    arrays = dict(bound.arrays)
    atr_clock = arrays["atr_available_ns"].copy()
    atr_clock[1, 0] = arrays["validity_ns"][1] + target.MINUTE_NS
    arrays["atr_available_ns"] = atr_clock
    bad = target.DynamicMtfStateSeries(MappingProxyType(arrays), bound.metadata)
    with pytest.raises(RuntimeError, match="AGE_ATR_CLOCK"):
        target.query_historical_dynamic_mtf_confirmed_run_age(bad,
            parent_rows=parents, issue_ns=issues, cutoff_ns=series.metadata["cutoff_ns"])


@pytest.mark.parametrize("cutoff_minutes", (1, 8))
def test_train_queries_never_fetch_missing_or_post_cutoff_label_rows(cutoff_minutes):
    frame = frame_from_closes([100., 101., 102., 103.], minutes=[4, 5, 8, 9])
    issues = START + np.array([0] if cutoff_minutes == 1 else [5], np.int64) * target.MINUTE_NS
    parents = np.arange(len(issues), dtype=np.int64)
    bound = target.bind_dynamic_mtf_state_parents(build(frame), parent_rows=parents, issue_ns=issues)
    cutoff = START + cutoff_minutes * target.MINUTE_NS
    stop = np.searchsorted(bound.arrays["validity_ns"], cutoff)
    # Include the full clock for exact lookup, but give each label channel only
    # its physically permitted prefix. Any post-cutoff gather raises IndexError.
    guarded = {key: (value[:stop] if value.ndim == 2 else value)
               for key, value in bound.arrays.items()}
    prefix = target.DynamicMtfStateSeries(guarded, bound.metadata)
    leads = np.array([1, 2, 3, 4], dtype=np.int64)
    kwargs = dict(parent_rows=parents, issue_ns=issues, cutoff_ns=cutoff)
    data = target.query_dynamic_mtf_states(prefix, **kwargs, lead_minutes=leads, include_first_flat_activity=True)
    current = target.query_historical_dynamic_mtf_current_state(prefix, **kwargs)
    assert np.array_equal(data["current_state"], current["current_state"])
    immature = data["validity_ns"] >= cutoff
    assert np.all(data["reason"].transpose(0, 2, 1)[immature] == target.IMMATURE)
    assert np.all(data["state"].transpose(0, 2, 1)[immature] == target.UNKNOWN)
    if cutoff_minutes == 1:
        assert not current["current_observed"].any()
        assert not data["first_flat_activity_observed"].any()
    else:
        assert np.all(data["reason"][0, :, 1] == target.QUERY_UNAVAILABLE)
        assert np.all(data["reason"][0, :, 0] != target.QUERY_UNAVAILABLE)
