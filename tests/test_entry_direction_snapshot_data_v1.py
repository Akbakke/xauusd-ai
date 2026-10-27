"""Synthetic193 adapter tests only; no genuine data or native model."""
import copy
import hashlib
import json
import numpy as np
import pytest

from gx1.contracts import entry_direction_snapshot_data_v1 as data


def _ns(value):
    return int(np.datetime64(value, "ns").astype(np.int64))


def _economics():
    from gx1.contracts.unified_exit_economics_objective_v2 import SECONDS_PER_YEAR
    value = {"schema_version": "gx1_entry_observed_market_v1_economics",
        "cost_authority_sha256": "a" * 64, "coverage_start_utc": "2024-01-01T00:00:00Z",
        "coverage_end_utc_exclusive": "2026-04-01T00:00:00Z",
        "round_trip_execution_cost_bps": 4.0, "annual_financing_cost_bps": [350.0, 125.0],
        "seconds_per_year": SECONDS_PER_YEAR, "historical_cost_truth_qualified": False,
        "capital_hurdle_included": False, "risk_utility_included": False}
    value["economics_sha256"] = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return value


def _observations(horizons=(1, 5, 15), rows=2):
    query = np.tile(horizons, rows)
    decisions = _ns("2025-01-15T12:05:00") + np.arange(rows, dtype=np.int64) * 300 * 10**9
    count = len(query)
    gross = np.tile(np.array([[17.123456789, -9.876543211]], np.float64), (count, 1))
    return {"endpoint_kind": "terminal_open_inclusive", "side_order": ("long", "short"),
        "query_entry_rows": np.repeat(np.arange(rows, dtype=np.int64), len(horizons)),
        "horizon_minutes": query, "terminal_gross_observed": np.ones((count, 2), bool),
        "path_observed": np.zeros((count, 2), bool),
        "terminal_gross_return_bps": gross,
        "terminal_time_ns": np.repeat(decisions, len(horizons)) + query * 60 * 10**9,
        "elapsed_wall_seconds": query * 60,
        **{k: np.full((count, 2), np.nan) for k in
            ("mfe_bps", "mae_before_first_mfe_upper_bps", "mae_before_first_mfe_lower_bps")}}, decisions


def _targets(observations, decisions):
    return data.direction_targets_from_observations(observations,
        parent_rows=np.arange(len(decisions), dtype=np.int64), decision_times_ns=decisions,
        economics=_economics(), split_end_exclusive_ns=_ns("2026-01-01T00:00:00"))


def test_direction_target_projection_exact_native_f32_then_f64_contrast():
    from gx1.contracts.entry_observed_market_v1 import build_dynamic_m5_batch_targets
    from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_HORIZON_MINUTES
    obs, decisions = _observations()
    projected = _targets(obs, decisions)
    full, _ = _observations(DYNAMIC_M1_PREFIX_HORIZON_MINUTES)
    closed = {**full, "endpoint_kind": "closed_m1_prefix"}
    native = build_dynamic_m5_batch_targets(observations=full, closed_observations=closed,
        economics=_economics(), batch_rows=2, query_minutes=DYNAMIC_M1_PREFIX_HORIZON_MINUTES)
    from gx1.contracts.entry_observed_market_v1 import DYNAMIC_M5_MARKOUT_TARGET_KEY
    expected = native[DYNAMIC_M5_MARKOUT_TARGET_KEY][:, [0, 4, 14], :]
    assert np.array_equal(projected["terminal_net_bps"], expected)
    assert projected["terminal_net_bps"].dtype == np.float32
    assert projected["contrast_bps"].dtype == np.float64
    assert np.array_equal(projected["contrast_bps"],
        (expected[:, :, 0].astype(np.float64) - expected[:, :, 1].astype(np.float64)) / 2)
    assert projected["contrast_observed"].all()  # Path is entirely unobserved.


def test_direction_target_projection_keeps_own_horizon_masks():
    obs, decisions = _observations()
    obs["terminal_gross_observed"][1] = False
    obs["terminal_gross_return_bps"][1] = np.nan
    result = _targets(obs, decisions)
    assert result["contrast_observed"].tolist() == [[True, False, True], [True, True, True]]
    assert np.array_equal(result["terminal_net_bps"][0, 1], np.zeros(3, np.float32))


def test_direction_target_projection_rejects_cutoff_endpoint():
    obs, decisions = _observations()
    with pytest.raises(RuntimeError, match="CUTOFF"):
        data.direction_targets_from_observations(obs,
            parent_rows=np.arange(2, dtype=np.int64), decision_times_ns=decisions,
            economics=_economics(), split_end_exclusive_ns=int(obs["terminal_time_ns"].max()))


def test_direction_target_projection_rejects_asymmetric_terminal_mask():
    obs, decisions = _observations()
    obs["terminal_gross_observed"][0, 1] = False
    obs["terminal_gross_return_bps"][0, 1] = np.nan
    with pytest.raises(RuntimeError, match="TARGET_QUERY"):
        _targets(obs, decisions)


def _surface(timeframe="H1", warmup=0):
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4, MULTI_TF_SHIFT
    length = dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)[timeframe]
    duration = int(MULTI_TF_SHIFT[timeframe].value)
    times = _ns("2025-01-01T00:00:00") + np.arange(length + 5, dtype=np.int64) * duration
    values = np.broadcast_to(np.arange(len(times), dtype=np.float32)[:, None],
        (len(times), len(MULTI_TF_PER_BAR_FEATURES_V4))).copy()
    return data.prepare_direction_mtf_surface(timeframe=timeframe, timestamps_ns=times,
        features=values, warmup_rows=warmup, sequence_bars=length), times, length


def test_direction_mtf_current_uses_closed_bar_at_decision_availability():
    surface, times, length = _surface()
    bar_start = np.array([times[length] + 30 * 60 * 10**9], np.int64)
    values, selected, selected_clock = data.gather_direction_mtf_current(surface, bar_start)
    assert selected.tolist() == [length - 1]
    assert selected_clock.tolist() == [int(times[length - 1])]
    assert values[0, 0] == length - 1


def test_direction_mtf_current_retains_full_native_window_warmup():
    surface, times, length = _surface(warmup=2)
    with pytest.raises(RuntimeError, match="WARMUP"):
        data.gather_direction_mtf_current(surface, np.array([times[length]], np.int64))


def test_direction_mtf_current_rejects_off_grid_m5_clock():
    surface, times, length = _surface()
    with pytest.raises(RuntimeError, match="M5_CLOCK_GRID"):
        data.gather_direction_mtf_current(surface, np.array([times[length] + 1], np.int64))


def test_direction_sequence_evidence_requires_both_actual_file_refs():
    check_names = (
        "all_values_finite_float32", "every_emitted_timestamp_maps_exactly_once_to_feature_surface",
        "every_sequence_equals_exact_source_surface_history_bit_identical",
        "every_snapshot_equals_exact_source_surface_current_row_bit_identical",
        "source_feature_surface_hash_matches_split_manifest",
        "source_feature_surface_manifest_hash_matches_split_manifest",
        "source_feature_surface_schema_and_dimensions_exact", "source_timestamps_strictly_increasing")
    audit = {"schema_version": "entry_model_native_sequence_source_reconstruction_audit_v1",
        "decision": "PASS", "rows": 2, "sequence_shape": [2, 96, 2725], "snapshot_shape": [2, 2725],
        "checks": dict.fromkeys(check_names, True), "parquet_path": "/fixture/train.parquet",
        "parquet_sha256": "a" * 64, "manifest_path": "/fixture/train.json", "manifest_sha256": "b" * 64,
        "feature_surface_sha256": "c" * 64, "sequence_source_chain_sha256": "d" * 64}
    args = {"parquet_binding": {"path": audit["parquet_path"], "sha256": audit["parquet_sha256"]},
        "manifest_binding": {"path": audit["manifest_path"], "sha256": audit["manifest_sha256"]}, "rows": 2}
    assert data.require_direction_sequence_snapshot_evidence(audit, **args)["seq_last_equals_snap"]
    bad = copy.deepcopy(audit)
    bad["manifest_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="SEQUENCE_SNAPSHOT_EVIDENCE"):
        data.require_direction_sequence_snapshot_evidence(bad, **args)


def test_direction_raw_snapshot_parquet_roundtrip_and_wrong_clock(tmp_path):
    """Actual Arrow list offsets, column selection and two bounded read batches."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    clocks = _ns("2025-01-15T12:00:00") + np.arange(3, dtype=np.int64) * 300 * 10**9
    snap = np.arange(3 * 2725, dtype=np.float64).reshape(3, 2725) / 17.0
    context = -np.arange(3 * 2542, dtype=np.float64).reshape(3, 2542) / 31.0
    categories = np.array([[0], [2], [3]], dtype=np.int64)
    table = pa.Table.from_arrays([
        pa.array(clocks).cast(pa.timestamp("ns", tz="UTC")),
        pa.array(snap.tolist(), type=pa.list_(pa.float64())),
        pa.array(context.tolist(), type=pa.list_(pa.float64())),
        pa.array(categories.tolist(), type=pa.list_(pa.int64())),
    ], names=["time", "snap", "ctx_cont", "ctx_cat"])
    path = tmp_path / "synthetic_snapshot.parquet"
    pq.write_table(table, path, row_group_size=2)
    binding = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    batches = list(data.iter_direction_raw_snapshot_batches(
        binding, expected_times_ns=clocks, block_rows=2))
    assert [len(batch["parents"]) for batch in batches] == [2, 1]
    assert np.array_equal(np.concatenate([batch["parents"] for batch in batches]),
        np.arange(3, dtype=np.int64))
    assert np.array_equal(np.concatenate([batch["bar_start_ns"] for batch in batches]), clocks)
    for name, expected, dtype in (
            ("snap_x", snap.astype(np.float32), np.float32),
            ("ctx_cont", context.astype(np.float32), np.float32),
            ("ctx_cat", categories, np.int64)):
        actual = np.concatenate([batch["raw"][name] for batch in batches])
        assert actual.dtype == dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()

    wrong_clock = clocks.copy()
    wrong_clock[-1] += 60 * 10**9
    with pytest.raises(RuntimeError, match="PHYSICAL_PARENT_CLOCK"):
        list(data.iter_direction_raw_snapshot_batches(
            binding, expected_times_ns=wrong_clock, block_rows=2))


def _last_m1_fixture():
    minute = 60 * 10**9
    start = _ns("2025-01-15T12:00:00")
    times = start + np.arange(15, dtype=np.int64) * minute
    source = {"bar_start_ns": times, "available_ns": times + minute,
        "contract_index": np.full(15, 2, np.int16), "assigned": np.ones(15, bool),
        "ofi_contracts": np.arange(15, dtype=np.float64), "ofi_valid": np.ones(15, bool),
        "receive_clock_excluded": np.zeros(15, bool),
        "book_event_state_excluded": np.zeros(15, bool)}
    source["ofi_contracts"][[4, 9, 14]] = [0.0, 11.0, -7.0]
    args = {"parent_rows": np.array([17, 42, 91], np.int64),
        "decision_times_ns": start + np.array([5, 10, 15], np.int64) * minute,
        "current_m5_contract_index": np.full(3, 2, np.int16)}
    return source, args


def _readonly_last_m1(source, args):
    for value in (*source.values(), *args.values()):
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return source, args


def test_last_closed_m1_ofi_exact_join_keeps_observed_zero_and_isolated_readonly_outputs():
    source, args = _readonly_last_m1(*_last_m1_fixture())
    before = {name: value.tobytes() for name, value in source.items()}
    result = data.gather_direction_last_closed_m1_ofi(source, **args)
    assert result["parent_rows"].tolist() == [17, 42, 91]
    assert result["source_row_indices"].tolist() == [4, 9, 14]
    assert result["last_closed_m1_ofi_contracts"].tolist() == [0.0, 11.0, -7.0]
    assert result["last_closed_m1_ofi_observed"].tolist() == [True, True, True]
    assert all(value.tobytes() == before[name] for name, value in source.items())
    inputs = [*source.values(), *args.values()]
    for value in result.values():
        assert not value.flags.writeable
        assert not any(np.shares_memory(value, item) for item in inputs)
        with pytest.raises(ValueError):
            value[0] = 0
    source["ofi_contracts"].setflags(write=True)
    source["ofi_contracts"][9] = 999.0
    assert result["last_closed_m1_ofi_contracts"][1] == 11.0


@pytest.mark.parametrize("name,replacement", [
    ("assigned", False), ("ofi_valid", False),
    ("receive_clock_excluded", True), ("book_event_state_excluded", True),
    ("contract_index", -1)])
def test_last_closed_m1_ofi_invalid_exact_row_never_uses_older_observed_value(name, replacement):
    source, args = _last_m1_fixture()
    source[name][9] = replacement
    source["ofi_contracts"][8] = 12345.0
    _readonly_last_m1(source, args)
    result = data.gather_direction_last_closed_m1_ofi(source, **args)
    assert result["parent_rows"].tolist() == [17, 42, 91]
    assert result["source_row_indices"].tolist() == [4, 9, 14]
    assert result["last_closed_m1_ofi_observed"].tolist() == [True, False, True]
    assert result["last_closed_m1_ofi_contracts"].tolist() == [0.0, 0.0, -7.0]


def test_last_closed_m1_ofi_missing_exact_minute_keeps_parent_without_asof_fallback():
    source, args = _last_m1_fixture()
    source = {name: np.delete(value, 9) for name, value in source.items()}
    _readonly_last_m1(source, args)
    result = data.gather_direction_last_closed_m1_ofi(source, **args)
    assert result["source_row_indices"].tolist() == [4, -1, 13]
    assert result["last_closed_m1_ofi_observed"].tolist() == [True, False, True]
    assert result["last_closed_m1_ofi_contracts"].tolist() == [0.0, 0.0, -7.0]


@pytest.mark.parametrize("contract", [3, -1])
def test_last_closed_m1_ofi_optional_authoritative_m5_contract_comparison(contract):
    source, args = _last_m1_fixture()
    args["current_m5_contract_index"][1] = contract
    _readonly_last_m1(source, args)
    result = data.gather_direction_last_closed_m1_ofi(source, **args)
    assert result["last_closed_m1_ofi_observed"].tolist() == [True, False, True]
    result = data.gather_direction_last_closed_m1_ofi(source,
        **{**args, "current_m5_contract_index": None})
    assert result["last_closed_m1_ofi_observed"].all()


@pytest.mark.parametrize("field,delta,reason", [
    ("bar_start_ns", 1, "LAST_M1_GRID"),
    ("available_ns", 60 * 10**9, "LAST_M1_EXACT_AVAILABILITY"),
    ("available_ns", -60 * 10**9, "LAST_M1_EXACT_AVAILABILITY")])
def test_last_closed_m1_ofi_rejects_shifted_or_future_source_clocks(field, delta, reason):
    source, args = _last_m1_fixture()
    source[field][9] += delta
    _readonly_last_m1(source, args)
    with pytest.raises(RuntimeError, match=reason):
        data.gather_direction_last_closed_m1_ofi(source, **args)


def test_last_closed_m1_ofi_rejects_off_grid_parent_decision():
    source, args = _last_m1_fixture()
    args["decision_times_ns"][1] += 60 * 10**9
    _readonly_last_m1(source, args)
    with pytest.raises(RuntimeError, match="LAST_M1_DECISION_GRID"):
        data.gather_direction_last_closed_m1_ofi(source, **args)


def test_last_closed_m1_ofi_nonfinite_only_legal_when_unobserved():
    source, args = _last_m1_fixture()
    source["ofi_contracts"][9] = np.nan
    _readonly_last_m1(source, args)
    with pytest.raises(RuntimeError, match="LAST_M1_OBSERVED_OFI_NONFINITE"):
        data.gather_direction_last_closed_m1_ofi(source, **args)
    source["ofi_valid"].setflags(write=True)
    source["ofi_valid"][9] = False
    source["ofi_valid"].setflags(write=False)
    result = data.gather_direction_last_closed_m1_ofi(source, **args)
    assert not result["last_closed_m1_ofi_observed"][1]
    assert result["last_closed_m1_ofi_contracts"][1] == 0.0


@pytest.mark.parametrize("location,key", [
    ("source", "ofi_contracts"), ("args", "parent_rows"),
    ("args", "decision_times_ns"), ("args", "current_m5_contract_index")])
def test_last_closed_m1_ofi_rejects_mutable_inputs(location, key):
    source, args = _readonly_last_m1(*_last_m1_fixture())
    (source if location == "source" else args)[key].setflags(write=True)
    with pytest.raises(RuntimeError, match="LAST_M1_.*(IMMUTABLE|CONTRACT_HEADER)"):
        data.gather_direction_last_closed_m1_ofi(source, **args)
