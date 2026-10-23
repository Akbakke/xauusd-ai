"""Focused metadata/source tests for the new204 price-coordinate authority."""
import json
import pytest

from gx1.contracts import gc_price_coordinate_contract_v1 as contract


@pytest.fixture(scope="module")
def catalog():
    return contract.build_gc_price_coordinate_source_contract()


def test_complete_immutable_registry_fields_and_masks_retained(catalog):
    registry = json.loads((contract._REPO / contract.REGISTRY_PATH).read_text())
    assert catalog["field_count"] == 1205
    assert catalog["source_column_count"] == 318
    assert catalog["coordinate_kind_counts"] == {
        "absolute_price_level_usd": 232, "absolute_price_level_ticks": 10, "unchanged": 963}
    for index, (actual, original) in enumerate(zip(catalog["fields"], registry["fields"], strict=True)):
        assert actual["registry_index"] == index
        for name in ("name", "availability_field", "table", "column", "owner"):
            assert actual[name] == original[name]
    assert catalog["numerical_transformation_implemented"] is False
    assert catalog["execution_authorized"] is False


@pytest.mark.parametrize("column,kind", [
    ("open_usd", "absolute_price_level_usd"),
    ("observed_proxy_ob_bear_high", "absolute_price_level_usd"),
    ("observed_buy_known_poc_usd", "absolute_price_level_usd"),
    ("observed_profile_session_observed_val_usd", "absolute_price_level_usd"),
    ("observed_book_opening_mid_tick", "absolute_price_level_ticks"),
    ("observed_book_closing_mid_tick", "absolute_price_level_ticks"),
    ("observed_book_mid_response_ticks", "unchanged"),
    ("rolling15_val_distance_ticks", "unchanged"),
    ("observed_event_spread_state_min_ticks", "unchanged"),
    ("observed_price_levels", "unchanged"),
    ("observed_proxy_ob_bull_origin_volume_contracts", "unchanged"),
    ("spot_basis_bps", "unchanged"),
    ("atr_usd", "unchanged"), ("range_usd", "unchanged"),
    ("new_unproved_price_usd", "unchanged"),
])
def test_level_identity_is_closed_producer_set_not_units_or_name(column, kind):
    assert contract._coordinate_kind(column) == kind


def test_price_producer_sets_are_disjoint_and_complete(catalog):
    groups = [contract.CANDLE_LEVEL_COLUMNS, contract.OB_LEVEL_COLUMNS,
        contract.HISTOGRAM_LEVEL_COLUMNS, contract.DYNAMIC_PROFILE_LEVEL_COLUMNS,
        contract.TICK_LEVEL_COLUMNS]
    assert [len(group) for group in groups] == [8, 8, 16, 32, 2]
    assert sum(map(len, groups)) == len(set().union(*groups)) == 66
    for item in catalog["fields"]:
        if item["coordinate_kind"] != "unchanged":
            assert item["price_producer_evidence"] in catalog["price_producer_evidence"]
        else:
            assert item["price_producer_evidence"] is None


def test_unit_owner_reuse_and_price_delta_mask_distinction(catalog):
    fields = {item["name"]: item for item in catalog["fields"]}
    assert fields["gc_m5_observed_open_usd"]["source_unit"] == "USD"
    assert fields["gc_m5_observed_book_mid_response_ticks"]["source_unit"] == "ticks"
    assert fields["gc_m5_observed_valid"]["source_unit"] == "binary"
    assert fields["gc_m5_observed_valid"]["coordinate_kind"] == "unchanged"
    assert fields["gc_m5_observed_valid"]["availability_field"] != "gc_m5_observed_valid"
    assert all(item["unit_evidence"] for item in catalog["fields"])
    assert catalog["unit_owner_functions"] == list(contract.UNIT_OWNER_FUNCTIONS)


def test_anchor_priority_keeps_all_levels_once_without_training_rank(catalog):
    levels = [item["name"] for item in catalog["fields"] if item["coordinate_kind"] != "unchanged"]
    priority = catalog["anchor_source_priority"]
    assert len(priority) == len(set(priority)) == 242
    assert priority == ["gc_book_close_usd"] + [name for name in levels if name != "gc_book_close_usd"]
    assert catalog["anchor_unobserved_source_id"] == -1
    assert "provenance, not model input" in catalog["anchor_source_id_semantics"]


def test_projected_profile_contract_is_from_actual_m5_snapshot(catalog):
    fields = {item["name"]: item for item in catalog["fields"]}
    profile = fields["gc_h4_observed_profile_rolling60_poc_usd"]
    assert profile["source_clock_column"] == "observed_profile_snapshot_available_ns"
    assert profile["contract_provenance"] == {
        "mode": "exact_m5_profile_snapshot_join", "source_table": "TRADE_FEATURES_M5.parquet",
        "source_clock_column": "available_ns", "contract_column": "contract_index",
        "join_clock_column": "observed_profile_snapshot_available_ns", "outer_tf_contract_is_authority": False}
    candle = fields["gc_h4_observed_close_usd"]
    assert candle["contract_provenance"]["source_table"] == "TRADE_FEATURES_H4.parquet"
    assert candle["contract_provenance"]["mode"] == "selected_source_table_row"
    requirements = catalog["future_adapter_requirements"]
    assert "state availability" in requirements["clock_meaning"]
    assert requirements["row_contract_metadata_required_for_numeric_transform"] is False
    assert requirements["same_contract_or_tradable_support_resistance_claimed"] is False
    assert "unknown" in requirements["per_row_contract_and_absolute_source_clock"]


def test_tick_units_link_uniform_roll_scale_and_allow_half_midpoints(catalog):
    conversion = catalog["tick_to_usd"]
    assert conversion["price_scale"] == 10**9
    assert conversion["formula"] == "level_ticks * tick_size / PRICE_SCALE"
    assert "metadata_identity" in conversion["tick_size_source"]
    assert "half ticks" in conversion["midpoint_grid"]


def test_unit_conflict_fails_closed():
    with pytest.raises(RuntimeError, match="UNIT_CONFLICT:x"):
        contract._merge_units({"x": "USD"}, {"x": ["first"]}, {"x": "ticks"}, "second")


def test_unknown_registry_column_and_mismatched_level_unit_fail_closed():
    field = {"name": "g", "availability_field": "g_observed", "table": "t",
        "column": "unknown", "owner": "family"}
    with pytest.raises(RuntimeError, match="UNIT_UNDECLARED:unknown"):
        contract._compile_fields([field], {}, {})
    field["column"] = "open_usd"
    with pytest.raises(RuntimeError, match="USD_UNIT:open_usd"):
        contract._compile_fields([field], {"open_usd": "ticks"}, {"open_usd": ["source"]})


def test_duplicate_value_or_availability_identity_fails_closed():
    field = {"name": "g", "availability_field": "g_observed", "table": "t",
        "column": "open_usd", "owner": "family"}
    with pytest.raises(RuntimeError, match="DUPLICATE_FIELD_OR_MASK"):
        contract._compile_fields([field, dict(field)], {"open_usd": "USD"}, {"open_usd": ["source"]})


def test_changed_bound_source_is_rejected_before_owner_import(monkeypatch):
    monkeypatch.setitem(contract.SOURCE_SHA256, contract.UNIT_OWNER_PATH, "0" * 64)
    with pytest.raises(RuntimeError, match="SOURCE_SHA256"):
        contract.build_gc_price_coordinate_source_contract()


def test_usd_units_cannot_promote_an_unproved_level():
    field = {"name": "g", "availability_field": "g_observed", "table": "t",
        "column": "atr_usd", "owner": "family"}
    record = contract._compile_fields([field], {"atr_usd": "USD"}, {"atr_usd": ["source"]})[0]
    assert record["coordinate_kind"] == "unchanged"
    assert record["price_producer_evidence"] is None
    assert "contract_provenance" not in record


def test_existing_decision_age_transform_is_explicit_in_map(catalog):
    fields = {item["name"]: item for item in catalog["fields"]}
    for name in ("gc_m5_observed_spot_basis_age_minutes",
            "gc_h4_observed_profile_snapshot_age_minutes"):
        assert fields[name]["source_transform"] == "decision_age_minutes"
        assert fields[name]["source_unit"] == "minutes"
        assert fields[name]["coordinate_kind"] == "unchanged"
