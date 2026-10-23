"""Synthetic mechanics for205, never a market or fitted-reference test."""
from copy import deepcopy
import numpy as np
import pytest

from gx1.contracts import gc_relative_price_coordinates_v1 as geometry
from gx1.contracts.gc_price_coordinate_contract_v1 import build_gc_price_coordinate_source_contract


@pytest.fixture(scope="module")
def source_contract():
    return build_gc_price_coordinate_source_contract()


@pytest.fixture(scope="module")
def names(source_contract):
    gc = ["ctx_cont." + item[key] for item in source_contract["fields"]
        for key in ("name", "availability_field")]
    return ["fixture.non_gc." + str(i) for i in range(2622 - len(gc))] + gc


@pytest.fixture(scope="module")
def layout(source_contract, names):
    return geometry.compile_gc_relative_price_layout(names, source_contract=source_contract)


def _set_level(matrix, layout, row, source_name, value):
    column = layout["anchor_source_priority"].index(source_name)
    matrix[row, layout["level_value_indices"][column]] = np.float32(value)
    matrix[row, layout["level_mask_indices"][column]] = 1
    return column


def _bits(value):
    return np.ascontiguousarray(value).view(np.uint32)


def test_layout_complete_pair_identity_and_fixed_priority(layout, source_contract, names):
    assert layout["local_field_names"] == names
    assert len(layout["registry_value_indices"]) == len(layout["registry_mask_indices"]) == 1205
    assert len(layout["level_value_indices"]) == 242
    assert len(layout["unchanged_indices"]) == 2622 - 242
    assert layout["level_units"].count("USD") == 232
    assert layout["level_units"].count("ticks") == 10
    assert layout["anchor_source_priority"] == source_contract["anchor_source_priority"]
    assert layout["max_block_rows"] == geometry.MAX_BLOCK_ROWS == 1024
    assert layout["anchor_source_id_is_model_input"] is False


def test_complete_all_level_log_projection_and_untouched_bits(layout):
    raw = np.zeros((1, 2622), np.float32)
    prices = np.linspace(1981, 2021, 242, dtype=np.float64)
    factors = np.asarray([0.1 if unit == "ticks" else 1 for unit in layout["level_units"]])
    raw[0, layout["level_value_indices"]] = (prices / factors).astype(np.float32)
    raw[0, layout["level_mask_indices"]] = 1
    raw[0, 0] = np.float32(-0.0)
    before = raw.copy()
    raw.setflags(write=False)
    result = geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)
    p = raw[0, layout["level_value_indices"]].astype(np.float64) * factors
    expected = np.log(p / p[0]).astype(np.float32)
    np.testing.assert_array_equal(_bits(result["local_values"][0, layout["level_value_indices"]]), _bits(expected))
    np.testing.assert_array_equal(_bits(result["local_values"][:, layout["unchanged_indices"]]),
        _bits(before[:, layout["unchanged_indices"]]))
    np.testing.assert_array_equal(_bits(raw), _bits(before))
    assert raw.flags.writeable is False
    assert result["anchor_source_id"].tolist() == [0]
    assert result["anchor_observed"].dtype == np.uint8
    assert result["anchor_usd"].dtype == np.float32
    assert result["anchor_source_id"].dtype == np.int64
    assert result["roundtrip_audit"]["observed_cells"] == 242
    assert result["roundtrip_audit"]["USD"]["observed_cells"] == 232
    assert result["roundtrip_audit"]["ticks"]["observed_cells"] == 10


def test_fallback_tick_anchor_all_missing_and_genuine_relative_zero(layout):
    raw = np.zeros((4, 2622), np.float32)
    _set_level(raw, layout, 0, "gc_book_close_usd", 2000)
    fallback = layout["anchor_source_priority"][1]
    _set_level(raw, layout, 1, fallback, 2200)
    tick_id = _set_level(raw, layout, 3, "gc_m5_observed_book_opening_mid_tick", 22345.5)
    result = geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)
    assert result["anchor_source_id"].tolist() == [0, 1, -1, tick_id]
    assert result["anchor_observed"].tolist() == [1, 1, 0, 1]
    np.testing.assert_array_equal(result["anchor_usd"], np.array([2000, 2200, 0, 2234.55], np.float32))
    for row, source_id in ((0, 0), (1, 1), (3, tick_id)):
        assert result["local_values"][row, layout["level_value_indices"][source_id]] == 0
        assert result["local_values"][row, layout["level_mask_indices"][source_id]] == 1
    np.testing.assert_array_equal(_bits(result["local_values"][2]), _bits(raw[2]))
    assert result["roundtrip_audit"]["ticks"]["max_anchor_storage_component_usd"] > 0
    assert result["roundtrip_audit"]["max_anchor_storage_abs_error_usd"] > 0
    assert result["roundtrip_audit"]["unobserved_anchor_rows"] == 1


def test_all_missing_audit_has_no_fabricated_kernel_or_error_cells(layout):
    raw = np.zeros((2, 2622), np.float32)
    result = geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)
    assert result["anchor_source_id"].tolist() == [-1, -1]
    assert result["anchor_observed"].tolist() == [0, 0]
    assert result["anchor_usd"].tolist() == [0, 0]
    for unit in ("USD", "ticks"):
        audit = result["roundtrip_audit"][unit]
        assert audit["observed_cells"] == 0
        assert audit["max_abs_error_native"] is None
        assert audit["max_error_to_positive_bound_ratio"] is None
        assert audit["inverse_to_float32_different_cells"] == 0


def test_noncontiguous_input_and_masked_negative_zero_are_preserved(layout):
    backing = np.zeros((2, 2630), np.float32)
    raw = backing[:, :2622]
    assert not raw.flags.c_contiguous
    masked_level = layout["level_value_indices"][3]
    raw[0, masked_level] = np.float32(-0.0)
    _set_level(raw, layout, 1, "gc_book_close_usd", 2000)
    result = geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)
    assert _bits(result["local_values"])[0, masked_level] == _bits(raw)[0, masked_level]
    np.testing.assert_array_equal(_bits(result["local_values"][:, layout["registry_mask_indices"]]),
        _bits(raw[:, layout["registry_mask_indices"]]))


def test_actual_reconstruction_absolute_relative_and_ulp_diagnostics(layout):
    raw = np.zeros((1, 2622), np.float32)
    _set_level(raw, layout, 0, "gc_book_close_usd", 2234.55)
    _set_level(raw, layout, 0, "gc_m5_observed_close_usd", 2345.67)
    _set_level(raw, layout, 0, "gc_m5_observed_book_opening_mid_tick", 23456.5)
    result = geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)
    for unit in ("USD", "ticks"):
        columns = [i for i, u in enumerate(layout["level_units"])
            if u == unit and raw[0, layout["level_mask_indices"][i]] == 1]
        indexes = np.asarray(layout["level_value_indices"])[columns]
        original = raw[0, indexes]
        f = 0.1 if unit == "ticks" else 1.0
        inverse = float(result["anchor_usd"][0]) * np.exp(
            result["local_values"][0, indexes].astype(np.float64)) / f
        actual = np.abs(inverse - original.astype(np.float64))
        ulp = original.astype(np.float64) - np.nextafter(original, np.float32(0)).astype(np.float64)
        audit = result["roundtrip_audit"][unit]
        assert audit["max_abs_error_native"] == float(np.max(actual))
        assert audit["max_abs_relative_error_native"] == float(np.max(actual / original))
        assert audit["max_source_native_float32_ulp_scaled_abs_error"] == float(np.max(actual / ulp))
        assert audit["inverse_to_float32_different_cells"] == int(np.count_nonzero(
            _bits(inverse.astype(np.float32)) != _bits(original)))
        assert audit["max_abs_error_native"] <= audit["max_bound_native"]
        assert audit["bound_violations"] == 0
    # A USD anchor already comes from an exact F32 source value. Tick conversion
    # is the separate case above that exercises extra anchor-storage rounding.
    assert result["roundtrip_audit"]["max_anchor_storage_abs_error_usd"] == 0
    assert result["roundtrip_audit"]["max_relative_log_storage_abs_error"] > 0
    assert "not_accuracy_gate" in result["roundtrip_audit"]["kind"]


@pytest.mark.parametrize("tick", [0, -1, True, 0.1, "100000000", np.int64(100000000)])
def test_tick_scale_requires_bound_positive_native_integer(layout, tick):
    with pytest.raises(RuntimeError, match="TICK_SIZE_RAW"):
        geometry.transform_gc_relative_price_block(np.zeros((1, 2622), np.float32), layout=layout, tick_size_raw=tick)


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_observed_nonpositive_level_fails_without_fallback_hiding_it(layout, value):
    raw = np.zeros((1, 2622), np.float32)
    _set_level(raw, layout, 0, "gc_book_close_usd", value)
    _set_level(raw, layout, 0, layout["anchor_source_priority"][1], 2000)
    with pytest.raises(RuntimeError, match="OBSERVED_LEVEL_NONPOSITIVE"):
        geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)


def test_invalid_masks_and_hidden_nonlevel_storage_fail_closed(layout):
    raw = np.zeros((1, 2622), np.float32)
    raw[0, layout["registry_mask_indices"][0]] = 0.5
    with pytest.raises(RuntimeError, match="MASK_DOMAIN"):
        geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)
    raw[:] = 0
    nonlevel = next(index for index in layout["registry_value_indices"] if index in layout["unchanged_indices"])
    raw[0, nonlevel] = 1
    with pytest.raises(RuntimeError, match="MASKED_STORAGE"):
        geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)


@pytest.mark.parametrize("raw", [np.zeros((0, 2622), np.float32),
    np.zeros((1025, 2622), np.float32), np.zeros((1, 2621), np.float32),
    np.zeros((1, 2622), np.float64), np.full((1, 2622), np.nan, np.float32),
    np.full((1, 2622), np.inf, np.float32)])
def test_block_shape_cap_dtype_and_finite_guards(layout, raw):
    with pytest.raises(RuntimeError, match="LOCAL_BLOCK"):
        geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)


def test_compile_rejects_alias_duplicates_missing_mask_and_changed_map(source_contract, names):
    duplicate = list(names);duplicate[0] = duplicate[1]
    with pytest.raises(RuntimeError, match="LOCAL_FIELDS"):
        geometry.compile_gc_relative_price_layout(duplicate, source_contract=source_contract)
    missing = list(names);missing[-1] = "fixture.replaced_mask"
    with pytest.raises(RuntimeError, match="COMPLETE_GC_FIELDS_AND_MASKS"):
        geometry.compile_gc_relative_price_layout(missing, source_contract=source_contract)
    changed = deepcopy(source_contract);changed["anchor_source_priority"].reverse()
    with pytest.raises(RuntimeError, match="SOURCE_MAP"):
        geometry.compile_gc_relative_price_layout(names, source_contract=changed)


def test_tampered_compiled_layout_is_rejected(layout):
    changed = deepcopy(layout);changed["level_value_indices"][0] = 0
    with pytest.raises(RuntimeError, match="LAYOUT"):
        geometry.transform_gc_relative_price_block(np.zeros((1, 2622), np.float32),
            layout=changed, tick_size_raw=100000000)


def test_ulp_diagnostics_cover_source_subnormal_and_maximum_without_epsilon(layout):
    raw = np.zeros((2, 2622), np.float32)
    tiny = np.nextafter(np.float32(0), np.float32(1))
    _set_level(raw, layout, 0, "gc_book_close_usd", tiny)
    _set_level(raw, layout, 1, "gc_book_close_usd", np.finfo(np.float32).max)
    result = geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)
    audit = result["roundtrip_audit"]["USD"]
    assert audit["observed_cells"] == 2
    assert audit["max_source_native_float32_ulp_scaled_abs_error"] == 0
    assert audit["inverse_to_float32_different_cells"] == 0
    assert audit["inverse_to_float32_nonfinite_cells"] == 0


def test_unrepresentable_positive_anchor_is_rejected_without_floor(layout):
    raw = np.zeros((1, 2622), np.float32)
    tiny = np.nextafter(np.float32(0), np.float32(1))
    _set_level(raw, layout, 0, "gc_m5_observed_book_opening_mid_tick", tiny)
    with pytest.raises(RuntimeError, match="ANCHOR32_UNREPRESENTABLE"):
        geometry.transform_gc_relative_price_block(raw, layout=layout, tick_size_raw=100000000)
