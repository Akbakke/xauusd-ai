"""Pure coordinate mechanics; no model, market payload, normalization fit or launch."""
from copy import deepcopy
import json

import numpy as np
import pytest

from gx1.contracts import gc_relative_price_coordinates_v1 as coordinates
from gx1.contracts import gc_price_coordinate_contract_v1 as source_owner
from gx1.contracts.entry_gc_mtf_history_v1 import gc_mtf_history_layout
from gx1.contracts.gc_price_coordinate_contract_v1 import build_gc_price_coordinate_source_contract
from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256


@pytest.fixture(scope="module")
def raw_layout():
    return gc_mtf_history_layout()


@pytest.fixture(scope="module")
def source_contract():
    return build_gc_price_coordinate_source_contract(source_mode="native_current")


@pytest.fixture(scope="module")
def layout(raw_layout, source_contract):
    return coordinates.compile_gc_native_relative_price_layout(history_layout=raw_layout,
        source_contract=source_contract, tick_size_raw=100000000)


def _bits(value):
    return np.ascontiguousarray(value).view(np.uint32)


def _sample(layout, tf):
    surface = layout["surfaces"][tf]
    data = np.zeros((3, surface["raw_width"]), np.float32)
    units = np.asarray([0.1 if unit == "ticks" else 1.0 for unit in surface["level_units"]])
    usd = np.linspace(1991.3, 2037.7, len(units))
    data[0, surface["level_value_indices"]] = (usd / units).astype(np.float32)
    data[0, surface["level_mask_indices"]] = 1
    chosen = surface["level_units"].index("ticks")
    data[1, surface["level_value_indices"][chosen]] = np.float32(22001.5)
    data[1, surface["level_mask_indices"][chosen]] = 1
    # Unobserved signed-zero storage is retained exactly.
    data[2, surface["level_value_indices"][0]] = np.float32(-0.0)
    return data, chosen


def test_complete_source_mapping_and_explicit_family_widths(layout, raw_layout, source_contract):
    assert layout["timeframes"] == raw_layout["timeframes"]
    assert layout["raw_layout_sha256"] == raw_layout["contract_sha256"]
    assert layout["tick_size_raw"] == 100000000
    assert layout["anchor_source_id_is_model_input"] is False
    assert coordinates.require_gc_native_relative_price_layout(layout, history_layout=raw_layout) == layout
    found = []
    source = {x["name"]: x for x in source_contract["fields"]}
    for tf in raw_layout["timeframes"]:
        frame = layout["surfaces"][tf]
        names = raw_layout["field_names"][tf]
        assert frame["raw_field_names"] == names
        assert frame["model_field_names"] == names + list(coordinates.NATIVE_ANCHOR_FIELDS)
        assert frame["raw_width"] == len(names)
        assert frame["model_width"] == len(names) + 2
        assert frame["anchor_value_index"] == len(names)
        assert frame["anchor_mask_index"] == len(names) + 1
        expected_priority = [name for name in source_contract["anchor_source_priority"] if name in names]
        assert frame["anchor_source_priority"] == expected_priority
        assert frame["anchor_family"] == source[expected_priority[0]]["owner"]
        assert frame["level_units"] == [source[name]["source_unit"] for name in expected_priority]
        assert [names[i] for i in frame["level_value_indices"]] == expected_priority
        assert [names[i] for i in frame["level_mask_indices"]] == [
            source[name]["availability_field"] for name in expected_priority]
        found.extend(expected_priority)
        for family, indices in raw_layout["family_indices"][tf].items():
            extra = [len(names), len(names) + 1] if family == frame["anchor_family"] else []
            assert frame["family_indices"][family] == indices + extra
        assert sorted(i for indices in frame["family_indices"].values() for i in indices) == list(range(len(names) + 2))
    assert len(found) == len(set(found)) == len(source_contract["anchor_source_priority"])
    assert set(found) == set(source_contract["anchor_source_priority"])


@pytest.mark.parametrize("tf", ["M5", "M15", "H1", "H4", "D1"])
def test_same_row_anchor_all_level_units_masks_and_f32_roundtrip(layout, tf, record_property):
    frame = layout["surfaces"][tf]
    raw, chosen = _sample(layout, tf)
    before = _bits(raw).copy()
    result = coordinates.transform_gc_native_relative_price_block(raw, layout=layout, timeframe=tf)
    assert result["values"].shape == (3, frame["model_width"])
    assert result["values"].dtype == np.float32
    np.testing.assert_array_equal(result["anchor_source_id"], [0, chosen, -1])
    np.testing.assert_array_equal(result["anchor_observed"], [1, 1, 0])
    np.testing.assert_array_equal(_bits(raw), before)
    np.testing.assert_array_equal(_bits(result["values"][:, frame["unchanged_indices"]]),
        _bits(raw[:, frame["unchanged_indices"]]))
    np.testing.assert_array_equal(_bits(result["values"][:, frame["anchor_value_index"]]),
        _bits(result["anchor_usd"]))
    np.testing.assert_array_equal(result["values"][:, frame["anchor_mask_index"]], [1, 1, 0])
    assert result["values"][2, frame["level_value_indices"][0]].view(np.uint32) == np.float32(-0.0).view(np.uint32)
    assert result["anchor_usd"][2] == 0
    for unit in ("USD", "ticks"):
        error = []
        for c, native_unit in enumerate(frame["level_units"]):
            if native_unit != unit:
                continue
            vi, mi = frame["level_value_indices"][c], frame["level_mask_indices"][c]
            observed = raw[:, mi] == 1
            factor = 0.1 if unit == "ticks" else 1.0
            inverse = result["anchor_usd"][observed].astype(np.float64) * np.exp(
                result["values"][observed, vi].astype(np.float64)) / factor
            error.extend(np.abs(inverse - raw[observed, vi].astype(np.float64)))
        audit = result["roundtrip_audit"][unit]
        assert audit["observed_cells"] == len(error)
        assert audit["max_abs_error_native"] == max(error)
        assert audit["bound_violations"] == 0
    assert result["roundtrip_audit"]["observed_anchor_rows"] == 2
    assert result["roundtrip_audit"]["unobserved_anchor_rows"] == 1
    record_property("roundtrip_audit_json", json.dumps({"timeframe": tf,
        "raw_width": frame["raw_width"], "model_width": frame["model_width"],
        "price_level_count": len(frame["level_units"]),
        "price_unit_counts": {unit: frame["level_units"].count(unit) for unit in ("USD", "ticks")},
        "anchor_family": frame["anchor_family"],
        "anchor_priority_first": frame["anchor_source_priority"][0],
        "roundtrip_audit": result["roundtrip_audit"]}, sort_keys=True))


def test_basis_nonprice_and_original_masks_keep_exact_bits(layout, source_contract):
    frame = layout["surfaces"]["M5"]
    raw, _ = _sample(layout, "M5")
    names = frame["raw_field_names"]
    basis = next(x for x in source_contract["fields"] if x["name"] in names
        and x["column"] in ("spot_basis_bps", "observed_spot_basis_bps"))
    assert basis["coordinate_kind"] == "unchanged"
    index = names.index(basis["name"])
    raw[:, index] = np.asarray([-4.25, 0.0, 3.125], np.float32)
    raw[:, names.index(basis["availability_field"])] = 1
    result = coordinates.transform_gc_native_relative_price_block(raw, layout=layout, timeframe="M5")
    np.testing.assert_array_equal(_bits(result["values"][:, index]), _bits(raw[:, index]))
    np.testing.assert_array_equal(_bits(result["values"][:, frame["registry_mask_indices"]]),
        _bits(raw[:, frame["registry_mask_indices"]]))


@pytest.mark.parametrize("tf", ["M5", "M15", "H1", "H4", "D1"])
def test_common_power_of_two_level_shift_only_changes_retained_anchor(layout, tf):
    frame = layout["surfaces"][tf]
    raw, _ = _sample(layout, tf)
    shifted = raw.copy()
    shifted[:, frame["level_value_indices"]] *= np.float32(2)
    original = coordinates.transform_gc_native_relative_price_block(raw, layout=layout, timeframe=tf)
    changed = coordinates.transform_gc_native_relative_price_block(shifted, layout=layout, timeframe=tf)
    np.testing.assert_array_equal(_bits(original["values"][:, :frame["raw_width"]]),
        _bits(changed["values"][:, :frame["raw_width"]]))
    np.testing.assert_array_equal(changed["anchor_usd"], original["anchor_usd"] * np.float32(2))


@pytest.mark.parametrize("tf", ["M5", "M15", "H1", "H4", "D1"])
def test_torch_pure_tensor_matches_numpy_and_keeps_input_gradient_path(layout, tf):
    import torch
    raw, _ = _sample(layout, tf)
    expected = coordinates.transform_gc_native_relative_price_block(raw, layout=layout, timeframe=tf)["values"]
    tensor = torch.tensor(raw[None], requires_grad=True)
    actual = coordinates.transform_gc_native_relative_price_tensor(tensor, layout=layout, timeframe=tf)
    assert actual.requires_grad and actual.shape == (1, *expected.shape)
    measured = actual.detach().numpy()[0]
    # Distinct log kernels may round one F32 coordinate step differently; this
    # is CPU tensor arithmetic, not a CUDA/native-model parity qualification.
    assert np.all(np.abs(measured.astype(np.float64) - expected.astype(np.float64)) <=
        np.abs(np.spacing(expected).astype(np.float64)))
    frame = layout["surfaces"][tf]
    np.testing.assert_array_equal(_bits(measured[:, frame["unchanged_indices"]]),
        _bits(raw[:, frame["unchanged_indices"]]))
    np.testing.assert_array_equal(_bits(tensor.detach().numpy()[0]), _bits(raw))


def test_native_window_can_exceed_snapshot_cpu_block_cap(layout):
    import torch
    frame = layout["surfaces"]["D1"]
    values = torch.zeros((16, 252, frame["raw_width"]), dtype=torch.float32)
    output = coordinates.transform_gc_native_relative_price_tensor(values, layout=layout, timeframe="D1")
    assert output.shape == (16, 252, frame["model_width"])
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("tick", [0, -1, True, 0.1, "100000000", np.int64(100000000)])
def test_native_tick_size_must_be_explicit_positive_integer(raw_layout, source_contract, tick):
    with pytest.raises(RuntimeError, match="TICK_SIZE_RAW"):
        coordinates.compile_gc_native_relative_price_layout(history_layout=raw_layout,
            source_contract=source_contract, tick_size_raw=tick)


def test_native_layout_tamper_and_resealed_family_drift_rejected(layout, raw_layout):
    changed = deepcopy(layout)
    changed["tick_size_raw"] *= 2
    raw, _ = _sample(layout, "M5")
    with pytest.raises(RuntimeError, match="NATIVE_LAYOUT"):
        coordinates.transform_gc_native_relative_price_block(raw, layout=changed, timeframe="M5")
    changed = deepcopy(layout)
    frame = changed["surfaces"]["M5"]
    frame["family_indices"][frame["anchor_family"]].reverse()
    changed["contract_sha256"] = canonical_sha256({k: v for k, v in changed.items() if k != "contract_sha256"})
    with pytest.raises(RuntimeError, match="NATIVE_LAYOUT_CANONICAL"):
        coordinates.require_gc_native_relative_price_layout(changed, history_layout=raw_layout)


@pytest.mark.parametrize("kind", ["zero", "negative", "mask", "hidden", "nan", "double_transform"])
def test_native_invalid_input_fails_without_fallback_or_mask_rewrite(layout, kind):
    import torch
    frame = layout["surfaces"]["M5"]
    raw, _ = _sample(layout, "M5")
    if kind == "zero": raw[0, frame["level_value_indices"][0]] = 0
    elif kind == "negative": raw[0, frame["level_value_indices"][0]] = -1
    elif kind == "mask": raw[0, frame["level_mask_indices"][0]] = 0.5
    elif kind == "hidden": raw[2, frame["level_value_indices"][0]] = 1
    elif kind == "nan": raw[0, frame["level_value_indices"][0]] = np.nan
    else: raw = coordinates.transform_gc_native_relative_price_block(raw, layout=layout, timeframe="M5")["values"]
    with pytest.raises(RuntimeError):
        coordinates.transform_gc_native_relative_price_block(raw, layout=layout, timeframe="M5")
    with pytest.raises(RuntimeError):
        coordinates.transform_gc_native_relative_price_tensor(torch.tensor(raw), layout=layout, timeframe="M5")


def test_snapshot_identity_is_preserved_and_native_binding_is_explicit(source_contract, layout):
    historical = build_gc_price_coordinate_source_contract()
    assert historical == build_gc_price_coordinate_source_contract(source_mode="snapshot_204")
    assert canonical_sha256(historical) == coordinates.SOURCE_MAP_CANONICAL_SHA256
    assert canonical_sha256(source_contract) == coordinates.NATIVE_SOURCE_MAP_CANONICAL_SHA256
    assert canonical_sha256(historical) != canonical_sha256(source_contract)
    assert "source_mode" not in historical
    assert source_contract["source_mode"] == layout["source_mode"] == "native_current"
    assert "source_map_file_sha256" not in layout
    assert layout["source_map"] == source_contract
    assert layout["source_sha256"] == source_contract["source_sha256"]
    assert source_contract["fields"] == historical["fields"]
    assert source_contract["anchor_source_priority"] == historical["anchor_source_priority"]
    assert source_contract["source_unit_counts"] == historical["source_unit_counts"]
    assert source_contract["tick_to_usd"] == historical["tick_to_usd"]
    assert [name for name in historical["source_sha256"]
        if historical["source_sha256"][name] != source_contract["source_sha256"][name]] == [
            "gx1/features/gc_native_context_v1.py"]
    assert source_owner.HISTORICAL_SOURCE_REVISION == "396fe41aae54b2d73eae636c0e7e24acc2b5f7da"


@pytest.mark.parametrize("mode", [None, "", "legacy", "native", 1])
def test_unknown_source_mode_fails_closed(mode):
    with pytest.raises(RuntimeError, match="SOURCE_MODE"):
        build_gc_price_coordinate_source_contract(source_mode=mode)


def test_native_source_tamper_has_no_historical_fallback(monkeypatch):
    monkeypatch.setitem(source_owner.NATIVE_SOURCE_SHA256,
        "gx1/features/gc_native_context_v1.py", "0" * 64)
    with pytest.raises(RuntimeError, match="SOURCE_SHA256:gx1/features/gc_native_context_v1.py"):
        build_gc_price_coordinate_source_contract(source_mode="native_current")
