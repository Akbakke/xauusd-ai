"""Small synthetic205 mechanics only; no market payload, native model or fitting authority."""
from copy import deepcopy
import numpy as np
import pytest

from gx1.contracts.entry_relative_price_snapshot_v1 import (
    MAX_BLOCK_ROWS, RAW_WIDTH, BASELINE_WIDTH, CANDIDATE_WIDTH,
    SIGNED_LOCAL_EMA, ANCHOR_FIELDS, _seal,
    compile_relative_price_snapshot_layout, project_raw_canonical_snapshot,
    snapshot_surface_views, transform_relative_price_snapshot_block,
    fit_prefix_snapshot_normalization, require_prefix_snapshot_normalization,
    encode_normalized_snapshot_block,
)
from gx1.contracts.gc_price_coordinate_contract_v1 import build_gc_price_coordinate_source_contract
from gx1.features.htf_features import MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4


@pytest.fixture(scope="module")
def source_contract():
    return build_gc_price_coordinate_source_contract()


@pytest.fixture(scope="module")
def layout(source_contract):
    return compile_relative_price_snapshot_layout(source_contract=source_contract)


def _raw_current(layout, rows=8):
    result = {}
    for surface, domain in layout["native"]["domains"].items():
        if surface == "seq_x":
            continue
        names = domain["field_names"]
        if surface == "ctx_cat":
            result[surface] = (np.arange(rows) % 4).astype(np.int64)[:, None]
            continue
        values = (2000 + np.arange(len(names))[None, :] / 100
                  + np.arange(rows)[:, None] / 2).astype(np.float32)
        binary = set(domain["binary_indices"]) | {
            i for i, name in enumerate(names)
            if name.removeprefix("ctx_cont.") in MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4}
        if surface in ("snap_x", "ctx_cont"):
            local_binary = set(layout["contract"]["surfaces"]["local"]["binary_fields"])
            prefix = "ctx_cont." if surface == "ctx_cont" else ""
            binary |= {i for i, name in enumerate(names) if prefix + name in local_binary}
        for i in binary:
            values[:, i] = np.arange(rows) % 2
        for i, enum in domain["enum_domains"].items():
            values[:, i] = np.asarray(enum)[np.arange(rows) % len(enum)]
        for i, name in enumerate(names):
            if name.removeprefix("ctx_cont.") in SIGNED_LOCAL_EMA:
                values[:, i] = np.asarray([-1, 0, 1])[np.arange(rows) % 3]
        for _, mask in domain["masking_pairs"]:
            values[:, mask] = 1
        for i, constant in domain["source_constants"].items():
            values[:, i] = constant
        result[surface] = values
    for alias in layout["native"]["physical_aliases"]:
        result["ctx_cont"][:, alias["ctx_cont_index"]] = result["snap_x"][:, alias["signal_index"]]
    return result


@pytest.fixture(scope="module")
def packet(layout):
    original = _raw_current(layout)
    values = project_raw_canonical_snapshot(original, layout=layout)
    candidate = transform_relative_price_snapshot_block(
        values, layout=layout, tick_size_raw=100000000)
    clocks = np.arange(len(values), dtype=np.int64) + 10
    baseline_norm = fit_prefix_snapshot_normalization(
        values, clocks, cutoff_ns=100, layout=layout, arm="baseline")
    relative_norm = fit_prefix_snapshot_normalization(
        candidate["values"], clocks, cutoff_ns=100, layout=layout, arm="relative",
        anchor_usd=candidate["anchor_usd"], anchor_observed=candidate["anchor_observed"])
    return original, values, candidate, clocks, baseline_norm, relative_norm


def test_raw_canonical_once_aliases_and_surface_views(layout, source_contract):
    all_gc_values = {"ctx_cont." + item["name"] for item in source_contract["fields"]}
    expected_binary = {"ctx_cont." + item["name"] for item in source_contract["fields"]
                       if item["source_unit"] == "binary"}
    declared_binary = set(layout["contract"]["surfaces"]["local"]["binary_fields"])
    assert len(expected_binary) == 210
    assert declared_binary.intersection(all_gc_values) == expected_binary
    original = _raw_current(layout)
    before = {name: values.copy() for name, values in original.items()}
    projected = project_raw_canonical_snapshot(original, layout=layout)
    assert projected.shape == (8, RAW_WIDTH) and projected.dtype == np.float32
    for spec in layout["contract"]["surfaces"].values():
        assert np.array_equal(projected[:, spec["start"]:spec["stop"]],
                              original[spec["source_surface"]][:, spec["source_indices"]])
    assert np.array_equal(projected[:, -1], original["ctx_cat"][:, 0])
    assert all(np.shares_memory(projected, view)
               for view in snapshot_surface_views(projected, layout=layout).values())
    assert all(np.array_equal(before[name].view(np.uint32), values.view(np.uint32))
               for name, values in original.items())


def test_full_physical_alias_mismatch_rejected(layout):
    raw = _raw_current(layout)
    domain = layout["native"]["domains"]["ctx_cont"]
    binary = set(domain["binary_indices"])
    alias = next(a for a in layout["native"]["physical_aliases"]
                 if a["ctx_cont_index"] not in binary
                 and a["ctx_cont_index"] not in domain["enum_domains"])
    raw["ctx_cont"][0, alias["ctx_cont_index"]] += np.float32(.25)
    with pytest.raises(RuntimeError, match="CURRENT_ALIAS_NOT_EXACT"):
        project_raw_canonical_snapshot(raw, layout=layout)


@pytest.mark.parametrize("violation", ["binary", "signed", "mask", "session"])
def test_active_source_domains_rejected(layout, packet, violation):
    _, values, _, _, norm, _ = packet
    bad = values.copy()
    names = layout["contract"]["raw_columns"]
    if violation == "binary":
        bad[0, names.index("local.ctx_cont.gc_book_ofi_valid")] = .5
    elif violation == "signed":
        bad[0, names.index("local.ctx_cont.m5_ema_stack_aligned_v2")] = 2
    elif violation == "mask":
        value, mask = layout["contract"]["surfaces"]["local"]["masking_pairs"][0]
        bad[0, mask] = 0
        bad[0, value] = 1
    else:
        bad[0, -1] = 4
    with pytest.raises(RuntimeError):
        encode_normalized_snapshot_block(bad, layout=layout, normalization=norm)


@pytest.mark.parametrize("violation", ["cutoff", "duplicate", "reverse"])
def test_prefix_decision_boundary(layout, packet, violation):
    _, values, _, clocks, _, _ = packet
    bad = clocks.copy()
    if violation == "cutoff":
        bad[-1] = 100
    elif violation == "duplicate":
        bad[-1] = bad[-2]
    else:
        bad = bad[::-1].copy()
    with pytest.raises(RuntimeError, match="PREFIX_CLOCKS"):
        fit_prefix_snapshot_normalization(values, bad, cutoff_ns=100, layout=layout, arm="baseline")


def test_output_dimensions_fixed_domains_and_source_id_exclusion(layout, packet):
    _, values, candidate, _, baseline_norm, relative_norm = packet
    baseline, names = encode_normalized_snapshot_block(values, layout=layout, normalization=baseline_norm)
    relative, rnames = encode_normalized_snapshot_block(
        candidate["values"], layout=layout, normalization=relative_norm,
        anchor_usd=candidate["anchor_usd"], anchor_observed=candidate["anchor_observed"])
    assert baseline.shape == (8, BASELINE_WIDTH) and relative.shape == (8, CANDIDATE_WIDTH)
    assert baseline.dtype == relative.dtype == np.float32
    assert rnames == names + list(ANCHOR_FIELDS)
    assert not any("source_id" in name for name in rnames)
    assert len(layout["contract"]["nominal_domains"]) == 7
    assert all(np.isfinite(a).all() for a in (baseline, relative))


def test_unchanged_fields_have_identical_prefix_statistics(layout, packet, source_contract):
    _, _, _, _, baseline, candidate = packet
    level_names = {"ctx_cont." + f["name"] for f in source_contract["fields"]
                   if f["coordinate_kind"] != "unchanged"}
    fields = layout["contract"]["surfaces"]["local"]["field_names"]
    unchanged = [i for i, name in enumerate(fields) if name not in level_names]
    for key in ("center", "scale", "binary_mask", "categorical_mask",
                "scale_source", "train_transformed_min", "train_transformed_max"):
        assert [baseline["surfaces"]["local"][key][i] for i in unchanged] == [
            candidate["surfaces"]["local"][key][i] for i in unchanged]
    for lane in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1"):
        assert baseline["surfaces"][lane] == candidate["surfaces"][lane]


def test_observed_relative_zero_is_not_missing(layout, packet):
    _, _, candidate, _, _, norm = packet
    spec = layout["contract"]["surfaces"]["local"]
    value = spec["field_names"].index("ctx_cont.gc_book_close_usd")
    mask = spec["field_names"].index("ctx_cont.gc_book_close_usd_observed")
    assert (candidate["values"][:, value] == 0).all()
    assert (candidate["values"][:, mask] == 1).all()
    stats = norm["surfaces"]["local"]
    assert stats["center"][value] == 0 and stats["scale"][value] == 1
    assert stats["binary_mask"][value] == 0
    assert stats["scale_source"][value] == "constant_train_unit_scale"


def test_signed_constant_prefix_and_absent_categories_keep_source_domains(layout, packet):
    _, raw, _, clocks, _, _ = packet
    values = raw.copy()
    names = layout["contract"]["raw_columns"]
    signed = names.index("local.ctx_cont.m5_ema_stack_aligned_v2")
    smc = names.index("local.smc_swing_state")
    values[:, signed] = 1
    values[:, smc] = 0
    norm = fit_prefix_snapshot_normalization(values, clocks, cutoff_ns=100, layout=layout, arm="baseline")
    stats = norm["surfaces"]["local"]
    assert stats["binary_mask"][signed] == 0
    assert stats["scale_source"][signed] == "constant_train_unit_scale"
    future = values[:1].copy()
    future[0, signed], future[0, smc], future[0, -1] = -1, 4, 3
    encoded, columns = encode_normalized_snapshot_block(future, layout=layout, normalization=norm)
    assert encoded[0, columns.index(names[signed])] == np.float32(np.arcsinh(-2))
    assert encoded[0, columns.index("local.smc_swing_state=4")] == 1
    assert encoded[0, columns.index("local.ctx_cat.session_id=3")] == 1


def test_anchor_absence_hardmask_and_flag_parity(layout, packet, source_contract):
    _, _, candidate, _, _, norm = packet
    values = candidate["values"][:1].copy()
    spec = layout["contract"]["surfaces"]["local"]
    positions = {name: i for i, name in enumerate(spec["field_names"])}
    for f in source_contract["fields"]:
        if f["coordinate_kind"] != "unchanged":
            values[0, positions["ctx_cont." + f["name"]]] = 0
            values[0, positions["ctx_cont." + f["availability_field"]]] = 0
    encoded, _ = encode_normalized_snapshot_block(
        values, layout=layout, normalization=norm,
        anchor_usd=np.zeros(1, np.float32), anchor_observed=np.zeros(1, np.uint8))
    assert (encoded[0, -2:] == 0).all()
    with pytest.raises(RuntimeError, match="ANCHOR_OBSERVED_PARITY"):
        encode_normalized_snapshot_block(
            candidate["values"][:1], layout=layout, normalization=norm,
            anchor_usd=np.zeros(1, np.float32), anchor_observed=np.zeros(1, np.uint8))


def test_no_observed_prefix_data_fails_without_fallback(layout, packet):
    _, _, candidate, clocks, _, _ = packet
    bad = candidate["values"].copy()
    value, mask = layout["contract"]["surfaces"]["local"]["masking_pairs"][0]
    bad[:, value] = 0
    bad[:, mask] = 0
    with pytest.raises(RuntimeError, match="NO_PHYSICAL_TRAIN_OBSERVATIONS"):
        fit_prefix_snapshot_normalization(
            bad, clocks, cutoff_ns=100, layout=layout, arm="relative",
            anchor_usd=candidate["anchor_usd"], anchor_observed=candidate["anchor_observed"])


def test_new_recipe_rejects_resealed_invalid_statistics(layout, packet):
    _, _, _, _, original, _ = packet
    norm = deepcopy(original)
    norm.pop("contract_sha256")
    norm["surfaces"]["local"]["center"][0] += 1
    with pytest.raises(RuntimeError, match="STATS_SHA256"):
        require_prefix_snapshot_normalization(_seal(norm), layout=layout)


def test_layout_source_identity_and_block_cap(layout, packet, source_contract):
    bad_source = deepcopy(source_contract)
    bad_source["field_count"] = 1204
    with pytest.raises(RuntimeError, match="SOURCE_MAP"):
        compile_relative_price_snapshot_layout(source_contract=bad_source)
    _, values, _, _, _, _ = packet
    too_many = np.repeat(values[:1], MAX_BLOCK_ROWS + 1, axis=0)
    with pytest.raises(RuntimeError, match="BLOCK_ROWS"):
        transform_relative_price_snapshot_block(too_many, layout=layout, tick_size_raw=100000000)
