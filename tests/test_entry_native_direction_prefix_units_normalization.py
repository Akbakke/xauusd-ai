"""Small synthetic surfaces for explicit October units; no genuine prefix or model.

The physical field identities are real. Eight fabricated rows exercise only the
low-level fitter/validator; they do not qualify the separate 29,849-row prefix.
"""
from copy import deepcopy

import numpy as np
import pytest

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.features import gc_native_context_v1 as gc
from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile


def _case(surface="gc_mtf_d1", *, rows=8):
    profile = units.october_prefix_units_profile()
    physical = compile_compact_entry_execution_profile()["physical_field_names"]
    names = list(history.gc_mtf_history_layout()["field_names"]["D1"]
                 if surface == "gc_mtf_d1" else
                 physical["mtf"] if surface.startswith("mtf_") else physical[surface])
    columns = units.prefix_units_surface_columns(profile, surface=surface, field_names=names)
    # Other observations are known constants, including genuine binary masks.
    raw = np.ones((rows, len(names)), dtype=np.float32)
    for index, entry in columns.items():
        raw[:, [index, entry["mask_index"]]] = 0.
    kwargs = dict(surface=surface, field_names=names, allow_constant_train_fields=True,
                  unobserved_units_profile=profile)
    return raw, kwargs, columns


def test_full_physical_m5_layout_keeps_single_apex_event():
    raw, kwargs, columns = _case("mtf_m5", rows=12)
    assert columns == {}
    index = kwargs["field_names"].index("geomchan_apex_proximity")
    raw[:, index] = 0.
    raw[7, index] = np.float32(0.5902255773544312)
    stats = norm.fit_surface_normalization(raw, **kwargs)
    _require(stats, kwargs)
    assert stats["field_names"][index] == "geomchan_apex_proximity"
    assert stats["scale_source"][index] == "source_bounded_unit_scale"
    assert stats["center"][index] == 0. and stats["scale"][index] == 1.
    transformed = norm.apply_surface_normalization(raw, stats)
    assert transformed[7, index] > 0.
    np.testing.assert_allclose(norm.invert_surface_normalization(transformed, stats), raw, atol=1e-6)


def _require(stats, kwargs):
    return norm.require_surface_normalization(
        stats, surface=kwargs["surface"], field_names=kwargs["field_names"],
        unobserved_units_profile=kwargs["unobserved_units_profile"])


def _reseal(stats):
    """Use the canonical digest, so negative tests go beyond a stale hash."""
    keys = ("field_names", "center", "scale", "binary_mask", "categorical_mask",
            "categorical_domains", "train_transformed_min", "train_transformed_max")
    stats["stats_sha256"] = norm._stats_sha256(
        **{key: stats[key] for key in keys},
        unobserved_units_profile_sha256=stats["unobserved_units_profile_sha256"],
        observed_fit_counts=stats["observed_fit_counts"])


@pytest.mark.parametrize("surface", ["gc_mtf_d1", "signal", "ctx_cont"])
def test_explicit_units_keep_exact_empty_support_and_default_native_rejects(surface):
    raw, kwargs, columns = _case(surface)
    original = raw.copy()
    stats = norm.fit_surface_normalization(raw, **kwargs)
    _require(stats, kwargs)
    with pytest.raises(RuntimeError):
        norm.require_surface_normalization(
            stats, surface=surface, field_names=kwargs["field_names"])
    assert stats["fit_row_count"] == len(raw) == 8
    assert stats["unobserved_units_profile_sha256"] == kwargs["unobserved_units_profile"]["contract_sha256"]
    assert "allow_unobserved_train_fields" not in stats
    assert "zero_support_profile_sha256" not in stats
    assert stats["unfitted_estimates"] == {entry["field"]: None for entry in columns.values()}
    assert stats["observed_fit_counts"] == [0 if i in columns else len(raw) for i in range(raw.shape[1])]
    for index, entry in columns.items():
        assert (stats["center"][index], stats["scale"][index]) == (0., 1.)
        assert stats["binary_mask"][index] == int(entry["binary"])
        assert stats["scale_source"][index] == (
            "semantic_binary_identity" if entry["binary"] else units.UNIT_SCALE_SOURCE)
        assert stats["train_transformed_min"][index] == stats["train_transformed_max"][index] == 0.
    resealed = deepcopy(stats)
    _reseal(resealed)
    assert resealed["stats_sha256"] == stats["stats_sha256"]
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("surface", ["gc_mtf_d1", "signal", "ctx_cont"])
def test_later_genuine_masks_zero_sign_and_tail_are_preserved_without_projection(surface):
    raw, kwargs, columns = _case(surface)
    stats = norm.fit_surface_normalization(raw, **kwargs)
    served = raw.copy()
    for index, entry in columns.items():
        served[1:, entry["mask_index"]] = 1.
        # Row 0 is unknown zero; row 1 is observed zero.
        served[2:, index] = 1. if entry["binary"] else 7.
        if entry["source_unit"] == "one_source_GC_price_tick":
            served[3, index] = -3.
            served[4, index] = np.float32(1e12)
    original, original_stats = served.copy(), deepcopy(stats)
    encoded = norm.apply_surface_normalization(served, stats)
    for index, entry in columns.items():
        expected = (served[:, index] if entry["binary"] else
                    np.arcsinh(served[:, index].astype(np.float64)).astype(np.float32))
        np.testing.assert_array_equal(encoded[:, index], expected)
        np.testing.assert_array_equal(encoded[:, entry["mask_index"]], served[:, entry["mask_index"]])
        assert encoded[0, index] == encoded[1, index] == 0.
        assert encoded[0, entry["mask_index"]] == 0. and encoded[1, entry["mask_index"]] == 1.
    np.testing.assert_allclose(norm.invert_surface_normalization(encoded, stats), served,
                               rtol=3e-6, atol=3e-6)
    np.testing.assert_array_equal(served, original)
    assert stats == original_stats


@pytest.mark.parametrize("fault", ["extra_missing", "whitelist_observed", "field_identity", "profile_identity"])
def test_fit_rejects_support_outside_the_exact_whitelist(fault):
    raw, kwargs, columns = _case()
    if fault == "extra_missing":
        index, mask = next((i, m) for i, m in gc.masking_pairs(kwargs["field_names"]) if i not in columns)
        raw[:, [index, mask]] = 0.
    elif fault == "whitelist_observed":
        # Even an observed numerical zero is support, and violates this profile.
        raw[0, next(iter(columns.values()))["mask_index"]] = 1.
    elif fault == "field_identity":
        kwargs["field_names"][next(iter(columns))] += "_wrong"
    else:
        kwargs["unobserved_units_profile"]["fit_prefix"]["rows"] += 1
    original = raw.copy()
    with pytest.raises(RuntimeError):
        norm.fit_surface_normalization(raw, **kwargs)
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("fault", ["profile_marker", "count", "nullable_estimate",
                                  "field_identity", "general_opt_in", "december_marker"])
def test_require_rejects_changed_bindings_and_foreign_markers(fault):
    raw, kwargs, columns = _case()
    stats = norm.fit_surface_normalization(raw, **kwargs)
    index, entry = next(iter(columns.items()))
    if fault == "profile_marker":
        stats["unobserved_units_profile_sha256"] = "0" * 64
    elif fault == "count":
        other = next(i for i, _ in gc.masking_pairs(kwargs["field_names"]) if i not in columns)
        stats["observed_fit_counts"][other] -= 1  # Plausible count, but not this sealed fit.
    elif fault == "nullable_estimate":
        stats["unfitted_estimates"][entry["field"]] = 0.
    elif fault == "field_identity":
        stats["field_names"][index] += "_wrong"
    elif fault == "general_opt_in":
        stats["allow_unobserved_train_fields"] = True
    else:
        stats["zero_support_profile_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        _require(stats, kwargs)


@pytest.mark.parametrize("coordinate", ["unit", "mask"])
def test_resealed_invalid_counts_still_fail_semantic_validation(coordinate):
    raw, kwargs, columns = _case()
    stats = norm.fit_surface_normalization(raw, **kwargs)
    previous_hash = stats["stats_sha256"]
    index, entry = next(iter(columns.items()))
    stats["observed_fit_counts"][index if coordinate == "unit" else entry["mask_index"]] = 1
    _reseal(stats)
    assert stats["stats_sha256"] != previous_hash
    with pytest.raises(RuntimeError, match="COUNTS"):
        _require(stats, kwargs)


def test_alias_sharing_requires_both_profiles_and_keeps_own_population_counts():
    signal_raw, signal_kwargs, signal_columns = _case("signal", rows=8)
    ctx_raw, ctx_kwargs, ctx_columns = _case("ctx_cont", rows=6)
    signal = norm.fit_surface_normalization(signal_raw, **signal_kwargs)
    ctx = norm.fit_surface_normalization(ctx_raw, **ctx_kwargs)
    signal_names, ctx_names = signal_kwargs["field_names"], ctx_kwargs["field_names"]
    aliases = []
    for index, entry in ctx_columns.items():
        for ctx_index in (index, entry["mask_index"]):
            field = ctx_names[ctx_index]
            aliases.append(dict(signal_field="ctx_cont." + field,
                                signal_index=signal_names.index("ctx_cont." + field),
                                ctx_cont_field=field, ctx_cont_index=ctx_index,
                                specialist="synthetic_source_identity"))
    original_ctx, original_signal, original_raw = deepcopy(ctx), deepcopy(signal), ctx_raw.copy()
    call = dict(temporal_aliases=aliases, ctx_cont_values=ctx_raw)
    with pytest.raises(RuntimeError, match="OPT_IN_REQUIRED"):
        norm.share_temporal_alias_stats_from_signal(ctx, signal, **call)
    shared = norm.share_temporal_alias_stats_from_signal(
        ctx, signal, **call, unobserved_units_profile=ctx_kwargs["unobserved_units_profile"])
    _require(shared, ctx_kwargs)
    assert shared["unobserved_units_profile_sha256"] == ctx["unobserved_units_profile_sha256"]
    assert shared["unfitted_estimates"] == ctx["unfitted_estimates"]
    assert shared["observed_fit_counts"] == ctx["observed_fit_counts"]
    assert all(shared["observed_fit_counts"][entry["mask_index"]] == 6 for entry in ctx_columns.values())
    assert all(signal["observed_fit_counts"][entry["mask_index"]] == 8 for entry in signal_columns.values())
    for missing_side in ("signal", "ctx_cont"):
        bad_signal, bad_ctx = deepcopy(signal), deepcopy(ctx)
        bad = bad_signal if missing_side == "signal" else bad_ctx
        bad.pop("unobserved_units_profile_sha256")
        with pytest.raises(RuntimeError):
            norm.share_temporal_alias_stats_from_signal(
                bad_ctx, bad_signal, **call,
                unobserved_units_profile=ctx_kwargs["unobserved_units_profile"])
    assert ctx == original_ctx and signal == original_signal
    np.testing.assert_array_equal(ctx_raw, original_raw)
