"""Synthetic full-width prefix, genuine layout and explicit research-only zero-support slots."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_normalization_v1 as owner
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from tests.test_entry_native_direction_prefix_normalization_v1 import _sync_current, prefix_case


def masked_prefix_case(monkeypatch):
    case = prefix_case()
    count = len(case["expected_prefix_parent_rows"])
    proof = {"rows": count, "cutoff_ns": case["cutoff_ns"], **{
        key + "_sha256": owner._hash_axis(case[key][:count]) for key in ("parent_rows", "bar_start_ns", "decision_ns")}}
    monkeypatch.setattr(owner, "DECEMBER_FIT_PREFIX", proof)
    profile = owner.december_zero_support_profile()
    for surface, raw in (("signal", case["entry_sequence_source"]["signal"]),
                         ("gc_mtf_d1", case["gc_surfaces"]["D1"]["features"])):
        for pair in profile["surfaces"][surface]:
            raw[:, [pair["value_index"], pair["mask_index"]]] = 0.
    _sync_current(case)
    return case, profile


def test_named_fit_excludes_only_the_three_exact_sets_and_keeps_storage_separate(monkeypatch):
    case, profile = masked_prefix_case(monkeypatch)
    legacy = owner.qualify_native_direction_prefix_population(**case)
    assert legacy["population"] is None and len(legacy["report"]["blockers"]) == 21
    qualified = owner.qualify_native_direction_prefix_population(**case, zero_support_profile=profile)
    assert len(qualified["report"]["excluded_zero_support"]) == 21
    fitted = owner.fit_native_direction_prefix_normalization(qualified)
    owner.require_december_normalization_bindings(fitted, profile)
    surfaces = {**fitted["base_normalization"]["surfaces"], **fitted["gc_mtf_history"]["normalization"]["surfaces"]}
    assert len(surfaces) == 12
    columns = owner.december_runtime_columns(fitted["base_normalization"], fitted["gc_mtf_history"], profile)
    with pytest.raises(RuntimeError, match="DECEMBER_PROFILE"):
        owner.december_runtime_columns(fitted["base_normalization"], fitted["gc_mtf_history"], None)
    for key, stats in surfaces.items():
        if key not in profile["surfaces"]:
            assert not {"zero_support_profile_sha256", "unfitted_estimates"}.intersection(stats)
            continue
        pairs = profile["surfaces"][key]
        excluded = [p["value_index"] for p in pairs]
        assert set(stats["unfitted_estimates"]) == {p["field"] for p in pairs}
        assert all(stats["unfitted_estimates"][p["field"]] is None for p in pairs)
        assert {i for i, role in enumerate(stats["scale_source"]) if role == owner.DECEMBER_ZERO_SCALE_SOURCE} == set(excluded)
        for p in pairs:
            i = p["value_index"]
            assert (stats["center"][i], stats["scale"][i]) == (0., 1.)
            assert qualified["report"]["support"][key]["observed_counts"][i] == 0
        raw = np.zeros((2, stats["field_count"]), np.float32)
        for pair in pairs:
            raw[1, [pair["value_index"], pair["mask_index"]]] = 1.
        neighbor = pairs[0]["field"].removesuffix("_active") + "_input_valid"
        for name in (neighbor, neighbor + "_observed"):
            raw[1, stats["field_names"].index(name)] = 1.
        projected = owner.project_december_zero_support(raw, surface=key, profile=profile)
        assert projected[1].sum() == 2 and raw[1].sum() == 16
        transformed = norm.apply_surface_normalization(raw, stats)
        indices = [j for p in pairs for j in (p["value_index"], p["mask_index"])]
        assert not transformed[..., indices].any()
        assert transformed[1, stats["field_names"].index(neighbor)] != transformed[0, stats["field_names"].index(neighbor)]
        shell = SimpleNamespace(_research_zero_columns=columns, _input_normalization_field_names={key: stats["field_names"]},
            _input_norm_categorical_domains={key: stats["categorical_domains"]},
            _input_norm_categorical_index_tuples={key: tuple(i for i, f in enumerate(stats["categorical_mask"]) if f)})
        for attr, dtype in (("center", torch.float32), ("scale", torch.float32), ("binary_mask", torch.bool),
                            ("categorical_mask", torch.bool)):
            setattr(shell, f"input_norm_{key}_{attr}", torch.tensor(stats[attr], dtype=dtype))
        tensor = torch.tensor(raw, requires_grad=True)
        out = EntryV10CtxHybridTransformer._normalize_input_surface(shell, tensor, surface=key)
        np.testing.assert_allclose(out.detach().numpy(), transformed, rtol=1e-6, atol=2e-6)
        grad = torch.autograd.grad(out.sum(), tensor)[0]
        assert not grad[..., indices].count_nonzero()
        bad = deepcopy(stats)
        bad["unfitted_estimates"][pairs[0]["field"]] = 0.
        with pytest.raises(RuntimeError, match="RESEARCH_ZERO_SUPPORT_ESTIMATES"):
            norm.require_surface_normalization(bad, surface=key, field_names=stats["field_names"])


def test_wrong_scope_or_any_other_unsupported_field_fails_closed(monkeypatch):
    case, profile = masked_prefix_case(monkeypatch)
    other = deepcopy(profile)
    other["surfaces"]["gc_mtf_d1"][0]["field"] += "_wrong"
    with pytest.raises(RuntimeError, match="DECEMBER_PROFILE"):
        owner.qualify_native_direction_prefix_population(**case, zero_support_profile=other)
    old = case["expected_prefix_parent_rows"]
    case["expected_prefix_parent_rows"] = old[:-1]
    with pytest.raises(RuntimeError):
        owner.qualify_native_direction_prefix_population(**case, zero_support_profile=profile)
    case["expected_prefix_parent_rows"] = old
    case["cutoff_ns"] -= 1
    with pytest.raises(RuntimeError, match="DECEMBER_FIT_PREFIX"):
        owner.qualify_native_direction_prefix_population(**case, zero_support_profile=profile)
    case["cutoff_ns"] += 1
    raw = case["gc_surfaces"]["D1"]["features"]
    values = profile["surfaces"]["gc_mtf_d1"]
    raw[:, values[0]["mask_index"]] = 1.
    other_value, other_mask = case["gc_layout"]["value_mask_pairs"]["D1"][0]
    raw[:, [other_value, other_mask]] = 0.
    q = owner.qualify_native_direction_prefix_population(**case, zero_support_profile=profile)
    assert q["population"] is None
    assert {b["reason"] for b in q["report"]["blockers"]} == {"expected_zero_support_but_observed", "no_observed_prefix_values"}
    with pytest.raises(RuntimeError, match="PREFIX_SUPPORT_BLOCKED"):
        owner.fit_native_direction_prefix_normalization(q)
