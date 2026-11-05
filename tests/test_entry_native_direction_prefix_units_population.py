"""Synthetic-only prefix unit routing; real 29,849-row identity remains immutable."""
from copy import deepcopy

import pytest

from gx1.contracts import entry_native_direction_prefix_normalization_v1 as owner
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from tests.test_entry_native_direction_prefix_normalization_v1 import prefix_case, _sync_current


def _case_without_declared_observations():
    case = prefix_case()
    spec = units.october_prefix_units_profile()
    source = case["entry_sequence_source"]["signal"]
    for key, array in (("signal", source),
                       ("gc_mtf_d1", case["gc_surfaces"]["D1"]["features"])):
        names = case["ordered_signal_names"] if key == "signal" else case["gc_layout"]["field_names"]["D1"]
        for index, entry in units.prefix_units_surface_columns(spec, surface=key, field_names=names).items():
            array[:, [index, entry["mask_index"]]] = 0.
    _sync_current(case)
    return case


def _synthetic_profile(monkeypatch, case):
    original = units.october_prefix_units_profile()
    profile = deepcopy(original)
    stop = len(case["expected_prefix_parent_rows"])
    profile["fit_prefix"] = {"rows": stop, "cutoff_ns": case["cutoff_ns"],
                             "parent_rows_sha256": owner._hash_axis(case["parent_rows"][:stop]),
                             "bar_start_ns_sha256": owner._hash_axis(case["bar_start_ns"][:stop]),
                             "decision_ns_sha256": owner._hash_axis(case["decision_ns"][:stop])}
    profile["contract_sha256"] = units._sha({k: v for k, v in profile.items() if k != "contract_sha256"})
    monkeypatch.setattr(units, "october_prefix_units_profile", lambda: deepcopy(profile))
    return profile


def test_actual_profile_rejects_synthetic_prefix_before_statistics():
    case = _case_without_declared_observations()
    with pytest.raises(RuntimeError, match="NATIVE_DIRECTION_PREFIX_UNITS_FIT_IDENTITY"):
        owner.qualify_native_direction_prefix_population(
            **case, unobserved_units_profile=units.october_prefix_units_profile())


def test_declared_units_only_qualify_exact_synthetic_prefix(monkeypatch):
    case = _case_without_declared_observations()
    profile = _synthetic_profile(monkeypatch, case)
    raw = case["entry_sequence_source"]["signal"].copy()
    gc_raw = case["gc_surfaces"]["D1"]["features"].copy()
    ordinary = owner.qualify_native_direction_prefix_population(**case)
    assert ordinary["population"] is None
    assert sum(b["reason"] == "no_observed_prefix_values" for b in ordinary["report"]["blockers"]) == 153
    qualified = owner.qualify_native_direction_prefix_population(**case, unobserved_units_profile=profile)
    report = qualified["report"]
    assert report["fit_allowed"] and report["blockers"] == []
    assert len(report["declared_unit_support"]) == 153
    assert "excluded_zero_support" not in report
    fitted = owner.fit_native_direction_prefix_normalization(qualified)
    assert owner.require_october_units_normalization_bindings(fitted, profile) == fitted
    assert fitted["native_launch_authorized"] is False
    assert fitted["genuine_model_fit_authorized"] is False
    assert (case["entry_sequence_source"]["signal"] == raw).all()
    assert (case["gc_surfaces"]["D1"]["features"] == gc_raw).all()
    with pytest.raises(RuntimeError):
        owner.require_october_units_normalization_bindings(fitted, None)


def test_declared_units_reject_observed_field_and_unlisted_missing_field(monkeypatch):
    case = _case_without_declared_observations()
    profile = _synthetic_profile(monkeypatch, case)
    names = case["ordered_signal_names"]
    entry = next(iter(units.prefix_units_surface_columns(profile, surface="signal", field_names=names).values()))
    index = names.index(entry["field"])
    mask = entry["mask_index"]
    case["entry_sequence_source"]["signal"][:, [index, mask]] = 1.
    _sync_current(case)
    report = owner.qualify_native_direction_prefix_population(**case, unobserved_units_profile=profile)["report"]
    assert not report["fit_allowed"]
    assert {"surface": "signal", "field": entry["field"],
            "reason": "expected_zero_support_but_observed"} in report["blockers"]
    case = _case_without_declared_observations()
    spare_index, spare_mask = next((vi, mi) for vi, mi in owner.gc.masking_pairs(names)
                                   if vi not in units.prefix_units_surface_columns(profile, surface="signal", field_names=names))
    case["entry_sequence_source"]["signal"][:, [spare_index, spare_mask]] = 0.
    _sync_current(case)
    report = owner.qualify_native_direction_prefix_population(**case, unobserved_units_profile=profile)["report"]
    assert not report["fit_allowed"]
    assert {"surface": "signal", "field": names[spare_index],
            "reason": "no_observed_prefix_values"} in report["blockers"]
