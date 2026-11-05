"""Metadata and synthetic mutation checks; no market arrays, fit or model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts import entry_native_direction_prefix_units_v1 as owner


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _reseal(profile):
    profile["contract_sha256"] = _sha({k: v for k, v in profile.items() if k != "contract_sha256"})
    return profile


@pytest.fixture(scope="module")
def native_field_names():
    # Source-owned ordered metadata only; these calls construct no model or data.
    from gx1.contracts.entry_gc_mtf_history_v1 import gc_mtf_history_layout
    from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile

    physical = compile_compact_entry_execution_profile()["physical_field_names"]
    return {"signal": physical["signal"], "ctx_cont": physical["ctx_cont"],
            "gc_mtf_d1": gc_mtf_history_layout()["field_names"]["D1"]}


def test_profile_is_fresh_sealed_metadata_bound_to_exact_plan():
    profile = owner.october_prefix_units_profile()
    raw = (Path(__file__).resolve().parents[1] / owner.PLAN_PATH).read_bytes()
    plan = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == profile["plan"]["sha256"] == owner.PLAN_SHA256
    assert profile["profile_name"] == "dynamic_m5_october_prefix_units_v1"
    assert profile["unit_scale_source"] == owner.UNIT_SCALE_SOURCE == "unobserved_prefix_unit_scale"
    assert profile["contract_sha256"] == _sha({k: v for k, v in profile.items() if k != "contract_sha256"})
    assert profile["fields"] == plan["fields"] and profile["surfaces"] == plan["surfaces"]
    assert all(value is False for value in profile["authority"].values())
    expected = owner.require_october_prefix_units_profile(profile)
    assert expected == profile and expected is not profile
    profile["fields"][0]["fixed_scale"] = 99.0
    assert owner.october_prefix_units_profile() == expected


@pytest.mark.parametrize("fault", [
    "field", "unit", "mask_index", "value_index", "width", "names_hash",
    "fit_prefix", "authority", "profile_name", "plan_sha", "extra", "numeric_type",
])
def test_resealed_profile_mutation_is_rejected(fault):
    profile = owner.october_prefix_units_profile()
    surface = profile["surfaces"]["signal"]
    if fault == "field":
        profile["fields"][0]["field"] += "_other"
    elif fault == "unit":
        profile["fields"][0]["source_unit"] = "one_contract"
    elif fault in ("mask_index", "value_index"):
        surface["columns"][0][fault] += 2
    elif fault == "width":
        surface["width"] += 1
    elif fault == "names_hash":
        surface["field_names_sha256"] = "0" * 64
    elif fault == "fit_prefix":
        profile["fit_prefix"]["rows"] += 1
    elif fault == "authority":
        profile["authority"]["training_enabled"] = True
    elif fault == "profile_name":
        profile["profile_name"] = "dynamic_mtf_december_gc_d1_bear_unobserved_v1"
    elif fault == "plan_sha":
        profile["plan"]["sha256"] = "0" * 64
    elif fault == "extra":
        profile["unregistered"] = True
    else:
        profile["fields"][0]["fixed_center"] = False  # False == 0.0 is insufficient identity.
    with pytest.raises(RuntimeError, match="PROFILE_IDENTITY"):
        owner.require_october_prefix_units_profile(_reseal(profile))


def test_changed_plan_bytes_rejected_before_profile_construction(tmp_path, monkeypatch):
    altered = tmp_path / "altered.json"
    altered.write_bytes(owner._PLAN_FILE.read_bytes() + b"\n")
    monkeypatch.setattr(owner, "_PLAN_FILE", altered)
    with pytest.raises(RuntimeError, match="PLAN_SHA"):
        owner.october_prefix_units_profile()


def test_all_153_value_mask_coordinates_match_actual_source_metadata(native_field_names):
    profile = owner.october_prefix_units_profile()
    total = 0
    for surface, names in native_field_names.items():
        columns = owner.prefix_units_surface_columns(profile, surface=surface, field_names=names)
        assert len(columns) == 51
        assert sum(entry["binary"] for entry in columns.values()) == 3
        expected = profile["surfaces"][surface]
        units = {entry["field"]: entry["source_unit"] for entry in profile["fields"]}
        assert _sha(list(names)) == expected["field_names_sha256"]
        for bound in expected["columns"]:
            index, mask_index = bound["value_index"], bound["mask_index"]
            entry = columns[index]
            assert set(entry) == {"field", "mask", "mask_index", "source_unit", "binary"}
            assert entry["field"] == names[index] == expected["value_name_prefix"] + bound["field"]
            assert entry["mask"] == names[mask_index] == entry["field"] + "_observed"
            assert entry["mask_index"] == mask_index
            assert entry["source_unit"] == units[bound["field"]]
            assert entry["binary"] is (entry["source_unit"] == "boolean_0_or_1")
        columns[next(iter(columns))]["mask_index"] = -1
        assert owner.prefix_units_surface_columns(profile, surface=surface, field_names=names) != columns
        total += len(columns)
    assert total == 153


@pytest.mark.parametrize("fault", ["width", "swap", "unbound_name", "duplicate", "not_string"])
def test_wrong_physical_names_fail_closed(native_field_names, fault):
    profile = owner.october_prefix_units_profile()
    names = list(native_field_names["signal"])
    pair = profile["surfaces"]["signal"]["columns"][0]
    if fault == "width":
        names.pop()
    elif fault == "swap":
        i, j = pair["value_index"], pair["mask_index"]
        names[i], names[j] = names[j], names[i]
    elif fault == "unbound_name":
        names[0] += "_changed"
    elif fault == "duplicate":
        names[0] = names[1]
    else:
        names[0] = 0
    with pytest.raises(RuntimeError, match="FIELD_NAMES"):
        owner.prefix_units_surface_columns(profile, surface="signal", field_names=names)


def test_mask_index_is_checked_even_after_matching_field_hash(native_field_names, monkeypatch):
    profile = owner.october_prefix_units_profile()
    changed = deepcopy(profile)
    pair = changed["surfaces"]["signal"]["columns"][0]
    pair["mask_index"] = changed["surfaces"]["signal"]["columns"][1]["mask_index"]
    # Isolate the index check from the independent exact-profile rejection above.
    monkeypatch.setattr(owner, "require_october_prefix_units_profile", lambda _profile: changed)
    with pytest.raises(RuntimeError, match="COLUMN_IDENTITY"):
        owner.prefix_units_surface_columns(profile, surface="signal", field_names=native_field_names["signal"])


def test_other_standard_surfaces_are_empty_and_unknown_surface_is_rejected():
    profile = owner.october_prefix_units_profile()
    others = set(owner.STANDARD_SURFACES) - set(profile["surfaces"])
    assert len(owner.STANDARD_SURFACES) == 12 and len(others) == 9
    for surface in others:
        assert owner.prefix_units_surface_columns(profile, surface=surface, field_names=[]) == {}
    with pytest.raises(RuntimeError, match="SURFACE"):
        owner.prefix_units_surface_columns(profile, surface="mtf_m30", field_names=[])
    profile["profile_name"] = "other"
    with pytest.raises(RuntimeError, match="PROFILE_IDENTITY"):
        owner.prefix_units_surface_columns(profile, surface="mtf_m5", field_names=[])


@pytest.mark.parametrize("key", [
    "rows", "cutoff_ns", "parent_rows_sha256", "bar_start_ns_sha256", "decision_ns_sha256",
])
@pytest.mark.parametrize("fault", ["missing", "different", "wrong_type"])
def test_wrong_or_missing_prefix_identity_rejected(key, fault):
    profile = owner.october_prefix_units_profile()
    prefix = dict(profile["fit_prefix"])
    if fault == "missing":
        prefix.pop(key)
    elif fault == "different":
        prefix[key] = prefix[key] + 1 if type(prefix[key]) is int else "0" * 64
    else:
        prefix[key] = float(prefix[key]) if type(prefix[key]) is int else prefix[key].encode()
    with pytest.raises(RuntimeError, match="FIT_IDENTITY"):
        owner.require_prefix_units_fit_identity(prefix, profile)


def test_exact_prefix_allows_extra_proof_metadata_without_mutation():
    profile = owner.october_prefix_units_profile()
    prefix = {**profile["fit_prefix"], "first_decision_ns": 1714521900000000000}
    checked = owner.require_prefix_units_fit_identity(prefix, profile)
    assert checked == profile["fit_prefix"] and checked is not profile["fit_prefix"]
    assert "first_decision_ns" in prefix and "first_decision_ns" not in checked
