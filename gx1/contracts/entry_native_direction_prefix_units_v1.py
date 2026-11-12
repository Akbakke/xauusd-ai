"""Exact sealed research prefix units; metadata only, with no fit authority.

The immutable preparation plan owns the fields, units and physical indices.
This module neither normalizes values nor imports a fitter or model.
"""
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path


PROFILE_NAME = "dynamic_m5_october_prefix_units_v1"
UNIT_SCALE_SOURCE = "unobserved_prefix_unit_scale"
PLAN_PATH = "configs/research/DYNAMIC_M5_OOF_PREFIX_UNITS_20261105_001.json"
PLAN_SHA256 = "83b70d1b610eef9a3de9c8cf4d0eaf45435bed88d112442e7e315032b3f8c8b2"
_PLAN_FILE = Path(__file__).resolve().parents[2] / PLAN_PATH
MATURE_FOLD1_PROFILE_NAME = "market_state_mature_fold1_prefix_units_v1"
MATURE_FOLD1_PLAN_PATH = "configs/research/MARKET_STATE_MATURE_FOLD1_PREFIX_UNITS_20261107_001.json"
MATURE_FOLD1_PLAN_SHA256 = "a19b4773b443573f3644ba4cc43e14652a929a8b97ea44391586dd5bf59ac630"
_MATURE_FOLD1_PLAN_FILE = Path(__file__).resolve().parents[2] / MATURE_FOLD1_PLAN_PATH
ANCHOR_SPAN_FOLD1_PROFILE_NAME = "market_state_mature_fold1_anchor_span_prefix_units_v1"
ANCHOR_SPAN_FOLD1_PARENT_SHA256 = "8fe7982aaf6ef8de947997cc945669cb58e94c6729282a38f57e7a967d3013bc"
ANCHOR_SPAN_FOLD1_SIGNAL_SHA256 = "99f626ba2a99113627ccc114b8635e6d984e344bf48d759f7058c2184c5d16db"
ANCHOR_SPAN_SIGNAL_FIELDS = (
    "chart.geomline_above_anchor_span_bars", "chart.geomline_below_anchor_span_bars",
)
ANCHOR_SPAN_INPUT_RESULT = {
    "path": "/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001/RESULT.json",
    "sha256": "8878d17a5bf114fcf9cd5eb0e74792ddc5df068b95b67951b594759d5b2b05bb",
}
STANDARD_SURFACES = (
    "signal", "ctx_cont",
    "mtf_m5", "mtf_m15", "mtf_h1", "mtf_h4", "mtf_d1",
    "gc_mtf_m5", "gc_mtf_m15", "gc_mtf_h1", "gc_mtf_h4", "gc_mtf_d1",
)
_PREFIX_KEYS = (
    "rows", "cutoff_ns", "parent_rows_sha256", "bar_start_ns_sha256", "decision_ns_sha256",
)


def _require(condition, code):
    if not condition:
        raise RuntimeError("NATIVE_DIRECTION_PREFIX_UNITS_" + code)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def october_prefix_units_profile():
    """Read the exact plan and return a fresh sealed metadata object."""
    raw = _PLAN_FILE.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == PLAN_SHA256, "PLAN_SHA")
    plan = json.loads(raw)
    _require(plan["schema_version"] == "gx1_exact_prefix_declared_units_preparation_v1"
             and plan["status"] == "PREPARATION_ONLY_NOT_NATIVE_ADMISSION"
             and all(value is False for value in plan["authority"].values()), "PLAN_SCOPE")
    fields = plan["fields"]
    names = [entry["field"] for entry in fields]
    _require(len(names) == len(set(names)) == 51
             and set(plan["surfaces"]) == {"signal", "ctx_cont", "gc_mtf_d1"}, "PLAN_FIELDS")
    for entry in fields:
        binary = entry["source_unit"] == "boolean_0_or_1"
        _require(entry["source_unit"] in plan["units_count"]
                 and entry["fixed_center"] == 0.0 and entry["fixed_scale"] == 1.0
                 and entry["unfitted_estimates"] is None
                 and entry["transform"] == ("identity" if binary else "asinh(x / one_declared_source_unit)"),
                 "PLAN_UNIT")
    for surface, spec in plan["surfaces"].items():
        columns = spec["columns"]
        _require([entry["field"] for entry in columns] == names
                 and type(spec["width"]) is int and spec["width"] > 0
                 and spec["value_name_prefix"] == ("ctx_cont." if surface == "signal" else ""),
                 "PLAN_COLUMNS")
        indices = [entry[key] for entry in columns for key in ("value_index", "mask_index")]
        _require(all(type(index) is int and 0 <= index < spec["width"] for index in indices)
                 and len(set(indices)) == len(indices), "PLAN_INDICES")
    value = {
        "schema_version": "gx1_native_direction_prefix_units_v1",
        "profile_name": PROFILE_NAME,
        "plan": {"path": PLAN_PATH, "sha256": PLAN_SHA256},
        "unit_scale_source": UNIT_SCALE_SOURCE,
        **{key: plan[key] for key in (
            "fit_prefix", "fields", "surfaces", "support_evidence", "unobserved_semantics", "authority")},
    }
    return {**value, "contract_sha256": _sha(value)}


def require_october_prefix_units_profile(profile):
    """Reject modified or resealed profiles; return an independent expected copy."""
    _require(isinstance(profile, Mapping), "PROFILE")
    expected = october_prefix_units_profile()
    try:
        actual = _canonical(dict(profile))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("NATIVE_DIRECTION_PREFIX_UNITS_PROFILE") from exc
    _require(actual == _canonical(expected), "PROFILE_IDENTITY")
    return expected


def mature_fold1_prefix_units_profile():
    """Bind the shorter fold-1 prefix to declared units, never to October statistics."""
    source = october_prefix_units_profile()
    raw = _MATURE_FOLD1_PLAN_FILE.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == MATURE_FOLD1_PLAN_SHA256, "MATURE_PLAN_SHA")
    plan = json.loads(raw)
    _require(set(plan) == {"schema_version", "plan_id", "status", "authority", "unit_source",
                           "support_evidence", "fit_prefix", "unobserved_surface_field_count",
                           "old_normalization_identity_reused"}
             and plan["schema_version"] == "gx1_mature_fold1_declared_units_preparation_v1"
             and plan["plan_id"] == "MARKET_STATE_MATURE_FOLD1_PREFIX_UNITS_20261107_001"
             and plan["status"] == "PREPARATION_ONLY_NOT_NATIVE_ADMISSION"
             and plan["authority"] == source["authority"]
             and plan["unit_source"] == {"path": PLAN_PATH, "sha256": PLAN_SHA256,
                                          "reuse_scope": "declared_source_units_and_physical_column_maps_only"}
             and plan["unobserved_surface_field_count"] == 153
             and plan["old_normalization_identity_reused"] is False,
             "MATURE_PLAN_SCOPE")
    fit = plan["fit_prefix"]
    _require(set(fit) == set(_PREFIX_KEYS) and type(fit["rows"]) is int and fit["rows"] == 29574
             and type(fit["cutoff_ns"]) is int and fit["cutoff_ns"] == 1727654400000000000
             and all(type(fit[key]) is str and len(fit[key]) == 64
                     and all(c in "0123456789abcdef" for c in fit[key]) for key in _PREFIX_KEYS[2:])
             and fit != source["fit_prefix"], "MATURE_FIT_PREFIX")
    evidence = plan["support_evidence"]
    _require(set(evidence) == {"path", "sha256"}
             and evidence["path"] == "/home/andre2/GX1_RUNS/MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001/QUALIFICATION_FOLD_1.json"
             and type(evidence["sha256"]) is str and len(evidence["sha256"]) == 64
             and all(c in "0123456789abcdef" for c in evidence["sha256"]),
             "MATURE_SUPPORT_EVIDENCE")
    value = {
        "schema_version": "gx1_native_direction_prefix_units_v1",
        "profile_name": MATURE_FOLD1_PROFILE_NAME,
        "plan": {"path": MATURE_FOLD1_PLAN_PATH, "sha256": MATURE_FOLD1_PLAN_SHA256},
        "unit_scale_source": UNIT_SCALE_SOURCE,
        "fit_prefix": fit,
        "fields": source["fields"],
        "surfaces": source["surfaces"],
        "support_evidence": evidence,
        "unobserved_semantics": source["unobserved_semantics"],
        "authority": plan["authority"],
    }
    return {**value, "contract_sha256": _sha(value)}


def require_mature_fold1_prefix_units_profile(profile):
    """Reject modified or resealed mature fold-1 profiles."""
    _require(isinstance(profile, Mapping), "PROFILE")
    expected = mature_fold1_prefix_units_profile()
    try:
        actual = _canonical(dict(profile))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("NATIVE_DIRECTION_PREFIX_UNITS_PROFILE") from exc
    _require(actual == _canonical(expected), "PROFILE_IDENTITY")
    return expected


def anchor_span_fold1_prefix_units_profile():
    """Remap the exact mature units after two physical signal-name insertions.

    The retained plan still declares the same source units. This third identity
    binds the completed physical extension without reading artifacts, fitting
    statistics, adding masks or authorizing execution.
    """
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile

    source = mature_fold1_prefix_units_profile()
    _require(source["contract_sha256"] == ANCHOR_SPAN_FOLD1_PARENT_SHA256
             and _sha({k: v for k, v in source.items() if k != "contract_sha256"})
             == ANCHOR_SPAN_FOLD1_PARENT_SHA256, "ANCHOR_SPAN_PARENT")
    names = list(compile_full_ta_entry_execution_profile()["source_field_names"]["signal"])
    _require(len(names) == 2727 and all(type(name) is str for name in names)
             and len(set(names)) == 2727
             and tuple(name for name in names if name in ANCHOR_SPAN_SIGNAL_FIELDS)
             == ANCHOR_SPAN_SIGNAL_FIELDS
             and _sha(names) == ANCHOR_SPAN_FOLD1_SIGNAL_SHA256, "ANCHOR_SPAN_SIGNAL_NAMES")
    parent_names = [name for name in names if name not in ANCHOR_SPAN_SIGNAL_FIELDS]
    bound = source["surfaces"]["signal"]
    _require(len(parent_names) == bound["width"] == 2725
             and _sha(parent_names) == bound["field_names_sha256"], "ANCHOR_SPAN_PARENT_NAMES")
    indices = {name: index for index, name in enumerate(names)}
    columns = []
    for column in bound["columns"]:
        field = bound["value_name_prefix"] + column["field"]
        mask = field + "_observed"
        _require(parent_names[column["value_index"]] == field
                 and parent_names[column["mask_index"]] == mask, "ANCHOR_SPAN_PARENT_COLUMN")
        columns.append({**column, "value_index": indices[field], "mask_index": indices[mask]})
    value = {k: v for k, v in source.items() if k != "contract_sha256"}
    value["profile_name"] = ANCHOR_SPAN_FOLD1_PROFILE_NAME
    value["surfaces"] = {**source["surfaces"], "signal": {
        **bound, "width": len(names), "field_names_sha256": _sha(names), "columns": columns,
    }}
    value["extension_provenance"] = {
        "schema_version": "gx1_mature_fold1_anchor_span_units_extension_v1",
        "parent_profile_name": MATURE_FOLD1_PROFILE_NAME,
        "parent_profile_sha256": ANCHOR_SPAN_FOLD1_PARENT_SHA256,
        "physical_input_result": dict(ANCHOR_SPAN_INPUT_RESULT),
        "inserted_signal_fields": list(ANCHOR_SPAN_SIGNAL_FIELDS),
        "mapping": "existing_signal_value_and_mask_indices_by_exact_field_name",
        "new_observation_masks": False,
        "statistics_fitted": False,
    }
    return {**value, "contract_sha256": _sha(value)}


def require_anchor_span_fold1_prefix_units_profile(profile):
    """Reject modified or resealed anchor-span identities, including provenance."""
    _require(isinstance(profile, Mapping), "PROFILE")
    expected = anchor_span_fold1_prefix_units_profile()
    try:
        actual = _canonical(dict(profile))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("NATIVE_DIRECTION_PREFIX_UNITS_PROFILE") from exc
    _require(actual == _canonical(expected), "PROFILE_IDENTITY")
    return expected


def require_prefix_units_profile(profile):
    """Allow only the three exact, independently sealed prefix unit identities."""
    _require(isinstance(profile, Mapping), "PROFILE")
    if profile.get("profile_name") == PROFILE_NAME:
        return require_october_prefix_units_profile(profile)
    if profile.get("profile_name") == MATURE_FOLD1_PROFILE_NAME:
        return require_mature_fold1_prefix_units_profile(profile)
    if profile.get("profile_name") == ANCHOR_SPAN_FOLD1_PROFILE_NAME:
        return require_anchor_span_fold1_prefix_units_profile(profile)
    _require(False, "PROFILE_IDENTITY")


def prefix_units_surface_columns(profile, *, surface, field_names):
    """Map exact physical value indices to their source unit and observation mask."""
    spec = require_prefix_units_profile(profile)
    _require(isinstance(surface, str) and surface in STANDARD_SURFACES, "SURFACE")
    if surface not in spec["surfaces"]:
        return {}
    bound = spec["surfaces"][surface]
    _require(isinstance(field_names, (list, tuple))
             and len(field_names) == bound["width"]
             and all(type(name) is str for name in field_names)
             and len(set(field_names)) == len(field_names), "FIELD_NAMES")
    names = list(field_names)
    _require(_sha(names) == bound["field_names_sha256"], "FIELD_NAMES_SHA")
    fields = {entry["field"]: entry for entry in spec["fields"]}
    result = {}
    for column in bound["columns"]:
        index, mask_index = column["value_index"], column["mask_index"]
        field = bound["value_name_prefix"] + column["field"]
        mask = field + "_observed"
        _require(names[index] == field and names[mask_index] == mask, "COLUMN_IDENTITY")
        unit = fields[column["field"]]["source_unit"]
        result[index] = {"field": field, "mask": mask, "mask_index": mask_index,
                         "source_unit": unit, "binary": unit == "boolean_0_or_1"}
    return result


def require_prefix_units_fit_identity(prefix, profile):
    """Bind the five exact prefix fields, allowing unrelated extra proof keys."""
    spec = require_prefix_units_profile(profile)
    expected = spec["fit_prefix"]
    _require(isinstance(prefix, Mapping)
             and all(key in prefix and type(prefix[key]) is type(expected[key])
                     and prefix[key] == expected[key] for key in _PREFIX_KEYS), "FIT_IDENTITY")
    return dict(expected)
