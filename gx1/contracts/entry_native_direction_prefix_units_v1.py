"""Exact October research prefix units; metadata only, with no fit authority.

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


def prefix_units_surface_columns(profile, *, surface, field_names):
    """Map exact physical value indices to their source unit and observation mask."""
    spec = require_october_prefix_units_profile(profile)
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
    spec = require_october_prefix_units_profile(profile)
    expected = spec["fit_prefix"]
    _require(isinstance(prefix, Mapping)
             and all(key in prefix and type(prefix[key]) is type(expected[key])
                     and prefix[key] == expected[key] for key in _PREFIX_KEYS), "FIT_IDENTITY")
    return dict(expected)
