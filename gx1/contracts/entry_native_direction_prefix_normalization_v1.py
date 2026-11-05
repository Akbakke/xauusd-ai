"""Target-free research prefix populations over unchanged native input owners.

This module loads no artifacts and grants no execution or fitting authority.
Callers supply verified raw surfaces and the bound full parent/clock identity.
Qualification retains all support evidence but returns no population on failure.
The separate fit stage requalifies selected values before any statistics call.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from collections.abc import Mapping
from pathlib import PurePosixPath

import numpy as np

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.features import gc_native_context_v1 as gc
from gx1.features.basic_v1 import session_native_binary_event_fields
from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile
from gx1.features.micro_structure_v1 import range_observation_binary_fields, range_observation_masking_pairs
from gx1.models.entry_v10 import entry_v10_input_normalization as native

SCHEMA_VERSION = "gx1_native_direction_prefix_normalization_v1"
QUALIFICATION_SCHEMA = SCHEMA_VERSION + "_population"
WINDOWS = dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)
BAR_NS = 300_000_000_000
MAX_NS = np.iinfo(np.int64).max
_SOURCE_LINEAGE_KEYS = {
    "dataset_run_id", "train_parquet_path", "train_parquet_sha256",
    "train_manifest_path", "train_manifest_sha256", "m5_prebuilt_path",
    "m5_prebuilt_sha256", "mtf_cache_manifest_path", "mtf_cache_manifest_sha256",
    "mtf_builder_version", "mtf_feature_names_sha256",
}


def _require(condition, reason):
    if not condition:
        raise RuntimeError("NATIVE_DIRECTION_PREFIX_" + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _copy(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _seal(value):
    return {**value, "contract_sha256": _sha(value)}


DECEMBER_ZERO_SUPPORT_NAME = "dynamic_mtf_december_gc_d1_bear_unobserved_v1"
DECEMBER_FIT_PREFIX = {
    "rows": 41866, "cutoff_ns": 1733011200000000000,
    "parent_rows_sha256": "4a344f16766979a7743e17d3d9a85dee9695d7f37b2abda9c0919802877b54c4",
    "bar_start_ns_sha256": "0f95bc0494d5c0060c7490d82146bbe982b2f23b8749b1347abc7acd6d722f03",
    "decision_ns_sha256": "5d1f2d13189650518a7bfe03cbedc423dc775d022112abb5e601a38f1859bd9a",
}
DECEMBER_ZERO_SCALE_SOURCE = "structurally_masked_no_train_statistics"


@lru_cache(maxsize=1)
def _december_surface_bytes():
    full = compile_compact_entry_execution_profile()["physical_field_names"]
    layout = history.gc_mtf_history_layout()
    values = ["gc_d1_observed_proxy_ob_bear_" + k for k in
              ("active", "created", "first_retouch", "low", "high", "age_bars", "origin_volume_contracts")]
    surfaces, widths, field_hashes = {}, {}, {}
    for key, names in (("signal", full["signal"]), ("ctx_cont", full["ctx_cont"]), ("gc_mtf_d1", layout["field_names"]["D1"])):
        fields = [("ctx_cont." if key == "signal" else "") + name for name in values]
        pairs = dict(gc.masking_pairs(names))
        entries = []
        for field in fields:
            loc = names.index(field)
            mask = pairs[loc]
            _require(names[mask] == field + "_observed", "DECEMBER_OBSERVATION_OWNER")
            entries.append({"field": field, "mask": names[mask], "value_index": loc, "mask_index": mask})
        surfaces[key] = entries
        widths[key] = len(names)
        field_hashes[key] = norm._field_names_sha256(names)
    return json.dumps({"surfaces": surfaces, "widths": widths, "field_names_sha256": field_hashes}, sort_keys=True)


def december_zero_support_profile():
    """Only these genuinely unsupported physical pairs stay unknown on this one historical fit and held-out use."""
    return _seal({"name": DECEMBER_ZERO_SUPPORT_NAME, "fit_prefix": _copy(DECEMBER_FIT_PREFIX),
                  "fit_statistics": None, "fixed_transform_inputs": {"center": 0.0, "scale": 1.0},
                  "model_storage": "zero_value_and_zero_observation_mask_on_all_batches_raw_unchanged",
                  **json.loads(_december_surface_bytes())})


def require_december_zero_support_profile(profile):
    expected = december_zero_support_profile()
    _require(isinstance(profile, Mapping) and dict(profile) == expected
             and profile["contract_sha256"] == _sha({k: v for k, v in profile.items() if k != "contract_sha256"}),
             "DECEMBER_PROFILE")
    return expected


def december_surface_columns(stats):
    spec = december_zero_support_profile()
    surface = stats["surface"]
    _require(stats.get("zero_support_profile_sha256") == spec["contract_sha256"] and surface in spec["surfaces"]
             and norm._field_names_sha256(stats["field_names"]) == spec["field_names_sha256"][surface],
             "DECEMBER_SURFACE_IDENTITY")
    return tuple(i for entry in spec["surfaces"][surface] for i in (entry["value_index"], entry["mask_index"]))


def project_december_zero_support(values, *, surface, profile):
    """Return a model-facing copy with both components of each excluded pair masked."""
    spec = require_december_zero_support_profile(profile)
    _require(surface in spec["surfaces"] and isinstance(values, np.ndarray) and values.dtype == np.float32
             and values.ndim > 0 and values.shape[-1] == spec["widths"][surface], "DECEMBER_PROJECT_SURFACE")
    entries = spec["surfaces"][surface]
    vi, mi = [e["value_index"] for e in entries], [e["mask_index"] for e in entries]
    val, mask = values[..., vi], values[..., mi]
    _require(np.isfinite(val).all() and np.isin(mask, (0., 1.)).all()
             and np.all(val[mask == 0.] == 0.), "DECEMBER_MASKED_STORAGE")
    binary = [k for k, e in enumerate(entries)
              if e["field"].endswith(("_bear_active", "_bear_created", "_bear_first_retouch"))]
    _require(np.isin(val[..., binary], (0., 1.)).all(), "DECEMBER_RAW_BINARY_DOMAIN")
    result = values.copy()
    result[..., vi + mi] = 0.
    return result


def december_runtime_columns(base, gc_history, profile):
    """Bind all structural zero buffers to one research prefix; an ordinary native constructor rejects them."""
    surfaces = dict(base["surfaces"])
    if gc_history is not None:
        surfaces.update(gc_history["normalization"]["surfaces"])
    marked = {k: v["zero_support_profile_sha256"] for k, v in surfaces.items() if "zero_support_profile_sha256" in v}
    if profile is None and not marked:
        return {}
    spec = require_december_zero_support_profile(profile)
    _require(marked == {k: spec["contract_sha256"] for k in spec["surfaces"]} and gc_history is not None
             and "relative_price" not in gc_history["normalization"]
             and base["lineage"]["train_row_count"] == base["lineage"]["entry_train_decision_row_count"] == spec["fit_prefix"]["rows"]
             and base["lineage"]["exit_train_decision_row_count"] == 0
             and gc_history["normalization"]["provenance"]["train_end_ns"] == spec["fit_prefix"]["cutoff_ns"],
             "DECEMBER_RUNTIME_SCOPE")
    return {k: december_surface_columns(surfaces[k]) for k in spec["surfaces"]}


def require_december_normalization_bindings(fitted, profile):
    spec = require_december_zero_support_profile(profile)
    report = fitted["prefix_qualification"]
    _require(fitted.get("zero_support_profile") == report.get("zero_support_profile") == spec
             and report["fit_allowed"] is True and report["blockers"] == []
             and all(report["proof"]["prefix"].get(k) == v for k, v in spec["fit_prefix"].items()),
             "DECEMBER_FIT_BINDING")
    december_runtime_columns(fitted["base_normalization"], fitted["gc_mtf_history"], spec)
    all_stats = {**fitted["base_normalization"]["surfaces"], **fitted["gc_mtf_history"]["normalization"]["surfaces"]}
    _require(set(all_stats) == set(report["support"]) == set(norm.EXPECTED_SURFACES) | {"gc_mtf_" + tf.lower() for tf in WINDOWS},
             "DECEMBER_FULL_SURFACES")
    unions = fitted["gc_mtf_history"]["normalization"]["provenance"]["row_union"]
    for tf in WINDOWS:
        measured = report["proof"]["gc"][tf]
        _require(unions[tf] == {k: measured[k] for k in ("rows", "native_row_indices_sha256", "available_ns_sha256")},
                 "DECEMBER_GC_ROW_UNION:" + tf)
    expected_missing = []
    for key, stat in all_stats.items():
        _require(stat["fit_row_count"] == report["support"][key]["population_rows"], "DECEMBER_SURFACE_FIT_ROWS:" + key)
        if key not in spec["surfaces"]:
            _require("zero_support_profile_sha256" not in stat, "DECEMBER_EXTRA_SURFACE")
            continue
        entries = spec["surfaces"][key]
        _require(stat.get("zero_support_profile_sha256") == spec["contract_sha256"]
                 and stat.get("unfitted_estimates") == {e["field"]: None for e in entries}, "DECEMBER_SURFACE_BINDING:" + key)
        for e in entries:
            i = e["value_index"]
            _require(stat["field_names"][i] == e["field"] and stat["scale_source"][i] == DECEMBER_ZERO_SCALE_SOURCE
                     and stat["center"][i] == 0. and stat["scale"][i] == 1.
                     and report["support"][key]["observed_counts"][i] == 0
                     and report["support"][key]["observed_min"][i] is None
                     and report["support"][key]["observed_max"][i] is None, "DECEMBER_FIT_EVIDENCE")
            expected_missing.append((key, e["field"]))
    recorded = report["excluded_zero_support"]
    _require(set(spec["surfaces"]) <= set(all_stats)
             and sorted(expected_missing) == sorted((r["surface"], r["field"]) for r in recorded)
             and all(r["reason"] == "no_observed_prefix_values" for r in recorded), "DECEMBER_EXCLUSION_BINDING")



def require_october_units_normalization_bindings(fitted, profile):
    """Validate an exact physical prefix fit without granting model-run authority."""
    spec = units.require_october_prefix_units_profile(profile)
    _require(isinstance(fitted, Mapping)
             and fitted.get("contract_sha256") == _sha({k: v for k, v in fitted.items() if k != "contract_sha256"})
             and fitted.get("schema_version") == SCHEMA_VERSION
             and fitted.get("native_launch_authorized") is False
             and fitted.get("genuine_model_fit_authorized") is False
             and fitted.get("old055_or223_normalization_identity_reused") is False,
             "UNOBSERVED_UNITS_FITTED_SEAL")
    report = fitted.get("prefix_qualification")
    _require(isinstance(report, Mapping)
             and report.get("contract_sha256") == _sha({k: v for k, v in report.items() if k != "contract_sha256"})
             and report.get("schema_version") == QUALIFICATION_SCHEMA
             and fitted.get("unobserved_units_profile") == report.get("unobserved_units_profile") == spec
             and fitted.get("zero_support_profile") is None and report.get("zero_support_profile") is None
             and report.get("fit_allowed") is True and report.get("blockers") == []
             and report.get("statistics_fits") == 0,
             "UNOBSERVED_UNITS_REPORT_BINDING")
    proof = report["proof"]
    units.require_prefix_units_fit_identity(proof["prefix"], spec)
    physical = compile_compact_entry_execution_profile()["physical_field_names"]
    base, gc_history = fitted["base_normalization"], fitted["gc_mtf_history"]
    _require(isinstance(base, Mapping) and isinstance(gc_history, Mapping)
             and "relative_price" not in gc_history["normalization"],
             "UNOBSERVED_UNITS_RAW_GC")
    expected_names = {"signal": physical["signal"], "ctx_cont": physical["ctx_cont"]}
    expected_names.update({"mtf_" + tf.lower(): physical["mtf"] for tf in WINDOWS})
    norm.require_input_normalization_contract(base, expected_field_names=expected_names,
        expected_ctx_cat_names=physical["ctx_cat"],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=spec)
    history.require_gc_mtf_history(gc_history, unobserved_units_profile=spec)
    _require(base["lineage"]["train_row_count"] == base["lineage"]["entry_train_decision_row_count"] == spec["fit_prefix"]["rows"]
             and base["lineage"]["exit_train_decision_row_count"] == 0
             and gc_history["normalization"]["provenance"]["train_end_ns"] == spec["fit_prefix"]["cutoff_ns"],
             "UNOBSERVED_UNITS_PREFIX_LINEAGE")
    all_stats = {**base["surfaces"], **gc_history["normalization"]["surfaces"]}
    _require(set(all_stats) == set(report["support"]) == set(units.STANDARD_SURFACES),
             "UNOBSERVED_UNITS_FULL_SURFACES")
    unions = gc_history["normalization"]["provenance"]["row_union"]
    for tf in WINDOWS:
        measured = proof["gc"][tf]
        _require(unions[tf] == {k: measured[k] for k in ("rows", "native_row_indices_sha256", "available_ns_sha256")},
                 "UNOBSERVED_UNITS_GC_ROW_UNION:" + tf)
    declared = []
    for key, stat in all_stats.items():
        measured = report["support"][key]
        _require(stat["field_names"] == measured["field_names"]
                 and stat["fit_row_count"] == measured["population_rows"]
                 and "zero_support_profile_sha256" not in stat,
                 "UNOBSERVED_UNITS_SURFACE_IDENTITY:" + key)
        columns = units.prefix_units_surface_columns(spec, surface=key, field_names=stat["field_names"])
        if not columns:
            _require("unobserved_units_profile_sha256" not in stat, "UNOBSERVED_UNITS_EXTRA_SURFACE:" + key)
            continue
        _require(stat.get("unobserved_units_profile_sha256") == spec["contract_sha256"]
                 and stat.get("unfitted_estimates") == {entry["field"]: None for entry in columns.values()}
                 and stat.get("observed_fit_counts") == measured["observed_counts"],
                 "UNOBSERVED_UNITS_STATS_BINDING:" + key)
        for index, entry in columns.items():
            mask = entry["mask_index"]
            _require(stat["scale_source"][index] == ("semantic_binary_identity" if entry["binary"] else units.UNIT_SCALE_SOURCE)
                     and stat["center"][index] == 0. and stat["scale"][index] == 1.
                     and measured["observed_counts"][index] == 0
                     and measured["observed_min"][index] is None and measured["observed_max"][index] is None
                     and measured["observed_counts"][mask] == measured["population_rows"]
                     and measured["observed_min"][mask] == measured["observed_max"][mask] == 0.,
                     "UNOBSERVED_UNITS_FIT_EVIDENCE:" + key + ":" + entry["field"])
            declared.append((key, entry["field"]))
    recorded = report.get("declared_unit_support")
    _require(len(declared) == 153 and isinstance(recorded, list)
             and sorted(declared) == sorted((item["surface"], item["field"]) for item in recorded)
             and all(item.get("reason") == "no_observed_prefix_values" for item in recorded),
             "UNOBSERVED_UNITS_DECLARATION_BINDING")
    return fitted


def _hash_axis(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<i8").tobytes()).hexdigest()


def _axis(value, name):
    _require(isinstance(value, np.ndarray) and value.dtype == np.int64
        and value.ndim == 1 and len(value) >= 2 and np.all(value >= 0)
        and np.all(value[1:] > value[:-1]), name)
    return value


def _ref(value):
    _require(isinstance(value, Mapping) and set(value) == {"path", "sha256"}, "SOURCE_REF")
    path, digest = value["path"], value["sha256"]
    _require(isinstance(path, str) and PurePosixPath(path).is_absolute()
        and ".." not in PurePosixPath(path).parts and isinstance(digest, str)
        and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "SOURCE_REF")


def _domains(names, surface):
    binary = (set(norm.MTF_SEMANTIC_BINARY_FIELDS).intersection(names)
        if surface.startswith("mtf_") else gc.binary_fields(names))
    binary |= range_observation_binary_fields(names) | session_native_binary_event_fields(names)
    if surface.startswith("gc_mtf_"):
        binary |= gc.mtf_history_binary_fields(names)
    categories = (norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS if surface == "signal" else
        norm.MTF_SEMANTIC_CATEGORICAL_DOMAINS if surface.startswith("mtf_") else {})
    return binary, {name: list(domain) for name, domain in categories.items() if name in names}


def _support(parts, names, surface):
    """Counts and observed support only; no quantile, scale or statistics fit."""
    parts, rows, width = norm._as_population_parts(parts, surface=surface)
    _require(width == len(names), "SUPPORT_WIDTH:" + surface)
    pairs = dict((*gc.masking_pairs(names), *range_observation_masking_pairs(names)))
    binary, categories = _domains(names, surface)
    counts = np.zeros(width, dtype=np.int64)
    nonzero = np.zeros(width, dtype=np.int64)
    lower, upper = np.full(width, np.inf), np.full(width, -np.inf)
    for matrix, indices, _source in parts:
        size = len(matrix) if indices is None else len(indices)
        for start in range(0, size, 1024):
            block = matrix[start:start + 1024] if indices is None else matrix[indices[start:start + 1024]]
            _require(np.isfinite(block).all(), "SUPPORT_NONFINITE:" + surface)
            for vi, mi in pairs.items():
                mask = block[:, mi]
                _require(np.isin(mask, (0., 1.)).all() and np.all(block[mask == 0., vi] == 0.),
                    "MASKED_STORAGE:" + surface + ":" + names[vi])
            for i, name in enumerate(names):
                column = block[:, i]
                if i in pairs:
                    column = column[block[:, pairs[i]] == 1.]
                if name in binary:
                    _require(np.isin(column, (0., 1.)).all(), "BINARY_DOMAIN:" + name)
                if name in categories:
                    _require(np.isin(column, categories[name]).all(), "CATEGORY_DOMAIN:" + name)
                counts[i] += len(column)
                nonzero[i] += np.count_nonzero(column)
                if len(column):
                    lower[i] = min(lower[i], float(column.min()))
                    upper[i] = max(upper[i], float(column.max()))
    return {
        "field_names": list(names), "population_rows": rows,
        "masking_pairs": [list(p) for p in sorted(pairs.items())],
        "observed_counts": counts.tolist(), "nonzero_observed_counts": nonzero.tolist(),
        "observed_min": [float(x) if counts[i] else None for i, x in enumerate(lower)],
        "observed_max": [float(x) if counts[i] else None for i, x in enumerate(upper)],
        "observed_constant": [bool(counts[i] and lower[i] == upper[i]) for i in range(width)],
        "binary_fields": sorted(binary), "categorical_domains": categories,
    }


def qualify_native_direction_prefix_population(
    *, parent_rows, bar_start_ns, decision_ns, expected_axes, cutoff_ns,
    expected_prefix_parent_rows, entry_sequence_source, snap_x, ctx_cont, ctx_cat,
    ordered_signal_names, multi_tf_sources, gc_surfaces, gc_layout,
    base_lineage, source_bindings, zero_support_profile=None, unobserved_units_profile=None,
    prefix_only_input_arrays=False,
):
    """Bind the full source axis; audit every prefix input before any stats fit.

    ``expected_axes`` has source={path,sha256}, rows, and raw little-endian I64
    parent_rows_sha256/bar_start_ns_sha256/decision_ns_sha256. It is supplied by
    the caller's immutable 245 metadata binding, not derived from label masks.
    No qualified population is returned when any observed support is absent.
    """
    parents = _axis(parent_rows, "PARENT_AXIS")
    starts, decisions = _axis(bar_start_ns, "BAR_AXIS"), _axis(decision_ns, "DECISION_AXIS")
    _require(len(parents) == len(starts) == len(decisions) and np.all(starts % BAR_NS == 0)
        and np.all(starts <= MAX_NS - BAR_NS) and np.array_equal(decisions, starts + BAR_NS), "DECISION_CLOCK")
    _require(isinstance(expected_axes, Mapping) and set(expected_axes) == {
        "source", "rows", "parent_rows_sha256", "bar_start_ns_sha256", "decision_ns_sha256"}, "EXPECTED_AXES_KEYS")
    _ref(expected_axes["source"])
    actual_axes = {"rows": len(parents), "parent_rows_sha256": _hash_axis(parents),
        "bar_start_ns_sha256": _hash_axis(starts), "decision_ns_sha256": _hash_axis(decisions)}
    _require(type(expected_axes["rows"]) is int and all(expected_axes[k] == v for k, v in actual_axes.items()), "FULL_SOURCE_AXIS_BINDING")
    _require(type(cutoff_ns) is int and 0 < cutoff_ns <= history.TEST_BOUNDARY_NS, "CUTOFF")
    stop = int(np.searchsorted(decisions, cutoff_ns, side="left"))
    supplied = _axis(expected_prefix_parent_rows, "PREFIX_PARENT_AXIS")
    _require(stop >= 2 and np.array_equal(supplied, parents[:stop]), "EXACT_PREFIX_MEMBERSHIP")
    _require(zero_support_profile is None or unobserved_units_profile is None, "PROFILE_MUTUAL_EXCLUSION")
    _require(type(prefix_only_input_arrays) is bool and (not prefix_only_input_arrays or
        unobserved_units_profile is not None), "PREFIX_ONLY_ARRAY_SCOPE")
    spec = None
    if zero_support_profile is not None or unobserved_units_profile is not None:
        prefix = {"rows": stop, "cutoff_ns": cutoff_ns, "parent_rows_sha256": _hash_axis(parents[:stop]),
                  "bar_start_ns_sha256": _hash_axis(starts[:stop]), "decision_ns_sha256": _hash_axis(decisions[:stop])}
        if zero_support_profile is not None:
            spec = require_december_zero_support_profile(zero_support_profile)
            _require(prefix == spec["fit_prefix"], "DECEMBER_FIT_PREFIX")
        else:
            spec = units.require_october_prefix_units_profile(unobserved_units_profile)
            units.require_prefix_units_fit_identity(prefix, spec)
    physical = compile_compact_entry_execution_profile()["physical_field_names"]
    names = list(ordered_signal_names)
    _require(names == list(physical["signal"]), "COMPLETE_SIGNAL_FIELDS")
    _require(isinstance(base_lineage, Mapping) and set(base_lineage) == _SOURCE_LINEAGE_KEYS
        and isinstance(source_bindings, Mapping) and bool(source_bindings), "SOURCE_BINDINGS")
    _require(isinstance(entry_sequence_source, Mapping) and set(entry_sequence_source) == {
        "signal", "times_ns", "sample_positions", "audit"}, "SEQUENCE_SOURCE_KEYS")
    source_times = _axis(entry_sequence_source["times_ns"], "SOURCE_CLOCK")
    positions = entry_sequence_source["sample_positions"]
    mapped_rows = stop if prefix_only_input_arrays else len(parents)
    mapped_starts = starts[:stop] if prefix_only_input_arrays else starts
    _require(isinstance(positions, np.ndarray) and positions.dtype == np.int64
        and positions.shape == (mapped_rows,) and np.all(positions >= native.MODEL_NATIVE_SEQ_LEN - 1)
        and np.all(positions < len(source_times)) and np.all(positions[1:] > positions[:-1])
        and np.all(source_times % BAR_NS == 0)
        and np.array_equal(source_times[positions], mapped_starts), "GLOBAL_SOURCE_POINTERS")
    for array, width, dtype in ((snap_x, len(names), np.float32),
            (ctx_cont, len(physical["ctx_cont"]), np.float32), (ctx_cat, len(physical["ctx_cat"]), np.int64)):
        _require(isinstance(array, np.ndarray) and array.dtype == dtype
            and array.shape == (mapped_rows, width), "CURRENT_SHAPE_DTYPE")
    aliases = native._derive_temporal_aliases(names)
    sequence = dict(entry_sequence_source, sample_positions=positions[:stop])
    signal, times, selected_positions, snap, ctx, cat, _audit = native._validate_source_backed_train_inputs(
        train_snap=snap_x[:stop], train_ctx_cont=ctx_cont[:stop], train_ctx_cat=ctx_cat[:stop],
        entry_sequence_source=sequence, ordered_signal_names=names,
        temporal_aliases=aliases, row_chunk=native.DEFAULT_ROW_CHUNK)
    local_parts, local_proof = native._select_entry_local_population_from_source(
        source_signal=signal, source_times_ns=times, source_positions=selected_positions,
        train_times_ns=starts[:stop], signal_names=names)
    local_ids = local_parts[0].row_indices
    _require(np.all(times[local_ids] <= MAX_NS - BAR_NS)
        and np.all(times[local_ids] + BAR_NS < cutoff_ns), "LOCAL_AVAILABILITY_CUTOFF")
    ctx_parts, cat_parts = [norm.MatrixPopulationPart(ctx, source="prefix_current_ctx")], [norm.MatrixPopulationPart(cat, source="prefix_current_cat")]
    support = {"signal": _support(local_parts, names, "signal"),
        "ctx_cont": _support(ctx_parts, list(physical["ctx_cont"]), "ctx_cont")}
    cat_proof = {"field_names": list(physical["ctx_cat"]), "rows": stop,
        "domains": {name: list(domain) for name, domain in norm.CTX_CAT_DOMAINS.items()}, "counts": {}}
    for i, name in enumerate(physical["ctx_cat"]):
        domain = norm.CTX_CAT_DOMAINS[name]
        _require(np.isin(cat[:, i], domain).all(), "CATEGORY_DOMAIN:" + name)
        cat_proof["counts"][name] = {str(value): int(np.count_nonzero(cat[:, i] == value)) for value in domain}
    _require(isinstance(multi_tf_sources, Mapping) and set(multi_tf_sources) == set(WINDOWS), "FIVE_NATIVE_TF")
    gc_layout = history.require_gc_mtf_history_layout(gc_layout)
    gc_surfaces_valid = history.require_gc_mtf_history_surfaces(gc_surfaces, layout=gc_layout)
    tf_parts, windows, tf_proofs, tf_ids, gc_proofs = {}, {}, {}, {}, {}
    for tf, length in WINDOWS.items():
        frame = multi_tf_sources[tf]
        _require(list(frame.columns) == list(physical["mtf"]), "COMPLETE_TF_FIELDS:" + tf)
        part, window, proof = native.select_shared_causal_mtf_fit_population(
            tf=tf, source=frame, entry_train_times_ns=starts[:stop],
            exit_train_times_ns=np.empty(0, dtype=np.int64), seq_len=length, require_exit_route=False)
        tf_times, _values, _warmup = native._extract_mtf_source(frame, tf=tf)
        ids = part.row_indices
        shift = int(native.MULTI_TF_SHIFT[tf].value)
        _require(np.all(tf_times[ids] <= MAX_NS - shift), "TF_CLOCK_OVERFLOW:" + tf)
        closes = tf_times[ids] + shift
        _require(np.all(closes < cutoff_ns), "TF_AVAILABILITY_CUTOFF:" + tf)
        surface = "mtf_" + tf.lower()
        tf_parts[tf], windows[tf], tf_ids[tf] = part, window, ids
        tf_proofs[tf] = {**proof, "available_ns_sha256": _hash_axis(closes)}
        support[surface] = _support(part, list(physical["mtf"]), surface)
        gc_surface = gc_surfaces_valid[tf]
        at = np.searchsorted(gc_surface["native_row_indices"], ids)
        _require(np.all(at < len(gc_surface["native_row_indices"])), "GC_MISSING_NATIVE_IDS:" + tf)
        _require(np.array_equal(gc_surface["native_row_indices"][at], ids)
            and np.array_equal(gc_surface["available_ns"][at], closes), "GC_NATIVE_ID_AVAILABILITY:" + tf)
        gc_name, gc_names = "gc_mtf_" + tf.lower(), gc_layout["field_names"][tf]
        gc_part = norm.MatrixPopulationPart(gc_surface["features"], row_indices=at, source=gc_name)
        support[gc_name] = _support(gc_part, gc_names, gc_name)
        gc_proofs[tf] = {"rows": len(ids), "native_row_indices_sha256": _hash_axis(ids),
            "available_ns_sha256": _hash_axis(closes), "selected_row_values_sha256": native._hash_selected_surface_rows(
                namespace="research_prefix_" + gc_name, matrix=gc_surface["features"], row_indices=at,
                timestamps_ns=gc_surface["available_ns"], field_names=gc_names, dtype="<f4")}
    blockers = [{"surface": surface, "field": name, "reason": "no_observed_prefix_values"}
        for surface, item in support.items() for name, count in zip(item["field_names"], item["observed_counts"]) if count == 0]
    for alias in aliases:
        si, ci = alias["signal_index"], alias["ctx_cont_index"]
        name = alias["signal_field"]
        if (name not in support["signal"]["binary_fields"] and name not in support["signal"]["categorical_domains"]
                and support["ctx_cont"]["observed_constant"][ci]
                and not support["signal"]["observed_constant"][si]):
            reason = ("inherited_continuous_alias_support_collapse"
                if support["ctx_cont"]["observed_counts"][ci] == stop else
                "partial_current_alias_constant_support_requires_statistics_validation")
            blockers.append({"surface": "ctx_cont", "field": alias["ctx_cont_field"], "reason": reason})
    proof = {
        "scope": "research_entry_only_input_prefix_not_wholeTRAIN_or_native_admission",
        "full_axes": {"source": _copy(expected_axes["source"]), **actual_axes},
        "prefix": {"cutoff_ns": cutoff_ns, "rows": stop,
            "parent_rows_sha256": _hash_axis(parents[:stop]), "bar_start_ns_sha256": _hash_axis(starts[:stop]),
            "decision_ns_sha256": _hash_axis(decisions[:stop]), "first_decision_ns": int(decisions[0]),
            "last_decision_ns": int(decisions[stop - 1]), "selection": "all_D_strictly_less_than_cutoff_no_labels"},
        "local": {**local_proof, "available_ns_sha256": _hash_axis(times[local_ids] + BAR_NS)},
        "current_context_values_sha256": native._hash_train_decision_rows(train_times_ns=starts[:stop],
            snap=snap, ctx_cont=ctx, ctx_cat=cat, row_chunk=native.DEFAULT_ROW_CHUNK),
        "ctx_cat": cat_proof, "temporal_aliases": aliases, "tf": tf_proofs, "gc": gc_proofs,
        "base_lineage": _copy(base_lineage), "source_bindings": _copy(source_bindings),
        "window_rows": dict(WINDOWS), "all_physical_fields_preserved": True,
    }
    extra = {}
    if zero_support_profile is not None:
        allowed = {(key, e["field"]) for key, entries in spec["surfaces"].items() for e in entries}
        excluded = [b for b in blockers if b["reason"] == "no_observed_prefix_values" and (b["surface"], b["field"]) in allowed]
        blockers = [b for b in blockers if b not in excluded]
        for key, field in sorted(allowed - {(b["surface"], b["field"]) for b in excluded}):
            blockers.append({"surface": key, "field": field, "reason": "expected_zero_support_but_observed"})
        extra = {"zero_support_profile": spec, "excluded_zero_support": excluded}
    elif unobserved_units_profile is not None:
        allowed = {(key, entry["field"]) for key in spec["surfaces"]
                   for entry in units.prefix_units_surface_columns(
                       spec, surface=key, field_names=support[key]["field_names"]).values()}
        _require(len(allowed) == 153, "UNOBSERVED_UNITS_EXPECTED_FIELDS")
        declared = [b for b in blockers if b["reason"] == "no_observed_prefix_values"
                    and (b["surface"], b["field"]) in allowed]
        blockers = [b for b in blockers if b not in declared]
        for key, field in sorted(allowed - {(b["surface"], b["field"]) for b in declared}):
            blockers.append({"surface": key, "field": field, "reason": "expected_zero_support_but_observed"})
        extra = {"unobserved_units_profile": spec, "declared_unit_support": declared}
    if prefix_only_input_arrays:
        proof["input_array_scope"] = "prefix_only_full_axis_hash_bound"
    report = _seal({"schema_version": QUALIFICATION_SCHEMA, "proof": proof, "support": support,
        "fit_allowed": not blockers, "blockers": blockers, "statistics_fits": 0, **extra})
    inputs = dict(parent_rows=parent_rows, bar_start_ns=bar_start_ns, decision_ns=decision_ns,
        expected_axes=_copy(expected_axes), cutoff_ns=cutoff_ns, expected_prefix_parent_rows=expected_prefix_parent_rows,
        entry_sequence_source=entry_sequence_source, snap_x=snap_x, ctx_cont=ctx_cont, ctx_cat=ctx_cat,
        ordered_signal_names=names, multi_tf_sources=multi_tf_sources, gc_surfaces=gc_surfaces,
        gc_layout=gc_layout, base_lineage=_copy(base_lineage), source_bindings=_copy(source_bindings))
    if zero_support_profile is not None:
        inputs["zero_support_profile"] = spec
    elif unobserved_units_profile is not None:
        inputs["unobserved_units_profile"] = spec
    if prefix_only_input_arrays:
        inputs["prefix_only_input_arrays"] = True
    population = None if blockers else {"inputs": inputs, "local_parts": local_parts,
        "ctx_parts": ctx_parts, "cat_parts": cat_parts, "tf_parts": tf_parts,
        "tf_ids": tf_ids, "windows": windows, "gc_surfaces": gc_surfaces_valid}
    return {"report": report, "population": population}


def fit_native_direction_prefix_normalization(qualification, *, gc_relative_coordinates=None):
    """Recheck exact prefix hashes/support, then fit native-compatible contracts."""
    _require(isinstance(qualification, Mapping) and set(qualification) == {"report", "population"}, "QUALIFICATION_KEYS")
    report, held = qualification["report"], qualification["population"]
    _require(isinstance(report, Mapping) and report.get("schema_version") == QUALIFICATION_SCHEMA
        and report.get("contract_sha256") == _sha({k: v for k, v in report.items() if k != "contract_sha256"}), "QUALIFICATION_HASH")
    _require(report.get("fit_allowed") is True and isinstance(held, Mapping), "PREFIX_SUPPORT_BLOCKED")
    fresh = qualify_native_direction_prefix_population(**held["inputs"])
    _require(fresh["report"] == report and fresh["population"] is not None, "QUALIFICATION_INPUTS_CHANGED")
    pop, inputs = fresh["population"], fresh["population"]["inputs"]
    profile = report.get("zero_support_profile")
    units_profile = report.get("unobserved_units_profile")
    _require(profile is None or units_profile is None, "PROFILE_MUTUAL_EXCLUSION")
    profile_args = {}
    if profile is not None:
        require_december_zero_support_profile(profile)
        _require(gc_relative_coordinates is None, "DECEMBER_RAW_GC_REQUIRED")
        profile_args["zero_support_profile"] = profile
    elif units_profile is not None:
        units.require_october_prefix_units_profile(units_profile)
        units.require_prefix_units_fit_identity(report["proof"]["prefix"], units_profile)
        _require(gc_relative_coordinates is None, "UNOBSERVED_UNITS_RAW_GC_REQUIRED")
        profile_args["unobserved_units_profile"] = units_profile
    physical = compile_compact_entry_execution_profile()["physical_field_names"]
    proof = report["proof"]
    surfaces = {
        "signal": norm.fit_surface_normalization(pop["local_parts"], surface="signal", field_names=physical["signal"],
            semantic_categorical_domains=norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS, allow_constant_train_fields=True, **profile_args),
        "ctx_cont": norm.fit_surface_normalization(pop["ctx_parts"], surface="ctx_cont", field_names=physical["ctx_cont"],
            semantic_categorical_domains=norm.CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS, allow_constant_train_fields=True, **profile_args),
    }
    surfaces["ctx_cont"] = norm.share_temporal_alias_stats_from_signal(surfaces["ctx_cont"], surfaces["signal"],
        temporal_aliases=proof["temporal_aliases"], ctx_cont_values=pop["ctx_parts"],
        unobserved_units_profile=units_profile)
    for tf in WINDOWS:
        name = "mtf_" + tf.lower()
        surfaces[name] = norm.fit_surface_normalization(pop["tf_parts"][tf], surface=name,
            field_names=physical["mtf"], semantic_categorical_domains=norm.MTF_SEMANTIC_CATEGORICAL_DOMAINS,
            allow_constant_train_fields=True, unobserved_units_profile=units_profile)
    prefix = proof["prefix"]
    starts = inputs["bar_start_ns"][:prefix["rows"]]
    first, last = native._timestamp_iso_utc(starts[0]), native._timestamp_iso_utc(starts[-1])
    lineage = {**proof["base_lineage"], "train_row_count": prefix["rows"],
        "entry_train_decision_row_count": prefix["rows"], "exit_train_decision_row_count": 0,
        "local_fit_row_count": proof["local"]["selected_unique_row_count"], "context_fit_row_count": prefix["rows"],
        "val_fit_row_count": 0, "test_fit_row_count": 0, "train_time_min_utc": first, "train_time_max_utc": last,
        "per_tf_seq_lens": dict(WINDOWS), "per_tf_shift_seconds": {tf: int(native.MULTI_TF_SHIFT[tf].total_seconds()) for tf in WINDOWS},
        "per_tf_fit_windows": pop["windows"]}
    base = norm.build_input_normalization_contract(fit_start_utc=first, fit_end_utc=last, surfaces=surfaces,
        ctx_cat=norm.fit_ctx_cat_contract(pop["cat_parts"], field_names=physical["ctx_cat"]),
        lineage=lineage, temporal_aliases=proof["temporal_aliases"],
        entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE, unobserved_units_profile=units_profile)
    gc_norm = history.fit_gc_mtf_history_normalization(pop["gc_surfaces"], pop["tf_ids"],
        train_start_ns=int(inputs["decision_ns"][0]), train_end_ns=inputs["cutoff_ns"], layout=inputs["gc_layout"],
        relative_price_coordinates=gc_relative_coordinates, **profile_args)
    norm.require_input_normalization_contract(base, expected_field_names={name: surfaces[name]["field_names"] for name in norm.EXPECTED_SURFACES},
        expected_ctx_cat_names=physical["ctx_cat"], expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=units_profile)
    gc_history = history.require_gc_mtf_history({"layout": inputs["gc_layout"], "normalization": gc_norm},
        unobserved_units_profile=units_profile)
    fitted = _seal({"schema_version": SCHEMA_VERSION, "base_normalization": base,
        "gc_mtf_history": gc_history, "prefix_qualification": _copy(report),
        "old055_or223_normalization_identity_reused": False,
        "native_launch_authorized": False, "genuine_model_fit_authorized": False, **profile_args})
    if profile is not None:
        require_december_normalization_bindings(fitted, profile)
    elif units_profile is not None:
        require_october_units_normalization_bindings(fitted, units_profile)
    return fitted
