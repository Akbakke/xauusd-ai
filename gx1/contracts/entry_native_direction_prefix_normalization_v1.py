"""Target-free research prefix populations over unchanged native input owners.

This module loads no artifacts and grants no execution or fitting authority.
Callers supply verified raw surfaces and the bound full parent/clock identity.
Qualification retains all support evidence but returns no population on failure.
The separate fit stage requalifies selected values before any statistics call.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import PurePosixPath

import numpy as np

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
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
    base_lineage, source_bindings,
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
    physical = compile_compact_entry_execution_profile()["physical_field_names"]
    names = list(ordered_signal_names)
    _require(names == list(physical["signal"]), "COMPLETE_SIGNAL_FIELDS")
    _require(isinstance(base_lineage, Mapping) and set(base_lineage) == _SOURCE_LINEAGE_KEYS
        and isinstance(source_bindings, Mapping) and bool(source_bindings), "SOURCE_BINDINGS")
    _require(isinstance(entry_sequence_source, Mapping) and set(entry_sequence_source) == {
        "signal", "times_ns", "sample_positions", "audit"}, "SEQUENCE_SOURCE_KEYS")
    source_times = _axis(entry_sequence_source["times_ns"], "SOURCE_CLOCK")
    positions = entry_sequence_source["sample_positions"]
    _require(isinstance(positions, np.ndarray) and positions.dtype == np.int64
        and positions.shape == parents.shape and np.all(positions >= native.MODEL_NATIVE_SEQ_LEN - 1)
        and np.all(positions < len(source_times)) and np.all(positions[1:] > positions[:-1])
        and np.all(source_times % BAR_NS == 0)
        and np.array_equal(source_times[positions], starts), "GLOBAL_SOURCE_POINTERS")
    for array, width, dtype in ((snap_x, len(names), np.float32),
            (ctx_cont, len(physical["ctx_cont"]), np.float32), (ctx_cat, len(physical["ctx_cat"]), np.int64)):
        _require(isinstance(array, np.ndarray) and array.dtype == dtype
            and array.shape == (len(parents), width), "FULL_CURRENT_SHAPE_DTYPE")
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
    report = _seal({"schema_version": QUALIFICATION_SCHEMA, "proof": proof, "support": support,
        "fit_allowed": not blockers, "blockers": blockers, "statistics_fits": 0})
    inputs = dict(parent_rows=parent_rows, bar_start_ns=bar_start_ns, decision_ns=decision_ns,
        expected_axes=_copy(expected_axes), cutoff_ns=cutoff_ns, expected_prefix_parent_rows=expected_prefix_parent_rows,
        entry_sequence_source=entry_sequence_source, snap_x=snap_x, ctx_cont=ctx_cont, ctx_cat=ctx_cat,
        ordered_signal_names=names, multi_tf_sources=multi_tf_sources, gc_surfaces=gc_surfaces,
        gc_layout=gc_layout, base_lineage=_copy(base_lineage), source_bindings=_copy(source_bindings))
    population = None if blockers else {"inputs": inputs, "local_parts": local_parts,
        "ctx_parts": ctx_parts, "cat_parts": cat_parts, "tf_parts": tf_parts,
        "tf_ids": tf_ids, "windows": windows, "gc_surfaces": gc_surfaces_valid}
    return {"report": report, "population": population}


def fit_native_direction_prefix_normalization(qualification):
    """Recheck exact prefix hashes/support, then fit native-compatible contracts."""
    _require(isinstance(qualification, Mapping) and set(qualification) == {"report", "population"}, "QUALIFICATION_KEYS")
    report, held = qualification["report"], qualification["population"]
    _require(isinstance(report, Mapping) and report.get("schema_version") == QUALIFICATION_SCHEMA
        and report.get("contract_sha256") == _sha({k: v for k, v in report.items() if k != "contract_sha256"}), "QUALIFICATION_HASH")
    _require(report.get("fit_allowed") is True and isinstance(held, Mapping), "PREFIX_SUPPORT_BLOCKED")
    fresh = qualify_native_direction_prefix_population(**held["inputs"])
    _require(fresh["report"] == report and fresh["population"] is not None, "QUALIFICATION_INPUTS_CHANGED")
    pop, inputs = fresh["population"], fresh["population"]["inputs"]
    physical = compile_compact_entry_execution_profile()["physical_field_names"]
    proof = report["proof"]
    surfaces = {
        "signal": norm.fit_surface_normalization(pop["local_parts"], surface="signal", field_names=physical["signal"],
            semantic_categorical_domains=norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS, allow_constant_train_fields=True),
        "ctx_cont": norm.fit_surface_normalization(pop["ctx_parts"], surface="ctx_cont", field_names=physical["ctx_cont"],
            semantic_categorical_domains=norm.CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS, allow_constant_train_fields=True),
    }
    surfaces["ctx_cont"] = norm.share_temporal_alias_stats_from_signal(surfaces["ctx_cont"], surfaces["signal"],
        temporal_aliases=proof["temporal_aliases"], ctx_cont_values=pop["ctx_parts"])
    for tf in WINDOWS:
        name = "mtf_" + tf.lower()
        surfaces[name] = norm.fit_surface_normalization(pop["tf_parts"][tf], surface=name,
            field_names=physical["mtf"], semantic_categorical_domains=norm.MTF_SEMANTIC_CATEGORICAL_DOMAINS,
            allow_constant_train_fields=True)
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
        entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    gc_norm = history.fit_gc_mtf_history_normalization(pop["gc_surfaces"], pop["tf_ids"],
        train_start_ns=int(inputs["decision_ns"][0]), train_end_ns=inputs["cutoff_ns"], layout=inputs["gc_layout"])
    norm.require_input_normalization_contract(base, expected_field_names={name: surfaces[name]["field_names"] for name in norm.EXPECTED_SURFACES},
        expected_ctx_cat_names=physical["ctx_cat"], expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    gc_history = history.require_gc_mtf_history({"layout": inputs["gc_layout"], "normalization": gc_norm})
    return _seal({"schema_version": SCHEMA_VERSION, "base_normalization": base,
        "gc_mtf_history": gc_history, "prefix_qualification": _copy(report),
        "old055_or223_normalization_identity_reused": False,
        "native_launch_authorized": False, "genuine_model_fit_authorized": False})
