"""205 bounded current snapshots and caller-scoped prefix normalization.

This is a new reference recipe, not native055 normalization or execution
authority. No loader, targets, model, optimizer or fit admission lives here.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import numpy as np

from gx1.contracts.entry_direction_snapshot_data_v1 import MAX_BLOCK_ROWS
from gx1.contracts.entry_direction_snapshot_reference_v1 import (
    PROFILE, _bits_equal, _require_current_surface,
)
from gx1.contracts.entry_dynamic_compact_input_influence_v1 import (
    _MTF, compile_dynamic_compact_influence_layout,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CONTINUOUS_TRANSFORM, SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS,
    MTF_SEMANTIC_CATEGORICAL_DOMAINS, _stats_sha256, _field_names_sha256,
    fit_surface_normalization, apply_surface_normalization,
)
from gx1.contracts.gc_sequence_research_v1 import encode_inputs
from gx1.features.htf_features import MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4
from gx1.features import gc_native_context_v1 as gc
from gx1.features import micro_structure_v1 as micro

SCHEMA_VERSION = "gx1_relative_price_snapshot_v1"
SOURCE_MAP_FILE_SHA256 = "c3dd04288d93d9d78d96ac7444db2f9ba887157142513334445eb3c75a138996"
SOURCE_MAP_CANONICAL_SHA256 = "1c7f117447a27d5167052477e8dbad8855182fea0c946b08e7978f70f56508ec"
RAW_WIDTH, BASELINE_WIDTH, CANDIDATE_WIDTH = 3318, 3345, 3347
SESSION = "local.ctx_cat.session_id"
ANCHOR_FIELDS = ("gc_relative.anchor_usd", "gc_relative.anchor_observed")
SIGNED_LOCAL_EMA = frozenset(
    tf + "_ema_stack_aligned_v2" for tf in ("m5", "m15", "h1", "h4", "d1"))
ARMS = ("baseline", "relative")
PREFIX_RECIPE = {
    "name": "fixed_prefix_parent_current_snapshot_asinh_unobserved_units_v2",
    "fit_population": "caller-qualified unique M5 parents with decision D < cutoff; repeated current HTF values retained",
    "label_masks_condition_normalization": False,
    "allow_constant_train_fields": True,
    "allow_unobserved_train_fields": True,
    "unobserved_policy": "declared identity or center0/unit_scale asinh; count0; no future statistics",
    "constant_policy": "existing constant_train_unit_scale; no future fallback",
    "missing": "source availability masks, never value != 0",
    "anchor": "observed prefix rows only; separate scalar stats; hardmask after apply",
    "native055_equivalence": False,
    "data_or_fit_admission": False,
}


def _require(condition, reason):
    if not condition:
        raise RuntimeError("RELATIVE_PRICE_SNAPSHOT_" + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _seal(value):
    return {**value, "contract_sha256": _sha(value)}


def _require_seal(value):
    _require(isinstance(value, Mapping), "CONTRACT")
    payload = dict(value)
    digest = payload.pop("contract_sha256", None)
    _require(digest == _sha(payload), "CONTRACT_SHA256")


def compile_relative_price_snapshot_layout(*, source_contract):
    """Compile exact source identities, without a normalization fit or055 stats."""
    from gx1.contracts.gc_relative_price_coordinates_v1 import compile_gc_relative_price_layout

    _require(_sha(source_contract) == SOURCE_MAP_CANONICAL_SHA256, "SOURCE_MAP")
    native = compile_dynamic_compact_influence_layout()
    profile = native["profile"]
    _require({k: profile[k] for k in PROFILE} == PROFILE, "PROFILE")
    selected = set(profile["ctx_cont_indices"])
    folded = {a["ctx_cont_index"] for a in native["physical_aliases"]
              if a["signal_index"] in profile["signal_indices"]
              and a["ctx_cont_index"] in selected}
    _require(folded == selected and len(folded) == 2525, "UNFOLDED_CONTEXT")
    _require(len(profile["signal_indices"]) == 2622
             and len(profile["mtf_indices"]) == 139, "ACTIVE_WIDTHS")
    cat = native["domains"]["ctx_cat"]
    _require(tuple(cat["field_names"]) == ("session_id",)
             and tuple(cat["enum_domains"][0]) == (0, 1, 2, 3), "SESSION")
    gc_binary_fields = {"ctx_cont." + item["name"] for item in source_contract["fields"]
                        if item["source_unit"] == "binary"}
    _require(len(gc_binary_fields) == 210, "GC_BINARY_VALUE_COUNT")
    surfaces, names, nominal = {}, [], {}
    for lane, physical, indices in (
        ("local", "snap_x", profile["signal_indices"]),
        *((lane, lane, profile["mtf_indices"]) for lane in _MTF),
    ):
        domain = native["domains"][physical]
        fields = [domain["field_names"][i] for i in indices]
        binary = [name for i, name in zip(indices, fields)
                  if i in domain["binary_indices"] or name in gc_binary_fields
                  or name.removeprefix("ctx_cont.") in MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4]
        declared = (SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS if lane == "local"
                    else MTF_SEMANTIC_CATEGORICAL_DOMAINS)
        categories = {name: list(declared[name]) for name in fields if name in declared}
        _require(not set(binary).intersection(categories), "SOURCE_DOMAIN_OVERLAP:" + lane)
        enums = {name: list(domain["enum_domains"][i])
                 for i, name in zip(indices, fields) if i in domain["enum_domains"]}
        if lane == "local":
            enums.update({name: [-1, 0, 1] for name in fields
                          if name.removeprefix("ctx_cont.") in SIGNED_LOCAL_EMA})
        pairs = [list(p) for p in (*gc.masking_pairs(fields),
                                  *micro.range_observation_masking_pairs(fields))]
        start = len(names)
        tokens = [lane + "." + name for name in fields]
        names.extend(tokens)
        nominal.update({lane + "." + name: domain for name, domain in categories.items()})
        surfaces[lane] = {
            "start": start, "stop": len(names), "field_names": fields,
            "source_surface": physical, "source_indices": list(indices),
            "binary_fields": binary, "categorical_domains": categories,
            "enum_domains": enums, "masking_pairs": pairs,
            "source_constants": {fields[j]: domain["source_constants"][i]
                                 for j, i in enumerate(indices) if i in domain["source_constants"]},
        }
    names.append(SESSION)
    nominal[SESSION] = [0, 1, 2, 3]
    encoded = [name + "=" + str(v) if name in nominal else name
               for name in names for v in (nominal[name] if name in nominal else (None,))]
    _require(len(names) == len(set(names)) == RAW_WIDTH
             and len(encoded) == len(set(encoded)) == BASELINE_WIDTH, "ENCODED_WIDTH")
    geometry = compile_gc_relative_price_layout(
        surfaces["local"]["field_names"], source_contract=source_contract)
    local_positions = {name: i for i, name in enumerate(surfaces["local"]["field_names"])}
    level_mask_indices = [local_positions["ctx_cont." + f["availability_field"]]
                          for f in source_contract["fields"] if f["coordinate_kind"] != "unchanged"]
    _require(len(level_mask_indices) == len(set(level_mask_indices)) == 242, "LEVEL_MASKS")
    contract = _seal({
        "schema_version": SCHEMA_VERSION, "profile": dict(PROFILE),
        "source_map_file_sha256": SOURCE_MAP_FILE_SHA256,
        "source_map_canonical_sha256": SOURCE_MAP_CANONICAL_SHA256,
        "native_source_layout_sha256": _sha(native),
        "geometry_layout_sha256": _sha(geometry),
        "raw_columns": names, "encoded_columns": encoded,
        "nominal_domains": nominal, "surfaces": surfaces, "recipe": dict(PREFIX_RECIPE),
        "baseline_width": BASELINE_WIDTH, "candidate_width": CANDIDATE_WIDTH,
        "anchor_fields": list(ANCHOR_FIELDS), "level_mask_indices": level_mask_indices,
        "all_local_coordinates_from_snap_x": True,
        "native055_equivalence": False, "execution_authorized": False,
    })
    return {"contract": contract, "native": native, "geometry": geometry}


def _require_layout(layout):
    _require(isinstance(layout, Mapping), "LAYOUT")
    contract = layout["contract"]
    _require_seal(contract)
    _require(contract["schema_version"] == SCHEMA_VERSION
             and contract["profile"] == PROFILE
             and contract["source_map_canonical_sha256"] == SOURCE_MAP_CANONICAL_SHA256
             and contract["recipe"] == PREFIX_RECIPE, "LAYOUT_IDENTITY")
    _require(_sha(layout["native"]) == contract["native_source_layout_sha256"]
             and _sha(layout["geometry"]) == contract["geometry_layout_sha256"], "LAYOUT_SOURCE")
    return contract


def snapshot_surface_views(values, *, layout):
    """Six adjacent column-range views; no advanced indexing or matrix copy."""
    contract = _require_layout(layout)
    _require(isinstance(values, np.ndarray) and values.dtype == np.float32
             and values.ndim == 2 and values.shape[1] == RAW_WIDTH
             and values.shape[0] > 0, "RAW_SHAPE")
    return {lane: values[:, spec["start"]:spec["stop"]]
            for lane, spec in contract["surfaces"].items()}


def _validate_values(values, contract):
    _require(isinstance(values, np.ndarray) and values.dtype == np.float32
             and values.ndim == 2 and values.shape[1] == RAW_WIDTH
             and values.shape[0] > 0, "RAW_SHAPE")
    for start in range(0, len(values), MAX_BLOCK_ROWS):
        block = values[start:start + MAX_BLOCK_ROWS]
        _require(np.isfinite(block).all(), "RAW_NONFINITE")
        for lane, spec in contract["surfaces"].items():
            current = block[:, spec["start"]:spec["stop"]]
            positions = {name: i for i, name in enumerate(spec["field_names"])}
            binary_indices = [positions[name] for name in spec["binary_fields"]]
            _require(np.isin(current[:, binary_indices], (0, 1)).all(), "BINARY:" + lane)
            for name, domain in spec["enum_domains"].items():
                _require(np.isin(current[:, positions[name]], domain).all(), "ENUM:" + lane + "." + name)
            if spec["masking_pairs"]:
                vi, mi = zip(*spec["masking_pairs"])
                observed, masked = current[:, mi], current[:, vi]
                _require(np.isin(observed, (0, 1)).all()
                         and not np.any((observed == 0) & (masked != 0)), "MASKED_STORAGE:" + lane)
            for name, constant in spec["source_constants"].items():
                _require((current[:, positions[name]] == constant).all(), "SOURCE_CONSTANT:" + lane + "." + name)
        _require(np.isin(block[:, -1], (0, 1, 2, 3)).all(), "SESSION_DOMAIN")


def project_raw_canonical_snapshot(raw_current, *, layout):
    """Full physical domain/alias checks, then active coordinates exactly once."""
    contract = _require_layout(layout)
    required = {"snap_x", "ctx_cont", "ctx_cat", *_MTF}
    _require(isinstance(raw_current, Mapping) and required <= set(raw_current)
             and set(raw_current) <= required | {"seq_x"}, "CURRENT_KEYS")
    snap = raw_current["snap_x"]
    _require(isinstance(snap, np.ndarray) and snap.ndim == 2
             and 0 < len(snap) <= MAX_BLOCK_ROWS, "BLOCK_ROWS")
    rows = len(snap)
    for surface in required:
        _require_current_surface(raw_current[surface], surface, layout=layout, rows=rows)
    if "seq_x" in raw_current:
        seq = raw_current["seq_x"]
        _require(isinstance(seq, np.ndarray) and seq.dtype == np.float32
                 and seq.ndim in (2, 3), "SEQ_CURRENT")
        if seq.ndim == 3:
            _require(seq.shape[1] > 0, "EMPTY_HISTORY")
            seq = seq[:, -1, :]
        _require(_bits_equal(seq, snap), "SEQ_LAST_SNAP_NOT_EXACT")
    aliases = layout["native"]["physical_aliases"]
    _require(_bits_equal(snap[:, [a["signal_index"] for a in aliases]],
                         raw_current["ctx_cont"][:, [a["ctx_cont_index"] for a in aliases]]),
             "CURRENT_ALIAS_NOT_EXACT")
    result = np.empty((rows, RAW_WIDTH), dtype=np.float32)
    for spec in contract["surfaces"].values():
        result[:, spec["start"]:spec["stop"]] = raw_current[spec["source_surface"]][:, spec["source_indices"]]
    result[:, -1] = raw_current["ctx_cat"][:, 0]
    _validate_values(result, contract)
    return result


def transform_relative_price_snapshot_block(raw_canonical, *, layout, tick_size_raw):
    """Bounded candidate block; geometry/source IDs remain explicit and separate."""
    from gx1.contracts.gc_relative_price_coordinates_v1 import transform_gc_relative_price_block

    contract = _require_layout(layout)
    _require(isinstance(raw_canonical, np.ndarray) and 0 < len(raw_canonical) <= MAX_BLOCK_ROWS, "BLOCK_ROWS")
    _validate_values(raw_canonical, contract)
    local = contract["surfaces"]["local"]
    geometry = transform_gc_relative_price_block(
        raw_canonical[:, local["start"]:local["stop"]],
        layout=layout["geometry"], tick_size_raw=tick_size_raw)
    result = raw_canonical.copy()
    result[:, local["start"]:local["stop"]] = geometry["local_values"]
    return {"values": result, **{key: geometry[key] for key in (
        "anchor_usd", "anchor_observed", "anchor_source_id", "roundtrip_audit")}}


def _validate_anchor(anchor_usd, anchor_observed, values, contract):
    rows = len(values)
    _require(isinstance(anchor_usd, np.ndarray) and anchor_usd.dtype == np.float32
             and anchor_usd.shape == (rows,) and np.isfinite(anchor_usd).all(), "ANCHOR_VALUES")
    _require(isinstance(anchor_observed, np.ndarray) and anchor_observed.dtype == np.uint8
             and anchor_observed.shape == (rows,) and np.isin(anchor_observed, (0, 1)).all(), "ANCHOR_OBSERVED")
    observed = anchor_observed == 1
    _require((anchor_usd[~observed] == 0).all()
             and (anchor_usd[observed] > 0).all(), "ANCHOR_MASKED_STORAGE")
    for start in range(0, rows, MAX_BLOCK_ROWS):
        masks = values[start:start + MAX_BLOCK_ROWS, contract["level_mask_indices"]]
        _require(np.array_equal(np.any(masks == 1, axis=1),
                                observed[start:start + MAX_BLOCK_ROWS]), "ANCHOR_OBSERVED_PARITY")
    return observed


def _fit_surface(values, spec, lane):
    return fit_surface_normalization(
        values, surface="prefix_snapshot_" + lane, field_names=spec["field_names"],
        row_count=len(values), semantic_binary_fields=spec["binary_fields"],
        semantic_categorical_domains=spec["categorical_domains"],
        allow_constant_train_fields=True, allow_unobserved_train_fields=True)


def fit_prefix_snapshot_normalization(
    prefix_values, prefix_decision_ns, *, cutoff_ns, layout, arm,
    anchor_usd=None, anchor_observed=None,
):
    """Fit only an already qualified prefix, never select it using target masks."""
    contract = _require_layout(layout)
    _require(arm in ARMS, "ARM")
    _validate_values(prefix_values, contract)
    clocks = prefix_decision_ns
    _require(isinstance(clocks, np.ndarray) and clocks.dtype == np.int64
             and clocks.shape == (len(prefix_values),) and len(clocks) >= 2
             and clocks.flags.c_contiguous and np.all(clocks[1:] > clocks[:-1])
             and type(cutoff_ns) is int and np.all(clocks < cutoff_ns), "PREFIX_CLOCKS")
    observed, anchor = None, None
    if arm == "relative":
        observed = _validate_anchor(anchor_usd, anchor_observed, prefix_values, contract)
        _require(np.count_nonzero(observed) >= 2, "NO_OBSERVED_PREFIX_ANCHOR")
        anchor = fit_surface_normalization(
            anchor_usd[observed, None], surface="prefix_snapshot_anchor",
            field_names=[ANCHOR_FIELDS[0]], row_count=int(observed.sum()),
            semantic_binary_fields=[], semantic_categorical_domains={},
            allow_constant_train_fields=True, allow_unobserved_train_fields=True)
    else:
        _require(anchor_usd is None and anchor_observed is None, "BASELINE_ANCHOR")
    surfaces = {lane: _fit_surface(prefix_values[:, spec["start"]:spec["stop"]], spec, lane)
                for lane, spec in contract["surfaces"].items()}
    return _seal({
        "schema_version": SCHEMA_VERSION + "_prefix_normalization",
        "layout_contract_sha256": contract["contract_sha256"], "arm": arm,
        "recipe": dict(PREFIX_RECIPE), "fit_row_count": len(clocks),
        "fit_cutoff_ns": cutoff_ns, "first_decision_ns": int(clocks[0]),
        "last_decision_ns": int(clocks[-1]),
        "decision_clock_sha256": hashlib.sha256(memoryview(clocks).cast("B")).hexdigest(),
        "surfaces": surfaces, "anchor_surface": anchor,
        "anchor_observed_fit_count": 0 if observed is None else int(observed.sum()),
    })


def _validate_surface_stats(stats, *, spec, lane, rows):
    """New recipe validation: native require_surface hardcodes another binary policy."""
    names = spec["field_names"]
    _require(stats["surface"] == "prefix_snapshot_" + lane
             and stats["field_names"] == names and stats["field_count"] == len(names)
             and stats["fit_row_count"] == rows
             and stats["field_names_sha256"] == _field_names_sha256(names)
             and stats["continuous_transform"] == CONTINUOUS_TRANSFORM, "STATS_IDENTITY:" + lane)
    arrays = {key: np.asarray(stats[key], dtype=dtype) for key, dtype in (
        ("center", np.float32), ("scale", np.float32), ("binary_mask", np.uint8),
        ("categorical_mask", np.uint8), ("train_transformed_min", np.float64),
        ("train_transformed_max", np.float64))}
    _require(all(a.shape == (len(names),) and np.isfinite(a).all() for a in arrays.values())
             and (arrays["scale"] > 0).all()
             and np.all(arrays["train_transformed_min"] <= arrays["train_transformed_max"]), "STATS_FINITE:" + lane)
    expected_binary = [int(name in spec["binary_fields"]) for name in names]
    expected_cat = [int(name in spec["categorical_domains"]) for name in names]
    _require(stats["binary_mask"] == expected_binary
             and stats["categorical_mask"] == expected_cat
             and stats["binary_field_count"] == sum(expected_binary)
             and stats["categorical_field_count"] == sum(expected_cat)
             and stats["categorical_domains"] == spec["categorical_domains"], "STATS_DOMAINS:" + lane)
    _require(len(stats["scale_source"]) == len(names), "STATS_SCALE_SOURCE:" + lane)
    counts = stats.get("observed_fit_counts")
    _require(stats.get("allow_unobserved_train_fields") is True
             and isinstance(counts, list) and len(counts) == len(names)
             and all(type(count) is int and 0 <= count <= rows for count in counts),
             "STATS_OBSERVED_COUNTS:" + lane)
    paired = {value for value, _mask in spec.get("masking_pairs", ())}
    for i, name in enumerate(names):
        binary, category = expected_binary[i], expected_cat[i]
        count = counts[i]
        _require(i in paired or count == rows, "STATS_UNPAIRED_COUNT:" + name)
        if count == 0:
            _require(i in paired and arrays["center"][i] == 0 and arrays["scale"][i] == 1
                     and arrays["train_transformed_min"][i] == 0
                     and arrays["train_transformed_max"][i] == 0,
                     "STATS_UNOBSERVED_UNITS:" + name)
        if binary or category:
            expected = "semantic_binary_identity" if binary else "categorical_embedding_identity"
            _require(arrays["center"][i] == 0 and arrays["scale"][i] == 1
                     and stats["scale_source"][i] == expected, "STATS_IDENTITY_UNITS:" + name)
        elif count == 0:
            _require(stats["scale_source"][i] == "unobserved_prefix_unit_scale",
                     "STATS_UNOBSERVED_SOURCE:" + name)
        else:
            _require(stats["scale_source"][i] in (
                "raw_iqr", "median_positive_abs_deviation", "constant_train_unit_scale"),
                "STATS_SCALE_SOURCE:" + name)
            if stats["scale_source"][i] == "constant_train_unit_scale":
                _require(arrays["scale"][i] == 1, "STATS_CONSTANT_SCALE:" + name)
    _require(_stats_sha256(field_names=names, **arrays,
                           categorical_domains=spec["categorical_domains"]) == stats["stats_sha256"],
             "STATS_SHA256:" + lane)


def require_prefix_snapshot_normalization(value, *, layout):
    contract = _require_layout(layout)
    _require_seal(value)
    _require(value["schema_version"] == SCHEMA_VERSION + "_prefix_normalization"
             and value["layout_contract_sha256"] == contract["contract_sha256"]
             and value["recipe"] == PREFIX_RECIPE and value["arm"] in ARMS
             and type(value["fit_row_count"]) is int and value["fit_row_count"] >= 2
             and value["first_decision_ns"] <= value["last_decision_ns"] < value["fit_cutoff_ns"],
             "PREFIX_IDENTITY")
    _require(set(value["surfaces"]) == set(contract["surfaces"]), "SURFACE_SET")
    for lane, spec in contract["surfaces"].items():
        _validate_surface_stats(value["surfaces"][lane], spec=spec, lane=lane, rows=value["fit_row_count"])
    if value["arm"] == "relative":
        rows = value["anchor_observed_fit_count"]
        _require(type(rows) is int and 2 <= rows <= value["fit_row_count"], "ANCHOR_FIT_COUNT")
        _validate_surface_stats(value["anchor_surface"], lane="anchor", rows=rows,
            spec={"field_names": [ANCHOR_FIELDS[0]], "binary_fields": [], "categorical_domains": {}})
    else:
        _require(value["anchor_surface"] is None and value["anchor_observed_fit_count"] == 0, "BASELINE_ANCHOR")
    return value


def encode_normalized_snapshot_block(
    values, *, layout, normalization, anchor_usd=None, anchor_observed=None,
):
    """Apply fixed prefix stats in a bounded block, then fixed-domain onehot."""
    contract = _require_layout(layout)
    norm = require_prefix_snapshot_normalization(normalization, layout=layout)
    _require(isinstance(values, np.ndarray) and 0 < len(values) <= MAX_BLOCK_ROWS, "BLOCK_ROWS")
    _validate_values(values, contract)
    normalized = np.empty_like(values)
    for lane, spec in contract["surfaces"].items():
        sl = slice(spec["start"], spec["stop"])
        normalized[:, sl] = apply_surface_normalization(values[:, sl], norm["surfaces"][lane])
    normalized[:, -1] = values[:, -1]
    result, names = encode_inputs(normalized, contract["raw_columns"], contract["nominal_domains"])
    _require(names == contract["encoded_columns"], "ENCODED_COLUMNS")
    if norm["arm"] == "relative":
        observed = _validate_anchor(anchor_usd, anchor_observed, values, contract)
        transformed = apply_surface_normalization(anchor_usd[:, None], norm["anchor_surface"])
        transformed[~observed, 0] = 0
        result = np.column_stack((result, transformed[:, 0], anchor_observed))
        names = [*names, *ANCHOR_FIELDS]
    else:
        _require(anchor_usd is None and anchor_observed is None, "BASELINE_ANCHOR")
    width = CANDIDATE_WIDTH if norm["arm"] == "relative" else BASELINE_WIDTH
    _require(result.shape == (len(values), width) and np.isfinite(result).all(), "ENCODED_RESULT")
    return result.astype(np.float32), names
