"""Bounded205 common-level GC coordinates; no fit, payload loader or native action.

All242 source-proven levels are transformed; all other local values and every
original mask retain their exact bits. The log uses the selected F64 USD anchor,
while the extra context stores F32 USD. The audit measures kernel/storage/unit
closure, not an independent accuracy threshold or a bit-exact inverse promise.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from gx1.contracts.entry_direction_snapshot_data_v1 import MAX_BLOCK_ROWS
from gx1.contracts.gc_price_coordinate_contract_v1 import SCHEMA_VERSION as SOURCE_SCHEMA
from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256

SCHEMA_VERSION = "gx1_gc_relative_price_layout_v1"
SOURCE_MAP_FILE_SHA256 = "c3dd04288d93d9d78d96ac7444db2f9ba887157142513334445eb3c75a138996"
SOURCE_MAP_CANONICAL_SHA256 = "6c62311ea684dee51f1fc2f394b1fce95d29940eecff5c1c7a3863245f96f148"
LOCAL_WIDTH = 2622
_FLOAT64_ROUNDING_OPERATIONS = 16
_U64 = np.finfo(np.float64).eps / 2
_GAMMA64 = _FLOAT64_ROUNDING_OPERATIONS * _U64 / (1 - _FLOAT64_ROUNDING_OPERATIONS * _U64)


def _require(condition, reason):
    if not condition:
        raise RuntimeError("GC_RELATIVE_PRICE_" + reason)


def compile_gc_relative_price_layout(local_field_names, *, source_contract):
    """Bind the complete canonical local inventory to the immutable204 source map."""
    _require(isinstance(source_contract, Mapping)
        and source_contract.get("schema_version") == SOURCE_SCHEMA
        and canonical_sha256(source_contract) == SOURCE_MAP_CANONICAL_SHA256, "SOURCE_MAP")
    names = list(local_field_names)
    _require(len(names) == LOCAL_WIDTH and len(set(names)) == LOCAL_WIDTH
        and all(isinstance(name, str) and name for name in names), "LOCAL_FIELDS")
    positions = {name: index for index, name in enumerate(names)}
    fields = source_contract["fields"]
    source_names = {"ctx_cont." + item[key] for item in fields for key in ("name", "availability_field")}
    _require(source_names <= set(names) and {name for name in names
        if name.startswith("ctx_cont.gc_")} == source_names, "COMPLETE_GC_FIELDS_AND_MASKS")
    values = [positions["ctx_cont." + item["name"]] for item in fields]
    masks = [positions["ctx_cont." + item["availability_field"]] for item in fields]
    by_name = {item["name"]: item for item in fields}
    priority = list(source_contract["anchor_source_priority"])
    levels = [by_name[name] for name in priority]
    _require(len(fields) == 1205 and len(priority) == 242
        and priority[0] == "gc_book_close_usd" and len(set(priority)) == 242, "SOURCE_COUNTS")
    level_values = [positions["ctx_cont." + item["name"]] for item in levels]
    level_masks = [positions["ctx_cont." + item["availability_field"]] for item in levels]
    units = [item["source_unit"] for item in levels]
    _require(units.count("USD") == 232 and units.count("ticks") == 10, "LEVEL_UNITS")
    level_set = set(level_values)
    value = {"schema_version": SCHEMA_VERSION, "source_map_file_sha256": SOURCE_MAP_FILE_SHA256,
        "source_map_contract_sha256": SOURCE_MAP_CANONICAL_SHA256, "local_field_names": names,
        "local_width": LOCAL_WIDTH, "registry_value_indices": values, "registry_mask_indices": masks,
        "level_value_indices": level_values, "level_mask_indices": level_masks, "level_units": units,
        "anchor_source_priority": priority, "price_scale": source_contract["tick_to_usd"]["price_scale"],
        "unchanged_indices": [index for index in range(LOCAL_WIDTH) if index not in level_set],
        "max_block_rows": MAX_BLOCK_ROWS, "anchor_source_id_is_model_input": False}
    return {**value, "contract_sha256": canonical_sha256(value)}


def _require_layout(layout):
    _require(isinstance(layout, Mapping) and layout.get("schema_version") == SCHEMA_VERSION
        and layout.get("source_map_contract_sha256") == SOURCE_MAP_CANONICAL_SHA256
        and layout.get("local_width") == LOCAL_WIDTH
        and layout.get("max_block_rows") == MAX_BLOCK_ROWS
        and layout.get("contract_sha256") == canonical_sha256(
            {key: value for key, value in layout.items() if key != "contract_sha256"}), "LAYOUT")


def _positive_finite(value, reason):
    _require(np.isfinite(value).all() and (value > 0).all(), reason)


def _group_audit(mask, *, actual_native, bound_native, actual_usd, bound_usd,
        kernel, anchor_component, relative_component, relative_native, ulp_scaled,
        inverse32_different, inverse32_nonfinite):
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {"observed_cells": 0, "max_abs_error_native": None, "max_bound_native": None,
            "max_error_to_positive_bound_ratio": None, "zero_bound_cells": 0,
            "bound_violations": 0, "max_abs_error_usd": None, "max_bound_usd": None,
            "max_kernel_closure_abs_usd": None, "max_anchor_storage_component_usd": None,
            "max_relative_storage_component_usd": None, "max_abs_relative_error_native": None,
            "max_source_native_float32_ulp_scaled_abs_error": None,
            "inverse_to_float32_different_cells": 0, "inverse_to_float32_nonfinite_cells": 0}
    actual, bound = actual_native[mask], bound_native[mask]
    positive = bound > 0
    return {"observed_cells": count, "max_abs_error_native": float(np.max(actual)),
        "max_bound_native": float(np.max(bound)), "max_error_to_positive_bound_ratio": (
            float(np.max(actual[positive] / bound[positive])) if np.any(positive) else None),
        "zero_bound_cells": int(np.count_nonzero(~positive)),
        "bound_violations": int(np.count_nonzero(actual > bound)),
        "max_abs_error_usd": float(np.max(actual_usd[mask])),
        "max_bound_usd": float(np.max(bound_usd[mask])),
        "max_kernel_closure_abs_usd": float(np.max(kernel[mask])),
        "max_anchor_storage_component_usd": float(np.max(anchor_component[mask])),
        "max_relative_storage_component_usd": float(np.max(relative_component[mask])),
        "max_abs_relative_error_native": float(np.max(relative_native[mask])),
        "max_source_native_float32_ulp_scaled_abs_error": float(np.max(ulp_scaled[mask])),
        "inverse_to_float32_different_cells": int(np.count_nonzero(inverse32_different[mask])),
        "inverse_to_float32_nonfinite_cells": int(np.count_nonzero(inverse32_nonfinite[mask]))}


def _roundtrip_audit(*, raw_native, price_usd, anchor64, anchor32, log64, log32, factors, tick_cells):
    """Measured a-posteriori decomposition, independent of a log/exp ULP assumption.

    E=exp64(log64), Es=exp64(F32log), R=fl64(A64*E), Rs=fl64(F32A*Es).
    |Rs-P| is bounded by measured |R-P|, |F32A-A64|*|E|,
    |F32A|*|Es-E| plus gamma16 ordinary-arithmetic allowance. Forward/back
    unit rounding is included in the native-unit bound. Kernel closure is
    reported separately; a large measured closure cannot establish accuracy.
    """
    for value in (raw_native, price_usd, anchor64, anchor32, factors):
        _positive_finite(value, "AUDIT_POSITIVE_FINITE")
    _require(np.isfinite(log64).all() and np.isfinite(log32).all(), "AUDIT_LOG_FINITE")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        exponential = np.exp(log64)
        stored_exponential = np.exp(log32.astype(np.float64))
        _positive_finite(exponential, "AUDIT_EXP_FINITE")
        _positive_finite(stored_exponential, "AUDIT_STORED_EXP_FINITE")
        seed = anchor64 * exponential
        reconstructed_usd = anchor32 * stored_exponential
        _positive_finite(seed, "AUDIT_SEED_FINITE")
        _positive_finite(reconstructed_usd, "AUDIT_RECONSTRUCTED_USD_FINITE")
        reconstructed_native = reconstructed_usd / factors
        _positive_finite(reconstructed_native, "AUDIT_RECONSTRUCTED_NATIVE_FINITE")
        kernel = np.abs(seed - price_usd)
        anchor_component = np.abs(anchor32 - anchor64) * np.abs(exponential)
        relative_component = np.abs(anchor32) * np.abs(stored_exponential - exponential)
        bound_usd = (kernel + anchor_component + relative_component
            + _GAMMA64 * (np.abs(price_usd) + np.abs(seed) + np.abs(reconstructed_usd)))
        bound_native = ((bound_usd + _GAMMA64 * np.abs(price_usd)) / factors
            + _GAMMA64 * np.abs(reconstructed_native))
        actual_usd = np.abs(reconstructed_usd - price_usd)
        actual_native = np.abs(reconstructed_native - raw_native)
    for value in (bound_usd, bound_native, actual_usd, actual_native):
        _require(np.isfinite(value).all() and (value >= 0).all(), "AUDIT_BOUND_FINITE")
    # Positive source F32 values have a finite predecessor, including subnormal
    # and maximum values. This fixed toward-zero ULP convention avoids an
    # infinite nextafter(maxF32,+inf) spacing. Cast overflow is a reported inverse
    # diagnostic; it does not change the finite encoded coordinates or clip them.
    raw32 = raw_native.astype(np.float32)
    predecessor = np.nextafter(raw32, np.float32(0), dtype=np.float32)
    source_ulp = raw_native - predecessor.astype(np.float64)
    _positive_finite(source_ulp, "SOURCE_F32_ULP")
    with np.errstate(over="ignore", under="ignore", invalid="raise", divide="raise"):
        inverse32 = reconstructed_native.astype(np.float32)
    relative_native = actual_native / raw_native
    ulp_scaled = actual_native / source_ulp
    _require(np.isfinite(relative_native).all() and np.isfinite(ulp_scaled).all(), "AUDIT_RELATIVE_OR_ULP_FINITE")
    inverse32_different = inverse32.view(np.uint32) != raw32.view(np.uint32)
    inverse32_nonfinite = ~np.isfinite(inverse32)
    kwargs = dict(actual_native=actual_native, bound_native=bound_native,
        actual_usd=actual_usd, bound_usd=bound_usd, kernel=kernel,
        anchor_component=anchor_component, relative_component=relative_component,
        relative_native=relative_native, ulp_scaled=ulp_scaled,
        inverse32_different=inverse32_different, inverse32_nonfinite=inverse32_nonfinite)
    usd = _group_audit(~tick_cells, **kwargs)
    ticks = _group_audit(tick_cells, **kwargs)
    _require(usd["bound_violations"] == ticks["bound_violations"] == 0, "AUDIT_DECOMPOSITION")
    return {"kind": "measured_kernel_storage_units_closure_not_accuracy_gate",
        "log_anchor": "selected_float64_usd", "stored_anchor": "float32_usd",
        "inverse_storage": "float64_native_units_with_separate_actual_float32_recast_diagnostics",
        "source_native_float32_ulp": "adjacent representable F32 step toward zero; includes subnormal and maximum values",
        "ordinary_float64_rounding_operations": _FLOAT64_ROUNDING_OPERATIONS,
        "gamma64": float(_GAMMA64), "observed_cells": len(raw_native),
        "max_relative_log_storage_abs_error": (float(np.max(np.abs(
            log32.astype(np.float64) - log64))) if len(log64) else None),
        "USD": usd, "ticks": ticks}


def transform_gc_relative_price_block(local_values, *, layout, tick_size_raw):
    """Transform one caller-owned canonical local block, without modifying it."""
    _require_layout(layout)
    _require(isinstance(local_values, np.ndarray) and local_values.dtype == np.float32
        and local_values.ndim == 2 and local_values.shape[1] == LOCAL_WIDTH
        and 0 < local_values.shape[0] <= MAX_BLOCK_ROWS
        and np.isfinite(local_values).all(), "LOCAL_BLOCK")
    _require(type(tick_size_raw) is int and 0 < tick_size_raw <= np.iinfo(np.int64).max,
        "TICK_SIZE_RAW")
    masks = local_values[:, layout["registry_mask_indices"]]
    values = local_values[:, layout["registry_value_indices"]]
    _require(np.isin(masks, (0, 1)).all(), "MASK_DOMAIN")
    _require((values[masks == 0] == 0).all(), "MASKED_STORAGE")
    observed = local_values[:, layout["level_mask_indices"]] == 1
    raw = local_values[:, layout["level_value_indices"]].astype(np.float64)
    _positive_finite(raw[observed], "OBSERVED_LEVEL_NONPOSITIVE_OR_NONFINITE")
    factor = tick_size_raw / layout["price_scale"]
    _positive_finite(np.asarray(factor), "TICK_USD_SCALE")
    tick_columns = np.asarray([unit == "ticks" for unit in layout["level_units"]])
    factors = np.where(tick_columns, factor, 1.0)
    _positive_finite(factors, "UNIT_FACTORS")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        prices = raw * factors[None, :]
    _positive_finite(prices[observed], "OBSERVED_USD_NONPOSITIVE_OR_NONFINITE")
    n = len(local_values)
    anchor_observed = np.any(observed, axis=1)
    row_ids = np.flatnonzero(anchor_observed)
    anchor64 = np.zeros(n, np.float64)
    source_ids = np.full(n, -1, np.int64)
    if len(row_ids):
        chosen = np.argmax(observed[row_ids], axis=1)
        source_ids[row_ids] = chosen
        anchor64[row_ids] = prices[row_ids, chosen]
    _positive_finite(anchor64[row_ids], "ANCHOR64")
    _require((anchor64[row_ids] <= np.finfo(np.float32).max).all(), "ANCHOR32_OVERFLOW")
    anchor = anchor64.astype(np.float32)
    _positive_finite(anchor[row_ids], "ANCHOR32_UNREPRESENTABLE")
    rows, columns = np.nonzero(observed)
    p = prices[rows, columns]
    a = anchor64[rows]
    _positive_finite(p, "LOG_PRICE")
    _positive_finite(a, "LOG_ANCHOR")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
        ratio = p / a
        _positive_finite(ratio, "LOG_RATIO")
        relative64 = np.log(ratio)
    _require(np.isfinite(relative64).all(), "RELATIVE64_FINITE")
    relative = relative64.astype(np.float32)
    _require(np.isfinite(relative).all(), "RELATIVE32_FINITE")
    output = np.array(local_values, copy=True, order="K")
    output[rows, np.asarray(layout["level_value_indices"])[columns]] = relative
    untouched = layout["unchanged_indices"]
    _require(np.array_equal(output[:, untouched].view(np.uint32),
        local_values[:, untouched].view(np.uint32)), "UNCHANGED_BITS")
    audit = _roundtrip_audit(raw_native=raw[rows, columns], price_usd=p,
        anchor64=a, anchor32=anchor[rows].astype(np.float64), log64=relative64,
        log32=relative, factors=factors[columns], tick_cells=tick_columns[columns])
    audit["observed_anchor_rows"] = int(np.count_nonzero(anchor_observed))
    audit["unobserved_anchor_rows"] = int(n - np.count_nonzero(anchor_observed))
    audit["max_anchor_storage_abs_error_usd"] = (float(np.max(np.abs(
        anchor[row_ids].astype(np.float64) - anchor64[row_ids]))) if len(row_ids) else None)
    return {"local_values": output, "anchor_usd": anchor,
        "anchor_observed": anchor_observed.astype(np.uint8), "anchor_source_id": source_ids,
        "roundtrip_audit": audit}
