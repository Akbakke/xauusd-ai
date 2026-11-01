"""Source-bound snapshot/native GC coordinates; no model, fit or payload loader.

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
NATIVE_SCHEMA_VERSION = "gx1_gc_native_relative_price_layout_v1"
NATIVE_SOURCE_MAP_CANONICAL_SHA256 = "f54f3ada955cea9206905bc6261cb25fcac55cba87f863818ead7f443abbc251"
NATIVE_ANCHOR_FIELDS = ("gc_relative_anchor_usd", "gc_relative_anchor_observed")
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
    return _transform_relative_price_values(local_values, layout=layout, tick_size_raw=tick_size_raw)


def _transform_relative_price_values(local_values, *, layout, tick_size_raw):
    """Shared source-unit/anchor/log kernel for snapshot and native row adapters."""
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


def compile_gc_native_relative_price_layout(*, history_layout, source_contract, tick_size_raw):
    """Bind own-TF raw row coordinates; append the real anchor and its mask.

    The existing native selector remains the sole row/clock owner. Each row's
    anchor is selected only from that same row and TF, by the existing immutable
    source priority. No current-M5 price is joined into historical TF rows.
    The two added model columns belong to the first priority source's declared
    family; this assignment is fixed even when another observed source anchors
    a row. The source ID remains provenance, never a model input.
    """
    from gx1.contracts.entry_gc_mtf_history_v1 import require_gc_mtf_history_layout
    raw = require_gc_mtf_history_layout(history_layout)
    _require(isinstance(source_contract, Mapping)
        and source_contract.get("schema_version") == SOURCE_SCHEMA
        and source_contract.get("source_mode") == "native_current"
        and canonical_sha256(source_contract) == NATIVE_SOURCE_MAP_CANONICAL_SHA256, "NATIVE_SOURCE_MAP")
    _require(type(tick_size_raw) is int and 0 < tick_size_raw <= np.iinfo(np.int64).max,
        "TICK_SIZE_RAW")
    records = {item["name"]: item for item in source_contract["fields"]}
    source_names = set(records)
    surfaces = {}
    consumed_sources = set()
    for tf in raw["timeframes"]:
        names = list(raw["field_names"][tf])
        _require(not set(NATIVE_ANCHOR_FIELDS).intersection(names), "NATIVE_ANCHOR_NAME_COLLISION")
        positions = {name: index for index, name in enumerate(names)}
        fields = [item for item in raw["source_fields"][tf] if item["name"] in source_names]
        for item in fields:
            record = records[item["name"]]
            _require(all(item[key] == record[key]
                for key in ("table", "column", "availability_field", "owner")), "NATIVE_SOURCE_MAPPING")
            _require(item["name"] not in consumed_sources, "NATIVE_SOURCE_DUPLICATE")
            consumed_sources.add(item["name"])
        local_sources = {item["name"] for item in fields}
        priority = [name for name in source_contract["anchor_source_priority"] if name in local_sources]
        _require(bool(priority), "NATIVE_LEVELS:" + tf)
        levels = [records[name] for name in priority]
        values = [positions[item["name"]] for item in levels]
        masks = [positions[item["availability_field"]] for item in levels]
        units = [item["source_unit"] for item in levels]
        _require(all(unit in ("USD", "ticks") for unit in units), "NATIVE_LEVEL_UNITS")
        width = len(names)
        anchor_family = levels[0]["owner"]
        family_indices = {name: list(indices) for name, indices in raw["family_indices"][tf].items()}
        _require(anchor_family in family_indices, "NATIVE_ANCHOR_FAMILY")
        family_indices[anchor_family].extend((width, width + 1))
        pairs = raw["value_mask_pairs"][tf]
        surfaces[tf] = {"raw_field_names": names,
            "model_field_names": names + list(NATIVE_ANCHOR_FIELDS),
            "raw_width": width, "model_width": width + 2,
            "family_indices": family_indices,
            "registry_value_indices": [pair[0] for pair in pairs],
            "registry_mask_indices": [pair[1] for pair in pairs],
            "binary_indices": [positions[name] for name in raw["binary_fields"][tf]],
            "level_value_indices": values, "level_mask_indices": masks, "level_units": units,
            "unchanged_indices": [index for index in range(width) if index not in set(values)],
            "anchor_source_priority": priority, "anchor_family": anchor_family,
            "anchor_value_index": width, "anchor_mask_index": width + 1,
            "price_scale": source_contract["tick_to_usd"]["price_scale"]}
    _require(consumed_sources == source_names
        and sum(len(s["level_value_indices"]) for s in surfaces.values())
            == len(source_contract["anchor_source_priority"]), "NATIVE_COMPLETE_SOURCE_MAPPING")
    value = {"schema_version": NATIVE_SCHEMA_VERSION, "timeframes": list(raw["timeframes"]),
        "raw_layout_sha256": raw["contract_sha256"],
        "source_mode": "native_current", "source_map": dict(source_contract),
        "source_sha256": dict(source_contract["source_sha256"]),
        "source_map_contract_sha256": NATIVE_SOURCE_MAP_CANONICAL_SHA256,
        "tick_size_raw": tick_size_raw, "price_scale": source_contract["tick_to_usd"]["price_scale"],
        "surfaces": surfaces, "anchor_source_id_is_model_input": False,
        "anchor_row_scope": "same_caller_selected_native_timeframe_row",
        "anchor_family_policy": "first_source_priority_declared_family",
        "transformed_scope": "five_native_gc_history_surfaces_only; current_context_and_signal_unchanged",
        "max_block_rows": MAX_BLOCK_ROWS}
    return {**value, "contract_sha256": canonical_sha256(value)}


def require_gc_native_relative_price_layout(value, *, history_layout):
    """Recompile source-bound metadata once at the caller's admission boundary."""
    from gx1.contracts.gc_price_coordinate_contract_v1 import build_gc_price_coordinate_source_contract
    _require(isinstance(value, Mapping), "NATIVE_LAYOUT")
    expected = compile_gc_native_relative_price_layout(history_layout=history_layout,
        source_contract=build_gc_price_coordinate_source_contract(source_mode="native_current"),
        tick_size_raw=value.get("tick_size_raw"))
    _require(dict(value) == expected, "NATIVE_LAYOUT_CANONICAL")
    return expected


def _native_surface(layout, timeframe):
    # The full source/layout join is validated by require_gc_native_relative_price_layout
    # at metadata/constructor admission. The hot path does no source-file I/O.
    _require(isinstance(layout, Mapping) and layout.get("schema_version") == NATIVE_SCHEMA_VERSION
        and layout.get("source_mode") == "native_current"
        and layout.get("source_map_contract_sha256") == NATIVE_SOURCE_MAP_CANONICAL_SHA256
        and layout.get("max_block_rows") == MAX_BLOCK_ROWS
        and layout.get("contract_sha256") == canonical_sha256(
            {key: value for key, value in layout.items() if key != "contract_sha256"}), "NATIVE_LAYOUT")
    _require(timeframe in layout["timeframes"] and timeframe in layout["surfaces"], "NATIVE_TIMEFRAME")
    _require(type(layout["tick_size_raw"]) is int
        and 0 < layout["tick_size_raw"] <= np.iinfo(np.int64).max, "TICK_SIZE_RAW")
    return layout["surfaces"][timeframe]


def transform_gc_native_relative_price_block(raw_values, *, layout, timeframe):
    """Transform a bounded pure NumPy block; raw fields/row clocks are unchanged.

    The returned values have two appended columns. The anchor is retained as a
    model input, so this is not a claim that every absolute GC level is removed.
    Missing levels stay unknown with their original storage bits and masks.
    """
    surface = _native_surface(layout, timeframe)
    _require(isinstance(raw_values, np.ndarray) and raw_values.dtype == np.float32
        and raw_values.ndim == 2 and raw_values.shape[1] == surface["raw_width"]
        and 0 < len(raw_values) <= MAX_BLOCK_ROWS and np.isfinite(raw_values).all(), "NATIVE_BLOCK")
    _require(np.isin(raw_values[:, surface["binary_indices"]], (0, 1)).all(), "NATIVE_BINARY_DOMAIN")
    result = _transform_relative_price_values(raw_values, layout=surface, tick_size_raw=layout["tick_size_raw"])
    model_values = np.concatenate((result["local_values"], result["anchor_usd"][:, None],
        result["anchor_observed"].astype(np.float32)[:, None]), axis=1)
    _require(model_values.shape == (len(raw_values), surface["model_width"])
        and model_values.dtype == np.float32, "NATIVE_MODEL_BLOCK")
    return {"values": model_values, "anchor_usd": result["anchor_usd"],
        "anchor_observed": result["anchor_observed"], "anchor_source_id": result["anchor_source_id"],
        "roundtrip_audit": result["roundtrip_audit"]}


def transform_gc_native_relative_price_tensor(raw_values, *, layout, timeframe):
    """Differentiable Torch counterpart; no model, fitting or normalization.

    Native batch/window geometry is caller-owned and can exceed the CPU audit's
    block-row cap. This function uses FP64 for unit conversion/log and stores
    F32 coordinates, exactly the numerical roles of the NumPy owner. The pure
    array audit measures F32 round-trip; CUDA kernel parity remains a separate
    qualification, not a claim made by this implementation.
    """
    import torch
    surface = _native_surface(layout, timeframe)
    _require(isinstance(raw_values, torch.Tensor) and raw_values.dtype == torch.float32
        and raw_values.ndim >= 2 and raw_values.shape[-1] == surface["raw_width"]
        and raw_values.numel() > 0 and bool(torch.isfinite(raw_values).all()), "NATIVE_TENSOR")
    flat = raw_values.reshape(-1, surface["raw_width"])
    masks = flat[:, surface["registry_mask_indices"]]
    values = flat[:, surface["registry_value_indices"]]
    binary = flat[:, surface["binary_indices"]]
    _require(bool(((masks == 0) | (masks == 1)).all())
        and bool(((binary == 0) | (binary == 1)).all()), "NATIVE_BINARY_DOMAIN")
    _require(bool((values[masks == 0] == 0).all()), "MASKED_STORAGE")
    observed = flat[:, surface["level_mask_indices"]] == 1
    raw = flat[:, surface["level_value_indices"]].to(torch.float64)
    _require(bool((raw[observed] > 0).all()), "OBSERVED_LEVEL_NONPOSITIVE_OR_NONFINITE")
    factor = layout["tick_size_raw"] / layout["price_scale"]
    factors = raw.new_tensor([factor if unit == "ticks" else 1.0 for unit in surface["level_units"]])
    prices = raw * factors
    _require(bool(torch.isfinite(prices[observed]).all()) and bool((prices[observed] > 0).all()),
        "OBSERVED_USD_NONPOSITIVE_OR_NONFINITE")
    anchor_observed = observed.any(dim=1)
    chosen = observed.to(torch.int64).argmax(dim=1)
    anchor64 = prices.gather(1, chosen[:, None]).squeeze(1)
    anchor64 = torch.where(anchor_observed, anchor64, torch.zeros_like(anchor64))
    _require(bool((anchor64[anchor_observed] <= torch.finfo(torch.float32).max).all()), "ANCHOR32_OVERFLOW")
    anchor = anchor64.to(torch.float32)
    _require(bool((anchor[anchor_observed] > 0).all()) and bool(torch.isfinite(anchor).all()),
        "ANCHOR32_UNREPRESENTABLE")
    rows, columns = torch.nonzero(observed, as_tuple=True)
    ratio = prices[rows, columns] / anchor64[rows]
    _require(bool(torch.isfinite(ratio).all()) and bool((ratio > 0).all()), "LOG_RATIO")
    relative = torch.log(ratio).to(torch.float32)
    _require(bool(torch.isfinite(relative).all()), "RELATIVE32_FINITE")
    output = flat.clone()
    level_indices = torch.tensor(surface["level_value_indices"], dtype=torch.int64, device=flat.device)
    output[rows, level_indices[columns]] = relative
    combined = torch.cat((output, anchor[:, None], anchor_observed.to(torch.float32)[:, None]), dim=1)
    return combined.reshape(*raw_values.shape[:-1], surface["model_width"])
