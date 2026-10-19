"""Canonical emitted future-target surface for model-native XAU Entry."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION = "entry_model_native_aux_targets_v7"
MODEL_NATIVE_AUX_FORECAST_HORIZONS = (1, 5, 12, 24)
# Same existing forecast task/head: four M5 returns plus independently masked
# executable LONG/SHORT markouts on the operator-retained1/5/20-day horizons.
MODEL_NATIVE_SUPPORT_HORIZONS_TRADING_DAYS = (1, 5, 20)
MODEL_NATIVE_SUPPORT_FORECAST_LAYOUT = tuple(
    {"horizon_trading_days": horizon, "direction": side,
     "target_semantics": "observed_executable_terminal_net_cash_markout_support_only"}
    for horizon in MODEL_NATIVE_SUPPORT_HORIZONS_TRADING_DAYS for side in ("long", "short"))
MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT = tuple(
    {"horizon_m5_bars": horizon, "target_column": f"y_forecast_ret_K{horizon}",
     "target_semantics": "observed_m5_close_return_bps"}
    for horizon in MODEL_NATIVE_AUX_FORECAST_HORIZONS) + MODEL_NATIVE_SUPPORT_FORECAST_LAYOUT
MODEL_NATIVE_FORECAST_OUTPUT_DIM = len(MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT)
MODEL_NATIVE_SUPPORT_FORECAST_DIM = len(MODEL_NATIVE_SUPPORT_FORECAST_LAYOUT)
SUPPORT_FORECAST_TARGET_BATCH_KEY = "entry_observed_market_support_forecast_bps"
SUPPORT_FORECAST_VALID_BATCH_KEY = "entry_observed_market_support_forecast_valid"
LEGACY_COMPLETE_EMISSION_SCOPE = "legacy_complete"
M5_PRIMARY_OBSERVED_EMISSION_SCOPE = "m5_primary_observed"
M5_DYNAMIC_OBSERVED_EMISSION_SCOPE = "m5_dynamic_observed"
M5_DYNAMIC_EMISSION_BATCH_KEY = "entry_m5_dynamic_emission"
M5_DYNAMIC_PHYSICAL_TARGET_BINDING_SCHEMA = "gx1_m5_dynamic_physical_target_binding_v1"
DYNAMIC_M5_PHYSICAL_BINDING_MODE = "original_physical_dynamic_curve_pointers"
DYNAMIC_M5_POINTER_COLUMNS = (
    "dynamic_entry_m1_row", "dynamic_first_prefix_observed",
    "dynamic_first_terminal_observed_long", "dynamic_first_terminal_observed_short",
    "dynamic_first_path_observed_long", "dynamic_first_path_observed_short",
    "dynamic_first_terminal_censor_reason", "dynamic_first_path_censor_reason",
)
DYNAMIC_M5_QUOTE_COLUMNS = (
    "time", "bid_open", "bid_high", "bid_low", "ask_open", "ask_high", "ask_low",
)
M5_FORECAST_VALID_BATCH_KEY = "entry_observed_market_m5_forecast_valid"
M5_PRIMARY_EMISSION_BATCH_KEY = "entry_m5_primary_emission"
M5_PHYSICAL_TARGET_BINDING_SCHEMA = "gx1_m5_primary_physical_target_binding_v1"
LEGACY_CAUSAL_M1_OBSERVED_COLUMN = "legacy_causal_m1_label_observed"
LEGACY_AUX_COMPLETE_COLUMN = "legacy_aux_target_complete"
MODEL_NATIVE_M5_EMISSION_PROOF_KEYS = (
    "candidate_rows_before_primary_availability",
    "primary_unavailable_candidate_rows_excluded",
    "primary_valid_rows_emitted",
    "legacy_causal_m1_observed_rows_emitted",
    "legacy_aux_complete_rows_emitted",
    "forecast_observed_rows_by_column",
)
MODEL_NATIVE_AUX_RISK_HORIZONS = (12, 48, 96)
MODEL_NATIVE_DIP_DIRECTIONS = ("long", "short")
MODEL_NATIVE_DIP_OUTPUT_TARGETS = ("dip_p50", "dip_p90", "recovery_p50")
MODEL_NATIVE_DIP_OUTPUT_DIM = (
    len(MODEL_NATIVE_DIP_DIRECTIONS)
    * len(MODEL_NATIVE_AUX_RISK_HORIZONS)
    * len(MODEL_NATIVE_DIP_OUTPUT_TARGETS)
)
MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS = tuple(
    f"y_dip_mae_{side}_K{horizon}"
    for side in MODEL_NATIVE_DIP_DIRECTIONS
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
)
MODEL_NATIVE_DIP_MFE_TARGET_COLUMNS = tuple(
    f"y_dip_mfe_{side}_K{horizon}"
    for side in MODEL_NATIVE_DIP_DIRECTIONS
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
)
MODEL_NATIVE_DIP_MFE_UPPER_SAFETY_CAP_BPS = 1000.0
MODEL_NATIVE_DIP_MAE_UPPER_SAFETY_CAP_BPS = 1000.0
# Full-path adverse risk covers the complete 96-bar window, unlike dip MAE
# which stops at the first favorable peak.  Keep its domain explicit and
# wider than the short-horizon dip caps; this is a validation bound, never a
# clipping instruction.
MODEL_NATIVE_TAIL_MAE_UPPER_SAFETY_CAP_BPS = 1500.0
MODEL_NATIVE_DIP_TARGET_COLUMNS = (
    *MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS,
    *MODEL_NATIVE_DIP_MFE_TARGET_COLUMNS,
)
MODEL_NATIVE_FORECAST_TARGET_COLUMNS = tuple(
    f"y_forecast_ret_K{horizon}"
    for horizon in MODEL_NATIVE_AUX_FORECAST_HORIZONS
)
MODEL_NATIVE_FORECAST_VALID_COLUMNS = tuple(
    name + "_observed" for name in MODEL_NATIVE_FORECAST_TARGET_COLUMNS
)
MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS = tuple(
    f"y_tail_mae_{side}_K{horizon}"
    for side in ("long", "short")
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
)
MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS = tuple(
    f"y_vol_fwd_K{horizon}" for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
)
MODEL_NATIVE_TIMING_DIRECTIONS = ("long", "short")
MODEL_NATIVE_TIMING_TARGETS = ("dip_bottom_frac", "time_to_mfe_frac")
MODEL_NATIVE_TURN_KIND_BY_DIRECTION = MappingProxyType(
    {"long": "BOTTOM", "short": "TOP"}
)
MODEL_NATIVE_TIMING_TARGET_COLUMNS = tuple(
    f"y_{target}_{direction}_K{horizon}"
    for direction in MODEL_NATIVE_TIMING_DIRECTIONS
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
    for target in MODEL_NATIVE_TIMING_TARGETS
)
MODEL_NATIVE_TIMING_OUTPUT_DIM = len(MODEL_NATIVE_TIMING_TARGET_COLUMNS)
MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS: tuple[str, ...] = ()
_TARGET_HORIZON_ITEMS = tuple(
    (name, horizon)
    for name in MODEL_NATIVE_DIP_TARGET_COLUMNS
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
    if name.endswith(f"_K{horizon}")
) + tuple(
    (name, horizon)
    for name in MODEL_NATIVE_FORECAST_TARGET_COLUMNS
    for horizon in MODEL_NATIVE_AUX_FORECAST_HORIZONS
    if name.endswith(f"_K{horizon}")
) + tuple(
    (name, horizon)
    for name in MODEL_NATIVE_TIMING_TARGET_COLUMNS
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
    if name.endswith(f"_K{horizon}")
) + tuple(
    (name, horizon)
    for name in MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
    if name.endswith(f"_K{horizon}")
) + tuple(
    (name, horizon)
    for name in MODEL_NATIVE_VOL_FORECAST_TARGET_COLUMNS
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
    if name.endswith(f"_K{horizon}")
)
MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN = MappingProxyType(
    dict(_TARGET_HORIZON_ITEMS)
)
MODEL_NATIVE_AUX_TARGET_COLUMNS = tuple(MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN)
MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS = max(
    MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN.values()
)
MODEL_NATIVE_AUX_EMISSION_PROOF_KEYS = (
    "incomplete_tail_rows_total",
    "candidate_rows_before_completeness",
    "incomplete_candidate_rows_excluded",
    "complete_rows_emitted",
)


def model_native_aux_target_contract_metadata(
    *, emission_scope: str = LEGACY_COMPLETE_EMISSION_SCOPE,
) -> dict[str, Any]:
    if emission_scope not in (LEGACY_COMPLETE_EMISSION_SCOPE, M5_PRIMARY_OBSERVED_EMISSION_SCOPE, M5_DYNAMIC_OBSERVED_EMISSION_SCOPE):
        raise RuntimeError("ENTRY_AUX_TARGET_EMISSION_SCOPE_INVALID")
    timing_layout = []
    index = 0
    for direction in MODEL_NATIVE_TIMING_DIRECTIONS:
        for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS:
            for target in MODEL_NATIVE_TIMING_TARGETS:
                timing_layout.append(
                    {
                        "index": index,
                        "direction": direction,
                        "market_turn": MODEL_NATIVE_TURN_KIND_BY_DIRECTION[direction],
                        "horizon_bars": horizon,
                        "target": target,
                        "target_column": f"y_{target}_{direction}_K{horizon}",
                    }
                )
                index += 1
    result = {
        "schema_version": MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION,
        "columns": list(MODEL_NATIVE_AUX_TARGET_COLUMNS),
        "forecast_output_layout": list(MODEL_NATIVE_FORECAST_OUTPUT_LAYOUT),
        "support_forecast_target_batch_key": SUPPORT_FORECAST_TARGET_BATCH_KEY,
        "support_forecast_valid_batch_key": SUPPORT_FORECAST_VALID_BATCH_KEY,
        "support_forecast_masked_per_horizon": True,
        "support_forecast_primary_entry_authority": False,
        "future_horizon_bars_by_column": {
            name: int(horizon)
            for name, horizon in MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN.items()
        },
        "max_future_horizon_bars": int(MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS),
        "spread_aware_risk_magnitudes_required": True,
        "target_value_domains": {
            "dip_mfe": {
                "columns": list(MODEL_NATIVE_DIP_MFE_TARGET_COLUMNS),
                "unit": "bps",
                "finite_on_complete_rows": True,
                "signed": True,
                "negative_values_preserved": True,
                "lower_bound_bps": None,
                "upper_safety_cap_bps": float(
                    MODEL_NATIVE_DIP_MFE_UPPER_SAFETY_CAP_BPS
                ),
            },
            "dip_mae": {
                "columns": list(MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS),
                "unit": "bps",
                "finite_on_complete_rows": True,
                "signed": False,
                "lower_bound_bps": 0.0,
                "upper_safety_cap_bps": float(
                    MODEL_NATIVE_DIP_MAE_UPPER_SAFETY_CAP_BPS
                ),
            },
            "tail_mae": {
                "columns": list(MODEL_NATIVE_TAIL_RISK_TARGET_COLUMNS),
                "unit": "bps",
                "finite_on_complete_rows": True,
                "signed": False,
                "lower_bound_bps": 0.0,
                "upper_safety_cap_bps": float(
                    MODEL_NATIVE_TAIL_MAE_UPPER_SAFETY_CAP_BPS
                ),
            },
        },
        "mid_price_timing_reference_only": True,
        "incomplete_value": "NaN_before_emission_only",
        "incomplete_rows_may_be_emitted": False,
        "turning_point_timing": {
            "output_name": "timing_pred",
            "output_dim": MODEL_NATIVE_TIMING_OUTPUT_DIM,
            "layout": timing_layout,
            "long_semantics": "adverse_low_before_favorable_peak_BOTTOM",
            "short_semantics": "adverse_high_before_favorable_trough_TOP",
            "live_direction_rule_authority": False,
            "final_fusion_evidence_required": True,
        },
        "offline_rl": "retired_replaced_by_entry_fitted_q",
        "extra_active_target_heads": list(MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS),
    }
    if emission_scope in (M5_PRIMARY_OBSERVED_EMISSION_SCOPE, M5_DYNAMIC_OBSERVED_EMISSION_SCOPE):
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_M5_PATH_TRAINING_PHASE
        result.update({
            "schema_version": "entry_model_native_aux_targets_m5_primary_emission_v1",
            "emission_scope": emission_scope,
            "training_phase": ENTRY_ONLY_M5_PATH_TRAINING_PHASE,
            "forecast_valid_columns": list(MODEL_NATIVE_FORECAST_VALID_COLUMNS),
            "m5_forecast_valid_batch_key": M5_FORECAST_VALID_BATCH_KEY,
            "legacy_causal_m1_observed_column": LEGACY_CAUSAL_M1_OBSERVED_COLUMN,
            "legacy_aux_complete_column": LEGACY_AUX_COMPLETE_COLUMN,
            "legacy_targets_phase_inactive": True,
            "active_m5_forecast_target_columns": list(MODEL_NATIVE_FORECAST_TARGET_COLUMNS),
            "inactive_legacy_aux_target_columns": [name for name in MODEL_NATIVE_AUX_TARGET_COLUMNS
                                                    if name not in MODEL_NATIVE_FORECAST_TARGET_COLUMNS],
            "primary_availability_independent_of_legacy_and_forecast_masks": True,
            "incomplete_value": "finite_zero_storage_behind_false_observation_mask",
            "incomplete_rows_may_be_emitted": True,
        })
    if emission_scope == M5_DYNAMIC_OBSERVED_EMISSION_SCOPE:
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
        from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
        result.update({
            "schema_version": "entry_model_native_aux_targets_m5_dynamic_emission_v1",
            "training_phase": ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
            "dynamic_target_contract": entry_observed_dynamic_m5_path_contract(),
            "dynamic_emission_batch_key": M5_DYNAMIC_EMISSION_BATCH_KEY,
            "primary_target_source": "bound_original_M1_quote_prefix_queries",
            "primary_availability_independent_of_legacy_and_forecast_masks": True,
        })
    return result


def require_auxiliary_emission_training_phase(
    value: Mapping[str, Any], *, entry_training_phase: str, context: str,
) -> None:
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    scope = value.get("emission_scope", LEGACY_COMPLETE_EMISSION_SCOPE)
    if (scope == M5_DYNAMIC_OBSERVED_EMISSION_SCOPE) != (entry_training_phase == ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE):
        raise RuntimeError(f"[{context}_DYNAMIC_M5_EMISSION_PHASE_MISMATCH]")
    if scope == M5_PRIMARY_OBSERVED_EMISSION_SCOPE:
        from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_M5_PATH_TRAINING_PHASE
        if entry_training_phase != ENTRY_ONLY_M5_PATH_TRAINING_PHASE:
            raise RuntimeError(f"[{context}_M5_EMISSION_PHASE_MISMATCH]")


def require_m5_primary_physical_target_receipt(
    value: Mapping[str, Any], *, design_binding: Mapping[str, str], context: str,
) -> dict[str, Any]:
    """Bind the current physical target sources, never accept a declared PASS."""
    from pathlib import Path
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_M5_PATH_TRAINING_PHASE
    from gx1.contracts.local_random_access_campaign_v2 import require_binding
    from gx1.features import gc_native_context_v1 as gc
    if (not isinstance(value, Mapping)
            or set(value) != {"schema_version", "design", "registry", "training_phase", "emission_scope", "test_accessed", "forbidden_access_attempts"}
            or value["schema_version"] != M5_PHYSICAL_TARGET_BINDING_SCHEMA
            or value["design"] != design_binding
            or value["training_phase"] != ENTRY_ONLY_M5_PATH_TRAINING_PHASE
            or value["emission_scope"] != M5_PRIMARY_OBSERVED_EMISSION_SCOPE
            or value["test_accessed"] is not False or value["forbidden_access_attempts"] != []):
        raise RuntimeError(f"[{context}_M5_PHYSICAL_TARGET_RECEIPT_INVALID]")
    require_binding(value["design"], label="M5 physical target design", verify_file=True)
    registry = require_binding(value["registry"], label="M5 current TRAIN registry", verify_file=True)
    if Path(registry["path"]) != Path(gc.REGISTRY_PATH) or registry["sha256"] != gc.REGISTRY_SHA256:
        raise RuntimeError(f"[{context}_M5_PHYSICAL_TARGET_REGISTRY_MISMATCH]")
    return dict(value)



def require_m5_dynamic_physical_target_receipt(
    value: Mapping[str, Any], *, design_binding: Mapping[str, str], context: str,
) -> dict[str, Any]:
    """Rejoin the immutable physical clock/pointers and their first-prefix proof.

    This checks every stored pointer/mask cell in bounded Arrow batches. The
    original quote-value/first-prefix reconstruction remains with the dynamic
    dataset provider; a self-declared physical receipt cannot replace it.
    """
    import hashlib
    import json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.contracts.entry_causal_m1_outcomes_v1 import dynamic_m1_prefix_contract
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.entry_observed_market_v1 import entry_observed_dynamic_m5_path_contract
    from gx1.contracts.local_random_access_campaign_v2 import require_binding, read_bound_json
    from gx1.features import gc_native_context_v1 as gc

    def fail(reason):
        raise RuntimeError(f"[{context}_M5_DYNAMIC_PHYSICAL_TARGET_{reason}]")

    def sha(raw):
        return (isinstance(raw, str) and len(raw) == 64
                and all(c in "0123456789abcdef" for c in raw))

    def utc(raw):
        stamp = pd.Timestamp(raw)
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.floor("min") != stamp:
            fail("CLOCK_INVALID")
        return stamp.tz_convert("UTC")

    required = {"schema_version", "design", "registry", "training_phase", "emission_scope",
                "target_contract", "splits", "test_accessed", "forbidden_access_attempts"}
    if (not isinstance(value, Mapping) or set(value) != required
            or value["schema_version"] != M5_DYNAMIC_PHYSICAL_TARGET_BINDING_SCHEMA
            or value["design"] != design_binding
            or value["training_phase"] != ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
            or value["emission_scope"] != M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
            or value["target_contract"] != entry_observed_dynamic_m5_path_contract()
            or value["test_accessed"] is not False or value["forbidden_access_attempts"] != []
            or not isinstance(value["splits"], Mapping) or set(value["splits"]) != {"train", "val"}):
        fail("RECEIPT_INVALID")
    design_file = require_binding(value["design"], label="dynamic M5 physical design", verify_file=True)
    registry = require_binding(value["registry"], label="dynamic M5 current TRAIN registry", verify_file=True)
    registry_path = (Path(gc.__file__).resolve().parents[2] / gc.REGISTRY_PATH).resolve(strict=True)
    if Path(registry["path"]) != registry_path or registry["sha256"] != gc.REGISTRY_SHA256:
        fail("REGISTRY_MISMATCH")
    design = read_bound_json(Path(design_file["path"]), design_file["sha256"])
    if design.get("scope", {}).get("entry_only_training_phase") != ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE:
        fail("PHASE_MISMATCH")
    calendar = design.get("calendar", {})
    sources = calendar.get("source_bindings", {})
    if not isinstance(sources, Mapping) or not {"train", "val"}.issubset(sources):
        fail("SOURCES_INVALID")
    shared_quote = None
    for split in ("train", "val"):
        source = sources[split]
        if not isinstance(source, Mapping):
            fail("SOURCE_INVALID")
        parquet_binding = require_binding(source.get("parquet"), label=f"dynamic M5 {split} physical pointers", verify_file=True)
        manifest_binding = require_binding(source.get("manifest"), label=f"dynamic M5 {split} manifest", verify_file=True)
        manifest = read_bound_json(Path(manifest_binding["path"]), manifest_binding["sha256"])
        extra = manifest.get("extra", {})
        emitted = extra.get("aux_head_target_contract")
        static = require_model_native_aux_target_emission_contract(emitted, context=context)
        rows = source.get("physical_rows")
        if (type(rows) is not int or rows <= 0 or not sha(source.get("clock_sha256"))
                or extra.get("rows") != rows or extra.get("emission_scope") != M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
                or static.get("emission_scope") != M5_DYNAMIC_OBSERVED_EMISSION_SCOPE
                or emitted["primary_valid_rows_emitted"] != rows
                or extra.get("gc_native_context") != gc.contract()):
            fail("MANIFEST_INVALID")
        cutoff = utc(calendar.get("train_control_cutoff" if split == "train"
                                  else "development_control_entry_end_exclusive"))
        meta = extra.get("dynamic_m5_supervision", {})
        meta_keys = {"schema_version", "contract", "source_quote_binding", "source_quote_columns",
            "index_binding", "pointer_columns", "pointer_row_clock", "first_prefix_horizon_minutes",
            "first_terminal_endpoint_kind", "first_path_endpoint_kind", "physical_population_rule",
            "later_horizon_availability_is_not_an_emission_gate", "closure_mode", "target_only_columns",
            "feature_inputs_include_future_masks", "dense_future_paths_materialized"}
        if (not isinstance(meta, Mapping) or set(meta) != meta_keys
                or meta["schema_version"] != "gx1_dynamic_m5_physical_target_pointers_v1"
                or meta["contract"] != dynamic_m1_prefix_contract()
                or meta["source_quote_columns"] != list(DYNAMIC_M5_QUOTE_COLUMNS)
                or meta["pointer_columns"] != list(DYNAMIC_M5_POINTER_COLUMNS)
                or meta["pointer_row_clock"] != "global_row_in_bound_shared_m1_quote_parquet"
                or type(meta["first_prefix_horizon_minutes"]) is not int or meta["first_prefix_horizon_minutes"] != 1
                or meta["first_terminal_endpoint_kind"] != "terminal_open_inclusive"
                or meta["first_path_endpoint_kind"] != "closed_m1_prefix"
                or meta["physical_population_rule"] != "any_genuine_first_prefix_terminal_or_path_component"
                or meta["later_horizon_availability_is_not_an_emission_gate"] is not True
                or meta["closure_mode"] != "unknown_gaps_only_no_inferred_market_calendar"
                or meta["target_only_columns"] is not True
                or meta["feature_inputs_include_future_masks"] is not False
                or meta["dense_future_paths_materialized"] is not False):
            fail("POINTER_MANIFEST_INVALID")
        quote, index = meta["source_quote_binding"], meta["index_binding"]
        index_keys = {"schema_version", "source_parquet_sha256", "source_clock_sha256", "source_rows",
            "indexed_clock_sha256", "indexed_rows", "source_row_offset", "split_end_exclusive",
            "clock_kind", "unknown_gap_count", "known_closure_count", "closure_authority_sha256"}
        if (not isinstance(quote, Mapping) or not {"path", "sha256", "size_bytes", "rows", "clock_sha256"}.issubset(quote)
                or not sha(quote.get("sha256")) or not sha(quote.get("clock_sha256"))
                or type(quote.get("rows")) is not int or quote["rows"] <= 0
                or type(quote.get("size_bytes")) is not int or quote["size_bytes"] <= 0
                or not isinstance(index, Mapping) or set(index) != index_keys
                or index["schema_version"] != meta["contract"]["schema_version"]
                or index["source_parquet_sha256"] != quote["sha256"]
                or index["source_clock_sha256"] != quote["clock_sha256"]
                or type(index["source_rows"]) is not int or index["source_rows"] != quote["rows"]
                or not sha(index["indexed_clock_sha256"])
                or type(index["indexed_rows"]) is not int or not 0 < index["indexed_rows"] <= quote["rows"]
                or type(index["source_row_offset"]) is not int or index["source_row_offset"] != 0
                or utc(index["split_end_exclusive"]) != cutoff
                or index["clock_kind"] != meta["contract"]["clock_kind"]
                or type(index["unknown_gap_count"]) is not int or not 0 <= index["unknown_gap_count"] < quote["rows"]
                or type(index["known_closure_count"]) is not int or index["known_closure_count"] != 0
                or index["closure_authority_sha256"] is not None):
            fail("QUOTE_INDEX_BINDING_INVALID")
        quote_path = Path(quote["path"])
        if (not quote_path.is_absolute() or quote_path.is_symlink()
                or quote_path != quote_path.resolve(strict=True)
                or quote_path.stat().st_size != quote["size_bytes"]):
            fail("QUOTE_SOURCE_INVALID")
        require_binding({"path": str(quote_path), "sha256": quote["sha256"]},
                        label="dynamic M5 original quote provenance", verify_file=True)
        if shared_quote is not None and quote != shared_quote:
            fail("SPLIT_QUOTE_LINEAGE_MISMATCH")
        shared_quote = quote
        window = source.get("declared_window", {})
        if not isinstance(window, Mapping) or set(window) != {"start", "end"}:
            fail("PHYSICAL_WINDOW_INVALID")
        low, high = utc(window["start"]), utc(window["end"])
        if high < low or high > cutoff:
            fail("PHYSICAL_WINDOW_INVALID")
        parquet = pq.ParquetFile(parquet_binding["path"])
        columns = ["time", *DYNAMIC_M5_POINTER_COLUMNS]
        if parquet.metadata.num_rows != rows or not set(columns).issubset(parquet.schema_arrow.names):
            fail("PHYSICAL_COLUMNS_INVALID")
        clock_hash, pointer_hash = hashlib.sha256(), hashlib.sha256()
        terminal_counts, path_counts = np.zeros(2, np.int64), np.zeros(2, np.int64)
        terminal_reasons, path_reasons = {}, {}
        seen, previous = 0, None
        reasons = {"observed", "fill_unavailable", "split_right_censor", "terminal_quote_missing",
                   "unknown_source_absence", "prefix_close_missing", "source_window_unavailable",
                   "closed_m1_prefix_only"}
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            frame = batch.to_pandas()
            clock = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True, errors="coerce")).as_unit("ns")
            if (clock.hasnans or not clock.is_unique or not clock.is_monotonic_increasing
                    or not clock.floor("5min").equals(clock)
                    or (previous is not None and len(clock) and clock[0] <= previous)
                    or np.any(clock < low) or np.any(clock > high)
                    or np.any(clock.asi8 + 300 * 10**9 >= cutoff.value)):
                fail("PHYSICAL_CLOCK_INVALID")
            if len(clock):
                previous = clock[-1]
            pointer = frame[DYNAMIC_M5_POINTER_COLUMNS[0]].to_numpy()
            valid = frame[DYNAMIC_M5_POINTER_COLUMNS[1]].to_numpy()
            terminal = frame[list(DYNAMIC_M5_POINTER_COLUMNS[2:4])].to_numpy()
            path = frame[list(DYNAMIC_M5_POINTER_COLUMNS[4:6])].to_numpy()
            if (pointer.dtype != np.dtype("int64") or np.any(pointer < 0) or np.any(pointer >= index["indexed_rows"])
                    or valid.dtype != np.bool_ or terminal.dtype != np.bool_ or path.dtype != np.bool_
                    or not valid.all() or not np.array_equal(valid, terminal.any(axis=1) | path.any(axis=1))):
                fail("PHYSICAL_MASK_OR_POINTER_INVALID")
            for column, masks, counts in ((DYNAMIC_M5_POINTER_COLUMNS[6], terminal, terminal_reasons),
                                           (DYNAMIC_M5_POINTER_COLUMNS[7], path, path_reasons)):
                for reason, observed in zip(frame[column].to_numpy(), masks):
                    if (not isinstance(reason, str) or reason not in reasons
                            or bool(np.any(observed)) != (reason == "observed")):
                        fail("PHYSICAL_CENSOR_REASON_INVALID")
                    counts[reason] = counts.get(reason, 0) + 1
            terminal_counts += terminal.sum(axis=0)
            path_counts += path.sum(axis=0)
            clock_hash.update(np.asarray(clock.asi8, dtype="<i8").tobytes())
            pointer_hash.update(np.asarray(pointer, dtype="<i8").tobytes())
            seen += len(frame)
        actual = {
            "source_parquet": parquet_binding, "source_manifest": manifest_binding,
            "rows": seen, "physical_clock_sha256": clock_hash.hexdigest(),
            "dynamic_m5_supervision": dict(meta), "first_prefix_observed_rows": seen,
            "first_terminal_observed_rows_by_side": dict(zip(("long", "short"), map(int, terminal_counts))),
            "first_path_observed_rows_by_side": dict(zip(("long", "short"), map(int, path_counts))),
            "first_terminal_censor_reason_counts": terminal_reasons,
            "first_path_censor_reason_counts": path_reasons,
            "entry_m1_pointer_sha256": pointer_hash.hexdigest(),
        }
        if seen != rows or actual["physical_clock_sha256"] != source["clock_sha256"]:
            fail("PHYSICAL_CLOCK_PROOF_MISMATCH")
        try:
            exact = json.dumps(value["splits"][split], sort_keys=True, allow_nan=False) == json.dumps(actual, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            exact = False
        if not exact:
            fail("SPLIT_PROOF_MISMATCH")
    return dict(value)


def require_m5_forecast_observation_storage(frame: Any, *, context: str):
    """Validate the four active M5 forecast cells without changing population."""
    import numpy as np
    required = (*MODEL_NATIVE_FORECAST_TARGET_COLUMNS, *MODEL_NATIVE_FORECAST_VALID_COLUMNS)
    if any(name not in frame for name in required):
        raise RuntimeError(f"[{context}_M5_FORECAST_OBSERVATION_COLUMNS_MISSING]")
    values = frame[list(MODEL_NATIVE_FORECAST_TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    raw_valid = frame[list(MODEL_NATIVE_FORECAST_VALID_COLUMNS)].to_numpy()
    if not np.isin(raw_valid, (False, True)).all():
        raise RuntimeError(f"[{context}_M5_FORECAST_OBSERVATION_DOMAIN_INVALID]")
    valid = raw_valid.astype(bool)
    if not np.isfinite(values).all() or np.any(values[~valid] != 0):
        raise RuntimeError(f"[{context}_M5_FORECAST_MASKED_STORAGE_INVALID]")
    return valid


def require_model_native_aux_target_contract(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    scope = value.get("emission_scope", LEGACY_COMPLETE_EMISSION_SCOPE) if isinstance(value, Mapping) else LEGACY_COMPLETE_EMISSION_SCOPE
    expected = model_native_aux_target_contract_metadata(emission_scope=scope)
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"[{context}_AUX_TARGET_CONTRACT_INVALID]")
    return expected


def require_model_native_aux_target_emission_contract(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate one emitted split contract and return its shared static core."""

    scope = value.get("emission_scope", LEGACY_COMPLETE_EMISSION_SCOPE) if isinstance(value, Mapping) else LEGACY_COMPLETE_EMISSION_SCOPE
    expected = model_native_aux_target_contract_metadata(emission_scope=scope)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_AUX_TARGET_EMISSION_CONTRACT_INVALID] not_mapping")
    observed = dict(value)
    proof_keys = (MODEL_NATIVE_M5_EMISSION_PROOF_KEYS if scope in (M5_PRIMARY_OBSERVED_EMISSION_SCOPE, M5_DYNAMIC_OBSERVED_EMISSION_SCOPE)
                  else MODEL_NATIVE_AUX_EMISSION_PROOF_KEYS)
    expected_keys = set(expected).union(proof_keys)
    if set(observed) != expected_keys:
        missing = sorted(expected_keys - set(observed))
        extra = sorted(set(observed) - expected_keys)
        raise RuntimeError(
            f"[{context}_AUX_TARGET_EMISSION_CONTRACT_INVALID] "
            f"missing={missing} extra={extra}"
        )
    static_observed = {key: observed[key] for key in expected}
    if static_observed != expected:
        raise RuntimeError(
            f"[{context}_AUX_TARGET_EMISSION_CONTRACT_INVALID] static_contract"
        )
    if scope in (M5_PRIMARY_OBSERVED_EMISSION_SCOPE, M5_DYNAMIC_OBSERVED_EMISSION_SCOPE):
        counts = {key: observed[key] for key in MODEL_NATIVE_M5_EMISSION_PROOF_KEYS[:-1]}
        by_column = observed["forecast_observed_rows_by_column"]
        emitted = counts["primary_valid_rows_emitted"]
        if (any(type(v) is not int or v < 0 for v in counts.values())
                or emitted <= 0
                or counts["candidate_rows_before_primary_availability"] != emitted + counts["primary_unavailable_candidate_rows_excluded"]
                or counts["legacy_causal_m1_observed_rows_emitted"] > emitted
                or counts["legacy_aux_complete_rows_emitted"] > emitted
                or not isinstance(by_column, Mapping)
                or set(by_column) != set(MODEL_NATIVE_FORECAST_TARGET_COLUMNS)
                or any(type(v) is not int or not 0 <= v <= emitted for v in by_column.values())):
            raise RuntimeError(f"[{context}_M5_AUX_TARGET_EMISSION_PROOF_INVALID]")
        return expected
    proof: dict[str, int] = {}
    for key in MODEL_NATIVE_AUX_EMISSION_PROOF_KEYS:
        raw = observed[key]
        if type(raw) is not int or raw < 0:
            raise RuntimeError(
                f"[{context}_AUX_TARGET_EMISSION_CONTRACT_INVALID] "
                f"{key}={raw!r}"
            )
        proof[key] = raw
    expected_incomplete = int(MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS)
    # The M1 fill-lifecycle completeness gate runs before this auxiliary-target
    # gate.  It may already have removed part of the final K=96 M5 tail, so
    # the number of *candidate* rows rejected here is bounded by the global
    # incomplete target tail; it is not necessarily equal to it.  The exact
    # candidate/emitted accounting below remains mandatory and prevents an
    # incomplete auxiliary target from reaching the trainer.
    if (
        proof["candidate_rows_before_completeness"] <= 0
        or proof["complete_rows_emitted"] <= 0
        or proof["incomplete_tail_rows_total"] != expected_incomplete
        or proof["incomplete_candidate_rows_excluded"] > expected_incomplete
        or proof["candidate_rows_before_completeness"]
        != proof["complete_rows_emitted"]
        + proof["incomplete_candidate_rows_excluded"]
    ):
        raise RuntimeError(
            f"[{context}_AUX_TARGET_EMISSION_CONTRACT_INVALID] "
            f"proof={proof} expected_incomplete={expected_incomplete}"
        )
    return expected
