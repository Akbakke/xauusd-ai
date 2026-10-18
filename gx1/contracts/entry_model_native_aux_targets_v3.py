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
    if emission_scope not in (LEGACY_COMPLETE_EMISSION_SCOPE, M5_PRIMARY_OBSERVED_EMISSION_SCOPE):
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
    if emission_scope == M5_PRIMARY_OBSERVED_EMISSION_SCOPE:
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
    return result


def require_auxiliary_emission_training_phase(
    value: Mapping[str, Any], *, entry_training_phase: str, context: str,
) -> None:
    if value.get("emission_scope", LEGACY_COMPLETE_EMISSION_SCOPE) == M5_PRIMARY_OBSERVED_EMISSION_SCOPE:
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
    proof_keys = (MODEL_NATIVE_M5_EMISSION_PROOF_KEYS if scope == M5_PRIMARY_OBSERVED_EMISSION_SCOPE
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
    if scope == M5_PRIMARY_OBSERVED_EMISSION_SCOPE:
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
