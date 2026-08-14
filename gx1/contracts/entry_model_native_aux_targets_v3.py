"""Canonical emitted future-target surface for model-native XAU Entry."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION = "entry_model_native_aux_targets_v6"
MODEL_NATIVE_AUX_FORECAST_HORIZONS = (1, 5, 12, 24)
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


def model_native_aux_target_contract_metadata() -> dict[str, Any]:
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
    return {
        "schema_version": MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION,
        "columns": list(MODEL_NATIVE_AUX_TARGET_COLUMNS),
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


def require_model_native_aux_target_contract(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    expected = model_native_aux_target_contract_metadata()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"[{context}_AUX_TARGET_CONTRACT_INVALID]")
    return expected


def require_model_native_aux_target_emission_contract(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate one emitted split contract and return its shared static core."""

    expected = model_native_aux_target_contract_metadata()
    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_AUX_TARGET_EMISSION_CONTRACT_INVALID] not_mapping")
    observed = dict(value)
    expected_keys = set(expected).union(MODEL_NATIVE_AUX_EMISSION_PROOF_KEYS)
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
    if (
        proof["candidate_rows_before_completeness"] <= 0
        or proof["complete_rows_emitted"] <= 0
        or proof["incomplete_tail_rows_total"] != expected_incomplete
        or proof["incomplete_candidate_rows_excluded"] != expected_incomplete
        or proof["candidate_rows_before_completeness"]
        != proof["complete_rows_emitted"]
        + proof["incomplete_candidate_rows_excluded"]
    ):
        raise RuntimeError(
            f"[{context}_AUX_TARGET_EMISSION_CONTRACT_INVALID] "
            f"proof={proof} expected_incomplete={expected_incomplete}"
        )
    return expected
