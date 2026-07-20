"""Canonical emitted future-target surface for model-native XAU Entry."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from gx1.contracts.entry_model_native_offline_rl_v1 import (
    ACTION_VALUE_TARGET_COLUMNS,
    HORIZON_BARS as OFFLINE_RL_HORIZON_BARS,
    offline_rl_contract_metadata,
)


MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION = "entry_model_native_aux_targets_v3"
MODEL_NATIVE_AUX_FORECAST_HORIZONS = (1, 5, 12, 24)
MODEL_NATIVE_AUX_RISK_HORIZONS = (12, 48, 96)
_TARGET_HORIZON_ITEMS = tuple(
    (f"y_dip_mae_{side}_K{horizon}", horizon)
    for side in ("long", "short")
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
) + tuple(
    (f"y_dip_mfe_{side}_K{horizon}", horizon)
    for side in ("long", "short")
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
) + tuple(
    (f"y_forecast_ret_K{horizon}", horizon)
    for horizon in MODEL_NATIVE_AUX_FORECAST_HORIZONS
) + tuple(
    (f"y_dip_bottom_frac_{side}_K{horizon}", horizon)
    for side in ("long", "short")
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
) + tuple(
    (f"y_time_to_mfe_frac_{side}_K{horizon}", horizon)
    for side in ("long", "short")
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
) + tuple(
    (f"y_tail_mae_{side}_K{horizon}", horizon)
    for side in ("long", "short")
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
) + tuple(
    (f"y_vol_fwd_K{horizon}", horizon)
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS
) + tuple(
    (name, horizon)
    for name in ACTION_VALUE_TARGET_COLUMNS
    for horizon in OFFLINE_RL_HORIZON_BARS
    if name.endswith(f"_K{horizon}")
)
MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN = MappingProxyType(
    dict(_TARGET_HORIZON_ITEMS)
)
MODEL_NATIVE_AUX_TARGET_COLUMNS = tuple(MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN)
MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS = max(
    MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN.values()
)


def model_native_aux_target_contract_metadata() -> dict[str, Any]:
    return {
        "schema_version": MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION,
        "columns": list(MODEL_NATIVE_AUX_TARGET_COLUMNS),
        "future_horizon_bars_by_column": {
            name: int(horizon)
            for name, horizon in MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN.items()
        },
        "max_future_horizon_bars": int(MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS),
        "spread_aware_risk_magnitudes_required": True,
        "mid_price_timing_reference_only": True,
        "incomplete_value": "NaN_before_emission_only",
        "incomplete_rows_may_be_emitted": False,
        "offline_rl": offline_rl_contract_metadata(),
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
