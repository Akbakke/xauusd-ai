"""Exact supervised objective surface for model-native XAU direction bundles."""

from __future__ import annotations

import math
from typing import Any, Mapping


SCHEMA_VERSION = "entry_model_native_training_objective_v1"

# Configurable losses whose zero value would advertise evidence that was never
# trained. Fixed positive loss multipliers are declared separately below.
REQUIRED_POSITIVE_LOSS_WEIGHTS = (
    "ENTRY_DIRECTION_CE_SCALE",
    "ENTRY_TAIL_DIRECTION_CE_WEIGHT",
    "ENTRY_SPECIALIST_GATE_ENTROPY_WEIGHT",
    "ENTRY_SPECIALIST_GATE_BALANCE_WEIGHT",
    "ENTRY_MTF_DIR_AUX_WEIGHT",
    "ENTRY_DIRECTION_UTILITY_MARGIN_WEIGHT",
    "ENTRY_DIRECTION_SIDE_UTILITY_CONVICTION_WEIGHT",
    "ENTRY_DIRECTION_UTILITY_TRADE_CONVICTION_WEIGHT",
    "ENTRY_DIRECTION_UTILITY_TRIAD_CE_WEIGHT",
    "ENTRY_AUX_PATH_WEIGHT",
    "ENTRY_AUX_MFE_WEIGHT",
    "ENTRY_AUX_TRADABLE_WEIGHT",
    "ENTRY_AUX_BAD_PATH_WEIGHT",
    "ENTRY_BAD_PATH_QUALITY_RANK_WEIGHT",
    "ENTRY_PATH_QUALITY_RANK_WEIGHT",
    "ENTRY_AUX_CLEAN_EDGE_WEIGHT",
    "ENTRY_AUX_SURVIVAL_WEIGHT",
    "ENTRY_CLEAN_EDGE_RANKING_WEIGHT",
    "ENTRY_HIER_TRADE_WEIGHT",
    "ENTRY_HIER_SIDE_WEIGHT",
    "ENTRY_HIER_UTILITY_WEIGHT",
    "ENTRY_HIER_BAD_PATH_WEIGHT",
    "ENTRY_HIER_MAE_WEIGHT",
    "ENTRY_HIER_SIDE_VALIDITY_WEIGHT",
    "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT",
    "ENTRY_OFFLINE_RL_Q_WEIGHT",
    "ENTRY_OFFLINE_RL_V_WEIGHT",
    "ENTRY_OFFLINE_RL_RANK_WEIGHT",
)

FIXED_POSITIVE_LOSS_WEIGHTS = {
    "tf_agreement": 0.3,
    "position_size": 0.2,
    "dip_forecast_timing_tail_vol_composite": 0.2,
}


def active_loss_weight_failures(weights: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    observed = set(weights)
    expected = set(REQUIRED_POSITIVE_LOSS_WEIGHTS)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    failures.extend(f"{name}=missing" for name in missing)
    failures.extend(f"{name}=unexpected" for name in extra)
    for name in REQUIRED_POSITIVE_LOSS_WEIGHTS:
        if name not in weights:
            continue
        try:
            value = float(weights[name])
        except (TypeError, ValueError):
            failures.append(f"{name}=non-numeric")
            continue
        if not math.isfinite(value):
            failures.append(f"{name}=non-finite")
        elif value <= 0.0:
            failures.append(f"{name}={value:g} expected >0")
    return failures


def require_training_objective_contract(
    value: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"[{context}_TRAINING_OBJECTIVE_MISSING]")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"[{context}_TRAINING_OBJECTIVE_SCHEMA_INVALID]")
    configurable = value.get("configurable_positive_loss_weights")
    if not isinstance(configurable, Mapping):
        raise RuntimeError(f"[{context}_TRAINING_OBJECTIVE_WEIGHTS_MISSING]")
    failures = active_loss_weight_failures(configurable)
    if failures:
        raise RuntimeError(
            f"[{context}_TRAINING_OBJECTIVE_WEIGHTS_INVALID] " + "; ".join(failures)
        )
    fixed = value.get("fixed_positive_loss_weights")
    if fixed != FIXED_POSITIVE_LOSS_WEIGHTS:
        raise RuntimeError(f"[{context}_TRAINING_OBJECTIVE_FIXED_WEIGHTS_INVALID]")
    if value.get("all_advertised_heads_supervised") is not True:
        raise RuntimeError(f"[{context}_TRAINING_OBJECTIVE_SUPERVISION_UNPROVEN]")
    return {
        "schema_version": SCHEMA_VERSION,
        "configurable_positive_loss_weights": {
            name: float(configurable[name])
            for name in REQUIRED_POSITIVE_LOSS_WEIGHTS
        },
        "fixed_positive_loss_weights": dict(FIXED_POSITIVE_LOSS_WEIGHTS),
        "all_advertised_heads_supervised": True,
    }


def training_objective_contract_metadata(
    weights: Mapping[str, Any],
) -> dict[str, Any]:
    return require_training_objective_contract(
        {
            "schema_version": SCHEMA_VERSION,
            "configurable_positive_loss_weights": dict(weights),
            "fixed_positive_loss_weights": dict(FIXED_POSITIVE_LOSS_WEIGHTS),
            "all_advertised_heads_supervised": True,
        },
        context="ENTRY_MODEL_NATIVE_EXPORT",
    )
