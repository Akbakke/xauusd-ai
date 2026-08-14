"""Hash-bound proposal for the model's per-timeframe input capacity.

The values in this module are tensor receptive-field capacities.  They are not
entry/exit rules, thresholds, labels, horizons, or promises that the trained
model will use every historical bar.  Integration is deliberately false until
the dataset, normalization, model, bundle, replay, and live owners migrate as
one fail-closed transaction.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from gx1.contracts.entry_exit_production_architecture_v1 import (
    PRODUCTION_MTF_PER_TF_WINDOW_BARS,
)


SCHEMA_VERSION = "gx1_entry_mtf_temporal_receptive_field_policy_v1"
DECISION = "PROPOSED_CAPACITY_NOT_INTEGRATED"
TIMEFRAME_ORDER = ("M5", "M15", "H1", "H4", "D1")
PROPOSED_MINIMUM_WINDOW_BARS = {
    "M5": 96,
    "M15": 96,
    "H1": 168,
    "H4": 180,
    "D1": 252,
}
BAR_DURATION_SECONDS = {
    "M5": 5 * 60,
    "M15": 15 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D1": 24 * 60 * 60,
}
NOMINAL_COVERAGE_LABEL = {
    "M5": "8_hours",
    "M15": "1_day",
    "H1": "1_week",
    "H4": "30_days",
    "D1": "252_daily_observations_approximately_one_trading_year",
}


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def current_window_observation() -> dict[str, int]:
    observed = {
        str(timeframe): int(bars)
        for timeframe, bars in PRODUCTION_MTF_PER_TF_WINDOW_BARS
    }
    if tuple(observed) != TIMEFRAME_ORDER or any(value <= 0 for value in observed.values()):
        raise RuntimeError("TEMPORAL_RECEPTIVE_FIELD_CURRENT_OWNER_INVALID")
    return observed


def temporal_receptive_field_policy() -> dict[str, Any]:
    current = current_window_observation()
    minimum = dict(PROPOSED_MINIMUM_WINDOW_BARS)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "timeframe_order": list(TIMEFRAME_ORDER),
        "current_owner": (
            "gx1.contracts.entry_exit_production_architecture_v1."
            "PRODUCTION_MTF_PER_TF_WINDOW_BARS"
        ),
        "current_window_bars": current,
        "proposed_minimum_window_bars": minimum,
        "bar_duration_seconds": dict(BAR_DURATION_SECONDS),
        "nominal_coverage_seconds": {
            timeframe: (
                None
                if timeframe == "D1"
                else minimum[timeframe] * BAR_DURATION_SECONDS[timeframe]
            )
            for timeframe in TIMEFRAME_ORDER
        },
        "nominal_coverage_label": dict(NOMINAL_COVERAGE_LABEL),
        "coverage_basis": {
            "intraday": "bar_count_times_native_cadence",
            "d1": (
                "252_closed_daily_observations; elapsed_wall_clock_depends_on_"
                "market_closures_and_is_approximately_one_trading_year"
            ),
        },
        "capacity_semantics": {
            "input_tensor_capacity_only": True,
            "decision_threshold": None,
            "trade_direction_authority": False,
            "entry_or_exit_timing_rule": False,
            "label_or_target_horizon": False,
            "forced_attention_or_usefulness": False,
            "model_learns_history_usefulness_from_train": True,
            "validation_or_test_selection_authority": False,
        },
        "causality": {
            "closed_bars_only": True,
            "right_edge": "decision_clock_available_closed_bar",
            "left_edge": "oldest_retained_closed_bar_at_declared_capacity",
            "future_rows": False,
        },
        "integration": {
            "integrated": False,
            "requires_atomic_contract_bump": True,
            "requires_dataset_rebuild": True,
            "requires_normalization_refit_and_rebind": True,
            "requires_model_positional_capacity_rebuild": True,
            "requires_candidate_retraining": True,
            "requires_bundle_replay_live_parity": True,
            "requires_registry_owner_coordination": True,
            "no_production_default_changes_in_this_wave": True,
        },
        "growth": {
            timeframe: {
                "current_bars": current[timeframe],
                "proposed_minimum_bars": minimum[timeframe],
                "added_bars": minimum[timeframe] - current[timeframe],
                "capacity_ratio": minimum[timeframe] / current[timeframe],
            }
            for timeframe in TIMEFRAME_ORDER
        },
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return payload


def require_temporal_receptive_field_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = temporal_receptive_field_policy()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError("TEMPORAL_RECEPTIVE_FIELD_POLICY_INVALID")
    return expected


__all__ = (
    "BAR_DURATION_SECONDS",
    "DECISION",
    "NOMINAL_COVERAGE_LABEL",
    "PROPOSED_MINIMUM_WINDOW_BARS",
    "SCHEMA_VERSION",
    "TIMEFRAME_ORDER",
    "canonical_json_sha256",
    "current_window_observation",
    "require_temporal_receptive_field_policy",
    "temporal_receptive_field_policy",
)
