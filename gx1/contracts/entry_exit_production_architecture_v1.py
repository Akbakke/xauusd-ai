"""One fail-closed production architecture for the shared Entry/Exit model.

This module is deliberately shape-only.  It does not load sources, caches,
datasets, lifecycle artifacts, or model state.  Production boundaries must
present their observed architecture here before doing any of those things.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_TIMEFRAME,
    ENTRY_EXIT_SHARED_ENCODER,
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    ENTRY_FEATURE_SEQUENCE_BARS,
    EXIT_DECISION_TIMEFRAME,
    EXIT_FEATURE_MAX_SEQUENCE_BARS,
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    SCHEMA_VERSION as INPUT_NORMALIZATION_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)
from gx1.features.htf_features import (
    HTF_V4_CACHE_SCHEMA_VERSION,
    HTF_V4_MATRIX_CONTRACT,
    MULTI_TF_FEATURE_COUNT_V4,
    MULTI_TF_TIMEFRAMES,
)


ENTRY_EXIT_PRODUCTION_ARCHITECTURE_SCHEMA_VERSION = (
    "gx1_entry_exit_production_architecture_v10"
)
# Derived from the one ordered per-TF surface owner (rule 13: a repeated
# literal is not ownership proof).  V29 Phase A stage 2: pre-V29 111 + 21
# event fields + 11 level-registry fields + 30 trendline-registry fields.
PRODUCTION_MTF_PER_TF_WIDTH = MULTI_TF_FEATURE_COUNT_V4
PRODUCTION_EXIT_SEQUENCE_BARS = 480
PRODUCTION_EXIT_MAX_PATH_BARS = 512
PRODUCTION_MTF_CACHE_TIMEFRAMES = ("M5", "M15", "H1", "H4", "D1")
PRODUCTION_MTF_PER_TF_WINDOW_BARS = (
    ("M5", 16),
    ("M15", 64),
    ("H1", 96),
    ("H4", 96),
    ("D1", 252),
)
PRODUCTION_ENTRY_MTF_ROUTE = ("M15", "H1", "H4", "D1")
PRODUCTION_EXIT_MTF_ROUTE = PRODUCTION_MTF_CACHE_TIMEFRAMES
PRODUCTION_SPECIALISTS = (
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
    "chart_geometry_encoder",
    "price_action_candle_encoder",
)


def entry_exit_production_architecture_contract() -> dict[str, Any]:
    """Return a fresh copy of the only production-selectable architecture."""

    return {
        "schema_version": ENTRY_EXIT_PRODUCTION_ARCHITECTURE_SCHEMA_VERSION,
        "schemas": {
            "signal": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
            "input_normalization": INPUT_NORMALIZATION_SCHEMA_VERSION,
            "mtf_cache": HTF_V4_CACHE_SCHEMA_VERSION,
            "mtf_matrix": HTF_V4_MATRIX_CONTRACT,
        },
        "shared_surface": {
            # Derived from the signal/context contract owners (rule 13; the
            # V29 stage-2 wiring grew the mandatory causal surface, so a
            # repeated literal here would be a second, driftable declaration).
            "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
            "snap_dim": MODEL_NATIVE_SIGNAL_DIM,
            "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
            "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        },
        "shared_encoder": "entry_v10_ctx_hybrid_transformer_shared_specialists_v2",
        "local_specialists": list(PRODUCTION_SPECIALISTS),
        "mtf_specialists": list(PRODUCTION_SPECIALISTS),
        "mtf": {
            "generation": "V4",
            "native_source_timeframe": "M5",
            "cache_timeframes": list(PRODUCTION_MTF_CACHE_TIMEFRAMES),
            "per_tf_widths": {
                timeframe: PRODUCTION_MTF_PER_TF_WIDTH
                for timeframe in PRODUCTION_MTF_CACHE_TIMEFRAMES
            },
            "per_tf_window_bars": dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
            "m5_route_consumption": {
                "entry": False,
                "exit": True,
            },
        },
        "entry": {
            "local_timeframe": "M5",
            "sequence_bars": 96,
            "mtf_route": list(PRODUCTION_ENTRY_MTF_ROUTE),
        },
        "exit": {
            "local_timeframe": "M1",
            "sequence_bars": PRODUCTION_EXIT_SEQUENCE_BARS,
            "mtf_route": list(PRODUCTION_EXIT_MTF_ROUTE),
            "max_path_bars": PRODUCTION_EXIT_MAX_PATH_BARS,
        },
    }


def current_entry_exit_architecture_observation() -> dict[str, Any]:
    """Project current code constants into the frozen production contract."""

    observed = entry_exit_production_architecture_contract()
    observed["schemas"] = {
        "signal": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "input_normalization": INPUT_NORMALIZATION_SCHEMA_VERSION,
        "mtf_cache": HTF_V4_CACHE_SCHEMA_VERSION,
        "mtf_matrix": HTF_V4_MATRIX_CONTRACT,
    }
    observed["shared_surface"] = {
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_dim": MODEL_NATIVE_SIGNAL_DIM,
        "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
    }
    observed["shared_encoder"] = ENTRY_EXIT_SHARED_ENCODER
    observed["local_specialists"] = list(MODEL_NATIVE_TRAINING_SPECIALISTS)
    observed["mtf_specialists"] = list(MODEL_NATIVE_TRAINING_SPECIALISTS)
    observed["mtf"] = {
        "generation": "V4",
        "native_source_timeframe": ENTRY_DECISION_TIMEFRAME,
        "cache_timeframes": list(MULTI_TF_TIMEFRAMES),
        "per_tf_widths": {
            timeframe: MULTI_TF_FEATURE_COUNT_V4
            for timeframe in MULTI_TF_TIMEFRAMES
        },
        "per_tf_window_bars": dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS),
        "m5_route_consumption": {
            "entry": "M5" in ENTRY_MTF_CONTEXT_TIMEFRAMES,
            "exit": "M5" in EXIT_MTF_CONTEXT_TIMEFRAMES,
        },
    }
    observed["entry"] = {
        "local_timeframe": ENTRY_DECISION_TIMEFRAME,
        "sequence_bars": ENTRY_FEATURE_SEQUENCE_BARS,
        "mtf_route": list(ENTRY_MTF_CONTEXT_TIMEFRAMES),
    }
    observed["exit"] = {
        "local_timeframe": EXIT_DECISION_TIMEFRAME,
        "sequence_bars": EXIT_FEATURE_SEQUENCE_BARS,
        "mtf_route": list(EXIT_MTF_CONTEXT_TIMEFRAMES),
        "max_path_bars": EXIT_FEATURE_MAX_SEQUENCE_BARS,
    }
    return observed


def _first_mismatch(
    expected: Any,
    observed: Any,
    *,
    path: str,
) -> tuple[str, Any, Any] | None:
    if isinstance(expected, dict):
        if not isinstance(observed, Mapping):
            return path, expected, observed
        expected_keys = list(expected)
        observed_keys = list(observed)
        if observed_keys != expected_keys:
            return f"{path}.__keys__", expected_keys, observed_keys
        for key in expected_keys:
            mismatch = _first_mismatch(
                expected[key],
                observed[key],
                path=f"{path}.{key}",
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            return path, expected, observed
        for index, expected_item in enumerate(expected):
            mismatch = _first_mismatch(
                expected_item,
                observed[index],
                path=f"{path}[{index}]",
            )
            if mismatch is not None:
                return mismatch
        return None
    if type(observed) is not type(expected) or observed != expected:
        return path, expected, observed
    return None


def require_entry_exit_production_architecture(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Reject any shape, route, schema, encoder, or specialist drift."""

    if not isinstance(context, str) or not context.strip():
        raise RuntimeError("ENTRY_EXIT_PRODUCTION_ARCHITECTURE_CONTEXT_INVALID")
    expected = entry_exit_production_architecture_contract()
    mismatch = _first_mismatch(expected, value, path="architecture")
    if mismatch is not None:
        path, expected_value, observed_value = mismatch
        raise RuntimeError(
            f"{context}_ENTRY_EXIT_PRODUCTION_ARCHITECTURE_MISMATCH: "
            f"field={path} observed={observed_value!r} "
            f"expected={expected_value!r}"
        )
    return deepcopy(expected)


__all__ = [
    "ENTRY_EXIT_PRODUCTION_ARCHITECTURE_SCHEMA_VERSION",
    "PRODUCTION_EXIT_MAX_PATH_BARS",
    "PRODUCTION_EXIT_SEQUENCE_BARS",
    "PRODUCTION_MTF_PER_TF_WIDTH",
    "PRODUCTION_MTF_PER_TF_WINDOW_BARS",
    "current_entry_exit_architecture_observation",
    "entry_exit_production_architecture_contract",
    "require_entry_exit_production_architecture",
]
