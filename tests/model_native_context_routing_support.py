"""Deterministic context-routing fixtures.

These temporal aliases are fixture evidence, not a production constant.
Production derives the overlap from each artifact's ordered signal names.
"""
from __future__ import annotations

import json

from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    classify_entry_specialist_feature,
    model_native_context_temporal_alias_policy,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    model_native_signal_contract_metadata,
)


# The fixed TRAIN-ranked remainder is retired: every code-owned candidate is
# selected. Derive the selected order exactly as the manifest materializer
# does, so this fixture cannot drift from the contract owner.
CANONICAL_SELECTED_FIELDS = (
    *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
)
_CTX_CONT_SET = set(MODEL_NATIVE_CTX_CONT_FIELDS)
TEMPORAL_ALIAS_SIGNAL_FIELDS = tuple(
    name
    for name in model_native_signal_contract_metadata(CANONICAL_SELECTED_FIELDS)[
        "fields"
    ]
    if name.startswith("ctx_cont.") and name.removeprefix("ctx_cont.") in _CTX_CONT_SET
)
TEMPORAL_ALIAS_CTX_CONT_FIELDS = tuple(
    name.removeprefix("ctx_cont.") for name in TEMPORAL_ALIAS_SIGNAL_FIELDS
)


_FILLER_PREFIX_BY_SPECIALIST = {
    "structure_swing_encoder": "structure.swing_fixture_",
    "smc_liquidity_encoder": "smc.liquidity_fixture_",
    "trend_ema_encoder": "trend.ema_fixture_",
    "vol_compression_encoder": "volatility.atr_fixture_",
    "momentum_flow_encoder": "momentum.flow_fixture_",
    "session_regime_encoder": "session.regime_fixture_",
    "chart_geometry_encoder": "chart.line_pattern_fixture_",
    "price_action_candle_encoder": "price_action.candle_fixture_",
}


def ordered_signal_names_for_specialist_indices(
    specialist_indices: dict[str, list[int]],
    *,
    temporal_alias_signal_fields: tuple[str, ...] = (
        TEMPORAL_ALIAS_SIGNAL_FIELDS
    ),
) -> list[str]:
    width = sum(len(indices) for indices in specialist_indices.values())
    fields: list[str | None] = [None] * width
    available = {
        name: list(indices) for name, indices in specialist_indices.items()
    }
    for signal_field in temporal_alias_signal_fields:
        owner = classify_entry_specialist_feature(signal_field)
        fields[available[owner].pop(0)] = signal_field
    for specialist in MODEL_NATIVE_TRAINING_SPECIALISTS:
        for ordinal, index in enumerate(available[specialist]):
            fields[index] = f"{_FILLER_PREFIX_BY_SPECIALIST[specialist]}{ordinal:03d}"
    if any(field is None for field in fields):
        raise RuntimeError("MODEL_NATIVE_TEST_SIGNAL_FIELD_GAP")
    return [str(field) for field in fields]


def context_routing_for_ordered_signal_names(
    ordered_signal_names: list[str],
) -> dict:
    routing = json.loads(
        json.dumps(MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT)
    )
    routing["temporal_alias_policy"] = (
        model_native_context_temporal_alias_policy(ordered_signal_names)
    )
    return routing
