"""One exact Q/V/advantage fixture for model-native runtime tests."""

from __future__ import annotations

import numpy as np

from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS,
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_model_native_offline_rl_v1 import (
    ACTION_ORDER,
    ACTION_VALUE_TARGET_COLUMNS,
    HORIZON_BARS,
    REWARD_SCALE_BPS,
    offline_rl_contract_metadata,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_BASE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
)


def offline_rl_evidence() -> dict[str, list[float]]:
    action_value = [0.4, 0.5, 0.6, -0.2, -0.1, 0.0, 0.0, 0.0, 0.0]
    expectile_value = [0.1, 0.2, 0.3]
    return {
        "action_value": action_value,
        "expectile_value": expectile_value,
        "action_advantage": [
            value - expectile_value[index % 3]
            for index, value in enumerate(action_value)
        ],
    }


def model_native_target_audit_evidence() -> dict[str, object]:
    return {
        "model_native_aux_target_contract": (
            model_native_aux_target_contract_metadata()
        ),
        "offline_rl_target_contract": {
            "decision": "PASS",
            "failures": [],
            "offline_rl_contract": offline_rl_contract_metadata(),
            "action_value_target_columns": list(ACTION_VALUE_TARGET_COLUMNS),
        },
        "target_head_contract": {
            "active_training_heads": list(MODEL_NATIVE_BASE_ACTIVE_HEADS),
            "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
            "extra_active_target_heads": list(
                MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
            ),
            "extra_active_target_head_liveness": {
                head: True for head in MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
            },
        },
    }


def offline_rl_prediction_columns(rows: int) -> dict[str, object]:
    rows = int(rows)
    ordinal = np.arange(rows, dtype=np.float64)
    base = np.mod(ordinal, 97.0) / 96.0
    rewards = np.zeros((rows, len(ACTION_ORDER), len(HORIZON_BARS)))
    columns: dict[str, object] = {}
    for horizon_index, horizon in enumerate(HORIZON_BARS):
        long_reward = 50.0 + 5.0 * base + horizon_index
        short_reward = -25.0 + 5.0 * base - horizon_index
        rewards[:, 0, horizon_index] = long_reward
        rewards[:, 1, horizon_index] = short_reward
        columns[f"y_action_value_long_K{horizon}"] = long_reward
        columns[f"y_action_value_short_K{horizon}"] = short_reward
        columns[f"y_action_value_flat_K{horizon}"] = np.zeros(rows)
    q_values = rewards / float(REWARD_SCALE_BPS)
    value = q_values.max(axis=1) - 0.05
    advantage = q_values - value[:, None, :]
    columns["action_value"] = [row.tolist() for row in q_values.reshape(rows, -1)]
    columns["expectile_value"] = [row.tolist() for row in value]
    columns["action_advantage"] = [
        row.tolist() for row in advantage.reshape(rows, -1)
    ]
    return columns


def offline_rl_prediction_row(index: int = 0) -> dict[str, object]:
    columns = offline_rl_prediction_columns(max(1, int(index) + 1))
    return {name: values[int(index)] for name, values in columns.items()}
