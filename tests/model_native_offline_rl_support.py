"""One exact Q/V/advantage fixture for model-native runtime tests."""

from __future__ import annotations


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
