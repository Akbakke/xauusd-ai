"""Exact internal offline-RL evidence contract for model-native XAU Entry."""

from __future__ import annotations

from typing import Any, Mapping

import torch


SCHEMA_VERSION = "entry_model_native_offline_rl_v2"
ACTION_ORDER = ("LONG", "SHORT", "FLAT")
HORIZON_BARS = (12, 48, 96)
ACTION_COUNT = len(ACTION_ORDER)
HORIZON_COUNT = len(HORIZON_BARS)
ACTION_VALUE_DIM = ACTION_COUNT * HORIZON_COUNT
EXPECTILE_VALUE_DIM = HORIZON_COUNT
ADVANTAGE_DIM = ACTION_VALUE_DIM
EXPECTILE_TAU = 0.8
REWARD_SCALE_BPS = 50.0
RANKING_MARGIN_BPS = 5.0
RANKING_MARGIN_SCALED = RANKING_MARGIN_BPS / REWARD_SCALE_BPS
UTILITY_MFE_WEIGHT = 0.35
UTILITY_MAE_WEIGHT = 1.15
UTILITY_PATH_WEIGHT = 0.25
ACTION_VALUE_TARGET_COLUMNS = tuple(
    f"y_action_value_{action.lower()}_K{horizon}"
    for action in ACTION_ORDER
    for horizon in HORIZON_BARS
)


def offline_rl_contract_metadata() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "full_counterfactual_contextual_bandit_internal_evidence",
        "action_order": list(ACTION_ORDER),
        "horizon_bars": list(HORIZON_BARS),
        "action_value_layout": "action_major_then_horizon",
        "action_value_dim": ACTION_VALUE_DIM,
        "expectile_value_dim": EXPECTILE_VALUE_DIM,
        "advantage_dim": ADVANTAGE_DIM,
        "expectile_tau": EXPECTILE_TAU,
        "reward_scale_bps": REWARD_SCALE_BPS,
        "ranking_margin_bps": RANKING_MARGIN_BPS,
        "target_columns": list(ACTION_VALUE_TARGET_COLUMNS),
        "flat_reward_bps": 0.0,
        "value_target": "expectile_of_detached_max_action_q_per_horizon",
        "advantage_formula": "Q(s,a,K)-V(s,K)",
        "q_target": "full_counterfactual_cost_adjusted_path_utility_bps",
        "q_objective": "direct_regression_without_bellman_backup",
        "ranking_target": "unique_counterfactual_reward_argmax_per_horizon",
        "ambiguous_reward_ties_ranked": False,
        "logged_behavior_objective": False,
        "separate_policy_or_direction_authority": False,
        "final_learned_fusion_required": True,
    }


def require_offline_rl_contract_metadata(
    value: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    expected = offline_rl_contract_metadata()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"[{context}_OFFLINE_RL_CONTRACT_INVALID]")
    return expected


def expectile_loss(diff: torch.Tensor, *, tau: float = EXPECTILE_TAU) -> torch.Tensor:
    if not 0.5 < float(tau) < 1.0:
        raise ValueError(f"expectile tau must be in (0.5, 1.0), got {tau}")
    weight = torch.where(diff >= 0, float(tau), 1.0 - float(tau))
    return (weight * diff.square()).mean()


def q_ranking_margin_loss(
    q_values: torch.Tensor,
    reward_targets: torch.Tensor,
    *,
    margin: float = RANKING_MARGIN_SCALED,
) -> torch.Tensor:
    if q_values.shape != reward_targets.shape:
        raise ValueError(
            f"Q/reward shapes differ: {tuple(q_values.shape)} vs "
            f"{tuple(reward_targets.shape)}"
        )
    if q_values.ndim != 3 or tuple(q_values.shape[1:]) != (
        ACTION_COUNT,
        HORIZON_COUNT,
    ):
        raise ValueError(
            "Q/reward tensors must have shape "
            f"(batch,{ACTION_COUNT},{HORIZON_COUNT})"
        )
    if float(margin) <= 0.0:
        raise ValueError("Q ranking margin must be positive")
    top_two = reward_targets.topk(k=2, dim=1).values
    unique_best = top_two[:, 0, :] > top_two[:, 1, :]
    best_action = reward_targets.argmax(dim=1, keepdim=True)
    best_q = torch.gather(q_values, 1, best_action)
    hinge = torch.relu(float(margin) - (best_q - q_values))
    mask = torch.ones_like(hinge)
    mask.scatter_(1, best_action, 0.0)
    per_horizon = (hinge * mask).sum(dim=1).div(float(ACTION_COUNT - 1))
    valid = unique_best.to(dtype=per_horizon.dtype)
    valid_count = valid.sum()
    if not bool(valid_count.item() > 0):
        return q_values.sum() * 0.0
    return (per_horizon * valid).sum() / valid_count
