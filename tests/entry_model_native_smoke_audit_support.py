from __future__ import annotations

import copy
import math
from typing import Any

from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_SMOKE_SPLITS,
    foundation_audit_policy_metadata,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    model_native_aux_target_contract_metadata,
)


CLASS_NAMES = ("LONG", "SHORT", "FLAT")
POLICY = foundation_audit_policy_metadata()["smoke_edge_pockets"]


def _wilson_lower(successes: int, trials: int) -> float:
    z_score = float(POLICY["wilson_z_score"])
    proportion = successes / trials
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / trials
    centre = proportion + z_squared / (2.0 * trials)
    radius = z_score * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return float((centre - radius) / denominator)


def _passing_direction_metrics(*, support_per_class: int, scope: str) -> dict[str, Any]:
    successes = support_per_class - 1
    rows = support_per_class * len(CLASS_NAMES)
    confusion = [
        [successes, 1, 0],
        [0, successes, 1],
        [1, 0, successes],
    ]
    class_precision = successes / support_per_class
    trade_rows = support_per_class * 2
    trade_successes = successes * 2
    trade_precision = trade_successes / trade_rows
    if scope == "global":
        minimum_trade_rows = int(POLICY["min_trade_rows"])
        minimum_trade_precision = float(POLICY["min_trade_direction_precision"])
        minimum_trade_wilson = float(POLICY["min_trade_precision_wilson_lower"])
        minimum_prediction_rows: int | None = int(
            POLICY["min_prediction_rows_per_class"]
        )
        minimum_class_wilson: float | None = float(
            POLICY["min_class_precision_wilson_lower"]
        )
    else:
        assert scope == "context"
        minimum_trade_rows = int(POLICY["min_context_trade_rows"])
        minimum_trade_precision = float(
            POLICY["min_context_trade_direction_precision"]
        )
        minimum_trade_wilson = float(
            POLICY["min_context_trade_precision_wilson_lower"]
        )
        minimum_prediction_rows = None
        minimum_class_wilson = None
    return {
        "decision": "PASS",
        "failures": [],
        "rows": rows,
        "accuracy": class_precision,
        "majority_baseline_accuracy": 1.0 / 3.0,
        "beats_majority_baseline": True,
        "balanced_accuracy": class_precision,
        "support_scope": scope,
        "wilson_confidence_level": float(POLICY["wilson_confidence_level"]),
        "wilson_z_score": float(POLICY["wilson_z_score"]),
        "trade_rows": trade_rows,
        "trade_successes": trade_successes,
        "minimum_trade_rows": minimum_trade_rows,
        "trade_coverage": trade_rows / rows,
        "trade_direction_precision": trade_precision,
        "minimum_trade_direction_precision": minimum_trade_precision,
        "trade_direction_precision_wilson_lower": _wilson_lower(
            trade_successes, trade_rows
        ),
        "minimum_trade_precision_wilson_lower": minimum_trade_wilson,
        "minimum_prediction_rows_per_class": minimum_prediction_rows,
        "minimum_class_precision_wilson_lower": minimum_class_wilson,
        "log_loss": 0.01,
        "label_counts": {name: support_per_class for name in CLASS_NAMES},
        "prediction_counts": {name: support_per_class for name in CLASS_NAMES},
        "precision": {name: class_precision for name in CLASS_NAMES},
        "precision_successes": {name: successes for name in CLASS_NAMES},
        "precision_wilson_lower": {
            name: _wilson_lower(successes, support_per_class)
            for name in CLASS_NAMES
        },
        "recall": {name: class_precision for name in CLASS_NAMES},
        "confusion_matrix": confusion,
    }


def _passing_context_slices() -> dict[str, Any]:
    sessions = sorted(POLICY["expected_sessions"])
    vol_regimes = ["high", "low"]
    return {
        "decision": "PASS",
        "failures": [],
        "minimum_rows_per_slice": int(POLICY["min_rows_per_context_slice"]),
        "minimum_trade_rows_per_slice": int(POLICY["min_context_trade_rows"]),
        "minimum_trade_direction_precision": float(
            POLICY["min_context_trade_direction_precision"]
        ),
        "minimum_trade_precision_wilson_lower": float(
            POLICY["min_context_trade_precision_wilson_lower"]
        ),
        "fields": {
            "session": {
                "values": sessions,
                "slices": {
                    name: _passing_direction_metrics(
                        support_per_class=100,
                        scope="context",
                    )
                    for name in sessions
                },
            },
            "vol_regime": {
                "values": vol_regimes,
                "slices": {
                    name: _passing_direction_metrics(
                        support_per_class=200,
                        scope="context",
                    )
                    for name in vol_regimes
                },
            },
        },
    }


def _passing_turning_point_evidence() -> dict[str, Any]:
    policy = copy.deepcopy(POLICY["turning_point_evidence"])
    layout = model_native_aux_target_contract_metadata()[
        "turning_point_timing"
    ]["layout"]
    rows = 100
    successes = 100
    wilson = _wilson_lower(successes, rows)
    pockets = {}
    for turn, direction, side in (
        ("BOTTOM", "LONG", "long"),
        ("TOP", "SHORT", "short"),
    ):
        timing_index = next(
            int(item["index"])
            for item in layout
            if item["direction"] == side
            and int(item["horizon_bars"])
            == int(policy["evaluation_horizon_bars"])
            and item["target"] == "dip_bottom_frac"
        )
        pockets[turn] = {
            "decision": "PASS",
            "failures": [],
            "model_direction": direction,
            "timing_output_index": timing_index,
            "evaluation_horizon_bars": int(policy["evaluation_horizon_bars"]),
            "near_turn_max_fraction": float(policy["near_turn_max_fraction"]),
            "rows": rows,
            "direction_successes": successes,
            "direction_precision": 1.0,
            "direction_precision_wilson_lower": wilson,
            "timing_successes": successes,
            "timing_precision": 1.0,
            "timing_precision_wilson_lower": wilson,
        }
    return {
        "decision": "PASS",
        "failures": [],
        "policy": policy,
        "layout": copy.deepcopy(layout),
        "target_alignment": [
            {
                **copy.deepcopy(item),
                "spearman": 0.90,
                "mae": 0.05,
                "decision": "PASS",
                "failures": [],
            }
            for item in layout
        ],
        "near_turn_pockets": pockets,
        "live_direction_rule_authority": False,
    }


def _passing_offline_rl_evidence() -> dict[str, Any]:
    policy = copy.deepcopy(POLICY["offline_rl_evidence"])
    rows = 200
    successes = 190
    return {
        "decision": "PASS",
        "failures": [],
        "policy": policy,
        "q_target_alignment": [
            {
                "action": action,
                "horizon_bars": horizon,
                "spearman": None if action == "FLAT" else 0.90,
                "mae_scaled": 0.05,
                "decision": "PASS",
                "failures": [],
            }
            for action in ("LONG", "SHORT", "FLAT")
            for horizon in (12, 48, 96)
        ],
        "reward_argmax_ranking": {
            f"K{horizon}": {
                "decision": "PASS",
                "failures": [],
                "unique_reward_rows": rows,
                "successes": successes,
                "accuracy": successes / rows,
            }
            for horizon in (12, 48, 96)
        },
        "value_vs_max_q": {
            f"K{horizon}": {
                "decision": "PASS",
                "failures": [],
                "spearman": 0.90,
            }
            for horizon in (12, 48, 96)
        },
        "advantage_max_abs_error": 0.0,
        "separate_direction_authority": False,
    }


def passing_smoke_audit_splits() -> dict[str, Any]:
    split = {
        "decision": "PASS",
        "failures": [],
        "rows": 1_200,
        "direction": _passing_direction_metrics(
            support_per_class=400,
            scope="global",
        ),
        "context_slice_contract": _passing_context_slices(),
        "turning_point_evidence": _passing_turning_point_evidence(),
        "offline_rl_evidence": _passing_offline_rl_evidence(),
    }
    return {
        name: copy.deepcopy(split)
        for name in FOUNDATION_AUDIT_SMOKE_SPLITS
    }


__all__ = ["passing_smoke_audit_splits"]
