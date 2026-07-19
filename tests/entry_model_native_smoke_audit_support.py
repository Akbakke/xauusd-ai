from __future__ import annotations

import copy
import math
from typing import Any

from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_SMOKE_SPLITS,
    foundation_audit_policy_metadata,
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
    }
    return {
        name: copy.deepcopy(split)
        for name in FOUNDATION_AUDIT_SMOKE_SPLITS
    }


__all__ = ["passing_smoke_audit_splits"]
