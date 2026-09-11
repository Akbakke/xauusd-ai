"""Couple the same EMA model's Entry decisions to its completed Exit outcomes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_ACTION_ORDER, replay_entry_fitted_q_policy,
)
from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    canonical_sha256,
)

SCHEMA_VERSION = "gx1_unified_exit_entry_policy_decisions_v1"


def build_entry_policy_decisions(
    *, predicted_q_bps: Any, entry_row_indices: Sequence[int],
    checkpoint_binding_sha256: str,
) -> dict[str, Any]:
    q = np.asarray(predicted_q_bps)
    rows = list(entry_row_indices)
    if (
        q.dtype != np.float32 or q.shape != (len(rows), 3) or not rows
        or any(type(row) is not int or row < 0 for row in rows)
        or rows != sorted(set(rows))
        or not isinstance(checkpoint_binding_sha256, str)
        or len(checkpoint_binding_sha256) != 64
        or any(c not in "0123456789abcdef" for c in checkpoint_binding_sha256)
    ):
        raise RuntimeError("UNIFIED_EXIT_ENTRY_POLICY_INPUT_INVALID")
    # The bound pair cohort contains an executable first state for both sides.
    # FLAT is always valid and terminates with zero return.
    actions = replay_entry_fitted_q_policy(
        predicted_q_bps=q, action_valid_mask=np.ones(q.shape, dtype=np.bool_),
    )
    value = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_binding_sha256": checkpoint_binding_sha256,
        "model_variant": "weight_ema",
        "action_order": list(ENTRY_FITTED_Q_ACTION_ORDER),
        "action_validity": "bound_both_side_pair_cohort_plus_flat",
        "entry_row_indices": rows,
        "entry_action_q_bps": q.tolist(),
        "action_indices": actions.tolist(),
        "tie_policy": "fail_closed",
        "test_data_used": False,
    }
    value["policy_sha256"] = canonical_sha256(value)
    return value


def require_entry_policy_decisions(
    value: Mapping[str, Any], *, entry_row_indices: Sequence[int],
    checkpoint_binding_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("UNIFIED_EXIT_ENTRY_POLICY_BINDING_INVALID")
    rebuilt = build_entry_policy_decisions(
        predicted_q_bps=np.asarray(value.get("entry_action_q_bps"), dtype=np.float32),
        entry_row_indices=entry_row_indices,
        checkpoint_binding_sha256=checkpoint_binding_sha256,
    )
    if rebuilt != dict(value):
        raise RuntimeError("UNIFIED_EXIT_ENTRY_POLICY_BINDING_INVALID")
    return rebuilt


def coupled_entry_exit_policy_metrics(
    *, entry_policy: Mapping[str, Any],
    trade_outcomes: Sequence[Mapping[str, Any]],
    full_cohort_authoritative: bool,
) -> dict[str, Any]:
    entry_policy = require_entry_policy_decisions(
        entry_policy, entry_row_indices=entry_policy["entry_row_indices"],
        checkpoint_binding_sha256=entry_policy["checkpoint_binding_sha256"],
    )
    if type(full_cohort_authoritative) is not bool:
        raise RuntimeError("UNIFIED_EXIT_ENTRY_POLICY_AUTHORITY_INVALID")
    rows = entry_policy["entry_row_indices"]
    actions = entry_policy["action_indices"]
    lookup = {}
    for outcome in trade_outcomes:
        key = (outcome["entry_row_index"], outcome["side_index"])
        if key in lookup:
            raise RuntimeError("UNIFIED_EXIT_ENTRY_POLICY_DUPLICATE_OUTCOME")
        lookup[key] = outcome
    if set(lookup) != {(row, side) for row in rows for side in range(2)}:
        raise RuntimeError("UNIFIED_EXIT_ENTRY_POLICY_OUTCOME_COHORT_INVALID")
    selected = [
        lookup[(row, action)] for row, action in zip(rows, actions) if action != 2
    ]
    values = [float(row["undiscounted_net_cash_pnl_bps"]) for row in selected]
    if not all(np.isfinite(values)):
        raise RuntimeError("UNIFIED_EXIT_ENTRY_POLICY_PNL_INVALID")
    complete = all(row["status"] == "EXITED" for row in selected)
    authoritative = full_cohort_authoritative and complete
    total = float(sum(values)) if authoritative else None
    return {
        "semantics": "equal_notional_independent_entry_opportunities_not_portfolio_return",
        "policy_sha256": entry_policy["policy_sha256"],
        "checkpoint_binding_sha256": entry_policy["checkpoint_binding_sha256"],
        "entry_decision_count": len(rows),
        "long_count": actions.count(0),
        "short_count": actions.count(1),
        "flat_count": actions.count(2),
        "selected_trade_count": len(selected),
        "selected_exited_count": sum(row["status"] == "EXITED" for row in selected),
        "selected_non_exited_count": sum(row["status"] != "EXITED" for row in selected),
        "full_cohort_authoritative": authoritative,
        "net_bps_sum": total,
        "mean_net_bps_per_entry": total / len(rows) if authoritative else None,
        "mean_net_bps_per_selected_trade": (
            total / len(selected) if authoritative and selected else None
        ),
        "positive_net_bps_observed": (
            authoritative and bool(selected) and total > 0
        ),
        "flat_return_bps": 0.0,
        "test_data_used": False,
    }
