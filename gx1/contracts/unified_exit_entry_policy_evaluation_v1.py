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



def marked_entry_exit_policy_metrics(
    *, entry_policy: Mapping[str, Any], trade_outcomes: Sequence[Mapping[str, Any]],
    full_cohort_authoritative: bool,
) -> dict[str, Any]:
    """Report observed liquidation NAV and a fixed-notional one-position replay.

    This reports the proposed execution rule; it does not change Entry targets
    or authorize checkpoint selection. Unknown gaps cannot stand in for month end.
    """
    original = coupled_entry_exit_policy_metrics(
        entry_policy=entry_policy, trade_outcomes=trade_outcomes,
        full_cohort_authoritative=full_cohort_authoritative,
    )
    lookup = {(o["entry_row_index"], o["side_index"]): o for o in trade_outcomes}
    row_ids, actions = entry_policy["entry_row_indices"], entry_policy["action_indices"]
    times = []
    for row in row_ids:
        time = lookup[row, 0].get("entry_fill_time_ns")
        if type(time) is not int or time <= 0 or lookup[row, 1].get("entry_fill_time_ns") != time:
            raise RuntimeError("UNIFIED_EXIT_MARKED_ENTRY_CLOCK_INVALID")
        times.append(time)
    if times != sorted(times):
        raise RuntimeError("UNIFIED_EXIT_MARKED_ENTRY_ORDER_INVALID")
    selected = [lookup[row, side] for row, side in zip(row_ids, actions) if side != 2]

    def total_for(outcomes):
        values, complete = [], full_cohort_authoritative
        for outcome in outcomes:
            mark = outcome.get("valuation")
            if not isinstance(mark, Mapping):
                raise RuntimeError("UNIFIED_EXIT_MARKED_VALUATION_MISSING")
            value = mark.get("net_cash_plus_open_value_bps")
            remaining = mark.get("remaining_liquidation_value_bps")
            eligible = outcome["status"] in {"EXITED", "RIGHT_CENSORED_SPLIT_END", "RIGHT_CENSORED_OBSERVATION_CUTOFF"}
            if eligible:
                if outcome["status"] == "RIGHT_CENSORED_OBSERVATION_CUTOFF" and (
                        type(mark.get("observation_cutoff_time_ns")) is not int
                        or type(mark.get("valuation_time_ns")) is not int
                        or mark["valuation_time_ns"] > mark["observation_cutoff_time_ns"]):
                    raise RuntimeError("UNIFIED_EXIT_MARKED_OBSERVATION_CUTOFF_INVALID")
                if (value is None or remaining is None or not np.isfinite(value)
                    or not np.isfinite(remaining)
                    or value != float(outcome["undiscounted_net_cash_pnl_bps"]) + remaining):
                    raise RuntimeError("UNIFIED_EXIT_MARKED_ACCOUNTING_INVALID")
                if (type(mark.get("valuation_time_ns")) is not int
                    or mark["valuation_time_ns"] < outcome["entry_fill_time_ns"]
                    or mark.get("model_exit_executed") is not (outcome["status"] == "EXITED")):
                    raise RuntimeError("UNIFIED_EXIT_MARKED_VALUATION_INVALID")
                if outcome["status"] == "EXITED" and (remaining != 0.0 or mark["valuation_time_ns"] != outcome.get("exit_decision_time_ns")):
                    raise RuntimeError("UNIFIED_EXIT_MARKED_REALIZED_VALUE_INVALID")
                values.append(float(value))
            else:
                complete = False
        return (float(sum(values)) if complete else None), bool(complete)

    independent_total, independent_complete = total_for(selected)
    executed, occupied_until, skipped = [], -1, 0
    for row, side, time in zip(row_ids, actions, times):
        if side == 2:
            continue
        if occupied_until is None or time < occupied_until:
            skipped += 1
            continue
        outcome = lookup[row, side]
        executed.append(outcome)
        # EXIT is processed before an Entry fill at the same timestamp.
        if outcome["status"] == "EXITED":
            occupied_until = outcome.get("exit_decision_time_ns")
            if type(occupied_until) is not int or occupied_until < time:
                raise RuntimeError("UNIFIED_EXIT_MARKED_EXIT_CLOCK_INVALID")
        else:
            occupied_until = None
    chronological_total, chronological_complete = total_for(executed)
    return {
        "schema_version": "gx1_entry_exit_marked_policy_metrics_v1",
        "policy_sha256": original["policy_sha256"],
        "checkpoint_binding_sha256": original["checkpoint_binding_sha256"],
        "valuation_basis": "observed_executable_liquidation_value_after_costs",
        "model_exit_fabricated": False,
        "used_for_early_stopping": False,
        "independent_opportunities": {
            "semantics": "equal_notional_opportunities_not_portfolio_return",
            "full_cohort_authoritative": independent_complete,
            "net_cash_plus_open_value_bps_sum": independent_total,
            "mean_bps_per_entry_opportunity": independent_total / len(row_ids) if independent_complete else None,
            "selected_trade_count": len(selected),
        },
        "single_position_replay": {
            "semantics": "single_position_fixed_notional_bps_not_compounded_account_return",
            "execution_order": "exit_before_entry_at_equal_timestamp_then_entry_row_order",
            "position_rule_trained": False,
            "full_cohort_authoritative": chronological_complete,
            "net_cash_plus_open_value_bps_sum": chronological_total,
            "executed_entry_row_indices": [o["entry_row_index"] for o in executed],
            "executed_trade_count": len(executed),
            "skipped_while_position_open_count": skipped,
            "entry_flat_decision_count": actions.count(2),
            "model_exited_count": sum(o["status"] == "EXITED" for o in executed),
            "open_position_count": int(bool(executed) and executed[-1]["status"] != "EXITED"),
        },
        "test_data_used": False,
    }
