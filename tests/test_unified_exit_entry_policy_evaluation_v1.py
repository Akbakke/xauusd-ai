import copy

import numpy as np
import pytest

from gx1.contracts.unified_exit_entry_policy_evaluation_v1 import (
    build_entry_policy_decisions, coupled_entry_exit_policy_metrics,
    require_entry_policy_decisions,
)


def _policy():
    return build_entry_policy_decisions(
        predicted_q_bps=np.array([[2, 1, 0], [0, 2, 1], [0, 1, 2]], dtype=np.float32),
        entry_row_indices=[0, 1, 2], checkpoint_binding_sha256="a" * 64,
    )


def _outcomes():
    returns = [[-5.0, 100.0], [100.0, 2.0], [100.0, 100.0]]
    return [
        {"entry_row_index": row, "side_index": side, "status": "EXITED",
         "undiscounted_net_cash_pnl_bps": returns[row][side]}
        for row in range(3) for side in range(2)
    ]


def test_entry_selected_side_and_flat_determine_return():
    result = coupled_entry_exit_policy_metrics(
        entry_policy=_policy(), trade_outcomes=_outcomes(), full_cohort_authoritative=True,
    )
    assert result["net_bps_sum"] == -3.0
    assert result["mean_net_bps_per_entry"] == -1.0
    assert result["mean_net_bps_per_selected_trade"] == -1.5
    assert result["flat_count"] == 1
    assert result["selected_trade_count"] == 2
    assert result["positive_net_bps_observed"] is False


def test_partial_rollout_never_produces_authoritative_policy_bps():
    rows = _outcomes()
    rows[0]["status"] = "RIGHT_CENSORED_DATA_BOUNDARY"
    result = coupled_entry_exit_policy_metrics(
        entry_policy=_policy(), trade_outcomes=rows, full_cohort_authoritative=False,
    )
    assert result["net_bps_sum"] is None
    assert result["mean_net_bps_per_entry"] is None
    assert result["positive_net_bps_observed"] is False


def test_ties_changed_actions_and_duplicate_outcomes_fail_closed():
    with pytest.raises(RuntimeError, match="TIED_ACTION"):
        build_entry_policy_decisions(
            predicted_q_bps=np.array([[1, 1, 0]], dtype=np.float32),
            entry_row_indices=[0], checkpoint_binding_sha256="a" * 64,
        )
    changed = copy.deepcopy(_policy())
    changed["action_indices"][0] = 1
    with pytest.raises(RuntimeError, match="BINDING_INVALID"):
        require_entry_policy_decisions(
            changed, entry_row_indices=[0, 1, 2], checkpoint_binding_sha256="a" * 64,
        )
    with pytest.raises(RuntimeError, match="DUPLICATE_OUTCOME"):
        coupled_entry_exit_policy_metrics(
            entry_policy=_policy(), trade_outcomes=_outcomes() + [_outcomes()[0]],
            full_cohort_authoritative=True,
        )



def _marked_case():
    from gx1.contracts.unified_exit_entry_policy_evaluation_v1 import marked_entry_exit_policy_metrics
    actions = [0, 1, 2, 1, 0]
    q = np.zeros((5, 3), dtype=np.float32)
    q[np.arange(5), actions] = 2.0
    policy = build_entry_policy_decisions(predicted_q_bps=q, entry_row_indices=list(range(5)),
                                          checkpoint_binding_sha256="a" * 64)
    times = [100, 150, 200, 200, 350]
    end = [200, 250, 220, 300, None]
    cash = [5.0, 1000.0, 0.0, 2.0, -0.5]
    outcomes = []
    for row in range(5):
        for side in (0, 1):
            closed = end[row] is not None
            remaining = 0.0 if closed else -20.0
            outcomes.append(dict(entry_row_index=row, side_index=side,
                status="EXITED" if closed else "RIGHT_CENSORED_SPLIT_END",
                entry_fill_time_ns=times[row], exit_decision_time_ns=end[row],
                undiscounted_net_cash_pnl_bps=cash[row],
                valuation=dict(remaining_liquidation_value_bps=remaining,
                    net_cash_plus_open_value_bps=cash[row]+remaining,
                    valuation_time_ns=end[row] if closed else 400,
                    model_exit_executed=closed)))
    return policy, outcomes, marked_entry_exit_policy_metrics


def test_open_losses_count_and_one_position_excludes_overlapping_winners():
    policy, outcomes, measure = _marked_case()
    result = measure(entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=True)
    assert result["independent_opportunities"]["net_cash_plus_open_value_bps_sum"] == 986.5
    sequential = result["single_position_replay"]
    assert sequential["executed_entry_row_indices"] == [0, 3, 4]
    assert sequential["net_cash_plus_open_value_bps_sum"] == -13.5
    assert sequential["skipped_while_position_open_count"] == 1
    assert sequential["entry_flat_decision_count"] == 1
    assert sequential["open_position_count"] == 1
    assert sequential["model_exited_count"] == 2
    assert result["used_for_early_stopping"] is False
    assert coupled_entry_exit_policy_metrics(entry_policy=policy, trade_outcomes=outcomes,
        full_cohort_authoritative=True)["net_bps_sum"] is None
    # No hindsight selection: changing a skipped trade cannot improve the replay.
    for row in outcomes:
        if row["entry_row_index"] == 1:
            row["undiscounted_net_cash_pnl_bps"] = -1000.0
            row["valuation"]["net_cash_plus_open_value_bps"] = -1000.0
    assert measure(entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=True)["single_position_replay"] == sequential


@pytest.mark.parametrize("status, complete", [("RIGHT_CENSORED_UNKNOWN_SOURCE_GAP", True),
                                              ("TRUNCATED_COMPUTE_GUARD_WALL", False)])
def test_unknown_future_or_incomplete_compute_cannot_become_complete_nav(status, complete):
    policy, outcomes, measure = _marked_case()
    for row in outcomes:
        if row["entry_row_index"] == 4:
            row["status"] = status
    result = measure(entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=complete)
    for key in ("independent_opportunities", "single_position_replay"):
        assert result[key]["full_cohort_authoritative"] is False
        assert result[key]["net_cash_plus_open_value_bps_sum"] is None


def test_marked_accounting_and_entry_order_are_validated():
    policy, outcomes, measure = _marked_case()
    outcomes[-2]["valuation"]["net_cash_plus_open_value_bps"] += 1.0
    with pytest.raises(RuntimeError, match="MARKED_ACCOUNTING_INVALID"):
        measure(entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=True)
    policy, outcomes, measure = _marked_case()
    for row in outcomes:
        if row["entry_row_index"] == 1:
            row["entry_fill_time_ns"] = 50
    with pytest.raises(RuntimeError, match="MARKED_ENTRY_ORDER_INVALID"):
        measure(entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=True)
