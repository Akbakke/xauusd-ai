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
