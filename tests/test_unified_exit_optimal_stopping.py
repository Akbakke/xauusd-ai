from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from gx1.contracts.unified_exit_optimal_stopping_v1 import (
    UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT,
    replay_unified_exit_q_policy,
    require_unified_exit_optimal_stopping_contract,
    unified_exit_optimal_stopping_contract,
    unified_exit_optimal_stopping_targets,
)


def _quotes() -> tuple[np.ndarray, np.ndarray]:
    count = UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT
    bid = np.linspace(100.0, 102.0, count, dtype=np.float64)
    ask = bid + 0.02
    return bid, ask


def test_optimal_stopping_contract_has_no_discount_margin_or_extra_lookahead() -> None:
    contract = unified_exit_optimal_stopping_contract()
    assert require_unified_exit_optimal_stopping_contract(
        contract, context="TEST"
    ) == contract
    assert contract["gamma"] == 1.0
    assert contract["hold_reward_bps"] == 0.0
    assert contract["static_margin_or_threshold"] is None
    assert contract["extra_lookahead_beyond_trajectory"] == 0
    assert contract["entry_and_exit_spread_included"] is True
    assert contract["financing_costs_included"] is False
    assert contract["slippage_included"] is False

    broken = deepcopy(contract)
    broken["gamma"] = 0.99
    with pytest.raises(RuntimeError, match="OPTIMAL_STOPPING_INVALID"):
        require_unified_exit_optimal_stopping_contract(broken, context="TEST")


@pytest.mark.parametrize("side_index", (0, 1))
def test_backward_dp_satisfies_exact_bellman_identities(side_index: int) -> None:
    bid, ask = _quotes()
    if side_index == 1:
        bid = bid[::-1].copy()
        ask = ask[::-1].copy()
    target = unified_exit_optimal_stopping_targets(
        side_index=side_index,
        entry_bid=99.98,
        entry_ask=100.02,
        bid_close=bid,
        ask_close=ask,
    )
    q = target["q_targets_bps"]
    value = target["state_value_bps"]
    assert np.array_equal(q[:-1, 0], value[1:])
    assert np.array_equal(q[:, 1], target["exit_pnl_bps"])
    assert np.array_equal(value[:-1], np.max(q[:-1], axis=1))
    assert target["action_valid_mask"][-1].tolist() == [False, True]
    assert target["action_equivalence_mask"][-1].tolist() == [False, True]
    assert target["action_equivalence_mask"][:-1, 0].all()


def test_long_short_use_executable_bid_ask_and_entry_spread() -> None:
    bid = np.full(512, 100.00, dtype=np.float64)
    ask = np.full(512, 100.04, dtype=np.float64)
    long = unified_exit_optimal_stopping_targets(
        side_index=0,
        entry_bid=99.98,
        entry_ask=100.02,
        bid_close=bid,
        ask_close=ask,
    )
    short = unified_exit_optimal_stopping_targets(
        side_index=1,
        entry_bid=99.98,
        entry_ask=100.02,
        bid_close=bid,
        ask_close=ask,
    )
    assert np.allclose(long["exit_pnl_bps"], (100.00 - 100.02) / 100.02 * 1e4)
    assert np.allclose(short["exit_pnl_bps"], (99.98 - 100.04) / 99.98 * 1e4)
    assert np.all(long["exit_pnl_bps"] < 0.0)
    assert np.all(short["exit_pnl_bps"] < 0.0)


def test_exact_action_equivalence_is_retained_without_tolerance() -> None:
    bid = np.full(512, 100.0, dtype=np.float64)
    ask = np.full(512, 100.02, dtype=np.float64)
    target = unified_exit_optimal_stopping_targets(
        side_index=0,
        entry_bid=99.98,
        entry_ask=100.02,
        bid_close=bid,
        ask_close=ask,
    )
    assert target["action_equivalence_mask"][:-1].all()
    assert target["action_equivalence_mask"][-1].tolist() == [False, True]


def test_q_policy_replay_exits_early_or_is_forced_at_terminal() -> None:
    bid, ask = _quotes()
    target = unified_exit_optimal_stopping_targets(
        side_index=0,
        entry_bid=99.98,
        entry_ask=100.02,
        bid_close=bid,
        ask_close=ask,
    )
    early = np.zeros((512, 2), dtype=np.float64)
    early[:, 0] = 1.0
    early[7, 1] = 2.0
    early_result = replay_unified_exit_q_policy(
        predicted_q_bps=early,
        targets=target,
    )
    assert early_result["exit_state_index"] == 7
    assert early_result["terminal_forced"] is False

    terminal = np.zeros((512, 2), dtype=np.float64)
    terminal[:, 0] = 1.0
    terminal[:, 1] = 0.0
    terminal_result = replay_unified_exit_q_policy(
        predicted_q_bps=terminal,
        targets=target,
    )
    assert terminal_result["exit_state_index"] == 511
    assert terminal_result["terminal_forced"] is True


def test_q_policy_exact_tie_fails_closed_without_hidden_preference() -> None:
    bid, ask = _quotes()
    target = unified_exit_optimal_stopping_targets(
        side_index=0,
        entry_bid=99.98,
        entry_ask=100.02,
        bid_close=bid,
        ask_close=ask,
    )
    tied = np.zeros((512, 2), dtype=np.float64)
    with pytest.raises(RuntimeError, match="TIED_ACTION"):
        replay_unified_exit_q_policy(
            predicted_q_bps=tied,
            targets=target,
        )


def test_future_quote_mutation_changes_targets_but_never_contract_inputs() -> None:
    bid, ask = _quotes()
    baseline = unified_exit_optimal_stopping_targets(
        side_index=0,
        entry_bid=99.98,
        entry_ask=100.02,
        bid_close=bid,
        ask_close=ask,
    )
    changed_bid = bid.copy()
    changed_bid[-1] += 0.01
    changed = unified_exit_optimal_stopping_targets(
        side_index=0,
        entry_bid=99.98,
        entry_ask=100.02,
        bid_close=changed_bid,
        ask_close=ask,
    )
    assert baseline["target_sha256"] != changed["target_sha256"]
    assert unified_exit_optimal_stopping_contract()[
        "future_outcomes_used_as_model_inputs"
    ] is False
