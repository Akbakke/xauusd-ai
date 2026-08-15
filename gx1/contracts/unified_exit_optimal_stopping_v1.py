"""Exact finite-horizon optimal-stopping targets for unified Exit.

The model state remains causal.  This owner consumes future executable quotes
only while constructing supervision for one already-declared 512-state
trajectory.  HOLD has zero reward and transitions to the next state; EXIT_NOW
terminates at the current executable quote.  The terminal state admits only
EXIT_NOW.  No discount, margin, indifference band or lookahead window exists.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_ACTION_ORDER,
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_SIDE_ORDER,
)


UNIFIED_EXIT_OPTIMAL_STOPPING_SCHEMA_VERSION = (
    "gx1_unified_exit_finite_horizon_optimal_stopping_v1"
)
UNIFIED_EXIT_OPTIMAL_STOPPING_TARGET_STREAM_SCHEMA_VERSION = (
    "gx1_unified_exit_optimal_stopping_target_stream_v1"
)
UNIFIED_EXIT_OPTIMAL_STOPPING_GAMMA = 1.0
UNIFIED_EXIT_OPTIMAL_STOPPING_TARGET_UNIT = "raw_bps"
UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT = UNIFIED_EXIT_MAX_PATH_BARS
UNIFIED_EXIT_OPTIMAL_STOPPING_ACTION_COUNT = len(UNIFIED_EXIT_ACTION_ORDER)
UNIFIED_EXIT_HOLD_ACTION_INDEX = UNIFIED_EXIT_ACTION_ORDER.index("HOLD")
UNIFIED_EXIT_EXIT_NOW_ACTION_INDEX = UNIFIED_EXIT_ACTION_ORDER.index("EXIT_NOW")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def unified_exit_optimal_stopping_contract() -> dict[str, Any]:
    payload = {
        "schema_version": UNIFIED_EXIT_OPTIMAL_STOPPING_SCHEMA_VERSION,
        "state_count": UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT,
        "action_order": list(UNIFIED_EXIT_ACTION_ORDER),
        "side_order": list(UNIFIED_EXIT_SIDE_ORDER),
        "target_unit": UNIFIED_EXIT_OPTIMAL_STOPPING_TARGET_UNIT,
        "gamma": UNIFIED_EXIT_OPTIMAL_STOPPING_GAMMA,
        "hold_reward_bps": 0.0,
        "hold_transition": "deterministic_next_authoritative_state",
        "exit_reward": "terminal_executable_trade_pnl_bps_at_current_state",
        "terminal_state_index": UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT - 1,
        "terminal_valid_actions": ["EXIT_NOW"],
        "long_entry_price": "entry_ask",
        "long_exit_price": "bid_close",
        "short_entry_price": "entry_bid",
        "short_exit_price": "ask_close",
        "entry_and_exit_spread_included": True,
        "financing_costs_included": False,
        "slippage_included": False,
        "future_outcomes_used_as_model_inputs": False,
        "static_margin_or_threshold": None,
        "extra_lookahead_beyond_trajectory": 0,
        "action_equivalence": "exact_float64_q_equality_no_tolerance",
        "invalid_terminal_hold_target": (
            "finite_exit_value_copy_masked_out_of_q_loss_and_policy"
        ),
        "bellman_equations": {
            "q_exit": "executable_trade_pnl_bps(state)",
            "q_hold_nonterminal": "0 + V(next_state)",
            "v_nonterminal": "max(q_hold,q_exit)",
            "v_terminal": "q_exit",
        },
    }
    payload["contract_sha256"] = _canonical_json_sha256(payload)
    return payload


def require_unified_exit_optimal_stopping_contract(
    value: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    expected = unified_exit_optimal_stopping_contract()
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise RuntimeError(f"[{context}_UNIFIED_EXIT_OPTIMAL_STOPPING_INVALID]")
    return expected


def _finite_entry_quotes(entry_bid: Any, entry_ask: Any) -> tuple[float, float]:
    try:
        bid = float(entry_bid)
        ask = float(entry_ask)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("UNIFIED_EXIT_OPTIMAL_ENTRY_QUOTES_INVALID") from exc
    if not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0.0 or ask <= bid:
        raise RuntimeError("UNIFIED_EXIT_OPTIMAL_ENTRY_QUOTES_INVALID")
    return bid, ask


def _finite_close_vector(value: Any, *, label: str) -> np.ndarray:
    raw = np.asarray(value)
    try:
        observed = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"UNIFIED_EXIT_OPTIMAL_{label}_INVALID"
        ) from exc
    if (
        observed.shape != (UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT,)
        or not np.isfinite(observed).all()
        or np.any(observed <= 0.0)
    ):
        raise RuntimeError(f"UNIFIED_EXIT_OPTIMAL_{label}_INVALID")
    return np.ascontiguousarray(observed)


def _target_sha256(
    *,
    side_index: int,
    entry_bid: float,
    entry_ask: float,
    arrays: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "schema_version": UNIFIED_EXIT_OPTIMAL_STOPPING_SCHEMA_VERSION,
                "side_index": side_index,
                "entry_bid": entry_bid,
                "entry_ask": entry_ask,
                "ordered_arrays": [
                    "exit_pnl_bps",
                    "q_targets_bps",
                    "state_value_bps",
                    "action_valid_mask",
                    "action_equivalence_mask",
                    "terminal_mask",
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    )
    digest.update(b"\x00")
    for name in (
        "exit_pnl_bps",
        "q_targets_bps",
        "state_value_bps",
    ):
        digest.update(np.ascontiguousarray(arrays[name], dtype="<f8").tobytes())
    for name in (
        "action_valid_mask",
        "action_equivalence_mask",
        "terminal_mask",
    ):
        digest.update(np.ascontiguousarray(arrays[name], dtype="u1").tobytes())
    return digest.hexdigest()


def unified_exit_optimal_stopping_targets(
    *,
    side_index: int,
    entry_bid: Any,
    entry_ask: Any,
    bid_close: Any,
    ask_close: Any,
) -> dict[str, Any]:
    """Backward-DP targets for one exact 512-state executable trajectory."""

    if isinstance(side_index, bool) or not isinstance(
        side_index, (int, np.integer)
    ) or int(side_index) not in (0, 1):
        raise RuntimeError("UNIFIED_EXIT_OPTIMAL_SIDE_INVALID")
    side = int(side_index)
    bid, ask = _finite_entry_quotes(entry_bid, entry_ask)
    bids = _finite_close_vector(bid_close, label="BID_CLOSE")
    asks = _finite_close_vector(ask_close, label="ASK_CLOSE")
    if np.any(asks <= bids):
        raise RuntimeError("UNIFIED_EXIT_OPTIMAL_EXECUTABLE_SPREAD_INVALID")

    exit_pnl = (
        (bids - ask) / ask * 10_000.0
        if side == 0
        else (bid - asks) / bid * 10_000.0
    )
    values = np.empty_like(exit_pnl)
    q_targets = np.empty(
        (
            UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT,
            UNIFIED_EXIT_OPTIMAL_STOPPING_ACTION_COUNT,
        ),
        dtype=np.float64,
    )
    action_valid = np.ones(q_targets.shape, dtype=np.bool_)
    terminal = np.zeros(
        UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT,
        dtype=np.bool_,
    )
    terminal[-1] = True
    action_valid[-1, UNIFIED_EXIT_HOLD_ACTION_INDEX] = False

    values[-1] = exit_pnl[-1]
    q_targets[-1] = exit_pnl[-1]
    for state in range(UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT - 2, -1, -1):
        q_targets[state, UNIFIED_EXIT_HOLD_ACTION_INDEX] = values[state + 1]
        q_targets[state, UNIFIED_EXIT_EXIT_NOW_ACTION_INDEX] = exit_pnl[state]
        values[state] = max(q_targets[state])

    equivalence = np.equal(q_targets, values[:, None]) & action_valid
    if (
        not np.isfinite(exit_pnl).all()
        or not np.isfinite(q_targets).all()
        or not np.isfinite(values).all()
        or not equivalence.any(axis=1).all()
        or equivalence[-1].tolist() != [False, True]
        or not np.array_equal(q_targets[:-1, 0], values[1:])
        or not np.array_equal(q_targets[:, 1], exit_pnl)
        or not np.array_equal(values[:-1], np.max(q_targets[:-1], axis=1))
        or values[-1] != exit_pnl[-1]
    ):
        raise RuntimeError("UNIFIED_EXIT_OPTIMAL_BELLMAN_IDENTITY_INVALID")
    arrays = {
        "exit_pnl_bps": np.ascontiguousarray(exit_pnl),
        "q_targets_bps": np.ascontiguousarray(q_targets),
        "state_value_bps": np.ascontiguousarray(values),
        "action_valid_mask": np.ascontiguousarray(action_valid),
        "action_equivalence_mask": np.ascontiguousarray(equivalence),
        "terminal_mask": np.ascontiguousarray(terminal),
    }
    return {
        "schema_version": UNIFIED_EXIT_OPTIMAL_STOPPING_SCHEMA_VERSION,
        "side_index": side,
        "side": UNIFIED_EXIT_SIDE_ORDER[side],
        "entry_bid": bid,
        "entry_ask": ask,
        **arrays,
        "target_sha256": _target_sha256(
            side_index=side,
            entry_bid=bid,
            entry_ask=ask,
            arrays=arrays,
        ),
    }


def update_unified_exit_optimal_stopping_target_stream(
    digest: Any,
    *,
    episode_index: int,
    entry_row_index: int,
    m1_start_row: int,
    targets: Mapping[str, Any],
) -> None:
    if not hasattr(digest, "update"):
        raise RuntimeError("UNIFIED_EXIT_OPTIMAL_STREAM_DIGEST_INVALID")
    if targets.get("schema_version") != UNIFIED_EXIT_OPTIMAL_STOPPING_SCHEMA_VERSION:
        raise RuntimeError("UNIFIED_EXIT_OPTIMAL_STREAM_TARGET_INVALID")
    target_sha = targets.get("target_sha256")
    if not isinstance(target_sha, str) or len(target_sha) != 64:
        raise RuntimeError("UNIFIED_EXIT_OPTIMAL_STREAM_TARGET_INVALID")
    digest.update(
        np.asarray(
            [
                int(episode_index),
                int(entry_row_index),
                int(targets["side_index"]),
                int(m1_start_row),
            ],
            dtype="<i8",
        ).tobytes()
    )
    digest.update(bytes.fromhex(target_sha))


def replay_unified_exit_q_policy(
    *,
    predicted_q_bps: Any,
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay one learned Q policy until EXIT_NOW or the forced terminal."""

    q = np.asarray(predicted_q_bps, dtype=np.float64)
    valid = np.asarray(targets.get("action_valid_mask"), dtype=np.bool_)
    exit_pnl = np.asarray(targets.get("exit_pnl_bps"), dtype=np.float64)
    expected_q_shape = (
        UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT,
        UNIFIED_EXIT_OPTIMAL_STOPPING_ACTION_COUNT,
    )
    if (
        q.shape != expected_q_shape
        or valid.shape != expected_q_shape
        or exit_pnl.shape != (UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT,)
        or not np.isfinite(q).all()
        or not np.isfinite(exit_pnl).all()
    ):
        raise RuntimeError("UNIFIED_EXIT_Q_POLICY_REPLAY_INPUT_INVALID")
    actions: list[int] = []
    for state in range(UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT):
        valid_indices = np.flatnonzero(valid[state])
        if valid_indices.size == 0:
            raise RuntimeError("UNIFIED_EXIT_Q_POLICY_ACTION_MASK_EMPTY")
        state_q = q[state, valid_indices]
        if np.count_nonzero(state_q == np.max(state_q)) != 1:
            raise RuntimeError("UNIFIED_EXIT_Q_POLICY_TIED_ACTION")
        action = int(valid_indices[int(np.argmax(state_q))])
        actions.append(action)
        if action == UNIFIED_EXIT_EXIT_NOW_ACTION_INDEX:
            return {
                "exit_state_index": state,
                "action_indices": actions,
                "realized_executable_pnl_bps": float(exit_pnl[state]),
                "terminal_forced": state
                == UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT - 1,
            }
    raise RuntimeError("UNIFIED_EXIT_Q_POLICY_DID_NOT_TERMINATE")


__all__ = (
    "UNIFIED_EXIT_EXIT_NOW_ACTION_INDEX",
    "UNIFIED_EXIT_HOLD_ACTION_INDEX",
    "UNIFIED_EXIT_OPTIMAL_STOPPING_GAMMA",
    "UNIFIED_EXIT_OPTIMAL_STOPPING_SCHEMA_VERSION",
    "UNIFIED_EXIT_OPTIMAL_STOPPING_STATE_COUNT",
    "UNIFIED_EXIT_OPTIMAL_STOPPING_TARGET_STREAM_SCHEMA_VERSION",
    "replay_unified_exit_q_policy",
    "require_unified_exit_optimal_stopping_contract",
    "unified_exit_optimal_stopping_contract",
    "unified_exit_optimal_stopping_targets",
    "update_unified_exit_optimal_stopping_target_stream",
)
