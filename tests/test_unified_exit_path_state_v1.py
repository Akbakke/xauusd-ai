from __future__ import annotations

import json
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

from gx1.models.entry_v10.direction_decision_contract import (
    CLOSED_M1_PATH_SCHEMA_VERSION,
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_PATH_FEATURE_DIM,
    canonical_unified_evidence_sha256,
    unified_exit_path_tensor,
)
from gx1.replay.unified_exit_path_state_v1 import (
    UnifiedExitPathState,
    first_full_closed_m1_bar_ts,
)
from tests.model_native_sizing_support import (
    runtime_head_prediction_columns,
)
from tests.unified_exit_input_support import (
    unified_exit_carry_fixture,
    unified_exit_input_fixture,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
)


_ENTRY_FILL = pd.Timestamp("2026-07-17T12:00:00Z")
_BUNDLE_SHA256 = "b" * 64


def _runtime_head(direction: str = "LONG") -> dict[str, object]:
    q_values = [3.0, 1.0, 0.0] if direction == "LONG" else [1.0, 3.0, 0.0]
    direction_index = 0 if direction == "LONG" else 1
    row = {
        "time": _ENTRY_FILL - pd.Timedelta(minutes=5),
        "entry_action_q_bps": q_values,
        "entry_action_q_margin_bps": 2.0,
        "model_direction_index": direction_index,
        "model_direction": direction,
        "session_id": 2,
        "session": "OVERLAP",
        "position_size_logit": -0.1,
        "position_size_pred": float(1.0 / (1.0 + np.exp(0.1))),
        "timing_pred": [0.0] * 12,
    }
    columns = runtime_head_prediction_columns(pd.DataFrame([row]))
    return json.loads(columns["runtime_head_evidence_json"][0])


def _open(direction: str = "LONG") -> UnifiedExitPathState:
    return UnifiedExitPathState.open_unit_normalized_research(
        entry_ts=_ENTRY_FILL,
        side="long" if direction == "LONG" else "short",
        entry_bid=3300.0,
        entry_ask=3300.2,
        v10_snapshot=_runtime_head(direction),
        replay_id=f"unit-{direction.lower()}",
        normalization_contract="unit_normalized_direction_exit_research_v1",
        model_identity_kind="bundle_sha256",
        model_identity_sha256=_BUNDLE_SHA256,
        input_normalization_sha256="6" * 64,
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
    )


def _closed_bar(timestamp: str, *, offset: float = 0.0) -> dict[str, object]:
    return {
        "schema_version": CLOSED_M1_PATH_SCHEMA_VERSION,
        "time": pd.Timestamp(timestamp).isoformat(),
        "complete": True,
        "source_path": "/immutable/unit/xauusd_m1.parquet",
        "source_sha256": "a" * 64,
        "bid_open": 3300.0 + offset,
        "bid_high": 3301.0 + offset,
        "bid_low": 3299.5 + offset,
        "bid_close": 3300.5 + offset,
        "ask_open": 3300.2 + offset,
        "ask_high": 3301.2 + offset,
        "ask_low": 3299.7 + offset,
        "ask_close": 3300.7 + offset,
        "mid_open": 3300.1 + offset,
        "mid_high": 3301.1 + offset,
        "mid_low": 3299.6 + offset,
        "mid_close": 3300.6 + offset,
        "volume": 12,
    }


def _exit_decision(
    state: UnifiedExitPathState,
    *,
    q_values: tuple[float, float] = (1.0, 0.0),
) -> dict[str, object]:
    envelope = state.build_closed_m1_path_evidence()
    input_envelope = unified_exit_input_fixture(
        entry_snapshot=state.entry_snapshot,
        exit_path_envelope=envelope,
        bundle_sha256=_BUNDLE_SHA256,
        decision_identity=state.replay_id,
        side=state.side,
        entry_bid=state.entry_bid,
        entry_ask=state.entry_ask,
        entry_decision_token_snapshot=(
            state.entry_decision_token_snapshot
        ),
    )
    action_index = 0 if q_values[0] > q_values[1] else 1
    decision: dict[str, object] = {
        "exit_action_q_bps": list(q_values),
        "exit_action_valid_mask": [True, True],
        "exit_action_index": action_index,
        "action": ("HOLD", "EXIT_NOW")[action_index],
        "decision_source": "unified_model",
        "exit_input_envelope": input_envelope,
        "exit_incremental_carry_envelope": unified_exit_carry_fixture(
            input_envelope=input_envelope,
            exit_path_envelope=envelope,
            previous_carry_envelope=(
                state.last_exit_decision[
                    "exit_incremental_carry_envelope"
                ]
                if state.last_exit_decision is not None
                else None
            ),
        ),
        "bundle_sha256": _BUNDLE_SHA256,
        "entry_snapshot_sha256": canonical_unified_evidence_sha256(
            state.entry_snapshot
        ),
        "exit_path_envelope_sha256": canonical_unified_evidence_sha256(
            envelope
        ),
        "exit_input_envelope_sha256": input_envelope[
            "input_envelope_sha256"
        ],
    }
    decision["output_evidence_sha256"] = canonical_unified_evidence_sha256(
        decision
    )
    return decision


def test_first_full_closed_m1_bar_timestamp_is_exact_and_utc_only() -> None:
    assert first_full_closed_m1_bar_ts(
        pd.Timestamp("2026-07-17T12:00:00Z")
    ) == pd.Timestamp("2026-07-17T12:00:00Z")
    assert first_full_closed_m1_bar_ts(
        pd.Timestamp("2026-07-17T12:00:00.001Z")
    ) == pd.Timestamp("2026-07-17T12:01:00Z")
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        first_full_closed_m1_bar_ts(pd.Timestamp("2026-07-17T12:00:00"))


def test_unit_path_state_stages_hash_binds_and_commits_one_bar() -> None:
    assert {field.name for field in fields(UnifiedExitPathState)} == {
        "replay_id",
        "entry_fill_ts",
        "side",
        "entry_bid",
        "entry_ask",
        "entry_snapshot",
        "entry_decision_token_snapshot",
        "normalization_contract",
        "bars_in_trade",
        "last_processed_m1_ts",
        "current_bid",
        "current_ask",
        "current_pnl_bps",
            "closed_m1_path",
            "full_path_chain_sha256",
            "last_exit_decision",
        "last_exit_input_envelope",
    }
    state = _open()
    staged = state.clone_for_exit_decision()
    staged.update_bar(**_closed_bar("2026-07-17T12:00:00Z"))

    assert state.bars_in_trade == 0
    assert staged.bars_in_trade == 1
    assert staged.current_pnl_bps == pytest.approx(
        (3300.5 - 3300.2) / 3300.2 * 10_000.0
    )
    envelope = staged.build_closed_m1_path_evidence()
    assert envelope["entry_fill_ts"] == _ENTRY_FILL.isoformat()
    assert envelope["bars_in_trade"] == 1
    assert envelope["path_rows_sha256"]

    decision = _exit_decision(staged)
    staged.bind_unified_exit_decision(
        decision,
        expected_bundle_sha256=_BUNDLE_SHA256,
        exit_input_envelope=decision["exit_input_envelope"],
    )
    state.commit_complete_exit_bar(
        staged,
        expected_bundle_sha256=_BUNDLE_SHA256,
    )
    assert state.bars_in_trade == 1
    assert state.last_exit_decision == decision
    staged.closed_m1_path[0]["bid_close"] = 1.0
    assert state.closed_m1_path[0]["bid_close"] == 3300.5


def test_path_state_fails_closed_on_side_clock_and_unbound_commit() -> None:
    with pytest.raises(ValueError, match="contract mode is stale"):
        UnifiedExitPathState.open_unit_normalized_research(
            entry_ts=_ENTRY_FILL,
            side="long",
            entry_bid=3300.0,
            entry_ask=3300.2,
            v10_snapshot=_runtime_head("LONG"),
            replay_id="stale-mode",
            normalization_contract="unit_normalized_direction_exit_research_v1",
            model_identity_kind="bundle_sha256",
            model_identity_sha256=_BUNDLE_SHA256,
            input_normalization_sha256="6" * 64,
            contract_mode="xau_seq513_model_native_direction_stale",
        )

    with pytest.raises(ValueError, match="differs from model direction"):
        UnifiedExitPathState.open_unit_normalized_research(
            entry_ts=_ENTRY_FILL,
            side="short",
            entry_bid=3300.0,
            entry_ask=3300.2,
            v10_snapshot=_runtime_head("LONG"),
            replay_id="wrong-side",
            normalization_contract="unit_normalized_direction_exit_research_v1",
            model_identity_kind="bundle_sha256",
            model_identity_sha256=_BUNDLE_SHA256,
            input_normalization_sha256="6" * 64,
            contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        )

    state = _open("SHORT")
    with pytest.raises(ValueError, match="first closed M1 row"):
        state.clone_for_exit_decision().update_bar(
            **_closed_bar("2026-07-17T12:01:00Z")
        )

    staged = state.clone_for_exit_decision()
    staged.update_bar(**_closed_bar("2026-07-17T12:00:00Z"))
    assert staged.current_pnl_bps == pytest.approx(
        (3300.0 - 3300.7) / 3300.0 * 10_000.0
    )
    with pytest.raises(ValueError, match="lacks its model decision"):
        state.commit_complete_exit_bar(
            staged,
            expected_bundle_sha256=_BUNDLE_SHA256,
        )

    decision = _exit_decision(staged, q_values=(0.0, 1.0))
    decision["exit_path_envelope_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        staged.bind_unified_exit_decision(
            decision,
            expected_bundle_sha256=_BUNDLE_SHA256,
            exit_input_envelope=decision["exit_input_envelope"],
        )


def test_path_state_rolls_tail_and_preserves_all_time_duration() -> None:
    state = _open()
    for offset in range(UNIFIED_EXIT_MAX_PATH_BARS + 1):
        state.update_bar(
            **_closed_bar(
                (_ENTRY_FILL + pd.Timedelta(minutes=offset)).isoformat()
            )
        )
    envelope = state.build_closed_m1_path_evidence()
    assert envelope["bars_in_trade"] == UNIFIED_EXIT_MAX_PATH_BARS + 1
    assert envelope["retained_path_length"] == UNIFIED_EXIT_MAX_PATH_BARS
    assert envelope["path_rows"][0]["time"] == (
        _ENTRY_FILL + pd.Timedelta(minutes=1)
    ).isoformat()
    assert len(envelope["full_path_chain_sha256"]) == 64

    tensor = unified_exit_path_tensor(
        path_rows=envelope["path_rows"],
        bars_in_trade=envelope["bars_in_trade"],
        entry_bid=state.entry_bid,
        entry_ask=state.entry_ask,
    )
    assert tensor.shape == (
        UNIFIED_EXIT_MAX_PATH_BARS,
        UNIFIED_EXIT_PATH_FEATURE_DIM,
    )
    assert tensor[0, -2] == pytest.approx(np.log1p(2))
    assert tensor[-1, -2] == pytest.approx(
        np.log1p(UNIFIED_EXIT_MAX_PATH_BARS + 1)
    )
