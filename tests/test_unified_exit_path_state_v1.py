from __future__ import annotations

import json
import math
from dataclasses import fields

import pandas as pd
import pytest

from gx1.models.entry_v10.direction_decision_contract import (
    CLOSED_M1_PATH_SCHEMA_VERSION,
    canonical_unified_evidence_sha256,
)
from gx1.replay.unified_exit_path_state_v1 import (
    UnifiedExitPathState,
    first_full_closed_m1_bar_ts,
)
from tests.model_native_offline_rl_support import (
    TEST_DIRECTION_CALIBRATION,
    TEST_PATH_CALIBRATION,
    offline_rl_prediction_row,
    runtime_head_prediction_columns,
)


_ENTRY_FILL = pd.Timestamp("2026-07-17T12:00:00Z")
_BUNDLE_SHA256 = "b" * 64


def _runtime_head(direction: str = "LONG") -> dict[str, object]:
    logits = [3.0, 1.0, 0.0] if direction == "LONG" else [1.0, 3.0, 0.0]
    row = {
        "time": _ENTRY_FILL - pd.Timedelta(minutes=5),
        "direction_logits": logits,
        "position_size_logit": -0.1,
        "timing_pred": [0.0] * 12,
        **offline_rl_prediction_row(),
    }
    columns = runtime_head_prediction_columns(
        pd.DataFrame([row]),
        {
            "direction_calibration": dict(TEST_DIRECTION_CALIBRATION),
            "path_calibration": dict(TEST_PATH_CALIBRATION),
        },
    )
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
    logits: tuple[float, float] = (1.0, 0.0),
) -> dict[str, object]:
    peak = max(logits)
    exponentials = [math.exp(value - peak) for value in logits]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    envelope = state.build_closed_m1_path_evidence()
    action_index = 0 if logits[0] > logits[1] else 1
    decision: dict[str, object] = {
        "exit_action_logits": list(logits),
        "exit_action_probs": probabilities,
        "exit_action_index": action_index,
        "action": ("HOLD", "EXIT_NOW")[action_index],
        "decision_source": "unified_model",
        "bundle_sha256": _BUNDLE_SHA256,
        "entry_snapshot_sha256": canonical_unified_evidence_sha256(
            state.entry_snapshot
        ),
        "exit_path_envelope_sha256": canonical_unified_evidence_sha256(
            envelope
        ),
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
        "normalization_contract",
        "bars_in_trade",
        "last_processed_m1_ts",
        "current_bid",
        "current_ask",
        "current_pnl_bps",
        "closed_m1_path",
        "last_exit_decision",
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
    with pytest.raises(ValueError, match="differs from model direction"):
        UnifiedExitPathState.open_unit_normalized_research(
            entry_ts=_ENTRY_FILL,
            side="short",
            entry_bid=3300.0,
            entry_ask=3300.2,
            v10_snapshot=_runtime_head("LONG"),
            replay_id="wrong-side",
            normalization_contract="unit_normalized_direction_exit_research_v1",
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

    decision = _exit_decision(staged, logits=(0.0, 1.0))
    decision["exit_path_envelope_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="content hash mismatch"):
        staged.bind_unified_exit_decision(
            decision,
            expected_bundle_sha256=_BUNDLE_SHA256,
        )
