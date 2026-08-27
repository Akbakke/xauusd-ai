from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gx1.contracts.current_audited_dataset_evidence_v1 import (
    CURRENT_AUDITED_DATASET_BLOCKER,
    CURRENT_AUDITED_DATASET_STATUS,
    require_blocked_launch_state_with_current_audited_dataset,
    require_current_audited_dataset_evidence,
)


REPO = Path(__file__).resolve().parents[1]
LAUNCH_STATE = REPO / "PROJECT_STATE_xau_direction_launch.json"


def _state() -> dict[str, object]:
    return json.loads(LAUNCH_STATE.read_text(encoding="utf-8"))


def test_current_v46_review_is_hash_bound_but_not_admitted() -> None:
    summary = require_blocked_launch_state_with_current_audited_dataset(_state())

    assert summary["status"] == CURRENT_AUDITED_DATASET_STATUS
    assert summary["blocker"] == CURRENT_AUDITED_DATASET_BLOCKER
    assert summary["dataset_run_id"] == "V46_20260825T170935Z"
    assert summary["report_count"] == 12


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (
            ("current_audited_dataset_evidence", "reports", "feature_audit", "sha256"),
            "0" * 64,
            "FEATURE_AUDIT_HASH_MISMATCH",
        ),
        (
            ("current_audited_dataset_evidence", "reports", "feature_audit", "path"),
            "/tmp/ENTRY_FEATURE_FOUNDATION_AUDIT_latest.json",
            "FEATURE_AUDIT_PATH_INVALID",
        ),
        (
            ("current_audited_dataset_evidence", "activation_allowed"),
            True,
            "EVIDENCE_STATUS_INVALID",
        ),
    ],
)
def test_current_review_rejects_mutation_and_mutable_discovery(
    path: tuple[str, ...], value: object, error: str
) -> None:
    evidence = copy.deepcopy(_state()["current_audited_dataset_evidence"])
    target: dict[str, object] = evidence
    for key in path[1:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value

    with pytest.raises(RuntimeError, match=error):
        require_current_audited_dataset_evidence(evidence)


def test_current_review_cannot_turn_the_launch_state_open() -> None:
    state = _state()
    state["dataset_admission_stage"] = "ADMITTED"

    with pytest.raises(RuntimeError, match="LAUNCH_STATE_NOT_FAIL_CLOSED"):
        require_blocked_launch_state_with_current_audited_dataset(state)
