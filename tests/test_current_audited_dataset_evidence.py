"""Current launch must not substitute a historical audited dataset."""
import json
from pathlib import Path

import pytest

from gx1.contracts.current_audited_dataset_evidence_v1 import (
    require_blocked_launch_state_with_current_audited_dataset,
    require_current_audited_dataset_evidence,
)

REPO = Path(__file__).resolve().parents[1]


def _state():
    return json.loads((REPO / "PROJECT_STATE_xau_direction_launch.json").read_text())


def test_no_historical_dataset_is_admitted_or_implicitly_selected():
    state = _state()
    assert state["decision"] == "BLOCK"
    assert state["pretraining_review_hold"]["activation_authority"] is False
    assert not {"current_source_technical_recipe", "active_candidate_training_session",
                "current_audited_dataset_evidence"} & state.keys()
    with pytest.raises(RuntimeError, match="EVIDENCE_MISSING"):
        require_blocked_launch_state_with_current_audited_dataset(state)


@pytest.mark.parametrize("field", [
    "dataset_event_id", "dataset_admission_stage", "accepted_dataset_dir",
    "accepted_dataset_terminal_evidence", "accepted_bundle_dir",
    "bundle_metadata_sha256", "current_smoke_launch_evidence", "accepted_via_vedtak",
])
def test_missing_current_evidence_cannot_open_launch(field):
    state = _state()
    state[field] = "UNAUTHORIZED"
    with pytest.raises(RuntimeError, match="LAUNCH_STATE_NOT_FAIL_CLOSED"):
        require_blocked_launch_state_with_current_audited_dataset(state)


@pytest.mark.parametrize("value", [None, {}, {"activation_allowed": True}])
def test_missing_or_partial_audited_evidence_is_rejected(value):
    with pytest.raises(RuntimeError, match="EVIDENCE_MISSING|EVIDENCE_SCHEMA_INVALID"):
        require_current_audited_dataset_evidence(value)
