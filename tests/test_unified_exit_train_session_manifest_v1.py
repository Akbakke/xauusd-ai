from __future__ import annotations

import pytest

from gx1.contracts.unified_exit_train_session_manifest_v1 import (
    canonical_sha256,
    require_train_session_manifest,
)


def test_preselection_manifest_cannot_claim_resume_or_epoch_eligibility() -> None:
    value = {
        "schema_version": "gx1_unified_exit_train_session_manifest_v1",
        "decision": "PASS_RESUME_PROOF_ELIGIBLE",
        "phase": "resume_proof",
        "entry_pairs_per_epoch": 16384,
        "transition_budget_per_epoch": 65536,
        "test_data_used": False,
    }
    value["manifest_sha256"] = canonical_sha256(value)
    with pytest.raises(RuntimeError):
        require_train_session_manifest(value, expected_phase="resume_proof")
