from __future__ import annotations

from gx1.scripts.verify_candidate_checkpoint_policy_v1 import run_policy_proof


def test_policy_proof_requires_one_complete_epoch_and_one_checkpoint() -> None:
    report = run_policy_proof()
    assert report["decision"] == "PASS"
    assert report["terminal_epoch"] == 1
    assert report["terminal_reason"] == "max_epochs_one_after_full_validation"
    assert report["early_stop"] is False
    assert [row["epoch"] for row in report["top_k"]] == [1]
