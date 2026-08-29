from __future__ import annotations

from gx1.scripts.verify_candidate_checkpoint_policy_v1 import run_policy_proof


def test_policy_proof_preserves_patience_across_resume_and_top_k() -> None:
    report = run_policy_proof()
    assert report["decision"] == "PASS"
    assert report["early_stop_epoch"] == 7
    assert report["early_stop_epoch_after_resume"] == 7
    assert [row["epoch"] for row in report["top_k"]] == [2, 3, 4]
