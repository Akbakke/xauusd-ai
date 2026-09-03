from __future__ import annotations

from gx1.scripts.verify_candidate_checkpoint_policy_v1 import run_policy_proof


def test_policy_proof_covers_max_epoch_and_early_stop_paths() -> None:
    report = run_policy_proof()
    assert report["decision"] == "PASS"
    assert report["terminal_epoch"] == 30
    assert report["terminal_reason"] == "max_epochs_thirty_after_full_validation"
    assert report["early_stop"] is False
    assert [row["epoch"] for row in report["top_k"]] == [30]
    assert report["early_stop_proof"]["terminal_epoch"] == 6
    assert report["early_stop_proof"]["terminal_reason"] == "early_stop_after_five_non_improvements"
    assert [row["epoch"] for row in report["early_stop_proof"]["top_k"]] == [1]
