import argparse
import subprocess
from pathlib import Path

from gx1.scripts.audit_entry_foundation_worktree_hygiene_v1 import (
    CATEGORY_ORDER,
    FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS,
    FOUNDATION_CLEANUP_REQUIRED_PATHS,
    classify_path,
    parse_porcelain_line,
    run,
)


REPO = Path("/home/andre2/src/GX1_ENGINE")
CONTROL = REPO / "scripts/entry_next_edge_control.sh"


def test_worktree_hygiene_classifies_foundation_and_review_paths() -> None:
    assert classify_path("docs/ENTRY_FOUNDATION_AUDIT_20260628.md") == "active_foundation_contract"
    assert classify_path("scripts/run_entry_foundation_seq146_smoke_train.sh") == "active_foundation_contract"
    assert classify_path("scripts/stage_entry_foundation_cleanup.sh") == "active_foundation_contract"
    assert classify_path("tests/test_entry_foundation_guardrails.py") == "active_foundation_contract"
    assert classify_path("tests/test_entry_foundation_state.py") == "active_foundation_contract"
    assert classify_path("tests/test_entry_handover_control.py") == "active_foundation_contract"
    assert classify_path("gx1/scripts/plan_entry_foundation_activation_v1.py") == "active_foundation_contract"
    assert classify_path("gx1/scripts/apply_entry_foundation_activation_v1.py") == "active_foundation_contract"
    assert classify_path("gx1/scripts/run_entry_foundation_activation_post_apply_v1.py") == "active_foundation_contract"
    assert classify_path("gx1/scripts/audit_entry_exit_handoff_readiness_v1.py") == "active_foundation_contract"
    assert classify_path("gx1/scripts/audit_entry_iql_replay_slices_v1.py") == "active_foundation_contract"
    assert classify_path("gx1/models/entry_v10/entry_v10_bundle.py") == "active_foundation_contract"
    assert classify_path("gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py") == "active_foundation_contract"
    assert classify_path("gx1/models/entry_v10/entry_v10_ctx_train_v3.py") == "active_foundation_contract"
    assert classify_path("docs/ENTRY_NEXT_EDGE_PLAN_20260627.md") == "legacy_tombstone_cleanup"
    assert classify_path("gx1/execution/oanda_client.py") == "unrelated_review"


def test_worktree_hygiene_parses_porcelain_line() -> None:
    row = parse_porcelain_line("?? gx1/scripts/verify_entry_training_readiness_v1.py")

    assert row["status"] == "??"
    assert row["path"] == "gx1/scripts/verify_entry_training_readiness_v1.py"
    assert row["category"] == "active_foundation_contract"


def test_worktree_hygiene_current_repo_report(tmp_path: Path) -> None:
    report = run(
        argparse.Namespace(
            out_dir=str(tmp_path),
            fail_on_dirty=False,
            quiet=True,
        )
    )

    assert report["decision"] in {"PASS_CLEAN_GIT", "BLOCKED_BY_DIRTY_GIT"}
    assert report["dirty_count"] >= 0
    assert set(report["categories"]) == set(CATEGORY_ORDER)
    assert "active_foundation_contract" in report["categories"]
    assert "review_before_stage_dirty_count" in report
    assert len(report["foundation_stage_paths"]) == report["foundation_cleanup_dirty_count"]
    assert len(report["review_hold_paths"]) == report["review_before_stage_dirty_count"]
    assert report["foundation_stage_summary"]["count"] == report["foundation_cleanup_dirty_count"]
    assert report["review_hold_summary"]["count"] == report["review_before_stage_dirty_count"]
    assert report["git_add_dry_run"]["returncode"] == 0
    assert report["git_add_dry_run"]["cached_unchanged"] is True
    assert report["stage_plan_safe"] is True
    assert all(report["stage_plan_checks"].values())
    assert report["stage_plan_diagnostics"]["stage_hold_overlap_count"] == 0
    assert report["stage_plan_diagnostics"]["git_add_dry_run_hold_overlap_count"] == 0
    assert report["foundation_cleanup_review_decision"] == "PASS"
    assert report["foundation_cleanup_required_review"]["missing_from_repo"] == []
    assert report["foundation_cleanup_required_review"]["dirty_missing_from_stage"] == []
    assert report["foundation_cleanup_required_review"]["ok_count"] == report["foundation_cleanup_required_review"]["required_path_count"]
    assert report["foundation_cleanup_critical_gate_review"]["missing_from_repo"] == []
    assert report["foundation_cleanup_critical_gate_review"]["dirty_missing_from_stage"] == []
    assert (
        report["foundation_cleanup_critical_gate_review"]["ok_count"]
        == report["foundation_cleanup_critical_gate_review"]["critical_gate_path_count"]
    )
    assert "scripts/entry_next_edge_control.sh" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/audit_entry_foundation_smoke_bundle_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/plan_entry_foundation_activation_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/apply_entry_foundation_activation_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/run_entry_foundation_activation_post_apply_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/verify_entry_candidate_readiness_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/verify_entry_foundation_guardrails_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/verify_entry_replay_readiness_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/materialize_entry_iql_distillation_contract_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/materialize_entry_iql_student_trade_log_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/materialize_entry_iql_replay_evidence_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/verify_entry_iql_replay_comparison_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/audit_entry_iql_replay_slices_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/audit_entry_exit_handoff_readiness_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/verify_entry_training_readiness_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "gx1/scripts/audit_entry_foundation_worktree_hygiene_v1.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_candidate_readiness.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_foundation_smoke_bundle_audit.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_foundation_guardrails.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_foundation_worktree_hygiene.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_iql_distillation_contract.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_iql_student_trade_log.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_iql_replay_comparison.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_iql_replay_evidence.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_iql_replay_slice_audit.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_exit_handoff_readiness.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_replay_readiness.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_training_readiness.py" in FOUNDATION_CLEANUP_CRITICAL_GATE_PATHS
    assert "tests/test_entry_foundation_guardrails.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "tests/test_entry_foundation_state.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "tests/test_entry_handover_control.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "gx1/scripts/plan_entry_foundation_activation_v1.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "gx1/scripts/apply_entry_foundation_activation_v1.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "gx1/scripts/run_entry_foundation_activation_post_apply_v1.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "gx1/scripts/audit_entry_exit_handoff_readiness_v1.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "gx1/scripts/audit_entry_iql_replay_slices_v1.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "gx1/models/entry_v10/entry_v10_bundle.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "gx1/models/entry_v10/entry_v10_ctx_train_v3.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "tests/test_entry_iql_replay_slice_audit.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    assert "tests/test_entry_exit_handoff_readiness.py" in FOUNDATION_CLEANUP_REQUIRED_PATHS
    if report["dirty_count"] == 0:
        assert report["foundation_cleanup_stage_ready"] is False
    else:
        assert report["foundation_cleanup_stage_ready"] is True
    assert report["foundation_cleanup_stage_command"][0:2] == ["git", "add"]
    assert report["foundation_cleanup_stage_preconditions"]["foundation_cleanup_review_decision"] == "PASS"
    post_stage = report["foundation_cleanup_post_stage_verification"]
    assert post_stage["decision"] in {"NOT_STAGED", "PASS_STAGED"}
    if post_stage["decision"] == "NOT_STAGED":
        assert post_stage["cached_count"] == 0
    resolution = report["clean_git_resolution"]
    assert resolution["decision"] in {
        "PASS_CLEAN_GIT",
        "READY_FOR_EXTERNAL_CLEAN_GIT_DECISION",
        "STAGE_FOUNDATION_CLEANUP_FIRST",
        "FIX_FOUNDATION_STAGE_PLAN_FIRST",
    }
    assert resolution["review_hold_count"] == report["review_before_stage_dirty_count"]
    assert resolution["review_hold_by_category"] == report["review_hold_summary"]["by_category"]
    assert "do not start smoke-train" in " ".join(resolution["forbidden_without_decision"])
    if post_stage["decision"] == "PASS_STAGED" and report["review_before_stage_dirty_count"]:
        assert resolution["decision"] == "READY_FOR_EXTERNAL_CLEAN_GIT_DECISION"
        assert resolution["requires_explicit_clean_git_decision"] is True
    elif post_stage["decision"] == "PASS_STAGED":
        assert post_stage["cached_count"] == len(report["foundation_stage_paths"])
        assert post_stage["stage_missing_from_cached_count"] == 0
        assert post_stage["cached_not_in_stage_count"] == 0
    assert post_stage["cached_hold_overlap_count"] == 0
    assert post_stage["cached_not_in_stage_count"] == 0
    assert Path(report["json_path"]).exists()
    assert Path(report["md_path"]).exists()
    assert Path(report["foundation_stage_paths_txt"]).exists()
    assert Path(report["review_hold_paths_txt"]).exists()
    assert Path(report["foundation_stage_status_tsv"]).exists()
    assert Path(report["review_hold_status_tsv"]).exists()
    assert Path(report["git_add_dry_run_txt"]).exists()


def test_stage_foundation_cleanup_dry_run_does_not_stage() -> None:
    before = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    result = subprocess.run(
        ["bash", str(CONTROL), "stage-foundation-cleanup", "--dry-run"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    if "Foundation cleanup stage-ready: true" in result.stdout:
        assert "git add --pathspec-from-file=" in result.stdout
    else:
        assert "Foundation cleanup stage-ready: false" in result.stdout
        assert "No foundation cleanup paths to stage." in result.stdout
    assert "Dry-run only; no git index changes made." in result.stdout
    assert cached.stdout == before.stdout


def test_stage_foundation_cleanup_apply_requires_vedtak_before_staging() -> None:
    before = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    result = subprocess.run(
        ["bash", str(CONTROL), "stage-foundation-cleanup", "--apply"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    cached = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert result.returncode == 2
    assert "--apply requires --vedtak" in result.stderr
    assert cached.stdout == before.stdout
