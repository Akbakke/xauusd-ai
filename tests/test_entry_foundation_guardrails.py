import argparse
import sys
from pathlib import Path

import pytest

from gx1.scripts.verify_entry_foundation_guardrails_v1 import CommandCase, _run_case, run


def test_command_case_rejects_forbidden_text() -> None:
    with pytest.raises(RuntimeError, match="forbidden text found"):
        _run_case(
            CommandCase(
                name="sample_forbidden_output",
                cmd=[sys.executable, "-c", "print('active foundation plus legacy body')"],
                expected_returncode=0,
                required_text="active foundation",
                forbidden_texts=("legacy body",),
            )
        )


def test_foundation_guardrails_current_repo_lock_control_and_handover(tmp_path: Path) -> None:
    report = run(argparse.Namespace(out_dir=str(tmp_path), quiet=True))

    assert report["decision"] == "PASS"
    assert report["promotion_shadow_live_allowed"] is False
    assert Path(report["out"]).exists()

    case_names = {case["name"] for case in report["cases"]}
    assert "control_readiness_report_active_foundation" in case_names
    assert "handover_points_at_foundation" in case_names
    assert "generic_train_blocked" in case_names

    handover_case = next(case for case in report["cases"] if case["name"] == "handover_points_at_foundation")
    assert handover_case["observed_returncode"] == 0
    assert handover_case["required_text"] == "active Entry foundation seq146"
    assert "OPEN-MORE WAVE ARMED" in handover_case["forbidden_texts"]

    checks = {row["name"]: row["ok"] for row in report["source_checks"]}
    assert checks["control_verify_dispatches_to_foundation_state"] is True
    assert checks["control_blocks_shadow_live_train_legacy_paths"] is True
    assert checks["control_exposes_foundation_adoption_candidate_report"] is True
    assert checks["control_exposes_foundation_activation_plan_report"] is True
    assert checks["control_exposes_vedtak_gated_foundation_activation_apply"] is True
    assert checks["control_exposes_vedtak_gated_foundation_activation_post_apply"] is True
    assert checks["handover_legacy_requires_explicit_env_token"] is True
    assert checks["handover_default_exits_before_legacy_body"] is True
    assert checks["handover_default_announces_active_foundation"] is True

    policy_checks = {row["name"]: row["ok"] for row in report["readiness_policy_checks"]}
    assert policy_checks["readiness_policy_snapshot_json_parseable"] is True
    assert policy_checks["readiness_policy_snapshot_report_only"] is True
    assert policy_checks["readiness_policy_command_set_exact"] is True
    assert policy_checks["readiness_policy_command_schema_complete"] is True
    assert policy_checks["readiness_policy_allowed_now_has_no_vedtak_placeholders"] is True
    assert policy_checks["readiness_policy_adoption_candidate_does_not_activate_without_vedtak"] is True
    assert policy_checks["readiness_policy_safe_now_verify"] is True
    assert policy_checks["readiness_policy_safe_now_foundation_activation_plan"] is True
    assert policy_checks["readiness_policy_safe_now_foundation_activation_apply_dry_run"] is True
    assert policy_checks["readiness_policy_safe_now_foundation_activation_post_apply_dry_run"] is True
    assert policy_checks["readiness_policy_safe_now_candidate_readiness_report"] is True
    assert policy_checks["readiness_policy_safe_now_iql_slice_audit"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_materialize"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_handoff"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_reconstruction_audit"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_state_reward_contract"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_split_leakage_audit"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_model_dataset_readiness"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_transformer_architecture_readiness"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_transformer_training_plan_readiness"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_transformer_trainer_wrapper_readiness"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_transformer_pretrain_manifest"] is True
    assert policy_checks["readiness_policy_safe_now_entry_exit_model_dataset_slice_robustness"] is True
    assert policy_checks["readiness_policy_blocks_foundation_activation_apply"] is True
    assert policy_checks["readiness_policy_blocks_foundation_activation_post_apply"] is True
    assert policy_checks["readiness_policy_blocks_smoke_train"] is True
    assert policy_checks["readiness_policy_blocks_candidate_train"] is True
    assert policy_checks["readiness_policy_blocks_selective_edge"] is True
    assert policy_checks["readiness_policy_blocks_replay_evidence"] is True
    assert policy_checks["readiness_policy_blocks_iql_distill"] is True
    assert policy_checks["readiness_policy_blocks_iql_student_trade_log"] is True
    assert policy_checks["readiness_policy_blocks_iql_replay_evidence"] is True
    assert policy_checks["readiness_policy_blocks_iql_compare"] is True
    assert policy_checks["readiness_policy_blocks_entry_exit_transformer_train"] is True
    assert policy_checks["readiness_policy_blocks_preview_shadow"] is True
    assert policy_checks["readiness_policy_blocks_start_shadow"] is True
    assert policy_checks["readiness_policy_blocks_live"] is True
    assert policy_checks["readiness_policy_candidate_train_declares_trainer"] is True
    assert policy_checks["readiness_policy_iql_distill_declares_iql_side_effect"] is True
    assert policy_checks["readiness_policy_entry_exit_transformer_train_declares_trainer"] is True
    assert policy_checks["readiness_policy_shadow_live_declares_live_touch"] is True
