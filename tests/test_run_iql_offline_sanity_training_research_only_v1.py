from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_run_iql_offline_sanity_training_research_only_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_no_forbidden_actions_keeps_adapter_r6_production_and_live_closed() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(
        adapter=True,
        r6=True,
        iql_production=True,
        package=True,
        freeze=True,
        promo=True,
        live=True,
        optuna=True,
        broad_sweep=True,
    )
    assert blocked["status_v1"] == "FAIL"
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "IQL_PRODUCTION_FORBIDDEN" in blocked["failures_v1"]
    assert "OPTUNA_FORBIDDEN" in blocked["failures_v1"]


def test_contract_reproduction_requires_expected_cohorts_and_field_counts() -> None:
    payload = {
        "state_allowlist_field_count_v1": 9,
        "state_denylist_field_count_v1": 22,
        "baseline_140_94_v1": {
            "selected_rows_v1": 140,
            "bad_count_audit_only_v1": 140,
            "tail_count_audit_only_v1": 94,
            "safety_status_v1": "CLEAN",
        },
        "safe_core_89_v1": {
            "selected_rows_v1": 89,
            "bad_count_audit_only_v1": 86,
            "tail_count_audit_only_v1": 55,
            "safety_status_v1": "CLEAN",
        },
        "source_safety_shielded_78_v1": {
            "selected_rows_v1": 78,
            "bad_count_audit_only_v1": 75,
            "tail_count_audit_only_v1": 55,
            "original_140_retained_v1": 75,
            "safety_status_v1": "CLEAN",
            "unsafe_row_blocked_v1": True,
        },
    }
    assert gate.validate_contract_reproducibility(payload)
    payload["source_safety_shielded_78_v1"]["selected_rows_v1"] = 79
    with pytest.raises(RuntimeError, match="IQL_SANITY_CONTRACT_REPRODUCTION_FAILED"):
        gate.validate_contract_reproducibility(payload)


def test_state_matrix_blocks_denied_tokens_and_fields() -> None:
    audit_rows = [
        {
            "raw_field_name_v1": "candidate_score_v1",
            "model_state_column_v1": "candidate_score_z_train_only_v1",
            "denied_field_v1": False,
        },
        {
            "raw_field_name_v1": "bad_label_v1",
            "model_state_column_v1": "",
            "denied_field_v1": True,
        },
    ]
    assert gate.validate_state_matrix(audit_rows, ["candidate_score_z_train_only_v1", "signal_r5_tail_score_v1"])
    with pytest.raises(RuntimeError, match="DENIED_STATE_TOKEN_IN_MATRIX"):
        gate.validate_state_matrix(audit_rows, ["bad_label_v1"])
    audit_rows[1]["model_state_column_v1"] = "bad_label_v1"
    with pytest.raises(RuntimeError, match="DENIED_FIELD_MAPPED_TO_STATE"):
        gate.validate_state_matrix(audit_rows, ["candidate_score_z_train_only_v1"])


def test_go_no_go_blocks_production_paths_and_marks_contextual_not_sequential() -> None:
    payload = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "research_only_contextual_iql_sanity_ran_v1": True,
        "sequential_iql_ready_v1": False,
        "deeper_research_allowed_next_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
    }
    assert gate.validate_go_no_go(payload)
    with pytest.raises(RuntimeError, match="FORBIDDEN_PATH_OPENED"):
        gate.validate_go_no_go(dict(payload, adapter_build_allowed_v1=True))
    with pytest.raises(RuntimeError, match="CONTEXTUAL_STATUS_CANNOT_MARK_SEQUENTIAL_READY"):
        gate.validate_go_no_go(dict(payload, sequential_iql_ready_v1=True))


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("GO_LIVE", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_materializer_writes_required_outputs_and_runs_contextual_only(tmp_path: Path) -> None:
    artifact_root = tmp_path / "RUN_IQL_OFFLINE_SANITY_TRAINING_RESEARCH_ONLY_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert summary["mode_v1"] == gate.SANITY_MODE
    assert summary["chosen_safety_shield_v1"] == gate.SAFETY_COHORT
    assert summary["training_status_v1"] == "CONTEXTUAL_ONE_STEP_SANITY_TRAINING_COMPLETED"
    assert summary["policy_selected_rows_v1"] > 0
    assert summary["policy_selected_rows_v1"] < 78
    assert summary["policy_safety_status_v1"] == "CLEAN"
    assert summary["no_shortcut_audit_status_v1"] == "PASS"
    assert summary["adapter_r6_iql_production_live_remain_blocked_v1"] is True
    assert summary["adapter_built_v1"] is False
    assert summary["r6_run_v1"] is False
    assert summary["iql_production_opened_v1"] is False

    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    transition = json.loads((artifact_root / "iql_offline_sanity_transition_or_contextual_audit_v1.json").read_text())
    assert transition["status_v1"] == gate.SANITY_MODE
    assert transition["true_sequential_iql_available_v1"] is False
    assert transition["no_fake_transitions_created_v1"] is True

    no_shortcut = json.loads((artifact_root / "iql_offline_sanity_no_shortcut_audit_v1.json").read_text())
    assert no_shortcut["status_v1"] == "PASS"
    assert no_shortcut["checks_v1"]["historical_v2_blueprint_absent_v1"] is True

    baseline = json.loads((artifact_root / "iql_offline_sanity_baseline_policy_comparison_v1.json").read_text())
    policies = {row["policy_name_v1"]: row for row in baseline["rows_v1"]}
    assert policies["IQL_CONTEXTUAL_ONE_STEP_POLICY"]["selected_rows_v1"] == summary["policy_selected_rows_v1"]
    assert policies["IQL_CONTEXTUAL_ONE_STEP_POLICY"]["safety_status_v1"] == "CLEAN"
    assert policies["IQL_CONTEXTUAL_ONE_STEP_POLICY"]["selected_rows_v1"] != policies[
        "ALWAYS_TAKE_WITHIN_78_SHIELD"
    ]["selected_rows_v1"]

    go = json.loads((artifact_root / "run_iql_offline_sanity_training_research_only_go_no_go_v1.json").read_text())
    assert go["status_v1"] == gate.FINAL_STATUS
    assert go["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert go["iql_production_allowed_v1"] is False
    assert go["adapter_build_allowed_v1"] is False
    assert go["r6_allowed_v1"] is False
