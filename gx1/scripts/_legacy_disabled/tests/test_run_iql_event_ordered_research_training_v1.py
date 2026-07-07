from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_run_iql_event_ordered_research_training_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/RUN_IQL_EVENT_ORDERED_RESEARCH_TRAINING_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("GO_LIVE", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "BUILD_ADAPTER_NOW")


def test_state_column_guard_blocks_denied_tokens() -> None:
    assert gate.validate_state_columns(["candidate_score_z_train_only_v1", "signal_tail_repair_v1"])
    with pytest.raises(RuntimeError, match="DENIED_STATE_TOKEN_IN_MATRIX"):
        gate.validate_state_columns(["candidate_score_z_train_only_v1", "reward_t_v1"])
    with pytest.raises(RuntimeError, match="DENIED_STATE_TOKEN_IN_MATRIX"):
        gate.validate_state_columns(["historical_v2_blueprint_v1"])
    with pytest.raises(RuntimeError, match="DENIED_STATE_TOKEN_IN_MATRIX"):
        gate.validate_state_columns(["row_id_v1"])


def test_go_no_go_blocks_production_adapter_r6_and_promotion() -> None:
    payload = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "event_ordered_deeper_research_allowed_next_v1": True,
        "full_lifecycle_sequential_iql_ready_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
    }
    assert gate.validate_go_no_go(payload)
    with pytest.raises(RuntimeError, match="FORBIDDEN_PATH_OPENED"):
        gate.validate_go_no_go(dict(payload, adapter_build_allowed_v1=True))
    with pytest.raises(RuntimeError, match="FORBIDDEN_PATH_OPENED"):
        gate.validate_go_no_go(dict(payload, policy_promotion_allowed_v1=True))
    with pytest.raises(RuntimeError, match="FULL_LIFECYCLE_IQL_OPENED"):
        gate.validate_go_no_go(dict(payload, full_lifecycle_sequential_iql_ready_v1=True))


def test_no_shortcut_guard_blocks_critical_failures() -> None:
    payload = {"status_v1": "PASS", "critical_failures_v1": []}
    assert gate.validate_no_shortcut(payload)
    with pytest.raises(RuntimeError, match="IQL_EVENT_ORDERED_TRAINING_NO_SHORTCUT_FAILED"):
        gate.validate_no_shortcut({"status_v1": "FAIL", "critical_failures_v1": ["reward leaked"]})


def test_reproducibility_guard_locks_transition_dataset_counts() -> None:
    payload = {
        "input_event_dataset_status_v1": "IQL_EVENT_ORDERED_TRANSITION_DATASET_READY_FOR_RESEARCH_TRAINING",
        "input_event_dataset_next_action_v1": gate.ACTION,
        "input_no_fake_transition_status_v1": "PASS",
        "rows_v1": 1914,
        "episodes_v1": 58,
        "nonterminal_transitions_v1": 1856,
        "terminal_rows_v1": 58,
        "cross_run_transitions_v1": 0,
        "state_feature_count_v1": 11,
        "take_trade_count_v1": 78,
        "skip_count_v1": 1836,
        "reward_sum_v1": 89.0,
        "research_only_event_ordered_v1": True,
    }
    assert gate.validate_reproducibility(payload)
    with pytest.raises(RuntimeError, match="IQL_EVENT_ORDERED_TRAINING_REPRODUCTION_FAILED"):
        gate.validate_reproducibility(dict(payload, cross_run_transitions_v1=1))


def test_materializer_writes_training_outputs_and_expected_metrics(tmp_path: Path) -> None:
    artifact_root = tmp_path / "RUN_IQL_EVENT_ORDERED_RESEARCH_TRAINING_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert summary["dataset_kind_v1"] == gate.DATASET_KIND
    assert summary["model_id_v1"] == gate.MODEL_ID
    assert summary["rows_v1"] == 1914
    assert summary["episodes_v1"] == 58
    assert summary["state_feature_count_v1"] == 11
    assert summary["policy_selected_rows_v1"] == 71
    assert summary["policy_reward_sum_v1"] == 91.75
    assert summary["policy_bad_tail_audit_only_v1"] == [70, 55]
    assert summary["policy_precision_audit_only_v1"] == pytest.approx(0.9859154929577465)
    assert summary["policy_safety_status_v1"] == "CLEAN"
    assert summary["contextual_reward_delta_v1"] == pytest.approx(1.25)
    assert summary["no_shortcut_audit_status_v1"] == "PASS"
    assert summary["research_only_event_ordered_not_full_lifecycle_iql_v1"] is True
    assert summary["adapter_r6_iql_production_live_remain_blocked_v1"] is True
    assert summary["adapter_built_v1"] is False
    assert summary["r6_run_v1"] is False
    assert summary["iql_production_opened_v1"] is False
    assert summary["package_built_v1"] is False
    assert summary["freeze_promo_live_run_v1"] is False

    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    manifest = json.loads((artifact_root / "iql_event_ordered_training_input_manifest_v1.json").read_text())
    assert manifest["no_implicit_latest_glob_selection_v1"] is True
    assert manifest["previous_artifacts_mutated_v1"] is False
    assert manifest["research_only_event_ordered_training_v1"] is True
    assert manifest["adapter_built_v1"] is False
    assert manifest["r6_run_v1"] is False
    assert manifest["iql_production_opened_v1"] is False

    repro = json.loads((artifact_root / "iql_event_ordered_training_reproducibility_audit_v1.json").read_text())
    assert repro["rows_v1"] == 1914
    assert repro["episodes_v1"] == 58
    assert repro["nonterminal_transitions_v1"] == 1856
    assert repro["terminal_rows_v1"] == 58
    assert repro["cross_run_transitions_v1"] == 0
    assert repro["take_trade_count_v1"] == 78
    assert repro["skip_count_v1"] == 1836
    assert repro["reward_sum_v1"] == pytest.approx(89.0)
    assert repro["adapter_r6_iql_production_live_remain_blocked_v1"] is True

    metrics = json.loads((artifact_root / "iql_event_ordered_training_metrics_v1.json").read_text())[
        "rows_v1"
    ]
    by_split = {row["split_id_v1"]: row for row in metrics}
    assert by_split["train"]["selected_take_rows_v1"] == 32
    assert by_split["train"]["total_reward_v1"] == pytest.approx(45.5)
    assert by_split["validation"]["selected_take_rows_v1"] == 29
    assert by_split["validation"]["total_reward_v1"] == pytest.approx(30.75)
    assert by_split["test"]["selected_take_rows_v1"] == 10
    assert by_split["test"]["total_reward_v1"] == pytest.approx(15.5)
    assert by_split["all"]["selected_take_rows_v1"] == 71
    assert by_split["all"]["bad_count_audit_only_v1"] == 70
    assert by_split["all"]["tail_count_audit_only_v1"] == 55
    assert by_split["all"]["safety_status_v1"] == "CLEAN"
    assert by_split["all"]["unsafe_boundary_row_selected_v1"] is False

    split_rows = json.loads((artifact_root / "iql_event_ordered_training_split_audit_v1.json").read_text())[
        "rows_v1"
    ]
    split_by_id = {row["split_id_v1"]: row for row in split_rows}
    assert split_by_id["train"]["episodes_v1"] == 28
    assert split_by_id["train"]["transitions_v1"] == 930
    assert split_by_id["train"]["take_trade_count_v1"] == 39
    assert split_by_id["validation"]["episodes_v1"] == 19
    assert split_by_id["validation"]["transitions_v1"] == 742
    assert split_by_id["validation"]["take_trade_count_v1"] == 29
    assert split_by_id["test"]["episodes_v1"] == 11
    assert split_by_id["test"]["transitions_v1"] == 242
    assert split_by_id["test"]["take_trade_count_v1"] == 10

    baselines = json.loads(
        (artifact_root / "iql_event_ordered_training_baseline_comparison_v1.json").read_text()
    )["rows_v1"]
    baseline_by_name = {row["policy_name_v1"]: row for row in baselines}
    assert baseline_by_name["CONTEXTUAL_IQL_SANITY_POLICY_FROM_PREVIOUS_GATE"]["total_reward_v1"] == pytest.approx(
        90.5
    )
    assert baseline_by_name["SOURCE_SAFETY_SHIELDED_78_POLICY"]["total_reward_v1"] == pytest.approx(89.0)
    assert baseline_by_name["EVENT_ORDERED_LINEAR_IQL_POLICY"]["total_reward_v1"] == pytest.approx(91.75)
    assert baseline_by_name["EVENT_ORDERED_LINEAR_IQL_POLICY"]["safety_status_v1"] == "CLEAN"

    no_shortcut = json.loads((artifact_root / "iql_event_ordered_training_no_shortcut_audit_v1.json").read_text())
    assert no_shortcut["status_v1"] == "PASS"
    assert no_shortcut["checks_v1"]["denied_fields_absent_from_state_and_next_state_v1"] is True
    assert no_shortcut["checks_v1"]["no_cross_run_transitions_v1"] is True
    assert no_shortcut["checks_v1"]["no_fake_next_state_v1"] is True
    assert no_shortcut["checks_v1"]["no_optuna_or_broad_sweep_v1"] is True

    go = json.loads((artifact_root / "run_iql_event_ordered_research_training_go_no_go_v1.json").read_text())
    assert go["status_v1"] == gate.FINAL_STATUS
    assert go["event_ordered_deeper_research_allowed_next_v1"] is True
    assert go["full_lifecycle_sequential_iql_ready_v1"] is False
    assert go["adapter_build_allowed_v1"] is False
    assert go["r6_allowed_v1"] is False
    assert go["iql_production_allowed_v1"] is False
    assert go["package_freeze_promo_live_allowed_v1"] is False
    assert go["policy_promotion_allowed_v1"] is False
