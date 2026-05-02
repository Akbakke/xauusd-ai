from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_build_iql_event_ordered_research_transition_dataset_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/BUILD_IQL_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_V1_20260428T000000Z_LOCK")]
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
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_state_column_guard_blocks_denied_tokens() -> None:
    assert gate.validate_state_columns(["candidate_score_z_train_only_v1", "signal_tail_repair_v1"])
    with pytest.raises(RuntimeError, match="DENIED_STATE_TOKEN_IN_MATRIX"):
        gate.validate_state_columns(["candidate_score_z_train_only_v1", "bad_label_v1"])
    with pytest.raises(RuntimeError, match="DENIED_STATE_TOKEN_IN_MATRIX"):
        gate.validate_state_columns(["historical_v2_blueprint_v1"])


def test_go_no_go_blocks_production_adapter_r6_and_lifecycle_iql() -> None:
    payload = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "event_ordered_research_training_allowed_next_v1": True,
        "full_lifecycle_sequential_iql_ready_v1": False,
        "iql_training_run_in_this_gate_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_production_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
    }
    assert gate.validate_go_no_go(payload)
    with pytest.raises(RuntimeError, match="FORBIDDEN_PATH_OPENED"):
        gate.validate_go_no_go(dict(payload, r6_allowed_v1=True))
    with pytest.raises(RuntimeError, match="FULL_LIFECYCLE_IQL_OPENED"):
        gate.validate_go_no_go(dict(payload, full_lifecycle_sequential_iql_ready_v1=True))


def test_no_fake_transition_audit_guard() -> None:
    audit = {
        "status_v1": "PASS",
        "critical_failures_v1": [],
        "checks_v1": {
            "no_synthetic_next_state_v1": True,
            "no_random_next_state_v1": True,
            "no_cross_run_next_state_v1": True,
            "no_transition_across_episode_boundary_v1": True,
            "row_identity_not_state_v1": True,
            "reward_not_state_v1": True,
            "future_label_not_state_v1": True,
            "historical_v2_blueprint_not_used_v1": True,
            "transformer_fields_absent_v1": True,
        },
    }
    assert gate.validate_no_fake_transition_audit(audit)
    broken = json.loads(json.dumps(audit))
    broken["checks_v1"]["no_synthetic_next_state_v1"] = False
    with pytest.raises(RuntimeError, match="NO_FAKE_TRANSITION_CHECK_FAILED"):
        gate.validate_no_fake_transition_audit(broken)


def test_materializer_writes_event_ordered_dataset_and_required_outputs(tmp_path: Path) -> None:
    artifact_root = tmp_path / "BUILD_IQL_EVENT_ORDERED_RESEARCH_TRANSITION_DATASET_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert summary["dataset_kind_v1"] == gate.DATASET_KIND
    assert summary["rows_v1"] == 1914
    assert summary["episodes_v1"] == 58
    assert summary["nonterminal_transitions_v1"] == 1856
    assert summary["terminal_rows_v1"] == 58
    assert summary["cross_run_transitions_v1"] == 0
    assert summary["state_next_state_allowlist_only_v1"] is True
    assert summary["take_trade_count_v1"] == 78
    assert summary["skip_count_v1"] == 1836
    assert summary["no_fake_transition_audit_status_v1"] == "PASS"
    assert summary["research_only_event_ordered_not_full_lifecycle_iql_v1"] is True
    assert summary["adapter_r6_iql_production_live_remain_blocked_v1"] is True
    assert summary["iql_training_run_v1"] is False

    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    manifest = json.loads((artifact_root / "iql_event_ordered_transition_input_manifest_v1.json").read_text())
    assert manifest["no_implicit_latest_glob_selection_v1"] is True
    assert manifest["previous_artifacts_mutated_v1"] is False
    assert manifest["iql_training_run_v1"] is False
    assert manifest["adapter_built_v1"] is False
    assert manifest["r6_run_v1"] is False

    dataset = json.loads((artifact_root / "iql_event_ordered_transition_dataset_v1.json").read_text())
    assert dataset["row_count_v1"] == 1914
    rows = dataset["rows_v1"]
    assert sum(1 for row in rows if row["done_v1"]) == 58
    assert all(not row["cross_run_transition_v1"] for row in rows)
    assert all(not row["state_contains_denied_fields_v1"] for row in rows)
    assert all(not row["next_state_contains_denied_fields_v1"] for row in rows)
    assert all(row["dataset_kind_v1"] == gate.DATASET_KIND for row in rows)
    assert {row["action_observed_or_inferred_v1"] for row in rows} == {
        "INFERRED_RESEARCH_ONLY_NOT_PRODUCTION_LOGGED_ACTION"
    }

    state_matrix = json.loads((artifact_root / "iql_event_ordered_state_matrix_v1.json").read_text())
    next_state_matrix = json.loads((artifact_root / "iql_event_ordered_next_state_matrix_v1.json").read_text())
    assert state_matrix["row_count_v1"] == 1914
    assert next_state_matrix["row_count_v1"] == 1914
    assert all("label" not in column.lower() for column in state_matrix["feature_columns_v1"])
    assert all("reward" not in column.lower() for column in state_matrix["feature_columns_v1"])

    no_fake = json.loads((artifact_root / "iql_event_ordered_no_fake_transition_audit_v1.json").read_text())
    assert no_fake["status_v1"] == "PASS"
    assert no_fake["checks_v1"]["no_cross_run_next_state_v1"] is True
    assert no_fake["checks_v1"]["historical_v2_blueprint_not_used_v1"] is True

    go = json.loads((artifact_root / "build_iql_event_ordered_research_transition_dataset_go_no_go_v1.json").read_text())
    assert go["status_v1"] == gate.FINAL_STATUS
    assert go["event_ordered_research_training_allowed_next_v1"] is True
    assert go["full_lifecycle_sequential_iql_ready_v1"] is False
    assert go["adapter_build_allowed_v1"] is False
    assert go["r6_allowed_v1"] is False
    assert go["iql_production_allowed_v1"] is False
