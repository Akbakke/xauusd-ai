from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_collect_or_reconstruct_iql_sequence_metadata_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/COLLECT_OR_RECONSTRUCT_IQL_SEQUENCE_METADATA_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_reproducibility_requires_prior_schema_and_sanity_values() -> None:
    payload = {
        "previous_transition_schema_status_v1": "IQL_TRANSITION_SCHEMA_PARTIAL_NEEDS_SEQUENCE_METADATA",
        "previous_sanity_status_v1": "IQL_OFFLINE_SANITY_PASS_CONTEXTUAL_ONLY_NEEDS_TRANSITION_DESIGN",
        "previous_sanity_no_shortcut_status_v1": "PASS",
        "dataset_rows_v1": 1914,
        "run_id_present_v1": True,
        "decision_timestamp_present_v1": True,
        "event_order_available_v1": True,
        "contextual_policy_selected_rows_v1": 76,
        "contextual_policy_bad_tail_audit_only_v1": [75, 55],
        "contextual_policy_safety_status_v1": "CLEAN",
        "source_safety_shielded_78_rows_v1": 78,
        "no_fake_transitions_were_created_in_previous_gate_v1": True,
    }
    assert gate.validate_reproducibility(payload)
    payload["event_order_available_v1"] = False
    with pytest.raises(RuntimeError, match="IQL_SEQUENCE_METADATA_REPRODUCTION_FAILED"):
        gate.validate_reproducibility(payload)


def test_event_order_and_next_row_validation_guards() -> None:
    order_rows = [
        {
            "run_id_v1": "run_a",
            "missing_timestamps_v1": 0,
            "timestamp_monotonic_after_sort_v1": True,
        }
    ]
    assert gate.validate_event_order(order_rows)
    with pytest.raises(RuntimeError, match="EVENT_ORDER_RECONSTRUCTION_FAILED"):
        gate.validate_event_order([{**order_rows[0], "missing_timestamps_v1": 1}])

    next_rows = [
        {
            "done_candidate_v1": False,
            "ambiguous_next_row_v1": False,
            "cross_run_transition_prevented_v1": True,
        },
        {
            "done_candidate_v1": True,
            "ambiguous_next_row_v1": False,
            "cross_run_transition_prevented_v1": True,
        },
    ]
    assert gate.validate_next_row_candidates(next_rows, expected_rows=2, expected_done=1)
    with pytest.raises(RuntimeError, match="DONE_ROW_COUNT_MISMATCH"):
        gate.validate_next_row_candidates(next_rows, expected_rows=2, expected_done=2)
    with pytest.raises(RuntimeError, match="AMBIGUOUS_NEXT_ROW"):
        gate.validate_next_row_candidates(
            [{**next_rows[0], "ambiguous_next_row_v1": True}, next_rows[1]],
            expected_rows=2,
            expected_done=1,
        )


def test_no_fake_transition_audit_and_go_no_go_block_forbidden_paths() -> None:
    audit = {
        "critical_failures_v1": [],
        "checks_v1": {"no_synthetic_next_state_v1": True},
    }
    assert gate.validate_no_fake_transition_audit(audit)
    with pytest.raises(RuntimeError, match="SYNTHETIC_NEXT_STATE_FORBIDDEN"):
        gate.validate_no_fake_transition_audit({"critical_failures_v1": [], "checks_v1": {}})

    go = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "true_transition_dataset_build_allowed_v1": False,
        "event_ordered_transition_dataset_build_allowed_v1": True,
        "true_sequential_iql_ready_v1": False,
        "event_ordered_research_ready_v1": True,
        "contextual_iql_research_still_valid_v1": True,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_production_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
    }
    assert gate.validate_go_no_go(go)
    with pytest.raises(RuntimeError, match="FORBIDDEN_PATH_OPENED"):
        gate.validate_go_no_go(dict(go, adapter_build_allowed_v1=True))
    with pytest.raises(RuntimeError, match="TRUE_TRANSITION_DATASET_ALLOWED_WITHOUT_TRUE_READY_STATUS"):
        gate.validate_go_no_go(dict(go, true_transition_dataset_build_allowed_v1=True))


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("GO_LIVE", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_materializer_writes_outputs_and_recommends_event_ordered_dataset(tmp_path: Path) -> None:
    artifact_root = tmp_path / "COLLECT_OR_RECONSTRUCT_IQL_SEQUENCE_METADATA_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert summary["event_order_valid_v1"] is True
    assert summary["next_state_can_be_constructed_v1"] == "YES_EVENT_ORDERED_RESEARCH_ONLY"
    assert summary["true_sequential_iql_ready_v1"] is False
    assert summary["event_ordered_research_transition_dataset_ready_v1"] is True
    assert summary["transition_dataset_kind_v1"] == gate.TRANSITION_KIND
    assert summary["expected_transition_rows_v1"] == 1914
    assert summary["expected_terminal_transitions_v1"] == 58
    assert summary["adapter_r6_iql_production_live_remain_blocked_v1"] is True
    assert summary["iql_training_run_v1"] is False

    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    inventory = json.loads((artifact_root / "iql_sequence_metadata_inventory_v1.json").read_text())
    by_name = {row["field_name_v1"]: row for row in inventory["rows_v1"]}
    assert by_name["run_id_v1"]["usable_for_episode_v1"] is True
    assert by_name["decision_timestamp_v1"]["usable_for_ordering_v1"] is True
    assert by_name["done_v1"]["present_v1"] is False
    assert by_name["logged_action_v1"]["present_v1"] is False

    next_rows = json.loads((artifact_root / "iql_next_row_candidate_audit_v1.json").read_text())
    assert next_rows["row_count_v1"] == 1914
    assert sum(1 for row in next_rows["rows_v1"] if row["done_candidate_v1"]) == 58
    assert all(row["cross_run_transition_prevented_v1"] for row in next_rows["rows_v1"])

    no_fake = json.loads((artifact_root / "iql_no_fake_transition_audit_v1.json").read_text())
    assert no_fake["status_v1"] == "PASS"
    assert no_fake["checks_v1"]["no_synthetic_next_state_v1"] is True

    go = json.loads((artifact_root / "collect_or_reconstruct_iql_sequence_metadata_go_no_go_v1.json").read_text())
    assert go["status_v1"] == gate.FINAL_STATUS
    assert go["event_ordered_transition_dataset_build_allowed_v1"] is True
    assert go["true_transition_dataset_build_allowed_v1"] is False
    assert go["iql_production_allowed_v1"] is False
