from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_design_iql_transition_and_episode_schema_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/DESIGN_IQL_TRANSITION_AND_EPISODE_SCHEMA_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_reproducibility_requires_previous_contextual_sanity_values() -> None:
    payload = {
        "dataset_rows_v1": 1914,
        "state_feature_count_v1": 11,
        "safety_shield_v1": "SOURCE_SAFETY_SHIELDED_78_RESEARCH_ELIGIBILITY",
        "contextual_policy_selected_rows_v1": 76,
        "contextual_policy_bad_tail_audit_only_v1": [75, 55],
        "contextual_policy_safety_status_v1": "CLEAN",
        "previous_sanity_no_shortcut_status_v1": "PASS",
        "baseline_140_94_v1": {"selected_rows_v1": 140},
        "safe_core_89_v1": {"selected_rows_v1": 89},
        "source_safety_shielded_78_v1": {"selected_rows_v1": 78},
        "contextual_only_because_transitions_missing_v1": True,
        "no_fake_transitions_created_v1": True,
    }
    assert gate.validate_reproducibility(payload)
    payload["contextual_policy_selected_rows_v1"] = 78
    with pytest.raises(RuntimeError, match="IQL_TRANSITION_SCHEMA_REPRODUCTION_FAILED"):
        gate.validate_reproducibility(payload)


def test_no_fake_transition_design_guard() -> None:
    assert gate.validate_no_fake_transition_design(
        {
            "recommended_design_v1": "TRANSITION_SCHEMA_NEEDS_SOURCE_METADATA",
            "true_sequential_iql_possible_v1": False,
            "fake_transitions_created_v1": False,
        }
    )
    with pytest.raises(RuntimeError, match="FAKE_TRANSITIONS_FORBIDDEN"):
        gate.validate_no_fake_transition_design(
            {
                "recommended_design_v1": "TRANSITION_SCHEMA_NEEDS_SOURCE_METADATA",
                "true_sequential_iql_possible_v1": False,
                "fake_transitions_created_v1": True,
            }
        )
    with pytest.raises(RuntimeError, match="SEQUENTIAL_READY_WITHOUT_REQUIRED_FIELDS"):
        gate.validate_no_fake_transition_design(
            {
                "recommended_design_v1": "TRUE_SEQUENTIAL_IQL_READY",
                "true_sequential_iql_possible_v1": False,
                "fake_transitions_created_v1": False,
            }
        )


def test_go_no_go_keeps_transition_build_and_production_paths_closed() -> None:
    payload = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "true_sequential_iql_ready_v1": False,
        "transition_dataset_build_allowed_next_v1": False,
        "sequence_metadata_required_before_transition_build_v1": True,
        "contextual_iql_research_allowed_v1": True,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_production_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
    }
    assert gate.validate_go_no_go(payload)
    with pytest.raises(RuntimeError, match="FORBIDDEN_PATH_OPENED"):
        gate.validate_go_no_go(dict(payload, r6_allowed_v1=True))
    with pytest.raises(RuntimeError, match="TRANSITION_DATASET_BUILD_ALLOWED_WITHOUT_READY_STATUS"):
        gate.validate_go_no_go(dict(payload, transition_dataset_build_allowed_next_v1=True))


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("GO_LIVE", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_materializer_writes_required_outputs_and_selects_metadata_gate(tmp_path: Path) -> None:
    artifact_root = tmp_path / "DESIGN_IQL_TRANSITION_AND_EPISODE_SCHEMA_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert summary["recommended_design_v1"] == gate.RECOMMENDED_DESIGN
    assert summary["true_sequential_iql_possible_v1"] is False
    assert summary["event_order_available_v1"] is True
    assert summary["fake_transitions_created_v1"] is False
    assert summary["transition_dataset_build_allowed_next_v1"] is False
    assert summary["adapter_r6_iql_production_live_remain_blocked_v1"] is True
    assert summary["iql_training_run_v1"] is False

    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    inventory = json.loads((artifact_root / "iql_transition_source_inventory_v1.json").read_text())
    by_name = {row["field_name_v1"]: row for row in inventory["rows_v1"]}
    assert by_name["decision_timestamp_v1"]["can_define_ordering_v1"] is True
    assert by_name["run_id_v1"]["can_define_episode_v1"] is True
    assert by_name["next_state_vector_v1"]["present_v1"] is False
    assert by_name["done"]["present_v1"] is False

    design = json.loads((artifact_root / "iql_recommended_transition_design_v1.json").read_text())
    assert design["fake_transitions_created_v1"] is False
    assert "true logged behavior action sequence" in design["missing_sequence_action_reward_fields_v1"]

    no_shortcut = json.loads((artifact_root / "iql_transition_schema_no_shortcut_audit_v1.json").read_text())
    assert no_shortcut["status_v1"] == "PASS"
    assert no_shortcut["checks_v1"]["fake_transitions_not_created_v1"] is True

    go = json.loads((artifact_root / "design_iql_transition_and_episode_schema_go_no_go_v1.json").read_text())
    assert go["status_v1"] == gate.FINAL_STATUS
    assert go["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert go["transition_dataset_build_allowed_next_v1"] is False
    assert go["adapter_build_allowed_v1"] is False
    assert go["r6_allowed_v1"] is False
