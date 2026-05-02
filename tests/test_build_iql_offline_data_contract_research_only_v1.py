from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_build_iql_offline_data_contract_research_only_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_no_forbidden_actions_blocks_training_and_production_paths() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(
        r6=True,
        adapter=True,
        iql_production=True,
        iql_training_now=True,
        package=True,
        freeze=True,
        promo=True,
        live=True,
        optuna=True,
        broad_sweep=True,
    )
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "IQL_PRODUCTION_FORBIDDEN" in blocked["failures_v1"]
    assert "IQL_TRAINING_FORBIDDEN_IN_CONTRACT_GATE" in blocked["failures_v1"]


def test_reproducibility_requires_140_safe_core_and_78_shield() -> None:
    payload = {
        "baseline_140_94_v1": {
            "selected_rows_v1": 140,
            "bad_count_audit_only_v1": 140,
            "tail_count_audit_only_v1": 94,
            "safety_status_v1": "CLEAN",
        },
        "safe_core_89_v1": {
            "selected_rows_v1": 89,
            "recovered_original_140_v1": 86,
            "extra_rows_v1": 3,
            "bad_count_audit_only_v1": 86,
            "tail_count_audit_only_v1": 55,
            "precision_audit_only_v1": 0.9662921348314607,
        },
        "source_safety_shielded_78_v1": {
            "selected_rows_v1": 78,
            "original_140_retained_v1": 75,
            "bad_count_audit_only_v1": 75,
            "tail_count_audit_only_v1": 55,
            "precision_audit_only_v1": 0.9615384615384616,
            "safety_status_v1": "CLEAN",
            "unsafe_row_blocked_v1": True,
        },
    }
    assert gate.validate_reproducibility(payload)
    payload["source_safety_shielded_78_v1"]["unsafe_row_blocked_v1"] = False
    with pytest.raises(RuntimeError, match="IQL_OFFLINE_DATA_CONTRACT_REPRODUCTION_FAILED"):
        gate.validate_reproducibility(payload)


def test_state_contract_blocks_labels_membership_blueprint_mfe_and_row_identity() -> None:
    inputs = gate._load_inputs()
    frame, _ = gate._frame_and_masks(inputs)
    rows = gate._state_contract_rows(frame)
    by_name = {row["field_name_v1"]: row for row in rows}

    for field in [
        "bad_label_v1",
        "tail_label_v1",
        "unsafe_audit_v1",
        "HISTORICAL_V2_BLUEPRINT",
        "student_oof_score_v1",
        "candidate_uid_v1",
        "selected_original_140_v1",
        "is_plus45_diagnostic_v1",
        "fifty_plus_mfe_risk_v1",
    ]:
        assert by_name[field]["allowed_as_state_v1"] is False

    assert by_name["candidate_score_v1"]["allowed_as_state_v1"] is True
    assert by_name["signal_r5_1_bad_score_v1"]["allowed_as_state_v1"] is True
    assert gate.validate_state_contract(rows)


def test_safety_shield_blocks_proxy_and_audit_only_state() -> None:
    payload = {
        "selected_rows_v1": 78,
        "unsafe_row_blocked_v1": True,
        "historical_v2_blueprint_allowed_v1": False,
        "membership_or_coverage_proxy_allowed_v1": False,
        "audit_only_veto_allowed_as_state_v1": False,
    }
    assert gate.validate_safety_shield(payload)
    with pytest.raises(RuntimeError, match="IQL_SAFETY_SHIELD_USES_BLOCKED_PROXY"):
        gate.validate_safety_shield(dict(payload, historical_v2_blueprint_allowed_v1=True))


def test_go_no_go_allows_only_research_sanity_next_and_blocks_production_paths() -> None:
    payload = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "research_only_iql_sanity_training_allowed_next_v1": True,
        "iql_training_run_in_this_gate_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
    }
    assert gate.validate_go_no_go(payload)
    with pytest.raises(RuntimeError, match="FORBIDDEN_DOWNSTREAM_PATH_OPENED"):
        gate.validate_go_no_go(dict(payload, r6_allowed_v1=True))
    with pytest.raises(RuntimeError, match="IQL_TRAINING_MUST_NOT_RUN"):
        gate.validate_go_no_go(dict(payload, iql_training_run_in_this_gate_v1=True))


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("RUN_IQL_ANYWAY", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "BUILD_ADAPTER_NOW")


def test_materializer_writes_required_outputs_and_keeps_live_paths_closed(tmp_path: Path) -> None:
    artifact_root = tmp_path / "BUILD_IQL_OFFLINE_DATA_CONTRACT_RESEARCH_ONLY_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert summary["chosen_research_only_eligibility_cohort_v1"] == gate.CHOSEN_RESEARCH_COHORT
    assert summary["source_safety_shielded_78_selected_bad_tail_v1"] == [78, 75, 55]
    assert summary["offline_iql_sanity_training_allowed_next_v1"] is True
    assert summary["iql_training_run_v1"] is False
    assert summary["iql_production_allowed_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["r6_run_v1"] is False

    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    state = json.loads((artifact_root / "iql_offline_state_contract_v1.json").read_text())
    by_name = {row["field_name_v1"]: row for row in state["rows_v1"]}
    assert by_name["bad_label_v1"]["allowed_as_state_v1"] is False
    assert by_name["HISTORICAL_V2_BLUEPRINT"]["allowed_as_state_v1"] is False
    assert by_name["candidate_score_v1"]["allowed_as_state_v1"] is True

    action = json.loads((artifact_root / "iql_offline_action_contract_v1.json").read_text())
    assert action["action_space_status_v1"] == "BINARY_ONLY_SIZING_ACTIONS_NOT_SUPPORTED_YET"

    shield = json.loads((artifact_root / "iql_offline_safety_shield_contract_v1.json").read_text())
    assert shield["primary_research_eligibility_cohort_v1"] == gate.CHOSEN_RESEARCH_COHORT
    assert shield["unsafe_row_blocked_v1"] is True
    assert shield["historical_v2_blueprint_allowed_v1"] is False

    go = json.loads((artifact_root / "build_iql_offline_data_contract_research_only_go_no_go_v1.json").read_text())
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["research_only_iql_sanity_training_allowed_next_v1"] is True
    assert go["iql_training_run_in_this_gate_v1"] is False
    assert go["iql_production_allowed_v1"] is False
    assert go["adapter_build_allowed_v1"] is False
    assert go["r6_allowed_v1"] is False
