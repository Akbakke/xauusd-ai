from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_refine_clean_as_of_safety_layer_to_retain_safe_core_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_retention_class_boundaries() -> None:
    assert gate.retention_class(unsafe_row_blocked=True, good_rows_cut=5) == "GREEN"
    assert gate.retention_class(unsafe_row_blocked=True, good_rows_cut=6) == "YELLOW"
    assert gate.retention_class(unsafe_row_blocked=True, good_rows_cut=11) == "ORANGE"
    assert gate.retention_class(unsafe_row_blocked=True, good_rows_cut=21) == "RED"
    assert gate.retention_class(unsafe_row_blocked=False, good_rows_cut=0) == "BLOCKED"
    assert gate.retention_class(unsafe_row_blocked=True, good_rows_cut=0, shortcut_or_leakage=True) == "BLOCKED"


def test_reproducibility_requires_prior_minimal_orange_cut() -> None:
    payload = {
        "safe_core_selected_rows_v1": 89,
        "safe_core_recovered_original_140_v1": 86,
        "safe_core_extra_rows_v1": 3,
        "safe_core_bad_count_audit_only_v1": 86,
        "safe_core_tail_count_audit_only_v1": 55,
        "safe_core_precision_audit_only_v1": 0.9662921348314607,
        "safe_core_safety_status_v1": "CLEAN",
        "unsafe_extra_without_hard_veto_rows_v1": 1,
        "prior_minimal_source_veto_blocks_unsafe_v1": True,
        "prior_minimal_source_veto_good_rows_cut_v1": 11,
        "prior_minimal_source_veto_retention_class_v1": "ORANGE",
    }
    assert gate.validate_reproducibility(payload)
    payload["prior_minimal_source_veto_good_rows_cut_v1"] = 5
    with pytest.raises(RuntimeError, match="REFINE_CLEAN_SAFETY_LAYER_REPRODUCTION_FAILED"):
        gate.validate_reproducibility(payload)


def test_candidate_metrics_require_candidates_and_no_clean_green_yellow() -> None:
    rows = [
        {
            "candidate_name_v1": name,
            "retention_class_v1": "BLOCKED",
            "proxy_leakage_risk_v1": False,
            "unsafe_row_blocked_v1": False,
            "uses_historical_v2_blueprint_v1": False,
            "uses_student_or_membership_proxy_v1": False,
            "adapter_ready_v1": False,
        }
        for name in [
            "BRANCH_LOCAL_SOURCE_HARD_VETO_V1",
            "SOURCE_CONFLUENCE_REFINED_VETO_V1",
            "RELAXED_SOURCE_THRESHOLD_VETO_V1",
            "GOOD_CORE_EXCEPTION_GUARD_SOURCE_VETO_V1",
            "LOW_SUPPORT_AWARE_SOURCE_VETO_V1",
            "MINIMAL_GREEN_SOURCE_VETO_V1",
            "YELLOW_REVIEW_SOURCE_VETO_V1",
        ]
    ]
    rows[1]["retention_class_v1"] = "ORANGE"
    rows[1]["unsafe_row_blocked_v1"] = True
    assert gate.validate_candidate_metrics(rows)
    rows[1]["retention_class_v1"] = "YELLOW"
    with pytest.raises(RuntimeError, match="UNEXPECTED_CLEAN_GREEN_OR_YELLOW_CANDIDATE"):
        gate.validate_candidate_metrics(rows)


def test_final_selection_keeps_orange_out_of_input_mapping() -> None:
    payload = {
        "selected_candidate_name_v1": gate.FINAL_REFINED_CANDIDATE,
        "retention_class_v1": "ORANGE",
        "adapter_input_mapping_allowed_next_v1": False,
        "uses_historical_v2_blueprint_v1": False,
    }
    assert gate.validate_final_selection(payload)
    payload["adapter_input_mapping_allowed_next_v1"] = True
    with pytest.raises(RuntimeError, match="ORANGE_REFINED_CANDIDATE_CANNOT_GO_TO_INPUT_MAPPING"):
        gate.validate_final_selection(payload)


def test_no_forbidden_actions_guard() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(
        r6=True,
        adapter=True,
        iql=True,
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
    assert "IQL_FORBIDDEN" in blocked["failures_v1"]
    assert "BROAD_SWEEP_FORBIDDEN" in blocked["failures_v1"]


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("GO_TO_MAPPING_ANYWAY", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_go_no_go_blocks_mapping_adapter_r6_iql() -> None:
    payload = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "input_mapping_allowed_next_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
    }
    assert gate.validate_go_no_go(payload)
    with pytest.raises(RuntimeError, match="INPUT_MAPPING_MUST_NOT_BE_ALLOWED"):
        gate.validate_go_no_go(dict(payload, input_mapping_allowed_next_v1=True))


def test_materializer_writes_required_outputs_and_keeps_adapter_closed(tmp_path: Path) -> None:
    artifact_root = tmp_path / "REFINE_CLEAN_AS_OF_SAFETY_LAYER_TO_RETAIN_SAFE_CORE_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["safe_core_rule_id_v1"] == gate.SAFE_CORE_RULE_ID
    assert summary["safe_core_selected_rows_v1"] == 89
    assert summary["safe_core_recovered_original_140_v1"] == 86
    assert summary["safe_core_extra_rows_v1"] == 3
    assert summary["safe_core_bad_tail_audit_only_v1"] == [86, 55]
    assert summary["prior_minimal_source_veto_good_rows_cut_v1"] == 11
    assert summary["selected_refined_candidate_v1"] == gate.FINAL_REFINED_CANDIDATE
    assert summary["unsafe_row_blocked_v1"] is True
    assert summary["good_rows_cut_v1"] == 11
    assert summary["retention_class_v1"] == "ORANGE"
    assert summary["adapter_readiness_preassessment_v1"] == "NOT_READY_STILL_ORANGE_DESTRUCTIVE"
    assert summary["adapter_r6_iql_remain_blocked_v1"] is True
    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION

    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    cut_rows = json.loads((artifact_root / "refine_clean_safety_layer_cut_11_good_rows_audit_v1.json").read_text())
    assert cut_rows["row_count_v1"] == 11
    assert any("HISTORICAL_V2_BLUEPRINT" in row["source_evidence_v1"] for row in cut_rows["rows_v1"])
    assert any(row["source_signals_that_can_protect_row_v1"] == "NONE_FOUND" for row in cut_rows["rows_v1"])

    metrics = json.loads((artifact_root / "refine_clean_safety_layer_candidate_metrics_v1.json").read_text())
    selected = next(row for row in metrics["rows_v1"] if row["candidate_name_v1"] == gate.FINAL_REFINED_CANDIDATE)
    assert selected["unsafe_row_blocked_v1"] is True
    assert selected["retention_class_v1"] == "ORANGE"
    assert selected["uses_historical_v2_blueprint_v1"] is False
    assert selected["adapter_ready_v1"] is False
    assert all(
        not row["adapter_ready_v1"]
        for row in metrics["rows_v1"]
        if row["uses_historical_v2_blueprint_v1"] or row["uses_student_or_membership_proxy_v1"]
    )

    go = json.loads((artifact_root / "refine_clean_as_of_safety_layer_to_retain_safe_core_go_no_go_v1.json").read_text())
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["input_mapping_allowed_next_v1"] is False
    assert go["adapter_build_allowed_v1"] is False
    assert go["r6_allowed_v1"] is False
    assert go["iql_allowed_v1"] is False
