from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_refine_140_94_hard_safety_veto_to_retain_safe_core_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/REFINE_140_94_HARD_SAFETY_VETO_TO_RETAIN_SAFE_CORE_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_reproducibility_requires_safe_core_and_prior_destructive_cut() -> None:
    payload = {
        "selected_rows_v1": 89,
        "recovered_original_140_rows_v1": 86,
        "extra_rows_v1": 3,
        "bad_count_audit_only_v1": 86,
        "tail_count_audit_only_v1": 55,
        "precision_audit_only_v1": 0.9662921348314607,
        "safety_status_v1": "CLEAN",
        "unsafe_extra_without_hard_veto_rows_v1": 1,
        "best_prior_deployable_destructive_candidate_v1": "SIGNAL_SHAPE_REFINED_NO_R5_TAIL_R5_BAD_SCORE_GE_099",
        "best_prior_deployable_destructive_safe_core_cut_v1": 21,
    }
    assert gate.validate_reproducibility(payload)
    payload["best_prior_deployable_destructive_safe_core_cut_v1"] = 5
    with pytest.raises(RuntimeError, match="REFINE_140_94_HARD_SAFETY_VETO_REPRODUCTION_FAILED"):
        gate.validate_reproducibility(payload)


def test_retention_tier_boundaries() -> None:
    assert gate.retention_tier(unsafe_row_blocked=True, good_rows_cut=5) == "GREEN"
    assert gate.retention_tier(unsafe_row_blocked=True, good_rows_cut=6) == "YELLOW"
    assert gate.retention_tier(unsafe_row_blocked=True, good_rows_cut=11) == "ORANGE"
    assert gate.retention_tier(unsafe_row_blocked=True, good_rows_cut=21) == "RED"
    assert gate.retention_tier(unsafe_row_blocked=False, good_rows_cut=0) == "BLOCKED"
    assert gate.retention_tier(unsafe_row_blocked=True, good_rows_cut=0, shortcut_or_leakage=True) == "BLOCKED"


def test_candidate_metrics_require_families_and_block_proxy_ready() -> None:
    rows = [
        {
            "veto_family_v1": family,
            "membership_proxy_risk_v1": False,
            "row_identity_risk_v1": False,
            "lineage_status_v1": "CONFIRMED_AS_OF_SIGNAL_SHAPE",
            "adapter_ready_v1": False,
        }
        for family in [
            "BRANCH_LOCAL_SIGNAL_SHAPE_VETO",
            "TWO_CONDITION_CONFLUENCE_VETO",
            "RELAXED_SIGNAL_SHAPE_THRESHOLD_VETO",
            "EXCEPTION_GUARDED_SIGNAL_SHAPE_VETO",
            "LOW_SUPPORT_AWARE_SIGNAL_SHAPE_VETO",
            "MINIMAL_DESTRUCTIVE_VETO",
            "DIAGNOSTIC_STUDENT_DISTANCE_COMPARISON",
        ]
    ]
    assert gate.validate_candidate_metrics(rows)
    rows[-1]["membership_proxy_risk_v1"] = True
    rows[-1]["adapter_ready_v1"] = True
    with pytest.raises(RuntimeError, match="MEMBERSHIP_PROXY_VETO_CANNOT_BE_ADAPTER_READY"):
        gate.validate_candidate_metrics(rows)


def test_final_selection_requires_lineage_confirmation_before_adapter() -> None:
    payload = {
        "selected_refined_veto_name_v1": "EXCEPTION_GUARDED_SIGNAL_SHAPE_REQUIRE_HISTORICAL_V2_BLUEPRINT_V1",
        "adapter_reopen_allowed_now_v1": False,
        "good_safe_core_rows_cut_v1": 3,
        "lineage_status_v1": "NEEDS_LINEAGE_CONFIRMATION",
    }
    assert gate.validate_final_selection(payload)
    payload["adapter_reopen_allowed_now_v1"] = True
    with pytest.raises(RuntimeError, match="ADAPTER_REOPEN_MUST_WAIT"):
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
    )
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "IQL_FORBIDDEN" in blocked["failures_v1"]


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("PROMOTE_NOW", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_materializer_writes_required_outputs_and_keeps_adapter_closed(tmp_path: Path) -> None:
    artifact_root = tmp_path / "REFINE_140_94_HARD_SAFETY_VETO_TO_RETAIN_SAFE_CORE_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)
    assert summary["safe_core_rule_id_v1"] == gate.SAFE_CORE_RULE_ID
    assert summary["selected_rows_v1"] == 89
    assert summary["recovered_original_140_rows_v1"] == 86
    assert summary["extra_rows_v1"] == 3
    assert summary["bad_tail_audit_only_v1"] == [86, 55]
    assert summary["unsafe_row_blocked_v1"] is True
    assert summary["good_safe_core_rows_cut_v1"] == 3
    assert summary["retention_tier_v1"] == "GREEN"
    assert summary["adapter_reopen_allowed_now_v1"] is False
    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    metrics = json.loads((artifact_root / "refine_140_94_refined_veto_candidate_metrics_v1.json").read_text())
    selected = next(
        row
        for row in metrics["rows_v1"]
        if row["candidate_name_v1"] == "EXCEPTION_GUARDED_SIGNAL_SHAPE_REQUIRE_HISTORICAL_V2_BLUEPRINT_V1"
    )
    assert selected["retention_tier_v1"] == "GREEN"
    assert selected["safe_core_rows_cut_v1"] == 3
    assert selected["lineage_status_v1"] == "NEEDS_LINEAGE_CONFIRMATION"
    assert selected["adapter_ready_v1"] is False
    assert any(
        row["veto_family_v1"] == "DIAGNOSTIC_STUDENT_DISTANCE_COMPARISON"
        and row["membership_proxy_risk_v1"] is True
        and row["adapter_ready_v1"] is False
        for row in metrics["rows_v1"]
    )

    go = json.loads((artifact_root / "refine_140_94_hard_safety_veto_to_retain_safe_core_go_no_go_v1.json").read_text())
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["adapter_reopen_allowed_now_v1"] is False
    assert go["r6_run_v1"] is False
    assert go["adapter_built_v1"] is False
    assert go["iql_run_v1"] is False
