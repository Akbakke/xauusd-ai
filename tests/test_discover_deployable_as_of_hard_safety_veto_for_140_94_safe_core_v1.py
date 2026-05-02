from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_discover_deployable_as_of_hard_safety_veto_for_140_94_safe_core_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_reproducibility_requires_safe_core_and_unsafe_extra() -> None:
    payload = {
        "selected_rows_v1": 89,
        "recovered_original_140_rows_v1": 86,
        "extra_rows_v1": 3,
        "bad_count_audit_only_v1": 86,
        "tail_count_audit_only_v1": 55,
        "precision_audit_only_v1": 0.9662921348314607,
        "safety_status_v1": "CLEAN",
        "hard_safety_veto_status_v1": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
        "unsafe_extra_without_hard_veto_rows_v1": 1,
    }
    assert gate.validate_reproducibility(payload)
    payload["unsafe_extra_without_hard_veto_rows_v1"] = 0
    with pytest.raises(RuntimeError, match="HARD_SAFETY_VETO_DISCOVERY_REPRODUCTION_FAILED"):
        gate.validate_reproducibility(payload)


def test_candidate_veto_metrics_require_all_families_and_block_proxy_ready() -> None:
    rows = [
        {
            "veto_family_v1": family,
            "candidate_adapter_ready_v1": False,
            "row_identity_risk_v1": False,
            "membership_coverage_proxy_risk_v1": False,
        }
        for family in [
            "SIGNAL_SHAPE_REFINED_VETO",
            "LOW_SUPPORT_OR_MISSING_ARTIFACT_VETO",
            "SAFE_CORE_DISTANCE_MARGIN_VETO",
            "BRANCH_SPECIFIC_VETO",
            "VETO_CONFLUENCE_RULE",
            "FALSE_POSITIVE_RISK_VETO",
        ]
    ]
    assert gate.validate_candidate_veto_metrics(rows)
    rows[0]["candidate_adapter_ready_v1"] = True
    rows[0]["membership_coverage_proxy_risk_v1"] = True
    with pytest.raises(RuntimeError, match="MEMBERSHIP_PROXY_VETO_CANNOT_BE_ADAPTER_READY"):
        gate.validate_candidate_veto_metrics(rows)


def test_final_selection_keeps_adapter_closed() -> None:
    payload = {
        "adapter_reopen_allowed_v1": False,
        "selected_veto_adapter_ready_v1": False,
    }
    assert gate.validate_final_selection(payload)
    payload["adapter_reopen_allowed_v1"] = True
    with pytest.raises(RuntimeError, match="ADAPTER_REOPEN_MUST_REMAIN_FALSE"):
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
    assert gate.validate_final_status(
        "140_94_VETO_FOUND_BUT_TOO_DESTRUCTIVE_TO_SAFE_CORE",
        "REFINE_140_94_HARD_SAFETY_VETO_TO_RETAIN_SAFE_CORE_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("PROMOTE_NOW", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_materializer_writes_required_outputs_and_does_not_reopen_adapter(tmp_path: Path) -> None:
    artifact_root = tmp_path / "DISCOVER_DEPLOYABLE_AS_OF_HARD_SAFETY_VETO_FOR_140_94_SAFE_CORE_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)
    assert summary["safe_core_rule_id_v1"] == gate.SAFE_CORE_RULE_ID
    assert summary["selected_rows_v1"] == 89
    assert summary["recovered_original_140_rows_v1"] == 86
    assert summary["extra_rows_v1"] == 3
    assert summary["bad_tail_audit_only_v1"] == [86, 55]
    assert summary["unsafe_extra_without_hard_veto_rows_v1"] == 1
    assert summary["selected_final_veto_v1"] is None
    assert summary["adapter_reopen_allowed_v1"] is False
    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    metrics = json.loads((artifact_root / "discover_140_94_candidate_veto_metrics_v1.json").read_text())
    assert any(
        row["veto_family_v1"] == "SIGNAL_SHAPE_REFINED_VETO"
        and row["unsafe_row_blocked_v1"] is True
        and row["safe_core_rows_accidentally_blocked_v1"] > gate.ACCEPTABLE_SAFE_CORE_BLOCK_LIMIT
        for row in metrics["rows_v1"]
    )
    assert any(
        row["membership_coverage_proxy_risk_v1"] is True and row["candidate_adapter_ready_v1"] is False
        for row in metrics["rows_v1"]
    )

    go = json.loads(
        (
            artifact_root
            / "discover_deployable_as_of_hard_safety_veto_for_140_94_safe_core_go_no_go_v1.json"
        ).read_text()
    )
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["adapter_reopen_allowed_v1"] is False
    assert go["r6_run_v1"] is False
    assert go["adapter_built_v1"] is False
    assert go["iql_run_v1"] is False
