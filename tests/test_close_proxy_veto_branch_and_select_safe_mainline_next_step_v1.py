from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_close_proxy_veto_branch_and_select_safe_mainline_next_step_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/CLOSE_PROXY_VETO_BRANCH_AND_SELECT_SAFE_MAINLINE_NEXT_STEP_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_branch_closure_blocks_adapter_r6_iql_and_blueprint_refinement() -> None:
    record = {
        "branch_closed_as_deployable_mainline_v1": True,
        "historical_v2_blueprint_rejected_as_adapter_input_now_v1": True,
        "refined_veto_preserved_as_diagnostic_only_v1": True,
        "fan_in_decision_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
        "continue_blueprint_refinement_without_new_as_of_sources_v1": False,
    }
    assert gate.validate_branch_closure(record)
    record["continue_blueprint_refinement_without_new_as_of_sources_v1"] = True
    with pytest.raises(RuntimeError, match="PROXY_VETO_BRANCH_CLOSURE_INVALID"):
        gate.validate_branch_closure(record)


def test_option_ranking_selects_clean_as_of_safety_layer_and_opens_nothing() -> None:
    rows = gate._rank_options(gate._next_direction_options())
    selected = [row for row in rows if row["selected_next_direction_v1"]]
    assert len(selected) == 1
    assert selected[0]["option_id_v1"] == "OPTION_B_BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS"
    assert selected[0]["rank_v1"] == 1
    assert all(row["opens_adapter_now_v1"] is False for row in rows)
    assert all(row["runs_r6_now_v1"] is False for row in rows)
    assert all(row["runs_iql_now_v1"] is False for row in rows)


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
        gate.validate_final_status("REFINE_BLUEPRINT_FOREVER", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "BUILD_ADAPTER_NOW")


def test_go_no_go_keeps_adapter_r6_iql_blocked() -> None:
    payload = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "proxy_veto_branch_closed_v1": True,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "iql_allowed_v1": False,
    }
    assert gate.validate_go_no_go(payload)
    with pytest.raises(RuntimeError, match="GO_NO_GO_MUST_KEEP_ADAPTER_R6_IQL_BLOCKED"):
        gate.validate_go_no_go(dict(payload, adapter_build_allowed_v1=True))
    with pytest.raises(RuntimeError, match="GO_NO_GO_MUST_CLOSE_PROXY_VETO_BRANCH"):
        gate.validate_go_no_go(dict(payload, proxy_veto_branch_closed_v1=False))


def test_materializer_writes_required_outputs_and_selects_safe_mainline(tmp_path: Path) -> None:
    artifact_root = tmp_path / "CLOSE_PROXY_VETO_BRANCH_AND_SELECT_SAFE_MAINLINE_NEXT_STEP_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["proxy_veto_branch_closed_v1"] is True
    assert summary["historical_v2_blueprint_deployable_now_v1"] is False
    assert summary["safe_core_rule_id_v1"] == gate.SAFE_CORE_RULE_ID
    assert summary["safe_core_selected_rows_v1"] == 89
    assert summary["safe_core_recovered_original_140_v1"] == 86
    assert summary["safe_core_bad_tail_audit_only_v1"] == [86, 55]
    assert summary["selected_next_mainline_direction_v1"] == (
        "OPTION_B_BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS"
    )
    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert summary["adapter_r6_iql_remain_blocked_v1"] is True
    assert summary["r6_run_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["iql_run_v1"] is False

    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name

    closure = json.loads((artifact_root / "close_proxy_veto_branch_closure_record_v1.json").read_text())
    assert closure["branch_closed_as_deployable_mainline_v1"] is True
    assert closure["fan_in_decision_allowed_v1"] is False
    assert "LANE_01_PROVENANCE_SOURCE_LINEAGE" in closure["blocking_lane_ids_v1"]
    assert "LANE_04_MEMBERSHIP_COVERAGE_PROXY_AUDIT" in closure["blocking_lane_ids_v1"]

    ranking = json.loads((artifact_root / "close_proxy_veto_option_ranking_v1.json").read_text())
    selected = [row for row in ranking["rows_v1"] if row["selected_next_direction_v1"]]
    assert selected[0]["rank_v1"] == 1
    assert selected[0]["option_id_v1"] == "OPTION_B_BUILD_CLEAN_AS_OF_SAFETY_FEATURE_LAYER_FROM_SOURCE_SIGNALS"

    go = json.loads(
        (artifact_root / "close_proxy_veto_branch_and_select_safe_mainline_next_step_go_no_go_v1.json").read_text()
    )
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["proxy_veto_branch_closed_v1"] is True
    assert go["adapter_build_allowed_v1"] is False
    assert go["r6_allowed_v1"] is False
    assert go["iql_allowed_v1"] is False
    assert go["further_v2_blueprint_refinement_recommended_v1"] is False
