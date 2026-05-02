from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_parallel_refined_veto_lineage_audit_lane_pack_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/PARALLEL_REFINED_VETO_LINEAGE_AUDIT_LANE_PACK_V1_20260428T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_lane_index_requires_exact_10_predefined_lanes_and_valid_statuses() -> None:
    rows = [
        {
            "lane_number_v1": idx,
            "lane_id_v1": lane_id,
            "lane_status_v1": "LANE_PASS_NO_BLOCKER_FOUND",
            "classification_v1": "TEST",
            "risk_level_v1": "LOW",
            "blocker_type_v1": "",
            "recommendation_v1": "TEST",
        }
        for idx, lane_id in enumerate(gate.LANES, start=1)
    ]
    assert gate.validate_lane_index(rows)

    bad_rows = list(rows)
    bad_rows[-1] = dict(bad_rows[-1], lane_id_v1="LANE_99_MUTATED")
    with pytest.raises(RuntimeError, match="LANE_INDEX_MUST_CONTAIN_EXACT_10_PREDEFINED_LANES"):
        gate.validate_lane_index(bad_rows)

    bad_status = list(rows)
    bad_status[0] = dict(bad_status[0], lane_status_v1="LANE_SECRET_PROMOTE")
    with pytest.raises(RuntimeError, match="UNKNOWN_LANE_STATUS"):
        gate.validate_lane_index(bad_status)


def test_no_forbidden_actions_guard_blocks_adapter_r6_iql_and_live_paths() -> None:
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
    assert "OPTUNA_FORBIDDEN" in blocked["failures_v1"]


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("OPEN_ADAPTER_DIRECTLY", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "BUILD_ADAPTER_NOW")


def test_lane_pack_go_no_go_cannot_reopen_adapter() -> None:
    go_no_go = {
        "adapter_reopen_allowed_v1": False,
        "r6_run_v1": False,
        "adapter_built_v1": False,
        "iql_run_v1": False,
    }
    assert gate.validate_no_adapter_reopen(go_no_go)
    with pytest.raises(RuntimeError, match="LANE_PACK_MUST_NOT_OPEN_ADAPTER_DIRECTLY"):
        gate.validate_no_adapter_reopen(dict(go_no_go, adapter_reopen_allowed_v1=True))
    with pytest.raises(RuntimeError, match="FORBIDDEN_SIDE_EFFECT_DETECTED"):
        gate.validate_no_adapter_reopen(dict(go_no_go, r6_run_v1=True))


def test_materializer_writes_required_outputs_and_keeps_adapter_closed(tmp_path: Path) -> None:
    artifact_root = tmp_path / "PARALLEL_REFINED_VETO_LINEAGE_AUDIT_LANE_PACK_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)

    assert summary["refined_veto_id_v1"] == gate.REFINED_VETO_ID
    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert summary["refined_veto_can_proceed_to_fan_in_v1"] is False
    assert summary["adapter_r6_iql_remain_blocked_v1"] is True
    assert summary["r6_run_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["iql_run_v1"] is False

    for name in gate.REQUIRED_GLOBAL_OUTPUTS:
        assert (artifact_root / name).exists(), name

    lane_index = json.loads((artifact_root / "parallel_refined_veto_lane_pack_lane_index_v1.json").read_text())
    assert lane_index["row_count_v1"] == 10
    lane_statuses = {row["lane_id_v1"]: row["lane_status_v1"] for row in lane_index["rows_v1"]}
    assert lane_statuses["LANE_01_PROVENANCE_SOURCE_LINEAGE"] == "LANE_BLOCKED_BY_MEMBERSHIP_OR_COVERAGE_PROXY_RISK"
    assert lane_statuses["LANE_02_AS_OF_RECONSTRUCTION"] == "LANE_BLOCKED_BY_AS_OF_RECONSTRUCTION_FAILURE"
    assert lane_statuses["LANE_04_MEMBERSHIP_COVERAGE_PROXY_AUDIT"] == "LANE_BLOCKED_BY_MEMBERSHIP_OR_COVERAGE_PROXY_RISK"
    assert lane_statuses["LANE_06_ROW_IDENTITY_ARTIFACT_SHORTCUT_AUDIT"] == (
        "LANE_BLOCKED_BY_ROW_IDENTITY_OR_ARTIFACT_SHORTCUT"
    )
    assert lane_statuses["LANE_09_ALTERNATIVE_AS_OF_VETO_WITHOUT_V2_BLUEPRINT"] == "LANE_PASS_NO_BLOCKER_FOUND"

    for lane_id in gate.LANES:
        lane_root = artifact_root / "lanes" / lane_id
        assert (lane_root / "lane_manifest_v1.json").exists()
        assert (lane_root / "lane_result_v1.json").exists()
        assert (lane_root / "lane_risk_audit_v1.json").exists()

    go = json.loads(
        (artifact_root / "parallel_refined_veto_lineage_audit_lane_pack_go_no_go_v1.json").read_text()
    )
    assert go["status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert go["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert go["adapter_reopen_allowed_v1"] is False
    assert go["r6_run_v1"] is False
    assert go["adapter_built_v1"] is False
    assert go["iql_run_v1"] is False

    summary_json = json.loads((artifact_root / "parallel_refined_veto_lane_pack_summary_v1.json").read_text())
    assert summary_json["historical_v2_blueprint_as_of_safe_assessment_v1"] == (
        "BLOCKED_OR_UNPROVEN_HISTORICAL_ARTIFACT_PROXY"
    )
    assert summary_json["historical_v2_blueprint_adapter_allowlist_assessment_v1"] == "NOT_ALLOWLIST_ELIGIBLE"
    assert summary_json["adapter_remains_blocked_v1"] is True
