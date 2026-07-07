from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_parallel_contextual_iql_state_action_research_lane_pack_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/PARALLEL_CONTEXTUAL_IQL_STATE_ACTION_RESEARCH_LANE_PACK_V1_20260429T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_lane_index_requires_exact_10_lanes_and_known_statuses() -> None:
    rows = [
        {
            "lane_number_v1": idx,
            "lane_id_v1": lane_id,
            "lane_status_v1": "LANE_PASS_PROMISING_FOR_NEXT_STAGE",
            "classification_v1": "TEST",
            "risk_level_v1": "LOW",
            "blocker_type_v1": "",
            "recommendation_v1": "TEST",
        }
        for idx, lane_id in enumerate(gate.LANES, start=1)
    ]
    assert gate.validate_lane_index(rows)

    bad_lane = list(rows)
    bad_lane[-1] = dict(bad_lane[-1], lane_id_v1="LANE_99_FAKE")
    with pytest.raises(RuntimeError, match="LANE_INDEX_MUST_CONTAIN_EXACT_10_PREDEFINED_LANES"):
        gate.validate_lane_index(bad_lane)

    bad_status = list(rows)
    bad_status[0] = dict(bad_status[0], lane_status_v1="LANE_PROMOTE_TO_LIVE")
    with pytest.raises(RuntimeError, match="UNKNOWN_LANE_STATUS"):
        gate.validate_lane_index(bad_status)


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(gate.FINAL_STATUS, gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("OPEN_ADAPTER_NOW", gate.NEXT_ACTION)
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(gate.FINAL_STATUS, "RUN_R6_NOW")


def test_no_forbidden_actions_guard_blocks_production_paths() -> None:
    payload = {
        "adapter_built_v1": False,
        "adapter_opened_v1": False,
        "r6_run_v1": False,
        "iql_production_opened_v1": False,
        "production_iql_training_run_v1": False,
        "policy_promotion_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "optuna_run_v1": False,
        "broad_sweep_run_v1": False,
    }
    assert gate.validate_no_forbidden_actions(payload)
    with pytest.raises(RuntimeError, match="FORBIDDEN_SIDE_EFFECT_DETECTED"):
        gate.validate_no_forbidden_actions(dict(payload, r6_run_v1=True))
    with pytest.raises(RuntimeError, match="FORBIDDEN_SIDE_EFFECT_DETECTED"):
        gate.validate_no_forbidden_actions(dict(payload, iql_production_opened_v1=True))
    with pytest.raises(RuntimeError, match="FORBIDDEN_SIDE_EFFECT_DETECTED"):
        gate.validate_no_forbidden_actions(dict(payload, optuna_run_v1=True))


def test_go_no_go_keeps_adapter_r6_iql_production_live_blocked() -> None:
    payload = {
        "status_v1": gate.FINAL_STATUS,
        "next_recommended_action_v1": gate.NEXT_ACTION,
        "adapter_r6_iql_production_live_remain_blocked_v1": True,
        "adapter_built_v1": False,
        "adapter_opened_v1": False,
        "r6_run_v1": False,
        "iql_production_opened_v1": False,
        "production_iql_training_run_v1": False,
        "policy_promotion_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "optuna_run_v1": False,
        "broad_sweep_run_v1": False,
    }
    assert gate.validate_go_no_go(payload)
    with pytest.raises(RuntimeError, match="LANE_PACK_MUST_KEEP_PRODUCTION_PATHS_BLOCKED"):
        gate.validate_go_no_go(dict(payload, adapter_r6_iql_production_live_remain_blocked_v1=False))
    with pytest.raises(RuntimeError, match="FORBIDDEN_SIDE_EFFECT_DETECTED"):
        gate.validate_go_no_go(dict(payload, policy_promotion_run_v1=True))


def test_no_shortcut_payload_blocks_critical_failures() -> None:
    assert gate.validate_no_shortcut_payload({"status_v1": "PASS", "critical_failures_v1": []})
    with pytest.raises(RuntimeError, match="CONTEXTUAL_IQL_LANE_PACK_NO_SHORTCUT_FAILED"):
        gate.validate_no_shortcut_payload({"status_v1": "FAIL", "critical_failures_v1": ["reward leaked"]})


def test_materializer_writes_lane_pack_outputs_and_selects_state_feature_rebuild(tmp_path: Path) -> None:
    artifact_root = (
        tmp_path / "PARALLEL_CONTEXTUAL_IQL_STATE_ACTION_RESEARCH_LANE_PACK_V1_20260429T000000Z_LOCK"
    )
    summary = gate.materialize(artifact_root)

    assert summary["final_status_v1"] == gate.FINAL_STATUS
    assert summary["next_recommended_action_v1"] == gate.NEXT_ACTION
    assert summary["selected_next_mainline_direction_v1"] == gate.NEXT_ACTION
    assert summary["contextual_remains_preferred_v1"] is True
    assert summary["adapter_r6_iql_production_live_remain_blocked_v1"] is True
    assert summary["iql_training_run_v1"] is False
    assert summary["state_feature_assessment_v1"] == "HIGHEST_LEVERAGE_NEXT_PATH"
    assert summary["transformer_assessment_v1"] == "TRANSFORMER_FEATURES_NOT_READY_NOT_BLOCKER"

    for name in gate.REQUIRED_GLOBAL_OUTPUTS:
        assert (artifact_root / name).exists(), name

    lane_index = json.loads((artifact_root / "contextual_iql_parallel_lane_index_v1.json").read_text())
    assert lane_index["row_count_v1"] == 10
    lane_statuses = {row["lane_id_v1"]: row["lane_status_v1"] for row in lane_index["rows_v1"]}
    assert lane_statuses["LANE_01_CONTEXTUAL_BASELINE_LOCK_AND_REPRO"] == "LANE_PASS_PROMISING_FOR_NEXT_STAGE"
    assert lane_statuses["LANE_02_AS_OF_SOURCE_STATE_FEATURE_EXPANSION"] == "LANE_PASS_PROMISING_FOR_NEXT_STAGE"
    assert lane_statuses["LANE_04_TRANSFORMER_FEATURE_LINEAGE_AUDIT"] == "LANE_BLOCKED_BY_MISSING_FEATURE_LINEAGE"
    assert lane_statuses["LANE_05_ACTION_SUPPORT_AND_BEHAVIOR_POLICY_AUDIT"] == "LANE_PASS_USEFUL_BUT_SECONDARY"
    assert lane_statuses["LANE_10_FAN_IN_RANKING_AND_NEXT_MAINLINE_DECISION"] == (
        "LANE_PASS_PROMISING_FOR_NEXT_STAGE"
    )

    for lane_id in gate.LANES:
        lane_root = artifact_root / lane_id
        for name in gate.REQUIRED_LANE_OUTPUTS:
            assert (lane_root / name).exists(), f"{lane_id}/{name}"

    repro = json.loads((artifact_root / "contextual_iql_parallel_lane_pack_reproducibility_audit_v1.json").read_text())
    assert repro["dataset_rows_v1"] == 1914
    assert repro["episodes_v1"] == 58
    assert repro["take_trade_count_v1"] == 78
    assert repro["skip_count_v1"] == 1836
    assert repro["contextual_equivalent_selected_rows_v1"] == 70
    assert repro["contextual_equivalent_reward_v1"] == pytest.approx(92.0)
    assert repro["contextual_equivalent_bad_tail_audit_only_v1"] == [69, 55]
    assert repro["event_ordered_fixed_policy_reward_v1"] == pytest.approx(91.75)

    l1 = json.loads((artifact_root / "LANE_01_CONTEXTUAL_BASELINE_LOCK_AND_REPRO/lane_result_v1.json").read_text())
    assert l1["classification_v1"] == "CONTEXTUAL_BASELINE_LOCKED"

    l2 = json.loads((artifact_root / "LANE_02_AS_OF_SOURCE_STATE_FEATURE_EXPANSION/lane_result_v1.json").read_text())
    assert l2["classification_v1"] == "BEST_NEXT_RESEARCH_LEVER"

    recommendation = json.loads((artifact_root / "contextual_iql_parallel_fan_in_recommendation_v1.json").read_text())
    assert recommendation["selected_path_v1"] == "STATE_FEATURE_REBUILD"
    assert recommendation["next_recommended_action_v1"] == gate.NEXT_ACTION

    go = json.loads(
        (artifact_root / "parallel_contextual_iql_state_action_research_lane_pack_go_no_go_v1.json").read_text()
    )
    assert go["status_v1"] == gate.FINAL_STATUS
    assert go["event_ordered_iql_parked_v1"] is True
    assert go["adapter_r6_iql_production_live_remain_blocked_v1"] is True
    assert go["adapter_built_v1"] is False
    assert go["r6_run_v1"] is False
    assert go["iql_production_opened_v1"] is False
    assert go["policy_promotion_run_v1"] is False
