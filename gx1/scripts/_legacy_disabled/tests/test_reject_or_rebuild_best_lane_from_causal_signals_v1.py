from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_reject_or_rebuild_best_lane_from_causal_signals_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [
            Path("/tmp/REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1_20260428T000000Z_LOCK"),
            Path("/tmp/STABILITY_RECHECK_BEST_LANE_185_139_BEFORE_R6_V1_20260427T200530Z_LOCK"),
        ]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/RUN_*_LOCK")])


def test_feature_denylist_blocks_labels_mfe_safe_recoverable_coverage_membership_and_selected_flags() -> None:
    blocked = [
        "bad_label_v1",
        "tail_label_v1",
        "post_outcome_mfe_v1",
        "safe_recoverable_v1",
        "coverage_proxy_member_v1",
        "lane_selected_v1",
        "selected_by_v2_v1",
    ]
    with pytest.raises(RuntimeError, match="FORBIDDEN_CAUSAL_FEATURE"):
        gate.validate_no_forbidden_feature_names(blocked)


def test_feature_allowlist_is_clean_for_causal_rebuild() -> None:
    features = gate.RAW_ALLOWED_FEATURES + list(gate.DERIVED_SIGNAL_FEATURES)
    assert gate.validate_no_forbidden_feature_names(features) is True


def test_reject_preserve_policy_requires_185_reject_and_plus45_diagnostic_only() -> None:
    rows = [
        {"item_v1": "LANE_08_185_139_MEMBERSHIP_BOUNDARY", "decision_v1": "REJECT_AS_DEPLOYABLE"},
        {"item_v1": "LANE_08_PLUS_45_ROWS", "decision_v1": "PRESERVE_AS_DIAGNOSTIC"},
        {"item_v1": "TAIL_REPAIRED_140_94", "decision_v1": "PRESERVE_AS_CAUSAL_CANDIDATE"},
    ]
    assert gate.validate_reject_preserve_policy(rows) is True
    rows[0]["decision_v1"] = "PRESERVE_AS_CAUSAL_CANDIDATE"
    with pytest.raises(RuntimeError, match="MUST_BE_REJECTED_AS_DEPLOYABLE"):
        gate.validate_reject_preserve_policy(rows)


def test_threshold_policy_cannot_use_full_dataset_selection() -> None:
    assert gate.validate_threshold_policy("INNER_GROUP_OOF_SUPERVISED_LABEL_TARGET_NO_HELDOUT_LEAKAGE")
    assert gate.validate_threshold_policy("FIXED_PRE_REGISTERED_RULE_NO_FULL_DATASET_THRESHOLD_SELECTION")
    with pytest.raises(RuntimeError, match="INVALID_THRESHOLD_POLICY"):
        gate.validate_threshold_policy("FULL_DATASET_THRESHOLD_SELECTED")


def test_final_status_and_next_action_are_allowlisted() -> None:
    assert gate.validate_final_status(
        "RETURN_TO_140_94_CAUSAL_BASELINE_BEST_CURRENT_OPTION",
        "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("PROMOTE_TO_LIVE", "RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1")
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status("RETURN_TO_140_94_CAUSAL_BASELINE_BEST_CURRENT_OPTION", "RUN_R6_NOW")


def test_forbidden_side_effects_are_blocked() -> None:
    clean = gate.validate_no_forbidden_actions()
    assert clean["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(r6=True, adapter=True, package=True, freeze=True, promo=True, live=True)
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "PACKAGE_BUILD_FORBIDDEN" in blocked["failures_v1"]


def test_materializer_writes_required_outputs_and_keeps_go_no_go_valid(tmp_path: Path) -> None:
    artifact_root = tmp_path / "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)
    assert summary["final_status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert summary["next_recommended_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert summary["r6_run_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["package_built_v1"] is False
    required = [
        "causal_rebuild_input_manifest_v1.json",
        "causal_rebuild_reject_preserve_audit_v1.json",
        "causal_signal_inventory_v1.csv",
        "causal_feature_allowlist_v1.json",
        "causal_feature_denylist_v1.json",
        "causal_feature_lineage_audit_v1.csv",
        "baseline_student_bestlane_comparison_v1.json",
        "causal_rebuild_candidate_definitions_v1.json",
        "causal_rebuild_candidate_oof_predictions_v1.csv",
        "causal_rebuild_candidate_metrics_v1.csv",
        "causal_rebuild_threshold_selection_audit_v1.json",
        "causal_rebuild_outcome_safety_audit_v1.csv",
        "causal_rebuild_plus45_diagnostic_audit_v1.csv",
        "causal_rebuild_near_miss_and_near_fail_rows_v1.csv",
        "causal_rebuild_unsafe_lookalike_audit_v1.json",
        "causal_rebuild_group_stability_audit_v1.csv",
        "causal_rebuild_anti_overfit_no_shortcut_audit_v1.json",
        "causal_rebuild_adapter_feasibility_audit_v1.json",
        "causal_rebuild_candidate_ranking_v1.json",
        "reject_or_rebuild_best_lane_from_causal_signals_recommendation_v1.json",
        "reject_or_rebuild_best_lane_from_causal_signals_go_no_go_v1.json",
    ]
    for name in required:
        assert (artifact_root / name).exists(), name
    go = json.loads((artifact_root / "reject_or_rebuild_best_lane_from_causal_signals_go_no_go_v1.json").read_text())
    assert go["status_v1"] == summary["final_status_v1"]
    ranking = json.loads((artifact_root / "causal_rebuild_candidate_ranking_v1.json").read_text())
    assert ranking["best_lane_185_139_role_v1"] == "COMPARATOR_DIAGNOSTIC_ONLY"
    assert ranking["anything_beats_140_94_honestly_v1"] is False
