from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_plus45_as_of_feature_gap_shadow_exploration_v1 as gate


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [
            Path("/tmp/PLUS45_AS_OF_FEATURE_GAP_SHADOW_EXPLORATION_V1_20260428T000000Z_LOCK"),
            Path("/tmp/RETURN_TO_140_94_CAUSAL_BASELINE_AND_PRECHECK_ADAPTER_V1_20260428T065344Z_LOCK"),
        ]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/*_LOCK")])


def test_no_forbidden_side_effects_are_default_and_blockable() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(
        r6=True,
        adapter=True,
        package=True,
        freeze=True,
        promo=True,
        live=True,
        optuna=True,
        production_model=True,
    )
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "PACKAGE_BUILD_FORBIDDEN" in blocked["failures_v1"]
    assert "PRODUCTION_MODEL_TRAINING_FORBIDDEN" in blocked["failures_v1"]


def test_diagnostic_only_policy_protects_mainline() -> None:
    assert gate.validate_diagnostic_only_policy(
        {
            "mainline_next_action_v1": gate.MAINLINE_NEXT_ACTION,
            "plus45_role_v1": "DIAGNOSTIC_ONLY_NOT_TARGET_FEATURE_FILTER_OR_THRESHOLD_OBJECTIVE",
            "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
            "coverage_proxy_role_v1": "COMPARATOR_ONLY_NOT_FEATURE_FILTER_OR_TARGET",
        }
    )
    with pytest.raises(RuntimeError, match="MAINLINE_NEXT_ACTION_MUST_REMAIN"):
        gate.validate_diagnostic_only_policy(
            {
                "mainline_next_action_v1": "RUN_R6_NOW",
                "plus45_role_v1": "DIAGNOSTIC_ONLY_NOT_TARGET_FEATURE_FILTER_OR_THRESHOLD_OBJECTIVE",
                "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
                "coverage_proxy_role_v1": "COMPARATOR_ONLY_NOT_FEATURE_FILTER_OR_TARGET",
            }
        )
    with pytest.raises(RuntimeError, match="PLUS45_MUST_REMAIN_DIAGNOSTIC_ONLY"):
        gate.validate_diagnostic_only_policy(
            {
                "mainline_next_action_v1": gate.MAINLINE_NEXT_ACTION,
                "plus45_role_v1": "TARGET",
                "best_lane_185_139_role_v1": "COMPARATOR_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
                "coverage_proxy_role_v1": "COMPARATOR_ONLY_NOT_FEATURE_FILTER_OR_TARGET",
            }
        )


def test_denylist_blocks_leakage_fields() -> None:
    blocked = [
        "bad_label_v1",
        "tail_label_v1",
        "post_outcome_mfe_v1",
        "safe_recoverable_v1",
        "coverage_proxy_membership_v1",
        "plus45_flag_v1",
        "lane_selected_v1",
        "selected_by_v2_v1",
        "candidate_uid_v1",
        "artifact_path_v1",
    ]
    with pytest.raises(RuntimeError, match="FORBIDDEN_PLUS45_SHADOW_FEATURE"):
        gate.validate_no_forbidden_feature_names(blocked)


def test_reference_feature_names_are_clean() -> None:
    assert gate.validate_no_forbidden_feature_names(gate.AS_OF_REFERENCE_FEATURES)


def test_cohort_counts_require_exact_140_185_45() -> None:
    payload = {
        "baseline_140_94_selected_rows_v1": 140,
        "best_lane_185_139_selected_rows_v1": 185,
        "plus45_rows_v1": 45,
        "plus45_bad_rows_audit_only_v1": 45,
        "plus45_tail_rows_audit_only_v1": 45,
    }
    assert gate.validate_cohort_counts(payload)
    payload["plus45_rows_v1"] = 44
    with pytest.raises(RuntimeError, match="PLUS45_COHORT_RECONSTRUCTION_FAILED"):
        gate.validate_cohort_counts(payload)


def test_final_shadow_status_and_action_are_allowed() -> None:
    assert gate.validate_final_shadow_status(
        "PLUS45_SHADOW_FOUND_ONLY_MEMBERSHIP_OR_COVERAGE_DEPENDENCY",
        "ARCHIVE_PLUS45_AS_DIAGNOSTIC_ONLY_AND_CONTINUE_140_94_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_SHADOW_STATUS_NOT_ALLOWED"):
        gate.validate_final_shadow_status("PROMOTE_NOW", "ARCHIVE_PLUS45_AS_DIAGNOSTIC_ONLY_AND_CONTINUE_140_94_V1")
    with pytest.raises(RuntimeError, match="SHADOW_NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_shadow_status("PLUS45_SHADOW_NO_ACTIONABLE_AS_OF_SIGNAL_FOUND", "RUN_R6_NOW")


def test_materializer_writes_required_outputs_and_valid_go_no_go(tmp_path: Path) -> None:
    artifact_root = tmp_path / "PLUS45_AS_OF_FEATURE_GAP_SHADOW_EXPLORATION_V1_20260428T000000Z_LOCK"
    summary = gate.materialize(artifact_root)
    assert summary["mainline_next_action_v1"] == gate.MAINLINE_NEXT_ACTION
    assert summary["r6_run_v1"] is False
    assert summary["adapter_built_v1"] is False
    assert summary["package_built_v1"] is False
    assert summary["plus45_rows_v1"] == 45
    for name in gate.REQUIRED_OUTPUTS:
        assert (artifact_root / name).exists(), name
    go = json.loads((artifact_root / "plus45_as_of_feature_gap_shadow_exploration_go_no_go_v1.json").read_text())
    assert go["status_v1"] in gate.ALLOWED_FINAL_SHADOW_STATUSES
    assert go["shadow_next_recommended_action_v1"] in gate.ALLOWED_SHADOW_NEXT_ACTIONS
    assert go["mainline_next_action_v1"] == gate.MAINLINE_NEXT_ACTION
    assert go["r6_run_v1"] is False
    assert go["adapter_built_v1"] is False
    assert go["plus45_role_v1"] == "DIAGNOSTIC_ONLY_NOT_TARGET_FEATURE_FILTER_OR_THRESHOLD_OBJECTIVE"
    cohort = json.loads((artifact_root / "plus45_shadow_cohort_reconstruction_v1.json").read_text())
    assert cohort["baseline_140_94_selected_rows_v1"] == 140
    assert cohort["best_lane_185_139_selected_rows_v1"] == 185
    assert cohort["plus45_rows_v1"] == 45
