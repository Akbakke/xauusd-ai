from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_build_model_to_learn_best_lane_membership_as_oof_target_v1 as gate


def test_teacher_target_freeze_requires_185_140_and_45() -> None:
    frame = pd.DataFrame(
        {
            "teacher_membership_v1": [True] * 185 + [False],
            "is_baseline_140_v1": [True] * 140 + [False] * 46,
            "is_added_45_v1": [False] * 140 + [True] * 45 + [False],
        }
    )
    assert gate.validate_teacher_target_freeze(frame) is True
    frame.loc[184, "teacher_membership_v1"] = False
    with pytest.raises(RuntimeError, match="TEACHER_TARGET_FREEZE_MISMATCH"):
        gate.validate_teacher_target_freeze(frame)


def test_teacher_target_freeze_rejects_added_baseline_overlap() -> None:
    added = [False] * 185
    added[139] = True
    for idx in range(141, 185):
        added[idx] = True
    frame = pd.DataFrame(
        {
            "teacher_membership_v1": [True] * 185,
            "is_baseline_140_v1": [True] * 140 + [False] * 45,
            "is_added_45_v1": added,
        }
    )
    with pytest.raises(RuntimeError, match="TEACHER_TARGET_ADDED_BASELINE_OVERLAP"):
        gate.validate_teacher_target_freeze(frame)


def test_denylist_blocks_label_mfe_safe_recoverable_membership_and_selected_features() -> None:
    bad_features = [
        "bad_label_v1",
        "tail_label_v1",
        "tail_10_50_mfe_v1",
        "safe_recoverable_v1",
        "coverage_proxy_member_v1",
        "lane_selected_v1",
        "selected_by_v2_v1",
        "protected_winner_status_v1",
    ]
    with pytest.raises(RuntimeError, match="TARGET_OR_OUTCOME_FEATURE_FORBIDDEN"):
        gate.validate_no_target_or_outcome_feature(bad_features)


def test_allowlisted_features_do_not_match_forbidden_patterns() -> None:
    used = gate.RAW_ALLOWED_FEATURES + list(gate.DERIVED_SIGNAL_FEATURES)
    assert gate.validate_no_target_or_outcome_feature(used) is True


def test_used_feature_must_pass_leakage_audit() -> None:
    rows = [
        {
            "feature_name_v1": "asof_signal__r5_tail_score_v1",
            "allowed_blocked_v1": "ALLOWED",
            "suspected_leakage_class_v1": "",
        }
    ]
    assert gate.validate_feature_policy(["asof_signal__r5_tail_score_v1"], rows) is True
    rows[0]["allowed_blocked_v1"] = "BLOCKED"
    rows[0]["suspected_leakage_class_v1"] = "OUTCOME_LABEL"
    with pytest.raises(RuntimeError, match="USED_FEATURE_FAILED_LEAKAGE_AUDIT"):
        gate.validate_feature_policy(["asof_signal__r5_tail_score_v1"], rows)


def test_oof_split_requires_no_train_validation_overlap() -> None:
    assert gate.validate_oof_split([1, 2, 3], [4, 5]) is True
    with pytest.raises(RuntimeError, match="OOF_TRAIN_VALIDATION_OVERLAP"):
        gate.validate_oof_split([1, 2, 3], [3, 4])


def test_threshold_selection_uses_inner_validation_scores_only() -> None:
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    y = np.array([0, 0, 1, 1])
    threshold, rows = gate.choose_threshold_from_inner_validation(scores, y)
    assert threshold in gate.THRESHOLD_GRID
    assert rows
    assert all("objective_v1" in row for row in rows)


def test_final_status_blocks_feature_leakage() -> None:
    status, next_action = gate.final_status_from_metrics(
        {"teacher_recall_v1": 1.0, "added_row_recall_v1": 1.0},
        leakage_clean=False,
        unsafe_selected=0,
    )
    assert status == "BLOCKED_BY_FEATURE_LEAKAGE_OR_TARGET_CONTAMINATION"
    assert next_action == "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1"


def test_final_status_flags_unsafe_lookalikes_before_adapter() -> None:
    status, next_action = gate.final_status_from_metrics(
        {"teacher_recall_v1": 1.0, "added_row_recall_v1": 1.0},
        leakage_clean=True,
        unsafe_selected=1,
    )
    assert status == "BEST_LANE_SIGNAL_REAL_BUT_STUDENT_OVERSELECTS_UNSAFE_LOOKALIKES"
    assert next_action == "DEEPEN_STUDENT_NEAR_MISS_UNSAFE_LOOKALIKE_AUDIT_V1"


def test_final_status_requires_added_row_recovery_for_adapter_feasible() -> None:
    status, next_action = gate.final_status_from_metrics(
        {"teacher_recall_v1": 0.71, "added_row_recall_v1": 0.0},
        leakage_clean=True,
        unsafe_selected=0,
    )
    assert status == "BEST_LANE_MEMBERSHIP_NOT_LEARNABLE_FROM_AS_OF_FEATURES"
    assert next_action == "REJECT_OR_REBUILD_BEST_LANE_FROM_CAUSAL_SIGNALS_V1"


def test_no_forbidden_actions_for_student_gate() -> None:
    clean = gate.validate_no_forbidden_actions(
        optuna=False,
        r6=False,
        adapter=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
    )
    blocked = gate.validate_no_forbidden_actions(
        optuna=True,
        r6=True,
        adapter=True,
        package=True,
        freeze=True,
        promo=True,
        live=True,
    )
    assert clean["status_v1"] == "PASS"
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]
    assert "ADAPTER_BUILD_FORBIDDEN" in blocked["failures_v1"]


def test_no_implicit_latest_or_glob_artifact_selection() -> None:
    assert gate.validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB") is True
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_selection("LATEST")
