from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts import run_r5_2_objective_v2_replay_with_oof_provenance_v1 as replay


def _score_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_uid": "c1",
                "trade_uid": "t1",
                "decision_timestamp": "2026-04-20T00:00:00Z",
                "trade_id": "trade-1",
                "run_id": "run-a",
                "was_row_in_train_for_scoring_model_v1": False,
            },
            {
                "candidate_uid": "c2",
                "trade_uid": "t2",
                "decision_timestamp": "2026-04-20T00:01:00Z",
                "trade_id": "trade-2",
                "run_id": "run-b",
                "was_row_in_train_for_scoring_model_v1": False,
            },
        ]
    )


def test_v2_replay_cannot_mark_row_decision_valid_when_scored_row_was_in_training_membership() -> None:
    scores = _score_rows()
    scores.loc[0, "was_row_in_train_for_scoring_model_v1"] = True

    result = replay.validate_no_in_sample_scoring(scores)

    assert result["status_v1"] == "FAIL"
    assert result["decision_valid_v1"] is False
    assert result["in_sample_scored_count_v1"] == 1


def test_v2_replay_writes_grouped_fold_assignment_contract() -> None:
    assignment = replay._fold_assignment(_score_rows(), fold_count=2)

    assert {"fold_id_v1", "group_key_v1", "split_policy_v1"}.issubset(assignment.columns)
    assert set(assignment["split_policy_v1"]) == {"DETERMINISTIC_BALANCED_GROUPED_OOF_BY_RUN_ID"}
    assert assignment.groupby("run_id")["fold_id_v1"].nunique().max() == 1


def test_v2_replay_writes_train_validation_membership_without_overlap() -> None:
    membership = pd.DataFrame(
        [
            {"candidate_uid_v1": "c1", "fold_id_v1": "fold_00", "is_train_v1": True, "is_validation_v1": False},
            {"candidate_uid_v1": "c2", "fold_id_v1": "fold_00", "is_train_v1": False, "is_validation_v1": True},
        ]
    )

    result = replay.validate_no_train_validation_overlap(membership)

    assert result["status_v1"] == "PASS"
    assert result["overlap_count_v1"] == 0


def test_v2_replay_writes_required_provenance_files(tmp_path: Path) -> None:
    for name in [
        "v2_oof_scores_v1.csv",
        "v2_oof_score_provenance_v1.csv",
        "v2_oof_fold_assignment_v1.csv",
        "v2_train_validation_membership_v1.csv",
    ]:
        (tmp_path / name).write_text("candidate_uid_v1\nc1\n", encoding="utf-8")
    (tmp_path / "v2_oof_score_source_manifest_v1.json").write_text(json.dumps({"ok": True}), encoding="utf-8")

    result = replay.validate_provenance_files(tmp_path)

    assert result["status_v1"] == "PASS"
    assert result["missing_files_v1"] == []


def test_v2_replay_fails_if_provenance_is_missing(tmp_path: Path) -> None:
    result = replay.validate_provenance_files(tmp_path)

    assert result["status_v1"] == "FAIL"
    assert "v2_oof_score_provenance_v1.csv" in result["missing_files_v1"]


def test_v2_replay_fails_if_train_validation_overlap_exists() -> None:
    membership = pd.DataFrame(
        [{"candidate_uid_v1": "c1", "fold_id_v1": "fold_00", "is_train_v1": True, "is_validation_v1": True}]
    )

    result = replay.validate_no_train_validation_overlap(membership)

    assert result["status_v1"] == "FAIL"
    assert result["decision_valid_v1"] is False


def test_v2_replay_rejects_full_sample_model_artifact_as_oof_without_proof() -> None:
    status = replay.classify_model_artifact_use(
        existing_artifact_fold_trained=False,
        existing_artifact_validation_only=False,
    )

    assert status == "EXISTING_V2_MODEL_ARTIFACTS_ARE_HISTORICAL_ONLY_FOR_OOF"


def test_v2_replay_can_describe_historical_only_artifact_without_decision_valid() -> None:
    assert replay.historical_v2_decision_status(has_oof_proof=False) == "HISTORICAL_V2_CAN_BE_COMPARATOR_ONLY"


def test_v2_oof_metric_denominator_invalid_blocks_decision_valid() -> None:
    metric = replay._metric_ratio("precision", numerator=0, denominator=0)

    assert metric["precision_decision_valid_v1"] is False
    assert metric["precision_denominator_status_v1"] == "EMPTY_DENOMINATOR"


def test_precision_and_worst_loso_denominators_are_tracked_separately() -> None:
    frame = pd.DataFrame(
        [
            {"run_id": "a"},
            {"run_id": "a"},
            {"run_id": "b"},
            {"run_id": "b"},
            {"run_id": "b"},
        ]
    )
    selected = pd.Series([True, False, True, True, True])
    bad = pd.Series([False, False, True, True, True])
    precision = replay._metric_ratio("precision", numerator=3, denominator=4)
    _, loso = replay._worst_loso(frame, selected, bad)

    assert precision["precision_denominator_v1"] == 4
    assert precision["precision_decision_valid_v1"] is False
    assert loso["worst_loso_denominator_v1"] == 1
    assert loso["worst_loso_decision_valid_v1"] is False


def test_no_dummy_synthetic_or_degraded_fallback_allowed() -> None:
    result = replay.validate_no_dummy_synthetic_fallback(dummy=True, synthetic=False, fallback=True)

    assert result["status_v1"] == "FAIL"
    assert result["decision_valid_v1"] is False
    assert set(result["failures_v1"]) == {"DUMMY_INPUT_FORBIDDEN", "DEGRADED_FALLBACK_FORBIDDEN"}


def test_no_implicit_latest_or_glob_artifact_selection() -> None:
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB_SELECTION_FORBIDDEN"):
        replay.validate_explicit_selection_policy("LATEST_FOLDER_WINS")


def test_old_invalid_v3_cannot_be_selected() -> None:
    status = replay.selected_v3_artifact_status(
        selected_for_decisioning=True,
        decision_valid_status="INVALID_FOR_OPTUNA_DECISIONING",
    )

    assert status == "BLOCK_SELECTED_INVALID_V3"


def test_historical_v2_95_61_cannot_be_decision_valid_without_oof_proof() -> None:
    assert replay.historical_v2_decision_status(has_oof_proof=False) == "HISTORICAL_V2_CAN_BE_COMPARATOR_ONLY"


def test_oof_replay_summary_compares_historical_v2_optuna_best_and_v3() -> None:
    historical, optuna_v3, rows = replay._delta_reports(
        {
            "bad_count_v1": 40,
            "tail_count_v1": 20,
            "row_overlap_with_optuna_best_v1": 12,
            "row_overlap_with_v3_v1": 4,
            "status_v1": "V2_OOF_REPLAY_DECISION_VALID_BUT_WEAK",
        }
    )

    assert historical["historical_v2_bad_v1"] == 95
    assert optuna_v3["optuna_best_bad_v1"] == 56
    assert optuna_v3["v3_best_bad_v1"] == 17
    assert {row["comparator_v1"] for row in rows} == {"historical_v2", "optuna_best", "v3_best", "v2_oof_replay"}


def test_historical_invalid_v3_artifact_not_selected_is_not_blocker() -> None:
    status = replay.selected_v3_artifact_status(
        selected_for_decisioning=False,
        decision_valid_status="INVALID_FOR_OPTUNA_DECISIONING",
    )

    assert status == "HISTORY_ONLY_NOT_BLOCKER"
