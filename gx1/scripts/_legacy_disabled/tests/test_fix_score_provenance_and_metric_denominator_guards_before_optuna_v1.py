from __future__ import annotations

import numpy as np
import pandas as pd

import gx1.scripts.materialize_constrained_optuna_objective_search_and_full_signal_forensics_v1 as optuna_forensics
import gx1.scripts.materialize_fix_score_provenance_and_metric_denominator_guards_before_optuna_v1 as fix
import gx1.scripts.run_r5_2_objective_v3_parallel_rebuild_runner_v1 as v3_runner


def _prediction(rows: int = 3) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "candidate_uid": [f"candidate_{idx}" for idx in range(rows)],
            "trade_uid": [f"trade_{idx}" for idx in range(rows)],
            "decision_timestamp": [f"2026-01-01T00:0{idx}:00Z" for idx in range(rows)],
        }
    )
    for field in fix.SCORE_FIELDS:
        frame[field] = 0.1
    return frame


def _valid_provenance(rows: int = 3) -> pd.DataFrame:
    pred = _prediction(rows)
    out = []
    for field in fix.SCORE_FIELDS:
        for idx, row in pred.iterrows():
            out.append(
                {
                    "candidate_uid": row["candidate_uid"],
                    "trade_uid": row["trade_uid"],
                    "decision_timestamp": row["decision_timestamp"],
                    "variant_id_v1": "variant_01",
                    "score_field_v1": field,
                    "fold_id_v1": idx + 1,
                    "group_key_v1": f"group_{idx}",
                    "train_validation_membership_v1": "VALIDATION",
                    "source_model_fold_v1": f"variant_01:{field}:fold_{idx + 1}",
                    "score_source_v1": "OOF",
                    "row_was_in_training_for_source_model_v1": False,
                    "in_sample_score_used_v1": False,
                    "fallback_score_used_v1": False,
                    "synthetic_score_used_v1": False,
                }
            )
    return pd.DataFrame(out)


def test_complete_valid_oof_provenance_passes() -> None:
    result = fix.validate_oof_score_provenance(_valid_provenance(), _prediction(), expected_rows=3)
    assert result["status_v1"] == "PASS"


def test_missing_oof_provenance_hard_fails() -> None:
    result = fix.validate_oof_score_provenance(pd.DataFrame(), _prediction(), expected_rows=3)
    assert result["status_v1"] == "FAIL_MISSING_PROVENANCE"


def test_in_sample_score_disguised_as_oof_hard_fails() -> None:
    provenance = _valid_provenance()
    provenance.loc[0, "in_sample_score_used_v1"] = True
    result = fix.validate_oof_score_provenance(provenance, _prediction(), expected_rows=3)
    assert result["status_v1"] == "FAIL_IN_SAMPLE_SCORE_USED"


def test_fold_id_missing_hard_fails() -> None:
    provenance = _valid_provenance()
    provenance.loc[0, "fold_id_v1"] = np.nan
    result = fix.validate_oof_score_provenance(provenance, _prediction(), expected_rows=3)
    assert result["status_v1"] == "FAIL_MISSING_PROVENANCE"


def test_training_leakage_hard_fails() -> None:
    provenance = _valid_provenance()
    provenance.loc[0, "row_was_in_training_for_source_model_v1"] = True
    result = fix.validate_oof_score_provenance(provenance, _prediction(), expected_rows=3)
    assert result["status_v1"] == "FAIL_TRAIN_VALIDATION_LEAKAGE"


def test_synthetic_or_fallback_provenance_hard_fails() -> None:
    provenance = _valid_provenance()
    provenance.loc[0, "synthetic_score_used_v1"] = True
    result = fix.validate_oof_score_provenance(provenance, _prediction(), expected_rows=3)
    assert result["status_v1"] == "FAIL_SYNTHETIC_SCORE_USED"
    provenance = _valid_provenance()
    provenance.loc[0, "fallback_score_used_v1"] = True
    result = fix.validate_oof_score_provenance(provenance, _prediction(), expected_rows=3)
    assert result["status_v1"] == "FAIL_FALLBACK_SCORE_USED"


def test_precision_empty_denominator_is_decision_invalid() -> None:
    metric = v3_runner._metric_ratio("precision", 0, 0, min_denominator=5)
    assert metric["precision_denominator_status_v1"] == "EMPTY_DENOMINATOR"
    assert metric["precision_decision_valid_v1"] is False
    assert pd.isna(metric["precision_v1"])


def test_precision_small_denominator_gets_status() -> None:
    metric = v3_runner._metric_ratio("precision", 1, 2, min_denominator=5)
    assert metric["precision_denominator_status_v1"] == "TOO_SMALL_DENOMINATOR"
    assert metric["precision_decision_valid_v1"] is False
    assert metric["precision_v1"] == 0.5


def test_valid_denominator_gives_normal_metric() -> None:
    metric = v3_runner._metric_ratio("precision", 4, 5, min_denominator=5)
    assert metric["precision_denominator_status_v1"] == "OK"
    assert metric["precision_decision_valid_v1"] is True
    assert metric["precision_v1"] == 0.8


def test_worst_loso_empty_selected_groups_is_not_one_point_zero_pass() -> None:
    frame = pd.DataFrame({"run_id": ["a", "b"], "bad": [True, False]})
    selected = pd.Series([False, False])
    result = v3_runner._worst_group_precision(frame, selected, frame["bad"], "run_id")
    assert result["worst_loso_denominator_status_v1"] == "EMPTY_DENOMINATOR"
    assert result["worst_loso_decision_valid_v1"] is False
    assert pd.isna(result["worst_loso_v1"])


def test_worst_loso_too_small_denominator_is_decision_invalid() -> None:
    frame = pd.DataFrame({"run_id": ["a", "b", "b", "b", "b", "b"], "bad": [True, True, True, True, True, True]})
    selected = pd.Series([True, False, False, False, False, False])
    result = v3_runner._worst_group_precision(frame, selected, frame["bad"], "run_id")
    assert result["worst_loso_denominator_status_v1"] == "TOO_SMALL_DENOMINATOR"
    assert result["worst_loso_decision_valid_v1"] is False


def test_optuna_candidate_metrics_stop_when_denominator_invalid() -> None:
    ledger = pd.DataFrame(
        {
            "candidate_uid": ["c1", "c2"],
            "bad_label_v1": [True, False],
            "tail_label_v1": [True, False],
            "split_loso_group_v1": ["g1", "g2"],
            "batch_v1": ["b1", "b1"],
            "repaired_flag_v1": [False, False],
            "high_mfe_flag_v1": [False, False],
            "dangerous_or_protected_v1": [False, False],
            "runner_flag_v1": [False, False],
        }
    )
    for column in [
        "r5_2_v3_oof_bad_score_v1",
        "r5_2_v2_bad_score_v1",
        "r5_bad_score_v1",
        "r5_1_bad_score_v1",
        "r5_2_v3_oof_tail_score_v1",
        "r5_2_v2_tail_score_v1",
        "r5_tail_score_v1",
        "r5_2_v3_oof_runner_protect_score_v1",
        "r5_2_v2_runner_protect_score_v1",
        "r5_runner_score_v1",
        "r5_1_runner_score_v1",
    ]:
        ledger[column] = 0.0
    params = {
        "w_v3_bad": 1.0,
        "w_v2_bad": 1.0,
        "w_r5_bad": 1.0,
        "w_r51_bad": 1.0,
        "w_v3_tail": 1.0,
        "w_v2_tail": 1.0,
        "w_r5_tail": 1.0,
        "w_v3_runner": 1.0,
        "w_v2_runner": 1.0,
        "w_r5_runner": 1.0,
        "w_r51_runner": 1.0,
        "bad_threshold": 10.0,
        "tail_threshold": 10.0,
        "risky_threshold": 10.0,
        "confirm_threshold": 10.0,
        "protection_threshold": 1.0,
        "exclude_all_50_plus": True,
    }
    metrics, selected = optuna_forensics._candidate_rule_metrics(ledger, params)
    assert not selected.any()
    assert metrics["precision_decision_valid_v1"] is False
    assert metrics["worst_loso_decision_valid_v1"] is False
    assert "METRIC_DENOMINATOR_INVALID" in metrics["fail_reason_v1"]
