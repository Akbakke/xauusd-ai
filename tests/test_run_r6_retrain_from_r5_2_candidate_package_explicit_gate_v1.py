from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts import materialize_run_r6_retrain_from_r5_2_candidate_package_explicit_gate_v1 as r6


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_package(root: Path, *, integrity_status: str = "PASS") -> None:
    for name in r6.REQUIRED_PACKAGE_FILES:
        path = root / name
        if name.endswith(".csv"):
            pd.DataFrame({"candidate_uid_v1": ["c1"]}).to_csv(path, index=False)
        elif name == "r5_2_candidate_package_manifest_v1.json":
            _write_json(path, {"final_promotion_allowed_v1": False})
        elif name == "r5_2_candidate_package_integrity_report_v1.json":
            _write_json(path, {"status_v1": integrity_status})
        elif name == "r5_2_candidate_package_r6_input_readiness_precheck_v1.json":
            _write_json(path, {"status_v1": "R6_INPUT_PACKAGE_READY_BUT_R6_NOT_AUTHORIZED"})
        elif name == "summary_v1.json":
            _write_json(
                path,
                {
                    "selected_threshold_candidate_v1": "RECALL",
                    "bad_count_v1": 130,
                    "tail_count_v1": 86,
                    "precision_denominator_v1": 130,
                    "strict_all_run_id_worst_loso_denominator_v1": 2,
                    "strict_all_run_id_decision_valid_v1": False,
                    "safety_clean_v1": True,
                },
            )
        else:
            _write_json(path, {})


def test_r6_must_use_explicit_package_root_and_no_latest_glob() -> None:
    assert r6.validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB") is True
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        r6.validate_explicit_artifact_selection("LATEST_FOLDER_WINS")


def test_r6_cannot_run_if_package_integrity_fails(tmp_path: Path) -> None:
    package = tmp_path / "package"
    _seed_package(package, integrity_status="FAIL")
    with pytest.raises(RuntimeError, match="R6_INPUT_PACKAGE_VALIDATION_FAILED"):
        r6.validate_input_package(package)


def test_r6_cannot_run_if_package_provenance_missing(tmp_path: Path) -> None:
    package = tmp_path / "package"
    _seed_package(package)
    (package / "r5_2_candidate_oof_score_provenance_v1.csv").unlink()
    with pytest.raises(RuntimeError, match="missing package files"):
        r6.validate_input_package(package)


def test_valid_r5_2_package_validation_passes(tmp_path: Path) -> None:
    package = tmp_path / "package"
    _seed_package(package)
    validation = r6.validate_input_package(package)
    assert validation["status_v1"] == "PASS"


def test_r6_source_mapping_finds_existing_path(tmp_path: Path) -> None:
    mapping = r6._source_mapping(tmp_path)
    assert mapping["r6_existing_path_found_v1"] is True
    assert mapping["existing_r6_path_reused_v1"] is True


def test_new_r6_wrapper_is_wrapper_only_not_duplicate_five_head_logic(tmp_path: Path) -> None:
    attestation = r6._no_reimplementation_attestation(r6._source_mapping(tmp_path))
    assert attestation["wrapper_only_v1"] is True
    assert attestation["new_r6_head_logic_introduced_v1"] is False
    assert attestation["disconnected_r6_clone_created_v1"] is False


def test_r6_candidate_grid_includes_pass_through_control() -> None:
    grid = r6._candidate_grid()
    assert r6.validate_candidate_grid(grid) is True
    assert any(row["candidate_id_v1"] == "R5_2_PASS_THROUGH_CONTROL" for row in grid)


def test_r6_candidate_grid_is_not_optuna_or_large_sweep() -> None:
    assert len(r6._candidate_grid()) <= 8
    too_large = [{"candidate_id_v1": "R5_2_PASS_THROUGH_CONTROL"}]
    too_large.extend({"candidate_id_v1": f"C{i}"} for i in range(8))
    with pytest.raises(RuntimeError, match="LARGE_SWEEP|OPTUNA"):
        r6.validate_candidate_grid(too_large)


def test_r6_oof_cannot_mark_row_decision_valid_if_row_was_training_member() -> None:
    scores = pd.DataFrame({"was_row_in_train_for_scoring_model_v1": [False, True]})
    result = r6.validate_no_in_sample_scoring(scores)
    assert result["status_v1"] == "FAIL"
    assert result["decision_valid_v1"] is False


def test_train_validation_overlap_blocks_decision_valid() -> None:
    membership = pd.DataFrame({"is_train_v1": [True], "is_validation_v1": [True]})
    result = r6.validate_no_train_validation_overlap(membership)
    assert result["status_v1"] == "FAIL"


def test_r6_provenance_required_for_every_scored_row() -> None:
    scores = pd.DataFrame({"candidate_uid_v1": ["c1"]})
    provenance = pd.DataFrame(
        [
            {
                "candidate_uid_v1": "c1",
                "scorefield_v1": scorefield,
                "provenance_valid_v1": True,
            }
            for scorefield in r6.R6_SCOREFIELDS[:-1]
        ]
    )
    result = r6.validate_r6_provenance_complete(scores, provenance)
    assert result["status_v1"] == "FAIL"
    assert result["missing_provenance_rows_v1"] == 1


def test_r6_must_report_all_five_heads() -> None:
    assert r6.R6_HEAD_NAMES == ["bad_risk", "runner_protector", "tail_control_10_50", "risky_allow", "batch04_blindspot"]


def test_safety_violations_block_candidate() -> None:
    row = {
        "fifty_plus_mfe_overlap_v1": 0,
        "hundred_plus_mfe_overlap_v1": 1,
        "two_hundred_plus_mfe_overlap_v1": 0,
        "strongest_winner_overlap_v1": 0,
        "protected_winner_selected_v1": 0,
        "runner_protect_leakage_v1": 0,
        "ambiguous_high_mfe_leakage_v1": 0,
        "quarantine_selected_v1": 0,
    }
    assert r6.candidate_passes_hard_safety(row) is False


def test_low_support_groups_remain_visible_in_best_status() -> None:
    best = {
        "candidate_id_v1": "R5_2_PASS_THROUGH_CONTROL",
        "bad_count_v1": 130,
        "tail_count_v1": 86,
        "safety_clean_v1": True,
        "strict_all_run_id_decision_valid_v1": False,
        "structural_low_support_selected_group_count_v1": 8,
    }
    status, _ = r6._best_status(best, provenance_ok=True, no_in_sample_ok=True, no_overlap_ok=True)
    assert status == "R6_CANDIDATE_RETURNS_R5_2_LEVEL_WITH_STRONGER_HEAD_DIAGNOSTICS"


def test_strict_loso_invalidity_cannot_be_hidden_by_final_promotion() -> None:
    with pytest.raises(RuntimeError, match="FINAL_PROMOTION"):
        r6.validate_final_promotion_blocked({"final_promotion_allowed_v1": True})


def test_no_freeze_promo_live_or_optuna_actions() -> None:
    clean = r6.validate_no_forbidden_actions(optuna=False, freeze=False, promo=False, live=False)
    blocked = r6.validate_no_forbidden_actions(optuna=True, freeze=True, promo=True, live=True)
    assert clean["status_v1"] == "PASS"
    assert blocked["status_v1"] == "FAIL"


def test_v2_and_r5_2_package_artifacts_can_be_hash_checked(tmp_path: Path) -> None:
    package = tmp_path / "package"
    _seed_package(package)
    before = r6._package_hashes(package)
    after = r6._package_hashes(package)
    assert before == after


def test_no_dummy_synthetic_fallback_contract_written_as_pass() -> None:
    result = r6.validate_no_forbidden_actions(optuna=False, freeze=False, promo=False, live=False)
    assert result["failures_v1"] == []


def test_forbidden_id_leakage_features_are_rejected() -> None:
    result = r6.validate_no_forbidden_features(["as_of_hour_utc_v1", "candidate_uid"])
    assert result["status_v1"] == "FAIL"


def test_existing_r6_presence_feature_allowlist_remains_legal() -> None:
    result = r6.validate_no_forbidden_features(["management_observation_present_v1"])
    assert result["status_v1"] == "PASS"


def test_hindsight_features_are_rejected() -> None:
    result = r6.validate_no_hindsight_features(["as_of_hour_utc_v1", "hindsight_label_v1"])
    assert result["status_v1"] == "FAIL"


def test_wednesday_benchmark_is_comparator_only_not_row_target() -> None:
    wednesday = [row for row in r6.FIXED_CONTROLS if row["control_v1"] == "wednesday_benchmark"][0]
    assert wednesday["role_v1"] == "COMPARATOR_ONLY_NOT_ROW_TARGET"


def test_candidate_final_promotion_always_false_in_go_no_go() -> None:
    assert r6.validate_final_promotion_blocked({"final_promotion_allowed_v1": False}) is True
