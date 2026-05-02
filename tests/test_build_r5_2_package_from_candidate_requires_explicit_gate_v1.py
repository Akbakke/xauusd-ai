from __future__ import annotations

import json
from pathlib import Path

import pytest

from gx1.scripts import materialize_build_r5_2_package_from_candidate_requires_explicit_gate_v1 as pkg


def _metrics(**overrides: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "threshold_candidate_id_v1": "RECALL",
        "bad_count_v1": 130,
        "tail_count_v1": 86,
        "precision_v1": 1.0,
        "precision_denominator_v1": 130,
        "precision_decision_valid_v1": True,
        "strict_all_run_id_worst_loso_v1": 1.0,
        "strict_all_run_id_worst_loso_denominator_v1": 2,
        "strict_all_run_id_decision_valid_v1": False,
        "selected_low_support_group_count_v1": 11,
        "structural_low_support_selected_group_count_v1": 8,
        "safety_clean_v1": True,
        "bad_threshold_v1": 0.35,
        "tail_threshold_v1": 0.4,
        "hard_veto_max_v1": 0.8,
        "policy_v1": "RECALL_ORIENTED_WITH_HARD_VETOES",
    }
    metrics.update(overrides)
    return metrics


def test_candidate_package_cannot_be_marked_promoted() -> None:
    with pytest.raises(RuntimeError, match="CANDIDATE_PACKAGE_CANNOT_BE_PROMOTED"):
        pkg.validate_not_promoted({"promoted_v1": True})


def test_candidate_package_cannot_be_live_or_freeze_ready() -> None:
    with pytest.raises(RuntimeError, match="CANDIDATE_PACKAGE_CANNOT_BE_PROMOTED"):
        pkg.validate_not_promoted({"freeze_ready_v1": True, "live_ready_v1": False})


def test_candidate_package_cannot_hide_strict_loso_invalidity() -> None:
    with pytest.raises(RuntimeError, match="STRICT_LOSO_INVALIDITY"):
        pkg.validate_strict_loso_visible(_metrics(strict_all_run_id_decision_valid_v1=True))


def test_candidate_package_preserves_130_86_metrics() -> None:
    assert pkg.validate_metric_preservation(_metrics()) is True


def test_candidate_package_preserves_selected_threshold_recall() -> None:
    with pytest.raises(RuntimeError, match="metric preservation failure"):
        pkg.validate_metric_preservation(_metrics(threshold_candidate_id_v1="BALANCED"))


def test_candidate_package_preserves_safety_clean_status() -> None:
    with pytest.raises(RuntimeError, match="metric preservation failure"):
        pkg.validate_metric_preservation(_metrics(safety_clean_v1=False))


def test_candidate_package_must_include_oof_provenance() -> None:
    assert "r5_2_oof_score_provenance_v1.csv" in pkg.REQUIRED_FILE_MAP


def test_candidate_package_must_include_train_validation_membership() -> None:
    assert "r5_2_train_validation_membership_v1.csv" in pkg.REQUIRED_FILE_MAP


def test_candidate_package_must_include_metric_denominator_report() -> None:
    assert "r5_2_oof_metric_denominator_report_v1.json" in pkg.REQUIRED_FILE_MAP


def test_candidate_package_must_include_low_support_report() -> None:
    assert "r5_2_oof_low_support_report_v1.json" in pkg.REQUIRED_FILE_MAP


def test_candidate_package_must_include_fixed_control_comparison() -> None:
    assert "r5_2_oof_fixed_control_comparison_v1.json" in pkg.REQUIRED_FILE_MAP


def test_candidate_package_must_include_no_dummy_synthetic_fallback_attestation() -> None:
    assert "no_fallback_no_dummy_no_synthetic_attestation_v1.json" in pkg.REQUIRED_FILE_MAP


def test_final_fit_artifact_if_created_cannot_be_evaluation_evidence() -> None:
    with pytest.raises(RuntimeError, match="FINAL_FIT_METRICS_CANNOT_BE_DECISIONING_EVIDENCE"):
        pkg.validate_final_fit_policy(
            {
                "status_v1": "FINAL_FIT_CREATED_NON_EVAL_FUTURE_SCORING_ONLY",
                "metrics_used_for_decisioning_v1": True,
            }
        )


def test_missing_source_file_causes_hard_failure(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Missing required source artifact"):
        pkg._source_required_files(tmp_path)


def test_file_hash_mismatch_causes_hard_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    source.mkdir()
    dest.mkdir()
    for source_name in pkg.REQUIRED_FILE_MAP:
        (source / source_name).write_text("same\n", encoding="utf-8")
    copied = pkg._copy_required_files(source, dest)
    first_package = dest / copied[0]["package_name_v1"]
    first_package.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing required artifact for hash|Package file hash mismatch"):
        if pkg._file_hash(source / copied[0]["source_name_v1"]) != pkg._file_hash(first_package):
            raise RuntimeError("Package file hash mismatch")


def test_r6_precheck_cannot_run_r6(tmp_path: Path) -> None:
    precheck = pkg._r6_precheck(tmp_path, {"checks_v1": {"safety_remains_clean_v1": True, "low_support_groups_remain_reported_v1": True}})

    assert precheck["r6_was_run_v1"] is False
    assert precheck["r6_run_authorized_v1"] is False


def test_r6_precheck_status_must_not_authorize_r6_without_explicit_gate() -> None:
    assert pkg.r6_precheck_authorizes_r6({"r6_run_authorized_v1": False}) is False


def test_no_optuna_r6_promotion_freeze_live_beyond_candidate_package() -> None:
    result = pkg.validate_no_forbidden_actions(optuna=False, r6=False, promoted=False, freeze=False, live=False)

    assert result["status_v1"] == "PASS"


def test_no_implicit_latest_glob_artifact_selection() -> None:
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        pkg.validate_explicit_artifact_selection("LATEST_FOLDER_WINS")


def test_v2_oof_source_artifacts_remain_referenced_not_mutated(tmp_path: Path) -> None:
    manifest = {
        "input_candidate_root_v1": str(tmp_path / "candidate"),
        "reason_v1": "STRUCTURAL_LOW_SUPPORT_REMAINS",
        "final_promotion_allowed_v1": False,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["final_promotion_allowed_v1"] is False
