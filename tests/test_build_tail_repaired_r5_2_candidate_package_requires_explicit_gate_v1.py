from __future__ import annotations

from pathlib import Path

import pytest

from gx1.scripts import materialize_build_tail_repaired_r5_2_candidate_package_requires_explicit_gate_v1 as pkg


def _metrics(**overrides: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "candidate_id_v1": "TAIL_TARGET_WEIGHT_REPAIR_CONSERVATIVE::RECALL",
        "threshold_candidate_id_v1": "RECALL",
        "bad_count_v1": 140,
        "tail_count_v1": 94,
        "precision_v1": 1.0,
        "precision_denominator_v1": 140,
        "precision_decision_valid_v1": True,
        "strict_all_run_id_worst_loso_v1": 1.0,
        "strict_all_run_id_worst_loso_denominator_v1": 2,
        "strict_all_run_id_decision_valid_v1": False,
        "selected_low_support_group_count_v1": 10,
        "structural_low_support_selected_group_count_v1": 7,
        "safety_clean_v1": True,
        "final_promotion_allowed_v1": False,
        "bad_threshold_v1": 0.35,
        "tail_threshold_v1": 0.4,
        "hard_veto_max_v1": 0.8,
        "policy_v1": "RECALL_ORIENTED_WITH_HARD_VETOES",
    }
    metrics.update(overrides)
    return metrics


def test_tail_repaired_candidate_package_cannot_be_marked_promoted() -> None:
    with pytest.raises(RuntimeError, match="CANNOT_BE_PROMOTED"):
        pkg.validate_not_promoted({"promoted_v1": True})


def test_tail_repaired_candidate_package_cannot_be_live_or_freeze_ready() -> None:
    with pytest.raises(RuntimeError, match="CANNOT_BE_PROMOTED"):
        pkg.validate_not_promoted({"freeze_ready_v1": True, "live_ready_v1": True})


def test_candidate_package_cannot_hide_strict_loso_invalidity() -> None:
    with pytest.raises(RuntimeError, match="STRICT_LOSO_INVALIDITY"):
        pkg.validate_strict_loso_visible(_metrics(strict_all_run_id_decision_valid_v1=True))


def test_candidate_package_preserves_140_94_oof_metrics() -> None:
    assert pkg.validate_metric_preservation(_metrics()) is True
    with pytest.raises(RuntimeError, match="metric preservation failure"):
        pkg.validate_metric_preservation(_metrics(bad_count_v1=139))


def test_candidate_package_preserves_selected_tail_repair_candidate() -> None:
    with pytest.raises(RuntimeError, match="metric preservation failure"):
        pkg.validate_metric_preservation(_metrics(candidate_id_v1="TAIL_TARGET_WEIGHT_REPAIR_BALANCED::RECALL"))


def test_candidate_package_preserves_safety_clean_status() -> None:
    with pytest.raises(RuntimeError, match="metric preservation failure"):
        pkg.validate_metric_preservation(_metrics(safety_clean_v1=False))


def test_candidate_package_must_include_required_provenance_and_reports() -> None:
    required_sources = set(pkg.COPY_FILE_MAP)
    assert "tail_repair_oof_score_provenance_v1.csv" in required_sources
    assert "tail_repair_train_validation_membership_v1.csv" in required_sources
    assert "tail_repair_metric_denominator_report_v1.json" in required_sources
    assert "tail_repair_low_support_report_v1.json" in required_sources
    assert "tail_repair_candidate_registry_v1.json" in required_sources
    assert "tail_gap_decomposition_v1.json" in required_sources
    assert "tail_specific_training_target_repair_design_v1.json" in required_sources
    assert "tail_repair_fixed_control_comparison_v1.json" in required_sources
    assert "tail_repair_anti_overfit_audit_v1.json" in required_sources


def test_candidate_package_must_generate_no_dummy_and_no_in_sample_attestations() -> None:
    assert "tail_repaired_r5_2_candidate_no_in_sample_decisioning_attestation_v1.json" in pkg.GENERATED_PACKAGE_FILES
    assert "tail_repaired_r5_2_candidate_no_fallback_no_dummy_no_synthetic_attestation_v1.json" in pkg.GENERATED_PACKAGE_FILES


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
    for source_name in pkg.COPY_FILE_MAP:
        (source / source_name).write_text("same\n", encoding="utf-8")
    copied = pkg._copy_required_files(source, dest)
    first = copied[0]
    package_path = dest / first["package_name_v1"]
    package_path.write_text("changed\n", encoding="utf-8")

    assert pkg._file_hash(source / first["source_name_v1"]) != pkg._file_hash(package_path)


def test_r6_precheck_cannot_run_or_authorize_r6(tmp_path: Path) -> None:
    precheck = pkg._r6_precheck(
        tmp_path,
        {
            "checks_v1": {
                "no_in_sample_decisioning_v1": True,
                "safety_remains_clean_v1": True,
                "low_support_groups_remain_reported_v1": True,
            }
        },
    )

    assert precheck["r6_was_run_v1"] is False
    assert precheck["r6_run_authorized_v1"] is False
    assert pkg.r6_precheck_authorizes_r6(precheck) is False


def test_no_optuna_r6_promotion_freeze_live() -> None:
    clean = pkg.validate_no_forbidden_actions(optuna=False, r6=False, promoted=False, freeze=False, live=False)
    blocked = pkg.validate_no_forbidden_actions(optuna=True, r6=True, promoted=True, freeze=True, live=True)

    assert clean["status_v1"] == "PASS"
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]


def test_no_implicit_latest_glob_artifact_selection() -> None:
    assert pkg.validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB") is True
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        pkg.validate_explicit_artifact_selection("LATEST_FOLDER_WINS")


def test_previous_artifact_hash_mismatch_is_reported() -> None:
    result = pkg.validate_input_artifacts_unchanged({"r5": "abc"}, {"r5": "def"})

    assert result["status_v1"] == "FAIL"
    assert result["changed_v1"] == ["r5"]

