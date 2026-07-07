from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts import materialize_best_lane_candidate_package_requires_explicit_gate_v1 as pkg


def _metrics(**overrides: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "lane_id_v1": "LANE_08_R5_2_GAP_ROWS_SAFE_ONLY",
        "bad_count_v1": 185,
        "tail_count_v1": 139,
        "precision_v1": 1.0,
        "precision_denominator_v1": 185,
        "precision_decision_valid_v1": True,
        "strict_all_run_id_worst_loso_denominator_v1": 2,
        "strict_all_run_id_decision_valid_v1": False,
        "selected_low_support_group_count_v1": 9,
        "structural_low_support_selected_group_count_v1": 7,
        "safety_clean_v1": True,
        "final_promotion_allowed_v1": False,
    }
    metrics.update(overrides)
    return metrics


def test_package_must_select_lane_08() -> None:
    assert pkg.validate_selected_lane(_metrics()) is True
    with pytest.raises(RuntimeError, match="METRIC_PRESERVATION"):
        pkg.validate_selected_lane(_metrics(lane_id_v1="LANE_07_R6_TAIL_HEAD_PLUS_RUN_ID_SUPPORT"))


def test_package_must_preserve_185_139_and_precision_denominator() -> None:
    with pytest.raises(RuntimeError, match="METRIC_PRESERVATION"):
        pkg.validate_selected_lane(_metrics(bad_count_v1=184))
    with pytest.raises(RuntimeError, match="METRIC_PRESERVATION"):
        pkg.validate_selected_lane(_metrics(precision_denominator_v1=184))


def test_package_must_preserve_strict_loso_low_support_and_safety() -> None:
    with pytest.raises(RuntimeError, match="METRIC_PRESERVATION"):
        pkg.validate_selected_lane(_metrics(strict_all_run_id_decision_valid_v1=True))
    with pytest.raises(RuntimeError, match="METRIC_PRESERVATION"):
        pkg.validate_selected_lane(_metrics(selected_low_support_group_count_v1=0))
    with pytest.raises(RuntimeError, match="METRIC_PRESERVATION"):
        pkg.validate_selected_lane(_metrics(safety_clean_v1=False))


def test_package_cannot_mark_final_promotion_freeze_or_live_true() -> None:
    with pytest.raises(RuntimeError, match="CANNOT_BE_PROMOTED"):
        pkg.validate_not_promoted({"final_promotion_allowed_v1": True})
    with pytest.raises(RuntimeError, match="CANNOT_BE_PROMOTED"):
        pkg.validate_not_promoted({"freeze_ready_v1": True, "live_ready_v1": True})


def test_no_optuna_broad_sweep_r6_freeze_promo_live() -> None:
    clean = pkg.validate_no_forbidden_actions(
        optuna=False,
        broad_sweep=False,
        r6=False,
        promoted=False,
        freeze=False,
        live=False,
    )
    blocked = pkg.validate_no_forbidden_actions(
        optuna=True,
        broad_sweep=True,
        r6=True,
        promoted=True,
        freeze=True,
        live=True,
    )
    assert clean["status_v1"] == "PASS"
    assert blocked["status_v1"] == "FAIL"
    assert "R6_FORBIDDEN" in blocked["failures_v1"]


def test_no_implicit_latest_glob_artifact_selection() -> None:
    assert pkg.validate_explicit_artifact_selection("EXPLICIT_ONLY_NO_LATEST_GLOB") is True
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        pkg.validate_explicit_artifact_selection("LATEST")


def test_missing_lane_result_summary_causes_hard_failure(tmp_path: Path) -> None:
    lane_pack = tmp_path / "lane_pack"
    (lane_pack / "lanes" / pkg.SELECTED_LANE_ID).mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Missing required source artifact"):
        pkg._source_required_files(lane_pack)


def test_missing_safety_denominator_or_low_support_report_causes_hard_failure(tmp_path: Path) -> None:
    lane_pack = tmp_path / "lane_pack"
    lane_dir = lane_pack / "lanes" / pkg.SELECTED_LANE_ID
    lane_dir.mkdir(parents=True)
    for source_name in pkg.LANE_FILE_MAP:
        if source_name not in {
            "lane_safety_report_v1.json",
            "lane_metric_denominator_report_v1.json",
            "lane_low_support_report_v1.json",
        }:
            (lane_dir / source_name).write_text("{}\n", encoding="utf-8")
    for source_name in pkg.ROOT_REFERENCE_FILE_MAP:
        (lane_pack / source_name).write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing required source artifact"):
        pkg._source_required_files(lane_pack)


def test_file_hash_mismatch_causes_hard_failure(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    lane_dir = source / "lanes" / pkg.SELECTED_LANE_ID
    lane_dir.mkdir(parents=True)
    dest.mkdir()
    for source_name in pkg.LANE_FILE_MAP:
        (lane_dir / source_name).write_text("same\n", encoding="utf-8")
    for source_name in pkg.ROOT_REFERENCE_FILE_MAP:
        (source / source_name).write_text("same\n", encoding="utf-8")
    copied = pkg._copy_required_files(source, dest)
    first = copied[0]
    package_path = dest / first["package_name_v1"]
    package_path.write_text("changed\n", encoding="utf-8")
    assert pkg._file_hash(Path(first["source_path_v1"])) != pkg._file_hash(package_path)


def test_lane_10_reproducibility_must_pass(tmp_path: Path) -> None:
    lane10 = tmp_path / "lanes" / "LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL"
    lane10.mkdir(parents=True)
    (lane10 / "lane_result_summary_v1.json").write_text(
        "{"
        '"lane_id_v1":"LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL",'
        '"bad_count_v1":140,"tail_count_v1":94,'
        '"rows_added_vs_140_94_v1":0,"rows_lost_vs_140_94_v1":0,'
        '"safety_clean_v1":true'
        "}\n",
        encoding="utf-8",
    )
    assert pkg.validate_lane10_reproducibility(tmp_path)["status_v1"] == "PASS"
    (lane10 / "lane_result_summary_v1.json").write_text(
        '{"lane_id_v1":"LANE_10_NULL_REPLAY_REPRODUCIBILITY_CONTROL","bad_count_v1":141}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="LANE_10_REPRODUCIBILITY"):
        pkg.validate_lane10_reproducibility(tmp_path)


def test_anti_overfit_audit_must_pass() -> None:
    anti = {
        "status_v1": "PARALLEL_LANE_PACK_STABLE_TRACK_PASS",
        "all_lanes_pre_registered_v1": True,
        "no_optuna_v1": True,
        "no_large_sweep_v1": True,
        "no_post_hoc_lane_mutation_v1": True,
        "no_in_sample_decisioning_v1": True,
        "strict_loso_visible_v1": True,
        "low_support_visible_v1": True,
        "no_dummy_synthetic_fallback_v1": True,
        "no_implicit_latest_glob_v1": True,
        "lane_10_reproducibility_pass_v1": True,
    }
    assert pkg.validate_anti_overfit(anti) is True
    anti["no_optuna_v1"] = False
    with pytest.raises(RuntimeError, match="ANTI_OVERFIT"):
        pkg.validate_anti_overfit(anti)


def _write_large_jump_inputs(tmp_path: Path, *, unsafe_column: str | None = None, unsupported: bool = False) -> tuple[Path, Path]:
    lane_dir = tmp_path / "lane"
    r6_root = tmp_path / "r6"
    lane_dir.mkdir()
    r6_root.mkdir()
    membership_rows = []
    score_rows = []
    for idx in range(185):
        uid = f"row_{idx:03d}"
        added = idx >= 140
        membership_rows.append(
            {
                "candidate_uid_v1": uid,
                "lane_selected_v1": True,
                "rows_added_vs_140_94_v1": added,
            }
        )
        score_rows.append(
            {
                "candidate_uid_v1": uid,
                "run_id_v1": "RUN_A",
                "bad_label_v1": True,
                "tail_label_v1": idx >= 46,
                "active_quarantine_v1": "ACTIVE_CANDIDATE",
                "source_evidence_v1": "" if unsupported and added else "R5_1_BAD_SCORE:SUPPORT",
                "run_id_policy_class_v1": "SUPPORT_REPAIRABLE_WITH_EXISTING_SAFE_SIGNALS",
                "safe_recoverable_v1": True,
                "training_opportunity_allowed_v1": True,
                "protected_winner_status_v1": False,
                "runner_protect_status_v1": False,
                "ambiguous_high_mfe_status_v1": False,
                "fifty_plus_mfe_risk_v1": False,
                "hundred_plus_mfe_risk_v1": False,
                "two_hundred_plus_mfe_risk_v1": False,
            }
        )
    if unsafe_column:
        for row in score_rows[140:]:
            row[unsafe_column] = True
            break
    pd.DataFrame(membership_rows).to_csv(lane_dir / "lane_scores_or_membership_v1.csv", index=False)
    pd.DataFrame(score_rows).to_csv(r6_root / "r6_tail_repaired_oof_scores_v1.csv", index=False)
    return lane_dir, r6_root


def test_large_jump_audit_inspects_added_rows_and_passes_when_clean(tmp_path: Path) -> None:
    lane_dir, r6_root = _write_large_jump_inputs(tmp_path)
    audit = pkg._large_jump_delta_audit(
        output_dir=tmp_path,
        lane_dir=lane_dir,
        r6_root=r6_root,
        r6_feature_hash={
            "feature_validation_v1": {"forbidden_features_v1": []},
            "hindsight_validation_v1": {"hindsight_features_v1": []},
        },
    )
    assert audit["status_v1"] == "LARGE_JUMP_SANITY_PASS"
    assert audit["added_rows_count_v1"] == 45
    assert audit["added_bad_rows_v1"] == 45
    assert audit["added_tail_rows_v1"] == 45
    assert pkg.validate_large_jump_audit(audit) is True


@pytest.mark.parametrize(
    "unsafe_column",
    [
        "protected_winner_status_v1",
        "runner_protect_status_v1",
        "ambiguous_high_mfe_status_v1",
        "fifty_plus_mfe_risk_v1",
    ],
)
def test_large_jump_audit_blocks_added_safety_conflicts(tmp_path: Path, unsafe_column: str) -> None:
    lane_dir, r6_root = _write_large_jump_inputs(tmp_path, unsafe_column=unsafe_column)
    audit = pkg._large_jump_delta_audit(
        output_dir=tmp_path,
        lane_dir=lane_dir,
        r6_root=r6_root,
        r6_feature_hash={
            "feature_validation_v1": {"forbidden_features_v1": []},
            "hindsight_validation_v1": {"hindsight_features_v1": []},
        },
    )
    assert audit["status_v1"] == "LARGE_JUMP_BLOCKED_BY_SAFETY_CONCERN"
    with pytest.raises(RuntimeError, match="LARGE_JUMP_SANITY"):
        pkg.validate_large_jump_audit(audit)


def test_large_jump_audit_blocks_added_quarantine_or_unsupported_rows(tmp_path: Path) -> None:
    lane_dir, r6_root = _write_large_jump_inputs(tmp_path, unsupported=True)
    audit = pkg._large_jump_delta_audit(
        output_dir=tmp_path,
        lane_dir=lane_dir,
        r6_root=r6_root,
        r6_feature_hash={
            "feature_validation_v1": {"forbidden_features_v1": []},
            "hindsight_validation_v1": {"hindsight_features_v1": []},
        },
    )
    assert audit["status_v1"] == "LARGE_JUMP_BLOCKED_BY_MISSING_EVIDENCE"

    rows = pd.read_csv(r6_root / "r6_tail_repaired_oof_scores_v1.csv")
    rows.loc[140, "active_quarantine_v1"] = "QUARANTINE_EXCLUDE"
    rows.to_csv(r6_root / "r6_tail_repaired_oof_scores_v1.csv", index=False)
    audit = pkg._large_jump_delta_audit(
        output_dir=tmp_path,
        lane_dir=lane_dir,
        r6_root=r6_root,
        r6_feature_hash={
            "feature_validation_v1": {"forbidden_features_v1": []},
            "hindsight_validation_v1": {"hindsight_features_v1": []},
        },
    )
    assert audit["status_v1"] == "LARGE_JUMP_BLOCKED_BY_SAFETY_CONCERN"


def test_large_jump_audit_blocks_hindsight_or_leakage_features(tmp_path: Path) -> None:
    lane_dir, r6_root = _write_large_jump_inputs(tmp_path)
    audit = pkg._large_jump_delta_audit(
        output_dir=tmp_path,
        lane_dir=lane_dir,
        r6_root=r6_root,
        r6_feature_hash={
            "feature_validation_v1": {"forbidden_features_v1": ["trade_id_v1"]},
            "hindsight_validation_v1": {"hindsight_features_v1": []},
        },
    )
    assert audit["status_v1"] == "LARGE_JUMP_BLOCKED_BY_LEAKAGE_CONCERN"


def test_r6_precheck_cannot_authorize_r6_without_explicit_gate(tmp_path: Path) -> None:
    for name in [
        "best_lane_candidate_selected_rows_v1.csv",
        "best_lane_candidate_scores_or_membership_v1.csv",
        "best_lane_candidate_lane_config_v1.json",
        "best_lane_candidate_lane_result_summary_v1.json",
        "best_lane_candidate_safety_report_v1.json",
        "best_lane_candidate_metric_denominator_report_v1.json",
        "best_lane_candidate_low_support_report_v1.json",
        "best_lane_candidate_membership_only_provenance_v1.json",
    ]:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    precheck = pkg._r6_precheck(tmp_path, "PASS", "LARGE_JUMP_SANITY_PASS")
    assert precheck["status_v1"] == "R6_INPUT_PACKAGE_REQUIRES_ADAPTER_FOR_LANE_MEMBERSHIP_INPUT"
    assert precheck["r6_run_authorized_v1"] is False
    assert pkg.r6_precheck_authorizes_r6(precheck) is False


def test_fixed_control_comparison_must_include_wednesday() -> None:
    rows = [{"control_v1": "wednesday", "bad_v1": 180, "tail_v1": 149}]
    assert pkg.validate_fixed_controls(rows) is True
    with pytest.raises(RuntimeError, match="WEDNESDAY"):
        pkg.validate_fixed_controls([{"control_v1": "tail_repaired_r5_2", "bad_v1": 140, "tail_v1": 94}])


def test_recommendation_prefers_stability_recheck_for_large_membership_jump() -> None:
    rec = pkg._recommendation(
        "PASS",
        "LARGE_JUMP_SANITY_PASS",
        {"status_v1": "R6_INPUT_PACKAGE_REQUIRES_ADAPTER_FOR_LANE_MEMBERSHIP_INPUT"},
    )
    assert rec["status_v1"] == "BEST_LANE_PACKAGE_READY_FOR_STABILITY_RECHECK_BEFORE_R6"
    assert rec["next_recommended_action_v1"] == "STABILITY_RECHECK_BEST_LANE_185_139_BEFORE_R6_V1"
