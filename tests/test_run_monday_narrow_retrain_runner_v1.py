import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts.run_monday_narrow_retrain_runner_v1 import PrelaunchValidationError, run_runner


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_fixture(
    tmp_path: Path,
    *,
    row_count: int = 1689,
    feature_count: int = 67,
    input_surface_kind: str = "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
    bridge_input: bool = False,
    forbidden_feature: bool = False,
    abort_case: bool = False,
) -> tuple[Path, Path, Path]:
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    spec_dir = reports_root / "MONDAY_NARROW_RETRAIN_RUNNER_SPEC_V1_20260424T155748Z"
    spec_dir.mkdir()
    data_dir = reports_root / "data"
    data_dir.mkdir()

    baseline_count = max(feature_count - 5, 0)
    baseline_features = [f"as_of_baseline_feature_{idx:03d}_v1" for idx in range(baseline_count)]
    proxy_features = [
        "as_of_pre_entry_vol_exp_comp_score_v1",
        "as_of_pre_entry_directional_asymmetry_score_v1",
        "as_of_pre_entry_swing_retracement_alignment_score_v1",
        "as_of_pre_entry_tail_leakage_pocket_score_v1",
        "as_of_pre_entry_runner_protection_guard_score_v1",
    ]
    features = baseline_features + proxy_features[: max(feature_count - baseline_count, 0)]
    if forbidden_feature and features:
        features[0] = "as_of_skip_xgb_illegal_feature_v1"

    raw = pd.DataFrame({"candidate_uid": [f"cand::{idx:05d}" for idx in range(row_count)]})
    for feature in features:
        raw[feature] = 0.0
    input_path = data_dir / ("bridge_training_surface.parquet" if bridge_input else "exact_training_surface.parquet")
    raw.to_parquet(input_path, index=False)

    labels = pd.DataFrame({"candidate_uid": [f"cand::{idx:05d}" for idx in range(1689)]})
    labels["r6_label_runner_protect_v1"] = False
    labels["r6_label_bad_risk_v1"] = bool(abort_case)
    labels["r6_label_tail_control_10_50_v1"] = bool(abort_case)
    labels["r6_label_risky_allow_v1"] = False
    labels["r6_label_batch04_blindspot_v1"] = False
    labels["r6_label_repaired_165_like_runner_v1"] = False
    labels["r6_label_runner_50_mfe_v1"] = False
    labels["r6_label_runner_100_mfe_v1"] = False
    labels["r6_label_runner_200_mfe_v1"] = False
    labels["r6_label_strong_low_mae_runner_v1"] = False
    labels["r6_label_runner_near_miss_v1"] = False
    if abort_case:
        labels.loc[0, "r6_label_repaired_165_like_runner_v1"] = True
    label_path = data_dir / "labels.parquet"
    labels.to_parquet(label_path, index=False)

    feature_table = pd.DataFrame(
        [
            {
                "feature_name_v1": feature,
                "feature_group_v1": "LOCKED_BASELINE" if idx < baseline_count else "LOCKED_NEW_PROXY",
                "manifest_order_v1": idx + 1,
                "legal_status_v1": "PRE_ENTRY_LEGAL",
                "source_surface_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
                "role_v1": "test feature",
                "pockets_helped_v1": "",
                "must_exclude_v1": False,
            }
            for idx, feature in enumerate(features)
        ]
    )
    feature_table.to_csv(spec_dir / "monday_narrow_retrain_feature_manifest_v1.csv", index=False)
    _write_json(
        spec_dir / "monday_narrow_retrain_feature_manifest_v1.json",
        {
            "baseline_feature_count_v1": baseline_count,
            "new_proxy_feature_count_v1": len(features) - baseline_count,
            "total_feature_count_v1": len(features),
            "baseline_features_v1": baseline_features,
            "new_proxy_features_v1": proxy_features,
            "explicit_exclusion_list_v1": {
                "bridge_only_rows_or_signals_v1": True,
                "management_exit_truth_v1": True,
                "policy_decision_log_fields_v1": True,
                "as_of_skip_xgb_fields_v1": "as_of_skip_xgb_*",
            },
        },
    )
    _write_json(
        spec_dir / "monday_narrow_retrain_runner_spec_v1.json",
        {
            "job_name_v1": "MONDAY_NARROW_RETRAIN_RUNNER_FIRST_SHADOW_ONLY_V1",
            "scope_v1": "NARROW_RUNNER_FIRST_SHADOW_ONLY",
            "training_now_v1": False,
            "runner_may_train_v1": False,
            "runner_may_prepare_config_v1": True,
            "input_artifact_v1": str(input_path),
            "input_surface_kind_v1": input_surface_kind,
            "expected_training_rows_v1": 1689,
            "label_artifact_v1": str(label_path),
            "label_contract_v1": "LOCKED_R6_HINDSIGHT_LABEL_SURFACE_FILTERED_TO_EXACT_ONLY_CANDIDATES",
            "locked_training_heads_v1": [
                {"head_id_v1": "runner_protector", "label_col_v1": "r6_label_runner_protect_v1"},
                {"head_id_v1": "bad_risk", "label_col_v1": "r6_label_bad_risk_v1"},
                {"head_id_v1": "tail_control_10_50", "label_col_v1": "r6_label_tail_control_10_50_v1"},
                {"head_id_v1": "risky_allow", "label_col_v1": "r6_label_risky_allow_v1"},
                {"head_id_v1": "batch04_blindspot", "label_col_v1": "r6_label_batch04_blindspot_v1"},
            ],
            "compare_against_inputs_v1": [
                {"reference_v1": "FROZEN_WEDNESDAY_R6_BENCHMARK", "kind_v1": "BENCHMARK", "id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"},
                {"reference_v1": "MONDAY_R5_1_SAFETY_REFERENCE", "kind_v1": "SAFETY_REFERENCE", "id_v1": "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like"},
                {"reference_v1": "MONDAY_NATIVE_R6_FAILURE_MINER", "kind_v1": "FAILURE_MINER", "id_v1": "FAILURE_MINER_DIAGNOSIS_ONLY"},
            ],
        },
    )
    _write_json(
        spec_dir / "monday_narrow_retrain_config_lock_v1.json",
        {
            "model_family_v1": "R6_STYLE_FIVE_HEAD_SHADOW_FAMILY",
            "base_model_v1": "XGBClassifier per head",
            "compact_grid_v1": True,
            "seed_v1": 20260422,
            "n_jobs_v1": 4,
            "training_mode_v1": "SHADOW_RESEARCH_ONLY_NOT_LIVE_NOT_CONTROLLER",
            "head_family_v1": ["runner_protector", "bad_risk", "tail_control_10_50", "risky_allow", "batch04_blindspot"],
            "default_model_hyperparams_v1": {
                "tree_method_v1": "hist",
                "n_estimators_v1": 4,
                "learning_rate_v1": 0.1,
                "max_depth_v1": 1,
            },
        },
    )
    _write_json(spec_dir / "monday_narrow_retrain_prelaunch_checklist_v1.json", {"checks_v1": []})
    _write_json(spec_dir / "monday_narrow_retrain_output_spec_v1.json", {"required_outputs_v1": []})
    _write_json(
        spec_dir / "monday_narrow_retrain_abort_rules_v1.json",
        {
            "abort_before_training_v1": ["bridge used as training surface"],
            "abort_or_reject_after_eval_v1": ["repaired_165_damage > 0"],
            "automatic_invalidators_v1": ["legality breach"],
        },
    )
    _write_json(spec_dir / "summary_v1.json", {"training_now_v1": False})
    return reports_root, spec_dir, reports_root / "OUT"


def test_runner_loads_and_materializes_scaffold_without_training(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path)

    summary = run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir, run_training=False)

    assert summary["training_started_v1"] is False
    assert summary["feature_count_v1"] == 67
    assert summary["training_rows_v1"] == 1689
    assert (output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_training_summary_v1.json").exists()
    assert (output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_model_config_manifest_v1.json").exists()
    assert (output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_feature_manifest_v1.csv").exists()
    assert (output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_eval_summary_v1.json").exists()
    assert (output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_compare_against_report_v1.csv").exists()
    assert (output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_pocket_report_v1.csv").exists()
    assert (output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_verdict_package_v1.json").exists()
    status = json.loads((output_dir / "status_v1.json").read_text(encoding="utf-8"))
    assert status["training_started_v1"] is False
    assert status["failed_check_count_v1"] == 0
    next_action = json.loads((output_dir / "next_agent_action_lock_v1.json").read_text(encoding="utf-8"))
    assert next_action["primary_action_v1"] == "USE_PROTECTOR_FIRST_PATH_INSTEAD"
    assert next_action["blocked_action_v1"] == "RUN_KNOWN_FAILED_NARROW_RETRAIN_WITHOUT_FORENSICS_OVERRIDE"


def test_training_run_requires_known_failed_forensics_override(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path)

    with pytest.raises(PrelaunchValidationError, match="known failed/no-go setup"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir, run_training=True)


def test_training_run_starts_only_with_explicit_flag_and_forensics_override(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path)

    summary = run_runner(
        reports_root=reports_root,
        spec_dir=spec_dir,
        output_dir=output_dir,
        run_training=True,
        allow_known_failed_narrow_rerun_for_forensics=True,
    )

    assert summary["training_started_v1"] is True
    assert summary["run_training_flag_v1"] is True
    assert (output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_model_bundle_v1.joblib").exists()
    assert (output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_policy_prediction_view_v1.parquet").exists()
    assert (output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_compare_against_report_v1.csv").exists()
    verdict = json.loads((output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_verdict_package_v1.json").read_text(encoding="utf-8"))
    assert verdict["verdict_v1"] in {
        "CANDIDATE_SAFE_BUT_NOT_BETTER",
        "CANDIDATE_FEATURES_INSUFFICIENT",
        "CANDIDATE_IMPROVES_AND_HOLDS_SAFETY",
        "CANDIDATE_IMPROVES_BUT_FAILS_SAFETY",
    }
    status = json.loads((output_dir / "status_v1.json").read_text(encoding="utf-8"))
    assert status["RUNNER_STATUS"] == "TRAINING_RUN_COMPLETED"


def test_prelaunch_blocks_wrong_surface_kind(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, input_surface_kind="READINESS_BRIDGE_SURFACE")

    with pytest.raises(PrelaunchValidationError, match="unexpected input surface kind"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_prelaunch_blocks_wrong_row_count(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, row_count=1688)

    with pytest.raises(PrelaunchValidationError, match="training row count"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_prelaunch_blocks_wrong_feature_count(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, feature_count=66)

    with pytest.raises(PrelaunchValidationError, match="feature manifest total|selected feature count"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_prelaunch_blocks_bridge_as_training_surface(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, bridge_input=True)

    with pytest.raises(PrelaunchValidationError, match="bridge path proposed"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_prelaunch_blocks_forbidden_selected_fields(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, forbidden_feature=True)

    with pytest.raises(PrelaunchValidationError, match="forbidden selected feature"):
        run_runner(reports_root=reports_root, spec_dir=spec_dir, output_dir=output_dir)


def test_training_run_materializes_abort_disqualification(tmp_path):
    reports_root, spec_dir, output_dir = _build_fixture(tmp_path, abort_case=True)

    summary = run_runner(
        reports_root=reports_root,
        spec_dir=spec_dir,
        output_dir=output_dir,
        run_training=True,
        allow_known_failed_narrow_rerun_for_forensics=True,
    )

    assert summary["training_started_v1"] is True
    assert summary["candidate_disqualified_v1"] is True
    verdict = json.loads((output_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_verdict_package_v1.json").read_text(encoding="utf-8"))
    assert "repaired_165_damage > 0" in verdict["hard_fail_reasons_v1"]
