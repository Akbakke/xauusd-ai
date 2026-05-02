import json
import sys
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_narrow_retrain_runner_spec_v1 import main


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materialize_monday_narrow_retrain_runner_spec_v1(tmp_path, monkeypatch):
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    job_dir = reports_root / "MONDAY_NARROW_RETRAIN_JOB_SPEC_V1_20260424T154329Z"
    scope_dir = reports_root / "MONDAY_NARROW_RETRAIN_SCOPE_PLAN_V1_20260424T150858Z"
    job_dir.mkdir()
    scope_dir.mkdir()

    baseline_features = [f"as_of_baseline_feature_{idx:02d}_v1" for idx in range(62)]
    proxy_features = {
        "as_of_pre_entry_vol_exp_comp_score_v1": {
            "role_v1": "vol role",
            "pockets_helped_v1": ["missed_10_50_tail_control_pocket"],
        },
        "as_of_pre_entry_directional_asymmetry_score_v1": {
            "role_v1": "directional role",
            "pockets_helped_v1": ["runner_near_miss_pocket"],
        },
        "as_of_pre_entry_swing_retracement_alignment_score_v1": {
            "role_v1": "swing role",
            "pockets_helped_v1": ["repaired_165_pocket"],
        },
        "as_of_pre_entry_tail_leakage_pocket_score_v1": {
            "role_v1": "tail role",
            "pockets_helped_v1": ["missed_10_50_tail_control_pocket"],
        },
        "as_of_pre_entry_runner_protection_guard_score_v1": {
            "role_v1": "guard role",
            "pockets_helped_v1": ["forensic_repaired_trade"],
        },
    }
    _write_json(
        scope_dir / "feature_set_lock_v1.json",
        {
            "baseline_training_features_v1": {
                "feature_count_v1": 62,
                "feature_families_v1": {"BASELINE": 62},
                "feature_names_v1": baseline_features,
            },
            "new_proxy_features_v1": proxy_features,
            "explicit_exclusions_v1": {
                "forbidden_sources_v1": [
                    "management/exit truth",
                    "policy-log / decision-log fields",
                    "bridge-only derived signals",
                    "deferred candidates",
                ],
                "forbidden_field_examples_v1": [
                    "last_peak_ts",
                    "management_policy_scores_or_decision_log_fields",
                    "bridge_only_rows_from_fullcoverage_r6_asof",
                ],
            },
        },
    )

    _write_json(
        job_dir / "training_job_spec_lock_v1.json",
        {
            "target_label_contract_v1": "LOCKED_R6_HINDSIGHT_LABEL_SURFACE_FILTERED_TO_EXACT_ONLY_CANDIDATES",
            "eval_package_v1": {
                "compare_against_v1": [
                    {"reference_v1": "FROZEN_WEDNESDAY_R6_BENCHMARK", "id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"},
                    {"reference_v1": "MONDAY_R5_1_SAFETY_REFERENCE", "id_v1": "R5_1_CANDIDATE_0241_R5_1_COMBINED_repaired_165_like"},
                    {"reference_v1": "MONDAY_NATIVE_R6_FAILURE_MINER", "id_v1": "FAILURE_MINER_DIAGNOSIS_ONLY"},
                ]
            },
        },
    )
    _write_json(
        job_dir / "training_input_contract_v1.json",
        {
            "valid_input_v1": {
                "training_feature_surface_path_v1": str(reports_root / "exact.parquet"),
                "training_feature_surface_kind_v1": "CANONICAL_EXACT_ONLY_ENTRY_RAW_STATE",
                "expected_training_row_count_v1": 1689,
                "expected_exact_label_intersection_v1": 1689,
            },
            "input_validation_hard_fail_v1": [
                "training surface path != locked exact-only parquet",
                "any bridge-only rows included in training population",
            ],
        },
    )
    _write_json(
        job_dir / "label_and_target_lock_v1.json",
        {
            "target_surface_v1": {"artifact_v1": str(reports_root / "labels.parquet")},
            "locked_training_heads_v1": [
                {"head_id_v1": "runner_protector", "label_col_v1": "r6_label_runner_protect_v1"},
                {"head_id_v1": "bad_risk", "label_col_v1": "r6_label_bad_risk_v1"},
            ],
        },
    )
    _write_json(
        job_dir / "model_and_training_configuration_spec_v1.json",
        {
            "model_family_v1": {"head_family_v1": ["runner_protector", "bad_risk"]},
            "training_mode_v1": "SHADOW_RESEARCH_ONLY_NOT_LIVE_NOT_CONTROLLER",
            "evaluation_structure_v1": {
                "walkforward_required_v1": True,
                "loso_required_v1": True,
                "rolling_window_required_v1": True,
                "batch_weeks_v1": 15,
                "batch04_batch05_reporting_v1": "BATCH_04 must be reported; BATCH_05 must be null if absent, not fail.",
            },
            "reproducibility_v1": {"seed_v1": 20260422, "n_jobs_v1": 4},
            "default_model_hyperparams_v1": {"tree_method_v1": "hist"},
            "must_hold_constant_vs_prior_r6_v1": ["five-head family continuity"],
            "allowed_to_change_in_this_narrow_slice_v1": ["feature matrix with the five new legal proxies included"],
            "not_allowed_to_change_v1": ["policy family"],
        },
    )
    _write_json(job_dir / "output_artifact_spec_v1.json", {"required_artifacts_v1": [{"artifact_name_v1": "summary.json"}]})
    _write_json(job_dir / "eval_verdict_matrix_v1.json", {"verdicts_v1": []})
    _write_json(job_dir / "pre_run_validation_checklist_v1.json", {"checks_v1": []})
    _write_json(job_dir / "post_run_eval_checklist_v1.json", {"checks_v1": []})
    _write_json(
        job_dir / "no_go_and_abort_protocol_v1.json",
        {
            "do_not_start_if_v1": ["bridge proposed as training surface"],
            "reject_after_run_if_v1": ["repaired_165_damage > 0"],
            "automatic_invalidators_v1": ["legality breach"],
        },
    )
    _write_json(job_dir / "summary_v1.json", {"exact_label_intersection_v1": 1689})
    _write_json(job_dir / "status_v1.json", {"failed_check_count_v1": 0})

    extension_dir = reports_root / "OUT"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_monday_narrow_retrain_runner_spec_v1.py",
            "--reports-root",
            str(reports_root),
            "--extension-dir",
            str(extension_dir),
        ],
    )
    main()

    summary = json.loads((extension_dir / "summary_v1.json").read_text(encoding="utf-8"))
    runner_spec = json.loads((extension_dir / "monday_narrow_retrain_runner_spec_v1.json").read_text(encoding="utf-8"))
    config_lock = json.loads((extension_dir / "monday_narrow_retrain_config_lock_v1.json").read_text(encoding="utf-8"))
    feature_manifest = json.loads((extension_dir / "monday_narrow_retrain_feature_manifest_v1.json").read_text(encoding="utf-8"))
    feature_df = pd.read_csv(extension_dir / "monday_narrow_retrain_feature_manifest_v1.csv")
    prelaunch = json.loads((extension_dir / "monday_narrow_retrain_prelaunch_checklist_v1.json").read_text(encoding="utf-8"))
    output_spec = json.loads((extension_dir / "monday_narrow_retrain_output_spec_v1.json").read_text(encoding="utf-8"))
    abort_rules = json.loads((extension_dir / "monday_narrow_retrain_abort_rules_v1.json").read_text(encoding="utf-8"))
    next_action = json.loads((extension_dir / "next_agent_action_lock_v1.json").read_text(encoding="utf-8"))
    status = json.loads((extension_dir / "status_v1.json").read_text(encoding="utf-8"))
    audit_df = pd.read_csv(extension_dir / "consistency_audit_v1.csv")

    assert summary["training_now_v1"] is False
    assert summary["total_feature_count_v1"] == 67
    assert runner_spec["runner_may_train_v1"] is False
    assert runner_spec["runner_may_prepare_config_v1"] is True
    assert runner_spec["expected_training_rows_v1"] == 1689
    assert config_lock["model_family_v1"] == "R6_STYLE_FIVE_HEAD_SHADOW_FAMILY"
    assert config_lock["seed_v1"] == 20260422
    assert feature_manifest["baseline_feature_count_v1"] == 62
    assert feature_manifest["new_proxy_feature_count_v1"] == 5
    assert len(feature_df) == 67
    assert not feature_df["feature_name_v1"].astype(str).str.contains("as_of_skip_xgb_").any()
    assert any(check["check_id_v1"] == "BRIDGE_NOT_USED_AS_TRAINING_SURFACE" for check in prelaunch["checks_v1"])
    assert any(row["output_id_v1"] == "VERDICT_PACKAGE" for row in output_spec["required_outputs_v1"])
    assert "repaired_165_damage > 0" in abort_rules["abort_or_reject_after_eval_v1"]
    assert next_action["primary_action_v1"] == "DO_NOT_RETRAIN_SAME_NARROW_SETUP_AGAIN"
    assert next_action["blocked_action_v1"] == "RUN_KNOWN_FAILED_NARROW_RETRAIN_AS_ACTIVE_PATH"
    assert status["runner_implementation_allowed_next_v1"] is False
    assert status["historical_context_only_v1"] is True
    assert status["failed_check_count_v1"] == 0
    assert audit_df["status_v1"].astype("string").eq("PASS").all()
