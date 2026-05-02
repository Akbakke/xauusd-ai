from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_narrow_retrain_scope_plan_v1 import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materialize_monday_narrow_retrain_scope_plan_v1(tmp_path, monkeypatch):
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    readiness_dir = reports_root / "MONDAY_RETRAIN_READINESS_RECHECK_AND_SCOPE_LOCK_V1_20260424T145118Z"
    bridge_dir = reports_root / "MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1_20260424T142808Z"
    diag_dir = reports_root / "MONDAY_R6_READONLY_DIAGNOSIS_AND_NEXT_STEP_LOCK_V1_20260424T120208Z"
    ledger_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"
    r6_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_ENTRY_RUNNER_FIRST_RETRAIN_V1"

    readiness_dir.mkdir()
    bridge_dir.mkdir()
    diag_dir.mkdir()
    ledger_dir.mkdir()
    r6_dir.mkdir()

    _write_json(readiness_dir / "readiness_decision_v1.json", {"decision_v1": "READY_TO_PLAN_NARROW_RETRAIN"})
    _write_json(
        readiness_dir / "summary_v1.json",
        {
            "readiness_decision_v1": "READY_TO_PLAN_NARROW_RETRAIN",
            "next_action_v1": "PLAN_NARROW_RETRAIN_NEXT",
        },
    )
    _write_json(
        readiness_dir / "narrow_retrain_scope_proposal_v1.json",
        {
            "scope_status_v1": "PLAN_ONLY_SCOPE_ALLOWED",
            "training_surface_v1": {
                "artifact_v1": str(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet"),
                "surface_kind_v1": "CANONICAL_EXACT_ONLY_TRAINING_SURFACE",
                "must_not_expand_with_bridge_rows_v1": True,
            },
        },
    )
    _write_json(
        readiness_dir / "retrain_contract_and_guard_recheck_v1.json",
        {
            "compare_against_v1": [
                "FROZEN_WEDNESDAY_R6_BENCHMARK",
                "MONDAY_R5_1_SAFETY_REFERENCE",
                "MONDAY_R6_FAILURE_MINER",
            ],
            "must_improve_over_monday_r6_v1": {
                "bad_blocks_gt_v1": 84,
                "precision_gte_v1": 0.9545,
                "tail_help_gt_v1": 84,
                "worst_loso_precision_gte_v1": 0.8889,
            },
            "must_keep_safe_v1": {
                "repaired_165_damage_v1": 0,
                "forensic_trade_must_stay_unblocked_v1": "TRUTH_MONFRI_WEEK_20260330_20260406:0:cand::000612:d2e2d6b7fb03",
                "fifty_plus_mfe_blocked_max_v1": 1,
                "hundred_plus_mfe_blocked_v1": 0,
                "two_hundred_plus_mfe_blocked_v1": 0,
                "strongest_winner_path_damage_v1": 0,
            },
        },
    )
    _write_json(
        bridge_dir / "summary_v1.json",
        {
            "bridge_surface_row_count_v1": 12,
            "bridge_only_row_count_v1": 2,
            "exact_canonical_row_count_v1": 10,
        },
    )
    _write_json(
        diag_dir / "summary_v1.json",
        {
            "monday_r6_role_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
            "correct_benchmark_v1": "FROZEN_R6_BENCHMARK",
            "correct_safety_reference_v1": "MONDAY_R5_1_SAFETY_REFERENCE",
        },
    )
    _write_json(
        reports_root / "truth_r5_loso_batch04_robustness_retrain_v1.json",
        {
            "selected_candidate_v1": {
                "should_not_take_block_count_v1": 66,
                "repaired_165_block_count_v1": 0,
                "fifty_plus_mfe_block_count_v1": 0,
                "hundred_plus_mfe_block_count_v1": 0,
                "two_hundred_plus_mfe_block_count_v1": 0,
            }
        },
    )

    raw_df = pd.DataFrame({"candidate_uid": [f"cand::{i:04d}" for i in range(10)]})
    raw_df.to_parquet(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet", index=False)
    pd.DataFrame({"candidate_uid": [f"cand::{i:04d}" for i in range(10)]}).to_parquet(
        r6_dir / "shadow_meta_all_trade_review_r6_entry_runner_first_hindsight_label_outcome_table_v1.parquet",
        index=False,
    )
    pd.DataFrame({"candidate_uid": [f"cand::{i:04d}" for i in range(10)]}).to_parquet(
        r6_dir / "shadow_meta_all_trade_review_r6_policy_prediction_view_v1.parquet",
        index=False,
    )

    contract_df = pd.DataFrame(
        [
            {
                "feature_name": "candidate_uid",
                "semantic_group": "IDENTITY",
                "as_of_safe_v1": True,
                "direct_only_allowed_v1": False,
                "leakage_risk_v1": "LOW",
            },
            {
                "feature_name": "as_of_skip_replay_window_ret_1_bps_v1",
                "semantic_group": "SHORT_HORIZON_PRE_ENTRY_CONTEXT",
                "as_of_safe_v1": True,
                "direct_only_allowed_v1": True,
                "leakage_risk_v1": "LOW",
            },
            {
                "feature_name": "as_of_skip_candidate_p_hat_v1",
                "semantic_group": "EXISTING_CANDIDATE_SNAPSHOT",
                "as_of_safe_v1": True,
                "direct_only_allowed_v1": True,
                "leakage_risk_v1": "LOW_SOURCE_SPECIFIC",
            },
            {
                "feature_name": "as_of_pre_entry_vol_exp_comp_score_v1",
                "semantic_group": "PRE_ENTRY_VOL_EXP_COMP_CONTEXT",
                "as_of_safe_v1": True,
                "direct_only_allowed_v1": True,
                "leakage_risk_v1": "LOW_DERIVED_FROM_AS_OF_ONLY",
            },
            {
                "feature_name": "as_of_pre_entry_directional_asymmetry_score_v1",
                "semantic_group": "PRE_ENTRY_DIRECTIONAL_ASYMMETRY_CONTEXT",
                "as_of_safe_v1": True,
                "direct_only_allowed_v1": True,
                "leakage_risk_v1": "LOW_DERIVED_FROM_AS_OF_ONLY",
            },
            {
                "feature_name": "as_of_pre_entry_swing_retracement_alignment_score_v1",
                "semantic_group": "PRE_ENTRY_SWING_RETRACEMENT_ALIGNMENT_CONTEXT",
                "as_of_safe_v1": True,
                "direct_only_allowed_v1": True,
                "leakage_risk_v1": "LOW_DERIVED_FROM_AS_OF_ONLY",
            },
            {
                "feature_name": "as_of_pre_entry_tail_leakage_pocket_score_v1",
                "semantic_group": "PRE_ENTRY_TAIL_LEAKAGE_POCKET_PROXY",
                "as_of_safe_v1": True,
                "direct_only_allowed_v1": True,
                "leakage_risk_v1": "LOW_IF_DERIVATION_LOCKED",
            },
            {
                "feature_name": "as_of_pre_entry_runner_protection_guard_score_v1",
                "semantic_group": "PRE_ENTRY_RUNNER_PROTECTION_GUARD",
                "as_of_safe_v1": True,
                "direct_only_allowed_v1": True,
                "leakage_risk_v1": "LOW_IF_DERIVATION_LOCKED",
            },
            {
                "feature_name": "as_of_skip_xgb_p_hat_v1",
                "semantic_group": "XGB_PREDICTION_FAMILY",
                "as_of_safe_v1": True,
                "direct_only_allowed_v1": True,
                "leakage_risk_v1": "TARGET_ADJACENT_REVIEW_REQUIRED",
            },
        ]
    )
    contract_df.to_csv(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_v1.csv", index=False)
    _write_json(
        ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_summary_v1.json",
        {"layer_name_v1": "EXACT_ONLY_CANONICAL_RAW_STATE"},
    )

    extension_dir = reports_root / "OUT"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_monday_narrow_retrain_scope_plan_v1.py",
            "--reports-root",
            str(reports_root),
            "--extension-dir",
            str(extension_dir),
        ],
    )
    main()

    summary = json.loads((extension_dir / "summary_v1.json").read_text(encoding="utf-8"))
    plan = json.loads((extension_dir / "narrow_retrain_plan_v1.json").read_text(encoding="utf-8"))
    training_surface = json.loads((extension_dir / "training_surface_lock_v1.json").read_text(encoding="utf-8"))
    feature_lock = json.loads((extension_dir / "feature_set_lock_v1.json").read_text(encoding="utf-8"))
    objective = json.loads((extension_dir / "training_objective_and_priority_lock_v1.json").read_text(encoding="utf-8"))
    eval_plan = json.loads((extension_dir / "eval_and_regression_guard_plan_v1.json").read_text(encoding="utf-8"))
    io_lock = json.loads((extension_dir / "training_run_inputs_and_outputs_lock_v1.json").read_text(encoding="utf-8"))
    next_action = json.loads((extension_dir / "next_agent_action_lock_v1.json").read_text(encoding="utf-8"))
    audit_df = pd.read_csv(extension_dir / "consistency_audit_v1.csv")

    assert summary["scope_v1"] == "NARROW_RUNNER_FIRST_SHADOW_ONLY"
    assert summary["training_now_v1"] is False
    assert plan["training_surface_v1"]["surface_kind_v1"] == "CANONICAL_EXACT_ONLY_TRAINING_SURFACE"
    assert training_surface["training_row_count_v1"] == 10
    assert training_surface["bridge_surface_not_allowed_v1"]["bridge_only_row_count_v1"] == 2
    assert training_surface["bridge_surface_not_allowed_v1"]["why_not_allowed_v1"]
    assert set(feature_lock["new_proxy_features_v1"].keys()) == {
        "as_of_pre_entry_vol_exp_comp_score_v1",
        "as_of_pre_entry_directional_asymmetry_score_v1",
        "as_of_pre_entry_swing_retracement_alignment_score_v1",
        "as_of_pre_entry_tail_leakage_pocket_score_v1",
        "as_of_pre_entry_runner_protection_guard_score_v1",
    }
    assert "as_of_skip_replay_window_ret_1_bps_v1" in feature_lock["baseline_training_features_v1"]["feature_names_v1"]
    assert "as_of_skip_xgb_p_hat_v1" not in feature_lock["baseline_training_features_v1"]["feature_names_v1"]
    assert objective["priority_order_v1"][0]["objective_v1"] == "RUNNER_PROTECTION_AND_REPAIRED_165_SAFETY"
    assert eval_plan["compare_against_v1"][0]["id_v1"] == "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"
    assert io_lock["inputs_v1"]["readiness_bridge_use_v1"] == "EVAL_ONLY_NOT_TRAINING"
    assert next_action["primary_action_v1"] == "AUTHORIZE_NARROW_RETRAIN_JOB_SPEC_ONLY"
    assert "NEXT_AGENT_MAY_PREPARE_TRAINING_BUT_NOT_RUN_IT" in next_action["supporting_actions_v1"]
    assert audit_df["status_v1"].astype("string").eq("PASS").all()
