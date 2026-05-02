from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_retrain_readiness_recheck_and_scope_lock_v1 import main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materialize_monday_retrain_readiness_recheck_and_scope_lock_v1(tmp_path, monkeypatch):
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    bridge_dir = reports_root / "MONDAY_ENTRY_TO_FAILURE_POCKET_BRIDGE_IMPLEMENTATION_V1_20260424T142808Z"
    selected_dir = reports_root / "MONDAY_SELECTED_PRE_ENTRY_PROXIES_AND_READINESS_PACK_V1_20260424T135846Z"
    narrow_dir = reports_root / "MONDAY_R6_NARROW_PRE_ENTRY_UPLIFT_IMPLEMENTATION_LOCK_V1_20260424T130635Z"
    legal_dir = reports_root / "MONDAY_R6_LEGAL_PRE_ENTRY_FEATURE_SPEC_AND_RETRAIN_PREREQS_LOCK_V1_20260424T122237Z"
    diag_dir = reports_root / "MONDAY_R6_READONLY_DIAGNOSIS_AND_NEXT_STEP_LOCK_V1_20260424T120208Z"
    ledger_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"

    bridge_dir.mkdir()
    selected_dir.mkdir()
    narrow_dir.mkdir()
    legal_dir.mkdir()
    diag_dir.mkdir()
    ledger_dir.mkdir()

    _write_json(
        bridge_dir / "summary_v1.json",
        {
            "bridge_only_row_count_v1": 2,
            "bridge_surface_row_count_v1": 12,
            "exact_canonical_row_count_v1": 10,
        },
    )
    _write_json(
        bridge_dir / "post_bridge_readiness_recheck_pack_v1.json",
        {
            "decision_v1": "READY_FOR_RETRAIN_READINESS_RECHECK",
            "fifty_plus_sufficiently_visible_v1": True,
            "forensic_repaired_trade_not_blind_v1": True,
            "legality_failure_count_v1": 0,
            "repaired_165_fully_trackable_v1": True,
            "retrain_now_v1": False,
            "runner_near_miss_fully_accounted_for_v1": True,
        },
    )
    _write_json(
        bridge_dir / "next_agent_action_lock_v1.json",
        {"primary_action_v1": "RUN_RETRAIN_READINESS_RECHECK_NEXT"},
    )
    pd.DataFrame(
        [
            {
                "pocket_id_v1": "repaired_165",
                "total_count_v1": 3,
                "exact_only_visible_count_v1": 1,
                "bridge_only_visible_count_v1": 2,
                "readiness_trackable_count_v1": 3,
                "rest_blind_count_v1": 0,
            },
            {
                "pocket_id_v1": "forensic_repaired_trade",
                "total_count_v1": 1,
                "exact_only_visible_count_v1": 0,
                "bridge_only_visible_count_v1": 1,
                "readiness_trackable_count_v1": 1,
                "rest_blind_count_v1": 0,
            },
            {
                "pocket_id_v1": "runner_near_miss",
                "total_count_v1": 4,
                "exact_only_visible_count_v1": 1,
                "bridge_only_visible_count_v1": 3,
                "readiness_trackable_count_v1": 4,
                "rest_blind_count_v1": 0,
            },
            {
                "pocket_id_v1": "fifty_plus_mfe_seed",
                "total_count_v1": 5,
                "exact_only_visible_count_v1": 3,
                "bridge_only_visible_count_v1": 2,
                "readiness_trackable_count_v1": 5,
                "rest_blind_count_v1": 0,
            },
        ]
    ).to_csv(bridge_dir / "failure_pocket_tagging_report_v1.csv", index=False)
    pd.DataFrame(
        [
            {"check_name_v1": "A", "status_v1": "PASS"},
            {"check_name_v1": "B", "status_v1": "PASS"},
        ]
    ).to_csv(bridge_dir / "legality_and_no_canonical_pollution_guard_report_v1.csv", index=False)

    _write_json(selected_dir / "summary_v1.json", {"status_v1": {"failed_check_count_v1": 0}})
    pd.DataFrame(
        [
            {"field_name_v1": "as_of_pre_entry_vol_exp_comp_score_v1", "coverage_rate_v1": 1.0},
            {"field_name_v1": "as_of_pre_entry_directional_asymmetry_score_v1", "coverage_rate_v1": 1.0},
            {"field_name_v1": "as_of_pre_entry_swing_retracement_alignment_score_v1", "coverage_rate_v1": 1.0},
            {"field_name_v1": "as_of_pre_entry_tail_leakage_pocket_score_v1", "coverage_rate_v1": 1.0},
            {"field_name_v1": "as_of_pre_entry_runner_protection_guard_score_v1", "coverage_rate_v1": 1.0},
        ]
    ).to_csv(selected_dir / "feature_coverage_and_null_policy_report_v1.csv", index=False)
    pd.DataFrame(
        [
            {"check_name_v1": "LEGALITY_A", "status_v1": "PASS"},
            {"check_name_v1": "LEGALITY_B", "status_v1": "PASS"},
        ]
    ).to_csv(selected_dir / "legality_and_leakage_test_report_v1.csv", index=False)

    _write_json(
        narrow_dir / "summary_v1.json",
        {
            "implementation_readiness_v1": "READY_TO_IMPLEMENT_NARROW_FEATURES",
            "monday_r6_role_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
        },
    )
    _write_json(
        legal_dir / "retrain_prerequisites_lock_v1.json",
        {
            "decision_v1": "READY_FOR_NARROW_IMPLEMENTATION_PHASE",
            "retrain_now_v1": False,
        },
    )
    _write_json(
        legal_dir / "next_retrain_contract_delta_v1.json",
        {
            "compare_against_v1": [
                "FROZEN_WEDNESDAY_R6_BENCHMARK",
                "MONDAY_R5_1_SAFETY_REFERENCE",
                "MONDAY_R6_FAILURE_MINER",
            ],
            "benchmark_direction_v1": {
                "bad_blocks_target_v1": 180,
                "precision_target_v1": 0.973,
                "tail_help_target_v1": 149,
                "worst_loso_target_v1": 0.9286,
            },
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
        diag_dir / "summary_v1.json",
        {
            "correct_benchmark_v1": "FROZEN_R6_BENCHMARK",
            "correct_safety_reference_v1": "MONDAY_R5_1_SAFETY_REFERENCE",
            "monday_r6_role_v1": "FAILURE_MINER_DIAGNOSIS_ONLY",
        },
    )

    pd.DataFrame(
        [
            {"field_name_v1": "candidate_uid"},
            {"field_name_v1": "as_of_pre_entry_vol_exp_comp_score_v1"},
            {"field_name_v1": "as_of_pre_entry_directional_asymmetry_score_v1"},
            {"field_name_v1": "as_of_pre_entry_swing_retracement_alignment_score_v1"},
            {"field_name_v1": "as_of_pre_entry_tail_leakage_pocket_score_v1"},
            {"field_name_v1": "as_of_pre_entry_runner_protection_guard_score_v1"},
        ]
    ).to_csv(ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_v1.csv", index=False)
    _write_json(
        ledger_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_contract_summary_v1.json",
        {"layer_name_v1": "EXACT_ONLY_CANONICAL_RAW_STATE_CONTRACT_SUMMARY_V1"},
    )

    extension_dir = reports_root / "OUT"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_monday_retrain_readiness_recheck_and_scope_lock_v1.py",
            "--reports-root",
            str(reports_root),
            "--extension-dir",
            str(extension_dir),
        ],
    )
    main()

    summary = json.loads((extension_dir / "summary_v1.json").read_text(encoding="utf-8"))
    decision = json.loads((extension_dir / "readiness_decision_v1.json").read_text(encoding="utf-8"))
    next_action = json.loads((extension_dir / "next_agent_action_lock_v1.json").read_text(encoding="utf-8"))
    contract_recheck = json.loads((extension_dir / "retrain_contract_and_guard_recheck_v1.json").read_text(encoding="utf-8"))
    boundary = json.loads((extension_dir / "readiness_vs_training_surface_boundary_lock_v1.json").read_text(encoding="utf-8"))
    prereq_df = pd.read_csv(extension_dir / "retrain_readiness_prerequisites_recheck_v1.csv")
    audit_df = pd.read_csv(extension_dir / "consistency_audit_v1.csv")

    assert summary["readiness_decision_v1"] == "READY_TO_PLAN_NARROW_RETRAIN"
    assert decision["decision_v1"] == "READY_TO_PLAN_NARROW_RETRAIN"
    assert decision["retrain_now_v1"] is False
    assert next_action["primary_action_v1"] == "PLAN_NARROW_RETRAIN_NEXT"
    assert "DO_NOT_USE_BRIDGE_AS_TRAINING_SURFACE" in next_action["supporting_actions_v1"]
    assert contract_recheck["contract_still_correct_v1"] is True
    assert "DO_NOT_USE_BRIDGE_AS_TRAINING_SURFACE" in contract_recheck["new_guardrails_after_bridge_v1"]
    assert boundary["retrain_readiness_can_open_without_training_surface_change_v1"] is True
    assert boundary["readiness_surface_v1"]["bridge_only_row_count_v1"] == 2
    assert prereq_df["status_v1"].astype("string").eq("PASS").all()
    assert audit_df["status_v1"].astype("string").eq("PASS").all()
