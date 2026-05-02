from __future__ import annotations

import json
from pathlib import Path

from gx1.scripts.materialize_truth_management_inventory_contract_v1 import (
    build_management_inventory_contract,
    write_management_inventory_contract,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def test_build_management_inventory_contract_materializes_expected_sections(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    r8_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260420_RUNTIME_RECOVERY_R8_HANDOFF_REALFIX"
    ext_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260420T133345Z_MANAGEMENT_AUDIT_EXTENSION_V1"
    r8_dir.mkdir(parents=True)
    ext_dir.mkdir(parents=True)

    _write_json(
        reports_root / "truth_trade_foundation_quality_v1.json",
        {
            "trade_count": 100,
            "outlook_v1": "POSITIVE_EDGE_HIGH_REGRET",
            "profitability": {"avg_pnl_bps": 2.5, "profit_factor": 1.1, "max_drawdown_bps": -100.0},
            "exit_efficiency": {"early_exit_regret_rate": 0.5, "early_exit_regret_count": 50},
            "hold_longer_pressure": {"extra_value_bps": {"mean": 12.0}},
            "quality_flags": {"clean_good_trade_mfe20_mae5_count": 20, "home_run_200bps_count": 2},
            "verdicts": {"exit_efficiency_status": "FAIL"},
        },
    )
    _write_json(
        reports_root / "truth_entry_skipability_pressure_v1.json",
        {
            "completed_zero_trade_runs": 3,
            "candidate_rich_zero_trade_runs": 3,
            "verdicts": {"zero_trade_acceptance_status": "FAIL"},
        },
    )
    _write_json(
        reports_root / "truth_continuous_market_opportunity_v1.json",
        {
            "opportunity_rich_zero_trade_runs_anchor": ["r1"],
            "verdicts": {"zero_trade_opportunity_rich_outlier_status": "FAIL"},
        },
    )
    _write_json(
        reports_root / "truth_management_rl_readiness_v1.json",
        {
            "downstream_management_ready": True,
            "downstream_runtime_recovery_fallback_detected": False,
        },
    )
    _write_json(
        reports_root / "truth_management_coarse_teacher_summary_v1.json",
        {
            "row_count_v1": 90,
            "binary_teacher_target_summary_v1": {"eligible_rows_v1": 40},
            "feedback_action_balance_status_v1": {"HOLD": "BALANCED_POSITIVE_AND_NEGATIVE"},
        },
    )
    _write_json(
        reports_root / "truth_management_coarse_feedback_benchmark_summary_v1.json",
        {
            "universe_counts_v1": {"split_counts_v1": {"TRAIN": 20}},
            "current_bucket_holdout_brier_improvement_v1": -0.1,
            "shadow_promotion_guard_v1": "BENCHMARK_ONLY_NO_RETRAIN",
        },
    )
    _write_json(
        reports_root / "truth_management_next_step_priority_v1.json",
        {
            "gates_v1": {"rl_training_gate_v1": "DO_NOT_START_RL_TRAINING_YET"},
            "recommended_execution_order_v1": ["a", "b"],
        },
    )
    _write_json(
        ext_dir / "shadow_meta_all_trade_review_management_policy_logging_summary_v1.json",
        {
            "instrumentation_status_v1": "BEVIST",
            "observed_action_counts_v1": {"HOLD": 80, "EXIT_NOW": 10},
            "behavior_policy_readiness_v1": "IKKE_ETABLERT",
            "propensity_readiness_v1": "IKKE_ETABLERT",
        },
    )
    _write_json(
        ext_dir / "shadow_meta_all_trade_review_management_regime_overlay_summary_v1.json",
        {
            "regime_consistency_status_v1": "IKKE_ETABLERT",
            "outcome_advantage_status_v1": "INDIKERT",
        },
    )
    _write_json(
        ext_dir / "shadow_meta_all_trade_review_management_outcome_quality_regime_audit_summary_v1.json",
        {"slice_count_v1": 4},
    )
    _write_json(
        r8_dir / "shadow_meta_all_trade_review_management_rl_readiness_status_v1.json",
        {"MANAGEMENT_RL_READINESS_STATUS": "OFFLINE_RL_READINESS_SUBSTRATE_ONLY"},
    )
    _write_json(
        r8_dir / "shadow_meta_all_trade_review_management_rl_sequence_status_v1.json",
        {"MANAGEMENT_RL_SEQUENCE_BLOCKER_STATUS": "NEXT_STEP_LINKS_MISSING_FOR_SOME_REALIZED_HOLD_ROWS"},
    )
    _write_json(
        r8_dir / "shadow_meta_all_trade_review_management_bandit_status_v1.json",
        {
            "MANAGEMENT_BANDIT_DM_CANDIDATE_ROW_COUNT_V1": 90,
            "MANAGEMENT_BANDIT_HOLD_EPISODE_RETURN_ROW_COUNT_V1": 80,
            "MANAGEMENT_BANDIT_EXIT_LOCAL_REWARD_ROW_COUNT_V1": 10,
            "MANAGEMENT_BANDIT_PROPENSITY_STATUS": "PROPENSITY_NOT_ESTABLISHED",
        },
    )
    _write_json(
        r8_dir / "shadow_meta_all_trade_review_management_exit_local_status_v1.json",
        {"MANAGEMENT_EXIT_LOCAL_BASELINE_STATUS": "EXIT_LOCAL_BASELINE_TRAINED", "BINARY_TARGET_STATUS_V1": "RUNNABLE"},
    )
    _write_json(
        r8_dir / "shadow_meta_all_trade_review_entry_actualization_status_v1.json",
        {"ENTRY_TO_MANAGEMENT_HANDOFF_STATUS": "HANDOFF_COVERAGE_NOT_FULLY_ESTABLISHED"},
    )
    _write_json(
        r8_dir / "shadow_meta_all_trade_review_entry_actual_take_to_management_handoff_summary_v1.json",
        {
            "management_core_v4_present_count_v1": 60,
            "management_bridge_diagnostic_only_count_v1": 2,
        },
    )

    contract = build_management_inventory_contract(reports_root)
    assert contract["headline_v1"]["trade_count_v1"] == 100
    assert contract["headline_v1"]["rl_policy_ready_v1"] is False
    assert len(contract["inventory_rows_v1"]) >= 10
    assert "management_review_labels_v1" in contract["labels_inventory_v1"]
    assert contract["top_improvements_v1"][0]["track_v1"] == "behavior_policy_and_propensity"

    written = write_management_inventory_contract(reports_root)
    assert Path(written["json_path"]).exists()
    assert Path(written["md_path"]).exists()
