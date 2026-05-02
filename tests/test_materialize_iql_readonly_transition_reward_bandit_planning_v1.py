from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import (
    FOUNDATION_DIRNAME,
    build_iql_readonly_planning,
    write_iql_readonly_planning_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_foundation_fixture(root: Path) -> Path:
    foundation_dir = root / FOUNDATION_DIRNAME
    foundation_dir.mkdir(parents=True)
    ledger_path = root / "canonical" / "shadow_meta_all_trade_review_ledger_closed_trades.parquet"
    ledger_path.parent.mkdir(parents=True)
    pd.DataFrame({"trade_uid": [f"t{i}" for i in range(1971)]}).to_parquet(ledger_path, index=False)

    _write_json(root / "truth_iql_foundation_mdp_contract_and_dataset_scaffold_v1.json", {"artifact_dir_v1": str(foundation_dir)})
    _write_json(
        foundation_dir / "iql_foundation_mdp_contract_v1.json",
        {
            "source_truth_v1": {
                "locked_ledger_source_file_v1": str(ledger_path),
            }
        },
    )
    _write_json(
        foundation_dir / "iql_foundation_summary_v1.json",
        {
            "management_mdp_verdict_v1": "PARTIAL_TRANSITIONS",
            "entry_iql_suitability_v1": "ENTRY_IQL_NOT_READY",
            "full_sequence_ready_transition_count_v1": 45,
            "bandit_only_row_count_v1": 1751,
            "hold_to_next_state_transition_count_v1": 0,
            "support_ood_verdict_v1": "SUPPORT_TOO_THIN",
            "bandit_support_verdict_v1": "SUPPORT_WEAK_BUT_USABLE",
            "training_harness_status_v1": "NOT_READY_FOR_IQL_TRAINING",
        },
    )
    _write_json(foundation_dir / "iql_foundation_status_v1.json", {"status_v1": "MATERIALIZED"})
    _write_json(
        foundation_dir / "iql_foundation_transition_linkage_audit_v1.json",
        {
            "full_sequence_ready_transition_count_v1": 45,
            "exact_next_management_state_count_v1": 0,
            "hold_to_next_state_transition_count_v1": 0,
            "bandit_only_row_count_v1": 1751,
            "primary_transition_gap_v1": "HOLD_NEXT_STATE_LINKS_NOT_LOGGED",
        },
    )
    _write_json(
        foundation_dir / "iql_foundation_support_ood_audit_v1.json",
        {
            "overall_support_verdict_v1": "SUPPORT_TOO_THIN",
            "bandit_support_verdict_v1": "SUPPORT_WEAK_BUT_USABLE",
        },
    )
    _write_json(
        foundation_dir / "iql_foundation_dataset_schema_v1.json",
        {
            "fields_v1": [
                {"field_name_v1": "next_state_vector", "status_v1": "PARTIAL"},
            ]
        },
    )
    _write_json(
        foundation_dir / "iql_foundation_baseline_comparator_spec_v1.json",
        {
            "baseline_calibration_status_v1": "PENDING_EXTERNAL_CALIBRATION",
            "baseline_comparator_presence_v1": {
                "no_rl_baseline_v1": {"status_v1": "REFERENCE_REGISTERED"},
                "r6_frozen_shadow_fallback_v1": {"status_v1": "REFERENCE_REGISTERED"},
            },
        },
    )
    _write_json(foundation_dir / "iql_foundation_training_harness_stub_v1.json", {"status_v1": "NOT_READY_FOR_IQL_TRAINING"})
    pd.DataFrame(
        [
            {
                "domain_v1": "ENTRY_IQL_FOUNDATION",
                "verdict_v1": "BANDIT_ONLY_READY",
            }
        ]
    ).to_csv(foundation_dir / "iql_foundation_mdp_domain_feasibility_audit_v1.csv", index=False)
    pd.DataFrame(
        [
            {
                "reward_candidate_v1": "REALIZED_PNL_REWARD",
                "formula_v1": "terminal_realized_pnl_bps",
                "coverage_rate_v1": 1.0,
                "distribution_count_v1": 1796,
                "hindsight_only_v1": True,
                "leakage_risk_v1": "LOW_IF_USED_ONLY_AS_REWARD",
                "verdict_v1": "USABLE_FOR_OFFLINE_RESEARCH",
            },
            {
                "reward_candidate_v1": "RUNNER_DAMAGE_PENALTY",
                "formula_v1": "-max(hindsight_hold_longer_extra_value_bps, 0)",
                "coverage_rate_v1": 1.0,
                "distribution_count_v1": 1796,
                "hindsight_only_v1": True,
                "leakage_risk_v1": "HIGH_COUNTERFACTUAL_HINDSIGHT_LOCALITY_NOT_LOCKED",
                "verdict_v1": "AUDIT_ONLY",
            },
        ]
    ).to_csv(foundation_dir / "iql_foundation_reward_audit_v1.csv", index=False)

    _write_json(root / "truth_r5_2_shadow_freeze_and_r6_failure_backlog_v1.json", {"freeze_id_v1": "R5_2_SHADOW_FREEZE_10176B84DF46B1F0_V1"})
    _write_json(root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json", {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"})
    return foundation_dir


def test_readonly_iql_planning_materializes_without_training_or_replay(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    foundation_dir = _write_foundation_fixture(reports_root)
    output_dir = reports_root / "IQL_READINESS" / "IQL_READONLY_TRANSITION_REWARD_BANDIT_PLANNING_V1_TEST"
    built_at = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)

    payload = build_iql_readonly_planning(
        reports_root,
        foundation_dir=foundation_dir,
        output_dir=output_dir,
        built_at=built_at,
        exit_manager_sha_before="same",
        exit_manager_sha_after="same",
    )

    assert payload["summary"]["management_status_v1"] == "PARTIAL_TRANSITIONS"
    assert payload["summary"]["strict_transition_count_v1"] == 45
    assert payload["summary"]["bandit_ready_row_count_v1"] == 1751
    assert payload["summary"]["hold_to_next_state_transition_count_v1"] == 0
    assert payload["summary"]["r7_started_v1"] is False
    assert payload["transition_gap"]["diagnosis_v1"] == "LOGGING_GAP_AND_SINGLE_SNAPSHOT_PROBLEM_INDICATED"
    assert set(payload["reward_draft_df"]["draft_status_v1"]) == {"LOCKABLE_AFTER_REVIEW", "AUDIT_ONLY"}
    assert payload["boundary_lock"]["r7_v1"]["status_v1"] == "NOT_STARTED"
    assert payload["non_interference"]["failed_check_count_v1"] == 0

    result = write_iql_readonly_planning_artifacts(
        reports_root,
        foundation_dir=foundation_dir,
        output_dir=output_dir,
        built_at=built_at,
    )

    assert Path(result["artifact_paths"]["summary"]).exists()
    assert Path(result["artifact_paths"]["non_interference_audit"]).exists()
    assert result["status"]["training_executed_v1"] is False
    assert result["status"]["replay_touched_v1"] is False
    assert result["status"]["r7_started_v1"] is False
