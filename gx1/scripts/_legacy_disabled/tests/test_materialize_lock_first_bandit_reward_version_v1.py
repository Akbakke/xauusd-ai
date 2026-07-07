from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_lock_first_bandit_reward_version_v1 import (
    FIRST_LOCKED_REWARD_VERSION_ID,
    SELECTED_REWARD_NAME,
    build_reward_lock,
    write_reward_lock_artifacts,
)
from gx1.scripts.materialize_iql_readonly_transition_reward_bandit_planning_v1 import FOUNDATION_DIRNAME


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_fixture(root: Path) -> tuple[Path, Path]:
    foundation = root / FOUNDATION_DIRNAME
    contract_lock = root / "IQL_INTEGRATION" / "IQL_REWARD_COMPARATOR_AND_BANDIT_CONTRACT_LOCK_V1_FIXTURE"
    foundation.mkdir(parents=True)
    contract_lock.mkdir(parents=True)

    _write_json(
        foundation / "iql_foundation_mdp_contract_v1.json",
        {"source_truth_v1": {"locked_ledger_source_file_v1": str(root / "ledger.parquet")}},
    )
    pd.DataFrame(
        [
            {
                "reward_candidate_v1": SELECTED_REWARD_NAME,
                "distribution_min_v1": -100.0,
                "distribution_max_v1": 200.0,
                "distribution_count_v1": 1796,
                "coverage_rate_v1": 1.0,
            }
        ]
    ).to_csv(foundation / "iql_foundation_reward_audit_v1.csv", index=False)

    _write_json(
        contract_lock / "iql_reward_comparator_bandit_contract_lock_summary_v1.json",
        {
            "bandit_dataset_contract_verdict_v1": "READY_TO_BUILD_AFTER_REWARD_LOCK",
            "training_harness_status_v1": "NOT_READY_FOR_IQL_TRAINING",
        },
    )
    pd.DataFrame(
        [
            {
                "reward_candidate_v1": "REALIZED_PNL_REWARD",
                "formula_draft_v1": "terminal_realized_pnl_bps",
                "sign_direction_v1": "MAXIMIZE",
                "required_inputs_v1": '["terminal_realized_pnl_bps"]',
                "coverage_rate_v1": 1.0,
                "coverage_count_v1": 1796,
                "hindsight_only_v1": True,
                "leakage_risk_v1": "LOW_IF_USED_ONLY_AS_REWARD",
                "hard_verdict_v1": "LOCKABLE_AFTER_REVIEW",
            },
            {
                "reward_candidate_v1": "MFE_CAPTURE_REWARD",
                "formula_draft_v1": "terminal_realized_pnl_bps / max(hindsight_peak_mfe_bps, eps)",
                "sign_direction_v1": "MAXIMIZE",
                "required_inputs_v1": '["terminal_realized_pnl_bps", "hindsight_peak_mfe_bps"]',
                "coverage_rate_v1": 1.0,
                "coverage_count_v1": 1796,
                "hindsight_only_v1": True,
                "leakage_risk_v1": "MEDIUM_HINDSIGHT_PATH_METRIC_REWARD_ONLY",
                "hard_verdict_v1": "LOCKABLE_AFTER_REVIEW",
            },
            {
                "reward_candidate_v1": "MAE_PENALTY_REWARD",
                "formula_draft_v1": "-abs(terminal_mae_bps)",
                "sign_direction_v1": "MAXIMIZE_LESS_NEGATIVE",
                "required_inputs_v1": '["terminal_mae_bps"]',
                "coverage_rate_v1": 1.0,
                "coverage_count_v1": 1796,
                "hindsight_only_v1": True,
                "leakage_risk_v1": "LOW_IF_USED_ONLY_AS_REWARD",
                "hard_verdict_v1": "LOCKABLE_AFTER_REVIEW",
            },
            {
                "reward_candidate_v1": "GIVEBACK_PENALTY_REWARD",
                "formula_draft_v1": "-hindsight_peak_to_exit_giveback_bps",
                "sign_direction_v1": "MAXIMIZE_LESS_NEGATIVE",
                "required_inputs_v1": '["hindsight_peak_to_exit_giveback_bps"]',
                "coverage_rate_v1": 1.0,
                "coverage_count_v1": 1796,
                "hindsight_only_v1": True,
                "leakage_risk_v1": "MEDIUM_HINDSIGHT_PATH_METRIC_REWARD_ONLY",
                "hard_verdict_v1": "LOCKABLE_AFTER_REVIEW",
            },
            {
                "reward_candidate_v1": "TAIL_CONTROL_REWARD",
                "formula_draft_v1": "terminal_realized_pnl_bps - 25*bad_trade - 75*cata_loser",
                "sign_direction_v1": "MAXIMIZE",
                "required_inputs_v1": '["terminal_realized_pnl_bps", "bad_trade", "cata_loser"]',
                "coverage_rate_v1": 1.0,
                "coverage_count_v1": 1796,
                "hindsight_only_v1": True,
                "leakage_risk_v1": "LOW_IF_USED_ONLY_AS_REWARD",
                "hard_verdict_v1": "LOCKABLE_AFTER_REVIEW",
            },
        ]
    ).to_csv(contract_lock / "iql_reward_contract_lock_review_v1.csv", index=False)
    _write_json(contract_lock / "iql_baseline_comparator_and_failcheck_lock_v1.json", {"lock_id_v1": "BASELINE_COMPARATOR_AND_FAILCHECK_LOCK_V1"})
    _write_json(root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json", {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"})
    return foundation, contract_lock


def test_lock_first_bandit_reward_version_is_fail_closed_and_readonly(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    foundation, contract_lock = _write_fixture(reports_root)
    output_dir = reports_root / "IQL_INTEGRATION" / "LOCK_FIRST_BANDIT_REWARD_VERSION_V1_FIXTURE"
    built_at = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)

    payload = build_reward_lock(
        reports_root,
        foundation_dir=foundation,
        contract_lock_dir=contract_lock,
        output_dir=output_dir,
        built_at=built_at,
        exit_manager_sha_before="same",
        exit_manager_sha_after="same",
        r6_sha_before="same",
        r6_sha_after="same",
    )

    assert payload["selection_lock"]["scalar_reward_locked_v1"] is True
    assert payload["selection_lock"]["selected_reward_name_v1"] == SELECTED_REWARD_NAME
    assert payload["selection_lock"]["selected_reward_version_id_v1"] == FIRST_LOCKED_REWARD_VERSION_ID
    assert payload["reward_contract"]["verdict_v1"] == "LOCKED_FOR_MANAGEMENT_BANDIT_RESEARCH_ONLY"
    assert payload["readiness_update"]["management_bandit_dataset_new_status_v1"] == "READY_TO_BUILD_WITH_LOCKED_REWARD"
    assert payload["readiness_update"]["sequence_iql_still_blocked_v1"] is True
    assert payload["non_interference"]["failed_check_count_v1"] == 0
    assert payload["consistency_df"]["status_v1"].eq("PASS").all()

    result = write_reward_lock_artifacts(
        reports_root,
        foundation_dir=foundation,
        contract_lock_dir=contract_lock,
        output_dir=output_dir,
        built_at=built_at,
    )

    assert Path(result["artifact_paths"]["reward_contract"]).exists()
    assert result["summary"]["reward_version_id_v1"] == FIRST_LOCKED_REWARD_VERSION_ID
    assert result["summary"]["bandit_dataset_ready_to_build_v1"] is True
    assert result["status"]["training_executed_v1"] is False
    assert result["status"]["r7_started_v1"] is False
