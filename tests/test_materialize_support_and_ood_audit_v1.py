from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_support_and_ood_audit_v1 import (
    build_support_and_ood_audit,
    write_support_and_ood_audit_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    iql = root / "IQL_INTEGRATION"
    dataset_dir = iql / "BUILD_MANAGEMENT_BANDIT_DATASET_V1_FIXTURE"
    eval_dir = iql / "RUN_FIRST_BANDIT_RESEARCH_EVAL_V1_FIXTURE"
    wait_dir = iql / "WAIT_STATE_AND_POST_REPLAY_READY_LOCK_V1_FIXTURE"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    wait_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.DataFrame.from_records(
        [
            {
                "row_id": "r1",
                "episode_id": "e1",
                "candidate_uid_exact": "c1",
                "decision_ts": "2026-04-22T00:00:00Z",
                "action": "HOLD",
                "action_id": 0,
                "reward": 1.0,
                "reward_version": "MGMT_BANDIT_REALIZED_PNL_BPS_V1",
                "state_feature_names": "[]",
                "state_vector": "[]",
                "source_policy_version": "fixture",
                "behavior_policy_status": "LOGGED",
                "support_status": "STRONG_FEATURE_SUPPORT",
                "as_of_schema_version": "fixture",
                "hindsight_outcome_backfill_version": "fixture",
                "eligibility_status": "INCLUDED",
                "exclusion_reason": "",
                "provenance_namespace": "fixture",
            },
            {
                "row_id": "r2",
                "episode_id": "e2",
                "candidate_uid_exact": "c2",
                "decision_ts": "2026-04-22T00:01:00Z",
                "action": "HOLD",
                "action_id": 0,
                "reward": 2.0,
                "reward_version": "MGMT_BANDIT_REALIZED_PNL_BPS_V1",
                "state_feature_names": "[]",
                "state_vector": "[]",
                "source_policy_version": "fixture",
                "behavior_policy_status": "LOGGED",
                "support_status": "MIXED_FEATURE_SUPPORT",
                "as_of_schema_version": "fixture",
                "hindsight_outcome_backfill_version": "fixture",
                "eligibility_status": "INCLUDED",
                "exclusion_reason": "",
                "provenance_namespace": "fixture",
            },
            {
                "row_id": "r3",
                "episode_id": "e3",
                "candidate_uid_exact": "c3",
                "decision_ts": "2026-04-22T00:02:00Z",
                "action": "EXIT_NOW",
                "action_id": 1,
                "reward": -1.0,
                "reward_version": "MGMT_BANDIT_REALIZED_PNL_BPS_V1",
                "state_feature_names": "[]",
                "state_vector": "[]",
                "source_policy_version": "fixture",
                "behavior_policy_status": "LOGGED",
                "support_status": "NOT_IN_HIGH_SCORE_HOLD_REVIEW_QUEUE",
                "as_of_schema_version": "fixture",
                "hindsight_outcome_backfill_version": "fixture",
                "eligibility_status": "INCLUDED",
                "exclusion_reason": "",
                "provenance_namespace": "fixture",
            },
            {
                "row_id": "r4",
                "episode_id": "e4",
                "candidate_uid_exact": "c4",
                "decision_ts": "2026-04-22T00:03:00Z",
                "action": "EXIT_NOW",
                "action_id": 1,
                "reward": -2.0,
                "reward_version": "MGMT_BANDIT_REALIZED_PNL_BPS_V1",
                "state_feature_names": "[]",
                "state_vector": "[]",
                "source_policy_version": "fixture",
                "behavior_policy_status": "LOGGED",
                "support_status": "NOT_IN_HIGH_SCORE_HOLD_REVIEW_QUEUE",
                "as_of_schema_version": "fixture",
                "hindsight_outcome_backfill_version": "fixture",
                "eligibility_status": "INCLUDED",
                "exclusion_reason": "",
                "provenance_namespace": "fixture",
            },
        ]
    )
    dataset_path = dataset_dir / "management_bandit_research_dataset_v1.parquet"
    dataset.to_parquet(dataset_path, index=False)
    _write_json(
        dataset_dir / "build_management_bandit_dataset_summary_v1.json",
        {
            "dataset_parquet_v1": str(dataset_path),
            "dataset_verdict_v1": "BANDIT_RESEARCH_DATASET_BUILT_WITH_LIMITATIONS",
            "support_ood_verdict_v1": "SUPPORT_TOO_THIN",
        },
    )
    _write_json(
        dataset_dir / "management_bandit_dataset_profile_v1.json",
        {
            "support_ood_verdict_from_foundation_v1": "SUPPORT_TOO_THIN",
        },
    )
    _write_json(
        dataset_dir / "build_management_bandit_dataset_contract_v1.json",
        {
            "source_paths_v1": {
                "locked_ledger_source_v1": str(root / "ledger.parquet"),
                "management_bandit_dm_view_v1": str(root / "dm.parquet"),
                "management_policy_log_v1": str(root / "policy.parquet"),
            }
        },
    )
    _write_json(
        eval_dir / "run_first_bandit_research_eval_summary_v1.json",
        {
            "support_ood_verdict_v1": "SUPPORT_TOO_THIN",
            "safety_verdict_v1": "NO_POSITIVE_CLAIM_ALLOWED",
            "final_verdict_v1": "WEAK_OR_INCONCLUSIVE_SIGNAL",
        },
    )
    _write_json(
        eval_dir / "first_bandit_failcheck_and_safety_review_v1.json",
        {
            "safety_verdict_v1": "NO_POSITIVE_CLAIM_ALLOWED",
        },
    )
    _write_json(
        wait_dir / "wait_state_post_replay_ready_lock_summary_v1.json",
        {
            "wait_state_verdict_v1": "RESEARCH_ONLY_WAIT_STATE",
        },
    )
    _write_json(root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json", {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"})
    return dataset_dir, eval_dir, wait_dir


def test_support_and_ood_audit_stays_fail_closed(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    dataset_dir, eval_dir, wait_dir = _write_fixture(reports_root)
    output_dir = reports_root / "IQL_INTEGRATION" / "SUPPORT_AND_OOD_AUDIT_V1_FIXTURE"
    built_at = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)

    payload = build_support_and_ood_audit(
        reports_root,
        dataset_dir=dataset_dir,
        eval_dir=eval_dir,
        wait_dir=wait_dir,
        output_dir=output_dir,
        built_at=built_at,
        exit_manager_sha_before="same",
        exit_manager_sha_after="same",
        r6_sha_before="same",
        r6_sha_after="same",
        ledger_sha_before="same",
        ledger_sha_after="same",
    )

    assert "SUPPORT_GLOBALLY_THIN" in payload["support_coverage"]["verdicts_v1"]
    assert "EXIT_NOW_SUPPORT_IS_PRIMARY_RISK" in payload["ood_action_audit"]["verdicts_v1"]
    assert "SEVERE_HOLD_DOMINANCE" in payload["action_imbalance"]["verdicts_v1"]
    assert payload["support_coverage"]["action_specific_support_v1"]["exit_now_non_weak_support_rows_v1"] == 0
    assert payload["subset_scan"]["verdict_v1"] == "LIMITED_SUBSET_RELIEF_ONLY"
    assert "NO_PHASE_UNLOCK" in payload["implications_lock"]["verdicts_v1"]
    assert payload["summary"]["research_only_status_v1"] == "RESEARCH_ONLY_STATUS_UNCHANGED"
    assert payload["summary"]["sequence_iql_status_v1"] == "SEQUENCE_IQL_STILL_BLOCKED"
    assert payload["summary"]["r7_status_v1"] == "R7_STILL_BLOCKED"
    assert payload["consistency"]["failed_check_count_v1"] == 0
    assert payload["non_interference"]["failed_check_count_v1"] == 0

    result = write_support_and_ood_audit_artifacts(
        reports_root,
        dataset_dir=dataset_dir,
        eval_dir=eval_dir,
        wait_dir=wait_dir,
        output_dir=output_dir,
        built_at=built_at,
    )
    assert Path(result["artifact_paths"]["summary"]).exists()
    assert result["summary"]["positive_policy_claim_allowed_v1"] is False
    assert result["status"]["training_executed_v1"] is False
