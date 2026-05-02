from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gx1.scripts.materialize_wait_state_and_post_replay_ready_lock_v1 import (
    build_wait_state_lock,
    write_wait_state_lock_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_fixture(root: Path) -> Path:
    eval_dir = root / "IQL_INTEGRATION" / "RUN_FIRST_BANDIT_RESEARCH_EVAL_V1_FIXTURE"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        eval_dir / "run_first_bandit_research_eval_summary_v1.json",
        {
            "final_verdict_v1": "WEAK_OR_INCONCLUSIVE_SIGNAL",
            "safety_verdict_v1": "NO_POSITIVE_CLAIM_ALLOWED",
            "signal_polarity_v1": "INCONCLUSIVE",
            "hold_rows_v1": 1751,
            "exit_now_rows_v1": 45,
            "support_ood_verdict_v1": "SUPPORT_TOO_THIN",
            "hard_gate_block_count_v1": 7,
            "failcheck_indeterminate_count_v1": 14,
            "sequence_iql_still_blocked_v1": True,
            "r7_still_blocked_v1": True,
        },
    )
    _write_json(
        eval_dir / "first_bandit_eval_final_verdict_v1.json",
        {
            "final_verdict_v1": "WEAK_OR_INCONCLUSIVE_SIGNAL",
            "signal_polarity_v1": "INCONCLUSIVE",
        },
    )
    _write_json(
        eval_dir / "first_bandit_failcheck_and_safety_review_v1.json",
        {
            "safety_verdict_v1": "NO_POSITIVE_CLAIM_ALLOWED",
            "hard_gate_block_count_v1": 7,
            "fail_count_v1": 1,
            "indeterminate_count_v1": 14,
        },
    )
    _write_json(
        eval_dir / "post_first_bandit_eval_status_update_v1.json",
        {
            "sequence_iql_still_blocked_v1": True,
            "r7_status_unchanged_blocked_v1": True,
        },
    )
    _write_json(
        eval_dir / "run_first_bandit_research_eval_manifest_v1.json",
        {
            "source_paths_v1": {
                "dataset_dir_v1": str(root / "dataset"),
                "eval_prep_dir_v1": str(root / "eval_prep"),
                "locked_ledger_source_v1": str(root / "ledger.parquet"),
            }
        },
    )
    _write_json(root / "truth_r6_shadow_freeze_and_path_dynamics_unblock_v1.json", {"freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"})
    return eval_dir


def test_wait_state_lock_keeps_replay_as_main_path(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    eval_dir = _write_fixture(reports_root)
    output_dir = reports_root / "IQL_INTEGRATION" / "WAIT_STATE_AND_POST_REPLAY_READY_LOCK_V1_FIXTURE"
    built_at = datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc)

    payload = build_wait_state_lock(
        reports_root,
        eval_dir=eval_dir,
        output_dir=output_dir,
        built_at=built_at,
        exit_manager_sha_before="same",
        exit_manager_sha_after="same",
        r6_sha_before="same",
        r6_sha_after="same",
    )

    assert payload["summary"]["wait_state_verdict_v1"] == "RESEARCH_ONLY_WAIT_STATE"
    assert payload["summary"]["main_priority_v1"] == "WAIT_FOR_REPLAY_THEN_REBUILD_CHAIN"
    assert "NO_POSITIVE_POLICY_CLAIM" in payload["wait_state"]["wait_state_verdicts_v1"]
    assert "LIMITED_BANDIT_RESEARCH_ONLY" in payload["bandit_limit"]["verdicts_v1"]
    assert payload["post_replay_gate"]["mode_v1"] == "PLAN_ONLY_NOT_EXECUTED_NOW"
    assert "R7_STILL_BLOCKED" in payload["r7_sequence"]["verdicts_v1"]
    assert "SEQUENCE_IQL_STILL_BLOCKED" in payload["r7_sequence"]["verdicts_v1"]
    assert payload["optional_tasks"]["verdict_v1"] == "ONLY_SMALL_RESEARCH_ALLOWED_WHILE_WAITING"
    assert payload["consistency"]["failed_check_count_v1"] == 0
    assert payload["non_interference"]["failed_check_count_v1"] == 0

    result = write_wait_state_lock_artifacts(
        reports_root,
        eval_dir=eval_dir,
        output_dir=output_dir,
        built_at=built_at,
    )
    assert Path(result["artifact_paths"]["summary"]).exists()
    assert result["summary"]["sequence_iql_status_v1"] == "SEQUENCE_IQL_STILL_BLOCKED"
    assert result["summary"]["r7_status_v1"] == "R7_STILL_BLOCKED"
    assert result["status"]["training_executed_v1"] is False
