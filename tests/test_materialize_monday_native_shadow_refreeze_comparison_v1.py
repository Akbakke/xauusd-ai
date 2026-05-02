from __future__ import annotations

import json
from pathlib import Path

from gx1.scripts.materialize_monday_native_shadow_refreeze_comparison_v1 import (
    ACTIVE_BENCHMARK_MATRIX,
    CONSISTENCY_AUDIT,
    CONTRACT,
    MANIFEST,
    REFREEZE_READINESS_MATRIX,
    REPORT,
    STATUS,
    SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_materialize_monday_native_shadow_refreeze_comparison_v1(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()

    ledger_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"
    ledger_dir.mkdir()
    _write_json(
        reports_root / "truth_downstream_canonical_rebuild_v1.json",
        {
            "ledger_dir": str(ledger_dir),
            "steps": [{"step": "all_trade_review_ledger", "status": "ok", "result": {"out_dir": str(ledger_dir)}}],
        },
    )
    _write_json(
        ledger_dir / "shadow_meta_all_trade_review_entry_rl_observability_summary_v1.json",
        {
            "observed_direct_entry_rows_v1": 1689,
            "logged_action_counts_v1": {"TAKE_NOW": 894, "SKIP": 462, "WAIT": 333},
            "opportunity_rich_zero_trade_run_count_v1": 2,
        },
    )
    _write_json(
        ledger_dir / "shadow_meta_all_trade_review_management_rl_readiness_summary_v1.json",
        {"management_rows_v1": 1842, "observation_feature_count_v1": 47},
    )
    _write_json(
        ledger_dir / "shadow_meta_all_trade_review_management_rl_sequence_summary_v1.json",
        {"strict_sequence_row_count_v1": 27},
    )
    _write_json(
        ledger_dir / "shadow_meta_all_trade_review_management_bandit_status_v1.json",
        {"MANAGEMENT_BANDIT_DM_CANDIDATE_ROW_COUNT_V1": 1790, "MANAGEMENT_BANDIT_TRAINER_RECOMMENDATION": "EXIT_LOCAL_REWARD_BASELINE_FIRST"},
    )
    _write_json(
        ledger_dir / "shadow_meta_all_trade_review_management_exit_local_status_v1.json",
        {"MANAGEMENT_EXIT_LOCAL_BASELINE_STATUS": "EXIT_LOCAL_BASELINE_TRAINED"},
    )

    comparator_dir = reports_root / "MONDAY_TOP_PRE_RL_BASELINE_COMPARATOR_V1_20260424T063650Z"
    comparator_dir.mkdir()
    _write_json(
        comparator_dir / "summary_v1.json",
        {
            "monday_trade_foundation_v1": {
                "trade_count": 1852,
                "avg_pnl_bps": -3.18,
                "profit_factor": 0.86,
                "max_drawdown_bps": -18653.25,
            },
            "benchmark_trade_foundation_v1": {
                "trade_count": 1971,
                "avg_pnl_bps": 2.47,
                "profit_factor": 1.13,
                "max_drawdown_bps": -9805.30,
            },
        },
    )

    snapshot_dir = reports_root / "MONDAY_WEDNESDAY_BENCHMARK_SNAPSHOT_V1_20260424T0900Z"
    r6_dir = snapshot_dir / "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
    r6_dir.mkdir(parents=True)
    _write_json(snapshot_dir / "summary_v1.json", {"copied_count_v1": 10})
    _write_json(
        r6_dir / "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json",
        {
            "freeze_id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
            "selected_candidate_v1": {"should_not_take_block_count_v1": 180, "tail_10_50_help_count_v1": 149},
            "batch05_v1": {"precision_v1": 0.9285714285714286},
        },
    )

    extension_dir = reports_root / "comparison_out"
    result = materialize(reports_root, extension_dir=extension_dir)
    assert result["status"]["not_live_gate_v1"] is True
    assert result["summary"]["decision_v1"] == "MONDAY_COMPARE_READY_REFREEZE_CHAIN_BLOCKED_BY_POLICY_LOGGING"
    for artifact in [CONTRACT, ACTIVE_BENCHMARK_MATRIX, REFREEZE_READINESS_MATRIX, SUMMARY, REPORT, MANIFEST, STATUS, CONSISTENCY_AUDIT]:
        assert (extension_dir / artifact).exists()
