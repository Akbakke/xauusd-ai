from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.audit_truth_trade_foundation_quality import build_trade_foundation_quality_summary


def _write_trade_frame(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_build_trade_foundation_quality_summary_aggregates_profitability(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    runs_root = reports_root / "runs"
    runs_root.mkdir(parents=True)

    run_a = runs_root / "E2E_SANITY_ORDERFIX_20250101_20250108"
    run_a.mkdir()
    _write_trade_frame(
        run_a / "trade_outcomes_E2E_SANITY_ORDERFIX_20250101_20250108_MERGED.parquet",
        [
            {
                "trade_id": "a1",
                "pnl_bps": 20.0,
                "mfe_bps": 40.0,
                "mae_bps": -4.0,
                "post_exit_mfe_bps": 5.0,
                "early_exit_regret": False,
                "session": "EU",
                "open_ts_utc": "2025-01-01T08:00:00+00:00",
                "close_ts_utc": "2025-01-01T08:30:00+00:00",
            },
            {
                "trade_id": "a2",
                "pnl_bps": -10.0,
                "mfe_bps": 15.0,
                "mae_bps": -25.0,
                "post_exit_mfe_bps": 8.0,
                "early_exit_regret": True,
                "session": "EU",
                "open_ts_utc": "2025-01-01T09:00:00+00:00",
                "close_ts_utc": "2025-01-01T09:30:00+00:00",
            },
        ],
    )

    run_b = runs_root / "E2E_SANITY_ORDERFIX_20250108_20250115"
    run_b.mkdir()
    _write_trade_frame(
        run_b / "trade_outcomes_E2E_SANITY_ORDERFIX_20250108_20250115_MERGED.parquet",
        [
            {
                "trade_id": "b1",
                "pnl_bps": 30.0,
                "mfe_bps": 60.0,
                "mae_bps": -3.0,
                "post_exit_mfe_bps": 12.0,
                "early_exit_regret": False,
                "session": "US",
                "open_ts_utc": "2025-01-08T14:00:00+00:00",
                "close_ts_utc": "2025-01-08T14:30:00+00:00",
            }
        ],
    )

    summary = build_trade_foundation_quality_summary(reports_root, sample_limit=5)

    assert summary["trade_count"] == 3
    assert summary["profitability"]["total_pnl_bps"] == 40.0
    assert summary["profitability"]["profit_factor"] == 5.0
    assert summary["quality_flags"]["clean_good_trade_mfe20_mae5_count"] == 2
    assert summary["quality_flags"]["home_run_200bps_count"] == 0
    assert summary["exit_efficiency"]["early_exit_regret_count"] == 1
    assert summary["hold_longer_pressure"]["meaningful_extra_value_10bps_count"] == 1
    assert summary["hold_longer_pressure"]["large_extra_value_50bps_count"] == 0
    assert summary["verdicts"]["profitability_status"] == "PASS"
    assert summary["worst_weeks_top10"][0]["run_id"] == "E2E_SANITY_ORDERFIX_20250101_20250108"


def test_build_trade_foundation_quality_summary_supports_monday_top_level_runs(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    reports_root.mkdir(parents=True)

    run_dir = reports_root / "TRUTH_MONFRI_WEEK_20260413_20260420"
    run_dir.mkdir()
    _write_trade_frame(
        run_dir / "trade_outcomes_TRUTH_MONFRI_WEEK_20260413_20260420_MERGED.parquet",
        [
            {
                "trade_id": "m1",
                "pnl_bps": 12.0,
                "mfe_bps": 35.0,
                "mae_bps": -4.0,
                "post_exit_mfe_bps": 6.0,
                "early_exit_regret": False,
                "session": "OVERLAP",
                "open_ts_utc": "2026-04-13T08:00:00+00:00",
                "close_ts_utc": "2026-04-13T08:30:00+00:00",
            }
        ],
    )

    summary = build_trade_foundation_quality_summary(reports_root, sample_limit=5)

    assert summary["trade_count"] == 1
    assert summary["profitability"]["total_pnl_bps"] == 12.0
    assert summary["best_weeks_top10"][0]["start_date"] == "2026-04-13"


def test_management_readiness_detects_namespaced_review_dir(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    runs_root = reports_root / "runs"
    runs_root.mkdir(parents=True)
    run_dir = runs_root / "E2E_SANITY_ORDERFIX_20250101_20250108"
    run_dir.mkdir()
    (run_dir / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "trade_uid": "u1",
                "trade_id": "t1",
                "pnl_bps": 10.0,
                "mfe_bps": 30.0,
                "mae_bps": -2.0,
                "exit_reason": "TP",
            }
        ]
    ).to_parquet(run_dir / "trade_outcomes_E2E_SANITY_ORDERFIX_20250101_20250108_MERGED.parquet", index=False)
    pd.DataFrame(
        [
            {
                "trade_uid": "u1",
                "decision": "LONG",
                "trainable_mask_v1": True,
                "decision_ts_utc": "2025-01-01T08:00:00+00:00",
                "side": "LONG",
                "session": "EU",
                "p_long": 0.7,
                "p_short": 0.1,
                "p_flat": 0.2,
                "p_hat": 0.7,
                "margin": 0.5,
                "uncertainty_score": 0.1,
                "entry_spread_bps": 5.0,
                "open_ts_utc": "2025-01-01T08:00:00+00:00",
                "close_ts_utc": "2025-01-01T08:30:00+00:00",
                "pnl_bps": 10.0,
                "mfe_bps": 30.0,
                "mae_bps": -2.0,
                "bars_in_trade": 6,
                "exit_reason": "TP",
                "mfe_threshold_bps": 10.0,
                "positive_exit": True,
                "cata": False,
                "never_mfe": False,
                "good_mfe_then_rot": False,
                "meta_allow_label_v1": True,
                "mfe_first_n_pred": 12.0,
                "path_quality_pred": 0.8,
            }
        ]
    ).to_parquet(run_dir / "shadow_meta_candidates_E2E_SANITY_ORDERFIX_20250101_20250108_MERGED.parquet", index=False)
    (run_dir / "shadow_meta_provenance_E2E_SANITY_ORDERFIX_20250101_20250108.json").write_text("{}\n", encoding="utf-8")

    review_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260411"
    review_dir.mkdir()
    for name in [
        "shadow_meta_all_trade_review_management_rl_readiness_status_v1.json",
        "shadow_meta_all_trade_review_management_bandit_status_v1.json",
    ]:
        (review_dir / name).write_text("{}\n", encoding="utf-8")
    (
        review_dir / "shadow_meta_all_trade_review_entry_actualization_status_v1.json"
    ).write_text(
        json.dumps({"ENTRY_TO_MANAGEMENT_HANDOFF_STATUS": "HANDOFF_COVERAGE_NOT_FULLY_ESTABLISHED"}) + "\n",
        encoding="utf-8",
    )
    (
        review_dir / "shadow_meta_all_trade_review_entry_actual_take_to_management_handoff_summary_v1.json"
    ).write_text(
        json.dumps(
            {
                "management_handoff_status_counts_v1": {
                    "ACTUAL_TAKE_WITH_PROVABLE_MANAGEMENT_HEAD": 10,
                    "ACTUAL_TAKE_WITHOUT_PROVABLE_MANAGEMENT_HEAD": 2,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"x": 1}]).to_parquet(
        review_dir / "shadow_meta_all_trade_review_management_policy_training_examples_core_v4.parquet",
        index=False,
    )

    from gx1.scripts.audit_truth_management_rl_readiness import build_truth_management_rl_readiness_summary

    summary = build_truth_management_rl_readiness_summary(reports_root, sample_limit=5)

    assert summary["downstream_review_dir"] == str(review_dir.resolve())
    assert bool(summary["downstream_management_ready"]) is True
    assert summary["entry_to_management_handoff_status_v1"] == "HANDOFF_COVERAGE_NOT_FULLY_ESTABLISHED"
    assert summary["actual_take_without_provable_management_head_count_v1"] == 2
    assert bool(summary["entry_to_management_handoff_fully_established_v1"]) is False
    assert summary["verdicts"]["entry_to_management_handoff_status"] == "FAIL"


def test_management_readiness_supports_monday_top_level_runs(tmp_path: Path) -> None:
    reports_root = tmp_path / "truth_root"
    reports_root.mkdir()
    run_dir = reports_root / "TRUTH_MONFRI_WEEK_20260413_20260420"
    run_dir.mkdir()
    (run_dir / "RUN_COMPLETED.json").write_text("{}\n", encoding="utf-8")

    pd.DataFrame(
        [
            {
                "trade_uid": "u1",
                "trade_id": "t1",
                "pnl_bps": 10.0,
                "mfe_bps": 30.0,
                "mae_bps": -2.0,
                "exit_reason": "TP",
            }
        ]
    ).to_parquet(run_dir / "trade_outcomes_TRUTH_MONFRI_WEEK_20260413_20260420_MERGED.parquet", index=False)
    pd.DataFrame(
        [
            {
                "trade_uid": "u1",
                "decision": "LONG",
                "trainable_mask_v1": True,
                "decision_ts_utc": "2026-04-13T08:00:00+00:00",
                "side": "LONG",
                "session": "OVERLAP",
                "p_long": 0.7,
                "p_short": 0.1,
                "p_flat": 0.2,
                "p_hat": 0.7,
                "margin": 0.5,
                "uncertainty_score": 0.1,
                "entry_spread_bps": 5.0,
                "open_ts_utc": "2026-04-13T08:00:00+00:00",
                "close_ts_utc": "2026-04-13T08:30:00+00:00",
                "pnl_bps": 10.0,
                "mfe_bps": 30.0,
                "mae_bps": -2.0,
                "bars_in_trade": 6,
                "exit_reason": "TP",
                "mfe_threshold_bps": 10.0,
                "positive_exit": True,
                "cata": False,
                "never_mfe": False,
                "good_mfe_then_rot": False,
                "meta_allow_label_v1": True,
                "mfe_first_n_pred": 12.0,
                "path_quality_pred": 0.8,
            }
        ]
    ).to_parquet(run_dir / "shadow_meta_candidates_TRUTH_MONFRI_WEEK_20260413_20260420_MERGED.parquet", index=False)
    (run_dir / "shadow_meta_provenance_TRUTH_MONFRI_WEEK_20260413_20260420.json").write_text("{}\n", encoding="utf-8")

    review_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260424"
    review_dir.mkdir()
    for name in [
        "shadow_meta_all_trade_review_management_rl_readiness_status_v1.json",
        "shadow_meta_all_trade_review_management_bandit_status_v1.json",
    ]:
        (review_dir / name).write_text("{}\n", encoding="utf-8")
    pd.DataFrame([{"x": 1}]).to_parquet(
        review_dir / "shadow_meta_all_trade_review_management_policy_training_examples_core_v4.parquet",
        index=False,
    )

    from gx1.scripts.audit_truth_management_rl_readiness import build_truth_management_rl_readiness_summary

    summary = build_truth_management_rl_readiness_summary(reports_root, sample_limit=5)

    assert summary["run_dir_count"] == 1
    assert summary["completed_runs"] == 1
    assert summary["trade_count"] == 1
    assert summary["zero_trade_run_count"] == 0
