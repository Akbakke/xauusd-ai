from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts.materialize_truth_rl_recommendation_candidate_v1 import (
    RECOMMENDATION_BATCH_REPLAY,
    RECOMMENDATION_STATUS,
    RECOMMENDATION_TRADE_VIEW,
    TOP_LEVEL_SUMMARY,
    build_rl_recommendation_candidate_payload,
    materialize_truth_rl_recommendation_candidate,
)


def _run_ids(count: int) -> list[str]:
    starts = pd.date_range("2025-01-01", periods=count, freq="7D")
    ends = starts + pd.Timedelta(days=7)
    return [
        f"E2E_SANITY_ORDERFIX_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        for start, end in zip(starts, ends)
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, list[str]]:
    reports_root = tmp_path / "reports"
    runs_root = reports_root / "runs"
    review_dir = reports_root / "review"
    unified_dir = reports_root / "unified"
    extension_dir = reports_root / "recommendation"
    review_dir.mkdir(parents=True)
    unified_dir.mkdir(parents=True)
    runs_root.mkdir(parents=True)
    run_ids = _run_ids(16)
    for run_id in run_ids:
        (runs_root / run_id).mkdir()

    ledger_df = pd.DataFrame(
        [
            {
                "run_id": run_ids[0],
                "candidate_uid": "cand-skip",
                "trade_uid": "trade-skip",
                "trade_id": "1",
                "realized_pnl_bps": -12.0,
                "hindsight_should_skip_trade_v1": True,
                "hindsight_should_hold_longer_v1": True,
                "hindsight_should_exit_earlier_v1": True,
                "hindsight_skip_trade_avoided_loss_bps_v1": 12.0,
                "hindsight_hold_longer_extra_value_bps_v1": 100.0,
                "hindsight_exit_earlier_saved_bps_v1": 50.0,
            },
            {
                "run_id": run_ids[1],
                "candidate_uid": "cand-exit",
                "trade_uid": "trade-exit",
                "trade_id": "2",
                "realized_pnl_bps": 1.0,
                "hindsight_should_skip_trade_v1": False,
                "hindsight_should_hold_longer_v1": True,
                "hindsight_should_exit_earlier_v1": True,
                "hindsight_skip_trade_avoided_loss_bps_v1": 0.0,
                "hindsight_hold_longer_extra_value_bps_v1": 100.0,
                "hindsight_exit_earlier_saved_bps_v1": 5.0,
            },
            {
                "run_id": run_ids[2],
                "candidate_uid": "cand-hold",
                "trade_uid": "trade-hold",
                "trade_id": "3",
                "realized_pnl_bps": 2.0,
                "hindsight_should_skip_trade_v1": False,
                "hindsight_should_hold_longer_v1": True,
                "hindsight_should_exit_earlier_v1": False,
                "hindsight_skip_trade_avoided_loss_bps_v1": 0.0,
                "hindsight_hold_longer_extra_value_bps_v1": 7.0,
                "hindsight_exit_earlier_saved_bps_v1": 0.0,
            },
            {
                "run_id": run_ids[3],
                "candidate_uid": "cand-keep",
                "trade_uid": "trade-keep",
                "trade_id": "4",
                "realized_pnl_bps": 3.0,
                "hindsight_should_skip_trade_v1": False,
                "hindsight_should_hold_longer_v1": False,
                "hindsight_should_exit_earlier_v1": False,
                "hindsight_skip_trade_avoided_loss_bps_v1": 0.0,
                "hindsight_hold_longer_extra_value_bps_v1": 0.0,
                "hindsight_exit_earlier_saved_bps_v1": 0.0,
            },
        ]
    )
    ledger_df.to_parquet(review_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet", index=False)
    pd.DataFrame({"candidate_uid": ledger_df["candidate_uid"]}).to_parquet(
        unified_dir / "shadow_meta_all_trade_review_rl_unified_episode_view_v1.parquet",
        index=False,
    )
    _write_json(
        unified_dir / "shadow_meta_all_trade_review_rl_unified_observability_summary_v1.json",
        {"failed_check_count_v1": 0},
    )
    _write_json(
        unified_dir / "shadow_meta_all_trade_review_rl_unified_observability_status_v1.json",
        {
            "UNIFIED_RL_OBSERVABILITY_STATUS": "READY_ENTRY_AND_MANAGEMENT_OBSERVABILITY",
            "ENTRY_PROPENSITY_STATUS": "NOT_ESTABLISHED",
            "MANAGEMENT_PROPENSITY_STATUS": "READY_DETERMINISTIC_LOGGED_ACTION_PROPENSITY",
        },
    )
    _write_json(
        reports_root / "truth_entry_skipability_pressure_v1.json",
        {"candidate_rich_zero_trade_run_ids": [run_ids[5], run_ids[6]]},
    )
    _write_json(
        reports_root / "truth_continuous_market_opportunity_v1.json",
        {"opportunity_rich_zero_trade_runs_anchor": [run_ids[6]]},
    )
    return reports_root, review_dir, unified_dir, extension_dir, run_ids


def test_build_rl_recommendation_candidate_prioritizes_non_overlapping_shadow_deltas(tmp_path: Path) -> None:
    reports_root, review_dir, unified_dir, _, _ = _build_fixture(tmp_path)

    payload = build_rl_recommendation_candidate_payload(
        reports_root=reports_root,
        review_dir=review_dir,
        unified_dir=unified_dir,
        batch_weeks=15,
    )

    summary = payload["summary_v1"]
    trade_view = payload["trade_view_v1_df"]
    batch_replay = payload["batch_replay_v1_df"]

    assert summary["failed_check_count_v1"] == 0
    assert summary["batch_count_v1"] == 2
    assert summary["baseline_trade_count_v1"] == 4
    assert summary["recommendation_counts_v1"] == {
        "SKIP_TRADE": 1,
        "EXIT_EARLIER": 1,
        "HOLD_LONGER": 1,
        "KEEP_BASELINE": 1,
    }
    assert summary["baseline_total_pnl_bps_v1"] == pytest.approx(-6.0)
    assert summary["priority_counterfactual_delta_bps_v1"] == pytest.approx(24.0)
    assert summary["shadow_upper_bound_pnl_bps_v1"] == pytest.approx(18.0)
    assert trade_view.loc[trade_view["trade_id"].eq("1"), "rl_priority_counterfactual_delta_bps_v1"].iloc[0] == 12.0
    assert trade_view.loc[trade_view["trade_id"].eq("2"), "rl_priority_counterfactual_delta_bps_v1"].iloc[0] == 5.0
    assert int(batch_replay["run_count_v1"].sum()) == 16
    assert int(batch_replay["candidate_rich_zero_trade_run_count_v1"].sum()) == 2
    assert int(batch_replay["opportunity_rich_zero_trade_run_count_v1"].sum()) == 1


def test_materialize_rl_recommendation_candidate_writes_contract_artifacts(tmp_path: Path) -> None:
    reports_root, review_dir, unified_dir, extension_dir, _ = _build_fixture(tmp_path)

    result = materialize_truth_rl_recommendation_candidate(
        reports_root,
        review_dir=review_dir,
        unified_dir=unified_dir,
        extension_dir=extension_dir,
        batch_weeks=15,
    )

    assert result["status"]["RL_RECOMMENDATION_CANDIDATE_STATUS"] == "READY_SHADOW_REPLAY_15WEEK"
    assert (extension_dir / RECOMMENDATION_TRADE_VIEW).exists()
    assert (extension_dir / RECOMMENDATION_BATCH_REPLAY).exists()
    assert (extension_dir / RECOMMENDATION_STATUS).exists()
    assert (reports_root / TOP_LEVEL_SUMMARY).exists()


def test_build_rl_recommendation_candidate_hard_fails_missing_required_truth_column(tmp_path: Path) -> None:
    reports_root, review_dir, unified_dir, _, _ = _build_fixture(tmp_path)
    ledger_path = review_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet"
    broken_ledger = pd.read_parquet(ledger_path).drop(columns=["hindsight_should_skip_trade_v1"])
    broken_ledger.to_parquet(ledger_path, index=False)

    with pytest.raises(KeyError, match="hindsight_should_skip_trade_v1"):
        build_rl_recommendation_candidate_payload(
            reports_root=reports_root,
            review_dir=review_dir,
            unified_dir=unified_dir,
            batch_weeks=15,
        )
