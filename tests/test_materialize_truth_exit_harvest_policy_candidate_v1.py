from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts.materialize_truth_exit_harvest_policy_candidate_v1 import (
    HARVEST_BATCH_REPLAY,
    HARVEST_MODEL_TARGET_VIEW,
    HARVEST_POLICY_VIEW,
    HARVEST_STATUS,
    TOP_LEVEL_SUMMARY,
    build_exit_harvest_policy_candidate_payload,
    materialize_truth_exit_harvest_policy_candidate,
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


def _row(
    *,
    run_id: str,
    candidate_uid: str,
    trade_id: str,
    pnl: float,
    peak_mfe: float,
    mae: float,
    giveback: float,
    should_skip: bool,
    should_hold: bool,
    should_exit_earlier: bool,
    skip_avoided: float,
    hold_extra: float,
    exit_saved: float,
    good_clean: str,
    bad_trade: str,
    good_exit: str,
    premature: str,
    late: str,
    exit_reason: str,
    outcome: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "candidate_uid": candidate_uid,
        "trade_uid": f"trade-{trade_id}",
        "trade_id": trade_id,
        "decision_timestamp": "2025-01-01T00:00:00+00:00",
        "entry_timestamp": "2025-01-01T00:01:00+00:00",
        "exit_timestamp": "2025-01-01T01:00:00+00:00",
        "realized_pnl_bps": pnl,
        "mfe_bps": peak_mfe,
        "mae_bps": mae,
        "hindsight_peak_mfe_bps_v1": peak_mfe,
        "hindsight_peak_to_exit_giveback_bps_v1": giveback,
        "hindsight_hold_longer_extra_value_bps_v1": hold_extra,
        "hindsight_exit_earlier_saved_bps_v1": exit_saved,
        "hindsight_skip_trade_avoided_loss_bps_v1": skip_avoided,
        "hindsight_should_skip_trade_v1": should_skip,
        "hindsight_should_hold_longer_v1": should_hold,
        "hindsight_should_exit_earlier_v1": should_exit_earlier,
        "good_trade": "FALSE" if bad_trade == "TRUE" else "TRUE",
        "good_trade_mfe20_mae5": good_clean,
        "bad_trade": bad_trade,
        "good_exit": good_exit,
        "premature_exit": premature,
        "late_exit": late,
        "exit_reason": exit_reason,
        "trade_outcome_class": outcome,
        "session": "US",
        "vol_regime": "HIGH",
        "trend_regime": "TREND_UP",
        "used_for_training": True,
        "used_for_validation": False,
        "used_for_holdout": False,
    }


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, list[str]]:
    reports_root = tmp_path / "reports"
    runs_root = reports_root / "runs"
    review_dir = reports_root / "review"
    recommendation_dir = reports_root / "recommendation"
    extension_dir = reports_root / "harvest"
    runs_root.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    recommendation_dir.mkdir(parents=True)
    run_ids = _run_ids(16)
    for run_id in run_ids:
        (runs_root / run_id).mkdir()

    ledger_df = pd.DataFrame(
        [
            _row(
                run_id=run_ids[0],
                candidate_uid="cand-skip",
                trade_id="1",
                pnl=-20.0,
                peak_mfe=25.0,
                mae=80.0,
                giveback=45.0,
                should_skip=True,
                should_hold=True,
                should_exit_earlier=True,
                skip_avoided=20.0,
                hold_extra=100.0,
                exit_saved=50.0,
                good_clean="FALSE",
                bad_trade="TRUE",
                good_exit="FALSE",
                premature="TRUE",
                late="FALSE",
                exit_reason="CATASTROPHIC_GUARD",
                outcome="cata",
            ),
            _row(
                run_id=run_ids[1],
                candidate_uid="cand-exit",
                trade_id="2",
                pnl=-5.0,
                peak_mfe=40.0,
                mae=60.0,
                giveback=45.0,
                should_skip=False,
                should_hold=False,
                should_exit_earlier=True,
                skip_avoided=0.0,
                hold_extra=0.0,
                exit_saved=15.0,
                good_clean="FALSE",
                bad_trade="FALSE",
                good_exit="FALSE",
                premature="FALSE",
                late="TRUE",
                exit_reason="CATASTROPHIC_GUARD",
                outcome="cata",
            ),
            _row(
                run_id=run_ids[2],
                candidate_uid="cand-homerun",
                trade_id="3",
                pnl=10.0,
                peak_mfe=250.0,
                mae=8.0,
                giveback=240.0,
                should_skip=False,
                should_hold=True,
                should_exit_earlier=False,
                skip_avoided=0.0,
                hold_extra=220.0,
                exit_saved=0.0,
                good_clean="TRUE",
                bad_trade="FALSE",
                good_exit="FALSE",
                premature="TRUE",
                late="FALSE",
                exit_reason="BE_PLUS_FLOOR",
                outcome="positive_exit",
            ),
            _row(
                run_id=run_ids[3],
                candidate_uid="cand-keep",
                trade_id="4",
                pnl=30.0,
                peak_mfe=40.0,
                mae=4.0,
                giveback=10.0,
                should_skip=False,
                should_hold=False,
                should_exit_earlier=False,
                skip_avoided=0.0,
                hold_extra=0.0,
                exit_saved=0.0,
                good_clean="TRUE",
                bad_trade="FALSE",
                good_exit="TRUE",
                premature="FALSE",
                late="FALSE",
                exit_reason="THRESHOLD",
                outcome="positive_exit",
            ),
        ]
    )
    ledger_df.to_parquet(review_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet", index=False)
    recommendation_df = ledger_df[["candidate_uid"]].copy()
    recommendation_df["rl_priority_recommendation_v1"] = [
        "SKIP_TRADE",
        "EXIT_EARLIER",
        "HOLD_LONGER",
        "KEEP_BASELINE",
    ]
    recommendation_df["rl_priority_counterfactual_delta_bps_v1"] = [20.0, 15.0, 220.0, 0.0]
    recommendation_df["rl_priority_entry_skip_delta_bps_v1"] = [20.0, 0.0, 0.0, 0.0]
    recommendation_df["rl_priority_exit_earlier_delta_bps_v1"] = [0.0, 15.0, 0.0, 0.0]
    recommendation_df["rl_priority_hold_longer_delta_bps_v1"] = [0.0, 0.0, 220.0, 0.0]
    recommendation_df["rl_recommendation_semantics_v1"] = "HINDSIGHT_SHADOW_UPPER_BOUND_NOT_LIVE_COUNTERFACTUAL_FILL"
    recommendation_df["unified_episode_coverage_status_v1"] = "COVERED_BY_UNIFIED_ENTRY_EPISODE"
    recommendation_df.to_parquet(
        recommendation_dir / "shadow_meta_all_trade_review_rl_recommendation_candidate_trade_view_v1.parquet",
        index=False,
    )
    _write_json(
        recommendation_dir / "shadow_meta_all_trade_review_rl_recommendation_candidate_summary_v1.json",
        {
            "failed_check_count_v1": 0,
            "baseline_trade_count_v1": 4,
            "priority_counterfactual_delta_bps_v1": 255.0,
        },
    )
    _write_json(
        recommendation_dir / "shadow_meta_all_trade_review_rl_recommendation_candidate_status_v1.json",
        {"RL_RECOMMENDATION_CANDIDATE_STATUS": "READY_SHADOW_REPLAY_15WEEK"},
    )
    return reports_root, review_dir, recommendation_dir, extension_dir, run_ids


def test_build_exit_harvest_policy_candidate_creates_model_targets(tmp_path: Path) -> None:
    reports_root, review_dir, recommendation_dir, _, _ = _build_fixture(tmp_path)

    payload = build_exit_harvest_policy_candidate_payload(
        reports_root=reports_root,
        review_dir=review_dir,
        recommendation_dir=recommendation_dir,
        batch_weeks=15,
    )

    summary = payload["summary_v1"]
    policy_view = payload["harvest_policy_view_v1_df"]
    target_view = payload["model_adjustment_target_view_v1_df"]
    batch_df = payload["batch_replay_v1_df"]

    assert summary["failed_check_count_v1"] == 0
    assert summary["trade_count_v1"] == 4
    assert summary["portfolio_capture_ratio_v1"] == pytest.approx(15.0 / 355.0)
    assert summary["harvest_priority_delta_bps_v1"] == pytest.approx(255.0)
    assert summary["home_run_200bps_opportunity_count_v1"] == 1
    assert summary["exit_harvest_policy_action_counts_v1"]["ENTRY_SUPPRESS_OR_DOWNSIZE"] == 1
    assert summary["exit_harvest_policy_action_counts_v1"]["HOLD_LONGER_HOME_RUN_RUNNER"] == 1
    assert policy_view.loc[
        policy_view["candidate_uid"].eq("cand-homerun"),
        "exit_harvest_policy_action_v1",
    ].iloc[0] == "HOLD_LONGER_HOME_RUN_RUNNER"
    assert target_view.loc[
        target_view["candidate_uid"].eq("cand-skip"),
        "entry_xgb_harvest_label_v1",
    ].iloc[0] == "REJECT_OR_LOW_SIZE"
    assert target_view.loc[
        target_view["candidate_uid"].eq("cand-homerun"),
        "exit_transformer_supervision_label_v1",
    ].iloc[0] == "HOLD_LONGER_OR_RUNNER_TRAIL"
    assert int(batch_df["run_count_v1"].sum()) == 16
    assert int(batch_df["trade_count_v1"].sum()) == 4


def test_materialize_exit_harvest_policy_candidate_writes_artifacts(tmp_path: Path) -> None:
    reports_root, review_dir, recommendation_dir, extension_dir, _ = _build_fixture(tmp_path)

    result = materialize_truth_exit_harvest_policy_candidate(
        reports_root,
        review_dir=review_dir,
        recommendation_dir=recommendation_dir,
        extension_dir=extension_dir,
        batch_weeks=15,
    )

    assert result["status"]["EXIT_HARVEST_POLICY_CANDIDATE_STATUS"] == "READY_FOR_RETRAIN_TARGET_REVIEW"
    assert (extension_dir / HARVEST_POLICY_VIEW).exists()
    assert (extension_dir / HARVEST_MODEL_TARGET_VIEW).exists()
    assert (extension_dir / HARVEST_BATCH_REPLAY).exists()
    assert (extension_dir / HARVEST_STATUS).exists()
    assert (reports_root / TOP_LEVEL_SUMMARY).exists()


def test_exit_harvest_policy_candidate_hard_fails_missing_truth_column(tmp_path: Path) -> None:
    reports_root, review_dir, recommendation_dir, _, _ = _build_fixture(tmp_path)
    ledger_path = review_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet"
    broken = pd.read_parquet(ledger_path).drop(columns=["hindsight_peak_mfe_bps_v1"])
    broken.to_parquet(ledger_path, index=False)

    with pytest.raises(KeyError, match="hindsight_peak_mfe_bps_v1"):
        build_exit_harvest_policy_candidate_payload(
            reports_root=reports_root,
            review_dir=review_dir,
            recommendation_dir=recommendation_dir,
            batch_weeks=15,
        )
