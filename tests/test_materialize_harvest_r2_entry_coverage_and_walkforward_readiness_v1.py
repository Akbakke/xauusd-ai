from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_harvest_r2_entry_coverage_and_walkforward_readiness_v1 import (
    AS_OF_TABLE,
    CONSISTENCY_AUDIT,
    COVERAGE_AUDIT,
    HINDSIGHT_LABEL_TABLE,
    MARKDOWN_REPORT,
    READINESS_MATRIX,
    SUMMARY,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _run_ids(count: int) -> list[str]:
    starts = pd.date_range("2025-01-01", periods=count, freq="7D")
    ends = starts + pd.Timedelta(days=7)
    return [
        f"E2E_SANITY_ORDERFIX_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
        for start, end in zip(starts, ends)
    ]


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    reports_root = tmp_path / "reports"
    review_dir = reports_root / "review"
    harvest_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_POLICY_CANDIDATE_R1"
    r2_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_RETRAIN_CANDIDATE_R2"
    extension_dir = reports_root / "readiness"
    runs_root = reports_root / "runs"
    for directory in [review_dir, harvest_dir, r2_dir, runs_root]:
        directory.mkdir(parents=True)

    run_ids = _run_ids(6)
    for run_id in run_ids:
        (runs_root / run_id).mkdir()

    ledger_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    entry_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    management_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    missing_entry = {5, 17, 31}
    missing_raw = {11, *missing_entry}
    missing_management = {7, 13}
    multiclass_labels = ["ALLOW_BASELINE", "PRIORITIZE_CLEAN_RUNNER", "REJECT_OR_LOW_SIZE"]

    for idx in range(36):
        run_id = run_ids[idx % len(run_ids)]
        candidate_uid = f"cand-{idx:03d}"
        should_skip = idx % 6 == 0
        strong_trade = idx % 6 in {1, 2}
        peak_mfe = 85.0 if strong_trade else (35.0 if idx % 3 else 12.0)
        mae_abs = 8.0 if strong_trade else (38.0 if should_skip else 18.0)
        realized = 24.0 if strong_trade else (-12.0 if should_skip else 4.0)
        harvest_action = "ENTRY_SUPPRESS_OR_DOWNSIZE" if should_skip else "HOLD_LONGER_RUNNER_TRAIL"
        exit_reason = "REPLAY_EOF" if idx == 17 else ("CATASTROPHIC_GUARD" if idx == 13 else "THRESHOLD")
        multiclass_target = "REJECT_OR_LOW_SIZE" if should_skip else ("PRIORITIZE_CLEAN_RUNNER" if strong_trade else "ALLOW_BASELINE")
        binary_target = not should_skip
        binary_pred = "FALSE" if should_skip or idx in {4, 8} else "TRUE"
        multiclass_pred = multiclass_target if idx % 5 else "ALLOW_BASELINE"
        train = idx < 22
        validation = 22 <= idx < 29
        holdout = idx >= 29
        ledger_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": candidate_uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "used_for_training": train,
                "used_for_validation": validation,
                "used_for_holdout": holdout,
                "realized_pnl_bps": realized,
                "mfe_bps": peak_mfe,
                "mae_bps": -mae_abs,
                "exit_reason": exit_reason,
                "trade_outcome_class": "never_mfe" if should_skip else "positive_exit",
                "session": "OVERLAP" if idx % 2 else "US",
                "vol_regime": "HIGH" if idx % 3 else "EXTREME",
                "trend_regime": "UP" if idx % 2 else "RANGE",
                "hindsight_entry_decision_review_v1": "SHOULD_SKIP_TRADE" if should_skip else "TAKE_WAS_OK",
                "hindsight_management_review_v1": "SHOULD_HOLD_LONGER" if strong_trade else "SHOULD_EXIT_EARLIER",
                "hindsight_should_skip_trade_v1": should_skip,
                "hindsight_should_hold_longer_v1": strong_trade,
                "hindsight_should_exit_earlier_v1": not strong_trade,
            }
        )
        policy_rows.append(
            {
                "candidate_uid": candidate_uid,
                "baseline_realized_pnl_bps_v1": realized,
                "peak_mfe_bps_v1": peak_mfe,
                "mae_abs_bps_v1": mae_abs,
                "giveback_bps_v1": max(peak_mfe - realized, 0.0),
                "harvest_capture_ratio_v1": max(realized, 0.0) / peak_mfe if peak_mfe else 0.0,
                "harvest_quality_bucket_v1": "ENTRY_FILTER" if should_skip else "EXIT_TOO_EARLY_UNDERHARVEST",
                "exit_harvest_policy_action_v1": harvest_action,
                "rl_priority_entry_skip_delta_bps_v1": 20.0 if should_skip else 0.0,
                "rl_priority_hold_longer_delta_bps_v1": 50.0 if strong_trade else 0.0,
                "rl_priority_exit_earlier_delta_bps_v1": 10.0 if should_skip else 0.0,
                "home_run_200bps_opportunity_v1": False,
                "runner_100bps_opportunity_v1": peak_mfe >= 100.0,
                "runner_50bps_opportunity_v1": peak_mfe >= 50.0,
            }
        )
        target_rows.append(
            {
                "candidate_uid": candidate_uid,
                "management_rl_harvest_action_label_v1": harvest_action,
            }
        )
        if idx not in missing_entry:
            entry_rows.append(
                {
                    "candidate_uid": candidate_uid,
                    "as_of_hour_utc_v1": idx % 24,
                    "as_of_session_v1": "LONDON" if idx % 2 else "NY",
                    "as_of_candidate_tradable_prob_v1": 0.8 if binary_target else 0.2,
                    "support_adverse_first_v1": should_skip,
                    "support_first_meaningful_mfe_bar_index_v1": 2 if strong_trade else 9,
                    "confirmation_delay_minutes_v1": 0.0 if strong_trade else 5.0,
                    "has_provable_confirmation_v1": strong_trade,
                    "wait_followthrough_status_v1": "HELPED" if not should_skip and not strong_trade else "DIRECT",
                    "teacher_should_wait_entry_v1": not should_skip and not strong_trade,
                }
            )
        if idx not in missing_raw:
            raw_rows.append(
                {
                    "candidate_uid": candidate_uid,
                    "as_of_skip_replay_h1_range_compression_ratio_v1": 0.25 if strong_trade else 0.75,
                    "as_of_skip_replay_window_directional_imbalance_60_bps_v1": 2.0 if strong_trade else -1.0,
                    "as_of_skip_candidate_margin_v1": 0.6 if binary_target else -0.4,
                    "as_of_skip_xgb_p_flat_v1": 0.1 if binary_target else 0.8,
                }
            )
        if idx not in missing_management:
            management_rows.append(
                {
                    "candidate_uid_exact_v1": candidate_uid,
                    "as_of_management_core_mfe_bps_so_far_v1": peak_mfe / 2.0,
                    "as_of_management_core_mae_bps_so_far_v1": mae_abs / 2.0,
                }
            )
        prob_true = 0.85 if binary_pred == "TRUE" else 0.15
        prediction_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": candidate_uid,
                "used_for_training": train,
                "used_for_validation": validation,
                "used_for_holdout": holdout,
                "entry_xgb_binary_take_target_v1": binary_target,
                "pred__entry_xgb_binary_take__label_v1": binary_pred,
                "pred__entry_xgb_binary_take__prob_false_v1": 1.0 - prob_true,
                "pred__entry_xgb_binary_take__prob_true_v1": prob_true,
                "entry_xgb_harvest_label_v1": multiclass_target,
                "pred__entry_xgb_harvest_label__label_v1": multiclass_pred,
                "pred__entry_xgb_harvest_label__prob_allow_baseline_v1": 0.8 if multiclass_pred == "ALLOW_BASELINE" else 0.1,
                "pred__entry_xgb_harvest_label__prob_prioritize_clean_runner_v1": 0.8 if multiclass_pred == "PRIORITIZE_CLEAN_RUNNER" else 0.1,
                "pred__entry_xgb_harvest_label__prob_reject_or_low_size_v1": 0.8 if multiclass_pred == "REJECT_OR_LOW_SIZE" else 0.1,
                "pred__entry_xgb_binary_take__feature_available_v1": idx not in missing_entry,
                "peak_mfe_bps_v1": peak_mfe,
                "baseline_realized_pnl_bps_v1": realized,
                "candidate_shadow_action_source_v1": "ENTRY_MODEL_SUPPRESS_FALLBACK" if idx % 7 == 0 else "MANAGEMENT_MODEL",
                "candidate_shadow_action_matches_harvest_target_v1": idx % 7 != 0 or should_skip,
            }
        )

    pd.DataFrame(ledger_rows).to_parquet(review_dir / "shadow_meta_all_trade_review_ledger_closed_trades.parquet", index=False)
    pd.DataFrame(entry_rows).to_parquet(review_dir / "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet", index=False)
    pd.DataFrame(raw_rows).to_parquet(review_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet", index=False)
    pd.DataFrame(management_rows).to_parquet(review_dir / "shadow_meta_all_trade_review_management_rl_row_semantics_view_v1.parquet", index=False)
    pd.DataFrame(policy_rows).to_parquet(harvest_dir / "shadow_meta_all_trade_review_exit_harvest_policy_candidate_trade_view_v1.parquet", index=False)
    pd.DataFrame(target_rows).to_parquet(harvest_dir / "shadow_meta_all_trade_review_harvest_model_adjustment_target_view_v1.parquet", index=False)
    pd.DataFrame(prediction_rows).to_parquet(r2_dir / "shadow_meta_all_trade_review_harvest_retrain_candidate_prediction_view_v1.parquet", index=False)
    _write_json(
        r2_dir / "shadow_meta_all_trade_review_harvest_retrain_candidate_summary_v1.json",
        {
            "candidate_shadow_action_match_rate_v1": 0.9,
            "candidate_to_target_delta_capture_ratio_v1": 0.8,
        },
    )
    return reports_root, review_dir, harvest_dir, r2_dir, extension_dir


def test_materialize_r2_entry_readiness_writes_separated_artifacts(tmp_path: Path) -> None:
    reports_root, review_dir, harvest_dir, r2_dir, extension_dir = _build_fixture(tmp_path)

    result = materialize(
        reports_root,
        review_dir=review_dir,
        harvest_dir=harvest_dir,
        r2_dir=r2_dir,
        extension_dir=extension_dir,
        batch_weeks=3,
        expected_ledger_count=36,
    )

    assert result["status"]["HARVEST_R2_ENTRY_READINESS_STATUS"] == "READY_SHADOW_RETRAIN_AUDIT_NOT_LIVE_GATE"
    assert (extension_dir / AS_OF_TABLE).exists()
    assert (extension_dir / HINDSIGHT_LABEL_TABLE).exists()
    assert (extension_dir / COVERAGE_AUDIT).exists()
    assert (extension_dir / READINESS_MATRIX).exists()
    assert (extension_dir / CONSISTENCY_AUDIT).exists()
    assert (extension_dir / SUMMARY).exists()
    assert (extension_dir / MARKDOWN_REPORT).exists()

    asof = pd.read_parquet(extension_dir / AS_OF_TABLE)
    labels = pd.read_parquet(extension_dir / HINDSIGHT_LABEL_TABLE)
    coverage = pd.read_csv(extension_dir / COVERAGE_AUDIT)

    assert len(asof) == 36
    assert len(labels) == 36
    assert not any(column.startswith("label_") or "hindsight" in column.lower() for column in asof.columns)
    assert {"label_should_not_take_v1", "label_strong_trade_candidate_v1"}.issubset(labels.columns)
    assert int((~asof["entry_observation_present_v1"]).sum()) == 3
    assert int((~asof["management_observation_present_v1"]).sum()) == 2
    assert "zero-trade/window edge" in set(coverage["entry_gap_reason_code_v1"])
    assert "missing AS_OF raw-state" in set(coverage["entry_gap_reason_code_v1"])

