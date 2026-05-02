from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.train_r5_entry_retrain_with_repaired_coverage_and_slice_robustness_v1 import R5_PROB
from gx1.scripts.train_r5_loso_batch04_robustness_retrain_v1 import (
    AS_OF_FEATURE_TABLE,
    BATCH04_FAILURE_ATTRIBUTION,
    CONSISTENCY_AUDIT,
    DECISION_MATRIX,
    HEAD_TO_HEAD,
    HINDSIGHT_OUTCOME_TABLE,
    LOSO_METRICS,
    ROBUST_STACK_BAKEOFF,
    SUMMARY,
    THRESHOLD_SEARCH,
    materialize,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_ids(count: int) -> list[str]:
    starts = pd.date_range("2025-01-01", periods=count, freq="7D")
    ends = starts + pd.Timedelta(days=7)
    return [f"E2E_SANITY_ORDERFIX_{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}" for s, e in zip(starts, ends)]


def _build_r5_fixture(tmp_path: Path) -> tuple[Path, Path]:
    reports_root = tmp_path / "reports"
    r5_dir = reports_root / "r5"
    runs_root = reports_root / "runs"
    r5_dir.mkdir(parents=True)
    runs_root.mkdir(parents=True)
    run_ids = _run_ids(5)
    for run_id in run_ids:
        (runs_root / run_id).mkdir()

    feature_names = [
        "as_of_candidate_tradable_prob_v1",
        "as_of_entry_candidate_path_quality_pred_v1",
        "as_of_candidate_mfe_first_n_pred_v1",
        "as_of_skip_candidate_p_flat_v1",
        "as_of_entry_candidate_margin_v1",
        "as_of_skip_replay_retracement_from_last_impulse_v1",
        "as_of_skip_replay_clv_v1",
        "as_of_skip_replay_window_range_15_bps_v1",
        "as_of_session_v1",
    ]
    asof_rows: list[dict[str, object]] = []
    hindsight_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    for idx in range(80):
        run_id = run_ids[idx % len(run_ids)]
        should = idx % 4 == 0
        strong = (idx % 5 == 3) or (idx % 13 == 0)
        repaired = idx in {70, 71, 72}
        peak = 160.0 if strong else (35.0 if should else 22.0)
        pnl = -60.0 if should else 25.0
        uid = f"cand-{idx:03d}"
        asof_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "used_for_training": idx < 50,
                "used_for_validation": 50 <= idx < 65,
                "used_for_holdout": idx >= 65,
                "as_of_candidate_tradable_prob_v1": 0.97 if strong else 0.72 + (idx % 7) * 0.02,
                "as_of_entry_candidate_path_quality_pred_v1": 0.88 if strong else 0.45 + (idx % 5) * 0.08,
                "as_of_candidate_mfe_first_n_pred_v1": 2.4 if strong else 1.1 + (idx % 4) * 0.2,
                "as_of_skip_candidate_p_flat_v1": 0.32 if strong else 0.55,
                "as_of_entry_candidate_margin_v1": 0.18 if strong else 0.04,
                "as_of_skip_replay_retracement_from_last_impulse_v1": 0.8 if strong else 0.25 + (idx % 5) * 0.1,
                "as_of_skip_replay_clv_v1": 0.7 if strong else 0.35,
                "as_of_skip_replay_window_range_15_bps_v1": 55.0 if strong else 105.0,
                "as_of_session_v1": "US" if idx % 2 else "EU",
                "entry_observation_present_v1": True,
                "entry_raw_state_present_v1": True,
                "entry_coverage_repair_applied_v1": repaired,
                "entry_coverage_repair_source_v1": "fixture_repair" if repaired else "original",
            }
        )
        hindsight_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "baseline_realized_pnl_bps_v1": pnl,
                "peak_mfe_bps_v1": peak,
                "mae_abs_bps_v1": 80.0 if should else 12.0,
                "giveback_bps_v1": 20.0,
                "hindsight_entry_decision_review_v1": "SHOULD_SKIP_TRADE" if should else "TAKE_WAS_OK",
                "hindsight_management_review_v1": "MANAGED_OK",
                "r5_label_should_not_take_v1": should,
                "r5_label_immediate_mae_risk_v1": should or idx % 3 == 0,
                "r5_label_runner_protect_v1": (not should) and (strong or repaired),
                "r5_label_strong_trade_candidate_v1": strong,
                "r5_label_tail_control_10_50_risk_v1": should and peak < 50.0,
                "r5_label_take_was_ok_v1": not should,
                "r5_label_bad_trade_but_high_runner_risk_v1": should and peak >= 50.0,
                "r5_label_wait_or_delay_advisory_v1": idx % 6 == 0,
                "r5_hindsight_label_contract_v1": "fixture",
            }
        )
        probs = {
            "should_not_take": 0.85 if should else 0.25,
            "immediate_MAE_risk": 0.82 if should else 0.30,
            "runner_protect": 0.88 if strong and not should else 0.25,
            "strong_trade_candidate": 0.90 if strong else 0.20,
            "tail_control_10_50_risk": 0.80 if should and peak < 50.0 else 0.20,
            "take_was_ok": 0.90 if not should else 0.20,
            "bad_trade_but_high_runner_risk": 0.75 if should and peak >= 50.0 else 0.10,
            "wait_or_delay_advisory": 0.70 if idx % 6 == 0 else 0.25,
        }
        pred_row: dict[str, object] = {
            "candidate_uid": uid,
            "no_entry_fallback_baseline__block_v1": False,
            "r2_fallback_reference__block_v1": idx % 5 == 3,
            "r3_fullcoverage_conservative__block_v1": should and idx % 2 == 0,
            "r4_current_reference__block_v1": should or idx % 5 == 3,
            "r5_selected_candidate__block_v1": should,
        }
        for label_id, column in R5_PROB.items():
            pred_row[column] = probs[label_id]
        pred_rows.append(pred_row)

    pd.DataFrame(asof_rows).to_parquet(r5_dir / "shadow_meta_all_trade_review_r5_entry_as_of_feature_table_v1.parquet", index=False)
    pd.DataFrame(hindsight_rows).to_parquet(r5_dir / "shadow_meta_all_trade_review_r5_entry_hindsight_label_outcome_table_v1.parquet", index=False)
    pd.DataFrame(pred_rows).to_parquet(r5_dir / "shadow_meta_all_trade_review_r5_entry_policy_prediction_view_v1.parquet", index=False)
    _write_json(
        r5_dir / "shadow_meta_all_trade_review_r5_entry_retrain_contract_v1.json",
        {
            "as_of_feature_names_v1": feature_names,
            "hindsight_label_columns_v1": [
                "r5_label_should_not_take_v1",
                "r5_label_immediate_mae_risk_v1",
                "r5_label_runner_protect_v1",
                "r5_label_strong_trade_candidate_v1",
                "r5_label_tail_control_10_50_risk_v1",
                "r5_label_take_was_ok_v1",
                "r5_label_bad_trade_but_high_runner_risk_v1",
                "r5_label_wait_or_delay_advisory_v1",
            ],
        },
    )
    _write_json(
        r5_dir / "shadow_meta_all_trade_review_r5_entry_summary_v1.json",
        {
            "coverage_v1": {
                "ledger_trade_count_v1": 80,
                "entry_coverage_v1": 80,
                "entry_raw_coverage_v1": 80,
                "missing_count_v1": 0,
                "synthetic_count_v1": 0,
                "repaired_rows_v1": 3,
            }
        },
    )
    loso_rows = []
    thresholds = {
        "should_not_take_threshold_v1": 0.80,
        "immediate_mae_threshold_v1": 0.85,
        "tail_control_threshold_v1": 0.80,
        "runner_protect_threshold_v1": 0.99,
        "strong_protect_threshold_v1": 0.99,
        "take_ok_protect_threshold_v1": 0.99,
        "take_ok_block_ceiling_v1": 0.45,
        "bad_risk_override_threshold_v1": 0.88,
    }
    for batch_idx, run_id in enumerate(run_ids, start=1):
        loso_rows.append(
            {
                "policy_name_v1": "R5_R2_PRESERVATION_AWARE_STACK",
                "holdout_slice_v1": f"BATCH_{batch_idx:02d}",
                "selected_policy_name_v1": "R5_R2_PRESERVATION_AWARE_STACK",
                "thresholds_json_v1": json.dumps(thresholds, sort_keys=True),
                "run_start_v1": run_id,
                "run_end_v1": run_id,
                "fifty_plus_mfe_block_count_v1": 3 if batch_idx == 4 else 0,
            }
        )
    pd.DataFrame(loso_rows).to_csv(r5_dir / "shadow_meta_all_trade_review_r5_entry_loso_v1.csv", index=False)
    return reports_root, r5_dir


def test_r5_loso_batch04_robustness_materializes(tmp_path: Path) -> None:
    reports_root, r5_dir = _build_r5_fixture(tmp_path)
    extension_dir = reports_root / "r5_1"
    result = materialize(
        reports_root,
        r5_dir=r5_dir,
        extension_dir=extension_dir,
        batch_weeks=1,
        n_estimators=20,
        early_stopping_rounds=5,
        n_jobs=1,
        expected_ledger_count=80,
    )
    assert result["status"]["not_live_gate"] is True
    for artifact in [
        AS_OF_FEATURE_TABLE,
        HINDSIGHT_OUTCOME_TABLE,
        BATCH04_FAILURE_ATTRIBUTION,
        THRESHOLD_SEARCH,
        ROBUST_STACK_BAKEOFF,
        LOSO_METRICS,
        HEAD_TO_HEAD,
        DECISION_MATRIX,
        CONSISTENCY_AUDIT,
        SUMMARY,
    ]:
        assert (extension_dir / artifact).exists()
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["coverage_v1"]["entry_coverage_v1"] == 80
    assert summary["coverage_v1"]["synthetic_count_v1"] == 0
    assert summary["status_v1"]["not_live_gate"] is True
    consistency = pd.read_csv(extension_dir / CONSISTENCY_AUDIT)
    assert not consistency["status_v1"].eq("FAIL").any()
    failure = pd.read_csv(extension_dir / BATCH04_FAILURE_ATTRIBUTION)
    assert len(failure) > 0


def test_r5_loso_batch05_absent_is_not_reported_as_fail(tmp_path: Path) -> None:
    reports_root, r5_dir = _build_r5_fixture(tmp_path)
    extension_dir = reports_root / "r5_1_compact"
    materialize(
        reports_root,
        r5_dir=r5_dir,
        extension_dir=extension_dir,
        batch_weeks=2,
        n_estimators=20,
        early_stopping_rounds=5,
        n_jobs=1,
        expected_ledger_count=80,
    )
    summary = json.loads((extension_dir / SUMMARY).read_text(encoding="utf-8"))
    assert summary["decision_v1"]["batch05_loso_pass_v1"] is None
