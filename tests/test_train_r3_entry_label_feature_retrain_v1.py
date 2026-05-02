from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gx1.scripts.train_r3_entry_label_feature_retrain_v1 import (
    R3_CONSISTENCY_AUDIT,
    R3_CONTRACT,
    R3_MODEL_METRICS,
    R3_POLICY_SAFETY,
    R3_PREDICTION_VIEW,
    R3_R2_FALLBACK_OVERLAP,
    R3_STATUS,
    R3_SUMMARY,
    R3_THRESHOLD_POLICY,
    R3_WALKFORWARD_METRICS,
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


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    reports_root = tmp_path / "reports"
    readiness_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260421T_HARVEST_R2_ENTRY_COVERAGE_AND_WALKFORWARD_READINESS_V1"
    extension_dir = reports_root / "r3"
    runs_root = reports_root / "runs"
    readiness_dir.mkdir(parents=True)
    runs_root.mkdir(parents=True)
    run_ids = _run_ids(8)
    for run_id in run_ids:
        (runs_root / run_id).mkdir()

    feature_names = [
        "as_of_signal_bad_v1",
        "as_of_signal_strong_v1",
        "as_of_signal_mae_v1",
        "as_of_session_v1",
    ]
    asof_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    missing = {7, 19, 53, 68}
    for idx in range(72):
        run_id = run_ids[idx % len(run_ids)]
        should_not = idx % 5 == 0
        strong = idx % 5 == 1
        immediate_mae = idx % 3 == 0
        bad_capture = idx % 4 == 0
        direct_take = idx % 5 in {1, 2}
        wait_helped = idx % 6 in {2, 3}
        train = idx < 48
        validation = 48 <= idx < 60
        holdout = idx >= 60
        available = idx not in missing
        asof_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": f"cand-{idx:03d}",
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "used_for_training": train,
                "used_for_validation": validation,
                "used_for_holdout": holdout,
                "entry_observation_present_v1": available,
                "entry_raw_state_present_v1": available,
                "management_observation_present_v1": True,
                "as_of_signal_bad_v1": float(should_not) + (idx % 3) / 10.0,
                "as_of_signal_strong_v1": float(strong) + (idx % 2) / 10.0,
                "as_of_signal_mae_v1": float(immediate_mae) + (idx % 4) / 10.0,
                "as_of_session_v1": "LONDON" if idx % 2 else "NY",
            }
        )
        label_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": f"cand-{idx:03d}",
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "baseline_realized_pnl_bps_v1": -10.0 if should_not else 20.0,
                "peak_mfe_bps_v1": 75.0 if strong else (25.0 if not should_not else 12.0),
                "mae_abs_bps_v1": 35.0 if immediate_mae else 8.0,
                "giveback_bps_v1": 60.0 if bad_capture else 5.0,
                "harvest_capture_ratio_v1": 0.2 if bad_capture else 0.8,
                "exit_harvest_policy_action_v1": "ENTRY_SUPPRESS_OR_DOWNSIZE" if should_not else "KEEP_BASELINE",
                "trade_outcome_class": "never_mfe" if should_not else "positive_exit",
                "exit_reason": "THRESHOLD",
                "label_should_not_take_v1": should_not,
                "label_strong_trade_candidate_v1": strong,
                "label_immediate_mae_risk_v1": immediate_mae,
                "label_good_mfe_bad_capture_v1": bad_capture,
                "label_direct_take_ok_v1": direct_take,
                "label_wait_would_have_helped_v1": wait_helped,
            }
        )

    pd.DataFrame(asof_rows).to_parquet(
        readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_as_of_feature_table_v1.parquet", index=False
    )
    pd.DataFrame(label_rows).to_parquet(
        readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_hindsight_label_table_v1.parquet", index=False
    )
    _write_json(
        readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_readiness_contract_v1.json",
        {
            "as_of_feature_names_v1": feature_names,
            "not_live_gate": True,
            "not_policy_truth": True,
        },
    )
    _write_json(
        readiness_dir / "shadow_meta_all_trade_review_harvest_r2_entry_readiness_summary_v1.json",
        {
            "readiness_v1": {
                "binary_entry_walkforward_min_balanced_accuracy_v1": 0.6,
                "multiclass_entry_walkforward_min_balanced_accuracy_v1": 0.4,
            },
            "safety_v1": {
                "entry_blocks_50_plus_mfe_count_v1": 10,
                "entry_helps_10_50_mfe_tail_control_count_v1": 8,
            },
        },
    )
    return reports_root, readiness_dir, extension_dir


def test_materialize_r3_entry_label_feature_retrain(tmp_path: Path) -> None:
    reports_root, readiness_dir, extension_dir = _build_fixture(tmp_path)

    result = materialize(
        reports_root,
        readiness_dir=readiness_dir,
        extension_dir=extension_dir,
        batch_weeks=4,
        n_estimators=25,
        early_stopping_rounds=5,
        learning_rate=0.1,
        max_depth=2,
        n_jobs=1,
        expected_ledger_count=72,
    )

    assert result["status"]["R3_ENTRY_LABEL_FEATURE_RETRAIN_STATUS"] == "TRAINED_SHADOW_RESEARCH_READY_NOT_LIVE_GATE"
    for artifact in [
        R3_PREDICTION_VIEW,
        R3_MODEL_METRICS,
        R3_WALKFORWARD_METRICS,
        R3_POLICY_SAFETY,
        R3_R2_FALLBACK_OVERLAP,
        R3_THRESHOLD_POLICY,
        R3_CONSISTENCY_AUDIT,
        R3_CONTRACT,
        R3_STATUS,
        R3_SUMMARY,
    ]:
        assert (extension_dir / artifact).exists()

    prediction = pd.read_parquet(extension_dir / R3_PREDICTION_VIEW)
    metrics = pd.read_csv(extension_dir / R3_MODEL_METRICS)
    contract = json.loads((extension_dir / R3_CONTRACT).read_text(encoding="utf-8"))

    assert len(prediction) == 72
    assert int(prediction["entry_r3_feature_available_v1"].sum()) == 68
    assert prediction["entry_r3_shadow_action_v1"].notna().all()
    assert set(contract["hindsight_target_columns_v1"]) == {
        "label_should_not_take_v1",
        "label_strong_trade_candidate_v1",
        "label_immediate_mae_risk_v1",
        "label_good_mfe_bad_capture_v1",
        "label_direct_take_ok_v1",
        "label_wait_would_have_helped_v1",
    }
    assert set(metrics["split_v1"]) == {"TRAIN", "VALIDATION", "HOLDOUT"}
