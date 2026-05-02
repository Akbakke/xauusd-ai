from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts.train_truth_harvest_retrain_candidate_v1 import (
    RETRAIN_METRICS,
    RETRAIN_PREDICTION_VIEW,
    RETRAIN_REPLAY_15WEEK,
    RETRAIN_STATUS,
    TOP_LEVEL_SUMMARY,
    _split_name_frame,
    build_harvest_retrain_candidate_payload,
    materialize_truth_harvest_retrain_candidate,
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


def _split_flags(index: int) -> tuple[bool, bool, bool]:
    if index < 18:
        return True, False, False
    if index < 24:
        return False, True, False
    return False, False, True


def _build_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    reports_root = tmp_path / "reports"
    review_dir = reports_root / "review"
    harvest_dir = reports_root / "harvest"
    extension_dir = reports_root / "retrain"
    runs_root = reports_root / "runs"
    review_dir.mkdir(parents=True)
    harvest_dir.mkdir(parents=True)
    runs_root.mkdir(parents=True)
    run_ids = _run_ids(16)
    for run_id in run_ids:
        (runs_root / run_id).mkdir()

    action_labels = [
        "DELAY_BE_PLUS_FLOOR_AND_RUNNER_TRAIL",
        "ENTRY_SUPPRESS_OR_DOWNSIZE",
        "EXIT_EARLIER_DAMAGE_CONTROL",
        "HOLD_LONGER_HOME_RUN_RUNNER",
        "HOLD_LONGER_RUNNER_TRAIL",
        "KEEP_BASELINE",
    ]
    exit_labels = [
        "HOLD_LONGER_OR_RUNNER_TRAIL",
        "NO_EXIT_TRAINING_ENTRY_FILTER",
        "EXIT_EARLIER_DAMAGE_CONTROL",
        "KEEP_BASELINE",
    ]
    entry_labels = ["PRIORITIZE_CLEAN_RUNNER", "REJECT_OR_LOW_SIZE", "ALLOW_BASELINE"]

    target_rows: list[dict[str, object]] = []
    policy_rows: list[dict[str, object]] = []
    entry_rows: list[dict[str, object]] = []
    management_rows: list[dict[str, object]] = []
    for idx in range(30):
        run_id = run_ids[idx % len(run_ids)]
        candidate_uid = f"cand-{idx:03d}"
        train, validation, holdout = _split_flags(idx)
        action = action_labels[idx % len(action_labels)]
        exit_label = exit_labels[idx % len(exit_labels)]
        entry_label = entry_labels[idx % len(entry_labels)]
        baseline = float((idx % 7) - 3)
        skip_delta = 20.0 if action == "ENTRY_SUPPRESS_OR_DOWNSIZE" else 0.0
        exit_delta = 12.0 if action == "EXIT_EARLIER_DAMAGE_CONTROL" else 0.0
        hold_delta = 40.0 if "HOLD_LONGER" in action or action.startswith("DELAY_") else 0.0
        reward = max(skip_delta, exit_delta, hold_delta)
        target_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": candidate_uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "used_for_training": train,
                "used_for_validation": validation,
                "used_for_holdout": holdout,
                "entry_xgb_harvest_label_v1": entry_label,
                "entry_xgb_binary_take_target_v1": entry_label != "REJECT_OR_LOW_SIZE",
                "entry_xgb_sample_weight_proposed_v1": 1.0 + reward / 100.0,
                "exit_transformer_supervision_label_v1": exit_label,
                "exit_transformer_target_extra_value_bps_v1": hold_delta,
                "exit_transformer_target_saved_bps_v1": exit_delta,
                "exit_transformer_sample_weight_proposed_v1": 1.0 + reward / 100.0,
                "management_rl_harvest_action_label_v1": action,
                "management_rl_harvest_reward_bps_raw_v1": reward,
                "management_rl_harvest_reward_bps_clipped_200_v1": reward,
                "harvest_model_update_family_v1": "RUNNER_HARVEST" if hold_delta else "ENTRY_FILTER",
                "harvest_quality_bucket_v1": "EXIT_TOO_EARLY_UNDERHARVEST" if hold_delta else "KEEP_BASELINE_HARVEST",
                "model_adjustment_contract_v1": "TEST_REAL_COLUMNS",
            }
        )
        policy_rows.append(
            {
                "run_id": run_id,
                "candidate_uid": candidate_uid,
                "trade_uid": f"trade-{idx:03d}",
                "trade_id": str(idx),
                "decision_timestamp": f"2025-01-{(idx % 28) + 1:02d}T12:00:00+00:00",
                "baseline_realized_pnl_bps_v1": baseline,
                "peak_mfe_bps_v1": 50.0 + idx,
                "mae_abs_bps_v1": 5.0 + (idx % 5),
                "giveback_bps_v1": 25.0 + idx,
                "harvest_capture_ratio_v1": 0.2,
                "harvest_quality_bucket_v1": "EXIT_TOO_EARLY_UNDERHARVEST" if hold_delta else "KEEP_BASELINE_HARVEST",
                "exit_harvest_policy_action_v1": action,
                "rl_priority_entry_skip_delta_bps_v1": skip_delta,
                "rl_priority_exit_earlier_delta_bps_v1": exit_delta,
                "rl_priority_hold_longer_delta_bps_v1": hold_delta,
                "management_rl_harvest_reward_bps_raw_v1": reward,
                "management_rl_harvest_reward_bps_clipped_200_v1": reward,
            }
        )
        entry_rows.append(
            {
                "candidate_uid": candidate_uid,
                "as_of_hour_utc_v1": idx % 24,
                "as_of_session_v1": "LONDON" if idx % 2 else "NY",
                "as_of_candidate_tradable_prob_v1": 0.2 + (idx % 10) / 20.0,
            }
        )
        management_rows.append(
            {
                "candidate_uid_exact_v1": candidate_uid,
                "as_of_hour_utc_v1": idx % 24,
                "as_of_session_v1": "LONDON" if idx % 2 else "NY",
                "as_of_management_core_mfe_bps_so_far_v1": 10.0 + idx,
                "as_of_management_core_mae_bps_so_far_v1": 2.0 + (idx % 4),
            }
        )

    pd.DataFrame(target_rows).to_parquet(harvest_dir / "shadow_meta_all_trade_review_harvest_model_adjustment_target_view_v1.parquet", index=False)
    pd.DataFrame(policy_rows).to_parquet(harvest_dir / "shadow_meta_all_trade_review_exit_harvest_policy_candidate_trade_view_v1.parquet", index=False)
    pd.DataFrame(entry_rows).to_parquet(review_dir / "shadow_meta_all_trade_review_entry_rl_observability_view_v1.parquet", index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid": f"cand-{idx:03d}",
                "as_of_skip_replay_h1_range_compression_ratio_v1": 0.5 + (idx % 6) / 10.0,
                "as_of_skip_replay_window_directional_imbalance_60_bps_v1": float((idx % 5) - 2),
                "as_of_skip_candidate_margin_v1": 0.1 + (idx % 4) / 10.0,
                "as_of_skip_xgb_p_flat_v1": 0.2 + (idx % 3) / 10.0,
            }
            for idx in range(30)
        ]
    ).to_parquet(review_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet", index=False)
    pd.DataFrame(management_rows).to_parquet(review_dir / "shadow_meta_all_trade_review_management_rl_row_semantics_view_v1.parquet", index=False)
    _write_json(
        review_dir / "shadow_meta_all_trade_review_entry_rl_observability_contract_v1.json",
        {"observation_feature_names_v1": ["as_of_hour_utc_v1", "as_of_session_v1", "as_of_candidate_tradable_prob_v1"]},
    )
    _write_json(
        review_dir / "shadow_meta_all_trade_review_management_rl_observation_contract_v1.json",
        {
            "observation_vector_feature_names_v1": [
                "as_of_hour_utc_v1",
                "as_of_session_v1",
                "as_of_management_core_mfe_bps_so_far_v1",
                "as_of_management_core_mae_bps_so_far_v1",
            ]
        },
    )
    _write_json(
        harvest_dir / "shadow_meta_all_trade_review_exit_harvest_policy_candidate_status_v1.json",
        {"EXIT_HARVEST_POLICY_CANDIDATE_STATUS": "READY_FOR_RETRAIN_TARGET_REVIEW"},
    )
    return reports_root, review_dir, harvest_dir, extension_dir


def test_build_harvest_retrain_candidate_trains_and_replays(tmp_path: Path) -> None:
    reports_root, review_dir, harvest_dir, extension_dir = _build_fixture(tmp_path)

    payload = build_harvest_retrain_candidate_payload(
        reports_root=reports_root,
        review_dir=review_dir,
        harvest_dir=harvest_dir,
        extension_dir=extension_dir,
        batch_weeks=15,
        n_estimators=20,
        early_stopping_rounds=5,
        learning_rate=0.1,
        max_depth=2,
        seed=7,
        n_jobs=1,
        entry_reject_probability_threshold=0.5,
        entry_feature_mode="rich_asof_raw",
    )

    summary = payload["summary_v1"]
    prediction_df = payload["prediction_df_v1"]
    replay_df = payload["replay_df_v1"]
    metrics_df = payload["metrics_df_v1"]

    assert summary["status_v1"]["HARVEST_RETRAIN_CANDIDATE_STATUS"] == "TRAINED_SHADOW_REPLAY_READY_NOT_PROMOTED"
    assert summary["model_count_v1"] == 5
    assert summary["candidate_count_v1"] == 30
    assert summary["entry_feature_mode_v1"] == "rich_asof_raw"
    assert summary["entry_rich_raw_feature_count_v1"] == 4
    assert summary["target_harvest_upper_bound_clipped_200_delta_bps_v1"] > 0.0
    assert len(prediction_df) == 30
    assert not replay_df.empty
    assert set(metrics_df["split_v1"]) == {"TRAIN", "VALIDATION", "HOLDOUT"}


def test_split_name_frame_allows_excluded_overlap_rows() -> None:
    frame = pd.DataFrame(
        {
            "used_for_training": [True, False, False, False],
            "used_for_validation": [False, True, False, False],
            "used_for_holdout": [False, False, True, False],
        }
    )
    out = _split_name_frame(frame)
    assert out.tolist() == ["TRAIN", "VALIDATION", "HOLDOUT", "EXCLUDED_NO_SPLIT"]


def test_split_name_frame_still_rejects_multi_flag_rows() -> None:
    frame = pd.DataFrame(
        {
            "used_for_training": [True],
            "used_for_validation": [True],
            "used_for_holdout": [False],
        }
    )
    with pytest.raises(ValueError, match="Expected at most one split flag per row"):
        _split_name_frame(frame)


def test_materialize_harvest_retrain_candidate_writes_artifacts(tmp_path: Path) -> None:
    reports_root, review_dir, harvest_dir, extension_dir = _build_fixture(tmp_path)

    result = materialize_truth_harvest_retrain_candidate(
        reports_root,
        review_dir=review_dir,
        harvest_dir=harvest_dir,
        extension_dir=extension_dir,
        batch_weeks=15,
        n_estimators=20,
        early_stopping_rounds=5,
        learning_rate=0.1,
        max_depth=2,
        seed=7,
        n_jobs=1,
    )

    assert result["status"]["HARVEST_RETRAIN_CANDIDATE_STATUS"] == "TRAINED_SHADOW_REPLAY_READY_NOT_PROMOTED"
    assert (extension_dir / RETRAIN_PREDICTION_VIEW).exists()
    assert (extension_dir / RETRAIN_REPLAY_15WEEK).exists()
    assert (extension_dir / RETRAIN_METRICS).exists()
    assert (extension_dir / RETRAIN_STATUS).exists()
    assert (reports_root / TOP_LEVEL_SUMMARY).exists()


def test_harvest_retrain_candidate_hard_fails_leaky_feature_contract(tmp_path: Path) -> None:
    reports_root, review_dir, harvest_dir, extension_dir = _build_fixture(tmp_path)
    _write_json(
        review_dir / "shadow_meta_all_trade_review_entry_rl_observability_contract_v1.json",
        {"observation_feature_names_v1": ["as_of_hour_utc_v1", "hindsight_should_skip_trade_v1"]},
    )

    with pytest.raises(ValueError, match="forbidden leakage"):
        build_harvest_retrain_candidate_payload(
            reports_root=reports_root,
            review_dir=review_dir,
            harvest_dir=harvest_dir,
            extension_dir=extension_dir,
            batch_weeks=15,
            n_estimators=20,
            early_stopping_rounds=5,
            learning_rate=0.1,
            max_depth=2,
            seed=7,
            n_jobs=1,
            entry_reject_probability_threshold=0.5,
            entry_feature_mode="rich_asof_raw",
        )


def test_harvest_retrain_candidate_hard_fails_missing_rich_entry_state(tmp_path: Path) -> None:
    reports_root, review_dir, harvest_dir, extension_dir = _build_fixture(tmp_path)
    (review_dir / "shadow_meta_all_trade_review_entry_skipability_raw_state_v1.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="requires shadow_meta_all_trade_review_entry_skipability_raw_state_v1"):
        build_harvest_retrain_candidate_payload(
            reports_root=reports_root,
            review_dir=review_dir,
            harvest_dir=harvest_dir,
            extension_dir=extension_dir,
            batch_weeks=15,
            n_estimators=20,
            early_stopping_rounds=5,
            learning_rate=0.1,
            max_depth=2,
            seed=7,
            n_jobs=1,
            entry_reject_probability_threshold=0.5,
            entry_feature_mode="rich_asof_raw",
        )
