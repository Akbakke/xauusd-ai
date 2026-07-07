import json
from pathlib import Path

import pandas as pd
import pytest

import gx1.scripts.materialize_parallel_monday_r6_recall_recovery_scan_v1 as scan


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frame() -> pd.DataFrame:
    rows = []
    for idx in range(6):
        should = idx in {0, 1, 2}
        selected = idx == 0
        rows.append(
            {
                "run_id": f"W{idx // 2}",
                "candidate_uid": scan.FORENSIC_REPAIRED_CANDIDATE_UID if idx == 5 else f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_timestamp": f"2026-01-0{idx + 1}T12:00:00Z",
                "split_scope_v1": "TRAIN" if idx < 3 else "HOLDOUT",
                "batch_scope_v1": "BATCH_01",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "label_should_not_take_v1": should,
                "take_was_ok_v1": not should,
                "tail_10_50_mfe_v1": idx in {0, 1},
                "fifty_plus_mfe_v1": idx == 4,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "is_repaired_165_v1": idx == 5,
                "r6_label_runner_near_miss_v1": False,
                "r5_selected_candidate__block_v1": selected,
                "r5_1_selected_candidate__block_v1": selected,
                "r5_2_selected_candidate__block_v1": selected,
                "selected_candidate_block_v1": selected,
                "asof_runner_guard_v1": False,
                "as_of_candidate_tradable_prob_v1": 0.2,
                "as_of_entry_candidate_path_quality_pred_v1": 0.2,
                "as_of_candidate_mfe_first_n_pred_v1": 0.2,
                "as_of_skip_candidate_p_flat_v1": 0.9,
                "pred__entry_r5_should_not_take__prob_true_v1": 0.92 if should else 0.1,
                "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.86 if should else 0.1,
                "pred__entry_r5_runner_protect__prob_true_v1": 0.1,
                "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.88 if idx in {0, 1} else 0.1,
                "r5_1_bad_blocker_score_v1": 0.9 if should else 0.1,
                "r5_1_runner_guard_score_v1": 0.1,
                "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.9 if selected else (0.7 if should else 0.1),
                "pred__entry_r5_2_runner_protector__prob_true_v1": 0.1,
                "pred__entry_r6_bad_risk__prob_true_v1": 0.9 if should else 0.1,
                "pred__entry_r6_runner_protector__prob_true_v1": 0.1 if idx != 5 else 0.99,
                "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.9 if idx in {0, 1} else 0.1,
                "pred__entry_r6_risky_allow__prob_true_v1": 0.9 if should else 0.1,
                "pred__entry_r6_batch04_blindspot__prob_true_v1": 0.1,
                "blocker_score_v1": 0.9 if should else 0.1,
                "runner_protector_score_v1": 0.1,
                "mae_abs_bps_v1": 45.0 if should else 5.0,
                "peak_mfe_bps_v1": 20.0 if should else 60.0,
            }
        )
    return pd.DataFrame(rows)


def _seed_inputs(root: Path) -> tuple[Path, Path]:
    score_dir = root / "MONDAY_R6_FOUNDATION_SCORE_REBUILD_V1_20260426T_CONTRACT_FIX_R5_R51_R52"
    r6_dir = root / "MONDAY_R6_REBUILD_ON_FOUNDATION_SCORES_V1_20260426T_CONTRACT_FIX_R6_FROM_FIXED_R52"
    score_dir.mkdir(parents=True)
    r6_dir.mkdir(parents=True)
    frame = _frame()
    frame.to_parquet(score_dir / scan.SCORE_FRAME, index=False)
    frame.to_parquet(r6_dir / scan.R6_TRAINING_FRAME, index=False)
    frame.to_parquet(r6_dir / scan.R6_PREDICTION_VIEW, index=False)
    pd.DataFrame(
        [
            {
                "policy_name_v1": "R6_CANDIDATE_00001_R6_RUNNER_FIRST_TWO_HEAD",
                "family_v1": "R6_RUNNER_FIRST_TWO_HEAD",
                "wednesday_safety_pass_v1": True,
                "wednesday_basic_safety_pass_v1": True,
                "hard_damage_count_v1": 0,
                "bad_threshold_v1": 0.85,
                "runner_threshold_v1": 0.7,
                "tail_threshold_v1": 0.5,
                "risky_threshold_v1": 0.5,
                "blindspot_threshold_v1": 0.7,
                "r5_2_runner_threshold_v1": 0.74,
                "use_r5_2_base_v1": False,
                "hard_asof_runner_guard_v1": False,
                "block_count_v1": 1,
                "bad_blocks_v1": 1,
                "tail_help_v1": 1,
                "precision_v1": 1.0,
                "worst_loso_v1": 1.0,
                "false_take_ok_blocks_v1": 0,
                "fifty_plus_mfe_blocked_v1": 0,
                "hundred_plus_mfe_blocked_v1": 0,
                "two_hundred_plus_mfe_blocked_v1": 0,
                "strongest_winner_damage_v1": 0,
                "repaired_165_damage_v1": 0,
                "quarantine_blocks_v1": 0,
                "runner_near_miss_blocked_v1": 0,
            }
        ]
    ).to_csv(r6_dir / scan.R6_GRID, index=False)
    _write_json(score_dir / scan.SCORE_SUMMARY, {"decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED", "r6_heads_trained_v1": False})
    _write_json(r6_dir / scan.R6_SUMMARY, {"decision_v1": "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER", "as_of_column_count_v1": 109})
    _write_json(r6_dir / scan.R6_COMPARE, {"verdict_v1": "MONDAY_R6_EXPLICIT_REBUILD_SAFE_BUT_NOT_BETTER"})
    return score_dir, r6_dir


def test_parallel_scan_requires_explicit_flag(tmp_path: Path) -> None:
    score_dir, r6_dir = _seed_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="requires --run-parallel-scan"):
        scan.materialize(reports_root=tmp_path, score_dir=score_dir, r6_dir=r6_dir, output_dir=tmp_path / "out")


def test_parallel_scan_materializes_all_lanes_without_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scan, "EXPECTED_ROW_COUNT", 6)
    monkeypatch.setattr(scan, "EXPECTED_ACTIVE_ROWS", 6)
    monkeypatch.setattr(scan, "EXPECTED_QUARANTINE_ROWS", 0)
    score_dir, r6_dir = _seed_inputs(tmp_path)
    out = tmp_path / "out"

    summary = scan.materialize(
        reports_root=tmp_path,
        score_dir=score_dir,
        r6_dir=r6_dir,
        output_dir=out,
        run_parallel_scan=True,
        max_workers=2,
        quick_scan=True,
    )

    assert summary["training_started_v1"] is False
    assert summary["new_baseline_built_v1"] is False
    assert summary["new_feature_surface_built_v1"] is False
    assert summary["lane_count_v1"] == 10
    assert summary["forensic_repaired_trade_present_v1"] is True
    for filename in scan.OUTPUT_FILES.values():
        assert (out / filename).exists()
    audit = pd.read_csv(out / scan.AUDIT)
    assert audit.set_index("check_v1").loc["NO_TRAINING", "status_v1"] == "PASS"
    lane_02 = pd.read_csv(out / scan.LANE_02)
    assert "rows_added_v1" in lane_02.columns
    leaderboard = pd.read_csv(out / scan.LEADERBOARD)
    assert not leaderboard.empty
