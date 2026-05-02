import json
from pathlib import Path

import pandas as pd

import gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 as score_rebuild


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _foundation_fixture(path: Path) -> None:
    rows = []
    for idx in range(4):
        rows.append(
            {
                "run_id": f"run_{idx}",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": f"trade{idx}",
                "decision_timestamp": f"2026-01-0{idx + 1}T12:00:00Z",
                "used_for_training": idx < 2,
                "used_for_validation": idx == 2,
                "used_for_holdout": idx == 3,
                "split_scope_v1": "TRAIN" if idx < 2 else ("VALIDATION" if idx == 2 else "QUARANTINE_EVAL_ONLY"),
                "calendar_quarantine_status_v1": "QUARANTINED" if idx == 3 else "ACTIVE_CANDIDATE",
                "as_of_candidate_tradable_prob_v1": 0.9 - idx * 0.1,
                "as_of_entry_candidate_path_quality_pred_v1": 0.8 - idx * 0.1,
                "as_of_candidate_mfe_first_n_pred_v1": 2.0 - idx * 0.2,
                "as_of_skip_candidate_p_flat_v1": 0.2 + idx * 0.1,
                "label_should_not_take_v1": idx in {1, 2},
                "take_was_ok_v1": idx in {0, 3},
                "fifty_plus_mfe_v1": idx in {0, 3},
                "hundred_plus_mfe_v1": idx == 0,
                "two_hundred_plus_mfe_v1": False,
                "tail_10_50_mfe_v1": idx == 2,
                "strongest_winner_path_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": False,
                "r5_label_should_not_take_v1": idx in {1, 2},
                "r5_label_immediate_mae_risk_v1": idx == 2,
                "r5_label_immediate_MAE_risk_v1": idx == 2,
                "r5_label_runner_protect_v1": idx in {0, 3},
                "r5_label_strong_trade_candidate_v1": idx == 0,
                "r5_label_tail_control_10_50_risk_v1": idx == 2,
                "r5_label_take_was_ok_v1": idx in {0, 3},
                "r5_label_bad_trade_but_high_runner_risk_v1": False,
                "r5_label_wait_or_delay_advisory_v1": idx == 2,
                "r5_2_label_bad_blocker_v1": idx in {1, 2},
                "r5_2_label_runner_protect_v1": idx in {0, 3},
                "r5_2_label_runner_50_mfe_v1": idx in {0, 3},
                "r5_2_label_runner_100_mfe_v1": idx == 0,
                "r5_2_label_runner_200_mfe_v1": False,
                "r5_2_label_repaired_165_like_runner_v1": False,
                "r5_2_label_strong_low_mae_runner_v1": idx == 0,
                "r5_2_batch04_hard_negative_runner_v1": False,
                "r5_2_hard_negative_like_asof_v1": False,
                "baseline_realized_pnl_bps_v1": 20.0 if idx in {0, 3} else -10.0,
                "peak_mfe_bps_v1": 80.0 if idx in {0, 3} else 5.0,
                "mae_abs_bps_v1": 10.0 if idx in {0, 3} else 55.0,
                "giveback_bps_v1": 5.0,
            }
        )
    frame = pd.DataFrame(rows)
    asof = frame[["run_id", "candidate_uid", "trade_uid", "trade_id", "decision_timestamp"]].copy()
    for idx in range(104):
        asof[f"as_of_dummy_{idx:03d}_v1"] = float(idx)
    frame.to_parquet(path / score_rebuild.FOUNDATION_FRAME, index=False)
    asof.to_parquet(path / score_rebuild.FOUNDATION_AS_OF, index=False)
    _write_json(
        path / score_rebuild.FOUNDATION_SUMMARY,
        {
            "decision_v1": "MONDAY_R6_ACTUAL_FULLCOVERAGE_FOUNDATION_BUILT",
            "row_count_v1": 4,
            "active_rows_v1": 3,
            "quarantine_rows_v1": 1,
        },
    )


def test_score_rebuild_requires_explicit_flag(tmp_path: Path) -> None:
    foundation = tmp_path / "foundation"
    foundation.mkdir()
    _foundation_fixture(foundation)
    out = tmp_path / "out"

    summary = score_rebuild.materialize(reports_root=tmp_path, foundation_dir=foundation, output_dir=out, run_score_rebuild=False)

    assert summary["decision_v1"] == "EXPLICIT_SCORE_REBUILD_FLAG_REQUIRED"
    assert summary["training_started_v1"] is False
    assert (out / score_rebuild.STATUS).exists()


def test_score_rebuild_writes_r5_r5_1_r5_2_outputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(score_rebuild, "EXPECTED_FOUNDATION_ROWS", 4)
    monkeypatch.setattr(score_rebuild, "EXPECTED_ACTIVE_ROWS", 3)
    monkeypatch.setattr(score_rebuild, "EXPECTED_QUARANTINE_ROWS", 1)
    foundation = tmp_path / "foundation"
    foundation.mkdir()
    _foundation_fixture(foundation)
    out = tmp_path / "out"

    summary = score_rebuild.materialize(reports_root=tmp_path, foundation_dir=foundation, output_dir=out, run_score_rebuild=True)

    assert summary["decision_v1"] == "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED"
    assert summary["training_started_v1"] is True
    assert summary["r5_head_count_v1"] == 8
    assert summary["r5_1_policy_materialized_v1"] is True
    assert summary["r5_2_head_count_v1"] == 2
    assert summary["r6_heads_trained_v1"] is False
    for filename in score_rebuild.OUTPUT_FILES.values():
        if filename == "models":
            continue
        assert (out / filename).exists()
    audit = pd.read_csv(out / score_rebuild.CONSISTENCY_AUDIT)
    assert audit.set_index("check_v1").loc["NO_R6_HEADS_TRAINED", "status_v1"] == "PASS"
