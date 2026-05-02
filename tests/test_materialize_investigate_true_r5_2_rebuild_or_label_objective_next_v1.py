import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_investigate_true_r5_2_rebuild_or_label_objective_next_v1 import OUTPUT_FILES, materialize
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import R5_2_BAD_PROB, R5_2_RUNNER_PROB


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for idx in range(410):
        selected = idx < 20
        missed = not selected
        high_mfe = missed and idx < 167
        label_bad = selected or missed
        r5_2_label_bad = not high_mfe
        r5_2_bad = 0.40 if selected else (0.20 if idx < 351 else 0.38)
        r5_2_runner = 0.15 if selected else (0.80 if idx >= 351 else 0.45)
        rows.append(
            {
                "run_id": "run_0",
                "candidate_uid": f"c{idx:04d}",
                "trade_uid": f"t{idx:04d}",
                "trade_id": f"trade{idx:04d}",
                "decision_timestamp": f"2026-01-01T12:{idx % 60:02d}:00Z",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "split_scope_v1": "TRAIN",
                "batch_scope_v1": "BATCH_04",
                "label_should_not_take_v1": label_bad,
                "tail_10_50_mfe_v1": missed and idx % 3 == 0,
                "take_was_ok_v1": False,
                "fifty_plus_mfe_v1": high_mfe,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": False,
                "peak_mfe_bps_v1": 80.0 if high_mfe else 8.0,
                "mae_abs_bps_v1": 55.0,
                "r5_2_label_bad_blocker_v1": r5_2_label_bad,
                "r5_2_label_runner_protect_v1": False,
                "r5_2_label_runner_50_mfe_v1": False,
                "r5_2_label_runner_100_mfe_v1": False,
                "r5_2_label_runner_200_mfe_v1": False,
                "r5_2_label_strong_low_mae_runner_v1": False,
                "r6_label_risky_allow_v1": missed,
                "pred__entry_r5_should_not_take__prob_true_v1": 0.90 if selected else 0.55,
                "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.80 if selected else 0.60,
                "pred__entry_r5_runner_protect__prob_true_v1": 0.20 if selected else 0.50,
                "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.80 if selected else 0.30,
                "pred__entry_r5_bad_trade_but_high_runner_risk__prob_true_v1": 0.10,
                "pred__entry_r5_strong_trade_candidate__prob_true_v1": 0.10,
                "pred__entry_r5_take_was_ok__prob_true_v1": 0.10 if selected else 0.55,
                "r5_1_bad_blocker_score_v1": 0.90 if selected else 0.60,
                "r5_1_runner_guard_score_v1": 0.20 if selected else 0.60,
                R5_2_BAD_PROB: r5_2_bad,
                R5_2_RUNNER_PROB: r5_2_runner,
                "r5_2_selected_candidate__block_v1": selected,
                "pred__entry_r6_bad_risk__prob_true_v1": 0.90 if selected else 0.70,
                "pred__entry_r6_risky_allow__prob_true_v1": 0.90 if selected else 0.65,
                "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.90 if selected else 0.30,
                "pred__entry_r6_runner_protector__prob_true_v1": 0.10 if selected else 0.60,
                "pred__entry_r6_batch04_blindspot__prob_true_v1": 0.10,
                "selected_candidate_block_v1": selected,
            }
        )
    frame = pd.DataFrame(rows)
    score = frame.drop(columns=["selected_candidate_block_v1"])
    r6 = frame.copy()
    return score, r6


def test_investigate_true_r5_2_rebuild_materializes_label_objective_lock(tmp_path: Path) -> None:
    score_dir = tmp_path / "score"
    r6_dir = tmp_path / "r6"
    score_dir.mkdir()
    r6_dir.mkdir()
    score, r6 = _frames()
    score.to_parquet(score_dir / "monday_r6_foundation_score_frame_v1.parquet", index=False)
    r6.to_parquet(r6_dir / "monday_r6_on_foundation_scores_training_frame_v1.parquet", index=False)
    _write_json(
        score_dir / "score_rebuild_summary_v1.json",
        {"r5_2_selected_policy_v1": {"params_v1": {"bad_threshold_v1": 0.35, "runner_max_v1": 0.20}}},
    )
    _write_json(r6_dir / "summary_v1.json", {"r6_training_started_v1": True})

    out = tmp_path / "out"
    summary = materialize(reports_root=tmp_path, output_dir=out, v3_score_dir=score_dir, v3_r6_dir=r6_dir, v3_forensics_dir=tmp_path / "forensics")

    assert summary["decision_v1"] == "LABEL_OBJECTIVE_FIX_REQUIRED_BEFORE_R5_2_REBUILD"
    assert summary["next_action_v1"] == "FIX_R5_2_LABEL_OBJECTIVE_FIRST"
    assert summary["missed_rows_traced_v1"] == 390
    assert summary["training_started_v1"] is False
    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()
    trace = pd.read_csv(out / "post_v3_missed_rows_true_root_trace_v1.csv")
    assert len(trace) == 390
    assert "R5_2_LABEL_EXCLUDES_HIGH_MFE_AMBIGUOUS_CASE" in set(trace["r5_2_first_exclusion_reason_v1"])
    audit = pd.read_csv(out / "consistency_audit_v1.csv")
    assert set(audit["status_v1"]) == {"PASS"}
