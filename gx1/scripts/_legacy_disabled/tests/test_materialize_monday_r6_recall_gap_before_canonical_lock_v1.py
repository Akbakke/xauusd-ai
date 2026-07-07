import json
from pathlib import Path

import pandas as pd

import gx1.scripts.materialize_monday_r6_recall_gap_before_canonical_lock_v1 as recall_gap


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _r6_fixture(path: Path) -> None:
    rows = []
    for idx in range(4):
        split = "TRAIN" if idx < 2 else ("VALIDATION" if idx == 2 else "HOLDOUT")
        should = idx in {0, 2, 3}
        selected = idx == 0
        rows.append(
            {
                "run_id": f"run_{idx}",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": f"trade{idx}",
                "decision_timestamp": f"2026-01-0{idx + 1}T12:00:00Z",
                "split_scope_v1": split,
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "label_should_not_take_v1": should,
                "tail_10_50_mfe_v1": should,
                "take_was_ok_v1": not should,
                "fifty_plus_mfe_v1": not should,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "r5_selected_candidate__block_v1": selected,
                "r5_1_selected_candidate__block_v1": selected,
                "r5_2_selected_candidate__block_v1": selected,
                "selected_candidate_block_v1": selected,
                "asof_runner_guard_v1": False,
                "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.9 if selected else 0.2,
                "pred__entry_r5_2_runner_protector__prob_true_v1": 0.1,
                "pred__entry_r6_bad_risk__prob_true_v1": 0.9 if selected else 0.3,
                "pred__entry_r6_runner_protector__prob_true_v1": 0.1,
                "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.9 if selected else 0.2,
                "pred__entry_r6_risky_allow__prob_true_v1": 0.99 if selected else 0.2,
                "pred__entry_r6_batch04_blindspot__prob_true_v1": 0.1,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_parquet(path / recall_gap.TRAINING_FRAME, index=False)
    frame.to_parquet(path / recall_gap.PREDICTION_VIEW, index=False)
    pd.DataFrame(
        [
            {
                "policy_name_v1": "safe",
                "family_v1": "R6",
                "wednesday_safety_pass_v1": True,
                "wednesday_basic_safety_pass_v1": True,
                "hard_damage_count_v1": 0,
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
            },
            {
                "policy_name_v1": "unsafe_recall",
                "family_v1": "R6",
                "wednesday_safety_pass_v1": False,
                "wednesday_basic_safety_pass_v1": False,
                "hard_damage_count_v1": 2,
                "block_count_v1": 3,
                "bad_blocks_v1": 3,
                "tail_help_v1": 3,
                "precision_v1": 0.75,
                "worst_loso_v1": None,
                "false_take_ok_blocks_v1": 1,
                "fifty_plus_mfe_blocked_v1": 1,
                "hundred_plus_mfe_blocked_v1": 0,
                "two_hundred_plus_mfe_blocked_v1": 0,
                "strongest_winner_damage_v1": 0,
                "repaired_165_damage_v1": 0,
            },
        ]
    ).to_csv(path / recall_gap.R6_GRID, index=False)
    _write_json(
        path / recall_gap.R6_SUMMARY,
        {
            "decision_v1": "MONDAY_R6_ON_FOUNDATION_SCORES_SAFE_BUT_NOT_BETTER",
            "selected_policy_source_v1": "R6_FAMILY_GRID_SAFE_FALLBACK",
        },
    )
    _write_json(
        path / recall_gap.R6_COMPARE,
        {
            "candidate_metrics_v1": {"bad_blocks_v1": 1, "tail_help_v1": 1, "precision_v1": 1.0},
            "candidate_worst_loso_v1": 1.0,
            "safety_failures_v1": [],
            "verdict_v1": "MONDAY_R6_EXPLICIT_REBUILD_SAFE_BUT_NOT_BETTER",
        },
    )


def test_recall_gap_materializes_no_training_diagnosis(tmp_path: Path) -> None:
    recall_gap.EXPECTED_ROW_COUNT = 4
    r6_dir = tmp_path / "r6"
    r6_dir.mkdir()
    _r6_fixture(r6_dir)
    out = tmp_path / "out"

    summary = recall_gap.materialize(reports_root=tmp_path, r6_dir=r6_dir, output_dir=out)

    assert summary["training_started_v1"] is False
    assert summary["decision_v1"] == "MONDAY_R6_RECALL_GAP_CONFIRMED_BEFORE_CANONICAL_LOCK"
    assert summary["selected_bad_blocks_v1"] == 1
    assert summary["missed_bad_rows_v1"] == 2
    assert summary["selected_blocks_outside_train_v1"] == 0
    for filename in recall_gap.OUTPUT_FILES.values():
        assert (out / filename).exists()
    audit = pd.read_csv(out / recall_gap.CONSISTENCY_AUDIT)
    assert audit.set_index("check_v1").loc["NO_TRAINING_RUN", "status_v1"] == "PASS"
    fallback = pd.read_csv(out / recall_gap.FALLBACK_WORD_AUDIT)
    assert "STALE_NAME_IN_GENERATED_ARTIFACT" in set(fallback["classification_v1"])
