import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_run_r6_from_v3_and_r6_head_recall_forensics_v1 import OUTPUT_FILES, materialize
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import (
    R5_2_BASE_MEMBERSHIP_CONTRACT_V3,
    R5_2_BAD_PROB,
    R5_2_RUNNER_PROB,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frame(*, v3: bool, r6: bool = False) -> pd.DataFrame:
    rows = []
    for idx in range(6):
        should = idx < 5
        base = idx < 3 or (v3 and idx == 3)
        selected = base if r6 else False
        rows.append(
            {
                "run_id": "run_0",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_timestamp": f"2026-01-01T12:0{idx}:00Z",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "label_should_not_take_v1": should,
                "tail_10_50_mfe_v1": idx in {0, 1, 3},
                "take_was_ok_v1": not should,
                "fifty_plus_mfe_v1": False,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "is_repaired_165_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": False,
                "pred__entry_r5_should_not_take__prob_true_v1": 0.9 if should else 0.1,
                "pred__entry_r5_runner_protect__prob_true_v1": 0.1,
                "r5_1_bad_blocker_score_v1": 0.9 if should else 0.1,
                "r5_1_runner_guard_score_v1": 0.1,
                R5_2_BAD_PROB: 0.6 if should else 0.1,
                R5_2_RUNNER_PROB: 0.1,
                "r5_2_selected_candidate__block_v1": base,
                "pred__entry_r6_bad_risk__prob_true_v1": 0.2,
                "pred__entry_r6_runner_protector__prob_true_v1": 0.1,
                "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.2,
                "pred__entry_r6_risky_allow__prob_true_v1": 0.2,
                "pred__entry_r6_batch04_blindspot__prob_true_v1": 0.1,
                "asof_runner_guard_v1": False,
                "selected_candidate_block_v1": selected,
            }
        )
    return pd.DataFrame(rows)


def _seed_score(path: Path, *, v3: bool) -> None:
    path.mkdir(parents=True)
    _frame(v3=v3).to_parquet(path / "monday_r6_foundation_score_frame_v1.parquet", index=False)
    _write_json(
        path / "score_rebuild_summary_v1.json",
        {"r5_2_selected_policy_v1": {"base_membership_active_contract_id_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V3["contract_id_v1"] if v3 else "V2"}},
    )


def _seed_r6(path: Path, score_dir: Path, *, v3: bool) -> None:
    path.mkdir(parents=True)
    frame = _frame(v3=v3, r6=True)
    frame.to_parquet(path / "monday_r6_on_foundation_scores_training_frame_v1.parquet", index=False)
    metrics = {
        "bad_blocks_v1": 4 if v3 else 3,
        "block_count_v1": 4 if v3 else 3,
        "tail_help_v1": 3 if v3 else 2,
        "precision_v1": 1.0,
        "false_take_ok_blocks_v1": 0,
        "fifty_plus_mfe_blocked_v1": 0,
        "hundred_plus_mfe_blocked_v1": 0,
        "two_hundred_plus_mfe_blocked_v1": 0,
        "strongest_winner_damage_v1": 0,
        "repaired_165_damage_v1": 0,
        "runner_near_miss_blocked_v1": 0,
    }
    selected = {
        "policy_name_v1": "R6_CANDIDATE_04504_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
        "family_v1": "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
        "params_v1": {
            "bad_threshold_v1": 0.85,
            "risky_threshold_v1": 0.85,
            "tail_threshold_v1": 0.85,
            "runner_threshold_v1": 0.30,
            "r5_2_runner_threshold_v1": 0.74,
            "blindspot_threshold_v1": 0.70,
            "use_r5_2_base_v1": True,
            "hard_asof_runner_guard_v1": True,
        },
        "metrics_v1": metrics,
        "candidate_worst_loso_v1": 1.0,
    }
    _write_json(
        path / "summary_v1.json",
        {
            "score_dir_v1": str(score_dir),
            "family_grid_selected_policy_v1": selected,
            "r6_training_started_v1": True,
            "r6_head_count_v1": 5,
            "as_of_column_count_v1": 109,
        },
    )
    _write_json(path / "compare_against_wednesday_r6_v1.json", {"verdict_v1": "SAFE_BUT_NOT_BETTER"})
    pd.DataFrame(
        [
            {
                "policy_name_v1": selected["policy_name_v1"],
                "family_v1": selected["family_v1"],
                "wednesday_safety_pass_v1": True,
                "hard_damage_count_v1": 0,
                "bad_blocks_v1": metrics["bad_blocks_v1"],
                "tail_help_v1": metrics["tail_help_v1"],
                "precision_v1": 1.0,
                "worst_loso_v1": 1.0,
            }
        ]
    ).to_csv(path / "r6_family_grid_replay_v1.csv", index=False)


def test_r6_from_v3_forensics_materializes(tmp_path: Path) -> None:
    v2_score = tmp_path / "v2_score"
    v3_score = tmp_path / "v3_score"
    v2_r6 = tmp_path / "v2_r6"
    v3_r6 = tmp_path / "v3_r6"
    _seed_score(v2_score, v3=False)
    _seed_score(v3_score, v3=True)
    _seed_r6(v2_r6, v2_score, v3=False)
    _seed_r6(v3_r6, v3_score, v3=True)

    out = tmp_path / "out"
    summary = materialize(reports_root=tmp_path, output_dir=out, v3_score_dir=v3_score, v2_score_dir=v2_score, v3_r6_dir=v3_r6, v2_r6_dir=v2_r6)

    assert summary["r6_v3_ran_v1"] is True
    assert summary["v3_contract_used_v1"] is True
    assert summary["bad_delta_vs_v2_v1"] == 1
    assert summary["tail_delta_vs_v2_v1"] == 1
    assert summary["v3_added_rows_selected_v1"] == 1
    assert summary["safety_ok_v1"] is True
    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()
    trace = pd.read_csv(out / "r6_v3_pass_through_audit_v1.csv")
    assert set(trace["first_fail_reason_v1"]) == {"SELECTED_BY_V3_R5_2_BASE"}
