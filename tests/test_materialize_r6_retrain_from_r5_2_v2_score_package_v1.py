import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_r6_retrain_from_r5_2_v2_score_package_v1 import OUTPUT_FILES, materialize
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import R5_2_BASE_MEMBERSHIP_CONTRACT_V2, R5_2_BAD_PROB, R5_2_RUNNER_PROB


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frame(v2: bool, r6: bool = False) -> pd.DataFrame:
    rows = []
    for idx in range(4):
        should = idx in {0, 1, 2}
        selected = idx == 0 or (v2 and idx in {1, 2})
        rows.append(
            {
                "run_id": f"W{idx}",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_timestamp": f"2026-01-01T12:0{idx}:00Z",
                "split_scope_v1": "TRAIN",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "batch_scope_v1": "BATCH_04" if idx < 2 else "BATCH_05",
                "label_should_not_take_v1": should,
                "tail_10_50_mfe_v1": idx in {0, 1},
                "take_was_ok_v1": not should,
                "fifty_plus_mfe_v1": False,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "is_repaired_165_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": False,
                "pred__entry_r5_should_not_take__prob_true_v1": 0.9 if should else 0.1,
                "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.8 if should else 0.1,
                "pred__entry_r5_runner_protect__prob_true_v1": 0.2,
                "r5_1_bad_blocker_score_v1": 0.9 if should else 0.1,
                "r5_1_runner_guard_score_v1": 0.2,
                R5_2_BAD_PROB: 0.5 if should else 0.1,
                R5_2_RUNNER_PROB: 0.2,
                "r5_2_selected_candidate__block_v1": selected,
                "pred__entry_r6_bad_risk__prob_true_v1": 0.9 if should else 0.1,
                "pred__entry_r6_runner_protector__prob_true_v1": 0.1,
                "pred__entry_r6_tail_control_10_50__prob_true_v1": 0.9 if idx in {0, 1} else 0.1,
                "pred__entry_r6_risky_allow__prob_true_v1": 0.99 if should else 0.1,
                "pred__entry_r6_batch04_blindspot__prob_true_v1": 0.1,
                "asof_runner_guard_v1": False,
                "selected_candidate_block_v1": selected if r6 else False,
            }
        )
    return pd.DataFrame(rows)


def _seed_score(path: Path, *, v2: bool) -> None:
    path.mkdir(parents=True)
    _frame(v2).to_parquet(path / "monday_r6_foundation_score_frame_v1.parquet", index=False)
    _write_json(
        path / "score_rebuild_summary_v1.json",
        {
            "r5_2_selected_policy_v1": {
                "base_membership_active_contract_id_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2["contract_id_v1"] if v2 else "V1",
            }
        },
    )


def _seed_r6(path: Path, *, v2: bool) -> None:
    path.mkdir(parents=True)
    frame = _frame(v2, r6=True)
    frame.to_parquet(path / "monday_r6_on_foundation_scores_training_frame_v1.parquet", index=False)
    frame.to_parquet(path / "monday_r6_on_foundation_scores_prediction_view_v1.parquet", index=False)
    metrics = {
        "bad_blocks_v1": 3 if v2 else 1,
        "block_count_v1": 3 if v2 else 1,
        "tail_help_v1": 2 if v2 else 1,
        "precision_v1": 1.0,
        "false_take_ok_blocks_v1": 0,
        "fifty_plus_mfe_blocked_v1": 0,
        "hundred_plus_mfe_blocked_v1": 0,
        "two_hundred_plus_mfe_blocked_v1": 0,
        "strongest_winner_damage_v1": 0,
        "repaired_165_damage_v1": 0,
        "runner_near_miss_blocked_v1": 0,
        "quarantine_blocks_v1": 0,
        "row_count_v1": 4,
    }
    selected = {
        "policy_name_v1": "R6_CANDIDATE_04504_R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
        "family_v1": "R6_R5_2_ULTRA_SAFE_TAIL_RISKY_ADDON",
        "params_v1": {
            "bad_threshold_v1": 0.85,
            "risky_threshold_v1": 0.99,
            "tail_threshold_v1": 0.85,
            "runner_threshold_v1": 0.3,
            "r5_2_runner_threshold_v1": 0.74,
            "blindspot_threshold_v1": 0.7,
            "hard_asof_runner_guard_v1": True,
            "use_r5_2_base_v1": True,
        },
        "metrics_v1": metrics,
        "candidate_worst_loso_v1": 1.0,
    }
    _write_json(
        path / "summary_v1.json",
        {
            "score_dir_v1": str(path.parent / "v2_score") if v2 else str(path.parent / "old_score"),
            "family_grid_selected_policy_v1": selected,
            "r6_training_started_v1": True,
            "not_freeze_or_promo_v1": True,
            "not_live_gate_v1": True,
            "as_of_column_count_v1": 109,
            "wednesday_locked_policy_replay_v1": {"r5_2_base_block_count_v1": 3 if v2 else 1},
            "r6_family_grid_replay_v1": {"max_observed_bad_blocks_v1": 3 if v2 else 1, "max_observed_tail_help_v1": 2 if v2 else 1},
        },
    )
    _write_json(path / "compare_against_wednesday_r6_v1.json", {"verdict_v1": "SAFE_BUT_NOT_BETTER", "safety_failures_v1": []})
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


def test_r6_retrain_from_v2_materializer_traces_added_rows(tmp_path: Path) -> None:
    old_score = tmp_path / "old_score"
    v2_score = tmp_path / "v2_score"
    old_r6 = tmp_path / "old_r6"
    v2_r6 = tmp_path / "v2_r6"
    _seed_score(old_score, v2=False)
    _seed_score(v2_score, v2=True)
    _seed_r6(old_r6, v2=False)
    _seed_r6(v2_r6, v2=True)

    out = tmp_path / "out"
    summary = materialize(
        reports_root=tmp_path,
        output_dir=out,
        v2_score_dir=v2_score,
        old_score_dir=old_score,
        v2_r6_dir=v2_r6,
        old_r6_dir=old_r6,
        v2_audit_dir=tmp_path / "audit",
    )

    assert summary["v2_score_package_used_v1"] is True
    assert summary["v2_contract_used_v1"] is True
    assert summary["bad_delta_vs_previous_v1"] == 2
    assert summary["tail_delta_vs_previous_v1"] == 1
    assert summary["v2_added_rows_selected_v1"] == 2
    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()
    trace = pd.read_csv(out / "v2_added_rows_r6_trace_v1.csv")
    assert set(trace["first_fail_reason_v1"]) == {"SELECTED_BY_V2_R5_2_BASE"}
