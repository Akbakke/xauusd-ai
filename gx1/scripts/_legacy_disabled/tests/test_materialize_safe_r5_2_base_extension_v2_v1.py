import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_safe_r5_2_base_extension_v2_v1 import OUTPUT_FILES, materialize
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import R5_2_BASE_MEMBERSHIP_CONTRACT_V2, R5_2_BAD_PROB, R5_2_RUNNER_PROB


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _frame(v2: bool) -> pd.DataFrame:
    rows = []
    for idx in range(79):
        base_old = idx < 76
        v2_add = v2 and idx in {1, 2}
        if v2:
            v2_add = idx in {76, 77}
        should = idx < 78
        rows.append(
            {
                "run_id": f"W{idx:03d}",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_timestamp": f"2026-01-01T12:{idx % 60:02d}:00Z",
                "split_scope_v1": "TRAIN",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "label_should_not_take_v1": should,
                "tail_10_50_mfe_v1": idx < 48 or (v2 and idx == 76),
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
                "r5_selected_candidate__block_v1": base_old,
                "r5_1_selected_candidate__block_v1": base_old,
                "r5_2_selected_candidate__block_v1": base_old or v2_add,
                "blocker_score_v1": 0.5 if should else 0.1,
                "runner_protector_score_v1": 0.2,
            }
        )
    return pd.DataFrame(rows)


def _seed_score_dir(path: Path, *, v2: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _frame(v2).to_parquet(path / "monday_r6_foundation_score_frame_v1.parquet", index=False)
    policy = {
        "metrics_v1": {"bad_blocks_v1": 78 if v2 else 76, "tail_help_v1": 49 if v2 else 48},
        "base_membership_active_contract_id_v1": R5_2_BASE_MEMBERSHIP_CONTRACT_V2["contract_id_v1"] if v2 else "V1",
        "v2_contract_applied_v1": v2,
        "base_metrics_before_contract_v1": {"block_count_v1": 1},
        "v1_contract_metrics_v1": {"block_count_v1": 1},
    }
    _write_json(path / "score_rebuild_summary_v1.json", {"r5_2_selected_policy_v1": policy})
    _write_json(
        path / "summary_v1.json",
        {
            "decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED",
            "explicit_score_rebuild_flag_v1": True,
            "r6_heads_trained_v1": False,
            "foundation_dir_v1": "foundation",
            "active_rows_v1": 4,
            "quarantine_rows_v1": 0,
            "as_of_column_count_v1": 109,
        },
    )


def test_safe_r5_2_base_extension_v2_materializes_pass(tmp_path: Path) -> None:
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    _seed_score_dir(old_dir, v2=False)
    _seed_score_dir(new_dir, v2=True)
    _write_json(
        scan_dir / "parallel_scan_aggregator_v1.json",
        {
            "best_safe_bad_candidate_v1": {
                "lane_v1": "LANE_02_R5_2_BASE_EXTENSION_V2_SCAN_V1",
                "rule_id_v1": "BASE_EXTENSION_R5_2_BAD",
                "params_json_v1": json.dumps(
                    {
                        "bad_source_v1": "R5_2_BAD",
                        "bad_threshold_v1": 0.35,
                        "immediate_mae_threshold_v1": 0.75,
                        "r5_runner_max_v1": 0.45,
                        "r5_1_runner_max_v1": 0.45,
                        "r5_2_runner_max_v1": 0.35,
                    }
                ),
            }
        },
    )
    pd.DataFrame({"lane_v1": ["LANE_02_R5_2_BASE_EXTENSION_V2_SCAN_V1"]}).to_csv(scan_dir / "lane_02_r5_2_base_extension_v2_scan_v1.csv", index=False)
    out = tmp_path / "out"

    summary = materialize(
        reports_root=tmp_path,
        output_dir=out,
        old_score_dir=old_dir,
        new_score_dir=new_dir,
        scan_dir=scan_dir,
        foundation_dir=tmp_path / "foundation",
    )

    assert summary["decision_v1"] == "R5_2_V2_SCORE_REBUILD_PASS"
    assert summary["next_action_v1"] == "RUN_R6_RETRAIN_FROM_R5_2_V2_SCORE_PACKAGE_EXPLICIT_FLAG"
    assert summary["bad_uplift_v1"] == 2
    assert summary["tail_uplift_v1"] == 1
    assert summary["v2_added_rows_v1"] == 2
    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()
    gate = json.loads((out / "r5_2_v2_score_rebuild_gate_v1.json").read_text())
    assert gate["checks_v1"]["contract_ok_v1"] is True
    added = pd.read_csv(out / "v2_added_rows_forensics_v1.csv")
    assert set(added["safe_or_reject_v1"]) == {"SAFE_RECOVERABLE"}
