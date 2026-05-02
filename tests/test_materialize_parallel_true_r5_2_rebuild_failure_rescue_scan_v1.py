import json
from pathlib import Path

import pandas as pd

import gx1.scripts.materialize_parallel_true_r5_2_rebuild_failure_rescue_scan_v1 as scan
from gx1.scripts.run_true_r5_2_rebuild_runner_v1 import (
    BASE_FLAG_COL,
    BAD_SCORE_COL,
    RISKY_SCORE_COL,
    RUNNER_SCORE_COL,
    TAIL_SCORE_COL,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_true_r5_2_run(root: Path) -> Path:
    source_dir = root / "source"
    true_dir = root / "true_r5_2"
    source_dir.mkdir(parents=True)
    true_dir.mkdir(parents=True)

    score_rows = []
    label_rows = []
    pred_rows = []
    for idx in range(6):
        is_v3 = idx in {0, 1}
        is_safe_recovery = idx == 2
        is_ambiguous_damage = idx == 3
        is_runner_damage = idx == 4
        should = idx in {0, 1, 2, 3, 4}
        tail = idx in {0, 1, 2, 3}
        bucket = "IGNORE_OR_MONITOR_ONLY"
        if is_ambiguous_damage:
            bucket = "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD"
        elif is_runner_damage:
            bucket = "RUNNER_PROTECT_TARGET"
        elif should:
            bucket = "STRONG_BAD_BLOCK_TARGET"

        score_rows.append(
            {
                "run_id": f"W{idx // 2}",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_timestamp": f"2026-01-01T12:0{idx}:00Z",
                "label_should_not_take_v1": should,
                "tail_10_50_mfe_v1": tail,
                "fifty_plus_mfe_v1": is_ambiguous_damage or is_runner_damage,
                "hundred_plus_mfe_v1": is_ambiguous_damage,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": is_runner_damage,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": is_runner_damage,
                "peak_mfe_bps_v1": 120.0 if (is_ambiguous_damage or is_runner_damage) else 18.0,
                "mae_abs_bps_v1": 40.0 if should else 3.0,
            }
        )
        label_rows.append(
            {
                "candidate_uid": f"c{idx}",
                "new_r5_2_label_bucket_v1": bucket,
                "bad_eligibility_target_v1": should and not (is_ambiguous_damage or is_runner_damage),
                "tail_eligibility_target_v1": tail and not (is_ambiguous_damage or is_runner_damage),
                "risky_attention_target_v1": should,
                "runner_protect_target_v1": is_runner_damage,
                "ambiguous_high_mfe_monitor_v1": is_ambiguous_damage,
                "sample_weight_v1": 3.0 if should else 1.0,
                "protection_weight_v1": 20.0 if (is_ambiguous_damage or is_runner_damage) else 0.0,
                "r5_2_v3_base_flag_v1": is_v3,
            }
        )
        pred_rows.append(
            {
                "candidate_uid": f"c{idx}",
                BAD_SCORE_COL: 0.90 if (is_v3 or is_safe_recovery) else (0.55 if is_ambiguous_damage else 0.10),
                TAIL_SCORE_COL: 0.80 if (is_v3 or is_safe_recovery or is_ambiguous_damage or is_runner_damage) else 0.10,
                RISKY_SCORE_COL: 0.70 if should else 0.10,
                RUNNER_SCORE_COL: 0.10 if (is_v3 or is_safe_recovery) else (0.30 if is_ambiguous_damage else 0.36),
                BASE_FLAG_COL: is_v3 or is_safe_recovery or is_ambiguous_damage or is_runner_damage,
            }
        )

    score_path = source_dir / "score_frame.parquet"
    label_path = source_dir / "label_table.csv"
    pd.DataFrame(score_rows).to_parquet(score_path, index=False)
    pd.DataFrame(label_rows).to_csv(label_path, index=False)
    pd.DataFrame(pred_rows).to_parquet(true_dir / "r5_2_prediction_view_v1.parquet", index=False)
    _write_json(true_dir / "manifest_v1.json", {"input_files_v1": {"score_frame_v1": str(score_path), "label_table_v1": str(label_path)}})
    return true_dir


def test_parallel_true_r5_2_rescue_scan_materializes_read_only_lanes(tmp_path: Path) -> None:
    true_dir = _seed_true_r5_2_run(tmp_path)
    out = tmp_path / "out"

    summary = scan.materialize(reports_root=tmp_path, true_r5_2_dir=true_dir, output_dir=out)

    assert summary["training_started_v1"] is False
    assert summary["r6_started_v1"] is False
    assert summary["raw_true_safety_pass_v1"] is False
    assert summary["decision_v1"] != "NOT_ESTABLISHED"
    assert summary["best_safe_bad_tail_v1"] is not None
    assert summary["best_safe_recovered_rows_v1"] >= 1
    for filename in scan.OUTPUT_FILES.values():
        assert (out / filename).exists()

    decision = json.loads((out / scan.OUTPUT_FILES["decision"]).read_text())
    assert decision["do_not_feed_raw_true_package_to_r6_v1"] is True
    audit = pd.read_csv(out / scan.OUTPUT_FILES["audit"])
    assert set(audit["status_v1"]) == {"PASS"}
    lane01 = pd.read_csv(out / scan.OUTPUT_FILES["lane01"])
    assert not lane01.empty
    lane07 = json.loads((out / scan.OUTPUT_FILES["lane07"]).read_text())
    assert lane07["layer_name"] == "LANE_07_SCORE_CALIBRATION_AND_MARGIN_SCAN_V1"
    assert lane07["damage_rows_v1"] >= 1
