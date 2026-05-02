import json
from pathlib import Path

import pandas as pd

import gx1.scripts.materialize_safe_true_r5_2_rescue_base_rule_v1 as rescue
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


def _seed_scan_lock(path: Path) -> None:
    _write_json(
        path / "rescue_or_retrain_decision_v1.json",
        {
            "best_safe_rescue_rule_v1": {
                "rule_id_v1": rescue.RULE_ID,
                "base_source_v1": "v3_union_plus_true_scores",
                "bad_threshold_v1": 0.85,
                "tail_threshold_v1": 0.70,
                "risky_threshold_v1": 0.65,
                "runner_cap_v1": 0.20,
                "consensus_min_v1": 1,
                "margin_min_v1": 0.0,
                "safety_pass_v1": True,
                "bad_v1": 3,
                "tail_v1": 3,
                "bad_delta_vs_v3_v1": 1,
                "tail_delta_vs_v3_v1": 1,
                "precision_v1": 1.0,
                "worst_loso_v1": 1.0,
                "repaired_like_overlap_v1": 0,
                "fifty_plus_overlap_v1": 0,
                "hundred_plus_overlap_v1": 0,
                "two_hundred_plus_overlap_v1": 0,
                "strongest_winner_overlap_v1": 0,
                "runner_near_miss_overlap_v1": 0,
                "ambiguous_high_mfe_included_v1": 0,
                "runner_protect_included_v1": 0,
            }
        },
    )


def _seed_true_run(root: Path) -> Path:
    source_dir = root / "source"
    true_dir = root / "true"
    source_dir.mkdir(parents=True)
    true_dir.mkdir(parents=True)
    score_rows = []
    label_rows = []
    pred_rows = []
    for idx in range(6):
        in_v3 = idx in {0, 1}
        safe_add = idx == 2
        ambiguous_damage = idx == 3
        runner_damage = idx == 4
        should = idx < 5
        tail = idx in {0, 1, 2, 3}
        bucket = "STRONG_BAD_BLOCK_TARGET" if should else "IGNORE_OR_MONITOR_ONLY"
        if ambiguous_damage:
            bucket = "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD"
        if runner_damage:
            bucket = "RUNNER_PROTECT_TARGET"
        score_rows.append(
            {
                "run_id": f"W{idx // 2}",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": str(idx),
                "decision_timestamp": f"2026-01-01T12:0{idx}:00Z",
                "label_should_not_take_v1": should,
                "tail_10_50_mfe_v1": tail,
                "fifty_plus_mfe_v1": ambiguous_damage or runner_damage,
                "hundred_plus_mfe_v1": ambiguous_damage,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": runner_damage,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_near_miss_v1": runner_damage,
                "peak_mfe_bps_v1": 120.0 if (ambiguous_damage or runner_damage) else 15.0,
                "mae_abs_bps_v1": 45.0 if should else 3.0,
            }
        )
        label_rows.append(
            {
                "candidate_uid": f"c{idx}",
                "new_r5_2_label_bucket_v1": bucket,
                "bad_eligibility_target_v1": should and not (ambiguous_damage or runner_damage),
                "tail_eligibility_target_v1": tail and not (ambiguous_damage or runner_damage),
                "risky_attention_target_v1": should,
                "runner_protect_target_v1": runner_damage,
                "ambiguous_high_mfe_monitor_v1": ambiguous_damage,
                "sample_weight_v1": 3.0,
                "protection_weight_v1": 20.0 if (ambiguous_damage or runner_damage) else 0.0,
                "r5_2_v3_base_flag_v1": in_v3,
            }
        )
        pred_rows.append(
            {
                "candidate_uid": f"c{idx}",
                BAD_SCORE_COL: 0.90 if (in_v3 or safe_add) else 0.55,
                TAIL_SCORE_COL: 0.80 if (in_v3 or safe_add or ambiguous_damage or runner_damage) else 0.10,
                RISKY_SCORE_COL: 0.70 if should else 0.10,
                RUNNER_SCORE_COL: 0.10 if (in_v3 or safe_add) else 0.35,
                BASE_FLAG_COL: in_v3 or safe_add or ambiguous_damage or runner_damage,
            }
        )
    score_path = source_dir / "score_frame.parquet"
    label_path = source_dir / "label_table.csv"
    pd.DataFrame(score_rows).to_parquet(score_path, index=False)
    pd.DataFrame(label_rows).to_csv(label_path, index=False)
    pd.DataFrame(pred_rows).to_parquet(true_dir / "r5_2_prediction_view_v1.parquet", index=False)
    _write_json(true_dir / "manifest_v1.json", {"input_files_v1": {"score_frame_v1": str(score_path), "label_table_v1": str(label_path)}})
    return true_dir


def test_safe_true_r5_2_rescue_materializes_package_and_blocks_raw_true(tmp_path: Path) -> None:
    scan_dir = tmp_path / "scan"
    true_dir = _seed_true_run(tmp_path)
    _seed_scan_lock(scan_dir)
    out = tmp_path / "out"

    summary = rescue.materialize(reports_root=tmp_path, true_r5_2_dir=true_dir, rescue_scan_dir=scan_dir, output_dir=out)

    assert summary["training_started_v1"] is False
    assert summary["r6_started_v1"] is False
    assert summary["decision_v1"] == "TRUE_R5_2_RESCUE_BASE_RULE_PASS"
    assert summary["rescued_bad_v1"] == 3
    assert summary["rescued_tail_v1"] == 3
    assert summary["raw_true_blocked_from_r6_v1"] is True
    for filename in rescue.OUTPUT_FILES.values():
        assert (out / filename).exists()

    manifest = json.loads((out / rescue.OUTPUT_FILES["r6_manifest"]).read_text())
    assert manifest["base_flag_for_r6_v1"] == rescue.RESCUE_BASE_FLAG_COL
    assert manifest["raw_true_base_flag_not_allowed_v1"] == BASE_FLAG_COL
    package = pd.read_parquet(out / rescue.OUTPUT_FILES["score_package"])
    assert rescue.RESCUE_BASE_FLAG_COL in package.columns
    assert int(package[rescue.RESCUE_BASE_FLAG_COL].sum()) == 3
    forensics = pd.read_csv(out / rescue.OUTPUT_FILES["forensics"])
    assert "RAW_TRUE_SAFETY_DAMAGE_ROW" in set(forensics["forensic_row_type_v1"])
