import json
import sys
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_monday_narrow_retrain_failure_forensics_v1 import main


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_materialize_monday_narrow_retrain_failure_forensics_v1(tmp_path, monkeypatch):
    reports_root = tmp_path / "reports"
    reports_root.mkdir(parents=True)
    run_dir = reports_root / "ALL_TRADE_REVIEW_LEDGER_20260424T170555Z_MONDAY_NARROW_RETRAIN_RUN_V1"
    run_dir.mkdir()

    rows = []
    for idx in range(12):
        block = idx in {0, 1, 2, 3, 4, 5}
        bad = idx in {0, 1, 6, 7}
        rows.append(
            {
                "candidate_uid": f"cand::{idx:04d}",
                "monday_narrow_block_v1": block,
                "r6_label_bad_risk_v1": bad,
                "r6_label_tail_control_10_50_v1": idx in {0, 8, 9},
                "r6_label_runner_50_mfe_v1": idx in {2, 3},
                "r6_label_runner_100_mfe_v1": idx == 3,
                "r6_label_runner_200_mfe_v1": idx == 4,
                "r6_label_strong_low_mae_runner_v1": idx == 5,
                "r6_label_runner_near_miss_v1": idx == 2,
                "r6_label_repaired_165_like_runner_v1": False,
                "r6_label_runner_protect_v1": idx in {2, 3, 4, 5},
                "pred__monday_narrow__runner_protector__prob_true_v1": 0.04 if block else 0.3,
                "pred__monday_narrow__bad_risk__prob_true_v1": 0.7 if block else 0.2,
                "pred__monday_narrow__tail_control_10_50__prob_true_v1": 0.1,
                "pred__monday_narrow__risky_allow__prob_true_v1": 0.2,
                "pred__monday_narrow__batch04_blindspot__prob_true_v1": 0.6 if idx == 2 else 0.1,
                "as_of_pre_entry_vol_exp_comp_score_v1": float(idx) / 10.0,
                "as_of_pre_entry_directional_asymmetry_score_v1": float(idx),
                "as_of_pre_entry_swing_retracement_alignment_score_v1": 1.0,
                "as_of_pre_entry_tail_leakage_pocket_score_v1": 0.3,
                "as_of_pre_entry_runner_protection_guard_score_v1": 0.6 if idx in {2, 3, 4, 5} else 0.4,
            }
        )
    pd.DataFrame(rows).to_parquet(run_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_policy_prediction_view_v1.parquet", index=False)
    _write_json(
        run_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_eval_summary_v1.json",
        {
            "global_metric_v1": {
                "bad_blocks_v1": 2,
                "tail_help_v1": 1,
                "global_precision_v1": 2 / 6,
                "worst_loso_precision_v1": 0.2,
                "fifty_plus_mfe_block_count_v1": 2,
                "hundred_plus_mfe_block_count_v1": 1,
                "two_hundred_plus_mfe_block_count_v1": 1,
                "strongest_winner_path_damage_v1": 2,
                "runner_near_miss_block_count_v1": 1,
            }
        },
    )
    pd.DataFrame(
        [
            {
                "reference_v1": "FROZEN_WEDNESDAY_R6_BENCHMARK",
                "id_v1": "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1",
                "kind_v1": "BENCHMARK",
                "bad_blocks_v1": 180,
                "tail_help_v1": 149,
                "global_precision_v1": 0.97,
                "worst_loso_precision_v1": 0.92,
                "repaired_165_damage_v1": 0,
                "fifty_plus_mfe_block_count_v1": 1,
                "hundred_plus_mfe_block_count_v1": 0,
                "two_hundred_plus_mfe_block_count_v1": 0,
                "strongest_winner_path_damage_v1": 0,
                "runner_near_miss_block_count_v1": None,
                "delta_vs_frozen_global_precision_v1": 0,
                "delta_vs_frozen_worst_loso_precision_v1": 0,
                "delta_vs_frozen_tail_help_v1": 0,
            },
            {
                "reference_v1": "MONDAY_NARROW_RETRAIN_CANDIDATE",
                "id_v1": "CURRENT_RUN",
                "kind_v1": "CANDIDATE",
                "bad_blocks_v1": 2,
                "tail_help_v1": 1,
                "global_precision_v1": 2 / 6,
                "worst_loso_precision_v1": 0.2,
                "repaired_165_damage_v1": 0,
                "fifty_plus_mfe_block_count_v1": 2,
                "hundred_plus_mfe_block_count_v1": 1,
                "two_hundred_plus_mfe_block_count_v1": 1,
                "strongest_winner_path_damage_v1": 2,
                "runner_near_miss_block_count_v1": 1,
                "delta_vs_frozen_global_precision_v1": -0.64,
                "delta_vs_frozen_worst_loso_precision_v1": -0.72,
                "delta_vs_frozen_tail_help_v1": -148,
            },
        ]
    ).to_csv(run_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_compare_against_report_v1.csv", index=False)
    pd.DataFrame(
        [
            {"pocket_v1": "50_plus_mfe_seed_pocket", "blocked_count_v1": 2, "hard_guard_v1": True},
            {"pocket_v1": "missed_should_not_take_pocket", "blocked_count_v1": 2, "hard_guard_v1": False},
        ]
    ).to_csv(run_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_pocket_report_v1.csv", index=False)
    pd.DataFrame(
        [
            {"scope_v1": "BATCH_01", "should_not_take_precision_v1": 0.2},
            {"scope_v1": "BATCH_02", "should_not_take_precision_v1": 0.5},
        ]
    ).to_csv(run_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_loso_metrics_v1.csv", index=False)
    _write_json(
        run_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_verdict_package_v1.json",
        {
            "verdict_v1": "CANDIDATE_FEATURES_INSUFFICIENT",
            "candidate_disqualified_v1": True,
            "hard_fail_reasons_v1": ["50+ blocked > 1"],
        },
    )
    pd.DataFrame({"feature_name_v1": ["as_of_pre_entry_runner_protection_guard_score_v1"]}).to_csv(
        run_dir / "shadow_meta_all_trade_review_monday_narrow_retrain_feature_manifest_v1.csv",
        index=False,
    )

    extension_dir = reports_root / "OUT"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_monday_narrow_retrain_failure_forensics_v1.py",
            "--reports-root",
            str(reports_root),
            "--run-dir",
            str(run_dir),
            "--extension-dir",
            str(extension_dir),
        ],
    )
    main()

    summary = json.loads((extension_dir / "summary_v1.json").read_text(encoding="utf-8"))
    decision = json.loads((extension_dir / "go_or_no_go_next_step_v1.json").read_text(encoding="utf-8"))
    collapse = json.loads((extension_dir / "narrow_retrain_failure_collapse_forensics_v1.json").read_text(encoding="utf-8"))
    runner = json.loads((extension_dir / "runner_protection_failure_analysis_v1.json").read_text(encoding="utf-8"))
    feature_review = pd.read_csv(extension_dir / "feature_proxy_behavior_review_v1.csv")
    audit = pd.read_csv(extension_dir / "consistency_audit_v1.csv")

    assert summary["run_verdict_v1"] == "CANDIDATE_FEATURES_INSUFFICIENT"
    assert summary["forensics_decision_v1"] == "STRENGTHEN_RUNNER_PROTECTION_BEFORE_ANY_NEW_RETRAIN"
    assert decision["decision_v1"] == "STRENGTHEN_RUNNER_PROTECTION_BEFORE_ANY_NEW_RETRAIN"
    assert "blocker heads" in collapse["collapse_explanation_v1"].lower()
    assert runner["overblocked_pockets_v1"]["runner_50_blocked_v1"] == 2
    assert not feature_review.empty
    assert audit["status_v1"].astype("string").eq("PASS").all()
