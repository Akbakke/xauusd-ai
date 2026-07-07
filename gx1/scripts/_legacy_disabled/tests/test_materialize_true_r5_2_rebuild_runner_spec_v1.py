import json
from pathlib import Path

import pandas as pd

from gx1.scripts.materialize_true_r5_2_rebuild_runner_spec_v1 import OUTPUT_FILES, materialize


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _label_table() -> pd.DataFrame:
    rows = []
    missed_plan = [
        ("STRONG_BAD_BLOCK_TARGET", 96, True, False),
        ("TAIL_CONTROL_TARGET", 147, False, True),
        ("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD", 130, True, False),
        ("RUNNER_PROTECT_TARGET", 17, True, False),
    ]
    idx = 0
    for bucket, count, bad_label, tail_label in missed_plan:
        for _ in range(count):
            rows.append(
                {
                    "candidate_uid": f"candidate_{idx:04d}",
                    "trade_uid": f"trade_{idx:04d}",
                    "decision_timestamp": f"2026-01-05T13:{idx % 60:02d}:00Z",
                    "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                    "label_should_not_take_v1": bad_label,
                    "tail_10_50_mfe_v1": tail_label,
                    "r5_2_v3_base_flag_v1": False,
                    "new_r5_2_label_bucket_v1": bucket,
                    "bad_eligibility_target_v1": bucket == "STRONG_BAD_BLOCK_TARGET",
                }
            )
            idx += 1
    while idx < 1914:
        rows.append(
            {
                "candidate_uid": f"candidate_{idx:04d}",
                "trade_uid": f"trade_{idx:04d}",
                "decision_timestamp": f"2026-01-05T13:{idx % 60:02d}:00Z",
                "calendar_quarantine_status_v1": "QUARANTINED" if idx >= 1852 else "ACTIVE_CANDIDATE",
                "label_should_not_take_v1": False,
                "tail_10_50_mfe_v1": False,
                "r5_2_v3_base_flag_v1": idx < 470,
                "new_r5_2_label_bucket_v1": "IGNORE_OR_MONITOR_ONLY",
                "bad_eligibility_target_v1": False,
            }
        )
        idx += 1
    return pd.DataFrame(rows)


def _write_label_objective_dir(path: Path) -> None:
    path.mkdir()
    _label_table().to_csv(path / "r5_2_pocket_label_table_v1.csv", index=False)
    _write_json(
        path / "r5_2_new_label_contract_v1.json",
        {"contract_id_v1": "R5_2_LABEL_OBJECTIVE_BAD_TAIL_ELIGIBILITY_WITH_HARD_WINNER_PROTECTION_V1"},
    )
    _write_json(
        path / "r5_2_objective_weighting_spec_v1.json",
        {
            "positive_class_weights_v1": {
                "STRONG_BAD_BLOCK_TARGET": 3.0,
                "TAIL_CONTROL_TARGET": 2.5,
                "RISKY_ALLOW_TARGET": 1.25,
            },
            "protection_costs_v1": {
                "RUNNER_PROTECT_TARGET": 10.0,
                "HUNDRED_OR_TWO_HUNDRED_MFE": 20.0,
                "STRONGEST_WINNER": 20.0,
                "REPAIRED_LIKE": 20.0,
            },
        },
    )
    _write_json(
        path / "r5_2_rebuild_experiment_spec_v1.json",
        {
            "recommended_design_v1": "POCKET_AWARE_MULTI_HEAD_R5_2_BAD_TAIL_ELIGIBILITY_WITH_RUNNER_PROTECTION",
            "model_family_v1": "XGB-style deterministic multi-head matching current stack",
            "heads_v1": [
                "bad_eligibility_head",
                "tail_10_50_eligibility_head",
                "risky_attention_head",
                "runner_protect_head",
            ],
            "split_eval_v1": "Existing train/validation/holdout/LOSO with quarantine eval-only.",
        },
    )


def test_true_r5_2_rebuild_runner_spec_materializes_complete_lock(tmp_path: Path) -> None:
    label_dir = tmp_path / "label_objective"
    score_dir = tmp_path / "score"
    r6_dir = tmp_path / "r6"
    score_dir.mkdir()
    r6_dir.mkdir()
    _write_label_objective_dir(label_dir)
    _write_json(
        score_dir / "summary_v1.json",
        {
            "foundation_dir_v1": "/tmp/foundation",
            "row_count_v1": 1914,
            "active_rows_v1": 1852,
            "quarantine_rows_v1": 62,
            "as_of_column_count_v1": 109,
            "base_feature_count_v1": 88,
            "r5_2_feature_count_v1": 99,
        },
    )

    out = tmp_path / "out"
    summary = materialize(output_dir=out, reports_root=tmp_path, label_objective_dir=label_dir, v3_score_dir=score_dir, old_v3_r6_dir=r6_dir)

    assert summary["decision_v1"] == "TRUE_R5_2_REBUILD_RUNNER_SPEC_COMPLETE"
    assert summary["next_action_v1"] == "IMPLEMENT_TRUE_R5_2_REBUILD_RUNNER_NEXT"
    assert summary["training_started_v1"] is False
    assert summary["r6_started_v1"] is False
    assert summary["missed_bad_tail_bucket_counts_v1"] == {
        "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD": 130,
        "RUNNER_PROTECT_TARGET": 17,
        "STRONG_BAD_BLOCK_TARGET": 96,
        "TAIL_CONTROL_TARGET": 147,
    }
    for filename in OUTPUT_FILES.values():
        assert (out / filename).exists()

    runner_spec = json.loads((out / "true_r5_2_rebuild_runner_spec_v1.json").read_text())
    assert runner_spec["future_runner_module_v1"] == "gx1.scripts.run_true_r5_2_rebuild_runner_v1"
    assert "--run-true-r5-2-rebuild" in runner_spec["future_command_template_v1"]

    audit = pd.read_csv(out / "consistency_audit_v1.csv")
    assert set(audit["status_v1"]) == {"PASS"}
