import json
from pathlib import Path

import pandas as pd

import gx1.scripts.materialize_design_r5_2_objective_v2_rebuild_next_v1 as design


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_investigation(path: Path) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "summary_v1.json",
        {
            "decision_v1": "R5_2_OBJECTIVE_V2_REBUILD",
            "current_r6_rescue_bad_tail_v1": [88, 57],
            "wednesday_bad_tail_v1": [180, 149],
            "gap_to_wednesday_bad_tail_v1": [92, 92],
        },
    )
    rows = []
    plan = [
        ("NOT_IN_RESCUED_R5_2_BASE", "R6_HEAD_SIGNAL_WEAK", 230),
        ("DANGEROUS_OR_PROTECTED", "R6_HEAD_SIGNAL_UNSAFE", 149),
        ("R6_COULD_RECOVER_BUT_BASE_GATE_BLOCKS", "R6_HEAD_SIGNAL_STRONG_BUT_BASE_BLOCKED", 5),
    ]
    idx = 0
    for bucket, signal, count in plan:
        for _ in range(count):
            rows.append(
                {
                    "candidate_uid": f"candidate_{idx:04d}",
                    "trade_uid": f"trade_{idx:04d}",
                    "decision_timestamp": f"2026-01-05T10:{idx % 60:02d}:00Z",
                    "label_should_not_take_v1": True,
                    "tail_10_50_mfe_v1": idx < 141,
                    "post_rescue_gap_bucket_v1": bucket,
                    "r6_head_signal_class_v1": signal,
                }
            )
            idx += 1
    pd.DataFrame(rows).to_csv(path / "post_rescue_recall_gap_map_v1.csv", index=False)
    _write_json(
        path / "r5_2_objective_v2_opportunity_scan_v1.json",
        {
            "raw_true_rebuild_findings_v1": {
                "raw_true_bad_tail_v1": [97, 60],
                "rescued_bad_tail_v1": [88, 57],
                "raw_true_safety_fail_v1": {
                    "fifty_plus_v1": 4,
                    "hundred_plus_v1": 2,
                    "strongest_v1": 1,
                    "ambiguous_v1": 3,
                    "runner_protect_v1": 1,
                },
            },
            "outside_base_simulation_safe_rule_count_v1": 0,
        },
    )
    _write_json(
        path / "r5_2_v2_vs_r6_outside_base_decision_matrix_v1.json",
        {
            "decision_v1": "R5_2_OBJECTIVE_V2_REBUILD",
            "base_gate_decision_v1": "R6_ADDON_ALLOWED_BUT_SCORES_TOO_WEAK",
        },
    )


def _seed_label_table(path: Path) -> None:
    path.mkdir(parents=True)
    rows = []
    bucket_plan = [
        ("STRONG_BAD_BLOCK_TARGET", 127),
        ("TAIL_CONTROL_TARGET", 198),
        ("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD", 130),
        ("RUNNER_PROTECT_TARGET", 462),
        ("IGNORE_OR_MONITOR_ONLY", 997),
    ]
    idx = 0
    for bucket, count in bucket_plan:
        for _ in range(count):
            rows.append(
                {
                    "candidate_uid": f"candidate_{idx:04d}",
                    "trade_uid": f"trade_{idx:04d}",
                    "decision_timestamp": f"2026-01-05T10:{idx % 60:02d}:00Z",
                    "new_r5_2_label_bucket_v1": bucket,
                    "label_should_not_take_v1": bucket in {"STRONG_BAD_BLOCK_TARGET", "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD", "RUNNER_PROTECT_TARGET"},
                    "tail_10_50_mfe_v1": bucket == "TAIL_CONTROL_TARGET",
                    "fifty_plus_mfe_v1": bucket in {"AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD", "RUNNER_PROTECT_TARGET"} and idx % 2 == 0,
                    "hundred_plus_mfe_v1": bucket == "RUNNER_PROTECT_TARGET" and idx % 5 == 0,
                    "two_hundred_plus_mfe_v1": bucket == "RUNNER_PROTECT_TARGET" and idx % 17 == 0,
                    "strongest_winner_path_v1": bucket == "RUNNER_PROTECT_TARGET" and idx % 3 == 0,
                    "r6_label_repaired_165_like_runner_v1": False,
                    "r6_label_runner_near_miss_v1": bucket == "RUNNER_PROTECT_TARGET" and idx % 11 == 0,
                }
            )
            idx += 1
    pd.DataFrame(rows).to_csv(path / "r5_2_pocket_label_table_v1.csv", index=False)


def test_design_r5_2_objective_v2_rebuild_next_materializes_spec(tmp_path: Path) -> None:
    investigation_dir = tmp_path / "investigation"
    label_dir = tmp_path / "label_objective"
    rescue_r6_dir = tmp_path / "r6"
    output_dir = tmp_path / "out"
    rescue_r6_dir.mkdir()
    _seed_investigation(investigation_dir)
    _seed_label_table(label_dir)

    summary = design.materialize(
        reports_root=tmp_path,
        investigation_dir=investigation_dir,
        label_objective_dir=label_dir,
        rescue_r6_dir=rescue_r6_dir,
        output_dir=output_dir,
    )

    assert summary["training_started_v1"] is False
    assert summary["r6_started_v1"] is False
    assert summary["new_baseline_built_v1"] is False
    assert summary["new_feature_surface_built_v1"] is False
    assert summary["decision_v1"] == "R5_2_OBJECTIVE_V2_DESIGN_READY_FOR_RUNNER_IMPLEMENTATION"
    assert summary["next_action_v1"] == "IMPLEMENT_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_RUNNER"
    assert summary["gap_bucket_counts_v1"] == {
        "NOT_IN_RESCUED_R5_2_BASE": 230,
        "DANGEROUS_OR_PROTECTED": 149,
        "R6_COULD_RECOVER_BUT_BASE_GATE_BLOCKS": 5,
    }
    for filename in design.OUTPUT_FILES.values():
        assert (output_dir / filename).exists()

    label_contract = json.loads((output_dir / "r5_2_objective_v2_label_contract_v1.json").read_text())
    assert len(label_contract["buckets_v1"]) == 6
    assert label_contract["ambiguous_high_mfe_bad_positive_allowed_v1"] is False

    weights = json.loads((output_dir / "r5_2_objective_v2_weight_and_cost_spec_v1.json").read_text())
    assert len(weights["candidate_weight_profiles_v1"]) >= 5

    architecture = json.loads((output_dir / "r5_2_objective_v2_model_architecture_spec_v1.json").read_text())
    assert "r5_2_v2_final_base_membership" in architecture["final_outputs_v1"]

    base_contract = json.loads((output_dir / "r5_2_objective_v2_base_membership_contract_v1.json").read_text())
    assert base_contract["final_base_rule_v1"] == "r5_2_v2_base_membership_pre_veto AND NOT r5_2_v2_hard_protection_veto"

    audit = pd.read_csv(output_dir / "consistency_audit_v1.csv")
    assert set(audit["status_v1"]) == {"PASS"}
