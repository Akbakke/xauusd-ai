import json
from pathlib import Path

import pandas as pd
import pytest

import gx1.scripts.run_r5_2_objective_v2_parallel_rebuild_runner_v1 as runner
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import SCORE_FRAME, SUMMARY as SCORE_SUMMARY


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _label_table(
    row_count: int = 1914,
    *,
    ambiguous_bad_positive: bool = False,
    runner_bad_positive: bool = False,
) -> pd.DataFrame:
    rows = []
    idx = 0
    plan = [
        ("STRONG_BAD_BLOCK_TARGET", 120),
        ("TAIL_CONTROL_TARGET", 180),
        ("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD", 130),
        ("RUNNER_PROTECT_TARGET", 450),
        ("IGNORE_OR_MONITOR_ONLY", 1034),
    ]
    for bucket, count in plan:
        for _ in range(count):
            risky = bucket == "IGNORE_OR_MONITOR_ONLY" and idx < 930
            hard = bucket == "RUNNER_PROTECT_TARGET" and idx % 5 == 0
            rows.append(
                {
                    "candidate_uid": f"candidate_{idx:04d}",
                    "trade_uid": f"trade_{idx:04d}",
                    "decision_timestamp": f"2026-01-05T10:{idx % 60:02d}:00Z",
                    "new_r5_2_label_bucket_v1": bucket,
                    "label_should_not_take_v1": bucket in {"STRONG_BAD_BLOCK_TARGET", "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD", "RUNNER_PROTECT_TARGET"},
                    "tail_10_50_mfe_v1": bucket == "TAIL_CONTROL_TARGET",
                    "bad_eligibility_target_v1": bucket == "STRONG_BAD_BLOCK_TARGET"
                    or (ambiguous_bad_positive and bucket == "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD" and idx == 300)
                    or (runner_bad_positive and bucket == "RUNNER_PROTECT_TARGET" and idx == 500),
                    "tail_eligibility_target_v1": bucket == "TAIL_CONTROL_TARGET",
                    "r6_label_risky_allow_v1": risky,
                    "fifty_plus_mfe_v1": bucket in {"AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD", "RUNNER_PROTECT_TARGET"},
                    "hundred_plus_mfe_v1": hard,
                    "two_hundred_plus_mfe_v1": bucket == "RUNNER_PROTECT_TARGET" and idx % 17 == 0,
                    "strongest_winner_path_v1": hard,
                    "r6_label_repaired_165_like_runner_v1": False,
                    "r6_label_runner_near_miss_v1": bucket == "RUNNER_PROTECT_TARGET" and idx % 11 == 0,
                }
            )
            idx += 1
    return pd.DataFrame(rows).iloc[:row_count].copy()


def _score_frame(label: pd.DataFrame, *, forbidden_feature: bool = False) -> pd.DataFrame:
    rows = []
    for idx, row in label.iterrows():
        record = {
            "run_id": f"run_{idx // 30:03d}",
            "candidate_uid": row["candidate_uid"],
            "trade_uid": row["trade_uid"],
            "trade_id": f"trade_{idx:04d}",
            "decision_timestamp": row["decision_timestamp"],
            "calendar_quarantine_status_v1": "QUARANTINED" if idx >= 1852 else "ACTIVE_CANDIDATE",
            "as_of_feature_a_v1": float(idx % 7),
            "as_of_feature_b_v1": float(idx % 11),
            "pred__entry_r5_should_not_take__prob_true_v1": 0.6,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.4,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.2,
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.5,
            "r5_1_bad_blocker_score_v1": 0.6,
            "r5_1_runner_guard_score_v1": 0.2,
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.4,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.2,
        }
        if forbidden_feature:
            record["as_of_hindsight_leak_v1"] = 1.0
        rows.append(record)
    return pd.DataFrame(rows)


def _variants(*, mismatch: bool = False) -> list[dict]:
    variants = []
    for idx, profile_id in enumerate(runner.EXPECTED_VARIANTS, start=1):
        actual_profile = "BROKEN_PROFILE" if mismatch and idx == 1 else profile_id
        variants.append(
            {
                "variant_id_v1": f"R5_2_OBJECTIVE_V2_VARIANT_{idx:02d}_{actual_profile}",
                "weights_v1": {
                    "profile_id_v1": actual_profile,
                    "bad_weight_v1": 3.0 + idx / 10,
                    "tail_weight_v1": 2.5 + idx / 10,
                    "risky_weight_v1": 1.0,
                    "runner_protect_weight_v1": 16.0,
                    "ambiguous_high_mfe_protection_weight_v1": 24.0,
                    "hard_protect_weight_v1": 32.0,
                },
                "veto_strictness_v1": {
                    "bad_recall_threshold_v1": 0.7,
                    "tail_recall_threshold_v1": 0.68,
                    "risky_attention_threshold_v1": 0.7,
                    "bad_tail_confirmation_threshold_v1": 0.5,
                    "runner_veto_threshold_v1": 0.2,
                    "ambiguous_veto_threshold_v1": 0.14,
                    "hard_winner_veto_threshold_v1": 0.1,
                },
                "expected_outputs_v1": [
                    "r5_2_v2_prediction_view_v1.parquet",
                    "r5_2_v2_score_package_v1.parquet",
                    "r5_2_v2_base_membership_v1.parquet",
                    "r5_2_v2_eval_summary_v1.json",
                    "r5_2_v2_downstream_r6_input_manifest_v1.json",
                ],
                "base_membership_rule_v1": "pre_veto recall/risky rule AND NOT hard protection veto",
            }
        )
    return variants


def _write_spec_dir(path: Path, *, variant_mismatch: bool = False, missing_veto: bool = False) -> None:
    path.mkdir(parents=True)
    _write_json(path / "r5_2_objective_v2_design_lock_v1.json", {"design_id_v1": runner.DESIGN_ID})
    _write_json(
        path / "r5_2_objective_v2_label_contract_v1.json",
        {
            "buckets_v1": [{"bucket_v1": bucket} for bucket in runner.REQUIRED_V2_BUCKETS],
            "ambiguous_high_mfe_bad_positive_allowed_v1": False,
        },
    )
    _write_json(path / "r5_2_objective_v2_weight_and_cost_spec_v1.json", {"candidate_weight_profiles_v1": []})
    _write_json(
        path / "r5_2_objective_v2_model_architecture_spec_v1.json",
        {"final_outputs_v1": [*runner.RECALL_OUTPUTS, *runner.PROTECTION_OUTPUTS, *runner.BASE_OUTPUTS]},
    )
    reason_codes = [] if missing_veto else ["VETO_HARD_WINNER", "VETO_HIGH_MFE_AMBIGUOUS", "VETO_RUNNER_PROTECT", "VETO_REPAIRED_OR_STRONGEST"]
    _write_json(
        path / "r5_2_objective_v2_base_membership_contract_v1.json",
        {
            "contract_id_v1": "R5_2_V2_BASE_MEMBERSHIP_TWO_STAGE_RECALL_HARD_PROTECTION_VETO",
            "final_base_rule_v1": "r5_2_v2_base_membership_pre_veto AND NOT r5_2_v2_hard_protection_veto",
            "veto_rule_v1": {"VETO_HARD_WINNER": "forensic strongest 100 200 repaired"},
            "reason_codes_v1": reason_codes,
        },
    )
    _write_json(path / "r5_2_objective_v2_target_table_spec_v1.json", {"row_coverage_v1": {"required_rows_v1": 1914}})
    pd.DataFrame([{"feature_family_v1": "existing_109_as_of_features", "role_v1": "REUSE_NOW"}]).to_csv(
        path / "r5_2_objective_v2_existing_feature_use_spec_v1.csv", index=False
    )
    _write_json(path / "r5_2_objective_v2_parallel_rebuild_run_spec_v1.json", {"variants_v1": _variants(mismatch=variant_mismatch)})
    _write_json(path / "r5_2_objective_v2_eval_and_gate_spec_v1.json", {"hard_fail_if_v1": []})
    _write_json(path / "r5_2_objective_v2_next_runner_spec_lock_v1.json", {"decision_v1": "IMPLEMENT_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_RUNNER"})
    _write_json(path / "manifest_v1.json", {"input_artifacts_v1": {}})


def _fixture(
    tmp_path: Path,
    *,
    label_rows: int = 1914,
    ambiguous_bad_positive: bool = False,
    runner_bad_positive: bool = False,
    forbidden_feature: bool = False,
    variant_mismatch: bool = False,
    missing_veto: bool = False,
) -> tuple[Path, Path, Path]:
    spec_dir = tmp_path / "spec"
    score_dir = tmp_path / "score"
    label_path = tmp_path / "label.csv"
    _write_spec_dir(spec_dir, variant_mismatch=variant_mismatch, missing_veto=missing_veto)
    score_dir.mkdir()
    labels = _label_table(label_rows, ambiguous_bad_positive=ambiguous_bad_positive, runner_bad_positive=runner_bad_positive)
    labels.to_csv(label_path, index=False)
    score = _score_frame(_label_table(1914), forbidden_feature=forbidden_feature)
    score.to_parquet(score_dir / SCORE_FRAME, index=False)
    _write_json(
        score_dir / SCORE_SUMMARY,
        {
            "row_count_v1": 1914,
            "active_rows_v1": 1852,
            "quarantine_rows_v1": 62,
            "as_of_column_count_v1": 109,
        },
    )
    return spec_dir, score_dir, label_path


def test_v2_parallel_runner_dry_prelaunch_writes_scaffold(tmp_path: Path) -> None:
    spec_dir, score_dir, label_path = _fixture(tmp_path)
    out = tmp_path / "out"

    summary = runner.materialize(
        reports_root=tmp_path,
        spec_dir=spec_dir,
        output_dir=out,
        foundation_score_dir=score_dir,
        label_table=label_path,
    )

    assert summary["decision_v1"] == "DRY_PRELAUNCH_COMPLETED"
    assert summary["prelaunch_status_v1"] == "PASS"
    assert summary["training_started_v1"] is False
    assert summary["parallel_execution_started_v1"] is False
    assert summary["variant_count_v1"] == 7
    assert summary["next_action_v1"] == "NEXT_AGENT_MAY_RUN_R5_2_OBJECTIVE_V2_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG"
    assert summary["blocked_action_v1"] == "RUN_PARALLEL_REBUILD_WITHOUT_EXPLICIT_FLAG"
    for filename in runner.DRY_OUTPUT_FILES.values():
        assert (out / filename).exists()
    variants = pd.read_csv(out / "v2_variant_config_manifest_v1.csv")
    assert set(variants["status_v1"]) == {"READY_FOR_EXPLICIT_RUN"}
    assert (out / "v2_downstream_r6_manifest_placeholder_v1.json").exists()


def test_v2_parallel_runner_execution_requires_explicit_flag_and_writes_outputs(tmp_path: Path) -> None:
    spec_dir, score_dir, label_path = _fixture(tmp_path)
    summary = runner.materialize(
        reports_root=tmp_path,
        spec_dir=spec_dir,
        output_dir=tmp_path / "out",
        foundation_score_dir=score_dir,
        label_table=label_path,
        run_parallel_rebuild=True,
    )

    assert summary["training_started_v1"] is True
    assert summary["parallel_execution_started_v1"] is True
    assert summary["r6_started_v1"] is False
    for filename in runner.EXECUTION_OUTPUT_FILES.values():
        assert (tmp_path / "out" / filename).exists()
    index = pd.read_csv(tmp_path / "out" / "v2_variant_training_outputs_index_v1.csv")
    assert index.shape[0] == 7


def test_v2_parallel_runner_variant_mismatch_hard_fails(tmp_path: Path) -> None:
    spec_dir, score_dir, label_path = _fixture(tmp_path, variant_mismatch=True)
    with pytest.raises(RuntimeError, match="variant profile"):
        runner.materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out", foundation_score_dir=score_dir, label_table=label_path)


def test_v2_parallel_runner_target_row_mismatch_hard_fails(tmp_path: Path) -> None:
    spec_dir, score_dir, label_path = _fixture(tmp_path, label_rows=1913)
    with pytest.raises(RuntimeError, match="Expected target table rows"):
        runner.materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out", foundation_score_dir=score_dir, label_table=label_path)


def test_v2_parallel_runner_ambiguous_bad_positive_hard_fails(tmp_path: Path) -> None:
    spec_dir, score_dir, label_path = _fixture(tmp_path, ambiguous_bad_positive=True)
    with pytest.raises(RuntimeError, match="Ambiguous high-MFE"):
        runner.materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out", foundation_score_dir=score_dir, label_table=label_path)


def test_v2_parallel_runner_runner_protect_bad_positive_hard_fails(tmp_path: Path) -> None:
    spec_dir, score_dir, label_path = _fixture(tmp_path, runner_bad_positive=True)
    with pytest.raises(RuntimeError, match="Runner-protect"):
        runner.materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out", foundation_score_dir=score_dir, label_table=label_path)


def test_v2_parallel_runner_forbidden_feature_hard_fails(tmp_path: Path) -> None:
    spec_dir, score_dir, label_path = _fixture(tmp_path, forbidden_feature=True)
    with pytest.raises(RuntimeError, match="Forbidden features"):
        runner.materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out", foundation_score_dir=score_dir, label_table=label_path)


def test_v2_parallel_runner_missing_veto_contract_hard_fails(tmp_path: Path) -> None:
    spec_dir, score_dir, label_path = _fixture(tmp_path, missing_veto=True)
    with pytest.raises(RuntimeError, match="hard protection veto contract"):
        runner.materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out", foundation_score_dir=score_dir, label_table=label_path)
