import json
from pathlib import Path

import pandas as pd
import pytest

import gx1.scripts.run_r5_2_objective_v3_parallel_rebuild_runner_v1 as runner


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
    plan = [
        ("STRONG_BAD_BLOCK_TARGET", 127),
        ("TAIL_CONTROL_TARGET", 198),
        ("AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD", 130),
        ("RUNNER_PROTECT_TARGET", 462),
        ("IGNORE_OR_MONITOR_ONLY", 997),
    ]
    idx = 0
    for bucket, count in plan:
        for _ in range(count):
            is_amb = bucket == "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD"
            is_runner = bucket == "RUNNER_PROTECT_TARGET"
            is_bad = bucket == "STRONG_BAD_BLOCK_TARGET"
            is_tail = bucket == "TAIL_CONTROL_TARGET"
            is_monitor = bucket == "IGNORE_OR_MONITOR_ONLY"
            hard = idx % 19 == 0
            rows.append(
                {
                    "candidate_uid": runner.FORENSIC_REPAIRED_CANDIDATE_UID if idx == 0 else f"candidate_{idx:04d}",
                    "trade_uid": f"trade_{idx:04d}",
                    "trade_id": f"T-{idx:04d}",
                    "decision_timestamp": f"2026-01-05T10:{idx % 60:02d}:00Z",
                    "run_id": f"run_{idx // 50:03d}",
                    "split_scope_v1": "TRAIN",
                    "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE" if idx < 1852 else "QUARANTINED",
                    "label_should_not_take_v1": is_bad or is_amb or is_runner,
                    "tail_10_50_mfe_v1": is_tail,
                    "new_r5_2_label_bucket_v1": bucket,
                    "bad_eligibility_target_v1": is_bad or (ambiguous_bad_positive and is_amb and idx == 350) or (runner_bad_positive and is_runner and idx == 500),
                    "tail_eligibility_target_v1": is_tail,
                    "risky_attention_target_v1": is_monitor and idx % 41 == 0,
                    "runner_protect_target_v1": is_runner,
                    "ambiguous_high_mfe_monitor_v1": is_amb,
                    "hundred_plus_mfe_v1": hard,
                    "two_hundred_plus_mfe_v1": hard and idx % 3 == 0,
                    "strongest_winner_path_v1": hard,
                    "r6_label_repaired_165_like_runner_v1": False,
                    "eval_only_flag_v1": hard,
                }
            )
            idx += 1
    return pd.DataFrame(rows).iloc[:row_count].copy()


def _score_frame(
    label: pd.DataFrame,
    *,
    forbidden_feature: bool = False,
    id_leak_feature: bool = False,
    synthetic_feature: bool = False,
) -> pd.DataFrame:
    rows = []
    for idx, row in label.iterrows():
        record = {
            "run_id": row["run_id"],
            "candidate_uid": row["candidate_uid"],
            "trade_uid": row["trade_uid"],
            "trade_id": row["trade_id"],
            "decision_timestamp": row["decision_timestamp"],
            "calendar_quarantine_status_v1": row["calendar_quarantine_status_v1"],
            "as_of_feature_a_v1": float(idx % 7),
            "as_of_feature_b_v1": float(idx % 11),
            "pred__entry_r5_should_not_take__prob_true_v1": 0.4,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.2,
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.5,
            "r5_1_bad_blocker_score_v1": 0.4,
            "r5_1_runner_guard_score_v1": 0.2,
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.4,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.2,
            "fifty_plus_mfe_v1": bool(idx % 13 == 0),
            "hundred_plus_mfe_v1": bool(idx % 37 == 0),
            "two_hundred_plus_mfe_v1": bool(idx % 101 == 0),
            "strongest_winner_path_v1": bool(idx % 29 == 0),
            "r6_label_repaired_165_like_runner_v1": False,
            "r6_label_runner_near_miss_v1": bool(idx % 43 == 0),
            "r5_2_label_high_mfe_tail_risk_ambiguous_v1": bool(idx % 17 == 0),
            "r5_2_label_runner_protect_v1": bool(idx % 31 == 0),
        }
        if forbidden_feature:
            record["as_of_hindsight_leak_v1"] = 1.0
        if id_leak_feature:
            record["as_of_candidate_id_v1"] = idx
        if synthetic_feature:
            record["as_of_dummy_score_v1"] = 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def _write_scan_dir(path: Path, v2_execution_dir: Path) -> None:
    path.mkdir(parents=True)
    _write_json(
        path / "summary_v1.json",
        {
            "decision_v1": "IMPLEMENT_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_RUNNER",
            "training_started_v1": False,
            "r6_started_v1": False,
        },
    )
    _write_json(path / "v3_or_r6_head_next_decision_v1.json", {"decision_v1": "IMPLEMENT_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_RUNNER"})
    _write_json(path / "next_action_lock_v1.json", {"next_action_v1": "IMPLEMENT_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_RUNNER"})
    _write_json(path / "manifest_v1.json", {"input_v2_execution_dir_v1": str(v2_execution_dir)})
    lane_rows = []
    for bucket, count in runner.EXPECTED_V3_BUCKET_COUNTS.items():
        for idx in range(count):
            lane_rows.append({"candidate_uid": f"{bucket}_{idx}", "gap_bucket_v1": bucket})
    pd.DataFrame(lane_rows).to_csv(path / "lane_01_v2_remaining_gap_trace_v1.csv", index=False)
    pd.DataFrame({"profile_id_v1": runner.V3_VARIANTS}).to_csv(path / "lane_09_v3_weight_profile_sim_scan_v1.csv", index=False)
    pd.DataFrame({"profile_id_v1": runner.V3_VARIANTS, "safety_pass_readonly_v1": True}).to_csv(
        path / "lane_10_high_mfe_winner_stress_scan_v1.csv", index=False
    )
    pd.DataFrame({"profile_id_v1": runner.V3_VARIANTS, "leaderboard_score_v1": range(len(runner.V3_VARIANTS))}).to_csv(
        path / "v3_design_leaderboard_v1.csv", index=False
    )


def _fixture(
    tmp_path: Path,
    *,
    label_rows: int = 1914,
    ambiguous_bad_positive: bool = False,
    runner_bad_positive: bool = False,
    forbidden_feature: bool = False,
    id_leak_feature: bool = False,
    synthetic_feature: bool = False,
    degraded_score_path: str | None = None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    label = _label_table(label_rows, ambiguous_bad_positive=ambiguous_bad_positive, runner_bad_positive=runner_bad_positive)
    score = _score_frame(label if len(label) == 1914 else _label_table(), forbidden_feature=forbidden_feature, id_leak_feature=id_leak_feature, synthetic_feature=synthetic_feature)
    v2_dir = tmp_path / "v2_execution"
    v2_dir.mkdir()
    label_path = tmp_path / "label.csv"
    score_path = tmp_path / (degraded_score_path or "score_package.parquet")
    summary_path = tmp_path / "foundation_summary.json"
    feature_inventory = tmp_path / "feature_inventory.csv"
    scan_dir = tmp_path / "scan"
    label.to_csv(label_path, index=False)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score.to_parquet(score_path, index=False)
    _write_json(summary_path, {"row_count_v1": 1914, "active_rows_v1": 1852, "quarantine_rows_v1": 62, "as_of_column_count_v1": 109})
    pd.DataFrame([{"feature_family_v1": "existing_109_as_of_features", "status_v1": "REUSE_NOW"}]).to_csv(feature_inventory, index=False)
    _write_json(
        v2_dir / "manifest_v1.json",
        {
            "input_artifacts_v1": {
                "foundation_summary_v1": str(summary_path),
                "label_table_v1": str(label_path),
                "spec_dir_v1": str(tmp_path),
            }
        },
    )
    _write_json(
        v2_dir / "best_v2_variant_downstream_r6_input_lock_v1.json",
        {
            "score_package_path_v1": str(score_path),
            "base_flag_for_r6_v1": "r5_2_v2_final_base_membership",
        },
    )
    _write_scan_dir(scan_dir, v2_dir)
    return scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory


def test_v3_parallel_runner_dry_prelaunch_writes_only_contract_artifacts(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path)
    out = tmp_path / "out"

    summary = runner.materialize(
        reports_root=tmp_path,
        scan_dir=scan_dir,
        output_dir=out,
        v2_execution_dir=v2_dir,
        score_package=score_path,
        foundation_summary=summary_path,
        label_table=label_path,
        feature_inventory=feature_inventory,
    )

    assert summary["decision_v1"] == "DRY_PRELAUNCH_COMPLETED"
    assert summary["training_started_v1"] is False
    assert summary["parallel_execution_started_v1"] is False
    assert summary["variant_count_v1"] == 10
    assert summary["foundation_rows_v1"] == 1914
    assert summary["target_table_rows_v1"] == 1914
    assert summary["forbidden_feature_count_v1"] == 0
    assert summary["id_leakage_feature_count_v1"] == 0
    assert summary["synthetic_or_dummy_input_count_v1"] == 0
    assert summary["hard_protection_veto_contract_present_v1"] is True
    assert summary["next_action_v1"] == "NEXT_AGENT_MAY_RUN_R5_2_OBJECTIVE_V3_PARALLEL_REBUILD_WITH_EXPLICIT_FLAG"
    for filename in runner.DRY_OUTPUT_FILES:
        assert (out / filename).exists()
    assert not list(out.glob("*.parquet"))
    assert not list(out.glob("*.joblib"))


def test_v3_parallel_runner_execution_requires_explicit_flag_and_writes_real_outputs(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path)
    summary = runner.materialize(
        reports_root=tmp_path,
        scan_dir=scan_dir,
        output_dir=tmp_path / "out",
        v2_execution_dir=v2_dir,
        score_package=score_path,
        foundation_summary=summary_path,
        label_table=label_path,
        feature_inventory=feature_inventory,
        run_parallel_rebuild=True,
    )

    assert summary["training_started_v1"] is True
    assert summary["parallel_execution_started_v1"] is True
    assert summary["r6_started_v1"] is False
    assert (tmp_path / "out" / "v3_variant_eval_and_safety_gate_v1.csv").exists()
    assert (tmp_path / "out" / "v3_generalization_and_overfit_eval_v1.csv").exists()
    assert (tmp_path / "out" / "v3_variant_leaderboard_v1.csv").exists()
    assert (tmp_path / "out" / "strategy_gate_after_v3_v1.json").exists()
    assert (tmp_path / "out" / "next_strategy_options_if_v3_too_weak_v1.json").exists()
    index = pd.read_csv(tmp_path / "out" / "v3_variant_outputs_index_v1.csv")
    assert index.shape[0] == 10
    assert Path(index.iloc[0]["score_package_path_v1"]).exists()
    first_variant = Path(index.iloc[0]["variant_dir_v1"])
    assert (first_variant / "model_manifest_v1.json").exists()
    assert (first_variant / "config_manifest_v1.json").exists()
    assert (first_variant / "label_weight_manifest_v1.csv").exists()
    assert (first_variant / "pocket_eval_report_v1.csv").exists()
    assert (first_variant / "safety_guard_report_v1.json").exists()
    assert (first_variant / "v3_oof_score_provenance_v1.csv").exists()
    assert (first_variant / "v3_oof_fold_assignment_v1.csv").exists()
    assert (first_variant / "v3_oof_score_source_manifest_v1.json").exists()
    assert (first_variant / "v3_train_validation_membership_v1.csv").exists()
    assert (first_variant / "status_v1.json").exists()
    assert (first_variant / "manifest_v1.json").exists()
    assert (first_variant / "consistency_audit_v1.csv").exists()
    assert (tmp_path / "out" / "active_score_artifact_selection_v1.json").exists()
    provenance = pd.read_csv(first_variant / "v3_oof_score_provenance_v1.csv")
    assert set(runner.V3_SCORE_FIELDS).issubset(set(provenance["score_field_v1"]))
    assert provenance["score_source_v1"].eq("OOF").all()
    assert not provenance["row_was_in_training_for_source_model_v1"].astype(bool).any()
    for column in [
        "model_source_identifier_v1",
        "feature_matrix_hash_v1",
        "label_table_hash_v1",
        "config_hash_v1",
        "seed_v1",
        "decision_valid_v1",
        "oof_provenance_status_v1",
    ]:
        assert column in provenance.columns
    assert provenance["decision_valid_v1"].astype(bool).all()


def test_v3_parallel_runner_explicit_oof_rerun_action_names_lock_root(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path)
    summary = runner.materialize(
        reports_root=tmp_path,
        scan_dir=scan_dir,
        v2_execution_dir=v2_dir,
        score_package=score_path,
        foundation_summary=summary_path,
        label_table=label_path,
        feature_inventory=feature_inventory,
        explicit_action=runner.EXPLICIT_OOF_RERUN_ACTION,
        write_oof_provenance=True,
        reject_in_sample_decision_scores=True,
        fail_on_missing_provenance=True,
        fail_on_degraded_fallback=True,
        fail_on_dummy_or_synthetic_input=True,
    )

    output_dir = Path(summary["output_dir_v1"])
    assert runner.EXPLICIT_OOF_RERUN_ACTION in output_dir.name
    assert output_dir.name.endswith("_LOCK")
    assert (output_dir / "v3_oof_score_provenance_v1.csv").exists()
    assert (output_dir / "v3_oof_fold_assignment_v1.csv").exists()
    assert (output_dir / "v3_oof_score_source_manifest_v1.json").exists()
    assert (output_dir / "v3_train_validation_membership_v1.csv").exists()
    assert (output_dir / "active_score_artifact_selection_v1.json").exists()


def test_v3_parallel_runner_missing_input_writes_blocked_status(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(runner.BlockedMissingRequiredInput):
        runner.materialize(reports_root=tmp_path, scan_dir=tmp_path / "missing_scan", output_dir=out)
    status = json.loads((out / "status_v1.json").read_text())
    assert status["status_v1"] == "BLOCKED_MISSING_REQUIRED_INPUT"


def test_v3_parallel_runner_degraded_1689_surface_hard_fails(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(
        tmp_path, degraded_score_path="bad_1689_exact_only_score.parquet"
    )
    with pytest.raises(RuntimeError, match="DEGRADED_FALLBACK_FORBIDDEN"):
        runner.materialize(
            reports_root=tmp_path,
            scan_dir=scan_dir,
            output_dir=tmp_path / "out",
            v2_execution_dir=v2_dir,
            score_package=score_path,
            foundation_summary=summary_path,
            label_table=label_path,
            feature_inventory=feature_inventory,
        )


@pytest.mark.parametrize("bad_path", ["diagnostic_score.parquet", "narrow_score.parquet", "protector_first_score.parquet"])
def test_v3_parallel_runner_diagnostic_narrow_protector_paths_hard_fail(tmp_path: Path, bad_path: str) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path, degraded_score_path=bad_path)
    with pytest.raises(RuntimeError, match="DEGRADED_FALLBACK_FORBIDDEN"):
        runner.materialize(
            reports_root=tmp_path,
            scan_dir=scan_dir,
            output_dir=tmp_path / "out",
            v2_execution_dir=v2_dir,
            score_package=score_path,
            foundation_summary=summary_path,
            label_table=label_path,
            feature_inventory=feature_inventory,
        )


def test_v3_parallel_runner_target_row_mismatch_hard_fails(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path, label_rows=1913)
    with pytest.raises(RuntimeError, match="Expected target table rows"):
        runner.materialize(
            reports_root=tmp_path,
            scan_dir=scan_dir,
            output_dir=tmp_path / "out",
            v2_execution_dir=v2_dir,
            score_package=score_path,
            foundation_summary=summary_path,
            label_table=label_path,
            feature_inventory=feature_inventory,
        )


def test_v3_parallel_runner_ambiguous_high_mfe_bad_positive_hard_fails(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path, ambiguous_bad_positive=True)
    with pytest.raises(RuntimeError, match="Ambiguous high-MFE"):
        runner.materialize(
            reports_root=tmp_path,
            scan_dir=scan_dir,
            output_dir=tmp_path / "out",
            v2_execution_dir=v2_dir,
            score_package=score_path,
            foundation_summary=summary_path,
            label_table=label_path,
            feature_inventory=feature_inventory,
        )


def test_v3_parallel_runner_runner_protect_bad_positive_hard_fails(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path, runner_bad_positive=True)
    with pytest.raises(RuntimeError, match="Runner-protect"):
        runner.materialize(
            reports_root=tmp_path,
            scan_dir=scan_dir,
            output_dir=tmp_path / "out",
            v2_execution_dir=v2_dir,
            score_package=score_path,
            foundation_summary=summary_path,
            label_table=label_path,
            feature_inventory=feature_inventory,
        )


def test_v3_parallel_runner_forbidden_feature_hard_fails(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path, forbidden_feature=True)
    with pytest.raises(RuntimeError, match="Forbidden feature"):
        runner.materialize(
            reports_root=tmp_path,
            scan_dir=scan_dir,
            output_dir=tmp_path / "out",
            v2_execution_dir=v2_dir,
            score_package=score_path,
            foundation_summary=summary_path,
            label_table=label_path,
            feature_inventory=feature_inventory,
        )


def test_v3_parallel_runner_id_leakage_hard_fails(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path, id_leak_feature=True)
    with pytest.raises(RuntimeError, match="ID leakage"):
        runner.materialize(
            reports_root=tmp_path,
            scan_dir=scan_dir,
            output_dir=tmp_path / "out",
            v2_execution_dir=v2_dir,
            score_package=score_path,
            foundation_summary=summary_path,
            label_table=label_path,
            feature_inventory=feature_inventory,
        )


def test_v3_parallel_runner_synthetic_input_hard_fails(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path, synthetic_feature=True)
    with pytest.raises(RuntimeError, match="SYNTHETIC_OR_DUMMY_INPUT_FORBIDDEN"):
        runner.materialize(
            reports_root=tmp_path,
            scan_dir=scan_dir,
            output_dir=tmp_path / "out",
            v2_execution_dir=v2_dir,
            score_package=score_path,
            foundation_summary=summary_path,
            label_table=label_path,
            feature_inventory=feature_inventory,
        )


def test_v3_parallel_runner_hard_veto_missing_hard_fails(tmp_path: Path) -> None:
    scan_dir, v2_dir, score_path, summary_path, label_path, feature_inventory = _fixture(tmp_path)
    score = pd.read_parquet(score_path).drop(columns=["fifty_plus_mfe_v1"])
    score.to_parquet(score_path, index=False)
    with pytest.raises(RuntimeError, match="Hard veto contract"):
        runner.materialize(
            reports_root=tmp_path,
            scan_dir=scan_dir,
            output_dir=tmp_path / "out",
            v2_execution_dir=v2_dir,
            score_package=score_path,
            foundation_summary=summary_path,
            label_table=label_path,
            feature_inventory=feature_inventory,
        )
