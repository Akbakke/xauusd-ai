import json
from pathlib import Path

import pandas as pd
import pytest

from gx1.scripts.run_true_r5_2_rebuild_runner_v1 import DRY_OUTPUT_FILES, materialize
from gx1.scripts.train_monday_r6_foundation_score_rebuild_v1 import SCORE_FRAME, SUMMARY as SCORE_SUMMARY


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _label_table(row_count: int = 1914, *, bucket_mismatch: bool = False, ambiguous_bad_positive: bool = False) -> pd.DataFrame:
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
            actual_bucket = "IGNORE_OR_MONITOR_ONLY" if bucket_mismatch and idx == 0 else bucket
            rows.append(
                {
                    "candidate_uid": f"candidate_{idx:04d}",
                    "trade_uid": f"trade_{idx:04d}",
                    "decision_timestamp": f"2026-01-05T13:{idx % 60:02d}:00Z",
                    "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                    "label_should_not_take_v1": bad_label,
                    "tail_10_50_mfe_v1": tail_label,
                    "r5_2_v3_base_flag_v1": False,
                    "new_r5_2_label_bucket_v1": actual_bucket,
                    "bad_eligibility_target_v1": actual_bucket == "STRONG_BAD_BLOCK_TARGET"
                    or (ambiguous_bad_positive and actual_bucket == "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD" and idx == 243),
                    "tail_eligibility_target_v1": actual_bucket == "TAIL_CONTROL_TARGET",
                    "risky_attention_target_v1": False,
                    "runner_protect_target_v1": actual_bucket == "RUNNER_PROTECT_TARGET",
                    "ambiguous_high_mfe_monitor_v1": actual_bucket == "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD",
                    "sample_weight_v1": 3.0 if actual_bucket == "STRONG_BAD_BLOCK_TARGET" else 2.5 if actual_bucket == "TAIL_CONTROL_TARGET" else 0.25,
                    "protection_weight_v1": 10.0 if actual_bucket == "RUNNER_PROTECT_TARGET" else 6.0 if actual_bucket == "AMBIGUOUS_HIGH_MFE_DO_NOT_REWARD_AS_BAD" else 0.0,
                    "hundred_plus_mfe_v1": False,
                    "two_hundred_plus_mfe_v1": False,
                    "strongest_winner_path_v1": False,
                    "r6_label_repaired_165_like_runner_v1": False,
                }
            )
            idx += 1
    while idx < row_count:
        rows.append(
            {
                "candidate_uid": f"candidate_{idx:04d}",
                "trade_uid": f"trade_{idx:04d}",
                "decision_timestamp": f"2026-01-05T13:{idx % 60:02d}:00Z",
                "calendar_quarantine_status_v1": "QUARANTINED" if idx >= 1852 else "ACTIVE_CANDIDATE",
                "label_should_not_take_v1": False,
                "tail_10_50_mfe_v1": False,
                "r5_2_v3_base_flag_v1": True,
                "new_r5_2_label_bucket_v1": "IGNORE_OR_MONITOR_ONLY",
                "bad_eligibility_target_v1": False,
                "tail_eligibility_target_v1": False,
                "risky_attention_target_v1": False,
                "runner_protect_target_v1": False,
                "ambiguous_high_mfe_monitor_v1": False,
                "sample_weight_v1": 0.25,
                "protection_weight_v1": 0.0,
                "hundred_plus_mfe_v1": False,
                "two_hundred_plus_mfe_v1": False,
                "strongest_winner_path_v1": False,
                "r6_label_repaired_165_like_runner_v1": False,
            }
        )
        idx += 1
    return pd.DataFrame(rows).iloc[:row_count].copy()


def _score_frame(row_count: int = 1914, *, forbidden_feature: bool = False) -> pd.DataFrame:
    label = _label_table(row_count)
    rows = []
    for idx, row in label.iterrows():
        record = {
            "run_id": f"run_{idx // 30:03d}",
            "candidate_uid": row["candidate_uid"],
            "trade_uid": row["trade_uid"],
            "trade_id": f"trade_{idx:04d}",
            "decision_timestamp": row["decision_timestamp"],
            "used_for_training": idx < 900,
            "used_for_validation": 900 <= idx < 1200,
            "used_for_holdout": idx >= 1200,
            "calendar_quarantine_status_v1": row["calendar_quarantine_status_v1"],
            "as_of_feature_a_v1": float(idx % 7),
            "as_of_feature_b_v1": float(idx % 11),
            "pred__entry_r5_should_not_take__prob_true_v1": 0.7,
            "pred__entry_r5_immediate_MAE_risk__prob_true_v1": 0.6,
            "pred__entry_r5_runner_protect__prob_true_v1": 0.2,
            "pred__entry_r5_tail_control_10_50_risk__prob_true_v1": 0.5,
            "r5_1_bad_blocker_score_v1": 0.7,
            "r5_1_runner_guard_score_v1": 0.2,
            "pred__entry_r5_2_bad_blocker__prob_true_v1": 0.4,
            "pred__entry_r5_2_runner_protector__prob_true_v1": 0.2,
        }
        if forbidden_feature:
            record["as_of_hindsight_leak_v1"] = 1.0
        rows.append(record)
    return pd.DataFrame(rows)


def _write_spec_dir(path: Path, score_dir: Path, label_path: Path) -> None:
    path.mkdir()
    input_contract = {
        "foundation_v1": {
            "score_package_dir_v1": str(score_dir),
            "row_count_v1": 1914,
            "active_rows_v1": 1852,
            "quarantine_rows_v1": 62,
            "as_of_column_count_v1": 109,
        },
        "label_table_v1": {"path_v1": str(label_path), "row_count_v1": 1914},
        "required_score_input_families_v1": {
            "r5_signals_v1": [
                "pred__entry_r5_should_not_take__prob_true_v1",
                "pred__entry_r5_immediate_MAE_risk__prob_true_v1",
                "pred__entry_r5_runner_protect__prob_true_v1",
                "pred__entry_r5_tail_control_10_50_risk__prob_true_v1",
            ],
            "r5_1_signals_v1": ["r5_1_bad_blocker_score_v1", "r5_1_runner_guard_score_v1"],
            "allowed_current_r5_2_inputs_v1": [
                "pred__entry_r5_2_bad_blocker__prob_true_v1",
                "pred__entry_r5_2_runner_protector__prob_true_v1",
            ],
        },
    }
    loader_spec = {
        "locked_weights_v1": {
            "bad_target_weight_v1": 3.0,
            "tail_target_weight_v1": 2.5,
            "runner_protect_weight_v1": 10.0,
            "hard_protection_weight_v1": 20.0,
        }
    }
    model_config = {"seed_v1": 20260426}
    r6_contract = {
        "r5_2_score_columns_for_r6_v1": ["pred__entry_r5_2_rebuild_bad_eligibility__prob_true_v1"],
        "base_membership_flags_for_r6_v1": ["r5_2_rebuilt_base_membership_v1"],
    }
    files = {
        "true_r5_2_rebuild_runner_spec_v1.json": {"job_name_v1": "fixture"},
        "r5_2_rebuild_input_contract_v1.json": input_contract,
        "r5_2_rebuild_label_and_weight_loader_spec_v1.json": loader_spec,
        "r5_2_rebuild_model_config_lock_v1.json": model_config,
        "r5_2_rebuild_eval_and_safety_guards_v1.json": {},
        "r5_2_rebuild_output_spec_v1.json": {},
        "r5_2_rebuild_prelaunch_checklist_v1.json": {},
        "r5_2_rebuild_abort_rules_v1.json": {},
        "downstream_r6_consumption_contract_v1.json": r6_contract,
    }
    for filename, payload in files.items():
        _write_json(path / filename, payload)


def _fixture(tmp_path: Path, *, score_rows: int = 1914, label_rows: int = 1914, bucket_mismatch: bool = False, ambiguous_bad_positive: bool = False, forbidden_feature: bool = False) -> tuple[Path, Path, Path]:
    score_dir = tmp_path / "score"
    label_dir = tmp_path / "label"
    spec_dir = tmp_path / "spec"
    score_dir.mkdir()
    label_dir.mkdir()
    label = _label_table(label_rows, bucket_mismatch=bucket_mismatch, ambiguous_bad_positive=ambiguous_bad_positive)
    label_path = label_dir / "r5_2_pocket_label_table_v1.csv"
    label.to_csv(label_path, index=False)
    _score_frame(score_rows, forbidden_feature=forbidden_feature).to_parquet(score_dir / SCORE_FRAME, index=False)
    _write_json(
        score_dir / SCORE_SUMMARY,
        {"row_count_v1": score_rows, "active_rows_v1": 1852, "quarantine_rows_v1": 62, "as_of_column_count_v1": 109},
    )
    _write_spec_dir(spec_dir, score_dir, label_path)
    return spec_dir, score_dir, label_path


def test_true_r5_2_rebuild_runner_dry_prelaunch_writes_scaffold(tmp_path: Path) -> None:
    spec_dir, _, _ = _fixture(tmp_path)
    out = tmp_path / "out"

    summary = materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=out)

    assert summary["decision_v1"] == "DRY_PRELAUNCH_COMPLETED"
    assert summary["prelaunch_status_v1"] == "PASS"
    assert summary["training_started_v1"] is False
    assert summary["next_action_v1"] == "NEXT_AGENT_MAY_RUN_TRUE_R5_2_REBUILD_WITH_EXPLICIT_FLAG"
    assert summary["blocked_action_v1"] == "RUN_TRAINING_WITHOUT_EXPLICIT_FLAG"
    assert summary["forbidden_feature_count_v1"] == 0
    for filename in DRY_OUTPUT_FILES.values():
        assert (out / filename).exists()
    assert (out / "r5_2_downstream_r6_input_manifest_placeholder_v1.json").exists()


def test_true_r5_2_rebuild_runner_row_count_hard_fails(tmp_path: Path) -> None:
    spec_dir, _, _ = _fixture(tmp_path, score_rows=1913)
    with pytest.raises(RuntimeError, match="Expected foundation rows"):
        materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out")


def test_true_r5_2_rebuild_runner_label_count_hard_fails(tmp_path: Path) -> None:
    spec_dir, _, _ = _fixture(tmp_path, label_rows=1913)
    with pytest.raises(RuntimeError, match="Expected label table rows"):
        materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out")


def test_true_r5_2_rebuild_runner_bucket_mismatch_hard_fails(tmp_path: Path) -> None:
    spec_dir, _, _ = _fixture(tmp_path, bucket_mismatch=True)
    with pytest.raises(RuntimeError, match="Expected missed bucket counts"):
        materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out")


def test_true_r5_2_rebuild_runner_ambiguous_bad_positive_hard_fails(tmp_path: Path) -> None:
    spec_dir, _, _ = _fixture(tmp_path, ambiguous_bad_positive=True)
    with pytest.raises(RuntimeError, match="Ambiguous high-MFE"):
        materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out")


def test_true_r5_2_rebuild_runner_forbidden_feature_hard_fails(tmp_path: Path) -> None:
    spec_dir, _, _ = _fixture(tmp_path, forbidden_feature=True)
    with pytest.raises(RuntimeError, match="Forbidden features"):
        materialize(reports_root=tmp_path, spec_dir=spec_dir, output_dir=tmp_path / "out")
