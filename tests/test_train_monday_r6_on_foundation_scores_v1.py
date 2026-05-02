import json
from pathlib import Path

import pandas as pd
import pytest

import gx1.scripts.train_monday_r6_on_foundation_scores_v1 as r6_rebuild


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _score_fixture(path: Path, row_count: int = 4) -> None:
    rows = []
    for idx in range(row_count):
        should = idx % 2 == 1
        active = idx != row_count - 1
        rows.append(
            {
                "run_id": f"run_{idx}",
                "candidate_uid": f"c{idx}",
                "trade_uid": f"t{idx}",
                "trade_id": f"trade{idx}",
                "decision_timestamp": f"2026-01-0{idx + 1}T12:00:00Z",
                "used_for_training": idx < 2,
                "used_for_validation": idx == 2,
                "used_for_holdout": idx >= 3,
                "split_scope_v1": "TRAIN" if idx < 2 else ("VALIDATION" if idx == 2 else "QUARANTINE_EVAL_ONLY"),
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE" if active else "QUARANTINED",
                "as_of_candidate_tradable_prob_v1": 0.9,
                "as_of_entry_candidate_path_quality_pred_v1": 0.8,
                "as_of_candidate_mfe_first_n_pred_v1": 2.0,
                "as_of_skip_candidate_p_flat_v1": 0.2,
                "label_should_not_take_v1": should,
                "take_was_ok_v1": not should,
                "label_strong_trade_candidate_v1": not should,
                "fifty_plus_mfe_v1": not should,
                "hundred_plus_mfe_v1": idx == 0,
                "two_hundred_plus_mfe_v1": False,
                "tail_10_50_mfe_v1": should,
                "strongest_winner_path_v1": False,
                "baseline_realized_pnl_bps_v1": -10.0 if should else 20.0,
                "peak_mfe_bps_v1": 20.0 if should else 80.0,
                "mae_abs_bps_v1": 55.0 if should else 10.0,
                "giveback_bps_v1": 5.0,
            }
        )
    frame = pd.DataFrame(rows)
    for column in r6_rebuild.FOUNDATION_SCORE_CONTEXT_COLUMNS:
        frame[column] = False if column.endswith("__block_v1") or column.endswith("selected_candidate__block_v1") else 0.2
    frame[r6_rebuild.R5_2_BAD_PROB] = [0.8, 0.9, 0.85, 0.1][:row_count]
    frame[r6_rebuild.R5_2_RUNNER_PROB] = [0.1, 0.1, 0.1, 0.9][:row_count]
    frame["blocker_score_v1"] = frame[r6_rebuild.R5_2_BAD_PROB]
    frame["runner_protector_score_v1"] = frame[r6_rebuild.R5_2_RUNNER_PROB]
    frame["r5_2_selected_candidate__block_v1"] = [True, True, True, False][:row_count]
    for column in r6_rebuild.R6_LABEL_COLUMNS:
        frame[column] = False
    frame["r6_label_bad_risk_v1"] = frame["label_should_not_take_v1"]
    frame["r6_label_tail_control_10_50_v1"] = frame["tail_10_50_mfe_v1"]
    frame["r6_label_runner_protect_v1"] = frame["take_was_ok_v1"]
    frame["r6_label_risky_allow_v1"] = frame["label_should_not_take_v1"]
    frame["r6_label_batch04_blindspot_v1"] = False
    frame.to_parquet(path / r6_rebuild.SCORE_FRAME, index=False)
    _write_json(
        path / r6_rebuild.SCORE_SUMMARY,
        {
            "decision_v1": "MONDAY_R5_R5_1_R5_2_SCORE_REBUILD_COMPLETED",
            "row_count_v1": row_count,
            "active_rows_v1": row_count - 1,
            "quarantine_rows_v1": 1,
            "as_of_column_count_v1": 109,
            "base_feature_count_v1": 4,
            "r5_head_count_v1": 8,
            "r5_1_policy_materialized_v1": True,
            "r5_2_head_count_v1": 2,
            "r6_heads_trained_v1": False,
        },
    )


def test_r6_rebuild_requires_explicit_flag(tmp_path: Path) -> None:
    score_dir = tmp_path / "score"
    score_dir.mkdir()
    out = tmp_path / "out"

    summary = r6_rebuild.materialize(reports_root=tmp_path, score_dir=score_dir, output_dir=out, run_r6_rebuild=False)

    assert summary["decision_v1"] == "EXPLICIT_R6_REBUILD_FLAG_REQUIRED"
    assert summary["training_started_v1"] is False
    assert summary["r6_training_started_v1"] is False
    assert (out / r6_rebuild.STATUS).exists()


def test_r6_rebuild_hard_fails_wrong_row_count(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(r6_rebuild, "EXPECTED_SCORE_ROWS", 4)
    monkeypatch.setattr(r6_rebuild, "EXPECTED_ACTIVE_ROWS", 3)
    monkeypatch.setattr(r6_rebuild, "EXPECTED_QUARANTINE_ROWS", 1)
    monkeypatch.setattr(r6_rebuild, "EXPECTED_BASE_FEATURES", 4)
    score_dir = tmp_path / "score"
    score_dir.mkdir()
    _score_fixture(score_dir, row_count=3)

    with pytest.raises(RuntimeError, match="Expected Monday R6 foundation score rows"):
        r6_rebuild.materialize(reports_root=tmp_path, score_dir=score_dir, output_dir=tmp_path / "out", run_r6_rebuild=True)


def test_r6_rebuild_writes_offline_outputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(r6_rebuild, "EXPECTED_SCORE_ROWS", 4)
    monkeypatch.setattr(r6_rebuild, "EXPECTED_ACTIVE_ROWS", 3)
    monkeypatch.setattr(r6_rebuild, "EXPECTED_QUARANTINE_ROWS", 1)
    monkeypatch.setattr(r6_rebuild, "EXPECTED_BASE_FEATURES", 4)

    def fake_train_r6_heads(**kwargs):
        frame = kwargs["frame"]
        pred = frame[["candidate_uid"]].copy()
        for column in r6_rebuild.R6_OUTPUT_COLUMNS:
            pred[column] = [0.9, 0.95, 0.85, 0.05]
        metrics = pd.DataFrame(
            [
                {
                    "model_tag_v1": kwargs["model_tag"],
                    "head_id_v1": "fixture",
                    "split_v1": "ALL",
                    "label_col_v1": "fixture",
                    "output_col_v1": r6_rebuild.R6_BAD_PROB,
                    "balanced_accuracy_v1": 1.0,
                }
            ]
        )
        return pred, metrics

    def fake_calibrate(frame):
        selected = pd.Series([False, True, True, False], index=frame.index)
        calibration = pd.DataFrame(
            [
                {
                    "all_wednesday_safety_pass_v1": True,
                    "all_wednesday_basic_safety_pass_v1": True,
                    "all_block_count_v1": 2,
                    "all_bad_blocks_v1": 2,
                    "all_tail_help_v1": 2,
                    "all_precision_v1": 1.0,
                    "all_worst_loso_v1": 1.0,
                    "trainval_safe_v1": True,
                }
            ]
        )
        policy = {
            "policy_name_v1": "FIXTURE_POLICY",
            "params_v1": {
                "asof_guard_tradable_min_v1": 0.94,
                "asof_guard_quality_min_v1": 0.70,
                "asof_guard_mfe_min_v1": 1.75,
                "asof_guard_flat_max_v1": 0.50,
            },
            "wednesday_safety_pass_v1": True,
        }
        return calibration, policy, selected

    monkeypatch.setattr(r6_rebuild, "_train_r6_heads", fake_train_r6_heads)
    monkeypatch.setattr(r6_rebuild, "_calibrate_policy", fake_calibrate)
    monkeypatch.setattr(
        r6_rebuild,
        "_calibration_safety_summary",
        lambda calibration, selected, compare: {
            "grid_candidate_count_v1": 1,
            "wednesday_safety_candidate_count_v1": 1,
            "wednesday_safety_and_better_candidate_count_v1": 0,
        },
    )
    monkeypatch.setattr(
        r6_rebuild,
        "_wednesday_locked_policy_replay",
        lambda frame: {
            "r5_2_base_block_count_v1": 3,
            "r6_addon_block_count_v1": 0,
            "wednesday_safety_pass_v1": True,
            "compare_v1": {"safety_failures_v1": []},
        },
    )
    monkeypatch.setattr(
        r6_rebuild,
        "_r6_family_grid_replay",
        lambda frame: (pd.DataFrame([{"policy_name_v1": "FIXTURE"}]), {"candidate_count_v1": 1, "wednesday_safety_candidate_count_v1": 1}),
    )
    score_dir = tmp_path / "score"
    score_dir.mkdir()
    _score_fixture(score_dir)
    out = tmp_path / "out"

    summary = r6_rebuild.materialize(reports_root=tmp_path, score_dir=score_dir, output_dir=out, run_r6_rebuild=True)

    assert summary["training_started_v1"] is True
    assert summary["r6_training_started_v1"] is True
    assert summary["r6_head_count_v1"] == 5
    assert summary["not_freeze_or_promo_v1"] is True
    assert "DO_NOT_USE_1689_EXACT_ONLY_AS_R6_BASELINE" in summary["blocked_action_v1"]
    for filename in r6_rebuild.OUTPUT_FILES.values():
        if filename == "models":
            continue
        assert (out / filename).exists()
    audit = pd.read_csv(out / r6_rebuild.AUDIT)
    assert audit.set_index("check_v1").loc["R6_HEADS_TRAINED", "status_v1"] == "PASS"
