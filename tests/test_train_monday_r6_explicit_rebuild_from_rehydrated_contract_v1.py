import json
from pathlib import Path

import pandas as pd

from gx1.scripts.train_monday_r6_explicit_rebuild_from_rehydrated_contract_v1 import (
    AS_OF_TABLE,
    HINDSIGHT_TABLE,
    OUTPUT_FILES,
    TRAINING_FRAME,
    TRUTH_TABLE,
    _assign_splits,
    _asof_runner_guard,
    _compare,
    _overlay_by_candidate,
    _wednesday_locked_policy_mask,
    materialize,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_monday_r6_explicit_rebuild_dry_run_requires_flag_and_writes_no_training_outputs(tmp_path: Path) -> None:
    monday_truth_dir = tmp_path / "MONDAY_R6_CANONICAL_TRUTH_V1_fixture"
    rehydrated_dir = tmp_path / "MONDAY_R6_REHYDRATED_WEDNESDAY_CONTRACT_V1_fixture"
    monday_truth_dir.mkdir()
    rehydrated_dir.mkdir()
    output_dir = tmp_path / "out"

    id_row = {
        "candidate_uid": "cand-001",
        "run_id": "MONDAY_RUN_2026W01",
        "trade_uid": "trade-001",
        "trade_id": "T001",
        "decision_timestamp": "2026-01-05T09:00:00Z",
    }
    pd.DataFrame([{**id_row, "as_of_candidate_tradable_prob_v1": 0.95, "as_of_skip_candidate_p_flat_v1": 0.10}]).to_parquet(
        rehydrated_dir / AS_OF_TABLE,
        index=False,
    )
    pd.DataFrame(
        [
            {
                **id_row,
                "baseline_realized_pnl_bps_v1": 24.0,
                "peak_mfe_bps_v1": 52.0,
                "mae_abs_bps_v1": 8.0,
                "giveback_bps_v1": 10.0,
            }
        ]
    ).to_parquet(rehydrated_dir / HINDSIGHT_TABLE, index=False)
    pd.DataFrame(
        [
            {
                "candidate_uid": "cand-001",
                "calendar_quarantine_status_v1": "ACTIVE_CANDIDATE",
                "calendar_quarantine_reason_v1": "",
                "truth_cata_or_friday_flat_damage_v1": False,
                "truth_exit_too_early_regret_replay_end_v1": False,
                "canonical_entry_ts_utc_v1": "2026-01-05T09:00:00Z",
            }
        ]
    ).to_parquet(monday_truth_dir / TRUTH_TABLE, index=False)

    status = materialize(
        reports_root=tmp_path,
        monday_truth_dir=monday_truth_dir,
        rehydrated_dir=rehydrated_dir,
        output_dir=output_dir,
        run_training=False,
    )

    assert status["training_started_v1"] is False
    assert status["decision_v1"] == "EXPLICIT_RUN_FLAG_REQUIRED"
    assert (output_dir / OUTPUT_FILES["status"]).exists()
    assert not (output_dir / TRAINING_FRAME).exists()


def test_assign_splits_keeps_quarantined_rows_eval_only() -> None:
    frame = pd.DataFrame(
        {
            "run_id": ["W01", "W02", "W03", "W04"],
            "calendar_quarantine_status_v1": [
                "ACTIVE_CANDIDATE",
                "ACTIVE_CANDIDATE",
                "QUARANTINED",
                "ACTIVE_CANDIDATE",
            ],
        }
    )

    split = _assign_splits(frame)

    quarantined = split["calendar_quarantine_status_v1"].eq("QUARANTINED")
    assert not split.loc[quarantined, "used_for_training"].any()
    assert not split.loc[quarantined, "used_for_validation"].any()
    assert split.loc[quarantined, "used_for_holdout"].all()
    assert set(split.loc[quarantined, "split_scope_v1"]) == {"QUARANTINE_EVAL_ONLY"}


def test_assign_splits_can_reuse_wednesday_run_split_contract() -> None:
    frame = pd.DataFrame(
        {
            "candidate_uid": ["c1", "c2", "c3", "c4"],
            "run_id": ["W01", "W01", "W02", "W03"],
            "calendar_quarantine_status_v1": [
                "ACTIVE_CANDIDATE",
                "ACTIVE_CANDIDATE",
                "ACTIVE_CANDIDATE",
                "QUARANTINED",
            ],
        }
    )
    split_reference = pd.DataFrame(
        {
            "candidate_uid": ["c1", "c2", "c3", "c4"],
            "run_id": ["W01", "W01", "W02", "W03"],
            "used_for_training": [True, False, False, True],
            "used_for_validation": [False, False, True, False],
            "used_for_holdout": [False, True, False, False],
        }
    )

    split = _assign_splits(frame, split_reference=split_reference)

    assert split.loc[split["candidate_uid"].eq("c1"), "used_for_training"].all()
    assert split.loc[split["candidate_uid"].eq("c2"), "used_for_holdout"].all()
    assert split.loc[split["candidate_uid"].eq("c3"), "used_for_validation"].all()
    quarantined = split["calendar_quarantine_status_v1"].eq("QUARANTINED")
    assert not split.loc[quarantined, "used_for_training"].any()
    assert split.loc[quarantined, "used_for_holdout"].all()


def test_compare_requires_wednesday_precision_and_worst_loso_safety() -> None:
    compare = _compare(
        {
            "bad_blocks_v1": 79,
            "tail_help_v1": 36,
            "precision_v1": 0.9634146341463414,
            "repaired_165_damage_v1": 0,
            "fifty_plus_mfe_blocked_v1": 1,
            "hundred_plus_mfe_blocked_v1": 0,
            "two_hundred_plus_mfe_blocked_v1": 0,
            "strongest_winner_damage_v1": 0,
        },
        worst_loso=0.7777777777777778,
    )

    assert compare["verdict_v1"] == "MONDAY_R6_EXPLICIT_REBUILD_RAN_BUT_FAILED_WEDNESDAY_SAFETY"
    assert compare["safety_failures_v1"] == ["precision_below_wednesday_r6", "worst_loso_below_wednesday_r6"]


def test_compare_flags_fifty_plus_over_wednesday_guardrail() -> None:
    compare = _compare(
        {
            "bad_blocks_v1": 40,
            "tail_help_v1": 20,
            "precision_v1": 1.0,
            "repaired_165_damage_v1": 0,
            "fifty_plus_mfe_blocked_v1": 2,
            "hundred_plus_mfe_blocked_v1": 0,
            "two_hundred_plus_mfe_blocked_v1": 0,
            "strongest_winner_damage_v1": 0,
        },
        worst_loso=1.0,
    )

    assert compare["verdict_v1"] == "MONDAY_R6_EXPLICIT_REBUILD_RAN_BUT_FAILED_WEDNESDAY_SAFETY"
    assert compare["safety_failures_v1"] == ["fifty_plus_mfe_blocked_v1>wednesday"]


def test_asof_runner_guard_accepts_selected_calibration_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "as_of_candidate_tradable_prob_v1": [0.88, 0.80],
            "as_of_entry_candidate_path_quality_pred_v1": [0.78, 0.78],
            "as_of_candidate_mfe_first_n_pred_v1": [1.90, 1.90],
            "as_of_skip_candidate_p_flat_v1": [0.20, 0.20],
        }
    )

    default_guard = _asof_runner_guard(frame)
    calibrated_guard = _asof_runner_guard(
        frame,
        {
            "asof_guard_tradable_min_v1": 0.87,
            "asof_guard_quality_min_v1": 0.75,
            "asof_guard_mfe_min_v1": 1.75,
            "asof_guard_flat_max_v1": 0.50,
        },
    )

    assert default_guard.tolist() == [False, False]
    assert calibrated_guard.tolist() == [True, False]


def test_wednesday_locked_policy_preserves_r5_2_base_block() -> None:
    frame = pd.DataFrame(
        {
            "r5_2_selected_candidate__block_v1": [True, False],
            "pred__entry_r6_bad_risk__prob_true_v1": [0.01, 0.99],
            "pred__entry_r6_risky_allow__prob_true_v1": [0.01, 0.99],
            "pred__entry_r6_tail_control_10_50__prob_true_v1": [0.01, 0.99],
            "pred__entry_r6_runner_protector__prob_true_v1": [0.99, 0.01],
            "pred__entry_r5_2_runner_protector__prob_true_v1": [0.99, 0.01],
            "as_of_candidate_tradable_prob_v1": [0.99, 0.50],
            "as_of_entry_candidate_path_quality_pred_v1": [0.99, 0.50],
            "as_of_candidate_mfe_first_n_pred_v1": [2.0, 0.0],
            "as_of_skip_candidate_p_flat_v1": [0.0, 1.0],
        }
    )

    mask = _wednesday_locked_policy_mask(frame)

    assert mask.tolist() == [True, True]


def test_overlay_by_candidate_uses_exact_label_source_for_matching_rows() -> None:
    frame = pd.DataFrame(
        {
            "candidate_uid": ["c1", "c2", "c3"],
            "r6_label_tail_control_10_50_v1": [False, False, False],
        }
    )
    source = pd.DataFrame(
        {
            "candidate_uid": ["c1", "c2"],
            "r6_label_tail_control_10_50_v1": [True, False],
        }
    )

    out, report = _overlay_by_candidate(frame, source, ["r6_label_tail_control_10_50_v1"])

    assert out["r6_label_tail_control_10_50_v1"].tolist() == [True, False, False]
    assert report["matched_rows_v1"] == 2
    assert report["overlaid_columns_v1"] == ["r6_label_tail_control_10_50_v1"]
