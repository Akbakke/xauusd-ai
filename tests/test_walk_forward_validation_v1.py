"""Tests for materialize_walk_forward_validation_v1."""
from __future__ import annotations

import pandas as pd
import pytest

from gx1.scripts import materialize_walk_forward_validation_v1 as gate


def test_fold_definitions_have_three_folds() -> None:
    assert len(gate.FOLD_DEFINITIONS) == 3


def test_fold_definitions_pass_validation() -> None:
    audit = gate.validate_fold_definitions(gate.FOLD_DEFINITIONS)
    assert audit["status_v1"] == "PASS"


def test_fold_definitions_validate_rejects_inverted_range() -> None:
    bad = [
        {
            "fold_id_v1": "BAD",
            "train_start_v1": 100,
            "train_end_v1": 50,  # < train_start
            "val_end_v1": 200,
            "test_end_v1": 300,
        }
    ]
    with pytest.raises(RuntimeError, match="FOLD_RANGE_INVALID"):
        gate.validate_fold_definitions(bad)


def test_fold_definitions_validate_rejects_empty_list() -> None:
    with pytest.raises(RuntimeError, match="FOLD_DEFINITIONS_EMPTY"):
        gate.validate_fold_definitions([])


def test_assign_fold_split_assigns_each_rank_correctly() -> None:
    uids = [f"UID_{i:04d}" for i in range(1, 11)]  # 1..10
    fold = {"train_start_v1": 1, "train_end_v1": 5, "val_end_v1": 7, "test_end_v1": 9}
    out = gate._assign_fold_split(uids, fold)
    # ranks 1-5 = train, 6-7 = val, 8-9 = test, 10 = hold_out
    assert out["UID_0001"] == "train"
    assert out["UID_0005"] == "train"
    assert out["UID_0006"] == "val"
    assert out["UID_0007"] == "val"
    assert out["UID_0008"] == "test"
    assert out["UID_0009"] == "test"
    assert out["UID_0010"] == "hold_out"


def test_assign_fold_split_handles_train_start_above_one() -> None:
    uids = [f"UID_{i:04d}" for i in range(1, 11)]
    fold = {"train_start_v1": 3, "train_end_v1": 6, "val_end_v1": 8, "test_end_v1": 10}
    out = gate._assign_fold_split(uids, fold)
    assert out["UID_0001"] == "hold_out"
    assert out["UID_0002"] == "hold_out"
    assert out["UID_0003"] == "train"
    assert out["UID_0006"] == "train"
    assert out["UID_0007"] == "val"
    assert out["UID_0009"] == "test"
    assert out["UID_0010"] == "test"


def test_override_split_column_drops_hold_out_rows() -> None:
    df = pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "B", "C", "D"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    uid_to_split = {"A": "train", "B": "test", "C": "hold_out", "D": "val"}
    out = gate._override_split_column(df, uid_to_split, "candidate_uid_v1")
    assert len(out) == 3
    assert "C" not in out["candidate_uid_v1"].values
    assert sorted(out["primary_split_v1"].unique()) == ["test", "train", "val"]


def test_stability_metrics_handles_empty_input() -> None:
    out = gate._stability_metrics([])
    assert out["n_v1"] == 0
    assert out["mean_v1"] is None


def test_stability_metrics_computes_basic_statistics() -> None:
    out = gate._stability_metrics([100.0, -50.0, 200.0])
    assert out["n_v1"] == 3
    assert out["n_positive_v1"] == 2
    assert out["mean_v1"] == pytest.approx(83.333, abs=0.01)
    assert out["min_v1"] == -50.0
    assert out["max_v1"] == 200.0


def test_classify_stability_stable_when_all_positive_and_high_mean() -> None:
    stats = gate._stability_metrics([300.0, 400.0, 500.0])
    assert gate._classify_stability(stats) == "STABLE_ALL_FOLDS_POSITIVE_MEAN_LIFT_OVER_200"


def test_classify_stability_partial_when_mostly_positive() -> None:
    stats = gate._stability_metrics([200.0, 300.0, -50.0])
    assert (
        gate._classify_stability(stats)
        == "PARTIAL_STABLE_MOSTLY_POSITIVE_MEAN_LIFT_OVER_100"
    )


def test_classify_stability_not_stable_when_negative_in_majority() -> None:
    stats = gate._stability_metrics([-1000.0, -500.0, 1000.0])
    assert gate._classify_stability(stats) == "NOT_STABLE"


def test_classify_stability_not_stable_when_mean_too_low() -> None:
    stats = gate._stability_metrics([10.0, 20.0, 30.0])
    # All positive but mean is way below 200 -> not classified as stable.
    assert gate._classify_stability(stats) == "NOT_STABLE"


def _mk_per_fold_result(fold_id, no_skip_real, skip_real, no_skip_iql, skip_iql):
    return {
        "fold_id_v1": fold_id,
        "split_v1": "test",
        "tuned_threshold_v1": 0.5,
        "trade_count_total_v1": 100,
        "trade_count_taken_v1": 60,
        "trade_count_skipped_v1": 40,
        "pnl_no_skip_realized_v1": no_skip_real,
        "pnl_skip_then_realized_v1": skip_real,
        "per_variant_v1": [
            {
                "reward_variant_v1": "GIVEBACK_PENALTY_REWARD",
                "split_v1": "test",
                "trade_count_total_v1": 100,
                "trade_count_taken_v1": 60,
                "trade_count_skipped_v1": 40,
                "pnl_no_skip_realized_v1": no_skip_real,
                "pnl_no_skip_iql_v1": no_skip_iql,
                "pnl_skip_then_realized_v1": skip_real,
                "pnl_skip_then_iql_v1": skip_iql,
                "lift_skip_only_v1": skip_real - no_skip_real,
                "lift_iql_only_v1": no_skip_iql - no_skip_real,
                "lift_combined_v1": skip_iql - no_skip_real,
                "lift_combined_minus_sum_of_components_v1": (
                    (skip_iql - no_skip_real)
                    - ((skip_real - no_skip_real) + (no_skip_iql - no_skip_real))
                ),
            }
        ],
    }


def test_go_no_go_pass_when_skip_stable_all_folds() -> None:
    # Skip lift > 200 on every fold.
    folds = [
        _mk_per_fold_result("F1", no_skip_real=-100.0, skip_real=400.0, no_skip_iql=100.0, skip_iql=500.0),
        _mk_per_fold_result("F2", no_skip_real=-200.0, skip_real=300.0, no_skip_iql=50.0, skip_iql=400.0),
        _mk_per_fold_result("F3", no_skip_real=-50.0, skip_real=350.0, no_skip_iql=200.0, skip_iql=550.0),
    ]
    status, action, _, headline = gate._go_no_go(folds)
    assert status == "WALK_FORWARD_VALIDATION_PASS_SKIP_STABLE_ALL_FOLDS"
    assert action == "DEFINE_PROMOTION_CRITERIA_BEFORE_PAPER_TRADING_V1"
    assert headline["skip_only_classification_v1"] == "STABLE_ALL_FOLDS_POSITIVE_MEAN_LIFT_OVER_200"


def test_go_no_go_pass_partial_when_skip_mostly_positive() -> None:
    folds = [
        _mk_per_fold_result("F1", -100.0, 200.0, 50.0, 250.0),
        _mk_per_fold_result("F2", -50.0, 150.0, 0.0, 200.0),
        _mk_per_fold_result("F3", 100.0, 50.0, 200.0, 150.0),  # skip -50
    ]
    status, action, _, headline = gate._go_no_go(folds)
    # Skip lifts: 300, 200, -50 -> mean=150, n_pos=2/3, min=-50 -> partial classification.
    assert headline["skip_only_classification_v1"] == "PARTIAL_STABLE_MOSTLY_POSITIVE_MEAN_LIFT_OVER_100"
    assert status == "WALK_FORWARD_VALIDATION_PASS_SKIP_PARTIAL_STABLE"


def test_go_no_go_partial_when_skip_not_stable() -> None:
    # Matches the actual gate run: skip wins fold 3, loses folds 1+2.
    folds = [
        _mk_per_fold_result("F1", 88.0, -1294.0, -783.0, -83.0),
        _mk_per_fold_result("F2", 1301.0, 816.0, 106.0, 353.0),
        _mk_per_fold_result("F3", -1862.0, -803.0, 219.0, 344.0),
    ]
    status, action, _, headline = gate._go_no_go(folds)
    assert status == "WALK_FORWARD_VALIDATION_PARTIAL_SKIP_NOT_STABLE"
    assert action == "REPAIR_RESEARCH_STACK_BEFORE_FURTHER_WORK_V1"
    assert headline["skip_only_classification_v1"] == "NOT_STABLE"


def test_go_no_go_raises_on_empty_folds() -> None:
    with pytest.raises(RuntimeError, match="WALK_FORWARD_PER_FOLD_RESULTS_MISSING"):
        gate._go_no_go([])


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("MADE_UP", "DEFINE_PROMOTION_CRITERIA_BEFORE_PAPER_TRADING_V1")


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "WALK_FORWARD_VALIDATION_PASS_SKIP_STABLE_ALL_FOLDS", "TRAIN_NOW"
        )


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))
