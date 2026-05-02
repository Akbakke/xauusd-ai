"""Tests for materialize_learn_trade_skip_meta_classifier_at_trade_open_v1."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_learn_trade_skip_meta_classifier_at_trade_open_v1 as gate


def test_should_skip_label_matches_v2_formula() -> None:
    yes = pd.Series({"pnl_bps": -100.0, "mae_bps": -80.0, "mfe_bps": 10.0})
    no_pnl_positive = pd.Series({"pnl_bps": 50.0, "mae_bps": -80.0, "mfe_bps": 10.0})
    no_mae_too_small = pd.Series({"pnl_bps": -10.0, "mae_bps": -30.0, "mfe_bps": 10.0})
    no_mfe_too_high = pd.Series({"pnl_bps": -10.0, "mae_bps": -80.0, "mfe_bps": 50.0})
    assert gate.compute_should_skip_label(yes) == 1
    assert gate.compute_should_skip_label(no_pnl_positive) == 0
    assert gate.compute_should_skip_label(no_mae_too_small) == 0
    assert gate.compute_should_skip_label(no_mfe_too_high) == 0


def test_should_skip_vectorized_matches_scalar() -> None:
    df = pd.DataFrame(
        {
            "pnl_bps": [-100.0, 50.0, -10.0, -200.0],
            "mae_bps": [-80.0, -80.0, -30.0, -100.0],
            "mfe_bps": [10.0, 10.0, 10.0, 5.0],
        }
    )
    out = gate.compute_should_skip_labels_vectorized(df)
    expected = np.array(
        [
            gate.compute_should_skip_label(df.iloc[i]) for i in range(len(df))
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(out, expected)
    assert out.tolist() == [1, 0, 0, 1]


def test_validate_label_formula_against_v2_contract_passes() -> None:
    fake_v2 = {
        "audit_only_labels_v1": [
            {
                "label_name_v2": "audit_should_have_skipped_v2",
                "formula_v2": "for each trade: 1 if pnl_bps < 0 AND mae_bps <= -50 AND mfe_bps < 25; else 0",
            }
        ]
    }
    audit = gate.validate_label_formula_against_v2_contract(fake_v2)
    assert audit["status_v1"] == "PASS"
    assert audit["all_tokens_present_v1"] is True


def test_validate_label_formula_raises_when_label_missing() -> None:
    with pytest.raises(RuntimeError, match="AUDIT_SHOULD_HAVE_SKIPPED_V2_NOT_FOUND"):
        gate.validate_label_formula_against_v2_contract({"audit_only_labels_v1": []})


def test_validate_label_formula_marks_fail_when_token_missing() -> None:
    fake_v2 = {
        "audit_only_labels_v1": [
            {
                "label_name_v2": "audit_should_have_skipped_v2",
                "formula_v2": "totally different formula",
            }
        ]
    }
    audit = gate.validate_label_formula_against_v2_contract(fake_v2)
    assert audit["status_v1"] == "FAIL"
    assert audit["all_tokens_present_v1"] is False


def test_zscore_handles_nan_via_median_imputation() -> None:
    s = pd.Series([1.0, 2.0, 3.0, np.nan, 5.0])
    cfg = {"transform": "z", "mean": 2.75, "std": 1.5, "median": 2.5}
    out = gate._zscore(s, cfg)
    assert np.isfinite(out).all()
    assert pytest.approx(out[3]) == (2.5 - 2.75) / 1.5


def test_passthrough_zero_one_with_sentinel_marks_nan() -> None:
    s = pd.Series([0.5, 0.7, np.nan])
    out = gate._passthrough_zero_one_with_sentinel(s)
    assert out[0] == 0.5
    assert out[1] == 0.7
    assert out[2] == gate.RECOVERY_SENTINEL_VALUE


def test_onehot_yields_correct_indicators() -> None:
    s = pd.Series(["EU", "US", "ASIA"])
    out = gate._onehot(s, ["ASIA", "EU", "US"])
    np.testing.assert_array_equal(out[0], [0, 1, 0])
    np.testing.assert_array_equal(out[1], [0, 0, 1])
    np.testing.assert_array_equal(out[2], [1, 0, 0])


def test_audit_no_shortcut_passes_on_clean_features() -> None:
    audit = gate.audit_no_shortcut_at_train_time(
        ["intercept", "p_long_entry_v1__pass_or_sentinel", "side__LONG"]
    )
    assert audit["status_v1"] == "PASS"


def test_audit_no_shortcut_rejects_pnl_bps_in_feature() -> None:
    with pytest.raises(RuntimeError, match="SKIP_CLASSIFIER_FEATURE_LEAK"):
        gate.audit_no_shortcut_at_train_time(["intercept", "running_pnl_bps_v1__z"])


def test_audit_no_shortcut_rejects_mae_in_feature() -> None:
    with pytest.raises(RuntimeError, match="SKIP_CLASSIFIER_FEATURE_LEAK"):
        gate.audit_no_shortcut_at_train_time(["mae_bps__z"])


def test_audit_no_shortcut_rejects_post_exit() -> None:
    with pytest.raises(RuntimeError, match="SKIP_CLASSIFIER_FEATURE_LEAK"):
        gate.audit_no_shortcut_at_train_time(["post_exit_drift_v2"])


def test_audit_no_shortcut_rejects_running_state_features() -> None:
    with pytest.raises(RuntimeError, match="SKIP_CLASSIFIER_FEATURE_LEAK"):
        gate.audit_no_shortcut_at_train_time(["running_mfe_bps_v1__z"])


def test_audit_split_isolation_passes_on_clean_assignment() -> None:
    df = pd.DataFrame(
        {
            "candidate_uid_v1": ["A", "B", "C"],
            "primary_split_v1": ["train", "val", "test"],
        }
    )
    audit = gate.audit_split_isolation(df)
    assert audit["status_v1"] == "PASS"


def test_evaluate_threshold_computes_confusion_correctly() -> None:
    per_trade = pd.DataFrame(
        {
            "should_skip_v1": [1, 1, 0, 0],
            "pnl_bps": [-100.0, -50.0, 10.0, 20.0],
        }
    )
    p_skip = np.array([0.8, 0.2, 0.7, 0.1])  # predicts row 0 and row 2 as skip
    out = gate._evaluate_threshold(per_trade, p_skip, threshold=0.5)
    assert out["tp_v1"] == 1  # row 0 actual=1 pred=1
    assert out["fp_v1"] == 1  # row 2 actual=0 pred=1
    assert out["fn_v1"] == 1  # row 1 actual=1 pred=0
    assert out["tn_v1"] == 1  # row 3 actual=0 pred=0
    assert out["trades_skipped_v1"] == 2
    assert out["trades_taken_v1"] == 2
    assert out["pnl_no_skip_v1"] == pytest.approx(-100.0 + -50.0 + 10.0 + 20.0)
    assert out["pnl_taken_v1"] == pytest.approx(-50.0 + 20.0)


def test_evaluate_threshold_skipping_none_matches_no_skip() -> None:
    per_trade = pd.DataFrame(
        {"should_skip_v1": [1, 0], "pnl_bps": [-10.0, 5.0]}
    )
    p_skip = np.array([0.1, 0.1])
    out = gate._evaluate_threshold(per_trade, p_skip, threshold=0.5)
    assert out["trades_skipped_v1"] == 0
    assert out["pnl_taken_v1"] == out["pnl_no_skip_v1"]


def test_evaluate_oracle_skip_excludes_actual_should_skip() -> None:
    per_trade = pd.DataFrame(
        {
            "should_skip_v1": [1, 0, 1, 0],
            "pnl_bps": [-100.0, 5.0, -50.0, 10.0],
        }
    )
    out = gate._evaluate_oracle_skip(per_trade)
    assert out["trades_skipped_v1"] == 2
    assert out["trades_taken_v1"] == 2
    assert out["pnl_no_skip_v1"] == pytest.approx(-135.0)
    assert out["pnl_taken_v1"] == pytest.approx(15.0)
    assert out["pnl_lift_vs_no_skip_v1"] == pytest.approx(150.0)


def test_ridge_fit_recovers_least_squares_solution() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 3))
    true_beta = np.array([0.5, -0.3, 0.8])
    y = X @ true_beta + rng.normal(scale=0.01, size=50)
    beta_hat = gate._ridge_fit(X, y, lam=1e-6)
    np.testing.assert_allclose(beta_hat, true_beta, atol=0.05)


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "MADE_UP", "COMBINE_SKIP_CLASSIFIER_WITH_EXIT_IQL_V2_V1"
        )


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "LEARN_TRADE_SKIP_META_CLASSIFIER_PASS_TUNED_THRESHOLD_LIFTS_TEST_PNL",
            "TRAIN_NOW",
        )


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    from pathlib import Path

    gate.validate_no_deprecated_revival(Path(gate.__file__))


def test_threshold_grid_includes_low_thresholds() -> None:
    """Predicted probs are biased toward 0 (label rate ~14% + ridge MSE), so
    the grid must include thresholds below 0.3 to make the policy useful."""
    assert min(gate.THRESHOLD_GRID) < 0.30
    assert 0.50 in gate.THRESHOLD_GRID


def _fake_test_metric(threshold, pnl_no_skip, pnl_taken, trades_skipped):
    return {
        "threshold_v1": threshold,
        "trade_count_v1": 100,
        "tp_v1": 0, "fp_v1": 0, "tn_v1": 0, "fn_v1": 0,
        "precision_v1": 0.0, "recall_v1": 0.0, "f1_v1": None,
        "trades_skipped_v1": trades_skipped, "trades_taken_v1": 100 - trades_skipped,
        "pnl_no_skip_v1": pnl_no_skip,
        "pnl_taken_v1": pnl_taken,
        "pnl_skipped_v1": pnl_no_skip - pnl_taken,
        "pnl_lift_vs_no_skip_v1": pnl_taken - pnl_no_skip,
    }


def test_go_no_go_pass_when_lift_above_100() -> None:
    locked = _fake_test_metric(0.15, pnl_no_skip=-100.0, pnl_taken=300.0, trades_skipped=20)
    oracle = {"pnl_lift_vs_no_skip_v1": 1000.0, "pnl_taken_v1": 900.0}
    status, action, _, headline = gate._go_no_go(locked, oracle)
    assert status == "LEARN_TRADE_SKIP_META_CLASSIFIER_PASS_TUNED_THRESHOLD_LIFTS_TEST_PNL"
    assert action == "COMBINE_SKIP_CLASSIFIER_WITH_EXIT_IQL_V2_V1"
    assert headline["captured_fraction_of_oracle_lift_v1"] == pytest.approx(0.4)


def test_go_no_go_partial_when_lift_within_50_bps() -> None:
    locked = _fake_test_metric(0.5, pnl_no_skip=-100.0, pnl_taken=-90.0, trades_skipped=2)
    oracle = {"pnl_lift_vs_no_skip_v1": 1000.0, "pnl_taken_v1": 900.0}
    status, _, _, _ = gate._go_no_go(locked, oracle)
    assert status == "LEARN_TRADE_SKIP_META_CLASSIFIER_PARTIAL_TUNED_THRESHOLD_TIES_REALIZED"


def test_go_no_go_partial_when_lift_negative() -> None:
    locked = _fake_test_metric(0.5, pnl_no_skip=-100.0, pnl_taken=-300.0, trades_skipped=10)
    oracle = {"pnl_lift_vs_no_skip_v1": 1000.0, "pnl_taken_v1": 900.0}
    status, action, _, _ = gate._go_no_go(locked, oracle)
    assert status == "LEARN_TRADE_SKIP_META_CLASSIFIER_PARTIAL_TUNED_THRESHOLD_DEGRADES_PNL"
    assert action == "REPAIR_SKIP_CLASSIFIER_BEFORE_PROMOTION_V1"
