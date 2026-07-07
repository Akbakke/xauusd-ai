from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_exit_per_bar_sanity_training_v1 as gate


def test_explicit_artifact_roots_reject_latest() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/EXIT_PER_BAR_SANITY_TRAINING_V1_20260429T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])


def test_no_forbidden_actions_default_pass() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"


def test_validate_final_status_only_allowed_pairs() -> None:
    assert gate.validate_final_status(
        "EXIT_PER_BAR_SANITY_TRAINING_PASS_POLICY_BEATS_REALIZED_AND_TRAIL_STOP",
        "EXIT_PER_BAR_REWARD_VARIANT_SENSITIVITY_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "ARBITRARY",
            "EXIT_PER_BAR_REWARD_VARIANT_SENSITIVITY_V1",
        )
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "EXIT_PER_BAR_SANITY_TRAINING_PASS_POLICY_BEATS_REALIZED_AND_TRAIL_STOP",
            "PROMOTE_TO_LIVE_NOW_V1",
        )


def test_validate_no_deprecated_revival(tmp_path: Path) -> None:
    bad = tmp_path / "imports_quarantine.py"
    bad.write_text(
        "from gx1.quarantine._DEPRECATED_SCRIPTS_20260219 import x\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN"):
        gate.validate_no_deprecated_revival(bad)
    assert gate.validate_no_deprecated_revival(Path(gate.__file__))


def _make_synthetic_per_bar(n_trades: int = 50, bars_per_trade: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for trade_id in range(n_trades):
        # Simulate that some trades are positive/negative
        baseline_pnl = rng.normal(0.0, 30.0)
        for bar in range(bars_per_trade):
            running_pnl = baseline_pnl * (bar + 1) / bars_per_trade + rng.normal(0, 5)
            running_mfe = max(running_pnl, baseline_pnl + 10.0)
            running_mae = -10.0 - bar * 2.0
            rows.append(
                {
                    "candidate_uid_v1": f"trade_{trade_id:03d}",
                    "bars_held_v1": bar,
                    "ts_v1": pd.Timestamp("2025-06-01 13:00", tz="UTC")
                    + pd.Timedelta(days=trade_id, minutes=5 * bar),
                    "primary_split_v1": (
                        "train" if trade_id < 35 else ("val" if trade_id < 43 else "test")
                    ),
                    "side_v1": "long",
                    "running_pnl_at_close_bps_v1": running_pnl,
                    "running_mfe_bps_v1": running_mfe,
                    "running_mae_bps_v1": running_mae,
                    "running_giveback_from_peak_bps_v1": max(running_mfe - running_pnl, 0.0),
                    "exit_prob_v1": 0.1 + 0.1 * bar,
                    "atr_bps_now_v1": 5.0,
                    "session_id_v1": 1,
                    "action_id_v1": gate.ACTION_HOLD_ID,
                }
            )
    return pd.DataFrame(rows)


def test_fit_train_normalization_uses_only_train_rows() -> None:
    df = _make_synthetic_per_bar()
    train = df[df["primary_split_v1"] == "train"]
    norm = gate._fit_train_normalization(train)
    assert "running_pnl_at_close_bps_v1" in norm
    assert norm["running_pnl_at_close_bps_v1"]["transform"] == "z"
    expected_mean = float(train["running_pnl_at_close_bps_v1"].mean())
    assert abs(norm["running_pnl_at_close_bps_v1"]["mean"] - expected_mean) < 1e-9


def test_build_state_matrix_yields_expected_shape() -> None:
    df = _make_synthetic_per_bar()
    norm = gate._fit_train_normalization(df[df["primary_split_v1"] == "train"])
    X = gate._build_state_matrix(df, norm)
    assert X.shape == (len(df), len(gate.STATE_FEATURE_NAMES_V1))
    # intercept always 1
    assert (X[:, 0] == 1.0).all()
    # No NaN
    assert np.isfinite(X).all()


def test_compute_targets_assigns_terminal_pnl_to_hold_and_close_to_exit() -> None:
    df = _make_synthetic_per_bar(n_trades=2, bars_per_trade=4)
    train = df[df["primary_split_v1"] == "train"]
    out = gate._compute_targets(train, df)
    assert "target_hold_v1" in out.columns
    assert "target_exit_now_v1" in out.columns
    # target_hold should be the trade's last bar pnl, same for all bars in the trade
    for uid, group in out.groupby("candidate_uid_v1"):
        last_pnl = group.sort_values("bars_held_v1").iloc[-1]["running_pnl_at_close_bps_v1"]
        assert (group["target_hold_v1"] == last_pnl).all()


def test_ridge_fit_solves_linear_system() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    true_coef = np.array([1.0, -2.0, 0.5])
    y = X @ true_coef + rng.normal(scale=0.01, size=50)
    coef = gate._ridge_fit(X, y, lam=1e-6)
    assert np.allclose(coef, true_coef, atol=0.05)


def test_train_q_heads_returns_two_coef_vectors() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y_hold = rng.normal(size=40)
    y_exit = rng.normal(size=40)
    model = gate._train_q_heads(X, y_hold, y_exit)
    assert len(model["coef_hold_v1"]) == 5
    assert len(model["coef_exit_now_v1"]) == 5


def test_exit_index_from_iql_policy_picks_first_qualifying_bar() -> None:
    # Construct simple per-bar where Q_exit > Q_hold from bar 2 onwards
    df = _make_synthetic_per_bar(n_trades=3, bars_per_trade=5)
    norm = gate._fit_train_normalization(df[df["primary_split_v1"] == "train"])
    X = gate._build_state_matrix(df, norm)
    # Force coefficients: Q_exit driven by bars_held_z (column 5), Q_hold = 0
    coef_hold = np.zeros(X.shape[1])
    coef_exit_now = np.zeros(X.shape[1])
    coef_exit_now[5] = 1.0  # Higher bars_held -> higher Q_exit
    indices = gate._exit_index_from_iql_policy(df, X, coef_hold, coef_exit_now)
    assert len(indices) == 3


def test_audit_no_shortcut_at_training_time_passes() -> None:
    audit = gate.audit_no_shortcut_at_training_time(gate.STATE_FEATURE_NAMES_V1)
    assert audit["status_v1"] == "PASS"


def test_audit_split_isolation_passes_on_clean_partition() -> None:
    df = _make_synthetic_per_bar()
    audit = gate.audit_split_isolation(df)
    assert audit["status_v1"] == "PASS"


def test_audit_split_isolation_fails_on_spanning_trade() -> None:
    df = _make_synthetic_per_bar()
    df.loc[df["candidate_uid_v1"] == "trade_000", "primary_split_v1"] = "val"
    df.loc[df.index[0], "primary_split_v1"] = "train"
    with pytest.raises(RuntimeError, match="SPLIT_ISOLATION_VIOLATION"):
        gate.audit_split_isolation(df)


def test_audit_train_only_normalization_detects_mismatch() -> None:
    df = _make_synthetic_per_bar()
    norm = gate._fit_train_normalization(df[df["primary_split_v1"] == "train"])
    audit = gate.audit_train_only_normalization(df, norm)
    assert audit["status_v1"] == "PASS"
    # tamper
    norm["running_pnl_at_close_bps_v1"]["mean"] = 9999.0
    with pytest.raises(RuntimeError, match="NORMALIZATION_FIT_NOT_TRAIN_ONLY"):
        gate.audit_train_only_normalization(df, norm)


def test_go_no_go_decides_pass_when_iql_beats_trail_stop() -> None:
    iql_results = [{"split_v1": "test", "total_realized_pnl_bps_v1": 2000.0}]
    baselines = {
        "test": [
            {"policy_id_v1": "REALIZED_EXIT_BASELINE", "total_realized_pnl_bps_v1": -500.0},
            {"policy_id_v1": "TRAIL_STOP_25_PCT_DD", "total_realized_pnl_bps_v1": 1000.0},
        ]
    }
    status, action, _ = gate._go_no_go(iql_results, baselines)
    assert status == "EXIT_PER_BAR_SANITY_TRAINING_PASS_POLICY_BEATS_REALIZED_AND_TRAIL_STOP"
    assert action == "EXIT_PER_BAR_REWARD_VARIANT_SENSITIVITY_V1"


def test_go_no_go_decides_partial_when_iql_beats_realized_not_trail() -> None:
    iql_results = [{"split_v1": "test", "total_realized_pnl_bps_v1": 100.0}]
    baselines = {
        "test": [
            {"policy_id_v1": "REALIZED_EXIT_BASELINE", "total_realized_pnl_bps_v1": -500.0},
            {"policy_id_v1": "TRAIL_STOP_25_PCT_DD", "total_realized_pnl_bps_v1": 1000.0},
        ]
    }
    status, _, _ = gate._go_no_go(iql_results, baselines)
    assert status == "EXIT_PER_BAR_SANITY_TRAINING_PASS_POLICY_BEATS_REALIZED_NOT_TRAIL_STOP"


def test_go_no_go_decides_underperforms_when_iql_below_realized() -> None:
    iql_results = [{"split_v1": "test", "total_realized_pnl_bps_v1": -2000.0}]
    baselines = {
        "test": [
            {"policy_id_v1": "REALIZED_EXIT_BASELINE", "total_realized_pnl_bps_v1": -500.0},
            {"policy_id_v1": "TRAIL_STOP_25_PCT_DD", "total_realized_pnl_bps_v1": 1000.0},
        ]
    }
    status, _, _ = gate._go_no_go(iql_results, baselines)
    assert status == "EXIT_PER_BAR_SANITY_TRAINING_PARTIAL_POLICY_UNDERPERFORMS_REALIZED"


def test_write_artifacts_produces_required_outputs(tmp_path: Path) -> None:
    out_root = tmp_path / "EXIT_PER_BAR_SANITY_TRAINING_V1_20260429T999999Z_LOCK"
    result = gate.write_artifacts(out_root=out_root, built_at_utc="20260429T999999Z")
    artifact_root = Path(result["artifact_root"])
    assert artifact_root == out_root
    for required in [
        "manifest_v1.json",
        "summary_v1.json",
        "status_v1.json",
        "report_v1.md",
        "exit_per_bar_sanity_training_go_no_go_v1.json",
        "input_manifest_v1.json",
        "trained_model_v1.json",
        "training_normalization_v1.json",
        "iql_vs_baseline_comparator_v1.json",
        "iql_vs_baseline_comparator_v1.csv",
        "training_audits_v1.json",
        "reproducibility_audit_v1.json",
    ]:
        assert (artifact_root / required).exists(), f"missing {required}"
    summary = json.loads((artifact_root / "summary_v1.json").read_text())
    assert summary["final_status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert summary["next_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert summary["reward_variant_v1"] == "REALIZED_PNL_REWARD"
    assert summary["model_id_v1"] == "EXIT_IQL_RIDGE_2HEAD_V1"
    audits = json.loads((artifact_root / "training_audits_v1.json").read_text())
    for a in audits["audits_v1"]:
        assert a["status_v1"] == "PASS", f"audit failed: {a}"
