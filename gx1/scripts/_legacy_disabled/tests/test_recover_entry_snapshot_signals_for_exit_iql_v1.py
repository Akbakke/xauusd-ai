"""Tests for materialize_recover_entry_snapshot_signals_for_exit_iql_v1.

We exercise the bridge-formula helper, per-week join with synthetic
parquets, the coverage thresholds, the no-shortcut/forbidden-actions
plumbing, and the deterministic concat order. We do NOT touch the real
DEFAULT_REPORTS_ROOT - tests use isolated tmp directories with
hand-crafted parquets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import (
    materialize_recover_entry_snapshot_signals_for_exit_iql_v1 as recover_gate,
)


# ---------------------------------------------------------------------------
# Bridge-formula unit tests
# ---------------------------------------------------------------------------


def test_bridge_fields_for_long_winner() -> None:
    out = recover_gate._compute_bridge_fields(
        np.array([0.7]), np.array([0.2]), np.array([0.1])
    )
    assert pytest.approx(out["p_long_entry_v1"][0]) == 0.7
    assert pytest.approx(out["p_hat_entry_v1"][0]) == 0.7
    assert pytest.approx(out["uncertainty_entry_v1"][0]) == 0.3
    assert pytest.approx(out["margin_entry_v1"][0]) == 0.5  # 0.7 - 0.2


def test_bridge_fields_for_short_winner() -> None:
    out = recover_gate._compute_bridge_fields(
        np.array([0.1]), np.array([0.6]), np.array([0.3])
    )
    assert pytest.approx(out["p_long_entry_v1"][0]) == 0.1
    assert pytest.approx(out["p_hat_entry_v1"][0]) == 0.6
    assert pytest.approx(out["uncertainty_entry_v1"][0]) == 0.4
    assert pytest.approx(out["margin_entry_v1"][0]) == 0.3  # 0.6 - 0.3


def test_bridge_fields_for_flat_winner_three_way_tie() -> None:
    out = recover_gate._compute_bridge_fields(
        np.array([1 / 3]), np.array([1 / 3]), np.array([1 / 3])
    )
    assert pytest.approx(out["p_hat_entry_v1"][0]) == 1 / 3
    assert pytest.approx(out["uncertainty_entry_v1"][0]) == 2 / 3
    assert pytest.approx(out["margin_entry_v1"][0], abs=1e-12) == 0.0


def test_bridge_fields_vectorized() -> None:
    pl = np.array([0.6, 0.1, 0.45])
    ps = np.array([0.3, 0.7, 0.45])
    pf = np.array([0.1, 0.2, 0.10])
    out = recover_gate._compute_bridge_fields(pl, ps, pf)
    np.testing.assert_allclose(out["p_long_entry_v1"], pl)
    np.testing.assert_allclose(out["p_hat_entry_v1"], np.maximum.reduce([pl, ps, pf]))
    np.testing.assert_allclose(out["uncertainty_entry_v1"], 1.0 - out["p_hat_entry_v1"])
    expected_margin = np.array([0.6 - 0.3, 0.7 - 0.2, 0.45 - 0.45])
    np.testing.assert_allclose(out["margin_entry_v1"], expected_margin)


# ---------------------------------------------------------------------------
# Per-week recovery on synthetic data
# ---------------------------------------------------------------------------


def _make_week(
    tmp_path: Path,
    week_name: str,
    trades: pd.DataFrame,
    xgb: pd.DataFrame | None,
) -> Path:
    week_dir = tmp_path / week_name
    week_dir.mkdir(parents=True, exist_ok=True)
    trades.to_parquet(week_dir / f"trade_outcomes_{week_name}_MERGED.parquet", index=False)
    if xgb is not None:
        xgb.to_parquet(
            week_dir / f"xgb_multi_horizon_predictions_{week_name}.parquet", index=False
        )
    return week_dir


def test_recover_week_full_match(tmp_path: Path) -> None:
    week = "TRUTH_MONFRI_WEEK_20250101_20250108"
    ts = pd.Timestamp("2025-01-02 13:00:00", tz="UTC")
    trades = pd.DataFrame(
        {
            "candidate_uid": [f"CAND_{week}_{i}" for i in range(2)],
            "trade_uid": [f"TRADE_{week}_{i}" for i in range(2)],
            "open_ts_utc": [ts, ts + pd.Timedelta(minutes=5)],
            "session": ["OVERLAP", "OVERLAP"],
            "side": ["long", "short"],
        }
    )
    xgb = pd.DataFrame(
        {
            "ts": [ts, ts + pd.Timedelta(minutes=5)],
            "head": ["OVERLAP", "OVERLAP"],
            "p_long": [0.6, 0.1],
            "p_short": [0.3, 0.7],
            "p_flat": [0.1, 0.2],
            "p_hat": [0.6, 0.7],
        }
    )
    week_dir = _make_week(tmp_path, week, trades, xgb)
    out, audit = recover_gate._recover_week(week_dir)
    assert audit["match_rate_v1"] == 1.0
    assert audit["matched_count_v1"] == 2
    assert audit["xgb_predictions_present_v1"] is True
    assert (out["recovery_status_v1"] == "RECOVERED_FROM_XGB_PREDICTIONS").all()
    assert pytest.approx(out["p_hat_entry_v1"].iloc[0]) == 0.6
    assert pytest.approx(out["margin_entry_v1"].iloc[1]) == 0.5  # 0.7 - 0.2


def test_recover_week_falls_back_when_session_does_not_match_head(tmp_path: Path) -> None:
    week = "TRUTH_MONFRI_WEEK_20250115_20250122"
    ts = pd.Timestamp("2025-01-16 14:00:00", tz="UTC")
    trades = pd.DataFrame(
        {
            "candidate_uid": [f"CAND_{week}_0"],
            "trade_uid": [f"TRADE_{week}_0"],
            "open_ts_utc": [ts],
            "session": ["US"],  # trade says US
            "side": ["long"],
        }
    )
    xgb = pd.DataFrame(
        {
            "ts": [ts],
            "head": ["OVERLAP"],  # but xgb head is OVERLAP at this ts
            "p_long": [0.55],
            "p_short": [0.30],
            "p_flat": [0.15],
            "p_hat": [0.55],
        }
    )
    week_dir = _make_week(tmp_path, week, trades, xgb)
    out, audit = recover_gate._recover_week(week_dir)
    assert audit["match_rate_v1"] == 1.0
    assert out["recovery_status_v1"].iloc[0] == "RECOVERED_FROM_XGB_PREDICTIONS"
    assert out["xgb_head_used_v1"].iloc[0] == "OVERLAP"
    assert pytest.approx(out["p_long_entry_v1"].iloc[0]) == 0.55
    assert pytest.approx(out["uncertainty_entry_v1"].iloc[0]) == pytest.approx(0.45)


def test_recover_week_marks_unmatched_when_xgb_missing_for_ts(tmp_path: Path) -> None:
    week = "TRUTH_MONFRI_WEEK_20250108_20250115"
    ts_trade = pd.Timestamp("2025-01-09 11:00:00", tz="UTC")
    ts_xgb = pd.Timestamp("2025-01-09 14:00:00", tz="UTC")  # different ts
    trades = pd.DataFrame(
        {
            "candidate_uid": [f"CAND_{week}_0"],
            "trade_uid": [f"TRADE_{week}_0"],
            "open_ts_utc": [ts_trade],
            "session": ["EU"],
            "side": ["long"],
        }
    )
    xgb = pd.DataFrame(
        {
            "ts": [ts_xgb],
            "head": ["EU"],
            "p_long": [0.5],
            "p_short": [0.3],
            "p_flat": [0.2],
            "p_hat": [0.5],
        }
    )
    week_dir = _make_week(tmp_path, week, trades, xgb)
    out, audit = recover_gate._recover_week(week_dir)
    assert audit["match_rate_v1"] == 0.0
    assert audit["matched_count_v1"] == 0
    assert out["recovery_status_v1"].iloc[0] == "NOT_RECOVERED_TS_NOT_IN_XGB"
    assert pd.isna(out["p_long_entry_v1"].iloc[0])


def test_recover_week_handles_missing_xgb_parquet(tmp_path: Path) -> None:
    week = "TRUTH_MONFRI_WEEK_20250122_20250129"
    trades = pd.DataFrame(
        {
            "candidate_uid": [f"CAND_{week}_0"],
            "trade_uid": [f"TRADE_{week}_0"],
            "open_ts_utc": [pd.Timestamp("2025-01-23 09:00:00", tz="UTC")],
            "session": ["EU"],
            "side": ["long"],
        }
    )
    week_dir = _make_week(tmp_path, week, trades, xgb=None)
    out, audit = recover_gate._recover_week(week_dir)
    assert audit["xgb_predictions_present_v1"] is False
    assert audit["match_rate_v1"] == 0.0
    assert (
        out["recovery_status_v1"].iloc[0] == "NOT_RECOVERED_XGB_PARQUET_MISSING"
    )


def test_recover_week_handles_missing_trade_outcomes(tmp_path: Path) -> None:
    week = "TRUTH_MONFRI_WEEK_20250129_20250205"
    week_dir = tmp_path / week
    week_dir.mkdir(parents=True, exist_ok=True)
    out, audit = recover_gate._recover_week(week_dir)
    assert audit["trade_outcomes_present_v1"] is False
    assert audit["trade_count_v1"] == 0
    assert out.empty


def test_recover_week_validates_required_columns(tmp_path: Path) -> None:
    week = "TRUTH_MONFRI_WEEK_20250205_20250212"
    week_dir = tmp_path / week
    week_dir.mkdir(parents=True, exist_ok=True)
    bad_trades = pd.DataFrame({"candidate_uid": ["X"], "trade_uid": ["Y"]})
    bad_trades.to_parquet(
        week_dir / f"trade_outcomes_{week}_MERGED.parquet", index=False
    )
    with pytest.raises(RuntimeError, match="TRADE_OUTCOMES_MISSING_COLUMNS"):
        recover_gate._recover_week(week_dir)


# ---------------------------------------------------------------------------
# go-no-go
# ---------------------------------------------------------------------------


def test_go_no_go_full_coverage_passes() -> None:
    coverage = {
        "match_rate_v1": 1.0,
        "total_trade_count_v1": 1000,
        "matched_trade_count_v1": 1000,
    }
    bridge = {"status_v1": "PASS"}
    status, next_action, _ = recover_gate._go_no_go(coverage, bridge)
    assert status == "RECOVER_ENTRY_SNAPSHOT_SIGNALS_PASS_FULL_COVERAGE_V1"
    assert next_action == "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1"


def test_go_no_go_partial_coverage_passes_when_above_threshold() -> None:
    coverage = {
        "match_rate_v1": 0.9922,
        "total_trade_count_v1": 1914,
        "matched_trade_count_v1": 1899,
    }
    bridge = {"status_v1": "PASS"}
    status, next_action, _ = recover_gate._go_no_go(coverage, bridge)
    assert status == "RECOVER_ENTRY_SNAPSHOT_SIGNALS_PARTIAL_COVERAGE_V1"
    assert next_action == "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1"


def test_go_no_go_blocks_below_threshold() -> None:
    coverage = {
        "match_rate_v1": 0.5,
        "total_trade_count_v1": 100,
        "matched_trade_count_v1": 50,
    }
    bridge = {"status_v1": "PASS"}
    status, next_action, _ = recover_gate._go_no_go(coverage, bridge)
    assert status == "RECOVER_ENTRY_SNAPSHOT_SIGNALS_BLOCKED_LOW_COVERAGE_V1"
    assert next_action == "HOLD_UNTIL_RECOVERY_COVERAGE_RESOLVED_V1"


def test_go_no_go_blocks_when_bridge_math_fails() -> None:
    coverage = {
        "match_rate_v1": 1.0,
        "total_trade_count_v1": 1,
        "matched_trade_count_v1": 1,
    }
    bridge = {"status_v1": "FAIL", "failures_v1": ["P_HAT_OUT_OF_RANGE"]}
    status, next_action, _ = recover_gate._go_no_go(coverage, bridge)
    assert status == "RECOVER_ENTRY_SNAPSHOT_SIGNALS_BLOCKED_LOW_COVERAGE_V1"
    assert next_action == "HOLD_UNTIL_RECOVERY_COVERAGE_RESOLVED_V1"


def test_validate_final_status_rejects_unknown_status() -> None:
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        recover_gate.validate_final_status(
            "MADE_UP", "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1"
        )


def test_validate_final_status_rejects_unknown_next_action() -> None:
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        recover_gate.validate_final_status(
            "RECOVER_ENTRY_SNAPSHOT_SIGNALS_PASS_FULL_COVERAGE_V1", "TRAIN_NOW"
        )


# ---------------------------------------------------------------------------
# Quarantine revival audit (script-text scan)
# ---------------------------------------------------------------------------


def test_validate_no_deprecated_revival_passes_on_self() -> None:
    script_path = (
        Path(recover_gate.__file__).resolve()
    )
    assert recover_gate.validate_no_deprecated_revival(script_path) is True


def test_validate_no_deprecated_revival_detects_quarantine_import(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        '"""x"""\nimport gx1.quarantine.something\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN"):
        recover_gate.validate_no_deprecated_revival(bad)


# ---------------------------------------------------------------------------
# Coverage and bridge audits
# ---------------------------------------------------------------------------


def test_coverage_audit_reports_thresholds_and_missing_xgb() -> None:
    per_week = [
        {
            "week_name_v1": "WEEK_A",
            "trade_count_v1": 10,
            "matched_count_v1": 10,
            "match_rate_v1": 1.0,
            "xgb_predictions_present_v1": True,
        },
        {
            "week_name_v1": "WEEK_B",
            "trade_count_v1": 5,
            "matched_count_v1": 0,
            "match_rate_v1": 0.0,
            "xgb_predictions_present_v1": False,
        },
        {
            "week_name_v1": "WEEK_C",
            "trade_count_v1": 8,
            "matched_count_v1": 7,
            "match_rate_v1": 7 / 8,  # below threshold 0.95
            "xgb_predictions_present_v1": True,
        },
    ]
    audit = recover_gate._coverage_audit(pd.DataFrame(), per_week)
    assert audit["total_trade_count_v1"] == 23
    assert audit["matched_trade_count_v1"] == 17
    assert audit["weeks_missing_xgb_v1"] == ["WEEK_B"]
    low = audit["weeks_below_threshold_v1"]
    assert any(r["week_name_v1"] == "WEEK_C" for r in low)


def test_bridge_audit_passes_on_synthetic_recovered_frame() -> None:
    rec = pd.DataFrame(
        {
            "recovery_status_v1": ["RECOVERED_FROM_XGB_PREDICTIONS"] * 3,
            "p_long_entry_v1": [0.6, 0.1, 0.4],
            "p_hat_entry_v1": [0.6, 0.7, 0.4],
            "uncertainty_entry_v1": [0.4, 0.3, 0.6],
            "margin_entry_v1": [0.3, 0.5, 0.0],
        }
    )
    audit = recover_gate._bridge_audit(rec)
    assert audit["status_v1"] == "PASS"
    assert audit["matched_row_count_v1"] == 3


def test_bridge_audit_fails_when_uncertainty_breaks_invariant() -> None:
    rec = pd.DataFrame(
        {
            "recovery_status_v1": ["RECOVERED_FROM_XGB_PREDICTIONS"],
            "p_long_entry_v1": [0.6],
            "p_hat_entry_v1": [0.6],
            # bug: uncertainty must be 1 - p_hat = 0.4
            "uncertainty_entry_v1": [0.9],
            "margin_entry_v1": [0.3],
        }
    )
    audit = recover_gate._bridge_audit(rec)
    assert audit["status_v1"] == "FAIL"
    assert "UNCERTAINTY_NOT_EQUAL_1_MINUS_P_HAT" in audit["failures_v1"]
