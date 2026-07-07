from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import (
    materialize_exit_quality_diagnostic_and_per_bar_decision_scaffold_v1 as gate,
)


def test_explicit_artifact_roots_reject_latest_and_glob() -> None:
    assert gate.validate_explicit_artifact_roots(
        [
            Path(
                "/tmp/EXIT_QUALITY_DIAGNOSTIC_AND_PER_BAR_DECISION_SCAFFOLD_V1_20260429T000000Z_LOCK"
            )
        ]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])


def test_no_forbidden_actions_default_pass() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"
    blocked = gate.validate_no_forbidden_actions(adapter=True, r6=True)
    assert blocked["status_v1"] == "FAIL"


def test_validate_final_status_only_allowed_pairs() -> None:
    assert gate.validate_final_status(
        "EXIT_QUALITY_DIAGNOSTIC_PASS_FULL_RECONSTRUCTION_AND_COUNTERFACTUALS_AVAILABLE",
        "TRAIN_EXIT_BANDIT_HOLD_EXIT_NOW_RESEARCH_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "ARBITRARY",
            "TRAIN_EXIT_BANDIT_HOLD_EXIT_NOW_RESEARCH_V1",
        )
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "EXIT_QUALITY_DIAGNOSTIC_PASS_FULL_RECONSTRUCTION_AND_COUNTERFACTUALS_AVAILABLE",
            "OPEN_R6_NOW",
        )


def test_validate_no_deprecated_revival_blocks_quarantine_imports(tmp_path: Path) -> None:
    bad = tmp_path / "imports_quarantine.py"
    bad.write_text(
        "from gx1.quarantine._DEPRECATED_SCRIPTS_20260219 import x\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN"):
        gate.validate_no_deprecated_revival(bad)
    good = tmp_path / "clean.py"
    good.write_text("import pandas as pd\n", encoding="utf-8")
    assert gate.validate_no_deprecated_revival(good)
    assert gate.validate_no_deprecated_revival(Path(gate.__file__))


def _make_synthetic_trades_and_m5():
    t0 = pd.Timestamp("2025-06-02 13:00:00", tz="UTC")
    bar_minutes = 5
    n_bars = 30
    times = [t0 + pd.Timedelta(minutes=bar_minutes * i) for i in range(n_bars)]
    rng = np.random.default_rng(7)
    base = 2700.0 + np.cumsum(rng.normal(0, 0.5, n_bars))
    m5 = pd.DataFrame(
        {
            "time": times,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.2,
            "volume": np.full(n_bars, 100.0),
        }
    )
    trades = pd.DataFrame(
        [
            {
                "candidate_uid": "cuid_001",
                "trade_uid": "tuid_001",
                "open_ts_utc": times[2],
                "close_ts_utc": times[7],
                "side": "long",
                "entry_price_used": float(m5.loc[2, "close"]),
                "exit_price_used": float(m5.loc[7, "close"]),
                "pnl_bps": 30.0,
                "mae_bps": -10.0,
                "mfe_bps": 50.0,
                "exit_reason": "THRESHOLD",
            },
            {
                "candidate_uid": "cuid_002",
                "trade_uid": "tuid_002",
                "open_ts_utc": times[10],
                "close_ts_utc": times[15],
                "side": "long",
                "entry_price_used": float(m5.loc[10, "close"]),
                "exit_price_used": float(m5.loc[15, "close"]),
                "pnl_bps": -45.0,
                "mae_bps": -60.0,
                "mfe_bps": 20.0,
                "exit_reason": "CATASTROPHIC_GUARD",
            },
            {
                "candidate_uid": "cuid_003",
                "trade_uid": "tuid_003",
                "open_ts_utc": times[20],
                "close_ts_utc": times[24],
                "side": "long",
                "entry_price_used": float(m5.loc[20, "close"]),
                "exit_price_used": float(m5.loc[24, "close"]),
                "pnl_bps": -350.0,
                "mae_bps": -400.0,
                "mfe_bps": 5.0,
                "exit_reason": "POLICY_FRIDAY_FLAT",
            },
        ]
    )
    return trades, m5


def test_per_bar_reconstruction_emits_one_row_per_bar_with_terminal_exit_now() -> None:
    trades, m5 = _make_synthetic_trades_and_m5()
    decisions, summary = gate._reconstruct_per_bar_trajectories(trades, m5)
    assert summary["reconstructed_trade_count_v1"] == 3
    assert summary["trades_in_m5_range_v1"] == 3
    assert summary["trades_out_of_m5_range_v1"] == 0
    assert summary["decision_row_count_v1"] > 0
    assert summary["exit_now_action_count_v1"] == 3  # one per trade
    assert summary["hold_action_count_v1"] == summary["decision_row_count_v1"] - 3
    # Each trade has terminal == True only on last bar
    for uid, group in decisions.groupby("candidate_uid_v1"):
        assert group["is_terminal_v1"].sum() == 1
        assert group["is_terminal_v1"].iloc[-1]


def test_giveback_ladder_includes_actual_and_levels_and_trail() -> None:
    trades = pd.DataFrame(
        {
            "pnl_bps": [10.0, -50.0, 80.0, -200.0, 0.0],
            "mfe_bps": [60.0, 5.0, 100.0, 50.0, 0.0],
            "mae_bps": [-10.0, -55.0, -5.0, -210.0, -2.0],
        }
    )
    table, summary = gate._giveback_ladder_counterfactual(trades)
    assert summary["actual_realized_pnl_bps_v1"] == -160.0
    scenarios = set(table["scenario_v1"])
    assert "ACTUAL_REALIZED" in scenarios
    for level in gate.GIVEBACK_LADDER_LEVELS:
        assert f"EXIT_AT_{int(level*100)}PCT_MFE" in scenarios
    assert any("TRAIL_EXIT_AT_PEAK_MINUS" in s for s in scenarios)


def test_cata_prevention_counterfactual_filters_correctly() -> None:
    trades = pd.DataFrame(
        {
            "candidate_uid": ["a", "b", "c"],
            "exit_reason": ["CATASTROPHIC_GUARD", "THRESHOLD", "CATASTROPHIC_GUARD"],
            "pnl_bps": [-40.0, 50.0, -100.0],
            "mfe_bps": [20.0, 60.0, 0.0],
            "mae_bps": [-50.0, -10.0, -110.0],
        }
    )
    table, summary = gate._cata_prevention_counterfactual(trades)
    assert summary["cata_count_v1"] == 2
    assert summary["cata_with_positive_mfe_window_v1"] == 1
    assert summary["actual_cata_total_pnl_bps_v1"] == -140.0
    assert summary["counterfactual_peak_mfe_total_pnl_bps_v1"] == 20.0
    assert summary["upper_bound_savings_bps_v1"] == pytest.approx(160.0)
    assert summary["mean_savings_per_cata_bps_v1"] == pytest.approx(80.0)
    assert len(table) == 2


def test_friday_flat_refinement_with_synthetic_monday_lookup() -> None:
    trades, m5 = _make_synthetic_trades_and_m5()
    table, summary = gate._friday_flat_refinement(trades, m5)
    assert summary["friday_flat_count_v1"] == 1
    assert "refined_policy_total_pnl_bps_v1" in summary
    assert summary["monday_open_lookup_available_v1"] >= 1
    assert len(table) == 1


def test_samstemte_feature_audit_returns_diff_table_and_summary() -> None:
    state_contract = {
        "rows_v1": [
            {"field_name_v1": "candidate_score_v1", "allowed_as_state_v1": True},
            {"field_name_v1": "signal_r5_bad_score_v1", "allowed_as_state_v1": True},
            {"field_name_v1": "bad_label_v1", "allowed_as_state_v1": False},
        ]
    }
    table, audit = gate._samstemte_feature_audit(state_contract)
    assert audit["exit_transformer_v1"]["feature_count_v1"] == 53
    assert audit["iql_state_v1"]["feature_count_v1"] == 2
    assert audit["samstemte_status_v1"] == "FEATURE_SETS_DIVERGE_NEED_HUB_DESIGN"
    assert "in_exit_transformer_v1" in table.columns
    assert "in_iql_state_v1" in table.columns


def test_go_no_go_handles_completeness_thresholds() -> None:
    base_per_bar = {"reconstruction_completeness_v1": 0.99, "decision_row_count_v1": 1000}
    base_friday = {"friday_flat_count_v1": 50, "monday_open_lookup_available_v1": 50}
    base_samstemte = {"samstemte_status_v1": "FEATURE_SETS_DIVERGE_NEED_HUB_DESIGN"}
    status, _, _ = gate._go_no_go(base_per_bar, {}, {}, base_friday, base_samstemte)
    assert status.startswith("EXIT_QUALITY_DIAGNOSTIC_PASS_FULL_RECONSTRUCTION")
    base_per_bar["reconstruction_completeness_v1"] = 0.40
    status, action, _ = gate._go_no_go(base_per_bar, {}, {}, base_friday, base_samstemte)
    assert status == "EXIT_QUALITY_DIAGNOSTIC_BLOCKED_BY_M5_GAP"
    base_per_bar["reconstruction_completeness_v1"] = 0.80
    status, action, _ = gate._go_no_go(base_per_bar, {}, {}, base_friday, base_samstemte)
    assert status == "EXIT_QUALITY_DIAGNOSTIC_PARTIAL_RECONSTRUCTION_GAP"


def test_write_artifacts_produces_required_outputs(tmp_path: Path) -> None:
    out_root = tmp_path / "EXIT_QUALITY_DIAGNOSTIC_AND_PER_BAR_DECISION_SCAFFOLD_V1_20260429T999999Z_LOCK"
    result = gate.write_artifacts(out_root=out_root, built_at_utc="20260429T999999Z")
    artifact_root = Path(result["artifact_root"])
    assert artifact_root == out_root
    for required in [
        "manifest_v1.json",
        "summary_v1.json",
        "status_v1.json",
        "report_v1.md",
        "exit_quality_diagnostic_and_per_bar_decision_scaffold_go_no_go_v1.json",
        "input_manifest_v1.json",
        "reproducibility_audit_v1.json",
    ]:
        assert (artifact_root / required).exists(), f"missing {required}"
    for sub in ("PER_BAR_TRAJECTORY_V1", "GIVEBACK_LADDER_V1", "CATA_PREVENTION_V1", "FRIDAY_FLAT_REFINEMENT_V1", "SAMSTEMTE_FEATURE_AUDIT_V1"):
        assert (artifact_root / sub).is_dir()
    summary = json.loads((artifact_root / "summary_v1.json").read_text())
    assert summary["row_count_invariant_v1"] is True
    assert summary["final_status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert summary["next_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert summary["trade_count_v1"] == gate.EXPECTED_FRAME_ROWS
    assert summary["per_bar_reconstruction_v1"]["completeness_v1"] >= 0.0
    assert summary["samstemte_feature_audit_v1"]["samstemte_status_v1"] in {
        "FEATURE_SETS_DIVERGE_NEED_HUB_DESIGN",
        "NOT_ESTABLISHED",
    }
