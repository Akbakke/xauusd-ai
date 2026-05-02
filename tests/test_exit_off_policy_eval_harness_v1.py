from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_exit_off_policy_eval_harness_v1 as gate


def test_explicit_artifact_roots_reject_latest() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/EXIT_OFF_POLICY_EVAL_HARNESS_V1_20260429T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])


def test_no_forbidden_actions_default_pass() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"


def test_validate_final_status_only_allowed_pairs() -> None:
    assert gate.validate_final_status(
        "EXIT_OFF_POLICY_EVAL_HARNESS_LOCKED_BASELINE_NUMBERS_AVAILABLE",
        "EXIT_PER_BAR_SANITY_TRAINING_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("ARBITRARY", "EXIT_PER_BAR_SANITY_TRAINING_V1")
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "EXIT_OFF_POLICY_EVAL_HARNESS_LOCKED_BASELINE_NUMBERS_AVAILABLE",
            "TRAIN_PRODUCTION_NOW_V1",
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


def _make_synthetic_per_bar() -> pd.DataFrame:
    """3 trades x 5 bars each, action=HOLD only (per-bar view)."""
    rows = []
    for trade_id, base_pnl in enumerate([20.0, -10.0, 50.0]):
        for bar in range(5):
            running_pnl = base_pnl + bar * 5.0
            running_mfe = max(running_pnl, base_pnl + 30.0)
            running_mae = -10.0 - bar * 2.0
            rows.append(
                {
                    "candidate_uid_v1": f"trade_{trade_id}",
                    "bars_held_v1": bar,
                    "running_pnl_at_close_bps_v1": running_pnl,
                    "running_mfe_bps_v1": running_mfe,
                    "running_mae_bps_v1": running_mae,
                    "running_giveback_from_peak_bps_v1": max(running_mfe - running_pnl, 0),
                    "exit_prob_v1": 0.1 + 0.15 * bar,
                    "primary_split_v1": "train",
                }
            )
    return pd.DataFrame(rows)


def test_realized_exit_picks_last_bar_per_trade() -> None:
    per_bar = _make_synthetic_per_bar()
    indices = gate._exit_index_realized_exit(per_bar)
    selected = per_bar.loc[indices.values]
    # Each selected row should have bars_held_v1 == 4 (last bar)
    assert (selected["bars_held_v1"].values == 4).all()
    assert len(selected) == 3


def test_always_exit_now_bar_0_picks_bar_0() -> None:
    per_bar = _make_synthetic_per_bar()
    indices = gate._exit_index_always_exit_now_bar_0(per_bar)
    selected = per_bar.loc[indices.values]
    assert (selected["bars_held_v1"].values == 0).all()


def test_peak_mfe_oracle_picks_max_mfe_bar() -> None:
    per_bar = _make_synthetic_per_bar()
    indices = gate._exit_index_peak_mfe_oracle(per_bar)
    selected = per_bar.loc[indices.values]
    # In synthetic data, running_mfe is max at bar 4 because pnl grows
    assert (selected["bars_held_v1"].values >= 0).all()
    # Peak mfe per trade
    for trade_id, group in per_bar.groupby("candidate_uid_v1"):
        sel_uid = selected[selected["candidate_uid_v1"] == trade_id].iloc[0]
        max_mfe_in_trade = group["running_mfe_bps_v1"].max()
        assert sel_uid["running_mfe_bps_v1"] == max_mfe_in_trade


def test_trail_stop_falls_back_to_realized_when_not_triggered() -> None:
    per_bar = _make_synthetic_per_bar()
    indices = gate._exit_index_trail_stop_25_pct_dd(per_bar)
    # Each trade should produce one exit index
    assert len(indices) == 3


def test_exit_prob_threshold_falls_back_when_no_qualifying_bar() -> None:
    per_bar = _make_synthetic_per_bar()
    # Threshold above all values forces fallback
    indices = gate._exit_index_exit_prob_threshold(per_bar, threshold=0.99)
    selected = per_bar.loc[indices.values]
    # All should be at last bar (realized exit fallback)
    assert (selected["bars_held_v1"].values == 4).all()


def test_evaluate_policy_computes_eight_metrics() -> None:
    per_bar = _make_synthetic_per_bar()
    indices = gate._exit_index_realized_exit(per_bar)
    metrics = gate.evaluate_policy(
        per_bar, indices, policy_id="TEST", split="train"
    )
    for key in [
        "trade_count_v1",
        "total_realized_pnl_bps_v1",
        "mean_realized_pnl_bps_v1",
        "mean_mfe_capture_ratio_v1",
        "mean_mae_burden_bps_v1",
        "mean_giveback_bps_v1",
        "cata_proxy_rate_v1",
        "mean_bars_to_exit_v1",
    ]:
        assert key in metrics


def test_audit_baseline_sanity_passes_on_consistent_results() -> None:
    realized = {
        "policy_id_v1": "REALIZED_EXIT_BASELINE",
        "total_realized_pnl_bps_v1": 100.0,
        "mean_realized_pnl_bps_v1": 5.0,
    }
    always_hold = {
        "policy_id_v1": "ALWAYS_HOLD_TO_REALIZED_END",
        "total_realized_pnl_bps_v1": 100.0,
        "mean_realized_pnl_bps_v1": 5.0,
    }
    bar0 = {
        "policy_id_v1": "ALWAYS_EXIT_NOW_AT_BAR_0",
        "total_realized_pnl_bps_v1": 0.0,
        "mean_realized_pnl_bps_v1": 0.0,
    }
    oracle = {
        "policy_id_v1": "PEAK_MFE_ORACLE",
        "total_realized_pnl_bps_v1": 200.0,
        "mean_realized_pnl_bps_v1": 10.0,
    }
    audit = gate.audit_baseline_sanity(
        {"train": [realized, always_hold, bar0, oracle]}
    )
    assert audit["status_v1"] == "PASS"


def test_audit_baseline_sanity_fails_on_oracle_dominance_violation() -> None:
    realized = {
        "policy_id_v1": "REALIZED_EXIT_BASELINE",
        "total_realized_pnl_bps_v1": 200.0,
        "mean_realized_pnl_bps_v1": 10.0,
    }
    always_hold = {
        "policy_id_v1": "ALWAYS_HOLD_TO_REALIZED_END",
        "total_realized_pnl_bps_v1": 200.0,
        "mean_realized_pnl_bps_v1": 10.0,
    }
    bar0 = {
        "policy_id_v1": "ALWAYS_EXIT_NOW_AT_BAR_0",
        "total_realized_pnl_bps_v1": 0.0,
        "mean_realized_pnl_bps_v1": 0.0,
    }
    oracle = {
        "policy_id_v1": "PEAK_MFE_ORACLE",
        "total_realized_pnl_bps_v1": 100.0,  # Violates oracle dominance
        "mean_realized_pnl_bps_v1": 5.0,
    }
    with pytest.raises(RuntimeError, match="BASELINE_SANITY_AUDIT_FAILED"):
        gate.audit_baseline_sanity(
            {"train": [realized, always_hold, bar0, oracle]}
        )


def test_audit_eval_state_leakage_check_passes_on_clean_fields() -> None:
    audit = gate.audit_eval_state_leakage_check(pd.DataFrame())
    assert audit["status_v1"] == "PASS"


def test_audit_eval_split_only_uses_split_data_passes_on_clean_partition() -> None:
    df = _make_synthetic_per_bar()
    audit = gate.audit_eval_split_only_uses_split_data(df)
    assert audit["status_v1"] == "PASS"


def test_audit_eval_split_fails_on_spanning_trade() -> None:
    df = _make_synthetic_per_bar()
    df.loc[df["candidate_uid_v1"] == "trade_0", "primary_split_v1"] = "val"
    df.loc[df.index[0], "primary_split_v1"] = "train"
    with pytest.raises(RuntimeError, match="EVAL_SPLIT_PARTITION_VIOLATION"):
        gate.audit_eval_split_only_uses_split_data(df)


def test_baseline_definitions_match_apply_baseline() -> None:
    per_bar = _make_synthetic_per_bar()
    for b in gate.BASELINE_DEFINITIONS:
        bid = b["baseline_id_v1"]
        indices = gate._apply_baseline(per_bar, bid)
        assert len(indices) == 3, f"baseline {bid} should produce 3 exit indices"


def test_apply_baseline_unknown_id_raises() -> None:
    per_bar = _make_synthetic_per_bar()
    with pytest.raises(RuntimeError, match="UNKNOWN_BASELINE_ID"):
        gate._apply_baseline(per_bar, "BOGUS")


def test_write_artifacts_produces_required_outputs(tmp_path: Path) -> None:
    out_root = tmp_path / "EXIT_OFF_POLICY_EVAL_HARNESS_V1_20260429T999999Z_LOCK"
    result = gate.write_artifacts(out_root=out_root, built_at_utc="20260429T999999Z")
    artifact_root = Path(result["artifact_root"])
    assert artifact_root == out_root
    for required in [
        "manifest_v1.json",
        "summary_v1.json",
        "status_v1.json",
        "report_v1.md",
        "exit_off_policy_eval_harness_go_no_go_v1.json",
        "input_manifest_v1.json",
        "baseline_definitions_v1.json",
        "metric_definitions_v1.json",
        "baseline_metrics_per_split_v1.json",
        "baseline_metrics_per_split_v1.csv",
        "eval_harness_audits_v1.json",
        "reproducibility_audit_v1.json",
    ]:
        assert (artifact_root / required).exists(), f"missing {required}"
    summary = json.loads((artifact_root / "summary_v1.json").read_text())
    assert summary["final_status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert summary["next_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert summary["training_blocked_v1"] is True
    assert summary["baseline_count_v1"] == 6
    assert summary["metric_count_v1"] == 8
    audits = json.loads((artifact_root / "eval_harness_audits_v1.json").read_text())
    for a in audits["audits_v1"]:
        assert a["status_v1"] == "PASS", f"audit failed: {a}"
