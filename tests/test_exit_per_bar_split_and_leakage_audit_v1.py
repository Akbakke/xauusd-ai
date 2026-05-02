from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_exit_per_bar_split_and_leakage_audit_v1 as gate


def test_explicit_artifact_roots_reject_latest() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1_20260429T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])


def test_no_forbidden_actions_default_pass() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"


def test_validate_final_status_only_allowed_pairs() -> None:
    assert gate.validate_final_status(
        "EXIT_PER_BAR_SPLIT_LOCKED_LEAKAGE_AUDIT_PASSED",
        "EXIT_OFF_POLICY_EVAL_HARNESS_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status("ARBITRARY", "EXIT_OFF_POLICY_EVAL_HARNESS_V1")
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "EXIT_PER_BAR_SPLIT_LOCKED_LEAKAGE_AUDIT_PASSED",
            "TRAIN_NOW_V1",
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


def test_validate_split_proportions_must_sum_to_one() -> None:
    assert gate.validate_split_proportions(0.7, 0.15, 0.15)
    with pytest.raises(RuntimeError, match="SPLIT_PROPORTIONS_DO_NOT_SUM_TO_1"):
        gate.validate_split_proportions(0.5, 0.3, 0.3)
    with pytest.raises(RuntimeError, match="SPLIT_PROPORTIONS_MUST_BE_POSITIVE"):
        gate.validate_split_proportions(0.7, 0.3, 0.0)


def _make_synthetic_augmented(n_trades: int = 20) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for i in range(n_trades):
        n_bars = int(rng.integers(3, 8))
        trade_open = pd.Timestamp("2025-06-01 13:00", tz="UTC") + pd.Timedelta(days=i)
        for bar in range(n_bars):
            for action_id, action_label in [(0, "HOLD"), (1, "EXIT_NOW")]:
                rows.append(
                    {
                        "candidate_uid_v1": f"TRUTH_MONFRI_WEEK_2025_w{i:03d}:0:cand:{i:03d}",
                        "trade_uid_v1": f"t{i}",
                        "trade_id": f"SIM-{i}",
                        "bars_held_v1": bar,
                        "ts_v1": trade_open + pd.Timedelta(minutes=5 * bar),
                        "is_terminal_for_action_v1": action_id == 1 or bar == n_bars - 1,
                        "action_id_v1": action_id,
                        "action_label_v1": action_label,
                        "behavior_propensity_v1": (
                            "LOGGED_EXIT_NOW_PROPENSITY_1"
                            if action_id == 1 and bar == n_bars - 1
                            else "COUNTERFACTUAL_EXIT_NOW_NO_PROPENSITY"
                            if action_id == 1
                            else "FORCED_TERMINAL_HOLD_DATA_LIMIT"
                            if bar == n_bars - 1
                            else "LOGGED_HOLD_PROPENSITY_1"
                        ),
                        "running_pnl_at_close_bps_v1": float(bar * 10 + i),
                        "running_mfe_bps_v1": float(bar * 12 + i),
                        "running_mae_bps_v1": float(-(bar + 1) * 5),
                        "exit_prob_v1": 0.1 + 0.1 * bar,
                        "atr_bps_now_v1": 5.0,
                        "session_id_v1": 1,
                        "row_id_per_bar_v1": i * 10 + bar,
                        "next_row_id_per_bar_v1": (i * 10 + bar + 1) if bar < n_bars - 1 and action_id == 0 else None,
                        "reward_realized_pnl_reward_v1": float(bar * 10 + i)
                        if action_id == 1 or bar == n_bars - 1
                        else 0.0,
                        "reward_mfe_capture_reward_v1": 0.5,
                        "reward_mae_penalty_reward_v1": 0.0,
                        "reward_giveback_penalty_reward_v1": 0.0,
                        "reward_transparent_combined_reward_v1": 0.0,
                    }
                )
    return pd.DataFrame(rows)


def test_assign_time_order_per_trade_split_partitions_trades() -> None:
    df = _make_synthetic_augmented(20)
    out = gate._assign_time_order_per_trade_split(df)
    assert "primary_split_v1" in out.columns
    counts_per_split = out.groupby("primary_split_v1")["candidate_uid_v1"].nunique()
    assert counts_per_split["train"] == 14
    assert counts_per_split["val"] == 3
    assert counts_per_split["test"] == 3
    # Each candidate appears in exactly one split
    per_uid = out.groupby("candidate_uid_v1")["primary_split_v1"].nunique()
    assert (per_uid == 1).all()


def test_assign_week_block_split_partitions_weeks() -> None:
    df = _make_synthetic_augmented(20)
    out = gate._assign_week_block_split(df)
    assert "sensitivity_week_split_v1" in out.columns
    week_counts = out.groupby("sensitivity_week_split_v1")["source_week_v1"].nunique()
    assert week_counts.sum() >= 3


def test_audit_intra_trade_integrity_passes_on_clean_split() -> None:
    df = _make_synthetic_augmented(20)
    df = gate._assign_time_order_per_trade_split(df)
    audit = gate.audit_intra_trade_integrity(df, "primary_split_v1")
    assert audit["status_v1"] == "PASS"
    assert audit["spanning_trade_count_v1"] == 0
    # break it deliberately
    bad = df.copy()
    bad.loc[0, "primary_split_v1"] = "test"
    bad.loc[1, "primary_split_v1"] = "train"
    bad.loc[0, "candidate_uid_v1"] = "u_shared"
    bad.loc[1, "candidate_uid_v1"] = "u_shared"
    with pytest.raises(RuntimeError, match="INTRA_TRADE_LEAKAGE"):
        gate.audit_intra_trade_integrity(bad, "primary_split_v1")


def test_audit_temporal_non_overlap_passes_on_time_ordered() -> None:
    df = _make_synthetic_augmented(20)
    df = gate._assign_time_order_per_trade_split(df)
    audit = gate.audit_temporal_non_overlap(df, "primary_split_v1")
    assert audit["status_v1"] == "PASS"


def test_audit_state_no_shortcut_recheck_blocks_forbidden() -> None:
    audit = gate.audit_state_no_shortcut_recheck(["running_pnl_at_close_bps_v1", "exit_prob_v1"])
    assert audit["status_v1"] == "PASS"
    with pytest.raises(RuntimeError, match="STATE_NO_SHORTCUT_RECHECK_FAIL"):
        gate.audit_state_no_shortcut_recheck(["pnl_bps", "exit_reason"])


def test_audit_reward_input_not_in_state_blocks_leak() -> None:
    audit = gate.audit_reward_input_not_in_state(["running_pnl_at_close_bps_v1"])
    assert audit["status_v1"] == "PASS"
    with pytest.raises(RuntimeError, match="REWARD_INPUT_LEAK_INTO_STATE"):
        gate.audit_reward_input_not_in_state(["pnl_bps", "running_pnl_at_close_bps_v1"])


def test_audit_action_balance_per_split_pass_on_balanced() -> None:
    df = _make_synthetic_augmented(20)
    df = gate._assign_time_order_per_trade_split(df)
    audit = gate.audit_action_balance_per_split(df, "primary_split_v1")
    assert audit["status_v1"] == "PASS"


def test_audit_action_balance_fails_on_imbalanced() -> None:
    df = _make_synthetic_augmented(20)
    df = gate._assign_time_order_per_trade_split(df)
    # Drop some EXIT_NOW rows to break balance
    bad = df[~((df["action_id_v1"] == 1) & (df.index < 5))].copy()
    with pytest.raises(RuntimeError, match="ACTION_BALANCE_FAIL_FOR_SPLIT"):
        gate.audit_action_balance_per_split(bad, "primary_split_v1")


def test_audit_propensity_distribution_blocks_degenerate_split() -> None:
    df = _make_synthetic_augmented(20)
    df = gate._assign_time_order_per_trade_split(df)
    # Force a split to lose one propensity label
    bad = df.copy()
    bad.loc[
        bad["behavior_propensity_v1"] == "LOGGED_EXIT_NOW_PROPENSITY_1",
        "behavior_propensity_v1",
    ] = "COUNTERFACTUAL_EXIT_NOW_NO_PROPENSITY"
    with pytest.raises(RuntimeError, match="PROPENSITY_DEGENERATE_SPLIT"):
        gate.audit_propensity_distribution(bad, "primary_split_v1")


def test_audit_next_row_pointer_cross_split_pass_on_clean_split() -> None:
    df = _make_synthetic_augmented(20)
    df = gate._assign_time_order_per_trade_split(df)
    audit = gate.audit_next_row_pointer_cross_split(df, "primary_split_v1")
    assert audit["status_v1"] in ("PASS", "PASS_TRIVIAL")


def test_state_columns_excludes_metadata_and_action_and_reward() -> None:
    df = _make_synthetic_augmented(20)
    df = gate._assign_time_order_per_trade_split(df)
    df = gate._assign_week_block_split(df)
    state_cols = gate._state_columns(df)
    assert "running_pnl_at_close_bps_v1" in state_cols
    assert "exit_prob_v1" in state_cols
    assert "action_id_v1" not in state_cols
    assert "primary_split_v1" not in state_cols
    assert "behavior_propensity_v1" not in state_cols
    assert all(not c.startswith("reward_") for c in state_cols)


def test_write_artifacts_produces_required_outputs(tmp_path: Path) -> None:
    out_root = tmp_path / "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1_20260429T999999Z_LOCK"
    result = gate.write_artifacts(out_root=out_root, built_at_utc="20260429T999999Z")
    artifact_root = Path(result["artifact_root"])
    assert artifact_root == out_root
    for required in [
        "manifest_v1.json",
        "summary_v1.json",
        "status_v1.json",
        "report_v1.md",
        "exit_per_bar_split_and_leakage_audit_go_no_go_v1.json",
        "input_manifest_v1.json",
        "leakage_audits_v1.json",
        "primary_split_summary_v1.json",
        "primary_split_summary_v1.csv",
        "sensitivity_week_split_summary_v1.json",
        "split_locked_augmented_dataset_v1.parquet",
        "primary_split_train_v1.parquet",
        "primary_split_val_v1.parquet",
        "primary_split_test_v1.parquet",
        "reproducibility_audit_v1.json",
    ]:
        assert (artifact_root / required).exists(), f"missing {required}"
    summary = json.loads((artifact_root / "summary_v1.json").read_text())
    assert summary["final_status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert summary["next_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert summary["training_blocked_v1"] is True
    leakage = json.loads((artifact_root / "leakage_audits_v1.json").read_text())
    for a in leakage["audits_v1"]:
        assert a["status_v1"] in ("PASS", "PASS_TRIVIAL"), f"audit failed: {a}"
