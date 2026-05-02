from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts import materialize_exit_action_support_augment_v1 as gate


def test_explicit_artifact_roots_reject_latest() -> None:
    assert gate.validate_explicit_artifact_roots(
        [Path("/tmp/EXIT_ACTION_SUPPORT_AUGMENT_V1_20260429T000000Z_LOCK")]
    )
    with pytest.raises(RuntimeError, match="IMPLICIT_LATEST_GLOB"):
        gate.validate_explicit_artifact_roots([Path("/tmp/latest")])


def test_no_forbidden_actions_default_pass() -> None:
    assert gate.validate_no_forbidden_actions()["status_v1"] == "PASS"


def test_validate_final_status_only_allowed_pairs() -> None:
    assert gate.validate_final_status(
        "EXIT_ACTION_SUPPORT_AUGMENT_LOCKED_DATASET_READY",
        "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1",
    )
    with pytest.raises(RuntimeError, match="FINAL_STATUS_NOT_ALLOWED"):
        gate.validate_final_status(
            "ARBITRARY",
            "EXIT_PER_BAR_SPLIT_AND_LEAKAGE_AUDIT_V1",
        )
    with pytest.raises(RuntimeError, match="NEXT_ACTION_NOT_ALLOWED"):
        gate.validate_final_status(
            "EXIT_ACTION_SUPPORT_AUGMENT_LOCKED_DATASET_READY",
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


def test_validate_state_no_shortcut_blocks_forbidden() -> None:
    assert gate.validate_state_no_shortcut(["running_pnl_at_close_bps_v1", "exit_prob_v1"])
    with pytest.raises(RuntimeError, match="FORBIDDEN_STATE_FIELD_IN_AUGMENTED_COLUMNS"):
        gate.validate_state_no_shortcut(["pnl_bps", "exit_reason"])


def test_validate_action_distribution_requires_balance() -> None:
    actions = pd.Series([0, 0, 1, 1])
    assert gate.validate_action_distribution(actions)
    with pytest.raises(RuntimeError, match="ACTION_VALUES_NOT_BINARY"):
        gate.validate_action_distribution(pd.Series([0, 1, 2]))
    with pytest.raises(RuntimeError, match="AUGMENTED_ACTION_COUNTS_MISMATCH"):
        gate.validate_action_distribution(pd.Series([0, 0, 0, 1]))


def test_compute_terminal_reward_at_bar_matches_specs() -> None:
    pnl = pd.Series([50.0, -20.0, 80.0])
    mfe = pd.Series([60.0, 10.0, 100.0])
    mae = pd.Series([-10.0, -50.0, -5.0])
    assert np.allclose(
        gate._compute_terminal_reward_at_bar(pnl, mfe, mae, "REALIZED_PNL_REWARD"),
        [50.0, -20.0, 80.0],
    )
    mfe_capture = gate._compute_terminal_reward_at_bar(pnl, mfe, mae, "MFE_CAPTURE_REWARD")
    assert np.allclose(mfe_capture, [50 / 60, -2.0, 80 / 100])
    assert np.allclose(
        gate._compute_terminal_reward_at_bar(pnl, mfe, mae, "MAE_PENALTY_REWARD"),
        [50 - 5, -20 - 25, 80 - 2.5],
    )
    assert np.allclose(
        gate._compute_terminal_reward_at_bar(pnl, mfe, mae, "GIVEBACK_PENALTY_REWARD"),
        [-10, -30, -20],
    )
    combined = gate._compute_terminal_reward_at_bar(
        pnl, mfe, mae, "TRANSPARENT_COMBINED_REWARD"
    )
    expected = [
        50 - 0.25 * 10 - 0.25 * max(60 - 50, 0),
        -20 - 0.25 * 50 - 0.25 * max(10 - (-20), 0),
        80 - 0.25 * 5 - 0.25 * max(100 - 80, 0),
    ]
    assert np.allclose(combined, expected)


def test_compute_terminal_reward_unknown_variant_raises() -> None:
    with pytest.raises(RuntimeError, match="UNKNOWN_REWARD_VARIANT"):
        gate._compute_terminal_reward_at_bar(
            pd.Series([1.0]), pd.Series([1.0]), pd.Series([-1.0]), "BOGUS"
        )


def _make_synthetic_state(n_bars: int = 5) -> pd.DataFrame:
    """One synthetic trade with n_bars bars."""
    return pd.DataFrame(
        {
            "candidate_uid_v1": ["c1"] * n_bars,
            "trade_uid_v1": ["t1"] * n_bars,
            "bars_held_v1": list(range(n_bars)),
            "is_terminal_v1": [False] * (n_bars - 1) + [True],
            "side_v1": ["long"] * n_bars,
            "ts_v1": pd.date_range("2025-06-02 13:00", periods=n_bars, freq="5min", tz="UTC"),
            "running_pnl_at_close_bps_v1": [10.0, 20.0, 35.0, 50.0, 30.0],
            "running_mfe_bps_v1": [10.0, 20.0, 35.0, 60.0, 60.0],
            "running_mae_bps_v1": [-2.0, -2.0, -2.0, -2.0, -5.0],
            "running_giveback_from_peak_bps_v1": [0.0, 0.0, 0.0, 10.0, 30.0],
            "trade_id": ["SIM-1"] * n_bars,
            "entry_session_v1": ["EU"] * n_bars,
            "entry_spread_bps_v1": [3.0] * n_bars,
            "exit_prob_v1": [0.1, 0.2, 0.3, 0.5, 0.7],
            "distance_from_peak_mfe_bps_v1": [0.0, 0.0, 0.0, 0.0, 30.0],
            "time_since_mfe_bars_v1": [0, 0, 0, 0, 1],
            "giveback_ratio_v1": [0.0, 0.0, 0.0, 0.0, 0.5],
            "atr_bps_now_v1": [4.0] * n_bars,
            "session_id_v1": [1] * n_bars,
            "vol_regime_id_v1": [1] * n_bars,
            "trend_slope_ema3_v1": [0.1] * n_bars,
            "spread_bps_dyn_v1": [3.0] * n_bars,
            "minutes_since_session_open_v1": [60, 65, 70, 75, 80],
            "entry_price_v1": [2700.0] * n_bars,
            "bar_close_v1": [2700.27, 2700.54, 2700.945, 2701.35, 2700.81],
            "bar_count_v1": [n_bars] * n_bars,
        }
    )


def test_augment_with_action_pairs_doubles_rows_and_assigns_actions() -> None:
    state = _make_synthetic_state(5)
    augmented = gate._augment_with_action_pairs(state)
    assert len(augmented) == 2 * len(state)
    counts = augmented["action_id_v1"].value_counts().to_dict()
    assert counts[gate.ACTION_HOLD_ID] == 5
    assert counts[gate.ACTION_EXIT_NOW_ID] == 5
    # HOLD at non-terminal bar must have reward = 0 (REALIZED_PNL flavor)
    hold_non_terminal = augmented[
        (augmented["action_id_v1"] == gate.ACTION_HOLD_ID)
        & (augmented["bars_held_v1"] < 4)
    ]
    assert (hold_non_terminal["reward_realized_pnl_reward_v1"] == 0.0).all()
    # HOLD at terminal bar = forced terminal hold = realized pnl
    hold_terminal = augmented[
        (augmented["action_id_v1"] == gate.ACTION_HOLD_ID)
        & (augmented["bars_held_v1"] == 4)
    ]
    assert hold_terminal["reward_realized_pnl_reward_v1"].iloc[0] == 30.0
    # EXIT_NOW at any bar = pnl at that bar's close
    exit_now = augmented[augmented["action_id_v1"] == gate.ACTION_EXIT_NOW_ID]
    expected_pnl_at_close = [10.0, 20.0, 35.0, 50.0, 30.0]
    assert list(exit_now["reward_realized_pnl_reward_v1"].values) == expected_pnl_at_close


def test_augment_propensity_labels_distinguish_logged_vs_counterfactual() -> None:
    state = _make_synthetic_state(5)
    augmented = gate._augment_with_action_pairs(state)
    propensity = augmented["behavior_propensity_v1"].value_counts().to_dict()
    assert propensity.get("LOGGED_HOLD_PROPENSITY_1") == 4  # 4 non-terminal HOLD bars
    assert propensity.get("FORCED_TERMINAL_HOLD_DATA_LIMIT") == 1
    assert propensity.get("LOGGED_EXIT_NOW_PROPENSITY_1") == 1
    assert propensity.get("COUNTERFACTUAL_EXIT_NOW_NO_PROPENSITY") == 4


def test_terminal_consistency_audit_passes_on_well_formed_augmented() -> None:
    state = _make_synthetic_state(5)
    augmented = gate._augment_with_action_pairs(state)
    augmented = gate._add_next_state_pointers(state, augmented)
    audit = gate._terminal_consistency_audit(augmented)
    assert audit["status_v1"] == "PASS"


def test_add_next_state_pointers_links_hold_to_next_bar_and_terminal_to_none() -> None:
    state = _make_synthetic_state(5)
    augmented = gate._augment_with_action_pairs(state)
    augmented = gate._add_next_state_pointers(state, augmented)
    # HOLD non-terminal must have next pointer
    hold_non_terminal = augmented[
        (augmented["action_id_v1"] == gate.ACTION_HOLD_ID)
        & (augmented["bars_held_v1"] < 4)
    ]
    assert hold_non_terminal["next_row_id_per_bar_v1"].notna().all()
    # EXIT_NOW always must have no next
    exit_now = augmented[augmented["action_id_v1"] == gate.ACTION_EXIT_NOW_ID]
    assert exit_now["next_row_id_per_bar_v1"].isna().all()
    # HOLD at terminal must also have no next
    hold_terminal = augmented[
        (augmented["action_id_v1"] == gate.ACTION_HOLD_ID)
        & (augmented["bars_held_v1"] == 4)
    ]
    assert hold_terminal["next_row_id_per_bar_v1"].isna().all()


def test_no_shortcut_audit_passes_on_augmented() -> None:
    state = _make_synthetic_state(5)
    augmented = gate._augment_with_action_pairs(state)
    augmented = gate._add_next_state_pointers(state, augmented)
    audit = gate._no_shortcut_audit(augmented)
    assert audit["status_v1"] == "PASS"


def test_action_balance_audit_passes_on_2x_augmented() -> None:
    state = _make_synthetic_state(5)
    augmented = gate._augment_with_action_pairs(state)
    audit = gate._action_balance_audit(augmented)
    assert audit["status_v1"] == "PASS"
    counts = audit["action_counts_v1"]
    assert int(counts["0"]) == int(counts["1"])


def test_write_artifacts_produces_required_outputs(tmp_path: Path) -> None:
    out_root = tmp_path / "EXIT_ACTION_SUPPORT_AUGMENT_V1_20260429T999999Z_LOCK"
    result = gate.write_artifacts(out_root=out_root, built_at_utc="20260429T999999Z")
    artifact_root = Path(result["artifact_root"])
    assert artifact_root == out_root
    for required in [
        "manifest_v1.json",
        "summary_v1.json",
        "status_v1.json",
        "report_v1.md",
        "exit_action_support_augment_go_no_go_v1.json",
        "input_manifest_v1.json",
        "no_shortcut_audit_v1.json",
        "action_balance_audit_v1.json",
        "reward_distribution_audit_v1.json",
        "reward_distribution_audit_v1.csv",
        "terminal_consistency_audit_v1.json",
        "join_coverage_audit_v1.json",
        "reproducibility_audit_v1.json",
        "augmented_per_bar_action_dataset_v1.parquet",
    ]:
        assert (artifact_root / required).exists(), f"missing {required}"
    summary = json.loads((artifact_root / "summary_v1.json").read_text())
    assert summary["final_status_v1"] in gate.ALLOWED_FINAL_STATUSES
    assert summary["next_action_v1"] in gate.ALLOWED_NEXT_ACTIONS
    assert summary["training_blocked_v1"] is True
    assert summary["augmentation_factor_v1"] == 2.0
    assert summary["reward_variant_count_v1"] == 5
    counts = summary["action_counts_v1"]
    assert int(counts["0"]) == int(counts["1"])
