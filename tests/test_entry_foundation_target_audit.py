import numpy as np
import pandas as pd

from gx1.scripts.audit_entry_foundation_targets_v1 import (
    EXPECTED_ACTIVE_AUX_HEADS,
    EXPECTED_BLOCKED_TARGET_HEADS,
    XAU_DIRECTION_REPAIR_TARGET_COLUMNS,
    _drift,
    _head_contract,
    _offline_rl_target_contract,
    _position_size_target_contract,
    _target_metrics,
    _xau_direction_repair_side_quality_contract,
    _xau_direction_repair_liveness,
)


def test_target_metrics_capture_bad_path_negative_path_quality_relation() -> None:
    df = pd.DataFrame(
        {
            "y_direction": [0, 1, 2, 0, 1],
            "y_tradable": [1, 1, 0, 1, 0],
            "y_bad_path": [0, 0, 1, 1, 1],
            "path_quality_bps": [20.0, 15.0, -10.0, -15.0, -20.0],
            "mae_first_n_bps": [2.0, 3.0, 12.0, 15.0, 20.0],
            "mfe_first_n_bps": [25.0, 18.0, 5.0, 2.0, 1.0],
            "y_clean_edge_bidir": [1, 1, 0, 0, 0],
            "y_survival_bidir": [1, 1, 0, 0, 0],
            "y_tail_mae_long_K48": [1, 2, 3, 4, 5],
            "y_tail_mae_short_K48": [5, 4, 3, 2, 1],
        }
    )

    row = _target_metrics(df, split="train", scope="split", value="ALL")

    assert row["n"] == 5
    assert row["y_direction_rates"]["long"] == 0.4
    assert row["y_bad_path_rate"] == 0.6
    assert row["bad_path_vs_path_quality_spearman"] < 0.0
    assert row["majority_label_baseline_acc"] == 0.4


def test_target_drift_compares_val_to_train_for_same_scope() -> None:
    train = {"split": "train", "scope": "split", "value": "ALL", "side": "ALL", "y_bad_path_rate": 0.2, "y_tradable_rate": 0.8, "path_quality_mean_bps": 5.0, "majority_label_baseline_acc": 0.4, "trade_label_rate": 0.7}
    val = {"split": "val", "scope": "split", "value": "ALL", "side": "ALL", "y_bad_path_rate": 0.3, "y_tradable_rate": 0.7, "path_quality_mean_bps": 4.0, "majority_label_baseline_acc": 0.5, "trade_label_rate": 0.6}

    rows = _drift([train, val])

    assert len(rows) == 1
    assert np.isclose(rows[0]["y_bad_path_rate_delta_vs_train"], 0.1)
    assert np.isclose(rows[0]["path_quality_mean_bps_delta_vs_train"], -1.0)


def test_target_head_contract_blocks_constant_hold_horizon_and_keeps_live_heads_active() -> None:
    rows = []
    for split in ("train", "val"):
        frame = {
            "split": [split, split, split],
            "y_direction": [0, 1, 2],
            "path_quality_bps": [3.0, 7.0, 11.0],
            "y_tf_agreement_score": [0.0, 0.5, 1.0],
            "y_position_size_target": [0.50, 0.75, 1.0],
            "y_hold_horizon_target": [0.5, 0.5, 0.5],
            "y_forecast_ret_K1": [1.0, 2.0, 3.0],
            "y_forecast_ret_K5": [1.0, 2.0, 3.0],
            "y_forecast_ret_K12": [1.0, 2.0, 3.0],
            "y_forecast_ret_K24": [1.0, 2.0, 3.0],
            "y_vol_fwd_K12": [1.0, 2.0, 3.0],
            "y_vol_fwd_K48": [1.0, 2.0, 3.0],
            "y_vol_fwd_K96": [1.0, 2.0, 3.0],
        }
        for side in ("long", "short"):
            for horizon in (12, 48, 96):
                frame[f"y_dip_mae_{side}_K{horizon}"] = [1.0, 2.0, 3.0]
                frame[f"y_dip_mfe_{side}_K{horizon}"] = [2.0, 3.0, 4.0]
                frame[f"y_dip_bottom_frac_{side}_K{horizon}"] = [0.0, 0.5, 1.0]
                frame[f"y_time_to_mfe_frac_{side}_K{horizon}"] = [0.1, 0.6, 0.9]
                frame[f"y_tail_mae_{side}_K{horizon}"] = [1.0, 2.0, 3.0]
        for horizon in (12, 48, 96):
            frame[f"y_action_value_long_K{horizon}"] = [10.0, -5.0, -5.0]
            frame[f"y_action_value_short_K{horizon}"] = [-5.0, 10.0, -5.0]
            frame[f"y_action_value_flat_K{horizon}"] = [0.0, 0.0, 0.0]
        rows.append(pd.DataFrame(frame))

    contract = _head_contract(rows)

    for head in EXPECTED_ACTIVE_AUX_HEADS:
        assert head in contract["active_training_heads"]
        assert contract["head_target_liveness"][head]["live_all_splits"] is True
    for head in EXPECTED_BLOCKED_TARGET_HEADS:
        assert head in contract["blocked_heads"]
        assert contract["head_target_liveness"][head]["live_all_splits"] is False

    offline_rl = _offline_rl_target_contract(rows)
    assert offline_rl["decision"] == "PASS"
    assert offline_rl["failures"] == []


def test_offline_rl_target_contract_rejects_missing_q_and_nonzero_flat() -> None:
    missing = pd.DataFrame({"split": ["train", "train", "train"]})
    assert _offline_rl_target_contract([missing])["decision"] == "FAIL"

    frame = {"split": ["train", "train", "train"]}
    for horizon in (12, 48, 96):
        frame[f"y_action_value_long_K{horizon}"] = [10.0, -5.0, -5.0]
        frame[f"y_action_value_short_K{horizon}"] = [-5.0, 10.0, -5.0]
        frame[f"y_action_value_flat_K{horizon}"] = [0.0, 0.0, 1.0]
    failed = _offline_rl_target_contract([pd.DataFrame(frame)])
    assert failed["decision"] == "FAIL"
    assert any("FLAT" in failure for failure in failed["failures"])


def test_xau_direction_repair_liveness_rejects_dead_present_columns() -> None:
    live_frames = []
    dead_frames = []
    for split in ("train", "val"):
        live_frames.append(
            pd.DataFrame(
                {
                    "split": [split, split, split],
                    "y_rising_channel_support_touch": [0.0, 1.0, 0.0],
                    "y_short_high_mae_low_mfe_early_failure": [0.0, 0.0, 1.0],
                }
            )
        )
        dead_frames.append(
            pd.DataFrame(
                {
                    "split": [split, split, split],
                    "y_rising_channel_support_touch": [0.0, 0.0, 0.0],
                }
            )
        )

    live = _xau_direction_repair_liveness(live_frames)
    dead = _xau_direction_repair_liveness(dead_frames)

    assert live["enabled"] is True
    assert live["live_all_present_columns_all_splits"] is True
    assert live["live_all_expected_columns_all_splits"] is False
    assert "y_trade" in live["missing_columns_any_split"]
    assert dead["enabled"] is True
    assert dead["live_all_present_columns_all_splits"] is False


def test_xau_direction_repair_liveness_requires_all_hierarchical_targets() -> None:
    frames = []
    for split in ("train", "val"):
        data = {"split": [split, split, split]}
        for idx, col in enumerate(XAU_DIRECTION_REPAIR_TARGET_COLUMNS):
            data[col] = [float(idx), float(idx + 1), float(idx + 2)]
        frames.append(pd.DataFrame(data))

    live = _xau_direction_repair_liveness(frames)
    del frames[0]["y_trade"]
    missing = _xau_direction_repair_liveness(frames)

    assert live["all_expected_columns_present_all_splits"] is True
    assert live["live_all_expected_columns_all_splits"] is True
    assert missing["all_expected_columns_present_all_splits"] is False
    assert "y_trade" in missing["missing_columns_any_split"]


def test_xau_direction_repair_side_quality_replaces_scalar_bad_path_monotonicity() -> None:
    frames = []
    for split in ("train", "val"):
        frame = pd.DataFrame(
            {
                "split": [split] * 6,
                "y_long_bad_path": [0, 0, 0, 1, 1, 1],
                "y_short_bad_path": [1, 1, 1, 0, 0, 0],
                "y_long_path_utility_bps": [30, 25, 20, -10, -15, -20],
                "y_short_path_utility_bps": [-30, -25, -20, 10, 15, 20],
                "y_long_expected_mae_bps": [2, 3, 4, 12, 13, 14],
                "y_short_expected_mae_bps": [14, 13, 12, 4, 3, 2],
                # Scalar bad-path is selected-side sparse in XAU repair and
                # is not the side-quality monotonicity surface.
                "y_bad_path": [0, 0, 1, 0, 0, 1],
                "path_quality_bps": [1, 2, 3, 4, 5, 6],
            }
        )
        frames.append(frame)

    contract = _xau_direction_repair_side_quality_contract(frames)

    assert contract["enabled"] is True
    assert contract["all_side_quality_checks_pass"] is True
    assert not contract["failures"]


def test_position_size_target_requires_exact_formula_and_positive_mae_magnitude() -> None:
    direction = np.asarray([0, 1, 2], dtype=np.int64)
    mfe = np.asarray([10.0, 2.0, 0.0], dtype=np.float64)
    mae = np.asarray([2.0, 6.0, 0.0], dtype=np.float64)
    atr = np.asarray([4.0, 4.0, 4.0], dtype=np.float64)
    expected = 1.0 / (1.0 + np.exp(-((mfe - mae) / (2.0 * atr))))
    expected[direction == 2] = 0.5
    frame = pd.DataFrame(
        {
            "split": ["train"] * 3,
            "y_direction": direction,
            "mfe_first_n_bps": mfe,
            "mae_first_n_bps": mae,
            "atr_bps": atr,
            "y_position_size_target": expected.astype(np.float32),
        }
    )

    passed = _position_size_target_contract([frame])
    assert passed["decision"] == "PASS"
    assert passed["mae_semantics"] == "non_negative_adverse_magnitude"
    assert passed["live_size_application_authority"] is False

    negative_mae = frame.copy()
    negative_mae.loc[0, "mae_first_n_bps"] = -2.0
    failed_mae = _position_size_target_contract([negative_mae])
    assert failed_mae["decision"] == "FAIL"
    assert any("non-negative adverse-magnitude" in row for row in failed_mae["failures"])

    wrong_formula = frame.copy()
    wrong_formula.loc[0, "y_position_size_target"] = 0.5
    failed_formula = _position_size_target_contract([wrong_formula])
    assert failed_formula["decision"] == "FAIL"
    assert any("formula mismatch" in row for row in failed_formula["failures"])
