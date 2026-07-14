import numpy as np
import pandas as pd

from gx1.scripts.audit_entry_foundation_targets_v1 import (
    DEFAULT_OUT_DIR,
    EXPECTED_ACTIVE_OPTIONAL_HEADS,
    EXPECTED_BLOCKED_OPTIONAL_HEADS,
    XAU_DIRECTION_REPAIR_TARGET_COLUMNS,
    _drift,
    _head_contract,
    _target_metrics,
    _xau_direction_repair_liveness,
)


def test_target_audit_default_out_dir_is_foundation_seq146() -> None:
    assert DEFAULT_OUT_DIR.name == "foundation_seq146"
    assert DEFAULT_OUT_DIR.parent.name == "entry_target_foundation_audit_20260628_v1"


def test_target_metrics_capture_bad_path_negative_path_quality_relation() -> None:
    df = pd.DataFrame(
        {
            "y_direction": [0, 1, 2, 0, 1],
            "y_tradable": [1, 1, 0, 1, 0],
            "y_bad_path": [0, 0, 1, 1, 1],
            "path_quality_bps": [20.0, 15.0, -10.0, -15.0, -20.0],
            "mae_first_n_bps": [-2.0, -3.0, -12.0, -15.0, -20.0],
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
                frame[f"y_tail_mae_{side}_K{horizon}"] = [1.0, 2.0, 3.0]
        rows.append(pd.DataFrame(frame))

    contract = _head_contract(rows)

    for head in EXPECTED_ACTIVE_OPTIONAL_HEADS:
        assert head in contract["active_training_heads"]
        assert contract["head_target_liveness"][head]["live_all_splits"] is True
    for head in EXPECTED_BLOCKED_OPTIONAL_HEADS:
        assert head in contract["blocked_heads"]
        assert contract["head_target_liveness"][head]["live_all_splits"] is False


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
