from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
    MODEL_NATIVE_AUX_TARGET_COLUMNS,
    MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN,
    _build_model_native_aux_head_targets,
    _position_size_target_from_path,
    _selected_side_bad_path_target,
    _validate_model_native_aux_head_targets,
    hierarchical_direction_label_contract,
    model_native_aux_target_contract_metadata,
)


BUILDER_PATH = Path("gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py")


def _spread_tape(n_rows: int = 130) -> pd.DataFrame:
    close = 2000.0 + np.arange(n_rows, dtype=np.float64) * 0.10
    high = close + 0.20
    low = close - 0.20
    return pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
            "bid_close": close - 0.05,
            "ask_close": close + 0.05,
            "bid_high": high - 0.05,
            "bid_low": low - 0.05,
            "ask_high": high + 0.05,
            "ask_low": low + 0.05,
        }
    )


def test_aux_targets_have_exact_horizons_and_no_fake_tail_values() -> None:
    frame = _spread_tape()
    targets, complete = _build_model_native_aux_head_targets(frame)

    assert tuple(targets) == MODEL_NATIVE_AUX_TARGET_COLUMNS
    assert MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS == 96
    assert complete.tolist() == ([True] * (len(frame) - 96) + [False] * 96)
    for name, horizon in MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN.items():
        values = targets[name]
        assert np.isfinite(values[: len(frame) - horizon]).all()
        assert np.isnan(values[len(frame) - horizon :]).all()
        assert not np.isinf(values).any()


def test_aux_target_contract_is_exact_and_spread_aware() -> None:
    contract = model_native_aux_target_contract_metadata()

    assert contract["schema_version"] == "entry_model_native_aux_targets_v4"
    assert len(contract["columns"]) == 46
    assert contract["columns"] == list(MODEL_NATIVE_AUX_TARGET_COLUMNS)
    assert contract["max_future_horizon_bars"] == 96
    assert contract["spread_aware_risk_magnitudes_required"] is True
    assert contract["mid_price_timing_reference_only"] is True
    assert contract["incomplete_rows_may_be_emitted"] is False
    assert contract["offline_rl"]["action_value_layout"] == "action_major_then_horizon"
    timing = contract["turning_point_timing"]
    assert timing["output_dim"] == 12
    assert timing["layout"][0]["market_turn"] == "BOTTOM"
    assert timing["layout"][6]["market_turn"] == "TOP"
    assert timing["live_direction_rule_authority"] is False


def test_model_native_group_a_recompute_is_memory_capped_and_explicit() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "_MODEL_NATIVE_GROUP_A_RECOMPUTE_WORKERS = 2" in source
    assert "workers=_MODEL_NATIVE_GROUP_A_RECOMPUTE_WORKERS" in source
    assert '"group_a_recompute_workers": _MODEL_NATIVE_GROUP_A_RECOMPUTE_WORKERS' in source


def test_aux_risk_magnitude_uses_executable_spread_path() -> None:
    frame = _spread_tape()
    targets, _ = _build_model_native_aux_head_targets(frame)

    expected_long_mfe = (
        (frame.loc[12, "bid_high"] - frame.loc[0, "ask_close"])
        / frame.loc[0, "ask_close"]
        * 1e4
    )
    expected_short_mfe = max(
        0.0,
        (frame.loc[0, "bid_close"] - frame.loc[1, "ask_low"])
        / frame.loc[0, "bid_close"]
        * 1e4,
    )
    assert targets["y_dip_mfe_long_K12"][0] == pytest.approx(expected_long_mfe)
    assert targets["y_dip_mfe_short_K12"][0] == pytest.approx(expected_short_mfe)


def test_action_values_are_full_counterfactual_spread_aware_path_utilities() -> None:
    frame = _spread_tape()
    targets, _ = _build_model_native_aux_head_targets(frame)

    entry_ask = frame.loc[0, "ask_close"]
    long_pnl = (frame.loc[12, "bid_close"] - entry_ask) / entry_ask * 1e4
    long_mfe = (frame.loc[12, "bid_high"] - entry_ask) / entry_ask * 1e4
    long_mae = max(
        0.0,
        (entry_ask - frame.loc[1:12, "bid_low"].min()) / entry_ask * 1e4,
    )
    expected_long = long_pnl + 0.35 * long_mfe - 1.15 * long_mae + 0.25 * (
        long_mfe - long_mae
    )

    assert targets["y_action_value_long_K12"][0] == pytest.approx(expected_long)
    assert targets["y_action_value_short_K12"][0] < 0.0
    assert targets["y_action_value_flat_K12"][0] == 0.0


def test_aux_target_validator_rejects_finite_incomplete_tail() -> None:
    targets, _ = _build_model_native_aux_head_targets(_spread_tape())
    broken = {name: values.copy() for name, values in targets.items()}
    broken["y_forecast_ret_K1"][-1] = 0.0

    with pytest.raises(RuntimeError, match="AUX_TARGET_COMPLETENESS_INVALID"):
        _validate_model_native_aux_head_targets(broken, n_rows=130)


def test_aux_target_builder_requires_bid_ask_high_low() -> None:
    frame = _spread_tape().drop(columns=["ask_low"])

    with pytest.raises(RuntimeError, match="AUX_SPREAD_TAPE_MISSING.*ask_low"):
        _build_model_native_aux_head_targets(frame)


def test_selected_side_bad_path_is_copied_from_future_outcome() -> None:
    selected_side = np.array([0, 1, -1, 0], dtype=np.int8)
    long_bad = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    short_bad = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    scalar_bad = _selected_side_bad_path_target(selected_side, long_bad, short_bad)

    assert scalar_bad.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_position_size_target_uses_selected_future_path() -> None:
    size = _position_size_target_from_path(
        mfe_first_n_bps=np.array([30.0, 5.0, 0.0, 1.0], dtype=np.float32),
        mae_first_n_bps=np.array([5.0, 20.0, 0.0, 1.0], dtype=np.float32),
        atr_bps=np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32),
        trade_mask=np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )

    assert size[0] > 0.5
    assert size[1] < 0.5
    assert size[2] == 0.5
    assert size[3] == 0.5


def test_position_size_target_decreases_monotonically_with_adverse_excursion() -> None:
    size = _position_size_target_from_path(
        mfe_first_n_bps=np.full(4, 20.0, dtype=np.float32),
        mae_first_n_bps=np.array([0.0, 10.0, 20.0, 40.0], dtype=np.float32),
        atr_bps=np.full(4, 10.0, dtype=np.float32),
        trade_mask=np.ones(4, dtype=np.float32),
    )

    assert np.all(np.diff(size) < 0.0)


@pytest.mark.parametrize(
    ("atr", "mask"),
    [
        (np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)),
        (np.array([np.nan], dtype=np.float32), np.array([1.0], dtype=np.float32)),
        (np.array([10.0], dtype=np.float32), np.array([0.5], dtype=np.float32)),
    ],
)
def test_position_size_target_fails_closed_on_invalid_evidence(atr, mask) -> None:
    with pytest.raises(ValueError, match="POSITION_SIZE_TARGET_INPUT_INVALID"):
        _position_size_target_from_path(
            mfe_first_n_bps=np.array([3.0], dtype=np.float32),
            mae_first_n_bps=np.array([1.0], dtype=np.float32),
            atr_bps=atr,
            trade_mask=mask,
        )


def test_position_size_target_rejects_signed_negative_mae() -> None:
    with pytest.raises(ValueError, match="mae_first_n_bps"):
        _position_size_target_from_path(
            mfe_first_n_bps=np.array([3.0], dtype=np.float32),
            mae_first_n_bps=np.array([-1.0], dtype=np.float32),
            atr_bps=np.array([10.0], dtype=np.float32),
            trade_mask=np.array([1.0], dtype=np.float32),
        )


def test_contract_forbids_feature_derived_core_target_rewrites() -> None:
    target = hierarchical_direction_label_contract()["hierarchical_direction_targets"]

    assert target["core_target_source"] == "future_path_and_utility_outcomes_only"
    assert target["feature_derived_core_rewrites_allowed"] is False
    assert target["utility_order_forcing_allowed"] is False
    assert target["structural_context_auxiliaries"]["may_change_core_targets"] is False


def test_builder_has_no_structural_side_or_utility_repair_primitive() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    assert "def _apply_structural_side_repair" not in source
    assert "def _apply_structural_utility_repair" not in source
    assert "_structural_utility_repair_masks" not in source
    assert "structural_short_to_long" not in source
    assert "structural_long_to_short" not in source
    assert "np.nan_to_num(_side_score" not in source
    assert "LONG_WINDOW_TEACHER" not in source
    assert "long_window_teacher" not in source
    assert "y_teacher_bad_long" not in source
    assert "y_teacher_winner_long" not in source
    assert "GX1_V10_SPREAD_AWARE_RISK_TARGETS" not in source
    assert "GX1_ENTRY_DIRECTION_TARGET_MODE" not in source
    assert "GX1_ENTRY_DIRECTION_UTILITY_" not in source


def test_builder_has_one_model_native_signal_path_and_no_context_soft_pass() -> None:
    source = BUILDER_PATH.read_text(encoding="utf-8")

    forbidden = (
        "XGBMultiheadModel",
        "XGBInputSanitizer",
        "proba_to_signal_bridge_v1",
        "xgb_bundle_path",
        "xgb_model_sha256",
        "xgb_bridge_source",
        "GX1_XGB_BUNDLE_DIR",
        "--xgb",
        "--neutral-xgb-bridge",
        "HARD_NEG_LONG_MIN_XGB_P_LONG",
        "hard_negative_uses_xgb_predictions",
        "hard_negative_long_xgb_p_long_min",
        "entry_runtime_gates",
        "flat_veto",
        "tradable_gate",
        "quality_gate",
        "allow_zero_ctx",
        "_feat.get(_k, 0.0)",
        'fillna("UNKNOWN")',
    )
    assert [token for token in forbidden if token in source] == []
    assert '"hard_negative_candidate_source": _hard_negative_candidate_source' in source
    assert '"neutral_xgb_bridge": False' in source
