from __future__ import annotations

from gx1.scripts import build_entry_v10_ctx_training_dataset_v3 as builder
from tests.entry_direction_target_policy_support import (
    entry_direction_target_policy_fixture,
)


def test_final_direction_label_horizon_is_owned_by_train_fit_policy() -> None:
    policy = entry_direction_target_policy_fixture()
    assert builder.final_direction_label_horizon_bars(policy) == policy[
        "selected_direction_horizon_bars"
    ]


def test_direction_label_contract_is_exact_train_fitted_executable_pnl() -> None:
    policy = entry_direction_target_policy_fixture()
    contract = builder.direction_label_contract(policy)

    assert contract["direction_label_horizon_bars"] == policy[
        "selected_direction_horizon_bars"
    ]
    assert contract["direction_target_mode"] == "train_fitted_executable_pnl_v1"
    assert contract["direction_side_score_formula"] == policy[
        "side_score_formula"
    ]
    assert contract["direction_tradable_edge_floor_bps"] == policy[
        "tradable_edge_floor_bps"
    ]
    assert contract["entry_direction_target_policy_sha256"] == policy[
        "policy_sha256"
    ]


def test_direction_target_has_no_environment_switch() -> None:
    source = builder.Path(builder.__file__).read_text(encoding="utf-8")

    assert "GX1_ENTRY_DIRECTION_TARGET_MODE" not in source
    assert "GX1_ENTRY_DIRECTION_UTILITY_" not in source
    assert '--early_move_threshold_bps' not in source
    assert 'V12_DIRECTION_TARGET_MODE = "train_fitted_executable_pnl_v1"' in source


def test_final_direction_label_horizon_array_uses_emitted_target_horizon() -> None:
    policy = entry_direction_target_policy_fixture()
    out = builder.final_direction_label_horizon_array(
        3,
        target_policy=policy,
    )

    assert out.dtype.name == "int32"
    assert out.tolist() == [policy["selected_direction_horizon_bars"]] * 3


def test_hierarchical_direction_label_contract_forbids_feature_rewrites() -> None:
    contract = builder.hierarchical_direction_label_contract()
    targets = contract["hierarchical_direction_targets"]

    assert targets["primary_head"] == "trade_vs_flat"
    assert targets["conditional_side_head"] == "long_vs_short_given_trade"
    assert targets["runtime_rule_free"] is True
    assert "y_countertrend_short_trap" in targets["target_columns"]
    assert "y_line_support_touch_held" in targets["target_columns"]
    assert "y_line_support_touch_mask" in targets["target_columns"]
    assert "y_line_resistance_touch_held" in targets["target_columns"]
    assert "y_line_resistance_touch_mask" in targets["target_columns"]
    assert "y_long_valid_trade" in targets["target_columns"]
    assert "y_short_valid_trade" in targets["target_columns"]
    assert "y_rising_channel_support_touch" not in targets["target_columns"]
    assert targets["core_target_source"] == "future_executable_pnl_outcomes_only"
    assert targets["feature_derived_core_rewrites_allowed"] is False
    assert targets["utility_order_forcing_allowed"] is False
    assert targets["structural_context_auxiliaries"]["enabled"] is True
    assert targets["structural_context_auxiliaries"]["may_change_core_targets"] is False
