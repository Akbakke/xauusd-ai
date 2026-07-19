from __future__ import annotations

from gx1.scripts import build_entry_v10_ctx_training_dataset_v3 as builder


def test_final_direction_label_horizon_matches_v11_direction_horizon() -> None:
    assert builder.final_direction_label_horizon_bars() == builder.V11_DIRECTION_HORIZON_BARS == 24


def test_direction_label_contract_is_exact_path_utility() -> None:
    contract = builder.direction_label_contract()

    assert contract["direction_label_horizon_bars"] == 24
    assert contract["direction_tradable_pnl_min_bps"] == builder.V11_TRADABLE_PNL_MIN_BPS
    assert contract["direction_target_mode"] == "path_utility_v2"
    assert contract["direction_label_source"] == "v12_spread_aware_path_utility_h24_plus_first10"
    assert "direction_utility_formula" in contract


def test_direction_target_has_no_environment_switch() -> None:
    source = builder.Path(builder.__file__).read_text(encoding="utf-8")

    assert "GX1_ENTRY_DIRECTION_TARGET_MODE" not in source
    assert "GX1_ENTRY_DIRECTION_UTILITY_" not in source
    assert 'V12_DIRECTION_TARGET_MODE = "path_utility_v2"' in source


def test_final_direction_label_horizon_array_uses_emitted_target_horizon() -> None:
    out = builder.final_direction_label_horizon_array(3)

    assert out.dtype.name == "int32"
    assert out.tolist() == [24, 24, 24]


def test_hierarchical_direction_label_contract_forbids_feature_rewrites() -> None:
    contract = builder.hierarchical_direction_label_contract()
    targets = contract["hierarchical_direction_targets"]

    assert targets["primary_head"] == "trade_vs_flat"
    assert targets["conditional_side_head"] == "long_vs_short_given_trade"
    assert targets["runtime_rule_free"] is True
    assert "y_countertrend_short_trap" in targets["target_columns"]
    assert "y_rising_channel_support_touch" in targets["target_columns"]
    assert targets["core_target_source"] == "future_path_and_utility_outcomes_only"
    assert targets["feature_derived_core_rewrites_allowed"] is False
    assert targets["utility_order_forcing_allowed"] is False
    assert targets["structural_context_auxiliaries"]["enabled"] is True
    assert targets["structural_context_auxiliaries"]["may_change_core_targets"] is False
