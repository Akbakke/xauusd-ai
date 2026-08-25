from __future__ import annotations

from gx1.contracts.entry_direction_target_policy_v1 import (
    ENTRY_DIRECTION_DIAGNOSTIC_OUTCOME_TARGET_MODE,
    entry_direction_diagnostic_outcome_contract,
)
from gx1.scripts import build_entry_v10_ctx_training_dataset_v3 as builder
from tests.entry_direction_target_policy_support import (
    entry_direction_target_policy_fixture,
)


def test_final_direction_label_horizon_is_owned_by_train_fit_policy() -> None:
    policy = entry_direction_target_policy_fixture()
    assert builder.diagnostic_outcome_horizon_bars(policy) == policy[
        "selected_direction_horizon_bars"
    ]


def test_diagnostic_outcome_contract_is_exact_train_fitted_executable_pnl() -> None:
    policy = entry_direction_target_policy_fixture()
    contract = builder.diagnostic_outcome_label_contract(policy)

    assert contract["diagnostic_outcome_horizon_bars"] == policy[
        "selected_direction_horizon_bars"
    ]
    assert contract == entry_direction_diagnostic_outcome_contract(policy)
    assert (
        contract["diagnostic_outcome_target_mode"]
        == ENTRY_DIRECTION_DIAGNOSTIC_OUTCOME_TARGET_MODE
    )
    assert contract["diagnostic_side_score_formula"] == policy[
        "side_score_formula"
    ]
    assert contract["diagnostic_tradable_edge_floor_bps"] == policy[
        "tradable_edge_floor_bps"
    ]
    assert contract["diagnostic_outcome_policy_sha256"] == policy[
        "policy_sha256"
    ]
    # The horizon labels are dataset diagnostics only; they are never the
    # Entry action target or a decision authority.
    assert contract["entry_action_authority"] is False


def test_direction_target_has_no_environment_switch() -> None:
    source = builder.Path(builder.__file__).read_text(encoding="utf-8")

    assert "GX1_ENTRY_DIRECTION_TARGET_MODE" not in source
    assert "GX1_ENTRY_DIRECTION_UTILITY_" not in source
    assert '--early_move_threshold_bps' not in source
    assert "entry_direction_diagnostic_outcome_contract" in source


def test_final_direction_label_horizon_array_uses_emitted_target_horizon() -> None:
    policy = entry_direction_target_policy_fixture()
    out = builder.diagnostic_outcome_horizon_array(
        3,
        target_policy=policy,
    )

    assert out.dtype.name == "int32"
    assert out.tolist() == [policy["selected_direction_horizon_bars"]] * 3


def test_representation_auxiliary_contract_forbids_feature_rewrites() -> None:
    contract = builder.representation_auxiliary_outcome_contract()
    targets = contract["representation_auxiliary_outcomes"]

    assert targets["enabled"] is True
    assert targets["entry_action_authority"] is False
    assert targets["runtime_rule_free"] is True
    assert "y_countertrend_short_trap" in targets["target_columns"]
    assert "y_line_support_touch_held" in targets["target_columns"]
    assert "y_line_support_touch_mask" in targets["target_columns"]
    assert "y_line_resistance_touch_held" in targets["target_columns"]
    assert "y_line_resistance_touch_mask" in targets["target_columns"]
    assert "y_long_valid_trade" in targets["target_columns"]
    assert "y_short_valid_trade" in targets["target_columns"]
    assert "y_rising_channel_support_touch" not in targets["target_columns"]
    assert targets["label_source"] == "future_executable_pnl_outcomes_only"
    assert targets["feature_derived_core_rewrites_allowed"] is False
    assert targets["utility_order_forcing_allowed"] is False
    assert targets["structural_context_auxiliaries"]["enabled"] is True
    assert targets["structural_context_auxiliaries"]["may_change_core_targets"] is False
