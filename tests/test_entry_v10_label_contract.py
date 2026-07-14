from __future__ import annotations

import importlib


def _builder(monkeypatch, mode: str = "v11_final_pnl_h24"):
    monkeypatch.setenv("GX1_ENTRY_DIRECTION_TARGET_MODE", mode)
    import gx1.scripts.build_entry_v10_ctx_training_dataset_v3 as builder

    return importlib.reload(builder)


def test_final_direction_label_horizon_matches_v11_direction_horizon(monkeypatch) -> None:
    builder = _builder(monkeypatch)
    assert builder.final_direction_label_horizon_bars() == builder.V11_DIRECTION_HORIZON_BARS == 24


def test_direction_label_contract_documents_final_emitted_target(monkeypatch) -> None:
    builder = _builder(monkeypatch, "v11_final_pnl_h24")
    contract = builder.direction_label_contract()

    assert contract["direction_label_horizon_bars"] == 24
    assert contract["direction_tradable_pnl_min_bps"] == builder.V11_TRADABLE_PNL_MIN_BPS
    assert contract["direction_label_source"] == "v11_spread_aware_final_pnl_at_direction_horizon"


def test_direction_label_contract_documents_path_utility_mode(monkeypatch) -> None:
    builder = _builder(monkeypatch, "path_utility_v2")
    contract = builder.direction_label_contract()

    assert contract["direction_target_mode"] == "path_utility_v2"
    assert contract["direction_label_source"] == "v12_spread_aware_path_utility_h24_plus_first10"
    assert "direction_utility_formula" in contract


def test_final_direction_label_horizon_array_uses_emitted_target_horizon(monkeypatch) -> None:
    builder = _builder(monkeypatch)
    out = builder.final_direction_label_horizon_array(3)

    assert out.dtype.name == "int32"
    assert out.tolist() == [24, 24, 24]


def test_hierarchical_direction_label_contract_documents_repair_targets(monkeypatch) -> None:
    builder = _builder(monkeypatch)
    contract = builder.hierarchical_direction_label_contract()
    targets = contract["hierarchical_direction_targets"]

    assert targets["primary_head"] == "trade_vs_flat"
    assert targets["conditional_side_head"] == "long_vs_short_given_trade"
    assert targets["runtime_rule_free"] is True
    assert "y_countertrend_short_trap" in targets["target_columns"]
    assert "y_rising_channel_support_touch" in targets["target_columns"]
    assert targets["structural_utility_repair"]["enabled"] is True
    assert "y_short_high_mae_low_mfe_early_failure" in targets["structural_utility_repair"]["anti_short_pockets"]
    assert "y_long_high_mae_low_mfe_early_failure" in targets["structural_utility_repair"]["anti_long_pockets"]
    assert targets["structural_utility_repair"]["utility_margin_bps"] > 0.0
