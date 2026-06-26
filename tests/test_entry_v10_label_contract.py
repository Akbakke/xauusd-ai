from __future__ import annotations

from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    V11_DIRECTION_HORIZON_BARS,
    V11_TRADABLE_PNL_MIN_BPS,
    direction_label_contract,
    final_direction_label_horizon_array,
    final_direction_label_horizon_bars,
)


def test_final_direction_label_horizon_matches_v11_direction_horizon() -> None:
    assert final_direction_label_horizon_bars() == V11_DIRECTION_HORIZON_BARS == 24


def test_direction_label_contract_documents_final_emitted_target() -> None:
    contract = direction_label_contract()

    assert contract["direction_label_horizon_bars"] == 24
    assert contract["direction_tradable_pnl_min_bps"] == V11_TRADABLE_PNL_MIN_BPS
    assert contract["direction_label_source"] == "v11_spread_aware_final_pnl_at_direction_horizon"


def test_final_direction_label_horizon_array_uses_emitted_target_horizon() -> None:
    out = final_direction_label_horizon_array(3)

    assert out.dtype.name == "int32"
    assert out.tolist() == [24, 24, 24]
