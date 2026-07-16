from __future__ import annotations

import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def _base_hier_batch() -> dict[str, torch.Tensor]:
    return {
        "y_trade": torch.tensor([1.0], dtype=torch.float32),
        "y_side": torch.tensor([0], dtype=torch.long),
        "y_side_mask": torch.tensor([1.0], dtype=torch.float32),
        "y_long_path_utility_bps": torch.tensor([0.0], dtype=torch.float32),
        "y_short_path_utility_bps": torch.tensor([0.0], dtype=torch.float32),
        "y_long_bad_path": torch.tensor([0.0], dtype=torch.float32),
        "y_short_bad_path": torch.tensor([0.0], dtype=torch.float32),
        "y_long_expected_mae_bps": torch.tensor([0.0], dtype=torch.float32),
        "y_short_expected_mae_bps": torch.tensor([0.0], dtype=torch.float32),
        "y_support_retest_continuation": torch.tensor([0.0], dtype=torch.float32),
        "y_resistance_retest_continuation": torch.tensor([0.0], dtype=torch.float32),
        "y_countertrend_short_trap": torch.tensor([0.0], dtype=torch.float32),
        "y_countertrend_long_trap": torch.tensor([0.0], dtype=torch.float32),
        "y_short_high_mae_low_mfe_early_failure": torch.tensor([0.0], dtype=torch.float32),
        "y_long_high_mae_low_mfe_early_failure": torch.tensor([0.0], dtype=torch.float32),
    }


def test_hierarchical_loss_folds_short_support_trap_into_short_bad_path() -> None:
    batch = _base_hier_batch()
    batch["y_support_retest_continuation"] = torch.tensor([1.0], dtype=torch.float32)
    batch["y_countertrend_short_trap"] = torch.tensor([1.0], dtype=torch.float32)
    out = {"side_bad_path_logit": torch.tensor([[0.0, 4.0]], dtype=torch.float32)}

    loss, stats = trainer._hierarchical_entry_loss(
        out,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )

    assert stats["hier_short_bad_target_rate"] == pytest.approx(1.0)
    assert stats["hier_long_bad_target_rate"] == pytest.approx(0.0)
    assert stats["hier_countertrend_short_trap_rate"] == pytest.approx(1.0)
    assert float(loss.detach().cpu().item()) < 0.20


def test_hierarchical_loss_folds_long_resistance_trap_into_long_bad_path() -> None:
    batch = _base_hier_batch()
    batch["y_resistance_retest_continuation"] = torch.tensor([1.0], dtype=torch.float32)
    batch["y_countertrend_long_trap"] = torch.tensor([1.0], dtype=torch.float32)
    out = {"side_bad_path_logit": torch.tensor([[4.0, 0.0]], dtype=torch.float32)}

    loss, stats = trainer._hierarchical_entry_loss(
        out,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )

    assert stats["hier_long_bad_target_rate"] == pytest.approx(1.0)
    assert stats["hier_short_bad_target_rate"] == pytest.approx(0.0)
    assert stats["hier_countertrend_long_trap_rate"] == pytest.approx(1.0)
    assert float(loss.detach().cpu().item()) < 0.20


def test_hierarchical_side_validity_learns_side_specific_valid_trade(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "ENTRY_HIER_SIDE_VALIDITY_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS", 10.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SIDE_VALIDITY_POS_WEIGHT_CAP", 8.0)
    batch = _base_hier_batch()
    batch["y_long_path_utility_bps"] = torch.tensor([25.0], dtype=torch.float32)
    batch["y_short_path_utility_bps"] = torch.tensor([-10.0], dtype=torch.float32)
    batch["y_short_bad_path"] = torch.tensor([1.0], dtype=torch.float32)

    out_good = {"side_validity_logit": torch.tensor([[4.0, -4.0]], dtype=torch.float32)}
    out_bad = {"side_validity_logit": torch.tensor([[-4.0, 4.0]], dtype=torch.float32)}

    good_loss, good_stats = trainer._hierarchical_entry_loss(
        out_good,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )
    bad_loss, bad_stats = trainer._hierarchical_entry_loss(
        out_bad,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )

    assert good_stats["hier_long_valid_target_rate"] == pytest.approx(1.0)
    assert good_stats["hier_short_valid_target_rate"] == pytest.approx(0.0)
    assert good_stats["hier_side_validity_loss"] < bad_stats["hier_side_validity_loss"]
    assert float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())


def test_hierarchical_side_validity_invalidates_early_failure_even_with_positive_utility(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "ENTRY_HIER_SIDE_VALIDITY_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_HIER_SIDE_VALIDITY_MIN_UTILITY_BPS", 10.0)
    batch = _base_hier_batch()
    batch["y_long_path_utility_bps"] = torch.tensor([40.0], dtype=torch.float32)
    batch["y_long_high_mae_low_mfe_early_failure"] = torch.tensor([1.0], dtype=torch.float32)

    out_good = {"side_validity_logit": torch.tensor([[-4.0, -4.0]], dtype=torch.float32)}
    out_bad = {"side_validity_logit": torch.tensor([[4.0, -4.0]], dtype=torch.float32)}

    good_loss, good_stats = trainer._hierarchical_entry_loss(
        out_good,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )
    bad_loss, bad_stats = trainer._hierarchical_entry_loss(
        out_bad,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )

    assert good_stats["hier_long_valid_target_rate"] == pytest.approx(0.0)
    assert bad_stats["hier_long_bad_target_rate"] == pytest.approx(1.0)
    assert float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())


def test_public_side_head_receives_direct_side_supervision(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "ENTRY_HIER_SIDE_WEIGHT", 1.0)
    batch = _base_hier_batch()

    out_good = {"public_side_logits": torch.tensor([[4.0, -4.0]], dtype=torch.float32)}
    out_bad = {"public_side_logits": torch.tensor([[-4.0, 4.0]], dtype=torch.float32)}

    good_loss, good_stats = trainer._hierarchical_entry_loss(
        out_good,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )
    bad_loss, bad_stats = trainer._hierarchical_entry_loss(
        out_bad,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )

    assert good_stats["hier_public_side_rows"] == pytest.approx(1.0)
    assert good_stats["hier_public_side_acc"] == pytest.approx(1.0)
    assert bad_stats["hier_public_side_acc"] == pytest.approx(0.0)
    assert good_stats["hier_public_side_loss"] < bad_stats["hier_public_side_loss"]
    assert float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())


def test_public_trade_head_receives_direct_trade_supervision(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "ENTRY_HIER_TRADE_WEIGHT", 1.0)
    batch = _base_hier_batch()

    out_good = {"public_trade_logit": torch.tensor([[4.0]], dtype=torch.float32)}
    out_bad = {"public_trade_logit": torch.tensor([[-4.0]], dtype=torch.float32)}

    good_loss, good_stats = trainer._hierarchical_entry_loss(
        out_good,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )
    bad_loss, bad_stats = trainer._hierarchical_entry_loss(
        out_bad,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )

    assert good_stats["hier_public_trade_loss"] < bad_stats["hier_public_trade_loss"]
    assert float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())


def test_trendline_rail_aux_loss_penalizes_rising_support_short_confidence(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT", 0.0)
    batch = _base_hier_batch()
    batch["y_rising_channel_support_touch"] = torch.tensor([1.0], dtype=torch.float32)
    batch["y_falling_channel_resistance_touch"] = torch.tensor([0.0], dtype=torch.float32)
    batch["y_countertrend_short_trap"] = torch.tensor([1.0], dtype=torch.float32)
    batch["y_countertrend_long_trap"] = torch.tensor([0.0], dtype=torch.float32)
    out = {"trendline_rail_logits": torch.tensor([[4.0, -4.0, 4.0, -4.0]], dtype=torch.float32)}

    high_short_probs = torch.tensor([[0.05, 0.90, 0.05]], dtype=torch.float32)
    low_short_probs = torch.tensor([[0.90, 0.05, 0.05]], dtype=torch.float32)
    high_loss, high_stats = trainer._trendline_rail_aux_loss(out, batch, high_short_probs, torch.device("cpu"))
    low_loss, low_stats = trainer._trendline_rail_aux_loss(out, batch, low_short_probs, torch.device("cpu"))

    assert high_stats["trendline_rail_rows"] == pytest.approx(1.0)
    assert high_stats["trendline_rising_rows"] == pytest.approx(1.0)
    assert low_stats["trendline_wrong_side_prob"] == pytest.approx(0.05)
    assert high_stats["trendline_wrong_side_prob"] == pytest.approx(0.90)
    assert float(high_loss.detach().cpu().item()) > float(low_loss.detach().cpu().item())


def test_trendline_rail_aux_loss_penalizes_final_short_margin_in_support_pocket(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT", 2.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_MARGIN", 0.75)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS", 15.0)

    batch = _base_hier_batch()
    batch["y_trade"] = torch.tensor([0.0], dtype=torch.float32)
    batch["y_side_mask"] = torch.tensor([0.0], dtype=torch.float32)
    batch["y_rising_channel_support_touch"] = torch.tensor([0.0], dtype=torch.float32)
    batch["y_falling_channel_resistance_touch"] = torch.tensor([0.0], dtype=torch.float32)
    batch["y_support_retest_continuation"] = torch.tensor([1.0], dtype=torch.float32)
    out_bad = {
        "trendline_rail_logits": torch.zeros((1, 4), dtype=torch.float32),
        "direction_logits": torch.tensor([[0.0, 3.0, 0.5]], dtype=torch.float32),
        "side_logits": torch.tensor([[0.0, 3.0]], dtype=torch.float32),
        "side_utility": torch.tensor([[0.0, 3.0]], dtype=torch.float32),
        "side_bad_path_logit": torch.tensor([[3.0, 0.0]], dtype=torch.float32),
        "trade_logit": torch.tensor([[3.0]], dtype=torch.float32),
    }
    out_good = {
        "trendline_rail_logits": torch.zeros((1, 4), dtype=torch.float32),
        "direction_logits": torch.tensor([[1.0, -3.0, 2.0]], dtype=torch.float32),
        "side_logits": torch.tensor([[3.0, 0.0]], dtype=torch.float32),
        "side_utility": torch.tensor([[3.0, 0.0]], dtype=torch.float32),
        "side_bad_path_logit": torch.tensor([[0.0, 3.0]], dtype=torch.float32),
        "trade_logit": torch.tensor([[-3.0]], dtype=torch.float32),
    }
    probs = torch.tensor([[0.30, 0.40, 0.30]], dtype=torch.float32)

    bad_loss, bad_stats = trainer._trendline_rail_aux_loss(out_bad, batch, probs, torch.device("cpu"))
    good_loss, good_stats = trainer._trendline_rail_aux_loss(out_good, batch, probs, torch.device("cpu"))

    assert bad_stats["trendline_rising_rows"] == pytest.approx(1.0)
    assert bad_stats["trendline_final_margin_loss"] > 0.0
    assert bad_stats["trendline_hier_margin_loss"] > 0.0
    assert bad_stats["trendline_flat_trade_loss"] > 0.0
    assert bad_stats["trendline_utility_margin_loss"] > 0.0
    assert float(bad_loss.detach().cpu().item()) > float(good_loss.detach().cpu().item())


def test_trendline_rail_aux_loss_utility_margin_is_independently_weighted(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_AUX_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_WRONG_SIDE_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_RISING_WRONG_SHORT_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_FALLING_WRONG_LONG_WEIGHT", 1.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_FINAL_MARGIN_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_HIER_MARGIN_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_FLAT_TRADE_WEIGHT", 0.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_WEIGHT", 2.0)
    monkeypatch.setattr(trainer, "ENTRY_TRENDLINE_RAIL_UTILITY_MARGIN_BPS", 15.0)

    batch = _base_hier_batch()
    batch["y_rising_channel_support_touch"] = torch.tensor([1.0], dtype=torch.float32)
    batch["y_falling_channel_resistance_touch"] = torch.tensor([0.0], dtype=torch.float32)
    out_bad = {
        "trendline_rail_logits": torch.zeros((1, 4), dtype=torch.float32),
        "side_utility": torch.tensor([[0.0, 3.0]], dtype=torch.float32),
    }
    out_good = {
        "trendline_rail_logits": torch.zeros((1, 4), dtype=torch.float32),
        "side_utility": torch.tensor([[3.0, 0.0]], dtype=torch.float32),
    }
    probs = torch.tensor([[0.30, 0.40, 0.30]], dtype=torch.float32)

    bad_loss, bad_stats = trainer._trendline_rail_aux_loss(out_bad, batch, probs, torch.device("cpu"))
    good_loss, good_stats = trainer._trendline_rail_aux_loss(out_good, batch, probs, torch.device("cpu"))

    assert bad_stats["trendline_hier_margin_loss"] == pytest.approx(0.0)
    assert bad_stats["trendline_utility_margin_loss"] > 0.0
    assert good_stats["trendline_utility_margin_loss"] > 0.0
    assert float(bad_loss.detach().cpu().item()) > float(good_loss.detach().cpu().item())
