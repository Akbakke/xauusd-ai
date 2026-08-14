from __future__ import annotations

import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def _base_hier_batch() -> dict[str, torch.Tensor]:
    return {
        "ctx_cat": torch.zeros((1, 5), dtype=torch.long),
        "y_trade": torch.tensor([1.0], dtype=torch.float32),
        "y_side": torch.tensor([0], dtype=torch.long),
        "y_side_mask": torch.tensor([1.0], dtype=torch.float32),
        "y_long_path_utility_bps": torch.tensor([0.0], dtype=torch.float32),
        "y_short_path_utility_bps": torch.tensor([0.0], dtype=torch.float32),
        "y_long_valid_trade": torch.tensor([0.0], dtype=torch.float32),
        "y_short_valid_trade": torch.tensor([0.0], dtype=torch.float32),
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


def _base_hier_out(**overrides: torch.Tensor) -> dict[str, torch.Tensor]:
    out = {
        "trade_logit": torch.zeros((1, 1), dtype=torch.float32),
        "side_logits": torch.zeros((1, 2), dtype=torch.float32),
        "side_utility": torch.zeros((1, 2), dtype=torch.float32),
        "side_bad_path_logit": torch.zeros((1, 2), dtype=torch.float32),
        "side_mae": torch.zeros((1, 2), dtype=torch.float32),
        "side_validity_logit": torch.zeros((1, 2), dtype=torch.float32),
    }
    out.update(overrides)
    return out


def _hier(out, batch):
    tasks, stats = trainer._hierarchical_entry_task_losses(
        out, batch, torch.device("cpu")
    )
    return sum(tasks.values()), stats


def test_hierarchical_loss_does_not_rewrite_short_bad_path_from_structure() -> None:
    batch = _base_hier_batch()
    batch["y_support_retest_continuation"] = torch.tensor([1.0], dtype=torch.float32)
    batch["y_countertrend_short_trap"] = torch.tensor([1.0], dtype=torch.float32)
    out = _base_hier_out(
        side_bad_path_logit=torch.tensor([[-4.0, -4.0]], dtype=torch.float32)
    )

    structural_loss, stats = _hier(out, batch)

    plain_loss, _ = _hier(out, _base_hier_batch())

    assert stats["hier_short_bad_target_rate"] == pytest.approx(0.0)
    assert stats["hier_long_bad_target_rate"] == pytest.approx(0.0)
    assert float(structural_loss.detach().cpu().item()) == pytest.approx(
        float(plain_loss.detach().cpu().item())
    )


def test_hierarchical_loss_does_not_rewrite_long_bad_path_from_structure() -> None:
    batch = _base_hier_batch()
    batch["y_resistance_retest_continuation"] = torch.tensor([1.0], dtype=torch.float32)
    batch["y_countertrend_long_trap"] = torch.tensor([1.0], dtype=torch.float32)
    out = _base_hier_out(
        side_bad_path_logit=torch.tensor([[-4.0, -4.0]], dtype=torch.float32)
    )

    structural_loss, stats = _hier(out, batch)

    plain_loss, _ = _hier(out, _base_hier_batch())

    assert stats["hier_long_bad_target_rate"] == pytest.approx(0.0)
    assert stats["hier_short_bad_target_rate"] == pytest.approx(0.0)
    assert float(structural_loss.detach().cpu().item()) == pytest.approx(
        float(plain_loss.detach().cpu().item())
    )


def test_hierarchical_side_validity_learns_side_specific_valid_trade(monkeypatch) -> None:
    batch = _base_hier_batch()
    batch["y_long_valid_trade"] = torch.tensor([1.0], dtype=torch.float32)

    out_good = _base_hier_out(
        side_validity_logit=torch.tensor([[4.0, -4.0]], dtype=torch.float32)
    )
    out_bad = _base_hier_out(
        side_validity_logit=torch.tensor([[-4.0, 4.0]], dtype=torch.float32)
    )

    good_loss, good_stats = _hier(out_good, batch)
    bad_loss, bad_stats = _hier(out_bad, batch)

    assert good_stats["hier_long_valid_target_rate"] == pytest.approx(1.0)
    assert good_stats["hier_short_valid_target_rate"] == pytest.approx(0.0)
    assert good_stats["hier_side_validity_loss"] < bad_stats["hier_side_validity_loss"]
    assert float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())


def test_hierarchical_side_validity_uses_outcome_targets_not_structural_rewrite(monkeypatch) -> None:
    batch = _base_hier_batch()
    batch["y_long_valid_trade"] = torch.tensor([1.0], dtype=torch.float32)
    batch["y_long_high_mae_low_mfe_early_failure"] = torch.tensor([1.0], dtype=torch.float32)

    out_good = _base_hier_out(
        side_validity_logit=torch.tensor([[4.0, -4.0]], dtype=torch.float32)
    )
    out_bad = _base_hier_out(
        side_validity_logit=torch.tensor([[-4.0, -4.0]], dtype=torch.float32)
    )

    good_loss, good_stats = _hier(out_good, batch)
    bad_loss, bad_stats = _hier(out_bad, batch)

    assert good_stats["hier_long_valid_target_rate"] == pytest.approx(1.0)
    assert bad_stats["hier_long_bad_target_rate"] == pytest.approx(0.0)
    assert float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())


def test_side_evidence_head_receives_direct_side_supervision(monkeypatch) -> None:
    batch = _base_hier_batch()

    out_good = _base_hier_out(side_logits=torch.tensor([[4.0, -4.0]], dtype=torch.float32))
    out_bad = _base_hier_out(side_logits=torch.tensor([[-4.0, 4.0]], dtype=torch.float32))

    good_loss, good_stats = _hier(out_good, batch)
    bad_loss, bad_stats = _hier(out_bad, batch)

    assert good_stats["hier_side_rows"] == pytest.approx(1.0)
    assert good_stats["hier_side_acc"] == pytest.approx(1.0)
    assert bad_stats["hier_side_acc"] == pytest.approx(0.0)
    assert good_stats["hier_side_loss"] < bad_stats["hier_side_loss"]
    assert float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())




def test_trade_evidence_head_receives_direct_outcome_supervision(monkeypatch) -> None:
    batch = _base_hier_batch()

    out_good = _base_hier_out(trade_logit=torch.tensor([[4.0]], dtype=torch.float32))
    out_bad = _base_hier_out(trade_logit=torch.tensor([[-4.0]], dtype=torch.float32))

    good_loss, good_stats = _hier(out_good, batch)
    bad_loss, bad_stats = _hier(out_bad, batch)

    assert good_stats["hier_trade_loss"] < bad_stats["hier_trade_loss"]
    assert float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())




def test_hierarchical_outcome_heads_are_direct_model_evidence(monkeypatch) -> None:
    batch = _base_hier_batch()
    batch["y_long_path_utility_bps"] = torch.tensor([20.0], dtype=torch.float32)
    batch["y_short_path_utility_bps"] = torch.tensor([-10.0], dtype=torch.float32)
    batch["y_long_bad_path"] = torch.tensor([0.0], dtype=torch.float32)
    batch["y_short_bad_path"] = torch.tensor([1.0], dtype=torch.float32)
    batch["y_long_expected_mae_bps"] = torch.tensor([2.0], dtype=torch.float32)
    batch["y_short_expected_mae_bps"] = torch.tensor([30.0], dtype=torch.float32)
    good = _base_hier_out(
        side_utility=torch.tensor([[20.0, -10.0]]),
        side_bad_path_logit=torch.tensor([[-8.0, 8.0]]),
        side_mae=torch.tensor([[2.0, 30.0]]),
    )
    bad = _base_hier_out(
        side_utility=-good["side_utility"],
        side_bad_path_logit=-good["side_bad_path_logit"],
        side_mae=torch.flip(good["side_mae"], dims=(1,)),
    )

    good_loss, good_stats = _hier(good, batch)
    bad_loss, bad_stats = _hier(bad, batch)

    assert good_stats["hier_utility_loss"] < bad_stats["hier_utility_loss"]
    assert good_stats["hier_bad_path_loss"] < bad_stats["hier_bad_path_loss"]
    assert good_stats["hier_mae_loss"] < bad_stats["hier_mae_loss"]
    assert float(good_loss.item()) < float(bad_loss.item())


def test_hierarchical_loss_fails_closed_without_mandatory_evidence_head() -> None:
    out = _base_hier_out()
    del out["side_logits"]

    with pytest.raises(KeyError, match="side_logits"):
        _hier(out, _base_hier_batch())


def _rail_batch() -> dict[str, torch.Tensor]:
    batch = _base_hier_batch()
    batch.update(
        {
            "y_line_support_touch_held": torch.tensor([1.0], dtype=torch.float32),
            "y_line_support_touch_mask": torch.tensor([1.0], dtype=torch.float32),
            "y_line_resistance_touch_held": torch.tensor([0.0], dtype=torch.float32),
            "y_line_resistance_touch_mask": torch.tensor([0.0], dtype=torch.float32),
            "y_countertrend_short_trap": torch.tensor([1.0], dtype=torch.float32),
            "y_countertrend_long_trap": torch.tensor([0.0], dtype=torch.float32),
            "y_short_high_mae_low_mfe_early_failure": torch.tensor([1.0], dtype=torch.float32),
            "y_long_high_mae_low_mfe_early_failure": torch.tensor([0.0], dtype=torch.float32),
        }
    )
    return batch


def test_trendline_rail_aux_loss_supervises_exact_six_label_contract(monkeypatch) -> None:
    out = {
        "trendline_rail_logits": torch.tensor(
            [[8.0, -8.0, 8.0, -8.0, 8.0, -8.0]],
            dtype=torch.float32,
        )
    }

    loss, stats = trainer._trendline_rail_aux_loss(out, _rail_batch(), torch.device("cpu"))

    assert float(loss.detach().cpu().item()) < 0.001
    assert stats["trendline_rail_rows"] == pytest.approx(1.0)
    assert stats["trendline_rising_rows"] == pytest.approx(1.0)
    assert stats["trendline_falling_rows"] == pytest.approx(0.0)


def test_trendline_rail_aux_loss_ignores_direction_and_utility_outputs(monkeypatch) -> None:
    rail_logits = torch.zeros((1, 6), dtype=torch.float32)
    out_long = {
        "trendline_rail_logits": rail_logits,
        "direction_logits": torch.tensor([[20.0, -20.0, -20.0]], dtype=torch.float32),
        "side_logits": torch.tensor([[20.0, -20.0]], dtype=torch.float32),
        "side_utility": torch.tensor([[20.0, -20.0]], dtype=torch.float32),
        "trade_logit": torch.tensor([[20.0]], dtype=torch.float32),
    }
    out_short = {
        "trendline_rail_logits": rail_logits,
        "direction_logits": torch.tensor([[-20.0, 20.0, -20.0]], dtype=torch.float32),
        "side_logits": torch.tensor([[-20.0, 20.0]], dtype=torch.float32),
        "side_utility": torch.tensor([[-20.0, 20.0]], dtype=torch.float32),
        "trade_logit": torch.tensor([[-20.0]], dtype=torch.float32),
    }

    long_loss, long_stats = trainer._trendline_rail_aux_loss(
        out_long, _rail_batch(), torch.device("cpu")
    )
    short_loss, short_stats = trainer._trendline_rail_aux_loss(
        out_short, _rail_batch(), torch.device("cpu")
    )

    assert float(long_loss.detach().cpu().item()) == pytest.approx(
        float(short_loss.detach().cpu().item())
    )
    assert long_stats == short_stats


def test_trendline_rail_aux_loss_fails_closed_on_incomplete_contract(monkeypatch) -> None:
    batch = _rail_batch()
    del batch["y_long_high_mae_low_mfe_early_failure"]

    with pytest.raises(
        RuntimeError,
        match="ENTRY_MODEL_NATIVE_ACTIVE_HEAD_TARGET_MISSING.*y_long_high_mae_low_mfe_early_failure",
    ):
        trainer._trendline_rail_aux_loss(
            {"trendline_rail_logits": torch.zeros((1, 6), dtype=torch.float32)},
            batch,
            torch.device("cpu"),
        )

    with pytest.raises(RuntimeError, match="ENTRY_TRENDLINE_RAIL_OUTPUT_DIM_MISMATCH"):
        trainer._trendline_rail_aux_loss(
            {"trendline_rail_logits": torch.zeros((1, 4), dtype=torch.float32)},
            _rail_batch(),
            torch.device("cpu"),
        )
