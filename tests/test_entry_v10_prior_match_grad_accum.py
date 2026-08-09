from __future__ import annotations

import pytest
import torch

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def _hier_batch(n: int) -> dict[str, torch.Tensor]:
    return {
        "ctx_cat": torch.zeros((n, 5), dtype=torch.long),
        "y_trade": torch.randint(0, 2, (n,)).float(),
        "y_side": torch.randint(0, 2, (n,)),
        "y_side_mask": torch.ones(n, dtype=torch.float32),
        "y_long_path_utility_bps": torch.zeros(n),
        "y_short_path_utility_bps": torch.zeros(n),
        "y_long_bad_path": torch.zeros(n),
        "y_short_bad_path": torch.zeros(n),
        "y_long_expected_mae_bps": torch.zeros(n),
        "y_short_expected_mae_bps": torch.zeros(n),
        "y_support_retest_continuation": torch.zeros(n),
        "y_resistance_retest_continuation": torch.zeros(n),
        "y_countertrend_short_trap": torch.zeros(n),
        "y_countertrend_long_trap": torch.zeros(n),
        "y_short_high_mae_low_mfe_early_failure": torch.zeros(n),
        "y_long_high_mae_low_mfe_early_failure": torch.zeros(n),
    }


def _hier_out(n: int) -> dict[str, torch.Tensor]:
    return {
        "trade_logit": torch.randn(n, 1),
        "side_logits": torch.randn(n, 2),
        "side_utility": torch.zeros(n, 2),
        "side_bad_path_logit": torch.zeros(n, 2),
        "side_mae": torch.zeros(n, 2),
        "side_validity_logit": torch.zeros(n, 2),
    }


def test_buffer_reset_clears_every_slot() -> None:
    buf = trainer._PriorMatchAccumBuffer()
    buf.append("direction_probs", torch.rand(4, 3))
    buf.append("ctx_cat", torch.zeros(4, 5, dtype=torch.long))
    buf.reset()
    for name in buf.__slots__:
        assert getattr(buf, name) is None


def test_buffer_read_on_empty_buffer_returns_the_live_tensor_unchanged() -> None:
    """At grad_accum_steps=1 the buffer is reset every micro-step, so every
    read must be a no-op: same object, not just equal values. This is the
    property that makes the fix byte-for-byte identical to pre-fix behavior
    whenever accumulation is off."""
    buf = trainer._PriorMatchAccumBuffer()
    live = torch.rand(4, 3, requires_grad=True)
    assert buf.read("direction_probs", live) is live


def test_buffer_append_detaches_and_read_concatenates() -> None:
    buf = trainer._PriorMatchAccumBuffer()
    first = torch.rand(4, 3, requires_grad=True)
    buf.append("direction_probs", first)
    assert buf.direction_probs.shape == (4, 3)
    assert not buf.direction_probs.requires_grad

    second = torch.rand(5, 3, requires_grad=True)
    combined = buf.read("direction_probs", second)
    assert combined.shape == (9, 3)
    assert combined.requires_grad

    combined.sum().backward()
    assert second.grad is not None
    assert second.grad.shape == (5, 3)


def test_batch_rate_sampling_floor_is_inert_at_micro_batch_but_not_at_accumulated_n() -> None:
    """The whole point of the buffer: sqrt(0.25/n) is inert (>0.02) at the fast
    micro-batch size (64) that made batch=640 both slow (~11x/row, measured
    2026-08-09) and briefly crash (CUDA illegal access, fixed 2deb50d0), but
    becomes non-inert (<=0.02) once ten such micro-steps accumulate to the
    n=625+ the anti-collapse fix requires."""
    assert trainer._batch_rate_sampling_floor(64) > 0.02
    assert trainer._batch_rate_sampling_floor(625) == pytest.approx(0.02, abs=1e-9)
    assert trainer._batch_rate_sampling_floor(640) <= 0.02


def test_ten_micro_steps_of_sixty_four_accumulate_to_the_required_n() -> None:
    buf = trainer._PriorMatchAccumBuffer()
    micro_batch = 64
    for _ in range(10):
        probs = torch.softmax(torch.randn(micro_batch, 3, requires_grad=True), dim=1)
        y = torch.randint(0, 3, (micro_batch,))
        buf.append("direction_probs", probs)
        buf.append("direction_y", y)
    assert buf.direction_probs.shape[0] == 640
    assert trainer._batch_rate_sampling_floor(int(buf.direction_probs.shape[0])) <= 0.02


def test_hierarchical_entry_loss_with_fresh_buffer_matches_no_buffer() -> None:
    torch.manual_seed(7)
    batch = _hier_batch(6)
    out = _hier_out(6)

    plain_loss, plain_stats = trainer._hierarchical_entry_loss(
        out,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
    )
    buffered_loss, buffered_stats = trainer._hierarchical_entry_loss(
        out,
        batch,
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
        prior_match_buffer=trainer._PriorMatchAccumBuffer(),
    )

    assert float(buffered_loss.detach().cpu().item()) == pytest.approx(
        float(plain_loss.detach().cpu().item())
    )
    for key in plain_stats:
        assert buffered_stats[key] == pytest.approx(plain_stats[key])


def test_hierarchical_entry_loss_appends_every_buffered_field_once_per_call() -> None:
    torch.manual_seed(11)
    buf = trainer._PriorMatchAccumBuffer()

    trainer._hierarchical_entry_loss(
        _hier_out(6),
        _hier_batch(6),
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
        prior_match_buffer=buf,
    )
    assert buf.trade_logit.shape[0] == 6
    assert buf.trade_y.shape[0] == 6
    assert buf.side_logits.shape[0] == 6
    assert buf.side_y.shape[0] == 6
    assert buf.side_mask.shape[0] == 6
    assert buf.ctx_cat.shape[0] == 6

    trainer._hierarchical_entry_loss(
        _hier_out(6),
        _hier_batch(6),
        torch.device("cpu"),
        trade_pos_weight=1.0,
        side_bad_path_pos_weight=1.0,
        prior_match_buffer=buf,
    )
    assert buf.trade_logit.shape[0] == 12
    assert buf.ctx_cat.shape[0] == 12
