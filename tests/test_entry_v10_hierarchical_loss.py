from __future__ import annotations

import pytest
import torch

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


# The Entry hierarchy heads (trade_logit / side_logits / side_utility /
# side_bad_path_logit / side_validity_logit) and their
# ``_hierarchical_entry_task_losses`` owner are retired: the Entry action
# target is the frozen fitted-Q teacher (``entry_action_q_bps``) and the
# decision authority is its unique argmax.  What survives here is the
# registry event auxiliary, whose loss must stay outcome-only.


def _event_batch() -> dict[str, torch.Tensor]:
    return {
        "ctx_cat": torch.zeros((1, MODEL_NATIVE_CTX_CAT_DIM), dtype=torch.long),
        "y_line_support_touch_held": torch.tensor([1.0], dtype=torch.float32),
        "y_line_support_touch_mask": torch.tensor([1.0], dtype=torch.float32),
        "y_line_resistance_touch_held": torch.tensor([0.0], dtype=torch.float32),
        "y_line_resistance_touch_mask": torch.tensor([0.0], dtype=torch.float32),
        "y_countertrend_short_trap": torch.tensor([1.0], dtype=torch.float32),
        "y_countertrend_long_trap": torch.tensor([0.0], dtype=torch.float32),
    }


def _event_width() -> int:
    return int(
        trainer._MODEL_NATIVE_ACTIVE_OUTPUT_WIDTHS["trendline_event_logits"]
    )


def test_trendline_event_aux_loss_supervises_exact_label_contract() -> None:
    width = _event_width()
    # Targets are (support_held, resistance_held, short_trap, long_trap) =
    # (1, 0, 1, 0); a confidently correct prediction must drive the masked
    # BCE to ~0.
    logits = torch.tensor([[8.0, -8.0, 8.0, -8.0]], dtype=torch.float32)
    assert logits.shape[1] == width

    loss, stats = trainer._trendline_event_aux_loss(
        {"trendline_event_logits": logits},
        _event_batch(),
        torch.device("cpu"),
    )

    assert float(loss.detach().cpu().item()) < 0.001
    assert stats["trendline_event_rows"] == pytest.approx(1.0)
    assert stats["trendline_support_rows"] == pytest.approx(1.0)
    assert stats["trendline_resistance_rows"] == pytest.approx(0.0)


def test_trendline_event_aux_loss_ignores_direction_and_utility_outputs() -> None:
    event_logits = torch.zeros((1, _event_width()), dtype=torch.float32)
    out_long = {
        "trendline_event_logits": event_logits,
        "entry_action_q_bps": torch.tensor(
            [[20.0, -20.0, -20.0]], dtype=torch.float32
        ),
        "position_size_logit": torch.tensor([[20.0]], dtype=torch.float32),
    }
    out_short = {
        "trendline_event_logits": event_logits,
        "entry_action_q_bps": torch.tensor(
            [[-20.0, 20.0, -20.0]], dtype=torch.float32
        ),
        "position_size_logit": torch.tensor([[-20.0]], dtype=torch.float32),
    }

    long_loss, long_stats = trainer._trendline_event_aux_loss(
        out_long, _event_batch(), torch.device("cpu")
    )
    short_loss, short_stats = trainer._trendline_event_aux_loss(
        out_short, _event_batch(), torch.device("cpu")
    )

    assert float(long_loss.detach().cpu().item()) == pytest.approx(
        float(short_loss.detach().cpu().item())
    )
    assert long_stats == short_stats


def test_trendline_event_aux_loss_fails_closed_on_incomplete_contract() -> None:
    width = _event_width()
    batch = _event_batch()
    del batch["y_countertrend_long_trap"]

    with pytest.raises(
        RuntimeError,
        match=(
            "ENTRY_MODEL_NATIVE_ACTIVE_HEAD_TARGET_MISSING.*"
            "y_countertrend_long_trap"
        ),
    ):
        trainer._trendline_event_aux_loss(
            {
                "trendline_event_logits": torch.zeros(
                    (1, width), dtype=torch.float32
                )
            },
            batch,
            torch.device("cpu"),
        )

    with pytest.raises(
        RuntimeError, match="ENTRY_TRENDLINE_EVENT_OUTPUT_DIM_MISMATCH"
    ):
        trainer._trendline_event_aux_loss(
            {
                "trendline_event_logits": torch.zeros(
                    (1, width - 1), dtype=torch.float32
                )
            },
            _event_batch(),
            torch.device("cpu"),
        )


def test_trendline_event_loss_is_masked_on_registry_touch_rows_only() -> None:
    # The two line-hold dims are supervised ONLY on registry touch-event rows.
    # Flipping the support prediction to a wrong answer must not move the loss
    # once that row carries no support touch event.
    logits = torch.tensor([[8.0, -8.0, 8.0, -8.0]], dtype=torch.float32)
    flipped = torch.tensor([[-8.0, -8.0, 8.0, -8.0]], dtype=torch.float32)
    batch = _event_batch()
    batch["y_line_support_touch_mask"] = torch.tensor(
        [0.0], dtype=torch.float32
    )

    masked_loss, _ = trainer._trendline_event_aux_loss(
        {"trendline_event_logits": logits}, batch, torch.device("cpu")
    )
    flipped_loss, _ = trainer._trendline_event_aux_loss(
        {"trendline_event_logits": flipped}, batch, torch.device("cpu")
    )
    assert float(masked_loss.detach().cpu().item()) == pytest.approx(
        float(flipped_loss.detach().cpu().item())
    )

    # On a genuine touch row the same flip is penalised.
    supervised_loss, _ = trainer._trendline_event_aux_loss(
        {"trendline_event_logits": flipped}, _event_batch(), torch.device("cpu")
    )
    assert float(supervised_loss.detach().cpu().item()) > float(
        flipped_loss.detach().cpu().item()
    )


def test_retired_entry_hierarchy_heads_cannot_reenter_the_trainer() -> None:
    for retired in (
        "_hierarchical_entry_task_losses",
        "_trendline_rail_aux_loss",
    ):
        assert not hasattr(trainer, retired)
    for retired_output in (
        "trade_logit",
        "side_logits",
        "side_utility",
        "side_bad_path_logit",
        "side_validity_logit",
        "trendline_rail_logits",
    ):
        assert retired_output not in trainer._MODEL_NATIVE_ACTIVE_OUTPUT_WIDTHS
