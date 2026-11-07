"""Real corrected TRAIN targets through the post-forward loss and score bridge."""
import math

import pytest
import torch

from gx1.contracts.market_state_five_endpoint_batch_v1 import evaluate_after_forward
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import (
    FOLD_BOUNDARIES_NS, load_bound_five_endpoint_targets,
)


def test_real_fit_row_preserves_active_unknown_side_and_separate_scores():
    bound = load_bound_five_endpoint_targets()
    logits = torch.zeros((1, 5, 3), dtype=torch.float32, requires_grad=True)
    result = evaluate_after_forward(logits, bound, bound.parent_rows[:1],
                                    bound.decision_ns[:1], fold_id=1,
                                    partition="fit")
    assert result["per_tf_counts"] == {"M5": 1, "M15": 1, "H1": 1,
                                        "H4": 1, "D1": 1}
    assert result["count"] == 5 and result["skip_update"] is False
    assert result["loss"].item() == pytest.approx((4 * math.log(1.5) + math.log(3)) / 5)
    for tf in ("M5", "M15", "H4", "D1"):
        assert result["score_sums"][tf]["activity"]["count"] == 1
        assert result["score_sums"][tf]["conditional_side"]["count"] == 0
        assert result["score_sums"][tf]["three_class"]["count"] == 0
    assert result["score_sums"]["H1"]["conditional_side"]["count"] == 1
    assert result["score_sums"]["H1"]["three_class"]["count"] == 1
    result["loss"].backward()
    assert logits.grad is not None and bool(torch.isfinite(logits.grad).all())
    for j in (0, 1, 3, 4):
        assert logits.grad[0, j, 0].item() == pytest.approx(logits.grad[0, j, 2].item())


def test_real_last_fit_row_has_empty_mature_support_and_skip_signal():
    bound = load_bound_five_endpoint_targets()
    at = int((bound.decision_ns < FOLD_BOUNDARIES_NS[0]).sum()) - 1
    assert FOLD_BOUNDARIES_NS[0] - int(bound.decision_ns[at]) == 300_000_000_000
    logits = torch.zeros((1, 5, 3), dtype=torch.float32, requires_grad=True)
    result = evaluate_after_forward(logits, bound, bound.parent_rows[at:at+1],
                                    bound.decision_ns[at:at+1], fold_id=1,
                                    partition="fit")
    assert result["count"] == 0 and result["skip_update"] is True
    assert result["loss"].item() == 0.0
    assert all(value == 0 for value in result["per_tf_counts"].values())
    assert all(metric["count"] == 0 for tf in result["score_sums"].values()
               for metric in tf.values())
