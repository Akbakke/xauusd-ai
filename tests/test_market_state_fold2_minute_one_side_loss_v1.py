"""Known-side support and equal-observation minute-one loss checks."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from gx1.contracts import market_state_fold2_minute_one_side_loss_v1 as candidate
from gx1.contracts.market_state_minute_query_candidate_v1 import minute_allowed_set_loss


def _batch(known: int):
    raw = torch.zeros((16, 5, 2), dtype=torch.float32, requires_grad=True)
    allowed = torch.full((16, 5), 7, dtype=torch.uint8)
    allowed[:, 0] = 5
    allowed[:known, 0] = 4
    eligible = torch.ones((16, 5), dtype=torch.bool)
    query = torch.ones((16, 5), dtype=torch.int64)
    return raw, allowed, eligible, query


def test_fixed_fit_support_is_bound_to_completed_minute_source_review():
    path = Path("docs/S_FOLD2_MINUTE_QUERY_SOURCE_REVIEW_20261115.json")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == candidate.SUPPORT_REVIEW_SHA256
    review = json.loads(path.read_text())
    assert review["minute1_all_TF"]["fit"]
    assert {tf: value["known_side"] for tf, value in
            review["minute1_all_TF"]["fit"].items()} == candidate.KNOWN_SIDE_FIT
    assert candidate.FIT_BATCHES == 2959
    assert candidate.contract()["known_side_fit_rows"] == candidate.KNOWN_SIDE_FIT


def test_known_side_gradient_has_same_weight_in_sparse_and_dense_batches():
    gradients = []
    for known in (1, 8):
        raw, allowed, ready, query = _batch(known)
        got = candidate.allowed_set_loss(raw, allowed, ready, ready, query)
        old = minute_allowed_set_loss(raw, allowed, ready, ready, query)
        assert got["per_tf_side_counts"]["M5"] == known
        assert got["loss"] > old["loss"]
        got["loss"].backward()
        gradients.append(float(raw.grad[0, 0, 1]))
        assert torch.count_nonzero(raw.grad[known:, 0, 1]) == 0
        assert torch.count_nonzero(raw.grad[:, 1:, :]) == 0
    expected = -0.5 * candidate.FIT_BATCHES / candidate.KNOWN_SIDE_FIT["M5"] / 5
    assert gradients == pytest.approx([expected, expected], abs=1e-8)


def test_query_schedule_is_fixed_before_any_label():
    raw, allowed, ready, query = _batch(1)
    query[0, 0] = 2
    with pytest.raises(ValueError, match="FIXED_QUERY_OR_BATCH"):
        candidate.allowed_set_loss(raw, allowed, ready, ready, query)
