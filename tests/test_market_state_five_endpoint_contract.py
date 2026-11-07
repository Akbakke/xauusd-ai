"""Five native endpoint masks, gradient, and proper-score denominators."""
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gx1.contracts.market_state_five_endpoint_contract_v1 import (
    CLASS_NAMES, LEAD_MINUTES, TIMEFRAMES, allowed_set_loss,
    five_endpoint_contract, proper_score_sums,
)


def tensors(bits=(1, 2, 4, 5, 7)):
    logits = torch.zeros((1, 5, 3), dtype=torch.float32, requires_grad=True)
    allowed = torch.tensor([bits], dtype=torch.uint8)
    eligible = torch.ones((1, 5), dtype=torch.bool)
    mature = torch.ones((1, 5), dtype=torch.bool)
    return logits, allowed, eligible, mature


def test_contract_and_all_five_allowed_sets_have_exact_loss_denominator():
    c = five_endpoint_contract()
    assert c["timeframes"] == list(TIMEFRAMES)
    assert c["future_validity_minutes_after_closed_M5_issue"] == list(LEAD_MINUTES)
    assert c["class_order"] == list(CLASS_NAMES) == ["DOWN", "FLAT", "UP"]
    assert len(c["contract_sha256"]) == 64
    logits, allowed, eligible, mature = tensors()
    result = allowed_set_loss(logits, allowed, eligible, mature)
    assert result["count"] == 4 and result["skip_update"] is False
    assert result["per_tf_counts"] == dict(zip(TIMEFRAMES, (1, 1, 1, 1, 0)))
    assert result["loss"].item() == pytest.approx((3 * math.log(3) + math.log(1.5)) / 4)
    result["loss"].backward()
    assert logits.grad is not None
    assert torch.all(logits.grad[0, 4] == 0)
    assert torch.all(torch.isfinite(logits.grad))


def test_maturity_and_eligibility_exclude_only_their_cells():
    logits, allowed, eligible, mature = tensors()
    mature[0, 0] = False
    eligible[0, 1] = False
    result = allowed_set_loss(logits, allowed, eligible, mature)
    assert result["count"] == 2
    assert result["per_tf_counts"] == dict(zip(TIMEFRAMES, (0, 0, 1, 1, 0)))
    assert result["loss"].item() == pytest.approx((math.log(3) + math.log(1.5)) / 2)
    assert not result["informative_mask"][0, 0]
    assert not result["informative_mask"][0, 1]


def test_active_mask_has_no_conditional_side_preference():
    side = torch.tensor(0.7, dtype=torch.float32, requires_grad=True)
    zero = torch.zeros((), dtype=torch.float32)
    # exp(DOWN)+exp(UP) stays one while their conditional split changes.
    row = torch.stack((F.logsigmoid(side), zero, F.logsigmoid(-side)))
    logits = row.view(1, 1, 3).expand(1, 5, 3)
    allowed = torch.tensor([[5, 7, 7, 7, 7]], dtype=torch.uint8)
    eligible = mature = torch.ones((1, 5), dtype=torch.bool)
    loss = allowed_set_loss(logits, allowed, eligible, mature)
    assert loss["loss"].item() == pytest.approx(math.log(2), abs=1e-6)
    loss["loss"].backward()
    assert side.grad is not None and abs(float(side.grad)) < 1e-6


def test_empty_support_is_zero_anchored_and_skips_update():
    logits, allowed, eligible, mature = tensors((7, 7, 7, 7, 7))
    result = allowed_set_loss(logits, allowed, eligible, mature)
    assert result["count"] == 0 and result["skip_update"] is True
    assert result["loss"].item() == 0 and result["loss"].requires_grad
    result["loss"].backward()
    assert logits.grad is not None and torch.all(logits.grad == 0)
    logits, allowed, eligible, mature = tensors()
    mature[:] = False
    assert allowed_set_loss(logits, allowed, eligible, mature)["skip_update"] is True


def test_score_rows_and_sums_are_separate_per_timeframe():
    p = np.array([
        [[.6, .2, .2], [.1, .8, .1], [.2, .3, .5], [.3, .1, .6], [.2, .4, .4]],
        [[.1, .3, .6], [.2, .2, .6], [.4, .2, .4], [.5, .2, .3], [.1, .7, .2]],
    ], dtype=np.float64)
    allowed = np.array([[1, 2, 4, 5, 7], [4, 7, 5, 1, 2]], dtype=np.uint8)
    eligible = np.ones((2, 5), dtype=np.bool_)
    mature = np.ones((2, 5), dtype=np.bool_)
    eligible[1, 3] = False
    mature[1, 2] = False
    score = proper_score_sums(p, allowed, eligible, mature)
    assert score["contract_sha256"] == five_endpoint_contract()["contract_sha256"]
    s = score["per_tf"]
    assert {tf: tuple(s[tf][metric]["count"] for metric in
                      ("activity", "conditional_side", "three_class")) for tf in TIMEFRAMES} == {
        "M5": (2, 2, 2), "M15": (1, 0, 1), "H1": (1, 1, 1),
        "H4": (1, 0, 0), "D1": (1, 0, 1)}
    assert s["M5"]["activity"]["brier_sum"] == pytest.approx(.04 + .09)
    assert s["M5"]["three_class"]["brier_sum"] == pytest.approx(.24 + .26)
    assert s["M5"]["three_class"]["log_sum"] == pytest.approx(-2 * math.log(.6))
    assert s["M5"]["conditional_side"]["brier_sum"] == pytest.approx(.25**2 + (1/7)**2)
    assert s["M5"]["conditional_side"]["log_sum"] == pytest.approx(-math.log(.75) - math.log(6/7))
    assert s["H4"]["activity"]["count"] == 1
    assert s["H4"]["conditional_side"]["count"] == 0
    assert s["D1"]["activity"]["brier_sum"] == pytest.approx((.3 - 0)**2)


def test_extreme_down_side_log_uses_direct_true_probability():
    p = np.full((1, 5, 3), 1/3, dtype=np.float64)
    p[0, 0] = [1e-20, 0.0, 1.0]
    bits = np.array([[1, 7, 7, 7, 7]], dtype=np.uint8)
    known = np.ones((1, 5), dtype=np.bool_)
    side = proper_score_sums(p, bits, known, known)["per_tf"]["M5"]["conditional_side"]
    assert side["count"] == 1
    assert side["log_sum"] == pytest.approx(-math.log(1e-20), abs=1e-12)
    assert side["zero_truth_probability_rows"] == 0


def test_invalid_inputs_fail_and_zero_truth_probability_is_visible():
    logits, allowed, eligible, mature = tensors()
    allowed[0, 0] = 3
    with pytest.raises(ValueError, match="ALLOWED"):
        allowed_set_loss(logits, allowed, eligible, mature)
    allowed[0, 0] = 1
    logits_bad = logits.detach().clone()
    logits_bad[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="LOGITS"):
        allowed_set_loss(logits_bad, allowed, eligible, mature)
    p = np.full((1, 5, 3), 1/3, dtype=np.float64)
    bits = np.full((1, 5), 7, dtype=np.uint8)
    e = m = np.ones((1, 5), dtype=np.bool_)
    p[0, 0] = [0.0, .5, .5]
    bits[0, 0] = 1
    score = proper_score_sums(p, bits, e, m)["per_tf"]["M5"]
    assert math.isinf(score["three_class"]["log_sum"])
    assert score["three_class"]["zero_truth_probability_rows"] == 1
    p[0, 1] = [.4, .4, .4]
    with pytest.raises(ValueError, match="PROBABILITIES"):
        proper_score_sums(p, bits, e, m)
    p[0, 1] = [1/3, 1/3, 1/3]
    p[0, 0] = [0.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="UNDEFINED_CONDITIONAL_SIDE"):
        proper_score_sums(p, bits, e, m)
