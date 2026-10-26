import math

import numpy as np
import pytest
from scipy.special import expit

from gx1.contracts.entry_direction_prior_offset_v1 import solve_prefix_prior_offset


@pytest.mark.parametrize("constant,p", [(-4., .2), (0., .5), (4., .8),
                                      (1000., .01), (-1000., .99)])
def test_constant_logits_equal_analytic_prior_logit(constant, p):
    z = np.full(19, constant)
    original = z.tobytes()
    state = solve_prefix_prior_offset(z, p, partition="train")
    expected = math.log(p) - math.log1p(-p) - constant
    assert state["offset"] == pytest.approx(expected, abs=2e-13)
    assert state["mean_probability_after"] == pytest.approx(p, abs=8e-15)
    assert state["bisection_steps"] == 64
    assert z.tobytes() == original


def test_heterogeneous_predictions_recover_known_common_offset():
    z = np.array([-3., -1., -.1, .4, 2., 5.])
    target = float(expit(z + .35).mean())
    state = solve_prefix_prior_offset(z, target, partition="train")
    assert state["offset"] == pytest.approx(.35, abs=2e-14)


def test_offset_preserves_all_pair_orderings_and_ties():
    z = np.array([-.1, .2, -.1, .7, -.4, .2])
    state = solve_prefix_prior_offset(z, .6, partition="train")
    shifted = z + state["offset"]
    np.testing.assert_array_equal(np.sign(z[:, None] - z),
                                  np.sign(shifted[:, None] - shifted))


@pytest.mark.parametrize("z", [
    np.array([0., 1.], np.float32), np.zeros((2, 1)),
    np.array([0., np.nan]), np.array([0., np.inf]),
    np.arange(8., dtype=np.float64)[::2], np.zeros(1),
])
def test_invalid_prefix_rejected(z):
    with pytest.raises(ValueError, match="PREFIX_LOGITS"):
        solve_prefix_prior_offset(z, .5, partition="train")


@pytest.mark.parametrize("p", [0., 1., float("nan"), float("inf"), True])
def test_invalid_prior_rejected(p):
    with pytest.raises(ValueError, match="PRIOR_OFFSET_PRIOR"):
        solve_prefix_prior_offset(np.zeros(2), p, partition="train")


def test_nontrain_calibration_rejected():
    with pytest.raises(ValueError, match="TRAIN_ONLY"):
        solve_prefix_prior_offset(np.zeros(2), .5, partition="later")


def test_unrepresentable_shift_fails_instead_of_claiming_calibration():
    with pytest.raises(ValueError, match="NUMERICAL_RESIDUAL"):
        solve_prefix_prior_offset(np.full(3, 1e20), .4, partition="train")
