"""Synthetic affine arithmetic only; no archives, checkpoints or models."""
import json
import math

import numpy as np
import pytest

from gx1.audit import ta_head_representation_accounting_v1 as owner


def inputs():
    h0, h1 = np.zeros((2, 128), np.float32), np.zeros((2, 128), np.float32)
    h0[:, 0], h1[:, 0] = [1, 3], [2, 5]
    w0, w1 = np.zeros(128, np.float32), np.zeros(128, np.float32)
    w0[0], w1[0] = 2, 4
    return [h0, h1, w0, w1, np.array([.5], np.float32),
            np.array([1.5], np.float32), np.array([2.5, 6.5], np.float32),
            np.array([9.5, 21.5], np.float32)]


def test_hand_computed_bilinear_terms_interaction_and_four_levels():
    data = inputs()
    for value in data:
        value.setflags(write=False)
    before = [(v.shape, v.dtype, v.tobytes(), v.flags.writeable) for v in data]
    random_before = np.random.get_state()
    result = owner.account(*data)
    assert result["rows"] == 2 and result["width"] == 128
    assert result["actual_delta"] == dict(mean=11., std=4., rms=math.sqrt(137),
                                          minimum=7., maximum=15.)
    expected = {
        "head_first": (4., 6.),
        "representation_first": (7., 3.),
        "symmetric": (5.5, 4.5),
    }
    for order, (weight_mean, representation_mean) in expected.items():
        terms = result["decompositions"][order]
        assert terms["bias"]["mean"] == 1.
        assert terms["head_weights"]["mean"] == weight_mean
        assert terms["representation"]["mean"] == representation_mean
        assert terms["rounding"]["rms"] == 0.
    assert result["interaction"]["mean"] == 3.
    assert result["interaction"]["minimum"] == 2.
    assert result["interaction"]["maximum"] == 4.
    logits = {
        "INITIAL_REPR_INITIAL_HEAD": (2.5, 6.5),
        "INITIAL_REPR_FINAL_HEAD": (5.5, 13.5),
        "FINAL_REPR_INITIAL_HEAD": (4.5, 10.5),
        "FINAL_REPR_FINAL_HEAD": (9.5, 21.5),
    }
    assert set(result["combinations"]) == set(logits)
    for name, levels in logits.items():
        got = result["combinations"][name]
        assert got["mean_logit"] == sum(levels) / 2
        expected_p = sum(1 / (1 + math.exp(-x)) for x in levels) / 2
        # Independent exp/logaddexp sigmoid formulas may differ by a few FP64 ulps.
        assert got["mean_p_up"] == pytest.approx(
            expected_p, rel=0., abs=4 * np.finfo(np.float64).eps)
    assert result["norms"]["weights"] == dict(initial_l2=2., final_l2=4.,
        delta_l2=2., max_abs_delta=2., cosine=1.)
    assert result["norms"]["representations"]["delta_l2"] == math.sqrt(5)
    assert result["max_replay_error"] == result["max_closure_error"] == 0.
    assert result["historical_causal_attribution"] is False
    assert [(v.shape, v.dtype, v.tobytes(), v.flags.writeable) for v in data] == before
    random_after = np.random.get_state()
    assert random_before[0] == random_after[0]
    assert np.array_equal(random_before[1], random_after[1])
    assert random_before[2:] == random_after[2:]
    json.dumps(result, allow_nan=False)


def test_identity_has_zero_allocations_and_defined_zero_norm_behavior():
    data = inputs()
    for value in data:
        value.fill(0.)
    result = owner.account(*data)
    assert result["actual_delta"]["rms"] == 0.
    assert all(term["rms"] == 0. for parts in result["decompositions"].values()
               for term in parts.values())
    assert all(v == dict(mean_logit=0., mean_p_up=.5)
               for v in result["combinations"].values())
    assert result["norms"]["weights"]["cosine"] is None
    assert result["norms"]["representations"]["cosine"] is None
    assert result["max_closure_error"] == result["max_replay_error"] == 0.
    json.dumps(result, allow_nan=False)


def test_bias_only_has_no_weight_or_representation_contribution():
    data = inputs()
    data[1], data[3] = data[0].copy(), data[2].copy()
    data[5] = np.array([-.5], np.float32)
    data[7] = np.array([1.5, 5.5], np.float32)
    result = owner.account(*data)
    assert result["actual_delta"]["mean"] == -1.
    for terms in result["decompositions"].values():
        assert terms["bias"]["mean"] == -1.
        for key in ("head_weights", "representation", "rounding"):
            assert terms[key]["rms"] == 0.
    assert result["interaction"]["rms"] == 0.


def test_rounding_residual_closes_to_native_logits_not_reconstructed_endpoints():
    data = inputs()
    data[0].fill(0.); data[1].fill(0.)
    data[0][:, 0], data[1][:, 0] = 1., 2.
    data[2].fill(0.); data[3].fill(0.)
    data[2][0] = data[3][0] = .25
    data[4].fill(0.); data[5].fill(0.)
    data[6].fill(np.nextafter(np.float32(.25), np.float32(1.)))
    data[7].fill(np.nextafter(np.float32(.5), np.float32(0.)))
    result = owner.account(*data)
    expected_rounding = (float(data[7][0]) - .5) - (float(data[6][0]) - .25)
    assert expected_rounding != 0.
    for terms in result["decompositions"].values():
        assert terms["rounding"]["mean"] == expected_rounding
        assert terms["representation"]["mean"] == .25
    assert result["actual_delta"]["mean"] == .25 + expected_rounding
    assert result["max_closure_error"] == 0.
    assert result["retained_endpoints"]["INITIAL"]["mean_logit"] != .25
    assert result["combinations"]["INITIAL_REPR_INITIAL_HEAD"]["mean_logit"] == .25


@pytest.mark.parametrize("phase, delta", [(6, 8e-6), (7, 2e-5)])
def test_native_replay_mismatch_rejects_both_endpoints(phase, delta):
    data = inputs()
    original = float(data[phase][0])
    data[phase][0] += np.float32(delta)
    error = float(data[phase][0]) - original
    assert error < owner.FP32_GAMMA129 * original
    assert error > owner.REPLAY_ATOL + owner.REPLAY_RTOL * float(data[phase][0])
    with pytest.raises(RuntimeError, match="NATIVE_REPLAY"):
        owner.account(*data)


def test_gamma_bound_is_required_even_when_absolute_tolerance_would_pass():
    data = inputs()
    for value in data:
        value.fill(0.)
    data[6][0] = np.nextafter(np.float32(0.), np.float32(1.))
    assert float(data[6][0]) < owner.REPLAY_ATOL
    with pytest.raises(RuntimeError, match="FP32_ENVELOPE:INITIAL"):
        owner.account(*data)


@pytest.mark.parametrize("fault", ["empty", "width", "rows", "weight", "bias", "logits",
                                    "dtype", "list", "nan", "infinity"])
def test_invalid_shapes_dtypes_and_values_reject_before_accounting(fault):
    data = inputs()
    if fault == "empty": data[0] = np.empty((0, 128), np.float32)
    elif fault == "width": data[0] = np.zeros((2, 127), np.float32)
    elif fault == "rows": data[1] = np.zeros((3, 128), np.float32)
    elif fault == "weight": data[2] = data[2][None, :]
    elif fault == "bias": data[4] = data[4][0]
    elif fault == "logits": data[6] = data[6][:, None]
    elif fault == "dtype": data[3] = data[3].astype(np.float64)
    elif fault == "list": data[5] = [1.5]
    elif fault == "nan": data[1][0, 7] = np.nan
    elif fault == "infinity": data[7][0] = np.inf
    with pytest.raises(RuntimeError, match="TA_HEAD_REPRESENTATION_"):
        owner.account(*data)
