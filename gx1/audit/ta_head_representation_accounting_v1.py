"""Pure affine endpoint accounting from retained FP32 arrays; no model or IO.

Ordered and symmetric allocations are algebraic, not unique historical causes.
The caller qualifies row, model and channel identities before calling account.
"""
from __future__ import annotations

import numpy as np

WIDTH = 128
REPLAY_RTOL = 1e-6
REPLAY_ATOL = 2e-6
CLOSURE_ATOL = 1e-12
FP32_GAMMA129 = 129 * float(np.finfo(np.float32).eps) / (1 - 129 * float(np.finfo(np.float32).eps))
COMBINATIONS = (
    "INITIAL_REPR_INITIAL_HEAD", "INITIAL_REPR_FINAL_HEAD",
    "FINAL_REPR_INITIAL_HEAD", "FINAL_REPR_FINAL_HEAD",
)


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError("TA_HEAD_REPRESENTATION_" + reason)


def _moments(values):
    _need(np.isfinite(values).all(), "FINITE_ACCOUNTING")
    return dict(mean=float(values.mean()), std=float(values.std()),
                rms=float(np.sqrt(np.mean(values * values))),
                minimum=float(values.min()), maximum=float(values.max()))


def _level(values):
    return dict(mean_logit=float(values.mean()),
                mean_p_up=float(np.exp(-np.logaddexp(0., -values)).mean()))


def _norms(initial, final):
    delta = final - initial
    a, b = float(np.linalg.norm(initial)), float(np.linalg.norm(final))
    return dict(initial_l2=a, final_l2=b, delta_l2=float(np.linalg.norm(delta)),
                max_abs_delta=float(np.abs(delta).max()),
                cosine=None if a == 0. or b == 0. else float(
                    np.sum(initial * final) / (a * b)))


def _replay(actual, calculated, representation, weight, bias, phase):
    error = actual - calculated
    envelope = FP32_GAMMA129 * (np.abs(representation) @ np.abs(weight) + abs(bias))
    tolerance = REPLAY_ATOL + REPLAY_RTOL * np.abs(actual)
    maximum = float(np.abs(error).max())
    scaled = float((np.abs(error) / tolerance).max())
    # Both fixed native tolerance and the qualified 129-operation FP32 bound
    # apply. A zero arithmetic envelope requires exactly zero replay error.
    _need(scaled <= 1., "NATIVE_REPLAY:" + phase)
    _need(np.all(np.abs(error) <= envelope), "FP32_ENVELOPE:" + phase)
    return error, dict(max_abs=maximum, max_scaled=scaled,
                       fp32_error_envelope_max=float(envelope.max()),
                       within_tolerance=True, within_fp32_error_envelope=True)


def account(initial_repr, final_repr, initial_weight, final_weight,
            initial_bias, final_bias, initial_logits, final_logits):
    """Return JSON-safe aggregate accounting for one caller-selected channel.

    Inputs are finite FP32 NumPy arrays: representations (N,128), weights
    (128,), biases (1,), and retained native logits (N,), with N > 0. No input
    is changed. Norms of representations are Frobenius norms over all rows.
    All four combinations use FP64 affine arithmetic; retained native endpoint
    means are separate. Explicit endpoint rounding closes each allocation to
    the actual retained logit change, rather than to a reconstructed endpoint.
    """
    values = (initial_repr, final_repr, initial_weight, final_weight,
              initial_bias, final_bias, initial_logits, final_logits)
    _need(all(isinstance(v, np.ndarray) and v.dtype == np.float32 for v in values),
          "FP32_ARRAYS")
    _need(initial_repr.ndim == 2 and initial_repr.shape[0] > 0
          and initial_repr.shape[1] == WIDTH, "REPRESENTATION_SHAPE")
    rows = initial_repr.shape[0]
    shapes = ((rows, WIDTH), (rows, WIDTH), (WIDTH,), (WIDTH,),
              (1,), (1,), (rows,), (rows,))
    _need(all(v.shape == shape for v, shape in zip(values, shapes)), "SHAPES")
    _need(all(np.isfinite(v).all() for v in values), "NONFINITE_INPUT")
    h0, h1, w0, w1, b0, b1, z0, z1 = (v.astype(np.float64) for v in values)
    bias0, bias1 = float(b0[0]), float(b1[0])
    affine = dict(zip(COMBINATIONS, (h0 @ w0 + bias0, h0 @ w1 + bias1,
                                    h1 @ w0 + bias0, h1 @ w1 + bias1)))
    error0, replay0 = _replay(z0, affine[COMBINATIONS[0]], h0, w0, bias0, "INITIAL")
    error1, replay1 = _replay(z1, affine[COMBINATIONS[3]], h1, w1, bias1, "FINAL")
    dh, dw = h1 - h0, w1 - w0
    db = np.full(rows, bias1 - bias0, dtype=np.float64)
    rounding, actual = error1 - error0, z1 - z0
    parts = dict(
        head_first=dict(bias=db, head_weights=h0 @ dw,
                        representation=dh @ w1, rounding=rounding),
        representation_first=dict(bias=db, head_weights=h1 @ dw,
                                  representation=dh @ w0, rounding=rounding),
        symmetric=dict(bias=db, head_weights=((h0 + h1) / 2) @ dw,
                       representation=dh @ ((w0 + w1) / 2), rounding=rounding),
    )
    closures = {name: float(np.abs(sum(terms.values()) - actual).max())
                for name, terms in parts.items()}
    maximum_closure = max(closures.values())
    _need(maximum_closure < CLOSURE_ATOL, "DECOMPOSITION_CLOSURE")
    return dict(
        rows=rows, width=WIDTH, arithmetic="FP64_from_retained_FP32",
        actual_delta=_moments(actual),
        decompositions={name: {term: _moments(value) for term, value in terms.items()}
                        for name, terms in parts.items()},
        interaction=_moments(dh @ dw),
        combinations={name: _level(value) for name, value in affine.items()},
        retained_endpoints=dict(INITIAL=_level(z0), FINAL=_level(z1)),
        norms=dict(weights=_norms(w0, w1), representations=_norms(h0, h1)),
        replay=dict(INITIAL=replay0, FINAL=replay1), closure_errors=closures,
        max_replay_error=max(replay0["max_abs"], replay1["max_abs"]),
        max_closure_error=maximum_closure,
        tolerances=dict(native_rtol=REPLAY_RTOL, native_atol=REPLAY_ATOL,
                        fp32_gamma129=FP32_GAMMA129, closure_atol=CLOSURE_ATOL),
        historical_causal_attribution=False,
    )
