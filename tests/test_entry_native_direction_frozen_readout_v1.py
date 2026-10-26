"""Synthetic frozen-head mechanics; no market payload or native model."""
import numpy as np
import pytest
from gx1.contracts import entry_native_direction_frozen_readout_v1 as owner


def sample():
    rng = np.random.RandomState(20260911)
    n = 1024
    hidden = rng.randint(-4, 5, size=(n, owner.WIDTH)).astype(np.float32)
    hidden[:, -2:] = np.asarray([0., 4.], np.float32)
    signs = []
    for j in range(3):
        label = hidden[:, j] > 0
        flipped = rng.uniform(size=n) < .4
        signs.append(np.where(label ^ flipped, 1, -1).astype(np.int8))
    sign = np.column_stack(signs)
    observed = np.ones((n, 3), np.bool_)
    observed[::11] = False
    sign[~observed] = 0
    sign[::17] = 0
    decision = 1714521600000000000 + (np.arange(n, dtype=np.int64) * 5 + 5) * owner.MINUTE_NS
    terminal = decision[:, None] + np.asarray(owner.HORIZONS, np.int64)[None, :] * owner.MINUTE_NS
    return hidden, {'parent_rows': np.arange(n, dtype=np.int64), 'sign': sign,
        'observed': observed, 'decision_ns': decision, 'terminal_time_ns': terminal,
        'cutoff_ns': int(decision[-1] + owner.MINUTE_NS),
        'original_weight': rng.normal(0, .02, size=(3, owner.WIDTH)).astype(np.float32),
        'original_bias': np.asarray([.1, -.1, .05], np.float32), 'split': 'train'}


def test_bce_gradient_matches_finite_differences():
    x = np.asarray([[1., 2., -1.], [0., -3., 1.], [-1., 1., 2.], [2., 0., -2.]])
    y = np.asarray([0., 1., 1., 0.])
    theta = np.asarray([.2, -.3, .1, .4])
    _, gradient = owner.binary_bce_gradient(theta, x, y)
    numerical = []
    for j in range(len(theta)):
        shift = np.zeros_like(theta); shift[j] = 1e-6
        numerical.append((owner.binary_bce_gradient(theta + shift, x, y)[0]
                          - owner.binary_bce_gradient(theta - shift, x, y)[0]) / 2e-6)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)


def test_extreme_logits_stay_finite_without_clipping():
    x = np.asarray([[-1.], [1.]])
    loss, gradient = owner.binary_bce_gradient(np.asarray([1000., 0.]), x, np.asarray([1., 0.]))
    assert loss == 1000. and np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient, [1., 0.])


def test_zero_feature_prior_is_stationary():
    x = np.zeros((6, 2), np.float64)
    y = np.asarray([0., 0., 0., 0., 1., 1.])
    loss, gradient = owner.binary_bce_gradient(np.asarray([0., 0., np.log(.5)]), x, y)
    expected = -(np.log(1 / 3) / 3 + 2 * np.log(2 / 3) / 3)
    assert abs(loss - expected) < 1e-15
    assert np.max(np.abs(gradient)) < 1e-15


def test_variance_contraction_keeps_affine_coordinates():
    hidden, _ = sample()
    contracted = hidden * np.float32(.125) + np.float32(2.)
    original = owner.transform_hidden(hidden, owner.fit_prefix_coordinates(hidden))
    transformed = owner.transform_hidden(contracted, owner.fit_prefix_coordinates(contracted))
    np.testing.assert_allclose(original, transformed, rtol=0, atol=1e-13)


def test_original_head_coordinate_algebra():
    hidden, args = sample()
    coordinates = owner.fit_prefix_coordinates(hidden)
    coef, intercept = owner.native_head_coordinates(args['original_weight'], args['original_bias'], coordinates)
    transformed = owner.transform_hidden(hidden, coordinates) @ coef.T + intercept
    raw = hidden.astype(np.float64) @ args['original_weight'].astype(np.float64).T + args['original_bias']
    np.testing.assert_allclose(transformed, raw, rtol=1e-12, atol=1e-12)


def test_target_ties_missing_and_exact_maturity():
    hidden, a = sample()
    eligible = owner.eligible_prefix_labels(a['parent_rows'], a['sign'], a['observed'],
        a['decision_ns'], a['terminal_time_ns'], cutoff_ns=a['cutoff_ns'])
    expected = a['observed'] & (a['sign'] != 0) & (a['terminal_time_ns'] < a['cutoff_ns'])
    np.testing.assert_array_equal(eligible, expected)
    assert not eligible[-1].any()
    assert np.any(a['observed'] & (a['sign'] == 0))
    assert np.any(~a['observed'])


@pytest.mark.parametrize('case,error', [
    ('oof', 'TRAIN_ONLY'), ('parent_order', 'PREFIX_PARENT_IDENTITY'),
    ('hidden_nan', 'HIDDEN_FP32_128'), ('hidden_dtype', 'HIDDEN_FP32_128'),
    ('missing_sign', 'SIGN_MISSING_STORAGE'), ('terminal_shift', 'TARGET_CLOCK'),
    ('decision_cutoff', 'DECISION_CUTOFF'), ('cutoff_after_train', 'PREFIX_CUTOFF'),
    ('single_class', 'PREFIX_BOTH_CLASSES'), ('weight_nonfinite', 'NATIVE_HEAD_FINITE'),
])
def test_rejects_invalid_prefix_before_fitting(case, error):
    hidden, a = sample()
    if case == 'oof': a['split'] = 'oof'
    elif case == 'parent_order': a['parent_rows'] = a['parent_rows'][::-1].copy()
    elif case == 'hidden_nan': hidden[0, 0] = np.nan
    elif case == 'hidden_dtype': hidden = hidden.astype(np.float64)
    elif case == 'missing_sign': a['sign'][0, 0] = 1
    elif case == 'terminal_shift': a['terminal_time_ns'][0, 0] += owner.MINUTE_NS
    elif case == 'decision_cutoff': a['cutoff_ns'] = int(a['decision_ns'][-1])
    elif case == 'cutoff_after_train': a['cutoff_ns'] = owner.TRAIN_END_NS + 1
    elif case == 'single_class': a['observed'][:, 0] = True; a['sign'][:, 0] = 1
    elif case == 'weight_nonfinite': a['original_weight'][0, 0] = np.inf
    with pytest.raises(RuntimeError, match='FROZEN_READOUT_' + error):
        owner.fit_frozen_direction_readout(hidden, **a)


@pytest.fixture(scope='module')
def paired_fits():
    hidden, a = sample()
    original = owner.fit_frozen_direction_readout(hidden, **a)
    contracted = hidden * np.float32(.125) + np.float32(2.)
    changed = owner.fit_frozen_direction_readout(contracted, **a)
    return hidden, contracted, original, changed


def test_six_fits_have_better_bce_and_small_gradients(paired_fits):
    for state in paired_fits[2:]:
        assert state['readout_fit_calls'] == 3 and state['native_optimizer_steps'] == 0
        assert state['predictive_value_qualified'] is False
        for d in state['fit_diagnostics']:
            assert d['fit_mean_bce'] <= min(d['prior_mean_bce'], d['original_head_mean_bce']) + 1e-10
            assert d['gradient_infinity_norm'] <= owner.GRADIENT_MAX
            assert d['objective_evaluations'] <= owner.SOLVER_OPTIONS['maxfun']
            assert d['native_coordinate_max_error'] <= d['native_coordinate_rounding_bound']


def test_pure_contraction_does_not_fake_prediction_loss(paired_fits):
    original = owner.predict_frozen_direction_readout(paired_fits[0], paired_fits[2])
    changed = owner.predict_frozen_direction_readout(paired_fits[1], paired_fits[3])
    np.testing.assert_allclose(original, changed, rtol=1e-7, atol=1e-7)


def test_fitted_state_arrays_are_readonly(paired_fits):
    state = paired_fits[2]
    for name in ('coefficient', 'intercept'):
        assert not state[name].flags.writeable
    for name in ('center', 'scale', 'constant'):
        assert not state['coordinates'][name].flags.writeable


def test_later_predictions_reuse_prefix_statistics(paired_fits):
    state = paired_fits[2]
    before = {k: state['coordinates'][k].tobytes() for k in ('center', 'scale', 'constant')}
    later = np.full((32, owner.WIDTH), 20., np.float32)
    prediction = owner.predict_frozen_direction_readout(later, state)
    assert prediction.shape == (32, 3) and np.isfinite(prediction).all()
    assert before == {k: state['coordinates'][k].tobytes() for k in before}
    assert state['coordinates']['fit_rows'] == 1024


def test_invalid_fitted_state_is_rejected(paired_fits):
    broken = dict(paired_fits[2], coefficient=np.zeros((3, 127), np.float64))
    with pytest.raises(RuntimeError, match='FROZEN_READOUT_FIT_COEFFICIENT'):
        owner.predict_frozen_direction_readout(paired_fits[0], broken)
