import numpy as np
import pytest

from gx1.audit.market_state_remaining_mtf_squeeze_oracle_v1 import (
    CACHE_COLUMNS, FIELDS, _decode_widths, closed_squeeze3_reference,
)

PARAMS = dict(means=[0.005, 0.05], variances=[0.0001, 0.0002],
              transition_probability=[[0.9, 0.1], [0.1, 0.9]],
              initial_probability=[0.5, 0.5])


def test_names_warmup_constant_and_causal_prefix():
    close = np.full(40, 2000.0)
    got = closed_squeeze3_reference(close, **PARAMS)
    assert tuple(got) == FIELDS and CACHE_COLUMNS == (26, 27, 28)
    assert all(x.dtype == np.float32 for x in got.values())
    for values in got.values():
        assert np.isnan(values[:19]).all()
    assert np.all(got[FIELDS[2]][19:] == 0.0)
    assert np.all(got[FIELDS[0]][19:] == np.arange(1, 22, dtype=np.float32))
    assert np.all(got[FIELDS[1]][19:] == np.arange(0, 21, dtype=np.float32))
    rng = np.random.default_rng(20261107)
    full_close = 2000.0 + np.cumsum(rng.normal(0, 0.5, 120))
    full = closed_squeeze3_reference(full_close, **PARAMS)
    prefix = closed_squeeze3_reference(full_close[:73], **PARAMS)
    for field in FIELDS:
        np.testing.assert_array_equal(prefix[field], full[field][:73])


def test_independent_probability_recursion_agrees_with_owner_log_odds():
    from gx1.features.volatility_squeeze_state_v1 import _advance_state_log_odds

    widths = np.array([0.004, 0.006, 0.055, 0.061, 0.004, 0.003])
    mu = np.array(PARAMS['means'])
    var = np.array(PARAMS['variances'])
    move = np.array(PARAMS['transition_probability'])
    start = np.array(PARAMS['initial_probability'])
    duration, age = _decode_widths(widths, mu, var, move, start)
    previous = None
    prior_active = None
    expected_duration = []
    expected_age = []
    bars = 0
    since_release = None
    for width in widths:
        density = -0.5 * (np.log(2 * np.pi * var) + (width - mu) ** 2 / var)
        previous = _advance_state_log_odds(
            previous, density[0] - density[1],
            log_transition=np.log(move), log_initial=np.log(start))
        active = previous >= 0
        since_release = 0 if since_release is None else since_release + 1
        if active:
            bars = bars + 1 if prior_active else 1
        else:
            if prior_active:
                since_release = 0
            bars = 0
        expected_duration.append(bars)
        expected_age.append(since_release)
        prior_active = active
    np.testing.assert_array_equal(duration, expected_duration)
    np.testing.assert_array_equal(age, expected_age)


def test_invalid_source_and_parameters_fail_closed():
    with pytest.raises(ValueError, match='CLOSED_CLOSE_INVALID'):
        closed_squeeze3_reference([100, 0], **PARAMS)
    with pytest.raises(ValueError, match='FROZEN_PARAMETERS_INVALID'):
        closed_squeeze3_reference([100] * 20, **{**PARAMS, 'initial_probability':[1, 0]})


def test_independent_bandwidth_matches_current_owner_on_closed_prices():
    from gx1.features.volatility_squeeze_state_v1 import bollinger_relative_bandwidth

    rng = np.random.default_rng(20261107)
    close = 2000 + np.cumsum(rng.normal(0, 0.8, 1200))
    expected = closed_squeeze3_reference(close, **PARAMS)[FIELDS[2]]
    observed = bollinger_relative_bandwidth(close).astype(np.float32)
    np.testing.assert_array_equal(expected, observed)
