import numpy as np
import pandas as pd
import pytest

from gx1.audit.market_state_remaining_mtf_momentum_simple5_oracle_v1 import (
    CACHE_COLUMNS, FIELDS, closed_momentum_simple5_reference,
)


def _sample(n=120):
    rng = np.random.default_rng(20261107)
    close = 2000.0 + np.cumsum(rng.normal(0, 0.4, n))
    high = close + 0.8
    low = close - 0.8
    rsi = 50 + 28 * np.sin(np.arange(n) / 6)
    rsi[:14] = np.nan
    atr = np.full(n, 1.2)
    atr[:13] = np.nan
    return high, low, close, rsi, atr


def test_warmup_events_units_and_prefix_causality():
    data = _sample()
    got = closed_momentum_simple5_reference(*data)
    assert tuple(got) == FIELDS and CACHE_COLUMNS == (111, 112, 123, 124, 125)
    assert all(x.dtype == np.float32 for x in got.values())
    assert np.isnan(got[FIELDS[0]][:14]).all()
    assert np.isnan(got[FIELDS[2]][:25]).all()
    assert np.isnan(got[FIELDS[3]][:33]).all()
    assert np.isnan(got[FIELDS[4]][:13]).all()
    assert np.isfinite(got[FIELDS[0]][20:]).any()
    assert np.all((got[FIELDS[4]][13:] >= 0) & (got[FIELDS[4]][13:] <= 1))
    prefix = closed_momentum_simple5_reference(*(x[:83] for x in data))
    for field in FIELDS:
        np.testing.assert_array_equal(prefix[field], got[field][:83])


def test_independent_parity_with_production_momentum_owner():
    from gx1.features.htf_features import _compute_v29_momentum_event_frame

    data = _sample(5000)
    h, l, c, rsi, atr = data
    idx = pd.RangeIndex(len(c))
    observed = _compute_v29_momentum_event_frame(
        high=pd.Series(h, index=idx), low=pd.Series(l, index=idx),
        close=pd.Series(c, index=idx), rsi=pd.Series(rsi, index=idx),
        mom_20_atr=pd.Series(np.zeros(len(c)), index=idx),
        atr_positive=pd.Series(atr, index=idx))
    expected = closed_momentum_simple5_reference(*data)
    for field in FIELDS:
        np.testing.assert_array_equal(expected[field], observed[field].to_numpy(dtype=np.float32))


def test_invalid_suffix_and_geometry():
    h, l, c, rsi, atr = _sample()
    rsi[30] = np.nan
    with pytest.raises(ValueError, match='SOURCE_INVALID'):
        closed_momentum_simple5_reference(h, l, c, rsi, atr)
    rsi[30] = 50
    h[30] = c[30] - 1
    with pytest.raises(ValueError, match='SOURCE_INVALID'):
        closed_momentum_simple5_reference(h, l, c, rsi, atr)


def test_raw_rsi_threshold_and_float64_lineage():
    h, l, c, rsi, atr = _sample(60)
    rsi[14:] = 50.0
    rsi[20] = 69.999999
    rsi[25] = 70.0
    got = closed_momentum_simple5_reference(h, l, c, rsi, atr)
    assert np.isnan(got[FIELDS[0]][20])
    assert got[FIELDS[0]][25] == 0.0
    assert got[FIELDS[1]][25] == 1.0
    with pytest.raises(ValueError, match="RAW_F64_REQUIRED"):
        closed_momentum_simple5_reference(h, l, c, rsi.astype(np.float32), atr)
    rsi[20] = np.inf
    with pytest.raises(ValueError, match="SOURCE_INVALID"):
        closed_momentum_simple5_reference(h, l, c, rsi, atr)
