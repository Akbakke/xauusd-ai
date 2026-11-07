import numpy as np
import pandas as pd
import pytest

from gx1.audit.market_state_remaining_mtf_local_returns_oracle_v1 import (
    CACHE_COLUMNS, FIELDS, closed_local_returns4_reference,
)


def test_warmup_units_and_flat_efficiency():
    close = np.full(40, 100.0)
    got = closed_local_returns4_reference(close)
    assert tuple(got) == FIELDS and CACHE_COLUMNS == (191, 192, 193, 228)
    assert all(x.dtype == np.float32 for x in got.values())
    assert np.isnan(got["ret_1"][0])
    assert np.isnan(got["ret_20"][:20]).all()
    assert np.isnan(got["rvol_20"][:20]).all()
    assert np.isnan(got["chart.local_kama_efficiency_30"]).all()
    assert np.all(got["ret_1"][1:] == 0.0)
    assert np.all(got["rvol_20"][20:] == 0.0)


def test_monotone_efficiency_and_prefix_causality():
    close = np.arange(100.0, 160.0)
    full = closed_local_returns4_reference(close)
    assert np.all(full["chart.local_kama_efficiency_30"][30:] == 1.0)
    assert full["ret_20"][20] == np.float32((120 / 100 - 1) * 10000)
    prefix = closed_local_returns4_reference(close[:43])
    for field in FIELDS:
        np.testing.assert_array_equal(prefix[field], full[field][:43])


def test_independent_formula_parity_with_production_owners():
    from gx1.features.technical_indicators_v1 import kaufman_efficiency_ratio

    rng = np.random.default_rng(20261107)
    close = 2000 + np.cumsum(rng.normal(0, 0.7, 1200))
    s = pd.Series(close)
    observed = {
        "ret_1": (s.pct_change(1) * 10000).to_numpy(dtype=np.float32),
        "ret_20": (s.pct_change(20) * 10000).to_numpy(dtype=np.float32),
        "rvol_20": (s.pct_change().rolling(20, min_periods=20).std()
                    * 10000 * np.sqrt(20)).to_numpy(dtype=np.float32),
        "chart.local_kama_efficiency_30":
            kaufman_efficiency_ratio(s, 30).to_numpy(dtype=np.float32),
    }
    expected = closed_local_returns4_reference(close)
    for field in FIELDS:
        np.testing.assert_allclose(expected[field], observed[field], rtol=2e-6,
                                   atol=2e-5, equal_nan=True)


def test_bad_close_fails_closed():
    with pytest.raises(ValueError, match="CLOSE_INVALID"):
        closed_local_returns4_reference([100.0, 0.0])
    with pytest.raises(ValueError, match="CLOSE_INVALID"):
        closed_local_returns4_reference([])
