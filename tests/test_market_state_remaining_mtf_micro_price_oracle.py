import numpy as np
import pytest

from gx1.audit.market_state_remaining_mtf_micro_price_oracle_v1 import (
    CACHE_COLUMNS, FIELDS, closed_micro_price5_reference,
)


def test_names_warmup_and_closed_bar_values():
    close = np.array([100, 102, 101, 105, 107, 106, 109], dtype=np.float64)
    high = close + 2
    low = close - 3
    got = closed_micro_price5_reference(high, low, close)
    assert tuple(got) == FIELDS and CACHE_COLUMNS == (218, 219, 220, 221, 222)
    assert all(x.dtype == np.float32 for x in got.values())
    assert np.isnan(got[FIELDS[0]][:3]).all()
    assert np.isnan(got[FIELDS[1]][:5]).all()
    assert np.isnan(got[FIELDS[2]][:2]).all()
    assert got[FIELDS[0]][3] == np.float32((105 / 100 - 1) * 10000)
    assert got[FIELDS[1]][5] == np.float32((106 / 100 - 1) * 10000)
    assert got[FIELDS[2]][2] == np.float32(((101 / 102 - 1) - (102 / 100 - 1)) * 10000)
    assert np.all(got[FIELDS[3]] == np.float32(2 / 5))
    assert np.all(got[FIELDS[4]] == 1.0)


def test_zero_range_is_masked_storage_zero_not_a_midpoint():
    close = np.array([100, 101, 101, 103, 104, 105], dtype=np.float64)
    high = close + 1
    low = close - 1
    high[2] = close[2]
    low[2] = close[2]
    got = closed_micro_price5_reference(high, low, close)
    assert got[FIELDS[3]][2] == 0.0
    assert got[FIELDS[4]][2] == 0.0
    assert got[FIELDS[3]][1] == np.float32(0.5)
    assert got[FIELDS[4]][1] == 1.0


def test_prefix_causality_and_invalid_geometry():
    close = np.linspace(100, 109, 10)
    high, low = close + 1, close - 1
    prefix = closed_micro_price5_reference(high[:7], low[:7], close[:7])
    full = closed_micro_price5_reference(high, low, close)
    for field in FIELDS:
        np.testing.assert_array_equal(prefix[field], full[field][:7])
    bad_high = high.copy()
    bad_high[3] = close[3] - 1
    with pytest.raises(ValueError, match="SOURCE_VALUES"):
        closed_micro_price5_reference(bad_high, low, close)
    with pytest.raises(ValueError, match="SOURCE_SHAPE"):
        closed_micro_price5_reference(high, low[:-1], close)


def test_independent_oracle_matches_producer_on_closed_synthetic_prefix():
    from gx1.features.micro_structure_v1 import compute_micro_structure_features

    rng = np.random.default_rng(20261107)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.4, 1200))
    width = rng.uniform(0.05, 0.9, len(close))
    high, low = close + width, close - width
    high[::97] = close[::97]
    low[::97] = close[::97]
    expected = closed_micro_price5_reference(high, low, close)
    observed = compute_micro_structure_features(high, low, close)
    for field in FIELDS:
        np.testing.assert_array_equal(expected[field], observed[field])
