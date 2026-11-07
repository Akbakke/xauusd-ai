"""Synthetic checks of an independent five-scalar HLC oracle."""
import numpy as np
import pandas as pd
import pytest

from gx1.audit.market_state_remaining_mtf_generic5_oracle_v1 import (
    CACHE_COLUMNS, GENERIC5_CACHE_NAMES, generic5_from_closed_hlc,
)
from gx1.features.htf_features import (
    MULTI_TF_V4_GENERIC_SCALAR_ADDITION_FEATURES,
    _compute_model_native_scalar_primitives_v4,
)


def _oscillating(n=280):
    i = np.arange(n, dtype=np.float64)
    close = 1900 + .12 * i + 3 * np.sin(.17 * i)
    high = close + 1 + .16 * (1 + np.sin(.23 * i))
    low = close - 1 - .13 * (1 + np.cos(.31 * i))
    return high, low, close


def test_exact_cache_contract():
    assert CACHE_COLUMNS == (230, 231, 232, 234, 238)
    assert GENERIC5_CACHE_NAMES == tuple(
        MULTI_TF_V4_GENERIC_SCALAR_ADDITION_FEATURES[j]
        for j in (0, 1, 2, 4, 8))


def test_warmup_masks_and_dimensionless_units():
    high, low, close = _oscillating()
    actual = generic5_from_closed_hlc(high, low, close)
    assert actual.shape == (280, 5) and actual.dtype == np.float32
    for j, first in enumerate((61, 28, 30, 19, 219)):
        assert np.isnan(actual[:first, j]).all()
        assert np.isfinite(actual[first:, j]).all()
    scaled = generic5_from_closed_hlc(high * 3.7, low * 3.7, close * 3.7)
    np.testing.assert_allclose(actual, scaled, rtol=4e-5, atol=4e-5,
                               equal_nan=True)


def test_range_population_z_score_and_zero_denominator_masks():
    widths = np.arange(1, 21, dtype=np.float64)
    close = np.full(20, 100., dtype=np.float64)
    high, low = close + widths / 2, close - widths / 2
    actual = generic5_from_closed_hlc(high, low, close)
    expected = (20 - 10.5) / np.std(widths, ddof=0)
    np.testing.assert_allclose(actual[-1, 3], np.float32(expected), rtol=0, atol=0)
    assert np.isnan(actual[:-1, 3]).all()
    flat = np.full(240, 100., dtype=np.float64)
    masked = generic5_from_closed_hlc(flat.copy(), flat.copy(), flat.copy())
    assert np.isnan(masked).all() and not np.isinf(masked).any()


def test_selected_positions_and_causal_prefix_parity():
    high, low, close = _oscillating()
    full = generic5_from_closed_hlc(high, low, close)
    positions = np.array([19, 28, 30, 61, 219, 220, 279], dtype=np.int64)
    selected = generic5_from_closed_hlc(high, low, close, positions=positions)
    assert np.array_equal(selected.view(np.uint32), full[positions].view(np.uint32))
    for cutoff in (20, 29, 31, 62, 220, 221, 280):
        prefix = generic5_from_closed_hlc(high[:cutoff], low[:cutoff], close[:cutoff])
        assert np.array_equal(prefix[-1].view(np.uint32), full[cutoff - 1].view(np.uint32))


def test_invalid_source_and_positions_fail_closed():
    high, low, close = _oscillating(4)
    with pytest.raises(ValueError, match='FLOAT64_HLC'):
        generic5_from_closed_hlc(high.astype(np.float32), low, close)
    with pytest.raises(ValueError, match='FINITE_POSITIVE_HLC'):
        generic5_from_closed_hlc(high, np.array([1., np.nan, 1., 1.]), close)
    with pytest.raises(ValueError, match='GEOMETRY'):
        generic5_from_closed_hlc(close - 1., low, close)
    with pytest.raises(ValueError, match='POSITIONS'):
        generic5_from_closed_hlc(high, low, close,
                                 positions=np.array([2, 1], dtype=np.int64))


def test_synthetic_producer_formula_parity():
    high, low, close = _oscillating(600)
    frame = pd.DataFrame({
        'open': close.copy(), 'high': high, 'low': low, 'close': close,
    }, index=pd.date_range('2024-01-01', periods=len(close), freq='5min', tz='UTC'))
    actual = generic5_from_closed_hlc(high, low, close)
    producer = _compute_model_native_scalar_primitives_v4(frame)
    expected = producer.loc[:, GENERIC5_CACHE_NAMES].to_numpy(dtype=np.float32)
    assert np.array_equal(np.isnan(actual), np.isnan(expected))
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
