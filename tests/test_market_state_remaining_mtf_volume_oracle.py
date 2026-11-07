"""Independent formula checks for three MTF price-update activity columns."""
import math

import numpy as np
import pandas as pd
import pytest

from gx1.audit.market_state_remaining_mtf_volume_oracle_v1 import (
    CACHE_COLUMNS, FIELD_NAMES, SOURCE_UNIT, WARMUP_ROWS,
    independent_closed_tf_price_update_oracle as oracle,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.features.volume_features import compute_volume_features


def _clock(n, *, step_ns=300_000_000_000):
    return np.arange(1, n + 1, dtype=np.int64) * step_ns


def test_exact_cache_columns_units_warmup_and_constant_case():
    assert CACHE_COLUMNS == (23, 24, 25)
    assert tuple(MULTI_TF_PER_BAR_FEATURES_V4[i] for i in CACHE_COLUMNS) == FIELD_NAMES
    assert WARMUP_ROWS == (19, 19, 95)
    assert SOURCE_UNIT == "positive_integer_price_updates_per_closed_tf_bar"
    result = oracle(np.full(100, 17, dtype=np.int64),
                    available_ns=_clock(100),
                    train_end_exclusive_ns=int(_clock(101)[-1]))
    values, available = result["values"], result["warmup_available"]
    assert values.dtype == np.float32 and available.dtype == np.bool_
    assert values.shape == available.shape == (100, 3)
    assert np.isnan(values[:19, :2]).all()
    assert np.isnan(values[:95, 2]).all()
    assert not available[:19, :2].any() and not available[:95, 2].any()
    np.testing.assert_array_equal(values[19:, :2], 0.0)
    np.testing.assert_array_equal(values[95:, 2], 0.5)
    assert result["output_unit"] == "dimensionless"


def test_one_price_update_spike_has_analytic_population_and_midrank_values():
    counts = np.ones(100, dtype=np.int64)
    counts[-1] = 2
    values = oracle(counts, available_ns=_clock(100))["values"]
    assert values[-1, 0] == pytest.approx(math.sqrt(19), rel=1e-6)
    assert values[-1, 1] == pytest.approx(1 / 7, rel=1e-6)
    assert values[-1, 2] == np.float32(95.5 / 96)
    assert values[95, 2] == np.float32(0.5)


def test_independent_oracle_matches_current_owner_on_synthetic_native_bars():
    rng = np.random.default_rng(17)
    counts = rng.integers(1, 1000, size=256, dtype=np.int64)
    counts[50:155] = 7  # constant windows and percentile ties
    counts[200] = 1_000_000  # no clipping
    expected = oracle(counts, available_ns=_clock(len(counts)))
    current = compute_volume_features(pd.DataFrame({"volume": counts.astype(np.float64)}))
    for j, name in enumerate(FIELD_NAMES):
        np.testing.assert_allclose(expected["values"][:, j], current[name],
                                   rtol=2e-6, atol=2e-6, equal_nan=True)
        np.testing.assert_array_equal(
            expected["warmup_available"][:, j], np.isfinite(current[name]))


def test_future_counts_and_clock_cadence_cannot_change_past_native_bar_values():
    counts = np.arange(1, 151, dtype=np.int64)
    baseline = oracle(counts, available_ns=_clock(len(counts)))
    changed = counts.copy()
    changed[120:] *= 100
    later = oracle(changed, available_ns=_clock(len(counts)))
    np.testing.assert_array_equal(
        baseline["values"][:120], later["values"][:120])
    daily = oracle(counts, available_ns=_clock(len(counts), step_ns=86_400_000_000_000))
    np.testing.assert_array_equal(baseline["values"], daily["values"])


@pytest.mark.parametrize("invalid", [
    np.array([1, 0, 3]), np.array([1, -2, 3]),
    np.array([1.0, 2.5, 3.0]), np.array([1.0, np.nan, 3.0]),
    np.array([1.0, np.inf, 3.0]), np.array([True, False, True]),
    np.array([1.0, 2**53 + 2.0, 3.0]),
    np.array([1, 2**53 + 1, 3], dtype=np.uint64),
])
def test_missing_or_noninteger_price_update_count_is_not_imputed(invalid):
    with pytest.raises(ValueError, match="PRICE_UPDATE_COUNT"):
        oracle(invalid, available_ns=_clock(3))


def test_clock_and_train_cutoff_fail_closed():
    counts = np.ones(4, dtype=np.int64)
    with pytest.raises(ValueError, match="AVAILABLE_CLOCK"):
        oracle(counts, available_ns=np.array([1, 2, 2, 4], dtype=np.int64))
    with pytest.raises(ValueError, match="AVAILABLE_CLOCK"):
        oracle(counts, available_ns=np.array([1., 2., 3., 4.]))
    with pytest.raises(ValueError, match="TRAIN_ONLY_CUTOFF"):
        oracle(counts, available_ns=_clock(4),
               train_end_exclusive_ns=int(_clock(4)[-1]))
