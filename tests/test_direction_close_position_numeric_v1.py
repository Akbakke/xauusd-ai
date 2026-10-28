import numpy as np
import pytest

from gx1.scripts.audit_direction_close_position_numeric_v1 import close_position_oracle


def test_20_bar_position_and_flat_range_warmup():
    high=np.arange(25,dtype=np.float64)+2.;low=np.arange(25,dtype=np.float64);close=low+1.
    result=close_position_oracle(high,low,close)
    assert np.isnan(result[:19]).all()
    assert result[19] == 20./21. and result[20] == 20./21.
    flat=np.full(21,100.)
    assert np.isnan(close_position_oracle(flat,flat,flat)).all()


def test_close_position_rejects_invalid_ohlc():
    with pytest.raises(RuntimeError,match='ORACLE_DOMAIN'):
        close_position_oracle(np.ones(21),np.full(21,2.),np.ones(21))
    with pytest.raises(RuntimeError,match='ORACLE_DOMAIN'):
        close_position_oracle(np.ones(19),np.ones(19),np.ones(19))


def test_numpy_oracle_matches_independent_rolling_reference():
    import pandas as pd
    rng = np.random.default_rng(17)
    low = rng.uniform(1800, 2000, size=80); high = low + rng.uniform(.1, 5, size=80)
    close = low + (high-low) * rng.uniform(size=80)
    expected_low = pd.Series(low).rolling(20, min_periods=20).min()
    expected_high = pd.Series(high).rolling(20, min_periods=20).max()
    expected = (pd.Series(close)-expected_low)/(expected_high-expected_low)
    actual = close_position_oracle(high,low,close)
    assert np.array_equal(actual.astype(np.float32), expected.to_numpy(dtype=np.float32), equal_nan=True)
