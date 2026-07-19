from __future__ import annotations

import numpy as np
import pytest

from gx1.features.swing_structure_v1 import (
    SWING_FEATURE_NAMES_V1,
    compute_swing_structure_features,
)


def _ohlc() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = np.array([100, 102, 104, 102, 101, 103, 105, 103, 102], dtype=np.float64)
    high = close + np.array([1, 1, 2, 1, 1, 1, 2, 1, 1], dtype=np.float64)
    low = close - 1.0
    return high, low, close


def test_swing_structure_is_causal_and_exact() -> None:
    high, low, close = _ohlc()
    observed = compute_swing_structure_features(high, low, close)
    assert tuple(observed) == SWING_FEATURE_NAMES_V1
    assert all(values.shape == close.shape for values in observed.values())
    assert all(np.isfinite(values).all() for values in observed.values())

    changed_high = high.copy()
    changed_low = low.copy()
    changed_close = close.copy()
    changed_high[-1] += 20.0
    changed_low[-1] -= 20.0
    changed_close[-1] += 5.0
    changed = compute_swing_structure_features(
        changed_high,
        changed_low,
        changed_close,
    )
    for name in SWING_FEATURE_NAMES_V1:
        np.testing.assert_array_equal(observed[name][:-1], changed[name][:-1])


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda h, _l, _c: h.__setitem__(2, np.nan), "NONFINITE"),
        (lambda h, _l, _c: h.__setitem__(2, 0.0), "NONPOSITIVE"),
        (
            lambda h, low_values, _c: low_values.__setitem__(2, h[2] + 1.0),
            "GEOMETRY",
        ),
    ],
)
def test_swing_structure_rejects_invalid_market_evidence(mutator, match: str) -> None:
    high, low, close = _ohlc()
    mutator(high, low, close)
    with pytest.raises(RuntimeError, match=match):
        compute_swing_structure_features(high, low, close)


def test_swing_structure_rejects_empty_or_invalid_parameters() -> None:
    with pytest.raises(RuntimeError, match="LENGTH"):
        compute_swing_structure_features([], [], [])
    high, low, close = _ohlc()
    with pytest.raises(RuntimeError, match="LOOKBACK"):
        compute_swing_structure_features(high, low, close, lookback=0)
    with pytest.raises(RuntimeError, match="ATR_PERIOD"):
        compute_swing_structure_features(high, low, close, atr_period=0)
