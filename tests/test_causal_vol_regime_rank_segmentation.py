"""The causal volatility-regime rank must not depend on how it is segmented.

``causal_vol_regime_bucket`` computes ``rolling(window, min_periods=1).rank``
in segments so its transient stays bounded: on the native M1 clock the
single-shot form allocated about 2.8 GiB and pushed the enriched producer past
the 10G ceiling. ``rolling().rank()`` recomputes each window independently, so a
segment carrying ``window - 1`` rows of preceding context reproduces the
whole-array result exactly. This pins that equality across several segment
boundaries and in the short-window region at the start where ``min_periods=1``
applies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_VOL_REGIME_WINDOW_BARS,
    _VOL_REGIME_RANK_SEGMENT_ROWS,
    causal_vol_regime_bucket,
)


def _whole_array_reference(values: np.ndarray) -> np.ndarray:
    """The pre-segmentation expression, computed in one pass."""

    percentile = (
        pd.Series(values, dtype=np.float64)
        .rolling(MODEL_NATIVE_VOL_REGIME_WINDOW_BARS, min_periods=1)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float64)
    )
    return np.clip(percentile * 5.0, 0.0, 4.99).astype(np.int64)


@pytest.mark.parametrize(
    "rows",
    [
        1,
        MODEL_NATIVE_VOL_REGIME_WINDOW_BARS - 1,
        MODEL_NATIVE_VOL_REGIME_WINDOW_BARS,
        _VOL_REGIME_RANK_SEGMENT_ROWS - 1,
        _VOL_REGIME_RANK_SEGMENT_ROWS,
        _VOL_REGIME_RANK_SEGMENT_ROWS + 1,
        _VOL_REGIME_RANK_SEGMENT_ROWS * 2 + 977,
    ],
)
def test_segmented_rank_equals_whole_array_rank(rows: int) -> None:
    rng = np.random.default_rng(20260806)
    values = np.abs(rng.lognormal(mean=1.0, sigma=0.4, size=rows)) + 0.01

    np.testing.assert_array_equal(
        causal_vol_regime_bucket(values),
        _whole_array_reference(values),
    )


def test_repeated_values_do_not_break_average_rank_ties() -> None:
    """Ties use method='average'; a segment boundary must not change them."""

    rows = _VOL_REGIME_RANK_SEGMENT_ROWS + MODEL_NATIVE_VOL_REGIME_WINDOW_BARS
    values = np.full(rows, 7.5, dtype=np.float64)
    values[::3] = 1.25

    np.testing.assert_array_equal(
        causal_vol_regime_bucket(values),
        _whole_array_reference(values),
    )


def test_segment_length_exceeds_the_rolling_window() -> None:
    assert isinstance(_VOL_REGIME_RANK_SEGMENT_ROWS, int)
    assert _VOL_REGIME_RANK_SEGMENT_ROWS > MODEL_NATIVE_VOL_REGIME_WINDOW_BARS
