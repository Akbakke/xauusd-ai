"""The V3 per-bar multi-timeframe contract.

The 513 signal fields are M5-only because their builders sit on 199 upstream
source fields, 194 of them derived, with dependencies between the families - so
producing them per timeframe means rebuilding the entire context pipeline. Two
owners have no such dependency: the candlestick family needs exactly
["open", "high", "low", "close", "time"] and swing structure is a pure function
of (high, low, close). V3 adds those to V2's 25 so the higher timeframes carry
real price geometry instead of a generic lens.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.features import htf_features as htf


def _bars(n: int, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 1.5, size=n))
    high = close + np.abs(rng.normal(0.0, 1.0, size=n))
    low = close - np.abs(rng.normal(0.0, 1.0, size=n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.date_range("2021-01-04", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": np.abs(rng.normal(500.0, 50.0, size=n)),
        },
        index=index,
    )


def test_v3_extends_v2_without_redefining_it() -> None:
    assert htf.MULTI_TF_FEATURE_COUNT_V2 == 25
    assert htf.MULTI_TF_FEATURE_COUNT_V3 == 90
    # V2 is a bit-identical prefix, so every artifact built on V2 stays valid.
    assert htf.MULTI_TF_PER_BAR_FEATURES_V3[:25] == htf.MULTI_TF_PER_BAR_FEATURES_V2
    assert len(set(htf.MULTI_TF_PER_BAR_FEATURES_V3)) == 90
    assert htf.MULTI_TF_FEATURE_NAMES_SHA256_V3 != htf.MULTI_TF_FEATURE_NAMES_SHA256_V2
    assert htf.HTF_V3_MATRIX_CONTRACT != htf.HTF_V2_MATRIX_CONTRACT


def test_v3_candlestick_names_come_from_the_owner() -> None:
    """A rename in the candlestick owner must not silently desync this contract."""
    from gx1.features.entry_candlestick_patterns_v1 import (
        CANDLESTICK_PATTERN_FEATURE_NAMES,
    )

    assert len(htf.MULTI_TF_PER_BAR_CANDLESTICK_V3) == len(
        CANDLESTICK_PATTERN_FEATURE_NAMES
    )
    for declared, owned in zip(
        htf.MULTI_TF_PER_BAR_CANDLESTICK_V3,
        CANDLESTICK_PATTERN_FEATURE_NAMES,
        strict=True,
    ):
        assert declared.endswith(owned.split(".", 1)[1])


def test_v3_holds_its_contract_at_every_resolution() -> None:
    """Exact width, exact order, and a single causal warmup prefix per timeframe."""
    m5 = _bars(20000, seed=7)
    for timeframe, rule in htf.MULTI_TF_RESAMPLE_RULES.items():
        resampled = (
            m5.resample(rule)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        if len(resampled) < 300:
            continue
        built = htf.compute_per_bar_features_v3(resampled)
        assert tuple(built.columns) == htf.MULTI_TF_PER_BAR_FEATURES_V3, timeframe
        matrix = built.to_numpy(dtype=np.float32)
        warmup = htf.validate_causal_feature_matrix(
            matrix,
            expected_width=htf.MULTI_TF_FEATURE_COUNT_V3,
            context=f"HTF_V3_{timeframe}",
        )
        assert np.isfinite(matrix[warmup:]).all(), timeframe


def test_v3_first_25_columns_equal_v2_exactly() -> None:
    """Adding families may not perturb a single existing value."""
    bars = _bars(6000, seed=11)
    v2 = htf.compute_per_bar_features_v2(bars).to_numpy(dtype=np.float64)
    v3 = htf.compute_per_bar_features_v3(bars).to_numpy(dtype=np.float64)

    assert v3.shape[1] == 90
    both_finite = np.isfinite(v2) & np.isfinite(v3[:, :25])
    assert np.array_equal(v2[both_finite], v3[:, :25][both_finite])
    assert np.array_equal(np.isfinite(v2), np.isfinite(v3[:, :25]))
