from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gx1.features.volume_features import (
    VOLUME_FEATURE_PREFIX_ROWS,
    VOLUME_FEATURE_REQUIRED_HISTORY_ROWS,
    VOLUME_FEATURE_NAMES,
    add_volume_features,
    compute_volume_features,
)


def _frame(n: int = 120) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "volume": np.arange(1, n + 1, dtype=np.float64),
            "close": 2000.0 + np.sin(np.arange(n, dtype=np.float64) / 7.0),
        }
    )


def test_volume_features_are_finite_causal_and_ordered() -> None:
    frame = _frame()
    baseline = compute_volume_features(frame)

    assert tuple(baseline) == tuple(VOLUME_FEATURE_NAMES)
    assert all(values.shape == (len(frame),) for values in baseline.values())
    assert all(np.isfinite(values).all() for values in baseline.values())
    assert baseline["vol_z_20"][0] == 0.0
    assert baseline["vol_ratio_5_20"][0] == 0.0
    # Mid-rank tie convention: a single-element warmup window ranks its own
    # row at the neutral 0.5, not 1.0.
    assert baseline["vol_pct_96"][0] == 0.5
    # Strictly increasing volume: mid-rank of the newest row in a full window
    # is (window - 1 + 0.5) / window.
    assert baseline["vol_pct_96"][119] == np.float32((95 + 0.5) / 96.0)

    future_start = 90
    changed = frame.copy()
    changed.loc[future_start:, "volume"] *= 100.0
    changed.loc[future_start:, "close"] += 500.0
    mutated = compute_volume_features(changed)
    for name in VOLUME_FEATURE_NAMES:
        np.testing.assert_array_equal(mutated[name][:future_start], baseline[name][:future_start])


def test_volume_feature_history_contract_matches_long_history_tail() -> None:
    assert VOLUME_FEATURE_REQUIRED_HISTORY_ROWS == 96
    assert VOLUME_FEATURE_PREFIX_ROWS == 95
    full = _frame(800)
    expected = compute_volume_features(full)
    bounded = compute_volume_features(full.tail(512 + VOLUME_FEATURE_PREFIX_ROWS))

    for name in VOLUME_FEATURE_NAMES:
        np.testing.assert_array_equal(
            bounded[name][-512:],
            expected[name][-512:],
        )


def test_add_volume_features_matches_exact_owner_outputs() -> None:
    frame = _frame()
    expected = compute_volume_features(frame)
    observed = add_volume_features(frame.copy())

    for name in VOLUME_FEATURE_NAMES:
        np.testing.assert_array_equal(observed[name].to_numpy(), expected[name])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("volume", np.nan, "VOLUME_FEATURE_SOURCE_NONFINITE"),
        ("volume", np.inf, "VOLUME_FEATURE_SOURCE_NONFINITE"),
        ("volume", -1.0, "VOLUME_FEATURE_SOURCE_VOLUME_NEGATIVE"),
        ("close", 0.0, "VOLUME_FEATURE_SOURCE_CLOSE_NOT_POSITIVE"),
        ("close", -np.inf, "VOLUME_FEATURE_SOURCE_NONFINITE"),
        ("volume", "bad", "VOLUME_FEATURE_SOURCE_NOT_NUMERIC"),
    ],
)
def test_volume_features_reject_invalid_sources(
    field: str,
    value: object,
    message: str,
) -> None:
    frame = _frame(8)
    if isinstance(value, str):
        frame[field] = frame[field].astype(object)
    frame.loc[3, field] = value
    with pytest.raises(RuntimeError, match=message):
        compute_volume_features(frame)


def test_volume_features_reject_missing_duplicate_and_empty_sources() -> None:
    with pytest.raises(RuntimeError, match="VOLUME_FEATURE_SOURCE_MISSING: volume"):
        compute_volume_features(_frame(8).drop(columns=["volume"]))
    with pytest.raises(RuntimeError, match="VOLUME_FEATURE_SOURCE_MISSING: close"):
        compute_volume_features(_frame(8).drop(columns=["close"]))
    with pytest.raises(RuntimeError, match="VOLUME_FEATURE_SOURCE_EMPTY"):
        compute_volume_features(_frame(0))

    duplicate = _frame(8)
    duplicate.insert(1, "volume", 1.0, allow_duplicates=True)
    with pytest.raises(RuntimeError, match="VOLUME_FEATURE_SOURCE_DUPLICATE: volume"):
        compute_volume_features(duplicate)
