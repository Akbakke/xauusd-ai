from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from gx1.features import volume_features
from gx1.features.volume_features import (
    VOLUME_FEATURE_PREFIX_ROWS,
    VOLUME_FEATURE_REQUIRED_HISTORY_ROWS,
    VOLUME_FEATURE_WARMUP_ROWS,
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


def test_volume_features_have_honest_warmup_are_causal_and_ordered() -> None:
    frame = _frame()
    baseline = compute_volume_features(frame)

    assert tuple(baseline) == tuple(VOLUME_FEATURE_NAMES)
    assert tuple(VOLUME_FEATURE_NAMES) == (
        "vol_z_20",
        "vol_ratio_5_20",
        "vol_pct_96",
    )
    assert all(values.shape == (len(frame),) for values in baseline.values())
    for name, values in baseline.items():
        warmup = VOLUME_FEATURE_WARMUP_ROWS[name]
        assert np.isnan(values[:warmup]).all()
        assert np.isfinite(values[warmup:]).all()
    # Strictly increasing volume: mid-rank of the newest row in a full window
    # is (window - 1 + 0.5) / window.
    assert baseline["vol_pct_96"][119] == np.float32((95 + 0.5) / 96.0)

    future_start = 90
    changed = frame.copy()
    changed.loc[future_start:, "volume"] *= 100.0
    mutated = compute_volume_features(changed)
    for name in VOLUME_FEATURE_NAMES:
        np.testing.assert_allclose(
            mutated[name][:future_start],
            baseline[name][:future_start],
            equal_nan=True,
        )


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


def test_short_frame_exposes_unavailable_features_as_nan() -> None:
    observed = compute_volume_features(_frame(4))

    for values in observed.values():
        assert np.isnan(values).all()


def test_constant_tick_count_has_zero_surprise_and_midrank_after_warmup() -> None:
    frame = _frame(120)
    frame["volume"] = 17.0
    observed = compute_volume_features(frame)

    np.testing.assert_array_equal(observed["vol_z_20"][19:], 0.0)
    np.testing.assert_array_equal(observed["vol_ratio_5_20"][19:], 0.0)
    np.testing.assert_array_equal(observed["vol_pct_96"][95:], 0.5)


def test_volume_owner_does_not_require_or_read_price_direction() -> None:
    volume_only = _frame(120)[["volume"]]
    expected = compute_volume_features(volume_only)
    with_price = volume_only.copy()
    with_price["close"] = np.linspace(1.0, 10_000.0, len(with_price))
    observed = compute_volume_features(with_price)

    for name in VOLUME_FEATURE_NAMES:
        np.testing.assert_array_equal(observed[name], expected[name])
    assert "signed_vol_z_20" not in inspect.getsource(volume_features)


def test_owner_is_native_row_clock_identical_for_m1_and_m5() -> None:
    m1 = _frame(120)
    m1.index = pd.date_range("2026-01-01", periods=len(m1), freq="1min", tz="UTC")
    m5 = m1.copy()
    m5.index = pd.date_range("2026-01-01", periods=len(m5), freq="5min", tz="UTC")

    observed_m1 = compute_volume_features(m1)
    observed_m5 = compute_volume_features(m5)
    for name in VOLUME_FEATURE_NAMES:
        np.testing.assert_array_equal(observed_m1[name], observed_m5[name])


def test_current_inclusive_zscore_is_not_static_clipped() -> None:
    frame = _frame(20)
    frame["volume"] = 1.0
    frame.loc[19, "volume"] = 1_000_000_000.0
    observed = compute_volume_features(frame)["vol_z_20"]
    values = frame["volume"].to_numpy(dtype=np.float64)
    expected = (values[-1] - values.mean()) / values.std(ddof=0)

    assert observed[-1] == pytest.approx(expected, rel=1e-6)
    source = inspect.getsource(volume_features)
    assert "_CLIP" not in source
    assert "np.clip" not in source


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
        ("volume", -1.0, "VOLUME_FEATURE_SOURCE_VOLUME_NOT_POSITIVE_INTEGER"),
        ("volume", 0.0, "VOLUME_FEATURE_SOURCE_VOLUME_NOT_POSITIVE_INTEGER"),
        ("volume", 1.5, "VOLUME_FEATURE_SOURCE_VOLUME_NOT_POSITIVE_INTEGER"),
        ("volume", True, "VOLUME_FEATURE_SOURCE_VOLUME_NOT_POSITIVE_INTEGER"),
        ("volume", "bad", "VOLUME_FEATURE_SOURCE_NOT_NUMERIC"),
    ],
)
def test_volume_features_reject_invalid_sources(
    field: str,
    value: object,
    message: str,
) -> None:
    frame = _frame(8)
    if isinstance(value, (str, bool, np.bool_)):
        frame[field] = frame[field].astype(object)
    frame.loc[3, field] = value
    with pytest.raises(RuntimeError, match=message):
        compute_volume_features(frame)


def test_volume_features_reject_missing_duplicate_and_empty_sources() -> None:
    with pytest.raises(RuntimeError, match="VOLUME_FEATURE_SOURCE_MISSING: volume"):
        compute_volume_features(_frame(8).drop(columns=["volume"]))
    with pytest.raises(RuntimeError, match="VOLUME_FEATURE_SOURCE_EMPTY"):
        compute_volume_features(_frame(0))

    duplicate = _frame(8)
    duplicate.insert(1, "volume", 1.0, allow_duplicates=True)
    with pytest.raises(RuntimeError, match="VOLUME_FEATURE_SOURCE_DUPLICATE: volume"):
        compute_volume_features(duplicate)
