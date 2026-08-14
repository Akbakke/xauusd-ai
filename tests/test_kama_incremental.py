"""Independent formula and continuation tests for the basic-v1 KAMA owner."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features.basic_v1 import _kama, _kama_np_chunk, kama_np


def _reference_kama(
    prices: np.ndarray,
    period: int,
    fast: int = 2,
    slow: int = 30,
) -> np.ndarray:
    """Literal independent implementation of the published KAMA recurrence."""

    source = np.asarray(prices, dtype=np.float64)
    if period == 1:
        return source.copy()
    out = np.full(len(source), np.nan, dtype=np.float64)
    if len(source) <= period:
        return out
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    previous = float(source[period - 1])
    for row in range(period, len(source)):
        window = source[row - period : row + 1]
        volatility = float(
            sum(
                abs(window[i] - window[i - 1])
                for i in range(1, len(window))
            )
        )
        change = abs(float(window[-1] - window[0]))
        efficiency = 1.0 if volatility == 0.0 else change / volatility
        smoothing = (
            efficiency * (fast_sc - slow_sc) + slow_sc
        ) ** 2
        previous = previous + smoothing * (float(source[row]) - previous)
        out[row] = previous
    return out


def test_kama_known_monotone_vector_has_honest_prefix_and_price_seed() -> None:
    prices = np.arange(100.0, 107.0, dtype=np.float64)
    got = kama_np(prices, period=3)

    smoothing = (2.0 / 3.0) ** 2  # ER is exactly one on this vector.
    expected = np.full(len(prices), np.nan, dtype=np.float64)
    previous = prices[2]
    for row in range(3, len(prices)):
        previous += smoothing * (prices[row] - previous)
        expected[row] = previous
    np.testing.assert_allclose(
        got,
        expected,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_kama_matches_independent_reference_on_nonmonotone_vector() -> None:
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0.03, 0.5, size=1_000))

    got = kama_np(prices, period=30)
    expected = _reference_kama(prices, period=30)

    np.testing.assert_allclose(
        got,
        expected,
        rtol=2e-15,
        atol=2e-14,
        equal_nan=True,
    )


def test_kama_chunk_state_is_exact_at_arbitrary_boundaries() -> None:
    rng = np.random.default_rng(7)
    prices = 2_000.0 + np.cumsum(rng.normal(0.0, 0.4, size=437))
    expected = kama_np(prices, period=30)

    chunks: list[np.ndarray] = []
    state = None
    boundaries = (0, 4, 29, 30, 31, 117, 203, len(prices))
    for start, stop in zip(boundaries, boundaries[1:]):
        values, state = _kama_np_chunk(
            prices[start:stop],
            period=30,
            state=state,
        )
        chunks.append(values)
    got = np.concatenate(chunks)

    np.testing.assert_array_equal(got, expected)
    assert state is not None
    assert state.observations == len(prices)
    assert len(state.history) == 30


def test_kama_period_one_is_identity_across_chunks() -> None:
    prices = np.asarray([3.0, 1.0, 4.0, 1.5], dtype=np.float64)
    first, state = _kama_np_chunk(prices[:2], period=1)
    second, state = _kama_np_chunk(prices[2:], period=1, state=state)

    np.testing.assert_array_equal(np.concatenate((first, second)), prices)
    assert state.value == prices[-1]


@pytest.mark.parametrize(
    "prices,error",
    [
        (np.asarray([100.0, np.nan, 102.0]), "KAMA_SOURCE_INVALID"),
        (np.asarray([100.0, np.inf, 102.0]), "KAMA_SOURCE_INVALID"),
    ],
)
def test_kama_rejects_unobserved_prices_instead_of_forward_filling(
    prices: np.ndarray,
    error: str,
) -> None:
    with pytest.raises(RuntimeError, match=error):
        kama_np(prices, period=3)


def test_kama_wrapper_preserves_container_and_index() -> None:
    prices = np.arange(100.0, 140.0, dtype=np.float64)
    array_result = _kama(prices, period=3)
    series = pd.Series(prices, index=pd.RangeIndex(10, 50))
    series_result = _kama(series, period=3)

    assert isinstance(array_result, np.ndarray)
    assert isinstance(series_result, pd.Series)
    assert series_result.index.equals(series.index)
    np.testing.assert_array_equal(series_result.to_numpy(), array_result)


def test_kama_source_has_no_partial_ema_forward_fill_or_sc_clamp() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "gx1"
        / "features"
        / "basic_v1.py"
    ).read_text(encoding="utf-8")
    assert "use simple EMA-like initialization" not in source
    assert "forward-fill" not in source
    assert "max(0.0, min(1.0" not in source
