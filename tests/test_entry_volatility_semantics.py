from __future__ import annotations

import numpy as np
import pytest

from gx1.features.entry_volatility_semantics_v1 import (
    atr_ratio_compression_pressure,
    atr_ratio_expansion_pressure,
    bollinger_expansion_pressure,
    bollinger_squeeze_pressure,
    center_atr_ratio,
)


def test_center_atr_ratio_is_signed_symmetric_and_fails_closed() -> None:
    ratio = np.asarray([0.5, 1.0, 2.0], dtype=np.float32)
    expected = float(np.tanh(1.0))
    np.testing.assert_allclose(
        center_atr_ratio(ratio), [-expected, 0.0, expected], atol=1e-7
    )
    with pytest.raises(RuntimeError, match="ENTRY_VOLATILITY_SEMANTICS"):
        center_atr_ratio(np.asarray([0.0], dtype=np.float32))


def test_atr14_over_atr100_has_symmetric_compression_expansion_semantics() -> None:
    ratio = np.asarray([0.5, 1.0, 2.0], dtype=np.float32)
    expected_pressure = float(np.tanh(1.0))
    np.testing.assert_allclose(
        atr_ratio_compression_pressure(ratio),
        [expected_pressure, 0.0, 0.0],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        atr_ratio_expansion_pressure(ratio),
        [0.0, 0.0, expected_pressure],
        atol=1e-7,
    )


def test_bollinger_relative_width_sign_is_not_inverted() -> None:
    relative_width = np.asarray([-0.8, 0.0, 0.8], dtype=np.float32)
    np.testing.assert_allclose(
        bollinger_squeeze_pressure(relative_width), [0.8, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        bollinger_expansion_pressure(relative_width), [0.0, 0.0, 0.8]
    )


@pytest.mark.parametrize("bad", [[0.0], [-0.5], [np.nan], [np.inf]])
def test_atr_ratio_semantics_fail_closed_on_impossible_input(bad: list[float]) -> None:
    with pytest.raises(RuntimeError, match="ENTRY_VOLATILITY_SEMANTICS"):
        atr_ratio_compression_pressure(np.asarray(bad, dtype=np.float32))
