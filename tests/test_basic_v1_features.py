"""Value and ownership contracts for the active basic-v1 surface.

Covers the 2026-08 forensic-audit repairs:
- ``_v1_ema_diff`` is an ATR-multiple (era-stable), not a USD spread that
  tracked gold's price level across the tape.

And the V30 (2026-08-13) noise-amplifier repairs (formula-based expectations,
the d71a8e57 repair-wave precedent):
- ``_v1_kama30_change_5_atr`` / ``_v1_tema20_change_3_atr`` are k-bar ATR-multiple
  changes (k=5 / k=3), not 5th/3rd-order finite differences: on an
  (asymptotically) linear tape the k-bar change equals k x the per-bar step
  while any order>=2 finite difference is identically 0 — an algebraic
  discriminator between the repair and the bug.
- ``_v1_bb10_bandwidth_change_3`` is the plain 3-bar change of the
  dimensionless bandwidth.
"""
import ast
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features.basic_v1 import (
    BASIC_V1_FIRST_FINITE_ROW,
    BASIC_V1_FEATURES,
    BASIC_V1_FEATURES_SHA256,
    BASIC_V1_FORMULA_CONTRACT,
    BASIC_V1_FORMULA_SHA256,
    BASIC_V1_SCHEMA_VERSION,
    PLUS5_FEATURES,
    PLUS5_FIRST_FINITE_ROW,
    _ema_np_chunk,
    _tema_np_chunk,
    build_basic_v1,
    compute_plus5_features,
)


RETIRED_BASIC_V1_FIELDS = frozenset(
    {
        "_v1_atr_regime_id",
        "_v1_atr_z_10_100",
        "_v1_body_share_1",
        "_v1_body_tr",
        "_v1_clv",
        "_v1_comp3_ratio",
        "_v1_close_ema_slope_3",
        "_v1_int_clv_atr",
        "_v1_int_ema_us",
        "_v1_int_r5_atr",
        "_v1_int_range_us",
        "_v1_kama_slope_30",
        "_v1_is_EU",
        "_v1_is_US",
        "_v1_lower_tr",
        "_v1_r1",
        "_v1_r3",
        "_v1_r5",
        "_v1_r8",
        "_v1_r12",
        "_v1_r24",
        "_v1_r48_z",
        "_v1_r1_q10_48",
        "_v1_r1_q90_48",
        "_v1_range_adr",
        "_v1_range_comp_20_100",
        "_v1_bb_bandwidth_delta_10",
        "_v1_ret_ema_diff_2_5",
        "_v1_ret_ema_ratio_5_34",
        "_v1_rsi14",
        "_v1_rsi14_z",
        "_v1_rsi2",
        "_v1_rsi2_gt_rsi14",
        "_v1_session_volatility_pressure",
        "_v1_spread_p",
        "_v1_spread_z",
        "_v1_tod_cos",
        "_v1_tod_sin",
        "_v1_tema_slope_20",
        "_v1_tr_1_over_atr_14",
        "_v1_upper_tr",
        "_v1_wick_imbalance",
    }
)


def _market_frame(periods: int = 5000) -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="5min")
    phase = np.arange(periods, dtype=np.float64) * 2.0 * np.pi / (288.0 * 4.0)
    close = (
        2_000.0
        + np.linspace(0.0, 10.0, periods)
        + 8.0 * np.sin(phase)
        + 1.5 * np.sin(phase * 0.31)
    )
    half_range = 0.4 + 0.25 * (1.0 + np.sin(phase * 0.37))
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + half_range,
            "low": close - half_range,
            "close": close,
            "volume": (100 + np.arange(periods) % 201).astype(np.float64),
            "spread_pct": np.full(periods, 1e-4, dtype=np.float64),
        },
        index=index,
    )


def test_basic_v1_emits_only_the_hash_bound_active_surface() -> None:
    out, names = build_basic_v1(_market_frame())

    assert tuple(names) == BASIC_V1_FEATURES
    assert {name for name in out if name.startswith("_v1_")} == set(
        BASIC_V1_FEATURES
    )
    assert BASIC_V1_SCHEMA_VERSION == "gx1_basic_v1_active_surface_v3"
    assert BASIC_V1_FEATURES_SHA256 == hashlib.sha256(
        "\n".join(BASIC_V1_FEATURES).encode("utf-8")
    ).hexdigest()
    formula_binding = (
        BASIC_V1_FORMULA_CONTRACT
        + tuple(
            f"basic_first_finite:{name}={BASIC_V1_FIRST_FINITE_ROW[name]}"
            for name in BASIC_V1_FEATURES
        )
        + tuple(
            f"plus5_first_finite:{name}={PLUS5_FIRST_FINITE_ROW[name]}"
            for name in PLUS5_FEATURES
        )
    )
    assert BASIC_V1_FORMULA_SHA256 == hashlib.sha256(
        "\n".join(formula_binding).encode("utf-8")
    ).hexdigest()
    assert out.attrs["basic_v1_contract"] == {
        "schema_version": BASIC_V1_SCHEMA_VERSION,
        "features": list(BASIC_V1_FEATURES),
        "features_sha256": BASIC_V1_FEATURES_SHA256,
        "formula_contract": list(BASIC_V1_FORMULA_CONTRACT),
        "formula_sha256": BASIC_V1_FORMULA_SHA256,
        "first_finite_row": dict(BASIC_V1_FIRST_FINITE_ROW),
    }


def test_retired_basic_v1_fields_have_no_gx1_source_consumer_or_producer() -> None:
    root = Path(__file__).resolve().parents[1] / "gx1"
    offenders: dict[str, list[str]] = {}
    retired_formula_fragments = (
        "rolling5760",
        "min2880",
        "q333_q667",
        "audit_seq513_source_cascade_v1",
        "materialize_cv3_modelrange_v1",
        "validate_seq513_source_cascade_proof",
    )
    for source_path in sorted(root.rglob("*.py")):
        source = source_path.read_text(encoding="utf-8")
        hits = sorted(name for name in RETIRED_BASIC_V1_FIELDS if name in source)
        hits.extend(
            fragment for fragment in retired_formula_fragments if fragment in source
        )
        if hits:
            offenders[str(source_path.relative_to(root.parent))] = hits
    assert offenders == {}

    tree = ast.parse(
        (root / "features" / "basic_v1.py").read_text(encoding="utf-8")
    )
    literal_outputs = {
        target.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, str)
        and target.slice.value.startswith("_v1_")
    }
    assert literal_outputs == set(BASIC_V1_FEATURES)


def test_ema_diff_is_atr_relative_and_price_level_invariant() -> None:
    base = _market_frame()
    # 2.0 is a power of two: every linear price operation scales exactly in
    # binary floating point, so the ATR-multiple must be (near-)identical.
    scaled = base.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] = scaled[column] * 2.0

    out_base, _ = build_basic_v1(base)
    out_scaled, _ = build_basic_v1(scaled)

    ema_base = out_base["_v1_ema_diff"].to_numpy(dtype=np.float64)
    ema_scaled = out_scaled["_v1_ema_diff"].to_numpy(dtype=np.float64)

    # Causal warmup: the ATR(14) rolling warmup propagates as a contiguous
    # NaN prefix (exact length is owned by the rolling implementation),
    # followed by an all-finite tail.
    first_finite = int(np.flatnonzero(np.isfinite(ema_base))[0])
    assert first_finite > 0
    assert np.isnan(ema_base[:first_finite]).all()
    assert np.isfinite(ema_base[first_finite:]).all()
    # Non-degenerate: the trend spread is actually live on this tape.
    assert np.nanmax(np.abs(ema_base)) > 0.0
    # Era-stability: doubling the price level must not change the feature.
    # A USD-scaled spread would double instead (the refuted date proxy).
    np.testing.assert_allclose(ema_scaled, ema_base, rtol=1e-9, equal_nan=True)


def _linear_frame(periods: int = 5000, step: float = 0.5) -> pd.DataFrame:
    """Strictly linear close with a nondegenerate, known positive range.

    The varying high-low span dominates both previous-close gaps. This keeps
    ATR positive while giving range-z a real denominator; expected normalized
    slopes use the independently emitted Wilder ATR on each row.
    """
    index = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="5min")
    close = 2_000.0 + step * np.arange(periods, dtype=np.float64)
    half_range = 2.0 + 0.2 * np.sin(np.arange(periods) * 0.13)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + half_range,
            "low": close - half_range,
            "close": close,
            "volume": np.full(periods, 100.0),
            "spread_pct": np.full(periods, 1e-4, dtype=np.float64),
        },
        index=index,
    )


def test_kama_tema_slopes_are_kbar_atr_multiple_changes() -> None:
    """Algebraic proof of the V30 repair, independent of the implementation.

    On a strictly linear tape both smoothers converge to lagged linear
    trajectories with the SAME per-bar step d (TEMA is a linear filter; KAMA
    has efficiency ratio exactly 1 on a monotone tape, so its smoothing
    constant is fixed and its transient decays geometrically).  In the
    converged tail the k-bar change is exactly k*d, so the ATR-multiple
    slope is k*d/ATR — while the retired order-k finite difference of a
    linear sequence is identically 0.  The equality below therefore both
    pins the repaired formula and refutes the noise-amplifier form.
    """
    step = 0.5
    out, _ = build_basic_v1(_linear_frame(step=step))
    tema_slope = out["_v1_tema20_change_3_atr"].to_numpy(dtype=np.float64)
    kama_slope = out["_v1_kama30_change_5_atr"].to_numpy(dtype=np.float64)

    # Honest causal warmup: one contiguous NaN prefix, then all-finite.
    for values in (tema_slope, kama_slope):
        first_finite = int(np.flatnonzero(np.isfinite(values))[0])
        assert first_finite > 0
        assert np.isnan(values[:first_finite]).all()
        assert np.isfinite(values[first_finite:]).all()

    tail = slice(-500, None)
    atr = out["_v1_atr14"].to_numpy(dtype=np.float64)[tail]
    np.testing.assert_allclose(
        tema_slope[tail], 3.0 * step / atr, rtol=1e-9
    )
    np.testing.assert_allclose(
        kama_slope[tail], 5.0 * step / atr, rtol=1e-9
    )


def test_bb_bandwidth_delta_10_is_plain_3bar_change() -> None:
    """Formula-based expectation: bw[t] - bw[t-3] on the closed row t where bw is
    the dimensionless 10-bar Bollinger bandwidth (4*std10 / (mean10+eps),
    defined only on a complete window, ddof=0).  The independent loop reduces
    each window in a fixed order; the retired ``np.diff(n=3)`` 3rd-order form
    fails the resulting plain-change identity."""

    frame = _market_frame()
    out, _ = build_basic_v1(frame)
    got = out["_v1_bb10_bandwidth_change_3"].to_numpy(dtype=np.float64)

    close = frame["close"].to_numpy(dtype=np.float64)
    mean10 = np.full(len(close), np.nan, dtype=np.float64)
    std10 = np.full(len(close), np.nan, dtype=np.float64)
    for row in range(9, len(close)):
        window = close[row - 9 : row + 1]
        mean10[row] = sum(window) / 10.0
        std10[row] = np.sqrt(sum((window - mean10[row]) ** 2) / 10.0)
    # Exact production algebra ((m+2s) - (m-2s), not the algebraically equal
    # 4s) so the identity is bit-tight.
    bw = ((mean10 + 2.0 * std10) - (mean10 - 2.0 * std10)) / mean10
    expected = bw - np.concatenate((np.full(3, np.nan), bw[:-3]))

    np.testing.assert_allclose(got, expected, rtol=1e-12, equal_nan=True)
    # Honest causal warmup prefix, then all-finite (no nan_to_num masking).
    first_finite = int(np.flatnonzero(np.isfinite(got))[0])
    assert first_finite > 0
    assert np.isnan(got[:first_finite]).all()
    assert np.isfinite(got[first_finite:]).all()


def test_active_range_and_bb_sources_keep_unknown_warmup_as_nan() -> None:
    out, _ = build_basic_v1(_market_frame())
    for name in (
        "_v1_range_z",
        "_v1_bb_squeeze_20_2",
    ):
        values = out[name].to_numpy(dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(values))
        assert len(finite) > 0, name
        first = int(finite[0])
        assert first > 0, name
        assert np.isnan(values[:first]).all(), name
        assert np.isfinite(values[first:]).all(), name


def _classic_ema_reference(values: np.ndarray, span: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    out = np.full(len(source), np.nan, dtype=np.float64)
    if len(source) < span:
        return out
    out[span - 1] = float(sum(source[:span])) / span
    alpha = 2.0 / (span + 1.0)
    for row in range(span, len(source)):
        out[row] = out[row - 1] + alpha * (source[row] - out[row - 1])
    return out


def _tema_reference(values: np.ndarray, period: int) -> np.ndarray:
    ema1 = _classic_ema_reference(values, period)
    ema2 = _classic_ema_reference(ema1[np.isfinite(ema1)], period)
    padded2 = np.full(len(values), np.nan, dtype=np.float64)
    padded2[period - 1 : period - 1 + len(ema2)] = ema2
    finite2 = padded2[np.isfinite(padded2)]
    ema3 = _classic_ema_reference(finite2, period)
    padded3 = np.full(len(values), np.nan, dtype=np.float64)
    first2 = 2 * (period - 1)
    padded3[first2 : first2 + len(ema3)] = ema3
    return 3.0 * ema1 - 3.0 * padded2 + padded3


def test_every_live_field_has_its_indicator_defined_nan_prefix() -> None:
    out, _ = build_basic_v1(_market_frame(700))

    assert set(BASIC_V1_FIRST_FINITE_ROW) == set(BASIC_V1_FEATURES)
    for name, expected_first in BASIC_V1_FIRST_FINITE_ROW.items():
        values = out[name].to_numpy(dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(values))
        assert len(finite), name
        assert int(finite[0]) == expected_first, name
        assert np.isnan(values[:expected_first]).all(), name
        assert np.isfinite(values[expected_first:]).all(), name


def test_known_vectors_pin_atr_vwap_parkinson_range_kurtosis_and_ema() -> None:
    frame = _market_frame(300)
    out, _ = build_basic_v1(frame)
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    volume = frame["volume"].to_numpy(dtype=np.float64)

    previous_close = np.concatenate(([close[0]], close[:-1]))
    true_range = np.maximum.reduce(
        (high - low, np.abs(high - previous_close), np.abs(low - previous_close))
    )
    atr = float(sum(true_range[:14])) / 14.0
    assert out["_v1_atr14"].iloc[13] == atr
    for row in range(14, 26):
        atr = ((13.0 * atr) + true_range[row]) / 14.0

    expected_vwap = float(np.dot(close[:48], volume[:48]) / sum(volume[:48]))
    expected_vwap_drift = (close[47] - expected_vwap) / expected_vwap
    assert out["_v1_vwap_drift48"].iloc[47] == pytest.approx(
        expected_vwap_drift,
        rel=2e-7,
    )

    expected_pk = np.sqrt(np.mean(np.log(high[:20] / low[:20]) ** 2)) / np.sqrt(
        4.0 * np.log(2.0)
    )
    assert out["_v1_pk_sigma20"].iloc[19] == pytest.approx(expected_pk)

    ranges = high[:48] - low[:48]
    expected_range_z = (ranges[-1] - np.mean(ranges)) / np.std(ranges, ddof=0)
    assert out["_v1_range_z"].iloc[47] == pytest.approx(expected_range_z)

    returns = close[1:49] / close[:48] - 1.0
    n = len(returns)
    deviations = returns - np.mean(returns)
    sum2 = float(sum(deviations**2))
    sum4 = float(sum(deviations**4))
    sample_variance = sum2 / (n - 1)
    factor1 = n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))
    factor2 = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    expected_kurtosis = factor1 * sum4 / sample_variance**2 - factor2
    assert out["_v1_kurt_r"].iloc[48] == pytest.approx(expected_kurtosis)

    ema3 = _classic_ema_reference(close, 3)
    ema6 = _classic_ema_reference(close, 6)
    expected_ema_slope = (ema3[5] - ema6[5]) / ema6[5]
    assert out["_v1_ema3_ema6_spread_frac"].iloc[5] == pytest.approx(
        expected_ema_slope
    )
    ema12 = _classic_ema_reference(close, 12)
    ema26 = _classic_ema_reference(close, 26)
    expected_diff = (ema12[25] - ema26[25]) / atr
    assert out["_v1_ema_diff"].iloc[25] == pytest.approx(expected_diff)


def test_plus5_fields_use_complete_indicator_windows_and_known_values() -> None:
    frame = _market_frame(300)
    out = compute_plus5_features(frame)
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)

    assert set(PLUS5_FIRST_FINITE_ROW) == set(PLUS5_FEATURES)
    assert out.attrs["plus5_contract"] == {
        "schema_version": BASIC_V1_SCHEMA_VERSION,
        "features": list(PLUS5_FEATURES),
        "formula_sha256": BASIC_V1_FORMULA_SHA256,
        "first_finite_row": dict(PLUS5_FIRST_FINITE_ROW),
    }
    for name, expected_first in PLUS5_FIRST_FINITE_ROW.items():
        values = out[name].to_numpy(dtype=np.float64)
        finite = np.flatnonzero(np.isfinite(values))
        assert int(finite[0]) == expected_first, name
        assert np.isnan(values[:expected_first]).all(), name
        assert np.isfinite(values[expected_first:]).all(), name

    returns = close[1:51] / close[:50] - 1.0
    assert out["std50"].iloc[50] == pytest.approx(
        np.std(returns, ddof=1),
        rel=2e-7,
    )
    assert out["roc20"].iloc[20] == pytest.approx(
        close[20] / close[0] - 1.0,
        rel=2e-7,
    )
    previous_close = np.concatenate(([close[0]], close[:-1]))
    true_range = np.maximum.reduce(
        (high - low, np.abs(high - previous_close), np.abs(low - previous_close))
    )
    assert out["atr"].iloc[13] == pytest.approx(np.mean(true_range[:14]))


@pytest.mark.parametrize("invalid_volume", [100.5, True])
def test_plus5_rejects_non_integer_tick_volume(invalid_volume: object) -> None:
    frame = _market_frame(80)
    if isinstance(invalid_volume, bool):
        frame["volume"] = frame["volume"].astype(object)
    frame.loc[frame.index[-1], "volume"] = invalid_volume

    with pytest.raises(RuntimeError, match="PLUS5_VOLUME_INVALID"):
        compute_plus5_features(frame)


def test_plus5_fields_are_prefix_future_invariant() -> None:
    frame = _market_frame(300)
    prefix = frame.iloc[:173].copy()
    full = compute_plus5_features(frame)
    before = compute_plus5_features(prefix)

    for name in PLUS5_FEATURES:
        np.testing.assert_array_equal(
            before[name].to_numpy(),
            full.loc[prefix.index, name].to_numpy(),
        )


def test_plus5_finite_windows_are_exact_with_50_row_overlap() -> None:
    frame = _market_frame(300)
    full = compute_plus5_features(frame)
    cut = 200
    bounded = compute_plus5_features(frame.iloc[cut - 50 :])

    for name in ("std50", "roc20", "_v1_vwap_drift48"):
        np.testing.assert_array_equal(
            bounded.loc[frame.index[cut] :, name].to_numpy(),
            full.loc[frame.index[cut] :, name].to_numpy(),
        )


def test_recursive_ema_and_tema_states_are_exact_across_chunk_boundaries() -> None:
    values = 2_000.0 + np.sin(np.arange(311) * 0.17) + np.arange(311) * 0.02
    ema_expected = _classic_ema_reference(values, 20)
    tema_expected = _tema_reference(values, 20)
    tema_one_shot, _tema_one_shot_state = _tema_np_chunk(values, 20)

    ema_parts: list[np.ndarray] = []
    tema_parts: list[np.ndarray] = []
    ema_state = None
    tema_state = None
    boundaries = (0, 1, 19, 20, 21, 57, 138, 230, len(values))
    for start, stop in zip(boundaries, boundaries[1:]):
        ema_values, ema_state = _ema_np_chunk(
            values[start:stop],
            20,
            state=ema_state,
        )
        tema_values, tema_state = _tema_np_chunk(
            values[start:stop],
            20,
            state=tema_state,
        )
        ema_parts.append(ema_values)
        tema_parts.append(tema_values)

    np.testing.assert_array_equal(np.concatenate(ema_parts), ema_expected)
    tema_chunked = np.concatenate(tema_parts)
    np.testing.assert_array_equal(tema_chunked, tema_one_shot)
    np.testing.assert_allclose(
        tema_chunked,
        tema_expected,
        rtol=2e-15,
        atol=2e-13,
        equal_nan=True,
    )


def test_finite_window_fields_are_exact_with_derived_118_row_overlap() -> None:
    frame = _market_frame(700)
    full, _ = build_basic_v1(frame)
    cut = 500
    # BB20 followed by its full 100-observation bandwidth baseline reaches
    # back exactly (20-1)+(100-1)=118 rows, the largest finite window.
    overlap = (20 - 1) + (100 - 1)
    bounded, _ = build_basic_v1(frame.iloc[cut - overlap :])
    finite_window_fields = (
        "_v1_pk_sigma20",
        "_v1_vwap_drift48",
        "_v1_bb10_bandwidth_change_3",
        "_v1_range_z",
        "_v1_kurt_r",
        "_v1_bb_squeeze_20_2",
    )
    for name in finite_window_fields:
        np.testing.assert_array_equal(
            bounded.loc[frame.index[cut] :, name].to_numpy(),
            full.loc[frame.index[cut] :, name].to_numpy(),
        )


def test_all_live_fields_are_prefix_and_future_invariant() -> None:
    frame = _market_frame(700)
    prefix = frame.iloc[:430].copy()
    full, _ = build_basic_v1(frame)
    before, _ = build_basic_v1(prefix)

    for name in BASIC_V1_FEATURES:
        np.testing.assert_array_equal(
            before[name].to_numpy(dtype=np.float64),
            full.loc[prefix.index, name].to_numpy(dtype=np.float64),
        )


def test_closed_row_is_current_decision_evidence_not_hidden_lag_one() -> None:
    from gx1.execution.v12_state_from_prebuilt import (
        PREBUILT_PAIR_TIMING_CONTRACT,
    )

    assert PREBUILT_PAIR_TIMING_CONTRACT["m1_availability"] == (
        "bar_start_plus_1min"
    )
    assert PREBUILT_PAIR_TIMING_CONTRACT["m5_availability"] == (
        "bar_start_plus_5min"
    )

    frame = _market_frame(300)
    changed = frame.copy()
    new_close = float(changed["close"].iloc[-1] + 3.0)
    changed.loc[changed.index[-1], ["open", "high", "low", "close"]] = (
        new_close - 0.1,
        new_close + 2.0,
        new_close - 2.0,
        new_close,
    )
    changed.loc[changed.index[-1], "volume"] *= 2.0

    baseline, _ = build_basic_v1(frame)
    current, _ = build_basic_v1(changed)
    for name in BASIC_V1_FEATURES:
        np.testing.assert_array_equal(
            current[name].iloc[:-1].to_numpy(dtype=np.float64),
            baseline[name].iloc[:-1].to_numpy(dtype=np.float64),
        )
        assert current[name].iloc[-1] != baseline[name].iloc[-1], name


def test_undefined_zero_denominators_fail_closed_instead_of_emitting_neutral() -> None:
    rows = 200
    index = pd.date_range("2026-01-01T00:00:00Z", periods=rows, freq="5min")
    flat = pd.DataFrame(
        {
            "open": np.full(rows, 2_000.0),
            "high": np.full(rows, 2_000.0),
            "low": np.full(rows, 2_000.0),
            "close": np.full(rows, 2_000.0),
            "volume": np.full(rows, 100.0),
            "spread_pct": np.full(rows, 1e-4),
        },
        index=index,
    )

    with pytest.raises(
        RuntimeError,
        match=r"BASIC_V1_FEATURE_UNAVAILABLE.*_v1_ema_diff",
    ):
        build_basic_v1(flat)


def test_m1_and_m5_use_one_formula_contract_with_independent_clock_values() -> None:
    m5 = _market_frame(500)
    m1 = m5.copy()
    m1.index = pd.date_range(m5.index[0], periods=len(m1), freq="1min")
    perturbation = 0.2 * np.sin(np.arange(len(m1), dtype=np.float64) * 0.17)
    close = m1["close"].to_numpy(dtype=np.float64) + perturbation
    half_range = 0.5 + 0.1 * np.cos(np.arange(len(m1)) * 0.11)
    m1["open"] = close - 0.1
    m1["high"] = close + half_range
    m1["low"] = close - half_range
    m1["close"] = close

    m1_out, _ = build_basic_v1(m1, decision_delay_seconds=60)
    m5_out, _ = build_basic_v1(m5, decision_delay_seconds=300)
    m1_plus = compute_plus5_features(m1)
    m5_plus = compute_plus5_features(m5)

    assert m1_out.attrs["basic_v1_contract"] == m5_out.attrs["basic_v1_contract"]
    for name in ("_v1_atr14", "_v1_vwap_drift48", "_v1_ema_diff"):
        assert not np.array_equal(
            m1_out[name].to_numpy(dtype=np.float64),
            m5_out[name].to_numpy(dtype=np.float64),
            equal_nan=True,
        )
    for name in PLUS5_FEATURES:
        assert not np.array_equal(
            m1_plus[name].to_numpy(dtype=np.float64),
            m5_plus[name].to_numpy(dtype=np.float64),
            equal_nan=True,
        )


def test_basic_v1_source_forbids_partial_warmup_and_neutral_fill_regressions() -> None:
    source_path = (
        Path(__file__).resolve().parents[1] / "gx1" / "features" / "basic_v1.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert re.search(r"min_periods\s*=\s*1(?:\D|$)", source) is None
    assert "int(win * 0.8)" not in source
    assert "span//2" not in source
    assert "returns.rolling(50, min_periods=2)" not in source
    assert "1e-12" not in source
    assert "ema_slope_shifted[0] = 0.0" not in source
    assert "kurt_result_shifted[0] = 0.0" not in source
    assert "np.nan_to_num(ema_slope_shifted" not in source
    assert "np.nan_to_num(kurt_result_shifted" not in source

    tree = ast.parse(source)
    protected_functions = {
        "compute_plus5_features",
        "build_basic_v1",
        "_ema_np_chunk",
        "_tema_np_chunk",
        "_kama_np_chunk",
    }
    forbidden_calls: dict[str, list[str]] = {}
    for function in (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in protected_functions
    ):
        calls = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                call_name = node.func.id
            else:
                continue
            if call_name in {"clip", "fillna", "nan_to_num"}:
                calls.append(call_name)
        if calls:
            forbidden_calls[function.name] = sorted(calls)
    assert forbidden_calls == {}
    build_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build_basic_v1"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"roll", "shift"}
        for node in ast.walk(build_node)
    )


def test_fidelity_renames_preserve_exact_specialist_semantics() -> None:
    from gx1.features.entry_specialist_feature_groups_v1 import (
        classify_entry_specialist_feature,
    )

    assert classify_entry_specialist_feature(
        "_v1_ema3_ema6_spread_frac"
    ) == "trend_ema_encoder"
    assert classify_entry_specialist_feature(
        "_v1_tema20_change_3_atr"
    ) == "trend_ema_encoder"
    assert classify_entry_specialist_feature(
        "_v1_kama30_change_5_atr"
    ) == "trend_ema_encoder"
    assert classify_entry_specialist_feature(
        "_v1_bb10_bandwidth_change_3"
    ) == "vol_compression_encoder"
