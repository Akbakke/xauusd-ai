from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CONT_MICRO_FIELDS,
    MODEL_NATIVE_CTX_CONT_SESSION_FIELDS,
    MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS,
    MODEL_NATIVE_CTX_CONT_SPREAD_DYNAMICS_FIELDS,
    MODEL_NATIVE_CTX_CONT_SWING_FIELDS,
    MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS,
    MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS,
    MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS,
)
from gx1.features.micro_structure_v1 import (
    MICRO_FEATURE_NAMES_V1,
    SPREAD_DYNAMICS_CAUSAL_WARMUP_ROWS_V1,
    SPREAD_DYNAMICS_FEATURE_NAMES_V1,
    SPREAD_DYNAMICS_SOURCE_COLUMNS_V1,
    SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1,
    compute_micro_structure_features,
    compute_spread_dynamics_features,
)
from gx1.features.entry_foundation_structure_v1 import FOUNDATION_EVENT_AGE_CAP
from gx1.features.regime_v4_features import REGIME_V4_SOURCE_COLS
from gx1.features.swing_structure_v1 import (
    SWING_FEATURE_NAMES_V1,
    SWING_V29_ADDITION_NAMES_V1,
    compute_swing_structure_features,
)
from gx1.features.volume_features import VOLUME_FEATURE_NAMES
from gx1.contracts.entry_model_native_state_v2 import (
    bucket_against_train_reference,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    _model_native_artifact_owner_fields,
)


ROOT = Path(__file__).resolve().parents[1]


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


def _v29_ohlc(close_values: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = np.asarray(close_values, dtype=np.float64)
    return close + 0.5, close - 0.5, close


# Fixture A: two confirmed swing highs (pivots at j=2 level 103.5 adopted bar 4,
# j=10 level 105.5 adopted bar 12), two confirmed swing lows (j=4 level 99.5
# adopted bar 6, j=12 level 102.5 adopted bar 14). High breaks at bars 8 and 14.
_V29_CLOSE_A = [
    100.0, 101.0, 103.0, 101.0, 100.0, 100.5, 102.0, 102.5, 104.0,
    104.8, 105.0, 104.0, 103.0, 103.5, 106.0, 106.5, 107.0,
]

# Fixture B: pivot lows at j=2 (99.5), j=6 (100.5, higher), j=10 (101.0,
# higher), j=14 (98.5, LOWER -> run reset), adopted at bars 4/8/12/16.
_V29_CLOSE_B = [
    103.0, 102.0, 100.0, 102.0, 103.0, 102.5, 101.0, 102.0, 103.0,
    102.5, 101.5, 103.0, 104.0, 103.0, 99.0, 101.0, 102.0, 103.0, 104.0,
]


def test_swing_v29_additions_are_opt_in_and_base_surface_is_unchanged() -> None:
    high, low, close = _v29_ohlc(_V29_CLOSE_A)
    base = compute_swing_structure_features(high, low, close)
    assert tuple(base) == SWING_FEATURE_NAMES_V1
    extended = compute_swing_structure_features(
        high, low, close, include_v29_additions=True
    )
    assert tuple(extended) == SWING_FEATURE_NAMES_V1 + SWING_V29_ADDITION_NAMES_V1
    assert not set(SWING_V29_ADDITION_NAMES_V1) & set(SWING_FEATURE_NAMES_V1)
    # The bound V1 surface is byte-identical with the additions on.
    for name in SWING_FEATURE_NAMES_V1:
        np.testing.assert_array_equal(base[name], extended[name])
    with pytest.raises(RuntimeError, match="V29_FLAG"):
        compute_swing_structure_features(
            high, low, close, include_v29_additions=1  # type: ignore[arg-type]
        )


def test_swing_v29_break_event_fires_once_per_level_and_rearms_on_new_pivot() -> None:
    high, low, close = _v29_ohlc(_V29_CLOSE_A)
    out = compute_swing_structure_features(
        high, low, close, include_v29_additions=True
    )
    # First close through 103.5 is bar 8; bars 9-10 stay above the broken
    # level but the event is disarmed (fires once per level). The new level
    # 105.5 (adopted bar 12) re-arms; first close through it is bar 14.
    assert np.flatnonzero(out["swing_high_break_event"]).tolist() == [8, 14]
    assert np.flatnonzero(out["swing_low_break_event"]).tolist() == []
    # Displacement is (close - level)/ATR at the event bar and 0 off-event.
    # At the event bar the level is still last_high, so the displacement
    # equals dist_last_swing_high_atr there — same formula, same ATR.
    assert np.flatnonzero(out["swing_break_displacement_atr"]).tolist() == [8, 14]
    for row in (8, 14):
        assert out["swing_break_displacement_atr"][row] > 0.0
        np.testing.assert_allclose(
            out["swing_break_displacement_atr"][row],
            out["dist_last_swing_high_atr"][row],
            rtol=1e-6,
        )
    # G2 ages: foundation _bars_since_event convention — cap-initialized,
    # 0 on the event bar, +1 capped after.
    ages = out["bars_since_swing_high_break"]
    assert ages[:8].tolist() == [float(FOUNDATION_EVENT_AGE_CAP)] * 8
    assert ages[8:].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0]
    assert out["bars_since_swing_low_break"].tolist() == [
        float(FOUNDATION_EVENT_AGE_CAP)
    ] * len(close)


def test_swing_v29_sequence_delta_and_counts_use_real_pivot_arithmetic() -> None:
    high, low, close = _v29_ohlc(_V29_CLOSE_A)
    out = compute_swing_structure_features(
        high, low, close, include_v29_additions=True
    )
    # No delta exists before the SECOND confirmed pivot on a side: honest NaN
    # warmup prefix, then finite forever.
    high_delta = out["swing_high_sequence_delta_atr"]
    assert np.isnan(high_delta[:12]).all()
    assert np.isfinite(high_delta[12:]).all()
    # delta = (last - prev)/ATR shares the ATR of dist_last_swing_high_atr
    # = (close - last)/ATR; at bar 12: last=105.5, prev=103.5, close=103.0.
    np.testing.assert_allclose(
        high_delta[12],
        out["dist_last_swing_high_atr"][12] * ((105.5 - 103.5) / (103.0 - 105.5)),
        rtol=1e-6,
    )
    low_delta = out["swing_low_sequence_delta_atr"]
    assert np.isnan(low_delta[:14]).all()
    assert np.isfinite(low_delta[14:]).all()
    assert (low_delta[14:] > 0.0).all()  # 102.5 is a higher low than 99.5
    # 105.5 > 103.5 is a higher high: the lower-highs run never starts.
    assert out["consecutive_lower_highs_count"].tolist() == [0.0] * len(close)
    hl_norm_1 = np.log1p(1.0) / np.log1p(float(FOUNDATION_EVENT_AGE_CAP))
    hl = out["consecutive_higher_lows_count"]
    np.testing.assert_allclose(hl[:14], 0.0, atol=0.0)
    np.testing.assert_allclose(hl[14:], np.float32(hl_norm_1), rtol=1e-6)


def test_swing_v29_higher_low_run_counts_and_resets_on_lower_low() -> None:
    high, low, close = _v29_ohlc(_V29_CLOSE_B)
    out = compute_swing_structure_features(
        high, low, close, include_v29_additions=True
    )
    cap = float(FOUNDATION_EVENT_AGE_CAP)
    norm = lambda k: np.float32(np.log1p(float(k)) / np.log1p(cap))  # noqa: E731
    hl = out["consecutive_higher_lows_count"]
    np.testing.assert_allclose(hl[:8], 0.0, atol=0.0)
    np.testing.assert_allclose(hl[8:12], norm(1), rtol=1e-6)
    np.testing.assert_allclose(hl[12:16], norm(2), rtol=1e-6)
    np.testing.assert_allclose(hl[16:], 0.0, atol=0.0)  # 98.5 < 101.0 resets
    # The armed level 101.0 (adopted bar 12) breaks down at bar 14 (close 99).
    assert np.flatnonzero(out["swing_low_break_event"]).tolist() == [14]
    assert out["swing_break_displacement_atr"][14] < 0.0
    np.testing.assert_allclose(
        out["swing_break_displacement_atr"][14],
        out["dist_last_swing_low_atr"][14],
        rtol=1e-6,
    )
    ages = out["bars_since_swing_low_break"]
    assert ages[13] == cap and ages[14] == 0.0 and ages[18] == 4.0


def test_swing_v29_additions_are_causal_and_future_append_invariant() -> None:
    high, low, close = _v29_ohlc(_V29_CLOSE_B)
    full = compute_swing_structure_features(
        high, low, close, include_v29_additions=True
    )
    keep = len(close) - 4
    prefix = compute_swing_structure_features(
        high[:keep], low[:keep], close[:keep], include_v29_additions=True
    )
    for name in SWING_FEATURE_NAMES_V1 + SWING_V29_ADDITION_NAMES_V1:
        np.testing.assert_array_equal(prefix[name], full[name][:keep])


def test_micro_structure_is_causal_exact_and_strict() -> None:
    high, low, close = _ohlc()
    observed = compute_micro_structure_features(high, low, close)
    assert tuple(observed) == MICRO_FEATURE_NAMES_V1
    assert all(np.isfinite(values).all() for values in observed.values())
    assert observed["micro_momentum_3"][:3].tolist() == [0.0, 0.0, 0.0]
    assert observed["micro_momentum_5"][:5].tolist() == [0.0] * 5
    # bps of the current close (repo ret_* convention), not raw USD diffs.
    np.testing.assert_allclose(
        observed["micro_momentum_3"][3:],
        (close[3:] - close[:-3]) / close[3:] * 1e4,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        observed["micro_acceleration"][2:],
        np.diff(np.diff(close)) / close[2:] * 1e4,
        rtol=1e-6,
    )
    # A zero-range bar has no close-location evidence: neutral 0.5, never the
    # fabricated "closed at high" 0.0.
    flat = compute_micro_structure_features(close.copy(), close.copy(), close.copy())
    assert flat["wick_ratio"].tolist() == [0.5] * len(close)

    changed_high = high.copy()
    changed_low = low.copy()
    changed_close = close.copy()
    changed_high[-1] += 20.0
    changed_low[-1] -= 20.0
    changed_close[-1] += 5.0
    changed = compute_micro_structure_features(
        changed_high,
        changed_low,
        changed_close,
    )
    for name in MICRO_FEATURE_NAMES_V1:
        np.testing.assert_array_equal(observed[name][:-1], changed[name][:-1])

    high[2] = np.nan
    with pytest.raises(RuntimeError, match="NONFINITE"):
        compute_micro_structure_features(high, low, close)


def _quote_frame() -> pd.DataFrame:
    """A quote tape with a moving spread and both asymmetry signs."""

    high, low, close = _ohlc()
    half_spread = np.array(
        [0.05, 0.04, 0.09, 0.05, 0.20, 0.06, 0.06, 0.11, 0.05],
        dtype=np.float64,
    )
    # Independent per-side extremes so the asymmetry is genuinely two-sided:
    # the ask range is wider on some bars and the bid range on others.
    ask_pad = np.array([0.3, 0.1, 0.4, 0.1, 0.5, 0.1, 0.2, 0.1, 0.3])
    bid_pad = np.array([0.1, 0.4, 0.1, 0.3, 0.1, 0.5, 0.1, 0.4, 0.1])
    return pd.DataFrame(
        {
            "close": close,
            "bid_close": close - half_spread,
            "ask_close": close + half_spread,
            "bid_high": high - half_spread,
            "bid_low": low - half_spread - bid_pad,
            "ask_high": high + half_spread + ask_pad,
            "ask_low": low + half_spread,
        }
    )


def test_spread_dynamics_is_causal_exact_and_strict() -> None:
    frame = _quote_frame()
    observed = compute_spread_dynamics_features(frame)
    assert tuple(observed) == SPREAD_DYNAMICS_FEATURE_NAMES_V1
    assert set(SPREAD_DYNAMICS_SOURCE_COLUMNS_V1) <= set(frame.columns)
    assert SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1 == ("spread_bps_delta_1",)

    bid_close = frame["bid_close"].to_numpy(dtype=np.float64)
    ask_close = frame["ask_close"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    # ONE spread owner: the level is the ctx spread_bps formula
    # ((ask - bid) / bid * 1e4), never a second definition invented here.
    spread_bps = (ask_close - bid_close) / bid_close * 1e4

    delta = observed["spread_bps_delta_1"]
    assert np.isnan(delta[:SPREAD_DYNAMICS_CAUSAL_WARMUP_ROWS_V1]).all()
    np.testing.assert_allclose(
        delta[1:],
        (spread_bps[1:] - spread_bps[:-1]).astype(np.float32),
        rtol=1e-5,
        atol=1e-5,
    )
    # It really moves on this tape - a constant field would pass a shape test.
    assert np.count_nonzero(delta[1:]) == len(delta) - 1

    np.testing.assert_allclose(
        observed["spread_intrabar_range_bps"],
        (
            (frame["ask_high"].to_numpy() - frame["bid_low"].to_numpy())
            / close
            * 1e4
        ).astype(np.float32),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        observed["quote_range_asymmetry_bps"],
        (
            (
                (frame["ask_high"].to_numpy() - frame["ask_low"].to_numpy())
                - (frame["bid_high"].to_numpy() - frame["bid_low"].to_numpy())
            )
            / close
            * 1e4
        ).astype(np.float32),
        rtol=1e-5,
    )
    # Signed, not an absolute magnitude: both signs must survive.
    assert observed["quote_range_asymmetry_bps"].min() < 0.0
    assert observed["quote_range_asymmetry_bps"].max() > 0.0
    # The envelope is non-negative by the quote geometry the producer enforces.
    assert (observed["spread_intrabar_range_bps"] >= 0.0).all()


def test_spread_dynamics_is_past_only_and_fails_closed() -> None:
    frame = _quote_frame()
    observed = compute_spread_dynamics_features(frame)

    # Causality: perturbing only the LAST bar's quotes may never change any
    # earlier row (the delta reads t and t-1; the other two read t alone).
    changed = frame.copy()
    changed.loc[changed.index[-1], "ask_close"] += 0.75
    changed.loc[changed.index[-1], "ask_high"] += 0.75
    changed.loc[changed.index[-1], "bid_low"] -= 0.75
    perturbed = compute_spread_dynamics_features(changed)
    for name in SPREAD_DYNAMICS_FEATURE_NAMES_V1:
        np.testing.assert_array_equal(observed[name][:-1], perturbed[name][:-1])

    for column in SPREAD_DYNAMICS_SOURCE_COLUMNS_V1:
        missing = frame.drop(columns=[column])
        with pytest.raises(RuntimeError, match="SOURCE_FIELDS_MISSING"):
            compute_spread_dynamics_features(missing)

    nonfinite = frame.copy()
    nonfinite.loc[nonfinite.index[2], "ask_high"] = np.nan
    with pytest.raises(RuntimeError, match="SPREAD_DYNAMICS_SOURCE_NONFINITE"):
        compute_spread_dynamics_features(nonfinite)

    # A crossed quote is a broken tape, not a value to smooth over.
    crossed = frame.copy()
    crossed.loc[crossed.index[3], "ask_high"] = (
        float(crossed.loc[crossed.index[3], "bid_high"]) - 1.0
    )
    with pytest.raises(RuntimeError, match="QUOTE_GEOMETRY_INVALID"):
        compute_spread_dynamics_features(crossed)


def test_entry_contract_is_the_only_context_subgroup_owner() -> None:
    assert MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS == (
        MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS
        + MODEL_NATIVE_CTX_CONT_MICRO_FIELDS
        + MODEL_NATIVE_CTX_CONT_SPREAD_DYNAMICS_FIELDS
        + MODEL_NATIVE_CTX_CONT_SWING_FIELDS
    )
    assert MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS == (
        MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS + MODEL_NATIVE_CTX_CONT_SESSION_FIELDS
    )
    # V30 (2026-08-13): 29 = 16 + H4_range_compression_ratio in the source
    # prefix subgroup (package 1) + the nine V29 swing event fields adopted
    # into MODEL_NATIVE_CTX_CONT_SWING_FIELDS (package 2) + the three
    # quote/spread-dynamics fields (package 4).
    assert len(MODEL_NATIVE_PREBUILT_CTX_CONT_FIELDS) == 29
    # The spread-dynamics block is its own declared subgroup, not a silent
    # extension of the five-field OHLC micro surface.
    assert MODEL_NATIVE_CTX_CONT_MICRO_FIELDS == tuple(MICRO_FEATURE_NAMES_V1)
    assert MODEL_NATIVE_CTX_CONT_SPREAD_DYNAMICS_FIELDS == (
        SPREAD_DYNAMICS_FEATURE_NAMES_V1
    )
    assert not (
        set(MODEL_NATIVE_CTX_CONT_MICRO_FIELDS)
        & set(MODEL_NATIVE_CTX_CONT_SPREAD_DYNAMICS_FIELDS)
    )
    assert len(MODEL_NATIVE_CTX_CAT_FIELDS) == 5


def test_active_context_has_no_future_or_soft_pass_through() -> None:
    htf_owner = (ROOT / "gx1/features/htf_features.py").read_text(encoding="utf-8")
    augment_owner = (ROOT / "gx1/execution/v12_ctx_augment_live.py").read_text(
        encoding="utf-8"
    )
    enriched_owner = (
        ROOT / "gx1/scripts/build_entry_exit_m1_enriched_frame_v1.py"
    ).read_text(encoding="utf-8")
    builder = (
        ROOT / "gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py"
    ).read_text(encoding="utf-8")

    for source in (htf_owner, augment_owner, enriched_owner):
        assert "shift(-" not in source
        assert "ctx-cont-dim" not in source
        assert "cv3-cross-source" not in source
    assert 'suffixes=("", "_tape")' not in builder
    assert 'if "is_ASIA" not in df.columns' not in builder
    assert "src_supplied" not in builder
    assert "fall back to canonical_v2" not in builder


def test_builder_artifact_field_owners_are_exact_and_disjoint() -> None:
    cv2_owned, source_owned = _model_native_artifact_owner_fields(
        MODEL_NATIVE_BASE_FIELDS
    )
    assert set(cv2_owned) == (
        (set(MODEL_NATIVE_BASE_FIELDS) - set(VOLUME_FEATURE_NAMES))
        | set(MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS)
    )
    assert set(source_owned) == (
        set(MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS)
        | (set(REGIME_V4_SOURCE_COLS) - {"D1_dist_from_ema200_atr"})
        | {"volume"}
    )
    assert set(cv2_owned).isdisjoint(source_owned)


def test_prebuilt_rank_bucket_uses_explicit_reference_without_missing_fallback() -> None:
    reference = np.array([1.0, 2.0, 3.0])
    observed = bucket_against_train_reference(
        np.array([1.0, 2.0, 3.0]),
        reference,
    )
    assert observed.dtype == np.int64
    assert observed.tolist() == [1, 3, 4]
    with pytest.raises(RuntimeError, match="NONFINITE"):
        bucket_against_train_reference(np.array([1.0, np.nan]), reference)
