from __future__ import annotations

import copy
import hashlib
import inspect
import json
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
    model_native_context_contract_metadata,
)
from gx1.features.micro_structure_v1 import (
    MICRO_CAUSAL_WARMUP_ROWS_V1,
    MICRO_FEATURE_NAMES_V1,
    MICRO_WARMUP_PREFIX_FIELDS_V1,
    MicroStructureCarryV1,
    SPREAD_DYNAMICS_CAUSAL_WARMUP_ROWS_V1,
    SPREAD_DYNAMICS_FEATURE_NAMES_V1,
    SPREAD_DYNAMICS_SOURCE_COLUMNS_V1,
    SPREAD_DYNAMICS_WARMUP_PREFIX_FIELDS_V1,
    SpreadDynamicsCarryV1,
    compute_micro_structure_features,
    compute_micro_structure_features_chunk,
    compute_spread_dynamics_features,
    compute_spread_dynamics_features_chunk,
    micro_structure_contract_metadata,
    require_micro_structure_contract_metadata,
)
from gx1.features.technical_indicators_v1 import classic_ema
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
from gx1.scripts.augment_forward_outcome_v2 import (
    trim_causal_context_warmup_prefix,
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
    for values in observed.values():
        finite = np.isfinite(values)
        assert not finite.any() or finite[int(np.argmax(finite)):].all()

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


def test_swing_has_no_bar_zero_pivot_and_raw_age_is_uncapped() -> None:
    tail = 99.95 - 0.05 * np.arange(615, dtype=np.float64)
    close = np.concatenate(
        (np.asarray([100.0, 101.0, 103.0, 101.0, 100.0]), tail)
    )
    out = compute_swing_structure_features(close + 0.5, close - 0.5, close)
    assert np.isnan(out["bars_since_swing_high"][:4]).all()
    assert np.isnan(out["dist_last_swing_high_atr"][:13]).all()
    assert np.isnan(out["bars_since_swing_low"]).all()
    assert out["bars_since_swing_high"][4] == 2.0
    assert out["bars_since_swing_high"][-1] == float(len(close) - 1 - 2)
    assert out["bars_since_swing_high"][-1] > 500.0


def test_swing_retracement_is_raw_and_not_clipped_to_unit_interval() -> None:
    high, low, close = _v29_ohlc(_V29_CLOSE_A)
    out = compute_swing_structure_features(high, low, close)
    assert out["swing_impulse_present"][14] == 1.0
    assert out["retracement_from_last_impulse"][14] > 1.0


def test_swing_owner_source_forbids_legacy_fill_cap_and_normalization() -> None:
    source = inspect.getsource(compute_swing_structure_features)
    helper = inspect.getsource(
        __import__(
            "gx1.features.event_age_v1",
            fromlist=["raw_event_age_bars"],
        ).raw_event_age_bars
    )
    for forbidden in (
        "min_periods=1",
        "np.clip",
        "_event_age_norm",
        "FOUNDATION_EVENT_AGE_CAP",
        "last_high = float(h[0])",
        "last_low = float(low_values[0])",
    ):
        assert forbidden not in source
    assert "wilder_atr(" in source
    assert "min(age" not in helper


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
    # The bar-8 break precedes the honest Wilder-14 ATR seed, so its
    # displacement is explicitly unavailable. Bar 14 is measured exactly.
    assert np.isnan(out["swing_high_break_displacement_atr"][:13]).all()
    assert out["swing_high_break_displacement_atr"][14] > 0.0
    np.testing.assert_allclose(
        out["swing_high_break_displacement_atr"][14],
        out["dist_last_swing_high_atr"][14],
        rtol=1e-6,
    )
    # Raw age is honestly unavailable before the first break.
    ages = out["bars_since_swing_high_break"]
    assert np.isnan(ages[:8]).all()
    assert ages[8:].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0]
    assert np.isnan(out["bars_since_swing_low_break"]).all()


def test_swing_v29_sequence_delta_and_counts_use_real_pivot_arithmetic() -> None:
    high, low, close = _v29_ohlc(_V29_CLOSE_A)
    out = compute_swing_structure_features(
        high, low, close, include_v29_additions=True
    )
    # No delta exists before the SECOND confirmed pivot on a side and the
    # Wilder denominator: one honest NaN prefix, then raw values.
    high_delta = out["swing_high_sequence_delta_atr"]
    assert np.isnan(high_delta[:13]).all()
    assert np.isfinite(high_delta[13:]).all()
    # delta = (last - prev)/ATR shares the ATR of dist_last_swing_high_atr
    # = (close - last)/ATR; row 13 is the first row where the second pivot and
    # the honest Wilder-14 denominator are both available.
    np.testing.assert_allclose(
        high_delta[13],
        out["dist_last_swing_high_atr"][13] * ((105.5 - 103.5) / (103.5 - 105.5)),
        rtol=1e-6,
    )
    low_delta = out["swing_low_sequence_delta_atr"]
    assert np.isnan(low_delta[:14]).all()
    assert np.isfinite(low_delta[14:]).all()
    assert (low_delta[14:] > 0.0).all()  # 102.5 is a higher low than 99.5
    # 105.5 > 103.5 is a higher high: the lower-highs run never starts.
    assert out["consecutive_lower_highs_count"].tolist() == [0.0] * len(close)
    hl = out["consecutive_higher_lows_count"]
    np.testing.assert_allclose(hl[:14], 0.0, atol=0.0)
    np.testing.assert_allclose(hl[14:], 1.0, rtol=0.0)


def test_swing_v29_higher_low_run_counts_and_resets_on_lower_low() -> None:
    high, low, close = _v29_ohlc(_V29_CLOSE_B)
    out = compute_swing_structure_features(
        high, low, close, include_v29_additions=True
    )
    hl = out["consecutive_higher_lows_count"]
    np.testing.assert_allclose(hl[:8], 0.0, atol=0.0)
    np.testing.assert_allclose(hl[8:12], 1.0, rtol=0.0)
    np.testing.assert_allclose(hl[12:16], 2.0, rtol=0.0)
    np.testing.assert_allclose(hl[16:], 0.0, atol=0.0)  # 98.5 < 101.0 resets
    # The armed level 101.0 (adopted bar 12) breaks down at bar 14 (close 99).
    assert np.flatnonzero(out["swing_low_break_event"]).tolist() == [14]
    assert out["swing_low_break_displacement_atr"][14] > 0.0
    np.testing.assert_allclose(
        out["swing_low_break_displacement_atr"][14],
        -out["dist_last_swing_low_atr"][14],
        rtol=1e-6,
    )
    ages = out["bars_since_swing_low_break"]
    assert np.isnan(ages[13])
    assert ages[14] == 0.0 and ages[18] == 4.0


def test_v30_package_8a_swing_emissions_are_raw_loop_state() -> None:
    """The extension exposes uncapped loop state without normalized aliases."""

    high, low, close = _v29_ohlc(_V29_CLOSE_B)
    out = compute_swing_structure_features(
        high, low, close, include_v29_additions=True
    )
    # (1) The two MISSING run counters complete the four-counter set.  Fixture
    # B's lows run 99.5 -> 100.5 -> 101.0 -> 98.5, so the higher-lows run is
    # 1, 2 then reset and the lower-lows run stays 0 until the 98.5 pivot is
    # adopted at bar 16, where it becomes 1.
    ll = out["consecutive_lower_lows_count"]
    np.testing.assert_allclose(ll[:16], 0.0, atol=0.0)
    np.testing.assert_allclose(ll[16:], 1.0, rtol=0.0)
    # The two counters on one side are mutually exclusive by construction
    # (strict > vs strict <), so they can never both be positive on a bar.
    assert not np.any(
        (out["consecutive_higher_lows_count"] > 0.0)
        & (out["consecutive_lower_lows_count"] > 0.0)
    )
    assert not np.any(
        (out["consecutive_higher_highs_count"] > 0.0)
        & (out["consecutive_lower_highs_count"] > 0.0)
    )
    # Fixture A's highs are 103.5 then 105.5 (a higher high), so the
    # higher-highs run starts exactly where the lower-highs run stays flat.
    high_a, low_a, close_a = _v29_ohlc(_V29_CLOSE_A)
    out_a = compute_swing_structure_features(
        high_a, low_a, close_a, include_v29_additions=True
    )
    assert out_a["consecutive_lower_highs_count"].tolist() == [0.0] * len(close_a)
    hh_a = out_a["consecutive_higher_highs_count"]
    np.testing.assert_allclose(hh_a[:12], 0.0, atol=0.0)
    np.testing.assert_allclose(hh_a[12:], 1.0, rtol=0.0)

    # (2) The intact flags ARE the G1 arming state with one honest NaN prefix.
    intact = out["swing_low_level_intact"]
    finite = np.isfinite(intact)
    first = int(np.argmax(finite))
    assert not finite[:first].any() and finite[first:].all()
    assert set(np.unique(intact[finite]).tolist()) <= {0.0, 1.0}
    breaks = out["swing_low_break_event"] > 0.0
    assert (intact[finite & breaks] == 0.0).all()
    delta_first = int(np.argmax(np.isfinite(out["swing_low_sequence_delta_atr"])))
    assert first <= delta_first
    high_first = int(np.argmax(np.isfinite(out["swing_high_level_intact"])))
    assert high_first <= int(
        np.argmax(np.isfinite(out["swing_high_sequence_delta_atr"]))
    )

    # (3) Ages and run lengths are raw and can exceed the old 96/500 caps;
    # no normalized age aliases are emitted.
    assert "bars_since_swing_high_norm" not in out
    assert "bars_since_swing_low_norm" not in out


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
    assert MICRO_CAUSAL_WARMUP_ROWS_V1 == 5
    assert MICRO_WARMUP_PREFIX_FIELDS_V1 == (
        "close_return_3_bps",
        "close_return_5_bps",
        "close_return_acceleration_1_bps",
        "close_distance_from_ema5_bps",
    )
    assert np.isnan(observed["close_return_3_bps"][:3]).all()
    assert np.isnan(observed["close_return_5_bps"][:5]).all()
    assert np.isnan(observed["close_return_acceleration_1_bps"][:2]).all()
    assert np.isnan(observed["close_distance_from_ema5_bps"][:4]).all()
    # Exact standard lag-close return formulas; no raw USD era proxy and no
    # misleading current-close denominator under a generic change name.
    np.testing.assert_allclose(
        observed["close_return_3_bps"][3:],
        (close[3:] / close[:-3] - 1.0) * 1e4,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        observed["close_return_acceleration_1_bps"][2:],
        (
            (close[2:] / close[1:-1] - 1.0)
            - (close[1:-1] / close[:-2] - 1.0)
        )
        * 1e4,
        rtol=1e-6,
    )
    expected_ema5 = classic_ema(pd.Series(close), 5).to_numpy(dtype=np.float64)
    np.testing.assert_allclose(
        observed["close_distance_from_ema5_bps"][4:],
        (close[4:] - expected_ema5[4:]) / close[4:] * 1e4,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        observed["close_distance_below_high_range_fraction"],
        (high - close) / (high - low),
        rtol=1e-6,
    )
    # A zero-range bar stores no fabricated location: zero is paired with an
    # explicit false availability mask, never epsilon-divided or set to 0.5.
    flat = compute_micro_structure_features(close.copy(), close.copy(), close.copy())
    assert (
        flat["close_distance_below_high_range_fraction"] == 0.0
    ).all()
    assert (flat["close_range_observed"] == 0.0).all()

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
        if name == "close_range_observed":
            assert changed[name][-1] == observed[name][-1] == 1.0
        else:
            assert changed[name][-1] != observed[name][-1]

    high[2] = np.nan
    with pytest.raises(RuntimeError, match="NONFINITE"):
        compute_micro_structure_features(high, low, close)


def test_micro_structure_chunk_carry_is_exact_and_bounded() -> None:
    high, low, close = _ohlc()
    expected = compute_micro_structure_features(high, low, close)
    boundaries = (1, 3, 4, 7, len(close))
    start = 0
    carry = MicroStructureCarryV1()
    chunks: dict[str, list[np.ndarray]] = {
        name: [] for name in MICRO_FEATURE_NAMES_V1
    }
    for stop in boundaries:
        values, carry = compute_micro_structure_features_chunk(
            high[start:stop],
            low[start:stop],
            close[start:stop],
            carry=carry,
        )
        for name in MICRO_FEATURE_NAMES_V1:
            chunks[name].append(values[name])
        assert len(carry.close_history) <= 5
        assert len(carry.ema_seed_values) < 5
        start = stop
    assert carry.rows_seen == len(close)
    assert carry.ema_value is not None
    for name in MICRO_FEATURE_NAMES_V1:
        np.testing.assert_array_equal(np.concatenate(chunks[name]), expected[name])

    with pytest.raises(RuntimeError, match="CARRY_SHAPE_INVALID"):
        compute_micro_structure_features_chunk(
            high[:1],
            low[:1],
            close[:1],
            carry=MicroStructureCarryV1(
                rows_seen=5,
                close_history=(float(close[4]),),
                ema_value=float(close[4]),
            ),
        )


def test_micro_structure_warmup_is_trimmed_as_one_honest_prefix() -> None:
    high, low, close = _ohlc()
    frame = pd.DataFrame(compute_micro_structure_features(high, low, close))
    trimmed = trim_causal_context_warmup_prefix(
        frame,
        list(MICRO_WARMUP_PREFIX_FIELDS_V1),
    )
    assert len(frame) - len(trimmed) == MICRO_CAUSAL_WARMUP_ROWS_V1
    assert trimmed.index[0] == MICRO_CAUSAL_WARMUP_ROWS_V1
    assert np.isfinite(
        trimmed[list(MICRO_WARMUP_PREFIX_FIELDS_V1)].to_numpy()
    ).all()


def test_micro_structure_internal_zero_range_has_explicit_mask_not_nan() -> None:
    high, low, close = _ohlc()
    high[6] = close[6]
    low[6] = close[6]
    observed = compute_micro_structure_features(high, low, close)
    assert observed["close_range_observed"][6] == 0.0
    assert observed["close_distance_below_high_range_fraction"][6] == 0.0
    for name in MICRO_FEATURE_NAMES_V1:
        assert np.isfinite(observed[name][MICRO_CAUSAL_WARMUP_ROWS_V1:]).all()

    frame = pd.DataFrame(observed)
    trimmed = trim_causal_context_warmup_prefix(
        frame,
        list(MICRO_WARMUP_PREFIX_FIELDS_V1),
    )
    assert len(frame) - len(trimmed) == MICRO_CAUSAL_WARMUP_ROWS_V1
    assert np.isfinite(trimmed.to_numpy(dtype=np.float64)).all()


def test_micro_structure_metadata_binds_names_formulas_warmup_and_carry() -> None:
    metadata = micro_structure_contract_metadata()
    assert require_micro_structure_contract_metadata(metadata) == metadata
    assert metadata["price_feature_names"] == list(MICRO_FEATURE_NAMES_V1)
    assert metadata["spread_feature_names"] == list(
        SPREAD_DYNAMICS_FEATURE_NAMES_V1
    )
    assert metadata["price_warmup_rows_until_all_fields_finite"] == 5
    assert metadata["price_field_nan_prefix_rows"] == {
        "close_return_3_bps": 3,
        "close_return_5_bps": 5,
        "close_return_acceleration_1_bps": 2,
        "close_distance_below_high_range_fraction": 0,
        "close_range_observed": 0,
        "close_distance_from_ema5_bps": 4,
    }
    assert metadata["classic_ema"]["span"] == 5
    assert len(str(metadata["formula_sha256"])) == 64
    assert len(str(metadata["carry_contract_sha256"])) == 64
    without_hash = copy.deepcopy(metadata)
    observed_hash = without_hash.pop("contract_sha256")
    expected_hash = hashlib.sha256(
        json.dumps(
            without_hash,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert observed_hash == expected_hash
    assert model_native_context_contract_metadata()[
        "micro_structure_owner"
    ] == metadata

    for key in (
        "price_feature_names",
        "formula_contract",
        "carry_contract",
        "classic_ema",
    ):
        changed = copy.deepcopy(metadata)
        if isinstance(changed[key], list):
            changed[key][0] = "tampered"
        elif key == "classic_ema":
            changed[key]["span"] = 6
        else:
            changed[key]["price"][0] = "tampered"
        with pytest.raises(RuntimeError, match="CONTRACT_MISMATCH"):
            require_micro_structure_contract_metadata(changed)


def test_micro_structure_runs_independently_on_native_m1_and_m5_clocks() -> None:
    rows = 30
    m1_close = 1900.0 + np.cumsum(
        np.sin(np.arange(rows, dtype=np.float64) / 2.0) + 0.25
    )
    m1_high = m1_close + 0.8
    m1_low = m1_close - 0.6
    m1 = compute_micro_structure_features(m1_high, m1_low, m1_close)

    # Independently closed five-minute bars, then the same owner. Computed M1
    # values are never sampled/rescaled into the M5 lane.
    m5_close = m1_close.reshape(-1, 5)[:, -1]
    m5_high = m1_high.reshape(-1, 5).max(axis=1)
    m5_low = m1_low.reshape(-1, 5).min(axis=1)
    m5 = compute_micro_structure_features(m5_high, m5_low, m5_close)
    assert m1["close_return_5_bps"].shape == (rows,)
    assert m5["close_return_5_bps"].shape == (rows // 5,)
    assert np.isnan(m5["close_return_5_bps"][:5]).all()
    np.testing.assert_allclose(
        m5["close_return_5_bps"][5:],
        (m5_close[5:] / m5_close[:-5] - 1.0) * 1e4,
        rtol=1e-6,
    )


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


def test_spread_dynamics_chunk_carry_is_exact_and_bounded() -> None:
    frame = _quote_frame()
    expected = compute_spread_dynamics_features(frame)
    carry = SpreadDynamicsCarryV1()
    pieces: dict[str, list[np.ndarray]] = {
        name: [] for name in SPREAD_DYNAMICS_FEATURE_NAMES_V1
    }
    start = 0
    for stop in (1, 4, 6, len(frame)):
        values, carry = compute_spread_dynamics_features_chunk(
            frame.iloc[start:stop],
            carry=carry,
        )
        for name in SPREAD_DYNAMICS_FEATURE_NAMES_V1:
            pieces[name].append(values[name])
        start = stop
    assert carry.rows_seen == len(frame)
    assert carry.previous_spread_bps is not None
    for name in SPREAD_DYNAMICS_FEATURE_NAMES_V1:
        np.testing.assert_array_equal(np.concatenate(pieces[name]), expected[name])


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
    # The spread-dynamics block is its own declared subgroup, not a silent
    # extension of the six-field OHLC micro surface.
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
    signal_owner = (
        ROOT / "gx1/contracts/entry_model_native_signal_v1.py"
    ).read_text(encoding="utf-8")
    builder = (
        ROOT / "gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py"
    ).read_text(encoding="utf-8")
    micro_owner = inspect.getsource(compute_micro_structure_features_chunk)

    for source in (htf_owner, augment_owner, enriched_owner):
        assert "shift(-" not in source
        assert "ctx-cont-dim" not in source
        assert "cv3-cross-source" not in source
    assert 'suffixes=("", "_tape")' not in builder
    assert 'if "is_ASIA" not in df.columns' not in builder
    assert "src_supplied" not in builder
    assert "fall back to canonical_v2" not in builder
    for forbidden in ("fillna", "np.clip", ".ewm", "eps", "0.5"):
        assert forbidden not in micro_owner
    assert "classic_ema(" in micro_owner
    assert "MICRO_WARMUP_PREFIX_FIELDS_V1" in augment_owner
    assert "list(MICRO_WARMUP_PREFIX_FIELDS_V1)" in augment_owner
    assert "list(MICRO_WARMUP_PREFIX_FIELDS_V1)" in builder
    assert '"micro_structure_owner": micro_structure_contract_metadata()' in (
        signal_owner
    )
    assert "if ctx_contract != exact_ctx:" in builder


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
