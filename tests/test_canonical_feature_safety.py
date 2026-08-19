import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
)
from gx1.features.basic_v1 import (
    _require_observed_spread_input,
    _validate_causal_feature_column,
)
from gx1.features.model_native_market_context_v1 import (
    derive_observed_spread_bps,
)
from gx1.features.smc_v1 import compute_smc_features
from gx1.execution.v12_ctx_augment_live import _add_spread_atr_bps
from gx1.scripts.materialize_build_canonical_features_v1 import add_high_level_basics


def test_high_level_rejects_degenerate_ohlc_before_plus5_features() -> None:
    n = 120
    df = pd.DataFrame(
        {
            "bid_close": np.full(n, 100.0),
            "ask_close": np.full(n, 100.1),
            "ask_high": np.full(n, 100.2),
            "bid_low": np.full(n, 99.8),
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.5),
            "low": np.full(n, 99.5),
            "close": np.full(n, 100.25),
            "volume": np.full(n, 10.0),
        }
    )
    df.loc[5, ["open", "high", "low", "close"]] = [90.0, 100.0, 100.0, 110.0]

    with pytest.raises(RuntimeError, match="PLUS5_OHLC_INVALID"):
        add_high_level_basics(df.copy())


def test_add_ctx_derives_spread_bps_from_valid_bid_ask() -> None:
    df = pd.DataFrame(
        {
            "bid_close": [100.0, 200.0],
            "ask_close": [100.1, 200.4],
        }
    )

    spread = derive_observed_spread_bps(df)

    np.testing.assert_allclose(spread, [10.0, 20.0])
    assert np.isfinite(spread).all()


@pytest.mark.parametrize(
    ("bid", "ask"),
    [(0.0, 10.0), (100.0, 99.5), (float("nan"), 100.0)],
)
def test_add_ctx_rejects_invalid_bid_ask_instead_of_zero_fill(
    bid: float, ask: float
) -> None:
    with pytest.raises(RuntimeError, match="MODEL_NATIVE_CONTEXT_(INVALID|NONFINITE)"):
        derive_observed_spread_bps(
            pd.DataFrame({"bid_close": [bid], "ask_close": [ask]})
        )


def test_add_ctx_existing_spread_bps_wins_over_bid_ask() -> None:
    df = pd.DataFrame(
        {
            "spread_bps": [1.25, 1.5],
            "bid_close": [100.0, 200.0],
            "ask_close": [101.0, 202.0],
        }
    )

    np.testing.assert_allclose(derive_observed_spread_bps(df), [1.25, 1.5])


def test_add_ctx_rejects_negative_existing_spread() -> None:
    with pytest.raises(RuntimeError, match="negative values"):
        derive_observed_spread_bps(pd.DataFrame({"spread_bps": [-2.0]}))


def test_add_ctx_spread_close_fallback_when_bid_ask_missing() -> None:
    df = pd.DataFrame({"spread": [0.05, 0.10], "close": [100.0, 200.0]})

    np.testing.assert_allclose(derive_observed_spread_bps(df), [5.0, 5.0])


def test_add_ctx_rejects_missing_spread_source() -> None:
    with pytest.raises(RuntimeError, match="observed spread requires"):
        derive_observed_spread_bps(pd.DataFrame({"close": [100.0]}))


def test_live_ctx_rejects_negative_bid_ask_glitches() -> None:
    df = pd.DataFrame(
        {
            "_v1_atr14": [1.0, 1.0],
            "high": [100.5, 100.5],
            "low": [99.5, 99.5],
            "close": [100.0, 100.0],
            "bid_close": [100.0, 100.0],
            "ask_close": [100.1, 99.5],
        }
    )

    with pytest.raises(RuntimeError, match="MODEL_NATIVE_CONTEXT_INVALID"):
        _add_spread_atr_bps(df)


def test_live_ctx_rejects_missing_atr_source_before_producing_features() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0],
            "bid_close": [100.0],
            "ask_close": [100.1],
        }
    )

    with pytest.raises(
        RuntimeError, match=r"MODEL_NATIVE_CONTEXT_MISSING\] ATR source fields"
    ):
        _add_spread_atr_bps(df)


def test_live_ctx_rank_formula_does_not_overwrite_canonical_atr() -> None:
    # A full Wilder-14 seed window: ctx_cont.atr_bps is the one Wilder ATR
    # owner (rule 19) and has no defined value on a two-row frame. The retired
    # `min_periods=1` partial window is exactly what made that look possible.
    frame = _ctx_atr_frame()
    canonical_atr = np.arange(len(frame), dtype=np.float64) + 9.0
    frame["atr"] = canonical_atr

    _add_spread_atr_bps(frame)

    # The offline `atr` column keeps its own producer's values: this helper
    # writes atr_bps and spread_bps only.
    np.testing.assert_array_equal(frame["atr"].to_numpy(), canonical_atr)
    warmup = 13
    assert np.isnan(frame["atr_bps"].to_numpy()[:warmup]).all()
    assert np.isfinite(frame["atr_bps"].to_numpy()[warmup:]).all()
    assert np.isfinite(frame["spread_bps"].to_numpy()).all()


def test_live_ctx_emits_no_regime_or_bucket_categorical() -> None:
    # The derived regime categoricals (trend_regime_id, atr_bucket,
    # spread_bucket) and the TRAIN rank reference that fitted the buckets are
    # retired: the live context augmenter emits raw continuous ATR/spread
    # evidence only, and the categorical contract is session_id alone.
    frame = _ctx_atr_frame()

    _add_spread_atr_bps(frame)

    for retired in (
        "trend_regime_id",
        "atr_bucket",
        "spread_bucket",
        "vol_regime_id",
        "H4_trend_sign_cat",
    ):
        assert retired not in frame.columns
        assert retired not in MODEL_NATIVE_CTX_CAT_FIELDS
    assert np.isfinite(frame[["atr_bps", "spread_bps"]].to_numpy()[13:]).all()


def test_basic_v1_spread_owner_does_not_require_slippage() -> None:
    df = pd.DataFrame(
        {
            "open": [100.0, 100.1],
            "high": [100.2, 100.3],
            "low": [99.8, 99.9],
            "close": [100.0, 100.1],
            "bid_close": [99.9, 100.0],
            "ask_close": [100.1, 100.2],
        },
        index=pd.date_range("2026-07-20", periods=2, freq="5min", tz="UTC"),
    )

    _require_observed_spread_input(df)

    assert "spread_pct" in df.columns
    assert "_v1_slip_bps" not in df.columns
    assert "_v1_cost_bps_est" not in df.columns


def test_basic_v1_ignores_post_order_slippage_as_a_feature_source() -> None:
    df = pd.DataFrame(
        {
            "close": [100.0, 100.1],
            "bid_close": [100.0, 100.0],
            "ask_close": [100.1, 100.2],
            "slippage_bps": [0.5, 0.75],
        }
    )

    _require_observed_spread_input(df)

    assert "spread_pct" in df.columns
    assert "_v1_slip_bps" not in df.columns
    assert "_v1_cost_bps_est" not in df.columns


def test_basic_v1_final_pack_preserves_only_causal_nan_prefix() -> None:
    values = np.asarray([np.nan, np.nan, 1.0, 2.0])

    observed = _validate_causal_feature_column(values, name="_v1_test")

    np.testing.assert_array_equal(observed, values)
    with pytest.raises(RuntimeError, match="BASIC_V1_FEATURE_NONFINITE_GAP"):
        _validate_causal_feature_column(
            np.asarray([np.nan, 1.0, np.nan]),
            name="_v1_test",
        )


@pytest.mark.parametrize(
    ("atr_values", "error_code"),
    [
        ([1.0, float("nan"), 1.0], "SMC_SOURCE_NONFINITE"),
        ([1.0, 0.0, 1.0], "SMC_SOURCE_INVALID"),
    ],
)
def test_smc_rejects_unavailable_atr_instead_of_using_sentinel(
    atr_values: list[float],
    error_code: str,
) -> None:
    frame = pd.DataFrame(
        {
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "atr": atr_values,
        }
    )

    with pytest.raises(RuntimeError, match=error_code):
        compute_smc_features(frame)


def test_smc_rejects_missing_atr_instead_of_using_one() -> None:
    frame = pd.DataFrame(
        {
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
        }
    )

    with pytest.raises(RuntimeError, match="SMC_SOURCE_MISSING"):
        compute_smc_features(frame)


def test_smc_preserves_only_a_causal_atr_warmup_prefix() -> None:
    frame = pd.DataFrame(
        {
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "atr": [np.nan, np.nan, 1.0, 1.0],
        }
    )

    observed = compute_smc_features(frame, include_v30_additions=True)

    assert observed["smc_sweep_up_depth_atr"].iloc[:2].isna().all()
    assert np.isfinite(observed["smc_sweep_up_depth_atr"].iloc[2:]).all()
    assert np.isfinite(observed["smc_bos_up"]).all()


def _smc_frame(bars: list[tuple[float, float, float]]) -> pd.DataFrame:
    """Build an OHLC+ATR frame from (high, low, close) rows; ATR=1.0 is an
    explicit test input (it only scales sweep size, unasserted here)."""
    high, low, close = (np.asarray(v, dtype=np.float64) for v in zip(*bars))
    return pd.DataFrame(
        {"high": high, "low": low, "close": close, "atr": np.ones(len(bars))}
    )


def test_smc_swing_state_partition_reaches_expansion_and_contraction() -> None:
    # swing_lookback=1 (explicit test input) → 3-bar pivot windows.
    # Confirmed pivots: SH 110(b1), SL 100(b3), SH 115(b5), SL 95(b6),
    # SH 110(b9), SL 100(b10) — all generic distinct prices, no ties.
    frame = _smc_frame(
        [
            (105.0, 104.0, 104.5),
            (110.0, 108.0, 109.0),   # SH pivot 110
            (106.0, 105.0, 105.5),
            (101.0, 100.0, 100.5),   # SL pivot 100
            (103.0, 101.5, 102.0),
            (115.0, 103.0, 110.0),   # SH pivot 115 (HH)
            (105.0, 95.0, 100.0),    # SL pivot 95 (LL)
            (104.0, 97.0, 100.0),
            (104.0, 98.0, 100.0),
            (110.0, 100.5, 105.0),   # SH pivot 110 (LH)
            (105.0, 100.0, 102.0),   # SL pivot 100 (HL)
            (106.0, 101.0, 103.0),
        ]
    )
    out = compute_smc_features(frame, swing_lookback=1)
    # b7-b9: HH+LL = 1 (two-sided expansion); b10: LH+LL = 3;
    # b11: LH+HL = 2 (contraction). Warmup rows stay 4.
    assert out["smc_swing_state"].tolist() == [4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 3, 2]


def test_smc_choch_requires_opposing_bos_not_pivot_pattern_flip() -> None:
    # Structure goes clean-up → mixed → clean-down, but that descriptive
    # pivot-pattern transition is not CHOCH. The next bar closes through the
    # opposing confirmed swing low; only that actual BOS changes character.
    frame = _smc_frame(
        [
            (105.0, 104.0, 104.5),
            (110.0, 106.0, 108.0),    # SH pivot 110
            (106.5, 105.8, 106.0),
            (102.0, 100.0, 101.0),    # SL pivot 100
            (107.0, 102.5, 105.0),
            (115.0, 107.0, 112.0),    # SH pivot 115 (HH)
            (109.0, 106.0, 107.0),
            (108.0, 105.0, 106.0),    # SL pivot 105 (HL) → state 0
            (110.0, 105.5, 107.0),
            (112.0, 106.0, 109.0),    # SH pivot 112 (LH) → state 2 (mixed)
            (108.0, 103.0, 105.0),    # SL pivot 103 (LL) → state 3
            (107.0, 103.5, 105.0),
            (104.0, 98.0, 99.0),      # opposing swing-low BOS → CHOCH down
        ]
    )
    out = compute_smc_features(frame, swing_lookback=1)
    state = out["smc_swing_state"].tolist()
    assert state[8] == 0 and state[10] == 2 and state[11] == 3
    choch = out["smc_choch"].to_numpy()
    assert choch[11] == 0.0
    assert out.loc[12, "smc_bos_down"] == 1.0
    assert choch[12] == 1.0
    assert float(choch.sum()) == 1.0


def test_smc_pivot_envelope_position_has_nan_warmup_and_raw_domain() -> None:
    # Strong uptrend: at b9 the confirmed swings are last_sh=110 (b4 pivot)
    # and last_sl=110.5 (b8 pivot) — last swing LOW above last swing HIGH.
    # The retired last_sh>last_sl validity fabricated 0.5 here; the 4-pivot
    # envelope [min(110.5, 99), max(110, 105)] = [99, 110.5] stays valid.
    frame = _smc_frame(
        [
            (101.0, 99.0, 100.0),
            (105.0, 101.0, 103.0),    # SH pivot 105
            (102.0, 100.5, 101.0),
            (101.5, 99.0, 100.0),     # SL pivot 99
            (110.0, 103.0, 108.0),    # SH pivot 110
            (109.5, 106.0, 108.0),
            (112.0, 107.0, 110.0),
            (115.0, 111.0, 114.0),
            (115.5, 110.5, 113.0),    # SL pivot 110.5 (above last SH 110)
            (116.0, 112.0, 115.0),    # SH pivot 116
            (110.0, 104.0, 106.0),
        ]
    )
    out = compute_smc_features(frame, swing_lookback=1)
    position = out["smc_pivot_envelope_position"].to_numpy()
    # Warmup is genuinely unavailable until both pivot pairs are confirmed.
    assert np.isnan(position[:9]).all()
    # b9: close 115 is above [99, 110.5], so the raw value remains >1.
    assert position[9] == pytest.approx((115.0 - 99.0) / 11.5, rel=1e-6)
    # b10: envelope [99, 116] (SH 116 confirmed) → interior value.
    assert position[10] == pytest.approx((106.0 - 99.0) / 17.0, rel=1e-6)


def test_smc_bos_fires_exactly_once_per_crossing() -> None:
    # Close crosses above the confirmed SH 105 at b4 and stays above through
    # b6: the crossing event fires only at b4 (the retired persistent state
    # emitted 1.0 on every bar above the level).
    frame = _smc_frame(
        [
            (100.0, 99.0, 99.5),
            (105.0, 100.0, 104.0),    # SH pivot 105
            (102.0, 101.0, 101.5),
            (103.0, 101.2, 102.0),
            (106.0, 102.0, 105.5),    # first close above 105 → event
            (107.0, 105.0, 106.5),    # still above → no event
            (108.0, 106.0, 107.0),    # still above → no event
        ]
    )
    out = compute_smc_features(frame, swing_lookback=1)
    assert out["smc_bos_up"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    # No swing low ever confirms in this monotone series.
    assert float(out["smc_bos_down"].to_numpy().sum()) == 0.0


def test_smc_sweep_event_refires_for_a_new_level_identity(monkeypatch) -> None:
    import gx1.features.smc_v1 as smc

    frame = _smc_frame(
        [
            (101.0, 99.0, 100.0),
            (105.0, 100.0, 104.0),   # first high level
            (104.0, 99.0, 103.0),
            (110.0, 100.0, 104.0),   # sweep first level; next high level
            (111.0, 101.0, 109.0),   # sweep newly confirmed level
            (109.0, 100.0, 108.0),
        ]
    )

    def fixed_pivots(_high, _low, _lookback):
        high_mask = np.zeros(len(frame), dtype=bool)
        low_mask = np.zeros(len(frame), dtype=bool)
        high_mask[[1, 3]] = True
        return high_mask, low_mask

    monkeypatch.setattr(smc, "_detect_swing_pivots", fixed_pivots)
    out = smc.compute_smc_features(
        frame,
        swing_lookback=1,
        include_v30_additions=True,
    )
    assert out.loc[3:4, "smc_sweep_up_state"].tolist() == [1.0, 1.0]
    assert out.loc[3:4, "smc_sweep_up_event"].tolist() == [1.0, 1.0]


def _ctx_atr_frame(rows: int = 40) -> pd.DataFrame:
    """A valid non-degenerate OHLC + two-sided quote frame.

    Deterministic, not random: the true range must move from bar to bar so a
    partial-window mean and a Wilder RMA cannot coincide by accident.
    """

    step = np.arange(rows, dtype=np.float64)
    close = 2000.0 + np.sin(step / 3.0) * 4.0 + step * 0.1
    half_range = 0.5 + (step % 5) * 0.4
    high = close + half_range
    low = close - half_range * 0.7
    half_spread = 0.05 + (step % 7) * 0.01
    return pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "bid_close": close - half_spread,
            "ask_close": close + half_spread,
        }
    )


def test_ctx_atr_bps_is_the_one_wilder_owner_without_partial_window() -> None:
    """ctx_cont.atr_bps must be the SAME ATR as `_v1_atr14` (rule 19).

    Regression for the 2026-08-19 repair: this owner used to run its own
    ``true_range.rolling(14, min_periods=1).mean()``, a simple moving average
    over a partial window, while `_v1_atr14` at index 0 of the same signal
    vector — and every ``*_atr``-normalized field on the surface — is the
    classic Wilder RMA from ``technical_indicators_v1.wilder_atr``.
    """

    from gx1.features.model_native_market_context_v1 import (
        derive_model_native_atr_spread_bps,
    )
    from gx1.features.technical_indicators_v1 import wilder_atr

    frame = _ctx_atr_frame()
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    # The comparison arm is exactly basic_v1's `_v1_atr14`:
    # wilder_atr(high, low, close, 14) on the same rows and the same clock.
    expected_atr = wilder_atr(
        frame["high"].astype(np.float64),
        frame["low"].astype(np.float64),
        frame["close"].astype(np.float64),
        14,
    ).to_numpy(dtype=np.float64)
    warmup = int(np.flatnonzero(np.isfinite(expected_atr))[0])
    assert warmup == 13

    derived = derive_model_native_atr_spread_bps(frame)
    atr = derived["atr"].to_numpy(dtype=np.float64)
    atr_bps = derived["atr_bps"].to_numpy(dtype=np.float64)

    # Bit-identical to the one ATR owner, on every row it is defined.
    np.testing.assert_array_equal(np.isnan(atr), np.isnan(expected_atr))
    np.testing.assert_array_equal(atr[warmup:], expected_atr[warmup:])
    # An honest unavailable prefix, never a partial-window mean that reads as a
    # converged ATR.
    assert np.isnan(atr[:warmup]).all()
    assert np.isnan(atr_bps[:warmup]).all()
    # atr_bps is that same ATR expressed over the bar CLOSE (see the
    # one-denominator regression below).
    np.testing.assert_array_equal(
        atr_bps[warmup:],
        (expected_atr / frame["close"].to_numpy(dtype=np.float64) * 1e4)[
            warmup:
        ],
    )
    # And it is NOT the retired midrange form.
    assert not np.array_equal(
        atr_bps[warmup:],
        (expected_atr / ((high + low) * 0.5) * 1e4)[warmup:],
    )


def test_ctx_atr_bps_uses_the_one_repository_bps_denominator() -> None:
    """`atr_bps` must be bps of CLOSE, the repository's one `*_bps` base.

    Regression for the 2026-08-19 one-concept/two-conventions repair: this
    owner divided the Wilder ATR by the bar midpoint ``(high + low) / 2``
    while the per-timeframe sibling ``atr_bps_14`` in the SAME signal
    vector divides by ``close`` -- and so does every other ``*_bps`` owner in
    this repository, including ``derive_observed_spread_bps`` in this very
    file.  Verified on 2026-08-19 from real emitted bytes, not from a restated
    literal: the per-TF ``atr_bps_14`` column of the V31 MULTI_TF_V4_CACHE
    reproduces as ``wilder_atr(...)/close*1e4`` to float32 resolution on all
    477,229 native M5 rows and misses ``/mid`` on 96.28% of them.

    The midrange form was a direction leak, not just a rescale: exactly
    ``atr/mid == (atr/close) * (close/mid)``, and ``close/mid - 1`` is a
    monotone re-expression of the intrabar close position (Spearman 0.9170 on
    the full declared M5 tape) -- a quantity this ctx vector already owns as
    ``close_distance_below_high_range_fraction``.
    """

    from gx1.features.model_native_market_context_v1 import (
        MODEL_NATIVE_ATR_CAUSAL_WARMUP_ROWS_V1,
        MODEL_NATIVE_ATR_PERIOD_V1,
        derive_model_native_atr_spread_bps,
    )
    from gx1.features.technical_indicators_v1 import wilder_atr

    frame = _ctx_atr_frame()
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    close = frame["close"].to_numpy(dtype=np.float64)
    warmup = MODEL_NATIVE_ATR_CAUSAL_WARMUP_ROWS_V1

    # The frame must actually separate the two conventions, or this test would
    # pass on a degenerate `mid == close` fixture and prove nothing.
    mid = (high + low) * 0.5
    assert np.abs(mid - close).min() > 0.0

    expected_atr = wilder_atr(
        frame["high"].astype(np.float64),
        frame["low"].astype(np.float64),
        frame["close"].astype(np.float64),
        MODEL_NATIVE_ATR_PERIOD_V1,
    ).to_numpy(dtype=np.float64)

    atr_bps = derive_model_native_atr_spread_bps(frame)["atr_bps"].to_numpy(
        dtype=np.float64
    )
    np.testing.assert_array_equal(
        atr_bps[warmup:], (expected_atr / close * 1e4)[warmup:]
    )
    # The retired midrange convention must be rejected on every defined row,
    # not merely "close enough".
    midrange_form = (expected_atr / mid * 1e4)[warmup:]
    assert np.all(atr_bps[warmup:] != midrange_form)

    # `spread_bps` in this same owner already uses a quoted CLOSE price, so the
    # two bps fields this function returns now share one base.
    spread_bps = derive_model_native_atr_spread_bps(frame)[
        "spread_bps"
    ].to_numpy(dtype=np.float64)
    bid = frame["bid_close"].to_numpy(dtype=np.float64)
    ask = frame["ask_close"].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(spread_bps, (ask - bid) / bid * 1e4)


def test_ctx_atr_warmup_contract_matches_the_owner_it_declares() -> None:
    """The declared prefix contract must be the ATR owner's real warmup."""

    from gx1.features.model_native_market_context_v1 import (
        MODEL_NATIVE_ATR_CAUSAL_WARMUP_ROWS_V1,
        MODEL_NATIVE_ATR_PERIOD_V1,
        MODEL_NATIVE_ATR_WARMUP_PREFIX_FIELDS_V1,
        derive_model_native_atr_spread_bps,
    )

    assert MODEL_NATIVE_ATR_WARMUP_PREFIX_FIELDS_V1 == ("atr_bps",)
    assert (
        MODEL_NATIVE_ATR_CAUSAL_WARMUP_ROWS_V1
        == MODEL_NATIVE_ATR_PERIOD_V1 - 1
    )
    frame = _ctx_atr_frame()
    derived = derive_model_native_atr_spread_bps(frame)
    for name in MODEL_NATIVE_ATR_WARMUP_PREFIX_FIELDS_V1:
        values = derived[name].to_numpy(dtype=np.float64)
        assert np.isnan(values[:MODEL_NATIVE_ATR_CAUSAL_WARMUP_ROWS_V1]).all()
        assert np.isfinite(
            values[MODEL_NATIVE_ATR_CAUSAL_WARMUP_ROWS_V1:]
        ).all()
    # `spread_bps` is defined on every row and must NOT gain a prefix.
    assert np.isfinite(derived["spread_bps"].to_numpy()).all()
    # Too few rows for a defined Wilder seed fails closed instead of emitting a
    # partial-window value.
    with pytest.raises(RuntimeError, match="MODEL_NATIVE_CONTEXT_SHORT"):
        derive_model_native_atr_spread_bps(
            _ctx_atr_frame(MODEL_NATIVE_ATR_CAUSAL_WARMUP_ROWS_V1)
        )


# ---------------------------------------------------------------------------
# High-level base block (gx1.scripts.materialize_build_canonical_features_v1
# .add_high_level_basics) — 2026-08-19 repair wave.
# ---------------------------------------------------------------------------


def _high_level_frame(rows: int = 320) -> pd.DataFrame:
    """A valid non-degenerate OHLCV + two-sided quote frame.

    Deterministic, not random, and long enough for the classic EMA200 seed so
    the warmup assertions below are exercised rather than skipped.
    """

    step = np.arange(rows, dtype=np.float64)
    close = 2500.0 + step * 0.1 + np.sin(step * 0.31) * 3.0
    open_ = close - 0.07 * np.cos(step * 0.2)
    high = np.maximum(open_, close) + 0.4 + 0.1 * (step % 5)
    low = np.minimum(open_, close) - 0.3 - 0.1 * (step % 7)
    half_spread = 0.05 + 0.01 * (step % 7)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(rows, 10.0),
            "bid_close": close - half_spread,
            "ask_close": close + half_spread,
            "ask_high": high + half_spread,
            "bid_low": low - half_spread,
        }
    )


def _first_finite_row(values: np.ndarray) -> int:
    finite = np.flatnonzero(np.isfinite(values))
    assert finite.size, "column is unavailable on every row"
    assert np.isfinite(values[finite[0]:]).all(), "non-contiguous warmup prefix"
    return int(finite[0])


def test_high_level_retires_the_fields_owned_by_another_specialist() -> None:
    """Four base columns were exact functions of live fields owned elsewhere.

    ``ret_5`` == ``ctx_cont.close_return_5_bps`` (micro_structure_v1),
    ``body_pct`` == ``abs(candle.raw_body_signed_range)`` and
    ``wick_asym`` == ``(upper_wick_share - lower_wick_share)`` over their sum
    (entry_candle_primitives_v1), and ``pos_vs_ema200``'s repaired form is the
    local EMA layer's own ``chart.local_price_vs_ema200_*``.  Rule 19: one
    specialist owner per field.
    """

    observed = add_high_level_basics(_high_level_frame())

    for retired in ("ret_5", "body_pct", "wick_asym", "pos_vs_ema200"):
        assert retired not in observed.columns
    # The retired unit spellings must not survive under their old names either.
    for retired_name in ("ema20_slope", "ema100_slope"):
        assert retired_name not in observed.columns
    # Both surviving return horizons are genuinely un-duplicated: the micro
    # family carries the 3-bar and 5-bar returns and the 1-bar acceleration,
    # never the 1-bar or 20-bar return itself.
    close = _high_level_frame()["close"].to_numpy(dtype=np.float64)
    for name, lag in (("ret_1", 1), ("ret_20", 20)):
        expected = (
            pd.Series(close).pct_change(lag).to_numpy(dtype=np.float64) * 1e4
        ).astype(np.float32)
        np.testing.assert_array_equal(
            observed[name].to_numpy(dtype=np.float32), expected
        )


def test_high_level_ema_block_is_the_classic_seeded_ema_over_positive_atr() -> None:
    """The trend block must be ATR multiples of the ONE classic EMA owner.

    Two independent repairs are pinned here.  (1) The block used
    ``close.ewm(span=..., adjust=False)`` — a seedless recursion that emits a
    value on row 0 — while every other EMA consumer on this surface uses the
    SMA-seeded ``technical_indicators_v1.classic_ema``.  (2) The slopes were
    ``delta / close * 1e4``, so their magnitude carried the volatility regime;
    they are now divided by the strictly-positive Wilder-14 ATR from
    ``wilder_atr14_positive``, never by an epsilon-floored price.
    """

    from gx1.features.technical_indicators_v1 import (
        classic_ema,
        wilder_atr14_positive,
    )

    frame = _high_level_frame()
    close = pd.Series(frame["close"].to_numpy(dtype=np.float64))
    _atr14, atr14_positive = wilder_atr14_positive(
        pd.Series(frame["high"].to_numpy(dtype=np.float64)),
        pd.Series(frame["low"].to_numpy(dtype=np.float64)),
        close,
    )
    observed = add_high_level_basics(frame.copy())

    for name, span, lookback in (
        ("ema20_slope_atr", 20, 5),
        ("ema100_slope_atr", 100, 20),
    ):
        ema = classic_ema(close, span)
        expected = (
            (ema - ema.shift(lookback)).div(atr14_positive)
        ).to_numpy(dtype=np.float64).astype(np.float32)
        values = observed[name].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(np.isnan(values), np.isnan(expected))
        np.testing.assert_array_equal(values, expected)
        # First finite row is DERIVED from the span and the lookback: the SMA
        # seed lands at span-1, the k-bar difference moves it to
        # span-1+lookback.  The seedless recursion had no warmup at all.
        assert _first_finite_row(values.astype(np.float64)) == span - 1 + lookback

    # The retired price-relative unit is not what is emitted.  A slope in ATR
    # multiples and the same slope in bps of price differ on every defined row
    # of this fixture.
    retired_ema20 = close.ewm(span=20, adjust=False).mean()
    retired_unit = (
        (retired_ema20 - retired_ema20.shift(5)) / np.maximum(close, 1e-9) * 1e4
    ).to_numpy(dtype=np.float64).astype(np.float32)
    emitted = observed["ema20_slope_atr"].to_numpy(dtype=np.float32)
    defined = np.isfinite(emitted) & np.isfinite(retired_unit)
    assert defined.any()
    assert not np.array_equal(emitted[defined], retired_unit[defined])


def test_high_level_atr_z_has_no_partial_window_no_epsilon_and_no_clip() -> None:
    """``atr50``/``atr_z`` must be full-window, unfloored and unclipped."""

    frame = _high_level_frame()
    high = pd.Series(frame["high"].to_numpy(dtype=np.float64))
    low = pd.Series(frame["low"].to_numpy(dtype=np.float64))
    close = pd.Series(frame["close"].to_numpy(dtype=np.float64))
    true_range = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr50 = true_range.rolling(50, min_periods=50).mean()
    expected_z = (
        (atr50 - atr50.rolling(50, min_periods=50).mean()).div(
            atr50.rolling(50, min_periods=50).std().where(
                atr50.rolling(50, min_periods=50).std() > 0.0
            )
        )
    ).to_numpy(dtype=np.float64).astype(np.float32)

    observed = add_high_level_basics(frame.copy())

    np.testing.assert_array_equal(
        observed["atr50"].to_numpy(dtype=np.float32),
        atr50.to_numpy(dtype=np.float64).astype(np.float32),
    )
    np.testing.assert_array_equal(
        observed["atr_z"].to_numpy(dtype=np.float32), expected_z
    )
    # Warmup is derived from the declared window, not from a min_periods
    # choice: the 50-bar mean lands at 49 and its own 50-bar moments at 98.
    assert _first_finite_row(observed["atr50"].to_numpy(dtype=np.float64)) == 49
    assert _first_finite_row(observed["atr_z"].to_numpy(dtype=np.float64)) == 98


def test_high_level_atr_z_is_unavailable_instead_of_epsilon_divided() -> None:
    """A constant true range leaves the z-score undefined, never 0.0."""

    rows = 160
    # Constant OHLC geometry and no gaps => constant true range => the rolling
    # standard deviation of atr50 is exactly 0 on every full window.
    close = np.full(rows, 2500.0)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 10.0),
            "bid_close": close - 0.05,
            "ask_close": close + 0.05,
            "ask_high": close + 1.05,
            "bid_low": close - 1.05,
        }
    )

    observed = add_high_level_basics(frame.copy())

    atr_z = observed["atr_z"].to_numpy(dtype=np.float64)
    assert np.isnan(atr_z).all(), "an undefined z-score must not read as 0.0"
    # Same convention for the two other retired epsilon floors on this block.
    rvol_60 = observed["rvol_60"].to_numpy(dtype=np.float64)
    vol_ratio = observed["vol_ratio"].to_numpy(dtype=np.float64)
    zero_vol = np.isfinite(rvol_60) & (rvol_60 == 0.0)
    assert zero_vol.any()
    assert np.isnan(vol_ratio[zero_vol]).all()


def test_high_level_range_reports_an_invalid_quote_instead_of_1e13_bps() -> None:
    """A non-positive mid is an invalid two-sided quote, not a small one."""

    frame = _high_level_frame(rows=120)
    frame.loc[7, "bid_close"] = -3.0
    frame.loc[7, "ask_close"] = 1.0

    observed = add_high_level_basics(frame.copy())

    bar_range = observed["range"].to_numpy(dtype=np.float64)
    assert np.isnan(bar_range[7])
    assert np.isfinite(np.delete(bar_range, 7)).all()
    # The retired epsilon floor turned exactly this row into a finite,
    # astronomically large "range".
    retired = (
        (frame.loc[7, "ask_high"] - frame.loc[7, "bid_low"])
        / max((frame.loc[7, "bid_close"] + frame.loc[7, "ask_close"]) / 2.0, 1e-9)
        * 1e4
    )
    assert abs(retired) > 1e9
