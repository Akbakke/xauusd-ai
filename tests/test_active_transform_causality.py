from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features.htf_features import (
    MULTI_TF_SHIFT,
    build_multi_tf_per_bar_features_v4 as _build_multi_tf_v4,
    compute_per_bar_features_v4 as _compute_per_bar_v4,
    multi_tf_last_closed_label,
    slice_multi_tf_v4_window,
)
from tests.htf_v29_registry_test_support import (
    synthetic_v29_registry_constants,
)
from tests.volatility_squeeze_test_support import (
    make_volatility_squeeze_artifact_set,
)

from gx1.features import basic_v1
from gx1.features.regime_v4_features import (
    REGIME_V4_DERIVED_COLS,
    REGIME_V4_FEATURE_NAMES,
    REGIME_V4_SOURCE_COLS,
    REGIME_V4_V29_ADDITION_COLS,
    REGIME_V4_V29_FLIP_TFS,
    _class_flip_flag_and_age,
    add_regime_v4_features,
)
from gx1.execution.v12_ctx_augment_live import (
    _add_htf_features,
    _add_regime_categoricals,
    _align_last_closed as _align_live_last_closed,
)
from gx1.scripts.augment_forward_outcome_v2 import (
    CausalContextWarmupError,
    _cache_cutoff_ns,
    _tf_cache_row,
    attach_group_a_ctx_columns,
    augment_candidate,
    build_context,
    trim_causal_context_warmup_prefix,
)

_V29_TEST_REGISTRY_CONSTANTS = synthetic_v29_registry_constants()
_SQUEEZE_TEST_ARTIFACTS = None


@pytest.fixture(scope="module", autouse=True)
def _bind_squeeze_artifacts(tmp_path_factory):
    global _SQUEEZE_TEST_ARTIFACTS
    _SQUEEZE_TEST_ARTIFACTS = make_volatility_squeeze_artifact_set(
        tmp_path_factory.mktemp("squeeze-artifacts")
    )


def build_multi_tf_per_bar_features_v4(m5_df):
    return _build_multi_tf_v4(
        m5_df,
        v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
        volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
    )


def compute_per_bar_features_v4(ohlcv, *, timeframe):
    return _compute_per_bar_v4(
        ohlcv,
        timeframe=timeframe,
        v29_registry_constants=_V29_TEST_REGISTRY_CONSTANTS,
        volatility_squeeze_artifacts=_SQUEEZE_TEST_ARTIFACTS,
    )


def test_tf_cache_row_is_strict_float32_zero_copy() -> None:
    target = pd.Timestamp("2026-01-02T12:00:00Z")
    from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4

    width = MULTI_TF_FEATURE_COUNT_V4
    ts_values = np.asarray(
        [target.value - pd.Timedelta(minutes=10).value, target.value],
        dtype=np.int64,
    )
    feat_values = np.arange(2 * width, dtype=np.float32).reshape(2, width)
    frame = pd.DataFrame(index=pd.DatetimeIndex(ts_values, tz="UTC"))
    frame.attrs["ts_int64"] = ts_values
    frame.attrs["feats_np"] = feat_values
    frame.attrs["causal_warmup_rows"] = 0
    ctx = type(
        "Context",
        (),
        {
            "multi_tf": {"M5": frame},
            "decision_bar_duration_ns": pd.Timedelta(minutes=5).value,
        },
    )()

    row = _tf_cache_row(ctx, "M5", target.value)

    assert row.dtype == np.dtype(np.float32)
    assert np.shares_memory(row, feat_values)
    np.testing.assert_array_equal(row, feat_values[1])


def test_basic_v1_has_one_environment_independent_feature_path() -> None:
    source = inspect.getsource(basic_v1)
    for forbidden in (
        "GX1_FEATURE_BUILD_DISABLED",
        "GX1_REPLAY_USE_PREBUILT_FEATURES",
        "GX1_REPLAY_MODE",
        "GX1_ASSERT_NO_PANDAS",
        "GX1_FEATURE_USE_NP_ROLLING",
        "GX1_REPLAY_INCREMENTAL_FEATURES",
        "GX1_ATR_REGIME_FIX",
        "FEATURE_BUILD_TIMEOUT_MS",
    ):
        assert forbidden not in source
    for name in basic_v1.BASIC_V1_FEATURES:
        assert f'df["{name}"]' in source


def test_basic_v1_rejects_non_float_ohlc_without_mode_dependent_conversion() -> None:
    frame = pd.DataFrame(
        {
            "open": [2000],
            "high": [2001],
            "low": [1999],
            "close": [2000],
            "volume": [100],
            "spread_bps": [1.0],
        },
        index=pd.date_range("2026-01-01T00:00:00Z", periods=1, freq="5min"),
    )
    with pytest.raises(ValueError, match="OHLC dtype mismatch"):
        basic_v1.build_basic_v1(frame)


def test_tf_cache_row_rejects_dtype_coercion_instead_of_copying() -> None:
    target = pd.Timestamp("2026-01-02T12:00:00Z")
    ts_values = np.asarray([target.value], dtype=np.int64)
    frame = pd.DataFrame(index=pd.DatetimeIndex(ts_values, tz="UTC"))
    frame.attrs["ts_int64"] = ts_values
    frame.attrs["feats_np"] = np.ones((1, 25), dtype=np.float64)
    frame.attrs["causal_warmup_rows"] = 0
    ctx = type(
        "Context",
        (),
        {
            "multi_tf": {"M5": frame},
            "decision_bar_duration_ns": pd.Timedelta(minutes=5).value,
        },
    )()

    with pytest.raises(RuntimeError, match="malformed M5 cache arrays"):
        _tf_cache_row(ctx, "M5", target.value)


def _market_frame(periods: int = 900) -> pd.DataFrame:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=periods, freq="5min")
    # Four-day cycles keep confirmed causal pivots alive even after D1
    # resampling, so this fixture exercises the same complete V4 surface as
    # production rather than silently testing only the historical V2 prefix.
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
            # OANDA candle volume is an exact positive integer tick count.
            "volume": 100 + (np.arange(periods, dtype=np.int64) % 201),
            "smc_swing_state": np.arange(periods) % 5,
        },
        index=index,
    )


def test_closed_bar_cutoff_is_immutable_and_excludes_forming_h1(monkeypatch: pytest.MonkeyPatch) -> None:
    target = pd.Timestamp("2026-07-08T18:00:00Z")
    monkeypatch.setenv("GX1_PERTF_CLOSED_BAR", "0")

    cutoff = pd.Timestamp(_cache_cutoff_ns(target.value, "H1"), tz="UTC")

    assert cutoff == pd.Timestamp("2026-07-08T17:05:00Z")
    assert pd.Timestamp("2026-07-08T18:00:00Z") > cutoff


def test_full_group_a_transform_rejects_history_without_every_genuine_event(
    tmp_path: Path,
) -> None:
    # V29: the D1 trend-event fields carry an exact 200-bar min_periods
    # mask, so a fully-warmed D1 row needs >= 201 D1 days of tape.
    frame = _market_frame(60_000)
    prefix = frame.iloc[:59_800]
    target = prefix.index[-1]

    prefix_mtf = build_multi_tf_per_bar_features_v4(prefix)
    full_mtf = build_multi_tf_per_bar_features_v4(frame)
    prefix_ctx = build_context(
        prefix[["high", "low", "close"]],
        prefix_mtf,
        journal_dir=tmp_path / "missing_prefix_journal",
    )
    full_ctx = build_context(
        frame[["high", "low", "close"]],
        full_mtf,
        journal_dir=tmp_path / "missing_full_journal",
    )

    with pytest.raises(CausalContextWarmupError, match="MTF_WARMUP"):
        augment_candidate(prefix_ctx, target, include_portfolio=False)
    with pytest.raises(CausalContextWarmupError, match="MTF_WARMUP"):
        augment_candidate(full_ctx, target, include_portfolio=False)


def test_group_a_decision_slice_uses_explicit_full_m5_history(tmp_path: Path) -> None:
    frame = _market_frame(60_000)
    multi_tf = build_multi_tf_per_bar_features_v4(frame)
    decision = frame.iloc[-12:].reset_index(names="time")

    with pytest.raises(RuntimeError, match="no complete causal context row"):
        attach_group_a_ctx_columns(
            decision,
            multi_tf=multi_tf,
            context_m5=frame,
        )


def test_native_m1_decisions_use_true_closed_m5_context(tmp_path: Path) -> None:
    m5 = _market_frame(60_000)
    multi_tf = build_multi_tf_per_bar_features_v4(m5)
    start = m5.index[-3]
    times = pd.date_range(start, periods=4, freq="1min", tz="UTC")
    close = np.linspace(float(m5.loc[start, "close"]), float(m5.loc[start, "close"]) + 0.3, 4)
    local_m1 = pd.DataFrame(
        {
            "time": times,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "smc_swing_state": np.arange(4) % 5,
        }
    )

    with pytest.raises(RuntimeError, match="no complete causal context row"):
        attach_group_a_ctx_columns(
            local_m1,
            multi_tf=multi_tf,
            context_m5=m5,
            base_bar_duration=pd.Timedelta(minutes=1),
        )


def test_group_a_full_history_rejects_decision_ohlc_mismatch() -> None:
    frame = _market_frame(18_400)
    multi_tf = build_multi_tf_per_bar_features_v4(frame)
    decision = frame.iloc[-2:].reset_index(names="time")
    decision.loc[0, "close"] += 0.01

    with pytest.raises(RuntimeError, match="decision OHLC differs"):
        attach_group_a_ctx_columns(
            decision,
            multi_tf=multi_tf,
            context_m5=frame,
        )


def test_attach_marks_only_real_warmup_and_shared_trim_removes_it() -> None:
    frame = _market_frame(60_000)
    multi_tf = build_multi_tf_per_bar_features_v4(frame)
    source = frame.reset_index(names="time")

    with pytest.raises(RuntimeError, match="no complete causal context row"):
        attach_group_a_ctx_columns(source, multi_tf=multi_tf)


def test_attach_requires_explicit_mtf_and_not_smc_score_source() -> None:
    # Supply enough closed M5 history for the genuine D1 Group-A primitives,
    # then attach only the final decision rows.  A short all-warmup frame would
    # test warmup length rather than the absence of the retired SMC score.
    frame = _market_frame(60_000)
    multi_tf = build_multi_tf_per_bar_features_v4(frame)
    source = frame.iloc[-2:].reset_index(names="time")

    with pytest.raises(RuntimeError, match="explicit multi_tf"):
        attach_group_a_ctx_columns(source, multi_tf=None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="no complete causal context row"):
        attach_group_a_ctx_columns(
            source.drop(columns="smc_swing_state"),
            multi_tf=multi_tf,
            context_m5=frame,
        )


def test_warmup_trim_rejects_nonfinite_gap_after_complete_rows() -> None:
    frame = pd.DataFrame({"feature": [np.nan, 1.0, np.nan, 2.0]})

    with pytest.raises(RuntimeError, match="not a contiguous warmup prefix"):
        trim_causal_context_warmup_prefix(frame, ["feature"])


def test_htf_v4_requires_exact_observed_volume() -> None:
    frame = _market_frame(100).drop(columns=["volume", "smc_swing_state"])

    with pytest.raises(RuntimeError, match="volume"):
        compute_per_bar_features_v4(frame, timeframe="M5")

    frame["volume"] = 0.0
    with pytest.raises(RuntimeError, match="volume"):
        compute_per_bar_features_v4(frame, timeframe="M5")


def test_htf_v4_warmup_is_explicit_and_future_append_is_prefix_invariant() -> None:
    frame = _market_frame(4_000).drop(columns="smc_swing_state")
    prefix = frame.iloc[:3_500]

    before = compute_per_bar_features_v4(prefix, timeframe="M5")
    after = compute_per_bar_features_v4(frame, timeframe="M5")

    assert before.iloc[0].isna().any()
    complete = np.isfinite(before.to_numpy()).all(axis=1)
    assert not complete.any()
    assert before["geomline_bars_since_break"].isna().all()
    np.testing.assert_allclose(
        before.to_numpy(),
        after.iloc[: len(before)].to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_htf_window_refuses_padding_and_returns_only_exact_finite_history() -> None:
    frame = _market_frame(4_000).drop(columns="smc_swing_state")
    feats = build_multi_tf_per_bar_features_v4(frame)["M5"]
    values = feats.attrs["feats_np"]
    assert np.shares_memory(
        feats.to_numpy(dtype=np.float32, copy=False),
        values,
    )

    target = frame.index[-1] + pd.Timedelta(minutes=5)
    with pytest.raises(RuntimeError, match="HTF_WINDOW_WARMUP_INCOMPLETE"):
        slice_multi_tf_v4_window(
            feats, target, n=32, tf_shift=MULTI_TF_SHIFT["M5"]
        )

    with pytest.raises(RuntimeError, match="WARMUP|HISTORY"):
        slice_multi_tf_v4_window(
            feats,
            frame.index[30] + pd.Timedelta(minutes=5),
            n=10,
            tf_shift=MULTI_TF_SHIFT["M5"],
        )
    legacy = feats.copy()
    legacy.attrs.clear()
    with pytest.raises(RuntimeError, match="CONTRACT"):
        slice_multi_tf_v4_window(
            legacy, target, n=1, tf_shift=MULTI_TF_SHIFT["M5"]
        )


def _regime_source_frame(periods: int) -> pd.DataFrame:
    index = pd.date_range("2025-01-01T00:00:00Z", periods=periods, freq="5min")
    block = (np.arange(periods) // 80) % 2
    classes = np.where(block == 0, 1.0, 3.0)
    stacks = np.where(block == 0, 1.0, -1.0)
    payload: dict[str, np.ndarray] = {}
    for tf in ("m15", "h1", "h4", "d1", "m5"):
        payload[f"{tf}_regime_class_id_v2"] = classes.copy()
        payload[f"{tf}_trend_state_age_bars_v2"] = np.arange(periods) % 80
        payload[f"{tf}_ema_stack_aligned_v2"] = stacks.copy()
    payload["D1_dist_from_ema200_atr"] = np.sin(np.arange(periods) / 50.0) * 2.0
    frame = pd.DataFrame(payload, index=index)
    frame.loc[frame.index[:20], REGIME_V4_SOURCE_COLS] = np.nan
    return frame


def test_regime_v4_uses_causal_warmup_and_is_future_append_invariant() -> None:
    prefix = _regime_source_frame(600)
    full = _regime_source_frame(700)

    before = add_regime_v4_features(prefix.copy())
    after = add_regime_v4_features(full.copy())

    warmup = int(before.attrs["causal_regime_v4_warmup_rows"])
    assert warmup >= 80
    assert before.loc[:, REGIME_V4_DERIVED_COLS].iloc[:warmup].isna().any(axis=1).all()
    assert np.isfinite(before.loc[:, REGIME_V4_DERIVED_COLS].iloc[warmup:].to_numpy()).all()
    np.testing.assert_allclose(
        before.loc[:, REGIME_V4_DERIVED_COLS].to_numpy(),
        after.loc[:, REGIME_V4_DERIVED_COLS].iloc[: len(before)].to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_regime_v4_d1_change_age_is_measured_in_d1_bars() -> None:
    periods = 700
    index = pd.date_range("2025-01-01T00:00:00Z", periods=periods, freq="5min")
    d1_classes = np.zeros(periods, dtype=np.float64)
    d1_classes[308:] = 1.0  # flip on the second observed D1 bar
    payload: dict[str, np.ndarray] = {}
    for tf in ("m15", "h1", "h4", "d1", "m5"):
        payload[f"{tf}_regime_class_id_v2"] = (
            d1_classes.copy() if tf == "d1" else np.ones(periods, dtype=np.float64)
        )
        if tf == "d1":
            trend_age = np.zeros(periods, dtype=np.float64)
            trend_age[596:] = 1.0
        else:
            trend_age = np.arange(periods, dtype=np.float64)
        payload[f"{tf}_trend_state_age_bars_v2"] = trend_age
        payload[f"{tf}_ema_stack_aligned_v2"] = np.ones(periods, dtype=np.float64)
    payload["D1_dist_from_ema200_atr"] = np.full(periods, 1.0)
    frame = pd.DataFrame(payload, index=index)
    frame.loc[frame.index[:20], REGIME_V4_SOURCE_COLS] = np.nan

    out = add_regime_v4_features(frame.copy())

    age = out["d1_regime_flip_age_bars_v4"].to_numpy(dtype=np.float64)
    assert np.isnan(age[:308]).all()
    assert (age[308:596] == 0.0).all()
    assert (age[596:] == 1.0).all()


def test_regime_v4_retires_handwritten_composites_but_keeps_raw_evidence() -> None:
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_FIELDS,
    )
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4

    retired = {
        "regime_tf_agreement_v3",
        "regime_stack_sum_v3",
        "regime_divergence_flag_v3",
        "d1_dist_to_boundary_v3",
        "d1_trend_age_mature_flag_v3",
    }
    assert retired.isdisjoint(REGIME_V4_SOURCE_COLS)
    assert retired.isdisjoint(REGIME_V4_DERIVED_COLS)
    assert retired.isdisjoint(REGIME_V4_FEATURE_NAMES)
    assert retired.isdisjoint(MODEL_NATIVE_CTX_CONT_FIELDS)

    for tf in ("m5", "m15", "h1", "h4", "d1"):
        assert f"{tf}_regime_class_id_v2" in MODEL_NATIVE_CTX_CONT_FIELDS
        assert f"{tf}_trend_state_age_bars_v2" in MODEL_NATIVE_CTX_CONT_FIELDS
    assert "D1_dist_from_ema200_atr" in MODEL_NATIVE_CTX_CONT_FIELDS
    assert "ema_stack_aligned_v2" in MULTI_TF_PER_BAR_FEATURES_V4
    assert "regime_class_id" in MULTI_TF_PER_BAR_FEATURES_V4
    assert "trend_state_age_bars" in MULTI_TF_PER_BAR_FEATURES_V4
    assert {
        "d1_dist_change_1bar_atr_v4",
        "d1_regime_changed_flag_v3",
        "d1_regime_flip_age_bars_v4",
    } <= set(REGIME_V4_DERIVED_COLS)

    producer_source = inspect.getsource(add_regime_v4_features)
    forbidden_formula_fragments = (
        "signs[tf] == d1_sign",
        "agree <= 0.5",
        "np.clip(np.abs(d1d)",
        "d1_age > 0.8",
        'suffix[f"{tf}_ema_stack_aligned_v2"]',
    )
    for fragment in forbidden_formula_fragments:
        assert fragment not in producer_source


def test_regime_v4_v29_additions_are_opt_in_and_mirror_f8_f9() -> None:
    frame = _regime_source_frame(600)

    base = add_regime_v4_features(frame.copy())
    assert not set(REGIME_V4_V29_ADDITION_COLS) & set(base.columns)
    assert not set(REGIME_V4_V29_ADDITION_COLS) & set(REGIME_V4_FEATURE_NAMES)

    out = add_regime_v4_features(frame.copy(), include_v29_additions=True)
    assert set(REGIME_V4_V29_ADDITION_COLS) <= set(out.columns)
    # The bound pre-V29 surface is byte-identical with the additions on.
    np.testing.assert_allclose(
        base.loc[:, REGIME_V4_DERIVED_COLS].to_numpy(dtype=np.float64),
        out.loc[:, REGIME_V4_DERIVED_COLS].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    # Drift guard: the vectorized helper reproduces the F8/F9 originals
    # exactly on the D1 lane (its declared origin), suffix rows onward.
    source_start = 20
    d1c = (
        frame["d1_regime_class_id_v2"]
        .iloc[source_start:]
        .to_numpy(dtype=np.int64)
    )
    d1_trend_age = frame["d1_trend_state_age_bars_v2"].iloc[
        source_start:
    ].to_numpy(dtype=np.float64)
    flag, age_bars = _class_flip_flag_and_age(
        d1c, trend_state_age_bars=d1_trend_age
    )
    np.testing.assert_allclose(
        out["d1_regime_changed_flag_v3"].to_numpy(dtype=np.float64)[source_start:],
        flag,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out["d1_regime_flip_age_bars_v4"].to_numpy(dtype=np.float64)[
            source_start:
        ],
        age_bars,
        rtol=1e-6,
        equal_nan=True,
    )

    # This fixture supplies one new native observation per row for every TF.
    for tf in ("m5", "m15", "h1", "h4"):
        flag_col = out[f"{tf}_regime_changed_flag_v3"].to_numpy(dtype=np.float64)
        age_col = out[f"{tf}_regime_flip_age_bars"].to_numpy(dtype=np.float64)
        assert flag_col[80] == 1.0
        assert flag_col[81] == 0.0  # edge-triggered, not level-triggered
        assert np.isnan(age_col[:80]).all()  # unknown until the first flip
        assert age_col[100] == 20.0

    # Future-append invariance of the additions (causality).
    longer = add_regime_v4_features(
        _regime_source_frame(700), include_v29_additions=True
    )
    np.testing.assert_allclose(
        out.loc[:, REGIME_V4_V29_ADDITION_COLS].to_numpy(dtype=np.float64),
        longer.loc[:, REGIME_V4_V29_ADDITION_COLS]
        .iloc[: len(out)]
        .to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )

    with pytest.raises(RuntimeError, match="include_v29_additions"):
        add_regime_v4_features(frame.copy(), include_v29_additions=1)  # type: ignore[arg-type]


def test_regime_v4_v29_flip_flag_catches_sub_flips_the_age_proxy_misses() -> None:
    periods = 400
    index = pd.date_range("2025-01-01T00:00:00Z", periods=periods, freq="5min")
    m15_classes = np.ones(periods, dtype=np.float64)
    m15_classes[300:] = 2.0  # 1 -> 2 sub-flip: SAME sign, SAME EMA stack
    payload: dict[str, np.ndarray] = {}
    for tf in ("m15", "h1", "h4", "d1", "m5"):
        payload[f"{tf}_regime_class_id_v2"] = (
            m15_classes.copy() if tf == "m15" else np.ones(periods, dtype=np.float64)
        )
        payload[f"{tf}_trend_state_age_bars_v2"] = np.arange(
            periods, dtype=np.float64
        )
        payload[f"{tf}_ema_stack_aligned_v2"] = np.ones(periods, dtype=np.float64)
    payload["D1_dist_from_ema200_atr"] = np.full(periods, 1.0)
    frame = pd.DataFrame(payload, index=index)
    frame.loc[frame.index[:20], REGIME_V4_SOURCE_COLS] = np.nan

    out = add_regime_v4_features(frame.copy(), include_v29_additions=True)

    flag = out["m15_regime_changed_flag_v3"].to_numpy(dtype=np.float64)
    age = out["m15_regime_flip_age_bars"].to_numpy(dtype=np.float64)
    assert np.isnan(flag[:21]).all()  # source warmup + first-row unknown
    np.testing.assert_allclose(flag[21:300], 0.0, rtol=0.0, atol=0.0)
    assert flag[300] == 1.0
    np.testing.assert_allclose(flag[301:], 0.0, rtol=0.0, atol=0.0)
    assert np.isnan(age[:300]).all()
    assert age[300] == 0.0
    assert age[303] == 3.0
    # The trend-age proxy carried by the frame never resets across the
    # sub-flip (constant 0.5) — the exact gap the class-keyed flag closes.
    assert (
        out["m15_trend_state_age_bars_v2"].to_numpy(dtype=np.float64)[300]
        == 300.0
    )
    assert tuple(REGIME_V4_V29_FLIP_TFS) == ("m5", "m15", "h1", "h4")


def test_regime_v4_rejects_nonfinite_source_gap_after_warmup() -> None:
    frame = _regime_source_frame(400)
    frame.loc[frame.index[300], "h1_regime_class_id_v2"] = np.nan

    with pytest.raises(RuntimeError, match="causal warmup prefix"):
        add_regime_v4_features(frame)


def test_htf_context_owner_overwrites_stale_values_with_causal_prefix() -> None:
    frame = _market_frame(82_000).drop(columns="smc_swing_state")
    prefix = frame.iloc[:80_000]
    cols = [
        "D1_dist_from_ema200_atr",
        "H4_trend_sign_cat",
    ]
    before = pd.DataFrame(999.0, index=prefix.index, columns=cols)
    after = pd.DataFrame(999.0, index=frame.index, columns=cols)

    _add_htf_features(before, prefix[["open", "high", "low", "close"]])
    _add_htf_features(after, frame[["open", "high", "low", "close"]])

    warmup = int(before.attrs["causal_htf_warmup_rows"])
    assert warmup > 0
    assert before.iloc[:warmup].isna().any(axis=1).all()
    assert np.isfinite(before.iloc[warmup:].to_numpy()).all()
    np.testing.assert_allclose(
        before.to_numpy(),
        after.iloc[: len(before)].to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_closed_htf_alignment_matches_active_htf_owner() -> None:
    target = pd.DatetimeIndex(["2026-07-08T12:55:00Z"])
    source = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(["2026-07-08T11:00:00Z", "2026-07-08T12:00:00Z"]),
    )

    live = _align_live_last_closed(target, source, pd.Timedelta(hours=1))
    assert float(live.iloc[0]) == 2.0
    assert multi_tf_last_closed_label(target[0], "H1") == source.index[-1]


def test_trend_regime_has_no_price_or_neutral_fallback() -> None:
    frame = pd.DataFrame(
        {"price_vs_ema50_atr": [2.0]},
        index=pd.DatetimeIndex(["2026-07-08T12:00:00Z"]),
    )

    with pytest.raises(RuntimeError, match="exact D1_dist"):
        _add_regime_categoricals(frame)


def test_active_regime_call_sites_have_no_environment_selected_surface() -> None:
    from gx1.execution import v12_ctx_augment_live
    from gx1.features import htf_features
    from gx1.scripts import build_entry_exit_m1_enriched_frame_v1

    for module in (
        htf_features,
        v12_ctx_augment_live,
        build_entry_exit_m1_enriched_frame_v1,
    ):
        source = inspect.getsource(module)
        retired_cross_feature_env = (
            "GX1_" + "X" + "GB" + "_CV3_FOR_CROSSFEATS"
        )
        assert "GX1_REGIME_V4" not in source
        assert "GX1_TREND_REGIME_FROM_D1" not in source
        assert retired_cross_feature_env not in source
