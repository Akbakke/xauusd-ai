from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.features.group_a_features import (
    daily_pivot_levels,
    realized_vol_percentile,
    vol_term_structure,
)
from gx1.features.htf_features import (
    HTF_V2_MATRIX_CONTRACT,
    MULTI_TF_SHIFT,
    attach_v2_mtf_per_bar_scalars,
    build_multi_tf_per_bar_features_v2,
    build_multi_tf_per_bar_features_v4,
    compute_per_bar_features_v2,
    get_last_n_at_or_before,
)
from gx1.features.regime_v4_features import (
    REGIME_V4_DERIVED_COLS,
    REGIME_V4_SOURCE_COLS,
    add_regime_v4_features,
)
from gx1.execution.v12_ctx_augment_live import (
    _add_htf_features,
    _add_regime_categoricals,
    _align_last_closed as _align_live_last_closed,
)
from gx1.scripts.add_ctx_cont_columns_to_prebuilt import (
    _align_last_closed as _align_offline_last_closed,
)
from gx1.scripts.augment_forward_outcome_v2 import (
    _build_atr_percentile_array,
    _cache_cutoff_ns,
    _tf_cache_row,
    attach_group_a_dip_struct_ctx_columns,
    augment_candidate,
    build_context,
    compute_smc_swing_dip_interaction,
    trim_causal_context_warmup_prefix,
)


def test_tf_cache_row_is_strict_float32_zero_copy() -> None:
    target = pd.Timestamp("2026-01-02T12:00:00Z")
    width = 111
    ts_values = np.asarray(
        [target.value - pd.Timedelta(minutes=10).value, target.value],
        dtype=np.int64,
    )
    feat_values = np.arange(2 * width, dtype=np.float32).reshape(2, width)
    frame = pd.DataFrame(index=pd.DatetimeIndex(ts_values, tz="UTC"))
    frame.attrs["ts_int64"] = ts_values
    frame.attrs["feats_np"] = feat_values
    frame.attrs["causal_warmup_rows"] = 0
    ctx = type("Context", (), {"multi_tf": {"M5": frame}})()

    row = _tf_cache_row(ctx, "M5", target.value)

    assert row.dtype == np.dtype(np.float32)
    assert np.shares_memory(row, feat_values)
    np.testing.assert_array_equal(row, feat_values[1])


def test_tf_cache_row_rejects_dtype_coercion_instead_of_copying() -> None:
    target = pd.Timestamp("2026-01-02T12:00:00Z")
    ts_values = np.asarray([target.value], dtype=np.int64)
    frame = pd.DataFrame(index=pd.DatetimeIndex(ts_values, tz="UTC"))
    frame.attrs["ts_int64"] = ts_values
    frame.attrs["feats_np"] = np.ones((1, 25), dtype=np.float64)
    frame.attrs["causal_warmup_rows"] = 0
    ctx = type("Context", (), {"multi_tf": {"M5": frame}})()

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
            "volume": np.linspace(100.0, 300.0, periods),
            "smc_swing_state": np.arange(periods) % 5,
        },
        index=index,
    )


def test_smc_swing_dip_interaction_is_static_and_prefix_invariant() -> None:
    state = np.asarray([0, 1, 2, 3], dtype=np.float64)
    dip = np.asarray([1.0, 0.8, 0.6, 0.4], dtype=np.float64)
    prefix = compute_smc_swing_dip_interaction(state, dip)
    extended = compute_smc_swing_dip_interaction(
        np.append(state, 4.0),
        np.append(dip, 0.2),
    )

    np.testing.assert_allclose(prefix, [0.0, 0.2, 0.3, 0.3])
    np.testing.assert_array_equal(prefix, extended[: len(prefix)])


@pytest.mark.parametrize(
    ("state", "dip"),
    [
        ([0.0, np.nan], [0.2, 0.3]),
        ([0.0, 5.0], [0.2, 0.3]),
        ([0.0, 1.5], [0.2, 0.3]),
        ([0.0, 1.0], [0.2, 1.1]),
        ([0.0], [0.2, 0.3]),
    ],
)
def test_smc_swing_dip_interaction_rejects_invalid_sources(
    state: list[float], dip: list[float]
) -> None:
    with pytest.raises(RuntimeError, match="CTX_CAUSALITY"):
        compute_smc_swing_dip_interaction(np.asarray(state), np.asarray(dip))


def test_atr_percentile_is_exactly_causal_when_future_rows_are_appended() -> None:
    frame = _market_frame(500)
    prefix = frame.iloc[:400]
    prefix_pct = _build_atr_percentile_array(
        prefix,
        prefix.index.view("int64"),
        window_days=365,
    )
    full_pct = _build_atr_percentile_array(
        frame,
        frame.index.view("int64"),
        window_days=365,
    )

    np.testing.assert_array_equal(prefix_pct, full_pct[: len(prefix)])
    assert prefix_pct[0] == 0.0
    assert np.unique(prefix_pct).size > 2


def test_closed_bar_cutoff_is_immutable_and_excludes_forming_h1(monkeypatch: pytest.MonkeyPatch) -> None:
    target = pd.Timestamp("2026-07-08T18:00:00Z")
    monkeypatch.setenv("GX1_PERTF_CLOSED_BAR", "0")

    cutoff = pd.Timestamp(_cache_cutoff_ns(target.value, "H1"), tz="UTC")

    assert cutoff == pd.Timestamp("2026-07-08T17:05:00Z")
    assert pd.Timestamp("2026-07-08T18:00:00Z") > cutoff


def test_full_group_a_transform_is_prefix_invariant(tmp_path: Path) -> None:
    frame = _market_frame(18_400)
    prefix = frame.iloc[:18_200]
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

    before = augment_candidate(prefix_ctx, target, include_portfolio=False)
    after = augment_candidate(full_ctx, target, include_portfolio=False)

    assert set(before) == set(after)
    for name in before:
        assert after[name] == pytest.approx(before[name], abs=1e-7), name


def test_group_a_decision_slice_uses_explicit_full_m5_history(tmp_path: Path) -> None:
    frame = _market_frame(18_400)
    multi_tf = build_multi_tf_per_bar_features_v4(frame)
    decision = frame.iloc[-12:].reset_index(names="time")

    augmented = attach_group_a_dip_struct_ctx_columns(
        decision,
        multi_tf=multi_tf,
        context_m5=frame,
    )
    full_ctx = build_context(
        frame[["high", "low", "close"]],
        multi_tf,
        journal_dir=tmp_path / "missing_full_history_journal",
    )
    expected = augment_candidate(
        full_ctx,
        pd.Timestamp(decision.loc[0, "time"]),
        include_portfolio=False,
    )

    assert int(augmented.attrs["causal_context_warmup_rows"]) == 0
    for name in ("dist_to_d1_hi_atr", "atr_ratio_m5_h4", "dip_proximity_d1_v3"):
        assert float(augmented.loc[0, name]) == pytest.approx(expected[name], abs=1e-7)


def test_group_a_full_history_rejects_decision_ohlc_mismatch() -> None:
    frame = _market_frame(18_400)
    multi_tf = build_multi_tf_per_bar_features_v4(frame)
    decision = frame.iloc[-2:].reset_index(names="time")
    decision.loc[0, "close"] += 0.01

    with pytest.raises(RuntimeError, match="decision OHLC differs"):
        attach_group_a_dip_struct_ctx_columns(
            decision,
            multi_tf=multi_tf,
            context_m5=frame,
        )


def test_attach_marks_only_real_warmup_and_shared_trim_removes_it() -> None:
    frame = _market_frame(18_400)
    multi_tf = build_multi_tf_per_bar_features_v4(frame)
    source = frame.reset_index(names="time")

    augmented = attach_group_a_dip_struct_ctx_columns(source, multi_tf=multi_tf)
    required = ["dist_to_R1_atr", "dip_confirmed_h1_v3", "struct_smc_swing_x_dip_v3"]
    trimmed = trim_causal_context_warmup_prefix(augmented, required)

    assert int(augmented.attrs["causal_context_warmup_rows"]) > 0
    assert len(trimmed) < len(augmented)
    assert np.isfinite(trimmed[required].to_numpy(dtype=np.float64)).all()


def test_attach_requires_explicit_mtf_and_exact_smc_source() -> None:
    frame = _market_frame(4_500)
    multi_tf = build_multi_tf_per_bar_features_v4(frame)
    source = frame.reset_index(names="time")

    with pytest.raises(RuntimeError, match="explicit multi_tf"):
        attach_group_a_dip_struct_ctx_columns(source, multi_tf=None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="exact SMC source missing"):
        attach_group_a_dip_struct_ctx_columns(
            source.drop(columns="smc_swing_state"),
            multi_tf=multi_tf,
        )


def test_group_a_public_owner_rejects_missing_evidence() -> None:
    frame = _market_frame(600)
    target = frame.index[-1]

    with pytest.raises(RuntimeError, match="exact TF cache"):
        vol_term_structure({}, target)
    with pytest.raises(RuntimeError, match="current_atr"):
        daily_pivot_levels(frame, target, 0.0)
    with pytest.raises(RuntimeError, match="exact M5 row"):
        realized_vol_percentile(frame, target + pd.Timedelta(minutes=1))


def test_warmup_trim_rejects_nonfinite_gap_after_complete_rows() -> None:
    frame = pd.DataFrame({"feature": [np.nan, 1.0, np.nan, 2.0]})

    with pytest.raises(RuntimeError, match="not a contiguous warmup prefix"):
        trim_causal_context_warmup_prefix(frame, ["feature"])


def test_htf_v2_requires_exact_observed_volume() -> None:
    frame = _market_frame(100).drop(columns=["volume", "smc_swing_state"])

    with pytest.raises(RuntimeError, match="volume"):
        compute_per_bar_features_v2(frame)

    frame["volume"] = 0.0
    with pytest.raises(RuntimeError, match="volume"):
        compute_per_bar_features_v2(frame)


def test_htf_v2_warmup_is_explicit_and_future_append_is_prefix_invariant() -> None:
    frame = _market_frame(160).drop(columns="smc_swing_state")
    prefix = frame.iloc[:120]

    before = compute_per_bar_features_v2(prefix)
    after = compute_per_bar_features_v2(frame)

    assert before.iloc[0].isna().any()
    warmup = int(np.argmax(np.isfinite(before.to_numpy()).all(axis=1)))
    assert warmup > 0
    assert np.isfinite(before.iloc[warmup:].to_numpy()).all()
    np.testing.assert_allclose(
        before.to_numpy(),
        after.iloc[: len(before)].to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_htf_window_refuses_padding_and_returns_only_exact_finite_history() -> None:
    frame = _market_frame(180).drop(columns="smc_swing_state")
    feats = build_multi_tf_per_bar_features_v2(frame)["M5"]
    assert feats.attrs["htf_feature_contract"] == HTF_V2_MATRIX_CONTRACT

    target = frame.index[-1] + pd.Timedelta(minutes=5)
    window = get_last_n_at_or_before(feats, target, n=32, tf_shift=MULTI_TF_SHIFT["M5"])
    assert window.shape == (32, feats.shape[1])
    assert np.isfinite(window).all()

    with pytest.raises(RuntimeError, match="WARMUP|HISTORY"):
        get_last_n_at_or_before(
            feats,
            frame.index[30] + pd.Timedelta(minutes=5),
            n=10,
            tf_shift=MULTI_TF_SHIFT["M5"],
        )
    legacy = feats.copy()
    legacy.attrs.clear()
    with pytest.raises(RuntimeError, match="CONTRACT"):
        get_last_n_at_or_before(legacy, target, n=1, tf_shift=MULTI_TF_SHIFT["M5"])


def test_htf_projection_uses_m5_decision_close_and_keeps_warmup_unavailable() -> None:
    frame = _market_frame(400).drop(columns="smc_swing_state")
    projected = attach_v2_mtf_per_bar_scalars(
        frame,
        frame.index.asi8,
        (("atr_bps_14", "atr_bps_14"),),
        tfs=("m5",),
    )["m5_atr_bps_14_v2"]
    direct = build_multi_tf_per_bar_features_v2(frame)["M5"]["atr_bps_14"].to_numpy()

    np.testing.assert_allclose(projected, direct, rtol=0.0, atol=0.0, equal_nan=True)
    assert np.isnan(projected[:13]).all()
    assert np.isfinite(projected[13:]).all()


def _regime_source_frame(periods: int) -> pd.DataFrame:
    index = pd.date_range("2025-01-01T00:00:00Z", periods=periods, freq="5min")
    block = (np.arange(periods) // 80) % 2
    classes = np.where(block == 0, 1.0, 3.0)
    stacks = np.where(block == 0, 1.0, -1.0)
    payload: dict[str, np.ndarray] = {}
    for tf in ("m15", "h1", "h4", "d1", "m5"):
        payload[f"{tf}_regime_class_id_v2"] = classes.copy()
        payload[f"{tf}_trend_age_bars_norm_v2"] = (np.arange(periods) % 501) / 500.0
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
    assert warmup >= 20 + 288
    assert before.loc[:, REGIME_V4_DERIVED_COLS].iloc[:warmup].isna().any(axis=1).all()
    assert np.isfinite(before.loc[:, REGIME_V4_DERIVED_COLS].iloc[warmup:].to_numpy()).all()
    np.testing.assert_allclose(
        before.loc[:, REGIME_V4_DERIVED_COLS].to_numpy(),
        after.loc[:, REGIME_V4_DERIVED_COLS].iloc[: len(before)].to_numpy(),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


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
        "D1_atr_percentile_252",
        "H1_range_compression_ratio",
        "M15_range_compression_ratio",
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


def test_closed_htf_alignment_observes_m5_decision_close_identically() -> None:
    target = pd.DatetimeIndex(["2026-07-08T12:55:00Z"])
    source = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(["2026-07-08T11:00:00Z", "2026-07-08T12:00:00Z"]),
    )

    live = _align_live_last_closed(target, source, pd.Timedelta(hours=1))
    offline = _align_offline_last_closed(target, source, pd.Timedelta(hours=1))

    assert float(live.iloc[0]) == 2.0
    assert float(offline.iloc[0]) == 2.0


def test_trend_regime_has_no_price_or_neutral_fallback() -> None:
    frame = pd.DataFrame(
        {"price_vs_ema50_atr": [2.0]},
        index=pd.DatetimeIndex(["2026-07-08T12:00:00Z"]),
    )

    with pytest.raises(RuntimeError, match="exact D1_dist"):
        _add_regime_categoricals(frame)


def test_active_regime_call_sites_have_no_environment_selected_surface() -> None:
    from gx1.execution import v12_ctx_augment_live, v12_state_from_prebuilt
    from gx1.scripts import add_ctx_cont_columns_to_prebuilt

    for module in (
        v12_ctx_augment_live,
        v12_state_from_prebuilt,
        add_ctx_cont_columns_to_prebuilt,
    ):
        source = inspect.getsource(module)
        retired_cross_feature_env = (
            "GX1_" + "X" + "GB" + "_CV3_FOR_CROSSFEATS"
        )
        assert "GX1_REGIME_V4" not in source
        assert "GX1_TREND_REGIME_FROM_D1" not in source
        assert retired_cross_feature_env not in source
