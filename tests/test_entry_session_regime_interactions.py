import numpy as np
import pytest

from gx1.features.entry_session_regime_interactions_v1 import (
    SESSION_REGIME_INTERACTION_FEATURE_NAMES,
    SESSION_REGIME_INTERACTION_SOURCE_FIELDS,
    build_entry_session_regime_interaction_layer,
    missing_session_regime_interaction_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature


def _matrix(names: list[str], n: int = 6) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    set_col("snap.ret_1", [0.5, 1.0, 3.0, 2.0, 4.0, -4.0])
    set_col("snap.ret_5", [1.0, 2.0, 8.0, 5.0, 8.0, -8.0])
    set_col("snap.ret_20", [2.0, 4.0, 18.0, 8.0, 15.0, -15.0])
    # The two session clocks must sum to a real SESSION_BOUNDARIES length per
    # row (ASIA=540, EU=300, OVERLAP=240, US=360); the layer fails closed
    # otherwise. Rows: ASIA, ASIA, EU, OVERLAP, US, US.
    set_col("ctx_cont.minutes_since_session_open", [0, 15, 120, 180, 240, 355])
    set_col("ctx_cont.minutes_to_next_session_boundary", [540, 525, 180, 60, 120, 5])
    set_col("ctx_cont.session_change_flag", [1, 0, 0, 0, 0, 1])
    set_col("ctx_cont.session_tradable", [0, 1, 1, 1, 1, 1])
    set_col("ctx_cont.is_ASIA", [1, 1, 0, 0, 0, 0])
    set_col("ctx_cont.is_asia_eu_overlap", [0, 1, 0, 0, 0, 0])
    set_col("ctx_cont.is_eu_us_overlap", [0, 0, 0, 1, 0, 0])
    set_col("ctx_cont.is_eu_only", [0, 0, 1, 0, 0, 0])
    set_col("ctx_cont.is_us_only", [0, 0, 0, 0, 1, 1])
    set_col("ctx_cont.spread_bps", [1, 2, 2, 8, 1, 10])
    set_col("ctx_cont.atr_bps", [20, 20, 20, 20, 20, 20])
    set_col("ctx_cont.micro_momentum_3", [0.1, 0.2, 0.6, 0.3, 0.8, -0.8])
    set_col("ctx_cont.micro_momentum_5", [0.1, 0.3, 0.7, 0.4, 0.9, -0.9])
    set_col("ctx_cont.micro_acceleration", [0.0, 0.1, 0.3, -0.2, 0.4, -0.4])
    set_col("ctx_cont.vol_pct_m5_1yr", [0.1, 0.2, 0.6, 0.9, 0.5, 0.95])
    set_col("ctx_cont.vol_pct_h1_1yr", [0.1, 0.3, 0.7, 0.8, 0.6, 0.9])
    set_col("ctx_cont.D1_atr_percentile_252", [0.2, 0.3, 0.7, 0.9, 0.5, 0.9])
    set_col("ctx_cat.vol_regime_id", [0, 1, 2, 3, 1, 3])
    set_col("ctx_cat.spread_bucket", [0, 0, 0, 2, 0, 2])
    set_col("ctx_cat.atr_bucket", [1, 1, 2, 4, 2, 4])
    set_col("ctx_cat.H4_trend_sign_cat", [0, 1, 2, 0, 2, 0])
    set_col("ctx_cont.atr_ratio_m5_m15", [0.5, 0.7, 1.2, 2.0, 0.8, 2.5])
    set_col("ctx_cont.atr_ratio_m15_d1", [0.5, 0.7, 1.2, 2.0, 0.8, 2.5])
    set_col("ctx_cont.atr_ratio_h1_d1", [0.5, 0.7, 1.2, 2.0, 0.8, 2.5])
    set_col("ctx_cont._v1h4_ema_diff", [-1.2, -0.2, 1.6, -0.8, 1.8, -1.8])
    set_col("ctx_cont._v1h4_slope5", [-0.6, 0.1, 1.0, -0.7, 1.2, -1.2])
    set_col("ctx_cont.d1_ema_slope_20_canon_v2", [-0.8, -0.1, 1.0, 0.8, 0.9, -1.0])
    set_col("ctx_cont.regime_tf_agreement_v3", [0.2, 0.4, 0.9, 0.2, 0.8, 0.1])
    set_col("ctx_cont.regime_stack_sum_v3", [-2, -1, 2, 1, 2, -2])
    set_col("ctx_cont.regime_divergence_flag_v3", [1, 0, 0, 1, 0, 1])
    set_col("ctx_cont.d1_dist_to_boundary_v3", [0.1, 0.4, 0.9, 0.1, 0.8, 0.05])
    set_col("ctx_cont.d1_dist_roc_288_v3", [-1.0, -0.2, 1.0, -0.8, 0.4, 1.2])
    set_col("ctx_cont.d1_regime_changed_flag_v3", [1, 0, 0, 1, 0, 1])
    set_col("ctx_cont.bars_since_d1_regime_change_v3", [0.0, 0.2, 0.8, 0.1, 0.9, 0.0])
    set_col("ctx_cont.d1_trend_age_mature_flag_v3", [0, 0, 0, 1, 0, 1])
    set_col("ctx_cont.m5_regime_class_id_v2", [1, 1, 2, 1, 2, 3])
    set_col("ctx_cont.m15_regime_class_id_v2", [1, 2, 2, 1, 2, 3])
    set_col("ctx_cont.h1_regime_class_id_v2", [2, 2, 2, 3, 2, 1])
    set_col("ctx_cont.h4_regime_class_id_v2", [3, 2, 2, 1, 2, 4])
    set_col("ctx_cont.d1_regime_class_id_v2", [3, 2, 2, 3, 2, 4])
    set_col("ctx_cont.h4_trend_age_bars_norm_v2", [0.3, 0.2, 0.5, 0.9, 0.7, 0.9])
    set_col("ctx_cont.d1_trend_age_bars_norm_v2", [0.4, 0.3, 0.6, 0.9, 0.8, 0.95])
    set_col("chart.foundation_structure_up_minus_down", [-1, -0.5, 1.2, 0.7, 1.5, -1.0])
    set_col("chart.foundation_bos_recent_balance", [-0.2, 0.1, 0.8, 0.2, 0.9, -0.8])
    set_col("chart.foundation_choch_recent_tau24", [0.5, 0.1, 0.0, 0.8, 0.0, 1.0])
    set_col("chart.foundation_sweep_reclaim_balance_proxy", [4, 2, 0, -4, 0, 5])
    set_col("chart.foundation_compression_release_trigger", [0, 1, 4, 5, 3, 5])
    set_col("chart.foundation_impulse_direction", [-0.5, -0.2, 1.5, 0.8, 1.8, -1.5])
    set_col("chart.foundation_pullback_depth_norm", [0.2, 0.4, 0.3, 0.8, 0.2, 0.9])
    return x


def test_session_regime_interaction_layer_is_finite_and_causal_shape() -> None:
    names = list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)
    out, out_names = build_entry_session_regime_interaction_layer(_matrix(names), names)
    idx = {name: i for i, name in enumerate(out_names)}

    assert tuple(out_names) == SESSION_REGIME_INTERACTION_FEATURE_NAMES
    assert len(SESSION_REGIME_INTERACTION_FEATURE_NAMES) == 68
    assert out.shape == (6, len(SESSION_REGIME_INTERACTION_FEATURE_NAMES))
    assert np.isfinite(out).all()
    assert out[0, idx["session_regime.session_opening_risk"]] == 1.0
    assert out[3, idx["session_regime.spread_cost_pressure"]] > out[2, idx["session_regime.spread_cost_pressure"]]
    assert out[3, idx["session_regime.spread_bucket_high_pressure"]] > out[2, idx["session_regime.spread_bucket_high_pressure"]]
    assert out[3, idx["session_regime.atr_bucket_high_pressure"]] > out[2, idx["session_regime.atr_bucket_high_pressure"]]
    assert out[3, idx["session_regime.spread_atr_bucket_joint_pressure"]] > out[2, idx["session_regime.spread_atr_bucket_joint_pressure"]]
    assert out[3, idx["session_regime.vol_regime_extreme_tail_pressure"]] > out[2, idx["session_regime.vol_regime_extreme_tail_pressure"]]
    assert out[2, idx["session_regime.regime_persistence_score"]] > out[3, idx["session_regime.regime_persistence_score"]]
    assert out[3, idx["session_regime.mtf_regime_divergence_pressure"]] > out[2, idx["session_regime.mtf_regime_divergence_pressure"]]
    assert out[2, idx["session_regime.regime_stack_bull_trend_pressure"]] > 0.0
    assert out[0, idx["session_regime.regime_stack_bear_trend_pressure"]] > 0.0
    assert out[2, idx["session_regime.h4_d1_regime_sign_agreement"]] == 1.0
    assert out[3, idx["session_regime.h4_d1_regime_sign_mismatch"]] == 1.0
    assert out[2, idx["session_regime.h4_d1_stack_bull_pressure"]] > 0.0
    assert out[5, idx["session_regime.h4_d1_stack_bear_pressure"]] > 0.0
    assert out[5, idx["session_regime.h4_d1_d1_roc_reversal_pressure"]] > out[2, idx["session_regime.h4_d1_d1_roc_reversal_pressure"]]
    assert out[3, idx["session_regime.regime_transition_tail_risk"]] > out[2, idx["session_regime.regime_transition_tail_risk"]]
    assert out[:, idx["session_regime.asia_mid_session_low_cost_permission"]].std() > 0.0
    assert out[:, idx["session_regime.asia_mid_session_low_cost_permission"]].max() > 0.0
    assert out[3, idx["session_regime.eu_us_overlap_divergence_risk"]] > 0.0
    assert out[4, idx["session_regime.us_momentum_followthrough_pressure"]] > 0.0
    assert out[3, idx["session_regime.overlap_momentum_vol_spread_risk"]] > 0.0
    assert out[2, idx["session_regime.eu_structure_regime_continuation_bias"]] > 0.0
    assert out[2, idx["session_regime.eu_h4_d1_structure_continuation_long"]] > 0.0
    assert out[3, idx["session_regime.overlap_h4_d1_liquidity_tail_risk"]] > 0.0
    assert out[4, idx["session_regime.session_momentum_structure_alignment_score"]] > 0.0
    # session_age_progress divides by the OWNING session's length, not 360:
    # EU row 120/300, OVERLAP row 180/240, US rows 240/360 and 355/360.
    np.testing.assert_allclose(
        out[:, idx["session_regime.session_age_progress_norm"]],
        np.asarray(
            [0.0, 15.0 / 540.0, 120.0 / 300.0, 180.0 / 240.0, 240.0 / 360.0, 355.0 / 360.0],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=0.0,
    )


def test_session_regime_layer_rejects_invalid_session_length() -> None:
    names = list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)
    x = _matrix(names)
    idx = {name: i for i, name in enumerate(names)}
    x[2, idx["ctx_cont.minutes_to_next_session_boundary"]] = 45.0  # sum 165

    with pytest.raises(
        RuntimeError,
        match="ENTRY_SESSION_REGIME_SESSION_LENGTH_INVALID",
    ):
        build_entry_session_regime_interaction_layer(x, names)


def test_open_boundary_risks_ignore_session_change_flag() -> None:
    # session_change was removed from both risk terms: at a session open the
    # clock already forces exp(0)=1, and at every other minute the flag must
    # not overwrite the true boundary distance. Toggling the flag column must
    # therefore leave the whole layer bit-identical.
    names = list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)
    base = _matrix(names)
    toggled = base.copy()
    idx = {name: i for i, name in enumerate(names)}
    toggled[:, idx["ctx_cont.session_change_flag"]] = (
        1.0 - toggled[:, idx["ctx_cont.session_change_flag"]]
    )

    base_out, _ = build_entry_session_regime_interaction_layer(base, names)
    toggled_out, _ = build_entry_session_regime_interaction_layer(toggled, names)

    np.testing.assert_allclose(toggled_out, base_out, rtol=0.0, atol=0.0)


def test_asia_mid_session_permission_is_ungated_by_session_tradable() -> None:
    # session_tradable is identically 0 on real ASIA rows, so the asia feature
    # consumes the ungated mid-session core: toggling the tradable flag must
    # not move it, while the gated stability output must move.
    names = list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)
    base = _matrix(names)
    toggled = base.copy()
    idx = {name: i for i, name in enumerate(names)}
    toggled[:, idx["ctx_cont.session_tradable"]] = (
        1.0 - toggled[:, idx["ctx_cont.session_tradable"]]
    )

    base_out, out_names = build_entry_session_regime_interaction_layer(base, names)
    toggled_out, _ = build_entry_session_regime_interaction_layer(toggled, names)
    out_idx = {name: i for i, name in enumerate(out_names)}

    np.testing.assert_allclose(
        toggled_out[:, out_idx["session_regime.asia_mid_session_low_cost_permission"]],
        base_out[:, out_idx["session_regime.asia_mid_session_low_cost_permission"]],
        rtol=0.0,
        atol=0.0,
    )
    # Row 1 is mid-session enough for a nonzero core, so the gate must bite.
    assert (
        toggled_out[1, out_idx["session_regime.session_mid_age_stability"]]
        != base_out[1, out_idx["session_regime.session_mid_age_stability"]]
    )


def test_session_flag_union_covers_every_minute_of_day() -> None:
    # active_session was removed from the layer because clip01(asia+eu+us) is
    # identically 1.  V30 package 3 (2026-08-13) unified the session clock: the
    # second hour-set definition this test used to import from
    # augment_forward_outcome_v2 is retired, and the four overlap flags are
    # derived from the ONE SESSION_BOUNDARIES partition by
    # session_detector.session_overlap_flags.  The proof is re-derived on that
    # owner here: exercise all 1440 minutes of the day and require the exact
    # layer expressions asia / eu / us to cover every one of them.
    import pandas as pd

    from gx1.time.session_detector import get_session, session_overlap_flags

    for minute in range(24 * 60):
        ts = pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(minutes=minute)
        flags = session_overlap_flags(ts)
        asia = float(get_session(ts) == "ASIA")
        # Exactly the layer's expressions (entry_session_regime_interactions_v1
        # and entry_foundation_structure_v1 both build eu/us this way).
        eu = min(
            1.0,
            flags["is_eu_only"]
            + flags["is_asia_eu_overlap"]
            + flags["is_eu_us_overlap"],
        )
        us = min(1.0, flags["is_us_only"] + flags["is_eu_us_overlap"])
        assert min(1.0, asia + eu + us) == 1.0, (
            f"minute {minute} not covered by asia/eu/us flag union"
        )


def test_session_overlap_flags_are_one_partition_and_all_live() -> None:
    # The four flags must stay mutually exclusive, all-zero on ASIA, and each
    # one live.  Rates are exact minute-of-day fractions of the partition, so
    # this is a proof, not a sample (rule 2f: no sampling error to clear).
    import pandas as pd

    from gx1.time.session_detector import (
        ASIA_EU_HANDOVER_MINUTES,
        SESSION_OVERLAP_FLAG_NAMES,
        get_session,
        session_overlap_flags,
    )

    counts = {name: 0 for name in SESSION_OVERLAP_FLAG_NAMES}
    for minute in range(24 * 60):
        ts = pd.Timestamp("2026-01-05", tz="UTC") + pd.Timedelta(minutes=minute)
        flags = session_overlap_flags(ts)
        assert tuple(flags) == SESSION_OVERLAP_FLAG_NAMES
        assert sum(flags.values()) <= 1.0, f"flags overlap at minute {minute}"
        if get_session(ts) == "ASIA":
            assert sum(flags.values()) == 0.0
        else:
            assert sum(flags.values()) == 1.0
        for name, value in flags.items():
            counts[name] += int(value)

    # EU 07:00-12:00 splits into the 120-minute Asia/EU handover window and the
    # 180-minute EU-only remainder; OVERLAP is 12:00-16:00 and US is 16:00-22:00.
    assert counts["is_asia_eu_overlap"] == ASIA_EU_HANDOVER_MINUTES == 120
    assert counts["is_eu_only"] == 180
    assert counts["is_eu_us_overlap"] == 240
    assert counts["is_us_only"] == 360
    for name, count in counts.items():
        assert count / (24 * 60) > 0.01, f"{name} below the 1% activity floor"


def test_session_regime_interaction_layer_rejects_unprefixed_aliases() -> None:
    names = list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)
    alias_names = [
        "spread_bps" if name == "ctx_cont.spread_bps" else name
        for name in names
    ]

    assert missing_session_regime_interaction_source_fields(alias_names) == [
        "ctx_cont.spread_bps"
    ]
    with pytest.raises(
        RuntimeError,
        match="ENTRY_SESSION_REGIME_INTERACTION_MISSING_SOURCE_FIELDS",
    ):
        build_entry_session_regime_interaction_layer(
            np.zeros((2, len(alias_names)), dtype=np.float32),
            alias_names,
        )


def test_session_regime_source_contract_and_routing_are_explicit() -> None:
    assert len(SESSION_REGIME_INTERACTION_SOURCE_FIELDS) == len(set(SESSION_REGIME_INTERACTION_SOURCE_FIELDS))
    assert len(SESSION_REGIME_INTERACTION_SOURCE_FIELDS) == 52
    assert "snap.ret_1" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "ctx_cont.spread_bps" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "ctx_cont.micro_acceleration" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "ctx_cat.vol_regime_id" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "ctx_cat.spread_bucket" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "ctx_cat.atr_bucket" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "ctx_cat.H4_trend_sign_cat" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "ctx_cont._v1h4_ema_diff" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "ctx_cont.d1_ema_slope_20_canon_v2" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "ctx_cont.regime_tf_agreement_v3" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "ctx_cont.d1_dist_roc_288_v3" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert "chart.foundation_sweep_reclaim_balance_proxy" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
    assert missing_session_regime_interaction_source_fields(SESSION_REGIME_INTERACTION_SOURCE_FIELDS) == []
    assert missing_session_regime_interaction_source_fields(
        [
            "spread_bps" if name == "ctx_cont.spread_bps" else name
            for name in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
        ]
    ) == ["ctx_cont.spread_bps"]
    missing = missing_session_regime_interaction_source_fields(
        name for name in SESSION_REGIME_INTERACTION_SOURCE_FIELDS if name != "ctx_cont.spread_bps"
    )
    assert missing == ["ctx_cont.spread_bps"]
    missing_names = [name for name in SESSION_REGIME_INTERACTION_SOURCE_FIELDS if name != "ctx_cont.spread_bps"]
    with pytest.raises(RuntimeError, match="ENTRY_SESSION_REGIME_INTERACTION_MISSING_SOURCE_FIELDS"):
        build_entry_session_regime_interaction_layer(
            np.zeros((2, len(missing_names)), dtype=np.float32),
            missing_names,
        )
    for feature in SESSION_REGIME_INTERACTION_FEATURE_NAMES:
        assert classify_entry_specialist_feature(feature) == "session_regime_encoder"


def test_session_regime_interaction_layer_rejects_nonfinite_inputs() -> None:
    names = list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)
    x = _matrix(names)
    idx = {name: i for i, name in enumerate(names)}
    x[1, idx["ctx_cont.spread_bps"]] = np.nan
    x[2, idx["ctx_cont.atr_bps"]] = 0.0
    x[3, idx["snap.ret_5"]] = np.inf
    x[4, idx["ctx_cat.vol_regime_id"]] = -np.inf

    with pytest.raises(
        RuntimeError,
        match="ENTRY_SESSION_REGIME_INTERACTION_SOURCE_NONFINITE",
    ):
        build_entry_session_regime_interaction_layer(x, names)


def test_session_regime_interaction_layer_does_not_read_future_rows() -> None:
    names = list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)
    base = _matrix(names)
    idx = {name: i for i, name in enumerate(names)}
    changed_future = base.copy()
    changed_future[-1, :] = base[-1, :] * -7.0 + 3.0
    # The perturbed future row must still carry a valid session clock (the
    # layer fails closed on a sum outside the named session lengths); use a
    # DIFFERENT valid pair than base so the row still changes.
    changed_future[-1, idx["ctx_cont.minutes_since_session_open"]] = 30.0
    changed_future[-1, idx["ctx_cont.minutes_to_next_session_boundary"]] = 210.0

    base_out, _ = build_entry_session_regime_interaction_layer(base, names)
    changed_out, _ = build_entry_session_regime_interaction_layer(changed_future, names)

    np.testing.assert_allclose(changed_out[:-1], base_out[:-1], rtol=0.0, atol=0.0)
