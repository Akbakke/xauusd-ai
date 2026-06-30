import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from gx1.features.entry_session_regime_interactions_v1 import (
    SESSION_REGIME_INTERACTION_FEATURE_NAMES,
    SESSION_REGIME_INTERACTION_SOURCE_FIELDS,
    build_entry_session_regime_interaction_layer,
    missing_session_regime_interaction_source_fields,
)
from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.scripts.materialize_entry_session_regime_interaction_manifest_v1 import run


def _matrix(names: list[str], n: int = 6) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    set_col("ret_1", [0.5, 1.0, 3.0, 2.0, 4.0, -4.0])
    set_col("ret_5", [1.0, 2.0, 8.0, 5.0, 8.0, -8.0])
    set_col("ret_20", [2.0, 4.0, 18.0, 8.0, 15.0, -15.0])
    set_col("ctx_cont.minutes_since_session_open", [0, 15, 120, 180, 240, 300])
    set_col("ctx_cont.minutes_to_next_session_boundary", [180, 90, 45, 10, 180, 5])
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


def _emitted_alias(name: str) -> str:
    for prefix in ("ctx_cont.", "ctx_cat."):
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


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


def test_session_regime_interaction_layer_accepts_emitted_context_aliases() -> None:
    names = list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)
    alias_names = [_emitted_alias(name) for name in names]
    canonical_out, canonical_names = build_entry_session_regime_interaction_layer(_matrix(names), names)
    alias_out, alias_feature_names = build_entry_session_regime_interaction_layer(_matrix(names), alias_names)

    assert tuple(alias_feature_names) == tuple(canonical_names)
    np.testing.assert_allclose(alias_out, canonical_out, rtol=0.0, atol=0.0)


def test_session_regime_source_contract_and_routing_are_explicit() -> None:
    assert len(SESSION_REGIME_INTERACTION_SOURCE_FIELDS) == len(set(SESSION_REGIME_INTERACTION_SOURCE_FIELDS))
    assert len(SESSION_REGIME_INTERACTION_SOURCE_FIELDS) == 52
    assert "ret_1" in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
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
            "spread_bps" if name == "ctx_cont.spread_bps" else
            "vol_regime_id" if name == "ctx_cat.vol_regime_id" else
            "spread_bucket" if name == "ctx_cat.spread_bucket" else
            name
            for name in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
        ]
    ) == []
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


def test_session_regime_interaction_layer_sanitizes_nonfinite_inputs() -> None:
    names = list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)
    x = _matrix(names)
    idx = {name: i for i, name in enumerate(names)}
    x[1, idx["ctx_cont.spread_bps"]] = np.nan
    x[2, idx["ctx_cont.atr_bps"]] = 0.0
    x[3, idx["ret_5"]] = np.inf
    x[4, idx["ctx_cat.vol_regime_id"]] = -np.inf

    out, out_names = build_entry_session_regime_interaction_layer(x, names)

    assert tuple(out_names) == SESSION_REGIME_INTERACTION_FEATURE_NAMES
    assert out.shape == (6, 68)
    assert np.isfinite(out).all()


def test_session_regime_interaction_layer_does_not_read_future_rows() -> None:
    names = list(SESSION_REGIME_INTERACTION_SOURCE_FIELDS)
    base = _matrix(names)
    changed_future = base.copy()
    changed_future[-1, :] = base[-1, :] * -7.0 + 3.0

    base_out, _ = build_entry_session_regime_interaction_layer(base, names)
    changed_out, _ = build_entry_session_regime_interaction_layer(changed_future, names)

    np.testing.assert_allclose(changed_out[:-1], base_out[:-1], rtol=0.0, atol=0.0)


def test_session_regime_manifest_script_is_report_only(tmp_path: Path) -> None:
    report = run(
        argparse.Namespace(
            out_dir=str(tmp_path),
            quiet=True,
            fail_on_audit_fail=True,
        )
    )

    assert report["decision"] == "READY_FOR_CHALLENGER_MANIFEST_REVIEW"
    assert report["training_allowed"] is False
    assert report["replay_allowed"] is False
    assert report["iql_allowed"] is False
    latest = tmp_path / "ENTRY_SESSION_REGIME_INTERACTION_MANIFEST_latest.json"
    data = json.loads(latest.read_text(encoding="utf-8"))
    assert data["selected_features"] == list(SESSION_REGIME_INTERACTION_FEATURE_NAMES)
    assert data["specialist"] == "session_regime_encoder"
