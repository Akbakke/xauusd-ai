import numpy as np
import pytest

from gx1.features.entry_specialist_feature_groups_v1 import classify_entry_specialist_feature
from gx1.features.entry_structure_swing_derivations_v1 import (
    STRUCTURE_SWING_DERIVATION_FEATURE_NAMES,
    STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS,
    build_entry_structure_swing_derivation_layer,
    missing_structure_swing_derivation_source_fields,
)


EXPECTED_STRUCTURE_SWING_DERIVATION_FEATURE_NAMES = (
    "chart.structure_swing_hh_hl_consistency_up",
    "chart.structure_swing_lh_ll_consistency_down",
    "chart.structure_swing_hh_hl_lh_ll_conflict",
    "chart.structure_swing_swing_leg_quality_up",
    "chart.structure_swing_swing_leg_quality_down",
    "chart.structure_swing_swing_leg_quality_balance",
    "chart.structure_swing_bos_choch_recency_alignment_up",
    "chart.structure_swing_bos_choch_recency_alignment_down",
    "chart.structure_swing_bos_choch_recency_conflict",
    "chart.structure_swing_bos_followthrough_up_quality",
    "chart.structure_swing_bos_followthrough_down_quality",
    "chart.structure_swing_bos_followthrough_balance",
    "chart.structure_swing_break_confirmation_up",
    "chart.structure_swing_break_confirmation_down",
    "chart.structure_swing_break_confirmation_balance",
    "chart.structure_swing_choch_failure_up_risk",
    "chart.structure_swing_choch_failure_down_risk",
    "chart.structure_swing_pullback_depth_quality",
    "chart.structure_swing_pullback_depth_phase_alignment_up",
    "chart.structure_swing_pullback_depth_phase_alignment_down",
    "chart.structure_swing_pullback_phase_continuation_up",
    "chart.structure_swing_pullback_phase_continuation_down",
    "chart.structure_swing_market_structure_regime_state",
    "chart.structure_swing_market_structure_regime_confidence",
    "chart.structure_swing_structure_compression_pressure",
    "chart.structure_swing_swing_compression_setup",
    "chart.structure_swing_mtf_structure_agreement",
    "chart.structure_swing_mtf_structure_divergence",
)


def _matrix(names: list[str], n: int = 8) -> np.ndarray:
    x = np.zeros((n, len(names)), dtype=np.float32)
    idx = {name: i for i, name in enumerate(names)}

    def set_col(name: str, values) -> None:
        x[:, idx[name]] = np.asarray(values, dtype=np.float32)

    set_col("chart.foundation_hh_state", [0.1, 0.8, 0.7, 0.4, 0.8, 0.1, 0.1, 0.1])
    set_col("chart.foundation_hl_state", [0.1, 0.7, 0.8, 0.4, 0.7, 0.1, 0.1, 0.1])
    set_col("chart.foundation_lh_state", [0.1, 0.1, 0.1, 0.5, 0.1, 0.8, 0.7, 0.8])
    set_col("chart.foundation_ll_state", [0.1, 0.1, 0.1, 0.5, 0.1, 0.7, 0.8, 0.7])
    set_col("chart.foundation_structure_up_minus_down", [0.0, 1.5, 1.4, 0.0, 1.3, -1.5, -1.4, -1.3])
    set_col("chart.foundation_bos_up_age_bars", [96, 0, 1, 20, 8, 96, 96, 96])
    set_col("chart.foundation_bos_down_age_bars", [96, 96, 96, 20, 96, 0, 1, 8])
    set_col("chart.foundation_bos_up_recent_tau24", [0.0, 1.0, 0.9, 0.1, 0.2, 0.0, 0.0, 0.0])
    set_col("chart.foundation_bos_down_recent_tau24", [0.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.9, 0.2])
    set_col("chart.foundation_bos_recent_balance", [0.0, 1.0, 0.9, 0.0, 0.2, -1.0, -0.9, -0.2])
    set_col("chart.foundation_choch_age_bars", [96, 96, 96, 12, 0, 96, 96, 0])
    set_col("chart.foundation_choch_recent_tau24", [0.0, 0.0, 0.0, 0.4, 1.0, 0.0, 0.0, 1.0])
    set_col("chart.foundation_impulse_direction", [0.0, 1.3, 1.2, 0.0, 1.0, -1.3, -1.2, -1.0])
    set_col("chart.foundation_impulse_age_proxy", [0.0, 0.2, 0.5, 0.2, 0.4, 0.2, 0.5, 0.4])
    set_col("chart.foundation_pullback_phase_up", [0.0, 0.5, 0.7, 0.5, 0.4, 0.0, 0.0, 0.0])
    set_col("chart.foundation_pullback_phase_down", [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.7, 0.4])
    set_col("chart.foundation_pullback_depth_norm", [0.2, 0.4, 0.5, 0.5, 0.4, 0.4, 0.5, 0.4])
    set_col("chart.foundation_impulse_pullback_alignment", [0.0, 0.8, 0.9, 0.0, 0.6, -0.8, -0.9, -0.6])
    set_col("chart.foundation_compression_state", [0.1, 0.2, 0.3, 0.9, 0.2, 0.2, 0.3, 0.2])
    set_col("chart.foundation_expansion_state", [0.0, 0.4, 0.5, 0.1, 0.2, 0.4, 0.5, 0.2])
    set_col("chart.foundation_compression_release_trigger", [0.0, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3, 0.1])
    set_col("ctx_cont.dist_last_swing_high_atr", [-4.0, 0.6, 0.4, -0.2, -0.3, -4.0, -4.0, -0.2])
    set_col("ctx_cont.dist_last_swing_low_atr", [4.0, 4.0, 4.0, 0.2, 0.3, -0.6, -0.4, 0.2])
    set_col("ctx_cont.bars_since_swing_high", [48, 1, 2, 4, 6, 48, 48, 4])
    set_col("ctx_cont.bars_since_swing_low", [48, 48, 48, 4, 6, 1, 2, 4])
    set_col("ctx_cont.struct_tf_agree_count_v3", [0, 5, 4, 1, 4, 5, 4, 4])

    for tf in ("m5", "m15", "h1", "h4", "d1"):
        set_col(f"ctx_cont.struct_continuation_up_{tf}_v3", [0.0, 0.9, 0.8, 0.6, 0.8, 0.0, 0.0, 0.1])
        set_col(f"ctx_cont.struct_pullback_in_uptrend_{tf}_v3", [0.0, 0.7, 0.6, 0.5, 0.7, 0.0, 0.0, 0.0])
        set_col(f"ctx_cont.struct_continuation_down_{tf}_v3", [0.0, 0.0, 0.0, 0.6, 0.0, 0.9, 0.8, 0.8])
        set_col(f"ctx_cont.struct_bounce_in_downtrend_{tf}_v3", [0.0, 0.0, 0.0, 0.5, 0.0, 0.7, 0.6, 0.7])
        set_col(f"ctx_cont.struct_pullback_depth_{tf}_v3", [0.2, 0.4, 0.5, 0.5, 0.4, 0.4, 0.5, 0.4])
    return x


def test_structure_swing_derivations_capture_quality_state_and_routing() -> None:
    names = list(STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS)
    x = _matrix(names)
    out, out_names = build_entry_structure_swing_derivation_layer(x, names)
    idx = {name: i for i, name in enumerate(out_names)}

    assert len(STRUCTURE_SWING_DERIVATION_FEATURE_NAMES) == 28
    assert STRUCTURE_SWING_DERIVATION_FEATURE_NAMES == EXPECTED_STRUCTURE_SWING_DERIVATION_FEATURE_NAMES
    assert out.shape == (8, len(EXPECTED_STRUCTURE_SWING_DERIVATION_FEATURE_NAMES))
    assert tuple(out_names) == EXPECTED_STRUCTURE_SWING_DERIVATION_FEATURE_NAMES
    assert np.isfinite(out).all()
    assert all(classify_entry_specialist_feature(name) == "structure_swing_encoder" for name in out_names)

    assert out[2, idx["chart.structure_swing_hh_hl_consistency_up"]] > out[2, idx["chart.structure_swing_lh_ll_consistency_down"]]
    assert out[6, idx["chart.structure_swing_lh_ll_consistency_down"]] > out[6, idx["chart.structure_swing_hh_hl_consistency_up"]]
    assert out[3, idx["chart.structure_swing_hh_hl_lh_ll_conflict"]] > out[1, idx["chart.structure_swing_hh_hl_lh_ll_conflict"]]
    assert out[2, idx["chart.structure_swing_swing_leg_quality_up"]] > out[2, idx["chart.structure_swing_swing_leg_quality_down"]]
    assert out[6, idx["chart.structure_swing_swing_leg_quality_down"]] > out[6, idx["chart.structure_swing_swing_leg_quality_up"]]
    assert out[1, idx["chart.structure_swing_bos_choch_recency_alignment_up"]] > 0.8
    assert out[5, idx["chart.structure_swing_bos_choch_recency_alignment_down"]] > 0.8
    assert out[3, idx["chart.structure_swing_bos_choch_recency_conflict"]] > out[1, idx["chart.structure_swing_bos_choch_recency_conflict"]]
    assert out[4, idx["chart.structure_swing_bos_choch_recency_conflict"]] > out[2, idx["chart.structure_swing_bos_choch_recency_conflict"]]
    assert out[1, idx["chart.structure_swing_bos_followthrough_up_quality"]] > 0.4
    assert out[5, idx["chart.structure_swing_bos_followthrough_down_quality"]] > 0.4
    assert out[1, idx["chart.structure_swing_break_confirmation_up"]] > 0.25
    assert out[5, idx["chart.structure_swing_break_confirmation_down"]] > 0.25
    assert out[1, idx["chart.structure_swing_break_confirmation_balance"]] > 0.25
    assert out[5, idx["chart.structure_swing_break_confirmation_balance"]] < -0.25
    assert out[4, idx["chart.structure_swing_choch_failure_up_risk"]] > 0.3
    assert out[7, idx["chart.structure_swing_choch_failure_down_risk"]] > 0.3
    assert out[2, idx["chart.structure_swing_pullback_depth_phase_alignment_up"]] > 0.4
    assert out[6, idx["chart.structure_swing_pullback_depth_phase_alignment_down"]] > 0.4
    assert out[2, idx["chart.structure_swing_pullback_phase_continuation_up"]] > 0.2
    assert out[6, idx["chart.structure_swing_pullback_phase_continuation_down"]] > 0.2
    assert out[1, idx["chart.structure_swing_pullback_phase_continuation_up"]] > out[1, idx["chart.structure_swing_pullback_phase_continuation_down"]]
    assert out[5, idx["chart.structure_swing_pullback_phase_continuation_down"]] > out[5, idx["chart.structure_swing_pullback_phase_continuation_up"]]
    assert out[2, idx["chart.structure_swing_market_structure_regime_state"]] > 0.5
    assert out[6, idx["chart.structure_swing_market_structure_regime_state"]] < -0.5
    assert out[1, idx["chart.structure_swing_mtf_structure_agreement"]] > out[3, idx["chart.structure_swing_mtf_structure_agreement"]]
    assert out[3, idx["chart.structure_swing_mtf_structure_divergence"]] > out[1, idx["chart.structure_swing_mtf_structure_divergence"]]
    assert out[3, idx["chart.structure_swing_structure_compression_pressure"]] > out[1, idx["chart.structure_swing_structure_compression_pressure"]]
    assert out[3, idx["chart.structure_swing_swing_compression_setup"]] > out[1, idx["chart.structure_swing_swing_compression_setup"]]


def test_structure_swing_derivations_reject_nonfinite_inputs() -> None:
    names = list(STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS)
    x = _matrix(names)
    idx = {name: i for i, name in enumerate(names)}
    x[1, idx["chart.foundation_hh_state"]] = np.nan
    x[2, idx["chart.foundation_hl_state"]] = np.inf
    x[3, idx["chart.foundation_lh_state"]] = -np.inf
    x[4, idx["chart.foundation_bos_up_recent_tau24"]] = np.nan
    x[5, idx["chart.foundation_pullback_depth_norm"]] = np.inf

    with pytest.raises(RuntimeError, match="STRUCTURE_SWING_DERIVATION_SOURCE_NONFINITE"):
        build_entry_structure_swing_derivation_layer(x, names)


def test_structure_swing_derivations_do_not_read_future_rows() -> None:
    names = list(STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS)
    base = _matrix(names)
    changed_future = base.copy()
    changed_future[3:, :] = 0.0
    changed_future[3:, names.index("chart.foundation_structure_up_minus_down")] = -2.0
    changed_future[3:, names.index("chart.foundation_choch_recent_tau24")] = 1.0

    out_base, out_names = build_entry_structure_swing_derivation_layer(base, names)
    out_changed, changed_names = build_entry_structure_swing_derivation_layer(changed_future, names)

    assert changed_names == out_names
    np.testing.assert_allclose(out_base[:3], out_changed[:3])


def test_structure_swing_derivation_source_contract_is_explicit() -> None:
    assert STRUCTURE_SWING_DERIVATION_FEATURE_NAMES == EXPECTED_STRUCTURE_SWING_DERIVATION_FEATURE_NAMES
    assert len(STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS) == len(set(STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS))
    assert "chart.foundation_bos_up_recent_tau24" in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS
    assert "ctx_cont.dist_last_swing_high_atr" in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS
    assert "ctx_cont.dist_last_swing_low_atr" in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS
    assert "ctx_cont.bars_since_swing_high" in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS
    assert "ctx_cont.bars_since_swing_low" in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS
    assert "ctx_cont.struct_continuation_up_h1_v3" in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS
    assert "ctx_cont.struct_pullback_depth_d1_v3" in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS
    assert missing_structure_swing_derivation_source_fields(STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS) == []
    missing = missing_structure_swing_derivation_source_fields(
        name for name in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS if name != "chart.foundation_hh_state"
    )
    assert missing == ["chart.foundation_hh_state"]


def test_structure_swing_derivations_fail_closed_on_missing_source_fields() -> None:
    names = [name for name in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS if name != "ctx_cont.dist_last_swing_high_atr"]
    x = np.zeros((4, len(names)), dtype=np.float32)

    try:
        build_entry_structure_swing_derivation_layer(x, names)
    except RuntimeError as exc:
        assert "structure/swing derivation required source fields missing" in str(exc)
        assert "ctx_cont.dist_last_swing_high_atr" in str(exc)
    else:
        raise AssertionError("builder should fail closed when required swing distance source is missing")
