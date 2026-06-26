import numpy as np

from gx1.audit.entry_transformer_feature_audit import (
    DERIVED_CANDIDATE_NAMES,
    _derived_candidate_matrix,
    _family,
)
from gx1.contracts.signal_bridge_v3 import ORDERED_CTX_CONT_NAMES_V3, ORDERED_SEQ_FIELDS_V3


def test_derived_candidate_matrix_keeps_names_and_core_formulas() -> None:
    seq_idx = {name: i for i, name in enumerate(ORDERED_SEQ_FIELDS_V3)}
    ctx_idx = {name: i for i, name in enumerate(ORDERED_CTX_CONT_NAMES_V3)}
    cand_idx = {name: i for i, name in enumerate(DERIVED_CANDIDATE_NAMES)}

    seq = np.zeros((2, 96, len(ORDERED_SEQ_FIELDS_V3)), dtype=np.float32)
    snap = np.zeros((2, len(ORDERED_SEQ_FIELDS_V3)), dtype=np.float32)
    ctx = np.zeros((2, len(ORDERED_CTX_CONT_NAMES_V3)), dtype=np.float32)

    seq[0, -1, seq_idx["smc_choch"]] = 1.0
    seq[1, -13, seq_idx["smc_choch"]] = 1.0
    seq[:, -12:, seq_idx["smc_bos_up"]] = 1.0
    seq[:, -48:, seq_idx["smc_sweep_down"]] = 1.0
    seq[:, -1, seq_idx["smc_sweep_size_atr"]] = 2.0
    snap[:, seq_idx["smc_bars_since_sweep"]] = [0.0, 24.0]
    snap[:, seq_idx["smc_premium_discount"]] = [0.25, 0.75]

    ctx[:, ctx_idx["dist_to_R1_atr"]] = [2.0, 4.0]
    ctx[:, ctx_idx["dist_to_R2_atr"]] = [3.0, 5.0]
    ctx[:, ctx_idx["dist_to_S1_atr"]] = [1.0, 6.0]
    ctx[:, ctx_idx["dist_to_S2_atr"]] = [8.0, 7.0]
    for name in ("dist_to_m5_hi_atr", "dist_to_m15_hi_atr", "dist_to_h1_hi_atr", "dist_to_h4_hi_atr", "dist_to_d1_hi_atr"):
        ctx[:, ctx_idx[name]] = 5.0
    for name in ("dist_to_m5_lo_atr", "dist_to_m15_lo_atr", "dist_to_h1_lo_atr", "dist_to_h4_lo_atr", "dist_to_d1_lo_atr"):
        ctx[:, ctx_idx[name]] = 2.0
    for name in ("dip_confirmed_m5_v3", "dip_confirmed_m15_v3", "dip_confirmed_h1_v3", "dip_confirmed_h4_v3", "dip_confirmed_d1_v3"):
        ctx[:, ctx_idx[name]] = 0.5
    for name in ("dip_proximity_h1_v3", "dip_proximity_h4_v3", "dip_proximity_d1_v3"):
        ctx[:, ctx_idx[name]] = 0.25
    ctx[:, ctx_idx["struct_tf_agree_count_v3"]] = [0.2, 0.4]

    out = _derived_candidate_matrix(seq, snap, ctx)

    assert out.shape == (2, len(DERIVED_CANDIDATE_NAMES))
    np.testing.assert_allclose(out[0, cand_idx["smc_choch_recent_tau12"]], 1.0)
    np.testing.assert_allclose(out[1, cand_idx["smc_choch_recent_tau12"]], np.exp(-12.0 / 12.0), rtol=1e-6)
    np.testing.assert_allclose(out[:, cand_idx["smc_bos_pressure_last12"]], [1.0, 1.0])
    np.testing.assert_allclose(out[:, cand_idx["smc_sweep_bull_pressure_last48"]], [1.0, 1.0])
    np.testing.assert_allclose(out[:, cand_idx["sr_nearest_pivot_abs_atr"]], [1.0, 4.0])
    np.testing.assert_allclose(out[:, cand_idx["dip_confirmed_mean_5tf"]], [0.5, 0.5])
    np.testing.assert_allclose(out[:, cand_idx["dip_proximity_mean_h1h4d1"]], [0.25, 0.25])


def test_family_classifies_structure_down_without_day_of_week_collision() -> None:
    assert _family("struct_continuation_down_h4_v3") == "structure_smc_swing"
    assert _family("struct_pullback_in_uptrend_d1_v3") == "structure_smc_swing"
    assert _family("dow_sin") == "session_time"
    assert _family("downside_pressure") == "other"
