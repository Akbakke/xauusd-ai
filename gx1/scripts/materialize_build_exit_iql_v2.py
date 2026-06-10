#!/usr/bin/env python3
"""Build exit-IQL V2: 2-action multi-head Q(s, a, K) on per-bar trade samples.

Mission
-------
Train the RL "smart head" that supervises and overrides the exit transformer
(V3) — same pattern as Entry-IQL guides V10. At each bar inside a held trade,
Exit-IQL recommends HOLD or EXIT_NOW based on per-bar state and counterfactual
forward-bar rewards.

Dataset built by `materialize_build_exit_iql_per_bar_dataset_v1.py`. Each row =
one (candidate, side, sampled bar_t) sample with:
  - candidate context features (regime, session, p_long, etc.)
  - per-bar trade state (bars_in_trade, current pnl/mfe/mae, drawdown, atr, ...)
  - EXIT_NOW reward (signed pnl at bar t)
  - HOLD reward per K-horizon (max future pnl in next K bars)

Action space (2)
----------------
  HOLD, EXIT_NOW

Reward variants (5)
-------------------
  R_RAW       : exit_now_reward,  hold_max_pnl_at_K (raw)
  R_NET05     : same minus 0.5 * future_drawdown (mild risk-adjustment)
  R_GATED     : HOLD reward only if expected MFE > threshold
  R_REGRET    : reward - oracle_max_pnl (negative regret signal)
  R_NET_REAL  : 0.5 * raw_reward - 0.5 * mae - cost (realistic)

Multi-head Q(s, a, K)
---------------------
For HOLD action, Q values per K-horizon (1, 4, 12, 24, 48, 96 bars) tell
the agent how much future MFE is achievable if it holds K more bars. EXIT
action's Q is K-independent (immediate reward).

This is RESEARCH-ONLY. No runtime promotion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import materialize_build_exit_iql_per_bar_dataset_v1 as exit_pipe


ACTION = "BUILD_EXIT_IQL_V2"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_PER_BAR_DIR = (
    DEFAULT_REPORTS_ROOT / "EXIT_IQL_PER_BAR_DATASET_V1"
)

# Override n_actions for 2-action exit policy.
N_ACTIONS_EXIT = 2
ACTION_HOLD_ID = 0
ACTION_EXIT_NOW_ID = 1
ACTION_LABELS_EXIT = {ACTION_HOLD_ID: "HOLD", ACTION_EXIT_NOW_ID: "EXIT_NOW"}

K_HORIZONS = exit_pipe.K_HORIZONS_EXIT  # SCALP: [1, 4, 12, 48, 144, 240] M1-bars (= 1min..4h lookahead)
# 2026-05-24 K-HORIZON MISMATCH (action required, NOT a config swap):
# Trener vil ha SCALP-K [1, 4, 12, 24, 48, 96]. Existing per-bar dataset has
# SWING-K [12, 60, 240, 480, 1440] (built for swing-trade analysis). Only K=12
# intersects -> 5/6 K-slots are degenerate (hold_K falls back to exit_now),
# which is why IQL_EXIT virtually never fires.
#
# RIGHT FIX: rebuild per-bar dataset with hold_max_pnl_K{1,4,12,24,48,96}_v1
# columns to match the scalp-aligned trainer. See
# materialize_build_exit_iql_per_bar_dataset_v2_m1.py.
#
# WRONG FIX: changing K_HORIZONS to [12, 60, 240, 480, 1440] would make trainer
# learn swing-trade exits — opposite of what we want for scalping.
N_K = len(K_HORIZONS)
K_PRIMARY = 48  # 2026-05-24: was 24, but new scalp K_HORIZONS=[1,4,12,48,144,240]. 48 M1=48min.
# NOTE: R_RAW and R_NET05 removed 2026-05-02 — they degenerate to "always HOLD"
# because their HOLD reward = max future MFE has no realization cost. Oracle
# itself becomes degenerate, the policy matches oracle perfectly (~99.6%) but
# the policy is unusable in live trading (trades would never close).
# R_NET_REAL is the production policy. R_GATED + R_REGRET are alternative
# perspectives that all include realistic exit-cost framing.
# 2026-05-24: R_PEAK_AWARE_{MILD,MED,HARSH} added. Penalise drawdown-from-peak,
# MAE, and bars-held more aggressively than R_NET_V2. Goal: trigger IQL_EXIT
# at MFE peaks instead of letting trades drift back to break-even.
REWARD_VARIANTS = [
    "R_NET_REAL", "R_GATED", "R_REGRET", "R_NET_V2",
    "R_PEAK_AWARE_MILD", "R_PEAK_AWARE_MED", "R_PEAK_AWARE_HARSH",
    # R_PEAK_QUALITY (analog til entry R_QUALITY): scale-invariant ratio.
    # r_hold = hold_K * max(0, 1 - drawdown/peak_mfe)
    # r_exit = exit_now - spread
    # Self-balancing: ingen hyperparametre. Modellen lærer "exit når quality faller"
    # via gradient, ikke via hardkodet giveback-terskel.
    "R_PEAK_QUALITY",
    # R_PEAK_QUALITY_QUAD: kvadratisk quality (sterkere straff for giveback).
    "R_PEAK_QUALITY_QUAD",
]
GATED_THRESHOLD_BPS = 10.0
R_NET_REAL_ALPHA = 0.5
R_NET_REAL_BETA = 0.5
R_NET_REAL_GAMMA = 2.0

# V9 Issue C: R_NET_V2 — improved reward shaping for Exit-IQL.
# Adds explicit MFE-giveback penalty, asymmetric vol-aware MAE, and capital cost.
# Designed to address: validation showed Exit-IQL active only 3.8% (over-conservative
# HOLD bias) AND lose-rate 24% on FORCED_TERMINAL — model needs stronger signal that
# "holding past peak is bad".
# R_NET_V2 = R_NET_REAL + giveback penalty + bars cost (incremental change).
# Calibrated so r_hold is ~15-25 bps worse than R_NET_REAL on avg — modest EXIT bias.
R_NET_V2_GIVEBACK = 0.3         # mild MFE-giveback penalty
R_NET_V2_BARS_COST = 0.02       # capital-binding cost per bar held

# 2026-05-24 R_PEAK_AWARE — Phase 6 OOT showed IQL_EXIT fired 2/14,581 times.
# Diagnosis: HOLD reward = max future PnL (perfect foresight) >> EXIT reward (current).
# Plus K-horizon mismatch made 5/6 K-slots degenerate.
# Fix: same skeleton as R_NET_V2 but stronger giveback + MAE + bars_cost so the
# expectile target actually moves below exit_now once we've passed peak.
#   r_hold = ALPHA*hold_K - BETA_MAE*|MAE| - giveback*drawdown - bars_cost*bars - 2*spread
#   r_exit = exit_now - 2*spread
R_PEAK_AWARE_PARAMS = {
    "R_PEAK_AWARE_MILD":  {"alpha": 0.5, "beta_mae": 0.3, "giveback": 0.5, "bars_cost": 0.02},
    "R_PEAK_AWARE_MED":   {"alpha": 0.5, "beta_mae": 0.5, "giveback": 1.0, "bars_cost": 0.05},
    "R_PEAK_AWARE_HARSH": {"alpha": 0.5, "beta_mae": 0.8, "giveback": 2.0, "bars_cost": 0.10},
}

# Cross-validation
N_FOLDS = 3
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SEED_V1 = 20260501
MIN_PER_CLASS_VAL = 50

AGGREGATORS = ["mean", "max"]

# Numeric state columns: per-bar trade state + carried candidate context.
NUMERIC_STATE_COLS_PER_BAR = [
    "bars_in_trade_v1",
    "current_unrealized_pnl_bps_v1",
    "current_mfe_bps_v1",
    "current_mae_bps_v1",
    "bars_since_mfe_peak_v1",
    "pnl_drawdown_from_peak_v1",
    "current_atr_bps_v1",
    "bar_return_bps_v1",
]
# NOTE: entry_spread_bps removed — POST-decision field, 99.7% NaN.
# Use canonical/chunk_0 `spread_bps` (at-bar pre-decision) instead.
NUMERIC_STATE_COLS_CANDIDATE = [
    "weekday_utc",
    "hour_utc",
    "atr_bps",
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "margin",
    "uncertainty_score",
    "tradable_prob",
    "mfe_first_n_pred",
    "path_quality_pred",
]
# CHUNK0 features dropped 2026-05-21: chunk_0_data parquet was missing from
# training inputs, so chunk0_v1 cols were zero-filled and contributed nothing.
# Confirmed by feature-importance analysis (gain=0, perm=0 across the board).
NUMERIC_STATE_COLS_CHUNK0: list[str] = []
# Canonical V10 features per held bar (basic_v1 + H1 + H4 + m5_phase + high-level)
NUMERIC_STATE_COLS_CANONICAL = [
    f"{c}{exit_pipe.CANONICAL_FEATURES_SUFFIX}"
    for c in [
        # 2026-05-21: pruned canon_v1 features (constant/duplicated per
        # feature-importance analysis): _v1_atr_regime_id,
        # _v1_bb_bandwidth_delta_10, _v1_lower_tr, _v1_spread_p, _v1_slip_bps,
        # _v1_spread_z, _v1_cost_bps_est, _v1_comp3_ratio.
        "_v1_r3", "_v1_r5", "_v1_r8", "_v1_r12", "_v1_r24", "_v1_r1", "_v1_r48_z",
        "_v1_atr14", "_v1_pk_sigma20", "_v1_ema_diff",
        "_v1_vwap_drift48", "_v1_rsi14_z", "_v1_rsi2", "_v1_rsi14",
        "_v1_rsi2_gt_rsi14", "_v1_ret_ema_diff_2_5",
        "_v1_close_ema_slope_3", "_v1_body_tr", "_v1_upper_tr",
        "_v1_wick_imbalance", "_v1_range_comp_20_100", "_v1_range_adr",
        "_v1_clv", "_v1_range_z",
        "_v1_kurt_r", "_v1_int_r5_atr", "_v1_int_vwap_h1", "_v1_int_slope_h4_atr",
        "_v1_int_clv_atr", "_v1_cost_bps_dyn", "_v1_tod_sin", "_v1_tod_cos",
        "_v1_r1_q90_48", "_v1_r1_q10_48", "_v1_atr_z_10_100", "_v1_tema_slope_20",
        "_v1_bb_squeeze_20_2", "_v1_kama_slope_30", "_v1_ret_ema_ratio_5_34",
        "_v1_body_share_1", "_v1_tr_1_over_atr_14",
        "_v1h1_ema_diff", "_v1h1_vwap_drift", "_v1h1_atr", "_v1h1_rsi14_z",
        "_v1h1_slope3", "_v1h1_slope5",
        "_v1h4_ema_diff", "_v1h4_atr", "_v1h4_rsi14_z", "_v1h4_slope3", "_v1h4_slope5",
        "m5_phase_0", "m5_phase_1", "m5_phase_2", "m5_phase_3", "m5_phase_4",
        "mid", "range", "body_pct", "wick_asym", "atr", "atr50", "atr_z", "std50",
        "ret_1", "ret_5", "ret_20", "roc20", "roc100", "rvol_20", "rvol_60",
        "ema20_slope", "ema100_slope", "pos_vs_ema200", "vol_ratio",
        # 2026-06-04 (Phase 0a / E2): wire 19 dead-zero canonical_v3 feats that ARE
        # joined into the per-bar state (the comprehension appends CANONICAL_FEATURES_SUFFIX
        # = _canon_v1) but were never in this allowlist, so the Exit-IQL state matrix
        # silently zero-filled them. All 19 verified present in canonical_features_v3
        # (113 cols) with real variance (live-merge check). smc_choch DROPPED (nnz 0.09%).
        # SMC structure (9):
        "smc_swing_state", "smc_bos_up", "smc_bos_down",
        "smc_sweep_up", "smc_sweep_down", "smc_sweep_size_atr",
        "smc_bars_since_sweep", "smc_premium_discount", "smc_premium_state",
        # HTF D1/M15 (9) — source cols carry an embedded _canon_v2 base, so post-join
        # name is X_canon_v2_canon_v1:
        "d1_rsi14_canon_v2", "d1_ema_slope_20_canon_v2", "d1_atr14_canon_v2",
        "d1_range_z_20_canon_v2", "d1_close_pct_in_20day_range_canon_v2",
        "d1_pct_change_5_canon_v2", "m15_rsi14_canon_v2",
        "m15_range_z_20_canon_v2", "m15_trend_sign_canon_v2",
        # M5/H1 cross-TF momentum (1):
        "m5h1_momentum",
    ]
]
# Phase 0a/O5 (2026-06-04): DROP 9 legacy _canon_v1 feats that have NO source in
# canonical_features_v3 (verified absent -> zero-filled at train, while live recomputes
# them -> a latent train/serve skew). Redundant with _v1_atr14 / _v1_atr_z / _v1_pk_sigma20.
# The cement bundle (which trained on them as constant zeros) is preserved on disk; this
# removes the skew + 9 dead inputs from the retrain contract. No env knob — the cement is
# the artifact, not code-reproduced here.
_O5_DEAD_CANON_V1 = {
    "_v1_vwap_drift48_canon_v1", "_v1_body_tr_canon_v1", "_v1_int_r5_atr_canon_v1",
    "_v1_int_slope_h4_atr_canon_v1", "_v1_int_clv_atr_canon_v1", "_v1h1_vwap_drift_canon_v1",
    "atr_canon_v1", "std50_canon_v1", "roc20_canon_v1",
}
NUMERIC_STATE_COLS_CANONICAL = [c for c in NUMERIC_STATE_COLS_CANONICAL if c not in _O5_DEAD_CANON_V1]
# V3 trade-state derivatives (per held bar)
NUMERIC_STATE_COLS_TRADE_DERIV = list(exit_pipe.TRADE_STATE_DERIVATIVE_COLS)
# V3 entry-snapshot fields carried through trade
NUMERIC_STATE_COLS_ENTRY_SNAPSHOT = list(exit_pipe.ENTRY_SNAPSHOT_COLS)
NUMERIC_STATE_COLS_CANDIDATE = (
    NUMERIC_STATE_COLS_CANDIDATE
    + NUMERIC_STATE_COLS_CHUNK0
    + NUMERIC_STATE_COLS_CANONICAL
    + NUMERIC_STATE_COLS_TRADE_DERIV
    + NUMERIC_STATE_COLS_ENTRY_SNAPSHOT
)

# ─────────────────────────────────────────────────────────────────────────────
# V2 (2026-05-22): C+prune strategy for Exit-IQL.
# Reads from AUGMENTED per_bar dataset (V2 cols joined by candidate_uid).
# Drops 57 features identified by Exit-IQL permutation importance.
# Adds 125 per-TF V2 + 28 group-A → ~250 pre-prune features.
# ─────────────────────────────────────────────────────────────────────────────

V2_BUILD_MODE_EXIT = bool(int(os.environ.get("GX1_BUILD_EXIT_IQL_V2", "0")))

# Exit-IQL FI drop list (57 features, perm-imp ≈ 0).
V2_DROP_FEATURES_EXIT = (
    "bad_path_prob", "m1_last_60bar_return_bps_v1", "m1_realized_vol_60bar_bps_v1",
    "hour_utc", "rolling_slope_since_entry_v1", "bars_in_trade_v1", "p_long",
    "v3_v8_family_logit_max", "path_quality_std", "m1_last_15bar_return_bps_v1",
    "mfe_first_n_pred", "direction_logit_flat", "mfe_decay_rate_v1",
    "tf_agreement_pred", "path_quality_pred", "p_flat", "margin",
    "m1_realized_vol_15bar_bps_v1", "hold_horizon_pred", "forward_bars_remaining_v1",
    "direction_logit_short", "pnl_acceleration_v1", "p_hat", "m5_phase_4_v1",
    "direction_logit_long", "giveback_acceleration_v1", "bar_return_bps_v1",
    "v3_signal_acceleration_v1", "m1_last_5bar_return_bps_v1", "current_atr_bps_v1",
    "m5_phase_2_v1", "pnl_velocity_v1", "entropy_entry_v1",
    "v3_max_consecutive_exits_v1", "m5_phase_1_v1", "tradable_prob",
    "m5_phase_0_v1", "m5_phase_3_v1", "v3_consecutive_exits_v1",
    "v3_v8_family_argmax", "uncertainty_score", "weekday_utc",
    "v12_capital_cost_bps_v1", "v10_p_long_at_entry_v1", "accepted",
    "p_long_entry_v1", "uncertainty_entry_v1", "margin_entry_v1",
    "v10_path_quality_at_entry_v1", "v10_path_quality_std_at_entry_v1",
    "v3_should_exit_decision_v1", "v10_position_size_at_entry_v1",
    "v10_mfe_pred_at_entry_v1", "v10_tf_agreement_at_entry_v1",
    "v10_hold_horizon_at_entry_v1", "v10_p_short_at_entry_v1", "p_hat_entry_v1",
    # giveback_ratio_v1 — kept (high importance per FI)
)

# V2 per-TF (125) and group-A (28) — same definitions as Entry-IQL builder
def _build_v2_per_tf_cols_exit():
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V2
    tfs = ("m5", "m15", "h1", "h4", "d1")
    cols = []
    for tf in tfs:
        for feat in MULTI_TF_PER_BAR_FEATURES_V2:
            base = f"{tf}_{feat}"
            cols.append(base if feat.endswith("_v2") else base + "_v2")
    return tuple(cols)

V2_PER_TF_COLS_EXIT = _build_v2_per_tf_cols_exit()
V2_GROUP_A_COLS_EXIT = (
    "is_asia_eu_overlap", "is_eu_us_overlap", "is_eu_only", "is_us_only",
    "atr_ratio_m5_h4", "atr_ratio_m15_d1", "atr_ratio_h1_d1", "atr_ratio_m5_m15",
    "vol_pct_m5_1yr", "vol_pct_h1_1yr",
    "dist_to_R1_atr", "dist_to_R2_atr", "dist_to_S1_atr", "dist_to_S2_atr",
    # 2026-05-24: added M15 + D1 dist_to_hi/lo (Bug 2 fix in group_a_features.py:199)
    "dist_to_m5_hi_atr",  "dist_to_m5_lo_atr",
    "dist_to_m15_hi_atr", "dist_to_m15_lo_atr",
    "dist_to_h1_hi_atr",  "dist_to_h1_lo_atr",
    "dist_to_h4_hi_atr",  "dist_to_h4_lo_atr",
    "dist_to_d1_hi_atr",  "dist_to_d1_lo_atr",
    "long_win_rate_last10",  "long_mean_pnl_last10",
    "long_n_consec_losses",  "long_time_since_last_close_min",
    "short_win_rate_last10", "short_mean_pnl_last10",
    "short_n_consec_losses", "short_time_since_last_close_min",
    # 2026-05-24 PM: dip + struct features (joined via augment_per_bar _v3 suffix).
    # Exit needs these to see "M15 turning down → strong-hold or exit-now" patterns.
    # 2026-05-24 PM 2: dropped dip_proximity_m5/m15 (saturated 99% nnz, low std).
    "dip_confirmed_m5_v3",
    "dip_confirmed_m15_v3",
    "dip_proximity_h1_v3",  "dip_confirmed_h1_v3",
    "dip_proximity_h4_v3",  "dip_confirmed_h4_v3",
    "dip_proximity_d1_v3",  "dip_confirmed_d1_v3",
    "struct_continuation_up_m5_v3",  "struct_pullback_in_uptrend_m5_v3",
    "struct_continuation_down_m5_v3","struct_bounce_in_downtrend_m5_v3",
    "struct_pullback_depth_m5_v3",
    "struct_continuation_up_m15_v3", "struct_pullback_in_uptrend_m15_v3",
    "struct_continuation_down_m15_v3","struct_bounce_in_downtrend_m15_v3",
    "struct_pullback_depth_m15_v3",
    "struct_continuation_up_h1_v3",  "struct_pullback_in_uptrend_h1_v3",
    "struct_continuation_down_h1_v3","struct_bounce_in_downtrend_h1_v3",
    "struct_pullback_depth_h1_v3",
    "struct_continuation_up_h4_v3",  "struct_pullback_in_uptrend_h4_v3",
    "struct_continuation_down_h4_v3","struct_bounce_in_downtrend_h4_v3",
    "struct_pullback_depth_h4_v3",
    "struct_continuation_up_d1_v3",  "struct_pullback_in_uptrend_d1_v3",
    "struct_continuation_down_d1_v3","struct_bounce_in_downtrend_d1_v3",
    "struct_pullback_depth_d1_v3",
    # 2026-05-24 PM: dropped struct_all_tf_pullback_v3 (0% nnz — never fires).
    "struct_tf_agree_count_v3",
    "struct_dip_x_uptrend_v3",   "struct_smc_swing_x_dip_v3",
)

if V2_BUILD_MODE_EXIT:
    # Drop weak features from PER_BAR + CANDIDATE state, then add V2 cols
    _per_bar_kept = [c for c in NUMERIC_STATE_COLS_PER_BAR if c not in V2_DROP_FEATURES_EXIT]
    _candidate_kept = tuple(c for c in NUMERIC_STATE_COLS_CANDIDATE if c not in V2_DROP_FEATURES_EXIT)
    NUMERIC_STATE_COLS_PER_BAR = _per_bar_kept
    NUMERIC_STATE_COLS_CANDIDATE = _candidate_kept + V2_PER_TF_COLS_EXIT + V2_GROUP_A_COLS_EXIT
    print(f"[V2_BUILD_MODE_EXIT] bar_state: {len(_per_bar_kept)} per-bar-kept + "
          f"{len(_candidate_kept)} candidate-kept + {len(V2_PER_TF_COLS_EXIT)} per-TF + "
          f"{len(V2_GROUP_A_COLS_EXIT)} group-A = "
          f"{len(NUMERIC_STATE_COLS_PER_BAR) + len(NUMERIC_STATE_COLS_CANDIDATE)} features",
          flush=True)
ONE_HOT_COLS = [
    "session",
    "vol_regime",
    "trend_regime",
    "decision_reason",
    "side_v1",  # adds per-bar side as one-hot
]

ALLOWED_FINAL_STATUSES = {
    "EXIT_IQL_V2_PASS_BEATS_BASELINE",
    "EXIT_IQL_V2_PARTIAL_TIES_BASELINE",
    "EXIT_IQL_V2_PARTIAL_DEGRADES_VS_BASELINE",
    "EXIT_IQL_V2_PARTIAL_DEGENERATE_ACTION_DIST",
    "EXIT_IQL_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
}
ALLOWED_NEXT_ACTIONS = {
    "BUILD_JOINT_ENTRY_AND_EXIT_IQL_V2",
    "REPAIR_EXIT_IQL_V2_BEFORE_FURTHER_WORK",
    "BUILD_EXIT_IQL_V2_RUNTIME_ADAPTER",
}

# Training budget presets — same shape as entry-IQL for consistency
BUDGET_PRESETS = {
    # 2026-06-10 (user vedtak): batch 256/512 was a 16-32x THROUGHPUT LEAK for the tiny 3-layer IQL
    # MLP — it trains via a manual in-memory minibatch loop (~13.7k iters/epoch at 256), so the GPU
    # sat at ~38% and Python-loop overhead dominated. Tiny net => BIG batch (SMART+MAXED rule). Batch
    # is now 4096 everywhere (~16x fewer iters, GPU saturated). NEVER drop a tiny IQL back to 256.
    "fast":   {"epochs_q":  30, "epochs_v": 15, "k_iter": 3, "batch": 4096, "hidden": 128, "n_hidden": 2},
    "medium": {"epochs_q":  60, "epochs_v": 25, "k_iter": 4, "batch": 4096, "hidden": 192, "n_hidden": 3},
    "full":   {"epochs_q": 100, "epochs_v": 40, "k_iter": 6, "batch": 4096, "hidden": 256, "n_hidden": 3},
}
TRAIN_TAU = 0.8
TRAIN_EPOCHS_Q = 30
TRAIN_EPOCHS_V = 15
TRAIN_K_VQ_ITERATIONS = 3
TRAIN_BATCH_SIZE = 512
TRAIN_HIDDEN_DIM = 128
TRAIN_N_HIDDEN = 2

_stamp = contract_gate._stamp
_utc_now = contract_gate._utc_now
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows


# ---------------------------------------------------------------------------
# Data loading + reward construction
# ---------------------------------------------------------------------------


def load_per_bar_dataset(per_week_dir: Path, sample_n_rows: int | None = None) -> pd.DataFrame:
    """Load per-bar exit dataset, optionally pre-subsampling per file to bound memory.

    For 20M+ row datasets, loading all weeks then subsampling causes OOM. When
    sample_n_rows is given, we instead read each parquet and stratified-sample
    proportionally — keeping ~sample_n_rows / n_files rows per file before concat.
    """
    parts: list[pd.DataFrame] = []
    parquets = sorted(per_week_dir.glob("exit_per_bar_*.parquet"))
    if not parquets:
        raise RuntimeError(f"[{ACTION}] no exit_per_bar_*.parquet found in {per_week_dir}")

    if sample_n_rows is not None and sample_n_rows > 0:
        # Subsample BEFORE concatenation to avoid OOM on large datasets.
        per_file_target = max(1, int(sample_n_rows / len(parquets)) + 1)
        print(f"[{ACTION}] streaming-subsample: target ~{per_file_target} rows per file × {len(parquets)} files", flush=True)
        rng = np.random.default_rng(20260501)
        for p in parquets:
            df_p = pd.read_parquet(p)
            if len(df_p) > per_file_target:
                idx = rng.choice(len(df_p), size=per_file_target, replace=False)
                df_p = df_p.iloc[idx].reset_index(drop=True)
            parts.append(df_p)
        df = pd.concat(parts, ignore_index=True)
        # Final shuffle + cap to exact sample_n_rows
        if len(df) > sample_n_rows:
            df = df.sample(n=sample_n_rows, random_state=20260501).reset_index(drop=True)
    else:
        for p in parquets:
            parts.append(pd.read_parquet(p))
        df = pd.concat(parts, ignore_index=True)
    return df


def build_reward_matrix(df: pd.DataFrame, *, variant: str) -> np.ndarray:
    """Return R of shape (n_rows, 2_actions, n_K).

    Action layout: 0=HOLD, 1=EXIT_NOW.
    EXIT_NOW reward is K-independent (broadcast across K).
    HOLD reward depends on K (max future pnl over K horizon).
    """
    n = len(df)
    R = np.zeros((n, N_ACTIONS_EXIT, N_K), dtype=np.float32)

    exit_now = pd.to_numeric(df["exit_now_reward_bps_v1"], errors="coerce").fillna(0.0).to_numpy()
    current_mae = pd.to_numeric(df["current_mae_bps_v1"], errors="coerce").fillna(0.0).to_numpy()
    drawdown = pd.to_numeric(df["pnl_drawdown_from_peak_v1"], errors="coerce").fillna(0.0).to_numpy()
    spread = (
        pd.to_numeric(df.get("entry_spread_bps"), errors="coerce")
        .fillna(1.5).clip(lower=0.0).astype(float).to_numpy()
        if "entry_spread_bps" in df.columns
        else np.full(n, 1.5, dtype=float)
    )
    # V9 Issue C: bars_held for R_NET_V2 capital cost.
    bars_held_v2 = (
        pd.to_numeric(df.get("bars_in_trade_v1"), errors="coerce").fillna(0.0).to_numpy()
        if "bars_in_trade_v1" in df.columns
        else np.zeros(n, dtype=float)
    )

    for ki, K in enumerate(K_HORIZONS):
        hold_col = f"hold_max_pnl_K{K}_v1"
        if hold_col in df.columns:
            hold_K = pd.to_numeric(df[hold_col], errors="coerce").fillna(0.0).to_numpy()
        else:
            hold_K = exit_now  # fallback degenerate

        if variant == "R_RAW":
            r_hold = hold_K
            r_exit = exit_now
        elif variant == "R_NET05":
            # Penalize HOLD by current drawdown-from-peak (worse if we've already given up MFE)
            r_hold = hold_K - 0.5 * drawdown
            r_exit = exit_now
        elif variant == "R_NET_REAL":
            r_hold = R_NET_REAL_ALPHA * hold_K - R_NET_REAL_BETA * np.abs(current_mae) - R_NET_REAL_GAMMA * spread
            r_exit = exit_now - R_NET_REAL_GAMMA * spread  # also pay exit spread
        elif variant == "R_GATED":
            r_hold = np.where(hold_K > GATED_THRESHOLD_BPS, hold_K, 0.0)
            r_exit = exit_now
        elif variant == "R_REGRET":
            # Negative regret: reward minus oracle_max (max(exit_now, hold_K))
            oracle = np.maximum(exit_now, hold_K)
            r_hold = hold_K - oracle  # ≤ 0
            r_exit = exit_now - oracle  # ≤ 0
        elif variant == "R_NET_V2":
            # V9 Issue C: R_NET_REAL + giveback + bars cost (incremental).
            # - giveback: penalize holding past MFE peak (regression of unrealized profit)
            # - bars_cost: small capital-binding cost (encourages closing stale positions)
            r_hold = (
                R_NET_REAL_ALPHA * hold_K
                - R_NET_REAL_BETA * np.abs(current_mae)
                - R_NET_REAL_GAMMA * spread
                - R_NET_V2_GIVEBACK * drawdown
                - R_NET_V2_BARS_COST * bars_held_v2
            )
            r_exit = exit_now - R_NET_REAL_GAMMA * spread
        elif variant in R_PEAK_AWARE_PARAMS:
            # R_PEAK_AWARE — addresses Phase 6 OOT finding that IQL_EXIT virtually never
            # fires (2/14,581) because HOLD reward (max future PnL = perfect foresight)
            # systematically dominates EXIT reward (current PnL).
            # Stronger giveback + MAE + bars_cost so r_hold drops below exit_now
            # once we've passed peak, encouraging the IQL to learn EXIT_NOW signals.
            p = R_PEAK_AWARE_PARAMS[variant]
            r_hold = (
                p["alpha"] * hold_K
                - p["beta_mae"] * np.abs(current_mae)
                - p["giveback"] * drawdown
                - p["bars_cost"] * bars_held_v2
                - R_NET_REAL_GAMMA * spread
            )
            r_exit = exit_now - R_NET_REAL_GAMMA * spread
        elif variant in ("R_PEAK_QUALITY", "R_PEAK_QUALITY_QUAD"):
            # Scale-invariant exit-quality (analog to entry R_QUALITY).
            # quality = max(0, 1 - drawdown_from_peak / peak_mfe)
            # When at peak: quality=1 → full HOLD reward
            # When given back 50% of peak: quality=0.5 → half HOLD reward
            # When given back 100% of peak: quality=0 → no HOLD reward → EXIT triggered
            # Self-balancing — replaces hardcoded R_PEAK_AWARE giveback/bars_cost lambdas.
            peak_mfe_col = pd.to_numeric(df.get("current_mfe_bps_v1"), errors="coerce").fillna(0.0).to_numpy()
            peak_mfe_safe = np.maximum(peak_mfe_col, 1.0)  # avoid div0; if no peak yet, quality≈1
            quality = np.clip(1.0 - drawdown / peak_mfe_safe, 0.0, 1.0)
            if variant == "R_PEAK_QUALITY_QUAD":
                quality = quality ** 2  # steeper falloff on giveback
            r_hold = hold_K * quality - R_NET_REAL_GAMMA * spread
            r_exit = exit_now - R_NET_REAL_GAMMA * spread
        else:
            raise ValueError(f"unknown reward variant: {variant}")

        r_hold = np.clip(r_hold, -500.0, 500.0)
        r_exit = np.clip(r_exit, -500.0, 500.0)
        R[:, ACTION_HOLD_ID, ki] = r_hold.astype(np.float32)
        R[:, ACTION_EXIT_NOW_ID, ki] = r_exit.astype(np.float32)
    return R


def build_state_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build state matrix with same data-integrity contract as entry-IQL trainer."""
    if "_canonical_features_complete_v1" in df.columns:
        complete_mask = df["_canonical_features_complete_v1"].fillna(False).astype(bool)
        if "_chunk0_features_complete_v1" in df.columns:
            complete_mask = complete_mask & df["_chunk0_features_complete_v1"].fillna(False).astype(bool)
        n_dropped = int((~complete_mask).sum())
        if n_dropped > 0:
            print(f"[BUILD_STATE_MATRIX] dropping {n_dropped:,} rows with incomplete features", flush=True)
        df = df.loc[complete_mask].reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    feature_names: list[str] = []
    nan_warnings: list[tuple[str, float]] = []
    for c in list(NUMERIC_STATE_COLS_PER_BAR) + list(NUMERIC_STATE_COLS_CANDIDATE):
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            nan_frac = s.isna().mean()
            if nan_frac > 0.05:
                nan_warnings.append((c, float(nan_frac)))
            parts.append(pd.DataFrame({c: s.fillna(0.0)}))
            feature_names.append(c)
        else:
            parts.append(pd.DataFrame({c: np.zeros(len(df))}))
            feature_names.append(c)
    if nan_warnings:
        msg = ", ".join(f"{n}={f:.1%}" for n, f in nan_warnings[:8])
        raise RuntimeError(
            f"[BUILD_STATE_MATRIX] {len(nan_warnings)} feature(s) have >5% NaN "
            f"(first 8 listed): {msg}. Investigate upstream pipeline before training."
        )
    for c in ONE_HOT_COLS:
        if c not in df.columns:
            continue
        s = df[c].astype("string").fillna("UNKNOWN")
        for v in sorted(s.unique().tolist()):
            col_name = f"{c}__{v}"
            parts.append(pd.DataFrame({col_name: (s == v).astype(np.float32).to_numpy()}))
            feature_names.append(col_name)
    X_df = pd.concat(parts, axis=1).reset_index(drop=True)
    X = X_df.to_numpy().astype(np.float32)
    return X, feature_names


def derive_oracle_action(R: np.ndarray, K_idx: int) -> np.ndarray:
    return R[:, :, K_idx].argmax(axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Folds, evaluation
# ---------------------------------------------------------------------------


def build_stratified_folds(
    n_rows: int, oracle_action: np.ndarray,
    *, n_folds: int = N_FOLDS, val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE, seed: int = SEED_V1,
    groups: np.ndarray | None = None,
) -> list[dict[str, np.ndarray]]:
    """Build train/val/test folds.

    EXIT-8 (2026-06-03 cross-model scan): when `groups` (per-row candidate_uid) is provided,
    use GroupShuffleSplit so ALL ~480 autocorrelated M1 bars of one trade land entirely on one
    side — the leaky StratifiedShuffleSplit otherwise scatters a single trade across train AND
    test, inflating the "OOT" estimate. groups=None (default) reproduces the cemented stratified
    split bit-for-bit. Group mode forfeits oracle-action stratification (GroupShuffleSplit can't
    stratify); that's the correct trade for an honest split here.
    """
    folds: list[dict[str, np.ndarray]] = []
    all_idx = np.arange(n_rows)
    g = np.asarray(groups) if groups is not None else None
    if g is not None and g.shape[0] != n_rows:
        raise ValueError(f"groups length {g.shape[0]} != n_rows {n_rows}")
    for fold_id_v1 in range(n_folds):
        non_train_size = val_size + test_size
        if g is not None:
            gss1 = GroupShuffleSplit(n_splits=1, test_size=non_train_size, random_state=seed + fold_id_v1)
            train_idx, valtest_idx = next(gss1.split(all_idx, oracle_action, groups=g))
            gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size / non_train_size, random_state=seed + fold_id_v1 + 7919)
            rel_val_idx, rel_test_idx = next(gss2.split(valtest_idx, oracle_action[valtest_idx], groups=g[valtest_idx]))
            val_idx = valtest_idx[rel_val_idx]
            test_idx = valtest_idx[rel_test_idx]
        else:
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=non_train_size, random_state=seed + fold_id_v1)
            train_idx, valtest_idx = next(sss1.split(all_idx, oracle_action))
            valtest_oracle = oracle_action[valtest_idx]
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / non_train_size, random_state=seed + fold_id_v1 + 7919)
            rel_val_idx, rel_test_idx = next(sss2.split(valtest_idx, valtest_oracle))
            val_idx = valtest_idx[rel_val_idx]
            test_idx = valtest_idx[rel_test_idx]
        folds.append({
            "fold_id_v1": f"FOLD_{fold_id_v1 + 1}",
            "train_idx_v1": train_idx,
            "val_idx_v1": val_idx,
            "test_idx_v1": test_idx,
        })
    return folds


def maybe_candidate_uid_groups(df: "pd.DataFrame") -> np.ndarray | None:
    """EXIT-8: ONE-truth resolver for the group-split key, shared by all exit call sites.

    Returns the per-row candidate_uid array when GX1_EXIT_IQL_GROUP_SPLIT is truthy, else None
    (= cement stratified split). Fail-closed: if the flag is set but candidate_uid is missing,
    RAISE rather than silently fall back to the leaky split.
    """
    if os.environ.get("GX1_EXIT_IQL_GROUP_SPLIT", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    if "candidate_uid" not in df.columns:
        raise RuntimeError(
            "GX1_EXIT_IQL_GROUP_SPLIT=1 but 'candidate_uid' is absent from the per-bar df — "
            "cannot do a leak-free group split; fix the dataset or unset the flag."
        )
    return df["candidate_uid"].to_numpy()


# R16 (2026-06-04): trades hold up to ~24h of autocorrelated M1 bars; the embargo must be >= the
# max trade duration so a train trade's full forward bar-window has closed before any val/test
# trade is decided (no straddle). 24h is a safe upper bound on the exit horizon.
EXIT_EMBARGO_NS = 24 * 3600 * 1_000_000_000


def build_chronological_group_folds(
    decision_ts: np.ndarray,          # per-row trade decision time (decision_ts_utc), len == n_rows
    groups: np.ndarray,               # per-row candidate_uid, len == n_rows
    n_folds: int = N_FOLDS,
    embargo_ns: int = EXIT_EMBARGO_NS,
) -> list[dict[str, np.ndarray]]:
    """R16: group-aware purged walk-forward chronological splits for the per-bar EXIT-IQL.

    The per-bar exit set has ~480 autocorrelated M1 bars per trade. An honest OOT split must
    (a) keep EVERY bar of a trade on ONE side (candidate_uid grouping), AND (b) be chronological in
    TRADE-decision time (train = earlier trades, val/test = later) with an embargo gap so no train
    trade's forward bar-window straddles a later split. Mirrors entry-IQL build_chronological_folds
    but at the TRADE (group) granularity. Row indices returned are into the ORIGINAL (unsorted) row
    order; fold N's test is the most-recent trade block (true OOT).
    """
    n_rows = len(decision_ts)
    if len(groups) != n_rows:
        raise ValueError(f"[EXIT_CHRONO_SPLIT] groups len {len(groups)} != n_rows {n_rows}")
    # Parse decision_ts -> int64 ns (robust to ISO str / datetime64 / int64-ns). NaT -> INT64_MIN.
    ts = pd.to_datetime(pd.Series(decision_ts), utc=True, errors="coerce").astype("int64").to_numpy()
    if np.any(ts <= 0):
        raise RuntimeError(
            "[EXIT_CHRONO_SPLIT] decision_ts has unparseable/empty values — cannot order trades by "
            "time for an honest OOT split."
        )
    g = np.asarray(groups)
    uniq, inv = np.unique(g, return_inverse=True)   # inv: row -> trade index into uniq
    n_trades = len(uniq)
    if n_trades < n_folds + 2:
        raise RuntimeError(
            f"[EXIT_CHRONO_SPLIT] only {n_trades} trades for {n_folds}+2 blocks — too few for an honest split."
        )
    # Per-trade decision time = min ts within the trade (decision_ts_utc is constant per trade;
    # min is robust to a stray differing row).
    trade_ts = np.full(n_trades, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(trade_ts, inv, ts)
    trade_order = np.argsort(trade_ts, kind="mergesort")      # chronological trade order
    blocks = np.array_split(trade_order, n_folds + 2)         # contiguous time-blocks of TRADE indices
    emb = max(0, int(embargo_ns))
    folds: list[dict[str, np.ndarray]] = []
    for k in range(n_folds):
        train_tr = np.concatenate(blocks[: k + 1])
        val_tr = blocks[k + 1]
        test_tr = blocks[k + 2]
        # Time-embargo: purge train trades decided within `emb` of the first val decision, and val
        # trades within `emb` of the first test decision, so no forward bar-window straddles a split.
        if emb and len(val_tr) and len(train_tr):
            val_start = int(trade_ts[val_tr].min())
            train_tr = train_tr[trade_ts[train_tr] <= (val_start - emb)]
        if emb and len(test_tr) and len(val_tr):
            test_start = int(trade_ts[test_tr].min())
            val_tr = val_tr[trade_ts[val_tr] <= (test_start - emb)]
        # Map trade indices -> ORIGINAL row indices (every bar of each trade lands together).
        train_idx = np.nonzero(np.isin(inv, train_tr))[0]
        val_idx = np.nonzero(np.isin(inv, val_tr))[0]
        test_idx = np.nonzero(np.isin(inv, test_tr))[0]
        folds.append({
            "fold_id_v1": f"FOLD_{k + 1}",
            "train_idx_v1": np.sort(train_idx),
            "val_idx_v1": np.sort(val_idx),
            "test_idx_v1": np.sort(test_idx),
        })
    return folds


def resolve_exit_folds(
    df: "pd.DataFrame", *, n_rows: int, oracle_action: np.ndarray, n_folds: int = N_FOLDS,
) -> list[dict[str, np.ndarray]]:
    """ONE-truth split resolver for ALL exit-IQL call sites. GX1_EXIT_IQL_SPLIT_MODE:
      - 'stratified' (DEFAULT, cement bit-parity): StratifiedShuffleSplit by oracle_action.
      - 'group' (EXIT-8): GroupShuffleSplit by candidate_uid (leak-free, but random-in-time).
      - 'chronological' (R16): group-aware purged walk-forward by decision_ts_utc + candidate_uid
        + embargo — true OOT, parity with entry-IQL build_chronological_folds.
    Back-compat: GX1_EXIT_IQL_GROUP_SPLIT=1 selects 'group' when SPLIT_MODE is unset. Fail-closed:
    'chronological'/'group' RAISE if their required columns are absent (never silently leak).
    """
    mode = os.environ.get("GX1_EXIT_IQL_SPLIT_MODE", "").strip().lower()
    if not mode:
        mode = "group" if os.environ.get("GX1_EXIT_IQL_GROUP_SPLIT", "0").strip().lower() in ("1", "true", "yes", "on") else "stratified"
    if mode == "chronological":
        for col in ("decision_ts_utc", "candidate_uid"):
            if col not in df.columns:
                raise RuntimeError(
                    f"[EXIT_SPLIT] GX1_EXIT_IQL_SPLIT_MODE=chronological needs '{col}' in the per-bar df — "
                    "cannot do an honest OOT split."
                )
        folds = build_chronological_group_folds(
            decision_ts=df["decision_ts_utc"].to_numpy(),
            groups=df["candidate_uid"].to_numpy(),
            n_folds=n_folds,
        )
        print(f"[EXIT_SPLIT] mode=chronological (decision_ts_utc + candidate_uid + {EXIT_EMBARGO_NS/3.6e12:.0f}h embargo) folds={len(folds)}", flush=True)
        return folds
    if mode == "group":
        if "candidate_uid" not in df.columns:
            raise RuntimeError("[EXIT_SPLIT] GX1_EXIT_IQL_SPLIT_MODE=group needs 'candidate_uid' in the per-bar df.")
        print("[EXIT_SPLIT] mode=group (GroupShuffleSplit by candidate_uid)", flush=True)
        return build_stratified_folds(n_rows=n_rows, oracle_action=oracle_action, n_folds=n_folds, groups=df["candidate_uid"].to_numpy())
    print("[EXIT_SPLIT] mode=stratified (cement StratifiedShuffleSplit, bit-parity)", flush=True)
    return build_stratified_folds(n_rows=n_rows, oracle_action=oracle_action, n_folds=n_folds, groups=None)


def maybe_per_trade_sample_weights(df: "pd.DataFrame") -> np.ndarray | None:
    """EXIT-8 part 2: per-trade 1/n_bars sample weight (mean-normalised to 1.0).

    A long-held trade contributes ~480 near-duplicate M1 bars to the per-bar MSE and would
    otherwise dominate short trades (survivor-length bias). Weighting each bar by 1/n_bars of
    its trade equalises every trade's total contribution. Returns None unless
    GX1_EXIT_IQL_PER_TRADE_WEIGHT is truthy (= cement unweighted). Mean-normalised so the loss
    magnitude / effective LR is unchanged. Fail-closed if flag set but candidate_uid missing.
    """
    if os.environ.get("GX1_EXIT_IQL_PER_TRADE_WEIGHT", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    if "candidate_uid" not in df.columns:
        raise RuntimeError(
            "GX1_EXIT_IQL_PER_TRADE_WEIGHT=1 but 'candidate_uid' is absent from the per-bar df — "
            "cannot compute a per-trade weight; fix the dataset or unset the flag."
        )
    counts = df.groupby("candidate_uid")["candidate_uid"].transform("size").to_numpy().astype(np.float64)
    w = 1.0 / np.maximum(1.0, counts)
    w = w * (float(len(w)) / float(w.sum()))   # mean-normalise to 1.0
    return w.astype(np.float32)


def maybe_degenerate_loss_mask(
    df: "pd.DataFrame", n_actions: int, k_horizons: "list[int]",
) -> np.ndarray | None:
    """EXIT-9: (n, n_actions*n_K) 0/1 Q-loss mask, or None.

    Zeroes the HOLD-action (row, K) cells whose forward window is clipped past the trade/data
    end (forward_bars_remaining_v1 < K). For those cells r_hold_K is computed on K_eff<K bars,
    so it is NOT a real K-horizon continuation reward and should not push gradient. EXIT_NOW
    cells stay 1 (current-unrealized-PnL is horizon-independent, always valid). Returns None
    unless GX1_EXIT_IQL_MASK_DEGENERATE is truthy (= cement, no masking). Fail-closed on a
    missing forward_bars_remaining_v1 column. Flatten order (action-major) matches R.view(n,-1).
    """
    if os.environ.get("GX1_EXIT_IQL_MASK_DEGENERATE", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return None
    if "forward_bars_remaining_v1" not in df.columns:
        raise RuntimeError(
            "GX1_EXIT_IQL_MASK_DEGENERATE=1 but 'forward_bars_remaining_v1' is absent from the "
            "per-bar df — cannot mask degenerate K-cells; fix the dataset or unset the flag."
        )
    n = len(df)
    nk = len(k_horizons)
    fbr = df["forward_bars_remaining_v1"].to_numpy().astype(np.float64)
    mask = np.ones((n, n_actions, nk), dtype=np.float32)
    for ki, K in enumerate(k_horizons):
        mask[fbr < float(K), ACTION_HOLD_ID, ki] = 0.0   # HOLD K-horizon clipped past trade end
    return mask.reshape(n, n_actions * nk)


def evaluate_policy_outcome(R: np.ndarray, predicted_action: np.ndarray, K_idx: int) -> dict[str, Any]:
    n = R.shape[0]
    per_row = R[np.arange(n), predicted_action, K_idx]
    counts = {f"n_action_{ai}_v1": int((predicted_action == ai).sum()) for ai in range(N_ACTIONS_EXIT)}
    counts["n_total_v1"] = int(n)
    fractions = {f"frac_action_{ai}_v1": float((predicted_action == ai).mean()) for ai in range(N_ACTIONS_EXIT)}
    return {
        "total_reward_bps_v1": float(per_row.sum()),
        "mean_reward_bps_v1": float(per_row.mean()) if n > 0 else 0.0,
        "counts_v1": counts, "fractions_v1": fractions,
        "min_action_frac_v1": float(min(fractions.values())) if fractions else 0.0,
    }


def evaluate_baselines(R: np.ndarray, K_idx: int) -> dict[str, dict[str, Any]]:
    n = R.shape[0]
    out: dict[str, dict[str, Any]] = {}
    out["always_hold_v1"] = evaluate_policy_outcome(R, np.zeros(n, dtype=np.int64), K_idx)
    out["always_exit_v1"] = evaluate_policy_outcome(R, np.ones(n, dtype=np.int64), K_idx)
    rng = np.random.default_rng(SEED_V1)
    out["random_uniform_v1"] = evaluate_policy_outcome(R, rng.integers(0, N_ACTIONS_EXIT, size=n), K_idx)
    out["oracle_v1"] = evaluate_policy_outcome(R, R[:, :, K_idx].argmax(axis=1), K_idx)
    return out


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------


def evaluate_one_fold(
    fold: dict[str, np.ndarray],
    X: np.ndarray, R: np.ndarray, oracle_action: np.ndarray,
    *,
    variant: str, artifact_root: Path,
    sample_weights: np.ndarray | None = None,   # EXIT-8 part 2: per-row weights (full-dataset)
    loss_mask: np.ndarray | None = None,        # EXIT-9: (n, n_actions*n_K) full-dataset Q-loss mask
) -> dict[str, Any]:
    fold_id_v1 = fold["fold_id_v1"]
    train_idx = fold["train_idx_v1"]
    val_idx = fold["val_idx_v1"]
    test_idx = fold["test_idx_v1"]
    X_train, R_train = X[train_idx], R[train_idx]
    X_val, R_val = X[val_idx], R[val_idx]
    X_test, R_test = X[test_idx], R[test_idx]
    # EXIT-8 part 2 / EXIT-9: subset the per-row weights + Q-loss mask to this fold's train rows.
    train_sample_weights = sample_weights[train_idx] if sample_weights is not None else None
    train_loss_mask = loss_mask[train_idx] if loss_mask is not None else None

    val_action_counts = {ai: int((oracle_action[val_idx] == ai).sum()) for ai in range(N_ACTIONS_EXIT)}
    val_class_guard_status = "OK" if min(val_action_counts.values()) >= MIN_PER_CLASS_VAL else "WARN_BELOW_MIN_PER_CLASS"

    print(f"[{ACTION}] {fold_id_v1} {variant} train={len(X_train)} val={len(X_val)} test={len(X_test)} "
          f"val_action_counts={val_action_counts} guard={val_class_guard_status}", flush=True)

    model = iql_core.train_multi_head_entry_iql(
        X_train, R_train,
        k_horizons=K_HORIZONS,
        n_actions=N_ACTIONS_EXIT,
        tau=TRAIN_TAU,
        epochs_q=TRAIN_EPOCHS_Q, epochs_v=TRAIN_EPOCHS_V,
        k_iterations=TRAIN_K_VQ_ITERATIONS,
        batch_size=TRAIN_BATCH_SIZE,
        hidden_dim=TRAIN_HIDDEN_DIM, n_hidden=TRAIN_N_HIDDEN,
        seed=SEED_V1, prefer_cuda=True,
        sample_weights=train_sample_weights,
        loss_mask=train_loss_mask,
    )

    K_primary_idx = K_HORIZONS.index(K_PRIMARY)
    eval_rows: list[dict[str, Any]] = []
    best_val_total = -np.inf
    best_combo: dict[str, Any] | None = None
    best_test_actions = None

    for agg in AGGREGATORS:
        val_actions = iql_core.policy_argmax_action(X_val, model, aggregator=agg)
        test_actions = iql_core.policy_argmax_action(X_test, model, aggregator=agg)
        for ki, K in enumerate(K_HORIZONS):
            val_metric = evaluate_policy_outcome(R_val, val_actions, ki)
            test_metric = evaluate_policy_outcome(R_test, test_actions, ki)
            eval_rows.append({
                "fold_id_v1": fold_id_v1, "variant_v1": variant,
                "aggregator_v1": agg, "k_horizon_v1": K,
                "val_total_reward_bps_v1": val_metric["total_reward_bps_v1"],
                "val_min_action_frac_v1": val_metric["min_action_frac_v1"],
                "val_action_fractions_v1": val_metric["fractions_v1"],
                "test_total_reward_bps_v1": test_metric["total_reward_bps_v1"],
                "test_min_action_frac_v1": test_metric["min_action_frac_v1"],
                "test_action_fractions_v1": test_metric["fractions_v1"],
            })
            if K == K_PRIMARY and val_metric["total_reward_bps_v1"] > best_val_total:
                best_val_total = val_metric["total_reward_bps_v1"]
                best_combo = {"aggregator_v1": agg, "k_horizon_v1": K}
                best_test_actions = test_actions

    test_baselines = evaluate_baselines(R_test, K_primary_idx)
    val_baselines = evaluate_baselines(R_val, K_primary_idx)

    model_dir = artifact_root / "trained_models_v1"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{variant}_{fold_id_v1}.pt"
    torch.save({
        "q_state_dict": model.q_net.state_dict(),
        "v_state_dict": model.v_net.state_dict(),
        "feature_means": model.feature_means.tolist(),
        "feature_stds": model.feature_stds.tolist(),
        "k_horizons": K_HORIZONS,
        "n_actions": N_ACTIONS_EXIT,
        "state_dim": model.state_dim,
        "variant": variant, "fold_id": fold_id_v1,
        "hidden_dim": TRAIN_HIDDEN_DIM, "n_hidden": TRAIN_N_HIDDEN,
        "dropout": iql_core.DEFAULT_DROPOUT,
        "schema_v1": "MULTI_HEAD_EXIT_IQL_V2_CHECKPOINT",
        "action_labels": ACTION_LABELS_EXIT,
    }, model_path)

    return {
        "fold_id_v1": fold_id_v1, "variant_v1": variant,
        "best_combo_v1": best_combo,
        "best_val_total_reward_bps_v1": float(best_val_total) if np.isfinite(best_val_total) else None,
        "best_test_metric_v1": (evaluate_policy_outcome(R_test, best_test_actions, K_primary_idx)
                                 if best_test_actions is not None else None),
        "val_class_guard_status_v1": val_class_guard_status,
        "val_action_counts_v1": val_action_counts,
        "val_baselines_v1": val_baselines, "test_baselines_v1": test_baselines,
        "all_evaluations_v1": eval_rows, "model_path_v1": str(model_path),
    }


def determine_status(per_fold_results: list[dict[str, Any]]) -> tuple[str, str, str, dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in per_fold_results:
        by_variant.setdefault(r["variant_v1"], []).append(r)
    best_variant = None
    best_mean_test = -np.inf
    variant_summaries: dict[str, dict[str, Any]] = {}
    for variant, results in by_variant.items():
        test_totals = [r["best_test_metric_v1"]["total_reward_bps_v1"] for r in results if r.get("best_test_metric_v1")]
        oracle_totals = [r["test_baselines_v1"]["oracle_v1"]["total_reward_bps_v1"] for r in results if r.get("test_baselines_v1")]
        hold_baseline_totals = [r["test_baselines_v1"]["always_hold_v1"]["total_reward_bps_v1"] for r in results if r.get("test_baselines_v1")]
        min_action_fracs = [r["best_test_metric_v1"]["min_action_frac_v1"] for r in results if r.get("best_test_metric_v1")]
        mean_test = float(np.mean(test_totals)) if test_totals else -np.inf
        mean_oracle = float(np.mean(oracle_totals)) if oracle_totals else 0.0
        mean_hold = float(np.mean(hold_baseline_totals)) if hold_baseline_totals else 0.0
        min_min_frac = float(np.min(min_action_fracs)) if min_action_fracs else 0.0
        variant_summaries[variant] = {
            "mean_test_reward_bps_v1": mean_test,
            "mean_oracle_reward_bps_v1": mean_oracle,
            "mean_always_hold_baseline_v1": mean_hold,
            "fraction_of_oracle_v1": (mean_test / mean_oracle) if mean_oracle != 0 else None,
            "min_action_frac_across_folds_v1": min_min_frac,
        }
        if mean_test > best_mean_test:
            best_mean_test = mean_test
            best_variant = variant
    headline = {
        "n_folds_v1": max(len(v) for v in by_variant.values()) if by_variant else 0,
        "n_variants_v1": len(by_variant),
        "best_variant_v1": best_variant,
        "best_mean_test_reward_bps_v1": float(best_mean_test) if np.isfinite(best_mean_test) else None,
        "variant_summaries_v1": variant_summaries,
    }
    best_summary = variant_summaries.get(best_variant or "", {})
    min_frac = best_summary.get("min_action_frac_across_folds_v1", 0.0)
    hold_baseline = best_summary.get("mean_always_hold_baseline_v1", 0.0)
    delta_hold = best_mean_test - hold_baseline
    if min_frac < 0.001:
        return ("EXIT_IQL_V2_PARTIAL_DEGENERATE_ACTION_DIST",
                "REPAIR_EXIT_IQL_V2_BEFORE_FURTHER_WORK",
                f"Best {best_variant} action distribution degenerate (min frac {min_frac:.4f})", headline)
    if delta_hold > 200.0:
        return ("EXIT_IQL_V2_PASS_BEATS_BASELINE",
                "BUILD_EXIT_IQL_V2_RUNTIME_ADAPTER",
                f"Best {best_variant} beats always-hold baseline by {delta_hold:+.0f} bps", headline)
    if abs(delta_hold) <= 200.0:
        return ("EXIT_IQL_V2_PARTIAL_TIES_BASELINE",
                "REPAIR_EXIT_IQL_V2_BEFORE_FURTHER_WORK",
                f"Best {best_variant} ties always-hold (delta {delta_hold:+.0f} bps)", headline)
    return ("EXIT_IQL_V2_PARTIAL_DEGRADES_VS_BASELINE",
            "REPAIR_EXIT_IQL_V2_BEFORE_FURTHER_WORK",
            f"Best {best_variant} underperforms always-hold by {-delta_hold:+.0f} bps", headline)


def write_artifacts(
    per_bar_dir: Path = DEFAULT_PER_BAR_DIR,
    out_root: Path | None = None,
    *,
    sample_n_rows: int | None = None,
    variants_subset: list[str] | None = None,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    per_week_dir = per_bar_dir / "per_week"
    if not per_week_dir.exists():
        return {
            "artifact_root": str(artifact_root),
            "summary": {
                "final_status_v1": "EXIT_IQL_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
                "next_action_v1": "REPAIR_EXIT_IQL_V2_BEFORE_FURTHER_WORK",
                "missing_input_v1": str(per_week_dir),
            },
        }

    # V2: load pre-built state + reward cache via env GX1_EXIT_IQL_V2_STATE_CACHE.
    # Cache built by scripts/prep_exit_iql_v2_state_cache.py with per-trade stratified sampling.
    import os as _os
    _state_cache_dir = _os.environ.get("GX1_EXIT_IQL_V2_STATE_CACHE", "").strip()
    if _state_cache_dir:
        from pathlib import Path as _P
        cache_dir = _P(_state_cache_dir)
        if not (cache_dir / "state_matrix.npy").is_file():
            raise FileNotFoundError(f"[{ACTION}] state-cache missing files: {cache_dir}")
        print(f"[{ACTION}] LOAD_STATE_CACHE: {cache_dir}", flush=True)
        X = np.load(cache_dir / "state_matrix.npy", mmap_mode="r").astype(np.float32, copy=False)
        feature_names = json.loads((cache_dir / "feature_names.json").read_text())
        rew_npz = np.load(cache_dir / "reward_matrices.npz")
        R_by_variant = {k: rew_npz[k] for k in rew_npz.files}
        cu_arr = np.load(cache_dir / "candidate_uid.npy", allow_pickle=True)
        df = pd.DataFrame({"candidate_uid": cu_arr})
        ts_path = cache_dir / "decision_ts_utc.npy"
        if ts_path.is_file():
            ts_ns = np.load(ts_path)
            df["decision_ts_utc"] = pd.to_datetime(ts_ns, utc=True)
        print(f"[{ACTION}] cache loaded: X={X.shape} features={len(feature_names)} "
              f"variants={list(R_by_variant.keys())}", flush=True)
        for variant in R_by_variant:
            print(f"[{ACTION}] cached reward {variant}: shape={R_by_variant[variant].shape} "
                  f"mean={float(R_by_variant[variant].mean()):.3f}", flush=True)
    else:
        print(f"[{ACTION}] loading per-bar exit dataset from {per_week_dir}", flush=True)
        df = load_per_bar_dataset(per_week_dir, sample_n_rows=sample_n_rows)
        print(f"[{ACTION}] loaded rows={len(df):,} cols={len(df.columns)}", flush=True)

        X, feature_names = build_state_matrix(df)
        print(f"[{ACTION}] state matrix: {X.shape}, features={len(feature_names)}", flush=True)

        R_by_variant = {}
        for variant in (variants_subset or REWARD_VARIANTS):
            R_by_variant[variant] = build_reward_matrix(df, variant=variant)
            print(f"[{ACTION}] reward matrix {variant}: shape={R_by_variant[variant].shape} "
                  f"mean={float(R_by_variant[variant].mean()):.3f}", flush=True)

    R_for_strat = R_by_variant.get("R_NET_REAL", next(iter(R_by_variant.values())))
    K_primary_idx = K_HORIZONS.index(K_PRIMARY)
    oracle_action = derive_oracle_action(R_for_strat, K_primary_idx)
    oracle_dist = {ai: int((oracle_action == ai).sum()) for ai in range(N_ACTIONS_EXIT)}
    print(f"[{ACTION}] oracle action (R_NET_REAL K={K_PRIMARY}) distribution: {oracle_dist}", flush=True)

    folds = build_stratified_folds(
        n_rows=len(df), oracle_action=oracle_action,
        groups=maybe_candidate_uid_groups(df),  # EXIT-8: leak-free group split when enabled
    )

    per_fold_results: list[dict[str, Any]] = []
    flat_evaluations: list[dict[str, Any]] = []
    _exit_sample_weights = maybe_per_trade_sample_weights(df)  # EXIT-8 part 2 (None = cement)
    _exit_loss_mask = maybe_degenerate_loss_mask(df, N_ACTIONS_EXIT, list(K_HORIZONS))  # EXIT-9 (None = cement)
    for variant in (variants_subset or REWARD_VARIANTS):
        for fold in folds:
            r = evaluate_one_fold(fold, X, R_by_variant[variant], oracle_action,
                                  variant=variant, artifact_root=artifact_root,
                                  sample_weights=_exit_sample_weights,
                                  loss_mask=_exit_loss_mask)
            per_fold_results.append(r)
            flat_evaluations.extend(r.get("all_evaluations_v1", []))

    _write_rows(artifact_root / "per_fold_per_variant_evaluations_v1.csv", flat_evaluations)
    status, next_action, recommendation, headline = determine_status(per_fold_results)

    summary = {
        "layer_name": "EXIT_IQL_V2_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "input_per_bar_dir_v1": str(per_bar_dir),
        "n_rows_v1": int(len(df)),
        "n_features_v1": int(X.shape[1]),
        "feature_names_v1": feature_names,
        "k_horizons_v1": K_HORIZONS,
        "k_primary_v1": K_PRIMARY,
        "n_actions_v1": N_ACTIONS_EXIT,
        "action_labels_v1": ACTION_LABELS_EXIT,
        "reward_variants_v1": list(R_by_variant.keys()),
        "oracle_action_distribution_v1": oracle_dist,
        "n_folds_v1": N_FOLDS,
        "seed_v1": SEED_V1,
        "min_per_class_val_v1": MIN_PER_CLASS_VAL,
        "per_fold_summary_v1": [
            {"fold_id_v1": r["fold_id_v1"], "variant_v1": r["variant_v1"],
             "best_combo_v1": r["best_combo_v1"], "best_test_metric_v1": r["best_test_metric_v1"],
             "val_class_guard_status_v1": r["val_class_guard_status_v1"],
             "val_action_counts_v1": r["val_action_counts_v1"],
             "test_baselines_v1": r["test_baselines_v1"]}
            for r in per_fold_results
        ],
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(artifact_root / "status_v1.json", {
        "layer_name": "EXIT_IQL_V2_STATUS_V1",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status, "next_action_v1": next_action,
    })
    _write_json(artifact_root / "manifest_v1.json", {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "per_fold_csv": str(artifact_root / "per_fold_per_variant_evaluations_v1.csv"),
            "trained_models_dir": str(artifact_root / "trained_models_v1"),
        },
    })
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    global TRAIN_EPOCHS_Q, TRAIN_EPOCHS_V, TRAIN_K_VQ_ITERATIONS, TRAIN_BATCH_SIZE, TRAIN_HIDDEN_DIM, TRAIN_N_HIDDEN
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION}.")
    parser.add_argument("--per-bar-dir", type=str, default=str(DEFAULT_PER_BAR_DIR))
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--sample-n-rows", type=int, default=None)
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--budget", type=str, default="fast", choices=list(BUDGET_PRESETS.keys()))
    parser.add_argument("--built-at-utc", type=str, default=None)
    parser.add_argument(
        "--vedtak", type=str, default=None,
        help="Explicit retrain decision id — rule 3, NEVER auto-retrain (gx1_guards fail-closed).",
    )
    args = parser.parse_args()
    # NEVER auto-retrain — fail-closed unless an explicit --vedtak is passed (CLAUDE.md rule 3).
    from gx1_guards.gates import require_retrain_vedtak, GateError
    try:
        require_retrain_vedtak(args.vedtak)
    except GateError as e:
        parser.error(str(e))

    preset = BUDGET_PRESETS[args.budget]
    TRAIN_EPOCHS_Q = preset["epochs_q"]
    TRAIN_EPOCHS_V = preset["epochs_v"]
    TRAIN_K_VQ_ITERATIONS = preset["k_iter"]
    TRAIN_BATCH_SIZE = preset["batch"]
    TRAIN_HIDDEN_DIM = preset["hidden"]
    TRAIN_N_HIDDEN = preset["n_hidden"]
    print(f"[{ACTION}] training budget '{args.budget}': {preset}", flush=True)

    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else None
    variants_subset = args.variants.split(",") if args.variants else None
    result = write_artifacts(
        per_bar_dir=Path(args.per_bar_dir).expanduser().resolve(),
        out_root=out_root,
        sample_n_rows=args.sample_n_rows,
        variants_subset=variants_subset,
        built_at_utc=args.built_at_utc,
    )
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
