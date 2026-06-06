#!/usr/bin/env python3
"""Build entry-IQL V2: 5-action multi-head Q(s, a, K) on full 2020-2026 candidate universe.

Mission
-------
Train the RL "smart head" that supervises and overrides the entry transformer
(V10) — same architectural pattern as exit-IQL guides exit-transformer (V3).

Uses the candidate forward-outcome dataset built by
`materialize_build_candidate_forward_outcome_dataset_v1.py` (~844k candidate
moments across 282 weekly XAUUSD replays, with side-aware spread-aware
forward-bar metrics for take-now and wait-6-bars actions, at 6 K-horizons:
[12, 24, 48, 96, 144, 192] M5-bars = [1h, 2h, 4h, 8h, 12h, 16h]).

Action space (5)
----------------
  SKIP, TAKE_LONG_NOW, TAKE_SHORT_NOW, WAIT_LONG, WAIT_SHORT

Reward variants (5, all trained in parallel for comparison)
-----------------------------------------------------------
  R_RAW    : mfe_at_K  (pure upside)
  R_NET05  : mfe_at_K - 0.5 * mae_before_mfe_at_K  (mild risk-adjustment)
  R_NET10  : mfe_at_K - 1.0 * mae_before_mfe_at_K  (strong risk-adjustment)
  R_GATED  : mfe_at_K * 1[mae_before_mfe_at_K < 30bps]  (binary realisability)
  R_PATH   : mfe_at_K * mfe_at_K / max(mfe_at_K, |mae_before_mfe_at_K|)
              (smooth path-quality)

For SKIP, reward = 0 in all variants (no trade taken, no upside, no downside).

Counterfactual setup
--------------------
For each candidate state, all 5 action rewards are observed (computed from
forward bars). This is fully-observable counterfactual bandit. The Q-network
learns to predict R[s, a, K] for all (a, K) cells — much richer signal than
standard partial-observation bandit.

Folds
-----
StratifiedShuffleSplit by oracle_action (= argmax_a of R_PRIMARY at K_PRIMARY)
with n_splits=3 and test_size=0.20. Per-fold per-class guards: each fold's
val partition must have >= MIN_PER_CLASS_VAL examples of each action label,
else fold is FLAGGED but training continues.

Output
------
  per_fold_per_variant_evaluations_v1.csv  — every (fold, variant, agg, K) row
  summary_v1.json                          — go/no-go status, headlines
  status_v1.json                           — single-line status for downstream
  manifest_v1.json                         — artifact paths
  trained_models_v1/<variant>_fold<i>.pt   — model checkpoints

This is RESEARCH-ONLY. No runtime promotion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sklearn.model_selection import StratifiedShuffleSplit

from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core
from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import materialize_build_candidate_forward_outcome_dataset_v1 as fwd_pipe


ACTION = "BUILD_ENTRY_IQL_V2"
DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
DEFAULT_FORWARD_OUTCOME_DIR = (
    DEFAULT_REPORTS_ROOT / "CANDIDATE_FORWARD_OUTCOME_RUN" / "v1_full"
)

K_HORIZONS = [12, 24, 48, 96, 144, 192]
N_K = len(K_HORIZONS)
K_PRIMARY = 96  # used to derive oracle_action_label_v1 for stratification

# NOTE: R_RAW, R_NET05, R_NET10, R_GATED, R_PATH removed 2026-05-02 — they all
# degenerate to "never SKIP" because their rewards are based on raw MFE / mild
# MAE-penalties, which never let SKIP=0 win against trade-action MFE > 0.
# Verified empirically: 5 of these 6 variants had min_action_frac = 0.0000 in
# the V3 training run. Only R_NET_REAL (alpha=0.5 MFE realisation + beta=0.5
# MAE penalty + gamma=2.0 spread cost) produces an action distribution where
# SKIP can win — the realistic production-deployable policy.
REWARD_VARIANTS = [
    "R_NET_REAL",
    "R_TERMINAL_K24",
    "R_TERMINAL_K96",
    # R_PATIENT_K96 = terminal_pnl@K96 - lambda * mae_before_mfe@K96 - 2*spread.
    # Linear MAE-pre penalty. Effect was small (~15% MAE reduction) — too gentle.
    "R_PATIENT_K96_LAM03",
    "R_PATIENT_K96_LAM05",
    "R_PATIENT_K96_LAM10",
    # R_CLEAN_K96 = binary threshold: full reward if MAE_pre < TOL else 0.
    # Forces Entry-IQL to learn "wait until the dip is done, then take".
    "R_CLEAN_K96_TOL10",
    "R_CLEAN_K96_TOL20",
    "R_CLEAN_K96_TOL40",
    # R_QUAD_K96 = quadratic MAE penalty: small MAE no effect, large MAE bombs.
    "R_QUAD_K96_LAM05",
    "R_QUAD_K96_LAM10",
    # R_QUALITY_K96 = scale-invariant entry-quality ratio.
    # r = terminal_pnl * max(0, 1 - MAE_pre/MFE) - 2*spread
    # Self-balancing: no tuning, the ratio MAE/MFE itself is the "entry cleanness" signal.
    "R_QUALITY_K96",
    # R_SOFT_K96 = sigmoid-binary (smoother than hard threshold).
    # r = terminal_pnl * sigmoid((TOL - MAE_pre) / scale) - 2*spread
    "R_SOFT_K96_TOL20",
    "R_SOFT_K96_TOL40",
    # R_ASYMMETRIC_K96 = logarithmic MAE penalty (gentle for small MAE, firm for large).
    # r = terminal_pnl - λ * log(1 + MAE_pre) - 2*spread
    "R_ASYMMETRIC_K96_LAM05",
    "R_ASYMMETRIC_K96_LAM10",
    # R_WAIT_OPP_K96 — uses counterfactual wait_long/wait_short_terminal_pnl@K96
    # to explicitly reward SKIP when waiting beats taking now.
    # r_long  = take_now_long_terminal_pnl@K96 - λ*MAE_pre - 2*spread
    # r_short = take_now_short_terminal_pnl@K96 - λ*MAE_pre - 2*spread
    # r_skip  = max(0, max(wait_long, wait_short) - max(r_long, r_short))
    # Goal: teach Entry-IQL "vent til dippen er ferdig" via SKIP > TAKE signal.
    "R_WAIT_OPP_K96_LAM05",
    "R_WAIT_OPP_K96_LAM10",
    # K48 variants: shorter wait horizon (more responsive). wait_K48 had +40 mean signal.
    "R_WAIT_OPP_K48_LAM05",
    "R_WAIT_OPP_K48_LAM10",
    "R_WAIT_OPP_K96_LAM20",  # stronger MAE penalty
    # 2026-05-24: aggressive λ — push MAE lower at cost of fewer trades
    "R_WAIT_OPP_K96_LAM30",
    "R_WAIT_OPP_K96_LAM50",
    # 2026-06-03 (vedtak entry_iql_perside_reward_20260603): _SYM = symmetric r_skip
    # (waited side penalized by its OWN MAE+spread). Fixes the ~98%-skip / per-side
    # anti-calibration. Candidate active variants for the regime-robust retrain.
    "R_WAIT_OPP_K96_LAM30_SYM",
    "R_WAIT_OPP_K96_LAM50_SYM",
    # R_HYBRID = R_WAIT_OPP + binary clean threshold (max selective)
    "R_HYBRID_K96_TOL20",
    "R_HYBRID_K96_TOL40",
    # R_V10_PQ_COND_K96 (2026-06-01, vedtak r_v10_pq_cond_design_20260601) —
    # V10 path-quality conditioned λ. Designed after live missed a 139-bps short
    # day where cement LAM50 (λ=5 fast) skipped every poll. Insight: V10's
    # path_quality_pred is already informative; use it to modulate the MAE
    # penalty. Clean path → small penalty (allow take), messy path → large
    # penalty (force skip). Encodes regime-awareness without a separate detector.
    #   λ_eff(s) = 8.0 - 14.0 * clip(v10_path_quality_pred, 0, 0.5)
    #   pq=0.0 → λ=8.0 (stricter than LAM50)
    #   pq=0.3 → λ=3.8 (between LAM30 and LAM50)
    #   pq=0.5+ → λ=1.0 (very tolerant, like LAM10)
    # r_long  = take_now_long_terminal_pnl@K96  - λ_eff * MAE_pre - 2*spread
    # r_short = take_now_short_terminal_pnl@K96 - λ_eff * MAE_pre - 2*spread
    # r_skip  = max(0, max(wait_long, wait_short) - max(r_long, r_short))
    "R_V10_PQ_COND_K96",
    # R_V10_MULTI_COND_K96 (2026-06-01, iteration after PQ_COND missed today's regime) —
    # Conditions λ on THREE signals at once: V10 path_quality + V10 mfe_pred + V10
    # tradable_prob. Insight from 2026-06-01: PQ_COND alone skipped today's 139-bps
    # short because V10 pq mean was only 0.38. But V10 ALSO predicted moderate
    # tradable_prob (0.84). The multi-signal cocktail captures more nuance:
    #   - High pq + high tradable + decent mfe_pred → very lenient λ
    #   - Any single signal weak → strict λ
    # λ_eff(s) = 8.0
    #          - 5.0 * clip(path_quality_pred, 0, 0.5)        # 0..2.5
    #          - 3.0 * clip(tradable_prob - 0.5, 0, 0.5)       # 0..1.5
    #          - 2.0 * clip(mfe_first_n_pred / 50.0, 0, 1.0)   # 0..2.0
    #   all-strong → λ ≈ 2.0 (very tolerant)
    #   mid → λ ≈ 5.0 (like LAM50)
    #   all-weak → λ ≈ 8.0 (stricter than cement)
    "R_V10_MULTI_COND_K96",
]
GATED_THRESHOLD_BPS = 30.0

# R_NET_REAL parameters: realistic-net reward.
# Exit policy realizes a FRACTION of MFE (alpha < 1), pays MAE-before-MFE penalty
# (beta), and pays entry+exit spread cost (gamma * entry_spread_bps).
# This makes SKIP competitive when forward path is choppy/costly.
R_NET_REAL_ALPHA_MFE = 0.5      # exit policy captures ~50% of peak MFE on average
R_NET_REAL_BETA_MAE = 0.5       # half-weight penalty on path-drawdown
R_NET_REAL_GAMMA_SPREAD = 2.0   # entry + exit each pay 1x spread

# Cross-validation
N_FOLDS = 3
VAL_SIZE = 0.15
TEST_SIZE = 0.15
SEED_V1 = 20260501

# Training budget — modulated by --budget flag. Default is "fast" so smoke-tests
# finish on CPU; override to "full" for production runs.
TRAIN_TAU = 0.8
TRAIN_EPOCHS_Q = 30
TRAIN_EPOCHS_V = 15
TRAIN_K_VQ_ITERATIONS = 3
TRAIN_BATCH_SIZE = 512
TRAIN_HIDDEN_DIM = 128
TRAIN_N_HIDDEN = 2

BUDGET_PRESETS = {
    "fast":   {"epochs_q":  30, "epochs_v": 15, "k_iter": 3, "batch": 512, "hidden": 128, "n_hidden": 2},
    "medium": {"epochs_q":  60, "epochs_v": 25, "k_iter": 4, "batch": 256, "hidden": 192, "n_hidden": 3},
    "full":   {"epochs_q": 100, "epochs_v": 40, "k_iter": 6, "batch": 256, "hidden": 256, "n_hidden": 3},
}

# Per-class assertion guards
MIN_PER_CLASS_VAL = 50  # warn (don't abort) if any action class has fewer than this

# Aggregator strategies for action selection (compared on val)
AGGREGATORS = ["mean", "max"]

# Numeric feature columns from candidate parquet (V10 outputs + state context).
# NOTE: entry_spread_bps and `side` are POST-decision fields (only populated for
# accepted candidates ~0.3% of rows), so they're EXCLUDED here. Use chunk_0
# `spread_bps` instead — at-bar pre-decision spread from raw bid/ask.
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
    # 2026-Q2 BIDIR additions: V10 v2 aux outputs needed for symmetric entry decisions.
    # `bad_path_prob` enables hard reward penalty when V10 says trade goes bad.
    # `direction_logit_*` are raw (pre-softmax) logits; complementary to the softmaxed
    # `p_long/p_short/p_flat` already in state — give the IQL access to V10's confidence
    # before normalization.
    "bad_path_prob",
    "direction_logit_long",
    "direction_logit_short",
    "direction_logit_flat",
    # 2026-05-27: dropped the old V10 v3+ aux-head cols (tf_agreement_pred /
    # path_quality_log_var / path_quality_std / position_size_pred / hold_horizon_pred)
    # — the COSTFIX V10 does NOT have those heads, so they'd be all-NaN and trip
    # build_state_matrix's >5%-NaN fail-closed abort. Replaced with COSTFIX V10's
    # actual new heads (full entry parity; base-named, carried via candidate->fwd-outcome).
    # 43 scalars: dip 18 / forecast 4 / timing 12 / tail_risk 6 / vol_forecast 3.
    *[f"v10_dip_{_i}" for _i in range(18)],
    *[f"v10_forecast_{_i}" for _i in range(4)],
    *[f"v10_timing_{_i}" for _i in range(12)],
    *[f"v10_tail_risk_{_i}" for _i in range(6)],
    *[f"v10_vol_forecast_{_i}" for _i in range(3)],
    # Portfolio features (improvement #1 2026-05-21) — simulated portfolio state
    # at decision_ts based on same-week sliding window. Makes Entry-IQL aware of
    # existing trades when scoring. Filled by forward-outcome builder.
    "portfolio_n_open_long_at_decision",
    "portfolio_n_open_short_at_decision",
    "portfolio_combined_pnl_bps_at_decision",
    "portfolio_time_since_last_entry_min",
]

# NUMERIC_STATE_COLS_CHUNK0 dropped 2026-05-21: chunk_0_data parquet was missing
# from training inputs all along, so every chunk0_v1 column was zero-filled
# during forward-outcome build. Feature-importance analysis confirmed all 55
# chunk0_v1 columns scored zero on XGB-gain AND permutation. Duplicates of the
# canon_v1 sister features already in the state vector — pure noise floor.
NUMERIC_STATE_COLS_CHUNK0: list[str] = []
# Canonical V10 features (basic_v1 + H1 + H4 + m5_phase + high-level), suffixed _canon_v1
# Names mirror canonical_features_v1.parquet columns.
NUMERIC_STATE_COLS_CANONICAL = [
    f"{c}{fwd_pipe.CANONICAL_FEATURES_SUFFIX}"
    for c in [
        # _v1_* basic_v1 features — canonical_v3 keeps the non-pruned subset.
        # 2026-05-21: 5 previously-dropped features RE-ADDED via PLUS5 augmenter
        # (_v1_vwap_drift48, _v1h1_vwap_drift, atr, std50, roc20).
        # 2026-05-21: 9 canon_v1 features pruned by feature-importance analysis
        # (constant/duplicated): _v1_atr_regime_id, _v1_bb_bandwidth_delta_10,
        # _v1_lower_tr, _v1_spread_p, _v1_slip_bps, _v1_spread_z,
        # _v1_cost_bps_est, _v1_comp3_ratio, smc_choch.
        "_v1_r3", "_v1_r5", "_v1_r8", "_v1_r12", "_v1_r24", "_v1_r1", "_v1_r48_z",
        "_v1_atr14", "_v1_pk_sigma20", "_v1_ema_diff",
        "_v1_rsi14_z", "_v1_rsi2", "_v1_rsi14",
        "_v1_rsi2_gt_rsi14", "_v1_ret_ema_diff_2_5",
        "_v1_close_ema_slope_3", "_v1_upper_tr",
        "_v1_wick_imbalance", "_v1_range_comp_20_100", "_v1_range_adr",
        "_v1_clv", "_v1_range_z",
        "_v1_kurt_r", "_v1_int_vwap_h1",
        "_v1_cost_bps_dyn", "_v1_tod_sin", "_v1_tod_cos",
        "_v1_r1_q90_48", "_v1_r1_q10_48", "_v1_atr_z_10_100", "_v1_tema_slope_20",
        "_v1_bb_squeeze_20_2", "_v1_kama_slope_30", "_v1_ret_ema_ratio_5_34",
        "_v1_body_share_1", "_v1_tr_1_over_atr_14",
        # PLUS5 2026-05-21: M5 48-period VWAP drift (mean-reversion signal)
        "_v1_vwap_drift48",
        # H1 multi-TF features
        "_v1h1_ema_diff", "_v1h1_atr", "_v1h1_rsi14_z",
        "_v1h1_slope3", "_v1h1_slope5",
        # PLUS5 2026-05-21: H1 24-period VWAP drift
        "_v1h1_vwap_drift",
        # 5 H4 multi-TF features
        "_v1h4_ema_diff", "_v1h4_atr", "_v1h4_rsi14_z", "_v1h4_slope3", "_v1h4_slope5",
        # 5 m5_phase one-hots
        "m5_phase_0", "m5_phase_1", "m5_phase_2", "m5_phase_3", "m5_phase_4",
        # high-level basics
        "mid", "range", "body_pct", "wick_asym", "atr50", "atr_z",
        "ret_1", "ret_5", "ret_20", "roc100", "rvol_20", "rvol_60",
        "ema20_slope", "ema100_slope", "pos_vs_ema200", "vol_ratio",
        # PLUS5 2026-05-21: raw atr + std50 + roc20 — momentum/vol features
        "atr", "std50", "roc20",
        # 8 SMC features (smc_choch pruned 2026-05-21 — constant in canonical_v3).
        "smc_swing_state", "smc_bos_up", "smc_bos_down",
        "smc_sweep_up", "smc_sweep_down", "smc_sweep_size_atr",
        "smc_bars_since_sweep", "smc_premium_discount",
        # 5 NEW canonical_v3 features (cyclic time + SMC×swing interaction)
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "smc_premium_state",
    ]
]
# Derived V3-equivalent feature
NUMERIC_STATE_COLS_DERIVED = ["entropy_v1"]

# 2026-05-24 DIP-CONFIRMED features (computed in build_state_matrix from existing cols).
# Captures "we are near a recent low AND momentum is turning up" — explicit dip-bottom signal.
# dip_confirmed_TF = clip(1 - dist_to_TF_lo_atr / 2, 0, 1) * sigmoid(TF_ema20_slope_atr_v2 * 5)
# Range: [0, 1]. High = "just bounced off recent low" → buy-the-dip candidate.
NUMERIC_STATE_COLS_DIP = [
    # DIP per TF — ALL 5 TFs after Bug 2 fix (dist_to_*_lo_atr alive for all).
    # 2026-05-24 PM: dropped dip_proximity_m5/m15_v3 (saturated 99% nnz, low std).
    # dip_confirmed_m5/m15 retain the slope-gated signal which IS informative.
    "dip_confirmed_m5_v3",
    "dip_confirmed_m15_v3",
    "dip_proximity_h1_v3",  "dip_confirmed_h1_v3",
    "dip_proximity_h4_v3",  "dip_confirmed_h4_v3",
    "dip_proximity_d1_v3",  "dip_confirmed_d1_v3",
]

# 2026-05-24 STRUCTURE per TF — ALL 5 TFs (HH/HL/LH/LL pattern detection).
# After Bug 1 fix: mom_5/mom_20 alive on M5/M15/H1/H4/D1.
NUMERIC_STATE_COLS_STRUCT = [
    # M5
    "struct_continuation_up_m5_v3",     "struct_pullback_in_uptrend_m5_v3",
    "struct_continuation_down_m5_v3",   "struct_bounce_in_downtrend_m5_v3",
    "struct_pullback_depth_m5_v3",
    # M15
    "struct_continuation_up_m15_v3",    "struct_pullback_in_uptrend_m15_v3",
    "struct_continuation_down_m15_v3",  "struct_bounce_in_downtrend_m15_v3",
    "struct_pullback_depth_m15_v3",
    # H1
    "struct_continuation_up_h1_v3",     "struct_pullback_in_uptrend_h1_v3",
    "struct_continuation_down_h1_v3",   "struct_bounce_in_downtrend_h1_v3",
    "struct_pullback_depth_h1_v3",
    # H4
    "struct_continuation_up_h4_v3",     "struct_pullback_in_uptrend_h4_v3",
    "struct_continuation_down_h4_v3",   "struct_bounce_in_downtrend_h4_v3",
    "struct_pullback_depth_h4_v3",
    # D1
    "struct_continuation_up_d1_v3",     "struct_pullback_in_uptrend_d1_v3",
    "struct_continuation_down_d1_v3",   "struct_bounce_in_downtrend_d1_v3",
    "struct_pullback_depth_d1_v3",
    # Multi-TF combo
    # 2026-05-24 PM: dropped struct_all_tf_pullback_v3 (0% nnz on 9.3M rows —
    # TFs are negatively correlated for pullback, all-5-align literally never happens).
    "struct_tf_agree_count_v3",           # soft 0-1 fraction of TFs agreeing (kept — 67.8% nnz)
    "struct_dip_x_uptrend_v3",            # avg dip (5 TFs) × M5 pullback
    "struct_smc_swing_x_dip_v3",          # SMC swing × M5 dip
]

NUMERIC_STATE_COLS = (
    NUMERIC_STATE_COLS_CANDIDATE
    + NUMERIC_STATE_COLS_CHUNK0
    + NUMERIC_STATE_COLS_CANONICAL
    + NUMERIC_STATE_COLS_DERIVED
    + NUMERIC_STATE_COLS_DIP
    + NUMERIC_STATE_COLS_STRUCT
)

# ─────────────────────────────────────────────────────────────────────────────
# V2 (2026-05-22): C+prune strategy. When GX1_BUILD_IQL_V2=1, input is the
# AUGMENTED forward-outcome dir (153 extra cols: 125 per-TF V2 + 28 group-A).
# State vector: drop 44 V1-confirmed-weak + add 153 V2 → ~291 features.
# Permutation importance (Phase C4b) prunes weak ones after preliminary training.
# ─────────────────────────────────────────────────────────────────────────────

V2_BUILD_MODE = bool(int(os.environ.get("GX1_BUILD_IQL_V2", "0")))

# Phase A drop list (44 weak features from PLUS5 ensemble permutation importance).
V2_DROP_FEATURES = (
    "smc_premium_discount_canon_v1", "_v1_r3_canon_v1", "_v1_range_adr_canon_v1",
    "smc_premium_state_canon_v1", "_v1h4_slope5_canon_v1", "_v1_range_z_canon_v1",
    "range_canon_v1", "_v1_pk_sigma20_canon_v1", "smc_swing_state_canon_v1",
    "_v1_rsi14_canon_v1", "ret_1_canon_v1", "vol_ratio_canon_v1", "_v1_r1_canon_v1",
    "_v1_cost_bps_dyn_canon_v1", "_v1h1_slope3_canon_v1", "_v1_rsi2_gt_rsi14_canon_v1",
    "_v1_kama_slope_30_canon_v1", "ret_20_canon_v1", "_v1_rsi2_canon_v1",
    "_v1_tr_1_over_atr_14_canon_v1", "_v1_tema_slope_20_canon_v1",
    "smc_bos_down_canon_v1", "m5_phase_0_canon_v1", "_v1_body_share_1_canon_v1",
    "_v1_clv_canon_v1", "smc_bars_since_sweep_canon_v1", "smc_sweep_down_canon_v1",
    "smc_sweep_size_atr_canon_v1", "_v1h4_slope3_canon_v1", "smc_sweep_up_canon_v1",
    "_v1_r48_z_canon_v1", "roc20_canon_v1", "_v1h1_slope5_canon_v1",
    "_v1_close_ema_slope_3_canon_v1", "_v1_r5_canon_v1", "_v1_wick_imbalance_canon_v1",
    "m5_phase_4_canon_v1", "body_pct_canon_v1", "m5_phase_3_canon_v1",
    "wick_asym_canon_v1", "smc_bos_up_canon_v1", "m5_phase_1_canon_v1",
    "m5_phase_2_canon_v1", "_v1_ret_ema_diff_2_5_canon_v1", "_v1_upper_tr_canon_v1",
)

# 125 per-TF V2 features (5 TFs × 25 V2 cache features).
# Naming matches augment_forward_outcome_v2.py:_pertf_name() output.
def _build_v2_per_tf_cols():
    from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V2
    tfs = ("m5", "m15", "h1", "h4", "d1")
    cols = []
    for tf in tfs:
        for feat in MULTI_TF_PER_BAR_FEATURES_V2:
            base = f"{tf}_{feat}"
            cols.append(base if feat.endswith("_v2") else base + "_v2")
    return tuple(cols)

V2_PER_TF_COLS = _build_v2_per_tf_cols()   # 125 cols

V2_GROUP_A_COLS = (
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
)   # 32 cols (was 28; +4 for M15/D1 hi/lo)

if V2_BUILD_MODE:
    # V2 final state-vector: V1 (minus drops) + 125 per-TF + 28 group-A
    _v1_kept = tuple(c for c in NUMERIC_STATE_COLS if c not in V2_DROP_FEATURES)
    NUMERIC_STATE_COLS = _v1_kept + V2_PER_TF_COLS + V2_GROUP_A_COLS
    print(f"[V2_BUILD_MODE] state vector: {len(_v1_kept)} V1-kept + "
          f"{len(V2_PER_TF_COLS)} per-TF + {len(V2_GROUP_A_COLS)} group-A "
          f"= {len(NUMERIC_STATE_COLS)} features (pre-prune)", flush=True)

# Categorical state columns to one-hot encode (string-valued candidate cols).
ONE_HOT_COLS = [
    "session",          # OVERLAP / US / EU / ASIA
    "vol_regime",       # EXTREME / HIGH / MEDIUM / LOW
    "trend_regime",     # TREND_UP / TREND_DOWN / TREND_NEUTRAL
    "decision_reason",  # flat_dominant / flat_veto / pre_quality / ...
]

ALLOWED_FINAL_STATUSES = {
    "ENTRY_IQL_V2_PASS_BEATS_REALIZED_BASELINE",
    "ENTRY_IQL_V2_PARTIAL_TIES_BASELINE",
    "ENTRY_IQL_V2_PARTIAL_DEGRADES_VS_BASELINE",
    "ENTRY_IQL_V2_PARTIAL_DEGENERATE_ACTION_DIST",
    "ENTRY_IQL_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
}
ALLOWED_NEXT_ACTIONS = {
    "BUILD_JOINT_ENTRY_AND_EXIT_IQL_V2",
    "REPAIR_ENTRY_IQL_V2_BEFORE_FURTHER_WORK",
    "BUILD_ENTRY_IQL_V2_RUNTIME_ADAPTER",
}

_stamp = contract_gate._stamp
_utc_now = contract_gate._utc_now
_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows


# ---------------------------------------------------------------------------
# Dataset loading + reward construction
# ---------------------------------------------------------------------------


def load_forward_outcome_dataset(per_week_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    parquets = sorted(per_week_dir.glob("forward_outcomes_*.parquet"))
    if not parquets:
        raise RuntimeError(f"[{ACTION}] no forward_outcomes_*.parquet found in {per_week_dir}")
    for p in parquets:
        parts.append(pd.read_parquet(p))
    df = pd.concat(parts, ignore_index=True)
    return df


def build_reward_matrix(df: pd.DataFrame, *, variant: str) -> np.ndarray:
    """Return R of shape (n_rows, 3_actions, n_K). NaN-filled to 0.

    Action layout follows iql_core.ACTION_*_ID (3-action):
      0=SKIP, 1=TAKE_LONG_NOW, 2=TAKE_SHORT_NOW

    SKIP semantically means "re-evaluate next M5 bar" — the runtime calls
    the adapter again on the next V10 candidate. Reward = 0 for SKIP.
    """
    n = len(df)
    R = np.zeros((n, iql_core.N_ACTIONS_V1, N_K), dtype=np.float32)
    # SKIP (action_id 0) → reward 0 across all K. Already initialised.

    # Per-row entry spread cost (used by R_NET_REAL only).
    # Falls back to 1.5 bps (XAUUSD typical median) if missing.
    entry_spread_bps = (
        pd.to_numeric(df.get("entry_spread_bps"), errors="coerce")
        .fillna(1.5).clip(lower=0.0).astype(float).to_numpy()
        if "entry_spread_bps" in df.columns
        else np.full(n, 1.5, dtype=float)
    )

    # 2026-Q2 BIDIR: hard penalty multiplier from V10 v2's bad_path probability.
    # If V10 predicts the trade goes bad (bad_path_prob → 1.0), TAKE rewards
    # are scaled toward zero, making SKIP comparatively attractive. The IQL learns
    # to internalize "trust V10's bad_path signal".
    # If column missing (legacy candidates without V10 v2 logits), no-op gate (= 1.0).
    bad_path_prob = (
        pd.to_numeric(df.get("bad_path_prob"), errors="coerce")
        .fillna(0.0).clip(lower=0.0, upper=1.0).astype(float).to_numpy()
        if "bad_path_prob" in df.columns
        else np.zeros(n, dtype=float)
    )
    bad_path_gate = (1.0 - bad_path_prob).astype(np.float32)  # 1.0 = no gate, 0.0 = full block

    if variant.startswith("R_WAIT_OPP_"):
        # Counterfactual-aware reward: explicitly credits SKIP when waiting beats taking.
        # 2026-06-03 (vedtak entry_iql_perside_reward_20260603): the `_SYM` family fixes the
        # r_skip ASYMMETRY — cement compared an UNPENALIZED best_wait against an
        # MAE+spread-penalized best_take, structurally over-crediting SKIP (~98% skip) and
        # corrupting per-side calibration. _SYM penalizes the waited side with ITS OWN MAE +
        # spread (the waited side eventually enters too), making r_skip a like-for-like compare.
        _sym = variant.endswith("_SYM")
        _base = variant[:-4] if _sym else variant
        _CFG = {
            "R_WAIT_OPP_K96_LAM05": (96, 0.5),
            "R_WAIT_OPP_K96_LAM10": (96, 1.0),
            "R_WAIT_OPP_K96_LAM20": (96, 2.0),
            "R_WAIT_OPP_K96_LAM30": (96, 3.0),
            "R_WAIT_OPP_K96_LAM50": (96, 5.0),
            "R_WAIT_OPP_K48_LAM05": (48, 0.5),
            "R_WAIT_OPP_K48_LAM10": (48, 1.0),
        }[_base]
        _K, _LAM = _CFG
        # _SYM (2026-06-03): TWO additional fixes folded in from the all-models scan —
        #  EIQL-R1: TRUE PER-K reward. The base variants compute ONE K and broadcast it to all
        #    6 Q-heads (multi-head Q + mean/max aggregator = no-op). _SYM instead gives each
        #    K-head its OWN horizon's outcome, re-activating the multi-head architecture.
        #  LBL-2: NO double-spread. take_now_*_terminal_pnl is already bid/ask-based (long exits
        #    on bid, short on ask -> round-trip spread embedded), so subtracting 2*entry_spread
        #    again double-charges. _SYM uses spread_coef=0; base keeps 2.0 for reproducibility.
        _spread_coef = 0.0 if _sym else 2.0

        def _wait_opp_at_K(K):
            tl = df[f"take_now_long_terminal_pnl_at_K{K}_v1"].astype(float).fillna(0.0).to_numpy()
            ts = df[f"take_now_short_terminal_pnl_at_K{K}_v1"].astype(float).fillna(0.0).to_numpy()
            ml = np.maximum(df[f"take_now_long_mae_before_mfe_bps_at_K{K}_v1"].astype(float).fillna(0.0).to_numpy(), 0.0)
            ms = np.maximum(df[f"take_now_short_mae_before_mfe_bps_at_K{K}_v1"].astype(float).fillna(0.0).to_numpy(), 0.0)
            wl = df[f"wait_long_terminal_pnl_at_K{K}_v1"].astype(float).fillna(0.0).to_numpy()
            ws = df[f"wait_short_terminal_pnl_at_K{K}_v1"].astype(float).fillna(0.0).to_numpy()
            rl = tl - _LAM * ml - _spread_coef * entry_spread_bps
            rs = ts - _LAM * ms - _spread_coef * entry_spread_bps
            if _sym:
                mwl = np.maximum(df[f"wait_long_mae_before_mfe_bps_at_K{K}_v1"].astype(float).fillna(0.0).to_numpy(), 0.0)
                mws = np.maximum(df[f"wait_short_mae_before_mfe_bps_at_K{K}_v1"].astype(float).fillna(0.0).to_numpy(), 0.0)
                wl = wl - _LAM * mwl - _spread_coef * entry_spread_bps
                ws = ws - _LAM * mws - _spread_coef * entry_spread_bps
            rk = np.maximum(0.0, np.maximum(wl, ws) - np.maximum(rl, rs))
            return (np.clip(rl, -500, 500) * bad_path_gate,
                    np.clip(rs, -500, 500) * bad_path_gate,
                    np.clip(rk, 0, 500))

        if _sym:
            # per-K: each Q-head ki gets the reward computed at its own horizon K_HORIZONS[ki]
            for ki in range(N_K):
                rl, rs, rk = _wait_opp_at_K(K_HORIZONS[ki])
                R[:, 0, ki] = rk.astype(np.float32)
                R[:, 1, ki] = rl.astype(np.float32)
                R[:, 2, ki] = rs.astype(np.float32)
        else:
            rl, rs, rk = _wait_opp_at_K(_K)   # single-K, broadcast (cement reproducibility)
            for ki in range(N_K):
                R[:, 0, ki] = rk.astype(np.float32)
                R[:, 1, ki] = rl.astype(np.float32)
                R[:, 2, ki] = rs.astype(np.float32)
        return R

    if variant == "R_V10_PQ_COND_K96":
        # V10 path-quality conditioned reward (vedtak r_v10_pq_cond_design_20260601).
        # Same structure as R_WAIT_OPP_K96_LAM* but λ varies per-row by V10's
        # path_quality_pred. Clean predicted path → small penalty (allow take),
        # messy → large penalty (force skip).
        _K = 96
        tl = df[f"take_now_long_terminal_pnl_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy()
        ts = df[f"take_now_short_terminal_pnl_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy()
        ml = np.maximum(df[f"take_now_long_mae_before_mfe_bps_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy(), 0.0)
        ms = np.maximum(df[f"take_now_short_mae_before_mfe_bps_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy(), 0.0)
        wl = df[f"wait_long_terminal_pnl_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy()
        ws = df[f"wait_short_terminal_pnl_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy()
        # Per-row λ from V10's path_quality_pred (head trained on path-quality target).
        # path_quality column in fwd-outcome df: "path_quality_pred" or "v10_path_quality_pred".
        v10_pq_col = None
        for c in ("path_quality_pred", "v10_path_quality_pred", "path_quality"):
            if c in df.columns:
                v10_pq_col = c
                break
        if v10_pq_col is None:
            raise RuntimeError(
                f"R_V10_PQ_COND_K96 requires V10 path_quality_pred column in df. "
                f"Available cols starting with 'v10' or 'path_q': "
                f"{[c for c in df.columns if c.startswith('v10') or c.startswith('path_q')][:10]}"
            )
        pq = pd.to_numeric(df[v10_pq_col], errors="coerce").fillna(0.5).astype(float).to_numpy()
        pq_clipped = np.clip(pq, 0.0, 0.5)
        lambda_eff = 8.0 - 14.0 * pq_clipped  # 1.0 (pq=0.5) to 8.0 (pq=0)
        r_long = tl - lambda_eff * ml - 2.0 * entry_spread_bps
        r_short = ts - lambda_eff * ms - 2.0 * entry_spread_bps
        best_take = np.maximum(r_long, r_short)
        best_wait = np.maximum(wl, ws)
        r_skip = np.maximum(0.0, best_wait - best_take)
        r_long = np.clip(r_long, -500, 500) * bad_path_gate
        r_short = np.clip(r_short, -500, 500) * bad_path_gate
        r_skip = np.clip(r_skip, 0, 500)
        for ki in range(N_K):
            R[:, 0, ki] = r_skip.astype(np.float32)  # SKIP
            R[:, 1, ki] = r_long.astype(np.float32)  # LONG
            R[:, 2, ki] = r_short.astype(np.float32)  # SHORT
        return R

    if variant == "R_V10_MULTI_COND_K96":
        # Multi-signal V10-conditioned reward (iter on R_V10_PQ_COND_K96).
        _K = 96
        tl = df[f"take_now_long_terminal_pnl_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy()
        ts = df[f"take_now_short_terminal_pnl_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy()
        ml = np.maximum(df[f"take_now_long_mae_before_mfe_bps_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy(), 0.0)
        ms = np.maximum(df[f"take_now_short_mae_before_mfe_bps_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy(), 0.0)
        wl = df[f"wait_long_terminal_pnl_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy()
        ws = df[f"wait_short_terminal_pnl_at_K{_K}_v1"].astype(float).fillna(0.0).to_numpy()
        # Three V10 signal columns. Tolerate naming variation across builds.
        def _get(col_options, default=0.5):
            for c in col_options:
                if c in df.columns:
                    return pd.to_numeric(df[c], errors="coerce").fillna(default).astype(float).to_numpy()
            raise RuntimeError(f"R_V10_MULTI_COND_K96 needs one of: {col_options}")
        pq = _get(("path_quality_pred", "v10_path_quality_pred"))
        tradable = _get(("tradable_prob", "v10_tradable_prob"))
        mfe_pred = _get(("mfe_first_n_pred", "v10_mfe_pred_at_entry", "mfe_at_entry_pred"))
        # λ_eff cocktail
        lambda_eff = (
            8.0
            - 5.0 * np.clip(pq, 0.0, 0.5)
            - 3.0 * np.clip(tradable - 0.5, 0.0, 0.5)
            - 2.0 * np.clip(mfe_pred / 50.0, 0.0, 1.0)
        )
        r_long = tl - lambda_eff * ml - 2.0 * entry_spread_bps
        r_short = ts - lambda_eff * ms - 2.0 * entry_spread_bps
        best_take = np.maximum(r_long, r_short)
        best_wait = np.maximum(wl, ws)
        r_skip = np.maximum(0.0, best_wait - best_take)
        r_long = np.clip(r_long, -500, 500) * bad_path_gate
        r_short = np.clip(r_short, -500, 500) * bad_path_gate
        r_skip = np.clip(r_skip, 0, 500)
        for ki in range(N_K):
            R[:, 0, ki] = r_skip.astype(np.float32)
            R[:, 1, ki] = r_long.astype(np.float32)
            R[:, 2, ki] = r_short.astype(np.float32)
        return R

    if variant.startswith("R_HYBRID_K96_TOL"):
        # R_HYBRID = R_WAIT_OPP + binary clean-threshold filter.
        # TAKE only allowed if (MAE_pre < TOL) AND (terminal_pnl > 0).
        # SKIP reward = wait - take counterfactual delta (R_WAIT_OPP style).
        _TOL = {"R_HYBRID_K96_TOL20": 20.0, "R_HYBRID_K96_TOL40": 40.0}[variant]
        tl = df["take_now_long_terminal_pnl_at_K96_v1"].astype(float).fillna(0.0).to_numpy()
        ts = df["take_now_short_terminal_pnl_at_K96_v1"].astype(float).fillna(0.0).to_numpy()
        ml = np.maximum(df["take_now_long_mae_before_mfe_bps_at_K96_v1"].astype(float).fillna(0.0).to_numpy(), 0.0)
        ms = np.maximum(df["take_now_short_mae_before_mfe_bps_at_K96_v1"].astype(float).fillna(0.0).to_numpy(), 0.0)
        wl = df["wait_long_terminal_pnl_at_K96_v1"].astype(float).fillna(0.0).to_numpy()
        ws = df["wait_short_terminal_pnl_at_K96_v1"].astype(float).fillna(0.0).to_numpy()
        # Binary gate: only credit TAKE if MAE_pre < TOL (clean entry only)
        clean_l = ml < _TOL
        clean_s = ms < _TOL
        r_long = np.where(clean_l, tl - 2.0 * entry_spread_bps, -50.0)
        r_short = np.where(clean_s, ts - 2.0 * entry_spread_bps, -50.0)
        best_take = np.maximum(r_long, r_short)
        best_wait = np.maximum(wl, ws)
        r_skip = np.maximum(0.0, best_wait - best_take)
        r_long = np.clip(r_long, -500, 500) * bad_path_gate
        r_short = np.clip(r_short, -500, 500) * bad_path_gate
        r_skip = np.clip(r_skip, 0, 500)
        for ki in range(N_K):
            R[:, 0, ki] = r_skip.astype(np.float32)
            R[:, 1, ki] = r_long.astype(np.float32)
            R[:, 2, ki] = r_short.astype(np.float32)
        return R

    for ai, (action_prefix, side) in enumerate([
        ("take_now", "long"),
        ("take_now", "short"),
    ], start=1):
        for ki, K in enumerate(K_HORIZONS):
            mfe_col = f"{action_prefix}_{side}_mfe_bps_at_K{K}_v1"
            mae_pre_col = f"{action_prefix}_{side}_mae_before_mfe_bps_at_K{K}_v1"
            mfe = df[mfe_col].astype(float).fillna(0.0).to_numpy()
            mae_pre = df[mae_pre_col].astype(float).fillna(0.0).to_numpy()
            mae_pre = np.maximum(mae_pre, 0.0)  # magnitude
            if variant == "R_RAW":
                r = mfe
            elif variant == "R_NET05":
                r = mfe - 0.5 * mae_pre
            elif variant == "R_NET10":
                r = mfe - 1.0 * mae_pre
            elif variant == "R_GATED":
                r = np.where(mae_pre < GATED_THRESHOLD_BPS, mfe, 0.0)
            elif variant == "R_PATH":
                denom = np.maximum(mfe, mae_pre)
                denom = np.where(denom < 1e-6, 1e-6, denom)
                r = mfe * mfe / denom
                r = np.where(mfe < 0, mfe, r)  # negative MFE → keep negative
            elif variant == "R_NET_REAL":
                # Realistic exit-aware reward: only fraction of MFE is captured,
                # MAE-before-MFE is penalised, and entry+exit spread cost subtracted.
                r = (R_NET_REAL_ALPHA_MFE * mfe
                     - R_NET_REAL_BETA_MAE * mae_pre
                     - R_NET_REAL_GAMMA_SPREAD * entry_spread_bps)
            elif variant in ("R_TERMINAL_K24", "R_TERMINAL_K96"):
                # Direct terminal-PnL reward at fixed K horizon (no MFE proxy).
                # Train target = "PnL you actually realize at K bars forward".
                # Avoids R_NET_REAL's MFE-vs-terminal-PnL gap that systematically
                # over-rewards SHORTS in uptrending markets (2026-05-24 diag).
                _K_TARGET = 24 if variant == "R_TERMINAL_K24" else 96
                term_pnl_col = f"{action_prefix}_{side}_terminal_pnl_at_K{_K_TARGET}_v1"
                if term_pnl_col not in df.columns:
                    raise RuntimeError(f"missing {term_pnl_col} (needed for {variant})")
                term_pnl = df[term_pnl_col].astype(float).fillna(0.0).to_numpy()
                # Same K-horizon shape: copy single-K reward into all K slots so
                # the IQL doesn't see K-varying signal (each K slot = terminal PnL @ K_TARGET).
                # Subtract spread (entry + exit each pay 1x).
                r = term_pnl - 2.0 * entry_spread_bps
            elif variant.startswith("R_PATIENT_K96_LAM"):
                # Patient-entry reward: penalises MAE BEFORE profit.
                # Goal: Entry-IQL learns to skip trades that immediately draw down,
                # and wait for the dip to finish before entering.
                #   r = terminal_pnl@K96 - λ * mae_before_mfe@K96 - 2*spread
                _LAM = {"R_PATIENT_K96_LAM03": 0.3,
                        "R_PATIENT_K96_LAM05": 0.5,
                        "R_PATIENT_K96_LAM10": 1.0}[variant]
                term_pnl_col = f"{action_prefix}_{side}_terminal_pnl_at_K96_v1"
                mae_pre_k96_col = f"{action_prefix}_{side}_mae_before_mfe_bps_at_K96_v1"
                if term_pnl_col not in df.columns:
                    raise RuntimeError(f"missing {term_pnl_col} (needed for {variant})")
                if mae_pre_k96_col not in df.columns:
                    raise RuntimeError(f"missing {mae_pre_k96_col} (needed for {variant})")
                term_pnl = df[term_pnl_col].astype(float).fillna(0.0).to_numpy()
                mae_pre_k96 = np.maximum(df[mae_pre_k96_col].astype(float).fillna(0.0).to_numpy(), 0.0)
                r = term_pnl - _LAM * mae_pre_k96 - 2.0 * entry_spread_bps
            elif variant.startswith("R_CLEAN_K96_TOL"):
                # R_CLEAN_K96 = binary-threshold MAE filter.
                # r = (terminal_pnl - 2*spread) if mae_pre < TOL else 0
                # Forces model to learn: only TAKE when entry is clean (MAE below tol).
                _TOL = {"R_CLEAN_K96_TOL10": 10.0,
                        "R_CLEAN_K96_TOL20": 20.0,
                        "R_CLEAN_K96_TOL40": 40.0}[variant]
                term_pnl_col = f"{action_prefix}_{side}_terminal_pnl_at_K96_v1"
                mae_pre_k96_col = f"{action_prefix}_{side}_mae_before_mfe_bps_at_K96_v1"
                term_pnl = df[term_pnl_col].astype(float).fillna(0.0).to_numpy()
                mae_pre_k96 = np.maximum(df[mae_pre_k96_col].astype(float).fillna(0.0).to_numpy(), 0.0)
                clean = mae_pre_k96 < _TOL
                r = np.where(clean, term_pnl - 2.0 * entry_spread_bps, 0.0)
            elif variant.startswith("R_QUAD_K96_LAM"):
                # R_QUAD_K96 = quadratic MAE penalty (small MAE ~free, large MAE bombs).
                # r = terminal_pnl - λ * (mae_pre / 10)² - 2*spread
                # Examples (LAM=1.0): MAE=10 → -1 bps, MAE=50 → -25 bps, MAE=200 → -400 bps
                _LAM = {"R_QUAD_K96_LAM05": 0.5,
                        "R_QUAD_K96_LAM10": 1.0}[variant]
                term_pnl_col = f"{action_prefix}_{side}_terminal_pnl_at_K96_v1"
                mae_pre_k96_col = f"{action_prefix}_{side}_mae_before_mfe_bps_at_K96_v1"
                term_pnl = df[term_pnl_col].astype(float).fillna(0.0).to_numpy()
                mae_pre_k96 = np.maximum(df[mae_pre_k96_col].astype(float).fillna(0.0).to_numpy(), 0.0)
                # Quadratic in (MAE/10) so small MAE essentially free
                r = term_pnl - _LAM * (mae_pre_k96 / 10.0) ** 2 - 2.0 * entry_spread_bps
            elif variant == "R_QUALITY_K96":
                # Scale-invariant entry-quality ratio. The ratio MAE_pre/MFE@K96 measures
                # how "clean" the entry was: tiny MAE relative to peak = high quality.
                # r = terminal_pnl * max(0, 1 - MAE_pre/MFE) - 2*spread
                # No hyperparameter; the data structure itself defines quality.
                term_pnl_col = f"{action_prefix}_{side}_terminal_pnl_at_K96_v1"
                mae_pre_k96_col = f"{action_prefix}_{side}_mae_before_mfe_bps_at_K96_v1"
                mfe_k96_col = f"{action_prefix}_{side}_mfe_bps_at_K96_v1"
                term_pnl = df[term_pnl_col].astype(float).fillna(0.0).to_numpy()
                mae_pre_k96 = np.maximum(df[mae_pre_k96_col].astype(float).fillna(0.0).to_numpy(), 0.0)
                mfe_k96 = np.maximum(df[mfe_k96_col].astype(float).fillna(0.0).to_numpy(), 1e-6)
                quality = np.clip(1.0 - mae_pre_k96 / mfe_k96, 0.0, 1.0)
                r = term_pnl * quality - 2.0 * entry_spread_bps
            elif variant.startswith("R_SOFT_K96_TOL"):
                # Sigmoid-soft binary: smooth transition around TOL instead of hard cliff.
                # r = terminal_pnl * sigmoid((TOL - MAE_pre) / scale) - 2*spread
                # scale=10 so transition spans ~30 bps around TOL.
                _TOL = {"R_SOFT_K96_TOL20": 20.0,
                        "R_SOFT_K96_TOL40": 40.0}[variant]
                _SCALE = 10.0
                term_pnl_col = f"{action_prefix}_{side}_terminal_pnl_at_K96_v1"
                mae_pre_k96_col = f"{action_prefix}_{side}_mae_before_mfe_bps_at_K96_v1"
                term_pnl = df[term_pnl_col].astype(float).fillna(0.0).to_numpy()
                mae_pre_k96 = np.maximum(df[mae_pre_k96_col].astype(float).fillna(0.0).to_numpy(), 0.0)
                weight = 1.0 / (1.0 + np.exp(-(_TOL - mae_pre_k96) / _SCALE))
                r = term_pnl * weight - 2.0 * entry_spread_bps
            elif variant.startswith("R_ASYMMETRIC_K96_LAM"):
                # Logarithmic MAE penalty: gentle for noise-level MAE, firm for large.
                # r = terminal_pnl - λ * 10 * log(1 + MAE_pre/10) - 2*spread
                # Examples (λ=1): MAE=10 → -6.9, MAE=50 → -17.9, MAE=200 → -30.4
                _LAM = {"R_ASYMMETRIC_K96_LAM05": 0.5,
                        "R_ASYMMETRIC_K96_LAM10": 1.0}[variant]
                term_pnl_col = f"{action_prefix}_{side}_terminal_pnl_at_K96_v1"
                mae_pre_k96_col = f"{action_prefix}_{side}_mae_before_mfe_bps_at_K96_v1"
                term_pnl = df[term_pnl_col].astype(float).fillna(0.0).to_numpy()
                mae_pre_k96 = np.maximum(df[mae_pre_k96_col].astype(float).fillna(0.0).to_numpy(), 0.0)
                r = term_pnl - _LAM * 10.0 * np.log1p(mae_pre_k96 / 10.0) - 2.0 * entry_spread_bps
            else:
                raise ValueError(f"unknown reward variant: {variant}")
            # Clip extreme values for stability (top 0.1% MFE values can be 500+ bps)
            r = np.clip(r, -500.0, 500.0)
            # 2026-Q2 BIDIR: apply bad_path gate. Multiplies the MFE reward by
            # (1 - bad_path_prob) so V10's bad-path predictions translate into
            # reduced incentive to TAKE.
            r = r * bad_path_gate
            R[:, ai, ki] = r.astype(np.float32)
    return R


def build_state_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return (X, feature_names). One-hot encodes categorical cols.

    DATA INTEGRITY:
      - Hard-fails if any required feature column has >5% NaN (indicates
        upstream pipeline failure).
      - Pre-converts numeric cols, fills NaN with 0.0 only as last-resort
        warmup-edge handling. The presence flags
        `_chunk0_features_complete_v1` / `_canonical_features_complete_v1`
        are inspected upstream to refuse rows with missing critical features.
    """
    # Refuse rows with incomplete features (canonical or chunk_0 missing).
    # These are warmup-edge or week-with-no-chunk_0 rows that should never
    # be used for training.
    if "_canonical_features_complete_v1" in df.columns:
        complete_mask = df["_canonical_features_complete_v1"].fillna(False).astype(bool)
        if "_chunk0_features_complete_v1" in df.columns:
            complete_mask = complete_mask & df["_chunk0_features_complete_v1"].fillna(False).astype(bool)
        n_dropped = int((~complete_mask).sum())
        if n_dropped > 0:
            print(f"[BUILD_STATE_MATRIX] dropping {n_dropped:,} rows with incomplete features", flush=True)
        df = df.loc[complete_mask].reset_index(drop=True)

    # 2026-05-24 DIP + STRUCTURE per TF — ALL 5 TFs (M5/M15/H1/H4/D1).
    # 2026-06-04 (Phase 0a / O2=A): compute the 36 dip/struct (+24 GROUP-A) ctx_cont
    # cols via the ONE-TRUTH helper attach_group_a_dip_struct_ctx_columns — the SAME
    # function the serving path (v12_state_from_prebuilt.py:428) and candidate-gen
    # (materialize_inference_batch_candidates_v3_v1.py:597) use, so the model sees
    # identical values at train and serve (no skew). The helper is IDEMPOTENT: if
    # candidate-gen already attached the 36 (the retrain path), it returns immediately
    # and PRESERVES the real values — fixing the dead-zero bug where the old inline
    # block recomputed them from per-TF inputs ABSENT in the COSTFIX frame and
    # overwrote the real values with zeros. Fail-loud: the helper raises if OHLC is
    # missing and the cols are not already present (better than silent zero-fill).
    df = df.copy()
    from gx1.scripts.augment_forward_outcome_v2 import attach_group_a_dip_struct_ctx_columns
    df = attach_group_a_dip_struct_ctx_columns(df, journal_label="entry_iql_v2_build")

    state_parts: list[pd.DataFrame] = []
    feature_names: list[str] = []
    nan_warnings: list[tuple[str, float]] = []
    absent_cols: list[str] = []      # Phase 0a/E6 fail-loud: allowlisted col not in frame
    constant_cols: list[str] = []    # present but std~0 (likely dead/degenerate)
    # Numeric features
    for c in NUMERIC_STATE_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            nan_frac = s.isna().mean()
            if nan_frac > 0.05:
                nan_warnings.append((c, float(nan_frac)))
            filled = s.fillna(0.0)
            if float(filled.std()) <= 1e-12:
                constant_cols.append(c)
            state_parts.append(pd.DataFrame({c: filled}))
            feature_names.append(c)
        else:
            absent_cols.append(c)
            state_parts.append(pd.DataFrame({c: np.zeros(len(df))}))
            feature_names.append(c)
    # Phase 0a/E6 (2026-06-04) — FAIL-LOUD: an allowlisted feature ABSENT from the frame
    # is the dead-zero footgun (the 36 dip/struct silently 0-filled until E1; the
    # 2026-05-19 LONG-bias collapse). Refuse to silent-0-fill a whole block. (This will
    # correctly error any build run on a frame lacking allowlisted cols — wire the source
    # upstream, e.g. attach_group_a_dip_struct_ctx_columns, or drop the col from the
    # allowlist before training.)
    if absent_cols:
        raise RuntimeError(
            f"[BUILD_STATE_MATRIX] {len(absent_cols)} allowlisted numeric feature(s) ABSENT "
            f"from the frame — refusing to silent-0-fill (dead-zero guard): {absent_cols[:20]}"
            + (f" (+{len(absent_cols)-20} more)" if len(absent_cols) > 20 else "")
            + ". Wire the source upstream or drop from NUMERIC_STATE_COLS before training."
        )
    # CONSTANT cols only WARN (not raise): a legit rare-binary flag can be all-0 in a given
    # training window (std=0) — flag for review rather than break the build (O7).
    if constant_cols:
        # P4 (2026-06-06): a CONSTANT col is now FAIL-CLOSED unless it's on the ONE-TRUTH
        # feature-liveness allowlist (structural/rare-binary). A non-allowlisted constant = a
        # silent-ignore / zeroed-handoff regression (the historical 36-dip/struct dead-zero) → RAISE.
        from gx1.audit.feature_liveness import KNOWN_ALLOWED_DEAD
        _new_const = [c for c in constant_cols if c not in KNOWN_ALLOWED_DEAD]
        if _new_const:
            raise RuntimeError(
                f"[BUILD_STATE_MATRIX] {len(_new_const)} numeric feature(s) CONSTANT (std~0) and NOT on "
                f"the feature-liveness allowlist — silent-ignore/zeroed-handoff regression: {_new_const[:20]}"
                + (f" (+{len(_new_const)-20} more)" if len(_new_const) > 20 else "")
                + ". Fix the upstream hand-off, or (if truly structural) add to KNOWN_ALLOWED_DEAD with a reason."
            )
        print(
            f"[BUILD_STATE_MATRIX][WARN] {len(constant_cols)} allowlisted constant feature(s) "
            f"(structural/rare-binary, OK): {constant_cols[:20]}", flush=True,
        )
    if nan_warnings:
        msg = ", ".join(f"{n}={f:.1%}" for n, f in nan_warnings[:8])
        raise RuntimeError(
            f"[BUILD_STATE_MATRIX] {len(nan_warnings)} feature(s) have >5% NaN "
            f"(first 8 listed): {msg}. Investigate upstream pipeline before training."
        )
    # One-hot features
    for c in ONE_HOT_COLS:
        if c not in df.columns:
            continue
        s = df[c].astype("string").fillna("UNKNOWN")
        unique_values = sorted(s.unique().tolist())
        for v in unique_values:
            col_name = f"{c}__{v}"
            state_parts.append(pd.DataFrame({col_name: (s == v).astype(np.float32).to_numpy()}))
            feature_names.append(col_name)
    X_df = pd.concat(state_parts, axis=1).reset_index(drop=True)
    X = X_df.to_numpy().astype(np.float32)
    return X, feature_names


def derive_oracle_action(
    R_primary: np.ndarray, K_primary_idx: int,
) -> np.ndarray:
    """Per-row: argmax over a of R_primary[:, a, K_primary_idx]. Returns (n,) action ids."""
    return R_primary[:, :, K_primary_idx].argmax(axis=1).astype(np.int64)


# ---------------------------------------------------------------------------
# Fold construction (stratified shuffle by oracle action, with weekly-blocked option)
# ---------------------------------------------------------------------------


def build_stratified_folds(
    n_rows: int,
    oracle_action: np.ndarray,
    n_folds: int = N_FOLDS,
    val_size: float = VAL_SIZE,
    test_size: float = TEST_SIZE,
    seed: int = SEED_V1,
) -> list[dict[str, np.ndarray]]:
    """Generate n_folds (train, val, test) splits via two-level StratifiedShuffleSplit.

    Each fold uses the same total population (n_rows) but with a different seed
    derived from the base seed. The val/test sets are STRATIFIED by action label
    so each fold's val has all 5 actions represented (when present in pop).
    """
    folds: list[dict[str, np.ndarray]] = []
    all_idx = np.arange(n_rows)
    for fold_id_v1 in range(n_folds):
        # First, split off (val + test) from train
        non_train_size = val_size + test_size
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=non_train_size, random_state=seed + fold_id_v1,
        )
        train_idx, valtest_idx = next(sss1.split(all_idx, oracle_action))
        # Then split (val + test) into val and test, stratified
        valtest_oracle = oracle_action[valtest_idx]
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size / non_train_size,
            random_state=seed + fold_id_v1 + 7919,
        )
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


def build_chronological_folds(
    decision_ts: np.ndarray,          # per-row decision timestamp (datetime64 / int64 ns), len == n_rows
    n_folds: int = N_FOLDS,
    embargo_bars: int = 192,          # >= longest forward horizon (16h @ M5 = 192 bars)
) -> list[dict[str, np.ndarray]]:
    """Purged walk-forward chronological splits with an embargo gap.

    2026-06-03 (RR4): the random StratifiedShuffleSplit leaks — neighbor candidates ~5 min
    apart share 8-16h forward windows, so a random test row's outcome overlaps a train row's.
    This sorts by decision time, builds EXPANDING-window walk-forward folds (train = past,
    val + test = later blocks), and PURGES `embargo_bars` rows at each train->val and val->test
    boundary so no forward window straddles the split. Fold N's test is the most recent block
    (the true OOT). Row indices returned are into the ORIGINAL (unsorted) row order.
    """
    n_rows = len(decision_ts)
    order = np.argsort(np.asarray(decision_ts), kind="mergesort")  # stable, chronological
    # (n_folds + 2) contiguous time-blocks: >=1 train block before the first val/test pair.
    blocks = np.array_split(order, n_folds + 2)
    folds: list[dict[str, np.ndarray]] = []
    emb = max(0, int(embargo_bars))
    for k in range(n_folds):
        train = np.concatenate(blocks[: k + 1]) if k + 1 > 0 else np.array([], dtype=int)
        val = blocks[k + 1]
        test = blocks[k + 2]
        # Purge: drop the last `emb` (most-recent) rows of train and of val so their forward
        # windows cannot reach into the following block. Blocks are time-ordered.
        train = train[:-emb] if emb and len(train) > emb else (np.array([], dtype=int) if emb >= len(train) else train)
        val_purged = val[:-emb] if emb and len(val) > emb else (np.array([], dtype=int) if emb >= len(val) else val)
        folds.append({
            "fold_id_v1": f"FOLD_{k + 1}",
            "train_idx_v1": np.sort(train),
            "val_idx_v1": np.sort(val_purged),
            "test_idx_v1": np.sort(test),
        })
    return folds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy_outcome(
    R: np.ndarray, predicted_action: np.ndarray, K_idx: int,
) -> dict[str, Any]:
    """Per-row reward = R[i, predicted_action[i], K_idx]. Sum + dist."""
    n = R.shape[0]
    per_row = R[np.arange(n), predicted_action, K_idx]
    counts = {
        f"n_action_{ai}_v1": int((predicted_action == ai).sum())
        for ai in range(iql_core.N_ACTIONS_V1)
    }
    counts["n_total_v1"] = int(n)
    fractions = {
        f"frac_action_{ai}_v1": float((predicted_action == ai).mean())
        for ai in range(iql_core.N_ACTIONS_V1)
    }
    return {
        "total_reward_bps_v1": float(per_row.sum()),
        "mean_reward_bps_v1": float(per_row.mean()) if n > 0 else 0.0,
        "counts_v1": counts,
        "fractions_v1": fractions,
        "min_action_frac_v1": float(min(fractions.values())) if fractions else 0.0,
    }


def evaluate_baselines(
    R: np.ndarray, K_idx: int,
) -> dict[str, dict[str, Any]]:
    """Baselines: ALWAYS_SKIP, ALWAYS_TAKE_LONG, RANDOM_UNIFORM, ORACLE."""
    n = R.shape[0]
    out: dict[str, dict[str, Any]] = {}
    # ALWAYS_SKIP
    a_skip = np.zeros(n, dtype=np.int64)
    out["always_skip_v1"] = evaluate_policy_outcome(R, a_skip, K_idx)
    # ALWAYS_TAKE_LONG
    a_take_long = np.full(n, iql_core.ACTION_TAKE_LONG_NOW_ID, dtype=np.int64)
    out["always_take_long_v1"] = evaluate_policy_outcome(R, a_take_long, K_idx)
    # RANDOM_UNIFORM
    rng = np.random.default_rng(SEED_V1)
    a_rand = rng.integers(0, iql_core.N_ACTIONS_V1, size=n)
    out["random_uniform_v1"] = evaluate_policy_outcome(R, a_rand, K_idx)
    # ORACLE: argmax over a
    a_oracle = R[:, :, K_idx].argmax(axis=1)
    out["oracle_v1"] = evaluate_policy_outcome(R, a_oracle, K_idx)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def evaluate_one_fold(
    fold: dict[str, np.ndarray],
    X: np.ndarray, R: np.ndarray, oracle_action: np.ndarray,
    *,
    variant: str,
    artifact_root: Path,
    balance_actions: bool = False,
    skip_weight_multiplier: float = 1.0,
    skip_oversample_factor: int = 1,
) -> dict[str, Any]:
    fold_id_v1 = fold["fold_id_v1"]
    train_idx = fold["train_idx_v1"]
    val_idx = fold["val_idx_v1"]
    test_idx = fold["test_idx_v1"]

    # V9 Issue 1 Plan B: oversample SKIP rows in training set. Class-weights
    # alone barely move SKIP-rate (4x: 0.93%, 10x: 0.985%) because oracle SKIP
    # has only 266/50824 rows (0.5%). Replicating the SKIP rows N times gives
    # the IQL actual gradient examples to learn from, not just a multiplier.
    skip_oversample_count = 0
    if skip_oversample_factor > 1:
        skip_mask = oracle_action[train_idx] == iql_core.ACTION_SKIP_ID
        skip_indices = train_idx[skip_mask]
        if len(skip_indices) > 0:
            extra_replicas = np.tile(skip_indices, skip_oversample_factor - 1)
            train_idx = np.concatenate([train_idx, extra_replicas])
            skip_oversample_count = int(len(extra_replicas))

    X_train = X[train_idx]
    R_train = R[train_idx]
    X_val = X[val_idx]
    R_val = R[val_idx]
    X_test = X[test_idx]
    R_test = R[test_idx]

    # Per-class guard on val
    val_action_counts = {
        ai: int((oracle_action[val_idx] == ai).sum())
        for ai in range(iql_core.N_ACTIONS_V1)
    }
    val_class_guard_status = (
        "OK" if min(val_action_counts.values()) >= MIN_PER_CLASS_VAL
        else "WARN_BELOW_MIN_PER_CLASS"
    )

    print(
        f"[{ACTION}] {fold_id_v1} {variant} "
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"val_action_counts={val_action_counts} "
        f"guard={val_class_guard_status}",
        flush=True,
    )

    # Inverse-frequency class weights to counter V10 v2's 83/17 long/short skew.
    # Without this the IQL collapses to always-LONG (16.75 bps fast baseline =
    # 57% of oracle). Q-loss + V-loss both honor sample_weights via expectile_loss.
    sample_weights = None
    train_class_weights: list[float] = []
    train_action_counts: dict[int, int] = {}
    if balance_actions:
        oa_train = oracle_action[train_idx]
        n_per_action = np.bincount(
            oa_train.astype(np.int64), minlength=iql_core.N_ACTIONS_V1,
        ).clip(min=1)
        class_weights = len(oa_train) / (iql_core.N_ACTIONS_V1 * n_per_action)
        # V9 Issue 1: bump SKIP class weight by additional multiplier (default 1.0
        # = inverse-frequency only). Wave 2 final showed Entry-IQL chose SKIP only
        # 0.73% of the time (target 5-10%); a 4-5× SKIP multiplier on top of the
        # ~64× inverse-frequency forces the IQL to attend to rare SKIP cases.
        if skip_weight_multiplier != 1.0:
            class_weights[iql_core.ACTION_SKIP_ID] *= float(skip_weight_multiplier)
        sample_weights = class_weights[oa_train].astype(np.float32)
        train_class_weights = [float(w) for w in class_weights]
        train_action_counts = {ai: int(n_per_action[ai]) for ai in range(iql_core.N_ACTIONS_V1)}
        print(
            f"[{ACTION}] {fold_id_v1} {variant} balance_actions: "
            f"train_action_counts={train_action_counts} "
            f"class_weights={train_class_weights} "
            f"skip_weight_multiplier={skip_weight_multiplier}",
            flush=True,
        )

    # Train multi-head IQL
    # EIQL-2 (2026-06-03 cross-model scan): optional small-weight margin-ranking term on the
    # MSE Q-loss. Default 0.0 = pure-MSE bit-parity with the cemented IQL; the regime-robust
    # retrain sets GX1_ENTRY_IQL_Q_RANKING_WEIGHT (e.g. 0.05) to sharpen the per-(state,K)
    # argmax separation the live policy reads, while keeping Q in bps for the Phase-6 gates.
    _q_rank_w = float(os.environ.get("GX1_ENTRY_IQL_Q_RANKING_WEIGHT", "0.0"))
    _q_rank_m = float(os.environ.get("GX1_ENTRY_IQL_Q_RANKING_MARGIN", "5.0"))
    model = iql_core.train_multi_head_entry_iql(
        X_train, R_train,
        k_horizons=K_HORIZONS,
        n_actions=iql_core.N_ACTIONS_V1,
        tau=TRAIN_TAU,
        epochs_q=TRAIN_EPOCHS_Q,
        epochs_v=TRAIN_EPOCHS_V,
        k_iterations=TRAIN_K_VQ_ITERATIONS,
        batch_size=TRAIN_BATCH_SIZE,
        hidden_dim=TRAIN_HIDDEN_DIM,
        n_hidden=TRAIN_N_HIDDEN,
        seed=SEED_V1,
        prefer_cuda=True,
        sample_weights=sample_weights,
        q_ranking_weight=_q_rank_w,
        q_ranking_margin=_q_rank_m,
    )

    # Evaluate per K-horizon × per aggregator on val and test
    K_primary_idx = K_HORIZONS.index(K_PRIMARY)
    eval_rows: list[dict[str, Any]] = []
    best_val_total = -np.inf
    best_combo: dict[str, Any] | None = None
    best_val_actions = None
    best_test_actions = None

    for agg in AGGREGATORS:
        val_actions = iql_core.policy_argmax_action(X_val, model, aggregator=agg)
        test_actions = iql_core.policy_argmax_action(X_test, model, aggregator=agg)
        for ki, K in enumerate(K_HORIZONS):
            val_metric = evaluate_policy_outcome(R_val, val_actions, ki)
            test_metric = evaluate_policy_outcome(R_test, test_actions, ki)
            eval_rows.append({
                "fold_id_v1": fold_id_v1,
                "variant_v1": variant,
                "aggregator_v1": agg,
                "k_horizon_v1": K,
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
                best_val_actions = val_actions
                best_test_actions = test_actions

    test_baselines = evaluate_baselines(R_test, K_primary_idx)
    val_baselines = evaluate_baselines(R_val, K_primary_idx)

    # Save model
    model_dir = artifact_root / "trained_models_v1"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{variant}_{fold_id_v1}.pt"
    torch.save(
        {
            "q_state_dict": model.q_net.state_dict(),
            "v_state_dict": model.v_net.state_dict(),
            "feature_means": model.feature_means.tolist(),
            "feature_stds": model.feature_stds.tolist(),
            "k_horizons": K_HORIZONS,
            "n_actions": iql_core.N_ACTIONS_V1,
            "state_dim": model.state_dim,
            "variant": variant,
            "fold_id": fold_id_v1,
            "hidden_dim": TRAIN_HIDDEN_DIM,
            "n_hidden": TRAIN_N_HIDDEN,
            "dropout": iql_core.DEFAULT_DROPOUT,
            "schema_v1": "MULTI_HEAD_ENTRY_IQL_V2_CHECKPOINT",
        },
        model_path,
    )

    return {
        "fold_id_v1": fold_id_v1,
        "variant_v1": variant,
        "best_combo_v1": best_combo,
        "best_val_total_reward_bps_v1": float(best_val_total) if np.isfinite(best_val_total) else None,
        "best_test_metric_v1": (
            evaluate_policy_outcome(R_test, best_test_actions, K_primary_idx)
            if best_test_actions is not None else None
        ),
        "val_class_guard_status_v1": val_class_guard_status,
        "val_action_counts_v1": val_action_counts,
        "val_baselines_v1": val_baselines,
        "test_baselines_v1": test_baselines,
        "all_evaluations_v1": eval_rows,
        "model_path_v1": str(model_path),
        "balance_actions_v1": bool(balance_actions),
        "skip_weight_multiplier_v1": float(skip_weight_multiplier),
        "skip_oversample_factor_v1": int(skip_oversample_factor),
        "skip_oversample_added_rows_v1": int(skip_oversample_count),
        "train_action_counts_v1": train_action_counts,
        "train_class_weights_v1": train_class_weights,
    }


def determine_status(per_fold_results: list[dict[str, Any]]) -> tuple[str, str, str, dict[str, Any]]:
    """Aggregate go/no-go across folds + variants. Picks BEST variant by mean test reward."""
    # Group by variant
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in per_fold_results:
        by_variant.setdefault(r["variant_v1"], []).append(r)

    best_variant = None
    best_mean_test = -np.inf
    variant_summaries: dict[str, dict[str, Any]] = {}
    for variant, results in by_variant.items():
        test_totals = []
        oracle_totals = []
        skip_baseline_totals = []
        min_action_fracs = []
        for r in results:
            test_metric = r.get("best_test_metric_v1")
            if test_metric:
                test_totals.append(test_metric["total_reward_bps_v1"])
                min_action_fracs.append(test_metric["min_action_frac_v1"])
            tb = r.get("test_baselines_v1", {})
            if tb.get("oracle_v1"):
                oracle_totals.append(tb["oracle_v1"]["total_reward_bps_v1"])
            if tb.get("always_skip_v1"):
                skip_baseline_totals.append(tb["always_skip_v1"]["total_reward_bps_v1"])
        mean_test = float(np.mean(test_totals)) if test_totals else -np.inf
        mean_oracle = float(np.mean(oracle_totals)) if oracle_totals else 0.0
        mean_skip = float(np.mean(skip_baseline_totals)) if skip_baseline_totals else 0.0
        min_min_frac = float(np.min(min_action_fracs)) if min_action_fracs else 0.0
        variant_summaries[variant] = {
            "mean_test_reward_bps_v1": mean_test,
            "mean_oracle_reward_bps_v1": mean_oracle,
            "mean_skip_baseline_reward_bps_v1": mean_skip,
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

    # Status determination
    best_summary = variant_summaries.get(best_variant or "", {})
    min_frac = best_summary.get("min_action_frac_across_folds_v1", 0.0)
    skip_baseline = best_summary.get("mean_skip_baseline_reward_bps_v1", 0.0)
    delta_skip = best_mean_test - skip_baseline

    # Degeneracy threshold: a class is genuinely degenerate only if it has
    # ZERO usage AND the oracle says it should have been used. Since R_NET_REAL
    # oracle distribution can have SKIP at ~1-3%, requiring 2% of policy
    # actions to be SKIP is too strict. Use 0.1% (3 examples per 3000) as the
    # absolute floor — anything below that means the action is fully ignored.
    if min_frac < 0.001:
        return (
            "ENTRY_IQL_V2_PARTIAL_DEGENERATE_ACTION_DIST",
            "REPAIR_ENTRY_IQL_V2_BEFORE_FURTHER_WORK",
            f"Best variant {best_variant} action distribution degenerate "
            f"(min frac {min_frac:.4f} < 0.1%) — IQL fully ignores at least one action.",
            headline,
        )
    if delta_skip > 200.0:
        return (
            "ENTRY_IQL_V2_PASS_BEATS_REALIZED_BASELINE",
            "BUILD_ENTRY_IQL_V2_RUNTIME_ADAPTER",
            f"Best variant {best_variant} beats always-skip baseline by "
            f"{delta_skip:+.0f} bps total mean test reward.",
            headline,
        )
    if abs(delta_skip) <= 200.0:
        return (
            "ENTRY_IQL_V2_PARTIAL_TIES_BASELINE",
            "REPAIR_ENTRY_IQL_V2_BEFORE_FURTHER_WORK",
            f"Best variant {best_variant} ties always-skip baseline "
            f"(delta {delta_skip:+.0f} bps).",
            headline,
        )
    return (
        "ENTRY_IQL_V2_PARTIAL_DEGRADES_VS_BASELINE",
        "REPAIR_ENTRY_IQL_V2_BEFORE_FURTHER_WORK",
        f"Best variant {best_variant} underperforms always-skip baseline by "
        f"{-delta_skip:+.0f} bps.",
        headline,
    )


def write_artifacts(
    forward_outcome_dir: Path = DEFAULT_FORWARD_OUTCOME_DIR,
    out_root: Path | None = None,
    *,
    sample_n_rows: int | None = None,
    variants_subset: list[str] | None = None,
    built_at_utc: str | None = None,
    balance_actions: bool = False,
    skip_weight_multiplier: float = 1.0,
    skip_oversample_factor: int = 1,
) -> dict[str, Any]:
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    per_week_dir = forward_outcome_dir / "per_week"
    if not per_week_dir.exists():
        return {
            "artifact_root": str(artifact_root),
            "summary": {
                "final_status_v1": "ENTRY_IQL_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
                "next_action_v1": "REPAIR_ENTRY_IQL_V2_BEFORE_FURTHER_WORK",
                "missing_input_v1": str(per_week_dir),
            },
        }

    print(f"[{ACTION}] loading forward-outcome dataset from {per_week_dir}", flush=True)
    df = load_forward_outcome_dataset(per_week_dir)
    print(f"[{ACTION}] loaded rows={len(df):,} cols={len(df.columns)}", flush=True)
    if sample_n_rows is not None and sample_n_rows < len(df):
        df = df.sample(n=sample_n_rows, random_state=SEED_V1).reset_index(drop=True)
        print(f"[{ACTION}] subsampled to {len(df):,} rows for smoke-test", flush=True)

    # Build state matrix once (shared across variants)
    X, feature_names = build_state_matrix(df)
    print(f"[{ACTION}] state matrix: {X.shape}, features={len(feature_names)}", flush=True)

    # Build R for each variant once
    R_by_variant: dict[str, np.ndarray] = {}
    for variant in (variants_subset or REWARD_VARIANTS):
        R_by_variant[variant] = build_reward_matrix(df, variant=variant)
        print(f"[{ACTION}] built reward matrix {variant}: shape={R_by_variant[variant].shape} "
              f"mean={float(R_by_variant[variant].mean()):.3f} max={float(R_by_variant[variant].max()):.1f}",
              flush=True)

    # Oracle action computed via R_NET_REAL @ K_PRIMARY (the realistic reward
    # variant where SKIP can win when forward path is too costly). R_RAW would
    # never let SKIP win since MFE >= 0 always — using it would give 0 SKIP
    # examples in folds and collapse the action space.
    if "R_NET_REAL" in R_by_variant:
        R_for_strat = R_by_variant["R_NET_REAL"]
        strat_variant = "R_NET_REAL"
    else:
        R_for_strat = build_reward_matrix(df, variant="R_NET_REAL")
        strat_variant = "R_NET_REAL_TEMP"
    K_primary_idx = K_HORIZONS.index(K_PRIMARY)
    oracle_action = derive_oracle_action(R_for_strat, K_primary_idx)
    oracle_dist = {ai: int((oracle_action == ai).sum()) for ai in range(iql_core.N_ACTIONS_V1)}
    print(f"[{ACTION}] oracle action (stratifier={strat_variant} K={K_PRIMARY}) distribution: {oracle_dist}", flush=True)

    # 2026-06-03 (RR4 / TUNE-03 Entry-half): chronological+embargo split is the HONEST
    # default. The legacy StratifiedShuffleSplit-by-action leaks: neighbor candidate bars
    # ~minutes apart with overlapping 8-16h forward windows straddle train/test, so its
    # "OOT" overstates edge. Chronological + embargo>=192 bars purges that overlap and makes
    # every downstream gate (Phase 6, per-side Spearman) honest. 'stratified' is still
    # selectable (GX1_ENTRY_IQL_SPLIT_MODE=stratified) ONLY to reproduce the pre-2026-06-03
    # cement bit-for-bit; it must never be used for a deploy candidate. Fail-closed: if
    # chronological is asked for but no decision-ts column exists, we RAISE (cannot do an
    # honest split without timestamps).
    _split_mode = os.environ.get("GX1_ENTRY_IQL_SPLIT_MODE", "chronological").strip().lower()
    if _split_mode == "chronological":
        _ts_col = next((c for c in ("decision_ts_utc", "decision_ts", "time", "ts_utc") if c in df.columns), None)
        if _ts_col is None:
            raise RuntimeError(f"[{ACTION}] GX1_ENTRY_IQL_SPLIT_MODE=chronological but no decision-ts "
                               f"column found in df (looked for decision_ts_utc/decision_ts/time/ts_utc)")
        _embargo = int(os.environ.get("GX1_ENTRY_IQL_EMBARGO_BARS", "192"))
        print(f"[{ACTION}] split=chronological (ts={_ts_col}, embargo={_embargo} bars) — honest OOT", flush=True)
        folds = build_chronological_folds(
            decision_ts=pd.to_datetime(df[_ts_col]).to_numpy(),
            n_folds=N_FOLDS, embargo_bars=_embargo,
        )
    else:
        folds = build_stratified_folds(
            n_rows=len(df), oracle_action=oracle_action,
            n_folds=N_FOLDS, val_size=VAL_SIZE, test_size=TEST_SIZE, seed=SEED_V1,
        )

    per_fold_results: list[dict[str, Any]] = []
    flat_evaluations: list[dict[str, Any]] = []
    for variant in (variants_subset or REWARD_VARIANTS):
        for fold in folds:
            r = evaluate_one_fold(
                fold, X, R_by_variant[variant], oracle_action,
                variant=variant, artifact_root=artifact_root,
                balance_actions=balance_actions,
                skip_weight_multiplier=skip_weight_multiplier,
                skip_oversample_factor=skip_oversample_factor,
            )
            per_fold_results.append(r)
            flat_evaluations.extend(r.get("all_evaluations_v1", []))

    _write_rows(artifact_root / "per_fold_per_variant_evaluations_v1.csv", flat_evaluations)

    status, next_action, recommendation, headline = determine_status(per_fold_results)

    summary = {
        "layer_name": "ENTRY_IQL_V2_SUMMARY_V1",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "headline_v1": headline,
        "input_forward_outcome_dir_v1": str(forward_outcome_dir),
        "n_rows_v1": int(len(df)),
        "n_features_v1": int(X.shape[1]),
        "feature_names_v1": feature_names,
        "k_horizons_v1": K_HORIZONS,
        "k_primary_v1": K_PRIMARY,
        "reward_variants_v1": list(R_by_variant.keys()),
        "oracle_action_distribution_v1": oracle_dist,
        "n_folds_v1": N_FOLDS,
        "val_size_v1": VAL_SIZE,
        "test_size_v1": TEST_SIZE,
        "seed_v1": SEED_V1,
        "min_per_class_val_v1": MIN_PER_CLASS_VAL,
        "balance_actions_v1": bool(balance_actions),
        "per_fold_summary_v1": [
            {
                "fold_id_v1": r["fold_id_v1"],
                "variant_v1": r["variant_v1"],
                "best_combo_v1": r["best_combo_v1"],
                "best_val_total_reward_bps_v1": r["best_val_total_reward_bps_v1"],
                "best_test_metric_v1": r["best_test_metric_v1"],
                "val_class_guard_status_v1": r["val_class_guard_status_v1"],
                "val_action_counts_v1": r["val_action_counts_v1"],
                "test_baselines_v1": r["test_baselines_v1"],
            }
            for r in per_fold_results
        ],
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
    }
    _write_json(artifact_root / "summary_v1.json", summary)
    _write_json(
        artifact_root / "status_v1.json",
        {
            "layer_name": "ENTRY_IQL_V2_STATUS_V1",
            "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
            "final_status_v1": status,
            "next_action_v1": next_action,
        },
    )
    _write_json(
        artifact_root / "manifest_v1.json",
        {
            "layer_id_v1": ACTION,
            "built_at_utc_v1": summary["built_at_utc_v1"],
            "output_dir_v1": str(artifact_root),
            "artifact_paths_v1": {
                "summary": str(artifact_root / "summary_v1.json"),
                "status": str(artifact_root / "status_v1.json"),
                "per_fold_csv": str(artifact_root / "per_fold_per_variant_evaluations_v1.csv"),
                "trained_models_dir": str(artifact_root / "trained_models_v1"),
            },
        },
    )
    return {"artifact_root": str(artifact_root), "summary": summary}


def main() -> None:
    global TRAIN_EPOCHS_Q, TRAIN_EPOCHS_V, TRAIN_K_VQ_ITERATIONS, TRAIN_BATCH_SIZE, TRAIN_HIDDEN_DIM, TRAIN_N_HIDDEN
    parser = argparse.ArgumentParser(description=f"Materialize {ACTION}.")
    parser.add_argument("--forward-outcome-dir", type=str, default=str(DEFAULT_FORWARD_OUTCOME_DIR))
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--sample-n-rows", type=int, default=None,
                        help="Subsample to N rows for smoke-testing (default: use all).")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-sep subset of reward variants to train (default: all 6).")
    parser.add_argument("--budget", type=str, default="fast", choices=list(BUDGET_PRESETS.keys()),
                        help="Training compute budget: fast (smoke), medium, full (production).")
    parser.add_argument("--balance-actions", action="store_true",
                        help="Inverse-frequency class weights on training rows to counter "
                             "V10 v2's 83/17 long/short skew. Required for entry-IQL retrain "
                             "after V3 v6; without it the IQL collapses to always-LONG.")
    parser.add_argument("--skip-weight-multiplier", type=float, default=1.0,
                        help="V9 Issue 1: extra multiplier on SKIP class-weight on top of "
                             "inverse-frequency (only active with --balance-actions). Wave 2 "
                             "final showed Entry-IQL chose SKIP only 0.73%% (target 5-10%%). "
                             "Try 4.0 to push effective SKIP weight from ~64x to ~256x.")
    parser.add_argument("--skip-oversample-factor", type=int, default=1,
                        help="V9 Issue 1 Plan B: replicate SKIP training rows N times. "
                             "Class-weights alone barely move SKIP-rate (4x and 10x both stay "
                             "around 1%%) because oracle SKIP has only 0.5%% of rows. "
                             "Try 10-20 to give the IQL actual gradient examples.")
    parser.add_argument("--built-at-utc", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override default SEED_V1 (used for ensemble training).")
    parser.add_argument(
        "--vedtak", type=str, default=None,
        help="Explicit user decision id authorizing this Entry-IQL build+train. "
             "REQUIRED (never auto-retrain). Any non-empty string from a deliberate human go.",
    )
    args = parser.parse_args()
    # NEVER auto-retrain — fail-closed unless an explicit --vedtak is passed.
    from gx1_guards.gates import require_retrain_vedtak, GateError
    try:
        require_retrain_vedtak(args.vedtak)
    except GateError as e:
        parser.error(str(e))
    # 2026-05-21 ensemble support: override module-level SEED_V1 so all seed-
    # dependent operations (fold split, IQL trainer, sampling) use the override.
    if args.seed is not None:
        global SEED_V1
        SEED_V1 = int(args.seed)
        print(f"[{ACTION}] SEED override: {SEED_V1}", flush=True)

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
        forward_outcome_dir=Path(args.forward_outcome_dir).expanduser().resolve(),
        out_root=out_root,
        sample_n_rows=args.sample_n_rows,
        variants_subset=variants_subset,
        built_at_utc=args.built_at_utc,
        balance_actions=args.balance_actions,
        skip_weight_multiplier=args.skip_weight_multiplier,
        skip_oversample_factor=args.skip_oversample_factor,
    )
    print(json.dumps(_jsonable(result["summary"]), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
