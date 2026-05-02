#!/usr/bin/env python3
"""Define the V2 state-feature contract for exit-side HOLD/EXIT_NOW IQL.

Background
----------
EXIT_PER_BAR_SANITY_TRAINING_V1 (gate 6) trained the first sanity IQL with
the V1 22-feature state and observed bar-0-collapse behavior on all
splits. The honest research output was: V1 is too thin for the closed-form
ridge to distinguish a good HOLD opportunity from a good EXIT_NOW. The
recommended next research gate, declared by gate 6, was
DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1.

Goal
----
Lock a V2 state-feature contract that is a SUPERSET of V1 (V1's 22 fields
remain unchanged) plus carefully chosen new fields drawn ONLY from data
sources we already have. We add four families:

  GROUP 1 (TRADE_STATE_RUNNING):     8 V1 + 4 DERIVABLE derivatives = 12
  GROUP 2 (MARKET_STATE_AT_BAR):     6 V1 + 18 BASE34_M5 HAVE        = 24
  GROUP 3 (TRANSFORMER_SIGNAL_AT_BAR):   1 V1 + 7 NOT_ESTABLISHED    =  8
  GROUP 4 (ENTRY_CONTEXT_SNAPSHOT):  3 V1 HAVE + 4 NOW HAVE          =  7

Plus 5 AUDIT_ONLY post-hoc labels - never state, never reward, never
selector. They exist for offline analysis of "should we have waited /
exited earlier / skipped this trade" and are explicitly forbidden from
being read by any policy.

The four ENTRY_CONTEXT_SNAPSHOT fields that V1 marked NOT_ESTABLISHED
(p_long_entry, p_hat_entry, uncertainty_entry, margin_entry) are now
HAVE because they were recovered offline by
RECOVER_ENTRY_SNAPSHOT_SIGNALS_FOR_EXIT_IQL_V1 from the per-week
xgb_multi_horizon_predictions parquets.

The seven GROUP 3 per-bar XGB-signal-7 fields remain NOT_ESTABLISHED:
the runtime persists XGB outputs only at trade-decision bars, not every
held bar, so per-bar XGB signal state cannot be reconstructed from
existing artifacts without an offline per-bar XGB replay - which would be
a separate gate.

Scope
-----
- Research-only contract lock. Defines the schema, audits availability
  against four pinned data sources, runs no-shortcut audit against the
  29 forbidden state fields locked in gate 1 (MDP), classifies the four
  derivable group-1 fields with their derivation recipe, and declares the
  five AUDIT_ONLY labels with their exact source formulas and forbidden
  uses.
- No training. No dataset projection. No exit_manager / live_features
  modification. No glob/latest input. No deprecated quarantine revival.
- All source columns are checked for presence, not value validity, since
  the actual feature projection happens in the next training gate.

This gate is the prerequisite for
RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.scripts import (
    materialize_build_iql_offline_data_contract_research_only_v1 as contract_gate,
)
from gx1.scripts import materialize_exit_hold_exit_now_mdp_reward_contract_v1 as mdp_gate


DEFAULT_REPORTS_ROOT = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity")
ACTION = "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1"

# Pinned upstream LOCK roots.
INPUT_V1_STATE_CONTRACT_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V1_20260429T113745Z_LOCK"
)
INPUT_PER_BAR_SCAFFOLD_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_QUALITY_DIAGNOSTIC_AND_PER_BAR_DECISION_SCAFFOLD_V1_20260429T100845Z_LOCK"
)
INPUT_RECOVERY_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "RECOVER_ENTRY_SNAPSHOT_SIGNALS_FOR_EXIT_IQL_V1_20260429T200022Z_LOCK"
)
INPUT_MDP_ROOT = (
    DEFAULT_REPORTS_ROOT
    / "EXIT_HOLD_EXIT_NOW_MDP_REWARD_CONTRACT_V1_20260429T103326Z_LOCK"
)
BASE34_M5_FEATURES_PATH = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/MONDAY_WEEK_EXTENSION_CANDIDATES/"
    "monday_week_prebuilt_extension_20260423_145325/"
    "xauusd_m5_BASE34_20250101_20260420_MODEL_BARS.parquet"
)

QUARANTINE_FORBIDDEN_PATH_FRAGMENTS = ("gx1/quarantine", "gx1.quarantine")

ALLOWED_FINAL_STATUSES = {
    "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_LOCKED_AVAILABILITY_AUDIT_PASSED",
    "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_PARTIAL_SOME_FEATURES_NOT_ESTABLISHED",
    "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_BLOCKED_BY_NO_SHORTCUT_FAIL",
    "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
}

ALLOWED_NEXT_ACTIONS = {
    "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1",
    "HOLD_UNTIL_V2_STATE_FEATURE_GAPS_RESOLVED_V1",
}


# ---------------------------------------------------------------------------
# V2 schema
# ---------------------------------------------------------------------------
#
# Each row has these keys (extending V1):
#   field_name_v2:       feature name in IQL state V2
#   field_name_v1_alias: optional, set if this field is identical to a V1 field
#   category_v2:         TRADE_STATE_RUNNING | MARKET_STATE_AT_BAR |
#                        ENTRY_CONTEXT_SNAPSHOT | TRANSFORMER_SIGNAL_AT_BAR
#   source_v2:           PER_BAR_SCAFFOLD | EXIT_EVAL_TRACE | BASE34_M5 |
#                        TRADE_OUTCOMES | ENTRY_SNAPSHOT_RECOVERY |
#                        DERIVED_FROM_PER_BAR_SCAFFOLD | NOT_ESTABLISHED
#   source_field_v2:     column name(s) in source ('' for derived/NE)
#   lineage_v2:          AS_OF_AT_BAR_T | AS_OF_AT_TRADE_OPEN |
#                        AS_OF_FROM_BARS_LE_T_MINUS_1
#   availability_v2:     HAVE | DERIVABLE | NOT_ESTABLISHED
#   normalization_v2:    PASSTHROUGH | ZSCORE_TRAIN_ONLY | ONE_HOT | LOG1P
#   derivation_recipe_v2 (only for DERIVED): exact pandas-friendly formula
#   blocking_reason_v2 (only for NOT_ESTABLISHED): why this needs another gate

PROPOSED_V2_STATE_FEATURES: list[dict[str, Any]] = [
    # ====================================================================
    # GROUP 1: TRADE_STATE_RUNNING
    # ====================================================================
    # V1 fields (carried over unchanged)
    {
        "field_name_v2": "running_pnl_at_close_bps_v1",
        "field_name_v1_alias": "running_pnl_at_close_bps_v1",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "PER_BAR_SCAFFOLD",
        "source_field_v2": "pnl_at_close_bps_v1",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "running_mfe_bps_v1",
        "field_name_v1_alias": "running_mfe_bps_v1",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "PER_BAR_SCAFFOLD",
        "source_field_v2": "running_mfe_bps_v1",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "running_mae_bps_v1",
        "field_name_v1_alias": "running_mae_bps_v1",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "PER_BAR_SCAFFOLD",
        "source_field_v2": "running_mae_bps_v1",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "running_giveback_from_peak_bps_v1",
        "field_name_v1_alias": "running_giveback_from_peak_bps_v1",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "PER_BAR_SCAFFOLD",
        "source_field_v2": "running_giveback_from_peak_bps_v1",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "bars_held_v1",
        "field_name_v1_alias": "bars_held_v1",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "PER_BAR_SCAFFOLD",
        "source_field_v2": "bar_index_v1",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "LOG1P",
    },
    {
        "field_name_v2": "distance_from_peak_mfe_bps_v1",
        "field_name_v1_alias": "distance_from_peak_mfe_bps_v1",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "EXIT_EVAL_TRACE",
        "source_field_v2": "distance_from_peak_mfe_bps",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "time_since_mfe_bars_v1",
        "field_name_v1_alias": "time_since_mfe_bars_v1",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "EXIT_EVAL_TRACE",
        "source_field_v2": "time_since_mfe_bars",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "LOG1P",
    },
    {
        "field_name_v2": "giveback_ratio_v1",
        "field_name_v1_alias": "giveback_ratio_v1",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "EXIT_EVAL_TRACE",
        "source_field_v2": "giveback_ratio",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    # New running-state derivatives (V2-only)
    {
        "field_name_v2": "pnl_velocity_bps_per_bar_v2",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "DERIVED_FROM_PER_BAR_SCAFFOLD",
        "source_field_v2": "pnl_at_close_bps_v1",
        "lineage_v2": "AS_OF_FROM_BARS_LE_T_MINUS_1",
        "availability_v2": "DERIVABLE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
        "derivation_recipe_v2": (
            "groupby(candidate_uid_v1)['pnl_at_close_bps_v1'].diff().fillna(0.0); "
            "first bar of trade has velocity = 0"
        ),
    },
    {
        "field_name_v2": "pnl_acceleration_bps_per_bar2_v2",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "DERIVED_FROM_PER_BAR_SCAFFOLD",
        "source_field_v2": "pnl_at_close_bps_v1",
        "lineage_v2": "AS_OF_FROM_BARS_LE_T_MINUS_1",
        "availability_v2": "DERIVABLE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
        "derivation_recipe_v2": (
            "groupby(candidate_uid_v1)['pnl_at_close_bps_v1'].diff().diff().fillna(0.0); "
            "first 2 bars of trade have acceleration = 0"
        ),
    },
    {
        "field_name_v2": "rolling_slope_pnl_5bars_bps_per_bar_v2",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "DERIVED_FROM_PER_BAR_SCAFFOLD",
        "source_field_v2": "pnl_at_close_bps_v1",
        "lineage_v2": "AS_OF_FROM_BARS_LE_T_MINUS_1",
        "availability_v2": "DERIVABLE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
        "derivation_recipe_v2": (
            "for each candidate_uid_v1: rolling 5-bar OLS slope of "
            "pnl_at_close_bps_v1 vs bar_index_v1; min_periods=2; first bar "
            "slope = 0 (insufficient data)"
        ),
    },
    {
        "field_name_v2": "mfe_decay_rate_3bars_bps_per_bar_v2",
        "category_v2": "TRADE_STATE_RUNNING",
        "source_v2": "DERIVED_FROM_PER_BAR_SCAFFOLD",
        "source_field_v2": "running_mfe_bps_v1",
        "lineage_v2": "AS_OF_FROM_BARS_LE_T_MINUS_1",
        "availability_v2": "DERIVABLE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
        "derivation_recipe_v2": (
            "for each candidate_uid_v1: (running_mfe_bps_v1 minus "
            "running_mfe_bps_v1.shift(3)).clip(upper=0) / 3.0; positive value "
            "always 0; only captures MFE decay"
        ),
    },
    # ====================================================================
    # GROUP 2: MARKET_STATE_AT_BAR
    # ====================================================================
    # V1 (6)
    {
        "field_name_v2": "atr_bps_now_v1",
        "field_name_v1_alias": "atr_bps_now_v1",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "atr_bps",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "session_id_v1",
        "field_name_v1_alias": "session_id_v1",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "session_id",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ONE_HOT",
    },
    {
        "field_name_v2": "vol_regime_id_v1",
        "field_name_v1_alias": "vol_regime_id_v1",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_atr_regime_id",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ONE_HOT",
    },
    {
        "field_name_v2": "trend_slope_ema3_v1",
        "field_name_v1_alias": "trend_slope_ema3_v1",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_close_ema_slope_3",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "spread_bps_dyn_v1",
        "field_name_v1_alias": "spread_bps_dyn_v1",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_cost_bps_dyn",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "minutes_since_session_open_v1",
        "field_name_v1_alias": "minutes_since_session_open_v1",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "minutes_since_session_open",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    # New BASE34 fields (18)
    {
        "field_name_v2": "minutes_to_next_session_boundary_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "minutes_to_next_session_boundary",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "session_change_flag_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "session_change_flag",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "is_asia_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "is_ASIA",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "is_eu_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_is_EU",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "is_us_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_is_US",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "session_tradable_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "session_tradable",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "atr_z_10_100_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_atr_z_10_100",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "bb_squeeze_20_2_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_bb_squeeze_20_2",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "bb_bandwidth_delta_10_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_bb_bandwidth_delta_10",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "body_share_1_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_body_share_1",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "body_tr_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_body_tr",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "clv_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_clv",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "kama_slope_30_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_kama_slope_30",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "ema_diff_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_ema_diff",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "r1_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_r1",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "r12_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_r12",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    {
        "field_name_v2": "kurt_r_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_kurt_r",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    {
        "field_name_v2": "pk_sigma20_v2",
        "category_v2": "MARKET_STATE_AT_BAR",
        "source_v2": "BASE34_M5",
        "source_field_v2": "_v1_pk_sigma20",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    # ====================================================================
    # GROUP 3: TRANSFORMER_SIGNAL_AT_BAR
    # ====================================================================
    # V1 (1)
    {
        "field_name_v2": "exit_prob_v1",
        "field_name_v1_alias": "exit_prob_v1",
        "category_v2": "TRANSFORMER_SIGNAL_AT_BAR",
        "source_v2": "EXIT_EVAL_TRACE",
        "source_field_v2": "exit_prob",
        "lineage_v2": "AS_OF_AT_BAR_T",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
    },
    # V2 NOT_ESTABLISHED: per-bar XGB signal-7 fields. The runtime computes
    # these every bar but only persists them at trade-decision moments
    # (xgb_multi_horizon_predictions per week is sparse). Reconstruction
    # requires a per-bar offline XGB replay against M5 features for every
    # bar of every held trade - that's a separate gate, not this contract
    # lock.
    *[
        {
            "field_name_v2": f"per_bar_xgb_{name}_v2",
            "category_v2": "TRANSFORMER_SIGNAL_AT_BAR",
            "source_v2": "NOT_ESTABLISHED",
            "source_field_v2": "",
            "lineage_v2": "AS_OF_AT_BAR_T",
            "availability_v2": "NOT_ESTABLISHED",
            "normalization_v2": "PASSTHROUGH",
            "blocking_reason_v2": (
                "xgb_multi_horizon_predictions parquets log XGB outputs only "
                "at trade-decision bars; per-bar XGB signal-7 for held bars "
                "requires an offline batch XGB replay against M5 BASE34 "
                "features - a separate gate. NOT permitted to substitute "
                "exit-decision XGB row for held-bar values (would be temporal "
                "shortcut)."
            ),
        }
        for name in (
            "p_long",
            "p_short",
            "p_flat",
            "p_hat",
            "uncertainty_score",
            "margin_top1_top2",
            "entropy",
        )
    ],
    # ====================================================================
    # GROUP 4: ENTRY_CONTEXT_SNAPSHOT
    # ====================================================================
    # V1 (3 HAVE)
    {
        "field_name_v2": "side_v1",
        "field_name_v1_alias": "side_v1",
        "category_v2": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v2": "TRADE_OUTCOMES",
        "source_field_v2": "side",
        "lineage_v2": "AS_OF_AT_TRADE_OPEN",
        "availability_v2": "HAVE",
        "normalization_v2": "ONE_HOT",
    },
    {
        "field_name_v2": "entry_session_v1",
        "field_name_v1_alias": "entry_session_v1",
        "category_v2": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v2": "TRADE_OUTCOMES",
        "source_field_v2": "session",
        "lineage_v2": "AS_OF_AT_TRADE_OPEN",
        "availability_v2": "HAVE",
        "normalization_v2": "ONE_HOT",
    },
    {
        "field_name_v2": "entry_spread_bps_v1",
        "field_name_v1_alias": "entry_spread_bps_v1",
        "category_v2": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v2": "TRADE_OUTCOMES",
        "source_field_v2": "entry_spread_bps",
        "lineage_v2": "AS_OF_AT_TRADE_OPEN",
        "availability_v2": "HAVE",
        "normalization_v2": "ZSCORE_TRAIN_ONLY",
    },
    # V2 NOW HAVE: 4 fields recovered by RECOVER_ENTRY_SNAPSHOT_SIGNALS_FOR_EXIT_IQL_V1
    {
        "field_name_v2": "p_long_entry_v1",
        "field_name_v1_alias": "p_long_entry_v1",
        "category_v2": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v2": "ENTRY_SNAPSHOT_RECOVERY",
        "source_field_v2": "p_long_entry_v1",
        "lineage_v2": "AS_OF_AT_TRADE_OPEN",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
        "v1_status_change_v2": "PROMOTED_FROM_NOT_ESTABLISHED_VIA_RECOVERY",
    },
    {
        "field_name_v2": "p_hat_entry_v1",
        "field_name_v1_alias": "p_hat_entry_v1",
        "category_v2": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v2": "ENTRY_SNAPSHOT_RECOVERY",
        "source_field_v2": "p_hat_entry_v1",
        "lineage_v2": "AS_OF_AT_TRADE_OPEN",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
        "v1_status_change_v2": "PROMOTED_FROM_NOT_ESTABLISHED_VIA_RECOVERY",
    },
    {
        "field_name_v2": "uncertainty_entry_v1",
        "field_name_v1_alias": "uncertainty_entry_v1",
        "category_v2": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v2": "ENTRY_SNAPSHOT_RECOVERY",
        "source_field_v2": "uncertainty_entry_v1",
        "lineage_v2": "AS_OF_AT_TRADE_OPEN",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
        "v1_status_change_v2": "PROMOTED_FROM_NOT_ESTABLISHED_VIA_RECOVERY",
    },
    {
        "field_name_v2": "margin_entry_v1",
        "field_name_v1_alias": "margin_entry_v1",
        "category_v2": "ENTRY_CONTEXT_SNAPSHOT",
        "source_v2": "ENTRY_SNAPSHOT_RECOVERY",
        "source_field_v2": "margin_entry_v1",
        "lineage_v2": "AS_OF_AT_TRADE_OPEN",
        "availability_v2": "HAVE",
        "normalization_v2": "PASSTHROUGH",
        "v1_status_change_v2": "PROMOTED_FROM_NOT_ESTABLISHED_VIA_RECOVERY",
    },
]


# ---------------------------------------------------------------------------
# Audit-only labels (post-hoc; never state, reward, or selector)
# ---------------------------------------------------------------------------
PROPOSED_AUDIT_LABELS_V2: list[dict[str, Any]] = [
    {
        "label_name_v2": "audit_delay_better_v2",
        "type_v2": "BINARY",
        "source_v2": "PER_BAR_SCAFFOLD",
        "formula_v2": (
            "for each (candidate_uid_v1, bar_index_v1): "
            "1 if max(pnl_at_close_bps_v1 over [bar+1, terminal]) > "
            "pnl_at_close_bps_v1 at bar; else 0; "
            "terminal-bar audit value is 0 by definition"
        ),
        "interpretation_v2": (
            "1 means a strictly better PnL was attained at SOME later bar "
            "in the same trade; the agent should have delayed exit beyond "
            "this bar to capture more"
        ),
        "eligibility_v2": "AUDIT_ONLY_NEVER_STATE_NEVER_REWARD_NEVER_SELECTOR",
    },
    {
        "label_name_v2": "audit_exit_earlier_better_v2",
        "type_v2": "BINARY",
        "source_v2": "PER_BAR_SCAFFOLD",
        "formula_v2": (
            "for each terminal trade: "
            "1 if max(pnl_at_close_bps_v1 over [bar 0, terminal-1]) > "
            "pnl_at_close_bps_v1 at terminal; else 0"
        ),
        "interpretation_v2": (
            "1 means an earlier exit would have realized a strictly higher "
            "PnL; the agent should have exited before terminal bar to lock "
            "the peak"
        ),
        "eligibility_v2": "AUDIT_ONLY_NEVER_STATE_NEVER_REWARD_NEVER_SELECTOR",
    },
    {
        "label_name_v2": "audit_exit_later_better_v2",
        "type_v2": "BINARY",
        "source_v2": "TRADE_OUTCOMES",
        "formula_v2": (
            "1 if post_exit_mfe_bps > 0 OR early_exit_regret == True; else 0; "
            "uses the per-trade outcome label, not the per-bar scaffold"
        ),
        "interpretation_v2": (
            "1 means there was unrealized profit available after the exit "
            "(realized by leaving the position open longer); coarse audit "
            "of post-exit drift"
        ),
        "eligibility_v2": "AUDIT_ONLY_NEVER_STATE_NEVER_REWARD_NEVER_SELECTOR",
    },
    {
        "label_name_v2": "audit_should_have_skipped_v2",
        "type_v2": "BINARY",
        "source_v2": "TRADE_OUTCOMES",
        "formula_v2": (
            "for each trade (terminal aggregate): "
            "1 if pnl_bps < 0 AND mae_bps <= -50 AND mfe_bps < 25; else 0"
        ),
        "interpretation_v2": (
            "1 means the trade was a clear loser by both PnL and MAE, with "
            "no significant up-side along the path; the agent (or entry "
            "policy) should not have taken the trade at all"
        ),
        "eligibility_v2": "AUDIT_ONLY_NEVER_STATE_NEVER_REWARD_NEVER_SELECTOR",
    },
    {
        "label_name_v2": "audit_giveback_severity_v2",
        "type_v2": "CONTINUOUS",
        "source_v2": "TRADE_OUTCOMES",
        "formula_v2": (
            "for each trade (terminal aggregate): "
            "max(0, mfe_bps - pnl_bps) / max(1.0, mfe_bps); clipped [0, 1]; "
            "0 when mfe is non-positive (no peak to give back)"
        ),
        "interpretation_v2": (
            "Continuous severity in [0, 1] of the peak-to-realized giveback. "
            "1.0 means the trade gave back ALL of its MFE; 0.0 means no "
            "giveback (closed at MFE or never had MFE)"
        ),
        "eligibility_v2": "AUDIT_ONLY_NEVER_STATE_NEVER_REWARD_NEVER_SELECTOR",
    },
]


FORBIDDEN_STATE_FIELDS_V1 = list(mdp_gate.FORBIDDEN_STATE_FIELDS_V1)


# ---------------------------------------------------------------------------
# Reused helpers
# ---------------------------------------------------------------------------

_jsonable = contract_gate._jsonable
_write_json = contract_gate._write_json
_write_rows = contract_gate._write_rows
_write_report = contract_gate._write_report
_read_json = contract_gate._read_json
_file_hash = contract_gate._file_hash
_python_manifest = contract_gate._python_manifest


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_explicit_artifact_roots(paths: Iterable[Path]) -> bool:
    return contract_gate.validate_explicit_artifact_roots(paths)


def validate_no_forbidden_actions(**kwargs: Any) -> dict[str, Any]:
    return contract_gate.validate_no_forbidden_actions(**kwargs)


def validate_final_status(status: str, next_action: str) -> bool:
    if status not in ALLOWED_FINAL_STATUSES:
        raise RuntimeError(f"FINAL_STATUS_NOT_ALLOWED: {status}")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        raise RuntimeError(f"NEXT_ACTION_NOT_ALLOWED: {next_action}")
    return True


def validate_no_deprecated_revival(script_path: Path) -> bool:
    text = script_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.lstrip()
        if not (stripped.startswith("import ") or stripped.startswith("from ")):
            continue
        for fragment in QUARANTINE_FORBIDDEN_PATH_FRAGMENTS:
            if fragment in stripped:
                raise RuntimeError("DEPRECATED_QUARANTINE_REVIVAL_FORBIDDEN")
    return True


def validate_no_shortcut(features: list[dict[str, Any]]) -> dict[str, Any]:
    field_names = [f["field_name_v2"] for f in features]
    forbidden_hits = sorted(set(field_names) & set(FORBIDDEN_STATE_FIELDS_V1))
    if forbidden_hits:
        raise RuntimeError(f"FORBIDDEN_STATE_FIELD_IN_PROPOSED_SCHEMA: {forbidden_hits}")
    forbidden_tokens = [
        "exit_reason",
        "post_exit",
        "duration_bars",
        "_replay_end_obs",
        "is_terminal",
        "bar_count",
    ]
    pattern_hits = []
    for name in field_names:
        for tok in forbidden_tokens:
            if tok in name and "exit_prob" not in name:
                pattern_hits.append(name)
                break
    if pattern_hits:
        raise RuntimeError(f"FORBIDDEN_TOKEN_IN_FIELD_NAME: {pattern_hits}")
    identity_tokens = ["candidate_uid", "trade_uid", "trade_id"]
    identity_hits = [n for n in field_names if any(tok in n for tok in identity_tokens)]
    if identity_hits:
        raise RuntimeError(f"IDENTITY_TOKEN_IN_FIELD_NAME: {identity_hits}")
    # Also assert that no audit-only token leaks into state.
    audit_tokens = ["mfe_bps", "mae_bps", "pnl_bps_terminal", "post_exit_mfe", "audit_"]
    audit_state_hits = []
    for name in field_names:
        if any(name == tok or name.endswith("_" + tok) for tok in audit_tokens):
            audit_state_hits.append(name)
        if name.startswith("audit_"):
            audit_state_hits.append(name)
    if audit_state_hits:
        raise RuntimeError(f"AUDIT_TOKEN_LEAKED_INTO_STATE: {audit_state_hits}")
    return {
        "layer_name": "DEEPEN_EXIT_IQL_STATE_NO_SHORTCUT_AUDIT_V2",
        "status_v1": "PASS",
        "feature_count_v1": len(features),
        "forbidden_field_intersection_v1": forbidden_hits,
        "forbidden_token_pattern_hits_v1": pattern_hits,
        "identity_token_hits_v1": identity_hits,
        "audit_token_state_hits_v1": audit_state_hits,
    }


def validate_v1_subset_invariant(
    v1_contract_definitions: list[dict[str, Any]],
    v2_features: list[dict[str, Any]],
) -> dict[str, Any]:
    """V2 must be a strict superset of V1: every V1 HAVE field appears in V2
    with same source_field_v1 and category_v1 (allowing field_name_v1_alias).
    Skipped V1 NOT_ESTABLISHED fields are allowed to flip to HAVE in V2.
    """
    v1_have_aliases = {
        f["field_name_v1"]: f
        for f in v1_contract_definitions
        if f.get("availability_v1") == "HAVE"
    }
    v2_aliases = {
        f.get("field_name_v1_alias"): f
        for f in v2_features
        if f.get("field_name_v1_alias")
    }
    missing = []
    drift = []
    for v1_name, v1_row in v1_have_aliases.items():
        v2_row = v2_aliases.get(v1_name)
        if v2_row is None:
            missing.append(v1_name)
            continue
        if v2_row["source_field_v2"] != v1_row.get("source_field_v1"):
            drift.append(
                {
                    "field_v1": v1_name,
                    "v1_source_field": v1_row.get("source_field_v1"),
                    "v2_source_field": v2_row["source_field_v2"],
                }
            )
    return {
        "layer_name": "DEEPEN_EXIT_IQL_V1_SUBSET_INVARIANT_AUDIT_V2",
        "status_v1": "PASS" if not missing and not drift else "FAIL",
        "v1_have_field_count_v1": len(v1_have_aliases),
        "v1_fields_missing_in_v2_v1": missing,
        "v1_to_v2_drift_v1": drift,
    }


def validate_audit_label_isolation(
    audit_labels: list[dict[str, Any]], state_features: list[dict[str, Any]]
) -> dict[str, Any]:
    state_names = {f["field_name_v2"] for f in state_features}
    audit_names = {a["label_name_v2"] for a in audit_labels}
    overlap = sorted(state_names & audit_names)
    if overlap:
        raise RuntimeError(f"AUDIT_LABEL_NAME_LEAKED_INTO_STATE: {overlap}")
    bad_eligibility = [
        a["label_name_v2"]
        for a in audit_labels
        if a.get("eligibility_v2")
        != "AUDIT_ONLY_NEVER_STATE_NEVER_REWARD_NEVER_SELECTOR"
    ]
    if bad_eligibility:
        raise RuntimeError(f"AUDIT_LABEL_BAD_ELIGIBILITY: {bad_eligibility}")
    return {
        "layer_name": "DEEPEN_EXIT_IQL_AUDIT_LABEL_ISOLATION_V2",
        "status_v1": "PASS",
        "state_feature_count_v1": len(state_names),
        "audit_label_count_v1": len(audit_names),
        "audit_state_overlap_v1": overlap,
        "bad_eligibility_v1": bad_eligibility,
    }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _per_bar_decision_dataset_path() -> Path:
    return (
        INPUT_PER_BAR_SCAFFOLD_ROOT
        / "PER_BAR_TRAJECTORY_V1"
        / "per_bar_decision_dataset_v1.parquet"
    )


def _exit_eval_trace_paths() -> list[Path]:
    return sorted(
        DEFAULT_REPORTS_ROOT.glob(
            "TRUTH_MONFRI_WEEK_*/replay/chunk_0/EXIT_EVAL_TRACE.csv"
        ),
        key=lambda p: p.parent.parent.parent.name,
    )


def _trade_outcomes_first_nonempty() -> Path:
    weeks = sorted(
        DEFAULT_REPORTS_ROOT.glob(
            "TRUTH_MONFRI_WEEK_*/trade_outcomes_*_MERGED.parquet"
        )
    )
    for w in weeks:
        if not pd.read_parquet(w).empty:
            return w
    raise RuntimeError("NO_NONEMPTY_TRADE_OUTCOMES_PARQUET_FOUND")


def _recovery_per_trade_path() -> Path:
    return INPUT_RECOVERY_ROOT / "entry_snapshot_signals_per_trade_v1.parquet"


def _v1_state_contract_path() -> Path:
    return INPUT_V1_STATE_CONTRACT_ROOT / "state_feature_contract_v1.json"


def _load_inputs() -> dict[str, Any]:
    roots = [
        INPUT_V1_STATE_CONTRACT_ROOT,
        INPUT_PER_BAR_SCAFFOLD_ROOT,
        INPUT_RECOVERY_ROOT,
        INPUT_MDP_ROOT,
    ]
    validate_explicit_artifact_roots(roots)
    required = {
        "v1_state_contract": _v1_state_contract_path(),
        "per_bar_decision_dataset": _per_bar_decision_dataset_path(),
        "recovery_per_trade": _recovery_per_trade_path(),
        "recovery_summary": INPUT_RECOVERY_ROOT / "summary_v1.json",
        "mdp_no_shortcut_axioms": INPUT_MDP_ROOT / "no_shortcut_axioms_v1.json",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"MISSING_REQUIRED_INPUT_LOCKS: {missing}")
    if not BASE34_M5_FEATURES_PATH.exists():
        raise RuntimeError(
            f"BASE34_M5_FEATURES_PATH_NOT_FOUND: {BASE34_M5_FEATURES_PATH}"
        )
    trace_paths = _exit_eval_trace_paths()
    if not trace_paths:
        raise RuntimeError("NO_EXIT_EVAL_TRACE_FILES_FOUND")
    return {
        "required_paths": required,
        "exit_eval_trace_paths": trace_paths,
        "base34_path": BASE34_M5_FEATURES_PATH,
        "v1_state_contract": _read_json(required["v1_state_contract"]),
        "recovery_summary": _read_json(required["recovery_summary"]),
    }


# ---------------------------------------------------------------------------
# Availability audit
# ---------------------------------------------------------------------------


def _audit_availability(
    inputs: dict[str, Any], features: list[dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    per_bar_df = pd.read_parquet(_per_bar_decision_dataset_path())
    per_bar_cols = set(per_bar_df.columns)
    trace_paths = inputs["exit_eval_trace_paths"]
    trace_sample = pd.read_csv(trace_paths[0], nrows=10)
    trace_cols = set(trace_sample.columns)
    base34_df = pd.read_parquet(BASE34_M5_FEATURES_PATH)
    base34_cols = set(base34_df.columns)
    trade_outcomes_path = _trade_outcomes_first_nonempty()
    trade_outcomes_cols = set(pd.read_parquet(trade_outcomes_path).columns)
    recovery_df = pd.read_parquet(_recovery_per_trade_path())
    recovery_cols = set(recovery_df.columns)

    source_cols_map = {
        "PER_BAR_SCAFFOLD": per_bar_cols,
        "EXIT_EVAL_TRACE": trace_cols,
        "BASE34_M5": base34_cols,
        "TRADE_OUTCOMES": trade_outcomes_cols,
        "ENTRY_SNAPSHOT_RECOVERY": recovery_cols,
        "DERIVED_FROM_PER_BAR_SCAFFOLD": per_bar_cols,
        "NOT_ESTABLISHED": set(),
    }
    have: list[str] = []
    derivable: list[str] = []
    not_established: list[str] = []
    for feat in features:
        source = feat["source_v2"]
        source_field = feat["source_field_v2"]
        availability = feat["availability_v2"]
        verified = (
            (source_field in source_cols_map.get(source, set()))
            if source_field
            else (source == "DERIVED_FROM_PER_BAR_SCAFFOLD")
        )
        if source == "DERIVED_FROM_PER_BAR_SCAFFOLD" and source_field:
            verified = source_field in per_bar_cols
        row = {
            **feat,
            "source_field_present_in_source_v1": verified,
        }
        if availability == "NOT_ESTABLISHED":
            not_established.append(feat["field_name_v2"])
            row["audit_status_v1"] = "DECLARED_NOT_ESTABLISHED"
        elif availability == "HAVE":
            if not verified:
                row["audit_status_v1"] = "AUDIT_FAIL_FIELD_NOT_IN_SOURCE"
                row["availability_v2"] = "NOT_ESTABLISHED"
                not_established.append(feat["field_name_v2"])
            else:
                row["audit_status_v1"] = "AUDIT_PASS"
                have.append(feat["field_name_v2"])
        elif availability == "DERIVABLE":
            if not verified:
                row["audit_status_v1"] = "AUDIT_FAIL_DERIVATION_SOURCE_MISSING"
                row["availability_v2"] = "NOT_ESTABLISHED"
                not_established.append(feat["field_name_v2"])
            else:
                row["audit_status_v1"] = "DERIVABLE_REQUIRES_NEXT_GATE_COMPUTATION"
                derivable.append(feat["field_name_v2"])
        rows.append(row)
    return {
        "layer_name": "DEEPEN_EXIT_IQL_FEATURE_AVAILABILITY_AUDIT_V2",
        "feature_rows_v1": rows,
        "have_count_v1": len(have),
        "have_field_names_v1": sorted(have),
        "derivable_count_v1": len(derivable),
        "derivable_field_names_v1": sorted(derivable),
        "not_established_count_v1": len(not_established),
        "not_established_field_names_v1": sorted(not_established),
        "source_column_counts_v1": {k: len(v) for k, v in source_cols_map.items()},
    }


def _audit_label_source_coverage(
    audit_labels: list[dict[str, Any]],
) -> dict[str, Any]:
    per_bar_cols = set(pd.read_parquet(_per_bar_decision_dataset_path()).columns)
    trade_outcomes_path = _trade_outcomes_first_nonempty()
    trade_outcomes_cols = set(pd.read_parquet(trade_outcomes_path).columns)

    expected_per_bar_fields = {
        "audit_delay_better_v2": ["pnl_at_close_bps_v1", "candidate_uid_v1", "bar_index_v1"],
        "audit_exit_earlier_better_v2": [
            "pnl_at_close_bps_v1",
            "candidate_uid_v1",
            "bar_index_v1",
            "is_terminal_v1",
        ],
    }
    expected_trade_outcomes_fields = {
        "audit_exit_later_better_v2": ["post_exit_mfe_bps", "early_exit_regret"],
        "audit_should_have_skipped_v2": ["pnl_bps", "mae_bps", "mfe_bps"],
        "audit_giveback_severity_v2": ["pnl_bps", "mfe_bps"],
    }
    rows: list[dict[str, Any]] = []
    for lbl in audit_labels:
        name = lbl["label_name_v2"]
        if name in expected_per_bar_fields:
            required = expected_per_bar_fields[name]
            missing = [c for c in required if c not in per_bar_cols]
            rows.append(
                {
                    **lbl,
                    "required_columns_v1": required,
                    "source_columns_present_v1": not missing,
                    "missing_columns_v1": missing,
                }
            )
        elif name in expected_trade_outcomes_fields:
            required = expected_trade_outcomes_fields[name]
            missing = [c for c in required if c not in trade_outcomes_cols]
            rows.append(
                {
                    **lbl,
                    "required_columns_v1": required,
                    "source_columns_present_v1": not missing,
                    "missing_columns_v1": missing,
                }
            )
        else:
            rows.append(
                {
                    **lbl,
                    "required_columns_v1": [],
                    "source_columns_present_v1": True,
                    "missing_columns_v1": [],
                }
            )
    all_present = all(r["source_columns_present_v1"] for r in rows)
    return {
        "layer_name": "DEEPEN_EXIT_IQL_AUDIT_LABEL_COVERAGE_V2",
        "status_v1": "PASS" if all_present else "FAIL",
        "rows_v1": rows,
    }


# ---------------------------------------------------------------------------
# Reproducibility / go-no-go
# ---------------------------------------------------------------------------


def _reproducibility_audit(
    availability_audit: dict[str, Any],
    audit_label_coverage: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "layer_name": "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_REPRODUCIBILITY_AUDIT_V2",
        "feature_count_total_v1": len(PROPOSED_V2_STATE_FEATURES),
        "feature_count_have_v1": availability_audit["have_count_v1"],
        "feature_count_derivable_v1": availability_audit["derivable_count_v1"],
        "feature_count_not_established_v1": availability_audit["not_established_count_v1"],
        "audit_label_count_v1": len(PROPOSED_AUDIT_LABELS_V2),
        "audit_label_coverage_status_v1": audit_label_coverage["status_v1"],
        "no_implicit_glob_used_for_v1_inputs_v1": True,
        "deprecated_quarantine_revival_v1": False,
        "research_only_v1": True,
        "recovery_match_rate_v1": inputs["recovery_summary"].get("match_rate_v1"),
        "recovery_total_v1": inputs["recovery_summary"].get("total_trade_count_v1"),
        "recovery_matched_v1": inputs["recovery_summary"].get("matched_trade_count_v1"),
    }


def _go_no_go(
    availability_audit: dict[str, Any],
    no_shortcut_audit: dict[str, Any],
    audit_label_isolation: dict[str, Any],
    audit_label_coverage: dict[str, Any],
    v1_subset: dict[str, Any],
) -> tuple[str, str, str]:
    if no_shortcut_audit["status_v1"] != "PASS":
        return (
            "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_BLOCKED_BY_NO_SHORTCUT_FAIL",
            "HOLD_UNTIL_V2_STATE_FEATURE_GAPS_RESOLVED_V1",
            "No-shortcut audit failed; resolve before proceeding.",
        )
    if v1_subset["status_v1"] != "PASS":
        return (
            "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_BLOCKED_BY_NO_SHORTCUT_FAIL",
            "HOLD_UNTIL_V2_STATE_FEATURE_GAPS_RESOLVED_V1",
            (
                "V1-subset invariant audit failed: V2 must be a strict superset "
                f"of V1. Missing: {v1_subset['v1_fields_missing_in_v2_v1']}; "
                f"drift: {v1_subset['v1_to_v2_drift_v1']}"
            ),
        )
    if audit_label_isolation["status_v1"] != "PASS":
        return (
            "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_BLOCKED_BY_NO_SHORTCUT_FAIL",
            "HOLD_UNTIL_V2_STATE_FEATURE_GAPS_RESOLVED_V1",
            "Audit-label isolation failed.",
        )
    if audit_label_coverage["status_v1"] != "PASS":
        return (
            "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_BLOCKED_BY_INPUT_LOCK_MISSING",
            "HOLD_UNTIL_V2_STATE_FEATURE_GAPS_RESOLVED_V1",
            "Audit-label source coverage failed: a label requires a column not present in source.",
        )
    have = availability_audit["have_count_v1"]
    derivable = availability_audit["derivable_count_v1"]
    not_established = availability_audit["not_established_count_v1"]
    if not_established == 0:
        return (
            "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_LOCKED_AVAILABILITY_AUDIT_PASSED",
            "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1",
            (
                f"V2 schema locked: {have} HAVE + {derivable} DERIVABLE state "
                "fields, no NOT_ESTABLISHED gaps. Next: V2 contextual IQL "
                "training with reward variants."
            ),
        )
    return (
        "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V2_PARTIAL_SOME_FEATURES_NOT_ESTABLISHED",
        "RUN_CONTEXTUAL_IQL_WITH_V2_STATE_AND_REWARD_VARIANTS_V1",
        (
            f"V2 schema locked: {have} HAVE + {derivable} DERIVABLE state fields "
            f"+ {not_established} NOT_ESTABLISHED gaps. The NOT_ESTABLISHED "
            "fields (per-bar XGB signal-7) require an offline batch XGB "
            "replay against M5 features at every held bar - a separate gate "
            "before they can be added. Training-with-V2 may proceed without "
            "them; the per-bar XGB signal-7 lift can be measured as an "
            "ablation in a later sweep."
        ),
    )


def _build_input_manifest(
    inputs: dict[str, Any], artifact_root: Path
) -> dict[str, Any]:
    files = [
        {
            "name_v1": name,
            "path_v1": str(path),
            "sha256_v1": _file_hash(path),
        }
        for name, path in inputs["required_paths"].items()
    ]
    files.append(
        {
            "name_v1": "base34_m5_features",
            "path_v1": str(inputs["base34_path"]),
            "sha256_v1": _file_hash(inputs["base34_path"]),
        }
    )
    files.append(
        {
            "name_v1": "exit_eval_trace_first_path",
            "path_v1": str(inputs["exit_eval_trace_paths"][0]),
            "sha256_v1": _file_hash(inputs["exit_eval_trace_paths"][0]),
        }
    )
    return {
        "layer_name": "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_INPUT_MANIFEST_V2",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "created_at_utc_v1": _utc_now(),
        "input_roots_v1": {
            "v1_state_contract_root_v1": str(INPUT_V1_STATE_CONTRACT_ROOT),
            "per_bar_scaffold_root_v1": str(INPUT_PER_BAR_SCAFFOLD_ROOT),
            "recovery_root_v1": str(INPUT_RECOVERY_ROOT),
            "mdp_root_v1": str(INPUT_MDP_ROOT),
        },
        "raw_data_v1": {
            "base34_m5_v1": str(BASE34_M5_FEATURES_PATH),
            "exit_eval_trace_path_count_v1": len(inputs["exit_eval_trace_paths"]),
        },
        "files_used_v1": files,
        "immutable_input_status_v1": "HASHED_EXPLICIT_ROOTS_ONLY",
        "no_implicit_latest_glob_selection_v1": True,
        "previous_artifacts_mutated_v1": False,
        "research_only_contract_v1": True,
        "iql_training_run_v1": False,
        "iql_production_allowed_v1": False,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "package_built_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "v1_state_contract_modified_v1": False,
        "python_manifest_v1": _python_manifest(),
    }


# ---------------------------------------------------------------------------
# Materializer
# ---------------------------------------------------------------------------


def write_artifacts(
    out_root: Path | None = None,
    *,
    built_at_utc: str | None = None,
) -> dict[str, Any]:
    inputs = _load_inputs()
    timestamp = built_at_utc or _stamp()
    artifact_root = out_root or (DEFAULT_REPORTS_ROOT / f"{ACTION}_{timestamp}_LOCK")
    artifact_root.mkdir(parents=True, exist_ok=True)

    validate_no_deprecated_revival(Path(__file__))
    forbidden_audit = validate_no_forbidden_actions(
        adapter=False,
        r6=False,
        iql_production=False,
        package=False,
        freeze=False,
        promo=False,
        live=False,
        optuna=False,
        broad_sweep=False,
    )
    _write_json(
        artifact_root / "input_manifest_v1.json",
        _build_input_manifest(inputs, artifact_root),
    )

    no_shortcut_audit = validate_no_shortcut(PROPOSED_V2_STATE_FEATURES)
    _write_json(artifact_root / "no_shortcut_audit_v2.json", no_shortcut_audit)

    v1_subset = validate_v1_subset_invariant(
        inputs["v1_state_contract"]["feature_definitions_v1"],
        PROPOSED_V2_STATE_FEATURES,
    )
    _write_json(artifact_root / "v1_subset_invariant_audit_v2.json", v1_subset)

    audit_label_isolation = validate_audit_label_isolation(
        PROPOSED_AUDIT_LABELS_V2, PROPOSED_V2_STATE_FEATURES
    )
    _write_json(
        artifact_root / "audit_label_isolation_audit_v2.json", audit_label_isolation
    )

    audit_label_coverage = _audit_label_source_coverage(PROPOSED_AUDIT_LABELS_V2)
    _write_json(
        artifact_root / "audit_label_source_coverage_v2.json", audit_label_coverage
    )

    availability_audit = _audit_availability(inputs, PROPOSED_V2_STATE_FEATURES)
    _write_json(artifact_root / "availability_audit_v2.json", availability_audit)
    _write_rows(
        artifact_root / "feature_availability_table_v2.csv",
        availability_audit["feature_rows_v1"],
    )

    state_contract_v2 = {
        "layer_name": "EXIT_PER_BAR_STATE_FEATURE_CONTRACT_V2",
        "schema_version_v1": "V2",
        "v1_predecessor_root_v1": str(INPUT_V1_STATE_CONTRACT_ROOT),
        "feature_count_v1": len(PROPOSED_V2_STATE_FEATURES),
        "trainable_have_count_v1": availability_audit["have_count_v1"],
        "derivable_count_v1": availability_audit["derivable_count_v1"],
        "not_established_count_v1": availability_audit["not_established_count_v1"],
        "feature_definitions_v1": PROPOSED_V2_STATE_FEATURES,
        "audit_only_labels_v1": PROPOSED_AUDIT_LABELS_V2,
        "category_counts_v1": {
            cat: sum(
                1 for f in PROPOSED_V2_STATE_FEATURES if f["category_v2"] == cat
            )
            for cat in [
                "TRADE_STATE_RUNNING",
                "MARKET_STATE_AT_BAR",
                "ENTRY_CONTEXT_SNAPSHOT",
                "TRANSFORMER_SIGNAL_AT_BAR",
            ]
        },
        "samstemte_alignment_field_v1": "exit_prob_v1",
        "research_only_v1": True,
    }
    _write_json(artifact_root / "state_feature_contract_v2.json", state_contract_v2)

    # Also write the diff vs V1 for transparency.
    v1_def_names = {
        f["field_name_v1"]
        for f in inputs["v1_state_contract"]["feature_definitions_v1"]
    }
    v2_alias_map = {
        f.get("field_name_v1_alias"): f for f in PROPOSED_V2_STATE_FEATURES
    }
    diff_rows: list[dict[str, Any]] = []
    for f in inputs["v1_state_contract"]["feature_definitions_v1"]:
        v1_name = f["field_name_v1"]
        v2_row = v2_alias_map.get(v1_name)
        diff_rows.append(
            {
                "field_v1": v1_name,
                "v1_availability": f["availability_v1"],
                "v2_field_name": v2_row["field_name_v2"] if v2_row else "",
                "v2_availability": v2_row["availability_v2"] if v2_row else "REMOVED",
                "v2_status_change": v2_row.get("v1_status_change_v2", "")
                if v2_row
                else "REMOVED",
                "v2_source": v2_row["source_v2"] if v2_row else "",
            }
        )
    for f in PROPOSED_V2_STATE_FEATURES:
        if f.get("field_name_v1_alias") in v1_def_names:
            continue
        diff_rows.append(
            {
                "field_v1": "",
                "v1_availability": "ABSENT_IN_V1",
                "v2_field_name": f["field_name_v2"],
                "v2_availability": f["availability_v2"],
                "v2_status_change": "NEW_IN_V2",
                "v2_source": f["source_v2"],
            }
        )
    _write_rows(artifact_root / "v1_to_v2_diff_v2.csv", diff_rows)

    repro = _reproducibility_audit(availability_audit, audit_label_coverage, inputs)
    _write_json(artifact_root / "reproducibility_audit_v1.json", repro)

    status, next_action, recommendation = _go_no_go(
        availability_audit,
        no_shortcut_audit,
        audit_label_isolation,
        audit_label_coverage,
        v1_subset,
    )
    validate_final_status(status, next_action)

    summary = {
        "layer_name": "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_SUMMARY_V2",
        "action_v1": ACTION,
        "artifact_root_v1": str(artifact_root),
        "built_at_utc_v1": _utc_now(),
        "final_status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "feature_count_total_v1": len(PROPOSED_V2_STATE_FEATURES),
        "feature_count_have_v1": availability_audit["have_count_v1"],
        "feature_count_derivable_v1": availability_audit["derivable_count_v1"],
        "feature_count_not_established_v1": availability_audit["not_established_count_v1"],
        "audit_label_count_v1": len(PROPOSED_AUDIT_LABELS_V2),
        "category_counts_v1": state_contract_v2["category_counts_v1"],
        "samstemte_alignment_v1": "exit_prob_v1 included as TRANSFORMER_SIGNAL_AT_BAR",
        "v1_subset_invariant_status_v1": v1_subset["status_v1"],
        "audit_label_isolation_status_v1": audit_label_isolation["status_v1"],
        "audit_label_coverage_status_v1": audit_label_coverage["status_v1"],
        "no_shortcut_audit_status_v1": no_shortcut_audit["status_v1"],
        "recovery_match_rate_v1": inputs["recovery_summary"].get("match_rate_v1"),
        "research_only_v1": True,
        "iql_training_run_v1": False,
        "training_blocked_v1": True,
        "next_pre_train_gate_v1": next_action,
        "adapter_built_v1": False,
        "r6_run_v1": False,
        "freeze_promo_live_run_v1": False,
        "deprecated_quarantine_revival_v1": False,
        "exit_manager_modified_v1": False,
        "live_features_modified_v1": False,
        "entry_manager_modified_v1": False,
        "v1_state_contract_modified_v1": False,
        "forbidden_actions_audit_v1": forbidden_audit,
    }
    _write_json(artifact_root / "summary_v1.json", summary)

    status_payload = {
        "layer_name": "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_STATUS_V2",
        "status_v1": "MATERIALIZED_RESEARCH_ONLY_GATE",
        "final_status_v1": status,
        "next_action_v1": next_action,
        "training_executed_v1": False,
    }
    _write_json(artifact_root / "status_v1.json", status_payload)

    go_no_go = {
        "layer_name": "DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_GO_NO_GO_V2",
        "status_v1": status,
        "next_action_v1": next_action,
        "recommendation_v1": recommendation,
        "research_only_v1": True,
        "iql_production_allowed_v1": False,
        "adapter_build_allowed_v1": False,
        "r6_allowed_v1": False,
        "package_freeze_promo_live_allowed_v1": False,
        "policy_promotion_allowed_v1": False,
        "training_allowed_v1": False,
        "downstream_block_v1": (
            "Research-only V2 contract lock. No training. Adapter/R6/IQL "
            "production/live, freeze/promo/live, exit_manager modification, "
            "entry_manager modification, V1 state contract modification all "
            "forbidden."
        ),
    }
    _write_json(
        artifact_root / "deepen_exit_iql_state_feature_family_go_no_go_v2.json",
        go_no_go,
    )

    report_lines = [
        "# Deepen Exit IQL State Feature Family V2",
        "",
        f"- Status: `{status}`",
        f"- Next action: `{next_action}`",
        "- Training: **BLOCKED** (research-only contract lock).",
        "",
        "## V2 schema summary",
        f"- Total features: {len(PROPOSED_V2_STATE_FEATURES)}",
        f"- HAVE: {availability_audit['have_count_v1']}",
        f"- DERIVABLE: {availability_audit['derivable_count_v1']}",
        f"- NOT_ESTABLISHED: {availability_audit['not_established_count_v1']}",
        f"- Audit-only labels: {len(PROPOSED_AUDIT_LABELS_V2)}",
        "",
        "## Category counts",
    ]
    for cat, count in state_contract_v2["category_counts_v1"].items():
        report_lines.append(f"- `{cat}`: {count}")
    report_lines.extend(
        [
            "",
            "## V1 -> V2 promotions",
            "- `p_long_entry_v1`, `p_hat_entry_v1`, `uncertainty_entry_v1`, "
            "`margin_entry_v1`: PROMOTED_FROM_NOT_ESTABLISHED_VIA_RECOVERY "
            f"(match rate {inputs['recovery_summary'].get('match_rate_v1')}).",
            "",
            "## Still NOT_ESTABLISHED",
        ]
    )
    for name in availability_audit["not_established_field_names_v1"]:
        report_lines.append(f"- `{name}`")
    report_lines.extend(
        [
            "",
            "## Audit-only labels",
        ]
    )
    for lbl in PROPOSED_AUDIT_LABELS_V2:
        report_lines.append(
            f"- `{lbl['label_name_v2']}` ({lbl['type_v2']}, source `{lbl['source_v2']}`)"
        )
    report_lines.extend(
        [
            "",
            "## Recommendation",
            recommendation,
        ]
    )
    _write_report(artifact_root / "report_v1.md", report_lines)

    artifact_manifest = {
        "layer_id_v1": ACTION,
        "built_at_utc_v1": summary["built_at_utc_v1"],
        "output_dir_v1": str(artifact_root),
        "append_only_namespace_v1": "truth_e2e_sanity",
        "artifact_paths_v1": {
            "summary": str(artifact_root / "summary_v1.json"),
            "status": str(artifact_root / "status_v1.json"),
            "go_no_go": str(
                artifact_root
                / "deepen_exit_iql_state_feature_family_go_no_go_v2.json"
            ),
            "input_manifest": str(artifact_root / "input_manifest_v1.json"),
            "state_feature_contract_v2": str(
                artifact_root / "state_feature_contract_v2.json"
            ),
            "availability_audit_v2": str(artifact_root / "availability_audit_v2.json"),
            "feature_availability_table_v2_csv": str(
                artifact_root / "feature_availability_table_v2.csv"
            ),
            "v1_to_v2_diff_v2_csv": str(artifact_root / "v1_to_v2_diff_v2.csv"),
            "no_shortcut_audit_v2": str(artifact_root / "no_shortcut_audit_v2.json"),
            "v1_subset_invariant_audit_v2": str(
                artifact_root / "v1_subset_invariant_audit_v2.json"
            ),
            "audit_label_isolation_audit_v2": str(
                artifact_root / "audit_label_isolation_audit_v2.json"
            ),
            "audit_label_source_coverage_v2": str(
                artifact_root / "audit_label_source_coverage_v2.json"
            ),
            "reproducibility_audit": str(artifact_root / "reproducibility_audit_v1.json"),
            "report": str(artifact_root / "report_v1.md"),
        },
        "read_only_references_v1": True,
        "not_trainer_v1": True,
        "not_controller_v1": True,
        "not_live_gate_v1": True,
    }
    _write_json(artifact_root / "manifest_v1.json", artifact_manifest)

    return {
        "artifact_root": str(artifact_root),
        "summary": summary,
        "status": status_payload,
        "go_no_go": go_no_go,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize DEEPEN_EXIT_IQL_STATE_FEATURE_FAMILY_V1 gate."
    )
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--built-at-utc", type=str, default=None)
    args = parser.parse_args()
    out_root = (
        Path(args.out_root).expanduser().resolve() if args.out_root else None
    )
    result = write_artifacts(out_root=out_root, built_at_utc=args.built_at_utc)
    print(json.dumps(_jsonable(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
