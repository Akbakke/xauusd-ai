#!/usr/bin/env python3
# ruff: noqa: E402
"""Build the exact model-native ENTRY_V10_CTX training dataset.

The emitted Entry surface is contract-locked to 34 genuine per-bar price-state
signals plus 479 selected specialist signals (513 total), a 96-bar sequence,
142 continuous context values, and five categorical context values.  Structure,
trend, liquidity, volatility, momentum, session, price action, path quality and
utility evidence remain learned inputs/targets; no external direction model or
runtime direction filter participates in this builder.

The exact canonical source parquet, canonical-v2 feature frame, exact
model-native selection manifest and the bid/ask market tape are mandatory.
Missing or mismatched contracts fail closed.  No compatibility fallback exists.

SECTION LINE INDEX (oppdater ved flytting; se ogsaa SYSTEM_MAP.md
"Pipeline- og ingredienskart"):
  ~594  exact model-native ctx142/cat5 gate
  ~628  _signal_build_contract_from_manifest
  ~674  _build_inline_seq_structure_extension (alle specialist-lag fra merged3)
  ~1799 is_ASIA-derivering ((session_id==0).astype(int8))
  ~1835 Entry-owned exact context ordering
  ~1982 df_ctx_cont-konstruksjon
  ~2096 merged3-assembly (labels/path/bad-path + ctx-join)
  ~2115 cv2-lasteliste (V3-navn minus computed-familier)
  ~2244 GROUP_A/DIP_STRUCT-attach (krever env GX1_V10_MULTI_TF_V4_CACHE_DIR;
        85 ms/rad serielt — se parallellmoenster i ranker-scriptet)
  ~2330 entry_smart_context (ENTRY_SMART_DERIVED)
  ~2337 ctx-komplett-sjekk (alle 142 maa finnes)
  ~3440 argparse
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import hashlib
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_CTX_CONT_ENTRY_SMART_DERIVED_FIELDS,
    MODEL_NATIVE_CTX_CONT_MICRO_FIELDS,
    MODEL_NATIVE_CTX_CONT_SESSION_FIELDS,
    MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS,
    MODEL_NATIVE_CTX_CONT_SWING_FIELDS,
    MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS,
    MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    model_native_context_contract_metadata,
    require_model_native_manifest,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_offline_rl_v1 import (
    HORIZON_BARS as OFFLINE_RL_HORIZON_BARS,
    UTILITY_MAE_WEIGHT,
    UTILITY_MFE_WEIGHT,
    UTILITY_PATH_WEIGHT,
)
from gx1.contracts.entry_structural_aux_label_signal_v1 import (
    STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS,
)
from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_CLOSURE_CONTRACT,
    validate_xau_tape_provenance_v1,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_AUX_FORECAST_HORIZONS,
    MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
    MODEL_NATIVE_AUX_RISK_HORIZONS,
    MODEL_NATIVE_AUX_TARGET_COLUMNS,
    MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN,
    MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION,
    MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS,
    MODEL_NATIVE_DIP_MAE_UPPER_SAFETY_CAP_BPS,
    MODEL_NATIVE_DIP_MFE_TARGET_COLUMNS,
    MODEL_NATIVE_DIP_MFE_UPPER_SAFETY_CAP_BPS,
    MODEL_NATIVE_TAIL_MAE_UPPER_SAFETY_CAP_BPS,
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
    apply_train_rank_reference_v2,
    load_train_rank_reference_v2,
    require_train_rank_source_market_identity_v2,
    validate_state_contract_metadata_v2,
)
from gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1 import (
    validate_signal_manifest_training_lineage,
)
from gx1.features.micro_structure_v1 import compute_micro_structure_features
from gx1.features.swing_structure_v1 import (
    SWING_ATR_PERIOD_V1,
    SWING_LOOKBACK_V1,
    compute_swing_structure_features,
)
from gx1.time.session_detector import (
    ASIA_SESSION_ID,
    SESSION_NAME_BY_ID,
    get_session_id_vectorized,
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    publish_bundle_directory_noreplace,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SHORT_INDEX,
    UNIFIED_EXIT_ACTION_ORDER,
    UNIFIED_EXIT_MAX_PATH_BARS,
    UNIFIED_EXIT_SIDE_ORDER,
)
from gx1.contracts.unified_exit_lifecycle_v1 import (
    UNIFIED_EXIT_LIFECYCLE_EPISODE_COLUMNS,
    UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
    UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS,
    canonical_json_sha256,
    require_unified_exit_m1_pair_authority,
    unified_exit_future_extrema as _unified_exit_future_extrema,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_FEATURE_ROW_CLOCK,
    entry_exit_shared_feature_base_contract,
    require_entry_exit_feature_surface_identity,
)
from gx1.contracts.entry_exit_feature_surface_v1 import (
    ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE,
    ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION,
    load_m1_feature_surface,
    load_m1_feature_surface_times,
)
from gx1.io.price_glitch_guard import assert_no_price_scale_glitch

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

PATH_QUALITY_HORIZON_BARS = 10
BAD_PATH_HORIZON_BARS = PATH_QUALITY_HORIZON_BARS
BAD_PATH_MAE_THRESHOLD_BPS = 6.0
BAD_PATH_MFE_THRESHOLD_BPS = 4.0
# 2026-05-26 — entry direction/tradable label re-tuned (user: "89% flat uaktuelt").
# The directional label = (pnl_at_horizon >= V11_TRADABLE_PNL_MIN_BPS) over
# V11_DIRECTION_HORIZON_BARS. Lowered 30→15 bps + horizon 10→24 (2h) → ~60% flat
# (was ~89%) so the model has real directional signal to learn + calibrate on.
# 15 bps ≈ 11 bps net after ~2.25 bps spread — a solid edge, not noise. bad_path /
# path_quality keep their own (10-bar) horizon — only the direction label changed.
V11_TRADABLE_PNL_MIN_BPS = 15.0
V11_DIRECTION_HORIZON_BARS = 24
# One immutable supervision contract.  These are future-outcome label semantics,
# never live direction rules, and callers cannot tune or replace them.
V12_DIRECTION_TARGET_MODE = "path_utility_v2"
# Compatibility names remain imported by immutable audit code, but the
# offline-RL contract is the single numerical owner of path-utility weights.
V12_DIRECTION_UTILITY_MFE_WEIGHT = UTILITY_MFE_WEIGHT
V12_DIRECTION_UTILITY_MAE_WEIGHT = UTILITY_MAE_WEIGHT
V12_DIRECTION_UTILITY_PATH_WEIGHT = UTILITY_PATH_WEIGHT
V12_DIRECTION_UTILITY_MIN_BPS = 15.0
V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS = 4.0

HARD_NEG_LONG_MIN_MFE_BPS = 10.0
HARD_NEG_LONG_MIN_MAE_BPS = 6.0
HARD_NEG_LONG_MAX_PATH_BPS = 8.0
DEAD_LONG_MAX_MFE_BPS = 0.5
DEAD_LONG_MIN_MAE_BPS = 6.0
TEASER_LONG_MIN_MFE_BPS = 0.5
TEASER_LONG_MAX_MFE_BPS = 10.0
TEASER_LONG_MIN_MAE_BPS = 6.0
TEASER_LONG_MAX_PATH_BPS = 4.0
CLEAN_EDGE_LONG_MFE_MIN_BPS = 14.0
CLEAN_EDGE_LONG_MAE_MAX_BPS = 4.0
CLEAN_EDGE_LONG_PATH_MIN_BPS = 16.0
SURVIVAL_LONG_MFE_MIN_BPS = 8.0
SURVIVAL_LONG_MAE_MAX_BPS = 6.0
SURVIVAL_LONG_PATH_MIN_BPS = 8.0

# The old map-derived hold-horizon target is blocked from the exact model-native
# dataset. It belonged to a different Exit policy and previously admitted a
# dead constant target. No hold-map input, target column, neutral fill or active
# head exists in this builder path.


def final_direction_label_horizon_bars() -> int:
    """Return the actual horizon used by the emitted final y_direction label."""
    return int(V11_DIRECTION_HORIZON_BARS)


def final_direction_label_horizon_array(n_rows: int) -> np.ndarray:
    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative, got {n_rows}")
    return np.full(int(n_rows), final_direction_label_horizon_bars(), dtype=np.int32)


DIRECTION_DATASET_STEM_SUFFIX = (
    f"__DIR_H{final_direction_label_horizon_bars():02d}B"
)


def direction_label_contract() -> Dict[str, Any]:
    return {
        "direction_target_mode": V12_DIRECTION_TARGET_MODE,
        "direction_label_source": "v12_spread_aware_path_utility_h24_plus_first10",
        "direction_label_horizon_bars": final_direction_label_horizon_bars(),
        "direction_tradable_pnl_min_bps": float(V11_TRADABLE_PNL_MIN_BPS),
        "direction_utility_formula": (
            "pnl_at_h + mfe_weight*mfe_first_n - mae_weight*mae_first_n "
            "+ path_weight*(mfe_first_n-mae_first_n)"
        ),
        "direction_utility_path_horizon_bars": int(PATH_QUALITY_HORIZON_BARS),
        "direction_utility_mfe_weight": float(V12_DIRECTION_UTILITY_MFE_WEIGHT),
        "direction_utility_mae_weight": float(V12_DIRECTION_UTILITY_MAE_WEIGHT),
        "direction_utility_path_weight": float(V12_DIRECTION_UTILITY_PATH_WEIGHT),
        "direction_utility_min_bps": float(V12_DIRECTION_UTILITY_MIN_BPS),
        "direction_utility_min_side_margin_bps": float(
            V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS
        ),
    }


def hierarchical_direction_label_contract() -> Dict[str, Any]:
    """Document the outcome-only direction and side-target contract.

    Structure/trend/geometry remain model inputs and optional representation
    auxiliaries. They are forbidden from rewriting direction, trade, side,
    utility, MAE or bad-path outcomes.
    """
    return {
        "hierarchical_direction_targets": {
            "enabled": True,
            "primary_head": "trade_vs_flat",
            "conditional_side_head": "long_vs_short_given_trade",
            "derived_compat_label": "y_direction",
            "side_order": ["long", "short"],
            "core_target_source": "future_path_and_utility_outcomes_only",
            "feature_derived_core_rewrites_allowed": False,
            "utility_order_forcing_allowed": False,
            "target_columns": [
                "y_trade",
                "y_side",
                "y_side_mask",
                "y_long_path_utility_bps",
                "y_short_path_utility_bps",
                "y_long_bad_path",
                "y_short_bad_path",
                "y_long_expected_mae_bps",
                "y_short_expected_mae_bps",
                "y_rising_channel_support_touch",
                "y_falling_channel_resistance_touch",
                "y_support_retest_continuation",
                "y_resistance_retest_continuation",
                "y_countertrend_short_trap",
                "y_countertrend_long_trap",
                "y_mtf_conflict_m5_vs_higher_side",
                "y_long_high_mae_low_mfe_early_failure",
                "y_short_high_mae_low_mfe_early_failure",
            ],
            "geometry_features_are_asof_closed_bar": True,
            "structural_context_auxiliaries": {
                "enabled": True,
                "may_change_core_targets": False,
                "semantics": "representation_and_slice_diagnostics_only",
            },
            "runtime_rule_free": True,
        }
    }


def _write_bytes_exclusive_fsync(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_model_native_aux_head_targets(
    targets: Mapping[str, np.ndarray],
    *,
    n_rows: int,
) -> np.ndarray:
    """Prove that each target exists only where its full future path exists."""
    if n_rows < 0:
        raise ValueError(f"MODEL_NATIVE_AUX_TARGET_ROW_COUNT_INVALID: {n_rows}")
    observed_columns = tuple(targets)
    if observed_columns != MODEL_NATIVE_AUX_TARGET_COLUMNS:
        missing = [
            name for name in MODEL_NATIVE_AUX_TARGET_COLUMNS if name not in targets
        ]
        extra = [
            name
            for name in observed_columns
            if name not in MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN
        ]
        raise RuntimeError(
            "MODEL_NATIVE_AUX_TARGET_COLUMNS_INVALID: "
            f"missing={missing} extra={extra} order_matches={observed_columns == MODEL_NATIVE_AUX_TARGET_COLUMNS}"
        )

    complete = np.ones(n_rows, dtype=bool)
    row_index = np.arange(n_rows, dtype=np.int64)
    for name in MODEL_NATIVE_AUX_TARGET_COLUMNS:
        horizon = int(MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN[name])
        values = np.asarray(targets[name], dtype=np.float64)
        if values.shape != (n_rows,):
            raise RuntimeError(
                f"MODEL_NATIVE_AUX_TARGET_SHAPE_INVALID: {name} got={values.shape} expected={(n_rows,)}"
            )
        if np.isinf(values).any():
            raise RuntimeError(f"MODEL_NATIVE_AUX_TARGET_INF_FORBIDDEN: {name}")
        expected_finite = row_index < max(0, n_rows - horizon)
        observed_finite = np.isfinite(values)
        if not np.array_equal(observed_finite, expected_finite):
            mismatch = int(np.count_nonzero(observed_finite != expected_finite))
            raise RuntimeError(
                "MODEL_NATIVE_AUX_TARGET_COMPLETENESS_INVALID: "
                f"column={name} horizon={horizon} mismatched_rows={mismatch}"
            )
        if not np.isnan(values[~expected_finite]).all():
            raise RuntimeError(f"MODEL_NATIVE_AUX_TARGET_TAIL_MUST_BE_NAN: {name}")
        complete_values = values[expected_finite]
        if name in MODEL_NATIVE_DIP_MFE_TARGET_COLUMNS:
            if np.any(
                complete_values > float(MODEL_NATIVE_DIP_MFE_UPPER_SAFETY_CAP_BPS)
            ):
                raise RuntimeError(
                    "MODEL_NATIVE_AUX_TARGET_DOMAIN_INVALID: "
                    f"{name} exceeds signed MFE upper safety cap"
                )
        elif name in MODEL_NATIVE_DIP_MAE_TARGET_COLUMNS:
            if np.any(complete_values < 0.0) or np.any(
                complete_values > float(MODEL_NATIVE_DIP_MAE_UPPER_SAFETY_CAP_BPS)
            ):
                raise RuntimeError(
                    "MODEL_NATIVE_AUX_TARGET_DOMAIN_INVALID: "
                    f"{name} must remain a non-negative MAE magnitude"
                )
        complete &= observed_finite

    expected_complete = row_index < max(
        0, n_rows - MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS
    )
    if not np.array_equal(complete, expected_complete):
        raise RuntimeError("MODEL_NATIVE_AUX_TARGET_UNION_COMPLETENESS_INVALID")
    return complete


def _build_model_native_aux_head_targets(
    frame: pd.DataFrame,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Build exact spread-aware future targets without incomplete-tail sentinels."""
    required = (
        "close",
        "high",
        "low",
        "bid_close",
        "ask_close",
        "bid_high",
        "bid_low",
        "ask_high",
        "ask_low",
    )
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"MODEL_NATIVE_AUX_SPREAD_TAPE_MISSING: {missing}")

    prices: Dict[str, np.ndarray] = {}
    n_rows = len(frame)
    for name in required:
        try:
            values = pd.to_numeric(frame[name], errors="raise").to_numpy(
                dtype=np.float64
            )
        except Exception as exc:
            raise RuntimeError(f"MODEL_NATIVE_AUX_PRICE_INVALID: {name}") from exc
        if (
            values.shape != (n_rows,)
            or not np.isfinite(values).all()
            or np.any(values <= 0.0)
        ):
            raise RuntimeError(f"MODEL_NATIVE_AUX_PRICE_INVALID: {name}")
        prices[name] = values
    for high_name, low_name in (
        ("high", "low"),
        ("bid_high", "bid_low"),
        ("ask_high", "ask_low"),
    ):
        if np.any(prices[high_name] < prices[low_name]):
            raise RuntimeError(
                f"MODEL_NATIVE_AUX_OHLC_GEOMETRY_INVALID: {high_name}/{low_name}"
            )

    computed: Dict[str, np.ndarray] = {}
    bps = 1e4
    close = prices["close"]

    def _store_prefix(
        name: str,
        prefix: np.ndarray,
        *,
        lower: float | None,
        upper: float | None,
    ) -> None:
        horizon = int(MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN[name])
        valid_rows = max(0, n_rows - horizon)
        values = np.asarray(prefix, dtype=np.float64)
        if values.shape != (valid_rows,) or not np.isfinite(values).all():
            raise RuntimeError(
                f"MODEL_NATIVE_AUX_TARGET_PREFIX_INVALID: {name} got={values.shape} expected={(valid_rows,)}"
            )
        full = np.full(n_rows, np.nan, dtype=np.float32)
        # Declared bounds are a contract to verify, never a value to rewrite.
        # Silently clamping a target is exactly how V24's signed-target
        # corruption reached training undetected, so an out-of-domain target
        # fails closed here and the dataset must be rebuilt.
        if lower is not None and float(values.min()) < float(lower):
            raise RuntimeError(
                f"MODEL_NATIVE_AUX_TARGET_BELOW_DECLARED_DOMAIN: {name} "
                f"min={float(values.min()):.9g} declared_lower={float(lower):.9g}"
            )
        if upper is not None and float(values.max()) > float(upper):
            raise RuntimeError(
                f"MODEL_NATIVE_AUX_TARGET_ABOVE_DECLARED_DOMAIN: {name} "
                f"max={float(values.max()):.9g} declared_upper={float(upper):.9g}"
            )
        full[:valid_rows] = values.astype(np.float32)
        computed[name] = full

    for horizon in MODEL_NATIVE_AUX_FORECAST_HORIZONS:
        valid_rows = max(0, n_rows - horizon)
        forecast = (
            (close[horizon : horizon + valid_rows] - close[:valid_rows])
            / close[:valid_rows]
            * bps
        )
        _store_prefix(
            f"y_forecast_ret_K{horizon}",
            forecast,
            lower=-1000.0,
            upper=1000.0,
        )

    one_bar_returns = (close[1:] / close[:-1]) - 1.0
    for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS:
        valid_rows = max(0, n_rows - horizon)
        if valid_rows:
            forward_vol = (
                pd.Series(one_bar_returns)
                .rolling(horizon, min_periods=horizon)
                .std(ddof=1)
                .to_numpy(dtype=np.float64)[horizon - 1 :]
                * bps
            )
        else:
            forward_vol = np.empty(0, dtype=np.float64)
        _store_prefix(
            f"y_vol_fwd_K{horizon}",
            forward_vol,
            lower=0.0,
            upper=1000.0,
        )

    high = prices["high"]
    low = prices["low"]
    for side in ("long", "short"):
        for horizon in MODEL_NATIVE_AUX_RISK_HORIZONS:
            valid_rows = max(0, n_rows - horizon)
            entry_mid = close[:valid_rows]
            entry_spread = (
                prices["ask_close"][:valid_rows]
                if side == "long"
                else prices["bid_close"][:valid_rows]
            )
            mfe_mid = np.full(valid_rows, -np.inf, dtype=np.float64)
            run_adverse_mid = np.zeros(valid_rows, dtype=np.float64)
            run_adverse_bar = np.zeros(valid_rows, dtype=np.float64)
            dip_bottom_bar = np.zeros(valid_rows, dtype=np.float64)
            mfe_bar = np.zeros(valid_rows, dtype=np.float64)
            run_adverse_spread = np.zeros(valid_rows, dtype=np.float64)
            mae_before_spread = np.zeros(valid_rows, dtype=np.float64)
            mfe_spread = np.full(valid_rows, -np.inf, dtype=np.float64)

            for offset in range(1, horizon + 1):
                future_high = high[offset : offset + valid_rows]
                future_low = low[offset : offset + valid_rows]
                if side == "long":
                    favorable_mid = (future_high - entry_mid) / entry_mid * bps
                    adverse_mid = (future_low - entry_mid) / entry_mid * bps
                    favorable_spread = (
                        (
                            prices["bid_high"][offset : offset + valid_rows]
                            - entry_spread
                        )
                        / entry_spread
                        * bps
                    )
                    adverse_spread = (
                        (prices["bid_low"][offset : offset + valid_rows] - entry_spread)
                        / entry_spread
                        * bps
                    )
                else:
                    favorable_mid = (entry_mid - future_low) / entry_mid * bps
                    adverse_mid = (entry_mid - future_high) / entry_mid * bps
                    favorable_spread = (
                        (entry_spread - prices["ask_low"][offset : offset + valid_rows])
                        / entry_spread
                        * bps
                    )
                    adverse_spread = (
                        (
                            entry_spread
                            - prices["ask_high"][offset : offset + valid_rows]
                        )
                        / entry_spread
                        * bps
                    )

                new_worst = adverse_mid < run_adverse_mid
                run_adverse_mid = np.minimum(run_adverse_mid, adverse_mid)
                run_adverse_bar = np.where(new_worst, offset, run_adverse_bar)
                new_peak = favorable_mid > mfe_mid
                dip_bottom_bar = np.where(new_peak, run_adverse_bar, dip_bottom_bar)
                mfe_bar = np.where(new_peak, offset, mfe_bar)
                mfe_mid = np.maximum(mfe_mid, favorable_mid)

                run_adverse_spread = np.minimum(run_adverse_spread, adverse_spread)
                mae_before_spread = np.where(
                    new_peak, -run_adverse_spread, mae_before_spread
                )
                mfe_spread = np.maximum(mfe_spread, favorable_spread)

            _store_prefix(
                f"y_dip_mae_{side}_K{horizon}",
                mae_before_spread,
                lower=0.0,
                upper=MODEL_NATIVE_DIP_MAE_UPPER_SAFETY_CAP_BPS,
            )
            _store_prefix(
                f"y_dip_mfe_{side}_K{horizon}",
                mfe_spread,
                lower=None,
                upper=MODEL_NATIVE_DIP_MFE_UPPER_SAFETY_CAP_BPS,
            )
            _store_prefix(
                f"y_dip_bottom_frac_{side}_K{horizon}",
                dip_bottom_bar / float(horizon),
                lower=0.0,
                upper=1.0,
            )
            _store_prefix(
                f"y_time_to_mfe_frac_{side}_K{horizon}",
                mfe_bar / float(horizon),
                lower=0.0,
                upper=1.0,
            )
            _store_prefix(
                f"y_tail_mae_{side}_K{horizon}",
                -run_adverse_spread,
                lower=0.0,
                upper=MODEL_NATIVE_TAIL_MAE_UPPER_SAFETY_CAP_BPS,
            )

            if side == "long":
                final_pnl_bps = (
                    (prices["bid_close"][horizon : horizon + valid_rows] - entry_spread)
                    / entry_spread
                    * bps
                )
            else:
                final_pnl_bps = (
                    (entry_spread - prices["ask_close"][horizon : horizon + valid_rows])
                    / entry_spread
                    * bps
                )
            full_path_mae_bps = -run_adverse_spread
            action_value_bps = (
                final_pnl_bps
                + UTILITY_MFE_WEIGHT * mfe_spread
                - UTILITY_MAE_WEIGHT * full_path_mae_bps
                + UTILITY_PATH_WEIGHT * (mfe_spread - full_path_mae_bps)
            )
            _store_prefix(
                f"y_action_value_{side}_K{horizon}",
                action_value_bps,
                lower=None,
                upper=None,
            )

    for horizon in OFFLINE_RL_HORIZON_BARS:
        _store_prefix(
            f"y_action_value_flat_K{horizon}",
            np.zeros(max(0, n_rows - horizon), dtype=np.float64),
            lower=None,
            upper=None,
        )

    ordered = {name: computed[name] for name in MODEL_NATIVE_AUX_TARGET_COLUMNS}
    complete = _validate_model_native_aux_head_targets(ordered, n_rows=n_rows)
    return ordered, complete


def _selected_side_bad_path_target(
    quality_side: np.ndarray,
    long_bad_path: np.ndarray,
    short_bad_path: np.ndarray,
) -> np.ndarray:
    side = np.asarray(quality_side)
    long_bad = np.asarray(long_bad_path, dtype=np.float32)
    short_bad = np.asarray(short_bad_path, dtype=np.float32)
    out = np.zeros_like(long_bad, dtype=np.float32)
    out[side == 0] = long_bad[side == 0]
    out[side == 1] = short_bad[side == 1]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _position_size_target_from_path(
    mfe_first_n_bps: np.ndarray,
    mae_first_n_bps: np.ndarray,
    atr_bps: np.ndarray,
    trade_mask: np.ndarray,
) -> np.ndarray:
    mfe = np.asarray(mfe_first_n_bps, dtype=np.float64)
    mae = np.asarray(mae_first_n_bps, dtype=np.float64)
    atr = np.asarray(atr_bps, dtype=np.float64)
    mask = np.asarray(trade_mask, dtype=np.float64)
    if not (mfe.shape == mae.shape == atr.shape == mask.shape):
        raise ValueError(
            "POSITION_SIZE_TARGET_SHAPE_MISMATCH: "
            f"mfe={mfe.shape} mae={mae.shape} atr={atr.shape} mask={mask.shape}"
        )
    invalid = {
        "mfe_first_n_bps": int(np.count_nonzero(~np.isfinite(mfe))),
        # Path builders expose MAE as an adverse *magnitude*.  Accepting a
        # signed/negative MAE here would silently reverse its sizing meaning.
        "mae_first_n_bps": int(np.count_nonzero(~np.isfinite(mae) | (mae < 0.0))),
        "atr_bps": int(np.count_nonzero(~np.isfinite(atr) | (atr <= 0.0))),
        "trade_mask": int(
            np.count_nonzero(~np.isfinite(mask) | ~np.isin(mask, (0.0, 1.0)))
        ),
    }
    invalid = {name: count for name, count in invalid.items() if count}
    if invalid:
        raise ValueError(f"POSITION_SIZE_TARGET_INPUT_INVALID: {invalid}")
    # MFE is signed favorable excursion after spread; MAE is a non-negative
    # adverse magnitude.  Worse adverse paths must therefore reduce, never
    # increase, the learned size target.
    signed_edge_atr = (mfe - mae) / (atr * 2.0)
    # Clipping the finite logit only prevents numerical exp overflow; it never
    # substitutes for missing/zero ATR or path evidence.
    bounded_logit = np.clip(signed_edge_atr, -80.0, 80.0)
    out = 1.0 / (1.0 + np.exp(-bounded_logit))
    out[mask == 0.0] = 0.5
    if not bool(np.isfinite(out).all()):
        raise RuntimeError("POSITION_SIZE_TARGET_NON_FINITE_OUTPUT")
    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------
def get_git_commit() -> str:
    """Get current git commit hash (best-effort)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _hard_gate_model_native_context() -> Dict[str, Any]:
    """Return the exact Entry-owned 142/5 context contract or fail closed."""

    ctx = model_native_context_contract_metadata()
    if tuple(ctx.get("ctx_cont_names") or ()) != MODEL_NATIVE_CTX_CONT_FIELDS:
        raise RuntimeError("MODEL_NATIVE_CTX_CONT_FIELDS_ORDER_MISMATCH")
    if tuple(ctx.get("ctx_cat_names") or ()) != MODEL_NATIVE_CTX_CAT_FIELDS:
        raise RuntimeError("MODEL_NATIVE_CTX_CAT_FIELDS_ORDER_MISMATCH")
    if tuple(ctx.get("ctx_cont_source_prefix_names") or ()) != (
        MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_FIELDS
    ):
        raise RuntimeError("MODEL_NATIVE_CTX_CONT_SOURCE_PREFIX_ORDER_MISMATCH")
    if tuple(ctx.get("ctx_cont_micro_features") or ()) != (
        MODEL_NATIVE_CTX_CONT_MICRO_FIELDS
    ):
        raise RuntimeError("MODEL_NATIVE_CTX_CONT_MICRO_ORDER_MISMATCH")
    if tuple(ctx.get("ctx_cont_swing_features") or ()) != (
        MODEL_NATIVE_CTX_CONT_SWING_FIELDS
    ):
        raise RuntimeError("MODEL_NATIVE_CTX_CONT_SWING_ORDER_MISMATCH")
    if tuple(ctx.get("ctx_cont_session_features") or ()) != (
        MODEL_NATIVE_CTX_CONT_SESSION_FIELDS
    ):
        raise RuntimeError("MODEL_NATIVE_CTX_CONT_SESSION_ORDER_MISMATCH")
    if int(ctx.get("ctx_cont_dim", -1)) != MODEL_NATIVE_CTX_CONT_DIM:
        raise RuntimeError("MODEL_NATIVE_CTX_CONT_DIM_MISMATCH")
    if int(ctx.get("ctx_cat_dim", -1)) != MODEL_NATIVE_CTX_CAT_DIM:
        raise RuntimeError("MODEL_NATIVE_CTX_CAT_DIM_MISMATCH")
    return ctx


def _model_native_artifact_owner_fields(
    active_base_signal_fields: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return exact (canonical-v2, source-prebuilt) field ownership."""

    from gx1.features.regime_v4_features import REGIME_V4_SOURCE_COLS
    from gx1.features.volume_features import VOLUME_FEATURE_NAMES

    volume_derived = set(VOLUME_FEATURE_NAMES)
    cv2_owned = tuple(
        dict.fromkeys(
            [name for name in active_base_signal_fields if name not in volume_derived]
            + list(MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS)
        )
    )
    source_owned = tuple(
        dict.fromkeys(
            list(MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS)
            + [
                name
                for name in REGIME_V4_SOURCE_COLS
                if name != "D1_dist_from_ema200_atr"
            ]
            + ["volume"]
        )
    )
    overlap = sorted(set(cv2_owned) & set(source_owned))
    if overlap:
        raise RuntimeError(f"MODEL_NATIVE_ARTIFACT_OWNER_OVERLAP: {overlap}")
    if len(cv2_owned) != len(set(cv2_owned)) or len(source_owned) != len(
        set(source_owned)
    ):
        raise RuntimeError("MODEL_NATIVE_ARTIFACT_OWNER_DUPLICATE")
    return cv2_owned, source_owned


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _signal_build_contract_from_manifest(
    manifest_path: Path,
) -> Dict[str, Any]:
    """Load the one accepted signal manifest without compatibility defaults."""

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"SEQ_STRUCTURE_MANIFEST_MISSING: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    contract = require_model_native_manifest(manifest, context="DATASET_BUILDER")
    return {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
        "base_fields": list(contract["base_fields"]),
        "model_native_signal_contract": contract,
    }


def _resolve_seq_structure_extension(
    *,
    manifest_path: Optional[Path],
) -> Tuple[List[str], Dict[str, Any]]:
    if manifest_path is None:
        raise RuntimeError("SEQ_STRUCTURE_EXACT_MANIFEST_REQUIRED")
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"SEQ_STRUCTURE_MANIFEST_MISSING: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = require_model_native_manifest(
        manifest, context="DATASET_BUILDER_EXTENSION"
    )
    features = list(contract["selected_fields"])
    meta = {
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_variant": manifest.get("manifest_variant"),
        "expected_seq_snap_width": manifest.get("expected_seq_snap_width"),
        "manifest_selected_feature_count": len(features),
        "mode": "mandatory_inline_common_causal_history_v1",
        "foundation_structure_feature_version": manifest.get(
            "foundation_structure_feature_version"
        ),
        "foundation_structure_feature_count": manifest.get(
            "foundation_structure_feature_count"
        ),
        "foundation_structure_missing_feature_count": manifest.get(
            "foundation_structure_missing_feature_count"
        ),
        "foundation_structure_all_required_selected": manifest.get(
            "foundation_structure_all_required_selected"
        ),
        "manifest": manifest,
    }
    return features, meta


def _build_inline_seq_structure_extension(
    merged3: pd.DataFrame,
    *,
    requested_features: List[str],
    ctx_cont_names: List[str],
    ctx_cat_names: Optional[List[str]] = None,
    source_parquet: Optional[Path],
    source_contract_label: Optional[str] = None,
    base_signal_fields: Sequence[str] = MODEL_NATIVE_BASE_FIELDS,
    precomputed_price_layer: Optional[Tuple[np.ndarray, List[str]]] = None,
    precomputed_candle_layer: Optional[Tuple[np.ndarray, List[str]]] = None,
    emit_offset: int = 0,
    support_memory_state: Optional[Mapping[str, float]] = None,
    return_support_memory_state: bool = False,
) -> (
    Tuple[np.ndarray, List[str], Dict[str, Any]]
    | Tuple[
        np.ndarray,
        List[str],
        Dict[str, Any],
        Dict[str, np.float32],
    ]
):
    if not requested_features:
        raise RuntimeError("SEQ_STRUCTURE_INLINE_REQUESTED_FEATURES_EMPTY")
    if source_parquet is None:
        raise RuntimeError("SEQ_STRUCTURE_INLINE_SOURCE_PARQUET_REQUIRED")
    if (
        isinstance(emit_offset, bool)
        or not isinstance(emit_offset, int)
        or emit_offset < 0
        or emit_offset >= len(merged3)
    ):
        raise RuntimeError("SEQ_STRUCTURE_INLINE_EMIT_OFFSET_INVALID")

    from gx1.features.entry_model_native_feature_layers_v1 import (
        MODEL_NATIVE_SPECIALIST_LAYER_FEATURES,
        build_candlestick_derived_layer,
        build_chart_layer,
        build_deep_interaction_layer,
        build_price_derived_layer,
    )
    from gx1.features.entry_chart_geometry_v1 import build_entry_chart_geometry_layer
    from gx1.features.entry_foundation_structure_v1 import (
        build_entry_foundation_structure_layer,
    )
    from gx1.features.entry_momentum_flow_v1 import build_entry_momentum_flow_layer
    from gx1.features.entry_mtf_confluence_v1 import build_entry_mtf_confluence_layer
    from gx1.features.entry_session_regime_interactions_v1 import (
        build_entry_session_regime_interaction_layer,
    )
    from gx1.features.entry_smc_liquidity_quality_v1 import (
        build_entry_smc_liquidity_quality_layer,
    )
    from gx1.features.entry_structure_swing_derivations_v1 import (
        build_entry_structure_swing_derivation_layer,
    )
    from gx1.features.entry_support_resistance_memory_v1 import (
        build_entry_support_resistance_memory_layer,
    )
    from gx1.features.entry_trend_ema_v1 import build_entry_trend_ema_layer
    from gx1.features.entry_vol_compression_v1 import build_entry_vol_compression_layer

    base_blocks: List[np.ndarray] = []
    base_names: List[str] = []
    for field in base_signal_fields:
        if field not in merged3.columns:
            raise RuntimeError(f"SEQ_STRUCTURE_INLINE_MISSING_SIGNAL_FIELD: {field}")
        base_blocks.append(merged3[field].astype(np.float32).to_numpy().reshape(-1, 1))
        base_names.append(f"snap.{field}")
    for field in ctx_cont_names:
        if field not in merged3.columns:
            raise RuntimeError(f"SEQ_STRUCTURE_INLINE_MISSING_CTX_FIELD: {field}")
        base_blocks.append(merged3[field].astype(np.float32).to_numpy().reshape(-1, 1))
        base_names.append(f"ctx_cont.{field}")
    for field in ctx_cat_names or []:
        if field not in merged3.columns:
            raise RuntimeError(f"SEQ_STRUCTURE_INLINE_MISSING_CTX_CAT_FIELD: {field}")
        base_blocks.append(merged3[field].astype(np.float32).to_numpy().reshape(-1, 1))
        base_names.append(f"ctx_cat.{field}")
    base_x = np.concatenate(base_blocks, axis=1).astype(np.float32, copy=False)

    chart_x, chart_names = build_chart_layer(base_x, base_names)
    if precomputed_price_layer is None:
        price_x, price_names = build_price_derived_layer(
            merged3[["time"]].copy(),
            Path(source_parquet),
        )
    else:
        price_x, price_names = precomputed_price_layer
        price_x = np.asarray(price_x, dtype=np.float32)
        price_names = list(price_names)
        if (
            price_x.ndim != 2
            or price_x.shape != (len(merged3), len(price_names))
            or not np.isfinite(price_x).all()
        ):
            raise RuntimeError("SEQ_STRUCTURE_INLINE_PRICE_LAYER_INVALID")
    chart_x = np.concatenate([chart_x, price_x], axis=1).astype(np.float32, copy=False)
    chart_names = list(chart_names) + list(price_names)
    if precomputed_candle_layer is None:
        candle_x, candle_names = build_candlestick_derived_layer(
            merged3[["time"]].copy(),
            Path(source_parquet),
        )
    else:
        candle_x, candle_names = precomputed_candle_layer
        candle_x = np.asarray(candle_x, dtype=np.float32)
        candle_names = list(candle_names)
        if (
            candle_x.ndim != 2
            or candle_x.shape != (len(merged3), len(candle_names))
            or not np.isfinite(candle_x).all()
        ):
            raise RuntimeError("SEQ_STRUCTURE_INLINE_CANDLE_LAYER_INVALID")
    chart_x = np.concatenate([chart_x, candle_x], axis=1).astype(np.float32, copy=False)
    chart_names = list(chart_names) + list(candle_names)
    chart_all_x = (
        np.concatenate([base_x, chart_x], axis=1).astype(np.float32, copy=False)
        if chart_x.shape[1]
        else base_x
    )
    chart_all_names = list(base_names) + list(chart_names)
    deep_x, deep_names = build_deep_interaction_layer(
        chart_all_x,
        chart_all_names,
        merged3[["time"]].copy(),
    )

    all_pieces = [base_x]
    all_names = list(base_names)
    if chart_x.shape[1]:
        all_pieces.append(chart_x)
        all_names.extend(chart_names)
    if deep_x.shape[1]:
        all_pieces.append(deep_x)
        all_names.extend(deep_names)
    all_x = np.concatenate(all_pieces, axis=1).astype(np.float32, copy=False)
    requested_set = set(requested_features)
    smart_generated_layers: List[Dict[str, Any]] = []

    def _append_generated_layer(
        label: str, layer_x: np.ndarray, layer_names: List[str]
    ) -> None:
        nonlocal all_x, all_names
        layer_x = np.asarray(layer_x, dtype=np.float32)
        if layer_x.ndim != 2:
            raise RuntimeError(
                f"SEQ_STRUCTURE_INLINE_SMART_LAYER_NOT_2D: {label} shape={layer_x.shape}"
            )
        if layer_x.shape[0] != all_x.shape[0]:
            raise RuntimeError(
                f"SEQ_STRUCTURE_INLINE_SMART_LAYER_ROW_MISMATCH: {label} "
                f"rows={layer_x.shape[0]} expected={all_x.shape[0]}"
            )
        existing = set(all_names)
        selected_pairs = [
            (i, name)
            for i, name in enumerate(layer_names)
            if name in requested_set and name not in existing
        ]
        if not selected_pairs:
            return
        selected_idx = [i for i, _ in selected_pairs]
        selected_names = [name for _, name in selected_pairs]
        selected_x = layer_x[:, selected_idx].astype(np.float32, copy=False)
        if not np.isfinite(selected_x).all():
            raise RuntimeError(f"SEQ_STRUCTURE_INLINE_SMART_LAYER_NONFINITE: {label}")
        all_x = np.concatenate([all_x, selected_x], axis=1).astype(
            np.float32, copy=False
        )
        all_names.extend(selected_names)
        smart_generated_layers.append(
            {
                "label": label,
                "feature_count": int(len(selected_names)),
                "features": selected_names,
            }
        )

    def _candlestick_layer_strict() -> Tuple[np.ndarray, List[str]]:
        # Reuse the already strict, exactly aligned candlestick layer above.
        # This avoids a second parquet read with a different normalization path.
        return candle_x.astype(np.float32, copy=False), list(candle_names)

    def _build_chart_geometry_smart_layer_strict(
        all_x: np.ndarray, all_names: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        return build_entry_chart_geometry_layer(all_x, all_names)

    smart_builders = {
        "foundation_cross_family_layer": build_entry_foundation_structure_layer,
        "trend_ema_smart_layer": build_entry_trend_ema_layer,
        "smc_liquidity_quality_layer": build_entry_smc_liquidity_quality_layer,
        "structure_swing_derivation_layer": build_entry_structure_swing_derivation_layer,
        "momentum_flow_smart_layer": build_entry_momentum_flow_layer,
        "session_regime_interaction_layer": build_entry_session_regime_interaction_layer,
        "vol_compression_smart_layer": build_entry_vol_compression_layer,
        "chart_geometry_smart2_layer": _build_chart_geometry_smart_layer_strict,
        "price_action_candle_smart3_layer": _candlestick_layer_strict,
        "support_resistance_memory_layer": build_entry_support_resistance_memory_layer,
        "mtf_confluence_layer": build_entry_mtf_confluence_layer,
    }
    next_support_memory_state: Dict[str, np.float32] = {}
    emit_applied = False
    for label, feature_names in MODEL_NATIVE_SPECIALIST_LAYER_FEATURES:
        if not any(
            name in requested_set and name not in set(all_names)
            for name in feature_names
        ):
            continue
        builder = smart_builders[label]
        if label == "price_action_candle_smart3_layer":
            smart_x, smart_names = builder()
        elif label == "support_resistance_memory_layer" and (
            emit_offset > 0 or return_support_memory_state
        ):
            if emit_offset > 0:
                all_x = all_x[emit_offset:]
                emit_applied = True
            smart_x, smart_names, next_support_memory_state = builder(
                all_x,
                all_names,
                memory_state=support_memory_state,
                return_memory_state=True,
            )
        else:
            smart_x, smart_names = builder(all_x, all_names)
        _append_generated_layer(label, smart_x, list(smart_names))

    if emit_offset > 0 and not emit_applied:
        all_x = all_x[emit_offset:]

    index = {name: i for i, name in enumerate(all_names)}
    missing = [name for name in requested_features if name not in index]
    if missing:
        raise RuntimeError(
            f"SEQ_STRUCTURE_INLINE_FEATURES_MISSING: {missing[:30]} total={len(missing)}"
        )

    selected = [name for name in requested_features]
    selected_cols: List[np.ndarray] = []
    for name in selected:
        selected_cols.append(all_x[:, index[name]].astype(np.float32, copy=False))
    out = np.column_stack(selected_cols).astype(np.float32, copy=False)
    if not np.isfinite(out).all():
        raise RuntimeError("SEQ_STRUCTURE_INLINE_NONFINITE_VALUES")

    meta = {
        "mode": "mandatory_inline_common_causal_history_v1",
        "features": list(selected),
        "feature_count": int(len(selected)),
        "base_matrix_dim": int(base_x.shape[1]),
        "chart_generated_dim": int(chart_x.shape[1]),
        "deep_generated_dim": int(deep_x.shape[1]),
        "smart_generated_dim": int(
            sum(row["feature_count"] for row in smart_generated_layers)
        ),
        "smart_generated_layers": smart_generated_layers,
        "available_feature_count": int(len(all_names)),
        "source_parquet_for_price_features": (
            str(source_contract_label)
            if source_contract_label is not None
            else (str(source_parquet) if source_parquet is not None else None)
        ),
        "missing_generated_features": [],
    }
    if return_support_memory_state:
        if not next_support_memory_state:
            raise RuntimeError("SEQ_STRUCTURE_INLINE_SUPPORT_STATE_MISSING")
        return out, selected, meta, next_support_memory_state
    return out, selected, meta


def _parse_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _split_min_max_from_ts_series(ts: pd.Series) -> Dict[str, Optional[str]]:
    t = pd.to_datetime(ts, utc=True, errors="coerce").dropna()
    if t.empty:
        return {"ts_min": None, "ts_max": None}
    return {"ts_min": str(t.min()), "ts_max": str(t.max())}


def _detect_time_col(df: pd.DataFrame) -> str:
    if "time" in df.columns:
        return "time"
    if "ts" in df.columns:
        return "ts"
    # Sometimes parquet index is time
    if "index" in df.columns:
        return "index"
    raise RuntimeError(
        "TIME_COLUMN_MISSING: canonical builder requires tz-aware UTC time column (time or ts)."
    )


def _normalize_time_utc(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    if out["time"].isna().any():
        raise RuntimeError(
            "TIME_PARSE_FAIL: time column could not be parsed to tz-aware UTC"
        )
    duplicate_count = int(out["time"].duplicated().sum())
    if duplicate_count:
        raise RuntimeError(f"TIME_DUPLICATE_ROWS: count={duplicate_count}")
    out = out.sort_values("time", kind="mergesort").copy()
    if len(out) == 0:
        raise RuntimeError("EMPTY_AFTER_TIME_NORMALIZATION")
    return out


# -----------------------------------------------------------------------------
# Market tape loading (canonical lane)
# -----------------------------------------------------------------------------
def _load_canonical_tape(
    *,
    tape_root: Path,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    required_cols: List[str],
) -> pd.DataFrame:
    """
    Load canonical M5 tape for [t_min, t_max] from a partitioned parquet dataset:
      .../xauusd_m5_bid_ask__CANONICAL/year=YYYY/part-000.parquet

    We avoid depending on manifest schema here; we trust the canonical lane path and parquet partitioning.
    """
    tape_root = tape_root.expanduser().resolve()
    if not tape_root.exists():
        raise RuntimeError(f"TAPE_ROOT_MISSING: {tape_root}")
    if not tape_root.is_dir():
        raise RuntimeError(f"TAPE_ROOT_NOT_DIR: {tape_root}")

    # Pull only years that intersect range
    y0 = int(pd.Timestamp(t_min).year)
    y1 = int(pd.Timestamp(t_max).year)
    files: List[Path] = []
    for y in range(y0, y1 + 1):
        p = tape_root / f"year={y}"
        if p.exists() and p.is_dir():
            files.extend(sorted(p.glob("*.parquet")))
    files = sorted(set(files))
    # If layout differs, fall back to recursive parquet scan (still deterministic)
    if not files:
        files = sorted(tape_root.rglob("*.parquet"))

    if not files:
        raise RuntimeError(f"TAPE_NO_FILES: no parquet files found under {tape_root}")

    # Read and filter
    df_list: List[pd.DataFrame] = []
    for fp in files:
        dfi = pd.read_parquet(fp, columns=list(set(["time"] + required_cols)))
        if "time" not in dfi.columns:
            # Some tape uses "ts"
            if "ts" in dfi.columns:
                dfi = dfi.rename(columns={"ts": "time"})
            else:
                raise RuntimeError(f"TAPE_TIME_MISSING: {fp}")
        dfi["time"] = pd.to_datetime(dfi["time"], utc=True, errors="coerce")
        invalid_times = int(dfi["time"].isna().sum())
        if invalid_times:
            raise RuntimeError(f"TAPE_TIME_PARSE_FAIL: path={fp} count={invalid_times}")
        dfi = dfi[(dfi["time"] >= t_min) & (dfi["time"] <= t_max)]
        if len(dfi):
            df_list.append(dfi)

    if not df_list:
        raise RuntimeError("TAPE_EMPTY_IN_RANGE")

    tape = pd.concat(df_list, ignore_index=True)
    tape = tape.sort_values("time", kind="mergesort")
    duplicate_count = int(tape["time"].duplicated().sum())
    if duplicate_count:
        raise RuntimeError(f"TAPE_DUPLICATE_TIME_ROWS: count={duplicate_count}")

    missing = [c for c in required_cols if c not in tape.columns]
    if missing:
        raise RuntimeError(f"TAPE_REQUIRED_COLS_MISSING: {missing}")

    numeric_columns: Dict[str, np.ndarray] = {}
    for column in required_cols:
        try:
            values = pd.to_numeric(tape[column], errors="raise").to_numpy(
                dtype=np.float64
            )
        except Exception as exc:
            raise RuntimeError(f"TAPE_REQUIRED_COLUMN_INVALID: {column}") from exc
        if not np.isfinite(values).all():
            raise RuntimeError(f"TAPE_REQUIRED_COLUMN_NONFINITE: {column}")
        if np.any(values <= 0.0):
            raise RuntimeError(f"TAPE_REQUIRED_COLUMN_NONPOSITIVE: {column}")
        numeric_columns[column] = values

    for open_name, high_name, low_name, close_name in (
        ("open", "high", "low", "close"),
        ("bid_open", "bid_high", "bid_low", "bid_close"),
        ("ask_open", "ask_high", "ask_low", "ask_close"),
    ):
        if high_name not in numeric_columns or low_name not in numeric_columns:
            continue
        high = numeric_columns[high_name]
        low = numeric_columns[low_name]
        invalid = high < low
        if open_name in numeric_columns:
            open_values = numeric_columns[open_name]
            invalid |= (high < open_values) | (low > open_values)
        if close_name in numeric_columns:
            close_values = numeric_columns[close_name]
            invalid |= (high < close_values) | (low > close_values)
        if invalid.any():
            raise RuntimeError(
                f"TAPE_OHLC_GEOMETRY_INVALID: family={high_name}/{low_name} "
                f"count={int(invalid.sum())}"
            )

    if tape["time"].dtype != "datetime64[ns, UTC]":
        # pandas sometimes shows tz-aware as dtype object, normalize again
        tape["time"] = pd.to_datetime(tape["time"], utc=True, errors="coerce")
        invalid_times = int(tape["time"].isna().sum())
        if invalid_times:
            raise RuntimeError(
                f"TAPE_TIME_PARSE_FAIL_AFTER_NORMALIZATION: count={invalid_times}"
            )

    if len(tape) == 0:
        raise RuntimeError("TAPE_EMPTY_AFTER_NORMALIZATION")

    return tape


def build_unified_exit_lifecycle_episodes(
    *,
    entry_rows: pd.DataFrame,
    closed_m1: pd.DataFrame,
    split_end: pd.Timestamp | str,
    target_lookahead_m1_steps: int,
    market_closure_contract: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build compact, replayable Exit episodes for every Entry row and side.

    The artifact stores no duplicated M1 paths and no precomputed model state.
    Each episode points into one hash-bound literal M1 source.  HOLD/EXIT_NOW
    targets are recomputable from that source, while the proof binds the exact
    target byte stream used by the trainer.
    """
    if market_closure_contract != CANONICAL_NATIVE_CLOSURE_CONTRACT:
        raise RuntimeError(
            "UNIFIED_EXIT_M1_MARKET_CLOSURE_PROOF_REQUIRED"
        )
    lookahead = target_lookahead_m1_steps
    if (
        isinstance(lookahead, bool)
        or not isinstance(lookahead, int)
        or lookahead <= 0
    ):
        raise RuntimeError(
            "UNIFIED_EXIT_TARGET_LOOKAHEAD_INVALID: explicit positive integer required"
        )
    if not isinstance(entry_rows, pd.DataFrame) or tuple(
        name for name in ("time",) if name in entry_rows.columns
    ) != ("time",):
        raise RuntimeError("UNIFIED_EXIT_ENTRY_ROWS_SCHEMA_INVALID")
    if not isinstance(closed_m1, pd.DataFrame):
        raise RuntimeError("UNIFIED_EXIT_M1_SOURCE_FRAME_REQUIRED")
    missing_m1 = [
        name
        for name in UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS
        if name not in closed_m1.columns
    ]
    if missing_m1:
        raise RuntimeError(
            f"UNIFIED_EXIT_M1_SOURCE_FIELDS_MISSING: {missing_m1}"
        )

    entry_time = pd.DatetimeIndex(
        pd.to_datetime(entry_rows["time"], utc=True, errors="coerce")
    ).as_unit("ns")
    if (
        len(entry_time) == 0
        or entry_time.hasnans
        or not entry_time.is_unique
        or not entry_time.is_monotonic_increasing
        or not entry_time.floor(f"{ENTRY_DECISION_BAR_SECONDS}s").equals(entry_time)
    ):
        raise RuntimeError("UNIFIED_EXIT_ENTRY_TIME_GEOMETRY_INVALID")

    m1 = closed_m1.loc[
        :,
        list(UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS),
    ].copy()
    assert_no_price_scale_glitch(
        m1,
        context="UNIFIED_EXIT_LIFECYCLE_TARGET_SOURCE",
    )
    m1_time = pd.DatetimeIndex(
        pd.to_datetime(m1.pop("time"), utc=True, errors="coerce")
    ).as_unit("ns")
    if (
        len(m1_time) == 0
        or m1_time.hasnans
        or not m1_time.is_unique
        or not m1_time.is_monotonic_increasing
        or not m1_time.floor(f"{EXIT_DECISION_BAR_SECONDS}s").equals(m1_time)
    ):
        raise RuntimeError("UNIFIED_EXIT_M1_TIME_GEOMETRY_INVALID")

    numeric = m1.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("UNIFIED_EXIT_M1_SOURCE_NONFINITE")
    for prefix in ("", "bid_", "ask_"):
        open_values = numeric[f"{prefix}open"].to_numpy(dtype=np.float64)
        high_values = numeric[f"{prefix}high"].to_numpy(dtype=np.float64)
        low_values = numeric[f"{prefix}low"].to_numpy(dtype=np.float64)
        close_values = numeric[f"{prefix}close"].to_numpy(dtype=np.float64)
        if (
            np.any(open_values <= 0.0)
            or np.any(close_values <= 0.0)
            or np.any(low_values <= 0.0)
            or np.any(high_values < np.maximum(open_values, close_values))
            or np.any(low_values > np.minimum(open_values, close_values))
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_M1_OHLC_GEOMETRY_INVALID: {prefix or 'mid_'}"
            )
    for suffix in ("open", "high", "low", "close"):
        if np.any(
            numeric[f"ask_{suffix}"].to_numpy(dtype=np.float64)
            <= numeric[f"bid_{suffix}"].to_numpy(dtype=np.float64)
        ):
            raise RuntimeError(
                f"UNIFIED_EXIT_M1_EXECUTABLE_SPREAD_INVALID: {suffix}"
            )
    volume = numeric["volume"].to_numpy(dtype=np.float64)
    if np.any(volume < 0.0) or not np.equal(volume, np.floor(volume)).all():
        raise RuntimeError("UNIFIED_EXIT_M1_VOLUME_INVALID")

    parsed_split_end = pd.Timestamp(split_end)
    if (
        pd.isna(parsed_split_end)
        or parsed_split_end.tz is None
        or parsed_split_end.utcoffset() != pd.Timedelta(0)
    ):
        raise RuntimeError("UNIFIED_EXIT_SPLIT_END_INVALID")
    parsed_split_end = parsed_split_end.as_unit("ns")

    bid_open = numeric["bid_open"].to_numpy(dtype=np.float64)
    ask_open = numeric["ask_open"].to_numpy(dtype=np.float64)
    bid_close = numeric["bid_close"].to_numpy(dtype=np.float64)
    ask_close = numeric["ask_close"].to_numpy(dtype=np.float64)
    future_bid, future_ask = _unified_exit_future_extrema(
        bid=bid_close,
        ask=ask_close,
        lookahead=lookahead,
    )
    current_bid = bid_close[:-lookahead]
    current_ask = ask_close[:-lookahead]
    long_targets = np.where(
        future_bid > current_bid,
        0,
        np.where(future_bid < current_bid, 1, -1),
    ).astype(np.int8)
    short_targets = np.where(
        future_ask < current_ask,
        0,
        np.where(future_ask > current_ask, 1, -1),
    ).astype(np.int8)

    path_state_count = int(UNIFIED_EXIT_MAX_PATH_BARS)
    required_rows = path_state_count + lookahead
    m1_ns = m1_time.asi8
    records: list[dict[str, Any]] = []
    skipped = {
        "missing_entry_available_m1_open": 0,
        "insufficient_m1_tail": 0,
        "crosses_split_end": 0,
    }
    target_stream = hashlib.sha256()
    target_counts = {
        UNIFIED_EXIT_ACTION_ORDER[0]: 0,
        UNIFIED_EXIT_ACTION_ORDER[1]: 0,
        "TIED_OMITTED": 0,
    }
    for entry_row_index, entry_timestamp in enumerate(entry_time):
        entry_available_at = entry_timestamp + pd.Timedelta(
            seconds=ENTRY_DECISION_BAR_SECONDS
        )
        start_row = int(np.searchsorted(m1_ns, entry_available_at.value))
        if start_row >= len(m1_ns) or m1_ns[start_row] != entry_available_at.value:
            skipped["missing_entry_available_m1_open"] += 1
            continue
        if start_row + required_rows > len(m1_ns):
            skipped["insufficient_m1_tail"] += 1
            continue
        required_last_row = start_row + required_rows - 1
        required_end_available_at = pd.Timestamp(
            m1_ns[required_last_row]
            + int(pd.Timedelta(seconds=EXIT_DECISION_BAR_SECONDS).value),
            unit="ns",
            tz="UTC",
        )
        if required_end_available_at > parsed_split_end:
            skipped["crosses_split_end"] += 1
            continue

        state_slice = slice(start_row, start_row + path_state_count)
        for side_index, side in enumerate(UNIFIED_EXIT_SIDE_ORDER):
            target = (
                long_targets[state_slice]
                if side == "long"
                else short_targets[state_slice]
            )
            hold_count = int(np.count_nonzero(target == 0))
            exit_count = int(np.count_nonzero(target == 1))
            tied_count = int(np.count_nonzero(target == -1))
            non_tied_count = hold_count + exit_count
            if non_tied_count == 0:
                raise RuntimeError(
                    "UNIFIED_EXIT_LIFECYCLE_EPISODE_ALL_TARGETS_TIED"
                )
            target_counts[UNIFIED_EXIT_ACTION_ORDER[0]] += hold_count
            target_counts[UNIFIED_EXIT_ACTION_ORDER[1]] += exit_count
            target_counts["TIED_OMITTED"] += tied_count
            target_stream.update(
                np.asarray(
                    [entry_row_index, side_index, start_row],
                    dtype="<i8",
                ).tobytes()
            )
            target_stream.update(np.ascontiguousarray(target).tobytes())
            records.append(
                {
                    "schema_version": (
                        UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
                    ),
                    "entry_row_index": np.int64(entry_row_index),
                    "entry_time": entry_timestamp,
                    "entry_available_at": entry_available_at,
                    "side_index": np.int8(side_index),
                    "side": side,
                    "entry_bid": np.float64(bid_open[start_row]),
                    "entry_ask": np.float64(ask_open[start_row]),
                    "m1_start_row": np.int64(start_row),
                    "m1_start_time": m1_time[start_row],
                    "path_state_count": np.int16(path_state_count),
                    "target_lookahead_m1_steps": np.int16(lookahead),
                    "non_tied_target_count": np.int16(non_tied_count),
                    "hold_target_count": np.int16(hold_count),
                    "exit_now_target_count": np.int16(exit_count),
                    "tied_target_count": np.int16(tied_count),
                }
            )

    if not records:
        raise RuntimeError("UNIFIED_EXIT_LIFECYCLE_NO_COMPLETE_EPISODES")
    if any(target_counts[name] <= 0 for name in UNIFIED_EXIT_ACTION_ORDER):
        raise RuntimeError(
            "UNIFIED_EXIT_LIFECYCLE_TARGET_CLASS_DEAD: "
            f"{target_counts}"
        )
    episodes = pd.DataFrame.from_records(
        records,
        columns=UNIFIED_EXIT_LIFECYCLE_EPISODE_COLUMNS,
    )
    proof = {
        "schema_version": UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION,
        "decision": "PASS",
        "entry_rows": int(len(entry_rows)),
        "eligible_entry_rows": int(len(episodes) // len(UNIFIED_EXIT_SIDE_ORDER)),
        "episode_rows": int(len(episodes)),
        "side_order": list(UNIFIED_EXIT_SIDE_ORDER),
        "action_order": list(UNIFIED_EXIT_ACTION_ORDER),
        "path_state_count": path_state_count,
        "target_lookahead_m1_steps": lookahead,
        "required_observed_m1_rows_per_episode": required_rows,
        "m1_row_clock": EXIT_FEATURE_ROW_CLOCK,
        "market_closure_contract": market_closure_contract,
        "target_counts": target_counts,
        "skipped_entry_rows": skipped,
        "target_stream_sha256": target_stream.hexdigest(),
        "split_end_utc": parsed_split_end.isoformat(),
        "entry_side_selection": "both_sides_for_every_causal_entry_snapshot",
        "future_outcomes_used_as_model_inputs": False,
        "path_values_duplicated_into_episode_artifact": False,
    }
    return episodes, proof


# -----------------------------------------------------------------------------
# Path quality (first N bars)
# -----------------------------------------------------------------------------
def _compute_path_quality_first_n(
    *,
    tape: pd.DataFrame,
    horizon_bars: int,
) -> pd.DataFrame:
    if horizon_bars < 1:
        raise RuntimeError("PATH_QUALITY_HORIZON_INVALID")

    cols = list(tape.columns)
    bid_col = "bid_close" if "bid_close" in cols else ("bid" if "bid" in cols else None)
    ask_col = "ask_close" if "ask_close" in cols else ("ask" if "ask" in cols else None)
    if bid_col is None or ask_col is None:
        raise RuntimeError(f"PATH_QUALITY_BID_ASK_MISSING: have={sorted(cols)[:60]}...")

    bid = tape[bid_col].astype(float).to_numpy()
    ask = tape[ask_col].astype(float).to_numpy()

    n = len(tape)
    if n <= horizon_bars:
        raise RuntimeError("PATH_QUALITY_TAPE_TOO_SHORT")

    entry_ask = ask[:-horizon_bars]
    entry_bid = bid[:-horizon_bars]

    mfe_long = np.empty(n - horizon_bars, dtype=np.float64)
    mae_long = np.empty(n - horizon_bars, dtype=np.float64)
    mfe_short = np.empty(n - horizon_bars, dtype=np.float64)
    mae_short = np.empty(n - horizon_bars, dtype=np.float64)

    for i in range(0, n - horizon_bars):
        w_bid = bid[i : i + horizon_bars + 1]
        w_ask = ask[i : i + horizon_bars + 1]
        max_bid = float(np.max(w_bid))
        min_bid = float(np.min(w_bid))
        max_ask = float(np.max(w_ask))
        min_ask = float(np.min(w_ask))

        mfe_long[i] = (
            (max_bid - entry_ask[i]) / np.clip(entry_ask[i], 1e-12, None) * 1e4
        )
        mae_long[i] = (
            (entry_ask[i] - min_bid) / np.clip(entry_ask[i], 1e-12, None) * 1e4
        )
        mfe_short[i] = (
            (entry_bid[i] - min_ask) / np.clip(entry_bid[i], 1e-12, None) * 1e4
        )
        mae_short[i] = (
            (max_ask - entry_bid[i]) / np.clip(entry_bid[i], 1e-12, None) * 1e4
        )

    out = pd.DataFrame(
        {
            "time": tape["time"].iloc[:-horizon_bars].to_numpy(),
            "mfe_long_first_n_bps": mfe_long,
            "mae_long_first_n_bps": mae_long,
            "mfe_short_first_n_bps": mfe_short,
            "mae_short_first_n_bps": mae_short,
            "path_quality_horizon_bars": np.int32(horizon_bars),
        }
    )
    return out


def _compute_bad_path_first_n(
    *,
    tape: pd.DataFrame,
    horizon_bars: int,
    adverse_threshold_bps: float,
    favorable_threshold_bps: float,
) -> pd.DataFrame:
    """V11 redesign: bad_path predicts ACTUAL LOSER, not trajectory shape.

    Old (V10): bad_path = MAE-threshold-hit BEFORE MFE-threshold-hit during trade.
              → predicts volatility, not outcome (anti-calibrated for trending markets).

    New (V11): bad_path = (final_PnL_at_horizon < 0).
              → predicts whether trade ends up a loser at the K-horizon.
              Direct outcome target, no path-shape confounder.
    """
    if horizon_bars < 1:
        raise RuntimeError("BAD_PATH_HORIZON_INVALID")

    cols = list(tape.columns)
    bid_col = "bid_close" if "bid_close" in cols else ("bid" if "bid" in cols else None)
    ask_col = "ask_close" if "ask_close" in cols else ("ask" if "ask" in cols else None)
    if bid_col is None or ask_col is None:
        raise RuntimeError(f"BAD_PATH_BID_ASK_MISSING: have={sorted(cols)[:60]}...")

    bid = tape[bid_col].astype(float).to_numpy()
    ask = tape[ask_col].astype(float).to_numpy()

    n = len(tape)
    if n <= horizon_bars:
        raise RuntimeError("BAD_PATH_TAPE_TOO_SHORT")

    entry_ask = ask[:-horizon_bars]
    entry_bid = bid[:-horizon_bars]
    horizon_bid = bid[horizon_bars:]
    horizon_ask = ask[horizon_bars:]
    pnl_long_at_horizon_bps = (
        (horizon_bid - entry_ask) / np.clip(entry_ask, 1e-12, None) * 1e4
    )
    pnl_short_at_horizon_bps = (
        (entry_bid - horizon_ask) / np.clip(entry_bid, 1e-12, None) * 1e4
    )

    out_long = (pnl_long_at_horizon_bps < 0).astype(np.float32)
    out_short = (pnl_short_at_horizon_bps < 0).astype(np.float32)

    return pd.DataFrame(
        {
            "time": tape["time"].iloc[:-horizon_bars].to_numpy(),
            "bad_path_long_first_n": out_long,
            "bad_path_short_first_n": out_short,
            "bad_path_horizon_bars": np.int32(horizon_bars),
            "bad_path_mae_threshold_bps": np.float32(
                adverse_threshold_bps
            ),  # kept for compat
            "bad_path_mfe_threshold_bps": np.float32(
                favorable_threshold_bps
            ),  # kept for compat
            "v11_pnl_long_at_horizon_bps": pnl_long_at_horizon_bps.astype(np.float32),
            "v11_pnl_short_at_horizon_bps": pnl_short_at_horizon_bps.astype(np.float32),
        }
    )


# -----------------------------------------------------------------------------
# Manifest writing
# -----------------------------------------------------------------------------
def _model_native_ctx_contract_metadata() -> Dict[str, Any]:
    return model_native_context_contract_metadata()


def _require_model_native_manifest_contract(
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the exact signal contract or reject a soft manifest."""

    if not isinstance(extra, dict):
        raise RuntimeError("MODEL_NATIVE_MANIFEST_EXTRA_CONTRACT_MISSING")
    if extra.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(
            "MODEL_NATIVE_MANIFEST_CONTRACT_MODE_INVALID: "
            f"got={extra.get('contract_mode')!r} expected={MODEL_NATIVE_CONTRACT_MODE!r}"
        )
    if extra.get("direction_logit_mode") != MODEL_NATIVE_DIRECTION_LOGIT_MODE:
        raise RuntimeError(
            "MODEL_NATIVE_MANIFEST_DIRECTION_MODE_INVALID: "
            f"got={extra.get('direction_logit_mode')!r} "
            f"expected={MODEL_NATIVE_DIRECTION_LOGIT_MODE!r}"
        )
    signal_contract = extra.get("model_native_signal_contract")
    if not isinstance(signal_contract, dict):
        raise RuntimeError("MODEL_NATIVE_MANIFEST_SIGNAL_CONTRACT_MISSING")
    require_model_native_signal_contract(
        signal_contract,
        context="DATASET_MANIFEST_WRITER",
    )

    signal_bridge = extra.get("signal_bridge")
    if not isinstance(signal_bridge, dict):
        raise RuntimeError("MODEL_NATIVE_MANIFEST_SIGNAL_SURFACE_MISSING")
    if signal_bridge.get("id") != MODEL_NATIVE_SIGNAL_SCHEMA_VERSION:
        raise RuntimeError(
            "MODEL_NATIVE_MANIFEST_SIGNAL_SCHEMA_INVALID: "
            f"got={signal_bridge.get('id')!r} expected={MODEL_NATIVE_SIGNAL_SCHEMA_VERSION!r}"
        )
    if int(signal_bridge.get("seq_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            "MODEL_NATIVE_MANIFEST_SEQ_WIDTH_INVALID: "
            f"got={signal_bridge.get('seq_input_dim')!r} expected={MODEL_NATIVE_SIGNAL_DIM}"
        )
    if int(signal_bridge.get("snap_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(
            "MODEL_NATIVE_MANIFEST_SNAP_WIDTH_INVALID: "
            f"got={signal_bridge.get('snap_input_dim')!r} expected={MODEL_NATIVE_SIGNAL_DIM}"
        )
    if signal_bridge.get("fields") != signal_contract.get("fields"):
        raise RuntimeError("MODEL_NATIVE_MANIFEST_ORDERED_FIELDS_MISMATCH")
    if signal_bridge.get("bridge_dim") != 0:
        raise RuntimeError(
            "MODEL_NATIVE_MANIFEST_BRIDGE_DIM_INVALID: "
            f"got={signal_bridge.get('bridge_dim')!r} expected=0"
        )
    if signal_bridge.get("bridge_source") is not None:
        raise RuntimeError(
            "MODEL_NATIVE_MANIFEST_BRIDGE_SOURCE_FORBIDDEN: "
            f"got={signal_bridge.get('bridge_source')!r}"
        )

    ctx_contract = extra.get("ctx_contract")
    if not isinstance(ctx_contract, dict):
        raise RuntimeError("MODEL_NATIVE_MANIFEST_CTX_CONTRACT_MISSING")
    exact_ctx = _model_native_ctx_contract_metadata()
    for key in (
        "tag",
        "ctx_cont_dim",
        "ctx_cat_dim",
        "ctx_cont_names",
        "ctx_cat_names",
    ):
        expected = exact_ctx[key]
        if ctx_contract.get(key) != expected:
            raise RuntimeError(
                "MODEL_NATIVE_MANIFEST_CTX_CONTRACT_INVALID: "
                f"{key}={ctx_contract.get(key)!r} expected={expected!r}"
            )

    state_contract = extra.get("model_native_state_contract")
    if not isinstance(state_contract, dict):
        raise RuntimeError("MODEL_NATIVE_MANIFEST_STATE_CONTRACT_MISSING")
    try:
        validate_state_contract_metadata_v2(state_contract, require_artifact=False)
    except RuntimeError as exc:
        raise RuntimeError(
            f"MODEL_NATIVE_MANIFEST_STATE_CONTRACT_INVALID: {exc}"
        ) from exc
    for key in (
        "rank_reference_npz",
        "rank_reference_sidecar_json",
        "rank_reference_source_parquet",
    ):
        if not str(state_contract.get(key) or "").strip():
            raise RuntimeError(f"MODEL_NATIVE_MANIFEST_STATE_FIELD_MISSING: {key}")
    for key in (
        "rank_reference_npz_sha256",
        "rank_reference_sidecar_sha256",
        "rank_reference_source_parquet_sha256",
    ):
        value = str(state_contract.get(key) or "").strip().lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise RuntimeError(f"MODEL_NATIVE_MANIFEST_STATE_HASH_INVALID: {key}")
    if int(state_contract.get("rank_reference_fit_row_count") or 0) <= 0:
        raise RuntimeError("MODEL_NATIVE_MANIFEST_STATE_ROW_COUNT_INVALID")
    return signal_contract


def _require_model_native_seq513_split_manifest_contract(
    *,
    splits: Dict[str, Any],
    extra: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate the exact split windows and model-native signal contract."""

    if set(splits) != {"train", "val", "test"}:
        raise RuntimeError(
            "MODEL_NATIVE_SPLIT_SET_INVALID: "
            f"got={sorted(splits)} expected=['test', 'train', 'val']"
        )
    for split_name in ("train", "val", "test"):
        window = splits.get(split_name)
        if not isinstance(window, dict) or set(window) != {"start", "end"}:
            raise RuntimeError(
                "MODEL_NATIVE_SPLIT_WINDOW_INVALID: "
                f"split={split_name} window={window!r}"
            )
        if (
            not str(window.get("start") or "").strip()
            or not str(window.get("end") or "").strip()
        ):
            raise RuntimeError(f"MODEL_NATIVE_SPLIT_WINDOW_EMPTY: split={split_name}")
    parsed = {
        f"{split_name}_{edge}": _parse_ts(str(splits[split_name][edge]))
        for split_name in ("train", "val", "test")
        for edge in ("start", "end")
    }
    if any(value is None for value in parsed.values()):
        raise RuntimeError("MODEL_NATIVE_SPLIT_WINDOW_TIMESTAMP_INVALID")
    if not (
        parsed["train_start"]
        <= parsed["train_end"]
        < parsed["val_start"]
        <= parsed["val_end"]
        < parsed["test_start"]
        <= parsed["test_end"]
    ):
        raise RuntimeError("MODEL_NATIVE_SPLIT_WINDOW_ORDER_INVALID")
    signal_contract = _require_model_native_manifest_contract(extra)
    state_contract = (
        extra.get("model_native_state_contract") if isinstance(extra, dict) else {}
    )
    state_fit_start = _parse_ts(str(state_contract.get("rank_fit_start_utc") or ""))
    state_fit_end = _parse_ts(str(state_contract.get("rank_fit_end_utc") or ""))
    if state_fit_start != parsed["train_start"] or state_fit_end != parsed["train_end"]:
        raise RuntimeError(
            "MODEL_NATIVE_STATE_TRAIN_WINDOW_MISMATCH: "
            f"state={state_fit_start}..{state_fit_end} "
            f"train={parsed['train_start']}..{parsed['train_end']}"
        )
    mtf_binding = extra.get("multi_tf_cache_binding")
    expected_mtf_keys = {
        "cache_dir",
        "manifest_path",
        "manifest_sha256",
        "cache_identity_sha256",
        "m5_prebuilt_source",
        "m5_prebuilt_source_sha256",
    }
    if not isinstance(mtf_binding, dict) or set(mtf_binding) != expected_mtf_keys:
        raise RuntimeError("MODEL_NATIVE_SPLIT_MTF_CACHE_BINDING_INVALID")
    for key in (
        "manifest_sha256",
        "cache_identity_sha256",
        "m5_prebuilt_source_sha256",
    ):
        value = str(mtf_binding.get(key) or "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise RuntimeError(
                f"MODEL_NATIVE_SPLIT_MTF_CACHE_HASH_INVALID: {key}"
            )
    manifest_path = Path(
        str(mtf_binding.get("manifest_path") or "")
    ).expanduser().resolve()
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or _sha256_file(manifest_path) != mtf_binding["manifest_sha256"]
        or manifest_path.parent
        != Path(str(mtf_binding["cache_dir"])).expanduser().resolve()
    ):
        raise RuntimeError("MODEL_NATIVE_SPLIT_MTF_CACHE_MANIFEST_MISMATCH")
    return signal_contract


def write_manifest(
    *,
    output_path: Path,
    build_command: List[str],
    source_parquet: Path,
    tape_root: Optional[Path],
    splits: Optional[Dict[str, Any]] = None,
    ts_min_max_by_split: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    source_parquet = Path(source_parquet).expanduser().resolve()
    if not source_parquet.is_file():
        raise RuntimeError(f"MODEL_NATIVE_SOURCE_PARQUET_MISSING: {source_parquet}")
    signal_contract: Dict[str, Any]
    if splits is not None:
        signal_contract = _require_model_native_seq513_split_manifest_contract(
            splits=splits,
            extra=extra,
        )
    else:
        signal_contract = _require_model_native_manifest_contract(extra)
    extra_ctx = extra["ctx_contract"]
    ctx_cont_micro = list(extra_ctx.get("ctx_cont_micro_features") or [])
    ctx_cont_swing = list(extra_ctx.get("ctx_cont_swing_features") or [])
    ctx_tag = str(extra_ctx["tag"])
    signal_bridge_id = MODEL_NATIVE_SIGNAL_SCHEMA_VERSION
    signal_bridge_sha = str(signal_contract["static_contract_sha256"])
    signal_bridge_fields = list(signal_contract["fields"])

    manifest: Dict[str, Any] = {
        "created_at": _utc_now_iso(),
        "git_commit": get_git_commit(),
        "output_data_path": str(output_path),
        "build_command": build_command,
        "inputs": {
            "source_parquet": str(source_parquet),
            "tape_root": str(tape_root) if tape_root is not None else None,
        },
        "feature_contract": {
            "ctx_tag": ctx_tag,
            "ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
            "ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
            "ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
            "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
            "ctx_cont_micro_features": list(ctx_cont_micro),
            "ctx_cont_swing_features": list(ctx_cont_swing),
            "signal_bridge_id": signal_bridge_id,
            "signal_bridge_contract_sha256": signal_bridge_sha,
            "signal_bridge_fields": signal_bridge_fields,
        },
        "splits": splits,
        "ts_min_max_by_split": ts_min_max_by_split or {},
        "notes": notes,
    }
    if splits is not None:
        manifest.update(
            {
                "schema_version": MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
                "manifest_variant": MODEL_NATIVE_CONTRACT_MODE,
                "expected_seq_snap_width": MODEL_NATIVE_SIGNAL_DIM,
            }
        )
    if extra:
        manifest["extra"] = extra

    manifest_path = output_path.parent / f"{output_path.stem}.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    log.info(f"MANIFEST WRITTEN: {manifest_path}")
    return manifest_path


def _model_native_state_contract(
    *,
    args: argparse.Namespace,
    feature_history_start: pd.Timestamp,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> Dict[str, Any]:
    entry_run_id = require_entry_run_id(getattr(args, "run_id", None))
    raw_npz = str(getattr(args, "model_native_rank_reference_npz", "") or "").strip()
    if not raw_npz:
        raise RuntimeError(
            "MODEL_NATIVE_RANK_REFERENCE_REQUIRED: pass an explicit audited NPZ"
        )
    npz_path = Path(raw_npz).expanduser().resolve()
    if not npz_path.is_file():
        raise RuntimeError(f"MODEL_NATIVE_RANK_REFERENCE_MISSING: {npz_path}")
    sidecar_path = npz_path.with_suffix(npz_path.suffix + ".json")
    if not sidecar_path.is_file():
        raise RuntimeError(
            f"MODEL_NATIVE_RANK_REFERENCE_SIDECAR_MISSING: {sidecar_path}"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        raise RuntimeError("MODEL_NATIVE_RANK_REFERENCE_SIDECAR_INVALID")
    npz_sha = _sha256_file(npz_path)
    sidecar_npz = Path(str(sidecar.get("out_npz") or "")).expanduser().resolve()
    if sidecar_npz != npz_path:
        raise RuntimeError(
            f"MODEL_NATIVE_RANK_REFERENCE_PATH_MISMATCH: sidecar={sidecar_npz} actual={npz_path}"
        )
    if str(sidecar.get("out_npz_sha256") or "").strip().lower() != npz_sha:
        raise RuntimeError(
            "MODEL_NATIVE_RANK_REFERENCE_SHA_MISMATCH: "
            f"sidecar={sidecar.get('out_npz_sha256')!r} actual={npz_sha} path={npz_path}"
        )
    source_path = Path(str(sidecar.get("source_parquet") or "")).expanduser().resolve()
    if not source_path.is_file():
        raise RuntimeError(f"MODEL_NATIVE_RANK_REFERENCE_SOURCE_MISSING: {source_path}")
    source_sha = _sha256_file(source_path)
    if str(sidecar.get("source_parquet_sha256") or "").strip().lower() != source_sha:
        raise RuntimeError(
            "MODEL_NATIVE_RANK_REFERENCE_SOURCE_SHA_MISMATCH: "
            f"sidecar={sidecar.get('source_parquet_sha256')!r} actual={source_sha} path={source_path}"
        )
    declared_builder_source = (
        Path(str(getattr(args, "source_parquet", "") or "")).expanduser().resolve()
    )
    declared_rank_source = (
        Path(str(getattr(args, "canonical_v2_parquet", "") or ""))
        .expanduser()
        .resolve()
    )
    if declared_rank_source != source_path:
        raise RuntimeError(
            "MODEL_NATIVE_RANK_REFERENCE_CANONICAL_SOURCE_MISMATCH: "
            f"rank_source={source_path} canonical_source={declared_rank_source}"
        )
    market_identity = require_train_rank_source_market_identity_v2(
        rank_source_parquet=source_path,
        model_source_parquet=declared_builder_source,
        history_start_utc=feature_history_start,
        fit_end_utc=train_end,
    )
    reference = load_train_rank_reference_v2(npz_path, expected_sha256=npz_sha)
    rank_run_id = str(reference.sidecar.get("entry_run_id") or "").strip()
    if rank_run_id != entry_run_id:
        raise RuntimeError(
            "MODEL_NATIVE_RANK_REFERENCE_RUN_ID_MISMATCH: "
            f"rank={rank_run_id!r} build={entry_run_id!r}"
        )
    sidecar_history_start = _parse_ts(str(sidecar.get("history_start_utc") or ""))
    if sidecar_history_start != feature_history_start:
        raise RuntimeError(
            "MODEL_NATIVE_RANK_REFERENCE_HISTORY_START_MISMATCH: "
            f"sidecar={sidecar_history_start} build={feature_history_start}"
        )
    if reference.fit_start_utc != train_start:
        raise RuntimeError(
            "MODEL_NATIVE_RANK_REFERENCE_FIT_START_MISMATCH: "
            f"sidecar={reference.fit_start_utc} build={train_start}"
        )
    if reference.fit_end_utc != train_end:
        raise RuntimeError(
            "MODEL_NATIVE_RANK_REFERENCE_FIT_END_MISMATCH: "
            f"sidecar={reference.fit_end_utc} build={train_end}"
        )
    if not feature_history_start <= train_start <= train_end:
        raise RuntimeError("MODEL_NATIVE_STATE_CONTRACT_TIME_ORDER_INVALID")
    if (
        not str(sidecar.get("fit_time_min") or "").strip()
        or not str(sidecar.get("fit_time_max") or "").strip()
    ):
        raise RuntimeError("MODEL_NATIVE_RANK_REFERENCE_TIME_RANGE_MISSING")
    return {
        "schema_version": MODEL_NATIVE_STATE_SCHEMA_VERSION,
        "source": "entry_v10_ctx_dataset_manifest",
        "feature_history_start_utc": str(feature_history_start),
        "rank_fit_start_utc": str(train_start),
        "rank_fit_end_utc": str(train_end),
        "rank_reference_npz": str(npz_path),
        "rank_reference_npz_sha256": npz_sha,
        "rank_reference_sidecar_sha256": reference.sidecar_sha256,
        "rank_reference_schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
        "rank_reference_sidecar_json": str(sidecar_path),
        "rank_reference_fit_row_count": int(reference.fit_row_count),
        "rank_reference_fit_time_min": str(sidecar["fit_time_min"]),
        "rank_reference_fit_time_max": str(sidecar["fit_time_max"]),
        "rank_reference_source_parquet": str(source_path),
        "rank_reference_source_parquet_sha256": source_sha,
        "rank_reference_model_source_market_identity": market_identity,
        "rank_reference_fit_scope": "train_only",
        "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
        "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
        "split_reset_allowed": False,
        "post_fit_rows_in_rank_reference": False,
        "runtime_rule_free": True,
        "entry_run_id": entry_run_id,
    }


# -----------------------------------------------------------------------------
# Label proof
# -----------------------------------------------------------------------------
_SESSION_ID_TO_NAME = SESSION_NAME_BY_ID


def _log_label_distribution_proof(df: pd.DataFrame, split: str) -> None:
    if split == "test":
        log.info(
            "[ENTRY_LABEL_DISTRIBUTION_PROOF] split=test "
            "status=withheld_until_final_candidate_evaluation"
        )
        return
    if "y_direction" not in df.columns:
        log.warning(
            "[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s status=no_y_direction", split
        )
        return
    y = df["y_direction"].astype(int)
    n = int(len(y))
    if n == 0:
        log.warning("[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s status=empty", split)
        return
    long_c = int((y == 0).sum())
    short_c = int((y == 1).sum())
    flat_c = int((y == 2).sum())
    log.info(
        "[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s n=%d long=%d (%.4f) short=%d (%.4f) flat=%d (%.4f)",
        split,
        n,
        long_c,
        long_c / n,
        short_c,
        short_c / n,
        flat_c,
        flat_c / n,
    )
    log.info(
        "[ENTRY_FLAT_LABEL_PROOF] split=%s flat=%d flat_rate=%.4f status=%s",
        split,
        flat_c,
        flat_c / n,
        "OK" if flat_c > 0 else "EMPTY",
    )
    if "ctx_cat" not in df.columns:
        return
    try:
        sess_ids = df["ctx_cat"].apply(
            lambda v: int(v[0]) if isinstance(v, (list, tuple)) and len(v) > 0 else None
        )
        df_s = pd.DataFrame({"y": y, "session_id": sess_ids}).dropna(
            subset=["session_id"]
        )
        if df_s.empty:
            return
        for sid, grp in df_s.groupby("session_id"):
            sid_int = int(sid)
            n_s = int(len(grp))
            long_s = int((grp["y"] == 0).sum())
            short_s = int((grp["y"] == 1).sum())
            flat_s = int((grp["y"] == 2).sum())
            log.info(
                "[ENTRY_LABEL_BY_SESSION_PROOF] split=%s session=%s session_id=%d n=%d long=%d (%.4f) short=%d (%.4f) flat=%d (%.4f)",
                split,
                _SESSION_ID_TO_NAME.get(sid_int, "UNKNOWN"),
                sid_int,
                n_s,
                long_s,
                long_s / n_s,
                short_s,
                short_s / n_s,
                flat_s,
                flat_s / n_s,
            )
    except Exception:
        return


# -----------------------------------------------------------------------------
# Core builder
# -----------------------------------------------------------------------------
# The Group-A worker is deliberately serial: it shares the full-history context
# and allocates only one bounded 4096-row result at a time. This is a capacity
# contract, not a tunable throughput preference.
_MODEL_NATIVE_GROUP_A_RECOMPUTE_WORKERS = 1
_MODEL_NATIVE_GROUP_A_CHECKPOINT_SCHEMA_VERSION = "entry_dataset_group_a_checkpoint_v3"
_MODEL_NATIVE_STREAMING_BATCH_SIZE = 512


def _align_native_m5_feature_surface(
    *,
    target_times: Sequence[Any],
    surface_times: pd.DatetimeIndex,
    surface_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return zero-copy M5 surface views for one exact contiguous source window."""

    target = pd.DatetimeIndex(
        pd.to_datetime(target_times, utc=True, errors="coerce")
    ).as_unit("ns")
    surface = pd.DatetimeIndex(surface_times).as_unit("ns")
    if (
        len(target) == 0
        or target.hasnans
        or not target.is_unique
        or not target.is_monotonic_increasing
        or not target.floor(f"{ENTRY_DECISION_BAR_SECONDS}s").equals(target)
    ):
        raise RuntimeError("ENTRY_M5_FEATURE_SURFACE_TARGET_TIME_INVALID")
    if (
        len(surface) == 0
        or surface.hasnans
        or not surface.is_unique
        or not surface.is_monotonic_increasing
        or not surface.floor(f"{ENTRY_DECISION_BAR_SECONDS}s").equals(surface)
    ):
        raise RuntimeError("ENTRY_M5_FEATURE_SURFACE_TIME_INVALID")
    expected_shapes = {
        "signal": (len(surface), MODEL_NATIVE_SIGNAL_DIM),
        "ctx_cont": (len(surface), MODEL_NATIVE_CTX_CONT_DIM),
        "ctx_cat": (len(surface), MODEL_NATIVE_CTX_CAT_DIM),
    }
    expected_dtypes = {
        "signal": np.dtype(np.float32),
        "ctx_cont": np.dtype(np.float32),
        "ctx_cat": np.dtype(np.int64),
    }
    if set(surface_arrays) != set(expected_shapes):
        raise RuntimeError("ENTRY_M5_FEATURE_SURFACE_ARRAY_SCHEMA_INVALID")
    for name, expected_shape in expected_shapes.items():
        values = surface_arrays[name]
        if (
            not isinstance(values, np.ndarray)
            or values.shape != expected_shape
            or values.dtype != expected_dtypes[name]
        ):
            raise RuntimeError(
                f"ENTRY_M5_FEATURE_SURFACE_{name.upper()}_INVALID: "
                f"shape={getattr(values, 'shape', None)} "
                f"dtype={getattr(values, 'dtype', None)}"
            )
    positions = surface.get_indexer(target)
    if np.any(positions < 0):
        raise RuntimeError(
            "ENTRY_M5_FEATURE_SURFACE_TIME_MISSING: "
            f"rows={int(np.count_nonzero(positions < 0))}"
        )
    if len(positions) > 1 and not np.all(np.diff(positions) == 1):
        raise RuntimeError("ENTRY_M5_FEATURE_SURFACE_WINDOW_NONCONTIGUOUS")
    window = slice(int(positions[0]), int(positions[-1]) + 1)
    if not surface[window].equals(target):
        raise RuntimeError("ENTRY_M5_FEATURE_SURFACE_WINDOW_IDENTITY_MISMATCH")
    return {name: values[window] for name, values in surface_arrays.items()}


def build_dataset_canonical(
    *,
    source_parquet: Path,
    tape_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_rows: Optional[int],
    seq_len: int,
    early_move_threshold_bps: float,
    canonical_v2_parquet: Path,
    seq_structure_manifest_path: Path,
    model_native_rank_reference_npz: Path,
    m5_feature_surface_times: pd.DatetimeIndex,
    m5_feature_surface_arrays: Mapping[str, np.ndarray],
    m5_feature_surface_binding: Mapping[str, Any],
    emit_start: pd.Timestamp,
    emit_end: pd.Timestamp,
    split_name: Optional[str] = None,
    output_path: Optional[Path] = None,  # V2 streaming-write target
    streaming_batch_size: int = _MODEL_NATIVE_STREAMING_BATCH_SIZE,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ctx = _hard_gate_model_native_context()
    signal_build_contract = _signal_build_contract_from_manifest(
        seq_structure_manifest_path
    )
    active_base_signal_fields = list(signal_build_contract["base_fields"])

    if seq_len != MODEL_NATIVE_SEQ_LEN:
        raise RuntimeError(
            f"MODEL_NATIVE_SEQ_LEN_INVALID: got={seq_len} expected={MODEL_NATIVE_SEQ_LEN}"
        )
    if start is None or end is None or start > end:
        raise RuntimeError(f"MODEL_RANGE_INVALID: start={start} end={end}")
    if (
        emit_start is None
        or emit_end is None
        or not (start <= emit_start <= emit_end <= end)
    ):
        raise RuntimeError(
            "MODEL_NATIVE_EMIT_WINDOW_INVALID: "
            f"history_start={start} emit_start={emit_start} emit_end={emit_end} computation_end={end}"
        )
    # 1) Resolve the one exact source parquet. Manifest indirection and BASE28
    # compatibility inputs are retired; lineage is bound by byte hash below.
    parquet_path = Path(source_parquet).expanduser().resolve()
    if not parquet_path.is_file():
        raise RuntimeError(f"SOURCE_PARQUET_MISSING: {parquet_path}")
    parquet_sha = _sha256_file(parquet_path)
    log.info("[SOURCE_PARQUET] %s sha256=%s", parquet_path, parquet_sha)

    # 2) Load parquet
    df = pd.read_parquet(parquet_path)
    df = df.reset_index(drop=False)
    time_col = _detect_time_col(df)
    df = _normalize_time_utc(df, time_col)
    assert_no_price_scale_glitch(
        df,
        context="MODEL_NATIVE_M5_SOURCE",
    )

    if "is_model_bar" in df.columns:
        raise RuntimeError(
            "SOURCE_PARQUET_MIXED_GRANULARITY_FORBIDDEN: exact M5 source required"
        )

    # filter by start/end
    if start is not None:
        df = df[df["time"] >= start]
    if end is not None:
        df = df[df["time"] <= end]

    if len(df) == 0:
        raise RuntimeError("NO_ROWS_AFTER_FILTERS")

    if df["time"].duplicated().any() or not df["time"].is_monotonic_increasing:
        raise RuntimeError("MODEL_NATIVE_HISTORY_TIME_ORDER_INVALID")
    train_rank_reference = load_train_rank_reference_v2(
        Path(model_native_rank_reference_npz).expanduser().resolve()
    )
    df = apply_train_rank_reference_v2(df, train_rank_reference)
    log.info(
        "[MODEL_NATIVE_TRAIN_ONLY_RANK] fit=%s..%s rows=%d history=%s..%s",
        train_rank_reference.fit_start_utc,
        train_rank_reference.fit_end_utc,
        train_rank_reference.fit_row_count,
        start,
        end,
    )

    # deterministic head
    if max_rows and len(df) > max_rows:
        df = df.head(int(max_rows)).copy()

    # 3) Validate the contracted session state and derive additional learned
    # session-context inputs from canonical UTC timestamps. Missing or malformed
    # session state is an upstream contract failure, never an ASIA/default value.
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if ts.isna().any():
        raise RuntimeError("TIME_PARSE_FAIL_FOR_SESSION_CONTEXT")
    if "session_id" not in df.columns:
        raise RuntimeError(
            "SESSION_ID_MISSING: canonical source must carry exact session state"
        )
    raw_session_id = pd.to_numeric(df["session_id"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(raw_session_id).all():
        raise RuntimeError(
            "SESSION_ID_NONFINITE: no missing-session default is allowed"
        )
    if not np.equal(raw_session_id, np.floor(raw_session_id)).all():
        raise RuntimeError("SESSION_ID_NONINTEGER: expected exact integer ids 0..3")
    session_id = raw_session_id.astype(np.int32)
    invalid_session_ids = sorted(
        set(session_id.tolist())
        - set(MODEL_NATIVE_CTX_CAT_DOMAINS["session_id"])
    )
    if invalid_session_ids:
        raise RuntimeError(f"SESSION_ID_OUT_OF_CONTRACT: {invalid_session_ids}")
    from gx1.time.session_detector import m5_decision_availability

    decision_ts = m5_decision_availability(ts)
    canonical_session_id = get_session_id_vectorized(decision_ts).to_numpy(dtype=np.int32)
    if not np.array_equal(session_id, canonical_session_id):
        mismatch = int(np.count_nonzero(session_id != canonical_session_id))
        raise RuntimeError(
            "SESSION_ID_CANONICAL_MISMATCH: "
            f"rows={mismatch}; source session state cannot pass through"
        )
    df["session_id"] = canonical_session_id
    n_unique_post = int(np.unique(session_id).size)
    if n_unique_post < 2:
        raise RuntimeError(
            f"SESSION_ID_DEGENERATE: unique={n_unique_post} "
            f"(expected >=2). Check prebuilt time/session pipeline."
        )
    # Log distribution proof
    _sess_counts = pd.Series(df["session_id"]).value_counts(dropna=False).sort_index()
    log.info("[SESSION_ID_DISTRIBUTION_PROOF] %s", _sess_counts.to_dict())
    df["is_ASIA"] = (
        canonical_session_id == ASIA_SESSION_ID
    ).astype(np.int8)
    # decision_ts is a DatetimeIndex, so the session helpers return Series
    # indexed by timestamp; df carries the window-filtered integer index.
    # Assign positionally — label alignment here silently yields all-NaN.
    df["minutes_since_session_open"] = get_session_minutes_since_open_vectorized(
        decision_ts
    ).to_numpy(dtype=np.float32)
    df["minutes_to_next_session_boundary"] = (
        get_session_minutes_to_next_boundary_vectorized(decision_ts).to_numpy(
            dtype=np.float32
        )
    )
    session_change = np.zeros(len(canonical_session_id), dtype=np.int8)
    if len(canonical_session_id) > 1:
        session_change[1:] = (
            canonical_session_id[1:] != canonical_session_id[:-1]
        ).astype(np.int8)
    df["session_change_flag"] = session_change
    df["session_tradable"] = (
        canonical_session_id != ASIA_SESSION_ID
    ).astype(np.int8)

    # The train-rank state contract above is the sole owner of exact ATR/spread
    # evidence. Never synthesize a second clipped volatility surface here.
    rank_context_names = ("atr", "atr_bps", "spread_bps")
    rank_context_missing = [
        name for name in rank_context_names if name not in df.columns
    ]
    if rank_context_missing:
        raise RuntimeError(f"MODEL_NATIVE_RANK_CONTEXT_MISSING: {rank_context_missing}")
    rank_context = df[list(rank_context_names)].to_numpy(dtype=np.float64)
    if not np.isfinite(rank_context).all():
        raise RuntimeError("MODEL_NATIVE_RANK_CONTEXT_NONFINITE")
    if np.any(rank_context[:, :2] <= 0.0) or np.any(rank_context[:, 2] < 0.0):
        raise RuntimeError("MODEL_NATIVE_RANK_CONTEXT_OUT_OF_RANGE")

    log.info(
        "[MODEL_NATIVE_SIGNAL] exact base fields=%d selected fields=%d total=%d rows=%d",
        len(active_base_signal_fields),
        len(signal_build_contract["model_native_signal_contract"]["selected_fields"]),
        len(signal_build_contract["model_native_signal_contract"]["fields"]),
        len(df),
    )

    # 6) Build ctx features
    ctx_cont_names = list(ctx["ctx_cont_source_prefix_names"])
    # ctx_cat is the unconditional five-field model-native contract; the stale
    # degenerate trend_regime_id base anchor is excluded. This also fixes the
    # H4 case: the v3 contract uses capital "H4_trend_sign_cat" exactly as add_ctx_cont emits it
    # (the v1 base used lowercase "h4_trend_sign_cat", which mismatched the prebuilt column).
    # Symmetric with the ctx_cont_names = MODEL_NATIVE_CTX_CONT_FIELDS upgrade later in this builder.
    ctx_cat_names = list(MODEL_NATIVE_CTX_CAT_FIELDS)

    # 7) Assemble per-bar signal dataframe (time aligned)
    df_sig = pd.DataFrame({"time": df["time"].to_numpy()})

    # 8) Labels from canonical tape lane (join by time)
    t_min = pd.Timestamp(df_sig["time"].min()).tz_convert("UTC")
    t_max = pd.Timestamp(df_sig["time"].max()).tz_convert("UTC")

    # Spread-aware risk supervision is mandatory: long enters ASK and marks BID;
    # short enters BID and marks ASK.  Mid OHLC remains only the timing reference.
    _risk_tape_cols = [
        "bid_close",
        "ask_close",
        "bid_high",
        "bid_low",
        "ask_high",
        "ask_low",
        "open",
        "high",
        "low",
        "close",
    ]
    tape = _load_canonical_tape(
        tape_root=tape_root,
        t_min=t_min,
        t_max=t_max,
        required_cols=_risk_tape_cols,
    )

    # Inner join tape to the source by time. Every exact M5 source timestamp
    # must resolve to one tape row; the tape may cover a wider time range.
    merged = df_sig.merge(tape, on="time", how="inner", validate="one_to_one")
    rows_source = int(len(df_sig))
    rows_tape = int(len(tape))
    rows_joined = int(len(merged))
    full_source_match = int(rows_source == rows_joined)
    log.info(
        "[ENTRY_TAPE_JOIN_PROOF] rows_source=%d rows_tape=%d rows_joined=%d full_source_match=%d",
        rows_source,
        rows_tape,
        rows_joined,
        full_source_match,
    )
    if not full_source_match:
        raise RuntimeError(
            f"TAPE_JOIN_STRICT_FAIL: rows_source={rows_source} rows_tape={rows_tape} rows_joined={rows_joined}"
        )

    # Micro-structure features (canonical tape OHLC), recomputed by their only
    # formula owner. Source copies are discarded below and can never win.
    if not all(c in merged.columns for c in ("close", "high", "low")):
        raise RuntimeError(
            "MICRO_FEATURES_MISSING: require close/high/low in canonical tape"
        )
    tape_feat = merged[["time", "close", "high", "low"]].copy().sort_values("time")
    close = tape_feat["close"].to_numpy(dtype=np.float64)
    high = tape_feat["high"].to_numpy(dtype=np.float64)
    low = tape_feat["low"].to_numpy(dtype=np.float64)
    for _name, _arr in compute_micro_structure_features(high, low, close).items():
        tape_feat[_name] = _arr

    # Swing-structure features (ATR-normalized) — ONE TRUTH: gx1.features.swing_structure_v1
    # (lookahead-safe confirmation lag). Shared with the live serve augmenter
    # (v12_ctx_augment_live._add_swing_features) so train == serve bit-for-bit; do NOT
    # re-implement the math here (2026-06-24 unification — was a 2nd, edge-divergent copy).
    _swing = compute_swing_structure_features(
        high,
        low,
        close,
        lookback=SWING_LOOKBACK_V1,
        atr_period=SWING_ATR_PERIOD_V1,
    )
    for _name, _arr in _swing.items():
        tape_feat[_name] = _arr
    log.info(
        "[ENTRY_SWING_PIVOT_PROOF] swing_resets_high=%d swing_resets_low=%d",
        int((np.diff(_swing["bars_since_swing_high"]) < 0).sum()),
        int((np.diff(_swing["bars_since_swing_low"]) < 0).sum()),
    )

    # Attach only canonical computed values to source rows (strict 1:1 time
    # alignment). Dropping source copies closes the former `_tape` pass-through.
    tape_context_names = list(MODEL_NATIVE_CTX_CONT_MICRO_FIELDS) + list(
        MODEL_NATIVE_CTX_CONT_SWING_FIELDS
    )
    source_context_copies = [name for name in tape_context_names if name in df.columns]
    if source_context_copies:
        log.info(
            "[ENTRY_TAPE_CONTEXT_RECOMPUTE] discarding_source_copies=%s",
            source_context_copies,
        )
        df = df.drop(columns=source_context_copies)
    df = df.merge(
        tape_feat[["time"] + tape_context_names],
        on="time",
        how="inner",
        validate="one_to_one",
    )
    if len(df) != rows_source:
        raise RuntimeError(
            f"MICRO_FEATURE_JOIN_FAIL: rows_source={rows_source} rows_after={len(df)}"
        )
    ctx_cont_names = (
        ctx_cont_names
        + list(MODEL_NATIVE_CTX_CONT_MICRO_FIELDS)
        + list(MODEL_NATIVE_CTX_CONT_SWING_FIELDS)
        + list(MODEL_NATIVE_CTX_CONT_SESSION_FIELDS)
    )
    log.info(
        "[ENTRY_MICRO_FEATURES_PROOF] names=%s count=%d",
        list(MODEL_NATIVE_CTX_CONT_MICRO_FIELDS),
        len(MODEL_NATIVE_CTX_CONT_MICRO_FIELDS),
    )
    log.info(
        "[ENTRY_SWING_FEATURES_PROOF] names=%s count=%d",
        list(MODEL_NATIVE_CTX_CONT_SWING_FIELDS),
        len(MODEL_NATIVE_CTX_CONT_SWING_FIELDS),
    )
    log.info(
        "[ENTRY_SESSION_CTX_PROOF] names=%s count=%d",
        list(MODEL_NATIVE_CTX_CONT_SESSION_FIELDS),
        len(MODEL_NATIVE_CTX_CONT_SESSION_FIELDS),
    )

    for name in ctx_cont_names:
        if name not in df.columns:
            raise RuntimeError(f"CTX_CONT_MISSING_IN_SOURCE: {name!r}")
    for name in ctx_cat_names:
        if name not in df.columns:
            raise RuntimeError(f"CTX_CAT_MISSING_IN_SOURCE: {name!r}")

    # Normalize ctx dtypes
    df_ctx_cont = df[ctx_cont_names].astype(np.float32)
    df_ctx_cat = df[ctx_cat_names].astype(np.int64)
    cont_matrix = df_ctx_cont.to_numpy()
    if not np.isfinite(cont_matrix).all():
        bad_mask = ~np.isfinite(cont_matrix)
        bad_columns = {
            name: int(count)
            for name, count in zip(ctx_cont_names, bad_mask.sum(axis=0))
            if count
        }
        first_bad_row = int(np.argwhere(bad_mask.any(axis=1))[0][0])
        first_bad_time = (
            str(df["time"].iloc[first_bad_row]) if "time" in df.columns else "unknown"
        )
        raise RuntimeError(
            "CTX_CONT_NONFINITE_IN_SOURCE: "
            f"columns={bad_columns} first_bad_row={first_bad_row} "
            f"first_bad_time={first_bad_time}"
        )
    if not np.isfinite(df_ctx_cat.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("CTX_CAT_NONFINITE_IN_SOURCE")

    path_quality = _compute_path_quality_first_n(
        tape=merged[
            ["time"]
            + [
                c
                for c in merged.columns
                if c in ("bid_close", "ask_close", "bid", "ask")
            ]
        ].copy(),
        horizon_bars=PATH_QUALITY_HORIZON_BARS,
    )
    bad_path = _compute_bad_path_first_n(
        tape=merged[
            ["time"]
            + [
                c
                for c in merged.columns
                if c in ("bid_close", "ask_close", "bid", "ask")
            ]
        ].copy(),
        horizon_bars=BAD_PATH_HORIZON_BARS,
        adverse_threshold_bps=BAD_PATH_MAE_THRESHOLD_BPS,
        favorable_threshold_bps=BAD_PATH_MFE_THRESHOLD_BPS,
    )

    # Direction, early-move, quality and bad-path targets are selected below
    # from the final spread-aware H=24 direction side. No bootstrap direction
    # or fixed-duration Exit label is admitted.
    merged2 = merged.merge(
        path_quality[
            [
                "time",
                "mfe_long_first_n_bps",
                "mae_long_first_n_bps",
                "mfe_short_first_n_bps",
                "mae_short_first_n_bps",
                "path_quality_horizon_bars",
            ]
        ],
        on="time",
        how="inner",
        validate="one_to_one",
    )
    merged2 = merged2.merge(
        bad_path[
            [
                "time",
                "bad_path_long_first_n",
                "bad_path_short_first_n",
                "bad_path_horizon_bars",
                "bad_path_mae_threshold_bps",
                "bad_path_mfe_threshold_bps",
                # NOTE: the H=10 v11_pnl_*_at_horizon_bps are NOT merged — the direction
                # label uses the dedicated H=24 v11_pnl_*_at_dir_horizon_bps below
                # (2026-05-26). bad_path head still uses its own bad_path_*_first_n.
            ]
        ],
        on="time",
        how="inner",
        validate="one_to_one",
    )
    # 2026-05-26: dedicated H=24 (2h) pnl-at-horizon for the DIRECTION/tradable label
    # (decoupled from bad_path's 10-bar horizon). Reuses the same spread-aware pnl fn.
    _dir_pnl = _compute_bad_path_first_n(
        tape=merged[
            ["time"]
            + [
                c
                for c in merged.columns
                if c in ("bid_close", "ask_close", "bid", "ask")
            ]
        ].copy(),
        horizon_bars=V11_DIRECTION_HORIZON_BARS,
        adverse_threshold_bps=BAD_PATH_MAE_THRESHOLD_BPS,
        favorable_threshold_bps=BAD_PATH_MFE_THRESHOLD_BPS,
    )[["time", "v11_pnl_long_at_horizon_bps", "v11_pnl_short_at_horizon_bps"]].rename(
        columns={
            "v11_pnl_long_at_horizon_bps": "v11_pnl_long_at_dir_horizon_bps",
            "v11_pnl_short_at_horizon_bps": "v11_pnl_short_at_dir_horizon_bps",
        }
    )
    merged2 = merged2.merge(_dir_pnl, on="time", how="inner", validate="one_to_one")
    if len(merged2) == 0:
        raise RuntimeError("LABEL_JOIN_EMPTY")

    # Re-attach ctx to merged2 (align by time)
    df_ctx = pd.DataFrame({"time": df["time"].to_numpy()})
    # Exact shared causal ATR is carried into the inline price layer.  It must
    # not be re-read from an older canonical parquet vintage.
    df_ctx["atr"] = df["atr"].to_numpy(dtype=np.float64)
    for i, name in enumerate(ctx_cont_names):
        df_ctx[name] = df_ctx_cont.iloc[:, i].to_numpy()
    for i, name in enumerate(ctx_cat_names):
        df_ctx[name] = df_ctx_cat.iloc[:, i].to_numpy()

    merged3 = merged2.merge(df_ctx, on="time", how="inner", validate="one_to_one")
    if len(merged3) != len(merged2):
        raise RuntimeError(
            f"CTX_JOIN_ROW_MISMATCH: before={len(merged2)} after={len(merged3)}"
        )

    # 8b) Join the canonical feature frame that owns the genuine per-bar base
    # signals and the complete continuous-context contract.
    canonical_v2_path = Path(canonical_v2_parquet).expanduser().resolve()
    if not canonical_v2_path.exists():
        raise RuntimeError(f"CANONICAL_V2_PARQUET_NOT_FOUND: {canonical_v2_path}")
    canonical_v2_sha256 = _sha256_file(canonical_v2_path)

    log.info(
        "[V2_CANONICAL_JOIN] loading canonical_v2 from %s sha256=%s",
        canonical_v2_path,
        canonical_v2_sha256,
    )
    # The input artifacts have disjoint field ownership. canonical-v2 owns
    # genuine base signals and its V2 context extension. The exact source
    # prebuilt owns V3/cyclic context, full-history regime sources and raw
    # volume. Never select a field from whichever artifact happens to have it.
    from gx1.features.volume_features import VOLUME_FEATURE_NAMES as _VOLUME_FEAT_NAMES

    cv2_owned, source_owned = _model_native_artifact_owner_fields(
        active_base_signal_fields
    )
    computed_overlap = sorted(
        (set(cv2_owned) | set(source_owned)) & set(merged3.columns)
    )
    if computed_overlap:
        raise RuntimeError(
            f"MODEL_NATIVE_INPUT_OWNER_OVERLAP: computed={computed_overlap}"
        )

    source_available = set(df.columns)
    source_missing = [name for name in source_owned if name not in source_available]
    if source_missing:
        raise RuntimeError(
            f"MODEL_NATIVE_SOURCE_OWNED_FIELDS_MISSING: {source_missing}"
        )
    source_extra = df[["time"] + list(source_owned)].copy()
    rows_before_source = len(merged3)
    merged3 = merged3.merge(
        source_extra,
        on="time",
        how="inner",
        validate="one_to_one",
    )
    if len(merged3) != rows_before_source:
        raise RuntimeError(
            "SOURCE_PARQUET_JOIN_ROW_MISMATCH: "
            f"before={rows_before_source} after={len(merged3)}"
        )

    import pyarrow.parquet as _pq_chk

    cv2_available = set(_pq_chk.read_schema(str(canonical_v2_path)).names)
    cv2_missing = [name for name in cv2_owned if name not in cv2_available]
    if cv2_missing:
        raise RuntimeError(
            f"MODEL_NATIVE_CANONICAL_V2_OWNED_FIELDS_MISSING: {cv2_missing}"
        )
    cv2_df = pd.read_parquet(canonical_v2_path, columns=["time"] + list(cv2_owned))
    cv2_df = _normalize_time_utc(cv2_df, "time")
    log.info(
        "[MODEL_NATIVE_INPUT_OWNERS] source=%d canonical_v2=%d cv2_rows=%d",
        len(source_owned),
        len(cv2_owned),
        len(cv2_df),
    )
    rows_pre = len(merged3)
    merged3 = merged3.merge(cv2_df, on="time", how="inner", validate="one_to_one")
    rows_post = len(merged3)
    if rows_post != rows_pre:
        raise RuntimeError(
            f"V2_CANONICAL_JOIN_ROW_MISMATCH: before={rows_pre} after={rows_post}"
        )
    log.info(
        "[V2_CANONICAL_JOIN] merged: rows_pre=%d rows_post=%d lost=%d",
        rows_pre,
        rows_post,
        rows_pre - rows_post,
    )
    if rows_post == 0:
        raise RuntimeError(
            "V2_CANONICAL_JOIN_EMPTY: no time overlap between merged3 and canonical_v2"
        )

    # Recompute long-lookback HTF and REGIME_V4 derived state on this exact
    # common-history frame. Canonical/source copies are never passed through as
    # model-native truth; both calls use the same owners as serving.
    from gx1.execution.v12_ctx_augment_live import _add_htf_features as _add_htf_common
    from gx1.features.regime_v4_features import (
        REGIME_V4_DERIVED_COLS as _REGIME_DERIVED,
        REGIME_V4_SOURCE_COLS as _REGIME_SOURCES,
        add_regime_v4_features as _add_regime_common,
    )

    _common_index = pd.DatetimeIndex(
        pd.to_datetime(merged3["time"], utc=True, errors="raise")
    )
    if (
        _common_index.hasnans
        or not _common_index.is_unique
        or not _common_index.is_monotonic_increasing
    ):
        raise RuntimeError("MODEL_NATIVE_COMMON_HISTORY_TIME_INVALID")
    # The HTF recompute must see the FULL tape history exactly like serving
    # does (live computes HTF state from complete canonical history): sourcing
    # it from the truncated common frame would leave the 252-day D1 percentile
    # NaN across the first ~year of TRAIN and fail the extension head.
    _htf_m5_src = _load_canonical_tape(
        tape_root=tape_root,
        t_min=pd.Timestamp("2020-01-01T00:00:00Z"),
        t_max=pd.Timestamp(_common_index.max()),
        required_cols=["open", "high", "low", "close"],
    )
    _common_m5 = _htf_m5_src.set_index("time")[
        ["open", "high", "low", "close"]
    ].sort_index()
    _htf_common = pd.DataFrame(index=_common_index)
    _add_htf_common(_htf_common, _common_m5)
    _htf_common_cols = (
        "D1_dist_from_ema200_atr",
        "D1_atr_percentile_252",
        "H1_range_compression_ratio",
        "M15_range_compression_ratio",
        "H4_trend_sign_cat",
    )
    _missing_htf_common = [
        name for name in _htf_common_cols if name not in _htf_common.columns
    ]
    if _missing_htf_common:
        raise RuntimeError(
            f"MODEL_NATIVE_COMMON_HISTORY_HTF_MISSING: {_missing_htf_common}"
        )
    for _name in _htf_common_cols:
        merged3[_name] = _htf_common[_name].to_numpy()

    _regime_source_without_d1 = [
        name for name in _REGIME_SOURCES if name != "D1_dist_from_ema200_atr"
    ]
    _missing_regime_sources = [
        name for name in _regime_source_without_d1 if name not in merged3.columns
    ]
    if _missing_regime_sources:
        raise RuntimeError(
            "MODEL_NATIVE_COMMON_HISTORY_REGIME_SOURCES_MISSING: "
            f"{_missing_regime_sources}"
        )
    _regime_common = merged3[_regime_source_without_d1].copy()
    _regime_common.index = _common_index
    _regime_common["D1_dist_from_ema200_atr"] = _htf_common[
        "D1_dist_from_ema200_atr"
    ].to_numpy()
    _add_regime_common(_regime_common)
    for _name in _REGIME_DERIVED:
        merged3[_name] = _regime_common[_name].to_numpy()
    _causal_regime_v4_warmup_rows = int(
        _regime_common.attrs.get("causal_regime_v4_warmup_rows", 0)
    )

    # ── GROUP-A market-parity (24) — 2026-05-26 ──────────────────────────────
    # Compute the 24 ctx_cont parity features (dip-distance, pivots, vol-term,
    # vol-percentile, session-overlap) through the shared
    # augment_forward_outcome_v2.augment_candidate owner. ATR is derived from
    # the same M5 multi-TF cache, so dataset surfaces remain identical.
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
        MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS,
    )

    # Recompute unconditionally over the explicit common-history frame.  Source
    # parquets may carry older/full-range derived values; trusting those would
    # make TRAIN and SERVE history semantics depend on an external build.
    from gx1.scripts.augment_forward_outcome_v2 import (
        attach_group_a_dip_struct_ctx_columns_parallel as _attach_group_a,
        trim_causal_context_warmup_prefix as _trim_context_warmup,
    )
    from gx1.features.htf_features import load_multi_tf_cache as _ga_load_cache

    _cache_dir_raw = os.environ.get("GX1_V10_MULTI_TF_V4_CACHE_DIR", "").strip()
    if not _cache_dir_raw:
        raise RuntimeError(
            "GX1_V10_MULTI_TF_V4_CACHE_DIR_REQUIRED: derived context may not use a default cache"
        )
    _cache_dir = Path(_cache_dir_raw).expanduser().resolve()
    if not _cache_dir.is_dir():
        raise RuntimeError(f"MULTI_TF_V4_CACHE_MISSING: {_cache_dir}")
    _verified_mtf_cache = _ga_load_cache(_cache_dir)
    _multi_tf_cache_binding = {
        "cache_dir": str(_cache_dir),
        "manifest_path": str((_cache_dir / "manifest.json").resolve()),
        "manifest_sha256": str(_verified_mtf_cache.manifest_sha256),
        "cache_identity_sha256": str(
            _verified_mtf_cache.cache_identity_sha256
        ),
        "m5_prebuilt_source": str(
            _verified_mtf_cache.m5_prebuilt_source
        ),
        "m5_prebuilt_source_sha256": str(
            _verified_mtf_cache.m5_prebuilt_source_sha256
        ),
    }
    _group_a_required = list(MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS) + list(
        MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS
    )
    merged3 = merged3.drop(
        columns=[name for name in _group_a_required if name in merged3.columns]
    )
    if output_path is None:
        raise RuntimeError(
            "MODEL_NATIVE_GROUP_A_CHECKPOINT_OUTPUT_REQUIRED: exact split output path is mandatory"
        )
    _group_a_checkpoint_payload = {
        "schema_version": _MODEL_NATIVE_GROUP_A_CHECKPOINT_SCHEMA_VERSION,
        "split_name": split_name,
        "source_parquet_sha256": parquet_sha,
        "canonical_v2_parquet_sha256": canonical_v2_sha256,
        "signal_manifest_sha256": _sha256_file(
            Path(seq_structure_manifest_path).expanduser().resolve()
        ),
        "multi_tf_manifest_sha256": _multi_tf_cache_binding[
            "manifest_sha256"
        ],
        "multi_tf_cache_identity_sha256": _multi_tf_cache_binding[
            "cache_identity_sha256"
        ],
        "rank_reference_sha256": _sha256_file(
            Path(model_native_rank_reference_npz).expanduser().resolve()
        ),
        "feature_history_start_utc": start.isoformat(),
        "feature_computation_end_utc": end.isoformat(),
        "emission_start_utc": emit_start.isoformat(),
        "emission_end_utc": emit_end.isoformat(),
        "output_path": str(Path(output_path).expanduser().resolve()),
    }
    _group_a_checkpoint_key = hashlib.sha256(
        json.dumps(
            _group_a_checkpoint_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _group_a_checkpoint_dir = Path(output_path).expanduser().resolve().parent / (
        f"_{Path(output_path).stem}_GROUP_A_CHECKPOINT"
    )
    merged3 = _attach_group_a(
        merged3,
        multi_tf=_verified_mtf_cache,
        journal_label="model_native_offline",
        workers=_MODEL_NATIVE_GROUP_A_RECOMPUTE_WORKERS,
        checkpoint_dir=_group_a_checkpoint_dir,
        checkpoint_key=_group_a_checkpoint_key,
        context_m5=_common_m5,
    )
    _group_a_checkpoint_meta = {
        **_group_a_checkpoint_payload,
        "checkpoint_key": _group_a_checkpoint_key,
        "complete_path": merged3.attrs.get("group_a_checkpoint_complete_path"),
        "complete_sha256": merged3.attrs.get(
            "group_a_checkpoint_complete_sha256"
        ),
    }
    if (
        not _group_a_checkpoint_meta["complete_path"]
        or not _group_a_checkpoint_meta["complete_sha256"]
    ):
        raise RuntimeError("MODEL_NATIVE_GROUP_A_CHECKPOINT_COMPLETION_MISSING")
    _causal_group_a_warmup_rows = int(
        merged3.attrs.get("causal_context_warmup_rows", 0)
    )
    _causal_required = list(
        dict.fromkeys(_group_a_required + list(_REGIME_SOURCES) + list(_REGIME_DERIVED))
    )
    _rows_before_context_trim = len(merged3)
    merged3 = _trim_context_warmup(merged3, _causal_required).reset_index(drop=True)
    _causal_context_warmup_rows_trimmed = _rows_before_context_trim - len(merged3)
    log.info(
        "[V10_GROUP_A_PARITY] computed %d parity + %d dip/struct features; trimmed warmup rows=%d",
        len(MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS),
        len(MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS),
        _causal_context_warmup_rows_trimmed,
    )

    # ── Volume / order-flow per-bar features (2026-05-26) ────────────────────
    # Derived from raw `volume` (+ `close`) via the SAME helper the live ctx
    # augmenter calls (gx1.features.volume_features) → identical train/serve
    # values. Adds vol_z_20 / vol_ratio_5_20 / vol_pct_96 / signed_vol_z_20 to
    # the genuine model-native per-bar base surface.
    if any(f not in merged3.columns for f in _VOLUME_FEAT_NAMES):
        from gx1.features.volume_features import add_volume_features as _add_vol

        if "volume" not in merged3.columns:
            raise RuntimeError(
                "V10_VOLUME_FEATURES: raw `volume` column missing from merged3 — cannot derive"
            )
        merged3 = merged3.sort_values("time").reset_index(drop=True)
        _add_vol(merged3)
        log.info(
            "[V10_VOLUME_FEATURES] computed %d volume/order-flow seq features (one-truth w/ serving)",
            len(_VOLUME_FEAT_NAMES),
        )

    # Exact 37-target future surface.  Magnitudes use executable bid/ask paths;
    # timing alone uses mid OHLC.  Every incomplete tail value remains NaN and is
    # removed by one union completeness mask immediately before emission.
    _head_target_arrays, _head_target_complete_mask = (
        _build_model_native_aux_head_targets(merged3)
    )
    for _name in MODEL_NATIVE_AUX_TARGET_COLUMNS:
        merged3[_name] = _head_target_arrays[_name]
    log.info(
        "[MODEL_NATIVE_AUX_TARGETS] schema=%s columns=%d max_future_horizon=%d "
        "complete_rows=%d incomplete_tail_rows=%d risk_magnitudes=spread_aware_bid_ask",
        MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION,
        len(MODEL_NATIVE_AUX_TARGET_COLUMNS),
        MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
        int(_head_target_complete_mask.sum()),
        int((~_head_target_complete_mask).sum()),
    )

    # ENTRY smart-context features promoted from audit-only candidates. These are
    # computed after SMC + Group-A + dip/struct source columns exist, before the
    # ctx_cont contract gate below.
    from gx1.features.entry_smart_context import (
        add_entry_smart_context_features as _add_entry_smart,
    )

    _add_entry_smart(merged3)
    log.info(
        "[V10_ENTRY_SMART_CTX] computed %d promoted ctx_cont features",
        len(MODEL_NATIVE_CTX_CONT_ENTRY_SMART_DERIVED_FIELDS),
    )

    # Verify all contracted signal and ctx_cont names are present after join.
    missing_sig = [f for f in active_base_signal_fields if f not in merged3.columns]
    if missing_sig:
        raise RuntimeError(
            f"V2_SIGNAL_FIELDS_MISSING after canonical_v2 join: {missing_sig}"
        )
    missing_ctx = [f for f in MODEL_NATIVE_CTX_CONT_FIELDS if f not in merged3.columns]
    if missing_ctx:
        raise RuntimeError(
            f"V2_CTX_CONT_FIELDS_MISSING after canonical_v2 join: {missing_ctx}"
        )

    # 9) Build the exact model-native seq + snap + context arrays.
    ctx_cont_names = list(MODEL_NATIVE_CTX_CONT_FIELDS)
    if len(ctx_cont_names) != MODEL_NATIVE_CTX_CONT_DIM:
        raise RuntimeError(
            f"MODEL_NATIVE_CTX_CONT_DIM_INVALID: got={len(ctx_cont_names)} expected={MODEL_NATIVE_CTX_CONT_DIM}"
        )
    if len(ctx_cat_names) != MODEL_NATIVE_CTX_CAT_DIM:
        raise RuntimeError(
            f"MODEL_NATIVE_CTX_CAT_DIM_INVALID: got={len(ctx_cat_names)} expected={MODEL_NATIVE_CTX_CAT_DIM}"
        )
    log.info("[V2_CTX_CONT_UPGRADE] ctx_cont_names_v2 count=%d", len(ctx_cont_names))
    seq_structure_requested, seq_structure_meta = _resolve_seq_structure_extension(
        manifest_path=seq_structure_manifest_path,
    )
    seq_structure_feature_names = list(seq_structure_requested)
    signal_fields_emitted = list(active_base_signal_fields) + list(
        seq_structure_feature_names
    )
    snap_fields_emitted = list(signal_fields_emitted)
    expected_model_native_fields = list(
        signal_build_contract["model_native_signal_contract"]["fields"]
    )
    if signal_fields_emitted != expected_model_native_fields:
        raise RuntimeError(
            "MODEL_NATIVE_SIGNAL_FIELD_ORDER_MISMATCH: emitted field order does not "
            "match the exact manifest contract"
        )
    times = merged3["time"].to_numpy()
    aligned_surface = _align_native_m5_feature_surface(
        target_times=times,
        surface_times=m5_feature_surface_times,
        surface_arrays=m5_feature_surface_arrays,
    )
    sig_mat = aligned_surface["signal"]
    ctx_cont_mat = aligned_surface["ctx_cont"]
    ctx_cat_mat = aligned_surface["ctx_cat"]
    seq_structure_meta.update(
        {
            "mode": ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE,
            "feature_surface": dict(m5_feature_surface_binding),
            "features": list(seq_structure_feature_names),
            "feature_count": len(seq_structure_feature_names),
            "inline_split_recomputation": False,
        }
    )
    log.info(
        "[ENTRY_M5_FEATURE_SURFACE] exact rows=%d signals=%d ctx_cont=%d ctx_cat=%d",
        len(merged3),
        int(sig_mat.shape[1]),
        int(ctx_cont_mat.shape[1]),
        int(ctx_cat_mat.shape[1]),
    )
    expected_seq_snap_width = seq_structure_meta.get("expected_seq_snap_width")
    if expected_seq_snap_width is not None:
        expected_seq_snap_width = int(expected_seq_snap_width)
        observed_seq_snap_width = int(sig_mat.shape[1])
        if observed_seq_snap_width != expected_seq_snap_width:
            raise RuntimeError(
                "SEQ_STRUCTURE_EXPECTED_WIDTH_MISMATCH: "
                f"manifest_variant={seq_structure_meta.get('manifest_variant')} "
                f"expected={expected_seq_snap_width} observed={observed_seq_snap_width} "
                f"manifest={seq_structure_meta.get('manifest_path')}"
            )
    if not np.isfinite(sig_mat).all():
        raise RuntimeError("MODEL_NATIVE_SIGNAL_NONFINITE")
    if not np.isfinite(ctx_cont_mat).all():
        raise RuntimeError("MODEL_NATIVE_CTX_CONT_NONFINITE")
    if not np.isfinite(ctx_cat_mat.astype(np.float64)).all():
        raise RuntimeError("MODEL_NATIVE_CTX_CAT_NONFINITE")

    target_rows = len(merged3)
    y_dir = np.full(target_rows, MODEL_DIRECTION_FLAT_INDEX, dtype=np.int32)
    y_early = np.zeros(target_rows, dtype=np.float32)
    y_qual = np.zeros(target_rows, dtype=np.float32)
    y_mae_first_n = np.zeros(target_rows, dtype=np.float32)
    y_mfe_first_n = np.zeros(target_rows, dtype=np.float32)
    y_path_quality = np.zeros(target_rows, dtype=np.float32)
    y_bad_path = np.zeros(target_rows, dtype=np.float32)
    y_label_horizon = final_direction_label_horizon_array(len(merged3))
    y_path_horizon = merged3["path_quality_horizon_bars"].astype(np.int32).to_numpy()

    # V10 v3+ TARGET 1: multi-TF trend-agreement score.
    # Computed from D1/H4/H1/M15/M5 sign signals; fraction of non-D1 TFs
    # whose sign matches D1's. Aux label for training (loss-weighting when
    # direction prediction is wrong under high TF-disagreement).
    # Spec: GX1_DATA/V10_V3_RETRAIN_TARGETS.md target 1.
    from gx1.features.tf_agreement_score import compute_tf_agreement_score

    y_tf_agreement = compute_tf_agreement_score(merged3).astype(np.float32).to_numpy()
    if len(y_tf_agreement) != len(merged3) or not np.isfinite(y_tf_agreement).all():
        raise RuntimeError("V3_TF_AGREEMENT_INVALID: target is missing or non-finite")
    if np.unique(y_tf_agreement).size < 2:
        raise RuntimeError("V3_TF_AGREEMENT_DEAD: target is constant")
    log.info(
        "[V3_TF_AGREEMENT] computed n=%d  mean=%.3f  std=%.3f  frac_full=%.3f  frac_zero=%.3f",
        len(y_tf_agreement),
        float(y_tf_agreement.mean()),
        float(y_tf_agreement.std()),
        float((y_tf_agreement == 1.0).mean()),
        float((y_tf_agreement == 0.0).mean()),
    )

    # Position-size depends on selected future MFE/MAE and tradable side, so the
    # real target is materialized after outcome-side selection below.
    y_position_size = np.full(len(merged3), 0.5, dtype=np.float32)

    # ---------------------------------------------------------------------------
    # Quality/tradability targets:
    # - We now intentionally collapse weak/ambiguous directional labels into FLAT.
    # - Main direction label follows the strict tradable side, not raw directional truth.
    # - This makes the dataset teach "only obvious edge" instead of "direction exists
    #   but maybe don't take it", which was too permissive for the current goal.
    # ---------------------------------------------------------------------------
    _mfe_long = merged3["mfe_long_first_n_bps"].astype(np.float32).to_numpy()
    _mae_long = merged3["mae_long_first_n_bps"].astype(np.float32).to_numpy()
    _mfe_short = merged3["mfe_short_first_n_bps"].astype(np.float32).to_numpy()
    _mae_short = merged3["mae_short_first_n_bps"].astype(np.float32).to_numpy()
    _bad_path_long = (
        merged3["bad_path_long_first_n"].astype(np.float32).to_numpy() > 0.5
    )
    # V2: also extract bad_path_short for symmetric BIDIR auxiliary labels.
    _bad_path_short = (
        merged3["bad_path_short_first_n"].astype(np.float32).to_numpy() > 0.5
    )
    _path_long = (_mfe_long - _mae_long).astype(np.float32)
    _path_short = (_mfe_short - _mae_short).astype(np.float32)
    _path_lead_long = (_path_long - _path_short).astype(np.float32)
    _path_lead_short = (_path_short - _path_long).astype(np.float32)
    _mfe_lead_long = (_mfe_long - _mfe_short).astype(np.float32)
    _mfe_lead_short = (_mfe_short - _mfe_long).astype(np.float32)

    # V11 redesign: tradable target is OUTCOME-based (final PnL ≥ threshold).
    # Old (V10): tradable required 6 trajectory conditions (mfe/mae/path/lead/...).
    #            Resulted in head_tradable being saturated (median pred ≈ 0.91)
    #            despite only ~5% of dataset being tradable. Anti-calibrated head.
    # New (V11): tradable = (final_pnl_at_horizon >= V11_TRADABLE_PNL_MIN_BPS).
    #            Direct outcome target, head learns to predict P(profitable trade).
    # 2026-05-26: threshold 30→15 bps + dedicated H=24 (2h) direction horizon
    # (V11_TRADABLE_PNL_MIN_BPS / V11_DIRECTION_HORIZON_BARS module consts) → ~60%
    # flat (was ~89%). Uses the v11_pnl_*_at_DIR_horizon_bps columns (H=24), NOT the
    # bad_path H=10 columns. (rule: 89% flat uaktuelt — give the model real signal.)
    _dir_long_col = "v11_pnl_long_at_dir_horizon_bps"
    _dir_short_col = "v11_pnl_short_at_dir_horizon_bps"
    if _dir_long_col not in merged3.columns or _dir_short_col not in merged3.columns:
        raise RuntimeError(
            f"V11_DIRECTION_PNL_MISSING: need {_dir_long_col}/{_dir_short_col} (H={V11_DIRECTION_HORIZON_BARS})"
        )
    _pnl_long_at_h = merged3[_dir_long_col].astype(np.float32).to_numpy()
    _pnl_short_at_h = merged3[_dir_short_col].astype(np.float32).to_numpy()

    # Learned target engineering, not a live rule: labels are chosen by one
    # immutable future side-utility formula.  There is no legacy target mode.
    _side_score_long = (
        _pnl_long_at_h
        + (float(V12_DIRECTION_UTILITY_MFE_WEIGHT) * _mfe_long)
        - (float(V12_DIRECTION_UTILITY_MAE_WEIGHT) * _mae_long)
        + (float(V12_DIRECTION_UTILITY_PATH_WEIGHT) * _path_long)
    ).astype(np.float32)
    _side_score_short = (
        _pnl_short_at_h
        + (float(V12_DIRECTION_UTILITY_MFE_WEIGHT) * _mfe_short)
        - (float(V12_DIRECTION_UTILITY_MAE_WEIGHT) * _mae_short)
        + (float(V12_DIRECTION_UTILITY_PATH_WEIGHT) * _path_short)
    ).astype(np.float32)
    if (
        not np.isfinite(_side_score_long).all()
        or not np.isfinite(_side_score_short).all()
    ):
        raise RuntimeError(
            "V12_DIRECTION_UTILITY_NONFINITE: future outcome components must be finite; "
            "no sentinel/fallback replacement is allowed"
        )
    _side_margin = (_side_score_long - _side_score_short).astype(np.float32)
    _tradable_long = (_side_score_long >= float(V12_DIRECTION_UTILITY_MIN_BPS)) & (
        _side_margin >= float(V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS)
    )
    _tradable_short = (_side_score_short >= float(V12_DIRECTION_UTILITY_MIN_BPS)) & (
        (-_side_margin) >= float(V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS)
    )
    if (
        not np.isfinite(_side_score_long).all()
        or not np.isfinite(_side_score_short).all()
    ):
        raise RuntimeError(
            "V12_DIRECTION_SCORE_NONFINITE: future side outcomes must be finite; "
            "no direction-label fallback is allowed"
        )
    # Side selection follows the better path-aware future utility target.
    _side = np.full(len(merged3), -1, dtype=np.int8)
    _only_long = _tradable_long & ~_tradable_short
    _only_short = _tradable_short & ~_tradable_long
    _both = _tradable_long & _tradable_short
    _side[_only_long] = MODEL_DIRECTION_LONG_INDEX
    _side[_only_short] = MODEL_DIRECTION_SHORT_INDEX
    if _both.any():
        _prefer_long = _side_score_long >= _side_score_short
        _prefer_short = _side_score_short > _side_score_long
        _side[_both & _prefer_long] = MODEL_DIRECTION_LONG_INDEX
        _side[_both & _prefer_short] = MODEL_DIRECTION_SHORT_INDEX

    _direction_side = _side.copy()

    # Mine hard examples from every row using realized future path evidence.
    # No external model score preselects which mistakes the Entry model sees.
    _hard_negative_candidate_source = "all_rows_future_path"
    _long_path_candidate = np.ones(len(merged3), dtype=bool)
    _short_path_candidate = np.ones(len(merged3), dtype=bool)
    _dead_negative_long = (
        _long_path_candidate
        & (_side != MODEL_DIRECTION_LONG_INDEX)
        & (_mfe_long <= float(DEAD_LONG_MAX_MFE_BPS))
        & (_mae_long >= float(DEAD_LONG_MIN_MAE_BPS))
    )
    _teaser_negative_long = (
        _long_path_candidate
        & (_side != MODEL_DIRECTION_LONG_INDEX)
        & (_mfe_long > float(TEASER_LONG_MIN_MFE_BPS))
        & (_mfe_long <= float(TEASER_LONG_MAX_MFE_BPS))
        & (
            _bad_path_long
            | (_mae_long >= float(TEASER_LONG_MIN_MAE_BPS))
            | (_path_long <= float(TEASER_LONG_MAX_PATH_BPS))
        )
    )
    _hard_negative_long = (
        _long_path_candidate
        & (_side != MODEL_DIRECTION_LONG_INDEX)
        & ~_dead_negative_long
        & ~_teaser_negative_long
        & (
            _bad_path_long
            | (
                (_mfe_long >= float(HARD_NEG_LONG_MIN_MFE_BPS))
                & (
                    (_mae_long >= float(HARD_NEG_LONG_MIN_MAE_BPS))
                    | (_path_long <= float(HARD_NEG_LONG_MAX_PATH_BPS))
                )
            )
        )
    )
    y_dead_negative_long = _dead_negative_long.astype(np.float32)
    y_teaser_negative_long = _teaser_negative_long.astype(np.float32)
    y_hard_negative_long = _hard_negative_long.astype(np.float32)
    _clean_edge_intrinsic = (
        (_direction_side == MODEL_DIRECTION_LONG_INDEX)
        & (_mfe_long >= float(CLEAN_EDGE_LONG_MFE_MIN_BPS))
        & (_mae_long <= float(CLEAN_EDGE_LONG_MAE_MAX_BPS))
        & (_path_long >= float(CLEAN_EDGE_LONG_PATH_MIN_BPS))
        & (~_bad_path_long)
    )
    y_clean_edge_long = _clean_edge_intrinsic.astype(np.float32)
    _survival_intrinsic = (
        (_direction_side == MODEL_DIRECTION_LONG_INDEX)
        & (_mfe_long >= float(SURVIVAL_LONG_MFE_MIN_BPS))
        & (_mae_long <= float(SURVIVAL_LONG_MAE_MAX_BPS))
        & (_path_long >= float(SURVIVAL_LONG_PATH_MIN_BPS))
    )
    y_survival_long = _survival_intrinsic.astype(np.float32)
    y_selector_long_mask = (
        _long_path_candidate
        | (_direction_side == MODEL_DIRECTION_LONG_INDEX)
    ).astype(
        np.float32
    )

    # ------------------------------------------------------------------------
    # V2: SHORT-side auxiliary labels (mirror of LONG-side for BIDIR symmetry)
    # ------------------------------------------------------------------------
    # Same all-row path-candidate logic as long-side, applied to short outcomes.
    # These feed V10's auxiliary heads (head_bad_path / head_clean_edge / head_survival)
    # symmetrically — without these, short-side rows would only have direction-supervision.
    _dead_negative_short = (
        _short_path_candidate
        & (_side != MODEL_DIRECTION_SHORT_INDEX)
        & (_mfe_short <= float(DEAD_LONG_MAX_MFE_BPS))
        & (_mae_short >= float(DEAD_LONG_MIN_MAE_BPS))
    )
    _teaser_negative_short = (
        _short_path_candidate
        & (_side != MODEL_DIRECTION_SHORT_INDEX)
        & (_mfe_short > float(TEASER_LONG_MIN_MFE_BPS))
        & (_mfe_short <= float(TEASER_LONG_MAX_MFE_BPS))
        & (
            _bad_path_short
            | (_mae_short >= float(TEASER_LONG_MIN_MAE_BPS))
            | (_path_short <= float(TEASER_LONG_MAX_PATH_BPS))
        )
    )
    _hard_negative_short = (
        _short_path_candidate
        & (_side != MODEL_DIRECTION_SHORT_INDEX)
        & ~_dead_negative_short
        & ~_teaser_negative_short
        & (
            _bad_path_short
            | (
                (_mfe_short >= float(HARD_NEG_LONG_MIN_MFE_BPS))
                & (
                    (_mae_short >= float(HARD_NEG_LONG_MIN_MAE_BPS))
                    | (_path_short <= float(HARD_NEG_LONG_MAX_PATH_BPS))
                )
            )
        )
    )
    y_dead_negative_short = _dead_negative_short.astype(np.float32)
    y_teaser_negative_short = _teaser_negative_short.astype(np.float32)
    y_hard_negative_short = _hard_negative_short.astype(np.float32)
    _clean_edge_short_intrinsic = (
        (_direction_side == MODEL_DIRECTION_SHORT_INDEX)
        & (_mfe_short >= float(CLEAN_EDGE_LONG_MFE_MIN_BPS))
        & (_mae_short <= float(CLEAN_EDGE_LONG_MAE_MAX_BPS))
        & (_path_short >= float(CLEAN_EDGE_LONG_PATH_MIN_BPS))
        & (~_bad_path_short)
    )
    y_clean_edge_short = _clean_edge_short_intrinsic.astype(np.float32)
    _survival_short_intrinsic = (
        (_direction_side == MODEL_DIRECTION_SHORT_INDEX)
        & (_mfe_short >= float(SURVIVAL_LONG_MFE_MIN_BPS))
        & (_mae_short <= float(SURVIVAL_LONG_MAE_MAX_BPS))
        & (_path_short >= float(SURVIVAL_LONG_PATH_MIN_BPS))
    )
    y_survival_short = _survival_short_intrinsic.astype(np.float32)
    y_selector_short_mask = (
        _short_path_candidate
        | (_direction_side == MODEL_DIRECTION_SHORT_INDEX)
    ).astype(np.float32)

    # ------------------------------------------------------------------------
    # V2: BIDIR-COMBINED labels for V10's single auxiliary heads
    # ------------------------------------------------------------------------
    # V10 has ONE head_clean_edge, ONE head_survival, ONE head_bad_path (not split per side).
    # Bidir supervision = OR across long-side and short-side label.
    y_clean_edge_bidir = np.maximum(y_clean_edge_long, y_clean_edge_short).astype(
        np.float32
    )
    y_survival_bidir = np.maximum(y_survival_long, y_survival_short).astype(np.float32)
    # y_bad_path is already direction-aware via merged3["y_bad_path"]:
    # it selects the canonical LONG/SHORT class and parks FLAT at zero.

    # Core direction/trade/side targets come only from future side utility and
    # must never be rewritten by structural context or stale teacher artifacts.
    y_tradable = (_direction_side != -1).astype(np.int32)
    y_dir = np.full(
        len(merged3),
        MODEL_DIRECTION_FLAT_INDEX,
        dtype=np.int32,
    )
    y_dir[_direction_side == MODEL_DIRECTION_LONG_INDEX] = (
        MODEL_DIRECTION_LONG_INDEX
    )
    y_dir[_direction_side == MODEL_DIRECTION_SHORT_INDEX] = (
        MODEL_DIRECTION_SHORT_INDEX
    )

    # Quality auxiliaries align to the strict tradable side. Non-obvious labels are
    # intentionally parked to zero/FLAT.
    _quality_side = _direction_side.copy()

    y_mfe_first_n = np.zeros_like(y_mfe_first_n)
    y_mae_first_n = np.zeros_like(y_mae_first_n)
    y_path_quality = np.zeros_like(y_path_quality)
    long_quality = _quality_side == MODEL_DIRECTION_LONG_INDEX
    short_quality = _quality_side == MODEL_DIRECTION_SHORT_INDEX
    y_mfe_first_n[long_quality] = _mfe_long[long_quality]
    y_mfe_first_n[short_quality] = _mfe_short[short_quality]
    y_mae_first_n[long_quality] = _mae_long[long_quality]
    y_mae_first_n[short_quality] = _mae_short[short_quality]
    y_path_quality[long_quality] = _path_long[long_quality]
    y_path_quality[short_quality] = _path_short[short_quality]

    # Early move: align to the directional-quality side instead of tradability side.
    y_early = np.zeros_like(y_early)
    y_early[_quality_side != -1] = (
        y_mfe_first_n[_quality_side != -1] >= float(early_move_threshold_bps)
    ).astype(np.float32)

    # Quality score stays non-negative but now reflects directional path quality.
    y_qual = np.zeros_like(y_qual)
    y_qual[_quality_side != -1] = np.maximum(
        0.0, y_path_quality[_quality_side != -1]
    ).astype(np.float32)

    def _sig_col(names: Sequence[str]) -> np.ndarray:
        for name in names:
            if name in signal_fields_emitted:
                idx = int(signal_fields_emitted.index(name))
                return sig_mat[:, idx].astype(np.float32, copy=False)
        raise RuntimeError(
            "XAU_STRUCTURAL_AUX_LABEL_SIGNAL_MISSING: expected one of "
            + repr(list(names))
        )

    def _structural_signal(requirement: str) -> np.ndarray:
        try:
            candidates = STRUCTURAL_AUX_LABEL_SIGNAL_REQUIREMENTS[requirement]
        except KeyError as exc:
            raise RuntimeError(
                "XAU_STRUCTURAL_AUX_LABEL_REQUIREMENT_UNKNOWN: "
                f"{requirement}"
            ) from exc
        return _sig_col(candidates)

    _trend_score = _structural_signal("trend_score")
    _trend_conflict = _structural_signal("trend_conflict")
    _long_trend_bias = _structural_signal("long_trend_bias")
    _short_trend_bias = _structural_signal("short_trend_bias")
    _structure_dir = _structural_signal("structure_direction")
    _support_prox = np.maximum.reduce(
        [
            _structural_signal("geometry_support_line_proximity"),
            _structural_signal("support_level_proximity"),
            _structural_signal("support_respect"),
            _structural_signal("support_reclaim"),
        ]
    )
    _resistance_prox = np.maximum.reduce(
        [
            _structural_signal("geometry_resistance_line_proximity"),
            _structural_signal("resistance_level_proximity"),
            _structural_signal("resistance_respect"),
            _structural_signal("resistance_reclaim"),
        ]
    )
    _channel_edge = _structural_signal("geometry_channel_edge")
    _channel_pos = _structural_signal("geometry_channel_position")
    _support_respect = np.maximum(
        _structural_signal("support_respect"),
        _structural_signal("support_liquidity_rejection"),
    )
    _resistance_respect = np.maximum(
        _structural_signal("resistance_respect"),
        _structural_signal("resistance_liquidity_rejection"),
    )
    _geom_long_prox = _structural_signal("geometry_long_fib_sr_proximity")
    _geom_short_prox = _structural_signal("geometry_short_fib_sr_proximity")

    _intraday_up = (
        (_trend_score >= 0.0)
        & (_long_trend_bias >= _short_trend_bias)
        & (_structure_dir >= -0.10)
    )
    _intraday_down = (
        (_trend_score <= 0.0)
        & (_short_trend_bias >= _long_trend_bias)
        & (_structure_dir <= 0.10)
    )
    _rising_support = (
        _intraday_up
        & (_support_prox >= 0.35)
        & (_support_prox >= _resistance_prox)
        & (
            (_channel_edge >= 0.15)
            | (_channel_pos <= 0.42)
            | (_support_respect >= 0.35)
            | (_geom_long_prox >= 0.35)
        )
    )
    _falling_resistance = (
        _intraday_down
        & (_resistance_prox >= 0.35)
        & (_resistance_prox >= _support_prox)
        & (
            (_channel_edge >= 0.15)
            | (_channel_pos >= 0.58)
            | (_resistance_respect >= 0.35)
            | (_geom_short_prox >= 0.35)
        )
    )

    _side_margin_bps = np.maximum(1.0, float(V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS))
    _long_high_mae_low_mfe = (_mae_long >= float(HARD_NEG_LONG_MIN_MAE_BPS)) & (
        (_mfe_long <= float(TEASER_LONG_MAX_MFE_BPS))
        | (_path_long <= float(HARD_NEG_LONG_MAX_PATH_BPS))
    )
    _short_high_mae_low_mfe = (_mae_short >= float(HARD_NEG_LONG_MIN_MAE_BPS)) & (
        (_mfe_short <= float(TEASER_LONG_MAX_MFE_BPS))
        | (_path_short <= float(HARD_NEG_LONG_MAX_PATH_BPS))
    )
    _support_retest_continuation = (
        _rising_support
        & (_side_score_long >= float(V12_DIRECTION_UTILITY_MIN_BPS))
        & ((_side_score_long - _side_score_short) >= _side_margin_bps)
        & (~_bad_path_long)
    )
    _resistance_retest_continuation = (
        _falling_resistance
        & (_side_score_short >= float(V12_DIRECTION_UTILITY_MIN_BPS))
        & ((_side_score_short - _side_score_long) >= _side_margin_bps)
        & (~_bad_path_short)
    )
    _countertrend_short_trap = _rising_support & (
        _bad_path_short
        | _short_high_mae_low_mfe
        | ((_side_score_long - _side_score_short) >= _side_margin_bps)
        | (_direction_side == MODEL_DIRECTION_LONG_INDEX)
    )
    _countertrend_long_trap = _falling_resistance & (
        _bad_path_long
        | _long_high_mae_low_mfe
        | ((_side_score_short - _side_score_long) >= _side_margin_bps)
        | (_direction_side == MODEL_DIRECTION_SHORT_INDEX)
    )
    _mtf_conflict_m5_vs_higher = (
        (_trend_conflict >= 0.45)
        | ((_trend_score > 0.15) & (_short_trend_bias > _long_trend_bias))
        | ((_trend_score < -0.15) & (_long_trend_bias > _short_trend_bias))
    )

    # Structure/trend labels below are representation/slice supervision only.
    # Core labels and side-specific future outcomes are copied without any
    # feature-derived rewrite or forced ordering.
    y_trade = y_tradable.astype(np.float32)
    y_side = np.where(
        y_dir == MODEL_DIRECTION_SHORT_INDEX,
        MODEL_DIRECTION_SHORT_INDEX,
        MODEL_DIRECTION_LONG_INDEX,
    ).astype(np.int8)
    y_side_mask = y_trade.astype(np.float32)
    y_long_path_utility_bps = _side_score_long.astype(np.float32)
    y_short_path_utility_bps = _side_score_short.astype(np.float32)
    y_long_bad_path = _bad_path_long.astype(np.float32)
    y_short_bad_path = _bad_path_short.astype(np.float32)
    y_long_expected_mae_bps = _mae_long.astype(np.float32)
    y_short_expected_mae_bps = _mae_short.astype(np.float32)
    y_clean_edge_bidir = np.maximum(y_clean_edge_long, y_clean_edge_short).astype(
        np.float32
    )
    y_survival_bidir = np.maximum(y_survival_long, y_survival_short).astype(np.float32)
    y_bad_path = _selected_side_bad_path_target(
        _quality_side,
        y_long_bad_path,
        y_short_bad_path,
    )
    if "atr_bps" not in merged3.columns:
        raise RuntimeError("V3_POSITION_SIZE_INPUT_MISSING: atr_bps")
    y_position_size = _position_size_target_from_path(
        y_mfe_first_n,
        y_mae_first_n,
        merged3["atr_bps"].astype(np.float32).to_numpy(),
        y_trade,
    )
    if len(y_position_size) != len(merged3) or not np.isfinite(y_position_size).all():
        raise RuntimeError("V3_POSITION_SIZE_INVALID: target is missing or non-finite")
    if np.unique(y_position_size).size < 2:
        raise RuntimeError("V3_POSITION_SIZE_DEAD: target is constant")
    log.info(
        "[V3_POSITION_SIZE] source=selected_future_path n=%d mean=%.3f p10=%.3f p50=%.3f p90=%.3f",
        len(y_position_size),
        float(y_position_size.mean()),
        float(np.percentile(y_position_size, 10)),
        float(np.percentile(y_position_size, 50)),
        float(np.percentile(y_position_size, 90)),
    )
    y_rising_channel_support_touch = _rising_support.astype(np.float32)
    y_falling_channel_resistance_touch = _falling_resistance.astype(np.float32)
    y_support_retest_continuation = _support_retest_continuation.astype(np.float32)
    y_resistance_retest_continuation = _resistance_retest_continuation.astype(
        np.float32
    )
    y_countertrend_short_trap = _countertrend_short_trap.astype(np.float32)
    y_countertrend_long_trap = _countertrend_long_trap.astype(np.float32)
    y_mtf_conflict_m5_vs_higher_side = _mtf_conflict_m5_vs_higher.astype(np.float32)
    y_long_high_mae_low_mfe_early_failure = _long_high_mae_low_mfe.astype(np.float32)
    y_short_high_mae_low_mfe_early_failure = _short_high_mae_low_mfe.astype(np.float32)

    log.info(
        "[ENTRY_HIER_LABEL_PROOF] trade=%.4f side_mask=%.4f rising_support=%.4f falling_resistance=%.4f "
        "countertrend_short_trap=%.4f countertrend_long_trap=%.4f mtf_conflict=%.4f",
        float(np.mean(y_trade)),
        float(np.mean(y_side_mask)),
        float(np.mean(y_rising_channel_support_touch)),
        float(np.mean(y_falling_channel_resistance_touch)),
        float(np.mean(y_countertrend_short_trap)),
        float(np.mean(y_countertrend_long_trap)),
        float(np.mean(y_mtf_conflict_m5_vs_higher_side)),
    )
    _split_tag = split_name or "full"
    _directional_long_rate = (
        float(np.mean(y_dir == MODEL_DIRECTION_LONG_INDEX))
        if len(y_dir)
        else 0.0
    )
    _directional_short_rate = (
        float(np.mean(y_dir == MODEL_DIRECTION_SHORT_INDEX))
        if len(y_dir)
        else 0.0
    )
    _directional_flat_rate = (
        float(np.mean(y_dir == MODEL_DIRECTION_FLAT_INDEX))
        if len(y_dir)
        else 0.0
    )
    log.info(
        "[ENTRY_DIRECTION_TARGET_SEMANTICS] split=%s source=future_path_utility_only "
        "target_mode=%s long_rate=%.6f short_rate=%.6f flat_rate=%.6f",
        _split_tag,
        V12_DIRECTION_TARGET_MODE,
        _directional_long_rate,
        _directional_short_rate,
        _directional_flat_rate,
    )
    log.info(
        "[ENTRY_DEAD_LONG_RULES] split=%s mfe_max=%.2f mae_min=%.2f rate=%.6f",
        _split_tag,
        float(DEAD_LONG_MAX_MFE_BPS),
        float(DEAD_LONG_MIN_MAE_BPS),
        float(np.mean(y_dead_negative_long)) if len(y_dead_negative_long) else 0.0,
    )
    log.info(
        "[ENTRY_TEASER_LONG_RULES] split=%s mfe_min=%.2f mfe_max=%.2f mae_min=%.2f path_max=%.2f rate=%.6f",
        _split_tag,
        float(TEASER_LONG_MIN_MFE_BPS),
        float(TEASER_LONG_MAX_MFE_BPS),
        float(TEASER_LONG_MIN_MAE_BPS),
        float(TEASER_LONG_MAX_PATH_BPS),
        float(np.mean(y_teaser_negative_long)) if len(y_teaser_negative_long) else 0.0,
    )
    log.info(
        "[ENTRY_HARD_NEG_LONG_RULES] split=%s candidate_source=%s "
        "mfe_min=%.2f mae_min=%.2f path_max=%.2f rate=%.6f",
        _split_tag,
        _hard_negative_candidate_source,
        float(HARD_NEG_LONG_MIN_MFE_BPS),
        float(HARD_NEG_LONG_MIN_MAE_BPS),
        float(HARD_NEG_LONG_MAX_PATH_BPS),
        float(np.mean(y_hard_negative_long)) if len(y_hard_negative_long) else 0.0,
    )
    log.info(
        "[ENTRY_CLEAN_EDGE_LONG_RULES] split=%s mfe_min=%.2f mae_max=%.2f path_min=%.2f rate=%.6f",
        _split_tag,
        float(CLEAN_EDGE_LONG_MFE_MIN_BPS),
        float(CLEAN_EDGE_LONG_MAE_MAX_BPS),
        float(CLEAN_EDGE_LONG_PATH_MIN_BPS),
        float(np.mean(y_clean_edge_long)) if len(y_clean_edge_long) else 0.0,
    )
    log.info(
        "[ENTRY_SURVIVAL_LONG_RULES] split=%s mfe_min=%.2f mae_max=%.2f path_min=%.2f rate=%.6f",
        _split_tag,
        float(SURVIVAL_LONG_MFE_MIN_BPS),
        float(SURVIVAL_LONG_MAE_MAX_BPS),
        float(SURVIVAL_LONG_PATH_MIN_BPS),
        float(np.mean(y_survival_long)) if len(y_survival_long) else 0.0,
    )
    # Tradable rate proof (split-aware)
    _tradable_rate = float(np.mean(y_tradable)) if len(y_tradable) else 0.0
    log.info(
        "[ENTRY_TRADABLE_RATE_PROOF] split=%s n_rows=%d tradable_rate=%.6f",
        _split_tag,
        len(y_tradable),
        _tradable_rate,
    )
    _quality_side_rate = (
        float(np.mean(_quality_side != -1)) if len(_quality_side) else 0.0
    )
    log.info(
        "[ENTRY_QUALITY_SIDE_RATE_PROOF] split=%s n_rows=%d quality_side_rate=%.6f",
        _split_tag,
        len(_quality_side),
        _quality_side_rate,
    )
    _side_long = y_dir == MODEL_DIRECTION_LONG_INDEX
    _side_short = y_dir == MODEL_DIRECTION_SHORT_INDEX
    _long_rate = float(np.mean(y_tradable[_side_long])) if _side_long.any() else 0.0
    _short_rate = float(np.mean(y_tradable[_side_short])) if _side_short.any() else 0.0
    log.info(
        "[ENTRY_TRADABLE_RATE_BY_SIDE] split=%s long_rate=%.6f short_rate=%.6f",
        _split_tag,
        _long_rate,
        _short_rate,
    )
    if "session_id" in merged3.columns:
        _sess = merged3["session_id"].astype(np.int64).to_numpy()
        for sid in sorted(set(_sess.tolist())):
            _mask = _sess == sid
            _rate = float(np.mean(y_tradable[_mask])) if _mask.any() else 0.0
            log.info(
                "[ENTRY_TRADABLE_RATE_BY_SESSION] split=%s session_id=%d session_name=%s rate=%.6f",
                _split_tag,
                int(sid),
                SESSION_NAME_BY_ID.get(int(sid), "UNK"),
                _rate,
            )

    n = len(merged3)
    if n < (seq_len + 1):
        raise RuntimeError(f"TOO_FEW_ROWS_FOR_SEQ: rows={n} seq_len={seq_len}")

    # V2 STREAMING-WRITE: Build rows in batches, write each batch to parquet via
    # pyarrow.parquet.ParquetWriter, and free memory between batches. The previous
    # accumulate-then-DataFrame pattern OOM'd at ~14 GB on full 311k rows because
    # 96-bar × 37-feature seq + ctx + labels per row × 311k rows ~= 25 GB DataFrame
    # in the 16 GB sandbox.
    import pyarrow as _pa
    import pyarrow.parquet as _pq

    def _to_list(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [_to_list(v) for v in x]
        return x

    emitted_time_index = pd.DatetimeIndex(
        pd.to_datetime(times, utc=True, errors="raise")
    )
    pre_emit_history_rows = int(
        emitted_time_index.searchsorted(emit_start, side="left")
    )
    if pre_emit_history_rows < (seq_len - 1):
        raise RuntimeError(
            "MODEL_NATIVE_COMMON_HISTORY_WARMUP_INSUFFICIENT: "
            f"clean_rows_before_emit={pre_emit_history_rows} required={seq_len - 1} "
            f"history_start={start} emit_start={emit_start}"
        )
    emit_mask = np.asarray(
        (emitted_time_index >= emit_start) & (emitted_time_index <= emit_end),
        dtype=bool,
    )
    emit_mask[: seq_len - 1] = False
    emitted_candidate_count_before_aux_completeness = int(emit_mask.sum())
    aux_target_incomplete_candidate_rows_excluded = int(
        np.count_nonzero(emit_mask & ~_head_target_complete_mask)
    )
    emit_mask &= _head_target_complete_mask
    emitted_candidate_count = int(emit_mask.sum())
    if emitted_candidate_count <= 0:
        raise RuntimeError(
            "MODEL_NATIVE_EMIT_WINDOW_EMPTY_AFTER_HISTORY_AND_TARGET_COMPLETENESS: "
            f"emit={emit_start}..{emit_end} history={start}..{end} "
            f"before_aux_completeness={emitted_candidate_count_before_aux_completeness}"
        )
    for target_name in MODEL_NATIVE_AUX_TARGET_COLUMNS:
        if not np.isfinite(_head_target_arrays[target_name][emit_mask]).all():
            raise RuntimeError(
                f"MODEL_NATIVE_AUX_TARGET_NONFINITE_EMISSION_FORBIDDEN: {target_name}"
            )
    log.info(
        "[MODEL_NATIVE_AUX_TARGET_EMIT_PROOF] candidates_before=%d excluded_incomplete=%d emitted=%d",
        emitted_candidate_count_before_aux_completeness,
        aux_target_incomplete_candidate_rows_excluded,
        emitted_candidate_count,
    )

    streaming_active = output_path is not None
    if streaming_active:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "[V2_STREAMING_WRITE] enabled: output=%s batch_size=%d total_emitted_rows~=%d",
            str(output_path),
            int(streaming_batch_size),
            emitted_candidate_count,
        )

    pq_writer = None
    pq_schema = None
    summary_rows: List[
        Dict[str, Any]
    ] = []  # only identity + labels (no seq/snap/ctx) — small
    total_written = 0

    def _emit_batch(rows_batch: List[Dict[str, Any]]) -> None:
        nonlocal pq_writer, pq_schema, total_written
        if not rows_batch:
            return
        df_b = pd.DataFrame(rows_batch)
        df_b["seq"] = df_b["seq"].apply(_to_list)
        df_b["snap"] = df_b["snap"].apply(_to_list)
        df_b["ctx_cont"] = df_b["ctx_cont"].apply(_to_list)
        df_b["ctx_cat"] = df_b["ctx_cat"].apply(_to_list)
        if streaming_active:
            if pq_schema is None:
                table = _pa.Table.from_pandas(df_b, preserve_index=False)
                pq_schema = table.schema
                pq_writer = _pq.ParquetWriter(
                    str(output_path), pq_schema, compression="snappy"
                )
                pq_writer.write_table(table)
            else:
                table = _pa.Table.from_pandas(
                    df_b, schema=pq_schema, preserve_index=False, safe=False
                )
                pq_writer.write_table(table)
        # Always keep tiny summary row (identity + labels) for downstream logging
        summary_rows.extend(
            df_b[
                [
                    c
                    for c in df_b.columns
                    if c not in ("seq", "snap", "ctx_cont", "ctx_cat")
                ]
            ].to_dict(orient="records")
        )
        total_written += len(df_b)
        del df_b, rows_batch[:]

    pending: List[Dict[str, Any]] = []
    # Start index at seq_len-1 so we have a full history ending at i
    for i in range(seq_len - 1, n):
        if not emit_mask[i]:
            continue
        seq = sig_mat[i - (seq_len - 1) : i + 1]  # V2: [seq_len, 37]
        snap = sig_mat[i]  # V2: [37]
        pending.append(
            {
                "time": times[i],
                "seq": seq,
                "snap": snap,
                "ctx_cont": ctx_cont_mat[i],
                "ctx_cat": ctx_cat_mat[i],
                "y_direction": y_dir[i],
                "y_early_move": y_early[i],
                "y_quality_score": y_qual[i],
                "y_bad_path": y_bad_path[i],
                "y_tradable": y_tradable[i],
                "y_trade": y_trade[i],
                "y_side": y_side[i],
                "y_side_mask": y_side_mask[i],
                "y_tf_agreement_score": y_tf_agreement[i],
                "y_position_size_target": y_position_size[i],
                "mae_first_n_bps": y_mae_first_n[i],
                "mfe_first_n_bps": y_mfe_first_n[i],
                "path_quality_bps": y_path_quality[i],
                "mfe_long_first_n_bps": _mfe_long[i],
                "mae_long_first_n_bps": _mae_long[i],
                "mfe_short_first_n_bps": _mfe_short[i],
                "mae_short_first_n_bps": _mae_short[i],
                "bad_path_long_first_n": float(_bad_path_long[i]),
                "bad_path_short_first_n": float(_bad_path_short[i]),
                "y_long_final_pnl_at_direction_horizon_bps": _pnl_long_at_h[i],
                "y_short_final_pnl_at_direction_horizon_bps": _pnl_short_at_h[i],
                "y_direction_target_mode_id": 1,
                "y_direction_long_score_bps": y_long_path_utility_bps[i],
                "y_direction_short_score_bps": y_short_path_utility_bps[i],
                "y_long_path_utility_bps": y_long_path_utility_bps[i],
                "y_short_path_utility_bps": y_short_path_utility_bps[i],
                "y_long_bad_path": y_long_bad_path[i],
                "y_short_bad_path": y_short_bad_path[i],
                "y_long_expected_mae_bps": y_long_expected_mae_bps[i],
                "y_short_expected_mae_bps": y_short_expected_mae_bps[i],
                "y_rising_channel_support_touch": y_rising_channel_support_touch[i],
                "y_falling_channel_resistance_touch": y_falling_channel_resistance_touch[
                    i
                ],
                "y_support_retest_continuation": y_support_retest_continuation[i],
                "y_resistance_retest_continuation": y_resistance_retest_continuation[i],
                "y_countertrend_short_trap": y_countertrend_short_trap[i],
                "y_countertrend_long_trap": y_countertrend_long_trap[i],
                "y_mtf_conflict_m5_vs_higher_side": y_mtf_conflict_m5_vs_higher_side[i],
                "y_long_high_mae_low_mfe_early_failure": y_long_high_mae_low_mfe_early_failure[
                    i
                ],
                "y_short_high_mae_low_mfe_early_failure": y_short_high_mae_low_mfe_early_failure[
                    i
                ],
                "y_dead_negative_long": y_dead_negative_long[i],
                "y_teaser_negative_long": y_teaser_negative_long[i],
                "y_hard_negative_long": y_hard_negative_long[i],
                "y_clean_edge_long": y_clean_edge_long[i],
                "y_survival_long": y_survival_long[i],
                "y_selector_long_mask": y_selector_long_mask[i],
                "y_dead_negative_short": y_dead_negative_short[i],
                "y_teaser_negative_short": y_teaser_negative_short[i],
                "y_hard_negative_short": y_hard_negative_short[i],
                "y_clean_edge_short": y_clean_edge_short[i],
                "y_survival_short": y_survival_short[i],
                "y_selector_short_mask": y_selector_short_mask[i],
                "y_clean_edge_bidir": y_clean_edge_bidir[i],
                "y_survival_bidir": y_survival_bidir[i],
                "label_horizon_bars": y_label_horizon[i],
                "path_quality_horizon_bars": y_path_horizon[i],
                # aux-head regression targets (dip/forecast/timing/tail/vol) — emit
                # so trainer's row.get(col) finds real values (not silent 0.0).
                **{
                    _c: _head_target_arrays[_c][i]
                    for _c in MODEL_NATIVE_AUX_TARGET_COLUMNS
                },
            }
        )
        if len(pending) >= streaming_batch_size:
            _emit_batch(pending)
            if total_written % (streaming_batch_size * 10) == 0:
                log.info("[V2_STREAMING_WRITE] flushed %d rows", total_written)
    _emit_batch(pending)

    if pq_writer is not None:
        pq_writer.close()
        log.info("[V2_STREAMING_WRITE] closed writer, total_rows=%d", total_written)

    df_out = pd.DataFrame(summary_rows)
    if len(df_out) == 0:
        raise RuntimeError("BUILD_EMPTY_OUTPUT")

    mae_missing = int(pd.isna(df_out["mae_first_n_bps"]).sum())
    mfe_missing = int(pd.isna(df_out["mfe_first_n_bps"]).sum())
    log.info(
        "[ENTRY_PATH_QUALITY_PROOF] rows=%d mae_missing=%d mfe_missing=%d",
        int(len(df_out)),
        mae_missing,
        mfe_missing,
    )
    log.info(
        "[ENTRY_INPUT_SCHEMA_PROOF] signal_dim=%d base_signal_dim=%d seq_structure_dim=%d ctx_cont_dim=%d ctx_cat_dim=%d",
        int(sig_mat.shape[1]),
        int(len(active_base_signal_fields)),
        int(len(seq_structure_feature_names)),
        int(len(ctx_cont_names)),
        int(len(ctx_cat_names)),
    )

    # 10) Metadata
    meta: Dict[str, Any] = {
        "rows": int(len(df_out)),
        "feature_history_start_utc": str(start),
        "feature_computation_end_utc": str(end),
        "emission_start_utc": str(emit_start),
        "emission_end_utc": str(emit_end),
        "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
        "split_reset_allowed": False,
        "causal_context_warmup_rows_trimmed": int(_causal_context_warmup_rows_trimmed),
        "causal_group_a_warmup_rows": int(_causal_group_a_warmup_rows),
        "group_a_checkpoint": _group_a_checkpoint_meta,
        "multi_tf_cache_binding": _multi_tf_cache_binding,
        "causal_regime_v4_warmup_rows": int(_causal_regime_v4_warmup_rows),
        "clean_history_rows_before_emission": int(pre_emit_history_rows),
        "seq_len": int(seq_len),
        "aux_head_target_contract": {
            **model_native_aux_target_contract_metadata(),
            "incomplete_tail_rows_total": int(
                np.count_nonzero(~_head_target_complete_mask)
            ),
            "candidate_rows_before_completeness": int(
                emitted_candidate_count_before_aux_completeness
            ),
            "incomplete_candidate_rows_excluded": int(
                aux_target_incomplete_candidate_rows_excluded
            ),
            "complete_rows_emitted": int(emitted_candidate_count),
        },
        **direction_label_contract(),
        **hierarchical_direction_label_contract(),
        "early_move_threshold_bps": float(early_move_threshold_bps),
        "source_frame": {
            "mode": "exact_source_parquet",
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_sha,
        },
        "canonical_v2_parquet": {
            "path": str(canonical_v2_path),
            "sha256": canonical_v2_sha256,
        },
        "contract_mode": signal_build_contract["contract_mode"],
        "direction_logit_mode": signal_build_contract["direction_logit_mode"],
        "model_native_signal_contract": signal_build_contract[
            "model_native_signal_contract"
        ],
        "tape_root": str(Path(tape_root).resolve()),
        "join_ratio_tape": float(rows_joined / max(1, rows_source)),
        "signal_bridge": {
            "id": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
            "bridge_source": None,
            "fields": list(signal_fields_emitted),
            "base_fields": list(active_base_signal_fields),
            "snap_fields": list(snap_fields_emitted),
            "seq_input_dim": int(sig_mat.shape[1]),
            "snap_input_dim": int(sig_mat.shape[1]),
            "base_seq_input_dim": int(len(active_base_signal_fields)),
            "seq_structure_extension_dim": int(len(seq_structure_feature_names)),
            "contract_sha256": signal_build_contract["model_native_signal_contract"][
                "static_contract_sha256"
            ],
            "bridge_dim": 0,
            "seq_structure_extension_v1": {
                "enabled": bool(seq_structure_feature_names),
                **{k: v for k, v in seq_structure_meta.items() if k != "manifest"},
            },
        },
        "ctx_contract": _model_native_ctx_contract_metadata(),
        "strict_entry_labels": {
            **direction_label_contract(),
            **hierarchical_direction_label_contract(),
            "direction_follows_tradable_side": True,
            "hard_negative_candidate_source": _hard_negative_candidate_source,
            "core_direction_target_provenance": {
                "source": "future_path_and_utility_outcomes_only",
                "feature_derived_rewrite_count": 0,
                "forced_utility_order_count": 0,
            },
            "hard_negative_long_mfe_min_bps": float(HARD_NEG_LONG_MIN_MFE_BPS),
            "hard_negative_long_mae_min_bps": float(HARD_NEG_LONG_MIN_MAE_BPS),
            "hard_negative_long_path_max_bps": float(HARD_NEG_LONG_MAX_PATH_BPS),
            "dead_long_mfe_max_bps": float(DEAD_LONG_MAX_MFE_BPS),
            "dead_long_mae_min_bps": float(DEAD_LONG_MIN_MAE_BPS),
            "teaser_long_mfe_min_bps": float(TEASER_LONG_MIN_MFE_BPS),
            "teaser_long_mfe_max_bps": float(TEASER_LONG_MAX_MFE_BPS),
            "teaser_long_mae_min_bps": float(TEASER_LONG_MIN_MAE_BPS),
            "teaser_long_path_max_bps": float(TEASER_LONG_MAX_PATH_BPS),
            "clean_edge_long_mfe_min_bps": float(CLEAN_EDGE_LONG_MFE_MIN_BPS),
            "clean_edge_long_mae_max_bps": float(CLEAN_EDGE_LONG_MAE_MAX_BPS),
            "clean_edge_long_path_min_bps": float(CLEAN_EDGE_LONG_PATH_MIN_BPS),
            "survival_long_mfe_min_bps": float(SURVIVAL_LONG_MFE_MIN_BPS),
            "survival_long_mae_max_bps": float(SURVIVAL_LONG_MAE_MAX_BPS),
            "survival_long_path_min_bps": float(SURVIVAL_LONG_PATH_MIN_BPS),
            "tradable_excludes_bad_path": True,
        },
        "parked_targets": {
            "bad_path": {
                "horizon_bars": int(BAD_PATH_HORIZON_BARS),
                "mae_threshold_bps": float(BAD_PATH_MAE_THRESHOLD_BPS),
                "mfe_threshold_bps": float(BAD_PATH_MFE_THRESHOLD_BPS),
            }
        },
    }

    return df_out, meta


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the exact model-native ENTRY_V10_CTX seq513/ctx142+5 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Output dataset path (.parquet)."
    )
    parser.add_argument(
        "--resume-exact-checkpoints",
        action="store_true",
        help=(
            "Resume only exact hash-bound Group-A checkpoints after a failed capped "
            "attempt; existing dataset splits remain forbidden."
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Explicit user decision ID bound into every immutable build artifact.",
    )

    # Deterministic filters
    parser.add_argument(
        "--start", type=str, required=True, help="Exact model range start (ISO UTC)."
    )
    parser.add_argument(
        "--end", type=str, required=True, help="Exact model range end (ISO UTC)."
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Deterministic: take first N rows after filtering.",
    )

    # Advanced dataset structure (the model-native sequence contract is explicit)
    parser.add_argument(
        "--seq_len",
        type=int,
        required=True,
        help=f"Exact model-native sequence length (required value: {MODEL_NATIVE_SEQ_LEN}).",
    )
    parser.add_argument(
        "--canonical_v2_parquet",
        type=str,
        required=True,
        help="Explicit canonical feature parquet containing the contracted base signals and context.",
    )

    parser.add_argument(
        "--source-parquet",
        type=str,
        required=True,
        help="Exact canonical M5 source parquet; byte-bound with no alternate resolver.",
    )
    parser.add_argument(
        "--seq-structure-manifest",
        type=str,
        required=True,
        help="Exact model-native selection manifest (34 + 479 = 513 ordered signals).",
    )
    parser.add_argument(
        "--feature-ranking-json",
        type=str,
        required=True,
        help="Exact immutable TRAIN-only feature-ranking JSON bound by the signal manifest.",
    )

    parser.add_argument(
        "--early_move_threshold_bps",
        type=float,
        required=True,
        help="Explicit early-move target threshold in bps, bound into the dataset contract.",
    )

    # Tape lane
    parser.add_argument(
        "--tape_root",
        type=str,
        required=True,
        help="Exact canonical tape lane root.",
    )
    parser.add_argument(
        "--m1-lifecycle-pair-manifest-json",
        type=str,
        required=True,
        help=(
            "Exact generation-local PAIR_MANIFEST.json binding lifecycle M1 "
            "to revalidated immutable native OANDA complete=true responses."
        ),
    )
    parser.add_argument(
        "--m1-lifecycle-pair-generation-root",
        type=str,
        required=True,
        help="Exact immutable canonical pair generation root.",
    )
    parser.add_argument(
        "--m1-feature-base-parquet",
        type=str,
        required=True,
        help=(
            "Exact row-level shared feature-base surface at M1; it must use "
            "the same ordered 513/142/5 contract as Entry."
        ),
    )
    parser.add_argument(
        "--m5-feature-base-parquet",
        type=str,
        required=True,
        help=(
            "Exact hash-bound native M5 feature surface consumed once by all "
            "Entry TRAIN/VAL/TEST split builds."
        ),
    )
    parser.add_argument(
        "--exit-lifecycle-dir",
        type=str,
        required=True,
        help="Fresh immutable output directory for unified Exit episode envelopes.",
    )
    parser.add_argument(
        "--exit-target-lookahead-m1-steps",
        type=int,
        required=True,
        help="Exact positive future M1 target lookahead owned by this dataset event.",
    )

    # Output splitting scaffolding (kept for parity)
    parser.add_argument(
        "--time_split",
        action="store_true",
        help="Write train/val/test outputs (time-based).",
    )
    parser.add_argument(
        "--train_start",
        type=str,
        default=None,
        help="Explicit train split start (ISO).",
    )
    parser.add_argument(
        "--train_end", type=str, default=None, help="Explicit train split end (ISO)."
    )
    parser.add_argument(
        "--val_start",
        type=str,
        default=None,
        help="Explicit validation split start (ISO).",
    )
    parser.add_argument(
        "--val_end", type=str, default=None, help="Explicit validation split end (ISO)."
    )
    parser.add_argument(
        "--test_start", type=str, default=None, help="Explicit test split start (ISO)."
    )
    parser.add_argument(
        "--test_end", type=str, default=None, help="Explicit test split end (ISO)."
    )
    parser.add_argument(
        "--model-native-rank-reference-npz",
        type=str,
        required=True,
        help=(
            "Audited model-native live-state rank reference produced from the same source frame."
        ),
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run: validate inputs/ctx, then exit.",
    )

    args = parser.parse_args()
    entry_run_id = require_entry_run_id(args.run_id)
    build_command = sys.argv.copy()

    main_signal_build_contract = _signal_build_contract_from_manifest(
        Path(args.seq_structure_manifest).expanduser().resolve()
    )

    # Hard gate: ONE UNIVERSE
    ctx = _hard_gate_model_native_context()
    log.info(
        f"[CTX_CONTRACT] OK: tag={ctx['tag']} cont={ctx['ctx_cont_dim']} cat={ctx['ctx_cat_dim']}"
    )

    if int(args.seq_len) != MODEL_NATIVE_SEQ_LEN:
        raise RuntimeError(
            f"MODEL_NATIVE_SEQ_LEN_INVALID: got={args.seq_len} expected={MODEL_NATIVE_SEQ_LEN}"
        )
    early_move_threshold_bps = float(args.early_move_threshold_bps)
    if not np.isfinite(early_move_threshold_bps) or early_move_threshold_bps < 0.0:
        raise RuntimeError(
            "EARLY_MOVE_THRESHOLD_INVALID: explicit finite non-negative bps value required"
        )

    start = _parse_ts(args.start)
    end = _parse_ts(args.end)
    if start is None or end is None or start > end:
        raise RuntimeError(f"MODEL_RANGE_INVALID: start={start} end={end}")
    if not args.time_split:
        raise RuntimeError(
            "MODEL_NATIVE_TIME_SPLIT_REQUIRED: TRAIN-only normalization has no single-frame fallback"
        )
    if args.max_rows is not None:
        raise RuntimeError(
            "MODEL_NATIVE_TIME_SPLIT_MAX_ROWS_FORBIDDEN: deterministic truncation would break common history"
        )
    train_start = _parse_ts(args.train_start)
    train_end = _parse_ts(args.train_end)
    val_start = _parse_ts(args.val_start)
    val_end = _parse_ts(args.val_end)
    test_start = _parse_ts(args.test_start)
    test_end = _parse_ts(args.test_end)
    split_points = (train_start, train_end, val_start, val_end, test_start, test_end)
    if any(point is None for point in split_points):
        raise RuntimeError("MODEL_NATIVE_SPLIT_WINDOW_MISSING")
    if not (
        start
        <= train_start
        <= train_end
        < val_start
        <= val_end
        < test_start
        <= test_end
        == end
    ):
        raise RuntimeError(
            "MODEL_NATIVE_SPLIT_WINDOWS_INVALID: expected one common history start and ordered, "
            "non-overlapping TRAIN/VAL/TEST windows ending exactly at --end"
        )
    state_contract = _model_native_state_contract(
        args=args,
        feature_history_start=start,
        train_start=train_start,
        train_end=train_end,
    )
    signal_lineage = validate_signal_manifest_training_lineage(
        manifest_path=Path(args.seq_structure_manifest),
        feature_ranking_path=Path(args.feature_ranking_json),
        expected_run_id=entry_run_id,
        expected_source_parquet=Path(args.source_parquet),
        expected_source_sha256=_sha256_file(Path(args.source_parquet)),
        expected_canonical_v2_parquet=Path(args.canonical_v2_parquet),
        expected_mtf_cache_dir=Path(
            os.environ["GX1_V10_MULTI_TF_V4_CACHE_DIR"]
        ),
        expected_history_start_utc=start,
        expected_time_max_utc=end,
        expected_train_start_utc=train_start.isoformat(),
        expected_train_end_utc=train_end.isoformat(),
    )
    if (
        signal_lineage["model_native_signal_contract"]
        != main_signal_build_contract["model_native_signal_contract"]
    ):
        raise RuntimeError("MODEL_NATIVE_SIGNAL_LINEAGE_CONTRACT_CHANGED")

    # Dataset build proof (will be written after output_path resolved)
    ctx_contract = _model_native_ctx_contract_metadata()
    proof_payload = {
        "entry_run_id": entry_run_id,
        "ctx_tag": ctx_contract["tag"],
        "ctx_cont_dim": int(len(MODEL_NATIVE_CTX_CONT_FIELDS)),
        "ctx_cat_dim": int(len(MODEL_NATIVE_CTX_CAT_FIELDS)),
        "signal_bridge_id": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "signal_bridge_contract_sha256": main_signal_build_contract[
            "model_native_signal_contract"
        ]["static_contract_sha256"],
        "contract_mode": main_signal_build_contract["contract_mode"],
        "direction_logit_mode": main_signal_build_contract["direction_logit_mode"],
        "model_native_signal_contract": main_signal_build_contract[
            "model_native_signal_contract"
        ],
        "signal_training_lineage": {
            key: value
            for key, value in signal_lineage.items()
            if key != "model_native_signal_contract"
        },
        "ctx_contract": ctx_contract,
        "model_native_state_contract": state_contract,
        "aux_head_target_contract": model_native_aux_target_contract_metadata(),
        **direction_label_contract(),
        **hierarchical_direction_label_contract(),
        "group_a_recompute_workers": _MODEL_NATIVE_GROUP_A_RECOMPUTE_WORKERS,
        "seq_structure_extension_v1": {
            "manifest_path": str(
                Path(args.seq_structure_manifest).expanduser().resolve()
            ),
            "mode": ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE,
        },
    }

    # One explicit source path. No truth-config, BASE28 resolution or manual
    # preference lane may alter the build input.
    source_parquet_path = Path(args.source_parquet).expanduser().resolve()
    if not source_parquet_path.is_file():
        raise RuntimeError(f"SOURCE_PARQUET_MISSING: {source_parquet_path}")
    proof_payload.update({"truth_source": "exact_source_parquet"})

    output_path = Path(args.output).resolve()
    if DIRECTION_DATASET_STEM_SUFFIX not in output_path.stem:
        output_path = output_path.with_name(
            f"{output_path.stem}{DIRECTION_DATASET_STEM_SUFFIX}{output_path.suffix}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proof_payload.update(
        {
            "source_parquet": str(source_parquet_path),
            "output_path": str(output_path),
        }
    )
    m1_lifecycle_source_path, m1_lifecycle_authority = (
        require_unified_exit_m1_pair_authority(
            pair_manifest_path=Path(
                args.m1_lifecycle_pair_manifest_json
            ),
            pair_generation_root=Path(
                args.m1_lifecycle_pair_generation_root
            ),
        )
    )
    m1_lifecycle_authority_sha256 = canonical_json_sha256(
        m1_lifecycle_authority
    )
    exit_lifecycle_dir = Path(args.exit_lifecycle_dir).expanduser()
    if (
        not exit_lifecycle_dir.is_absolute()
        or exit_lifecycle_dir.exists()
        or exit_lifecycle_dir.is_symlink()
        or exit_lifecycle_dir.parent.is_symlink()
        or not exit_lifecycle_dir.parent.is_dir()
    ):
        raise RuntimeError(
            "UNIFIED_EXIT_LIFECYCLE_OUTPUT_INVALID: fresh absolute directory "
            f"with an existing real parent required: {exit_lifecycle_dir}"
        )
    if (
        isinstance(args.exit_target_lookahead_m1_steps, bool)
        or int(args.exit_target_lookahead_m1_steps) <= 0
    ):
        raise RuntimeError(
            "UNIFIED_EXIT_TARGET_LOOKAHEAD_INVALID: explicit positive integer required"
        )
    exit_target_lookahead = int(args.exit_target_lookahead_m1_steps)
    m1_lifecycle_source_sha256 = _sha256_file(
        m1_lifecycle_source_path
    )
    m1_feature_base_path = Path(args.m1_feature_base_parquet).expanduser().resolve()
    m1_feature_base_sha256 = _sha256_file(m1_feature_base_path)
    m1_feature_base_manifest_path = Path(
        str(m1_feature_base_path) + ".manifest.json"
    )
    if not m1_feature_base_manifest_path.is_file():
        raise RuntimeError("DATASET_BUILDER_M1_FEATURE_BASE_MANIFEST_MISSING")
    m1_feature_base_manifest_sha256 = _sha256_file(
        m1_feature_base_manifest_path
    )
    m1_feature_base_manifest = json.loads(
        m1_feature_base_manifest_path.read_text(encoding="utf-8")
    )
    m1_manifest_without_hash = dict(m1_feature_base_manifest)
    m1_declared_manifest_sha256 = m1_manifest_without_hash.pop(
        "manifest_sha256", None
    )
    if (
        m1_feature_base_manifest.get("schema_version")
        != "gx1_entry_exit_m1_feature_surface_v1"
        or m1_feature_base_manifest.get("decision") != "PASS"
        or m1_feature_base_manifest.get("dataset_run_id") != entry_run_id
        or m1_feature_base_manifest.get("pair_generation_id")
        != m1_lifecycle_authority.get("pair_generation_id")
        or m1_feature_base_manifest.get("output_parquet")
        != str(m1_feature_base_path)
        or m1_feature_base_manifest.get("output_parquet_sha256")
        != m1_feature_base_sha256
        or m1_declared_manifest_sha256
        != canonical_json_sha256(m1_manifest_without_hash)
    ):
        raise RuntimeError(
            "DATASET_BUILDER_M1_FEATURE_BASE_MANIFEST_CONTRACT_INVALID"
        )
    signal_manifest_path = Path(args.seq_structure_manifest).expanduser().resolve()
    require_entry_exit_feature_surface_identity(
        m1_feature_base_manifest,
        expected_timeframe="M1",
        expected_ordered_fields=main_signal_build_contract[
            "model_native_signal_contract"
        ]["fields"],
        expected_signal_manifest_path=str(signal_manifest_path),
        expected_signal_manifest_sha256=_sha256_file(signal_manifest_path),
        expected_rank_reference_sha256=state_contract[
            "rank_reference_npz_sha256"
        ],
        context="DATASET_BUILDER_M1_VS_ENTRY_M5",
    )
    m1_feature_times = load_m1_feature_surface_times(
        m1_feature_base_path,
        context="DATASET_BUILDER",
    )
    m1_source_times = pd.DatetimeIndex(
        pd.to_datetime(
            pd.read_parquet(m1_lifecycle_source_path, columns=["time"])["time"],
            utc=True,
            errors="coerce",
        )
    ).as_unit("ns")
    if (
        len(m1_feature_times) < len(m1_source_times)
        or not m1_source_times.isin(m1_feature_times).all()
        or m1_feature_times[-1] != m1_source_times[-1]
        or m1_feature_base_manifest.get("rows") != len(m1_feature_times)
    ):
        raise RuntimeError("DATASET_BUILDER_M1_FEATURE_BASE_TIME_MISMATCH")
    m5_feature_base_path = Path(args.m5_feature_base_parquet).expanduser().resolve()
    if m5_feature_base_path.is_symlink() or not m5_feature_base_path.is_file():
        raise RuntimeError("DATASET_BUILDER_M5_FEATURE_BASE_MISSING")
    m5_feature_base_sha256 = _sha256_file(m5_feature_base_path)
    m5_feature_base_manifest_path = Path(
        str(m5_feature_base_path) + ".manifest.json"
    )
    if (
        m5_feature_base_manifest_path.is_symlink()
        or not m5_feature_base_manifest_path.is_file()
    ):
        raise RuntimeError("DATASET_BUILDER_M5_FEATURE_BASE_MANIFEST_MISSING")
    m5_feature_base_manifest_sha256 = _sha256_file(
        m5_feature_base_manifest_path
    )
    m5_feature_base_manifest = json.loads(
        m5_feature_base_manifest_path.read_text(encoding="utf-8")
    )
    m5_manifest_without_hash = dict(m5_feature_base_manifest)
    m5_declared_manifest_sha256 = m5_manifest_without_hash.pop(
        "manifest_sha256", None
    )
    if (
        m5_feature_base_manifest.get("schema_version")
        != ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION
        or m5_feature_base_manifest.get("decision") != "PASS"
        or m5_feature_base_manifest.get("dataset_run_id") != entry_run_id
        or m5_feature_base_manifest.get("pair_generation_id")
        != m1_lifecycle_authority.get("pair_generation_id")
        or m5_feature_base_manifest.get("output_parquet")
        != str(m5_feature_base_path)
        or m5_feature_base_manifest.get("output_parquet_sha256")
        != m5_feature_base_sha256
        or m5_declared_manifest_sha256
        != canonical_json_sha256(m5_manifest_without_hash)
    ):
        raise RuntimeError(
            "DATASET_BUILDER_M5_FEATURE_BASE_MANIFEST_CONTRACT_INVALID"
        )
    require_entry_exit_feature_surface_identity(
        m5_feature_base_manifest,
        expected_timeframe="M5",
        expected_ordered_fields=main_signal_build_contract[
            "model_native_signal_contract"
        ]["fields"],
        expected_signal_manifest_path=str(signal_manifest_path),
        expected_signal_manifest_sha256=_sha256_file(signal_manifest_path),
        expected_rank_reference_sha256=state_contract[
            "rank_reference_npz_sha256"
        ],
        context="DATASET_BUILDER_ENTRY_M5_FEATURE_SURFACE",
    )
    m5_feature_times_expected = load_m1_feature_surface_times(
        m5_feature_base_path,
        context="DATASET_BUILDER_ENTRY_M5",
        expected_bar_seconds=ENTRY_DECISION_BAR_SECONDS,
    )
    m5_source_times = pd.DatetimeIndex(
        pd.to_datetime(
            pd.read_parquet(source_parquet_path, columns=["time"])["time"],
            utc=True,
            errors="coerce",
        )
    ).as_unit("ns")
    if (
        not m5_feature_times_expected.equals(m5_source_times)
        or m5_feature_base_manifest.get("rows")
        != len(m5_feature_times_expected)
    ):
        raise RuntimeError("DATASET_BUILDER_M5_FEATURE_BASE_TIME_MISMATCH")
    m5_feature_surface_binding = {
        "schema_version": ENTRY_EXIT_M5_FEATURE_SURFACE_SCHEMA_VERSION,
        "path": str(m5_feature_base_path),
        "sha256": m5_feature_base_sha256,
        "manifest_path": str(m5_feature_base_manifest_path),
        "manifest_sha256": m5_feature_base_manifest_sha256,
        "dataset_run_id": entry_run_id,
        "pair_generation_id": m1_lifecycle_authority.get("pair_generation_id"),
        "rows": len(m5_feature_times_expected),
        "time_alignment": "exact_entry_m5_source_timeline",
        "signal_manifest_sha256": _sha256_file(signal_manifest_path),
        "rank_reference_sha256": state_contract["rank_reference_npz_sha256"],
        "inline_split_recomputation": False,
    }
    proof_payload["entry_m5_feature_surface"] = m5_feature_surface_binding
    proof_payload["seq_structure_extension_v1"]["mode"] = (
        ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE
    )
    proof_payload["unified_exit_lifecycle"] = {
        "schema_version": (
            UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
        ),
        "m1_source_path": str(m1_lifecycle_source_path),
        "m1_source_sha256": m1_lifecycle_source_sha256,
        "m1_feature_base_path": str(m1_feature_base_path),
        "m1_feature_base_sha256": m1_feature_base_sha256,
        "m1_feature_base_manifest_path": str(m1_feature_base_manifest_path),
        "m1_feature_base_manifest_sha256": m1_feature_base_manifest_sha256,
        "feature_base_time_alignment": "exact_m1_source_timestamp_subset_with_causal_prefix",
        "m1_authority": m1_lifecycle_authority,
        "m1_authority_sha256": m1_lifecycle_authority_sha256,
        "output_dir": str(exit_lifecycle_dir),
        "target_lookahead_m1_steps": exit_target_lookahead,
        "m1_row_clock": EXIT_FEATURE_ROW_CLOCK,
        "path_state_count": int(UNIFIED_EXIT_MAX_PATH_BARS),
        "side_order": list(UNIFIED_EXIT_SIDE_ORDER),
        "action_order": list(UNIFIED_EXIT_ACTION_ORDER),
        "shared_feature_base_contract": (
            entry_exit_shared_feature_base_contract()
        ),
    }

    tape_root = Path(args.tape_root).expanduser().resolve()
    if not tape_root.is_dir():
        raise RuntimeError(f"CANONICAL_TAPE_ROOT_MISSING: {tape_root}")
    xau_tape_provenance = validate_xau_tape_provenance_v1(
        tape_root,
        expected_run_id=entry_run_id,
        require_current=True,
    )
    canonical_v2_path = Path(args.canonical_v2_parquet).expanduser().resolve()
    if not canonical_v2_path.is_file():
        raise RuntimeError(f"CANONICAL_V2_PARQUET_NOT_FOUND: {canonical_v2_path}")
    proof_payload.update(
        {
            "tape_root": str(tape_root),
            "xau_tape_provenance": xau_tape_provenance,
            "time_start_utc": str(start),
            "time_end_utc": str(end),
        }
    )
    proof_path = output_path.parent / "DATASET_BUILD_PROOF.json"
    proof_bytes = (
        json.dumps(proof_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if proof_path.exists() or proof_path.is_symlink():
        if not args.resume_exact_checkpoints:
            raise RuntimeError(
                "DATASET_BUILD_PROOF_ALREADY_EXISTS: choose a fresh output directory "
                f"or use explicit exact-checkpoint recovery: {proof_path}"
            )
        if proof_path.is_symlink() or not proof_path.is_file():
            raise RuntimeError("DATASET_BUILD_PROOF_RESUME_PATH_INVALID")
        if proof_path.read_bytes() != proof_bytes:
            raise RuntimeError("DATASET_BUILD_PROOF_RESUME_IDENTITY_MISMATCH")
        log.info("[DATASET_BUILD_PROOF] exact resume identity validated %s", proof_path)
    else:
        if args.resume_exact_checkpoints:
            raise RuntimeError("DATASET_BUILD_PROOF_RESUME_MISSING")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(proof_path, flags, 0o644)
        try:
            view = memoryview(proof_bytes)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError(f"short dataset build proof write: {proof_path}")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        log.info("[DATASET_BUILD_PROOF] wrote %s", proof_path)

    if args.dry_run:
        log.info("[DRY_RUN] Inputs exist and CTX contract is valid. Exiting.")
        write_manifest(
            output_path=output_path,
            build_command=build_command,
            source_parquet=source_parquet_path,
            tape_root=tape_root,
            notes="DRY_RUN only.",
            extra={
                "start": args.start,
                "end": args.end,
                "max_rows": args.max_rows,
                "time_split": bool(args.time_split),
                "seq_len": int(args.seq_len),
                "aux_head_target_contract": model_native_aux_target_contract_metadata(),
                **direction_label_contract(),
                **hierarchical_direction_label_contract(),
                "early_move_threshold_bps": early_move_threshold_bps,
                "contract_mode": main_signal_build_contract["contract_mode"],
                "direction_logit_mode": main_signal_build_contract[
                    "direction_logit_mode"
                ],
                "model_native_signal_contract": main_signal_build_contract[
                    "model_native_signal_contract"
                ],
                "signal_bridge": {
                    "id": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
                    "fields": list(
                        main_signal_build_contract["model_native_signal_contract"][
                            "fields"
                        ]
                    ),
                    "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                    "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
                    "bridge_dim": 0,
                    "bridge_source": None,
                },
                "ctx_contract": _model_native_ctx_contract_metadata(),
                "seq_structure_extension_v1": {
                    "manifest_path": str(
                        Path(args.seq_structure_manifest).expanduser().resolve()
                    )
                    if args.seq_structure_manifest
                    else None,
                    "mode": ENTRY_M5_FEATURE_SURFACE_CONSUMPTION_MODE,
                    "feature_surface": m5_feature_surface_binding,
                    "inline_split_recomputation": False,
                },
                "model_native_state_contract": state_contract,
                "xau_tape_provenance": xau_tape_provenance,
                "entry_run_id": entry_run_id,
            },
        )
        return

    m5_surface_storage = tempfile.TemporaryDirectory(
        prefix=".entry_m5_feature_surface.",
        dir=str(output_path.parent),
    )
    m5_feature_surface_times, m5_feature_surface_arrays = load_m1_feature_surface(
        m5_feature_base_path,
        context="DATASET_BUILDER_ENTRY_M5",
        storage_dir=Path(m5_surface_storage.name),
        expected_bar_seconds=ENTRY_DECISION_BAR_SECONDS,
    )
    if not m5_feature_surface_times.equals(m5_feature_times_expected):
        raise RuntimeError("DATASET_BUILDER_M5_FEATURE_BASE_LOAD_TIME_MISMATCH")

    splits = {
        "train": {"start": str(train_start), "end": str(train_end)},
        "val": {"start": str(val_start), "end": str(val_end)},
        "test": {"start": str(test_start), "end": str(test_end)},
    }
    base = output_path
    out_dir = base.parent
    stem = base.stem

    metas: Dict[str, Any] = {}
    ts_min_max_by_split: Dict[str, Dict[str, Optional[str]]] = {}
    rank_reference_path = (
        Path(args.model_native_rank_reference_npz).expanduser().resolve()
    )
    closed_m1_lifecycle = pd.read_parquet(
        m1_lifecycle_source_path,
        columns=list(UNIFIED_EXIT_LIFECYCLE_REQUIRED_M1_COLUMNS),
    )
    assert_no_price_scale_glitch(
        closed_m1_lifecycle,
        context="UNIFIED_EXIT_M1_SOURCE",
    )
    lifecycle_staging = tempfile.TemporaryDirectory(
        prefix=f".{exit_lifecycle_dir.name}.staging.",
        dir=str(exit_lifecycle_dir.parent),
    )
    lifecycle_stage_dir = Path(lifecycle_staging.name).resolve(strict=True)
    lifecycle_split_bindings: dict[str, dict[str, Any]] = {}

    # Each split is emitted from the same history anchor.  The computation end
    # advances only to that split's own boundary, so no later split can affect
    # either features or labels and label horizons cannot cross a boundary.
    for split_name, (s0, s1) in {
        "train": (train_start, train_end),
        "val": (val_start, val_end),
        "test": (test_start, test_end),
    }.items():
        log.info(
            "[BUILD_COMMON_HISTORY] split=%s history_start=%s emit=%s..%s",
            split_name,
            start,
            s0,
            s1,
        )
        out = out_dir / f"{stem}_{split_name}.parquet"
        df_built, meta = build_dataset_canonical(
            source_parquet=source_parquet_path,
            tape_root=tape_root,
            start=start,
            end=s1,
            emit_start=s0,
            emit_end=s1,
            model_native_rank_reference_npz=rank_reference_path,
            m5_feature_surface_times=m5_feature_surface_times,
            m5_feature_surface_arrays=m5_feature_surface_arrays,
            m5_feature_surface_binding=m5_feature_surface_binding,
            max_rows=None,
            seq_len=int(args.seq_len),
            early_move_threshold_bps=early_move_threshold_bps,
            split_name=split_name,
            canonical_v2_parquet=canonical_v2_path,
            output_path=out,
            seq_structure_manifest_path=Path(args.seq_structure_manifest)
            .expanduser()
            .resolve(),
        )
        _log_label_distribution_proof(df_built, split=split_name)
        lifecycle_episodes, lifecycle_proof = (
            build_unified_exit_lifecycle_episodes(
                entry_rows=df_built,
                closed_m1=closed_m1_lifecycle,
                split_end=s1,
                target_lookahead_m1_steps=exit_target_lookahead,
                market_closure_contract=m1_lifecycle_authority[
                    "native_m1_market_closure_contract"
                ],
            )
        )
        lifecycle_parquet = (
            lifecycle_stage_dir
            / f"{split_name}_unified_exit_lifecycle.parquet"
        )
        lifecycle_episodes.to_parquet(lifecycle_parquet, index=False)
        with lifecycle_parquet.open("rb") as handle:
            os.fsync(handle.fileno())
        lifecycle_proof.update(
            {
                "entry_run_id": entry_run_id,
                "split": split_name,
                "entry_dataset_path": str(out.resolve(strict=True)),
                "entry_dataset_sha256": _sha256_file(out),
                "m1_source_path": str(m1_lifecycle_source_path),
                "m1_source_sha256": m1_lifecycle_source_sha256,
                "m1_feature_base_path": str(m1_feature_base_path),
                "m1_feature_base_sha256": m1_feature_base_sha256,
                "m1_feature_base_manifest_path": str(
                    m1_feature_base_manifest_path
                ),
                "m1_feature_base_manifest_sha256": (
                    m1_feature_base_manifest_sha256
                ),
                "m1_authority_sha256": (
                    m1_lifecycle_authority_sha256
                ),
                "lifecycle_parquet": lifecycle_parquet.name,
                "lifecycle_parquet_sha256": _sha256_file(
                    lifecycle_parquet
                ),
                "lifecycle_parquet_rows": int(len(lifecycle_episodes)),
            }
        )
        lifecycle_manifest = (
            lifecycle_stage_dir
            / f"{split_name}_unified_exit_lifecycle.manifest.json"
        )
        _write_bytes_exclusive_fsync(
            lifecycle_manifest,
            (
                json.dumps(
                    lifecycle_proof,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        lifecycle_split_bindings[split_name] = {
            "entry_dataset_path": lifecycle_proof["entry_dataset_path"],
            "entry_dataset_sha256": lifecycle_proof[
                "entry_dataset_sha256"
            ],
            "lifecycle_parquet": lifecycle_parquet.name,
            "lifecycle_parquet_sha256": lifecycle_proof[
                "lifecycle_parquet_sha256"
            ],
            "lifecycle_manifest": lifecycle_manifest.name,
            "lifecycle_manifest_sha256": _sha256_file(
                lifecycle_manifest
            ),
            "episode_rows": int(len(lifecycle_episodes)),
            "target_counts": lifecycle_proof["target_counts"],
            "target_stream_sha256": lifecycle_proof[
                "target_stream_sha256"
            ],
        }
        metas[split_name] = deepcopy(meta)
        ts_min_max_by_split[split_name] = _split_min_max_from_ts_series(
            df_built["time"]
        )
        write_manifest(
            output_path=out,
            build_command=build_command,
            source_parquet=source_parquet_path,
            tape_root=tape_root,
            splits=splits,
            ts_min_max_by_split=ts_min_max_by_split,
            notes=(
                f"Canonical common-history build completed for split={split_name}; "
                "no split-local feature reset."
            ),
            extra={
                **metas[split_name],
                "model_native_state_contract": state_contract,
                "xau_tape_provenance": xau_tape_provenance,
                "entry_run_id": entry_run_id,
            },
        )

    lifecycle_root_manifest = {
        "schema_version": (
            UNIFIED_EXIT_LIFECYCLE_EPISODE_SCHEMA_VERSION
        ),
        "decision": "PASS",
        "entry_run_id": entry_run_id,
        "m1_source_path": str(m1_lifecycle_source_path),
        "m1_source_sha256": m1_lifecycle_source_sha256,
        "m1_feature_base_path": str(m1_feature_base_path),
        "m1_feature_base_sha256": m1_feature_base_sha256,
        "m1_feature_base_manifest_path": str(m1_feature_base_manifest_path),
        "m1_feature_base_manifest_sha256": m1_feature_base_manifest_sha256,
        "m1_authority": m1_lifecycle_authority,
        "m1_authority_sha256": m1_lifecycle_authority_sha256,
        "path_state_count": int(UNIFIED_EXIT_MAX_PATH_BARS),
        "target_lookahead_m1_steps": exit_target_lookahead,
        "m1_row_clock": EXIT_FEATURE_ROW_CLOCK,
        "shared_feature_base_contract": (
            entry_exit_shared_feature_base_contract()
        ),
        "side_order": list(UNIFIED_EXIT_SIDE_ORDER),
        "action_order": list(UNIFIED_EXIT_ACTION_ORDER),
        "splits": lifecycle_split_bindings,
    }
    lifecycle_root_manifest_path = (
        lifecycle_stage_dir / "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json"
    )
    _write_bytes_exclusive_fsync(
        lifecycle_root_manifest_path,
        (
            json.dumps(
                lifecycle_root_manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )
    expected_lifecycle_inventory = {
        "UNIFIED_EXIT_LIFECYCLE_MANIFEST.json",
        *(
            f"{split}_unified_exit_lifecycle{suffix}"
            for split in ("train", "val", "test")
            for suffix in (".parquet", ".manifest.json")
        ),
    }
    observed_lifecycle_inventory = {
        path.name for path in lifecycle_stage_dir.iterdir()
    }
    if (
        observed_lifecycle_inventory != expected_lifecycle_inventory
        or any(
            path.is_symlink() or not path.is_file()
            for path in lifecycle_stage_dir.iterdir()
        )
    ):
        raise RuntimeError(
            "UNIFIED_EXIT_LIFECYCLE_STAGING_INVENTORY_INVALID"
        )
    _fsync_directory(lifecycle_stage_dir)
    publish_bundle_directory_noreplace(
        lifecycle_stage_dir,
        exit_lifecycle_dir,
    )
    lifecycle_staging.cleanup()
    del m5_feature_surface_arrays
    m5_surface_storage.cleanup()
    log.info("[DATASET_BUILD] Common-history TRAIN/VAL/TEST build complete")


if __name__ == "__main__":
    main()
