"""
EXIT_IO_V8_REGIME_M1L512 contract.

Extension of EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512 (155 features) for the 2026-06-03
regime-everywhere wave — mirrors the V10 entry-side REGIME_V4 onto the exit transformer so the
V3 exit model gets the SAME explicit multi-TF regime CONDITIONING + regime-CHANGE-DETECTION
the entry side now has (previously the exit transformer saw regime ONLY via the multi-TF seq
encoders, with no explicit regime ctx — the gap-register finding).

  Δ vs V7:
    + 18 REGIME_V4 features (SAME names + computation as the entry V10 ctx — one truth:
        gx1.features.regime_v4_features.REGIME_V4_FEATURE_NAMES):
        per-TF regime class (m5/m15/h1/h4/d1) + per-TF trend-age (5) — multi-TF regime STATE;
        regime_tf_agreement / regime_stack_sum / regime_divergence_flag — cross-TF state;
        d1_dist_roc_288 / d1_dist_to_boundary / d1_regime_changed_flag /
        bars_since_d1_regime_change / d1_trend_age_mature_flag — regime CHANGE-DETECTION.
      Decision-context features broadcast across the 512-M1 window (one value per sample, like
      V7 treats group-A / dip-struct) — NOT recomputed per M1 bar. For the exit, the
      change-detection features answer "is the regime I entered now shifting → take profit
      before the reversal."

Total: 155 + 18 = 173 features per M1 bar, × 512 M1 window.

Backward-compat:
  - First 155 features identical to V7 — V8 transformer can prefix-init from V7.
  - REGIME_V4 appended at the tail (indices 155..172).

One-truth: the regime tail == gx1.features.regime_v4_features.REGIME_V4_FEATURE_NAMES
(the SAME list the entry V10 ctx_cont uses). Asserted below.
"""
from __future__ import annotations

from typing import List

from gx1.exits.contracts.exit_io_v1_ctx36 import compute_feature_names_hash
from gx1.exits.contracts.exit_io_v7_volume_dipstruct_m1l512 import (
    EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURES,
    compute_m5_phase_index,
    compute_m5_phase_onehot,
)
from gx1.features.regime_v4_features import REGIME_V4_FEATURE_NAMES


EXIT_IO_V8_REGIME_M1L512_IO_VERSION = "EXIT_IO_V8_REGIME_M1L512"
EXIT_IO_V8_REGIME_M1L512_DEFAULT_WINDOW_LEN = 512

# Appended group (one-truth reference, not re-listed literally).
EXIT_IO_V8_REGIME_FEATURES: List[str] = list(REGIME_V4_FEATURE_NAMES)  # 16

EXIT_IO_V8_REGIME_M1L512_FEATURES: List[str] = (
    list(EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURES)
    + EXIT_IO_V8_REGIME_FEATURES
)
EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT = len(EXIT_IO_V8_REGIME_M1L512_FEATURES)
EXIT_IO_V8_REGIME_M1L512_FEATURE_TO_INDEX = {
    name: idx for idx, name in enumerate(EXIT_IO_V8_REGIME_M1L512_FEATURES)
}
EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH = compute_feature_names_hash(
    EXIT_IO_V8_REGIME_M1L512_FEATURES
)


def assert_exit_io_v8_regime_m1l512_contract() -> None:
    feats = EXIT_IO_V8_REGIME_M1L512_FEATURES
    if len(feats) != EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT:
        raise RuntimeError(
            f"[EXIT_IO_V8_CONTRACT] len mismatch: {len(feats)} != "
            f"{EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT}"
        )
    # V7 prefix invariance (warm-start from V7).
    v7 = EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURES
    if feats[: len(v7)] != list(v7):
        raise RuntimeError("[EXIT_IO_V8_CONTRACT] V7 prefix not preserved")
    # ONE-TRUTH: regime tail must equal the canonical REGIME_V4 list (no drift vs the entry V10).
    n_reg = len(EXIT_IO_V8_REGIME_FEATURES)
    if feats[-n_reg:] != list(REGIME_V4_FEATURE_NAMES):
        raise RuntimeError("[EXIT_IO_V8_CONTRACT] regime tail drift vs REGIME_V4_FEATURE_NAMES")
    # no duplicate feature names
    if len(set(feats)) != len(feats):
        dupes = [f for f in feats if feats.count(f) > 1]
        raise RuntimeError(f"[EXIT_IO_V8_CONTRACT] duplicate features: {sorted(set(dupes))}")
    got = compute_feature_names_hash(feats)
    if got != EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_IO_V8_CONTRACT] hash mismatch: got={got} "
            f"expected={EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH}"
        )


__all__ = [
    "EXIT_IO_V8_REGIME_M1L512_IO_VERSION",
    "EXIT_IO_V8_REGIME_M1L512_DEFAULT_WINDOW_LEN",
    "EXIT_IO_V8_REGIME_FEATURES",
    "EXIT_IO_V8_REGIME_M1L512_FEATURES",
    "EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT",
    "EXIT_IO_V8_REGIME_M1L512_FEATURE_TO_INDEX",
    "EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH",
    "assert_exit_io_v8_regime_m1l512_contract",
    "compute_m5_phase_index",
    "compute_m5_phase_onehot",
]


assert_exit_io_v8_regime_m1l512_contract()
