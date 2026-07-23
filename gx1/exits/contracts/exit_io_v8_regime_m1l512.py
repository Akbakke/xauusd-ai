"""
EXIT_IO_V8_REGIME_M1L512 contract.

Semantic V2 of the 173-field V8 layout.  The feature order remains the V7
155-field prefix plus 18 REGIME_V4 fields, but cadence is now explicit and is
part of the IO version so legacy broadcast artifacts cannot be served:

  Δ vs V7:
    + 18 REGIME_V4 features (SAME names + computation as the entry V10 ctx — one truth:
        gx1.features.regime_v4_features.REGIME_V4_FEATURE_NAMES):
        per-TF regime class (m5/m15/h1/h4/d1) + per-TF trend-age (5) — multi-TF regime STATE;
        regime_tf_agreement / regime_stack_sum / regime_divergence_flag — cross-TF state;
        d1_dist_roc_288 / d1_dist_to_boundary / d1_regime_changed_flag /
        bars_since_d1_regime_change / d1_trend_age_mature_flag — regime CHANGE-DETECTION.
  Cadence:
    - 4 volume fields are computed natively at M1 cadence for every historical
      row, with the exact 95-row causal prefix.
    - 24 group-A + 36 dip/structure + 18 regime fields are historical
      decision-context at every M1 row.  An M1 bar labelled T maps to
      floor(T+1 minute, 5 minutes)-5 minutes, the newest fully closed M5 row.
    - No current-sample broadcast is allowed.  This lets the transformer see
      how structure/regime/confluence changed along the 512-bar path without
      lookahead.

Total: 155 + 18 = 173 features per M1 bar, × 512 M1 window.

Weight initialization:
  - First 155 feature positions remain identical to V7, so weights may be
    prefix-initialized.  A V7/V8-legacy dataset or bundle is not IO-compatible
    because its broadcast cadence had different meaning.
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


EXIT_IO_V8_REGIME_M1L512_IO_VERSION = (
    "EXIT_IO_V8_REGIME_PER_M1_CLOSED_M5_V2"
)
EXIT_IO_V8_CONTEXT_CADENCE_CONTRACT = (
    "m1_native_volume_95_prefix__historical_per_m1_latest_closed_m5_context_v1"
)
EXIT_IO_V8_REGIME_M1L512_DEFAULT_WINDOW_LEN = 512

# Appended group (one-truth reference, not re-listed literally).
EXIT_IO_V8_REGIME_FEATURES: List[str] = list(REGIME_V4_FEATURE_NAMES)  # 18

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
    "EXIT_IO_V8_CONTEXT_CADENCE_CONTRACT",
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
