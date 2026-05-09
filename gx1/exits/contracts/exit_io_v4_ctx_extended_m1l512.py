"""
EXIT_IO_V4_CTX_EXTENDED_M1L512 contract.

Extension of EXIT_IO_V3_CTX36_M1L512_PHASE5 for the 2026-Q2 stack rebuild,
designed to symmetrically match V10 v2's option α (extended multi-TF ctx).

Deltas vs V3:
  - V1 ctx36 base (53 features) preserved at indices 0..52
  - 5 m5_phase features at indices 53..57 (same as V3 PHASE5)
  - **NEW**: 22 multi-TF context features at indices 58..79 (V2 extension —
    explicit H1/H4/D1/M15 indicators sourced from canonical_v2)

Total: 80 features per M1 bar (vs 58 in V3), × 512 M1 window.

Backward-compat:
  - First 58 features are IDENTICAL to V3's contract (same names, same order).
  - V4 transformer can load V3 weights as a prefix initialization if desired.
  - The 22 new features are appended; their projection layer is freshly trained.

Why
---
V3 had only 11 multi-TF ctx features (D1×2, H1×1, M15×1, micro/swing/session).
The 2026-Q2 audit found this is sparse vs the 84 features available in
canonical_features_v2. By adding the same 22 explicit H1/H4/D1/M15 indicators
as V10 v2 (signal_bridge_v2), the exit transformer gets symmetric multi-TF
context for the trade-decision moment.
"""

from __future__ import annotations

from typing import Any, List

from gx1.exits.contracts.exit_io_v1_ctx36 import compute_feature_names_hash
from gx1.exits.contracts.exit_io_v3_ctx36_m1l512_phase5 import (
    EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES,
    compute_m5_phase_index,
    compute_m5_phase_onehot,
)


EXIT_IO_V4_CTX_EXTENDED_M1L512_IO_VERSION = "EXIT_IO_V4_CTX_EXTENDED_M1L512"
EXIT_IO_V4_CTX_EXTENDED_M1L512_DEFAULT_WINDOW_LEN = 512

# Same 22 extension features as signal_bridge_v2's CTX_CONT V2 extension.
# Order MUST match signal_bridge_v2.ORDERED_CTX_CONT_V2_EXTENSION for cross-stack consistency.
EXIT_IO_V4_CTX_EXTENSION_FEATURES: List[str] = [
    # H1 (6)
    "_v1h1_ema_diff",
    "_v1h1_atr",
    "_v1h1_rsi14_z",
    "_v1h1_slope3",
    "_v1h1_slope5",
    "_v1h1_vwap_drift",
    # H4 (5)
    "_v1h4_ema_diff",
    "_v1h4_atr",
    "_v1h4_rsi14_z",
    "_v1h4_slope3",
    "_v1h4_slope5",
    # D1 (6, from canonical_v2)
    "d1_atr14_canon_v2",
    "d1_rsi14_canon_v2",
    "d1_ema_slope_20_canon_v2",
    "d1_range_z_20_canon_v2",
    "d1_close_pct_in_20day_range_canon_v2",
    "d1_pct_change_5_canon_v2",
    # M15 (5, from canonical_v2)
    "m15_atr14_canon_v2",
    "m15_rsi14_canon_v2",
    "m15_ema_slope_5_canon_v2",
    "m15_range_z_20_canon_v2",
    "m15_trend_sign_canon_v2",
]

EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES: List[str] = (
    list(EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES)
    + list(EXIT_IO_V4_CTX_EXTENSION_FEATURES)
)
EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_COUNT = len(EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES)
EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_TO_INDEX = {
    name: idx for idx, name in enumerate(EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES)
}
EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_NAMES_HASH = compute_feature_names_hash(
    EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES
)


def assert_exit_io_v4_ctx_extended_m1l512_contract() -> None:
    if (
        len(EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES)
        != EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_COUNT
    ):
        raise RuntimeError(
            f"[EXIT_IO_V4_CONTRACT] len mismatch: "
            f"{len(EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES)} != "
            f"{EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_COUNT}"
        )
    # V3 prefix invariance: first 58 features must be IDENTICAL to V3 PHASE5
    for idx, v3_name in enumerate(EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES):
        if EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES[idx] != v3_name:
            raise RuntimeError(
                f"[EXIT_IO_V4_CONTRACT] V3 prefix drift at index {idx}: "
                f"{EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES[idx]} != {v3_name}"
            )
    got = compute_feature_names_hash(EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES)
    if got != EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_IO_V4_CONTRACT] hash mismatch: "
            f"got={got} expected={EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_NAMES_HASH}"
        )


# Re-export V3's m5_phase helpers for convenience (V4 reuses identical phase semantics)
__all__ = [
    "EXIT_IO_V4_CTX_EXTENDED_M1L512_IO_VERSION",
    "EXIT_IO_V4_CTX_EXTENDED_M1L512_DEFAULT_WINDOW_LEN",
    "EXIT_IO_V4_CTX_EXTENSION_FEATURES",
    "EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES",
    "EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_COUNT",
    "EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_TO_INDEX",
    "EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_NAMES_HASH",
    "assert_exit_io_v4_ctx_extended_m1l512_contract",
    "compute_m5_phase_index",
    "compute_m5_phase_onehot",
]


assert_exit_io_v4_ctx_extended_m1l512_contract()
