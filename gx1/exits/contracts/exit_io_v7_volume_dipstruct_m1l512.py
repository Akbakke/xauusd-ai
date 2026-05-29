"""
EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512 contract.

Extension of EXIT_IO_V6_CTX_V3CANONICAL_M1L512 (91 features) for the 2026-Q2
V3↔Exit-IQL parity wave — mirrors the V10 entry-side rebuild onto the exit
transformer. Adds the three feature groups V6 lacked but the entry V10 has:

  Δ vs V6:
    + 4 volume/order-flow features (SAME helper as V10 — one truth):
        vol_z_20, vol_ratio_5_20, vol_pct_96, signed_vol_z_20
    + 24 group-A market-parity features (SAME names as the entry V10 +
        signal_bridge_v3.ORDERED_CTX_CONT_GROUP_A_PARITY — one truth):
        dist_to_{tf}_{hi,lo}_atr (top/bottom distance), daily pivots
        (R1/R2/S1/S2), vol-term, vol-percentile, session-overlap.
    + 36 dip/struct parity features (SAME names as the Entry/Exit-IQL state +
        signal_bridge_v3.ORDERED_CTX_CONT_DIP_STRUCT — one truth):
        dip_confirmed/proximity per TF (8) + struct continuation/pullback/
        bounce/depth per TF + cross-TF combos (28).
      These are decision-context features broadcast across the 512-M1 window
      (one value per sample, like V10 treats ctx) — NOT recomputed per M1 bar.

Total: 91 + 4 + 24 + 36 = 155 features per M1 bar, × 512 M1 window.

Backward-compat:
  - First 91 features identical to V6 — V7 transformer can prefix-init from V6.
  - Volume/group-A/dip-struct appended at the tail (indices 91..154).

One-truth: the volume slice == gx1.features.volume_features.VOLUME_FEATURE_NAMES;
the dip/struct slice == gx1.contracts.signal_bridge_v3.ORDERED_CTX_CONT_DIP_STRUCT.
Both asserted below.
"""
from __future__ import annotations

from typing import List

from gx1.exits.contracts.exit_io_v1_ctx36 import compute_feature_names_hash
from gx1.exits.contracts.exit_io_v6_ctx_v3canonical_m1l512 import (
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_DEFAULT_WINDOW_LEN,
    compute_m5_phase_index,
    compute_m5_phase_onehot,
)
from gx1.features.volume_features import VOLUME_FEATURE_NAMES
from gx1.contracts.signal_bridge_v3 import (
    ORDERED_CTX_CONT_DIP_STRUCT,
    ORDERED_CTX_CONT_GROUP_A_PARITY,
)


EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_IO_VERSION = "EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512"
EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_DEFAULT_WINDOW_LEN = 512

# Appended groups (one-truth references, not re-listed literally).
EXIT_IO_V7_VOLUME_FEATURES: List[str] = list(VOLUME_FEATURE_NAMES)          # 4
# 2026-05-26: GROUP-A market-parity (24) added to the EXIT too (full symmetry with
# the entry V10) — gives the exit transformer the SAME top/bottom + pivot vision:
# dist_to_{tf}_{hi,lo}_atr (distance to recent tops/bottoms), daily pivots
# (R1/R2/S1/S2), vol-term, vol-percentile, session-overlap. Critical for exits:
# "take profit near resistance / hold near support." Same source (augment_candidate).
EXIT_IO_V7_GROUP_A_FEATURES: List[str] = list(ORDERED_CTX_CONT_GROUP_A_PARITY)  # 24
EXIT_IO_V7_DIP_STRUCT_FEATURES: List[str] = list(ORDERED_CTX_CONT_DIP_STRUCT)  # 36

EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURES: List[str] = (
    list(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES)
    + EXIT_IO_V7_VOLUME_FEATURES
    + EXIT_IO_V7_GROUP_A_FEATURES
    + EXIT_IO_V7_DIP_STRUCT_FEATURES
)
EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURE_COUNT = len(
    EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURES
)
EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURE_TO_INDEX = {
    name: idx for idx, name in enumerate(EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURES)
}
EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURE_NAMES_HASH = compute_feature_names_hash(
    EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURES
)


def assert_exit_io_v7_volume_dipstruct_m1l512_contract() -> None:
    feats = EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURES
    if len(feats) != EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURE_COUNT:
        raise RuntimeError(
            f"[EXIT_IO_V7_CONTRACT] len mismatch: {len(feats)} != "
            f"{EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURE_COUNT}"
        )
    # V6 prefix invariance (warm-start from V6).
    v6 = EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES
    if feats[: len(v6)] != list(v6):
        raise RuntimeError("[EXIT_IO_V7_CONTRACT] V6 prefix not preserved")
    # ONE-TRUTH: tail slices must equal their canonical sources (no drift).
    # Layout: [V6] + [volume 4] + [group_a 24] + [dip_struct 36].
    n_ds = len(EXIT_IO_V7_DIP_STRUCT_FEATURES)
    n_ga = len(EXIT_IO_V7_GROUP_A_FEATURES)
    n_vol = len(EXIT_IO_V7_VOLUME_FEATURES)
    if feats[-n_ds:] != list(ORDERED_CTX_CONT_DIP_STRUCT):
        raise RuntimeError("[EXIT_IO_V7_CONTRACT] dip/struct tail drift vs ORDERED_CTX_CONT_DIP_STRUCT")
    if feats[-(n_ds + n_ga):-n_ds] != list(ORDERED_CTX_CONT_GROUP_A_PARITY):
        raise RuntimeError("[EXIT_IO_V7_CONTRACT] group-A slice drift vs ORDERED_CTX_CONT_GROUP_A_PARITY")
    if feats[-(n_ds + n_ga + n_vol):-(n_ds + n_ga)] != list(VOLUME_FEATURE_NAMES):
        raise RuntimeError("[EXIT_IO_V7_CONTRACT] volume slice drift vs VOLUME_FEATURE_NAMES")
    # no duplicate feature names
    if len(set(feats)) != len(feats):
        dupes = [f for f in feats if feats.count(f) > 1]
        raise RuntimeError(f"[EXIT_IO_V7_CONTRACT] duplicate features: {sorted(set(dupes))}")
    got = compute_feature_names_hash(feats)
    if got != EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_IO_V7_CONTRACT] hash mismatch: got={got} "
            f"expected={EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURE_NAMES_HASH}"
        )


__all__ = [
    "EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_IO_VERSION",
    "EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_DEFAULT_WINDOW_LEN",
    "EXIT_IO_V7_VOLUME_FEATURES",
    "EXIT_IO_V7_DIP_STRUCT_FEATURES",
    "EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURES",
    "EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURE_COUNT",
    "EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURE_TO_INDEX",
    "EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512_FEATURE_NAMES_HASH",
    "assert_exit_io_v7_volume_dipstruct_m1l512_contract",
    "compute_m5_phase_index",
    "compute_m5_phase_onehot",
]


assert_exit_io_v7_volume_dipstruct_m1l512_contract()
