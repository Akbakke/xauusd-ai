"""
EXIT_IO_V6_CTX_V3CANONICAL_M1L512 contract.

Extension of EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512 (89 features) for the
canonical_v3 stack rebuild. Adds the 5 new features that exist in canonical_v3
but were unused by every layer prior to this contract:

  Δ vs V5:
    + hour_sin, hour_cos          (cyclic time-of-day from canonical_v3)
    + dow_sin, dow_cos            (cyclic day-of-week from canonical_v3)
    + smc_premium_state           (SMC × swing-state interaction from canonical_v3)

Total: 94 features per M1 bar (vs 89 in V5), × 512 M1 window.

Backward-compat:
  - First 89 features identical to V5 contract — V6 transformer can prefix-init
    from V5 weights for warm-start.
  - V3/V4 prefixes preserved via V5 chain.
"""
from __future__ import annotations

from typing import List

from gx1.exits.contracts.exit_io_v1_ctx36 import compute_feature_names_hash
from gx1.exits.contracts.exit_io_v5_ctx_extended_smc_m1l512 import (
    EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES,
    compute_m5_phase_index,
    compute_m5_phase_onehot,
)


EXIT_IO_V6_CTX_V3CANONICAL_M1L512_IO_VERSION = "EXIT_IO_V6_CTX_V3CANONICAL_M1L512"
EXIT_IO_V6_CTX_V3CANONICAL_M1L512_DEFAULT_WINDOW_LEN = 512

# 3 features that V5 inherited from canonical_v2 but are PRUNED in canonical_v3.
# Including them in V6 would zero-pad them in the V3 v7 dataset (silent fallback);
# better to drop them so the contract is rent and matches canonical_v3's schema.
EXIT_IO_V6_DROPPED_FROM_V5: List[str] = [
    "m15_atr14_canon_v2",
    "m15_ema_slope_5_canon_v2",
    "_v1h1_vwap_drift",
]

# 5 new features added in canonical_v3 (cyclic time + SMC × swing interaction).
EXIT_IO_V6_V3_EXTENSION_FEATURES: List[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "smc_premium_state",
]

# V6 = (V5 minus 3 v3-pruned) + 5 v3-new = 89 - 3 + 5 = 91 features.
EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES: List[str] = (
    [f for f in EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES if f not in EXIT_IO_V6_DROPPED_FROM_V5]
    + list(EXIT_IO_V6_V3_EXTENSION_FEATURES)
)
EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT = len(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES)
EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_TO_INDEX = {
    name: idx for idx, name in enumerate(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES)
}
EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_NAMES_HASH = compute_feature_names_hash(
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES
)


def assert_exit_io_v6_ctx_v3canonical_m1l512_contract() -> None:
    if (
        len(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES)
        != EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT
    ):
        raise RuntimeError(
            f"[EXIT_IO_V6_CONTRACT] len mismatch: "
            f"{len(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES)} != "
            f"{EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT}"
        )
    # No strict V5 prefix invariance: V6 drops 3 features that V5 inherited from
    # canonical_v2 but were pruned in canonical_v3. Verify the 3 dropped features
    # are in fact absent + the 5 new features are present.
    for dropped in EXIT_IO_V6_DROPPED_FROM_V5:
        if dropped in EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES:
            raise RuntimeError(
                f"[EXIT_IO_V6_CONTRACT] dropped feature still present: {dropped}"
            )
    for added in EXIT_IO_V6_V3_EXTENSION_FEATURES:
        if added not in EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES:
            raise RuntimeError(
                f"[EXIT_IO_V6_CONTRACT] new v3-feature missing: {added}"
            )
    got = compute_feature_names_hash(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES)
    if got != EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_IO_V6_CONTRACT] hash mismatch: "
            f"got={got} expected={EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_NAMES_HASH}"
        )


__all__ = [
    "EXIT_IO_V6_CTX_V3CANONICAL_M1L512_IO_VERSION",
    "EXIT_IO_V6_CTX_V3CANONICAL_M1L512_DEFAULT_WINDOW_LEN",
    "EXIT_IO_V6_V3_EXTENSION_FEATURES",
    "EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES",
    "EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_COUNT",
    "EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_TO_INDEX",
    "EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_NAMES_HASH",
    "assert_exit_io_v6_ctx_v3canonical_m1l512_contract",
    "compute_m5_phase_index",
    "compute_m5_phase_onehot",
]


assert_exit_io_v6_ctx_v3canonical_m1l512_contract()
