"""
EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512 contract.

Extension of EXIT_IO_V4_CTX_EXTENDED_M1L512 (which itself extends V3) for the
2026-Q2 stack rebuild's Smart Money Concept (SMC) feature addition.

Deltas vs V4:
  - V3 prefix (58 features) preserved at indices 0..57
  - V2 multi-TF extension (22 features) preserved at indices 58..79
  - **NEW**: 9 SMC features at indices 80..88 (HH/HL state, BOS, CHOCH,
    liquidity sweep up/down + size + bars-since, premium/discount score)

Total: 89 features per M1 bar (vs 80 in V4, 58 in V3), × 512 M1 window.

Backward-compat:
  - First 58 features identical to V3 contract.
  - First 80 features identical to V4 contract.
  - V5 transformer can prefix-init from V4 weights.
"""
from __future__ import annotations

from typing import List

from gx1.exits.contracts.exit_io_v1_ctx36 import compute_feature_names_hash
from gx1.exits.contracts.exit_io_v4_ctx_extended_m1l512 import (
    EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES,
    compute_m5_phase_index,
    compute_m5_phase_onehot,
)
from gx1.features.smc_v1 import SMC_FEATURE_NAMES


EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_IO_VERSION = "EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512"
EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_DEFAULT_WINDOW_LEN = 512

# Reuse the SMC feature list from gx1/features/smc_v1.py — single source of truth.
EXIT_IO_V5_SMC_EXTENSION_FEATURES: List[str] = list(SMC_FEATURE_NAMES)

EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES: List[str] = (
    list(EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES)
    + list(EXIT_IO_V5_SMC_EXTENSION_FEATURES)
)
EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_COUNT = len(EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES)
EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_TO_INDEX = {
    name: idx for idx, name in enumerate(EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES)
}
EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_NAMES_HASH = compute_feature_names_hash(
    EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES
)


def assert_exit_io_v5_ctx_extended_smc_m1l512_contract() -> None:
    if (
        len(EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES)
        != EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_COUNT
    ):
        raise RuntimeError(
            f"[EXIT_IO_V5_CONTRACT] len mismatch: "
            f"{len(EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES)} != "
            f"{EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_COUNT}"
        )
    # V4 prefix invariance
    for idx, v4_name in enumerate(EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES):
        if EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES[idx] != v4_name:
            raise RuntimeError(
                f"[EXIT_IO_V5_CONTRACT] V4 prefix drift at index {idx}: "
                f"{EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES[idx]} != {v4_name}"
            )
    got = compute_feature_names_hash(EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES)
    if got != EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_IO_V5_CONTRACT] hash mismatch: "
            f"got={got} expected={EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_NAMES_HASH}"
        )


__all__ = [
    "EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_IO_VERSION",
    "EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_DEFAULT_WINDOW_LEN",
    "EXIT_IO_V5_SMC_EXTENSION_FEATURES",
    "EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES",
    "EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_COUNT",
    "EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_TO_INDEX",
    "EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_NAMES_HASH",
    "assert_exit_io_v5_ctx_extended_smc_m1l512_contract",
    "compute_m5_phase_index",
    "compute_m5_phase_onehot",
]


assert_exit_io_v5_ctx_extended_smc_m1l512_contract()
