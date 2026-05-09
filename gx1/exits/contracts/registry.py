from __future__ import annotations

from typing import Any, Dict, Optional

from gx1.exits.contracts.exit_io_v1_ctx36 import (
    EXIT_IO_V1_CTX36_FEATURE_NAMES_HASH,
    EXIT_IO_V1_CTX36_IO_VERSION,
    assert_exit_io_v1_ctx36_contract,
)
from gx1.exits.contracts.exit_io_v1_ctx36_features import EXIT_IO_V1_CTX36_FEATURES
from gx1.exits.contracts.exit_io_v3_ctx36_m1l512_phase5 import (
    EXIT_IO_V3_CTX36_M1L512_PHASE5_DEFAULT_WINDOW_LEN,
    EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_NAMES_HASH,
    EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES,
    EXIT_IO_V3_CTX36_M1L512_PHASE5_IO_VERSION,
    assert_exit_io_v3_ctx36_m1l512_phase5_contract,
)
from gx1.exits.contracts.exit_io_v4_ctx_extended_m1l512 import (
    EXIT_IO_V4_CTX_EXTENDED_M1L512_DEFAULT_WINDOW_LEN,
    EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_NAMES_HASH,
    EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES,
    EXIT_IO_V4_CTX_EXTENDED_M1L512_IO_VERSION,
    assert_exit_io_v4_ctx_extended_m1l512_contract,
)
from gx1.exits.contracts.exit_io_v5_ctx_extended_smc_m1l512 import (
    EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_DEFAULT_WINDOW_LEN,
    EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_NAMES_HASH,
    EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES,
    EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_IO_VERSION,
    assert_exit_io_v5_ctx_extended_smc_m1l512_contract,
)
from gx1.exits.contracts.exit_io_v6_ctx_v3canonical_m1l512 import (
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_DEFAULT_WINDOW_LEN,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_NAMES_HASH,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES,
    EXIT_IO_V6_CTX_V3CANONICAL_M1L512_IO_VERSION,
    assert_exit_io_v6_ctx_v3canonical_m1l512_contract,
)


def get_exit_io_contract(io_version: Optional[str] = None) -> Dict[str, Any]:
    resolved = (
        str(io_version or EXIT_IO_V6_CTX_V3CANONICAL_M1L512_IO_VERSION).strip()
        or EXIT_IO_V6_CTX_V3CANONICAL_M1L512_IO_VERSION
    )
    if resolved == EXIT_IO_V1_CTX36_IO_VERSION:
        assert_exit_io_v1_ctx36_contract()
        return {
            "io_version": EXIT_IO_V1_CTX36_IO_VERSION,
            "feature_names": list(EXIT_IO_V1_CTX36_FEATURES),
            "feature_count": len(EXIT_IO_V1_CTX36_FEATURES),
            "feature_hash": EXIT_IO_V1_CTX36_FEATURE_NAMES_HASH,
            "default_window_len": 8,
        }
    if resolved == EXIT_IO_V3_CTX36_M1L512_PHASE5_IO_VERSION:
        assert_exit_io_v3_ctx36_m1l512_phase5_contract()
        return {
            "io_version": EXIT_IO_V3_CTX36_M1L512_PHASE5_IO_VERSION,
            "feature_names": list(EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES),
            "feature_count": len(EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES),
            "feature_hash": EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_NAMES_HASH,
            "default_window_len": int(EXIT_IO_V3_CTX36_M1L512_PHASE5_DEFAULT_WINDOW_LEN),
        }
    if resolved == EXIT_IO_V4_CTX_EXTENDED_M1L512_IO_VERSION:
        assert_exit_io_v4_ctx_extended_m1l512_contract()
        return {
            "io_version": EXIT_IO_V4_CTX_EXTENDED_M1L512_IO_VERSION,
            "feature_names": list(EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES),
            "feature_count": len(EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURES),
            "feature_hash": EXIT_IO_V4_CTX_EXTENDED_M1L512_FEATURE_NAMES_HASH,
            "default_window_len": int(EXIT_IO_V4_CTX_EXTENDED_M1L512_DEFAULT_WINDOW_LEN),
        }
    if resolved == EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_IO_VERSION:
        assert_exit_io_v5_ctx_extended_smc_m1l512_contract()
        return {
            "io_version": EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_IO_VERSION,
            "feature_names": list(EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES),
            "feature_count": len(EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURES),
            "feature_hash": EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_FEATURE_NAMES_HASH,
            "default_window_len": int(EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512_DEFAULT_WINDOW_LEN),
        }
    if resolved == EXIT_IO_V6_CTX_V3CANONICAL_M1L512_IO_VERSION:
        assert_exit_io_v6_ctx_v3canonical_m1l512_contract()
        return {
            "io_version": EXIT_IO_V6_CTX_V3CANONICAL_M1L512_IO_VERSION,
            "feature_names": list(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES),
            "feature_count": len(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURES),
            "feature_hash": EXIT_IO_V6_CTX_V3CANONICAL_M1L512_FEATURE_NAMES_HASH,
            "default_window_len": int(EXIT_IO_V6_CTX_V3CANONICAL_M1L512_DEFAULT_WINDOW_LEN),
        }
    raise RuntimeError(f"[EXIT_CONTRACT_UNSUPPORTED] io_version={resolved}")
