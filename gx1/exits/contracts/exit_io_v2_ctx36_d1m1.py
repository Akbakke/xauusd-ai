"""
EXIT_IO_V2_CTX36_D1M1 contract.

V2 keeps the same 53 feature columns as V1 but changes the temporal contract
to a long M1 lookback window.
"""

from __future__ import annotations

from gx1.exits.contracts.exit_io_v1_ctx36 import compute_feature_names_hash
from gx1.exits.contracts.exit_io_v1_ctx36_features import (
    EXIT_IO_V1_CTX36_FEATURES,
    EXIT_IO_V1_CTX36_FEATURE_COUNT,
    EXIT_IO_V1_CTX36_FEATURE_TO_INDEX,
)

EXIT_IO_V2_CTX36_D1M1_IO_VERSION = "EXIT_IO_V2_CTX36_D1M1"
EXIT_IO_V2_CTX36_D1M1_DEFAULT_WINDOW_LEN = 1440
EXIT_IO_V2_CTX36_D1M1_FEATURES = list(EXIT_IO_V1_CTX36_FEATURES)
EXIT_IO_V2_CTX36_D1M1_FEATURE_COUNT = int(EXIT_IO_V1_CTX36_FEATURE_COUNT)
EXIT_IO_V2_CTX36_D1M1_FEATURE_TO_INDEX = dict(EXIT_IO_V1_CTX36_FEATURE_TO_INDEX)
EXIT_IO_V2_CTX36_D1M1_FEATURE_NAMES_HASH = compute_feature_names_hash(EXIT_IO_V2_CTX36_D1M1_FEATURES)


def assert_exit_io_v2_ctx36_d1m1_contract() -> None:
    if len(EXIT_IO_V2_CTX36_D1M1_FEATURES) != EXIT_IO_V2_CTX36_D1M1_FEATURE_COUNT:
        raise RuntimeError(
            f"[EXIT_CONTRACT] len mismatch: {len(EXIT_IO_V2_CTX36_D1M1_FEATURES)} != {EXIT_IO_V2_CTX36_D1M1_FEATURE_COUNT}"
        )
    h = compute_feature_names_hash(EXIT_IO_V2_CTX36_D1M1_FEATURES)
    if h != EXIT_IO_V2_CTX36_D1M1_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_CONTRACT] hash mismatch: got={h} expected={EXIT_IO_V2_CTX36_D1M1_FEATURE_NAMES_HASH}"
        )


assert_exit_io_v2_ctx36_d1m1_contract()
