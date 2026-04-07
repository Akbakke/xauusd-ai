"""
EXIT_IO_V2_CTX36_M1L512 contract.

V2 keeps the same 53 feature columns as V1 but changes the temporal contract
to a longer M1 lookback window of 512 bars.
"""

from __future__ import annotations

from gx1.exits.contracts.exit_io_v1_ctx36 import compute_feature_names_hash
from gx1.exits.contracts.exit_io_v1_ctx36_features import (
    EXIT_IO_V1_CTX36_FEATURES,
    EXIT_IO_V1_CTX36_FEATURE_COUNT,
    EXIT_IO_V1_CTX36_FEATURE_TO_INDEX,
)

EXIT_IO_V2_CTX36_M1L512_IO_VERSION = "EXIT_IO_V2_CTX36_M1L512"
EXIT_IO_V2_CTX36_M1L512_DEFAULT_WINDOW_LEN = 512
EXIT_IO_V2_CTX36_M1L512_FEATURES = list(EXIT_IO_V1_CTX36_FEATURES)
EXIT_IO_V2_CTX36_M1L512_FEATURE_COUNT = int(EXIT_IO_V1_CTX36_FEATURE_COUNT)
EXIT_IO_V2_CTX36_M1L512_FEATURE_TO_INDEX = dict(EXIT_IO_V1_CTX36_FEATURE_TO_INDEX)
EXIT_IO_V2_CTX36_M1L512_FEATURE_NAMES_HASH = compute_feature_names_hash(EXIT_IO_V2_CTX36_M1L512_FEATURES)


def assert_exit_io_v2_ctx36_m1l512_contract() -> None:
    if len(EXIT_IO_V2_CTX36_M1L512_FEATURES) != EXIT_IO_V2_CTX36_M1L512_FEATURE_COUNT:
        raise RuntimeError(
            f"[EXIT_CONTRACT] len mismatch: {len(EXIT_IO_V2_CTX36_M1L512_FEATURES)} != {EXIT_IO_V2_CTX36_M1L512_FEATURE_COUNT}"
        )
    h = compute_feature_names_hash(EXIT_IO_V2_CTX36_M1L512_FEATURES)
    if h != EXIT_IO_V2_CTX36_M1L512_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_CONTRACT] hash mismatch: got={h} expected={EXIT_IO_V2_CTX36_M1L512_FEATURE_NAMES_HASH}"
        )


assert_exit_io_v2_ctx36_m1l512_contract()
