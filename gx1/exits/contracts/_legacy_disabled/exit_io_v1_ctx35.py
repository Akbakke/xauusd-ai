"""
EXIT_IO_V1_CTX35 contract (SSoT for transformer exit IO).
"""

from __future__ import annotations

from hashlib import sha256
from typing import List, Sequence

from gx1.exits.contracts.exit_io_v1_ctx35_features import (
    EXIT_IO_V1_CTX35_FEATURES,
    EXIT_IO_V1_CTX35_FEATURE_COUNT,
)

EXIT_IO_V1_CTX35_IO_VERSION = "EXIT_IO_V1_CTX35"
EXIT_IO_V1_CTX35_FEATURE_NAMES_HASH = "e948b1cb34bbac64"


def compute_feature_names_hash(names: Sequence[str]) -> str:
    payload = "\n".join(names).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def assert_exit_io_v1_ctx35_contract() -> None:
    if len(EXIT_IO_V1_CTX35_FEATURES) != EXIT_IO_V1_CTX35_FEATURE_COUNT:
        raise RuntimeError(
            f"[EXIT_CONTRACT] len mismatch: {len(EXIT_IO_V1_CTX35_FEATURES)} != {EXIT_IO_V1_CTX35_FEATURE_COUNT}"
        )
    h = compute_feature_names_hash(EXIT_IO_V1_CTX35_FEATURES)
    if h != EXIT_IO_V1_CTX35_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_CONTRACT] hash mismatch: got={h} expected={EXIT_IO_V1_CTX35_FEATURE_NAMES_HASH}"
        )


def required_exit_columns_v1_ctx35() -> List[str]:
    return list(EXIT_IO_V1_CTX35_FEATURES)


assert_exit_io_v1_ctx35_contract()
