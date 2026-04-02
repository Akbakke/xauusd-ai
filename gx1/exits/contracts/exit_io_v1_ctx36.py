"""
EXIT_IO_V1_CTX36 contract (SSoT for transformer exit IO).
"""

from __future__ import annotations

from hashlib import sha256
from typing import List, Sequence

from gx1.exits.contracts.exit_io_v1_ctx36_features import (
    EXIT_IO_V1_CTX36_FEATURES,
    EXIT_IO_V1_CTX36_FEATURE_COUNT,
    EXIT_IO_V1_CTX36_FEATURE_TO_INDEX,
)

EXIT_IO_V1_CTX36_IO_VERSION = "EXIT_IO_V1_CTX36"
EXIT_IO_V1_CTX36_FEATURE_NAMES_HASH = "1be2fe5a6150e8c1"


def compute_feature_names_hash(names: Sequence[str]) -> str:
    payload = "\n".join(names).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def assert_exit_io_v1_ctx36_contract() -> None:
    if len(EXIT_IO_V1_CTX36_FEATURES) != EXIT_IO_V1_CTX36_FEATURE_COUNT:
        raise RuntimeError(
            f"[EXIT_CONTRACT] len mismatch: {len(EXIT_IO_V1_CTX36_FEATURES)} != {EXIT_IO_V1_CTX36_FEATURE_COUNT}"
        )
    h = compute_feature_names_hash(EXIT_IO_V1_CTX36_FEATURES)
    if h != EXIT_IO_V1_CTX36_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_CONTRACT] hash mismatch: got={h} expected={EXIT_IO_V1_CTX36_FEATURE_NAMES_HASH}"
        )


def required_exit_columns_v1_ctx36() -> List[str]:
    return list(EXIT_IO_V1_CTX36_FEATURES)


def exit_feature_index_v1_ctx36(name: str) -> int:
    try:
        return int(EXIT_IO_V1_CTX36_FEATURE_TO_INDEX[name])
    except KeyError as exc:
        raise KeyError(f"[EXIT_CONTRACT] unknown EXIT_IO_V1_CTX36 feature: {name}") from exc


assert_exit_io_v1_ctx36_contract()
