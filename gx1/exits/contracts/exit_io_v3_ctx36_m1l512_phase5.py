"""
EXIT_IO_V3_CTX36_M1L512_PHASE5 contract.

Appends explicit M5 phase features to the existing M1L512 contract so the exit
transformer can distinguish fresh model-bar XGB signal from carried M1 bars.
"""

from __future__ import annotations

from typing import Any, List

from gx1.exits.contracts.exit_io_v1_ctx36 import compute_feature_names_hash
from gx1.exits.contracts.exit_io_v1_ctx36_features import (
    EXIT_IO_V1_CTX36_FEATURE_COUNT,
    EXIT_IO_V1_CTX36_FEATURE_TO_INDEX,
    EXIT_IO_V1_CTX36_FEATURES,
)

EXIT_IO_V3_CTX36_M1L512_PHASE5_IO_VERSION = "EXIT_IO_V3_CTX36_M1L512_PHASE5"
EXIT_IO_V3_CTX36_M1L512_PHASE5_DEFAULT_WINDOW_LEN = 512
EXIT_IO_V3_CTX36_M1L512_PHASE5_EXTRA_FEATURES: List[str] = [
    "m5_phase_0",
    "m5_phase_1",
    "m5_phase_2",
    "m5_phase_3",
    "m5_phase_4",
]
EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES = list(EXIT_IO_V1_CTX36_FEATURES) + list(
    EXIT_IO_V3_CTX36_M1L512_PHASE5_EXTRA_FEATURES
)
EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_COUNT = len(EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES)
EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_TO_INDEX = {
    name: idx for idx, name in enumerate(EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES)
}
EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_NAMES_HASH = compute_feature_names_hash(
    EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES
)


def compute_m5_phase_index(ts_val: Any) -> int:
    minute = int(getattr(ts_val, "minute", 0))
    return int(minute % 5)


def compute_m5_phase_onehot(ts_val: Any) -> List[float]:
    phase_idx = compute_m5_phase_index(ts_val)
    return [1.0 if phase_idx == i else 0.0 for i in range(5)]


def assert_exit_io_v3_ctx36_m1l512_phase5_contract() -> None:
    if len(EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES) != EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_COUNT:
        raise RuntimeError(
            f"[EXIT_CONTRACT] len mismatch: {len(EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES)} != "
            f"{EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_COUNT}"
        )
    if EXIT_IO_V1_CTX36_FEATURE_COUNT != len(EXIT_IO_V1_CTX36_FEATURES):
        raise RuntimeError("[EXIT_CONTRACT] V1 base feature count mismatch")
    for name, idx in EXIT_IO_V1_CTX36_FEATURE_TO_INDEX.items():
        if EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_TO_INDEX.get(name) != idx:
            raise RuntimeError(
                f"[EXIT_CONTRACT] V1 base feature index drift for {name}: "
                f"{EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_TO_INDEX.get(name)} != {idx}"
            )
    got = compute_feature_names_hash(EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURES)
    if got != EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_CONTRACT] hash mismatch: got={got} expected={EXIT_IO_V3_CTX36_M1L512_PHASE5_FEATURE_NAMES_HASH}"
        )


assert_exit_io_v3_ctx36_m1l512_phase5_contract()
