"""
EXIT_IO_V0_CTX19 contract (SSoT for transformer exit IO).
"""

from __future__ import annotations

from hashlib import sha256
from typing import List, Sequence

EXIT_IO_V0_CTX19_IO_VERSION = "EXIT_IO_V0_CTX19"
EXIT_IO_V0_CTX19_FEATURE_COUNT = 19
EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH = "7d63043349359419"

EXIT_IO_V0_CTX19_FEATURES: List[str] = [
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
    "p_long_entry",
    "p_hat_entry",
    "uncertainty_entry",
    "entropy_entry",
    "margin_entry",
    "pnl_bps_now",
    "mfe_bps",
    "mae_bps",
    "dd_from_mfe_bps",
    "bars_held",
    "time_since_mfe_bars",
    "atr_bps_now",
]


def compute_feature_names_hash(names: Sequence[str]) -> str:
    # MUST match how the bundle hash was produced.
    payload = "\n".join(names).encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def assert_exit_io_v0_ctx19_contract() -> None:
    if len(EXIT_IO_V0_CTX19_FEATURES) != EXIT_IO_V0_CTX19_FEATURE_COUNT:
        raise RuntimeError(
            f"[EXIT_CONTRACT] len mismatch: {len(EXIT_IO_V0_CTX19_FEATURES)} != {EXIT_IO_V0_CTX19_FEATURE_COUNT}"
        )
    h = compute_feature_names_hash(EXIT_IO_V0_CTX19_FEATURES)
    if h != EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH:
        raise RuntimeError(
            f"[EXIT_CONTRACT] hash mismatch: got={h} expected={EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH}"
        )


def required_exit_columns_v0_ctx19() -> List[str]:
    return list(EXIT_IO_V0_CTX19_FEATURES)


assert_exit_io_v0_ctx19_contract()
