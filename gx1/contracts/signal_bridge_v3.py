"""
XGB_SIGNAL_BRIDGE_V3 contract (SSoT for V10 v3 entry transformer + XGB v5).

Built on signal_bridge_v2, adapted for canonical_v3:

  Δ vs v2 — ctx_cont (V10 macro context):
    DROP from v2 (3 features pruned in canonical_v3):
      - m15_atr14_canon_v2          (kept atr50 in canonical_v3)
      - m15_ema_slope_5_canon_v2    (kept ema20_slope in canonical_v3)
      - _v1h1_vwap_drift            (kept _v1h1_ema_diff in canonical_v3)
    ADD (5 new features in canonical_v3):
      - hour_sin, hour_cos          (cyclic time-of-day)
      - dow_sin, dow_cos            (cyclic day-of-week)
      - smc_premium_state           (SMC × swing-state interaction)
    NET: 43 → 45 ctx_cont features

  Δ vs v2 — per-bar PRICE_STATE (V10 seq + snap input):
    DROP `atr` (pruned in v3 — duplicate of `_v1_atr14`)
    ADD `_v1_atr14` (already canonical name; replaces `atr` cleanly)
    NET: 30 → 30 (same dim, one substitution)

  Δ vs v2 — ctx_cat: UNCHANGED (6)
  Δ vs v2 — anchor (p_long/p_short/p_flat at indices 0..2): UNCHANGED
  Δ vs v2 — bridge (7 fields): UNCHANGED

Backward-compat with v2:
  - Indices 0..6 of ORDERED_SEQ_FIELDS_V3 (the XGB-bridge fields) are IDENTICAL
    to v2's ORDERED_BRIDGE_FIELDS_V2, so V10's anchor logic continues to work.
  - ctx_cont prefix (v1's 21 features) is unchanged — partial-load of v1
    bundles still works.

Hard rules (unchanged from v1/v2):
  - Fixed, explicit ordered schema with fixed dimension.
  - No fallback: in TRUTH/SMOKE, any mismatch must hard-fail.
  - Independent of raw feature universes.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


SIGNAL_BRIDGE_ID_V3 = "XGB_SIGNAL_BRIDGE_V3"


# ---------------------------------------------------------------------------
# XGB-bridge fields (UNCHANGED from v2)
# ---------------------------------------------------------------------------

XGB_PROB_FIELDS_ORDERED_V3: List[str] = [
    "p_long",
    "p_short",
    "p_flat",
]

ORDERED_BRIDGE_FIELDS_V3: List[str] = [
    *XGB_PROB_FIELDS_ORDERED_V3,
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
]
BRIDGE_DIM_V3 = len(ORDERED_BRIDGE_FIELDS_V3)  # 7

ANCHOR_FIELDS_V3 = ("p_long", "p_short", "p_flat")
ANCHOR_INDICES_V3 = tuple(ORDERED_BRIDGE_FIELDS_V3.index(f) for f in ANCHOR_FIELDS_V3)
assert ANCHOR_INDICES_V3 == (0, 1, 2), "anchor indices must remain (0,1,2)"


# ---------------------------------------------------------------------------
# Per-bar PRICE-STATE features (option β — substitution: `atr` → `_v1_atr14`)
# ---------------------------------------------------------------------------
# 30 features per M5 bar. Identical to v2 except `atr` is replaced with
# `_v1_atr14` (the v3-pruning kept `_v1_atr14` and dropped the redundant `atr`).
PER_BAR_PRICE_STATE_FIELDS_V3: List[str] = [
    # Volatility / momentum / returns (10) — `atr` dropped, `_v1_atr14` substituted
    "_v1_atr14",                      # M5 ATR(14) — was `atr` in v2
    "atr_z",                          # ATR z-score over 50 bars
    "ret_1",                          # last bar return (bps)
    "ret_5",                          # 5-bar return (bps)
    "ret_20",                         # 20-bar return
    "rvol_20",                        # 20-bar realized vol (bps)
    "body_pct",                       # body / bar-range
    "wick_asym",                      # (upper - lower) / wick-total
    "ema20_slope",                    # EMA(20) slope
    "pos_vs_ema200",                  # (close - ema200) / ema200 (bps)
    # _v1 family (10 — `_v1_atr14` moved up, replaced here by `_v1_pk_sigma20`)
    "_v1_pk_sigma20",                 # Parkinson sigma — substitute slot
    "_v1_ema_diff",                   # canonical _v1 EMA diff
    "_v1_close_ema_slope_3",          # short-term momentum
    "_v1_clv",                        # close location value
    "_v1_range_z",                    # range z-score
    "_v1_kama_slope_30",              # KAMA slope
    "_v1_tema_slope_20",              # TEMA slope
    "_v1_bb_squeeze_20_2",            # Bollinger squeeze
    "_v1_bb_bandwidth_delta_10",      # BB bandwidth change
    "_v1_body_share_1",               # body share
    "_v1_kurt_r",                     # return kurtosis
    # SMC features (9 — UNCHANGED from v2)
    "smc_swing_state",
    "smc_bos_up",
    "smc_bos_down",
    "smc_choch",
    "smc_sweep_up",
    "smc_sweep_down",
    "smc_sweep_size_atr",
    "smc_bars_since_sweep",
    "smc_premium_discount",
]
PRICE_STATE_DIM_V3 = len(PER_BAR_PRICE_STATE_FIELDS_V3)  # 30


# ---------------------------------------------------------------------------
# SEQ + SNAP fields
# ---------------------------------------------------------------------------
ORDERED_SEQ_FIELDS_V3: List[str] = ORDERED_BRIDGE_FIELDS_V3 + PER_BAR_PRICE_STATE_FIELDS_V3
SEQ_SIGNAL_DIM_V3 = len(ORDERED_SEQ_FIELDS_V3)  # 37 (same as v2)

ORDERED_SNAP_FIELDS_V3: List[str] = list(ORDERED_SEQ_FIELDS_V3)
SNAP_SIGNAL_DIM_V3 = len(ORDERED_SNAP_FIELDS_V3)  # 37

DEFAULT_SEQ_LEN_V3 = 96  # unchanged from v2


# ---------------------------------------------------------------------------
# CTX_CONT fields (option α — drop 3 v2 features pruned in canonical_v3, add 5 new)
# ---------------------------------------------------------------------------
# v1 prefix (21 features) UNCHANGED — preserves prefix-load compat
ORDERED_CTX_CONT_V1_PREFIX: List[str] = [
    "atr_bps",
    "spread_bps",
    "D1_dist_from_ema200_atr",
    "H1_range_compression_ratio",
    "D1_atr_percentile_252",
    "M15_range_compression_ratio",
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
    "wick_ratio",
    "distance_ema_fast",
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "retracement_from_last_impulse",
    "is_ASIA",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "session_change_flag",
    "session_tradable",
]

# v2 extension reduced — drop 3 features pruned in canonical_v3
ORDERED_CTX_CONT_V2_EXTENSION_RETAINED: List[str] = [
    # H1 (5 retained — _v1h1_vwap_drift dropped)
    "_v1h1_ema_diff",
    "_v1h1_atr",
    "_v1h1_rsi14_z",
    "_v1h1_slope3",
    "_v1h1_slope5",
    # H4 (5, UNCHANGED)
    "_v1h4_ema_diff",
    "_v1h4_atr",
    "_v1h4_rsi14_z",
    "_v1h4_slope3",
    "_v1h4_slope5",
    # D1 (6, UNCHANGED)
    "d1_atr14_canon_v2",
    "d1_rsi14_canon_v2",
    "d1_ema_slope_20_canon_v2",
    "d1_range_z_20_canon_v2",
    "d1_close_pct_in_20day_range_canon_v2",
    "d1_pct_change_5_canon_v2",
    # M15 (3 retained — m15_atr14_canon_v2 + m15_ema_slope_5_canon_v2 dropped)
    "m15_rsi14_canon_v2",
    "m15_range_z_20_canon_v2",
    "m15_trend_sign_canon_v2",
]

# v3 NEW additions (5 features new in canonical_v3)
ORDERED_CTX_CONT_V3_EXTENSION: List[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "smc_premium_state",
]

ORDERED_CTX_CONT_NAMES_V3: List[str] = (
    ORDERED_CTX_CONT_V1_PREFIX
    + ORDERED_CTX_CONT_V2_EXTENSION_RETAINED
    + ORDERED_CTX_CONT_V3_EXTENSION
)
CTX_CONT_DIM_V3 = len(ORDERED_CTX_CONT_NAMES_V3)  # 21 + 19 + 5 = 45


# ---------------------------------------------------------------------------
# CTX_CAT fields (UNCHANGED from v2)
# ---------------------------------------------------------------------------
ORDERED_CTX_CAT_NAMES_V3: List[str] = [
    "session_id",
    "trend_regime_id",
    "vol_regime_id",
    "atr_bucket",
    "spread_bucket",
    "H4_trend_sign_cat",
]
CTX_CAT_DIM_V3 = len(ORDERED_CTX_CAT_NAMES_V3)  # 6


# ---------------------------------------------------------------------------
# Schema hashes
# ---------------------------------------------------------------------------
def _hash_fields(fields: List[str]) -> str:
    return hashlib.sha256(("|".join(fields)).encode("utf-8")).hexdigest()


SEQ_CONTRACT_SHA256_V3 = _hash_fields(ORDERED_SEQ_FIELDS_V3)
SNAP_CONTRACT_SHA256_V3 = _hash_fields(ORDERED_SNAP_FIELDS_V3)
CTX_CONT_CONTRACT_SHA256_V3 = _hash_fields(ORDERED_CTX_CONT_NAMES_V3)
CTX_CAT_CONTRACT_SHA256_V3 = _hash_fields(ORDERED_CTX_CAT_NAMES_V3)

CONTRACT_SHA256_V3 = hashlib.sha256(
    "|".join([
        SEQ_CONTRACT_SHA256_V3,
        SNAP_CONTRACT_SHA256_V3,
        CTX_CONT_CONTRACT_SHA256_V3,
        CTX_CAT_CONTRACT_SHA256_V3,
    ]).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class SignalBridgeV3Contract:
    seq_fields: tuple[str, ...]
    snap_fields: tuple[str, ...]
    ctx_cont_names: tuple[str, ...]
    ctx_cat_names: tuple[str, ...]
    seq_dim: int
    snap_dim: int
    ctx_cont_dim: int
    ctx_cat_dim: int
    default_seq_len: int
    bridge_dim: int
    anchor_indices: tuple[int, ...]
    schema_hash: str


CONTRACT_V3 = SignalBridgeV3Contract(
    seq_fields=tuple(ORDERED_SEQ_FIELDS_V3),
    snap_fields=tuple(ORDERED_SNAP_FIELDS_V3),
    ctx_cont_names=tuple(ORDERED_CTX_CONT_NAMES_V3),
    ctx_cat_names=tuple(ORDERED_CTX_CAT_NAMES_V3),
    seq_dim=SEQ_SIGNAL_DIM_V3,
    snap_dim=SNAP_SIGNAL_DIM_V3,
    ctx_cont_dim=CTX_CONT_DIM_V3,
    ctx_cat_dim=CTX_CAT_DIM_V3,
    default_seq_len=DEFAULT_SEQ_LEN_V3,
    bridge_dim=BRIDGE_DIM_V3,
    anchor_indices=ANCHOR_INDICES_V3,
    schema_hash=CONTRACT_SHA256_V3,
)


def assert_signal_bridge_v3_contract() -> None:
    """Hard-fail check that the v3 contract is internally consistent."""
    if SEQ_SIGNAL_DIM_V3 != len(ORDERED_SEQ_FIELDS_V3):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_V3] seq dim mismatch: {SEQ_SIGNAL_DIM_V3} != {len(ORDERED_SEQ_FIELDS_V3)}"
        )
    if SNAP_SIGNAL_DIM_V3 != len(ORDERED_SNAP_FIELDS_V3):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_V3] snap dim mismatch: {SNAP_SIGNAL_DIM_V3} != {len(ORDERED_SNAP_FIELDS_V3)}"
        )
    if CTX_CONT_DIM_V3 != len(ORDERED_CTX_CONT_NAMES_V3):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_V3] ctx_cont dim mismatch: {CTX_CONT_DIM_V3} != {len(ORDERED_CTX_CONT_NAMES_V3)}"
        )
    if CTX_CAT_DIM_V3 != len(ORDERED_CTX_CAT_NAMES_V3):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_V3] ctx_cat dim mismatch: {CTX_CAT_DIM_V3} != {len(ORDERED_CTX_CAT_NAMES_V3)}"
        )
    for expected_idx, name in enumerate(ANCHOR_FIELDS_V3):
        actual = ORDERED_SEQ_FIELDS_V3.index(name)
        if actual != expected_idx:
            raise RuntimeError(
                f"[SIGNAL_BRIDGE_V3] anchor index drift: {name} at {actual} (expected {expected_idx})"
            )
    if ORDERED_SEQ_FIELDS_V3 != ORDERED_SNAP_FIELDS_V3:
        raise RuntimeError("[SIGNAL_BRIDGE_V3] seq/snap field lists must match")


assert_signal_bridge_v3_contract()


# ---------------------------------------------------------------------------
# v2-compatible aliases — same shape so existing call-sites can swap import path
# ---------------------------------------------------------------------------
SIGNAL_BRIDGE_ID = SIGNAL_BRIDGE_ID_V3
ORDERED_FIELDS = ORDERED_SEQ_FIELDS_V3  # 37 fields per bar
SEQ_SIGNAL_DIM = SEQ_SIGNAL_DIM_V3  # 37
SNAP_SIGNAL_DIM = SNAP_SIGNAL_DIM_V3  # 37
CONTRACT_SHA256 = CONTRACT_SHA256_V3

ORDERED_CTX_CONT_NAMES_EXTENDED = ORDERED_CTX_CONT_NAMES_V3  # 45
ORDERED_CTX_CAT_NAMES_EXTENDED = ORDERED_CTX_CAT_NAMES_V3  # 6
N_CTX_CONT_EXTENDED = CTX_CONT_DIM_V3  # 45
N_CTX_CAT_EXTENDED = CTX_CAT_DIM_V3  # 6

ORDERED_CTX_CONT_NAMES_BASELINE = ORDERED_CTX_CONT_V1_PREFIX[:2]
N_CTX_CONT_BASELINE = len(ORDERED_CTX_CONT_NAMES_BASELINE)
ORDERED_CTX_CAT_NAMES_BASELINE = ORDERED_CTX_CAT_NAMES_V3[:5]
N_CTX_CAT_BASELINE = len(ORDERED_CTX_CAT_NAMES_BASELINE)

CTX_CONT_COL_D1_DIST = "D1_dist_from_ema200_atr"
CTX_CONT_COL_H1_COMP = "H1_range_compression_ratio"
CTX_CONT_COL_D1_ATR_PCTL252 = "D1_atr_percentile_252"
CTX_CONT_COL_M15_COMP = "M15_range_compression_ratio"
CTX_CAT_COL_H4_TREND_SIGN = "H4_trend_sign_cat"

ALLOWED_CTX_CONT_DIMS = tuple(range(CTX_CONT_DIM_V3, 65))
ALLOWED_CTX_CAT_DIMS = (CTX_CAT_DIM_V3,)


@dataclass(frozen=True)
class SignalBridgeContract:
    bridge_id: str
    ordered_fields: Tuple[str, ...]
    seq_dim: int
    snap_dim: int
    sha256: str


CONTRACT = SignalBridgeContract(
    bridge_id=SIGNAL_BRIDGE_ID,
    ordered_fields=tuple(ORDERED_FIELDS),
    seq_dim=int(SEQ_SIGNAL_DIM),
    snap_dim=int(SNAP_SIGNAL_DIM),
    sha256=str(CONTRACT_SHA256),
)


def _is_truth_or_smoke() -> bool:
    mode = os.getenv("GX1_RUN_MODE", "").upper()
    return os.getenv("GX1_TRUTH_MODE", "0") == "1" or mode in {"TRUTH", "SMOKE"}


def validate_seq_signal(seq_x: np.ndarray, *, context: str = "unknown") -> None:
    if seq_x is None:
        raise RuntimeError(f"[SIGNAL_BRIDGE_FAIL_V3] seq_x is None (context={context})")
    arr = np.asarray(seq_x)
    if _is_truth_or_smoke() and arr.dtype not in (np.float32, np.float64):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] seq_x invalid dtype={arr.dtype} (context={context})"
        )
    if arr.ndim != 3:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] seq_x.ndim mismatch: expected=3 got={arr.ndim} "
            f"shape={getattr(arr, 'shape', None)} (context={context})"
        )
    if int(arr.shape[-1]) != int(SEQ_SIGNAL_DIM):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] seq_x feature dim mismatch: expected={SEQ_SIGNAL_DIM} "
            f"got={int(arr.shape[-1])} (context={context})"
        )
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] seq_x non-finite count={n_bad} (context={context})"
        )


def validate_snap_signal(snap_x: np.ndarray, *, context: str = "unknown") -> None:
    if snap_x is None:
        raise RuntimeError(f"[SIGNAL_BRIDGE_FAIL_V3] snap_x is None (context={context})")
    arr = np.asarray(snap_x)
    if _is_truth_or_smoke() and arr.dtype not in (np.float32, np.float64):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] snap_x invalid dtype={arr.dtype} (context={context})"
        )
    if arr.ndim != 2:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] snap_x.ndim mismatch: expected=2 got={arr.ndim} "
            f"shape={getattr(arr, 'shape', None)} (context={context})"
        )
    if int(arr.shape[-1]) != int(SNAP_SIGNAL_DIM):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] snap_x feature dim mismatch: expected={SNAP_SIGNAL_DIM} "
            f"got={int(arr.shape[-1])} (context={context})"
        )
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] snap_x non-finite count={n_bad} (context={context})"
        )


def validate_contract_in_truth() -> None:
    if not _is_truth_or_smoke():
        return
    if not ORDERED_FIELDS or len(set(ORDERED_FIELDS)) != len(ORDERED_FIELDS):
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL_V3] ORDERED_FIELDS invalid")
    if SEQ_SIGNAL_DIM <= 0 or SNAP_SIGNAL_DIM <= 0:
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL_V3] signal dims invalid")
    if CONTRACT.sha256 != CONTRACT_SHA256:
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL_V3] contract sha mismatch")
    if not ORDERED_CTX_CONT_NAMES_EXTENDED or len(set(ORDERED_CTX_CONT_NAMES_EXTENDED)) != len(ORDERED_CTX_CONT_NAMES_EXTENDED):
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL_V3] ORDERED_CTX_CONT_NAMES_EXTENDED invalid")
    if not ORDERED_CTX_CAT_NAMES_EXTENDED or len(set(ORDERED_CTX_CAT_NAMES_EXTENDED)) != len(ORDERED_CTX_CAT_NAMES_EXTENDED):
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL_V3] ORDERED_CTX_CAT_NAMES_EXTENDED invalid")


def get_canonical_ctx_contract() -> Dict[str, object]:
    return {
        "ctx_cont_dim": int(N_CTX_CONT_EXTENDED),
        "ctx_cat_dim": int(N_CTX_CAT_EXTENDED),
        "ctx_cont_names": list(ORDERED_CTX_CONT_NAMES_EXTENDED),
        "ctx_cat_names": list(ORDERED_CTX_CAT_NAMES_EXTENDED),
        "tag": "CTX45CAT6_V3",
        "source": "signal_bridge_v3_full_contract",
        "ctx_cont_rule": "bundle-driven ctx_cont_dim must equal 45 and match the v3 ordered list",
        "ctx_cat_rule": "bundle-driven ctx_cat_dim must equal 6 and match the v3 ordered list",
    }


def validate_bundle_ctx_contract_in_strict(
    expected_ctx_cont_dim: int,
    expected_ctx_cat_dim: int,
    ordered_ctx_cont_names: Sequence[str],
    ordered_ctx_cat_names: Sequence[str],
    *,
    context: str = "bundle_meta",
) -> None:
    if not _is_truth_or_smoke():
        return
    if expected_ctx_cont_dim != N_CTX_CONT_EXTENDED:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] {context} expected_ctx_cont_dim={expected_ctx_cont_dim} "
            f"!= v3 contract dim={N_CTX_CONT_EXTENDED}"
        )
    if expected_ctx_cat_dim != N_CTX_CAT_EXTENDED:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] {context} expected_ctx_cat_dim={expected_ctx_cat_dim} "
            f"!= v3 contract dim={N_CTX_CAT_EXTENDED}"
        )
    meta_cont = list(ordered_ctx_cont_names)[:expected_ctx_cont_dim]
    contract_cont = list(ORDERED_CTX_CONT_NAMES_EXTENDED)
    if meta_cont != contract_cont:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] bundle {context} ordered_ctx_cont_names mismatch"
        )
    meta_cat = list(ordered_ctx_cat_names)[:expected_ctx_cat_dim]
    contract_cat = list(ORDERED_CTX_CAT_NAMES_EXTENDED)
    if meta_cat != contract_cat:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V3] bundle {context} ordered_ctx_cat_names mismatch"
        )
