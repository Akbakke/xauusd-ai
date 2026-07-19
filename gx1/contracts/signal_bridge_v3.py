"""
XGB_SIGNAL_BRIDGE_V3 contract (SSoT for V10 v3 entry transformer + XGB v5).

Historically derived from the retired signal_bridge_v2 (module deleted; git
history retains it), adapted for canonical_v3:

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
    ADD 4 volume/order-flow features (2026-05-26): vol_z_20, vol_ratio_5_20,
        vol_pct_96, signed_vol_z_20 — from gx1.features.volume_features.
    NET: 30 → 34 (one substitution + 4 volume); SEQ 37 → 41 (7 bridge + 34)

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
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS,
    MODEL_NATIVE_CTX_CONT_ENTRY_SMART_DERIVED_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
    MODEL_NATIVE_CTX_CONT_REGIME_FIELDS,
    MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS,
    MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS,
    MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS,
)


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
PER_BAR_PRICE_STATE_FIELDS_V3: List[str] = list(MODEL_NATIVE_BASE_FIELDS)
PRICE_STATE_DIM_V3 = len(PER_BAR_PRICE_STATE_FIELDS_V3)  # 34 (was 30; +4 volume)


# ---------------------------------------------------------------------------
# SEQ + SNAP fields
# ---------------------------------------------------------------------------
ORDERED_SEQ_FIELDS_V3: List[str] = ORDERED_BRIDGE_FIELDS_V3 + PER_BAR_PRICE_STATE_FIELDS_V3
SEQ_SIGNAL_DIM_V3 = len(ORDERED_SEQ_FIELDS_V3)  # 41 (was 37; +4 volume)

ORDERED_SNAP_FIELDS_V3: List[str] = list(ORDERED_SEQ_FIELDS_V3)
SNAP_SIGNAL_DIM_V3 = len(ORDERED_SNAP_FIELDS_V3)  # 41

DEFAULT_SEQ_LEN_V3 = 96  # unchanged from v2


# ---------------------------------------------------------------------------
# CTX_CONT fields (option α — drop 3 v2 features pruned in canonical_v3, add 5 new)
# ---------------------------------------------------------------------------
# v1 prefix (21 features) UNCHANGED — preserves prefix-load compat
ORDERED_CTX_CONT_V1_PREFIX: List[str] = list(
    MODEL_NATIVE_CTX_CONT_V1_PREFIX_FIELDS
)

# v2 extension reduced — drop 3 features pruned in canonical_v3
ORDERED_CTX_CONT_V2_EXTENSION_RETAINED: List[str] = list(
    MODEL_NATIVE_CTX_CONT_V2_EXTENSION_FIELDS
)

# v3 NEW additions (5 features new in canonical_v3)
ORDERED_CTX_CONT_V3_EXTENSION: List[str] = list(
    MODEL_NATIVE_CTX_CONT_V3_EXTENSION_FIELDS
)

# 2026-05-26 — GROUP-A market-parity extension (24). Gives V10 the SAME market/
# structure features the Entry/Exit-IQL decide on (dip-distance, pivots, vol-term,
# vol-percentile, session-overlap) so the transformer represents everything the
# IQL acts on — no asymmetry. Portfolio features (long/short_*) are deliberately
# EXCLUDED (IQL-only state, would be 0 in training → train/serve skew). Order +
# names MUST match group_a_features.GROUP_A_FEATURE_NAMES market subset.
ORDERED_CTX_CONT_GROUP_A_PARITY: List[str] = list(
    MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS
)

# DIP/STRUCT parity extension (36). Gives model-native Entry explicit
# dip/proximity and HH/HL/LH/LL continuation/pullback/bounce/depth evidence,
# plus cross-TF combinations, from augment_forward_outcome_v2._dip_struct_5tf.
ORDERED_CTX_CONT_DIP_STRUCT: List[str] = list(
    MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS
)

# 2026-06-26 — ENTRY smart-context promotion (19). These started as audit-only
# nonlinear summaries of already-active seq/ctx inputs. They are now promoted to
# ctx_cont because diagnostics found consistent edge in S/R proximity, SMC
# recency/pressure, liquidity proximity, and multi-TF dip aggregation. Computed by
# gx1.features.entry_smart_context in builder + batch/live inference.
ORDERED_CTX_CONT_ENTRY_SMART_DERIVED: List[str] = list(
    MODEL_NATIVE_CTX_CONT_ENTRY_SMART_DERIVED_FIELDS
)

# REGIME_V4 tail: 18 multi-TF regime conditioning/change-detection features.
# This is mandatory in the active contract and shares its exact owner with EXIT_IO_V8.
ORDERED_CTX_CONT_REGIME_V4: List[str] = list(MODEL_NATIVE_CTX_CONT_REGIME_FIELDS)

ORDERED_CTX_CONT_NAMES_V3: List[str] = list(MODEL_NATIVE_CTX_CONT_FIELDS)
CTX_CONT_DIM_V3 = len(ORDERED_CTX_CONT_NAMES_V3)  # exact active contract: 142


# ---------------------------------------------------------------------------
# CTX_CAT fields (UNCHANGED from v2)
# ---------------------------------------------------------------------------
_CTX_CAT_ALL_V3: List[str] = [
    "session_id",
    "trend_regime_id",
    *MODEL_NATIVE_CTX_CAT_FIELDS[1:],
]
# Phase 0a/R4 (2026-06-04, audit + user vedtak): when REGIME_V4 is ON, DROP the trend_regime_id
# categorical. It was degenerate (const=1 on the price_vs_ema50 basis; const-bucket-2 on the D1
# basis over 2025 OOT — a hardcoded ±1.0-ATR cut). Trend is now carried by the CONTINUOUS
# D1_dist_from_ema200_atr (ctx_cont) + the 18 MULTI-TF REGIME_V4 features (per-TF regime class
# m15/h1/h4/d1 + trend-age + cross-TF agreement) — "all smart, no hardcoded bucket". Multi-TF is
# UNAFFECTED (ctx_cat is a separate shared-vocab embedding from the seq branches; REGIME_V4 adds
# per-TF regime). The degenerate legacy trend bucket is never part of the active contract.
ORDERED_CTX_CAT_NAMES_V3: List[str] = list(MODEL_NATIVE_CTX_CAT_FIELDS)
CTX_CAT_DIM_V3 = len(ORDERED_CTX_CAT_NAMES_V3)  # exact active contract: 5


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
    # ONE-TRUTH: the volume slice of the price-state list MUST equal the canonical
    # VOLUME_FEATURE_NAMES (no silent drift between module and contract).
    from gx1.features.volume_features import VOLUME_FEATURE_NAMES
    vol_slice = PER_BAR_PRICE_STATE_FIELDS_V3[-len(VOLUME_FEATURE_NAMES):]
    if vol_slice != list(VOLUME_FEATURE_NAMES):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_V3] volume-field drift: contract tail {vol_slice} "
            f"!= VOLUME_FEATURE_NAMES {list(VOLUME_FEATURE_NAMES)}"
        )


assert_signal_bridge_v3_contract()


# ---------------------------------------------------------------------------
# v2-compatible aliases — same shape so existing call-sites can swap import path
# ---------------------------------------------------------------------------
SIGNAL_BRIDGE_ID = SIGNAL_BRIDGE_ID_V3
ORDERED_FIELDS = ORDERED_SEQ_FIELDS_V3  # 41 fields per bar (7 bridge + 34 price-state)
SEQ_SIGNAL_DIM = SEQ_SIGNAL_DIM_V3  # 41
SNAP_SIGNAL_DIM = SNAP_SIGNAL_DIM_V3  # 41
CONTRACT_SHA256 = CONTRACT_SHA256_V3

ORDERED_CTX_CONT_NAMES_EXTENDED = ORDERED_CTX_CONT_NAMES_V3
ORDERED_CTX_CAT_NAMES_EXTENDED = ORDERED_CTX_CAT_NAMES_V3  # 6
N_CTX_CONT_EXTENDED = CTX_CONT_DIM_V3
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

ALLOWED_CTX_CONT_DIMS = tuple(range(CTX_CONT_DIM_V3, CTX_CONT_DIM_V3 + 21))  # bundle-driven headroom
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


def validate_seq_signal(seq_x: np.ndarray, *, context: str = "unknown") -> None:
    if seq_x is None:
        raise RuntimeError(f"[SIGNAL_BRIDGE_FAIL_V3] seq_x is None (context={context})")
    arr = np.asarray(seq_x)
    if arr.dtype not in (np.float32, np.float64):
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
    if arr.dtype not in (np.float32, np.float64):
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
        "tag": f"CTX6CAT{int(N_CTX_CAT_EXTENDED)}",
        "source": "signal_bridge_v3_full_contract",
        "ctx_cont_rule": (
            f"bundle-driven ctx_cont_dim must equal {int(N_CTX_CONT_EXTENDED)} "
            "and match the v3 ordered list"
        ),
        "ctx_cat_rule": (
            f"bundle-driven ctx_cat_dim must equal {int(N_CTX_CAT_EXTENDED)} "
            "and match the v3 ordered list"
        ),
    }


def validate_bundle_ctx_contract_in_strict(
    expected_ctx_cont_dim: int,
    expected_ctx_cat_dim: int,
    ordered_ctx_cont_names: Sequence[str],
    ordered_ctx_cat_names: Sequence[str],
    *,
    context: str = "bundle_meta",
) -> None:
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
