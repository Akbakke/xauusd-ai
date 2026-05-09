"""
XGB_SIGNAL_BRIDGE_V2 contract (SSoT for V10 v2 entry transformer).

This is the option α + β + γ extension of signal_bridge_v1, designed for the
2026-Q2 GX1 stack rebuild:

  α — extended ctx_cont (V10's macro context features)
       21 → 43 features (added explicit H1/H4/D1/M15 indicators from canonical_v2)

  β — extended per-bar SEQUENCE features
       7 → 30 features per M5 bar
       (V10 v1 only saw XGB outputs per bar; V10 v2 also sees price-state
        features per bar, similar to V3's per-M1-bar 58-feature input.)

  γ — extended sequence length
       seq_len 30 → 96 (2.5h → 8h, full session, symmetric with V3's M1L512)

Backward-compat with v1:
  - Indices 0..6 of ORDERED_SEQ_FIELDS_V2 are IDENTICAL to v1's ORDERED_FIELDS,
    so V10's anchor logic (`_ANCHOR_IDX = (p_long_idx, p_short_idx, p_flat_idx)`)
    continues to work without modification.
  - ctx_cat_dim unchanged (6 categorical features).
  - ctx_cont_dim PREFIX matches v1's 21 features (V2's 22 new features appended).

Hard rules (unchanged from v1):
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


SIGNAL_BRIDGE_ID_V2 = "XGB_SIGNAL_BRIDGE_V2"


# ---------------------------------------------------------------------------
# XGB-bridge fields (must occupy indices 0..6 — V10 anchor relies on this)
# ---------------------------------------------------------------------------

# proba[:, 0] = LONG, proba[:, 1] = SHORT, proba[:, 2] = FLAT  (per XGBMultiheadModel)
XGB_PROB_FIELDS_ORDERED_V2: List[str] = [
    "p_long",
    "p_short",
    "p_flat",
]

ORDERED_BRIDGE_FIELDS_V2: List[str] = [
    *XGB_PROB_FIELDS_ORDERED_V2,
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
]
BRIDGE_DIM_V2 = len(ORDERED_BRIDGE_FIELDS_V2)  # 7

# Anchor indices — explicit so callers can validate
ANCHOR_FIELDS_V2 = ("p_long", "p_short", "p_flat")
ANCHOR_INDICES_V2 = tuple(ORDERED_BRIDGE_FIELDS_V2.index(f) for f in ANCHOR_FIELDS_V2)
assert ANCHOR_INDICES_V2 == (0, 1, 2), "anchor indices must remain (0,1,2)"


# ---------------------------------------------------------------------------
# Per-bar PRICE-STATE features (option β: extension of v1's seq input)
# ---------------------------------------------------------------------------
# 30 additional features per M5 bar describing price/momentum/volatility/range/structure.
# All sourced from canonical_features_v2 (joined per-bar at training time).
# 2026-05-03 update: dropped `roc20` (dup of ret_20) and `_v1_spread_z` (constant),
# added 9 SMC features (HH/HL state, BOS, CHOCH, sweep, premium/discount).
PER_BAR_PRICE_STATE_FIELDS_V2: List[str] = [
    # Volatility / momentum / returns (10)
    "atr",                            # M5 ATR(14)
    "atr_z",                          # ATR z-score over 50 bars
    "ret_1",                          # last bar return (bps)
    "ret_5",                          # 5-bar return (bps)
    "ret_20",                         # 20-bar return (replaces roc20 — same thing)
    "rvol_20",                        # 20-bar realized vol (bps)
    "body_pct",                       # body / bar-range
    "wick_asym",                      # (upper - lower) / wick-total
    "ema20_slope",                    # EMA(20) slope
    "pos_vs_ema200",                  # (close - ema200) / ema200 (bps)
    # _v1 family (11)
    "_v1_atr14",                      # canonical _v1 ATR
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
    # SMC features (9 — Smart Money Concept structure + liquidity)
    "smc_swing_state",                # 0=HH+HL up, 1=up-bias, 2=down-bias, 3=LH+LL down, 4=mixed
    "smc_bos_up",                     # break of structure up (close > last swing high)
    "smc_bos_down",                   # break of structure down
    "smc_choch",                      # change of character flag (up↔down flip)
    "smc_sweep_up",                   # liquidity sweep up (false breakout above swing high)
    "smc_sweep_down",                 # liquidity sweep down
    "smc_sweep_size_atr",             # magnitude of last sweep, ATR-normalized
    "smc_bars_since_sweep",           # bars since last sweep
    "smc_premium_discount",           # 0..1 position in last_swing_low..last_swing_high range
]
PRICE_STATE_DIM_V2 = len(PER_BAR_PRICE_STATE_FIELDS_V2)  # 30


# ---------------------------------------------------------------------------
# SEQ + SNAP fields (per-bar full feature vector)
# ---------------------------------------------------------------------------
# Both seq (per-bar over seq_len bars) and snap (current bar) use the same schema.
ORDERED_SEQ_FIELDS_V2: List[str] = ORDERED_BRIDGE_FIELDS_V2 + PER_BAR_PRICE_STATE_FIELDS_V2
SEQ_SIGNAL_DIM_V2 = len(ORDERED_SEQ_FIELDS_V2)  # 30

ORDERED_SNAP_FIELDS_V2: List[str] = list(ORDERED_SEQ_FIELDS_V2)
SNAP_SIGNAL_DIM_V2 = len(ORDERED_SNAP_FIELDS_V2)  # 30

# Locked seq length for V10 v2 (8h M5 = 96 bars)
DEFAULT_SEQ_LEN_V2 = 96


# ---------------------------------------------------------------------------
# CTX_CONT fields (option α: macro-context features per decision)
# ---------------------------------------------------------------------------
# v1 prefix (21 features, unchanged order so prefix-loading works for legacy data)
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

# v2 extension (22 NEW multi-TF context features)
ORDERED_CTX_CONT_V2_EXTENSION: List[str] = [
    # H1 (6)
    "_v1h1_ema_diff",
    "_v1h1_atr",
    "_v1h1_rsi14_z",
    "_v1h1_slope3",
    "_v1h1_slope5",
    "_v1h1_vwap_drift",
    # H4 (5)
    "_v1h4_ema_diff",
    "_v1h4_atr",
    "_v1h4_rsi14_z",
    "_v1h4_slope3",
    "_v1h4_slope5",
    # D1 (6, from canonical_v2)
    "d1_atr14_canon_v2",
    "d1_rsi14_canon_v2",
    "d1_ema_slope_20_canon_v2",
    "d1_range_z_20_canon_v2",
    "d1_close_pct_in_20day_range_canon_v2",
    "d1_pct_change_5_canon_v2",
    # M15 (5, from canonical_v2)
    "m15_atr14_canon_v2",
    "m15_rsi14_canon_v2",
    "m15_ema_slope_5_canon_v2",
    "m15_range_z_20_canon_v2",
    "m15_trend_sign_canon_v2",
]

ORDERED_CTX_CONT_NAMES_V2: List[str] = (
    ORDERED_CTX_CONT_V1_PREFIX + ORDERED_CTX_CONT_V2_EXTENSION
)
CTX_CONT_DIM_V2 = len(ORDERED_CTX_CONT_NAMES_V2)  # 43


# ---------------------------------------------------------------------------
# CTX_CAT fields (unchanged from v1)
# ---------------------------------------------------------------------------
ORDERED_CTX_CAT_NAMES_V2: List[str] = [
    "session_id",
    "trend_regime_id",
    "vol_regime_id",
    "atr_bucket",
    "spread_bucket",
    "H4_trend_sign_cat",
]
CTX_CAT_DIM_V2 = len(ORDERED_CTX_CAT_NAMES_V2)  # 6


# ---------------------------------------------------------------------------
# Schema hashes (for runtime validation)
# ---------------------------------------------------------------------------
def _hash_fields(fields: List[str]) -> str:
    return hashlib.sha256(("|".join(fields)).encode("utf-8")).hexdigest()


SEQ_CONTRACT_SHA256_V2 = _hash_fields(ORDERED_SEQ_FIELDS_V2)
SNAP_CONTRACT_SHA256_V2 = _hash_fields(ORDERED_SNAP_FIELDS_V2)
CTX_CONT_CONTRACT_SHA256_V2 = _hash_fields(ORDERED_CTX_CONT_NAMES_V2)
CTX_CAT_CONTRACT_SHA256_V2 = _hash_fields(ORDERED_CTX_CAT_NAMES_V2)

# Combined hash spanning all 4 schemas — signature for the v2 V10 bundle
CONTRACT_SHA256_V2 = hashlib.sha256(
    "|".join([
        SEQ_CONTRACT_SHA256_V2,
        SNAP_CONTRACT_SHA256_V2,
        CTX_CONT_CONTRACT_SHA256_V2,
        CTX_CAT_CONTRACT_SHA256_V2,
    ]).encode("utf-8")
).hexdigest()


# ---------------------------------------------------------------------------
# Public introspection helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SignalBridgeV2Contract:
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


CONTRACT_V2 = SignalBridgeV2Contract(
    seq_fields=tuple(ORDERED_SEQ_FIELDS_V2),
    snap_fields=tuple(ORDERED_SNAP_FIELDS_V2),
    ctx_cont_names=tuple(ORDERED_CTX_CONT_NAMES_V2),
    ctx_cat_names=tuple(ORDERED_CTX_CAT_NAMES_V2),
    seq_dim=SEQ_SIGNAL_DIM_V2,
    snap_dim=SNAP_SIGNAL_DIM_V2,
    ctx_cont_dim=CTX_CONT_DIM_V2,
    ctx_cat_dim=CTX_CAT_DIM_V2,
    default_seq_len=DEFAULT_SEQ_LEN_V2,
    bridge_dim=BRIDGE_DIM_V2,
    anchor_indices=ANCHOR_INDICES_V2,
    schema_hash=CONTRACT_SHA256_V2,
)


def assert_signal_bridge_v2_contract() -> None:
    """Hard-fail check that the v2 contract is internally consistent."""
    if SEQ_SIGNAL_DIM_V2 != len(ORDERED_SEQ_FIELDS_V2):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_V2] seq dim mismatch: {SEQ_SIGNAL_DIM_V2} != {len(ORDERED_SEQ_FIELDS_V2)}"
        )
    if SNAP_SIGNAL_DIM_V2 != len(ORDERED_SNAP_FIELDS_V2):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_V2] snap dim mismatch: {SNAP_SIGNAL_DIM_V2} != {len(ORDERED_SNAP_FIELDS_V2)}"
        )
    if CTX_CONT_DIM_V2 != len(ORDERED_CTX_CONT_NAMES_V2):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_V2] ctx_cont dim mismatch: {CTX_CONT_DIM_V2} != {len(ORDERED_CTX_CONT_NAMES_V2)}"
        )
    if CTX_CAT_DIM_V2 != len(ORDERED_CTX_CAT_NAMES_V2):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_V2] ctx_cat dim mismatch: {CTX_CAT_DIM_V2} != {len(ORDERED_CTX_CAT_NAMES_V2)}"
        )
    # Anchor invariant: p_long/p_short/p_flat must remain at indices 0,1,2
    for expected_idx, name in enumerate(ANCHOR_FIELDS_V2):
        actual = ORDERED_SEQ_FIELDS_V2.index(name)
        if actual != expected_idx:
            raise RuntimeError(
                f"[SIGNAL_BRIDGE_V2] anchor index drift: {name} at {actual} (expected {expected_idx})"
            )
    # Seq/snap must share the same schema
    if ORDERED_SEQ_FIELDS_V2 != ORDERED_SNAP_FIELDS_V2:
        raise RuntimeError("[SIGNAL_BRIDGE_V2] seq/snap field lists must match")


assert_signal_bridge_v2_contract()


# ---------------------------------------------------------------------------
# v1-compatible aliases (so runtime files can swap import path with no logic
# change). These intentionally shadow the v1 names so that existing call sites
# like `from gx1.contracts.signal_bridge_v2 import ORDERED_FIELDS, SEQ_SIGNAL_DIM`
# work identically.
# ---------------------------------------------------------------------------
SIGNAL_BRIDGE_ID = SIGNAL_BRIDGE_ID_V2
ORDERED_FIELDS = ORDERED_SEQ_FIELDS_V2  # 37 fields per bar
SEQ_SIGNAL_DIM = SEQ_SIGNAL_DIM_V2  # 37
SNAP_SIGNAL_DIM = SNAP_SIGNAL_DIM_V2  # 37
CONTRACT_SHA256 = CONTRACT_SHA256_V2

ORDERED_CTX_CONT_NAMES_EXTENDED = ORDERED_CTX_CONT_NAMES_V2  # 43
ORDERED_CTX_CAT_NAMES_EXTENDED = ORDERED_CTX_CAT_NAMES_V2  # 6
N_CTX_CONT_EXTENDED = CTX_CONT_DIM_V2  # 43
N_CTX_CAT_EXTENDED = CTX_CAT_DIM_V2  # 6

ORDERED_CTX_CONT_NAMES_BASELINE = ORDERED_CTX_CONT_V1_PREFIX[:2]
N_CTX_CONT_BASELINE = len(ORDERED_CTX_CONT_NAMES_BASELINE)
ORDERED_CTX_CAT_NAMES_BASELINE = ORDERED_CTX_CAT_NAMES_V2[:5]
N_CTX_CAT_BASELINE = len(ORDERED_CTX_CAT_NAMES_BASELINE)

CTX_CONT_COL_D1_DIST = "D1_dist_from_ema200_atr"
CTX_CONT_COL_H1_COMP = "H1_range_compression_ratio"
CTX_CONT_COL_D1_ATR_PCTL252 = "D1_atr_percentile_252"
CTX_CONT_COL_M15_COMP = "M15_range_compression_ratio"
CTX_CAT_COL_H4_TREND_SIGN = "H4_trend_sign_cat"

ALLOWED_CTX_CONT_DIMS = tuple(range(CTX_CONT_DIM_V2, 65))
ALLOWED_CTX_CAT_DIMS = (CTX_CAT_DIM_V2,)


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
    """Validate seq_x for SIGNAL_BRIDGE_V2 (3D [B, T, 37], finite)."""
    if seq_x is None:
        raise RuntimeError(f"[SIGNAL_BRIDGE_FAIL_V2] seq_x is None (context={context})")
    arr = np.asarray(seq_x)
    if _is_truth_or_smoke() and arr.dtype not in (np.float32, np.float64):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] seq_x invalid dtype={arr.dtype} (context={context})"
        )
    if arr.ndim != 3:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] seq_x.ndim mismatch: expected=3 got={arr.ndim} "
            f"shape={getattr(arr, 'shape', None)} (context={context})"
        )
    if int(arr.shape[-1]) != int(SEQ_SIGNAL_DIM):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] seq_x feature dim mismatch: expected={SEQ_SIGNAL_DIM} "
            f"got={int(arr.shape[-1])} (context={context})"
        )
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] seq_x non-finite count={n_bad} (context={context})"
        )


def validate_snap_signal(snap_x: np.ndarray, *, context: str = "unknown") -> None:
    """Validate snap_x for SIGNAL_BRIDGE_V2 (2D [B, 37], finite)."""
    if snap_x is None:
        raise RuntimeError(f"[SIGNAL_BRIDGE_FAIL_V2] snap_x is None (context={context})")
    arr = np.asarray(snap_x)
    if _is_truth_or_smoke() and arr.dtype not in (np.float32, np.float64):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] snap_x invalid dtype={arr.dtype} (context={context})"
        )
    if arr.ndim != 2:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] snap_x.ndim mismatch: expected=2 got={arr.ndim} "
            f"shape={getattr(arr, 'shape', None)} (context={context})"
        )
    if int(arr.shape[-1]) != int(SNAP_SIGNAL_DIM):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] snap_x feature dim mismatch: expected={SNAP_SIGNAL_DIM} "
            f"got={int(arr.shape[-1])} (context={context})"
        )
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] snap_x non-finite count={n_bad} (context={context})"
        )


def validate_contract_in_truth() -> None:
    """TRUTH/SMOKE invariant: v2 contract must be stable and non-empty."""
    if not _is_truth_or_smoke():
        return
    if not ORDERED_FIELDS or len(set(ORDERED_FIELDS)) != len(ORDERED_FIELDS):
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL_V2] ORDERED_FIELDS invalid (empty/dup)")
    if SEQ_SIGNAL_DIM <= 0 or SNAP_SIGNAL_DIM <= 0:
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL_V2] signal dims invalid (<=0)")
    if CONTRACT.sha256 != CONTRACT_SHA256:
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL_V2] contract sha mismatch (internal)")
    if not ORDERED_CTX_CONT_NAMES_EXTENDED or len(set(ORDERED_CTX_CONT_NAMES_EXTENDED)) != len(ORDERED_CTX_CONT_NAMES_EXTENDED):
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL_V2] ORDERED_CTX_CONT_NAMES_EXTENDED invalid")
    if not ORDERED_CTX_CAT_NAMES_EXTENDED or len(set(ORDERED_CTX_CAT_NAMES_EXTENDED)) != len(ORDERED_CTX_CAT_NAMES_EXTENDED):
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL_V2] ORDERED_CTX_CAT_NAMES_EXTENDED invalid")


def get_canonical_ctx_contract() -> Dict[str, object]:
    """Return the canonical v2 ONE-UNIVERSE ctx contract (43/6)."""
    return {
        "ctx_cont_dim": int(N_CTX_CONT_EXTENDED),
        "ctx_cat_dim": int(N_CTX_CAT_EXTENDED),
        "ctx_cont_names": list(ORDERED_CTX_CONT_NAMES_EXTENDED),
        "ctx_cat_names": list(ORDERED_CTX_CAT_NAMES_EXTENDED),
        "tag": "CTX43CAT6_V2",
        "source": "signal_bridge_v2_full_contract",
        "ctx_cont_rule": "bundle-driven ctx_cont_dim must equal 43 and match the v2 ordered list",
        "ctx_cat_rule": "bundle-driven ctx_cat_dim must equal 6 and match the v2 ordered list",
    }


def validate_bundle_ctx_contract_in_strict(
    expected_ctx_cont_dim: int,
    expected_ctx_cat_dim: int,
    ordered_ctx_cont_names: Sequence[str],
    ordered_ctx_cat_names: Sequence[str],
    *,
    context: str = "bundle_meta",
) -> None:
    """TRUTH/SMOKE hard-gate for the v2 43/6 contract."""
    if not _is_truth_or_smoke():
        return
    if expected_ctx_cont_dim != N_CTX_CONT_EXTENDED:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] {context} expected_ctx_cont_dim={expected_ctx_cont_dim} "
            f"!= v2 contract dim={N_CTX_CONT_EXTENDED}"
        )
    if expected_ctx_cat_dim != N_CTX_CAT_EXTENDED:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] {context} expected_ctx_cat_dim={expected_ctx_cat_dim} "
            f"!= v2 contract dim={N_CTX_CAT_EXTENDED}"
        )
    meta_cont = list(ordered_ctx_cont_names)[:expected_ctx_cont_dim]
    contract_cont = list(ORDERED_CTX_CONT_NAMES_EXTENDED)
    if meta_cont != contract_cont:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] bundle {context} ordered_ctx_cont_names mismatch: "
            f"contract={contract_cont!r} meta={meta_cont!r}"
        )
    meta_cat = list(ordered_ctx_cat_names)[:expected_ctx_cat_dim]
    contract_cat = list(ORDERED_CTX_CAT_NAMES_EXTENDED)
    if meta_cat != contract_cat:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL_V2] bundle {context} ordered_ctx_cat_names mismatch: "
            f"contract={contract_cat!r} meta={meta_cat!r}"
        )
