"""
EXIT_TRANSFORMER_V0 – ML-only exit transformer (no rules/router/critic).

Provides:
- ExitTransformerV0 (PyTorch model)
- ExitTransformerDecider (inference)
- train_from_exits_jsonl (retained Exit training utility)
- save/load artifacts (model/config/sha)

Hard gates:
- feature_dim must be EXIT_IO_FEATURE_COUNT
- window_len must match config; validate every window
- No fallbacks; missing artifacts/files -> RuntimeError/FileNotFoundError
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gx1.exits.contracts.exit_io_v1_ctx36 import (
    EXIT_IO_V1_CTX36_FEATURES,
    EXIT_IO_V1_CTX36_IO_VERSION,
    compute_feature_names_hash,
    assert_exit_io_v1_ctx36_contract,
)
from gx1.exits.contracts.exit_io_v3_ctx36_m1l512_phase5 import (
    EXIT_IO_V3_CTX36_M1L512_PHASE5_IO_VERSION,
)
from gx1.exits.contracts.registry import get_exit_io_contract

# -----------------------------------------------------------------------------
# IO CONTRACT
# -----------------------------------------------------------------------------
_DEFAULT_EXIT_CONTRACT = get_exit_io_contract(EXIT_IO_V3_CTX36_M1L512_PHASE5_IO_VERSION)
EXIT_IO_FEATURE_COUNT = int(_DEFAULT_EXIT_CONTRACT["feature_count"])
EXIT_IO_VERSION = str(_DEFAULT_EXIT_CONTRACT["io_version"])
EXIT_FEATURE_NAMES_HASH = str(_DEFAULT_EXIT_CONTRACT["feature_hash"])

log = logging.getLogger(__name__)
_exit_features_proof_logged = False
assert_exit_io_v1_ctx36_contract()

# -----------------------------------------------------------------------------
# Optional torch
# -----------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_WINDOW_LEN = int(_DEFAULT_EXIT_CONTRACT["default_window_len"])
DEFAULT_D_MODEL = 64
DEFAULT_N_LAYERS = 2
DEFAULT_N_HEADS = 4
DEFAULT_THRESHOLD = 0.5

LAST_GO_FILENAME = "LAST_GO.txt"
TRUTH_E2E_REPORTS = "reports/truth_e2e_sanity"
EXIT_LABEL_FAMILY_NAMES: List[str] = [
    "HARD_DETERIORATION_EDGE_DEATH",
    "BREAK_EVEN_RED_PROTECTION",
    "PROFIT_GIVEBACK_PROTECTION",
    "NO_EDGE_ADVERSE_COLLAPSE",
]
EXIT_LABEL_FAMILY_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(EXIT_LABEL_FAMILY_NAMES)}
EXIT_LABEL_FAMILY_IGNORE_INDEX = -100
EXIT_AUX_FAMILY_NAMES: List[str] = [
    "HARD_DETERIORATION_EDGE_DEATH",
    "BREAK_EVEN_RED_PROTECTION",
    "NO_EDGE_ADVERSE_COLLAPSE",
]
EXIT_AUX_FAMILY_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(EXIT_AUX_FAMILY_NAMES)}
EXIT_AUX_FAMILY_IGNORE_INDEX = -100
EXIT_PROFIT_HEAD_BUCKET_NAMES: List[str] = [
    "PROFIT_PROTECT_LIKE",
    "HARD_DETERIORATION_LIKE",
    "BREAK_EVEN_RED_LIKE",
    "EARLY_WEAK_ADVERSE",
    "NORMAL_THRESHOLD_MASS",
]
DEFAULT_FAMILY_AUX_LOSS_WEIGHT = 0.0
DEFAULT_PROFIT_PROTECT_HEAD_LOSS_WEIGHT = 0.50

# ── Dip-analysis head layout (V3 exit) — does the current drawdown recover?
# Output index = flatten over (horizon, quantile). recovery = max(pnl after now
# within K) − pnl_now. p10 = downside-recovery-risk (don't hold a dip that won't
# recover). Trainer pinball loss + consumers use this layout. See memory
# project_gx1_dip_aware_entry_timing.
V3_DIP_HORIZONS = (12, 48, 144)        # M1 bars
V3_DIP_QUANTILES = ("recovery_p50", "recovery_p10")
V3_DIP_HEAD_DIM = len(V3_DIP_HORIZONS) * len(V3_DIP_QUANTILES)  # = 6

# ── New aux heads (2026-05-26) — mirror of V10 entry heads onto the exit side.
# All M1-horizon, forward-looking from the current in-trade bar, single-side
# (the open trade's pnl is already side-signed). Gated, state-dict-safe, default
# OFF (bundles trained without them stay bit-identical). Targets computed in the
# exit labeler `_attach_labels_to_exit_records`; pinball/smooth_l1 in the trainer.
V3_NEW_HEAD_HORIZONS = (12, 48, 144)   # M1 bars (match dip horizons)
# timing: WHEN the dip recovers / WHEN the worst-adverse hits (frac of K ∈[0,1]).
V3_TIMING_TARGETS = ("time_to_recovery_frac", "time_to_worst_frac")
V3_TIMING_HEAD_DIM = len(V3_NEW_HEAD_HORIZONS) * len(V3_TIMING_TARGETS)  # = 6
# tail-risk: p90 of worst further-adverse pnl over K (pinball q=0.9) — cut before tail.
V3_TAIL_RISK_QUANTILE = 0.9
V3_TAIL_RISK_HEAD_DIM = len(V3_NEW_HEAD_HORIZONS)  # = 3
# vol-forecast: forward realized vol (bps) over K — regime/hold sizing.
V3_VOL_FORECAST_HEAD_DIM = len(V3_NEW_HEAD_HORIZONS)  # = 3
# forecast: cumulative forward pnl (bps) over K — exit-timing signal.
V3_FORECAST_HEAD_DIM = len(V3_NEW_HEAD_HORIZONS)  # = 3

# -----------------------------------------------------------------------------
# LAST_GO resolver
# -----------------------------------------------------------------------------
def get_last_go_exits_dataset(gx1_data: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve exits jsonl from LAST_GO pointer (TRUTH E2E).
    Returns: dict(go_run_dir, go_run_id, exits_jsonl_path)
    """
    base = Path(gx1_data or os.environ.get("GX1_DATA", "")).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError("GX1_DATA not set or not a directory")
    last_go_path = base / TRUTH_E2E_REPORTS / LAST_GO_FILENAME
    if not last_go_path.exists():
        raise FileNotFoundError(f"LAST_GO pointer missing: {last_go_path}")
    go_run_dir = Path(last_go_path.read_text().strip()).expanduser().resolve()
    if not go_run_dir.is_dir():
        raise FileNotFoundError(f"LAST_GO run dir does not exist: {go_run_dir}")
    go_run_id = go_run_dir.name
    chunk_dir = go_run_dir / "replay" / "chunk_0"
    exits_dir = chunk_dir / "logs" / "exits"
    exits_jsonl_path = exits_dir / f"exits_{go_run_id}.jsonl"
    if not exits_jsonl_path.exists():
        candidates = sorted(exits_dir.glob("exits_*.jsonl")) if exits_dir.exists() else []
        if len(candidates) == 1:
            exits_jsonl_path = candidates[0]
        elif not candidates:
            raise FileNotFoundError(f"Exits jsonl not found: {exits_jsonl_path}")
        else:
            raise FileNotFoundError(
                f"Exits jsonl ambiguous: expected {exits_jsonl_path} or single exits_*.jsonl, found {[p.name for p in candidates]}"
            )
    return {
        "go_run_dir": go_run_dir,
        "go_run_id": go_run_id,
        "exits_jsonl_path": exits_jsonl_path,
    }


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def config_hash_v3(window_len: int, d_model: int, n_layers: int) -> str:
    h = hashlib.sha256()
    h.update(f"{window_len}:{d_model}:{n_layers}".encode("utf-8"))
    return h.hexdigest()[:16]


def validate_window_v3(window: np.ndarray, window_len: int, feat_dim: int, context: str = "") -> None:
    if window.ndim != 2 or window.shape[0] != window_len or window.shape[1] != feat_dim:
        if os.getenv("GX1_EXIT_HASH_GUARD_BYPASS") == "1":
            log.warning(
                "[EXIT_WINDOW_SHAPE_BYPASS] context=%s expected=%s got=%s",
                context,
                (window_len, feat_dim),
                window.shape,
            )
            return
        raise RuntimeError(
            f"EXIT_WINDOW_SHAPE_MISMATCH[{context}]: expected {(window_len, feat_dim)}, got {window.shape}"
        )


def _make_deterministic(seed: int = 42) -> None:
    if not TORCH_AVAILABLE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
    np.random.seed(seed)
    random.seed(seed)


def _attach_labels_to_exit_records(
    records: List[Dict[str, Any]],
    *,
    min_hold_bars: int = 2,
    capture_ratio_threshold: float = 0.8,
    enable_capture_label: bool = False,
    det_max_per_trade: int = 4,
    postprog_mfe_min_long: float = 6.0,
    postprog_mfe_min_short: float = 5.0,
    postprog_dd_min_long: float = 20.0,
    postprog_dd_min_short: float = 15.0,
    postprog_giveback_ratio_min_long: float = 0.50,
    postprog_giveback_ratio_min_short: float = 0.45,
    postprog_giveback_ratio_floor: float = 0.10,
    postprog_pnl_frac_floor_long: float = 0.35,
    postprog_pnl_frac_floor_short: float = 0.35,
    postprog_pnl_abs_floor_long: float = 0.5,
    postprog_pnl_abs_floor_short: float = 1.0,
    early_long_recovery_max_bars: float = 3.0,
    early_long_recovery_mfe_cap_bps: float = 1.0,
    early_long_recovery_future_max_bps: float = 10.0,
    early_long_recovery_negative_weight: float = 1.75,
    no_edge_adverse_mfe_cap_bps: float = 1.0,
    no_edge_adverse_dd_min_bps: float = 18.0,
    no_edge_adverse_pnl_min_bps: float = -12.0,
    no_edge_adverse_slope_max: float = -4.0,
    no_edge_adverse_pflat_max: float = 0.22,
    no_edge_adverse_margin_max: float = 0.16,
    no_edge_adverse_unc_min: float = 0.55,
    profit_protect_mfe_min_long: float = 30.0,
    profit_protect_mfe_min_short: float = 20.0,
    profit_protect_dd_min_bps: float = 20.0,
    profit_protect_giveback_ratio_min: float = 0.30,
    profit_protect_pnl_min_bps: float = 0.0,
    profit_protect_pnl_max_bps: float = 120.0,
    profit_protect_pnl_frac_mfe_max: float = 0.70,
    profit_protect_time_since_mfe_min_bars: float = 2.0,
    profit_protect_future_headroom_abs_max_bps: float = 10.0,
    profit_protect_future_headroom_frac_max: float = 0.25,
    profit_protect_mask_signal_pflat_min: float = 0.35,
    profit_protect_mask_signal_unc_min: float = 0.52,
    profit_protect_mask_signal_margin_max: float = 0.22,
) -> None:
    """
    Edge-death / giveback labels with protect-earned-MFE semantics.

    Primary truth:
      - trade had edge (MFE arm),
      - edge is deteriorating now (giveback/stale/back-to-flat),
      - should exit to protect earned MFE.

    Secondary truth (optional, disabled by default):
      - profit-capture based on peak proximity.

    Redesign note:
      - Profit-protect labels are hindsight-aware and should teach
        deterioration after earned MFE, not generic early close.
      - A profit-protect positive is therefore suppressed when the same
        state still has meaningful future recovery left from current pnl.

    For each trade bar i:
      - future_max_pnl = max(pnl_bps_now from i to trade end)
      - capture_ratio = pnl_bps_now / future_max_pnl
      - optional capture label = 1 if capture_ratio >= capture_ratio_threshold
    NaN/inf-safe: non-finite values or non-positive future_max_pnl -> label 0.
    Adds deterministic post-progress deterioration labels (relabel-truth) when a trade:
      1) has meaningful progress (MFE arm),
      2) then shows clear deterioration by one of:
         - giveback after progress,
         - stale after peak,
         - back to near-flat/red.
    Side asymmetry is explicit (short can trigger earlier than long).
    """
    if not records:
        return

    def _get_scalar(rec: Dict[str, Any], key: str) -> float:
        if key in rec:
            val = rec[key]
        else:
            val = (rec.get("scalars") or {}).get(key)
        if val is None:
            raise RuntimeError(f"[EXIT_LABELER_MISSING_SCALAR] key={key}")
        try:
            out = float(val)
        except Exception as e:
            raise RuntimeError(f"[EXIT_LABELER_SCALAR_PARSE_FAIL] key={key} err={e}")
        if not math.isfinite(out):
            raise RuntimeError(f"[EXIT_LABELER_SCALAR_NONFINITE] key={key} val={val}")
        return out

    def _get_scalar_optional(rec: Dict[str, Any], key: str) -> Optional[float]:
        val = rec.get(key)
        if val is None:
            val = (rec.get("scalars") or {}).get(key)
        if val is None:
            return None
        try:
            out = float(val)
        except Exception:
            return None
        if not math.isfinite(out):
            return None
        return out

    def _get_trade_teacher_state(recs_for_trade: List[Dict[str, Any]]) -> Dict[str, Any]:
        first = recs_for_trade[0] if recs_for_trade else {}
        final_reason = str(
            first.get("teacher_final_exit_reason")
            or first.get("teacher_exit_reason")
            or first.get("final_exit_reason")
            or ""
        ).strip().upper()
        final_pnl = _get_scalar_optional(first, "teacher_final_pnl_bps")
        final_mfe = _get_scalar_optional(first, "teacher_final_mfe_bps")
        final_mae = _get_scalar_optional(first, "teacher_final_mae_bps")
        final_duration_bars = _get_scalar_optional(first, "teacher_duration_bars")
        post_exit_mfe = _get_scalar_optional(first, "teacher_post_exit_mfe_bps")
        post_exit_mfe_replay_end = _get_scalar_optional(first, "teacher_post_exit_mfe_bps_replay_end_obs")
        if post_exit_mfe is None:
            post_exit_mfe = post_exit_mfe_replay_end
        if not final_reason and final_pnl is None and final_mfe is None:
            return {
                "available": False,
                "final_reason": "",
                "final_pnl_bps": math.nan,
                "final_mfe_bps": math.nan,
                "final_mae_bps": math.nan,
                "final_duration_bars": math.nan,
                "post_exit_mfe_bps": math.nan,
                "retention_frac": math.nan,
                "profit_eligible": False,
                "is_cata": False,
                "is_eof_positive": False,
                "is_eof_negative": False,
                "is_policy_positive": False,
                "is_negative_close": False,
                "good_close": False,
                "bad_rot": False,
            }
        final_pnl_f = float(final_pnl) if final_pnl is not None else math.nan
        final_mfe_f = float(final_mfe) if final_mfe is not None else math.nan
        final_mae_f = float(final_mae) if final_mae is not None else math.nan
        final_duration_bars_f = float(final_duration_bars) if final_duration_bars is not None else math.nan
        post_exit_mfe_f = float(post_exit_mfe) if post_exit_mfe is not None else math.nan
        retention_frac = (
            float(final_pnl_f / final_mfe_f)
            if math.isfinite(final_pnl_f) and math.isfinite(final_mfe_f) and final_mfe_f > 0.0
            else math.nan
        )
        profit_eligible = bool(math.isfinite(final_mfe_f) and final_mfe_f >= teacher_trade_profit_mfe_min)
        is_cata = final_reason == "CATASTROPHIC_GUARD"
        is_eof_positive = final_reason == "REPLAY_EOF" and math.isfinite(final_pnl_f) and final_pnl_f > 0.0
        is_eof_negative = final_reason == "REPLAY_EOF" and math.isfinite(final_pnl_f) and final_pnl_f < 0.0
        is_policy_positive = final_reason in {"POLICY_FRIDAY_FLAT", "POLICY_WEEK_FLAT"} and math.isfinite(final_pnl_f) and final_pnl_f > 0.0
        is_negative_close = math.isfinite(final_pnl_f) and final_pnl_f < 0.0
        good_close = bool(
            profit_eligible
            and math.isfinite(final_pnl_f)
            and final_pnl_f >= teacher_positive_close_min_pnl_bps
            and math.isfinite(retention_frac)
            and retention_frac >= teacher_positive_retention_min
            and (
                not math.isfinite(post_exit_mfe_f)
                or post_exit_mfe_f <= teacher_positive_post_exit_headroom_max_bps
            )
            and (
                final_reason in {"MODEL_EXIT", "POLICY_FRIDAY_FLAT", "POLICY_WEEK_FLAT", "REPLAY_EOF"}
                or final_reason == ""
            )
        )
        bad_rot = bool(
            profit_eligible
            and (
                is_cata
                or is_eof_negative
                or is_negative_close
                or (
                    math.isfinite(retention_frac)
                    and retention_frac <= teacher_rot_retention_max
                    and math.isfinite(final_pnl_f)
                    and final_pnl_f <= teacher_rot_final_pnl_max_bps
                )
            )
        )
        return {
            "available": True,
            "final_reason": final_reason,
            "final_pnl_bps": final_pnl_f,
            "final_mfe_bps": final_mfe_f,
            "final_mae_bps": final_mae_f,
            "final_duration_bars": final_duration_bars_f,
            "post_exit_mfe_bps": post_exit_mfe_f,
            "retention_frac": retention_frac,
            "profit_eligible": profit_eligible,
            "is_cata": is_cata,
            "is_eof_positive": is_eof_positive,
            "is_eof_negative": is_eof_negative,
            "is_policy_positive": is_policy_positive,
            "is_negative_close": is_negative_close,
            "good_close": good_close,
            "bad_rot": bad_rot,
        }

    label_variant = os.environ.get("GX1_EXIT_LABEL_VARIANT", "EXIT_LABEL_DET_V1").strip().upper()
    sweep_profile = os.environ.get("GX1_EXIT_LABEL_SWEEP_PROFILE", "").strip().upper()
    main_label_profile = os.environ.get("GX1_EXIT_MAIN_LABEL_PROFILE", "MAIN_V1_CONTROL").strip().upper()
    profit_dd_extra_bps = 0.0
    profit_gb_extra = 0.0
    be_de_max = -0.03
    be_require_signal_extreme = False
    suppress_weak_band_max = 0.02
    suppress_flat_min = 0.44
    suppress_mild_dd_max = 14.0
    suppress_mild_gb_max = 0.85
    signal_not_supportive_de_max = 0.02
    signal_not_supportive_pflat_min = 0.40
    signal_not_supportive_margin_max = 0.20
    signal_not_supportive_unc_min = 0.55
    signal_not_supportive_phat_max = 0.60
    signal_dead_strong_de_max = 0.0
    signal_dead_strong_pflat_min = 0.45
    signal_dead_strong_margin_max = 0.12
    signal_dead_strong_unc_min = 0.60
    no_edge_adverse_enabled = True
    main_use_profit_positive = False
    main_allow_hard_deterioration = True
    main_allow_break_even_red = True
    main_allow_no_edge_adverse = True
    main_allow_late_exit = True
    main_allow_boundary_exit = True
    main_allow_failfast_exit = True
    main_allow_capture = True
    main_profit_min_pnl_bps = 0.0
    main_hard_det_min_pnl_bps = -9999.0
    main_break_even_min_pnl_bps = -9999.0
    main_no_edge_min_pnl_bps = -9999.0
    weekly_teacher_mode = False
    try:
        teacher_trade_profit_mfe_min = float(os.environ.get("GX1_EXIT_WEEKLY_TEACHER_MFE_MIN", "8.0"))
    except Exception:
        teacher_trade_profit_mfe_min = 8.0
    try:
        teacher_positive_close_min_pnl_bps = float(os.environ.get("GX1_EXIT_WEEKLY_TEACHER_POS_PNL_MIN", "8.0"))
    except Exception:
        teacher_positive_close_min_pnl_bps = 8.0
    try:
        teacher_positive_retention_min = float(os.environ.get("GX1_EXIT_WEEKLY_TEACHER_POS_RETENTION_MIN", "0.45"))
    except Exception:
        teacher_positive_retention_min = 0.45
    try:
        teacher_positive_post_exit_headroom_max_bps = float(os.environ.get("GX1_EXIT_WEEKLY_TEACHER_POS_HEADROOM_MAX", "12.0"))
    except Exception:
        teacher_positive_post_exit_headroom_max_bps = 12.0
    try:
        teacher_rot_retention_max = float(os.environ.get("GX1_EXIT_WEEKLY_TEACHER_ROT_RETENTION_MAX", "0.20"))
    except Exception:
        teacher_rot_retention_max = 0.20
    try:
        teacher_rot_final_pnl_max_bps = float(os.environ.get("GX1_EXIT_WEEKLY_TEACHER_ROT_PNL_MAX", "6.0"))
    except Exception:
        teacher_rot_final_pnl_max_bps = 6.0
    try:
        weekly_teacher_positive_weight = float(os.environ.get("GX1_EXIT_WEEKLY_TEACHER_POS_WEIGHT", "1.50"))
    except Exception:
        weekly_teacher_positive_weight = 1.50
    try:
        weekly_teacher_good_hold_weight = float(os.environ.get("GX1_EXIT_WEEKLY_TEACHER_GOOD_HOLD_WEIGHT", "1.15"))
    except Exception:
        weekly_teacher_good_hold_weight = 1.15
    if sweep_profile in {"A_PROFIT_STRICT", "EXIT_LABEL_SWEEP_A_PROFIT_STRICT"}:
        profit_dd_extra_bps = 2.0
        profit_gb_extra = 0.05
    elif sweep_profile in {"B_BE_STRICT", "EXIT_LABEL_SWEEP_B_BE_STRICT"}:
        be_de_max = -0.05
        be_require_signal_extreme = True
    elif sweep_profile in {"C_EARLY_SUPPRESS", "EXIT_LABEL_SWEEP_C_EARLY_SUPPRESS"}:
        suppress_weak_band_max = 0.03
        suppress_flat_min = 0.40
        suppress_mild_dd_max = 16.0
        suppress_mild_gb_max = 0.90
    elif sweep_profile in {"D_HYBRID", "EXIT_LABEL_SWEEP_D_HYBRID"}:
        profit_dd_extra_bps = 2.0
        profit_gb_extra = 0.05
        suppress_weak_band_max = 0.03
        suppress_flat_min = 0.40
        suppress_mild_dd_max = 16.0
        suppress_mild_gb_max = 0.90
    if main_label_profile == "MAIN_V1_CONTROL":
        pass
    elif main_label_profile == "MAIN_V2_PROGRESS_STRICT":
        min_hold_bars = max(int(min_hold_bars), 3)
        postprog_mfe_min_long = 10.0
        postprog_mfe_min_short = 8.0
        postprog_dd_min_long = 24.0
        postprog_dd_min_short = 18.0
        postprog_giveback_ratio_min_long = 0.60
        postprog_giveback_ratio_min_short = 0.55
        det_max_per_trade = min(int(det_max_per_trade), 3)
    elif main_label_profile == "MAIN_V3_SIGNAL_STRICT":
        min_hold_bars = max(int(min_hold_bars), 3)
        be_de_max = -0.06
        be_require_signal_extreme = True
        signal_not_supportive_de_max = 0.0
        signal_not_supportive_pflat_min = 0.44
        signal_not_supportive_margin_max = 0.16
        signal_not_supportive_unc_min = 0.58
        signal_not_supportive_phat_max = 0.58
        signal_dead_strong_de_max = -0.02
        signal_dead_strong_pflat_min = 0.50
        signal_dead_strong_margin_max = 0.10
        signal_dead_strong_unc_min = 0.62
        suppress_weak_band_max = 0.01
        suppress_flat_min = 0.48
        suppress_mild_dd_max = 12.0
        suppress_mild_gb_max = 0.75
    elif main_label_profile == "MAIN_V4_NO_EDGE_STRICT":
        min_hold_bars = max(int(min_hold_bars), 3)
        no_edge_adverse_mfe_cap_bps = 0.5
        no_edge_adverse_dd_min_bps = 24.0
        no_edge_adverse_pnl_min_bps = -20.0
        no_edge_adverse_slope_max = -6.0
        no_edge_adverse_pflat_max = 0.18
        no_edge_adverse_margin_max = 0.12
        no_edge_adverse_unc_min = 0.60
        det_max_per_trade = min(int(det_max_per_trade), 3)
    elif main_label_profile == "MAIN_V5_HYBRID_STRICT":
        min_hold_bars = max(int(min_hold_bars), 3)
        postprog_mfe_min_long = 10.0
        postprog_mfe_min_short = 8.0
        postprog_dd_min_long = 24.0
        postprog_dd_min_short = 18.0
        postprog_giveback_ratio_min_long = 0.60
        postprog_giveback_ratio_min_short = 0.55
        be_de_max = -0.06
        be_require_signal_extreme = True
        signal_not_supportive_de_max = 0.0
        signal_not_supportive_pflat_min = 0.44
        signal_not_supportive_margin_max = 0.16
        signal_not_supportive_unc_min = 0.58
        signal_not_supportive_phat_max = 0.58
        signal_dead_strong_de_max = -0.02
        signal_dead_strong_pflat_min = 0.50
        signal_dead_strong_margin_max = 0.10
        signal_dead_strong_unc_min = 0.62
        no_edge_adverse_mfe_cap_bps = 0.5
        no_edge_adverse_dd_min_bps = 24.0
        no_edge_adverse_pnl_min_bps = -20.0
        no_edge_adverse_slope_max = -6.0
        no_edge_adverse_pflat_max = 0.18
        no_edge_adverse_margin_max = 0.12
        no_edge_adverse_unc_min = 0.60
        suppress_weak_band_max = 0.01
        suppress_flat_min = 0.48
        suppress_mild_dd_max = 12.0
        suppress_mild_gb_max = 0.75
        det_max_per_trade = min(int(det_max_per_trade), 3)
    elif main_label_profile == "MAIN_R4_PROFIT_ONLY":
        min_hold_bars = max(int(min_hold_bars), 3)
        main_use_profit_positive = True
        main_allow_hard_deterioration = False
        main_allow_break_even_red = False
        main_allow_no_edge_adverse = False
        main_allow_late_exit = False
        main_allow_boundary_exit = False
        main_allow_failfast_exit = False
        main_allow_capture = False
        main_profit_min_pnl_bps = 0.5
        profit_protect_mfe_min_long = 22.0
        profit_protect_mfe_min_short = 18.0
        profit_protect_dd_min_bps = 16.0
        profit_protect_giveback_ratio_min = 0.22
        profit_protect_time_since_mfe_min_bars = 1.5
        profit_protect_future_headroom_abs_max_bps = 6.0
        profit_protect_future_headroom_frac_max = 0.16
        det_max_per_trade = min(int(det_max_per_trade), 2)
    elif main_label_profile == "MAIN_R4_PROFIT_FAILFAST":
        min_hold_bars = max(int(min_hold_bars), 3)
        main_use_profit_positive = True
        main_allow_hard_deterioration = False
        main_allow_break_even_red = False
        main_allow_no_edge_adverse = True
        main_allow_late_exit = False
        main_allow_boundary_exit = False
        main_allow_failfast_exit = False
        main_allow_capture = False
        main_profit_min_pnl_bps = 0.0
        main_no_edge_min_pnl_bps = -14.0
        profit_protect_mfe_min_long = 20.0
        profit_protect_mfe_min_short = 16.0
        profit_protect_dd_min_bps = 14.0
        profit_protect_giveback_ratio_min = 0.20
        profit_protect_time_since_mfe_min_bars = 1.0
        profit_protect_future_headroom_abs_max_bps = 8.0
        profit_protect_future_headroom_frac_max = 0.22
        no_edge_adverse_mfe_cap_bps = 0.75
        no_edge_adverse_dd_min_bps = 20.0
        no_edge_adverse_pnl_min_bps = -10.0
        no_edge_adverse_slope_max = -5.0
        no_edge_adverse_pflat_max = 0.20
        no_edge_adverse_margin_max = 0.14
        no_edge_adverse_unc_min = 0.58
        det_max_per_trade = min(int(det_max_per_trade), 2)
    elif main_label_profile == "MAIN_R4_GIVEBACK_STRICT":
        min_hold_bars = max(int(min_hold_bars), 3)
        main_use_profit_positive = True
        main_allow_hard_deterioration = True
        main_allow_break_even_red = True
        main_allow_no_edge_adverse = True
        main_allow_late_exit = False
        main_allow_boundary_exit = False
        main_allow_failfast_exit = False
        main_allow_capture = False
        main_profit_min_pnl_bps = 0.0
        main_hard_det_min_pnl_bps = 1.0
        main_break_even_min_pnl_bps = -1.0
        main_no_edge_min_pnl_bps = -14.0
        postprog_mfe_min_long = 9.0
        postprog_mfe_min_short = 7.0
        postprog_dd_min_long = 20.0
        postprog_dd_min_short = 16.0
        postprog_giveback_ratio_min_long = 0.50
        postprog_giveback_ratio_min_short = 0.45
        profit_protect_mfe_min_long = 18.0
        profit_protect_mfe_min_short = 14.0
        profit_protect_dd_min_bps = 12.0
        profit_protect_giveback_ratio_min = 0.18
        profit_protect_time_since_mfe_min_bars = 1.0
        profit_protect_future_headroom_abs_max_bps = 8.0
        profit_protect_future_headroom_frac_max = 0.20
        det_max_per_trade = min(int(det_max_per_trade), 3)
    elif main_label_profile == "MAIN_R4_HYBRID_BALANCED":
        min_hold_bars = max(int(min_hold_bars), 3)
        main_use_profit_positive = True
        main_allow_hard_deterioration = True
        main_allow_break_even_red = True
        main_allow_no_edge_adverse = True
        main_allow_late_exit = False
        main_allow_boundary_exit = False
        main_allow_failfast_exit = False
        main_allow_capture = False
        main_profit_min_pnl_bps = 0.0
        main_hard_det_min_pnl_bps = 0.0
        main_break_even_min_pnl_bps = -2.0
        main_no_edge_min_pnl_bps = -16.0
        postprog_mfe_min_long = 8.0
        postprog_mfe_min_short = 6.0
        postprog_dd_min_long = 18.0
        postprog_dd_min_short = 14.0
        postprog_giveback_ratio_min_long = 0.45
        postprog_giveback_ratio_min_short = 0.40
        profit_protect_mfe_min_long = 16.0
        profit_protect_mfe_min_short = 12.0
        profit_protect_dd_min_bps = 10.0
        profit_protect_giveback_ratio_min = 0.15
        profit_protect_time_since_mfe_min_bars = 1.0
        profit_protect_future_headroom_abs_max_bps = 10.0
        profit_protect_future_headroom_frac_max = 0.25
        det_max_per_trade = min(int(det_max_per_trade), 3)
    elif main_label_profile == "MAIN_R4_HYBRID_WIDE":
        min_hold_bars = max(int(min_hold_bars), 2)
        main_use_profit_positive = True
        main_allow_hard_deterioration = True
        main_allow_break_even_red = True
        main_allow_no_edge_adverse = True
        main_allow_late_exit = False
        main_allow_boundary_exit = False
        main_allow_failfast_exit = False
        main_allow_capture = False
        main_profit_min_pnl_bps = -0.25
        main_hard_det_min_pnl_bps = -0.5
        main_break_even_min_pnl_bps = -3.0
        main_no_edge_min_pnl_bps = -18.0
        postprog_mfe_min_long = 7.0
        postprog_mfe_min_short = 5.0
        postprog_dd_min_long = 16.0
        postprog_dd_min_short = 12.0
        postprog_giveback_ratio_min_long = 0.40
        postprog_giveback_ratio_min_short = 0.35
        profit_protect_mfe_min_long = 14.0
        profit_protect_mfe_min_short = 10.0
        profit_protect_dd_min_bps = 8.0
        profit_protect_giveback_ratio_min = 0.12
        profit_protect_time_since_mfe_min_bars = 1.0
        profit_protect_future_headroom_abs_max_bps = 12.0
        profit_protect_future_headroom_frac_max = 0.30
        det_max_per_trade = min(int(det_max_per_trade), 4)
    elif main_label_profile == "MAIN_R5_WEEKLY_MONETIZE_ROT":
        min_hold_bars = max(int(min_hold_bars), 3)
        weekly_teacher_mode = True
        main_use_profit_positive = True
        main_allow_hard_deterioration = False
        main_allow_break_even_red = False
        main_allow_no_edge_adverse = False
        main_allow_late_exit = False
        main_allow_boundary_exit = False
        main_allow_failfast_exit = False
        main_allow_capture = False
        main_profit_min_pnl_bps = 0.0
        postprog_mfe_min_long = 10.0
        postprog_mfe_min_short = 8.0
        postprog_dd_min_long = 16.0
        postprog_dd_min_short = 12.0
        postprog_giveback_ratio_min_long = 0.35
        postprog_giveback_ratio_min_short = 0.30
        profit_protect_mfe_min_long = 12.0
        profit_protect_mfe_min_short = 10.0
        profit_protect_dd_min_bps = 8.0
        profit_protect_giveback_ratio_min = 0.15
        profit_protect_pnl_min_bps = 2.0
        profit_protect_pnl_max_bps = 90.0
        profit_protect_pnl_frac_mfe_max = 0.75
        profit_protect_time_since_mfe_min_bars = 4.0
        profit_protect_future_headroom_abs_max_bps = 12.0
        profit_protect_future_headroom_frac_max = 0.22
        det_max_per_trade = min(int(det_max_per_trade), 2)
    else:
        raise RuntimeError(f"[EXIT_MAIN_LABEL_PROFILE_FAIL] unsupported profile={main_label_profile!r}")
    intraday_variants = {"EXIT_LABEL_INTRADAY_H30", "EXIT_LABEL_INTRADAY_FAILFAST_H30"}
    intraday_h = 30 if label_variant in intraday_variants else None
    failfast = label_variant == "EXIT_LABEL_INTRADAY_FAILFAST_H30"
    failfast_loss_bps = -8.0
    failfast_grace_bars = 2
    late_capture_threshold_drop = 0.20
    late_exit_pnl_floor = -5.0
    boundary_minutes = 60.0
    boundary_pnl_floor = -2.0
    log.info(
        "[EXIT_LABEL_VARIANT] variant=%s sweep_profile=%s main_label_profile=%s intraday_h=%s failfast=%s loss_bps=%.1f grace_bars=%d "
        "profit_dd_extra=%.1f profit_gb_extra=%.2f be_de_max=%.2f be_extreme=%s "
        "signal={ns_de<=%.2f,ns_pflat>=%.2f,ns_margin<=%.2f,ns_unc>=%.2f,ns_phat<=%.2f,dead_de<=%.2f,dead_pflat>=%.2f,dead_margin<=%.2f,dead_unc>=%.2f} "
        "suppress={weak_band_max=%.2f,flat_min=%.2f,dd_max=%.1f,gb_max=%.2f} "
        "postprog={hold>=%d,long(mfe>=%.1f,dd>=%.1f,gb>=%.2f),short(mfe>=%.1f,dd>=%.1f,gb>=%.2f)} "
        "no_edge={enabled=%s,mfe<=%.1f,dd>=%.1f,pnl<=%.1f,slope<=%.1f,pflat<=%.2f,margin<=%.2f,unc>=%.2f}",
        label_variant,
        sweep_profile,
        main_label_profile,
        intraday_h,
        failfast,
        failfast_loss_bps,
        failfast_grace_bars,
        profit_dd_extra_bps,
        profit_gb_extra,
        be_de_max,
        be_require_signal_extreme,
        signal_not_supportive_de_max,
        signal_not_supportive_pflat_min,
        signal_not_supportive_margin_max,
        signal_not_supportive_unc_min,
        signal_not_supportive_phat_max,
        signal_dead_strong_de_max,
        signal_dead_strong_pflat_min,
        signal_dead_strong_margin_max,
        signal_dead_strong_unc_min,
        suppress_weak_band_max,
        suppress_flat_min,
        suppress_mild_dd_max,
        suppress_mild_gb_max,
        int(min_hold_bars),
        postprog_mfe_min_long,
        postprog_dd_min_long,
        postprog_giveback_ratio_min_long,
        postprog_mfe_min_short,
        postprog_dd_min_short,
        postprog_giveback_ratio_min_short,
        no_edge_adverse_enabled,
        no_edge_adverse_mfe_cap_bps,
        no_edge_adverse_dd_min_bps,
        no_edge_adverse_pnl_min_bps,
        no_edge_adverse_slope_max,
        no_edge_adverse_pflat_max,
        no_edge_adverse_margin_max,
        no_edge_adverse_unc_min,
    )

    trades: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        if "should_exit" in rec:
            continue
        io = rec.get("io_features") or (rec.get("io") or {}).get("io_features")
        if io is None:
            continue
        scalars = rec.get("scalars") or {}
        if not scalars:
            continue
        tid = rec.get("trade_uid") or rec.get("trade_id")
        if not tid:
            continue
        trades.setdefault(tid, []).append(rec)

    n_main_pos = 0
    n_profit_pos = 0
    n_profit_mask = 0
    n_pos_capture = 0
    n_main_pos_deterioration = 0
    n_pos_both = 0
    n_trades_with_main_pos = 0
    n_trades_with_profit_pos = 0
    bars_per_trade: List[int] = []
    pos_bars: List[float] = []
    pos_counts_per_trade: List[int] = []
    capture_ratios: List[float] = []
    det_counts_per_trade: List[int] = []
    det_reason_counts: Dict[str, int] = {
        "HARD_DETERIORATION_EDGE_DEATH": 0,
        "BREAK_EVEN_RED_PROTECTION": 0,
        "PROFIT_GIVEBACK_PROTECTION": 0,
        "NO_EDGE_ADVERSE_COLLAPSE": 0,
        "OTHER_DETERMINISTIC": 0,
    }
    det_suppressed_early_rest_signature = 0
    det_suppressed_near_early_family_total = 0
    det_suppressed_near_early_family_by_reason: Dict[str, int] = {
        "HARD_DETERIORATION_EDGE_DEATH": 0,
        "BREAK_EVEN_RED_PROTECTION": 0,
        "PROFIT_GIVEBACK_PROTECTION": 0,
    }
    profit_protect_future_recovery_suppressed = 0
    profit_protect_normal_threshold_suppressed = 0
    early_long_recovery_signature_seen = 0
    early_long_recovery_signature_positive_pre = 0
    early_long_recovery_signature_positive_suppressed = 0
    early_long_recovery_signature_downweighted = 0
    no_edge_adverse_collapse_seen = 0
    no_edge_adverse_collapse_pos = 0
    be_pos_pre_strict = 0
    be_pos_post_strict = 0
    no_edge_pos_pre_strict = 0
    total_pos_pre_be_strict = 0
    total_pos_post_be_strict = 0
    teacher_trade_available = 0
    teacher_trade_profit_eligible = 0
    teacher_trade_good_close = 0
    teacher_trade_bad_rot = 0
    active_io_version = str(EXIT_IO_VERSION)

    for tid, recs in trades.items():
        if not recs:
            continue
        if any("should_exit" in r for r in recs):
            continue

        def _sort_key(r: Dict[str, Any]) -> Tuple[float, str]:
            bars_val = (r.get("scalars") or {}).get("bars_held", r.get("bars_held", 0))
            ts_val = r.get("ts") or ""
            return (float(bars_val) if math.isfinite(float(bars_val)) else 0.0, str(ts_val))

        recs = sorted(recs, key=_sort_key)
        teacher_state = _get_trade_teacher_state(recs)
        if teacher_state.get("available"):
            teacher_trade_available += 1
            if bool(teacher_state.get("profit_eligible")):
                teacher_trade_profit_eligible += 1
            if bool(teacher_state.get("good_close")):
                teacher_trade_good_close += 1
            if bool(teacher_state.get("bad_rot")):
                teacher_trade_bad_rot += 1
        pnl = np.array([_get_scalar(r, "pnl_bps_now") for r in recs], dtype=np.float32)
        bars = np.array([_get_scalar(r, "bars_held") for r in recs], dtype=np.float32)
        mfe = np.array([_get_scalar(r, "mfe_bps") for r in recs], dtype=np.float32)
        distance_from_peak = np.array(
            [
                _get_scalar_optional(r, "distance_from_peak_mfe_bps")
                if _get_scalar_optional(r, "distance_from_peak_mfe_bps") is not None
                else _get_scalar(r, "dd_from_mfe_bps")
                for r in recs
            ],
            dtype=np.float32,
        )
        giveback_ratio = np.array([_get_scalar(r, "giveback_ratio") for r in recs], dtype=np.float32)
        time_since_mfe = np.array([_get_scalar(r, "time_since_mfe_bars") for r in recs], dtype=np.float32)
        side_norm = str(recs[0].get("side") or "").strip().lower()
        is_short = side_norm == "short"

        directional_edge_sig = np.full(len(recs), np.nan, dtype=np.float32)
        p_flat_sig = np.full(len(recs), np.nan, dtype=np.float32)
        margin_sig = np.full(len(recs), np.nan, dtype=np.float32)
        uncertainty_sig = np.full(len(recs), np.nan, dtype=np.float32)
        p_hat_sig = np.full(len(recs), np.nan, dtype=np.float32)
        slope_sig = np.full(len(recs), np.nan, dtype=np.float32)
        active_io_version = str(EXIT_IO_VERSION)
        try:
            active_contract = None
            first_rec = recs[0] if recs else {}
            active_io_version = str(first_rec.get("io_version") or first_rec.get("exit_ml_io_version") or EXIT_IO_VERSION)
            try:
                active_contract = get_exit_io_contract(active_io_version)
            except Exception:
                active_contract = None
            active_feature_names = (
                list(active_contract.get("feature_names") or [])
                if isinstance(active_contract, dict)
                else list(EXIT_IO_V1_CTX36_FEATURES)
            )
            feat_idx = {name: i for i, name in enumerate(active_feature_names)}
            required = {
                "p_long": feat_idx.get("p_long"),
                "p_short": feat_idx.get("p_short"),
                "p_flat": feat_idx.get("p_flat"),
                "margin_top1_top2": feat_idx.get("margin_top1_top2"),
                "uncertainty_score": feat_idx.get("uncertainty_score"),
                "p_hat": feat_idx.get("p_hat"),
                "rolling_slope_since_entry": feat_idx.get("rolling_slope_since_entry"),
            }
            if all(v is not None for v in required.values()):
                i_pl = int(required["p_long"])
                i_ps = int(required["p_short"])
                i_pf = int(required["p_flat"])
                i_margin = int(required["margin_top1_top2"])
                i_unc = int(required["uncertainty_score"])
                i_phat = int(required["p_hat"])
                i_slope = int(required["rolling_slope_since_entry"])
                for i, rec in enumerate(recs):
                    io = rec.get("io_features") or (rec.get("io") or {}).get("io_features")
                    if io is None:
                        continue
                    arr = np.asarray(io, dtype=np.float32)
                    row = arr[-1] if arr.ndim == 2 else arr if arr.ndim == 1 else None
                    if row is None or int(row.shape[-1]) <= max(i_pl, i_ps, i_pf):
                        continue
                    p_long_i = float(row[i_pl])
                    p_short_i = float(row[i_ps])
                    p_flat_i = float(row[i_pf])
                    margin_i = float(row[i_margin])
                    uncertainty_i = float(row[i_unc])
                    p_hat_i = float(row[i_phat])
                    slope_i = float(row[i_slope])
                    if not all(
                        math.isfinite(v)
                        for v in (p_long_i, p_short_i, p_flat_i, margin_i, uncertainty_i, p_hat_i, slope_i)
                    ):
                        continue
                    directional_edge_sig[i] = float(p_short_i - p_long_i) if is_short else float(p_long_i - p_short_i)
                    p_flat_sig[i] = float(p_flat_i)
                    margin_sig[i] = float(margin_i)
                    uncertainty_sig[i] = float(uncertainty_i)
                    p_hat_sig[i] = float(p_hat_i)
                    slope_sig[i] = float(slope_i)
        except Exception:
            pass

        minutes_to_boundary = np.array(
            [
                _get_scalar_optional(r, "minutes_to_next_session_boundary")
                if _get_scalar_optional(r, "minutes_to_next_session_boundary") is not None
                else np.nan
                for r in recs
            ],
            dtype=np.float32,
        )

        bars_per_trade.append(len(recs))
        for r in recs:
            r["should_exit"] = 0.0
            r["profit_protect_should_exit"] = 0.0
            r["profit_protect_train_mask"] = 0.0
            r["sample_weight"] = 1.0
            r["exit_label_family"] = "NEGATIVE"
            r["exit_aux_family"] = "NEGATIVE"
            r["profit_head_bucket"] = "OTHER"

        n = len(recs)
        if intraday_h is None:
            future_max = np.maximum.accumulate(pnl[::-1])[::-1]
        else:
            future_max = np.zeros(n, dtype=np.float32)
            for i in range(n):
                end = min(n, i + intraday_h + 1)
                future_max[i] = float(np.nanmax(pnl[i:end])) if end > i else float(pnl[i])

        # ── V3 dip-head targets (2026-05-26): forward recovery within K M1 bars ──
        # recovery_K[i] = max(pnl[i+1 .. i+K]) − pnl[i], clipped ≥0. Teaches the
        # exit dip-head whether a current drawdown recovers → don't exit at a dip
        # bottom. Pinball loss turns it into recovery_p50/p10 (downside) quantiles.
        # O(K)-vectorized per trade; gated by enable_dip_head at train time.
        for _Kdip in (12, 48, 144):
            _fwd = np.full(n, -np.inf, dtype=np.float64)
            for _k in range(1, _Kdip + 1):
                if n > _k:
                    _sh = np.full(n, -np.inf, dtype=np.float64)
                    _sh[: n - _k] = pnl[_k:]
                    _fwd = np.maximum(_fwd, _sh)
            _rec = np.where(np.isfinite(_fwd), _fwd - pnl, 0.0)
            _rec = np.clip(_rec, 0.0, 1000.0)
            for _i in range(n):
                recs[_i][f"y_dip_recovery_K{_Kdip}"] = float(_rec[_i])

        # ── V3 NEW-head targets (2026-05-26) — EXIT-ORIENTED, forward-from-in-trade ──
        # All on the trade's own side-signed pnl trajectory. Codes "the market has
        # turned against the open position → take profit". Horizons match dip (M1).
        #   forecast_K   = pnl[i+K] − pnl[i]                 (<0 → market moving against us)
        #   tail_mae_K   = pnl[i] − min(pnl[i+1..i+K])       (worst further drawdown ahead)
        #   t_recov_frac = argmax(pnl[i+1..i+K]) / K          (WHEN the forward peak hits)
        #   t_worst_frac = argmin(pnl[i+1..i+K]) / K          (WHEN the worst-adverse hits)
        #   vol_fwd_K    = std(Δpnl over next K) (bps)        (forward realized vol)
        # Gated by enable_*_head at train time; pinball(tail)/smooth_l1(rest) in trainer.
        _pnl64 = pnl.astype(np.float64)
        for _K in (12, 48, 144):
            _fmax = np.full(n, -np.inf); _fmin = np.full(n, np.inf)
            _amax = np.zeros(n); _amin = np.zeros(n); _kth = np.full(n, np.nan)
            for _k in range(1, _K + 1):
                if n > _k:
                    _sh = np.full(n, np.nan, dtype=np.float64); _sh[: n - _k] = _pnl64[_k:]
                    _fin = np.isfinite(_sh)
                    _amax = np.where(_fin & (_sh > _fmax), float(_k), _amax)
                    _fmax = np.where(_fin, np.maximum(_fmax, _sh), _fmax)
                    _amin = np.where(_fin & (_sh < _fmin), float(_k), _amin)
                    _fmin = np.where(_fin, np.minimum(_fmin, _sh), _fmin)
                    if _k == _K:
                        _kth = _sh
            _fc = np.where(np.isfinite(_kth), _kth - _pnl64, 0.0)
            _tail = np.where(np.isfinite(_fmin), _pnl64 - _fmin, 0.0)
            _ttr = _amax / float(_K); _ttw = _amin / float(_K)
            for _i in range(n):
                recs[_i][f"y_v3_forecast_K{_K}"] = float(np.clip(np.nan_to_num(_fc[_i]), -1000.0, 1000.0))
                recs[_i][f"y_v3_tail_mae_K{_K}"] = float(np.clip(_tail[_i], 0.0, 1000.0))
                recs[_i][f"y_v3_time_to_recovery_frac_K{_K}"] = float(np.clip(_ttr[_i], 0.0, 1.0))
                recs[_i][f"y_v3_time_to_worst_frac_K{_K}"] = float(np.clip(_ttw[_i], 0.0, 1.0))
        # forward realized vol of per-bar pnl changes (direction-agnostic)
        _dpnl = np.diff(_pnl64, prepend=_pnl64[:1] if n else np.zeros(0, dtype=np.float64))
        _svol = pd.Series(_dpnl)
        for _K in (12, 48, 144):
            _fstd = (_svol.rolling(_K).std().shift(-_K)).to_numpy()
            _fstd = np.clip(np.nan_to_num(_fstd, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1000.0)
            for _i in range(n):
                recs[_i][f"y_v3_vol_fwd_K{_K}"] = float(_fstd[_i])

        main_pos_count = 0
        profit_pos_count = 0
        det_count = 0
        failfast_trigger_idx: Optional[int] = None
        if failfast:
            for i in range(n):
                cur = float(pnl[i])
                if math.isfinite(cur) and cur <= failfast_loss_bps:
                    failfast_trigger_idx = i
                    break

        for i in range(n):
            if bars[i] < min_hold_bars:
                continue
            cur = float(pnl[i])
            fmax = float(future_max[i])
            if not (math.isfinite(cur) and math.isfinite(fmax)):
                continue
            de_now = float("nan")
            p_flat_now = float("nan")
            margin_now = float("nan")
            unc_now = float("nan")
            capture_ratio = None
            is_capture = False
            if enable_capture_label and fmax > 0.0:
                capture_ratio = cur / fmax
                if math.isfinite(capture_ratio):
                    threshold = capture_ratio_threshold
                    if intraday_h is not None and float(bars[i]) >= intraday_h:
                        threshold = max(0.0, threshold - late_capture_threshold_drop)
                    is_capture = capture_ratio >= threshold

            late_exit = False
            if intraday_h is not None and float(bars[i]) >= intraday_h and math.isfinite(cur) and cur >= late_exit_pnl_floor:
                late_exit = True

            boundary_exit = False
            if intraday_h is not None:
                mins = float(minutes_to_boundary[i]) if math.isfinite(float(minutes_to_boundary[i])) else math.nan
                if math.isfinite(mins) and mins <= boundary_minutes and math.isfinite(cur) and cur >= boundary_pnl_floor:
                    boundary_exit = True

            failfast_exit = False
            if failfast and intraday_h is not None and failfast_trigger_idx is not None:
                if i >= (failfast_trigger_idx + failfast_grace_bars) and float(bars[i]) <= intraday_h:
                    failfast_exit = True

            det = False
            det_reason: Optional[str] = None
            profit_train_mask = False
            profit_positive = False
            try:
                mfe_now = float(mfe[i])
                dd_now = float(distance_from_peak[i])
                gb_now = float(giveback_ratio[i])
                tsm_now = float(time_since_mfe[i])
                if (
                    math.isfinite(mfe_now)
                    and math.isfinite(dd_now)
                    and math.isfinite(gb_now)
                    and math.isfinite(tsm_now)
                ):
                    mfe_min = float(postprog_mfe_min_short if is_short else postprog_mfe_min_long)
                    dd_min = float(postprog_dd_min_short if is_short else postprog_dd_min_long)
                    gb_min = float(postprog_giveback_ratio_min_short if is_short else postprog_giveback_ratio_min_long)
                    pnl_frac_floor = float(postprog_pnl_frac_floor_short if is_short else postprog_pnl_frac_floor_long)
                    pnl_abs_floor = float(postprog_pnl_abs_floor_short if is_short else postprog_pnl_abs_floor_long)
                    de_now = float(directional_edge_sig[i]) if i < len(directional_edge_sig) else float("nan")
                    p_flat_now = float(p_flat_sig[i]) if i < len(p_flat_sig) else float("nan")
                    margin_now = float(margin_sig[i]) if i < len(margin_sig) else float("nan")
                    unc_now = float(uncertainty_sig[i]) if i < len(uncertainty_sig) else float("nan")
                    p_hat_now = float(p_hat_sig[i]) if i < len(p_hat_sig) else float("nan")
                    slope_now = float(slope_sig[i]) if i < len(slope_sig) else float("nan")
                    if not math.isfinite(slope_now):
                        slope_now_opt = _get_scalar_optional(recs[i], "rolling_slope_since_entry")
                        slope_now = float(slope_now_opt) if slope_now_opt is not None else 0.0
                    has_progress = mfe_now >= mfe_min
                    has_actual_giveback = gb_now >= float(postprog_giveback_ratio_floor)
                    pnl_frac_limit = float(mfe_now * pnl_frac_floor)
                    near_flat_or_red_limit = float(min(pnl_abs_floor, pnl_frac_limit))
                    future_headroom = float(max(0.0, fmax - cur))
                    future_headroom_frac = float(future_headroom / mfe_now) if mfe_now > 0.0 else float("inf")
                    signal_not_supportive = (
                        math.isfinite(de_now)
                        and de_now <= float(signal_not_supportive_de_max)
                        and (
                            (math.isfinite(p_flat_now) and p_flat_now >= float(signal_not_supportive_pflat_min))
                            or (math.isfinite(margin_now) and margin_now <= float(signal_not_supportive_margin_max))
                            or (math.isfinite(unc_now) and unc_now >= float(signal_not_supportive_unc_min))
                            or (math.isfinite(p_hat_now) and p_hat_now <= float(signal_not_supportive_phat_max))
                        )
                    )
                    signal_dead_strong = (
                        math.isfinite(de_now)
                        and de_now <= float(signal_dead_strong_de_max)
                        and (
                            (math.isfinite(p_flat_now) and p_flat_now >= float(signal_dead_strong_pflat_min))
                            or (math.isfinite(margin_now) and margin_now <= float(signal_dead_strong_margin_max))
                            or (math.isfinite(unc_now) and unc_now >= float(signal_dead_strong_unc_min))
                        )
                    )
                    hard_deterioration = (
                        has_progress
                        and has_actual_giveback
                        and signal_dead_strong
                        and (
                            (dd_now >= (dd_min + 5.0) and gb_now >= (gb_min + 0.10))
                            or (dd_now >= max(12.0, dd_min * 0.8) and gb_now >= 0.85)
                        )
                    )
                    break_even_red_pre = (
                        has_progress
                        and has_actual_giveback
                        and signal_not_supportive
                        and cur <= near_flat_or_red_limit
                    )
                    break_even_red = (
                        break_even_red_pre
                        and signal_dead_strong
                        and math.isfinite(de_now)
                        and de_now <= be_de_max
                    )
                    if break_even_red and be_require_signal_extreme:
                        break_even_red = (
                            (math.isfinite(p_flat_now) and p_flat_now >= 0.50)
                            or (math.isfinite(margin_now) and margin_now <= 0.10)
                            or (math.isfinite(unc_now) and unc_now >= 0.62)
                        )
                    profit_mfe_min = float(profit_protect_mfe_min_short if is_short else profit_protect_mfe_min_long)
                    profit_dd_min = max(float(profit_protect_dd_min_bps), float(dd_min + profit_dd_extra_bps))
                    profit_gb_min = max(float(profit_protect_giveback_ratio_min), float(gb_min + profit_gb_extra))
                    profit_pnl_cap = min(float(profit_protect_pnl_max_bps), float(mfe_now * profit_protect_pnl_frac_mfe_max))
                    profit_state = (
                        has_progress
                        and has_actual_giveback
                        and mfe_now >= profit_mfe_min
                        and dd_now >= profit_dd_min
                        and gb_now >= profit_gb_min
                        and tsm_now >= float(profit_protect_time_since_mfe_min_bars)
                        and cur >= float(profit_protect_pnl_min_bps)
                        and cur <= profit_pnl_cap
                    )
                    no_edge_adverse_collapse = (
                        bool(no_edge_adverse_enabled)
                        and (not is_short)
                        and mfe_now <= float(no_edge_adverse_mfe_cap_bps)
                        and dd_now >= float(no_edge_adverse_dd_min_bps)
                        and cur <= float(no_edge_adverse_pnl_min_bps)
                        and slope_now <= float(no_edge_adverse_slope_max)
                        and math.isfinite(de_now)
                        and de_now <= 0.02
                        and math.isfinite(p_flat_now)
                        and p_flat_now <= float(no_edge_adverse_pflat_max)
                        and (
                            (math.isfinite(margin_now) and margin_now <= float(no_edge_adverse_margin_max))
                            or (math.isfinite(unc_now) and unc_now >= float(no_edge_adverse_unc_min))
                        )
                    )
                    if no_edge_adverse_collapse:
                        no_edge_adverse_collapse_seen += 1
                        no_edge_pos_pre_strict += 1
                    if break_even_red_pre:
                        be_pos_pre_strict += 1

                    profit_signal_not_supportive = (
                        math.isfinite(de_now)
                        and de_now <= 0.05
                        and (
                            (math.isfinite(p_flat_now) and p_flat_now >= float(profit_protect_mask_signal_pflat_min))
                            or (math.isfinite(unc_now) and unc_now >= float(profit_protect_mask_signal_unc_min))
                            or (math.isfinite(margin_now) and margin_now <= float(profit_protect_mask_signal_margin_max))
                        )
                    )
                    break_even_red_floor = bool(
                        break_even_red_pre
                        and (
                            future_headroom <= float(profit_protect_future_headroom_abs_max_bps)
                            and future_headroom_frac <= float(profit_protect_future_headroom_frac_max)
                        )
                    )
                    profit_giveback = bool(profit_state and profit_signal_not_supportive)
                    if profit_giveback and (
                        future_headroom > float(profit_protect_future_headroom_abs_max_bps)
                        or future_headroom_frac > float(profit_protect_future_headroom_frac_max)
                    ):
                        profit_giveback = False
                        profit_protect_future_recovery_suppressed += 1

                    normal_threshold_mass = bool(
                        has_progress
                        and cur > profit_pnl_cap
                        and dd_now < profit_dd_min
                        and gb_now < profit_gb_min
                    )
                    if has_progress and has_actual_giveback and normal_threshold_mass:
                        profit_protect_normal_threshold_suppressed += 1
                    profit_train_mask = bool(profit_state)
                    profit_positive = bool(profit_train_mask and (profit_giveback or break_even_red_floor))
                    if weekly_teacher_mode and bool(teacher_state.get("available")):
                        if not bool(teacher_state.get("profit_eligible")):
                            profit_train_mask = False
                            profit_positive = False
                        elif bool(teacher_state.get("good_close")):
                            profit_train_mask = bool(profit_state)
                            profit_positive = False
                            recs[i]["sample_weight"] = float(
                                max(float(recs[i].get("sample_weight", 1.0)), float(weekly_teacher_good_hold_weight))
                            )
                        elif bool(teacher_state.get("bad_rot")):
                            profit_train_mask = bool(profit_state)
                            profit_positive = bool(profit_train_mask and (profit_giveback or break_even_red_floor))
                            if profit_positive:
                                recs[i]["sample_weight"] = float(
                                    max(float(recs[i].get("sample_weight", 1.0)), float(weekly_teacher_positive_weight))
                                )

                    det_pre_strict = hard_deterioration or break_even_red_pre or no_edge_adverse_collapse
                    should_label_pre_be_strict = bool(det_pre_strict or late_exit or boundary_exit or failfast_exit or is_capture)
                    if should_label_pre_be_strict:
                        total_pos_pre_be_strict += 1
                    det = hard_deterioration or break_even_red or no_edge_adverse_collapse
                    if break_even_red:
                        be_pos_post_strict += 1

                    pre_suppress_reason: Optional[str] = None
                    if det:
                        if hard_deterioration:
                            pre_suppress_reason = "HARD_DETERIORATION_EDGE_DEATH"
                        elif break_even_red:
                            pre_suppress_reason = "BREAK_EVEN_RED_PROTECTION"
                        else:
                            pre_suppress_reason = "NO_EDGE_ADVERSE_COLLAPSE"
                    if det:
                        weak_misalign_band = math.isfinite(de_now) and (-0.03 < de_now <= 0.02)
                        flat_dominant = math.isfinite(p_flat_now) and p_flat_now >= 0.44
                        moderate_deterioration = dd_now <= 10.0 and gb_now <= 0.65
                        still_above_near_flat_floor = cur > near_flat_or_red_limit
                        indecision_dominant = (
                            (math.isfinite(p_flat_now) and p_flat_now >= 0.42)
                            or (math.isfinite(unc_now) and unc_now >= 0.58)
                        )
                        if weak_misalign_band and flat_dominant and moderate_deterioration and still_above_near_flat_floor:
                            det = False
                            det_reason = None
                            det_suppressed_early_rest_signature += 1
                        elif (
                            (math.isfinite(de_now) and (-0.03 < de_now <= suppress_weak_band_max))
                            and ((math.isfinite(p_flat_now) and p_flat_now >= suppress_flat_min) or indecision_dominant)
                            and dd_now <= suppress_mild_dd_max
                            and gb_now <= suppress_mild_gb_max
                            and still_above_near_flat_floor
                        ):
                            det = False
                            det_reason = None
                            det_suppressed_near_early_family_total += 1
                            if pre_suppress_reason:
                                det_suppressed_near_early_family_by_reason[pre_suppress_reason] = int(
                                    det_suppressed_near_early_family_by_reason.get(pre_suppress_reason, 0) + 1
                                )
                    if det:
                        if hard_deterioration:
                            det_reason = "HARD_DETERIORATION_EDGE_DEATH"
                        elif break_even_red:
                            det_reason = "BREAK_EVEN_RED_PROTECTION"
                        else:
                            det_reason = "NO_EDGE_ADVERSE_COLLAPSE"
                    if det and det_count >= det_max_per_trade:
                        det = False
                        det_reason = None
                    # Early suppression operates on the pre-strict deterministic label state,
                    # before the final main_reason is derived below.
                    should_label = bool(should_label_pre_be_strict)
            except Exception:
                det = False
                det_reason = None
                should_label = False
                profit_train_mask = False
                profit_positive = False
                hard_deterioration = False
                break_even_red_pre = False
                no_edge_adverse_collapse = False
                normal_threshold_mass = False

            weak_directional_long = math.isfinite(de_now) and de_now <= 0.05
            indecision_like = (
                (math.isfinite(p_flat_now) and p_flat_now >= 0.28)
                or (math.isfinite(unc_now) and unc_now >= 0.54)
                or (math.isfinite(margin_now) and margin_now <= 0.12)
            )
            early_long_recovery_signature = (
                (not is_short)
                and float(bars[i]) <= float(early_long_recovery_max_bars)
                and float(mfe[i]) <= float(early_long_recovery_mfe_cap_bps)
                and cur < 0.0
                and (
                    fmax >= float(early_long_recovery_future_max_bps)
                    or (weak_directional_long and indecision_like)
                )
            )
            if early_long_recovery_signature:
                early_long_recovery_signature_seen += 1
                if should_label:
                    early_long_recovery_signature_positive_pre += 1
                    det = False
                    det_reason = None
                    late_exit = False
                    boundary_exit = False
                    failfast_exit = False
                    is_capture = False
                    should_label = False
                    early_long_recovery_signature_positive_suppressed += 1
                profit_train_mask = False
                profit_positive = False
                recs[i]["sample_weight"] = float(max(float(recs[i].get("sample_weight", 1.0)), float(early_long_recovery_negative_weight)))
                early_long_recovery_signature_downweighted += 1

            main_reason: Optional[str] = None
            if bool(main_use_profit_positive) and profit_positive and math.isfinite(cur) and cur >= float(main_profit_min_pnl_bps):
                main_reason = "PROFIT_GIVEBACK_PROTECTION"
            elif bool(main_allow_hard_deterioration) and hard_deterioration and math.isfinite(cur) and cur >= float(main_hard_det_min_pnl_bps):
                main_reason = "HARD_DETERIORATION_EDGE_DEATH"
            elif bool(main_allow_break_even_red) and break_even_red and math.isfinite(cur) and cur >= float(main_break_even_min_pnl_bps):
                main_reason = "BREAK_EVEN_RED_PROTECTION"
            elif bool(main_allow_no_edge_adverse) and no_edge_adverse_collapse and math.isfinite(cur) and cur >= float(main_no_edge_min_pnl_bps):
                main_reason = "NO_EDGE_ADVERSE_COLLAPSE"
            elif bool(main_allow_late_exit) and late_exit:
                main_reason = "OTHER_DETERMINISTIC"
            elif bool(main_allow_boundary_exit) and boundary_exit:
                main_reason = "OTHER_DETERMINISTIC"
            elif bool(main_allow_failfast_exit) and failfast_exit:
                main_reason = "OTHER_DETERMINISTIC"
            elif bool(main_allow_capture) and is_capture:
                main_reason = "OTHER_DETERMINISTIC"

            should_label = bool(main_reason)
            if should_label:
                total_pos_post_be_strict += 1

            recs[i]["profit_protect_train_mask"] = 1.0 if profit_train_mask else 0.0
            recs[i]["profit_protect_should_exit"] = 1.0 if profit_positive else 0.0
            recs[i]["exit_label_family"] = "PROFIT_GIVEBACK_PROTECTION" if profit_positive else recs[i].get("exit_label_family", "NEGATIVE")
            if profit_positive:
                if break_even_red_floor and not profit_giveback:
                    recs[i]["profit_head_bucket"] = "BREAK_EVEN_RED_LIKE"
                else:
                    recs[i]["profit_head_bucket"] = "PROFIT_PROTECT_LIKE"
            elif hard_deterioration:
                recs[i]["profit_head_bucket"] = "HARD_DETERIORATION_LIKE"
            elif break_even_red_pre:
                recs[i]["profit_head_bucket"] = "BREAK_EVEN_RED_LIKE"
            elif no_edge_adverse_collapse:
                recs[i]["profit_head_bucket"] = "EARLY_WEAK_ADVERSE"
            elif normal_threshold_mass:
                recs[i]["profit_head_bucket"] = "NORMAL_THRESHOLD_MASS"
            else:
                recs[i]["profit_head_bucket"] = "OTHER"
            if profit_train_mask:
                n_profit_mask += 1
            if profit_positive:
                profit_pos_count += 1
                n_profit_pos += 1
                recs[i]["exit_label_family"] = "PROFIT_GIVEBACK_PROTECTION"

            if should_label:
                recs[i]["should_exit"] = 1.0
                recs[i]["exit_label_family"] = str(main_reason or "OTHER_DETERMINISTIC")
                recs[i]["exit_aux_family"] = str(main_reason or "NEGATIVE")
                main_pos_count += 1
                n_main_pos += 1
                pos_bars.append(float(bars[i]))
                if is_capture and capture_ratio is not None:
                    n_pos_capture += 1
                    capture_ratios.append(float(capture_ratio))
                if str(main_reason or "").upper() in {
                    "HARD_DETERIORATION_EDGE_DEATH",
                    "BREAK_EVEN_RED_PROTECTION",
                    "NO_EDGE_ADVERSE_COLLAPSE",
                    "PROFIT_GIVEBACK_PROTECTION",
                }:
                    n_main_pos_deterioration += 1
                    if str(main_reason) != "PROFIT_GIVEBACK_PROTECTION":
                        det_count += 1
                    if main_reason:
                        det_reason_counts[str(main_reason)] = int(det_reason_counts.get(str(main_reason), 0) + 1)
                        if main_reason == "NO_EDGE_ADVERSE_COLLAPSE":
                            no_edge_adverse_collapse_pos += 1
                elif late_exit or boundary_exit or failfast_exit or is_capture:
                    det_reason_counts["OTHER_DETERMINISTIC"] = int(det_reason_counts.get("OTHER_DETERMINISTIC", 0) + 1)
                if is_capture and det:
                    n_pos_both += 1

            if not should_label and not profit_positive:
                recs[i]["exit_label_family"] = "NEGATIVE"

        if main_pos_count > 0:
            n_trades_with_main_pos += 1
        if profit_pos_count > 0:
            n_trades_with_profit_pos += 1
        pos_counts_per_trade.append(int(main_pos_count))
        det_counts_per_trade.append(int(det_count))

        for src, new in zip(trades[tid], recs):
            src["should_exit"] = new.get("should_exit", 0.0)
            src["sample_weight"] = new.get("sample_weight", 1.0)
            src["profit_protect_should_exit"] = new.get("profit_protect_should_exit", 0.0)
            src["profit_protect_train_mask"] = new.get("profit_protect_train_mask", 0.0)
            src["exit_label_family"] = new.get("exit_label_family", "NEGATIVE")
            src["exit_aux_family"] = new.get("exit_aux_family", "NEGATIVE")
            src["profit_head_bucket"] = new.get("profit_head_bucket", "OTHER")

    total_records = len(records)
    n_trades = len(trades)
    n_main_positive = sum(1 for r in records if r.get("should_exit", 0) == 1.0)
    n_profit_positive = sum(1 for r in records if r.get("profit_protect_should_exit", 0) == 1.0)
    n_profit_mask_total = sum(1 for r in records if r.get("profit_protect_train_mask", 0) == 1.0)
    exit_rate = n_main_positive / total_records if total_records else 0.0
    profit_rate = n_profit_positive / total_records if total_records else 0.0
    profit_mask_rate = n_profit_mask_total / total_records if total_records else 0.0
    avg_pos_per_pos_trade = (n_main_positive / n_trades_with_main_pos) if n_trades_with_main_pos else 0.0
    bars_min = min(bars_per_trade) if bars_per_trade else 0
    bars_med = int(np.median(bars_per_trade)) if bars_per_trade else 0
    bars_max = max(bars_per_trade) if bars_per_trade else 0
    pos_bars_min = min(pos_bars) if pos_bars else 0.0
    pos_bars_med = float(np.median(pos_bars)) if pos_bars else 0.0
    pos_bars_max = max(pos_bars) if pos_bars else 0.0
    capture_ratio_med = float(np.median(capture_ratios)) if capture_ratios else 0.0
    pos_counts_med = float(np.median(pos_counts_per_trade)) if pos_counts_per_trade else 0.0
    pos_counts_p90 = float(np.quantile(pos_counts_per_trade, 0.9)) if pos_counts_per_trade else 0.0
    pos_counts_p99 = float(np.quantile(pos_counts_per_trade, 0.99)) if pos_counts_per_trade else 0.0
    det_counts_med = float(np.median(det_counts_per_trade)) if det_counts_per_trade else 0.0
    det_counts_p90 = float(np.quantile(det_counts_per_trade, 0.9)) if det_counts_per_trade else 0.0

    log.info(
        "[EXIT_LABELER_PROOF] io=%s total_records=%d n_trades=%d n_trades_with_main_pos=%d n_main_pos=%d exit_rate=%.6f avg_pos_per_pos_trade=%.3f pct_trades_with_main_pos=%.3f bars_per_trade(min/med/max)=%d/%d/%d pos_bars(min/med/max)=%.2f/%.2f/%.2f pos_counts(p50/p90/p99)=%.2f/%.2f/%.2f capture_ratio_med=%.3f capture_ratio_threshold=%.2f det_pos=%d det_pos_rate=%.6f det_counts(p50/p90)=%.2f/%.2f min_hold_bars=%d",
        active_io_version,
        total_records,
        n_trades,
        n_trades_with_main_pos,
        n_main_positive,
        exit_rate,
        avg_pos_per_pos_trade,
        (n_trades_with_main_pos / n_trades) if n_trades else 0.0,
        bars_min,
        bars_med,
        bars_max,
        pos_bars_min,
        pos_bars_med,
        pos_bars_max,
        pos_counts_med,
        pos_counts_p90,
        pos_counts_p99,
        capture_ratio_med,
        capture_ratio_threshold,
        n_main_pos_deterioration,
        (n_main_pos_deterioration / total_records) if total_records else 0.0,
        det_counts_med,
        det_counts_p90,
        min_hold_bars,
    )
    log.info(
        "[EXIT_LABELER_PROFIT_HEAD_PROOF] profit_pos=%d profit_rate=%.6f profit_mask=%d profit_mask_rate=%.6f trades_with_profit_pos=%d",
        n_profit_positive,
        profit_rate,
        n_profit_mask_total,
        profit_mask_rate,
        n_trades_with_profit_pos,
    )
    log.info(
        "[EXIT_LABELER_DET_PROOF] capture_pos=%d det_pos=%d both=%d family_counts=%s det_suppressed_early_rest_signature=%d det_suppressed_near_early_family_total=%d det_suppressed_near_early_family_by_reason=%s profit_protect_future_recovery_suppressed=%d profit_protect_normal_threshold_suppressed=%d early_long_recovery_signature={seen=%d,pos_pre=%d,pos_suppressed=%d,downweighted=%d,max_bars=%.1f,mfe_cap_bps=%.1f,future_max_bps=%.1f,neg_weight=%.2f} be_strict_proof={be_pos_pre=%d,be_pos_post=%d,total_pos_pre=%d,total_pos_post=%d} no_edge_adverse_collapse={seen=%d,pos_pre=%d,pos=%d,mfe_cap=%.1f,dd_min=%.1f,pnl_min=%.1f,slope_max=%.1f,pflat_max=%.2f,margin_max=%.2f,unc_min=%.2f} edge_family_thresholds={long:{mfe_bps=%.1f,dd_bps=%.1f,gb=%.2f,pnl_frac=%.2f,pnl_abs=%.1f},short:{mfe_bps=%.1f,dd_bps=%.1f,gb=%.2f,pnl_frac=%.2f,pnl_abs=%.1f},gb_floor=%.2f} profit_protect_thresholds={long_mfe=%.1f,short_mfe=%.1f,dd_bps=%.1f,gb=%.2f,pnl_min=%.1f,pnl_max=%.1f,pnl_frac_max=%.2f,time_since_mfe=%.1f,future_headroom_abs=%.1f,future_headroom_frac=%.2f} max_per_trade=%d",
        n_pos_capture,
        n_main_pos_deterioration,
        n_pos_both,
        det_reason_counts,
        int(det_suppressed_early_rest_signature),
        int(det_suppressed_near_early_family_total),
        det_suppressed_near_early_family_by_reason,
        int(profit_protect_future_recovery_suppressed),
        int(profit_protect_normal_threshold_suppressed),
        int(early_long_recovery_signature_seen),
        int(early_long_recovery_signature_positive_pre),
        int(early_long_recovery_signature_positive_suppressed),
        int(early_long_recovery_signature_downweighted),
        float(early_long_recovery_max_bars),
        float(early_long_recovery_mfe_cap_bps),
        float(early_long_recovery_future_max_bps),
        float(early_long_recovery_negative_weight),
        int(be_pos_pre_strict),
        int(be_pos_post_strict),
        int(total_pos_pre_be_strict),
        int(total_pos_post_be_strict),
        int(no_edge_adverse_collapse_seen),
        int(no_edge_pos_pre_strict),
        int(no_edge_adverse_collapse_pos),
        float(no_edge_adverse_mfe_cap_bps),
        float(no_edge_adverse_dd_min_bps),
        float(no_edge_adverse_pnl_min_bps),
        float(no_edge_adverse_slope_max),
        float(no_edge_adverse_pflat_max),
        float(no_edge_adverse_margin_max),
        float(no_edge_adverse_unc_min),
        postprog_mfe_min_long,
        postprog_dd_min_long,
        postprog_giveback_ratio_min_long,
        postprog_pnl_frac_floor_long,
        postprog_pnl_abs_floor_long,
        postprog_mfe_min_short,
        postprog_dd_min_short,
        postprog_giveback_ratio_min_short,
        postprog_pnl_frac_floor_short,
        postprog_pnl_abs_floor_short,
        postprog_giveback_ratio_floor,
        profit_protect_mfe_min_long,
        profit_protect_mfe_min_short,
        profit_protect_dd_min_bps,
        profit_protect_giveback_ratio_min,
        profit_protect_pnl_min_bps,
        profit_protect_pnl_max_bps,
        profit_protect_pnl_frac_mfe_max,
        profit_protect_time_since_mfe_min_bars,
        profit_protect_future_headroom_abs_max_bps,
        profit_protect_future_headroom_frac_max,
        det_max_per_trade,
    )
    log.info(
        "[EXIT_LABELER_WEEKLY_TEACHER_PROOF] enabled=%s trade_available=%d trade_profit_eligible=%d trade_good_close=%d trade_bad_rot=%d teacher={profit_mfe_min=%.1f,pos_pnl_min=%.1f,pos_retention_min=%.2f,pos_headroom_max=%.1f,rot_retention_max=%.2f,rot_pnl_max=%.1f,pos_weight=%.2f,good_hold_weight=%.2f}",
        bool(weekly_teacher_mode),
        int(teacher_trade_available),
        int(teacher_trade_profit_eligible),
        int(teacher_trade_good_close),
        int(teacher_trade_bad_rot),
        float(teacher_trade_profit_mfe_min),
        float(teacher_positive_close_min_pnl_bps),
        float(teacher_positive_retention_min),
        float(teacher_positive_post_exit_headroom_max_bps),
        float(teacher_rot_retention_max),
        float(teacher_rot_final_pnl_max_bps),
        float(weekly_teacher_positive_weight),
        float(weekly_teacher_good_hold_weight),
    )
    if exit_rate < 0.02:
        log.warning("[EXIT_LABELER_TOO_SPARSE] exit_rate=%.6f", exit_rate)
    if exit_rate > 0.20:
        log.warning("[EXIT_LABELER_TOO_DENSE] exit_rate=%.6f", exit_rate)
    if exit_rate > 0.5:
        log.warning("[EXIT_LABELER_PROOF] exit_rate high (%.4f); labels may be too dense", exit_rate)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
if TORCH_AVAILABLE:

    class ExitTransformerV0(nn.Module):
        """Small transformer encoder: input (B, T, D) -> EXIT prob (B,).

        V12.2 multi-TF mode: when enable_multi_tf=True, the model takes 5
        additional per-TF inputs (M5/M15/H1/H4/D1) and fuses them with the
        base M1 sequence's last token via a second-stage MLP. When disabled
        (default), state_dict matches v8 exactly so existing v8 bundles load
        with strict=True.
        """

        def __init__(
            self,
            input_dim: int = EXIT_IO_FEATURE_COUNT,
            window_len: int = DEFAULT_WINDOW_LEN,
            d_model: int = 64,
            n_heads: int = DEFAULT_N_HEADS,
            n_layers: int = DEFAULT_N_LAYERS,
            dropout: float = 0.1,
            family_head_dim: int = len(EXIT_AUX_FAMILY_NAMES),
            profit_protect_head_dim: int = 1,
            # ── V12.2 multi-TF (default OFF — v8-bit-identical) ──
            # When enabled, model expects all 5 TF tensors in forward().
            # Base sequence is M1 (window_len=512 by default), so M5 is added
            # here as a NEW stream (not present in V10 multi-TF since V10's
            # base IS M5). Coverage at default lengths: M5=8h, M15=24h,
            # H1=4d, H4=16d, D1=~3mo.
            enable_multi_tf: bool = False,
            m5_seq_dim: int = 0,
            m15_seq_dim: int = 0,
            h1_seq_dim: int = 0,
            h4_seq_dim: int = 0,
            d1_seq_dim: int = 0,
            m5_seq_len: int = 96,
            m15_seq_len: int = 96,
            h1_seq_len: int = 96,
            h4_seq_len: int = 96,
            d1_seq_len: int = 96,
            multi_tf_num_layers: int = 2,
            multi_tf_scale: float = 0.5,
            # ── Distillation Q-head (V13 prep) ────────────────────────
            # When enabled, adds nn.Linear(d_model, 2) producing q_per_action
            # mirroring Exit-IQL Q-values [q_hold, q_exit]. Zero-init for
            # stable baseline output. State-dict matches v_FIXED exactly when
            # disabled (no params added).
            enable_q_head: bool = False,
            # ── Positional encoding (temporal order) ──────────────────
            # When enabled, adds sinusoidal positional encoding to the base
            # M1 sequence (and each per-TF stream) BEFORE the transformer.
            # Without it, self-attention is permutation-invariant and the
            # 512-bar window is treated as an unordered set — defeating the
            # purpose of "look 512 bars back in time". Default OFF so older
            # bundles (trained without PE) keep bit-identical forward
            # behaviour; the buffer is persistent=False so state_dict is
            # unchanged either way (strict load stays valid).
            enable_pos_enc: bool = False,
            # ── Dip-analysis head (2026-05-26) ────────────────────────
            # When enabled, adds nn.Linear(d_model, V3_DIP_HEAD_DIM=6) producing a
            # distributional dip/recovery forecast: for each horizon K∈{12,48,144}
            # two pinball quantiles (recovery_p50, recovery_p10) of "how much does
            # the current adverse excursion recover, in bps". Makes the exit
            # transformer EXPLICITLY represent dip-state — don't exit at a dip
            # bottom. Trained via _v3_dip_loss (pinball) in the multitask loop.
            # State-dict matches when off (gated, default OFF).
            enable_dip_head: bool = False,
            # ── New aux heads (2026-05-26) — mirror of V10 entry heads ────
            # timing(6)/tail_risk(3)/vol_forecast(3)/forecast(3), all gated,
            # state-dict-safe, default OFF. See V3_*_HEAD_DIM layout.
            enable_timing_head: bool = False,
            enable_tail_risk_head: bool = False,
            enable_vol_forecast_head: bool = False,
            enable_forecast_head: bool = False,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.window_len = window_len
            self.d_model = d_model
            self.family_head_dim = int(family_head_dim)
            self.profit_protect_head_dim = int(profit_protect_head_dim)
            self.enable_q_head = bool(enable_q_head)
            self.enable_dip_head = bool(enable_dip_head)
            self.enable_timing_head = bool(enable_timing_head)
            self.enable_tail_risk_head = bool(enable_tail_risk_head)
            self.enable_vol_forecast_head = bool(enable_vol_forecast_head)
            self.enable_forecast_head = bool(enable_forecast_head)
            self.proj = nn.Linear(input_dim, d_model)
            enc = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)

            # ── Multi-TF (V12.2) — only built when enabled ──
            self.enable_multi_tf = bool(enable_multi_tf)
            if self.enable_multi_tf:
                if min(m5_seq_dim, m15_seq_dim, h1_seq_dim, h4_seq_dim, d1_seq_dim) <= 0:
                    raise RuntimeError(
                        f"MULTI_TF_DIM_INVALID: all of m5/m15/h1/h4/d1_seq_dim must be >0. "
                        f"Got m5={m5_seq_dim} m15={m15_seq_dim} h1={h1_seq_dim} "
                        f"h4={h4_seq_dim} d1={d1_seq_dim}"
                    )
                self._mtf_dims = {
                    "m5": int(m5_seq_dim), "m15": int(m15_seq_dim),
                    "h1": int(h1_seq_dim), "h4": int(h4_seq_dim), "d1": int(d1_seq_dim),
                }
                self._mtf_lens = {
                    "m5": int(m5_seq_len), "m15": int(m15_seq_len),
                    "h1": int(h1_seq_len), "h4": int(h4_seq_len), "d1": int(d1_seq_len),
                }
                self.m5_proj = nn.Linear(int(m5_seq_dim), d_model)
                self.m15_proj = nn.Linear(int(m15_seq_dim), d_model)
                self.h1_proj = nn.Linear(int(h1_seq_dim), d_model)
                self.h4_proj = nn.Linear(int(h4_seq_dim), d_model)
                self.d1_proj = nn.Linear(int(d1_seq_dim), d_model)
                def _mk_enc():
                    layer = nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                        dropout=dropout, activation="gelu", batch_first=True,
                    )
                    return nn.TransformerEncoder(layer, num_layers=int(multi_tf_num_layers))
                self.m5_encoder = _mk_enc()
                self.m15_encoder = _mk_enc()
                self.h1_encoder = _mk_enc()
                self.h4_encoder = _mk_enc()
                self.d1_encoder = _mk_enc()
                # V12.2 v2: ADDITIVE residual fusion. multi_tf_fuse operates
                # ONLY on multi-TF pools (not on z_v8). Output is a correction
                # added to z_v8 via multi_tf_scale. Zero-init last layer so
                # initial correction ≈ 0 → model starts exactly as v8 baseline.
                self.multi_tf_fuse = nn.Sequential(
                    nn.Linear(5 * d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                )
                nn.init.zeros_(self.multi_tf_fuse[-1].bias)
                nn.init.normal_(self.multi_tf_fuse[-1].weight, std=0.01)
                self.register_buffer("multi_tf_scale", torch.tensor(float(multi_tf_scale)))

            # ── Positional encoding buffers (persistent=False → not in state_dict) ──
            self.enable_pos_enc = bool(enable_pos_enc)
            if self.enable_pos_enc:
                self.register_buffer(
                    "pos_enc", self._sinusoidal_pe(int(window_len), d_model), persistent=False
                )
                if self.enable_multi_tf:
                    for tf in ("m5", "m15", "h1", "h4", "d1"):
                        self.register_buffer(
                            f"pos_enc_{tf}",
                            self._sinusoidal_pe(int(self._mtf_lens[tf]), d_model),
                            persistent=False,
                        )

            self.head = nn.Linear(d_model, 1)
            self.profit_protect_head = nn.Linear(d_model, self.profit_protect_head_dim)
            self.family_head = nn.Linear(d_model, self.family_head_dim)

            # ── Distillation Q-head (V13 prep) ────────────────────────
            # Only instantiated when enable_q_head=True. Zero-init for stable
            # baseline (no Q-signal) until KL-loss pulls it during training.
            if self.enable_q_head:
                self.q_head = nn.Linear(d_model, 2)  # [q_hold, q_exit]
                nn.init.zeros_(self.q_head.weight)
                nn.init.zeros_(self.q_head.bias)

            # ── Dip-analysis head: 6 outputs (K{12,48,144} × {recovery_p50,p10}) ──
            if self.enable_dip_head:
                self.dip_head = nn.Linear(d_model, V3_DIP_HEAD_DIM)
            # ── New aux heads (2026-05-26) — mirror of V10 entry heads ──
            if self.enable_timing_head:
                self.timing_head = nn.Linear(d_model, V3_TIMING_HEAD_DIM)        # 6
            if self.enable_tail_risk_head:
                self.tail_risk_head = nn.Linear(d_model, V3_TAIL_RISK_HEAD_DIM)  # 3
            if self.enable_vol_forecast_head:
                self.vol_forecast_head = nn.Linear(d_model, V3_VOL_FORECAST_HEAD_DIM)  # 3
            if self.enable_forecast_head:
                self.forecast_head = nn.Linear(d_model, V3_FORECAST_HEAD_DIM)    # 3

        @staticmethod
        def _sinusoidal_pe(seq_len: int, d_model: int) -> "torch.Tensor":
            """Standard sinusoidal positional encoding, shape (1, seq_len, d_model)."""
            pe = torch.zeros(int(seq_len), int(d_model))
            position = torch.arange(0, int(seq_len), dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, int(d_model), 2, dtype=torch.float32)
                * (-math.log(10000.0) / float(d_model))
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            n_cos = pe[:, 1::2].size(1)
            pe[:, 1::2] = torch.cos(position * div_term[:n_cos])
            return pe.unsqueeze(0)  # (1, seq_len, d_model)

        def _add_pe(self, t: "torch.Tensor", buf_name: str) -> "torch.Tensor":
            """Add positional encoding (sliced to current seq len) if enabled."""
            if not getattr(self, "enable_pos_enc", False):
                return t
            pe = getattr(self, buf_name)
            return t + pe[:, : t.size(1)]

        def forward(self, x: "torch.Tensor", **mtf_kwargs) -> "torch.Tensor":
            return torch.sigmoid(self.forward_logits(x, **mtf_kwargs))

        def forward_q_per_action(self, x: "torch.Tensor", **mtf_kwargs) -> "torch.Tensor":
            """Returns (B, 2) tensor [q_hold, q_exit]. Only valid if enable_q_head=True."""
            if not self.enable_q_head:
                raise RuntimeError("forward_q_per_action called but enable_q_head=False")
            return self.q_head(self._encode(x, **mtf_kwargs))

        def forward_dip(self, x: "torch.Tensor", **mtf_kwargs) -> "torch.Tensor":
            """Returns (B,) dip/recovery regression. Only valid if enable_dip_head=True."""
            if not self.enable_dip_head:
                raise RuntimeError("forward_dip called but enable_dip_head=False")
            return self.dip_head(self._encode(x, **mtf_kwargs)).squeeze(-1)

        def forward_timing(self, x: "torch.Tensor", **mtf_kwargs) -> "torch.Tensor":
            """Returns (B, V3_TIMING_HEAD_DIM). Only valid if enable_timing_head=True."""
            if not self.enable_timing_head:
                raise RuntimeError("forward_timing called but enable_timing_head=False")
            return self.timing_head(self._encode(x, **mtf_kwargs))

        def forward_tail_risk(self, x: "torch.Tensor", **mtf_kwargs) -> "torch.Tensor":
            """Returns (B, V3_TAIL_RISK_HEAD_DIM). Only valid if enable_tail_risk_head=True."""
            if not self.enable_tail_risk_head:
                raise RuntimeError("forward_tail_risk called but enable_tail_risk_head=False")
            return self.tail_risk_head(self._encode(x, **mtf_kwargs))

        def forward_vol_forecast(self, x: "torch.Tensor", **mtf_kwargs) -> "torch.Tensor":
            """Returns (B, V3_VOL_FORECAST_HEAD_DIM). Only valid if enable_vol_forecast_head=True."""
            if not self.enable_vol_forecast_head:
                raise RuntimeError("forward_vol_forecast called but enable_vol_forecast_head=False")
            return self.vol_forecast_head(self._encode(x, **mtf_kwargs))

        def forward_forecast(self, x: "torch.Tensor", **mtf_kwargs) -> "torch.Tensor":
            """Returns (B, V3_FORECAST_HEAD_DIM). Only valid if enable_forecast_head=True."""
            if not self.enable_forecast_head:
                raise RuntimeError("forward_forecast called but enable_forecast_head=False")
            return self.forecast_head(self._encode(x, **mtf_kwargs))

        def _encode(self, x: "torch.Tensor",
                    seq_m5=None, seq_m15=None, seq_h1=None, seq_h4=None, seq_d1=None
                    ) -> "torch.Tensor":
            h = self.proj(x)
            h = self._add_pe(h, "pos_enc")
            h = self.transformer(h)
            z_v8 = h[:, -1]   # base v8 pooling = last M1 token
            if (self.enable_multi_tf
                    and all(t is not None for t in (seq_m5, seq_m15, seq_h1, seq_h4, seq_d1))):
                m5_pool = self.m5_encoder(self._add_pe(self.m5_proj(seq_m5), "pos_enc_m5")).mean(dim=1)
                m15_pool = self.m15_encoder(self._add_pe(self.m15_proj(seq_m15), "pos_enc_m15")).mean(dim=1)
                h1_pool = self.h1_encoder(self._add_pe(self.h1_proj(seq_h1), "pos_enc_h1")).mean(dim=1)
                h4_pool = self.h4_encoder(self._add_pe(self.h4_proj(seq_h4), "pos_enc_h4")).mean(dim=1)
                d1_pool = self.d1_encoder(self._add_pe(self.d1_proj(seq_d1), "pos_enc_d1")).mean(dim=1)
                # V12.2 v2: ADDITIVE residual. mtf_fuse on multi-TF only;
                # output added to z_v8 with scale.
                mtf_combined = torch.cat(
                    [m5_pool, m15_pool, h1_pool, h4_pool, d1_pool], dim=1,
                )
                mtf_correction = self.multi_tf_fuse(mtf_combined)
                scale = float(self.multi_tf_scale.item())
                return z_v8 + scale * mtf_correction
            return z_v8

        def forward_logits(self, x: "torch.Tensor", **mtf_kwargs) -> "torch.Tensor":
            return self.head(self._encode(x, **mtf_kwargs)).squeeze(-1)

        def forward_profit_protect_logits(self, x: "torch.Tensor", **mtf_kwargs) -> "torch.Tensor":
            return self.profit_protect_head(self._encode(x, **mtf_kwargs)).squeeze(-1)

        def forward_family_logits(self, x: "torch.Tensor", **mtf_kwargs) -> "torch.Tensor":
            return self.family_head(self._encode(x, **mtf_kwargs))

else:
    ExitTransformerV0 = None  # type: ignore


# -----------------------------------------------------------------------------
# Artifact save/load helpers
# -----------------------------------------------------------------------------
def save_exit_transformer_artifacts(
    model: Any,
    out_dir: Path,
    window_len: int,
    d_model: int,
    n_layers: int,
    feature_names_hash: Optional[str] = None,
    input_dim: Optional[int] = None,
    exit_io_version: str = EXIT_IO_VERSION,
) -> Tuple[Path, Path, str]:
    if not TORCH_AVAILABLE or model is None:
        raise RuntimeError("PyTorch and model required to save")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "exit_transformer_v0.pt"
    torch.save(model.state_dict(), model_path)
    with open(model_path, "rb") as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()

    contract = get_exit_io_contract(exit_io_version)
    cfg = {
        "window_len": int(window_len),
        "input_dim": int(input_dim if input_dim is not None else contract["feature_count"]),
        "d_model": int(d_model),
        "n_layers": int(n_layers),
        "n_heads": int(DEFAULT_N_HEADS),
        "dropout": 0.1,
        "family_head_dim": int(len(EXIT_AUX_FAMILY_NAMES)),
        "family_head_names": list(EXIT_AUX_FAMILY_NAMES),
        "profit_protect_head_dim": 1,
        "exit_ml_io_version": str(contract["io_version"]),
        "feature_names_hash": str(feature_names_hash or contract["feature_hash"]),
    }

    # V12.2: serialize multi-TF config so the bundle can be reloaded as
    # multi-TF at live inference time. Inferred from the model class itself —
    # v8 models (no multi-TF) will write enabled=False.
    enable_mtf = bool(getattr(model, "enable_multi_tf", False))
    mtf_dims = getattr(model, "_mtf_dims", None) or {}
    mtf_lens = getattr(model, "_mtf_lens", None) or {}
    cfg["multi_tf"] = {
        "enabled": enable_mtf,
        "m5_seq_dim": int(mtf_dims.get("m5", 0)) if enable_mtf else 0,
        "m15_seq_dim": int(mtf_dims.get("m15", 0)) if enable_mtf else 0,
        "h1_seq_dim": int(mtf_dims.get("h1", 0)) if enable_mtf else 0,
        "h4_seq_dim": int(mtf_dims.get("h4", 0)) if enable_mtf else 0,
        "d1_seq_dim": int(mtf_dims.get("d1", 0)) if enable_mtf else 0,
        "m5_seq_len": int(mtf_lens.get("m5", 96)) if enable_mtf else 96,
        "m15_seq_len": int(mtf_lens.get("m15", 96)) if enable_mtf else 96,
        "h1_seq_len": int(mtf_lens.get("h1", 96)) if enable_mtf else 96,
        "h4_seq_len": int(mtf_lens.get("h4", 96)) if enable_mtf else 96,
        "d1_seq_len": int(mtf_lens.get("d1", 96)) if enable_mtf else 96,
        "feature_contract": "MULTI_TF_PER_BAR_V1" if enable_mtf else None,
    }

    cfg_path = out_dir / "exit_transformer_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return model_path, cfg_path, model_sha


class ExitTransformerDecider:
    """Lightweight inference wrapper."""

    def __init__(
        self,
        model: Any,
        config: Dict[str, Any],
        model_sha: Optional[str] = None,
        *,
        profit_protect_head_available: bool = False,
        family_head_available: bool = False,
        family_head_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.window_len = int(config.get("window_len", DEFAULT_WINDOW_LEN))
        self.input_dim = int(config.get("input_dim", EXIT_IO_FEATURE_COUNT))
        self.model_sha = model_sha
        self.feature_names_hash = config.get("feature_names_hash")
        self.profit_protect_head_available = bool(profit_protect_head_available)
        self.family_head_available = bool(family_head_available)
        self.family_head_names = list(family_head_names or [])

    def predict(self, window: Any) -> Tuple[float, Optional[float], Optional[float]]:
        details = self.predict_details(window)
        return (
            float(details["prob_close"]),
            details.get("logit"),
            details.get("raw_score"),
        )

    def predict_details(self, window: Any) -> Dict[str, Any]:
        arr = np.asarray(window, dtype=np.float32)
        validate_window_v3(arr, self.window_len, self.input_dim, context="inference")
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for inference")
        out: Dict[str, Any] = {
            "prob_close": None,
            "logit": None,
            "raw_score": None,
            "profit_protect_prob": None,
            "family_probs": None,
            "family_top_name": None,
        }
        with torch.no_grad():
            x = torch.from_numpy(arr).unsqueeze(0)
            if hasattr(self.model, "forward_logits"):
                logit = float(self.model.forward_logits(x).detach().cpu().numpy().reshape(-1)[0])
                try:
                    prob = 1.0 / (1.0 + math.exp(-logit))
                except OverflowError:
                    prob = 0.0 if logit < 0 else 1.0
                out["prob_close"] = float(prob)
                out["logit"] = logit
                out["raw_score"] = logit
                if self.profit_protect_head_available and hasattr(self.model, "forward_profit_protect_logits"):
                    profit_logit = float(self.model.forward_profit_protect_logits(x).detach().cpu().numpy().reshape(-1)[0])
                    try:
                        out["profit_protect_prob"] = float(1.0 / (1.0 + math.exp(-profit_logit)))
                    except OverflowError:
                        out["profit_protect_prob"] = 0.0 if profit_logit < 0 else 1.0
                if self.family_head_available and hasattr(self.model, "forward_family_logits"):
                    family_logits = self.model.forward_family_logits(x).detach().cpu().numpy().reshape(-1)
                    if family_logits.size > 0:
                        max_logit = float(np.max(family_logits))
                        exp_logits = np.exp(family_logits - max_logit)
                        probs = exp_logits / max(float(np.sum(exp_logits)), 1e-12)
                        out["family_probs"] = [float(v) for v in probs.tolist()]
                        top_idx = int(np.argmax(probs))
                        if 0 <= top_idx < len(self.family_head_names):
                            out["family_top_name"] = self.family_head_names[top_idx]
                return out
            prob = float(self.model(x).detach().cpu().numpy().reshape(-1)[0])
            out["prob_close"] = prob
        return out


def load_exit_transformer_decider(model_path: Path) -> ExitTransformerDecider:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required to load exit transformer")

    model_path = Path(model_path)
    if model_path.is_dir():
        model_file = model_path / "exit_transformer_v0.pt"
        cfg_path = model_path / "exit_transformer_config.json"
    else:
        model_file = model_path
        cfg_path = model_path.parent / "exit_transformer_config.json"

    if not model_file.exists():
        raise FileNotFoundError(model_file)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    window_len = int(cfg.get("window_len", DEFAULT_WINDOW_LEN))
    io_ver = str(cfg.get("exit_ml_io_version", "") or EXIT_IO_VERSION).strip() or EXIT_IO_VERSION
    contract = get_exit_io_contract(io_ver)
    input_dim = int(cfg.get("input_dim", contract["feature_count"]))
    feat_hash = cfg.get("feature_names_hash", "")
    if feat_hash and feat_hash != contract["feature_hash"]:
        if os.getenv("GX1_EXIT_HASH_GUARD_BYPASS") == "1":
            log.warning(
                "[EXIT_IO_CONTRACT_BYPASS] enabled=1 feature_names_hash=%s contract_feature_hash=%s",
                feat_hash,
                contract["feature_hash"],
            )
        else:
            raise RuntimeError(
                f"[EXIT_IO_CONTRACT_VIOLATION] feature_names_hash mismatch: {feat_hash} != {contract['feature_hash']}"
            )
    if input_dim != int(contract["feature_count"]):
        raise RuntimeError(
            f"[EXIT_IO_CONTRACT_VIOLATION] input_dim mismatch: {input_dim} != {int(contract['feature_count'])}"
        )

    model = ExitTransformerV0(
        input_dim=input_dim,
        window_len=window_len,
        d_model=int(cfg.get("d_model", DEFAULT_D_MODEL)),
        n_heads=int(cfg.get("n_heads", DEFAULT_N_HEADS)),
        n_layers=int(cfg.get("n_layers", DEFAULT_N_LAYERS)),
        dropout=float(cfg.get("dropout", 0.1)),
        family_head_dim=int(cfg.get("family_head_dim", len(EXIT_AUX_FAMILY_NAMES))),
        profit_protect_head_dim=int(cfg.get("profit_protect_head_dim", 1)),
    )
    state = torch.load(model_file, map_location="cpu")
    load_result = model.load_state_dict(state, strict=False)
    missing_keys = set(load_result.missing_keys)
    unexpected_keys = set(load_result.unexpected_keys)
    allowed_missing = {
        "family_head.weight",
        "family_head.bias",
        "profit_protect_head.weight",
        "profit_protect_head.bias",
    }
    if unexpected_keys:
        raise RuntimeError(f"[EXIT_MODEL_LOAD_UNEXPECTED_KEYS] unexpected={sorted(unexpected_keys)}")
    if missing_keys and not missing_keys.issubset(allowed_missing):
        raise RuntimeError(f"[EXIT_MODEL_LOAD_MISSING_KEYS] missing={sorted(missing_keys)}")
    model.eval()

    with open(model_file, "rb") as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()

    return ExitTransformerDecider(
        model=model,
        config=cfg,
        model_sha=model_sha,
        profit_protect_head_available=hasattr(model, "forward_profit_protect_logits"),
        family_head_available=hasattr(model, "forward_family_logits"),
        family_head_names=list(cfg.get("family_head_names", []) or []),
    )

def verify_exit_transformer_artifacts(model_dir: Path) -> Dict[str, Any]:
    """
    Lightweight artifact verification for exit_transformer_v0 bundles.
    """
    failures: List[str] = []
    model_dir = Path(model_dir)
    model_file = model_dir / "exit_transformer_v0.pt"
    cfg_path = model_dir / "exit_transformer_config.json"
    if not model_file.exists():
        failures.append(f"missing_model_file:{model_file}")
    if not cfg_path.exists():
        failures.append(f"missing_config_file:{cfg_path}")
    cfg = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
        except Exception as exc:
            failures.append(f"config_read_failed:{exc}")
    if cfg:
        io_ver = str(cfg.get("exit_ml_io_version", "") or EXIT_IO_VERSION).strip() or EXIT_IO_VERSION
        contract = get_exit_io_contract(io_ver)
        feat_hash = cfg.get("feature_names_hash")
        if feat_hash and feat_hash != contract["feature_hash"]:
            failures.append(f"feature_hash_mismatch:{feat_hash}!={contract['feature_hash']}")
        input_dim = int(cfg.get("input_dim", -1))
        if input_dim != int(contract["feature_count"]):
            failures.append(f"input_dim_mismatch:{input_dim}!={int(contract['feature_count'])}")
    if model_file.exists() and cfg_path.exists():
        try:
            load_exit_transformer_decider(model_dir)
        except Exception as exc:
            failures.append(f"model_load_failed:{exc}")
    return {"passed": not failures, "failures": failures}


# -----------------------------------------------------------------------------
# Dataset loader / builder
# -----------------------------------------------------------------------------
def _load_exits_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _focus_exit_training_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep all positive labels and prune obviously uninteresting negative hold states.

    This keeps the training universe focused on actual decision neighborhoods:
    trades with real progress, deterioration, or non-trivial live PnL. Fully dead
    hold bars add a lot of class imbalance but little exit signal.
    """
    try:
        focus_mfe_min = float(os.environ.get("GX1_EXIT_FOCUS_MFE_MIN", "8.0"))
    except Exception:
        focus_mfe_min = 8.0
    try:
        focus_dd_min = float(os.environ.get("GX1_EXIT_FOCUS_DD_MIN", "8.0"))
    except Exception:
        focus_dd_min = 8.0
    try:
        focus_gb_min = float(os.environ.get("GX1_EXIT_FOCUS_GB_MIN", "0.20"))
    except Exception:
        focus_gb_min = 0.20
    try:
        focus_abs_pnl_min = float(os.environ.get("GX1_EXIT_FOCUS_ABS_PNL_MIN", "4.0"))
    except Exception:
        focus_abs_pnl_min = 4.0
    try:
        focus_bg_stride = max(1, int(os.environ.get("GX1_EXIT_FOCUS_BG_STRIDE", "20")))
    except Exception:
        focus_bg_stride = 20

    log.info(
        "[EXIT_TRAIN_FOCUS_CFG] mfe_min=%.2f dd_min=%.2f gb_min=%.2f abs_pnl_min=%.2f bg_stride=%d",
        float(focus_mfe_min),
        float(focus_dd_min),
        float(focus_gb_min),
        float(focus_abs_pnl_min),
        int(focus_bg_stride),
    )

    focused: List[Dict[str, Any]] = []
    kept_pos = 0
    kept_neg = 0
    sampled_background_neg = 0
    dropped_neg = 0
    negative_counter = 0

    for rec in records:
        should_exit = float(rec.get("should_exit", 0.0) or 0.0)
        profit_exit = float(rec.get("profit_protect_should_exit", 0.0) or 0.0)
        if should_exit > 0.5 or profit_exit > 0.5:
            focused.append(rec)
            kept_pos += 1
            continue

        io = rec.get("io_features")
        if io is None:
            io = (rec.get("io") or {}).get("io_features")
        if io is None:
            dropped_neg += 1
            continue
        arr = np.asarray(io, dtype=np.float32)
        if arr.ndim == 2:
            row = arr[-1]
        elif arr.ndim == 1:
            row = arr
        else:
            dropped_neg += 1
            continue
        try:
            mfe_bps = float(row[exit_feature_index_v1_ctx36("mfe_bps")])
            dd_bps = float(row[exit_feature_index_v1_ctx36("dd_from_mfe_bps")])
            giveback_ratio = float(row[exit_feature_index_v1_ctx36("giveback_ratio")])
            pnl_bps_now = float(row[exit_feature_index_v1_ctx36("pnl_bps_now")])
            bars_held = float(row[exit_feature_index_v1_ctx36("bars_held")])
        except Exception:
            dropped_neg += 1
            continue
        if not all(math.isfinite(v) for v in (mfe_bps, dd_bps, giveback_ratio, pnl_bps_now, bars_held)):
            dropped_neg += 1
            continue

        in_decision_neighborhood = bool(
            bars_held >= 2.0
            and (
                mfe_bps >= float(focus_mfe_min)
                or dd_bps >= float(focus_dd_min)
                or giveback_ratio >= float(focus_gb_min)
                or abs(pnl_bps_now) >= float(focus_abs_pnl_min)
            )
        )
        if in_decision_neighborhood:
            focused.append(rec)
            kept_neg += 1
        else:
            negative_counter += 1
            if bars_held >= 2.0 and (negative_counter % int(focus_bg_stride)) == 0:
                focused.append(rec)
                sampled_background_neg += 1
            else:
                dropped_neg += 1

    log.info(
        "[EXIT_TRAIN_FOCUS] total=%d kept=%d kept_pos=%d kept_neg=%d sampled_background_neg=%d dropped_neg=%d keep_rate=%.6f",
        len(records),
        len(focused),
        kept_pos,
        kept_neg,
        sampled_background_neg,
        dropped_neg,
        float(len(focused) / len(records)) if records else math.nan,
    )
    return focused


def _build_windows_from_exits(
    records: List[Dict[str, Any]],
    window_len: int,
    stride: int,
    exit_io_version: str = EXIT_IO_VERSION,
    return_targets: bool = False,
) -> Tuple[Any, ...]:
    contract = get_exit_io_contract(exit_io_version)
    feature_count = int(contract["feature_count"])
    feats: List[np.ndarray] = []
    main_labels: List[float] = []
    profit_labels: List[float] = []
    profit_masks: List[float] = []
    ts_list: List[pd.Timestamp] = []
    weights: List[float] = []
    family_ids: List[int] = []
    aux_family_names: List[str] = []
    label_family_names: List[str] = []
    profit_bucket_names: List[str] = []

    for r in records:
        io = r.get("io_features")
        if io is None:
            io = (r.get("io") or {}).get("io_features")
        label = r.get("should_exit")
        if io is None or label is None:
            continue
        sample_weight = r.get("sample_weight", 1.0)
        try:
            sample_weight_f = float(sample_weight)
        except Exception:
            sample_weight_f = 1.0
        if not math.isfinite(sample_weight_f) or sample_weight_f <= 0.0:
            sample_weight_f = 1.0
        aux_fam_name = str(r.get("exit_aux_family") or "NEGATIVE").strip().upper()
        fam_id = EXIT_AUX_FAMILY_TO_ID.get(aux_fam_name, EXIT_AUX_FAMILY_IGNORE_INDEX)
        label_fam_name = str(r.get("exit_label_family") or "NEGATIVE").strip().upper()
        profit_bucket_name = str(r.get("profit_head_bucket") or "OTHER").strip().upper()
        profit_label = float(r.get("profit_protect_should_exit", 0.0) or 0.0)
        profit_mask = float(r.get("profit_protect_train_mask", 0.0) or 0.0)
        arr = np.asarray(io, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != feature_count:
            raise RuntimeError(
                f"EXIT_IO_DIM_MISMATCH expected_input_dim={feature_count} got_input_dim={int(arr.shape[-1])}"
            )
        if arr.shape[0] < window_len:
            continue
        ts_raw = r.get("ts") or r.get("event_ts") or r.get("time")
        try:
            ts_val = pd.Timestamp(ts_raw) if ts_raw else pd.NaT
        except Exception:
            ts_val = pd.NaT
        for s in range(0, arr.shape[0] - window_len + 1, stride):
            feats.append(arr[s : s + window_len])
            main_labels.append(float(label))
            profit_labels.append(float(profit_label))
            profit_masks.append(float(profit_mask))
            ts_list.append(ts_val)
            weights.append(sample_weight_f)
            family_ids.append(int(fam_id))
            aux_family_names.append(aux_fam_name)
            label_family_names.append(label_fam_name)
            profit_bucket_names.append(profit_bucket_name)

    if not feats:
        base = (
            np.zeros((0, window_len, feature_count), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
            np.zeros(0, dtype=np.float32),
        )
        if not return_targets:
            return base
        return base + (np.zeros(0, dtype=np.int64), [], [], [])

    base = (
        np.stack(feats),
        np.asarray(main_labels, dtype=np.float32),
        np.asarray(profit_labels, dtype=np.float32),
        np.asarray(profit_masks, dtype=np.float32),
        ts_list,
        np.asarray(weights, dtype=np.float32),
    )
    if not return_targets:
        return base
    return base + (np.asarray(family_ids, dtype=np.int64), aux_family_names, label_family_names, profit_bucket_names)


def _family_distribution(values: List[str], known_names: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {name: 0 for name in known_names}
    counts["NEGATIVE_OR_OTHER"] = 0
    for val in values:
        name = str(val).strip().upper()
        if name in counts:
            counts[name] = int(counts.get(name, 0) + 1)
        else:
            counts["NEGATIVE_OR_OTHER"] = int(counts.get("NEGATIVE_OR_OTHER", 0) + 1)
    return counts


# -----------------------------------------------------------------------------
# Score compression audit
# -----------------------------------------------------------------------------
def _score_stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _audit_one_head(
    *,
    head_name: str,
    model: "ExitTransformerV0",
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    train_mask: Optional[np.ndarray] = None,
    val_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    train_mask_arr = np.ones_like(y_tr, dtype=np.float32) if train_mask is None else np.asarray(train_mask, dtype=np.float32)
    val_mask_arr = np.ones_like(y_va, dtype=np.float32) if val_mask is None else np.asarray(val_mask, dtype=np.float32)
    train_keep = train_mask_arr > 0.5
    val_keep = val_mask_arr > 0.5
    X_tr_eff = X_tr[train_keep]
    y_tr_eff = y_tr[train_keep]
    X_va_eff = X_va[val_keep]
    y_va_eff = y_va[val_keep]

    def _forward(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return np.zeros(0, dtype=np.float32)
        with torch.no_grad():
            x = torch.from_numpy(arr).to(device)
            if head_name == "main_head":
                return model.forward_logits(x).detach().cpu().numpy()
            if head_name == "profit_protect_head":
                return model.forward_profit_protect_logits(x).detach().cpu().numpy()
            raise RuntimeError(f"[EXIT_AUDIT_UNKNOWN_HEAD] head={head_name}")

    logits_tr = _forward(X_tr_eff)
    logits_va = _forward(X_va_eff)
    probs_tr = 1.0 / (1.0 + np.exp(-logits_tr)) if logits_tr.size else np.zeros(0)
    probs_va = 1.0 / (1.0 + np.exp(-logits_va)) if logits_va.size else np.zeros(0)

    combined_y = np.concatenate([y_tr_eff, y_va_eff]) if (y_tr_eff.size or y_va_eff.size) else np.zeros(0)
    label_rate = float(np.mean(combined_y)) if combined_y.size else math.nan
    train_pos_rate = float(np.mean(y_tr_eff)) if y_tr_eff.size else math.nan
    val_pos_rate = float(np.mean(y_va_eff)) if y_va_eff.size else math.nan
    prob_stats = _score_stats(probs_va if probs_va.size else probs_tr)
    logit_stats = _score_stats(logits_va if logits_va.size else logits_tr)

    suspected_root_cause = "unknown"
    if math.isfinite(label_rate) and label_rate < 0.05 and prob_stats["std"] < 0.002:
        suspected_root_cause = "class_imbalance_unweighted_bce"
    effective_X = X_va_eff if X_va_eff.size else X_tr_eff
    effective_window_len = int(effective_X.shape[1]) if effective_X.ndim == 3 and effective_X.shape[0] > 0 else int(DEFAULT_WINDOW_LEN)
    effective_input_dim = int(effective_X.shape[2]) if effective_X.ndim == 3 and effective_X.shape[0] > 0 else int(EXIT_IO_FEATURE_COUNT)

    return {
        "head_name": head_name,
        "scope": "training_windows_val_split_or_train_fallback",
        "score_source": "validation" if X_va_eff.size else "train_fallback",
        "n_train_windows": int(X_tr_eff.shape[0]),
        "n_val_windows": int(X_va_eff.shape[0]),
        "window_len": int(effective_window_len),
        "input_dim": int(effective_input_dim),
        "feature_names_hash": EXIT_FEATURE_NAMES_HASH,
        "label_rate": label_rate,
        "train_pos_rate": train_pos_rate,
        "val_pos_rate": val_pos_rate,
        "logit_mean": logit_stats["mean"],
        "logit_std": logit_stats["std"],
        "prob_mean": prob_stats["mean"],
        "prob_std": prob_stats["std"],
        "suspected_root_cause": suspected_root_cause,
        "train_mask_rate": float(np.mean(train_keep.astype(np.float32))) if train_mask_arr.size else math.nan,
        "val_mask_rate": float(np.mean(val_keep.astype(np.float32))) if val_mask_arr.size else math.nan,
    }


def _audit_score_compression(
    model: "ExitTransformerV0",
    X_tr: np.ndarray,
    y_main_tr: np.ndarray,
    y_profit_tr: np.ndarray,
    profit_mask_tr: np.ndarray,
    profit_buckets_tr: List[str],
    X_va: np.ndarray,
    y_main_va: np.ndarray,
    y_profit_va: np.ndarray,
    profit_mask_va: np.ndarray,
    profit_buckets_va: List[str],
) -> Dict[str, Any]:
    def _audit_profit_head_buckets() -> Dict[str, Any]:
        model.eval()
        device = next(model.parameters()).device
        if X_va.size:
            X_eval = X_va
            buckets_eval = profit_buckets_va
            scope = "validation"
        else:
            X_eval = X_tr
            buckets_eval = profit_buckets_tr
            scope = "train_fallback"
        if X_eval.size == 0:
            return {"scope": scope, "buckets": {}}
        with torch.no_grad():
            logits = model.forward_profit_protect_logits(torch.from_numpy(X_eval).to(device)).detach().cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
        bucket_stats: Dict[str, Any] = {}
        for bucket in list(EXIT_PROFIT_HEAD_BUCKET_NAMES) + ["OTHER"]:
            idx = [i for i, name in enumerate(buckets_eval) if str(name).strip().upper() == bucket]
            if not idx:
                bucket_stats[bucket] = {"count": 0}
                continue
            arr = probs[np.asarray(idx, dtype=np.int64)]
            bucket_stats[bucket] = {
                "count": int(arr.size),
                "prob_mean": float(np.mean(arr)),
                "prob_std": float(np.std(arr)),
                "prob_p90": float(np.quantile(arr, 0.90)),
                "ge_0_5_count": int(np.sum(arr >= 0.5)),
                "ge_0_5_rate": float(np.mean(arr >= 0.5)),
            }
        return {"scope": scope, "buckets": bucket_stats}

    return {
        "main_head": _audit_one_head(
            head_name="main_head",
            model=model,
            X_tr=X_tr,
            y_tr=y_main_tr,
            X_va=X_va,
            y_va=y_main_va,
        ),
        "profit_protect_head": _audit_one_head(
            head_name="profit_protect_head",
            model=model,
            X_tr=X_tr,
            y_tr=y_profit_tr,
            X_va=X_va,
            y_va=y_profit_va,
            train_mask=profit_mask_tr,
            val_mask=profit_mask_va,
        ),
        "profit_protect_head_bucket_audit": _audit_profit_head_buckets(),
    }


def audit_score_compression_from_exits_jsonl(
    exits_jsonl_path: Path,
    model_path: Path,
    out_dir: Optional[Path] = None,
    window_len: int = DEFAULT_WINDOW_LEN,
    stride: int = 1,
) -> Dict[str, Any]:
    """
    Standalone audit: attach labels if missing, build windows, run model,
    and emit score-compression stats to out_dir.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    exits_jsonl_path = Path(exits_jsonl_path)
    records = _load_exits_jsonl(exits_jsonl_path)
    _attach_labels_to_exit_records(records)
    records = _focus_exit_training_records(records)
    X, y_main, y_profit, profit_mask, _, _, _, _, profit_bucket_names = _build_windows_from_exits(
        records, window_len, stride, return_targets=True
    )
    if X.size == 0:
        raise RuntimeError("No audit windows produced")

    decider = load_exit_transformer_decider(model_path)
    model = decider.model

    n = X.shape[0]
    split = max(1, n * 9 // 10)
    X_tr, y_main_tr, y_profit_tr, profit_mask_tr = X[:split], y_main[:split], y_profit[:split], profit_mask[:split]
    X_va, y_main_va, y_profit_va, profit_mask_va = X[split:], y_main[split:], y_profit[split:], profit_mask[split:]
    profit_buckets_tr = profit_bucket_names[:split]
    profit_buckets_va = profit_bucket_names[split:]

    audit = _audit_score_compression(
        model,
        X_tr,
        y_main_tr,
        y_profit_tr,
        profit_mask_tr,
        profit_buckets_tr,
        X_va,
        y_main_va,
        y_profit_va,
        profit_mask_va,
        profit_buckets_va,
    )
    if out_dir is None:
        out_dir = model_path if model_path.is_dir() else model_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SCORE_COMPRESSION_AUDIT.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    log.info(
        "[EXIT_SCORE_COMPRESSION_AUDIT] main_label_rate=%.6f main_prob_mean=%.6f main_prob_std=%.6f profit_label_rate=%.6f profit_prob_mean=%.6f profit_prob_std=%.6f",
        audit["main_head"]["label_rate"],
        audit["main_head"]["prob_mean"],
        audit["main_head"]["prob_std"],
        audit["profit_protect_head"]["label_rate"],
        audit["profit_protect_head"]["prob_mean"],
        audit["profit_protect_head"]["prob_std"],
    )
    return audit


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_from_exits_jsonl(
    exits_jsonl_path: Path,
    out_dir: Optional[Path] = None,
    window_len: Optional[int] = None,
    d_model: int = DEFAULT_D_MODEL,
    n_layers: int = DEFAULT_N_LAYERS,
    n_heads: int = DEFAULT_N_HEADS,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_fold: int = 9,
    seed: int = 42,
    stride: int = 1,
    gx1_data: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required")

    extra = dict(_)
    requested_io_version = str(extra.get("exit_io_version", "") or "").strip()
    exit_io_version = requested_io_version or EXIT_IO_VERSION
    contract = get_exit_io_contract(exit_io_version)
    if window_len is None:
        window_len = int(contract["default_window_len"])
    window_len = int(window_len)

    exits_jsonl_path = Path(exits_jsonl_path)
    records = _load_exits_jsonl(exits_jsonl_path)

    # Attach deterministic labels if missing
    _attach_labels_to_exit_records(records)
    records = _focus_exit_training_records(records)

    _make_deterministic(seed)
    X, y_main, y_profit, profit_mask, ts_list, w, family_y, aux_family_names, label_family_names, profit_bucket_names = _build_windows_from_exits(
        records,
        window_len,
        stride,
        exit_io_version=exit_io_version,
        return_targets=True,
    )
    if X.size == 0:
        raise RuntimeError("No training windows produced")

    dataset_sha = hashlib.sha256(exits_jsonl_path.read_bytes()).hexdigest()
    n = X.shape[0]
    if n < 3:
        raise RuntimeError("[EXIT_SPLIT_FAIL] insufficient windows for train/val/test split")
    pos_idx = np.where(y_main > 0)[0]
    if pos_idx.size == 0:
        raise RuntimeError("[EXIT_SPLIT_FAIL] no positive labels in dataset")
    last_pos_idx = int(pos_idx[-1])
    if last_pos_idx < n - 1:
        ts_last = ts_list[last_pos_idx] if last_pos_idx < len(ts_list) else pd.NaT
        log.warning(
            "[EXIT_TRAIN_TRUNCATE] tail_no_pos=1 old_n=%d new_n=%d last_pos_idx=%d last_pos_ts=%s",
            n,
            last_pos_idx + 1,
            last_pos_idx,
            str(ts_last),
        )
        n = last_pos_idx + 1
        X = X[:n]
        y_main = y_main[:n]
        y_profit = y_profit[:n]
        profit_mask = profit_mask[:n]
        ts_list = ts_list[:n]
        w = w[:n]
        family_y = family_y[:n]
        aux_family_names = aux_family_names[:n]
        label_family_names = label_family_names[:n]
        profit_bucket_names = profit_bucket_names[:n]
    val_frac = 1.0 / float(val_fold + 1)
    try:
        test_frac = float(os.environ.get("GX1_EXIT_TRAIN_TEST_FRACTION", "0.05"))
    except Exception:
        test_frac = 0.05
    test_size = max(1, int(n * test_frac))
    val_size = max(1, int(n * val_frac))
    if test_size + val_size >= n:
        test_size = max(1, n // 20)
        val_size = max(1, n // 10)
    train_end = max(1, n - (val_size + test_size))
    val_end = max(train_end + 1, n - test_size)

    def _pos_count(arr: np.ndarray) -> int:
        return int(np.sum(arr)) if arr.size else 0

    def _time_range(ts_slice: List[pd.Timestamp]) -> Tuple[str, str]:
        ts_clean = [t for t in ts_slice if isinstance(t, pd.Timestamp) and t is not pd.NaT]
        if not ts_clean:
            return ("NA", "NA")
        return (str(min(ts_clean)), str(max(ts_clean)))

    # Ensure test has positives (expand test backward if needed)
    while val_end > train_end + 1 and _pos_count(y_main[val_end:]) == 0:
        val_end -= 1
    if _pos_count(y_main[val_end:]) == 0:
        raise RuntimeError("[EXIT_SPLIT_FAIL] test split has 0 positive labels")

    # Ensure val has positives (expand val backward if needed)
    while train_end > 1 and _pos_count(y_main[train_end:val_end]) == 0:
        train_end -= 1
    if _pos_count(y_main[train_end:val_end]) == 0:
        raise RuntimeError("[EXIT_SPLIT_FAIL] val split has 0 positive labels")

    X_tr = X[:train_end]
    y_main_tr = y_main[:train_end]
    y_profit_tr = y_profit[:train_end]
    profit_mask_tr = profit_mask[:train_end]
    w_tr = w[:train_end]
    fy_tr = family_y[:train_end]
    profit_buckets_tr = profit_bucket_names[:train_end]

    X_va = X[train_end:val_end]
    y_main_va = y_main[train_end:val_end]
    y_profit_va = y_profit[train_end:val_end]
    profit_mask_va = profit_mask[train_end:val_end]
    w_va = w[train_end:val_end]
    fy_va = family_y[train_end:val_end]
    profit_buckets_va = profit_bucket_names[train_end:val_end]

    X_te = X[val_end:]
    y_main_te = y_main[val_end:]
    y_profit_te = y_profit[val_end:]
    profit_mask_te = profit_mask[val_end:]
    w_te = w[val_end:]
    fy_te = family_y[val_end:]

    def _log_split(
        name: str,
        y_main_split: np.ndarray,
        y_profit_split: np.ndarray,
        profit_mask_split: np.ndarray,
        ts_split: List[pd.Timestamp],
    ) -> None:
        total = int(y_main_split.size)
        main_pos = _pos_count(y_main_split)
        profit_keep = profit_mask_split > 0.5
        profit_keep_count = int(np.sum(profit_keep))
        profit_pos = int(np.sum(y_profit_split[profit_keep])) if profit_keep_count else 0
        t_min, t_max = _time_range(ts_split)
        log.info(
            "[EXIT_TRAIN_SPLIT] split=%s total=%d main_pos=%d main_pos_rate=%.6f profit_mask=%d profit_mask_rate=%.6f profit_pos=%d profit_pos_rate=%.6f time_range=%s..%s",
            name,
            total,
            main_pos,
            float(main_pos / total) if total else math.nan,
            profit_keep_count,
            float(profit_keep_count / total) if total else math.nan,
            profit_pos,
            float(profit_pos / profit_keep_count) if profit_keep_count else math.nan,
            t_min,
            t_max,
        )

    _log_split("train", y_main_tr, y_profit_tr, profit_mask_tr, ts_list[:train_end])
    _log_split("val", y_main_va, y_profit_va, profit_mask_va, ts_list[train_end:val_end])
    _log_split("test", y_main_te, y_profit_te, profit_mask_te, ts_list[val_end:])
    log.info(
        "[EXIT_TRAIN_FAMILY_DISTRIBUTION] total=%s train=%s val=%s test=%s",
        _family_distribution(aux_family_names, EXIT_AUX_FAMILY_NAMES),
        _family_distribution(
            [EXIT_AUX_FAMILY_NAMES[int(i)] if 0 <= int(i) < len(EXIT_AUX_FAMILY_NAMES) else "NEGATIVE_OR_OTHER" for i in fy_tr],
            EXIT_AUX_FAMILY_NAMES,
        ),
        _family_distribution(
            [EXIT_AUX_FAMILY_NAMES[int(i)] if 0 <= int(i) < len(EXIT_AUX_FAMILY_NAMES) else "NEGATIVE_OR_OTHER" for i in fy_va],
            EXIT_AUX_FAMILY_NAMES,
        ),
        _family_distribution(
            [EXIT_AUX_FAMILY_NAMES[int(i)] if 0 <= int(i) < len(EXIT_AUX_FAMILY_NAMES) else "NEGATIVE_OR_OTHER" for i in fy_te],
            EXIT_AUX_FAMILY_NAMES,
        ),
    )
    log.info(
        "[EXIT_TRAIN_SAMPLE_WEIGHT_PROOF] train(mean/med/max)=%.3f/%.3f/%.3f val(mean/med/max)=%.3f/%.3f/%.3f test(mean/med/max)=%.3f/%.3f/%.3f",
        float(np.mean(w_tr)) if w_tr.size else 1.0,
        float(np.median(w_tr)) if w_tr.size else 1.0,
        float(np.max(w_tr)) if w_tr.size else 1.0,
        float(np.mean(w_va)) if w_va.size else 1.0,
        float(np.median(w_va)) if w_va.size else 1.0,
        float(np.max(w_va)) if w_va.size else 1.0,
        float(np.mean(w_te)) if w_te.size else 1.0,
        float(np.median(w_te)) if w_te.size else 1.0,
        float(np.max(w_te)) if w_te.size else 1.0,
    )

    device_pref = os.environ.get("GX1_EXIT_TRAIN_DEVICE", "").strip().lower()
    if device_pref in ("cuda", "cuda:0", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("[EXIT_TRAIN_DEVICE] CUDA requested but not available")
        device = torch.device("cuda")
    elif device_pref in ("cpu", ""):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_pref)
    model = ExitTransformerV0(
        input_dim=int(contract["feature_count"]),
        window_len=window_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        family_head_dim=len(EXIT_AUX_FAMILY_NAMES),
        profit_protect_head_dim=1,
    ).to(device)
    log.info(
        "[EXIT_TRAIN_DEVICE] device=%s input_dim=%d window_len=%d feature_hash=%s feature_names=%s",
        str(device),
        int(contract["feature_count"]),
        int(window_len),
        str(contract["feature_hash"]),
        list(contract["feature_names"]),
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_main_pos = float(np.sum(y_main_tr))
    n_main_neg = float(max(0.0, y_main_tr.size - n_main_pos))
    main_pos_weight = torch.tensor(n_main_neg / max(1.0, n_main_pos), dtype=torch.float32)
    profit_keep_train = profit_mask_tr > 0.5
    n_profit_train = int(np.sum(profit_keep_train))
    n_profit_pos = float(np.sum(y_profit_tr[profit_keep_train])) if n_profit_train else 0.0
    n_profit_neg = float(max(0.0, n_profit_train - n_profit_pos))
    profit_pos_weight = torch.tensor(n_profit_neg / max(1.0, n_profit_pos), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=main_pos_weight, reduction="none")
    profit_loss_fn = nn.BCEWithLogitsLoss(pos_weight=profit_pos_weight, reduction="none")
    family_loss_fn = nn.CrossEntropyLoss(ignore_index=EXIT_AUX_FAMILY_IGNORE_INDEX, reduction="none")
    try:
        family_aux_loss_weight = float(
            os.environ.get("GX1_EXIT_FAMILY_AUX_LOSS_WEIGHT", str(DEFAULT_FAMILY_AUX_LOSS_WEIGHT))
        )
    except Exception:
        family_aux_loss_weight = float(DEFAULT_FAMILY_AUX_LOSS_WEIGHT)
    try:
        profit_protect_head_loss_weight = float(
            os.environ.get("GX1_EXIT_PROFIT_PROTECT_HEAD_LOSS_WEIGHT", str(DEFAULT_PROFIT_PROTECT_HEAD_LOSS_WEIGHT))
        )
    except Exception:
        profit_protect_head_loss_weight = float(DEFAULT_PROFIT_PROTECT_HEAD_LOSS_WEIGHT)

    early_stop = os.environ.get("GX1_EXIT_TRAIN_EARLY_STOP", "0").strip() == "1"
    try:
        early_stop_patience = int(os.environ.get("GX1_EXIT_TRAIN_EARLY_STOP_PATIENCE", "5"))
    except Exception:
        early_stop_patience = 5
    try:
        early_stop_min_delta = float(os.environ.get("GX1_EXIT_TRAIN_EARLY_STOP_MIN_DELTA", "0.0"))
    except Exception:
        early_stop_min_delta = 0.0

    best_state = None
    best_vloss = float("inf")
    bad_epochs = 0
    train_main_loss_last = math.nan
    train_profit_loss_last = math.nan
    train_aux_loss_last = math.nan
    val_main_loss_last = math.nan
    val_profit_loss_last = math.nan
    val_aux_loss_last = math.nan

    for epoch in range(epochs):
        model.train()
        epoch_main_loss_sum = 0.0
        epoch_profit_loss_sum = 0.0
        epoch_aux_loss_sum = 0.0
        epoch_sample_count = 0
        for i in range(0, X_tr.shape[0], batch_size):
            bx = torch.from_numpy(X_tr[i : i + batch_size]).to(device)
            by = torch.from_numpy(y_main_tr[i : i + batch_size]).to(device)
            bp = torch.from_numpy(y_profit_tr[i : i + batch_size]).to(device)
            bpm = torch.from_numpy(profit_mask_tr[i : i + batch_size]).to(device)
            bw = torch.from_numpy(w_tr[i : i + batch_size]).to(device)
            bf = torch.from_numpy(fy_tr[i : i + batch_size]).to(device)
            opt.zero_grad()
            logits = model.forward_logits(bx)
            profit_logits = model.forward_profit_protect_logits(bx)
            family_logits = model.forward_family_logits(bx)
            main_loss = (loss_fn(logits, by) * bw).mean()
            profit_mask_batch = (bpm > 0.5).float()
            if float((bw * profit_mask_batch).sum().detach().cpu().item()) > 0.0:
                profit_raw = profit_loss_fn(profit_logits, bp)
                profit_loss = ((profit_raw * bw) * profit_mask_batch).sum() / (
                    ((bw * profit_mask_batch).sum()) + 1e-8
                )
            else:
                profit_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
            family_raw = family_loss_fn(family_logits, bf)
            family_mask = (bf != EXIT_AUX_FAMILY_IGNORE_INDEX).float()
            if float((bw * family_mask).sum().detach().cpu().item()) > 0.0:
                aux_loss = ((family_raw * bw) * family_mask).sum() / (((bw * family_mask).sum()) + 1e-8)
            else:
                aux_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
            loss = (
                main_loss
                + (float(profit_protect_head_loss_weight) * profit_loss)
                + (float(family_aux_loss_weight) * aux_loss)
            )
            loss.backward()
            opt.step()
            bs = int(by.shape[0])
            epoch_sample_count += bs
            epoch_main_loss_sum += float(main_loss.detach().cpu().item()) * bs
            epoch_profit_loss_sum += float(profit_loss.detach().cpu().item()) * bs
            epoch_aux_loss_sum += float(aux_loss.detach().cpu().item()) * bs

        train_main_loss_last = epoch_main_loss_sum / max(1, epoch_sample_count)
        train_profit_loss_last = epoch_profit_loss_sum / max(1, epoch_sample_count)
        train_aux_loss_last = epoch_aux_loss_sum / max(1, epoch_sample_count)

        if early_stop and X_va.size:
            model.eval()
            with torch.no_grad():
                va_tensor = torch.from_numpy(X_va).to(device)
                ya_tensor = torch.from_numpy(y_main_va).to(device)
                yp_tensor = torch.from_numpy(y_profit_va).to(device)
                ypm_tensor = torch.from_numpy(profit_mask_va).to(device)
                wa_tensor = torch.from_numpy(w_va).to(device)
                yfa_tensor = torch.from_numpy(fy_va).to(device)
                va_logits = model.forward_logits(va_tensor)
                va_profit_logits = model.forward_profit_protect_logits(va_tensor)
                va_family_logits = model.forward_family_logits(va_tensor)
                va_main_loss = (loss_fn(va_logits, ya_tensor) * wa_tensor).mean()
                va_profit_mask = (ypm_tensor > 0.5).float()
                if float((wa_tensor * va_profit_mask).sum().detach().cpu().item()) > 0.0:
                    va_profit_raw = profit_loss_fn(va_profit_logits, yp_tensor)
                    va_profit_loss = ((va_profit_raw * wa_tensor) * va_profit_mask).sum() / (
                        ((wa_tensor * va_profit_mask).sum()) + 1e-8
                    )
                else:
                    va_profit_loss = torch.zeros((), dtype=va_main_loss.dtype, device=va_main_loss.device)
                va_family_raw = family_loss_fn(va_family_logits, yfa_tensor)
                va_family_mask = (yfa_tensor != EXIT_AUX_FAMILY_IGNORE_INDEX).float()
                if float((wa_tensor * va_family_mask).sum().detach().cpu().item()) > 0.0:
                    va_aux_loss = ((va_family_raw * wa_tensor) * va_family_mask).sum() / (
                        ((wa_tensor * va_family_mask).sum()) + 1e-8
                    )
                else:
                    va_aux_loss = torch.zeros((), dtype=va_main_loss.dtype, device=va_main_loss.device)
                val_main_loss_last = float(va_main_loss.detach().cpu().item())
                val_profit_loss_last = float(va_profit_loss.detach().cpu().item())
                val_aux_loss_last = float(va_aux_loss.detach().cpu().item())
                vloss_epoch = float(
                    (
                        va_main_loss
                        + (float(profit_protect_head_loss_weight) * va_profit_loss)
                        + (float(family_aux_loss_weight) * va_aux_loss)
                    ).detach().cpu().item()
                )
            if vloss_epoch + early_stop_min_delta < best_vloss:
                best_vloss = vloss_epoch
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    log.info(
                        "[EXIT_TRAIN_EARLY_STOP] epoch=%d best_vloss=%.6f patience=%d",
                        epoch + 1,
                        best_vloss,
                        early_stop_patience,
                    )
                    break

    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
        vloss = best_vloss
    else:
        model.eval()
        with torch.no_grad():
            if X_va.size:
                va_tensor = torch.from_numpy(X_va).to(device)
                ya_tensor = torch.from_numpy(y_main_va).to(device)
                yp_tensor = torch.from_numpy(y_profit_va).to(device)
                ypm_tensor = torch.from_numpy(profit_mask_va).to(device)
                wa_tensor = torch.from_numpy(w_va).to(device)
                yfa_tensor = torch.from_numpy(fy_va).to(device)
                va_logits = model.forward_logits(va_tensor)
                va_profit_logits = model.forward_profit_protect_logits(va_tensor)
                va_family_logits = model.forward_family_logits(va_tensor)
                va_main_loss = (loss_fn(va_logits, ya_tensor) * wa_tensor).mean()
                va_profit_mask = (ypm_tensor > 0.5).float()
                if float((wa_tensor * va_profit_mask).sum().detach().cpu().item()) > 0.0:
                    va_profit_raw = profit_loss_fn(va_profit_logits, yp_tensor)
                    va_profit_loss = ((va_profit_raw * wa_tensor) * va_profit_mask).sum() / (
                        ((wa_tensor * va_profit_mask).sum()) + 1e-8
                    )
                else:
                    va_profit_loss = torch.zeros((), dtype=va_main_loss.dtype, device=va_main_loss.device)
                va_family_raw = family_loss_fn(va_family_logits, yfa_tensor)
                va_family_mask = (yfa_tensor != EXIT_AUX_FAMILY_IGNORE_INDEX).float()
                if float((wa_tensor * va_family_mask).sum().detach().cpu().item()) > 0.0:
                    va_aux_loss = ((va_family_raw * wa_tensor) * va_family_mask).sum() / (
                        ((wa_tensor * va_family_mask).sum()) + 1e-8
                    )
                else:
                    va_aux_loss = torch.zeros((), dtype=va_main_loss.dtype, device=va_main_loss.device)
                val_main_loss_last = float(va_main_loss.detach().cpu().item())
                val_profit_loss_last = float(va_profit_loss.detach().cpu().item())
                val_aux_loss_last = float(va_aux_loss.detach().cpu().item())
                vloss = float(
                    (
                        va_main_loss
                        + (float(profit_protect_head_loss_weight) * va_profit_loss)
                        + (float(family_aux_loss_weight) * va_aux_loss)
                    ).detach().cpu().item()
                )
            else:
                vloss = math.nan
                val_main_loss_last = math.nan
                val_profit_loss_last = math.nan
                val_aux_loss_last = math.nan

    audit = _audit_score_compression(
        model,
        X_tr,
        y_main_tr,
        y_profit_tr,
        profit_mask_tr,
        profit_buckets_tr,
        X_va,
        y_main_va,
        y_profit_va,
        profit_mask_va,
        profit_buckets_va,
    )

    base = Path(gx1_data or os.environ["GX1_DATA"])
    if out_dir is None:
        out_dir = base / "models" / "exit_transformer_v0" / f"{exit_io_version}__WL{window_len}__{dataset_sha}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path, cfg_path, sha = save_exit_transformer_artifacts(
        model,
        out_dir,
        window_len,
        d_model,
        n_layers,
        feature_names_hash=str(contract["feature_hash"]),
        input_dim=int(contract["feature_count"]),
        exit_io_version=exit_io_version,
    )

    train_report_path = out_dir / "TRAIN_REPORT.json"
    family_distribution_total = _family_distribution(label_family_names, EXIT_LABEL_FAMILY_NAMES)
    family_distribution_positive_only = _family_distribution(
        [name for name in label_family_names if name in EXIT_LABEL_FAMILY_TO_ID],
        EXIT_LABEL_FAMILY_NAMES,
    )
    aux_family_distribution_total = _family_distribution(aux_family_names, EXIT_AUX_FAMILY_NAMES)
    aux_family_distribution_splits = {
        "train": _family_distribution(
            [EXIT_AUX_FAMILY_NAMES[int(i)] if 0 <= int(i) < len(EXIT_AUX_FAMILY_NAMES) else "NEGATIVE_OR_OTHER" for i in fy_tr],
            EXIT_AUX_FAMILY_NAMES,
        ),
        "val": _family_distribution(
            [EXIT_AUX_FAMILY_NAMES[int(i)] if 0 <= int(i) < len(EXIT_AUX_FAMILY_NAMES) else "NEGATIVE_OR_OTHER" for i in fy_va],
            EXIT_AUX_FAMILY_NAMES,
        ),
        "test": _family_distribution(
            [EXIT_AUX_FAMILY_NAMES[int(i)] if 0 <= int(i) < len(EXIT_AUX_FAMILY_NAMES) else "NEGATIVE_OR_OTHER" for i in fy_te],
            EXIT_AUX_FAMILY_NAMES,
        ),
    }
    report = {
        "dataset": str(exits_jsonl_path),
        "dataset_sha256": dataset_sha,
        "exit_ml_io_version": exit_io_version,
        "window_len": int(window_len),
        "feature_names_hash": str(contract["feature_hash"]),
        "main_label_profile": str(os.environ.get("GX1_EXIT_MAIN_LABEL_PROFILE", "MAIN_V1_CONTROL")).strip().upper() or "MAIN_V1_CONTROL",
        "n_samples": int(X.shape[0]),
        "main_label_rate": float(np.mean(y_main)),
        "exit_rate": float(np.mean(y_main)),
        "profit_protect_label_rate": float(np.mean(y_profit)),
        "profit_protect_train_mask_rate": float(np.mean(profit_mask)),
        "family_label_distribution_total": family_distribution_total,
        "family_label_distribution_positive_only": family_distribution_positive_only,
        "aux_family_distribution_total": aux_family_distribution_total,
        "profit_head_bucket_distribution_total": _family_distribution(profit_bucket_names, EXIT_PROFIT_HEAD_BUCKET_NAMES),
        "family_label_distribution_splits": {
            "train": _family_distribution(
                [name for name in label_family_names[:train_end]],
                EXIT_LABEL_FAMILY_NAMES,
            ),
            "val": _family_distribution(
                [name for name in label_family_names[train_end:val_end]],
                EXIT_LABEL_FAMILY_NAMES,
            ),
            "test": _family_distribution(
                [name for name in label_family_names[val_end:]],
                EXIT_LABEL_FAMILY_NAMES,
            ),
        },
        "aux_family_distribution_splits": aux_family_distribution_splits,
        "profit_head_bucket_distribution_splits": {
            "train": _family_distribution(profit_buckets_tr, EXIT_PROFIT_HEAD_BUCKET_NAMES),
            "val": _family_distribution(profit_buckets_va, EXIT_PROFIT_HEAD_BUCKET_NAMES),
            "test": _family_distribution(profit_bucket_names[val_end:], EXIT_PROFIT_HEAD_BUCKET_NAMES),
        },
        "losses": {
            "main_head_train_loss": float(train_main_loss_last),
            "profit_protect_head_train_loss": float(train_profit_loss_last),
            "family_aux_train_loss": float(train_aux_loss_last),
            "main_head_val_loss": float(val_main_loss_last),
            "profit_protect_head_val_loss": float(val_profit_loss_last),
            "family_aux_val_loss": float(val_aux_loss_last),
            "profit_protect_head_loss_weight": float(profit_protect_head_loss_weight),
            "family_aux_loss_weight": float(family_aux_loss_weight),
            "total_val_loss": float(vloss),
        },
        "val_loss": vloss,
        "score_compression_audit": audit,
        "profit_protect_head_normal_threshold_mass_activation": audit.get("profit_protect_head_bucket_audit", {}).get("buckets", {}).get("NORMAL_THRESHOLD_MASS", {}),
        "model_path": str(model_path),
        "config_path": str(cfg_path),
        "model_sha256": sha,
    }

    with open(out_dir / "SCORE_COMPRESSION_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    report["config_sha256"] = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    report["score_compression_audit_sha256"] = hashlib.sha256((out_dir / "SCORE_COMPRESSION_AUDIT.json").read_bytes()).hexdigest()

    log.info(
        "[EXIT_SCORE_COMPRESSION_AUDIT] main_label_rate=%.6f main_prob_mean=%.6f main_prob_std=%.6f profit_label_rate=%.6f profit_prob_mean=%.6f profit_prob_std=%.6f",
        audit["main_head"]["label_rate"],
        audit["main_head"]["prob_mean"],
        audit["main_head"]["prob_std"],
        audit["profit_protect_head"]["label_rate"],
        audit["profit_protect_head"]["prob_mean"],
        audit["profit_protect_head"]["prob_std"],
    )
    log.info(
        "[EXIT_TRAINING_PROOF] input_dim=%d feature_hash=%s main_train_pos_rate=%.6f main_val_pos_rate=%.6f profit_train_pos_rate=%.6f profit_val_pos_rate=%.6f main_prob_mean=%.6f main_prob_std=%.6f profit_prob_mean=%.6f profit_prob_std=%.6f",
        int(EXIT_IO_FEATURE_COUNT),
        EXIT_FEATURE_NAMES_HASH,
        audit["main_head"]["train_pos_rate"],
        audit["main_head"]["val_pos_rate"],
        audit["profit_protect_head"]["train_pos_rate"],
        audit["profit_protect_head"]["val_pos_rate"],
        audit["main_head"]["prob_mean"],
        audit["main_head"]["prob_std"],
        audit["profit_protect_head"]["prob_mean"],
        audit["profit_protect_head"]["prob_std"],
    )
    log.info("[EXIT_TRAIN_DONE] out_dir=%s val_loss=%s", out_dir, vloss)
    with open(train_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["artifact_verify"] = verify_exit_transformer_artifacts(out_dir)
    with open(train_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return dict(report, train_report_path=train_report_path)
