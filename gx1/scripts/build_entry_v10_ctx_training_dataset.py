#!/usr/bin/env python3
"""
Build ENTRY_V10_CTX training dataset (canonical, CTX6CAT6 base with optional ctx_cont extensions).

SSoT / ONE UNIVERSE:
- ctx contract: CTX6CAT6 base (ctx_cat=6 fixed; ctx_cont base=6, extended here to the
  active canonical 21-dim runtime contract via micro + swing + session timing features)
- signal bridge: XGB_SIGNAL_BRIDGE_V1 (7-dim)
- Inputs must be canonical:
  - BASE28 prebuilt via CURRENT_MANIFEST.json (manifest-only resolution; sha256 verify; no direct parquet path)
  - canonical XGB bundle (universal multihead v2; ordered_features must match the active 34-feature contract)
  - canonical market tape lane (bid/ask) for deterministic label building (close after N bars)

Outputs:
- time: tz-aware UTC timestamp
- seq: list/ndarray shaped [seq_len, 7]  (signal bridge sequence)
- snap: ndarray shaped [7]              (signal bridge snapshot)
- ctx_cont: ndarray shaped [dynamic canonical ctx_cont dim; active path = 21]
- ctx_cat: ndarray shaped [6]
- y_direction: int32 (0/1/2)            (label computed from tape with fixed-hold exit; hold-bars configurable)
- y_early_move: float32 (0/1)           (label computed from tape within horizon=hold_bars)
- y_quality_score: float32              (e.g. abs pnl bps over horizon)
- y_bad_path: float32 (0/1, parked in canonical train recipe)

NO FALLBACKS unless explicitly allowed by CLI flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import hashlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gx1.contracts.signal_bridge_v1 import get_canonical_ctx_contract
from gx1.contracts.signal_bridge_v1 import (
    ORDERED_FIELDS as SIGNAL_FIELDS,
    CONTRACT_SHA256 as SIGNAL_CONTRACT_SHA256,
)
from gx1.time.session_detector import (
    get_session_minutes_since_open_vectorized,
    get_session_minutes_to_next_boundary_vectorized,
    get_session_vectorized,
)
from gx1.utils.canonical_prebuilt_resolver import resolve_base28_canonical_from_manifest
from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel, proba_to_signal_bridge_v1
from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PATH_QUALITY_HORIZON_BARS = 10
BAD_PATH_HORIZON_BARS = PATH_QUALITY_HORIZON_BARS
BAD_PATH_MAE_THRESHOLD_BPS = 6.0
BAD_PATH_MFE_THRESHOLD_BPS = 4.0
MICRO_FEATURE_NAMES = [
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
    "wick_ratio",
    "distance_ema_fast",
]
SWING_FEATURE_NAMES = [
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "retracement_from_last_impulse",
]
SESSION_CTX_CONT_NAMES = [
    "is_ASIA",
    "minutes_since_session_open",
    "minutes_to_next_session_boundary",
    "session_change_flag",
    "session_tradable",
]
SWING_ATR_PERIOD = 14

# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------
def get_git_commit() -> str:
    """Get current git commit hash (best-effort)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _hard_gate_ctx6cat6() -> Dict[str, Any]:
    """Fail-fast: ensure the canonical ctx base contract is CTX6CAT6 (ctx_cont_base=6, ctx_cat=6)."""
    ctx = get_canonical_ctx_contract()
    tag = str(ctx.get("tag", ""))
    cont = int(ctx.get("ctx_cont_dim", -1))
    cat = int(ctx.get("ctx_cat_dim", -1))
    if tag != "CTX6CAT6" or cont != 6 or cat != 6:
        raise RuntimeError(
            f"CTX_CONTRACT_SPLIT_BRAIN: expected CTX6CAT6 base (ctx_cont_base=6, ctx_cat=6) "
            f"but got tag={tag} cont={cont} cat={cat}"
        )
    # Names must exist for stable column mapping
    if "ctx_cont_names" not in ctx or "ctx_cat_names" not in ctx:
        raise RuntimeError("CTX_CONTRACT_INVALID: missing ctx_cont_names/ctx_cat_names in contract")
    if len(ctx["ctx_cont_names"]) != 6 or len(ctx["ctx_cat_names"]) != 6:
        raise RuntimeError("CTX_CONTRACT_INVALID: ctx base names length must be 6/6")
    return ctx


def _ensure_inputs_exist(base28_manifest: Path, xgb_bundle: Path) -> None:
    if not base28_manifest.exists():
        raise RuntimeError(f"INPUT_MISSING: base28_manifest not found: {base28_manifest}")
    if not xgb_bundle.exists():
        raise RuntimeError(f"INPUT_MISSING: xgb_bundle not found: {xgb_bundle}")
    if base28_manifest.suffix.lower() != ".json":
        raise RuntimeError(f"INPUT_INVALID: base28_manifest must be a .json manifest file: {base28_manifest}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_ts(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _split_min_max_from_ts_series(ts: pd.Series) -> Dict[str, Optional[str]]:
    t = pd.to_datetime(ts, utc=True, errors="coerce").dropna()
    if t.empty:
        return {"ts_min": None, "ts_max": None}
    return {"ts_min": str(t.min()), "ts_max": str(t.max())}


def _detect_time_col(df: pd.DataFrame) -> str:
    if "time" in df.columns:
        return "time"
    if "ts" in df.columns:
        return "ts"
    # Sometimes parquet index is time
    if "index" in df.columns:
        return "index"
    raise RuntimeError(
        "TIME_COLUMN_MISSING: canonical builder requires tz-aware UTC time column (time or ts)."
    )


def _normalize_time_utc(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out["time"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    if out["time"].isna().any():
        raise RuntimeError("TIME_PARSE_FAIL: time column could not be parsed to tz-aware UTC")
    out = out.dropna(subset=["time"]).sort_values("time")
    out = out[~out["time"].duplicated()].copy()
    if len(out) == 0:
        raise RuntimeError("EMPTY_AFTER_TIME_NORMALIZATION")
    return out


def _resolve_gx1_data_root() -> Path:
    gx1_data = os.environ.get("GX1_DATA", "").strip()
    if not gx1_data:
        raise RuntimeError("GX1_DATA not set; required to resolve canonical market tape lane")
    root = Path(gx1_data).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"GX1_DATA invalid dir: {root}")
    return root


# -----------------------------------------------------------------------------
# Market tape loading (canonical lane)
# -----------------------------------------------------------------------------
def _load_canonical_tape(
    *,
    tape_root: Path,
    t_min: pd.Timestamp,
    t_max: pd.Timestamp,
    required_cols: List[str],
) -> pd.DataFrame:
    """
    Load canonical M5 tape for [t_min, t_max] from a partitioned parquet dataset:
      .../xauusd_m5_bid_ask__CANONICAL/year=YYYY/part-000.parquet

    We avoid depending on manifest schema here; we trust the canonical lane path and parquet partitioning.
    """
    tape_root = tape_root.expanduser().resolve()
    if not tape_root.exists():
        raise RuntimeError(f"TAPE_ROOT_MISSING: {tape_root}")
    if not tape_root.is_dir():
        raise RuntimeError(f"TAPE_ROOT_NOT_DIR: {tape_root}")

    # Pull only years that intersect range
    y0 = int(pd.Timestamp(t_min).year)
    y1 = int(pd.Timestamp(t_max).year)
    files: List[Path] = []
    for y in range(y0, y1 + 1):
        p = tape_root / f"year={y}"
        if p.exists() and p.is_dir():
            files.extend(sorted(p.glob("*.parquet")))
            files.extend(sorted(p.glob("part-*.parquet")))
    # If layout differs, fall back to recursive parquet scan (still deterministic)
    if not files:
        files = sorted(tape_root.rglob("*.parquet"))

    if not files:
        raise RuntimeError(f"TAPE_NO_FILES: no parquet files found under {tape_root}")

    # Read and filter
    df_list: List[pd.DataFrame] = []
    for fp in files:
        dfi = pd.read_parquet(fp, columns=list(set(["time"] + required_cols)))
        if "time" not in dfi.columns:
            # Some tape uses "ts"
            if "ts" in dfi.columns:
                dfi = dfi.rename(columns={"ts": "time"})
            else:
                raise RuntimeError(f"TAPE_TIME_MISSING: {fp}")
        dfi["time"] = pd.to_datetime(dfi["time"], utc=True, errors="coerce")
        dfi = dfi.dropna(subset=["time"])
        dfi = dfi[(dfi["time"] >= t_min) & (dfi["time"] <= t_max)]
        if len(dfi):
            df_list.append(dfi)

    if not df_list:
        raise RuntimeError("TAPE_EMPTY_IN_RANGE")

    tape = pd.concat(df_list, ignore_index=True)
    tape = tape.sort_values("time")
    tape = tape[~tape["time"].duplicated()].copy()

    missing = [c for c in required_cols if c not in tape.columns]
    if missing:
        raise RuntimeError(f"TAPE_REQUIRED_COLS_MISSING: {missing}")

    if tape["time"].dtype != "datetime64[ns, UTC]":
        # pandas sometimes shows tz-aware as dtype object, normalize again
        tape["time"] = pd.to_datetime(tape["time"], utc=True, errors="coerce")
        tape = tape.dropna(subset=["time"])

    if len(tape) == 0:
        raise RuntimeError("TAPE_EMPTY_AFTER_NORMALIZATION")

    return tape


# -----------------------------------------------------------------------------
# Labels (simple fixed-hold exit, consistent with the "close after N bars" sanity exit)
# -----------------------------------------------------------------------------
def _compute_labels_fixed_hold(
    *,
    tape: pd.DataFrame,
    horizon_bars: int,
    early_move_threshold_bps: float,
    flat_threshold_bps: float,
) -> pd.DataFrame:
    """
    Labels are computed from bid/ask close (or bid/ask) with a fixed hold:
    - y_direction: 0=LONG, 1=SHORT, 2=FLAT
      LONG  if long_edge_bps  >= +flat_threshold_bps and long_edge_bps >= short_edge_bps
      SHORT if short_edge_bps >= +flat_threshold_bps and short_edge_bps >  long_edge_bps
      FLAT  otherwise (no clear edge or abs return below flat threshold)
    - y_early_move: 1 if max favorable move within horizon >= threshold (direction-aware)
    - y_quality_score: pnl_bps clipped for chosen direction (direction-aware)
    """
    if horizon_bars < 1:
        raise RuntimeError("HORIZON_INVALID")

    # We need bid/ask "close" like columns. Common names:
    # - bid_close / ask_close, or bid / ask
    cols = list(tape.columns)
    bid_col = "bid_close" if "bid_close" in cols else ("bid" if "bid" in cols else None)
    ask_col = "ask_close" if "ask_close" in cols else ("ask" if "ask" in cols else None)
    if bid_col is None or ask_col is None:
        raise RuntimeError(f"TAPE_BID_ASK_COLS_MISSING: have={sorted(cols)[:60]}...")

    bid = tape[bid_col].astype(float).to_numpy()
    ask = tape[ask_col].astype(float).to_numpy()

    n = len(tape)
    if n <= horizon_bars:
        raise RuntimeError("TAPE_TOO_SHORT_FOR_HORIZON")

    entry_ask = ask[:-horizon_bars]
    entry_bid = bid[:-horizon_bars]
    exit_bid = bid[horizon_bars:]
    exit_ask = ask[horizon_bars:]

    # pnl in bps (spread-aware)
    # LONG: buy at ask, sell at bid
    pnl_long_bps = (exit_bid - entry_ask) / np.clip(entry_ask, 1e-12, None) * 1e4
    # SHORT: sell at bid, buy at ask
    pnl_short_bps = (entry_bid - exit_ask) / np.clip(entry_bid, 1e-12, None) * 1e4

    # Early move (direction-aware):
    # LONG: max bid over horizon vs entry_ask
    # SHORT: min ask over horizon vs entry_bid
    max_fav_bid = np.empty(n - horizon_bars, dtype=np.float64)
    min_fav_ask = np.empty(n - horizon_bars, dtype=np.float64)
    for i in range(0, n - horizon_bars):
        bid_window = bid[i : i + horizon_bars + 1]
        ask_window = ask[i : i + horizon_bars + 1]
        max_fav_bid[i] = float(np.max(bid_window))
        min_fav_ask[i] = float(np.min(ask_window))
    mfe_long_bps = (max_fav_bid - entry_ask) / np.clip(entry_ask, 1e-12, None) * 1e4
    mfe_short_bps = (entry_bid - min_fav_ask) / np.clip(entry_bid, 1e-12, None) * 1e4

    if flat_threshold_bps < 0:
        raise RuntimeError("FLAT_THRESHOLD_INVALID")
    y_direction = np.full_like(pnl_long_bps, 2, dtype=np.int32)
    long_edge = pnl_long_bps
    short_edge = pnl_short_bps
    thr = float(flat_threshold_bps)
    long_mask = (long_edge >= thr) & (long_edge >= short_edge)
    short_mask = (short_edge >= thr) & (short_edge > long_edge)
    y_direction[long_mask] = 0
    y_direction[short_mask] = 1
    thr_early = float(early_move_threshold_bps)
    y_early = np.zeros_like(pnl_long_bps, dtype=np.float32)
    y_early[long_mask] = (mfe_long_bps[long_mask] >= thr_early).astype(np.float32)
    y_early[short_mask] = (mfe_short_bps[short_mask] >= thr_early).astype(np.float32)

    y_quality = np.zeros_like(pnl_long_bps, dtype=np.float32)
    y_quality[long_mask] = np.clip(pnl_long_bps[long_mask], 0.0, 5000.0).astype(np.float32)
    y_quality[short_mask] = np.clip(pnl_short_bps[short_mask], 0.0, 5000.0).astype(np.float32)

    out = pd.DataFrame(
        {
            "time": tape["time"].iloc[:-horizon_bars].to_numpy(),
            "y_direction": y_direction.astype(np.int32),
            "y_early_move": y_early,
            "y_quality_score": y_quality,
            "label_horizon_bars": np.int32(horizon_bars),
        }
    )
    return out


# -----------------------------------------------------------------------------
# Path quality (first N bars)
# -----------------------------------------------------------------------------
def _compute_path_quality_first_n(
    *,
    tape: pd.DataFrame,
    horizon_bars: int,
) -> pd.DataFrame:
    if horizon_bars < 1:
        raise RuntimeError("PATH_QUALITY_HORIZON_INVALID")

    cols = list(tape.columns)
    bid_col = "bid_close" if "bid_close" in cols else ("bid" if "bid" in cols else None)
    ask_col = "ask_close" if "ask_close" in cols else ("ask" if "ask" in cols else None)
    if bid_col is None or ask_col is None:
        raise RuntimeError(f"PATH_QUALITY_BID_ASK_MISSING: have={sorted(cols)[:60]}...")

    bid = tape[bid_col].astype(float).to_numpy()
    ask = tape[ask_col].astype(float).to_numpy()

    n = len(tape)
    if n <= horizon_bars:
        raise RuntimeError("PATH_QUALITY_TAPE_TOO_SHORT")

    entry_ask = ask[:-horizon_bars]
    entry_bid = bid[:-horizon_bars]

    mfe_long = np.empty(n - horizon_bars, dtype=np.float64)
    mae_long = np.empty(n - horizon_bars, dtype=np.float64)
    mfe_short = np.empty(n - horizon_bars, dtype=np.float64)
    mae_short = np.empty(n - horizon_bars, dtype=np.float64)

    for i in range(0, n - horizon_bars):
        w_bid = bid[i : i + horizon_bars + 1]
        w_ask = ask[i : i + horizon_bars + 1]
        max_bid = float(np.max(w_bid))
        min_bid = float(np.min(w_bid))
        max_ask = float(np.max(w_ask))
        min_ask = float(np.min(w_ask))

        mfe_long[i] = (max_bid - entry_ask[i]) / np.clip(entry_ask[i], 1e-12, None) * 1e4
        mae_long[i] = (entry_ask[i] - min_bid) / np.clip(entry_ask[i], 1e-12, None) * 1e4
        mfe_short[i] = (entry_bid[i] - min_ask) / np.clip(entry_bid[i], 1e-12, None) * 1e4
        mae_short[i] = (max_ask - entry_bid[i]) / np.clip(entry_bid[i], 1e-12, None) * 1e4

    out = pd.DataFrame(
        {
            "time": tape["time"].iloc[:-horizon_bars].to_numpy(),
            "mfe_long_first_n_bps": mfe_long,
            "mae_long_first_n_bps": mae_long,
            "mfe_short_first_n_bps": mfe_short,
            "mae_short_first_n_bps": mae_short,
            "path_quality_horizon_bars": np.int32(horizon_bars),
        }
    )
    return out


def _compute_bad_path_first_n(
    *,
    tape: pd.DataFrame,
    horizon_bars: int,
    adverse_threshold_bps: float,
    favorable_threshold_bps: float,
) -> pd.DataFrame:
    if horizon_bars < 1:
        raise RuntimeError("BAD_PATH_HORIZON_INVALID")

    cols = list(tape.columns)
    bid_col = "bid_close" if "bid_close" in cols else ("bid" if "bid" in cols else None)
    ask_col = "ask_close" if "ask_close" in cols else ("ask" if "ask" in cols else None)
    if bid_col is None or ask_col is None:
        raise RuntimeError(f"BAD_PATH_BID_ASK_MISSING: have={sorted(cols)[:60]}...")

    bid = tape[bid_col].astype(float).to_numpy()
    ask = tape[ask_col].astype(float).to_numpy()

    n = len(tape)
    if n <= horizon_bars:
        raise RuntimeError("BAD_PATH_TAPE_TOO_SHORT")

    entry_ask = ask[:-horizon_bars]
    entry_bid = bid[:-horizon_bars]
    thr_adv = float(adverse_threshold_bps)
    thr_fav = float(favorable_threshold_bps)
    out_long = np.zeros(n - horizon_bars, dtype=np.float32)
    out_short = np.zeros(n - horizon_bars, dtype=np.float32)

    for i in range(0, n - horizon_bars):
        w_bid = bid[i : i + horizon_bars + 1]
        w_ask = ask[i : i + horizon_bars + 1]

        long_fav = (w_bid - entry_ask[i]) / np.clip(entry_ask[i], 1e-12, None) * 1e4
        long_adv = (entry_ask[i] - w_bid) / np.clip(entry_ask[i], 1e-12, None) * 1e4
        short_fav = (entry_bid[i] - w_ask) / np.clip(entry_bid[i], 1e-12, None) * 1e4
        short_adv = (w_ask - entry_bid[i]) / np.clip(entry_bid[i], 1e-12, None) * 1e4

        long_fav_idx = np.flatnonzero(long_fav >= thr_fav)
        long_adv_idx = np.flatnonzero(long_adv >= thr_adv)
        short_fav_idx = np.flatnonzero(short_fav >= thr_fav)
        short_adv_idx = np.flatnonzero(short_adv >= thr_adv)

        first_long_fav = int(long_fav_idx[0]) if len(long_fav_idx) else None
        first_long_adv = int(long_adv_idx[0]) if len(long_adv_idx) else None
        first_short_fav = int(short_fav_idx[0]) if len(short_fav_idx) else None
        first_short_adv = int(short_adv_idx[0]) if len(short_adv_idx) else None

        out_long[i] = float(
            first_long_adv is not None and (first_long_fav is None or first_long_adv < first_long_fav)
        )
        out_short[i] = float(
            first_short_adv is not None and (first_short_fav is None or first_short_adv < first_short_fav)
        )

    return pd.DataFrame(
        {
            "time": tape["time"].iloc[:-horizon_bars].to_numpy(),
            "bad_path_long_first_n": out_long,
            "bad_path_short_first_n": out_short,
            "bad_path_horizon_bars": np.int32(horizon_bars),
            "bad_path_mae_threshold_bps": np.float32(adverse_threshold_bps),
            "bad_path_mfe_threshold_bps": np.float32(favorable_threshold_bps),
        }
    )


# -----------------------------------------------------------------------------
# Manifest writing
# -----------------------------------------------------------------------------
def write_manifest(
    *,
    output_path: Path,
    build_command: List[str],
    base28_manifest: Path,
    xgb_bundle: Path,
    tape_root: Optional[Path],
    splits: Optional[Dict[str, Any]] = None,
    ts_min_max_by_split: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    ctx = _hard_gate_ctx6cat6()
    extra_ctx = (extra or {}).get("ctx_contract") or {}
    ctx_cont_dim = int(extra_ctx.get("ctx_cont_dim") or ctx.get("ctx_cont_dim") or 6)
    ctx_cat_dim = int(extra_ctx.get("ctx_cat_dim") or ctx.get("ctx_cat_dim") or 6)
    ctx_cont_base_dim = int(extra_ctx.get("ctx_cont_base_dim") or ctx.get("ctx_cont_dim") or 6)
    ctx_cont_micro = list(extra_ctx.get("ctx_cont_micro_features") or [])
    ctx_cont_swing = list(extra_ctx.get("ctx_cont_swing_features") or [])

    manifest: Dict[str, Any] = {
        "created_at": _utc_now_iso(),
        "git_commit": get_git_commit(),
        "output_data_path": str(output_path),
        "build_command": build_command,
        "inputs": {
            "base28_manifest": str(base28_manifest),
            "xgb_bundle": str(xgb_bundle),
            "xgb_model_file": str((xgb_bundle / "xgb_universal_multihead_v2.joblib").resolve()),
            "xgb_model_sha256": extra.get("xgb_model_sha256") if extra else None,
            "tape_root": str(tape_root) if tape_root is not None else None,
        },
        "feature_contract": {
            "ctx_tag": str(ctx["tag"]),
            "ctx_cont_dim": int(ctx_cont_dim),
            "ctx_cat_dim": int(ctx_cat_dim),
            "ctx_cont_base_dim": int(ctx_cont_base_dim),
            "ctx_cont_micro_features": list(ctx_cont_micro),
            "ctx_cont_swing_features": list(ctx_cont_swing),
            "signal_bridge_id": "XGB_SIGNAL_BRIDGE_V1",
            "signal_bridge_contract_sha256": SIGNAL_CONTRACT_SHA256,
            "signal_bridge_fields": list(SIGNAL_FIELDS),
        },
        "splits": splits,
        "ts_min_max_by_split": ts_min_max_by_split or {},
        "notes": notes,
    }
    if extra:
        manifest["extra"] = extra

    manifest_path = output_path.parent / f"{output_path.stem}.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    log.info(f"MANIFEST WRITTEN: {manifest_path}")
    return manifest_path


# -----------------------------------------------------------------------------
# Label proof
# -----------------------------------------------------------------------------
_SESSION_ID_TO_NAME = {
    0: "ASIA",
    1: "EU",
    2: "OVERLAP",
    3: "US",
}


def _log_label_distribution_proof(df: pd.DataFrame, split: str) -> None:
    if "y_direction" not in df.columns:
        log.warning("[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s status=no_y_direction", split)
        return
    y = df["y_direction"].astype(int)
    n = int(len(y))
    if n == 0:
        log.warning("[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s status=empty", split)
        return
    long_c = int((y == 0).sum())
    short_c = int((y == 1).sum())
    flat_c = int((y == 2).sum())
    log.info(
        "[ENTRY_LABEL_DISTRIBUTION_PROOF] split=%s n=%d long=%d (%.4f) short=%d (%.4f) flat=%d (%.4f)",
        split,
        n,
        long_c,
        long_c / n,
        short_c,
        short_c / n,
        flat_c,
        flat_c / n,
    )
    log.info(
        "[ENTRY_FLAT_LABEL_PROOF] split=%s flat=%d flat_rate=%.4f status=%s",
        split,
        flat_c,
        flat_c / n,
        "OK" if flat_c > 0 else "EMPTY",
    )
    if "ctx_cat" not in df.columns:
        return
    try:
        sess_ids = df["ctx_cat"].apply(
            lambda v: int(v[0]) if isinstance(v, (list, tuple)) and len(v) > 0 else None
        )
        df_s = pd.DataFrame({"y": y, "session_id": sess_ids}).dropna(subset=["session_id"])
        if df_s.empty:
            return
        for sid, grp in df_s.groupby("session_id"):
            sid_int = int(sid)
            n_s = int(len(grp))
            long_s = int((grp["y"] == 0).sum())
            short_s = int((grp["y"] == 1).sum())
            flat_s = int((grp["y"] == 2).sum())
            log.info(
                "[ENTRY_LABEL_BY_SESSION_PROOF] split=%s session=%s session_id=%d n=%d long=%d (%.4f) short=%d (%.4f) flat=%d (%.4f)",
                split,
                _SESSION_ID_TO_NAME.get(sid_int, "UNKNOWN"),
                sid_int,
                n_s,
                long_s,
                long_s / n_s,
                short_s,
                short_s / n_s,
                flat_s,
                flat_s / n_s,
            )
    except Exception:
        return


# -----------------------------------------------------------------------------
# Core builder
# -----------------------------------------------------------------------------
def build_dataset_canonical(
    *,
    base28_manifest_path: Path,
    xgb_bundle_path: Path,
    tape_root: Path,
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    max_rows: Optional[int],
    seq_len: int,
    horizon_bars: int,
    early_move_threshold_bps: float,
    flat_threshold_bps: float,
    allow_zero_ctx: bool,
    split_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ctx = _hard_gate_ctx6cat6()
    _ensure_inputs_exist(base28_manifest_path, xgb_bundle_path)

    if seq_len < 2:
        raise RuntimeError("SEQ_LEN_INVALID: must be >=2")
    if horizon_bars < 1:
        raise RuntimeError("HORIZON_INVALID: must be >=1")

    # 1) Resolve BASE28 manifest + sha
    manifest_info = resolve_base28_canonical_from_manifest(str(base28_manifest_path))
    parquet_path = Path(manifest_info["parquet_path"]).expanduser().resolve()
    parquet_sha = manifest_info["parquet_sha256"]

    if not parquet_path.exists():
        raise RuntimeError(f"BASE28_PARQUET_MISSING: {parquet_path}")

    # 2) Load BASE28 parquet
    df = pd.read_parquet(parquet_path)
    df = df.reset_index(drop=False)
    time_col = _detect_time_col(df)
    df = _normalize_time_utc(df, time_col)

    # filter by start/end
    if start is not None:
        df = df[df["time"] >= start]
    if end is not None:
        df = df[df["time"] <= end]

    if len(df) == 0:
        raise RuntimeError("NO_ROWS_AFTER_FILTERS")

    # deterministic head
    if max_rows and len(df) > max_rows:
        df = df.head(int(max_rows)).copy()

    # 3) Enforce XGB feature contract order (BASE28 + session-context extensions).
    contract_path = project_root / "gx1" / "xgb" / "contracts" / "xgb_input_features_base28_v1.json"
    contract_obj = json.loads(contract_path.read_text(encoding="utf-8"))
    features = contract_obj.get("features") or contract_obj.get("ordered_features") or []
    if len(features) < 28:
        raise RuntimeError("FEATURE_CONTRACT_INVALID_LEN")

    # Derive session-context features from canonical UTC timestamps when absent or degenerate.
    ts = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if ts.isna().any():
        raise RuntimeError("TIME_PARSE_FAIL_FOR_SESSION_CONTEXT")
    session_map = {"ASIA": 0, "EU": 1, "OVERLAP": 2, "US": 3}
    session_recompute = False
    if "session_id" not in df.columns:
        session_recompute = True
    else:
        # Degenerate if < 2 unique non-null values
        n_unique = int(pd.Series(df["session_id"]).dropna().nunique())
        if n_unique < 2:
            session_recompute = True
    if session_recompute:
        df["session_id"] = get_session_vectorized(ts).map(session_map).astype(np.int32)
        log.info("[SESSION_ID_RECOMPUTE] recomputed session_id from timestamp (reason=%s)", "missing" if "session_id" not in df.columns else "degenerate")
    # Hard-fail if still degenerate after recompute
    n_unique_post = int(pd.Series(df["session_id"]).dropna().nunique())
    if n_unique_post < 2:
        raise RuntimeError(
            f"SESSION_ID_DEGENERATE_AFTER_RECOMPUTE: unique={n_unique_post} "
            f"(expected >=2). Check prebuilt time/session pipeline."
        )
    # Log distribution proof
    _sess_counts = pd.Series(df["session_id"]).value_counts(dropna=False).sort_index()
    log.info("[SESSION_ID_DISTRIBUTION_PROOF] %s", _sess_counts.to_dict())
    if "is_ASIA" not in df.columns:
        df["is_ASIA"] = (df["session_id"].astype(int) == 0).astype(np.int8)
    if "minutes_since_session_open" not in df.columns:
        df["minutes_since_session_open"] = get_session_minutes_since_open_vectorized(ts).astype(np.float32)
    if "minutes_to_next_session_boundary" not in df.columns:
        df["minutes_to_next_session_boundary"] = get_session_minutes_to_next_boundary_vectorized(ts).astype(np.float32)
    if "session_change_flag" not in df.columns:
        sid = df["session_id"].astype(np.int32)
        df["session_change_flag"] = (sid.diff().fillna(0) != 0).astype(np.int8)
    if "session_tradable" not in df.columns:
        df["session_tradable"] = (df["session_id"].astype(int) != 0).astype(np.int8)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError(f"BASE28_FEATURES_MISSING: {missing}")

    df_features = df[features].copy()
    if len(df_features) == 0:
        raise RuntimeError("NO_ROWS_AFTER_FEATURE_SELECT")

    # 4) Load canonical XGB bundle + sanitizer
    model_path = Path(xgb_bundle_path) / "xgb_universal_multihead_v2.joblib"
    sanitizer_cfg = project_root / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_base28_v1.json"
    if not model_path.exists():
        raise RuntimeError(f"XGB_MODEL_MISSING: {model_path}")
    if not sanitizer_cfg.exists():
        raise RuntimeError(f"SANITIZER_CONFIG_MISSING: {sanitizer_cfg}")

    xgb_model_sha256 = _sha256_file(model_path)
    log.info("[XGB_MODEL_SHA256] %s %s", model_path, xgb_model_sha256)

    model = XGBMultiheadModel.load(str(model_path))
    model_features = list(getattr(model, "feature_list", []) or [])
    if model_features != features:
        raise RuntimeError(
            "XGB_FEATURE_CONTRACT_MISMATCH: contract features != model.feature_list "
            f"(contract_len={len(features)} model_len={len(model_features)})"
        )
    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_cfg))

    # sanitize (contract-ordered)
    x_array, _stats = sanitizer.sanitize(df_features, feature_list=features, allow_nan_fill=False)
    if x_array is None or len(df_features) != len(df):
        raise RuntimeError("SANITIZER_OUTPUT_INVALID")
    if np.isnan(x_array).any() or np.isinf(x_array).any():
        raise RuntimeError("SANITIZER_FAIL_NONFINITE: sanitized features contain NaN/Inf")
    df_features_sanitized = pd.DataFrame(
        x_array,
        columns=features,
        index=df_features.index,
    )

    # 5) Predict per session head (ASIA routes to OVERLAP if no ASIA head exists)
    session_series = df["session_id"].fillna(2).astype(int) if "session_id" in df.columns else None
    session_map = {0: "OVERLAP", 1: "EU", 2: "OVERLAP", 3: "US"}

    bridge_all = np.zeros((len(df), 7), dtype=np.float64)

    def _run_for_session(sess_name: str, idx: np.ndarray) -> None:
        if idx.size == 0:
            return
        probs = model.predict_proba(
            df_features_sanitized.iloc[idx],
            session=sess_name,
            feature_list=features,
        )
        # Expect attributes or dict-like; support both
        if hasattr(probs, "p_long"):
            pl = np.asarray(probs.p_long, dtype=np.float64)
            ps = np.asarray(probs.p_short, dtype=np.float64)
            pf = np.asarray(probs.p_flat, dtype=np.float64)
        else:
            pl = np.asarray(probs["p_long"], dtype=np.float64)
            ps = np.asarray(probs["p_short"], dtype=np.float64)
            pf = np.asarray(probs["p_flat"], dtype=np.float64)
        bridge_input = np.column_stack([pl, ps, pf])
        bridge = proba_to_signal_bridge_v1(bridge_input)
        if bridge.shape[1] != 7:
            raise RuntimeError(f"[BRIDGE_DIM_MISMATCH] expected bridge_dim=7, got shape={bridge.shape}")
        bridge_all[idx, :] = bridge

    if session_series is not None:
        for sid, name in session_map.items():
            mask = session_series.values == sid
            idx = np.where(mask)[0]
            _run_for_session(name, idx)
    else:
        _run_for_session("US", np.arange(len(df), dtype=np.int64))

    # Bridge proof log/meta
    log.info("[BRIDGE_PROOF] proba_dim=%d -> bridge_dim=%d rows=%d", 3, 7, bridge_all.shape[0])

    # 6) Build ctx features
    ctx_cont_names = list(ctx["ctx_cont_names"])
    ctx_cat_names = list(ctx["ctx_cat_names"])

    # 7) Assemble per-bar signal dataframe (time aligned)
    df_sig = pd.DataFrame({"time": df["time"].to_numpy()})
    for i, field in enumerate(SIGNAL_FIELDS):
        df_sig[field] = bridge_all[:, i]

    # 8) Labels from canonical tape lane (join by time)
    t_min = pd.Timestamp(df_sig["time"].min()).tz_convert("UTC")
    t_max = pd.Timestamp(df_sig["time"].max()).tz_convert("UTC")

    tape = _load_canonical_tape(
        tape_root=tape_root,
        t_min=t_min,
        t_max=t_max,
        required_cols=["bid_close", "ask_close", "open", "high", "low", "close"],
    )

    # Inner join tape to BASE28 by time
    # We keep only matching times (deterministic). Require strict 1:1 match.
    merged = df_sig.merge(tape, on="time", how="inner", validate="one_to_one")
    rows_base28 = int(len(df_sig))
    rows_tape = int(len(tape))
    rows_joined = int(len(merged))
    exact_match = int(rows_base28 == rows_tape == rows_joined)
    log.info(
        "[ENTRY_TAPE_JOIN_PROOF] rows_base28=%d rows_tape=%d rows_joined=%d exact_match=%d",
        rows_base28,
        rows_tape,
        rows_joined,
        exact_match,
    )
    if not exact_match:
        raise RuntimeError(
            f"TAPE_JOIN_STRICT_FAIL: rows_base28={rows_base28} rows_tape={rows_tape} rows_joined={rows_joined}"
        )

    # Micro-structure features (canonical tape OHLC)
    eps = 1e-9
    if not all(c in merged.columns for c in ("close", "high", "low")):
        raise RuntimeError("MICRO_FEATURES_MISSING: require close/high/low in canonical tape")
    tape_feat = merged[["time", "close", "high", "low"]].copy().sort_values("time")
    close = tape_feat["close"].astype(float)
    high = tape_feat["high"].astype(float)
    low = tape_feat["low"].astype(float)

    tape_feat["micro_momentum_3"] = close - close.shift(3)
    tape_feat["micro_momentum_5"] = close - close.shift(5)
    tape_feat["micro_acceleration"] = (close - close.shift(1)) - (close.shift(1) - close.shift(2))
    tape_feat["wick_ratio"] = (high - close) / (high - low + eps)
    ema_fast = close.ewm(span=5, adjust=False).mean()
    tape_feat["distance_ema_fast"] = close - ema_fast

    for name in MICRO_FEATURE_NAMES:
        if name not in tape_feat.columns:
            raise RuntimeError(f"MICRO_FEATURE_MISSING: {name}")

    # Swing-structure features (ATR-normalized)
    prev_close = close.shift(1).fillna(close)
    tr = np.maximum(
        (high - low).abs(),
        np.maximum((high - prev_close).abs(), (low - prev_close).abs()),
    )
    atr = tr.rolling(window=SWING_ATR_PERIOD, min_periods=1).mean()
    atr_safe = atr.clip(lower=eps)

    pivot_high = high > pd.concat(
        [high.shift(1), high.shift(2), high.shift(-1), high.shift(-2)], axis=1
    ).max(axis=1, skipna=True)
    pivot_low = low < pd.concat(
        [low.shift(1), low.shift(2), low.shift(-1), low.shift(-2)], axis=1
    ).min(axis=1, skipna=True)
    log.info(
        "[ENTRY_SWING_PIVOT_PROOF] pivot_highs=%d pivot_lows=%d",
        int(pivot_high.sum()),
        int(pivot_low.sum()),
    )

    n = int(len(tape_feat))
    last_high_vals = np.empty(n, dtype=np.float64)
    last_low_vals = np.empty(n, dtype=np.float64)
    last_high_idx = np.empty(n, dtype=np.int64)
    last_low_idx = np.empty(n, dtype=np.int64)

    last_high = float(high.iloc[0])
    last_low = float(low.iloc[0])
    last_hi_i = 0
    last_lo_i = 0
    for i in range(n):
        if bool(pivot_high.iloc[i]):
            last_high = float(high.iloc[i])
            last_hi_i = i
        if bool(pivot_low.iloc[i]):
            last_low = float(low.iloc[i])
            last_lo_i = i
        last_high_vals[i] = last_high
        last_low_vals[i] = last_low
        last_high_idx[i] = last_hi_i
        last_low_idx[i] = last_lo_i

    idx = np.arange(n, dtype=np.int64)
    bars_since_high = (idx - last_high_idx).astype(np.float32)
    bars_since_low = (idx - last_low_idx).astype(np.float32)

    dist_last_swing_high_atr = (close.to_numpy() - last_high_vals) / atr_safe.to_numpy()
    dist_last_swing_low_atr = (close.to_numpy() - last_low_vals) / atr_safe.to_numpy()

    denom = np.maximum((last_high_vals - last_low_vals), eps)
    retracement = np.zeros(n, dtype=np.float64)
    up_mask = last_high_idx > last_low_idx
    down_mask = last_low_idx > last_high_idx
    retracement[up_mask] = (last_high_vals[up_mask] - close.to_numpy()[up_mask]) / denom[up_mask]
    retracement[down_mask] = (close.to_numpy()[down_mask] - last_low_vals[down_mask]) / denom[down_mask]
    retracement = np.clip(retracement, 0.0, 1.0)

    tape_feat["dist_last_swing_high_atr"] = dist_last_swing_high_atr.astype(np.float32)
    tape_feat["dist_last_swing_low_atr"] = dist_last_swing_low_atr.astype(np.float32)
    tape_feat["bars_since_swing_high"] = bars_since_high.astype(np.float32)
    tape_feat["bars_since_swing_low"] = bars_since_low.astype(np.float32)
    tape_feat["retracement_from_last_impulse"] = retracement.astype(np.float32)

    for name in SWING_FEATURE_NAMES:
        if name not in tape_feat.columns:
            raise RuntimeError(f"SWING_FEATURE_MISSING: {name}")

    # Attach micro features to BASE28 rows (strict 1:1 time alignment)
    df = df.merge(
        tape_feat[["time"] + list(MICRO_FEATURE_NAMES) + list(SWING_FEATURE_NAMES)],
        on="time",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_tape"),
    )
    if len(df) != rows_base28:
        raise RuntimeError(
            f"MICRO_FEATURE_JOIN_FAIL: rows_base28={rows_base28} rows_after={len(df)}"
        )
    for name in list(MICRO_FEATURE_NAMES) + list(SWING_FEATURE_NAMES):
        tape_name = f"{name}_tape"
        if name not in df.columns and tape_name in df.columns:
            df[name] = df[tape_name]
            df.drop(columns=[tape_name], inplace=True)

    ctx_cont_names = (
        ctx_cont_names
        + list(MICRO_FEATURE_NAMES)
        + list(SWING_FEATURE_NAMES)
        + list(SESSION_CTX_CONT_NAMES)
    )
    log.info(
        "[ENTRY_MICRO_FEATURES_PROOF] names=%s count=%d",
        list(MICRO_FEATURE_NAMES),
        len(MICRO_FEATURE_NAMES),
    )
    log.info(
        "[ENTRY_SWING_FEATURES_PROOF] names=%s count=%d",
        list(SWING_FEATURE_NAMES),
        len(SWING_FEATURE_NAMES),
    )
    log.info(
        "[ENTRY_SESSION_CTX_PROOF] names=%s count=%d",
        list(SESSION_CTX_CONT_NAMES),
        len(SESSION_CTX_CONT_NAMES),
    )

    for name in ctx_cont_names:
        if name not in df.columns:
            if allow_zero_ctx:
                df[name] = 0.0
            else:
                raise RuntimeError(f"CTX_CONT_MISSING_IN_BASE28: '{name}' not found (use --allow_zero_ctx to force zeros)")
    for name in ctx_cat_names:
        if name not in df.columns:
            if allow_zero_ctx:
                df[name] = 0
            else:
                raise RuntimeError(f"CTX_CAT_MISSING_IN_BASE28: '{name}' not found (use --allow_zero_ctx to force zeros)")

    # Normalize ctx dtypes
    df_ctx_cont = df[ctx_cont_names].astype(np.float32)
    df_ctx_cat = df[ctx_cat_names].astype(np.int64)

    # Compute labels on merged tape
    labels = _compute_labels_fixed_hold(
        tape=merged[["time"] + [c for c in merged.columns if c in ("bid_close", "ask_close", "bid", "ask")]].copy(),
        horizon_bars=horizon_bars,
        early_move_threshold_bps=early_move_threshold_bps,
        flat_threshold_bps=flat_threshold_bps,
    )
    path_quality = _compute_path_quality_first_n(
        tape=merged[["time"] + [c for c in merged.columns if c in ("bid_close", "ask_close", "bid", "ask")]].copy(),
        horizon_bars=PATH_QUALITY_HORIZON_BARS,
    )
    bad_path = _compute_bad_path_first_n(
        tape=merged[["time"] + [c for c in merged.columns if c in ("bid_close", "ask_close", "bid", "ask")]].copy(),
        horizon_bars=BAD_PATH_HORIZON_BARS,
        adverse_threshold_bps=BAD_PATH_MAE_THRESHOLD_BPS,
        favorable_threshold_bps=BAD_PATH_MFE_THRESHOLD_BPS,
    )

    # Align signals to labels (labels are shorter by horizon_bars)
    merged2 = merged.merge(
        labels[["time", "y_direction", "y_early_move", "y_quality_score", "label_horizon_bars"]],
        on="time",
        how="inner",
        validate="one_to_one",
    )
    merged2 = merged2.merge(
        path_quality[
            [
                "time",
                "mfe_long_first_n_bps",
                "mae_long_first_n_bps",
                "mfe_short_first_n_bps",
                "mae_short_first_n_bps",
                "path_quality_horizon_bars",
            ]
        ],
        on="time",
        how="inner",
        validate="one_to_one",
    )
    merged2 = merged2.merge(
        bad_path[
            [
                "time",
                "bad_path_long_first_n",
                "bad_path_short_first_n",
                "bad_path_horizon_bars",
                "bad_path_mae_threshold_bps",
                "bad_path_mfe_threshold_bps",
            ]
        ],
        on="time",
        how="inner",
        validate="one_to_one",
    )
    if len(merged2) == 0:
        raise RuntimeError("LABEL_JOIN_EMPTY")

    # Select MAE/MFE based on true direction (flat -> 0)
    y_dir = merged2["y_direction"].to_numpy()
    mae_first_n = np.where(
        y_dir == 0,
        merged2["mae_long_first_n_bps"].to_numpy(),
        np.where(y_dir == 1, merged2["mae_short_first_n_bps"].to_numpy(), 0.0),
    )
    mfe_first_n = np.where(
        y_dir == 0,
        merged2["mfe_long_first_n_bps"].to_numpy(),
        np.where(y_dir == 1, merged2["mfe_short_first_n_bps"].to_numpy(), 0.0),
    )
    merged2["mae_first_n_bps"] = mae_first_n.astype(np.float32)
    merged2["mfe_first_n_bps"] = mfe_first_n.astype(np.float32)
    merged2["path_quality_bps"] = (merged2["mfe_first_n_bps"] - merged2["mae_first_n_bps"]).astype(np.float32)
    bad_path_dir = np.where(
        y_dir == 0,
        merged2["bad_path_long_first_n"].to_numpy(),
        np.where(y_dir == 1, merged2["bad_path_short_first_n"].to_numpy(), 0.0),
    )
    merged2["y_bad_path"] = bad_path_dir.astype(np.float32)

    # Re-attach ctx to merged2 (align by time)
    df_ctx = pd.DataFrame({"time": df["time"].to_numpy()})
    for i, name in enumerate(ctx_cont_names):
        df_ctx[name] = df_ctx_cont.iloc[:, i].to_numpy()
    for i, name in enumerate(ctx_cat_names):
        df_ctx[name] = df_ctx_cat.iloc[:, i].to_numpy()

    merged3 = merged2.merge(df_ctx, on="time", how="inner", validate="one_to_one")
    if len(merged3) == 0:
        raise RuntimeError("CTX_JOIN_EMPTY")

    # 9) Build advanced structure: seq + snap + ctx arrays per sample
    # We use signal dims only (7 fields), and build rolling window of length seq_len.
    sig_mat = merged3[list(SIGNAL_FIELDS)].astype(np.float32).to_numpy()
    times = merged3["time"].to_numpy()

    ctx_cont_mat = merged3[ctx_cont_names].astype(np.float32).to_numpy()
    ctx_cat_mat = merged3[ctx_cat_names].astype(np.int64).to_numpy()

    y_dir = merged3["y_direction"].astype(np.int32).to_numpy()
    # Snapshot of original labels before any relabel rules for RAW pocket proofs.
    y_dir_raw = y_dir.copy()
    y_early = merged3["y_early_move"].astype(np.float32).to_numpy()
    y_qual = merged3["y_quality_score"].astype(np.float32).to_numpy()
    y_mae_first_n = merged3["mae_first_n_bps"].astype(np.float32).to_numpy()
    y_mfe_first_n = merged3["mfe_first_n_bps"].astype(np.float32).to_numpy()
    y_path_quality = merged3["path_quality_bps"].astype(np.float32).to_numpy()
    y_bad_path = merged3["y_bad_path"].astype(np.float32).to_numpy()
    y_label_horizon = merged3["label_horizon_bars"].astype(np.int32).to_numpy()
    y_path_horizon = merged3["path_quality_horizon_bars"].astype(np.int32).to_numpy()

    # ---------------------------------------------------------------------------
    # Poison-short relabel: H4_uptrend + negative micro_momentum + price below EMA
    #
    # Bars matching this signature are systematically mislabeled by the model as
    # SHORT, but price consistently moves UP (they are LONG or FLAT in truth).
    # Action: force y_direction=2 (FLAT) and zero y_quality_score so these bars
    # contribute no signal for the SHORT class during training.
    # Rationale: H4_trend_sign_cat=2 (H4 uptrend), micro_momentum_5 < -1.5
    # (short-term pullback), distance_ema_fast < -0.3 (price below fast EMA)
    # — this pocket was confirmed regime-stable across 2025 and 2026 runs.
    # Only applied when these three columns are available in merged3.
    # ---------------------------------------------------------------------------
    _POISON_SHORT_COLS = ("H4_trend_sign_cat", "micro_momentum_5", "distance_ema_fast")
    if all(c in merged3.columns for c in _POISON_SHORT_COLS):
        _h4 = merged3["H4_trend_sign_cat"].to_numpy()
        _mom5 = merged3["micro_momentum_5"].astype(np.float32).to_numpy()
        _dist_ema = merged3["distance_ema_fast"].astype(np.float32).to_numpy()
        _poison_mask = (
            (_h4 == 2)
            & (_mom5 < -1.5)
            & (_dist_ema < -0.3)
        )
        _n_poison = int(_poison_mask.sum())
        if _n_poison > 0:
            y_dir = y_dir.copy()
            y_qual = y_qual.copy()
            y_early = y_early.copy()
            y_dir[_poison_mask] = 2      # relabel to FLAT
            y_qual[_poison_mask] = 0.0   # zero quality weight
            y_early[_poison_mask] = 0.0  # no early-move credit
        log.info(
            "[POISON_SHORT_RELABEL] n_rows=%d n_poison_relabeled=%d "
            "criteria=H4_trend_sign_cat==2_AND_micro_momentum_5<-1.5_AND_distance_ema_fast<-0.3",
            len(merged3),
            _n_poison,
        )
    else:
        log.warning(
            "[POISON_SHORT_RELABEL_SKIP] missing_cols=%s",
            [c for c in _POISON_SHORT_COLS if c not in merged3.columns],
        )

    # ---------------------------------------------------------------------------
    # Group B relabel: H4=2 + high ATR + short-like local setup
    # Criteria (explicit):
    # - H4_trend_sign_cat == 2
    # - high ATR: atr_bps >= 9.4949 OR atr_bucket >= 4 OR D1_atr_percentile_252 >= 0.8254
    # - short-like: micro_momentum_3 < 0, micro_momentum_5 < 0, distance_ema_fast < 0
    # Action: force y_direction=2 (FLAT) and zero quality/early-move.
    # ---------------------------------------------------------------------------
    _GROUP_B_COLS = (
        "H4_trend_sign_cat",
        "atr_bps",
        "atr_bucket",
        "D1_atr_percentile_252",
        "micro_momentum_3",
        "micro_momentum_5",
        "distance_ema_fast",
    )
    if all(c in merged3.columns for c in _GROUP_B_COLS):
        _h4 = merged3["H4_trend_sign_cat"].to_numpy()
        _atr_bps = merged3["atr_bps"].astype(np.float32).to_numpy()
        _atr_bucket = merged3["atr_bucket"].astype(np.int64).to_numpy()
        _d1_atr = merged3["D1_atr_percentile_252"].astype(np.float32).to_numpy()
        _mom3 = merged3["micro_momentum_3"].astype(np.float32).to_numpy()
        _mom5 = merged3["micro_momentum_5"].astype(np.float32).to_numpy()
        _dist_ema = merged3["distance_ema_fast"].astype(np.float32).to_numpy()

        _high_atr = (_atr_bps >= 9.4949) | (_atr_bucket >= 4) | (_d1_atr >= 0.8254)
        _short_like = (_mom3 < 0.0) & (_mom5 < 0.0) & (_dist_ema < 0.0)
        _group_b_mask = (_h4 == 2) & _high_atr & _short_like
        _n_group_b = int(_group_b_mask.sum())
        if _n_group_b > 0:
            y_dir = y_dir.copy()
            y_qual = y_qual.copy()
            y_early = y_early.copy()
            y_dir[_group_b_mask] = 2
            y_qual[_group_b_mask] = 0.0
            y_early[_group_b_mask] = 0.0
        log.info(
            "[GROUP_B_RELABEL] n_rows=%d n_group_b_relabeled=%d "
            "criteria=H4_trend_sign_cat==2_AND_high_ATR_AND_micro_momentum_3<0_AND_micro_momentum_5<0_AND_distance_ema_fast<0 "
            "high_ATR=(atr_bps>=9.4949 OR atr_bucket>=4 OR D1_atr_percentile_252>=0.8254)",
            len(merged3),
            _n_group_b,
        )
    else:
        log.warning(
            "[GROUP_B_RELABEL_SKIP] missing_cols=%s",
            [c for c in _GROUP_B_COLS if c not in merged3.columns],
        )

    # ---------------------------------------------------------------------------
    # OVERLAP short tail relabel: session in {EU, OVERLAP} + short + high ATR + weak short setup
    # Criteria (explicit):
    # - session_id == 2 (OVERLAP)
    # - y_direction == 1 (SHORT)
    # - high ATR: atr_bps >= 9.4949 OR atr_bucket >= 4 OR D1_atr_percentile_252 >= 0.8254
    # - weak short setup: micro_momentum_3 >= 0, micro_momentum_5 >= 0, distance_ema_fast >= 0
    # Action: force y_direction=2 (FLAT) and zero quality/early-move.
    # ---------------------------------------------------------------------------
    _OVERLAP_TAIL_COLS = (
        "session_id",
        "atr_bps",
        "atr_bucket",
        "D1_atr_percentile_252",
        "micro_momentum_3",
        "micro_momentum_5",
        "distance_ema_fast",
    )
    if all(c in merged3.columns for c in _OVERLAP_TAIL_COLS):
        _sess = merged3["session_id"].astype(np.int64).to_numpy()
        _atr_bps = merged3["atr_bps"].astype(np.float32).to_numpy()
        _atr_bucket = merged3["atr_bucket"].astype(np.int64).to_numpy()
        _d1_atr = merged3["D1_atr_percentile_252"].astype(np.float32).to_numpy()
        _mom3 = merged3["micro_momentum_3"].astype(np.float32).to_numpy()
        _mom5 = merged3["micro_momentum_5"].astype(np.float32).to_numpy()
        _dist_ema = merged3["distance_ema_fast"].astype(np.float32).to_numpy()

        _high_atr = (_atr_bps >= 9.4949) | (_atr_bucket >= 4) | (_d1_atr >= 0.8254)
        _weak_short_setup = (_mom3 >= 0.0) & (_mom5 >= 0.0) & (_dist_ema >= 0.0)
        _drift_chop_setup = (np.abs(_mom3) <= 0.5) & (np.abs(_mom5) <= 0.5) & (np.abs(_dist_ema) <= 0.2)
        _session_mask = (_sess == 1) | (_sess == 2)  # EU=1, OVERLAP=2
        _short_mask = (y_dir == 1)
        _no_early_edge = (y_early == 0.0)
        _overlap_tail_mask = _session_mask & _short_mask & _high_atr & _no_early_edge & (_weak_short_setup | _drift_chop_setup)
        _n_overlap_tail = int(_overlap_tail_mask.sum())
        if _n_overlap_tail > 0:
            y_dir = y_dir.copy()
            y_qual = y_qual.copy()
            y_early = y_early.copy()
            y_dir[_overlap_tail_mask] = 2
            y_qual[_overlap_tail_mask] = 0.0
            y_early[_overlap_tail_mask] = 0.0
        log.info(
            "[OVERLAP_SHORT_TAIL_RELABEL] n_rows=%d n_overlap_short_tail_relabeled=%d "
            "criteria=session_id_in_{EU,OVERLAP}_AND_y_direction==1_AND_high_ATR_AND_y_early_move==0_AND("
            "micro_momentum_3>=0_AND_micro_momentum_5>=0_AND_distance_ema_fast>=0 OR "
            "abs(micro_momentum_3)<=0.5_AND_abs(micro_momentum_5)<=0.5_AND_abs(distance_ema_fast)<=0.2) "
            "high_ATR=(atr_bps>=9.4949 OR atr_bucket>=4 OR D1_atr_percentile_252>=0.8254)",
            len(merged3),
            _n_overlap_tail,
        )
    else:
        log.warning(
            "[OVERLAP_SHORT_TAIL_RELABEL_SKIP] missing_cols=%s",
            [c for c in _OVERLAP_TAIL_COLS if c not in merged3.columns],
        )

    # ---------------------------------------------------------------------------
    # OVERLAP short residual relabel (confirmed pocket, post-B1; H4 in {0,2}):
    # Criteria (explicit):
    # - session_id == OVERLAP (2)
    # - y_direction == SHORT
    # - H4_trend_sign_cat in {0, 2}
    # - atr_bucket == 3
    # - micro_momentum_5 < 0
    # - distance_ema_fast < 0
    # Action: force y_direction=2 (FLAT) and zero quality/early-move.
    # ---------------------------------------------------------------------------
    _OVERLAP_SHORT_RESIDUAL_COLS = (
        "session_id",
        "H4_trend_sign_cat",
        "atr_bucket",
        "micro_momentum_3",
        "micro_momentum_5",
        "distance_ema_fast",
    )
    if all(c in merged3.columns for c in _OVERLAP_SHORT_RESIDUAL_COLS):
        _sess = merged3["session_id"].astype(np.int64).to_numpy()
        _h4 = merged3["H4_trend_sign_cat"].astype(np.int64).to_numpy()
        _atr_bucket = merged3["atr_bucket"].astype(np.int64).to_numpy()
        _mom5 = merged3["micro_momentum_5"].astype(np.float32).to_numpy()
        _dist_ema = merged3["distance_ema_fast"].astype(np.float32).to_numpy()

        _is_overlap = (_sess == 2)  # OVERLAP
        _is_short = (y_dir == 1)       # SHORT (effective, after earlier relabels)
        _is_short_raw = (y_dir_raw == 1)  # SHORT (raw, before any relabels)
        _h4_0_2 = (_h4 == 0) | (_h4 == 2)
        _raw_mask = (
            _is_overlap
            & _is_short_raw
            & _h4_0_2
            & (_atr_bucket == 3)
            & (_mom5 < 0.0)
            & (_dist_ema < 0.0)
        )
        _mask = (
            _is_overlap
            & _is_short
            & _h4_0_2
            & (_atr_bucket == 3)
            & (_mom5 < 0.0)
            & (_dist_ema < 0.0)
        )
        _n_overlap_short_raw = int(_raw_mask.sum())
        _n_overlap_short_eff = int(_mask.sum())
        _overlap_short_total = int((_is_overlap & _is_short).sum())
        if _n_overlap_short_eff > 0:
            y_dir = y_dir.copy()
            y_qual = y_qual.copy()
            y_early = y_early.copy()
            y_dir[_mask] = 2
            y_qual[_mask] = 0.0
            y_early[_mask] = 0.0
        log.info(
            "[OVERLAP_SHORT_RESIDUAL_RAW_PROOF] n_rows=%d n_raw_match=%d "
            "criteria=session_id==OVERLAP_AND_y_direction==SHORT_AND_H4_trend_sign_cat_in_{0,2}_AND_atr_bucket==3_"
            "AND_micro_momentum_5<0_AND_distance_ema_fast<0",
            len(merged3),
            _n_overlap_short_raw,
        )
        log.info(
            "[OVERLAP_SHORT_RESIDUAL_EFFECTIVE_PROOF] n_rows=%d n_effective_relabel=%d "
            "share_of_overlap_short=%.6f",
            len(merged3),
            _n_overlap_short_eff,
            (_n_overlap_short_eff / _overlap_short_total) if _overlap_short_total > 0 else 0.0,
        )
    else:
        log.warning(
            "[OVERLAP_SHORT_RESIDUAL_RELABEL_SKIP] missing_cols=%s",
            [c for c in _OVERLAP_SHORT_RESIDUAL_COLS if c not in merged3.columns],
        )

    # ---------------------------------------------------------------------------
    # OVERLAP/EU short residual variant matrix (proof-only, no relabel)
    # Evaluated on RAW vs EFFECTIVE labels for clean comparison.
    # ---------------------------------------------------------------------------
    _OVERLAP_SHORT_MATRIX_COLS = (
        "session_id",
        "H4_trend_sign_cat",
        "atr_bucket",
        "micro_momentum_3",
        "micro_momentum_5",
        "distance_ema_fast",
    )
    if all(c in merged3.columns for c in _OVERLAP_SHORT_MATRIX_COLS):
        _sess = merged3["session_id"].astype(np.int64).to_numpy()
        _h4 = merged3["H4_trend_sign_cat"].astype(np.int64).to_numpy()
        _atr_bucket = merged3["atr_bucket"].astype(np.int64).to_numpy()
        _mom3 = merged3["micro_momentum_3"].astype(np.float32).to_numpy()
        _mom5 = merged3["micro_momentum_5"].astype(np.float32).to_numpy()
        _dist_ema = merged3["distance_ema_fast"].astype(np.float32).to_numpy()

        _is_short = (y_dir == 1)
        _is_short_raw = (y_dir_raw == 1)

        _sess_overlap = (_sess == 2)
        _sess_eu = (_sess == 1)
        _sess_eu_overlap = _sess_overlap | _sess_eu

        _atr_3 = (_atr_bucket == 3)
        _atr_4 = (_atr_bucket == 4)
        _atr_34 = _atr_3 | _atr_4

        _h4_0 = (_h4 == 0)
        _h4_0_2 = (_h4 == 0) | (_h4 == 2)

        _mom3_neg = (_mom3 < 0.0)
        _mom5_neg = (_mom5 < 0.0)
        _mom3_5_neg = _mom3_neg & _mom5_neg
        _dist_neg = (_dist_ema < 0.0)
        _dist_neg_strong = (_dist_ema < -0.3)

        _variants = [
            {
                "name": "O_S3_H4_0_MOM35_DISTNEG",
                "sess": _sess_overlap,
                "atr": _atr_3,
                "h4": _h4_0,
                "mom": _mom3_5_neg,
                "dist": _dist_neg,
                "share_label": "overlap_short",
            },
            {
                "name": "O_S34_H4_0_MOM35_DISTNEG",
                "sess": _sess_overlap,
                "atr": _atr_34,
                "h4": _h4_0,
                "mom": _mom3_5_neg,
                "dist": _dist_neg,
                "share_label": "overlap_short",
            },
            {
                "name": "EU_S3_H4_0_MOM35_DISTNEG",
                "sess": _sess_eu,
                "atr": _atr_3,
                "h4": _h4_0,
                "mom": _mom3_5_neg,
                "dist": _dist_neg,
                "share_label": "eu_short",
            },
            {
                "name": "EUO_S3_H4_0_MOM35_DISTNEG",
                "sess": _sess_eu_overlap,
                "atr": _atr_3,
                "h4": _h4_0,
                "mom": _mom3_5_neg,
                "dist": _dist_neg,
                "share_label": "eu_or_overlap_short",
            },
            {
                "name": "O_S3_H4_0_MOM5_DISTNEG",
                "sess": _sess_overlap,
                "atr": _atr_3,
                "h4": _h4_0,
                "mom": _mom5_neg,
                "dist": _dist_neg,
                "share_label": "overlap_short",
            },
            {
                "name": "O_S3_H4_0_MOM35_DISTNEG_STRONG",
                "sess": _sess_overlap,
                "atr": _atr_3,
                "h4": _h4_0,
                "mom": _mom3_5_neg,
                "dist": _dist_neg_strong,
                "share_label": "overlap_short",
            },
            {
                "name": "O_S3_H4_0_MOM35_NODIST",
                "sess": _sess_overlap,
                "atr": _atr_3,
                "h4": _h4_0,
                "mom": _mom3_5_neg,
                "dist": np.ones(len(merged3), dtype=bool),
                "share_label": "overlap_short",
            },
            {
                "name": "O_S3_H4_0OR2_MOM35_DISTNEG",
                "sess": _sess_overlap,
                "atr": _atr_3,
                "h4": _h4_0_2,
                "mom": _mom3_5_neg,
                "dist": _dist_neg,
                "share_label": "overlap_short",
            },
            {
                "name": "EUO_S34_H4_0_MOM35_DISTNEG",
                "sess": _sess_eu_overlap,
                "atr": _atr_34,
                "h4": _h4_0,
                "mom": _mom3_5_neg,
                "dist": _dist_neg,
                "share_label": "eu_or_overlap_short",
            },
            {
                "name": "O_S3_H4_0OR2_MOM35_DISTNEG_ACTIVE",
                "sess": _sess_overlap,
                "atr": _atr_3,
                "h4": _h4_0_2,
                "mom": _mom3_5_neg,
                "dist": _dist_neg,
                "share_label": "overlap_short",
            },
            {
                "name": "O_S3_H4_0OR2_MOM5_DISTNEG",
                "sess": _sess_overlap,
                "atr": _atr_3,
                "h4": _h4_0_2,
                "mom": _mom5_neg,
                "dist": _dist_neg,
                "share_label": "overlap_short",
            },
            {
                "name": "O_S3_H4_0OR2_MOM35_NODIST",
                "sess": _sess_overlap,
                "atr": _atr_3,
                "h4": _h4_0_2,
                "mom": _mom3_5_neg,
                "dist": np.ones(len(merged3), dtype=bool),
                "share_label": "overlap_short",
            },
            {
                "name": "O_S34_H4_0OR2_MOM35_DISTNEG",
                "sess": _sess_overlap,
                "atr": _atr_34,
                "h4": _h4_0_2,
                "mom": _mom3_5_neg,
                "dist": _dist_neg,
                "share_label": "overlap_short",
            },
        ]

        _share_denoms = {
            "overlap_short": int((_sess_overlap & _is_short).sum()),
            "eu_short": int((_sess_eu & _is_short).sum()),
            "eu_or_overlap_short": int((_sess_eu_overlap & _is_short).sum()),
        }

        for v in _variants:
            _mask_raw = _is_short_raw & v["sess"] & v["atr"] & v["h4"] & v["mom"] & v["dist"]
            _mask_eff = _is_short & v["sess"] & v["atr"] & v["h4"] & v["mom"] & v["dist"]
            _n_raw = int(_mask_raw.sum())
            _n_eff = int(_mask_eff.sum())
            _share_label = v["share_label"]
            _denom = _share_denoms.get(_share_label, 0)
            _share = (_n_eff / _denom) if _denom > 0 else 0.0
            log.info(
                "[OVERLAP_SHORT_MATRIX_RAW_PROOF] variant=%s n_rows=%d n_raw_match=%d",
                v["name"],
                len(merged3),
                _n_raw,
            )
            log.info(
                "[OVERLAP_SHORT_MATRIX_EFFECTIVE_PROOF] variant=%s n_rows=%d n_effective_match=%d "
                "share_%s=%.6f",
                v["name"],
                len(merged3),
                _n_eff,
                _share_label,
                _share,
            )
    else:
        log.warning(
            "[OVERLAP_SHORT_MATRIX_SKIP] missing_cols=%s",
            [c for c in _OVERLAP_SHORT_MATRIX_COLS if c not in merged3.columns],
        )

    # ---------------------------------------------------------------------------
    # OVERLAP long relabel
    #
    # Canonical lane keeps exactly one general relabel profile here:
    #   V4_OL_LONG_B1
    #
    # B1 criteria, explicitly:
    # - session_id == OVERLAP
    # - y_direction == LONG
    # - H4_trend_sign_cat == 0
    # - atr_bucket in {3,4}
    # - D1_atr_percentile_252 >= 0.5 (mid/high)
    # - micro_momentum_3 < 0
    # - micro_momentum_5 < 0
    # - distance_ema_fast is clearly negative (stricter than < 0), i.e. dist_ema_fast < -0.3
    # ---------------------------------------------------------------------------
    _OVERLAP_LONG_COLS = (
        "session_id",
        "H4_trend_sign_cat",
        "atr_bucket",
        "D1_atr_percentile_252",
        "micro_momentum_3",
        "micro_momentum_5",
        "distance_ema_fast",
    )
    if all(c in merged3.columns for c in _OVERLAP_LONG_COLS):
        _sess = merged3["session_id"].astype(np.int64).to_numpy()
        _h4 = merged3["H4_trend_sign_cat"].astype(np.int64).to_numpy()
        _atr_bucket = merged3["atr_bucket"].astype(np.int64).to_numpy()
        _d1_atr = merged3["D1_atr_percentile_252"].astype(np.float32).to_numpy()
        _mom3 = merged3["micro_momentum_3"].astype(np.float32).to_numpy()
        _mom5 = merged3["micro_momentum_5"].astype(np.float32).to_numpy()
        _dist_ema = merged3["distance_ema_fast"].astype(np.float32).to_numpy()

        _is_overlap = (_sess == 2)  # OVERLAP
        _is_long = (y_dir == 0)     # LONG (effective, after earlier relabels)
        _is_long_raw = (y_dir_raw == 0)  # LONG (raw, before any relabels)
        _h4_0 = (_h4 == 0)
        _atr_34 = (_atr_bucket == 3) | (_atr_bucket == 4)
        _d1_mid_high = (_d1_atr >= 0.5)
        _mom_neg = (_mom3 < 0.0) & (_mom5 < 0.0)
        # Use existing natural cutoff for "clearly negative" distance_ema_fast.
        _dist_neg_strong = (_dist_ema < -0.3)
        _raw_mask = _is_overlap & _is_long_raw & _h4_0 & _atr_34 & _d1_mid_high & _mom_neg & _dist_neg_strong
        _mask = _is_overlap & _is_long & _h4_0 & _atr_34 & _d1_mid_high & _mom_neg & _dist_neg_strong

        _n_overlap_long_raw = int(_raw_mask.sum())
        _n_overlap_long_eff = int(_mask.sum())
        _overlap_long_total = int((_is_overlap & _is_long).sum())
        if _n_overlap_long_eff > 0:
            y_dir = y_dir.copy()
            y_qual = y_qual.copy()
            y_early = y_early.copy()
            y_dir[_mask] = 2
            y_qual[_mask] = 0.0
            y_early[_mask] = 0.0
        log.info(
            "[OVERLAP_LONG_RELABEL_PROOF] variant=%s n_rows=%d n_raw_match=%d "
            "n_effective_relabel=%d share_of_overlap_long=%.6f",
            "V4_OL_LONG_B1",
            len(merged3),
            _n_overlap_long_raw,
            _n_overlap_long_eff,
            (_n_overlap_long_eff / _overlap_long_total) if _overlap_long_total > 0 else 0.0,
        )
    else:
        log.warning(
            "[OVERLAP_LONG_RELABEL_SKIP] missing_cols=%s",
            [c for c in _OVERLAP_LONG_COLS if c not in merged3.columns],
        )

    # ---------------------------------------------------------------------------
    # US short relabel
    #
    # Canonical lane keeps this parked at BASELINE (no relabel).
    # ---------------------------------------------------------------------------
    _US_SHORT_COLS = (
        "session_id",
        "H4_trend_sign_cat",
        "atr_bucket",
        "D1_atr_percentile_252",
        "micro_momentum_3",
        "micro_momentum_5",
        "distance_ema_fast",
    )
    if all(c in merged3.columns for c in _US_SHORT_COLS):
        _sess = merged3["session_id"].astype(np.int64).to_numpy()

        _is_us = (_sess == 3)
        _is_short = (y_dir == 1)
        _raw_mask = np.zeros(len(merged3), dtype=bool)
        _mask = np.zeros(len(merged3), dtype=bool)
        _n_us_short_raw = int(_raw_mask.sum())
        _n_us_short_eff = int(_mask.sum())
        _us_short_total = int((_is_us & _is_short).sum())
        log.info(
            "[US_SHORT_RELABEL_PROOF] variant=%s n_rows=%d n_raw_match=%d n_effective_relabel=%d "
            "share_of_us_short=%.6f",
            "V4_US_SHORT_BASELINE",
            len(merged3),
            _n_us_short_raw,
            _n_us_short_eff,
            (_n_us_short_eff / _us_short_total) if _us_short_total > 0 else 0.0,
        )
    else:
        log.warning(
            "[US_SHORT_RELABEL_SKIP] missing_cols=%s",
            [c for c in _US_SHORT_COLS if c not in merged3.columns],
        )

    # ---------------------------------------------------------------------------
    # Quality/tradability targets (post-relabel):
    # - Main direction label remains directional truth after explicit relabel vetoes.
    # - Early-path quality decides y_tradable and auxiliary quality targets.
    # - This keeps "direction" separate from "should we actually take it now?",
    #   which is already handled by the tradable / quality heads in runtime.
    # ---------------------------------------------------------------------------
    # Relabel veto = any rule that flipped y_dir from raw label (all relabels only force FLAT)
    relabel_veto = (y_dir != y_dir_raw)
    y_dir_directional = y_dir.copy()

    _mfe_long = merged3["mfe_long_first_n_bps"].astype(np.float32).to_numpy()
    _mae_long = merged3["mae_long_first_n_bps"].astype(np.float32).to_numpy()
    _mfe_short = merged3["mfe_short_first_n_bps"].astype(np.float32).to_numpy()
    _mae_short = merged3["mae_short_first_n_bps"].astype(np.float32).to_numpy()
    _path_long = (_mfe_long - _mae_long).astype(np.float32)
    _path_short = (_mfe_short - _mae_short).astype(np.float32)

    _tradable_long = (_mfe_long >= 16.0) & (_mae_long <= 3.0) & (_path_long >= 6.0)
    _tradable_short = (_mfe_short >= 16.0) & (_mae_short <= 3.0) & (_path_short >= 6.0)

    # Choose side based on best path quality (tie-break by MFE)
    _side = np.full(len(merged3), -1, dtype=np.int8)  # -1 none, 0 long, 1 short
    _only_long = _tradable_long & ~_tradable_short
    _only_short = _tradable_short & ~_tradable_long
    _both = _tradable_long & _tradable_short
    _side[_only_long] = 0
    _side[_only_short] = 1
    if _both.any():
        _prefer_long = _path_long >= _path_short
        _prefer_short = _path_short > _path_long
        _side[_both & _prefer_long] = 0
        _side[_both & _prefer_short] = 1
        _tie = _both & (~_prefer_long) & (~_prefer_short)
        if _tie.any():
            _side[_tie & (_mfe_long >= _mfe_short)] = 0
            _side[_tie & (_mfe_short > _mfe_long)] = 1

    # Apply relabel veto to tradability as well: explicit poison pockets stay non-tradable
    _side[relabel_veto] = -1

    # Final tradability / quality targets
    y_tradable = (_side != -1).astype(np.int32)
    y_dir = y_dir_directional

    # Quality auxiliaries align to the post-relabel directional side, not the stricter
    # tradable side. This keeps the main direction truth and the quality heads in the
    # same semantic world while preserving tradability as its own, stricter runtime gate.
    _quality_side = np.full(len(merged3), -1, dtype=np.int8)  # -1 none, 0 long, 1 short
    _quality_side[y_dir_directional == 0] = 0
    _quality_side[y_dir_directional == 1] = 1

    y_mfe_first_n = np.zeros_like(y_mfe_first_n)
    y_mae_first_n = np.zeros_like(y_mae_first_n)
    y_path_quality = np.zeros_like(y_path_quality)
    y_mfe_first_n[_quality_side == 0] = _mfe_long[_quality_side == 0]
    y_mfe_first_n[_quality_side == 1] = _mfe_short[_quality_side == 1]
    y_mae_first_n[_quality_side == 0] = _mae_long[_quality_side == 0]
    y_mae_first_n[_quality_side == 1] = _mae_short[_quality_side == 1]
    y_path_quality[_quality_side == 0] = _path_long[_quality_side == 0]
    y_path_quality[_quality_side == 1] = _path_short[_quality_side == 1]

    # Early move: align to the directional-quality side instead of tradability side.
    y_early = np.zeros_like(y_early)
    y_early[_quality_side != -1] = (
        y_mfe_first_n[_quality_side != -1] >= float(early_move_threshold_bps)
    ).astype(np.float32)

    # Quality score stays non-negative but now reflects directional path quality.
    y_qual = np.zeros_like(y_qual)
    y_qual[_quality_side != -1] = np.maximum(0.0, y_path_quality[_quality_side != -1]).astype(np.float32)

    _directional_long_rate = float(np.mean(y_dir == 0)) if len(y_dir) else 0.0
    _directional_short_rate = float(np.mean(y_dir == 1)) if len(y_dir) else 0.0
    _directional_flat_rate = float(np.mean(y_dir == 2)) if len(y_dir) else 0.0
    log.info(
        "[ENTRY_DIRECTION_TARGET_SEMANTICS] split=%s source=post_relabel_directional "
        "long_rate=%.6f short_rate=%.6f flat_rate=%.6f relabel_veto_rate=%.6f",
        split_name or "full",
        _directional_long_rate,
        _directional_short_rate,
        _directional_flat_rate,
        float(np.mean(relabel_veto)) if len(relabel_veto) else 0.0,
    )

    # Tradable rate proof (split-aware)
    _split_tag = split_name or "full"
    _tradable_rate = float(np.mean(y_tradable)) if len(y_tradable) else 0.0
    log.info(
        "[ENTRY_TRADABLE_RATE_PROOF] split=%s n_rows=%d tradable_rate=%.6f",
        _split_tag,
        len(y_tradable),
        _tradable_rate,
    )
    _quality_side_rate = float(np.mean(_quality_side != -1)) if len(_quality_side) else 0.0
    log.info(
        "[ENTRY_QUALITY_SIDE_RATE_PROOF] split=%s n_rows=%d quality_side_rate=%.6f",
        _split_tag,
        len(_quality_side),
        _quality_side_rate,
    )
    _side_long = (y_dir == 0)
    _side_short = (y_dir == 1)
    _long_rate = float(np.mean(y_tradable[_side_long])) if _side_long.any() else 0.0
    _short_rate = float(np.mean(y_tradable[_side_short])) if _side_short.any() else 0.0
    log.info(
        "[ENTRY_TRADABLE_RATE_BY_SIDE] split=%s long_rate=%.6f short_rate=%.6f",
        _split_tag,
        _long_rate,
        _short_rate,
    )
    if "session_id" in merged3.columns:
        _sess = merged3["session_id"].astype(np.int64).to_numpy()
        _sess_names = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
        for sid in sorted(set(_sess.tolist())):
            _mask = (_sess == sid)
            _rate = float(np.mean(y_tradable[_mask])) if _mask.any() else 0.0
            log.info(
                "[ENTRY_TRADABLE_RATE_BY_SESSION] split=%s session_id=%d session_name=%s rate=%.6f",
                _split_tag,
                int(sid),
                _sess_names.get(int(sid), "UNK"),
                _rate,
            )

    n = len(merged3)
    if n < (seq_len + 1):
        raise RuntimeError(f"TOO_FEW_ROWS_FOR_SEQ: rows={n} seq_len={seq_len}")

    rows: List[Dict[str, Any]] = []
    # Start index at seq_len-1 so we have a full history ending at i
    for i in range(seq_len - 1, n):
        seq = sig_mat[i - (seq_len - 1) : i + 1]  # [seq_len, 7]
        snap = sig_mat[i]  # [7]
        rows.append(
            {
                "time": times[i],
                "seq": seq,
                "snap": snap,
                "ctx_cont": ctx_cont_mat[i],
                "ctx_cat": ctx_cat_mat[i],
                "y_direction": y_dir[i],
                "y_early_move": y_early[i],
                "y_quality_score": y_qual[i],
                "y_bad_path": y_bad_path[i],
                "y_tradable": y_tradable[i],
                "mae_first_n_bps": y_mae_first_n[i],
                "mfe_first_n_bps": y_mfe_first_n[i],
                "path_quality_bps": y_path_quality[i],
                "label_horizon_bars": y_label_horizon[i],
                "path_quality_horizon_bars": y_path_horizon[i],
            }
        )

    df_out = pd.DataFrame(rows)
    if len(df_out) == 0:
        raise RuntimeError("BUILD_EMPTY_OUTPUT")

    # Parquet can only serialize list-like columns (PyArrow); convert arrays to lists
    # -------------------------------------------------------------------------
    # Parquet-safe serialization: enforce pure Python lists (no numpy arrays)
    # -------------------------------------------------------------------------
    def _to_list(x):
        # Handles numpy arrays, lists, tuples; returns pure Python list (deep)
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return [_to_list(v) for v in x]
        return x

    df_out["seq"] = df_out["seq"].apply(_to_list)
    df_out["snap"] = df_out["snap"].apply(_to_list)
    df_out["ctx_cont"] = df_out["ctx_cont"].apply(_to_list)
    df_out["ctx_cat"] = df_out["ctx_cat"].apply(_to_list)

    mae_missing = int(pd.isna(df_out["mae_first_n_bps"]).sum())
    mfe_missing = int(pd.isna(df_out["mfe_first_n_bps"]).sum())
    log.info(
        "[ENTRY_PATH_QUALITY_PROOF] rows=%d mae_missing=%d mfe_missing=%d",
        int(len(df_out)),
        mae_missing,
        mfe_missing,
    )
    log.info(
        "[ENTRY_INPUT_SCHEMA_PROOF] signal_dim=7 ctx_cont_dim=%d ctx_cat_dim=6",
        int(len(ctx_cont_names)),
    )

    # 10) Metadata
    _hold_bars = int(horizon_bars)
    meta: Dict[str, Any] = {
        "rows": int(len(df_out)),
        "seq_len": int(seq_len),
        "hold_bars": _hold_bars,
        "early_move_threshold_bps": float(early_move_threshold_bps),
        "flat_threshold_bps": float(flat_threshold_bps),
        "base28_manifest": {
            "path": str(base28_manifest_path),
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_sha,
        },
        "xgb_bundle": str(Path(xgb_bundle_path).resolve()),
        "xgb_model_sha256": xgb_model_sha256,
        "tape_root": str(Path(tape_root).resolve()),
        "join_ratio_tape": float(rows_joined / max(1, rows_base28)),
        "signal_bridge": {
            "id": "XGB_SIGNAL_BRIDGE_V1",
            "fields": list(SIGNAL_FIELDS),
            "contract_sha256": SIGNAL_CONTRACT_SHA256,
            "proba_dim_seen": 3,
            "bridge_dim": 7,
        },
        "ctx_contract": {
            "tag": ctx["tag"],
            "ctx_cont_dim": int(len(ctx_cont_names)),
            "ctx_cat_dim": int(ctx["ctx_cat_dim"]),
            "ctx_cont_names": list(ctx_cont_names),
            "ctx_cat_names": list(ctx_cat_names),
            "allow_zero_ctx": bool(allow_zero_ctx),
            "ctx_cont_base_dim": int(ctx["ctx_cont_dim"]),
            "ctx_cont_micro_features": list(MICRO_FEATURE_NAMES),
            "ctx_cont_swing_features": list(SWING_FEATURE_NAMES),
            "ctx_cont_session_features": list(SESSION_CTX_CONT_NAMES),
        },
        "lane_contract": {
            "entry_admission_policy": "OVERLAP_LONG_REPLACES_OLDEST_OVERLAP_SHORT_WHEN_FULL",
            "entry_runtime_gates": [
                "flat_veto",
                "tradable_gate",
                "quality_gate",
            ],
            "max_open_trades": 10,
        },
        "parked_targets": {
            "bad_path": {
                "horizon_bars": int(BAD_PATH_HORIZON_BARS),
                "mae_threshold_bps": float(BAD_PATH_MAE_THRESHOLD_BPS),
                "mfe_threshold_bps": float(BAD_PATH_MFE_THRESHOLD_BPS),
            }
        },
        "base28_feature_contract": {
            "features": list(features),
            "contract_path": str(contract_path),
        },
    }

    return df_out, meta


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ENTRY_V10_CTX training dataset (canonical, CTX6CAT6-only; advanced seq/snap/ctx structure)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--truth-config",
        type=str,
        required=False,
        help="Path to canonical truth config JSON. If provided, base28_manifest and xgb_bundle are resolved from it.",
    )
    parser.add_argument(
        "--base28_manifest",
        type=str,
        required=False,
        help="Path to BASE28_CANONICAL CURRENT_MANIFEST.json (manifest-only resolution). Optional when --truth-config is set.",
    )
    parser.add_argument(
        "--xgb_bundle",
        type=str,
        required=False,
        help="Path to canonical XGB bundle directory (universal multihead v2; locked). Optional when --truth-config is set.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output dataset path (.parquet).")

    # Deterministic filters
    parser.add_argument("--start", type=str, default=None, help="Start datetime (ISO; UTC recommended).")
    parser.add_argument("--end", type=str, default=None, help="End datetime (ISO; UTC recommended).")
    parser.add_argument("--max_rows", type=int, default=None, help="Deterministic: take first N rows after filtering.")

    # Advanced dataset structure
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length for seq feature (default: 30).")

    # Labels (fixed-hold)
    parser.add_argument("--hold-bars", dest="hold_bars", type=int, default=3, help="Fixed-hold label horizon in M5 bars (default: 3). Must be between 1 and 50.")
    parser.add_argument("--early_move_threshold_bps", type=float, default=4.0, help="Early-move threshold in bps (default: 4.0).")
    parser.add_argument("--flat_threshold_bps", type=float, default=3.0, help="Flat threshold in bps for 3-class labels (default: 3.0).")

    # Tape lane
    parser.add_argument(
        "--tape_root",
        type=str,
        default="",
        help="Override canonical tape lane root. Default resolves from $GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL",
    )

    # CTX handling
    parser.add_argument(
        "--allow_zero_ctx",
        action="store_true",
        help="If ctx columns are missing in BASE28, fill zeros instead of hard-fail (NOT recommended for final).",
    )

    # Output splitting scaffolding (kept for parity)
    parser.add_argument("--time_split", action="store_true", help="Write train/val/test outputs (time-based).")
    parser.add_argument("--train_start", type=str, default="2025-01-01T00:00:00Z", help="Train split start (ISO).")
    parser.add_argument("--train_end", type=str, default="2025-09-30T23:59:59Z", help="Train split end (ISO).")
    parser.add_argument("--val_start", type=str, default="2025-10-01T00:00:00Z", help="Val split start (ISO).")
    parser.add_argument("--val_end", type=str, default="2025-11-30T23:59:59Z", help="Val split end (ISO).")
    parser.add_argument("--test_start", type=str, default="2025-12-01T00:00:00Z", help="Test split start (ISO).")
    parser.add_argument("--test_end", type=str, default="2025-12-31T23:59:59Z", help="Test split end (ISO).")

    parser.add_argument("--dry_run", action="store_true", help="Dry run: validate inputs/ctx, then exit.")

    args = parser.parse_args()
    build_command = sys.argv.copy()

    # Hard gate: ONE UNIVERSE
    ctx = _hard_gate_ctx6cat6()
    log.info(f"[CTX_CONTRACT] OK: tag={ctx['tag']} cont={ctx['ctx_cont_dim']} cat={ctx['ctx_cat_dim']}")

    hold_bars = int(args.hold_bars)
    if hold_bars < 1 or hold_bars > 50:
        raise ValueError(f"HOLD_BARS_INVALID: {hold_bars} (must be 1..50)")
    log.info("[LABEL_HOLD] hold_bars=%d", hold_bars)
    flat_threshold_bps = float(args.flat_threshold_bps)
    if flat_threshold_bps < 0:
        raise ValueError(f"FLAT_THRESHOLD_INVALID: {flat_threshold_bps} (must be >=0)")
    log.info("[LABEL_FLAT] flat_threshold_bps=%.4f", flat_threshold_bps)

    # Dataset build proof (will be written after output_path resolved)
    proof_payload = {
        "ctx_tag": ctx.get("tag"),
        "ctx_cont_dim": int(ctx.get("ctx_cont_dim", -1)),
        "ctx_cat_dim": int(ctx.get("ctx_cat_dim", -1)),
        "signal_bridge_id": "XGB_SIGNAL_BRIDGE_V1",
        "signal_bridge_contract_sha256": str(SIGNAL_CONTRACT_SHA256),
        "hold_bars": hold_bars,
        "flat_threshold_bps": float(flat_threshold_bps),
    }

    # Resolve SSoT inputs (truth-config or manual)
    truth_config_path: Optional[Path] = None
    if args.truth_config:
        truth_config_path = Path(args.truth_config).expanduser().resolve()
        if args.base28_manifest or args.xgb_bundle:
            raise SystemExit("[SPLIT_BRAIN_ARGS] truth-config provided but base28_manifest/xgb_bundle also supplied")
        if not truth_config_path.exists():
            raise RuntimeError(f"TRUTH_CONFIG_MISSING: {truth_config_path}")
        truth_obj = json.loads(truth_config_path.read_text())
        base28_manifest_path = Path(
            truth_obj.get(
                "canonical_prebuilt_manifest",
                "/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json",
            )
        ).expanduser().resolve()
        canonical_xgb_bundle_dir = str(truth_obj.get("canonical_xgb_bundle_dir") or "").strip()
        if not canonical_xgb_bundle_dir:
            raise RuntimeError(
                f"TRUTH_CONFIG_MISSING_CANONICAL_XGB_BUNDLE: canonical_xgb_bundle_dir missing in {truth_config_path}"
            )
        xgb_bundle_path = Path(canonical_xgb_bundle_dir).expanduser().resolve()
        xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
        if xgb_override:
            override_path = Path(xgb_override).expanduser().resolve()
            if override_path != xgb_bundle_path:
                log.info(
                    "[XGB_OVERRIDE] truth_config_bundle=%s override_bundle=%s",
                    xgb_bundle_path,
                    override_path,
                )
            xgb_bundle_path = override_path
        log.info(
            "[TRUTH_CONFIG] Using truth-config %s -> base28_manifest=%s xgb_bundle=%s",
            truth_config_path,
            base28_manifest_path,
            xgb_bundle_path,
        )
        proof_payload.update(
            {
                "truth_config_path": str(truth_config_path),
                "truth_source": "truth_config",
            }
        )
    else:
        if not args.base28_manifest or not args.xgb_bundle:
            raise SystemExit("Both --base28_manifest and --xgb_bundle are required when --truth-config is not provided")
        base28_manifest_path = Path(args.base28_manifest).resolve()
        xgb_bundle_path = Path(args.xgb_bundle).resolve()
        xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
        if xgb_override:
            override_path = Path(xgb_override).expanduser().resolve()
            if override_path != xgb_bundle_path:
                raise RuntimeError(
                    f"[XGB_OVERRIDE_MISMATCH] GX1_XGB_BUNDLE_DIR={override_path} != --xgb_bundle={xgb_bundle_path}"
                )
            xgb_bundle_path = override_path
        proof_payload.update({"truth_config_path": None, "truth_source": "manual_args"})
    _ensure_inputs_exist(base28_manifest_path, xgb_bundle_path)

    output_path = Path(args.output).resolve()
    hold_suffix = f"HOLD_{hold_bars:02d}B"
    if hold_suffix not in output_path.stem:
        output_path = output_path.with_name(f"{output_path.stem}__{hold_suffix}{output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write proof payload
    try:
        proof_path = output_path.parent / "DATASET_BUILD_PROOF.json"
        with open(proof_path, "w") as f:
            json.dump(proof_payload, f, indent=2)
        log.info("[DATASET_BUILD_PROOF] wrote %s", proof_path)
    except Exception as e:
        log.warning("[DATASET_BUILD_PROOF] failed to write proof file: %s", e)
    proof_payload.update(
        {
            "base28_manifest_path": str(base28_manifest_path),
            "xgb_bundle_path": str(xgb_bundle_path),
            "output_path": str(output_path),
        }
    )

    model_file = xgb_bundle_path / "xgb_universal_multihead_v2.joblib"
    if not model_file.exists():
        raise RuntimeError(f"XGB_MODEL_MISSING: {model_file}")
    xgb_model_sha256 = _sha256_file(model_file)
    proof_payload["xgb_model_sha256"] = xgb_model_sha256
    log.info(
        "[XGB_BUNDLE_PROOF] xgb_bundle=%s model_file=%s model_sha256=%s",
        xgb_bundle_path,
        model_file,
        xgb_model_sha256,
    )

    start = _parse_ts(args.start)
    end = _parse_ts(args.end)

    # Tape root resolution
    if args.tape_root.strip():
        tape_root = Path(args.tape_root).expanduser().resolve()
    else:
        gx1_data = _resolve_gx1_data_root()
        tape_root = gx1_data / "data" / "oanda" / "canonical" / "xauusd_m5_bid_ask__CANONICAL"

    if args.dry_run:
        log.info("[DRY_RUN] Inputs exist and CTX contract is valid. Exiting.")
        write_manifest(
            output_path=output_path,
            build_command=build_command,
            base28_manifest=base28_manifest_path,
            xgb_bundle=xgb_bundle_path,
            tape_root=tape_root,
            notes="DRY_RUN only.",
            extra={
                "start": args.start,
                "end": args.end,
                "max_rows": args.max_rows,
                "time_split": bool(args.time_split),
                "seq_len": int(args.seq_len),
                "hold_bars": int(hold_bars),
                "early_move_threshold_bps": float(args.early_move_threshold_bps),
                "allow_zero_ctx": bool(args.allow_zero_ctx),
                "xgb_model_sha256": xgb_model_sha256,
            },
        )
        return

    if args.time_split:
        train_start = _parse_ts(args.train_start)
        train_end = _parse_ts(args.train_end)
        val_start = _parse_ts(args.val_start)
        val_end = _parse_ts(args.val_end)
        test_start = _parse_ts(args.test_start)
        test_end = _parse_ts(args.test_end)

        splits = {
            "train": {"start": str(train_start), "end": str(train_end)},
            "val": {"start": str(val_start), "end": str(val_end)},
            "test": {"start": str(test_start), "end": str(test_end)},
        }

        base = output_path
        out_dir = base.parent
        stem = base.stem

        metas: Dict[str, Any] = {}
        ts_min_max_by_split: Dict[str, Dict[str, Optional[str]]] = {}

        for split_name, (s0, s1) in {
            "train": (train_start, train_end),
            "val": (val_start, val_end),
            "test": (test_start, test_end),
        }.items():
            log.info(f"[BUILD] split={split_name} start={s0} end={s1}")
            df_built, meta = build_dataset_canonical(
                base28_manifest_path=base28_manifest_path,
                xgb_bundle_path=xgb_bundle_path,
                tape_root=tape_root,
                start=s0,
                end=s1,
                max_rows=args.max_rows,
                seq_len=int(args.seq_len),
                horizon_bars=int(hold_bars),
                early_move_threshold_bps=float(args.early_move_threshold_bps),
                flat_threshold_bps=float(flat_threshold_bps),
                allow_zero_ctx=bool(args.allow_zero_ctx),
                split_name=split_name,
            )
            _log_label_distribution_proof(df_built, split=split_name)
            out = out_dir / f"{stem}_{split_name}.parquet"
            df_built.to_parquet(out, index=False)
            metas[split_name] = deepcopy(meta)

            ts_min_max_by_split[split_name] = _split_min_max_from_ts_series(df_built["time"])

            write_manifest(
                output_path=out,
                build_command=build_command,
                base28_manifest=base28_manifest_path,
                xgb_bundle=xgb_bundle_path,
                tape_root=tape_root,
                splits=splits,
                ts_min_max_by_split=ts_min_max_by_split,
                notes=f"Canonical build completed for split={split_name}.",
                extra=metas[split_name],
            )

        log.info("[DATASET_BUILD] Time-split build complete!")
        return

    # Single output
    df_built, meta = build_dataset_canonical(
        base28_manifest_path=base28_manifest_path,
        xgb_bundle_path=xgb_bundle_path,
        tape_root=tape_root,
        start=start,
        end=end,
        max_rows=args.max_rows,
        seq_len=int(args.seq_len),
        horizon_bars=int(hold_bars),
        early_move_threshold_bps=float(args.early_move_threshold_bps),
        flat_threshold_bps=float(flat_threshold_bps),
        allow_zero_ctx=bool(args.allow_zero_ctx),
        split_name="full",
    )
    _log_label_distribution_proof(df_built, split="SINGLE")
    df_built.to_parquet(output_path, index=False)
    log.info(f"✅ Saved dataset: {output_path}")

    ts_min_max_by_split = {"SINGLE": _split_min_max_from_ts_series(df_built["time"])}

    write_manifest(
        output_path=output_path,
        build_command=build_command,
        base28_manifest=base28_manifest_path,
        xgb_bundle=xgb_bundle_path,
        tape_root=tape_root,
        splits=None,
        ts_min_max_by_split=ts_min_max_by_split,
        notes="Canonical build completed (single).",
        extra=meta,
    )

    log.info("[DATASET_BUILD] Dataset build complete!")


if __name__ == "__main__":
    main()
