#!/usr/bin/env python3
"""
Build ENTRY_V10_CTX training dataset (canonical, CTX6CAT6-only).

SSoT / ONE UNIVERSE:
- ctx contract: CTX6CAT6 (6/6) is the ONLY valid universe
- signal bridge: XGB_SIGNAL_BRIDGE_V1 (7-dim)
- Inputs must be canonical:
  - BASE28 prebuilt via CURRENT_MANIFEST.json (manifest-only resolution; sha256 verify; no direct parquet path)
  - canonical XGB bundle (universal multihead v2; ordered_features=BASE28; locked sessions)
  - canonical market tape lane (bid/ask) for deterministic label building (close after N bars)

Outputs (advanced structure, compatible with "old idea"):
- time: tz-aware UTC timestamp
- seq: list/ndarray shaped [seq_len, 7]  (signal bridge sequence)
- snap: ndarray shaped [7]              (signal bridge snapshot)
- ctx_cont: ndarray shaped [6]
- ctx_cat: ndarray shaped [6]
- y_direction: float32 (0/1)            (label computed from tape with fixed-hold exit)
- y_early_move: float32 (0/1)           (label computed from tape within horizon)
- y_quality_score: float32              (e.g. abs pnl bps over horizon)

NO FALLBACKS unless explicitly allowed by CLI flags.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
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
from gx1.utils.canonical_prebuilt_resolver import resolve_base28_canonical_from_manifest
from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel
from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


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
    """Fail-fast: ensure the canonical ctx contract is CTX6CAT6 (6/6)."""
    ctx = get_canonical_ctx_contract()
    tag = str(ctx.get("tag", ""))
    cont = int(ctx.get("ctx_cont_dim", -1))
    cat = int(ctx.get("ctx_cat_dim", -1))
    if tag != "CTX6CAT6" or cont != 6 or cat != 6:
        raise RuntimeError(
            f"CTX_CONTRACT_SPLIT_BRAIN: expected CTX6CAT6 (6/6) but got tag={tag} cont={cont} cat={cat}"
        )
    # Names must exist for stable column mapping
    if "ctx_cont_names" not in ctx or "ctx_cat_names" not in ctx:
        raise RuntimeError("CTX_CONTRACT_INVALID: missing ctx_cont_names/ctx_cat_names in contract")
    if len(ctx["ctx_cont_names"]) != 6 or len(ctx["ctx_cat_names"]) != 6:
        raise RuntimeError("CTX_CONTRACT_INVALID: ctx names length must be 6/6")
    return ctx


def _ensure_inputs_exist(base28_manifest: Path, xgb_bundle: Path) -> None:
    if not base28_manifest.exists():
        raise RuntimeError(f"INPUT_MISSING: base28_manifest not found: {base28_manifest}")
    if not xgb_bundle.exists():
        raise RuntimeError(f"INPUT_MISSING: xgb_bundle not found: {xgb_bundle}")
    if base28_manifest.suffix.lower() != ".json":
        raise RuntimeError(f"INPUT_INVALID: base28_manifest must be a .json manifest file: {base28_manifest}")


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
) -> pd.DataFrame:
    """
    Labels are computed from bid/ask close (or bid/ask) with a fixed hold:
    - y_direction: 1 if pnl_bps > 0 for LONG (entry at ask, exit at bid after horizon)
    - y_early_move: 1 if max favorable move within horizon >= threshold (LONG: max bid - entry ask)
    - y_quality_score: abs(pnl_bps) clipped (float32)
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
    exit_bid = bid[horizon_bars:]

    # pnl in bps for LONG: (exit_bid - entry_ask)/entry_ask * 1e4
    pnl_bps = (exit_bid - entry_ask) / np.clip(entry_ask, 1e-12, None) * 1e4

    # Early move: within horizon, max favorable bid vs entry_ask
    # Compute rolling max of bid over next horizon bars (inclusive)
    # For each i in [0, n-horizon-1], consider bid[i+1 .. i+horizon] or [i .. i+horizon]? Use i..i+horizon for simplicity.
    max_fav = np.empty(n - horizon_bars, dtype=np.float64)
    for i in range(0, n - horizon_bars):
        window = bid[i : i + horizon_bars + 1]
        max_fav[i] = float(np.max(window))
    mfe_bps = (max_fav - entry_ask) / np.clip(entry_ask, 1e-12, None) * 1e4

    y_direction = (pnl_bps > 0.0).astype(np.float32)
    y_early = (mfe_bps >= float(early_move_threshold_bps)).astype(np.float32)
    y_quality = np.clip(np.abs(pnl_bps), 0.0, 5000.0).astype(np.float32)

    out = pd.DataFrame(
        {
            "time": tape["time"].iloc[:-horizon_bars].to_numpy(),
            "y_direction": y_direction,
            "y_early_move": y_early,
            "y_quality_score": y_quality,
            "label_horizon_bars": np.int32(horizon_bars),
        }
    )
    return out


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

    manifest: Dict[str, Any] = {
        "created_at": _utc_now_iso(),
        "git_commit": get_git_commit(),
        "output_data_path": str(output_path),
        "build_command": build_command,
        "inputs": {
            "base28_manifest": str(base28_manifest),
            "xgb_bundle": str(xgb_bundle),
            "tape_root": str(tape_root) if tape_root is not None else None,
        },
        "feature_contract": {
            "ctx_tag": str(ctx["tag"]),
            "ctx_cont_dim": int(ctx["ctx_cont_dim"]),
            "ctx_cat_dim": int(ctx["ctx_cat_dim"]),
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
    allow_zero_ctx: bool,
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

    # 3) Enforce BASE28 contract order
    contract_path = project_root / "gx1" / "xgb" / "contracts" / "xgb_input_features_base28_v1.json"
    contract_obj = json.loads(contract_path.read_text(encoding="utf-8"))
    features = contract_obj.get("features") or contract_obj.get("ordered_features") or []
    if len(features) != 28:
        raise RuntimeError("FEATURE_CONTRACT_INVALID_LEN")
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

    model = XGBMultiheadModel.load(str(model_path))
    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_cfg))

    # sanitize (contract-ordered)
    x_array, _stats = sanitizer.sanitize(df_features, feature_list=features, allow_nan_fill=False)
    if x_array is None or len(df_features) != len(df):
        # df_features derived from df; should match
        raise RuntimeError("SANITIZER_OUTPUT_INVALID")

    # 5) Predict per session head (if session_id exists) else US
    session_series = df["session_id"].fillna(2).astype(int) if "session_id" in df.columns else None
    session_map = {0: "EU", 1: "OVERLAP", 2: "US"}

    p_long = np.zeros(len(df), dtype=np.float64)
    p_short = np.zeros(len(df), dtype=np.float64)
    p_flat = np.zeros(len(df), dtype=np.float64)

    def _run_for_session(sess_name: str, idx: np.ndarray) -> None:
        if idx.size == 0:
            return
        probs = model.predict_proba(df_features.iloc[idx], session=sess_name, feature_list=features)
        # Expect attributes or dict-like; support both
        if hasattr(probs, "p_long"):
            p_long[idx] = np.asarray(probs.p_long, dtype=np.float64)
            p_short[idx] = np.asarray(probs.p_short, dtype=np.float64)
            p_flat[idx] = np.asarray(probs.p_flat, dtype=np.float64)
        else:
            p_long[idx] = np.asarray(probs["p_long"], dtype=np.float64)
            p_short[idx] = np.asarray(probs["p_short"], dtype=np.float64)
            p_flat[idx] = np.asarray(probs["p_flat"], dtype=np.float64)

    if session_series is not None:
        for sid, name in session_map.items():
            mask = session_series.values == sid
            idx = np.where(mask)[0]
            _run_for_session(name, idx)
    else:
        _run_for_session("US", np.arange(len(df), dtype=np.int64))

    eps = 1e-12
    p_hat = np.maximum(p_long, p_short)
    top2 = np.sort(np.stack([p_long, p_short, p_flat], axis=0), axis=0)
    margin_top1_top2 = top2[-1] - top2[-2]
    entropy = -(
        p_long * np.log(np.clip(p_long, eps, 1.0))
        + p_short * np.log(np.clip(p_short, eps, 1.0))
        + p_flat * np.log(np.clip(p_flat, eps, 1.0))
    )
    uncertainty_score = entropy

    # 6) Build ctx features
    ctx_cont_names = list(ctx["ctx_cont_names"])
    ctx_cat_names = list(ctx["ctx_cat_names"])

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

    # 7) Assemble per-bar signal dataframe (time aligned)
    df_sig = pd.DataFrame({"time": df["time"].to_numpy()})
    for field in SIGNAL_FIELDS:
        if field == "p_long":
            df_sig[field] = p_long
        elif field == "p_short":
            df_sig[field] = p_short
        elif field == "p_flat":
            df_sig[field] = p_flat
        elif field == "p_hat":
            df_sig[field] = p_hat
        elif field == "uncertainty_score":
            df_sig[field] = uncertainty_score
        elif field == "margin_top1_top2":
            df_sig[field] = margin_top1_top2
        elif field == "entropy":
            df_sig[field] = entropy
        else:
            raise RuntimeError(f"SIGNAL_FIELD_UNKNOWN: {field}")

    # 8) Labels from canonical tape lane (join by time)
    t_min = pd.Timestamp(df_sig["time"].min()).tz_convert("UTC")
    t_max = pd.Timestamp(df_sig["time"].max()).tz_convert("UTC")

    tape = _load_canonical_tape(
        tape_root=tape_root,
        t_min=t_min,
        t_max=t_max,
        required_cols=["bid_close", "ask_close"],
    )

    # Inner join tape to BASE28 by time
    # We keep only matching times (deterministic). Gate join ratio.
    merged = df_sig.merge(tape, on="time", how="inner", validate="one_to_one")
    join_ratio = float(len(merged) / max(1, len(df_sig)))
    if join_ratio < 0.995:
        raise RuntimeError(f"TAPE_JOIN_RATIO_FAIL: join_ratio={join_ratio:.6f} (<0.995)")

    # Compute labels on merged tape
    labels = _compute_labels_fixed_hold(
        tape=merged[["time"] + [c for c in merged.columns if c in ("bid_close", "ask_close", "bid", "ask")]].copy(),
        horizon_bars=horizon_bars,
        early_move_threshold_bps=early_move_threshold_bps,
    )

    # Align signals to labels (labels are shorter by horizon_bars)
    merged2 = merged.merge(labels[["time", "y_direction", "y_early_move", "y_quality_score", "label_horizon_bars"]], on="time", how="inner", validate="one_to_one")
    if len(merged2) == 0:
        raise RuntimeError("LABEL_JOIN_EMPTY")

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

    y_dir = merged3["y_direction"].astype(np.float32).to_numpy()
    y_early = merged3["y_early_move"].astype(np.float32).to_numpy()
    y_qual = merged3["y_quality_score"].astype(np.float32).to_numpy()

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
            }
        )

    df_out = pd.DataFrame(rows)
    if len(df_out) == 0:
        raise RuntimeError("BUILD_EMPTY_OUTPUT")

    # Parquet can only serialize list-like columns (PyArrow); convert arrays to lists
    df_out["seq"] = df_out["seq"].apply(lambda a: a.tolist() if hasattr(a, "tolist") else list(a))
    df_out["snap"] = df_out["snap"].apply(lambda a: a.tolist() if hasattr(a, "tolist") else list(a))
    df_out["ctx_cont"] = df_out["ctx_cont"].apply(lambda a: a.tolist() if hasattr(a, "tolist") else list(a))
    df_out["ctx_cat"] = df_out["ctx_cat"].apply(lambda a: a.tolist() if hasattr(a, "tolist") else list(a))

    # 10) Metadata
    meta: Dict[str, Any] = {
        "rows": int(len(df_out)),
        "seq_len": int(seq_len),
        "horizon_bars": int(horizon_bars),
        "early_move_threshold_bps": float(early_move_threshold_bps),
        "base28_manifest": {
            "path": str(base28_manifest_path),
            "parquet_path": str(parquet_path),
            "parquet_sha256": parquet_sha,
        },
        "xgb_bundle": str(Path(xgb_bundle_path).resolve()),
        "tape_root": str(Path(tape_root).resolve()),
        "join_ratio_tape": float(join_ratio),
        "signal_bridge": {
            "id": "XGB_SIGNAL_BRIDGE_V1",
            "fields": list(SIGNAL_FIELDS),
            "contract_sha256": SIGNAL_CONTRACT_SHA256,
        },
        "ctx_contract": {
            "tag": ctx["tag"],
            "ctx_cont_dim": int(ctx["ctx_cont_dim"]),
            "ctx_cat_dim": int(ctx["ctx_cat_dim"]),
            "ctx_cont_names": list(ctx_cont_names),
            "ctx_cat_names": list(ctx_cat_names),
            "allow_zero_ctx": bool(allow_zero_ctx),
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
        "--base28_manifest",
        type=str,
        required=True,
        help="Path to BASE28_CANONICAL CURRENT_MANIFEST.json (manifest-only resolution).",
    )
    parser.add_argument(
        "--xgb_bundle",
        type=str,
        required=True,
        help="Path to canonical XGB bundle directory (universal multihead v2; locked).",
    )
    parser.add_argument("--output", type=str, required=True, help="Output dataset path (.parquet).")

    # Deterministic filters
    parser.add_argument("--start", type=str, default=None, help="Start datetime (ISO; UTC recommended).")
    parser.add_argument("--end", type=str, default=None, help="End datetime (ISO; UTC recommended).")
    parser.add_argument("--max_rows", type=int, default=None, help="Deterministic: take first N rows after filtering.")

    # Advanced dataset structure
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length for seq feature (default: 30).")

    # Labels (fixed-hold)
    parser.add_argument("--horizon_bars", type=int, default=3, help="Label horizon in M5 bars (default: 3).")
    parser.add_argument("--early_move_threshold_bps", type=float, default=4.0, help="Early-move threshold in bps (default: 4.0).")

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

    base28_manifest_path = Path(args.base28_manifest).resolve()
    xgb_bundle_path = Path(args.xgb_bundle).resolve()
    _ensure_inputs_exist(base28_manifest_path, xgb_bundle_path)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                "horizon_bars": int(args.horizon_bars),
                "early_move_threshold_bps": float(args.early_move_threshold_bps),
                "allow_zero_ctx": bool(args.allow_zero_ctx),
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
                horizon_bars=int(args.horizon_bars),
                early_move_threshold_bps=float(args.early_move_threshold_bps),
                allow_zero_ctx=bool(args.allow_zero_ctx),
            )
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
        horizon_bars=int(args.horizon_bars),
        early_move_threshold_bps=float(args.early_move_threshold_bps),
        allow_zero_ctx=bool(args.allow_zero_ctx),
    )
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