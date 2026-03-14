"""
EXIT_TRANSFORMER_V0 – ML-only exit transformer (no rules/router/critic).

Provides:
- ExitTransformerV0 (PyTorch model)
- ExitTransformerDecider (inference)
- train_from_exits_jsonl (training utility; used by run_truth_e2e_sanity --train_exit_transformer_v0_from_last_go)
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
    EXIT_IO_V1_CTX36_FEATURE_NAMES_HASH,
    EXIT_IO_V1_CTX36_IO_VERSION,
    EXIT_IO_V1_CTX36_FEATURE_COUNT,
    compute_feature_names_hash,
    assert_exit_io_v1_ctx36_contract,
)

# -----------------------------------------------------------------------------
# IO CONTRACT
# -----------------------------------------------------------------------------
EXIT_IO_FEATURE_COUNT = EXIT_IO_V1_CTX36_FEATURE_COUNT
EXIT_IO_VERSION = EXIT_IO_V1_CTX36_IO_VERSION
EXIT_FEATURE_NAMES_HASH = EXIT_IO_V1_CTX36_FEATURE_NAMES_HASH

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
DEFAULT_WINDOW_LEN = 8
DEFAULT_D_MODEL = 64
DEFAULT_N_LAYERS = 2
DEFAULT_N_HEADS = 4
DEFAULT_THRESHOLD = 0.5

LAST_GO_FILENAME = "LAST_GO.txt"
TRUTH_E2E_REPORTS = "reports/truth_e2e_sanity"

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


# -----------------------------------------------------------------------------
# Deterministic EXIT LABELER (V0)
# -----------------------------------------------------------------------------
def _label_exit_deterministic_v0(
    *,
    idx: int,
    pnl_bps: np.ndarray,
    mfe_bps: np.ndarray,
    mae_bps: np.ndarray,
    dd_from_mfe_bps: np.ndarray,
    bars_held: np.ndarray,
    horizon: int = 12,
    min_hold_bars: int = 2,
    loss_bps: float = 40.0,
    recovery_bps: float = 20.0,
    mfe_arm_bps: float = 35.0,
    giveback_bps: float = 18.0,
) -> int:
    """
    Deterministic binary exit label.

    Rules:
    1) Respect min_hold_bars
    2) Cut-loser: future MAE <= -loss_bps before any recovery >= +recovery_bps
    3) Giveback: already reached MFE >= mfe_arm_bps AND dd_from_mfe_now >= giveback_bps
    """
    if bars_held[idx] < min_hold_bars:
        return 0

    # Giveback protector (uses info available "now")
    if mfe_bps[idx] >= mfe_arm_bps and dd_from_mfe_bps[idx] >= giveback_bps:
        return 1

    # Cut-loser lookahead (deterministic horizon)
    end = min(idx + horizon + 1, pnl_bps.size)
    future_pnl = pnl_bps[idx:end]

    hit_loss = np.any(future_pnl <= -loss_bps)
    hit_recovery = np.any(future_pnl >= recovery_bps)

    if hit_loss and not hit_recovery:
        return 1

    return 0


def _attach_labels_to_exit_records(
    records: List[Dict[str, Any]],
    *,
    lookahead_h_bars: int = 10,
    uplift_eps_bps: float = 20.0,
    min_hold_bars: int = 2,
    max_pos_per_trade: int = 10,
    hazard_band_bars: int = 10,
    max_positive_fraction_per_trade: float = 0.25,
    capture_ratio_threshold: float = 0.8,
    det_mfe_bps: float = 2.0,
    det_dd_from_mfe_bps: float = 80.0,
    det_giveback_ratio: float = 1.0,
    det_time_since_mfe_bars: float = 150.0,
    det_giveback_accel_min: float = 0.02,
    det_dd_accel_min: float = 3.0,
    det_stagnation_bars: int = 20,
    det_pnl_max_bps: float = 5.0,
    det_pnl_frac_of_mfe: float = 0.5,
    det_min_bars_held: int = 300,
    det_regret_ratio: float = 0.5,
    det_max_per_trade: int = 3,
) -> None:
    """
    Profit-capture labels based on peak proximity.
    For each trade bar i:
      - future_max_pnl = max(pnl_bps_now from i to trade end)
      - capture_ratio = pnl_bps_now / future_max_pnl
      - exit_label = 1 if capture_ratio >= capture_ratio_threshold (default 0.8)
    NaN/inf-safe: non-finite values or non-positive future_max_pnl -> label 0.
    Adds post-peak deterioration labels when a strong winner shows structural giveback.
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

    label_variant = os.environ.get("GX1_EXIT_LABEL_VARIANT", "EXIT_LABEL_DET_V1").strip().upper()
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
        "[EXIT_LABEL_VARIANT] variant=%s intraday_h=%s failfast=%s loss_bps=%.1f grace_bars=%d",
        label_variant,
        intraday_h,
        failfast,
        failfast_loss_bps,
        failfast_grace_bars,
    )

    # Group by trade_uid (fallback trade_id)
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

    n_pos = 0
    n_pos_capture = 0
    n_pos_deterioration = 0
    n_pos_both = 0
    n_trades_with_pos = 0
    bars_per_trade: List[int] = []
    pos_bars: List[float] = []
    pos_counts_per_trade: List[int] = []
    capture_ratios: List[float] = []
    det_counts_per_trade: List[int] = []

    for tid, recs in trades.items():
        if not recs:
            continue
        if any("should_exit" in r for r in recs):
            continue

        # Deterministic ordering: sort by bars_held then ts string
        def _sort_key(r):
            bars_val = (r.get("scalars") or {}).get("bars_held", r.get("bars_held", 0))
            ts_val = r.get("ts") or ""
            return (float(bars_val) if math.isfinite(float(bars_val)) else 0.0, str(ts_val))
        recs = sorted(recs, key=_sort_key)

        pnl = np.array([_get_scalar(r, "pnl_bps_now") for r in recs], dtype=np.float32)
        bars = np.array([_get_scalar(r, "bars_held") for r in recs], dtype=np.float32)
        mfe = np.array([_get_scalar(r, "mfe_bps") for r in recs], dtype=np.float32)
        dd_from_mfe = np.array([_get_scalar(r, "dd_from_mfe_bps") for r in recs], dtype=np.float32)
        distance_from_peak = []
        for r in recs:
            try:
                distance_from_peak.append(_get_scalar(r, "distance_from_peak_mfe_bps"))
            except Exception:
                distance_from_peak.append(_get_scalar(r, "dd_from_mfe_bps"))
        distance_from_peak = np.array(distance_from_peak, dtype=np.float32)
        giveback_ratio = np.array([_get_scalar(r, "giveback_ratio") for r in recs], dtype=np.float32)
        time_since_mfe = np.array([_get_scalar(r, "time_since_mfe_bars") for r in recs], dtype=np.float32)
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

        n = len(recs)
        if intraday_h is None:
            future_max = np.maximum.accumulate(pnl[::-1])[::-1]
        else:
            future_max = np.zeros(n, dtype=np.float32)
            for i in range(n):
                end = min(n, i + intraday_h + 1)
                future_max[i] = float(np.nanmax(pnl[i:end])) if end > i else float(pnl[i])
        # Track last time a new pnl high was made (stagnation proxy)
        max_so_far = -float("inf")
        last_high_idx = -1
        last_high_idx_arr = np.zeros(n, dtype=np.int32)
        for i in range(n):
            val = float(pnl[i])
            if math.isfinite(val) and val >= max_so_far:
                max_so_far = val
                last_high_idx = i
            last_high_idx_arr[i] = last_high_idx
        pos_count = 0
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
            capture_ratio = None
            is_capture = False
            if fmax > 0.0:
                capture_ratio = cur / fmax
                if math.isfinite(capture_ratio):
                    threshold = capture_ratio_threshold
                    if intraday_h is not None and float(bars[i]) >= intraday_h:
                        threshold = max(0.0, threshold - late_capture_threshold_drop)
                    is_capture = capture_ratio >= threshold

            late_exit = False
            if intraday_h is not None and float(bars[i]) >= intraday_h:
                if math.isfinite(cur) and cur >= late_exit_pnl_floor:
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

            # Post-peak deterioration detection (profit-protection / regret minimization)
            det = False
            try:
                if (
                    math.isfinite(float(mfe[i]))
                    and math.isfinite(float(distance_from_peak[i]))
                    and math.isfinite(float(giveback_ratio[i]))
                    and math.isfinite(float(time_since_mfe[i]))
                ):
                    stagnation = (i - int(last_high_idx_arr[i])) >= det_stagnation_bars
                    pnl_limit = min(det_pnl_max_bps, float(det_pnl_frac_of_mfe) * float(mfe[i]))
                    regret_ratio = float(distance_from_peak[i]) / max(1.0, float(mfe[i]))
                    det = (
                        float(bars[i]) >= det_min_bars_held
                        and float(mfe[i]) >= det_mfe_bps
                        and float(distance_from_peak[i]) >= det_dd_from_mfe_bps
                        and float(giveback_ratio[i]) >= det_giveback_ratio
                        and float(time_since_mfe[i]) >= det_time_since_mfe_bars
                        and regret_ratio >= det_regret_ratio
                        and stagnation
                        and float(cur) <= pnl_limit
                    )
                    if det and det_count >= det_max_per_trade:
                        det = False
            except Exception:
                det = False

            if is_capture or det or late_exit or boundary_exit or failfast_exit:
                recs[i]["should_exit"] = 1.0
                pos_count += 1
                n_pos += 1
                pos_bars.append(float(bars[i]))
                if is_capture and capture_ratio is not None:
                    n_pos_capture += 1
                    capture_ratios.append(float(capture_ratio))
                if det:
                    n_pos_deterioration += 1
                    det_count += 1
                if is_capture and det:
                    n_pos_both += 1
        if pos_count > 0:
            n_trades_with_pos += 1
        pos_counts_per_trade.append(int(pos_count))
        det_counts_per_trade.append(int(det_count))

        # write back to original list objects
        for src, new in zip(trades[tid], recs):
            src["should_exit"] = new["should_exit"]

    total_records = len(records)
    n_trades = len(trades)
    n_positive = sum(1 for r in records if r.get("should_exit", 0) == 1.0)
    exit_rate = n_positive / total_records if total_records else 0.0
    avg_pos_per_pos_trade = (n_positive / n_trades_with_pos) if n_trades_with_pos else 0.0
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
        "[EXIT_LABELER_PROOF] io=EXIT_IO_V1_CTX36 total_records=%d n_trades=%d n_trades_with_pos=%d n_pos=%d exit_rate=%.6f avg_pos_per_pos_trade=%.3f pct_trades_with_pos=%.3f bars_per_trade(min/med/max)=%d/%d/%d pos_bars(min/med/max)=%.2f/%.2f/%.2f pos_counts(p50/p90/p99)=%.2f/%.2f/%.2f capture_ratio_med=%.3f capture_ratio_threshold=%.2f det_pos=%d det_pos_rate=%.6f det_counts(p50/p90)=%.2f/%.2f min_hold_bars=%d",
        total_records,
        n_trades,
        n_trades_with_pos,
        n_positive,
        exit_rate,
        avg_pos_per_pos_trade,
        (n_trades_with_pos / n_trades) if n_trades else 0.0,
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
        n_pos_deterioration,
        (n_pos_deterioration / total_records) if total_records else 0.0,
        det_counts_med,
        det_counts_p90,
        min_hold_bars,
    )
    log.info(
        "[EXIT_LABELER_DET_PROOF] capture_pos=%d det_pos=%d both=%d det_thresholds={mfe_bps=%.1f dd_bps=%.1f giveback_ratio=%.2f time_since_mfe_bars=%.1f regret_ratio=%.2f giveback_accel_min=%.2f dd_accel_min=%.1f stagnation_bars=%d pnl_max=%.1f pnl_frac_of_mfe=%.2f min_bars=%d max_per_trade=%d}",
        n_pos_capture,
        n_pos_deterioration,
        n_pos_both,
        det_mfe_bps,
        det_dd_from_mfe_bps,
        det_giveback_ratio,
        det_time_since_mfe_bars,
        det_regret_ratio,
        det_giveback_accel_min,
        det_dd_accel_min,
        det_stagnation_bars,
        det_pnl_max_bps,
        det_pnl_frac_of_mfe,
        det_min_bars_held,
        det_max_per_trade,
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
        """Small transformer encoder: input (B, T, D) -> EXIT prob (B,)."""

        def __init__(
            self,
            input_dim: int = EXIT_IO_FEATURE_COUNT,
            window_len: int = DEFAULT_WINDOW_LEN,
            d_model: int = 64,
            n_heads: int = DEFAULT_N_HEADS,
            n_layers: int = DEFAULT_N_LAYERS,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.window_len = window_len
            self.d_model = d_model
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
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.sigmoid(self.forward_logits(x))

        def forward_logits(self, x: "torch.Tensor") -> "torch.Tensor":
            h = self.proj(x)
            h = self.transformer(h)
            return self.head(h[:, -1]).squeeze(-1)

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
    feature_names_hash: str = EXIT_FEATURE_NAMES_HASH,
    input_dim: Optional[int] = None,
) -> Tuple[Path, Path, str]:
    if not TORCH_AVAILABLE or model is None:
        raise RuntimeError("PyTorch and model required to save")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "exit_transformer_v0.pt"
    torch.save(model.state_dict(), model_path)
    with open(model_path, "rb") as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()

    cfg = {
        "window_len": int(window_len),
        "input_dim": int(input_dim if input_dim is not None else EXIT_IO_FEATURE_COUNT),
        "d_model": int(d_model),
        "n_layers": int(n_layers),
        "n_heads": int(DEFAULT_N_HEADS),
        "dropout": 0.1,
        "exit_ml_io_version": EXIT_IO_VERSION,
        "feature_names_hash": feature_names_hash,
    }
    cfg_path = out_dir / "exit_transformer_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return model_path, cfg_path, model_sha


class ExitTransformerDecider:
    """Lightweight inference wrapper."""

    def __init__(self, model: Any, config: Dict[str, Any], model_sha: Optional[str] = None) -> None:
        self.model = model
        self.config = config
        self.window_len = int(config.get("window_len", DEFAULT_WINDOW_LEN))
        self.input_dim = int(config.get("input_dim", EXIT_IO_FEATURE_COUNT))
        self.model_sha = model_sha
        self.feature_names_hash = config.get("feature_names_hash")
        try:
            cfg_temp = float(config.get("exit_logit_temperature", 1.0))
        except Exception:
            cfg_temp = 1.0
        env_temp = os.environ.get("GX1_EXIT_LOGIT_TEMPERATURE", "").strip()
        try:
            env_temp_val = float(env_temp) if env_temp else None
        except Exception:
            env_temp_val = None
        self.logit_temperature = float(env_temp_val if env_temp_val is not None else cfg_temp)
        if not math.isfinite(self.logit_temperature) or self.logit_temperature <= 0.0:
            self.logit_temperature = 1.0

    def predict(self, window: Any) -> Tuple[float, Optional[float], Optional[float]]:
        arr = np.asarray(window, dtype=np.float32)
        validate_window_v3(arr, self.window_len, self.input_dim, context="inference")
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for inference")
        with torch.no_grad():
            x = torch.from_numpy(arr).unsqueeze(0)
            if hasattr(self.model, "forward_logits"):
                logit = float(self.model.forward_logits(x).detach().cpu().numpy().reshape(-1)[0])
                temp = self.logit_temperature
                if not math.isfinite(temp) or temp <= 0.0:
                    temp = 1.0
                logit_scaled = logit / temp
                try:
                    prob = 1.0 / (1.0 + math.exp(-logit_scaled))
                except OverflowError:
                    prob = 0.0 if logit_scaled < 0 else 1.0
                return float(prob), logit, logit_scaled
            prob = float(self.model(x).detach().cpu().numpy().reshape(-1)[0])
        return prob, None, None


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
    input_dim = int(cfg.get("input_dim", EXIT_IO_FEATURE_COUNT))
    feat_hash = cfg.get("feature_names_hash", "")
    io_ver = cfg.get("exit_ml_io_version", "")
    if io_ver and io_ver != EXIT_IO_VERSION:
        raise RuntimeError(f"[EXIT_IO_CONTRACT_VIOLATION] exit_ml_io_version mismatch: {io_ver} != {EXIT_IO_VERSION}")
    if feat_hash and feat_hash != EXIT_FEATURE_NAMES_HASH:
        if os.getenv("GX1_EXIT_HASH_GUARD_BYPASS") == "1":
            log.warning(
                "[EXIT_IO_CONTRACT_BYPASS] enabled=1 feature_names_hash=%s contract_feature_hash=%s",
                feat_hash,
                EXIT_FEATURE_NAMES_HASH,
            )
        else:
            raise RuntimeError(
                f"[EXIT_IO_CONTRACT_VIOLATION] feature_names_hash mismatch: {feat_hash} != {EXIT_FEATURE_NAMES_HASH}"
            )

    model = ExitTransformerV0(
        input_dim=input_dim,
        window_len=window_len,
        d_model=int(cfg.get("d_model", DEFAULT_D_MODEL)),
        n_heads=int(cfg.get("n_heads", DEFAULT_N_HEADS)),
        n_layers=int(cfg.get("n_layers", DEFAULT_N_LAYERS)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    state = torch.load(model_file, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with open(model_file, "rb") as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()

    return ExitTransformerDecider(model=model, config=cfg, model_sha=model_sha)


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
        io_ver = cfg.get("exit_ml_io_version")
        if io_ver and io_ver != EXIT_IO_VERSION:
            failures.append(f"io_version_mismatch:{io_ver}!={EXIT_IO_VERSION}")
        feat_hash = cfg.get("feature_names_hash")
        if feat_hash and feat_hash != EXIT_FEATURE_NAMES_HASH:
            failures.append(f"feature_hash_mismatch:{feat_hash}!={EXIT_FEATURE_NAMES_HASH}")
        input_dim = int(cfg.get("input_dim", -1))
        if input_dim != EXIT_IO_FEATURE_COUNT:
            failures.append(f"input_dim_mismatch:{input_dim}!={EXIT_IO_FEATURE_COUNT}")
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


def _build_windows_from_exits(
    records: List[Dict[str, Any]],
    window_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    feats: List[np.ndarray] = []
    labels: List[float] = []
    ts_list: List[pd.Timestamp] = []

    for r in records:
        io = r.get("io_features")
        if io is None:
            io = (r.get("io") or {}).get("io_features")
        label = r.get("should_exit")
        if io is None or label is None:
            continue
        arr = np.asarray(io, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != EXIT_IO_FEATURE_COUNT:
            raise RuntimeError("EXIT_IO_DIM_MISMATCH")
        if arr.shape[0] < window_len:
            continue
        ts_raw = r.get("ts") or r.get("event_ts") or r.get("time")
        try:
            ts_val = pd.Timestamp(ts_raw) if ts_raw else pd.NaT
        except Exception:
            ts_val = pd.NaT
        for s in range(0, arr.shape[0] - window_len + 1, stride):
            feats.append(arr[s : s + window_len])
            labels.append(float(label))
            ts_list.append(ts_val)

    if not feats:
        return (
            np.zeros((0, window_len, EXIT_IO_FEATURE_COUNT), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
        )

    return np.stack(feats), np.asarray(labels, dtype=np.float32), ts_list


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


def _audit_score_compression(
    model: "ExitTransformerV0",
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> Dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        logits_tr = model.forward_logits(torch.from_numpy(X_tr).to(device)).detach().cpu().numpy()
        probs_tr = 1.0 / (1.0 + np.exp(-logits_tr))
        logits_va = (
            model.forward_logits(torch.from_numpy(X_va).to(device)).detach().cpu().numpy()
            if X_va.size
            else np.zeros(0)
        )
        probs_va = 1.0 / (1.0 + np.exp(-logits_va)) if X_va.size else np.zeros(0)

    label_rate = float(np.mean(np.concatenate([y_tr, y_va])) if (y_tr.size or y_va.size) else math.nan)
    train_pos_rate = float(np.mean(y_tr)) if y_tr.size else math.nan
    val_pos_rate = float(np.mean(y_va)) if y_va.size else math.nan

    prob_stats = _score_stats(probs_va if probs_va.size else probs_tr)
    logit_stats = _score_stats(logits_va if logits_va.size else logits_tr)

    suspected_root_cause = "unknown"
    if math.isfinite(label_rate) and label_rate < 0.05 and prob_stats["std"] < 0.002:
        suspected_root_cause = "class_imbalance_unweighted_bce"

    return {
        "label_rate": label_rate,
        "train_pos_rate": train_pos_rate,
        "val_pos_rate": val_pos_rate,
        "logit_mean": logit_stats["mean"],
        "logit_std": logit_stats["std"],
        "prob_mean": prob_stats["mean"],
        "prob_std": prob_stats["std"],
        "suspected_root_cause": suspected_root_cause,
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
    X, y, _ = _build_windows_from_exits(records, window_len, stride)
    if X.size == 0:
        raise RuntimeError("No audit windows produced")

    decider = load_exit_transformer_decider(model_path)
    model = decider.model

    n = X.shape[0]
    split = max(1, n * 9 // 10)
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    audit = _audit_score_compression(model, X_tr, y_tr, X_va, y_va)
    if out_dir is None:
        out_dir = model_path if model_path.is_dir() else model_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "SCORE_COMPRESSION_AUDIT.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    log.info(
        "[EXIT_SCORE_COMPRESSION_AUDIT] label_rate=%.6f logit_mean=%.6f logit_std=%.6f prob_mean=%.6f prob_std=%.6f train_pos_rate=%.6f val_pos_rate=%.6f suspected_root_cause=%s",
        audit["label_rate"],
        audit["logit_mean"],
        audit["logit_std"],
        audit["prob_mean"],
        audit["prob_std"],
        audit["train_pos_rate"],
        audit["val_pos_rate"],
        audit["suspected_root_cause"],
    )
    return audit


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_from_exits_jsonl(
    exits_jsonl_path: Path,
    out_dir: Optional[Path] = None,
    window_len: int = DEFAULT_WINDOW_LEN,
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

    exits_jsonl_path = Path(exits_jsonl_path)
    records = _load_exits_jsonl(exits_jsonl_path)

    # Attach deterministic labels if missing
    _attach_labels_to_exit_records(records)

    _make_deterministic(seed)
    X, y, ts_list = _build_windows_from_exits(records, window_len, stride)
    if X.size == 0:
        raise RuntimeError("No training windows produced")

    dataset_sha = hashlib.sha256(exits_jsonl_path.read_bytes()).hexdigest()
    n = X.shape[0]
    if n < 3:
        raise RuntimeError("[EXIT_SPLIT_FAIL] insufficient windows for train/val/test split")
    pos_idx = np.where(y > 0)[0]
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
        y = y[:n]
        ts_list = ts_list[:n]
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
    while val_end > train_end + 1 and _pos_count(y[val_end:]) == 0:
        val_end -= 1
    if _pos_count(y[val_end:]) == 0:
        raise RuntimeError("[EXIT_SPLIT_FAIL] test split has 0 positive labels")

    # Ensure val has positives (expand val backward if needed)
    while train_end > 1 and _pos_count(y[train_end:val_end]) == 0:
        train_end -= 1
    if _pos_count(y[train_end:val_end]) == 0:
        raise RuntimeError("[EXIT_SPLIT_FAIL] val split has 0 positive labels")

    X_tr, y_tr = X[:train_end], y[:train_end]
    X_va, y_va = X[train_end:val_end], y[train_end:val_end]
    X_te, y_te = X[val_end:], y[val_end:]

    def _log_split(name: str, y_split: np.ndarray, ts_split: List[pd.Timestamp]) -> None:
        total = int(y_split.size)
        pos = _pos_count(y_split)
        neg = total - pos
        pos_rate = float(pos / total) if total else math.nan
        t_min, t_max = _time_range(ts_split)
        log.info(
            "[EXIT_TRAIN_SPLIT] split=%s total=%d pos=%d neg=%d pos_rate=%.6f time_range=%s..%s",
            name,
            total,
            pos,
            neg,
            pos_rate,
            t_min,
            t_max,
        )

    _log_split("train", y_tr, ts_list[:train_end])
    _log_split("val", y_va, ts_list[train_end:val_end])
    _log_split("test", y_te, ts_list[val_end:])

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
        input_dim=EXIT_IO_FEATURE_COUNT,
        window_len=window_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)
    log.info(
        "[EXIT_TRAIN_DEVICE] device=%s input_dim=%d window_len=%d feature_hash=%s feature_names=%s",
        str(device),
        int(EXIT_IO_FEATURE_COUNT),
        int(window_len),
        EXIT_FEATURE_NAMES_HASH,
        list(EXIT_IO_V1_CTX36_FEATURES),
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_pos = float(np.sum(y_tr))
    n_neg = float(max(0.0, y_tr.size - n_pos))
    pos_weight = torch.tensor(n_neg / max(1.0, n_pos), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

    for epoch in range(epochs):
        model.train()
        for i in range(0, X_tr.shape[0], batch_size):
            bx = torch.from_numpy(X_tr[i : i + batch_size]).to(device)
            by = torch.from_numpy(y_tr[i : i + batch_size]).to(device)
            opt.zero_grad()
            logits = model.forward_logits(bx)
            loss = loss_fn(logits, by)
            loss.backward()
            opt.step()

        if early_stop and X_va.size:
            model.eval()
            with torch.no_grad():
                vloss_epoch = float(
                    np.mean(
                        (model(torch.from_numpy(X_va).to(device)).detach().cpu().numpy() - y_va) ** 2
                    )
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
            vloss = (
                float(
                    np.mean(
                        (model(torch.from_numpy(X_va).to(device)).detach().cpu().numpy() - y_va) ** 2
                    )
                )
                if X_va.size
                else math.nan
            )

    audit = _audit_score_compression(model, X_tr, y_tr, X_va, y_va)

    base = Path(gx1_data or os.environ["GX1_DATA"])
    if out_dir is None:
        out_dir = base / "models" / "exit_transformer_v0" / dataset_sha
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path, cfg_path, sha = save_exit_transformer_artifacts(
        model,
        out_dir,
        window_len,
        d_model,
        n_layers,
        feature_names_hash=EXIT_FEATURE_NAMES_HASH,
        input_dim=EXIT_IO_FEATURE_COUNT,
    )

    train_report_path = out_dir / "TRAIN_REPORT.json"
    report = {
        "dataset": str(exits_jsonl_path),
        "dataset_sha256": dataset_sha,
        "n_samples": int(X.shape[0]),
        "exit_rate": float(np.mean(y)),
        "val_loss": vloss,
        "score_compression_audit": audit,
        "model_path": str(model_path),
        "config_path": str(cfg_path),
        "model_sha256": sha,
    }

    with open(train_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(out_dir / "SCORE_COMPRESSION_AUDIT.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    log.info(
        "[EXIT_SCORE_COMPRESSION_AUDIT] label_rate=%.6f logit_mean=%.6f logit_std=%.6f prob_mean=%.6f prob_std=%.6f train_pos_rate=%.6f val_pos_rate=%.6f suspected_root_cause=%s",
        audit["label_rate"],
        audit["logit_mean"],
        audit["logit_std"],
        audit["prob_mean"],
        audit["prob_std"],
        audit["train_pos_rate"],
        audit["val_pos_rate"],
        audit["suspected_root_cause"],
    )
    log.info(
        "[EXIT_TRAINING_PROOF] input_dim=%d feature_hash=%s train_pos_rate=%.6f val_pos_rate=%.6f logit_mean=%.6f logit_std=%.6f prob_mean=%.6f prob_std=%.6f",
        int(EXIT_IO_FEATURE_COUNT),
        EXIT_FEATURE_NAMES_HASH,
        audit["train_pos_rate"],
        audit["val_pos_rate"],
        audit["logit_mean"],
        audit["logit_std"],
        audit["prob_mean"],
        audit["prob_std"],
    )
    return dict(report, train_report_path=train_report_path)

    log.info("[EXIT_TRAIN_DONE] out_dir=%s val_loss=%s", out_dir, vloss)
    return report
