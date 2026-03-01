"""
EXIT_TRANSFORMER_V0 – ML-only exit transformer (no rules/router/critic).

Provides:
- ExitTransformerV0 (PyTorch model)
- ExitTransformerDecider (inference)
- train_from_exits_jsonl (training utility; used by run_truth_e2e_sanity --train_exit_transformer_v0_from_last_go)
- save/load artifacts (model/config/sha)

Hard gates:
- feature_dim must be 19 (EXIT_IO_FEATURE_COUNT)
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
from gx1.exits.contracts.exit_io_v0_ctx19 import (
    EXIT_IO_V0_CTX19_FEATURES,
    EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH,
    EXIT_IO_V0_CTX19_IO_VERSION,
    EXIT_IO_V0_CTX19_FEATURE_COUNT,
    compute_feature_names_hash,
    assert_exit_io_v0_ctx19_contract,
)

# -----------------------------------------------------------------------------
# IO CONTRACT
# -----------------------------------------------------------------------------
EXIT_IO_FEATURE_COUNT = EXIT_IO_V0_CTX19_FEATURE_COUNT
EXIT_IO_VERSION = EXIT_IO_V0_CTX19_IO_VERSION
EXIT_FEATURE_NAMES_HASH = EXIT_IO_V0_CTX19_FEATURE_NAMES_HASH

log = logging.getLogger(__name__)
_exit_features_proof_logged = False
assert_exit_io_v0_ctx19_contract()

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
    lookahead_h_bars: int = 6,
    uplift_eps_bps: float = 8.0,
    min_hold_bars: int = 2,
    max_pos_per_trade: int = 10,
    hazard_band_bars: int = 8,
    max_positive_fraction_per_trade: float = 0.25,
) -> None:
    """
    Lookahead-based labels with per-bar condition and capped hazard band per trade.
    For each trade:
      - Candidate if bars_held>=min_hold AND future uplift (max pnl in horizon minus current pnl) <= uplift_eps_bps.
      - Keep up to max_pos_per_trade candidates with smallest uplift (tie: earliest index).
      - Mark a hazard band of hazard_band_bars following each chosen event (inclusive).
      - Cap positives to max_positive_fraction_per_trade of trade length (earliest bands first).
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

    # Group by trade_uid (fallback trade_id)
    trades: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        if "should_exit" in rec:
            continue
        tid = rec.get("trade_uid") or rec.get("trade_id")
        if not tid:
            continue
        trades.setdefault(tid, []).append(rec)

    n_pos = 0
    n_trades_with_pos = 0
    bars_per_trade: List[int] = []
    pos_bars: List[float] = []
    pos_uplift: List[float] = []
    pos_counts_per_trade: List[int] = []

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

        bars_per_trade.append(len(recs))
        for r in recs:
            r["should_exit"] = 0.0

        # Collect candidates with uplift condition
        candidates = []
        n = len(recs)
        for i in range(n):
            if bars[i] < min_hold_bars:
                continue
            j_end = min(i + lookahead_h_bars, n - 1)
            best_future = float(np.max(pnl[i : j_end + 1]))
            uplift = best_future - float(pnl[i])
            if uplift <= uplift_eps_bps:
                candidates.append((uplift, i))

        if candidates:
            # Sort by smallest uplift, then earliest index
            candidates.sort(key=lambda t: (t[0], t[1]))
            chosen = candidates[:max_pos_per_trade]
            # Build hazard band mask
            mask = np.zeros(len(recs), dtype=np.float32)
            cap_pos = int(max(1, math.floor(max_positive_fraction_per_trade * len(recs))))
            for _, idx in chosen:
                band_end = min(idx + hazard_band_bars, len(recs))
                for j in range(idx, band_end):
                    if mask.sum() >= cap_pos:
                        break
                    mask[j] = 1.0
                if mask.sum() >= cap_pos:
                    break
            for i, r in enumerate(recs):
                if mask[i] >= 1.0:
                    r["should_exit"] = 1.0
                    n_pos += 1
                    pos_bars.append(float(bars[i]))
            if mask.sum() > 0:
                n_trades_with_pos += 1
            pos_counts_per_trade.append(int(mask.sum()))

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
    pos_uplift_med = float(np.median(pos_uplift)) if pos_uplift else 0.0
    pos_counts_med = float(np.median(pos_counts_per_trade)) if pos_counts_per_trade else 0.0
    pos_counts_p90 = float(np.quantile(pos_counts_per_trade, 0.9)) if pos_counts_per_trade else 0.0
    pos_counts_p99 = float(np.quantile(pos_counts_per_trade, 0.99)) if pos_counts_per_trade else 0.0

    log.info(
        "[EXIT_LABELER_PROOF] io=EXIT_IO_V0_CTX19 total_records=%d n_trades=%d n_trades_with_pos=%d n_pos=%d exit_rate=%.6f avg_pos_per_pos_trade=%.3f pct_trades_with_pos=%.3f bars_per_trade(min/med/max)=%d/%d/%d pos_bars(min/med/max)=%.2f/%.2f/%.2f pos_counts(p50/p90/p99)=%.2f/%.2f/%.2f pos_uplift_med=%.2f lookahead_h_bars=%d uplift_eps_bps=%.2f min_hold_bars=%d max_pos_per_trade=%d hazard_band_bars=%d max_pos_frac=%.3f",
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
        pos_uplift_med,
        lookahead_h_bars,
        uplift_eps_bps,
        min_hold_bars,
        max_pos_per_trade,
        hazard_band_bars,
        max_positive_fraction_per_trade,
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
            h = self.proj(x)
            h = self.transformer(h)
            return torch.sigmoid(self.head(h[:, -1]).squeeze(-1))

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

    def predict(self, window: Any) -> Tuple[float, Optional[float], Optional[float]]:
        arr = np.asarray(window, dtype=np.float32)
        validate_window_v3(arr, self.window_len, self.input_dim, context="inference")
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for inference")
        with torch.no_grad():
            x = torch.from_numpy(arr).unsqueeze(0)
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
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[float] = []

    for r in records:
        io = r.get("io_features")
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
        for s in range(0, arr.shape[0] - window_len + 1, stride):
            feats.append(arr[s : s + window_len])
            labels.append(float(label))

    if not feats:
        return (
            np.zeros((0, window_len, EXIT_IO_FEATURE_COUNT), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )

    return np.stack(feats), np.asarray(labels, dtype=np.float32)


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
    X, y = _build_windows_from_exits(records, window_len, stride)
    if X.size == 0:
        raise RuntimeError("No training windows produced")

    dataset_sha = hashlib.sha256(exits_jsonl_path.read_bytes()).hexdigest()
    n = X.shape[0]
    split = max(1, n * val_fold // (val_fold + 1))

    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    device = torch.device("cpu")
    model = ExitTransformerV0(
        input_dim=EXIT_IO_FEATURE_COUNT,
        window_len=window_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for _ in range(epochs):
        model.train()
        for i in range(0, X_tr.shape[0], batch_size):
            bx = torch.from_numpy(X_tr[i : i + batch_size]).to(device)
            by = torch.from_numpy(y_tr[i : i + batch_size]).to(device)
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        vloss = (
            float(np.mean((model(torch.from_numpy(X_va)).numpy() - y_va) ** 2))
            if X_va.size
            else math.nan
        )

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

    report = {
        "dataset": str(exits_jsonl_path),
        "dataset_sha256": dataset_sha,
        "n_samples": int(X.shape[0]),
        "exit_rate": float(np.mean(y)),
        "val_loss": vloss,
        "model_path": str(model_path),
        "config_path": str(cfg_path),
        "model_sha256": sha,
    }

    with open(out_dir / "TRAIN_REPORT.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info("[EXIT_TRAIN_DONE] out_dir=%s val_loss=%s", out_dir, vloss)
    return report