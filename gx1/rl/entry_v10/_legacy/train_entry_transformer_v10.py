#!/usr/bin/env python3
"""
Train ENTRY_V10_CTX Transformer (ONE UNIVERSE: 6/6 + signal bridge 7/7).

- STRICT 6/6 context only (ctx_cont_dim=6, ctx_cat_dim=6). No 2/4/6 or 5/6.
- Signal bridge: seq_x [B,T,7], snap_x [B,7] from gx1.contracts.signal_bridge_v1.
- Uses EntryV10CtxDataset and EntryV10CtxHybridTransformer.
- Bundles to: $GX1_DATA/models/models/entry_v10_ctx/TRANSFORMER_ENTRY_V10_CTX__CTX6CAT6_<ts>/
  with model_state_dict.pt, MASTER_TRANSFORMER_LOCK.json, bundle_metadata.json + self-check.

NO FALLBACKS. NO LEGACY. TRUTH/SMOKE hard-fail on any other dims.
"""

raise RuntimeError(
    "RL_DISABLED: Reinforcement Learning is explicitly disabled. "
    "Focus is ENTRY/EXIT transformer + live trade plumbing only."
)

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from gx1.contracts.signal_bridge_v1 import (
    CONTRACT_SHA256 as SIGNAL_BRIDGE_CONTRACT_SHA256,
    SEQ_SIGNAL_DIM,
    SNAP_SIGNAL_DIM,
)

from gx1.rl.entry_v10.dataset_v10 import EntryV10Dataset
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
    CTX_CAT_DIM as MODEL_CTX_CAT_DIM,
    CTX_CONT_DIM as MODEL_CTX_CONT_DIM,
)


# -----------------------------------------------------------------------------
# Multiprocessing start method safety (macOS compatibility + predictable workers)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def _resolve_gx1_data(gx1_data_cli: str = "") -> Path:
    base = Path(gx1_data_cli or os.environ.get("GX1_DATA", "")).expanduser().resolve()
    if not base.is_dir():
        raise RuntimeError(
            "GX1_DATA must be set (or passed via --gx1-data) and point to an existing directory. "
            f"Got: {base}"
        )
    return base


def _set_strict_deterministic_mode(seed: int, device: torch.device) -> None:
    """
    Best-effort determinism for reproducibility (training speed will drop).
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        log.warning(f"[DETERMINISM] torch.use_deterministic_algorithms(True) failed: {e}")


def _init_weights_safe(model: nn.Module) -> None:
    """
    Conservative init to reduce risk of exploding activations early.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


def _assert_finite(name: str, t: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        raise RuntimeError(f"NON_FINITE_TENSOR: {name} has {bad} NaN/Inf values; shape={tuple(t.shape)}")


def _torch_load_weights_only(path: Path) -> Dict[str, torch.Tensor]:
    """
    torch.load(..., weights_only=True) exists in newer torch. Fall back if not supported.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


# -----------------------------------------------------------------------------
# Core training loops (CTX)
# -----------------------------------------------------------------------------
def train_epoch_ctx(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        seq_x = batch["seq_x"].to(device)
        snap_x = batch["snap_x"].to(device)
        ctx_cat = batch["ctx_cat"].to(device)
        ctx_cont = batch["ctx_cont"].to(device)
        y = batch.get("y_direction", batch.get("y")).to(device)

        _assert_finite("seq_x", seq_x)
        _assert_finite("snap_x", snap_x)
        _assert_finite("ctx_cont", ctx_cont)
        _assert_finite("y", y)

        optimizer.zero_grad(set_to_none=True)
        out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)

        direction_logit = out["direction_logit"].squeeze(-1)
        _assert_finite("direction_logit", direction_logit)

        loss = criterion(direction_logit, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"NON_FINITE_LOSS: loss={loss.item()}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = int(y.shape[0])
        total_loss += float(loss) * bs
        n_samples += bs

    return {"loss": (total_loss / max(1, n_samples))}


def validate_ctx(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_samples = 0

    preds: list[float] = []
    targets: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            ctx_cat = batch["ctx_cat"].to(device)
            ctx_cont = batch["ctx_cont"].to(device)
            y = batch.get("y_direction", batch.get("y")).to(device)

            out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont)
            direction_logit = out["direction_logit"].squeeze(-1)

            loss = criterion(direction_logit, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NON_FINITE_VAL_LOSS: loss={loss.item()}")

            bs = int(y.shape[0])
            total_loss += float(loss) * bs
            n_samples += bs

            probs = torch.sigmoid(direction_logit)
            _assert_finite("val_probs", probs)

            probs_np = probs.detach().cpu().numpy()
            if probs_np.size > 1 and float(np.std(probs_np)) < 1e-6:
                raise RuntimeError(
                    f"CRITICAL: validation predictions nearly constant (std={np.std(probs_np):.2e}). NO FALLBACK."
                )

            preds.extend([float(x) for x in probs_np.reshape(-1)])
            targets.extend([float(x) for x in y.detach().cpu().numpy().reshape(-1)])

    avg_loss = total_loss / max(1, n_samples)
    preds_np = np.asarray(preds, dtype=np.float64)
    targets_np = np.asarray(targets, dtype=np.float64)

    try:
        auc = float(roc_auc_score(targets_np, preds_np))
    except Exception:
        auc = 0.0

    pred_bin = (preds_np > 0.5).astype(np.int64)
    acc = float(accuracy_score(targets_np.astype(np.int64), pred_bin))

    return {"loss": float(avg_loss), "auc": auc, "accuracy": acc}


# -----------------------------------------------------------------------------
# ENTRY_V10_CTX training (STRICT 6/6)
# -----------------------------------------------------------------------------
def train_entry_v10_ctx_transformer(
    train_parquet: Path,
    val_parquet: Path,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seq_len: int,
    early_stopping_patience: int,
    gx1_data: str,
    seed: int,
    strict_deterministic: bool,
    num_workers: int,
) -> Path:
    """
    Train ENTRY_V10_CTX STRICT 6/6 and create a versioned bundle under GX1_DATA.

    Returns:
        Path to bundle directory.
    """
    # Hard lock to model contract (STRICT)
    ctx_cont_dim = int(MODEL_CTX_CONT_DIM)
    ctx_cat_dim = int(MODEL_CTX_CAT_DIM)

    base = _resolve_gx1_data(gx1_data)

    train_parquet = Path(train_parquet).expanduser().resolve()
    val_parquet = Path(val_parquet).expanduser().resolve()
    _require_file(train_parquet, "train_parquet")
    _require_file(val_parquet, "val_parquet")

    dev = torch.device(device)
    log.info(f"[CTX] device={dev} seq_len={seq_len} ctx_cont_dim={ctx_cont_dim} ctx_cat_dim={ctx_cat_dim}")

    if strict_deterministic:
        log.info("[CTX] strict_deterministic=1 (repro mode)")
        _set_strict_deterministic_mode(seed=seed, device=dev)
        num_workers = 0

    # Dataset: must match STRICT dims. If your dataset supports variable dims, we pin here.
    train_ds = EntryV10Dataset(
        train_parquet,
        seq_len=seq_len,
        device=str(dev),
    )
    val_ds = EntryV10Dataset(
        val_parquet,
        seq_len=seq_len,
        device=str(dev),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(int(num_workers) > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(int(num_workers) > 0),
    )

    # Hard assert first batch ctx dims
    first_batch = next(iter(train_loader))
    if first_batch["seq_x"].shape[0] == 0:
        raise ValueError("CTX_DIM_MISMATCH: first batch has 0 samples (dataset/batch)")
    got_cont = int(first_batch["ctx_cont"].shape[1])
    got_cat = int(first_batch["ctx_cat"].shape[1])
    if got_cont != 6 or got_cat != 6:
        raise ValueError(
            f"CTX_DIM_MISMATCH: first batch ctx_cont len={got_cont} ctx_cat len={got_cat} expected canonical 6/6"
        )
    if got_cont != ctx_cont_dim or got_cat != ctx_cat_dim:
        raise ValueError(
            f"CTX_DIM_MISMATCH: first batch ctx_cont len={got_cont} expected={ctx_cont_dim} "
            f"ctx_cat len={got_cat} expected={ctx_cat_dim} (dataset/batch)"
        )

    # Model config must match STRICT model signature
    model_config: Dict[str, Any] = {
        "max_seq_len": int(seq_len),
        "num_layers": 3,
        "d_model": 128,
        "n_heads": 4,
        "dim_feedforward": 512,
        "dropout": 0.05,
    }

    model = EntryV10CtxHybridTransformer(**model_config).to(dev)
    _init_weights_safe(model)

    criterion = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    epochs_wo = 0

    for epoch in range(int(epochs)):
        tr = train_epoch_ctx(model, train_loader, criterion, opt, dev)
        va = validate_ctx(model, val_loader, criterion, dev)
        scheduler.step(va["loss"])

        log.info(
            f"[CTX] epoch {epoch+1}/{epochs} "
            f"train_loss={tr['loss']:.6f} val_loss={va['loss']:.6f} auc={va['auc']:.4f} acc={va['accuracy']:.4f}"
        )

        if va["loss"] < best_val_loss:
            best_val_loss = float(va["loss"])
            best_epoch = int(epoch + 1)
            epochs_wo = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_wo += 1

        if epochs_wo >= int(early_stopping_patience):
            log.info(f"[CTX] early stop: patience={early_stopping_patience} best_epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("[CTX] TRAIN_FAIL: no best_state captured")

    ts = _utc_ts_compact()
    out_dir = (
        base
        / "models"
        / "models"
        / "entry_v10_ctx"
        / f"TRANSFORMER_ENTRY_V10_CTX__IOV1_CTX6CAT6_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model_state_dict.pt"
    torch.save(best_state, model_path)
    model_sha256 = _sha256_file(model_path)
    created_utc = _iso_utc_now()

    lock = {
        "version": "entry_v10_ctx_signal_only_lock_v1",
        "created_at_utc": created_utc,
        "signal_bridge_contract_sha256": SIGNAL_BRIDGE_CONTRACT_SHA256,
        "seq_input_dim": int(SEQ_SIGNAL_DIM),
        "snap_input_dim": int(SNAP_SIGNAL_DIM),
        "seq_len": int(seq_len),
        "ctx_cont_dim": int(ctx_cont_dim),
        "ctx_cat_dim": int(ctx_cat_dim),
        "model_path_relative": "model_state_dict.pt",
        "model_sha256": model_sha256,
    }
    (out_dir / "MASTER_TRANSFORMER_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8"
    )

    bundle_metadata = {
        "created_at_utc": created_utc,
        "mode": "entry_v10_ctx_strict_6_6",
        "model_config": model_config,
        "seq_input_dim": int(SEQ_SIGNAL_DIM),
        "snap_input_dim": int(SNAP_SIGNAL_DIM),
        "seq_len": int(seq_len),
        "ctx_cont_dim": int(ctx_cont_dim),
        "ctx_cat_dim": int(ctx_cat_dim),
        "supports_context_features": True,
        "signal_bridge_contract_sha256": SIGNAL_BRIDGE_CONTRACT_SHA256,
        "train_data_path": str(train_parquet),
        "val_data_path": str(val_parquet),
        "train_data_sha256": _sha256_file(train_parquet),
        "val_data_sha256": _sha256_file(val_parquet),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "seed": int(seed),
        "strict_deterministic": bool(strict_deterministic),
        "num_workers": int(num_workers),
    }
    (out_dir / "bundle_metadata.json").write_text(
        json.dumps(bundle_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Self-check: rebuild + strict load + dummy forward
    log.info("[CTX] self-check: rebuild + strict load + dummy forward")
    model2 = EntryV10CtxHybridTransformer(**model_config)
    state = _torch_load_weights_only(model_path)
    model2.load_state_dict(state, strict=True)
    model2.eval()
    with torch.no_grad():
        B = 2
        dummy_seq = torch.zeros(B, int(seq_len), int(SEQ_SIGNAL_DIM), dtype=torch.float32)
        dummy_snap = torch.zeros(B, int(SNAP_SIGNAL_DIM), dtype=torch.float32)
        dummy_cat = torch.zeros(B, int(ctx_cat_dim), dtype=torch.long)
        dummy_cont = torch.zeros(B, int(ctx_cont_dim), dtype=torch.float32)
        _ = model2(dummy_seq, dummy_snap, ctx_cat=dummy_cat, ctx_cont=dummy_cont)

    log.info(f"[CTX] bundle OK: {out_dir}")
    return out_dir


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ENTRY_V10_CTX (6/6 + signal bridge 7/7). ONE UNIVERSE only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--train-parquet", type=Path, required=True, help="Training dataset parquet")
    parser.add_argument("--val-parquet", type=Path, required=True, help="Validation dataset parquet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument(
        "--strict-deterministic",
        action="store_true",
        help="Repro mode: deterministic algorithms (best-effort), num_workers forced to 0.",
    )
    parser.add_argument("--gx1-data", type=str, default="", help="Override GX1_DATA for bundling")

    args = parser.parse_args()

    dev = torch.device(args.device)

    if args.strict_deterministic:
        _set_strict_deterministic_mode(seed=int(args.seed), device=dev)

    bundle_dir = train_entry_v10_ctx_transformer(
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=str(args.device),
        seq_len=int(args.seq_len),
        early_stopping_patience=int(args.early_stopping_patience),
        gx1_data=str(args.gx1_data),
        seed=int(args.seed),
        strict_deterministic=bool(args.strict_deterministic),
        num_workers=int(args.num_workers),
    )
    log.info(f"[DONE] CTX bundle: {bundle_dir}")


if __name__ == "__main__":
    main()