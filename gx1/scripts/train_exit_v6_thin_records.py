#!/usr/bin/env python3
"""V3 v6 exit transformer trainer — improvements over v5.

Built on top of train_exit_v5_thin_records.py with these enhancements (per
2026-Q2 audit findings, see project_gx1_audit_findings_2026q2.md):

  + Multi-task loss: train main + profit_protect + family heads jointly
  + Focal loss (γ=2) for class imbalance (replaces pos_weight)
  + Label smoothing (ε=0.05)
  + Cosine LR schedule with linear warmup (10% of total steps)
  + Mixed precision (bfloat16 autocast on cuda)
  + EMA of model weights (decay=0.999)
  + Optional attention pooling (vs last-token)
  + Optional torch.compile

Reuses ThinRecordDataset + LabeledThinDataset from v5 trainer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.exits.training.thin_record_dataset import ThinRecordDataset
from gx1.policy.exit_transformer_v0 import (
    ExitTransformerV0,
    _attach_labels_to_exit_records,
    EXIT_AUX_FAMILY_TO_ID,
    EXIT_AUX_FAMILY_IGNORE_INDEX,
)
from gx1.exits.contracts.registry import get_exit_io_contract
from gx1.scripts.train_exit_v5_thin_records import (
    LabeledThinDataset, split_records_by_time,
)


ACTION = "TRAIN_EXIT_V6_THIN_RECORDS"

DEFAULT_DATASET_DIR = Path("/home/andre2/GX1_DATA/data/training/exit_v3_v5_training_2020_2026")
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/models/exit_transformer_v0")


# ---------------------------------------------------------------------------
# Multi-task labeled dataset
# ---------------------------------------------------------------------------


class MultiTaskThinDataset(LabeledThinDataset):
    """LabeledThinDataset that also returns auxiliary labels for multi-task training.

    Family classes come from `EXIT_AUX_FAMILY_TO_ID` (currently 3-class). Records
    whose `exit_aux_family` is unknown / "NEGATIVE" map to the IGNORE_INDEX
    sentinel so torch's cross_entropy skips them.
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        rec = self.records[idx]
        io = self.base._reconstruct_io_features(rec)
        x = torch.from_numpy(io)
        y_main = torch.tensor(float(rec.get("should_exit", 0.0) or 0.0), dtype=torch.float32)
        y_profit = torch.tensor(float(rec.get("profit_protect_should_exit", 0.0) or 0.0), dtype=torch.float32)
        family_str = str(rec.get("exit_aux_family") or "").strip().upper()
        family_idx = EXIT_AUX_FAMILY_TO_ID.get(family_str, EXIT_AUX_FAMILY_IGNORE_INDEX)
        y_family = torch.tensor(family_idx, dtype=torch.long)
        w = torch.tensor(float(rec.get("sample_weight", 1.0) or 1.0), dtype=torch.float32)
        return x, {"main": y_main, "profit": y_profit, "family": y_family}, w


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def focal_bce_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal BCE: BCE × (1 - p_t)^gamma. Optionally with label smoothing.

    p_t = p if target=1 else (1-p). Down-weights easy examples.
    """
    if label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    focal = ((1 - p_t).clamp(min=1e-6) ** gamma) * bce
    if weight is not None:
        focal = focal * weight
    return focal.mean()


# ---------------------------------------------------------------------------
# Cosine LR schedule with warmup
# ---------------------------------------------------------------------------


def make_cosine_warmup_scheduler(optimizer, *, total_steps: int, warmup_frac: float = 0.1):
    warmup_steps = max(1, int(total_steps * warmup_frac))
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class ModelEMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def evaluate_multitask(model, loader, device, *, focal_gamma: float, label_smoothing: float,
                       loss_weights: Dict[str, float], use_amp: bool) -> Dict[str, float]:
    model.eval()
    losses_main, ys_main, preds_main = [], [], []
    losses_profit, losses_family = [], []
    with torch.no_grad():
        for bi, (x, y_dict, w) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            y_main = y_dict["main"].to(device, non_blocking=True)
            y_profit = y_dict["profit"].to(device, non_blocking=True)
            y_family = y_dict["family"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                h = model._encode(x)
                logit_main = model.head(h).squeeze(-1)
                logit_profit = model.profit_protect_head(h).squeeze(-1)
                logit_family = model.family_head(h)

                loss_main = focal_bce_loss(logit_main, y_main, gamma=focal_gamma,
                                           weight=w, label_smoothing=label_smoothing)
                loss_profit = focal_bce_loss(logit_profit, y_profit, gamma=focal_gamma,
                                             weight=w, label_smoothing=label_smoothing)
                fam_per = F.cross_entropy(
                    logit_family, y_family, label_smoothing=label_smoothing,
                    reduction="none", ignore_index=EXIT_AUX_FAMILY_IGNORE_INDEX,
                )
                fam_mask = (y_family != EXIT_AUX_FAMILY_IGNORE_INDEX).to(fam_per.dtype)
                fam_n = fam_mask.sum().clamp(min=1.0)
                loss_family = (fam_per * w * fam_mask).sum() / fam_n

            losses_main.append(float(loss_main.item()))
            losses_profit.append(float(loss_profit.item()))
            losses_family.append(float(loss_family.item()))
            ys_main.append(y_main.detach().cpu().float().numpy())
            preds_main.append(torch.sigmoid(logit_main.float()).detach().cpu().numpy())
    if not losses_main:
        return {"loss_main": float("nan"), "n": 0}
    y_arr = np.concatenate(ys_main); p_arr = np.concatenate(preds_main)
    weighted_total = (
        loss_weights.get("main", 1.0) * float(np.mean(losses_main))
        + loss_weights.get("profit", 0.5) * float(np.mean(losses_profit))
        + loss_weights.get("family", 0.3) * float(np.mean(losses_family))
    )
    return {
        "loss_main": float(np.mean(losses_main)),
        "loss_profit": float(np.mean(losses_profit)),
        "loss_family": float(np.mean(losses_family)),
        "loss_total": weighted_total,
        "loss": weighted_total,  # alias for early-stop logic
        "n": int(y_arr.size),
        "pos_rate": float(y_arr.mean()),
        "pred_mean": float(p_arr.mean()),
        "acc_at_05": float(((p_arr >= 0.5).astype(np.float32) == y_arr).mean()),
    }


def train_one_epoch_multitask(model, loader, optimizer, scheduler, device, *,
                              focal_gamma: float, label_smoothing: float,
                              loss_weights: Dict[str, float], ema: Optional[ModelEMA],
                              use_amp: bool, log_every: int = 200) -> Dict[str, float]:
    model.train()
    running_main, running_profit, running_family, running_n = 0.0, 0.0, 0.0, 0
    t0 = time.time()
    for bi, (x, y_dict, w) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        y_main = y_dict["main"].to(device, non_blocking=True)
        y_profit = y_dict["profit"].to(device, non_blocking=True)
        y_family = y_dict["family"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            h = model._encode(x)
            logit_main = model.head(h).squeeze(-1)
            logit_profit = model.profit_protect_head(h).squeeze(-1)
            logit_family = model.family_head(h)

            loss_main = focal_bce_loss(logit_main, y_main, gamma=focal_gamma,
                                       weight=w, label_smoothing=label_smoothing)
            loss_profit = focal_bce_loss(logit_profit, y_profit, gamma=focal_gamma,
                                         weight=w, label_smoothing=label_smoothing)
            fam_per = F.cross_entropy(
                logit_family, y_family, label_smoothing=label_smoothing,
                reduction="none", ignore_index=EXIT_AUX_FAMILY_IGNORE_INDEX,
            )
            fam_mask = (y_family != EXIT_AUX_FAMILY_IGNORE_INDEX).to(fam_per.dtype)
            fam_n = fam_mask.sum().clamp(min=1.0)
            loss_family = (fam_per * w * fam_mask).sum() / fam_n
            loss = (loss_weights.get("main", 1.0) * loss_main
                    + loss_weights.get("profit", 0.5) * loss_profit
                    + loss_weights.get("family", 0.3) * loss_family)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

        running_main += float(loss_main.item()) * y_main.size(0)
        running_profit += float(loss_profit.item()) * y_main.size(0)
        running_family += float(loss_family.item()) * y_main.size(0)
        running_n += int(y_main.size(0))
        if (bi + 1) % log_every == 0:
            avg = running_main / max(running_n, 1)
            rate = running_n / max(time.time() - t0, 1e-6)
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"   batch {bi+1}: avg_main={avg:.4f} lr={cur_lr:.2e} rate={rate:.0f}/s", flush=True)
    return {
        "train_loss_main": running_main / max(running_n, 1),
        "train_loss_profit": running_profit / max(running_n, 1),
        "train_loss_family": running_family / max(running_n, 1),
        "n": running_n, "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--bundle-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20,
                        help="Default 20 with early-stop patience 4. Cosine LR schedule "
                             "auto-adjusts to total steps across all epochs.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Default 256 for RTX 3090 (24GB). v5 used 64 which left "
                             "~7% VRAM utilization on table. lr should scale linearly with batch "
                             "(default lr=1.2e-3 for batch=256 vs lr=3e-4 for batch=64).")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1.2e-3,
                        help="Default 1.2e-3 (linear LR scaling for batch=256 vs v5's 3e-4 at batch=64). "
                             "Cosine decay schedule applied on top.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128, help="Capacity bump from v5 default 64")
    parser.add_argument("--n-layers", type=int, default=6, help="Capacity bump from v5 default 4")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--records-limit", type=int, default=None)
    parser.add_argument("--train-cutoff", type=str, default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--val-cutoff", type=str, default="2025-09-01T00:00:00+00:00")
    parser.add_argument("--exit-io-version", type=str, default="EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512")
    parser.add_argument("--early-stop-patience", type=int, default=4,
                        help="Stop after N non-improving val epochs. With epochs=20 + patience=4, "
                             "training runs ~6-12 epochs typical depending on convergence speed.")

    # New v6 knobs
    parser.add_argument("--side-balance-weight", action="store_true",
                        help="Weight samples inversely by side frequency (counter long/short skew)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (0 = standard BCE)")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--loss-weight-main", type=float, default=1.0)
    parser.add_argument("--loss-weight-profit", type=float, default=0.5)
    parser.add_argument("--loss-weight-family", type=float, default=0.3)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use bfloat16 autocast on cuda")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Wrap model in torch.compile (PyTorch 2.x)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[{ACTION}] device={device} dataset={args.dataset_dir}", flush=True)

    # Build the offset index when EXIT_LABEL_WORKERS > 1 so spawn workers can
    # seek+read their slice from JSONL without inheriting parent's records dict.
    _build_offset_idx = int(os.environ.get("EXIT_LABEL_WORKERS", "1")) > 1
    base = ThinRecordDataset(
        args.dataset_dir,
        records_limit=args.records_limit,
        build_offset_index=_build_offset_idx,
    )
    print(f"[{ACTION}] loaded {len(base.records):,} records  matrix={base.matrix.shape}  "
          f"input_dim={base.input_dim} window_len={base.window_len}"
          f"  offset_index={'yes' if _build_offset_idx else 'no'}", flush=True)

    train_recs, val_recs, test_recs = split_records_by_time(
        base.records, train_cutoff=args.train_cutoff, val_cutoff=args.val_cutoff,
    )
    print(f"[{ACTION}] split (group-by-trade): train={len(train_recs):,} val={len(val_recs):,} test={len(test_recs):,}",
          flush=True)

    train_ds = MultiTaskThinDataset(base, records=train_recs)
    val_ds = MultiTaskThinDataset(base, records=val_recs, flatten_scalars=False)
    test_ds = MultiTaskThinDataset(base, records=test_recs, flatten_scalars=False)

    train_pos = float(np.mean([float(r.get("should_exit", 0.0) or 0.0) for r in train_recs])) if train_recs else 0.0
    val_pos = float(np.mean([float(r.get("should_exit", 0.0) or 0.0) for r in val_recs])) if val_recs else 0.0
    print(f"[{ACTION}] should_exit pos rate: train={train_pos:.4f} val={val_pos:.4f}", flush=True)

    if args.side_balance_weight:
        n_long = sum(1 for r in train_recs if r.get("side") == "long")
        n_short = sum(1 for r in train_recs if r.get("side") == "short")
        n_total = max(n_long + n_short, 1)
        long_w = n_total / max(2 * n_long, 1)
        short_w = n_total / max(2 * n_short, 1)
        print(f"[{ACTION}] side balance: long_w={long_w:.3f} short_w={short_w:.3f}", flush=True)
        for rs in (train_recs, val_recs, test_recs):
            for r in rs:
                base_w = float(r.get("sample_weight", 1.0) or 1.0)
                r["sample_weight"] = base_w * (long_w if r.get("side") == "long" else short_w)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = ExitTransformerV0(
        input_dim=int(base.input_dim), window_len=int(base.window_len),
        d_model=int(args.d_model), n_heads=int(args.n_heads), n_layers=int(args.n_layers),
        dropout=float(args.dropout),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{ACTION}] model: ExitTransformerV0 d_model={args.d_model} n_layers={args.n_layers} "
          f"params={n_params:,}", flush=True)

    if args.torch_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[{ACTION}] torch.compile applied (reduce-overhead)", flush=True)
        except Exception as e:
            print(f"[{ACTION}] torch.compile failed: {e}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = make_cosine_warmup_scheduler(optimizer, total_steps=total_steps, warmup_frac=args.warmup_frac)
    print(f"[{ACTION}] LR schedule: cosine + {int(args.warmup_frac*100)}% warmup ({total_steps} total steps)",
          flush=True)

    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)
    if ema is not None:
        print(f"[{ACTION}] EMA enabled (decay={args.ema_decay})", flush=True)

    use_amp = bool(args.mixed_precision and device.type == "cuda")
    if use_amp:
        print(f"[{ACTION}] mixed precision (bfloat16) enabled", flush=True)

    loss_weights = {"main": args.loss_weight_main, "profit": args.loss_weight_profit, "family": args.loss_weight_family}
    print(f"[{ACTION}] loss weights: {loss_weights}  focal_gamma={args.focal_gamma}  label_smooth={args.label_smoothing}",
          flush=True)

    best_val_loss = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    history = []
    for epoch in range(args.epochs):
        print(f"[{ACTION}] === epoch {epoch+1}/{args.epochs} ===", flush=True)
        train_stats = train_one_epoch_multitask(
            model, train_loader, optimizer, scheduler, device,
            focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
            loss_weights=loss_weights, ema=ema, use_amp=use_amp,
        )
        # Evaluate using EMA weights if available
        eval_model = copy.deepcopy(model) if ema is not None else model
        if ema is not None:
            ema.apply_to(eval_model)
        val_stats = evaluate_multitask(
            eval_model, val_loader, device,
            focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
            loss_weights=loss_weights, use_amp=use_amp,
        )
        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})
        print(f"[{ACTION}] epoch {epoch+1}: train_main={train_stats['train_loss_main']:.4f} "
              f"val_main={val_stats['loss_main']:.4f} val_total={val_stats['loss_total']:.4f} "
              f"val_acc={val_stats['acc_at_05']:.4f} ({train_stats['elapsed_s']:.1f}s)", flush=True)
        if val_stats["loss_total"] < best_val_loss - 1e-4:
            best_val_loss = val_stats["loss_total"]
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            epochs_no_improve = 0
            print(f"[{ACTION}]   new best val_loss={best_val_loss:.4f}", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"[{ACTION}] early stop after {args.early_stop_patience} non-improving epochs", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_eval_model = copy.deepcopy(model) if ema is not None else model
    if ema is not None:
        ema.apply_to(final_eval_model)
    test_stats = evaluate_multitask(
        final_eval_model, test_loader, device,
        focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
        loss_weights=loss_weights, use_amp=use_amp,
    )
    print(f"[{ACTION}] test: loss_main={test_stats['loss_main']:.4f} acc={test_stats['acc_at_05']:.4f} "
          f"n={test_stats['n']}", flush=True)

    bundle_name = args.bundle_name or f"EXIT_V6_THIN__BIDIR_2026Q2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle_dir = Path(args.out_dir) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    state_path = bundle_dir / "exit_transformer_v0.pt"
    torch.save(model.state_dict(), state_path)
    state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()

    contract = get_exit_io_contract(args.exit_io_version)
    config = {
        "input_dim": int(base.input_dim), "window_len": int(base.window_len),
        "d_model": int(args.d_model), "n_layers": int(args.n_layers), "n_heads": int(args.n_heads),
        "dropout": float(args.dropout),
        "exit_ml_io_version": str(contract["io_version"]),
        "feature_names_hash": str(contract["feature_hash"]),
        "feature_count": int(contract["feature_count"]),
    }
    (bundle_dir / "transformer_config.json").write_text(json.dumps(config, indent=2))

    manifest = {
        "action_v1": ACTION,
        "bundle_name": bundle_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "exit_io_version": str(contract["io_version"]),
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "d_model": args.d_model, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "dropout": args.dropout, "seed": args.seed,
            "focal_gamma": args.focal_gamma, "label_smoothing": args.label_smoothing,
            "warmup_frac": args.warmup_frac, "ema_decay": args.ema_decay if ema is not None else None,
            "mixed_precision": use_amp, "torch_compile": args.torch_compile,
            "loss_weights": loss_weights, "side_balance_weight": args.side_balance_weight,
            "train_cutoff": args.train_cutoff, "val_cutoff": args.val_cutoff,
        },
        "best_val_loss": float(best_val_loss),
        "test_metrics": test_stats,
        "n_train": len(train_recs), "n_val": len(val_recs), "n_test": len(test_recs),
        "train_pos_rate": train_pos,
        "model_state_dict_sha256": state_sha,
        "history": history,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    print(f"[{ACTION}] bundle saved at {bundle_dir}", flush=True)
    print(json.dumps(manifest, indent=2, default=float))


if __name__ == "__main__":
    main()
