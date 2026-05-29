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

DEFAULT_DATASET_DIR = Path("/home/andre2/GX1_DATA/data/training/exit_v3_v7_training_2020_2026_canonical_v3")
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


_MTF_KEYS = ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1")
_V3_DIP_HORIZONS = (12, 48, 144)
_V3_DIP_QUANTILES = (0.5, 0.1)  # recovery_p50, recovery_p10 — order matches V3_DIP_QUANTILES


def _extract_mtf(y_dict, device):
    """Pull the per-TF windows out of the batch so the multi-TF branches actually
    get fed (bug fix 2026-05-26: _encode was called without them → branches untrained)."""
    return {k: y_dict[k].to(device, non_blocking=True) for k in _MTF_KEYS if k in y_dict}


_V3_NEW_HORIZONS = (12, 48, 144)
_V3_TAIL_QUANTILE = 0.9


def _v3_aux_loss(model, h, y_dict, device):
    """Combined V3 aux-head loss (2026-05-26): dip(6,pinball) + timing(6,smooth_l1)
    + tail-risk(3,pinball q=0.9) + vol-forecast(3,smooth_l1) + forecast(3,smooth_l1).
    Each term is 0 when its head/targets are absent (gated). Mirror of V10's
    dip_forecast_loss. Targets are exit-oriented (forward-from-in-trade pnl)."""
    total = torch.zeros((), device=device)
    # dip (6) — pinball quantiles of forward recovery
    if getattr(model, "enable_dip_head", False) and "dip_recovery_K12" in y_dict:
        cols, qs = [], []
        for K in _V3_DIP_HORIZONS:
            for q in _V3_DIP_QUANTILES:
                cols.append(y_dict[f"dip_recovery_K{K}"].to(device, non_blocking=True)); qs.append(q)
        tgt = torch.stack(cols, dim=1).float()
        q = torch.tensor(qs, device=device, dtype=tgt.dtype).view(1, -1)
        err = tgt - model.dip_head(h).float()
        total = total + torch.maximum(q * err, (q - 1.0) * err).mean()
    # timing (6) — smooth_l1 on {time_to_recovery_frac, time_to_worst_frac} × K
    if getattr(model, "enable_timing_head", False) and "v3_time_to_recovery_frac_K12" in y_dict:
        tg = [y_dict[f"v3_{nm}_K{K}"].to(device, non_blocking=True)
              for K in _V3_NEW_HORIZONS for nm in ("time_to_recovery_frac", "time_to_worst_frac")]
        total = total + F.smooth_l1_loss(model.timing_head(h).float(), torch.stack(tg, dim=1).float())
    # tail-risk (3) — pinball q=0.9 of worst further drawdown
    if getattr(model, "enable_tail_risk_head", False) and "v3_tail_mae_K12" in y_dict:
        tg = [y_dict[f"v3_tail_mae_K{K}"].to(device, non_blocking=True) for K in _V3_NEW_HORIZONS]
        t = torch.stack(tg, dim=1).float(); err = t - model.tail_risk_head(h).float()
        total = total + torch.maximum(_V3_TAIL_QUANTILE * err, (_V3_TAIL_QUANTILE - 1.0) * err).mean()
    # vol-forecast (3) — smooth_l1
    if getattr(model, "enable_vol_forecast_head", False) and "v3_vol_fwd_K12" in y_dict:
        tg = [y_dict[f"v3_vol_fwd_K{K}"].to(device, non_blocking=True) for K in _V3_NEW_HORIZONS]
        total = total + F.smooth_l1_loss(model.vol_forecast_head(h).float(), torch.stack(tg, dim=1).float())
    # forecast (3) — smooth_l1 on forward pnl change
    if getattr(model, "enable_forecast_head", False) and "v3_forecast_K12" in y_dict:
        tg = [y_dict[f"v3_forecast_K{K}"].to(device, non_blocking=True) for K in _V3_NEW_HORIZONS]
        total = total + F.smooth_l1_loss(model.forecast_head(h).float(), torch.stack(tg, dim=1).float())
    return total


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
                h = model._encode(x, **_extract_mtf(y_dict, device))
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
    out = {
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
    # Discrimination metrics on the exit-now (should_exit) head. acc is useless on a
    # ~4%-positive class (majority baseline ~0.96); these are the real verdict.
    out.update(_exit_discrimination_metrics(y_arr, p_arr))
    return out


def _exit_discrimination_metrics(y_arr: np.ndarray, p_arr: np.ndarray) -> Dict[str, float]:
    """Precision/recall/F1@0.5 + ROC-AUC + PR-AUC for the rare exit-now class.

    For a ~4%-positive task, PR-AUC (average precision) is the most informative
    single number. Falls back to NaN-safe manual precision/recall if sklearn is
    unavailable; AUCs need both classes present.
    """
    y = y_arr.astype(np.int32)
    pred = (p_arr >= 0.5).astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and np.isfinite(prec) and np.isfinite(rec)) else float("nan")
    roc = float("nan")
    pr = float("nan")
    if y.min() != y.max():  # both classes present
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            roc = float(roc_auc_score(y, p_arr))
            pr = float(average_precision_score(y, p_arr))
        except Exception:
            pass
    return {
        "precision_at_05": float(prec),
        "recall_at_05": float(rec),
        "f1_at_05": float(f1),
        "roc_auc": roc,
        "pr_auc": pr,
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
            h = model._encode(x, **_extract_mtf(y_dict, device))
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
            loss = loss + loss_weights.get("dip", 0.2) * _v3_aux_loss(model, h, y_dict, device)

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
# Standalone entry point — REMOVED (2026-05-25).
# This module is now a LIBRARY ONLY: its shared losses / dataset class / eval /
# scheduler / EMA are imported by train_exit_v6_disk_thin.py, the ONE true
# multi-TF V3 trainer. The old single-TF training main() was deleted because a
# single-TF V3 bundle is rejected by the live loader (requires multi_tf.enabled)
# and violates the multi-TF mandate — keeping it was a duplicate, wrong "truth".
# ---------------------------------------------------------------------------
def main() -> None:
    raise SystemExit(
        "train_exit_v6_thin_records is LIBRARY-ONLY. The single-TF V3 trainer was "
        "removed (multi-TF is mandatory; single-TF bundles fail the live loader). "
        "Use: python -m gx1.scripts.train_exit_v6_disk_thin --enable-multi-tf "
        "--enable-pos-enc ..."
    )


if __name__ == "__main__":
    main()
