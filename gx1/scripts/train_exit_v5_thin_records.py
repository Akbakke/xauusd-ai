#!/usr/bin/env python3
"""V3-style exit transformer trainer for the V5 thin-record dataset.

Reads the thin-record dataset built by `materialize_build_v3_training_dataset_v1.py`,
reconstructs (window_len=512, input_dim=89) io_features per sample via
`ThinRecordDataset`, derives binary `should_exit` labels via the existing
`_attach_labels_to_exit_records` from `gx1.policy.exit_transformer_v0`, and
trains an `ExitTransformerV0` model.

Saves the bundle to a directory compatible with the existing exit transformer
loader (model_state_dict.pt + transformer_config.json + manifest.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.exits.training.thin_record_dataset import ThinRecordDataset
from gx1.policy.exit_transformer_v0 import ExitTransformerV0, _attach_labels_to_exit_records
from gx1.exits.contracts.registry import get_exit_io_contract


ACTION = "TRAIN_EXIT_V5_THIN_RECORDS"

DEFAULT_DATASET_DIR = Path("/home/andre2/GX1_DATA/data/training/exit_v3_v7_training_2020_2026_canonical_v3")
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/models/exit_transformer_v0")


# Module-level worker base. Set in the parent BEFORE Pool fork so workers
# inherit the ThinRecordDataset (mmap arrays + records dict) via copy-on-write
# fork — avoids pickling 1.5M+ records to each worker.
_WORKER_BASE: Optional[ThinRecordDataset] = None
_WORKER_IO_VERSION: str = ""


def _label_one_trade(args: Tuple[str, List[Dict[str, Any]]]) -> Tuple[str, List[Dict[str, Any]]]:
    uid, trade_recs = args
    base = _WORKER_BASE
    if base is None:
        raise RuntimeError("[LabeledThinDataset] worker base not initialized (fork issue)")
    for rec in trade_recs:
        rec["io_features"] = base._reconstruct_io_features(rec).tolist()
    _attach_labels_to_exit_records(trade_recs)
    for rec in trade_recs:
        rec.pop("io_features", None)
    return uid, trade_recs


# ---------------------------------------------------------------------------
# Spawn-based parallel labeler (no fork-COW issues)
#
# Workers are FRESH Python processes. They open ThinRecordDataset with
# load_records_eagerly=False (matrix mmap + overlay_index only) and read their
# trade-records on demand from the JSONL file via seek+byte_length.
#
# Result: each worker uses ~150 MB instead of 5-9 GB, and we can run 8-16
# workers on a 16 GB host with no swap pressure.
# ---------------------------------------------------------------------------


def _spawn_init_worker(dataset_dir: str) -> None:
    """Worker initializer (spawn): open a fresh ThinRecordDataset with no records.

    The matrix and overlays mmaps + overlay_index are tiny per-worker (<200 MB).
    Records are read lazily per-task from JSONL.
    """
    global _WORKER_BASE, _WORKER_IO_VERSION
    _WORKER_BASE = ThinRecordDataset(
        dataset_dir,
        load_records_eagerly=False,
        build_offset_index=False,
    )
    _WORKER_IO_VERSION = str(_WORKER_BASE.manifest.get("io_version", ""))


def _spawn_label_one_trade(
    args: Tuple[str, List[Tuple[int, int]]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """Worker task: read a trade's records from JSONL, label, return.

    `args` = (trade_uid, [(byte_offset, byte_length), ...])
    """
    uid, byte_ranges = args
    base = _WORKER_BASE
    if base is None or base.records_path is None:
        raise RuntimeError("[spawn labeler] worker not initialized")

    trade_recs: List[Dict[str, Any]] = []
    with open(base.records_path, "rb") as f:
        for byte_off, byte_len in byte_ranges:
            f.seek(byte_off)
            line = f.read(byte_len)
            stripped = line.strip()
            if stripped:
                trade_recs.append(json.loads(stripped))

    # Mirror v5/v6 LabeledThinDataset.flatten_scalars=True behavior so the
    # labeler can index io_features by feature-name correctly.
    for rec in trade_recs:
        sc = rec.get("scalars") or {}
        for k, v in sc.items():
            rec.setdefault(k, v)
        rec.setdefault("exit_ml_io_version", _WORKER_IO_VERSION)

    # Reconstruct io_features → label → clear
    for rec in trade_recs:
        rec["io_features"] = base._reconstruct_io_features(rec).tolist()
    _attach_labels_to_exit_records(trade_recs)
    for rec in trade_recs:
        rec.pop("io_features", None)
    return uid, trade_recs


def _label_via_jsonl_spawn(
    dataset_dir: Path,
    uid_to_offsets: Dict[str, List[Tuple[int, int]]],
    n_workers: int,
    *,
    n_trades: int,
    chunksize: int = 16,
) -> Dict[str, List[Dict[str, Any]]]:
    """Parallel-label trades via spawn workers reading JSONL on demand.

    Returns {trade_uid: [labeled_record, ...]}.
    """
    jobs = list(uid_to_offsets.items())
    out: Dict[str, List[Dict[str, Any]]] = {}
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_spawn_init_worker,
        initargs=(str(dataset_dir),),
    ) as pool:
        for ti, (uid, labeled) in enumerate(
            pool.imap_unordered(_spawn_label_one_trade, jobs, chunksize=chunksize)
        ):
            out[uid] = labeled
            if (ti + 1) % 5000 == 0:
                print(
                    f"   {ti+1:,}/{n_trades:,} trades labeled ({time.time()-t0:.1f}s)",
                    flush=True,
                )
    return out


class LabeledThinDataset(Dataset):
    """Wraps ThinRecordDataset; attaches `should_exit` labels via the V3 labeler.

    The V3 labeler (`_attach_labels_to_exit_records`) requires per-record
    `io_features` arrays so it can index XGB-bridge fields (p_flat, p_hat,
    uncertainty, margin) that drive its no-edge / signal-not-supportive logic.

    Our thin records don't carry io_features. We reconstruct them per-trade
    during labeling, then clear them to keep memory bounded. Peak: ~2 MB per
    trade (48 records × 512 × 89 float32) — released after labeling.
    """

    def __init__(
        self, base: ThinRecordDataset, *,
        records: Optional[List[Dict[str, Any]]] = None,
        flatten_scalars: bool = True,
    ) -> None:
        self.base = base
        self.records = records if records is not None else base.records
        if flatten_scalars:
            for rec in self.records:
                sc = rec.get("scalars") or {}
                for k, v in sc.items():
                    rec.setdefault(k, v)
                # Tell the labeler which exit_io contract to use so it can index
                # io_features by feature-name correctly.
                rec.setdefault("exit_ml_io_version", base.manifest.get("io_version", ""))

        # Process trade-by-trade: reconstruct io_features → label → clear.
        from collections import defaultdict
        trade_idx: Dict[str, List[int]] = defaultdict(list)
        for idx, rec in enumerate(self.records):
            tid = rec.get("trade_uid") or rec.get("trade_id")
            if tid:
                trade_idx[tid].append(idx)

        n_trades = len(trade_idx)
        # Default 1 (sequential). Multi-worker labeling COWs the 2.5M records
        # dict in each worker → ~5-9 GB RSS each → swap-thrash on 16 GB hosts.
        # Smoke-tested at 10k records (2.6× speedup) but doesn't scale to full
        # train split. Set EXIT_LABEL_WORKERS=N>1 only on hosts with >32 GB RAM.
        n_workers = int(os.environ.get("EXIT_LABEL_WORKERS", "1"))
        print(
            f"[LabeledThinDataset] labeling {n_trades:,} trades "
            f"({len(self.records):,} records) with {n_workers} workers...",
            flush=True,
        )
        t0 = time.time()

        if n_workers > 1:
            # Spawn-based labeler reads from JSONL — needs base.uid_to_offsets
            if getattr(base, "uid_to_offsets", None) is None:
                raise RuntimeError(
                    "[LabeledThinDataset] parallel labeling requires "
                    "ThinRecordDataset(build_offset_index=True). Got base "
                    "without uid_to_offsets."
                )
            # Build subset offsets for this split's uids only
            my_uids = list(trade_idx.keys())
            subset_offsets = {
                uid: base.uid_to_offsets[uid]
                for uid in my_uids
                if uid in base.uid_to_offsets
            }
            missing = len(my_uids) - len(subset_offsets)
            if missing:
                print(
                    f"[LabeledThinDataset] WARN: {missing} uids missing from offset index "
                    "— falling back to sequential for those.",
                    flush=True,
                )
            labeled_dict = _label_via_jsonl_spawn(
                base.dataset_dir, subset_offsets, n_workers, n_trades=n_trades,
            )
            # Replace records in self.records by index. Labeled trade_recs are
            # in JSONL-order (per uid), and trade_idx[uid] holds same-order indices,
            # so we can zip directly.
            for uid in my_uids:
                if uid in labeled_dict:
                    indices = trade_idx[uid]
                    labeled = labeled_dict[uid]
                    if len(indices) != len(labeled):
                        raise RuntimeError(
                            f"[LabeledThinDataset] uid={uid}: index count "
                            f"{len(indices)} != labeled count {len(labeled)}"
                        )
                    for j, rec in zip(indices, labeled):
                        self.records[j] = rec
        else:
            for ti, (tid, idxs) in enumerate(trade_idx.items()):
                trade_recs = [self.records[i] for i in idxs]
                for rec in trade_recs:
                    rec["io_features"] = base._reconstruct_io_features(rec).tolist()
                _attach_labels_to_exit_records(trade_recs)
                for rec in trade_recs:
                    rec.pop("io_features", None)
                if (ti + 1) % 5000 == 0:
                    print(f"   {ti+1:,}/{n_trades:,} trades labeled ({time.time()-t0:.1f}s)", flush=True)
        print(f"[LabeledThinDataset] labeling done in {time.time()-t0:.1f}s", flush=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        io = self.base._reconstruct_io_features(rec)
        x = torch.from_numpy(io)
        y = torch.tensor(float(rec.get("should_exit", 0.0) or 0.0), dtype=torch.float32)
        w = torch.tensor(float(rec.get("sample_weight", 1.0) or 1.0), dtype=torch.float32)
        return x, y, w


def split_records_by_time(
    records: List[Dict[str, Any]], *, train_cutoff: str, val_cutoff: str,
) -> Tuple[List, List, List]:
    """Split records into train/val/test by trade_uid (not record-by-record).

    Each trade is assigned to exactly one split based on its FIRST record's
    timestamp. This prevents leakage where records from the same trade end up
    in both train and test (records spanning a cutoff boundary).
    """
    train_ts = pd.Timestamp(train_cutoff, tz="UTC")
    val_ts = pd.Timestamp(val_cutoff, tz="UTC")

    # Group by trade and find earliest timestamp per trade
    from collections import defaultdict
    trades: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        tid = rec.get("trade_uid") or rec.get("trade_id") or ""
        trades[tid].append(rec)

    train, val, test = [], [], []
    for tid, trade_recs in trades.items():
        # Use earliest record's timestamp to assign trade to split
        try:
            tss = []
            for r in trade_recs:
                ts = pd.Timestamp(r.get("ts"))
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                tss.append(ts)
            trade_first_ts = min(tss)
        except Exception:
            train.extend(trade_recs)
            continue
        if trade_first_ts < train_ts:
            train.extend(trade_recs)
        elif trade_first_ts < val_ts:
            val.extend(trade_recs)
        else:
            test.extend(trade_recs)
    return train, val, test


def evaluate(model, loader, device, *, pos_weight: Optional[torch.Tensor] = None,
             max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    losses, ys, preds = [], [], []
    with torch.no_grad():
        for bi, (x, y, w) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x, y, w = x.to(device, non_blocking=True), y.to(device, non_blocking=True), w.to(device, non_blocking=True)
            logits = model.forward_logits(x)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, weight=w, pos_weight=pos_weight, reduction="mean"
            )
            losses.append(float(loss.item()))
            ys.append(y.detach().cpu().numpy())
            preds.append(torch.sigmoid(logits).detach().cpu().numpy())
    if not losses:
        return {"loss": float("nan"), "n": 0, "pos_rate": float("nan"), "acc_at_05": float("nan")}
    y_arr = np.concatenate(ys)
    p_arr = np.concatenate(preds)
    return {
        "loss": float(np.mean(losses)),
        "n": int(y_arr.size),
        "pos_rate": float(y_arr.mean()),
        "pred_mean": float(p_arr.mean()),
        "acc_at_05": float(((p_arr >= 0.5).astype(np.float32) == y_arr).mean()),
    }


def train_one_epoch(model, loader, optimizer, device, *, pos_weight: Optional[torch.Tensor] = None,
                    log_every: int = 200) -> Dict[str, float]:
    model.train()
    running_loss, running_n = 0.0, 0
    t0 = time.time()
    for bi, (x, y, w) in enumerate(loader):
        x, y, w = x.to(device, non_blocking=True), y.to(device, non_blocking=True), w.to(device, non_blocking=True)
        logits = model.forward_logits(x)
        loss = F.binary_cross_entropy_with_logits(
            logits, y, weight=w, pos_weight=pos_weight, reduction="mean"
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += float(loss.item()) * y.size(0)
        running_n += int(y.size(0))
        if (bi + 1) % log_every == 0:
            avg = running_loss / max(running_n, 1)
            rate = running_n / max(time.time() - t0, 1e-6)
            print(f"   batch {bi+1}: avg_loss={avg:.4f} rate={rate:.0f}/s", flush=True)
    return {"train_loss": running_loss / max(running_n, 1), "n": running_n, "elapsed_s": time.time() - t0}


def main() -> None:
    parser = argparse.ArgumentParser(description=ACTION)
    parser.add_argument("--dataset-dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--bundle-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 = single-process (recommended; mp workers each replicate the 800MB matrix mmap + 2.5M record dict)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos-weight", type=float, default=None,
                        help="BCE pos_weight for class imbalance. If None, auto-set to (1-pos_rate)/pos_rate")
    parser.add_argument("--side-balance-weight", action="store_true",
                        help="Weight samples inversely by side frequency to counter long/short skew (~5x for short)")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--records-limit", type=int, default=None)
    parser.add_argument("--train-cutoff", type=str, default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--val-cutoff", type=str, default="2025-09-01T00:00:00+00:00")
    parser.add_argument("--exit-io-version", type=str, default="EXIT_IO_V5_CTX_EXTENDED_SMC_M1L512")
    parser.add_argument("--early-stop-patience", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[{ACTION}] device={device} dataset={args.dataset_dir}", flush=True)

    base = ThinRecordDataset(args.dataset_dir, records_limit=args.records_limit)
    print(f"[{ACTION}] loaded {len(base.records):,} records  matrix={base.matrix.shape}  "
          f"input_dim={base.input_dim} window_len={base.window_len}", flush=True)

    train_recs, val_recs, test_recs = split_records_by_time(
        base.records, train_cutoff=args.train_cutoff, val_cutoff=args.val_cutoff,
    )
    print(f"[{ACTION}] split: train={len(train_recs):,} val={len(val_recs):,} test={len(test_recs):,}",
          flush=True)

    train_ds = LabeledThinDataset(base, records=train_recs)
    val_ds = LabeledThinDataset(base, records=val_recs, flatten_scalars=False)
    test_ds = LabeledThinDataset(base, records=test_recs, flatten_scalars=False)

    train_pos = float(np.mean([float(r.get("should_exit", 0.0) or 0.0) for r in train_recs])) if train_recs else 0.0
    val_pos = float(np.mean([float(r.get("should_exit", 0.0) or 0.0) for r in val_recs])) if val_recs else 0.0
    print(f"[{ACTION}] should_exit pos rate: train={train_pos:.4f} val={val_pos:.4f}", flush=True)

    # Side-class balance weighting (counter the 83/17 long/short skew from V10 v2)
    if args.side_balance_weight:
        n_long = sum(1 for r in train_recs if r.get("side") == "long")
        n_short = sum(1 for r in train_recs if r.get("side") == "short")
        n_total = n_long + n_short
        long_w = float(n_total) / max(2 * n_long, 1)
        short_w = float(n_total) / max(2 * n_short, 1)
        print(f"[{ACTION}] side balance: long_w={long_w:.3f} short_w={short_w:.3f} "
              f"(n_long={n_long:,} n_short={n_short:,})", flush=True)
        for rs in (train_recs, val_recs, test_recs):
            for r in rs:
                base_w = float(r.get("sample_weight", 1.0) or 1.0)
                r["sample_weight"] = base_w * (long_w if r.get("side") == "long" else short_w)

    # Class imbalance pos_weight for BCE
    if args.pos_weight is not None:
        pw = float(args.pos_weight)
    else:
        # Auto: (1 - p) / p where p = train_pos rate
        pw = float((1.0 - train_pos) / max(train_pos, 1e-6))
        pw = max(1.0, min(pw, 50.0))  # clamp to [1, 50]
    pos_weight_t = torch.tensor(pw, device=device, dtype=torch.float32)
    print(f"[{ACTION}] BCE pos_weight={pw:.3f}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = ExitTransformerV0(
        input_dim=int(base.input_dim),
        window_len=int(base.window_len),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{ACTION}] model: ExitTransformerV0 input_dim={base.input_dim} d_model={args.d_model} "
          f"n_layers={args.n_layers} params={n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    history = []
    for epoch in range(args.epochs):
        print(f"[{ACTION}] === epoch {epoch+1}/{args.epochs} ===", flush=True)
        train_stats = train_one_epoch(model, train_loader, optimizer, device, pos_weight=pos_weight_t)
        val_stats = evaluate(model, val_loader, device, pos_weight=pos_weight_t)
        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})
        print(f"[{ACTION}] epoch {epoch+1}: train_loss={train_stats['train_loss']:.4f} "
              f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc_at_05']:.4f} "
              f"({train_stats['elapsed_s']:.1f}s)", flush=True)
        if val_stats["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_stats["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"[{ACTION}]   new best val_loss={best_val_loss:.4f}", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"[{ACTION}] early stop after {args.early_stop_patience} non-improving epochs",
                      flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_stats = evaluate(model, test_loader, device, pos_weight=pos_weight_t)
    print(f"[{ACTION}] test: loss={test_stats['loss']:.4f} acc={test_stats['acc_at_05']:.4f} "
          f"n={test_stats['n']} pos_rate={test_stats['pos_rate']:.4f}", flush=True)

    bundle_name = args.bundle_name or f"EXIT_V5_THIN__BIDIR_2026Q2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    bundle_dir = Path(args.out_dir) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    state_path = bundle_dir / "exit_transformer_v0.pt"
    torch.save(model.state_dict(), state_path)
    state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()

    contract = get_exit_io_contract(args.exit_io_version)
    config = {
        "input_dim": int(base.input_dim),
        "window_len": int(base.window_len),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
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
