#!/usr/bin/env python3
"""V3 v6 exit-transformer trainer with disk-spilling labeler.

Same model + losses as `train_exit_v6_thin_records.py` (multi-task, focal,
EMA, cosine LR, bf16 AMP) but uses the disk-only data flow from
`gx1.exits.training.disk_labeled_dataset` so parent RAM stays ~500 MB
regardless of dataset size.

Background — why this exists
----------------------------
The in-memory v6 trainer OOM-killed at val labeling on a 16 GB host because
the parent process held: full records dict (8 GB) + 3 split copies + spawn
output dict + page cache → 12+ GB. See project_gx1_v3_v6_retrain_2026q2.md.

This trainer:
  1. Builds ThinRecordDataset with load_records_eagerly=False (mmap-only).
  2. Splits trade UIDs by timestamp via one streaming pass over records.jsonl.
  3. For each split, spawn-labels trades and streams labeled records to a
     per-split labeled JSONL on disk.
  4. Trains with DiskMultiTaskThinDataset that reads one labeled record per
     __getitem__ via byte-offset seek.

Use `--label-spill-dir <path>` to control where labeled JSONL is written
(default: <dataset-dir>/labeled_v6_<utc-stamp>/).
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
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.exits.training.thin_record_dataset import ThinRecordDataset
from gx1.exits.training.disk_labeled_dataset import (
    DiskMultiTaskThinDataset,
    compute_side_weights_from_offsets,
    flatten_uid_offsets,
    label_uids_to_disk,
    scan_labeled_pos_rate,
    split_uids_by_time,
)
from gx1.policy.exit_transformer_v0 import ExitTransformerV0
from gx1.exits.contracts.registry import get_exit_io_contract
# V12.2 multi-TF feature contract (imported unconditionally — used in
# both v8 and v9 config-save paths, value is 19).
from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT

# Reuse the existing v6 components (same training math, same loss / EMA / scheduler).
from gx1.scripts.train_exit_v6_thin_records import (
    ModelEMA,
    evaluate_multitask,
    make_cosine_warmup_scheduler,
    train_one_epoch_multitask,
)


ACTION = "TRAIN_EXIT_V6_DISK_THIN_RECORDS"

DEFAULT_DATASET_DIR = Path("/home/andre2/GX1_DATA/data/training/exit_v3_v7_training_2020_2026_canonical_v3")
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/models/exit_transformer_v0")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _label_split(
    name: str,
    dataset_dir: Path,
    uid_to_offsets: Dict[str, List],
    *,
    spill_dir: Path,
    n_workers: int,
    chunksize: int,
) -> tuple[Path, Dict[str, List], List]:
    """Label one split to disk and return (labeled_jsonl, uid->offsets, flat-offsets)."""
    out_jsonl = spill_dir / f"{name}_labeled.jsonl"
    print(
        f"[{ACTION}] labeling split={name} ({len(uid_to_offsets):,} trades) "
        f"-> {out_jsonl} (workers={n_workers}, chunksize={chunksize})",
        flush=True,
    )
    t0 = time.time()
    label_offsets = label_uids_to_disk(
        dataset_dir, uid_to_offsets,
        out_jsonl=out_jsonl, n_workers=n_workers, chunksize=chunksize,
    )
    elapsed = time.time() - t0
    n_records = sum(len(v) for v in label_offsets.values())
    size_mb = out_jsonl.stat().st_size / (1024 * 1024)
    print(
        f"[{ACTION}] split={name} done in {elapsed:.1f}s "
        f"({n_records:,} records, {size_mb:.1f} MB on disk)",
        flush=True,
    )
    flat = flatten_uid_offsets(label_offsets)
    return out_jsonl, label_offsets, flat


def main() -> None:
    parser = argparse.ArgumentParser(description=ACTION)
    # rule 4 (no silent defaults): an exit retrain MUST point at an explicit, freshly-built
    # dataset. The old default (DEFAULT_DATASET_DIR = cement-vintage exit_v3_v7) silently
    # trained a retrain on stale data — the 2026-06-05 readiness-audit footgun.
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Fresh exit training dataset dir (REQUIRED — no stale default).")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--bundle-name", type=str, default=None)
    parser.add_argument("--label-spill-dir", type=str, default=None,
                        help="Where to write labeled JSONL files (default: "
                             "<dataset-dir>/labeled_v6_<utc-stamp>/).")
    parser.add_argument("--keep-spill-dir", action="store_true",
                        help="Don't delete spill dir after training (default: delete).")
    parser.add_argument("--reuse-spill-dir", type=str, default=None,
                        help="Reuse an existing labeled-JSONL spill dir (skip labeling). "
                             "Useful for re-training without re-labeling.")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6,
                        help="DataLoader workers. 2026-05-26: DiskLabeledThinDataset is now "
                             "worker-safe (per-process file handle + pickle-safe state), so "
                             ">0 enables parallel prefetch (was the V3-train bottleneck; ~6.8x). "
                             "Set 0 only for debug.")
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-cutoff", type=str, default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--val-cutoff", type=str, default="2025-09-01T00:00:00+00:00")
    parser.add_argument("--exit-io-version", type=str, default="EXIT_IO_V7_VOLUME_DIPSTRUCT_M1L512",
                        help="V7 contract (131 feats = 91 V6 + 4 volume + 36 dip/struct, 2026-05-26 parity "
                             "wave). Default so a re-run trains on the full 131-feature matrix.")
    parser.add_argument("--early-stop-patience", type=int, default=4)

    parser.add_argument("--side-balance-weight", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--loss-weight-main", type=float, default=1.0)
    parser.add_argument("--loss-weight-profit", type=float, default=0.5)
    parser.add_argument("--loss-weight-family", type=float, default=0.3)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--label-chunksize", type=int, default=4,
                        help="imap_unordered chunksize for spawn labeler. Smaller "
                             "= less pickle buffer pressure. Default 4 (was 16 in v6 in-memory).")
    parser.add_argument("--smoke-trades-per-split", type=int, default=None,
                        help="If set, truncate each split to N trades for smoke testing.")
    # V12.2 multi-TF — MANDATORY (2026-05-26): the --enable-multi-tf on/off flag
    # was REMOVED. Multi-TF×5 (M5/M15/H1/H4/D1, V2 25-feat) is ALWAYS on; it must
    # never be possible to accidentally train a single-TF V3 exit transformer.
    # --m5-prebuilt-path is therefore required. (rule: multi_tf_always_mandatory)
    parser.add_argument("--m5-prebuilt-path", type=str, required=True,
                        help="REQUIRED: canonical_v3 M5 OHLC parquet for resampling the "
                             "M5/M15/H1/H4/D1 multi-TF features. Multi-TF is mandatory.")
    parser.add_argument("--multi-tf-seq-len", type=int, default=96,
                        help="V12.2: bars per TF (default 96).")
    parser.add_argument("--multi-tf-scale", type=float, default=0.5,
                        help="V12.2: multi-TF pool scale in cross-fusion.")
    parser.add_argument("--enable-pos-enc", action=argparse.BooleanOptionalAction, default=True,
                        help="Sinusoidal positional encoding on the base M1 window AND every "
                             "per-TF stream so the transformer uses temporal ORDER (default ON). "
                             "Without it the encoder is permutation-invariant — blind to bar order. "
                             "--no-enable-pos-enc restores the old order-blind behaviour.")
    parser.add_argument("--enable-dip-head", action=argparse.BooleanOptionalAction, default=True,
                        help="V3 dip-analysis head (6: K{12,48,144}×{recovery_p50,p10}, pinball loss "
                             "vs forward recovery) so the exit transformer represents dip-bottom/recovery "
                             "— don't exit at a dip bottom. Default ON for the dip-aware rebuild.")
    # New aux heads (2026-05-26) — mirror of V10 entry heads, exit-oriented. Default ON.
    parser.add_argument("--enable-timing-head", action=argparse.BooleanOptionalAction, default=True,
                        help="V3 timing head (6: time_to_recovery/worst_frac × K{12,48,144}).")
    parser.add_argument("--enable-tail-risk-head", action=argparse.BooleanOptionalAction, default=True,
                        help="V3 tail-risk head (3: p90 worst further-drawdown × K). Cut before tail.")
    parser.add_argument("--enable-vol-forecast-head", action=argparse.BooleanOptionalAction, default=True,
                        help="V3 vol-forecast head (3: forward realized vol bps × K).")
    parser.add_argument("--enable-forecast-head", action=argparse.BooleanOptionalAction, default=True,
                        help="V3 forecast head (3: forward pnl change × K). <0 → market turning against trade.")
    # V2 fast-train extras
    parser.add_argument("--init-from-state-dict", type=str, default=None,
                        help="V2 warm-start: load .pt state_dict into model after construction "
                             "(strict=False). Use scripts/warm_start_v3_v10_v2_from_v9.py to build. "
                             "For smoke-date filtering use --train-cutoff and --val-cutoff (existing).")
    parser.add_argument(
        "--vedtak", type=str, default=None,
        help="Explicit user decision id authorizing this retrain. REQUIRED "
             "(never auto-retrain). Any non-empty string from a deliberate human go.",
    )
    args = parser.parse_args()
    # NEVER auto-retrain — fail-closed unless an explicit --vedtak is passed.
    from gx1_guards.gates import require_retrain_vedtak, GateError
    try:
        require_retrain_vedtak(args.vedtak)
    except GateError as e:
        parser.error(str(e))
    # Multi-TF is MANDATORY — force on (toggle removed). m5-prebuilt-path is required
    # by argparse, so multi-TF can never be silently single-TF.
    args.enable_multi_tf = True

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"[{ACTION}] device={device} dataset={args.dataset_dir}", flush=True)

    # Stage 1: build ThinRecordDataset WITHOUT loading records into memory.
    base = ThinRecordDataset(
        args.dataset_dir,
        load_records_eagerly=False,
        build_offset_index=False,
    )
    records_path = base.records_path
    if records_path is None:
        raise RuntimeError(f"[{ACTION}] base.records_path is None — dataset corrupted")
    print(
        f"[{ACTION}] base loaded (mmap only)  matrix={base.matrix.shape}  "
        f"input_dim={base.input_dim} window_len={base.window_len}",
        flush=True,
    )

    # Stage 2: split UIDs by timestamp (one streaming pass).
    print(f"[{ACTION}] splitting trades by timestamp (one pass over records.jsonl)...", flush=True)
    t0 = time.time()
    train_uids, val_uids, test_uids = split_uids_by_time(
        records_path, train_cutoff=args.train_cutoff, val_cutoff=args.val_cutoff,
    )
    n_train_recs = sum(len(v) for v in train_uids.values())
    n_val_recs = sum(len(v) for v in val_uids.values())
    n_test_recs = sum(len(v) for v in test_uids.values())
    print(
        f"[{ACTION}] split done in {time.time()-t0:.1f}s: "
        f"train={len(train_uids):,} trades / {n_train_recs:,} records, "
        f"val={len(val_uids):,} / {n_val_recs:,}, "
        f"test={len(test_uids):,} / {n_test_recs:,}",
        flush=True,
    )

    if args.smoke_trades_per_split is not None:
        N = args.smoke_trades_per_split
        train_uids = dict(list(train_uids.items())[:N])
        val_uids = dict(list(val_uids.items())[:N])
        test_uids = dict(list(test_uids.items())[:N])
        print(
            f"[{ACTION}] SMOKE: truncated to {N} trades per split "
            f"(train={len(train_uids)} val={len(val_uids)} test={len(test_uids)})",
            flush=True,
        )

    # Stage 3: side-balance weights (computed without loading records).
    side_weights: Optional[Dict[str, float]] = None
    if args.side_balance_weight:
        sw, n_long, n_short = compute_side_weights_from_offsets(records_path, train_uids)
        side_weights = sw
        print(
            f"[{ACTION}] side balance: long_w={sw['long']:.3f} short_w={sw['short']:.3f} "
            f"(n_long_recs={n_long:,} n_short_recs={n_short:,})",
            flush=True,
        )

    # Stage 4: prepare label-spill dir.
    if args.reuse_spill_dir is not None:
        spill_dir = Path(args.reuse_spill_dir)
        if not spill_dir.exists():
            raise FileNotFoundError(f"[{ACTION}] --reuse-spill-dir {spill_dir} does not exist")
        print(f"[{ACTION}] REUSING existing spill dir: {spill_dir}", flush=True)
        # Skip labeling; rebuild offset maps by re-streaming each labeled JSONL.
        train_jsonl, val_jsonl, test_jsonl = (
            spill_dir / "train_labeled.jsonl",
            spill_dir / "val_labeled.jsonl",
            spill_dir / "test_labeled.jsonl",
        )
        train_offsets = _scan_jsonl_offsets(train_jsonl)
        val_offsets = _scan_jsonl_offsets(val_jsonl)
        test_offsets = _scan_jsonl_offsets(test_jsonl)
    else:
        spill_root = (
            Path(args.label_spill_dir)
            if args.label_spill_dir
            else Path(args.dataset_dir) / f"labeled_v6_{_utc_stamp()}"
        )
        spill_root.mkdir(parents=True, exist_ok=True)
        print(f"[{ACTION}] spill dir: {spill_root}", flush=True)

        # 2026-05-26: default raised 4→14 — labeling is CPU-bound, host has 18 cores;
        # the labeler is ~26M per-bar evals so more workers ≈ linear until ~14.
        # Set EXIT_LABEL_WORKERS=1 to fall back to serial (debug/low-RAM hosts).
        n_workers = int(os.environ.get("EXIT_LABEL_WORKERS", str(min(14, (os.cpu_count() or 8)))))
        chunksize = int(args.label_chunksize)
        print(f"[{ACTION}] EXIT_LABEL_WORKERS={n_workers} chunksize={chunksize}", flush=True)

        train_jsonl, _, train_offsets = _label_split(
            "train", Path(args.dataset_dir), train_uids,
            spill_dir=spill_root, n_workers=n_workers, chunksize=chunksize,
        )
        val_jsonl, _, val_offsets = _label_split(
            "val", Path(args.dataset_dir), val_uids,
            spill_dir=spill_root, n_workers=n_workers, chunksize=chunksize,
        )
        test_jsonl, _, test_offsets = _label_split(
            "test", Path(args.dataset_dir), test_uids,
            spill_dir=spill_root, n_workers=n_workers, chunksize=chunksize,
        )
        spill_dir = spill_root

    # Stage 5: pos-rate (informational; auto pos_weight not used in v6 — focal handles imbalance).
    train_pos = scan_labeled_pos_rate(train_jsonl, train_offsets[:50000])  # bound the scan
    print(f"[{ACTION}] train should_exit pos_rate (sampled): {train_pos:.4f}", flush=True)

    # V12.2: pre-build multi-TF feature tables (M5/M15/H1/H4/D1) from canonical_v3
    # M5 prebuilt. Resample once, cache in memory, share across train/val/test datasets.
    mtf_feats: Optional[Dict[str, Any]] = None
    mtf_shift: Optional[Dict[str, Any]] = None
    v3_v2_mode: bool = False
    _mtf_feature_count_actual: int = 0
    if args.enable_multi_tf:
        # MANDATORY V2: 5 TFs × 25 feats. V1 (17-feat) path + GX1_V3_MULTI_TF_V2
        # env-gate removed 2026-05-26 (rule: multi_tf_always_mandatory).
        v3_v2_mode = True
        from gx1.features.htf_features import (
            build_multi_tf_per_bar_features,
            build_multi_tf_per_bar_features_v2,
            MULTI_TF_SHIFT, MULTI_TF_FEATURE_COUNT, MULTI_TF_FEATURE_COUNT_V2,
        )
        import pandas as _pd
        print(f"[{ACTION}] V12.2: pre-building multi-TF features from {args.m5_prebuilt_path} (v2={v3_v2_mode})", flush=True)
        load_cols = ["time", "open", "high", "low", "close"]
        if v3_v2_mode:
            import pyarrow.parquet as _pq
            if "volume" in _pq.ParquetFile(args.m5_prebuilt_path).schema_arrow.names:
                load_cols.append("volume")
        m5 = _pd.read_parquet(args.m5_prebuilt_path, columns=load_cols)
        m5["time"] = _pd.to_datetime(m5["time"], utc=True)
        m5 = m5.set_index("time").sort_index()
        for c in ("open", "high", "low", "close"):
            m5[c] = m5[c].astype(np.float32)
        if "volume" in m5.columns:
            m5["volume"] = m5["volume"].astype(np.float32)
        if v3_v2_mode:
            mtf_feats = build_multi_tf_per_bar_features_v2(m5)
        else:
            mtf_feats = build_multi_tf_per_bar_features(m5)
        mtf_shift = MULTI_TF_SHIFT
        del m5
        import gc; gc.collect()
        _mtf_feature_count_actual = int(MULTI_TF_FEATURE_COUNT_V2 if v3_v2_mode else MULTI_TF_FEATURE_COUNT)
        for tf, df in mtf_feats.items():
            print(f"[{ACTION}]   {tf}: {len(df):,} bars × {df.shape[1]} feats", flush=True)

    # Stage 6: build datasets.
    _mtf_kwargs = (
        dict(multi_tf_feats=mtf_feats, multi_tf_shift=mtf_shift,
              multi_tf_seq_len=int(args.multi_tf_seq_len))
        if args.enable_multi_tf else {}
    )
    train_ds = DiskMultiTaskThinDataset(
        base, labeled_jsonl=train_jsonl, record_offsets=train_offsets,
        side_weights=side_weights,
        **_mtf_kwargs,
    )
    val_ds = DiskMultiTaskThinDataset(
        base, labeled_jsonl=val_jsonl, record_offsets=val_offsets,
        side_weights=None,  # eval doesn't need side balance
        **_mtf_kwargs,
    )
    test_ds = DiskMultiTaskThinDataset(
        base, labeled_jsonl=test_jsonl, record_offsets=test_offsets,
        side_weights=None,
        **_mtf_kwargs,
    )
    print(
        f"[{ACTION}] datasets ready: train={len(train_ds):,} val={len(val_ds):,} "
        f"test={len(test_ds):,}",
        flush=True,
    )

    # FAST_TRAIN: global TF32 / bf16 / compile toggles, plus DataLoader prefetch bump.
    try:
        from gx1.utils.fast_train import apply_global_speedups, loader_kwargs as _ft_loader_kwargs
        _ft_report = apply_global_speedups()
        print(f"[{ACTION}] FAST_TRAIN: {_ft_report}", flush=True)
    except Exception as _e:
        print(f"[{ACTION}] FAST_TRAIN init failed: {_e!r}", flush=True)
        _ft_loader_kwargs = None
    if _ft_loader_kwargs is not None:
        _dl_kwargs = _ft_loader_kwargs(args.num_workers, use_cuda=(device.type == "cuda"))
    else:
        _dl_kwargs = dict(num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=(args.num_workers > 0),
                          prefetch_factor=(2 if args.num_workers > 0 else None))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **_dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **_dl_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, **_dl_kwargs,
    )

    # Stage 7: model.
    _model_mtf_kwargs = {}
    if args.enable_multi_tf:
        # Use the dim that was actually built (V1=17, V2=25) — not the V1 constant.
        _mtf_dim = int(_mtf_feature_count_actual)
        _model_mtf_kwargs = dict(
            enable_multi_tf=True,
            m5_seq_dim=_mtf_dim, m15_seq_dim=_mtf_dim,
            h1_seq_dim=_mtf_dim, h4_seq_dim=_mtf_dim, d1_seq_dim=_mtf_dim,
            m5_seq_len=int(args.multi_tf_seq_len), m15_seq_len=int(args.multi_tf_seq_len),
            h1_seq_len=int(args.multi_tf_seq_len), h4_seq_len=int(args.multi_tf_seq_len),
            d1_seq_len=int(args.multi_tf_seq_len),
            multi_tf_scale=float(args.multi_tf_scale),
        )
        _v2_tag = " (V2)" if v3_v2_mode else " (V1)"
        print(f"[{ACTION}] V12.2 model{_v2_tag}: multi-TF M5/M15/H1/H4/D1 "
              f"dim={_mtf_dim} len={args.multi_tf_seq_len} scale={args.multi_tf_scale}", flush=True)
    model = ExitTransformerV0(
        input_dim=int(base.input_dim), window_len=int(base.window_len),
        d_model=int(args.d_model), n_heads=int(args.n_heads), n_layers=int(args.n_layers),
        dropout=float(args.dropout),
        enable_pos_enc=bool(args.enable_pos_enc),
        enable_dip_head=bool(args.enable_dip_head),
        enable_timing_head=bool(args.enable_timing_head),
        enable_tail_risk_head=bool(args.enable_tail_risk_head),
        enable_vol_forecast_head=bool(args.enable_vol_forecast_head),
        enable_forecast_head=bool(args.enable_forecast_head),
        **_model_mtf_kwargs,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{ACTION}] model: ExitTransformerV0 d_model={args.d_model} "
        f"n_layers={args.n_layers} params={n_params:,}",
        flush=True,
    )

    # V2 warm-start: load state_dict BEFORE torch.compile (avoids _orig_mod prefix issue).
    if args.init_from_state_dict:
        _isd_path = Path(args.init_from_state_dict)
        if not _isd_path.is_file():
            raise FileNotFoundError(f"[INIT_STATE_DICT_MISSING] {_isd_path}")
        print(f"[{ACTION}] [WARM_START] loading: {_isd_path}", flush=True)
        _isd = torch.load(_isd_path, map_location="cpu", weights_only=True)
        _isd = {k.removeprefix("_orig_mod."): v for k, v in _isd.items()}
        _ld = model.load_state_dict(_isd, strict=False)
        print(
            f"[{ACTION}] [WARM_START] loaded {len(_isd)} keys. "
            f"missing={len(_ld.missing_keys)} unexpected={len(_ld.unexpected_keys)}",
            flush=True,
        )

    # torch.compile via either --torch-compile flag OR GX1_FAST_TRAIN env.
    _compile_from_env = False
    try:
        from gx1.utils.fast_train import compile_enabled as _ft_compile_enabled
        _compile_from_env = _ft_compile_enabled()
    except Exception:
        pass
    if args.torch_compile or _compile_from_env:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[{ACTION}] torch.compile applied (source={'cli' if args.torch_compile else 'fast_train_env'})", flush=True)
        except Exception as e:
            print(f"[{ACTION}] torch.compile failed: {e}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    scheduler = make_cosine_warmup_scheduler(
        optimizer, total_steps=total_steps, warmup_frac=args.warmup_frac,
    )
    print(
        f"[{ACTION}] LR schedule: cosine + {int(args.warmup_frac*100)}% warmup "
        f"({total_steps} total steps)",
        flush=True,
    )

    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)
    if ema is not None:
        print(f"[{ACTION}] EMA enabled (decay={args.ema_decay})", flush=True)

    use_amp = bool(args.mixed_precision and device.type == "cuda")
    if use_amp:
        print(f"[{ACTION}] mixed precision (bfloat16) enabled", flush=True)

    loss_weights = {
        "main": args.loss_weight_main,
        "profit": args.loss_weight_profit,
        "family": args.loss_weight_family,
    }
    print(
        f"[{ACTION}] loss weights: {loss_weights}  focal_gamma={args.focal_gamma}  "
        f"label_smooth={args.label_smoothing}",
        flush=True,
    )

    # ── Bundle dir + config written BEFORE the loop so the best weights are
    #    checkpointed to disk on every improvement. Fixes failure-mode #8: the
    #    old code held best_state only in RAM and saved at natural end, so any
    #    kill/reboot mid-training lost the whole run (burned 7 BASE76 runs). ──
    bundle_name = args.bundle_name or f"EXIT_V6_DISK__BIDIR_2026Q2_{_utc_stamp()}"
    bundle_dir = Path(args.out_dir) / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)
    state_path = bundle_dir / "exit_transformer_v0.pt"
    contract = get_exit_io_contract(args.exit_io_version)
    _bundle_mtf_dim = int(_mtf_feature_count_actual) if args.enable_multi_tf else 0
    _bundle_v2 = bool(args.enable_multi_tf and v3_v2_mode)
    config = {
        "input_dim": int(base.input_dim), "window_len": int(base.window_len),
        "d_model": int(args.d_model), "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads), "dropout": float(args.dropout),
        "enable_pos_enc": bool(args.enable_pos_enc),
        "enable_dip_head": bool(args.enable_dip_head),
        "enable_timing_head": bool(args.enable_timing_head),
        "enable_tail_risk_head": bool(args.enable_tail_risk_head),
        "enable_vol_forecast_head": bool(args.enable_vol_forecast_head),
        "enable_forecast_head": bool(args.enable_forecast_head),
        "exit_ml_io_version": str(contract["io_version"]),
        "feature_names_hash": str(contract["feature_hash"]),
        "feature_count": int(contract["feature_count"]),
        "multi_tf": {
            "enabled": bool(args.enable_multi_tf),
            "v2_mode": _bundle_v2,
            "m5_seq_dim": _bundle_mtf_dim, "m15_seq_dim": _bundle_mtf_dim,
            "h1_seq_dim": _bundle_mtf_dim, "h4_seq_dim": _bundle_mtf_dim,
            "d1_seq_dim": _bundle_mtf_dim,
            "m5_seq_len": int(args.multi_tf_seq_len), "m15_seq_len": int(args.multi_tf_seq_len),
            "h1_seq_len": int(args.multi_tf_seq_len), "h4_seq_len": int(args.multi_tf_seq_len),
            "d1_seq_len": int(args.multi_tf_seq_len),
            "feature_contract": (
                "MULTI_TF_PER_BAR_V2" if _bundle_v2
                else ("MULTI_TF_PER_BAR_V1" if args.enable_multi_tf else None)
            ),
            "multi_tf_scale": float(args.multi_tf_scale),
        },
    }
    (bundle_dir / "transformer_config.json").write_text(json.dumps(config, indent=2))

    def _persist_best(state: Dict[str, torch.Tensor]) -> str:
        """Atomically write best weights to disk; returns sha256 (fix #8)."""
        tmp = state_path.with_suffix(".pt.tmp")
        torch.save({k: v for k, v in state.items()}, tmp)
        os.replace(tmp, state_path)
        return hashlib.sha256(state_path.read_bytes()).hexdigest()

    # Stage 8: training loop.
    best_val_loss = math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    state_sha = ""
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []
    for epoch in range(args.epochs):
        print(f"[{ACTION}] === epoch {epoch+1}/{args.epochs} ===", flush=True)
        train_stats = train_one_epoch_multitask(
            model, train_loader, optimizer, scheduler, device,
            focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
            loss_weights=loss_weights, ema=ema, use_amp=use_amp,
        )
        eval_model = copy.deepcopy(model) if ema is not None else model
        if ema is not None:
            ema.apply_to(eval_model)
        val_stats = evaluate_multitask(
            eval_model, val_loader, device,
            focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
            loss_weights=loss_weights, use_amp=use_amp,
        )
        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})
        print(
            f"[{ACTION}] epoch {epoch+1}: train_main={train_stats['train_loss_main']:.4f} "
            f"val_main={val_stats['loss_main']:.4f} val_total={val_stats['loss_total']:.4f} "
            f"val_acc={val_stats['acc_at_05']:.4f} "
            f"PR_AUC={val_stats.get('pr_auc', float('nan')):.4f} "
            f"ROC_AUC={val_stats.get('roc_auc', float('nan')):.4f} "
            f"prec={val_stats.get('precision_at_05', float('nan')):.3f} "
            f"recall={val_stats.get('recall_at_05', float('nan')):.3f} "
            f"({train_stats['elapsed_s']:.1f}s)",
            flush=True,
        )
        if val_stats["loss_total"] < best_val_loss - 1e-4:
            best_val_loss = val_stats["loss_total"]
            _eval_to_save = eval_model._orig_mod if hasattr(eval_model, "_orig_mod") else eval_model
            best_state = {k: v.detach().cpu().clone() for k, v in _eval_to_save.state_dict().items()}
            epochs_no_improve = 0
            state_sha = _persist_best(best_state)  # checkpoint to disk immediately (fix #8)
            print(f"[{ACTION}]   new best val_loss={best_val_loss:.4f} — checkpointed to disk", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(
                    f"[{ACTION}] early stop after {args.early_stop_patience} "
                    "non-improving epochs",
                    flush=True,
                )
                break

    if best_state is not None:
        # best_state was saved from the UNCOMPILED module (_orig_mod, non-prefixed
        # keys — see _eval_to_save above). Load into the underlying module so a
        # torch.compile'd `model` (OptimizedModule, expects _orig_mod. prefix)
        # doesn't fail with a key-prefix mismatch. (GX1_FAST_TRAIN=1 path.)
        _load_target = model._orig_mod if hasattr(model, "_orig_mod") else model
        _load_target.load_state_dict(best_state)

    final_eval_model = copy.deepcopy(model) if ema is not None else model
    if ema is not None:
        ema.apply_to(final_eval_model)
    test_stats = evaluate_multitask(
        final_eval_model, test_loader, device,
        focal_gamma=args.focal_gamma, label_smoothing=args.label_smoothing,
        loss_weights=loss_weights, use_amp=use_amp,
    )
    print(
        f"[{ACTION}] test: loss_main={test_stats['loss_main']:.4f} "
        f"acc={test_stats['acc_at_05']:.4f} n={test_stats['n']}",
        flush=True,
    )

    # Stage 9: weights + config already on disk (checkpointed at each best, fix #8).
    # If the run never improved (best_state is None), persist final weights now so
    # the bundle is never empty.
    if best_state is None:
        _final = model._orig_mod if hasattr(model, "_orig_mod") else model
        best_state = {k: v.detach().cpu().clone() for k, v in _final.state_dict().items()}
        state_sha = _persist_best(best_state)

    manifest = {
        "action_v1": ACTION,
        "bundle_name": bundle_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(args.dataset_dir),
        "label_spill_dir": str(spill_dir),
        "exit_io_version": str(contract["io_version"]),
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "d_model": args.d_model, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "dropout": args.dropout, "seed": args.seed,
            "focal_gamma": args.focal_gamma, "label_smoothing": args.label_smoothing,
            "warmup_frac": args.warmup_frac, "ema_decay": args.ema_decay if ema is not None else None,
            "mixed_precision": use_amp, "torch_compile": args.torch_compile,
            "loss_weights": loss_weights, "side_balance_weight": args.side_balance_weight,
            "side_weights_long": (side_weights or {}).get("long"),
            "side_weights_short": (side_weights or {}).get("short"),
            "label_workers": int(os.environ.get("EXIT_LABEL_WORKERS", "4")),
            "label_chunksize": args.label_chunksize,
            "train_cutoff": args.train_cutoff, "val_cutoff": args.val_cutoff,
        },
        "best_val_loss": float(best_val_loss),
        "test_metrics": test_stats,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "n_train_trades": len(train_uids),
        "n_val_trades": len(val_uids),
        "n_test_trades": len(test_uids),
        "train_pos_rate_sampled": float(train_pos),
        "model_state_dict_sha256": state_sha,
        "history": history,
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    print(f"[{ACTION}] bundle saved at {bundle_dir}", flush=True)

    # Stage 10: spill-dir cleanup.
    if not args.keep_spill_dir and args.reuse_spill_dir is None:
        try:
            for p in spill_dir.glob("*"):
                p.unlink()
            spill_dir.rmdir()
            print(f"[{ACTION}] cleaned up spill dir: {spill_dir}", flush=True)
        except Exception as e:
            print(f"[{ACTION}] spill-dir cleanup warning: {e}", flush=True)

    print(json.dumps(manifest, indent=2, default=float))


def _scan_jsonl_offsets(path: Path) -> List:
    """Rebuild flat record offsets from an existing labeled JSONL (for --reuse-spill-dir)."""
    offsets: List = []
    with path.open("rb") as f:
        byte_off = 0
        for line in f:
            ll = len(line)
            if line.strip():
                offsets.append((byte_off, ll))
            byte_off += ll
    return offsets


if __name__ == "__main__":
    main()
