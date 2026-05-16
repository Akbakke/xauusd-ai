#!/usr/bin/env python3
"""Phase B Stage 3a — V10 q_head fine-tune via Entry-IQL Q-target distillation.

Strategy: keep V12.2 v_FIXED backbone FROZEN; train only the new `q_head` layer
(zero-init in patched model) to mirror Entry-IQL's q_per_action_v1 distribution.

Inputs:
  - V12.2 v_FIXED bundle (entry_v10/ENTRY_V10_MULTI_TF_v2_FIXED_*) — warmstart
  - V10 training dataset (v10_v3_6yr_dataset__HOLD_03B_*.parquet)
  - Entry-IQL decisions.parquet (q_skip_v1, q_take_long_v1, q_take_short_v1)
  - canonical_v3 M5 prebuilt (for multi-TF cache, same as training)

Loss: KL(softmax(q_per_action / T), softmax(iql_q_target / T)) * T²
Default T=3.0, batch=256, lr=3e-4, epochs=3.

Output bundle: ENTRY_V10_MULTI_TF_v2_DISTILLED_<ts>/
  - model_state_dict.pt   (backbone + new q_head)
  - bundle_metadata.json  (multi_tf.enabled=true, enable_q_head=true)
  - distill_summary.json  (loss curve, n_rows_with_q_target, etc.)

Usage:
  PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 -u \\
    gx1/scripts/distill_v1_v10_q_head_finetune.py \\
    --decisions /home/andre2/GX1_DATA/reports/truth_e2e_sanity/V12_2_REPORTS_*/ENTRY_IQL_INFERENCE_FOR_V12_*/decisions.parquet
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path("/home/andre2/src/GX1_ENGINE")
sys.path.insert(0, str(REPO))

from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    EntryV10CtxDataset, _multi_tf_kwargs_from_batch,
    SEQ_SIGNAL_DIM, SNAP_SIGNAL_DIM,
)


DEFAULT_V_FIXED_BUNDLE = Path(
    "/home/andre2/GX1_DATA/models/models/entry_v10_ctx/"
    "ENTRY_V10_MULTI_TF_v2_FIXED_20260513T214805Z"
)
DEFAULT_TRAIN_PARQUET = Path(
    "/home/andre2/GX1_DATA/data/training/entry_v10_ctx_v3_dataset_6yr/"
    "v10_v3_6yr_dataset__HOLD_03B_train.parquet"
)
DEFAULT_VAL_PARQUET = Path(
    "/home/andre2/GX1_DATA/data/training/entry_v10_ctx_v3_dataset_6yr/"
    "v10_v3_6yr_dataset__HOLD_03B_val.parquet"
)
DEFAULT_CANONICAL_M5 = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)
OUTPUT_PARENT = Path("/home/andre2/GX1_DATA/models/models/entry_v10_ctx")


class QTargetWrapper(Dataset):
    """Wraps EntryV10CtxDataset; adds iql_q_target tensor per row.

    Decisions.parquet has q_skip_v1, q_take_long_v1, q_take_short_v1 per
    decision_ts_utc. We pre-compute a numpy array of Q-targets aligned with
    the underlying dataset's row order, then return it alongside the batch.
    """

    def __init__(self, base: EntryV10CtxDataset, q_targets: np.ndarray):
        assert len(q_targets) == len(base), f"q_targets {len(q_targets)} != base {len(base)}"
        self.base = base
        self.q_targets = q_targets

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        batch = self.base[i]
        batch["iql_q_target"] = torch.tensor(self.q_targets[i], dtype=torch.float32)
        return batch


def build_q_target_array(parquet_path: Path, decisions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, q_targets) where mask[i]=True iff row i has Q-target match.

    Join is on the row's 'time' against decisions.decision_ts_utc.
    """
    df_times = pd.read_parquet(parquet_path, columns=["time"])
    # Both UTC; coerce to ns-precision datetime64 for set lookup
    df_times["time"] = pd.to_datetime(df_times["time"], utc=True).astype("datetime64[ns, UTC]")
    decisions = decisions.copy()
    decisions["decision_ts_utc"] = pd.to_datetime(decisions["decision_ts_utc"], utc=True).astype("datetime64[ns, UTC]")
    # Build dict: ts_utc → (q_skip, q_long, q_short)
    q_map: dict[pd.Timestamp, tuple[float, float, float]] = {}
    for ts, qs, ql, qsh in zip(
        decisions["decision_ts_utc"].to_numpy(),
        decisions["q_skip_v1"].to_numpy(),
        decisions["q_take_long_v1"].to_numpy(),
        decisions["q_take_short_v1"].to_numpy(),
    ):
        q_map[ts] = (float(qs), float(ql), float(qsh))
    q_targets = np.zeros((len(df_times), 3), dtype=np.float32)
    mask = np.zeros(len(df_times), dtype=bool)
    for i, t in enumerate(df_times["time"].to_numpy()):
        hit = q_map.get(t)
        if hit is not None:
            q_targets[i] = hit
            mask[i] = True
    return mask, q_targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v-fixed-bundle", type=Path, default=DEFAULT_V_FIXED_BUNDLE)
    ap.add_argument("--train-parquet", type=Path, default=DEFAULT_TRAIN_PARQUET)
    ap.add_argument("--val-parquet",   type=Path, default=DEFAULT_VAL_PARQUET)
    ap.add_argument("--canonical-m5",  type=Path, default=DEFAULT_CANONICAL_M5)
    ap.add_argument("--decisions",     type=Path, required=True,
                    help="V12.2 Entry-IQL decisions.parquet (with q_skip_v1, q_take_long_v1, q_take_short_v1)")
    ap.add_argument("--out-bundle-name", type=str, default=None,
                    help="Default: ENTRY_V10_MULTI_TF_v2_DISTILLED_<ts>")
    ap.add_argument("--epochs",     type=int,   default=3)
    ap.add_argument("--batch-size", type=int,   default=256)
    ap.add_argument("--lr",         type=float, default=3e-4)
    ap.add_argument("--temperature", type=float, default=3.0,
                    help="KL distillation temperature (T). Higher = softer targets.")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--max-train-rows", type=int, default=0,
                    help="Cap training rows for quick experiments (0 = no cap)")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"[DISTILL_V10_QHEAD] device={device}", flush=True)

    # ── Load decisions + build Q-target arrays ──────────────────────
    print(f"[1/5] Loading decisions: {args.decisions}", flush=True)
    decisions = pd.read_parquet(args.decisions,
                                columns=["decision_ts_utc","q_skip_v1","q_take_long_v1","q_take_short_v1"])
    print(f"  decisions rows = {len(decisions):,}", flush=True)

    print(f"[2/5] Joining train rows to Q-targets (time → decision_ts_utc)...", flush=True)
    train_mask, train_qs = build_q_target_array(args.train_parquet, decisions)
    val_mask,   val_qs   = build_q_target_array(args.val_parquet,   decisions)
    print(f"  train rows w/ Q-target: {train_mask.sum():,} / {len(train_mask):,} "
          f"({100*train_mask.mean():.1f}%)", flush=True)
    print(f"  val   rows w/ Q-target: {val_mask.sum():,} / {len(val_mask):,} "
          f"({100*val_mask.mean():.1f}%)", flush=True)
    if train_mask.sum() == 0:
        raise RuntimeError("No training rows have Q-target match. Check decisions.parquet time format.")

    # ── Filter parquets to rows-with-Q-target and write tmp parquets ──
    # (EntryV10CtxDataset reads from a parquet path. Easiest: build filtered parquets.)
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="distill_v10_qhead_"))
    print(f"[3/5] Building filtered parquets in {tmp_dir}", flush=True)
    # Filter train: keep only rows with Q-target; keep q_targets in same order.
    train_df = pd.read_parquet(args.train_parquet)
    train_df = train_df[train_mask].reset_index(drop=True)
    train_qs = train_qs[train_mask]
    assert len(train_df) == len(train_qs), "alignment broken"
    if args.max_train_rows and len(train_df) > args.max_train_rows:
        # Sample indices once, apply to BOTH train_df and train_qs.
        rng = np.random.default_rng(1337)
        idx = rng.choice(len(train_df), size=args.max_train_rows, replace=False)
        idx = np.sort(idx)  # preserve chronological order
        train_df = train_df.iloc[idx].reset_index(drop=True)
        train_qs = train_qs[idx]
    train_pq = tmp_dir / "train.parquet"
    train_df.to_parquet(train_pq, index=False)
    val_df = pd.read_parquet(args.val_parquet)
    val_df = val_df[val_mask].reset_index(drop=True)
    val_qs = val_qs[val_mask]
    assert len(val_df) == len(val_qs), "val alignment broken"
    val_pq = tmp_dir / "val.parquet"
    val_df.to_parquet(val_pq, index=False)
    print(f"  train_filtered: {len(train_df):,} rows", flush=True)
    print(f"  val_filtered:   {len(val_df):,} rows", flush=True)
    del train_df, val_df  # free memory

    # ── Build Datasets (with multi-TF cache from canonical) ─────────
    print(f"[4/5] Building Datasets (multi-TF cache from canonical_v3)...", flush=True)
    base_train = EntryV10CtxDataset(
        parquet_path=train_pq, seq_len=96, allow_constant_labels=True,
        enable_multi_tf=True, m5_prebuilt_path=args.canonical_m5,
        multi_tf_seq_len=96,
    )
    base_val = EntryV10CtxDataset(
        parquet_path=val_pq, seq_len=96, allow_constant_labels=True,
        enable_multi_tf=True, m5_prebuilt_path=args.canonical_m5,
        multi_tf_seq_len=96,
    )
    train_ds = QTargetWrapper(base_train, train_qs)
    val_ds   = QTargetWrapper(base_val, val_qs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"  train batches per epoch: {len(train_loader)}  val batches: {len(val_loader)}", flush=True)

    # ── Build model with enable_q_head=True; warmstart from v_FIXED ──
    print(f"[5/5] Loading V12.2 v_FIXED + initialising q_head...", flush=True)
    meta_path = args.v_fixed_bundle / "bundle_metadata.json"
    meta = json.loads(meta_path.read_text())
    mtf = meta["multi_tf"]
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=int(meta["seq_input_dim"]),
        snap_input_dim=int(meta["snap_input_dim"]),
        seq_len=int(meta["seq_len"]),
        ctx_cont_dim=int(meta["ctx_cont_dim"]),
        ctx_cat_dim=int(meta["ctx_cat_dim"]),
        enable_multi_tf=True,
        m15_seq_dim=int(mtf["m15_seq_dim"]), h1_seq_dim=int(mtf["h1_seq_dim"]),
        h4_seq_dim=int(mtf["h4_seq_dim"]),   d1_seq_dim=int(mtf["d1_seq_dim"]),
        m15_seq_len=int(mtf["m15_seq_len"]), h1_seq_len=int(mtf["h1_seq_len"]),
        h4_seq_len=int(mtf["h4_seq_len"]),   d1_seq_len=int(mtf["d1_seq_len"]),
        enable_q_head=True,
    ).to(device)
    state = torch.load(args.v_fixed_bundle / "model_state_dict.pt", map_location=device)
    incompatible = model.load_state_dict(state, strict=False)
    # Expected: missing=q_head.{weight,bias} (zero-init), unexpected=[]
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys in state_dict: {incompatible.unexpected_keys}")
    expected_missing = {"q_head.weight", "q_head.bias"}
    actual_missing = set(incompatible.missing_keys)
    if actual_missing != expected_missing:
        raise RuntimeError(f"Missing keys mismatch: got {actual_missing}, expected {expected_missing}")
    print(f"  ✓ v_FIXED warmstart OK (q_head zero-init)", flush=True)

    # ── Freeze backbone ─────────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.q_head.parameters():
        p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params (q_head only): {n_trainable:,} / {n_total:,}", flush=True)

    optimizer = torch.optim.AdamW(model.q_head.parameters(), lr=args.lr, weight_decay=0.0)
    T = float(args.temperature)

    # ── Training loop ───────────────────────────────────────────────
    def run_epoch(loader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        total_kl, n_batches = 0.0, 0
        for batch in loader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            ctx_cat = batch["ctx_cat"].to(device).long()
            ctx_cont = batch["ctx_cont"].to(device).float()
            mtf_kwargs = _multi_tf_kwargs_from_batch(batch, device)
            q_target = batch["iql_q_target"].to(device)  # (B, 3) — [q_skip, q_long, q_short]
            with torch.set_grad_enabled(train_mode):
                out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf_kwargs)
                q_pred = out["q_per_action"]  # (B, 3)
                kl = F.kl_div(
                    F.log_softmax(q_pred / T, dim=-1),
                    F.softmax(q_target / T, dim=-1),
                    reduction="batchmean",
                ) * (T ** 2)
            if train_mode:
                optimizer.zero_grad()
                kl.backward()
                optimizer.step()
            total_kl += float(kl.item())
            n_batches += 1
        return {"kl_loss": total_kl / max(1, n_batches)}

    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        train_stats = run_epoch(train_loader, train_mode=True)
        val_stats = run_epoch(val_loader, train_mode=False)
        elapsed = time.time() - t0
        print(f"  epoch {epoch+1}/{args.epochs}  train_kl={train_stats['kl_loss']:.4f}  "
              f"val_kl={val_stats['kl_loss']:.4f}  elapsed={elapsed/60:.1f}min", flush=True)
        history.append({"epoch": epoch+1, "train": train_stats, "val": val_stats,
                        "elapsed_min": elapsed/60})

    # ── Save distilled bundle ───────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_name = args.out_bundle_name or f"ENTRY_V10_MULTI_TF_v2_DISTILLED_{ts}"
    out_bundle = OUTPUT_PARENT / out_name
    out_bundle.mkdir(parents=True, exist_ok=True)
    # Copy bundle_metadata.json + add enable_q_head=true
    new_meta = dict(meta)
    new_meta["enable_q_head"] = True
    new_meta["distill"] = {
        "warmstart_bundle": str(args.v_fixed_bundle),
        "decisions_parquet": str(args.decisions),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "temperature": T,
        "history": history,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": int(len(train_ds)),
        "n_val_rows": int(len(val_ds)),
    }
    # Copy peripheral files from v_FIXED into new bundle
    for fname in ("xgb_universal_multihead_v2_meta.json", "lock.json"):
        src = args.v_fixed_bundle / fname
        if src.exists():
            (out_bundle / fname).write_bytes(src.read_bytes())
    (out_bundle / "bundle_metadata.json").write_text(json.dumps(new_meta, indent=2))
    torch.save(model.state_dict(), out_bundle / "model_state_dict.pt")
    (out_bundle / "distill_summary.json").write_text(json.dumps({
        "history": history, "final_val_kl": history[-1]["val"]["kl_loss"] if history else None,
    }, indent=2))
    print(f"\n[DISTILL_V10_QHEAD] DONE → {out_bundle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
