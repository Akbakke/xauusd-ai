#!/usr/bin/env python3
"""Re-score forward-outcome dataset with V10 V2 prelim outputs.

Reads each per-week parquet in the existing forward-outcome dir, runs V10 V2
prelim forward on each candidate (matched by time to V10 training dataset for
pre-built nested cols), and writes a new dataset with V10-derived cols replaced.

V10-derived cols replaced (12):
  p_long, p_short, p_flat, p_hat
  tradable_prob, mfe_first_n_pred, path_quality_pred, bad_path_prob
  direction_logit_long, direction_logit_short, direction_logit_flat
  path_quality_std  (path_quality_log_var + tf_agreement_pred skipped — disabled in prelim)

Usage:
    python -m gx1.scripts.rescore_forward_outcome_with_v10_v2 \\
        --fwd-in   /path/to/CANDIDATE_FORWARD_OUTCOME_..._LOCK \\
        --fwd-out  /path/to/CANDIDATE_FORWARD_OUTCOME_..._LOCK_V10V2_RESCORED \\
        --v10-bundle /path/to/ENTRY_V10_V2_PRELIM_* \\
        --v10-training-dir /path/to/entry_v10_ctx_v3plus_dataset_6yr
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

V10_OUT_COLS = [
    "p_long", "p_short", "p_flat", "p_hat",
    "tradable_prob", "mfe_first_n_pred", "path_quality_pred", "bad_path_prob",
    "direction_logit_long", "direction_logit_short", "direction_logit_flat",
    "path_quality_std",
]


def _build_v10_outputs_per_time(
    v10_training_dir: Path,
    v10_bundle_dir: Path,
    m5_prebuilt_path: Path,
    device_str: str = "cuda",
    batch_size: int = 256,
) -> pd.DataFrame:
    """Run V10 V2 prelim forward on the entire V10 training+val+test dataset.

    Returns a DataFrame indexed by time with the 12 V10 output cols.
    """
    from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset
    from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle

    print(f"[LOAD_BUNDLE] {v10_bundle_dir.name}", flush=True)
    bundle = load_entry_v10_ctx_bundle(bundle_dir=v10_bundle_dir, device=device_str, xgb_models=None)
    model = bundle.transformer_model
    model.eval()
    device = torch.device(device_str)

    # Discover h4_seq_len + d1_seq_len from bundle metadata so dataset matches.
    mtf = bundle.metadata.get("multi_tf", {})
    h4_len = int(mtf.get("h4_seq_len", 96))
    d1_len = int(mtf.get("d1_seq_len", 96))
    print(f"[BUNDLE] multi_tf v2_mode={mtf.get('v2_mode')} h4_len={h4_len} d1_len={d1_len}", flush=True)

    out_rows: List[Dict] = []
    for split_name in ("train", "val", "test"):
        parquet_path = v10_training_dir / f"v10_v3plus_6yr_dataset__HOLD_03B_{split_name}.parquet"
        if not parquet_path.is_file():
            print(f"[SKIP] {split_name}: {parquet_path} not found", flush=True)
            continue
        print(f"[LOAD_DS] {split_name}: {parquet_path.name}", flush=True)
        ds = EntryV10CtxDataset(
            parquet_path=parquet_path,
            seq_len=96,
            allow_constant_labels=True,
            enable_multi_tf=True,
            m5_prebuilt_path=m5_prebuilt_path,
            multi_tf_seq_len=96,
            per_tf_seq_lens={"H4": h4_len, "D1": d1_len},
        )
        times = pd.to_datetime(ds.df["time"], utc=True).values
        n = len(ds)
        print(f"[INFER] {split_name}: {n:,} samples", flush=True)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))
        idx = 0
        t0 = time.time()
        with torch.no_grad():
            for batch in loader:
                seq_x = batch["seq_x"].to(device, non_blocking=True)
                snap_x = batch["snap_x"].to(device, non_blocking=True)
                ctx_cont = batch["ctx_cont"].to(device, non_blocking=True)
                ctx_cat = batch["ctx_cat"].to(device, non_blocking=True)
                kw = {}
                for k in ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1"):
                    if k in batch:
                        kw[k] = batch[k].to(device, non_blocking=True)
                out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **kw)
                logits = out["direction_logits"]
                probs = torch.softmax(logits, dim=1)
                p_long, p_short, p_flat = probs[:, 0], probs[:, 1], probs[:, 2]
                p_hat = probs.max(dim=1).values
                tradable_prob = torch.sigmoid(out["tradable_logit"]).squeeze(-1) if "tradable_logit" in out else torch.zeros_like(p_long)
                mfe_pred = out.get("mfe_first_n")
                if mfe_pred is not None and mfe_pred.dim() > 1:
                    mfe_pred = mfe_pred.squeeze(-1)
                path_pred = out.get("path_quality")
                if path_pred is not None and path_pred.dim() > 1:
                    path_pred = path_pred.squeeze(-1)
                bad_path_prob = torch.sigmoid(out["bad_path_logit"]).squeeze(-1) if "bad_path_logit" in out else torch.zeros_like(p_long)
                B = seq_x.shape[0]
                for j in range(B):
                    row = {
                        "time": pd.Timestamp(times[idx + j]),
                        "p_long_v2": float(p_long[j].item()),
                        "p_short_v2": float(p_short[j].item()),
                        "p_flat_v2": float(p_flat[j].item()),
                        "p_hat_v2": float(p_hat[j].item()),
                        "tradable_prob_v2": float(tradable_prob[j].item()),
                        "mfe_first_n_pred_v2": float(mfe_pred[j].item()) if mfe_pred is not None else 0.0,
                        "path_quality_pred_v2": float(path_pred[j].item()) if path_pred is not None else 0.0,
                        "bad_path_prob_v2": float(bad_path_prob[j].item()),
                        "direction_logit_long_v2": float(logits[j, 0].item()),
                        "direction_logit_short_v2": float(logits[j, 1].item()),
                        "direction_logit_flat_v2": float(logits[j, 2].item()),
                        "path_quality_std_v2": float(path_pred[j].std().item()) if path_pred is not None else 0.0,
                    }
                    out_rows.append(row)
                idx += B
                if idx % (batch_size * 50) < batch_size:
                    elapsed = time.time() - t0
                    rate = idx / max(1e-6, elapsed)
                    eta = (n - idx) / max(1, rate)
                    print(f"  [{split_name}] {idx:,}/{n:,} ({100*idx/n:.1f}%) rate={rate:.0f}/s eta={eta:.0f}s", flush=True)

    df = pd.DataFrame(out_rows)
    df = df.set_index("time").sort_index()
    print(f"[INFER_DONE] total rows: {len(df):,}", flush=True)
    return df


def _join_into_forward_outcome(
    fwd_in_dir: Path,
    fwd_out_dir: Path,
    v10_outputs: pd.DataFrame,
) -> None:
    """For each per-week parquet, join V10_v2 outputs by decision_ts_utc and write to new dir."""
    fwd_out_dir.mkdir(parents=True, exist_ok=True)
    per_week_in = fwd_in_dir / "per_week"
    per_week_out = fwd_out_dir / "per_week"
    per_week_out.mkdir(parents=True, exist_ok=True)
    week_paths = sorted(per_week_in.glob("*.parquet"))
    print(f"[JOIN] {len(week_paths)} per-week parquets to process", flush=True)

    v10_idx = v10_outputs.index  # DatetimeIndex
    v10_idx_int = v10_idx.values.astype("datetime64[ns]").astype(np.int64)
    v10_arr = v10_outputs.to_numpy(dtype=np.float32)
    v10_cols = list(v10_outputs.columns)

    n_matched = 0
    n_total = 0
    t0 = time.time()
    for i, wp in enumerate(week_paths):
        df = pd.read_parquet(wp)
        n_total += len(df)
        # Match by decision_ts_utc → time
        ts = pd.to_datetime(df["decision_ts_utc"], utc=True).values.astype("datetime64[ns]").astype(np.int64)
        # searchsorted to find indices in v10_idx_int
        right = np.searchsorted(v10_idx_int, ts, side="left")
        # Determine which are exact matches (ts[k] == v10_idx_int[right[k]])
        in_range = right < len(v10_idx_int)
        match_mask = np.zeros(len(ts), dtype=bool)
        match_mask[in_range] = (v10_idx_int[right[in_range]] == ts[in_range])
        n_matched += int(match_mask.sum())
        # Build new V10 cols
        new_cols = np.full((len(ts), len(v10_cols)), np.nan, dtype=np.float32)
        if match_mask.any():
            matched_idx = right[match_mask]
            new_cols[match_mask] = v10_arr[matched_idx]
        for j, col in enumerate(v10_cols):
            df[col] = new_cols[:, j]
        # Write
        out_path = per_week_out / wp.name
        df.to_parquet(out_path, index=False)
        if (i + 1) % 20 == 0 or i + 1 == len(week_paths):
            elapsed = time.time() - t0
            print(f"  [JOIN] {i+1}/{len(week_paths)} weeks, matched={n_matched:,}/{n_total:,} ({100*n_matched/max(1,n_total):.1f}%), elapsed={elapsed:.0f}s", flush=True)
    print(f"[JOIN_DONE] {len(week_paths)} weeks, total matched: {n_matched:,}/{n_total:,}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fwd-in", type=Path, required=True)
    p.add_argument("--fwd-out", type=Path, required=True)
    p.add_argument("--v10-bundle", type=Path, required=True)
    p.add_argument("--v10-training-dir", type=Path, required=True)
    p.add_argument("--m5-prebuilt", type=Path,
                   default=Path("/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/xauusd_m5_CANONICAL_V3_2020_2026.parquet"))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    print(f"[RESCORE_V10V2] fwd-in: {args.fwd_in}", flush=True)
    print(f"[RESCORE_V10V2] fwd-out: {args.fwd_out}", flush=True)
    print(f"[RESCORE_V10V2] device: {args.device}", flush=True)
    v10_outputs = _build_v10_outputs_per_time(
        args.v10_training_dir, args.v10_bundle, args.m5_prebuilt,
        device_str=args.device, batch_size=args.batch_size,
    )
    _join_into_forward_outcome(args.fwd_in, args.fwd_out, v10_outputs)
    print("[DONE]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
