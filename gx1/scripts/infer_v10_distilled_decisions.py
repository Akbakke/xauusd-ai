#!/usr/bin/env python3
"""Run V10 distilled q_head over baseline Entry-IQL decisions.parquet.

Generates a parallel decisions.parquet where action_label_v1 / q_per_action
come from V10's distilled q_head (Phase 3a) instead of the Entry-IQL teacher.
Used as the --entry-iql-decisions input for Phase 6 distilled A/B comparison
against the baseline PLUS5-ensemble decisions.

Inputs:
  - V10 distilled bundle (state_dict has q_head.weight)
  - baseline decisions.parquet (provides {candidate_uid, decision_ts_utc} mapping)
  - V10 train + val parquets (provides V10 inputs aligned by 'time' = decision_ts)
  - canonical_v3 M5 prebuilt (multi-TF cache, same as training)

Output:
  decisions.parquet with cols
      candidate_uid, decision_ts_utc,
      q_skip_v1, q_take_long_v1, q_take_short_v1,
      iql_entry_advantage_over_skip_v1, iql_entry_action_v1, action_label_v1

Coverage caveat: only candidates whose decision_ts matches a row in the V10
train+val parquets get a distilled prediction. Other candidates are skipped
(typically 30-40% of the forward-outcome candidates). Phase 6 evaluates the
intersection.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path("/home/andre2/src/GX1_ENGINE")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    EntryV10CtxDataset, _multi_tf_kwargs_from_batch,
)


ACTION_LABELS_V1 = ["SKIP", "TAKE_LONG_NOW", "TAKE_SHORT_NOW"]

DEFAULT_V10_BUNDLE = Path(
    "/home/andre2/GX1_DATA/models/models/entry_v10_ctx/"
    "ENTRY_V10_V3PLUS_DISTILLED_20260521T132037Z"
)
DEFAULT_BASELINE_DECISIONS = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "IQL_ENTRY_Q_TARGETS_V3PLUS_PLUS5_ENS5_20260521T131431Z_LOCK/decisions.parquet"
)
DEFAULT_TRAIN_PARQUET = Path(
    "/home/andre2/GX1_DATA/data/training/entry_v10_ctx_v3plus_dataset_6yr/"
    "v10_v3plus_6yr_dataset__HOLD_03B_train.parquet"
)
DEFAULT_VAL_PARQUET = Path(
    "/home/andre2/GX1_DATA/data/training/entry_v10_ctx_v3plus_dataset_6yr/"
    "v10_v3plus_6yr_dataset__HOLD_03B_val.parquet"
)
DEFAULT_CANONICAL_M5 = Path(
    "/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
    "xauusd_m5_CANONICAL_V3_2020_2026.parquet"
)


def build_filtered_parquet(v10_pq: Path, ts_to_uid: dict, tmp_dir: Path, tag: str) -> tuple[Path, np.ndarray] | None:
    """Filter V10 parquet to rows whose 'time' is in ts_to_uid; return path + uid_array."""
    df_times = pd.read_parquet(v10_pq, columns=["time"])
    times_arr = pd.to_datetime(df_times["time"], utc=True).astype("datetime64[ns, UTC]").to_numpy()
    mask = np.array([t in ts_to_uid for t in times_arr], dtype=bool)
    if not mask.any():
        return None
    matched_uids = np.array([ts_to_uid[t] for t in times_arr[mask]], dtype=object)
    matched_times = times_arr[mask]
    print(f"  {tag}: {mask.sum():,} / {len(mask):,} rows match baseline decisions", flush=True)
    df = pd.read_parquet(v10_pq)
    df = df.loc[mask].reset_index(drop=True)
    out_pq = tmp_dir / f"v10_filtered_{tag}.parquet"
    df.to_parquet(out_pq, index=False)
    del df
    return out_pq, matched_uids, matched_times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v10-bundle",   type=Path, default=DEFAULT_V10_BUNDLE)
    ap.add_argument("--baseline-decisions", type=Path, default=DEFAULT_BASELINE_DECISIONS)
    ap.add_argument("--train-parquet", type=Path, default=DEFAULT_TRAIN_PARQUET)
    ap.add_argument("--val-parquet",  type=Path, default=DEFAULT_VAL_PARQUET)
    ap.add_argument("--canonical-m5", type=Path, default=DEFAULT_CANONICAL_M5)
    ap.add_argument("--out",          type=Path, required=True)
    ap.add_argument("--batch-size",   type=int, default=512)
    ap.add_argument("--device",       type=str, default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    print(f"[INFER_V10_DISTILLED] device={device}", flush=True)

    # ── Load distilled V10 (must have q_head.weight in state_dict) ──────
    print(f"[1/4] Loading V10 distilled: {args.v10_bundle.name}", flush=True)
    meta = json.loads((args.v10_bundle / "bundle_metadata.json").read_text())
    mtf = meta["multi_tf"]
    state = torch.load(args.v10_bundle / "model_state_dict.pt", map_location=device)
    aux_flags = dict(
        enable_tf_agreement_head="head_tf_agreement.weight" in state,
        enable_path_quality_variance_head="head_path_quality_log_var.weight" in state,
        enable_position_size_head="head_position_size.weight" in state,
        enable_hold_horizon_head="head_hold_horizon.weight" in state,
    )
    enable_q_head = "q_head.weight" in state
    if not enable_q_head:
        raise RuntimeError(f"{args.v10_bundle.name} has no q_head — not a distilled bundle")
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
        **aux_flags,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"  ✓ loaded; aux_heads {sum(aux_flags.values())}/4", flush=True)

    # ── Build candidate_uid → decision_ts map from baseline decisions ──
    print(f"[2/4] Reading baseline decisions: {args.baseline_decisions}", flush=True)
    base = pd.read_parquet(args.baseline_decisions, columns=["candidate_uid", "decision_ts_utc"])
    base["decision_ts_utc"] = pd.to_datetime(base["decision_ts_utc"], utc=True).astype("datetime64[ns, UTC]")
    n_base = len(base)
    print(f"  baseline candidates: {n_base:,}", flush=True)
    # WARNING: multiple candidates may share the same decision_ts. Take last
    # to match V10 distill script's join semantics.
    ts_to_uid: dict = {}
    for ts, uid in zip(base["decision_ts_utc"].to_numpy(), base["candidate_uid"].to_numpy()):
        ts_to_uid[ts] = uid

    # ── Filter V10 train + val parquets ─────────────────────────────────
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="infer_v10_distilled_"))
    print(f"[3/4] Building filtered V10 parquets in {tmp_dir}", flush=True)
    chunks = []
    for v10_pq, tag in ((args.train_parquet, "train"), (args.val_parquet, "val")):
        res = build_filtered_parquet(v10_pq, ts_to_uid, tmp_dir, tag)
        if res is not None:
            chunks.append(res)
    if not chunks:
        raise RuntimeError("No V10 rows matched any baseline candidate. Check time formats.")

    # ── Run inference + save decisions ──────────────────────────────────
    print(f"[4/4] Inference (batch={args.batch_size})", flush=True)
    out_rows = []
    t0 = time.time()
    for pq_path, uids, times in chunks:
        base_ds = EntryV10CtxDataset(
            parquet_path=pq_path, seq_len=96, allow_constant_labels=True,
            enable_multi_tf=True, m5_prebuilt_path=args.canonical_m5,
            multi_tf_seq_len=96,
        )
        loader = DataLoader(base_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        idx = 0
        for batch in loader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            ctx_cat = batch["ctx_cat"].to(device).long()
            ctx_cont = batch["ctx_cont"].to(device).float()
            mtf_kwargs = _multi_tf_kwargs_from_batch(batch, device)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(seq_x, snap_x, ctx_cat=ctx_cat, ctx_cont=ctx_cont, **mtf_kwargs)
                q = out["q_per_action"].float().cpu().numpy()  # (B, 3)
            for j in range(q.shape[0]):
                qs, ql, qsh = float(q[j, 0]), float(q[j, 1]), float(q[j, 2])
                a_id = int(np.argmax(q[j]))
                out_rows.append({
                    "candidate_uid": uids[idx],
                    "decision_ts_utc": pd.Timestamp(times[idx]),
                    "q_skip_v1": qs,
                    "q_take_long_v1": ql,
                    "q_take_short_v1": qsh,
                    "iql_entry_advantage_over_skip_v1": max(ql, qsh) - qs,
                    "iql_entry_action_v1": a_id,
                    "action_label_v1": ACTION_LABELS_V1[a_id],
                })
                idx += 1
        del base_ds, loader
    elapsed = time.time() - t0
    out_df = pd.DataFrame(out_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"  ✓ {len(out_df):,} distilled decisions written in {elapsed/60:.1f} min")
    print(f"  → {args.out}")

    # Action distribution
    print("\n  action distribution:")
    print(out_df["action_label_v1"].value_counts())
    print(f"\n  baseline coverage: {len(out_df)}/{n_base} = {100*len(out_df)/n_base:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
