#!/usr/bin/env python3
"""Standalone OOT evaluator for a trained V3 exit-transformer bundle.

Re-labels the TEST split (OOT) of a V7 exit dataset and runs evaluate_multitask
on a saved bundle to report the REAL discrimination verdict (PR-AUC / recall /
precision on the rare should_exit class) — the metrics raw loss + acc@0.5 hide.

MUST be run as a module (`python -m gx1.scripts.eval_v3_exit_oot ...`): the
disk labeler uses mp spawn, and a path-run script gets re-executed as __main__
in spawn children (cascade). As a package module, children import it by name.

Usage:
  python -m gx1.scripts.eval_v3_exit_oot \
      --dataset-dir <V7 dataset> --bundle <.../EXIT_V6_DISK__...> \
      --m5-prebuilt-path <canonical_v3 m5>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--m5-prebuilt-path", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--train-cutoff", default="2025-01-01T00:00:00+00:00")
    ap.add_argument("--val-cutoff", default="2025-09-01T00:00:00+00:00")
    ap.add_argument("--label-workers", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    from torch.utils.data import DataLoader
    from gx1.scripts.train_exit_v6_disk_thin import (
        ThinRecordDataset, split_uids_by_time, _label_split,
        DiskMultiTaskThinDataset, _utc_stamp,
    )
    from gx1.scripts.train_exit_v6_thin_records import evaluate_multitask
    from gx1.policy.exit_transformer_v0 import ExitTransformerV0
    from gx1.features.htf_features import build_multi_tf_per_bar_features_v2, MULTI_TF_SHIFT
    import pyarrow.parquet as pq

    DSET = Path(args.dataset_dir)
    BUN = Path(args.bundle)
    cfg = json.load(open(BUN / "transformer_config.json"))
    mtf = cfg["multi_tf"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = ThinRecordDataset(DSET, load_records_eagerly=False, build_offset_index=False)
    tr, va, te = split_uids_by_time(
        base.records_path, train_cutoff=args.train_cutoff, val_cutoff=args.val_cutoff,
    )
    uids = {"test": te, "val": va}[args.split]
    print(f"[VERDICT] split={args.split} trades={len(uids):,}", flush=True)
    spill = DSET / f"eval_spill_{_utc_stamp()}"
    spill.mkdir(parents=True, exist_ok=True)
    jsonl, _, offs = _label_split(
        args.split, DSET, uids, spill_dir=spill,
        n_workers=args.label_workers, chunksize=32,
    )

    cols = ["time", "open", "high", "low", "close"]
    if "volume" in pq.ParquetFile(args.m5_prebuilt_path).schema_arrow.names:
        cols.append("volume")
    m5 = pd.read_parquet(args.m5_prebuilt_path, columns=cols)
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.set_index("time").sort_index()
    for c in ("open", "high", "low", "close"):
        m5[c] = m5[c].astype(np.float32)
    if "volume" in m5.columns:
        m5["volume"] = m5["volume"].astype(np.float32)
    mtf_feats = build_multi_tf_per_bar_features_v2(m5)
    kw = dict(multi_tf_feats=mtf_feats, multi_tf_shift=MULTI_TF_SHIFT, multi_tf_seq_len=96)
    ds = DiskMultiTaskThinDataset(base, labeled_jsonl=jsonl, record_offsets=offs, side_weights=None, **kw)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=6, pin_memory=True)
    print(f"[VERDICT] records={len(ds):,}", flush=True)

    D = int(mtf["m5_seq_dim"])
    model = ExitTransformerV0(
        input_dim=cfg["input_dim"], window_len=cfg["window_len"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_layers=cfg["n_layers"], dropout=cfg["dropout"],
        enable_pos_enc=cfg["enable_pos_enc"], enable_dip_head=cfg["enable_dip_head"],
        enable_timing_head=cfg["enable_timing_head"], enable_tail_risk_head=cfg["enable_tail_risk_head"],
        enable_vol_forecast_head=cfg["enable_vol_forecast_head"], enable_forecast_head=cfg["enable_forecast_head"],
        enable_multi_tf=True, m5_seq_dim=D, m15_seq_dim=D, h1_seq_dim=D, h4_seq_dim=D, d1_seq_dim=D,
        m5_seq_len=96, m15_seq_len=96, h1_seq_len=96, h4_seq_len=96, d1_seq_len=96,
        multi_tf_scale=float(mtf["multi_tf_scale"]),
    ).to(dev)
    sd = torch.load(BUN / "exit_transformer_v0.pt", map_location="cpu", weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    ld = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"[VERDICT] loaded: missing={len(ld.missing_keys)} unexpected={len(ld.unexpected_keys)}", flush=True)

    st = evaluate_multitask(
        model, loader, dev, focal_gamma=2.0, label_smoothing=0.05,
        loss_weights={"main": 1.0, "profit": 0.5, "family": 0.3}, use_amp=True,
    )
    print(f"\n===== V3 EXIT OOT ({args.split}) VERDICT =====")
    for k in ("n", "pos_rate", "pred_mean", "acc_at_05", "precision_at_05",
              "recall_at_05", "f1_at_05", "roc_auc", "pr_auc"):
        print(f"  {k:16s} = {st.get(k)}")
    if st.get("pr_auc") and st.get("pos_rate"):
        print(f"  PR-AUC lift over base = {st['pr_auc'] / max(st['pos_rate'], 1e-9):.2f}x")


if __name__ == "__main__":
    main()
