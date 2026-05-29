#!/usr/bin/env python3
"""Post-hoc temperature calibration for V10 v2 entry transformer.

Fits a single scalar temperature T such that softmax(direction_logits / T)
minimizes negative log-likelihood (NLL) on a calibration set, plus reports
Expected Calibration Error (ECE) before and after.

Saves the temperature as `calibration_temperature.json` alongside the V10 bundle:
    {
      "temperature": 0.93,
      "ece_before": 0.072,
      "ece_after": 0.041,
      "nll_before": 1.043,
      "nll_after": 0.987,
      "calibration_set_size": 5000,
      "fitted_at_utc": "2026-05-04T..."
    }

At inference time, divide direction_logits by `temperature` before softmax to
get calibrated probabilities. This does NOT change any training; it only
tweaks how raw logits are converted to probabilities.

USAGE
    python -m gx1.scripts.calibrate_v10_v2_temperature \
        --bundle-dir /home/andre2/GX1_DATA/models/models/entry_v10_ctx/ENTRY_V10_CTX__RETRAIN_2026Q2_BIDIR_SMC_20260503T134214Z \
        --calibration-set-size 5000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute ECE: average gap between confidence and accuracy across bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_weight = mask.mean()
        ece += abs(bin_conf - bin_acc) * bin_weight
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Optimize T via L-BFGS to minimize NLL of softmax(logits/T)."""
    log_T = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        T = log_T.exp()
        loss = F.cross_entropy(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_T.exp().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="V10 v2 post-hoc temperature calibration")
    parser.add_argument("--bundle-dir", type=str, required=True)
    parser.add_argument("--calibration-set-size", type=int, default=5000)
    parser.add_argument("--prebuilt", type=str,
                        default="/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V2_PREBUILT/xauusd_m5_CANONICAL_V2_2020_2026.parquet")
    parser.add_argument("--xgb-bundle", type=str,
                        default="/home/andre2/GX1_DATA/models/models/xgb_universal_multihead_v4__BIDIR_RSI_SMC_PRUNED_20260503T103505Z")
    parser.add_argument("--xgb-feature-contract", type=str,
                        default=str(REPO_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_canonical_v2_v1.json"))
    parser.add_argument("--xgb-sanitizer-config", type=str,
                        default=str(REPO_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_canonical_v2_v1.json"))
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu recommended (don't compete with GPU training)")
    parser.add_argument("--lookahead-bars", type=int, default=12,
                        help="Bars ahead used for label derivation (matches XGB v4 training)")
    parser.add_argument("--threshold-bps", type=float, default=6.0,
                        help="Bps threshold for long/short label (matches XGB v4)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    meta = json.loads((bundle_dir / "bundle_metadata.json").read_text())
    print(f"[calibrate] V10 bundle: {bundle_dir.name}", flush=True)
    print(f"[calibrate] dims: seq={meta['seq_input_dim']} ctx_cont={meta['ctx_cont_dim']} seq_len={meta['seq_len']}",
          flush=True)

    # Load V10 model
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
    model = EntryV10CtxHybridTransformer(
        seq_input_dim=meta["seq_input_dim"], snap_input_dim=meta["snap_input_dim"],
        ctx_cont_dim=meta["ctx_cont_dim"], ctx_cat_dim=meta["ctx_cat_dim"], seq_len=meta["seq_len"],
    )
    state = torch.load(bundle_dir / "model_state_dict.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(state); model.eval(); model.to(args.device)

    # Load canonical_v2 + run XGB once on full set (cached)
    print(f"[calibrate] loading prebuilt + running XGB v4 inference...", flush=True)
    cv2 = pd.read_parquet(args.prebuilt)
    if "time" in cv2.columns and not isinstance(cv2.index, pd.DatetimeIndex):
        cv2["time"] = pd.to_datetime(cv2["time"], utc=True)
        cv2 = cv2.set_index("time")
    cv2 = cv2.sort_index()

    # Use the inference-batch's helpers to get bridge fields per M5 bar
    from gx1.scripts.materialize_inference_batch_candidates_v2_v1 import run_xgb_inference, build_v10_input_matrices
    bridge = run_xgb_inference(
        cv2, xgb_bundle=Path(args.xgb_bundle),
        feature_contract=Path(args.xgb_feature_contract),
        sanitizer_config=Path(args.xgb_sanitizer_config),
    )
    per_bar, ctx_cont, ctx_cat = build_v10_input_matrices(cv2, bridge)

    # Build labels: long if M5_close at t+lookahead > close[t] + threshold; short if <; else flat
    # canonical_v2 prebuilt is feature-only; load M5 tape to get raw close prices.
    n = len(cv2)
    print(f"[calibrate] loading M5 tape for close prices (canonical_v2 is feature-only)...", flush=True)
    M5_TAPE_ROOT = "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL"
    from gx1.scripts import materialize_build_candidate_forward_outcome_dataset_v1 as fwd_pipe
    m5 = fwd_pipe.load_m5_tape(Path(M5_TAPE_ROOT))
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.set_index("time").sort_index()
    # Mid-price = (bid_close + ask_close) / 2
    m5_close = (m5["bid_close"].to_numpy() + m5["ask_close"].to_numpy()) / 2.0
    m5_time_ns = m5.index.astype("int64").to_numpy()
    cv2_time_ns = cv2.index.astype("int64").to_numpy()
    # Map cv2 timestamps → m5 indices via searchsorted
    pos = np.searchsorted(m5_time_ns, cv2_time_ns, side="left")
    pos = np.clip(pos, 0, len(m5_time_ns) - 1)
    closes = m5_close[pos]
    print(f"[calibrate] mapped {len(closes):,} M5 closes from tape (range {closes.min():.2f}..{closes.max():.2f})", flush=True)
    future_close = np.roll(closes, -args.lookahead_bars)
    future_close[-args.lookahead_bars:] = closes[-args.lookahead_bars:]
    delta_bps = (future_close - closes) / np.maximum(closes, 1e-9) * 10000
    labels = np.full(n, 2, dtype=np.int64)  # default flat
    labels[delta_bps > args.threshold_bps] = 0   # long
    labels[delta_bps < -args.threshold_bps] = 1  # short

    # Sample a calibration set from the validation period (last 20% by time)
    val_start = int(n * 0.80)
    valid_idx = np.arange(max(meta["seq_len"], val_start), n - args.lookahead_bars)
    n_sample = min(int(args.calibration_set_size), len(valid_idx))
    sample_idx = np.random.choice(valid_idx, size=n_sample, replace=False)
    print(f"[calibrate] calibration set: {n_sample} samples from validation period", flush=True)

    # Run V10 inference on sample
    print(f"[calibrate] V10 forward...", flush=True)
    seq_len_v10 = int(meta["seq_len"])
    BATCH = 64
    all_logits, all_labels = [], []
    with torch.no_grad():
        t0 = time.time()
        for batch_start in range(0, n_sample, BATCH):
            batch = sample_idx[batch_start: batch_start + BATCH]
            seq_x = np.zeros((len(batch), seq_len_v10, per_bar.shape[1]), dtype=np.float32)
            for bi, di in enumerate(batch):
                seq_x[bi] = per_bar[di - seq_len_v10 + 1: di + 1]
            snap_x = per_bar[batch]
            cont_x = ctx_cont[batch]
            cat_x = ctx_cat[batch]
            out = model(
                seq_x=torch.from_numpy(seq_x).to(args.device),
                snap_x=torch.from_numpy(snap_x).to(args.device),
                ctx_cont=torch.from_numpy(cont_x).to(args.device),
                ctx_cat=torch.from_numpy(cat_x).to(args.device),
            )
            all_logits.append(out["direction_logits"].cpu().numpy())
            all_labels.append(labels[batch])
            if (batch_start // BATCH) % 20 == 0:
                print(f"   {batch_start + len(batch):,}/{n_sample:,}", flush=True)
    print(f"[calibrate] forward done in {time.time()-t0:.1f}s", flush=True)

    logits_np = np.concatenate(all_logits, axis=0).astype(np.float32)
    labels_np = np.concatenate(all_labels, axis=0).astype(np.int64)

    # Pre-calibration metrics
    probs_pre = F.softmax(torch.from_numpy(logits_np), dim=-1).numpy()
    nll_pre = float(F.cross_entropy(torch.from_numpy(logits_np), torch.from_numpy(labels_np)).item())
    ece_pre = expected_calibration_error(probs_pre, labels_np)
    acc_pre = float((probs_pre.argmax(axis=1) == labels_np).mean())
    print(f"\n[calibrate] BEFORE: NLL={nll_pre:.4f}  ECE={ece_pre:.4f}  acc={acc_pre:.4f}", flush=True)

    # Fit temperature
    T = fit_temperature(torch.from_numpy(logits_np), torch.from_numpy(labels_np))
    print(f"[calibrate] fitted T = {T:.4f}", flush=True)

    # Post-calibration metrics
    probs_post = F.softmax(torch.from_numpy(logits_np / T), dim=-1).numpy()
    nll_post = float(F.cross_entropy(torch.from_numpy(logits_np / T), torch.from_numpy(labels_np)).item())
    ece_post = expected_calibration_error(probs_post, labels_np)
    acc_post = float((probs_post.argmax(axis=1) == labels_np).mean())
    print(f"[calibrate] AFTER:  NLL={nll_post:.4f}  ECE={ece_post:.4f}  acc={acc_post:.4f}", flush=True)

    # Save calibration artifact
    out = {
        "temperature": float(T),
        "nll_before": nll_pre, "nll_after": nll_post,
        "ece_before": ece_pre, "ece_after": ece_post,
        "accuracy_before": acc_pre, "accuracy_after": acc_post,  # accuracy is unchanged by T (monotone)
        "calibration_set_size": int(n_sample),
        "lookahead_bars": int(args.lookahead_bars),
        "threshold_bps": float(args.threshold_bps),
        "seed": int(args.seed),
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Apply at inference: probs = softmax(direction_logits / temperature)",
    }
    out_path = bundle_dir / "calibration_temperature.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[calibrate] saved: {out_path}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
