#!/usr/bin/env python3
"""V3 exit-transformer bundle evaluator.

Loads a trained V3 bundle (V5 thin-record format) and produces a comprehensive
evaluation report against a held-out split of the V3 dataset:

  - Confusion matrix at threshold 0.5
  - Per-side breakdown (long/short)
  - Calibration curve + ECE (Expected Calibration Error)
  - Per-regime breakdown (session, year, vol_regime if available)
  - Threshold sweep (precision/recall/F1 at thresholds 0.1..0.9)
  - PnL-aware metric: at each threshold, what would we have earned/saved?
    Approximation: should_exit=1 → exit at current pnl_bps_now;
                   should_exit=0 → continue and earn teacher_final_pnl_bps

USAGE
    python -m gx1.scripts.evaluate_exit_v5_bundle \
        --bundle-dir /home/andre2/GX1_DATA/models/exit_transformer_v0/EXIT_V5_THIN__BIDIR_2026Q2_* \
        --split test \
        --records-limit 50000

Designed to compare baselines (V5) vs improvements (V6). Output is JSON +
human-readable summary printed to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gx1.exits.training.thin_record_dataset import ThinRecordDataset
from gx1.policy.exit_transformer_v0 import ExitTransformerV0, _attach_labels_to_exit_records


def load_bundle(bundle_dir: Path) -> Tuple[ExitTransformerV0, Dict[str, Any]]:
    config = json.loads((bundle_dir / "transformer_config.json").read_text())
    model = ExitTransformerV0(
        input_dim=config["input_dim"], window_len=config["window_len"],
        d_model=config["d_model"], n_heads=config["n_heads"],
        n_layers=config["n_layers"], dropout=config.get("dropout", 0.1),
    )
    state_path = bundle_dir / "exit_transformer_v0.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Tuple[float, List[Dict[str, float]]]:
    """ECE for binary: bin by predicted prob, compute |conf - acc| × bin_weight."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins_info = []
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (probs > lo) & (probs <= hi)
        if mask.sum() == 0:
            bins_info.append({"lo": float(lo), "hi": float(hi), "n": 0,
                              "conf": float("nan"), "acc": float("nan")})
            continue
        bin_conf = float(probs[mask].mean())
        bin_acc = float(labels[mask].mean())
        bin_weight = float(mask.mean())
        ece += abs(bin_conf - bin_acc) * bin_weight
        bins_info.append({"lo": float(lo), "hi": float(hi), "n": int(mask.sum()),
                          "conf": bin_conf, "acc": bin_acc})
    return float(ece), bins_info


def threshold_sweep(probs: np.ndarray, labels: np.ndarray) -> List[Dict[str, float]]:
    out = []
    for thr in np.arange(0.05, 1.0, 0.05):
        pred = (probs >= thr).astype(np.int32)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out.append({"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": float(prec), "recall": float(rec), "f1": float(f1),
                    "n_positives": int(pred.sum())})
    return out


def pnl_aware_eval(records: List[Dict[str, Any]], probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """If should_exit predicted: exit at pnl_bps_now (locked-in pnl).
    Else: continue and earn teacher_final_pnl_bps.

    Compares against:
    - Baseline: never exit early (always earn teacher_final_pnl_bps)
    - Oracle: exit when teacher_final_pnl_bps < pnl_bps_now (perfect hindsight)
    """
    pnl_now = np.array([float((r.get("scalars") or {}).get("pnl_bps_now", 0.0) or 0.0) for r in records])
    pnl_final = np.array([float(r.get("teacher_final_pnl_bps", 0.0) or 0.0) for r in records])
    pred = probs >= threshold

    realized = np.where(pred, pnl_now, pnl_final)
    baseline = pnl_final
    oracle = np.maximum(pnl_now, pnl_final)
    return {
        "threshold": float(threshold),
        "realized_pnl_mean_bps": float(realized.mean()),
        "baseline_pnl_mean_bps": float(baseline.mean()),
        "oracle_pnl_mean_bps": float(oracle.mean()),
        "pct_of_oracle": float(realized.mean() / max(oracle.mean(), 1e-6)),
        "pct_vs_baseline": float((realized.mean() - baseline.mean()) / max(abs(baseline.mean()), 1e-6)),
        "n_exits_taken": int(pred.sum()),
        "n_total": int(len(records)),
    }


def per_regime_breakdown(
    records: List[Dict[str, Any]], probs: np.ndarray, labels: np.ndarray,
) -> Dict[str, Any]:
    """Group accuracy + ECE by side, year, session (if available)."""
    out: Dict[str, Any] = {}

    # Side breakdown
    by_side: Dict[str, Tuple[List[float], List[int]]] = defaultdict(lambda: ([], []))
    for r, p, y in zip(records, probs, labels):
        side = str(r.get("side", "unknown"))
        by_side[side][0].append(float(p))
        by_side[side][1].append(int(y))
    out["side"] = {}
    for side, (ps, ys) in by_side.items():
        ps_arr, ys_arr = np.array(ps), np.array(ys)
        if len(ps_arr) == 0:
            continue
        acc = float(((ps_arr >= 0.5) == ys_arr.astype(bool)).mean())
        pos_rate = float(ys_arr.mean())
        out["side"][side] = {"n": len(ps_arr), "pos_rate": pos_rate, "acc": acc,
                              "pred_mean": float(ps_arr.mean())}

    # Year breakdown
    by_year: Dict[int, Tuple[List[float], List[int]]] = defaultdict(lambda: ([], []))
    for r, p, y in zip(records, probs, labels):
        ts = r.get("ts", "")
        try:
            year = int(str(ts)[:4])
        except Exception:
            continue
        by_year[year][0].append(float(p))
        by_year[year][1].append(int(y))
    out["year"] = {}
    for year, (ps, ys) in sorted(by_year.items()):
        ps_arr, ys_arr = np.array(ps), np.array(ys)
        if len(ps_arr) == 0:
            continue
        acc = float(((ps_arr >= 0.5) == ys_arr.astype(bool)).mean())
        out["year"][str(year)] = {"n": len(ps_arr), "pos_rate": float(ys_arr.mean()),
                                   "acc": acc, "pred_mean": float(ps_arr.mean())}

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 exit transformer bundle evaluator")
    parser.add_argument("--bundle-dir", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str,
                        default="/home/andre2/GX1_DATA/data/training/exit_v3_v5_training_2020_2026")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--train-cutoff", type=str, default="2025-01-01T00:00:00+00:00")
    parser.add_argument("--val-cutoff", type=str, default="2025-09-01T00:00:00+00:00")
    parser.add_argument("--records-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-json", type=str, default=None,
                        help="Output JSON path (default: <bundle_dir>/eval_<split>_<timestamp>.json)")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    print(f"[eval] bundle: {bundle_dir.name}", flush=True)

    # Load model
    model, config = load_bundle(bundle_dir)
    model.to(args.device)
    print(f"[eval] model loaded: input_dim={config['input_dim']} window_len={config['window_len']} "
          f"d_model={config['d_model']} n_layers={config['n_layers']}", flush=True)

    # Load dataset
    base = ThinRecordDataset(args.dataset_dir, records_limit=args.records_limit)
    print(f"[eval] dataset: {len(base.records):,} records", flush=True)

    # Split records by trade timestamp (matches v5/v6 trainer logic)
    from gx1.scripts.train_exit_v5_thin_records import split_records_by_time
    train, val, test = split_records_by_time(
        base.records, train_cutoff=args.train_cutoff, val_cutoff=args.val_cutoff,
    )
    eval_records = {"train": train, "val": val, "test": test}[args.split]
    print(f"[eval] split={args.split} → {len(eval_records):,} records", flush=True)

    # Attach labels (need io_features placeholder for labeler — same approach as v5/v6 trainer)
    print(f"[eval] labeling records (per-trade io_features reconstruction)...", flush=True)
    from collections import defaultdict
    trades = defaultdict(list)
    for rec in eval_records:
        sc = rec.get("scalars") or {}
        for k, v in sc.items():
            rec.setdefault(k, v)
        rec.setdefault("exit_ml_io_version", base.manifest.get("io_version", ""))
        tid = rec.get("trade_uid") or rec.get("trade_id")
        if tid:
            trades[tid].append(rec)
    t0 = time.time()
    for ti, (tid, trade_recs) in enumerate(trades.items()):
        for rec in trade_recs:
            rec["io_features"] = base._reconstruct_io_features(rec).tolist()
        _attach_labels_to_exit_records(trade_recs)
        for rec in trade_recs:
            rec.pop("io_features", None)
    print(f"[eval] labeled in {time.time()-t0:.1f}s", flush=True)

    # Run inference
    print(f"[eval] running inference (batch={args.batch_size}, device={args.device})...", flush=True)
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        for i in range(0, len(eval_records), args.batch_size):
            batch_recs = eval_records[i:i + args.batch_size]
            xs = np.stack([base._reconstruct_io_features(r) for r in batch_recs]).astype(np.float32)
            x = torch.from_numpy(xs).to(args.device)
            logits = model.forward_logits(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend([float(r.get("should_exit", 0.0) or 0.0) for r in batch_recs])
            if (i // args.batch_size) % 20 == 0:
                rate = (i + len(batch_recs)) / max(time.time() - t0, 1e-6)
                print(f"   {i + len(batch_recs):,}/{len(eval_records):,} ({rate:.0f}/s)", flush=True)
    print(f"[eval] inference done in {time.time()-t0:.1f}s", flush=True)

    probs_arr = np.concatenate(all_probs)
    labels_arr = np.array(all_labels)

    # Compute metrics
    print(f"\n[eval] === METRICS ===", flush=True)

    # Overall
    pos_rate = float(labels_arr.mean())
    pred_mean = float(probs_arr.mean())
    acc_at_thr = float(((probs_arr >= args.threshold) == labels_arr.astype(bool)).mean())
    print(f"  pos_rate: {pos_rate:.4f}  pred_mean: {pred_mean:.4f}  acc@0.5: {acc_at_thr:.4f}", flush=True)

    # Confusion at threshold
    pred = probs_arr >= args.threshold
    tp = int(((pred == 1) & (labels_arr == 1)).sum())
    fp = int(((pred == 1) & (labels_arr == 0)).sum())
    fn = int(((pred == 0) & (labels_arr == 1)).sum())
    tn = int(((pred == 0) & (labels_arr == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"  confusion@{args.threshold}: TP={tp} FP={fp} FN={fn} TN={tn}", flush=True)
    print(f"  precision: {prec:.4f}  recall: {rec:.4f}  f1: {f1:.4f}", flush=True)

    # ECE
    ece, bins_info = expected_calibration_error(probs_arr, labels_arr)
    print(f"  ECE: {ece:.4f}", flush=True)

    # Threshold sweep
    sweep = threshold_sweep(probs_arr, labels_arr)
    best_f1 = max(sweep, key=lambda x: x["f1"])
    print(f"  best F1 threshold: {best_f1['threshold']:.2f}  f1={best_f1['f1']:.4f}  prec={best_f1['precision']:.4f}  rec={best_f1['recall']:.4f}", flush=True)

    # Per-regime breakdown
    regime = per_regime_breakdown(eval_records, probs_arr, labels_arr)
    print(f"\n  per-side:", flush=True)
    for side, stats in regime.get("side", {}).items():
        print(f"    {side:>6}: n={stats['n']:>6,}  pos_rate={stats['pos_rate']:.4f}  acc={stats['acc']:.4f}  pred_mean={stats['pred_mean']:.4f}", flush=True)
    print(f"\n  per-year:", flush=True)
    for year, stats in regime.get("year", {}).items():
        print(f"    {year}: n={stats['n']:>6,}  pos_rate={stats['pos_rate']:.4f}  acc={stats['acc']:.4f}  pred_mean={stats['pred_mean']:.4f}", flush=True)

    # PnL-aware
    pnl = pnl_aware_eval(eval_records, probs_arr, threshold=args.threshold)
    print(f"\n  PnL-aware @ thr={args.threshold}:", flush=True)
    print(f"    realized: {pnl['realized_pnl_mean_bps']:+.2f} bps", flush=True)
    print(f"    baseline (never exit): {pnl['baseline_pnl_mean_bps']:+.2f} bps", flush=True)
    print(f"    oracle (perfect): {pnl['oracle_pnl_mean_bps']:+.2f} bps", flush=True)
    print(f"    pct of oracle: {pnl['pct_of_oracle']:.1%}", flush=True)
    print(f"    delta vs baseline: {pnl['pct_vs_baseline']:+.1%}", flush=True)

    # Save full report
    out = {
        "bundle_dir": str(bundle_dir),
        "split": args.split,
        "n_records": len(eval_records),
        "config": config,
        "overall": {"pos_rate": pos_rate, "pred_mean": pred_mean, "acc_at_threshold": acc_at_thr,
                    "threshold": args.threshold,
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "precision": prec, "recall": rec, "f1": f1, "ece": ece},
        "threshold_sweep": sweep,
        "best_f1": best_f1,
        "regime_breakdown": regime,
        "pnl_aware": pnl,
        "calibration_bins": bins_info,
    }
    out_json = args.out_json or str(bundle_dir / f"eval_{args.split}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    Path(out_json).write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[eval] report saved → {out_json}", flush=True)


if __name__ == "__main__":
    main()
