#!/usr/bin/env python3
"""V12.2 V10 multi-TF hyperparameter sweep.

Sweeps lr × multi_tf_scale × (optionally multi_tf_num_layers) on a stratified
subsample of the V10 training set. Each trial:
  - 3 epochs (no early stopping — we want raw val_loss trajectory)
  - 60k subsampled rows (stratified: preserves long/short/flat ratios)
  - Same model size, same dataset (deterministic except for sampling seed)

Output:
  - per-trial log under /tmp/v10_sweep_<TS>/trial_<i>.log
  - sweep_results.json with all trials sorted by best val_loss
  - prints winner config at the end

Total time estimate:
  - 9 trials × 3 epochs × ~7 min/epoch (60k subsample) = ~3 hours
  - Multi-TF cache persists across trials (built once)

Usage:
  python3 gx1/scripts/v10_multi_tf_sweep.py \\
      --epochs 3 --subsample-rows 60000 \\
      --output-dir /tmp/v10_sweep_$(date -u +%Y%m%dT%H%M%SZ)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO = Path("/home/andre2/src/GX1_ENGINE")
TRAINER = REPO / "gx1/models/entry_v10/entry_v10_ctx_train_v3.py"
# 2026-05-26: repointed to the current ctx=105 dataset (old v3_dataset_6yr deleted in cleanup).
DATASET_DIR = Path("/home/andre2/GX1_DATA/data/training/entry_v10_ctx_BASE80_PARITY_NORELABEL_6yr_20260526T102912Z")
TRAIN_PARQUET = DATASET_DIR / "v10_base80_parity_norelabel_6yr_dataset__HOLD_03B_train.parquet"
M5_PREBUILT = Path("/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/xauusd_m5_CANONICAL_V3_2020_2026.parquet")


# Hyperparam grid: lr × multi_tf_scale. (multi_tf_num_layers fixed at 2.)
DEFAULT_GRID = [
    {"lr": 1e-5, "scale": 0.10},
    {"lr": 1e-5, "scale": 0.25},
    {"lr": 1e-5, "scale": 0.50},
    {"lr": 3e-5, "scale": 0.10},
    {"lr": 3e-5, "scale": 0.25},
    {"lr": 3e-5, "scale": 0.50},
    {"lr": 1e-4, "scale": 0.10},
    {"lr": 1e-4, "scale": 0.25},
    {"lr": 1e-4, "scale": 0.50},
]


def run_trial(
    *, lr: float, scale: float, out_root: Path, trial_idx: int,
    epochs: int, subsample_rows: int, batch_size: int, num_workers: int,
) -> Dict[str, Any]:
    """Run one training trial, return {best_val, epoch_losses}."""
    trial_dir = out_root / f"trial_{trial_idx:02d}_lr{lr:.0e}_scale{scale:.2f}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "train.log"
    bundle_dir = trial_dir / "bundle"

    cmd = [
        "python3", "-u", str(TRAINER),
        "--train",
        "--dataset_dir", str(DATASET_DIR),
        "--dataset_train_parquet", str(TRAIN_PARQUET),
        "--out_bundle_dir", str(bundle_dir),
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", f"{lr}",
        "--early-stopping-patience", "999",   # disable, we want raw curves
        "--device", "cuda",
        # --enable-multi-tf removed 2026-05-26 (multi-TF is now mandatory; flag deleted)
        "--m5-prebuilt-path", str(M5_PREBUILT),
        "--multi-tf-scale", f"{scale}",
        "--subsample-rows", str(subsample_rows),
        "--num-workers", str(num_workers),
    ]
    env = {**os.environ, "GX1_DATA": "/home/andre2/GX1_DATA",
           "PYTHONPATH": str(REPO)}

    t0 = time.time()
    print(f"\n=== TRIAL {trial_idx:02d}: lr={lr:.0e}  scale={scale:.2f} ===")
    print(f"  log: {log_path}")
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(REPO))
    elapsed = time.time() - t0

    # Parse val losses from log
    text = log_path.read_text()
    val_pattern = re.compile(r"split=val epoch=(\d+).*?total=([\d.]+)")
    val_losses = [(int(m.group(1)), float(m.group(2))) for m in val_pattern.finditer(text)]
    delta_pattern = re.compile(r"split=train epoch=(\d+).*?delta_abs_mean=([\d.]+)")
    deltas = [(int(m.group(1)), float(m.group(2))) for m in delta_pattern.finditer(text)]

    result = {
        "trial_idx": trial_idx,
        "lr": lr,
        "scale": scale,
        "elapsed_sec": elapsed,
        "exit_code": proc.returncode,
        "val_losses": val_losses,
        "best_val_loss": min((v for _, v in val_losses), default=None),
        "final_train_delta": deltas[-1][1] if deltas else None,
        "log_path": str(log_path),
        "bundle_dir": str(bundle_dir),
    }
    print(f"  done in {elapsed:.0f}s  best_val={result['best_val_loss']}  "
          f"final_delta={result['final_train_delta']}")
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: /tmp/v10_sweep_<TS>)")
    p.add_argument("--epochs", type=int, default=3,
                   help="Epochs per trial (default 3 — enough to see direction)")
    p.add_argument("--subsample-rows", type=int, default=60000,
                   help="Stratified train subsample for fast trials (0=all rows)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers per trial (default 0). 2-4 helps if "
                        "CPU is bottleneck; each worker duplicates multi-TF cache "
                        "(~33MB), so memory usage scales with workers.")
    p.add_argument("--config-grid", type=str, default=None,
                   help="JSON file with list of {lr, scale} dicts; uses default 3x3 grid if omitted")
    args = p.parse_args()

    if args.config_grid:
        grid = json.loads(Path(args.config_grid).read_text())
    else:
        grid = DEFAULT_GRID

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = args.output_dir or Path(f"/tmp/v10_sweep_{ts}")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"V10 multi-TF sweep: {len(grid)} trials × {args.epochs} epochs")
    print(f"  subsample: {args.subsample_rows:,} rows")
    print(f"  output:    {out_root}")
    print(f"  trainer:   {TRAINER}")
    (out_root / "sweep_config.json").write_text(json.dumps({
        "epochs": args.epochs, "subsample_rows": args.subsample_rows,
        "batch_size": args.batch_size, "grid": grid,
        "started_at": ts,
    }, indent=2))

    results: List[Dict[str, Any]] = []
    for i, cfg in enumerate(grid):
        try:
            r = run_trial(
                lr=cfg["lr"], scale=cfg["scale"],
                out_root=out_root, trial_idx=i,
                epochs=args.epochs, subsample_rows=args.subsample_rows,
                batch_size=args.batch_size, num_workers=args.num_workers,
            )
            results.append(r)
            # Save after each trial in case sweep is interrupted
            (out_root / "sweep_results.json").write_text(json.dumps(results, indent=2, default=str))
        except Exception as exc:
            print(f"  TRIAL {i} FAILED: {exc}")
            results.append({"trial_idx": i, "lr": cfg["lr"], "scale": cfg["scale"], "error": str(exc)})

    # Rank by best_val_loss (lower is better, None = failed)
    ranked = sorted(
        [r for r in results if r.get("best_val_loss") is not None],
        key=lambda r: r["best_val_loss"],
    )

    print("\n" + "=" * 75)
    print("SWEEP RESULTS (ranked by best val_loss)")
    print("=" * 75)
    print(f"{'rank':>4} {'lr':>8} {'scale':>6} {'best_val':>10} {'final_delta':>14} {'time(s)':>8}")
    for rank, r in enumerate(ranked, 1):
        print(f"{rank:>4} {r['lr']:>8.0e} {r['scale']:>6.2f} {r['best_val_loss']:>10.4f} "
              f"{r['final_train_delta']:>14.1f} {r['elapsed_sec']:>8.0f}")
    if ranked:
        winner = ranked[0]
        print("\n🏆 WINNER:")
        print(f"  lr={winner['lr']:.0e}  scale={winner['scale']:.2f}  "
              f"best_val={winner['best_val_loss']:.4f}  "
              f"final_delta={winner['final_train_delta']:.1f}")
        print(f"  → Full training: --lr {winner['lr']} --multi-tf-scale {winner['scale']}")
        (out_root / "WINNER.json").write_text(json.dumps(winner, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
