#!/usr/bin/env python3
"""Feature-importance analysis for Exit-IQL bar_state.

Twin to analyze_entry_iql_feature_importance.py — runs XGBoost + permutation
importance on per_bar parquet rows to identify which features predict
r_exit_now_v1 (per-bar exit reward).

Output: drop_list of low-importance features → candidate features to remove
from Exit-IQL state in V2 rebuild.

Run:
  PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 -u \\
    gx1/scripts/analyze_exit_iql_feature_importance.py \\
    [--per-bar-dir <V3TRACKED_LOCK>] [--n-max 100000]
"""
from __future__ import annotations
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import xgboost as xgb

DEFAULT_PER_BAR_DIR = Path(
    "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
    "BUILD_EXIT_IQL_PER_BAR_DATASET_V12_V3PLUS_FULL_20260519T012648Z_LOCK_"
    "V3TRACKED_20260519T022946Z_LOCK"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bar-dir", type=Path, default=DEFAULT_PER_BAR_DIR)
    ap.add_argument("--n-max", type=int, default=80_000,
                    help="Subsample bars for analysis (default 80K)")
    ap.add_argument("--out-dir", type=Path,
                    default="/home/andre2/GX1_DATA/reports/truth_e2e_sanity/EXIT_FEATURE_IMPORTANCE_ANALYSIS")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EXIT_FI] sampling per_bar parquets from {args.per_bar_dir.name}")
    files = sorted((args.per_bar_dir / "per_week").glob("exit_per_bar_m1_*.parquet"))
    # Sample 20 random weeks for speed
    rng = np.random.default_rng(42)
    if len(files) > 20:
        idx = rng.choice(len(files), size=20, replace=False)
        files = [files[i] for i in sorted(idx)]
    print(f"[EXIT_FI] reading {len(files)} weekly parquets...", flush=True)

    parts = []
    for f in files:
        d = pd.read_parquet(f)
        # Drop forced_terminal + never_fire (match Exit-IQL training filter)
        d = d[(d.get("is_never_fire_v1", 0) == 0) & (d.get("forced_terminal_v1", 0) == 0)]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    print(f"[EXIT_FI] total rows after filter: {len(df):,}")

    if len(df) > args.n_max:
        df = df.sample(n=args.n_max, random_state=42).reset_index(drop=True)
        print(f"[EXIT_FI] sub-sampled to {len(df):,}")

    # Target: r_exit_now_v1 (per-bar exit reward)
    target_col = "r_exit_now_v1"
    if target_col not in df.columns:
        print(f"[EXIT_FI] ERROR: {target_col} missing")
        return 1
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).to_numpy()

    # Feature cols: numeric only, exclude identifiers + targets
    EXCLUDE = {
        "candidate_uid", "bar_idx_v1", "bar_ts_ns_v1", "side", "side_v1",
        "is_never_fire_v1", "forced_terminal_v1",
    }
    EXCLUDE.update([c for c in df.columns if c.startswith("r_") or c.startswith("hold_max_pnl_K")])
    EXCLUDE.add(target_col)
    EXCLUDE.add("exit_now_reward_bps_v1")
    feat_cols = [c for c in df.columns
                 if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    print(f"[EXIT_FI] feature cols: {len(feat_cols)}")

    X = df[feat_cols].fillna(0.0).astype(np.float32)
    print(f"[EXIT_FI] X shape: {X.shape}  y shape: {y.shape}")

    # XGBoost gain importance
    print("[EXIT_FI] fitting XGBoost...")
    t0 = time.time()
    xg = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                          tree_method="hist", n_jobs=8, random_state=42)
    xg.fit(X, y)
    print(f"  done in {time.time()-t0:.1f}s")
    gain = dict(zip(feat_cols, xg.feature_importances_.tolist()))

    # Permutation importance on held-out 20%
    print("[EXIT_FI] running permutation importance on held-out subset...")
    n_val = int(len(X) * 0.2)
    rf = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=8, random_state=42)
    rf.fit(X.iloc[:-n_val], y[:-n_val])
    t1 = time.time()
    perm = permutation_importance(rf, X.iloc[-n_val:], y[-n_val:],
                                  n_repeats=3, n_jobs=8, random_state=42)
    print(f"  done in {time.time()-t1:.1f}s")
    perm_mean = dict(zip(feat_cols, perm.importances_mean.tolist()))

    # Combined ranking
    rows = []
    for c in feat_cols:
        rows.append({"feature": c, "gain": gain[c], "perm": perm_mean[c],
                     "combined": gain[c] + perm_mean[c]})
    rank_df = pd.DataFrame(rows).sort_values("combined", ascending=False).reset_index(drop=True)
    rank_df.to_csv(args.out_dir / "feature_importance_ranking.csv", index=False)

    # Top + bottom
    print("\n[EXIT_FI] TOP 20:")
    for _, r in rank_df.head(20).iterrows():
        print(f"  {r['feature']:<50}  gain={r['gain']:>10.4f}  perm={r['perm']:>+8.4f}")
    print("\n[EXIT_FI] BOTTOM 20 (drop candidates):")
    for _, r in rank_df.tail(20).iterrows():
        print(f"  {r['feature']:<50}  gain={r['gain']:>10.4f}  perm={r['perm']:>+8.4f}")

    # Drop list: features with gain ≈ 0 AND perm ≈ 0
    drop_list = rank_df[(rank_df["gain"] < 0.001) & (rank_df["perm"].abs() < 0.0005)]["feature"].tolist()
    with open(args.out_dir / "drop_list.txt", "w") as f:
        f.write("\n".join(drop_list) + "\n")
    keep_list = rank_df[~rank_df["feature"].isin(drop_list)]["feature"].tolist()
    with open(args.out_dir / "keep_list.txt", "w") as f:
        f.write("\n".join(keep_list) + "\n")

    print(f"\n[EXIT_FI] drop_list: {len(drop_list)} features")
    print(f"[EXIT_FI] keep_list: {len(keep_list)} features")
    print(f"[EXIT_FI] outputs → {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
