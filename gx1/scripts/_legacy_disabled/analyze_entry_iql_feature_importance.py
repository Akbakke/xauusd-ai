#!/usr/bin/env python3
"""Feature-importance analysis for Entry-IQL state vector.

Goal: identify which of the 182 features actually predict reward (R_NET_REAL)
vs noise. Drop low-importance features to reduce model variance and overfit.

Methods used:
  1. XGBoost GBT regressor → gain-based importance (causality proxy)
  2. Permutation importance on held-out 20% → robust to colinearity
  3. Combined ranking → recommend top-K features to keep

Input: forward-outcome PLUS5 dataset.
Output: feature_importance_ranking.csv + drop_list.txt

Run:
  PYTHONPATH=/home/andre2/src/GX1_ENGINE python3 -u \\
    gx1/scripts/analyze_entry_iql_feature_importance.py \\
    --forward-outcome-dir <PLUS5_FWD>
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import xgboost as xgb

REPO = Path("/home/andre2/src/GX1_ENGINE")
sys.path.insert(0, str(REPO))

from gx1.scripts import materialize_build_entry_iql_v2 as v2_train


def load_data(fwd_dir: Path, n_max: int | None = None) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    files = sorted((fwd_dir / "per_week").glob("forward_outcomes_*.parquet"))
    print(f"[FEAT_IMP] loading {len(files)} per-week files", flush=True)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"[FEAT_IMP]   total rows: {len(df):,}", flush=True)
    if n_max and len(df) > n_max:
        df = df.sample(n=n_max, random_state=20260501).reset_index(drop=True)
        print(f"[FEAT_IMP]   sub-sampled to {len(df):,}", flush=True)

    # Use R_NET_REAL at K=96 (4h) for both LONG and SHORT as target.
    # Reward = best of long_max_mfe or short_max_mfe (tradeable signal proxy).
    def _col(name):
        if name not in df.columns:
            return pd.Series(np.zeros(len(df), dtype=np.float64))
        return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    long_term = _col("take_now_long_terminal_pnl_at_K96_v1")
    short_term = _col("take_now_short_terminal_pnl_at_K96_v1")
    # Use abs(reward) as "tradeability" target: large magnitudes (in either
    # direction) = high-information signals worth predicting.
    target = np.maximum(long_term.abs(), short_term.abs()).values.astype(np.float32)
    print(f"[FEAT_IMP] target stats: mean={target.mean():.2f} std={target.std():.2f} max={target.max():.2f}",
          flush=True)

    # Use trainer's NUMERIC_STATE_COLS (182 features) — same as production input
    state_cols = v2_train.NUMERIC_STATE_COLS
    X_data = np.zeros((len(df), len(state_cols)), dtype=np.float32)
    found_cols = []
    for i, c in enumerate(state_cols):
        if c in df.columns:
            X_data[:, i] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values
            found_cols.append(c)
    print(f"[FEAT_IMP]   matched {len(found_cols)}/{len(state_cols)} feature cols", flush=True)
    return df, target, state_cols


def xgb_importance(X: np.ndarray, y: np.ndarray, feat_names: list[str]) -> pd.DataFrame:
    print(f"[FEAT_IMP] training XGBoost regressor (gain-based importance)...", flush=True)
    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, random_state=20260501,
        tree_method="hist", n_jobs=-1, verbosity=0,
    )
    model.fit(X, y)
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    # Map back to feature names
    out = []
    for i, f in enumerate(feat_names):
        k = f"f{i}"
        out.append({"feature": f, "xgb_gain": float(gain.get(k, 0.0))})
    df = pd.DataFrame(out).sort_values("xgb_gain", ascending=False).reset_index(drop=True)
    return df


def permutation_imp(X: np.ndarray, y: np.ndarray, feat_names: list[str],
                    n_repeats: int = 5) -> pd.DataFrame:
    print(f"[FEAT_IMP] training RF for permutation importance (n_repeats={n_repeats})...",
          flush=True)
    # Smaller forest for speed
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=20,
        n_jobs=-1, random_state=20260501,
    )
    # Train on 80%
    n_train = int(0.8 * len(X))
    rng = np.random.default_rng(20260501)
    idx = rng.permutation(len(X))
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    rf.fit(X[train_idx], y[train_idx])
    print(f"[FEAT_IMP]   train R² = {rf.score(X[train_idx], y[train_idx]):.4f}", flush=True)
    print(f"[FEAT_IMP]   val R²   = {rf.score(X[val_idx], y[val_idx]):.4f}", flush=True)
    print(f"[FEAT_IMP]   computing permutation importance (slow)...", flush=True)
    perm = permutation_importance(rf, X[val_idx], y[val_idx],
                                   n_repeats=n_repeats, n_jobs=-1, random_state=20260501)
    out = []
    for i, f in enumerate(feat_names):
        out.append({"feature": f, "perm_imp_mean": float(perm.importances_mean[i]),
                    "perm_imp_std": float(perm.importances_std[i])})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-outcome-dir", type=str, required=True)
    ap.add_argument("--n-max", type=int, default=30000)
    ap.add_argument("--out-dir", type=str,
                    default="/home/andre2/GX1_DATA/reports/truth_e2e_sanity/FEATURE_IMPORTANCE_ANALYSIS")
    ap.add_argument("--keep-top-frac", type=float, default=0.60,
                    help="Recommend keeping this fraction of top features")
    args = ap.parse_args()

    fwd_dir = Path(args.forward_outcome_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, y, feat_names = load_data(fwd_dir, n_max=args.n_max)
    print(f"[FEAT_IMP] X shape: {len(df)} × {len(feat_names)}, y range [{y.min():.2f}, {y.max():.2f}]",
          flush=True)
    # Build matrix
    X = np.zeros((len(df), len(feat_names)), dtype=np.float32)
    for i, c in enumerate(feat_names):
        if c in df.columns:
            X[:, i] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values
    # Clip to remove extreme outliers (same approach as winsorize)
    X = np.clip(X, np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0))

    xgb_df = xgb_importance(X, y, feat_names)
    perm_df = permutation_imp(X, y, feat_names, n_repeats=3)
    merged = xgb_df.merge(perm_df, on="feature")
    # Combined rank: average rank of xgb_gain (desc) + perm_imp_mean (desc)
    merged["xgb_rank"] = merged["xgb_gain"].rank(ascending=False, method="dense")
    merged["perm_rank"] = merged["perm_imp_mean"].rank(ascending=False, method="dense")
    merged["combined_rank"] = (merged["xgb_rank"] + merged["perm_rank"]) / 2.0
    merged = merged.sort_values("combined_rank").reset_index(drop=True)

    out_csv = out_dir / "feature_importance_ranking.csv"
    merged.to_csv(out_csv, index=False)
    print(f"[FEAT_IMP] wrote ranking -> {out_csv}", flush=True)

    n_keep = int(args.keep_top_frac * len(feat_names))
    keep_list = merged.head(n_keep)["feature"].tolist()
    drop_list = merged.tail(len(feat_names) - n_keep)["feature"].tolist()
    (out_dir / "keep_list.txt").write_text("\n".join(keep_list))
    (out_dir / "drop_list.txt").write_text("\n".join(drop_list))
    print(f"[FEAT_IMP] keep top {n_keep}/{len(feat_names)} ({args.keep_top_frac*100:.0f}%)",
          flush=True)
    print(f"[FEAT_IMP] TOP 20:")
    for i, row in merged.head(20).iterrows():
        print(f"  {row['feature']:<55s}  gain={row['xgb_gain']:.4f}  perm={row['perm_imp_mean']:+.4f}")
    print(f"[FEAT_IMP] BOTTOM 20 (candidates for drop):")
    for i, row in merged.tail(20).iterrows():
        print(f"  {row['feature']:<55s}  gain={row['xgb_gain']:.4f}  perm={row['perm_imp_mean']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
