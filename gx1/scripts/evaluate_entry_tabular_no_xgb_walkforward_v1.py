"""Walk-forward validation for the no-XGB tabular Entry baseline.

This is the next gate after the single train/val/test LightGBM baseline. It
trains only on rows before each fold, uses a pre-fold validation tail for early
stopping, and evaluates the fold with the same selective edge/PnL metrics used
for ET/XGB comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

from gx1.audit.entry_transformer_feature_audit import _stack_list_column
from gx1.scripts.evaluate_entry_selective_edge_v1 import (
    CLASS_NAMES,
    SESSION_NAMES,
    _evaluate_predictions,
    _json_default,
    _parse_float_list,
    _split_files,
    _spread_aware_path,
)
from gx1.scripts.evaluate_entry_tabular_no_xgb_baseline_v1 import (
    _check_no_xgb_feature_names,
    _predict_proba,
    _selected_feature_names,
    _train_lightgbm,
    XGB_SIGNAL_FIELD_COUNT,
)


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    start: pd.Timestamp
    end: pd.Timestamp


def _parse_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _parse_folds(raw: str) -> list[FoldSpec]:
    folds: list[FoldSpec] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            fold_id, bounds = part.split("=", 1)
        else:
            fold_id, bounds = f"fold_{len(folds)+1:02d}", part
        if ":" not in bounds:
            raise SystemExit(f"fold must be id=start:end or start:end, got {part!r}")
        start_s, end_s = bounds.split(":", 1)
        start = _parse_timestamp(start_s)
        end = _parse_timestamp(end_s)
        if end <= start:
            raise SystemExit(f"fold end must be after start: {part!r}")
        folds.append(FoldSpec(fold_id=fold_id.strip(), start=start, end=end))
    if not folds:
        raise SystemExit("no folds parsed")
    return folds


def _default_folds() -> list[FoldSpec]:
    raw = ",".join([
        "2023H1=2023-01-01:2023-07-01",
        "2023H2=2023-07-01:2024-01-01",
        "2024H1=2024-01-01:2024-07-01",
        "2024H2=2024-07-01:2025-01-01",
        "2025H1=2025-01-01:2025-07-01",
        "2025H2=2025-07-01:2026-01-01",
        "2026YTD=2026-01-01:2026-05-01",
        "2026HOLDOUT=2026-05-01:2026-06-13",
    ])
    return _parse_folds(raw)


def _load_split_matrix(parquet_path: Path, split_name: str, feature_names: list[str] | None) -> dict[str, Any]:
    cols = ["time", "snap", "ctx_cont", "ctx_cat", "y_direction", "label_horizon_bars", "path_quality_bps"]
    df = pd.read_parquet(parquet_path, columns=cols)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    ctx_cat_raw = _stack_list_column(df["ctx_cat"], np.int64)
    df["session_id"] = ctx_cat_raw[:, 0].astype(int)
    df["session"] = df["session_id"].map(SESSION_NAMES).fillna("UNKNOWN")
    df["source_split"] = split_name

    snap = _stack_list_column(df["snap"], np.float32)[:, XGB_SIGNAL_FIELD_COUNT:]
    ctx_cont = _stack_list_column(df["ctx_cont"], np.float32)
    ctx_cat = ctx_cat_raw.astype(np.float32)
    x = np.concatenate([snap, ctx_cont, ctx_cat], axis=1).astype(np.float32, copy=False)
    y = df["y_direction"].to_numpy(np.int64)

    if feature_names is None:
        feature_names, categorical_idx = _selected_feature_names(parquet_path)
        if len(feature_names) != x.shape[1]:
            raise RuntimeError(f"feature name mismatch for {parquet_path}: names={len(feature_names)} x={x.shape[1]}")
        _check_no_xgb_feature_names(feature_names)
    else:
        categorical_idx = []

    base_df = df.drop(columns=["snap", "ctx_cont", "ctx_cat"]).reset_index(drop=True)
    return {
        "x": x,
        "y": y,
        "base_df": base_df,
        "feature_names": feature_names,
        "categorical_idx": categorical_idx,
    }


def _load_all_data(dataset_dir: Path, splits: list[str]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], list[int]]:
    files = _split_files(dataset_dir, splits)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    dfs: list[pd.DataFrame] = []
    feature_names: list[str] | None = None
    categorical_idx: list[int] = []
    for split in splits:
        loaded = _load_split_matrix(files[split], split, feature_names)
        if feature_names is None:
            feature_names = list(loaded["feature_names"])
            categorical_idx = list(loaded["categorical_idx"])
        xs.append(loaded["x"])
        ys.append(loaded["y"])
        dfs.append(loaded["base_df"])
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    df = pd.concat(dfs, axis=0, ignore_index=True)
    order = np.argsort(df["time"].to_numpy(), kind="mergesort")
    x = x[order]
    y = y[order]
    df = df.iloc[order].reset_index(drop=True)
    assert feature_names is not None
    return x, y, df, feature_names, categorical_idx


def _fold_indices(
    *,
    times: pd.Series,
    fold: FoldSpec,
    val_tail_days: int,
    min_val_rows: int,
    min_train_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = pd.to_datetime(times, utc=True)
    pre = np.flatnonzero((t < fold.start).to_numpy())
    eval_idx = np.flatnonzero(((t >= fold.start) & (t < fold.end)).to_numpy())
    if len(pre) < min_train_rows + min_val_rows:
        raise RuntimeError(f"{fold.fold_id}: insufficient pre-fold rows: {len(pre)}")
    tail_start = fold.start - pd.Timedelta(days=val_tail_days)
    val_idx = np.flatnonzero(((t >= tail_start) & (t < fold.start)).to_numpy())
    if len(val_idx) < min_val_rows:
        val_idx = pre[-min_val_rows:]
    train_idx = pre[~np.isin(pre, val_idx, assume_unique=True)]
    if len(train_idx) < min_train_rows:
        raise RuntimeError(f"{fold.fold_id}: insufficient train rows after validation tail: {len(train_idx)}")
    if len(eval_idx) == 0:
        raise RuntimeError(f"{fold.fold_id}: no eval rows in {fold.start}..{fold.end}")
    return train_idx, val_idx, eval_idx


def _maybe_cap_train_indices(indices: np.ndarray, max_train_rows: int, seed: int) -> np.ndarray:
    if max_train_rows <= 0 or len(indices) <= max_train_rows:
        return indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=max_train_rows, replace=False)
    return np.sort(chosen)


def _run_one_model(
    *,
    model_name: str,
    fold: FoldSpec,
    x: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    categorical_idx: list[int],
    source_parquet: Path,
    top_fracs: list[float],
    extra_cost_bps_values: list[float],
    args: argparse.Namespace,
    shuffled: bool,
    model_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], np.ndarray]:
    y_train = y[train_idx]
    if shuffled:
        fold_seed = int(hashlib.sha256(fold.fold_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(int(args.seed) + fold_seed)
        y_train = np.array(y_train, copy=True)
        rng.shuffle(y_train)

    t0 = perf_counter()
    model = _train_lightgbm(
        x_train=x[train_idx],
        y_train=y_train,
        x_val=x[val_idx],
        y_val=y[val_idx],
        categorical_idx=categorical_idx,
        seed=int(args.seed),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        num_leaves=int(args.num_leaves),
        max_depth=int(args.max_depth),
        min_child_samples=int(args.min_child_samples),
        n_jobs=int(args.n_jobs),
        early_stopping_rounds=int(args.early_stopping_rounds),
    )
    train_seconds = perf_counter() - t0

    model_path = model_dir / f"{fold.fold_id}__{model_name}.joblib"
    joblib.dump(model, model_path)

    eval_df = df.iloc[eval_idx].reset_index(drop=True)
    probs = _predict_proba(model, x[eval_idx])
    path = _spread_aware_path(
        source_parquet=source_parquet,
        sample_times=eval_df["time"],
        horizons=eval_df["label_horizon_bars"].to_numpy(np.int64),
    )
    rows, summary = _evaluate_predictions(
        split=fold.fold_id,
        model_name=model_name,
        df=eval_df,
        probs=probs,
        path=path,
        top_fracs=top_fracs,
        extra_cost_bps_values=extra_cost_bps_values,
    )
    fold_meta = {
        "fold_id": fold.fold_id,
        "fold_start": str(fold.start),
        "fold_end": str(fold.end),
        "model_path": str(model_path),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "eval_rows": int(len(eval_idx)),
        "train_time_min": str(df.iloc[train_idx]["time"].min()),
        "train_time_max": str(df.iloc[train_idx]["time"].max()),
        "val_time_min": str(df.iloc[val_idx]["time"].min()),
        "val_time_max": str(df.iloc[val_idx]["time"].max()),
        "eval_time_min": str(eval_df["time"].min()),
        "eval_time_max": str(eval_df["time"].max()),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "train_seconds": float(train_seconds),
        "shuffled_train_labels": bool(shuffled),
    }
    for row in rows:
        row.update(fold_meta)
    summary.update(fold_meta)
    return rows, summary, fold_meta, model.feature_importances_


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    data_splits = [s.strip() for s in str(args.data_splits).split(",") if s.strip()]
    eval_folds = _parse_folds(args.folds) if args.folds else _default_folds()
    top_fracs = _parse_float_list(args.top_fracs)
    extra_cost_bps_values = _parse_float_list(args.extra_cost_bps)

    x, y, df, feature_names, categorical_idx = _load_all_data(dataset_dir, data_splits)
    _check_no_xgb_feature_names(feature_names)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    fold_metas: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []
    for fold in eval_folds:
        train_idx, val_idx, eval_idx = _fold_indices(
            times=df["time"],
            fold=fold,
            val_tail_days=int(args.val_tail_days),
            min_val_rows=int(args.min_val_rows),
            min_train_rows=int(args.min_train_rows),
        )
        train_idx = _maybe_cap_train_indices(train_idx, int(args.max_train_rows), int(args.seed))
        fold_models = [("lightgbm_tabular_no_xgb_wf", False)]
        if args.include_shuffle_control:
            fold_models.append(("lightgbm_tabular_no_xgb_wf_shuffled_labels", True))
        for model_name, shuffled in fold_models:
            rows, summary, meta, importances = _run_one_model(
                model_name=model_name,
                fold=fold,
                x=x,
                y=y,
                df=df,
                train_idx=train_idx,
                val_idx=val_idx,
                eval_idx=eval_idx,
                categorical_idx=categorical_idx,
                source_parquet=source_parquet,
                top_fracs=top_fracs,
                extra_cost_bps_values=extra_cost_bps_values,
                args=args,
                shuffled=shuffled,
                model_dir=model_dir,
            )
            all_rows.extend(rows)
            summaries.append(summary)
            fold_metas.append(meta)
            for feature, importance in zip(feature_names, importances):
                importance_rows.append({
                    "fold_id": fold.fold_id,
                    "model": model_name,
                    "feature": feature,
                    "importance": int(importance),
                })

    metrics = pd.DataFrame(all_rows)
    metrics_path = out_dir / "walkforward_selective_edge_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    importance = pd.DataFrame(importance_rows)
    importance_path = out_dir / "walkforward_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    if not importance.empty:
        agg_importance = (
            importance[importance["model"] == "lightgbm_tabular_no_xgb_wf"]
            .groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False)
        )
        agg_importance.to_csv(out_dir / "walkforward_feature_importance_mean.csv", index=False)

    summary = {
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "out_dir": str(out_dir),
        "metrics_csv": str(metrics_path),
        "feature_importance_csv": str(importance_path),
        "data_splits": data_splits,
        "folds": [
            {"fold_id": f.fold_id, "start": str(f.start), "end": str(f.end)}
            for f in eval_folds
        ],
        "top_fracs": top_fracs,
        "extra_cost_bps": extra_cost_bps_values,
        "feature_policy": {
            "included": ["snap[7:]", "ctx_cont", "ctx_cat"],
            "n_features": len(feature_names),
            "n_categorical_features": len(categorical_idx),
            "excluded_xgb_derived_snap_fields": [
                "p_long",
                "p_short",
                "p_flat",
                "p_hat",
                "uncertainty_score",
                "margin_top1_top2",
                "entropy",
            ],
        },
        "training_policy": {
            "val_tail_days": int(args.val_tail_days),
            "min_val_rows": int(args.min_val_rows),
            "min_train_rows": int(args.min_train_rows),
            "max_train_rows": int(args.max_train_rows),
            "include_shuffle_control": bool(args.include_shuffle_control),
        },
        "params": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "max_depth": int(args.max_depth),
            "min_child_samples": int(args.min_child_samples),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "n_jobs": int(args.n_jobs),
            "seed": int(args.seed),
        },
        "fold_summaries": summaries,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--source-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--folds", default="", help="Comma-separated id=start:end folds. Empty uses semiyear defaults.")
    ap.add_argument("--top-fracs", default="0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--extra-cost-bps", default="0,10")
    ap.add_argument("--val-tail-days", type=int, default=30)
    ap.add_argument("--min-val-rows", type=int, default=2500)
    ap.add_argument("--min-train-rows", type=int, default=50000)
    ap.add_argument("--max-train-rows", type=int, default=0, help="0 means all eligible prior rows")
    ap.add_argument("--include-shuffle-control", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.035)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--min-child-samples", type=int, default=250)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--early-stopping-rounds", type=int, default=60)
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
