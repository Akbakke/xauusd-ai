"""Train and evaluate a no-XGB tabular Entry baseline.

This is a Phase 2 gate for the 2026-06-27 Entry plan. It trains a LightGBM
multiclass classifier on existing V10 dataset columns while excluding the XGB
signal bridge probability fields. It reuses the selective edge/PnL evaluator so
the result is directly comparable with ET and XGB-only artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from gx1.audit.entry_transformer_feature_audit import _stack_list_column
from gx1.contracts.signal_bridge_v3 import ORDERED_SEQ_FIELDS_V3
from gx1.scripts.evaluate_entry_selective_edge_v1 import (
    _evaluate_predictions,
    _json_default,
    _load_base_frame,
    _parse_float_list,
    _split_files,
    _spread_aware_path,
)


XGB_SIGNAL_FIELD_COUNT = 7
XGB_DERIVED_FIELD_NAMES = set(ORDERED_SEQ_FIELDS_V3[:XGB_SIGNAL_FIELD_COUNT])
SUSPICIOUS_NAME_PARTS = ("xgb", "p_long", "p_short", "p_flat", "prob", "signal_bridge")


def _manifest_for_split(parquet_path: Path) -> Path | None:
    candidate = parquet_path.with_suffix(".manifest.json")
    return candidate if candidate.exists() else None


def _ctx_names_from_manifest(parquet_path: Path) -> tuple[list[str], list[str]]:
    manifest_path = _manifest_for_split(parquet_path)
    if manifest_path is None:
        return [], []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_contract = manifest.get("feature_contract") if isinstance(manifest.get("feature_contract"), dict) else {}
    ctx_contract = None
    if isinstance(manifest.get("extra"), dict):
        ctx_contract = manifest["extra"].get("ctx_contract")
    if isinstance(manifest.get("build_metadata"), dict):
        ctx_contract = ctx_contract or manifest["build_metadata"].get("ctx_contract")
    if not isinstance(ctx_contract, dict):
        ctx_contract = feature_contract
    ctx_cont_names = list(ctx_contract.get("ctx_cont_names") or feature_contract.get("ctx_cont_names") or [])
    ctx_cat_names = list(ctx_contract.get("ctx_cat_names") or feature_contract.get("ctx_cat_names") or [])
    return ctx_cont_names, ctx_cat_names


def _selected_feature_names(parquet_path: Path) -> tuple[list[str], list[int]]:
    ctx_cont_names, ctx_cat_names = _ctx_names_from_manifest(parquet_path)
    snap_names = list(ORDERED_SEQ_FIELDS_V3[XGB_SIGNAL_FIELD_COUNT:])
    selected = [f"snap.{name}" for name in snap_names]
    selected.extend([f"ctx_cont.{name}" for name in ctx_cont_names] if ctx_cont_names else [])
    selected.extend([f"ctx_cat.{name}" for name in ctx_cat_names] if ctx_cat_names else [])
    categorical_idx = list(range(len(snap_names) + len(ctx_cont_names), len(selected)))
    return selected, categorical_idx


def _check_no_xgb_feature_names(feature_names: list[str]) -> None:
    suspicious = []
    for name in feature_names:
        low = name.lower()
        if any(part in low for part in SUSPICIOUS_NAME_PARTS):
            suspicious.append(name)
    if suspicious:
        raise RuntimeError(f"refusing no-XGB tabular baseline with suspicious feature names: {suspicious[:20]}")


def _load_tabular_xy(
    parquet_path: Path,
    *,
    max_rows: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    df = pd.read_parquet(parquet_path, columns=["snap", "ctx_cont", "ctx_cat", "y_direction"])
    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        sample = df.sample(n=max_rows, random_state=seed).sort_index()
        df = sample.reset_index(drop=True)
    snap = _stack_list_column(df["snap"], np.float32)[:, XGB_SIGNAL_FIELD_COUNT:]
    ctx_cont = _stack_list_column(df["ctx_cont"], np.float32)
    ctx_cat = _stack_list_column(df["ctx_cat"], np.int64).astype(np.float32)
    y = df["y_direction"].to_numpy(np.int64)

    feature_names, categorical_idx = _selected_feature_names(parquet_path)
    if not feature_names:
        snap_names = [f"snap.{name}" for name in ORDERED_SEQ_FIELDS_V3[XGB_SIGNAL_FIELD_COUNT:]]
        ctx_cont_names = [f"ctx_cont.{i}" for i in range(ctx_cont.shape[1])]
        ctx_cat_names = [f"ctx_cat.{i}" for i in range(ctx_cat.shape[1])]
        feature_names = snap_names + ctx_cont_names + ctx_cat_names
        categorical_idx = list(range(len(snap_names) + len(ctx_cont_names), len(feature_names)))
    if len(feature_names) != snap.shape[1] + ctx_cont.shape[1] + ctx_cat.shape[1]:
        snap_names = [f"snap.{name}" for name in ORDERED_SEQ_FIELDS_V3[XGB_SIGNAL_FIELD_COUNT:]]
        ctx_cont_names = [f"ctx_cont.{i}" for i in range(ctx_cont.shape[1])]
        ctx_cat_names = [f"ctx_cat.{i}" for i in range(ctx_cat.shape[1])]
        feature_names = snap_names + ctx_cont_names + ctx_cat_names
        categorical_idx = list(range(len(snap_names) + len(ctx_cont_names), len(feature_names)))
    _check_no_xgb_feature_names(feature_names)

    x = np.concatenate([snap, ctx_cont, ctx_cat], axis=1).astype(np.float32, copy=False)
    return x, y, feature_names, categorical_idx


def _load_tabular_x(parquet_path: Path, feature_names: list[str]) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=["snap", "ctx_cont", "ctx_cat"])
    snap = _stack_list_column(df["snap"], np.float32)[:, XGB_SIGNAL_FIELD_COUNT:]
    ctx_cont = _stack_list_column(df["ctx_cont"], np.float32)
    ctx_cat = _stack_list_column(df["ctx_cat"], np.int64).astype(np.float32)
    x = np.concatenate([snap, ctx_cont, ctx_cat], axis=1).astype(np.float32, copy=False)
    if x.shape[1] != len(feature_names):
        raise RuntimeError(f"feature shape mismatch for {parquet_path}: x={x.shape[1]} names={len(feature_names)}")
    return x


def _train_lightgbm(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    categorical_idx: list[int],
    seed: int,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    min_child_samples: int,
    n_jobs: int,
    early_stopping_rounds: int,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    callbacks = [
        lgb.early_stopping(early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=50),
    ]
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="multi_logloss",
        categorical_feature=categorical_idx,
        callbacks=callbacks,
    )
    return model


def _predict_proba(model: lgb.LGBMClassifier, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise RuntimeError(f"unexpected probability shape: {probs.shape}")
    row_sum = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, np.maximum(row_sum, 1e-12))


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    top_fracs = _parse_float_list(args.top_fracs)
    extra_cost_bps_values = _parse_float_list(args.extra_cost_bps)
    files = _split_files(dataset_dir, sorted(set(["train", "val", *splits])))

    max_train_rows = None if int(args.max_train_rows) <= 0 else int(args.max_train_rows)
    x_train, y_train, feature_names, categorical_idx = _load_tabular_xy(
        files["train"],
        max_rows=max_train_rows,
        seed=int(args.seed),
    )
    shuffled_train_labels = bool(args.shuffle_train_labels)
    if shuffled_train_labels:
        rng = np.random.default_rng(int(args.seed))
        y_train = np.array(y_train, copy=True)
        rng.shuffle(y_train)
    x_val, y_val, _, _ = _load_tabular_xy(files["val"], max_rows=None, seed=int(args.seed))
    model = _train_lightgbm(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
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

    model_path = out_dir / "lightgbm_no_xgb_model.joblib"
    joblib.dump(model, model_path)

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for split in splits:
        parquet_path = files[split]
        df = _load_base_frame(parquet_path)
        x = x_val if split == "val" else _load_tabular_x(parquet_path, feature_names)
        probs = _predict_proba(model, x)
        path = _spread_aware_path(
            source_parquet=source_parquet,
            sample_times=df["time"],
            horizons=df["label_horizon_bars"].to_numpy(np.int64),
        )
        model_name = "lightgbm_tabular_no_xgb_shuffled_labels" if shuffled_train_labels else "lightgbm_tabular_no_xgb"
        rows, summary = _evaluate_predictions(
            split=split,
            model_name=model_name,
            df=df,
            probs=probs,
            path=path,
            top_fracs=top_fracs,
            extra_cost_bps_values=extra_cost_bps_values,
        )
        all_rows.extend(rows)
        summaries.append(summary)

    metrics = pd.DataFrame(all_rows)
    metrics_path = out_dir / "selective_edge_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_path = out_dir / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    summary = {
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "metrics_csv": str(metrics_path),
        "feature_importance_csv": str(importance_path),
        "model": "lightgbm_tabular_no_xgb_shuffled_labels" if shuffled_train_labels else "lightgbm_tabular_no_xgb",
        "splits": splits,
        "top_fracs": top_fracs,
        "extra_cost_bps": extra_cost_bps_values,
        "feature_policy": {
            "included": ["snap[7:]", "ctx_cont", "ctx_cat"],
            "excluded_xgb_derived_snap_fields": ORDERED_SEQ_FIELDS_V3[:XGB_SIGNAL_FIELD_COUNT],
            "n_features": len(feature_names),
            "n_categorical_features": len(categorical_idx),
        },
        "train": {
            "train_file": str(files["train"]),
            "val_file": str(files["val"]),
            "n_train_rows_used": int(len(y_train)),
            "n_val_rows": int(len(y_val)),
            "max_train_rows": max_train_rows,
            "shuffled_train_labels": shuffled_train_labels,
            "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
            "classes": [int(x) for x in model.classes_],
        },
        "params": model.get_params(),
        "summaries": summaries,
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
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--top-fracs", default="0.01,0.02,0.05,0.10,0.20")
    ap.add_argument("--extra-cost-bps", default="0,10")
    ap.add_argument("--max-train-rows", type=int, default=0, help="0 means all train rows")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-estimators", type=int, default=700)
    ap.add_argument("--learning-rate", type=float, default=0.035)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--max-depth", type=int, default=-1)
    ap.add_argument("--min-child-samples", type=int, default=250)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--early-stopping-rounds", type=int, default=75)
    ap.add_argument("--shuffle-train-labels", action="store_true")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
