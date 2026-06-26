"""Model-based diagnostics for Entry Transformer inputs.

This is an audit/research tool, not a production trainer. It builds a compact
tabular view from the exact V10 training dataset surfaces and answers:

* Which active feature families carry learnable y_direction signal?
* Do audit-only derived candidates add signal?
* Is the primary 3-bar direction target aligned with longer forward horizons?

The tool intentionally does not mutate model bundles or train/serve contracts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.audit.entry_transformer_feature_audit import (
    DERIVED_CANDIDATE_NAMES,
    _derived_candidate_matrix,
    _family,
    _safe_float,
    _split_files,
    _stack_list_column,
)
from gx1.contracts.signal_bridge_v3 import (
    ORDERED_CTX_CAT_NAMES_V3,
    ORDERED_CTX_CONT_NAMES_V3,
    ORDERED_SEQ_FIELDS_V3,
)
from gx1.features.htf_features import (
    MULTI_TF_PER_BAR_FEATURES_V2,
    MULTI_TF_SHIFT,
    load_multi_tf_v2_cache,
)


DEFAULT_REBUILD = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260626_spreadfix"
)
DEFAULT_DATASET_DIR = DEFAULT_REBUILD / "v10_dataset_6yr"
DEFAULT_MTF_CACHE = DEFAULT_REBUILD / "MULTI_TF_V2_CACHE"
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/entry_transformer_feature_diagnostics_20260626_spreadfix")

TF_ORDER = ("M5", "M15", "H1", "H4", "D1")
FORECAST_RET_COLUMNS = ("y_forecast_ret_K1", "y_forecast_ret_K5", "y_forecast_ret_K12", "y_forecast_ret_K24")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    source: str
    base_feature: str
    group: str
    active_contract: bool


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _selected_indices(total_rows: int, max_rows: int, seed: int) -> np.ndarray | None:
    if max_rows <= 0 or max_rows >= total_rows:
        return None
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(total_rows, size=int(max_rows), replace=False).astype(np.int64))


def _local_indices_for_batch(selected: np.ndarray | None, offset: int, batch_len: int) -> np.ndarray | slice | None:
    if selected is None:
        return slice(None)
    end = int(offset + batch_len)
    left = int(np.searchsorted(selected, offset, side="left"))
    right = int(np.searchsorted(selected, end, side="left"))
    if right <= left:
        return None
    return selected[left:right] - int(offset)


def _as_ns_utc(times: pd.Series) -> np.ndarray:
    return pd.to_datetime(times, utc=True).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _mtf_current_matrix(times_ns: np.ndarray, mtf_cache: dict[str, pd.DataFrame] | None) -> tuple[np.ndarray, list[FeatureSpec]]:
    specs: list[FeatureSpec] = []
    cols: list[np.ndarray] = []
    if not mtf_cache:
        return np.empty((len(times_ns), 0), dtype=np.float32), specs
    for tf in TF_ORDER:
        df = mtf_cache.get(tf)
        if df is None:
            continue
        ts = np.asarray(df.attrs.get("ts_int64"), dtype=np.int64)
        feats = np.asarray(df.attrs.get("feats_np"), dtype=np.float32)
        if ts.ndim != 1 or feats.ndim != 2 or feats.shape[1] != len(MULTI_TF_PER_BAR_FEATURES_V2):
            raise RuntimeError(f"bad MTF cache shape for {tf}: ts={ts.shape} feats={feats.shape}")
        cutoffs = np.asarray(times_ns, dtype=np.int64) - int(MULTI_TF_SHIFT[tf].value)
        right = np.searchsorted(ts, cutoffs, side="right") - 1
        valid = right >= 0
        safe = np.clip(right, 0, len(ts) - 1)
        current = np.where(valid[:, None], feats[safe], 0.0).astype(np.float32, copy=False)
        cols.append(current)
        for name in MULTI_TF_PER_BAR_FEATURES_V2:
            specs.append(
                FeatureSpec(
                    name=f"mtf_{tf.lower()}__{name}",
                    source=f"mtf_{tf.lower()}_current",
                    base_feature=str(name),
                    group=f"mtf_current:{_family(str(name))}",
                    active_contract=True,
                )
            )
    if not cols:
        return np.empty((len(times_ns), 0), dtype=np.float32), specs
    return np.concatenate(cols, axis=1), specs


def _seq_summary_matrix(seq: np.ndarray) -> tuple[np.ndarray, list[FeatureSpec]]:
    windows = (12, 48, 96)
    cols: list[np.ndarray] = []
    specs: list[FeatureSpec] = []
    for window in windows:
        tail = seq[:, -window:, :]
        cols.append(tail.mean(axis=1))
        for name in ORDERED_SEQ_FIELDS_V3:
            specs.append(
                FeatureSpec(
                    name=f"seq_mean{window}__{name}",
                    source=f"seq_mean{window}",
                    base_feature=str(name),
                    group=f"seq_summary:{_family(str(name))}",
                    active_contract=True,
                )
            )
    delta12 = seq[:, -1, :] - seq[:, -13, :]
    cols.append(delta12)
    for name in ORDERED_SEQ_FIELDS_V3:
        specs.append(
            FeatureSpec(
                name=f"seq_delta12__{name}",
                source="seq_delta12",
                base_feature=str(name),
                group=f"seq_summary:{_family(str(name))}",
                active_contract=True,
            )
        )
    return np.concatenate(cols, axis=1).astype(np.float32, copy=False), specs


def _batch_feature_matrix(
    pdf: pd.DataFrame,
    *,
    include_seq_summary: bool,
    include_derived: bool,
    mtf_cache: dict[str, pd.DataFrame] | None,
) -> tuple[np.ndarray, np.ndarray, list[FeatureSpec], dict[str, np.ndarray]]:
    seq = _stack_list_column(pdf["seq"], np.float32)
    snap = _stack_list_column(pdf["snap"], np.float32)
    ctx = _stack_list_column(pdf["ctx_cont"], np.float32)
    cat = _stack_list_column(pdf["ctx_cat"], np.int64).astype(np.float32, copy=False)
    y = pd.to_numeric(pdf["y_direction"], errors="coerce").to_numpy(dtype=np.int64)
    times_ns = _as_ns_utc(pdf["time"])

    parts: list[np.ndarray] = []
    specs: list[FeatureSpec] = []

    parts.append(snap)
    specs.extend(
        FeatureSpec(f"snap__{name}", "snap", str(name), f"snap:{_family(str(name))}", True)
        for name in ORDERED_SEQ_FIELDS_V3
    )

    if include_seq_summary:
        seq_summary, seq_specs = _seq_summary_matrix(seq)
        parts.append(seq_summary)
        specs.extend(seq_specs)

    parts.append(ctx)
    specs.extend(
        FeatureSpec(f"ctx__{name}", "ctx_cont", str(name), f"ctx:{_family(str(name))}", True)
        for name in ORDERED_CTX_CONT_NAMES_V3
    )

    parts.append(cat)
    specs.extend(
        FeatureSpec(f"ctx_cat__{name}", "ctx_cat", str(name), "ctx_cat", True)
        for name in ORDERED_CTX_CAT_NAMES_V3
    )

    if include_derived:
        derived = _derived_candidate_matrix(seq, snap, ctx)
        parts.append(derived)
        specs.extend(
            FeatureSpec(
                f"derived__{name}",
                "derived_candidate",
                str(name),
                f"derived:{_family(str(name))}",
                str(name) in ORDERED_CTX_CONT_NAMES_V3,
            )
            for name in DERIVED_CANDIDATE_NAMES
        )

    mtf, mtf_specs = _mtf_current_matrix(times_ns, mtf_cache)
    if mtf.shape[1]:
        parts.append(mtf)
        specs.extend(mtf_specs)

    targets: dict[str, np.ndarray] = {}
    for col in ("label_horizon_bars", "path_quality_horizon_bars", *FORECAST_RET_COLUMNS):
        if col in pdf.columns:
            targets[col] = pd.to_numeric(pdf[col], errors="coerce").to_numpy(dtype=np.float64)

    x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, y, specs, targets


def _load_split_matrix(
    path: Path,
    *,
    max_rows: int,
    seed: int,
    batch_size: int,
    include_seq_summary: bool,
    include_derived: bool,
    mtf_cache: dict[str, pd.DataFrame] | None,
) -> tuple[np.ndarray, np.ndarray, list[FeatureSpec], pd.DataFrame]:
    pf = pq.ParquetFile(path)
    selected = _selected_indices(pf.metadata.num_rows, max_rows, seed)
    base_cols = ["time", "seq", "snap", "ctx_cont", "ctx_cat", "y_direction"]
    schema = set(pf.schema_arrow.names)
    target_cols = [c for c in ("label_horizon_bars", "path_quality_horizon_bars", *FORECAST_RET_COLUMNS) if c in schema]
    cols = base_cols + target_cols

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    target_parts: list[pd.DataFrame] = []
    specs: list[FeatureSpec] | None = None
    offset = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        pdf = batch.to_pandas()
        local = _local_indices_for_batch(selected, offset, len(pdf))
        offset += len(pdf)
        if local is None:
            continue
        if not isinstance(local, slice):
            pdf = pdf.iloc[local].reset_index(drop=True)
        x, y, batch_specs, targets = _batch_feature_matrix(
            pdf,
            include_seq_summary=include_seq_summary,
            include_derived=include_derived,
            mtf_cache=mtf_cache,
        )
        if specs is None:
            specs = batch_specs
        elif [s.name for s in specs] != [s.name for s in batch_specs]:
            raise RuntimeError("feature spec changed across batches")
        x_parts.append(x)
        y_parts.append(y)
        if targets:
            target_parts.append(pd.DataFrame(targets))
    if specs is None:
        raise RuntimeError(f"no rows loaded from {path}")
    target_df = pd.concat(target_parts, ignore_index=True) if target_parts else pd.DataFrame()
    return np.vstack(x_parts), np.concatenate(y_parts), specs, target_df


def _metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict[str, float | int | None]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

    out: dict[str, float | int | None] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
    }
    try:
        out["log_loss"] = float(log_loss(y_true, proba, labels=[0, 1, 2]))
    except Exception:
        out["log_loss"] = None
    return out


def _make_model(kind: str, *, seed: int, n_jobs: int, xgb_estimators: int):
    if kind == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=int(xgb_estimators),
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            min_child_weight=20,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=int(seed),
            n_jobs=int(n_jobs),
        )
    if kind == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=180,
            max_depth=4,
            learning_rate=0.05,
            l2_regularization=1.0,
            min_samples_leaf=200,
            early_stopping=True,
            random_state=int(seed),
        )
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1500, C=0.5, class_weight="balanced", random_state=int(seed), n_jobs=int(n_jobs)),
        )
    raise ValueError(f"unknown model kind: {kind}")


def _predict_proba(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x), dtype=np.float64)
        proba = np.clip(proba, 0.0, 1.0)
        row_sum = proba.sum(axis=1, keepdims=True)
        bad = (~np.isfinite(row_sum)) | (row_sum <= 0.0)
        if np.any(bad):
            proba[bad[:, 0], :] = 1.0 / max(proba.shape[1], 1)
            row_sum = proba.sum(axis=1, keepdims=True)
        return proba / row_sum
    raise RuntimeError(f"model has no predict_proba: {type(model)}")


def _fit_eval(
    *,
    model_kind: str,
    feature_set: str,
    columns: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_jobs: int,
    xgb_estimators: int,
) -> tuple[list[dict[str, Any]], Any]:
    model = _make_model(model_kind, seed=seed, n_jobs=n_jobs, xgb_estimators=xgb_estimators)
    model.fit(x_train[:, columns], y_train)
    rows: list[dict[str, Any]] = []
    for split, x, y in (("val", x_val, y_val), ("test", x_test, y_test)):
        proba = _predict_proba(model, x[:, columns])
        pred = np.argmax(proba, axis=1)
        rows.append(
            {
                "model": model_kind,
                "feature_set": feature_set,
                "n_features": int(len(columns)),
                "split": split,
                **_metrics(y, proba, pred),
            }
        )
    return rows, model


def _majority_metrics(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> list[dict[str, Any]]:
    vals, counts = np.unique(y_train, return_counts=True)
    majority = int(vals[np.argmax(counts)])
    rows: list[dict[str, Any]] = []
    for split, y in (("val", y_val), ("test", y_test)):
        proba = np.zeros((len(y), 3), dtype=np.float64)
        proba[:, majority] = 1.0
        pred = np.full(len(y), majority, dtype=np.int64)
        rows.append({"model": "majority", "feature_set": "none", "n_features": 0, "split": split, **_metrics(y, proba, pred)})
    return rows


def _spec_frame(specs: Sequence[FeatureSpec]) -> pd.DataFrame:
    return pd.DataFrame([s.__dict__ for s in specs])


def _horizon_rows(split: str, y: np.ndarray, targets: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if targets.empty:
        return rows
    for col in ("label_horizon_bars", "path_quality_horizon_bars"):
        if col in targets:
            s = pd.to_numeric(targets[col], errors="coerce")
            rows.append(
                {
                    "split": split,
                    "target": col,
                    "kind": "horizon_constant_check",
                    "n": int(s.notna().sum()),
                    "mean": _safe_float(float(s.mean())),
                    "std": _safe_float(float(s.std(ddof=0))),
                    "min": _safe_float(float(s.min())),
                    "max": _safe_float(float(s.max())),
                    "unique_count": int(s.nunique(dropna=True)),
                }
            )
    for col in FORECAST_RET_COLUMNS:
        if col not in targets:
            continue
        r = pd.to_numeric(targets[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(r)
        if not finite.any():
            continue
        sign_pred = np.full(len(r), 2, dtype=np.int64)
        sign_pred[r > 0.0] = 0
        sign_pred[r < 0.0] = 1
        rows.append(
            {
                "split": split,
                "target": col,
                "kind": "forward_return_sign_vs_y_direction",
                "n": int(finite.sum()),
                "mean": _safe_float(float(np.nanmean(r))),
                "std": _safe_float(float(np.nanstd(r))),
                "min": _safe_float(float(np.nanmin(r))),
                "max": _safe_float(float(np.nanmax(r))),
                "sign_accuracy_vs_y_direction": float((sign_pred[finite] == y[finite]).mean()),
                "long_rate_if_ret_positive": _safe_float(float((y[(r > 0.0) & finite] == 0).mean())) if np.any((r > 0.0) & finite) else None,
                "short_rate_if_ret_negative": _safe_float(float((y[(r < 0.0) & finite] == 1).mean())) if np.any((r < 0.0) & finite) else None,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    split_files = _split_files(dataset_dir)
    mtf_cache = load_multi_tf_v2_cache(Path(args.mtf_cache).expanduser()) if args.mtf_cache else None

    x_train, y_train, specs, train_targets = _load_split_matrix(
        split_files["train"],
        max_rows=int(args.max_train_rows),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        include_seq_summary=not args.no_seq_summary,
        include_derived=not args.no_derived,
        mtf_cache=mtf_cache,
    )
    x_val, y_val, val_specs, val_targets = _load_split_matrix(
        split_files["val"],
        max_rows=int(args.max_eval_rows),
        seed=int(args.seed) + 1,
        batch_size=int(args.batch_size),
        include_seq_summary=not args.no_seq_summary,
        include_derived=not args.no_derived,
        mtf_cache=mtf_cache,
    )
    x_test, y_test, test_specs, test_targets = _load_split_matrix(
        split_files["test"],
        max_rows=int(args.max_eval_rows),
        seed=int(args.seed) + 2,
        batch_size=int(args.batch_size),
        include_seq_summary=not args.no_seq_summary,
        include_derived=not args.no_derived,
        mtf_cache=mtf_cache,
    )
    if [s.name for s in specs] != [s.name for s in val_specs] or [s.name for s in specs] != [s.name for s in test_specs]:
        raise RuntimeError("feature specs differ across splits")

    spec_df = _spec_frame(specs)
    spec_df.to_csv(out_dir / "diagnostic_feature_manifest.csv", index=False)

    metric_rows = _majority_metrics(y_train, y_val, y_test)
    all_cols = np.arange(x_train.shape[1], dtype=np.int64)
    metric_rows_full, full_model = _fit_eval(
        model_kind=args.model,
        feature_set="all",
        columns=all_cols,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        xgb_estimators=int(args.xgb_estimators),
    )
    metric_rows.extend(metric_rows_full)

    family_rows: list[dict[str, Any]] = []
    group_to_cols = {
        str(group): spec_df.index[spec_df["group"].astype(str) == str(group)].to_numpy(dtype=np.int64)
        for group in sorted(spec_df["group"].astype(str).unique())
    }
    for group, cols in group_to_cols.items():
        if len(cols) < int(args.min_family_features):
            continue
        rows, _ = _fit_eval(
            model_kind=args.model,
            feature_set=f"family_only:{group}",
            columns=cols,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            seed=int(args.seed),
            n_jobs=int(args.n_jobs),
            xgb_estimators=max(40, int(args.xgb_estimators) // 2),
        )
        metric_rows.extend(rows)
        family_rows.extend(rows)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "model_feature_set_metrics.csv", index=False)

    importance = np.asarray(getattr(full_model, "feature_importances_", np.zeros(x_train.shape[1])), dtype=np.float64)
    if importance.shape[0] != x_train.shape[1]:
        importance = np.zeros(x_train.shape[1], dtype=np.float64)
    imp_df = spec_df.copy()
    imp_df["importance"] = importance
    imp_df.sort_values("importance", ascending=False).to_csv(out_dir / "xgb_feature_importance.csv", index=False)
    fam_imp = (
        imp_df.groupby("group", dropna=False)
        .agg(n_features=("name", "count"), importance_sum=("importance", "sum"), importance_mean=("importance", "mean"))
        .sort_values("importance_sum", ascending=False)
        .reset_index()
    )
    fam_imp.to_csv(out_dir / "family_importance.csv", index=False)

    horizon = pd.DataFrame(
        _horizon_rows("train_sample", y_train, train_targets)
        + _horizon_rows("val", y_val, val_targets)
        + _horizon_rows("test", y_test, test_targets)
    )
    horizon.to_csv(out_dir / "horizon_label_diagnostics.csv", index=False)

    best_family_test = (
        metrics[(metrics["split"] == "test") & metrics["feature_set"].astype(str).str.startswith("family_only:")]
        .sort_values("accuracy", ascending=False)
        .head(12)
        .to_dict(orient="records")
    )
    top_features = imp_df.sort_values("importance", ascending=False).head(30).to_dict(orient="records")
    summary = {
        "dataset_dir": str(dataset_dir),
        "mtf_cache": str(args.mtf_cache) if args.mtf_cache else None,
        "out_dir": str(out_dir),
        "model": args.model,
        "samples": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
            "n_features": int(x_train.shape[1]),
        },
        "label_counts": {
            "train": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "val": {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
            "test": {str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        },
        "all_feature_metrics": metrics[(metrics["feature_set"] == "all")].to_dict(orient="records"),
        "best_family_test_accuracy": best_family_test,
        "top_features": top_features,
        "outputs": {
            "diagnostic_feature_manifest": str(out_dir / "diagnostic_feature_manifest.csv"),
            "model_feature_set_metrics": str(out_dir / "model_feature_set_metrics.csv"),
            "xgb_feature_importance": str(out_dir / "xgb_feature_importance.csv"),
            "family_importance": str(out_dir / "family_importance.csv"),
            "horizon_label_diagnostics": str(out_dir / "horizon_label_diagnostics.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--mtf-cache", default=str(DEFAULT_MTF_CACHE))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--model", choices=("xgb", "hgb", "logreg"), default="xgb")
    ap.add_argument("--max-train-rows", type=int, default=80_000)
    ap.add_argument("--max-eval-rows", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--xgb-estimators", type=int, default=120)
    ap.add_argument("--min-family-features", type=int, default=1)
    ap.add_argument("--no-seq-summary", action="store_true")
    ap.add_argument("--no-derived", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
