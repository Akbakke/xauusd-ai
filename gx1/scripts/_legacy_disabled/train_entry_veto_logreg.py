#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a simple logistic-regression veto filter on top of a fixed ENTRY base.

Input: existing replay artifacts (trade_journal + chunk_0_data) from a run directory.
Output: model.pkl + metrics.json + coefficients.csv (for transparency).

This is a minimal, post-hoc training step. No trading logic or replay is modified.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


WORKSPACE_ROOT = Path(__file__).parent.parent.parent
DEFAULT_CANONICAL_TRUTH = "gx1/configs/canonical_truth_signal_only.json"


def _resolve_chunk_dir(replay_dir: Path) -> Tuple[Path, Path]:
    """Return (run_dir, chunk_dir) given a replay dir or run dir."""
    replay_dir = replay_dir.expanduser().resolve()
    if replay_dir.name == "chunk_0":
        return replay_dir.parent.parent, replay_dir
    if (replay_dir / "replay" / "chunk_0").exists():
        return replay_dir, replay_dir / "replay" / "chunk_0"
    if (replay_dir / "chunk_0").exists():
        return replay_dir.parent, replay_dir / "chunk_0"
    raise FileNotFoundError(f"Could not resolve chunk_0 under {replay_dir}")


def _find_one(pattern: str, chunk_dir: Path) -> Path:
    matches = sorted(chunk_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing {pattern} in {chunk_dir}")
    return matches[0]


def _resolve_truth_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def _load_canonical_truth(path: Path) -> Dict:
    path = _resolve_truth_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_veto_target(
    df: pd.DataFrame,
    *,
    pnl_bps_max: float = -2.0,
    mfe_mae_ratio_max: float = 0.2,
    mfe_bps_max: float = 1.5,
    bars_in_trade_min: int = 100,
) -> pd.Series:
    """
    Conservative "bad residual trade" label.

    veto = 1 if:
      - pnl_bps <= pnl_bps_max (negative trade), AND
      - any of:
          * mfe_mae_ratio <= mfe_mae_ratio_max (weak forward vs adverse)
          * mfe_bps <= mfe_bps_max (never got meaningful MFE)
          * bars_in_trade >= bars_in_trade_min (slow/dragging trade)
    """
    cond_bad_pnl = df["pnl_bps"] <= pnl_bps_max
    cond_ratio = df["mfe_mae_ratio"] <= mfe_mae_ratio_max
    cond_low_mfe = df["mfe_bps"] <= mfe_bps_max
    cond_long = df["bars_in_trade"] >= bars_in_trade_min
    veto = cond_bad_pnl & (cond_ratio | cond_low_mfe | cond_long)
    return veto.astype(int)


def _session_from_id(session_id: float) -> str:
    mapping = {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}
    try:
        return mapping.get(int(session_id), "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def _prepare_dataset(
    chunk_dir: Path,
    include_entry_margin: bool,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    trade_journal = _find_one("trade_journal_*.parquet", chunk_dir)
    chunk_data = chunk_dir / "chunk_0_data.parquet"
    if not chunk_data.exists():
        raise FileNotFoundError(f"Missing chunk_0_data.parquet in {chunk_dir}")

    trades = pd.read_parquet(trade_journal, engine="pyarrow")
    candles = pd.read_parquet(chunk_data, engine="pyarrow")

    trades["open_ts_utc"] = pd.to_datetime(trades["open_ts_utc"], utc=True, errors="coerce")
    # Drop malformed timestamps and align to 5-minute bars to match replay candles
    trades = trades.dropna(subset=["open_ts_utc"]).copy()
    trades["open_ts_utc"] = trades["open_ts_utc"].dt.floor("5min")
    candles["time"] = pd.to_datetime(candles["time"], utc=True)

    feature_cols = [
        "time",
        "H4_trend_sign_cat",
        "atr_bucket",
        "D1_atr_percentile_252",
        "micro_momentum_3",
        "micro_momentum_5",
        "distance_ema_fast",
        "session_id",
    ]
    missing = [c for c in feature_cols if c not in candles.columns]
    if missing:
        raise RuntimeError(f"Missing required features in chunk_0_data.parquet: {missing}")

    merged = trades.merge(candles[feature_cols], left_on="open_ts_utc", right_on="time", how="left")
    if merged["time"].isna().any():
        missing = int(merged["time"].isna().sum())
        raise RuntimeError(f"Time merge failed for {missing} trades; open_ts_utc not found in chunk_0_data")

    # Ensure session string exists
    if "session" not in merged.columns or merged["session"].isna().any():
        merged["session"] = merged["session_id"].apply(_session_from_id)

    # Build target
    required_target = ["pnl_bps", "mfe_bps", "mfe_mae_ratio", "bars_in_trade"]
    missing_target = [c for c in required_target if c not in merged.columns]
    if missing_target:
        raise RuntimeError(f"Missing target columns in trade_journal: {missing_target}")

    merged["veto_label"] = _build_veto_target(merged)

    # Feature spec
    categorical = ["session", "side", "H4_trend_sign_cat", "atr_bucket"]
    numeric = [
        "D1_atr_percentile_252",
        "micro_momentum_3",
        "micro_momentum_5",
        "distance_ema_fast",
    ]
    include_margin = (
        include_entry_margin
        and "margin_top1_top2" in merged.columns
        and merged["margin_top1_top2"].notna().any()
    )
    if include_margin:
        numeric.append("margin_top1_top2")

    feature_spec = {"categorical": categorical, "numeric": numeric}

    # Drop rows with missing feature values
    keep_cols = categorical + numeric + ["veto_label"]
    clean = merged.dropna(subset=keep_cols).copy()

    return clean, feature_spec


def _build_pipeline(feature_spec: Dict[str, List[str]]) -> Pipeline:
    categorical = feature_spec["categorical"]
    numeric = feature_spec["numeric"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )


def _get_feature_names(pipeline: Pipeline, feature_spec: Dict[str, List[str]]) -> List[str]:
    pre = pipeline.named_steps["preprocess"]
    categorical = feature_spec["categorical"]
    numeric = feature_spec["numeric"]

    cat_features = []
    if categorical:
        ohe = pre.named_transformers_["cat"]
        cat_features = ohe.get_feature_names_out(categorical).tolist()
    return cat_features + numeric


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train entry veto logistic regression from replay artifacts")
    ap.add_argument(
        "--replay-dir",
        required=True,
        help="Replay run dir (containing replay/chunk_0) or chunk_0 dir",
    )
    ap.add_argument(
        "--canonical-truth",
        default=DEFAULT_CANONICAL_TRUTH,
        help="Path to canonical_truth_signal_only.json",
    )
    ap.add_argument(
        "--output-dir",
        default="/home/andre2/GX1_DATA/models/models/entry_v10_ctx_veto",
        help="Root output directory for veto model artifacts",
    )
    ap.add_argument(
        "--run-id",
        default="ENTRY_V10_CTX_VETO_LOGREG_20260322_B1",
        help="Run id name used for output subdir",
    )
    ap.add_argument(
        "--include-entry-margin",
        action="store_true",
        help="Include margin_top1_top2 from trade_journal if present",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = ap.parse_args()

    run_dir, chunk_dir = _resolve_chunk_dir(Path(args.replay_dir))
    canonical_truth_path = _resolve_truth_path(Path(args.canonical_truth))
    truth_cfg = _load_canonical_truth(canonical_truth_path)

    log.info("Resolved run_dir=%s", run_dir)
    log.info("Resolved chunk_dir=%s", chunk_dir)

    df, feature_spec = _prepare_dataset(chunk_dir, include_entry_margin=args.include_entry_margin)

    X = df[feature_spec["categorical"] + feature_spec["numeric"]]
    y = df["veto_label"].astype(int).to_numpy()

    if y.sum() == 0 or y.sum() == len(y):
        raise RuntimeError("Veto label is degenerate (all zeros or all ones); adjust target definition")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=args.seed,
        stratify=y,
    )

    pipeline = _build_pipeline(feature_spec)
    pipeline.fit(X_train, y_train)

    p_train = pipeline.predict_proba(X_train)[:, 1]
    p_val = pipeline.predict_proba(X_val)[:, 1]

    train_metrics = _metrics(y_train, p_train)
    val_metrics = _metrics(y_val, p_val)

    # Feature coefficients
    feature_names = _get_feature_names(pipeline, feature_spec)
    coef = pipeline.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "odds_ratio": np.exp(coef),
    }).sort_values("coef", ascending=False)

    # Output
    out_root = Path(args.output_dir).expanduser().resolve() / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, out_root / "model.pkl")
    coef_df.to_csv(out_root / "coefficients.csv", index=False)

    report = {
        "run_id": args.run_id,
        "replay_run_dir": str(run_dir),
        "chunk_dir": str(chunk_dir),
        "canonical_truth_file": str(canonical_truth_path),
        "entry_base_bundle": truth_cfg.get("canonical_transformer_bundle_dir"),
        "target_definition": {
            "pnl_bps_max": -2.0,
            "mfe_mae_ratio_max": 0.2,
            "mfe_bps_max": 1.5,
            "bars_in_trade_min": 100,
            "formula": "veto = (pnl_bps<=-2.0) AND (mfe_mae_ratio<=0.2 OR mfe_bps<=1.5 OR bars_in_trade>=100)",
        },
        "n_samples": int(len(y)),
        "class_balance": {
            "veto_1": int(y.sum()),
            "veto_0": int(len(y) - y.sum()),
            "veto_rate": float(y.mean()),
        },
        "class_balance_train": {
            "veto_1": int(y_train.sum()),
            "veto_0": int(len(y_train) - y_train.sum()),
            "veto_rate": float(y_train.mean()),
        },
        "class_balance_val": {
            "veto_1": int(y_val.sum()),
            "veto_0": int(len(y_val) - y_val.sum()),
            "veto_rate": float(y_val.mean()),
        },
        "features": feature_spec,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "notes": {
            "include_entry_margin": "margin_top1_top2" in feature_spec["numeric"],
        },
    }

    (out_root / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Console summary
    log.info("Saved model to: %s", out_root / "model.pkl")
    log.info("Saved coefficients to: %s", out_root / "coefficients.csv")
    log.info("Saved metrics to: %s", out_root / "metrics.json")

    log.info("Train metrics: %s", train_metrics)
    log.info("Val metrics: %s", val_metrics)

    log.info("Top positive coefficients:\n%s", coef_df.head(10).to_string(index=False))
    log.info("Top negative coefficients:\n%s", coef_df.tail(10).to_string(index=False))

    # Print classification report for visibility
    y_val_pred = (p_val >= 0.5).astype(int)
    log.info("Validation classification report:\n%s", classification_report(y_val, y_val_pred, digits=3))


if __name__ == "__main__":
    main()
