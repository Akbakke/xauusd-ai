#!/usr/bin/env python
"""
train_exit_critic_v1.py

Trener en ExitCritic-modell på exit-datasettet fra build_exit_dataset_v1.py.

Antakelser om datasettet (Parquet):
- Én rad per trade (ikke per bar).
- Kolonner (typisk):
    - trade_id
    - entry_time (datetime64[ns])
    - exit_time  (datetime64[ns])
    - exit_reason (str)
    - reward_bps (float)
    - max_favorable_excursion_bps (float)
    - max_adverse_excursion_bps (float)
    - bars_held (int)
    - label (int)  # MÅ finnes – target for ExitCritic
    - feat_* (float)  # features fra entry/exit-snapshots

Scriptet:
- Leser Parquet-datasett.
- Finner feature-kolonner automatisk (numeric cols, ekskl. id/tid/label).
- Splitter i train/val (time-based hvis entry_time finnes, ellers random).
- Trener en XGBoost-klassifiserer.
- Lagrer modell + metadata + enkel treningsrapport (Markdown).
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as e:
    raise SystemExit(
        "xgboost er ikke installert. Installer med:\n"
        "  pip install xgboost\n"
        f"Original feil: {e}"
    )

from sklearn.metrics import classification_report, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ExitCritic V1 (XGBoost).")

    parser.add_argument(
        "--dataset",
        type=str,
        default="data/rl/exit_critic/exit_critic_dataset_v1.parquet",
        help="Path til exit-critic Parquet-datasett (default: data/rl/exit_critic/exit_critic_dataset_v1.parquet)."
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="exit_label_id",
        help="Navn på label-kolonnen i datasettet (default: exit_label_id)."
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/exit_critic/exit_critic_xgb_v1.json",
        help="Filsti for lagret XGBoost-modell (JSON)."
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default="models/exit_critic/exit_critic_xgb_v1.meta.json",
        help="Filsti for metadata om modellen (JSON)."
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/rl/exit_critic/EXIT_CRITIC_XGB_V1_REPORT.md",
        help="Filsti for treningsrapport (Markdown, default: reports/rl/exit_critic/EXIT_CRITIC_XGB_V1_REPORT.md)."
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Andel av data som brukes til validering (0–1)."
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for shuffle hvis vi ikke bruker time-based split."
    )
    parser.add_argument(
        "--use-time-split",
        action="store_true",
        help="Bruk time-based split basert på entry_time hvis tilgjengelig."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="XGBoost max_depth."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Antall trær i XGBoost."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="Learning rate (eta) for XGBoost."
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.9,
        help="Subsample-rate for rader."
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="Subsample-rate for features."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for XGBoost. 'cuda' krever GPU og riktig CUDA-setup."
    )

    return parser.parse_args()


def ensure_dir(path: str) -> None:
    if not path:
        return
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def infer_feature_columns(
    df: pd.DataFrame,
    label_col: str,
    extra_exclude: List[str] = None,
) -> List[str]:
    """
    Velger feature-kolonner:
    - Numeric dtype (int/float/bool)
    - Ekskluderer id/tidsfelt/label som ikke skal inn i modellen.
    """
    exclude = {
        label_col,
        "trade_id",
        "entry_time",
        "exit_time",
        "exit_reason",
    }
    if extra_exclude:
        exclude.update(extra_exclude)

    feature_cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        # Skip categorical columns
        if pd.api.types.is_categorical_dtype(df[col]):
            continue
        # bruk kun rene numeriske features
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)
        except (TypeError, ValueError):
            # Skip non-numeric types
            continue

    if not feature_cols:
        raise ValueError(
            "Fant ingen feature-kolonner. "
            "Sjekk at build_exit_dataset_v1 faktisk lager numeriske feat_* kolonner."
        )

    return sorted(feature_cols)


def time_based_split(
    df: pd.DataFrame,
    label_col: str,
    val_frac: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splitt på tid: sørg for at val-settet er siste del av perioden.
    Forutsetter at entry_time finnes.
    """
    if "entry_time" not in df.columns:
        raise ValueError("Kan ikke bruke time-based split: 'entry_time' mangler i datasettet.")

    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    n = len(df_sorted)
    split_idx = int(n * (1.0 - val_frac))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(f"Ugyldig split-indeks ({split_idx}) for n={n}, val_frac={val_frac}")

    train_df = df_sorted.iloc[:split_idx].copy()
    val_df = df_sorted.iloc[split_idx:].copy()
    return train_df, val_df


def random_split(
    df: pd.DataFrame,
    label_col: str,
    val_frac: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Random split (fallback hvis vi ikke bruker tid).
    """
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    split_idx = int(n * (1.0 - val_frac))
    train_df = df_shuffled.iloc[:split_idx].copy()
    val_df = df_shuffled.iloc[split_idx:].copy()
    return train_df, val_df


def make_model(
    num_classes: int,
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    device: str,
) -> xgb.XGBClassifier:
    """
    Lager XGBoost-klassifiserer for ExitCritic.
    Klassifiserer støtter både binary og multi-class via num_class.
    """
    if num_classes <= 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
    else:
        objective = "multi:softprob"
        eval_metric = "mlogloss"  # Use mlogloss for multi-class

    if device == "cuda":
        tree_method = "gpu_hist"
        predictor = "gpu_predictor"
    else:
        tree_method = "hist"
        predictor = "auto"

    model = xgb.XGBClassifier(
        objective=objective,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        tree_method=tree_method,
        predictor=predictor,
        eval_metric=eval_metric,
        n_jobs=-1,  # bruk alle tilgjengelige cores
    )
    return model


def train_and_eval(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    val_frac: float,
    use_time_split: bool,
    seed: int,
    model_kwargs: dict,
):
    """
    Trener modell og returnerer:
    - model
    - metrics dict
    - train/val feature-kolonner (for metadata)
    """
    # dropp rader uten label
    df = df.dropna(subset=[label_col]).copy()

    # sørg for int-labels
    df[label_col] = df[label_col].astype(int)

    # split
    if use_time_split and "entry_time" in df.columns:
        train_df, val_df = time_based_split(df, label_col, val_frac)
        split_type = "time"
    else:
        train_df, val_df = random_split(df, label_col, val_frac, seed)
        split_type = "random"

    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values

    X_val = val_df[feature_cols].values
    y_val = val_df[label_col].values

    num_classes = int(df[label_col].nunique())
    model = make_model(num_classes=num_classes, **model_kwargs)

    print(f"[ExitCritic] Trener modell med {len(X_train)} train-rader, {len(X_val)} val-rader.")
    print(f"[ExitCritic] Antall klasser: {num_classes}, split-type: {split_type}")
    print(f"[ExitCritic] Antall features: {len(feature_cols)}")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )

    # Preds for rapport
    if num_classes <= 2:
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = model.predict_proba(X_val)
        y_pred = np.argmax(y_prob, axis=1)

    cls_report = classification_report(y_val, y_pred, digits=4)
    cm = confusion_matrix(y_val, y_pred)

    # grunnleggende metrics i dictionary (for metadata + rapport)
    metrics = {
        "split_type": split_type,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "num_classes": num_classes,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "unique_labels": sorted(int(x) for x in df[label_col].unique()),
    }

    return model, metrics, feature_cols


def write_report(
    path: str,
    dataset_path: str,
    label_col: str,
    feature_cols: List[str],
    metrics: dict,
    model_kwargs: dict,
) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# EXIT CRITIC XGB V1 – Training Report\n\n")
        f.write(f"**Dataset:** `{dataset_path}`\n\n")
        f.write(f"**Label column:** `{label_col}`\n\n")
        f.write("## Model Config\n\n")
        f.write("```json\n")
        json.dump(model_kwargs, f, indent=2)
        f.write("\n```\n\n")

        f.write("## Data & Split\n\n")
        f.write(f"- Split type: {metrics['split_type']}\n")
        f.write(f"- Train rows: {metrics['n_train']}\n")
        f.write(f"- Val rows: {metrics['n_val']}\n")
        f.write(f"- Num classes: {metrics['num_classes']}\n")
        f.write(f"- Labels: {metrics['unique_labels']}\n\n")

        f.write("## Features\n\n")
        for col in feature_cols:
            f.write(f"- {col}\n")
        f.write("\n")

        f.write("## Classification Report (Validation)\n\n")
        f.write("```text\n")
        f.write(metrics["classification_report"])
        f.write("\n```\n\n")

        f.write("## Confusion Matrix (Validation)\n\n")
        f.write("```text\n")
        f.write(json.dumps(metrics["confusion_matrix"], indent=2))
        f.write("\n```\n")


def main():
    args = parse_args()

    print(f"[ExitCritic] Leser datasett: {args.dataset}")
    df = pd.read_parquet(args.dataset)

    if args.label_col not in df.columns:
        raise SystemExit(
            f"Label-kolonnen '{args.label_col}' finnes ikke i datasettet.\n"
            f"Tilgjengelige kolonner: {list(df.columns)}"
        )

    feature_cols = infer_feature_columns(
        df,
        label_col=args.label_col,
        extra_exclude=[],
    )

    model_kwargs = dict(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        device=args.device,
    )

    model, metrics, feature_cols = train_and_eval(
        df=df,
        label_col=args.label_col,
        feature_cols=feature_cols,
        val_frac=args.val_frac,
        use_time_split=args.use_time_split,
        seed=args.random_seed,
        model_kwargs=model_kwargs,
    )

    # Lagre modell
    ensure_dir(args.output_model)
    print(f"[ExitCritic] Lagrer modell til: {args.output_model}")
    model.save_model(args.output_model)

    # Lagre metadata
    label_mapping = {
        "HOLD": 0,
        "EXIT_NOW": 1,
        "SCALP_PROFIT": 2,
    }
    metadata = {
        "dataset_path": args.dataset,
        "label_col": args.label_col,
        "feature_cols": feature_cols,
        "label_mapping": label_mapping,
        "model_path": args.output_model,
        "model_kwargs": model_kwargs,
        "train_stats": {
            "n_rows": int(len(df)),
            "split_type": metrics["split_type"],
            "n_train": metrics["n_train"],
            "n_val": metrics["n_val"],
            "num_classes": metrics["num_classes"],
            "unique_labels": metrics["unique_labels"],
        },
    }
    ensure_dir(args.output_metadata)
    with open(args.output_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[ExitCritic] Metadata lagret til: {args.output_metadata}")

    # Lag rapport (Markdown)
    write_report(
        path=args.output_report,
        dataset_path=args.dataset,
        label_col=args.label_col,
        feature_cols=feature_cols,
        metrics=metrics,
        model_kwargs=model_kwargs,
    )
    print(f"[ExitCritic] Rapport lagret til: {args.output_report}")
    print("[ExitCritic] Ferdig.")


if __name__ == "__main__":
    main()
