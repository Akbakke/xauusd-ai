import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import joblib


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    return df


def prepare_features(df: pd.DataFrame):
    """
    Forventer minst følgende kolonner i df:
      - atr_pct (float)
      - spread_pct (float)
      - atr_bucket (str / kategori)
      - regime (str / kategori)
      - session (str / kategori)
      - best_policy ("RULE5" / "RULE6A")
      - pnl_rule5_bps (float)
      - pnl_rule6a_bps (float)
    """

    required_cols = [
        "atr_pct",
        "spread_pct",
        "atr_bucket",
        "regime",
        "session",
        "best_policy",
        "pnl_rule5_bps",
        "pnl_rule6a_bps",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset mangler kolonner: {missing}")

    # Dataset should already be cleaned by build_exit_router_dataset.py
    # Only drop rows with NaN in critical fields (best_policy, pnl_rule5_bps, pnl_rule6a_bps)
    # atr_pct, atr_bucket, spread_pct, regime, session should already be filled
    critical_cols = ["best_policy", "pnl_rule5_bps", "pnl_rule6a_bps"]
    df = df.dropna(subset=critical_cols).copy()

    # Target: 0 = RULE5, 1 = RULE6A
    y = (df["best_policy"] == "RULE6A").astype(int).values

    # Features
    numeric_features = ["atr_pct", "spread_pct"]
    categorical_features = ["atr_bucket", "regime", "session"]

    X = df[numeric_features + categorical_features].copy()

    # Meta for EV-beregning
    meta = df[["pnl_rule5_bps", "pnl_rule6a_bps", "quarter", "trade_id"]].copy()

    feature_spec = {
        "numeric": numeric_features,
        "categorical": categorical_features,
    }

    return X, y, meta, feature_spec


def build_preprocessor(feature_spec):
    numeric_features = feature_spec["numeric"]
    categorical_features = feature_spec["categorical"]

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_models(X, y, feature_spec, random_state=42):
    preprocessor = build_preprocessor(feature_spec)

    # Logistic Regression
    log_reg = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )

    log_reg_clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", log_reg),
        ]
    )

    # Decision Tree (liten, forklarbar)
    tree = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
    )

    tree_clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", tree),
        ]
    )

    # Train/val-split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=y,
    )

    log_reg_clf.fit(X_train, y_train)
    tree_clf.fit(X_train, y_train)

    return {
        "log_reg": log_reg_clf,
        "tree": tree_clf,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }


def evaluate_router_model(y_true, p_rule6a, meta_val, threshold=0.5):
    """
    Simuler EV/trade for:
      - alltid RULE5
      - oracle (perfekt valg)
      - modellen (RULE6A hvis p>=threshold)
    """

    df = meta_val.copy()
    df = df.reset_index(drop=True)

    if len(df) != len(y_true):
        raise ValueError("meta_val og y_true har ulik lengde")

    df["y_true"] = y_true
    df["p_rule6a"] = p_rule6a

    # baseline: alltid RULE5
    df["pnl_baseline_rule5"] = df["pnl_rule5_bps"]

    # oracle: alltid velg best av de to
    df["pnl_oracle"] = np.where(
        df["pnl_rule6a_bps"] > df["pnl_rule5_bps"],
        df["pnl_rule6a_bps"],
        df["pnl_rule5_bps"],
    )

    # modell: velg RULE6A hvis p>=threshold, ellers RULE5
    df["use_rule6a_model"] = df["p_rule6a"] >= threshold
    df["pnl_model"] = np.where(
        df["use_rule6a_model"], df["pnl_rule6a_bps"], df["pnl_rule5_bps"]
    )

    ev_baseline = df["pnl_baseline_rule5"].mean()
    ev_oracle = df["pnl_oracle"].mean()
    ev_model = df["pnl_model"].mean()

    upl_model_vs_baseline = ev_model - ev_baseline
    upl_oracle_vs_baseline = ev_oracle - ev_baseline

    # accuracy if we interpret model decision as policy prediction
    y_pred_policy = df["use_rule6a_model"].astype(int).values
    acc = accuracy_score(df["y_true"].values, y_pred_policy)
    cm = confusion_matrix(df["y_true"].values, y_pred_policy)

    metrics = {
        "threshold": float(threshold),
        "ev_per_trade": {
            "baseline_rule5": float(ev_baseline),
            "oracle": float(ev_oracle),
            "model": float(ev_model),
        },
        "uplift_vs_baseline": {
            "model_minus_baseline": float(upl_model_vs_baseline),
            "oracle_minus_baseline": float(upl_oracle_vs_baseline),
        },
        "classification": {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
        },
    }

    return metrics, df


def export_tree_rules(tree_pipeline: Pipeline, feature_spec, out_path: Path):
    """
    Eksporter treet som lesbare regler (tekst).
    Dette er det vi senere kan bruke til å oversette til ren if/else-Python.
    """

    preprocessor: ColumnTransformer = tree_pipeline.named_steps["preprocessor"]
    clf: DecisionTreeClassifier = tree_pipeline.named_steps["clf"]

    # Hent feature-navn etter preprocessing
    numeric_features = feature_spec["numeric"]
    categorical_features = feature_spec["categorical"]

    # ColumnTransformer: vi må hente navn i riktig rekkefølge
    feature_names_out = []

    # numeric
    feature_names_out.extend(numeric_features)

    # categorical (OneHotEncoder)
    cat_transformer = preprocessor.named_transformers_["cat"]
    ohe: OneHotEncoder = cat_transformer.named_steps["onehot"]
    cat_ohe_names = ohe.get_feature_names_out(categorical_features).tolist()
    feature_names_out.extend(cat_ohe_names)

    tree_text = export_text(clf, feature_names=feature_names_out)

    out_path.write_text(tree_text, encoding="utf-8")
    return tree_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-path",
        type=str,
        default="gx1/data/exit_router/exit_router_training_dataset.parquet",
        help="Path til treningsdataset (parquet/csv).",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="gx1/analysis/exit_router_models",
        help="Katalog for lagring av modeller og metrics.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold på p(RULE6A) for å velge RULE6A.",
    )
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[router_training] Laster dataset: {dataset_path}")
    df = load_dataset(dataset_path)

    print(f"[router_training] Antall rader før filtrering: {len(df)}")
    X, y, meta, feature_spec = prepare_features(df)
    print(f"[router_training] Antall rader etter filtrering: {len(X)}")

    print("[router_training] Trener modeller (logistic + decision tree)...")
    trained = train_models(X, y, feature_spec)

    log_reg_clf = trained["log_reg"]
    tree_clf = trained["tree"]
    X_val = trained["X_val"]
    y_val = trained["y_val"]

    # Vi trenger samme subset av meta som X_val
    # train_test_split har shufflet, så vi lager en matcher basert på indeks
    # For enkelhet antar vi at X og meta hadde samme index rekkefølge ved split
    _, meta_val = train_test_split(
        meta,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    meta_val = meta_val.reset_index(drop=True)

    # Evaluer logistic
    print("[router_training] Evaluerer Logistic Regression...")
    p_lr = log_reg_clf.predict_proba(X_val)[:, 1]
    lr_metrics, lr_df = evaluate_router_model(
        y_true=y_val,
        p_rule6a=p_lr,
        meta_val=meta_val,
        threshold=args.threshold,
    )

    print("[router_training] Logistic Regression metrics:")
    print(json.dumps(lr_metrics, indent=2))

    y_pred_lr = (p_lr >= args.threshold).astype(int)
    print("\n[router_training] Logistic Regression classification report:")
    print(classification_report(y_val, y_pred_lr, target_names=["RULE5", "RULE6A"]))

    # Evaluer decision tree
    print("[router_training] Evaluerer Decision Tree...")
    p_tree = tree_clf.predict_proba(X_val)[:, 1]
    tree_metrics, tree_df = evaluate_router_model(
        y_true=y_val,
        p_rule6a=p_tree,
        meta_val=meta_val,
        threshold=args.threshold,
    )

    print("[router_training] Decision Tree metrics:")
    print(json.dumps(tree_metrics, indent=2))

    y_pred_tree = (p_tree >= args.threshold).astype(int)
    print("\n[router_training] Decision Tree classification report:")
    print(classification_report(y_val, y_pred_tree, target_names=["RULE5", "RULE6A"]))

    # Eksporter tree-regler
    tree_rules_path = output_dir / "exit_router_decision_tree_rules.txt"
    tree_rules = export_tree_rules(tree_clf, feature_spec, tree_rules_path)
    print(f"\n[router_training] Decision Tree regler lagret til: {tree_rules_path}")
    print(tree_rules)

    # Lagre modeller og metrics
    log_reg_path = output_dir / "exit_router_logistic.pkl"
    tree_path = output_dir / "exit_router_tree.pkl"
    metrics_path = output_dir / "exit_router_metrics.json"

    joblib.dump(log_reg_clf, log_reg_path)
    joblib.dump(tree_clf, tree_path)

    all_metrics = {
        "logistic": lr_metrics,
        "decision_tree": tree_metrics,
        "threshold": args.threshold,
        "n_samples": int(len(X)),
    }

    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    print(f"[router_training] Lagret modeller til:")
    print(f"  - {log_reg_path}")
    print(f"  - {tree_path}")
    print(f"[router_training] Lagret metrics til: {metrics_path}")
    print("[router_training] Ferdig.")


if __name__ == "__main__":
    main()
