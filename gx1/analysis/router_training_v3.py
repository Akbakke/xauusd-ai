import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import joblib


# ---------------------------
# Data loading & preparation
# ---------------------------

def load_dataset_v3(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix == ".parquet":
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == ".csv":
        df = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

    return df


def _clean_distance_to_range(series: pd.Series) -> pd.Series:
    """
    V4-A: gjør distance_to_range robust og "first-class".

    Regler:
      - Coerce til numeric
      - NaN -> 0.5 (mid-range nøytral)
      - Hvis data ser normalisert ut (~[0,1]) så klipp til [0,1]
        (heuristikk: 5-95 percentil i [-0.1, 1.2])
      - Ellers behold rå float-verdier
    """
    s = pd.to_numeric(series, errors="coerce")

    # NaN -> midrange
    if s.isna().any():
        s = s.fillna(0.5)

    # Heuristikk for normalisert range-feature
    q05, q95 = s.quantile([0.05, 0.95])
    if q05 >= -0.1 and q95 <= 1.2:
        s = s.clip(0.0, 1.0)

    return s.astype(float)


def prepare_features_v3(df: pd.DataFrame):
    """
    Forventer minst:
      - policy_best (target)
      - atr_pct, spread_pct, atr_bucket, farm_regime, session,
        distance_to_range, micro_volatility, volume_stability
      - pnl_rule5_bps, pnl_rule6a_bps (for EV-evaluering)

    V4-A endring:
      - distance_to_range renses eksplisitt og brukes som first-class feature.
    """

    required_cols = [
        "policy_best",
        "atr_pct",
        "spread_pct",
        "atr_bucket",
        "farm_regime",
        "session",
        "distance_to_range",
        "range_edge_dist_atr",
        "micro_volatility",
        "volume_stability",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset mangler kolonner: {missing}")

    # Dropp rader uten target eller med UNKNOWN
    df = df.dropna(subset=["policy_best"]).copy()
    df = df[df["policy_best"] != "UNKNOWN"].copy()

    # ---- V4-A: robust range feature ----
    df["distance_to_range"] = _clean_distance_to_range(df["distance_to_range"])

    # Sørg for numeriske kolonner er numeriske (konservativt)
    df["atr_pct"] = pd.to_numeric(df["atr_pct"], errors="coerce")
    df["spread_pct"] = pd.to_numeric(df["spread_pct"], errors="coerce")
    df["range_edge_dist_atr"] = pd.to_numeric(df["range_edge_dist_atr"], errors="coerce").fillna(0.0).clip(0.0, 10.0)
    df["micro_volatility"] = pd.to_numeric(df["micro_volatility"], errors="coerce").fillna(0.0)
    df["volume_stability"] = pd.to_numeric(df["volume_stability"], errors="coerce").fillna(0.0)

    # Target: policy_best (kan være RULE5, RULE6A, senere flere)
    y = df["policy_best"].astype(str).values

    numeric_features = [
        "atr_pct",
        "spread_pct",
        "distance_to_range",
        "range_edge_dist_atr",
        "micro_volatility",
        "volume_stability",
    ]
    categorical_features = [
        "atr_bucket",
        "farm_regime",
        "session",
    ]

    X = df[numeric_features + categorical_features].copy()

    # Meta for EV-beregning (brukes bare hvis vi er i RULE5 vs RULE6A-mode)
    meta_cols = ["trade_id", "quarter", "pnl_rule5_bps", "pnl_rule6a_bps", "policy_best", "policy_actual"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[meta_cols].copy()

    feature_spec = {
        "numeric": numeric_features,
        "categorical": categorical_features,
    }

    return X, y, meta, feature_spec


def build_preprocessor(feature_spec):
    numeric_features = feature_spec["numeric"]
    categorical_features = feature_spec["categorical"]

    numeric_transformer = "passthrough"

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


# ---------------------------
# EV-evaluering (for RULE5 vs RULE6A)
# ---------------------------

def evaluate_router_ev_binary(
    clf: Pipeline,
    X_val: pd.DataFrame,
    df_val: pd.DataFrame,
    threshold: float = 0.5,
):
    """
    Evaluer EV/trade og EV/day for en binary router mellom RULE5 og RULE6A.
    Forutsetter at:
      - y_true ∈ {RULE5, RULE6A}
      - df_val inneholder 'pnl_rule5_bps' og 'pnl_rule6a_bps'
    """

    if not {"pnl_rule5_bps", "pnl_rule6a_bps"}.issubset(df_val.columns):
        raise ValueError("Mangler pnl_rule5_bps / pnl_rule6a_bps for EV-evaluering.")

    df = df_val.copy().reset_index(drop=True)

    # Dropp rader uten PnL-informasjon
    mask_valid = df["pnl_rule5_bps"].notna() & df["pnl_rule6a_bps"].notna()
    df = df[mask_valid].copy()

    if len(df) == 0:
        print("[router_training_v3] ⚠️  Ingen trades med både pnl_rule5_bps og pnl_rule6a_bps i valideringssettet. Skipper EV-evaluering.")
        return None

    # Viktig: align X_val med filtrerte rader. X_val er i samme rekkefølge som df_val etter reset_index.
    if hasattr(mask_valid, "values"):
        X_val = X_val.iloc[mask_valid.values].copy()
    else:
        X_val = X_val.iloc[mask_valid].copy()

    # Pred proba for RULE6A
    proba = clf.predict_proba(X_val)
    classes = clf.named_steps["clf"].classes_
    classes = list(classes)
    if "RULE6A" not in classes or "RULE5" not in classes:
        raise ValueError("Model classes må inneholde RULE5 og RULE6A for EV-evaluering.")

    idx_rule6a = classes.index("RULE6A")
    p_rule6a = proba[:, idx_rule6a]

    # Baseline: alltid RULE5
    df["pnl_baseline_rule5"] = df["pnl_rule5_bps"]

    # Oracle: velg best av de to
    df["pnl_oracle"] = np.where(
        df["pnl_rule6a_bps"] > df["pnl_rule5_bps"],
        df["pnl_rule6a_bps"],
        df["pnl_rule5_bps"],
    )

    # Modell: RULE6A hvis p>=threshold, ellers RULE5
    df["use_rule6a_model"] = p_rule6a >= threshold
    df["pnl_model"] = np.where(
        df["use_rule6a_model"], df["pnl_rule6a_bps"], df["pnl_rule5_bps"]
    )

    ev_baseline = df["pnl_baseline_rule5"].mean()
    ev_oracle = df["pnl_oracle"].mean()
    ev_model = df["pnl_model"].mean()

    uplift_model_vs_baseline = ev_model - ev_baseline
    uplift_oracle_vs_baseline = ev_oracle - ev_baseline

    # Approx EV/day: placeholder (som før)
    n_trades = len(df)
    ev_day_model = ev_model * 1.0  # placeholder

    # Classification metrics (bruk policy_best hvis finnes, ellers policy_actual)
    y_true_series = None
    if "policy_best" in df.columns:
        y_true_series = df["policy_best"]
    elif "policy_actual" in df.columns:
        y_true_series = df["policy_actual"]

    if y_true_series is not None:
        y_true = y_true_series.values.astype(str)
        y_pred = np.where(df["use_rule6a_model"], "RULE6A", "RULE5")
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=["RULE5", "RULE6A"]).tolist()
    else:
        acc = None
        cm = None

    metrics = {
        "threshold": float(threshold),
        "n_trades": int(n_trades),
        "ev_per_trade": {
            "baseline_rule5": float(ev_baseline),
            "oracle": float(ev_oracle),
            "model": float(ev_model),
        },
        "uplift_vs_baseline": {
            "model_minus_baseline": float(uplift_model_vs_baseline),
            "oracle_minus_baseline": float(uplift_oracle_vs_baseline),
        },
        "ev_per_day_approx": {
            "model": float(ev_day_model),
        },
        "classification": {
            "accuracy": float(acc) if acc is not None else None,
            "confusion_matrix": cm,
        },
        "routing": {
            "pct_rule6a_model": float(100.0 * df["use_rule6a_model"].mean()),
        },
    }

    return metrics


def compute_ev_metrics_full(
    df: pd.DataFrame,
    clf: Pipeline,
    feature_spec,
) -> Optional[Dict[str, float]]:
    """
    Evaluer EV-baseline, tre (modell) og oracle over alle trades
    som har både pnl_rule5_bps og pnl_rule6a_bps tilgjengelig.
    """
    required_cols = {"pnl_rule5_bps", "pnl_rule6a_bps"}
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Mangler kolonner for EV-evaluering: {missing_required}")

    feature_cols = feature_spec["numeric"] + feature_spec["categorical"]
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Mangler feature-kolonner for EV-evaluering: {missing_features}")

    ev_df = df[
        df["pnl_rule5_bps"].notnull() & df["pnl_rule6a_bps"].notnull()
    ].copy()

    if ev_df.empty:
        print("[router_training_v3] ⚠️  Ingen trades med både pnl_rule5_bps og pnl_rule6a_bps for full EV-evaluering.")
        return None

    # V4-A: sørg for distance_to_range er renset også her
    if "distance_to_range" in ev_df.columns:
        ev_df["distance_to_range"] = _clean_distance_to_range(ev_df["distance_to_range"])
    if "micro_volatility" in ev_df.columns:
        ev_df["micro_volatility"] = pd.to_numeric(ev_df["micro_volatility"], errors="coerce").fillna(0.0)
    if "volume_stability" in ev_df.columns:
        ev_df["volume_stability"] = pd.to_numeric(ev_df["volume_stability"], errors="coerce").fillna(0.0)
    if "atr_pct" in ev_df.columns:
        ev_df["atr_pct"] = pd.to_numeric(ev_df["atr_pct"], errors="coerce")
    if "spread_pct" in ev_df.columns:
        ev_df["spread_pct"] = pd.to_numeric(ev_df["spread_pct"], errors="coerce")

    ev_features = ev_df[feature_cols].copy()
    tree_preds = clf.predict(ev_features)

    baseline_pnl = ev_df["pnl_rule5_bps"].astype(float)
    tree_pnl = np.where(
        tree_preds == "RULE6A",
        ev_df["pnl_rule6a_bps"],
        ev_df["pnl_rule5_bps"],
    )

    policy_best = ev_df.get("policy_best")
    if policy_best is not None:
        policy_best = policy_best.astype(str)
        oracle_pnl = np.where(
            policy_best == "RULE6A",
            ev_df["pnl_rule6a_bps"],
            np.where(
                policy_best == "RULE5",
                ev_df["pnl_rule5_bps"],
                np.maximum(ev_df["pnl_rule5_bps"], ev_df["pnl_rule6a_bps"]),
            ),
        )
    else:
        oracle_pnl = np.maximum(ev_df["pnl_rule5_bps"], ev_df["pnl_rule6a_bps"])

    baseline_ev = float(np.mean(baseline_pnl))
    tree_ev = float(np.mean(tree_pnl))
    oracle_ev = float(np.mean(oracle_pnl))

    ev_summary = {
        "baseline_rule5_ev": baseline_ev,
        "tree_ev_per_trade": tree_ev,
        "oracle_ev": oracle_ev,
        "uplift_tree_vs_baseline": tree_ev - baseline_ev,
        "uplift_oracle_vs_tree": oracle_ev - tree_ev,
        "uplift_oracle_vs_baseline": oracle_ev - baseline_ev,
        "n_samples_ev": int(len(ev_df)),
    }

    return ev_summary


# ---------------------------
# Training
# ---------------------------

def train_tree_v3(X, y, feature_spec, random_state=42):
    preprocessor = build_preprocessor(feature_spec)

    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=random_state,
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clf", tree),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=y,
    )

    clf.fit(X_train, y_train)

    return clf, X_train, X_val, y_train, y_val


def export_tree_rules_v3(tree_pipeline: Pipeline, feature_spec, out_path: Path) -> str:
    """
    Eksporter V3-treet som lesbare regler – dette brukes senere for å
    hardkode HYBRID_ROUTER_V3 i Python.
    """
    preprocessor: ColumnTransformer = tree_pipeline.named_steps["preprocessor"]
    clf: DecisionTreeClassifier = tree_pipeline.named_steps["clf"]

    numeric_features = feature_spec["numeric"]
    categorical_features = feature_spec["categorical"]

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
        default="gx1/data/exit_policy/exit_policy_training_dataset_v3.parquet",
        help="Path til V3 exit-policy treningsdataset (parquet/csv).",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="gx1/analysis/exit_router_models_v3",
        help="Katalog for lagring av modeller og metrics.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Threshold på p(RULE6A) for RULE6A vs RULE5 i EV-evaluering (binary mode).",
    )
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[router_training_v3] Laster dataset: {dataset_path}")
    df = load_dataset_v3(dataset_path)
    print(f"[router_training_v3] Antall rader totalt: {len(df)}")

    X, y, meta, feature_spec = prepare_features_v3(df)
    print(f"[router_training_v3] Antall rader etter filtrering: {len(X)}")
    print(f"[router_training_v3] Unike policies (policy_best): {sorted(pd.unique(y))}")

    print("[router_training_v3] Trener Decision Tree (V3 + V4-A range-awareness)...")
    clf, X_train, X_val, y_train, y_val = train_tree_v3(X, y, feature_spec)

    # Klassifikasjonsevaluering
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred).tolist()

    print("\n[router_training_v3] Klassifikasjons-metrics:")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rad = true, kol = pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    # EV-evaluering hvis vi kun har RULE5/RULE6A
    policies_unique = sorted(pd.unique(y))
    ev_metrics = None
    if set(policies_unique).issubset({"RULE5", "RULE6A"}):
        print("[router_training_v3] Kjører EV-evaluering for RULE5 vs RULE6A...")
        # Split meta på samme måte som X/y
        _, meta_val = train_test_split(
            meta,
            test_size=0.3,
            random_state=42,
            stratify=y,
        )
        meta_val = meta_val.reset_index(drop=True)

        X_val_df = pd.DataFrame(X_val, columns=X.columns)
        ev_metrics = evaluate_router_ev_binary(
            clf=clf,
            X_val=X_val_df,
            df_val=meta_val,
            threshold=args.threshold,
        )
        print("[router_training_v3] EV-metrics:")
        print(json.dumps(ev_metrics, indent=2))
    else:
        print("[router_training_v3] Skipper EV-evaluering (ikke ren RULE5/RULE6A).")

    print("[router_training_v3] Kjører full EV-evaluering basert på policy_best og hypotetiske PnL-felt...")
    df_for_ev = load_dataset_v3(dataset_path)
    full_ev_summary = compute_ev_metrics_full(df_for_ev, clf, feature_spec)
    if full_ev_summary:
        print("[router_training_v3] Full EV-metrics:")
        print(json.dumps(full_ev_summary, indent=2))
    else:
        print("[router_training_v3] Full EV-evaluering hoppet over (mangler nødvendige data).")

    # Eksporter tre-regler
    tree_rules_path = output_dir / "exit_router_v3_tree_rules.txt"
    tree_text = export_tree_rules_v3(clf, feature_spec, tree_rules_path)
    print(f"\n[router_training_v3] Decision Tree regler lagret til: {tree_rules_path}")
    print(tree_text)

    # Lagre modell
    model_path = output_dir / "exit_router_v3_tree.pkl"
    joblib.dump(clf, model_path)

    # Lagre metrics
    metrics_path = output_dir / "exit_router_v3_metrics.json"
    all_metrics = {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "tree_ev_per_trade": full_ev_summary["tree_ev_per_trade"] if full_ev_summary else None,
        "baseline_rule5_ev": full_ev_summary["baseline_rule5_ev"] if full_ev_summary else None,
        "oracle_ev": full_ev_summary["oracle_ev"] if full_ev_summary else None,
        "uplift_tree_vs_baseline": full_ev_summary["uplift_tree_vs_baseline"] if full_ev_summary else None,
        "uplift_oracle_vs_tree": full_ev_summary["uplift_oracle_vs_tree"] if full_ev_summary else None,
        "uplift_oracle_vs_baseline": full_ev_summary["uplift_oracle_vs_baseline"] if full_ev_summary else None,
        "n_samples_ev": full_ev_summary["n_samples_ev"] if full_ev_summary else 0,
        "n_samples_total": int(len(X)),
        "policies": policies_unique,
        "threshold_binary_ev": args.threshold,
        "ev_metrics_binary": ev_metrics,
        "full_ev_metrics": full_ev_summary,
        "v4_range_awareness": {
            "enabled": True,
            "feature": "distance_to_range",
            "nan_fill": 0.5,
            "clip_if_normalized_q05_q95": [-0.1, 1.2],
        },
    }
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")

    print(f"[router_training_v3] Lagret modell til: {model_path}")
    print(f"[router_training_v3] Lagret metrics til: {metrics_path}")
    print("[router_training_v3] Ferdig.")


if __name__ == "__main__":
    main()
