# gx1/analysis/router_threshold_sweep.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from router_training import (
    load_dataset,
    prepare_features,
    evaluate_router_model,
)

def threshold_sweep(
    model_path: str,
    dataset_path: str,
    output_path: str,
    thresholds=None
):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    # Load model
    model = joblib.load(model_path)

    # Load dataset
    df = load_dataset(Path(dataset_path))
    X, y, meta, _ = prepare_features(df)

    # Predict probabilities
    p = model.predict_proba(X)[:, 1]

    results = []

    for t in thresholds:
        metrics, df_eval = evaluate_router_model(
            y_true=y,
            p_rule6a=p,
            meta_val=meta,
            threshold=t,
        )

        n_rule6a = int(df_eval["use_rule6a_model"].sum())
        n_total = len(df_eval)
        pct_rule6a = 100.0 * n_rule6a / n_total

        results.append({
            "threshold": float(t),
            "ev_trade": metrics["ev_per_trade"]["model"],
            "ev_day": metrics["ev_per_trade"]["model"] * (n_total / 194),  # approximate
            "baseline_ev_trade": metrics["ev_per_trade"]["baseline_rule5"],
            "oracle_ev_trade": metrics["ev_per_trade"]["oracle"],
            "uplift_vs_baseline": metrics["uplift_vs_baseline"]["model_minus_baseline"],
            "pct_rule6a": pct_rule6a,
            "n_trades": n_total,
        })

    # Save
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)

    print(f"Sweep saved to: {output_path}")
    print(df_out)


if __name__ == "__main__":
    threshold_sweep(
        model_path="gx1/analysis/exit_router_models/exit_router_tree.pkl",
        dataset_path="gx1/data/exit_router/exit_router_training_dataset.parquet",
        output_path="gx1/analysis/exit_router_models/router_threshold_sweep_results.csv",
        thresholds=np.linspace(0.05, 0.95, 19),
    )
