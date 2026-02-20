#!/usr/bin/env python3
"""
Minimal XGBoost repro test (isolated, no Optuna, no multiprocessing).

If this hangs → environment / XGB / BLAS problem
If this works → fork + multiprocessing problem
"""
import xgboost as xgb
import numpy as np
import time

print("=== Minimal XGBoost Test ===")
print(f"XGBoost version: {xgb.__version__}")

# Generate minimal data
X = np.random.randn(1000, 10).astype(np.float32)
y = np.random.randint(0, 2, size=1000).astype(np.int32)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X finite: {np.isfinite(X).all()}, y finite: {np.isfinite(y).all()}")

dtrain = xgb.DMatrix(X, label=y)
print(f"DMatrix created: {dtrain.num_row()} rows, {dtrain.num_col()} cols")

params = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "nthread": 1,
    "verbosity": 2,
}

print("\n=== BEFORE TRAIN ===")
start = time.perf_counter()
bst = xgb.train(params, dtrain, num_boost_round=50)
elapsed = time.perf_counter() - start
print(f"\n=== AFTER TRAIN (took {elapsed:.2f}s) ===")
print("✅ SUCCESS: XGBoost training completed")
