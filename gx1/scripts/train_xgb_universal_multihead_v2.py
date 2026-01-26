#!/usr/bin/env python3
"""
Train Universal Multi-head XGB v2 on multiyear data.

Trains one XGB head per session (EU, US, OVERLAP) with 3-class labels:
- LONG: long payoff >= threshold and short payoff < threshold
- SHORT: short payoff >= threshold and long payoff < threshold
- FLAT: otherwise

All heads are stored in a single artifact.

Usage:
    python3 gx1/scripts/train_xgb_universal_multihead_v2.py --years 2020 2021 2022 2023 2024 2025 --sessions EU US OVERLAP
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, accuracy_score, f1_score
    import xgboost as xgb
    from joblib import dump as joblib_dump
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install scikit-learn xgboost joblib")
    sys.exit(1)

from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer
from gx1.time.session_detector import get_session_vectorized, get_session_stats


def resolve_gx1_data_dir() -> Path:
    """Resolve GX1_DATA directory."""
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    default = WORKSPACE_ROOT.parent / "GX1_DATA"
    return default


def resolve_prebuilt_for_year(year: int, gx1_data: Path) -> Optional[Path]:
    """Resolve prebuilt parquet path for a given year."""
    candidates = [
        gx1_data / "data" / "data" / "prebuilt" / "TRIAL160" / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def load_contracts() -> Tuple[List[str], str, XGBInputSanitizer, str, str]:
    """Load feature contract and sanitizer."""
    feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
    if not feature_contract_path.exists():
        raise FileNotFoundError(f"Feature contract not found: {feature_contract_path}")
    
    with open(feature_contract_path, "r") as f:
        feature_contract = json.load(f)
    
    features = feature_contract.get("features", [])
    schema_hash = feature_contract.get("schema_hash", "unknown")
    feature_contract_sha = compute_file_sha256(feature_contract_path)
    
    sanitizer_config_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_v1.json"
    if not sanitizer_config_path.exists():
        raise FileNotFoundError(f"Sanitizer config not found: {sanitizer_config_path}")
    
    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_config_path))
    sanitizer_sha = compute_file_sha256(sanitizer_config_path)
    
    return features, schema_hash, sanitizer, feature_contract_sha, sanitizer_sha


def compute_payoffs(
    df: pd.DataFrame,
    lookahead_bars: int = 12,
    spread_bps: float = 2.0,
) -> tuple:
    """
    Compute long and short payoffs (ATR-normalized).
    
    Returns:
        (long_payoffs, short_payoffs, valid_mask)
    """
    # Get price column
    if "close" in df.columns:
        close = df["close"].values
    elif "mid" in df.columns:
        close = df["mid"].values
    else:
        raise ValueError("No 'close' or 'mid' column found")
    
    # Get ATR for normalization
    if "_v1_atr" in df.columns:
        atr = df["_v1_atr"].values
    elif "atr" in df.columns:
        atr = df["atr"].values
    else:
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[0], returns])
        atr = pd.Series(returns).rolling(14).std().values * np.sqrt(14) * close
        atr = np.nan_to_num(atr, nan=0.001)
    
    atr = np.maximum(atr, close * 0.0001)
    
    # Future close
    future_close = np.roll(close, -lookahead_bars)
    
    # Spread cost
    spread_cost = close * (spread_bps / 10000)
    
    # Payoffs
    long_payoff = (future_close - close - spread_cost) / atr
    short_payoff = (close - future_close - spread_cost) / atr
    
    # Valid mask (not last lookahead_bars rows)
    valid_mask = np.ones(len(close), dtype=bool)
    valid_mask[-lookahead_bars:] = False
    
    return long_payoff, short_payoff, valid_mask


def create_3class_labels(
    df: pd.DataFrame,
    lookahead_bars: int = 12,
    threshold_atr_mult: float = 0.5,
    spread_bps: float = 2.0,
) -> np.ndarray:
    """
    Create 3-class labels (LONG=0, SHORT=1, FLAT=2) based on payoff.
    
    Args:
        df: DataFrame with 'close' (or 'mid') and optionally 'atr' columns
        lookahead_bars: Number of bars to look ahead
        threshold_atr_mult: Threshold as multiple of ATR
        spread_bps: Spread in basis points (for cost adjustment)
    
    Returns:
        Labels array: 0=LONG, 1=SHORT, 2=FLAT
    """
    # Get price column
    if "close" in df.columns:
        close = df["close"].values
    elif "mid" in df.columns:
        close = df["mid"].values
    else:
        raise ValueError("No 'close' or 'mid' column found")
    
    # Get ATR for normalization (or use simple proxy)
    if "_v1_atr" in df.columns:
        atr = df["_v1_atr"].values
    elif "atr" in df.columns:
        atr = df["atr"].values
    else:
        # Simple proxy: rolling std of returns * sqrt(14)
        returns = np.diff(close) / close[:-1]
        returns = np.concatenate([[0], returns])
        atr = pd.Series(returns).rolling(14).std().values * np.sqrt(14) * close
        atr = np.nan_to_num(atr, nan=0.001)
    
    # Ensure ATR is positive
    atr = np.maximum(atr, close * 0.0001)  # Min 1 bps
    
    # Calculate future returns
    future_close = np.roll(close, -lookahead_bars)
    
    # Spread cost in price terms
    spread_cost = close * (spread_bps / 10000)
    
    # Long payoff: buy now, sell later
    long_payoff = (future_close - close - spread_cost) / atr
    
    # Short payoff: sell now, buy later
    short_payoff = (close - future_close - spread_cost) / atr
    
    # Threshold
    threshold = threshold_atr_mult
    
    # Classify
    labels = np.full(len(close), 2, dtype=int)  # Default FLAT
    
    # LONG: long_payoff >= T and short_payoff < T
    long_mask = (long_payoff >= threshold) & (short_payoff < threshold)
    labels[long_mask] = 0
    
    # SHORT: short_payoff >= T and long_payoff < T
    short_mask = (short_payoff >= threshold) & (long_payoff < threshold)
    labels[short_mask] = 1
    
    # Mark last rows as invalid
    labels[-lookahead_bars:] = -1
    
    return labels


def get_session_mask(df: pd.DataFrame, session: str) -> np.ndarray:
    """
    Get boolean mask for rows in a specific session.
    
    Uses SSoT session detector based on timestamp.
    """
    # Find timestamp column
    ts_col = None
    for col in ["timestamp", "ts", "time", "datetime"]:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col is None:
        # Fallback to feature columns
        session_cols = {
            "EU": ["is_EU", "_v1_is_EU", "_v1_session_tag_EU"],
            "US": ["is_US", "_v1_is_US", "_v1_session_tag_US"],
            "OVERLAP": ["is_OVERLAP", "_v1_is_OVERLAP", "_v1_session_tag_OVERLAP"],
            "ASIA": ["is_ASIA", "_v1_is_ASIA", "_v1_session_tag_ASIA"],
        }
        for col in session_cols.get(session, []):
            if col in df.columns:
                return df[col].values.astype(bool)
        # Ultimate fallback
        return np.ones(len(df), dtype=bool)
    
    # Use SSoT session detector
    sessions = get_session_vectorized(df[ts_col])
    return (sessions == session).values


def calibrate_threshold(
    long_payoffs: np.ndarray,
    short_payoffs: np.ndarray,
    target_flat_min: float = 0.70,
    target_flat_max: float = 0.90,
    min_class_rate: float = 0.02,
    thresholds_to_try: list = None,
) -> tuple:
    """
    Calibrate label threshold to achieve target class balance.
    
    Args:
        long_payoffs: Array of long payoffs (ATR-normalized)
        short_payoffs: Array of short payoffs (ATR-normalized)
        target_flat_min: Minimum target FLAT rate
        target_flat_max: Maximum target FLAT rate
        min_class_rate: Minimum rate for LONG and SHORT
        thresholds_to_try: List of thresholds to test
    
    Returns:
        (best_threshold, class_rates_dict)
    """
    if thresholds_to_try is None:
        thresholds_to_try = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    
    best_threshold = 0.25  # Default
    best_score = -1
    best_rates = None
    
    for threshold in thresholds_to_try:
        # Classify with this threshold
        n_total = len(long_payoffs)
        
        long_mask = (long_payoffs >= threshold) & (short_payoffs < threshold)
        short_mask = (short_payoffs >= threshold) & (long_payoffs < threshold)
        flat_mask = ~long_mask & ~short_mask
        
        long_rate = long_mask.sum() / n_total
        short_rate = short_mask.sum() / n_total
        flat_rate = flat_mask.sum() / n_total
        
        # Check constraints
        if flat_rate < target_flat_min or flat_rate > target_flat_max:
            continue
        if long_rate < min_class_rate or short_rate < min_class_rate:
            continue
        
        # Score: prefer balanced LONG/SHORT and FLAT near middle of target
        balance_score = 1.0 - abs(long_rate - short_rate)
        flat_score = 1.0 - abs(flat_rate - (target_flat_min + target_flat_max) / 2) / 0.10
        score = balance_score + flat_score
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_rates = {
                "LONG": long_rate,
                "SHORT": short_rate,
                "FLAT": flat_rate,
            }
    
    if best_rates is None:
        # No threshold met constraints, use default
        threshold = 0.25
        n_total = len(long_payoffs)
        long_mask = (long_payoffs >= threshold) & (short_payoffs < threshold)
        short_mask = (short_payoffs >= threshold) & (long_payoffs < threshold)
        flat_mask = ~long_mask & ~short_mask
        best_rates = {
            "LONG": long_mask.sum() / n_total,
            "SHORT": short_mask.sum() / n_total,
            "FLAT": flat_mask.sum() / n_total,
        }
        best_threshold = threshold
    
    return best_threshold, best_rates


def main():
    parser = argparse.ArgumentParser(
        description="Train Universal Multi-head XGB v2"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to train on"
    )
    parser.add_argument(
        "--sessions",
        type=str,
        nargs="+",
        default=["EU", "US", "OVERLAP"],
        help="Sessions to train heads for"
    )
    parser.add_argument(
        "--n-bars-per-year",
        type=int,
        default=None,
        help="Limit bars per year"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--lookahead-bars",
        type=int,
        default=12,
        help="Lookahead bars for label creation (12 = 1 hour)"
    )
    parser.add_argument(
        "--threshold-atr-mult",
        type=float,
        default=0.5,
        help="Threshold as ATR multiple for LONG/SHORT classification"
    )
    parser.add_argument(
        "--spread-bps",
        type=float,
        default=2.0,
        help="Spread in basis points"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="XGBoost max_depth"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="XGBoost n_estimators"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="XGBoost learning_rate"
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=10,
        help="Early stopping rounds"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("TRAIN UNIVERSAL MULTI-HEAD XGB V2")
    print("=" * 60)
    
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA: {gx1_data}")
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = gx1_data / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")
    
    # Load contracts
    print("\nLoading contracts...")
    features, schema_hash, sanitizer, feature_contract_sha, sanitizer_sha = load_contracts()
    print(f"  Features: {len(features)}")
    print(f"  Schema hash: {schema_hash}")
    print(f"  Sessions: {args.sessions}")
    
    # Load multihead output contract
    output_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_multihead_output_contract_v1.json"
    output_contract_sha = compute_file_sha256(output_contract_path) if output_contract_path.exists() else None
    
    # Collect data per year
    print("\nCollecting data...")
    year_data: Dict[int, pd.DataFrame] = {}
    
    for year in args.years:
        prebuilt_path = resolve_prebuilt_for_year(year, gx1_data)
        if not prebuilt_path:
            print(f"  WARNING: No prebuilt for {year}")
            continue
        
        print(f"  Loading {year}...")
        df = pd.read_parquet(prebuilt_path)
        
        if args.n_bars_per_year and len(df) > args.n_bars_per_year:
            step = len(df) // args.n_bars_per_year
            df = df.iloc[::step][:args.n_bars_per_year]
        
        # Compute payoffs (labels will be created per-head with calibrated threshold)
        long_payoff, short_payoff, valid_mask = compute_payoffs(
            df,
            lookahead_bars=args.lookahead_bars,
            spread_bps=args.spread_bps,
        )
        df["_long_payoff"] = long_payoff
        df["_short_payoff"] = short_payoff
        df["_valid_mask"] = valid_mask
        
        # Compute session for each row
        df["_session"] = get_session_vectorized(df.get("timestamp", df.get("ts", df.index)))
        session_stats = get_session_stats(df["_session"])
        print(f"    Rows: {len(df)}")
        print(f"    Sessions: EU={session_stats['percentages'].get('EU', 0):.1f}%, "
              f"US={session_stats['percentages'].get('US', 0):.1f}%, "
              f"OVERLAP={session_stats['percentages'].get('OVERLAP', 0):.1f}%, "
              f"ASIA={session_stats['percentages'].get('ASIA', 0):.1f}%")
        
        # Sanitize features
        try:
            X, stats = sanitizer.sanitize(df, features, allow_nan_fill=True, nan_fill_value=0.0)
            df["_X_sanitized"] = list(X)
            print(f"    Clip rate: {stats.clip_rate_pct:.2f}%")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
        
        year_data[year] = df
    
    if not year_data:
        print("ERROR: No data collected")
        return 1
    
    # Train one head per session
    print("\nTraining heads...")
    heads: Dict[str, Any] = {}
    head_metrics: Dict[str, Dict[str, Any]] = {}
    threshold_by_head: Dict[str, float] = {}
    
    xgb_params = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": args.seed,
        "n_jobs": -1,
        "early_stopping_rounds": args.early_stopping_rounds,
    }
    
    for session in args.sessions:
        print(f"\n  Training {session} head...")
        
        # Collect session data using SSoT session detector
        all_long_payoffs = []
        all_short_payoffs = []
        all_X = []
        session_bars_per_year = {}
        
        for year, df in year_data.items():
            # Get session mask using computed _session column
            session_mask = (df["_session"] == session) & df["_valid_mask"]
            n_session_bars = session_mask.sum()
            session_bars_per_year[year] = n_session_bars
            
            if n_session_bars == 0:
                continue
            
            # Collect payoffs for threshold calibration
            long_payoffs = df.loc[session_mask, "_long_payoff"].values
            short_payoffs = df.loc[session_mask, "_short_payoff"].values
            X = np.array([x for x, v in zip(df["_X_sanitized"], session_mask) if v])
            
            all_long_payoffs.append(long_payoffs)
            all_short_payoffs.append(short_payoffs)
            all_X.append(X)
        
        if not all_X:
            print(f"    WARNING: No data for {session}, skipping")
            continue
        
        all_long_payoffs = np.concatenate(all_long_payoffs)
        all_short_payoffs = np.concatenate(all_short_payoffs)
        X_all = np.vstack(all_X)
        
        total_bars = len(X_all)
        print(f"    Total session bars: {total_bars}")
        for year, count in session_bars_per_year.items():
            print(f"      {year}: {count} bars")
        
        # HARD FAIL if too few bars
        if total_bars < 50000:
            print(f"    ERROR: Too few bars for {session}: {total_bars} < 50000")
            return 1
        
        # Calibrate threshold for this head
        print(f"    Calibrating threshold...")
        best_threshold, class_rates = calibrate_threshold(
            all_long_payoffs,
            all_short_payoffs,
            target_flat_min=0.70,
            target_flat_max=0.90,
            min_class_rate=0.02,
        )
        threshold_by_head[session] = best_threshold
        print(f"      Selected threshold: {best_threshold}")
        print(f"      LONG: {class_rates['LONG']:.1%}, SHORT: {class_rates['SHORT']:.1%}, FLAT: {class_rates['FLAT']:.1%}")
        
        # Create labels with calibrated threshold
        long_mask = (all_long_payoffs >= best_threshold) & (all_short_payoffs < best_threshold)
        short_mask = (all_short_payoffs >= best_threshold) & (all_long_payoffs < best_threshold)
        
        y_all = np.full(len(all_long_payoffs), 2, dtype=int)  # FLAT
        y_all[long_mask] = 0  # LONG
        y_all[short_mask] = 1  # SHORT
        
        # Class counts
        class_counts = {0: (y_all == 0).sum(), 1: (y_all == 1).sum(), 2: (y_all == 2).sum()}
        
        # Time-based split
        n_train = int(len(X_all) * (1 - args.val_split))
        X_train, X_val = X_all[:n_train], X_all[n_train:]
        y_train, y_val = y_all[:n_train], y_all[n_train:]
        
        # Train
        print(f"    Training XGBoost...")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        # Evaluate
        y_val_pred = model.predict_proba(X_val)
        val_logloss = log_loss(y_val, y_val_pred)
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        
        print(f"    Val LogLoss: {val_logloss:.4f}, Accuracy: {val_accuracy:.2%}")
        
        heads[session] = model
        head_metrics[session] = {
            "n_samples": total_bars,
            "bars_per_year": session_bars_per_year,
            "threshold_atr_mult": best_threshold,
            "class_distribution": {
                "LONG": class_counts[0] / total_bars,
                "SHORT": class_counts[1] / total_bars,
                "FLAT": class_counts[2] / total_bars,
            },
            "val_logloss": val_logloss,
            "val_accuracy": val_accuracy,
        }
        
        # Save feature importance
        importance_df = pd.DataFrame({
            "feature": features[:len(model.feature_importances_)],
            "gain": model.feature_importances_,
        }).sort_values("gain", ascending=False)
        
        importance_path = output_dir / f"xgb_universal_multihead_v2_feature_importance_{session}.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"    Feature importance: {importance_path.name}")
    
    if not heads:
        print("ERROR: No heads trained")
        return 1
    
    # Verify heads diverge
    print("\n  Verifying head divergence...")
    divergence_check_passed = True
    head_signatures = {}
    
    # Get 2025 data for verification
    if 2025 in year_data:
        df_2025 = year_data[2025]
        for session in heads.keys():
            session_mask = (df_2025["_session"] == session) & df_2025["_valid_mask"]
            if session_mask.sum() < 1000:
                continue
            
            X_check = np.array([x for x, v in zip(df_2025["_X_sanitized"], session_mask) if v])[:5000]
            proba = heads[session].predict_proba(X_check)
            
            sig = {
                "p_long_mean": float(proba[:, 0].mean()),
                "p_short_mean": float(proba[:, 1].mean()),
                "p_flat_mean": float(proba[:, 2].mean()),
            }
            head_signatures[session] = sig
            print(f"    {session}: p_long={sig['p_long_mean']:.4f}, p_short={sig['p_short_mean']:.4f}, p_flat={sig['p_flat_mean']:.4f}")
        
        # Check if heads are too similar
        sessions_list = list(head_signatures.keys())
        for i in range(len(sessions_list)):
            for j in range(i + 1, len(sessions_list)):
                s1, s2 = sessions_list[i], sessions_list[j]
                diff = abs(head_signatures[s1]["p_long_mean"] - head_signatures[s2]["p_long_mean"])
                if diff < 1e-4:
                    print(f"    ⚠️  WARNING: {s1} and {s2} are very similar (diff={diff:.6f})")
    
    if not divergence_check_passed:
        print("  ❌ Heads are identical - session filtering may not be working")
    else:
        print("  ✅ Head divergence check passed")
    
    # Save model
    print("\nSaving model...")
    model_data = {
        "heads": heads,
        "feature_list": features,
        "meta": {
            "version": "xgb_universal_multihead_v2",
            "created_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            "schema_hash": schema_hash,
            "feature_contract_sha256": feature_contract_sha,
            "sanitizer_sha256": sanitizer_sha,
            "output_contract_sha256": output_contract_sha,
            "sessions": list(heads.keys()),
            "n_features": len(features),
            "training": {
                "years": args.years,
                "lookahead_bars": args.lookahead_bars,
                "threshold_by_head": threshold_by_head,
                "spread_bps": args.spread_bps,
                "val_split": args.val_split,
                "seed": args.seed,
            },
            "xgb_params": {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"},
            "head_metrics": head_metrics,
            "head_signatures": head_signatures,
        },
    }
    
    model_path = output_dir / "xgb_universal_multihead_v2.joblib"
    try:
        from joblib import dump as joblib_dump
    except ImportError:
        import joblib
        joblib_dump = joblib.dump
    
    joblib_dump(model_data, model_path)
    model_sha = compute_file_sha256(model_path)
    print(f"  Model: {model_path}")
    print(f"  Model SHA256: {model_sha[:16]}...")
    
    # Save metadata (convert numpy types to Python types)
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    meta_path = output_dir / "xgb_universal_multihead_v2_meta.json"
    meta_for_json = convert_to_serializable(model_data["meta"])
    with open(meta_path, "w") as f:
        json.dump(meta_for_json, f, indent=2)
    meta_sha = compute_file_sha256(meta_path)
    print(f"  Metadata: {meta_path}")
    print(f"  Meta SHA256: {meta_sha[:16]}...")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Sessions: {list(heads.keys())}")
    for session, metrics in head_metrics.items():
        print(f"  {session}: LogLoss={metrics['val_logloss']:.4f}, Acc={metrics['val_accuracy']:.2%}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
