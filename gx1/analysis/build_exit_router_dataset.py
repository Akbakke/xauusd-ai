#!/usr/bin/env python3
"""
Build training dataset for ML-based exit router.

This script extracts features and labels from hybrid exit replay trade logs
to train a model that predicts whether a trade should use RULE5 or RULE6A.

Input: Trade logs from hybrid exit replays (Q1-Q4 2025)
Output: Training dataset with features and labels for exit routing decisions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import json

def load_hybrid_trade_logs(quarters: List[str]) -> pd.DataFrame:
    """Load trade logs from hybrid exit replays for specified quarters."""
    base_path = Path("gx1/wf_runs")
    all_trades = []
    
    for quarter in quarters:
        # Try both possible directory structures
        possible_dirs = [
            base_path / f"FARM_V2B_EXIT_HYBRID_{quarter}",
            base_path / f"GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_{quarter}",
        ]
        
        for run_dir in possible_dirs:
            if run_dir.exists():
                csv_files = list(run_dir.glob("*merged.csv"))
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                    df['quarter'] = quarter
                    all_trades.append(df)
                    print(f"✅ Loaded {len(df)} trades from {quarter}")
                    break
    
    if not all_trades:
        raise FileNotFoundError("No hybrid trade logs found")
    
    combined = pd.concat(all_trades, ignore_index=True)
    return combined

def extract_routing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features relevant for exit routing decisions.
    
    Features include:
    - Entry-time features: ATR, spread, regime, session
    - Market conditions: volatility, trend
    - Trade characteristics: entry price, side
    """
    features = pd.DataFrame()
    
    # Basic trade info
    features['trade_id'] = df.get('trade_id', df.index)
    features['quarter'] = df.get('quarter', 'UNKNOWN')
    
    # Entry time features (from trade log)
    if 'entry_time' in df.columns:
        features['entry_time'] = pd.to_datetime(df['entry_time'])
        features['entry_hour'] = features['entry_time'].dt.hour
        features['entry_dayofweek'] = features['entry_time'].dt.dayofweek
    
    # ATR and spread features (from entry diagnostics)
    features['atr_bps'] = df.get('atr_bps', np.nan)
    features['spread_bps'] = df.get('spread_bps', np.nan)
    
    # Regime and session
    features['regime'] = df.get('farm_entry_vol_regime', df.get('regime', 'UNKNOWN'))
    features['session'] = df.get('farm_entry_session', df.get('session', 'UNKNOWN'))
    
    # Trade characteristics
    features['side'] = df.get('side', 'UNKNOWN')
    features['entry_price'] = df.get('entry_price', np.nan)
    
    # Entry model outputs (if available)
    features['p_long'] = df.get('p_long', np.nan)
    features['p_short'] = df.get('p_short', np.nan)
    features['margin'] = df.get('margin', np.nan)
    
    # Label: which exit profile was actually used
    features['exit_profile'] = df.get('exit_profile', 'UNKNOWN')
    features['label'] = (features['exit_profile'] == 'FARM_EXIT_V2_RULES_ADAPTIVE_v1').astype(int)
    
    # Outcome metrics (for evaluation)
    features['pnl_bps'] = df.get('pnl_bps', np.nan)
    features['win'] = (features['pnl_bps'] > 0).astype(int)
    features['bars_held'] = df.get('bars_held', np.nan)
    
    return features

def calculate_atr_spread_percentiles(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate percentile thresholds for ATR and spread."""
    atr_values = df['atr_bps'].dropna()
    spread_values = df['spread_bps'].dropna()
    
    if len(atr_values) == 0 or len(spread_values) == 0:
        return {'atr_p50': 0.0, 'atr_p75': 0.0, 'spread_p40': 0.0, 'spread_p60': 0.0}
    
    return {
        'atr_p50': np.percentile(atr_values, 50),
        'atr_p75': np.percentile(atr_values, 75),
        'spread_p40': np.percentile(spread_values, 40),
        'spread_p60': np.percentile(spread_values, 60),
    }

def build_dataset(quarters: List[str], output_dir: Path) -> None:
    """Build exit router training dataset from hybrid trade logs."""
    print("=" * 80)
    print("BUILDING EXIT ROUTER TRAINING DATASET")
    print("=" * 80)
    print()
    
    # Load trade logs
    print("Loading trade logs...")
    df = load_hybrid_trade_logs(quarters)
    
    # Filter to closed long trades only
    df_closed = df[df.get('exit_reason', '') != 'REPLAY_END'].copy()
    df_long = df_closed[df_closed.get('side', '') == 'long'].copy()
    
    print(f"✅ Total closed long trades: {len(df_long)}")
    print()
    
    # Extract features
    print("Extracting routing features...")
    features = extract_routing_features(df_long)
    
    # Calculate percentiles for reference
    percentiles = calculate_atr_spread_percentiles(features)
    print(f"📊 ATR percentiles: P50={percentiles['atr_p50']:.2f}, P75={percentiles['atr_p75']:.2f}")
    print(f"📊 Spread percentiles: P40={percentiles['spread_p40']:.2f}, P60={percentiles['spread_p60']:.2f}")
    print()
    
    # Label distribution
    label_counts = features['label'].value_counts()
    print("📈 Label distribution:")
    print(f"   RULE5 (label=0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(features)*100:.1f}%)")
    print(f"   RULE6A (label=1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(features)*100:.1f}%)")
    print()
    
    # Save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as Parquet (efficient)
    parquet_path = output_dir / "exit_router_training_dataset.parquet"
    features.to_parquet(parquet_path, index=False)
    print(f"✅ Saved Parquet: {parquet_path}")
    
    # Save as CSV (human-readable)
    csv_path = output_dir / "exit_router_training_dataset.csv"
    features.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")
    
    # Save metadata
    metadata = {
        'quarters': quarters,
        'total_trades': int(len(features)),
        'rule5_count': int(label_counts.get(0, 0)),
        'rule6a_count': int(label_counts.get(1, 0)),
        'rule5_pct': float(label_counts.get(0, 0) / len(features) * 100 if len(features) > 0 else 0),
        'rule6a_pct': float(label_counts.get(1, 0) / len(features) * 100 if len(features) > 0 else 0),
        'percentiles': {k: float(v) for k, v in percentiles.items()},
        'feature_columns': list(features.columns),
    }
    
    metadata_path = output_dir / "exit_router_dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata: {metadata_path}")
    print()
    
    # Summary statistics
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print()
    print(f"Total trades: {len(features)}")
    print(f"Features: {len(features.columns)}")
    print(f"RULE5 trades: {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(features)*100:.1f}%)")
    print(f"RULE6A trades: {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(features)*100:.1f}%)")
    print()
    print(f"Output directory: {output_dir}")
    print()
    print("=" * 80)
    print("DATASET BUILD COMPLETE")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Build exit router training dataset")
    parser.add_argument(
        '--quarters',
        nargs='+',
        default=['Q1', 'Q2', 'Q3', 'Q4'],
        help='Quarters to include (default: Q1 Q2 Q3 Q4)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='gx1/data/exit_router',
        help='Output directory for dataset files (default: gx1/data/exit_router)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    build_dataset(args.quarters, output_dir)

if __name__ == "__main__":
    main()
