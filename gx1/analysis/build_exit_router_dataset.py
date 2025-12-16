#!/usr/bin/env python3
"""
Build training dataset for ML-based exit router.

This script compares RULE5 and RULE6A replays to build a dataset with
both pnl_rule5_bps and pnl_rule6a_bps for each trade, allowing us to
train a model that predicts which exit policy would be best.

Input: Trade logs from separate RULE5 and RULE6A replays (Q1-Q4 2025)
Output: Training dataset with features and labels for exit routing decisions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import json

def load_rule5_trade_logs(quarters: List[str]) -> pd.DataFrame:
    """Load trade logs from RULE5 replays for specified quarters."""
    base_path = Path("gx1/wf_runs")
    all_trades = []
    
    for quarter in quarters:
        possible_dirs = [
            base_path / f"FARM_V2B_EXIT_A_{quarter}_RULE5",
            base_path / f"GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_{quarter}_RULE5",
        ]
        
        for run_dir in possible_dirs:
            if run_dir.exists():
                csv_files = list(run_dir.glob("*merged.csv"))
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                    df['quarter'] = quarter
                    all_trades.append(df)
                    print(f"✅ Loaded RULE5: {len(df)} trades from {quarter}")
                    break
    
    if not all_trades:
        raise FileNotFoundError("No RULE5 trade logs found")
    
    combined = pd.concat(all_trades, ignore_index=True)
    return combined

def load_rule6a_trade_logs(quarters: List[str]) -> pd.DataFrame:
    """Load trade logs from RULE6A (ADAPTIVE) replays for specified quarters."""
    base_path = Path("gx1/wf_runs")
    all_trades = []
    
    for quarter in quarters:
        possible_dirs = [
            base_path / f"FARM_V2B_EXIT_A_{quarter}_RULE6",
            base_path / f"GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_{quarter}_RULE6",
            base_path / f"GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_ADAPTIVE_{quarter}",
        ]
        
        for run_dir in possible_dirs:
            if run_dir.exists():
                csv_files = list(run_dir.glob("*merged.csv"))
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                    df['quarter'] = quarter
                    all_trades.append(df)
                    print(f"✅ Loaded RULE6A: {len(df)} trades from {quarter}")
                    break
    
    if not all_trades:
        raise FileNotFoundError("No RULE6A trade logs found")
    
    combined = pd.concat(all_trades, ignore_index=True)
    return combined

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

def extract_routing_features_from_hybrid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from hybrid replay (fallback when separate replays not available).
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

def merge_rule5_rule6a_trades(df_rule5: pd.DataFrame, df_rule6a: pd.DataFrame) -> pd.DataFrame:
    """
    Merge RULE5 and RULE6A trade logs on entry_time and quarter to get both PnL values for each trade.
    Note: trade_ids may differ between replays, so we match on entry_time instead.
    """
    # Filter to closed long trades only
    df5_closed = df_rule5[df_rule5.get('exit_reason', '') != 'REPLAY_END'].copy()
    df5_long = df5_closed[df5_closed.get('side', '') == 'long'].copy()
    
    df6_closed = df_rule6a[df_rule6a.get('exit_reason', '') != 'REPLAY_END'].copy()
    df6_long = df6_closed[df6_closed.get('side', '') == 'long'].copy()
    
    # Convert entry_time to datetime for matching
    df5_long['entry_time'] = pd.to_datetime(df5_long['entry_time'])
    df6_long['entry_time'] = pd.to_datetime(df6_long['entry_time'])
    
    # Select columns that exist in RULE5 data
    rule5_cols = ['trade_id', 'quarter', 'entry_time', 'atr_bps', 'entry_price', 'pnl_bps']
    if 'spread_bps' in df5_long.columns:
        rule5_cols.append('spread_bps')
    if 'farm_entry_vol_regime' in df5_long.columns:
        rule5_cols.append('farm_entry_vol_regime')
    if 'farm_entry_session' in df5_long.columns:
        rule5_cols.append('farm_entry_session')
    if 'regime' in df5_long.columns:
        rule5_cols.append('regime')
    if 'session' in df5_long.columns:
        rule5_cols.append('session')
    
    # Merge on entry_time and quarter (trade_ids may differ between replays)
    merged = pd.merge(
        df5_long[rule5_cols].copy(),
        df6_long[['entry_time', 'quarter', 'pnl_bps']].copy(),
        on=['entry_time', 'quarter'],
        how='inner',
        suffixes=('_rule5', '_rule6a')
    )
    
    return merged

def extract_routing_features(df_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features relevant for exit routing decisions from merged RULE5/RULE6A data.
    
    Features include:
    - Entry-time features: ATR, spread, regime, session
    - Market conditions: volatility, trend
    - Trade characteristics: entry price, side
    - Labels: best_policy (RULE5 or RULE6A based on which had better PnL)
    """
    features = pd.DataFrame()
    
    # Basic trade info
    features['trade_id'] = df_merged['trade_id'].values
    features['quarter'] = df_merged['quarter'].values
    
    # Entry time features
    if 'entry_time' in df_merged.columns:
        entry_times = pd.to_datetime(df_merged['entry_time'])
        features['entry_time'] = entry_times.values
        features['entry_hour'] = entry_times.dt.hour.values
        features['entry_dayofweek'] = entry_times.dt.dayofweek.values
    
    # ATR and spread (convert to percentiles later)
    if 'atr_bps' in df_merged.columns:
        features['atr_bps'] = df_merged['atr_bps'].values
    else:
        features['atr_bps'] = np.nan
    
    if 'spread_bps' in df_merged.columns:
        features['spread_bps'] = df_merged['spread_bps'].values
    else:
        features['spread_bps'] = np.nan
    
    # Regime and session (try different column names)
    if 'farm_entry_vol_regime' in df_merged.columns:
        features['regime'] = df_merged['farm_entry_vol_regime'].values
    elif 'regime' in df_merged.columns:
        features['regime'] = df_merged['regime'].values
    else:
        features['regime'] = np.full(len(df_merged), 'UNKNOWN')
    
    if 'farm_entry_session' in df_merged.columns:
        features['session'] = df_merged['farm_entry_session'].values
    elif 'session' in df_merged.columns:
        features['session'] = df_merged['session'].values
    else:
        features['session'] = np.full(len(df_merged), 'UNKNOWN')
    
    # Entry price
    if 'entry_price' in df_merged.columns:
        features['entry_price'] = df_merged['entry_price'].values
    else:
        features['entry_price'] = np.nan
    
    # PnL for both policies
    features['pnl_rule5_bps'] = df_merged['pnl_bps_rule5'].values
    features['pnl_rule6a_bps'] = df_merged['pnl_bps_rule6a'].values
    
    # Determine best policy (label)
    features['best_policy'] = np.where(
        features['pnl_rule6a_bps'] > features['pnl_rule5_bps'],
        'RULE6A',
        'RULE5'
    )
    
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

def create_atr_bucket(atr_bps: float, percentiles: Dict[str, float]) -> str:
    """Create ATR bucket based on percentiles."""
    if pd.isna(atr_bps):
        return 'UNKNOWN'
    p50 = percentiles.get('atr_p50', 0.0)
    p75 = percentiles.get('atr_p75', 0.0)
    
    if atr_bps <= p50:
        return 'LOW'
    elif atr_bps <= p75:
        return 'MEDIUM'
    else:
        return 'HIGH'

def convert_to_percentiles(df: pd.DataFrame, percentiles: Dict[str, float]) -> pd.DataFrame:
    """Convert ATR and spread from bps to percentiles."""
    df = df.copy()
    
    # Calculate ATR percentiles
    atr_values = df['atr_bps'].dropna()
    if len(atr_values) > 0:
        df['atr_pct'] = df['atr_bps'].apply(lambda x: (atr_values <= x).sum() / len(atr_values) * 100 if not pd.isna(x) else np.nan)
        df['atr_bucket'] = df['atr_bps'].apply(lambda x: create_atr_bucket(x, percentiles))
    else:
        df['atr_pct'] = np.nan
        df['atr_bucket'] = 'UNKNOWN'
    
    # Calculate spread percentiles
    spread_values = df['spread_bps'].dropna()
    if len(spread_values) > 0:
        df['spread_pct'] = df['spread_bps'].apply(lambda x: (spread_values <= x).sum() / len(spread_values) * 100 if not pd.isna(x) else np.nan)
    else:
        df['spread_pct'] = np.nan
    
    return df

def build_dataset(quarters: List[str], output_dir: Path, use_separate_replays: bool = True) -> None:
    """Build exit router training dataset by comparing RULE5 and RULE6A replays."""
    print("=" * 80)
    print("BUILDING EXIT ROUTER TRAINING DATASET")
    print("=" * 80)
    print()
    
    if use_separate_replays:
        # Try to load separate RULE5 and RULE6A replays
        try:
            print("Loading separate RULE5 and RULE6A replays...")
            df_rule5 = load_rule5_trade_logs(quarters)
            df_rule6a = load_rule6a_trade_logs(quarters)
            
            print(f"✅ RULE5 trades: {len(df_rule5)}")
            print(f"✅ RULE6A trades: {len(df_rule6a)}")
            print()
            
            # Merge on trade_id to get both PnL values
            print("Merging RULE5 and RULE6A trades...")
            df_merged = merge_rule5_rule6a_trades(df_rule5, df_rule6a)
            print(f"✅ Matched trades: {len(df_merged)}")
            print()
            
            # Extract features
            print("Extracting routing features...")
            features = extract_routing_features(df_merged)
            
        except FileNotFoundError as e:
            print(f"⚠️  Separate replays not found: {e}")
            print("Falling back to hybrid replay data...")
            use_separate_replays = False
    
    if not use_separate_replays:
        # Fallback to hybrid replay
        print("Loading hybrid trade logs...")
        df = load_hybrid_trade_logs(quarters)
        
        # Filter to closed long trades only
        df_closed = df[df.get('exit_reason', '') != 'REPLAY_END'].copy()
        df_long = df_closed[df_closed.get('side', '') == 'long'].copy()
        
        print(f"✅ Total closed long trades: {len(df_long)}")
        print()
        
        # Extract features (without pnl_rule5_bps and pnl_rule6a_bps)
        print("Extracting routing features...")
        features = extract_routing_features_from_hybrid(df_long)
    
    # Calculate percentiles for ATR/spread buckets
    percentiles = calculate_atr_spread_percentiles(features)
    print(f"📊 ATR percentiles: P50={percentiles['atr_p50']:.2f}, P75={percentiles['atr_p75']:.2f}")
    print(f"📊 Spread percentiles: P40={percentiles['spread_p40']:.2f}, P60={percentiles['spread_p60']:.2f}")
    print()
    
    # Convert to percentiles and create buckets
    if 'atr_bps' in features.columns:
        features = convert_to_percentiles(features, percentiles)
    
    # Clean NaN values before saving
    print("Cleaning NaN values...")
    core_numeric = ["atr_pct", "spread_pct"]
    core_cat = ["atr_bucket", "regime", "session"]
    
    # Hard requirement: atr_pct MUST exist, otherwise drop the trade
    initial_count = len(features)
    features = features.dropna(subset=["atr_pct"]).copy()
    dropped_count = initial_count - len(features)
    if dropped_count > 0:
        print(f"⚠️  Dropped {dropped_count} trades with missing atr_pct")
    
    # For spread_pct: better to assume "high spread" than to drop
    if 'spread_pct' in features.columns:
        features["spread_pct"] = features["spread_pct"].fillna(100.0)
        print(f"✅ Filled {features['spread_pct'].isna().sum()} missing spread_pct values with 100.0 (high spread)")
    
    # For categories: fill with 'UNKNOWN'
    for col in core_cat:
        if col in features.columns:
            nan_count = features[col].isna().sum()
            features[col] = features[col].fillna("UNKNOWN")
            if nan_count > 0:
                print(f"✅ Filled {nan_count} missing {col} values with 'UNKNOWN'")
    print()
    
    # Label distribution
    if len(features) > 0:
        if 'best_policy' in features.columns:
            label_counts = features['best_policy'].value_counts()
            print("📈 Best policy distribution:")
            print(f"   RULE5: {label_counts.get('RULE5', 0)} ({label_counts.get('RULE5', 0)/len(features)*100:.1f}%)")
            print(f"   RULE6A: {label_counts.get('RULE6A', 0)} ({label_counts.get('RULE6A', 0)/len(features)*100:.1f}%)")
        elif 'label' in features.columns:
            label_counts = features['label'].value_counts()
            print("📈 Label distribution:")
            print(f"   RULE5 (label=0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(features)*100:.1f}%)")
            print(f"   RULE6A (label=1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(features)*100:.1f}%)")
    else:
        print("⚠️  No matched trades found!")
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
        'use_separate_replays': use_separate_replays,
        'percentiles': {k: float(v) for k, v in percentiles.items()},
        'feature_columns': list(features.columns),
    }
    
    if 'best_policy' in features.columns:
        label_counts = features['best_policy'].value_counts()
        metadata['rule5_count'] = int(label_counts.get('RULE5', 0))
        metadata['rule6a_count'] = int(label_counts.get('RULE6A', 0))
        metadata['rule5_pct'] = float(label_counts.get('RULE5', 0) / len(features) * 100 if len(features) > 0 else 0)
        metadata['rule6a_pct'] = float(label_counts.get('RULE6A', 0) / len(features) * 100 if len(features) > 0 else 0)
    elif 'label' in features.columns:
        label_counts = features['label'].value_counts()
        metadata['rule5_count'] = int(label_counts.get(0, 0))
        metadata['rule6a_count'] = int(label_counts.get(1, 0))
        metadata['rule5_pct'] = float(label_counts.get(0, 0) / len(features) * 100 if len(features) > 0 else 0)
        metadata['rule6a_pct'] = float(label_counts.get(1, 0) / len(features) * 100 if len(features) > 0 else 0)
    
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
    if 'best_policy' in features.columns:
        label_counts = features['best_policy'].value_counts()
        print(f"RULE5 trades: {label_counts.get('RULE5', 0)} ({label_counts.get('RULE5', 0)/len(features)*100:.1f}%)")
        print(f"RULE6A trades: {label_counts.get('RULE6A', 0)} ({label_counts.get('RULE6A', 0)/len(features)*100:.1f}%)")
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
    parser.add_argument(
        '--use-separate-replays',
        action='store_true',
        default=True,
        help='Use separate RULE5 and RULE6A replays (default: True)'
    )
    parser.add_argument(
        '--use-hybrid-replay',
        action='store_true',
        default=False,
        help='Use hybrid replay instead of separate replays (default: False)'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    use_separate = args.use_separate_replays and not args.use_hybrid_replay
    build_dataset(args.quarters, output_dir, use_separate_replays=use_separate)

if __name__ == "__main__":
    main()
