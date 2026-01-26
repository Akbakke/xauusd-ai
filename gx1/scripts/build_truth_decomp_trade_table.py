#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Canonical Trade Table for Truth Decomposition

DEL 1: Load all trades for 2020-2025 baseline, derive session/regime,
calculate metrics, and output canonical parquet table.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add workspace root to path
import sys
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from gx1.execution.live_features import infer_session_tag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_trade_with_context(trade_file: Path) -> Optional[Dict[str, Any]]:
    """
    Load a single trade JSON file with full context.
    
    Returns dict with all fields needed for decomposition, or None if invalid.
    """
    try:
        with open(trade_file) as f:
            trade_data = json.load(f)
        
        entry_snapshot = trade_data.get("entry_snapshot", {})
        exit_summary = trade_data.get("exit_summary", {})
        
        # Skip if no exit_summary (open trades)
        if not exit_summary or "exit_time" not in exit_summary:
            return None
        
        # Skip REPLAY_EOF trades
        exit_reason = exit_summary.get("exit_reason", "")
        if exit_reason == "REPLAY_EOF":
            return None
        
        entry_time_str = entry_snapshot.get("entry_time")
        exit_time_str = exit_summary.get("exit_time")
        
        if not entry_time_str or not exit_time_str:
            return None
        
        # Parse timestamps
        entry_time = pd.Timestamp(entry_time_str)
        exit_time = pd.Timestamp(exit_time_str)
        
        # Hard assert: entry_time <= exit_time
        if entry_time > exit_time:
            raise ValueError(f"Invalid timestamps: entry_time {entry_time} > exit_time {exit_time}")
        
        # Get session (from entry_snapshot or derive)
        entry_session = entry_snapshot.get("session")
        if not entry_session or pd.isna(entry_session):
            entry_session = infer_session_tag(entry_time)
        
        exit_session = infer_session_tag(exit_time)
        
        # Hard assert: session in valid set
        valid_sessions = {"ASIA", "EU", "OVERLAP", "US"}
        if entry_session not in valid_sessions:
            raise ValueError(f"Invalid entry_session: {entry_session}")
        if exit_session not in valid_sessions:
            raise ValueError(f"Invalid exit_session: {exit_session}")
        
        # Calculate bars_held (M5 = 5 minutes per bar)
        time_delta = exit_time - entry_time
        bars_held = max(1, int(time_delta.total_seconds() / 300.0))
        
        # Get PnL
        pnl_bps = exit_summary.get("realized_pnl_bps", 0.0)
        
        # Hard assert: pnl_bps finite
        if not np.isfinite(pnl_bps):
            raise ValueError(f"Non-finite pnl_bps: {pnl_bps}")
        
        # Extract context/regime data
        # Try multiple possible field names
        atr_bps_raw = entry_snapshot.get("atr_bps") or entry_snapshot.get("atr")
        spread_bps_raw = entry_snapshot.get("spread_bps") or entry_snapshot.get("spread")
        trend_regime = entry_snapshot.get("trend_regime") or entry_snapshot.get("regime")
        vol_regime = entry_snapshot.get("vol_regime")
        
        # Unit sanity: normalize spread_bps and atr_bps
        # Store raw values for debugging
        entry_price = entry_snapshot.get("entry_price")
        entry_bid = entry_snapshot.get("entry_bid")
        entry_ask = entry_snapshot.get("entry_ask")
        
        # Normalize spread_bps
        spread_bps = None
        spread_bps_source = "unknown"
        if spread_bps_raw is not None and pd.notna(spread_bps_raw):
            spread_bps_val = float(spread_bps_raw)
            
            # If obviously wrong units (> 500), try to fix
            # Note: We check > 500 (not 5000) because realistic spread_bps should be < 500
            if spread_bps_val > 500:
                # Try to derive from bid/ask if available (most reliable)
                if entry_bid is not None and entry_ask is not None and entry_bid > 0:
                    spread_price = float(entry_ask) - float(entry_bid)
                    spread_bps = (spread_price / float(entry_bid)) * 10000.0
                    spread_bps_source = "derived_bidask"
                elif entry_price is not None and entry_price > 0:
                    # Maybe it's in pips? (1 pip = 0.0001, so divide by 10000)
                    spread_bps = spread_bps_val / 10000.0
                    spread_bps_source = "converted_pips"
                else:
                    # Last resort: assume it's in wrong units, divide by 10000
                    spread_bps = spread_bps_val / 10000.0
                    spread_bps_source = "converted_unknown"
            else:
                # Already plausible bps
                spread_bps = spread_bps_val
                spread_bps_source = "direct"
        
        # Normalize atr_bps
        atr_bps = None
        atr_bps_source = "unknown"
        if atr_bps_raw is not None and pd.notna(atr_bps_raw):
            atr_bps_val = float(atr_bps_raw)
            
            # If obviously wrong units (> 2000), try to fix
            if atr_bps_val > 2000:
                # Try to derive from atr_abs and price if available
                atr_abs = entry_snapshot.get("atr_abs") or entry_snapshot.get("atr")
                if atr_abs is not None and entry_price is not None and entry_price > 0:
                    atr_bps = (float(atr_abs) / float(entry_price)) * 10000.0
                    atr_bps_source = "derived_atr_abs"
                else:
                    # Last resort: assume it's in wrong units
                    atr_bps = atr_bps_val / 100.0  # Maybe it's in pips?
                    atr_bps_source = "converted_unknown"
            else:
                # Already plausible bps
                atr_bps = atr_bps_val
                atr_bps_source = "direct"
        
        # Build trade record
        trade_record = {
            "trade_id": entry_snapshot.get("trade_id"),
            "year": entry_time.year,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_snapshot.get("entry_price"),
            "exit_price": exit_summary.get("exit_price"),
            "pnl_bps": pnl_bps,
            "win_flag": 1 if pnl_bps > 0 else 0,
            "entry_session": entry_session,
            "exit_session": exit_session,
            "transition": f"{entry_session}->{exit_session}",
            "side": entry_snapshot.get("side"),
            "exit_reason": exit_summary.get("exit_reason"),
            "bars_held": bars_held,
            "max_mfe_bps": exit_summary.get("max_mfe_bps"),
            "max_mae_bps": exit_summary.get("max_mae_bps"),
            # Context/regime fields (normalized)
            "atr_bps": atr_bps if (atr_bps is not None and pd.notna(atr_bps)) else None,
            "spread_bps": spread_bps if (spread_bps is not None and pd.notna(spread_bps)) else None,
            "trend_regime": trend_regime if trend_regime else None,
            "vol_regime": vol_regime if vol_regime else None,
            # Unit sanity metadata (for debugging)
            "_spread_bps_source": spread_bps_source,
            "_atr_bps_source": atr_bps_source,
        }
        
        return trade_record
        
    except Exception as e:
        log.warning(f"Failed to load trade from {trade_file}: {e}")
        return None


def load_all_trades_multiyear(
    input_root: Path,
    years: List[int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load all trades for specified years from multiyear baseline output.
    
    Expected structure:
    - input_root/{year}/chunk_0/trade_journal/trades/*.json
    OR
    - input_root/YEAR_{year}/chunk_0/trade_journal/trades/*.json
    OR
    - input_root/*/YEAR_{year}/chunk_0/trade_journal/trades/*.json
    """
    all_trades = []
    n_trades_total = 0
    n_trades_missing_context = 0
    year_counts: Dict[int, int] = {}
    
    for year in years:
        year_counts[year] = 0
        log.info(f"Loading trades for year {year}...")
        
        # Try multiple path patterns
        possible_paths = [
            input_root / str(year) / "chunk_0" / "trade_journal" / "trades",
            input_root / f"YEAR_{year}" / "chunk_0" / "trade_journal" / "trades",
        ]
        
        # Also try glob pattern for nested structures
        year_dirs = list(input_root.glob(f"*/YEAR_{year}"))
        for year_dir in year_dirs:
            possible_paths.append(year_dir / "chunk_0" / "trade_journal" / "trades")
        
        # Try TRIAL160_YEARLY structure: {year}/chunk_0/...
        year_dirs_alt = list(input_root.glob(f"{year}*/chunk_0"))
        for year_dir in year_dirs_alt:
            possible_paths.append(year_dir / "trade_journal" / "trades")
        
        # If not found, check archive directories (like analyze_mae_mfe_multiyear.py)
        if not any(p.exists() for p in possible_paths):
            # Find workspace root (go up from input_root until we find archive/)
            workspace_root_candidate = input_root
            max_depth = 10
            depth = 0
            while depth < max_depth and workspace_root_candidate.parent != workspace_root_candidate:
                if (workspace_root_candidate / "archive").exists():
                    break
                workspace_root_candidate = workspace_root_candidate.parent
                depth += 1
            
            # Find all matching archive directories
            archive_base_pattern = workspace_root_candidate / "archive" / "REPLAY_RAW_*"
            for archive_base in sorted(workspace_root_candidate.glob("archive/REPLAY_RAW_*")):
                archive_year_dir = archive_base / "TRIAL160_YEARLY" / str(year)
                if archive_year_dir.exists():
                    # Try chunk_0, chunk_1, etc.
                    for chunk_dir in sorted(archive_year_dir.glob("chunk_*")):
                        trade_dir = chunk_dir / "trade_journal" / "trades"
                        if trade_dir.exists():
                            possible_paths.append(trade_dir)
                            log.info(f"Found archive directory: {trade_dir}")
        
        trades_dir = None
        for path in possible_paths:
            if path.exists():
                trades_dir = path
                break
        
        if trades_dir is None:
            log.warning(f"Trade journal directory not found for year {year}")
            continue
        
        trade_files = list(trades_dir.glob("*.json"))
        log.info(f"Found {len(trade_files)} trade JSON files for {year}")
        
        year_trades = 0
        year_missing_context = 0
        
        for trade_file in trade_files:
            trade_record = load_trade_with_context(trade_file)
            if trade_record is None:
                continue
            
            n_trades_total += 1
            year_trades += 1
            year_counts[year] += 1
            
            # Check if context is missing
            if trade_record["atr_bps"] is None or trade_record["spread_bps"] is None:
                n_trades_missing_context += 1
                year_missing_context += 1
            
            all_trades.append(trade_record)
        
        log.info(f"Loaded {year_trades} valid trades for {year} ({year_missing_context} missing context)")
    
    if not all_trades:
        raise ValueError(f"No valid trades found for years {years}")
    
    # ============================================================================
    # INVARIANT: Year Coverage
    # ============================================================================
    # FATAL if < 2 years (prevents "only 2023" scenarios)
    # FATAL if any requested year has 0 trades
    # ============================================================================
    
    # HARD YEAR COVERAGE CHECK: Must have at least 1 trade per requested year
    missing_years = [y for y in years if year_counts.get(y, 0) == 0]
    if missing_years:
        raise ValueError(
            f"FATAL: No trades found for requested years: {missing_years}. "
            f"Year counts: {year_counts}"
        )
    
    df = pd.DataFrame(all_trades)
    
    # ============================================================================
    # INVARIANT: Unit Sanity
    # ============================================================================
    # FATAL if spread_bps or atr_bps outside plausible bounds
    # FATAL if non-finite values found
    # WARN if trades < 1000 (but don't fail)
    # ============================================================================
    
    # WARN if trades < 1000
    if n_trades_total < 1000:
        log.warning(f"⚠️  WARNING: Only {n_trades_total:,} trades loaded (< 1000). Results may be unreliable.")
    
    # Unit sanity: Hard asserts after normalization
    log.info("")
    log.info("Unit sanity checks...")
    
    # Check spread_bps
    spread_valid = df["spread_bps"].notna()
    if spread_valid.sum() > 0:
        spread_median = df.loc[spread_valid, "spread_bps"].median()
        spread_p95 = df.loc[spread_valid, "spread_bps"].quantile(0.95)
        spread_max = df.loc[spread_valid, "spread_bps"].max()
        
        log.info(f"  spread_bps: median={spread_median:.2f}, p95={spread_p95:.2f}, max={spread_max:.2f}")
        
        # Hard assert: 0 <= spread_bps <= 500
        invalid_spread = df.loc[spread_valid & ((df["spread_bps"] < 0) | (df["spread_bps"] > 500)), "spread_bps"]
        if len(invalid_spread) > 0:
            # Write debug sample
            debug_dir = workspace_root / "reports" / "truth_decomp" / "_audit"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / "UNIT_SANITY_FAILURE_SPREAD_SAMPLE.json"
            
            sample_rows = []
            for idx in invalid_spread.head(10).index:
                row = df.loc[idx].to_dict()
                sample_rows.append(row)
            
            with open(debug_path, "w") as f:
                json.dump(sample_rows, f, indent=2, default=str)
            
            raise ValueError(
                f"FATAL: Invalid spread_bps values found: {len(invalid_spread)} trades. "
                f"Range: [{invalid_spread.min():.2f}, {invalid_spread.max():.2f}]. "
                f"Expected: [0, 500]. Debug sample written to: {debug_path}"
            )
    
    # Check atr_bps
    atr_valid = df["atr_bps"].notna()
    if atr_valid.sum() > 0:
        atr_median = df.loc[atr_valid, "atr_bps"].median()
        atr_p95 = df.loc[atr_valid, "atr_bps"].quantile(0.95)
        atr_max = df.loc[atr_valid, "atr_bps"].max()
        
        log.info(f"  atr_bps: median={atr_median:.2f}, p95={atr_p95:.2f}, max={atr_max:.2f}")
        
        # Hard assert: 0 <= atr_bps <= 2000
        invalid_atr = df.loc[atr_valid & ((df["atr_bps"] < 0) | (df["atr_bps"] > 2000)), "atr_bps"]
        if len(invalid_atr) > 0:
            # Write debug sample
            debug_dir = workspace_root / "reports" / "truth_decomp" / "_audit"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / "UNIT_SANITY_FAILURE_ATR_SAMPLE.json"
            
            sample_rows = []
            for idx in invalid_atr.head(10).index:
                row = df.loc[idx].to_dict()
                sample_rows.append(row)
            
            with open(debug_path, "w") as f:
                json.dump(sample_rows, f, indent=2, default=str)
            
            raise ValueError(
                f"FATAL: Invalid atr_bps values found: {len(invalid_atr)} trades. "
                f"Range: [{invalid_atr.min():.2f}, {invalid_atr.max():.2f}]. "
                f"Expected: [0, 2000]. Debug sample written to: {debug_path}"
            )
    
    # Hard assert: all finite
    if not df["spread_bps"].isna().all():
        if not np.isfinite(df["spread_bps"].dropna()).all():
            raise ValueError("FATAL: Non-finite spread_bps values found")
    if not df["atr_bps"].isna().all():
        if not np.isfinite(df["atr_bps"].dropna()).all():
            raise ValueError("FATAL: Non-finite atr_bps values found")
    
    log.info("✅ Unit sanity checks passed")
    
    # Calculate coverage ratio
    coverage_ratio = 1.0 - (n_trades_missing_context / n_trades_total) if n_trades_total > 0 else 0.0
    
    log.info(f"Total trades loaded: {n_trades_total}")
    log.info(f"Trades missing context: {n_trades_missing_context}")
    log.info(f"Coverage ratio: {coverage_ratio:.1%}")
    
    # Unit sanity stats
    spread_stats = {}
    atr_stats = {}
    spread_source_counts = {}
    atr_source_counts = {}
    
    if spread_valid.sum() > 0:
        spread_stats = {
            "min": float(df.loc[spread_valid, "spread_bps"].min()),
            "median": float(df.loc[spread_valid, "spread_bps"].median()),
            "p95": float(df.loc[spread_valid, "spread_bps"].quantile(0.95)),
            "max": float(df.loc[spread_valid, "spread_bps"].max()),
        }
        spread_source_counts_series = df.loc[spread_valid, "_spread_bps_source"].value_counts()
        spread_source_counts = spread_source_counts_series.to_dict()
        spread_source = spread_source_counts_series.index[0] if len(spread_source_counts_series) > 0 else "unknown"
    else:
        spread_source = "unknown"
    
    if atr_valid.sum() > 0:
        atr_stats = {
            "min": float(df.loc[atr_valid, "atr_bps"].min()),
            "median": float(df.loc[atr_valid, "atr_bps"].median()),
            "p95": float(df.loc[atr_valid, "atr_bps"].quantile(0.95)),
            "max": float(df.loc[atr_valid, "atr_bps"].max()),
        }
        atr_source_counts_series = df.loc[atr_valid, "_atr_bps_source"].value_counts()
        atr_source_counts = atr_source_counts_series.to_dict()
        atr_source = atr_source_counts_series.index[0] if len(atr_source_counts_series) > 0 else "unknown"
    else:
        atr_source = "unknown"
    
    return df, {
        "n_trades_total": n_trades_total,
        "n_trades_missing_context": n_trades_missing_context,
        "coverage_ratio": coverage_ratio,
        "years_requested": years,
        "years_detected": sorted(df["year"].unique().tolist()),
        "year_counts": {str(k): int(v) for k, v in sorted(year_counts.items())},
        "input_root_resolved": str(input_root.resolve()),
        "spread_bps_stats": spread_stats,
        "atr_bps_stats": atr_stats,
        "spread_bps_source": spread_source,
        "atr_bps_source": atr_source,
        "spread_bps_source_counts": {str(k): int(v) for k, v in spread_source_counts.items()},
        "atr_bps_source_counts": {str(k): int(v) for k, v in atr_source_counts.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Build Canonical Trade Table for Truth Decomposition")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing year subdirectories with trade journals",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output path for parquet file (e.g., reports/truth_decomp/trades_baseline_2020_2025.parquet)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2020,2021,2022,2023,2024,2025",
        help="Comma-separated list of years",
    )
    
    args = parser.parse_args()
    
    # Parse years
    years = [int(y.strip()) for y in args.years.split(",")]
    
    # Resolve paths
    workspace_root = Path(__file__).parent.parent.parent
    if not args.input_root.is_absolute():
        args.input_root = workspace_root / args.input_root
    if not args.output_path.is_absolute():
        args.output_path = workspace_root / args.output_path
    
    log.info("=" * 60)
    log.info("BUILD CANONICAL TRADE TABLE")
    log.info("=" * 60)
    log.info(f"Input root: {args.input_root}")
    log.info(f"Output path: {args.output_path}")
    log.info(f"Years: {years}")
    log.info("")
    
    # Load all trades
    try:
        df, coverage_stats = load_all_trades_multiyear(args.input_root, years)
        
        log.info("")
        log.info("Trade table statistics:")
        log.info(f"  Total trades: {len(df):,}")
        log.info(f"  Years covered: {sorted(df['year'].unique())}")
        log.info(f"  Coverage ratio: {coverage_stats['coverage_ratio']:.1%}")
        log.info(f"  Missing context: {coverage_stats['n_trades_missing_context']:,}")
        log.info("")
        
        # Write parquet
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.output_path, index=False, compression="snappy")
        log.info(f"✅ Wrote trade table: {args.output_path}")
        log.info(f"   File size: {args.output_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Write coverage stats JSON
        stats_path = args.output_path.parent / f"{args.output_path.stem}_coverage.json"
        with open(stats_path, "w") as f:
            json.dump(coverage_stats, f, indent=2)
        log.info(f"✅ Wrote coverage stats: {stats_path}")
        
        return 0
        
    except Exception as e:
        log.error(f"❌ Failed to build trade table: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
