#!/usr/bin/env python3
"""
ENTRY_V10 Label Quality Analysis

Analyzes V10 p_long predictions vs actual trade PnL from replay results.
Divides predictions into quantiles and shows real trade outcomes.

Usage:
    python -m gx1.tools.debug.analyze_entry_v10_label_quality \
        --v10-dataset data/entry_v10/entry_v10_dataset_val.parquet \
        --v10-model models/entry_v10/entry_v10_transformer.pt \
        --v10-meta models/entry_v10/entry_v10_transformer_meta.json \
        --replay-trade-log runs/replay_shadow/SNIPER_P4_1_V10_HYBRID/trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv \
        --output-report reports/rl/entry_v10/ENTRY_V10_LABEL_QUALITY.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from gx1.rl.entry_v10.dataset_v10 import EntryV10Dataset
from gx1.models.entry_v10.entry_v10_hybrid_transformer import EntryV10HybridTransformer


def load_v10_model(model_path: Path, meta_path: Path, device: torch.device) -> torch.nn.Module:
    """Load V10 Transformer model."""
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Get variant from metadata (default to v10 for backward compatibility)
    variant = meta.get("variant", "v10")
    
    # Use default config matching training
    model = EntryV10HybridTransformer(
        seq_input_dim=meta.get("seq_feature_count", 16),
        snap_input_dim=meta.get("snap_feature_count", 88),
        max_seq_len=meta.get("seq_len", 30),
        variant=variant,
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    return model


def get_v10_predictions(
    model: torch.nn.Module,
    dataset: EntryV10Dataset,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get V10 predictions with metadata."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_preds = []
    all_targets = []
    session_ids = []
    vol_regime_ids = []
    trend_regime_ids = []
    timestamps = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            seq_x = batch["seq_x"].to(device)
            snap_x = batch["snap_x"].to(device)
            session_id = batch["session_id"].to(device)
            vol_regime_id = batch["vol_regime_id"].to(device)
            trend_regime_id = batch["trend_regime_id"].to(device)
            y = batch.get("y_direction", batch.get("y")).to(device)
            
            output = model(seq_x, snap_x, session_id, vol_regime_id, trend_regime_id)
            direction_logit = output["direction_logit"].squeeze(-1)
            probs = torch.sigmoid(direction_logit)
            
            # V10.1 design: HARD FEIL on NaN/Inf - no fallback
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                raise RuntimeError(
                    "CRITICAL: NaN/Inf in label quality analysis predictions. "
                    "NO FALLBACK - HARD FEIL."
                )
            
            probs_np = probs.cpu().numpy()
            # Check for constant predictions (std < 1e-6)
            if len(probs_np) > 1 and np.std(probs_np) < 1e-6:
                raise RuntimeError(
                    "CRITICAL: Predictions are effectively constant "
                    f"(std={np.std(probs_np):.2e}). NO FALLBACK - HARD FEIL."
                )
            
            all_preds.extend(probs_np)
            all_targets.extend(y.cpu().numpy())
            session_ids.extend(session_id.cpu().numpy())
            vol_regime_ids.extend(vol_regime_id.cpu().numpy())
            trend_regime_ids.extend(trend_regime_id.cpu().numpy())
    
    # Load timestamps from dataset
    df = pd.read_parquet(dataset.parquet_path)
    if "ts" in df.columns:
        timestamps = df["ts"].values
    elif "time" in df.columns:
        timestamps = df["time"].values
    else:
        timestamps = np.arange(len(df))
    
    return (
        np.array(all_preds),
        np.array(all_targets),
        np.array(session_ids),
        np.array(vol_regime_ids),
        np.array(trend_regime_ids),
        np.array(timestamps),
    )


def load_replay_trade_log(trade_log_path: Path) -> pd.DataFrame:
    """
    Load replay trade log and extract PnL data.
    
    Supports both:
    - CSV trade log (e.g., trade_log.csv)
    - Trade journal directory (trade_journal_index.csv or trade_journal/trades/*.json)
    """
    print(f"[LOAD] Loading trade log from: {trade_log_path}")
    
    # Check if it's a directory (trade journal)
    if trade_log_path.is_dir():
        # Try trade_journal_index.csv first
        index_csv = trade_log_path / "trade_journal_index.csv"
        if index_csv.exists():
            print(f"[LOAD] Loading from trade journal index: {index_csv}")
            df = pd.read_csv(index_csv, on_bad_lines='skip', engine='python')
        else:
            # Try trade_journal subdirectory
            journal_dir = trade_log_path / "trade_journal"
            if journal_dir.exists():
                index_csv = journal_dir / "trade_journal_index.csv"
                if index_csv.exists():
                    print(f"[LOAD] Loading from trade journal index: {index_csv}")
                    df = pd.read_csv(index_csv, on_bad_lines='skip', engine='python')
                else:
                    # Load from JSON files
                    trades_dir = journal_dir / "trades"
                    if trades_dir.exists():
                        print(f"[LOAD] Loading from trade journal JSON files: {trades_dir}")
                        trades = []
                        for json_file in sorted(trades_dir.glob("*.json")):
                            try:
                                with open(json_file, "r") as f:
                                    trade = json.load(f)
                                    if isinstance(trade, dict):
                                        # Convert trade dict to row
                                        row = {
                                            "trade_id": trade.get("trade_id", ""),
                                            "entry_time": trade.get("entry_time", ""),
                                            "exit_time": trade.get("exit_time", ""),
                                            "pnl_bps": trade.get("pnl_bps", None),
                                            "side": trade.get("side", ""),
                                        }
                                        # Add entry prediction if available
                                        entry = trade.get("entry", {})
                                        if "p_long_v10" in entry:
                                            row["p_long_v10"] = entry["p_long_v10"]
                                        elif "p_long_v10_1" in entry:
                                            row["p_long_v10"] = entry["p_long_v10_1"]
                                        elif "p_long" in entry:
                                            row["p_long_v10"] = entry["p_long"]
                                        trades.append(row)
                            except Exception as e:
                                print(f"[LOAD] Warning: Failed to load {json_file}: {e}")
                        df = pd.DataFrame(trades)
                    else:
                        raise FileNotFoundError(f"No trade journal found in {trade_log_path}")
            else:
                raise FileNotFoundError(f"No trade journal found in {trade_log_path}")
    else:
        # Assume CSV file
        df = pd.read_csv(trade_log_path, on_bad_lines='skip', engine='python')
    
    # Filter closed trades
    closed = df[
        df["exit_time"].notna() &
        df["pnl_bps"].notna() &
        (df["pnl_bps"] != "")
    ].copy()
    
    # Convert pnl_bps to float
    closed["pnl_bps"] = pd.to_numeric(closed["pnl_bps"], errors='coerce')
    closed = closed[closed["pnl_bps"].notna()]
    
    # Convert entry_time to datetime
    if "entry_time" in closed.columns:
        closed["entry_time"] = pd.to_datetime(closed["entry_time"], errors='coerce')
    
    print(f"[LOAD] Loaded {len(closed)} closed trades")
    return closed


def match_predictions_to_trades(
    preds: np.ndarray,
    timestamps: np.ndarray,
    trade_log: pd.DataFrame,
    time_tolerance_minutes: int = 5,
) -> pd.DataFrame:
    """Match V10 predictions to actual trades by timestamp."""
    print(f"[MATCH] Matching {len(preds)} predictions to {len(trade_log)} trades...")
    
    # Create DataFrame with predictions
    pred_df = pd.DataFrame({
        "ts": pd.to_datetime(timestamps, utc=True),
        "p_long_v10": preds,
    })
    
    # Normalize timestamps - remove timezone if present, ensure UTC
    pred_df["ts"] = pd.to_datetime(pred_df["ts"], utc=True)
    
    # Merge on timestamp (within tolerance)
    trade_log["entry_time"] = pd.to_datetime(trade_log["entry_time"], errors='coerce', utc=True)
    trade_log = trade_log[trade_log["entry_time"].notna()]
    
    # Ensure both are timezone-aware UTC
    if pred_df["ts"].dt.tz is None:
        pred_df["ts"] = pred_df["ts"].dt.tz_localize("UTC")
    else:
        pred_df["ts"] = pred_df["ts"].dt.tz_convert("UTC")
    
    if trade_log["entry_time"].dt.tz is None:
        trade_log["entry_time"] = trade_log["entry_time"].dt.tz_localize("UTC")
    else:
        trade_log["entry_time"] = trade_log["entry_time"].dt.tz_convert("UTC")
    
    # Merge with tolerance
    merged = pd.merge_asof(
        trade_log.sort_values("entry_time"),
        pred_df.sort_values("ts"),
        left_on="entry_time",
        right_on="ts",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=time_tolerance_minutes),
    )
    
    # Filter out unmatched
    matched = merged[merged["p_long_v10"].notna()].copy()
    
    print(f"[MATCH] Matched {len(matched)} trades ({len(matched)/len(trade_log)*100:.1f}%)")
    return matched


def analyze_quantiles(
    matched_trades: pd.DataFrame,
    n_quantiles: int = 5,
    min_trades_per_quantile: int = 30,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze trade PnL by V10/V10.1 p_long quantiles.
    
    Args:
        matched_trades: DataFrame with matched trades
        n_quantiles: Number of quantiles
        min_trades_per_quantile: Minimum trades required per quantile (HARD FAIL if below)
    
    Returns:
        Tuple of (quantile_stats DataFrame, quantile_metadata dict)
    """
    print(f"[ANALYZE] Analyzing {n_quantiles} quantiles (min_trades_per_quantile={min_trades_per_quantile})...")
    
    # Create quantiles
    matched_trades["quantile"] = pd.qcut(
        matched_trades["p_long_v10"],
        q=n_quantiles,
        labels=[f"Q{i+1}" for i in range(n_quantiles)],
        duplicates="drop",
    )
    
    # Group by quantile
    quantile_stats = []
    quantile_metadata = {
        "n_total_trades": len(matched_trades),
        "n_quantiles": n_quantiles,
        "min_trades_per_quantile": min_trades_per_quantile,
        "insufficient_data_quantiles": [],
    }
    
    for q in matched_trades["quantile"].cat.categories:
        q_trades = matched_trades[matched_trades["quantile"] == q]
        n_trades = len(q_trades)
        
        # HARD FAIL: Check minimum trades per quantile
        if n_trades < min_trades_per_quantile:
            print(f"⚠️  WARNING: Quantile {q} has only {n_trades} trades (< {min_trades_per_quantile})")
            quantile_metadata["insufficient_data_quantiles"].append({
                "quantile": q,
                "n_trades": n_trades,
                "status": "insufficient_data",
            })
            # Mark as insufficient but continue (don't fail hard here, let user decide)
            stats = {
                "quantile": q,
                "n_trades": n_trades,
                "status": "insufficient_data",
                "p_long_min": float(q_trades["p_long_v10"].min()) if n_trades > 0 else None,
                "p_long_max": float(q_trades["p_long_v10"].max()) if n_trades > 0 else None,
                "p_long_mean": float(q_trades["p_long_v10"].mean()) if n_trades > 0 else None,
                "pnl_mean": None,
                "pnl_median": None,
                "pnl_std": None,
                "win_rate": None,
                "pnl_p05": None,
                "pnl_p95": None,
            }
            quantile_stats.append(stats)
            continue
        
        # Sufficient data - compute stats
        stats = {
            "quantile": q,
            "n_trades": n_trades,
            "status": "success",
            "p_long_min": float(q_trades["p_long_v10"].min()),
            "p_long_max": float(q_trades["p_long_v10"].max()),
            "p_long_mean": float(q_trades["p_long_v10"].mean()),
            "pnl_mean": float(q_trades["pnl_bps"].mean()),
            "pnl_median": float(q_trades["pnl_bps"].median()),
            "pnl_std": float(q_trades["pnl_bps"].std()),
            "win_rate": float((q_trades["pnl_bps"] > 0).mean()),
            "pnl_p05": float(q_trades["pnl_bps"].quantile(0.05)),
            "pnl_p95": float(q_trades["pnl_bps"].quantile(0.95)),
        }
        quantile_stats.append(stats)
    
    return pd.DataFrame(quantile_stats), quantile_metadata


def analyze_regime_weakest(
    matched_trades: pd.DataFrame,
    regime_col: str = "trend_regime",
) -> pd.DataFrame:
    """Analyze weakest regimes (NEUTRAL×MEDIUM/HIGH)."""
    print(f"[REGIME] Analyzing weakest regimes...")
    
    # Map regime IDs to names if needed
    trend_map = {0: "TREND_UP", 1: "TREND_DOWN", 2: "TREND_NEUTRAL", 3: "CHOP", 4: "MIXED"}
    vol_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
    
    # Add regime name columns if they don't exist
    if "trend_regime" not in matched_trades.columns and "trend_regime_id" in matched_trades.columns:
        matched_trades["trend_regime"] = matched_trades["trend_regime_id"].map(trend_map).fillna("UNKNOWN")
    if "vol_regime" not in matched_trades.columns and "vol_regime_id" in matched_trades.columns:
        matched_trades["vol_regime"] = matched_trades["vol_regime_id"].map(vol_map).fillna("UNKNOWN")
    
    # Filter weakest regimes
    # Get columns safely (use DataFrame columns, not .get() which returns scalar)
    trend_col = matched_trades["trend_regime"] if "trend_regime" in matched_trades.columns else pd.Series(["UNKNOWN"] * len(matched_trades), index=matched_trades.index)
    vol_col = matched_trades["vol_regime"] if "vol_regime" in matched_trades.columns else pd.Series(["UNKNOWN"] * len(matched_trades), index=matched_trades.index)
    
    weakest = matched_trades[
        (trend_col.astype(str).str.contains("NEUTRAL", case=False, na=False)) &
        (vol_col.astype(str).isin(["MEDIUM", "HIGH"]))
    ].copy()
    
    if len(weakest) == 0:
        print("[REGIME] No weakest regime trades found")
        return pd.DataFrame()
    
    print(f"[REGIME] Found {len(weakest)} trades in NEUTRAL×MEDIUM/HIGH")
    
    # Auto-determine threshold (median of p_long_v10)
    threshold_auto = matched_trades["p_long_v10"].median()
    print(f"[REGIME] Using auto threshold: {threshold_auto:.3f}")
    
    # Compare V10-selected vs random
    v10_selected = weakest[weakest["p_long_v10"] >= threshold_auto]
    random_sample = weakest.sample(n=min(len(v10_selected), len(weakest)), random_state=42)
    
    stats = {
        "regime": "NEUTRAL×MEDIUM/HIGH",
        "threshold": threshold_auto,
        "n_total": len(weakest),
        "n_v10_selected": len(v10_selected),
        "n_random": len(random_sample),
        "v10_pnl_mean": v10_selected["pnl_bps"].mean() if len(v10_selected) > 0 else 0.0,
        "random_pnl_mean": random_sample["pnl_bps"].mean() if len(random_sample) > 0 else 0.0,
        "v10_pnl_median": v10_selected["pnl_bps"].median() if len(v10_selected) > 0 else 0.0,
        "random_pnl_median": random_sample["pnl_bps"].median() if len(random_sample) > 0 else 0.0,
        "difference_mean": (v10_selected["pnl_bps"].mean() - random_sample["pnl_bps"].mean()) if len(v10_selected) > 0 and len(random_sample) > 0 else 0.0,
        "difference_median": (v10_selected["pnl_bps"].median() - random_sample["pnl_bps"].median()) if len(v10_selected) > 0 and len(random_sample) > 0 else 0.0,
    }
    
    return pd.DataFrame([stats])


def generate_report(
    quantile_stats: pd.DataFrame,
    regime_stats: pd.DataFrame,
    matched_trades: pd.DataFrame,
    output_path: Path,
    variant: str = "v10",
    output_json: Optional[Path] = None,
) -> None:
    """Generate markdown report."""
    # Determine title based on variant
    if variant == "v10_1":
        title = "# ENTRY_V10.1 Label Quality Analysis (seq_len=90)"
        model_desc = "ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)"
    else:
        title = "# ENTRY_V10 Label Quality Analysis"
        model_desc = "ENTRY_V10 HYBRID"
    
    lines = [
        title,
        "",
        "**Date:** " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        f"**Model:** {model_desc}",
        "",
        "## Summary",
        "",
        f"- **Total Matched Trades:** {len(matched_trades):,}",
        f"- **V10 p_long Range:** [{matched_trades['p_long_v10'].min():.3f}, {matched_trades['p_long_v10'].max():.3f}]",
        f"- **Mean PnL (all trades):** {matched_trades['pnl_bps'].mean():.2f} bps",
        f"- **Median PnL (all trades):** {matched_trades['pnl_bps'].median():.2f} bps",
        f"- **Win Rate (all trades):** {(matched_trades['pnl_bps'] > 0).mean()*100:.1f}%",
        "",
        "## Quantile Analysis",
        "",
        "Trade PnL by V10 p_long quantiles:",
        "",
        "| Quantile | N Trades | p_long Range | Mean PnL | Median PnL | Win Rate | P05 | P95 |",
        "|----------|----------|--------------|----------|-----------|----------|-----|-----|",
    ]
    
    for _, row in quantile_stats.iterrows():
        lines.append(
            f"| {row['quantile']} | {row['n_trades']:,} | "
            f"[{row['p_long_min']:.3f}, {row['p_long_max']:.3f}] | "
            f"{row['pnl_mean']:.2f} | {row['pnl_median']:.2f} | "
            f"{row['win_rate']*100:.1f}% | {row['pnl_p05']:.2f} | {row['pnl_p95']:.2f} |"
        )
    
    lines.extend([
        "",
        "### Interpretation",
        "",
        "If higher p_long quantiles show better trade PnL, V10 is learning useful patterns.",
        "If quantiles are similar, labels may not capture real trade outcomes well.",
        "",
    ])
    
    # Regime analysis
    if len(regime_stats) > 0:
        row = regime_stats.iloc[0]
        lines.extend([
            "## Weakest Regime Analysis (NEUTRAL×MEDIUM/HIGH)",
            "",
            f"**Threshold (auto):** {row['threshold']:.3f}",
            "",
            "| Metric | V10 Selected | Random | Difference |",
            "|--------|--------------|--------|------------|",
        ])
        
        lines.append(
            f"| Mean PnL | {row['v10_pnl_mean']:.2f} bps | {row['random_pnl_mean']:.2f} bps | "
            f"{row['difference_mean']:.2f} bps |"
        )
        lines.append(
            f"| Median PnL | {row['v10_pnl_median']:.2f} bps | {row['random_pnl_median']:.2f} bps | "
            f"{row['difference_median']:.2f} bps |"
        )
        lines.append(
            f"| N Trades | {row['n_v10_selected']:,} | {row['n_random']:,} | - |"
        )
        
        lines.extend([
            "",
            "### Interpretation",
            "",
            "If V10-selected trades have similar PnL to random, there may be no edge in this regime.",
            f"Difference: {row['difference_mean']:.2f} bps mean, {row['difference_median']:.2f} bps median.",
            "",
        ])
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n[REPORT] Report saved to: {output_path}")
    
    # Write JSON output for threshold engine
    if output_json:
        import json as json_module
        json_output = {
            "variant": variant,
            "n_total_trades": len(matched_trades),
            "quantile_stats": quantile_stats.to_dict(orient="records"),
            "regime_stats": regime_stats.to_dict(orient="records") if len(regime_stats) > 0 else [],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json_module.dump(json_output, f, indent=2)
        print(f"[REPORT] JSON output saved to: {output_json}")


def find_latest_replay_log() -> Optional[Path]:
    """Auto-detect latest replay trade log."""
    search_dirs = [
        project_root / "runs/replay_shadow/SNIPER_P4_1_V10_HYBRID",
        project_root / "runs/replay_shadow/SNIPER_P4_1",
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        # Look for merged trade log first
        merged_log = search_dir / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"
        if merged_log.exists():
            return merged_log
        
        # Look for any trade_log CSV
        trade_logs = list(search_dir.glob("trade_log*.csv"))
        if trade_logs:
            # Return most recent
            return max(trade_logs, key=lambda p: p.stat().st_mtime)
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Analyze ENTRY_V10 label quality")
    parser.add_argument("--v10-dataset", type=Path, required=True, help="V10 dataset parquet")
    parser.add_argument("--v10-model", type=Path, required=True, help="V10 model path")
    parser.add_argument("--v10-meta", type=Path, required=True, help="V10 metadata path")
    parser.add_argument("--replay-trade-log", type=Path, default=None, help="Replay trade log CSV or trade journal directory (auto-detect if not provided)")
    parser.add_argument("--trade-journal", type=Path, default=None, help="Trade journal directory (alternative to --replay-trade-log)")
    parser.add_argument("--output-report", type=Path, required=True, help="Output report path")
    parser.add_argument("--n-quantiles", type=int, default=5, help="Number of quantiles")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    
    args = parser.parse_args()
    
    # Handle trade journal parameter (takes precedence)
    if args.trade_journal:
        args.replay_trade_log = args.trade_journal
    
    # Auto-detect replay log if not provided
    if args.replay_trade_log is None or not args.replay_trade_log.exists():
        print("[AUTO] Replay log not provided or not found, auto-detecting...")
        args.replay_trade_log = find_latest_replay_log()
        if args.replay_trade_log is None:
            print("❌ ERROR: Could not find replay trade log. Please provide --replay-trade-log or --trade-journal")
            return 1
        print(f"[AUTO] Found replay log: {args.replay_trade_log}")
    
    # Validate all required files exist
    if not args.v10_dataset.exists():
        print(f"❌ ERROR: V10 dataset not found: {args.v10_dataset}")
        return 1
    if not args.v10_model.exists():
        print(f"❌ ERROR: V10 model not found: {args.v10_model}")
        return 1
    if not args.v10_meta.exists():
        print(f"❌ ERROR: V10 metadata not found: {args.v10_meta}")
        return 1
    if not args.replay_trade_log.exists():
        print(f"❌ ERROR: Replay trade log/journal not found: {args.replay_trade_log}")
        return 1
    
    device = torch.device(args.device)
    
    # Load V10 model
    print("[LOAD] Loading V10 model...")
    v10_model = load_v10_model(args.v10_model, args.v10_meta, device)
    
    # Load dataset (get seq_len from metadata if available)
    with open(args.v10_meta, "r") as f:
        v10_meta = json.load(f)
    seq_len = v10_meta.get("seq_len", 30)
    print(f"[LOAD] Loading V10 dataset (seq_len={seq_len})...")
    dataset = EntryV10Dataset(args.v10_dataset, seq_len=seq_len, device=device)
    
    # Get predictions
    print("[PRED] Getting V10 predictions...")
    preds, targets, session_ids, vol_regime_ids, trend_regime_ids, timestamps = get_v10_predictions(
        v10_model, dataset, device
    )
    
    # Load trade log
    trade_log = load_replay_trade_log(args.replay_trade_log)
    
    # Match predictions to trades
    matched = match_predictions_to_trades(preds, timestamps, trade_log)
    
    if len(matched) == 0:
        print("❌ No matched trades found!")
        return 1
    
    # Get variant from metadata
    with open(args.v10_meta, "r") as f:
        v10_meta_data = json.load(f)
    variant = v10_meta_data.get("variant", "v10")
    
    # Analyze quantiles
    quantile_stats, quantile_metadata = analyze_quantiles(
        matched, n_quantiles=args.n_quantiles, min_trades_per_quantile=30
    )
    
    # Check for insufficient data quantiles
    if quantile_metadata.get("insufficient_data_quantiles"):
        print(f"\n⚠️  WARNING: {len(quantile_metadata['insufficient_data_quantiles'])} quantiles have insufficient data")
        for q_info in quantile_metadata["insufficient_data_quantiles"]:
            print(f"   - {q_info['quantile']}: {q_info['n_trades']} trades (< 30 required)")
    
    # Analyze weakest regimes
    regime_stats = analyze_regime_weakest(matched)
    
    # Generate report
    output_json = args.output_report.parent / f"{args.output_report.stem}.json"
    generate_report(quantile_stats, regime_stats, matched, args.output_report, variant=variant, output_json=output_json)
    
    print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

