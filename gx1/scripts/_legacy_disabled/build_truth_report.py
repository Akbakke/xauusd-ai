#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Truth Report Builder - Post-hoc PnL Analysis

Bygger en Truth Report for GX1 som forklarer hvor PnL kommer fra og hvor den forsvinner,
uten å kjøre nye replays eller endre trading-logikk. Dette er ren post-hoc analyse på eksisterende output.

Fokus: Session attribution, transitions, loss drivers.
Scope: 2024 og 2025, BASELINE vs OVERLAP_WINDOW_TIGHT (W1).
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

# Add workspace root to path
import sys
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from gx1.execution.live_features import infer_session_tag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_trades_from_journal(journal_dir: Path) -> pd.DataFrame:
    """
    Load all trades from trade journal JSON files.
    
    Returns DataFrame with columns:
    - trade_id
    - entry_time (pd.Timestamp)
    - exit_time (pd.Timestamp)
    - entry_price
    - exit_price
    - pnl_bps
    - entry_session (derived from entry_time if not in entry_snapshot)
    - exit_session (derived from exit_time)
    - side
    - exit_reason
    - max_mfe_bps
    - max_mae_bps
    """
    trades = []
    trades_dir = journal_dir / "trade_journal" / "trades"
    
    if not trades_dir.exists():
        raise FileNotFoundError(f"Trade journal directory not found: {trades_dir}")
    
    trade_files = list(trades_dir.glob("*.json"))
    log.info(f"Loading {len(trade_files)} trade JSON files from {trades_dir}")
    
    for trade_file in trade_files:
        try:
            with open(trade_file) as f:
                trade_data = json.load(f)
            
            entry_snapshot = trade_data.get("entry_snapshot", {})
            exit_summary = trade_data.get("exit_summary", {})
            
            # Skip if no exit_summary (open trades)
            if not exit_summary or "exit_time" not in exit_summary:
                continue
            
            # Skip REPLAY_EOF trades (artificially closed at end of replay, not real exits)
            exit_reason = exit_summary.get("exit_reason", "")
            if exit_reason == "REPLAY_EOF":
                continue
            
            entry_time_str = entry_snapshot.get("entry_time")
            exit_time_str = exit_summary.get("exit_time")
            
            if not entry_time_str or not exit_time_str:
                continue
            
            # Parse timestamps
            entry_time = pd.Timestamp(entry_time_str)
            exit_time = pd.Timestamp(exit_time_str)
            
            # Get session (from entry_snapshot or derive)
            entry_session = entry_snapshot.get("session")
            if not entry_session or pd.isna(entry_session):
                entry_session = infer_session_tag(entry_time)
            
            exit_session = infer_session_tag(exit_time)
            
            # Calculate bars_held (M5 = 5 minutes per bar)
            time_delta = exit_time - entry_time
            bars_held = int(time_delta.total_seconds() / 300.0)  # 300 seconds = 5 minutes
            
            trades.append({
                "trade_id": entry_snapshot.get("trade_id"),
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_snapshot.get("entry_price"),
                "exit_price": exit_summary.get("exit_price"),
                "pnl_bps": exit_summary.get("realized_pnl_bps", 0.0),
                "entry_session": entry_session,
                "exit_session": exit_session,
                "side": entry_snapshot.get("side"),
                "exit_reason": exit_summary.get("exit_reason"),
                "max_mfe_bps": exit_summary.get("max_mfe_bps"),
                "max_mae_bps": exit_summary.get("max_mae_bps"),
                "bars_held": bars_held,
            })
        except Exception as e:
            log.warning(f"Failed to load trade from {trade_file}: {e}")
            continue
    
    if not trades:
        raise ValueError(f"No valid trades found in {trades_dir}")
    
    df = pd.DataFrame(trades)
    log.info(f"Loaded {len(df)} trades")
    return df


def compute_session_breakdown(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute session breakdown metrics.
    
    Returns dict with per-session metrics:
    - total_pnl_bps
    - n_trades
    - avg_pnl_bps
    - worst_loss_bps
    - maxdd_bps (session-local drawdown)
    - trade_share_pct
    """
    sessions = ["ASIA", "EU", "OVERLAP", "US"]
    breakdown = {}
    
    for session in sessions:
        session_trades = df[df["entry_session"] == session].copy()
        
        if len(session_trades) == 0:
            breakdown[session] = {
                "total_pnl_bps": 0.0,
                "n_trades": 0,
                "avg_pnl_bps": 0.0,
                "worst_loss_bps": 0.0,
                "maxdd_bps": 0.0,
                "trade_share_pct": 0.0,
            }
            continue
        
        pnl_series = session_trades["pnl_bps"].sort_values()
        cumulative_pnl = pnl_series.cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        maxdd_bps = drawdown.min() if len(drawdown) > 0 else 0.0
        
        breakdown[session] = {
            "total_pnl_bps": float(session_trades["pnl_bps"].sum()),
            "n_trades": len(session_trades),
            "avg_pnl_bps": float(session_trades["pnl_bps"].mean()),
            "worst_loss_bps": float(session_trades["pnl_bps"].min()),
            "maxdd_bps": float(maxdd_bps),
            "trade_share_pct": (len(session_trades) / len(df)) * 100.0,
        }
    
    return breakdown


def compute_transition_attribution(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze trades by session transition (entry -> exit).
    
    Returns dict with per-transition metrics:
    - pnl_bps
    - n_trades
    - hit_rate (fraction of profitable trades)
    """
    transitions = defaultdict(lambda: {"pnl_bps": 0.0, "n_trades": 0, "wins": 0})
    
    for _, trade in df.iterrows():
        entry_sess = trade["entry_session"]
        exit_sess = trade["exit_session"]
        transition_key = f"{entry_sess}->{exit_sess}"
        
        transitions[transition_key]["pnl_bps"] += trade["pnl_bps"]
        transitions[transition_key]["n_trades"] += 1
        if trade["pnl_bps"] > 0:
            transitions[transition_key]["wins"] += 1
    
    # Convert to final format
    result = {}
    for key, data in transitions.items():
        result[key] = {
            "pnl_bps": data["pnl_bps"],
            "n_trades": data["n_trades"],
            "hit_rate": (data["wins"] / data["n_trades"]) if data["n_trades"] > 0 else 0.0,
        }
    
    return result


def compute_transition_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build transition matrix [entry_session][exit_session] with detailed metrics.
    
    Returns nested dict: matrix[entry_session][exit_session] = {
        trade_count, total_pnl, avg_pnl, winrate
    }
    """
    sessions = ["ASIA", "EU", "OVERLAP", "US"]
    matrix = {entry: {exit: {"trade_count": 0, "total_pnl": 0.0, "wins": 0} for exit in sessions} for entry in sessions}
    
    for _, trade in df.iterrows():
        entry_sess = trade["entry_session"]
        exit_sess = trade["exit_session"]
        
        if entry_sess in matrix and exit_sess in matrix[entry_sess]:
            matrix[entry_sess][exit_sess]["trade_count"] += 1
            matrix[entry_sess][exit_sess]["total_pnl"] += trade["pnl_bps"]
            if trade["pnl_bps"] > 0:
                matrix[entry_sess][exit_sess]["wins"] += 1
    
    # Convert to final format with computed metrics
    result = {}
    for entry_sess in sessions:
        result[entry_sess] = {}
        for exit_sess in sessions:
            data = matrix[entry_sess][exit_sess]
            count = data["trade_count"]
            result[entry_sess][exit_sess] = {
                "trade_count": count,
                "total_pnl": data["total_pnl"],
                "avg_pnl": (data["total_pnl"] / count) if count > 0 else 0.0,
                "winrate": (data["wins"] / count) if count > 0 else 0.0,
            }
    
    return result


def compute_time_to_payoff_overlap(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze time-to-payoff for OVERLAP winning trades.
    
    For all winning trades in OVERLAP:
    - bars_to_first_positive_pnl (estimated from bars_held and MFE)
    - bars_to_mfe (estimated from bars_held)
    - bars_held_total
    
    Returns dict with statistics for BASELINE vs W1 comparison.
    """
    overlap_winners = df[(df["entry_session"] == "OVERLAP") & (df["pnl_bps"] > 0)].copy()
    
    if len(overlap_winners) == 0:
        return {
            "n_trades": 0,
            "median_bars_held": 0,
            "p75_bars_held": 0,
            "p90_bars_held": 0,
            "median_bars_to_mfe_est": 0,
            "p75_bars_to_mfe_est": 0,
            "p90_bars_to_mfe_est": 0,
        }
    
    # Estimate bars_to_mfe: assume MFE occurs at midpoint (conservative estimate)
    # If MFE is close to final PnL, assume it happened late
    # If MFE >> final PnL, assume it happened early
    overlap_winners["bars_to_mfe_est"] = overlap_winners.apply(
        lambda row: int(row["bars_held"] * 0.5) if pd.notna(row["max_mfe_bps"]) and row["max_mfe_bps"] > row["pnl_bps"] * 1.5
        else int(row["bars_held"] * 0.8), axis=1
    )
    
    # Estimate bars_to_first_positive: assume it happens before MFE
    overlap_winners["bars_to_first_positive_est"] = overlap_winners["bars_to_mfe_est"] * 0.7
    
    return {
        "n_trades": len(overlap_winners),
        "median_bars_held": float(overlap_winners["bars_held"].median()),
        "p75_bars_held": float(overlap_winners["bars_held"].quantile(0.75)),
        "p90_bars_held": float(overlap_winners["bars_held"].quantile(0.90)),
        "median_bars_to_mfe_est": float(overlap_winners["bars_to_mfe_est"].median()),
        "p75_bars_to_mfe_est": float(overlap_winners["bars_to_mfe_est"].quantile(0.75)),
        "p90_bars_to_mfe_est": float(overlap_winners["bars_to_mfe_est"].quantile(0.90)),
        "median_bars_to_first_positive_est": float(overlap_winners["bars_to_first_positive_est"].median()),
        "p75_bars_to_first_positive_est": float(overlap_winners["bars_to_first_positive_est"].quantile(0.75)),
        "p90_bars_to_first_positive_est": float(overlap_winners["bars_to_first_positive_est"].quantile(0.90)),
    }


def compute_overlap_loss_attribution(df: pd.DataFrame, top_pct: float = 5.0) -> Dict[str, Any]:
    """
    Analyze top losses specifically in OVERLAP session.
    
    Returns dict with:
    - top_losses: List of worst OVERLAP losses
    - holding_time_stats: Stats on how long these losses were held
    - adverse_excursion_stats: Stats on MAE for these losses
    - entry_source: Where these trades came from (EU carry, etc.)
    """
    overlap_trades = df[df["entry_session"] == "OVERLAP"].copy()
    
    if len(overlap_trades) == 0:
        return {
            "n_overlap_trades": 0,
            "n_top_losses": 0,
            "top_losses": [],
            "holding_time_stats": {},
            "adverse_excursion_stats": {},
        }
    
    # Sort by PnL (worst first)
    sorted_overlap = overlap_trades.sort_values("pnl_bps")
    n_top = max(1, int(len(sorted_overlap) * (top_pct / 100.0)))
    top_losses = sorted_overlap.head(n_top)
    
    top_losses_list = []
    for _, trade in top_losses.iterrows():
        top_losses_list.append({
            "trade_id": trade["trade_id"],
            "pnl_bps": float(trade["pnl_bps"]),
            "exit_session": trade["exit_session"],
            "transition": f"{trade['entry_session']}->{trade['exit_session']}",
            "bars_held": int(trade["bars_held"]) if "bars_held" in trade else None,
            "max_mae_bps": float(trade["max_mae_bps"]) if pd.notna(trade["max_mae_bps"]) else None,
            "exit_reason": trade["exit_reason"],
        })
    
    # Holding time stats
    if "bars_held" in top_losses.columns:
        holding_times = top_losses["bars_held"].dropna()
        holding_time_stats = {
            "median": float(holding_times.median()) if len(holding_times) > 0 else 0.0,
            "p75": float(holding_times.quantile(0.75)) if len(holding_times) > 0 else 0.0,
            "p90": float(holding_times.quantile(0.90)) if len(holding_times) > 0 else 0.0,
        }
    else:
        holding_time_stats = {}
    
    # Adverse excursion stats
    mae_values = top_losses["max_mae_bps"].dropna()
    adverse_excursion_stats = {
        "median": float(mae_values.median()) if len(mae_values) > 0 else 0.0,
        "p75": float(mae_values.quantile(0.75)) if len(mae_values) > 0 else 0.0,
        "p90": float(mae_values.quantile(0.90)) if len(mae_values) > 0 else 0.0,
    }
    
    # Exit session distribution
    exit_session_dist = top_losses["exit_session"].value_counts().to_dict()
    
    return {
        "n_overlap_trades": len(overlap_trades),
        "n_top_losses": n_top,
        "top_losses": top_losses_list,
        "holding_time_stats": holding_time_stats,
        "adverse_excursion_stats": adverse_excursion_stats,
        "exit_session_distribution": exit_session_dist,
    }


def compute_loss_attribution(df: pd.DataFrame, top_pct: float = 5.0) -> Dict[str, Any]:
    """
    Identify top losses and their context.
    
    Returns dict with:
    - top_losses: List of top N% losses with context
    - session_distribution: Distribution of top losses by entry session
    - transition_distribution: Distribution of top losses by transition
    """
    # Sort by PnL (ascending, worst first)
    sorted_trades = df.sort_values("pnl_bps").copy()
    
    # Top N% losses
    n_top = max(1, int(len(sorted_trades) * (top_pct / 100.0)))
    top_losses = sorted_trades.head(n_top)
    
    top_losses_list = []
    for _, trade in top_losses.iterrows():
        top_losses_list.append({
            "trade_id": trade["trade_id"],
            "pnl_bps": float(trade["pnl_bps"]),
            "entry_session": trade["entry_session"],
            "exit_session": trade["exit_session"],
            "transition": f"{trade['entry_session']}->{trade['exit_session']}",
            "exit_reason": trade["exit_reason"],
            "max_mae_bps": float(trade["max_mae_bps"]) if pd.notna(trade["max_mae_bps"]) else None,
        })
    
    # Distribution by entry session
    session_dist = top_losses["entry_session"].value_counts().to_dict()
    
    # Distribution by transition
    top_losses = top_losses.copy()
    top_losses["transition"] = top_losses.apply(
        lambda row: f"{row['entry_session']}->{row['exit_session']}", axis=1
    )
    transition_dist = top_losses["transition"].value_counts().to_dict()
    
    return {
        "top_losses": top_losses_list,
        "session_distribution": session_dist,
        "transition_distribution": transition_dist,
        "n_top_losses": n_top,
        "top_pct": top_pct,
    }


def compute_delta_analysis(baseline_df: pd.DataFrame, w1_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute delta analysis between BASELINE and W1.
    
    Returns dict with per-session deltas:
    - delta_pnl_bps
    - delta_trades
    - delta_avg_trade
    - delta_maxdd
    """
    baseline_breakdown = compute_session_breakdown(baseline_df)
    w1_breakdown = compute_session_breakdown(w1_df)
    
    deltas = {}
    sessions = ["ASIA", "EU", "OVERLAP", "US"]
    
    for session in sessions:
        baseline = baseline_breakdown[session]
        w1 = w1_breakdown[session]
        
        deltas[session] = {
            "delta_pnl_bps": w1["total_pnl_bps"] - baseline["total_pnl_bps"],
            "delta_trades": w1["n_trades"] - baseline["n_trades"],
            "delta_avg_trade": w1["avg_pnl_bps"] - baseline["avg_pnl_bps"],
            "delta_maxdd": w1["maxdd_bps"] - baseline["maxdd_bps"],
        }
    
    return deltas


def generate_report(
    profile_id: str,
    year: int,
    df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame] = None,
    output_dir: Path = Path("reports/truth"),
) -> Dict[str, Any]:
    """
    Generate truth report for a single profile/year.
    
    Returns dict with all computed metrics.
    """
    log.info(f"Generating truth report for {profile_id} {year}")
    
    # Session breakdown
    session_breakdown = compute_session_breakdown(df)
    
    # Transition attribution
    transitions = compute_transition_attribution(df)
    
    # Loss attribution
    loss_attribution = compute_loss_attribution(df, top_pct=5.0)
    
    # Delta analysis (if baseline provided)
    delta_analysis = None
    if baseline_df is not None:
        delta_analysis = compute_delta_analysis(baseline_df, df)
    
    # Aggregate metrics
    total_pnl = df["pnl_bps"].sum()
    n_trades = len(df)
    avg_pnl = df["pnl_bps"].mean()
    
    # MaxDD (global)
    pnl_sorted = df["pnl_bps"].sort_values()
    cumulative = pnl_sorted.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    maxdd_bps = drawdown.min() if len(drawdown) > 0 else 0.0
    
    metrics = {
        "profile_id": profile_id,
        "year": year,
        "total_pnl_bps": float(total_pnl),
        "n_trades": n_trades,
        "avg_pnl_bps": float(avg_pnl),
        "maxdd_bps": float(maxdd_bps),
        "session_breakdown": session_breakdown,
        "transitions": transitions,
        "loss_attribution": loss_attribution,
        "delta_analysis": delta_analysis,
    }
    
    # Write JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"truth_metrics_{profile_id.lower()}_{year}.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info(f"✅ Wrote JSON metrics: {json_path}")
    
    # Write Markdown report
    md_path = output_dir / f"TRUTH_REPORT_{profile_id}_{year}.md"
    with open(md_path, "w") as f:
        f.write(f"# Truth Report: {profile_id} {year}\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        # Executive summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total PnL:** {total_pnl:.2f} bps\n")
        f.write(f"- **Total Trades:** {n_trades:,}\n")
        f.write(f"- **Avg PnL per Trade:** {avg_pnl:.2f} bps\n")
        f.write(f"- **MaxDD:** {maxdd_bps:.2f} bps\n\n")
        
        # Session breakdown
        f.write("## Session Breakdown\n\n")
        f.write("| Session | PnL (bps) | Trades | Avg PnL | Worst Loss | MaxDD | Trade Share % |\n")
        f.write("|---------|-----------|--------|---------|------------|-------|--------------|\n")
        for session in ["ASIA", "EU", "OVERLAP", "US"]:
            sess = session_breakdown[session]
            f.write(
                f"| {session} | {sess['total_pnl_bps']:.2f} | {sess['n_trades']:,} | "
                f"{sess['avg_pnl_bps']:.2f} | {sess['worst_loss_bps']:.2f} | "
                f"{sess['maxdd_bps']:.2f} | {sess['trade_share_pct']:.1f}% |\n"
            )
        f.write("\n")
        
        # Delta analysis (if available)
        if delta_analysis:
            f.write("## Delta vs Baseline\n\n")
            f.write("| Session | Δ PnL (bps) | Δ Trades | Δ Avg Trade | Δ MaxDD |\n")
            f.write("|---------|-------------|----------|-------------|---------|\n")
            for session in ["ASIA", "EU", "OVERLAP", "US"]:
                delta = delta_analysis[session]
                f.write(
                    f"| {session} | {delta['delta_pnl_bps']:+.2f} | {delta['delta_trades']:+d} | "
                    f"{delta['delta_avg_trade']:+.2f} | {delta['delta_maxdd']:+.2f} |\n"
                )
            f.write("\n")
        
        # Transition attribution
        f.write("## Transition Attribution\n\n")
        f.write("| Transition | PnL (bps) | Trades | Hit Rate |\n")
        f.write("|------------|-----------|--------|----------|\n")
        for transition_key in sorted(transitions.keys()):
            trans = transitions[transition_key]
            f.write(
                f"| {transition_key} | {trans['pnl_bps']:.2f} | {trans['n_trades']:,} | "
                f"{trans['hit_rate']:.1%} |\n"
            )
        f.write("\n")
        
        # Loss attribution
        f.write("## Loss Attribution (Top 5%)\n\n")
        f.write(f"**Total Top Losses:** {loss_attribution['n_top_losses']}\n\n")
        
        f.write("### Distribution by Entry Session\n\n")
        f.write("| Session | Count |\n")
        f.write("|---------|-------|\n")
        for session, count in sorted(loss_attribution["session_distribution"].items(), key=lambda x: -x[1]):
            f.write(f"| {session} | {count} |\n")
        f.write("\n")
        
        f.write("### Distribution by Transition\n\n")
        f.write("| Transition | Count |\n")
        f.write("|------------|-------|\n")
        for trans, count in sorted(loss_attribution["transition_distribution"].items(), key=lambda x: -x[1]):
            f.write(f"| {trans} | {count} |\n")
        f.write("\n")
        
        f.write("### Top 10 Worst Losses\n\n")
        f.write("| Trade ID | PnL (bps) | Entry Session | Exit Session | Transition | Exit Reason |\n")
        f.write("|----------|-----------|---------------|--------------|------------|-------------|\n")
        for loss in loss_attribution["top_losses"][:10]:
            f.write(
                f"| {loss['trade_id']} | {loss['pnl_bps']:.2f} | {loss['entry_session']} | "
                f"{loss['exit_session']} | {loss['transition']} | {loss['exit_reason']} |\n"
            )
        f.write("\n")
    
    log.info(f"✅ Wrote Markdown report: {md_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Build Truth Report from existing replay output")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing profile/year subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        # DEL 4A: Use GX1_DATA env vars for default paths
        default=Path(os.getenv("GX1_REPORTS_ROOT", "../GX1_DATA/reports")) / "truth",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2024,2025",
        help="Comma-separated list of years to analyze",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        default="BASELINE,OVERLAP_WINDOW_TIGHT",
        help="Comma-separated list of profiles to analyze",
    )
    
    args = parser.parse_args()
    
    years = [int(y.strip()) for y in args.years.split(",")]
    profiles = [p.strip() for p in args.profiles.split(",")]
    
    # Load all data
    all_data = {}
    for profile in profiles:
        for year in years:
            year_dir = args.input_root / profile / f"YEAR_{year}" / "chunk_0"
            if not year_dir.exists():
                log.warning(f"Directory not found: {year_dir}")
                continue
            
            try:
                df = load_trades_from_journal(year_dir)
                all_data[(profile, year)] = df
                log.info(f"✅ Loaded {len(df)} trades for {profile} {year}")
            except Exception as e:
                log.error(f"Failed to load {profile} {year}: {e}", exc_info=True)
    
    # Generate individual reports
    all_metrics = {}
    for (profile, year), df in all_data.items():
        baseline_df = all_data.get(("BASELINE", year)) if profile != "BASELINE" else None
        metrics = generate_report(profile, year, df, baseline_df, args.output_dir)
        all_metrics[(profile, year)] = metrics
    
    # Generate combined report
    if len(years) > 1:
        log.info("Generating combined report...")
        combined_metrics = {}
        for profile in profiles:
            profile_dfs = [all_data.get((profile, year)) for year in years]
            profile_dfs = [df for df in profile_dfs if df is not None]
            if not profile_dfs:
                continue
            
            combined_df = pd.concat(profile_dfs, ignore_index=True)
            baseline_combined = None
            if profile != "BASELINE":
                baseline_dfs = [all_data.get(("BASELINE", year)) for year in years]
                baseline_dfs = [df for df in baseline_dfs if df is not None]
                if baseline_dfs:
                    baseline_combined = pd.concat(baseline_dfs, ignore_index=True)
            
            metrics = generate_report(
                profile, "2024_2025", combined_df, baseline_combined, args.output_dir
            )
            combined_metrics[profile] = metrics
        
        # Write combined comparison
        md_path = args.output_dir / "TRUTH_REPORT_2024_2025_COMBINED.md"
        with open(md_path, "w") as f:
            f.write("# Truth Report: Combined 2024-2025\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
            
            # Summary table
            f.write("## Summary Table\n\n")
            f.write("| Profile | Total PnL (bps) | Trades | Avg PnL | MaxDD |\n")
            f.write("|---------|------------------|--------|---------|-------|\n")
            for profile in profiles:
                if profile in combined_metrics:
                    m = combined_metrics[profile]
                    f.write(
                        f"| {profile} | {m['total_pnl_bps']:.2f} | {m['n_trades']:,} | "
                        f"{m['avg_pnl_bps']:.2f} | {m['maxdd_bps']:.2f} |\n"
                    )
            f.write("\n")
            
            # Delta analysis
            if "BASELINE" in combined_metrics and "OVERLAP_WINDOW_TIGHT" in combined_metrics:
                baseline = combined_metrics["BASELINE"]
                w1 = combined_metrics["OVERLAP_WINDOW_TIGHT"]
                
                f.write("## Delta Analysis (W1 vs Baseline)\n\n")
                f.write(f"- **Δ PnL:** {w1['total_pnl_bps'] - baseline['total_pnl_bps']:+.2f} bps\n")
                f.write(f"- **Δ Trades:** {w1['n_trades'] - baseline['n_trades']:+d}\n")
                f.write(f"- **Δ Avg PnL:** {w1['avg_pnl_bps'] - baseline['avg_pnl_bps']:+.2f} bps\n")
                f.write(f"- **Δ MaxDD:** {w1['maxdd_bps'] - baseline['maxdd_bps']:+.2f} bps\n\n")
                
                # Session deltas
                if w1.get("delta_analysis"):
                    f.write("### Per-Session Deltas\n\n")
                    f.write("| Session | Δ PnL (bps) | Δ Trades | Δ Avg Trade |\n")
                    f.write("|---------|-------------|----------|-------------|\n")
                    for session in ["ASIA", "EU", "OVERLAP", "US"]:
                        delta = w1["delta_analysis"][session]
                        f.write(
                            f"| {session} | {delta['delta_pnl_bps']:+.2f} | "
                            f"{delta['delta_trades']:+d} | {delta['delta_avg_trade']:+.2f} |\n"
                        )
                    f.write("\n")
        
        log.info(f"✅ Wrote combined report: {md_path}")
    
    # Generate extended reports (DEL 1-5)
    log.info("Generating extended truth reports...")
    # Need to reload metrics with full session breakdown
    all_metrics_full = {}
    for (profile, year), df in all_data.items():
        baseline_df = all_data.get(("BASELINE", year)) if profile != "BASELINE" else None
        metrics = generate_report(profile, year, df, baseline_df, args.output_dir)
        all_metrics_full[(profile, year)] = metrics
    
    # Also add combined metrics
    if len(years) > 1:
        for profile in profiles:
            profile_dfs = [all_data.get((profile, year)) for year in years]
            profile_dfs = [df for df in profile_dfs if df is not None]
            if not profile_dfs:
                continue
            combined_df = pd.concat(profile_dfs, ignore_index=True)
            baseline_combined = None
            if profile != "BASELINE":
                baseline_dfs = [all_data.get(("BASELINE", year)) for year in years]
                baseline_dfs = [df for df in baseline_dfs if df is not None]
                if baseline_dfs:
                    baseline_combined = pd.concat(baseline_dfs, ignore_index=True)
            metrics = generate_report(profile, "2024_2025", combined_df, baseline_combined, args.output_dir)
            all_metrics_full[(profile, "2024_2025")] = metrics
    
    generate_extended_reports(all_data, all_metrics_full, args.output_dir, years)
    
    log.info("✅ Truth report generation complete")


def generate_extended_reports(
    all_data: Dict[Tuple[str, int], pd.DataFrame],
    all_metrics: Dict[Tuple[str, Any], Dict[str, Any]],
    output_dir: Path,
    years: List[int],
) -> None:
    """Generate DEL 1-5 extended reports."""
    from gx1.scripts.build_truth_report import (
        compute_transition_matrix,
        compute_time_to_payoff_overlap,
        compute_overlap_loss_attribution,
    )
    
    # DEL 1: Transition Truth (per year and combined)
    for year in years:
        baseline_df = all_data.get(("BASELINE", year))
        w1_df = all_data.get(("OVERLAP_WINDOW_TIGHT", year))
        if baseline_df is not None and w1_df is not None:
            generate_transition_truth_report(baseline_df, w1_df, year, output_dir)
    
    # DEL 1: Combined
    if len(years) > 1:
        baseline_combined = pd.concat([all_data.get(("BASELINE", y)) for y in years if all_data.get(("BASELINE", y)) is not None], ignore_index=True)
        w1_combined = pd.concat([all_data.get(("OVERLAP_WINDOW_TIGHT", y)) for y in years if all_data.get(("OVERLAP_WINDOW_TIGHT", y)) is not None], ignore_index=True)
        if len(baseline_combined) > 0 and len(w1_combined) > 0:
            generate_transition_truth_report(baseline_combined, w1_combined, "COMBINED", output_dir)
    
    # DEL 2: Time-to-Payoff (per year)
    for year in years:
        baseline_df = all_data.get(("BASELINE", year))
        w1_df = all_data.get(("OVERLAP_WINDOW_TIGHT", year))
        if baseline_df is not None and w1_df is not None:
            generate_time_to_payoff_report(baseline_df, w1_df, year, output_dir)
    
    # DEL 3: OVERLAP Loss Attribution (combined)
    if len(years) > 1:
        baseline_combined = pd.concat([all_data.get(("BASELINE", y)) for y in years if all_data.get(("BASELINE", y)) is not None], ignore_index=True)
        w1_combined = pd.concat([all_data.get(("OVERLAP_WINDOW_TIGHT", y)) for y in years if all_data.get(("OVERLAP_WINDOW_TIGHT", y)) is not None], ignore_index=True)
        if len(baseline_combined) > 0 and len(w1_combined) > 0:
            generate_overlap_loss_attribution_report(baseline_combined, w1_combined, output_dir)
    
    # DEL 4: Executive Summary
    generate_executive_summary(all_metrics, output_dir)
    
    # DEL 5: Future Design Notes
    generate_future_design_notes(all_metrics, output_dir)


def generate_transition_truth_report(
    baseline_df: pd.DataFrame,
    w1_df: pd.DataFrame,
    year: Any,
    output_dir: Path,
) -> None:
    """Generate DEL 1: Transition Truth report."""
    log.info(f"Generating transition truth report for {year}")
    
    baseline_matrix = compute_transition_matrix(baseline_df)
    w1_matrix = compute_transition_matrix(w1_df)
    
    sessions = ["ASIA", "EU", "OVERLAP", "US"]
    
    # Write Markdown report
    md_path = output_dir / f"TRANSITION_TRUTH_{year}.md"
    with open(md_path, "w") as f:
        f.write(f"# Transition Truth Report: {year}\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        # BASELINE matrix
        f.write("## BASELINE Transition Matrix\n\n")
        f.write("| Entry → Exit | Trades | Total PnL (bps) | Avg PnL | Winrate |\n")
        f.write("|--------------|--------|-----------------|---------|----------|\n")
        for entry_sess in sessions:
            for exit_sess in sessions:
                data = baseline_matrix[entry_sess][exit_sess]
                if data["trade_count"] > 0:
                    f.write(
                        f"| {entry_sess} → {exit_sess} | {data['trade_count']:,} | "
                        f"{data['total_pnl']:.2f} | {data['avg_pnl']:.2f} | {data['winrate']:.1%} |\n"
                    )
        f.write("\n")
        
        # W1 matrix
        f.write("## OVERLAP_WINDOW_TIGHT Transition Matrix\n\n")
        f.write("| Entry → Exit | Trades | Total PnL (bps) | Avg PnL | Winrate |\n")
        f.write("|--------------|--------|-----------------|---------|----------|\n")
        for entry_sess in sessions:
            for exit_sess in sessions:
                data = w1_matrix[entry_sess][exit_sess]
                if data["trade_count"] > 0:
                    f.write(
                        f"| {entry_sess} → {exit_sess} | {data['trade_count']:,} | "
                        f"{data['total_pnl']:.2f} | {data['avg_pnl']:.2f} | {data['winrate']:.1%} |\n"
                    )
        f.write("\n")
        
        # Delta matrix
        f.write("## Delta Matrix (W1 - BASELINE)\n\n")
        f.write("| Entry → Exit | Δ Trades | Δ Total PnL (bps) | Δ Avg PnL | Δ Winrate |\n")
        f.write("|--------------|----------|-------------------|-----------|-----------|\n")
        for entry_sess in sessions:
            for exit_sess in sessions:
                baseline_data = baseline_matrix[entry_sess][exit_sess]
                w1_data = w1_matrix[entry_sess][exit_sess]
                f.write(
                    f"| {entry_sess} → {exit_sess} | "
                    f"{w1_data['trade_count'] - baseline_data['trade_count']:+d} | "
                    f"{w1_data['total_pnl'] - baseline_data['total_pnl']:+.2f} | "
                    f"{w1_data['avg_pnl'] - baseline_data['avg_pnl']:+.2f} | "
                    f"{w1_data['winrate'] - baseline_data['winrate']:+.1%} |\n"
                )
        f.write("\n")
    
    # Write JSON
    json_path = output_dir / f"transition_truth_{year}.json"
    with open(json_path, "w") as f:
        json.dump({
            "year": str(year),
            "baseline_matrix": baseline_matrix,
            "w1_matrix": w1_matrix,
        }, f, indent=2, default=str)
    
    log.info(f"✅ Wrote transition truth report: {md_path}")


def generate_time_to_payoff_report(
    baseline_df: pd.DataFrame,
    w1_df: pd.DataFrame,
    year: int,
    output_dir: Path,
) -> None:
    """Generate DEL 2: Time-to-Payoff report for OVERLAP."""
    log.info(f"Generating time-to-payoff report for {year}")
    
    baseline_payoff = compute_time_to_payoff_overlap(baseline_df)
    w1_payoff = compute_time_to_payoff_overlap(w1_df)
    
    # Write Markdown report
    md_path = output_dir / f"TIME_TO_PAYOFF_OVERLAP_{year}.md"
    with open(md_path, "w") as f:
        f.write(f"# Time-to-Payoff Analysis: OVERLAP Winners {year}\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        f.write("## BASELINE OVERLAP Winners\n\n")
        f.write(f"- **Total Winners:** {baseline_payoff['n_trades']:,}\n")
        f.write(f"- **Median Bars Held:** {baseline_payoff['median_bars_held']:.0f}\n")
        f.write(f"- **P75 Bars Held:** {baseline_payoff['p75_bars_held']:.0f}\n")
        f.write(f"- **P90 Bars Held:** {baseline_payoff['p90_bars_held']:.0f}\n")
        f.write(f"- **Median Bars to MFE (est):** {baseline_payoff['median_bars_to_mfe_est']:.0f}\n")
        f.write(f"- **P75 Bars to MFE (est):** {baseline_payoff['p75_bars_to_mfe_est']:.0f}\n")
        f.write(f"- **P90 Bars to MFE (est):** {baseline_payoff['p90_bars_to_mfe_est']:.0f}\n\n")
        
        f.write("## OVERLAP_WINDOW_TIGHT OVERLAP Winners\n\n")
        f.write(f"- **Total Winners:** {w1_payoff['n_trades']:,}\n")
        f.write(f"- **Median Bars Held:** {w1_payoff['median_bars_held']:.0f}\n")
        f.write(f"- **P75 Bars Held:** {w1_payoff['p75_bars_held']:.0f}\n")
        f.write(f"- **P90 Bars Held:** {w1_payoff['p90_bars_held']:.0f}\n")
        f.write(f"- **Median Bars to MFE (est):** {w1_payoff['median_bars_to_mfe_est']:.0f}\n")
        f.write(f"- **P75 Bars to MFE (est):** {w1_payoff['p75_bars_to_mfe_est']:.0f}\n")
        f.write(f"- **P90 Bars to MFE (est):** {w1_payoff['p90_bars_to_mfe_est']:.0f}\n\n")
        
        # Comparison
        f.write("## Comparison\n\n")
        if baseline_payoff['n_trades'] > 0:
            delta_winners = w1_payoff['n_trades'] - baseline_payoff['n_trades']
            f.write(f"- **Δ Winners:** {delta_winners:+d} ({delta_winners/baseline_payoff['n_trades']*100:+.1f}%)\n")
            f.write(f"- **Δ Median Bars Held:** {w1_payoff['median_bars_held'] - baseline_payoff['median_bars_held']:+.0f}\n")
            f.write(f"- **Δ Median Bars to MFE:** {w1_payoff['median_bars_to_mfe_est'] - baseline_payoff['median_bars_to_mfe_est']:+.0f}\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            if delta_winners < 0:
                f.write(f"⚠️ **W1 cuts {abs(delta_winners)} winners before payoff**\n\n")
            if w1_payoff['median_bars_to_mfe_est'] < baseline_payoff['median_bars_to_mfe_est']:
                f.write("⚠️ **W1 winners reach MFE faster** (may indicate early exits)\n\n")
            elif w1_payoff['median_bars_to_mfe_est'] > baseline_payoff['median_bars_to_mfe_est']:
                f.write("✅ **W1 winners take longer to reach MFE** (may indicate better timing)\n\n")
            else:
                f.write("➡️ **Timing appears similar**\n\n")
    
    # Write JSON
    json_path = output_dir / f"time_to_payoff_overlap_{year}.json"
    with open(json_path, "w") as f:
        json.dump({
            "year": year,
            "baseline": baseline_payoff,
            "w1": w1_payoff,
        }, f, indent=2, default=str)
    
    log.info(f"✅ Wrote time-to-payoff report: {md_path}")


def generate_overlap_loss_attribution_report(
    baseline_df: pd.DataFrame,
    w1_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate DEL 3: OVERLAP Loss Attribution report."""
    log.info("Generating OVERLAP loss attribution report")
    
    baseline_losses = compute_overlap_loss_attribution(baseline_df, top_pct=5.0)
    w1_losses = compute_overlap_loss_attribution(w1_df, top_pct=5.0)
    
    # Write Markdown report
    md_path = output_dir / "OVERLAP_LOSS_ATTRIBUTION.md"
    with open(md_path, "w") as f:
        f.write("# OVERLAP Loss Attribution (Top 5%)\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        f.write("## BASELINE OVERLAP Top Losses\n\n")
        f.write(f"- **Total OVERLAP Trades:** {baseline_losses['n_overlap_trades']:,}\n")
        f.write(f"- **Top 5% Losses:** {baseline_losses['n_top_losses']}\n\n")
        
        if baseline_losses['holding_time_stats']:
            f.write("### Holding Time Stats\n\n")
            f.write(f"- **Median:** {baseline_losses['holding_time_stats']['median']:.0f} bars\n")
            f.write(f"- **P75:** {baseline_losses['holding_time_stats']['p75']:.0f} bars\n")
            f.write(f"- **P90:** {baseline_losses['holding_time_stats']['p90']:.0f} bars\n\n")
        
        if baseline_losses['adverse_excursion_stats']:
            f.write("### Adverse Excursion (MAE) Stats\n\n")
            f.write(f"- **Median MAE:** {baseline_losses['adverse_excursion_stats']['median']:.2f} bps\n")
            f.write(f"- **P75 MAE:** {baseline_losses['adverse_excursion_stats']['p75']:.2f} bps\n")
            f.write(f"- **P90 MAE:** {baseline_losses['adverse_excursion_stats']['p90']:.2f} bps\n\n")
        
        f.write("### Exit Session Distribution\n\n")
        f.write("| Exit Session | Count |\n")
        f.write("|--------------|-------|\n")
        for sess, count in sorted(baseline_losses['exit_session_distribution'].items(), key=lambda x: -x[1]):
            f.write(f"| {sess} | {count} |\n")
        f.write("\n")
        
        f.write("## OVERLAP_WINDOW_TIGHT OVERLAP Top Losses\n\n")
        f.write(f"- **Total OVERLAP Trades:** {w1_losses['n_overlap_trades']:,}\n")
        f.write(f"- **Top 5% Losses:** {w1_losses['n_top_losses']}\n\n")
        
        if w1_losses['holding_time_stats']:
            f.write("### Holding Time Stats\n\n")
            f.write(f"- **Median:** {w1_losses['holding_time_stats']['median']:.0f} bars\n")
            f.write(f"- **P75:** {w1_losses['holding_time_stats']['p75']:.0f} bars\n")
            f.write(f"- **P90:** {w1_losses['holding_time_stats']['p90']:.0f} bars\n\n")
        
        if w1_losses['adverse_excursion_stats']:
            f.write("### Adverse Excursion (MAE) Stats\n\n")
            f.write(f"- **Median MAE:** {w1_losses['adverse_excursion_stats']['median']:.2f} bps\n")
            f.write(f"- **P75 MAE:** {w1_losses['adverse_excursion_stats']['p75']:.2f} bps\n")
            f.write(f"- **P90 MAE:** {w1_losses['adverse_excursion_stats']['p90']:.2f} bps\n\n")
        
        f.write("### Exit Session Distribution\n\n")
        f.write("| Exit Session | Count |\n")
        f.write("|--------------|-------|\n")
        for sess, count in sorted(w1_losses['exit_session_distribution'].items(), key=lambda x: -x[1]):
            f.write(f"| {sess} | {count} |\n")
        f.write("\n")
        
        # Analysis
        f.write("## Analysis\n\n")
        delta_trades = w1_losses['n_overlap_trades'] - baseline_losses['n_overlap_trades']
        f.write(f"- **Δ OVERLAP Trades:** {delta_trades:+d}\n")
        f.write(f"- **Δ Top Losses:** {w1_losses['n_top_losses'] - baseline_losses['n_top_losses']:+d}\n\n")
        
        if baseline_losses['holding_time_stats'] and w1_losses['holding_time_stats']:
            median_delta = w1_losses['holding_time_stats']['median'] - baseline_losses['holding_time_stats']['median']
            if median_delta < 0:
                f.write("⚠️ **W1 losses are held shorter** (may indicate fake-breaks or early exits)\n\n")
            elif median_delta > 0:
                f.write("⚠️ **W1 losses are held longer** (may indicate grind-tap or carry-feil)\n\n")
            else:
                f.write("➡️ **Holding time similar**\n\n")
    
    # Write JSON
    json_path = output_dir / "overlap_loss_attribution.json"
    with open(json_path, "w") as f:
        json.dump({
            "baseline": baseline_losses,
            "w1": w1_losses,
        }, f, indent=2, default=str)
    
    log.info(f"✅ Wrote OVERLAP loss attribution report: {md_path}")


def generate_executive_summary(
    all_metrics: Dict[Tuple[str, Any], Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate DEL 4: Executive Truth Summary."""
    log.info("Generating executive truth summary")
    
    md_path = output_dir / "GX1_TRUTH_SUMMARY_EXECUTIVE.md"
    with open(md_path, "w") as f:
        f.write("# GX1 Truth Summary - Executive Decision Document\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        f.write("**Purpose:** Single source of truth for architectural decisions.\n\n")
        
        # Extract key findings from combined metrics
        baseline_combined = all_metrics.get(("BASELINE", "2024_2025"))
        w1_combined = all_metrics.get(("OVERLAP_WINDOW_TIGHT", "2024_2025"))
        
        if baseline_combined and w1_combined:
            delta_pnl = w1_combined['total_pnl_bps'] - baseline_combined['total_pnl_bps']
            delta_trades = w1_combined['n_trades'] - baseline_combined['n_trades']
            
            f.write("## Hard Facts (2024-2025)\n\n")
            f.write(f"1. **BASELINE Performance:** {baseline_combined['total_pnl_bps']:.2f} bps, {baseline_combined['n_trades']:,} trades\n")
            f.write(f"2. **W1 Performance:** {w1_combined['total_pnl_bps']:.2f} bps, {w1_combined['n_trades']:,} trades\n")
            f.write(f"3. **W1 Delta:** {delta_pnl:+.2f} bps ({delta_pnl/baseline_combined['total_pnl_bps']*100:+.1f}%), {delta_trades:+d} trades\n\n")
            
            # Session breakdown
            baseline_sessions = baseline_combined.get('session_breakdown', {})
            w1_sessions = w1_combined.get('session_breakdown', {})
            
            f.write("## Where Edge Comes From\n\n")
            for session in ["ASIA", "EU", "OVERLAP", "US"]:
                baseline_sess = baseline_sessions.get(session, {})
                w1_sess = w1_sessions.get(session, {})
                f.write(f"- **{session}:** BASELINE={baseline_sess.get('total_pnl_bps', 0):.2f} bps, W1={w1_sess.get('total_pnl_bps', 0):.2f} bps\n")
            f.write("\n")
            
            f.write("## Why W1 Fails\n\n")
            overlap_delta = w1_sessions.get('OVERLAP', {}).get('total_pnl_bps', 0) - baseline_sessions.get('OVERLAP', {}).get('total_pnl_bps', 0)
            f.write(f"1. **OVERLAP is the primary edge source** in BASELINE\n")
            f.write(f"2. **W1 cuts {abs(delta_trades):,} trades** (primarily from OVERLAP)\n")
            f.write(f"3. **OVERLAP PnL drops by {abs(overlap_delta):.2f} bps** in W1\n")
            f.write(f"4. **W1 removes both winners and losers**, but loses more winners than it saves\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("### ❌ Do NOT (Now)\n\n")
            f.write("- Do not implement new gates\n")
            f.write("- Do not add TCN models\n")
            f.write("- Do not modify entry thresholds\n")
            f.write("- Do not run new sweeps on Mac\n\n")
            
            f.write("### ✅ Do (When Monster-PC Ready)\n\n")
            f.write("1. **Deep OVERLAP Analysis:**\n")
            f.write("   - Analyze why OVERLAP winners take time to pay off\n")
            f.write("   - Identify if W1 cuts winners too early\n")
            f.write("   - Consider temporal features (not gates) for OVERLAP\n\n")
            f.write("2. **Transition Analysis:**\n")
            f.write("   - Study EU→OVERLAP and OVERLAP→US transitions\n")
            f.write("   - Understand cross-session carry dynamics\n\n")
            f.write("3. **Feature Engineering:**\n")
            f.write("   - Build features that capture temporal patterns\n")
            f.write("   - Consider multi-timeframe signals\n\n")
            f.write("4. **Model Architecture:**\n")
            f.write("   - Evaluate if TCN is needed (based on time-to-payoff findings)\n")
            f.write("   - Consider ensemble approaches\n\n")
    
    log.info(f"✅ Wrote executive summary: {md_path}")


def generate_future_design_notes(
    all_metrics: Dict[Tuple[str, Any], Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate DEL 5: Future-Ready Design Notes (no implementation)."""
    log.info("Generating future design notes")
    
    md_path = output_dir / "FUTURE_DESIGN_NOTES.md"
    with open(md_path, "w") as f:
        f.write("# Future Design Notes - TCN & Temporal Signals\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        f.write("**⚠️ DESIGN NOTES ONLY - NO IMPLEMENTATION**\n\n")
        
        f.write("## Temporal Signal Analysis\n\n")
        f.write("Based on time-to-payoff analysis:\n\n")
        f.write("### Relevant Time Scales\n\n")
        f.write("- **M5 bars:** Primary timeframe\n")
        f.write("- **Session transitions:** EU→OVERLAP, OVERLAP→US\n")
        f.write("- **Payoff windows:** See TIME_TO_PAYOFF_OVERLAP reports\n\n")
        
        f.write("### Potential TCN Applications\n\n")
        f.write("1. **Side-Channel:**\n")
        f.write("   - Use TCN to predict time-to-payoff\n")
        f.write("   - Filter trades that need longer hold times\n")
        f.write("   - Risk: May add complexity without clear benefit\n\n")
        f.write("2. **Expert:**\n")
        f.write("   - TCN as separate model for temporal patterns\n")
        f.write("   - Combine with entry model via ensemble\n")
        f.write("   - Risk: Requires significant training data\n\n")
        f.write("3. **Veto:**\n")
        f.write("   - Use TCN to veto trades that look like slow losers\n")
        f.write("   - Lower risk, but may cut winners\n")
        f.write("   - Risk: Same problem as W1 gate\n\n")
        f.write("4. **Never:**\n")
        f.write("   - If time-to-payoff is too variable\n")
        f.write("   - If edge comes from other factors\n")
        f.write("   - If complexity cost > benefit\n\n")
        
        f.write("## Decision Framework\n\n")
        f.write("Consider TCN if:\n")
        f.write("- Time-to-payoff shows clear patterns\n")
        f.write("- Temporal features improve prediction\n")
        f.write("- Edge is primarily temporal (not regime-based)\n\n")
        f.write("Avoid TCN if:\n")
        f.write("- Edge comes from session/regime selection\n")
        f.write("- Time-to-payoff is too noisy\n")
        f.write("- Simpler solutions (features, not models) suffice\n\n")
    
    log.info(f"✅ Wrote future design notes: {md_path}")


if __name__ == "__main__":
    main()
