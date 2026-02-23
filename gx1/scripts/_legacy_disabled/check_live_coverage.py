#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNIPER Live Coverage Checker

Checks consistency between:
- OANDA (open + closed trades)
- Trade journal (local logging)
- RL dataset (shadow + real trades)

Answers: "Får vi faktisk med HELE bildet?" for a given trading day.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gx1.execution.oanda_client import OandaClient, OandaClientConfig
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.utils.env_loader import load_dotenv_if_present

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_oanda_trades(
    client: OandaClient,
    instrument: str,
    date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load OANDA trades (open + closed).
    
    Returns:
        (open_trades_df, closed_trades_df)
    """
    log.info("Fetching OANDA trades...")
    
    # Fetch open trades
    try:
        open_response = client.get_open_trades()
        open_trades = open_response.get("trades", [])
        log.info(f"Found {len(open_trades)} open trades in OANDA")
    except Exception as e:
        log.warning(f"Failed to fetch open trades: {e}")
        open_trades = []
    
    # Fetch closed trades (last 200)
    try:
        closed_response = client.get_trades(state="CLOSED", instrument=instrument, count=200)
        closed_trades = closed_response.get("trades", [])
        log.info(f"Found {len(closed_trades)} closed trades in OANDA (last 200)")
    except Exception as e:
        log.warning(f"Failed to fetch closed trades: {e}")
        closed_trades = []
    
    # Filter to instrument
    open_trades = [t for t in open_trades if t.get("instrument") == instrument]
    closed_trades = [t for t in closed_trades if t.get("instrument") == instrument]
    
    # Filter to date if provided
    if date:
        date_obj = pd.to_datetime(date).date()
        open_trades = [
            t for t in open_trades
            if pd.to_datetime(t.get("openTime", "")).date() == date_obj
        ]
        closed_trades = [
            t for t in closed_trades
            if pd.to_datetime(t.get("openTime", "")).date() == date_obj
        ]
    
    # Build DataFrames
    open_rows = []
    for t in open_trades:
        open_rows.append({
            "oanda_trade_id": str(t.get("id", "")),
            "state": "OPEN",
            "open_time": pd.to_datetime(t.get("openTime", "")),
            "close_time": None,
            "units": int(float(t.get("currentUnits", 0))),
            "realizedPL": None,
        })
    
    closed_rows = []
    for t in closed_trades:
        closed_rows.append({
            "oanda_trade_id": str(t.get("id", "")),
            "state": "CLOSED",
            "open_time": pd.to_datetime(t.get("openTime", "")),
            "close_time": pd.to_datetime(t.get("closeTime", "")) if t.get("closeTime") else None,
            "units": int(float(t.get("initialUnits", 0))),
            "realizedPL": float(t.get("realizedPL", 0.0)),
        })
    
    open_df = pd.DataFrame(open_rows) if open_rows else pd.DataFrame(columns=["oanda_trade_id", "state", "open_time", "close_time", "units", "realizedPL"])
    closed_df = pd.DataFrame(closed_rows) if closed_rows else pd.DataFrame(columns=["oanda_trade_id", "state", "open_time", "close_time", "units", "realizedPL"])
    
    return open_df, closed_df


def load_trade_journal_index(run_dir: Path, date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load trade journal index CSV."""
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if not index_path.exists():
        log.warning(f"Trade journal index not found: {index_path}")
        return None
    
    try:
        df = pd.read_csv(index_path)
        
        # Parse timestamps
        if "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(df["entry_time"])
        if "exit_time" in df.columns:
            df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
        
        # Filter to date if provided
        if date and "entry_time" in df.columns:
            date_obj = pd.to_datetime(date).date()
            df = df[df["entry_time"].dt.date == date_obj]
        
        # Add is_open column
        df["is_open"] = df["exit_time"].isna() | (df["exit_time"] == "")
        
        # Try to extract oanda_trade_id from extra JSON files
        if "trade_id" in df.columns:
            oanda_ids = []
            for trade_id in df["trade_id"]:
                trade_json_path = run_dir / "trade_journal" / "trades" / f"{trade_id}.json"
                oanda_id = None
                if trade_json_path.exists():
                    try:
                        with open(trade_json_path, "r") as f:
                            trade_data = json.load(f)
                            extra = trade_data.get("extra", {})
                            oanda_id = extra.get("oanda_trade_id")
                    except Exception:
                        pass
                oanda_ids.append(oanda_id)
            df["oanda_trade_id"] = oanda_ids
        
        return df
    except Exception as e:
        log.error(f"Failed to load trade journal index: {e}")
        return None


def load_rl_dataset(dataset_path: Path, date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load RL dataset."""
    if not dataset_path.exists():
        log.warning(f"RL dataset not found: {dataset_path}")
        return None
    
    try:
        df = pd.read_parquet(dataset_path)
        
        # Filter to date if provided
        if date:
            # Try to find timestamp column
            ts_col = None
            for col in ["candle_time", "ts", "time", "timestamp"]:
                if col in df.columns:
                    ts_col = col
                    break
            
            if ts_col:
                df[ts_col] = pd.to_datetime(df[ts_col])
                date_obj = pd.to_datetime(date).date()
                df = df[df[ts_col].dt.date == date_obj]
        
        return df
    except Exception as e:
        log.error(f"Failed to load RL dataset: {e}")
        return None


def match_trades(
    oanda_df: pd.DataFrame,
    journal_df: pd.DataFrame,
    match_tolerance_seconds: int = 300,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Match OANDA trades with journal trades.
    
    Returns:
        (matched_pairs, mismatches)
    """
    matched = []
    mismatches = []
    
    for _, oanda_row in oanda_df.iterrows():
        oanda_id = str(oanda_row["oanda_trade_id"])
        oanda_open_time = oanda_row["open_time"]
        oanda_units = oanda_row["units"]
        
        # Try to match by oanda_trade_id first
        journal_match = None
        if "oanda_trade_id" in journal_df.columns:
            journal_matches = journal_df[journal_df["oanda_trade_id"] == oanda_id]
            if len(journal_matches) > 0:
                journal_match = journal_matches.iloc[0]
        
        # If no match by ID, try by time + units
        if journal_match is None and "entry_time" in journal_df.columns:
            time_diff = abs((journal_df["entry_time"] - oanda_open_time).dt.total_seconds())
            units_match = journal_df["units"] == oanda_units
            candidates = journal_df[(time_diff <= match_tolerance_seconds) & units_match]
            if len(candidates) > 0:
                journal_match = candidates.iloc[0]
        
        if journal_match is not None:
            matched.append({
                "oanda_trade_id": oanda_id,
                "journal_trade_id": journal_match.get("trade_id", "N/A"),
                "state": oanda_row["state"],
            })
        else:
            mismatches.append({
                "oanda_trade_id": oanda_id,
                "state": oanda_row["state"],
                "open_time": oanda_open_time,
                "units": oanda_units,
                "issue": "MISSING_IN_JOURNAL",
            })
    
    # Check for journal trades not in OANDA
    if "oanda_trade_id" in journal_df.columns:
        journal_with_oanda_id = journal_df[journal_df["oanda_trade_id"].notna()]
        for _, journal_row in journal_with_oanda_id.iterrows():
            oanda_id = str(journal_row["oanda_trade_id"])
            if oanda_id not in oanda_df["oanda_trade_id"].values:
                is_open = journal_row.get("is_open", False)
                if is_open:
                    mismatches.append({
                        "journal_trade_id": journal_row.get("trade_id", "N/A"),
                        "oanda_trade_id": oanda_id,
                        "state": "LOCAL_ONLY_OPEN",
                        "issue": "LOCAL_ONLY_OPEN",
                    })
    
    return matched, mismatches


def match_journal_rl(
    journal_df: pd.DataFrame,
    rl_df: pd.DataFrame,
    match_tolerance_seconds: int = 300,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Match journal trades with RL dataset trades.
    
    Returns:
        (matched_pairs, mismatches)
    """
    matched = []
    mismatches = []
    
    # Filter to closed trades in journal
    closed_journal = journal_df[journal_df["is_open"] == False]
    
    # Filter to real trades in RL (action_taken == 1)
    if "action_taken" in rl_df.columns:
        real_rl = rl_df[rl_df["action_taken"] == 1]
    else:
        real_rl = rl_df
    
    # Find timestamp column in RL
    rl_ts_col = None
    for col in ["candle_time", "ts", "time", "timestamp"]:
        if col in rl_df.columns:
            rl_ts_col = col
            break
    
    for _, journal_row in closed_journal.iterrows():
        journal_trade_id = journal_row.get("trade_id", "")
        journal_entry_time = journal_row.get("entry_time")
        
        if pd.isna(journal_entry_time):
            continue
        
        # Try to match by timestamp
        rl_match = None
        if rl_ts_col:
            time_diff = abs((pd.to_datetime(real_rl[rl_ts_col]) - journal_entry_time).dt.total_seconds())
            candidates = real_rl[time_diff <= match_tolerance_seconds]
            if len(candidates) > 0:
                rl_match = candidates.iloc[0]
        
        if rl_match is not None:
            matched.append({
                "journal_trade_id": journal_trade_id,
                "rl_timestamp": rl_match[rl_ts_col] if rl_ts_col else "N/A",
            })
        else:
            mismatches.append({
                "journal_trade_id": journal_trade_id,
                "entry_time": journal_entry_time,
                "issue": "MISSING_IN_RL",
            })
    
    # Check for RL trades not in journal
    if rl_ts_col:
        for _, rl_row in real_rl.iterrows():
            rl_ts = pd.to_datetime(rl_row[rl_ts_col])
            # Try to match by timestamp
            if "entry_time" in journal_df.columns:
                time_diff = abs((pd.to_datetime(journal_df["entry_time"]) - rl_ts).dt.total_seconds())
                candidates = journal_df[time_diff <= match_tolerance_seconds]
                if len(candidates) == 0:
                    mismatches.append({
                        "rl_timestamp": rl_ts,
                        "issue": "MISSING_IN_JOURNAL",
                    })
    
    return matched, mismatches


def generate_report(
    date: str,
    run_dir: Path,
    dataset_path: Path,
    oanda_open: pd.DataFrame,
    oanda_closed: pd.DataFrame,
    journal_df: Optional[pd.DataFrame],
    rl_df: Optional[pd.DataFrame],
    oanda_journal_matched: List[Dict],
    oanda_journal_mismatches: List[Dict],
    journal_rl_matched: List[Dict],
    journal_rl_mismatches: List[Dict],
) -> str:
    """Generate markdown report."""
    lines = []
    
    lines.append(f"# SNIPER Live Coverage Check – {date}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(f"**Run Directory:** {run_dir}")
    lines.append(f"**RL Dataset:** {dataset_path}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Section 1: Counts
    lines.append("## 1. Counts")
    lines.append("")
    
    lines.append("### OANDA")
    lines.append(f"- **Open trades:** {len(oanda_open)}")
    lines.append(f"- **Closed trades (last 200, filtered to date):** {len(oanda_closed)}")
    lines.append("")
    
    if journal_df is not None:
        lines.append("### Trade Journal")
        lines.append(f"- **Total trades:** {len(journal_df)}")
        open_count = journal_df["is_open"].sum() if "is_open" in journal_df.columns else 0
        closed_count = len(journal_df) - open_count
        lines.append(f"- **Open trades:** {open_count}")
        lines.append(f"- **Closed trades:** {closed_count}")
    else:
        lines.append("### Trade Journal")
        lines.append("⚠️ **Trade journal index not found**")
    lines.append("")
    
    if rl_df is not None:
        lines.append("### RL Dataset")
        lines.append(f"- **Total rows:** {len(rl_df)}")
        if "action_taken" in rl_df.columns:
            real_count = (rl_df["action_taken"] == 1).sum()
            shadow_count = (rl_df["action_taken"] == 0).sum()
            lines.append(f"- **Real trades (action_taken=1):** {real_count}")
            lines.append(f"- **Shadow-only (action_taken=0):** {shadow_count}")
        else:
            lines.append("- **Real trades:** Unknown (action_taken column missing)")
            lines.append("- **Shadow-only:** Unknown")
    else:
        lines.append("### RL Dataset")
        lines.append("⚠️ **RL dataset not found**")
    lines.append("")
    
    # Section 2: OANDA ↔ Journal consistency
    lines.append("## 2. OANDA ↔ Journal Consistency")
    lines.append("")
    
    if len(oanda_journal_mismatches) == 0:
        lines.append("✅ **All OANDA trades are present in trade journal.**")
    else:
        lines.append("⚠️ **Mismatches found:**")
        lines.append("")
        lines.append("| OANDA Trade ID | State | Issue | Open Time | Units |")
        lines.append("|----------------|-------|-------|-----------|-------|")
        for mismatch in oanda_journal_mismatches:
            oanda_id = mismatch.get("oanda_trade_id", "N/A")
            state = mismatch.get("state", "N/A")
            issue = mismatch.get("issue", "N/A")
            open_time = mismatch.get("open_time", "N/A")
            if isinstance(open_time, pd.Timestamp):
                open_time = open_time.strftime("%Y-%m-%d %H:%M")
            units = mismatch.get("units", "N/A")
            lines.append(f"| {oanda_id} | {state} | {issue} | {open_time} | {units} |")
    
    lines.append("")
    lines.append(f"**Matched:** {len(oanda_journal_matched)} trades")
    lines.append("")
    
    # Section 3: Journal ↔ RL consistency
    lines.append("## 3. Journal ↔ RL Consistency")
    lines.append("")
    
    if journal_df is None or rl_df is None:
        lines.append("⚠️ **Cannot check consistency - missing data sources.**")
    else:
        if len(journal_rl_mismatches) == 0:
            lines.append("✅ **All closed journal trades are represented in RL dataset.**")
        else:
            lines.append("⚠️ **Mismatches found:**")
            lines.append("")
            lines.append("| Trade ID / Timestamp | Issue |")
            lines.append("|----------------------|-------|")
            for mismatch in journal_rl_mismatches:
                trade_id = mismatch.get("journal_trade_id", mismatch.get("rl_timestamp", "N/A"))
                if isinstance(trade_id, pd.Timestamp):
                    trade_id = trade_id.strftime("%Y-%m-%d %H:%M")
                issue = mismatch.get("issue", "N/A")
                lines.append(f"| {trade_id} | {issue} |")
    
    lines.append("")
    lines.append(f"**Matched:** {len(journal_rl_matched)} trades")
    lines.append("")
    
    # Section 4: Conclusion
    lines.append("## 4. Conclusion")
    lines.append("")
    
    all_consistent = (
        len(oanda_journal_mismatches) == 0 and
        len(journal_rl_mismatches) == 0 and
        journal_df is not None and
        rl_df is not None
    )
    
    if all_consistent:
        lines.append("✅ **All layers are consistent.**")
        lines.append("")
        lines.append("- All OANDA trades are logged in trade journal")
        lines.append("- All closed journal trades are represented in RL dataset")
    else:
        issues = []
        if len(oanda_journal_mismatches) > 0:
            issues.append(f"{len(oanda_journal_mismatches)} OANDA trades missing in journal")
        if len(journal_rl_mismatches) > 0:
            issues.append(f"{len(journal_rl_mismatches)} journal trades missing in RL dataset")
        if journal_df is None:
            issues.append("Trade journal index missing")
        if rl_df is None:
            issues.append("RL dataset missing")
        
        lines.append("⚠️ **Inconsistencies detected:**")
        lines.append("")
        for issue in issues:
            lines.append(f"- {issue}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `gx1/scripts/check_live_coverage.py`*")
    
    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SNIPER Live Coverage Checker"
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("runs/live_demo/SNIPER_20251226_113527"),
        help="SNIPER run directory",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/rl/sniper_shadow_rl_dataset_LIVE_DAY1_WITH_TRADES.parquet"),
        help="Path to RL dataset Parquet file",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date to check (YYYY-MM-DD, optional - inferred from dataset if not provided)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports/live"),
        help="Output directory for report",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="XAU_USD",
        help="Instrument symbol (default: XAU_USD)",
    )
    
    args = parser.parse_args()
    
    log.info(f"Checking live coverage for: {args.date or 'all dates'}")
    log.info(f"Run directory: {args.run_dir}")
    log.info(f"Dataset: {args.dataset}")
    
    # Determine date
    if args.date is None:
        # Try to infer from dataset
        if args.dataset.exists():
            try:
                df = pd.read_parquet(args.dataset)
                ts_col = None
                for col in ["candle_time", "ts", "time", "timestamp"]:
                    if col in df.columns:
                        ts_col = col
                        break
                if ts_col:
                    df[ts_col] = pd.to_datetime(df[ts_col])
                    first_date = df[ts_col].min().date()
                    args.date = first_date.strftime("%Y-%m-%d")
                    log.info(f"Inferred date from dataset: {args.date}")
            except Exception as e:
                log.warning(f"Failed to infer date from dataset: {e}")
                args.date = datetime.now().strftime("%Y-%m-%d")
        else:
            args.date = datetime.now().strftime("%Y-%m-%d")
    
    # Load OANDA credentials
    load_dotenv_if_present()
    try:
        credentials = load_oanda_credentials(prod_baseline=False)
        log.info(f"Loaded OANDA credentials: env={credentials.env}")
    except Exception as e:
        log.error(f"Failed to load OANDA credentials: {e}")
        log.warning("Continuing without OANDA data...")
        credentials = None
    
    # Load OANDA trades
    oanda_open = pd.DataFrame()
    oanda_closed = pd.DataFrame()
    if credentials:
        try:
            config = OandaClientConfig(
                api_key=credentials.api_token,
                account_id=credentials.account_id,
                env=credentials.env,
            )
            client = OandaClient(config)
            oanda_open, oanda_closed = load_oanda_trades(client, args.instrument, args.date)
        except Exception as e:
            log.error(f"Failed to fetch OANDA trades: {e}")
    
    # Load trade journal
    journal_df = load_trade_journal_index(args.run_dir, args.date)
    
    # Load RL dataset
    rl_df = load_rl_dataset(args.dataset, args.date)
    
    # Perform matching
    oanda_all = pd.DataFrame()
    if not oanda_open.empty:
        oanda_all = pd.concat([oanda_all, oanda_open], ignore_index=True)
    if not oanda_closed.empty:
        oanda_all = pd.concat([oanda_all, oanda_closed], ignore_index=True)
    
    oanda_journal_matched, oanda_journal_mismatches = match_trades(
        oanda_all,
        journal_df if journal_df is not None else pd.DataFrame(),
    )
    
    journal_rl_matched, journal_rl_mismatches = match_journal_rl(
        journal_df if journal_df is not None else pd.DataFrame(),
        rl_df if rl_df is not None else pd.DataFrame(),
    )
    
    # Generate report
    report = generate_report(
        date=args.date,
        run_dir=args.run_dir,
        dataset_path=args.dataset,
        oanda_open=oanda_open,
        oanda_closed=oanda_closed,
        journal_df=journal_df,
        rl_df=rl_df,
        oanda_journal_matched=oanda_journal_matched,
        oanda_journal_mismatches=oanda_journal_mismatches,
        journal_rl_matched=journal_rl_matched,
        journal_rl_mismatches=journal_rl_mismatches,
    )
    
    # Write report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"LIVE_COVERAGE_{args.date.replace('-', '_')}.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    log.info(f"Report saved: {report_path}")
    log.info("✅ Coverage check complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())

