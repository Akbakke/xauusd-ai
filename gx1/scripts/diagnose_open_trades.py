#!/usr/bin/env python3
"""
Diagnose open trades in merged trade journal.

Identifies why trades remain open after merge and classifies them into buckets:
- Bucket A: Closed in chunk but merge chose open version
- Bucket B: Close event in logs but journal missing
- Bucket C: Never closed in chunk (EOF-close missed)
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def load_merged_index(run_dir: Path) -> pd.DataFrame:
    """Load merged trade journal index."""
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if not index_path.exists():
        log.error("Merged index not found: %s", index_path)
        return pd.DataFrame()
    
    df = pd.read_csv(index_path)
    log.info("Loaded merged index: %d trades", len(df))
    return df


def find_open_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Find trades without exit_reason or exit_time."""
    open_mask = df["exit_reason"].isna() | (df["exit_reason"] == "") | df["exit_time"].isna() | (df["exit_time"] == "")
    open_trades = df[open_mask].copy()
    log.info("Found %d open trades", len(open_trades))
    return open_trades


def load_chunk_journals(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load trade journal indices from all chunks."""
    chunk_indices = {}
    
    # Check parallel chunks
    parallel_chunks_dir = run_dir / "parallel_chunks"
    if parallel_chunks_dir.exists():
        for chunk_dir in sorted(parallel_chunks_dir.glob("chunk_*")):
            chunk_id = chunk_dir.name
            chunk_index_path = chunk_dir / "trade_journal" / "trade_journal_index.csv"
            
            if chunk_index_path.exists():
                try:
                    chunk_df = pd.read_csv(chunk_index_path)
                    chunk_indices[chunk_id] = chunk_df
                    log.info("Loaded chunk %s index: %d trades", chunk_id, len(chunk_df))
                except Exception as e:
                    log.warning("Failed to load chunk %s index: %s", chunk_id, e)
    
    # Check legacy location (gx1/live/trade_journal)
    legacy_index = Path("gx1/live/trade_journal/trade_journal_index.csv")
    if legacy_index.exists():
        try:
            legacy_df = pd.read_csv(legacy_index)
            chunk_indices["legacy"] = legacy_df
            log.info("Loaded legacy index: %d trades", len(legacy_df))
        except Exception as e:
            log.warning("Failed to load legacy index: %s", e)
    
    return chunk_indices


def check_chunk_logs(run_dir: Path, trade_id: str) -> Dict[str, any]:
    """Check chunk logs for close events for a trade_id."""
    log_events = {
        "request_close": False,
        "exit_summary": False,
        "REPLAY_EOF": False,
        "chunk_with_close": None,
        "log_lines": [],
    }
    
    # Check parallel chunk logs
    parallel_chunks_dir = run_dir / "parallel_chunks"
    if parallel_chunks_dir.exists():
        for chunk_dir in sorted(parallel_chunks_dir.glob("chunk_*")):
            chunk_log = chunk_dir / f"{chunk_dir.name}.log"
            if chunk_log.exists():
                try:
                    with open(chunk_log, "r") as f:
                        log_content = f.read()
                        
                        # Check for close events
                        if trade_id in log_content:
                            # Look for specific patterns
                            if f"request_close.*{trade_id}" in log_content or f"Close.*{trade_id}" in log_content:
                                log_events["request_close"] = True
                                log_events["chunk_with_close"] = chunk_dir.name
                            
                            if f"exit_summary.*{trade_id}" in log_content or f"EXIT.*{trade_id}" in log_content:
                                log_events["exit_summary"] = True
                                log_events["chunk_with_close"] = chunk_dir.name
                            
                            if f"REPLAY_EOF.*{trade_id}" in log_content or f"EOF.*{trade_id}" in log_content:
                                log_events["REPLAY_EOF"] = True
                                log_events["chunk_with_close"] = chunk_dir.name
                            
                            # Extract relevant log lines
                            for line in log_content.split("\n"):
                                if trade_id in line and ("close" in line.lower() or "exit" in line.lower() or "EOF" in line.lower()):
                                    log_events["log_lines"].append(line.strip())
                                    if len(log_events["log_lines"]) >= 5:
                                        break
                except Exception as e:
                    log.debug("Failed to read chunk log %s: %s", chunk_log, e)
    
    return log_events


def classify_open_trade(
    trade_id: str,
    merged_row: pd.Series,
    chunk_indices: Dict[str, pd.DataFrame],
    run_dir: Path,
) -> Tuple[str, Dict]:
    """
    Classify an open trade into bucket A, B, or C.
    
    Returns:
        (bucket, details_dict)
    """
    details = {
        "trade_id": trade_id,
        "merged_exit_reason": merged_row.get("exit_reason"),
        "merged_exit_time": merged_row.get("exit_time"),
        "merged_source": merged_row.get("source_chunk", "unknown"),
    }
    
    # Bucket A: Check if closed version exists in chunks
    closed_in_chunk = False
    chunk_with_close = None
    
    for chunk_id, chunk_df in chunk_indices.items():
        chunk_trades = chunk_df[chunk_df["trade_id"] == trade_id]
        if len(chunk_trades) > 0:
            chunk_row = chunk_trades.iloc[0]
            chunk_exit_reason = chunk_row.get("exit_reason")
            chunk_exit_time = chunk_row.get("exit_time")
            
            if pd.notna(chunk_exit_reason) and chunk_exit_reason != "":
                closed_in_chunk = True
                chunk_with_close = chunk_id
                details["chunk_exit_reason"] = chunk_exit_reason
                details["chunk_exit_time"] = chunk_exit_time
                details["chunk_with_close"] = chunk_id
                break
    
    if closed_in_chunk:
        return "A", details
    
    # Bucket B: Check logs for close events
    log_events = check_chunk_logs(run_dir, trade_id)
    if log_events["request_close"] or log_events["exit_summary"] or log_events["REPLAY_EOF"]:
        details.update(log_events)
        return "B", details
    
    # Bucket C: Never closed
    return "C", details


def diagnose_open_trades(run_dir: Path) -> Dict:
    """Main diagnosis function."""
    run_dir = Path(run_dir)
    
    # Load merged index
    merged_df = load_merged_index(run_dir)
    if merged_df.empty:
        return {"error": "No merged index found"}
    
    # Find open trades
    open_trades = find_open_trades(merged_df)
    if len(open_trades) == 0:
        log.info("✅ No open trades found!")
        return {
            "run_dir": str(run_dir),
            "total_trades": len(merged_df),
            "open_trades": 0,
            "bucket_counts": {},
            "bucket_details": {},
        }
    
    log.info("Analyzing %d open trades...", len(open_trades))
    
    # Load chunk indices
    chunk_indices = load_chunk_journals(run_dir)
    
    # Classify each open trade
    bucket_counts = defaultdict(int)
    bucket_details = defaultdict(list)
    
    for _, row in open_trades.iterrows():
        trade_id = row["trade_id"]
        bucket, details = classify_open_trade(trade_id, row, chunk_indices, run_dir)
        bucket_counts[bucket] += 1
        bucket_details[bucket].append(details)
    
    # Build report
    report = {
        "run_dir": str(run_dir),
        "total_trades": len(merged_df),
        "open_trades": len(open_trades),
        "bucket_counts": dict(bucket_counts),
        "bucket_details": {k: v[:10] for k, v in bucket_details.items()},  # Top 10 per bucket
    }
    
    return report


def write_reports(run_dir: Path, report: Dict):
    """Write CSV and Markdown reports."""
    output_dir = run_dir / "trade_journal"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV report
    csv_data = []
    for bucket, details_list in report.get("bucket_details", {}).items():
        for details in details_list:
            csv_data.append({
                "bucket": bucket,
                "trade_id": details.get("trade_id"),
                "entry_time": details.get("entry_time", ""),
                "merged_exit_reason": details.get("merged_exit_reason", ""),
                "merged_exit_time": details.get("merged_exit_time", ""),
                "chunk_exit_reason": details.get("chunk_exit_reason", ""),
                "chunk_exit_time": details.get("chunk_exit_time", ""),
                "chunk_with_close": details.get("chunk_with_close", ""),
                "merged_source": details.get("merged_source", ""),
            })
    
    if csv_data:
        csv_df = pd.DataFrame(csv_data)
        csv_path = output_dir / "open_trades_report.csv"
        csv_df.to_csv(csv_path, index=False)
        log.info("Wrote CSV report: %s", csv_path)
    
    # Markdown report
    md_path = output_dir / "open_trades_report.md"
    with open(md_path, "w") as f:
        f.write("# Open Trades Diagnosis Report\n\n")
        f.write(f"**Run Directory:** `{report['run_dir']}`\n\n")
        f.write(f"**Total Trades:** {report['total_trades']}\n")
        f.write(f"**Open Trades:** {report['open_trades']}\n\n")
        
        f.write("## Bucket Classification\n\n")
        for bucket, count in sorted(report["bucket_counts"].items()):
            bucket_name = {
                "A": "Closed in chunk but merge chose open",
                "B": "Close event in logs but journal missing",
                "C": "Never closed in chunk",
            }.get(bucket, bucket)
            f.write(f"- **Bucket {bucket}** ({bucket_name}): {count} trades\n")
        
        f.write("\n## Sample Trades (Top 10 per bucket)\n\n")
        for bucket in ["A", "B", "C"]:
            if bucket in report["bucket_details"]:
                f.write(f"### Bucket {bucket}\n\n")
                for i, details in enumerate(report["bucket_details"][bucket][:10], 1):
                    f.write(f"{i}. **{details['trade_id']}**\n")
                    f.write(f"   - Merged exit_reason: `{details.get('merged_exit_reason', 'N/A')}`\n")
                    if "chunk_exit_reason" in details:
                        f.write(f"   - Chunk exit_reason: `{details['chunk_exit_reason']}` (from {details.get('chunk_with_close', 'unknown')})\n")
                    if "chunk_with_close" in details:
                        f.write(f"   - Close found in chunk: `{details['chunk_with_close']}`\n")
                    f.write("\n")
    
    log.info("Wrote Markdown report: %s", md_path)


def main():
    parser = argparse.ArgumentParser(description="Diagnose open trades in merged trade journal")
    parser.add_argument("--run", type=Path, required=True, help="Run directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.run.exists():
        log.error("Run directory does not exist: %s", args.run)
        return 1
    
    log.info("Diagnosing open trades in: %s", args.run)
    
    report = diagnose_open_trades(args.run)
    
    if "error" in report:
        log.error("Diagnosis failed: %s", report["error"])
        return 1
    
    write_reports(args.run, report)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Open Trades Diagnosis Summary")
    print("=" * 60)
    print(f"Total trades: {report['total_trades']}")
    print(f"Open trades: {report['open_trades']}")
    print("\nBucket breakdown:")
    for bucket, count in sorted(report["bucket_counts"].items()):
        bucket_name = {
            "A": "Closed in chunk but merge chose open",
            "B": "Close event in logs but journal missing",
            "C": "Never closed in chunk",
        }.get(bucket, bucket)
        print(f"  Bucket {bucket} ({bucket_name}): {count}")
    
    print(f"\n✅ Reports written to: {args.run / 'trade_journal'}")
    
    return 0


if __name__ == "__main__":
    exit(main())

