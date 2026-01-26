#!/usr/bin/env python3
"""
Build ENTRY_V10.1 Edge Buckets from Label Quality Analysis

Reads JSON output from analyze_entry_v10_label_quality.py and builds edge-buckets
for aggressive sizing based on expected PnL per quantile bin.

Usage:
    python -m gx1.rl.entry_v10.build_v10_1_edge_buckets \
        --label-quality-json reports/rl/entry_v10/ENTRY_V10_1_LABEL_QUALITY_2025.json \
        --output-json data/entry_v10/entry_v10_1_edge_buckets_2025_flat.json \
        --min-trades-per-bin 30
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

from gx1.rl.entry_v10.thresholds_v10 import compute_v10_1_edge_buckets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build ENTRY_V10.1 edge buckets from label quality analysis"
    )
    parser.add_argument(
        "--label-quality-json",
        type=Path,
        required=True,
        help="Path to label quality JSON (from analyze_entry_v10_label_quality.py)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output path for edge buckets JSON",
    )
    parser.add_argument(
        "--min-trades-per-bin",
        type=int,
        default=30,
        help="Minimum trades required per bin (default: 30)",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.label_quality_json.exists():
        log.error(f"Label quality JSON not found: {args.label_quality_json}")
        return 1
    
    # Compute edge buckets
    log.info("Computing edge buckets from label quality analysis...")
    edge_buckets = compute_v10_1_edge_buckets(
        label_quality_json_path=args.label_quality_json,
        min_trades_per_bin=args.min_trades_per_bin,
    )
    
    # Save output
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(edge_buckets, f, indent=2)
    
    log.info(f"✅ Edge buckets saved to {args.output_json}")
    
    # Print summary
    total_bins = 0
    for session in ["EU", "OVERLAP", "US"]:
        for regime, bins in edge_buckets.get(session, {}).items():
            total_bins += len(bins)
    
    log.info(f"Summary: {total_bins} total bins across all sessions/regimes")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

