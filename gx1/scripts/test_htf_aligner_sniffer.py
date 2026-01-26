#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sniffer test for HTF aligner stateful alignment.

Verifies that stateful HTF alignment is used correctly in replay:
- htf_align_call_count_total ≈ 2× n_bars (H1 + H4)
- fallback_count == 0
- Fails if deviation > 20%
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Sniffer test for HTF aligner")
    parser.add_argument("--perf-json", type=Path, required=True, help="Path to perf JSON file")
    parser.add_argument("--tolerance", type=float, default=0.20, help="Tolerance for call count deviation (default: 0.20 = 20%%)")
    args = parser.parse_args()
    
    with open(args.perf_json) as f:
        data = json.load(f)
    
    # Aggregate metrics across chunks
    total_bars = sum(c.get("total_bars", 0) for c in data.get("chunks", []))
    total_align_calls = sum(c.get("htf_align_call_count", 0) for c in data.get("chunks", []))
    total_fallback = sum(c.get("htf_align_fallback_count", 0) for c in data.get("chunks", []))
    
    expected_calls = total_bars * 2  # H1 + H4
    deviation = abs(total_align_calls - expected_calls) / expected_calls if expected_calls > 0 else 1.0
    
    print("=" * 60)
    print("HTF Aligner Sniffer Test")
    print("=" * 60)
    print(f"Perf JSON: {args.perf_json}")
    print(f"Total bars: {total_bars:,}")
    print(f"HTF align calls: {total_align_calls:,} (expected: {expected_calls:,})")
    print(f"Deviation: {deviation*100:.1f}%")
    print(f"Fallback count: {total_fallback}")
    print()
    
    # Check 1: Call count deviation
    if deviation > args.tolerance:
        print(f"❌ FAIL: Call count deviation ({deviation*100:.1f}%) exceeds tolerance ({args.tolerance*100:.1f}%)")
        print(f"   Expected: {expected_calls:,}, Got: {total_align_calls:,}")
        sys.exit(1)
    else:
        print(f"✅ PASS: Call count deviation ({deviation*100:.1f}%) within tolerance ({args.tolerance*100:.1f}%)")
    
    # Check 2: Fallback count
    if total_fallback > 0:
        print(f"❌ FAIL: Fallback count is {total_fallback} (expected: 0)")
        sys.exit(1)
    else:
        print(f"✅ PASS: Fallback count is 0")
    
    print()
    print("=" * 60)
    print("✅ ALL CHECKS PASSED")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
