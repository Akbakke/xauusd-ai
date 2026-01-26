#!/usr/bin/env python3
"""
Combine SNIPER quarter runs into a single full-year run directory.

This creates a symbolic structure that portfolio combination script can use.
"""

import shutil
import sys
from pathlib import Path
from typing import List


def combine_quarter_runs(
    q1_dir: Path,
    q2_dir: Path,
    q3_dir: Path,
    q4_dir: Path,
    output_dir: Path,
) -> None:
    """Combine four quarter runs into one full-year run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create parallel_chunks directory
    chunks_dir = output_dir / "parallel_chunks"
    chunks_dir.mkdir(exist_ok=True)
    
    # Copy chunks from each quarter
    chunk_idx = 0
    for quarter_dir, quarter_name in [
        (q1_dir, "Q1"),
        (q2_dir, "Q2"),
        (q3_dir, "Q3"),
        (q4_dir, "Q4"),
    ]:
        if not quarter_dir.exists():
            print(f"WARNING: {quarter_name} directory not found: {quarter_dir}", file=sys.stderr)
            continue
        
        # Find parallel_chunks in quarter dir
        quarter_chunks = quarter_dir / "parallel_chunks"
        if not quarter_chunks.exists():
            print(f"WARNING: No parallel_chunks in {quarter_name}: {quarter_dir}", file=sys.stderr)
            continue
        
        # Copy each chunk
        for chunk_dir in sorted(quarter_chunks.glob("chunk_*")):
            new_chunk_name = f"chunk_{chunk_idx}"
            new_chunk_dir = chunks_dir / new_chunk_name
            shutil.copytree(chunk_dir, new_chunk_dir, dirs_exist_ok=True)
            print(f"Copied {quarter_name} {chunk_dir.name} → {new_chunk_name}")
            chunk_idx += 1
    
    print(f"\n✅ Combined {chunk_idx} chunks into {output_dir}")
    print(f"   Total chunks: {chunk_idx}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine SNIPER quarter runs")
    parser.add_argument("--q1-dir", type=Path, required=True)
    parser.add_argument("--q2-dir", type=Path, required=True)
    parser.add_argument("--q3-dir", type=Path, required=True)
    parser.add_argument("--q4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    
    args = parser.parse_args()
    
    combine_quarter_runs(
        args.q1_dir,
        args.q2_dir,
        args.q3_dir,
        args.q4_dir,
        args.output_dir,
    )

