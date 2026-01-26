#!/usr/bin/env python3
"""
Summarize replay performance metrics from chunk logs.

Parses [REPLAY_PERF_SUMMARY] lines from all chunk logs and generates
an aggregated performance report.

Usage:
    python -m gx1.tools.analysis.summarize_replay_perf_v10_1_threshold018_2025 \
        --log-dir data/replay/sniper/entry_v10_1_flat_threshold0_18/2025/FLAT_THRESHOLD_0_18_FIXED_20260101_180231/logs \
        --output-report reports/rl/entry_v10/ENTRY_V10_1_REPLAY_PERF_2025_THRESHOLD_0_18_FIXED.md
"""

import argparse
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_perf_summary_line(line: str) -> Optional[Dict]:
    """
    Parse [REPLAY_PERF_SUMMARY] line.
    
    Format: [REPLAY_PERF_SUMMARY] bars=X trades=Y total=Z bars_per_sec=W 
            feat=A (B%) model=C (D%) journal=E (F%) other=G (H%)
    """
    pattern = (
        r"\[REPLAY_PERF_SUMMARY\]\s+"
        r"bars=(\d+)\s+"
        r"trades=(\d+)\s+"
        r"total=([\d.]+)s\s+"
        r"bars_per_sec=([\d.]+)\s+"
        r"feat=([\d.]+)s\s+\(([\d.]+)%\)\s+"
        r"model=([\d.]+)s\s+\(([\d.]+)%\)\s+"
        r"journal=([\d.]+)s\s+\(([\d.]+)%\)\s+"
        r"other=([\d.]+)s\s+\(([\d.]+)%\)"
    )
    
    match = re.search(pattern, line)
    if not match:
        return None
    
    return {
        "bars": int(match.group(1)),
        "trades": int(match.group(2)),
        "total_time": float(match.group(3)),
        "bars_per_sec": float(match.group(4)),
        "feat_time": float(match.group(5)),
        "feat_pct": float(match.group(6)),
        "model_time": float(match.group(7)),
        "model_pct": float(match.group(8)),
        "journal_time": float(match.group(9)),
        "journal_pct": float(match.group(10)),
        "other_time": float(match.group(11)),
        "other_pct": float(match.group(12)),
    }


def find_perf_summaries(log_dir: Path) -> List[Tuple[str, Dict]]:
    """
    Find all [REPLAY_PERF_SUMMARY] lines in log files.
    
    Returns:
        List of (chunk_id, perf_dict) tuples.
    """
    summaries = []
    
    # Look for chunk_*.log files
    log_files = sorted(log_dir.glob("chunk_*.log"))
    if not log_files:
        # Try alternative location
        log_files = sorted(log_dir.glob("**/chunk_*.log"))
    
    for log_file in log_files:
        chunk_id = log_file.stem  # e.g., "chunk_0"
        log.info(f"Scanning {log_file.name}...")
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    perf = parse_perf_summary_line(line)
                    if perf:
                        summaries.append((chunk_id, perf))
                        log.info(f"Found [REPLAY_PERF_SUMMARY] in {chunk_id}")
        except Exception as e:
            log.warning(f"Failed to read {log_file}: {e}")
    
    return summaries


def aggregate_perf_summaries(summaries: List[Tuple[str, Dict]]) -> Dict:
    """
    Aggregate performance summaries across all chunks.
    
    Returns:
        Aggregated metrics dictionary.
    """
    if not summaries:
        raise ValueError("No performance summaries found")
    
    total_bars = sum(s["bars"] for _, s in summaries)
    total_trades = sum(s["trades"] for _, s in summaries)
    total_time = sum(s["total_time"] for _, s in summaries)
    total_feat_time = sum(s["feat_time"] for _, s in summaries)
    total_model_time = sum(s["model_time"] for _, s in summaries)
    total_journal_time = sum(s["journal_time"] for _, s in summaries)
    total_other_time = sum(s["other_time"] for _, s in summaries)
    
    # Calculate percentages based on total time
    if total_time > 0:
        feat_pct = 100 * total_feat_time / total_time
        model_pct = 100 * total_model_time / total_time
        journal_pct = 100 * total_journal_time / total_time
        other_pct = 100 * total_other_time / total_time
        bars_per_sec = total_bars / total_time
    else:
        feat_pct = model_pct = journal_pct = other_pct = 0.0
        bars_per_sec = 0.0
    
    return {
        "n_chunks": len(summaries),
        "total_bars": total_bars,
        "total_trades": total_trades,
        "total_time": total_time,
        "bars_per_sec": bars_per_sec,
        "feat_time": total_feat_time,
        "feat_pct": feat_pct,
        "model_time": total_model_time,
        "model_pct": model_pct,
        "journal_time": total_journal_time,
        "journal_pct": journal_pct,
        "other_time": total_other_time,
        "other_pct": other_pct,
        "chunk_details": summaries,
    }


def identify_bottleneck(agg: Dict) -> str:
    """Identify the primary bottleneck."""
    if agg["journal_pct"] > 30:
        return "journaling"
    elif agg["model_pct"] > 30:
        return "model_inference"
    elif agg["feat_pct"] > 30:
        return "feature_building"
    elif agg["other_pct"] > 30:
        return "other"
    else:
        return "well_distributed"


def generate_report(agg: Dict, output_path: Path, log_dir: Path) -> None:
    """Generate markdown performance report."""
    bottleneck = identify_bottleneck(agg)
    
    # TL;DR at the top
    if bottleneck == "journaling":
        tldr = f"**TL;DR:** Journaling (JSON+disk I/O) står for {agg['journal_pct']:.1f}% av tiden."
    elif bottleneck == "model_inference":
        tldr = f"**TL;DR:** Modell-inferens står for {agg['model_pct']:.1f}% av tiden."
    elif bottleneck == "feature_building":
        tldr = f"**TL;DR:** Featurebygging står for {agg['feat_pct']:.1f}% av tiden."
    elif bottleneck == "other":
        tldr = f"**TL;DR:** Annet (ikke målt) står for {agg['other_pct']:.1f}% av tiden."
    else:
        tldr = "**TL;DR:** Ingen enkelt komponent dominerer (>30%). Tiden er godt fordelt."
    
    lines = [
        "# ENTRY_V10.1 FLAT THRESHOLD 0.18 FULLYEAR 2025 Replay Performance Profile (FIXED)",
        "",
        tldr,
        "",
        "**Date:** Generated from completed replay run",
        "",
        "**Configuration:**",
        "- Policy: GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18",
        "- Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)",
        "- Threshold: min_prob_long=0.18, p_side_min.long=0.18",
        "- Sizing: FLAT (baseline sizing, no aggressive overlays)",
        "- Exit: ExitCritic V1 + RULE5/RULE6A",
        "- Workers: 7 (parallel replay)",
        "- Run: FLAT_THRESHOLD_0_18_FIXED_20260101_180231",
        "",
        "## Performance Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Total Bars Processed** | {agg['total_bars']:,} |",
        f"| **Total Trades** | {agg['total_trades']:,} |",
        f"| **Total Time** | {agg['total_time']:.1f} seconds ({agg['total_time']/60:.1f} minutes) |",
        f"| **Bars per Second** | {agg['bars_per_sec']:.2f} |",
        f"| **Chunks Processed** | {agg['n_chunks']} |",
        "",
        "## Time Distribution",
        "",
        "| Component | Time (seconds) | Percentage |",
        "|-----------|---------------|------------|",
        f"| **Feature Building** | {agg['feat_time']:.1f} | {agg['feat_pct']:.1f}% |",
        f"| **Model Inference** | {agg['model_time']:.1f} | {agg['model_pct']:.1f}% |",
        f"| **Trade Journaling** | {agg['journal_time']:.1f} | {agg['journal_pct']:.1f}% |",
        f"| **Other** | {agg['other_time']:.1f} | {agg['other_pct']:.1f}% |",
        f"| **Total** | {agg['total_time']:.1f} | 100.0% |",
        "",
        "## Bottleneck Analysis",
        "",
    ]
    
    if bottleneck == "journaling":
        lines.extend([
            f"⚠️  **Primary Bottleneck: Trade Journaling ({agg['journal_pct']:.1f}%)**",
            "",
            "Trade journaling (JSON serialization + disk I/O) is the dominant time consumer.",
            "This includes:",
            "- Writing per-trade JSON files",
            "- Serializing entry_snapshot, feature_context, exit_summary",
            "- Disk I/O operations",
            "",
            "**Recommendations:**",
            "- Consider batching JSON writes (write multiple trades per file)",
            "- Use binary formats (parquet) instead of JSON for trade journal",
            "- Reduce JSON size (remove redundant fields, compress)",
            "- Batch writes instead of per-trade writes",
            "- Consider async I/O for journal writes",
        ])
    elif bottleneck == "model_inference":
        lines.extend([
            f"⚠️  **Primary Bottleneck: Model Inference ({agg['model_pct']:.1f}%)**",
            "",
            "Model inference (V10.1 Hybrid: XGBoost + Transformer) is the dominant time consumer.",
            "",
            "**Recommendations:**",
            "- Pre-load models in memory (avoid repeated loading)",
            "- Batch predictions if possible (process multiple bars at once)",
            "- Use model quantization or optimization",
            "- Consider GPU acceleration for Transformer inference",
            "- Profile specific model components (XGBoost vs Transformer)",
        ])
    elif bottleneck == "feature_building":
        lines.extend([
            f"⚠️  **Primary Bottleneck: Feature Building ({agg['feat_pct']:.1f}%)**",
            "",
            "Feature building (build_live_entry_features) is the dominant time consumer.",
            "",
            "**Recommendations:**",
            "- Cache computed features where possible",
            "- Optimize pandas operations (use vectorization)",
            "- Profile specific feature calculations",
            "- Consider pre-computing static features",
        ])
    elif bottleneck == "other":
        lines.extend([
            f"⚠️  **Primary Bottleneck: Other ({agg['other_pct']:.1f}%)**",
            "",
            "Unmeasured time (other) is significant. This may include:",
            "- Exit evaluation",
            "- Trade management logic",
            "- Python overhead",
            "- Uninstrumented code paths",
            "",
            "**Recommendations:**",
            "- Add more timing hooks to identify what 'other' includes",
            "- Profile with cProfile or py-spy to find hotspots",
        ])
    else:
        lines.extend([
            "✅ **No Single Bottleneck Identified**",
            "",
            "Performance is well-distributed across components.",
            "No single component exceeds 30% of total time.",
            "",
            "**Recommendations:**",
            "- Continue monitoring as code evolves",
            "- Consider micro-optimizations across all components",
        ])
    
    lines.extend([
        "",
        "## Per-Chunk Breakdown",
        "",
        "| Chunk | Bars | Trades | Time (s) | Feat % | Model % | Journal % | Other % |",
        "|-------|------|--------|----------|--------|---------|-----------|---------|",
    ])
    
    for chunk_id, perf in agg["chunk_details"]:
        lines.append(
            f"| {chunk_id} | {perf['bars']:,} | {perf['trades']:,} | {perf['total_time']:.1f} | "
            f"{perf['feat_pct']:.1f}% | {perf['model_pct']:.1f}% | {perf['journal_pct']:.1f}% | {perf['other_pct']:.1f}% |"
        )
    
    lines.extend([
        "",
        "## Notes",
        "",
        "- Performance metrics are aggregated across all parallel chunks.",
        "- Each chunk ran independently with its own timing counters.",
        "- Total time is the sum of all chunk times (not wall-clock time).",
        "- Bars per second is calculated from total bars / total time.",
        "",
        f"- **Log Directory:** `{log_dir}`",
        "",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    log.info(f"✅ Performance report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize replay performance metrics from chunk logs"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Directory containing chunk log files (chunk_*.log)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("reports/rl/entry_v10/ENTRY_V10_1_REPLAY_PERF_2025_THRESHOLD_0_18_FIXED.md"),
        help="Path to save the performance report",
    )
    
    args = parser.parse_args()
    
    if not args.log_dir.exists():
        log.error(f"Log directory does not exist: {args.log_dir}")
        return 1
    
    # Find all performance summaries
    log.info(f"Scanning log directory: {args.log_dir}")
    summaries = find_perf_summaries(args.log_dir)
    
    if not summaries:
        log.error("No [REPLAY_PERF_SUMMARY] lines found in log files")
        log.error("Make sure log files contain performance metrics from replay run")
        return 1
    
    log.info(f"Found {len(summaries)} performance summaries")
    
    # Aggregate
    agg = aggregate_perf_summaries(summaries)
    
    # Generate report
    generate_report(agg, args.output_report, args.log_dir)
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Total Bars: {agg['total_bars']:,}")
    print(f"Total Trades: {agg['total_trades']:,}")
    print(f"Total Time: {agg['total_time']:.1f}s ({agg['total_time']/60:.1f} min)")
    print(f"Bars/sec: {agg['bars_per_sec']:.2f}")
    print()
    print("Time Distribution:")
    print(f"  Feature Building: {agg['feat_time']:.1f}s ({agg['feat_pct']:.1f}%)")
    print(f"  Model Inference:  {agg['model_time']:.1f}s ({agg['model_pct']:.1f}%)")
    print(f"  Trade Journaling: {agg['journal_time']:.1f}s ({agg['journal_pct']:.1f}%)")
    print(f"  Other:            {agg['other_time']:.1f}s ({agg['other_pct']:.1f}%)")
    print()
    bottleneck = identify_bottleneck(agg)
    print(f"Primary Bottleneck: {bottleneck}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

