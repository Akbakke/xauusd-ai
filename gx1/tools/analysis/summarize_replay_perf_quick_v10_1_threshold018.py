#!/usr/bin/env python3
"""
Summarize quick replay performance metrics from log file.

Parses [REPLAY_PERF_SUMMARY] line from a single-worker replay log and generates
a performance report.

Usage:
    python -m gx1.tools.analysis.summarize_replay_perf_quick_v10_1_threshold018 \
        --log-file /tmp/profile_v10_1_threshold018_quick.log \
        --output-report reports/rl/entry_v10/ENTRY_V10_1_REPLAY_PERF_QUICK_THRESHOLD_0_18.md
"""

import argparse
import re
import logging
from pathlib import Path
from typing import Dict, Optional

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


def identify_bottleneck(perf: Dict) -> tuple:
    """Identify the primary bottleneck and return (name, percentage)."""
    if perf["journal_pct"] > 30:
        return ("journaling", perf["journal_pct"])
    elif perf["model_pct"] > 30:
        return ("model_inference", perf["model_pct"])
    elif perf["feat_pct"] > 30:
        return ("feature_building", perf["feat_pct"])
    elif perf["other_pct"] > 30:
        return ("other", perf["other_pct"])
    else:
        return ("well_distributed", None)


def generate_report(perf: Dict, output_path: Path, log_file: Path) -> None:
    """Generate markdown performance report."""
    bottleneck_name, bottleneck_pct = identify_bottleneck(perf)
    
    # TL;DR at the top
    if bottleneck_name == "journaling":
        tldr = f"**TL;DR:** Journaling (JSON+disk I/O) står for {bottleneck_pct:.1f}% av tiden."
    elif bottleneck_name == "model_inference":
        tldr = f"**TL;DR:** Modell-inferens står for {bottleneck_pct:.1f}% av tiden."
    elif bottleneck_name == "feature_building":
        tldr = f"**TL;DR:** Featurebygging står for {bottleneck_pct:.1f}% av tiden."
    elif bottleneck_name == "other":
        tldr = f"**TL;DR:** Annet (ikke målt) står for {bottleneck_pct:.1f}% av tiden."
    else:
        tldr = "**TL;DR:** Ingen enkelt komponent dominerer (>30%). Tiden er godt fordelt."
    
    lines = [
        "# ENTRY_V10.1 FLAT THRESHOLD 0.18 Quick Performance Profile",
        "",
        tldr,
        "",
        "**Date:** Generated from quick replay run (1 month, single worker)",
        "",
        "**Configuration:**",
        "- Policy: GX1_V12_SNIPER_ENTRY_V10_1_FLAT_THRESHOLD_0_18",
        "- Entry: ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)",
        "- Threshold: min_prob_long=0.18, p_side_min.long=0.18",
        "- Sizing: FLAT (baseline sizing, no aggressive overlays)",
        "- Exit: ExitCritic V1 + RULE5/RULE6A",
        "- Workers: 1 (single worker, no parallel chunks)",
        "- Period: 1 month (2025-01-01 → 2025-02-01)",
        "",
        "## Performance Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Total Bars Processed** | {perf['bars']:,} |",
        f"| **Total Trades** | {perf['trades']:,} |",
        f"| **Total Time** | {perf['total_time']:.1f} seconds ({perf['total_time']/60:.1f} minutes) |",
        f"| **Bars per Second** | {perf['bars_per_sec']:.2f} |",
        "",
        "## Time Distribution",
        "",
        "| Component | Time (seconds) | Percentage |",
        "|-----------|---------------|------------|",
        f"| **Feature Building** | {perf['feat_time']:.1f} | {perf['feat_pct']:.1f}% |",
        f"| **Model Inference** | {perf['model_time']:.1f} | {perf['model_pct']:.1f}% |",
        f"| **Trade Journaling** | {perf['journal_time']:.1f} | {perf['journal_pct']:.1f}% |",
        f"| **Other** | {perf['other_time']:.1f} | {perf['other_pct']:.1f}% |",
        f"| **Total** | {perf['total_time']:.1f} | 100.0% |",
        "",
        "## Bottleneck Analysis",
        "",
    ]
    
    if bottleneck_name == "journaling":
        lines.extend([
            f"⚠️  **Primary Bottleneck: Trade Journaling ({bottleneck_pct:.1f}%)**",
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
    elif bottleneck_name == "model_inference":
        lines.extend([
            f"⚠️  **Primary Bottleneck: Model Inference ({bottleneck_pct:.1f}%)**",
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
    elif bottleneck_name == "feature_building":
        lines.extend([
            f"⚠️  **Primary Bottleneck: Feature Building ({bottleneck_pct:.1f}%)**",
            "",
            "Feature building (build_live_entry_features) is the dominant time consumer.",
            "",
            "**Recommendations:**",
            "- Cache computed features where possible",
            "- Optimize pandas operations (use vectorization)",
            "- Profile specific feature calculations",
            "- Consider pre-computing static features",
        ])
    elif bottleneck_name == "other":
        lines.extend([
            f"⚠️  **Primary Bottleneck: Other ({bottleneck_pct:.1f}%)**",
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
        "## Notes",
        "",
        "- This is a quick profile run on 1 month of data with a single worker.",
        "- Results are indicative but may vary for FULLYEAR runs.",
        "- For FULLYEAR profiling, run: `scripts/profile_sniper_entry_v10_1_flat_threshold018_2025.sh`",
        "",
        f"- **Log File:** `{log_file}`",
        "",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    log.info(f"✅ Performance report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize quick replay performance metrics from log file"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Log file containing [REPLAY_PERF_SUMMARY] line",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("reports/rl/entry_v10/ENTRY_V10_1_REPLAY_PERF_QUICK_THRESHOLD_0_18.md"),
        help="Path to save the performance report",
    )
    
    args = parser.parse_args()
    
    if not args.log_file.exists():
        log.error(f"Log file does not exist: {args.log_file}")
        return 1
    
    # Find performance summary
    log.info(f"Scanning log file: {args.log_file}")
    perf_summary = None
    
    try:
        with open(args.log_file, "r", encoding="utf-8") as f:
            for line in f:
                perf = parse_perf_summary_line(line)
                if perf:
                    perf_summary = perf
                    log.info("Found [REPLAY_PERF_SUMMARY]")
    except Exception as e:
        log.error(f"Failed to read log file: {e}")
        return 1
    
    if not perf_summary:
        log.error("No [REPLAY_PERF_SUMMARY] line found in log file")
        log.error("Make sure the replay completed and logged performance metrics")
        return 1
    
    # Generate report
    generate_report(perf_summary, args.output_report, args.log_file)
    
    # Print summary to console
    bottleneck_name, bottleneck_pct = identify_bottleneck(perf_summary)
    
    print("\n" + "=" * 80)
    print("QUICK PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Total Bars: {perf_summary['bars']:,}")
    print(f"Total Trades: {perf_summary['trades']:,}")
    print(f"Total Time: {perf_summary['total_time']:.1f}s ({perf_summary['total_time']/60:.1f} min)")
    print(f"Bars/sec: {perf_summary['bars_per_sec']:.2f}")
    print()
    print("Time Distribution:")
    print(f"  Feature Building: {perf_summary['feat_time']:.1f}s ({perf_summary['feat_pct']:.1f}%)")
    print(f"  Model Inference:  {perf_summary['model_time']:.1f}s ({perf_summary['model_pct']:.1f}%)")
    print(f"  Trade Journaling: {perf_summary['journal_time']:.1f}s ({perf_summary['journal_pct']:.1f}%)")
    print(f"  Other:            {perf_summary['other_time']:.1f}s ({perf_summary['other_pct']:.1f}%)")
    print()
    if bottleneck_pct:
        print(f"Primary Bottleneck: {bottleneck_name} ({bottleneck_pct:.1f}%)")
    else:
        print(f"Primary Bottleneck: {bottleneck_name}")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

