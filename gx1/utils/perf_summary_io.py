"""
Performance summary I/O utilities for replay.
Provides deterministic JSON/Markdown writing with atomic file operations.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON file atomically (write to .tmp, then rename).
    
    Parameters
    ----------
    path : Path
        Target JSON file path
    data : Dict[str, Any]
        Data to write (will be JSON serialized)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    tmp.replace(path)


def write_md(path: Path, md: str) -> None:
    """
    Write Markdown file atomically (write to .tmp, then rename).
    
    Parameters
    ----------
    path : Path
        Target Markdown file path
    md : str
        Markdown content to write
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(md)
    tmp.replace(path)


def format_md_summary(data: Dict[str, Any]) -> str:
    """
    Format performance summary data as Markdown.
    
    Parameters
    ----------
    data : Dict[str, Any]
        Performance summary data (same structure as JSON)
    
    Returns
    -------
    str
        Formatted Markdown string
    """
    lines = []
    lines.append("# Replay Performance Summary")
    lines.append("")
    
    # Basic info
    lines.append(f"**Start Time:** {data.get('start_time', 'N/A')}")
    lines.append(f"**End Time:** {data.get('end_time', 'N/A')}")
    
    duration_sec = data.get('duration_sec', 0.0)
    lines.append(f"**Duration:** {duration_sec:.1f} seconds ({duration_sec/60:.1f} minutes)")
    lines.append("")
    
    bars_processed = data.get('bars_processed', 0)
    bars_total = data.get('bars_total', 0)
    percent_done = (bars_processed / bars_total * 100.0) if bars_total > 0 else 0.0
    lines.append(f"**Bars Processed:** {bars_processed}/{bars_total} ({percent_done:.1f}%)")
    lines.append(f"**Completed:** {data.get('completed', False)}")
    
    if data.get('early_stop_reason'):
        lines.append(f"**Early Stop Reason:** {data.get('early_stop_reason')}")
    
    if data.get('last_timestamp_processed'):
        lines.append(f"**Last Timestamp Processed:** {data.get('last_timestamp_processed')}")
    
    lines.append(f"**Trades Total:** {data.get('trades_total', 0)}")
    lines.append(f"**Trades Open at End:** {data.get('trades_open_at_end', 0)}")
    lines.append("")
    
    # Chunk info (if present)
    if 'chunk_id' in data:
        lines.append(f"**Chunk ID:** {data.get('chunk_id')}")
        if 'window_start' in data:
            lines.append(f"**Window Start:** {data.get('window_start')}")
        if 'window_end' in data:
            lines.append(f"**Window End:** {data.get('window_end')}")
        lines.append("")
    
    # Runner perf metrics
    runner_perf = data.get('runner_perf_metrics', {})
    if runner_perf:
        lines.append("## Runner Performance Metrics")
        lines.append("")
        for key, value in runner_perf.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")
    
    # Scaling snapshot
    feat_time_sec = runner_perf.get('feat_time_sec', 0.0)
    if bars_total > 0:
        lines.append("## Scaling Snapshot")
        lines.append("")
        lines.append(f"- **Bars/second:** {bars_total / max(duration_sec, 0.001):.2f}")
        lines.append(f"- **Feature time per bar:** {(feat_time_sec / max(bars_total, 1)) * 1000:.4f} ms")
        lines.append(f"- **Total bars:** {bars_total:,}")
        lines.append(f"- **Duration:** {duration_sec:.1f}s ({duration_sec/60:.1f} min)")
        lines.append(f"- **Feature time:** {feat_time_sec:.1f}s")
        lines.append("")
    
    # Feature top blocks
    feature_top_blocks = data.get('feature_top_blocks', [])
    if feature_top_blocks:
        lines.append("## Feature Top Blocks (Top 15)")
        lines.append("")
        lines.append("*Breakdown of {:.1f}s total feature time*".format(feat_time_sec))
        lines.append("")
        lines.append("| Name | Total (sec) | Share (%) | Calls |")
        lines.append("|------|-------------|-----------|-------|")
        for block in feature_top_blocks[:15]:
            name = block.get('name', 'N/A')
            total_sec = block.get('total_sec', 0.0)
            count = block.get('count', 0)
            share_pct = block.get('share_of_feat_time_pct', 0.0)
            lines.append(f"| {name} | {total_sec:.4f} | {share_pct:.2f} | {count:,} |")
        lines.append("")
    
    # Top pandas ops
    top_pandas_ops = data.get('top_pandas_ops', [])
    if top_pandas_ops:
        lines.append("## Top Pandas Operations")
        lines.append("")
        lines.append("| Name | Total (sec) | Calls |")
        lines.append("|------|-------------|-------|")
        for op in top_pandas_ops[:10]:
            name = op.get('name', 'N/A')
            total_sec = op.get('total_sec', 0.0)
            count = op.get('count', 0)
            lines.append(f"| {name} | {total_sec:.4f} | {count:,} |")
        lines.append("")
    
    return "\n".join(lines)


