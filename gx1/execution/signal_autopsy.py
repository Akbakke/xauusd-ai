"""
Signal Autopsy: Aggregate ENTRY_SIGNAL_TRACE.jsonl into SIGNAL_AUTOPSY_SUMMARY.json.

Reads chunk_0/ENTRY_SIGNAL_TRACE.jsonl (streaming), collects n_signals,
n_attempted, n_taken, block_code_counts, first_events. Writes deterministic JSON.

TRUTH/SMOKE fail-fast:
- n_entry_signals > 0 but trace has 0 lines → RuntimeError
- Any line missing final_block_code → RuntimeError
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger(__name__)


def write_signal_autopsy_summary(
    chunk_dir: Path,
    run_id: str,
    n_entry_signals: int,
    *,
    first_n_events: int = 20,
    is_truth_or_smoke: bool = False,
) -> Dict[str, Any]:
    """
    Read ENTRY_SIGNAL_TRACE.jsonl, aggregate, write SIGNAL_AUTOPSY_SUMMARY.json.

    Args:
        chunk_dir: chunk_0 directory
        run_id: run identifier
        n_entry_signals: expected from metrics/killchain (for fail-fast)
        first_n_events: how many first events to include
        is_truth_or_smoke: if True, enforce fail-fast rules

    Returns:
        summary dict with n_signals, n_attempted, n_taken, block_code_counts, first_events
    """
    trace_path = Path(chunk_dir) / "ENTRY_SIGNAL_TRACE.jsonl"
    out_path = Path(chunk_dir) / "SIGNAL_AUTOPSY_SUMMARY.json"

    n_signals = 0
    n_attempted = 0
    n_taken = 0
    block_code_counts: Counter[str] = Counter()
    first_events: List[Dict[str, Any]] = []

    if not trace_path.exists():
        if is_truth_or_smoke and n_entry_signals > 0:
            raise RuntimeError(
                f"[ENTRY_SIGNAL_TRACE_FAIL] n_entry_signals={n_entry_signals} > 0 but "
                f"ENTRY_SIGNAL_TRACE.jsonl is missing (empty/not written). "
                "Trace must be written when GX1_ENTRY_SIGNAL_TRACE=1."
            )
        summary = {
            "run_id": run_id,
            "n_signals": 0,
            "n_attempted": 0,
            "n_taken": 0,
            "block_code_counts": {},
            "first_events": [],
            "trace_exists": False,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
        return summary

    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("[SIGNAL_AUTOPSY] Invalid JSON line: %s", e)
                continue
            n_signals += 1
            if event.get("entry_attempted"):
                n_attempted += 1
            if event.get("entry_taken"):
                n_taken += 1
            fc = event.get("final_block_code")
            if fc is None or fc == "":
                if is_truth_or_smoke:
                    raise RuntimeError(
                        f"[ENTRY_SIGNAL_TRACE_FAIL] Line {n_signals} missing final_block_code. "
                        "Every trace line must have final_block_code."
                    )
                fc = "BLOCK_UNKNOWN"
            block_code_counts[fc] += 1
            if len(first_events) < first_n_events:
                first_events.append({
                    "time": event.get("time"),
                    "final_block_code": fc,
                })

    if is_truth_or_smoke and n_entry_signals > 0 and n_signals == 0:
        raise RuntimeError(
            f"[ENTRY_SIGNAL_TRACE_FAIL] n_entry_signals={n_entry_signals} > 0 but "
            "trace has 0 lines. Trace must capture every entry signal."
        )

    top_block = block_code_counts.most_common(1)
    top_block_code = top_block[0][0] if top_block else "NONE"
    top_block_count = top_block[0][1] if top_block else 0

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "n_signals": n_signals,
        "n_attempted": n_attempted,
        "n_taken": n_taken,
        "block_code_counts": dict(sorted(block_code_counts.items())),
        "top_block_code": top_block_code,
        "top_block_count": top_block_count,
        "first_events": first_events,
        "trace_exists": True,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True, ensure_ascii=False)
    log.info("[SIGNAL_AUTOPSY] Wrote %s (signals=%d attempted=%d taken=%d top=%s(%d))",
             out_path, n_signals, n_attempted, n_taken, top_block_code, top_block_count)
    return summary
