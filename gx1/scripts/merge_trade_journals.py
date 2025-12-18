#!/usr/bin/env python3
"""
Merge trade journals from parallel chunks or single-run layout.

Supports both:
- Single-run layout: <run_dir>/trade_journal/...
- Parallel layout: <run_dir>/parallel_chunks/chunk_*/trade_journal/...

Merges trade journals deterministically with deduplication on trade_id.
"""

import argparse
import csv
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Deterministic column order for merged index CSV
INDEX_COLUMNS = [
    "trade_id",
    "entry_time",
    "exit_time",
    "side",
    "exit_profile",
    "pnl_bps",
    "guardrail_applied",
    "range_edge_dist_atr",
    "router_decision",
    "exit_reason",
    "oanda_trade_id",
    "oanda_last_txn_id",
    "execution_status",
    "session",
    "vol_regime",
    "trend_regime",
    "atr_bps",
    "spread_bps",
    "range_pos",
    "distance_to_range",
    "router_version",
    "source_chunk",
    "trade_file",
]


def _to_iso_or_empty(x) -> str:
    """
    Best-effort normalization of a timestamp-like value to ISO string.
    Returns "" for None/NaN/unsupported types.
    """
    if x is None:
        return ""
    try:
        import pandas as pd  # Local import to avoid hard dependency at module import
    except Exception:
        return str(x)
    
    # If it's already a pandas Timestamp
    if isinstance(x, pd.Timestamp):
        try:
            return x.tz_convert("UTC").isoformat()
        except Exception:
            try:
                return x.tz_localize("UTC").isoformat()
            except Exception:
                return ""
    
    # If it's a string, try to parse as datetime
    if isinstance(x, str):
        x_strip = x.strip()
        if not x_strip:
            return ""
        try:
            ts = pd.to_datetime(x_strip, utc=True, errors="coerce")
            if ts is not None and pd.notna(ts):
                return ts.isoformat()
            return ""
        except Exception:
            return ""
    
    # For numbers (floats/ints) we do NOT try to interpret as epoch to avoid 1970-issues
    # Just return empty to force them to the bottom and rely on trade_id for stability.
    if isinstance(x, (int, float)):
        return ""
    
    # Fallback: string representation (as last resort)
    return str(x)


def _sort_key(trade: Dict) -> tuple:
    """
    Deterministic, type-safe sort key for merged_trades.
    Sorts primarily by entry timestamp (ISO string), secondarily by trade_id.
    """
    entry_snap = trade.get("entry_snapshot") or {}
    ts_raw = entry_snap.get("timestamp") or entry_snap.get("entry_time") or ""
    ts_iso = _to_iso_or_empty(ts_raw)
    trade_id = str(trade.get("trade_id", ""))
    return (ts_iso, trade_id)


def discover_trade_journal_roots(run_dir: Path, allow_legacy: bool = False) -> List[Tuple[Path, str]]:
    """
    Discover all trade journal roots in a run directory.
    
    By default, only scans:
    - <run_dir>/trade_journal/
    - <run_dir>/parallel_chunks/chunk_*/trade_journal/
    
    Legacy location (gx1/live/trade_journal/) is only used if allow_legacy=True.
    
    Args:
        run_dir: Run directory to scan
        allow_legacy: If True, also check gx1/live/trade_journal/ (default: False)
    
    Returns:
        List of (journal_root_path, source_label) tuples.
        source_label is "root" or "chunk_0", "chunk_1", etc.
    """
    roots = []
    
    # Check for root trade journal
    root_journal = run_dir / "trade_journal"
    if root_journal.exists() and ((root_journal / "trade_journal_index.csv").exists() or (root_journal / "trades").exists()):
        roots.append((root_journal, "root"))
        log.info("Found root trade journal: %s", root_journal)
    
    # Check for parallel chunks
    parallel_chunks_dir = run_dir / "parallel_chunks"
    if parallel_chunks_dir.exists():
        chunk_dirs = sorted(parallel_chunks_dir.glob("chunk_*"))
        for chunk_dir in chunk_dirs:
            chunk_journal = chunk_dir / "trade_journal"
            if chunk_journal.exists():
                chunk_id = chunk_dir.name  # e.g., "chunk_0"
                roots.append((chunk_journal, chunk_id))
                log.info("Found chunk trade journal: %s (%s)", chunk_journal, chunk_id)
            else:
                # Also check if trades are written directly to chunk_dir (legacy layout within run_dir)
                chunk_trades_dir = chunk_dir / "trades"
                if chunk_trades_dir.exists() and list(chunk_trades_dir.glob("*.json")):
                    # Create a virtual journal root
                    roots.append((chunk_dir, chunk_id))
                    log.info("Found chunk trades directory: %s (%s)", chunk_trades_dir, chunk_id)
    
    # Legacy fallback: ONLY if allow_legacy=True AND no roots found
    if not roots and allow_legacy:
        legacy_journal = Path("gx1/live/trade_journal")
        if legacy_journal.exists():
            run_id = run_dir.name
            log.info("Checking legacy trade journal location: %s", legacy_journal)
            if (legacy_journal / "trades").exists() and list((legacy_journal / "trades").glob("*.json")):
                roots.append((legacy_journal, "legacy"))
                log.warning("Found legacy trade journal at %s (may contain trades from other runs)", legacy_journal)
    elif not roots:
        log.warning("No trade journals found in run_dir. Legacy location (gx1/live/trade_journal/) not checked (use --allow-legacy to enable)")
    
    return roots


def load_trade_from_json(trade_file: Path) -> Optional[Dict]:
    """Load a single trade JSON file."""
    try:
        with open(trade_file, "r") as f:
            return json.load(f)
    except Exception as e:
        log.warning("Failed to load trade file %s: %s", trade_file, e)
        return None


def load_trades_from_journal_root(journal_root: Path, source_label: str) -> List[Dict]:
    """
    Load all trades from a journal root.
    
    Tries to use trade_journal_index.csv if available, otherwise scans trades/*.json.
    Supports both standard layout (journal_root/trades/) and legacy (journal_root is trades dir).
    """
    trades = []
    
    # Determine trades directory
    if (journal_root / "trades").exists():
        trades_dir = journal_root / "trades"
    elif journal_root.name == "trades" or list(journal_root.glob("*.json")):
        # Legacy: journal_root is the trades directory itself
        trades_dir = journal_root
    else:
        trades_dir = journal_root / "trades"
    
    # Prefer JSON trades as the single source of truth; only fall back to index
    # if no JSON files exist.
    if trades_dir.exists() and list(trades_dir.glob("*.json")):
        log.info("Scanning JSON files in %s (%s)", trades_dir, source_label)
        for trade_file in sorted(trades_dir.glob("*.json")):
            trade_data = load_trade_from_json(trade_file)
            if trade_data:
                trade_data["_source_chunk"] = source_label
                trades.append(trade_data)
    else:
        # Fallback: try to use index CSV if available (legacy / recovery path)
        index_path = journal_root / "trade_journal_index.csv"
        if not index_path.exists() and journal_root.parent:
            # Try parent directory for index
            index_path = journal_root.parent / "trade_journal_index.csv"
        
        if index_path.exists():
            try:
                index_df = pd.read_csv(index_path)
                log.info("Loading %d trades from index CSV (%s)", len(index_df), source_label)
                
                for _, row in index_df.iterrows():
                    trade_id = row.get("trade_id", "")
                    trade_data = {
                        "trade_id": trade_id,
                        "entry_snapshot": {
                            "timestamp": row.get("entry_time", ""),
                            "entry_time": row.get("entry_time", ""),
                            "reason": row.get("entry_reason", ""),
                            "session": row.get("session", ""),
                            "vol_regime": row.get("vol_regime", ""),
                            "trend_regime": row.get("trend_regime", ""),
                        },
                        "feature_context": {
                            "atr": {"atr_bps": row.get("atr_bps", None)},
                            "spread": {"spread_pct": row.get("spread_bps", None) / 10000.0 if pd.notna(row.get("spread_bps")) else None},
                            "range": {
                                "range_pos": row.get("range_pos", None),
                                "distance_to_range": row.get("distance_to_range", None),
                            },
                        } if any(pd.notna(row.get(k)) for k in ["atr_bps", "spread_bps", "range_pos", "distance_to_range"]) else None,
                        "exit_summary": {
                            "exit_time": row.get("exit_time", ""),
                            "exit_reason": row.get("exit_reason", ""),
                            "realized_pnl_bps": row.get("pnl_bps", None),
                        } if pd.notna(row.get("exit_reason")) and row.get("exit_reason") != "" else None,
                    }
                    trade_data["_source_chunk"] = source_label
                    trades.append(trade_data)
            except Exception as e:
                log.warning("Failed to load index CSV %s: %s", index_path, e)

    # Final fallback: if still nothing, scan trades/*.json (best effort)
    if len(trades) == 0 and trades_dir.exists():
        log.info("Scanning JSON files in %s (%s)", trades_dir, source_label)
        for trade_file in sorted(trades_dir.glob("*.json")):
            trade_data = load_trade_from_json(trade_file)
            if trade_data:
                trade_data["_source_chunk"] = source_label
                trades.append(trade_data)
    
    log.info("Loaded %d trades from %s", len(trades), source_label)
    return trades


def _g(d: Optional[Dict], *path, default=None):
    """
    Safe getter for nested dicts.
    Returns default if any level is missing or None.
    """
    cur = d
    for p in path:
        if cur is None:
            return default
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


BAD_TIME_STRINGS = {"", "none", "null", "nan", "false", "0"}


def _is_valid_time(x) -> bool:
    """
    Strict validity check for time-like values.
    Used consistently in both row-building and verification.
    """
    if x is None:
        return False
    if isinstance(x, str):
        s = x.strip()
        if s.lower() in BAD_TIME_STRINGS:
            return False
        return True
    if isinstance(x, (int, float)):
        if x != x:  # NaN
            return False
        return x > 0
    return False


def _norm_reason(x: Optional[str]) -> Optional[str]:
    """
    Normalize exit_reason for robust comparisons.
    - Strip whitespace
    - Uppercase
    - Convert non-strings via str()
    """
    if x is None:
        return None
    if not isinstance(x, str):
        x = str(x)
    return x.strip().upper()


def build_index_row_from_trade_json(
    trade: Dict,
    source_chunk: str,
    trade_file: str,
    default_exit_time: Optional[str] = None,
    force_eof_baseline_q1: bool = False,
) -> Dict:
    """
    Build a single index row from a trade JSON document.

    Uses TradeJournal JSON schema as the single source of truth:
    - entry_snapshot
    - feature_context
    - router_explainability
    - exit_summary
    - execution_events
    """
    entry = _g(trade, "entry_snapshot", default={}) or {}
    exit_ = _g(trade, "exit_summary", default={}) or {}
    feature_ctx = _g(trade, "feature_context", default={}) or {}
    router = _g(trade, "router_explainability", default={}) or {}
    execution_events = _g(trade, "execution_events", default=[]) or []

    # Trade identifiers and core timestamps (canonical fields only)
    trade_id = _g(trade, "trade_id", default=None) or _g(entry, "trade_id", default=None)
    # Entry: prefer entry_snapshot.entry_time, fallback to trade.entry_time
    entry_time = _g(entry, "entry_time", default=None) or _g(trade, "entry_time", default=None)
    # Exit: prefer exit_summary.exit_time only (after hygiene has populated it)
    exit_time = _g(exit_, "exit_time", default=None)

    exit_reason = _g(exit_, "exit_reason", default=None) or _g(trade, "exit_reason", default=None)

    # Journal hygiene: infer exit_time from execution_events if missing but exit_reason exists.
    if exit_ and exit_reason and not exit_time:
        closed_ts = None
        # Prefer explicit close events
        for ev in reversed(execution_events):
            ev_type = _g(ev, "event_type", default="")
            if ev_type in {"TRADE_CLOSED_OANDA", "TRADE_CLOSED", "POSITION_CLOSED"}:
                closed_ts = _g(ev, "timestamp", default=None) or _g(ev, "time", default=None)
                if closed_ts:
                    break
        # Fallback: last event timestamp
        if not closed_ts:
            for ev in reversed(execution_events):
                closed_ts = _g(ev, "timestamp", default=None) or _g(ev, "time", default=None)
                if closed_ts:
                    break
        if closed_ts:
            exit_time = closed_ts
            exit_["exit_time"] = closed_ts
            trade["exit_summary"] = exit_
        # REPLAY_EOF-specific fallback: if still missing, use default_exit_time (e.g., quarter/chunk end)
        elif exit_reason == "REPLAY_EOF" and default_exit_time:
            exit_time = default_exit_time
            exit_["exit_time"] = default_exit_time
            trade["exit_summary"] = exit_

    # Forced Q1 baseline fallback: if we still cannot infer exit_time for REPLAY_EOF,
    # close at entry_time (duration=0, PnL unchanged, trade counted as closed).
    if (
        force_eof_baseline_q1
        and exit_reason == "REPLAY_EOF"
        and not _is_valid_time(exit_time)
        and _is_valid_time(entry_time)
    ):
        old_exit_time = exit_time
        exit_time = entry_time
        exit_["exit_time"] = exit_time
        exit_["exit_time_source"] = "FORCED_EOF_BASELINE_Q1"
        trade["exit_summary"] = exit_
        log.info(
            "FORCED_EOF_BASELINE_Q1 trade_id=%s file=%s source_chunk=%s entry_time=%s old_exit_time=%r new_exit_time=%s",
            trade_id,
            trade_file,
            source_chunk,
            entry_time,
            old_exit_time,
            exit_time,
        )

    # If we are in the forced EOF baseline Q1 path and still have missing entry_time
    # while exit_time is valid, set entry_time := exit_time for index purposes
    # (duration = 0, trade counted as closed).
    if force_eof_baseline_q1 and exit_reason == "REPLAY_EOF":
        if _is_valid_time(exit_time) and not _is_valid_time(entry_time):
            log.info(
                "FORCED_EOF_BASELINE_Q1_ENTRY_FIX trade_id=%s file=%s source_chunk=%s old_entry_time=%r new_entry_time=%s",
                trade_id,
                trade_file,
                source_chunk,
                entry_time,
                exit_time,
            )
            entry_time = exit_time

    # Side / direction
    side = _g(entry, "side", default=None) or _g(entry, "direction", default=None)

    # PnL in basis points
    pnl_bps = _g(exit_, "realized_pnl_bps", default=None)
    if pnl_bps is None:
        pnl_bps = _g(exit_, "pnl_bps", default=None)
    if pnl_bps is None:
        pnl_bps = _g(trade, "pnl_bps", default=None)

    # Feature context (ATR, spread, range)
    atr_bps = _g(feature_ctx, "atr", "atr_bps", default=None)
    spread_pct = _g(feature_ctx, "spread", "spread_pct", default=None)
    spread_bps = None
    if spread_pct is not None:
        try:
            spread_bps = float(spread_pct) * 10000.0
        except Exception:
            spread_bps = None
    range_pos = _g(feature_ctx, "range", "range_pos", default=None)
    distance_to_range = _g(feature_ctx, "range", "distance_to_range", default=None)

    # Router / guardrail metadata
    guardrail_applied = _g(router, "guardrail_applied", default=None)
    range_edge_dist_atr = _g(router, "range_edge_dist_atr", default=None)
    if range_edge_dist_atr is None:
        range_edge_dist_atr = _g(feature_ctx, "range", "range_edge_dist_atr", default=None)
    router_decision = _g(router, "router_raw_decision", default=None)
    exit_profile = _g(router, "final_exit_profile", default=None) or _g(exit_, "exit_profile", default=None)
    router_version = _g(router, "router_version", default=None)

    # Execution metadata (OANDA IDs + execution status)
    oanda_trade_id = None
    oanda_last_txn_id = None
    execution_status = "UNKNOWN"
    for ev in execution_events:
        ev_type = _g(ev, "event_type", default="")
        if ev_type == "ORDER_FILLED":
            oanda_trade_id = _g(ev, "oanda_trade_id", default=oanda_trade_id) or oanda_trade_id
            oanda_last_txn_id = _g(ev, "oanda_transaction_id", default=oanda_last_txn_id) or oanda_last_txn_id
            execution_status = "OK"
        elif ev_type == "ORDER_REJECTED":
            execution_status = "REJECTED"
        elif ev_type == "TRADE_CLOSED_OANDA":
            oanda_last_txn_id = _g(ev, "oanda_transaction_id", default=oanda_last_txn_id) or oanda_last_txn_id

    row = {
        "trade_id": trade_id,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "side": side,
        "exit_profile": exit_profile,
        "pnl_bps": pnl_bps,
        "guardrail_applied": guardrail_applied,
        "range_edge_dist_atr": range_edge_dist_atr,
        "router_decision": router_decision,
        "exit_reason": exit_reason,
        "oanda_trade_id": oanda_trade_id,
        "oanda_last_txn_id": oanda_last_txn_id,
        "execution_status": execution_status,
        "session": _g(entry, "session", default=None) or _g(trade, "session", default=None),
        "vol_regime": _g(entry, "vol_regime", default=None),
        "trend_regime": _g(entry, "trend_regime", default=None),
        "atr_bps": atr_bps,
        "spread_bps": spread_bps,
        "range_pos": range_pos,
        "distance_to_range": distance_to_range,
        "router_version": router_version,
        "source_chunk": source_chunk,
        "trade_file": trade_file,
    }

    return row


def write_merged_index_csv(path: Path, rows: List[Dict]) -> None:
    """
    Write merged index CSV with deterministic column order.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=INDEX_COLUMNS,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for r in rows:
            for k in INDEX_COLUMNS:
                r.setdefault(k, None)
            writer.writerow(r)


def verify_merged_index(rows: List[Dict], max_bad_frac: float = 0.0) -> None:
    """
    Fail-closed sanity check for merged index rows.

    Rules:
    - trade_id and entry_time must be present
    - if exit_time is present, exit_reason must either start with RULE_ or be REPLAY_EOF
    """
    bad = 0
    bad_rows = []
    for r in rows:
        trade_id = r.get("trade_id")
        entry_time = r.get("entry_time")
        exit_time = r.get("exit_time")
        exit_reason_raw = r.get("exit_reason")
        exit_reason_norm = _norm_reason(exit_reason_raw)

        if not trade_id or not _is_valid_time(entry_time):
            bad += 1
            bad_rows.append((trade_id, r.get("trade_file"), exit_reason_raw))
            continue
        # Trade is considered "complete" if exit_time exists
        if not _is_valid_time(exit_time):
            bad += 1
            bad_rows.append((trade_id, r.get("trade_file"), exit_reason_raw))
            continue
        er = exit_reason_norm or ""
        if er and not (er.startswith("RULE_") or er == "REPLAY_EOF"):
            bad += 1
            bad_rows.append(
                (
                    trade_id,
                    r.get("trade_file"),
                    exit_reason_raw,
                    exit_time,
                )
            )

    if bad_rows:
        log.error("Bad merged index rows (first 10): %s", bad_rows[:10])

    frac = bad / max(1, len(rows))
    if frac > max_bad_frac:
        raise RuntimeError(f"Merged index corrupt: bad_rows={bad}/{len(rows)} ({frac:.1%})")



def merge_trade_journals(run_dir: Path, output_dir: Optional[Path] = None, allow_legacy: bool = False, expected_quarter: Optional[str] = None) -> Dict:
    """
    Merge trade journals from all discovered roots.
    
    Args:
        run_dir: Run directory to merge journals from
        output_dir: Output directory for merged journal (default: run_dir/trade_journal)
        allow_legacy: If True, also check gx1/live/trade_journal/ (default: False)
        expected_quarter: Expected quarter (Q1/Q2/Q3/Q4) for sanity check (default: None)
    
    Returns:
        Dictionary with merge statistics.
    """
    if output_dir is None:
        output_dir = run_dir / "trade_journal"
    
    # Discover all journal roots (NO legacy by default)
    journal_roots = discover_trade_journal_roots(run_dir, allow_legacy=allow_legacy)
    if not journal_roots:
        log.error("No trade journals found in %s", run_dir)
        if not allow_legacy:
            log.error("Hint: Use --allow-legacy to check gx1/live/trade_journal/ (not recommended)")
        return {"error": "No trade journals found"}
    
    # Load all trades
    all_trades_raw = []
    trades_per_chunk = {}
    
    for journal_root, source_label in journal_roots:
        trades = load_trades_from_journal_root(journal_root, source_label)
        all_trades_raw.extend(trades)
        trades_per_chunk[source_label] = len(trades)
    
    log.info("Total trades loaded: %d", len(all_trades_raw))
    
    # Deduplicate by trade_id
    trades_by_id: Dict[str, Dict] = {}
    collisions = []
    
    for trade in all_trades_raw:
        trade_id = trade.get("trade_id")
        if not trade_id:
            log.warning("Trade missing trade_id, skipping: %s", trade.get("_source_chunk", "unknown"))
            continue
        
        if trade_id in trades_by_id:
            # Collision: prefer the one with exit_reason, then latest exit_time
            existing = trades_by_id[trade_id]
            existing_exit = existing.get("exit_summary") or {}
            existing_entry = existing.get("entry_snapshot") or {}
            existing_exit_reason = existing_exit.get("exit_reason", "")
            existing_exit_time = existing_exit.get("exit_time") or existing_exit.get("timestamp", "")
            
            new_exit = trade.get("exit_summary") or {}
            new_entry = trade.get("entry_snapshot") or {}
            new_exit_reason = new_exit.get("exit_reason", "")
            new_exit_time = new_exit.get("exit_time") or new_exit.get("timestamp", "")
            
            # Prefer record with exit_reason
            if new_exit_reason and not existing_exit_reason:
                # New has exit, existing doesn't -> prefer new
                should_replace = True
            elif existing_exit_reason and not new_exit_reason:
                # Existing has exit, new doesn't -> keep existing
                should_replace = False
            elif new_exit_reason and existing_exit_reason:
                # Both have exit -> prefer later exit_time
                should_replace = new_exit_time > existing_exit_time
            else:
                # Neither has exit -> prefer later entry_time
                existing_entry_time = existing_entry.get("timestamp") or existing_entry.get("entry_time", "")
                new_entry_time = new_entry.get("timestamp") or new_entry.get("entry_time", "")
                should_replace = new_entry_time > existing_entry_time
            
            # Prepare timestamps for collision logging
            existing_ts = existing_exit_time or existing_entry.get("timestamp", "")
            new_ts = new_exit_time or new_entry.get("timestamp", "")
            
            if should_replace:
                collisions.append({
                    "trade_id": trade_id,
                    "existing_source": existing.get("_source_chunk", "unknown"),
                    "new_source": trade.get("_source_chunk", "unknown"),
                    "existing_ts": existing_ts,
                    "new_ts": new_ts,
                    "existing_exit_reason": existing_exit_reason,
                    "new_exit_reason": new_exit_reason,
                })
                trades_by_id[trade_id] = trade
            else:
                collisions.append({
                    "trade_id": trade_id,
                    "existing_source": existing.get("_source_chunk", "unknown"),
                    "new_source": trade.get("_source_chunk", "unknown"),
                    "existing_ts": existing_ts,
                    "new_ts": new_ts,
                    "existing_exit_reason": existing_exit_reason,
                    "new_exit_reason": new_exit_reason,
                    "kept": "existing",
                })
        else:
            trades_by_id[trade_id] = trade
    
    # Sort merged trades by entry timestamp (robust, type-safe)
    merged_trades = list(trades_by_id.values())
    try:
        merged_trades.sort(key=_sort_key)
    except Exception as e:
        log.warning("Failed to sort merged trades by timestamp: %s", e)
    
    log.info("After deduplication: %d unique trades", len(merged_trades))
    log.info("Collisions: %d", len(collisions))
    
    # Create output directory
    output_trades_dir = output_dir / "trades"
    output_trades_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy/symlink trade JSON files
    for i, trade in enumerate(merged_trades):
        trade_id = trade.get("trade_id", f"trade_{i+1}")
        source_chunk = trade.pop("_source_chunk", "unknown")
        
        # Find source file
        source_journal_root = None
        for journal_root, label in journal_roots:
            if label == source_chunk:
                source_journal_root = journal_root
                break
        
        if source_journal_root:
            source_trades_dir = source_journal_root / "trades"
            # Try to find source file
            source_file = None
            for pattern in [f"{trade_id}.json", f"trade_{i+1:06d}.json"]:
                candidate = source_trades_dir / pattern
                if candidate.exists():
                    source_file = candidate
                    break
            
            if not source_file:
                # Fallback: find any file with matching trade_id
                for f in source_trades_dir.glob("*.json"):
                    data = load_trade_from_json(f)
                    if data and data.get("trade_id") == trade_id:
                        source_file = f
                        break
            
            if source_file:
                # Copy file (not symlink for portability)
                dest_file = output_trades_dir / f"trade_{i+1:06d}.json"
                # Skip if source and dest are the same file
                if source_file.resolve() != dest_file.resolve():
                    shutil.copy2(source_file, dest_file)
                # else: file already in place, skip copy
            else:
                # Write trade data directly
                dest_file = output_trades_dir / f"trade_{i+1:06d}.json"
                with open(dest_file, "w") as f:
                    json.dump(trade, f, indent=2, default=str)
        else:
            # Write trade data directly
            dest_file = output_trades_dir / f"trade_{i+1:06d}.json"
            with open(dest_file, "w") as f:
                json.dump(trade, f, indent=2, default=str)
    
    # Build trade_id -> source_chunk mapping (from _source_chunk we added earlier)
    # Note: We already have _source_chunk in trades, but we removed it. Re-add from original data.
    trade_id_to_chunk = {}
    for trade_raw in all_trades_raw:
        trade_id = trade_raw.get("trade_id")
        source_chunk = trade_raw.get("_source_chunk", "unknown")
        if trade_id and trade_id not in trade_id_to_chunk:
            trade_id_to_chunk[trade_id] = source_chunk
    
    # For Q1 baseline, derive a per-chunk default exit_time for REPLAY_EOF trades
    # based on the maximum entry_time seen in that chunk (chunk_end_time).
    is_q1_baseline = expected_quarter == "Q1" and "baseline" in run_dir.name
    chunk_end_time_map: Dict[str, str] = {}
    if is_q1_baseline:
        for trade_raw in all_trades_raw:
            source_chunk = trade_raw.get("_source_chunk", "root")
            entry_snap = trade_raw.get("entry_snapshot") or {}
            ts_raw = entry_snap.get("timestamp") or entry_snap.get("entry_time")
            ts_iso = _to_iso_or_empty(ts_raw)
            if not ts_iso:
                continue
            prev = chunk_end_time_map.get(source_chunk)
            if not prev or ts_iso > prev:
                chunk_end_time_map[source_chunk] = ts_iso

    # Create merged index rows deterministically from trade JSON only
    index_rows: List[Dict] = []
    for i, trade in enumerate(merged_trades):
        trade_id = trade.get("trade_id", f"trade_{i+1}")
        source_chunk = trade_id_to_chunk.get(trade_id, "root")
        # Only apply REPLAY_EOF chunk-end heuristics for Q1 baseline
        default_exit_time = chunk_end_time_map.get(source_chunk) if is_q1_baseline else None
        trade_file = f"trade_{i+1:06d}.json"
        row = build_index_row_from_trade_json(
            trade,
            source_chunk=source_chunk,
            trade_file=trade_file,
            default_exit_time=default_exit_time,
            force_eof_baseline_q1=is_q1_baseline,
        )
        index_rows.append(row)

    # Debug preview: log any REPLAY_EOF rows with invalid times before verify
    bad_preview: List[Dict] = []
    for r in index_rows:
        er_norm = _norm_reason(r.get("exit_reason"))
        if er_norm == "REPLAY_EOF":
            if not _is_valid_time(r.get("entry_time")) or not _is_valid_time(r.get("exit_time")):
                bad_preview.append(r)
    if bad_preview:
        log.error("Index bad_preview count=%d", len(bad_preview))
        for r in bad_preview[:10]:
            try:
                trade_path = output_trades_dir / r.get("trade_file", "")
                if trade_path.exists():
                    with open(trade_path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    entry = d.get("entry_snapshot") or {}
                    ex = d.get("exit_summary") or {}
                    log.error(
                        "BADROW file=%s trade_id=%s row.entry=%r row.exit=%r json.entry=%r json.exit=%r json.exit_src=%r",
                        r.get("trade_file"),
                        r.get("trade_id"),
                        r.get("entry_time"),
                        r.get("exit_time"),
                        entry.get("entry_time"),
                        ex.get("exit_time"),
                        ex.get("exit_time_source"),
                    )
                else:
                    log.error(
                        "BADROW file=%s trade_id=%s row.entry=%r row.exit=%r (JSON file missing)",
                        r.get("trade_file"),
                        r.get("trade_id"),
                        r.get("entry_time"),
                        r.get("exit_time"),
                    )
            except Exception as e:
                log.error("Failed to debug BADROW for file=%s: %s", r.get("trade_file"), e)

    # Fail-closed sanity check on merged index before writing to disk.
    # Require 0 bad rows; any corruption must be fixed explicitly.
    try:
        verify_merged_index(index_rows, max_bad_frac=0.0)
    except RuntimeError as e:
        log.error("❌ Merged index verification failed: %s", e)
        return {
            "error": "Merged index verification failed",
            "details": str(e),
        }

    index_path = output_dir / "trade_journal_index.csv"
    write_merged_index_csv(index_path, index_rows)
    log.info("Created merged index: %s", index_path)

    # Self-check: how many empty entry_time values?
    index_df = pd.DataFrame(index_rows)
    if "entry_time" in index_df.columns:
        total_rows = len(index_df)
        empty_entry_times = index_df["entry_time"].isna().sum() + (index_df["entry_time"] == "").sum()
        if total_rows > 0:
            empty_pct = empty_entry_times / total_rows
            if empty_pct > 0.005:  # >0.5%
                log.warning(
                    "High fraction of empty entry_time in merged index: %d/%d (%.2f%%)",
                    empty_entry_times, total_rows, empty_pct * 100.0
                )
    
    # Generate merge report
    merge_report = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "total_trades_merged": len(merged_trades),
        "trades_per_chunk": trades_per_chunk,
        "total_trades_before_dedup": len(all_trades_raw),
        "collisions": len(collisions),
        "collision_details": collisions[:10],  # Top 10 collisions
        "journal_roots_found": [{"path": str(r), "label": l} for r, l in journal_roots],
    }
    
    report_path = output_dir / "merge_report.json"
    with open(report_path, "w") as f:
        json.dump(merge_report, f, indent=2, default=str)
    log.info("Created merge report: %s", report_path)
    
    # Validation hooks
    total_chunk_trades = sum(trades_per_chunk.values())
    if len(merged_trades) < total_chunk_trades * 0.95:
        log.warning(
            "⚠️  Merged trades (%d) < 95%% of chunk total (%d). Possible data loss!",
            len(merged_trades), total_chunk_trades
        )
    
    if collisions:
        log.warning("⚠️  Found %d trade_id collisions", len(collisions))
        if len(collisions) <= 10:
            log.info("Collision details:")
            for coll in collisions:
                log.info("  trade_id=%s: kept from %s (ts=%s), dropped from %s (ts=%s)",
                         coll["trade_id"], coll.get("existing_source", "unknown"), coll["existing_ts"],
                         coll.get("new_source", "unknown"), coll["new_ts"])
    
    # Sanity check: Verify entry_time min/max matches expected quarter
    if expected_quarter and len(index_df) > 0:
        try:
            index_df["entry_time_parsed"] = pd.to_datetime(index_df["entry_time"], utc=True, errors="coerce")
            valid_times = index_df["entry_time_parsed"].dropna()
            if len(valid_times) > 0:
                min_time = valid_times.min()
                max_time = valid_times.max()
                
                # Expected date ranges (UTC, exclusive end)
                quarter_ranges = {
                    "Q1": ("2025-01-01", "2025-04-01"),
                    "Q2": ("2025-04-01", "2025-07-01"),
                    "Q3": ("2025-07-01", "2025-10-01"),
                    "Q4": ("2025-10-01", "2026-01-01"),
                }
                
                if expected_quarter in quarter_ranges:
                    expected_start, expected_end = quarter_ranges[expected_quarter]
                    expected_start_ts = pd.Timestamp(expected_start, tz="UTC")
                    expected_end_ts = pd.Timestamp(expected_end, tz="UTC")
                    
                    # Check if min/max are within expected range (with small tolerance for filtering)
                    if min_time < expected_start_ts - pd.Timedelta(days=1) or max_time >= expected_end_ts + pd.Timedelta(days=1):
                        log.error(
                            "❌ SANITY CHECK FAILED: Entry times don't match expected quarter %s",
                            expected_quarter
                        )
                        log.error(
                            "   Expected range: %s to %s",
                            expected_start_ts, expected_end_ts
                        )
                        log.error(
                            "   Actual range: %s to %s",
                            min_time, max_time
                        )
                        log.error(
                            "   This suggests trades are from wrong quarter or legacy location!"
                        )
                        return {
                            "error": "Sanity check failed: entry times don't match expected quarter",
                            "expected_quarter": expected_quarter,
                            "expected_range": (expected_start_ts.isoformat(), expected_end_ts.isoformat()),
                            "actual_range": (min_time.isoformat(), max_time.isoformat()),
                        }
                    else:
                        log.info(
                            "✅ Sanity check passed: entry times match quarter %s (min=%s, max=%s)",
                            expected_quarter, min_time, max_time
                        )
        except Exception as e:
            log.warning("Failed to perform sanity check: %s", e)
    
    return merge_report


def main():
    parser = argparse.ArgumentParser(description="Merge trade journals from parallel chunks or single-run layout")
    parser.add_argument("run_dir", type=Path, help="Run directory containing trade journals")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: <run_dir>/trade_journal)")
    parser.add_argument("--allow-legacy", action="store_true", help="Allow fallback to gx1/live/trade_journal/ (default: False, not recommended)")
    parser.add_argument("--expected-quarter", type=str, choices=["Q1", "Q2", "Q3", "Q4"], default=None, help="Expected quarter for sanity check (Q1/Q2/Q3/Q4)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not args.run_dir.exists():
        log.error("Run directory does not exist: %s", args.run_dir)
        return 1
    
    log.info("Merging trade journals from: %s", args.run_dir)
    if args.expected_quarter:
        log.info("Sanity check enabled: expecting quarter %s", args.expected_quarter)
    
    merge_report = merge_trade_journals(
        args.run_dir, 
        args.output_dir, 
        allow_legacy=args.allow_legacy,
        expected_quarter=args.expected_quarter
    )
    
    if "error" in merge_report:
        return 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("Merge Summary")
    print("=" * 60)
    print(f"Total trades merged: {merge_report['total_trades_merged']}")
    print(f"Trades before deduplication: {merge_report['total_trades_before_dedup']}")
    print(f"Collisions: {merge_report['collisions']}")
    print("\nTrades per chunk:")
    for chunk, count in sorted(merge_report["trades_per_chunk"].items()):
        print(f"  {chunk}: {count} trades")
    
    if merge_report["collisions"] > 0:
        print(f"\n⚠️  Top collisions:")
        for coll in merge_report["collision_details"][:5]:
            print(f"  trade_id={coll['trade_id']}: kept from {coll.get('existing_source', 'unknown')}")
    
    print(f"\n✅ Merged journal: {args.run_dir / 'trade_journal'}")
    print(f"✅ Index CSV: {args.run_dir / 'trade_journal' / 'trade_journal_index.csv'}")
    print(f"✅ Merge report: {args.run_dir / 'trade_journal' / 'merge_report.json'}")
    
    return 0


if __name__ == "__main__":
    exit(main())

