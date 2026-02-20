#!/usr/bin/env python3
"""
OHLC Schema Gate for GX1 (TRUTH/SMOKE).

Purpose:
- Verify that the data file (--data) contains OHLC columns before workers start.
- Fail early with a clear capsule if OHLC columns are missing.

Contract (when truth_or_smoke=True):
- Reads schema from data_path (not full load, just schema).
- Requires one of two valid OHLC column families:
  A) {open, high, low, close} (direct)
  B) {candles.open, candles.high, candles.low, candles.close} (prefixed)
- If not satisfied: write OHLC_SCHEMA_FATAL.json capsule and exit code 2.
- Always write OHLC_SCHEMA_AUDIT.md atomically to output_dir (or /tmp fallback).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

# --- Utility functions (copied from env_identity_gate for self-containment) ---
def _now_utc_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _safe_mkdir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False


def _fallback_dir() -> Path:
    root = Path(tempfile.gettempdir()) / "gx1_ohlc_schema_gate"
    if not _safe_mkdir(root):
        root = Path("/tmp")
    run_dir = root / f"run_{_now_utc_compact()}_{os.getpid()}"
    _safe_mkdir(run_dir)
    return run_dir


def _choose_report_dir(output_dir: Path) -> Path:
    try:
        if output_dir is not None and _safe_mkdir(output_dir):
            return output_dir
    except Exception:
        pass
    return _fallback_dir()


def _write_text_atomic(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
# ---------------------------------------------------------------------------


def _read_schema_columns(data_path: Path) -> Optional[List[str]]:
    """
    Read column names from parquet file schema (fast, no full load).
    Returns None if file cannot be read.
    """
    if not data_path.exists():
        return None
    
    try:
        if pq is not None:
            # Use pyarrow (fast, schema-only)
            parquet_file = pq.ParquetFile(data_path)
            columns = [field.name for field in parquet_file.schema]
            return columns
        else:
            # Fallback: read first row with pandas (slower but works)
            import pandas as pd
            df_sample = pd.read_parquet(data_path, nrows=1)
            return list(df_sample.columns)
    except Exception:
        return None


def _check_ohlc_columns(columns: List[str]) -> Dict[str, Any]:
    """
    Check if columns contain required OHLC fields.
    
    Returns dict with:
    - has_ohlc_direct: bool (open, high, low, close)
    - has_ohlc_candles: bool (candles.open, candles.high, etc.)
    - missing_fields: List[str] (which OHLC fields are missing)
    - score: int (10 if direct, 8 if candles, 0 if none)
    """
    columns_lower = [c.lower() for c in columns]
    columns_set = set(columns_lower)
    columns_original = {c.lower(): c for c in columns}  # Map lowercase -> original case
    
    # Direct OHLC columns (case-insensitive check)
    ohlc_direct_lower = {"open", "high", "low", "close"}
    ohlc_direct_found = {col for col in ohlc_direct_lower if col in columns_set}
    has_ohlc_direct = len(ohlc_direct_found) == 4
    
    # Candles-prefixed OHLC (case-insensitive)
    ohlc_candles_lower = {"candles.open", "candles.high", "candles.low", "candles.close"}
    ohlc_candles_found = {col for col in ohlc_candles_lower if col in columns_set}
    has_ohlc_candles = len(ohlc_candles_found) == 4
    
    # Find missing fields (for direct)
    missing_direct = ohlc_direct_lower - ohlc_direct_found
    missing_candles = ohlc_candles_lower - ohlc_candles_found
    
    # Score: 10 if direct, 8 if candles, 0 if none
    score = 0
    if has_ohlc_direct:
        score = 10
    elif has_ohlc_candles:
        score = 8
    
    # Get original case column names for missing fields
    missing_fields_direct = [columns_original.get(m, m) for m in missing_direct]
    missing_fields_candles = [columns_original.get(m, m) for m in missing_candles]
    
    return {
        "has_ohlc_direct": has_ohlc_direct,
        "has_ohlc_candles": has_ohlc_candles,
        "ohlc_direct_found": sorted([columns_original.get(c, c) for c in ohlc_direct_found]),
        "ohlc_candles_found": sorted([columns_original.get(c, c) for c in ohlc_candles_found]),
        "missing_fields_direct": sorted(missing_fields_direct),
        "missing_fields_candles": sorted(missing_fields_candles),
        "score": score,
    }


def _render_md(
    data_path: Path,
    columns: List[str],
    ohlc_check: Dict[str, Any],
    file_size: int,
) -> str:
    """Renders the OHLC_SCHEMA_AUDIT report in Markdown format."""
    status = "✅ PASS" if (ohlc_check["has_ohlc_direct"] or ohlc_check["has_ohlc_candles"]) else "❌ FAIL"
    
    lines = [
        f"# OHLC Schema Audit",
        f"",
        f"**Generated:** {datetime.utcnow().isoformat()}Z",
        f"**Data Path:** `{data_path}`",
        f"**File Size:** {file_size:,} bytes ({file_size/(1024*1024):.1f} MB)",
        f"**Total Columns:** {len(columns)}",
        f"",
        f"## Status",
        f"",
        f"**Result:** {status}",
        f"",
        f"## OHLC Column Check",
        f"",
        f"- **Has OHLC Direct (open/high/low/close):** {ohlc_check['has_ohlc_direct']}",
        f"- **Has OHLC Candles (candles.open/high/low/close):** {ohlc_check['has_ohlc_candles']}",
        f"- **Score:** {ohlc_check['score']}",
        f"",
    ]
    
    if ohlc_check["ohlc_direct_found"]:
        lines.append(f"### Direct OHLC Columns Found")
        for col in ohlc_check["ohlc_direct_found"]:
            lines.append(f"- `{col}`")
        lines.append("")
    
    if ohlc_check["ohlc_candles_found"]:
        lines.append(f"### Candles-Prefixed OHLC Columns Found")
        for col in ohlc_check["ohlc_candles_found"]:
            lines.append(f"- `{col}`")
        lines.append("")
    
    if ohlc_check["missing_fields_direct"]:
        lines.append(f"### Missing Direct OHLC Fields")
        for col in ohlc_check["missing_fields_direct"]:
            lines.append(f"- `{col}`")
        lines.append("")
    
    if ohlc_check["missing_fields_candles"]:
        lines.append(f"### Missing Candles-Prefixed OHLC Fields")
        for col in ohlc_check["missing_fields_candles"]:
            lines.append(f"- `{col}`")
        lines.append("")
    
    lines.extend([
        f"## Column Sample (First 50)",
        f"",
        f"```",
        f"{', '.join(sorted(columns)[:50])}",
        f"```",
        f"",
        f"---",
        f"*Report generated by ohlc_schema_gate.py at {datetime.utcnow().isoformat()}Z*",
    ])
    
    return "\n".join(lines)


def run_ohlc_schema_gate_or_fatal(
    output_dir: Path,
    truth_or_smoke: bool,
    data_path: Path,
) -> None:
    """
    Run OHLC schema gate over data file and hard-fail (exit code 2) in TRUTH/SMOKE if OHLC missing.
    """
    if not truth_or_smoke:
        return
    
    report_dir = _choose_report_dir(output_dir)
    fatal_path = report_dir / "OHLC_SCHEMA_FATAL.json"
    md_path = report_dir / "OHLC_SCHEMA_AUDIT.md"
    
    # Read schema (fast, no full load)
    columns = _read_schema_columns(data_path)
    if columns is None:
        msg = f"Failed to read schema from data file: {data_path}"
        capsule = {
            "status": "FAIL",
            "message": msg,
            "data_path": str(data_path),
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)
    
    # Get file size
    try:
        file_size = data_path.stat().st_size
    except Exception:
        file_size = 0
    
    # Check OHLC
    ohlc_check = _check_ohlc_columns(columns)
    
    # Write audit report (always)
    md_content = _render_md(
        data_path=data_path,
        columns=columns,
        ohlc_check=ohlc_check,
        file_size=file_size,
    )
    _write_text_atomic(md_path, md_content)
    
    # Hard-fail if OHLC missing
    if not (ohlc_check["has_ohlc_direct"] or ohlc_check["has_ohlc_candles"]):
        # Determine which family is closer (for hint)
        if ohlc_check["score"] == 0:
            # Neither family present
            missing_all = sorted(set(ohlc_check["missing_fields_direct"] + ohlc_check["missing_fields_candles"]))
            hint = (
                f"You are pointing to a prebuilt features file (no OHLC columns). "
                f"Use --data to point to RAW data file (with open/high/low/close columns), "
                f"and --prebuilt-parquet (or GX1_REPLAY_PREBUILT_FEATURES_PATH) for prebuilt features. "
                f"Example raw data: /home/andre2/GX1_DATA/data/data/raw/xauusd_m5_2025_bid_ask.parquet"
            )
        else:
            hint = "OHLC columns partially present but incomplete."
        
        msg = (
            f"Data file missing required OHLC columns. "
            f"Required: either {{open, high, low, close}} OR {{candles.open, candles.high, candles.low, candles.close}}. "
            f"Missing direct: {ohlc_check['missing_fields_direct']}, "
            f"Missing candles: {ohlc_check['missing_fields_candles']}"
        )
        
        capsule = {
            "status": "FAIL",
            "message": msg,
            "hint": hint,
            "data_path": str(data_path),
            "file_size": file_size,
            "n_columns": len(columns),
            "detected_columns_sample": sorted(columns)[:50],
            "missing_fields_direct": ohlc_check["missing_fields_direct"],
            "missing_fields_candles": ohlc_check["missing_fields_candles"],
            "ohlc_direct_found": ohlc_check["ohlc_direct_found"],
            "ohlc_candles_found": ohlc_check["ohlc_candles_found"],
            "sys.executable": sys.executable,
            "sys.version": sys.version,
            "cwd": str(Path.cwd()),
            "report_dir": str(report_dir),
            "report_path": str(md_path),
        }
        _write_json_atomic(fatal_path, capsule)
        raise SystemExit(2)
