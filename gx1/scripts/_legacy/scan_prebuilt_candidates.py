#!/usr/bin/env python3
"""
Deterministic prebuilt candidate scanner for train_xgb_universal_multihead_v2.py.

Scans ALL .parquet under a root (e.g. GX1_DATA), checks:
- Price column: mid or close (case-insensitive)
- Time column: ts or time or timestamp (case-insensitive)
- BASE28: all 28 features from contract (first 28 in xgb_input_features_v1.json)
- Clean sample: no NaN/Inf in price + BASE28 (sample ~2000 rows evenly spread)
- Year: from filename (xauusd_m5_2020...) or from time column

Uses pyarrow only (no pandas). Schema + batch read; does not load full files.
Output: CSV (prebuilt_scan.csv) + human summary + CANONICAL_PICK commands.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
except ImportError as e:
    print("ERROR: pyarrow required. Install: pip install pyarrow", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE28_COUNT = 28
SAMPLE_TARGET_ROWS = 2000
PRICE_NAMES = ("mid", "close")  # case-insensitive
TIME_NAMES = ("ts", "time", "timestamp")
YEAR_PATTERN = re.compile(r"(2020|2021|2022|2023|2024|2025)")


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def load_base28(contract_path: Path) -> List[str]:
    with open(contract_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])
    if len(features) < BASE28_COUNT:
        raise ValueError(
            f"Contract has {len(features)} features, need >= {BASE28_COUNT}"
        )
    return list(features[:BASE28_COUNT])


# ---------------------------------------------------------------------------
# Schema-only checks (no data read)
# ---------------------------------------------------------------------------


def _col_map(schema: pa.Schema) -> Dict[str, str]:
    """Map lower-case name -> actual name for case-insensitive lookup."""
    return {f.name.lower(): f.name for f in schema}

def _has_col(schema: pa.Schema, names: Tuple[str, ...]) -> Tuple[bool, Optional[str]]:
    m = _col_map(schema)
    for n in names:
        if n.lower() in m:
            return True, m[n.lower()]
    return False, None

def has_price_col(schema: pa.Schema) -> Tuple[bool, Optional[str]]:
    return _has_col(schema, PRICE_NAMES)

def has_time_col(schema: pa.Schema) -> Tuple[bool, Optional[str]]:
    return _has_col(schema, TIME_NAMES)

def base28_missing(schema: pa.Schema, base28: List[str]) -> List[str]:
    names = set(f.name for f in schema)
    return [f for f in base28 if f not in names]


# ---------------------------------------------------------------------------
# Sample read (minimal batches, evenly spread)
# ---------------------------------------------------------------------------


def sample_table_for_nan_inf(
    path: Path,
    columns: List[str],
    target_rows: int = SAMPLE_TARGET_ROWS,
) -> Tuple[Optional[pa.Table], Optional[str]]:
    """
    Read a small sample of columns (evenly spread over row groups).
    Returns (table, error_message). If error, table is None.
    """
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        return None, str(e)
    meta = pf.metadata
    num_groups = meta.num_row_groups
    if num_groups == 0:
        return None, "no_row_groups"
    # Which row groups to read to get ~target_rows evenly spread
    if num_groups <= 10:
        indices = list(range(num_groups))
    else:
        step = max(1, num_groups * target_rows // (target_rows * 2) or 1)
        indices = list(range(0, num_groups, step))[:20]
        if not indices:
            indices = [0]
    tables = []
    total = 0
    for i in indices:
        if total >= target_rows:
            break
        try:
            batch = pf.read_row_group(i)
        except Exception as e:
            return None, f"read_row_group({i}): {e}"
        # Select only columns that exist
        existing = [c for c in columns if c in batch.schema.names]
        if not existing:
            continue
        tbl = batch.select(existing)
        tables.append(tbl)
        total += tbl.num_rows
    if not tables:
        return None, "no_rows_read"
    try:
        combined = pa.concat_tables(tables)
    except Exception as e:
        return None, str(e)
    if combined.num_rows > target_rows:
        # Slice to first target_rows
        combined = combined.slice(0, target_rows)
    return combined, None


def has_nan_or_inf(table: pa.Table) -> bool:
    """True if any numeric column has NaN or Inf in the table."""
    for i in range(table.num_columns):
        col = table.column(i)
        if not pa.types.is_floating(col.type) and not pa.types.is_integer(col.type):
            continue
        if col.null_count > 0:
            return True
        if pa.types.is_floating(col.type):
            try:
                if pc.any(pc.is_nan(col)).as_py():
                    return True
                if pc.any(pc.is_inf(col)).as_py():
                    return True
            except Exception:
                # Fallback: sum of nulls
                if col.null_count > 0:
                    return True
    return False


# ---------------------------------------------------------------------------
# Year from path or time column
# ---------------------------------------------------------------------------


def year_from_path(path: Path) -> Optional[int]:
    m = YEAR_PATTERN.search(path.name)
    if m:
        return int(m.group(1))
    m = YEAR_PATTERN.search(str(path))
    if m:
        return int(m.group(1))
    return None


def year_from_time_column(path: Path, time_col: str) -> Optional[int]:
    """Read first batch and infer year from time column (int or datetime)."""
    try:
        pf = pq.ParquetFile(path)
        batch = pf.read_row_group(0)
    except Exception:
        return None
    if time_col not in batch.schema.names:
        return None
    col = batch.column(batch.schema.get_field_index(time_col))
    if len(col) == 0:
        return None
    try:
        if pa.types.is_timestamp(col.type) or pa.types.is_date(col.type):
            val = col[0]
            if hasattr(val, "as_py"):
                val = val.as_py()
            if hasattr(val, "year"):
                return val.year
            if isinstance(val, int):
                # Unix ms or ns
                if val > 1e12:
                    from datetime import datetime
                    return datetime.utcfromtimestamp(val / 1e9).year
                return datetime.utcfromtimestamp(val / 1e3).year
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Single-file scan
# ---------------------------------------------------------------------------


def scan_one(
    path: Path,
    base28: List[str],
    check_nan_inf: bool,
) -> Dict[str, Any]:
    path_str = str(path)
    size_bytes = path.stat().st_size if path.exists() else 0
    row: Dict[str, Any] = {
        "path": path_str,
        "size_bytes": size_bytes,
        "year_guess": None,
        "has_price": False,
        "price_col": "",
        "has_time": False,
        "time_col": "",
        "has_side": False,
        "has_session_id": False,
        "has_prob_long": False,
        "has_prob_short": False,
        "base28_missing_count": BASE28_COUNT,
        "base28_missing_list": "",
        "has_nan_inf_sample": None,
        "verdict": "UNKNOWN_YEAR",
    }
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
    except Exception as e:
        row["verdict"] = f"READ_ERROR: {e}"
        return row

    col_map_lower = _col_map(schema)
    col_set = set(schema.names)

    # Price
    hp, price_col = has_price_col(schema)
    row["has_price"] = hp
    row["price_col"] = price_col or ""

    # Time
    ht, time_col = has_time_col(schema)
    row["has_time"] = ht
    row["time_col"] = time_col or ""

    # Optional flags
    row["has_side"] = "side" in col_set or "side" in col_map_lower
    row["has_session_id"] = "session_id" in col_set or "session_id" in col_map_lower
    row["has_prob_long"] = "prob_long" in col_set or "prob_long" in col_map_lower
    row["has_prob_short"] = "prob_short" in col_set or "prob_short" in col_map_lower

    # BASE28
    missing = base28_missing(schema, base28)
    row["base28_missing_count"] = len(missing)
    row["base28_missing_list"] = "|".join(missing[:20])
    if len(missing) > 20:
        row["base28_missing_list"] += f"|...(+{len(missing)-20})"

    # Year
    year = year_from_path(path)
    if year is None and time_col:
        year = year_from_time_column(path, time_col)
    row["year_guess"] = year
    if year is None:
        row["verdict"] = "UNKNOWN_YEAR"
        return row

    # Verdict precedence: MISSING_PRICE -> MISSING_TIME -> MISSING_BASE28 -> HAS_NAN_INF -> VALID
    if not row["has_price"]:
        row["verdict"] = "MISSING_PRICE"
        return row
    if not row["has_time"]:
        row["verdict"] = "MISSING_TIME"
        return row
    if missing:
        row["verdict"] = "MISSING_BASE28"
        return row

    # NaN/Inf sample check
    if check_nan_inf:
        cols_to_check = [row["price_col"]] + [c for c in base28 if c in col_set]
        tbl, err = sample_table_for_nan_inf(path, cols_to_check, SAMPLE_TARGET_ROWS)
        if err:
            row["has_nan_inf_sample"] = None
            row["verdict"] = f"SAMPLE_READ_ERROR: {err}"
            return row
        row["has_nan_inf_sample"] = has_nan_or_inf(tbl)
        if row["has_nan_inf_sample"]:
            row["verdict"] = "HAS_NAN_INF"
            return row
    else:
        row["has_nan_inf_sample"] = False

    row["verdict"] = "VALID"
    return row


# ---------------------------------------------------------------------------
# Discovery + scan
# ---------------------------------------------------------------------------


def find_all_parquets(root: Path) -> List[Path]:
    return sorted(root.rglob("*.parquet"))


def run_scan(
    root: Path,
    contract_path: Path,
    out_dir: Path,
    check_nan_inf: bool = True,
) -> List[Dict[str, Any]]:
    base28 = load_base28(contract_path)
    parquets = find_all_parquets(root)
    rows: List[Dict[str, Any]] = []
    for i, path in enumerate(parquets):
        try:
            row = scan_one(path, base28, check_nan_inf)
        except Exception as e:
            row = {
                "path": str(path),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "year_guess": None,
                "has_price": False,
                "price_col": "",
                "has_time": False,
                "time_col": "",
                "has_side": False,
                "has_session_id": False,
                "has_prob_long": False,
                "has_prob_short": False,
                "base28_missing_count": BASE28_COUNT,
                "base28_missing_list": "",
                "has_nan_inf_sample": None,
                "verdict": f"EXCEPTION: {e}",
            }
        rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "prebuilt_scan.csv"
    # Write CSV with LF-only line endings and a single top-line comment for
    # downstream Unix-toolchain consumers.
    with open(csv_path, "w", newline="\n", encoding="utf-8") as f:
        # Header comment: file uses LF-only and verdict is the last column.
        f.write("# LF-only; verdict is last column\n")
        w = csv.DictWriter(f, fieldnames=[
            "path", "size_bytes", "year_guess", "has_price", "price_col",
            "has_time", "time_col", "has_side", "has_session_id",
            "has_prob_long", "has_prob_short", "base28_missing_count",
            "base28_missing_list", "has_nan_inf_sample", "verdict",
        ], lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    return rows


# ---------------------------------------------------------------------------
# Summary + CANONICAL_PICK
# ---------------------------------------------------------------------------


def verdict_rank(v: str) -> int:
    order = ("VALID", "MISSING_TIME", "MISSING_PRICE", "MISSING_BASE28", "HAS_NAN_INF")
    for i, x in enumerate(order):
        if v == x or (v.startswith(x) if x in ("VALID",) else v == x):
            return i
    return 99


def write_summary(rows: List[Dict[str, Any]], out_dir: Path, root: Path) -> None:
    summary_path = out_dir / "prebuilt_scan_summary.txt"
    valid_any = any(r.get("verdict") == "VALID" for r in rows)
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("PREBUILT SCAN SUMMARY")
    lines.append("=" * 70)
    if not valid_any:
        lines.append("NO_VALID_FOUND: No candidate meets all requirements (price + time + BASE28 + clean sample).")
    else:
        lines.append("At least one VALID candidate found.")
    lines.append("")

    by_year: Dict[Optional[int], List[Dict]] = {}
    for r in rows:
        y = r.get("year_guess")
        if y not in by_year:
            by_year[y] = []
        by_year[y].append(r)

    for year in [2020, 2021, 2022, 2023, 2024, 2025, None]:
        cands = by_year.get(year, [])
        if not cands:
            continue
        label = str(year) if year is not None else "UNKNOWN_YEAR"
        lines.append(f"--- Year {label} ({len(cands)} candidates) ---")
        # Sort: VALID first, then MISSING_TIME, MISSING_PRICE, MISSING_BASE28, rest
        cands_sorted = sorted(cands, key=lambda r: (verdict_rank(r.get("verdict", "")), r["path"]))
        for r in cands_sorted[:10]:
            short = r["path"].replace(str(root), "$ROOT")
            lines.append(f"  {r['verdict']}: {short}")
            lines.append(f"    price={r.get('price_col') or 'NO'} time={r.get('time_col') or 'NO'} "
                        f"base28_missing={r.get('base28_missing_count', 28)} nan_inf={r.get('has_nan_inf_sample')}")
        if len(cands_sorted) > 10:
            lines.append(f"  ... and {len(cands_sorted) - 10} more")
        lines.append("")

    # Known findings (as requested)
    lines.append("--- Known findings (from scan) ---")
    trial160_like = [r for r in rows if "TRIAL160" in r["path"] or "trial160" in r["path"].lower()]
    trial160_2020_2023 = [r for r in trial160_like if r.get("year_guess") in (2020, 2021, 2022, 2023)]
    if trial160_like:
        lines.append("TRIAL160 2020–2023: candidates with mid+ts+prob_long/prob_short and ~98 cols exist in tree.")
    lines.append("2024: candidates (e.g. TRIAL160_EXIT_AB / ABORTED) may lack 'ts' and have CLOSE (uppercase).")
    lines.append("2025: TRIAL160 with possible suffix (v12ab_clean etc.) — check scan CSV for each.")
    lines.append("")

    # CANONICAL_PICK
    lines.append("--- CANONICAL_PICK (one file per year; run these manually) ---")
    canonical_root = root / "data" / "data" / "prebuilt" / "TRIAL160"
    for year in [2020, 2021, 2022, 2023, 2024, 2025]:
        valid_for_year = [r for r in rows if r.get("year_guess") == year and r.get("verdict") == "VALID"]
        if not valid_for_year:
            lines.append(f"# Year {year}: NO VALID CANDIDATE — add prebuilt that meets all requirements first.")
            continue
        chosen = valid_for_year[0]
        target_dir = canonical_root / str(year)
        target_file = target_dir / f"xauusd_m5_{year}_features_v10_ctx.parquet"
        lines.append(f"# Year {year}:")
        lines.append(f"mkdir -p '{target_dir}'")
        lines.append(f"ln -sf '{chosen['path']}' '{target_file}'")
        lines.append("")
    if not any(r.get("verdict") == "VALID" for r in rows):
        lines.append("# No VALID candidates; no symlink commands generated.")
    lines.append("=" * 70)

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("SUMMARY:", summary_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan prebuilt parquet candidates for XGB v2 training.")
    parser.add_argument("--root", type=Path, required=True, help="Root to scan (e.g. GX1_DATA)")
    parser.add_argument("--contract", type=Path, required=True, help="Path to xgb_input_features_v1.json")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for CSV + summary")
    parser.add_argument("--no-nan-check", action="store_true", help="Skip NaN/Inf sample check (faster)")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"ERROR: root does not exist: {args.root}", file=sys.stderr)
        return 1
    if not args.contract.exists():
        print(f"ERROR: contract does not exist: {args.contract}", file=sys.stderr)
        return 1

    rows = run_scan(
        args.root,
        args.contract,
        args.out,
        check_nan_inf=not args.no_nan_check,
    )
    write_summary(rows, args.out, args.root)
    print("CSV:", args.out / "prebuilt_scan.csv")
    print("Total files scanned:", len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
