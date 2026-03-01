# ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# # ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# if False:
#     r"""
# #!/usr/bin/env python3
# raise RuntimeError("LEGACY_DISABLED: Use gx1/scripts/README_TRUTH_XGB.md (TRUTH XGB lane only).")
# """
# Find and verify raw candle parquets (XAUUSD M5 2020-2025).
# Output: CANDLES_AUDIT.md, candles_paths.json, install_candles_canonical.sh
# to /tmp/candles_src_root_audit_<UTC>/. No delete/move; only report + commented cp -a.
# """
# 
# from __future__ import annotations
# 
# import json
# import re
# import sys
# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple
# 
# try:
#     import pyarrow.parquet as pq
# except ImportError:
#     print("ERROR: pyarrow required", file=sys.stderr)
#     sys.exit(1)
# 
# GX1_DATA = Path("/home/andre2/GX1_DATA")
# CANONICAL_TARGET = Path("/home/andre2/GX1_DATA/data/oanda/years")
# REQUIRED_BID_ASK = [
#     "bid_open", "bid_high", "bid_low", "bid_close",
#     "ask_open", "ask_high", "ask_low", "ask_close",
# ]
# YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
# 
# 
# def utc_stamp() -> str:
#     return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
# 
# 
# def year_from_path(p: Path) -> Optional[int]:
#     m = re.search(r"(2020|2021|2022|2023|2024|2025)", p.name)
#     if m:
#         return int(m.group(1))
#     # Try parent dir name
#     m = re.search(r"(2020|2021|2022|2023|2024|2025)", str(p))
#     return int(m.group(1)) if m else None
# 
# 
# def find_candle_candidates(root: Path) -> List[Path]:
#     out: List[Path] = []
#     for p in root.rglob("*.parquet"):
#         if not p.is_file():
#             continue
#         s = str(p).lower()
#         if "oanda" not in s:
#             continue
#         name = p.name.lower()
#         # xauusd_m5_*bid*ask*.parquet or {YEAR}.parquet
#         if "xauusd" in name and "bid" in name and "ask" in name:
#             out.append(p)
#         if name in ("2020.parquet", "2021.parquet", "2022.parquet", "2023.parquet", "2024.parquet", "2025.parquet"):
#             out.append(p)
#     return sorted(set(out))
# 
# 
# def verify_candidate(path: Path) -> Tuple[str, str, Optional[str], Optional[str], Dict[str, Any]]:
#     """
#     Returns: (status, reason, first_ts, last_ts, extra).
#     status: VALID | INVALID
#     """
#     extra: Dict[str, Any] = {}
#     try:
#         pf = pq.ParquetFile(path)
#         schema = pf.schema_arrow
#         names = set(schema.names)
#         names_lower = {n.lower(): n for n in names}
# 
#         # Required bid/ask columns
#         missing = []
#         for c in REQUIRED_BID_ASK:
#             if c not in names and c not in names_lower:
#                 missing.append(c)
#         if missing:
#             return "INVALID", f"missing_columns:{','.join(missing[:4])}", None, None, extra
# 
#         # Index / timestamp column (often first or named index/timestamp/time/ts)
#         idx_candidates = [n for n in schema.names if n.lower() in ("index", "timestamp", "time", "ts", "datetime")]
#         if not idx_candidates:
#             idx_candidates = [schema.names[0]] if schema.names else []
#         if not idx_candidates:
#             return "INVALID", "no_index_candidate", None, None, extra
# 
#         col_name = idx_candidates[0]
#         batch0 = pf.read_row_group(0)
#         if col_name not in batch0.schema.names:
#             return "INVALID", "index_not_in_first_batch", None, None, extra
# 
#         col = batch0.column(batch0.schema.get_field_index(col_name))
#         if len(col) == 0:
#             return "INVALID", "empty_first_batch", None, None, extra
# 
#         dtype = str(col.type)
#         if "timestamp" not in dtype.lower() and "date" not in dtype.lower():
#             return "INVALID", f"index_not_timestamp:{dtype}", None, None, extra
# 
#         # tz=UTC
#         if "utc" not in dtype.lower() and "utc" not in dtype:
#             extra["index_dtype"] = dtype
#             # Could still be UTC if stored as naive UTC; we don't read values here
#             # Consider valid if timestamp type
#             pass
# 
#         first_ts = None
#         last_ts = None
#         try:
#             first_val = col[0]
#             if hasattr(first_val, "as_py"):
#                 first_ts = str(first_val.as_py())
#             else:
#                 first_ts = str(first_val)
#         except Exception:
#             pass
#         try:
#             last_val = col[len(col) - 1]
#             if hasattr(last_val, "as_py"):
#                 last_ts = str(last_val.as_py())
#             else:
#                 last_ts = str(last_val)
#         except Exception:
#             pass
# 
#         # Last row group for last_ts if single batch
#         if pf.metadata.num_row_groups > 1:
#             try:
#                 last_batch = pf.read_row_group(pf.metadata.num_row_groups - 1)
#                 if col_name in last_batch.schema.names:
#                     c = last_batch.column(last_batch.schema.get_field_index(col_name))
#                     if len(c) > 0:
#                         v = c[len(c) - 1]
#                         last_ts = str(v.as_py()) if hasattr(v, "as_py") else str(v)
#             except Exception:
#                 pass
# 
#         return "VALID", "ok", first_ts, last_ts, extra
#     except Exception as e:
#         return "INVALID", str(e)[:80], None, None, extra
# 
# 
# def main() -> int:
#     out_dir = Path(f"/tmp/candles_src_root_audit_{utc_stamp()}")
#     out_dir.mkdir(parents=True, exist_ok=True)
# 
#     print("RECOMMENDED_CANDLES_SRC_ROOT=(computed below)")
#     print("CANONICAL_CANDLES_TARGET=" + str(CANONICAL_TARGET))
#     print("")
# 
#     candidates = find_candle_candidates(GX1_DATA)
#     by_year: Dict[int, List[Dict]] = {y: [] for y in YEARS}
# 
#     for p in candidates:
#         y = year_from_path(p)
#         if y not in by_year:
#             continue
#         size = p.stat().st_size if p.exists() else 0
#         status, reason, first_ts, last_ts, extra = verify_candidate(p)
#         by_year[y].append({
#             "path": str(p),
#             "year": y,
#             "size_bytes": size,
#             "status": status,
#             "reason": reason,
#             "first_ts": first_ts,
#             "last_ts": last_ts,
#         })
# 
#     # Best per year: VALID only, then max size, then prefer longer ts range (we don't have exact range here, use size as proxy)
#     best_by_year: Dict[int, Dict] = {}
#     for y in YEARS:
#         valid = [c for c in by_year[y] if c["status"] == "VALID"]
#         if not valid:
#             best_by_year[y] = {"status": "MISSING_YEAR", "path": None, "size_bytes": None, "first_ts": None, "last_ts": None}
#             continue
#         valid.sort(key=lambda c: (c["size_bytes"], c["first_ts"] or "", c["last_ts"] or ""), reverse=True)
#         best_by_year[y] = valid[0]
# 
#     # Recommended SRC_ROOT: common parent of all best paths if same, else "per_year"
#     best_paths = [best_by_year[y]["path"] for y in YEARS if best_by_year[y].get("path")]
#     if not best_paths:
#         recommended_root = "MISSING"
#     else:
#         common = Path(best_paths[0]).parent
#         for s in best_paths[1:]:
#             common = common.parents[0] if common != Path(s).parent else common
#         # If all under same parent (e.g. quarantine/.../oanda/years) use that
#         if all(Path(s).parent == Path(best_paths[0]).parent for s in best_paths):
#             recommended_root = str(Path(best_paths[0]).parent)
#         elif all(Path(s).parent.parent == Path(best_paths[0]).parent.parent for s in best_paths):
#             recommended_root = str(Path(best_paths[0]).parent.parent)
#         else:
#             recommended_root = "per_year"
#     if recommended_root == "per_year":
#         recommended_root = str(Path(best_paths[0]).parent) if best_paths else "MISSING"
# 
#     # CANDLES_AUDIT.md
#     md = [
#         "# Candles source audit (XAUUSD M5 2020-2025)",
#         "",
#         f"Generated: {utc_stamp()}",
#         "",
#         "## Recommended SRC_ROOT",
#         "",
#         f"`{recommended_root}`",
#         "",
#         "## Best canonical candidate per year",
#         "",
#         "| Year | BEST_VALID path | size_bytes | first_ts | last_ts | status |",
#         "|------|-----------------|------------|----------|---------|--------|",
#     ]
#     for y in YEARS:
#         b = best_by_year[y]
#         if b.get("status") == "MISSING_YEAR":
#             md.append(f"| {y} | (none) | - | - | - | MISSING_YEAR |")
#         else:
#             path = b.get("path", "")
#             size = b.get("size_bytes", "")
#             first = (b.get("first_ts") or "")[:24]
#             last = (b.get("last_ts") or "")[:24]
#             md.append(f"| {y} | {path} | {size} | {first} | {last} | VALID |")
#     md.append("")
#     md.append("## All candidates (by year)")
#     for y in YEARS:
#         md.append(f"### {y}")
#         for c in by_year[y]:
#             md.append(f"- [{c['status']}] {c['path']} ({c['size_bytes']} bytes) — {c['reason']}")
#         md.append("")
#     (out_dir / "CANDLES_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
# 
#     # candles_paths.json
#     payload = {
#         "candidates_by_year": {str(y): by_year[y] for y in YEARS},
#         "best_by_year": {str(y): best_by_year[y] for y in YEARS},
#         "recommended_src_root": recommended_root,
#     }
#     (out_dir / "candles_paths.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
# 
#     # install_candles_canonical.sh (commented cp -a only)
#     sh_lines = [
#         "#!/usr/bin/env bash",
#         "# Copy BEST_VALID candle parquets to canonical path. Review before uncommenting.",
#         "# No rm, no mv. Target: " + str(CANONICAL_TARGET),
#         "",
#     ]
#     for y in YEARS:
#         b = best_by_year[y]
#         if b.get("status") == "MISSING_YEAR" or not b.get("path"):
#             sh_lines.append(f"# Year {y}: MISSING_YEAR — no cp")
#             continue
#         src = b["path"]
#         dst = CANONICAL_TARGET / str(y) / f"xauusd_m5_{y}_bid_ask.parquet"
#         sh_lines.append(f"# mkdir -p '{CANONICAL_TARGET / str(y)}'")
#         sh_lines.append(f"# cp -a '{src}' '{dst}'")
#         sh_lines.append("")
#     (out_dir / "install_candles_canonical.sh").write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
# 
#     print("OUT_DIR:", out_dir)
#     print("")
#     print("RECOMMENDED_CANDLES_SRC_ROOT=" + recommended_root)
#     print("CANONICAL_CANDLES_TARGET=" + str(CANONICAL_TARGET))
#     return 0
# 
# 
# if __name__ == "__main__":
#     sys.exit(main())
# 
#     """
