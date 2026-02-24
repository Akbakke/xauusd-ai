#!/usr/bin/env python3
raise RuntimeError("LEGACY_DISABLED: Use gx1/scripts/README_TRUTH_XGB.md (TRUTH XGB lane only).")
"""
ONE TRUTH BASE28 path audit.
Finds and classifies candles + prebuilt paths; identifies legacy.
Writes PATH_AUDIT.md, paths.json, cleanup_plan.sh to /tmp/base28_path_audit_<UTC>/.
Does NOT delete or move anything; cleanup_plan.sh contains COMMENTED suggestions only.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow required. pip install pyarrow", file=sys.stderr)
    sys.exit(1)

# SSoT roots
GX1_ENGINE = Path("/home/andre2/src/GX1_ENGINE")
GX1_DATA = Path("/home/andre2/GX1_DATA")
CONTRACT_PATH = GX1_ENGINE / "gx1/xgb/contracts/xgb_input_features_v1.json"
BASE28_COUNT = 28
SAMPLE_ROWS = 50_000

# Reserved candle columns (case-insensitive); prebuilt must NOT contain these
RESERVED_LOWER = {
    "open", "high", "low", "close", "volume",
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_contract_base28() -> List[str]:
    with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    features = data.get("features", [])
    if len(features) < BASE28_COUNT:
        raise ValueError(f"Contract has {len(features)} features, need >= {BASE28_COUNT}")
    return list(features[:BASE28_COUNT])


def find_candles(root: Path) -> List[Path]:
    out: List[Path] = []
    # Primary: oanda/years/{YEAR}/xauusd_m5_{YEAR}_bid_ask.parquet
    for p in root.rglob("oanda/years/*/xauusd_m5_*_bid_ask.parquet"):
        if p.is_file():
            out.append(p)
    # Alternative: oanda/years/xauusd_m5_*_bid_ask.parquet (no year subdir)
    for p in (root / "data" / "oanda" / "years").glob("xauusd_m5_*_bid_ask.parquet"):
        if p.is_file():
            out.append(p)
    for p in (root / "oanda" / "years").glob("xauusd_m5_*_bid_ask.parquet"):
        if p.is_file():
            out.append(p)
    # Alternative: oanda/years/{YEAR}.parquet
    for p in root.rglob("oanda/years/*.parquet"):
        if p.is_file() and p.name != "xauusd_m5_*_bid_ask.parquet":
            out.append(p)
    return sorted(set(out))


def find_prebuilt(root: Path) -> List[Path]:
    out: List[Path] = []
    for p in root.rglob("*.parquet"):
        if not p.is_file():
            continue
        name = p.name.lower()
        # prebuilt/**/xauusd_m5_*_features*.parquet, features/**/xauusd_m5_*_features*.parquet, **/TRIAL160/**/xauusd_m5_*_features*.parquet
        if "xauusd_m5_" in name and "_features" in name:
            path_str = str(p)
            if "prebuilt" in path_str or "features" in path_str or "trial160" in path_str.lower():
                out.append(p)
    return sorted(set(out))


def classify_path(p: Path, root: Path) -> str:
    s = str(p)
    if "quarantine" in s:
        return "QUARANTINE"
    if "archive" in s:
        return "ARCHIVE"
    # Canonical targets
    if "prebuilt" in s and "BASE28_CANONICAL" in s:
        return "PREBUILT_CANONICAL"
    if "oanda" in s and "years" in s and "bid_ask" in s:
        return "RAW_CANDLES"
    if "oanda" in s and "years" in s:
        return "RAW_CANDLES"
    if "prebuilt" in s or "TRIAL160" in s or "features" in s:
        return "PREBUILT_LEGACY"
    return "UNKNOWN"


def year_from_path(p: Path) -> Optional[int]:
    m = re.search(r"(2020|2021|2022|2023|2024|2025)", p.name)
    return int(m.group(1)) if m else None


def check_candles_index(path: Path) -> Tuple[bool, str]:
    """Check that parquet has DatetimeIndex with tz=UTC. Uses minimal read."""
    try:
        t = pq.read_table(path, columns=[])
        n = t.num_rows
    except Exception as e:
        return False, str(e)
    # Try to read index column (often first column or 'index')
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        names = schema.names
        if not names:
            return False, "no_columns"
        # Read first column (often index) with few rows
        first_col = names[0]
        batch = pf.read_row_group(0)
        if first_col not in batch.schema.names:
            return False, "first_col_missing"
        col = batch.column(batch.schema.get_field_index(first_col))
        if len(col) == 0:
            return False, "empty"
        dtype = col.type
        if not (str(dtype).startswith("timestamp") or str(dtype).startswith("date")):
            return False, f"index_not_timestamp:{dtype}"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def check_prebuilt(
    path: Path,
    base28: List[str],
) -> Dict[str, Any]:
    """Checks: index tz, reserved cols, contract cols, mid, NaN/Inf sample."""
    result: Dict[str, Any] = {
        "index_utc": None,
        "index_reason": "",
        "fatal_collision": False,
        "reserved_found": [],
        "contract_cols_ok": False,
        "contract_missing_count": BASE28_COUNT,
        "has_mid": False,
        "nan_inf_sample": None,
        "sample_rows": 0,
    }
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        names = set(schema.names)
        names_lower = {n.lower(): n for n in names}

        # Reserved (case-insensitive): open/high/low/close/volume, bid_*, ask_*
        for c in names:
            cl = c.lower()
            if cl in RESERVED_LOWER or cl.startswith("bid_") or cl.startswith("ask_"):
                result["reserved_found"].append(c)
                result["fatal_collision"] = True

        # Contract BASE28
        missing = [f for f in base28 if f not in names]
        result["contract_missing_count"] = len(missing)
        result["contract_cols_ok"] = len(missing) == 0

        # mid
        result["has_mid"] = "mid" in names or "mid" in names_lower

        # Index: first column or common names
        idx_candidates = [n for n in names if n.lower() in ("index", "timestamp", "ts", "time", "datetime")]
        if not idx_candidates and names:
            idx_candidates = [schema.names[0]]
        if idx_candidates:
            col_name = idx_candidates[0]
            batch = pf.read_row_group(0)
            if col_name in batch.schema.names:
                col = batch.column(batch.schema.get_field_index(col_name))
                dtype = str(col.type)
                result["index_utc"] = "timestamp" in dtype.lower() or "date" in dtype.lower()
                result["index_reason"] = dtype if result["index_utc"] else f"not_ts:{dtype}"
            else:
                result["index_reason"] = "index_col_not_in_first_batch"
        else:
            result["index_reason"] = "no_index_candidate"

        # NaN/Inf sample (up to SAMPLE_ROWS from first row groups)
        cols_to_check = [base28[i] for i in range(min(BASE28_COUNT, len(base28))) if base28[i] in names]
        if result["has_mid"] and "mid" in names:
            cols_to_check = ["mid"] + cols_to_check
        if cols_to_check:
            tables = []
            total = 0
            for i in range(min(pf.metadata.num_row_groups, 20)):
                if total >= SAMPLE_ROWS:
                    break
                batch = pf.read_row_group(i)
                existing = [c for c in cols_to_check if c in batch.schema.names]
                if existing:
                    tbl = batch.select(existing)
                    tables.append(tbl)
                    total += tbl.num_rows
            if tables:
                import pyarrow as pa
                import pyarrow.compute as pc
                combined = pa.concat_tables(tables)
                if combined.num_rows > SAMPLE_ROWS:
                    combined = combined.slice(0, SAMPLE_ROWS)
                result["sample_rows"] = combined.num_rows
                has_bad = False
                for j in range(combined.num_columns):
                    col = combined.column(j)
                    try:
                        nc = getattr(col, "null_count", 0)
                        if (callable(nc) and nc() > 0) or (not callable(nc) and nc > 0):
                            has_bad = True
                            break
                    except Exception:
                        pass
                    if hasattr(col.type, "id") and getattr(col.type, "id", None) in (7, 8):
                        try:
                            if pc.any(pc.is_nan(col)).as_py() or pc.any(pc.is_inf(col)).as_py():
                                has_bad = True
                                break
                        except Exception:
                            pass
                result["nan_inf_sample"] = has_bad
    except Exception as e:
        result["index_reason"] = str(e)
    return result


def find_legacy_scripts(repo: Path) -> List[Dict[str, Any]]:
    patterns = ["prebuilt", "features", "ctx", "prune14", "prune20", "trial160", "baseline"]
    out = []
    for py in repo.rglob("gx1/scripts/**/*.py"):
        if not py.is_file():
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
            text_lower = text.lower()
            for pat in patterns:
                if pat.lower() in text_lower:
                    rel = py.relative_to(repo) if repo in py.parents else py
                    out.append({
                        "path": str(rel),
                        "matches": pat,
                        "reason": f"contains '{pat}'",
                    })
                    break
        except Exception:
            pass
    for py in repo.rglob("gx1/**/*.py"):
        if "scripts" in str(py):
            continue
        if not py.is_file():
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="ignore")
            text_lower = text.lower()
            for pat in patterns:
                if pat.lower() in text_lower:
                    rel = py.relative_to(repo) if repo in py.parents else py
                    out.append({"path": str(rel), "matches": pat, "reason": f"contains '{pat}'"})
                    break
        except Exception:
            pass
    # Dedupe by path
    seen = set()
    deduped = []
    for x in out:
        if x["path"] not in seen:
            seen.add(x["path"])
            deduped.append(x)
    return deduped


def main() -> int:
    out_dir = Path(f"/tmp/base28_path_audit_{utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    base28 = load_contract_base28()

    # --- Candles ---
    candles = find_candles(GX1_DATA)
    candles_by_year: Dict[int, List[Dict]] = {}
    for p in candles:
        size = p.stat().st_size if p.exists() else 0
        ok, reason = check_candles_index(p)
        y = year_from_path(p)
        entry = {"path": str(p), "size_bytes": size, "index_utc_ok": ok, "index_reason": reason}
        if y not in candles_by_year:
            candles_by_year[y] = []
        candles_by_year[y].append(entry)

    # --- Prebuilt ---
    prebuilt_all = find_prebuilt(GX1_DATA)
    prebuilt_candidates: List[Dict[str, Any]] = []
    for p in prebuilt_all:
        size = p.stat().st_size if p.exists() else 0
        kind = classify_path(p, GX1_DATA)
        checks = check_prebuilt(p, base28)
        y = year_from_path(p)
        prebuilt_candidates.append({
            "path": str(p),
            "size_bytes": size,
            "year": y,
            "classification": kind,
            **checks,
        })

    # --- Legacy scripts ---
    legacy_scripts = find_legacy_scripts(GX1_ENGINE)
    # Exclude the canonical builder
    legacy_scripts = [s for s in legacy_scripts if "build_prebuilt_base28_universal" not in s["path"]]

    # Best candles per year (canonical pattern preferred)
    canonical_candles_root = str(GX1_DATA / "data" / "oanda" / "years")
    best_candles: Dict[int, Dict] = {}
    for y in [2020, 2021, 2022, 2023, 2024, 2025]:
        cands = candles_by_year.get(y, [])
        # Prefer path matching {YEAR}/xauusd_m5_{YEAR}_bid_ask.parquet
        canonical_style = [c for c in cands if f"/{y}/" in c["path"] and f"xauusd_m5_{y}_bid_ask" in c["path"]]
        if canonical_style:
            best = canonical_style[0]
            for c in canonical_style:
                if c.get("index_utc_ok") and (not best.get("index_utc_ok") or c.get("size_bytes", 0) > best.get("size_bytes", 0)):
                    best = c
        else:
            best = cands[0] if cands else {}
        best_candles[y] = best

    # --- PATH_AUDIT.md ---
    md_lines = [
        "# ONE TRUTH BASE28 Path Audit",
        "",
        f"Generated: {utc_stamp()}",
        "",
        "## Canonical roots (recommended)",
        "",
        f"- **Canonical raw candles root:** `{GX1_DATA}/data/oanda/years`",
        f"  - Pattern: `{{YEAR}}/xauusd_m5_{{YEAR}}_bid_ask.parquet`",
        "",
        f"- **Canonical prebuilt output root:** `{GX1_DATA}/data/data/prebuilt/BASE28_CANONICAL`",
        "",
        "## Candles by year (best candidate)",
        "",
    ]
    for y in [2020, 2021, 2022, 2023, 2024, 2025]:
        b = best_candles.get(y, {})
        if not b:
            md_lines.append(f"- **{y}:** (no candidate found)")
        else:
            path = b.get("path", "")
            ok = b.get("index_utc_ok", False)
            reason = b.get("index_reason", "")
            md_lines.append(f"- **{y}:** `{path}`")
            md_lines.append(f"  - index_utc_ok: {ok}, reason: {reason}")
        md_lines.append("")

    md_lines.append("## Prebuilt candidates (legacy / quarantine / canonical)")
    md_lines.append("")
    for c in prebuilt_candidates[:200]:  # cap for readability
        p = c["path"]
        cl = c["classification"]
        fat = " FATAL_COLLISION" if c.get("fatal_collision") else ""
        md_lines.append(f"- [{cl}{fat}] `{p}`")
        md_lines.append(f"  - year: {c.get('year')}, index_utc: {c.get('index_utc')}, contract_ok: {c.get('contract_cols_ok')}, has_mid: {c.get('has_mid')}, nan_inf_sample: {c.get('nan_inf_sample')}")
    if len(prebuilt_candidates) > 200:
        md_lines.append(f"- ... and {len(prebuilt_candidates) - 200} more (see paths.json)")
    md_lines.append("")

    md_lines.append("## Legacy scripts (prebuilt/features/ctx/prune/trial160/baseline)")
    md_lines.append("")
    for s in legacy_scripts:
        md_lines.append(f"- `{s['path']}` — {s['reason']}")
    md_lines.append("")

    (out_dir / "PATH_AUDIT.md").write_text("\n".join(md_lines), encoding="utf-8")

    # --- paths.json ---
    paths_payload = {
        "raw_candles_by_year": {str(y): [e["path"] for e in candles_by_year.get(y, [])] for y in [2020, 2021, 2022, 2023, 2024, 2025]},
        "best_candles_per_year": {str(y): best_candles.get(y, {}) for y in [2020, 2021, 2022, 2023, 2024, 2025]},
        "prebuilt_candidates": prebuilt_candidates,
        "legacy_scripts": legacy_scripts,
    }
    (out_dir / "paths.json").write_text(json.dumps(paths_payload, indent=2), encoding="utf-8")

    # --- cleanup_plan.sh (commented only) ---
    sh_lines = [
        "#!/usr/bin/env bash",
        "# ONE TRUTH BASE28 — suggested cleanup (COMMENTED; do not run as-is)",
        "# Do NOT touch RAW_CANDLES. Do NOT touch gx1/scripts/prebuilt/build_prebuilt_base28_universal.py",
        "",
        "# Example: move legacy prebuilts to _legacy (review paths first):",
        "# mv /path/to/old_prebuilt /path/to/prebuilt/_legacy/",
        "",
        "# Example: move legacy scripts to archive (review first):",
        "# mv gx1/scripts/old_builder.py gx1/scripts/archive/",
        "",
    ]
    for c in prebuilt_candidates[:80]:
        if c["classification"] in ("PREBUILT_LEGACY", "QUARANTINE", "ARCHIVE"):
            p = c["path"]
            sh_lines.append(f"# mv '{p}' '{GX1_DATA}/data/data/prebuilt/_legacy/'  # review before uncomment")
    if len([c for c in prebuilt_candidates if c["classification"] in ("PREBUILT_LEGACY", "QUARANTINE", "ARCHIVE")]) > 80:
        sh_lines.append("# ... (more in paths.json; add mv manually if needed)")
    (out_dir / "cleanup_plan.sh").write_text("\n".join(sh_lines) + "\n", encoding="utf-8")

    print("OUT_DIR:", out_dir)
    print("PATH_AUDIT.md:", out_dir / "PATH_AUDIT.md")
    print("paths.json:", out_dir / "paths.json")
    print("cleanup_plan.sh:", out_dir / "cleanup_plan.sh")
    print("")
    print("ONE TRUTH (use these going forward):")
    print(f"CANONICAL_RAW_CANDLES_ROOT={GX1_DATA}/data/oanda/years")
    print(f"CANONICAL_PREBUILT_OUTPUT_ROOT={GX1_DATA}/data/data/prebuilt/BASE28_CANONICAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
