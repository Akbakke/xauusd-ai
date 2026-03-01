# ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# # ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# if False:
#     r"""
# #!/usr/bin/env python3
# raise RuntimeError("LEGACY_DISABLED: Use gx1/scripts/README_TRUTH_XGB.md (TRUTH XGB lane only).")
# from __future__ import annotations
# 
# """
# pick_base28_features_from_legacy.py
# 
# Pick best legacy prebuilt/feature parquet per year (2020..2025) for BASE28,
# validate against canonical candles and contract, write pick report and install script.
# When a year has no valid legacy candidate (index matches candles), generate BASE28
# from canonical candles via build_basic_v1 and write to BASE28_FEATURES.
# 
# SSoT:
# - Candles: {GX1_DATA}/data/oanda/years/{YEAR}/xauusd_m5_{YEAR}_bid_ask.parquet
# - BASE28 contract: gx1/xgb/contracts/xgb_input_features_base28_v1.json
# - Target: {GX1_DATA}/data/data/prebuilt/BASE28_FEATURES/{YEAR}/xauusd_m5_{YEAR}_features_base28.parquet
# 
# Output: /tmp/base28_features_pick_<UTC>/ with PICK.md, pick.json, install_base28_features.sh.
# """
# 
# import json
# import re
# import subprocess
# import sys
# from datetime import datetime, timezone
# from pathlib import Path
# 
# import numpy as np
# import pandas as pd
# 
# # Paths (override via env or default)
# import os as _os
# GX1_DATA = Path(_os.environ.get("GX1_DATA", "/home/andre2/GX1_DATA")).resolve()
# GX1_ENGINE = Path(__file__).resolve().parents[3]
# 
# RESERVED_CANDLE_COLUMNS = {
#     "open", "high", "low", "close", "volume",
#     "bid_open", "bid_high", "bid_low", "bid_close",
#     "ask_open", "ask_high", "ask_low", "ask_close",
# }
# YEARS = list(range(2020, 2026))
# 
# 
# def load_contract():
#     p = GX1_ENGINE / "gx1/xgb/contracts/xgb_input_features_base28_v1.json"
#     with open(p, "r", encoding="utf-8") as f:
#         c = json.load(f)
#     return c["features"]
# 
# 
# def candles_index_path(year: int) -> Path:
#     return GX1_DATA / "data/oanda/years" / str(year) / f"xauusd_m5_{year}_bid_ask.parquet"
# 
# 
# def load_candles_index(year: int):
#     p = candles_index_path(year)
#     if not p.exists():
#         return None
#     df = pd.read_parquet(p)
#     if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None or str(df.index.tz) != "UTC":
#         return None
#     return df.index
# 
# 
# def load_candles_full(year: int) -> pd.DataFrame | None:
#     """Load full candles df for generation; None if missing or invalid."""
#     p = candles_index_path(year)
#     if not p.exists():
#         return None
#     df = pd.read_parquet(p)
#     if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None or str(df.index.tz) != "UTC":
#         return None
#     for c in ("bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"):
#         if c not in df.columns:
#             return None
#     return df
# 
# 
# def generate_base28_from_candles(year: int, base28_features: list) -> pd.DataFrame:
#     """
#     Build BASE28 features from canonical candles using build_basic_v1.
#     Returns DataFrame with DatetimeIndex UTC and exactly BASE28 columns (contract order).
#     Hard-fail if pipeline adds reserved columns to output set, or NaN/Inf in BASE28.
#     """
#     candles = load_candles_full(year)
#     if candles is None:
#         raise RuntimeError(f"Cannot load canonical candles for year {year}")
# 
#     # OHLC from bid/ask mid (deterministic)
#     df = pd.DataFrame(
#         {
#             "open": ((candles["bid_open"] + candles["ask_open"]) / 2).astype(np.float64),
#             "high": ((candles["bid_high"] + candles["ask_high"]) / 2).astype(np.float64),
#             "low": ((candles["bid_low"] + candles["ask_low"]) / 2).astype(np.float64),
#             "close": ((candles["bid_close"] + candles["ask_close"]) / 2).astype(np.float64),
#         },
#         index=candles.index,
#     )
#     df["ts"] = df.index
# 
#     # build_basic_v1 requires FEATURE_STATE (HTF cache) and can hit FEATURE_BUILD_TIMEOUT_MS
#     from gx1.features.feature_state import FeatureState
#     from gx1.utils.feature_context import set_feature_state, reset_feature_state
#     from gx1.features.basic_v1 import build_basic_v1
# 
#     feature_state = FeatureState()
#     token = set_feature_state(feature_state)
#     try:
#         old_timeout = _os.environ.get("FEATURE_BUILD_TIMEOUT_MS")
#         _os.environ["FEATURE_BUILD_TIMEOUT_MS"] = "600000"  # 10 min for full-year build
#         try:
#             feature_df, _ = build_basic_v1(df)
#         finally:
#             if old_timeout is None:
#                 _os.environ.pop("FEATURE_BUILD_TIMEOUT_MS", None)
#             else:
#                 _os.environ["FEATURE_BUILD_TIMEOUT_MS"] = old_timeout
#     finally:
#         reset_feature_state(token)
# 
#     missing = [f for f in base28_features if f not in feature_df.columns]
#     if missing:
#         raise RuntimeError(f"build_basic_v1 did not produce BASE28 columns: {missing}")
# 
#     out = feature_df[base28_features].copy()
#     out.index = feature_df.index
# 
#     # No reserved in output (case-insensitive)
#     lower = {c.lower() for c in out.columns}
#     if lower & RESERVED_CANDLE_COLUMNS:
#         raise RuntimeError(f"Pipeline produced reserved columns in output: {sorted(lower & RESERVED_CANDLE_COLUMNS)}")
# 
#     if not np.isfinite(out.to_numpy()).all():
#         raise RuntimeError(f"Generated BASE28 for {year} contains NaN/Inf")
# 
#     out = out.sort_index()
#     return out
# 
# 
# def find_candidate_parquets():
#     root = GX1_DATA
#     keywords = ("prebuilt", "features", "archive", "quarantine", "trial160", "baseline")
#     pattern = re.compile(r"xauusd.*m5|m5.*xauusd", re.I)
#     parquets = []
#     for path in root.rglob("*.parquet"):
#         try:
#             s = path.as_posix().lower()
#             if not any(k in s for k in keywords):
#                 continue
#             if not pattern.search(path.name):
#                 continue
#             parquets.append(path.resolve())
#         except OSError:
#             continue
#     return parquets
# 
# 
# def year_from_path(path: Path) -> int | None:
#     """Return single year in 2020..2025 if path clearly refers to one year; else None."""
#     s = path.as_posix()
#     years_found = set()
#     for y in YEARS:
#         if f"/{y}/" in s or f"_{y}." in s or f"_{y}_" in s or path.name.startswith(f"{y}_"):
#             years_found.add(y)
#     if len(years_found) == 1:
#         return years_found.pop()
#     return None
# 
# 
# def validate_candidate(path: Path, year: int, candle_index: pd.DatetimeIndex, base28_set: set) -> tuple[bool, str, dict]:
#     """
#     Returns (ok, reason, info_dict with rows, first_ts, last_ts, bytes).
#     Source may have extra columns (e.g. CLOSE, other features); we only require
#     it has all BASE28 columns, index match, and no NaN/Inf in BASE28. Output to
#     target will be BASE28 columns only.
#     """
#     try:
#         df = pd.read_parquet(path)
#     except Exception as e:
#         return False, f"read_parquet: {e}", {}
# 
#     info = {"rows": len(df), "bytes": path.stat().st_size}
# 
#     if not isinstance(df.index, pd.DatetimeIndex):
#         return False, "index not DatetimeIndex", info
#     if df.index.tz is None or str(df.index.tz) != "UTC":
#         return False, "index not UTC", info
#     if not candle_index.equals(df.index):
#         return False, "index does not match candles 1:1", info
# 
#     have = set(df.columns)
#     missing = base28_set - have
#     if missing:
#         return False, f"missing BASE28 columns: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}", info
# 
#     # BASE28 output must not contain reserved; contract has no reserved, so ok.
#     # Source may have CLOSE/other reserved; we only write BASE28 to target.
#     sub = df[list(base28_set)]
#     if not np.isfinite(sub.to_numpy()).all():
#         return False, "NaN/Inf in BASE28 columns", info
# 
#     info["first_ts"] = str(df.index[0])
#     info["last_ts"] = str(df.index[-1])
#     return True, "ok", info
# 
# 
# def main():
#     base28_features = load_contract()
#     base28_set = set(base28_features)
# 
#     out_dir = Path(f"/tmp/base28_features_pick_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
#     out_dir.mkdir(parents=True, exist_ok=True)
# 
#     target_root = GX1_DATA / "data/data/prebuilt/BASE28_FEATURES"
#     target_root.mkdir(parents=True, exist_ok=True)
# 
#     parquets = find_candidate_parquets()
#     by_year = {y: [] for y in YEARS}
#     for p in parquets:
#         y = year_from_path(p)
#         if y is not None:
#             by_year[y].append(p)
# 
#     best_by_year = {}
#     rejected = []
#     pick_lines = []
# 
#     for year in YEARS:
#         candle_idx = load_candles_index(year)
#         if candle_idx is None:
#             pick_lines.append((year, None, "no canonical candles index"))
#             rejected.append({"year": year, "path": None, "reason": "no canonical candles"})
#             continue
# 
#         valid = []
#         for path in by_year.get(year, []):
#             ok, reason, info = validate_candidate(path, year, candle_idx, base28_set)
#             if ok:
#                 valid.append((path, info["rows"], info["bytes"], info.get("first_ts"), info.get("last_ts")))
#             else:
#                 rejected.append({"year": year, "path": str(path), "reason": reason})
# 
#         if not valid:
#             pick_lines.append((year, None, "no valid candidate (will generate from candles)"))
#             continue
# 
#         valid.sort(key=lambda x: (-x[1], -x[2]))
#         best_path, rows, bytes_, first_ts, last_ts = valid[0]
#         best_by_year[str(year)] = {
#             "src": str(best_path),
#             "rows": rows,
#             "bytes": bytes_,
#             "first_ts": first_ts,
#             "last_ts": last_ts,
#             "reason": "best valid (rows then bytes)",
#             "generated": False,
#         }
#         pick_lines.append((year, best_path, rows, first_ts, last_ts, "ok"))
# 
#     # Generate from candles for years with no valid legacy
#     for year in YEARS:
#         if str(year) in best_by_year:
#             continue
#         if load_candles_index(year) is None:
#             continue
#         try:
#             df_gen = generate_base28_from_candles(year, base28_features)
#         except Exception as e:
#             print(f"[GENERATE] {year}: failed: {e}", file=sys.stderr)
#             continue
#         dst = target_root / str(year) / f"xauusd_m5_{year}_features_base28.parquet"
#         dst.parent.mkdir(parents=True, exist_ok=True)
#         df_gen.to_parquet(dst)
#         best_by_year[str(year)] = {
#             "src": "(generated)",
#             "rows": len(df_gen),
#             "bytes": dst.stat().st_size,
#             "first_ts": str(df_gen.index[0]),
#             "last_ts": str(df_gen.index[-1]),
#             "reason": "generated from candles",
#             "generated": True,
#         }
#         print(f"[GENERATE] {year}: wrote {dst} rows={len(df_gen)}")
# 
#     # Rebuild pick_lines so PICK.md reflects legacy + generated
#     pick_lines_final = []
#     for year in YEARS:
#         if str(year) in best_by_year:
#             b = best_by_year[str(year)]
#             pick_lines_final.append((year, b["src"], b["rows"], b["first_ts"], b["last_ts"], b["reason"]))
#         else:
#             pick_lines_final.append((year, None, "no canonical candles or generation failed"))
#     pick_lines = pick_lines_final
# 
#     # PICK.md
#     md = ["# BASE28 features pick from legacy", "", "| year | chosen_src | rows | first_ts | last_ts | reason |", "|------|------------|------|----------|---------|--------|"]
#     for t in pick_lines:
#         if len(t) == 3:
#             year, src, reason = t
#             md.append(f"| {year} | (none) | - | - | - | {reason} |")
#         else:
#             year, src, rows, first_ts, last_ts, reason = t
#             md.append(f"| {year} | {src} | {rows} | {first_ts} | {last_ts} | {reason} |")
#     (out_dir / "PICK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
# 
#     # pick.json
#     pick_json = {"best_by_year": best_by_year, "rejected": rejected}
#     (out_dir / "pick.json").write_text(json.dumps(pick_json, indent=2) + "\n", encoding="utf-8")
# 
#     # install_base28_features.sh (audit: sources and targets; script writes BASE28-only to target)
#     sh_lines = [
#         "# Generated by pick_base28_features_from_legacy.py",
#         "# Target files are written as BASE28 columns only (selected from chosen source).",
#         "# Sources below; re-run this script to regenerate target parquets.",
#         "",
#         "set -e",
#         "FEATURES_ROOT=/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_FEATURES",
#         "",
#         "# Chosen sources (script wrote BASE28-only to target):",
#         "# ---",
#     ]
#     for year in YEARS:
#         if str(year) not in best_by_year:
#             sh_lines.append(f"# {year}: (none)")
#             continue
#         src = best_by_year[str(year)]["src"]
#         dst = target_root / str(year) / f"xauusd_m5_{year}_features_base28.parquet"
#         sh_lines.append(f"# {year}: '{src}' -> '{dst}'")
#     sh_lines.append("# ---")
#     (out_dir / "install_base28_features.sh").write_text("\n".join(sh_lines) + "\n", encoding="utf-8")
# 
#     print(f"OUT_DIR: {out_dir}")
#     print(f"PICK.md: {out_dir / 'PICK.md'}")
#     print(f"pick.json: {out_dir / 'pick.json'}")
#     print(f"install_base28_features.sh: {out_dir / 'install_base28_features.sh'}")
# 
#     # Write target: legacy = select BASE28 from source; generated already written above
#     for year in YEARS:
#         if str(year) not in best_by_year:
#             continue
#         b = best_by_year[str(year)]
#         if b.get("generated"):
#             continue
#         src = Path(b["src"])
#         dst = target_root / str(year) / f"xauusd_m5_{year}_features_base28.parquet"
#         dst.parent.mkdir(parents=True, exist_ok=True)
#         df = pd.read_parquet(src)
#         df[base28_features].to_parquet(dst)
#         print(f"[WRITE] {year}: {src.name} -> {dst} (BASE28 columns only)")
# 
#     print("FEATURES_ROOT=/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_FEATURES")
# 
#     # Build canonical prebuilt only when all years are filled
#     missing_years = [y for y in YEARS if str(y) not in best_by_year]
#     if missing_years:
#         print(f"[BUILD] Skipped (missing years: {missing_years}). Fill sources then run build manually.")
#     else:
#         build_script = GX1_ENGINE / "gx1/scripts/prebuilt/build_prebuilt_base28_universal.py"
#         out_parquet = GX1_DATA / "data/data/prebuilt/BASE28_CANONICAL/xauusd_m5_BASE28_2020_2025.parquet"
#         contract = GX1_ENGINE / "gx1/xgb/contracts/xgb_input_features_base28_v1.json"
#         data_root = GX1_DATA / "data"
#         cmd = [
#             sys.executable,
#             str(build_script),
#             "--data-root", str(data_root),
#             "--features-root", str(target_root),
#             "--contract", str(contract),
#             "--years", "2020..2025",
#             "--out", str(out_parquet),
#         ]
#         print("[BUILD] Running BASE28 canonical prebuilt builder...")
#         r = subprocess.run(cmd)
#         if r.returncode != 0:
#             sys.exit(r.returncode)
# 
# 
# if __name__ == "__main__":
#     main()
# 
#     """
