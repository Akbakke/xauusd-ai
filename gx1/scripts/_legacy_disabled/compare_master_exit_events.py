# ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# # ARCHIVED (NON-EXECUTABLE): kept for reference; wrapped to be syntactically parseable.
# if False:
#     r"""
# #!/home/andre2/venvs/gx1/bin/python
# raise RuntimeError("LEGACY_DISABLED: Use gx1/scripts/README_TRUTH_XGB.md (TRUTH XGB lane only).")
# """
# Deterministic comparison of MASTER_EXIT_V1 exits between two runs.
# Proves whether guardrails (min_hold_bars, min_mfe_bps_to_arm) affect exit timing.
# 
# Reads: chunk_footer.json (EXIT_TUNING_CAPSULE), replay/chunk_0/logs/exits/exits_*.jsonl.
# Output: key=value lines; ends with VERDICT=NO_CHANGE_DETECTED | TIMING_SHIFT_DETECTED |
#   PNL_SHIFT_DETECTED | GUARDRAIL_VIOLATION.
# 
# Usage:
#   /home/andre2/venvs/gx1/bin/python -m gx1.scripts.compare_master_exit_events
#   /home/andre2/venvs/gx1/bin/python -m gx1.scripts.compare_master_exit_events --run-a <dir> --run-b <dir>
# """
# 
# from __future__ import annotations
# 
# import argparse
# import json
# import sys
# from collections import Counter
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple
# 
# DEFAULT_RUN_A = "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/E2E_SANITY_20260219_084126"
# DEFAULT_RUN_B = "/home/andre2/GX1_DATA/reports/truth_e2e_sanity/E2E_SANITY_20260219_101036"
# MAX_JSONL_LINES = 50_000
# 
# 
# def _load_json(path: Path) -> dict:
#     if not path.exists():
#         print(f"Missing: {path}", file=sys.stderr)
#         sys.exit(1)
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)
# 
# 
# def _get_first(rec: dict, *keys: str) -> Any:
#     for k in keys:
#         if k in rec and rec[k] is not None:
#             return rec[k]
#     return None
# 
# 
# def _parse_exit_record(rec: dict) -> dict:
#     """Extract fields robustly; missing -> None."""
#     reason = _get_first(rec, "reason", "close_reason", "exit_reason")
#     source = rec.get("source")
#     bars_held = _get_first(rec, "bars_held", "bars_in_trade", "barsInTrade", "holding_bars", "n_bars_held")
#     mfe_bps = _get_first(rec, "mfe_bps", "mfe_bps_now", "mfe_bps_peak")
#     mfe_atr = _get_first(rec, "mfe_atr", "mfe_in_atr", "mfe_atr_units")
#     pnl_bps = _get_first(rec, "pnl_bps", "pnl_bps_now", "pnl_bps_at_exit")
#     trail_gap_atr = _get_first(rec, "trail_gap_atr", "trail_gap")
#     trailing_active = rec.get("trailing_active")
#     trail_armed = rec.get("trail_armed")
#     return {
#         "reason": str(reason) if reason is not None else None,
#         "source": str(source) if source is not None else None,
#         "bars_held": int(bars_held) if bars_held is not None else None,
#         "mfe_bps": float(mfe_bps) if mfe_bps is not None else None,
#         "mfe_atr": float(mfe_atr) if mfe_atr is not None else None,
#         "pnl_bps": float(pnl_bps) if pnl_bps is not None else None,
#         "trail_gap_atr": float(trail_gap_atr) if trail_gap_atr is not None else None,
#         "trailing_active": trailing_active,
#         "trail_armed": trail_armed,
#     }
# 
# 
# def _load_exits_jsonl(chunk_dir: Path) -> List[dict]:
#     exits_dir = chunk_dir / "logs" / "exits"
#     if not exits_dir.exists():
#         return []
#     files = sorted(
#         exits_dir.glob("exits_*.jsonl"),
#         key=lambda p: p.stat().st_mtime,
#         reverse=True,
#     )
#     if not files:
#         return []
#     out: List[dict] = []
#     with open(files[0], "r", encoding="utf-8") as f:
#         for i, line in enumerate(f):
#             if i >= MAX_JSONL_LINES:
#                 break
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 rec = json.loads(line)
#                 out.append(_parse_exit_record(rec))
#             except json.JSONDecodeError:
#                 continue
#     return out
# 
# 
# def _percentile_sorted(sorted_list: List[float], p: float) -> Optional[float]:
#     if not sorted_list:
#         return None
#     idx = int(len(sorted_list) * p / 100.0)
#     idx = min(idx, len(sorted_list) - 1)
#     return sorted_list[idx]
# 
# 
# def _stats(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
#     clean = [v for v in values if v is not None]
#     if not clean:
#         return None, None, None, None
#     s = sorted(clean)
#     return (
#         min(s),
#         _percentile_sorted(s, 50),
#         _percentile_sorted(s, 90),
#         max(s),
#     )
# 
# 
# def main() -> int:
#     ap = argparse.ArgumentParser(
#         description="Compare MASTER_EXIT_V1 exits between two runs (guardrail / timing analysis)."
#     )
#     ap.add_argument("--run-a", type=str, default=DEFAULT_RUN_A, help="Baseline run dir")
#     ap.add_argument("--run-b", type=str, default=DEFAULT_RUN_B, help="Tuned run dir")
#     args = ap.parse_args()
# 
#     run_a = Path(args.run_a).expanduser().resolve()
#     run_b = Path(args.run_b).expanduser().resolve()
#     if not run_a.exists():
#         print(f"Missing RUN_A: {run_a}", file=sys.stderr)
#         sys.exit(1)
#     if not run_b.exists():
#         print(f"Missing RUN_B: {run_b}", file=sys.stderr)
#         sys.exit(1)
# 
#     chunk_a = run_a / "replay" / "chunk_0"
#     chunk_b = run_b / "replay" / "chunk_0"
#     footer_a_path = chunk_a / "chunk_footer.json"
#     footer_b_path = chunk_b / "chunk_footer.json"
#     if not footer_a_path.exists():
#         print(f"Missing: {footer_a_path}", file=sys.stderr)
#         sys.exit(1)
#     if not footer_b_path.exists():
#         print(f"Missing: {footer_b_path}", file=sys.stderr)
#         sys.exit(1)
# 
#     footer_a = _load_json(footer_a_path)
#     footer_b = _load_json(footer_b_path)
#     capsule_a = footer_a.get("exit_tuning_capsule")
#     capsule_b = footer_b.get("exit_tuning_capsule")
#     if isinstance(capsule_a, dict):
#         capsule_a = {k: v for k, v in capsule_a.items()}
#     else:
#         capsule_a = None
#     if isinstance(capsule_b, dict):
#         capsule_b = {k: v for k, v in capsule_b.items()}
#     else:
#         capsule_b = None
# 
#     exits_a = _load_exits_jsonl(chunk_a)
#     exits_b = _load_exits_jsonl(chunk_b)
# 
#     reason_counts_a = Counter(r.get("reason") or "" for r in exits_a)
#     reason_counts_b = Counter(r.get("reason") or "" for r in exits_b)
# 
#     print(f"RUN_A_DIR={run_a}")
#     print(f"RUN_B_DIR={run_b}")
#     print("RUN_A_EXIT_TUNING_CAPSULE=" + (json.dumps(capsule_a, sort_keys=True) if capsule_a else ""))
#     print("RUN_B_EXIT_TUNING_CAPSULE=" + (json.dumps(capsule_b, sort_keys=True) if capsule_b else ""))
#     print("EXIT_REASON_COUNTS_A=" + json.dumps(dict(reason_counts_a), sort_keys=True))
#     print("EXIT_REASON_COUNTS_B=" + json.dumps(dict(reason_counts_b), sort_keys=True))
# 
#     trail_a = [r for r in exits_a if (r.get("reason") or "") == "MASTER_TRAIL"]
#     trail_b = [r for r in exits_b if (r.get("reason") or "") == "MASTER_TRAIL"]
# 
#     def _bars_held_list(rows: List[dict]) -> List[Optional[float]]:
#         return [r.get("bars_held") if r.get("bars_held") is not None else None for r in rows]
# 
#     def _mfe_bps_list(rows: List[dict]) -> List[Optional[float]]:
#         return [r.get("mfe_bps") for r in rows]
# 
#     def _pnl_bps_list(rows: List[dict]) -> List[Optional[float]]:
#         return [r.get("pnl_bps") for r in rows]
# 
#     bars_a = [x for x in _bars_held_list(trail_a) if x is not None]
#     bars_b = [x for x in _bars_held_list(trail_b) if x is not None]
#     mfe_a = [x for x in _mfe_bps_list(trail_a) if x is not None]
#     mfe_b = [x for x in _mfe_bps_list(trail_b) if x is not None]
#     pnl_a = [x for x in _pnl_bps_list(trail_a) if x is not None]
#     pnl_b = [x for x in _pnl_bps_list(trail_b) if x is not None]
# 
#     def _fmt(v: Optional[float]) -> str:
#         if v is None:
#             return ""
#         return str(v)
# 
#     # Bars held stats (MASTER_TRAIL only)
#     for label, vals in [("A", bars_a), ("B", bars_b)]:
#         mn, p50, p90, mx = _stats([float(x) for x in vals] if vals else [])
#         print(f"{label}_MIN_BARS_HELD={_fmt(mn)}")
#         print(f"{label}_P50_BARS_HELD={_fmt(p50)}")
#         print(f"{label}_P90_BARS_HELD={_fmt(p90)}")
#         print(f"{label}_MAX_BARS_HELD={_fmt(mx)}")
# 
#     # MFE bps at exit (MASTER_TRAIL only)
#     for label, vals in [("A", mfe_a), ("B", mfe_b)]:
#         mn, p50, p90, mx = _stats([float(x) for x in vals] if vals else [])
#         print(f"{label}_MIN_MFE_BPS_AT_EXIT={_fmt(mn)}")
#         print(f"{label}_P50_MFE_BPS_AT_EXIT={_fmt(p50)}")
#         print(f"{label}_P90_MFE_BPS_AT_EXIT={_fmt(p90)}")
#         print(f"{label}_MAX_MFE_BPS_AT_EXIT={_fmt(mx)}")
# 
#     # PNL bps at exit (MASTER_TRAIL only)
#     for label, vals in [("A", pnl_a), ("B", pnl_b)]:
#         mn, p50, p90, mx = _stats([float(x) for x in vals] if vals else [])
#         print(f"{label}_MIN_PNL_BPS_AT_EXIT={_fmt(mn)}")
#         print(f"{label}_P50_PNL_BPS_AT_EXIT={_fmt(p50)}")
#         print(f"{label}_P90_PNL_BPS_AT_EXIT={_fmt(p90)}")
#         print(f"{label}_MAX_PNL_BPS_AT_EXIT={_fmt(mx)}")
# 
#     # Guardrail tripwire counts
#     min_hold_a = int(capsule_a.get("min_hold_bars", 0)) if capsule_a else 0
#     min_hold_b = int(capsule_b.get("min_hold_bars", 0)) if capsule_b else 0
#     min_mfe_bps_a = float(capsule_a.get("min_mfe_bps_to_arm", 0)) if capsule_a else 0.0
#     min_mfe_bps_b = float(capsule_b.get("min_mfe_bps_to_arm", 0)) if capsule_b else 0.0
# 
#     count_bars_violation = 0
#     for r in trail_a:
#         bh = r.get("bars_held")
#         if bh is not None and bh < min_hold_a:
#             count_bars_violation += 1
#     for r in trail_b:
#         bh = r.get("bars_held")
#         if bh is not None and bh < min_hold_b:
#             count_bars_violation += 1
# 
#     count_mfe_violation = 0
#     for r in trail_a:
#         mfe = r.get("mfe_bps")
#         if mfe is not None and mfe < min_mfe_bps_a:
#             count_mfe_violation += 1
#     for r in trail_b:
#         mfe = r.get("mfe_bps")
#         if mfe is not None and mfe < min_mfe_bps_b:
#             count_mfe_violation += 1
# 
#     print(f"COUNT_TRAIL_WITH_BARS_HELD_LT_MIN_HOLD_BARS={count_bars_violation}")
#     print(f"COUNT_TRAIL_WITH_MFE_BPS_LT_MIN_MFE_BPS_TO_ARM={count_mfe_violation}")
# 
#     # VERDICT
#     verdict = "NO_CHANGE_DETECTED"
#     if count_bars_violation > 0 or count_mfe_violation > 0:
#         verdict = "GUARDRAIL_VIOLATION"
#     else:
#         bars_same = (bars_a == bars_b) or (
#             _stats([float(x) for x in bars_a]) == _stats([float(x) for x in bars_b])
#             if bars_a and bars_b
#             else (not bars_a and not bars_b)
#         )
#         pnl_same = (pnl_a == pnl_b) or (
#             _stats([float(x) for x in pnl_a]) == _stats([float(x) for x in pnl_b])
#             if pnl_a and pnl_b
#             else (not pnl_a and not pnl_b)
#         )
#         if not bars_same:
#             verdict = "TIMING_SHIFT_DETECTED"
#         elif not pnl_same:
#             verdict = "PNL_SHIFT_DETECTED"
#         elif dict(reason_counts_a) != dict(reason_counts_b):
#             verdict = "TIMING_SHIFT_DETECTED"
# 
#     print(f"VERDICT={verdict}")
#     return 0 if verdict == "NO_CHANGE_DETECTED" or verdict == "TIMING_SHIFT_DETECTED" or verdict == "PNL_SHIFT_DETECTED" else 1
# 
# 
# if __name__ == "__main__":
#     sys.exit(main())
# 
#     """
