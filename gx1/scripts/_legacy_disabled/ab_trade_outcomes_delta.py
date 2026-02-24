#!/home/andre2/venvs/gx1/bin/python
raise RuntimeError("LEGACY_DISABLED: Use gx1/scripts/README_TRUTH_XGB.md (TRUTH XGB lane only).")
# -*- coding: utf-8 -*-
"""
TRUTH-grade A/B trade-outcomes delta analyzer for Exit ML runs.

Deterministic, no network, no legacy replay imports. Works with best-effort
when optional columns are missing; use --strict to hard-fail on missing key columns.

Outputs: AB_TRADE_DELTA.md, AB_TRADE_DELTA.csv, AB_TRADE_DELTA_META.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Hard rules (SSoT)
GX1_DATA = "/home/andre2/GX1_DATA"
AB_OUTPUT_ROOT = "/home/andre2/GX1_DATA/reports/ab_fullyear_2025_exit_ml"

# Canonical column names and fallback source columns (first present wins)
CANONICAL_PNL_BPS = ["pnl_bps", "pnl_bps_at_exit", "pnl_bps_close"]
CANONICAL_BARS_HELD = ["duration_bars", "bars_held", "hold_bars", "n_bars_held"]
CANONICAL_EXIT_REASON = ["exit_reason", "exit_type_reason", "reason"]
ENTRY_TS_CANDIDATES = ["entry_time", "open_ts", "entry_ts"]
EXIT_TS_CANDIDATES = ["exit_time", "close_ts", "exit_ts"]
SIDE_CANDIDATES = ["side", "direction"]
ENTRY_PRICE_CANDIDATES = ["entry_price", "open_price", "price_at_entry"]
EXIT_PRICE_CANDIDATES = ["exit_price", "close_price", "price_at_exit"]

# Numeric columns to delta if present in both (canonical name -> candidate list)
NUMERIC_DELTA_CANDIDATES: Dict[str, List[str]] = {
    "pnl_bps": CANONICAL_PNL_BPS,
    "pnl": ["pnl", "pnl_total"],
    "pnl_usd": ["pnl_usd", "pnl_dollars"],
    "mfe_bps": ["mfe_bps"],
    "mae_bps": ["mae_bps"],
    "bars_held": CANONICAL_BARS_HELD,
    "entry_price": ENTRY_PRICE_CANDIDATES,
    "exit_price": EXIT_PRICE_CANDIDATES,
    "mfe_price": ["mfe_price"],
    "mae_price": ["mae_price"],
}


def _utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_path(p: str) -> Path:
    return Path(p).expanduser().resolve()


def _first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _find_merged_parquet(run_dir: Path) -> Path:
    """Find trade_outcomes_*_MERGED.parquet in run root. If multiple, pick newest by mtime; warn."""
    candidates = sorted(run_dir.glob("trade_outcomes_*_MERGED.parquet"), key=lambda x: x.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"[AB_DELTA] No trade_outcomes_*_MERGED.parquet in {run_dir}. "
            "Ensure run dir is a TRUTH run root with merged artifacts."
        )
    if len(candidates) > 1:
        chosen = candidates[-1]
        print(
            f"[AB_DELTA] Multiple parquets in {run_dir}; using newest: {chosen.name}",
            file=sys.stderr,
        )
        return chosen
    return candidates[0]


def _build_join_key_composite(df: pd.DataFrame, suffix: str) -> pd.Series:
    """Build composite key series: entry_ts|side|entry_price|exit_ts|exit_price with NA for missing."""
    entry_col = _first_present(df, ENTRY_TS_CANDIDATES)
    exit_col = _first_present(df, EXIT_TS_CANDIDATES)
    side_col = _first_present(df, SIDE_CANDIDATES)
    entry_price_col = _first_present(df, ENTRY_PRICE_CANDIDATES)
    exit_price_col = _first_present(df, EXIT_PRICE_CANDIDATES)

    def row_key(row: pd.Series) -> str:
        parts = [
            str(row[entry_col]) if entry_col else "NA",
            str(row[side_col]) if side_col else "NA",
            str(row[entry_price_col]) if entry_price_col else "NA",
            str(row[exit_col]) if exit_col else "NA",
            str(row[exit_price_col]) if exit_price_col else "NA",
        ]
        return "|".join(parts)

    return df.apply(row_key, axis=1)


def _resolve_canonical_columns(df: pd.DataFrame, canonical_to_candidates: Dict[str, List[str]]) -> Dict[str, str]:
    """Return dict canonical_name -> source_column (first present)."""
    out: Dict[str, str] = {}
    for canonical, candidates in canonical_to_candidates.items():
        src = _first_present(df, candidates)
        if src is not None:
            out[canonical] = src
    return out


def _load_and_join(
    path_a: Path,
    path_b: Path,
    strict: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, Dict[str, Any], Dict[str, str], Dict[str, str]]:
    """
    Load both parquets, determine join key, inner join. Return (joined_df, df_a, df_b, join_strategy, meta, map_a, map_b).
    """
    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    join_key_col: Optional[str] = None
    key_col_a: Optional[str] = None
    key_col_b: Optional[str] = None

    # Prefer trade_id / trade_uid only if they will match across runs (same values in both).
    # In A/B runs, trade_id and trade_uid often embed run_id (A_... vs B_...) so they don't match.
    # Try composite first when both have entry_ts and exit_ts; else try trade_uid then trade_id.
    has_entry = _first_present(df_a, ENTRY_TS_CANDIDATES) and _first_present(df_b, ENTRY_TS_CANDIDATES)
    has_exit = _first_present(df_a, EXIT_TS_CANDIDATES) and _first_present(df_b, EXIT_TS_CANDIDATES)
    if has_entry and has_exit:
        join_key_col = None  # use composite
    elif "trade_uid" in df_a.columns and "trade_uid" in df_b.columns:
        join_key_col = "trade_uid"
        key_col_a = "trade_uid"
        key_col_b = "trade_uid"
    elif "trade_id" in df_a.columns and "trade_id" in df_b.columns:
        join_key_col = "trade_id"
        key_col_a = "trade_id"
        key_col_b = "trade_id"

    if join_key_col is not None:
        join_strategy = join_key_col
        df_a = df_a.copy()
        df_b = df_b.copy()
        df_a["_join_key"] = df_a[key_col_a].astype(str)
        df_b["_join_key"] = df_b[key_col_b].astype(str)
    else:
        join_strategy = "composite"
        required_for_composite = []
        for name, cands in [
            ("entry_ts", ENTRY_TS_CANDIDATES),
            ("exit_ts", EXIT_TS_CANDIDATES),
        ]:
            if _first_present(df_a, cands) is None or _first_present(df_b, cands) is None:
                required_for_composite.append(name)
        if strict and required_for_composite:
            raise ValueError(
                f"[AB_DELTA] --strict: composite key requires entry_ts and exit_ts in both. "
                f"Missing in at least one: {required_for_composite}. "
                f"A columns: {list(df_a.columns)}. B columns: {list(df_b.columns)}."
            )
        df_a = df_a.copy()
        df_b = df_b.copy()
        df_a["_join_key"] = _build_join_key_composite(df_a, "A")
        df_b["_join_key"] = _build_join_key_composite(df_b, "B")

    merged = df_a.merge(df_b, on="_join_key", how="inner", suffixes=("_A", "_B"))
    only_a = len(df_a) - merged["_join_key"].nunique()
    only_b = len(df_b) - merged["_join_key"].nunique()
    # Actually only_in_A = rows in A not in merged; only_in_B = rows in B not in merged
    in_a_not_b = set(df_a["_join_key"]) - set(merged["_join_key"])
    in_b_not_a = set(df_b["_join_key"]) - set(merged["_join_key"])
    only_in_a = len(in_a_not_b)
    only_in_b = len(in_b_not_a)
    matched = len(merged)

    meta = {
        "join_strategy": join_strategy,
        "n_a": int(len(df_a)),
        "n_b": int(len(df_b)),
        "matched": matched,
        "only_in_A": only_in_a,
        "only_in_B": only_in_b,
    }

    map_a = _resolve_canonical_columns(df_a, {k: v for k, v in NUMERIC_DELTA_CANDIDATES.items()})
    map_a["exit_reason"] = _first_present(df_a, CANONICAL_EXIT_REASON) or ""
    map_b = _resolve_canonical_columns(df_b, {k: v for k, v in NUMERIC_DELTA_CANDIDATES.items()})
    map_b["exit_reason"] = _first_present(df_b, CANONICAL_EXIT_REASON) or ""

    return merged, df_a, df_b, join_strategy, meta, map_a, map_b


def _add_deltas(
    merged: pd.DataFrame,
    map_a: Dict[str, str],
    map_b: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Add A_, B_, delta_ columns for each canonical metric present in both. Return (merged, delta_columns_used)."""
    delta_used: Dict[str, str] = {}
    for canonical, candidates in NUMERIC_DELTA_CANDIDATES.items():
        src_a = map_a.get(canonical)
        src_b = map_b.get(canonical)
        if src_a is None or src_b is None:
            continue
        col_a = src_a + "_A" if src_a + "_A" in merged.columns else src_a
        col_b = src_b + "_B" if src_b + "_B" in merged.columns else src_b
        if col_a not in merged.columns or col_b not in merged.columns:
            continue
        a_vals = pd.to_numeric(merged[col_a], errors="coerce")
        b_vals = pd.to_numeric(merged[col_b], errors="coerce")
        merged["A_" + canonical] = a_vals
        merged["B_" + canonical] = b_vals
        merged["delta_" + canonical] = b_vals - a_vals
        delta_used[canonical] = f"{col_a} vs {col_b}"
    return merged, delta_used


def _exit_transition_table(
    merged: pd.DataFrame,
    map_a: Dict[str, str],
    map_b: Dict[str, str],
) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
    """If exit_reason in both: (counts A->B, mean delta_pnl_bps per transition)."""
    reason_a = map_a.get("exit_reason")
    reason_b = map_b.get("exit_reason")
    if not reason_a or not reason_b:
        return None
    col_a = reason_a + "_A" if reason_a + "_A" in merged.columns else reason_a
    col_b = reason_b + "_B" if reason_b + "_B" in merged.columns else reason_b
    if col_a not in merged.columns or col_b not in merged.columns:
        return None
    counts = merged.groupby([col_a, col_b]).size().reset_index(name="count")
    counts = counts.rename(columns={col_a: "A_reason", col_b: "B_reason"})
    mean_pnl = None
    if "delta_pnl_bps" in merged.columns:
        mean_pnl = (
            merged.groupby([col_a, col_b])["delta_pnl_bps"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={col_a: "A_reason", col_b: "B_reason", "mean": "mean_delta_pnl_bps", "count": "n"})
        )
    return (counts, mean_pnl)


def _delta_stats(series: pd.Series) -> Dict[str, float]:
    """Summary stats for a delta series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {}
    return {
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p10": float(s.quantile(0.10)),
        "p90": float(s.quantile(0.90)),
        "min": float(s.min()),
        "max": float(s.max()),
        "count": int(len(s)),
    }


def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Render a simple markdown table without pandas.to_markdown."""
    if not headers or not rows:
        return ""
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(c)))
    sep = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    line = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
    out = [sep, line]
    for row in rows:
        out.append("| " + " | ".join(str(row[i]).ljust(col_widths[i]) if i < len(row) else "" for i in range(len(headers))) + " |")
    return "\n".join(out)


def _atomic_write(path: Path, content: str, encoding: str = "utf-8") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding=encoding)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def run_analyzer(
    run_a: Path,
    run_b: Path,
    out_dir: Path,
    ab_run_id: str,
    max_rows: int,
    strict: bool,
) -> Dict[str, Any]:
    """
    Run the A/B trade-outcomes delta analyzer. Writes MD, CSV, META to out_dir/ab_run_id/.
    Returns metadata dict (for smoke / programmatic use).
    """
    path_a = _find_merged_parquet(run_a)
    path_b = _find_merged_parquet(run_b)

    merged, df_a, df_b, join_strategy, meta, map_a, map_b = _load_and_join(path_a, path_b, strict)
    merged, delta_used = _add_deltas(merged, map_a, map_b)
    transition = _exit_transition_table(merged, map_a, map_b)

    meta["run_a"] = str(run_a)
    meta["run_b"] = str(run_b)
    meta["path_a"] = str(path_a)
    meta["path_b"] = str(path_b)
    meta["column_mapping_a"] = map_a
    meta["column_mapping_b"] = map_b
    meta["delta_columns_used"] = delta_used

    out_sub = out_dir / ab_run_id
    out_sub.mkdir(parents=True, exist_ok=True)

    # CSV: full joined table
    csv_path = out_sub / "AB_TRADE_DELTA.csv"
    merged.to_csv(csv_path.with_suffix(csv_path.suffix + ".tmp"), index=False)
    os.replace(csv_path.with_suffix(csv_path.suffix + ".tmp"), csv_path)

    # Delta stats for report
    delta_pnl_stats: Dict[str, float] = {}
    delta_bars_stats: Dict[str, float] = {}
    if "delta_pnl_bps" in merged.columns:
        delta_pnl_stats = _delta_stats(merged["delta_pnl_bps"])
    elif "delta_pnl" in merged.columns:
        delta_pnl_stats = _delta_stats(merged["delta_pnl"])
    if "delta_bars_held" in merged.columns:
        delta_bars_stats = _delta_stats(merged["delta_bars_held"])
    meta["delta_pnl_bps_stats"] = delta_pnl_stats
    meta["delta_bars_held_stats"] = delta_bars_stats

    # Top winners / worst by delta_pnl_bps (or delta_pnl)
    sort_col = "delta_pnl_bps" if "delta_pnl_bps" in merged.columns else "delta_pnl"
    if sort_col not in merged.columns:
        sort_col = ""
    top_winners: List[List[Any]] = []
    top_losers: List[List[Any]] = []
    if sort_col:
        merged_sorted = merged.sort_values(sort_col, ascending=False)
        cols_show = ["_join_key", sort_col]
        for c in ["A_pnl_bps", "B_pnl_bps", "pnl_bps_A", "pnl_bps_B", "exit_reason_A", "exit_reason_B"]:
            if c in merged.columns and c not in cols_show:
                cols_show.append(c)
        cols_show = [c for c in cols_show if c in merged_sorted.columns][:8]
        winners = merged_sorted.head(max_rows)[cols_show]
        losers = merged_sorted.tail(max_rows)[cols_show].iloc[::-1]
        top_winners = winners.values.tolist()
        top_losers = losers.values.tolist()
        meta["top_winners_columns"] = cols_show
        meta["top_losers_columns"] = cols_show

    # Build markdown
    md_lines = [
        "# A/B Trade Outcomes Delta",
        "",
        f"**AB_RUN_ID:** {ab_run_id}",
        "",
        "## Run dirs",
        "",
        f"- **A (baseline):** `{run_a}`",
        f"- **B (ML-frozen):** `{run_b}`",
        "",
        "## Join strategy",
        "",
        f"- Key: **{join_strategy}**",
        "",
        "## Coverage",
        "",
        _md_table(
            ["Metric", "Value"],
            [
                ["Trades in A", meta["n_a"]],
                ["Trades in B", meta["n_b"]],
                ["Matched", meta["matched"]],
                ["Only in A", meta["only_in_A"]],
                ["Only in B", meta["only_in_B"]],
            ],
        ),
        "",
        "## Delta stats",
        "",
    ]

    if delta_pnl_stats:
        md_lines.append("### delta_pnl_bps (B − A)")
        md_lines.append("")
        md_lines.append(_md_table(["Stat", "Value"], [[k, v] for k, v in delta_pnl_stats.items()]))
        md_lines.append("")
    if delta_bars_stats:
        md_lines.append("### delta_bars_held (B − A)")
        md_lines.append("")
        md_lines.append(_md_table(["Stat", "Value"], [[k, v] for k, v in delta_bars_stats.items()]))
        md_lines.append("")

    if transition is not None:
        counts_df, mean_pnl_df = transition
        md_lines.append("## Exit reason transitions (A → B)")
        md_lines.append("")
        md_lines.append(_md_table(list(counts_df.columns), counts_df.values.tolist()))
        md_lines.append("")
        if mean_pnl_df is not None and len(mean_pnl_df) > 0:
            md_lines.append("### Mean delta_pnl_bps by transition")
            md_lines.append("")
            md_lines.append(_md_table(list(mean_pnl_df.columns), mean_pnl_df.values.tolist()))
            md_lines.append("")

    if top_winners:
        md_lines.append("## Top winners by " + sort_col)
        md_lines.append("")
        md_lines.append(_md_table(meta.get("top_winners_columns", []), top_winners))
        md_lines.append("")
    if top_losers:
        md_lines.append("## Top losers by " + sort_col)
        md_lines.append("")
        md_lines.append(_md_table(meta.get("top_losers_columns", []), top_losers))
        md_lines.append("")

    md_path = out_sub / "AB_TRADE_DELTA.md"
    _atomic_write(md_path, "\n".join(md_lines))

    meta_path = out_sub / "AB_TRADE_DELTA_META.json"
    _atomic_write_json(meta_path, meta)

    # Console summary
    print(f"[AB_DELTA] Join: {join_strategy} | A={meta['n_a']} B={meta['n_b']} matched={meta['matched']} only_A={meta['only_in_A']} only_B={meta['only_in_B']}")
    if delta_pnl_stats:
        print(f"[AB_DELTA] delta_pnl_bps: mean={delta_pnl_stats.get('mean', 'n/a')} median={delta_pnl_stats.get('median', 'n/a')} min={delta_pnl_stats.get('min', 'n/a')} max={delta_pnl_stats.get('max', 'n/a')}")
    print(f"[AB_DELTA] Output: {out_sub}")
    print(f"[AB_DELTA]   {md_path.name}")
    print(f"[AB_DELTA]   {csv_path.name}")
    print(f"[AB_DELTA]   {meta_path.name}")

    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="TRUTH A/B trade-outcomes delta analyzer (Exit ML runs).")
    parser.add_argument("--run-a", type=str, required=True, help="Run A (baseline) root dir")
    parser.add_argument("--run-b", type=str, required=True, help="Run B (ML-frozen) root dir")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=AB_OUTPUT_ROOT,
        help=f"Output root (default: {AB_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--ab-run-id",
        type=str,
        default="",
        help="AB run id for output subdir (default: AB_TRADE_DELTA_<utc_ts>)",
    )
    parser.add_argument("--max-rows", type=int, default=50, help="Max rows in top/worst tables (default 50)")
    parser.add_argument("--strict", action="store_true", help="Hard-fail on missing key columns")
    args = parser.parse_args()

    run_a = _resolve_path(args.run_a)
    run_b = _resolve_path(args.run_b)
    out_dir = _resolve_path(args.out_dir)
    ab_run_id = args.ab_run_id or f"AB_TRADE_DELTA_{_utc_ts_compact()}"

    if not run_a.is_dir():
        print(f"[AB_DELTA] Run A not a directory: {run_a}", file=sys.stderr)
        return 1
    if not run_b.is_dir():
        print(f"[AB_DELTA] Run B not a directory: {run_b}", file=sys.stderr)
        return 1

    try:
        run_analyzer(run_a, run_b, out_dir, ab_run_id, args.max_rows, args.strict)
        return 0
    except Exception as e:
        print(f"[AB_DELTA] {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
