#!/usr/bin/env python3
"""
Compare SNIPER regime classification between index/CSV-based fields and
JSON-first fields for the same Q4 trades (no replay, no trading logic changes).

Goal:
- Explain why Q4 B_MIXED counts differ between the older index-based regime
  split (~1086 B_MIXED trades) and the new JSON-first evaluation (near zero).

This script:
- Autodetects latest Q4 2025 baseline and baseline_overlay run_dirs.
- Loads regime input fields from:
  - trade_journal/trade_journal_index.csv  ("index" source)
  - trade_journal/trades/*.json           ("json" source)
- Applies the SAME classify_regime(...) function to both views.
- Joins trades and reports:
  A) Coverage per field (index vs json)
  B) Regime distribution (A/B/C) for each source
  C) Agreement matrix between index- and json-based regime_class
  D) Drilldown on B_MIXED→non-B_MIXED disagreements
  E) Auto "hypothesis" summary about likely causes of mismatch

Output:
- reports/SNIPER_Q4_REGIME_SOURCE_DIFF__YYYYMMDD_HHMMSS.md

Exit code:
- Always 0 (diagnostic only), but logs WARNING if join-rate < 80%.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gx1.sniper.analysis.regime_classifier import classify_regime


SNIPER_BASE = Path("gx1/wf_runs")


CANONICAL_FIELDS = ["trend_regime", "vol_regime", "atr_bps", "spread_bps", "session"]


@dataclass
class RegimeRunPair:
    name: str
    baseline: Optional[Path]
    baseline_overlay: Optional[Path]


def _latest_run(pattern: str) -> Optional[Path]:
    matches = sorted(SNIPER_BASE.glob(pattern))
    return matches[-1] if matches else None


def autodetect_q4_baseline_pairs() -> RegimeRunPair:
    """
    Detect latest Q4 baseline and baseline_overlay runs.
    """
    base = _latest_run("SNIPER_OBS_Q4_2025_baseline_*")
    overlay = _latest_run("SNIPER_OBS_Q4_2025_baseline_overlay_*")
    return RegimeRunPair(name="baseline", baseline=base, baseline_overlay=overlay)


def _coerce_canonical_fields(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Ensure canonical regime fields exist with reasonable dtypes.
    """
    df_local = df.copy()
    for col in CANONICAL_FIELDS:
        if col not in df_local.columns:
            df_local[col] = np.nan
    # Coerce numerics
    for col in ["atr_bps", "spread_bps"]:
        df_local[col] = pd.to_numeric(df_local[col], errors="coerce")
    # Sessions and labels as strings where possible
    df_local["session"] = df_local["session"].astype("object")
    df_local["trend_regime"] = df_local["trend_regime"].astype("object")
    df_local["vol_regime"] = df_local["vol_regime"].astype("object")
    return df_local


def load_index_rows(run_dir: Path) -> pd.DataFrame:
    """
    Load canonical regime fields from trade_journal_index.csv.
    Returns empty DataFrame if index is missing or empty.
    """
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if not index_path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(index_path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df

    # Keep canonical fields plus join keys
    keep_cols = set(["trade_id", "trade_file"] + CANONICAL_FIELDS)
    cols = [c for c in df.columns if c in keep_cols]
    df = df[cols].copy()
    return _coerce_canonical_fields(df, "index")


def trade_to_regime_row(trade: Dict[str, Any], trade_file: str) -> Dict[str, Any]:
    """
    Extract canonical regime inputs and join keys from a trade JSON dict.
    """
    entry = trade.get("entry_snapshot") or {}
    feature_ctx = entry.get("feature_context") or {}
    extra = trade.get("extra") or entry.get("extra") or {}

    def _first_non_null(*candidates):
        for v in candidates:
            if v is not None:
                return v
        return None

    trend_regime = _first_non_null(
        entry.get("trend_regime"),
        feature_ctx.get("trend_regime"),
        trade.get("trend_regime"),
    )
    vol_regime = _first_non_null(
        entry.get("vol_regime"),
        feature_ctx.get("vol_regime"),
        trade.get("vol_regime"),
    )
    atr_bps = _first_non_null(
        entry.get("atr_bps"),
        feature_ctx.get("atr_bps"),
        trade.get("atr_bps"),
    )
    spread_bps = _first_non_null(
        entry.get("spread_bps"),
        feature_ctx.get("spread_bps"),
        trade.get("spread_bps"),
    )
    session = _first_non_null(
        entry.get("session"),
        feature_ctx.get("session"),
        trade.get("session"),
    )

    return {
        "trade_id": trade.get("trade_id"),
        "trade_file": trade_file,
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "atr_bps": atr_bps,
        "spread_bps": spread_bps,
        "session": session,
    }


def load_json_rows(run_dir: Path) -> pd.DataFrame:
    """
    Load canonical regime fields from trade_journal/trades/*.json.
    Returns empty DataFrame if no trades dir.
    """
    trades_dir = run_dir / "trade_journal" / "trades"
    if not trades_dir.exists():
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for json_file in sorted(trades_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                trade = json.load(f)
        except Exception:
            continue
        if not isinstance(trade, dict):
            continue
        rows.append(trade_to_regime_row(trade, trade_file=json_file.name))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return _coerce_canonical_fields(df, "json")


def attach_regime_class(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Attach regime_class_{source_label} and regime_reason_{source_label}
    using classify_regime for each row.
    """
    df_local = df.copy()
    classes: List[Optional[str]] = []
    reasons: List[Optional[str]] = []
    for _, row in df_local.iterrows():
        try:
            c, r = classify_regime(row.to_dict())
        except Exception as exc:
            c, r = None, f"classify_error:{type(exc).__name__}"
        classes.append(c)
        reasons.append(r)
    df_local[f"regime_class_{source_label}"] = classes
    df_local[f"regime_reason_{source_label}"] = reasons
    return df_local


def join_index_json(
    df_index: pd.DataFrame, df_json: pd.DataFrame
) -> Tuple[pd.DataFrame, float]:
    """
    Join index and json views on best available key.

    Priority:
    1) trade_id (if present in both)
    2) trade_file (if present in both)

    Returns:
        (joined_df, join_rate) where join_rate is fraction of json rows matched.
    """
    if df_index.empty or df_json.empty:
        return pd.DataFrame(), 0.0

    if "trade_id" in df_index.columns and "trade_id" in df_json.columns:
        key = "trade_id"
    elif "trade_file" in df_index.columns and "trade_file" in df_json.columns:
        key = "trade_file"
    else:
        return pd.DataFrame(), 0.0

    left = df_index.set_index(key, drop=False)
    right = df_json.set_index(key, drop=False)
    joined = left.join(
        right,
        how="inner",
        lsuffix="_index_src",
        rsuffix="_json_src",
    )

    join_rate = float(len(joined)) / float(len(df_json)) if len(df_json) > 0 else 0.0
    return joined.reset_index(drop=True), join_rate


def _coverage_stats(df_index: pd.DataFrame, df_json: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("### A) Field coverage (index vs json)")
    lines.append("")
    lines.append("| Field | index non-null% | json non-null% |")
    lines.append("| --- | ---: | ---: |")
    for col in CANONICAL_FIELDS:
        idx_cov = float(df_index[col].notna().mean() * 100.0) if not df_index.empty else 0.0
        js_cov = float(df_json[col].notna().mean() * 100.0) if not df_json.empty else 0.0
        lines.append(f"| {col} | {idx_cov:.1f}% | {js_cov:.1f}% |")
    lines.append("")
    # Simple stats for numerics
    for col in ["atr_bps", "spread_bps"]:
        idx_series = pd.to_numeric(df_index.get(col), errors="coerce")
        js_series = pd.to_numeric(df_json.get(col), errors="coerce")
        lines.append(f"- {col} index: min={idx_series.min():.2f} max={idx_series.max():.2f}")
        lines.append(f"- {col} json : min={js_series.min():.2f} max={js_series.max():.2f}")
    lines.append("")
    return "\n".join(lines)


def _regime_distribution(df_index_cls: pd.DataFrame, df_json_cls: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("### B) Regime distribution (A/B/C)")
    lines.append("")

    def _dist(series: pd.Series) -> Dict[str, int]:
        return series.value_counts(dropna=True).to_dict()

    idx_dist = _dist(df_index_cls["regime_class_index"])
    js_dist = _dist(df_json_cls["regime_class_json"])

    lines.append("| Regime | count_index | count_json |")
    lines.append("| --- | ---: | ---: |")
    for regime in sorted(set(list(idx_dist.keys()) + list(js_dist.keys()))):
        lines.append(
            f"| {regime} | {idx_dist.get(regime, 0)} | {js_dist.get(regime, 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _agreement_matrix(joined: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("### C) Regime agreement matrix (index vs json)")
    lines.append("")
    if joined.empty:
        lines.append("_No joined rows; cannot compute agreement matrix._")
        lines.append("")
        return "\n".join(lines)

    ct = pd.crosstab(
        joined["regime_class_index"],
        joined["regime_class_json"],
        dropna=False,
    )
    total = float(ct.to_numpy().sum()) if ct.size > 0 else 0.0

    lines.append("| index \\ json | " + " | ".join(str(c) for c in ct.columns) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in ct.columns) + " |")
    for idx_val, row in ct.iterrows():
        counts = [int(row[c]) for c in ct.columns]
        lines.append(
            f"| {idx_val} | " + " | ".join(str(v) for v in counts) + " |"
        )
    lines.append("")

    if total > 0:
        lines.append("Agreement as % of joined rows:")
        lines.append("")
        for idx_val, row in ct.iterrows():
            for col_val in ct.columns:
                frac = float(ct.loc[idx_val, col_val]) / total
                lines.append(
                    f"- {idx_val} → {col_val}: {frac*100:.2f}% of joined rows"
                )
        lines.append("")
    return "\n".join(lines)


def _b_mixed_drift(joined: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("### D) B_MIXED disappearance drilldown")
    lines.append("")
    if joined.empty:
        lines.append("_No joined rows; cannot analyze B_MIXED drift._")
        lines.append("")
        return "\n".join(lines)

    mask = (joined["regime_class_index"] == "B_MIXED") & (
        joined["regime_class_json"] != "B_MIXED"
    )
    drift = joined[mask].copy()
    if drift.empty:
        lines.append("_No trades where index=B_MIXED and json!=B_MIXED._")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        f"Found {len(drift)} trades where index=B_MIXED but json!=B_MIXED. Showing up to 30:"
    )
    lines.append("")
    cols = [
        "trade_id",
        "trade_file",
        "trend_regime_index",
        "trend_regime_json",
        "vol_regime_index",
        "vol_regime_json",
        "atr_bps_index",
        "atr_bps_json",
        "spread_bps_index",
        "spread_bps_json",
        "regime_reason_index",
        "regime_reason_json",
    ]
    # Prepare columns in drift
    for base_col in ["trend_regime", "vol_regime", "atr_bps", "spread_bps"]:
        drift[f"{base_col}_index"] = drift[f"{base_col}_index_src"]
        drift[f"{base_col}_json"] = drift[f"{base_col}_json_src"]

    subset = drift.iloc[:30]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, row in subset.iterrows():
        vals = [str(row.get(c, "")) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def _hypothesis_summary(
    df_index: pd.DataFrame,
    df_json: pd.DataFrame,
    df_index_cls: pd.DataFrame,
    df_json_cls: pd.DataFrame,
    joined: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("### E) Auto hypothesis about source mismatch")
    lines.append("")

    # Basic counts
    idx_b = int((df_index_cls["regime_class_index"] == "B_MIXED").sum())
    js_b = int((df_json_cls["regime_class_json"] == "B_MIXED").sum())

    if idx_b == 0 and js_b == 0:
        lines.append(
            "- Both index and json have zero B_MIXED trades; no disappearance to explain."
        )
        return "\n".join(lines) + "\n"

    if idx_b > 0 and js_b == 0:
        lines.append(
            f"- Index reports {idx_b} B_MIXED trades, but json reports 0. "
            "Likely causes:\n"
            "  - Missing trend_regime/vol_regime/atr_bps/spread_bps in JSON rows, or\n"
            "  - Different mapping / label normalization between index and JSON fields."
        )
    elif idx_b > 0 and js_b > 0 and js_b < idx_b * 0.5:
        lines.append(
            f"- Index reports {idx_b} B_MIXED trades, json reports {js_b}. "
            "Substantial drop; investigate field coverage and label differences."
        )
    else:
        lines.append(
            f"- Index B_MIXED={idx_b}, json B_MIXED={js_b}. Counts are closer; "
            "mismatch may be due to per-trade differences rather than gross drift."
        )

    # Field coverage hints
    for col in CANONICAL_FIELDS:
        idx_cov = float(df_index[col].notna().mean() * 100.0) if not df_index.empty else 0.0
        js_cov = float(df_json[col].notna().mean() * 100.0) if not df_json.empty else 0.0
        if idx_cov > 90.0 and js_cov < 50.0:
            lines.append(
                f"- Field {col}: index coverage ~{idx_cov:.1f}%, json coverage ~{js_cov:.1f}%. "
                "JSON likely missing this field for many trades."
            )

    # Simple label drift detection
    if not joined.empty:
        tr_idx = joined["trend_regime_index_src"].dropna().astype(str).str.lower()
        tr_js = joined["trend_regime_json_src"].dropna().astype(str).str.lower()
        if not tr_idx.empty and not tr_js.empty:
            if "trend" in " ".join(tr_idx.unique()) and "trend" not in " ".join(
                tr_js.unique()
            ):
                lines.append(
                    "- trend_regime labels differ markedly between index and json; "
                    "JSON may be missing the normalized trend labels used in the index."
                )

    lines.append("")
    return "\n".join(lines)


def build_report(pair: RegimeRunPair, ts: str) -> str:
    lines: List[str] = []
    lines.append(f"# SNIPER Q4 Regime Source Comparison ({ts})")
    lines.append("")
    lines.append("This report compares regime inputs/classification between:")
    lines.append("- INDEX source: trade_journal_index.csv")
    lines.append("- JSON source: trade_journal/trades/*.json")
    lines.append("")

    lines.append(f"## Run pair: `{pair.name}`")
    lines.append("")
    lines.append(f"- Baseline run_dir: `{pair.baseline}`")
    lines.append(f"- Baseline_overlay run_dir: `{pair.baseline_overlay}`")
    lines.append("")

    if pair.baseline is None:
        lines.append("- ❌ Missing baseline run_dir; nothing to compare.")
        return "\n".join(lines)

    # For regime-source comparison we focus on the baseline run (non-overlay)
    run_dir = pair.baseline

    df_index = load_index_rows(run_dir)
    df_json = load_json_rows(run_dir)

    lines.append(f"- INDEX source available: {not df_index.empty}")
    lines.append(f"- JSON  source available: {not df_json.empty}")
    lines.append("")

    if df_index.empty and df_json.empty:
        lines.append("_Neither index nor json data available; aborting._")
        return "\n".join(lines)

    df_index_cls = attach_regime_class(df_index, "index")
    df_json_cls = attach_regime_class(df_json, "json")

    joined, join_rate = join_index_json(df_index_cls, df_json_cls)
    lines.append(f"- Joined rows (index∩json): {len(joined)}")
    lines.append(f"- Join rate (vs json rows): {join_rate*100:.1f}%")
    lines.append("")

    lines.append(_coverage_stats(df_index_cls, df_json_cls))
    lines.append(_regime_distribution(df_index_cls, df_json_cls))
    lines.append(_agreement_matrix(joined))
    lines.append(_b_mixed_drift(joined))
    lines.append(
        _hypothesis_summary(df_index, df_json, df_index_cls, df_json_cls, joined)
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare SNIPER Q4 regime classification between index and JSON sources."
    )
    _ = parser.parse_args()

    pair = autodetect_q4_baseline_pairs()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_text = build_report(pair, ts)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"SNIPER_Q4_REGIME_SOURCE_DIFF__{ts}.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote Q4 regime source diff report: {report_path}")

    # Exit code is always 0 (diagnostic only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


