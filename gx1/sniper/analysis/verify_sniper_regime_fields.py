#!/usr/bin/env python3
"""
Verify that SNIPER trade journals contain the FARM-style regime and microstructure fields
needed for offline regime classification (no replay, no metric recomputation).

This script:
- Autodetects the latest SNIPER OBS run per (quarter, variant) or uses --run-dirs.
- Reads trade_journal_index.csv for each run_dir.
- Audits presence and coverage of:
  - trend_regime
  - vol_regime
  - atr_bps
  - spread_bps
  - session
  - range_pos, distance_to_range, range_edge_dist_atr
  - router_decision (if present)
- Writes a markdown report under reports/:
  SNIPER_2025_REGIME_FIELDS_AUDIT__YYYYMMDD_HHMMSS.md
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


EXPECTED_FIELDS = [
    "trend_regime",
    "vol_regime",
    "atr_bps",
    "spread_bps",
    "session",
    "range_pos",
    "distance_to_range",
    "range_edge_dist_atr",
    "router_decision",
]


def _autodetect_run_dirs() -> List[Path]:
    """
    Autodetect latest SNIPER OBS run per (quarter, variant) from gx1/wf_runs.
    Pattern: SNIPER_OBS_Qx_2025_<variant>_YYYYMMDD_HHMMSS
    """
    base = Path("gx1/wf_runs")
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    variants = ["baseline", "guarded"]
    run_dirs: List[Path] = []

    for q in quarters:
        for v in variants:
            pattern = f"SNIPER_OBS_{q}_2025_{v}_*"
            matches = sorted(base.glob(pattern))
            if matches:
                run_dirs.append(matches[-1])
    return run_dirs


def _coverage(df: pd.DataFrame, col: str) -> float:
    """Return fraction of rows where column is non-null and non-empty (0–1)."""
    if col not in df.columns or df.empty:
        return 0.0
    s = df[col]
    non_null = s.notna()
    if s.dtype == object:
        non_null &= s.astype(str).str.strip().ne("")
    return float(non_null.mean())


def audit_run_dir(run_dir: Path) -> Tuple[Path, List[str], Dict[str, float]]:
    """
    Audit a single SNIPER run_dir.

    Returns:
        (run_dir, columns, coverage_per_field)
    """
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if not index_path.exists():
        return run_dir, [], {f: 0.0 for f in EXPECTED_FIELDS}

    df = pd.read_csv(index_path)
    cols = list(df.columns)
    cov: Dict[str, float] = {}
    for field in EXPECTED_FIELDS:
        cov[field] = _coverage(df, field)
    return run_dir, cols, cov


def build_report(
    audits: List[Tuple[Path, List[str], Dict[str, float]]],
    ts: str,
) -> str:
    """Build markdown report text from audit results."""
    lines: List[str] = []
    lines.append(f"# SNIPER 2025 Regime Fields Audit ({ts})")
    lines.append("")
    lines.append("This report verifies that SNIPER trade journals contain the FARM-style")
    lines.append("regime and microstructure fields needed for offline regime analysis.")
    lines.append("")

    # List audited run_dirs
    lines.append("## Audited runs")
    lines.append("")
    for run_dir, cols, _ in audits:
        lines.append(f"- `{run_dir}` ({len(cols)} columns in index)")
    if not audits:
        lines.append("- _None detected_")
    lines.append("")

    # Columns (union)
    all_columns = set()
    for _, cols, _ in audits:
        all_columns.update(cols)
    lines.append("## Index columns (union across runs)")
    lines.append("")
    if all_columns:
        for col in sorted(all_columns):
            lines.append(f"- `{col}`")
    else:
        lines.append("- _No columns found (no indexes)_")
    lines.append("")

    # Coverage per field per run
    lines.append("## Field coverage per run")
    lines.append("")
    for run_dir, _, cov in audits:
        lines.append(f"### `{run_dir}`")
        lines.append("")
        if not cov:
            lines.append("- No index found.")
            lines.append("")
            continue
        lines.append("| Field | Coverage |")
        lines.append("| --- | --- |")
        for field in EXPECTED_FIELDS:
            coverage = cov.get(field, 0.0)
            lines.append(f"| `{field}` | {coverage*100:.1f}% |")
        lines.append("")

    # Momentum note
    lines.append("## Momentum / trend proxies")
    lines.append("")
    if "trend_regime" in all_columns:
        lines.append(
            "- `trend_regime` is present and can be used as a primary **momentum / trend proxy**."
        )
    else:
        lines.append(
            "- `trend_regime` is missing; consider using other features (e.g. range or PnL path) as momentum proxy."
        )
    if "vol_regime" in all_columns and "atr_bps" in all_columns:
        lines.append(
            "- `vol_regime` and `atr_bps` are available for volatility/energy classification."
        )
    lines.append("")

    # Overall conclusion
    lines.append("## Conclusion")
    lines.append("")
    # Check if core fields have high coverage in all audited runs
    core_fields = ["trend_regime", "vol_regime", "atr_bps", "spread_bps"]
    ok_core = True
    for _, _, cov in audits:
        if not cov:
            ok_core = False
            break
        for f in core_fields:
            if cov.get(f, 0.0) < 0.8:
                ok_core = False
                break
        if not ok_core:
            break

    if audits and ok_core:
        lines.append(
            "**OK to classify regimes**: `trend_regime`, `vol_regime`, `atr_bps`, "
            "`spread_bps` have high coverage (>= 80%) in all audited runs."
        )
    elif audits:
        lines.append(
            "**Not ready for strict regime classification**: one or more core fields "
            "have low coverage (< 80%). Consider tightening journaling before relying "
            "on regime splits."
        )
    else:
        lines.append(
            "**No SNIPER runs found** under `gx1/wf_runs`; cannot assess regime fields."
        )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify SNIPER trade journal regime fields (no replay)."
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="*",
        default=None,
        help="Explicit SNIPER run directories to audit. "
        "If omitted, autodetect latest Q1–Q4 baseline/guarded runs.",
    )
    args = parser.parse_args()

    if args.run_dirs:
        run_dirs = list(args.run_dirs)
    else:
        run_dirs = _autodetect_run_dirs()

    audits: List[Tuple[Path, List[str], Dict[str, float]]] = []
    for rd in run_dirs:
        run_dir, cols, cov = audit_run_dir(rd)
        audits.append((run_dir, cols, cov))

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_text = build_report(audits, ts)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"SNIPER_2025_REGIME_FIELDS_AUDIT__{ts}.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote regime fields audit: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


