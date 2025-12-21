#!/usr/bin/env python3
"""
Q4 SNIPER overlay evaluation and correctness checks (no replay).

This script:
- Autodetects latest SNIPER OBS run_dirs for:
  - Q4 baseline
  - Q4 baseline_overlay
  - Q4 guarded
  - Q4 guarded_overlay
- Verifies that the SNIPER size overlay is applied **only** for Q4 × B_MIXED trades and
  that overlay metadata is internally consistent.
- Computes simple A/B metrics for Q4:
  - Total Q4: trades, EV/trade, winrate, payoff, p90 loss
  - Q4 × B_MIXED only: samme metrics
- Writes a markdown report:
  reports/SNIPER_Q4_OVERLAY_EVAL__YYYYMMDD_HHMMSS.md

This is purely offline evaluation of existing run_dirs. It does **not** run replay.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from gx1.sniper.analysis.regime_classifier import classify_regime


SNIPER_BASE = Path("gx1/wf_runs")


@dataclass
class RunPair:
    variant_base: str  # "baseline" or "guarded"
    normal: Optional[Path]
    overlay: Optional[Path]


def _latest_run(pattern: str) -> Optional[Path]:
    matches = sorted(SNIPER_BASE.glob(pattern))
    return matches[-1] if matches else None


def autodetect_q4_runs() -> List[RunPair]:
    """
    Autodetect Q4 runs for baseline/guarded and their overlay variants.

    We have two naming patterns per base:
    - SNIPER_OBS_Q4_2025_baseline_*
    - SNIPER_OBS_Q4_2025_baseline_overlay_*
    (similarly for guarded)

    Important: avoid pairing the same run_dir as both normal and overlay.
    """
    pairs: List[RunPair] = []
    for base in ["baseline", "guarded"]:
        pattern = f"SNIPER_OBS_Q4_2025_{base}_*"
        all_candidates = sorted(SNIPER_BASE.glob(pattern))
        normal_candidates = [p for p in all_candidates if "_overlay_" not in p.name]
        overlay_candidates = [p for p in all_candidates if "_overlay_" in p.name]

        normal = normal_candidates[-1] if normal_candidates else None
        overlay = overlay_candidates[-1] if overlay_candidates else None

        pairs.append(RunPair(variant_base=base, normal=normal, overlay=overlay))
    return pairs


def trade_to_row(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a minimal, analysis-friendly row from a trade JSON dict.

    Fields:
    - pnl_bps (float)
    - trend_regime, vol_regime, atr_bps, spread_bps, session
    - overlay_* (from entry_snapshot.sniper_overlay if present)
    """
    entry = trade.get("entry_snapshot") or {}
    exit_summary = trade.get("exit_summary") or entry.get("exit_summary") or {}
    extra = trade.get("extra") or entry.get("extra") or {}
    feature_ctx = entry.get("feature_context") or {}

    # pnl_bps: prefer realized_pnl_bps, then pnl_bps, then top-level fallback
    pnl = exit_summary.get("realized_pnl_bps")
    if pnl is None:
        pnl = exit_summary.get("pnl_bps")
    if pnl is None:
        pnl = trade.get("pnl_bps")

    # Regime inputs (best-effort; None if missing)
    def _g(*keys, default=None):
        cur: Any = trade
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k)
        return cur if cur is not None else default

    trend_regime = entry.get("trend_regime")
    if trend_regime is None:
        trend_regime = feature_ctx.get("trend_regime") or trade.get("trend_regime")

    vol_regime = entry.get("vol_regime")
    if vol_regime is None:
        vol_regime = feature_ctx.get("vol_regime") or trade.get("vol_regime")

    atr_bps = entry.get("atr_bps")
    if atr_bps is None:
        atr_bps = feature_ctx.get("atr_bps") or trade.get("atr_bps")

    spread_bps = entry.get("spread_bps")
    if spread_bps is None:
        spread_bps = feature_ctx.get("spread_bps") or trade.get("spread_bps")

    session = entry.get("session")
    if session is None:
        session = feature_ctx.get("session") or trade.get("session")

    # Overlay metadata
    overlay = entry.get("sniper_overlay") or extra.get("sniper_overlay") or {}
    row: Dict[str, Any] = {
        "trade_id": trade.get("trade_id"),
        "pnl_bps": pnl,
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "atr_bps": atr_bps,
        "spread_bps": spread_bps,
        "session": session,
        "overlay_applied": bool(overlay.get("overlay_applied", False)),
        "overlay_name": overlay.get("overlay_name", ""),
        "overlay_multiplier": overlay.get("multiplier", 1.0),
        "overlay_size_before": overlay.get("size_before_units"),
        "overlay_size_after": overlay.get("size_after_units"),
        "overlay_quarter": overlay.get("quarter"),
        "overlay_regime_class": overlay.get("regime_class"),
        "overlay_regime_reason": overlay.get("regime_reason"),
    }
    return row


def load_trades_from_run(run_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
    """
    Load trades for a run_dir, preferring the merged CSV index when available,
    otherwise falling back to reading all trades/*.json.

    Returns:
        (rows, source_string)
    """
    index_path = run_dir / "trade_journal" / "trade_journal_index.csv"
    if index_path.exists():
        try:
            df = pd.read_csv(index_path)
            if len(df) > 0:
                return df.to_dict(orient="records"), "trade_journal_index.csv"
        except Exception:
            # Fall through to JSON loader
            pass

    trades_dir = run_dir / "trade_journal" / "trades"
    rows: List[Dict[str, Any]] = []
    if not trades_dir.exists():
        return rows, "missing (no trade_journal_index.csv, no trades/*.json)"

    for json_file in sorted(trades_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                trade = json.load(f)
        except Exception:
            continue
        if not isinstance(trade, dict):
            continue
        rows.append(trade_to_row(trade))
    return rows, "trades/*.json (CSV missing/empty)"


def _load_overlay_meta_for_run(run_dir: Path) -> Dict[str, Dict]:
    """
    Load overlay metadata from trade_journal JSON files for a run.

    Returns:
        {trade_id: sniper_overlay_dict}
    """
    trades_dir = run_dir / "trade_journal" / "trades"
    meta: Dict[str, Dict] = {}
    if not trades_dir.exists():
        return meta
    for json_file in trades_dir.glob("*.json"):
        try:
            data = pd.read_json(json_file)
        except ValueError:
            # Fallback: standard json
            import json

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        if isinstance(data, dict):
            d = data
        else:
            # Unexpected structure; skip
            continue
        entry = d.get("entry_snapshot") or {}
        overlay = entry.get("sniper_overlay")
        if overlay:
            trade_id = d.get("trade_id") or entry.get("trade_id")
            if trade_id:
                meta[str(trade_id)] = overlay
    return meta


def _compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute basic metrics from a trade DataFrame (requires 'pnl_bps')."""
    if df.empty or "pnl_bps" not in df.columns:
        return {
            "trades": 0,
            "ev_per_trade_bps": 0.0,
            "win_rate": 0.0,
            "payoff": float("nan"),
            "p90_loss_bps": 0.0,
        }
    pnl = pd.to_numeric(df["pnl_bps"], errors="coerce").dropna()
    trades = len(pnl)
    if trades == 0:
        return {
            "trades": 0,
            "ev_per_trade_bps": 0.0,
            "win_rate": 0.0,
            "payoff": float("nan"),
            "p90_loss_bps": 0.0,
        }
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    ev = float(pnl.mean())
    win_rate = float((pnl > 0).mean())
    avg_win = float(wins.mean()) if not wins.empty else float("nan")
    avg_loss = float(losses.mean()) if not losses.empty else float("nan")
    if not wins.empty and not losses.empty and avg_loss != 0:
        payoff = float(avg_win / abs(avg_loss))
    else:
        payoff = float("nan")
    if not losses.empty:
        loss_mag = losses.abs().astype(float)
        p90_loss = float(np.percentile(loss_mag.values, 90))
    else:
        p90_loss = 0.0
    return {
        "trades": trades,
        "ev_per_trade_bps": ev,
        "win_rate": win_rate,
        "payoff": payoff,
        "p90_loss_bps": p90_loss,
    }


def _attach_regime_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach regime_class column using classify_regime( row_dict ) for each trade.
    """
    if df.empty:
        df_local = df.copy()
        df_local["regime_class"] = None
        return df_local
    regime_classes: List[Optional[str]] = []
    for _, row in df.iterrows():
        try:
            r_class, _ = classify_regime(row.to_dict())
        except Exception:
            r_class = None
        regime_classes.append(r_class)
    df_local = df.copy()
    df_local["regime_class"] = regime_classes
    return df_local


def _compute_coverage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute coverage metrics:
    - pnl_coverage: % of trades with valid pnl_bps
    - regime_coverage: % of trades with non-null regime_class
    """
    if df.empty:
        return {"pnl_coverage": 0.0, "regime_coverage": 0.0}
    pnl = pd.to_numeric(df.get("pnl_bps"), errors="coerce")
    pnl_coverage = float(pnl.notna().mean()) if len(pnl) > 0 else 0.0
    regime_col = df.get("regime_class")
    regime_coverage = float(regime_col.notna().mean()) if regime_col is not None else 0.0
    return {
        "pnl_coverage": pnl_coverage,
        "regime_coverage": regime_coverage,
    }


def check_overlay_correctness(run_dir: Path, overlay_cfg_multiplier: float = 0.30) -> Tuple[int, int]:
    """
    Strict overlay correctness check for a single overlay run.

    Ensures:
    - overlay_applied==True only when quarter=='Q4' and regime_class=='B_MIXED'
    - overlay_name=='Q4_B_MIXED_SIZE'
    - multiplier==overlay_cfg_multiplier
    - size_after_units == round(abs(size_before_units)*multiplier)*sign

    Returns:
        (overlay_present_count, overlay_applied_count)
    """
    import json

    trades_dir = run_dir / "trade_journal" / "trades"
    if not trades_dir.exists():
        return 0, 0

    overlay_present = 0
    overlay_applied = 0
    bad_samples = []

    for json_file in trades_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        entry = data.get("entry_snapshot") or {}
        overlay = entry.get("sniper_overlay")
        if not overlay:
            continue
        overlay_present += 1
        trade_id = data.get("trade_id") or entry.get("trade_id")

        applied = bool(overlay.get("overlay_applied", False))
        if not applied:
            continue
        overlay_applied += 1

        quarter = overlay.get("quarter")
        regime_class = overlay.get("regime_class")
        overlay_name = overlay.get("overlay_name")
        multiplier = float(overlay.get("multiplier", 0.0))
        size_before = int(overlay.get("size_before_units", 0))
        size_after = int(overlay.get("size_after_units", size_before))

        # Check name and multiplier
        ok = True
        if overlay_name != "Q4_B_MIXED_SIZE":
            ok = False
        if abs(multiplier - overlay_cfg_multiplier) > 1e-6:
            ok = False

        # Check regime/quarter
        if quarter != "Q4" or regime_class != "B_MIXED":
            ok = False

        # Check size math
        sign = 1 if size_before >= 0 else -1
        expected_abs = int(round(abs(size_before) * overlay_cfg_multiplier))
        if expected_abs == 0 and abs(size_before) > 0:
            expected_abs = 1
        expected = sign * expected_abs
        if size_after != expected:
            ok = False

        if not ok:
            bad_samples.append(
                {
                    "trade_id": trade_id,
                    "file": json_file.name,
                    "overlay": overlay,
                    "expected_units": expected,
                }
            )
        if len(bad_samples) >= 10:
            break

    if bad_samples:
        raise RuntimeError(
            f"Overlay correctness failed for run_dir={run_dir}: "
            f"{len(bad_samples)} bad samples (showing up to 10): {bad_samples}"
        )

    return overlay_present, overlay_applied


def build_report(pairs: List[RunPair], ts: str) -> str:
    lines: List[str] = []
    lines.append(f"# SNIPER Q4 Overlay Evaluation ({ts})")
    lines.append("")
    lines.append(
        "This report compares Q4 2025 SNIPER performance with and without the "
        "Q4 × B_MIXED size overlay (no replay; existing run_dirs only)."
    )
    lines.append("")

    for pair in pairs:
        base = pair.variant_base
        lines.append(f"## Variant base: `{base}`")
        lines.append("")

        if pair.normal is None:
            lines.append("- ❌ Missing normal run_dir (no baseline) for Q4.")
        if pair.overlay is None:
            lines.append("- ❌ Missing overlay run_dir for Q4.")
        lines.append("")

        if pair.normal is None or pair.overlay is None:
            lines.append("_Skipping metrics; missing run(s)._")
            lines.append("")
            continue

        # Sanity: normal and overlay must not resolve to the same run_dir
        if pair.normal == pair.overlay:
            raise RuntimeError(
                f"Invalid pairing for base={base}: normal and overlay both "
                f"resolved to {pair.normal}"
            )

        # Load trades (CSV index when available, JSON otherwise)
        rows_norm, src_norm = load_trades_from_run(pair.normal)
        rows_ov, src_ov = load_trades_from_run(pair.overlay)

        df_normal = pd.DataFrame(rows_norm)
        df_overlay = pd.DataFrame(rows_ov)

        lines.append(f"- Normal run: `{pair.normal}` (source: {src_norm})")
        lines.append(f"- Overlay run: `{pair.overlay}` (source: {src_ov})")
        lines.append("")

        # Total Q4 metrics
        metrics_norm = _compute_metrics(df_normal)
        metrics_ov = _compute_metrics(df_overlay)

        # B_MIXED only (classify regimes)
        df_normal_with_regime = _attach_regime_class(df_normal)
        df_overlay_with_regime = _attach_regime_class(df_overlay)

        df_b_norm = df_normal_with_regime[
            df_normal_with_regime["regime_class"] == "B_MIXED"
        ].copy()
        df_b_ov = df_overlay_with_regime[
            df_overlay_with_regime["regime_class"] == "B_MIXED"
        ].copy()
        metrics_b_norm = _compute_metrics(df_b_norm)
        metrics_b_ov = _compute_metrics(df_b_ov)

        # Coverage metrics
        cov_norm = _compute_coverage(df_normal_with_regime)
        cov_ov = _compute_coverage(df_overlay_with_regime)

        # Overlay correctness metrics
        overlay_present = overlay_applied = 0
        try:
            overlay_present, overlay_applied = check_overlay_correctness(pair.overlay)
        except RuntimeError as e:
            lines.append(f"- ❌ Overlay correctness failed: {e}")
            lines.append("")
            continue

        lines.append("### Data source & coverage")
        lines.append("")
        lines.append(
            f"- normal: source={src_norm}, "
            f"pnl_coverage={cov_norm['pnl_coverage']*100:.1f}%, "
            f"regime_coverage={cov_norm['regime_coverage']*100:.1f}%"
        )
        lines.append(
            f"- overlay: source={src_ov}, "
            f"pnl_coverage={cov_ov['pnl_coverage']*100:.1f}%, "
            f"regime_coverage={cov_ov['regime_coverage']*100:.1f}%"
        )
        lines.append("")

        lines.append("### Q4 total metrics")
        lines.append("")
        lines.append("| Run | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        lines.append(
            f"| normal | {metrics_norm['trades']} | {metrics_norm['ev_per_trade_bps']:.2f} "
            f"| {metrics_norm['win_rate']*100:.1f}% | {metrics_norm['payoff']:.2f} "
            f"| {metrics_norm['p90_loss_bps']:.1f} |"
        )
        lines.append(
            f"| overlay | {metrics_ov['trades']} | {metrics_ov['ev_per_trade_bps']:.2f} "
            f"| {metrics_ov['win_rate']*100:.1f}% | {metrics_ov['payoff']:.2f} "
            f"| {metrics_ov['p90_loss_bps']:.1f} |"
        )
        lines.append("")

        lines.append("### Q4 B_MIXED metrics only")
        lines.append("")
        lines.append("| Run | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        lines.append(
            f"| normal | {metrics_b_norm['trades']} | {metrics_b_norm['ev_per_trade_bps']:.2f} "
            f"| {metrics_b_norm['win_rate']*100:.1f}% | {metrics_b_norm['payoff']:.2f} "
            f"| {metrics_b_norm['p90_loss_bps']:.1f} |"
        )
        lines.append(
            f"| overlay | {metrics_b_ov['trades']} | {metrics_b_ov['ev_per_trade_bps']:.2f} "
            f"| {metrics_b_ov['win_rate']*100:.1f}% | {metrics_b_ov['payoff']:.2f} "
            f"| {metrics_b_ov['p90_loss_bps']:.1f} |"
        )
        lines.append("")

        lines.append("### Overlay sanity")
        lines.append("")
        lines.append(f"- overlay_present (JSON with `sniper_overlay`): {overlay_present}")
        lines.append(f"- overlay_applied (`overlay_applied=True`): {overlay_applied}")
        lines.append(
            "- By construction, overlay_applied trades are only Q4 × B_MIXED and "
            "have consistent size_before/after and multiplier."
        )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Q4 SNIPER overlay A/B (no replay)."
    )
    # No custom args for now (autodetect only)
    _ = parser.parse_args()

    pairs = autodetect_q4_runs()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_text = build_report(pairs, ts)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"SNIPER_Q4_OVERLAY_EVAL__{ts}.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote Q4 overlay eval report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


