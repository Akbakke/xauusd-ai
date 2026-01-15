#!/usr/bin/env python3
"""
Generate FULLYEAR kill-chain analysis report (where trades die).

This is READ-ONLY reporting based on chunk footers produced by replay runs.

Usage:
  python3 scripts/generate_killchain_analysis.py <run_dir> [output_path]

Where:
  run_dir: Directory containing chunk_*/chunk_footer.json and perf_*.json (optional)
  output_path: Optional markdown output path. Defaults to:
    reports/replay_eval/FULLYEAR_KILLCHAIN_ANALYSIS_<run_id>.md
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


KILLCHAIN_REASON_ORDER = [
    "BLOCK_BELOW_THRESHOLD",
    "BLOCK_SESSION",
    "BLOCK_VOL",
    "BLOCK_REGIME",
    "BLOCK_RISK",
    "BLOCK_POSITION_LIMIT",
    "BLOCK_COOLDOWN",
    "BLOCK_UNKNOWN",
]


@dataclass
class KillchainAggregate:
    version: int
    counters: Dict[str, int]
    reasons: Dict[str, int]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _find_run_id(run_dir: Path, footers: List[Dict[str, Any]]) -> str:
    for footer in footers:
        run_id = footer.get("run_id")
        if isinstance(run_id, str) and run_id:
            return run_id
    # Fallback to directory name
    return run_dir.name


def _aggregate_killchain(run_dir: Path) -> Tuple[KillchainAggregate, Dict[str, Any]]:
    footer_paths = sorted(run_dir.glob("chunk_*/chunk_footer.json"))
    if not footer_paths:
        raise FileNotFoundError(f"No chunk footers found under: {run_dir}")

    footers: List[Dict[str, Any]] = []
    for p in footer_paths:
        d = _load_json(p)
        if d is not None:
            footers.append(d)

    if not footers:
        raise RuntimeError(f"Failed to load any chunk footers under: {run_dir}")

    version = int(footers[0].get("killchain_version", 1))
    counter_keys = [
        "killchain_n_entry_pred_total",
        "killchain_n_above_threshold",
        "killchain_n_after_session_guard",
        "killchain_n_after_vol_guard",
        "killchain_n_after_regime_guard",
        "killchain_n_after_risk_sizing",
        "killchain_n_trade_create_attempts",
        "killchain_n_trade_created",
    ]
    counters = {k: 0 for k in counter_keys}
    reasons: Dict[str, int] = {k: 0 for k in KILLCHAIN_REASON_ORDER}

    for f in footers:
        version = int(f.get("killchain_version", version))
        for k in counter_keys:
            try:
                counters[k] += int(f.get(k, 0))
            except Exception:
                pass
        rc = f.get("killchain_block_reason_counts", {}) or {}
        if isinstance(rc, dict):
            for rk, rv in rc.items():
                if rk not in reasons:
                    # Keep stable set; bucket everything else into UNKNOWN
                    rk = "BLOCK_UNKNOWN"
                try:
                    reasons[rk] += int(rv)
                except Exception:
                    pass

    ctx = {
        "run_id": _find_run_id(run_dir, footers),
        "policy_path": None,
        "dataset_path": None,
        "prebuilt_path": None,
        "features_file_sha256": None,
        "git_head": None,
    }

    # Best-effort context from perf json (if present)
    perf_files = sorted(run_dir.glob("perf_*.json"))
    if perf_files:
        perf = _load_json(perf_files[0]) or {}
        ctx["run_id"] = perf.get("run_id", ctx["run_id"])
        ctx["policy_path"] = perf.get("policy_path")
        ctx["dataset_path"] = perf.get("data_path")
        ctx["prebuilt_path"] = perf.get("prebuilt_path")
        ctx["features_file_sha256"] = perf.get("features_file_sha256")
        ctx["git_head"] = perf.get("git_head")

    return KillchainAggregate(version=version, counters=counters, reasons=reasons), ctx


def _top_reasons(reasons: Dict[str, int], n: int = 5) -> List[Tuple[str, int]]:
    items = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:n]


def _pct(part: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{(part / float(total)) * 100.0:.1f}%"


def generate_report(run_dir: Path, output_path: Path) -> None:
    agg, ctx = _aggregate_killchain(run_dir)

    total_pred = int(agg.counters["killchain_n_entry_pred_total"])
    above_thr = int(agg.counters["killchain_n_above_threshold"])
    created = int(agg.counters["killchain_n_trade_created"])

    top_reason = _top_reasons(agg.reasons, n=1)[0][0] if agg.reasons else None
    top5 = _top_reasons(agg.reasons, n=5)

    smoking_gun = "UNKNOWN"
    if above_thr > 0 and created == 0:
        smoking_gun = "POST_GATES_KILL_ALL"
    elif above_thr == 0 and created == 0:
        smoking_gun = "NO_SIGNAL_ABOVE_THRESHOLD"
    elif created > 0:
        smoking_gun = "TRADES_CREATED"

    lines: List[str] = []
    lines.append("# FULLYEAR KILLCHAIN ANALYSIS")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Run ID:** {ctx.get('run_id')}")
    lines.append(f"**Source:** {run_dir}")
    lines.append("")
    lines.append("## RUN_CTX")
    lines.append("")
    lines.append(f"- **root:** {run_dir.parent.parent}")
    lines.append(f"- **head:** {ctx.get('git_head') or 'unknown'}")
    lines.append(f"- **policy path:** {ctx.get('policy_path') or 'unknown'}")
    lines.append(f"- **dataset path:** {ctx.get('dataset_path') or 'unknown'}")
    lines.append(f"- **prebuilt path:** {ctx.get('prebuilt_path') or 'unknown'}")
    lines.append(f"- **prebuilt sha:** {ctx.get('features_file_sha256') or 'unknown'}")
    lines.append("")
    lines.append("## Kill-chain (counts + % of n_entry_pred_total)")
    lines.append("")
    lines.append("| Step | Count | % of entry_pred_total |")
    lines.append("|------|-------|------------------------|")
    for k in [
        "killchain_n_entry_pred_total",
        "killchain_n_above_threshold",
        "killchain_n_after_session_guard",
        "killchain_n_after_vol_guard",
        "killchain_n_after_regime_guard",
        "killchain_n_after_risk_sizing",
        "killchain_n_trade_create_attempts",
        "killchain_n_trade_created",
    ]:
        v = int(agg.counters.get(k, 0))
        lines.append(f"| {k} | {v:,} | {_pct(v, total_pred)} |")
    lines.append("")
    lines.append("## Block reasons (Top 5)")
    lines.append("")
    lines.append("| Reason | Count |")
    lines.append("|--------|-------|")
    for r, c in top5:
        lines.append(f"| {r} | {c:,} |")
    lines.append("")
    lines.append("## Smoking gun")
    lines.append("")
    lines.append(f"- **smoking_gun:** {smoking_gun}")
    lines.append(f"- **top_block_reason:** {top_reason}")
    if above_thr > 0 and created == 0:
        lines.append(
            "- **conclusion:** n_above_threshold > 0 and n_trade_created == 0 → post-gates are killing all potential trades."
        )
    elif above_thr == 0 and created == 0:
        lines.append(
            "- **conclusion:** n_above_threshold == 0 → threshold/score distribution is the primary limiter (not post-gates)."
        )
    else:
        lines.append(
            "- **conclusion:** trades are being created → use per-trade analysis next (winners/losers, tails, sessions/regimes)."
        )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✅ Wrote: {output_path}")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: generate_killchain_analysis.py <run_dir> [output_path]")
        return 2

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return 2

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        # Best-effort run_id from perf json or directory name
        perf_files = sorted(run_dir.glob("perf_*.json"))
        run_id = run_dir.name
        if perf_files:
            perf = _load_json(perf_files[0]) or {}
            run_id = perf.get("run_id", run_id)
        output_path = Path("reports/replay_eval") / f"FULLYEAR_KILLCHAIN_ANALYSIS_{run_id}.md"

    generate_report(run_dir, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

