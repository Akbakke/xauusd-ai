#!/usr/bin/env python3
"""V12 daily trade review — turn structured trade journal into actionable insight.

For each completed (and still-open) trade in the trade journal, reconstructs the
full narrative: entry context → bar-by-bar trajectory → key moments → exit.
Outputs:

  - REVIEW.md        : day-level markdown with summary tables + per-trade rows
  - trades_metrics.csv : one row per trade (for batch analysis / retraining feed)
  - trades/<id>.md   : per-trade detail (narrative + key moments)

Designed to be cron-friendly: idempotent rebuild of today's review every N min.

Inputs:
  - /home/andre2/GX1_DATA/reports/v12_paper_runs/trade_journal/trades/*.json
  - /home/andre2/GX1_DATA/reports/v12_paper_runs/trade_journal/trade_journal_index.csv
  - /home/andre2/GX1_DATA/reports/v12_paper_runs/v12_paper_journal_<DATE>_<SUFFIX>.jsonl
    (optional, used for pre-entry context if present)
  - /home/andre2/GX1_DATA/reports/v12_paper_runs/counterfactual_reports/*.jsonl
    (optional, used to cross-reference SKIPs ≥25h ago)

Run:
  python3 gx1/execution/v12_daily_trade_review.py [--date 20260512] [--suffix live]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger("v12_review")

JOURNAL_DIR = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs")
TRADE_JSON_DIR = JOURNAL_DIR / "trade_journal" / "trades"
INDEX_CSV = JOURNAL_DIR / "trade_journal" / "trade_journal_index.csv"
OPEN_TRADES_DIR = JOURNAL_DIR / "open_trades"
REVIEW_BASE = JOURNAL_DIR / "daily_reviews"


def _load_state(trade_id: str) -> dict | None:
    """Read the runner's persisted TradeState for an open trade (if present)."""
    p = OPEN_TRADES_DIR / f"open_trade_{trade_id}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _backfill_entry_from_state(trade_id: str) -> dict | None:
    """For trades that were already open when TradeJournal was retrofitted,
    entry_snapshot will be None. Pull what we have from open_trade_<id>.json
    (the runner's persisted TradeState) — it preserves side/entry/v10_snapshot.
    """
    s = _load_state(trade_id)
    if s is None:
        return None
    v10 = s.get("v10_snapshot") or {}
    dp = v10.get("direction_probs") or [0.0, 0.0, 0.0]
    return {
        "trade_id": trade_id,
        "entry_time": s.get("entry_ts"),
        "side": s.get("side"),
        "entry_price": s.get("entry_ask") if s.get("side") == "long" else s.get("entry_bid"),
        "entry_bid": s.get("entry_bid"),
        "entry_ask": s.get("entry_ask"),
        "entry_spread_bps": s.get("entry_spread_bps"),
        "entry_score": {
            "v10_p_long": dp[0] if len(dp) > 0 else 0.0,
            "v10_p_short": dp[1] if len(dp) > 1 else 0.0,
            "v10_path_quality_pred": v10.get("path_quality"),
            "v10_mfe_pred_at_entry": v10.get("mfe_first_n"),
            "v10_tradable_prob": v10.get("tradable_prob"),
            "v10_bad_path_prob": v10.get("bad_path_prob"),
            "decision_ts": v10.get("decision_ts"),
        },
        "_backfilled_from_state": True,
    }


# ── trade analysis ─────────────────────────────────────────────────────────

def _bar_metrics(bars: list[dict]) -> dict[str, Any]:
    """Derive trade-level metrics from per-bar decision trace."""
    if not bars:
        return {}

    n = len(bars)
    pnls = [b["current_pnl_bps"] for b in bars]
    mfes = [b["cum_mfe_bps"] for b in bars]
    maes = [b["cum_mae_bps"] for b in bars]
    v3_probs = [b.get("v3_should_exit_prob") or 0.0 for b in bars]

    max_mfe = max(mfes)
    max_mae = min(maes)
    final_pnl = pnls[-1]
    mfe_peak_bar = next((i for i, m in enumerate(mfes) if m == max_mfe), 0)
    mae_worst_bar = next((i for i, m in enumerate(maes) if m == max_mae), 0)

    # V3 alarms
    v3_max_prob = max(v3_probs) if v3_probs else 0.0
    v3_first_alarm_bar = next((i for i, p in enumerate(v3_probs) if p > 0.5), -1)

    # IQL held while V3 alarmed (≥3 bars)
    iql_held_through_v3 = sum(
        1 for b in bars
        if (b.get("v3_should_exit_prob") or 0) > 0.5 and b.get("iql_action") != "EXIT_NOW"
    )

    # MFE giveback ratio: (max_mfe - final_pnl) / max_mfe
    giveback_bps = max_mfe - final_pnl if max_mfe > 0 else 0.0
    giveback_pct = giveback_bps / max_mfe if max_mfe > 0 else 0.0

    # Jojo count: each time pnl crosses 0 (long-only proxy for swings)
    jojo = sum(
        1 for i in range(1, n)
        if (pnls[i - 1] >= 0) != (pnls[i] >= 0)
    )

    return {
        "n_bars": n,
        "max_mfe_bps": max_mfe,
        "max_mae_bps": max_mae,
        "final_pnl_bps": final_pnl,
        "mfe_peak_bar": mfe_peak_bar,
        "mae_worst_bar": mae_worst_bar,
        "v3_max_prob": v3_max_prob,
        "v3_first_alarm_bar": v3_first_alarm_bar,
        "iql_held_through_v3_count": iql_held_through_v3,
        "mfe_giveback_bps": giveback_bps,
        "mfe_giveback_pct": giveback_pct,
        "pnl_zero_crossings": jojo,
    }


def _classify_pattern(metrics: dict, exit_summary: dict | None) -> list[str]:
    """Tag a trade with diagnostic patterns for easier filtering."""
    tags = []
    final = metrics.get("final_pnl_bps", 0.0)
    mfe = metrics.get("max_mfe_bps", 0.0)
    mae = metrics.get("max_mae_bps", 0.0)
    giveback = metrics.get("mfe_giveback_pct", 0.0)

    if exit_summary is None:
        tags.append("STILL_OPEN")
    elif final > 0:
        tags.append("WINNER")
    else:
        tags.append("LOSER")

    if mfe >= 20 and giveback >= 0.5:
        tags.append("MFE_GIVEBACK_50PCT")
    if mfe >= 15 and final <= 0:
        tags.append("MFE_HIT_BUT_NEGATIVE_EXIT")
    if mae <= -20 and final > 0:
        tags.append("RECOVERY_FROM_DEEP_MAE")
    if mae <= -20 and final <= mae * 0.8:
        tags.append("RAN_TO_STOP")
    if metrics.get("iql_held_through_v3_count", 0) >= 5:
        tags.append("IQL_HELD_THROUGH_V3_ALARM")
    if metrics.get("pnl_zero_crossings", 0) >= 4:
        tags.append("JOJO_4PLUS")
    return tags


# ── trade JSON → summary row ───────────────────────────────────────────────

def trade_summary_row(trade_json: dict) -> dict[str, Any]:
    """Flatten a trade journal into a single-row metrics dict."""
    entry = trade_json.get("entry_snapshot") or {}
    exit_s = trade_json.get("exit_summary")
    bars = trade_json.get("v12_bar_decisions") or []
    entry_score = entry.get("entry_score") or {}

    metrics = _bar_metrics(bars)
    tags = _classify_pattern(metrics, exit_s)

    return {
        "trade_id": trade_json.get("trade_id") or "",
        "side": entry.get("side", ""),
        "entry_time": entry.get("entry_time", ""),
        "entry_price": entry.get("entry_price", ""),
        "entry_spread_bps": entry.get("entry_spread_bps", ""),
        "decision_ts": entry_score.get("decision_ts", ""),
        "v10_p_long": entry_score.get("v10_p_long", ""),
        "v10_p_short": entry_score.get("v10_p_short", ""),
        "v10_path_quality": entry_score.get("v10_path_quality_pred", ""),
        "v10_mfe_pred": entry_score.get("v10_mfe_pred_at_entry", ""),
        "v10_tradable": entry_score.get("v10_tradable_prob", ""),
        "v10_bad_path": entry_score.get("v10_bad_path_prob", ""),
        "q_take_long": entry_score.get("q_take_long", ""),
        "q_take_short": entry_score.get("q_take_short", ""),
        "q_skip": entry_score.get("q_skip", ""),
        "q_advantage_at_entry": entry_score.get("advantage_over_skip", ""),
        # Outcome
        "exit_time": (exit_s or {}).get("exit_time", ""),
        "exit_price": (exit_s or {}).get("exit_price", ""),
        "exit_reason": (exit_s or {}).get("exit_reason", "STILL_OPEN"),
        "realized_pnl_bps": (exit_s or {}).get("realized_pnl_bps", metrics.get("final_pnl_bps", 0.0)),
        # Bar-trace derived
        "n_bars": metrics.get("n_bars", 0),
        "max_mfe_bps": metrics.get("max_mfe_bps", 0.0),
        "max_mae_bps": metrics.get("max_mae_bps", 0.0),
        "mfe_peak_bar": metrics.get("mfe_peak_bar", -1),
        "mae_worst_bar": metrics.get("mae_worst_bar", -1),
        "mfe_giveback_bps": metrics.get("mfe_giveback_bps", 0.0),
        "mfe_giveback_pct": metrics.get("mfe_giveback_pct", 0.0),
        "v3_max_prob": metrics.get("v3_max_prob", 0.0),
        "v3_first_alarm_bar": metrics.get("v3_first_alarm_bar", -1),
        "iql_held_through_v3_count": metrics.get("iql_held_through_v3_count", 0),
        "pnl_zero_crossings": metrics.get("pnl_zero_crossings", 0),
        "tags": ",".join(tags),
    }


# ── per-trade detail markdown ──────────────────────────────────────────────

def render_trade_detail(trade_json: dict, summary: dict, out_path: Path) -> None:
    entry = trade_json.get("entry_snapshot") or {}
    exit_s = trade_json.get("exit_summary")
    bars = trade_json.get("v12_bar_decisions") or []
    score = entry.get("entry_score") or {}
    tid = trade_json.get("trade_id") or "?"

    lines: list[str] = []
    lines.append(f"# Trade #{tid}\n")
    lines.append(f"**Tags:** `{summary['tags']}`\n")

    lines.append("## Entry\n")
    lines.append(f"- **Time:** {entry.get('entry_time')}")
    lines.append(f"- **Side:** {entry.get('side')}  ·  **Price:** {entry.get('entry_price')}  ·  **Spread:** {entry.get('entry_spread_bps'):.2f} bps")
    lines.append(f"- **Decision M5 bucket:** {score.get('decision_ts')}")
    lines.append("")
    lines.append("### V10 outputs at entry")
    lines.append(f"- p_long: **{score.get('v10_p_long', 0):.3f}**  ·  p_short: {score.get('v10_p_short', 0):.3f}")
    lines.append(f"- path_quality: {score.get('v10_path_quality_pred', 0):.3f}  ·  mfe_pred: {score.get('v10_mfe_pred_at_entry', 0):.2f} bps")
    lines.append(f"- tradable_prob: {score.get('v10_tradable_prob', 0):.3f}  ·  bad_path_prob: {score.get('v10_bad_path_prob', 0):.3f}")
    lines.append("")
    lines.append("### Entry-IQL Q-values")
    lines.append(f"- Q_take_long: **{score.get('q_take_long', 0):+.2f}**  ·  Q_take_short: {score.get('q_take_short', 0):+.2f}  ·  Q_skip: {score.get('q_skip', 0):+.4f}")
    lines.append(f"- Advantage over skip: **{score.get('advantage_over_skip', 0):+.2f}** bps  (long: {score.get('advantage_over_skip_long', 0):+.2f}, short: {score.get('advantage_over_skip_short', 0):+.2f})")
    lines.append("")

    if bars:
        lines.append("## In-trade trajectory\n")
        lines.append("| bar | time | bid | pnl | mfe | mae | dd_from_peak | v3_prob | v3_consec | iql | source |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        # Sample at most ~30 bars across the trade for readability
        n = len(bars)
        step = max(1, n // 30)
        sampled_idx = list(range(0, n, step))
        # always include MFE peak + MAE worst bars
        if summary["mfe_peak_bar"] not in sampled_idx and summary["mfe_peak_bar"] >= 0:
            sampled_idx.append(summary["mfe_peak_bar"])
        if summary["mae_worst_bar"] not in sampled_idx and summary["mae_worst_bar"] >= 0:
            sampled_idx.append(summary["mae_worst_bar"])
        if summary["v3_first_alarm_bar"] not in sampled_idx and summary["v3_first_alarm_bar"] >= 0:
            sampled_idx.append(summary["v3_first_alarm_bar"])
        if n - 1 not in sampled_idx:
            sampled_idx.append(n - 1)
        for i in sorted(set(sampled_idx)):
            b = bars[i]
            mark = ""
            if i == summary["mfe_peak_bar"]:
                mark = " 🟢MFE"
            if i == summary["mae_worst_bar"]:
                mark += " 🔴MAE"
            if i == summary["v3_first_alarm_bar"]:
                mark += " ⚠️V3"
            ts = b.get("timestamp", "?")[11:16]
            dd = (b.get("cum_mfe_bps", 0) - b.get("current_pnl_bps", 0))
            lines.append(
                f"| {b.get('bars_in_trade', i)}{mark} | {ts} | {b.get('bid', 0):.2f} | "
                f"{b.get('current_pnl_bps', 0):+.2f} | {b.get('cum_mfe_bps', 0):.2f} | "
                f"{b.get('cum_mae_bps', 0):.2f} | {dd:.2f} | "
                f"{b.get('v3_should_exit_prob', 0):.3f} | {b.get('v3_consecutive_exits', 0)} | "
                f"{b.get('iql_action', '?')} | {b.get('iql_decision_source', '?')} |"
            )
        lines.append("")

    lines.append("## Exit\n")
    if exit_s:
        lines.append(f"- **Time:** {exit_s.get('exit_time')}")
        lines.append(f"- **Price:** {exit_s.get('exit_price'):.2f}  ·  **Reason:** {exit_s.get('exit_reason')}")
        lines.append(f"- **Realized PnL:** {exit_s.get('realized_pnl_bps'):+.2f} bps")
        lines.append(f"- **Max MFE / MAE:** {exit_s.get('max_mfe_bps'):.2f} / {exit_s.get('max_mae_bps'):.2f} bps")
        lines.append(f"- **Intratrade drawdown from MFE peak:** {exit_s.get('intratrade_drawdown_bps'):.2f} bps")
    else:
        lines.append(f"- **STILL OPEN** (bars_in_trade={summary['n_bars']})")
        lines.append(f"- Current unrealized: **{summary['realized_pnl_bps']:+.2f} bps**")
    lines.append("")

    lines.append("## Verdikt\n")
    tags = summary["tags"].split(",")
    if "MFE_HIT_BUT_NEGATIVE_EXIT" in tags:
        lines.append("- ⚠️ **MFE peak ble ikke kapitalisert** — modellen så peak ≥15 bps men endte i minus.")
    if "MFE_GIVEBACK_50PCT" in tags:
        lines.append(f"- ⚠️ **Giveback ≥50%** — MFE peak {summary['max_mfe_bps']:.1f} bps → final {summary['realized_pnl_bps']:+.1f} bps (giveback {summary['mfe_giveback_pct']*100:.0f}%).")
    if "RECOVERY_FROM_DEEP_MAE" in tags:
        lines.append(f"- ✅ **Recovery fra deep MAE** — bunn {summary['max_mae_bps']:.1f} bps → final {summary['realized_pnl_bps']:+.1f}. IQL holdt rett.")
    if "IQL_HELD_THROUGH_V3_ALARM" in tags:
        lines.append(f"- 🟡 **IQL ignored V3** — V3 sa exit i {summary['iql_held_through_v3_count']} bars, IQL holdt. Verifiser om dette var riktig (Phase 6 sa NO_TRAIL er edge).")
    if "JOJO_4PLUS" in tags:
        lines.append(f"- ⚠️ **{summary['pnl_zero_crossings']} svingninger** rundt break-even — roller-coaster mønster.")
    if "RAN_TO_STOP" in tags:
        lines.append(f"- 🔴 **Løp til stop-like deep MAE** — IQL holdt for lenge i tap, eller markedet snudde aldri.")
    if not any(t in tags for t in ["MFE_GIVEBACK_50PCT", "MFE_HIT_BUT_NEGATIVE_EXIT", "RECOVERY_FROM_DEEP_MAE",
                                    "IQL_HELD_THROUGH_V3_ALARM", "JOJO_4PLUS", "RAN_TO_STOP"]):
        lines.append("- (Ingen patologiske mønstre flagget)")
    lines.append("")

    out_path.write_text("\n".join(lines))


# ── day-level review markdown ──────────────────────────────────────────────

def render_day_review(rows: list[dict], date_str: str, out_dir: Path) -> None:
    closed = [r for r in rows if r["exit_reason"] != "STILL_OPEN"]
    still_open = [r for r in rows if r["exit_reason"] == "STILL_OPEN"]
    winners = [r for r in closed if (r["realized_pnl_bps"] or 0) > 0]
    losers = [r for r in closed if (r["realized_pnl_bps"] or 0) <= 0]

    total_realized = sum(float(r["realized_pnl_bps"] or 0) for r in closed)
    total_unrealized = sum(float(r["realized_pnl_bps"] or 0) for r in still_open)

    # Cluster detection: trades opened within 5 min, same side
    sorted_rows = sorted(rows, key=lambda r: r["entry_time"] or "")
    clusters: list[list[dict]] = []
    for r in sorted_rows:
        if not r["entry_time"]:
            continue
        if clusters and clusters[-1]:
            last = clusters[-1][-1]
            last_t = datetime.fromisoformat(last["entry_time"].replace("Z", "+00:00"))
            this_t = datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00"))
            if r["side"] == last["side"] and (this_t - last_t).total_seconds() < 300:
                clusters[-1].append(r)
                continue
        clusters.append([r])
    multi_clusters = [c for c in clusters if len(c) >= 2]

    lines = []
    lines.append(f"# V12 Daily Trade Review — {date_str}\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")

    lines.append("## Day summary\n")
    lines.append(f"- **Trades opened today:** {len(rows)}  ·  **closed:** {len(closed)}  ·  **still open:** {len(still_open)}")
    if closed:
        wr = 100.0 * len(winners) / len(closed)
        lines.append(f"- **Win rate (closed):** {wr:.1f}%  ({len(winners)}W / {len(losers)}L)")
    lines.append(f"- **Realized PnL (sum bps):** {total_realized:+.2f}")
    lines.append(f"- **Unrealized PnL (sum bps):** {total_unrealized:+.2f}")
    if rows:
        mfes = [float(r["max_mfe_bps"] or 0) for r in rows]
        maes = [float(r["max_mae_bps"] or 0) for r in rows]
        med = lambda xs: sorted(xs)[len(xs) // 2] if xs else 0
        lines.append(f"- **Median MFE peak:** {med(mfes):.1f} bps  ·  **Median MAE worst:** {med(maes):.1f} bps")
        lines.append(f"- **Trades with MFE peak ≥20 bps:** {sum(1 for m in mfes if m >= 20)}")
        lines.append(f"- **Trades with MAE ≤-20 bps:** {sum(1 for m in maes if m <= -20)}")
    lines.append("")

    # Clusters
    if multi_clusters:
        lines.append("## Concentration / clustering\n")
        for i, c in enumerate(multi_clusters, 1):
            sides = c[0]["side"]
            tstart = c[0]["entry_time"][:19]
            tend = c[-1]["entry_time"][:19]
            ids = ", ".join(str(r["trade_id"]) for r in c)
            total_pnl = sum(float(r["realized_pnl_bps"] or 0) for r in c)
            lines.append(f"- **Cluster {i}:** {len(c)}× {sides.upper()} from {tstart} to {tend}  →  combined PnL {total_pnl:+.2f} bps  ({ids})")
        lines.append("\n*(Same-side trades within 5 min of each other = 1 V12 signal × N entries, ikke uavhengige.)*\n")

    # Per-trade table
    lines.append("## Per-trade summary\n")
    lines.append("| id | side | entry | bars | MFE | MAE | giveback% | V3 max | realized | tags |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        entry_t = (r["entry_time"] or "")[11:16] if r["entry_time"] else "?"
        rea = r["realized_pnl_bps"]
        rea_str = f"{float(rea):+.1f}" if rea != "" and rea is not None else "?"
        tag_short = r["tags"].replace(",", " · ")[:60]
        lines.append(
            f"| {r['trade_id']} | {r['side']} | {entry_t} | {r['n_bars']} | "
            f"{float(r['max_mfe_bps'] or 0):.1f} | {float(r['max_mae_bps'] or 0):.1f} | "
            f"{float(r['mfe_giveback_pct'] or 0)*100:.0f}% | "
            f"{float(r['v3_max_prob'] or 0):.2f} | "
            f"{rea_str} | {tag_short} |"
        )
    lines.append("")

    # Pattern findings
    lines.append("## Pattern findings\n")
    flagged = {
        "MFE_GIVEBACK_50PCT": [],
        "MFE_HIT_BUT_NEGATIVE_EXIT": [],
        "RECOVERY_FROM_DEEP_MAE": [],
        "IQL_HELD_THROUGH_V3_ALARM": [],
        "JOJO_4PLUS": [],
        "RAN_TO_STOP": [],
    }
    for r in rows:
        for t in r["tags"].split(","):
            if t in flagged:
                flagged[t].append(r["trade_id"])
    for tag, ids in flagged.items():
        if ids:
            lines.append(f"- **{tag}** ({len(ids)}): {', '.join(str(i) for i in ids)}")
    if not any(flagged.values()):
        lines.append("- (Ingen patologiske mønstre i dagens trades)")
    lines.append("")

    lines.append("## Per-trade detail\n")
    for r in rows:
        lines.append(f"- [Trade #{r['trade_id']}](trades/{r['trade_id']}.md) — {r['side']} {r['n_bars']}b  pnl={float(r['realized_pnl_bps'] or 0):+.1f}")
    lines.append("")

    lines.append("---\n")
    lines.append("_Datakilder: `trade_journal/trades/*.json`, `trade_journal_index.csv`._\n")
    lines.append("_Bruk denne reviewen til å seede retraining data for IQL — patologiske mønstre er retraining-signal._\n")

    (out_dir / "REVIEW.md").write_text("\n".join(lines))


# ── main ───────────────────────────────────────────────────────────────────

def _generate_review(date_str: str, out_dir: Path) -> int:
    (out_dir / "trades").mkdir(parents=True, exist_ok=True)
    LOG.info(f"Reading trade journals from {TRADE_JSON_DIR}")
    rows = []
    files = sorted(TRADE_JSON_DIR.glob("*.json"))
    iso_date = _to_iso_date(date_str)
    for jf in files:
        try:
            data = json.loads(jf.read_text())
        except Exception as e:
            LOG.warning(f"failed to parse {jf.name}: {e}")
            continue
        # If entry_snapshot is missing (trade resumed before TradeJournal retrofit),
        # backfill from the runner's persisted TradeState (open_trade_<id>.json).
        if data.get("entry_snapshot") is None:
            bf = _backfill_entry_from_state(data.get("trade_id") or "")
            if bf is not None:
                data["entry_snapshot"] = bf
        # Filter to today's trades. Prefer entry_snapshot.entry_time, fall back to
        # first bar timestamp.
        entry_t = (data.get("entry_snapshot") or {}).get("entry_time", "")
        if not entry_t and data.get("v12_bar_decisions"):
            entry_t = data["v12_bar_decisions"][0].get("timestamp", "")
        if not entry_t or not entry_t.startswith(iso_date):
            continue
        s = trade_summary_row(data)
        if not s["entry_time"]:
            s["entry_time"] = entry_t
        # For open trades, override journal-derived metrics with the runner's
        # cumulative TradeState (it preserves the full pre-retrofit history).
        if s["exit_reason"] == "STILL_OPEN":
            state = _load_state(data.get("trade_id") or "")
            if state is not None:
                s["n_bars"] = int(state.get("bars_in_trade") or s["n_bars"])
                s["max_mfe_bps"] = float(state.get("cum_mfe_bps") or s["max_mfe_bps"])
                s["max_mae_bps"] = float(state.get("cum_mae_bps") or s["max_mae_bps"])
                s["realized_pnl_bps"] = float(state.get("current_pnl_bps") or s["realized_pnl_bps"])
                s["v3_max_prob"] = float(state.get("v3_max_prob_in_trade") or s["v3_max_prob"])
                # Recompute giveback against the corrected MFE peak
                final = float(s["realized_pnl_bps"] or 0)
                mfe = float(s["max_mfe_bps"] or 0)
                s["mfe_giveback_bps"] = mfe - final if mfe > 0 else 0.0
                s["mfe_giveback_pct"] = (mfe - final) / mfe if mfe > 0 else 0.0
                # Re-classify tags after metric correction
                s["tags"] = ",".join(_classify_pattern(
                    {
                        "max_mfe_bps": s["max_mfe_bps"],
                        "max_mae_bps": s["max_mae_bps"],
                        "final_pnl_bps": s["realized_pnl_bps"],
                        "mfe_giveback_pct": s["mfe_giveback_pct"],
                        "iql_held_through_v3_count": s["iql_held_through_v3_count"],
                        "pnl_zero_crossings": s["pnl_zero_crossings"],
                    },
                    None,   # still open
                ))
        rows.append(s)
        render_trade_detail(data, s, out_dir / "trades" / f"{data.get('trade_id')}.md")

    rows.sort(key=lambda r: r["entry_time"] or "")
    LOG.info(f"  {len(rows)} trades from {date_str}")

    # Write CSV
    if rows:
        csv_path = out_dir / "trades_metrics.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        LOG.info(f"  wrote {csv_path}")

    # Day review
    render_day_review(rows, date_str, out_dir)
    LOG.info(f"  wrote {out_dir / 'REVIEW.md'}")
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="V12 daily trade review")
    p.add_argument("--date", type=str, default=None,
                   help="YYYYMMDD (default: today UTC)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: GX1_DATA/.../daily_reviews/<date>/)")
    p.add_argument("--loop", action="store_true",
                   help="Regenerate review every --interval seconds. "
                        "If --date is omitted, follows the current UTC date "
                        "(auto-rotates at 00:00 UTC).")
    p.add_argument("--interval", type=int, default=120,
                   help="Seconds between rebuilds in --loop mode (default: 120)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ")

    if args.loop:
        import time
        while True:
            date_str = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
            out_dir = Path(args.out_dir) if args.out_dir else REVIEW_BASE / date_str
            try:
                _generate_review(date_str, out_dir)
            except Exception as exc:
                LOG.error(f"review generation failed: {exc}")
            time.sleep(args.interval)
    else:
        date_str = args.date or datetime.now(timezone.utc).strftime("%Y%m%d")
        out_dir = Path(args.out_dir) if args.out_dir else REVIEW_BASE / date_str
        _generate_review(date_str, out_dir)
    return 0


def _to_iso_date(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"


if __name__ == "__main__":
    raise SystemExit(main())
