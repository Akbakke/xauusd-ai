#!/usr/bin/env python3
"""TRUE exit-chain net-capture — productized SAMPLED nightly tier (research tool).

WHY THIS EXISTS
---------------
The hourly counterfactual daemon enriches every SKIP with a CHEAP, vectorized
Strategy-F-FLOOR managed net (``proposed_side_managed_net_bps``, produced by
``gx1.execution.v12_counterfactual_replay.giveback_managed_net``). That floor
applies ONLY the model-free Strategy-F overlay (PROFIT-LOCK + BREAKEVEN-CUT) and
is an UPPER BOUND on managed re-capture: the LIVE exit chain ALSO runs V3
should_exit + Exit-IQL Q + Strategy-F rules 3/4, which exit EARLIER on winners.

This module GROUNDS that floor on the highest-value SKIPs by running the REAL
exit chain forward bar-by-bar — the exact live per-bar primitive
``V12Pipeline.make_exit_decision`` (v12_pipeline.py:550). At ~37 s / replay it is
INFEASIBLE for the ~12k SKIPs/day, so it is SAMPLED: the top-K SKIPs (ranked by
the cheap managed-floor signal) get the true chain. It runs in the NIGHTLY loop,
never the hourly cf-daemon.

ONE-TRUTH REUSE (rule 7 — never reimplement the exit logic)
-----------------------------------------------------------
- ``V12Pipeline.make_exit_decision``  (gx1/execution/v12_pipeline.py:550) — the
  live per-M1-bar exit primitive (advances TradeState, runs V3 multi-TF, Exit-IQL
  Q, Strategy-F overlay, returns HOLD / EXIT_NOW).
- ``TradeState.open`` / ``.update_bar``  (gx1/execution/v12_trade_state.py:116/174).
- ``load_m1_window``                    (gx1/execution/v12_counterfactual_replay.py:62).
- ``giveback_managed_net``              (gx1/execution/v12_counterfactual_replay.py:170)
  — the SAME cheap floor the daemon writes; used here for the floor-vs-true gap.
- Bundles resolve ONLY through ``PROJECT_STATE_artifacts.json`` via
  ``V12Pipeline.load_default`` → ``gx1_guards`` (rule 8). No glob / latest / mtime.

This is a gx1/research/ tool (NOT live-chain): it takes no trading decision,
touches no contract, retrains nothing. It only MEASURES the chain forward on
historical SKIPs and writes an analysis parquet + meta sidecar.

DETERMINISM: stable candidate ordering (sort by managed-floor desc, then ts, then
proposed_side); models are eval-mode; the only inputs are the fixed M1 tape +
frozen v10 snapshot from the journal. No RNG, no Date.now in the numbers (the
wall-clock appears only in the meta sidecar's ``generated_at``).

CLI
---
  python -m gx1.research.exit_netcapture \
      --skip-source <counterfactual_*.jsonl | parquet of (ts,side[,managed_net])> \
      --top-k 20 --out <parquet>

Falls back to the raw-peak signal (``proposed_side_mfe_bps``) for RANKING when a
report predates the managed-net enrichment, and ALWAYS computes the managed floor
itself so the floor-vs-true gap is reported regardless of report vintage. If the
pipeline load or every replay fails, it writes the managed-floor-only view and
exits non-fatally (the caller decides) — it NEVER fabricates a true-net number.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Live-parity env (mirror the reference scratch script /tmp/exit_netcap/step3_continuation.py).
# PURE_PHASE6 + REGIME_V4 = the live serve config. Strategy-F ENABLED (default) so the
# true chain == the live discipline (we want the TRUE managed exit, not a continuation).
os.environ.setdefault("GX1_PURE_PHASE6", "1")
os.environ.setdefault("GX1_REGIME_V4", "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# Default horizon: the 4h managed-net judging window the cheap floor uses
# (CORRECT_SKIP_K=240 M1 bars in v12_counterfactual_replay).
DEFAULT_MAX_BARS = 240


# ── candidate selection ──────────────────────────────────────────────────────
def _coerce_ts(v: Any) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(v, utc=True)
        return ts if pd.notna(ts) else None
    except Exception:
        return None


def load_skip_candidates(skip_source: Path) -> pd.DataFrame:
    """Load SKIP candidates from a counterfactual jsonl OR a (ts,side) parquet.

    Returns a frame with at least: ts (UTC tz-aware), proposed_side,
    managed_floor_bps (the ranking signal; may be NaN), raw_peak_bps, bid, ask,
    v10_snapshot (dict or None). One row per replayable SKIP.
    """
    src = Path(skip_source)
    if src.suffix == ".parquet":
        df = pd.read_parquet(src)
        df = df.rename(columns={"timestamp": "ts", "time": "ts"})
        if "ts" not in df.columns or "side" not in df.columns:
            raise ValueError(
                f"parquet skip-source must have (ts,side) columns; got {list(df.columns)}")
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["proposed_side"] = df["side"].astype(str).str.lower()
        if "managed_net_bps" in df.columns:
            df["managed_floor_bps"] = pd.to_numeric(df["managed_net_bps"], errors="coerce")
        elif "proposed_side_managed_net_bps" in df.columns:
            df["managed_floor_bps"] = pd.to_numeric(
                df["proposed_side_managed_net_bps"], errors="coerce")
        else:
            df["managed_floor_bps"] = np.nan
        for c in ("raw_peak_bps", "bid", "ask"):
            if c not in df.columns:
                df[c] = np.nan
        if "v10_snapshot" not in df.columns:
            df["v10_snapshot"] = [None] * len(df)
        return df[["ts", "proposed_side", "managed_floor_bps",
                   "raw_peak_bps", "bid", "ask", "v10_snapshot"]]

    # jsonl counterfactual report
    rows: list[dict[str, Any]] = []
    for line in src.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        decision = (ev.get("v12_decision") or {}).get("action", "")
        order_status = ev.get("order_status", "")
        is_skip = decision == "SKIP" or order_status == "BLOCKED_BY_GATE"
        if not is_skip:
            continue
        ts = _coerce_ts(ev.get("ts_utc"))
        side = ev.get("proposed_side")
        if ts is None or side not in ("long", "short"):
            continue
        snap = (ev.get("v12_decision") or {}).get("_v10_snapshot")
        rows.append(dict(
            ts=ts,
            proposed_side=side,
            managed_floor_bps=ev.get("proposed_side_managed_net_bps"),
            raw_peak_bps=ev.get("proposed_side_mfe_bps"),
            bid=ev.get("bid"),
            ask=ev.get("ask"),
            v10_snapshot=snap,
        ))
    if not rows:
        return pd.DataFrame(columns=["ts", "proposed_side", "managed_floor_bps",
                                     "raw_peak_bps", "bid", "ask", "v10_snapshot"])
    df = pd.DataFrame(rows)
    df["managed_floor_bps"] = pd.to_numeric(df["managed_floor_bps"], errors="coerce")
    df["raw_peak_bps"] = pd.to_numeric(df["raw_peak_bps"], errors="coerce")
    return df


def rank_top_k(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Top-K by the managed-floor signal (the action-2 signal). Falls back to the
    raw peak when the report predates managed-net enrichment. DETERMINISTIC:
    stable secondary sort by ts then proposed_side."""
    if df.empty:
        return df
    df = df.copy()
    # rank key: managed floor if any present, else raw peak
    if df["managed_floor_bps"].notna().any():
        df["_rank_signal"] = df["managed_floor_bps"]
        df["_rank_source"] = "managed_floor"
    else:
        df["_rank_signal"] = df["raw_peak_bps"]
        df["_rank_source"] = "raw_peak"
    df = df.sort_values(
        ["_rank_signal", "ts", "proposed_side"],
        ascending=[False, True, True], kind="mergesort"
    ).head(int(top_k)).reset_index(drop=True)
    return df


# ── snapshot reconstruction (mirror the reference parse_snapshot) ─────────────
def parse_snapshot(raw: Any) -> dict[str, Any] | None:
    """The journal serialized numpy arrays as strings; rebuild the dict TradeState
    expects (direction_probs/logits as float lists). Identical to the reference
    /tmp/exit_netcap/step3_continuation.py:parse_snapshot."""
    if not isinstance(raw, dict):
        return None
    s = dict(raw)
    for k in ("direction_probs", "direction_logits"):
        v = s.get(k)
        if isinstance(v, str):
            try:
                s[k] = [float(x) for x in v.strip("[]").split()]
            except ValueError:
                pass
    return s


# ── the core primitive ───────────────────────────────────────────────────────
def replay_exit_chain_forward(
    pipe,
    m1: pd.DataFrame,
    entry_ts: pd.Timestamp,
    side: str,
    entry_bid: float,
    entry_ask: float,
    v10_snapshot: dict[str, Any] | None,
    max_bars: int = DEFAULT_MAX_BARS,
    trade_id: str | None = None,
) -> dict[str, Any]:
    """Step the REAL live exit chain forward from (entry_ts, side) for up to
    max_bars M1 bars, returning the TRUE managed net.

    Reuses the live one-truth primitive ``pipe.make_exit_decision`` (rule 7) — the
    exact function the live runner steps each M1 bar for an open trade. Opens a
    TradeState at the candidate bar on the proposed side (frozen v10 snapshot), then
    HOLD/EXIT per the chain (V3 + Exit-IQL + Strategy-F) until it says EXIT_NOW or
    the horizon caps. Returns the net at the chain's chosen exit.

    Returns dict: {net_bps, exit_bar, exit_reason, n_bars, status}. On any failure
    status carries the reason and net_bps is None (never fabricated).
    """
    if side not in ("long", "short"):
        return dict(status="bad_side", net_bps=None, exit_bar=None,
                    exit_reason=None, n_bars=0)
    if not entry_bid or not entry_ask or entry_bid <= 0 or entry_ask <= entry_bid:
        return dict(status="no_entry_px", net_bps=None, exit_bar=None,
                    exit_reason=None, n_bars=0)

    ent_min = pd.Timestamp(entry_ts).floor("min")
    cap_until = ent_min + pd.Timedelta(minutes=int(max_bars))
    bars = m1.loc[(m1.index >= ent_min) & (m1.index <= cap_until)]
    if len(bars) < 2:
        return dict(status="no_bars", net_bps=None, exit_bar=None,
                    exit_reason=None, n_bars=len(bars))

    try:
        trade = __import__(
            "gx1.execution.v12_trade_state", fromlist=["TradeState"]
        ).TradeState.open_unit_normalized_research(
            entry_ts=ent_min, side=side,
            entry_bid=float(entry_bid), entry_ask=float(entry_ask),
            v10_snapshot=v10_snapshot or {}, trade_id=trade_id,
            normalization_contract="unit_normalized_direction_exit_research_v1")
    except Exception as exc:
        return dict(status=f"open_err:{exc}", net_bps=None, exit_bar=None,
                    exit_reason=None, n_bars=0)

    exit_pnl = None
    exit_bar = None
    exit_reason = None
    last_pnl = None
    n = 0
    for ts, b in bars.iterrows():
        bid = float(b["bid_close"])
        ask = float(b["ask_close"])
        m1_close = float(b["close"]) if "close" in b else (bid + ask) / 2.0
        try:
            ed = pipe.make_exit_decision(trade, pd.Timestamp(ts), bid, ask, m1_close)
        except Exception as exc:
            return dict(status=f"replay_err:{exc}", net_bps=None, exit_bar=None,
                        exit_reason=None, n_bars=n)
        n += 1
        last_pnl = ed.get("current_pnl_bps")
        if ed.get("action_id") == 1:  # EXIT_NOW
            exit_pnl = ed.get("current_pnl_bps")
            exit_bar = ts
            exit_reason = ed.get("decision_source") or "EXIT_NOW"
            break
    if exit_pnl is None:
        # never chose EXIT within horizon → realize at the horizon-capped bar
        exit_pnl = last_pnl
        exit_bar = bars.index[-1]
        exit_reason = "horizon_cap"
    if exit_pnl is None:
        return dict(status="no_pnl", net_bps=None, exit_bar=None,
                    exit_reason=None, n_bars=n)
    return dict(status="ok", net_bps=float(exit_pnl),
                exit_bar=pd.Timestamp(exit_bar).isoformat(),
                exit_reason=str(exit_reason), n_bars=int(n))


# ── driver ───────────────────────────────────────────────────────────────────
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def run(skip_source: Path, top_k: int, out: Path, max_bars: int) -> dict[str, Any]:
    """Load top-K SKIPs, run the true exit chain on each, write per-candidate
    results + a meta sidecar. Returns a summary dict (also printed)."""
    from gx1.execution.v12_counterfactual_replay import (
        load_m1_window, giveback_managed_net, CORRECT_SKIP_K)

    cand = rank_top_k(load_skip_candidates(skip_source), top_k)
    n_cand = len(cand)
    rank_source = cand["_rank_source"].iloc[0] if n_cand else "none"
    print(f"[exit_netcap] {n_cand} top-K SKIP candidates from {skip_source.name} "
          f"(ranked by {rank_source})", flush=True)
    if n_cand == 0:
        print("[exit_netcap] NO replayable SKIP candidates — nothing to do.", flush=True)
        return dict(status="no_candidates", n=0, out=None)

    # ── load the live pipeline ONCE (rule 8: contract-resolved bundles) ──
    print("[exit_netcap] loading V12Pipeline (one-time)...", flush=True)
    t0 = time.time()
    try:
        from gx1.execution.v12_pipeline import V12Pipeline
        pipe = V12Pipeline.load_default()
    except Exception as exc:
        # Pipeline load failed → fall back to the managed-floor-only view (no fabricated true-net).
        print(f"[exit_netcap] PIPELINE LOAD FAILED: {exc}", flush=True)
        return _write_floor_only(cand, out, skip_source, max_bars,
                                 reason=f"pipeline_load_failed:{exc}")
    load_s = time.time() - t0
    print(f"[exit_netcap] pipeline loaded in {load_s:.1f}s  "
          f"cutoff={pipe.prebuilt_loader.cutoff_ts}", flush=True)

    # M1 tape covering all candidates + the horizon tail
    start = cand["ts"].min() - pd.Timedelta(hours=1)
    end = cand["ts"].max() + pd.Timedelta(minutes=int(max_bars) + 60)
    m1 = load_m1_window(start, end)
    if m1 is None or len(m1) == 0:
        return _write_floor_only(cand, out, skip_source, max_bars,
                                 reason="empty_m1_tape")
    m1 = m1.set_index("time").sort_index()
    m1_time_arr = m1.index.values
    print(f"[exit_netcap] M1 tape: {len(m1)} bars  {start} -> {end}", flush=True)

    results: list[dict[str, Any]] = []
    per_replay_ms: list[float] = []
    for _, r in cand.iterrows():
        ts = pd.Timestamp(r["ts"])
        side = r["proposed_side"]
        snap = parse_snapshot(r["v10_snapshot"])
        bid = r.get("bid")
        ask = r.get("ask")
        # managed FLOOR (same cheap one-truth the daemon uses) at the candidate bar.
        idx = int(np.searchsorted(m1_time_arr, ts.floor("min").value, side="left"))
        managed_floor = r.get("managed_floor_bps")
        if managed_floor is None or (isinstance(managed_floor, float) and np.isnan(managed_floor)):
            mf = giveback_managed_net(m1.reset_index(), idx, side, int(max_bars)) \
                if idx < len(m1) else None
            managed_floor = mf
        raw_peak = r.get("raw_peak_bps")

        t_rep = time.time()
        res = replay_exit_chain_forward(
            pipe, m1, ts, side,
            entry_bid=float(bid) if bid else 0.0,
            entry_ask=float(ask) if ask else 0.0,
            v10_snapshot=snap, max_bars=int(max_bars),
            trade_id=f"netcap_{ts.isoformat()}_{side}")
        dt_ms = (time.time() - t_rep) * 1000.0
        if res["status"] == "ok":
            per_replay_ms.append(dt_ms)
        true_net = res.get("net_bps")
        overstate = (float(managed_floor) - float(true_net)) \
            if (managed_floor is not None and true_net is not None) else None
        results.append(dict(
            ts=ts.isoformat(), proposed_side=side, status=res["status"],
            true_net_bps=true_net,
            managed_floor_bps=(float(managed_floor) if managed_floor is not None else None),
            raw_peak_bps=(float(raw_peak) if raw_peak is not None and pd.notna(raw_peak) else None),
            floor_minus_true_bps=overstate,
            exit_bar=res.get("exit_bar"), exit_reason=res.get("exit_reason"),
            n_bars=res.get("n_bars"), replay_ms=round(dt_ms, 1),
        ))
        mf_s = f"{managed_floor:.1f}" if managed_floor is not None else "NA"
        tn_s = f"{true_net:.1f}" if true_net is not None else "NA"
        rp_s = f"{raw_peak:.1f}" if raw_peak is not None and pd.notna(raw_peak) else "NA"
        print(f"[exit_netcap]  {ts} {side}: true_net={tn_s}  managed_floor={mf_s}  "
              f"raw_peak={rp_s}  ({res['status']}, {res.get('n_bars')} bars, {dt_ms:.0f}ms)",
              flush=True)

    out_df = pd.DataFrame(results)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)

    ok = out_df[out_df["status"] == "ok"]
    summary = _summarize(ok, per_replay_ms)
    _write_meta(out, skip_source, n_cand, len(ok), max_bars, rank_source,
                load_s, summary, per_replay_ms)
    _print_summary(out, summary, ok, per_replay_ms)
    return dict(status="ok", n=n_cand, out=str(out), **summary)


def _summarize(ok: pd.DataFrame, per_replay_ms: list[float]) -> dict[str, Any]:
    if len(ok) == 0:
        return dict(n_ok=0)
    tn = ok["true_net_bps"].dropna()
    fmt = ok["floor_minus_true_bps"].dropna()
    s = dict(
        n_ok=int(len(ok)),
        true_net_sum_bps=round(float(tn.sum()), 1),
        true_net_median_bps=round(float(tn.median()), 2),
        true_net_mean_bps=round(float(tn.mean()), 2),
        true_net_positive=int((tn > 0).sum()),
        managed_floor_sum_bps=round(float(ok["managed_floor_bps"].dropna().sum()), 1),
        managed_floor_median_bps=round(float(ok["managed_floor_bps"].dropna().median()), 2)
        if ok["managed_floor_bps"].notna().any() else None,
    )
    if len(fmt):
        s["floor_overstatement_median_bps"] = round(float(fmt.median()), 2)
        s["floor_overstatement_mean_bps"] = round(float(fmt.mean()), 2)
        # overstatement RATIO on winners (managed floor > 0)
        win = ok[(ok["managed_floor_bps"].fillna(0) > 0) & ok["true_net_bps"].notna()]
        if len(win):
            ratio = win["managed_floor_bps"] / win["true_net_bps"].replace(0, np.nan)
            s["floor_vs_true_ratio_median"] = round(float(ratio.median()), 2)
    return s


def _write_meta(out: Path, skip_source: Path, n_cand: int, n_ok: int, max_bars: int,
                rank_source: str, load_s: float, summary: dict, per_replay_ms: list) -> None:
    meta = dict(
        tool="gx1.research.exit_netcapture",
        generated_at=pd.Timestamp.utcnow().isoformat(),
        commit=_git_commit(),
        skip_source=str(skip_source),
        rank_source=rank_source,
        n_candidates=n_cand, n_ok=n_ok, max_bars=int(max_bars),
        pipeline_load_s=round(load_s, 1),
        per_replay_ms_median=round(float(np.median(per_replay_ms)), 1) if per_replay_ms else None,
        per_replay_ms_mean=round(float(np.mean(per_replay_ms)), 1) if per_replay_ms else None,
        bundles=_resolve_bundles(),
        summary=summary,
    )
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _resolve_bundles() -> dict[str, str]:
    """Record the contract-resolved bundles for provenance (rule 8)."""
    try:
        from gx1_guards.artifacts import load_decision_artifact
        return {role: str(load_decision_artifact(role))
                for role in ("xgb", "v10_entry", "v3_exit", "exit_iql")}
    except Exception as exc:
        return {"error": str(exc)}


def _write_floor_only(cand: pd.DataFrame, out: Path, skip_source: Path,
                      max_bars: int, reason: str) -> dict[str, Any]:
    """Pipeline/tape load failed → write the managed-FLOOR-only view (NO fabricated
    true-net). The caller (nightly leg) treats this as a SKIP, not a crash."""
    print(f"[exit_netcap] FALLBACK floor-only ({reason}) — NOT fabricating true-net.",
          flush=True)
    rows = []
    for _, r in cand.iterrows():
        rows.append(dict(
            ts=pd.Timestamp(r["ts"]).isoformat(), proposed_side=r["proposed_side"],
            status=f"floor_only:{reason}", true_net_bps=None,
            managed_floor_bps=(float(r["managed_floor_bps"])
                               if pd.notna(r["managed_floor_bps"]) else None),
            raw_peak_bps=(float(r["raw_peak_bps"]) if pd.notna(r["raw_peak_bps"]) else None),
            floor_minus_true_bps=None, exit_bar=None, exit_reason=None,
            n_bars=0, replay_ms=None))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    _write_meta(out, skip_source, len(cand), 0, max_bars, "floor_only", 0.0,
                dict(n_ok=0, reason=reason), [])
    return dict(status="floor_only", reason=reason, n=len(cand), n_ok=0, out=str(out))


def _print_summary(out: Path, summary: dict, ok: pd.DataFrame, per_replay_ms: list) -> None:
    print()
    print(f"=== TRUE exit-chain net-capture ({summary.get('n_ok', 0)} faithful replays) ===")
    if summary.get("n_ok", 0) == 0:
        print("  no faithful replays — see per-candidate status in", out)
        return
    print(f"  true_net      SUM={summary['true_net_sum_bps']}  "
          f"MEDIAN={summary['true_net_median_bps']}  MEAN={summary['true_net_mean_bps']}  "
          f"positive={summary['true_net_positive']}/{summary['n_ok']}")
    print(f"  managed_floor SUM={summary['managed_floor_sum_bps']}  "
          f"MEDIAN={summary.get('managed_floor_median_bps')}")
    if "floor_overstatement_median_bps" in summary:
        print(f"  PROXY OVERSTATEMENT (floor - true): "
              f"MEDIAN={summary['floor_overstatement_median_bps']} bps  "
              f"MEAN={summary['floor_overstatement_mean_bps']} bps  "
              f"ratio_median={summary.get('floor_vs_true_ratio_median')}")
    if per_replay_ms:
        print(f"  per-replay ms: median={np.median(per_replay_ms):.0f}  "
              f"mean={np.mean(per_replay_ms):.0f}  max={np.max(per_replay_ms):.0f}")
    print(f"  → {out}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-source", required=True, type=Path,
                    help="enriched counterfactual_*.jsonl OR parquet of (ts,side[,managed_net])")
    ap.add_argument("--top-k", type=int, default=20,
                    help="number of highest managed-floor SKIPs to run the true chain on")
    ap.add_argument("--out", required=True, type=Path,
                    help="output parquet of per-candidate true-net results")
    ap.add_argument("--max-bars", type=int, default=DEFAULT_MAX_BARS,
                    help=f"horizon cap in M1 bars (default {DEFAULT_MAX_BARS} = 4h)")
    args = ap.parse_args(argv)
    if not args.skip_source.exists():
        print(f"[exit_netcap] skip-source not found: {args.skip_source}", file=sys.stderr)
        return 2
    res = run(args.skip_source, args.top_k, args.out, args.max_bars)
    # Non-fatal: floor-only / no-candidates still exit 0 (analysis leg is best-effort).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
