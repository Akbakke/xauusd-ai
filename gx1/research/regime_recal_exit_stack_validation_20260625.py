"""Validate the regime-adaptive LONG-sizing recalibration THROUGH the exit stack.

The earlier sweep (+0.048 Sharpe 2026) measured the de-size on held-to-K96 TERMINAL.
This re-derives it on EXIT-STACK realized PnL (the phase6 exit-replay: Exit-IQL R_NET_REAL
+ LWR + hard-stop + Strategy-F), on the LIVE CONVICTION-GATE book (NOT the argmax book —
the gotcha), with the multiplier rebuilt in CLOSE-ORDER (a take's mult uses only takes that
CLOSED before its entry) using the ONE-TRUTH gx1.features.regime_adaptive_sizing_v1 tracker.

Inputs:
  - per_candidate_V12_OFF.csv  (phase6 exit-replay on the conviction-gate decisions): realized_pnl_bps, exit_bar
  - decisions_conviction_gate.parquet: candidate_uid, decision_ts_utc, gate_side, trend_regime, margin, atr_bps

Compares cum-equity / Sharpe / DD / worst-trade WITH vs WITHOUT the recal, on 2026 (after a
2025-H2 deque warm-up). Base sizing = live margin^2 x inv-ATR (clamp [0.5,2]); recal multiplies
LONG size only. Run after the phase6 gate-book replay writes per_candidate_V12_OFF.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.features.regime_adaptive_sizing_v1 import RegimeLongWinTracker, TREND_REGIMES

SCOPE = Path("/home/andre2/GX1_DATA/reports/v12_paper_runs/_regime_recal_2026scope")
PER_CAND = SCOPE / "phase6_gatebook" / "per_candidate_V12_OFF.csv"
DECISIONS = SCOPE / "decisions_conviction_gate.parquet"

# live sizing (margin^2 x inv-ATR), launcher op
POW, REF, ATR_REF, ATR_FLOOR, MIN_M, MAX_M = 2.0, 0.3318, 18.0, 6.0, 0.5, 2.0
# recal swept-best
K, BASE, FLOOR, WARMUP = 25, 0.55, 0.30, 12


def base_size(margin, atr_bps):
    m = (max(float(margin), 0.0) ** POW) / REF
    if atr_bps and atr_bps > 0:
        m *= ATR_REF / max(float(atr_bps), ATR_FLOOR)
    return float(np.clip(m, MIN_M, MAX_M))


def metrics(pnl_weighted, label):
    pnl = np.asarray(pnl_weighted, dtype=float)
    if len(pnl) == 0:
        return f"{label}: (no trades)"
    ev = pnl.mean()
    sharpe = ev / (pnl.std() + 1e-9)
    eq = np.cumsum(pnl)
    dd = float((eq - np.maximum.accumulate(eq)).min())
    return (f"{label}: n={len(pnl)} EV={ev:+.2f} Sharpe={sharpe:+.3f} "
            f"total={pnl.sum():+.0f} DD={dd:+.0f} worst={pnl.min():+.1f}")


def main():
    if not PER_CAND.exists():
        print(f"WAIT: {PER_CAND} not written yet (phase6 still running)"); sys.exit(2)
    pc = pd.read_csv(PER_CAND)
    dec = pd.read_parquet(DECISIONS)
    print(f"per_candidate rows={len(pc)}  cols={[c for c in pc.columns if 'pnl' in c.lower() or 'exit' in c.lower() or 'uid' in c.lower()][:8]}")

    key = "candidate_uid" if "candidate_uid" in pc.columns else pc.columns[0]
    df = pc.merge(dec[["candidate_uid", "decision_ts_utc", "gate_side", "trend_regime",
                       "margin", "atr_bps", "action_label_v1"]], on=key, how="inner")
    # only the gate-book TAKES that were scored (realized present), drop SKIP/zero
    df = df[df["action_label_v1"].isin(["TAKE_LONG_NOW", "TAKE_SHORT_NOW"])].copy()
    df["entry_ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    df["realized"] = pd.to_numeric(df["realized_pnl_bps"], errors="coerce")
    df = df.dropna(subset=["realized", "entry_ts"])
    eb = pd.to_numeric(df.get("exit_bar", 0), errors="coerce").fillna(0).clip(lower=0)
    df["close_ts"] = df["entry_ts"] + pd.to_timedelta(eb, unit="m")
    df["base_sz"] = [base_size(m, a) for m, a in zip(df["margin"], df["atr_bps"])]
    df["is_long"] = df["gate_side"].eq("long")
    df["regime"] = df["trend_regime"].astype(str)
    print(f"gate-book takes scored: {len(df)}  longs={int(df['is_long'].sum())} shorts={int((~df['is_long']).sum())}")
    print(f"  date range: {df['entry_ts'].min()} -> {df['entry_ts'].max()}")

    # ── event-ordered close-order deque (mult at entry uses only takes closed before) ──
    tracker = RegimeLongWinTracker(k=K, base=BASE, floor=FLOOR, warmup=WARMUP)
    df = df.sort_values("entry_ts").reset_index(drop=True)
    events = []  # (ts, kind, idx)
    for i, r in df.iterrows():
        events.append((r["entry_ts"], 0, i))   # open
        events.append((r["close_ts"], 1, i))   # close
    events.sort(key=lambda e: (e[0], e[1]))     # close before open at same ts is fine; opens read past only
    recal_mult = np.ones(len(df))
    for ts, kind, i in events:
        if kind == 0:  # open: read mult (longs only)
            if df.at[i, "is_long"]:
                recal_mult[i] = tracker.multiplier(df.at[i, "regime"])
        else:           # close: update deque (longs only)
            if df.at[i, "is_long"] and df.at[i, "regime"] in TREND_REGIMES:
                tracker.record(df.at[i, "regime"], bool(df.at[i, "realized"] > 0))
    df["recal_mult"] = recal_mult

    df["w_off"] = df["realized"] * df["base_sz"]
    df["w_on"] = df["realized"] * df["base_sz"] * df["recal_mult"]

    for period, mask in [("2026", df["entry_ts"] >= "2026-01-01"),
                          ("2025-H2(warmup)", df["entry_ts"] < "2026-01-01")]:
        sub = df[mask]
        print(f"\n=== {period} (n={len(sub)}) ===")
        print("  " + metrics(sub["w_off"], "WITHOUT recal"))
        print("  " + metrics(sub["w_on"], "WITH    recal"))
        lo = sub[sub["is_long"]]
        if len(lo):
            print(f"  longs: n={len(lo)} mean recal_mult={lo['recal_mult'].mean():.3f} "
                  f"(de-sized {100*(lo['recal_mult']<1.0).mean():.0f}% of longs)")
    out = SCOPE / "regime_recal_exit_stack_result.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
