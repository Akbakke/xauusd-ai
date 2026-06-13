"""Nightly operating-point A/B: LIVE open-more(−100) + margin²-sizing vs the prior conviction67 baseline.

For each LIVE trade (journal entry, suffix open100_conv_sized) joined to its resolved verdict
(counterfactual_reports/trade_verdicts_*.jsonl → realized_pnl_bps), compute what the OLD conviction67
operating point would have done ON THE SAME candidate: TAKE only if raw_adv ≥ −37.71 (the old gate),
sized by raw_adv (the old sizing), vs LIVE which TAKES at raw_adv ≥ −100 sized by margin². Tally the
daily + cumulative delta — so when live data starts (Monday) the nightly report shows whether opening
to −100 + margin² actually beats conviction67 on REAL trades. No new replay; pure journal arithmetic.

Genuinely-new nightly helper (no existing script compares operating points on the live journal).
Reversible levers being graded: GX1_CONVICTION_THR (−100 vs −37.71), GX1_SIZING_CONV_SRC (margin² vs raw_adv).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

# Old conviction67 operating point (the baseline being compared against).
OLD_THR = float(os.environ.get("GX1_OPCMP_OLD_THR", "-37.71"))
OLD_SIZ_LO = float(os.environ.get("GX1_OPCMP_OLD_LO", "-37.71"))
OLD_SIZ_HI = float(os.environ.get("GX1_OPCMP_OLD_HI", "-13.99"))
OLD_MAX, OLD_MIN = 2.0, 0.5

PAPER = "/home/andre2/GX1_DATA/reports/v12_paper_runs"


def _old_raw_adv_mult(raw_adv: float) -> float:
    span = max(OLD_SIZ_HI - OLD_SIZ_LO, 1e-6)
    frac = min(max((raw_adv - OLD_SIZ_LO) / span, 0.0), 1.0)
    return min(max(1.0 + (OLD_MAX - 1.0) * frac, OLD_MIN), OLD_MAX)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-suffix", default="open100_conv_sized_skipasia_pure_phase6")
    ap.add_argument("--out", default=f"{PAPER}/nightly_learning/op_comparison.json")
    args = ap.parse_args()

    # 1. resolved verdicts (realized_pnl_bps per trade_id) for the live suffix
    verdicts = {}
    for fp in glob.glob(f"{PAPER}/counterfactual_reports/trade_verdicts_*{args.live_suffix}*.jsonl"):
        for line in open(fp):
            try:
                v = json.loads(line)
            except Exception:
                continue
            if v.get("resolved") and v.get("realized_pnl_bps") is not None:
                verdicts[v["trade_id"]] = v

    # 2. live entries (raw_adv + margin + live units_mult) from the journals
    entries = {}
    for fp in glob.glob(f"{PAPER}/v12_paper_journal_*{args.live_suffix}*.jsonl"):
        for line in open(fp):
            try:
                e = json.loads(line)
            except Exception:
                continue
            tid = e.get("trade_id") or e.get("order_id")
            if tid and ("sizing_raw_adv" in e or "sizing_margin" in e):
                entries[tid] = e

    if not verdicts or not entries:
        out = {"status": "no_live_data_yet", "n_verdicts": len(verdicts), "n_entries": len(entries),
               "note": "open100+margin² goes live Monday; this leg activates once trades resolve."}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"[op-cmp] {out['status']} (verdicts={len(verdicts)} entries={len(entries)})")
        return 0

    by_day = defaultdict(lambda: {"n_live": 0, "live_bps": 0.0, "n_old_take": 0, "old_bps": 0.0,
                                  "n_new_trades": 0, "new_bps": 0.0})
    tot = {"n_live": 0, "live_bps": 0.0, "n_old_take": 0, "old_bps": 0.0, "n_new_trades": 0, "new_bps": 0.0}
    for tid, v in verdicts.items():
        e = entries.get(tid)
        if e is None:
            continue
        pnl = float(v["realized_pnl_bps"])
        live_mult = float(e.get("units_multiplier_applied", 1.0) or 1.0)
        raw_adv = float(e.get("sizing_raw_adv", 0.0) or 0.0)
        day = str(v.get("ts_utc", ""))[:10]
        live_pnl = pnl * live_mult
        d = by_day[day]
        for acc in (d, tot):
            acc["n_live"] += 1
            acc["live_bps"] += live_pnl
        if raw_adv >= OLD_THR:                       # conviction67 would also take it
            old_pnl = pnl * _old_raw_adv_mult(raw_adv)
            for acc in (d, tot):
                acc["n_old_take"] += 1
                acc["old_bps"] += old_pnl
        else:                                        # the trade open-more ADDED (conviction67 skipped)
            for acc in (d, tot):
                acc["n_new_trades"] += 1
                acc["new_bps"] += live_pnl

    out = {
        "status": "ok", "live_suffix": args.live_suffix, "baseline": "conviction67 (thr -37.71, raw_adv sizing)",
        "cumulative": {**tot, "delta_open100_minus_conv67_bps": round(tot["live_bps"] - tot["old_bps"], 1)},
        "per_day": {day: {**v, "delta_bps": round(v["live_bps"] - v["old_bps"], 1)} for day, v in sorted(by_day.items())},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    c = out["cumulative"]
    print(f"[op-cmp] LIVE open100+margin² {c['live_bps']:.0f}bps ({c['n_live']} trades) vs conv67 "
          f"{c['old_bps']:.0f}bps ({c['n_old_take']} would-take) | Δ={c['delta_open100_minus_conv67_bps']:+.0f}bps "
          f"| open-more added {c['n_new_trades']} trades worth {c['new_bps']:+.0f}bps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
