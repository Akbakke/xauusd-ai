"""CONVICTION audit (user vedtak 2026-06-23): does the system trade too many low-conviction (small
LONG-vs-SHORT difference) setups? 7 parts, OOT-honest, on forward_outcome (K96, held-to-horizon).

DATA CAVEAT: the entry-IQL Q-values / raw_adv (the LIVE gate metric, thr -37.71) are NOT in
forward_outcome (it's the pre-IQL candidate set). The sizing code documents raw_adv as a near-DEAD
signal (corr 0.04 w/ realized) vs `margin` (V10 top1-top2 conviction) at corr 0.36. So conviction here
= `margin` (the stronger, available signal that sizing uses) + signed dir-gap p_long-p_short. KEY lesson
from the tail-risk audit: report UNIT- AND SIZE-weighted (margin^2 sizing already up-weights conviction
→ a stricter margin-gate may be redundant). Live sizing: clip(margin^2/0.3318 * 14/max(atr,14),0.5,2).
RAM-safe; realized-exit replay = phase6 gate (flagged). No cement touched.
"""
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path

FO = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
K = "K96"
DIP = [f"v10_dip_{i}" for i in range(18)]
TIMING = [f"v10_timing_{i}" for i in range(12)]
OUT = {}


def live_size(margin, atr):
    return np.clip((np.maximum(margin, 0.0) ** 2) / 0.3318 * 14.0 / np.maximum(atr, 14.0), 0.5, 2.0)


def sharpe(p): return float(np.mean(p)/(np.std(p)+1e-9)) if len(p) else float("nan")
def pf(p):
    w = p[p > 0].sum(); l = -p[p < 0].sum(); return float(w/(l+1e-9))
def mdd(p, t):
    if len(p) == 0: return 0.0
    cum = np.cumsum(p[np.argsort(t)]); return float((np.maximum.accumulate(cum)-cum).max())


def load():
    files = sorted(glob.glob(f"{FO}/*.parquet"))
    av = set(pd.read_parquet(files[0]).columns)
    lab = [f"take_now_{s}_{m}_at_{K}_v1" for s in ("long", "short") for m in ("terminal_pnl", "mae_bps", "mfe_bps", "time_to_mfe")]
    cols = [c for c in (DIP+TIMING+["p_long","p_short","p_flat","margin","uncertainty_score","atr_bps",
            "_v1_atr14_canon_v1","trend_regime","decision_ts_utc"]+lab) if c in av]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    long = (df["p_long"] >= df["p_short"]).values
    for m in ("terminal_pnl", "mae_bps", "mfe_bps", "time_to_mfe"):
        df[m] = np.where(long, df[f"take_now_long_{m}_at_{K}_v1"].astype(float), df[f"take_now_short_{m}_at_{K}_v1"].astype(float))
    df = df.dropna(subset=["terminal_pnl", "margin", "p_long"]).reset_index(drop=True)
    df["long"] = (df["p_long"] >= df["p_short"]).astype(int)
    df["atrbps"] = (df["atr_bps"].astype(float) if "atr_bps" in df.columns else df["_v1_atr14_canon_v1"].astype(float)).fillna(14.0)
    df["dgap"] = (df["p_long"].astype(float) - df["p_short"].astype(float)).abs()  # LONG-vs-SHORT difference
    df["sz"] = live_size(df["margin"].astype(float).values, df["atrbps"].values)
    return df


def main():
    df = load()
    pnl = df["terminal_pnl"].astype(float).values
    mae = df["mae_bps"].astype(float).values
    mfe = df["mfe_bps"].astype(float).values
    dur = df["time_to_mfe"].astype(float).values
    marg = df["margin"].astype(float).values
    dgap = df["dgap"].values
    sz = df["sz"].values
    ts = df["ts"].values
    early = (df["ts"] <= df["ts"].median()).values
    late = (df["ts"] > df["ts"].median()).values
    o26 = (df["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")).values
    print(f"all candidate bars n={len(df)}  margin med={np.median(marg):.3f}  dgap med={np.median(dgap):.3f}\n")

    # ── PART 1: margin deciles ───────────────────────────────────────────────
    print("=" * 64 + "\nPART 1 — MARGIN DECILES (expectancy vs conviction)\n" + "=" * 64)
    dec = pd.qcut(marg, 10, labels=False, duplicates="drop")
    tot = pnl.sum()
    print(f"  {'decile':7s} {'n':>6s} {'win%':>5s} {'exp':>7s} {'PF':>5s} {'Sharpe':>7s} {'MAE':>6s} {'MFE':>6s} {'dur':>5s} {'PnL%':>6s}")
    p1 = {}
    for d in range(10):
        m = dec == d
        if m.sum() == 0: continue
        p1[f"D{d+1}"] = dict(n=int(m.sum()), win=round(float((pnl[m] > 0).mean()), 3), exp=round(float(pnl[m].mean()), 2),
                             pf=round(pf(pnl[m]), 2), sharpe=round(sharpe(pnl[m]), 3), mae=round(float(mae[m].mean()), 1),
                             mfe=round(float(mfe[m].mean()), 1), pnl_share=round(float(pnl[m].sum()/tot*100), 1))
        print(f"  D{d+1:<6d} {m.sum():6d} {(pnl[m]>0).mean()*100:4.1f}% {pnl[m].mean():+7.2f} {pf(pnl[m]):5.2f} "
              f"{sharpe(pnl[m]):7.3f} {mae[m].mean():6.1f} {mfe[m].mean():6.1f} {np.nanmean(dur[m]):5.0f} {pnl[m].sum()/tot*100:5.1f}%")
    OUT["part1_margin_deciles"] = p1

    # ── PART 2: gate threshold sweep (margin pctile), UNIT + SIZE weighted ────
    print("\n" + "=" * 64 + "\nPART 2 — MARGIN-GATE THRESHOLD SWEEP (keep margin>=pctile)\n" +
          "  (raw_adv live-gate not reconstructable from FO; this is the margin-gate)\n" + "=" * 64)
    def gate_report(splitname, smask):
        pv, sv, tv = pnl[smask], sz[smask], ts[smask]
        mv = marg[smask]
        base_u = dict(n=len(pv), pnl=round(float(pv.sum())), pf=round(pf(pv), 2), sharpe=round(sharpe(pv), 3), dd=round(mdd(pv, tv)))
        base_s = dict(pnl=round(float((sv*pv).sum())), pf=round(pf(sv*pv), 2), sharpe=round(sharpe(sv*pv), 3), dd=round(mdd(sv*pv, tv)))
        print(f"  [{splitname}] base n={base_u['n']} | UNIT Sharpe={base_u['sharpe']} PF={base_u['pf']} DD={base_u['dd']} "
              f"| SIZE Sharpe={base_s['sharpe']} PF={base_s['pf']} PnL={base_s['pnl']}")
        for pct in (10, 20, 30, 40, 50, 60, 70, 80, 90):
            cut = np.percentile(mv, pct); keep = mv >= cut
            u_sh = sharpe(pv[keep]); s_sh = sharpe((sv*pv)[keep])
            print(f"     keep margin>={pct:2d}pct (top {100-pct}%): n={keep.sum():6d} "
                  f"UNIT Sh={u_sh:.3f} PF={pf(pv[keep]):.2f} | SIZE Sh={s_sh:.3f} PF={pf((sv*pv)[keep]):.2f} PnL={(sv*pv)[keep].sum():.0f}")
            OUT.setdefault(f"part2_{splitname}", {})[f"keep>={pct}"] = dict(n=int(keep.sum()), unit_sharpe=round(u_sh, 3),
                                                                            size_sharpe=round(s_sh, 3), size_pnl=round(float((sv*pv)[keep].sum())))
    for sn, sm in (("OOT-early", early), ("OOT-late", late), ("2026", o26)):
        gate_report(sn, sm)
    print("  NOTE: realized-exit replay = phase6 gate (heavy/separate) — NOT inline.")

    # ── PART 3: dir-gap segments ─────────────────────────────────────────────
    print("\n" + "=" * 64 + "\nPART 3 — DIR-GAP |p_long-p_short| SEGMENTS (best trades from big Q-diff?)\n" + "=" * 64)
    segs = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    qp = np.array([np.percentile(dgap, p) for p in range(0, 101)])
    print(f"  {'gap-seg':10s} {'n':>6s} {'exp':>7s} {'PF':>5s} {'Sharpe':>7s} {'MAE':>6s} {'MFE':>6s}")
    for lo, hi in segs:
        m = (dgap >= np.percentile(dgap, lo)) & (dgap < np.percentile(dgap, hi)) if hi < 100 else (dgap >= np.percentile(dgap, lo))
        if m.sum() == 0: continue
        print(f"  {lo:3d}-{hi:<3d}%   {m.sum():6d} {pnl[m].mean():+7.2f} {pf(pnl[m]):5.2f} {sharpe(pnl[m]):7.3f} {mae[m].mean():6.1f} {mfe[m].mean():6.1f}")
        OUT.setdefault("part3", {})[f"{lo}-{hi}"] = dict(n=int(m.sum()), exp=round(float(pnl[m].mean()), 2), sharpe=round(sharpe(pnl[m]), 3))

    # ── PART 4: bad trades — are losers low-conviction? ──────────────────────
    print("\n" + "=" * 64 + "\nPART 4 — ARE LOSERS LOW-CONVICTION? (bottom-10% vs top-10%)\n" + "=" * 64)
    qual = mfe / np.maximum(mae, 1)
    for nm, score in (("by PnL", pnl), ("by MFE/MAE", qual)):
        lo = score <= np.percentile(score, 10); hi = score >= np.percentile(score, 90)
        print(f"  [{nm}] bottom-10%: margin={marg[lo].mean():.3f} dgap={dgap[lo].mean():.3f} | "
              f"top-10%: margin={marg[hi].mean():.3f} dgap={dgap[hi].mean():.3f}")
        OUT.setdefault("part4", {})[nm] = dict(bottom_margin=round(float(marg[lo].mean()), 3), top_margin=round(float(marg[hi].mean()), 3),
                                               bottom_dgap=round(float(dgap[lo].mean()), 3), top_dgap=round(float(dgap[hi].mean()), 3))

    # ── PART 5: "gulltrade" pattern (LONG, MAE>100, ends -20..+20) ───────────
    print("\n" + "=" * 64 + "\nPART 5 — 'VISUALLY-WRONG' LONGS (MAE>100, terminal -20..+20)\n" + "=" * 64)
    glt = (df["long"].values == 1) & (mae >= 100) & (pnl >= -20) & (pnl <= 20)
    win = pnl >= np.percentile(pnl, 90)
    if glt.sum() > 0:
        atrp = pd.Series(df["atrbps"].values).rank(pct=True).values
        print(f"  pattern n={int(glt.sum())} ({glt.mean()*100:.2f}% of bars):")
        print(f"    margin={marg[glt].mean():.3f}  dgap={dgap[glt].mean():.3f}  ATR-pctile={atrp[glt].mean():.2f}  "
              f"uncertainty={df['uncertainty_score'].values[glt].mean():.3f}")
        print(f"  vs TOP-10% winners: margin={marg[win].mean():.3f} dgap={dgap[win].mean():.3f} ATR-pctile={atrp[win].mean():.2f}")
        # regime mix
        if "trend_regime" in df.columns:
            print(f"    pattern regime mix: {pd.Series(df['trend_regime'].values[glt]).value_counts(normalize=True).round(2).to_dict()}")
        OUT["part5"] = dict(n=int(glt.sum()), patt_margin=round(float(marg[glt].mean()), 3), win_margin=round(float(marg[win].mean()), 3),
                            patt_dgap=round(float(dgap[glt].mean()), 3), win_dgap=round(float(dgap[win].mean()), 3))

    # ── PART 6: counterfactual keep top-X% by margin, UNIT + SIZE, 3 splits ──
    print("\n" + "=" * 64 + "\nPART 6 — COUNTERFACTUAL: keep top-X% by margin (fewer, more convinced)\n" + "=" * 64)
    for sn, sm in (("OOT-late", late), ("2026", o26)):
        pv, sv, tv, mv = pnl[sm], sz[sm], ts[sm], marg[sm]
        print(f"  [{sn}] (UNIT then SIZE-weighted)")
        for topf in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            cut = np.percentile(mv, (1-topf)*100); keep = mv >= cut
            wu = pv[keep]; ws = (sv*pv)[keep]
            print(f"     top {int(topf*100):3d}%: n={keep.sum():6d} | UNIT PnL={wu.sum():8.0f} Sh={sharpe(wu):.3f} PF={pf(wu):.2f} DD={mdd(wu,tv[keep]):.0f} "
                  f"| SIZE PnL={ws.sum():8.0f} Sh={sharpe(ws):.3f} PF={pf(ws):.2f} DD={mdd(ws,tv[keep]):.0f}")
            OUT.setdefault(f"part6_{sn}", {})[f"top{int(topf*100)}"] = dict(n=int(keep.sum()), unit_sharpe=round(sharpe(wu), 3),
                                                                            size_sharpe=round(sharpe(ws), 3), size_pnl=round(float(ws.sum())))
    print("  NOTE: realized-exit replay = phase6 gate (heavy/separate) — NOT inline.")

    Path("/tmp/conviction_audit.json").write_text(json.dumps(OUT, indent=2, default=str))
    print("\nWROTE /tmp/conviction_audit.json")


if __name__ == "__main__":
    main()
