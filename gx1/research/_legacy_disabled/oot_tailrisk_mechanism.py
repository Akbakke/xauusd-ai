"""MECHANISM: why does a tail_risk SKIP help when inverse-ATR SIZING already uses vol? (user vedtak 2026-06-23)
7 parts on forward_outcome (taken-proxy top-20% margin, K96 held-to-horizon, UNIT base + LIVE sizing applied).
RAM-safe sequential (no t-SNE). Realized-exit replay = phase6 gate (flagged, not inline). No cement touched.

LIVE sizing (exact, v12_paper_runner.size_units, op-point both/pow2/ref0.3318/atr_ref=floor=14/clamp[0.5,2]):
  m = clip( (margin^2)/0.3318 * 14/max(atr_bps,14), 0.5, 2.0 )
"""
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

FO = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
K = "K96"
RNG = np.random.default_rng(20260623)
TAIL = [f"v10_tail_risk_{i}" for i in range(6)]
DIP = [f"v10_dip_{i}" for i in range(18)]
TIMING = [f"v10_timing_{i}" for i in range(12)]
ATRF = ["atr_bps", "_v1_atr14_canon_v1", "atr_canon_v1", "atr50_canon_v1", "atr_z_canon_v1",
        "_v1h1_atr_canon_v1", "_v1h4_atr_canon_v1", "rvol_20_canon_v1", "_v1_pk_sigma20_canon_v1"]
OUT = {}


def live_size_mult(margin, atr_bps):
    m = (np.maximum(margin, 0.0) ** 2) / 0.3318
    vol = 14.0 / np.maximum(atr_bps, 14.0)
    return np.clip(m * vol, 0.5, 2.0)


def load():
    files = sorted(glob.glob(f"{FO}/*.parquet"))
    av = set(pd.read_parquet(files[0]).columns)
    lab = [f"take_now_{s}_{m}_at_{K}_v1" for s in ("long", "short") for m in ("terminal_pnl", "mae_bps", "mfe_bps", "time_to_mfe")]
    cols = [c for c in (TAIL + DIP + TIMING + ATRF + ["p_long", "p_short", "margin", "uncertainty_score",
            "decision_ts_utc"] + lab) if c in av]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    long = (df["p_long"] >= df["p_short"]).values
    for m in ("terminal_pnl", "mae_bps", "mfe_bps", "time_to_mfe"):
        df[m] = np.where(long, df[f"take_now_long_{m}_at_{K}_v1"].astype(float), df[f"take_now_short_{m}_at_{K}_v1"].astype(float))
    df = df.dropna(subset=["terminal_pnl", "mae_bps", "p_long"]).reset_index(drop=True)
    df = df[df["margin"].astype(float) >= np.nanpercentile(df["margin"].astype(float), 80)].reset_index(drop=True)
    df["atrbps"] = df["atr_bps"].astype(float).fillna(df["_v1_atr14_canon_v1"].astype(float)) if "atr_bps" in df.columns else df["_v1_atr14_canon_v1"].astype(float)
    df["tail"] = np.abs(df[TAIL].astype(float).values).mean(axis=1)
    df["size"] = live_size_mult(df["margin"].astype(float).values, df["atrbps"].values)
    return df


def sharpe(p): return float(np.mean(p) / (np.std(p) + 1e-9))
def pf(p): w = p[p > 0].sum(); l = -p[p < 0].sum(); return float(w / (l + 1e-9))
def mdd(p, t):
    cum = np.cumsum(p[np.argsort(t)]); return float((np.maximum.accumulate(cum) - cum).max())


def main():
    df = load()
    pnl = df["terminal_pnl"].astype(float).values
    mae = df["mae_bps"].astype(float).values
    mfe = df["mfe_bps"].astype(float).values
    dur = df["time_to_mfe"].astype(float).values
    atr = df["atrbps"].values
    sz = df["size"].values
    ts = df["ts"].values
    print(f"taken-proxy n={len(df)}  atr_bps med={np.median(atr):.1f}  size med={np.median(sz):.2f}")

    # ── PART 1: ATR-percentile buckets ───────────────────────────────────────
    print("\n" + "=" * 64 + "\nPART 1 — ATR-PERCENTILE BUCKETS (knee where expectancy collapses?)\n" + "=" * 64)
    pcts = [0, 50, 70, 80, 90, 95, 99, 100]
    edges = np.percentile(atr, pcts)
    print(f"  {'bucket':9s} {'n':>6s} {'exp':>7s} {'Sharpe':>7s} {'PF':>5s} {'avgMAE':>7s} {'avgMFE':>6s} {'win%':>5s} {'avgSize':>7s} {'PnL%tot':>8s}")
    tot = pnl.sum()
    p1 = {}
    for i in range(len(pcts) - 1):
        m = (atr >= edges[i]) & (atr < edges[i + 1]) if i < len(pcts) - 2 else (atr >= edges[i])
        if m.sum() == 0: continue
        row = dict(n=int(m.sum()), exp=round(float(pnl[m].mean()), 2), sharpe=round(sharpe(pnl[m]), 3),
                   pf=round(pf(pnl[m]), 2), mae=round(float(mae[m].mean()), 1), mfe=round(float(mfe[m].mean()), 1),
                   win=round(float((pnl[m] > 0).mean()), 3), size=round(float(sz[m].mean()), 2),
                   pnl_share=round(float(pnl[m].sum() / tot * 100), 1))
        p1[f"{pcts[i]}-{pcts[i+1]}"] = row
        print(f"  {pcts[i]:3d}-{pcts[i+1]:<3d}   {row['n']:6d} {row['exp']:+7.2f} {row['sharpe']:7.3f} {row['pf']:5.2f} "
              f"{row['mae']:7.1f} {row['mfe']:6.1f} {row['win']*100:4.1f}% {row['size']:7.2f} {row['pnl_share']:7.1f}%")
    OUT["part1"] = p1

    # ── PART 2: tail_risk residual vs ATR ────────────────────────────────────
    print("\n" + "=" * 64 + "\nPART 2 — tail_risk RESIDUAL (after ATR): info beyond volatility?\n" + "=" * 64)
    A = df[[c for c in ATRF if c in df.columns]].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    lr = LinearRegression().fit(A, df["tail"].values)
    resid = df["tail"].values - lr.predict(A)
    print(f"  tail_risk ~ ATR  R²={lr.score(A, df['tail'].values):.3f}")
    early = (df["ts"] <= df["ts"].median()).values; late = ~early
    def auc1(x, y):
        x = x.reshape(-1, 1)
        c = HistGradientBoostingClassifier(max_depth=3, max_iter=150, min_samples_leaf=150, early_stopping=True, random_state=0).fit(x[early], y[early])
        yv = y[late]
        return round(roc_auc_score(yv, c.predict_proba(x[late])[:, 1]), 3) if len(np.unique(yv)) > 1 else float("nan")
    targets = {"PnL>0": (pnl > 0).astype(int), "MAE>=150": (mae >= 150).astype(int),
               "MFE>median": (mfe > np.median(mfe)).astype(int),
               "quality(MFE/MAE>median)": ((mfe / np.maximum(mae, 1)) > np.median(mfe / np.maximum(mae, 1))).astype(int)}
    print(f"  {'target':26s} {'rawTail':>8s} {'ATR(multi)':>11s} {'tailResid':>10s}")
    for tn, y in targets.items():
        a_tail = auc1(df["tail"].values, y)
        a_atr = auc1(atr, y)
        a_res = auc1(resid, y)
        print(f"  {tn:26s} {a_tail:8.3f} {a_atr:11.3f} {a_res:10.3f}")
        OUT.setdefault("part2", {})[tn] = dict(raw_tail=a_tail, atr=a_atr, resid=a_res)

    # ── PART 3: removed-20% vs kept-80% ──────────────────────────────────────
    print("\n" + "=" * 64 + "\nPART 3 — REMOVED top-20% tail_risk vs KEPT 80%\n" + "=" * 64)
    cut = np.percentile(df["tail"].values, 80)
    rem = df["tail"].values >= cut; kep = ~rem
    for nm, m in (("REMOVED(top20% tail)", rem), ("KEPT(80%)", kep)):
        print(f"  {nm:22s} n={m.sum():5d} netPnL={pnl[m].sum():8.0f} Sharpe={sharpe(pnl[m]):.3f} PF={pf(pnl[m]):.2f} "
              f"win={(pnl[m]>0).mean():.3f} MFE={mfe[m].mean():.0f} MAE={mae[m].mean():.0f}")
    # DD attribution: full-portfolio DD vs DD of kept-only
    dd_full = mdd(pnl, ts); dd_kept = mdd(pnl[kep], ts[kep])
    loss_rem = -pnl[rem][pnl[rem] < 0].sum(); loss_all = -pnl[pnl < 0].sum()
    gain_rem = pnl[rem][pnl[rem] > 0].sum(); gain_all = pnl[pnl > 0].sum()
    print(f"  >> removed cohort = {loss_rem/loss_all*100:.1f}% of all GROSS LOSSES, {gain_rem/gain_all*100:.1f}% of all GROSS GAINS")
    print(f"  >> net PnL of removed = {pnl[rem].sum():.0f} (negative = removing them ADDS PnL)")
    print(f"  >> portfolio maxDD full={dd_full:.0f} -> kept-only={dd_kept:.0f} ({(1-dd_kept/dd_full)*100:+.1f}% DD change)")
    OUT["part3"] = dict(removed_net=round(float(pnl[rem].sum())), pct_losses=round(float(loss_rem/loss_all*100), 1),
                        pct_gains=round(float(gain_rem/gain_all*100), 1), dd_full=round(dd_full), dd_kept=round(dd_kept))

    # ── PART 4: sizing vs skip (A-F), 3 splits, SIZE-WEIGHTED PnL ─────────────
    print("\n" + "=" * 64 + "\nPART 4 — SIZING vs SKIP (size-weighted PnL; is gain from less exposure or eliminating trades?)\n" + "=" * 64)
    margin = df["margin"].astype(float).values
    m_conv = (np.maximum(margin, 0.0) ** 2) / 0.3318
    m_vol = 14.0 / np.maximum(atr, 14.0)
    tail_pct = pd.Series(df["tail"].values).rank(pct=True).values
    def variants(mask):
        a = np.clip(m_conv * m_vol, 0.5, 2.0)                       # A current
        b = np.clip(m_conv * m_vol**2, 0.5, 2.0)                    # B double vol penalty
        c = np.clip(m_conv * m_vol**3, 0.5, 2.0)                    # C triple
        d = a * (df["tail"].values < cut)                           # D tail skip (top20 -> 0)
        e = a * np.clip(1.0 - 0.8 * tail_pct, 0.2, 1.0)            # E tail linear sizing (no skip)
        f = a * np.clip(1.0 - 0.8 * tail_pct, 0.2, 1.0) * (tail_pct < 0.95)  # F tail sizing + skip top5%
        return {"A_current": a, "B_2xvol": b, "C_3xvol": c, "D_tailskip20": d, "E_tailsize": e, "F_tailsize+skip5": f}
    for sname, mask in (("OOT-early", (df["ts"] <= df["ts"].median()).values),
                        ("OOT-late", (df["ts"] > df["ts"].median()).values),
                        ("2026", (df["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")).values)):
        print(f"  [{sname}] (size-weighted: PnL=Σ size·pnl)")
        pv = pnl[mask]; tv = ts[mask]
        for vn, szv in variants(mask).items():
            s = szv[mask]; wp = s * pv
            print(f"     {vn:18s} PnL={wp.sum():9.0f} Sharpe={sharpe(wp):.3f} PF={pf(wp):.2f} DD={mdd(wp, tv):8.0f} exposure={s.sum():.0f}")
            OUT.setdefault(f"part4_{sname}", {})[vn] = dict(pnl=round(float(wp.sum())), sharpe=round(sharpe(wp), 3),
                                                            pf=round(pf(wp), 2), exposure=round(float(s.sum())))
    print("  NOTE: realized-exit replay = phase6 gate (heavy/separate) — NOT inline.")

    # ── PART 5: trade quality prediction ─────────────────────────────────────
    print("\n" + "=" * 64 + "\nPART 5 — TRADE QUALITY: does V10 predict quality > direction (~0.62)?\n" + "=" * 64)
    q1 = mfe / np.maximum(mae, 1); q2 = pnl / np.maximum(mae, 1); q3 = pnl / np.maximum(dur, 1)
    qual = {"q1=MFE/MAE": q1, "q2=PnL/MAE": q2, "q3=PnL/dur": q3}
    sigs = {"tail_risk": df["tail"].values, "ATR": atr,
            "dip_heads": df[[c for c in DIP if c in df.columns]].astype(float).values,
            "timing_heads": df[[c for c in TIMING if c in df.columns]].astype(float).values,
            "conviction": df[["margin", "uncertainty_score", "p_long", "p_short"]].astype(float).values}
    def aucX(X, y):
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        c = HistGradientBoostingClassifier(max_depth=3, max_iter=150, min_samples_leaf=150, early_stopping=True, random_state=0).fit(X[early], y[early])
        yv = y[late]
        return round(roc_auc_score(yv, c.predict_proba(X[late])[:, 1]), 3) if len(np.unique(yv)) > 1 else float("nan")
    for qn, qv in qual.items():
        ybin = (qv > np.median(qv)).astype(int)
        print(f"  predict high {qn}: " + "  ".join(f"{s}={aucX(v, ybin)}" for s, v in sigs.items()))
        OUT.setdefault("part5", {})[qn] = {s: aucX(v, ybin) for s, v in sigs.items()}

    # ── PART 6: exit-stack interaction PROXY (static thresholds on K96 path) ──
    print("\n" + "=" * 64 + "\nPART 6 — EXIT-STACK PROXY on the REMOVED cohort (static path-threshold estimate)\n" + "=" * 64)
    r = rem
    hardstop = (mae[r] >= 80)                                   # -80 would have triggered (adverse >=80)
    sf_lock = (mfe[r] >= 30) & ((mfe[r] - pnl[r]) / np.maximum(mfe[r], 1) >= 0.30)  # Strategy-F profit-lock path
    lwr_hold = (pnl[r] >= 15) & ((mfe[r] - pnl[r]) / np.maximum(mfe[r], 1) < 0.30)  # LWR would HOLD
    print(f"  removed cohort n={int(r.sum())} (held-to-K96 path):")
    print(f"    hard-stop(-80) would trigger: {hardstop.mean()*100:.1f}%  (caps these at -80 vs full K96 MAE {mae[r].mean():.0f})")
    print(f"    Strategy-F profit-lock path:   {sf_lock.mean()*100:.1f}%  (would bank MFE before giveback)")
    print(f"    LWR would HOLD (near peak):     {lwr_hold.mean()*100:.1f}%")
    # how many removed-losers does hard-stop NOT save (MAE between threshold and -80, ended neg, never hit -80)?
    rem_loss = r & (pnl < 0)
    not_stopped = (mae[rem_loss] < 80)
    print(f"    removed LOSERS not reached by -80 stop (MAE<80): {not_stopped.mean()*100:.1f}% -> these are exit-INVISIBLE, skip-only")
    OUT["part6"] = dict(hardstop_pct=round(float(hardstop.mean()*100), 1), sf_pct=round(float(sf_lock.mean()*100), 1),
                        lwr_pct=round(float(lwr_hold.mean()*100), 1), loser_below80_pct=round(float(not_stopped.mean()*100), 1))

    Path("/tmp/tailrisk_mechanism.json").write_text(json.dumps(OUT, indent=2, default=str))
    print("\nWROTE /tmp/tailrisk_mechanism.json")


if __name__ == "__main__":
    main()
