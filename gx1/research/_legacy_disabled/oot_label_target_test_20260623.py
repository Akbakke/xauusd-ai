"""LABEL/TARGET test (2026-06-23, user): "is the entry model weak, or are we training against the wrong
target?" Does entry-score (raw_adv) predict terminal@K96 best, or does it predict MFE / a shorter-horizon
return / the optimal-exit ceiling better — i.e. is the model finding opportunities the held-to-K96 LABEL
then dilutes?

Reuses the cached regime universe (live-policy TAKEs: FOLD_1, conv-gate+DIPFIX, features + raw_adv + side)
and joins multi-K terminal/MFE (chosen side) from forward_outcome. entry-score = raw_adv (the live gate).
HONEST DATA LIMITS (no replay): no 15/30-min label (min K=12=60min); "current-exit" and "LWR" returns need
the per-trade M1 PATH (exit replay) — NOT computed here. Proxies: optimal_exit ≈ MFE@K96 (exit at peak);
fixed-horizon exits = terminal@K12/K24/K48 (deployable time-exit policies). Strict OOT throughout (train
pre-2026 -> test 2026). Targets binarized at median (top-half) for AUC; rank-IC = spearman(pred, continuous).
"""
from __future__ import annotations
import glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from gx1.research.oot_live_policy_falsification_20260623 import FO

REG = "/tmp/regime_universe_20260623.parquet"
KS = ["K12", "K24", "K48", "K96", "K144", "K192"]

# fixed pre-entry feature set (regime-universe cols; constant across the target-swap)
_FEATS = ["raw_adv", "margin", "p_long", "p_short", "take_minus_alt", "atr_pctl", "atr_z_canon_v1",
          "_v1_atr_z_10_100_canon_v1", "_v1_bb_squeeze_20_2_canon_v1", "vol_ratio_canon_v1", "trend_str",
          "trend_align", "_v1h1_slope5_canon_v1", "_v1h4_slope5_canon_v1", "d1_ema_slope_20_canon_v2_canon_v1",
          "ema100_slope_canon_v1", "ema20_slope_canon_v1", "hour_utc", "range_pos_proxy", "dist_hi96_proxy",
          "breakout96_proxy"]


def feat_cols(d):
    return [c for c in _FEATS if c in d.columns and d[c].notna().any()]


def build():
    u = pd.read_parquet(REG); u["ts"] = pd.to_datetime(u["ts"], utc=True)
    u = u.drop(columns=[c for c in u.columns if c.startswith("take_now_")])  # drop regime-universe's stale K96 joins (avoid merge collision)
    cols = ["candidate_uid"] + [f"take_now_{s}_{m}_at_{k}_v1" for s in ("long", "short")
                                for m in ("terminal_pnl", "mfe_bps", "mae_bps", "time_to_mfe") for k in KS]
    av = set(pd.read_parquet(sorted(glob.glob(f"{FO}/*.parquet"))[0]).columns)
    cols = [c for c in cols if c in av]
    lab = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(f"{FO}/*.parquet"))],
                    ignore_index=True).drop_duplicates("candidate_uid")
    d = u.merge(lab, left_on="uid", right_on="candidate_uid", how="left")
    isL = d["side"] == "long"
    for m in ("terminal_pnl", "mfe_bps", "mae_bps", "time_to_mfe"):
        for k in KS:
            cl, cs = f"take_now_long_{m}_at_{k}_v1", f"take_now_short_{m}_at_{k}_v1"
            if cl in d and cs in d:
                d[f"{m}_{k}"] = np.where(isL, d[cl], d[cs])
    return d


# TARGET set: current label + shorter time-exits + MFE (opportunity/optimal ceiling) + MAE
TARGETS = {
    "terminal_K96 (CURRENT label)": "terminal_pnl_K96",
    "terminal_K12 (60min exit)": "terminal_pnl_K12",
    "terminal_K24 (120min exit)": "terminal_pnl_K24",
    "terminal_K48 (4h exit)": "terminal_pnl_K48",
    "MFE_K12 (best within 60min)": "mfe_bps_K12",
    "MFE_K24 (best within 120min)": "mfe_bps_K24",
    "MFE_K96 (optimal-exit ceiling)": "mfe_bps_K96",
    "MAE_K96 (drawdown)": "mae_bps_K96",
}


def del1(d):
    print("\n" + "=" * 96 + "\nDEL 1 — entry-score (raw_adv) vs each target  [2026 OOT]\n" + "=" * 96)
    s = d["raw_adv"].values
    d = d.copy(); d["_dec"] = pd.qcut(d["raw_adv"].rank(method="first"), 10, labels=False)
    rows = {}
    print(f"  {'target':34s} {'Spearman':>9} {'Pearson':>9} {'dec-mono':>9} {'topD-edge':>10}")
    for lab, col in TARGETS.items():
        if col not in d: continue
        y = d[col].values
        m = ~np.isnan(y)
        sp = float(spearmanr(s[m], y[m]).correlation); pe = float(np.corrcoef(s[m], y[m])[0, 1])
        decmean = d.groupby("_dec")[col].mean()
        mono = float(spearmanr(decmean.index, decmean.values).correlation)   # +1 = perfectly monotone up
        topd = float(decmean.iloc[-1] - d[col].mean())
        rows[lab] = dict(spearman=round(sp, 4), pearson=round(pe, 4), decile_mono=round(mono, 3), topdecile_edge=round(topd, 1))
        print(f"  {lab:34s} {sp:>+9.4f} {pe:>+9.4f} {mono:>+9.3f} {topd:>+10.1f}")
    return rows


def del2(d):
    print("\n" + "=" * 96 + "\nDEL 2 — raw_adv deciles: what does score rise WITH?  [2026 OOT]\n" + "=" * 96)
    d = d.copy(); d["_dec"] = pd.qcut(d["raw_adv"].rank(method="first"), 10, labels=False) + 1
    cols = {"term_K96": "terminal_pnl_K96", "MFE_K96(opt)": "mfe_bps_K96", "MAE_K96": "mae_bps_K96",
            "term_K24": "terminal_pnl_K24", "term_K48": "terminal_pnl_K48", "t2mfe_K96": "time_to_mfe_K96"}
    g = d.groupby("_dec").agg({**{c: "mean" for c in cols.values()}, "raw_adv": "mean"})
    print(f"  {'D':>2} {'raw_adv':>8} " + " ".join(f"{k:>12}" for k in cols))
    for dec, r in g.iterrows():
        print(f"  {dec:>2} {r['raw_adv']:>+8.1f} " + " ".join(f"{r[c]:>+12.1f}" for c in cols.values()))
    # which target has steepest monotone rise with score-decile (normalized spearman already in del1)
    return {str(k): {c: round(float(v[cc]), 1) for c, cc in cols.items()} for k, v in g.iterrows()}


def cohorts(d, OUT):
    n = len(d)
    losers = d[d["terminal_pnl_K96"] <= 0]
    # DEL3: good-trade-looks-bad: terminal<=0 BUT MFE>=100 AND optimal(=MFE)>=75
    g = d[(d["terminal_pnl_K96"] <= 0) & (d["mfe_bps_K96"] >= 100) & (d["mfe_bps_K96"] >= 75)]
    d2 = d.copy(); d2["score_pct"] = d2["raw_adv"].rank(pct=True)
    g_pct = float(d2.loc[g.index, "score_pct"].mean()) if len(g) else float("nan")
    OUT["del3_good_looks_bad"] = dict(n=int(len(g)), pct_of_losers=round(100 * len(g) / max(len(losers), 1), 1),
                                      pct_of_all=round(100 * len(g) / n, 1), mean_score=round(float(g["raw_adv"].mean()), 1) if len(g) else None,
                                      mean_score_pctile=round(g_pct, 3), mean_mfe=round(float(g["mfe_bps_K96"].mean()), 1) if len(g) else None)
    # DEL4: bad-trade-looks-good: terminal>0 BUT MFE<20 OR optimal≈terminal (mfe-terminal<10)
    b = d[(d["terminal_pnl_K96"] > 0) & ((d["mfe_bps_K96"] < 20) | ((d["mfe_bps_K96"] - d["terminal_pnl_K96"]) < 10))]
    winners = d[d["terminal_pnl_K96"] > 0]
    OUT["del4_bad_looks_good"] = dict(n=int(len(b)), pct_of_winners=round(100 * len(b) / max(len(winners), 1), 1),
                                      pct_of_all=round(100 * len(b) / n, 1), mean_score=round(float(b["raw_adv"].mean()), 1) if len(b) else None,
                                      mean_score_pctile=round(float(d2.loc[b.index, "score_pct"].mean()), 3) if len(b) else None)
    print("\n" + "=" * 96 + "\nDEL 3 — GOOD trades that look BAD (terminal<=0 & MFE>=100 & optimal>=75)\n" + "=" * 96)
    print(f"  {OUT['del3_good_looks_bad']}")
    print("\nDEL 4 — BAD trades that look GOOD (terminal>0 & (MFE<20 or optimal≈terminal))")
    print(f"  {OUT['del4_bad_looks_good']}")
    print(f"  (overall mean score pctile = 0.500 by construction; GOOD-looks-BAD pctile {OUT['del3_good_looks_bad']['mean_score_pctile']}, "
          f"BAD-looks-GOOD pctile {OUT['del4_bad_looks_good']['mean_score_pctile']})")


def del5(d):
    """TARGET-SWAP: identical features, swap target, STRICT OOT (train pre-2026 -> test 2026)."""
    print("\n" + "=" * 96 + "\nDEL 5 — TARGET-SWAP (identical features, strict OOT pre-2026->2026)\n" + "=" * 96)
    feats = feat_cols(d)
    pre = d["ts"] < pd.Timestamp("2026-01-01", tz="UTC"); te = ~pre
    Xtr = d.loc[pre, feats].fillna(d[feats].median()); Xte = d.loc[te, feats].fillna(d[feats].median())
    OUT = {}
    print(f"  features={len(feats)}  n_train={int(pre.sum())}  n_test={int(te.sum())}")
    print(f"  {'target':34s} {'AUC(top-half)':>13} {'rankIC':>8} {'topDret':>9} {'topDsharpe':>11}")
    for lab, col in TARGETS.items():
        if col not in d: continue
        ytr_c = d.loc[pre, col].values; yte_c = d.loc[te, col].values
        mtr = ~np.isnan(ytr_c); mte = ~np.isnan(yte_c)
        if mtr.sum() < 500 or mte.sum() < 200: continue
        thr = np.nanmedian(ytr_c[mtr])
        ytr = (ytr_c[mtr] > thr).astype(int); yte = (yte_c[mte] > np.nanmedian(yte_c[mte])).astype(int)
        if len(np.unique(ytr)) < 2: continue
        clf = GradientBoostingClassifier(max_depth=3, n_estimators=150, subsample=0.7, random_state=0)
        clf.fit(Xtr.values[mtr], ytr)
        pr = clf.predict_proba(Xte.values[mte])[:, 1]
        auc = float(roc_auc_score(yte, pr))
        ic = float(spearmanr(pr, yte_c[mte]).correlation)               # rank-IC vs continuous target
        topmask = pr >= np.quantile(pr, 0.90)
        topret = float(np.nanmean(yte_c[mte][topmask]))
        topsh = float(np.nanmean(yte_c[mte][topmask]) / (np.nanstd(yte_c[mte][topmask]) + 1e-9))
        OUT[lab] = dict(auc=round(auc, 4), rank_ic=round(ic, 4), top_decile_return=round(topret, 1), top_decile_sharpe=round(topsh, 4))
        print(f"  {lab:34s} {auc:>13.4f} {ic:>+8.4f} {topret:>+9.1f} {topsh:>+11.4f}")
    return OUT


def main():
    d = build()
    d26 = d[d["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")].dropna(subset=["terminal_pnl_K96", "raw_adv"]).copy()
    OUT = {"meta": dict(n_takes_total=int(len(d)), n_2026=int(len(d26)),
                        note="entry-score=raw_adv; optimal_exit≈MFE@K96; current-exit/LWR need M1 path (not computed)")}
    print(f"TAKEs total={len(d)}  2026 OOT={len(d26)}")
    OUT["del1"] = del1(d26)
    OUT["del2"] = del2(d26)
    cohorts(d26, OUT)
    OUT["del5_target_swap"] = del5(d)
    with open("/tmp/label_target_test_20260623.json", "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print("\n→ /tmp/label_target_test_20260623.json")


if __name__ == "__main__":
    main()
