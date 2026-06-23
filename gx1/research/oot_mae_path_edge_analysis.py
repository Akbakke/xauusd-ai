"""FULL MAE/path-edge analysis (user vedtak 2026-06-23): is the biggest UNEXPLOITED edge predicting
MAE/path-quality rather than DIRECTION? 6 parts on the largest OOT base (forward_outcome_clean, K96).

Base = the ENTRY's inherent held-to-K96 path (pre-exit) for the model's chosen side on the TAKEN-proxy
population (top-quintile margin = conviction-gate selectivity, side=argmax direction). This isolates
ENTRY path quality before the Exit-IQL manages it — exactly the "can V10 flag a bad-path entry ex-ante?"
question. NOTE: head_survival / head_clean_edge are NOT exported to forward_outcome → use the available
path heads (bad_path_prob, path_quality_pred/std/log_var, mfe_first_n_pred, tradable_prob) + tail_risk/
timing/forecast/dip/direction heads. READ-ONLY; no cement touched. Direction-AUC reference ≈ 0.60-0.62.
"""
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

FO = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
K = "K96"
RNG = np.random.default_rng(20260623)

PATH_HEADS = ["bad_path_prob", "path_quality_pred", "path_quality_std", "path_quality_log_var",
              "mfe_first_n_pred", "tradable_prob"]
TAIL = [f"v10_tail_risk_{i}" for i in range(6)]
TIMING = [f"v10_timing_{i}" for i in range(12)]
FORECAST = [f"v10_forecast_{i}" for i in range(4)]
DIP = [f"v10_dip_{i}" for i in range(18)]
DIRECTION = ["direction_logit_long", "direction_logit_short", "direction_logit_flat", "p_long", "p_short",
             "margin", "uncertainty_score"]
ALL_HEADS = PATH_HEADS + TAIL + TIMING + FORECAST + DIP + DIRECTION


def load():
    files = sorted(glob.glob(f"{FO}/*.parquet"))
    av = set(pd.read_parquet(files[0]).columns)
    lab = []
    for side in ("long", "short"):
        for m in ("terminal_pnl", "mae_bps", "mfe_bps", "time_to_mfe", "mae_before_mfe_bps"):
            lab.append(f"take_now_{side}_{m}_at_{K}_v1")
    need = [c for c in (ALL_HEADS + lab + ["decision_ts_utc", "trend_regime",
            "d1_ema_slope_20_canon_v2_canon_v1", "_v1h4_ema_diff_canon_v1", "_v1h1_ema_diff_canon_v1"]) if c in av]
    df = pd.concat([pd.read_parquet(f, columns=need) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    # side = argmax direction; realized = that side's labels
    long = (df["p_long"] >= df["p_short"]).values
    df["side"] = np.where(long, "long", "short")
    for m in ("terminal_pnl", "mae_bps", "mfe_bps", "time_to_mfe"):
        lc = f"take_now_long_{m}_at_{K}_v1"; sc = f"take_now_short_{m}_at_{K}_v1"
        df[m] = np.where(long, df[lc].astype(float), df[sc].astype(float))
    df = df.dropna(subset=["terminal_pnl", "mae_bps", "mfe_bps"]).reset_index(drop=True)
    return df, [h for h in ALL_HEADS if h in df.columns]


def main():
    df, heads = load()
    # TAKEN-proxy: top-quintile margin (conviction-gate selectivity)
    thr = np.nanpercentile(df["margin"].astype(float).values, 80)
    taken = df[df["margin"].astype(float) >= thr].reset_index(drop=True)
    med = taken["ts"].median()
    print(f"loaded {len(df)} candidate bars; TAKEN-proxy (top-20% margin) = {len(taken)}; heads avail={len(heads)}")
    mae = taken["mae_bps"].astype(float).values
    mfe = taken["mfe_bps"].astype(float).values
    pnl = taken["terminal_pnl"].astype(float).values
    dur = taken["time_to_mfe"].astype(float).values

    # ── PART 1: MAE segmentation ──────────────────────────────────────────────
    print("\n" + "=" * 70 + "\nPART 1 — MAE SEGMENTATION (taken-proxy, side=argmax, held-to-K96)\n" + "=" * 70)
    bins = [("A: MAE<50", mae < 50), ("B: 50-150", (mae >= 50) & (mae < 150)),
            ("C: 150-300", (mae >= 150) & (mae < 300)), ("D: MAE>=300", mae >= 300)]
    print(f"  {'group':14s} {'n':>7s} {'%':>5s} {'win%':>6s} {'exp(mean)':>9s} {'medPnL':>7s} {'avgMFE':>7s} {'avgDur':>7s}")
    seg = {}
    for name, m in bins:
        if m.sum() == 0:
            continue
        seg[name] = dict(n=int(m.sum()), win=float((pnl[m] > 0).mean()), exp=float(pnl[m].mean()),
                         med=float(np.median(pnl[m])), mfe=float(mfe[m].mean()), dur=float(np.nanmean(dur[m])))
        print(f"  {name:14s} {m.sum():7d} {m.mean()*100:4.1f}% {(pnl[m]>0).mean()*100:5.1f}% "
              f"{pnl[m].mean():+9.2f} {np.median(pnl[m]):+7.1f} {mfe[m].mean():7.1f} {np.nanmean(dur[m]):7.1f}")
    badmask = mae >= 150
    print(f"  >> good paths A+B: n={int((~badmask).sum())} exp={pnl[~badmask].mean():+.2f}  |  "
          f"bad paths C+D: n={int(badmask.sum())} exp={pnl[badmask].mean():+.2f}  win={(pnl[badmask]>0).mean():.3f}")

    # ── PART 2: does V10 predict MAE? (corr + PI) ─────────────────────────────
    print("\n" + "=" * 70 + "\nPART 2 — DO V10 HEADS PREDICT MAE/MFE/DURATION? (corr + PI vs C+D)\n" + "=" * 70)
    rows = []
    for h in heads:
        v = taken[h].astype(float).values
        if np.nanstd(v) < 1e-9:
            continue
        c_mae = np.corrcoef(v, mae)[0, 1]
        c_mfe = np.corrcoef(v, mfe)[0, 1]
        c_dur = np.corrcoef(np.nan_to_num(v), np.nan_to_num(dur))[0, 1]
        rows.append((h, c_mae, c_mfe, c_dur))
    rows.sort(key=lambda x: -abs(x[1]))
    print(f"  {'head':28s} {'corr_MAE':>9s} {'corr_MFE':>9s} {'corr_DUR':>9s}")
    for h, cm, cf, cd in rows[:18]:
        print(f"  {h:28s} {cm:+9.3f} {cf:+9.3f} {cd:+9.3f}")
    # PI vs bad-path (C+D), OOT
    e = (taken["ts"] <= med).values; l = (taken["ts"] > med).values
    yb = (mae >= 150).astype(int)
    Xtr = taken.loc[e, heads].astype(np.float32).replace([np.inf, -np.inf], np.nan); ytr = yb[e]
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=250, learning_rate=0.05,
                                         min_samples_leaf=200, early_stopping=True, random_state=42).fit(Xtr, ytr)
    Xte = taken.loc[l, heads].astype(np.float32).replace([np.inf, -np.inf], np.nan); yte = yb[l]
    nte = min(len(Xte), 40000); s = RNG.choice(len(Xte), nte, replace=False)
    pi = permutation_importance(clf, Xte.iloc[s], yte[s], n_repeats=4, random_state=0, scoring="roc_auc", n_jobs=3)
    imp = sorted(zip(heads, pi.importances_mean), key=lambda x: -x[1])
    print("  TOP heads by OOT permutation-importance for predicting BAD-PATH (C+D):")
    for h, v in imp[:12]:
        print(f"    {h:28s} {v:+.5f}")

    # ── PART 3: separation A+B vs C+D ─────────────────────────────────────────
    print("\n" + "=" * 70 + "\nPART 3 — SEPARATION: good-path(A+B) vs bad-path(C+D)  [vs direction-AUC ~0.62]\n" + "=" * 70)

    def auc_pair(cols, tr, te):
        Xt = taken.loc[tr, cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        Xv = taken.loc[te, cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        yt, yv = yb[tr], yb[te]
        if len(np.unique(yt)) < 2 or len(np.unique(yv)) < 2:
            return None
        c = HistGradientBoostingClassifier(max_depth=4, max_iter=250, learning_rate=0.05,
                                           min_samples_leaf=200, early_stopping=True, random_state=42).fit(Xt, yt)
        p = c.predict_proba(Xv)[:, 1]
        return dict(auc=roc_auc_score(yv, p), pr=average_precision_score(yv, p),
                    ks=ks_2samp(p[yv == 1], p[yv == 0]).statistic, brier=brier_score_loss(yv, p), base=yv.mean())
    for name, cols in [("path_heads_only", PATH_HEADS), ("tail_risk", TAIL),
                       ("ALL V10 heads", heads), ("direction_heads", [c for c in DIRECTION if c in heads])]:
        cols = [c for c in cols if c in heads]
        r1 = auc_pair(cols, e, l); r2 = auc_pair(cols, l, e)
        if r1 and r2:
            print(f"  {name:18s} AUC={r1['auc']:.3f}/{r2['auc']:.3f}  PR={r1['pr']:.3f}/{r2['pr']:.3f}  "
                  f"KS={r1['ks']:.3f}/{r2['ks']:.3f}  brier={r1['brier']:.3f} (base={r1['base']:.3f})")
    # single best head
    singles = [(h, auc_pair([h], e, l)) for h in heads]
    singles = [(h, r["auc"]) for h, r in singles if r]
    singles.sort(key=lambda x: -x[1])
    print("  best SINGLE heads for bad-path AUC (fit e→l):", [(h, round(a, 3)) for h, a in singles[:6]])

    # ── PART 4: long in bearish regime ────────────────────────────────────────
    print("\n" + "=" * 70 + "\nPART 4 — LONG TRADES BY TREND REGIME (entry D1/H4/H1)\n" + "=" * 70)
    lo = df[df["side"] == "long"].reset_index(drop=True)
    lmae = lo["mae_bps"].astype(float).values; lmfe = lo["mfe_bps"].astype(float).values
    lpnl = lo["terminal_pnl"].astype(float).values
    d1 = np.sign(lo["d1_ema_slope_20_canon_v2_canon_v1"].astype(float).values)
    h4 = np.sign(lo["_v1h4_ema_diff_canon_v1"].astype(float).values)
    h1 = np.sign(lo["_v1h1_ema_diff_canon_v1"].astype(float).values)
    for tfn, tf in (("D1", d1), ("H4", h4), ("H1", h1)):
        print(f"  {tfn} trend at entry (long trades):")
        for lab, mk in (("UP", tf > 0), ("DOWN", tf < 0)):
            if mk.sum():
                print(f"     {lab:5s} n={mk.sum():6d}  exp={lpnl[mk].mean():+7.2f}  MAE={lmae[mk].mean():6.1f}  "
                      f"MFE={lmfe[mk].mean():6.1f}  win={(lpnl[mk]>0).mean():.3f}")
    # longs that END POSITIVE but had EXTREME MAE first
    epw = (lpnl > 0) & (lmae >= 150)
    print(f"  >> longs that WIN but with EXTREME MAE>=150 first: n={int(epw.sum())} ({epw.mean()*100:.1f}% of longs)  "
          f"mean exp={lpnl[epw].mean():+.1f} mean MAE={lmae[epw].mean():.0f} mean MFE={lmfe[epw].mean():.0f}")
    # can V10 heads flag them ex-ante? AUC of heads for (extreme-MAE) within winning longs
    win = lpnl > 0
    yext = (lmae[win] >= 150).astype(int)
    Hh = [h for h in heads if h in lo.columns]
    if len(np.unique(yext)) > 1:
        n = win.sum(); idx = np.where(win)[0]; spl = int(n * 0.5)
        Xw = lo.iloc[idx][Hh].astype(np.float32).replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
        c = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05,
                                           min_samples_leaf=200, early_stopping=True, random_state=42).fit(Xw.iloc[:spl], yext[:spl])
        a = roc_auc_score(yext[spl:], c.predict_proba(Xw.iloc[spl:])[:, 1])
        print(f"  >> can V10 heads predict 'winning-long-with-extreme-MAE' ex-ante? OOT AUC={a:.3f} (base={yext.mean():.3f})")

    # ── PART 5: counterfactual filter ─────────────────────────────────────────
    print("\n" + "=" * 70 + "\nPART 5 — COUNTERFACTUAL FILTER (remove worst-PATH trades, OOT-late)\n" + "=" * 70)
    oot = taken[(taken["ts"] > med)].reset_index(drop=True)
    o_pnl = oot["terminal_pnl"].astype(float).values
    o_ts = oot["ts"].values
    order = np.argsort(o_ts)
    def stats(keep):
        v = o_pnl[keep]
        if len(v) == 0:
            return None
        cum = np.cumsum(v[np.argsort(o_ts[keep])])
        mdd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
        wins = v[v > 0].sum(); losses = -v[v < 0].sum()
        return dict(n=len(v), total=float(v.sum()), mean=float(v.mean()),
                    sharpe=float(v.mean() / (v.std() + 1e-9)), pf=float(wins / (losses + 1e-9)), mdd=mdd)
    base = stats(np.ones(len(oot), bool))
    print(f"  BASELINE (no filter): n={base['n']} total={base['total']:.0f} mean={base['mean']:+.2f} "
          f"Sharpe={base['sharpe']:.3f} PF={base['pf']:.2f} maxDD={base['mdd']:.0f}")
    # rankers: higher score = WORSE path → drop the top X%
    bp = oot["bad_path_prob"].astype(float).values
    pq = -oot["path_quality_pred"].astype(float).values  # low quality = bad → negate so high=bad
    tr_score = oot[[c for c in TAIL if c in oot.columns]].astype(float).abs().mean(axis=1).values
    # combined V10 model: GBM(all heads on EARLY)->predict bad-path, score OOT
    yb_e = (taken["mae_bps"].astype(float).values[(taken["ts"] <= med).values] >= 150).astype(int)
    Xe = taken.loc[(taken["ts"] <= med).values, heads].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    cm = HistGradientBoostingClassifier(max_depth=4, max_iter=250, learning_rate=0.05,
                                        min_samples_leaf=200, early_stopping=True, random_state=42).fit(Xe, yb_e)
    combo = cm.predict_proba(oot[heads].astype(np.float32).replace([np.inf, -np.inf], np.nan))[:, 1]
    rankers = {"bad_path_prob": bp, "low_path_quality": pq, "tail_risk_mag": tr_score, "COMBINED_V10": combo}
    for rname, score in rankers.items():
        print(f"  filter by {rname} (drop worst-path X%):")
        for frac in (0.05, 0.10, 0.15, 0.20, 0.25):
            cut = np.nanpercentile(score, 100 * (1 - frac))
            keep = score < cut
            st = stats(keep)
            if st:
                print(f"     drop {int(frac*100):2d}%: n={st['n']:5d} total={st['total']:8.0f} mean={st['mean']:+6.2f} "
                      f"Sharpe={st['sharpe']:.3f} PF={st['pf']:.2f} maxDD={st['mdd']:7.0f}")

    # ── PART 6: decomposition ─────────────────────────────────────────────────
    print("\n" + "=" * 70 + "\nPART 6 — LOSS DECOMPOSITION: wrong-direction vs bad-path\n" + "=" * 70)
    losers = pnl < 0
    # wrong-direction = loser whose MFE never got meaningfully positive (never went our way)
    wrong_dir = losers & (mfe < 20)
    # right-direction-bad-path = loser that DID reach decent MFE (went our way) but ended negative
    bad_path_loss = losers & (mfe >= 20)
    tot_loss = -pnl[losers].sum()
    print(f"  total losers: n={int(losers.sum())} ({losers.mean()*100:.1f}%)  total loss={tot_loss:.0f} bps")
    print(f"  WRONG-DIRECTION (MFE<20, never went our way): n={int(wrong_dir.sum())} "
          f"loss={-pnl[wrong_dir].sum():.0f} = {(-pnl[wrong_dir].sum())/tot_loss*100:.1f}% of losses")
    print(f"  RIGHT-DIR-BAD-PATH (MFE>=20 then gave back/MAE): n={int(bad_path_loss.sum())} "
          f"loss={-pnl[bad_path_loss].sum():.0f} = {(-pnl[bad_path_loss].sum())/tot_loss*100:.1f}% of losses")
    print(f"  mean MAE: wrong-dir={mae[wrong_dir].mean():.0f}  bad-path-loss={mae[bad_path_loss].mean():.0f}")


if __name__ == "__main__":
    main()
