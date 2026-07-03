"""REGIME-QUALIFIER hypothesis (2026-06-23): is the edge regime-dependent — is there an unexploited
"WHEN to trade" signal above V10, orthogonal to direction? NOT about better long/short prediction.

Reuses the cached LIVE-POLICY universe (FOLD_1, conviction-gate -37.71 + DIPFIX) from
oot_live_policy_falsification_20260623.py and ENRICHES each TAKE with its regime features (read from
forward_outcome by candidate_uid) + MFE/time_to_mfe. NO new model, NO replay — pandas + a small GBM
for the OOT separability test only. Outcome = terminal_pnl@K96 held-to-horizon (entry-side label).

HONEST FEATURE COVERAGE: native (model's actual inputs) = trend_regime, vol_regime, atr_bps/atr_z/
bb_squeeze/vol_ratio, multi-TF slopes (h1/h4/d1/ema), session. NOT native (flagged): ADX, true
distance-to-D1/H4-swing-high/low, breakout — derived as ROLLING-window proxies from the close series
(labelled *_proxy), never fabricated as if exact.

Del 4 DISCIPLINE (AGENTS.md LABEL-REWARD PRE-GATE): the GOOD/BAD-environment separability MUST hold
STRICT OOT (fit pre-2026 → test 2026 AND reverse) and INCREMENTALLY over raw_adv+margin — else a regime
that only separates in-sample is a futile filter (the trough-label trap).
"""
from __future__ import annotations
import glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from gx1.research.oot_live_policy_falsification_20260623 import live_size, sharpe, wsharpe, wmean, FO, K, LONG, SHORT

UNIV = "/tmp/live_policy_universe_20260623.parquet"
REGIME_COLS = ["candidate_uid", "trend_regime", "vol_regime", "atr_bps", "atr_z_canon_v1",
               "_v1_atr_z_10_100_canon_v1", "_v1_bb_squeeze_20_2_canon_v1", "vol_ratio_canon_v1",
               "_v1_atr_regime_id_canon_v1", "_v1h1_slope5_canon_v1", "_v1h4_slope5_canon_v1",
               "d1_ema_slope_20_canon_v2_canon_v1", "ema100_slope_canon_v1", "m15_trend_sign_canon_v2_canon_v1",
               "hour_utc", "close"]
MFE = [f"take_now_{s}_{m}_at_{K}_v1" for s in ("long", "short") for m in ("mfe_bps", "time_to_mfe", "mae_before_mfe_bps")]


def enrich():
    u = pd.read_parquet(UNIV)
    u = u[u.is_take == 1].copy()
    u["ts"] = pd.to_datetime(u["ts"], utc=True)
    files = sorted(glob.glob(f"{FO}/*.parquet"))
    av = set(pd.read_parquet(files[0]).columns)
    rc = [c for c in REGIME_COLS + MFE if c in av]
    reg = pd.concat([pd.read_parquet(f, columns=rc) for f in files], ignore_index=True)
    reg = reg.drop_duplicates("candidate_uid")
    d = u.merge(reg, left_on="uid", right_on="candidate_uid", how="left")
    # final-side MFE / time-to-mfe / mae_before_mfe
    isL = d["side"] == "long"
    for m in ("mfe_bps", "time_to_mfe", "mae_before_mfe_bps"):
        cl, cs = f"take_now_long_{m}_at_{K}_v1", f"take_now_short_{m}_at_{K}_v1"
        if cl in d and cs in d:
            d[m] = np.where(isL, d[cl], d[cs])
    # ── derived regime axes ──
    d["atr_pctl"] = d["atr_bps"].rank(pct=True)
    d["trend_str"] = d[["_v1h1_slope5_canon_v1", "_v1h4_slope5_canon_v1", "d1_ema_slope_20_canon_v2_canon_v1"]].abs().mean(axis=1)
    # multi-TF trend alignment (persistence proxy): sign agreement of h1/h4/d1 slopes
    sgn = np.sign(d[["_v1h1_slope5_canon_v1", "_v1h4_slope5_canon_v1", "d1_ema_slope_20_canon_v2_canon_v1"]].fillna(0).values)
    d["trend_align"] = np.abs(sgn.sum(axis=1))  # 3 = all agree, 1 = split
    # structure/breakout PROXIES from the close series (NOT true D1 swings)
    d = d.sort_values("ts").reset_index(drop=True)
    roll_hi = d["close"].rolling(96, min_periods=20).max()
    roll_lo = d["close"].rolling(96, min_periods=20).min()
    rng = (roll_hi - roll_lo).replace(0, np.nan)
    d["dist_hi96_proxy"] = (roll_hi - d["close"]) / rng        # 0=at high, 1=at low
    d["dist_lo96_proxy"] = (d["close"] - roll_lo) / rng
    d["range_pos_proxy"] = d["dist_lo96_proxy"]                 # 0=low,1=high of 96-bar range
    d["breakout96_proxy"] = (d["close"] >= roll_hi.shift(1)).astype(int) - (d["close"] <= roll_lo.shift(1)).astype(int)
    # market type (data-driven from compression/expansion/trend)
    sq = d["_v1_bb_squeeze_20_2_canon_v1"]; az = d["atr_z_canon_v1"]
    sq_lo = sq < sq.quantile(0.33); az_hi = az > az.quantile(0.67)
    mt = np.where(az_hi, "EXPANSION", np.where(sq_lo, "SQUEEZE",
         np.where((d["trend_align"] == 3) & (d["trend_str"] > d["trend_str"].median()), "TREND",
         np.where(d["trend_align"] <= 1, "RANGE", "TRANSITION"))))
    d["market_type"] = mt
    print(f"[enrich] TAKEs={len(d)}  regime-cols joined={len([c for c in rc if c!='candidate_uid'])}  "
          f"pnl/mfe non-null={d['pnl'].notna().mean():.3f}/{d.get('mfe_bps', pd.Series([np.nan])).notna().mean():.3f}")
    return d


def bstats(df):
    p, w = df["pnl"].values, df["sz"].values
    if len(df) < 30:
        return dict(n=len(df), note="n<30")
    return dict(n=int(len(df)), win=round(float((p > 0).mean()), 3), exp=round(float(p.mean()), 1),
                sharpe=round(sharpe(p), 3), mae=round(float(df["mae"].mean()), 1),
                mfe=round(float(df["mfe_bps"].mean()), 1) if "mfe_bps" in df else None,
                t2mfe=round(float(df["time_to_mfe"].mean()), 1) if "time_to_mfe" in df else None,
                wexp=round(wmean(p, w), 1), wsharpe=round(wsharpe(p, w), 3))


def qbucket(d, col, n=4):
    try:
        return pd.qcut(d[col].rank(method="first"), n, labels=[f"Q{i+1}" for i in range(n)])
    except Exception:
        return pd.Series(["NA"] * len(d), index=d.index)


def seg(d, by, OUT, key, label):
    print(f"\n-- {label} --")
    res = {}
    groups = d.groupby(by, observed=True) if d[by].dtype == object or str(d[by].dtype).startswith("category") else d.assign(_b=qbucket(d, by)).groupby("_b", observed=True)
    gcol = by if (d[by].dtype == object or str(d[by].dtype).startswith("category")) else "_b"
    src = d if gcol == by else d.assign(_b=qbucket(d, by))
    for g, sub in src.groupby(gcol, observed=True):
        s = bstats(sub); res[str(g)] = s
        if "win" in s:
            print(f"   {str(g):14s} n={s['n']:6d} win={s['win']:.3f} exp={s['exp']:+7.1f} sharpe={s['sharpe']:+.3f} "
                  f"mae={s['mae']:6.1f} mfe={s['mfe']:6.1f} | wexp={s['wexp']:+7.1f} wsharpe={s['wsharpe']:+.3f}")
    OUT.setdefault("del1_segments", {})[key] = res
    return res


NUMF = ["atr_pctl", "atr_z_canon_v1", "_v1_atr_z_10_100_canon_v1", "_v1_bb_squeeze_20_2_canon_v1",
        "vol_ratio_canon_v1", "trend_str", "trend_align", "_v1h1_slope5_canon_v1", "_v1h4_slope5_canon_v1",
        "d1_ema_slope_20_canon_v2_canon_v1", "ema100_slope_canon_v1", "hour_utc", "range_pos_proxy",
        "dist_hi96_proxy", "breakout96_proxy"]


def oot_separability(d, OUT):
    """STRICT OOT: can regime features predict trade WIN, and incrementally over raw_adv+margin?"""
    d = d.dropna(subset=["pnl"]).copy()
    d["y"] = (d["pnl"] > 0).astype(int)
    feats = [c for c in NUMF if c in d.columns]
    X = d[feats].fillna(d[feats].median())
    base = d[["raw_adv", "margin"]].fillna(0.0)
    pre = d["ts"] < pd.Timestamp("2026-01-01", tz="UTC")
    arms = {}
    for name, tr, te in [("fit_pre2026_test_2026", pre, ~pre), ("fit_2026_test_pre", ~pre, pre)]:
        if tr.sum() < 500 or te.sum() < 200:
            arms[name] = {"note": "insufficient"}; continue
        def auc(cols):
            g = GradientBoostingClassifier(max_depth=3, n_estimators=120, subsample=0.7, random_state=0)
            g.fit(cols[tr.values], d["y"].values[tr.values])
            pr = g.predict_proba(cols[te.values])[:, 1]
            yt = d["y"].values[te.values]
            return round(float(roc_auc_score(yt, pr)), 4), round(float(average_precision_score(yt, pr)), 4), g
        a_reg, pr_reg, g = auc(X.values)
        a_base, pr_base, _ = auc(base.values)
        a_both, pr_both, _ = auc(np.hstack([X.values, base.values]))
        imp = sorted(zip(feats, g.feature_importances_), key=lambda t: -t[1])[:8]
        arms[name] = dict(base_rate=round(float(d["y"].values[te.values].mean()), 3),
                          auc_regime=a_reg, prauc_regime=pr_reg, auc_rawadv_margin=a_base,
                          auc_combined=a_both, incremental_auc=round(a_both - a_base, 4),
                          top_features=[(f, round(float(i), 3)) for f, i in imp])
        print(f"\n  [{name}] base_rate={arms[name]['base_rate']}  AUC: regime={a_reg} | raw_adv+margin={a_base} | "
              f"combined={a_both} (incr {arms[name]['incremental_auc']:+.4f})  PR-AUC regime={pr_reg}")
        print("     top regime feats:", [f"{f}:{i:.2f}" for f, i in imp[:6]])
    OUT["del4_oot_separability"] = arms


def bucket_persistence(d, OUT):
    """Is the regime EXPECTANCY ORDERING stable OOT? rank buckets by exp on pre-2026, corr with 2026."""
    d = d.dropna(subset=["pnl"]).copy()
    pre = d["ts"] < pd.Timestamp("2026-01-01", tz="UTC")
    out = {}
    for axis in ["market_type", "trend_regime", "vol_regime", "session"]:
        if axis not in d: continue
        ep = d[pre].groupby(axis, observed=True)["pnl"].mean()
        eo = d[~pre].groupby(axis, observed=True)["pnl"].mean()
        common = ep.index.intersection(eo.index)
        if len(common) >= 3:
            rho = float(spearmanr(ep[common], eo[common]).correlation)
            out[axis] = dict(rank_corr_pre_vs_2026=round(rho, 3),
                             pre={str(k): round(float(v), 1) for k, v in ep[common].items()},
                             y2026={str(k): round(float(v), 1) for k, v in eo[common].items()})
    OUT["del4_bucket_persistence"] = out
    print("\n-- regime expectancy PERSISTENCE (rank-corr pre-2026 vs 2026; >0 = stable ordering) --")
    for a, v in out.items():
        print(f"   {a:14s} rank_corr={v['rank_corr_pre_vs_2026']:+.3f}")


def main():
    d = enrich()
    d.to_parquet("/tmp/regime_universe_20260623.parquet", index=False)
    OUT = {"meta": dict(n_takes=int(len(d)), outcome="terminal_pnl@K96 held-to-horizon",
                        global_win=round(float((d["pnl"] > 0).mean()), 3), global_exp=round(float(d["pnl"].mean()), 1))}
    print("\n" + "=" * 90 + "\nDEL 1 — REGIME SEGMENTATION (live-policy TAKEs)\n" + "=" * 90)
    print(f"GLOBAL: n={len(d)} win={OUT['meta']['global_win']} exp={OUT['meta']['global_exp']:+.1f}")
    for by, lab, key in [("trend_regime", "TREND regime (native)", "trend_regime"),
                         ("vol_regime", "VOL regime (native)", "vol_regime"),
                         ("market_type", "MARKET TYPE (derived)", "market_type"),
                         ("session", "SESSION", "session"),
                         ("atr_pctl", "ATR percentile (quartile)", "atr_pctl"),
                         ("trend_str", "TREND strength (quartile)", "trend_str"),
                         ("trend_align", "TREND alignment h1/h4/d1", "trend_align"),
                         ("_v1_bb_squeeze_20_2_canon_v1", "BB squeeze / compression (quartile)", "bb_squeeze"),
                         ("atr_z_canon_v1", "ATR-z / expansion (quartile)", "atr_z"),
                         ("range_pos_proxy", "RANGE position proxy (quartile)", "range_pos_proxy"),
                         ("breakout96_proxy", "BREAKOUT-96 proxy", "breakout96_proxy")]:
        if by in d.columns:
            seg(d, by, OUT, key, lab)

    # ── DEL 2: V10 calibration per market_type ──
    print("\n" + "=" * 90 + "\nDEL 2 — V10 CALIBRATION per market_type (mean p_chosen vs realized win; margin vs win)\n" + "=" * 90)
    d["p_chosen"] = np.where(d["side"] == "long", d["p_long"], d["p_short"])
    cal = {}
    for g, sub in d.dropna(subset=["pnl"]).groupby("market_type", observed=True):
        cal[str(g)] = dict(n=int(len(sub)), mean_p_chosen=round(float(sub["p_chosen"].mean()), 3),
                           realized_win=round(float((sub["pnl"] > 0).mean()), 3),
                           calib_gap=round(float(sub["p_chosen"].mean() - (sub["pnl"] > 0).mean()), 3),
                           mean_margin=round(float(sub["margin"].mean()), 3),
                           corr_margin_win=round(float(np.corrcoef(sub["margin"], (sub["pnl"] > 0))[0, 1]), 3) if len(sub) > 50 else None,
                           mae=round(float(sub["mae"].mean()), 1), mfe=round(float(sub["mfe_bps"].mean()), 1))
        print(f"   {str(g):12s} n={cal[str(g)]['n']:6d}  p_chosen={cal[str(g)]['mean_p_chosen']:.3f} "
              f"win={cal[str(g)]['realized_win']:.3f} gap={cal[str(g)]['calib_gap']:+.3f}  "
              f"margin={cal[str(g)]['mean_margin']:.3f} corr(margin,win)={cal[str(g)]['corr_margin_win']}  "
              f"mae={cal[str(g)]['mae']:.1f} mfe={cal[str(g)]['mfe']:.1f}")
    OUT["del2_calibration"] = cal

    # ── DEL 3: "visually wrong" cohort (MAE>100 then recovers to +PnL) vs clean winners ──
    print("\n" + "=" * 90 + "\nDEL 3 — 'VISUALLY WRONG' cohort (MAE>100 & pnl>0) vs CLEAN winners (MAE<30 & pnl>0)\n" + "=" * 90)
    dd = d.dropna(subset=["pnl", "mae"])
    wrong = dd[(dd["mae"] > 100) & (dd["pnl"] > 0)]
    clean = dd[(dd["mae"] < 30) & (dd["pnl"] > 0)]
    def prof(c):
        return dict(n=int(len(c)), margin=round(float(c["margin"].mean()), 3), raw_adv=round(float(c["raw_adv"].mean()), 1),
                    atr_pctl=round(float(c["atr_pctl"].mean()), 3), trend_align=round(float(c["trend_align"].mean()), 2),
                    bb_squeeze=round(float(c["_v1_bb_squeeze_20_2_canon_v1"].mean()), 4),
                    atr_z=round(float(c["atr_z_canon_v1"].mean()), 3), mfe=round(float(c["mfe_bps"].mean()), 1),
                    pnl=round(float(c["pnl"].mean()), 1),
                    mkt=c["market_type"].value_counts(normalize=True).round(3).to_dict(),
                    sess=c["session"].value_counts(normalize=True).round(3).to_dict())
    OUT["del3"] = dict(visually_wrong=prof(wrong), clean_winners=prof(clean))
    for nm, c in [("VISUALLY-WRONG (MAE>100,+)", wrong), ("CLEAN (MAE<30,+)", clean)]:
        p = prof(c)
        print(f"   {nm:26s} n={p['n']:5d} margin={p['margin']:.3f} raw_adv={p['raw_adv']:+.1f} atr_pctl={p['atr_pctl']:.2f} "
              f"trend_align={p['trend_align']:.2f} atr_z={p['atr_z']:+.2f} mfe={p['mfe']:.0f}")
        print(f"        market: {p['mkt']}  session: {p['sess']}")

    # ── DEL 4: OOT separability + persistence ──
    print("\n" + "=" * 90 + "\nDEL 4 — META-REGIME: OOT separability (can features say WHEN to trade?)\n" + "=" * 90)
    oot_separability(d, OUT)
    bucket_persistence(d, OUT)

    with open("/tmp/regime_qualifier_20260623.json", "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print("\n→ /tmp/regime_qualifier_20260623.json + /tmp/regime_universe_20260623.parquet")


if __name__ == "__main__":
    main()
