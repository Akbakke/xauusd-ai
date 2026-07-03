"""FALSIFY "raw_adv overconviction collapse" — does the model's MOST EXTREME score = its WEAKEST area?

User falsification task 2026-06-24. We want MOTBEVIS, not confirmation. Observation: raw_adv deciles
show an inverted-U on terminal AND MFE in the 2026 OOT TAKE cohort (D7 67.6 / D8 76.1 / D9 49.8 /
D10 29.7; MFE D7 154 / D8 148 / D9 113 / D10 93). The highest-score trades are NOT the best trades.

REUSE (rule 7, no rebuild — the heavy real-inference build is done):
  • PRIMARY cohort = /tmp/regime_universe_20260623.parquet (61822 conviction-gated LIVE TAKEs;
    action-side outcomes pnl=terminal@K96, mfe_bps=MFE@K96, mae=MAE@K96, + trend_regime/vol_regime/
    market_type + raw_adv). This reproduces the user's exact decile numbers (parity-checked 2026-06-24).
  • FULL-RANGE cross-check = /tmp/live_policy_universe_20260623.parquet (147491 all candidates;
    raw_adv vs pnl_argmax) — to test whether the top-of-takes inverted-U is real degradation or an
    artifact of the conviction-gate's range-restriction (the margin-floor lesson: the gate already
    excises raw_adv's negative-slope region).

raw_adv = max(q_long,q_short) − q_skip  (the LIVE conviction-gate metric; entry-IQL Q's, FOLD_1).
Outcome = forward terminal_pnl@K96 HELD-TO-HORIZON (the cheap ENTRY-SIDE label, NOT the exit-stack).
So a degradation that shows on terminal AND MFE = ENTRY; terminal-only (MFE holds) = EXIT/path.

Splits (Del 10): 2020-22 = FOLD_1 in-sample (chronological walk-forward per contract/memory);
2023-25 = OOT pre-2026; 2026 = STRICT OOT (gold mega-uptrend; OOT for ALL folds — the decisive test).

NO retrain, NO new model, NO new feature, NO deploy. Diagnostics only.
Run: .venv/bin/python -m gx1.research.oot_raw_adv_inverted_u_20260624
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

PRIMARY = "/tmp/regime_universe_20260623.parquet"
FULL = "/tmp/live_policy_universe_20260623.parquet"
OUT_JSON = "/tmp/raw_adv_inverted_u_20260624.json"
SCORE = "raw_adv"
RNG = np.random.default_rng(20260624)

SPLITS = {
    "ALL": lambda ts: ts == ts,
    "2020-22 in-sample": lambda ts: ts < pd.Timestamp("2023-01-01", tz="UTC"),
    "2023-25 OOT-pre": lambda ts: (ts >= pd.Timestamp("2023-01-01", tz="UTC")) & (ts < pd.Timestamp("2026-01-01", tz="UTC")),
    "2026 strict-OOT": lambda ts: ts >= pd.Timestamp("2026-01-01", tz="UTC"),
}


def sharpe(p):
    p = np.asarray(p, float)
    return float(p.mean() / (p.std() + 1e-9)) if len(p) else float("nan")


def bucket_stats(g, out="pnl"):
    p = g[out].values.astype(float)
    return dict(n=int(len(p)), mean=round(float(p.mean()), 2), median=round(float(np.median(p)), 2),
                win=round(float((p > 0).mean()), 4), sharpe=round(sharpe(p), 4),
                mfe=round(float(g["mfe_bps"].mean()), 1), mae=round(float(g["mae"].mean()), 1))


# ── DEL 1: 100 percentiles + 20 ventiles ───────────────────────────────────
def del1_percentiles(d, nbins, out="pnl"):
    d = d.dropna(subset=[out, SCORE, "mfe_bps", "mae"]).copy()
    if len(d) < nbins * 5:
        return {"note": f"n={len(d)}<{nbins*5}"}
    d["b"] = pd.qcut(d[SCORE].rank(method="first"), nbins, labels=False) + 1
    rows = []
    for b in range(1, nbins + 1):
        g = d[d["b"] == b]
        st = bucket_stats(g, out)
        st["bucket"] = b
        st["score_lo"] = round(float(g[SCORE].min()), 2)
        st["score_hi"] = round(float(g[SCORE].max()), 2)
        rows.append(st)
    df = pd.DataFrame(rows)
    return dict(
        rows=rows,
        argmax_mean_bucket=int(df.loc[df["mean"].idxmax(), "bucket"]),
        argmax_median_bucket=int(df.loc[df["median"].idxmax(), "bucket"]),
        argmax_sharpe_bucket=int(df.loc[df["sharpe"].idxmax(), "bucket"]),
        argmax_win_bucket=int(df.loc[df["win"].idxmax(), "bucket"]),
        top_bucket_mean=float(df.iloc[-1]["mean"]), peak_mean=float(df["mean"].max()),
        top_vs_peak_mean=round(float(df.iloc[-1]["mean"] - df["mean"].max()), 2),
    )


# ── DEL 2: nested top-tail ──────────────────────────────────────────────────
def del2_top_tail(d, out="pnl"):
    d = d.dropna(subset=[out, SCORE, "mfe_bps", "mae"]).copy()
    res = {}
    for top in [50, 25, 10, 5, 2, 1, 0.5]:
        thr = np.percentile(d[SCORE], 100 - top)
        g = d[d[SCORE] >= thr]
        st = bucket_stats(g, out)
        st["score_floor"] = round(float(thr), 2)
        res[f"top{top}%"] = st
    keys = [f"top{t}%" for t in [50, 25, 10, 5, 2, 1, 0.5]]
    means = [res[k]["mean"] for k in keys]
    sharpes = [res[k]["sharpe"] for k in keys]
    res["_monotone_mean_rising_toward_extreme"] = bool(all(means[i + 1] >= means[i] for i in range(len(means) - 1)))
    res["_monotone_sharpe_rising_toward_extreme"] = bool(all(sharpes[i + 1] >= sharpes[i] for i in range(len(sharpes) - 1)))
    res["_signal_worse_at_extreme(top0.5_vs_top10)_mean"] = round(res["top0.5%"]["mean"] - res["top10%"]["mean"], 2)
    res["_signal_worse_at_extreme(top0.5_vs_top10)_sharpe"] = round(res["top0.5%"]["sharpe"] - res["top10%"]["sharpe"], 4)
    return res


# ── DEL 3: score → realized (monotonicity, isotonic, platt) ─────────────────
def del3_calibration(d, out="pnl", nbins=30):
    d = d.dropna(subset=[out, SCORE]).copy()
    x = d[SCORE].values.astype(float)
    y = d[out].values.astype(float)
    yw = (y > 0).astype(int)
    sp = spearmanr(x, y).correlation
    # isotonic (forced non-decreasing) — compare its top-knot prediction to the ACTUAL top-bucket mean
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, y)
    # platt (logistic on win)
    plat = LogisticRegression(max_iter=1000).fit(x.reshape(-1, 1), yw)
    # find score-percentile where rolling-mean outcome peaks (smoothed over quantile bins)
    d["b"] = pd.qcut(d[SCORE].rank(method="first"), nbins, labels=False)
    gb = d.groupby("b").agg(score=(SCORE, "mean"), realized=(out, "mean"), n=(out, "size"))
    peak_b = int(gb["realized"].idxmax())
    # "point where more score stops helping" = the score at the peak bin (beyond it, realized declines)
    knots = [10, 50, 90, 95, 99]
    iso_pred = {f"p{k}": round(float(iso.predict([np.percentile(x, k)])[0]), 2) for k in knots}
    return dict(
        n=int(len(d)), spearman_score_vs_terminal=round(float(sp), 4),
        platt_slope=round(float(plat.coef_[0][0]), 6),
        platt_win_at_p50=round(float(plat.predict_proba([[np.percentile(x, 50)]])[0][1]), 4),
        platt_win_at_p99=round(float(plat.predict_proba([[np.percentile(x, 99)]])[0][1]), 4),
        isotonic_pred_at_knots=iso_pred,
        peak_realized_bin=peak_b, n_bins=nbins,
        peak_score_value=round(float(gb.loc[peak_b, "score"]), 2),
        peak_score_percentile=round(100.0 * (peak_b + 0.5) / nbins, 1),
        realized_at_peak=round(float(gb.loc[peak_b, "realized"]), 2),
        realized_at_top_bin=round(float(gb["realized"].iloc[-1]), 2),
        decline_top_vs_peak=round(float(gb["realized"].iloc[-1] - gb["realized"].max()), 2),
        overconfident_right_tail=bool(gb["realized"].iloc[-1] < gb["realized"].max() - 1e-9),
    )


# ── DEL 9/10: EXTREME (P99-100) vs HIGH (P90-99) vs NORMAL (P60-90) + bootstrap ─
def groups_by_pct(d, score=SCORE):
    x = d[score].values.astype(float)
    p60, p90, p99 = np.percentile(x, [60, 90, 99])
    return dict(
        NORMAL_p60_90=d[(x >= p60) & (x < p90)],
        HIGH_p90_99=d[(x >= p90) & (x < p99)],
        EXTREME_p99_100=d[x >= p99],
    )


def boot_diff_iid(a, b, nboot=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or len(b) < 5:
        return None
    diffs = np.empty(nboot)
    for i in range(nboot):
        diffs[i] = RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return dict(point=round(float(a.mean() - b.mean()), 2), ci_lo=round(float(lo), 2),
                ci_hi=round(float(hi), 2), p_value=round(float(p), 4),
                ci_excludes_zero=bool(lo > 0 or hi < 0))


def boot_diff_block(d, out="pnl", nboot=10000):
    """Block bootstrap by calendar DAY (autocorrelation-aware): resample days, recompute the P90/P99
    cutoffs WITHIN each resample, recompute EXTREME−HIGH mean-diff. The conservative, honest test —
    the extreme-score trades cluster in a few episodes, so the i.i.d. CI understates uncertainty."""
    d = d.dropna(subset=[out, SCORE]).copy()
    d["day"] = pd.to_datetime(d["ts"], utc=True).dt.floor("D")
    days = d["day"].unique()
    by_day = {dd: d[d["day"] == dd][[SCORE, out]].values for dd in days}
    base_ext = d[d[SCORE] >= np.percentile(d[SCORE], 99)][out].mean()
    base_high = d[(d[SCORE] >= np.percentile(d[SCORE], 90)) & (d[SCORE] < np.percentile(d[SCORE], 99))][out].mean()
    diffs = []
    for _ in range(nboot):
        pick = RNG.choice(len(days), len(days))
        arr = np.concatenate([by_day[days[k]] for k in pick])
        sc, yy = arr[:, 0], arr[:, 1]
        p90, p99 = np.percentile(sc, [90, 99])
        ext = yy[sc >= p99]
        high = yy[(sc >= p90) & (sc < p99)]
        if len(ext) >= 3 and len(high) >= 3:
            diffs.append(ext.mean() - high.mean())
    if len(diffs) < 100:
        return None
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return dict(point=round(float(base_ext - base_high), 2), ci_lo=round(float(lo), 2),
                ci_hi=round(float(hi), 2), p_value=round(float(p), 4),
                ci_excludes_zero=bool(lo > 0 or hi < 0), n_days=int(len(days)))


def del9(d):
    grp = groups_by_pct(d)
    summary = {k: bucket_stats(v) for k, v in grp.items()}
    ext_t = grp["EXTREME_p99_100"]["pnl"].dropna().values
    high_t = grp["HIGH_p90_99"]["pnl"].dropna().values
    ext_m = grp["EXTREME_p99_100"]["mfe_bps"].dropna().values
    high_m = grp["HIGH_p90_99"]["mfe_bps"].dropna().values
    return dict(
        groups=summary,
        EXTREME_minus_HIGH_terminal_iid=boot_diff_iid(ext_t, high_t),
        EXTREME_minus_HIGH_terminal_block=boot_diff_block(d, "pnl"),
        EXTREME_minus_HIGH_MFE_iid=boot_diff_iid(ext_m, high_m),
        extreme_worse_than_high_terminal=bool(np.mean(ext_t) < np.mean(high_t)) if len(ext_t) and len(high_t) else None,
        extreme_worse_than_high_mfe=bool(np.mean(ext_m) < np.mean(high_m)) if len(ext_m) and len(high_m) else None,
    )


# ── DEL 8: feature attribution HIGH vs EXTREME (Cohen's d) ───────────────────
FEATS = ["trend_str", "trend_align", "atr_pctl", "atr_bps", "atr_z_canon_v1", "vol_ratio_canon_v1",
         "ema100_slope_canon_v1", "d1_ema_slope_20_canon_v2_canon_v1", "_v1h1_slope5_canon_v1",
         "_v1h4_slope5_canon_v1", "m15_trend_sign_canon_v2_canon_v1", "_v1_bb_squeeze_20_2_canon_v1",
         "dist_hi96_proxy", "dist_lo96_proxy", "range_pos_proxy", "breakout96_proxy",
         "p_long", "p_short", "margin", "dgap"]


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return None
    ps = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (ps + 1e-12))


def del8(d):
    grp = groups_by_pct(d)
    ext, high = grp["EXTREME_p99_100"], grp["HIGH_p90_99"]
    out = []
    for f in FEATS:
        if f not in d.columns:
            continue
        dd = cohens_d(ext[f].values, high[f].values)
        if dd is None:
            continue
        out.append(dict(feature=f, d_extreme_vs_high=round(dd, 3),
                        extreme_mean=round(float(np.nanmean(ext[f].values)), 4),
                        high_mean=round(float(np.nanmean(high[f].values)), 4)))
    out.sort(key=lambda r: -abs(r["d_extreme_vs_high"]))
    return out


def fmt_pct_table(rep):
    if "rows" not in rep:
        return "  " + str(rep)
    lines = []
    for r in rep["rows"]:
        lines.append(f"  B{r['bucket']:3d} [{r['score_lo']:+8.2f},{r['score_hi']:+8.2f}] n={r['n']:4d} "
                     f"mean={r['mean']:+7.2f} med={r['median']:+7.2f} win={r['win']:.3f} "
                     f"sh={r['sharpe']:+.3f} MFE={r['mfe']:6.1f} MAE={r['mae']:6.1f}")
    return "\n".join(lines)


def main():
    prim = pd.read_parquet(PRIMARY)
    prim["ts"] = pd.to_datetime(prim["ts"], utc=True)
    full = pd.read_parquet(FULL)
    full["ts"] = pd.to_datetime(full["ts"], utc=True)
    # full-range cohort outcome = argmax-side terminal; give it pnl/mfe/mae-compatible cols for reuse
    full = full.rename(columns={"pnl_argmax": "pnl_full"})
    OUT = {"meta": dict(primary=PRIMARY, n_takes=int(len(prim)), full=FULL, n_candidates=int(len(full)),
                        score=SCORE, outcome="terminal_pnl@K96 held-to-horizon (entry-side label)",
                        cohort="conviction-gated LIVE TAKE universe (FOLD_1, thr -37.71, DIPFIX both)")}

    for sname, fn in SPLITS.items():
        d = prim[fn(prim["ts"])].copy()
        if len(d) < 500:
            continue
        print("\n" + "=" * 90)
        print(f"SPLIT: {sname}   takes={len(d)}   (long={int((d.side=='long').sum())} short={int((d.side=='short').sum())})")
        print("=" * 90)
        S = OUT.setdefault(sname, {})

        # DEL 1 — 20 ventiles (printed) + 100 percentiles (json, argmax summary printed)
        v20 = del1_percentiles(d, 20, "pnl")
        p100 = del1_percentiles(d, 100, "pnl")
        S["del1_ventiles20"] = v20
        S["del1_percentiles100"] = {k: v for k, v in p100.items() if k != "rows"}
        S["del1_percentiles100_rows"] = p100.get("rows")
        print("\n[DEL1] raw_adv VENTILES (20) — terminal@K96:")
        print(fmt_pct_table(v20))
        print(f"  → ventile: argmax(mean)=B{v20.get('argmax_mean_bucket')} argmax(median)=B{v20.get('argmax_median_bucket')} "
              f"argmax(Sharpe)=B{v20.get('argmax_sharpe_bucket')} argmax(win)=B{v20.get('argmax_win_bucket')}  "
              f"top_vs_peak_mean={v20.get('top_vs_peak_mean')}")
        print(f"  → percentile(100): argmax(mean)=P{p100.get('argmax_mean_bucket')} argmax(Sharpe)=P{p100.get('argmax_sharpe_bucket')} "
              f"top_vs_peak_mean={p100.get('top_vs_peak_mean')}  (Q4: D10 decile-artifact? peak%≈{p100.get('argmax_mean_bucket')})")

        # DEL 2 — nested top-tail
        S["del2_top_tail"] = del2_top_tail(d, "pnl")
        print("\n[DEL2] nested top-tail (by raw_adv) — terminal@K96:")
        for k in [f"top{t}%" for t in [50, 25, 10, 5, 2, 1, 0.5]]:
            r = S["del2_top_tail"][k]
            print(f"  {k:7s} floor={r['score_floor']:+8.2f} n={r['n']:5d} exp={r['mean']:+7.2f} sh={r['sharpe']:+.3f} "
                  f"MFE={r['mfe']:6.1f} MAE={r['mae']:6.1f} win={r['win']:.3f}")
        print(f"  → quality rises monotonically toward extreme? mean={S['del2_top_tail']['_monotone_mean_rising_toward_extreme']} "
              f"sharpe={S['del2_top_tail']['_monotone_sharpe_rising_toward_extreme']}")
        print(f"  → SIGNAL WORSE AT EXTREME (top0.5% − top10%): mean={S['del2_top_tail']['_signal_worse_at_extreme(top0.5_vs_top10)_mean']} "
              f"sharpe={S['del2_top_tail']['_signal_worse_at_extreme(top0.5_vs_top10)_sharpe']}")

        # DEL 3 — calibration terminal + DEL 4 — MFE
        S["del3_calibration_terminal"] = del3_calibration(d, "pnl")
        S["del4_calibration_mfe"] = del3_calibration(d, "mfe_bps")
        c3, c4 = S["del3_calibration_terminal"], S["del4_calibration_mfe"]
        print(f"\n[DEL3] score→TERMINAL: spearman={c3['spearman_score_vs_terminal']:+.4f} platt_slope={c3['platt_slope']:+.5f} "
              f"win@p50={c3['platt_win_at_p50']:.3f}→win@p99={c3['platt_win_at_p99']:.3f}")
        print(f"        peak realized at score-pctl≈{c3['peak_score_percentile']}% (val {c3['peak_score_value']:+.1f}); "
              f"top-bin {c3['realized_at_top_bin']:+.1f} vs peak {c3['realized_at_peak']:+.1f} (decline {c3['decline_top_vs_peak']:+.1f}); "
              f"overconfident_right_tail={c3['overconfident_right_tail']}")
        print(f"[DEL4] score→MFE:      spearman={c4['spearman_score_vs_terminal']:+.4f} "
              f"peak MFE at score-pctl≈{c4['peak_score_percentile']}%; top-bin {c4['realized_at_top_bin']:.1f} vs peak {c4['realized_at_peak']:.1f} "
              f"(decline {c4['decline_top_vs_peak']:+.1f})  → MFE-collapse-at-top={c4['overconfident_right_tail']}")
        print("        ATTRIBUTION: terminal↓ AND MFE↓ ⇒ ENTRY problem; terminal↓ & MFE flat ⇒ EXIT/path problem.")

        # DEL 9 — EXTREME vs HIGH vs NORMAL + bootstrap
        S["del9_extreme_vs_high"] = del9(d)
        g = S["del9_extreme_vs_high"]
        print("\n[DEL9] NORMAL(P60-90) / HIGH(P90-99) / EXTREME(P99-100) — terminal@K96:")
        for k in ["NORMAL_p60_90", "HIGH_p90_99", "EXTREME_p99_100"]:
            s = g["groups"][k]
            print(f"  {k:16s} n={s['n']:5d} exp={s['mean']:+7.2f} sh={s['sharpe']:+.3f} win={s['win']:.3f} MFE={s['mfe']:6.1f} MAE={s['mae']:6.1f}")
        bt, bb, bm = g["EXTREME_minus_HIGH_terminal_iid"], g["EXTREME_minus_HIGH_terminal_block"], g["EXTREME_minus_HIGH_MFE_iid"]
        if bt:
            print(f"  EXTREME−HIGH terminal  iid:  Δ={bt['point']:+.2f} 95%CI[{bt['ci_lo']:+.2f},{bt['ci_hi']:+.2f}] p={bt['p_value']:.4f} excl0={bt['ci_excludes_zero']}")
        if bb:
            print(f"  EXTREME−HIGH terminal  BLOCK(day): Δ={bb['point']:+.2f} 95%CI[{bb['ci_lo']:+.2f},{bb['ci_hi']:+.2f}] p={bb['p_value']:.4f} excl0={bb['ci_excludes_zero']} (n_days={bb['n_days']})")
        if bm:
            print(f"  EXTREME−HIGH MFE       iid:  Δ={bm['point']:+.2f} 95%CI[{bm['ci_lo']:+.2f},{bm['ci_hi']:+.2f}] p={bm['p_value']:.4f} excl0={bm['ci_excludes_zero']}")

        # DEL 8 — feature attribution
        S["del8_feature_attribution"] = del8(d)
        print("\n[DEL8] EXTREME vs HIGH — top feature differences (Cohen's d, sorted):")
        for r in S["del8_feature_attribution"][:10]:
            print(f"  {r['feature']:34s} d={r['d_extreme_vs_high']:+.3f}  extreme={r['extreme_mean']:+.4f} high={r['high_mean']:+.4f}")

        # DEL 6 — regime split (top-tail within regime) — only where n suffices
        S["del6_by_trend"] = {}
        S["del6_by_vol"] = {}
        for col, store in [("trend_regime", "del6_by_trend"), ("vol_regime", "del6_by_vol")]:
            for lvl in sorted(d[col].dropna().unique()):
                sub = d[d[col] == lvl]
                if len(sub) < 1500:
                    S[store][str(lvl)] = {"note": f"n={len(sub)}<1500"}
                    continue
                grp = groups_by_pct(sub)
                S[store][str(lvl)] = {k: bucket_stats(v) for k, v in grp.items()}
        print("\n[DEL6] EXTREME(P99-100) exp by regime (universal or regime-specific?):")
        for col, store in [("trend_regime", "del6_by_trend"), ("vol_regime", "del6_by_vol")]:
            for lvl, v in S[store].items():
                if "EXTREME_p99_100" in v:
                    e, h = v["EXTREME_p99_100"], v["HIGH_p90_99"]
                    print(f"  {col}={lvl:14s} HIGH exp={h['mean']:+7.2f}(n{h['n']}) EXTREME exp={e['mean']:+7.2f}(n{e['n']}) "
                          f"Δ={e['mean']-h['mean']:+7.2f}  EXTREME<HIGH={e['mean']<h['mean']}")

        # DEL 7 — long vs short
        S["del7_by_side"] = {}
        for sd in ["long", "short"]:
            sub = d[d.side == sd]
            if len(sub) < 1000:
                S["del7_by_side"][sd] = {"note": f"n={len(sub)}<1000"}
                continue
            S["del7_by_side"][sd] = dict(top_tail=del2_top_tail(sub, "pnl"),
                                         extreme_vs_high={k: bucket_stats(v) for k, v in groups_by_pct(sub).items()})
        print("\n[DEL7] LONG vs SHORT — is the inverted-U one-sided?")
        for sd, v in S["del7_by_side"].items():
            if "extreme_vs_high" in v:
                e, h = v["extreme_vs_high"]["EXTREME_p99_100"], v["extreme_vs_high"]["HIGH_p90_99"]
                tt = v["top_tail"]
                print(f"  {sd:5s} HIGH exp={h['mean']:+7.2f} EXTREME exp={e['mean']:+7.2f} Δ={e['mean']-h['mean']:+7.2f} EXTREME<HIGH={e['mean']<h['mean']} "
                      f"| top0.5%−top10% mean={tt['_signal_worse_at_extreme(top0.5_vs_top10)_mean']:+.2f}")

    # ── DEL 1 full-range cross-check (range-restriction artifact test) ──────
    print("\n" + "=" * 90)
    print("[DEL1-FULLRANGE] all 147491 candidates (raw_adv full range, argmax-side terminal) — 20 ventiles")
    print("  tests whether the top-of-TAKES inverted-U is real degradation or the gate's range-restriction")
    print("=" * 90)
    OUT["fullrange"] = {}
    for sname, fn in {"2023-25 OOT-pre": SPLITS["2023-25 OOT-pre"], "2026 strict-OOT": SPLITS["2026 strict-OOT"]}.items():
        fd = full[fn(full["ts"])].dropna(subset=["pnl_full", SCORE]).copy()
        fd["b"] = pd.qcut(fd[SCORE].rank(method="first"), 20, labels=False) + 1
        rows = []
        for b in range(1, 21):
            g = fd[fd["b"] == b]
            rows.append(dict(b=b, n=int(len(g)), score_lo=round(float(g[SCORE].min()), 1),
                             score_hi=round(float(g[SCORE].max()), 1), mean=round(float(g["pnl_full"].mean()), 2),
                             win=round(float((g["pnl_full"] > 0).mean()), 4), sharpe=round(sharpe(g["pnl_full"].values), 4)))
        OUT["fullrange"][sname] = rows
        peak = max(rows, key=lambda r: r["mean"])
        print(f"\n  {sname} (n={len(fd)}, gate floor raw_adv≥-37.71 ≈ ventile where score_lo crosses -37.71):")
        for r in rows:
            mark = "  <-- gate floor" if r["score_lo"] <= -37.71 <= r["score_hi"] else ""
            print(f"    B{r['b']:2d} [{r['score_lo']:+8.1f},{r['score_hi']:+8.1f}] n={r['n']:5d} mean={r['mean']:+7.2f} win={r['win']:.3f} sh={r['sharpe']:+.3f}{mark}")
        print(f"    → full-range peak mean at B{peak['b']} (score≈[{peak['score_lo']},{peak['score_hi']}]); top ventile mean={rows[-1]['mean']:+.2f}")

    with open(OUT_JSON, "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
