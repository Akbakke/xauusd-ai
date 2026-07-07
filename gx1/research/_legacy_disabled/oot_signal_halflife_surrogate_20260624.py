"""FALSIFY "slow learning / adaptivity is the lever" — PART 2: surrogate signal half-life (Del 1/8/9).

User falsification 2026-06-24 — try to KILL adaptivity. We CANNOT cheaply re-fit V10/IQL per window
(cost + rule 3), so we measure the DATA's learnable signal decay with an identical-each-time surrogate
(HistGradientBoosting on the same decision-time features → long-side terminal@K96), varying ONLY the
training window, testing on a FIXED future period. This BOUNDS what ANY model (incl. adaptive retraining)
could exploit: if a surrogate fit on recent/short data can't beat one fit on old/long data OOT, adaptivity
can't help. Corroborated by the ALREADY-DONE real-model fold comparison (oldest FOLD_1 best OOT, recency
refuted P=0.71). Reuses forward_outcome_clean (no chain retrain; surrogate is a throwaway research probe).

Del1/8 = window sweep (5y/3y/2y/1y/6m/3m → fixed 2026 + 2025H2 test) + bootstrap. Del9 = walk-forward
adaptivity ceiling (static-2022 vs expanding-monthly vs trailing-2y/6m monthly retrain). NO new feature/
model/label for the chain; cement+live UNCHANGED.
Run: .venv/bin/python -m gx1.research.oot_signal_halflife_surrogate_20260624
"""
from __future__ import annotations
import glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

FO = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
OUT_JSON = "/tmp/signal_halflife_surrogate_20260624.json"
RNG = np.random.default_rng(20260624)

FEATURES = ["ema20_slope_canon_v1", "ema100_slope_canon_v1", "_v1h1_slope5_canon_v1", "_v1h4_slope5_canon_v1",
            "d1_ema_slope_20_canon_v2_canon_v1", "pos_vs_ema200_canon_v1", "m15_trend_sign_canon_v2_canon_v1",
            "_v1_rsi14_canon_v1", "_v1_rsi2_canon_v1", "_v1h1_rsi14_z_canon_v1", "_v1h4_rsi14_z_canon_v1",
            "d1_rsi14_canon_v2_canon_v1", "m15_rsi14_canon_v2_canon_v1", "atr_z_canon_v1", "_v1_atr_z_10_100_canon_v1",
            "vol_ratio_canon_v1", "_v1_bb_squeeze_20_2_canon_v1", "_v1_bb_bandwidth_delta_10_canon_v1",
            "_v1_vwap_drift48_canon_v1", "m5h1_momentum_canon_v1", "_v1h1_ema_diff_canon_v1", "_v1h4_ema_diff_canon_v1",
            "p_long", "p_short", "margin", "atr_bps"]
LONG = "take_now_long_terminal_pnl_at_K96_v1"
SHORT = "take_now_short_terminal_pnl_at_K96_v1"


def sharpe(p):
    p = np.asarray(p, float)
    return float(p.mean() / (p.std() + 1e-9)) if len(p) > 1 else float("nan")


def load():
    cols = list(dict.fromkeys(["candidate_uid", "decision_ts_utc", LONG, SHORT] + FEATURES))
    parts = []
    for fp in sorted(glob.glob(f"{FO}/*.parquet")):
        d = pd.read_parquet(fp, columns=[c for c in cols if c])
        if len(d):
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    df = df.dropna(subset=[LONG, SHORT]).reset_index(drop=True)
    return df


def fit_predict(tr, te):
    feats = [f for f in FEATURES if f in tr.columns]
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05,
                                      l2_regularization=1.0, random_state=0)
    m.fit(tr[feats].values, tr[LONG].values)
    return m.predict(te[feats].values)


def eval_pred(te, pred):
    y = te[LONG].values
    ic = spearmanr(pred, y).correlation
    thr = np.percentile(pred, 90)
    topd = y[pred >= thr]
    try:
        auc = roc_auc_score((te[LONG].values > te[SHORT].values).astype(int), pred)
    except Exception:
        auc = np.nan
    return dict(n=int(len(te)), auc=round(float(auc), 4), rankIC=round(float(ic), 4),
                topD_ret=round(float(topd.mean()), 2), topD_sharpe=round(sharpe(topd), 4),
                exp=round(float(y.mean()), 2), n_topD=int(len(topd)))


def window_start(end, label):
    months = {"5y": 60, "3y": 36, "2y": 24, "1y": 12, "6m": 6, "3m": 3}[label]
    return end - pd.DateOffset(months=months)


def main():
    df = load()
    print(f"[data] {len(df)} decision rows  {df.ts.min()} .. {df.ts.max()}  feats={len([f for f in FEATURES if f in df.columns])}")
    OUT = {"meta": dict(n=len(df), span=f"{df.ts.min()}..{df.ts.max()}", target=LONG, surrogate="HistGBM(long_term)")}

    # ── DEL 1 / DEL 8: window sweep, fixed future test ──
    tests = {"2026 strict-OOT": ("2026-01-01", None), "2025-H2": ("2025-07-01", "2026-01-01")}
    OUT["del1_window_sweep"] = {}
    preds_for_boot = {}
    for tname, (t0, t1) in tests.items():
        te = df[df.ts >= pd.Timestamp(t0, tz="UTC")] if t1 is None else df[(df.ts >= pd.Timestamp(t0, tz="UTC")) & (df.ts < pd.Timestamp(t1, tz="UTC"))]
        end = pd.Timestamp(t0, tz="UTC")
        print("\n" + "=" * 100); print(f"DEL1/8 WINDOW SWEEP — test={tname} (n={len(te)}); train windows END at {t0}, vary lookback"); print("=" * 100)
        print(f"  {'window':6s} {'train_n':>8} {'AUC':>7} {'rankIC':>8} {'topD_ret':>9} {'topD_Sharpe':>12} {'exp_all':>8}")
        rows = {}
        preds_for_boot[tname] = {}
        for w in ["5y", "3y", "2y", "1y", "6m", "3m"]:
            tr = df[(df.ts >= window_start(end, w)) & (df.ts < end)]
            if len(tr) < 2000:
                rows[w] = {"note": f"train_n={len(tr)}<2000"}; continue
            pred = fit_predict(tr, te)
            r = eval_pred(te, pred); r["train_n"] = int(len(tr))
            rows[w] = r
            preds_for_boot[tname][w] = pred
            print(f"  {w:6s} {r['train_n']:8d} {r['auc']:7.4f} {r['rankIC']:+8.4f} {r['topD_ret']:+9.2f} {r['topD_sharpe']:+12.4f} {r['exp']:+8.2f}")
        OUT["del1_window_sweep"][tname] = rows

    # ── DEL 8: bootstrap difference (short 1y vs long 3y) on 2026 top-decile return + rankIC ──
    print("\n" + "=" * 100); print("DEL8 — BOOTSTRAP: is SHORT/recent (1y) better than LONG (3y) on 2026? (block by day)"); print("=" * 100)
    te26 = df[df.ts >= pd.Timestamp("2026-01-01", tz="UTC")].reset_index(drop=True)
    OUT["del8_bootstrap"] = {}
    if "1y" in preds_for_boot["2026 strict-OOT"] and "3y" in preds_for_boot["2026 strict-OOT"]:
        p1, p3, y = preds_for_boot["2026 strict-OOT"]["1y"], preds_for_boot["2026 strict-OOT"]["3y"], te26[LONG].values
        day = te26.ts.dt.floor("D").values
        udays = np.unique(day)
        # top-decile-return difference (1y − 3y), block bootstrap by day
        def topd_ret(pred, mask):
            sub = pred[mask]; ysub = y[mask]
            if len(sub) < 20: return np.nan
            return ysub[sub >= np.percentile(sub, 90)].mean()
        diffs = []
        for _ in range(5000):
            pick = RNG.choice(len(udays), len(udays))
            mask = np.isin(day, udays[pick])
            d = topd_ret(p1, mask) - topd_ret(p3, mask)
            if not np.isnan(d): diffs.append(d)
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        pval = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
        base = topd_ret(p1, np.ones(len(y), bool)) - topd_ret(p3, np.ones(len(y), bool))
        OUT["del8_bootstrap"] = dict(point_1y_minus_3y_topD_ret=round(float(base), 2), ci_lo=round(float(lo), 2),
                                     ci_hi=round(float(hi), 2), p_value=round(float(pval), 4), ci_excludes_zero=bool(lo > 0 or hi < 0))
        print(f"  1y − 3y top-decile return on 2026: Δ={base:+.2f} bps  95%CI[{lo:+.2f},{hi:+.2f}] p={pval:.4f} "
              f"excl0={OUT['del8_bootstrap']['ci_excludes_zero']}  → shorter-better? {'YES' if base>0 and (lo>0) else 'NO'}")

    # ── DEL 9: adaptivity ceiling (walk-forward monthly) ──
    print("\n" + "=" * 100); print("DEL9 — ADAPTIVITY CEILING: walk-forward monthly top-decile-long, static vs retrained"); print("=" * 100)
    df["ym"] = df.ts.dt.to_period("M")
    test_months = [p for p in sorted(df.ym.unique()) if p >= pd.Period("2023-01")]
    arms = {"static_2020_2022": None, "expanding": "expanding", "trailing_2y": 24, "trailing_6m": 6}
    # pre-fit the static model once
    static_tr = df[df.ts < pd.Timestamp("2023-01-01", tz="UTC")]
    feats = [f for f in FEATURES if f in df.columns]
    static_model = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    static_model.fit(static_tr[feats].values, static_tr[LONG].values)
    results = {a: {"ret": [], "monthly_mean": []} for a in arms}
    for p in test_months:
        te = df[df.ym == p]
        if len(te) < 50: continue
        m_start = p.to_timestamp().tz_localize("UTC")
        for a, spec in arms.items():
            if a == "static_2020_2022":
                pred = static_model.predict(te[feats].values)
            else:
                tr = df[df.ts < m_start] if spec == "expanding" else df[(df.ts >= m_start - pd.DateOffset(months=spec)) & (df.ts < m_start)]
                if len(tr) < 2000: continue
                mm = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, l2_regularization=1.0, random_state=0)
                mm.fit(tr[feats].values, tr[LONG].values)
                pred = mm.predict(te[feats].values)
            y = te[LONG].values
            topd = y[pred >= np.percentile(pred, 90)]
            results[a]["ret"].extend(topd.tolist())
            results[a]["monthly_mean"].append(float(topd.mean()))
    OUT["del9_adaptivity"] = {}
    print(f"  {'arm':18s} {'n_trades':>9} {'topD_exp':>9} {'pooled_Sharpe':>14} {'monthly_Sharpe':>15} {'total_bps':>10}")
    for a in arms:
        r = np.array(results[a]["ret"]); mm = np.array(results[a]["monthly_mean"])
        if len(r) == 0: continue
        OUT["del9_adaptivity"][a] = dict(n=int(len(r)), topD_exp=round(float(r.mean()), 2), pooled_sharpe=round(sharpe(r), 4),
                                         monthly_sharpe=round(sharpe(mm), 4), total=round(float(r.sum()), 0))
        v = OUT["del9_adaptivity"][a]
        print(f"  {a:18s} {v['n']:9d} {v['topD_exp']:+9.2f} {v['pooled_sharpe']:+14.4f} {v['monthly_sharpe']:+15.4f} {v['total']:+10.0f}")
    base = OUT["del9_adaptivity"].get("static_2020_2022", {})
    if base:
        best_adaptive = max((v for k, v in OUT["del9_adaptivity"].items() if k != "static_2020_2022"),
                            key=lambda v: v["topD_exp"], default=None)
        if best_adaptive:
            gain = best_adaptive["topD_exp"] - base["topD_exp"]
            OUT["del9_adaptivity"]["MAX_ADAPTIVITY_GAIN_bps_per_trade"] = round(float(gain), 2)
            print(f"\n  → MAX adaptivity gain (best adaptive − static) = {gain:+.2f} bps/trade on top-decile  "
                  f"({'material' if gain > 5 else 'NEGLIGIBLE/none → hypothesis weakened'})")

    with open(OUT_JSON, "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
