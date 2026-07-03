"""FALSIFY "missing edge = slow learning / signal drift / non-stationarity" — PART 1 (no training).

User falsification 2026-06-24 (try to KILL the adaptivity hypothesis). This part measures the ACTUAL
deployed signal (raw_adv from the live FOLD_1 V10+IQL, fixed model trained ~2020-22) over 2020-2026 —
the most faithful decay/drift test (no surrogate). Reuses /tmp/live_policy_universe (147491 candidates,
raw_adv + argmax-side terminal) + forward_outcome (features for PSI + the IQL-conflict cell). The
surrogate window-sweep / adaptivity ceiling (Del 1/8/9) lives in oot_signal_halflife_surrogate_20260624.py.

Covers: Del2 decay-curve, Del3 rolling-OOT monthly distribution (April-concentrated?), Del4 score-scale
drift, Del5 rank-stability, Del6 feature PSI, Del7 prediction drift, + the FINAL all-TF-down & p_long≥0.75
cell split by IQL argmax action (does IQL hold info V10 lacks = a model-conflict, NOT a regime filter?).

raw_adv = max(q_long,q_short)−q_skip (live gate metric). Outcome = argmax-side terminal_pnl@K96 (entry-side
label). NO retrain, NO new model/feature/label. Diagnostics only; cement+live UNCHANGED.
Run: .venv/bin/python -m gx1.research.oot_signal_drift_20260624
"""
from __future__ import annotations
import glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

UNIV = "/tmp/live_policy_universe_20260623.parquet"
FO = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
OUT_JSON = "/tmp/signal_drift_20260624.json"

# FO columns: TF slopes + ema200 (cell), long/short outcomes, regime, PSI feature subset
FO_SLOPES = dict(m15="m15_trend_sign_canon_v2_canon_v1", h1="_v1h1_slope5_canon_v1",
                 h4="_v1h4_slope5_canon_v1", d1="d1_ema_slope_20_canon_v2_canon_v1",
                 ema200="pos_vs_ema200_canon_v1")
FO_OUT = ["take_now_long_terminal_pnl_at_K96_v1", "take_now_long_mfe_bps_at_K96_v1", "take_now_long_mae_bps_at_K96_v1"]
PSI_FEATS = ["ema20_slope_canon_v1", "ema100_slope_canon_v1", "_v1h1_slope5_canon_v1", "_v1h4_slope5_canon_v1",
             "d1_ema_slope_20_canon_v2_canon_v1", "pos_vs_ema200_canon_v1", "_v1_rsi14_canon_v1",
             "_v1h1_rsi14_z_canon_v1", "d1_rsi14_canon_v2_canon_v1", "atr_z_canon_v1", "vol_ratio_canon_v1",
             "_v1_atr_z_10_100_canon_v1", "_v1_bb_squeeze_20_2_canon_v1", "_v1_vwap_drift48_canon_v1", "p_long"]


def sharpe(p):
    p = np.asarray(p, float)
    return float(p.mean() / (p.std() + 1e-9)) if len(p) > 1 else float("nan")


def load_fo():
    cols = ["candidate_uid", "decision_ts_utc", "p_long", "p_short", "margin", "vol_regime", "trend_regime"] \
        + list(FO_SLOPES.values()) + FO_OUT + PSI_FEATS
    cols = list(dict.fromkeys(cols))
    parts = []
    for fp in sorted(glob.glob(f"{FO}/*.parquet")):
        d = pd.read_parquet(fp, columns=[c for c in cols if c is not None])
        if len(d):
            parts.append(d)
    fo = pd.concat(parts, ignore_index=True)
    fo["ts"] = pd.to_datetime(fo["decision_ts_utc"], utc=True)
    fo["ym"] = fo["ts"].dt.strftime("%Y-%m")
    return fo


def psi(expected, actual, bins=10):
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if len(expected) < 50 or len(actual) < 50:
        return np.nan
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return np.nan
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / len(expected) + 1e-6
    a = np.histogram(actual, qs)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    u = pd.read_parquet(UNIV)
    u["ts"] = pd.to_datetime(u["ts"], utc=True)
    u["ym"] = u["ts"].dt.strftime("%Y-%m")
    u["yr"] = u["ts"].dt.year
    u["iql_argmax"] = np.where((u.q_long >= u.q_skip) & (u.q_long >= u.q_short), "TAKE_LONG",
                       np.where((u.q_short >= u.q_skip) & (u.q_short > u.q_long), "TAKE_SHORT", "SKIP"))
    # outcome for the SCORE analyses = argmax-side terminal (raw_adv is the argmax advantage)
    uu = u.dropna(subset=["pnl_argmax", "raw_adv"]).copy()
    OUT = {"meta": dict(n=len(u), span=f"{u.ts.min()}..{u.ts.max()}", model="live FOLD_1 (fixed, ~2020-22 train)")}

    # ── DEL 3 + DEL 2: monthly rolling-OOT of the REAL deployed signal ──
    print("=" * 100); print("DEL3/DEL2 — MONTHLY OOT of the deployed signal (raw_adv→terminal@K96): rankIC / topD-Sharpe / expectancy")
    print("=" * 100)
    rows = []
    for ym, g in uu.groupby("ym"):
        if len(g) < 60:
            continue
        ic = spearmanr(g.raw_adv, g.pnl_argmax).correlation
        thr = np.percentile(g.raw_adv, 90)
        topd = g[g.raw_adv >= thr]["pnl_argmax"].values
        rows.append(dict(ym=ym, n=len(g), rankIC=round(float(ic), 4), topD_exp=round(float(topd.mean()), 1),
                         topD_sharpe=round(sharpe(topd), 3), exp_all=round(float(g.pnl_argmax.mean()), 1)))
    md = pd.DataFrame(rows)
    OUT["del3_monthly"] = rows
    for r in rows:
        flag = "  <== APR" if r["ym"].endswith("-04") and r["ym"].startswith("2026") else ""
        print(f"  {r['ym']} n={r['n']:4d} rankIC={r['rankIC']:+.3f} topD_exp={r['topD_exp']:+7.1f} topD_sh={r['topD_sharpe']:+.3f} exp_all={r['exp_all']:+7.1f}{flag}")
    # decay vs months-since-training (OOT from 2023-01)
    oot = md[md.ym >= "2023-01"].copy()
    oot["msince"] = (oot.ym.str[:4].astype(int) - 2023) * 12 + oot.ym.str[5:7].astype(int)
    slope_ic = np.polyfit(oot.msince, oot.rankIC, 1)[0] if len(oot) > 6 else np.nan
    slope_sh = np.polyfit(oot.msince, oot.topD_sharpe.fillna(0), 1)[0] if len(oot) > 6 else np.nan
    OUT["del2_decay"] = dict(rankIC_slope_per_month=round(float(slope_ic), 5), topD_sharpe_slope_per_month=round(float(slope_sh), 5),
                             rankIC_first6m=round(float(oot.head(6).rankIC.mean()), 4), rankIC_last6m=round(float(oot.tail(6).rankIC.mean()), 4),
                             rankIC_corr_with_time=round(float(np.corrcoef(oot.msince, oot.rankIC)[0, 1]), 4))
    print(f"\n  DEL2 decay: rankIC slope/month={OUT['del2_decay']['rankIC_slope_per_month']:+.5f} "
          f"(corr w/ time {OUT['del2_decay']['rankIC_corr_with_time']:+.3f}); first-6m rankIC {OUT['del2_decay']['rankIC_first6m']:+.3f} "
          f"vs last-6m {OUT['del2_decay']['rankIC_last6m']:+.3f}  → smooth decay? {'NO' if abs(OUT['del2_decay']['rankIC_corr_with_time'])<0.3 else 'maybe'}")
    print(f"  DEL3 episode-concentration: worst months = {md.nsmallest(5,'rankIC')[['ym','rankIC','n']].values.tolist()}")
    print(f"    2026-04 rankIC={md.loc[md.ym=='2026-04','rankIC'].values}  vs median monthly rankIC={md.rankIC.median():+.3f}")

    # ── DEL 4: score-scale drift over time ──
    print("\n" + "=" * 100); print("DEL4 — raw_adv SCALE drift (monthly percentiles): does score-MEANING move?"); print("=" * 100)
    rows = []
    for ym, g in uu.groupby("ym"):
        if len(g) < 60:
            continue
        p = np.percentile(g.raw_adv, [50, 75, 90, 95, 99])
        rows.append(dict(ym=ym, p50=round(p[0], 1), p75=round(p[1], 1), p90=round(p[2], 1), p95=round(p[3], 1), p99=round(p[4], 1)))
    OUT["del4_score_scale"] = rows
    for r in rows[::3]:
        print(f"  {r['ym']} p50={r['p50']:+8.1f} p75={r['p75']:+8.1f} p90={r['p90']:+8.1f} p95={r['p95']:+8.1f} p99={r['p99']:+8.1f}")
    sd = pd.DataFrame(rows)
    print(f"  → p99 range across months: [{sd.p99.min():.0f}, {sd.p99.max():.0f}] (×{sd.p99.max()/max(sd.p99.median(),1):.1f} vs median {sd.p99.median():.0f}); "
          f"p50 range [{sd.p50.min():.0f},{sd.p50.max():.0f}] → SCALE DRIFTS = a static threshold means different things per month")

    # ── DEL 5: rank-stability (fixed within-month percentile → outcome) ──
    print("\n" + "=" * 100); print("DEL5 — RANK-STABILITY: does top-X% (within-month) mean the same over years?"); print("=" * 100)
    rows = []
    for ym, g in uu.groupby("ym"):
        if len(g) < 100:
            continue
        rec = dict(ym=ym, n=len(g))
        for top, lbl in [(10, "t10"), (5, "t5"), (1, "t1")]:
            sub = g[g.raw_adv >= np.percentile(g.raw_adv, 100 - top)]["pnl_argmax"].values
            rec[f"{lbl}_exp"] = round(float(sub.mean()), 1)
            rec[f"{lbl}_win"] = round(float((sub > 0).mean()), 3)
        rows.append(rec)
    OUT["del5_rank_stability"] = rows
    rk = pd.DataFrame(rows)
    print("  yearly mean of within-month top-X% expectancy / win:")
    rk["yr"] = rk.ym.str[:4]
    for yr, g in rk.groupby("yr"):
        print(f"   {yr}: top10 exp={g.t10_exp.mean():+6.1f} win={g.t10_win.mean():.3f} | top5 exp={g.t5_exp.mean():+6.1f} win={g.t5_win.mean():.3f} | top1 exp={g.t1_exp.mean():+6.1f} win={g.t1_win.mean():.3f}")

    # ── DEL 7: prediction drift (score variance + entropy + p_long) ──
    print("\n" + "=" * 100); print("DEL7 — PREDICTION DRIFT: score variance / entropy / mean p_long over time"); print("=" * 100)
    rows = []
    for ym, g in uu.groupby("ym"):
        if len(g) < 60:
            continue
        x = g.raw_adv.values
        h, _ = np.histogram(x, bins=20, range=(np.percentile(x, 1), np.percentile(x, 99)))
        pdist = h / max(h.sum(), 1) + 1e-9
        ent = float(-(pdist * np.log(pdist)).sum())
        rows.append(dict(ym=ym, std=round(float(x.std()), 1), entropy=round(ent, 3),
                         mean_plong=round(float(g.p_long.mean()), 3), frac_take=round(float((g.raw_adv >= -37.71).mean()), 3)))
    OUT["del7_prediction_drift"] = rows
    for r in rows[::3]:
        print(f"  {r['ym']} score_std={r['std']:7.1f} entropy={r['entropy']:.3f} mean_p_long={r['mean_plong']:.3f} frac_above_gate={r['frac_take']:.3f}")

    # ── DEL 6: feature PSI (drift vs 2020-22 baseline) ──
    print("\n" + "=" * 100); print("DEL6 — FEATURE DRIFT (PSI vs 2020-22 baseline; >0.25 = large shift)"); print("=" * 100)
    fo = load_fo()
    base = fo[fo.ts < pd.Timestamp("2023-01-01", tz="UTC")]
    psi_rows = {}
    for f in PSI_FEATS:
        if f not in fo.columns:
            continue
        per_month = {}
        for ym, g in fo[fo.ts >= pd.Timestamp("2023-01-01", tz="UTC")].groupby("ym"):
            per_month[ym] = psi(base[f].values, g[f].values)
        vals = [v for v in per_month.values() if not np.isnan(v)]
        psi_rows[f] = dict(mean_psi=round(float(np.mean(vals)), 3) if vals else np.nan,
                           max_psi=round(float(np.max(vals)), 3) if vals else np.nan,
                           psi_2026=round(float(np.nanmean([per_month.get(f"2026-{m:02d}", np.nan) for m in range(1, 6)])), 3))
    OUT["del6_psi"] = psi_rows
    for f, v in sorted(psi_rows.items(), key=lambda kv: -(kv[1]["mean_psi"] if not np.isnan(kv[1]["mean_psi"]) else -1)):
        print(f"  {f:42s} mean_PSI={v['mean_psi']:6.3f} max={v['max_psi']:6.3f} 2026={v['psi_2026']:6.3f}")

    # ── FINAL: all-TF-down & p_long≥0.75 cell, split by IQL argmax action ──
    print("\n" + "=" * 100); print("FINAL — cell: M15<0 & H1<0 & H4<0 & D1<0 & below-EMA200 & V10 p_long≥0.75, split by IQL action"); print("=" * 100)
    m = u.merge(fo[["candidate_uid"] + list(FO_SLOPES.values()) + FO_OUT + ["p_long"]].rename(columns={"p_long": "p_long_fo"}),
                left_on="uid", right_on="candidate_uid", how="inner")
    cond = ((m[FO_SLOPES["m15"]] < 0) & (m[FO_SLOPES["h1"]] < 0) & (m[FO_SLOPES["h4"]] < 0) &
            (m[FO_SLOPES["d1"]] < 0) & (m[FO_SLOPES["ema200"]] < 0) & (m["p_long"] >= 0.75))
    cell = m[cond].copy()
    cell["long_term"] = cell["take_now_long_terminal_pnl_at_K96_v1"]
    cell["long_mfe"] = cell["take_now_long_mfe_bps_at_K96_v1"]
    cell["long_mae"] = cell["take_now_long_mae_bps_at_K96_v1"]
    cell = cell.dropna(subset=["long_term"])

    def grp(d):
        p = d.long_term.values
        return dict(n=int(len(d)), exp=round(float(p.mean()), 2), sharpe=round(sharpe(p), 4),
                    win=round(float((p > 0).mean()), 4), median=round(float(np.median(p)), 2),
                    mfe=round(float(d.long_mfe.mean()), 1), mae=round(float(d.long_mae.mean()), 1))

    def report_cell(dd, tag):
        print(f"\n  [{tag}]  cell n={len(dd)}  (whole-universe LONG-side baseline exp={m.dropna(subset=['take_now_long_terminal_pnl_at_K96_v1']).take_now_long_terminal_pnl_at_K96_v1.mean():+.1f})")
        out = {}
        for a in ["TAKE_LONG", "SKIP", "TAKE_SHORT"]:
            sub = dd[dd.iql_argmax == a]
            if len(sub) >= 10:
                out[a] = grp(sub)
                s = out[a]
                print(f"    IQL={a:10s} n={s['n']:4d} exp={s['exp']:+7.2f} sh={s['sharpe']:+.3f} win={s['win']:.3f} med={s['median']:+7.2f} MFE={s['mfe']:6.1f} MAE={s['mae']:6.1f}")
            else:
                out[a] = {"n": int(len(sub)), "note": "n<10"}
                print(f"    IQL={a:10s} n={len(sub):4d}  (n<10)")
        if "TAKE_LONG" in out and "SKIP" in out and "n" in out["SKIP"] and out["SKIP"]["n"] >= 10:
            d_as = out["TAKE_LONG"]["exp"] - out["SKIP"]["exp"]
            print(f"    → IQL-conflict signal: TAKE_LONG − SKIP exp = {d_as:+.2f} bps  "
                  f"({'IQL HAS extra info (B worse)' if d_as > 5 else 'NO extra info — refuted' if abs(d_as) <= 5 else 'SKIP BETTER (inverted)'})")
        return out

    OUT["final_cell_all"] = report_cell(cell, "ALL years")
    OUT["final_cell_2026"] = report_cell(cell[cell.yr == 2026], "2026 strict-OOT")
    OUT["final_cell_pre2026"] = report_cell(cell[cell.yr < 2026], "pre-2026")

    with open(OUT_JSON, "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
