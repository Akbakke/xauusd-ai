"""Regime-conditional conviction-threshold experiment (2026-06-18, user vedtak: same family as the
live DIPFIX/skip-ASIA selection edge — NOT direction, so not bounded by the 0.62 info ceiling).

Question: does a per-(session × vol_regime) conviction threshold beat the GLOBAL live op (raw_adv >= −37.71)
on OOT-2026 realized PnL? Tune the per-regime thresholds on TRAIN, freeze, evaluate on the 2026 holdout
(strict OOT) + a robustness split. Skeptical of overfit (per-regime tuning has many DOF — the project's
track record is that most selection refits fail the honest gate).

METRIC: forward_outcome take-side terminal PnL @K96 (PRE-exit, held-to-horizon). This OVERSTATES vs the
real through-exit (~2× for the marginal band) but is fine for the RELATIVE per-regime comparison + cheapest
-first (don't spend the heavy through-exit cap-3 gate until this shows promise). If a per-regime policy
beats global here, the through-exit gate is the confirmation step.
"""
import glob
import numpy as np
import pandas as pd

DEC = ("/home/andre2/GX1_DATA/reports/v12_paper_runs/nightly_learning/candidate_gates/runs/"
       "entry_iql_volbal_20260611_20260615T180412Z/entry_decisions/"
       "ENTRY_IQL_INFERENCE_FOR_V12_20260615T180414Z/decisions.parquet")
FO_DIR = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
K = "K96"
GLOBAL_THR = -37.71
GRID = [-100.0, -67.0, -37.71, -20.0, -10.0, 0.0, 10.0, 20.0]


def _load():
    d = pd.read_parquet(DEC, columns=["candidate_uid", "decision_ts_utc",
                                       "q_skip_v1", "q_take_long_v1", "q_take_short_v1"])
    d["ts"] = pd.to_datetime(d["decision_ts_utc"], utc=True)
    qL, qS, qSk = d["q_take_long_v1"].values, d["q_take_short_v1"].values, d["q_skip_v1"].values
    d["raw_adv"] = np.maximum(qL, qS) - qSk
    d["side"] = np.where(qL >= qS, "long", "short")
    cols = ["decision_ts_utc", "session", "vol_regime", "trend_regime",
            f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1"]
    fo = pd.concat([pd.read_parquet(f, columns=cols) for f in glob.glob(f"{FO_DIR}/*.parquet")],
                   ignore_index=True)
    fo["ts"] = pd.to_datetime(fo["decision_ts_utc"], utc=True)
    fo = fo.drop(columns=["decision_ts_utc"]).drop_duplicates("ts")
    d = d.merge(fo, on="ts", how="left")
    lp = d[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float).values
    sp = d[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float).values
    d["realized"] = np.where(d["side"].values == "long", lp, sp)   # take-side terminal PnL (fwd)
    d = d[~np.isnan(d["realized"].values) & d["session"].notna() & d["vol_regime"].notna()].reset_index(drop=True)
    d = d[d["session"] != "ASIA"].reset_index(drop=True)           # skip-ASIA is live
    d["regime"] = d["session"].astype(str) + "x" + d["vol_regime"].astype(str)
    return d


def _eval(df, thr_fn):
    """Apply a threshold function (regime -> thr) and return total/mean/win/n on df."""
    thr = df["regime"].map(thr_fn).fillna(GLOBAL_THR).values
    take = df["raw_adv"].values >= thr
    v = df["realized"].values[take]
    if len(v) == 0:
        return dict(total=0.0, mean=0.0, win=0.0, n=0)
    return dict(total=float(v.sum()), mean=float(v.mean()), win=float((v > 0).mean()), n=int(len(v)))


def _tune(train):
    """Per-regime: pick the grid thr maximizing TOTAL realized on train. Returns {regime: thr}."""
    out = {}
    for rg, g in train.groupby("regime"):
        ra = g["raw_adv"].values; rz = g["realized"].values
        best_thr, best_tot = GLOBAL_THR, -1e18
        for t in GRID:
            v = rz[ra >= t]
            tot = v.sum() if len(v) else 0.0
            if tot > best_tot:
                best_tot, best_thr = tot, t
        out[rg] = best_thr
    return out


def main():
    d = _load()
    print(f"[regime-thr] rows={len(d)} (non-ASIA)  regimes={d['regime'].nunique()}")
    train = d[d["ts"] <= pd.Timestamp("2025-06-01", tz="UTC")]
    test = d[d["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")]
    early = d[d["ts"] <= d["ts"].median()]; late = d[d["ts"] > d["ts"].median()]
    print(f"  train(<=2025-06)={len(train)}  test(2026)={len(test)}")

    glob_fn = (lambda rg: GLOBAL_THR)
    tuned = _tune(train)
    print("\n[per-regime tuned thresholds] (maximize TRAIN total realized; grid", GRID, ")")
    for rg in sorted(tuned, key=lambda r: tuned[r]):
        print(f"   {rg:22s} thr={tuned[rg]:+7.2f}   (train n={int((train['regime']==rg).sum())})")

    print("\n=== 2026 HOLDOUT (strict OOT) ===")
    g = _eval(test, glob_fn); r = _eval(test, lambda rg: tuned.get(rg, GLOBAL_THR))
    iqlnat = _eval(test, lambda rg: 0.0)
    print(f"  GLOBAL -37.71   : total={g['total']:9.0f}  mean={g['mean']:6.2f}  win={g['win']:.3f}  n={g['n']}")
    print(f"  PER-REGIME tuned: total={r['total']:9.0f}  mean={r['mean']:6.2f}  win={r['win']:.3f}  n={r['n']}")
    print(f"  IQL-natural (0) : total={iqlnat['total']:9.0f}  mean={iqlnat['mean']:6.2f}  win={iqlnat['win']:.3f}  n={iqlnat['n']}")
    dtot = 100*(r['total']-g['total'])/abs(g['total']) if g['total'] else 0
    print(f"  >>> per-regime vs global on 2026: total {dtot:+.1f}%  (mean {r['mean']-g['mean']:+.2f}, win {r['win']-g['win']:+.3f})")

    print("\n=== ROBUSTNESS: tune on LATE half, test on EARLY half ===")
    tuned2 = _tune(late)
    g2 = _eval(early, glob_fn); r2 = _eval(early, lambda rg: tuned2.get(rg, GLOBAL_THR))
    dtot2 = 100*(r2['total']-g2['total'])/abs(g2['total']) if g2['total'] else 0
    print(f"  GLOBAL : total={g2['total']:9.0f} mean={g2['mean']:6.2f} win={g2['win']:.3f}")
    print(f"  PER-RG : total={r2['total']:9.0f} mean={r2['mean']:6.2f} win={r2['win']:.3f}  >>> {dtot2:+.1f}% total")

    # per-regime 2026 quality at the GLOBAL op (which regimes are weak — the lever's premise)
    print("\n=== per-regime 2026 realized at GLOBAL -37.71 (weak regimes = candidates to tighten) ===")
    t = test[test["raw_adv"] >= GLOBAL_THR]
    for rg, g3 in sorted(t.groupby("regime"), key=lambda kv: kv[1]["realized"].mean()):
        v = g3["realized"].values
        print(f"   {rg:22s} n={len(v):5d} mean={v.mean():7.2f} win={(v>0).mean():.3f} total={v.sum():8.0f}")

    # ── DEFENSIVE policy: tighten to IQL-natural (thr=0) ONLY regimes whose MARGINAL band [-37.71,0)
    #    is robustly negative-EV (mean<0 on BOTH train halves) — low-DOF, only cuts proven-bad marginal takes.
    def marg_mean(df, rg):
        m = (df["regime"] == rg) & (df["raw_adv"] >= GLOBAL_THR) & (df["raw_adv"] < 0.0)
        v = df.loc[m, "realized"].values
        return (v.mean() if len(v) >= 30 else np.nan, len(v))
    th = d[d["ts"] <= d["ts"].median()]; tl = d[d["ts"] > d["ts"].median()]
    print("\n=== DEFENSIVE: per-regime MARGINAL-band [-37.71,0) EV — train-half-1 vs half-2 (robustness) ===")
    defensive = {}
    for rg in sorted(d["regime"].unique()):
        e, ne = marg_mean(th, rg); l, nl = marg_mean(tl, rg)
        bad = (not np.isnan(e)) and (not np.isnan(l)) and e < 0 and l < 0
        if bad:
            defensive[rg] = 0.0
        flag = "  <<< TIGHTEN (both halves neg)" if bad else ""
        print(f"   {rg:22s} h1_marg_mean={e:7.2f}(n{ne}) h2_marg_mean={l:7.2f}(n{nl}){flag}")
    print(f"\n  defensive tightens {len(defensive)} regimes to thr=0: {sorted(defensive)}")
    gd = _eval(test, glob_fn); rd = _eval(test, lambda rg: defensive.get(rg, GLOBAL_THR))
    ddtot = 100*(rd['total']-gd['total'])/abs(gd['total']) if gd['total'] else 0
    print(f"  2026 GLOBAL    : total={gd['total']:9.0f} mean={gd['mean']:6.2f} win={gd['win']:.3f} n={gd['n']}")
    print(f"  2026 DEFENSIVE : total={rd['total']:9.0f} mean={rd['mean']:6.2f} win={rd['win']:.3f} n={rd['n']}  >>> total {ddtot:+.1f}%")
    print("  (FWD metric overstates + penalizes exit-managed shorts; a positive here = run the through-exit cap-3 gate to confirm.)")


if __name__ == "__main__":
    main()
