"""BAD_LONG vs GOOD_LONG entry-separability (2026-06-23, user). Can "early mean-reversion LONGs that go
hard against us and never develop edge" be told apart from real winners AT ENTRY TIME?

Cohort = live-policy LONG TAKEs (FOLD_1, conv-gate+DIPFIX). Primary split = 2026 OOT; OOT robustness =
train pre-2026 LONGs -> test 2026 LONGs.
  BAD_LONG  = [(MAE>=100 & MFE<=30) OR (slut-PnL in [-20,+20])] AND NOT GOOD   (disjoint)
  GOOD_LONG = MFE>=80 & PnL>0
Features are PRE-ENTRY ONLY (V10 heads, direction probs, margin, raw_adv, regime/EMA-dist/accel). MAE/MFE/
PnL are used for LABELS ONLY, never as features (no leakage). Reuses the cached regime universe + joins the
V10 head block from forward_outcome. NO new model, NO replay. "clean_edge" is NOT a distinct V10 head —
closest = tradable_prob / path_quality_pred (flagged).

KEY TEST (AGENTS.md pre-gate discipline): separability must hold on HELD-OUT data — 5-fold CV within 2026
AND strict pre-2026->2026 OOT — else outcome-derived labels fit in-sample trivially. Bar: AUC>0.60.
"""
from __future__ import annotations
import glob, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from gx1.research.oot_live_policy_falsification_20260623 import FO, K

REG = "/tmp/regime_universe_20260623.parquet"
import pyarrow.parquet as pq
ALLCOLS = pq.ParquetFile(sorted(glob.glob(f"{FO}/*.parquet"))[0]).schema.names
V10 = [c for c in ALLCOLS if c.startswith("v10_")]
SCALAR_HEADS = [c for c in ["tradable_prob", "path_quality_pred", "path_quality_std", "uncertainty_score",
                            "hold_horizon_pred", "p_hat"] if c in ALLCOLS]
EMADIST = [c for c in ["_v1_ema_diff_canon_v1", "pos_vs_ema200_canon_v1", "_v1h1_ema_diff_canon_v1",
                       "_v1h4_ema_diff_canon_v1", "_v1_ret_ema_ratio_5_34_canon_v1",
                       "_v1h1_slope3_canon_v1", "_v1h4_slope3_canon_v1", "_v1_close_ema_slope_3_canon_v1"] if c in ALLCOLS]
JOIN = ["candidate_uid"] + V10 + SCALAR_HEADS + EMADIST


def build():
    u = pd.read_parquet(REG)
    u["ts"] = pd.to_datetime(u["ts"], utc=True)
    u = u[u.side == "long"].copy()
    reg = pd.concat([pd.read_parquet(f, columns=[c for c in JOIN if c in ALLCOLS]) for f in sorted(glob.glob(f"{FO}/*.parquet"))],
                    ignore_index=True).drop_duplicates("candidate_uid")
    d = u.merge(reg, left_on="uid", right_on="candidate_uid", how="left")
    # derived accel proxies (slope3 - slope5 = recent vs slightly-older = acceleration)
    if "_v1h1_slope3_canon_v1" in d and "_v1h1_slope5_canon_v1" in d:
        d["accel_h1"] = d["_v1h1_slope3_canon_v1"] - d["_v1h1_slope5_canon_v1"]
    if "_v1h4_slope3_canon_v1" in d and "_v1h4_slope5_canon_v1" in d:
        d["accel_h4"] = d["_v1h4_slope3_canon_v1"] - d["_v1h4_slope5_canon_v1"]
    # cohorts (labels from OUTCOME only)
    good = (d["mfe_bps"] >= 80) & (d["pnl"] > 0)
    bad_raw = ((d["mae"] >= 100) & (d["mfe_bps"] <= 30)) | (d["pnl"].between(-20, 20))
    d["coh"] = np.where(good, "GOOD", np.where(bad_raw & ~good, "BAD", "OTHER"))
    return d


# pre-entry feature set (NO outcome cols)
def feat_cols(d):
    base = V10 + SCALAR_HEADS + EMADIST + [
        "p_long", "p_short", "margin", "raw_adv", "atr_pctl", "trend_str", "trend_align",
        "_v1h1_slope5_canon_v1", "_v1h4_slope5_canon_v1", "d1_ema_slope_20_canon_v2_canon_v1",
        "ema100_slope_canon_v1", "ema20_slope_canon_v1", "dist_hi96_proxy", "range_pos_proxy",
        "breakout96_proxy", "_v1_bb_squeeze_20_2_canon_v1", "atr_z_canon_v1", "vol_ratio_canon_v1",
        "accel_h1", "accel_h4", "hour_utc"]
    return [c for c in base if c in d.columns and d[c].notna().any()]


def univ_auc(d, cols):
    y = (d["coh"] == "BAD").astype(int).values
    rows = []
    for c in cols:
        x = d[c].fillna(d[c].median()).values
        if np.std(x) < 1e-12:
            continue
        a = roc_auc_score(y, x)
        rows.append((c, round(float(a), 4), round(float(abs(a - 0.5)), 4),
                     round(float(d.loc[d.coh == "BAD", c].mean() - d.loc[d.coh == "GOOD", c].mean()), 4)))
    return sorted(rows, key=lambda r: -r[2])


def classify(d_tr, d_te, cols, label):
    Xtr = d_tr[cols].fillna(d_tr[cols].median())
    Xte = d_te[cols].fillna(d_tr[cols].median())
    ytr = (d_tr["coh"] == "BAD").astype(int).values
    yte = (d_te["coh"] == "BAD").astype(int).values
    out = {}
    for nm, clf in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))),
                    ("gbm", GradientBoostingClassifier(max_depth=3, n_estimators=150, subsample=0.7, random_state=0))]:
        clf.fit(Xtr.values, ytr)
        pr = clf.predict_proba(Xte.values)[:, 1]
        thr = np.quantile(pr, 0.90)
        topdec = yte[pr >= thr]
        out[nm] = dict(auc=round(float(roc_auc_score(yte, pr)), 4),
                       prauc=round(float(average_precision_score(yte, pr)), 4),
                       base_rate=round(float(yte.mean()), 4),
                       top_decile_precision=round(float(topdec.mean()), 4) if len(topdec) else None,
                       n_train=len(ytr), n_test=len(yte))
    print(f"  [{label}] base_rate(BAD)={out['logreg']['base_rate']}  "
          f"logreg AUC={out['logreg']['auc']} PR={out['logreg']['prauc']} top10%prec={out['logreg']['top_decile_precision']}  | "
          f"gbm AUC={out['gbm']['auc']} top10%prec={out['gbm']['top_decile_precision']}")
    return out


def main():
    d = build()
    pre = d["ts"] < pd.Timestamp("2026-01-01", tz="UTC")
    o26 = ~pre
    d26 = d[o26]
    OUT = {"meta": dict(n_long_2026=int(o26.sum()),
                        n_bad_2026=int((d26.coh == "BAD").sum()), n_good_2026=int((d26.coh == "GOOD").sum()),
                        n_other_2026=int((d26.coh == "OTHER").sum()),
                        clean_edge_note="no distinct V10 clean_edge head; proxied by tradable_prob/path_quality_pred")}
    print("=" * 92)
    print(f"2026 LONG cohort: total={int(o26.sum())}  BAD={OUT['meta']['n_bad_2026']}  "
          f"GOOD={OUT['meta']['n_good_2026']}  OTHER(excluded)={OUT['meta']['n_other_2026']}")
    cohort26 = d26[d26.coh.isin(["BAD", "GOOD"])].copy()
    cols = feat_cols(d)

    # ── Del 1-3: group means BAD vs GOOD (V10 heads summary + direction + regime) ──
    print("\n-- BAD vs GOOD group means (key signals; Δ = BAD−GOOD) --")
    keysig = [c for c in ["p_long", "p_short", "margin", "raw_adv", "p_hat", "tradable_prob",
                          "path_quality_pred", "uncertainty_score", "hold_horizon_pred",
                          "pos_vs_ema200_canon_v1", "_v1_ema_diff_canon_v1", "atr_pctl", "trend_str",
                          "d1_ema_slope_20_canon_v2_canon_v1", "_v1h4_slope5_canon_v1", "accel_h4",
                          "range_pos_proxy", "atr_z_canon_v1", "_v1_bb_squeeze_20_2_canon_v1"] if c in cohort26.columns]
    gm = cohort26.groupby("coh")[keysig].mean()
    means = {}
    for c in keysig:
        b, g = float(gm.loc["BAD", c]), float(gm.loc["GOOD", c])
        means[c] = dict(BAD=round(b, 4), GOOD=round(g, 4), delta=round(b - g, 4))
        print(f"   {c:34s} BAD={b:+9.4f}  GOOD={g:+9.4f}  Δ={b-g:+9.4f}")
    OUT["del1to3_group_means"] = means
    # also V10 dip/timing/tail/forecast argmax distribution
    for blk, pref in [("dip", "v10_dip_"), ("timing", "v10_timing_"), ("tail_risk", "v10_tail_risk_"), ("forecast", "v10_forecast_")]:
        bc = [c for c in cohort26.columns if c.startswith(pref)]
        if bc:
            am = cohort26[bc].values.argmax(1)
            cohort26[f"{blk}_argmax"] = am
            ct = cohort26.groupby("coh")[f"{blk}_argmax"].mean()
            print(f"   V10 {blk:10s} argmax mean: BAD={ct.get('BAD',float('nan')):.2f} GOOD={ct.get('GOOD',float('nan')):.2f}")

    # ── Del 4: top-20 separating (univariate AUC on the 2026 cohort) ──
    top = univ_auc(cohort26, cols)[:20]
    OUT["del4_top20_separating"] = [{"feat": f, "auc": a, "sep": s, "delta_bad_minus_good": dlt} for f, a, s, dlt in top]
    print("\n-- DEL4 TOP-20 SEPARATING (univariate AUC vs BAD; sep=|AUC-.5|) --")
    for f, a, s, dlt in top:
        print(f"   {f:34s} AUC={a:.3f} sep={s:.3f} Δ(BAD−GOOD)={dlt:+.4f}")

    # ── Del 5: classifier — within-2026 5-fold CV + pre2026->2026 OOT ──
    print("\n-- DEL5 CLASSIFIER (BAD vs GOOD, pre-entry only) --")
    # within-2026 CV
    Xc = cohort26[cols].fillna(cohort26[cols].median())
    yc = (cohort26["coh"] == "BAD").astype(int).values
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    cvout = {}
    for nm, clf in [("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))),
                    ("gbm", GradientBoostingClassifier(max_depth=3, n_estimators=150, subsample=0.7, random_state=0))]:
        pr = cross_val_predict(clf, Xc.values, yc, cv=cv, method="predict_proba")[:, 1]
        thr = np.quantile(pr, 0.90)
        cvout[nm] = dict(auc=round(float(roc_auc_score(yc, pr)), 4), prauc=round(float(average_precision_score(yc, pr)), 4),
                         base_rate=round(float(yc.mean()), 4), top_decile_precision=round(float(yc[pr >= thr].mean()), 4))
    OUT["del5_within2026_cv"] = cvout
    print(f"  [within-2026 5fold-CV] base_rate(BAD)={cvout['logreg']['base_rate']}  "
          f"logreg AUC={cvout['logreg']['auc']} PR={cvout['logreg']['prauc']} top10%prec={cvout['logreg']['top_decile_precision']}  | "
          f"gbm AUC={cvout['gbm']['auc']} top10%prec={cvout['gbm']['top_decile_precision']}")
    # strict OOT: train pre-2026 longs -> test 2026 longs
    cohpre = d[pre & d.coh.isin(["BAD", "GOOD"])].copy()
    OUT["del5_oot_pre2026_to_2026"] = classify(cohpre, cohort26, cols, "OOT pre2026->2026") if len(cohpre) > 300 else {"note": "insufficient pre-2026 cohort"}

    best_auc = max(cvout["logreg"]["auc"], cvout["gbm"]["auc"])
    OUT["VERDICT"] = dict(best_within2026_cv_auc=best_auc,
                          separable_above_0_60=bool(best_auc > 0.60),
                          conclusion=("BAD_LONGS separable at entry (AUC>0.60) — selection filter possible"
                                      if best_auc > 0.60 else
                                      "BAD_LONGS NOT separable from GOOD at entry (AUC<=0.60) — problem is EXIT/TIMING, not selection"))
    print(f"\n{'='*92}\nVERDICT: best within-2026 CV AUC = {best_auc}  -> {OUT['VERDICT']['conclusion']}\n{'='*92}")
    with open("/tmp/bad_long_separability_20260623.json", "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print("→ /tmp/bad_long_separability_20260623.json")


if __name__ == "__main__":
    main()
