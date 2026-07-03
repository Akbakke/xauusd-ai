"""CATASTROPHIC-PATH AUDIT (user vedtak 2026-06-23): are C/D trades a separate setup-family
identifiable BEFORE entry, or just the tail of one distribution? 6 parts, OOT-honest.

RAM-SAFE: single sequential process, bounded sampling (t-SNE on 25k). umap/hdbscan/shap NOT installed
→ sklearn equivalents (t-SNE + KMeans/GMM/DBSCAN for clustering; permutation-importance + XGB/LGBM gain
instead of SHAP) — stated honestly. Base = taken-proxy (top-20% margin, side=argmax) on forward_outcome
(held-to-K96 ENTRY path, pre-exit). Direction-AUC reference ≈ 0.62. NO cement touched.

Labels: Label_150 = MAE>=150bps ; Label_300 = MAE>=300bps  (the C/D catastrophe bands).
"""
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

FO = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
K = "K96"
RNG = np.random.default_rng(20260623)
OUT = {}

PATH = ["bad_path_prob", "path_quality_pred", "mfe_first_n_pred", "tradable_prob"]
TAIL = [f"v10_tail_risk_{i}" for i in range(6)]
TIMING = [f"v10_timing_{i}" for i in range(12)]
FORE = [f"v10_forecast_{i}" for i in range(4)]
DIP = [f"v10_dip_{i}" for i in range(18)]
DIRC = ["direction_logit_long", "direction_logit_short", "direction_logit_flat", "p_long", "p_short",
        "margin", "uncertainty_score"]
ATR = ["atr_bps", "atr_canon_v1", "atr50_canon_v1", "atr_z_canon_v1", "_v1_atr14_canon_v1",
       "_v1h1_atr_canon_v1", "_v1h4_atr_canon_v1", "rvol_20_canon_v1", "rvol_60_canon_v1",
       "_v1_pk_sigma20_canon_v1", "vol_ratio_canon_v1"]
REGIME = ["vol_regime__LOW", "vol_regime__MEDIUM", "vol_regime__HIGH", "vol_regime__EXTREME",
          "trend_regime__TREND_UP", "trend_regime__TREND_NEUTRAL", "trend_regime__TREND_DOWN"]
SESSION = ["session__ASIA", "session__EU", "session__OVERLAP", "session__US", "hour_sin_canon_v1", "hour_cos_canon_v1"]
ENTRY_FEATS = PATH + TAIL + TIMING + FORE + DIP + DIRC + ATR + REGIME + SESSION


def load():
    files = sorted(glob.glob(f"{FO}/*.parquet"))
    av = set(pd.read_parquet(files[0]).columns)
    lab = [f"take_now_{s}_{m}_at_{K}_v1" for s in ("long", "short") for m in ("terminal_pnl", "mae_bps", "mfe_bps")]
    feats = [c for c in ENTRY_FEATS if c in av]
    df = pd.concat([pd.read_parquet(f, columns=[c for c in (feats + lab + ["decision_ts_utc"]) if c in av])
                    for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    long = (df["p_long"] >= df["p_short"]).values
    for m in ("terminal_pnl", "mae_bps", "mfe_bps"):
        df[m] = np.where(long, df[f"take_now_long_{m}_at_{K}_v1"].astype(float), df[f"take_now_short_{m}_at_{K}_v1"].astype(float))
    df = df.dropna(subset=["terminal_pnl", "mae_bps", "p_long"]).reset_index(drop=True)
    thr = np.nanpercentile(df["margin"].astype(float).values, 80)
    df = df[df["margin"].astype(float) >= thr].reset_index(drop=True)  # taken-proxy
    return df, feats


def Xy(df, feats):
    X = df[feats].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    return X


def main():
    df, feats = load()
    mae = df["mae_bps"].astype(float).values
    pnl = df["terminal_pnl"].astype(float).values
    y150 = (mae >= 150).astype(int)
    y300 = (mae >= 300).astype(int)
    ts = df["ts"].values
    med = df["ts"].median()
    early = (df["ts"] <= med).values
    late = (df["ts"] > med).values
    oot26 = (df["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")).values
    print(f"taken-proxy n={len(df)} feats={len(feats)} | Label_150 base={y150.mean():.4f} | Label_300 base={y300.mean():.4f}")
    print(f"deps: umap/hdbscan/shap ABSENT -> sklearn t-SNE+KMeans/GMM/DBSCAN + permutation-importance/gain\n")
    X = Xy(df, feats)
    Xs = StandardScaler().fit_transform(X)

    # ── PART 1: clustering (t-SNE viz proxy + KMeans/GMM/DBSCAN on standardized entry space) ──
    print("=" * 64 + "\nPART 1 — CLUSTERING: are C/D over-represented in some cluster?\n" + "=" * 64)
    band = np.where(mae < 50, "A", np.where(mae < 150, "B", np.where(mae < 300, "C", "D")))
    cd = (mae >= 150)
    nsamp = min(25000, len(df))
    si = RNG.choice(len(df), nsamp, replace=False)
    km = KMeans(n_clusters=10, n_init=4, random_state=0).fit_predict(Xs[si])
    gm = GaussianMixture(n_components=10, covariance_type="diag", random_state=0, max_iter=80).fit(Xs[si]).predict(Xs[si])
    cd_s = cd[si]
    p1 = {}
    for nm, lab in (("KMeans-10", km), ("GMM-10", gm)):
        rows = []
        for c in np.unique(lab):
            m = lab == c
            rows.append((int(c), int(m.sum()), float(cd_s[m].mean())))
        rows.sort(key=lambda x: -x[2])
        base = cd_s.mean()
        print(f"  [{nm}] base C+D rate={base:.3f}. Clusters by C+D-rate (size, C+D%):")
        for c, n, r in rows[:6]:
            print(f"    cluster {c:2d}: n={n:5d}  C+D={r*100:5.1f}%  (lift {r/base:.2f}x)")
        topr = rows[0]
        p1[nm] = dict(base=round(base, 4), top_cluster_cdrate=round(topr[2], 4), top_lift=round(topr[2]/base, 2),
                      max_cluster_cdrate=round(max(r for _, _, r in rows), 4))
    OUT["part1_clusters"] = p1

    # ── PART 2: catastrophic classifier (XGB / LGBM / LR), strict OOT + 2026 ──
    print("\n" + "=" * 64 + "\nPART 2 — CATASTROPHIC CLASSIFIER (fit EARLY → test LATE & 2026)\n" + "=" * 64)
    from sklearn.metrics import roc_curve
    pre26 = ~oot26

    def fit_eval(y, name):
        res = {}
        # robust OOT arms: chrono early->late (only if early has both classes) + pre2026->2026 (true forward)
        arms = []
        if len(np.unique(y[early])) > 1 and len(np.unique(y[late])) > 1:
            arms.append(("early", "late", early, late))
        if len(np.unique(y[pre26])) > 1 and len(np.unique(y[oot26])) > 1:
            arms.append(("pre2026", "2026", pre26, oot26))
        if not arms:
            print(f"  [{name}] no valid OOT arm (too rare for any split)"); return res
        for trn, tsn, trm, tem in arms:
            spw = float((y[trm] == 0).sum() / max(1, (y[trm] == 1).sum()))
            for mdl, tag in ((xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                                 colsample_bytree=0.8, eval_metric="logloss", n_jobs=4,
                                                 scale_pos_weight=spw), "XGB"),
                             (lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                                 colsample_bytree=0.8, n_jobs=4, verbose=-1, class_weight="balanced"), "LGBM"),
                             (LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5), "LR")):
                Xt = (Xs if tag == "LR" else X)[trm]
                Xv = (Xs if tag == "LR" else X)[tem]
                try:
                    mdl.fit(Xt, y[trm])
                except Exception as e:
                    print(f"  [{name} {tag} {trn}->{tsn}] fit failed: {e}"); continue
                p = mdl.predict_proba(Xv)[:, 1]
                yv = y[tem]
                auc = roc_auc_score(yv, p); pr = average_precision_score(yv, p)
                fpr, tpr, _ = roc_curve(yv, p)
                rec = {f"recall@FPR{int(f*100)}%": round(float(np.interp(f, fpr, tpr)), 3) for f in (0.01, 0.05, 0.10)}
                res[f"{tag}_{trn}->{tsn}"] = dict(auc=round(auc, 4), pr_auc=round(pr, 4), base=round(yv.mean(), 4),
                                                  brier=round(brier_score_loss(yv, p), 4), **rec)
                print(f"  [{name} {tag} {trn}->{tsn}] AUC={auc:.3f} PR={pr:.3f} (base {yv.mean():.3f}) "
                      f"rec@1%={rec['recall@FPR1%']} @5%={rec['recall@FPR5%']} @10%={rec['recall@FPR10%']}")
        return res
    OUT["part2_L150"] = fit_eval(y150, "L150")
    OUT["part2_L300"] = fit_eval(y300, "L300")

    # ── PART 3: feature importance (PI + XGB gain) for catastrophe ──
    print("\n" + "=" * 64 + "\nPART 3 — FEATURE IMPORTANCE for catastrophe (PI on LATE; XGB gain)\n" + "=" * 64)
    def fam(c):
        if c in TAIL: return "tail_risk"
        if c in DIP: return "dip"
        if c in TIMING: return "timing"
        if c in FORE: return "forecast"
        if c in PATH: return "path_head"
        if c in DIRC: return "direction/conv"
        if c in ATR: return "ATR/vol"
        if c in REGIME: return "regime"
        if c in SESSION: return "session"
        return "other"
    for y, nm in ((y150, "L150"), (y300, "L300")):
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=250, learning_rate=0.05,
                                             min_samples_leaf=100, early_stopping=True, random_state=42).fit(X[early], y[early])
        nte = min(40000, late.sum()); s = RNG.choice(np.where(late)[0], nte, replace=False)
        pi = permutation_importance(clf, X[s][:, [feats.index(f) for f in feats]], y[s], n_repeats=4,
                                    random_state=0, scoring="roc_auc", n_jobs=3)
        imp = sorted(zip(feats, pi.importances_mean), key=lambda x: -x[1])
        famsum = {}
        for f, v in imp:
            famsum[fam(f)] = famsum.get(fam(f), 0.0) + max(v, 0.0)
        print(f"  [{nm}] BY FAMILY (sum positive OOT-PI): " +
              " ".join(f"{k}={v:.3f}" for k, v in sorted(famsum.items(), key=lambda x: -x[1])))
        print(f"  [{nm}] TOP 15: " + ", ".join(f"{f}({v:+.4f})" for f, v in imp[:15]))
        OUT[f"part3_{nm}_top"] = [(f, round(float(v), 5)) for f, v in imp[:50]]
        OUT[f"part3_{nm}_family"] = {k: round(v, 5) for k, v in famsum.items()}

    # ── PART 4: discrete or continuous? KS/JS/Wasserstein A+B vs C+D ──
    print("\n" + "=" * 64 + "\nPART 4 — TWO DISTRIBUTIONS or ONE TAIL? (A+B vs C+D)\n" + "=" * 64)
    def js(a, b, bins=60):
        lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
        ha, _ = np.histogram(a, bins=bins, range=(lo, hi), density=True)
        hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
        ha = ha/ha.sum()+1e-12; hb = hb/hb.sum()+1e-12; m = 0.5*(ha+hb)
        kl = lambda p, q: np.sum(p*np.log(p/q))
        return float(0.5*kl(ha, m)+0.5*kl(hb, m))
    tail_mag = np.abs(df[TAIL].astype(float).values).mean(axis=1) if all(t in df.columns for t in TAIL) else df["bad_path_prob"].astype(float).values
    feats_test = {"MAE": mae, "tail_risk_mag": tail_mag, "margin": df["margin"].astype(float).values,
                  "p_long": df["p_long"].astype(float).values, "p_short": df["p_short"].astype(float).values,
                  "ATR_bps": df["atr_bps"].astype(float).values if "atr_bps" in df.columns else df["_v1_atr14_canon_v1"].astype(float).values}
    p4 = {}
    for nm, v in feats_test.items():
        a = v[~cd]; b = v[cd]
        ksst = ks_2samp(a, b).statistic; jsd = js(a, b); wd = wasserstein_distance(a, b)
        p4[nm] = dict(ks=round(ksst, 3), js=round(jsd, 3), wasserstein=round(float(wd), 2))
        print(f"  {nm:14s} KS={ksst:.3f}  JS={jsd:.3f}  Wasserstein={wd:.2f}")
    OUT["part4"] = p4
    # bimodality on MAE: GMM 2-comp vs 1-comp BIC
    m2 = mae.reshape(-1, 1)
    bic1 = GaussianMixture(1, random_state=0).fit(m2).bic(m2)
    bic2 = GaussianMixture(2, random_state=0).fit(m2).bic(m2)
    print(f"  MAE GMM BIC: 1-comp={bic1:.0f}  2-comp={bic2:.0f}  -> {'2-COMP (bimodal-ish)' if bic2 < bic1 else '1-COMP (continuous tail)'}")
    OUT["part4_bic"] = dict(bic1=round(bic1), bic2=round(bic2), favors=("2comp" if bic2 < bic1 else "1comp"))

    # ── PART 5: counterfactual portfolio (drop worst-X% by catastrophe score), 3 splits ──
    print("\n" + "=" * 64 + "\nPART 5 — COUNTERFACTUAL PORTFOLIO (drop worst-X% by L150 score)\n" + "=" * 64)
    cls = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", n_jobs=4,
                            scale_pos_weight=float((y150[early] == 0).sum()/max(1, (y150[early] == 1).sum()))).fit(X[early], y150[early])
    def port(mask, name):
        idx = np.where(mask)[0]
        if len(idx) < 50:
            print(f"  [{name}] n<50 skip"); return
        score = cls.predict_proba(X[idx])[:, 1]
        pv = pnl[idx]; tv = ts[idx]
        order = np.argsort(tv)
        def stat(keep):
            v = pv[keep]
            if len(v) == 0: return None
            cum = np.cumsum(v[np.argsort(tv[keep])])
            mdd = float((np.maximum.accumulate(cum)-cum).max())
            w = v[v > 0].sum(); l = -v[v < 0].sum()
            return dict(n=len(v), total=round(float(v.sum())), mean=round(float(v.mean()), 2),
                        sharpe=round(float(v.mean()/(v.std()+1e-9)), 3), pf=round(float(w/(l+1e-9)), 2), mdd=round(mdd))
        base = stat(np.ones(len(idx), bool))
        print(f"  [{name}] base: {base}")
        for frac in (0.01, 0.02, 0.05, 0.10, 0.15):
            cut = np.nanpercentile(score, 100*(1-frac))
            st = stat(score < cut)
            print(f"     drop {int(frac*100):2d}%: {st}")
            OUT.setdefault(f"part5_{name}", {})[f"drop{int(frac*100)}"] = st
        OUT.setdefault(f"part5_{name}", {})["base"] = base
    port(late, "OOT-late")
    port(oot26, "2026")
    # NOTE: realized-replay through the actual Exit-IQL stack = the phase6 gate (heavy, separate) — NOT run here.
    print("  NOTE: realized-replay through the live Exit-stack = phase6 gate (heavy/separate) — flagged, not inline.")

    # ── PART 6: tail_risk vs ATR (is it just volatility?) ──
    print("\n" + "=" * 64 + "\nPART 6 — IS tail_risk JUST VOLATILITY? (partial out ATR)\n" + "=" * 64)
    from sklearn.linear_model import LinearRegression
    atr_cols = [c for c in ATR if c in df.columns]
    A = df[atr_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    lr = LinearRegression().fit(A, tail_mag)
    r2 = lr.score(A, tail_mag)
    resid = tail_mag - lr.predict(A)  # tail_risk with ATR removed
    aucs = {}
    for nm, v in (("ATR_only", A), ("tail_risk_only", tail_mag.reshape(-1, 1)), ("tail_risk_resid(ATR-out)", resid.reshape(-1, 1))):
        c = HistGradientBoostingClassifier(max_depth=3, max_iter=150, min_samples_leaf=100, early_stopping=True,
                                           random_state=0).fit(v[early], y150[early])
        a = roc_auc_score(y150[late], c.predict_proba(v[late])[:, 1])
        aucs[nm] = round(a, 3)
    print(f"  tail_risk ~ ATR  R²={r2:.3f}  (1.0 = tail_risk fully explained by ATR)")
    print(f"  L150 OOT-AUC: ATR_only={aucs['ATR_only']}  tail_risk_only={aucs['tail_risk_only']}  "
          f"tail_risk_resid(ATR-removed)={aucs['tail_risk_resid(ATR-out)']}")
    OUT["part6_tailrisk_vs_atr"] = dict(r2=round(r2, 3), **aucs)

    Path("/tmp/cpath_audit.json").write_text(json.dumps(OUT, indent=2, default=str))
    print("\nWROTE /tmp/cpath_audit.json")


if __name__ == "__main__":
    main()
