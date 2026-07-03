"""ADVERSARIAL REFUTATION of the mtf_disambig OOT-separability claim.

Claim under attack: a buy-the-dip-LONG-win target in the dip-in-uptrend population is
OOT-SEPARABLE (AUC 0.7018 / 0.8642) using CURRENT candidate/context features + multi-TF
trend-disambiguation interactions. Top feats: p_short, _v1h1_ema_diff, x_mag_h1_h4,
d1_close_pct_in_20day_range, _v1h4_ema_diff, d1_rsi14, p_long, ema100_slope.

We reuse the EXACT population + target convention from oot_pregate_dip_reversal_structural.py
(the refuted-trough pre-gate) so this is apples-to-apples with the load-bearing label.

Attacks:
  (0) REPRODUCE the claimed 2-split result with the disambig feature set (GBM, seed 42).
  (1) 3rd MIDDLE-OUT time split + different seed (1234) + different model (LogReg + RF).
  (2) LEAKAGE hunt: drop the top OOT feature, re-test; forbidden-token assert on every feat.
  (3) EMBARGO: purge K96 (= 96 M5 bars = 8h) horizon around the split boundary on BOTH sides
      to kill label-window autocorrelation bleed.
  (4) LOAD-BEARING high-conviction tail: AUC on the top-conviction subset only (margin tail),
      not blanket AUC.

Gate (same as pre-gate): separable iff BOTH directional OOT AUC >= 0.58, AND survives embargo,
AND holds on the conviction tail. Default verdict = REFUTED if fragile on any leg.
"""
import glob, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
FO_DIR = ("/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
          "forward_outcome_clean/per_week")
K = "K96"
SAMPLE_CAP = 250_000

# raw cols we read (no forward/label leak into features)
TS = "decision_ts_utc"
RAW = [
    TS, "side", "trend_regime",
    "p_long", "p_short", "margin", "uncertainty_score",
    "_v1_close_ema_slope_3_canon_v1", "_v1_ema_diff_canon_v1",
    "m15_trend_sign_canon_v2_canon_v1",
    "_v1h1_ema_diff_canon_v1", "_v1h1_slope3_canon_v1",
    "_v1h4_ema_diff_canon_v1", "_v1h4_slope3_canon_v1",
    "d1_ema_slope_20_canon_v2_canon_v1",
    "d1_close_pct_in_20day_range_canon_v2_canon_v1", "d1_rsi14_canon_v2_canon_v1",
    "ema20_slope_canon_v1", "ema100_slope_canon_v1", "pos_vs_ema200_canon_v1",
    "_v1_r5_canon_v1", "_v1_rsi14_canon_v1",
]
LABEL = [f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1"]


def load():
    files = sorted(glob.glob(f"{FO_DIR}/*.parquet"))
    need = sorted(set(RAW + LABEL))
    parts = [pd.read_parquet(f, columns=need) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df[TS], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


def build_disambig(df):
    """Construct the mtf_disambig feature set described in the claim."""
    # per-TF slope sign proxies (the chain's real per-TF trend cols)
    s_m5 = df["_v1_close_ema_slope_3_canon_v1"].astype(float)
    s_m15 = df["m15_trend_sign_canon_v2_canon_v1"].astype(float)
    s_h1 = df["_v1h1_ema_diff_canon_v1"].astype(float)
    s_h4 = df["_v1h4_ema_diff_canon_v1"].astype(float)
    s_d1 = df["d1_ema_slope_20_canon_v2_canon_v1"].astype(float)
    sg = lambda x: np.sign(x.fillna(0.0))
    sm5, sm15, sh1, sh4, sd1 = sg(s_m5), sg(s_m15), sg(s_h1), sg(s_h4), sg(s_d1)

    f = pd.DataFrame(index=df.index)
    # raw current candidate/context features (claim says "CURRENT ... features")
    for c in ["p_long", "p_short", "margin", "uncertainty_score",
              "_v1h1_ema_diff_canon_v1", "_v1h4_ema_diff_canon_v1",
              "d1_close_pct_in_20day_range_canon_v2_canon_v1", "d1_rsi14_canon_v2_canon_v1",
              "ema20_slope_canon_v1", "ema100_slope_canon_v1", "pos_vs_ema200_canon_v1",
              "_v1_rsi14_canon_v1", "_v1_r5_canon_v1", "_v1h1_slope3_canon_v1",
              "_v1h4_slope3_canon_v1", "d1_ema_slope_20_canon_v2_canon_v1"]:
        f[c] = df[c].astype(float)
    # cross-TF ema-slope SIGN PRODUCTS
    f["sgn_m5_m15"] = sm5 * sm15
    f["sgn_m15_h1"] = sm15 * sh1
    f["sgn_h1_h4"] = sh1 * sh4
    f["sgn_h4_d1"] = sh4 * sd1
    f["sgn_m5_d1"] = sm5 * sd1
    # n-TFs-up count
    n_up = ((sm5 > 0).astype(int) + (sm15 > 0).astype(int) + (sh1 > 0).astype(int)
            + (sh4 > 0).astype(int) + (sd1 > 0).astype(int))
    f["n_tfs_up"] = n_up
    # all-HTF-up flag & D1up-vs-(H1/H4 down) conflict flags
    f["all_htf_up"] = ((sh1 > 0) & (sh4 > 0) & (sd1 > 0)).astype(int)
    f["d1up_h1h4down_conflict"] = ((sd1 > 0) & (sh1 < 0) & (sh4 < 0)).astype(int)
    f["d1up_h1down_conflict"] = ((sd1 > 0) & (sh1 < 0)).astype(int)
    # slope-MAGNITUDE products  (x_mag_h1_h4 is the claim's named top feature)
    f["x_mag_h1_h4"] = s_h1.fillna(0.0) * s_h4.fillna(0.0)
    f["x_mag_h4_d1"] = s_h4.fillna(0.0) * s_d1.fillna(0.0)
    # pos_vs_ema200 x slope products
    pe = df["pos_vs_ema200_canon_v1"].astype(float).fillna(0.0)
    f["pe200_x_ema100slope"] = pe * df["ema100_slope_canon_v1"].astype(float).fillna(0.0)
    f["pe200_x_h1slope"] = pe * s_h1.fillna(0.0)
    # d1-range-pos x n-up
    f["d1range_x_nup"] = df["d1_close_pct_in_20day_range_canon_v2_canon_v1"].astype(float).fillna(0.0) * n_up
    return f


def dip_uptrend_target(df):
    uptrend = ((df["trend_regime"] == "TREND_UP") & (df["pos_vs_ema200_canon_v1"] > 0)
               & (df["ema100_slope_canon_v1"] > 0))
    dip = (df["_v1_r5_canon_v1"] < 0) | (df["_v1_rsi14_canon_v1"] < 45)
    pop = (uptrend & dip).fillna(False)
    sub = df[pop].reset_index(drop=True)
    lp = sub[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float)
    sp = sub[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float)
    y = ((lp > 0) | (lp > sp)).astype(int)
    keep = sub[f"take_now_long_terminal_pnl_at_{K}_v1"].notna()
    return sub[keep].reset_index(drop=True), y[keep].reset_index(drop=True)


def _fit_auc(Xtr, ytr, Xte, yte, model, seed):
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(Xtr.replace([np.inf, -np.inf], np.nan))
    Xte = imp.transform(Xte.replace([np.inf, -np.inf], np.nan))
    if model == "gbm":
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                             l2_regularization=1.0, min_samples_leaf=200,
                                             early_stopping=True, validation_fraction=0.15,
                                             random_state=seed)
    elif model == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=200,
                                     n_jobs=4, random_state=seed)
    else:  # logreg
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    rng = np.random.default_rng(seed)
    if len(Xtr) > SAMPLE_CAP:
        sel = rng.choice(len(Xtr), SAMPLE_CAP, replace=False)
        Xtr, ytr = Xtr[sel], np.asarray(ytr)[sel]
    clf.fit(Xtr, ytr)
    if len(np.unique(yte)) < 2:
        return float("nan")
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def main():
    df = load()
    sub, y = dip_uptrend_target(df)
    F = build_disambig(sub)
    feat = list(F.columns)
    F["ts"] = sub["ts"].values
    F["_y"] = y.values
    F["margin_abs"] = sub["margin"].astype(float).abs().values
    print(f"population n={len(F)}  base_rate(LONG-win)={F['_y'].mean():.4f}  nfeat={len(feat)}")

    # LEAKAGE GUARD
    FORBIDDEN = ("terminal_pnl", "mfe", "mae", "forward", "time_to_mfe", "wait_",
                 "take_now_", "_outcome", "trough", "bars_to")
    for c in feat:
        assert not any(t in c for t in FORBIDDEN), f"LEAKAGE: {c}"
    print("leakage-guard: no forward token in any feature  OK")

    ts = F["ts"]
    out = {}

    # ── ATTACK 0: reproduce claimed 2-split (median, GBM seed 42) ───────────────
    med = ts.median()
    early, late = ts <= med, ts > med
    a0_el = _fit_auc(F.loc[early, feat], F.loc[early, "_y"], F.loc[late, feat], F.loc[late, "_y"], "gbm", 42)
    a0_le = _fit_auc(F.loc[late, feat], F.loc[late, "_y"], F.loc[early, feat], F.loc[early, "_y"], "gbm", 42)
    out["attack0_repro_2split_gbm42"] = dict(fit_early_test_late=round(a0_el, 4),
                                             fit_late_test_early=round(a0_le, 4), split=str(med))
    print(f"[A0 repro 2-split GBM42] EL={a0_el:.4f}  LE={a0_le:.4f}")

    # ── ATTACK 1: 3rd MIDDLE-OUT split + new seed + new models ──────────────────
    q1, q2 = ts.quantile(1/3), ts.quantile(2/3)
    mid = (ts > q1) & (ts <= q2)
    edges = ~mid
    res1 = {}
    for model in ("gbm", "rf", "logreg"):
        # middle-out: train edges, test middle  AND  train middle, test edges
        mo = _fit_auc(F.loc[edges, feat], F.loc[edges, "_y"], F.loc[mid, feat], F.loc[mid, "_y"], model, 1234)
        om = _fit_auc(F.loc[mid, feat], F.loc[mid, "_y"], F.loc[edges, feat], F.loc[edges, "_y"], model, 1234)
        # also reproduce 2-split with new seed+model
        el = _fit_auc(F.loc[early, feat], F.loc[early, "_y"], F.loc[late, feat], F.loc[late, "_y"], model, 1234)
        le = _fit_auc(F.loc[late, feat], F.loc[late, "_y"], F.loc[early, feat], F.loc[early, "_y"], model, 1234)
        res1[model] = dict(middle_out_train_edges=round(mo, 4), train_middle_test_edges=round(om, 4),
                           split2_EL=round(el, 4), split2_LE=round(le, 4))
        print(f"[A1 {model} seed1234] midout(train_edges->mid)={mo:.4f} (train_mid->edges)={om:.4f} | 2split EL={el:.4f} LE={le:.4f}")
    out["attack1_middleout_newseed_newmodel"] = res1

    # ── ATTACK 2: drop the top feature (p_short per claim) + leakage re-test ────
    for drop in ("p_short", "p_long", "x_mag_h1_h4"):
        ft2 = [c for c in feat if c != drop]
        el = _fit_auc(F.loc[early, ft2], F.loc[early, "_y"], F.loc[late, ft2], F.loc[late, "_y"], "gbm", 42)
        le = _fit_auc(F.loc[late, ft2], F.loc[late, "_y"], F.loc[early, ft2], F.loc[early, "_y"], "gbm", 42)
        out[f"attack2_drop_{drop}"] = dict(EL=round(el, 4), LE=round(le, 4))
        print(f"[A2 drop {drop}] EL={el:.4f}  LE={le:.4f}")

    # ── ATTACK 3: EMBARGO K96 (8h) around split boundary, both sides ────────────
    embargo = pd.Timedelta(hours=8)
    e_tr = ts <= (med - embargo)   # train early, purge last 8h
    e_te = ts > (med + embargo)    # test late, purge first 8h
    a3_el = _fit_auc(F.loc[e_tr, feat], F.loc[e_tr, "_y"], F.loc[e_te, feat], F.loc[e_te, "_y"], "gbm", 42)
    l_tr = ts > (med + embargo)
    l_te = ts <= (med - embargo)
    a3_le = _fit_auc(F.loc[l_tr, feat], F.loc[l_tr, "_y"], F.loc[l_te, feat], F.loc[l_te, "_y"], "gbm", 42)
    out["attack3_embargo8h"] = dict(EL=round(a3_el, 4), LE=round(a3_le, 4),
                                    purged=int((~(e_tr | e_te)).sum()))
    print(f"[A3 embargo8h] EL={a3_el:.4f}  LE={a3_le:.4f}")

    # ── ATTACK 4: LOAD-BEARING high-conviction tail (top-20% margin) ────────────
    thr = F["margin_abs"].quantile(0.80)
    tail = F["margin_abs"] >= thr
    # train on full early, evaluate ONLY on high-conviction late tail
    Xtr, ytr = F.loc[early, feat], F.loc[early, "_y"]
    te_mask = late & tail
    a4_el = _fit_auc(Xtr, ytr, F.loc[te_mask, feat], F.loc[te_mask, "_y"], "gbm", 42)
    Xtr2, ytr2 = F.loc[late, feat], F.loc[late, "_y"]
    te2 = early & tail
    a4_le = _fit_auc(Xtr2, ytr2, F.loc[te2, feat], F.loc[te2, "_y"], "gbm", 42)
    out["attack4_conviction_tail"] = dict(EL=round(a4_el, 4), LE=round(a4_le, 4),
                                          tail_n_late=int(te_mask.sum()), tail_n_early=int(te2.sum()),
                                          tail_base_rate_late=round(float(F.loc[te_mask, "_y"].mean()), 4))
    print(f"[A4 conviction-tail top20%] EL={a4_el:.4f}  LE={a4_le:.4f}  base_rate_late={F.loc[te_mask,'_y'].mean():.4f}")

    # verdict
    legs = []
    legs.append(min(a0_el, a0_le))                                   # repro
    legs.append(min(res1["gbm"]["middle_out_train_edges"], res1["gbm"]["train_middle_test_edges"]))
    legs.append(min(res1["logreg"]["split2_EL"], res1["logreg"]["split2_LE"]))
    legs.append(min(a3_el, a3_le))                                   # embargo
    legs.append(min(a4_el, a4_le))                                   # tail
    worst = float(np.nanmin(legs))
    survives = bool(worst >= 0.58 and a3_el >= 0.58 and a3_le >= 0.58
                    and min(a4_el, a4_le) >= 0.58)
    out["verdict"] = dict(survives=survives, worst_oot_auc_across_attacks=round(worst, 4),
                          gate="BOTH directional OOT >= 0.58 on every attack leg")
    print(f"\n=== VERDICT survives={survives}  worst_OOT_AUC={worst:.4f} ===")
    Path("/tmp/oot_refute_mtf_disambig.json").write_text(json.dumps(out, indent=2, default=str))
    print("WROTE /tmp/oot_refute_mtf_disambig.json")


if __name__ == "__main__":
    main()
