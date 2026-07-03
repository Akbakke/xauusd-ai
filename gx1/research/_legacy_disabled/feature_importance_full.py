"""FULL feature importance (user vedtak 2026-06-23): what does the ENTRY chain ACTUALLY weight?
OOT permutation-importance (the dip-struct pre-gate's method) of the entry-IQL state vector on the
load-bearing DIRECTION outcome (long_pnl>short_pnl @K96), strict OOT (fit early→test late), by FAMILY
and top-N. Run TWICE: (1) ALL features, (2) RAW-only (exclude the model's OWN outputs p_long/V10-heads/
conviction — the 'fitted self-signal' leak suspects) so we see what the raw market features carry.
Plus XGB-bridge gain importance. READ-ONLY; no cement touched."""
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

FO_DIR = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
ENTRY_SUMMARY = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/entry_iql_volbal_20260611/summary_v1.json"
RNG = np.random.default_rng(20260623)
SAMPLE_CAP = 200_000
K = "K96"

# model's OWN outputs (fitted self-signal — exclude in the RAW-only run)
SELF_PREFIX = ("p_long", "p_short", "p_flat", "p_hat", "margin", "uncertainty", "tradable_prob",
               "direction_logit", "v10_", "bad_path_prob", "path_quality", "mfe_first_n", "hold_horizon")


def family(c):
    cl = c.lower()
    if any(c.startswith(p) or p in cl for p in SELF_PREFIX):
        return "MODEL_self_output(V10/conv)"
    if cl.startswith("session") or cl.startswith("is_") or "hour" in cl or "dow" in cl or "weekday" in cl:
        return "session/time"
    if cl.startswith("vol_regime") or cl.startswith("trend_regime") or "regime" in cl:
        return "regime"
    if cl.startswith("smc_") or "swing" in cl or "_v3" in cl or "dip_" in cl or "struct_" in cl:
        return "SMC/structure"
    if cl.startswith("dist_to_r") or cl.startswith("dist_to_s") or "_hi_atr" in cl or "_lo_atr" in cl or "pdh" in cl or "pdl" in cl:
        return "pivots/S-R"
    if "vwap" in cl:
        return "vwap"
    if "vol_z" in cl or "vol_ratio" in cl or "vol_pct" in cl or "signed_vol" in cl:
        return "volume"
    if any(k in cl for k in ("h1_", "h4_", "d1_", "m15_", "_v1h1", "_v1h4")):
        return "multi-TF(H1/H4/D1/M15)"
    return "M5 price/trend"


def main():
    names = json.load(open(ENTRY_SUMMARY))["feature_names_v1"]
    files = sorted(glob.glob(f"{FO_DIR}/*.parquet"))
    avail = set(pd.read_parquet(files[0]).columns)
    feats = [n for n in names if n in avail]
    lab = [f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1"]
    df = pd.concat([pd.read_parquet(f, columns=sorted(set(feats + lab + ["decision_ts_utc"]))) for f in files],
                   ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    lp = df[lab[0]].astype(float).values
    sp = df[lab[1]].astype(float).values
    ok = ~(np.isnan(lp) | np.isnan(sp))
    df = df[ok].reset_index(drop=True)
    y = (df[lab[0]].astype(float).values > df[lab[1]].astype(float).values).astype(int)
    df = df.sort_values("ts").reset_index(drop=True)
    med = df["ts"].median()
    early = (df["ts"] <= med).values
    late = (df["ts"] > med).values
    y = (df[lab[0]].astype(float).values > df[lab[1]].astype(float).values).astype(int)
    print(f"entry features covered: {len(feats)}/{len(names)}  rows={len(df)}  split@{med}  dir base={y.mean():.3f}")

    def run_pi(cols, tag):
        Xtr = df.loc[early, cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        ytr = y[early]
        if len(Xtr) > SAMPLE_CAP:
            s = RNG.choice(len(Xtr), SAMPLE_CAP, replace=False); Xtr = Xtr.iloc[s]; ytr = ytr[s]
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                             l2_regularization=1.0, min_samples_leaf=200,
                                             early_stopping=True, validation_fraction=0.15, random_state=42)
        clf.fit(Xtr, ytr)
        Xte = df.loc[late, cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        yte = y[late]
        nte = min(len(Xte), 40000)
        sel = RNG.choice(len(Xte), nte, replace=False)
        auc = roc_auc_score(yte[sel], clf.predict_proba(Xte.iloc[sel])[:, 1])
        pi = permutation_importance(clf, Xte.iloc[sel], yte[sel], n_repeats=4, random_state=0,
                                    scoring="roc_auc", n_jobs=3)
        imp = sorted(zip(cols, pi.importances_mean), key=lambda x: -x[1])
        # aggregate by family
        fam = {}
        for c, v in imp:
            fam.setdefault(family(c), 0.0)
            fam[family(c)] += max(v, 0.0)
        print(f"\n===== {tag}  (OOT AUC={auc:.3f}, n_feat={len(cols)}) =====")
        print("  BY FAMILY (sum of positive OOT-PI):")
        for f, v in sorted(fam.items(), key=lambda x: -x[1]):
            print(f"    {f:34s} {v:+.4f}")
        print("  TOP 20 individual features (OOT permutation-AUC importance):")
        for c, v in imp[:20]:
            print(f"    {c:46s} {v:+.5f}  [{family(c)}]")
        return dict(auc=round(auc, 4), top=[(c, round(float(v), 5)) for c, v in imp[:25]],
                    by_family={f: round(v, 5) for f, v in fam.items()})

    out = {}
    out["ALL"] = run_pi(feats, "ALL entry features")
    raw = [c for c in feats if family(c) != "MODEL_self_output(V10/conv)"]
    out["RAW_only"] = run_pi(raw, "RAW market features ONLY (exclude V10/conviction self-output)")

    # ── XGB bridge gain importance ────────────────────────────────────────────
    print("\n===== XGB-BRIDGE gain importance =====")
    try:
        from gx1.runtime import gx1_guards  # noqa
    except Exception:
        pass
    try:
        import joblib
        xgb_dir = json.load(open("PROJECT_STATE_artifacts.json"))["active"]["xgb"]["path"]
        # try to locate the booster(s)
        found = False
        for pat in ("**/*.json", "**/*.ubj", "**/*.pkl", "**/*.joblib", "**/*.model"):
            for mp in glob.glob(f"{xgb_dir}/{pat}", recursive=True):
                if any(k in mp.lower() for k in ("booster", "xgb", "model")) and not mp.endswith("summary_v1.json"):
                    try:
                        import xgboost as xgb
                        bst = xgb.Booster(); bst.load_model(mp)
                        g = bst.get_score(importance_type="gain")
                        top = sorted(g.items(), key=lambda x: -x[1])[:20]
                        print(f"  [{Path(mp).name}] top gain features:")
                        for fn, gv in top:
                            print(f"    {fn:40s} {gv:9.1f}")
                        found = True
                        break
                    except Exception as e:
                        continue
            if found:
                break
        if not found:
            print("  (XGB booster not directly loadable as a Booster — multihead custom format; "
                  "gain available via the trainer's saved importance if present)")
            for j in glob.glob(f"{xgb_dir}/**/*importance*", recursive=True)[:3]:
                print("   importance artifact:", j)
    except Exception as e:
        print("  XGB gain failed:", e)

    Path("/tmp/feature_importance_full.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nWROTE /tmp/feature_importance_full.json")


if __name__ == "__main__":
    main()
