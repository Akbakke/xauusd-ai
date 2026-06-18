"""Meta-labeling pre-gate (2026-06-18, user vedtak): can a secondary model predict P(the bot's chosen-side
take WINS through exit) on the conviction-gated traded subset, well enough that PRUNING the predicted-losers
RAISES realized PnL/win OOT? Cheapest-first per AGENTS.md (don't build a filter on a label that doesn't
separate OOT; a fitted filter that drops profitable trades is the classic failure).

LOAD-BEARING target = REAL through-exit realized PnL (volbal baseline, the cemented IQL-natural takes),
NOT held-to-K96 forward outcome (which penalizes exit-managed trades). Strict 2026 holdout + robustness +
the economic prune test (does dropping the bottom-K% predicted-win actually improve OOT total/win, or does
it cut +EV trades?). Also compares the full meta-model to a conviction-only baseline (margin/uncertainty/
raw_adv = what the existing gate already uses) — incremental value is the question.
"""
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

BASELINE = ("/home/andre2/GX1_DATA/reports/v12_paper_runs/nightly_learning/candidate_gates/"
            "baselines/volbal_per_candidate_V12_OFF.csv")
DEC = ("/home/andre2/GX1_DATA/reports/v12_paper_runs/nightly_learning/candidate_gates/runs/"
       "entry_iql_volbal_20260611_20260615T180412Z/entry_decisions/"
       "ENTRY_IQL_INFERENCE_FOR_V12_20260615T180414Z/decisions.parquet")
FO_DIR = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
RNG = np.random.default_rng(42)

CONV = ["p_long", "p_short", "margin", "uncertainty_score", "p_hat", "tradable_prob",
        "path_quality_pred", "bad_path_prob"]
PRICE = ["ema20_slope_canon_v1", "ema100_slope_canon_v1", "pos_vs_ema200_canon_v1", "_v1_rsi14_canon_v1",
         "_v1_r5_canon_v1", "_v1_r24_canon_v1", "roc20_canon_v1", "atr_bps", "atr_z_canon_v1",
         "_v1h1_ema_diff_canon_v1", "_v1h4_ema_diff_canon_v1", "_v1h1_slope5_canon_v1", "_v1h4_slope5_canon_v1",
         "d1_ema_slope_20_canon_v2_canon_v1", "d1_close_pct_in_20day_range_canon_v2_canon_v1",
         "m15_rsi14_canon_v2_canon_v1"]
V10H = ([f"v10_dip_{i}" for i in range(18)] + [f"v10_forecast_{i}" for i in range(4)]
        + [f"v10_timing_{i}" for i in range(12)] + [f"v10_tail_risk_{i}" for i in range(2)]
        + ["direction_logit_long", "direction_logit_short"])


def main():
    b = pd.read_csv(BASELINE, usecols=["candidate_uid", "realized_pnl_bps", "side_v1"])
    d = pd.read_parquet(DEC, columns=["candidate_uid", "decision_ts_utc", "q_skip_v1",
                                      "q_take_long_v1", "q_take_short_v1"])
    d["ts"] = pd.to_datetime(d["decision_ts_utc"], utc=True)
    d["raw_adv"] = np.maximum(d["q_take_long_v1"].values, d["q_take_short_v1"].values) - d["q_skip_v1"].values
    m = b.merge(d[["candidate_uid", "ts", "raw_adv"]], on="candidate_uid", how="inner")
    foc = sorted(set(["decision_ts_utc"] + CONV + PRICE + V10H))
    files = glob.glob(f"{FO_DIR}/*.parquet")
    avail = [c for c in foc if c in pd.read_parquet(files[0]).columns]   # read schema ONCE
    fo = pd.concat([pd.read_parquet(f, columns=avail) for f in files], ignore_index=True)
    fo["ts"] = pd.to_datetime(fo["decision_ts_utc"], utc=True)
    fo = fo.drop(columns=["decision_ts_utc"]).drop_duplicates("ts")
    df = m.merge(fo, on="ts", how="left").sort_values("ts").reset_index(drop=True)
    feats_all = [c for c in CONV + PRICE + V10H + ["raw_adv"] if c in df.columns]
    feats_conv = [c for c in CONV + ["raw_adv"] if c in df.columns]
    real = df["realized_pnl_bps"].astype(float).values
    df["_win"] = (real > 0).astype(int)
    df["_bigloss"] = (real < -30).astype(int)
    print(f"[meta] traded IQL-natural takes (through-exit) n={len(df)}  win={df['_win'].mean():.3f}  "
          f"bigloss(<-30)={df['_bigloss'].mean():.3f}  mean_pnl={real.mean():.2f}")
    tr = df["ts"] <= pd.Timestamp("2025-06-01", tz="UTC"); te = df["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")
    med = df["ts"].median(); e = df["ts"] <= med; l = df["ts"] > med
    print(f"  train(<=2025-06)={int(tr.sum())}  test(2026)={int(te.sum())}")

    def fit_auc(cols, y, trm, tem):
        Xtr = df.loc[trm, cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        Xte = df.loc[tem, cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        ytr, yte = y[trm.values], y[tem.values]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return float("nan"), None
        c = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                           l2_regularization=1.0, min_samples_leaf=200,
                                           early_stopping=True, random_state=42)
        c.fit(Xtr, ytr)
        p = c.predict_proba(Xte)[:, 1]
        return roc_auc_score(yte, p), p

    print("\n=== AUC: predict take-WIN (through-exit) on the traded subset, strict OOT ===")
    for name, y in (("WIN(>0)", df["_win"].values), ("BIG-LOSS(<-30)", df["_bigloss"].values)):
        a_full, _ = fit_auc(feats_all, y, tr, te)
        a_conv, _ = fit_auc(feats_conv, y, tr, te)
        ae, _ = fit_auc(feats_all, y, e, l); al, _ = fit_auc(feats_all, y, l, e)
        print(f"  [{name:14s}] 2026: full={a_full:.3f}  conv-only={a_conv:.3f}  | robustness full e→l={ae:.3f} l→e={al:.3f}")

    # ECONOMIC prune test (load-bearing): meta-model predicts WIN; drop the bottom-K% on the 2026 test;
    # does remaining total realized / win / worst improve, or did we cut +EV trades?
    print("\n=== ECONOMIC prune test (2026 holdout, REAL through-exit PnL) ===")
    _, p_test = fit_auc(feats_all, df["_win"].values, tr, te)
    if p_test is None:
        print("  (degenerate)"); return
    rte = real[te.values]
    base_tot, base_win, base_n = rte.sum(), (rte > 0).mean(), len(rte)
    print(f"  take ALL (current)     : n={base_n} total={base_tot:8.0f} mean={rte.mean():6.2f} win={base_win:.3f} worst={rte.min():.1f}")
    order = np.argsort(p_test)  # ascending predicted win-prob = most-likely-loser first
    for K in (0.05, 0.10, 0.20):
        n_drop = int(K * len(p_test))
        keep = np.ones(len(p_test), bool); keep[order[:n_drop]] = False
        v = rte[keep]; dropped = rte[~keep]
        print(f"  drop bottom-{int(K*100):2d}% (n={n_drop}): kept_total={v.sum():8.0f} ({100*(v.sum()-base_tot)/abs(base_tot):+.1f}%) "
              f"kept_mean={v.mean():6.2f} kept_win={(v>0).mean():.3f} | DROPPED mean={dropped.mean():6.2f} win={(dropped>0).mean():.3f} "
              f"({'CUTS LOSERS ✓' if dropped.mean() < 0 else 'CUTS +EV TRADES ✗'})")
    print("  Verdict: meta-labeling helps ONLY if the DROPPED trades are net-negative (cut losers, not winners).")


if __name__ == "__main__":
    main()
