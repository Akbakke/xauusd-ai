"""PRE-GATE (user vedtak 2026-06-25): before retraining the V10 direction head on a
regime/MTF-horizon-conditioned target, prove that conditioning actually SEPARATES the
realized direction on a STRICT out-of-time split (both directions). If it is ~0.5 or
INVERTS OOT, the direction-retrain is futile (the model can't learn what the signal
can't predict) — exactly the 2026-06-14 LABEL-REWARD PRE-GATE rule.

Tests the user's vision directly:
  (1) Does trend_regime predict realized direction OOT? ("short in short-regime, long in long")
  (2) Does a horizon-weighted MTF vote add over the current V10 direction OOT?
  (3) HORIZON-MATCH: do SHORT-TF signs (m5/m15) predict the SHORT horizon (K12/K24) and
      LONG-TF signs (h4/d1) the LONG horizon (K96)? ("d1 short while m5-h4 long -> long
      short-term; m5/m15 short while higher long -> quick short")
  (4) INCREMENTAL: V10-alone vs V10+MTF+regime, fit-train/eval-test AUC, BOTH split dirs.

Target = realized_long_wins@K = (take_now_long_terminal_pnl_at_K > 0). Read-only on
forward_outcome (no retrain). Verdict: conditioning is worth a retrain ONLY if it adds
stable OOT separation (dAUC >~0.02 both splits, no sign-flip).
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FWD = "/home/andre2/GX1_DATA/runs/FASE2B_CLEAN_20260608/forward_outcome_clean/per_week"
HORIZONS = [12, 24, 48, 96]
# per-TF trend-sign sources (short->long horizon)
TF_SLOPE = {
    "m5": "ema20_slope_canon_v1",
    "m15": "m15_trend_sign_canon_v2_canon_v1",
    "h1": "_v1h1_slope5_canon_v1",
    "h4": "_v1h4_slope5_canon_v1",
    "d1": "d1_ema_slope_20_canon_v2_canon_v1",
}
REG_MAP = {"TREND_UP": 1.0, "TREND_NEUTRAL": 0.0, "TREND_DOWN": -1.0}


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    m = np.isfinite(s) & np.isfinite(y)
    if m.sum() < 200 or len(np.unique(y[m])) < 2:
        return np.nan
    return roc_auc_score(y[m], s[m])


def main():
    cols = (["candidate_uid", "decision_ts_utc", "trend_regime",
             "direction_logit_long", "direction_logit_short", "p_long", "p_short"]
            + list(TF_SLOPE.values())
            + [f"take_now_long_terminal_pnl_at_K{k}_v1" for k in HORIZONS])
    files = sorted(glob.glob(FWD + "/*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=[c for c in cols if c]) for f in files],
                   ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"rows={len(df)}  {df['ts'].min()} -> {df['ts'].max()}")

    # signals
    df["v10_dir"] = df["direction_logit_long"] - df["direction_logit_short"]
    df["reg_sign"] = df["trend_regime"].map(REG_MAP).fillna(0.0)
    for tf, c in TF_SLOPE.items():
        df[f"sgn_{tf}"] = np.sign(pd.to_numeric(df[c], errors="coerce"))
    df["mtf_vote_eq"] = df[[f"sgn_{tf}" for tf in TF_SLOPE]].mean(axis=1)            # equal-weight all 5
    df["mtf_short"] = df[["sgn_m5", "sgn_m15"]].mean(axis=1)                          # tactical TFs
    df["mtf_long"] = df[["sgn_h4", "sgn_d1"]].mean(axis=1)                            # strategic TFs
    # horizon-weighted vote: short TFs weighted for short horizon intent
    df["mtf_hw"] = (0.35*df["sgn_m5"] + 0.25*df["sgn_m15"] + 0.2*df["sgn_h1"]
                    + 0.12*df["sgn_h4"] + 0.08*df["sgn_d1"])

    periods = {"2020-2024": df["ts"] < "2025-01-01", "2025-2026": df["ts"] >= "2025-01-01"}

    print("\n========== (1)/(2)/(3) single-signal AUC for realized_long_wins, per period+horizon ==========")
    sigs = ["v10_dir", "reg_sign", "mtf_vote_eq", "mtf_hw", "mtf_short", "mtf_long"]
    for k in HORIZONS:
        y = (pd.to_numeric(df[f"take_now_long_terminal_pnl_at_K{k}_v1"], errors="coerce") > 0).astype(float)
        print(f"\n--- horizon K{k} ({'~%dh'%(k*5//60) if k*5>=60 else '%dm'%(k*5)}) ---")
        print(f"  {'signal':12s} {'2020-24':>9s} {'2025-26':>9s}  flip?")
        for s in sigs:
            a1, a2 = auc(y[periods['2020-2024']], df[s][periods['2020-2024']]), auc(y[periods['2025-2026']], df[s][periods['2025-2026']])
            flip = "INVERTS" if (np.isfinite(a1) and np.isfinite(a2) and (a1-0.5)*(a2-0.5) < 0) else ""
            print(f"  {s:12s} {a1:9.3f} {a2:9.3f}  {flip}")

    print("\n========== (4) INCREMENTAL: V10 vs V10+MTF+regime (fit-train / eval-test, BOTH dirs) ==========")
    feat_v10 = ["v10_dir"]
    feat_full = ["v10_dir", "reg_sign", "mtf_hw", "mtf_short", "mtf_long", "sgn_h1"]
    for k in [24, 96]:
        y = (pd.to_numeric(df[f"take_now_long_terminal_pnl_at_K{k}_v1"], errors="coerce") > 0).astype(float)
        print(f"\n--- horizon K{k} ---")
        for train_p, test_p in [("2020-2024", "2025-2026"), ("2025-2026", "2020-2024")]:
            tr, te = periods[train_p], periods[test_p]
            def fit_eval(feats):
                X = df[feats].replace([np.inf, -np.inf], np.nan)
                m_tr = tr & X.notna().all(axis=1) & y.notna()
                m_te = te & X.notna().all(axis=1) & y.notna()
                if m_tr.sum() < 500 or m_te.sum() < 500:
                    return np.nan
                lr = LogisticRegression(max_iter=500)
                lr.fit(X[m_tr], y[m_tr])
                return roc_auc_score(y[m_te], lr.predict_proba(X[m_te])[:, 1])
            a_v10, a_full = fit_eval(feat_v10), fit_eval(feat_full)
            d = a_full - a_v10 if (np.isfinite(a_v10) and np.isfinite(a_full)) else np.nan
            print(f"  fit {train_p} -> test {test_p}:  V10={a_v10:.3f}  V10+MTF+reg={a_full:.3f}  dAUC={d:+.3f}")

    print("\nVERDICT GUIDE: regime/MTF conditioning is worth the direction-retrain ONLY if "
          "(a) the signals DON'T invert OOT in (1)-(3), AND (b) dAUC >~+0.02 in BOTH split "
          "directions in (4). Sign-flip or dAUC<=0 => futile (consistent with the regime-OOT-inversion finding).")


if __name__ == "__main__":
    main()
