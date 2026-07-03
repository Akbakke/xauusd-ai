"""Entry-quality / MAE-MFE timing analysis — does LAM50 OVER-SKIP recoverable-dip edge, and which
decision-time FEATURES separate good-timed entries from straight-to-MAE ones?

Motivation (user 2026-06-09): cemented LAM50 trades too LITTLE — its λ=5 MAE-before-MFE penalty
blanket-skips entries that go adverse before recovering (the mean-reversion dips V10 correctly flags).
Goal: measure offline (forward_outcome records the MAE/MFE outcome for EVERY candidate, taken or
skipped — zero live risk) (a) how much GOOD edge LAM50 leaves on the table, (b) the conviction
frontier (open the gate ↔ edge ↔ MAE-profile ↔ slot-util), (c) which features to GATE ON for quality
instead of a blanket λ. Then the fix is a feature-conditional entry reward (penalize straight-to-MAE,
REWARD recoverable dips that reach MFE) — learned, not a hardcoded threshold.

Reads forward_outcome (per-week) + the entry-IQL decisions.parquet. Research-only; touches no model.
Usage: python -m gx1.research.entry_quality_mae_mfe_analysis [--forward-outcome-dir DIR] [--decisions PARQUET]
       [--k 96] [--recoverable-atr-frac 0.5] [--feature-importance]
"""
import argparse
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/andre2/src/GX1_ENGINE")

# Defaults = the CLEAN (WS2) cemented chain. Repointed 2026-06-10: the old REGIME_V4 forward_outcome_clean
# + entry_decisions_lam50 dumps were quarantined to runs/_SUPERSEDED_20260610/. DEF_DEC = the current
# conviction-gate entry decisions (the live entry operating point); pass --decisions explicitly to analyse
# a different policy's dump.
DEF_FO = "/home/andre2/GX1_DATA/runs/FASE2B_CLEAN_20260608/forward_outcome_clean/per_week"
DEF_DEC = "/home/andre2/GX1_DATA/runs/FASE2B_CLEAN_20260608/entry_decisions_conviction20/decisions.parquet"
# April-2026 x10-REPAIRED 2026-06 — no exclusion anywhere; April is valid data.

# Decision-time feature candidates (decision-time only; NO outcomes, NO entry-IQL q/advantage = circular).
FEATURE_HINTS = ("p_long", "p_short", "p_flat", "p_hat", "margin", "uncertainty", "tradable_prob",
                 "mfe_first_n_pred", "path_quality_pred", "bad_path_prob", "direction_logit",
                 "tf_agreement", "path_quality_log_var", "path_quality_std", "vol_regime",
                 "trend_regime", "atr_bps", "entry_spread_bps", "dip", "struct", "ema", "rsi",
                 "vol_", "regime", "hour_utc", "weekday_utc", "smc", "session_id")
EXCLUDE_SUBSTR = ("terminal_pnl", "mae_before_mfe", "_mae_", "_mfe_", "time_to_mfe", "forward_bars",
                  "wait_", "take_now_", "q_take", "q_skip", "advantage", "confidence_softmax",
                  "action_", "_uid", "trade_id", "decision_reason", "decision_ts", "run_id",
                  "accepted", "policy_lane")


def load(fo_dir, dec_path, K):
    files = sorted(glob.glob(f"{fo_dir}/*.parquet"))
    if not files:
        raise SystemExit(f"no forward_outcome parquets under {fo_dir}")
    keep_out = []
    for side in ("long", "short"):
        for m in ("terminal_pnl", "mfe_bps", "mae_bps", "mae_before_mfe_bps", "time_to_mfe"):
            keep_out.append(f"take_now_{side}_{m}_at_K{K}_v1")
    fo = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    dec = pd.read_parquet(dec_path)
    df = fo.merge(dec, on="candidate_uid", how="inner", suffixes=("", "_dec"))
    ts = df["candidate_uid"].str.extract(r"(\d{8}T\d{6})$")[0]
    df["entry_ts"] = pd.to_datetime(ts, format="%Y%m%dT%H%M%S", utc=True)
    # April-2026 is x10-REPAIRED (2026-06) → NO exclusion; use ALL data.
    return df, keep_out


def policy_side_outcome(df, K):
    """Outcome on the side the POLICY prefers (argmax q_take_long/short) — what it WOULD take."""
    ql = pd.to_numeric(df["q_take_long_v1"], errors="coerce")
    qs = pd.to_numeric(df["q_take_short_v1"], errors="coerce")
    side = np.where(ql >= qs, "long", "short")
    out = {}
    for m in ("terminal_pnl", "mfe_bps", "mae_bps", "mae_before_mfe_bps", "time_to_mfe"):
        lv = pd.to_numeric(df[f"take_now_long_{m}_at_K{K}_v1"], errors="coerce").to_numpy()
        sv = pd.to_numeric(df[f"take_now_short_{m}_at_K{K}_v1"], errors="coerce").to_numpy()
        out[m] = np.where(side == "long", lv, sv)
    out["side"] = side
    return out


def classify(term, mae_before_mfe, atr_bps, recov_frac, spread):
    """BAD = term<=spread (no net edge). GOOD split: RECOVERABLE if it went meaningfully adverse
    before recovering (mae_before_mfe >= recov_frac*atr), else CLEAN (well-timed)."""
    cost = np.maximum(spread, 0.0)
    good = term > cost
    adverse_thr = np.maximum(recov_frac * np.maximum(atr_bps, 1.0), 5.0)
    recoverable = good & (mae_before_mfe >= adverse_thr)
    clean = good & ~recoverable
    bad = ~good
    cls = np.where(clean, "CLEAN", np.where(recoverable, "RECOVERABLE", "BAD"))
    return cls


def agg_cls(term, cls, mask=None):
    m = np.ones(len(term), bool) if mask is None else mask
    t = term[m]; c = cls[m]
    n = int(m.sum())
    out = {"n": n, "mean_bps": round(float(np.nanmean(t)), 2) if n else float("nan"),
           "total_bps": round(float(np.nansum(t)), 0)}
    for k in ("CLEAN", "RECOVERABLE", "BAD"):
        kk = (c == k)
        out[k] = {"n": int(kk.sum()), "pct": round(100 * kk.mean(), 1) if n else 0.0,
                  "total_bps": round(float(np.nansum(t[kk])), 0)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward-outcome-dir", default=DEF_FO)
    ap.add_argument("--decisions", default=DEF_DEC)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--recoverable-atr-frac", type=float, default=0.5)
    ap.add_argument("--feature-importance", action="store_true")
    ap.add_argument("--validate-relmae", action="store_true",
                    help="Offline label-validation of R_RELMAE design: does relMAE separate "
                         "recoverable vs bad (vs raw MAE), and do features predict good-edge (held-out AUC)?")
    args = ap.parse_args()

    df, _ = load(args.forward_outcome_dir, args.decisions, args.k)
    o = policy_side_outcome(df, args.k)
    term = o["terminal_pnl"]
    maebm = o["mae_before_mfe_bps"]
    atr = pd.to_numeric(df.get("atr_bps", pd.Series(np.full(len(df), 20.0))), errors="coerce").fillna(20.0).to_numpy()
    spread = pd.to_numeric(df.get("entry_spread_bps", pd.Series(np.full(len(df), 1.5))), errors="coerce").fillna(1.5).to_numpy()
    cls = classify(term, maebm, atr, args.recoverable_atr_frac, spread)
    action = df["action_label_v1"].astype(str).to_numpy()
    took = np.array([a.startswith("TAKE") for a in action])
    skipped = ~took
    adv = pd.to_numeric(df["advantage_over_skip_v1"], errors="coerce").fillna(0.0).to_numpy()

    print(f"\n===== ENTRY-QUALITY / MAE-MFE @ K{args.k} (recoverable if mae_before_mfe>={args.recoverable_atr_frac}*atr) =====")
    print(f"candidates (ALL data, April repaired): {len(df)}   took={took.sum()} ({100*took.mean():.1f}%)   skipped={skipped.sum()}")
    print("\n-- ALL candidates, policy-preferred side, by timing class --")
    a = agg_cls(term, cls)
    for k in ("CLEAN", "RECOVERABLE", "BAD"):
        print(f"   {k:11s}: n={a[k]['n']:6d} ({a[k]['pct']:4.1f}%)  total={a[k]['total_bps']:.0f} bps")

    print("\n-- LAM50 SKIP analysis: of what it SKIPPED, how much was GOOD (over-skip) vs correctly BAD --")
    s = agg_cls(term, cls, skipped)
    good_skipped = s["CLEAN"]["n"] + s["RECOVERABLE"]["n"]
    good_skipped_bps = s["CLEAN"]["total_bps"] + s["RECOVERABLE"]["total_bps"]
    print(f"   skipped={s['n']}  -> GOOD(clean+recov)={good_skipped} ({100*good_skipped/max(1,s['n']):.1f}%)  "
          f"BAD(correctly)={s['BAD']['n']} ({100*s['BAD']['n']/max(1,s['n']):.1f}%)")
    print(f"   ⚠ LEFT ON TABLE (sum terminal of skipped-but-GOOD): {good_skipped_bps:.0f} bps "
          f"(of which RECOVERABLE-dips: {s['RECOVERABLE']['total_bps']:.0f} bps over {s['RECOVERABLE']['n']} trades)")
    t = agg_cls(term, cls, took)
    print(f"   for comparison TAKEN={t['n']}: mean={t['mean_bps']} bps  GOOD={t['CLEAN']['n']+t['RECOVERABLE']['n']} "
          f"BAD={t['BAD']['n']} ({100*t['BAD']['n']/max(1,t['n']):.1f}%)")

    print("\n-- CONVICTION FRONTIER (rank ALL candidates by advantage_over_skip; take top-X%) --")
    print("   coverage  n_take  mean_bps  total_bps  %CLEAN %RECOV %BAD  (vs LAM50 actual take-rate above)")
    order = np.argsort(-adv)
    for cov in (0.02, 0.05, 0.083, 0.12, 0.20, 0.35, 0.50, 0.75, 1.00):
        ntake = max(1, int(cov * len(df)))
        idx = order[:ntake]
        tt = term[idx]; cc = cls[idx]
        pc = {k: 100 * (cc == k).mean() for k in ("CLEAN", "RECOVERABLE", "BAD")}
        print(f"   {cov*100:5.1f}%   {ntake:6d}  {np.nanmean(tt):8.2f}  {np.nansum(tt):9.0f}   "
              f"{pc['CLEAN']:5.1f} {pc['RECOVERABLE']:5.1f} {pc['BAD']:5.1f}")
    print("   (advantage_over_skip is the policy's OWN conviction; a feature-based score may rank better"
          " — see --feature-importance.)")

    if args.validate_relmae:
        validate_relmae(df, o, term, spread, cls, atr, adv)

    if args.feature_importance:
        run_feature_importance(df, term, spread)


def rank_auc(score, pos):
    """Tie-aware ROC-AUC (Mann-Whitney) — higher score => more likely pos. No sklearn dep."""
    score = np.asarray(score, float)
    pos = np.asarray(pos, bool)
    ok = ~np.isnan(score)
    score, pos = score[ok], pos[ok]
    npos, nneg = int(pos.sum()), int((~pos).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()  # average ranks for ties
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def validate_relmae(df, o, term, spread, cls, atr, adv):
    """OFFLINE label-validation of the R_RELMAE design BEFORE any retrain (measure-then-learn):
    (1) does relMAE = mae_before_mfe/(mae_before_mfe+mfe) separate recoverable-good from bad, and is it
    a BETTER separator than the raw mae_before_mfe LAM50 penalizes? (2) do decision-time features predict
    good-edge on a HELD-OUT (chronological) split? Honest: relMAE can FAIL on losers that peak then crash
    (low relMAE) — the AUC shows whether terminal must carry that, not relMAE."""
    eps = 1.0
    maebm = np.maximum(o["mae_before_mfe_bps"], 0.0)
    mfe = np.maximum(o["mfe_bps"], 0.0)
    relmae = maebm / (maebm + mfe + eps)
    good = term > spread
    print("\n===== R_RELMAE VALIDATION (offline label-check, measure-then-learn) =====")
    print("-- relMAE by timing class (design wants BAD high, CLEAN/RECOVERABLE low) --")
    for k in ("CLEAN", "RECOVERABLE", "BAD"):
        m = (cls == k)
        if m.sum():
            r = relmae[m]
            print(f"   {k:11s}: n={m.sum():6d}  relMAE mean={np.nanmean(r):.3f} median={np.nanmedian(r):.3f} "
                  f"p25={np.nanpercentile(r,25):.3f} p75={np.nanpercentile(r,75):.3f}")
    print("\n-- separation power: AUC for predicting BAD (term<=spread). Higher = better separator --")
    auc_rel = rank_auc(relmae, ~good)
    auc_raw = rank_auc(maebm, ~good)
    auc_relabs = rank_auc(relmae, cls == "BAD")  # vs BAD only (excl. recoverable from negatives)
    print(f"   relMAE         -> BAD:  AUC={auc_rel:.3f}")
    print(f"   raw mae_bef_mfe-> BAD:  AUC={auc_raw:.3f}   (LAM50's signal — Δ={auc_rel-auc_raw:+.3f})")
    print(f"   relMAE separates CLEAN+RECOV vs BAD: {'YES' if auc_rel>0.6 else 'WEAK' if auc_rel>0.55 else 'NO'} "
          f"(and better than raw MAE: {'YES' if auc_rel>auc_raw+0.01 else 'NO'})")
    print("   ⚠ honest check — relMAE of the BAD class that PEAKED-then-crashed (low relMAE losers):")
    bad = ~good
    bad_lowrel = bad & (relmae < 0.4)
    print(f"     BAD with relMAE<0.4 (relMAE misses these; terminal must catch them): "
          f"{bad_lowrel.sum()} / {bad.sum()} ({100*bad_lowrel.sum()/max(1,bad.sum()):.1f}% of BAD)")

    print("\n-- do decision-time FEATURES predict good-edge? HELD-OUT (chronological 70/30) AUC --")
    feat_cols = [c for c in df.columns
                 if any(h in c for h in FEATURE_HINTS) and not any(x in c for x in EXCLUDE_SUBSTR)]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in feat_cols if X[c].notna().mean() > 0.5 and X[c].std() > 0]
    X = X[keep].fillna(X[keep].median())
    y = np.asarray(good).astype(int)
    order = np.argsort(df["entry_ts"].to_numpy())  # chronological
    cut = int(0.7 * len(order))
    tr, te = order[:cut], order[cut:]
    print(f"   features={len(keep)}  train={len(tr)} test={len(te)}  test good-rate={y[te].mean():.3f}")
    try:
        import xgboost as xgb
        dtr = xgb.DMatrix(X.iloc[tr].to_numpy(), label=y[tr], feature_names=keep)
        dte = xgb.DMatrix(X.iloc[te].to_numpy(), feature_names=keep)
        bst = xgb.train({"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
                         "subsample": 0.8, "colsample_bytree": 0.8}, dtr, num_boost_round=150)
        pred = bst.predict(dte)
        auc_feat = rank_auc(pred, y[te].astype(bool))
        print(f"   MULTIVARIATE feature->good-edge HELD-OUT AUC = {auc_feat:.3f}  "
              f"({'usable for a conditional reward' if auc_feat>0.6 else 'WEAK — conditioning would be noise' if auc_feat<0.55 else 'borderline'})")
    except Exception as e:
        print(f"   xgboost held-out failed ({e!r})")
    print("   univariate AUC (top edge predictors -> good-edge), on full set:")
    for f in ("p_hat", "uncertainty_score", "margin", "path_quality_pred"):
        if f in df.columns:
            s = pd.to_numeric(df[f], errors="coerce").to_numpy()
            print(f"     {f:20s} AUC={rank_auc(s, good):.3f}")
    dipcols = [c for c in df.columns if c.startswith("v10_dip_")]
    if dipcols:
        dipmean = pd.to_numeric(df[dipcols].apply(pd.to_numeric, errors='coerce').mean(axis=1), errors="coerce").to_numpy()
        print(f"     {'v10_dip_* (mean)':20s} AUC={rank_auc(dipmean, good):.3f}")

    print("\n-- ORACLE upper-bound vs ACHIEVABLE @ 20% coverage (DO NOT read %BAD=0 of the reward rows as real) --")
    print("   ⚠ relMAE/LAM50 rows rank by the reward LABEL, which CONTAINS future `term`, and BAD is DEFINED")
    print("     by `term` -> %BAD≈0 is a TAUTOLOGY (oracle). The POLICY ranks by FEATURES at decision time,")
    print(f"     bounded by the held-out feature AUC ~0.607 above. The 'adv' row is the honest decision-time rank.")
    cov = 0.20; ntake = int(cov * len(df))
    for lam in (20.0, 40.0, 60.0):
        idx = np.argsort(-(term - lam * relmae))[:ntake]
        print(f"   [ORACLE] relMAE λ={lam:4.0f}: mean_term={np.nanmean(term[idx]):7.2f}  total={np.nansum(term[idx]):9.0f}  %BAD={100*(cls[idx]=='BAD').mean():4.1f}")
    idx = np.argsort(-(term - 5.0 * maebm))[:ntake]
    print(f"   [ORACLE] LAM50 ref   : mean_term={np.nanmean(term[idx]):7.2f}  total={np.nansum(term[idx]):9.0f}  %BAD={100*(cls[idx]=='BAD').mean():4.1f}")
    idx = np.argsort(-adv)[:ntake]
    print(f"   [ACHIEVABLE] adv     : mean_term={np.nanmean(term[idx]):7.2f}  total={np.nansum(term[idx]):9.0f}  "
          f"%BAD={100*(cls[idx]=='BAD').mean():4.1f}  <- honest decision-time conviction ranking")


def run_feature_importance(df, term, spread):
    print("\n-- FEATURE IMPORTANCE: which decision-time features predict GOOD edge (term>spread)? --")
    feat_cols = [c for c in df.columns
                 if any(h in c for h in FEATURE_HINTS) and not any(x in c for x in EXCLUDE_SUBSTR)]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in feat_cols if X[c].notna().mean() > 0.5 and X[c].std() > 0]
    X = X[keep].fillna(X[keep].median())
    y = (term > spread).astype(int)
    print(f"   features used: {len(keep)}  positive rate: {y.mean():.3f}")
    try:
        import xgboost as xgb
        dm = xgb.DMatrix(X.to_numpy(), label=np.asarray(y), feature_names=keep)
        bst = xgb.train({"max_depth": 4, "eta": 0.1, "objective": "binary:logistic",
                         "eval_metric": "auc", "subsample": 0.8, "colsample_bytree": 0.8},
                        dm, num_boost_round=120)
        gain = bst.get_score(importance_type="gain")
        top = sorted(gain.items(), key=lambda kv: -kv[1])[:20]
        print("   TOP features by gain (predict good-edge):")
        for f, g in top:
            print(f"     {f:38s} gain={g:8.1f}")
    except Exception as e:
        print(f"   xgboost unavailable ({e!r}); univariate |corr| with good-edge instead:")
        corr = X.apply(lambda c: np.corrcoef(c, y)[0, 1]).abs().sort_values(ascending=False)
        for f, c in corr.head(20).items():
            print(f"     {f:38s} |corr|={c:.3f}")


if __name__ == "__main__":
    main()
