"""OOT pre-gate (user vedtak 2026-06-23, sparked by trade #755: p_long=0.96 but -24 MAE / +8 top /
-180 bps after; bad_path_prob=0.575 flagged the rough path). Three questions, strict OOT both halves:

  A) CALIBRATION — is V10 p_long calibrated? bin p_long → actual OOT long-win-rate + mean PnL + mean MAE.
     (#755 was p_long=0.96 with K96 long-PnL negative — is high p_long systematically meaningful?)
  B) PREDICTIVENESS — does bad_path_prob / path_quality SEPARATE rough/losing longs OOT (GBM AUC, both dirs,
     vs the p_long/margin conviction baseline)? targets: LOSS (long_pnl<0), BIG_MAE (long_mae<=-20), BIG_LOSS(<-30).
  C) MONEY TEST — within the leans-long population (proxy for take-long candidates), does sizing DOWN on high
     bad_path improve RISK-ADJUSTED realized PnL OOT (PnL retained vs MAE-exposure cut), without nuking wins?

Read-only research on forward_outcome_clean (K96). Touches NO cement/contract. Mirrors the dip-struct
pre-gate's load + OOT-split conventions. A size-down overlay would be serve-time/reversible IF this is green.
"""
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FO_DIR = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
RNG = np.random.default_rng(20260623)
SAMPLE_CAP = 250_000
K = "K96"


def load():
    files = sorted(glob.glob(f"{FO_DIR}/*.parquet"))
    avail = set(pd.read_parquet(files[0]).columns)
    need = [c for c in ["decision_ts_utc", "p_long", "p_short", "margin", "uncertainty_score",
                        "bad_path_prob", "path_quality_pred", "trend_regime",
                        f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1",
                        f"take_now_long_mae_bps_at_{K}_v1", f"take_now_long_mfe_bps_at_{K}_v1"] if c in avail]
    df = pd.concat([pd.read_parquet(f, columns=need) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    df = df.dropna(subset=[f"take_now_long_terminal_pnl_at_{K}_v1", "p_long", "bad_path_prob"]).reset_index(drop=True)
    return df.sort_values("ts").reset_index(drop=True)


def main():
    df = load()
    med = df["ts"].median()
    late = df["ts"] > med
    early = df["ts"] <= med
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float).values
    mae = df[f"take_now_long_mae_bps_at_{K}_v1"].astype(float).values
    bp = df["bad_path_prob"].astype(float).values
    pq = df["path_quality_pred"].astype(float).values
    pl = df["p_long"].astype(float).values
    ps = df["p_short"].astype(float).values
    print(f"rows={len(df)}  split@{med}  K={K}  (long terminal-PnL horizon)")
    print(f"bad_path_prob: mean={bp.mean():.3f} std={bp.std():.3f} | path_quality: mean={pq.mean():.3f}")

    # ── A. CALIBRATION of p_long (OOT-late) ───────────────────────────────────
    print("\n=== A. p_long CALIBRATION (OOT = late half; does high p_long mean a long actually wins @K96?) ===")
    print(f"  {'p_long bin':12s} {'n':>7s} {'long_win%':>9s} {'mean_pnl':>9s} {'mean_MAE':>9s}")
    lat = late.values
    for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.01)]:
        m = lat & (pl >= lo) & (pl < hi)
        if m.sum() == 0:
            continue
        print(f"  [{lo:.2f},{hi:.2f}) {m.sum():7d} {(lp[m] > 0).mean()*100:8.1f}% {lp[m].mean():9.2f} {mae[m].mean():9.2f}")
    # the #755 cell: p_long>=0.95
    m = lat & (pl >= 0.95)
    print(f"  -> high-conviction p_long>=0.95 (the #755 cell): n={m.sum()} win={ (lp[m]>0).mean()*100:.1f}% meanPnL={lp[m].mean():.2f} meanMAE={mae[m].mean():.2f}")

    # ── B. PREDICTIVENESS (GBM AUC, both OOT dirs) ────────────────────────────
    print("\n=== B. does bad_path/path_quality SEPARATE rough/losing longs OOT? (AUC, fit e→l / l→e) ===")
    targets = {
        "LOSS(long_pnl<0)": (lp < 0).astype(int),
        "BIG_MAE(<=-20bps)": (mae <= -20).astype(int),
        "BIG_LOSS(<-30bps)": (lp < -30).astype(int),
    }
    arms = {
        "bad_path_only": ["bad_path_prob"],
        "path_quality_only": ["path_quality_pred"],
        "badpath+pathq": ["bad_path_prob", "path_quality_pred"],
        "conviction_base(p_long,p_short,margin)": ["p_long", "p_short", "margin"],
        "base+badpath+pathq": ["p_long", "p_short", "margin", "bad_path_prob", "path_quality_pred"],
    }

    def auc(cols, y, tr, te):
        Xtr = df.loc[tr, cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        Xte = df.loc[te, cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        ytr, yte = y[tr], y[te]
        if len(Xtr) > SAMPLE_CAP:
            s = RNG.choice(len(Xtr), SAMPLE_CAP, replace=False); Xtr = Xtr.iloc[s]; ytr = ytr[s]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return float("nan")
        c = HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.05,
                                           l2_regularization=1.0, min_samples_leaf=200,
                                           early_stopping=True, validation_fraction=0.15, random_state=42)
        c.fit(Xtr, ytr)
        return roc_auc_score(yte, c.predict_proba(Xte)[:, 1])

    ev, lv = early.values, late.values
    report = {}
    for tname, y in targets.items():
        print(f"  [{tname}  base={y.mean():.3f}]")
        report[tname] = {}
        for aname, cols in arms.items():
            e, l = auc(cols, y, ev, lv), auc(cols, y, lv, ev)
            sep = bool(e >= 0.58 and l >= 0.58)
            report[tname][aname] = dict(oot=[round(e, 4), round(l, 4)], sep=sep)
            print(f"     {aname:42s} {e:.3f} / {l:.3f}  sep>=0.58={sep}")

    # ── C. MONEY TEST: size-down on high bad_path, OOT, leans-long population ──
    print("\n=== C. SIZING MONEY TEST (leans-long pop, realized long terminal-PnL @K96, OOT-late) ===")
    leans_long = pl > ps
    hi_conv = leans_long & (df["margin"].astype(float).values >= np.nanpercentile(df["margin"].astype(float).values, 67))
    for popname, popmask in (("leans_long(all)", leans_long), ("leans_long & top-tercile-margin (≈taken)", hi_conv)):
        m = lat & popmask
        if m.sum() == 0:
            print(f"  [{popname}] n=0"); continue
        pnl = lp[m]; ma = mae[m]; bpm = bp[m]
        def metrics(size):
            tot = float(np.sum(size * pnl))
            mae_exp = float(np.sum(size * np.abs(ma)))   # size-weighted MAE exposure
            ratio = tot / mae_exp if mae_exp > 0 else float("nan")
            return tot, mae_exp, ratio
        flat = np.ones(len(pnl))
        # bad_path-down: linear taper 1 → 0.5 as bad_path 0→1 (clip), reversible-overlay-style
        bp_down = np.clip(1.0 - 0.5 * bpm, 0.5, 1.0)
        # bucket: bad_path>0.5 → 0.5x else 1x (the simplest serve-time rule)
        bp_bucket = np.where(bpm > 0.5, 0.5, 1.0)
        print(f"\n  [{popname}] n={m.sum()}  (mean bad_path={bpm.mean():.3f})")
        for sname, sz in (("flat 1.0", flat), ("bp_taper(1→0.5)", bp_down), ("bp_bucket(>0.5→0.5x)", bp_bucket)):
            tot, mexp, ratio = metrics(sz)
            print(f"     {sname:22s} total_PnL={tot:9.0f}  MAE_exposure={mexp:9.0f}  PnL/MAE={ratio:+.4f}  units={sz.sum():.0f}")
        # how much PnL retained vs MAE cut for the bucket rule
        ft, fm, _ = metrics(flat); bt, bm, _ = metrics(bp_bucket)
        print(f"     -> bucket vs flat: PnL retained={bt/ft*100:.1f}%  MAE-exposure cut to={bm/fm*100:.1f}%  "
              f"(GREEN if PnL% > MAE%, i.e. risk-adjusted improves)")

    Path("/tmp/oot_pregate_badpath.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nWROTE /tmp/oot_pregate_badpath.json")


if __name__ == "__main__":
    main()
