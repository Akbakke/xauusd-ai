"""SWING PIVOT PRE-GATE — is DIRECTION more predictable at swing horizons (days) than at the current 8h?

User pivot 2026-06-24 (scalp→swing). Before recommending a full swing rebuild, prove the swing label
SEPARATES OOT (AGENTS.md LABEL-REWARD PRE-GATE). If multi-day direction-AUC ≈ the 8h ceiling (~0.62) or
worse OOT, swing is just a longer version of the same ceiling. If it RISES with horizon, swing is a real
new edge frontier worth the rebuild.

Forward returns computed from the M5 canonical tape (bar-count shifts → skips weekends naturally); features
from forward_outcome (RAW MARKET features only — chain probs p_long/p_short/margin EXCLUDED so the cross-
horizon comparison is apples-to-apples, not favouring the 8h-tuned chain). Strict OOT BOTH directions
(train<2026→test2026 AND train≥2023→test 2020-22). Honest n: long-horizon NON-OVERLAPPING samples are scarce
(that scarcity IS part of the swing verdict) — test subsampled to ~non-overlapping, low-n cells flagged.

NO retrain, NO new chain model/feature/label. Diagnostic pre-gate only; cement+live UNCHANGED.
Run: .venv/bin/python -m gx1.research.oot_swing_horizon_pregate_20260624
"""
from __future__ import annotations
import glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from gx1.research.oot_signal_halflife_surrogate_20260624 import FEATURES, sharpe

TAPE = "/home/andre2/GX1_DATA/runs/PREBUILT_REBUILD_20260613/cv3/xauusd_m5_CANONICAL_V3_2020_2026.parquet"
FO = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
OUT_JSON = "/tmp/swing_horizon_pregate_20260624.json"
# market-only features (drop the 8h-tuned chain probs for a fair cross-horizon test)
MFEAT = [f for f in FEATURES if f not in ("p_long", "p_short", "margin")]
HORIZONS = {"8h_K96(curr)": 96, "1d": 288, "2d": 576, "3d": 864, "5d": 1440, "10d": 2880}


def load_tape():
    t = pd.read_parquet(TAPE, columns=["time", "close"])
    t["time"] = pd.to_datetime(t["time"], utc=True)
    t = t.sort_values("time").reset_index(drop=True)
    for lbl, h in HORIZONS.items():
        t[f"fwd_{lbl}"] = (t["close"].shift(-h) - t["close"]) / t["close"] * 1e4  # bps
    return t


def load_fo():
    cols = list(dict.fromkeys(["decision_ts_utc"] + MFEAT))
    parts = []
    for fp in sorted(glob.glob(f"{FO}/*.parquet")):
        d = pd.read_parquet(fp, columns=[c for c in cols if c])
        if len(d):
            parts.append(d)
    fo = pd.concat(parts, ignore_index=True)
    fo["time"] = pd.to_datetime(fo["decision_ts_utc"], utc=True)
    return fo


def evaluate(tr, te, ycol):
    feats = [f for f in MFEAT if f in tr.columns]
    m = HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.05, l2_regularization=1.0, random_state=0)
    m.fit(tr[feats].values, tr[ycol].values)
    pred = m.predict(te[feats].values)
    y = te[ycol].values
    try:
        auc = roc_auc_score((y > 0).astype(int), pred)
    except Exception:
        auc = np.nan
    ic = spearmanr(pred, y).correlation
    topd = y[pred >= np.percentile(pred, 90)]
    # naive momentum baseline: predict up if recent trend up (ema100 slope)
    mom_auc = np.nan
    if "ema100_slope_canon_v1" in te.columns:
        try:
            mom_auc = roc_auc_score((y > 0).astype(int), te["ema100_slope_canon_v1"].fillna(0).values)
        except Exception:
            pass
    return dict(n_test=int(len(te)), auc=round(float(auc), 4), mom_auc=round(float(mom_auc), 4),
                rankIC=round(float(ic), 4), topD_ret=round(float(topd.mean()), 1), topD_sharpe=round(sharpe(topd), 3),
                base_mean=round(float(y.mean()), 1), median_abs_move=round(float(np.median(np.abs(y))), 1))


def main():
    tape = load_tape()
    fo = load_fo()
    df = fo.merge(tape[["time"] + [f"fwd_{l}" for l in HORIZONS]], on="time", how="inner")
    df = df.sort_values("time").reset_index(drop=True)
    print(f"[data] merged decision bars={len(df)}  {df.time.min()} .. {df.time.max()}  market-feats={len([f for f in MFEAT if f in df.columns])}")
    OUT = {"meta": dict(n=len(df), horizons=HORIZONS, features="market-only (no chain probs)", note="fwd returns from M5 tape, bar-shift")}

    splits = {
        "OOT-2026 (train<2026 → test 2026)": (lambda t: t < pd.Timestamp("2026-01-01", tz="UTC"), lambda t: t >= pd.Timestamp("2026-01-01", tz="UTC")),
        "OOT-early (train≥2023 → test 2020-22)": (lambda t: t >= pd.Timestamp("2023-01-01", tz="UTC"), lambda t: t < pd.Timestamp("2023-01-01", tz="UTC")),
    }
    for sname, (trf, tef) in splits.items():
        print("\n" + "=" * 104)
        print(f"SPLIT: {sname}")
        print("=" * 104)
        print(f"  {'horizon':14s} {'n_test_eff':>10} {'AUC':>7} {'mom_AUC':>8} {'rankIC':>8} {'topD_ret':>9} {'topD_Sh':>8} {'drift':>7} {'med|move|':>9}")
        OUT.setdefault(sname, {})
        for lbl, h in HORIZONS.items():
            yc = f"fwd_{lbl}"
            d = df.dropna(subset=[yc]).copy()
            tr = d[trf(d.time)]
            te = d[tef(d.time)]
            # subsample to ~non-overlapping for honest eval; dense-ish train for fit
            te = te.iloc[::h].copy()                       # non-overlapping test
            tr = tr.iloc[::max(6, h // 24)].copy()         # dense train (overlap ok for fitting)
            if len(te) < 12 or len(tr) < 500:
                OUT[sname][lbl] = {"n_test": int(len(te)), "n_train": int(len(tr)), "note": "LOW-N inconclusive"}
                print(f"  {lbl:14s} {len(te):>10} {'':>7} {'':>8} {'':>8}   LOW-N (n_train={len(tr)}) — inconclusive")
                continue
            r = evaluate(tr, te, yc)
            r["n_train"] = int(len(tr))
            r["low_n"] = bool(r["n_test"] < 30)
            OUT[sname][lbl] = r
            flag = "  <LOW-N>" if r["low_n"] else ""
            print(f"  {lbl:14s} {r['n_test']:>10} {r['auc']:>7.4f} {r['mom_auc']:>8.4f} {r['rankIC']:>+8.4f} "
                  f"{r['topD_ret']:>+9.1f} {r['topD_sharpe']:>+8.3f} {r['base_mean']:>+7.1f} {r['median_abs_move']:>9.1f}{flag}")

    # trend persistence: autocorr of non-overlapping h-period returns (does gold trend more at multi-day?)
    print("\n" + "=" * 104); print("TREND PERSISTENCE — lag-1 autocorr of NON-OVERLAPPING h-period returns (>0 = trend-follow edge)")
    print("=" * 104)
    OUT["persistence"] = {}
    for lbl, h in HORIZONS.items():
        s = tape[f"fwd_{lbl}"].dropna().iloc[::h].values
        if len(s) > 30:
            ac = float(np.corrcoef(s[:-1], s[1:])[0, 1])
            OUT["persistence"][lbl] = dict(n=int(len(s)), autocorr=round(ac, 4))
            print(f"  {lbl:14s} n={len(s):5d}  lag1_autocorr={ac:+.4f}  ({'TRENDING' if ac > 0.05 else 'MEAN-REVERT' if ac < -0.05 else '~random'})")

    with open(OUT_JSON, "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print(f"\n→ {OUT_JSON}")


if __name__ == "__main__":
    main()
