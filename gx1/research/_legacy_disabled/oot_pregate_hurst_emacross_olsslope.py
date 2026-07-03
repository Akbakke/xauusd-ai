"""OOT pre-gate (user vedtak 2026-06-23, LABEL-REWARD PRE-GATE rule, AGENTS.md):
do 3 candidate features add SEPARATION the chain's existing trend features lack?

Candidates (computed from PRICE on the M5 canonical tape, asof-mapped to decision bars — leak-safe):
  1. HURST exponent (rolling, structure-function / variance-of-differences; H>0.5 trending, <0.5 mean-revert)
  2. EMA20×EMA50 cross  = ema20_50_diff_atr (the built-but-DEAD htf_features V3 col; (ema20-ema50)/atr14, clip ±15)
  3. OLS regression slope (rolling least-squares slope of close, ATR-normalized) + R² (trend quality)

Method (mirrors gx1/research/oot_pregate_dip_reversal_structural.py — ONE-TRUTH methodology):
  Target  : load-bearing entry targets on forward_outcome_clean decision bars, K96 family:
              DIRECTION  = long_terminal_pnl > short_terminal_pnl   (the side the bot picks)
              BIG_LOSS   = long_terminal_pnl < -30 bps  (→ skip the toxic take)
              HIGH_MFE   = long_mfe > median             (→ selection / size up)
  Split   : strict out-of-time BOTH directions (fit early→test late AND fit late→test early), split @ median ts.
  Model   : HistGradientBoosting (same hyperparams as the dip-struct pre-gate).
  Gate    : a candidate "earns a place" ONLY if it adds ΔAUC over the pure-price-trend baseline AND
            baseline+candidate clears OOT both-halves >= 0.58 (direction) — i.e. NON-REDUNDANT + separable.
  Leakage : asserted — no forward/label token in any feature; candidates use only past/current bars.
"""
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FO_DIR = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
M5_PREBUILT = ("/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
               "xauusd_m5_CANONICAL_V3_2020_2026.parquet")
RNG = np.random.default_rng(20260623)
SAMPLE_CAP = 250_000
K = "K96"
GATE = 0.58


# ── 1. candidate feature computation on the M5 tape ───────────────────────────
@njit(cache=True)
def _rolling_hurst(close, W, lags):
    """Per-bar Hurst via structure-function: sd(close[i]-close[i-lag]) ~ lag^H.
    H = slope of log(sd) vs log(lag) over a trailing window of length W. ~0.5=random walk."""
    n = close.shape[0]
    out = np.full(n, np.nan)
    nl = lags.shape[0]
    lx = np.empty(nl)
    for j in range(nl):
        lx[j] = np.log(lags[j])
    lx_mean = lx.mean()
    lx_var = 0.0
    for j in range(nl):
        lx_var += (lx[j] - lx_mean) ** 2
    for t in range(W, n):
        ly = np.empty(nl)
        ok = True
        for j in range(nl):
            lag = lags[j]
            # variance of lag-differences within the trailing window [t-W, t]
            s = 0.0
            s2 = 0.0
            cnt = 0
            for i in range(t - W + lag, t + 1):
                d = close[i] - close[i - lag]
                s += d
                s2 += d * d
                cnt += 1
            if cnt < 2:
                ok = False
                break
            var = (s2 - s * s / cnt) / (cnt - 1)
            if var <= 0.0:
                ok = False
                break
            ly[j] = 0.5 * np.log(var)
        if not ok:
            continue
        ly_mean = ly.mean()
        cov = 0.0
        for j in range(nl):
            cov += (lx[j] - lx_mean) * (ly[j] - ly_mean)
        out[t] = cov / lx_var  # slope = H
    return out


@njit(cache=True)
def _rolling_ols(close, W):
    """Per-bar least-squares slope of close vs bar-index over trailing window W, + R².
    Returns (slope_per_bar, r2)."""
    n = close.shape[0]
    slope = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    # x = 0..W-1, precompute x stats (constant)
    sx = 0.0
    sxx = 0.0
    for i in range(W):
        sx += i
        sxx += i * i
    xbar = sx / W
    sxx_c = sxx - sx * sx / W
    for t in range(W - 1, n):
        sy = 0.0
        sxy = 0.0
        syy = 0.0
        for i in range(W):
            yv = close[t - W + 1 + i]
            sy += yv
            sxy += i * yv
            syy += yv * yv
        sxy_c = sxy - sx * sy / W
        syy_c = syy - sy * sy / W
        b = sxy_c / sxx_c if sxx_c > 0 else 0.0
        slope[t] = b
        if syy_c > 0:
            r2[t] = (sxy_c * sxy_c) / (sxx_c * syy_c)
        else:
            r2[t] = 0.0
    return slope, r2


def build_candidates() -> pd.DataFrame:
    m5 = pd.read_parquet(M5_PREBUILT, columns=["time", "close", "high", "low"])
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time").reset_index(drop=True)
    close = m5["close"].to_numpy(np.float64)
    high = m5["high"].to_numpy(np.float64)
    low = m5["low"].to_numpy(np.float64)
    # build-truth ATR(14) SMA-TR (same as the dip-struct pre-gate / step1 parity)
    tr = pd.concat([(m5["high"] - m5["low"]).abs(),
                    (m5["high"] - m5["close"].shift(1)).abs(),
                    (m5["low"] - m5["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().to_numpy(np.float64)
    atr_safe = np.maximum(atr, 1e-3)
    out = pd.DataFrame({"time": m5["time"]})

    # 1) HURST (two windows)
    for W in (128, 256):
        out[f"hurst_{W}"] = _rolling_hurst(close, W, np.array([1, 2, 4, 8, 16, 32], dtype=np.int64))

    # 2) EMA20×EMA50 cross — the built-but-dead ema20_50_diff_atr (htf_features.py:715 formula), M5
    ema20 = pd.Series(close).ewm(span=20, adjust=False).mean().to_numpy()
    ema50 = pd.Series(close).ewm(span=50, adjust=False).mean().to_numpy()
    out["ema20_50_diff_atr"] = np.clip((ema20 - ema50) / atr_safe, -15, 15)
    # also the EMA50×EMA200 distance for completeness (no dedicated col in cement)
    ema200 = pd.Series(close).ewm(span=200, adjust=False).mean().to_numpy()
    out["ema50_200_diff_atr"] = np.clip((ema50 - ema200) / atr_safe, -30, 30)

    # 3) OLS regression slope (two windows), ATR-normalized total-move-over-window + R²
    for W in (20, 50):
        s, r2 = _rolling_ols(close, W)
        out[f"ols_slope_{W}_atr"] = np.clip(s * W / atr_safe, -30, 30)  # total fitted move over window, ATR units
        out[f"ols_r2_{W}"] = r2
    return out


CAND_GROUPS = {
    "HURST": ["hurst_128", "hurst_256"],
    "EMA20x50_cross": ["ema20_50_diff_atr", "ema50_200_diff_atr"],
    "OLS_slope": ["ols_slope_20_atr", "ols_r2_20", "ols_slope_50_atr", "ols_r2_50"],
}
ALL_CANDS = sum(CAND_GROUPS.values(), [])

# pure-price-trend BASELINE (the chain's EXISTING trend features — NO conviction/leak cols)
BASELINE = ["ema20_slope_canon_v1", "ema100_slope_canon_v1", "pos_vs_ema200_canon_v1",
            "_v1_rsi14_canon_v1", "_v1h1_ema_diff_canon_v1", "_v1h4_ema_diff_canon_v1",
            "_v1_ema_diff_canon_v1", "m15_trend_sign_canon_v2_canon_v1",
            "d1_ema_slope_20_canon_v2_canon_v1", "_v1_r5_canon_v1", "_v1_r24_canon_v1", "atr_bps"]
LABELS = [f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1",
          f"take_now_long_mfe_bps_at_{K}_v1"]


def load_decisions():
    files = sorted(glob.glob(f"{FO_DIR}/*.parquet"))
    avail = set(pd.read_parquet(files[0]).columns)
    need = [c for c in sorted(set(BASELINE + LABELS + ["decision_ts_utc"])) if c in avail]
    base_have = [c for c in BASELINE if c in avail]
    df = pd.concat([pd.read_parquet(f, columns=need) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    return df, base_have


def main():
    print("[1/4] building candidate features on M5 tape ...")
    m5 = build_candidates()
    cand_cols = [c for c in m5.columns if c != "time"]
    # sanity: candidates are alive + Hurst centered near 0.5
    for c in CAND_GROUPS["HURST"]:
        v = m5[c].dropna()
        print(f"  {c}: mean={v.mean():.3f} std={v.std():.3f} (≈0.5=random walk)")

    print("[2/4] loading forward_outcome decision bars ...")
    df, base_have = load_decisions()
    print(f"  decision rows={len(df)}  baseline cols available={len(base_have)}/{len(BASELINE)}")

    # asof-map candidates onto decision bars (cand time <= decision ts → leak-safe)
    ts_arr = m5["time"].values.astype("datetime64[ns]").astype(np.int64)
    dec = df["ts"].values.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(ts_arr, dec, side="right") - 1
    ok = idx >= 0
    df = df[ok].reset_index(drop=True); idx = idx[ok]
    for c in cand_cols:
        df[c] = m5[c].to_numpy()[idx]

    # LEAKAGE GUARD
    FORBIDDEN = ("terminal_pnl", "mfe", "mae", "forward", "take_now", "_outcome", "trough", "bars_to")
    for c in base_have + ALL_CANDS + ["ema50_200_diff_atr"]:
        assert not any(t in c for t in FORBIDDEN), f"LEAKAGE: {c}"

    df = df.dropna(subset=[f"take_now_long_terminal_pnl_at_{K}_v1",
                           f"take_now_short_terminal_pnl_at_{K}_v1"]).reset_index(drop=True)
    df = df.sort_values("ts").reset_index(drop=True)
    med = df["ts"].median()
    early = (df["ts"] <= med).values
    late = (df["ts"] > med).values
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float).values
    sp = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float).values
    mfe = df[f"take_now_long_mfe_bps_at_{K}_v1"].astype(float).values
    TARGETS = {
        "DIRECTION(long>short)": (lp > sp).astype(int),
        "BIG_LOSS(<-30→skip)": (lp < -30.0).astype(int),
        "HIGH_MFE(>median)": (mfe > np.nanmedian(mfe)).astype(int),
    }
    print(f"[3/4] n={len(df)}  split@{med}  early={int(early.sum())} late={int(late.sum())}")

    def auc(cols, y, tr, te):
        Xtr = df.loc[tr, cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        Xte = df.loc[te, cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        ytr, yte = y[tr], y[te]
        if len(Xtr) > SAMPLE_CAP:
            s = RNG.choice(len(Xtr), SAMPLE_CAP, replace=False); Xtr = Xtr.iloc[s]; ytr = ytr[s]
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            return float("nan")
        c = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                           l2_regularization=1.0, min_samples_leaf=200,
                                           early_stopping=True, validation_fraction=0.15, random_state=42)
        c.fit(Xtr, ytr)
        return roc_auc_score(yte, c.predict_proba(Xte)[:, 1])

    print("[4/4] OOT separability (both dirs). ΔAUC = baseline+cand − baseline (the NON-REDUNDANCY test):\n")
    report = {}
    for tname, y in TARGETS.items():
        br = float(y.mean())
        b_el, b_le = auc(base_have, y, early, late), auc(base_have, y, late, early)
        b_avg = (b_el + b_le) / 2
        print(f"=== {tname}  base_rate={br:.3f} ===")
        print(f"  {'arm':26s} OOT(e→l / l→e)   avg     ΔvsBase   sep>={GATE}")
        print(f"  {'baseline(price-trend)':26s} {b_el:.3f} / {b_le:.3f}   {b_avg:.3f}    —")
        report[tname] = {"base_rate": br, "baseline_oot": [round(b_el, 4), round(b_le, 4)], "arms": {}}
        arms = [("+HURST", CAND_GROUPS["HURST"]),
                ("+EMA20x50", CAND_GROUPS["EMA20x50_cross"]),
                ("+OLS_slope", CAND_GROUPS["OLS_slope"]),
                ("+ALL3", ALL_CANDS),
                ("HURST_alone", CAND_GROUPS["HURST"]),
                ("EMA20x50_alone", CAND_GROUPS["EMA20x50_cross"]),
                ("OLS_alone", CAND_GROUPS["OLS_slope"])]
        for aname, cc in arms:
            cols = (base_have + cc) if aname.startswith("+") else cc
            e, l = auc(cols, y, early, late), auc(cols, y, late, early)
            avg = (e + l) / 2
            delta = avg - b_avg if aname.startswith("+") else float("nan")
            sep = bool(e >= GATE and l >= GATE)
            ds = f"{delta:+.4f}" if aname.startswith("+") else "  (alone)"
            print(f"  {aname:26s} {e:.3f} / {l:.3f}   {avg:.3f}   {ds}    {sep}")
            report[tname]["arms"][aname] = dict(oot=[round(e, 4), round(l, 4)], avg=round(avg, 4),
                                                delta_vs_base=(round(delta, 4) if aname.startswith("+") else None),
                                                separable=sep)
        print()

    Path("/tmp/oot_pregate_hurst_emacross_ols.json").write_text(json.dumps(report, indent=2, default=str))
    # VERDICT
    print("=== VERDICT (per candidate: does it add ΔAUC on the load-bearing DIRECTION target?) ===")
    d = report["DIRECTION(long>short)"]["arms"]
    for label, arm in (("HURST", "+HURST"), ("EMA20x50", "+EMA20x50"), ("OLS_slope", "+OLS_slope")):
        delta = d[arm]["delta_vs_base"]
        verdict = "EARNS A PLACE" if (delta is not None and delta >= 0.005 and d[arm]["separable"]) else "FUTILE / redundant"
        print(f"  {label:12s} Δdir-AUC={delta:+.4f}  baseline+cand sep={d[arm]['separable']}  → {verdict}")
    print("\nWROTE /tmp/oot_pregate_hurst_emacross_ols.json")


if __name__ == "__main__":
    main()
