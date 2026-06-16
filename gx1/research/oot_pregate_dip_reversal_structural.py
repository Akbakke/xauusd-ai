"""OOT pre-gate: do STRUCTURAL features separate the dip-in-uptrend LONG-win target
where TREND features cannot?  (user vedtak LABEL-REWARD PRE-GATE, AGENTS.md)

Target  : "a LONG on this dip would WIN" within the dip-in-uptrend population.
Split   : strict out-of-time, BOTH directions (fit early/test late AND fit late/test early).
Gate    : separable_oot=True iff BOTH OOT AUC >= 0.58.
Model   : GBM (HistGradientBoosting), modest depth, <=250k sample.

Structural features (computed from PRICE on the M5 canonical tape, asof-mapped to decision bars):
  - sweep-then-RECLAIM up/down displacement  (gx1.features.smc_v1, GX1_SMC_SWEEP_RECLAIM)
  - round-number $-level signed ATR-distance + magnet  (gx1.features.group_a_features.round_number_levels)
  - FVG-proximity (M5 nearest unfilled 3-bar gap, signed ATR-dist + active)  (mirror of augment_forward_outcome_v2._fvg_5tf, M5 leg)
  - recent swing-low ATR-distance + bars-since  (gx1.features.smc_v1._detect_swing_pivots, SWING_LOOKBACK=3)
  - stored base SMC cols (sweep_down/size/bars_since/premium_discount) — already structural

Trend features (CURRENT chain cols, the "can trend separate it" arm):
  ema20/100 slope, pos_vs_ema200, rsi14, multi-TF ema_diff (m5/h1/h4), m15 trend sign, d1 ema slope,
  recent returns r5/r24, p_long/p_short/margin/uncertainty (XGB+V10 conviction).

NO forward-looking trough/forward_outcome column is used as a feature (leakage guard asserted).
"""
import os
os.environ["GX1_SMC_SWEEP_RECLAIM"] = "1"  # arm reclaim names in smc_v1

import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from gx1.features.smc_v1 import compute_smc_features, SMC_FEATURE_NAMES, SWING_LOOKBACK, _detect_swing_pivots
from gx1.features.group_a_features import round_number_levels

FO_DIR = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
M5_PREBUILT = ("/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
               "xauusd_m5_CANONICAL_V3_2020_2026.parquet")
RNG = np.random.default_rng(20260615)
SAMPLE_CAP = 250_000
K = "K96"  # load-bearing horizon (cement entry-reward K96 family)

# ── 1. structural feature computation on the M5 tape ──────────────────────────
def build_m5_structural() -> pd.DataFrame:
    m5 = pd.read_parquet(M5_PREBUILT, columns=["time", "open", "high", "low", "close",
                                               "smc_sweep_up", "smc_sweep_down",
                                               "smc_sweep_size_atr", "smc_premium_discount"])
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time").reset_index(drop=True)
    high = m5["high"].to_numpy(np.float64); low = m5["low"].to_numpy(np.float64)
    close = m5["close"].to_numpy(np.float64)
    # SMA-14 TR atr — the build-truth atr used by step1_roundwall_ab (SMC parity proven there)
    tr = pd.concat([(m5["high"] - m5["low"]).abs(),
                    (m5["high"] - m5["close"].shift(1)).abs(),
                    (m5["low"] - m5["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().to_numpy(np.float64)
    atr_safe = np.maximum(atr, 1e-3)
    m5["_atr_build"] = atr.astype(np.float32)

    # SMC base + sweep-reclaim (one-truth)
    smc = compute_smc_features(m5, atr_col="_atr_build")
    out = pd.DataFrame({"time": m5["time"]})
    for c in ["smc_sweep_down", "smc_sweep_size_atr", "smc_bars_since_sweep",
              "smc_premium_discount", "smc_sweep_reclaim_up_displacement_atr",
              "smc_sweep_reclaim_down_displacement_atr"]:
        out[c] = smc[c].to_numpy(np.float64)

    # round-number signed ATR-distance + magnet (vectorized mirror of round_number_levels)
    for g in (100.0, 50.0, 25.0, 10.0):
        nearest = np.round(close / g) * g
        out[f"dist_to_round_{int(g)}_atr"] = (close - nearest) / atr_safe
    # magnet (decayed) — match round_number_levels weights/scale exactly
    W = {100.0: 1.0, 50.0: 0.6, 25.0: 0.35, 10.0: 0.2}
    magnet = np.zeros(len(close))
    for g, w in W.items():
        nearest = np.round(close / g) * g
        d = (close - nearest) / atr_safe
        magnet += w * np.exp(-np.abs(d) / 0.5)
    out["round_magnet_score"] = magnet
    # spot-check vs scalar helper on 5 rows
    for i in RNG.integers(20, len(close), size=5):
        ref = round_number_levels(float(close[i]), float(atr[i]))
        assert abs(ref["dist_to_round_100_atr"] - out["dist_to_round_100_atr"].iloc[i]) < 1e-6, "round-num parity"
        assert abs(ref["round_magnet_score"] - out["round_magnet_score"].iloc[i]) < 1e-5, "magnet parity"

    # FVG M5: nearest UNFILLED 3-bar gap, signed ATR-dist + active (mirror of _fvg_5tf M5 leg).
    # Gap at anchor j is COMPLETED at bar f=j+2; only completed gaps are eligible at decision bar k>f.
    # Numba-jit a bounded backward scan (last LOOKBACK completed gaps) — O(n*LOOKBACK), fast.
    from numba import njit
    n = len(close)
    CLIP, TAU, LOOKBACK = 8.0, 1.5, 240

    @njit(cache=True)
    def _fvg_scan(hi, lo, cp_arr, atr, clip, lookback):
        nn = hi.shape[0]
        dist = np.full(nn, clip)
        for k in range(2, nn):
            cp = cp_arr[k]; a = atr[k]
            sgn = clip
            j0 = k - 2 - lookback
            if j0 < 0:
                j0 = 0
            for j in range(k - 3, j0 - 1, -1):   # f=j+2 must be < k → j <= k-3
                f = j + 2
                # bullish gap hi[j] < lo[f]
                if hi[j] < lo[f]:
                    gap_lo = hi[j]; gap_hi = lo[f]
                    filled = False
                    for t in range(f + 1, k):
                        if lo[t] <= gap_lo:
                            filled = True; break
                    if not filled:
                        if cp >= gap_hi:
                            s = (cp - gap_hi) / a
                        elif cp <= gap_lo:
                            s = -(gap_lo - cp) / a
                        else:
                            s = 0.0
                        if abs(s) < abs(sgn):
                            sgn = s
                # bearish gap lo[j] > hi[f]
                if lo[j] > hi[f]:
                    gap_lo = hi[f]; gap_hi = lo[j]
                    filled = False
                    for t in range(f + 1, k):
                        if hi[t] >= gap_hi:
                            filled = True; break
                    if not filled:
                        if cp <= gap_lo:
                            s = -(gap_lo - cp) / a
                        elif cp >= gap_hi:
                            s = (cp - gap_hi) / a
                        else:
                            s = 0.0
                        if abs(s) < abs(sgn):
                            sgn = s
            if sgn > clip:
                sgn = clip
            if sgn < -clip:
                sgn = -clip
            dist[k] = sgn
        return dist

    fvg_dist = _fvg_scan(high, low, close, atr_safe, CLIP, LOOKBACK)
    out["m5_dist_to_unfilled_fvg_atr"] = fvg_dist
    out["m5_fvg_active"] = np.exp(-np.abs(fvg_dist) / TAU)

    # recent swing structure (confirmed pivots only — no look-ahead at decision bar).
    # Track the TWO most-recent confirmed swing lows AND highs so we can encode Dow market
    # structure: a HIGHER-LOW (last_sl > prev_sl) = uptrend intact (buy the dip); a LOWER-LOW
    # = trend breaking (falling knife). This is the classic manual-trader dip-vs-knife read.
    sh_mask, sl_mask = _detect_swing_pivots(high, low, SWING_LOOKBACK)
    last_sl_price = np.nan; prev_sl_price = np.nan; last_sl_bar = -1
    last_sh_price = np.nan; prev_sh_price = np.nan
    dist_sl = np.full(n, np.nan); bars_sl = np.full(n, 999.0)
    hl_flag = np.full(n, np.nan); hl_mag = np.full(n, np.nan)
    hh_flag = np.full(n, np.nan); struct_score = np.full(n, np.nan)
    for i in range(n):
        # a pivot at bar p is only CONFIRMED at p+SWING_LOOKBACK (needs future bars) → use confirmation lag
        conf = i - SWING_LOOKBACK
        if conf >= 0 and sl_mask[conf]:
            prev_sl_price = last_sl_price; last_sl_price = low[conf]; last_sl_bar = conf
        if conf >= 0 and sh_mask[conf]:
            prev_sh_price = last_sh_price; last_sh_price = high[conf]
        if last_sl_bar >= 0:
            dist_sl[i] = (close[i] - last_sl_price) / atr_safe[i]
            bars_sl[i] = float(i - last_sl_bar)
        if np.isfinite(last_sl_price) and np.isfinite(prev_sl_price):
            hl_flag[i] = 1.0 if last_sl_price > prev_sl_price else 0.0
            hl_mag[i] = (last_sl_price - prev_sl_price) / atr_safe[i]
        if np.isfinite(last_sh_price) and np.isfinite(prev_sh_price):
            hh_flag[i] = 1.0 if last_sh_price > prev_sh_price else 0.0
        if np.isfinite(hl_flag[i]) and np.isfinite(hh_flag[i]):
            struct_score[i] = hl_flag[i] + hh_flag[i]   # 2 = HH+HL (clean uptrend), 0 = LL+LH (downtrend)
    out["dist_to_swing_low_atr"] = dist_sl
    out["bars_since_swing_low"] = np.clip(bars_sl, 0, 999)
    out["hl_higher_low_flag"] = hl_flag
    out["hl_higher_low_mag_atr"] = hl_mag
    out["hl_higher_high_flag"] = hh_flag
    out["hl_structure_score"] = struct_score

    # ── FIBONACCI retrace zone of the last confirmed swing (from premium_discount = (close-last_low)/
    # (last_high-last_low) ∈ [0,1]). In an uptrend pullback the 50–61.8% retrace buy-zone = pd∈[~0.35,0.55];
    # a deep retrace (pd<0.21 ≈ beyond 78.6%) means the up-swing is likely broken. premium_discount itself
    # is ALREADY in the baseline structural arm — these are the nonlinear zone encodings on top of it.
    pd_arr = out["smc_premium_discount"].to_numpy(np.float64)
    out["fib_dist_382"] = np.abs(pd_arr - 0.382)
    out["fib_dist_500"] = np.abs(pd_arr - 0.500)
    out["fib_dist_618"] = np.abs(pd_arr - 0.618)
    out["fib_in_golden"] = ((pd_arr >= 0.35) & (pd_arr <= 0.55)).astype(np.float64)
    out["fib_deep_retrace"] = (pd_arr < 0.214).astype(np.float64)

    # ── PDH/PDL/PWH/PWL: prior COMPLETED day/week high-low, signed ATR-distance. Leak-safe via a
    # shift of the period-bin index to its "available-from" time (period end), then a backward asof-map:
    # at bar t we only ever see periods that FULLY completed before t. Heavily-watched manual levels.
    tdf = pd.DataFrame({"time": m5["time"].values, "high": high, "low": low}).set_index("time")
    def _prior_period(rule, delta):
        agg = pd.DataFrame({"ph": tdf["high"].resample(rule).max(),
                            "pl": tdf["low"].resample(rule).min()}).dropna()
        agg.index = agg.index + delta            # available only AFTER the period fully completes
        a = pd.merge_asof(pd.DataFrame({"time": m5["time"].values}), agg.reset_index(),
                          on="time", direction="backward")
        return a["ph"].to_numpy(np.float64), a["pl"].to_numpy(np.float64)
    pdh, pdl = _prior_period("1D", pd.Timedelta(days=1))
    pwh, pwl = _prior_period("1W", pd.Timedelta(days=7))
    out["dist_to_pdh_atr"] = (close - pdh) / atr_safe
    out["dist_to_pdl_atr"] = (close - pdl) / atr_safe
    out["dist_to_pwh_atr"] = (close - pwh) / atr_safe
    out["dist_to_pwl_atr"] = (close - pwl) / atr_safe
    return out


# ── Tier-1 NEW feature families (the eye-vs-model candidates under test) ───────
FIB_COLS = ["smc_premium_discount", "fib_dist_382", "fib_dist_500", "fib_dist_618",
            "fib_in_golden", "fib_deep_retrace"]
HL_COLS = ["hl_higher_low_flag", "hl_higher_low_mag_atr", "hl_higher_high_flag",
           "hl_structure_score", "dist_to_swing_low_atr", "bars_since_swing_low"]
PDHL_COLS = ["dist_to_pdh_atr", "dist_to_pdl_atr", "dist_to_pwh_atr", "dist_to_pwl_atr"]


# ── 2. load forward_outcome decision bars (trend cols + label + ts) ───────────
TREND_COLS = ["ema20_slope_canon_v1", "ema100_slope_canon_v1", "pos_vs_ema200_canon_v1",
              "_v1_rsi14_canon_v1", "_v1h1_ema_diff_canon_v1", "_v1h4_ema_diff_canon_v1",
              "_v1_ema_diff_canon_v1", "m15_trend_sign_canon_v2_canon_v1",
              "d1_ema_slope_20_canon_v2_canon_v1", "_v1_r5_canon_v1", "_v1_r24_canon_v1",
              "p_long", "p_short", "margin", "uncertainty_score", "atr_bps"]
LABEL_COLS = [f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1"]
CTX_COLS = ["decision_ts_utc", "side", "trend_regime", "ema100_slope_canon_v1",
            "pos_vs_ema200_canon_v1", "_v1_r5_canon_v1", "_v1_rsi14_canon_v1"]


def load_decisions() -> pd.DataFrame:
    files = sorted(glob.glob(f"{FO_DIR}/*.parquet"))
    need = sorted(set(TREND_COLS + LABEL_COLS + CTX_COLS))
    parts = []
    for f in files:
        parts.append(pd.read_parquet(f, columns=need))
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    return df


def main():
    print("[1/5] building M5 structural features ...")
    m5 = build_m5_structural()
    struct_cols = [c for c in m5.columns if c != "time"]
    print(f"  m5 rows={len(m5)} struct_cols={len(struct_cols)}")

    print("[2/5] loading decision bars ...")
    df = load_decisions()
    print(f"  decision rows={len(df)}")

    # asof-map structural cols onto each decision bar (label <= ts; same as compute_reclaim_cols)
    ts_arr = m5["time"].values.astype("datetime64[ns]").astype(np.int64)
    dec_ns = df["ts"].values.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(ts_arr, dec_ns, side="right") - 1
    ok = idx >= 0
    df = df[ok].reset_index(drop=True); idx = idx[ok]
    for c in struct_cols:
        df[c] = m5[c].to_numpy()[idx]

    # ── 3. DIP-IN-UPTREND population ──────────────────────────────────────────
    # uptrend = regime TREND_UP AND structurally above 200EMA AND ema100 rising;
    # dip = a short-term pullback (recent 5-bar return < 0 OR rsi14 < 45 OR price in lower premium/discount).
    uptrend = (df["trend_regime"] == "TREND_UP") & (df["pos_vs_ema200_canon_v1"] > 0) & (df["ema100_slope_canon_v1"] > 0)
    dip = (df["_v1_r5_canon_v1"] < 0) | (df["_v1_rsi14_canon_v1"] < 45)
    pop = uptrend & dip
    pop = pop.fillna(False)
    df = df[pop].reset_index(drop=True)
    print(f"[3/5] dip-in-uptrend population rows={len(df)}")

    # ── 4. TARGET: a LONG on this dip would WIN ───────────────────────────────
    long_pnl = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float)
    short_pnl = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float)
    y = ((long_pnl > 0) | (long_pnl > short_pnl)).astype(int)
    df["_y"] = y.values
    df = df.dropna(subset=[f"take_now_long_terminal_pnl_at_{K}_v1"]).reset_index(drop=True)
    y = df["_y"].values
    base_rate = float(y.mean())
    print(f"  target base_rate(LONG-win)={base_rate:.4f}  n={len(df)}")

    # LEAKAGE GUARD: assert no forward/label column leaked into either feature set
    FORBIDDEN = ("terminal_pnl", "mfe", "mae", "forward", "time_to_mfe", "wait_", "take_now_",
                 "_outcome", "trough", "bars_to")
    feat_struct = [c for c in struct_cols]
    feat_trend = [c for c in TREND_COLS]
    for fset in (feat_struct, feat_trend, FIB_COLS, HL_COLS, PDHL_COLS):
        for c in fset:
            assert not any(tok in c for tok in FORBIDDEN), f"LEAKAGE: forward col in features: {c}"

    # ── 5. OOT splits + GBM ───────────────────────────────────────────────────
    df = df.sort_values("ts").reset_index(drop=True)
    med = df["ts"].median()
    early = df["ts"] <= med
    late = df["ts"] > med
    print(f"  split @ {med}  early_n={int(early.sum())} late_n={int(late.sum())}")

    def fit_eval(feat_cols, name, train_mask, test_mask):
        Xtr = df.loc[train_mask, feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        Xte = df.loc[test_mask, feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        ytr = df.loc[train_mask, "_y"].values; yte = df.loc[test_mask, "_y"].values
        # subsample train to <=SAMPLE_CAP
        if len(Xtr) > SAMPLE_CAP:
            sel = RNG.choice(len(Xtr), SAMPLE_CAP, replace=False)
            Xtr = Xtr.iloc[sel]; ytr = ytr[sel]
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                             l2_regularization=1.0, min_samples_leaf=200,
                                             early_stopping=True, validation_fraction=0.15,
                                             random_state=42)
        clf.fit(Xtr, ytr)
        auc_is = roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1])
        auc_oot = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]) if len(np.unique(yte)) > 1 else float("nan")
        # permutation importance (cheap: gain proxy via single-feature drop on OOT not needed; use built-in)
        return clf, auc_is, auc_oot, len(Xtr), len(Xte)

    NEW_COMBINED = list(dict.fromkeys(FIB_COLS + HL_COLS + PDHL_COLS))
    # decompose trend into chain-CONVICTION (p_long/p_short/margin/uncertainty = the chain's OWN fitted
    # output — a known fitted-leak suspect) vs PURE-PRICE trend (ema/rsi/returns/atr — no chain output).
    CONV_COLS = ["p_long", "p_short", "margin", "uncertainty_score"]
    TREND_PRICE_COLS = [c for c in feat_trend if c not in CONV_COLS]
    results = {}
    for name, cols in (("structural", feat_struct), ("trend", feat_trend),
                       ("structural+trend", feat_struct + feat_trend),
                       ("conviction_only", CONV_COLS), ("trend_price_only", TREND_PRICE_COLS),
                       ("T1_fib", FIB_COLS), ("T1_higher_low", HL_COLS), ("T1_pdh_pdl", PDHL_COLS),
                       ("T1_new_combined", NEW_COMBINED),
                       ("T1_new+trend", list(dict.fromkeys(NEW_COMBINED + feat_trend))),
                       ("T1_new+price_trend", list(dict.fromkeys(NEW_COMBINED + TREND_PRICE_COLS))),
                       ("T1_new+struct+trend", list(dict.fromkeys(feat_struct + NEW_COMBINED + feat_trend)))):
        clf_el, is_el, oot_el, ntr_el, nte_el = fit_eval(cols, name, early, late)   # fit early, test late
        clf_le, is_le, oot_le, ntr_le, nte_le = fit_eval(cols, name, late, early)   # fit late, test early
        results[name] = dict(in_sample=(is_el + is_le) / 2,
                             oot_fit_early_test_late=oot_el, oot_fit_late_test_early=oot_le,
                             n_train_early=ntr_el, n_test_late=nte_el,
                             n_train_late=ntr_le, n_test_early=nte_le,
                             separable=bool(oot_el >= 0.58 and oot_le >= 0.58))
        print(f"[{name}] in_sample={results[name]['in_sample']:.4f}  "
              f"OOT(fit_early→late)={oot_el:.4f}  OOT(fit_late→early)={oot_le:.4f}  "
              f"separable={results[name]['separable']}")

    # top features (permutation importance on OOT, fit-early model) over BASELINE-struct + NEW Tier-1
    PI_COLS = list(dict.fromkeys(feat_struct + NEW_COMBINED))
    from sklearn.inspection import permutation_importance
    Xtr = df.loc[early, PI_COLS].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    ytr = df.loc[early, "_y"].values
    if len(Xtr) > SAMPLE_CAP:
        sel = RNG.choice(len(Xtr), SAMPLE_CAP, replace=False); Xtr = Xtr.iloc[sel]; ytr = ytr[sel]
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                         l2_regularization=1.0, min_samples_leaf=200,
                                         early_stopping=True, validation_fraction=0.15, random_state=42)
    clf.fit(Xtr, ytr)
    Xte = df.loc[late, PI_COLS].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    yte = df.loc[late, "_y"].values
    nte = min(len(Xte), 50000)
    sel = RNG.choice(len(Xte), nte, replace=False)
    pi = permutation_importance(clf, Xte.iloc[sel], yte[sel], n_repeats=5, random_state=0,
                                scoring="roc_auc", n_jobs=4)
    imp = sorted(zip(PI_COLS, pi.importances_mean), key=lambda x: -x[1])
    print("[top features (baseline-struct + NEW Tier-1) by OOT permutation-AUC importance]")
    for f, v in imp[:12]:
        print(f"  {f:42s} {v:+.5f}")

    out = dict(results=results, base_rate=base_rate, n_total=len(df),
               split_median=str(med), top_struct=[(f, round(float(v), 5)) for f, v in imp[:12]])
    Path("/tmp/oot_pregate_dip_struct.json").write_text(json.dumps(out, indent=2, default=str))
    print("WROTE /tmp/oot_pregate_dip_struct.json")


if __name__ == "__main__":
    main()
