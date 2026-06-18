"""ARM = dedicated_pattern_model. Tests the user's "dedicated pattern-AI fed into the transformers" idea.

Builds a COMBINED numerical pattern feature set on the M5 tape (LEAK-SAFE: bar t uses only bars <= t):
  CANDLESTICK: doji, hammer, shooting-star, bullish/bearish engulfing, marubozu, inside/outside bar,
               3-bar momentum (3 up / 3 down), body-vs-prev, gap, pin-bar reversal, harami.
  CHART      : swing-pivot structure (HH/HL/LH/LL via CONFIRMED pivots), double-top/bottom proximity,
               breakout above recent N-bar high / below recent N-bar low, channel position, consolidation.
  FIB        : retrace zone of last confirmed swing (premium_discount), 382/500/618 zone distances,
               golden-pocket flag, deep-retrace flag, fib-extension proximity.

Then, STRICTLY incremental over a RAW-OHLC BASELINE (the per-bar OHLC shape the V10 transformer already
sees): body/wick/range/returns/ema_slope/atr/rsi from the forward_outcome (~15 cols).

Three deliverables:
  (1) dedicated pattern-ONLY GBM predicting dir_up (standalone AUC; ~0.5 = noise).
  (2) ENSEMBLE pattern-output with the chain's own p_long: does p_long + pattern_score beat p_long alone OOT?
  (3) incremental dAUC of (baseline + all patterns) over baseline alone, on the 2026 holdout, + robustness
      (fit_early->late, fit_late->early). leak/redundancy check.

Strict split: fit ts<=2025-06-01 / TEST ts>=2026-01-01. Subsample train <=200k. GBM as specified.
"""
import glob
import json
import numpy as np
import pandas as pd
from numba import njit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FO_DIR = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
M5_PREBUILT = ("/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
               "xauusd_m5_CANONICAL_V3_2020_2026.parquet")
RNG = np.random.default_rng(42)
SAMPLE_CAP = 200_000
K = "K96"
SWING_LB = 3  # confirmed-pivot lookback (same as smc_v1 SWING_LOOKBACK)

GBM_KW = dict(max_depth=4, max_iter=300, learning_rate=0.05, l2_regularization=1.0,
              min_samples_leaf=200, early_stopping=True, validation_fraction=0.15, random_state=42)

# ── BASELINE raw-OHLC features already in the FO (what the transformer sees per-bar) ───────
BASELINE_COLS = ["body_pct_canon_v1", "wick_asym_canon_v1", "_v1_wick_imbalance_canon_v1",
                 "_v1_body_share_1_canon_v1", "range_canon_v1", "ret_1_canon_v1", "ret_5_canon_v1",
                 "ret_20_canon_v1", "ema20_slope_canon_v1", "ema100_slope_canon_v1",
                 "pos_vs_ema200_canon_v1", "atr_canon_v1", "_v1_rsi14_canon_v1", "_v1_r5_canon_v1",
                 "_v1_r24_canon_v1"]
LABEL_COLS = [f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1"]
CHAIN_COLS = ["p_long", "p_short", "margin", "uncertainty_score"]


# ── 1. PATTERN feature computation on the M5 tape (leak-safe) ──────────────────────────────
@njit(cache=True)
def _swing_pivots(high, low, lb):
    n = high.shape[0]
    sh = np.zeros(n, np.bool_); sl = np.zeros(n, np.bool_)
    for i in range(lb, n - lb):
        ishigh = True; islow = True
        for k in range(1, lb + 1):
            if high[i] <= high[i - k] or high[i] <= high[i + k]:
                ishigh = False
            if low[i] >= low[i - k] or low[i] >= low[i + k]:
                islow = False
        sh[i] = ishigh; sl[i] = islow
    return sh, sl


@njit(cache=True)
def _chart_structure(high, low, close, atr, sh_mask, sl_mask, lb):
    """Confirmed-pivot Dow structure + breakout/double-top-bottom/channel — all leak-safe.
    A pivot at bar p is confirmed only at p+lb (needs lb future bars). At decision bar i we use the
    pivot only once conf=i-lb >= its position, i.e. we read sh_mask/sl_mask at index conf."""
    n = high.shape[0]
    hl_flag = np.full(n, np.nan); ll_flag = np.full(n, np.nan)
    hh_flag = np.full(n, np.nan); lh_flag = np.full(n, np.nan)
    struct = np.full(n, np.nan)
    dist_sl = np.full(n, np.nan); dist_sh = np.full(n, np.nan)
    bars_sl = np.full(n, 999.0); bars_sh = np.full(n, 999.0)
    dbl_top = np.zeros(n); dbl_bot = np.zeros(n)
    last_sl = np.nan; prev_sl = np.nan; last_sl_bar = -1
    last_sh = np.nan; prev_sh = np.nan; last_sh_bar = -1
    for i in range(n):
        conf = i - lb
        if conf >= 0:
            if sl_mask[conf]:
                prev_sl = last_sl; last_sl = low[conf]; last_sl_bar = conf
            if sh_mask[conf]:
                prev_sh = last_sh; last_sh = high[conf]; last_sh_bar = conf
        a = atr[i]
        if a < 1e-6:
            a = 1e-6
        if last_sl_bar >= 0:
            dist_sl[i] = (close[i] - last_sl) / a
            bars_sl[i] = float(i - last_sl_bar)
        if last_sh_bar >= 0:
            dist_sh[i] = (close[i] - last_sh) / a
            bars_sh[i] = float(i - last_sh_bar)
        if last_sl == last_sl and prev_sl == prev_sl:
            hl_flag[i] = 1.0 if last_sl > prev_sl else 0.0
            ll_flag[i] = 1.0 if last_sl < prev_sl else 0.0
            # double-bottom: two recent swing lows within 0.5 ATR of each other
            if abs(last_sl - prev_sl) / a < 0.5:
                dbl_bot[i] = 1.0
        if last_sh == last_sh and prev_sh == prev_sh:
            hh_flag[i] = 1.0 if last_sh > prev_sh else 0.0
            lh_flag[i] = 1.0 if last_sh < prev_sh else 0.0
            if abs(last_sh - prev_sh) / a < 0.5:
                dbl_top[i] = 1.0
        if hl_flag[i] == hl_flag[i] and hh_flag[i] == hh_flag[i]:
            struct[i] = hl_flag[i] + hh_flag[i]  # 2=HH+HL clean up, 0=LL+LH clean down
    return (hl_flag, ll_flag, hh_flag, lh_flag, struct, dist_sl, dist_sh, bars_sl, bars_sh,
            dbl_top, dbl_bot)


def build_m5_patterns() -> pd.DataFrame:
    m5 = pd.read_parquet(M5_PREBUILT, columns=["time", "open", "high", "low", "close"])
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time").reset_index(drop=True)
    o = m5["open"].to_numpy(np.float64); h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64); c = m5["close"].to_numpy(np.float64)
    n = len(c)
    # SMA-14 TR atr (build-truth)
    tr = pd.concat([(m5["high"] - m5["low"]).abs(),
                    (m5["high"] - m5["close"].shift(1)).abs(),
                    (m5["low"] - m5["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().to_numpy(np.float64)
    atr_safe = np.maximum(atr, 1e-6)
    rng = np.maximum(h - l, 1e-9)
    body = c - o
    abody = np.abs(body)
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    out = pd.DataFrame({"time": m5["time"]})

    # ── CANDLESTICK patterns (single + 2/3-bar; all leak-safe, only bars <= t) ──
    body_frac = abody / rng
    out["cdl_doji"] = (body_frac < 0.1).astype(np.float64)
    out["cdl_hammer"] = ((lower_wick > 2.0 * abody) & (upper_wick < abody) & (body_frac < 0.5)).astype(np.float64)
    out["cdl_shooting_star"] = ((upper_wick > 2.0 * abody) & (lower_wick < abody) & (body_frac < 0.5)).astype(np.float64)
    out["cdl_marubozu"] = (body_frac > 0.9).astype(np.float64)
    out["cdl_upper_wick_frac"] = upper_wick / rng
    out["cdl_lower_wick_frac"] = lower_wick / rng
    out["cdl_body_frac"] = body_frac
    out["cdl_body_dir"] = np.sign(body)
    # prev-bar relations (shifted, leak-safe)
    po = np.roll(o, 1); pc = np.roll(c, 1); ph = np.roll(h, 1); pl = np.roll(l, 1)
    po[0] = o[0]; pc[0] = c[0]; ph[0] = h[0]; pl[0] = l[0]
    pbody_lo = np.minimum(po, pc); pbody_hi = np.maximum(po, pc)
    bull_eng = (c > o) & (pc < po) & (c >= pbody_hi) & (o <= pbody_lo)
    bear_eng = (c < o) & (pc > po) & (o >= pbody_hi) & (c <= pbody_lo)
    out["cdl_bull_engulf"] = bull_eng.astype(np.float64)
    out["cdl_bear_engulf"] = bear_eng.astype(np.float64)
    # harami (inside body)
    out["cdl_bull_harami"] = ((c > o) & (pc < po) & (np.maximum(o, c) <= pbody_hi) & (np.minimum(o, c) >= pbody_lo)).astype(np.float64)
    out["cdl_bear_harami"] = ((c < o) & (pc > po) & (np.maximum(o, c) <= pbody_hi) & (np.minimum(o, c) >= pbody_lo)).astype(np.float64)
    # inside / outside bar
    out["cdl_inside_bar"] = ((h <= ph) & (l >= pl)).astype(np.float64)
    out["cdl_outside_bar"] = ((h >= ph) & (l <= pl)).astype(np.float64)
    # gap (open vs prev close, ATR-scaled)
    out["cdl_gap_atr"] = (o - pc) / atr_safe
    # 3-bar momentum (3 consecutive up / down closes)
    up1 = c > pc
    up2 = pc > np.roll(c, 2); up2[:2] = False
    up3 = np.roll(c, 2) > np.roll(c, 3); up3[:3] = False
    out["cdl_3up"] = (up1 & up2 & up3).astype(np.float64)
    out["cdl_3down"] = ((~up1) & (~up2) & (~up3)).astype(np.float64)
    # pin-bar reversal at extremes (long lower wick after down move = bullish pin)
    out["cdl_bull_pin"] = ((lower_wick > 0.6 * rng) & (c < np.roll(c, 5))).astype(np.float64)
    out["cdl_bear_pin"] = ((upper_wick > 0.6 * rng) & (c > np.roll(c, 5))).astype(np.float64)
    out["cdl_body_vs_atr"] = abody / atr_safe

    # ── CHART patterns (confirmed swing pivots; breakout; channel) ──
    sh_mask, sl_mask = _swing_pivots(h, l, SWING_LB)
    (hl, ll, hh, lh, struct, dist_sl, dist_sh, bars_sl, bars_sh,
     dbl_top, dbl_bot) = _chart_structure(h, l, c, atr_safe, sh_mask, sl_mask, SWING_LB)
    out["chart_hl_flag"] = hl; out["chart_ll_flag"] = ll
    out["chart_hh_flag"] = hh; out["chart_lh_flag"] = lh
    out["chart_structure_score"] = struct
    out["chart_dist_swing_low_atr"] = dist_sl
    out["chart_dist_swing_high_atr"] = dist_sh
    out["chart_bars_since_swing_low"] = np.clip(bars_sl, 0, 999)
    out["chart_bars_since_swing_high"] = np.clip(bars_sh, 0, 999)
    out["chart_double_top"] = dbl_top
    out["chart_double_bottom"] = dbl_bot
    # breakout above/below recent N-bar high/low (use bars < t: shift the rolling extreme by 1)
    for N in (20, 50):
        roll_hi = pd.Series(h).rolling(N).max().shift(1).to_numpy()
        roll_lo = pd.Series(l).rolling(N).min().shift(1).to_numpy()
        out[f"chart_breakout_hi_{N}"] = ((c > roll_hi)).astype(np.float64)
        out[f"chart_breakout_lo_{N}"] = ((c < roll_lo)).astype(np.float64)
        # channel position: where is close within recent N-bar range [0,1]
        denom = np.maximum(roll_hi - roll_lo, 1e-9)
        out[f"chart_channel_pos_{N}"] = np.clip((c - roll_lo) / denom, 0, 1)
        out[f"chart_dist_to_hi_{N}_atr"] = (c - roll_hi) / atr_safe
        out[f"chart_dist_to_lo_{N}_atr"] = (c - roll_lo) / atr_safe
    # consolidation: recent range compression (last 20-bar range / last 100-bar range)
    r20 = pd.Series(h).rolling(20).max().shift(1).to_numpy() - pd.Series(l).rolling(20).min().shift(1).to_numpy()
    r100 = pd.Series(h).rolling(100).max().shift(1).to_numpy() - pd.Series(l).rolling(100).min().shift(1).to_numpy()
    out["chart_range_comp_20_100"] = r20 / np.maximum(r100, 1e-9)

    # ── FIB patterns: retrace zone of last confirmed swing (premium_discount style) ──
    # premium_discount = (close - last_swing_low) / (last_swing_high - last_swing_low), leak-safe via conf.
    pd_arr = np.full(n, np.nan)
    last_sl_p = np.nan; last_sh_p = np.nan
    for i in range(n):
        conf = i - SWING_LB
        if conf >= 0:
            if sl_mask[conf]:
                last_sl_p = l[conf]
            if sh_mask[conf]:
                last_sh_p = h[conf]
        if last_sl_p == last_sl_p and last_sh_p == last_sh_p and (last_sh_p - last_sl_p) > 1e-9:
            pd_arr[i] = (c[i] - last_sl_p) / (last_sh_p - last_sl_p)
    out["fib_premium_discount"] = pd_arr
    out["fib_dist_382"] = np.abs(pd_arr - 0.382)
    out["fib_dist_500"] = np.abs(pd_arr - 0.500)
    out["fib_dist_618"] = np.abs(pd_arr - 0.618)
    out["fib_in_golden"] = ((pd_arr >= 0.35) & (pd_arr <= 0.55)).astype(np.float64)
    out["fib_deep_retrace"] = (pd_arr < 0.214).astype(np.float64)
    out["fib_extension"] = (pd_arr > 1.0).astype(np.float64)  # broke above swing high (1.0 ext)
    out["fib_below_low"] = (pd_arr < 0.0).astype(np.float64)  # broke below swing low

    return out


PATTERN_GROUPS = {
    "candlestick": None,  # filled at runtime (cdl_*)
    "chart": None,        # chart_*
    "fib": None,          # fib_*
}


def load_decisions() -> pd.DataFrame:
    files = sorted(glob.glob(f"{FO_DIR}/*.parquet"))
    need = sorted(set(BASELINE_COLS + LABEL_COLS + CHAIN_COLS + ["decision_ts_utc"]))
    parts = [pd.read_parquet(f, columns=need) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    return df


def fit_eval(df, feat_cols, train_mask, test_mask):
    Xtr = df.loc[train_mask, feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    Xte = df.loc[test_mask, feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    ytr = df.loc[train_mask, "_y"].values; yte = df.loc[test_mask, "_y"].values
    if len(Xtr) > SAMPLE_CAP:
        sel = RNG.choice(len(Xtr), SAMPLE_CAP, replace=False)
        Xtr = Xtr.iloc[sel]; ytr = ytr[sel]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None, float("nan"), float("nan")
    clf = HistGradientBoostingClassifier(**GBM_KW)
    clf.fit(Xtr, ytr)
    auc_oot = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    return clf, clf, auc_oot


def main():
    print("[1/4] building M5 patterns ...")
    m5 = build_m5_patterns()
    pat_cols = [c for c in m5.columns if c != "time"]
    cdl = [c for c in pat_cols if c.startswith("cdl_")]
    chart = [c for c in pat_cols if c.startswith("chart_")]
    fib = [c for c in pat_cols if c.startswith("fib_")]
    print(f"  m5 rows={len(m5)} pattern_cols={len(pat_cols)} (cdl={len(cdl)} chart={len(chart)} fib={len(fib)})")

    print("[2/4] loading decisions + asof-map ...")
    df = load_decisions()
    ts_arr = m5["time"].values.astype("datetime64[ns]").astype(np.int64)
    dec_ns = df["ts"].values.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(ts_arr, dec_ns, side="right") - 1
    ok = idx >= 0
    df = df[ok].reset_index(drop=True); idx = idx[ok]
    for col in pat_cols:
        df[col] = m5[col].to_numpy()[idx]
    print(f"  decision rows={len(df)}")

    # ── TARGET: dir_up = long_terminal_pnl > short_terminal_pnl (the spec target) ──
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float)
    sp = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float)
    df = df[lp.notna() & sp.notna()].reset_index(drop=True)
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float).values
    sp = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float).values
    df["_y"] = (lp > sp).astype(int)
    base_rate = float(df["_y"].mean())
    print(f"  dir_up base_rate={base_rate:.4f} n={len(df)}")

    # LEAKAGE GUARD
    FORBIDDEN = ("terminal_pnl", "mfe", "mae", "forward", "time_to", "wait_", "take_now_",
                 "_outcome", "trough", "bars_to", "_pnl")
    for fset in (BASELINE_COLS, pat_cols, CHAIN_COLS):
        for c in fset:
            assert not any(tok in c for tok in FORBIDDEN), f"LEAKAGE: {c}"

    # ── SPLITS ──
    df = df.sort_values("ts").reset_index(drop=True)
    train_main = df["ts"] <= pd.Timestamp("2025-06-01", tz="UTC")   # strict-split train
    test_2026 = df["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")    # strict 2026 holdout
    med = df["ts"].median()
    early = df["ts"] <= med
    late = df["ts"] > med
    print(f"  strict: train_n={int(train_main.sum())} test2026_n={int(test_2026.sum())}  "
          f"robust split@{med} early_n={int(early.sum())} late_n={int(late.sum())}")

    ALL_PAT = cdl + chart + fib

    # ====================================================================================
    # DELIVERABLE 1: dedicated pattern-ONLY GBM standalone AUC on 2026 (~0.5 = noise)
    # ====================================================================================
    print("\n[3/4] DELIVERABLE 1 — standalone pattern-only AUC (dir_up, strict 2026 holdout):")
    pat_clf, _, standalone_auc = fit_eval(df, ALL_PAT, train_main, test_2026)
    print(f"  standalone all-patterns AUC(2026) = {standalone_auc:.4f}")
    _, _, cdl_auc = fit_eval(df, cdl, train_main, test_2026)
    _, _, chart_auc = fit_eval(df, chart, train_main, test_2026)
    _, _, fib_auc = fit_eval(df, fib, train_main, test_2026)
    print(f"    candlestick-only={cdl_auc:.4f}  chart-only={chart_auc:.4f}  fib-only={fib_auc:.4f}")

    # ====================================================================================
    # DELIVERABLE 3: incremental dAUC of baseline+patterns over baseline alone (2026) + robustness
    # ====================================================================================
    print("\n[3/4] DELIVERABLE 3 — incremental over RAW-OHLC baseline:")
    _, _, base_2026 = fit_eval(df, BASELINE_COLS, train_main, test_2026)
    _, _, plus_2026 = fit_eval(df, BASELINE_COLS + ALL_PAT, train_main, test_2026)
    dauc_2026 = plus_2026 - base_2026
    print(f"  baseline AUC(2026)={base_2026:.4f}  baseline+patterns AUC(2026)={plus_2026:.4f}  "
          f"dAUC(2026)={dauc_2026:+.4f}")
    # robustness: fit_early->late, fit_late->early
    _, _, base_el = fit_eval(df, BASELINE_COLS, early, late)
    _, _, plus_el = fit_eval(df, BASELINE_COLS + ALL_PAT, early, late)
    _, _, base_le = fit_eval(df, BASELINE_COLS, late, early)
    _, _, plus_le = fit_eval(df, BASELINE_COLS + ALL_PAT, late, early)
    robust_el = plus_el - base_el
    robust_le = plus_le - base_le
    print(f"  robust fit_early->late: base={base_el:.4f} plus={plus_el:.4f} dAUC={robust_el:+.4f}")
    print(f"  robust fit_late->early: base={base_le:.4f} plus={plus_le:.4f} dAUC={robust_le:+.4f}")

    # per-group incremental over baseline (which family, if any, adds?)
    for gname, gcols in (("+candlestick", cdl), ("+chart", chart), ("+fib", fib)):
        _, _, g26 = fit_eval(df, BASELINE_COLS + gcols, train_main, test_2026)
        print(f"  baseline{gname} AUC(2026)={g26:.4f}  dAUC={g26 - base_2026:+.4f}")

    # ====================================================================================
    # DELIVERABLE 2: ENSEMBLE pattern-output with chain p_long — beat p_long alone on 2026?
    # ====================================================================================
    print("\n[4/4] DELIVERABLE 2 — does a pattern-expert ADD to the chain's own read (p_long)?:")
    # chain p_long as a standalone direction score
    p_long_oot = df.loc[test_2026, "p_long"].astype(float).values
    y_oot = df.loc[test_2026, "_y"].values
    auc_plong = roc_auc_score(y_oot, p_long_oot) if len(np.unique(y_oot)) > 1 else float("nan")
    # ensemble A: GBM on [p_long, p_short, margin, uncertainty] alone (chain read) vs + pattern_score
    _, _, auc_chain = fit_eval(df, CHAIN_COLS, train_main, test_2026)
    # build pattern_score OOS: train pattern GBM on train_main, predict on ALL rows -> use as feature
    if pat_clf is not None:
        # OOS pattern score for ensemble: refit on train_main, score test (already have on test);
        # to feed as a feature without train-leak, score the WHOLE df with the train_main-fit model.
        Xall = df[ALL_PAT].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        df["pattern_score"] = pat_clf.predict_proba(Xall)[:, 1]
        _, _, auc_chain_plus_pat = fit_eval(df, CHAIN_COLS + ["pattern_score"], train_main, test_2026)
        # also raw-OHLC baseline patterns score added to chain (the honest "expert head" feed)
        _, _, auc_chain_plus_allpat = fit_eval(df, CHAIN_COLS + ALL_PAT, train_main, test_2026)
    else:
        auc_chain_plus_pat = float("nan"); auc_chain_plus_allpat = float("nan")
    print(f"  p_long-only AUC(2026)={auc_plong:.4f}")
    print(f"  chain-read GBM (p_long/p_short/margin/uncert) AUC(2026)={auc_chain:.4f}")
    print(f"  chain-read + pattern_score AUC(2026)={auc_chain_plus_pat:.4f}  "
          f"dAUC={auc_chain_plus_pat - auc_chain:+.4f}")
    print(f"  chain-read + ALL-pattern-cols AUC(2026)={auc_chain_plus_allpat:.4f}  "
          f"dAUC={auc_chain_plus_allpat - auc_chain:+.4f}")

    # redundancy check: how much of pattern_score is explained by baseline OHLC?
    # if a GBM on BASELINE predicts pattern_score with high R2, patterns ARE the OHLC the transformer sees.
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    tr = df[train_main]
    Xb = tr[BASELINE_COLS].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    if "pattern_score" in df.columns and len(Xb) > 1000:
        sel = RNG.choice(len(Xb), min(SAMPLE_CAP, len(Xb)), replace=False)
        reg = HistGradientBoostingRegressor(max_depth=4, max_iter=200, learning_rate=0.05,
                                            l2_regularization=1.0, random_state=42)
        reg.fit(Xb.iloc[sel], tr["pattern_score"].values[sel])
        te = df[test_2026]
        Xbte = te[BASELINE_COLS].astype(np.float32).replace([np.inf, -np.inf], np.nan)
        r2_redund = r2_score(te["pattern_score"].values, reg.predict(Xbte))
    else:
        r2_redund = float("nan")
    print(f"\n  REDUNDANCY: R2(baseline-OHLC -> pattern_score) on 2026 = {r2_redund:.4f} "
          f"(high = patterns are just raw-OHLC the transformer already sees)")

    out = dict(base_rate=base_rate, n_total=len(df),
               n_train=int(train_main.sum()), n_test_2026=int(test_2026.sum()),
               standalone_all=round(standalone_auc, 4), standalone_cdl=round(cdl_auc, 4),
               standalone_chart=round(chart_auc, 4), standalone_fib=round(fib_auc, 4),
               baseline_2026=round(base_2026, 4), plus_2026=round(plus_2026, 4),
               dauc_2026=round(dauc_2026, 4), robust_el=round(robust_el, 4), robust_le=round(robust_le, 4),
               auc_plong=round(auc_plong, 4), auc_chain=round(auc_chain, 4),
               auc_chain_plus_pat=round(auc_chain_plus_pat, 4),
               auc_chain_plus_allpat=round(auc_chain_plus_allpat, 4),
               r2_redundancy=round(r2_redund, 4),
               pattern_cols=ALL_PAT)
    import pathlib
    pathlib.Path("/tmp/oot_pattern_ai_arm.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nWROTE /tmp/oot_pattern_ai_arm.json")
    print(json.dumps({k: v for k, v in out.items() if k != "pattern_cols"}, indent=2))


if __name__ == "__main__":
    main()
