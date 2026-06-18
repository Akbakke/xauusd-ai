"""OOT chart-patterns arm (2026-06-17/18): build NUMERICAL multi-bar GEOMETRIC chart patterns on the
M5 tape (leak-safe, CONFIRMED swing pivots only) and test whether they are INCREMENTAL over a
raw-OHLC baseline for dir_up (= long_terminal_pnl > short_terminal_pnl at K96) on a STRICT OOT split.

Patterns (all use ONLY bars <= t):
  - FVG: signed ATR-distance to nearest UNFILLED 3-bar fair-value gap (completed gaps only) + active.
  - BULL/BEAR FLAG: strong impulse leg (ATR-scaled displacement over impulse window) + tight
    counter-trend consolidation (low range / channel). Signed flag-direction score.
  - CUP-AND-HANDLE: U-shaped base (deep-then-recovered swing-low retrace) + small recent pullback.
  - ASC/DESC TRIANGLE: converging swing highs/lows — flat-top + rising-lows (asc) / flat-bottom +
    falling-highs (desc); slope of recent swing-high line and swing-low line.
  - DOUBLE-TOP/BOTTOM: two similar recent swing extremes (price-match within tol) + position.
  - HEAD-AND-SHOULDERS: 3 swings, middle most extreme (signed: HS top vs inverse-HS bottom).

BASELINE = raw-OHLC features pulled from the forward_outcome (body/wick/range/returns/ema_slope/atr/rsi
+ multi-TF ema_diff) — the V10 transformer already sees per-bar OHLC shape over 96-bar 5-TF sequences,
so a pattern that is just a function of raw OHLC is REDUNDANT. The load-bearing test = baseline+patterns
vs baseline alone, OOT.

Model: HistGradientBoostingClassifier(max_depth=4,max_iter=300,lr=0.05,l2=1.0,min_samples_leaf=200,
early_stopping=True). Strict split: fit ts<=2025-06-01 / TEST ts>=2026-01-01. Subsample train <=200k.
Robustness: fit_early->late and fit_late->early on the median split.
"""
import glob
import json
import numpy as np
import pandas as pd
from pathlib import Path
from numba import njit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from gx1.features.smc_v1 import _detect_swing_pivots, SWING_LOOKBACK

FO_DIR = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
M5_PREBUILT = ("/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
               "xauusd_m5_CANONICAL_V3_2020_2026.parquet")
RNG = np.random.default_rng(20260618)
SAMPLE_CAP = 200_000
K = "K96"

# ── raw-OHLC BASELINE cols (from forward_outcome — what the V10 transformer already sees) ──
BASELINE_COLS = [
    "body_pct_canon_v1", "wick_asym_canon_v1", "_v1_wick_imbalance_canon_v1", "_v1_body_share_1_canon_v1",
    "range_canon_v1", "_v1_range_z_canon_v1", "ret_1_canon_v1", "ret_5_canon_v1", "ret_20_canon_v1",
    "_v1_r5_canon_v1", "_v1_r24_canon_v1", "roc20_canon_v1", "roc100_canon_v1",
    "ema20_slope_canon_v1", "ema100_slope_canon_v1", "pos_vs_ema200_canon_v1",
    "atr_canon_v1", "atr50_canon_v1", "atr_z_canon_v1", "_v1_rsi14_canon_v1", "_v1_rsi2_canon_v1",
    "_v1_ema_diff_canon_v1", "_v1h1_ema_diff_canon_v1", "_v1h4_ema_diff_canon_v1",
]
LABEL_COLS = [f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1"]
CTX_COLS = ["decision_ts_utc"]


# ── numba pattern kernels (all leak-safe: bar t uses only bars <= t) ─────────────────────────
@njit(cache=True)
def _fvg_scan(hi, lo, cp, atr, clip, lookback):
    nn = hi.shape[0]
    dist = np.full(nn, clip)
    for k in range(2, nn):
        c = cp[k]; a = atr[k]; sgn = clip
        j0 = k - 2 - lookback
        if j0 < 0:
            j0 = 0
        for j in range(k - 3, j0 - 1, -1):   # f=j+2 < k → j<=k-3
            f = j + 2
            if hi[j] < lo[f]:                 # bullish gap
                gap_lo = hi[j]; gap_hi = lo[f]; filled = False
                for t in range(f + 1, k):
                    if lo[t] <= gap_lo:
                        filled = True; break
                if not filled:
                    if c >= gap_hi:
                        s = (c - gap_hi) / a
                    elif c <= gap_lo:
                        s = -(gap_lo - c) / a
                    else:
                        s = 0.0
                    if abs(s) < abs(sgn):
                        sgn = s
            if lo[j] > hi[f]:                 # bearish gap
                gap_lo = hi[f]; gap_hi = lo[j]; filled = False
                for t in range(f + 1, k):
                    if hi[t] >= gap_hi:
                        filled = True; break
                if not filled:
                    if c <= gap_lo:
                        s = -(gap_lo - c) / a
                    elif c >= gap_hi:
                        s = (c - gap_hi) / a
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


@njit(cache=True)
def _flag_scan(hi, lo, cp, atr, imp, cons):
    """BULL/BEAR FLAG: strong impulse leg over [t-imp-cons, t-cons] + tight consolidation over [t-cons, t].
    Returns signed score = impulse_disp_atr * (1 - consol_range_ratio), sign = impulse direction.
    Leak-safe: all bars < t? we use [..t] inclusive (current bar) — only past+current, no future."""
    nn = hi.shape[0]
    out = np.zeros(nn)
    for t in range(imp + cons, nn):
        a = atr[t]
        # impulse leg: close displacement over the impulse window (ending cons bars ago)
        i0 = t - imp - cons; i1 = t - cons
        imp_disp = (cp[i1] - cp[i0]) / a
        # impulse range (how clean/strong) — max-min over impulse window
        imp_hi = hi[i0]; imp_lo = lo[i0]
        for j in range(i0, i1 + 1):
            if hi[j] > imp_hi:
                imp_hi = hi[j]
            if lo[j] < imp_lo:
                imp_lo = lo[j]
        imp_range = (imp_hi - imp_lo) / a
        # consolidation: range over the last cons bars (should be TIGHT, counter or sideways)
        c_hi = hi[i1]; c_lo = lo[i1]
        for j in range(i1, t + 1):
            if hi[j] > c_hi:
                c_hi = hi[j]
            if lo[j] < c_lo:
                c_lo = lo[j]
        cons_range = (c_hi - c_lo) / a
        # flag = strong impulse AND tight consolidation relative to impulse range
        if imp_range > 1e-6:
            tightness = 1.0 - min(cons_range / imp_range, 1.0)
        else:
            tightness = 0.0
        # require meaningful impulse; score signed by impulse direction
        if abs(imp_disp) > 1.0 and tightness > 0.0:
            out[t] = imp_disp * tightness
    return out


@njit(cache=True)
def _swing_slopes(hi, lo, sh_idx, sl_idx, atr, lag):
    """ASC/DESC TRIANGLE + DOUBLE/HS patterns from confirmed swing sequences.
    sh_idx/sl_idx: arrays where entry>=0 marks a CONFIRMED swing high/low at that bar (already lagged).
    Returns per-bar: sh_slope, sl_slope (ATR-normalized slope of last-2 swing highs / lows),
    double_top, double_bottom, hns (signed head&shoulders), tri_conv (convergence score)."""
    nn = hi.shape[0]
    sh_slope = np.zeros(nn); sl_slope = np.zeros(nn)
    dtop = np.zeros(nn); dbot = np.zeros(nn); hns = np.zeros(nn); conv = np.zeros(nn)
    # rolling lists of last confirmed swing-high / swing-low (bar, price)
    sh_bars = np.full(8, -1); sh_pr = np.zeros(8)
    sl_bars = np.full(8, -1); sl_pr = np.zeros(8)
    nsh = 0; nsl = 0
    for t in range(nn):
        conf = t - lag
        if conf >= 0:
            if sh_idx[conf] > 0.5:
                # push (shift right, keep last 8)
                for q in range(7, 0, -1):
                    sh_bars[q] = sh_bars[q - 1]; sh_pr[q] = sh_pr[q - 1]
                sh_bars[0] = conf; sh_pr[0] = hi[conf]; nsh += 1
            if sl_idx[conf] > 0.5:
                for q in range(7, 0, -1):
                    sl_bars[q] = sl_bars[q - 1]; sl_pr[q] = sl_pr[q - 1]
                sl_bars[0] = conf; sl_pr[0] = lo[conf]; nsl += 1
        a = atr[t]
        # swing-high slope (last 2 highs): (newer - older) / (bars) / atr
        if sh_bars[0] >= 0 and sh_bars[1] >= 0 and sh_bars[0] != sh_bars[1]:
            sh_slope[t] = (sh_pr[0] - sh_pr[1]) / (sh_bars[0] - sh_bars[1]) / a * 100.0
        if sl_bars[0] >= 0 and sl_bars[1] >= 0 and sl_bars[0] != sl_bars[1]:
            sl_slope[t] = (sl_pr[0] - sl_pr[1]) / (sl_bars[0] - sl_bars[1]) / a * 100.0
        # convergence: asc triangle = sh_slope~0 (flat top) & sl_slope>0 (rising lows) → +;
        # desc = sl_slope~0 & sh_slope<0 → -. encode as sl_slope - sh_slope (rising lows vs falling highs)
        conv[t] = sl_slope[t] - sh_slope[t]
        # double-top: last 2 swing highs within tol AND recent (within window)
        if sh_bars[0] >= 0 and sh_bars[1] >= 0:
            d = abs(sh_pr[0] - sh_pr[1]) / a
            if d < 0.5 and (t - sh_bars[0]) < 60:
                dtop[t] = 1.0 - d / 0.5
        if sl_bars[0] >= 0 and sl_bars[1] >= 0:
            d = abs(sl_pr[0] - sl_pr[1]) / a
            if d < 0.5 and (t - sl_bars[0]) < 60:
                dbot[t] = 1.0 - d / 0.5
        # head & shoulders (3 swing highs: middle highest) signed -1; inverse (3 lows: middle lowest) +1
        if sh_bars[0] >= 0 and sh_bars[1] >= 0 and sh_bars[2] >= 0:
            l, m, r = sh_pr[2], sh_pr[1], sh_pr[0]   # oldest=left, mid=head, newest=right
            if m > l and m > r and abs(l - r) / a < 0.6:
                hns[t] = -((m - max(l, r)) / a)       # bearish HS top
        if sl_bars[0] >= 0 and sl_bars[1] >= 0 and sl_bars[2] >= 0:
            l, m, r = sl_pr[2], sl_pr[1], sl_pr[0]
            if m < l and m < r and abs(l - r) / a < 0.6:
                hns[t] = (min(l, r) - m) / a           # bullish inverse-HS bottom (overrides if set)
    return sh_slope, sl_slope, dtop, dbot, hns, conv


@njit(cache=True)
def _cup_handle(cp, hi, lo, atr, win, handle):
    """CUP-AND-HANDLE: U-shaped base over [t-win, t-handle] (deep mid-trough then recovery to ~start
    level) + small recent pullback (handle) over [t-handle, t]. Returns score (depth*recovery*handle_tight)."""
    nn = cp.shape[0]
    out = np.zeros(nn)
    for t in range(win, nn):
        a = atr[t]
        s0 = t - win; s1 = t - handle
        left = cp[s0]; right = cp[s1]
        # trough = min low in the base
        trough = lo[s0]; tpos = s0
        for j in range(s0, s1 + 1):
            if lo[j] < trough:
                trough = lo[j]; tpos = j
        depth = (min(left, right) - trough) / a
        rim_match = 1.0 - min(abs(left - right) / a / 2.0, 1.0)     # left & right rims similar
        # trough roughly centered (U not L/J): position within [0.25,0.75]
        frac = (tpos - s0) / max(win - handle, 1)
        centered = 1.0 if 0.2 <= frac <= 0.8 else 0.0
        # recovery: current price back near the rim
        recov = 1.0 - min(abs(cp[t] - right) / a, 1.0)
        # handle: small recent pullback (tight range over handle window, below rim)
        h_hi = hi[s1]; h_lo = lo[s1]
        for j in range(s1, t + 1):
            if hi[j] > h_hi:
                h_hi = hi[j]
            if lo[j] < h_lo:
                h_lo = lo[j]
        handle_range = (h_hi - h_lo) / a
        handle_tight = 1.0 if handle_range < depth * 0.5 else 0.0
        if depth > 1.0 and centered > 0.0:
            out[t] = depth * rim_match * recov * (0.5 + 0.5 * handle_tight)
    return out


def build_patterns() -> pd.DataFrame:
    m5 = pd.read_parquet(M5_PREBUILT, columns=["time", "open", "high", "low", "close"])
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time").reset_index(drop=True)
    high = m5["high"].to_numpy(np.float64); low = m5["low"].to_numpy(np.float64)
    close = m5["close"].to_numpy(np.float64)
    tr = pd.concat([(m5["high"] - m5["low"]).abs(),
                    (m5["high"] - m5["close"].shift(1)).abs(),
                    (m5["low"] - m5["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().to_numpy(np.float64)
    atr_s = np.maximum(atr, 1e-3)
    n = len(close)

    out = pd.DataFrame({"time": m5["time"]})

    # FVG
    fvg = _fvg_scan(high, low, close, atr_s, 8.0, 240)
    out["cp_fvg_dist_atr"] = fvg
    out["cp_fvg_active"] = np.exp(-np.abs(fvg) / 1.5)

    # BULL/BEAR FLAG (impulse 12 bars, consolidation 8 bars)
    out["cp_flag_score"] = _flag_scan(high, low, close, atr_s, 12, 8)

    # confirmed-swing-derived geometric patterns (lag = SWING_LOOKBACK confirmation)
    sh_mask, sl_mask = _detect_swing_pivots(high, low, SWING_LOOKBACK)
    sh_f = sh_mask.astype(np.float64); sl_f = sl_mask.astype(np.float64)
    sh_slope, sl_slope, dtop, dbot, hns, conv = _swing_slopes(
        high, low, sh_f, sl_f, atr_s, SWING_LOOKBACK)
    out["cp_sh_slope"] = sh_slope
    out["cp_sl_slope"] = sl_slope
    out["cp_triangle_conv"] = conv
    out["cp_double_top"] = dtop
    out["cp_double_bottom"] = dbot
    out["cp_head_shoulders"] = hns

    # CUP-AND-HANDLE (base window 80 bars, handle 16 bars)
    out["cp_cup_handle"] = _cup_handle(close, high, low, atr_s, 80, 16)
    return out


PATTERN_COLS = ["cp_fvg_dist_atr", "cp_fvg_active", "cp_flag_score", "cp_sh_slope", "cp_sl_slope",
                "cp_triangle_conv", "cp_double_top", "cp_double_bottom", "cp_head_shoulders",
                "cp_cup_handle"]


def load_decisions() -> pd.DataFrame:
    files = sorted(glob.glob(f"{FO_DIR}/*.parquet"))
    need = sorted(set(BASELINE_COLS + LABEL_COLS + CTX_COLS))
    avail = set(pd.read_parquet(files[0]).columns)
    miss = [c for c in need if c not in avail]
    if miss:
        print(f"  WARN missing FO cols (dropping): {miss}")
    need = [c for c in need if c in avail]
    df = pd.concat([pd.read_parquet(f, columns=need) for f in files], ignore_index=True)
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
        return float("nan")
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                         l2_regularization=1.0, min_samples_leaf=200,
                                         early_stopping=True, validation_fraction=0.15, random_state=42)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def main():
    print("[1/4] building chart patterns on M5 tape ...")
    pat = build_patterns()
    # liveness: every pattern must vary on a shuffled sample
    samp = pat[PATTERN_COLS].sample(min(50000, len(pat)), random_state=1)
    live = {c: float(samp[c].std()) for c in PATTERN_COLS}
    dead = [c for c, s in live.items() if not np.isfinite(s) or s < 1e-9]
    print(f"  pattern std (shuffled): { {k: round(v,4) for k,v in live.items()} }")
    print(f"  DEAD patterns: {dead}")

    print("[2/4] loading decision bars + asof-map patterns ...")
    df = load_decisions()
    ts_arr = pat["time"].values.astype("datetime64[ns]").astype(np.int64)
    dec_ns = df["ts"].values.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(ts_arr, dec_ns, side="right") - 1
    ok = idx >= 0
    df = df[ok].reset_index(drop=True); idx = idx[ok]
    for c in PATTERN_COLS:
        df[c] = pat[c].to_numpy()[idx]

    # target dir_up = long_pnl > short_pnl
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float)
    sp = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float)
    df = df[lp.notna() & sp.notna()].reset_index(drop=True)
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float).values
    sp = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float).values
    df["_y"] = (lp > sp).astype(int)
    base_rate = float(df["_y"].mean())
    print(f"  rows={len(df)} dir_up base_rate={base_rate:.4f}")

    # LEAKAGE GUARD
    FORBIDDEN = ("terminal_pnl", "mfe", "mae", "forward", "wait_", "take_now_", "_outcome", "trough", "bars_to")
    for c in BASELINE_COLS + PATTERN_COLS:
        assert not any(tok in c for tok in FORBIDDEN), f"LEAKAGE: {c}"

    base_avail = [c for c in BASELINE_COLS if c in df.columns]
    print(f"  baseline cols available={len(base_avail)}/{len(BASELINE_COLS)}")

    # ── 3. STRICT split (fit<=2025-06-01 / test>=2026-01-01) ──
    df = df.sort_values("ts").reset_index(drop=True)
    train_strict = df["ts"] <= pd.Timestamp("2025-06-01", tz="UTC")
    test_2026 = df["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")
    print(f"[3/4] strict split: train_n={int(train_strict.sum())} test_2026_n={int(test_2026.sum())}")

    base_auc = fit_eval(df, base_avail, train_strict, test_2026)
    plus_auc = fit_eval(df, base_avail + PATTERN_COLS, train_strict, test_2026)
    standalone = fit_eval(df, PATTERN_COLS, train_strict, test_2026)
    dauc = plus_auc - base_auc
    print(f"  baseline_2026_auc      = {base_auc:.5f}")
    print(f"  plus_patterns_2026_auc = {plus_auc:.5f}")
    print(f"  incremental dAUC(2026) = {dauc:+.5f}")
    print(f"  standalone pattern AUC = {standalone:.5f}  (~0.5 = noise)")

    # ── 4. robustness: median split both directions (baseline+patterns vs baseline) ──
    med = df["ts"].median()
    early = df["ts"] <= med; late = df["ts"] > med
    print(f"[4/4] robustness median split @ {med}")
    b_el = fit_eval(df, base_avail, early, late); p_el = fit_eval(df, base_avail + PATTERN_COLS, early, late)
    b_le = fit_eval(df, base_avail, late, early); p_le = fit_eval(df, base_avail + PATTERN_COLS, late, early)
    d_el = p_el - b_el; d_le = p_le - b_le
    print(f"  fit_early->late: base={b_el:.5f} plus={p_el:.5f} dAUC={d_el:+.5f}")
    print(f"  fit_late->early: base={b_le:.5f} plus={p_le:.5f} dAUC={d_le:+.5f}")

    # standalone per-pattern correlation to baseline-residual? quick per-feature standalone AUC
    print("[per-pattern standalone OOT-2026 AUC]")
    per = {}
    for c in PATTERN_COLS:
        a = fit_eval(df, [c], train_strict, test_2026)
        per[c] = a
        print(f"  {c:24s} {a:.4f}")

    clears = bool(np.isfinite(dauc) and dauc > 0 and np.isfinite(d_el) and d_el > 0
                  and np.isfinite(d_le) and d_le > 0 and standalone > 0.5 and not dead)

    out = dict(base_rate=base_rate, n=len(df),
               baseline_2026_auc=base_auc, plus_patterns_2026_auc=plus_auc,
               incremental_dauc_2026=dauc, standalone_pattern_auc=standalone,
               robust_dauc_early_late=d_el, robust_dauc_late_early=d_le,
               per_pattern_standalone=per, dead_patterns=dead, clears=clears)
    Path("/tmp/oot_chart_patterns_arm.json").write_text(json.dumps(out, indent=2, default=str))
    print("WROTE /tmp/oot_chart_patterns_arm.json")
    print(f"CLEARS_GATE={clears}")


if __name__ == "__main__":
    main()
