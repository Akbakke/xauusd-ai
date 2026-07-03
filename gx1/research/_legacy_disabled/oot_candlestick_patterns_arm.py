"""OOT INCREMENTAL test — ARM = numerical CANDLESTICK patterns (leak-safe, bars<=t).

Question (honest): the V10 transformer ALREADY sees per-bar OHLC shape (body_pct, wick imbalance,
close-open, range, returns, ema/atr/rsi) over 96-bar 5-TF sequences. Do EXPLICIT candlestick patterns
(doji, engulfing, hammer/shooting-star, 3-soldiers/3-crows, morning/evening star, harami, marubozu,
piercing/dark-cloud) add anything BEYOND that raw OHLC, OOT — or are they redundant?

Method (mirrors gx1/research/oot_pregate_dip_reversal_structural.py):
  - build numeric candlestick features on the M5 canonical tape; each uses ONLY bars<=t (no future).
  - asof-map onto each FO decision bar (searchsorted side='right'-1).
  - BASELINE = ~15 raw-OHLC cols already in the FO (what the transformer already sees).
  - TEST = BASELINE+patterns vs BASELINE alone, OOT. Targets: dir_up + take_win.
  - splits: STRICT (fit ts<=2025-06-01 / TEST ts>=2026-01-01) = the load-bearing 2026 holdout,
    + robustness (fit early/test late AND fit late/test early on the median split).
  - model: HistGradientBoostingClassifier(depth4, iter300, lr.05, l2=1.0, leaf200, early_stop, seed42).
  - leak guard: assert no forward/label token in any feature; standalone-pattern AUC (~0.5 = noise).
"""
import glob, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FO_DIR = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
M5_PREBUILT = ("/home/andre2/GX1_DATA/data/data/prebuilt/CANONICAL_V3_PREBUILT/"
               "xauusd_m5_CANONICAL_V3_2020_2026.parquet")
RNG = np.random.default_rng(42)
SAMPLE_CAP = 200_000
K = "K96"

# ── BASELINE = raw-OHLC features the transformer already sees (~15 cols, from the FO) ─────────────
BASELINE_COLS = [
    "body_pct_canon_v1", "wick_asym_canon_v1", "_v1_body_share_1_canon_v1", "_v1_wick_imbalance_canon_v1",
    "range_canon_v1", "_v1_range_z_canon_v1", "ret_1_canon_v1", "ret_5_canon_v1", "ret_20_canon_v1",
    "ema20_slope_canon_v1", "ema100_slope_canon_v1", "atr_canon_v1", "_v1_rsi14_canon_v1",
    "_v1_lower_tr_canon_v1", "_v1_tr_1_over_atr_14_canon_v1",
]

LABEL_COLS = [f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1"]


# ── 1. NUMERIC CANDLESTICK PATTERNS on the M5 tape (each strength uses ONLY bars<=t) ──────────────
def build_m5_candles() -> pd.DataFrame:
    m5 = pd.read_parquet(M5_PREBUILT, columns=["time", "open", "high", "low", "close"])
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time").reset_index(drop=True)
    o = m5["open"].to_numpy(np.float64); h = m5["high"].to_numpy(np.float64)
    l = m5["low"].to_numpy(np.float64);  c = m5["close"].to_numpy(np.float64)
    n = len(o)

    rng = np.maximum(h - l, 1e-9)            # full bar range (hi-lo), floored
    body = c - o                             # signed body
    abody = np.abs(body)
    body_pct = abody / rng                   # body as fraction of range  [0,1]
    upper_wick = h - np.maximum(o, c)        # upper shadow
    lower_wick = np.minimum(o, c) - l        # lower shadow
    up_pct = upper_wick / rng
    lo_pct = lower_wick / rng
    bull = (c > o).astype(np.float64)        # 1 bull, 0 bear (doji counts as not-bull)
    bear = (c < o).astype(np.float64)
    # ATR(14) on this tape for scale-normalised "is this a big bar" gating (bars<=t)
    tr = pd.concat([(m5["high"] - m5["low"]).abs(),
                    (m5["high"] - m5["close"].shift(1)).abs(),
                    (m5["low"] - m5["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().to_numpy(np.float64)
    atr_s = np.maximum(atr, 1e-9)
    rng_atr = rng / atr_s                     # bar size in ATR units (significance gate)

    def sh(a, k):                             # shift forward k bars (prior bar = bars<=t safe)
        out = np.empty_like(a); out[:k] = np.nan; out[k:] = a[:-k]; return out

    o1, h1, l1, c1 = sh(o, 1), sh(h, 1), sh(l, 1), sh(c, 1)
    o2, c2 = sh(o, 2), sh(c, 2)
    rng1 = np.maximum(h1 - l1, 1e-9)
    body1 = c1 - o1; abody1 = np.abs(body1)
    bull1 = (c1 > o1).astype(np.float64); bear1 = (c1 < o1).astype(np.float64)
    body1_pct = abody1 / rng1

    out = pd.DataFrame({"time": m5["time"]})

    # DOJI — small body relative to range; strength = 1 - body_pct (higher = more doji-like), gated big-enough bar
    out["cs_doji"] = np.clip(1.0 - body_pct / 0.10, 0.0, 1.0) * np.clip(rng_atr, 0.0, 1.0)

    # MARUBOZU — almost no wick; strength = body_pct (near 1) signed by direction
    marub = np.clip((body_pct - 0.85) / 0.15, 0.0, 1.0)
    out["cs_marubozu_bull"] = marub * bull
    out["cs_marubozu_bear"] = marub * bear

    # HAMMER — long lower wick, small upper wick, small body near top (bullish reversal strength, continuous)
    out["cs_hammer"] = (np.clip((lo_pct - 0.5) / 0.5, 0.0, 1.0)
                        * np.clip(1.0 - up_pct / 0.15, 0.0, 1.0)
                        * np.clip(1.0 - body_pct / 0.35, 0.0, 1.0))
    # SHOOTING STAR — long upper wick, small lower wick, small body near bottom (bearish reversal)
    out["cs_shooting_star"] = (np.clip((up_pct - 0.5) / 0.5, 0.0, 1.0)
                               * np.clip(1.0 - lo_pct / 0.15, 0.0, 1.0)
                               * np.clip(1.0 - body_pct / 0.35, 0.0, 1.0))
    # signed net wick-reversal (hammer positive / star negative) — single continuous lever
    out["cs_wick_reversal_signed"] = out["cs_hammer"] - out["cs_shooting_star"]

    # ENGULFING — current body fully engulfs prior opposite-color body; strength = size ratio (capped), gated dir flip
    body_top = np.maximum(o, c); body_bot = np.minimum(o, c)
    body_top1 = np.maximum(o1, c1); body_bot1 = np.minimum(o1, c1)
    engulf_geom = ((body_top >= body_top1) & (body_bot <= body_bot1)).astype(np.float64)
    size_ratio = np.clip(abody / np.maximum(abody1, 1e-9) - 1.0, 0.0, 3.0) / 3.0
    out["cs_engulf_bull"] = engulf_geom * bull * bear1 * (0.4 + 0.6 * size_ratio)   # bull engulfs prior bear
    out["cs_engulf_bear"] = engulf_geom * bear * bull1 * (0.4 + 0.6 * size_ratio)   # bear engulfs prior bull
    out["cs_engulf_signed"] = out["cs_engulf_bull"] - out["cs_engulf_bear"]

    # HARAMI — current body INSIDE prior (larger) body (inside bar); strength = how deep inside, gated opposite color
    harami_geom = ((body_top <= body_top1) & (body_bot >= body_bot1) & (abody1 > 1e-9)).astype(np.float64)
    inside_frac = np.clip(1.0 - abody / np.maximum(abody1, 1e-9), 0.0, 1.0)
    out["cs_harami_bull"] = harami_geom * bull * bear1 * inside_frac   # bull harami in prior bear
    out["cs_harami_bear"] = harami_geom * bear * bull1 * inside_frac   # bear harami in prior bull

    # THREE WHITE SOLDIERS / THREE BLACK CROWS — 3 consecutive same-dir closes, each close > (or <) prior close,
    #   each a real body. strength = mean body_pct over the 3 (conviction).
    up3 = ((c > c1) & (c1 > c2) & (bull > 0) & (bull1 > 0) & ((c2 > o2))).astype(np.float64)
    dn3 = ((c < c1) & (c1 < c2) & (bear > 0) & (bear1 > 0) & ((c2 < o2))).astype(np.float64)
    body2_pct = np.abs(c2 - o2) / np.maximum(sh(rng, 2), 1e-9)
    mean_body3 = (body_pct + body1_pct + body2_pct) / 3.0
    out["cs_three_white_soldiers"] = up3 * mean_body3
    out["cs_three_black_crows"] = dn3 * mean_body3

    # MORNING / EVENING STAR — 3-bar reversal: big bar1, small-body bar2 (star), strong bar3 in opposite dir.
    #   morning: bar(t-2) big bear, bar(t-1) small body gapping down, bar(t) big bull closing into bar(t-2) body.
    big1 = (np.abs(c2 - o2) / np.maximum(sh(rng, 2), 1e-9))       # body_pct of t-2
    small_star = np.clip(1.0 - body1_pct / 0.30, 0.0, 1.0)        # t-1 small body
    midpoint2 = (o2 + c2) / 2.0
    morning = ((c2 < o2) & (c > o) & (c >= midpoint2)).astype(np.float64) * big1 * small_star * body_pct
    evening = ((c2 > o2) & (c < o) & (c <= midpoint2)).astype(np.float64) * big1 * small_star * body_pct
    out["cs_morning_star"] = morning
    out["cs_evening_star"] = evening

    # PIERCING / DARK-CLOUD — 2-bar: piercing = prior bear, current bull opens below prior low/close and closes
    #   above prior body midpoint (but below prior open). dark-cloud = mirror. strength = penetration depth.
    mid1 = (o1 + c1) / 2.0
    pierce_pen = np.clip((c - mid1) / np.maximum(abody1, 1e-9), 0.0, 1.0)
    piercing = ((bear1 > 0) & (bull > 0) & (o < c1) & (c > mid1) & (c < o1)).astype(np.float64) * pierce_pen
    cloud_pen = np.clip((mid1 - c) / np.maximum(abody1, 1e-9), 0.0, 1.0)
    darkcloud = ((bull1 > 0) & (bear > 0) & (o > c1) & (c < mid1) & (c > o1)).astype(np.float64) * cloud_pen
    out["cs_piercing"] = piercing
    out["cs_dark_cloud"] = darkcloud

    # NET signed candlestick reversal score (all bullish patterns +, bearish -) — one aggregate lever
    out["cs_net_bull_signal"] = (out["cs_engulf_bull"] + out["cs_hammer"] + out["cs_harami_bull"]
                                 + out["cs_three_white_soldiers"] + out["cs_morning_star"]
                                 + out["cs_piercing"] + out["cs_marubozu_bull"])
    out["cs_net_bear_signal"] = (out["cs_engulf_bear"] + out["cs_shooting_star"] + out["cs_harami_bear"]
                                 + out["cs_three_black_crows"] + out["cs_evening_star"]
                                 + out["cs_dark_cloud"] + out["cs_marubozu_bear"])
    out["cs_net_signed"] = out["cs_net_bull_signal"] - out["cs_net_bear_signal"]
    return out


PATTERN_COLS = [
    "cs_doji", "cs_marubozu_bull", "cs_marubozu_bear", "cs_hammer", "cs_shooting_star",
    "cs_wick_reversal_signed", "cs_engulf_bull", "cs_engulf_bear", "cs_engulf_signed",
    "cs_harami_bull", "cs_harami_bear", "cs_three_white_soldiers", "cs_three_black_crows",
    "cs_morning_star", "cs_evening_star", "cs_piercing", "cs_dark_cloud",
    "cs_net_bull_signal", "cs_net_bear_signal", "cs_net_signed",
]


def load_decisions() -> pd.DataFrame:
    files = sorted(glob.glob(f"{FO_DIR}/*.parquet"))
    need = sorted(set(BASELINE_COLS + LABEL_COLS + ["decision_ts_utc"]))
    df = pd.concat([pd.read_parquet(f, columns=need) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    return df


def fit_auc(df, feat_cols, ycol, train_mask, test_mask):
    Xtr = df.loc[train_mask, feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    Xte = df.loc[test_mask, feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    ytr = df.loc[train_mask, ycol].values; yte = df.loc[test_mask, ycol].values
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
    print("[1/4] building M5 candlestick patterns ...")
    m5 = build_m5_candles()
    print(f"  m5 rows={len(m5)} pattern_cols={len(PATTERN_COLS)}")

    print("[2/4] loading decision bars + asof-map ...")
    df = load_decisions()
    ts_arr = m5["time"].values.astype("datetime64[ns]").astype(np.int64)
    dec_ns = df["ts"].values.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(ts_arr, dec_ns, side="right") - 1
    ok = idx >= 0
    df = df[ok].reset_index(drop=True); idx = idx[ok]
    for c in PATTERN_COLS:
        df[c] = m5[c].to_numpy()[idx]
    print(f"  decision rows={len(df)}")

    # targets
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float)
    sp = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float)
    good = lp.notna() & sp.notna()
    df = df[good].reset_index(drop=True)
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float)
    sp = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float)
    df["dir_up"] = (lp > sp).astype(int)
    df["take_win"] = (lp > 0).astype(int)
    df = df.sort_values("ts").reset_index(drop=True)

    # LEAK GUARD
    FORBIDDEN = ("terminal_pnl", "mfe", "mae", "forward", "wait_", "take_now_", "_outcome", "trough", "bars_to")
    for c in BASELINE_COLS + PATTERN_COLS:
        assert not any(tok in c for tok in FORBIDDEN), f"LEAK: {c}"

    # splits
    STRICT_TR = df["ts"] <= pd.Timestamp("2025-06-01", tz="UTC")
    STRICT_TE = df["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")
    med = df["ts"].median()
    early = df["ts"] <= med; late = df["ts"] > med
    print(f"[3/4] strict: train_n={int(STRICT_TR.sum())} test2026_n={int(STRICT_TE.sum())}  "
          f"median split @ {med}")

    res = {}
    for ycol in ("dir_up", "take_win"):
        base_strict = fit_auc(df, BASELINE_COLS, ycol, STRICT_TR, STRICT_TE)
        plus_strict = fit_auc(df, BASELINE_COLS + PATTERN_COLS, ycol, STRICT_TR, STRICT_TE)
        stand_strict = fit_auc(df, PATTERN_COLS, ycol, STRICT_TR, STRICT_TE)
        # robustness on incremental dAUC (plus - base) both directions on median split
        base_el = fit_auc(df, BASELINE_COLS, ycol, early, late)
        plus_el = fit_auc(df, BASELINE_COLS + PATTERN_COLS, ycol, early, late)
        base_le = fit_auc(df, BASELINE_COLS, ycol, late, early)
        plus_le = fit_auc(df, BASELINE_COLS + PATTERN_COLS, ycol, late, early)
        res[ycol] = dict(
            base_2026=base_strict, plus_2026=plus_strict, dauc_2026=plus_strict - base_strict,
            standalone_2026=stand_strict,
            dauc_early_late=plus_el - base_el, dauc_late_early=plus_le - base_le,
            base_early_late=base_el, plus_early_late=plus_el,
            base_late_early=base_le, plus_late_early=plus_le,
        )
        print(f"\n[{ycol}] base_2026={base_strict:.4f} plus_2026={plus_strict:.4f} "
              f"dAUC_2026={plus_strict-base_strict:+.4f} standalone={stand_strict:.4f}")
        print(f"  robust dAUC fit_early→late={plus_el-base_el:+.4f}  fit_late→early={plus_le-base_le:+.4f}")

    # redundancy probe: how well do baseline raw-OHLC cols PREDICT the candlestick pattern signal?
    # if baseline reconstructs cs_net_signed with high R^2, the patterns are raw-OHLC redundancy.
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    tr = STRICT_TR.values; te = STRICT_TE.values
    Xtr = df.loc[tr, BASELINE_COLS].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    Xte = df.loc[te, BASELINE_COLS].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    redun = {}
    for tgt in ("cs_net_signed", "cs_engulf_signed", "cs_wick_reversal_signed"):
        ytr = df.loc[tr, tgt].astype(np.float32).values; yte = df.loc[te, tgt].astype(np.float32).values
        if len(Xtr) > SAMPLE_CAP:
            sel = RNG.choice(len(Xtr), SAMPLE_CAP, replace=False)
            xt, yt = Xtr.iloc[sel], ytr[sel]
        else:
            xt, yt = Xtr, ytr
        rg = HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=0.05,
                                           l2_regularization=1.0, min_samples_leaf=200,
                                           early_stopping=True, random_state=42)
        rg.fit(xt, yt)
        redun[tgt] = float(r2_score(yte, rg.predict(Xte)))
    print(f"\n[redundancy] baseline→pattern R^2 (2026): {redun}")

    out = dict(results=res, redundancy_baseline_to_pattern_r2=redun,
               n_total=len(df), patterns_built=PATTERN_COLS, baseline_cols=BASELINE_COLS,
               strict_train_n=int(STRICT_TR.sum()), strict_test2026_n=int(STRICT_TE.sum()))
    Path("/tmp/oot_candlestick_arm.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nWROTE /tmp/oot_candlestick_arm.json")


if __name__ == "__main__":
    main()
