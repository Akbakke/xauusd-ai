"""OOT pre-gate ARM: PROPER multi-swing FIBONACCI retracement vs raw-OHLC baseline.

The chain already has a SIMPLE premium/discount fib (smc_premium_discount + the template's
fib_dist_382/500/618/in_golden/deep_retrace built on it) — and it was OOT-REFUTED. This arm builds
a PROPER fib anchored on the last CONFIRMED significant swing leg (high->low or low->high via swing
pivots), with the full ladder 0.236/0.382/0.5/0.618/0.786 + the 1.272/1.618 extensions, encoded as:
  - signed ATR-distance to each retracement level (5) + each extension (2)
  - in-golden-zone (0.382-0.618) flag
  - which-level-nearest (ordinal) + abs ATR-dist to nearest level
  - retracement depth = where current close sits in the leg (0=at-anchor-extreme, 1=full-retrace)
  - leg direction (up-swing being retraced down vs down-swing retraced up) + leg size in ATR + bars-since
Leak-safe: pivots confirmed only at p+SWING_LOOKBACK (uses no future bars at the decision bar).

LOAD-BEARING TEST = INCREMENTAL over a RAW-OHLC BASELINE (body/wick/range/returns/ema_slope/atr/rsi
from the FO — what the V10 transformer already sees per-bar). Win = dAUC(2026)>0 AND both robustness
splits (fit_early->late, fit_late->early) positive AND survives leak/redundancy (standalone ~0.5 = noise,
and proper-fib must beat the simple-fib that already failed).
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
RNG = np.random.default_rng(20260617)
SAMPLE_CAP = 200_000
K = "K96"

# Raw-OHLC baseline (what the V10 transformer already sees per-bar; pattern must beat THIS, OOT)
BASELINE_COLS = ["body_pct_canon_v1", "wick_asym_canon_v1", "_v1_wick_imbalance_canon_v1",
                 "_v1_body_share_1_canon_v1", "range_canon_v1", "ret_1_canon_v1", "ret_5_canon_v1",
                 "ret_20_canon_v1", "_v1_r5_canon_v1", "_v1_r24_canon_v1", "roc20_canon_v1",
                 "roc100_canon_v1", "ema20_slope_canon_v1", "ema100_slope_canon_v1",
                 "atr_canon_v1", "_v1_rsi14_canon_v1", "pos_vs_ema200_canon_v1"]

# The SIMPLE fib already in the chain (OOT-refuted) — reference arm to prove proper > simple (or not)
SIMPLE_FIB_COLS = ["smc_premium_discount", "fib_dist_382", "fib_dist_500", "fib_dist_618",
                   "fib_in_golden", "fib_deep_retrace"]

FIB_RETR = (0.236, 0.382, 0.5, 0.618, 0.786)
FIB_EXT = (1.272, 1.618)


@njit(cache=True)
def _proper_fib_core(high, low, close, atr, sh_mask, sl_mask, conf_lag, swing_min_atr):
    """PROPER multi-swing fib anchored on the last CONFIRMED swing LEG.

    A pivot at bar p is confirmed at p+conf_lag (leak-safe — needs conf_lag future bars to confirm, so
    only used from bar p+conf_lag onward). The 'last significant leg' = the move between the two
    most-recent confirmed pivots of OPPOSITE type whose magnitude >= swing_min_atr*atr (filters
    micro-wiggle). anchor_b = the most-recent extreme; anchor_a = the prior opposite extreme the move
    started from. up-leg (low->high) retraced DOWN; down-leg (high->low) retraced UP.

    Outputs per bar (all leak-safe): retr_depth, leg_dir(+1 up/-1 down/0 none), leg_size_atr,
    bars_since_b, signed ATR-dist to each of 5 retr levels + 2 ext levels, nearest_level_ord (0..6),
    nearest_abs_atr, in_golden(0/1).
    """
    nb = high.shape[0]
    NR = 5
    NE = 2
    retr_depth = np.full(nb, np.nan)
    leg_dir = np.zeros(nb, np.float64)
    leg_size_atr = np.full(nb, np.nan)
    bars_since_b = np.full(nb, 999.0)
    dretr = np.full((nb, NR), np.nan)
    dext = np.full((nb, NE), np.nan)
    nearest_ord = np.full(nb, -1.0)
    nearest_abs = np.full(nb, np.nan)
    in_golden = np.full(nb, np.nan)
    retr_levels = np.array([0.236, 0.382, 0.5, 0.618, 0.786])
    ext_levels = np.array([1.272, 1.618])

    last_sh_p = np.nan; last_sh_b = -1
    last_sl_p = np.nan; last_sl_b = -1

    for i in range(nb):
        conf = i - conf_lag
        if conf >= 0:
            if sh_mask[conf]:
                last_sh_p = high[conf]; last_sh_b = conf
            if sl_mask[conf]:
                last_sl_p = low[conf]; last_sl_b = conf

        a_atr = atr[i]
        if a_atr < 1e-3:
            a_atr = 1e-3
        cp = close[i]

        if last_sh_b < 0 or last_sl_b < 0:
            continue
        # the leg = move between the two most-recent OPPOSITE confirmed pivots.
        # later pivot = anchor_b (most recent extreme), earlier = anchor_a (origin extreme).
        if last_sh_b > last_sl_b:
            # most-recent extreme is a HIGH -> up-leg (low->high), retraced DOWN
            a_b = last_sh_b; a_b_p = last_sh_p     # later extreme (high)
            a_a_p = last_sl_p                       # earlier extreme (low)
            ldir = 1.0
        else:
            # most-recent extreme is a LOW -> down-leg (high->low), retraced UP
            a_b = last_sl_b; a_b_p = last_sl_p
            a_a_p = last_sh_p
            ldir = -1.0
        leg = a_b_p - a_a_p     # signed: up-leg>0, down-leg<0
        leg_abs = leg if leg >= 0 else -leg
        if leg_abs < swing_min_atr * a_atr or leg_abs < 1e-9:
            continue   # micro-wiggle leg — leave NaN (no valid fib structure)

        leg_dir[i] = ldir
        leg_size_atr[i] = leg_abs / a_atr
        bars_since_b[i] = float(i - a_b)

        # retracement depth: 0 at anchor_b extreme (no retrace), 1 at full retrace back to anchor_a.
        # price at retrace fraction r = a_b_p - r*leg  (for up-leg: pulls down from high; down-leg: pulls up).
        # depth = (a_b_p - cp)/leg for up; (a_b_p - cp)/leg also works for down since leg<0 flips sign.
        depth = (a_b_p - cp) / leg
        retr_depth[i] = depth
        in_golden[i] = 1.0 if (depth >= 0.382 and depth <= 0.618) else 0.0

        best_abs = 1e18; best_ord = -1.0
        for r in range(NR):
            lvl_price = a_b_p - retr_levels[r] * leg
            d = (cp - lvl_price) / a_atr     # signed ATR-dist: + above level, - below
            dretr[i, r] = d
            ad = d if d >= 0 else -d
            if ad < best_abs:
                best_abs = ad; best_ord = float(r)
        for e in range(NE):
            lvl_price = a_b_p - ext_levels[e] * leg   # extension BEYOND anchor_b in leg direction
            # extension beyond the recent extreme: for up-leg fib ext target is ABOVE a_b_p (1.272 etc.)
            # a_b_p - 1.272*leg for up-leg (leg>0) = below a_a... that's wrong. ext = a_a_p + ext*leg.
            lvl_price = a_a_p + ext_levels[e] * leg
            d = (cp - lvl_price) / a_atr
            dext[i, e] = d
            ad = d if d >= 0 else -d
            if ad < best_abs:
                best_abs = ad; best_ord = float(NR + e)
        nearest_ord[i] = best_ord
        nearest_abs[i] = best_abs

    return (retr_depth, leg_dir, leg_size_atr, bars_since_b, dretr, dext,
            nearest_ord, nearest_abs, in_golden)


def build_proper_fib() -> pd.DataFrame:
    m5 = pd.read_parquet(M5_PREBUILT, columns=["time", "open", "high", "low", "close",
                                               "smc_premium_discount"])
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time").reset_index(drop=True)
    high = m5["high"].to_numpy(np.float64)
    low = m5["low"].to_numpy(np.float64)
    close = m5["close"].to_numpy(np.float64)
    # SMA-14 TR atr (build-truth atr used by SMC)
    tr = pd.concat([(m5["high"] - m5["low"]).abs(),
                    (m5["high"] - m5["close"].shift(1)).abs(),
                    (m5["low"] - m5["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().to_numpy(np.float64)

    sh_mask, sl_mask = _detect_swing_pivots(high, low, SWING_LOOKBACK)
    (retr_depth, leg_dir, leg_size_atr, bars_since_b, dretr, dext,
     nearest_ord, nearest_abs, in_golden) = _proper_fib_core(
        high, low, close, atr, sh_mask, sl_mask, SWING_LOOKBACK, 0.75)

    out = pd.DataFrame({"time": m5["time"]})
    out["pfib_retr_depth"] = retr_depth
    out["pfib_leg_dir"] = leg_dir
    out["pfib_leg_size_atr"] = leg_size_atr
    out["pfib_bars_since_b"] = np.clip(bars_since_b, 0, 999)
    for r, lv in enumerate(FIB_RETR):
        out[f"pfib_dist_{int(lv*1000)}_atr"] = dretr[:, r]
    for e, lv in enumerate(FIB_EXT):
        out[f"pfib_dist_ext_{int(lv*1000)}_atr"] = dext[:, e]
    out["pfib_nearest_level_ord"] = nearest_ord
    out["pfib_nearest_abs_atr"] = nearest_abs
    out["pfib_in_golden"] = in_golden

    # the SIMPLE fib (template) on smc_premium_discount — the OOT-refuted reference
    pd_arr = m5["smc_premium_discount"].to_numpy(np.float64)
    out["smc_premium_discount"] = pd_arr
    out["fib_dist_382"] = np.abs(pd_arr - 0.382)
    out["fib_dist_500"] = np.abs(pd_arr - 0.500)
    out["fib_dist_618"] = np.abs(pd_arr - 0.618)
    out["fib_in_golden"] = ((pd_arr >= 0.35) & (pd_arr <= 0.55)).astype(np.float64)
    out["fib_deep_retrace"] = (pd_arr < 0.214).astype(np.float64)
    return out


PROPER_FIB_COLS = (["pfib_retr_depth", "pfib_leg_dir", "pfib_leg_size_atr", "pfib_bars_since_b"]
                   + [f"pfib_dist_{int(lv*1000)}_atr" for lv in FIB_RETR]
                   + [f"pfib_dist_ext_{int(lv*1000)}_atr" for lv in FIB_EXT]
                   + ["pfib_nearest_level_ord", "pfib_nearest_abs_atr", "pfib_in_golden"])


def load_decisions() -> pd.DataFrame:
    files = sorted(glob.glob(f"{FO_DIR}/*.parquet"))
    label_cols = [f"take_now_long_terminal_pnl_at_{K}_v1", f"take_now_short_terminal_pnl_at_{K}_v1"]
    need = sorted(set(BASELINE_COLS + label_cols + ["decision_ts_utc"]))
    parts = [pd.read_parquet(f, columns=need) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["ts"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
    return df


def fit_eval(df, feat_cols, train_mask, test_mask):
    Xtr = df.loc[train_mask, feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    Xte = df.loc[test_mask, feat_cols].astype(np.float32).replace([np.inf, -np.inf], np.nan)
    ytr = df.loc[train_mask, "_y"].values
    yte = df.loc[test_mask, "_y"].values
    if len(Xtr) > SAMPLE_CAP:
        sel = RNG.choice(len(Xtr), SAMPLE_CAP, replace=False)
        Xtr = Xtr.iloc[sel]; ytr = ytr[sel]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=300, learning_rate=0.05,
                                         l2_regularization=1.0, min_samples_leaf=200,
                                         early_stopping=True, validation_fraction=0.15,
                                         random_state=42)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def main():
    print("[1/4] building PROPER fib on M5 tape ...")
    fib = build_proper_fib()
    fib_cols_all = [c for c in fib.columns if c != "time"]
    # liveness sanity: how many proper-fib cols vary
    nz = {c: float(fib[c].notna().mean()) for c in PROPER_FIB_COLS}
    print("  proper-fib non-nan coverage:", {k: round(v, 3) for k, v in list(nz.items())[:6]}, "...")

    print("[2/4] loading decision bars ...")
    df = load_decisions()
    print(f"  decision rows={len(df)}")

    ts_arr = fib["time"].values.astype("datetime64[ns]").astype(np.int64)
    dec_ns = df["ts"].values.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(ts_arr, dec_ns, side="right") - 1
    ok = idx >= 0
    df = df[ok].reset_index(drop=True); idx = idx[ok]
    fib_np = {c: fib[c].to_numpy() for c in fib_cols_all}
    for c in fib_cols_all:
        df[c] = fib_np[c][idx]

    # TARGET: dir_up = long_pnl > short_pnl
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float)
    sp = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float)
    df = df[lp.notna() & sp.notna()].reset_index(drop=True)
    lp = df[f"take_now_long_terminal_pnl_at_{K}_v1"].astype(float).values
    sp = df[f"take_now_short_terminal_pnl_at_{K}_v1"].astype(float).values
    df["_y"] = (lp > sp).astype(int)
    base_rate = float(df["_y"].mean())
    print(f"  target dir_up base_rate={base_rate:.4f}  n={len(df)}")

    # LEAKAGE GUARD
    FORBIDDEN = ("terminal_pnl", "mfe", "mae", "forward", "wait_", "take_now_", "_outcome",
                 "trough", "bars_to", "_y", "decision_ts")
    for fset in (BASELINE_COLS, PROPER_FIB_COLS, SIMPLE_FIB_COLS):
        for c in fset:
            assert not any(tok in c for tok in FORBIDDEN), f"LEAKAGE: {c}"

    df = df.sort_values("ts").reset_index(drop=True)

    # ── STRICT split per task: fit ts<=2025-06-01 / TEST ts>=2026-01-01 ──
    train_strict = df["ts"] <= pd.Timestamp("2025-06-01", tz="UTC")
    test_2026 = df["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")
    print(f"[3/4] strict split: train(<=2025-06)={int(train_strict.sum())}  test(2026)={int(test_2026.sum())}")

    base_2026 = fit_eval(df, BASELINE_COLS, train_strict, test_2026)
    plus_2026 = fit_eval(df, BASELINE_COLS + PROPER_FIB_COLS, train_strict, test_2026)
    standalone_2026 = fit_eval(df, PROPER_FIB_COLS, train_strict, test_2026)
    simple_2026 = fit_eval(df, BASELINE_COLS + SIMPLE_FIB_COLS, train_strict, test_2026)
    dauc_2026 = plus_2026 - base_2026
    dauc_simple = simple_2026 - base_2026
    print(f"  baseline 2026 AUC          = {base_2026:.4f}")
    print(f"  baseline+PROPER-fib 2026   = {plus_2026:.4f}  (dAUC={dauc_2026:+.4f})")
    print(f"  baseline+SIMPLE-fib 2026   = {simple_2026:.4f}  (dAUC={dauc_simple:+.4f})  [the refuted one]")
    print(f"  PROPER-fib STANDALONE 2026 = {standalone_2026:.4f}  (~0.5 = noise)")

    # ── ROBUSTNESS: median split both directions ──
    med = df["ts"].median()
    early = df["ts"] <= med
    late = df["ts"] > med
    base_el = fit_eval(df, BASELINE_COLS, early, late)
    plus_el = fit_eval(df, BASELINE_COLS + PROPER_FIB_COLS, early, late)
    base_le = fit_eval(df, BASELINE_COLS, late, early)
    plus_le = fit_eval(df, BASELINE_COLS + PROPER_FIB_COLS, late, early)
    robust_el = plus_el - base_el
    robust_le = plus_le - base_le
    print(f"[4/4] robustness (median split @ {med}):")
    print(f"  fit_early->late: base={base_el:.4f} +fib={plus_el:.4f}  dAUC={robust_el:+.4f}")
    print(f"  fit_late->early: base={base_le:.4f} +fib={plus_le:.4f}  dAUC={robust_le:+.4f}")

    clears = bool(dauc_2026 > 0 and robust_el > 0 and robust_le > 0
                  and 0.45 <= standalone_2026 <= 0.62)
    out = dict(arm="fibonacci_proper",
               patterns_built=PROPER_FIB_COLS,
               baseline_2026_auc=round(base_2026, 4),
               plus_patterns_2026_auc=round(plus_2026, 4),
               incremental_dauc_2026=round(dauc_2026, 4),
               simple_fib_2026_auc=round(simple_2026, 4),
               simple_fib_dauc_2026=round(dauc_simple, 4),
               standalone_pattern_auc=round(standalone_2026, 4),
               robust_dauc_early_late=round(robust_el, 4),
               robust_dauc_late_early=round(robust_le, 4),
               clears_gate=clears,
               base_rate=round(base_rate, 4))
    Path("/tmp/oot_pregate_fibonacci_proper.json").write_text(json.dumps(out, indent=2, default=str))
    print("WROTE /tmp/oot_pregate_fibonacci_proper.json")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
