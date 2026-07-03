"""S/R RESIDUAL FALSIFICATION (2026-06-24, user). DIAGNOSTIC ONLY — no model, no prod features, no deploy.

HYPOTHESIS: today's models (V10 transformer + XGB + Entry-IQL) don't fully understand structural price
levels, so a systematic share of their ERRORS clusters near liquidity pools / support-resistance. If true,
S/R distances should EXPLAIN the model's residuals (BAD takes systematically NEARER levels) and add
INCREMENTAL OOT information over today's score / conviction / margin. If neither holds, KILL the S/R track.

WHY A NEW FILE (rule 7): the prior S/R refutations (round-number wall, FVG, fibonacci) tested S/R as a
DIRECTIONAL feature added to the chain. This is a different, sharper question — a RESIDUAL analysis: do the
model's *mistakes* live near levels? No existing research script frames the take-universe residuals against
mechanically-built D/W/M + swing + session + equal-high/low levels. Reuses the cached live-policy take
universe + the M5 canonical tape (read-only); builds nothing into the production stack.

POPULATION  = the ACTUAL live-policy take universe (FOLD_1 conviction-gate −37.71 + DIPFIX), 61,822 takes,
              cached /tmp/regime_universe_20260623.parquet (built by oot_live_policy_falsification_20260623).
              These are the trades the live model actually CHOOSES — the right substrate for "model errors".
OUTCOME     = forward_outcome terminal_pnl @K96 for the chosen side (held-to-horizon entry-side label — the
              same cheap label every other entry-side falsification used; exit stack is downstream + common).
LEVELS      = built from a continuous M5 canonical tape, NO lookahead: every level uses ONLY bars strictly
              before the entry's period (prev-period extremes) or confirmed >=W bars before entry (swings).

Aggressive falsification (DEL 10): S/R survives ONLY if >=1 of the 5 conditions holds AND survives a
day-block bootstrap. Otherwise REJECT. The default verdict is REJECT; the data must overturn it.

Run: .venv/bin/python -m gx1.research.oot_sr_residual_falsification_20260624
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

UNIV = "/tmp/regime_universe_20260623.parquet"
M5TAPE = "/home/andre2/GX1_DATA/runs/PREBUILT_REBUILD_20260613/canonical_features_v2.parquet"
OUT_JSON = "/tmp/sr_residual_falsification_20260624.json"
RNG = np.random.default_rng(0)

# zone widths in ATR units (DEL 8)
ZONES = [0.1, 0.25, 0.5]
PRIMARY_Z = 0.25

# the structural levels we test. type 'H' = resistance-like (above), 'L' = support-like (below)
HIGH_LEVELS = ["pdh", "pwh", "pmh", "swing_high", "asia_high", "london_high", "eqh"]
LOW_LEVELS = ["pdl", "pwl", "pml", "swing_low", "asia_low", "london_low", "eql"]


# ══════════════════════════════════════════════════════════════════════════
# DEL 1 — STRUCTURAL LEVELS (no lookahead)
# ══════════════════════════════════════════════════════════════════════════
def period_levels(m5: pd.DataFrame, key: str) -> pd.DataFrame:
    """prev-period high/low active from each period's START (prev period is fully closed by then)."""
    s = m5[["time", "high", "low"]].copy()
    if key == "D":
        s["k"] = s["time"].dt.floor("D")
    elif key == "W":
        s["k"] = s["time"].dt.to_period("W").dt.start_time.dt.tz_localize("UTC")
    elif key == "M":
        s["k"] = s["time"].dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    agg = s.groupby("k").agg(hi=("high", "max"), lo=("low", "min")).reset_index()
    agg["phi"] = agg["hi"].shift(1)
    agg["plo"] = agg["lo"].shift(1)
    agg = agg.dropna(subset=["phi", "plo"])
    return agg[["k", "phi", "plo"]].rename(columns={"k": "active_from"})


def swing_levels(m5: pd.DataFrame, tf: str = "1h", W: int = 3) -> pd.DataFrame:
    """Confirmed fractal swing high/low on the TF grid. A swing at bar i is the strict extreme over
    [i-W, i+W]; it becomes KNOWN only at bar i+W (confirmation). Series = last-confirmed swing H / L,
    plus an 'equal-highs/lows' pool level = tightest pair among the last 10 confirmed swings within
    0.10*ATR(H1). All active_from times are confirmation times (>= the swing bar) → no lookahead."""
    s = m5.set_index("time")
    hi = s["high"].resample(tf, label="right", closed="right").max()
    lo = s["low"].resample(tf, label="right", closed="right").min()
    df = pd.DataFrame({"high": hi, "low": lo}).dropna()
    t = df.index
    hh, ll = df["high"].values, df["low"].values
    n = len(df)
    atr = (df["high"] - df["low"]).rolling(14, min_periods=3).mean().bfill().values  # H1 ATR (price units)
    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    eqh = np.full(n, np.nan)
    eql = np.full(n, np.nan)
    rec_h: list[float] = []  # last-10 confirmed swing-high prices
    rec_l: list[float] = []
    cur_sh = np.nan
    cur_sl = np.nan
    for i in range(n):
        # confirm any swing whose center is i-W (known now, at i)
        c = i - W
        if c >= W:
            win_h = hh[c - W:c + W + 1]
            win_l = ll[c - W:c + W + 1]
            if hh[c] == win_h.max() and hh[c] > hh[c - W:c].max() and hh[c] >= hh[c + 1:c + W + 1].max():
                cur_sh = hh[c]
                rec_h.append(hh[c])
                rec_h[:] = rec_h[-10:]
            if ll[c] == win_l.min() and ll[c] < ll[c - W:c].min() and ll[c] <= ll[c + 1:c + W + 1].min():
                cur_sl = ll[c]
                rec_l.append(ll[c])
                rec_l[:] = rec_l[-10:]
        last_sh[i] = cur_sh
        last_sl[i] = cur_sl
        tol = 0.10 * atr[i]
        eqh[i] = _tightest_pair(rec_h, tol)
        eql[i] = _tightest_pair(rec_l, tol)
    return pd.DataFrame({"active_from": t, "swing_high": last_sh, "swing_low": last_sl,
                         "eqh": eqh, "eql": eql})


def _tightest_pair(vals: list[float], tol: float) -> float:
    """Level of the tightest pair within tol (an 'equal highs/lows' liquidity pool); NaN if none."""
    if len(vals) < 2 or not np.isfinite(tol) or tol <= 0:
        return np.nan
    a = np.sort(np.asarray(vals))
    gaps = np.diff(a)
    j = int(np.argmin(gaps))
    return float((a[j] + a[j + 1]) / 2) if gaps[j] <= tol else np.nan


def session_levels(m5: pd.DataFrame) -> pd.DataFrame:
    """Asia (00–08 UTC) / London (08–16 UTC) session high/low of the most-recent COMPLETED session,
    active from the session close (08:00 / 16:00). No lookahead."""
    s = m5[["time", "high", "low"]].copy()
    s["day"] = s["time"].dt.floor("D")
    s["hr"] = s["time"].dt.hour
    out = []
    for name, (h0, h1) in {"asia": (0, 8), "london": (8, 16)}.items():
        seg = s[(s["hr"] >= h0) & (s["hr"] < h1)]
        g = seg.groupby("day").agg(hi=("high", "max"), lo=("low", "min")).reset_index()
        g["active_from"] = g["day"] + pd.Timedelta(hours=h1)
        out.append(g[["active_from", "hi", "lo"]].rename(columns={"hi": f"{name}_high", "lo": f"{name}_low"}))
    a, l = out
    return a.merge(l, on="active_from", how="outer").sort_values("active_from")


def build_levels(u: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
    """Asof-join every structural level + the tape entry price onto each entry ts (backward = no lookahead)."""
    u = u.sort_values("ts").reset_index(drop=True)
    # entry price from the SAME tape the levels come from (avoids the universe-vs-tape close mismatch)
    px = m5[["time", "close"]].rename(columns={"close": "P"}).sort_values("time")
    d = pd.merge_asof(u, px, left_on="ts", right_on="time", direction="backward")
    for key, (nh, nl) in {"D": ("pdh", "pdl"), "W": ("pwh", "pwl"), "M": ("pmh", "pml")}.items():
        pl = period_levels(m5, key).rename(columns={"phi": nh, "plo": nl}).sort_values("active_from")
        d = pd.merge_asof(d.sort_values("ts"), pl, left_on="ts", right_on="active_from",
                          direction="backward").drop(columns=["active_from"])
    sw = swing_levels(m5).sort_values("active_from")
    d = pd.merge_asof(d.sort_values("ts"), sw, left_on="ts", right_on="active_from",
                      direction="backward").drop(columns=["active_from"])
    se = session_levels(m5).sort_values("active_from")
    d = pd.merge_asof(d.sort_values("ts"), se, left_on="ts", right_on="active_from",
                      direction="backward").drop(columns=["active_from"])
    return d


def add_distances(d: pd.DataFrame) -> pd.DataFrame:
    """Signed gap (price − level) in bps and ATR units, + |gap| in ATR, for every level."""
    P = d["P"].values
    atr = d["atr"].clip(lower=1.0).values  # atr_bps (M5 ATR14), floor 1 bps to avoid blowups
    for lv in HIGH_LEVELS + LOW_LEVELS:
        L = d[lv].values
        gap_bps = (P - L) / P * 1e4              # >0 = price ABOVE the level
        gap_atr = gap_bps / atr
        d[f"{lv}_gap_atr"] = gap_atr
        d[f"{lv}_abs_atr"] = np.abs(gap_atr)
    return d


# ══════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════
def sharpe(p):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return float(np.mean(p) / (np.std(p) + 1e-9)) if len(p) else float("nan")


def odds_ratio(near_loser, n_loser, near_winner, n_winner):
    """Haldane-corrected OR of being NEAR a level for losers vs winners (+95% CI)."""
    a, b = near_loser + 0.5, (n_loser - near_loser) + 0.5
    c, dd = near_winner + 0.5, (n_winner - near_winner) + 0.5
    orr = (a / b) / (c / dd)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / dd)
    lo, hi = np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)
    return float(orr), float(lo), float(hi)


def day_block_boot(values_by_day, stat_fn, n=10000):
    """Day-block bootstrap: resample whole UTC days w/ replacement, recompute stat. Returns (est, lo, hi, p)."""
    days = list(values_by_day.keys())
    arrs = [values_by_day[dd] for dd in days]
    base = stat_fn(np.concatenate(arrs)) if arrs else float("nan")
    nd = len(days)
    boots = np.empty(n)
    for b in range(n):
        idx = RNG.integers(0, nd, nd)
        cat = np.concatenate([arrs[j] for j in idx])
        boots[b] = stat_fn(cat)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided p vs 0 (effect-size null)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(base), float(lo), float(hi), float(min(p, 1.0))


# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 100)
    print("S/R RESIDUAL FALSIFICATION — diagnostic only (no model / no prod feature / no deploy)")
    print("=" * 100)
    u = pd.read_parquet(UNIV)
    u["ts"] = pd.to_datetime(u["ts"], utc=True)
    u = u[u["side"].isin(["long", "short"])].copy()
    u = u[np.isfinite(u["pnl"])].copy()
    m5 = pd.read_parquet(M5TAPE, columns=["time", "high", "low", "close"])
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time").reset_index(drop=True)
    print(f"universe takes: {len(u)}  | LONG {int((u.side=='long').sum())}  SHORT {int((u.side=='short').sum())}")
    print(f"M5 tape: {m5.time.min()} → {m5.time.max()} ({len(m5)} bars)")

    d = build_levels(u, m5)
    d = add_distances(d)
    d["day"] = d["ts"].dt.floor("D")
    d["year"] = d["ts"].dt.year
    d["loss"] = (d["pnl"] < 0).astype(int)
    d["win"] = (d["pnl"] > 0).astype(int)
    # directional score (chosen-side V10 prob), conviction (IQL raw_adv), margin (V10 top1-top2)
    d["dir_score"] = np.where(d["side"] == "long", d["p_long"], d["p_short"])
    print(f"rows w/ entry price + levels: {int(d['P'].notna().sum())}  "
          f"loss-rate={d['loss'].mean():.3f}  mean pnl={d['pnl'].mean():.2f} bps")

    OUT = {"meta": dict(n=len(d), n_long=int((d.side == "long").sum()), n_short=int((d.side == "short").sum()),
                        loss_rate=round(float(d.loss.mean()), 4), mean_pnl_bps=round(float(d.pnl.mean()), 2),
                        primary_zone_atr=PRIMARY_Z, label="terminal_pnl@K96", population="live-policy FOLD_1 takes")}

    # ── DEL 2 — residual terciles GOOD/MID/BAD by terminal return ──────────
    print("\n" + "─" * 100 + "\nDEL 2 — RESIDUAL TERCILES (GOOD top30% / MID / BAD bottom30% by terminal pnl)")
    q30, q70 = d["pnl"].quantile([0.30, 0.70])
    d["coh"] = np.where(d["pnl"] <= q30, "BAD", np.where(d["pnl"] >= q70, "GOOD", "MID"))
    del2 = {}
    print(f"  thresholds: BAD<= {q30:.1f} bps  GOOD>= {q70:.1f} bps   "
          f"(BAD n={int((d.coh=='BAD').sum())} GOOD n={int((d.coh=='GOOD').sum())})")
    print(f"  {'level':12s} {'BADnear%':>9s} {'GOODnr%':>9s} {'BADmed|d|':>10s} {'GOODmed|d|':>11s} "
          f"{'Δmean|d|':>9s} {'MWU_p':>8s}   (near=|gap|<=0.25ATR; |d| in ATR; Δ=BAD−GOOD, neg=BAD nearer)")
    for lv in HIGH_LEVELS + LOW_LEVELS:
        col = f"{lv}_abs_atr"
        b = d.loc[d.coh == "BAD", col].dropna()
        g = d.loc[d.coh == "GOOD", col].dropna()
        if len(b) < 50 or len(g) < 50:
            continue
        nb = float((b <= PRIMARY_Z).mean())
        ng = float((g <= PRIMARY_Z).mean())
        try:
            _, mwu_p = mannwhitneyu(b, g, alternative="two-sided")
        except ValueError:
            mwu_p = float("nan")
        dmean = float(b.mean() - g.mean())
        del2[lv] = dict(bad_near_pct=round(nb, 4), good_near_pct=round(ng, 4),
                        bad_med_abs_atr=round(float(b.median()), 3), good_med_abs_atr=round(float(g.median()), 3),
                        delta_mean_abs_atr=round(dmean, 4), mwu_p=round(float(mwu_p), 5),
                        bad_p25=round(float(b.quantile(.25)), 3), bad_p75=round(float(b.quantile(.75)), 3))
        flag = " <<" if (mwu_p < 0.05 and dmean < 0 and nb > ng) else ""
        print(f"  {lv:12s} {nb*100:8.2f}% {ng*100:8.2f}% {b.median():10.3f} {g.median():11.3f} "
              f"{dmean:9.4f} {mwu_p:8.4f}{flag}")
    OUT["del2_terciles"] = del2

    # ── DEL 3 / DEL 4 — LONG-near-highs / SHORT-near-lows: losers vs winners odds ratio ──
    for side, levels, lbl in [("long", HIGH_LEVELS, "DEL 3 — LONG errors near RESISTANCE (highs)"),
                              ("short", LOW_LEVELS, "DEL 4 — SHORT errors near SUPPORT (lows)")]:
        print("\n" + "─" * 100 + f"\n{lbl}   (near = within 0.25 ATR, on the respect side)")
        ds = d[(d.side == side) & np.isfinite(d.pnl)]
        los = ds[ds.loss == 1]
        win = ds[ds.win == 1]
        res = {}
        print(f"  side={side}  losers={len(los)} winners={len(win)}")
        print(f"  {'level':12s} {'loser_near%':>11s} {'winner_near%':>12s} {'OR':>6s} {'95%CI':>16s}")
        for lv in levels:
            gap = f"{lv}_gap_atr"
            # respect side: for highs price BELOW level within zone (approaching resistance from below);
            # for lows price ABOVE level within zone (approaching support from above)
            if lv in HIGH_LEVELS:
                near_l = ((los[gap] <= 0) & (los[gap] >= -PRIMARY_Z)).sum()
                near_w = ((win[gap] <= 0) & (win[gap] >= -PRIMARY_Z)).sum()
            else:
                near_l = ((los[gap] >= 0) & (los[gap] <= PRIMARY_Z)).sum()
                near_w = ((win[gap] >= 0) & (win[gap] <= PRIMARY_Z)).sum()
            nl = int(np.isfinite(los[gap]).sum())
            nw = int(np.isfinite(win[gap]).sum())
            if nl < 50 or nw < 50:
                continue
            orr, lo, hi = odds_ratio(int(near_l), nl, int(near_w), nw)
            res[lv] = dict(loser_near_pct=round(near_l / nl, 4), winner_near_pct=round(near_w / nw, 4),
                           odds_ratio=round(orr, 3), ci=[round(lo, 3), round(hi, 3)],
                           n_loser=nl, n_winner=nw)
            flag = " <<" if lo > 1.0 else (" (inv)" if hi < 1.0 else "")
            print(f"  {lv:12s} {near_l/nl*100:10.2f}% {near_w/nw*100:11.2f}% {orr:6.2f} "
                  f"[{lo:6.2f},{hi:6.2f}]{flag}")
        OUT[f"del{'3' if side=='long' else '4'}_{side}_oddsratio"] = res

    # ── DEL 5 — liquidity pools: loss rate near vs far ──────────────────────
    print("\n" + "─" * 100 + "\nDEL 5 — LIQUIDITY POOLS (equal-highs/lows, Asia/London H/L): loss-rate near vs far")
    pools = ["eqh", "eql", "asia_high", "asia_low", "london_high", "london_low"]
    base_loss = float(d.loss.mean())
    del5 = {}
    print(f"  baseline loss-rate = {base_loss:.4f}")
    print(f"  {'pool':12s} {'n_near':>7s} {'near_loss':>9s} {'far_loss':>9s} {'lift':>7s} {'near_pnl':>9s}")
    for pl in pools:
        col = f"{pl}_abs_atr"
        m = np.isfinite(d[col])
        near = m & (d[col] <= PRIMARY_Z)
        far = m & (d[col] > PRIMARY_Z)
        if near.sum() < 50:
            continue
        nl_ = float(d.loc[near, "loss"].mean())
        fl_ = float(d.loc[far, "loss"].mean())
        del5[pl] = dict(n_near=int(near.sum()), near_loss=round(nl_, 4), far_loss=round(fl_, 4),
                        lift=round(nl_ - fl_, 4), near_mean_pnl=round(float(d.loc[near, "pnl"].mean()), 2))
        flag = " <<" if nl_ - fl_ > 0.02 else ""
        print(f"  {pl:12s} {int(near.sum()):7d} {nl_:9.4f} {fl_:9.4f} {nl_-fl_:+7.4f} "
              f"{d.loc[near,'pnl'].mean():9.2f}{flag}")
    OUT["del5_liquidity_pools"] = del5

    # ── DEL 6 — breakout vs rejection (vs prev-day level), expectancy/Sharpe/winrate ──
    print("\n" + "─" * 100 + "\nDEL 6 — BREAKOUT vs REJECTION (relative to prev-day high/low; zone 0.5 ATR)")
    Z6 = 0.5
    # nearest prev-day level by abs distance; classify breakout (price beyond level) vs rejection (within zone, respect side)
    cat = np.full(len(d), "none", dtype=object)
    gh, gl = d["pdh_gap_atr"].values, d["pdl_gap_atr"].values  # >0 = above the level
    # breakout up: above pdh within zone; breakout down: below pdl within zone
    bo = ((gh >= 0) & (gh <= Z6)) | ((gl <= 0) & (gl >= -Z6))
    # rejection: below pdh within zone (testing resistance) or above pdl within zone (testing support)
    rj = ((gh < 0) & (gh >= -Z6)) | ((gl > 0) & (gl <= Z6))
    cat[rj] = "rejection"
    cat[bo] = "breakout"  # breakout takes precedence if both (price sitting right on the level)
    d["brk_cat"] = cat
    del6 = {}
    for c in ["breakout", "rejection", "none"]:
        for side in ["long", "short", "all"]:
            sub = d[(d.brk_cat == c)] if side == "all" else d[(d.brk_cat == c) & (d.side == side)]
            if len(sub) < 50:
                continue
            del6[f"{c}_{side}"] = dict(n=len(sub), mean_pnl=round(float(sub.pnl.mean()), 2),
                                       win=round(float(sub.win.mean()), 4), sharpe=round(sharpe(sub.pnl), 4))
    print(f"  {'cat_side':22s} {'n':>6s} {'mean_pnl':>9s} {'winrate':>8s} {'sharpe':>7s}")
    for k, v in del6.items():
        print(f"  {k:22s} {v['n']:6d} {v['mean_pnl']:9.2f} {v['win']:8.4f} {v['sharpe']:7.4f}")
    OUT["del6_breakout_rejection"] = del6

    # ── DEL 7 — incremental OOT info: base (score/conviction/margin) vs +S/R ──
    print("\n" + "─" * 100 + "\nDEL 7 — INCREMENTAL OOT INFO (logistic loss-pred; base vs base+S/R), strict OOT both directions")
    base_feats = ["dir_score", "raw_adv", "margin", "atr"]
    sr_feats = [f"{lv}_gap_atr" for lv in HIGH_LEVELS + LOW_LEVELS] + \
               [f"{lv}_abs_atr" for lv in HIGH_LEVELS + LOW_LEVELS]
    sr_feats = [c for c in sr_feats if d[c].notna().mean() > 0.5]
    dd = d.dropna(subset=base_feats + ["pnl"]).copy()
    for c in sr_feats:
        dd[c] = dd[c].fillna(dd[c].median())
    pre = dd[dd.year < 2026]
    o26 = dd[dd.year == 2026]
    del7 = {}

    def fit_eval(tr, te, feats, y="loss"):
        Xtr, Xte = tr[feats].values, te[feats].values
        ytr, yte = tr[y].values, te[y].values
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        clf.fit(Xtr, ytr)
        pr = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, pr)
        ric = spearmanr(pr, -te["pnl"].values).correlation  # higher loss-prob ↔ lower pnl
        return float(auc), float(ric), pr

    for name, tr, te in [("pre2026→2026", pre, o26), ("2026→pre2026", o26, pre)]:
        if len(tr) < 500 or len(te) < 500:
            del7[name] = {"note": "insufficient split"}
            continue
        a0, r0, _ = fit_eval(tr, te, base_feats)
        a1, r1, _ = fit_eval(tr, te, base_feats + sr_feats)
        del7[name] = dict(n_train=len(tr), n_test=len(te),
                          auc_base=round(a0, 4), auc_base_sr=round(a1, 4), delta_auc=round(a1 - a0, 4),
                          rankic_base=round(r0, 4), rankic_base_sr=round(r1, 4), delta_rankic=round(r1 - r0, 4))
        print(f"  [{name}] base AUC={a0:.4f} → +S/R {a1:.4f}  ΔAUC={a1-a0:+.4f}   "
              f"base RankIC={r0:+.4f} → +S/R {r1:+.4f}  ΔRankIC={r1-r0:+.4f}")
    OUT["del7_incremental_oot"] = del7

    # ── DEL 8 — zones ±0.1/0.25/0.5 ATR: is the error concentrated in tight zones? ──
    print("\n" + "─" * 100 + "\nDEL 8 — ZONE WIDTH SWEEP (loss-rate near each level type at 0.1/0.25/0.5 ATR)")
    del8 = {}
    print(f"  baseline loss = {base_loss:.4f}")
    print(f"  {'level':12s}" + "".join(f"  z={z}:loss(n)" for z in ZONES))
    for lv in HIGH_LEVELS + LOW_LEVELS:
        col = f"{lv}_abs_atr"
        m = np.isfinite(d[col])
        row = {}
        line = f"  {lv:12s}"
        for z in ZONES:
            near = m & (d[col] <= z)
            if near.sum() < 30:
                row[str(z)] = None
                line += f"   --      "
                continue
            lr = float(d.loc[near, "loss"].mean())
            row[str(z)] = dict(loss=round(lr, 4), n=int(near.sum()), lift=round(lr - base_loss, 4))
            line += f"  {lr:.3f}({int(near.sum())})"
        del8[lv] = row
        print(line)
    OUT["del8_zone_sweep"] = del8

    # ── DEL 9 — day-block bootstrap (10k) on the strongest candidate effects ──
    print("\n" + "─" * 100 + "\nDEL 9 — DAY-BLOCK BOOTSTRAP (10,000 resamples) on the strongest effects")
    del9 = {}
    # (a) DEL2: BAD−GOOD mean |dist| for the level with the most negative Δmean (BAD nearer)
    cand = sorted(del2.items(), key=lambda kv: kv[1]["delta_mean_abs_atr"])
    if cand:
        lv = cand[0][0]
        col = f"{lv}_abs_atr"
        sub = d[d.coh.isin(["BAD", "GOOD"]) & np.isfinite(d[col])][["day", "coh", col]].copy()
        vbd = {dd_: grp[col].where(grp.coh == "BAD").values for dd_, grp in sub.groupby("day")}
        vgd = {dd_: grp[col].where(grp.coh == "GOOD").values for dd_, grp in sub.groupby("day")}

        def stat_diff(_):  # recomputed per resample inside boot via closure on resampled days
            return np.nan
        # custom boot: resample days, compute mean(BAD)-mean(GOOD)
        days = sub["day"].unique()
        bd = {dd_: grp.loc[grp.coh == "BAD", col].values for dd_, grp in sub.groupby("day")}
        gd = {dd_: grp.loc[grp.coh == "GOOD", col].values for dd_, grp in sub.groupby("day")}
        base = float(sub.loc[sub.coh == "BAD", col].mean() - sub.loc[sub.coh == "GOOD", col].mean())
        nd = len(days)
        boots = np.empty(10000)
        dl = list(days)
        bd_a = [bd[x] for x in dl]
        gd_a = [gd[x] for x in dl]
        for bi in range(10000):
            idx = RNG.integers(0, nd, nd)
            bb = np.concatenate([bd_a[j] for j in idx])
            gg = np.concatenate([gd_a[j] for j in idx])
            boots[bi] = (bb.mean() if len(bb) else np.nan) - (gg.mean() if len(gg) else np.nan)
        boots = boots[np.isfinite(boots)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        del9["del2_BAD_minus_GOOD_meandist"] = dict(level=lv, effect=round(base, 4),
                                                    ci=[round(float(lo), 4), round(float(hi), 4)],
                                                    p=round(float(min(p, 1)), 5),
                                                    interp="negative & CI<0 ⇒ BAD systematically nearer this level")
        print(f"  (a) {lv}: mean|dist| BAD−GOOD = {base:+.4f} ATR  95%CI[{lo:+.4f},{hi:+.4f}]  p={min(p,1):.4f}")

    # (b) DEL5: loss-rate lift near the strongest pool
    if del5:
        plv = max(del5.items(), key=lambda kv: kv[1]["lift"])[0]
        col = f"{plv}_abs_atr"
        sub = d[np.isfinite(d[col])][["day", "loss", col]].copy()
        sub["near"] = (sub[col] <= PRIMARY_Z).astype(int)
        days = sub["day"].unique()
        nd = len(days)
        gnear = {x: g.loc[g.near == 1, "loss"].values for x, g in sub.groupby("day")}
        gfar = {x: g.loc[g.near == 0, "loss"].values for x, g in sub.groupby("day")}
        dl = list(days)
        na = [gnear[x] for x in dl]
        fa = [gfar[x] for x in dl]
        base = float(sub.loc[sub.near == 1, "loss"].mean() - sub.loc[sub.near == 0, "loss"].mean())
        boots = np.empty(10000)
        for bi in range(10000):
            idx = RNG.integers(0, nd, nd)
            nn = np.concatenate([na[j] for j in idx])
            ff = np.concatenate([fa[j] for j in idx])
            boots[bi] = (nn.mean() if len(nn) else np.nan) - (ff.mean() if len(ff) else np.nan)
        boots = boots[np.isfinite(boots)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
        del9["del5_pool_lossrate_lift"] = dict(pool=plv, effect=round(base, 4),
                                               ci=[round(float(lo), 4), round(float(hi), 4)],
                                               p=round(float(min(p, 1)), 5),
                                               interp="positive & CI>0 ⇒ errors cluster near this pool")
        print(f"  (b) {plv}: loss-rate(near−far) = {base:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  p={min(p,1):.4f}")
    OUT["del9_bootstrap"] = del9

    # ── DEL 10 — falsification verdict ──────────────────────────────────────
    print("\n" + "═" * 100 + "\nDEL 10 — FALSIFICATION VERDICT")
    # C1: BAD systematically nearer SOME level (DEL2 MWU<.05 & BAD nearer) AND bootstrap CI<0
    c1_terc = [lv for lv, v in del2.items() if v["mwu_p"] < 0.05 and v["delta_mean_abs_atr"] < 0
               and v["bad_near_pct"] > v["good_near_pct"]]
    boot_a = del9.get("del2_BAD_minus_GOOD_meandist", {})
    c1 = bool(c1_terc) and boot_a.get("ci", [0, 0])[1] < 0
    # C2: model errors cluster at liquidity pools (DEL5 lift>.02 sig) AND bootstrap CI>0
    c2_pools = [pl for pl, v in del5.items() if v["lift"] > 0.02]
    boot_b = del9.get("del5_pool_lossrate_lift", {})
    c2 = bool(c2_pools) and boot_b.get("ci", [0, 0])[0] > 0
    # C3: breakout/rejection explains residuals (one category systematically negative expectancy while other positive)
    bo_a = del6.get("breakout_all", {}).get("mean_pnl", 0)
    rj_a = del6.get("rejection_all", {}).get("mean_pnl", 0)
    c3 = (bo_a < 0) != (rj_a < 0) and abs(bo_a - rj_a) > 5.0
    # C4: S/R gives incremental OOT info (ΔAUC>.01 AND ΔRankIC>.01 in BOTH directions)
    d7 = OUT["del7_incremental_oot"]
    def ok(x):
        return isinstance(x, dict) and x.get("delta_auc", -1) > 0.01 and x.get("delta_rankic", -1) > 0.01
    c4 = all(ok(d7.get(k, {})) for k in ["pre2026→2026", "2026→pre2026"])
    # C5: effects survive day-block bootstrap (either (a) or (b) significant)
    c5 = (boot_a.get("p", 1) < 0.05 and boot_a.get("ci", [0, 0])[1] < 0) or \
         (boot_b.get("p", 1) < 0.05 and boot_b.get("ci", [0, 0])[0] > 0)
    conds = {"C1_BAD_nearer_levels": c1, "C2_errors_cluster_pools": c2,
             "C3_breakout_rejection_residual": c3, "C4_incremental_OOT_info": c4,
             "C5_survives_dayblock_bootstrap": c5}
    survives = any(conds.values()) and (c4 or c5)  # the load-bearing OOT/bootstrap legs must carry it
    OUT["del10_verdict"] = dict(conditions=conds, c1_levels=c1_terc, c2_pools=c2_pools,
                                survives=bool(survives))
    for k, v in conds.items():
        print(f"   {('PASS' if v else 'fail'):4s}  {k}")
    print("\n" + ("█ S/R TRACK SURVIVES — at least one falsification condition held + carried by OOT/bootstrap."
                  if survives else
                  "▒ S/R TRACK REJECTED — no condition survived the load-bearing OOT/bootstrap test. KILL IT."))
    print("═" * 100)

    with open(OUT_JSON, "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print(f"→ {OUT_JSON}")


if __name__ == "__main__":
    main()
