#!/usr/bin/env python3
"""READ-ONLY cross-asset predictive-value SPIKE (vedtak cross_asset_features_20260609).

Question: does the DOLLAR (EUR_USD + USD_JPY, the #1 gold driver) carry INCREMENTAL
predictive value for XAU at the K96 (2h) decision horizon — beyond what the V10/XGB chain
already captures? Gold ≈ inverse dollar; if the dollar adds signal, cross-asset features
are worth building (Tier 2a). If not, the XAU-only ceiling on this axis is real.

ISOLATION (rule 6 feature-only exception): other-instrument data is FETCHED READ-ONLY and
written ONLY under GX1_DATA/research/cross_asset_spike_20260609/. It is NEVER traded, never
enters the trading chain, never shared. XAU stays the sole traded instrument.

No model training. No runtime modification. Does not touch the live chain or the V3 retrain.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from gx1.scripts.materialize_backfill_xauusd_m1_repair_v1 import _client_from_env  # reuse OANDA client

OUT = Path("/home/andre2/GX1_DATA/research/cross_asset_spike_20260609")
OUT.mkdir(parents=True, exist_ok=True)
FO_DIR = Path("/home/andre2/GX1_DATA/runs/FASE2B_CLEAN_20260608/forward_outcome_clean/per_week")
START = pd.Timestamp("2020-11-01", tz="UTC")
END = pd.Timestamp("2026-06-08", tz="UTC")


def fetch_h1(client, instrument: str) -> pd.DataFrame:
    """H1 candles START..END via chunked OANDA (cached to OUT)."""
    cache = OUT / f"{instrument}_H1.parquet"
    if cache.exists():
        d = pd.read_parquet(cache)
        return d.set_index(pd.to_datetime(d["time"], utc=True))
    parts, cur = [], START
    while cur < END:
        nxt = min(cur + pd.Timedelta(days=120), END)  # ~2880 H1 < 5000 cap
        try:
            df = client.get_candles(instrument, "H1", from_ts=cur, to_ts=nxt,
                                    include_mid=True, exclude_incomplete=True)
            if df is not None and len(df):
                parts.append(df)
        except Exception as e:
            print(f"  {instrument} {cur.date()}: {str(e)[:60]}")
        cur = nxt
    d = pd.concat(parts)
    if not isinstance(d.index, pd.DatetimeIndex):
        d = d.set_index(pd.to_datetime(d["time"], utc=True)) if "time" in d.columns else d
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d.reset_index().rename(columns={"index": "time"}).to_parquet(cache, index=False)
    print(f"  fetched {instrument} H1: {len(d)} bars {d.index.min()} -> {d.index.max()}")
    return d


def main() -> int:
    print("[SPIKE] fetching dollar legs (EUR_USD, USD_JPY) H1 from OANDA...", flush=True)
    cl = _client_from_env()
    eur = fetch_h1(cl, "EUR_USD")["close"].astype(float)
    jpy = fetch_h1(cl, "USD_JPY")["close"].astype(float)
    # USD strength: dollar UP = EUR_USD down + USD_JPY up. log-returns, summed (both +ve for stronger USD).
    df = pd.DataFrame({"eur": eur, "jpy": jpy}).dropna()
    df["usd_logret"] = -np.log(df["eur"]).diff() + np.log(df["jpy"]).diff()
    usd = pd.DataFrame(index=df.index)
    usd["usd_mom6"] = df["usd_logret"].rolling(6).sum()     # last 6h dollar move
    usd["usd_mom24"] = df["usd_logret"].rolling(24).sum()   # last 24h
    usd["usd_vs_ma200"] = (np.log(1/df["eur"]) + np.log(df["jpy"])) - (np.log(1/df["eur"]) + np.log(df["jpy"])).rolling(200).mean()
    usd = usd.dropna()
    print(f"[SPIKE] USD features built: {len(usd)} H1 rows {usd.index.min()} -> {usd.index.max()}", flush=True)

    # XAU candidates + forward K96 outcomes + V10 direction
    import glob
    cols = ["decision_ts_utc", "p_long",
            "take_now_long_terminal_pnl_at_K96_v1", "take_now_short_terminal_pnl_at_K96_v1"]
    fs = sorted(glob.glob(str(FO_DIR / "*.parquet")))
    parts = []
    for f in fs:
        try:
            parts.append(pd.read_parquet(f, columns=cols))
        except Exception:
            pass
    cand = pd.concat(parts, ignore_index=True)
    cand["ts"] = pd.to_datetime(cand["decision_ts_utc"], utc=True)
    cand = cand.dropna(subset=["ts", "p_long", "take_now_long_terminal_pnl_at_K96_v1",
                               "take_now_short_terminal_pnl_at_K96_v1"]).sort_values("ts")
    # exclude corrupt April 2026 (honest)
    apr = (cand["ts"] >= pd.Timestamp("2026-04-01", tz="UTC")) & (cand["ts"] <= pd.Timestamp("2026-04-21", tz="UTC"))
    cand = cand[~apr]
    print(f"[SPIKE] XAU candidates (April-excluded): {len(cand):,}", flush=True)

    # asof-align dollar features onto each candidate (causal: last H1 bar <= decision ts)
    usd_flat = usd.rename_axis("uts").reset_index().sort_values("uts")
    m = pd.merge_asof(cand.sort_values("ts"), usd_flat, left_on="ts", right_on="uts", direction="backward")
    m = m.dropna(subset=["usd_mom6", "usd_mom24", "usd_vs_ma200"])
    L = m["take_now_long_terminal_pnl_at_K96_v1"].to_numpy()
    S = m["take_now_short_terminal_pnl_at_K96_v1"].to_numpy()
    xau_long_better = (L > S).astype(int)          # was LONG the better XAU direction?
    xau_dir_ret = (L - S) / 2.0                     # signed XAU directional payoff proxy (bps)

    out = []
    def line(s): out.append(s); print(s, flush=True)
    line(f"\n=== CROSS-ASSET SPIKE RESULTS (n={len(m):,}, 2020-11..2026, April-excluded) ===")
    # (a) contemporaneous-to-forward correlation: dollar momentum vs XAU forward dir payoff
    for c in ["usd_mom6", "usd_mom24", "usd_vs_ma200"]:
        r = np.corrcoef(m[c].to_numpy(), xau_dir_ret)[0, 1]
        line(f"  corr(XAU_K96_dir_payoff, {c}) = {r:+.4f}   (expect NEGATIVE: dollar up -> gold long worse)")
    # (b) incremental direction lift: AUC(V10) vs AUC(V10 + usd) for predicting xau_long_better
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    X_v10 = m[["p_long"]].to_numpy()
    X_usd = m[["p_long", "usd_mom6", "usd_mom24", "usd_vs_ma200"]].to_numpy()
    y = xau_long_better
    Xtr_a, Xte_a, Xtr_b, Xte_b, ytr, yte = train_test_split(X_v10, X_usd, y, test_size=0.3, shuffle=False)
    auc_v10 = roc_auc_score(yte, LogisticRegression(max_iter=500).fit(Xtr_a, ytr).predict_proba(Xte_a)[:, 1])
    auc_usd = roc_auc_score(yte, LogisticRegression(max_iter=500).fit(Xtr_b, ytr).predict_proba(Xte_b)[:, 1])
    line(f"\n  direction AUC (chronological holdout 30%): V10-only={auc_v10:.4f}  V10+USD={auc_usd:.4f}  LIFT={auc_usd-auc_v10:+.4f}")
    # (c) selection: V10-pick win-rate/bps when dollar AGREES vs DISAGREES with the trade
    v10_long = (m["p_long"] > 0.5).to_numpy()
    realized = np.where(v10_long, L, S)                       # bps if we follow V10's call
    dollar_bear = (m["usd_mom24"] < 0).to_numpy()             # dollar weakening -> bullish gold
    agree = (v10_long & dollar_bear) | (~v10_long & ~dollar_bear)  # V10-long+dollar-down, or V10-short+dollar-up
    line(f"\n  follow-V10 realized K96, split by dollar agreement (usd_mom24 sign):")
    line(f"    AGREE   (n={agree.sum():,}): mean={realized[agree].mean():+.1f} bps  win={(realized[agree]>0).mean():.3f}")
    line(f"    DISAGREE(n={(~agree).sum():,}): mean={realized[~agree].mean():+.1f} bps  win={(realized[~agree]>0).mean():.3f}")
    line(f"    => gap (agree - disagree) = {realized[agree].mean()-realized[~agree].mean():+.1f} bps")
    (OUT / "SPIKE_RESULTS.txt").write_text("\n".join(out) + "\n")
    line(f"\n[SPIKE] wrote {OUT/'SPIKE_RESULTS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
