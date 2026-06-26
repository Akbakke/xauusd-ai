"""Decisive behavioral test (user vedtak 2026-06-26): does the 6-year-trained entry-IQL
(FOLD_FINAL, train ~2020-2026) LONG the down-regime LESS than the deployed FOLD_1
(chronological block-0, ~2020-mid-2021)?

The user's claim: the live bot keeps longing dips in a down market because the deployed
brain only ever saw the 2020-22 up-trend. If true, FOLD_FINAL (which saw the 2025-26
down-regime) should take FEWER longs in TREND_DOWN — especially in 2025-2026.

Read-only. Joins trend_regime + realized terminal pnl from the SOURCE forward_outcome
per_week onto each model's phase1 inference (action_label_v1 / advantage_over_skip_v1 /
q_*). No retrain, no contract touch.
"""
from __future__ import annotations
import glob, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
INF = {
    "FOLD_FINAL_6yr": sys.argv[1],   # trained on ALL 6yr incl 2025-26 -> IN-SAMPLE on block[4]
    "FOLD_1_deployed": sys.argv[2],  # trained block[0] (~2020-21) -> OOT on block[4]
}
if len(sys.argv) > 3:
    INF["FOLD_3_thru2024"] = sys.argv[3]  # trained block[0-2] (~thru 2024) -> OOT on block[4]
# conviction-gate live thr (op #2): raw_adv >= -67 -> TAKE; side = argmax take-Q
GATE_THR = -67.0


def load_inf(d):
    files = sorted(glob.glob(d.rstrip("/") + "/per_week/*.parquet"))
    cols = ["candidate_uid", "decision_ts_utc", "action_label_v1", "raw_action_label_v1",
            "advantage_over_skip_v1", "q_skip_v1", "q_take_long_v1", "q_take_short_v1"]
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    return df


HARD_STOP = -80.0  # live GX1_EXIT_HARD_STOP_BPS


def load_src():
    files = sorted(glob.glob(SRC + "/*.parquet"))
    cols = ["candidate_uid", "decision_ts_utc", "trend_regime", "vol_regime",
            "take_now_long_terminal_pnl_at_K96_v1", "take_now_short_terminal_pnl_at_K96_v1",
            "take_now_long_mae_bps_at_K96_v1", "take_now_short_mae_bps_at_K96_v1"]
    out = []
    for f in files:
        avail = pq.ParquetFile(f).schema_arrow.names
        out.append(pd.read_parquet(f, columns=[c for c in cols if c in avail]))
    return pd.concat(out, ignore_index=True)


def gated_action(df):
    """Replay the live conviction-gate (op#2): TAKE if raw_adv>=thr, side=argmax(take-Q)."""
    a = np.where(df["advantage_over_skip_v1"].to_numpy() >= GATE_THR,
                 np.where(df["q_take_long_v1"].to_numpy() >= df["q_take_short_v1"].to_numpy(),
                          "TAKE_LONG", "TAKE_SHORT"),
                 "SKIP")
    return a


def dist(df, label):
    n = len(df)
    if n == 0:
        return f"  {label:22s} n=0"
    argmax = df["action_label_v1"].value_counts(normalize=True)
    gated = pd.Series(gated_action(df)).value_counts(normalize=True)
    return (f"  {label:22s} n={n:6d} | ARGMAX skip={argmax.get('SKIP',0):.2f} "
            f"long={argmax.get('TAKE_LONG',0):.2f} short={argmax.get('TAKE_SHORT',0):.2f} "
            f"|| GATED(thr{GATE_THR:.0f}) take={1-gated.get('SKIP',0):.2f} "
            f"long={gated.get('TAKE_LONG',0):.2f} short={gated.get('TAKE_SHORT',0):.2f}")


def main():
    src = load_src().drop_duplicates(["candidate_uid"])
    src["ts"] = pd.to_datetime(src["decision_ts_utc"], utc=True)
    print(f"src rows={len(src)}  {src['ts'].min()} -> {src['ts'].max()}")
    src["period"] = np.where(src["ts"] >= "2025-01-01", "2025-2026(DOWN)", "2020-2024")

    models = {}
    for name, d in INF.items():
        m = load_inf(d).drop_duplicates(["candidate_uid"])
        j = m.merge(src[["candidate_uid", "trend_regime", "vol_regime", "period",
                         "take_now_long_terminal_pnl_at_K96_v1",
                         "take_now_short_terminal_pnl_at_K96_v1",
                         "take_now_long_mae_bps_at_K96_v1",
                         "take_now_short_mae_bps_at_K96_v1"]],
                    on="candidate_uid", how="inner")
        models[name] = j
        print(f"{name}: inf={len(m)} joined={len(j)}")

    for period in ["2020-2024", "2025-2026(DOWN)"]:
        print(f"\n================= PERIOD {period} =================")
        for reg in ["TREND_DOWN", "TREND_NEUTRAL", "TREND_UP"]:
            print(f"\n--- regime {reg} ---")
            for name, j in models.items():
                sub = j[(j["period"] == period) & (j["trend_regime"] == reg)]
                print(dist(sub, name))

    # ── LOAD-BEARING: high-conviction subset (the bot trades the conviction tail, not the
    #    blanket average). advantage_over_skip_v1 is on its own scale (P90~0.9-7.5), so the
    #    live -67 thr saturates here; use a percentile cut to approximate live coverage. ──
    print("\n================= LOAD-BEARING: high-conviction longs in TREND_DOWN, 2025-2026 =================")
    print("(side=argmax take-Q; TERMINAL vs STOP-ADJ where MAE<=-80 realizes the live hard-stop)")
    for cov in [0.35, 0.10]:
        print(f"\n--- top {int(cov*100)}% conviction (by advantage_over_skip_v1, within all candidates) ---")
        for name, j in models.items():
            sub = j[(j["period"] == "2025-2026(DOWN)") & (j["trend_regime"] == "TREND_DOWN")].copy()
            if len(sub) == 0:
                print(f"  {name:22s} n=0"); continue
            thr = j["advantage_over_skip_v1"].quantile(1 - cov)  # global conviction cut
            hc = sub[sub["advantage_over_skip_v1"] >= thr]
            side = np.where(hc["q_take_long_v1"].to_numpy() >= hc["q_take_short_v1"].to_numpy(),
                            "LONG", "SHORT")
            hc = hc.assign(side=side)
            longs = hc[hc["side"] == "LONG"]
            term = pd.to_numeric(longs["take_now_long_terminal_pnl_at_K96_v1"], errors="coerce")
            mae = pd.to_numeric(longs["take_now_long_mae_bps_at_K96_v1"], errors="coerce")  # +magnitude
            stop_adj = np.where(mae >= -HARD_STOP, HARD_STOP, term)  # MAE breaches 80bps -> -80 stop
            hit = (mae >= -HARD_STOP).mean()
            print(f"  {name:22s} n={len(hc):5d} long-share={ (hc['side']=='LONG').mean():.3f} "
                  f"longs_n={len(longs):4d} | TERM mean={term.mean():6.1f} win={(term>0).mean():.3f} "
                  f"| STOP-ADJ mean={np.nanmean(stop_adj):6.1f} | %hit-(-80)={hit:.3f}")

    # ── CLEAN OOT: block[4] = FOLD_3's true held-out (last 1/5 by time). FOLD_1 & FOLD_3 are
    #    BOTH OOT here; FOLD_FINAL is IN-SAMPLE (trained on it) — shown only as a memorization
    #    ceiling. This is the FAIR test of "does more recent training help the down-regime". ──
    any_j = next(iter(models.values()))
    src_ts = src.set_index("candidate_uid")["ts"]
    b4_cut = src["ts"].quantile(0.80)  # block[4] start (np.array_split into 5 -> last fifth)
    print(f"\n================= CLEAN OOT: block[4] (ts>={b4_cut}) TREND_DOWN =================")
    print("FOLD_1 & FOLD_3 = OOT (fair) | FOLD_FINAL = IN-SAMPLE (memorized, ceiling only)")
    for cov in [0.35, 0.10]:
        print(f"\n--- top {int(cov*100)}% conviction ---")
        for name, j in models.items():
            jj = j.copy()
            jj["ts"] = jj["candidate_uid"].map(src_ts)
            sub = jj[(jj["ts"] >= b4_cut) & (jj["trend_regime"] == "TREND_DOWN")].copy()
            if len(sub) == 0:
                print(f"  {name:22s} n=0"); continue
            thr = j["advantage_over_skip_v1"].quantile(1 - cov)
            hc = sub[sub["advantage_over_skip_v1"] >= thr]
            if len(hc) == 0:
                print(f"  {name:22s} n=0 (none clear conviction bar)"); continue
            side = np.where(hc["q_take_long_v1"].to_numpy() >= hc["q_take_short_v1"].to_numpy(), "LONG", "SHORT")
            hc = hc.assign(side=side)
            longs = hc[hc["side"] == "LONG"]
            term = pd.to_numeric(longs["take_now_long_terminal_pnl_at_K96_v1"], errors="coerce")
            mae = pd.to_numeric(longs["take_now_long_mae_bps_at_K96_v1"], errors="coerce")
            stop_adj = np.where(mae >= -HARD_STOP, HARD_STOP, term)
            tag = "[IN-SAMPLE]" if "FINAL" in name else "[OOT]"
            print(f"  {name:22s}{tag:11s} n={len(hc):4d} long-share={(hc['side']=='LONG').mean():.3f} "
                  f"longs_n={len(longs):4d} | TERM mean={term.mean():6.1f} win={(term>0).mean():.3f} "
                  f"| STOP-ADJ={np.nanmean(stop_adj):6.1f} %hit={(mae>=-HARD_STOP).mean():.3f}")

    print("\nREAD: if both folds still long TREND_DOWN ~the same AND the longs are net-positive "
          "even stop-adjusted, the 6yr retrain did NOT change behavior because longing the down-regime "
          "PAYS in the data — the pain is the MAE/stop path, an EXIT problem, not an entry-direction one. "
          "The block[4] OOT FOLD_3-vs-FOLD_1 row is the FAIR verdict on whether more recent data helps.")


if __name__ == "__main__":
    main()
