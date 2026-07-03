"""CORRECTED (post /batch bug-hunt 2026-06-26): FOLD_FINAL(6yr) vs FOLD_1(deployed-early) vs
FOLD_3(thru~2024) behavior on the down-regime, using the ACTUAL live gate.

Bugs the audit caught in v1 (all fixed here):
  - LIVE gate is raw_adv = max(q_take_long, q_take_short) - q_skip  (UN-clipped), thr = -37.71
    (live op, launch_live_practice.sh:119). v1 used advantage_over_skip_v1 (clipped at 0) with -67
    -> saturated to a no-op (take=1.00 for everyone). FIXED.
  - action labels are TAKE_LONG_NOW / TAKE_SHORT_NOW (v1 argmax panel was all-zero). FIXED (dropped panel).
  - FOLD_FINAL is OUT-OF-SAMPLE on the recent down-regime: its train ended ~2026-02-10 (val+test = last
    12%), and the TREND_DOWN cluster is ~April 2026 -> held-out. v1 wrongly tagged it in-sample
    ("memorization"). FIXED: all three folds are OOS on the down-regime; flagged.
  - compare all folds at the SAME absolute live gate (not per-fold global quantiles). FIXED.
NOTE: in this data trend_regime==TREND_DOWN is essentially ONE cluster (~April 2026) -> small sample.
Read-only. Cement/live untouched.
"""
from __future__ import annotations
import glob, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
INF = {"FOLD_FINAL_6yr": sys.argv[1], "FOLD_1_deployed": sys.argv[2]}
if len(sys.argv) > 3:
    INF["FOLD_3_thru2024"] = sys.argv[3]
LIVE_THR = -37.71      # live conviction gate on UN-clipped raw_adv
HARD_STOP = 80.0       # MAE (positive magnitude) >= 80 -> -80 stop fires
# per-fold train-end (chronological): block[0]=0.2, block[0:3]=0.6, final=0.88 of time-sorted candidates
FOLD_TRAIN_END_Q = {"FOLD_1_deployed": 0.20, "FOLD_3_thru2024": 0.60, "FOLD_FINAL_6yr": 0.88}


def load_inf(d):
    files = sorted(glob.glob(d.rstrip("/") + "/per_week/*.parquet"))
    cols = ["candidate_uid", "q_skip_v1", "q_take_long_v1", "q_take_short_v1"]
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def load_src():
    files = sorted(glob.glob(SRC + "/*.parquet"))
    cols = ["candidate_uid", "decision_ts_utc", "trend_regime",
            "take_now_long_terminal_pnl_at_K96_v1", "take_now_short_terminal_pnl_at_K96_v1",
            "take_now_long_mae_bps_at_K96_v1", "take_now_short_mae_bps_at_K96_v1"]
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def main():
    src = load_src().drop_duplicates(["candidate_uid"])
    src["ts"] = pd.to_datetime(src["decision_ts_utc"], utc=True)
    ts_q = {k: src["ts"].quantile(q) for k, q in FOLD_TRAIN_END_Q.items()}
    print(f"src rows={len(src)}  {src['ts'].min()} -> {src['ts'].max()}")
    dn = src[src["trend_regime"] == "TREND_DOWN"]
    print(f"TREND_DOWN rows={len(dn)}  span {dn['ts'].min()} -> {dn['ts'].max()}  "
          f"(median {dn['ts'].quantile(0.5)})")
    for k, q in ts_q.items():
        print(f"  {k}: train-end ~ {q}  -> down-regime OOS? {(dn['ts'] > q).mean()*100:.0f}% of down rows are after train-end")

    models = {}
    for name, d in INF.items():
        m = load_inf(d).drop_duplicates(["candidate_uid"])
        j = m.merge(src, on="candidate_uid", how="inner")
        j["raw_adv"] = np.maximum(j["q_take_long_v1"], j["q_take_short_v1"]) - j["q_skip_v1"]
        j["gated"] = j["raw_adv"] >= LIVE_THR
        j["side"] = np.where(j["q_take_long_v1"] >= j["q_take_short_v1"], "LONG", "SHORT")
        models[name] = j
        print(f"{name}: joined={len(j)}  overall take-rate @{LIVE_THR}={j['gated'].mean()*100:.1f}%")

    print(f"\n========== LIVE GATE (raw_adv>={LIVE_THR}) on TREND_DOWN — what the bot actually trades ==========")
    print("(all folds OOS here; realized = side's terminal; STOP-ADJ: MAE>=80 -> -80)")
    hdr = f"  {'fold':18s} {'take%':>6s} {'long%':>6s} {'takes_n':>8s} | {'TERM':>7s} {'win':>6s} {'STOP-ADJ':>9s} {'%hit-80':>8s}"
    print(hdr)
    for name, j in models.items():
        sub = j[j["trend_regime"] == "TREND_DOWN"]
        g = sub[sub["gated"]]
        if len(g) == 0:
            print(f"  {name:18s}  (no gated takes)"); continue
        longn = (g["side"] == "LONG").mean()
        # realized per take: long->long pnl/mae, short->short pnl/mae
        term = np.where(g["side"] == "LONG",
                        pd.to_numeric(g["take_now_long_terminal_pnl_at_K96_v1"], errors="coerce"),
                        pd.to_numeric(g["take_now_short_terminal_pnl_at_K96_v1"], errors="coerce"))
        mae = np.where(g["side"] == "LONG",
                       pd.to_numeric(g["take_now_long_mae_bps_at_K96_v1"], errors="coerce"),
                       pd.to_numeric(g["take_now_short_mae_bps_at_K96_v1"], errors="coerce"))
        stop_adj = np.where(mae >= HARD_STOP, -HARD_STOP, term)
        print(f"  {name:18s} {sub['gated'].mean()*100:6.1f} {longn*100:6.1f} {len(g):8d} | "
              f"{np.nanmean(term):7.1f} {np.nanmean(term>0):6.3f} {np.nanmean(stop_adj):9.1f} "
              f"{np.nanmean(mae>=HARD_STOP):8.3f}")

    print(f"\n========== GATED LONGS ONLY on TREND_DOWN (the cluster you flagged) ==========")
    print(hdr.replace('long%','     ').replace('take%','longs%'))
    for name, j in models.items():
        sub = j[j["trend_regime"] == "TREND_DOWN"]
        g = sub[sub["gated"] & (sub["side"] == "LONG")]
        if len(g) == 0:
            print(f"  {name:18s}  (no gated longs)"); continue
        term = pd.to_numeric(g["take_now_long_terminal_pnl_at_K96_v1"], errors="coerce")
        mae = pd.to_numeric(g["take_now_long_mae_bps_at_K96_v1"], errors="coerce")
        stop_adj = np.where(mae >= HARD_STOP, -HARD_STOP, term)
        share = (sub["gated"] & (sub["side"] == "LONG")).sum() / max(1, sub["gated"].sum())
        print(f"  {name:18s} {share*100:6.1f} {'':6s} {len(g):8d} | {term.mean():7.1f} "
              f"{(term>0).mean():6.3f} {np.nanmean(stop_adj):9.1f} {(mae>=HARD_STOP).mean():8.3f}")

    # ---- CLEAN: region OOS for ALL THREE folds (ts > FOLD_FINAL train-end) = the April-2026 down cluster.
    #      Removes FOLD_FINAL's home-field advantage (it trained on 86% of all-history TREND_DOWN). ----
    OOS_CUT = ts_q["FOLD_FINAL_6yr"]
    print(f"\n========== CLEAN ALL-OOS region (ts>{OOS_CUT}) TREND_DOWN — the ONLY fair fold comparison ==========")
    print("(all 3 folds genuinely OOS here; this is the April-2026 down cluster, small sample)")
    print(hdr)
    for name, j in models.items():
        sub = j[(j["trend_regime"] == "TREND_DOWN") & (j["ts"] > OOS_CUT)]
        g = sub[sub["gated"]]
        if len(g) == 0:
            print(f"  {name:18s}  n=0"); continue
        term = np.where(g["side"] == "LONG",
                        pd.to_numeric(g["take_now_long_terminal_pnl_at_K96_v1"], errors="coerce"),
                        pd.to_numeric(g["take_now_short_terminal_pnl_at_K96_v1"], errors="coerce"))
        mae = np.where(g["side"] == "LONG",
                       pd.to_numeric(g["take_now_long_mae_bps_at_K96_v1"], errors="coerce"),
                       pd.to_numeric(g["take_now_short_mae_bps_at_K96_v1"], errors="coerce"))
        stop_adj = np.where(mae >= HARD_STOP, -HARD_STOP, term)
        print(f"  {name:18s} {sub['gated'].mean()*100:6.1f} {(g['side']=='LONG').mean()*100:6.1f} {len(g):8d} | "
              f"{np.nanmean(term):7.1f} {np.nanmean(term>0):6.3f} {np.nanmean(stop_adj):9.1f} "
              f"{np.nanmean(mae>=HARD_STOP):8.3f}  (n_down={len(sub)})")

    print("\nVERDICT GUIDE: the CLEAN ALL-OOS row is the only fair test (removes FOLD_FINAL home-field). "
          "If FOLD_FINAL beats FOLD_1 there at the same gate -> the 6yr IQL helped OOS (small sample); "
          "if ~equal -> more years didn't move it. Supersedes v1's wrong 'in-sample memorization' dismissal.")


if __name__ == "__main__":
    main()
