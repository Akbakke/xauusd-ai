"""FOLD-DEPLOY comparison + margin-floor fold-invariance (2026-06-23, user "ja" to fold run).

Companion to oot_live_policy_falsification_20260623.py (reuses its live-policy reconstruction + helpers).
The live bot deploys FOLD_1 — trained on the EARLIEST block only (chronological walk-forward, n_folds=3,
5 equal-count time-blocks: FOLD_k trains blocks[0:k+1], tests block[k+1]). So:
  FOLD_1 train≈2020-11..2022-03 (OLDEST), test block2 (2023-06..2024-11)
  FOLD_2 train≈2020-11..2023-06,         test block3 (2024-11..2025-11)
  FOLD_3 train≈2020-11..2024-11 (NEWEST), test block4 (2025-11..2026-05)
=> 2026 is genuine OOT for ALL three (none trained past ~2024-11). Two questions:
  (1) DEPLOY: at EQUAL coverage (match FOLD_1's live take-rate), which fold's policy picks better trades
      on 2026 (the live regime) and on each fold's own held-out test block?
  (2) INVARIANCE: is the margin-floor verdict (no-op) the same across folds?
margin is V10-derived (fold-INDEPENDENT) — identical across folds; only raw_adv / the take-SET differ.
Outcome = terminal_pnl@K96 held-to-horizon (entry-side label), same caveat as the parent analysis.
NO retrain, NO deploy, NO heavy exit replay.
"""
from __future__ import annotations
import glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

# reuse the parent's env pins (DIPFIX both, THR -37.71) + helpers (one-truth)
from gx1.research.oot_live_policy_falsification_20260623 import (
    live_size, sharpe, wsharpe, wmean, BUNDLE, VARIANT, AGG, FO, K, THR, SKIP, LONG, SHORT)
from gx1.runtime.entry_iql_v2_adapter import EntryIQLV2Adapter
from gx1.scripts.augment_forward_outcome_v2 import attach_group_a_dip_struct_ctx_columns
from gx1.execution.v12_entry_iql_live import apply_dipfix_overlay

FOLDS = ["FOLD_1", "FOLD_2", "FOLD_3"]
# fold OOT test-block date bounds (approx, from forward_outcome 5-quantile; for labeling only)
TEST_BLOCK = {"FOLD_1": ("2023-06-23", "2024-11-22"),
              "FOLD_2": ("2024-11-22", "2025-11-10"),
              "FOLD_3": ("2025-11-10", "2026-05-26")}


def build():
    adapters = {f: EntryIQLV2Adapter.load(artifact_root=BUNDLE, variant=VARIANT, fold_id=f,
                                          aggregator=AGG, beta=1.0, prefer_cuda=False, min_advantage_bps=0.0)
                for f in FOLDS}
    files = sorted(glob.glob(f"{FO}/*.parquet"))
    rows = []
    for wi, fp in enumerate(files):
        df = pd.read_parquet(fp)
        if len(df) == 0:
            continue
        if "time" not in df.columns and "decision_ts_utc" in df.columns:
            df["time"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
        df = attach_group_a_dip_struct_ctx_columns(df, journal_label="foldcmp")
        cds = df.to_dict("records")
        preds = {f: adapters[f].predict(cds) for f in FOLDS}
        for i, c in enumerate(cds):
            tl = c.get(f"take_now_long_terminal_pnl_at_{K}_v1")
            ts_ = c.get(f"take_now_short_terminal_pnl_at_{K}_v1")
            atr = c.get("atr_bps", c.get("_v1_atr14_canon_v1", 14.0))
            base = dict(uid=c.get("candidate_uid"), ts=c.get("decision_ts_utc"), session=c.get("session"),
                        margin=float(c.get("margin", np.nan)),
                        atr=float(atr) if atr is not None and not pd.isna(atr) else 14.0,
                        tl=float(tl) if tl is not None and not pd.isna(tl) else np.nan,
                        ts_=float(ts_) if ts_ is not None and not pd.isna(ts_) else np.nan)
            for f in FOLDS:
                q = preds[f][i].q_per_action_v1
                qs, ql, qsh = float(q[SKIP]), float(q[LONG]), float(q[SHORT])
                rows.append(dict(**base, fold=f, raw_adv=max(ql, qsh) - qs,
                                 side_argmax=(LONG if ql >= qsh else SHORT)))
        if (wi + 1) % 50 == 0:
            print(f"  ...{wi+1}/{len(files)} weeks")
    u = pd.DataFrame(rows)
    u["ts"] = pd.to_datetime(u["ts"], utc=True)
    u["sz"] = live_size(u["margin"].values, u["atr"].values)
    return u


def main():
    u = build()
    u.to_parquet("/tmp/fold_comparison_universe_20260623.parquet", index=False)
    o26 = u["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")
    OUT = {"meta": dict(n_per_fold=int(len(u) / 3), thr_fold1=THR, K=K)}

    # equal-coverage: FOLD_1 take-frac at live THR (gate only, pre-dipfix) → match per fold
    f1 = u[u.fold == "FOLD_1"]
    target_cov = float((f1["raw_adv"] >= THR).mean())
    thr_f = {"FOLD_1": THR}
    for f in FOLDS[1:]:
        thr_f[f] = float(np.quantile(u[u.fold == f]["raw_adv"].values, 1 - target_cov))
    OUT["equal_coverage_target"] = round(target_cov, 4)
    OUT["thr_per_fold"] = {f: round(v, 3) for f, v in thr_f.items()}
    print(f"\nequal-coverage target (FOLD_1 gate frac @ {THR}) = {target_cov:.3f}")
    print("per-fold THR (gate-only, equal coverage):", {f: round(v, 2) for f, v in thr_f.items()})

    def arm(d):
        p, w = d["pnl"].values, d["sz"].values
        return dict(n=len(d), win=round(float((p > 0).mean()), 3), exp=round(float(p.mean()), 1),
                    wexp=round(wmean(p, w), 1), wsharpe=round(wsharpe(p, w), 4),
                    wtot=round(float((p * w).sum()), 0))

    # ── Q1 DEPLOY: each fold's gate-policy (equal coverage, NO dipfix for a clean fold-vs-fold) on slices ──
    print("\n" + "=" * 84 + "\nQ1 — DEPLOY: each fold @ equal coverage, realized terminal_pnl@K96\n" + "=" * 84)
    for f in FOLDS:
        d = u[u.fold == f].copy()
        take = d[d["raw_adv"] >= thr_f[f]].copy()
        take["pnl"] = np.where(take["side_argmax"] == LONG, take["tl"], take["ts_"])
        take = take.dropna(subset=["pnl"])
        # slices: 2026 (OOT for all) + this fold's OWN held-out test block
        lo, hi = TEST_BLOCK[f]
        own = take[(take["ts"] >= pd.Timestamp(lo, tz="UTC")) & (take["ts"] < pd.Timestamp(hi, tz="UTC"))]
        t26 = take[take["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")]
        OUT.setdefault("Q1_deploy", {})[f] = dict(thr=round(thr_f[f], 3),
            on_2026=arm(t26), on_own_testblock=arm(own), own_block=f"{lo}..{hi}")
        print(f"  {f}: thr={thr_f[f]:+.2f}")
        print(f"      on 2026 (OOT all folds):  {arm(t26)}")
        print(f"      on own test block {lo[:7]}..{hi[:7]}: {arm(own)}")

    # ── Q2 INVARIANCE: margin-floor A vs B per fold on 2026 ──
    print("\n" + "=" * 84 + "\nQ2 — MARGIN-FLOOR INVARIANCE: Arm A (gate) vs B (+margin>=0.47) per fold, 2026\n" + "=" * 84)
    for f in FOLDS:
        d = u[(u.fold == f) & o26].copy()
        take = d[d["raw_adv"] >= thr_f[f]].copy()
        take["pnl"] = np.where(take["side_argmax"] == LONG, take["tl"], take["ts_"])
        take = take.dropna(subset=["pnl"])
        A = take
        B = take[take["margin"] >= 0.47]
        dropped = take[take["margin"] < 0.47]
        OUT.setdefault("Q2_marginfloor_2026", {})[f] = dict(A=arm(A), B=arm(B),
            dropped_n=int(len(dropped)), dropped_win=round(float((dropped["pnl"] > 0).mean()), 3) if len(dropped) else None,
            dropped_exp=round(float(dropped["pnl"].mean()), 1) if len(dropped) else None)
        print(f"  {f}:  A {arm(A)}")
        print(f"        B {arm(B)}   dropped n={len(dropped)} win={ (dropped['pnl']>0).mean() if len(dropped) else float('nan'):.3f} exp={dropped['pnl'].mean() if len(dropped) else float('nan'):+.1f}")

    with open("/tmp/fold_comparison_20260623.json", "w") as fp:
        json.dump(OUT, fp, indent=2, default=str)
    print("\n→ /tmp/fold_comparison_20260623.json + /tmp/fold_comparison_universe_20260623.parquet")


if __name__ == "__main__":
    main()
