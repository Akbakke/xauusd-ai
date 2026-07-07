"""FALSIFY the conviction signal on the ACTUAL LIVE-POLICY entry universe (2026-06-23, user falsification).

WHY A NEW FILE (rule 7): oot_conviction_gate_audit.py measures on the FULL all-candidate set with a
MARGIN-percentile gate, and its raw_adv≈0.04 number is QUOTED from a code comment — raw_adv is NOT in
forward_outcome, so that script could not measure it. This analysis REQUIRES entry-IQL inference to obtain
the live gate metric (raw_adv = best_take_q − q_skip) and reconstruct the EXACT live policy:
  • conviction gate: raw_adv ≥ GX1_CONVICTION_THR (−37.71), side = argmax(TAKE-Q)  [v12_entry_iql_live.predict L589-605]
  • DIPFIX overlay (one-truth apply_dipfix_overlay, live env mode=both)
  • max_trades / portfolio cap IGNORED at entry level (per user)
The outcome is forward_outcome terminal_pnl @K96 (HELD-TO-HORIZON, the cheap entry-side label) — NOT the
realized exit-stack PnL (that needs the heavy phase6 replay this analysis is explicitly trying to avoid).
So every verdict here is about ENTRY-SIDE SIGNAL QUALITY, with the exit stack downstream + ~common to all
signals. SIDE comes from the entry-IQL argmax(TAKE-Q) (NOT the V10 p_long lean the old audit used — they
differ exactly on the DIPFIX/forced-short cells).

Reports Del 1 (corr/spearman/MI table — on TAKE universe AND all-candidates), Del 2 (raw_adv + margin
deciles), Del 3 (margin among raw_adv-PASS + loss attribution), Del 4 (Arm A/B/C entry-selection sim),
split OOT-late + 2026. NO retrain, NO deploy, NO heavy exit replay.

Run: GX1_ENTRY_DIPFIX=1 GX1_ENTRY_DIPFIX_MODE=both .venv/bin/python -m gx1.research.oot_live_policy_falsification_20260623
"""
from __future__ import annotations

import glob
import json
import os

# Live op-env MUST be set before importing v12_entry_iql_live (flags read at import) — match live EXACTLY.
# CONVICTION_THR default in the module is −34.2; the LIVE value is −37.71 (launcher pin) — pin it here or
# the analysis silently measures the wrong gate (caught in smoke 2026-06-23).
os.environ.setdefault("GX1_CONVICTION_GATE", "1")
os.environ.setdefault("GX1_CONVICTION_THR", "-37.71")
os.environ.setdefault("GX1_ENTRY_DIPFIX", "1")
os.environ.setdefault("GX1_ENTRY_DIPFIX_MODE", "both")
os.environ.setdefault("GX1_REGIME_V4", "1")
os.environ.setdefault("GX1_TREND_REGIME_FROM_D1", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

from gx1.runtime.entry_iql_v2_adapter import EntryIQLV2Adapter
from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core
from gx1.scripts.augment_forward_outcome_v2 import attach_group_a_dip_struct_ctx_columns
from gx1.execution.v12_entry_iql_live import apply_dipfix_overlay, CONVICTION_THR

# ── live constants (one-truth) ────────────────────────────────────────────
BUNDLE = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/entry_iql_volbal_20260611"
VARIANT, FOLD, AGG = "R_WAIT_OPP_K96_LAM50_SYM", "FOLD_1", "mean"   # live deploys FOLD_1
FO = "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/forward_outcome_clean/per_week"
K = "K96"
THR = float(CONVICTION_THR)   # −37.71 live
assert abs(THR - (-37.71)) < 1e-6, f"THR={THR} ≠ live −37.71 — export GX1_CONVICTION_THR=-37.71 before run"
SKIP, LONG, SHORT = iql_core.ACTION_SKIP_ID, iql_core.ACTION_TAKE_LONG_NOW_ID, iql_core.ACTION_TAKE_SHORT_NOW_ID
SIZE_REF, ATR_REF = 0.3318, 14.0   # live margin²-sizing (contract operating_point)


def live_size(margin, atr):  # matches contract + oot_conviction_gate_audit.live_size
    return np.clip((np.maximum(margin, 0.0) ** 2) / SIZE_REF * ATR_REF / np.maximum(atr, ATR_REF), 0.5, 2.0)


def sharpe(p):
    p = np.asarray(p, float)
    return float(np.mean(p) / (np.std(p) + 1e-9)) if len(p) else float("nan")


def wsharpe(p, w):  # size-weighted Sharpe: weighted mean / weighted std of per-trade pnl
    p, w = np.asarray(p, float), np.asarray(w, float)
    if len(p) == 0 or w.sum() == 0:
        return float("nan")
    m = np.average(p, weights=w)
    v = np.average((p - m) ** 2, weights=w)
    return float(m / (np.sqrt(v) + 1e-9))


def wmean(p, w):
    p, w = np.asarray(p, float), np.asarray(w, float)
    return float(np.average(p, weights=w)) if (len(p) and w.sum() > 0) else float("nan")


def build_universe():
    """Run entry-IQL inference on every forward_outcome week → the LIVE-POLICY entry universe."""
    adapter = EntryIQLV2Adapter.load(artifact_root=BUNDLE, variant=VARIANT, fold_id=FOLD,
                                     aggregator=AGG, beta=1.0, prefer_cuda=False, min_advantage_bps=0.0)
    files = sorted(glob.glob(f"{FO}/*.parquet"))
    rows = []
    n_dipfix = 0
    for wi, fp in enumerate(files):
        df = pd.read_parquet(fp)
        if len(df) == 0:
            continue
        if "time" not in df.columns and "decision_ts_utc" in df.columns:
            df["time"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
        df = attach_group_a_dip_struct_ctx_columns(df, journal_label="falsification")
        cds = df.to_dict("records")
        recs = adapter.predict(cds)
        for r, c in zip(recs, cds):
            q = r.q_per_action_v1
            q_skip, q_long, q_short = float(q[SKIP]), float(q[LONG]), float(q[SHORT])
            raw_adv = max(q_long, q_short) - q_skip                       # live gate metric
            side_argmax = LONG if q_long >= q_short else SHORT
            a_id = side_argmax if raw_adv >= THR else SKIP                # conviction gate (live L593)
            a_id, mk = apply_dipfix_overlay(a_id, c)                       # one-truth DIPFIX
            if mk:
                n_dipfix += 1
            # outcome / state for the FINAL action's side (long→long label, short→short label)
            tl = c.get(f"take_now_long_terminal_pnl_at_{K}_v1")
            ts_ = c.get(f"take_now_short_terminal_pnl_at_{K}_v1")
            mael = c.get(f"take_now_long_mae_bps_at_{K}_v1")
            maes = c.get(f"take_now_short_mae_bps_at_{K}_v1")
            side = "long" if a_id == LONG else "short" if a_id == SHORT else "skip"
            pnl = (tl if side == "long" else ts_ if side == "short" else np.nan)
            mae = (mael if side == "long" else maes if side == "short" else np.nan)
            q_chosen = q_long if a_id == LONG else q_short if a_id == SHORT else max(q_long, q_short)
            q_alt = q_short if a_id == LONG else q_long if a_id == SHORT else min(q_long, q_short)
            atr = c.get("atr_bps", c.get("_v1_atr14_canon_v1", 14.0))
            rows.append(dict(
                uid=c.get("candidate_uid"), ts=c.get("decision_ts_utc"), session=c.get("session"),
                action=a_id, is_take=int(a_id != SKIP), side=side,
                raw_adv=raw_adv, q_skip=q_skip, q_long=q_long, q_short=q_short,
                take_minus_skip=q_chosen - q_skip, take_minus_alt=q_chosen - q_alt,
                margin=float(c.get("margin", np.nan)),
                p_long=float(c.get("p_long", np.nan)), p_short=float(c.get("p_short", np.nan)),
                pnl=float(pnl) if pnl is not None and not pd.isna(pnl) else np.nan,
                mae=float(mae) if mae is not None and not pd.isna(mae) else np.nan,
                atr=float(atr) if atr is not None and not pd.isna(atr) else 14.0,
                # all-candidates argmax-side outcome (for the full-range signal comparison)
                pnl_argmax=float(tl if side_argmax == LONG else ts_) if (tl is not None and ts_ is not None and not pd.isna(tl) and not pd.isna(ts_)) else np.nan,
            ))
        if (wi + 1) % 50 == 0:
            print(f"  ...{wi+1}/{len(files)} weeks, rows={len(rows)}")
    u = pd.DataFrame(rows)
    u["ts"] = pd.to_datetime(u["ts"], utc=True)
    u["dgap"] = (u["p_long"] - u["p_short"]).abs()
    u["sz"] = live_size(u["margin"].values, u["atr"].values)
    print(f"[universe] candidates={len(u)}  TAKEs(conv+dipfix)={int(u.is_take.sum())}  dipfix_fired={n_dipfix}")
    return u


SIGNALS = ["raw_adv", "margin", "p_long", "dgap", "take_minus_skip", "take_minus_alt"]


def corr_table(df, outcome_col="pnl"):
    d = df.dropna(subset=[outcome_col] + SIGNALS)
    y = d[outcome_col].values.astype(float)
    out = {}
    if len(d) < 50:
        return {"n": int(len(d)), "note": "n<50 insufficient"}
    mi = mutual_info_regression(d[SIGNALS].values.astype(float), y, random_state=0)
    for i, s in enumerate(SIGNALS):
        x = d[s].values.astype(float)
        out[s] = dict(pearson=round(float(np.corrcoef(x, y)[0, 1]), 4),
                      spearman=round(float(spearmanr(x, y).correlation), 4),
                      mutual_info=round(float(mi[i]), 4))
    out["n"] = int(len(d))
    return out


def decile_report(df, by, outcome="pnl"):
    d = df.dropna(subset=[outcome, by, "sz"]).copy()
    if len(d) < 100:
        return {"note": f"n={len(d)}<100"}
    d["dec"] = pd.qcut(d[by].rank(method="first"), 10, labels=False) + 1
    res = []
    for dec in range(1, 11):
        g = d[d["dec"] == dec]
        p, w = g[outcome].values, g["sz"].values
        res.append(dict(decile=dec, n=int(len(g)),
                        win=round(float((p > 0).mean()), 4), exp=round(float(p.mean()), 2),
                        sharpe=round(sharpe(p), 4), mae=round(float(g["mae"].mean()), 1),
                        wexp=round(wmean(p, w), 2), wsharpe=round(wsharpe(p, w), 4)))
    return res


def arm_metrics(d):
    p, w = d["pnl"].values, d["sz"].values
    return dict(n=int(len(d)), win=round(float((p > 0).mean()), 4), exp=round(float(p.mean()), 2),
                sharpe=round(sharpe(p), 4), wexp=round(wmean(p, w), 2), wsharpe=round(wsharpe(p, w), 4),
                total=round(float(p.sum()), 0), wtotal=round(float((p * w).sum()), 0))


def main():
    u = build_universe()
    u.to_parquet("/tmp/live_policy_universe_20260623.parquet", index=False)
    med = u["ts"].median()
    masks = {"OOT-late (ts>median)": u["ts"] > med, "2026 (ts>=2026-01-01)": u["ts"] >= pd.Timestamp("2026-01-01", tz="UTC")}
    OUT = {"meta": dict(thr=THR, K=K, fold=FOLD, variant=VARIANT, n_candidates=int(len(u)),
                        n_take=int(u.is_take.sum()), median_ts=str(med),
                        outcome="terminal_pnl@K96 HELD-TO-HORIZON (NOT exit-stack)")}

    for split, m in masks.items():
        takes = u[m & (u.is_take == 1)]
        allc = u[m]
        print(f"\n{'='*78}\nSPLIT: {split}   candidates={int(m.sum())}  TAKEs={len(takes)}\n{'='*78}")

        # ── DEL 1: corr table (TAKE universe = live reality; all-cand = fair full-range comparison) ──
        del1_take = corr_table(takes, "pnl")
        del1_all = corr_table(allc.assign(pnl=allc["pnl_argmax"]), "pnl")
        OUT.setdefault("del1_take_universe", {})[split] = del1_take
        OUT.setdefault("del1_all_candidates", {})[split] = del1_all
        print("DEL1 — signal vs terminal_pnl@K96")
        print(f"  TAKE universe (live, raw_adv range-restricted ≥{THR}):  n={del1_take.get('n')}")
        for s in SIGNALS:
            if s in del1_take:
                print(f"    {s:16s} pearson={del1_take[s]['pearson']:+.4f}  spearman={del1_take[s]['spearman']:+.4f}  MI={del1_take[s]['mutual_info']:.4f}")
        print(f"  ALL candidates (full range, argmax-side outcome):  n={del1_all.get('n')}")
        for s in SIGNALS:
            if s in del1_all:
                print(f"    {s:16s} pearson={del1_all[s]['pearson']:+.4f}  spearman={del1_all[s]['spearman']:+.4f}  MI={del1_all[s]['mutual_info']:.4f}")

        # ── DEL 2: deciles raw_adv + margin (on TAKE universe) ──
        OUT.setdefault("del2_raw_adv_deciles", {})[split] = decile_report(takes, "raw_adv")
        OUT.setdefault("del2_margin_deciles", {})[split] = decile_report(takes, "margin")
        for by in ("raw_adv", "margin"):
            print(f"\nDEL2 — {by} deciles (TAKE universe):")
            rep = OUT[f"del2_{by}_deciles"][split]
            if isinstance(rep, list):
                for r in rep:
                    print(f"    D{r['decile']:2d} n={r['n']:5d} win={r['win']:.3f} exp={r['exp']:+7.2f} "
                          f"sharpe={r['sharpe']:+.3f} mae={r['mae']:6.1f} | wexp={r['wexp']:+7.2f} wsharpe={r['wsharpe']:+.3f}")

        # ── DEL 3: among raw_adv-PASS (= the whole TAKE universe), margin buckets + loss attribution ──
        d3 = takes.dropna(subset=["pnl", "margin"]).copy()
        if len(d3) >= 100:
            d3["mb"] = pd.qcut(d3["margin"].rank(method="first"), 5, labels=["Q1lo", "Q2", "Q3", "Q4", "Q5hi"])
            buck = {}
            tot_loss = float(d3.loc[d3.pnl < 0, "pnl"].sum())
            for b in ["Q1lo", "Q2", "Q3", "Q4", "Q5hi"]:
                g = d3[d3.mb == b]
                loss = float(g.loc[g.pnl < 0, "pnl"].sum())
                buck[b] = dict(n=int(len(g)), margin_lo=round(float(g.margin.min()), 3), margin_hi=round(float(g.margin.max()), 3),
                               win=round(float((g.pnl > 0).mean()), 4), exp=round(float(g.pnl.mean()), 2),
                               neg_pnl_sum=round(loss, 0), pct_of_all_losses=round(100 * loss / (tot_loss - 1e-9), 1))
            OUT.setdefault("del3_margin_buckets_among_rawadv_pass", {})[split] = buck
            print(f"\nDEL3 — margin buckets among raw_adv-PASS (n={len(d3)}, total_neg_pnl={tot_loss:.0f}):")
            for b, v in buck.items():
                print(f"    {b:5s} margin[{v['margin_lo']:.2f},{v['margin_hi']:.2f}] n={v['n']:5d} win={v['win']:.3f} "
                      f"exp={v['exp']:+7.2f} negPnL={v['neg_pnl_sum']:+8.0f} ({v['pct_of_all_losses']:4.1f}% of losses)")

        # ── DEL 4: Arm A (live gate) / B (+margin floor 0.47) / C (margin primary, raw_adv removed, same N) ──
        armA = takes.dropna(subset=["pnl"])                                   # conviction gate + DIPFIX
        armB = armA[armA["margin"] >= 0.47]                                   # + margin floor
        nA = len(armA)
        pool = allc.dropna(subset=["pnl_argmax", "margin"]).copy()            # ALL candidates, argmax side
        pool = pool.assign(pnl=pool["pnl_argmax"])
        armC = pool.nlargest(nA, "margin") if nA <= len(pool) else pool       # top-N_A by margin, raw_adv ignored
        OUT.setdefault("del4_arms", {})[split] = dict(
            A_live_gate=arm_metrics(armA), B_plus_margin_floor=arm_metrics(armB),
            C_margin_primary_sameN=arm_metrics(armC),
            C_margin_thr=round(float(armC["margin"].min()), 4) if len(armC) else None)
        print(f"\nDEL4 — Arms (size-weighted in wexp/wsharpe):")
        for nm, a in OUT["del4_arms"][split].items():
            if isinstance(a, dict) and "n" in a:
                print(f"    {nm:24s} n={a['n']:5d} win={a['win']:.3f} exp={a['exp']:+7.2f} sharpe={a['sharpe']:+.3f} "
                      f"| wexp={a['wexp']:+7.2f} wsharpe={a['wsharpe']:+.3f} wtotal={a['wtotal']:+.0f}")

    with open("/tmp/live_policy_falsification_20260623.json", "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print("\n→ /tmp/live_policy_falsification_20260623.json  +  /tmp/live_policy_universe_20260623.parquet")


if __name__ == "__main__":
    main()
