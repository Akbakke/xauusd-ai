"""ENSEMBLE EPISTEMIC-DISAGREEMENT FALSIFICATION (2026-06-24, user). DIAGNOSTIC ONLY — no model, no prod
feature, no deploy. The LAST honest entry-side probe after 11 refuted tracks.

HYPOTHESIS: the live policy uses ONE fold (FOLD_1). The 3 folds are an ensemble; their DISAGREEMENT is a
model-EPISTEMIC-uncertainty signal that is DISTINCT from conviction (raw_adv = one model's margin). If the
ensemble knows something the single fold doesn't, then:
  (A) FOLD_1 takes where folds DISAGREE on side should be systematically WORSE (epistemic blind spots), and
  (B) an ensemble-VOTE policy (≥2/3 agree) should beat the single FOLD_1 policy OOT, and
  (C) a UNANIMITY gate (3/3 agree+take) should yield a cleaner book, and
  (D) ensemble agreement should add INCREMENTAL OOT info over raw_adv/margin.
If none survive strict-OOT + day-block bootstrap, KILL it — folds are redundant (fold-comparison already
found F1−F3 not significant, P=0.71; this measures whether disagreement is informative WHERE it occurs).

WHY A NEW FILE (rule 7): oot_fold_comparison_20260623 compares fold deploy CHOICE (which single fold);
oot_live_policy_falsification_20260623 runs ONE fold. Neither measures cross-fold AGREEMENT as an
abstention/selection signal. Reuses both modules' one-truth constants + adapter + DIPFIX (read-only).

Population: all 147,491 forward_outcome candidates, 3-fold entry-IQL inference. Outcome = terminal_pnl @K96
for the acted side (held-to-horizon entry label). Conviction gate −37.71 + DIPFIX = exact live policy.

Run: .venv/bin/python -m gx1.research.oot_ensemble_disagreement_20260624
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

# importing this module sets the live op-env (GX1_CONVICTION_THR=-37.71, DIPFIX=1, REGIME_V4=1) on import
from gx1.research.oot_live_policy_falsification_20260623 import FO, K, BUNDLE, VARIANT, AGG, THR
from gx1.runtime.entry_iql_v2_adapter import EntryIQLV2Adapter
from gx1.scripts import entry_iql_multi_head_gpu_core_v1 as iql_core
from gx1.scripts.augment_forward_outcome_v2 import attach_group_a_dip_struct_ctx_columns
from gx1.execution.v12_entry_iql_live import apply_dipfix_overlay

SKIP, LONG, SHORT = iql_core.ACTION_SKIP_ID, iql_core.ACTION_TAKE_LONG_NOW_ID, iql_core.ACTION_TAKE_SHORT_NOW_ID
FOLDS = ["FOLD_1", "FOLD_2", "FOLD_3"]
OUT_JSON = "/tmp/ensemble_disagreement_20260624.json"
RNG = np.random.default_rng(0)


def sharpe(p):
    p = np.asarray(p, float); p = p[np.isfinite(p)]
    return float(np.mean(p) / (np.std(p) + 1e-9)) if len(p) else float("nan")


def book(df, pnl_col="pnl"):
    p = df[pnl_col].dropna().values
    return dict(n=int(len(p)), mean_pnl=round(float(np.mean(p)), 2) if len(p) else None,
               win=round(float((p > 0).mean()), 4) if len(p) else None,
               sharpe=round(sharpe(p), 4), total=round(float(np.sum(p)), 1) if len(p) else None)


def build():
    adapters = {f: EntryIQLV2Adapter.load(artifact_root=BUNDLE, variant=VARIANT, fold_id=f,
                                           aggregator=AGG, beta=1.0, prefer_cuda=False, min_advantage_bps=0.0)
                for f in FOLDS}
    files = sorted(glob.glob(f"{FO}/*.parquet"))
    rows = []
    for fp in files:
        df = pd.read_parquet(fp)
        if len(df) == 0:
            continue
        if "time" not in df.columns and "decision_ts_utc" in df.columns:
            df["time"] = pd.to_datetime(df["decision_ts_utc"], utc=True)
        df = attach_group_a_dip_struct_ctx_columns(df, journal_label="ensemble")
        cds = df.to_dict("records")
        preds = {f: adapters[f].predict(cds) for f in FOLDS}
        for i, c in enumerate(cds):
            tl = c.get(f"take_now_long_terminal_pnl_at_{K}_v1")
            ts_ = c.get(f"take_now_short_terminal_pnl_at_{K}_v1")
            tl = float(tl) if tl is not None and not pd.isna(tl) else np.nan
            ts_ = float(ts_) if ts_ is not None and not pd.isna(ts_) else np.nan
            rec = dict(uid=c.get("candidate_uid"), ts=c.get("decision_ts_utc"), session=c.get("session"),
                       margin=float(c.get("margin", np.nan)), atr=float(c.get("atr_bps", 14.0) or 14.0),
                       p_long=float(c.get("p_long", np.nan)), tl=tl, ts_=ts_)
            for f in FOLDS:
                q = preds[f][i].q_per_action_v1
                qs, ql, qsh = float(q[SKIP]), float(q[LONG]), float(q[SHORT])
                side = LONG if ql >= qsh else SHORT
                radv = max(ql, qsh) - qs
                a_gate = side if radv >= THR else SKIP
                a_fin, _ = apply_dipfix_overlay(a_gate, c)   # one-truth DIPFIX (live op)
                rec[f"radv_{f}"] = radv
                rec[f"side_{f}"] = "long" if side == LONG else "short"   # directional argmax (pre-gate)
                rec[f"take_{f}"] = int(a_gate != SKIP)                   # conviction-gate pass
                rec[f"act_{f}"] = ("long" if a_fin == LONG else "short" if a_fin == SHORT else "skip")
            rows.append(rec)
    d = pd.DataFrame(rows)
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d["year"] = d["ts"].dt.year
    d["day"] = d["ts"].dt.floor("D")
    # ensemble epistemic features (on directional argmax + gate vote)
    sides = d[[f"side_{f}" for f in FOLDS]].values
    d["n_long_vote"] = (sides == "long").sum(1)
    d["agree_side"] = (d["n_long_vote"] == 3) | (d["n_long_vote"] == 0)      # unanimous direction
    d["maj_side"] = np.where(d["n_long_vote"] >= 2, "long", "short")
    d["n_take"] = d[[f"take_{f}" for f in FOLDS]].sum(1)                      # how many folds gate-pass
    d["radv_std"] = d[[f"radv_{f}" for f in FOLDS]].std(1)                    # epistemic spread of conviction
    d["radv_mean"] = d[[f"radv_{f}" for f in FOLDS]].mean(1)
    return d


def pnl_for(side, tl, ts_):
    return np.where(side == "long", tl, np.where(side == "short", ts_, np.nan))


def main():
    print("=" * 100)
    print("ENSEMBLE EPISTEMIC-DISAGREEMENT FALSIFICATION — entry-side, diagnostic only")
    print("=" * 100)
    d = build()
    d["pnl_f1"] = pnl_for(d["act_FOLD_1"].values, d["tl"].values, d["ts_"].values)   # live FOLD_1 book pnl
    n_all = len(d)
    f1_takes = d[d["act_FOLD_1"] != "skip"].copy()
    print(f"candidates={n_all:,}  FOLD_1 live takes={len(f1_takes):,}  "
          f"(2026 takes={int((f1_takes.year==2026).sum())})")
    # cross-fold agreement on FOLD_1's chosen direction among FOLD_1 takes
    OUT = {"meta": dict(n_candidates=n_all, n_f1_takes=len(f1_takes), label="terminal_pnl@K96",
                        thr=THR, folds=FOLDS)}

    # ─ ARM A — disagreement-as-abstention WITHIN FOLD_1 takes ─
    print("\n" + "─" * 100 + "\nARM A — FOLD_1 takes split by cross-fold SIDE agreement (all 3 argmax same side?)")
    ft = f1_takes.copy()
    ft["pnl"] = pnl_for(ft["act_FOLD_1"].values, ft["tl"].values, ft["ts_"].values)
    # does FOLD_1's chosen side match the unanimous direction?
    ft["folds_agree_f1side"] = (ft.apply(lambda r: (r["n_long_vote"] == 3 and r["act_FOLD_1"] == "long")
                                         or (r["n_long_vote"] == 0 and r["act_FOLD_1"] == "short"), axis=1))
    armA = {}
    for split, lbl in [(slice(None), "ALL"), (ft.year < 2026, "pre2026"), (ft.year == 2026, "2026")]:
        sub = ft if isinstance(split, slice) else ft[split]
        ag = sub[sub.folds_agree_f1side]
        dis = sub[~sub.folds_agree_f1side]
        armA[lbl] = dict(agree=book(ag), disagree=book(dis),
                         disagree_frac=round(float((~sub.folds_agree_f1side).mean()), 4))
        print(f"  [{lbl:7s}] agree n={armA[lbl]['agree']['n']:5d} pnl={armA[lbl]['agree']['mean_pnl']} "
              f"win={armA[lbl]['agree']['win']}  |  DISAGREE n={armA[lbl]['disagree']['n']:5d} "
              f"pnl={armA[lbl]['disagree']['mean_pnl']} win={armA[lbl]['disagree']['win']}  "
              f"(disagree_frac={armA[lbl]['disagree_frac']})")
    OUT["armA_disagreement_abstention"] = armA

    # ─ ARM B — ensemble-VOTE policy vs FOLD_1 policy (the "super smart" version) ─
    print("\n" + "─" * 100 + "\nARM B — ensemble-VOTE policy (>=2/3 gate-pass, majority side, +DIPFIX) vs FOLD_1, by year")
    dv = d.copy()
    # ensemble take = majority of folds gate-pass; side = majority argmax; then DIPFIX via the live overlay is
    # already baked per-fold act_*; here vote on the GATE+side directly (equal-weight ensemble decision)
    vote_take = dv["n_take"] >= 2
    dv["ens_act"] = np.where(vote_take, dv["maj_side"], "skip")
    dv["pnl_ens"] = pnl_for(dv["ens_act"].values, dv["tl"].values, dv["ts_"].values)
    armB = {}
    for yr, lbl in [(None, "ALL"), ("pre", "pre2026"), (2026, "2026")]:
        if lbl == "pre2026":
            sub = dv[dv.year < 2026]
        elif lbl == "2026":
            sub = dv[dv.year == 2026]
        else:
            sub = dv
        f1 = sub[sub.act_FOLD_1 != "skip"].assign(pnl=lambda x: pnl_for(x.act_FOLD_1.values, x.tl.values, x.ts_.values))
        en = sub[sub.ens_act != "skip"].assign(pnl=lambda x: x.pnl_ens)
        armB[lbl] = dict(fold1=book(f1), ensemble=book(en))
        print(f"  [{lbl:7s}] FOLD_1: n={armB[lbl]['fold1']['n']:5d} pnl={armB[lbl]['fold1']['mean_pnl']} "
              f"win={armB[lbl]['fold1']['win']} Sh={armB[lbl]['fold1']['sharpe']}  ||  "
              f"ENSEMBLE: n={armB[lbl]['ensemble']['n']:5d} pnl={armB[lbl]['ensemble']['mean_pnl']} "
              f"win={armB[lbl]['ensemble']['win']} Sh={armB[lbl]['ensemble']['sharpe']}")
    OUT["armB_ensemble_vote_policy"] = armB

    # ─ ARM C — UNANIMITY gate (3/3 take + agree side) as a quality filter on FOLD_1 takes ─
    print("\n" + "─" * 100 + "\nARM C — UNANIMITY filter: FOLD_1 takes where ALL 3 folds take AND agree side")
    ft["unanimous"] = (ft.n_take == 3) & ft.agree_side & ft.folds_agree_f1side
    armC = {}
    for split, lbl in [(ft.year < 2026, "pre2026"), (ft.year == 2026, "2026")]:
        sub = ft[split]
        uni = sub[sub.unanimous]
        rest = sub[~sub.unanimous]
        armC[lbl] = dict(unanimous=book(uni), non_unanimous=book(rest),
                         uni_frac=round(float(sub.unanimous.mean()), 4))
        print(f"  [{lbl:7s}] UNANIMOUS n={armC[lbl]['unanimous']['n']:5d} pnl={armC[lbl]['unanimous']['mean_pnl']} "
              f"win={armC[lbl]['unanimous']['win']} Sh={armC[lbl]['unanimous']['sharpe']}  |  rest n={armC[lbl]['non_unanimous']['n']:5d} "
              f"pnl={armC[lbl]['non_unanimous']['mean_pnl']} win={armC[lbl]['non_unanimous']['win']}  "
              f"(uni_frac={armC[lbl]['uni_frac']})")
    OUT["armC_unanimity_filter"] = armC

    # ─ ARM D — incremental OOT info: base (single-fold conviction) vs +ensemble epistemics ─
    print("\n" + "─" * 100 + "\nARM D — incremental OOT info on FOLD_1 takes (loss-pred), strict OOT both directions")
    ft["loss"] = (ft["pnl"] < 0).astype(int)
    ft["dir_score"] = ft["p_long"]
    base = ["radv_FOLD_1", "margin", "atr", "dir_score"]
    ens = ["n_take", "radv_std", "radv_mean", "radv_FOLD_2", "radv_FOLD_3",
           "n_long_vote", "agree_side"]
    dd = ft.dropna(subset=base + ["pnl"]).copy()
    dd["agree_side"] = dd["agree_side"].astype(int)
    for c in ens:
        dd[c] = dd[c].fillna(dd[c].median())
    armD = {}

    def fe(tr, te, feats):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(tr[feats].values, tr.loss.values)
        pr = clf.predict_proba(te[feats].values)[:, 1]
        return float(roc_auc_score(te.loss.values, pr)), float(spearmanr(pr, -te.pnl.values).correlation)

    pre = dd[dd.year < 2026]; o26 = dd[dd.year == 2026]
    for nm, tr, te in [("pre→2026", pre, o26), ("2026→pre", o26, pre)]:
        if len(tr) < 500 or len(te) < 500:
            armD[nm] = {"note": "insufficient"}; continue
        a0, r0 = fe(tr, te, base); a1, r1 = fe(tr, te, base + ens)
        armD[nm] = dict(auc_base=round(a0, 4), auc_ens=round(a1, 4), delta_auc=round(a1 - a0, 4),
                        rankic_base=round(r0, 4), rankic_ens=round(r1, 4), delta_rankic=round(r1 - r0, 4))
        print(f"  [{nm}] base AUC={a0:.4f}→+ens {a1:.4f} ΔAUC={a1-a0:+.4f} | "
              f"base RankIC={r0:+.4f}→+ens {r1:+.4f} ΔRankIC={r1-r0:+.4f}")
    OUT["armD_incremental_oot"] = armD

    # ─ ARM E — day-block bootstrap on the agreement→pnl gap (2026) ─
    print("\n" + "─" * 100 + "\nARM E — DAY-BLOCK BOOTSTRAP (10k) on agree−disagree mean-pnl gap (2026 FOLD_1 takes)")
    sub = ft[(ft.year == 2026) & np.isfinite(ft.pnl)][["day", "folds_agree_f1side", "pnl"]].copy()
    days = sub.day.unique()
    ag_by = {x: g.loc[g.folds_agree_f1side, "pnl"].values for x, g in sub.groupby("day")}
    di_by = {x: g.loc[~g.folds_agree_f1side, "pnl"].values for x, g in sub.groupby("day")}
    dl = list(days); aga = [ag_by[x] for x in dl]; dia = [di_by[x] for x in dl]; nd = len(dl)
    base_gap = float(sub.loc[sub.folds_agree_f1side, "pnl"].mean() - sub.loc[~sub.folds_agree_f1side, "pnl"].mean())
    boots = np.empty(10000)
    for b in range(10000):
        idx = RNG.integers(0, nd, nd)
        a = np.concatenate([aga[j] for j in idx]); di = np.concatenate([dia[j] for j in idx])
        boots[b] = (a.mean() if len(a) else np.nan) - (di.mean() if len(di) else np.nan)
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    OUT["armE_bootstrap"] = dict(agree_minus_disagree_pnl=round(base_gap, 2),
                                 ci=[round(float(lo), 2), round(float(hi), 2)], p=round(float(min(p, 1)), 5),
                                 n_disagree_2026=int((~sub.folds_agree_f1side).sum()))
    print(f"  agree−disagree mean pnl (2026) = {base_gap:+.2f} bps  95%CI[{lo:+.2f},{hi:+.2f}]  p={min(p,1):.4f}  "
          f"(n_disagree={int((~sub.folds_agree_f1side).sum())})")

    # ─ VERDICT ─
    print("\n" + "═" * 100 + "\nVERDICT")
    a26 = armA.get("2026", {})
    # A: disagreement takes materially worse 2026 (>10 bps gap) AND bootstrap CI>0
    cA = (a26.get("agree", {}).get("mean_pnl") or 0) - (a26.get("disagree", {}).get("mean_pnl") or 0) > 10 \
        and OUT["armE_bootstrap"]["ci"][0] > 0
    # B: ensemble vote beats FOLD_1 on 2026 (higher per-take pnl AND not just fewer trades) — Sharpe up
    b26 = armB.get("2026", {})
    cB = (b26.get("ensemble", {}).get("mean_pnl") or -1) > (b26.get("fold1", {}).get("mean_pnl") or 0) + 3 \
        and (b26.get("ensemble", {}).get("sharpe") or 0) > (b26.get("fold1", {}).get("sharpe") or 0)
    # C: unanimity filter lifts 2026 per-take pnl materially (>5 bps) while keeping a usable book (>30% kept)
    c26 = armC.get("2026", {})
    cC = (c26.get("unanimous", {}).get("mean_pnl") or 0) - (c26.get("non_unanimous", {}).get("mean_pnl") or 0) > 5 \
        and c26.get("uni_frac", 0) > 0.3
    # D: incremental OOT info both directions (ΔAUC>.01 & ΔRankIC>.01)
    def ok(x):
        return isinstance(x, dict) and x.get("delta_auc", -1) > 0.01 and x.get("delta_rankic", -1) > 0.01
    cD = all(ok(armD.get(k, {})) for k in ["pre→2026", "2026→pre"])
    # E: bootstrap-significant agreement effect
    cE = OUT["armE_bootstrap"]["p"] < 0.05 and OUT["armE_bootstrap"]["ci"][0] > 0
    conds = {"A_disagreement_worse_2026": bool(cA), "B_ensemble_beats_fold1_2026": bool(cB),
             "C_unanimity_quality_lift": bool(cC), "D_incremental_OOT_info": bool(cD),
             "E_survives_dayblock_bootstrap": bool(cE)}
    survives = (cD or cE) and any(conds.values())   # OOT/bootstrap legs are load-bearing
    OUT["verdict"] = dict(conditions=conds, survives=bool(survives))
    for k, v in conds.items():
        print(f"   {('PASS' if v else 'fail'):4s}  {k}")
    print("\n" + ("█ ENSEMBLE-DISAGREEMENT SURVIVES — a real entry-side epistemic selector."
                  if survives else
                  "▒ ENSEMBLE-DISAGREEMENT REJECTED — folds are redundant; disagreement is not a tradeable "
                  "selector OOT. KILL IT."))
    print("═" * 100)
    with open(OUT_JSON, "w") as f:
        json.dump(OUT, f, indent=2, default=str)
    print(f"→ {OUT_JSON}")


if __name__ == "__main__":
    main()
