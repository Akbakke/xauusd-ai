"""POST-HOC OOT evaluator — measure P1 (skip/down-weight ASIA) edge + P3 (Strategy-F / runner-clip)
on a gated per-candidate CSV, with ZERO retrain and ZERO serve change.

This EXTENDS the existing post-hoc grader honest_phase6_gate.py (corrupt-April exclusion, skip-ASIA
subset, HARDENED 4-check verdict) with the two things it lacked:
  (1) the validator's PER-YEAR floor (GATE_CFG 12.0 bps / 0.85 win) so a chain that collapses in one
      regime can't pass on a pooled mean — imported as ONE-TRUTH from
      gx1.scripts.v12.v12_phase6_joint_validation (_per_year_metrics / _entry_year / GATE_CFG);
  (2) a HOLD-BUCKET breakdown on exit_bar — the P3 tool: if learned giveback-protection
      (R_PEAK_QUALITY) or the hardcoded Strategy-F overlay CLIPS the 60-240-bar runners on the
      near-oracle R_NET_REAL policy, it shows up here as a per-bucket mean_bps regression.

PHILOSOPHY: this is a MEASUREMENT ruler, NOT a deployment mechanism. Edges measured here are then
BAKED INTO the policy via reward shaping (R_PEAK_QUALITY exit reward / session-conditional entry
reward) — the smart-AI route — rather than shipped as hardcoded serve wrappers. Session SSoT =
gx1.time.session_detector (ASIA = 22:00-07:00 UTC). Reads CSVs only; touches no model/contract.

Usage:
  python -m gx1.research.posthoc_session_strategyf_eval <candidate_per_candidate.csv> [--baseline <cement.csv>]
Exit 0 iff the skip-ASIA arm PASSES the hardened verdict vs baseline AND the per-year floor; else 1.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/andre2/src/GX1_ENGINE")
from gx1.time.session_detector import get_session_vectorized  # SSoT session classification
# ONE-TRUTH per-year floor (do not re-implement the 12.0/0.85 thresholds — import them):
from gx1.scripts.v12.v12_phase6_joint_validation import (  # noqa: E402
    GATE_CFG, _entry_year, _per_year_metrics,
)

# Same corrupt-April window the honest gate excludes (x10 data-corruption tripwire).
CORRUPT_APRIL = (pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-04-21", tz="UTC"))
# Default cement baseline (matches honest_phase6_gate.py:52). 33.61 bps / 0.971 skip-ASIA reference.
CEMENT_BASELINE = ("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/"
                   "PHASE6_OOT_COSTFIX_HEADS_20260529T043258Z_LOCK/per_candidate_V12_OFF.csv")
# The LIVE fase2b cement = the ACTUAL rollback the clean wave would replace (its per-candidate CSV).
FASE2B_BASELINE = ("/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
                   "phase6_lam50/per_candidate_V12_OFF.csv")
# Hold-bucket edges on exit_bar (M1 bars). The 60-240 buckets are the runner band the overlay clips.
HOLD_BUCKETS = [(0, 30), (30, 60), (60, 120), (120, 240), (240, np.inf)]


def load(csv_path, keep_april=False):
    """Read a per-candidate CSV; decode entry_ts/session/year from candidate_uid. By DEFAULT drops the
    corrupt-April window (cement/pre-repair chains had x10 fakes there). keep_april=True KEEPS it — use
    for a REPAIRED clean chain (April is valid OOT post-repair, handover §9); the impossible(>1000bps)
    count in the kept window then doubles as a REPAIR CHECK (a good x10 repair => 0 impossible there)."""
    pc = pd.read_csv(csv_path)
    ts = pc["candidate_uid"].str.extract(r"(\d{8}T\d{6})$")[0]
    pc["entry_ts"] = pd.to_datetime(ts, format="%Y%m%dT%H%M%S", utc=True)
    pc["session"] = get_session_vectorized(pc["entry_ts"]).values
    pc["win"] = (pc["realized_pnl_bps"] > 0).astype(float)
    apr = (pc["entry_ts"] >= CORRUPT_APRIL[0]) & (pc["entry_ts"] < CORRUPT_APRIL[1])
    clean = pc.copy() if keep_april else pc[~apr].copy()
    clean.attrs["raw_n"] = len(pc)
    clean.attrs["april_kept"] = keep_april
    clean.attrs["april_removed"] = 0 if keep_april else int(apr.sum())
    clean.attrs["april_bps_removed"] = 0.0 if keep_april else float(pc.loc[apr, "realized_pnl_bps"].sum())
    clean.attrs["impossible"] = int((clean["realized_pnl_bps"].abs() > 1000).sum())
    return clean


def agg(df):
    if len(df) == 0:
        return dict(n=0, win=float("nan"), mean_bps=float("nan"), median_bps=float("nan"),
                    total_bps=0.0, max_bps=float("nan"))
    return dict(n=len(df), win=round(df["win"].mean(), 4),
                mean_bps=round(df["realized_pnl_bps"].mean(), 2),
                median_bps=round(df["realized_pnl_bps"].median(), 2),
                total_bps=round(df["realized_pnl_bps"].sum(), 0),
                max_bps=round(df["realized_pnl_bps"].max(), 0))


def per_year_floor(df):
    """Worst-year mean/win vs GATE_CFG floor (12.0 bps / 0.85), reusing the validator's ONE-truth fn."""
    py = _per_year_metrics(df)  # {year: {n, mean, median, win}}
    if not py:
        return dict(ok=False, worst_mean=float("nan"), worst_win=float("nan"), by_year={})
    worst_mean = min(v["mean"] for v in py.values())
    worst_win = min(v["win"] for v in py.values())
    return dict(
        ok=(worst_mean >= GATE_CFG["min_year_mean_bps"] and worst_win >= GATE_CFG["min_year_win"]),
        worst_mean=round(worst_mean, 2), worst_win=round(worst_win, 4),
        by_year={int(y): dict(n=v["n"], mean=round(v["mean"], 1), win=round(v["win"], 3))
                 for y, v in sorted(py.items())},
    )


def hold_buckets(df):
    """Per-hold-bucket n/win/mean_bps on exit_bar — surfaces runner-clipping in the 60-240 band."""
    if "exit_bar" not in df.columns:
        return None
    eb = pd.to_numeric(df["exit_bar"], errors="coerce")
    out = {}
    for lo, hi in HOLD_BUCKETS:
        sub = df[(eb >= lo) & (eb < hi)]
        out[f"{lo}-{'inf' if hi == np.inf else int(hi)}"] = agg(sub)
    return out


def summarize(df, label):
    """All P1/P3 measurement slices for one chain."""
    sessions = ["ASIA", "EU", "OVERLAP", "US"]
    return dict(
        label=label,
        raw_n=df.attrs.get("raw_n"), april_removed=df.attrs.get("april_removed"),
        april_bps_removed=df.attrs.get("april_bps_removed"), impossible=df.attrs.get("impossible"),
        blanket=agg(df),
        skip_asia=agg(df[df["session"] != "ASIA"]),
        per_session={s: agg(df[df["session"] == s]) for s in sessions},
        per_year_blanket=per_year_floor(df),
        per_year_skipasia=per_year_floor(df[df["session"] != "ASIA"]),
        buckets_blanket=hold_buckets(df),
        buckets_skipasia=hold_buckets(df[df["session"] != "ASIA"]),
    )


def _row(a):
    return (f"n={a['n']:6d}  win={a['win']:.3f}  mean_bps={a['mean_bps']:8.2f}  "
            f"median={a['median_bps']:7.2f}  total={a['total_bps']:.0f}")


def show(r):
    print(f"\n===== {r['label']} =====")
    print(f"  raw takes={r['raw_n']}  corrupt-April removed={r['april_removed']} "
          f"({r['april_bps_removed']:.0f} bps)  ⚠impossible(>1000bps)={r['impossible']}")
    print(f"  {'BLANKET':14s}: {_row(r['blanket'])}")
    print(f"  {'skip-ASIA (P1)':14s}: {_row(r['skip_asia'])}")
    print("  per-session:")
    for s, a in r["per_session"].items():
        print(f"    {s:8s}: {_row(a)}")
    for tag, key in (("blanket", "per_year_blanket"), ("skip-ASIA", "per_year_skipasia")):
        py = r[key]
        flag = "PASS" if py["ok"] else "FAIL"
        print(f"  per-year floor [{tag}] [{flag}] worst_mean={py['worst_mean']} (>= {GATE_CFG['min_year_mean_bps']})"
              f"  worst_win={py['worst_win']} (>= {GATE_CFG['min_year_win']})  by_year={py['by_year']}")
    if r["buckets_skipasia"]:
        print("  hold-buckets [skip-ASIA] (P3 runner-clip check):")
        for b, a in r["buckets_skipasia"].items():
            print(f"    bars {b:8s}: {_row(a)}")


def verdict(new, base):
    """HARDENED skip-ASIA verdict vs baseline (mirrors honest_phase6_gate.py:66-78) + per-year floor."""
    nb, bb = new["skip_asia"], base["skip_asia"]
    db = nb["mean_bps"] - bb["mean_bps"]
    dw = nb["win"] - bb["win"]
    cov = nb["n"] / max(1, bb["n"])
    py = new["per_year_skipasia"]
    checks = {
        "no_impossible_trade(>1000bps)": new["impossible"] == 0,
        "coverage>=80% of baseline N": cov >= 0.80,
        "win-rate non-regression (Δ>=-1pp)": dw >= -0.01,
        "expectancy non-regression (Δ>=0)": db >= 0.0,
        f"per-year mean>={GATE_CFG['min_year_mean_bps']} & win>={GATE_CFG['min_year_win']}": py["ok"],
    }
    print("\n=== VERDICT (skip-ASIA honest, HARDENED + per-year) ===")
    print(f"  Δ mean_bps={db:+.2f}  Δ win={dw:+.4f}  coverage={cov:.2f}  "
          f"impossible={new['impossible']}  worst_year_mean={py['worst_mean']}  worst_year_win={py['worst_win']}")
    for k, ok in checks.items():
        print(f"    [{'PASS' if ok else 'FAIL'}] {k}")
    ok_all = all(checks.values())
    print(f"  => {'PASS — new ≥ baseline on ALL checks' if ok_all else 'FAIL — see failed checks'}")
    return ok_all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("candidate_csv", help="gated chain per_candidate_V12_OFF.csv to grade")
    ap.add_argument("--baseline", default=CEMENT_BASELINE,
                    help="COSTFIX-era baseline per_candidate CSV (informational)")
    ap.add_argument("--fase2b-baseline", default=FASE2B_BASELINE,
                    help="LIVE fase2b cement baseline = the ACTUAL rollback (decisive). '' to skip.")
    ap.add_argument("--no-repaired-april-view", action="store_true",
                    help="suppress the repaired-April-INCLUSIVE candidate view (shown by default)")
    args = ap.parse_args()

    # Candidate, April-EXCLUDED = apples-to-apples basis vs the (still-corrupt-April) cement baselines.
    new = summarize(load(args.candidate_csv, keep_april=False), "CANDIDATE chain (April-EXCLUDED)")
    show(new)
    # Candidate INCLUDING the repaired April = the wave's new OOT capability + a repair check (handover §9
    # says April is valid post-repair; the launcher claims 'no April-skip'). Verdict stays April-excluded
    # because the cement baselines' April is still the x10-corrupt data.
    if not args.no_repaired_april_view:
        new_apr = summarize(load(args.candidate_csv, keep_april=True),
                            "CANDIDATE incl. REPAIRED April (wave OOT; impossible=0 => x10 repair OK)")
        show(new_apr)
        if new_apr["impossible"] > 0:
            print(f"  ⚠⚠ {new_apr['impossible']} impossible(>1000bps) trades in the KEPT window — the x10 "
                  f"April repair MAY HAVE FAILED; do NOT trust the April-inclusive view until checked.")

    decisive, last_v = None, None
    for tag, path in (("COSTFIX cement (informational)", args.baseline),
                      ("LIVE fase2b cement (THE ROLLBACK — decisive)", args.fase2b_baseline)):
        if not path:
            continue
        try:
            base = summarize(load(path, keep_april=False), f"BASELINE — {tag}")
        except Exception as e:
            print(f"\n[skip baseline {tag}: {e!r}]")
            continue
        print(f"\n########## CANDIDATE (April-excl) vs {tag} ##########")
        show(base)
        last_v = verdict(new, base)
        if "fase2b" in tag:
            decisive = last_v
    if decisive is None:  # fase2b baseline missing → fall back to the last baseline verdict we computed
        decisive = bool(last_v) if last_v is not None else False
    print(f"\n=== DECISIVE GATE (clean chain vs LIVE fase2b rollback, April-excluded): "
          f"{'PASS — flip candidate' if decisive else 'FAIL — keep cement'} ===")
    print("Honest cement reference: blanket ~28.4 bps/94% win; skip-ASIA ~33.6 bps/97% win.")
    sys.exit(0 if decisive else 1)


if __name__ == "__main__":
    main()
