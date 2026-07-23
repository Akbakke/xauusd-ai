"""Feature-liveness re-audit of the active Exit-IQL state vector.

The model-native Entry has its own exact 513-signal liveness contract. This
driver separately audits the retained Exit-IQL state plus optional XGB gain.

This driver loads a SHUFFLED sample of each state vector's ACTUAL built values and runs the same tested
`_dead_cols` checker, classifying every Exit feature ALIVE / CONST /
ALLOWLISTED. It exits nonzero on any unallowlisted dead feature.

⚠ RAM: this loads data. Use a MODEST --sample-n (default 200k, NOT the full 5M) and run only when no heavy build
holds RAM (AGENTS.md SMART+MAXED RAM-headroom rule — an OOM crashed the PC 2026-06-10). Check `free -g` first.

Usage:
  python -m gx1.audit.full_state_reaudit --sample-n 200000 \
    --exit-per-bar-dir <…/exit_per_bar_scored_clean> \
    [--xgb-bundle <bundle-owned-contracts>]
"""
import argparse
import os
import re
import sys
import traceback

import numpy as np
import pandas as pd

from gx1.audit.feature_liveness import audit_iql_state_liveness, KNOWN_ALLOWED_DEAD

def _exit_wave() -> str:
    """Resolve the retained Exit-IQL wave from the artifact contract."""
    from gx1_guards.artifacts import load_decision_artifact

    return os.path.dirname(str(load_decision_artifact("exit_iql")))


def _audit_state(name, X, names):
    """Run the one-truth audit_iql_state_liveness on a built state matrix; return n_dead."""
    rep = audit_iql_state_liveness(X, names, role=name, raise_on_fail=False)  # warn-loud, don't raise
    dead, n_zero, n_feat, n_rows = rep["dead"], rep["n_zero"], rep["n_features"], rep["n_rows"]
    print(f"\n===== {name}: {n_feat} features, {n_rows} rows =====")
    print(f"  ALIVE={n_feat-len(dead)}  DEAD(un-allowlisted)={len(dead)}  all-zero-cols={n_zero}")
    if dead:
        print("  ⚠ DEAD / un-allowlisted (FAIL):")
        for d in dead[:60]:
            print(f"      {d}")
        if len(dead) > 60:
            print(f"      … +{len(dead)-60} more")
    return len(dead)


# ── --detail mode (2026-06-11): per-TF EMA / M5-M15 market-state / MAE liveness on the SAME frames
# the audit already loads (no second data-load). Answers "is EMA operative on ALL TFs / are M5-M15 state
# + MAE features actually alive". Piggybacks arms 1 (entry forward_outcome) + 2 (exit per-bar). ──
_TFS_ORDER = ["m5", "m15", "h1", "h4", "d1"]
_DETAIL_DEAD_STD = 1e-8


def _col_stat(s):
    # 2026-06-11: dtype-aware — string/categorical columns (trend_regime/vol_regime) used to coerce
    # to all-NaN and print "<<< DEAD/CONST", a FALSE positive (the same print-class that produced the
    # earlier false "entry regime-blind" memory). Categoricals are judged by nunique; their liveness
    # is properly audited via the one-hots inside the model state vectors.
    if not pd.api.types.is_numeric_dtype(s):
        vals = s.dropna().astype(str)
        nu = int(vals.nunique())
        nz = float(len(vals)) / max(len(s), 1)
        return 0.0, nz, nu, (nu <= 1)
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 0.0, 0, True
    std = float(np.std(a))
    nz = float(np.mean(np.abs(a) > 1e-12))
    nu = int(np.unique(np.round(a, 8)).size)
    return std, nz, nu, (std < _DETAIL_DEAD_STD)


def _tf_of(col):
    m = re.match(r"^(m5|m15|h1|h4|d1)_", col.lower())
    return m.group(1) if m else None


def _detail_block(df, title, pats):
    print(f"\n----- DETAIL: {title} -----")
    any_dead = []
    for label, pat in pats:
        rx = re.compile(pat, re.I)
        cols = sorted([c for c in df.columns if rx.search(c)])
        if not cols:
            print(f"  [{label}] (no matching cols)")
            continue
        by_tf = {}
        for c in cols:
            by_tf.setdefault(_tf_of(c) or "(scalar)", []).append(c)
        ntf = len([k for k in by_tf if k != "(scalar)"])
        print(f"  [{label}] {len(cols)} cols, {ntf} TFs:")
        for tf in _TFS_ORDER + ["(scalar)"]:
            for c in by_tf.get(tf, []):
                std, nz, nu, dead = _col_stat(df[c])
                flag = "  <<< DEAD/CONST" if dead else ("  <all-zero>" if nz < 1e-3 else "")
                if dead:
                    any_dead.append(c)
                print(f"      {c:46s} std={std:.3e} nz={nz*100:5.1f}% nuniq={nu:>6}{flag}")
    if any_dead:
        print(f"  ⚠ DEAD/CONST in '{title}': {any_dead}")
    return any_dead


_EXIT_DETAIL_PATS = [
    ("MAE", r"\bmae\b|_mae|mae_bps|mae_bef"), ("MFE", r"\bmfe\b|_mfe|mfe_bps"),
    ("dd_from_mfe", r"dd_from_mfe|drawdown_from"), ("giveback", r"giveback"),
    ("m5_phase/hold", r"m5_phase|bars_in|bars_since|hold"),
    ("exit-MTF EMA", r"ema\d+_dist_atr|ema\d+_slope_atr|ema_stack"),
]


def main():
    import os
    from pathlib import Path
    # Match the A3 exit build's env (aug64 + regime) so the EXIT lazy-join computes the same 64 aug64 features.
    for k in ("GX1_EXIT_AUGMENT_64", "GX1_REGIME_V4", "GX1_TREND_REGIME_FROM_D1"):
        os.environ.setdefault(k, "1")
    _exit_wave_dir = _exit_wave()
    print(f"[reaudit] exit-wave={_exit_wave_dir}")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-n", type=int, default=200_000)
    ap.add_argument("--exit-per-bar-dir", default=f"{_exit_wave_dir}/exit_per_bar_scored_clean")
    ap.add_argument("--canonical-features", default=f"{_exit_wave_dir}/CANONICAL_FEATURES_V3_PLUS5/canonical_features_v3_plus5.parquet")
    ap.add_argument("--reports-root", default="/home/andre2/GX1_DATA/reports/truth_e2e_sanity")  # chunk0 (empty) — build default
    ap.add_argument("--xgb-bundle", default=None)
    ap.add_argument("--detail", action="store_true",
                    help="also print per-TF EMA / M5-M15 market-state / MAE liveness detail on the loaded frames")
    args = ap.parse_args()
    total_dead = 0
    failures = []   # a THROWN arm is a FAIL, never a silent pass (the 2026-06-10 false-pass bug)

    # Exit-IQL state. Use the build's one-truth lazy-join loader (canonical-suffix + 64 aug64 + chunk0) →
    #    build_state_matrix — same path as the A3 build (materialize_build_exit_iql_v3_m1:470). Raw scored parquets
    #    LACK the _v1_*_canon_v1 + aug64 cols (they are lazy-joined), so a plain read would false-miss 86 cols.
    try:
        from gx1.scripts.materialize_build_exit_iql_v3_m1 import (
            build_state_matrix as exit_bsm, load_per_bar_dataset_lazy_join)
        pbd = Path(args.exit_per_bar_dir)
        per_week = pbd / "per_week" if (pbd / "per_week").exists() else pbd
        dfe = load_per_bar_dataset_lazy_join(per_week, reports_root=Path(args.reports_root),
                                             canonical_path=Path(args.canonical_features), sample_n_rows=args.sample_n)
        Xe, namese = exit_bsm(dfe)
        total_dead += _audit_state("EXIT-IQL state", Xe, namese)
        if args.detail:
            _detail_block(dfe, "EXIT trade-state MAE/MFE + exit-MTF EMA", _EXIT_DETAIL_PATS)
    except Exception as e:
        print(f"\n[EXIT-IQL audit FAILED: {e!r}]")
        traceback.print_exc()
        failures.append("EXIT-IQL")

    # Optional XGB gain check.
    if args.xgb_bundle:
        try:
            from gx1.audit.feature_liveness import audit_xgb_gain
            xdead = audit_xgb_gain(args.xgb_bundle)
            print(f"\n===== XGB: {len(xdead)} features 0-gain in ALL heads (un-allowlisted) =====")
            for d in xdead[:60]:
                print(f"      {d}")
            total_dead += len(xdead)
        except Exception as e:
            print(f"\n[XGB audit FAILED: {e!r}]")
            failures.append("XGB")
    else:
        print("\n[XGB audit skipped — pass --xgb-bundle]")

    # Verdict: PASS only if every attempted arm RAN and found no un-allowlisted dead. A thrown arm = FAIL
    # (so a broken audit can NEVER read as PASS — the false-pass bug found 2026-06-10).
    ok = (total_dead == 0) and not failures
    print(f"\n=== FULL RE-AUDIT: un-allowlisted DEAD={total_dead}  failed_arms={failures or 'none'} "
          f"=> {'PASS — all audited arms alive/allowlisted' if ok else 'FAIL — see dead features / failed arms above'} ===")
    print(f"(allowlist KNOWN_ALLOWED_DEAD has {len(KNOWN_ALLOWED_DEAD)} entries)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
