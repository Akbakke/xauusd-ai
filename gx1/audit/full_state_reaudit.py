"""FULL feature-liveness re-audit of EVERY feature in EVERY model's state vector (user vedtak 2026-06-10).

Closes the rule-9 COVERAGE GAP: gx1.audit.feature_liveness only auto-checks XGB-gain + V10 ctx/snap/multi-TF.
There is NO liveness check on the Entry-IQL (197) or Exit-IQL (209) STATE vectors — that is how 36 self-computed
dip/struct features stayed CONST-ZERO, unallowlisted, for months ([[project_gx1_dipstruct_silent_zero...]]).

This driver loads a SHUFFLED sample of each state vector's ACTUAL built values and runs the same tested
`_dead_cols` checker, classifying EVERY feature ALIVE / CONST / ALLOWLISTED. Exit-nonzero on any un-allowlisted
dead feature. It ALSO doubles as the dip/struct DATA-TRACE: pass --mtf-cache and see whether the dip/struct come
ALIVE (→ the bug was the cache not being delivered) or stay zero (→ the cache content itself is zero).

⚠ RAM: this loads data. Use a MODEST --sample-n (default 200k, NOT the full 5M) and run only when no heavy build
holds RAM (AGENTS.md SMART+MAXED RAM-headroom rule — an OOM crashed the PC 2026-06-10). Check `free -g` first.

Usage:
  python -m gx1.audit.full_state_reaudit --sample-n 200000 \
    --forward-outcome-dir <…/forward_outcome_clean/per_week> --mtf-cache <…/MULTI_TF_V2_CACHE> \
    --exit-per-bar-dir <…/exit_per_bar_scored_clean> [--xgb-bundle <…> --xgb-contract <…>]
"""
import argparse
import glob
import re
import sys
import traceback

import numpy as np
import pandas as pd

from gx1.audit.feature_liveness import audit_iql_state_liveness, KNOWN_ALLOWED_DEAD

WS2 = "/home/andre2/GX1_DATA/runs/FASE2B_CLEAN_20260608"


def _sample_weeks(per_week_dir, sample_n, seed=1337):
    """Bounded-RAM shuffled sample of ~sample_n rows from a per-week parquet dir.

    Subsamples EACH parquet to ~sample_n/n_files rows BEFORE concat (the proven pattern from
    load_per_bar_dataset, materialize_build_exit_iql_v2) so the full dataset is never materialized —
    full temporal coverage, bounded memory. Glob-agnostic, so it serves both the forward_outcome and
    exit_per_bar dirs. Final shuffle-cap to exactly sample_n (rule-9: shuffled, not a consecutive batch).
    """
    files = sorted(glob.glob(f"{per_week_dir}/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquets under {per_week_dir}")
    rng = np.random.default_rng(seed)
    per_file = max(1, sample_n // len(files) + 1)
    parts = []
    for f in files:
        d = pd.read_parquet(f)
        if len(d) > per_file:
            d = d.iloc[rng.choice(len(d), size=per_file, replace=False)]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=seed)
    return df.reset_index(drop=True)


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
    a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 0.0, 0, True
    std = float(np.std(a)); nz = float(np.mean(np.abs(a) > 1e-12)); nu = int(np.unique(np.round(a, 8)).size)
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
            print(f"  [{label}] (no matching cols)"); continue
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


def _ema_distinct(df):
    print("\n----- DETAIL: EMA cross-TF DISTINCTNESS (corr>0.98 = TFs NOT distinct = a bug) -----")
    for base in ["ema50_dist_atr", "ema20_dist_atr", "ema20_50_diff_atr"]:
        ser = {}
        for tf in _TFS_ORDER:
            cand = [c for c in df.columns if c.lower().startswith(tf + "_") and base in c.lower()]
            if cand:
                ser[tf] = pd.to_numeric(df[cand[0]], errors="coerce").to_numpy(np.float64)
        if not ser:
            print(f"  {base}: NOT PRESENT in any TF (not wired)"); continue
        if len(ser) < 2:
            print(f"  {base}: only in {list(ser)}"); continue
        tfs = list(ser); print(f"  {base}: present in {tfs}")
        for i in range(len(tfs)):
            for j in range(i + 1, len(tfs)):
                a, b = ser[tfs[i]], ser[tfs[j]]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() > 100 and np.std(a[m]) > 0 and np.std(b[m]) > 0:
                    r = float(np.corrcoef(a[m], b[m])[0, 1])
                    print(f"      {tfs[i]}~{tfs[j]} corr={r:+.3f}{'  <<< NOT DISTINCT' if abs(r) > 0.98 else ''}")


_ENTRY_DETAIL_PATS = [
    ("ema_dist", r"ema\d+_dist_atr"), ("ema_slope", r"ema\d+_slope_atr"),
    ("ema_stack", r"ema_stack_aligned"), ("ema20x50_cross(V3)", r"ema20_50_diff_atr"),
    ("dip", r"dip"), ("struct", r"struct"), ("regime", r"regime"),
    ("momentum", r"_mom_?\d|momentum"), ("dist_to_hilo", r"dist_to_.*(lo|hi)"),
    ("MAE-before-MFE", r"mae_before_mfe|mae_bef_mfe"), ("mfe", r"mfe"),
]
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-n", type=int, default=200_000)
    ap.add_argument("--forward-outcome-dir", default=f"{WS2}/forward_outcome_clean/per_week")
    ap.add_argument("--mtf-cache", default=f"{WS2}/MULTI_TF_V2_CACHE")
    ap.add_argument("--exit-per-bar-dir", default=f"{WS2}/exit_per_bar_scored_clean")
    ap.add_argument("--canonical-features", default=f"{WS2}/CANONICAL_FEATURES_V3_PLUS5/canonical_features_v3_plus5.parquet")
    ap.add_argument("--reports-root", default="/home/andre2/GX1_DATA/reports/truth_e2e_sanity")  # chunk0 (empty) — build default
    ap.add_argument("--xgb-bundle", default=None)
    ap.add_argument("--xgb-contract", default=None)
    ap.add_argument("--detail", action="store_true",
                    help="also print per-TF EMA / M5-M15 market-state / MAE liveness detail on the loaded frames")
    args = ap.parse_args()
    total_dead = 0
    failures = []   # a THROWN arm is a FAIL, never a silent pass (the 2026-06-10 false-pass bug)

    # 1) ENTRY-IQL 197-state. Replicate the build (materialize_build_entry_iql_v2:873-876): map decision_ts_utc->time,
    #    attach_ (dip/struct from the MTF cache), build_state_matrix → _dead_cols. Also the dip/struct DATA-TRACE.
    try:
        from gx1.scripts.augment_forward_outcome_v2 import attach_group_a_dip_struct_ctx_columns
        from gx1.scripts.materialize_build_entry_iql_v2 import build_state_matrix as entry_bsm
        df = _sample_weeks(args.forward_outcome_dir, args.sample_n)
        if "time" not in df.columns and "decision_ts_utc" in df.columns:
            df["time"] = pd.to_datetime(df["decision_ts_utc"], utc=True)   # build does this before attach_
        df = attach_group_a_dip_struct_ctx_columns(df, cache_dir=args.mtf_cache, journal_label="reaudit")
        X, names = entry_bsm(df)
        total_dead += _audit_state("ENTRY-IQL state", X, names)
        if args.detail:
            _detail_block(df, "ENTRY per-TF EMA + M5/M15 market-state + MAE", _ENTRY_DETAIL_PATS)
            _ema_distinct(df)
    except Exception as e:
        print(f"\n[ENTRY-IQL audit FAILED: {e!r}]"); traceback.print_exc(); failures.append("ENTRY-IQL")

    # 2) EXIT-IQL 209-state. Use the build's ONE-TRUTH lazy-join loader (canonical-suffix + 64 aug64 + chunk0) →
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
        print(f"\n[EXIT-IQL audit FAILED: {e!r}]"); traceback.print_exc(); failures.append("EXIT-IQL")

    # 3) XGB gain (existing one-truth check)
    if args.xgb_bundle and args.xgb_contract:
        try:
            from gx1.audit.feature_liveness import audit_xgb_gain
            xdead = audit_xgb_gain(args.xgb_bundle, args.xgb_contract)
            print(f"\n===== XGB: {len(xdead)} features 0-gain in ALL heads (un-allowlisted) =====")
            for d in xdead[:60]:
                print(f"      {d}")
            total_dead += len(xdead)
        except Exception as e:
            print(f"\n[XGB audit FAILED: {e!r}]"); failures.append("XGB")
    else:
        print("\n[XGB audit skipped — pass --xgb-bundle + --xgb-contract]")

    # Verdict: PASS only if every attempted arm RAN and found no un-allowlisted dead. A thrown arm = FAIL
    # (so a broken audit can NEVER read as PASS — the false-pass bug found 2026-06-10).
    ok = (total_dead == 0) and not failures
    print(f"\n=== FULL RE-AUDIT: un-allowlisted DEAD={total_dead}  failed_arms={failures or 'none'} "
          f"=> {'PASS — all audited arms alive/allowlisted' if ok else 'FAIL — see dead features / failed arms above'} ===")
    print(f"(allowlist KNOWN_ALLOWED_DEAD has {len(KNOWN_ALLOWED_DEAD)} entries)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
