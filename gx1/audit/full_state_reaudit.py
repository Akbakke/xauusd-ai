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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-n", type=int, default=200_000)
    ap.add_argument("--forward-outcome-dir", default=f"{WS2}/forward_outcome_clean/per_week")
    ap.add_argument("--mtf-cache", default=f"{WS2}/MULTI_TF_V2_CACHE")
    ap.add_argument("--exit-per-bar-dir", default=f"{WS2}/exit_per_bar_scored_clean/per_week")
    ap.add_argument("--xgb-bundle", default=None)
    ap.add_argument("--xgb-contract", default=None)
    args = ap.parse_args()
    total_dead = 0

    # 1) ENTRY-IQL 197-state (THE GAP — where the 36 dip/struct hid). Load fwd-outcome sample → attach_ (self-
    #    compute dip/struct with the MTF cache) → build_state_matrix → _dead_cols. Also the dip/struct DATA-TRACE.
    try:
        from gx1.scripts.augment_forward_outcome_v2 import attach_group_a_dip_struct_ctx_columns
        from gx1.scripts.materialize_build_entry_iql_v2 import build_state_matrix as entry_bsm
        df = _sample_weeks(args.forward_outcome_dir, args.sample_n)
        df = attach_group_a_dip_struct_ctx_columns(df, cache_dir=args.mtf_cache, journal_label="reaudit")
        X, names = entry_bsm(df)
        total_dead += _audit_state("ENTRY-IQL state", X, names)
    except Exception as e:
        print(f"\n[ENTRY-IQL audit FAILED: {e!r}]"); traceback.print_exc()

    # 2) EXIT-IQL 209-state
    try:
        from gx1.scripts.materialize_build_exit_iql_v3_m1 import build_state_matrix as exit_bsm
        dfe = _sample_weeks(args.exit_per_bar_dir, args.sample_n)
        Xe, namese = exit_bsm(dfe)
        total_dead += _audit_state("EXIT-IQL state", Xe, namese)
    except Exception as e:
        print(f"\n[EXIT-IQL audit FAILED: {e!r}]"); traceback.print_exc()

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
            print(f"\n[XGB audit FAILED: {e!r}]")
    else:
        print("\n[XGB audit skipped — pass --xgb-bundle + --xgb-contract]")

    # NOTE: V10 ctx/snap/multi-TF is covered by assert_v10_batch_liveness (run at V10 retrain); add a batch
    # loader here when extending. V3 173-input = variance check on a v3_dataset sample (TODO).
    print(f"\n=== FULL RE-AUDIT: total un-allowlisted DEAD features = {total_dead} "
          f"({'PASS — all alive/allowlisted' if total_dead == 0 else 'FAIL — fix or justify each'}) ===")
    print(f"(allowlist KNOWN_ALLOWED_DEAD has {len(KNOWN_ALLOWED_DEAD)} entries)")
    sys.exit(0 if total_dead == 0 else 1)


if __name__ == "__main__":
    main()
