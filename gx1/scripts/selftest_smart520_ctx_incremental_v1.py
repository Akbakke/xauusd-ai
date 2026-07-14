#!/usr/bin/env python3
"""Self-test for the ASYNC smart-context serving path (serving-wave gap 3).

Proves, on the REAL live prebuilts, the four claims the async design rests on
(gx1/execution/v12_smart_entry_live.py + v12_smart520_state_live.py):

  A. OVERRIDE CAUSALITY (hard): recomputing the frame-override tables
     (compute_bucket_ctx_cat_full_frame + compute_htf_ctx_full_frame) on an
     EXTENDED cv3 leaves every pre-existing row bit-identical — so the live
     age>0 path's cheap fresh recompute is exact by construction.
  B. MTF SPLICE BIT-IDENTITY (hard, per TF in SMART520_MTF_SPLICE_TFS):
     append_multi_tf_incremental (warmup-tail rebuild + splice) reproduces the
     full build_multi_tf_from_cv3 float32 arrays EXACTLY for M5/M15/H1 at
     context ages 1..3 and across M15/H1 period boundaries. H4/D1 are never
     spliced (EMA-200 cannot converge on any sane tail) — their forming-bar
     residual vs full rebuild is MEASURED and reported verbatim.
  C. DECISION-STATE EQUIVALENCE (measured, reported verbatim): a decision on
     the newest bar using an age-1..3 snapshot (+ incremental extension) vs the
     blocking full-refresh reference — quantifies the ONLY residual staleness
     (H4/D1 forming rows consumed by the group-A per-TF features).
  D. THREAD SMOKE (hard): predict_live_bar during an in-flight background
     refresh — no exception, deterministic results off the grabbed snapshot,
     atomic swap visible afterwards, post-swap decision == blocking reference.

Extend-don't-fork note (CLAUDE.md rule 7): verify_smart520_serve_parity_v1 was
considered — it is the PINNED live-vs-OFFLINE train==serve launch gate with a
fixed artifact schema; this self-test proves INTERNAL incremental==full
equivalence + thread behavior and must be re-runnable without touching the
gate's artifact. Report: SMART520_CTX_ASYNC_SELFTEST_<ts>.json (+ _latest).

Run capped (full prebuilt load):
  scripts/gx1_capped_run.sh --mem 34G -- .venv/bin/python -m \
      gx1.scripts.selftest_smart520_ctx_incremental_v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("GX1_REGIME_V4", "1")
os.environ.setdefault("GX1_TREND_REGIME_FROM_D1", "1")

import numpy as np
import pandas as pd

DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/smart520_ctx_async_selftest")


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max()) \
        if len(a) and len(b) else 0.0


def _build_overrides(cv3: pd.DataFrame) -> pd.DataFrame:
    from gx1.execution.v12_smart520_state_live import (
        Smart520StateContract,
        compute_bucket_ctx_cat_full_frame,
        compute_htf_ctx_full_frame,
    )
    state_contract = Smart520StateContract.legacy()
    return pd.concat(
        [
            compute_bucket_ctx_cat_full_frame(cv3, state_contract),
            compute_htf_ctx_full_frame(cv3, state_contract),
        ],
        axis=1,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ages", default="1,2,3")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--skip-decision", action="store_true",
                    help="skip tests C/D (no adapter/model load) — A/B only, faster")
    args = ap.parse_args()
    ages = [int(x) for x in args.ages.split(",") if x.strip()]

    t0 = time.time()
    failures: list[str] = []
    report: dict = {
        "schema_version": "smart520_ctx_async_selftest_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ages": ages,
    }
    from gx1.execution.v12_smart520_state_live import (
        SMART520_MTF_SPLICE_TFS,
        SMART520_MTF_SPLICE_WARMUP_M5,
        append_multi_tf_incremental,
        build_multi_tf_from_cv3,
    )
    report["splice_tfs"] = list(SMART520_MTF_SPLICE_TFS)
    report["splice_warmup_m5"] = SMART520_MTF_SPLICE_WARMUP_M5

    from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader
    loader = PrebuiltStateLoader()
    loader.load()
    loader._refresh_enabled = False
    cv3 = loader._cv3
    report["cv3"] = {"rows": int(len(cv3)), "cutoff": str(cv3.index[-1])}
    print(f"[selftest] prebuilts loaded: {len(cv3)} rows, cutoff={cv3.index[-1]} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ── TEST A: override causality (hard) ────────────────────────────────────
    max_age = max(ages)
    ovr_T = _build_overrides(cv3)
    ovr_C = _build_overrides(cv3.iloc[:-max_age])
    common = ovr_C.index
    a_diffs: dict[str, float] = {}
    for col in ovr_C.columns:
        a = ovr_C[col].to_numpy()
        b = ovr_T.loc[common, col].to_numpy()
        d = _max_abs(a.astype(np.float64), b.astype(np.float64))
        a_diffs[col] = d
        if d > 0.0:
            failures.append(f"TEST A: override col '{col}' changed on pre-existing rows "
                            f"(max_abs_diff={d:.3e}) — causality claim BROKEN")
    report["test_a_override_causality"] = {
        "compared_rows": int(len(common)), "extended_by_bars": max_age,
        "max_abs_diff_per_col": a_diffs,
        "pass": all(v == 0.0 for v in a_diffs.values()),
    }
    print(f"[selftest] TEST A done: max col diff = "
          f"{max(a_diffs.values()):.3e} ({time.time()-t0:.0f}s)", flush=True)

    # ── TEST B: MTF splice bit-identity (hard for splice TFs) ───────────────
    # Base 1: the live cutoff. Base 2: the newest ON-THE-HOUR bar (guarantees the
    # spliced gap crosses an H1+M15 period boundary at age 1).
    b_results = []
    minutes = cv3.index.minute
    on_hour_pos = int(np.flatnonzero(minutes == 0)[-1])
    bases = [("cutoff", len(cv3)), ("h1_boundary", on_hour_pos + 1)]
    for base_name, pos_T in bases:
        cv3_T = cv3.iloc[:pos_T]
        full_T = build_multi_tf_from_cv3(cv3_T)
        base_ages = ages if base_name == "cutoff" else [1]
        for age in base_ages:
            snap = build_multi_tf_from_cv3(cv3.iloc[:pos_T - age])
            spliced, flag = append_multi_tf_incremental(cv3_T, snap)
            rec = {"base": base_name, "end_ts": str(cv3_T.index[-1]), "age": age,
                   "spliced": bool(flag), "per_tf": {}}
            for tf, full_df in full_T.items():
                s_df = spliced[tf]
                ts_eq = np.array_equal(np.asarray(s_df.attrs["ts_int64"]),
                                       np.asarray(full_df.attrs["ts_int64"]))
                if tf in SMART520_MTF_SPLICE_TFS:
                    feats_eq = ts_eq and np.array_equal(np.asarray(s_df.attrs["feats_np"]),
                                                        np.asarray(full_df.attrs["feats_np"]))
                    md = (0.0 if feats_eq else
                          _max_abs(s_df.attrs["feats_np"], full_df.attrs["feats_np"])
                          if ts_eq else float("nan"))
                    rec["per_tf"][tf] = {"bit_identical": bool(feats_eq), "max_abs_diff": md}
                    if not feats_eq:
                        failures.append(
                            f"TEST B: {tf} splice NOT bit-identical (base={base_name}, age={age}, "
                            f"max_abs_diff={md:.3e}) — remove {tf} from SMART520_MTF_SPLICE_TFS "
                            f"or raise the warmup")
                else:
                    # H4/D1 keep snapshot rows — measure the forming-row residual
                    n = min(len(s_df.attrs["feats_np"]), len(full_df.attrs["feats_np"]))
                    md = _max_abs(s_df.attrs["feats_np"][-2:],
                                  full_df.attrs["feats_np"][-2:]) if n >= 2 else float("nan")
                    rows_missing = int(len(full_df) - len(s_df))
                    rec["per_tf"][tf] = {"unspliced_forming_rows_max_abs_diff": md,
                                         "rows_missing_vs_full": rows_missing}
            b_results.append(rec)
            print(f"[selftest] TEST B {base_name} age={age}: "
                  f"{json.dumps(rec['per_tf'])} ({time.time()-t0:.0f}s)", flush=True)
    report["test_b_mtf_splice"] = b_results

    if args.skip_decision:
        return _finish(report, failures, args.out_dir, t0)

    # ── TEST C: decision-state equivalence age 0 vs 1..3 (measured) ─────────
    from gx1.execution.v12_smart_entry_live import SmartEntryLiveInference
    adapter = SmartEntryLiveInference.load(device="cpu")
    end_ts = cv3.index[-1]
    adapter.refresh_multi_tf(cv3)                      # blocking reference (age 0)
    ctx_T = adapter._ctx
    t_dec = time.time()
    frame_ref = adapter._prepare_anchored_frame(loader, cv3, end_ts,
                                                ctx_T.frame_overrides, ctx_T.multi_tf)
    states_ref = adapter._builder.build_states(frame_ref, [end_ts])
    head_ref = adapter.forward_states(states_ref, multi_tf=ctx_T.multi_tf)[0]
    dec_ref_s = time.time() - t_dec
    c_results = []
    stale_snaps = {}
    for age in ages:
        snap_C = adapter._build_ctx_snapshot(cv3.iloc[:-age])
        stale_snaps[age] = snap_C
        t_dec = time.time()
        mtf, ovr, eff_age, spliced = adapter._effective_context(cv3, snap_C, end_ts)
        frame = adapter._prepare_anchored_frame(loader, cv3, end_ts, ovr, mtf)
        states = adapter._builder.build_states(frame, [end_ts])
        head = adapter.forward_states(states, multi_tf=mtf)[0]
        dec_s = time.time() - t_dec
        blocks = {b: _max_abs(states[b][0], states_ref[b][0])
                  for b in ("seq", "snap", "ctx_cont", "ctx_cat")}
        heads = {k: abs(float(head[k]) - float(head_ref[k]))
                 for k in ("p_long", "p_short", "p_flat", "edge_score",
                           "path_quality_pred", "tradable_prob",
                           "mfe_first_n_pred", "bad_path_prob")}
        c_results.append({"age": eff_age, "mtf_spliced": bool(spliced),
                          "decision_seconds": round(dec_s, 2),
                          "state_max_abs_diff": blocks, "forward_abs_diff": heads})
        print(f"[selftest] TEST C age={eff_age}: state={json.dumps(blocks)} "
              f"decision={dec_s:.1f}s ({time.time()-t0:.0f}s)", flush=True)
    report["test_c_decision_equivalence"] = {
        "end_ts": str(end_ts),
        "reference_decision_seconds": round(dec_ref_s, 2),
        "snapshot_build_seconds": round(ctx_T.build_seconds, 1),
        "per_age": c_results,
        "note": "residual staleness = H4/D1 forming MTF rows consumed by group-A per-TF "
                "features (never spliced); overrides + M5/M15/H1 MTF are exact",
    }

    # ── TEST D: thread smoke — refresh in flight during predict ─────────────
    adapter._install_ctx_snapshot(stale_snaps[max_age])          # stale by max_age
    scheduled = adapter.maybe_schedule_ctx_refresh(cv3)
    in_flight_heads = []
    err = None
    try:
        for _ in range(2):
            h = adapter.predict_live_bar(loader, end_ts)
            in_flight_heads.append({k: h[k] for k in
                                    ("p_long", "p_short", "edge_score",
                                     "context_age_m5_bars", "context_refresh_in_flight",
                                     "context_mtf_incremental")})
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        failures.append(f"TEST D: predict_live_bar raised during in-flight refresh: {err}")
    thread = adapter._ctx_refresh_thread
    if thread is not None:
        thread.join(timeout=900)
    swap_ok = adapter._ctx is not None and adapter._ctx.cv3_cutoff == cv3.index[-1]
    if not scheduled:
        failures.append("TEST D: background refresh was not scheduled on a stale snapshot")
    if not swap_ok:
        failures.append(f"TEST D: snapshot swap missing after join "
                        f"(ctx cutoff={adapter._ctx.cv3_cutoff if adapter._ctx else None})")
    post = adapter.predict_live_bar(loader, end_ts)
    post_matches_ref = all(
        abs(float(post[k]) - float(head_ref[k])) == 0.0
        for k in ("p_long", "p_short", "p_flat", "edge_score",
                  "path_quality_pred", "mfe_first_n_pred"))
    if int(post["context_age_m5_bars"]) != 0:
        failures.append(f"TEST D: post-swap age={post['context_age_m5_bars']} != 0")
    if not post_matches_ref:
        failures.append("TEST D: post-swap decision != blocking reference (must be bit-equal)")
    report["test_d_thread_smoke"] = {
        "scheduled": bool(scheduled), "error_during_inflight": err,
        "in_flight_heads": in_flight_heads, "swap_visible_after_join": bool(swap_ok),
        "post_swap_age": int(post["context_age_m5_bars"]),
        "post_swap_equals_blocking_reference": bool(post_matches_ref),
    }
    print(f"[selftest] TEST D done: swap_ok={swap_ok} post==ref={post_matches_ref} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return _finish(report, failures, args.out_dir, t0)


def _finish(report: dict, failures: list[str], out_dir: Path, t0: float) -> int:
    report["failures"] = failures
    report["decision"] = "PASS" if not failures else "FAIL"
    report["runtime_s"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"SMART520_CTX_ASYNC_SELFTEST_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    (out_dir / "SMART520_CTX_ASYNC_SELFTEST_latest.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"decision": report["decision"], "failures": failures,
                      "report": str(out_path), "runtime_s": report["runtime_s"]},
                     indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
