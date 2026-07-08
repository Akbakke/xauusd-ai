#!/usr/bin/env python3
"""TRAIN==SERVE parity gate for the LIVE smart520 entry serving path — MANDATORY
before anything downstream of the serving wave opens (vedtak
SMART_JOINT_POLICY_PROMOTION_20260708, serving-wave gap: parity proof).

What it proves, bar-for-bar, on N historical M5 bars sampled from the offline
TEST split:

  LEG 1 (STATE, hard tolerance 1e-5): the LIVE state builder
  (gx1.execution.v12_smart520_state_live.Smart520StateBuilder) run on the LIVE
  cv3/BASE28 prebuilts reproduces the OFFLINE dataset rows
  (v10_dataset_smart_candidate_julyext_20260705 test split) exactly:
  seq (96,520) + snap (520) + ctx_cont (142) + ctx_cat (5). FAIL-LOUD per
  column name on any deviation.
  EXCEPTION — the FRAME-END target (the live decision-bar situation): the
  offline dataset's swing features use 2-bar-LOOKAHEAD pivot confirmation
  (add_ctx_cont convention, see _compute_offline_swing_block), so the last <=2
  bars' pivot state at the decision bar is PHYSICALLY unknowable at decision
  time — the offline row used future bars the live frame cannot have. That
  bar's diffs are measured and reported VERBATIM as decision_bar_tail_delta
  (advisory, never zero by construction) instead of hard-failing; every
  interior bar (>=2 succeeding bars in frame) must still match at 1e-5.

  LEG 2 (FORWARD, reported + gated at --forward-tol): the LIVE adapter
  (gx1.execution.v12_smart_entry_live.SmartEntryLiveInference) forwards those
  live states through the contract-resolved calibrated bundle and must
  reproduce the PINNED promotion-evidence predictions
  (reports/joint_smart_policy_replay_20260708/heads_rerun/
  selective_edge_predictions.parquet): p_long/p_short/p_flat/edge_score +
  path_quality_pred/tradable_prob/mfe_first_n_pred/bad_path_prob.
  NOTE: the pinned predictions were computed on CUDA; this gate runs the live
  CPU path, so LEG 2 tolerance covers numeric-backend drift ONLY — LEG 1 is the
  bit-level state proof.

Run under the capped runner (heavy: full prebuilt load + augmenters ~7 min):
  scripts/gx1_capped_run.sh --mem 34G .venv/bin/python -m \
      gx1.scripts.verify_smart520_serve_parity_v1 --n-bars 64

Writes SMART520_SERVE_PARITY_<ts>.json + _latest.json under --out-dir and exits
non-zero on FAIL. The launcher's smart-serving gate requires the _latest.json
decision == PASS for the ACTIVE bundle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# CPU-only + deterministic before torch import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Live-serve parity flags (same pins as launch_live_practice.sh / the daemon drop-in)
os.environ.setdefault("GX1_REGIME_V4", "1")
os.environ.setdefault("GX1_TREND_REGIME_FROM_D1", "1")

import numpy as np
import pandas as pd

DEFAULT_DATASET_DIR = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_smart_candidate_julyext_20260705"
)
DEFAULT_PINNED_PREDICTIONS = Path(
    "/home/andre2/GX1_DATA/reports/joint_smart_policy_replay_20260708/"
    "heads_rerun/selective_edge_predictions.parquet"
)
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/smart520_serve_parity_v1")

STATE_BLOCKS = ("seq", "snap", "ctx_cont", "ctx_cat")
FORWARD_COLS = (
    "p_long", "p_short", "p_flat", "edge_score",
    "path_quality_pred", "tradable_prob", "mfe_first_n_pred", "bad_path_prob",
)


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except Exception:
        return "unknown"


def _load_offline_rows(dataset_dir: Path, split: str, times: pd.DatetimeIndex) -> pd.DataFrame:
    """Stream the split parquet batch-wise and keep ONLY the target rows.
    (A one-shot filtered to_table materializes every row group's nested
    (96,520) seq lists ≈ 14+GB — OOM'd the 34G-capped gate 2026-07-08.)"""
    import pyarrow.parquet as pq
    matches = sorted(dataset_dir.glob(f"*_{split}.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one *_{split}.parquet in {dataset_dir}, got {matches}")
    want = set(times.tz_convert("UTC").asi8)
    pf = pq.ParquetFile(matches[0])
    kept: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=512, columns=["time", "seq", "snap", "ctx_cont", "ctx_cat"]):
        ts = pd.to_datetime(pd.Series(batch.column("time").to_pandas()), utc=True)
        mask = ts.astype("int64").isin(want)
        if mask.any():
            df_b = batch.to_pandas()
            df_b["time"] = ts
            kept.append(df_b.loc[mask.to_numpy()])
    if not kept:
        return pd.DataFrame(columns=["seq", "snap", "ctx_cont", "ctx_cat"]).set_index(
            pd.DatetimeIndex([], tz="UTC"))
    df = pd.concat(kept, ignore_index=True)
    return df.set_index("time").sort_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--split", default="test")
    ap.add_argument("--pinned-predictions", type=Path, default=DEFAULT_PINNED_PREDICTIONS)
    ap.add_argument("--n-bars", type=int, default=64)
    ap.add_argument("--min-ts", default="2026-02-01",
                    help="earliest sampled bar (default 2026-02-01: past the anchored-frame "
                         "EMA-convergence window of the extension price layer; the state frame "
                         "anchor itself is the split start — see v12_smart520_state_live)")
    ap.add_argument("--max-ts", default="")
    ap.add_argument("--state-tol", type=float, default=1e-5)
    ap.add_argument("--forward-tol", type=float, default=1e-3,
                    help="LEG-2 numeric-backend (CPU-serve vs pinned-CUDA) tolerance")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dump-npz", type=Path, default=None,
                    help="debug: dump live+offline state matrices for offline diffing")
    args = ap.parse_args()

    t0 = time.time()
    failures: list[str] = []
    report: dict = {
        "schema_version": "smart520_serve_parity_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "pinned_predictions": str(args.pinned_predictions),
        "state_tol": args.state_tol,
        "forward_tol": args.forward_tol,
        "env_pins": {k: os.environ.get(k) for k in ("GX1_REGIME_V4", "GX1_TREND_REGIME_FROM_D1")},
    }

    # ── contract-resolved adapter (fail-closed rule 8) ───────────────────────
    from gx1.execution.v12_smart_entry_live import SmartEntryLiveInference
    from gx1.execution.v12_smart520_state_live import (
        SEQ_LEN_SMART520, SMART520_STATE_FRAME_ANCHOR_UTC,
    )
    from gx1.contracts.signal_bridge_v3 import (
        ORDERED_CTX_CONT_NAMES_V3, ORDERED_CTX_CAT_NAMES_V3,
    )
    adapter = SmartEntryLiveInference.load(device="cpu")
    report["bundle_dir"] = str(adapter.bundle_dir)
    report["operating_point"] = adapter.operating_point
    report["frame_anchor_utc"] = str(SMART520_STATE_FRAME_ANCHOR_UTC)
    signal_names = adapter._builder.ordered_signal_names

    # ── live prebuilts (frozen snapshot — deterministic) ─────────────────────
    from gx1.execution.v12_state_from_prebuilt import PrebuiltStateLoader
    loader = PrebuiltStateLoader()
    loader.load()
    loader._refresh_enabled = False
    cutoff = loader.cutoff_ts
    report["live_prebuilt_cutoff"] = str(cutoff)
    print(f"[parity] live prebuilts loaded (cutoff={cutoff}, {time.time()-t0:.0f}s)", flush=True)

    # ── target bars: evenly spaced sample of the offline split times ─────────
    matches = sorted(args.dataset_dir.glob(f"*_{args.split}.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one *_{args.split}.parquet in {args.dataset_dir}")
    all_times = pd.to_datetime(
        pd.read_parquet(matches[0], columns=["time"])["time"], utc=True
    ).sort_values()
    lo = pd.Timestamp(args.min_ts, tz="UTC")
    hi = min(pd.Timestamp(args.max_ts, tz="UTC") if args.max_ts else all_times.iloc[-1],
             pd.Timestamp(cutoff))
    pool = all_times[(all_times >= lo) & (all_times <= hi)].reset_index(drop=True)
    if len(pool) < args.n_bars:
        raise RuntimeError(f"only {len(pool)} offline bars in [{lo}, {hi}] — need {args.n_bars}")
    pick = np.linspace(0, len(pool) - 1, args.n_bars).round().astype(int)
    targets = pd.DatetimeIndex(pool.iloc[sorted(set(pick.tolist()))])
    report["n_bars"] = int(len(targets))
    report["target_range"] = [str(targets[0]), str(targets[-1])]
    print(f"[parity] {len(targets)} target bars: {targets[0]} .. {targets[-1]}", flush=True)

    # ── LIVE anchored frame + states (shared one-truth adapter path) ─────────
    t_max = targets[-1]
    frame = adapter.build_anchored_frame(loader, t_max)
    report["live_frame"] = {"rows": int(len(frame)),
                            "start": str(frame["time"].iloc[0]), "end": str(frame["time"].iloc[-1])}
    print(f"[parity] anchored frame prepared: {len(frame)} rows ({time.time()-t0:.0f}s)", flush=True)
    states = adapter._builder.build_states(frame, list(targets))
    print(f"[parity] live states built ({time.time()-t0:.0f}s)", flush=True)

    # ── LEG 1: state parity vs offline dataset rows ──────────────────────────
    off = _load_offline_rows(args.dataset_dir, args.split, targets)
    missing = [str(t) for t in targets if t not in off.index]
    if missing:
        failures.append(f"offline rows missing for {len(missing)} targets: {missing[:3]}")
    block_max: dict[str, float] = {b: 0.0 for b in STATE_BLOCKS}
    worst: dict[str, dict] = {}
    per_bar_rows = []
    frame_end_ts = pd.Timestamp(frame["time"].iloc[-1])
    tail_delta: dict = {}
    # per-COLUMN max diff across bars (diagnostic: which families skew)
    col_max_snap = np.zeros(len(signal_names))
    col_max_seq = np.zeros(len(signal_names))
    col_max_ctx = np.zeros(len(ORDERED_CTX_CONT_NAMES_V3))
    _dump: dict[str, np.ndarray] = {}
    for k, ts in enumerate(targets):
        if ts not in off.index:
            continue
        row = off.loc[ts]
        o_seq = np.asarray([np.asarray(x, dtype=np.float64) for x in row["seq"]])
        o_snap = np.asarray(row["snap"], dtype=np.float64)
        o_ctx = np.asarray(row["ctx_cont"], dtype=np.float64)
        o_cat = np.asarray(row["ctx_cat"], dtype=np.int64)
        l_seq = states["seq"][k].astype(np.float64)
        l_snap = states["snap"][k].astype(np.float64)
        l_ctx = states["ctx_cont"][k].astype(np.float64)
        l_cat = states["ctx_cat"][k]
        if o_seq.shape != l_seq.shape:
            failures.append(f"{ts}: seq shape live={l_seq.shape} offline={o_seq.shape}")
            continue
        d_seq = np.abs(l_seq - o_seq)
        d_snap = np.abs(l_snap - o_snap)
        d_ctx = np.abs(l_ctx - o_ctx)
        d_cat = np.abs(l_cat - o_cat).max() if o_cat.shape == l_cat.shape else 999
        if ts == frame_end_ts:
            # decision-bar tail delta (advisory — see module docstring): the
            # offline swing convention peeks 2 bars ahead; the live frame ends here.
            _ctx_off = np.argsort(d_ctx)[::-1][:8]
            _snap_off = np.argsort(d_snap)[::-1][:8]
            tail_delta = {
                "time": str(ts),
                "block_max_abs_diff": {"seq": float(d_seq.max()), "snap": float(d_snap.max()),
                                       "ctx_cont": float(d_ctx.max()), "ctx_cat": float(d_cat)},
                "worst_ctx": [{"col": ORDERED_CTX_CONT_NAMES_V3[i], "diff": float(d_ctx[i])}
                              for i in _ctx_off if d_ctx[i] > args.state_tol],
                "worst_snap": [{"col": signal_names[i], "diff": float(d_snap[i])}
                               for i in _snap_off if d_snap[i] > args.state_tol],
                "note": "irreducible at decision time: offline swing pivots are 2-bar-lookahead-"
                        "confirmed (add_ctx_cont convention); the promoted evidence itself carries "
                        "this property — measured verbatim, not pass/fail",
            }
            per_bar_rows.append({"time": str(ts), "seq": float(d_seq.max()),
                                 "snap": float(d_snap.max()), "ctx_cont": float(d_ctx.max()),
                                 "ctx_cat": float(d_cat), "decision_bar_tail": True})
            continue
        col_max_snap = np.maximum(col_max_snap, d_snap)
        col_max_seq = np.maximum(col_max_seq, d_seq.max(axis=0))
        col_max_ctx = np.maximum(col_max_ctx, d_ctx)
        if args.dump_npz is not None:
            _dump[f"live_seq::{ts.isoformat()}"] = l_seq.astype(np.float32)
            _dump[f"off_seq::{ts.isoformat()}"] = o_seq.astype(np.float32)
            _dump[f"live_ctx::{ts.isoformat()}"] = l_ctx.astype(np.float32)
            _dump[f"off_ctx::{ts.isoformat()}"] = o_ctx.astype(np.float32)
        vals = {"seq": float(d_seq.max()), "snap": float(d_snap.max()),
                "ctx_cont": float(d_ctx.max()), "ctx_cat": float(d_cat)}
        per_bar_rows.append({"time": str(ts), **vals})
        for b, v in vals.items():
            if v > block_max[b]:
                block_max[b] = v
                if b == "seq":
                    r, c = np.unravel_index(int(d_seq.argmax()), d_seq.shape)
                    worst[b] = {"time": str(ts), "row_offset": int(r) - (SEQ_LEN_SMART520 - 1),
                                "col": signal_names[int(c)], "diff": float(d_seq.max())}
                elif b == "snap":
                    c = int(d_snap.argmax())
                    worst[b] = {"time": str(ts), "col": signal_names[c], "diff": float(d_snap.max())}
                elif b == "ctx_cont":
                    c = int(d_ctx.argmax())
                    worst[b] = {"time": str(ts), "col": ORDERED_CTX_CONT_NAMES_V3[c],
                                "diff": float(d_ctx.max())}
                else:
                    worst[b] = {"time": str(ts),
                                "col": ORDERED_CTX_CAT_NAMES_V3[int(np.abs(l_cat - o_cat).argmax())]
                                if o_cat.shape == l_cat.shape else "shape",
                                "diff": float(d_cat)}
    def _top(names, arr, k=25):
        order = np.argsort(arr)[::-1][:k]
        return [{"col": str(names[i]), "max_abs_diff": float(arr[i])}
                for i in order if arr[i] > args.state_tol]
    report["decision_bar_tail_delta"] = tail_delta
    report["state_parity"] = {
        "block_max_abs_diff": block_max, "worst": worst, "tolerance": args.state_tol,
        "top_offenders": {
            "snap": _top(signal_names, col_max_snap),
            "seq": _top(signal_names, col_max_seq),
            "ctx_cont": _top(list(ORDERED_CTX_CONT_NAMES_V3), col_max_ctx),
        },
    }
    if args.dump_npz is not None:
        args.dump_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.dump_npz, **_dump)
        print(f"[parity] debug dump -> {args.dump_npz}", flush=True)
    for b, v in block_max.items():
        tol = 0.0 if b == "ctx_cat" else args.state_tol
        if v > tol:
            failures.append(
                f"STATE block '{b}' max_abs_diff={v:.3e} > tol={tol:.0e} (worst: {worst.get(b)})"
            )
    print(f"[parity] LEG1 state: {json.dumps(block_max)} ({time.time()-t0:.0f}s)", flush=True)

    # ── LEG 2: forward parity vs pinned promotion-evidence predictions ───────
    heads = adapter.forward_states(states)
    print(f"[parity] LEG2 forward done ({time.time()-t0:.0f}s)", flush=True)
    pinned = pd.read_parquet(args.pinned_predictions)
    pinned["time"] = pd.to_datetime(pinned["time"], utc=True)
    pinned = pinned[pinned["model"] == "candidate"].set_index("time").sort_index()
    fwd_max: dict[str, float] = {c: 0.0 for c in FORWARD_COLS}
    fwd_worst: dict[str, str] = {}
    n_fwd = 0
    side_mismatch = []
    take_mismatch = []
    thr = float(adapter.operating_point["edge_score_threshold"])
    sessions = {str(s) for s in adapter.operating_point.get("sessions") or []}
    tail_fwd: dict = {}
    for h in heads:
        ts = h["time"]
        if ts not in pinned.index:
            failures.append(f"pinned prediction missing for {ts}")
            continue
        p = pinned.loc[ts]
        if ts == frame_end_ts:
            # decision-bar: forward delta follows the tail state delta — advisory
            tail_fwd = {c: abs(float(h[c]) - float(p[c])) for c in FORWARD_COLS if c in p}
            tail_fwd["take_flip"] = bool(
                ((h["edge_score"] >= thr) and ({0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}.get(h["session_id"]) in sessions))
                != ((float(p["edge_score"]) >= thr) and (str(p["session"]) in sessions)))
            continue
        n_fwd += 1
        for c in FORWARD_COLS:
            if c not in p:
                continue
            d = abs(float(h[c]) - float(p[c]))
            if d > fwd_max[c]:
                fwd_max[c] = d
                fwd_worst[c] = str(ts)
        if int(h["trade_side"]) != int(p["trade_side"]):
            side_mismatch.append(str(ts))
        live_take = (h["edge_score"] >= thr) and (
            {0: "ASIA", 1: "EU", 2: "OVERLAP", 3: "US"}.get(h["session_id"]) in sessions)
        pin_take = (float(p["edge_score"]) >= thr) and (str(p["session"]) in sessions)
        if live_take != pin_take:
            take_mismatch.append(str(ts))
    report["forward_parity"] = {
        "n_compared": n_fwd, "max_abs_diff": fwd_max, "worst_ts": fwd_worst,
        "trade_side_mismatches": side_mismatch, "take_decision_mismatches": take_mismatch,
        "tolerance": args.forward_tol,
        "decision_bar_tail_forward_delta": tail_fwd,
        "note": "pinned=CUDA fp32 evidence run; live=CPU fp32 — LEG2 bounds backend drift only",
    }
    for c, v in fwd_max.items():
        if v > args.forward_tol:
            failures.append(f"FORWARD '{c}' max_abs_diff={v:.3e} > tol={args.forward_tol:.0e} "
                            f"(worst ts {fwd_worst.get(c)})")
    if side_mismatch:
        failures.append(f"FORWARD trade_side mismatches: {len(side_mismatch)} ({side_mismatch[:3]})")
    if take_mismatch:
        failures.append(f"FORWARD take/skip decision mismatches: {len(take_mismatch)} ({take_mismatch[:3]})")
    print(f"[parity] LEG2 forward: {json.dumps({k: round(v, 9) for k, v in fwd_max.items()})}", flush=True)

    # ── verdict ───────────────────────────────────────────────────────────────
    report["per_bar_state_diffs"] = per_bar_rows
    report["failures"] = failures
    report["decision"] = "PASS" if not failures else "FAIL"
    report["runtime_s"] = round(time.time() - t0, 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out_dir / f"SMART520_SERVE_PARITY_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    (args.out_dir / "SMART520_SERVE_PARITY_latest.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({
        "decision": report["decision"],
        "n_bars": report["n_bars"],
        "state_block_max_abs_diff": block_max,
        "forward_max_abs_diff": fwd_max,
        "decision_bar_tail_delta": tail_delta.get("block_max_abs_diff"),
        "decision_bar_tail_forward_delta": tail_fwd,
        "failures": failures,
        "report": str(out_path),
        "runtime_s": report["runtime_s"],
    }, indent=2), flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
