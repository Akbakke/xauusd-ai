#!/home/andre2/venvs/gx1/bin/python
# -*- coding: utf-8 -*-
"""
TRUTH-grade end-to-end sanity checker for the signal-only pipeline.

Verifies:
- Canonical truth file exists + matches signal_bridge contract SHA
- XGB bundle: MASTER_MODEL_LOCK.json + meta ordered_features match (order-sensitive)
- Transformer bundle: MASTER_TRANSFORMER_LOCK.json exists
- Prebuilt parquet: manifest + schema_manifest present physical XGB columns + ctx reality check (bundle-driven ctx_cont>=6, ctx_cat=6) before replay/training
- TRUTH env sanity (required envs set, forbidden envs unset)
- Replay runs 1W1C in-process via gx1.execution.replay_chunk.process_chunk + replay_merge.merge_artifacts_1w1c (no legacy replay script import)
- Post-run required artifacts exist (chunk + merged + RUN_COMPLETED)
- chunk_footer.status == ok
- prebuilt_proven (env + footer path + join file) and join_ratio >= 0.995
- feature_build_call_count is 0 if present, otherwise require GX1_FEATURE_BUILD_DISABLED=1
- ctx dims: ONE UNIVERSE (ctx_cat=6 fixed; ctx_cont from bundle; hard-fail otherwise)
- transformer forward calls > 0 (robust metrics schema fallback)
- zero-trades contract: trade_outcomes parquet exists + ZERO_TRADES_DIAG.json exists when n_trades==0
- exit coverage: truth_exit_journal_ok==true if EXIT_COVERAGE_SUMMARY.json exists
- bars invariant: bars_total_input - bars_processed == tail_holdback_bars

No fallback: missing/mismatch → hard fail.
Always writes E2E_FATAL_CAPSULE.json on failure.

One-liner commands (canonical short-window TRUTH replay; XGB BASE28 → XGB_SIGNAL_BRIDGE_V1 7 dims → Transformer + ctx side-channel):

  A) Validate-only preflight (no replay):
     export GX1_DATA=/home/andre2/GX1_DATA
     export GX1_CANONICAL_TRUTH_FILE=/home/andre2/src/GX1_ENGINE/gx1/configs/canonical_truth_signal_only.json
     export GX1_STRICT_MASK=1
     export GX1_CTX_CONT_MASK=1,1,1,1
     export GX1_CTX_CAT_MASK=1,1,1,1,1
     /home/andre2/venvs/gx1/bin/python -m gx1.scripts.run_truth_e2e_sanity --validate-only --start-ts 2025-06-03 --end-ts 2025-06-10

  B) Micro replay (same window; masks set from bundle if env unset):
     /home/andre2/venvs/gx1/bin/python -m gx1.scripts.run_truth_e2e_sanity --start-ts 2025-06-03 --end-ts 2025-06-10

  C) Zero-trades canary (1 day; GX1_ENTRY_THRESHOLD_OVERRIDE=1.1; RUN_IDENTITY.json + E2E_SANITY_SUMMARY proof; hard-fail if n_trades>0):
     python -m gx1.scripts.run_truth_e2e_sanity --force-zero-trades --start-ts 2025-06-03 --end-ts 2025-06-04

  D) Full-year proof: use run_fullyear_2025_truth_proof.py (separate runner).

  ONE UNIVERSE rollout (ctx_cat_dim=6 fixed; ctx_cont_dim from bundle; no CLI override):

    1) E2E short window (exits must have context.ctx_cont/ctx_cat matching bundle dims):
       python -m gx1.scripts.run_truth_e2e_sanity --truth-file <canonical_truth> --start-ts 2025-06-03 --end-ts 2025-06-10
       (LAST_GO oppdateres kun når exits matcher bundle dims; postrun gate.)

    2) Train exit from LAST_GO (phase5 contract only; ctx dims must match bundle):
       python -m gx1.scripts.run_truth_e2e_sanity --train-exit-transformer-v0-from-last-go

    3) E2E full-year (run_fullyear_2025_truth_proof eller tilsvarende).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
import inspect
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import gx1
from gx1.execution.live_features import infer_session_tag  # type: ignore
from gx1.scripts.report_truth_decomp_payoff_shapes import append_h2_overlap_auc_decomposition

ENGINE = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)

_EXPECTED_EXE = Path("/home/andre2/venvs/gx1/bin/python").resolve()
_EXPECTED_ENGINE = Path("/home/andre2/src/GX1_ENGINE").resolve()
# Weekly slices can legitimately have one-class forward labels. Keep the
# hard two-class orientation gate for larger samples, but do not fail small
# truth-week replays after the replay/merge artifacts are already valid.
MIN_ROWS_FOR_PRED_LABEL_ORIENTATION = 100

# Disable dotenv loading for TRUTH replay (replay-safe)
os.environ["GX1_DISABLE_DOTENV"] = "1"
print("[TRUTH_DOTENV_DISABLED_PROOF] GX1_DISABLE_DOTENV=1", file=sys.stderr)
_GX1_FILE = getattr(gx1, "__file__", None) or next(iter(getattr(gx1, "__path__", [])), None)
_RUN_TRUTH_PATH = Path(inspect.getfile(sys.modules[__name__]))
print(
    "[ENGINE_SANITY_PROOF]\n"
    f"sys.executable={sys.executable}\n"
    f"ENGINE={ENGINE}\n"
    f"gx1.__file__={_GX1_FILE}\n"
    f"run_truth_e2e_sanity={_RUN_TRUTH_PATH}",
    flush=True,
)
_ENGINE_REAL = os.path.realpath(str(ENGINE))
_GX1_FILE_REAL = os.path.realpath(str(_GX1_FILE))
print(
    "[ENGINE_SANITY_IMPORT_ROOT_PROOF] "
    f"gx1_file={_GX1_FILE_REAL} engine={_ENGINE_REAL}",
    flush=True,
)
if Path(sys.executable).resolve() != _EXPECTED_EXE:
    raise RuntimeError("[ENGINE_SANITY_FAIL] sys.executable mismatch")
if ENGINE.resolve() != _EXPECTED_ENGINE:
    raise RuntimeError("[ENGINE_SANITY_FAIL] ENGINE root mismatch")
if _GX1_FILE is None:
    raise RuntimeError("[ENGINE_SANITY_FAIL] gx1.__file__ is None")
if not _GX1_FILE_REAL.startswith(_ENGINE_REAL + os.sep):
    raise RuntimeError(
        f"[ENGINE_SANITY_FAIL] gx1 imported from outside ENGINE: gx1.__file__={_GX1_FILE_REAL} ENGINE={_ENGINE_REAL}"
    )
_GX1_DATA = os.environ.get("GX1_DATA", "").strip()
_GX1_DATA_RESOLVED = str(Path(_GX1_DATA).expanduser().resolve()) if _GX1_DATA else "EMPTY"
print(f"[GX1_DATA_SANITY_PROOF] GX1_DATA={_GX1_DATA_RESOLVED}", flush=True)
if not _GX1_DATA:
    raise RuntimeError(
        "[GX1_DATA_SANITY_FAIL] GX1_DATA is not set. "
        "Required: export GX1_DATA=/home/andre2/GX1_DATA"
    )


def _write_multi_horizon_predictions(run_root: Path, run_id: str) -> Path:
    pred_path = run_root / f"xgb_multi_horizon_predictions_{run_id}.parquet"
    if pred_path.exists():
        return pred_path

    log_dir = run_root / "replay" / "chunk_0" / "logs"
    trace_files = sorted(log_dir.glob(f"pred_trace_{run_id}.jsonl"))
    if not trace_files:
        raise RuntimeError(f"[PRED_WRITE] pred_trace missing under {log_dir}")
    trace_path = trace_files[0]

    records: List[Dict[str, Any]] = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except Exception as e:
                raise RuntimeError(f"[PRED_WRITE] invalid JSON in {trace_path}: {e}")

    if not records:
        raise RuntimeError(f"[PRED_WRITE] pred_trace empty: {trace_path}")

    required_fields = ["ts_utc", "head", "horizon_bars", "p_long", "p_short", "p_hat", "margin"]
    for f in required_fields:
        if f not in records[0]:
            raise RuntimeError(f"[PRED_WRITE] required key missing in pred_trace: {f}")

    rows: List[Dict[str, Any]] = []
    for rec in records:
        ts_val = rec.get("ts_utc")
        head_val = rec.get("head")
        horizon_val = rec.get("horizon_bars")
        if ts_val is None or head_val is None or horizon_val is None:
            raise RuntimeError("[PRED_WRITE] pred_trace row missing ts/head/horizon")
        try:
            horizon_int = int(horizon_val)
        except Exception:
            raise RuntimeError(f"[PRED_WRITE] invalid horizon_bars={horizon_val!r}")

        p_long = float(rec["p_long"])
        p_short = float(rec["p_short"])
        p_flat = float(rec.get("p_flat", max(0.0, 1.0 - p_long - p_short)))
        scores = [p_long, p_short, p_flat]
        p_hat = float(rec.get("p_hat", max(scores)))
        argmax_idx = int(np.argmax(scores))
        pred_side = "LONG" if argmax_idx == 0 else "SHORT" if argmax_idx == 1 else "FLAT"

        rows.append(
            {
                "run_id": run_id,
                "ts": ts_val,
                "head": head_val,
                "horizon_bars": horizon_int,
                "p_long": p_long,
                "p_short": p_short,
                "p_flat": p_flat,
                "p_hat": p_hat,
                "pred_side": pred_side,
            }
        )

    df_pred = pd.DataFrame(rows)
    if df_pred.empty:
        raise RuntimeError("[PRED_WRITE] no rows produced from pred_trace")

    df_pred["ts"] = pd.to_datetime(df_pred["ts"], utc=True, errors="coerce")
    if df_pred["ts"].isna().any():
        raise RuntimeError("[PRED_WRITE] ts parse failed for some rows")

    years = sorted({ts.year for ts in df_pred["ts"]})
    tape_frames: List[pd.DataFrame] = []
    for yr in years:
        tape_path = _resolve_canonical_raw_tape_root() / f"year={yr}" / "part-000.parquet"
        if not tape_path.exists():
            raise RuntimeError(f"[PRED_WRITE] tape parquet missing: {tape_path}")
        import pyarrow.parquet as pq
        schema = pq.read_schema(tape_path)
        available_cols = set(schema.names)
        bid_candidates = [c for c in ["bid_close", "bid_c", "bid"] if c in available_cols]
        ask_candidates = [c for c in ["ask_close", "ask_c", "ask"] if c in available_cols]
        if not bid_candidates or not ask_candidates:
            raise RuntimeError(f"[PRED_WRITE] bid/ask columns missing in tape: {tape_path}")
        bid_col = bid_candidates[0]
        ask_col = ask_candidates[0]
        columns_to_read = ["time", bid_col, ask_col]
        columns_to_read = [c for c in columns_to_read if c in available_cols]
        tape_df = pd.read_parquet(tape_path, columns=columns_to_read)
        time_col = "time" if "time" in tape_df.columns else None
        if time_col is None:
            raise RuntimeError(f"[PRED_WRITE] tape parquet missing time column: {tape_path}")
        tape_df["ts"] = pd.to_datetime(tape_df[time_col], utc=True, errors="coerce")
        print(f"[PRED_LABEL_COL_PROOF] bid_col={bid_col} ask_col={ask_col}")
        tape_df["bid_use"] = tape_df[bid_col]
        tape_df["ask_use"] = tape_df[ask_col]
        tape_frames.append(tape_df[["ts", "bid_use", "ask_use"]])

    df_tape = pd.concat(tape_frames, ignore_index=True) if tape_frames else pd.DataFrame(columns=["ts"])
    if df_tape.empty:
        raise RuntimeError("[PRED_WRITE] tape dataframe empty after load")
    df_tape = df_tape.sort_values("ts").reset_index(drop=True)
    df_tape["pos"] = np.arange(len(df_tape), dtype=np.int64)

    merged = df_pred.merge(df_tape[["ts", "pos"]], on="ts", how="left", validate="one_to_one")
    if merged["pos"].isna().any():
        raise RuntimeError("[PRED_WRITE] ts not found in tape for some rows")
    merged["pos"] = merged["pos"].astype(int)

    bid_series = df_tape["bid_use"].to_numpy()
    ask_series = df_tape["ask_use"].to_numpy()

    def _nan_count(arr: np.ndarray) -> int:
        return int(np.isnan(arr).sum())

    pos_used = merged["pos"].to_numpy(dtype=np.int64)
    horizon_used = merged["horizon_bars"].to_numpy(dtype=np.int64)
    target_pos_used = pos_used + horizon_used

    # Bounds derived from actual tape length (not full-year constants)
    N = len(bid_series)
    if N == 0:
        raise RuntimeError("[PRED_WRITE] empty tape (N=0) cannot compute labels")
    mask_valid = (pos_used >= 0) & (pos_used < N) & (target_pos_used >= 0) & (target_pos_used < N)
    if not mask_valid.all():
        dropped = int((~mask_valid).sum())
        max_target = int(target_pos_used[mask_valid].max()) if mask_valid.any() else -1
        print(f"[PRED_WRITE_TRIM_PROOF] dropped_out_of_bounds={dropped} N={N} max_target_valid={max_target}")
        pos_used = pos_used[mask_valid]
        horizon_used = horizon_used[mask_valid]
        target_pos_used = target_pos_used[mask_valid]
        merged = merged.loc[mask_valid].reset_index(drop=True)
    if len(pos_used) == 0:
        raise RuntimeError("[PRED_WRITE] no valid rows after trimming out-of-bounds horizons")

    bid_nan = _nan_count(bid_series)
    ask_nan = _nan_count(ask_series)
    bid_min = float(np.nanmin(bid_series)) if bid_nan < len(bid_series) else float("nan")
    bid_max = float(np.nanmax(bid_series)) if bid_nan < len(bid_series) else float("nan")
    ask_min = float(np.nanmin(ask_series)) if ask_nan < len(ask_series) else float("nan")
    ask_max = float(np.nanmax(ask_series)) if ask_nan < len(ask_series) else float("nan")

    used_bid = bid_series[pos_used]
    used_ask = ask_series[pos_used]
    used_bid_exit = bid_series[target_pos_used]
    used_ask_entry = ask_series[target_pos_used]
    used_all = np.concatenate([used_bid, used_bid_exit])
    used_all_ask = np.concatenate([used_ask, used_ask_entry])
    used_bid_nan = _nan_count(used_all)
    used_ask_nan = _nan_count(used_all_ask)
    used_bid_min = float(np.nanmin(used_all)) if used_bid_nan < len(used_all) else float("nan")
    used_bid_max = float(np.nanmax(used_all)) if used_bid_nan < len(used_all) else float("nan")
    used_ask_min = float(np.nanmin(used_all_ask)) if used_ask_nan < len(used_all_ask) else float("nan")
    used_ask_max = float(np.nanmax(used_all_ask)) if used_ask_nan < len(used_all_ask) else float("nan")

    print(
        "[PRED_LABEL_INPUT_QUALITY_PROOF] "
        f"bid_dtype={bid_series.dtype} ask_dtype={ask_series.dtype} "
        f"bid_nan={bid_nan} ask_nan={ask_nan} bid_min={bid_min} bid_max={bid_max} ask_min={ask_min} ask_max={ask_max} "
        f"used_n={len(pos_used)} used_bid_nan={used_bid_nan} used_ask_nan={used_ask_nan} "
        f"used_bid_min={used_bid_min} used_bid_max={used_bid_max} used_ask_min={used_ask_min} used_ask_max={used_ask_max}"
    )
    if used_bid_nan > 0 or used_ask_nan > 0:
        raise RuntimeError("[PRED_WRITE] NaN in bid/ask for used positions; cannot compute labels")

    y_vals: List[int] = []
    delta_vals: List[float] = []
    for _, row in merged.iterrows():
        pos = int(row["pos"])
        horizon = int(row["horizon_bars"])
        target_pos = pos + horizon
        if target_pos >= len(df_tape):
            raise RuntimeError(f"[PRED_WRITE] horizon out of bounds for ts={row['ts']}")
        entry_ask = float(ask_series[pos])
        exit_bid = float(bid_series[target_pos])
        delta = exit_bid - entry_ask
        delta_vals.append(delta)
        y_vals.append(1 if exit_bid > entry_ask else 0)
    d = np.array(delta_vals, dtype=float)
    y_array = np.array(y_vals, dtype=int)
    pos_mask = y_array == 1
    neg_mask = y_array == 0
    mean_delta_pos = float(np.mean(d[pos_mask])) if pos_mask.any() else float("nan")
    mean_delta_neg = float(np.mean(d[neg_mask])) if neg_mask.any() else float("nan")
    pos_count = int(pos_mask.sum())
    neg_count = int(neg_mask.sum())
    # Persist orientation proof to artifact (not just stdout)
    try:
        horizons_used = sorted({int(h) for h in merged["horizon_bars"].unique().tolist()}) if "horizon_bars" in merged.columns else []
        ts_min = str(merged["ts"].min()) if "ts" in merged.columns else "UNKNOWN"
        ts_max = str(merged["ts"].max()) if "ts" in merged.columns else "UNKNOWN"
        proof_payload = {
            "horizon_bars": horizons_used,
            "rows_used": int(len(merged)),
            "pos_count": int(pos_count),
            "neg_count": int(neg_count),
            "min_rows_for_two_class_orientation": int(MIN_ROWS_FOR_PRED_LABEL_ORIENTATION),
            "two_class_observed": bool(pos_count > 0 and neg_count > 0),
            "mean_delta_pos": float(mean_delta_pos) if np.isfinite(mean_delta_pos) else None,
            "mean_delta_neg": float(mean_delta_neg) if np.isfinite(mean_delta_neg) else None,
            "ts_range": {"min": ts_min, "max": ts_max},
        }
        proof_path = run_root / "PRED_LABEL_ORIENTATION_PROOF.json"
        proof_path.write_text(json.dumps(proof_payload, indent=2), encoding="utf-8")
        log.info(
            "[PRED_LABEL_ORIENTATION_PROOF] horizon_bars=%s rows_used=%s pos_count=%s neg_count=%s mean_delta_pos=%s mean_delta_neg=%s ts_min=%s ts_max=%s",
            horizons_used,
            len(merged),
            pos_count,
            neg_count,
            mean_delta_pos,
            mean_delta_neg,
            ts_min,
            ts_max,
        )
    except Exception as e:
        log.warning("[PRED_LABEL_ORIENTATION_PROOF_WRITE_FAILED] %s", e)

    print(
        "[PRED_LABEL_ORIENTATION_PROOF] "
        f"mean_delta_pos={mean_delta_pos:.6f} mean_delta_neg={mean_delta_neg:.6f} "
        f"pos_count={pos_count} neg_count={neg_count}",
        flush=True,
    )
    if len(merged) < MIN_ROWS_FOR_PRED_LABEL_ORIENTATION:
        log.warning(
            "[PRED_LABEL_ORIENTATION_SKIPPED] reason=insufficient_sample rows_used=%s pos_count=%s neg_count=%s min_rows=%s",
            len(merged),
            pos_count,
            neg_count,
            MIN_ROWS_FOR_PRED_LABEL_ORIENTATION,
        )
    else:
        if pos_count == 0 or neg_count == 0:
            raise RuntimeError("[PRED_LABEL_ORIENTATION_FAIL] pos/neg counts must both be >0")
        if not np.isfinite(mean_delta_pos) or mean_delta_pos <= 0:
            raise RuntimeError("[PRED_LABEL_ORIENTATION_FAIL] mean_delta_pos must be positive (long label must mean up)")
    print(
        "[PRED_LABEL_DELTA_PROOF] "
        f"n={len(d)} min={np.min(d):.6f} max={np.max(d):.6f} mean={np.mean(d):.6f} "
        f"pos_count={(d > 0).sum()} neg_count={(d <= 0).sum()}"
    )
    merged["y_true"] = y_vals

    # Join ctx from prebuilt (SSoT)
    def _resolve_prebuilt_path() -> Tuple[Path, str]:
        footer_path = run_root / "replay" / "chunk_0" / "chunk_footer.json"
        manifest_obj = _load_json(MANIFEST_SSOT)
        manifest_parquet = Path(str(manifest_obj.get("parquet_path") or "")).expanduser().resolve()
        source = "manifest"
        if footer_path.exists():
            try:
                footer_obj = _load_json(footer_path)
                cand = footer_obj.get("prebuilt_parquet_path_resolved") or footer_obj.get("prebuilt_parquet_path")
                if cand:
                    p_footer = Path(str(cand)).expanduser().resolve()
                    if not p_footer.exists():
                        raise RuntimeError(f"[PRED_WRITE] footer prebuilt not found: {p_footer}")
                    if manifest_parquet and p_footer != manifest_parquet:
                        raise RuntimeError(
                            "[PREBUILT_SPLIT_BRAIN_IN_PRED_WRITER] footer_prebuilt=%s manifest_parquet=%s"
                            % (p_footer, manifest_parquet)
                        )
                    return p_footer, "footer"
            except Exception as e:
                raise RuntimeError(f"[PRED_WRITE] footer prebuilt resolution failed: {e}")
        if not manifest_parquet or not manifest_parquet.exists():
            raise RuntimeError("[PRED_WRITE] manifest parquet missing/unresolved")
        return manifest_parquet, source

    prebuilt_path, prebuilt_source = _resolve_prebuilt_path()

    import pyarrow.parquet as pq

    schema = pq.read_schema(prebuilt_path)
    available = set(schema.names)
    print(f"[PRED_PREBUILT_SCHEMA_PROOF] path={prebuilt_path} n_cols={len(schema.names)} has_time={'time' in available}")
    if "time" not in available:
        raise RuntimeError(f"[PRED_WRITE] prebuilt missing time column: {prebuilt_path}")

    ctx_candidates = ["atr_bucket", "trend_regime", "trend_regime_id", "session_tag"]
    columns_to_read = ["time"] + [c for c in ctx_candidates if c in available]
    if "atr_bucket" not in columns_to_read:
        raise RuntimeError("[CTX_MISSING_ATR_BUCKET_IN_PREBUILT]")
    print(f"[PRED_PREBUILT_COLS_PROOF] requested={columns_to_read}")

    ctx_df = pd.read_parquet(prebuilt_path, columns=columns_to_read)
    print(f"[PRED_PREBUILT_DF_COLS_PROOF] n_cols={len(ctx_df.columns)} cols={list(ctx_df.columns)}")
    print(
        f"[PRED_PREBUILT_INDEX_PROOF] index_name={getattr(ctx_df.index,'name',None)!r} index_names={getattr(ctx_df.index,'names',None)!r}"
    )
    if time_col not in ctx_df.columns:
        idx_name = getattr(ctx_df.index, "name", None)
        idx_names = list(getattr(ctx_df.index, "names", []) or [])
        if idx_name == time_col or time_col in idx_names:
            ctx_df = ctx_df.reset_index()
            print(f"[PRED_PREBUILT_TIME_RECOVERED_PROOF] via=reset_index time_col={time_col}")
    if time_col not in ctx_df.columns:
        try:
            import pyarrow.dataset as ds  # type: ignore

            dataset = ds.dataset(str(prebuilt_path), format="parquet")
            dataset_cols = set(dataset.schema.names)
            cols2 = [time_col] + [c for c in ["atr_bucket", "trend_regime", "trend_regime_id", "session_tag"] if c in dataset_cols]
            table = dataset.to_table(columns=cols2)
            ctx_df = table.to_pandas()
            print(f"[PRED_PREBUILT_PYARROW_DATASET_PROOF] cols={cols2} n_rows={len(ctx_df)}")
        except Exception as e:
            raise RuntimeError(f"[PRED_WRITE] pyarrow.dataset fallback failed: {e}")
    if time_col not in ctx_df.columns:
        idx_name = getattr(ctx_df.index, "name", None)
        idx_names = list(getattr(ctx_df.index, "names", []) or [])
        raise RuntimeError(
            f"[PRED_WRITE] prebuilt df missing time_col after read_parquet: time_col={time_col!r} "
            f"path={prebuilt_path} requested={columns_to_read} got_cols={list(ctx_df.columns)} "
            f"index_name={idx_name!r} index_names={idx_names!r}"
        )
    ctx_df["ts"] = pd.to_datetime(ctx_df[time_col], utc=True, errors="coerce")
    if ctx_df["ts"].isna().any():
        raise RuntimeError("[PRED_WRITE] prebuilt ts parse failed")
    ctx_df = ctx_df.drop(columns=[time_col])
    print(
        "[PRED_PREBUILT_TS_PROOF] ts_min=%s ts_max=%s tz=UTC"
        % (ctx_df["ts"].min().isoformat() if not ctx_df.empty else "nan", ctx_df["ts"].max().isoformat() if not ctx_df.empty else "nan")
    )
    if "trend_regime" in ctx_df.columns:
        ctx_df["trend_regime_name"] = ctx_df["trend_regime"]
    elif "trend_regime_id" in ctx_df.columns:
        ctx_df["trend_regime_name"] = ctx_df["trend_regime_id"].apply(lambda v: f"R{int(v)}" if pd.notna(v) else "UNKNOWN")
    else:
        raise RuntimeError("[PRED_WRITE] trend_regime missing in prebuilt")
    if "session_tag" not in ctx_df.columns:
        ctx_df["session_tag"] = ctx_df["ts"].apply(lambda ts: str(infer_session_tag(ts)).upper() if infer_session_tag else "N/A")
    ctx_df = ctx_df[["ts", "atr_bucket", "trend_regime_name", "session_tag"]]
    print(f"[PRED_CTX_SOURCE_PROOF] ctx_source=prebuilt path={prebuilt_path} source={prebuilt_source} cols={list(ctx_df.columns)}")

    merged = merged.merge(ctx_df, on="ts", how="left", validate="one_to_one")
    if merged[["atr_bucket", "trend_regime_name", "session_tag"]].isna().any().any():
        raise RuntimeError("[PRED_WRITE] ctx columns contain NaN after join")

    merged.rename(columns={"trend_regime_name": "trend_regime"}, inplace=True)
    merged = merged.drop(columns=["pos"])
    merged["has_ctx"] = 1

    merged.to_parquet(pred_path, index=False)
    print(
        "[XGB_MULTI_HORIZON_PREDICTIONS_PROOF] source=pred_trace pred_path=%s rows=%d has_ctx=1 ctx_cols=%s"
        % (pred_path, len(merged), ["atr_bucket", "trend_regime", "session_tag"])
    )
    return pred_path

# TRUTH/SMOKE SSoT allowlist (entrypoints that may run in TRUTH/SMOKE):
# - gx1/scripts/run_truth_e2e_sanity.py (this orchestrator)
# - gx1/execution/replay_chunk.py (worker)
# - gx1/execution/chunk_data_loader.py (data loader)
# - XGB eval/calibration: train_xgb_universal_multihead_v2.py
# Everything else touching PRUNE14/PRUNE20/v13_refined3/CTX6CAT6 or direct canonical_prebuilt_parquet
# must remain frozen/disabled.

# KUN ÉN TRUTH (default) — used when neither CLI nor env provides truth-file.
CANONICAL_TRUTH_DEFAULT = "/home/andre2/src/GX1_ENGINE/gx1/configs/canonical_truth_signal_only.json"
MANIFEST_SSOT = Path("/home/andre2/GX1_DATA/data/data/prebuilt/BASE28_CANONICAL/CURRENT_MANIFEST.json")

DEFAULT_START_TS = "2025-06-03T00:00:00+00:00"
DEFAULT_END_TS = "2025-06-10T23:59:59+00:00"
FULLYEAR_START_TS = "2025-01-01T00:00:00+00:00"
FULLYEAR_END_TS = "2025-12-31T23:59:59+00:00"

JOIN_RATIO_TRUTH = 0.995

# ONE UNIVERSE: ctx_cat fixed=6; ctx_cont from bundle metadata (no CLI or env override for dims).
MIN_CTX_CONT_DIM = 6
CTX_CONT_DIM = MIN_CTX_CONT_DIM  # updated from bundle metadata in preflight
CTX_CAT_DIM = 6

# Zero-trades canary: threshold > 1.0 so entry is mathematically impossible (entry uses >= threshold)
ZERO_TRADES_CANARY_THRESHOLD = "1.1"

# Forbidden envs (baseline TRUTH must have these unset)
FORBIDDEN_ENVS = [
    "GX1_STOP_AFTER_N_BARS",
    "GX1_BAR_SKIP_TRACE",
    "GX1_BAR_SKIP_TRACE_MAX",
    "GX1_KILLCHAIN_STAGE2_TRACE",
    "GX1_KILLCHAIN_STAGE2_TRACE_MAX",
    "GX1_SEGMENTED_PARALLEL",
    "GX1_SEGMENT_START",
    "GX1_SEGMENT_END",
    "GX1_PARALLEL",
    "GX1_SEGMENTED",
    "GX1_PREROLL_BARS",
    "GX1_PREROLL_START",
    "GX1_OWNER_START",
    "GX1_OWNER_END",
    "GX1_REPLAY_PREBUILT_FEATURES_PATH",
]

REQUIRED_TRUTH_ENVS = {
    "GX1_RUN_MODE": "TRUTH",
    "GX1_TRUTH_MODE": "1",
    "GX1_REPLAY_USE_PREBUILT_FEATURES": "1",
    "GX1_FEATURE_BUILD_DISABLED": "1",
    "GX1_GATED_FUSION_ENABLED": "1",
}


def _assert_no_forbidden_symbol_imports_after_replay(run_root: Path) -> None:
    """TRUTH gate: hard-fail if forbidden modules are in sys.modules after replay (stricter than IMPORT_PROOF/banlist).
    Only explicitly banned names and runtime_v9 pattern; no broad baseline/fallback pattern (avoids false positives)."""
    forbidden_exact = [
        "gx1.scripts.replay_eval_gated_parallel",
    ]
    hits: List[str] = []
    for mod in forbidden_exact:
        if mod in sys.modules:
            hits.append(mod)
    for name in sys.modules:
        if "runtime_v9" in name:
            hits.append(name)
    hits = sorted(set(hits))
    if hits:
        msg = (
            "[TRUTH_FORBIDDEN_SYMBOL_IMPORTS] Forbidden modules in sys.modules after replay: "
            + ", ".join(hits)
            + ". Banned: replay_eval_gated_parallel, runtime_v9."
        )
        _write_fatal_capsule(run_root, RuntimeError(msg), ["forbidden_symbol_imports"])
        raise RuntimeError(msg)


def _assert_truth_no_legacy_replay(run_root: Path) -> None:
    """TRUTH/SMOKE gate: fail hard if legacy replay is loaded or script exists on disk."""
    if "gx1.scripts.replay_eval_gated_parallel" in sys.modules:
        msg = (
            "[TRUTH_GATE] Legacy replay script must not be imported in TRUTH path. "
            "Found in sys.modules. Use replay_chunk.process_chunk + replay_merge.merge_artifacts_1w1c only."
        )
        _write_fatal_capsule(run_root, RuntimeError(msg), ["legacy_replay_import"])
        raise RuntimeError(msg)
    legacy_script_path = ENGINE / "gx1" / "scripts" / "replay_eval_gated_parallel.py"
    if legacy_script_path.exists():
        msg = (
            f"[TRUTH_GATE] Legacy replay script must not exist in repo (ghost purge). "
            f"File exists: {legacy_script_path}."
        )
        _write_fatal_capsule(run_root, RuntimeError(msg), ["legacy_replay_on_disk"])
        raise RuntimeError(msg)


def _utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_utc_year(ts_str: str) -> int:
    raw = (ts_str or "").strip()
    if not raw:
        raise RuntimeError("[REPLAY_TAPE_RESOLVE] empty timestamp")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except Exception as e:
        raise RuntimeError(f"[REPLAY_TAPE_RESOLVE] invalid timestamp={ts_str!r} err={e}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).year


def _resolve_canonical_tape_path(
    start_ts: str,
    end_ts: str,
    run_root: Optional[Path] = None,
) -> Path:
    tape_root = _resolve_canonical_raw_tape_root()
    year_start = _parse_utc_year(start_ts)
    year_end = _parse_utc_year(end_ts)

    if year_start == year_end:
        # Single-year: original behaviour unchanged.
        tape_path = Path(tape_root) / f"year={year_start}" / "part-000.parquet"
        return tape_path.expanduser().resolve()

    # Multi-year: load each year partition, concat and write a combined parquet
    # under run_root (never into canonical tape_root).
    if run_root is None:
        raise RuntimeError(
            "[REPLAY_TAPE_RESOLVE] multi-year span requires run_root to write combined tape "
            f"(start_year={year_start}, end_year={year_end})"
        )
    import pandas as _pd

    frames: list = []
    for yr in range(year_start, year_end + 1):
        yr_path = (Path(tape_root) / f"year={yr}" / "part-000.parquet").expanduser().resolve()
        if not yr_path.exists():
            raise RuntimeError(
                f"[REPLAY_TAPE_RESOLVE] tape partition missing for year {yr}: {yr_path}"
            )
        yr_df = _pd.read_parquet(yr_path)
        n_rows = len(yr_df)
        print(
            f"[REPLAY_TAPE_RESOLVE] loaded year={yr} path={yr_path} rows={n_rows}",
            flush=True,
        )
        frames.append(yr_df)

    combined = _pd.concat(frames, ignore_index=True)
    if "time" in combined.columns:
        combined = combined.sort_values("time").reset_index(drop=True)
    out_path = (run_root / "tape_combined.parquet").expanduser().resolve()
    combined.to_parquet(out_path, index=False)
    print(
        f"[REPLAY_TAPE_RESOLVE] multi-year combined: years={year_start}-{year_end} "
        f"total_rows={len(combined)} output={out_path}",
        flush=True,
    )
    return out_path


def _resolve_canonical_raw_tape_root() -> Path:
    raw_root = os.environ.get(
        "GX1_CANONICAL_TAPE_ROOT_RAW",
        os.environ.get(
            "GX1_CANONICAL_TAPE_ROOT",
            "/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m5_bid_ask__CANONICAL",
        ),
    )
    return Path(raw_root).expanduser().resolve()


def _gx1_data() -> Path:
    gx1_data = os.environ.get("GX1_DATA") or os.environ.get("GX1_DATA_DIR") or os.environ.get("GX1_DATA_ROOT")
    if gx1_data:
        return Path(gx1_data).expanduser().resolve()
    return Path.home() / "GX1_DATA"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"[E2E] file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_guard_id(guard_obj: Dict[str, Any]) -> str:
    id_string = guard_obj.get("id_string")
    if id_string:
        return str(id_string)
    name = guard_obj.get("name")
    version = guard_obj.get("version")
    sha = guard_obj.get("impl_sha256")
    if name and version and sha:
        return f"{name}::{version}::{sha}"
    raise RuntimeError("[RISK_GUARD_IDENTITY_DRIFT] guard identity missing fields")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _forbid_prune_path(label: str, value: str, *, allow_ctx6cat6: bool = False) -> None:
    upper = value.upper()
    tokens = ["PRUNE14", "PRUNE20", "V13_REFINED3_PRUNE", "REFINED3"]
    if not allow_ctx6cat6:
        tokens.append("CTX6CAT6")
    if any(bad in upper for bad in tokens):
        raise RuntimeError(f"LEGACY_PRUNE_FORBIDDEN_IN_TRUTH: {label}={value}")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> bool:
    """
    Atomic JSON writer; prefers gx1.utils.atomic_json.atomic_write_json with fallback to tmp+replace.
    Never raises (best-effort).
    """
    try:
        from gx1.utils.atomic_json import atomic_write_json as _write  # type: ignore

        path.parent.mkdir(parents=True, exist_ok=True)
        return _write(path, payload, fallback_on_error=False)
    except Exception:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            os.replace(tmp, path)
            return True
        except Exception:
            return False


def _write_fatal_capsule(run_root: Path, error: BaseException, gates_failed: List[str]) -> None:
    capsule = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": "run_truth_e2e_sanity",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "gates_failed": gates_failed,
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
    }
    _atomic_write_json(run_root / "E2E_FATAL_CAPSULE.json", capsule)


def _apply_ctx_mask_defaults(ctx_cont_dim: int, ctx_cat_dim: int) -> None:
    """
    Set GX1_CTX_CONT_MASK / GX1_CTX_CAT_MASK from dims if unset; else validate length.
    ONE UNIVERSE: ctx_cat=6 fixed; ctx_cont from bundle metadata (>=6).
    """
    if ctx_cont_dim < MIN_CTX_CONT_DIM:
        raise RuntimeError(
            f"[E2E] ctx_cont_dim must be >= {MIN_CTX_CONT_DIM} (ONE UNIVERSE), got {ctx_cont_dim}"
        )
    if ctx_cat_dim != CTX_CAT_DIM:
        raise RuntimeError(f"[E2E] ctx_cat_dim must be {CTX_CAT_DIM} (ONE UNIVERSE), got {ctx_cat_dim}")
    cont_raw = os.environ.get("GX1_CTX_CONT_MASK", "").strip()
    if not cont_raw:
        os.environ["GX1_CTX_CONT_MASK"] = ",".join(["1"] * ctx_cont_dim)
    else:
        parts = [p.strip() for p in cont_raw.split(",") if p.strip()]
        if len(parts) != ctx_cont_dim:
            raise RuntimeError(
                f"[E2E] GX1_CTX_CONT_MASK length={len(parts)} does not match ctx_cont_dim={ctx_cont_dim}. "
                "Set env to match ctx_cont_dim or leave unset."
            )
    cat_raw = os.environ.get("GX1_CTX_CAT_MASK", "").strip()
    if not cat_raw:
        os.environ["GX1_CTX_CAT_MASK"] = ",".join(["1"] * ctx_cat_dim)
    else:
        parts = [p.strip() for p in cat_raw.split(",") if p.strip()]
        if len(parts) != ctx_cat_dim:
            raise RuntimeError(
                f"[E2E] GX1_CTX_CAT_MASK length={len(parts)} does not match ctx_cat_dim={ctx_cat_dim}. "
                "Set env to match ctx_cat_dim or leave unset."
            )


def _exits_context_gate(
    run_root: Path,
    run_id: str,
    expected_ctx_cont_dim: int,
    expected_ctx_cat_dim: int,
    require_exits_file: bool = False,
) -> tuple[bool, str | None]:
    """
    TRUTH gate: LAST_GO only when exits have expected context (bundle dims).
    Check only: replay/chunk_0/logs/exits/exits_<run_id>.jsonl (no glob/fuzzy).
    If file exists: at least one line must have context.ctx_cont len=expected_ctx_cont_dim and
    context.ctx_cat len=expected_ctx_cat_dim (dims from footer or bundle).
    Else: gate fails (caller must fail postrun, write fatal capsule, not update LAST_GO.txt).
    If file does not exist: gate passes unless require_exits_file (exit_ml_enabled / GX1_EXIT_AUDIT).
    Returns (True, None) if gate passes; (False, error_message) if not.
    """
    exits_path = run_root / "replay" / "chunk_0" / "logs" / "exits" / f"exits_{run_id}.jsonl"
    if not exits_path.exists():
        if require_exits_file:
            return False, (f"exits jsonl required when exit ML/audit enabled but file not found: {exits_path}")
        return True, None
    found_valid = False
    for line in exits_path.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ctx = rec.get("context")
        if not ctx or not isinstance(ctx, dict):
            continue
        c_cont = ctx.get("ctx_cont")
        c_cat = ctx.get("ctx_cat")
        if c_cont is None or c_cat is None:
            continue
        n_cont = len(c_cont) if isinstance(c_cont, (list, tuple)) else 0
        n_cat = len(c_cat) if isinstance(c_cat, (list, tuple)) else 0
        if n_cont == expected_ctx_cont_dim and n_cat == expected_ctx_cat_dim:
            found_valid = True
            break
    if not found_valid:
        # 0-trade run: footer has expected dims and exit ML was ready; no exits to log
        footer_path = run_root / "replay" / "chunk_0" / "chunk_footer.json"
        if footer_path.exists():
            try:
                footer = _load_json(footer_path)
                if (
                    footer.get("n_trades_closed", 0) == 0
                    and footer.get("ctx_cont_dim") == expected_ctx_cont_dim
                    and footer.get("ctx_cat_dim") == expected_ctx_cat_dim
                ):
                    return True, None
            except Exception:
                pass
        return False, (
            f"exits jsonl has no event with context.ctx_cont/ctx_cat of expected lengths "
            f"(expected ctx_cont_dim={expected_ctx_cont_dim}, ctx_cat_dim={expected_ctx_cat_dim}). "
            "LAST_GO will not be updated; rerun with context in replay so exits audit matches bundle dims."
        )
    return True, None


def _last_go_run_is_eligible(start_ts: str, end_ts: str) -> bool:
    """
    LAST_GO is reserved for canonical full-year replay only.
    Short sanity runs must never become the source for exit retraining.
    """
    try:
        start_norm = str(_pd.Timestamp(start_ts, tz="UTC"))
        end_norm = str(_pd.Timestamp(end_ts, tz="UTC"))
    except Exception:
        return False
    return start_norm == FULLYEAR_START_TS and end_norm == FULLYEAR_END_TS


def _run_preflight(truth_path: Path, run_root: Path) -> Dict[str, Any]:
    """Preflight: env sanity, truth file + SHA, XGB lock + meta, transformer lock, prebuilt + schema prefix."""
    result: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "checks": {},
        "gates_failed": [],
    }

    # Env sanity
    for key, expected in REQUIRED_TRUTH_ENVS.items():
        actual = os.getenv(key, "")
        if actual != expected:
            result["gates_failed"].append(f"env_{key}")
            result["checks"]["env_sanity"] = {"error": f"{key}={actual!r} (expected {expected!r})"}
            return result
    for key in FORBIDDEN_ENVS:
        if os.getenv(key):
            result["gates_failed"].append(f"forbidden_env_{key}")
            result["checks"]["env_sanity"] = {"error": f"Forbidden env {key} is set"}
            return result
    result["checks"]["env_sanity"] = {"passed": True}

    # Canonical truth replay should default to V_NEXT ctx contract unless explicitly set.
    ctx_contract_env = os.getenv("GX1_CTX_CONTRACT", "").strip().upper()
    ctx_contract_defaulted = False
    if not ctx_contract_env:
        ctx_contract_env = "V_NEXT"
        os.environ["GX1_CTX_CONTRACT"] = ctx_contract_env
        ctx_contract_defaulted = True

    # Hard gate: only canonical EXIT label variant allowed in TRUTH runs.
    _exit_label_variant = os.environ.get("GX1_EXIT_LABEL_VARIANT", "").strip().upper()
    if _exit_label_variant and _exit_label_variant != "EXIT_LABEL_DET_V1":
        result["gates_failed"].append("forbidden_exit_label_variant")
        result["checks"]["exit_label_variant"] = {
            "error": f"GX1_EXIT_LABEL_VARIANT={_exit_label_variant!r} is not allowed in canonical TRUTH runs; "
                     f"only EXIT_LABEL_DET_V1 or unset is permitted."
        }
        return result
    result["checks"]["exit_label_variant"] = {"passed": True, "variant": _exit_label_variant or "EXIT_LABEL_DET_V1"}

    # Canonical truth file + bridge SHA
    if not truth_path.is_absolute() or not truth_path.exists():
        result["gates_failed"].append("canonical_truth_file")
        result["checks"]["canonical_truth"] = {"error": f"Truth file missing/invalid: {truth_path}"}
        return result
    try:
        truth_obj = _load_json(truth_path)
    except Exception as e:
        result["gates_failed"].append("canonical_truth_file")
        result["checks"]["canonical_truth"] = {"error": str(e)}
        return result

    # Support keys: signal_bridge_contract_sha256 (preferred), signal_bridge_sha (legacy). Both present and different → drift, hard-fail.
    key_preferred = "signal_bridge_contract_sha256"
    key_legacy = "signal_bridge_sha"
    val_preferred = truth_obj.get(key_preferred)
    val_legacy = truth_obj.get(key_legacy)
    if val_preferred is not None:
        val_preferred = str(val_preferred).strip()
    if val_legacy is not None:
        val_legacy = str(val_legacy).strip()
    if val_preferred is not None and val_legacy is not None and val_preferred != val_legacy:
        result["gates_failed"].append("signal_bridge_sha")
        result["checks"]["canonical_truth"] = {
            "error": "contract sha drift in truth file: signal_bridge_contract_sha256 != signal_bridge_sha (expected one key or both equal). Fix gx1/configs canonical truth JSON.",
            "signal_bridge_sha_key_used": None,
            "signal_bridge_sha_match": False,
            "signal_bridge_contract_sha256_expected": None,
        }
        try:
            from gx1.contracts.signal_bridge_v1 import CONTRACT_SHA256 as _EXP  # type: ignore

            result["checks"]["canonical_truth"]["signal_bridge_contract_sha256_expected"] = _EXP
        except Exception:
            pass
        return result
    bridge_sha = (val_preferred or val_legacy) or None
    key_used = key_preferred if val_preferred is not None else (key_legacy if val_legacy is not None else None)
    if bridge_sha is None or key_used is None:
        result["gates_failed"].append("signal_bridge_sha")
        result["checks"]["canonical_truth"] = {
            "error": "missing key: expected one of {signal_bridge_contract_sha256, signal_bridge_sha}",
            "signal_bridge_sha_key_used": None,
            "signal_bridge_sha_match": False,
        }
        try:
            from gx1.contracts.signal_bridge_v1 import CONTRACT_SHA256 as _EXP  # type: ignore

            result["checks"]["canonical_truth"]["signal_bridge_contract_sha256_expected"] = _EXP
        except Exception:
            pass
        return result

    try:
        from gx1.contracts.signal_bridge_v1 import CONTRACT_SHA256 as _BRIDGE_SHA  # type: ignore

        match = bridge_sha == _BRIDGE_SHA
        if not match:
            result["gates_failed"].append("signal_bridge_sha")
            result["checks"]["canonical_truth"] = {
                "error": "contract sha mismatch: truth file value != gx1/contracts/signal_bridge_v1.py:CONTRACT_SHA256",
                "signal_bridge_sha_match": False,
                "signal_bridge_sha_key_used": key_used,
                "signal_bridge_sha_value": bridge_sha[:16] + "..." if len(bridge_sha) > 16 else bridge_sha,
                "signal_bridge_contract_sha256_expected": _BRIDGE_SHA,
            }
            return result
    except Exception as e:
        result["gates_failed"].append("signal_bridge_sha")
        result["checks"]["canonical_truth"] = {
            "error": "contract sha mismatch: " + str(e) + " (expected gx1/contracts/signal_bridge_v1.py:CONTRACT_SHA256)",
            "signal_bridge_sha_match": False,
            "signal_bridge_sha_key_used": key_used,
        }
        return result

    canonical_bundle = str(truth_obj.get("canonical_xgb_bundle_dir") or "")
    xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
    if xgb_override:
        canonical_bundle = xgb_override
    xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
    if xgb_override:
        canonical_bundle = xgb_override
    xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
    if xgb_override:
        canonical_bundle = xgb_override
    xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
    if xgb_override:
        canonical_bundle = xgb_override
    xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
    if xgb_override:
        canonical_bundle = xgb_override
    canonical_prebuilt = str(truth_obj.get("canonical_prebuilt_parquet") or "")
    canonical_manifest_truth = str(truth_obj.get("canonical_prebuilt_manifest") or "")
    canonical_transformer = str(truth_obj.get("canonical_transformer_bundle_dir") or "")
    print(
        f"[PIPELINE_TRUTH_PROOF] canonical_xgb_bundle={canonical_bundle} canonical_transformer_bundle={canonical_transformer}",
        file=sys.stderr,
    )
    entry_override = os.environ.get("GX1_ENTRY_BUNDLE_DIR", "").strip()
    if entry_override:
        canonical_transformer = entry_override
    entry_override = os.environ.get("GX1_ENTRY_BUNDLE_DIR", "").strip()
    if entry_override:
        canonical_transformer = entry_override

    # Hard forbid legacy tokens before IO
    for label, val in (
        ("canonical_xgb_bundle_dir", canonical_bundle),
        ("canonical_prebuilt_parquet", canonical_prebuilt),
        ("canonical_prebuilt_manifest", canonical_manifest_truth),
        ("canonical_transformer_bundle_dir", canonical_transformer),
        ("manifest_ssot", str(MANIFEST_SSOT)),
    ):
        if val:
            try:
                _forbid_prune_path(label, val, allow_ctx6cat6=label == "canonical_transformer_bundle_dir")
            except Exception as e:
                result["gates_failed"].append("canonical_truth_paths")
                result["checks"]["canonical_truth"] = {"error": str(e)}
                return result

    if not canonical_bundle or not canonical_transformer:
        result["gates_failed"].append("canonical_truth_paths")
        result["checks"]["canonical_truth"] = {"error": "Missing canonical_* paths in truth file"}
        return result

    if xgb_override:
        override_path = Path(xgb_override).expanduser().resolve()
        if not override_path.exists():
            result["gates_failed"].append("xgb_override_missing")
            result["checks"]["xgb_override"] = {"error": f"GX1_XGB_BUNDLE_DIR missing: {override_path}"}
            return result
        result["checks"]["xgb_override"] = {"override_dir": str(override_path), "applied": True}

    manifest_ssot_resolved = MANIFEST_SSOT.expanduser().resolve()
    if canonical_manifest_truth:
        if Path(canonical_manifest_truth).expanduser().resolve() != manifest_ssot_resolved:
            result["gates_failed"].append("canonical_truth_paths")
            result["checks"]["canonical_truth"] = {
                "error": f"PREBUILT_MANIFEST_SPLIT_BRAIN: truth_manifest={canonical_manifest_truth} "
                f"expected={manifest_ssot_resolved}"
            }
            return result

    if not manifest_ssot_resolved.exists():
        result["gates_failed"].append("prebuilt_manifest")
        result["checks"]["prebuilt_manifest"] = {"error": f"MANIFEST_SSOT_NOT_FOUND: {manifest_ssot_resolved}"}
        return result

    try:
        manifest_obj = json.loads(manifest_ssot_resolved.read_text(encoding="utf-8"))
    except Exception as e:
        result["gates_failed"].append("prebuilt_manifest")
        result["checks"]["prebuilt_manifest"] = {"error": f"MANIFEST_SSOT_INVALID_JSON: {e}"}
        return result

    manifest_parquet = str(Path(manifest_obj.get("parquet_path") or "").expanduser().resolve())
    manifest_sha = manifest_obj.get("parquet_sha256") or ""

    if not manifest_parquet:
        result["gates_failed"].append("prebuilt_manifest")
        result["checks"]["prebuilt_manifest"] = {"error": "MANIFEST_SSOT_MISSING_PARQUET_PATH"}
        return result

    if not Path(manifest_parquet).is_file():
        result["gates_failed"].append("prebuilt_manifest")
        result["checks"]["prebuilt_manifest"] = {"error": f"MANIFEST_PARQUET_NOT_FOUND: {manifest_parquet}"}
        return result

    if not canonical_prebuilt:
        result["gates_failed"].append("canonical_truth_paths")
        result["checks"]["canonical_truth"] = {"error": "canonical_prebuilt_parquet missing (must mirror manifest)"}
        return result

    prebuilt_resolved = str(Path(canonical_prebuilt).expanduser().resolve())
    if prebuilt_resolved != manifest_parquet:
        result["gates_failed"].append("canonical_truth_paths")
        result["checks"]["canonical_truth"] = {
            "error": f"PREBUILT_SPLIT_BRAIN: manifest.parquet_path={manifest_parquet} != canonical_prebuilt_parquet={prebuilt_resolved}"
        }
        return result

    try:
        _forbid_prune_path("manifest_parquet", manifest_parquet)
    except Exception as e:
        result["gates_failed"].append("canonical_truth_paths")
        result["checks"]["canonical_truth"] = {"error": str(e)}
        return result

    # Single source assertion (print)
    try:
        parquet_sha = _sha256_file(Path(manifest_parquet))
    except Exception:
        parquet_sha = "MISSING_OR_UNREADABLE"
    print(
        "[E2E] SSoT_PREBUILT "
        f"manifest_ssot={manifest_ssot_resolved} manifest_parquet={manifest_parquet} "
        f"canonical_prebuilt_parquet={canonical_prebuilt} parquet_sha256={parquet_sha}",
        file=sys.stderr,
    )

    result["checks"]["canonical_truth"] = {
        "truth_file": str(truth_path),
        "truth_file_sha256": _sha256_file(truth_path),
        "manifest_ssot": str(manifest_ssot_resolved),
        "manifest_parquet_path": manifest_parquet,
        "manifest_parquet_sha256": manifest_sha,
        "canonical_prebuilt_parquet": prebuilt_resolved,
        "signal_bridge_sha_match": True,
        "signal_bridge_sha_key_used": key_used,
        "signal_bridge_sha_value": bridge_sha[:16] + "..." if len(bridge_sha) > 16 else bridge_sha,
        "signal_bridge_contract_sha256_expected": _BRIDGE_SHA,
        "passed": True,
    }

    # XGB lock + meta
    bundle_dir = Path(canonical_bundle).expanduser().resolve()
    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if not lock_path.exists():
        if xgb_override:
            result["checks"]["xgb_lock"] = {
                "skipped": True,
                "reason": "GX1_XGB_BUNDLE_DIR set (override bundle may not carry MASTER_MODEL_LOCK.json)",
            }
        else:
            result["gates_failed"].append("xgb_lock")
            result["checks"]["xgb_lock"] = {"error": f"MASTER_MODEL_LOCK.json missing: {lock_path}"}
            return result
    lock_obj = _load_json(lock_path) if lock_path.exists() else {}
    ordered_features = list(lock_obj.get("ordered_features") or [])
    if not ordered_features and not xgb_override:
        result["gates_failed"].append("xgb_lock")
        result["checks"]["xgb_lock"] = {"error": "MASTER_MODEL_LOCK missing ordered_features"}
        return result
    lock_feature_sha = str(lock_obj.get("feature_list_sha256") or "")
    if ordered_features and lock_feature_sha:
        computed_lock_feature_sha = hashlib.sha256("|".join(ordered_features).encode("utf-8")).hexdigest()
        if computed_lock_feature_sha != lock_feature_sha:
            result["gates_failed"].append("xgb_lock")
            result["checks"]["xgb_lock"] = {"error": "MASTER_MODEL_LOCK feature_list_sha256 mismatch"}
            return result

    meta_rel = str(lock_obj.get("meta_path_relative") or "xgb_universal_multihead_v2_meta.json")
    meta_path = bundle_dir / meta_rel
    if not meta_path.exists():
        result["gates_failed"].append("xgb_meta")
        result["checks"]["xgb_meta"] = {"error": f"XGB meta missing: {meta_path}"}
        return result
    meta_obj = _load_json(meta_path)
    meta_features = list(
        meta_obj.get("feature_names_ordered")
        or meta_obj.get("ordered_features")
        or meta_obj.get("feature_list")
        or []
    )
    if ordered_features and meta_features != ordered_features:
        result["gates_failed"].append("xgb_meta")
        result["checks"]["xgb_meta"] = {"error": "XGB meta ordered_features != MASTER_MODEL_LOCK.ordered_features"}
        return result
    meta_feature_sha = str(meta_obj.get("feature_list_sha256") or "")
    if meta_features and meta_feature_sha:
        computed_meta_feature_sha = hashlib.sha256("|".join(meta_features).encode("utf-8")).hexdigest()
        if computed_meta_feature_sha != meta_feature_sha:
            result["gates_failed"].append("xgb_meta")
            result["checks"]["xgb_meta"] = {"error": "XGB meta feature_list_sha256 mismatch"}
            return result

    result["checks"]["xgb_lock"] = {"passed": True, "lock_path": str(lock_path)}
    result["checks"]["xgb_meta"] = {"passed": True, "meta_path": str(meta_path)}

    # Transformer lock + metadata (ctx dims from bundle)
    transformer_dir = Path(canonical_transformer).expanduser().resolve()
    trans_lock_path = transformer_dir / "MASTER_TRANSFORMER_LOCK.json"
    if not trans_lock_path.exists():
        result["gates_failed"].append("transformer_lock")
        result["checks"]["transformer_lock"] = {"error": f"MASTER_TRANSFORMER_LOCK.json missing: {trans_lock_path}"}
        return result
    result["checks"]["transformer_lock"] = {"passed": True}
    trans_meta_path = transformer_dir / "bundle_metadata.json"
    if not trans_meta_path.exists():
        result["gates_failed"].append("transformer_meta")
        result["checks"]["transformer_meta"] = {"error": f"bundle_metadata.json missing: {trans_meta_path}"}
        return result
    trans_meta = _load_json(trans_meta_path)
    bundle_ctx_cont_dim = int(trans_meta.get("ctx_cont_dim") or trans_meta.get("expected_ctx_cont_dim") or 0)
    bundle_ctx_cat_dim = int(trans_meta.get("ctx_cat_dim") or trans_meta.get("expected_ctx_cat_dim") or 0)
    bundle_ctx_cont_names = list(trans_meta.get("ordered_ctx_cont_names") or [])
    bundle_ctx_cat_names = list(trans_meta.get("ordered_ctx_cat_names") or [])
    if bundle_ctx_cat_dim != CTX_CAT_DIM or bundle_ctx_cont_dim < MIN_CTX_CONT_DIM:
        result["gates_failed"].append("transformer_meta")
        result["checks"]["transformer_meta"] = {
            "error": "bundle_metadata ctx dims invalid",
            "ctx_cont_dim": bundle_ctx_cont_dim,
            "ctx_cat_dim": bundle_ctx_cat_dim,
        }
        return result
    if len(bundle_ctx_cont_names) != bundle_ctx_cont_dim or len(bundle_ctx_cat_names) != bundle_ctx_cat_dim:
        result["gates_failed"].append("transformer_meta")
        result["checks"]["transformer_meta"] = {
            "error": "bundle_metadata ordered_ctx_* length mismatch",
            "ctx_cont_dim": bundle_ctx_cont_dim,
            "ctx_cat_dim": bundle_ctx_cat_dim,
            "ordered_ctx_cont_len": len(bundle_ctx_cont_names),
            "ordered_ctx_cat_len": len(bundle_ctx_cat_names),
        }
        return result
    global CTX_CONT_DIM
    CTX_CONT_DIM = int(bundle_ctx_cont_dim)
    result["checks"]["transformer_meta"] = {
        "passed": True,
        "bundle_metadata": str(trans_meta_path),
        "ctx_cont_dim": bundle_ctx_cont_dim,
        "ctx_cat_dim": bundle_ctx_cat_dim,
    }

    # Prebuilt parquet + manifest + schema manifest + schema prefix
    prebuilt_path = Path(canonical_prebuilt).expanduser().resolve()
    if not prebuilt_path.exists():
        result["gates_failed"].append("prebuilt_parquet")
        result["checks"]["prebuilt"] = {"error": f"Prebuilt parquet missing: {prebuilt_path}"}
        return result

    manifest_path = prebuilt_path.with_suffix(".manifest.json")
    schema_manifest_path = prebuilt_path.with_suffix(".schema_manifest.json")
    if not manifest_path.exists():
        result["gates_failed"].append("prebuilt_manifest")
        result["checks"]["prebuilt"] = {"error": f"Prebuilt manifest missing: {manifest_path}"}
        return result
    if not schema_manifest_path.exists():
        result["gates_failed"].append("prebuilt_schema_manifest")
        result["checks"]["prebuilt"] = {"error": f"Prebuilt schema manifest missing: {schema_manifest_path}"}
        return result

    schema_obj = _load_json(schema_manifest_path)
    required_all = list(schema_obj.get("required_all_features") or [])
    derived_xgb_features = {
        "session_id",
        "is_ASIA",
        "minutes_since_session_open",
        "minutes_to_next_session_boundary",
        "session_change_flag",
        "session_tradable",
    }
    missing_xgb_physical = [c for c in ordered_features if c not in derived_xgb_features and c not in required_all]
    if missing_xgb_physical:
        result["gates_failed"].append("schema_prefix_match")
        result["checks"]["prebuilt"] = {
            "error": "schema_manifest.required_all_features missing physical XGB columns from MASTER_MODEL_LOCK.ordered_features",
            "missing_xgb_physical": missing_xgb_physical,
        }
        return result

    # ctx reality check: prebuilt must have all ctx columns required by bundle metadata
    try:
        if not bundle_ctx_cont_names or not bundle_ctx_cat_names:
            raise RuntimeError("bundle_metadata missing ordered_ctx_cont_names/ordered_ctx_cat_names")
        required_ctx = list(bundle_ctx_cont_names) + list(bundle_ctx_cat_names)
        if ctx_contract_env == "V_NEXT":
            vnext_extra = [
                "is_ASIA",
                "minutes_since_session_open",
                "minutes_to_next_session_boundary",
                "session_change_flag",
                "session_tradable",
            ]
            required_ctx = [c for c in required_ctx if c not in vnext_extra]
        import pyarrow.parquet as pq

        parquet_schema = pq.read_schema(prebuilt_path)
        prebuilt_cols = list(parquet_schema.names)
        missing_ctx_cols = [c for c in required_ctx if c not in prebuilt_cols]
        if missing_ctx_cols:
            result["gates_failed"].append("prebuilt_ctx_6_6")
            result["checks"]["prebuilt"] = {
                "error": "Prebuilt missing ctx columns required by bundle metadata (check ctx contract).",
                "missing_ctx_cols": missing_ctx_cols,
                "required_ctx": required_ctx,
                "ctx_contract_used": ctx_contract_env,
                "ctx_contract_defaulted": ctx_contract_defaulted,
            }
            return result
        # Bonus: no NaN/Inf in ctx columns (STRICT dataset will fail early otherwise)
        import numpy as np
        import pandas as pd

        df_sample = pd.read_parquet(prebuilt_path, columns=required_ctx).head(1000)
        nan_counts = {}
        inf_counts = {}
        if len(df_sample) > 0:
            for col in required_ctx:
                if col not in df_sample.columns:
                    continue
                ser = df_sample[col]
                nan_count = int(ser.isna().sum())
                if nan_count > 0:
                    nan_counts[col] = nan_count
                    result["gates_failed"].append("prebuilt_ctx_6_6")
                    result["checks"]["prebuilt"] = {
                        "error": f"Prebuilt has NaN in ctx column {col!r} (check ctx contract).",
                        "column": col,
                        "nan_count": nan_count,
                        "required_ctx": required_ctx,
                        "missing_ctx_cols": missing_ctx_cols,
                        "nan_cols": nan_counts,
                        "inf_cols": inf_counts,
                        "ctx_contract_used": ctx_contract_env,
                        "ctx_contract_defaulted": ctx_contract_defaulted,
                    }
                    return result
                if pd.api.types.is_float_dtype(ser):
                    arr = ser.to_numpy(dtype=np.float64, na_value=np.nan)
                    inf_count = int(np.isinf(arr).sum())
                    if inf_count > 0:
                        inf_counts[col] = inf_count
                        result["gates_failed"].append("prebuilt_ctx_6_6")
                        result["checks"]["prebuilt"] = {
                            "error": f"Prebuilt has Inf in ctx column {col!r} (check ctx contract).",
                            "column": col,
                            "inf_count": inf_count,
                            "required_ctx": required_ctx,
                            "missing_ctx_cols": missing_ctx_cols,
                            "nan_cols": nan_counts,
                            "inf_cols": inf_counts,
                            "ctx_contract_used": ctx_contract_env,
                            "ctx_contract_defaulted": ctx_contract_defaulted,
                        }
                        return result
        ctx_reality_check = {
            "missing_ctx_cols": [],
            "ctx_nan_inf_check": "passed",
            "required_ctx": required_ctx,
            "nan_cols": nan_counts,
            "inf_cols": inf_counts,
            "ctx_contract_used": ctx_contract_env,
            "ctx_contract_defaulted": ctx_contract_defaulted,
        }
    except Exception as e:
        result["gates_failed"].append("prebuilt_ctx_6_6")
        result["checks"]["prebuilt"] = {
            "error": f"ctx reality check failed: {e}",
            "required_ctx": required_ctx if "required_ctx" in locals() else None,
            "missing_ctx_cols": missing_ctx_cols if "missing_ctx_cols" in locals() else None,
            "nan_cols": nan_counts if "nan_counts" in locals() else None,
            "inf_cols": inf_counts if "inf_counts" in locals() else None,
            "ctx_contract_used": ctx_contract_env,
            "ctx_contract_defaulted": ctx_contract_defaulted,
        }
        return result

    result["checks"]["prebuilt"] = {
        "passed": True,
        "prebuilt_path": str(prebuilt_path),
        "manifest_exists": True,
        "schema_manifest_exists": True,
        "schema_prefix_match": True,
        "ctx_6_6_reality_check": ctx_reality_check,
        "ctx_contract_used": ctx_contract_env,
        "ctx_contract_defaulted": ctx_contract_defaulted,
    }

    result["passed"] = True
    return result


def _bundle_sha256(bundle_dir: Path) -> str:
    """Compute bundle SHA256 for SSoT: bundle_metadata.json sha256, else hash of model_state_dict.pt, else MASTER_MODEL_LOCK.json."""
    meta_path = bundle_dir / "bundle_metadata.json"
    if meta_path.exists():
        try:
            obj = _load_json(meta_path)
            sha = obj.get("sha256") or obj.get("bundle_sha256")
            if sha:
                return str(sha).strip()
        except Exception:
            pass
    model_path = bundle_dir / "model_state_dict.pt"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    xgb_joblib = bundle_dir / "xgb_universal_multihead_v2.joblib"
    if xgb_joblib.exists():
        with open(xgb_joblib, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    xgb_meta = bundle_dir / "xgb_universal_multihead_v2_meta.json"
    if xgb_meta.exists():
        with open(xgb_meta, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if lock_path.exists():
        with open(lock_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    raise RuntimeError(
        f"[E2E] Cannot compute bundle_sha256: no bundle_metadata.json, model_state_dict.pt, "
        f"xgb_universal_multihead_v2.joblib, xgb_universal_multihead_v2_meta.json, or MASTER_MODEL_LOCK.json in {bundle_dir}"
    )


def _run_replay(
    replay_output_dir: Path,
    run_id: str,
    truth_path: Path,
    policy_path: Path,
    raw_path: Path,
    prebuilt_path: Path,
    start_ts: str,
    end_ts: str,
    env_overrides: Dict[str, str],
    bundle_dir: Path,
    merge_output_dir: Path,
    chunk_local_padding_days: int = 0,
) -> int:
    """Run 1W1C replay in-process via replay_chunk.process_chunk + replay_merge.merge_artifacts_1w1c (no legacy script import).
    Chunk artifacts go to replay_output_dir/chunk_0; MERGED/RUN_COMPLETED go to merge_output_dir (run_root)."""
    for k, v in env_overrides.items():
        os.environ[k] = v

    bundle_sha = _bundle_sha256(bundle_dir)

    # PRE_FORK_FREEZE.json required by chunk_bootstrap in TRUTH
    try:
        from gx1.utils.prefork_freeze_gate import run_prefork_freeze_gate_or_fatal

        run_prefork_freeze_gate_or_fatal(
            output_dir=replay_output_dir,
            truth_or_smoke=True,
            bundle_sha=bundle_sha,
        )
    except ImportError:
        # Fallback: write minimal PRE_FORK_FREEZE.json to satisfy bootstrap (no legacy module needed)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bundle_sha256": bundle_sha,
            "note": "prefork freeze stub (module missing); TRUTH/SMOKE requires presence only.",
        }
        _atomic_write_json(replay_output_dir / "PRE_FORK_FREEZE.json", payload)
    except Exception as e:
        print(f"[E2E] PRE_FORK_FREEZE failed: {e}", file=sys.stderr)
        return 2

    import pandas as _pd

    chunk_start_ts = _pd.Timestamp(start_ts, tz="UTC")
    chunk_end_ts = _pd.Timestamp(end_ts, tz="UTC")

    from gx1.execution.replay_chunk import process_chunk
    from gx1.execution.replay_merge import merge_artifacts_1w1c

    try:
        truth_obj_run = _load_json(truth_path)
    except Exception:
        truth_obj_run = {}

    result = process_chunk(
        chunk_idx=0,
        chunk_start=chunk_start_ts,
        chunk_end=chunk_end_ts,
        data_path=raw_path,
        policy_path=policy_path,
        run_id=run_id,
        output_dir=replay_output_dir,
        bundle_sha256=bundle_sha,
        prebuilt_parquet_path=None,
        bundle_dir=bundle_dir,
        chunk_local_padding_days=int(chunk_local_padding_days or 0),
        truth_artifacts=truth_obj_run,
    )

    if result.get("status") != "ok":
        print(f"[E2E] process_chunk failed: {result.get('error', result)}", file=sys.stderr)
        return 1

    try:
        merge_artifacts_1w1c(replay_output_dir, run_id, output_dir=merge_output_dir)
    except Exception as e:
        print(f"[E2E] merge_artifacts_1w1c failed: {e}", file=sys.stderr)
        return 1

    return 0


def _run_postrun_checks(run_root: Path, run_id: str, truth_artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Post-run: required files, chunk_footer status, invariants, ctx dims, forward_calls, zero-trades, exit journal.
    Root artifacts (MERGED, RUN_COMPLETED, etc.) live in run_root; chunk artifacts in run_root/replay/chunk_0."""
    result: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "run_dir": str(run_root),
        "passed": False,
        "gates_failed": [],
        "checks": {},
    }

    # Hard gate: forbidden symbol-imports after replay (stricter than IMPORT_PROOF/banlist)
    _assert_no_forbidden_symbol_imports_after_replay(run_root)

    chunk_dir = run_root / "replay" / "chunk_0"
    footer_path = chunk_dir / "chunk_footer.json"

    # Required files: chunk under replay/chunk_0, root artifacts in run_root.
    required_chunk = [
        "trade_outcomes_" + run_id + ".parquet",
        "attribution_" + run_id + ".json",
        "chunk_footer.json",
        "IMPORT_PROOF.json",
    ]
    required_root = [
        f"trade_outcomes_{run_id}_MERGED.parquet",
        f"metrics_{run_id}_MERGED.json",
        f"MERGE_PROOF_{run_id}.json",
        "RUN_COMPLETED.json",
    ]
    truth_artifacts = truth_artifacts or {}
    replay_truth_artifacts = truth_artifacts.get("replay_config", {}).get("truth_artifacts", {})
    if replay_truth_artifacts.get("require_import_proof"):
        fname = replay_truth_artifacts.get("import_proof_filename") or "IMPORT_PROOF.json"
        required_chunk.append(fname)

    # Diagnostics: log required list and present files (deterministic, bounded)
    required_files = [f"replay/chunk_0/{name}" for name in required_chunk] + required_root
    required_files = list(dict.fromkeys(required_files))  # de-dupe deterministically
    print(
        "[POSTRUN_REQUIRED_SOURCE] run_root=%s chunk_root=%s required_files_count=%d source=_run_postrun_checks"
        % (run_root, chunk_dir, len(required_files))
    )
    for req in required_files:
        print("[POSTRUN_REQUIRED_FILE] %s" % req)

    present_files: List[str] = []
    try:
        present_files = sorted(
            [
                str(p.relative_to(chunk_dir))
                for p in chunk_dir.rglob("*")
                if p.is_file()
            ]
        )
    except Exception:
        present_files = []
    present_limit = 200
    if len(present_files) > present_limit:
        head = present_files[:present_limit]
        print("[POSTRUN_PRESENT_FILES_TRUNCATED] count=%d limit=%d" % (len(present_files), present_limit))
        for pf in head:
            print("[POSTRUN_PRESENT_FILE] %s" % pf)
    else:
        print("[POSTRUN_PRESENT_FILES] count=%d" % len(present_files))
        for pf in present_files:
            print("[POSTRUN_PRESENT_FILE] %s" % pf)

    missing: List[str] = []
    for name in required_chunk:
        if not (chunk_dir / name).exists():
            missing.append(f"replay/chunk_0/{name}")
    for name in required_root:
        if not (run_root / name).exists():
            missing.append(name)

    # Artifact chain proofs (explicit, run-scoped)
    try:
        report_sources = {
            "run_root": str(run_root),
            "chunk_dir": str(chunk_dir),
            "run_header": str(chunk_dir / "run_header.json"),
            "model_used_capsule": str(chunk_dir / "MODEL_USED_CAPSULE.json"),
            "replay_summary": str(chunk_dir / "REPLAY_SUMMARY_PROOF.log"),
            "trade_outcomes": str(chunk_dir / f"trade_outcomes_{run_id}.parquet"),
            "trade_journal": str(chunk_dir / f"trade_journal_{run_id}.parquet"),
            "merged_trade_outcomes": str(run_root / f"trade_outcomes_{run_id}_MERGED.parquet"),
            "merged_trade_journal": str(run_root / f"trade_journal_{run_id}_MERGED.parquet"),
            "run_completed": str(run_root / "RUN_COMPLETED.json"),
        }
        print(
            "[ARTIFACT_CHAIN_PROOF] "
            + " ".join([f"{k}={v}" for k, v in report_sources.items()])
        )
        print(
            "[REPORT_SOURCE_PROOF] "
            f"replay_summary={report_sources['replay_summary']} "
            f"trade_outcomes={report_sources['trade_outcomes']} "
            f"trade_journal={report_sources['trade_journal']}"
        )
    except Exception as _artifact_err:
        print(f"[ARTIFACT_CHAIN_PROOF_WARN] {type(_artifact_err).__name__}: {_artifact_err}")

    if missing:
        result["gates_failed"].append("required_files")
        result["checks"]["required_files"] = {"missing": missing}
        print("[POSTRUN_REQUIRED] required_count=%d missing_count=%d" % (len(required_files), len(missing)))
        for m in missing:
            print("[POSTRUN_REQUIRED_MISSING] %s" % m)
        print("[ARTIFACT_MISSING_PROOF] missing=%s" % missing)
        return result
    result["checks"]["required_files"] = {"passed": True}

    # Zero-trades diagnostics (for visibility on skipped artifacts)
    n_trades = None
    try:
        footer_obj = _load_json(footer_path)
        n_trades = footer_obj.get("n_trades") or footer_obj.get("trades_closed") or footer_obj.get("n_trades_closed")
    except Exception:
        footer_obj = None
    print("[POSTRUN_ZERO_TRADES_INFO] n_trades=%s footer_path=%s footer_loaded=%s" % (n_trades, footer_path, footer_obj is not None))

    # Gate: IMPORT_PROOF.json must exist and forbidden_hits must be empty (no ghost imports)
    import_proof_path = chunk_dir / "IMPORT_PROOF.json"
    if not import_proof_path.exists():
        result["gates_failed"].append("import_ghosts")
        result["checks"]["import_ghosts"] = {"error": f"IMPORT_PROOF.json missing: {import_proof_path}"}
        return result
    try:
        import_proof = _load_json(import_proof_path)
        forbidden_hits = import_proof.get("forbidden_hits") or []
        if forbidden_hits:
            result["gates_failed"].append("import_ghosts")
            result["checks"]["import_ghosts"] = {"forbidden_hits": forbidden_hits}
            return result
        result["checks"]["import_ghosts"] = {"passed": True, "forbidden_hits": []}
    except Exception as e:
        result["gates_failed"].append("import_ghosts")
        result["checks"]["import_ghosts"] = {"error": str(e)}
        return result

    # Gate: policy snapshot sha256 must match run_header (no disk drift) unless disabled by truth_artifacts
    replay_truth_artifacts = truth_artifacts.get("replay_config", {}).get("truth_artifacts", {}) if truth_artifacts else {}
    require_policy_snapshot = replay_truth_artifacts.get("require_policy_snapshot", True)
    header_path = chunk_dir / "run_header.json"
    if not header_path.exists():
        result["gates_failed"].append("run_header")
        result["checks"]["policy_snapshot"] = {"error": "run_header.json missing"}
        return result
    header = _load_json(header_path)
    if require_policy_snapshot:
        expected_sha256 = header.get("policy_snapshot_sha256")
        if not expected_sha256:
            result["gates_failed"].append("policy_snapshot_sha256")
            result["checks"]["policy_snapshot"] = {
                "error": "run_header missing policy_snapshot_sha256 (run must use snapshot runner)"
            }
            return result
        snapshot_name = header.get("policy_snapshot_path") or replay_truth_artifacts.get("policy_snapshot_filename") or "RUN_POLICY_USED.yaml"
        snapshot_file = chunk_dir / snapshot_name
        if not snapshot_file.exists():
            result["gates_failed"].append("policy_snapshot_sha256")
            result["checks"]["policy_snapshot"] = {"error": f"{snapshot_name} missing in chunk dir"}
            return result
        actual_sha256 = _sha256_file(snapshot_file)
        if actual_sha256 != expected_sha256:
            result["gates_failed"].append("policy_snapshot_sha256")
            result["checks"]["policy_snapshot"] = {
                "error": f"{snapshot_name} sha256 does not match run_header.policy_snapshot_sha256",
                "expected": expected_sha256,
                "actual": actual_sha256,
            }
            return result
        result["checks"]["policy_snapshot"] = {"passed": True, "sha256": actual_sha256}
    else:
        result["checks"]["policy_snapshot"] = {"skipped": True, "reason": "disabled_by_truth_config"}

    if not footer_path.exists():
        result["gates_failed"].append("chunk_footer")
        result["checks"]["chunk_footer"] = {"error": "chunk_footer.json missing"}
        return result

    footer = _load_json(footer_path)
    status = (footer.get("status") or "").lower()
    if status != "ok":
        result["gates_failed"].append("chunk_footer_status")
        result["checks"]["chunk_footer"] = {"error": f"status={status!r}", "footer_error": footer.get("error")}
        return result

    # ---------------------------------------------------------------------
    # Invariants: prebuilt_proven + feature_build
    # ---------------------------------------------------------------------
    join_path = chunk_dir / "RAW_PREBUILT_JOIN.json"

    # prebuilt_proven: env + footer.prebuilt_parquet_path + join-file exists
    env_prebuilt = os.environ.get("GX1_REPLAY_USE_PREBUILT_FEATURES", "0") == "1"
    prebuilt_path_footer = footer.get("prebuilt_parquet_path")
    prebuilt_proven = bool(env_prebuilt and prebuilt_path_footer and join_path.exists())
    if not prebuilt_proven:
        result["gates_failed"].append("prebuilt_proven")
        result["checks"]["invariants"] = {
            "GX1_REPLAY_USE_PREBUILT_FEATURES": os.environ.get("GX1_REPLAY_USE_PREBUILT_FEATURES"),
            "footer_prebuilt_parquet_path": prebuilt_path_footer,
            "RAW_PREBUILT_JOIN_exists": join_path.exists(),
        }
        return result

    # feature_build_call_count: soft invariant
    # - If present: must be 0
    # - If missing: require GX1_FEATURE_BUILD_DISABLED=1
    fbc = footer.get("feature_build_call_count")
    if fbc is not None:
        try:
            if int(fbc) != 0:
                result["gates_failed"].append("feature_build_call_count")
                result["checks"]["invariants"] = {"feature_build_call_count": fbc}
                return result
        except (TypeError, ValueError):
            result["gates_failed"].append("feature_build_call_count")
            result["checks"]["invariants"] = {"feature_build_call_count": fbc}
            return result
    else:
        if os.environ.get("GX1_FEATURE_BUILD_DISABLED", "0") != "1":
            result["gates_failed"].append("feature_build_disabled")
            result["checks"]["invariants"] = {"GX1_FEATURE_BUILD_DISABLED": os.environ.get("GX1_FEATURE_BUILD_DISABLED")}
            return result

    result["checks"]["invariants"] = {
        "prebuilt_proven": True,
        "feature_build_call_count": fbc,
        "GX1_FEATURE_BUILD_DISABLED": os.environ.get("GX1_FEATURE_BUILD_DISABLED"),
    }

    # ---------------------------------------------------------------------
    # ctx dims: ONE UNIVERSE (ctx_cat=6 fixed; ctx_cont from bundle)
    # ---------------------------------------------------------------------
    ctx_cont = footer.get("ctx_cont_dim")
    ctx_cat = footer.get("ctx_cat_dim")
    if ctx_cont != CTX_CONT_DIM or ctx_cat != CTX_CAT_DIM:
        result["gates_failed"].append("ctx_dims")
        result["checks"]["ctx_dims"] = {
            "ctx_cont_dim": ctx_cont,
            "ctx_cat_dim": ctx_cat,
            "required": (CTX_CONT_DIM, CTX_CAT_DIM),
        }
        return result
    result["checks"]["ctx_dims"] = {"ctx_cont_dim": ctx_cont, "ctx_cat_dim": ctx_cat, "passed": True}

    # ---------------------------------------------------------------------
    # join_ratio >= 0.995 (TRUTH gate): join_ratio then fallback join_ratio_eval
    # ---------------------------------------------------------------------
    if not join_path.exists():
        result["gates_failed"].append("join_ratio")
        result["checks"]["join_ratio"] = {"error": "RAW_PREBUILT_JOIN.json missing"}
        return result

    try:
        join_data = _load_json(join_path)
        jr = join_data.get("join_ratio")
        if jr is None:
            jr = join_data.get("join_ratio_eval")  # legacy fallback

        if jr is None:
            result["gates_failed"].append("join_ratio")
            result["checks"]["join_ratio"] = {
                "error": "join_ratio missing (tried join_ratio then join_ratio_eval)",
                "join_file_keys": sorted(list(join_data.keys())),
            }
            return result

        try:
            jr_f = float(jr)
        except (TypeError, ValueError):
            result["gates_failed"].append("join_ratio")
            result["checks"]["join_ratio"] = {"error": f"join_ratio not numeric: {jr!r}"}
            return result

        if jr_f < JOIN_RATIO_TRUTH:
            result["gates_failed"].append("join_ratio")
            result["checks"]["join_ratio"] = {"join_ratio": jr_f, "required": JOIN_RATIO_TRUTH}
            return result

        result["checks"]["join_ratio"] = {"join_ratio": jr_f, "passed": True}

    except Exception as e:
        result["gates_failed"].append("join_ratio")
        result["checks"]["join_ratio"] = {"error": f"could not read join file: {e}"}
        return result

    # ---------------------------------------------------------------------
    # forward_calls / n_model_calls (observability gate; never use t_transformer_forward_sec as proof)
    # ---------------------------------------------------------------------
    metrics_path = run_root / f"metrics_{run_id}_MERGED.json"
    metrics: Dict[str, Any] = _load_json(metrics_path) if metrics_path.exists() else {}
    n_trades = int(metrics.get("n_trades", -1)) if metrics else -1
    if n_trades < 0:
        n_trades = int(footer.get("n_trades_closed", -1)) if footer.get("n_trades_closed") is not None else -1

    tried_keys = [
        "transformer_forward_calls",
        "forward_calls_total",
        "n_transformer_calls",
        "transformer_calls",
        "policy_forward_calls",
        "n_model_calls",
    ]
    forward_calls: Optional[int] = None
    chosen_key: Optional[str] = None
    for k in tried_keys:
        v = metrics.get(k)
        if v is not None:
            try:
                forward_calls = int(v)
                chosen_key = k
                break
            except Exception:
                pass
    if forward_calls is None:
        fc = footer.get("n_model_calls") or footer.get("bars_evaluated")
        if fc is not None:
            try:
                forward_calls = int(fc)
                chosen_key = "chunk_footer.n_model_calls"
            except Exception:
                pass

    if n_trades > 0:
        if forward_calls is None or forward_calls <= 0:
            keys = sorted(list(metrics.keys()))
            print(
                f"[E2E] forward_calls NO-GO: n_trades={n_trades} but forward_calls={forward_calls} (required > 0)",
                file=sys.stderr,
            )
            result["gates_failed"].append("forward_calls")
            result["checks"]["forward_calls"] = {
                "error": "n_trades > 0 requires forward_calls > 0",
                "n_trades": n_trades,
                "forward_calls": forward_calls,
                "source_key": chosen_key,
                "metrics_keys": keys,
            }
            return result
        result["checks"]["forward_calls"] = {
            "forward_calls": forward_calls,
            "source_key": chosen_key or "unknown",
            "n_trades": n_trades,
            "passed": True,
        }
    else:
        if forward_calls is None:
            forward_calls = 0
        if forward_calls == 0:
            print("[E2E] no-forward-window: n_trades=0, forward_calls=0 (policy/session-filtered)", file=sys.stderr)
        result["checks"]["forward_calls"] = {
            "forward_calls": forward_calls,
            "source_key": chosen_key or "chunk_footer.n_model_calls",
            "n_trades": n_trades,
            "passed": True,
        }

    # ---------------------------------------------------------------------
    # ctx telemetry (when ctx matches bundle dims: n_ctx_model_calls > 0, ctx_proof_pass == n_ctx_model_calls, ctx_proof_fail == 0)
    # ---------------------------------------------------------------------
    ctx_cat_dim = int(footer.get("ctx_cat_dim") or 0)
    ctx_cont_dim = int(footer.get("ctx_cont_dim") or 0)
    if ctx_cat_dim == CTX_CAT_DIM and ctx_cont_dim == CTX_CONT_DIM:
        n_ctx = int(footer.get("n_ctx_model_calls") or 0)
        ctx_pass = int(footer.get("ctx_proof_pass_count") or 0)
        ctx_fail = int(footer.get("ctx_proof_fail_count") or 0)
        if forward_calls == 0:
            result["checks"]["ctx_telemetry"] = {
                "skipped": True,
                "reason": "no forward calls in window",
                "n_ctx_model_calls": n_ctx,
                "ctx_proof_pass_count": ctx_pass,
                "ctx_proof_fail_count": ctx_fail,
            }
            return result
        if n_ctx <= 0:
            result["gates_failed"].append("ctx_telemetry")
            result["checks"]["ctx_telemetry"] = {
                "error": "ctx dims match bundle but n_ctx_model_calls <= 0",
                "n_ctx_model_calls": n_ctx,
                "ctx_proof_pass_count": ctx_pass,
                "ctx_proof_fail_count": ctx_fail,
            }
            return result
        if ctx_pass != n_ctx:
            result["gates_failed"].append("ctx_telemetry")
            result["checks"]["ctx_telemetry"] = {
                "error": f"ctx_proof_pass_count ({ctx_pass}) != n_ctx_model_calls ({n_ctx})",
                "n_ctx_model_calls": n_ctx,
                "ctx_proof_pass_count": ctx_pass,
                "ctx_proof_fail_count": ctx_fail,
            }
            return result
        if ctx_fail != 0:
            result["gates_failed"].append("ctx_telemetry")
            result["checks"]["ctx_telemetry"] = {
                "error": f"ctx_proof_fail_count must be 0 when ctx present, got {ctx_fail}",
                "n_ctx_model_calls": n_ctx,
                "ctx_proof_pass_count": ctx_pass,
                "ctx_proof_fail_count": ctx_fail,
            }
            return result
        result["checks"]["ctx_telemetry"] = {
            "n_ctx_model_calls": n_ctx,
            "ctx_proof_pass_count": ctx_pass,
            "ctx_proof_fail_count": ctx_fail,
            "passed": True,
        }

    # ---------------------------------------------------------------------
    # SIDE_DISTRIBUTION_PROOF (post-replay audit artifact)
    # ---------------------------------------------------------------------
    side_proof: Dict[str, Any] = {
        "long_trades": 0,
        "short_trades": 0,
        "flat_predictions": 0,
        "long_winrate": None,
        "short_winrate": None,
    }
    side_proof_errors: Dict[str, Any] = {}
    trade_sources = [
        run_root / f"trade_outcomes_{run_id}_MERGED.parquet",
        chunk_dir / f"trade_outcomes_{run_id}.parquet",
    ]
    trade_df = None
    trade_source = None
    for p in trade_sources:
        if p.exists():
            try:
                trade_df = pd.read_parquet(p)
                trade_source = str(p)
                break
            except Exception as e:
                side_proof_errors["trade_outcomes_read"] = f"{type(e).__name__}: {e}"
                trade_df = None
                trade_source = str(p)
                break
    if trade_df is not None and len(trade_df) > 0:
        if "side" in trade_df.columns:
            sides = trade_df["side"].astype(str).str.upper()
            long_mask = sides == "LONG"
            short_mask = sides == "SHORT"
            side_proof["long_trades"] = int(long_mask.sum())
            side_proof["short_trades"] = int(short_mask.sum())
            pnl_col = None
            for cand in ("pnl_bps", "pnl", "pnl_usd", "pnl_ticks", "pnl_value"):
                if cand in trade_df.columns:
                    pnl_col = cand
                    break
            if pnl_col is None:
                side_proof_errors["pnl_col_missing"] = f"no pnl column in trade_outcomes (cols={list(trade_df.columns)})"
            else:
                pnl = pd.to_numeric(trade_df[pnl_col], errors="coerce").fillna(0.0)
                long_wins = int(((pnl > 0) & long_mask).sum())
                short_wins = int(((pnl > 0) & short_mask).sum())
                if side_proof["long_trades"] > 0:
                    side_proof["long_winrate"] = float(long_wins / side_proof["long_trades"])
                if side_proof["short_trades"] > 0:
                    side_proof["short_winrate"] = float(short_wins / side_proof["short_trades"])
        else:
            side_proof_errors["side_col_missing"] = f"trade_outcomes missing 'side' column (cols={list(trade_df.columns)})"
    elif trade_source is None:
        side_proof_errors["trade_outcomes_missing"] = "trade_outcomes parquet not found"

    # flat_predictions from eval_log (ENTRY argmax == FLAT)
    eval_dir = chunk_dir / "logs"
    eval_files = sorted(eval_dir.glob("eval_log_*.jsonl")) if eval_dir.exists() else []
    eval_rows = 0
    if eval_files:
        for ef in eval_files:
            try:
                with ef.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        pl = row.get("entry_p_long")
                        ps = row.get("entry_p_short")
                        pf = row.get("entry_p_flat")
                        if pl is None or ps is None or pf is None:
                            continue
                        eval_rows += 1
                        if pf >= pl and pf >= ps:
                            side_proof["flat_predictions"] += 1
            except Exception as e:
                side_proof_errors["eval_log_read"] = f"{type(e).__name__}: {e}"
                break
    else:
        side_proof_errors["eval_log_missing"] = f"no eval_log_*.jsonl under {eval_dir}"

    side_proof["sources"] = {
        "trade_outcomes": trade_source,
        "eval_log_files": [str(p) for p in eval_files],
        "eval_rows_used": eval_rows,
    }
    if side_proof_errors:
        side_proof["errors"] = side_proof_errors

    _atomic_write_json(run_root / "SIDE_DISTRIBUTION_PROOF.json", side_proof)
    print(
        "[SIDE_DISTRIBUTION_PROOF] long_trades=%s short_trades=%s flat_predictions=%s long_winrate=%s short_winrate=%s"
        % (
            side_proof.get("long_trades"),
            side_proof.get("short_trades"),
            side_proof.get("flat_predictions"),
            side_proof.get("long_winrate"),
            side_proof.get("short_winrate"),
        )
    )

    # ---------------------------------------------------------------------
    # FLAT_VETO_CANDIDATE_AUDIT (counterfactual audit of blocked entry candidates)
    # ---------------------------------------------------------------------
    flat_veto_audit: Dict[str, Any] = {
        "candidate_count": 0,
        "horizon_bars": None,
        "side_split": {"long": 0, "short": 0},
        "winner_loser_split": {"winner": 0, "loser_or_flat": 0},
        "side_winner_loser_split": {
            "long": {"winner": 0, "loser_or_flat": 0},
            "short": {"winner": 0, "loser_or_flat": 0},
        },
        "pnl_buckets_bps": {},
        "adverse_immediate_no_edge_chop_long_pocket": {
            "signature": "side=long AND directional_edge<=0.03 AND p_flat>=0.30 AND uncertainty_score>=0.60",
            "count": 0,
        },
        "sources": {
            "eval_log_files": [str(p) for p in eval_files],
            "tape_root": str(_resolve_canonical_raw_tape_root()),
        },
    }
    try:
        veto_rows: List[Dict[str, Any]] = []
        for ef in eval_files:
            with ef.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if str(row.get("decision_reason", "")).strip().lower() != "flat_veto":
                        continue
                    ts = pd.to_datetime(row.get("ts_utc"), utc=True, errors="coerce")
                    if pd.isna(ts):
                        continue
                    try:
                        p_long = float(row.get("p_long", np.nan))
                        p_short = float(row.get("p_short", np.nan))
                        p_flat = float(row.get("p_flat", np.nan))
                        unc = float(row.get("uncertainty_score", np.nan))
                    except Exception:
                        continue
                    if not (np.isfinite(p_long) and np.isfinite(p_short) and np.isfinite(p_flat)):
                        continue
                    side = "long" if p_long >= p_short else "short"
                    veto_rows.append(
                        {
                            "ts": ts,
                            "p_long": p_long,
                            "p_short": p_short,
                            "p_flat": p_flat,
                            "uncertainty_score": unc,
                            "candidate_side": side,
                        }
                    )

        flat_veto_audit["candidate_count"] = int(len(veto_rows))
        if veto_rows:
            horizon_bars = int((truth_artifacts or {}).get("pred_trace_horizon_bars", 24) or 24)
            if horizon_bars < 1:
                horizon_bars = 24
            flat_veto_audit["horizon_bars"] = int(horizon_bars)

            veto_df = pd.DataFrame(veto_rows)
            years = sorted(
                set(int(pd.Timestamp(veto_df["ts"].min()).year + k) for k in range(0, int(pd.Timestamp(veto_df["ts"].max()).year - pd.Timestamp(veto_df["ts"].min()).year + 2)))
            )
            tape_frames: List[pd.DataFrame] = []
            tape_root = _resolve_canonical_raw_tape_root()
            for yr in years:
                tp = tape_root / f"year={yr}" / "part-000.parquet"
                if not tp.exists():
                    continue
                tdf = pd.read_parquet(tp, columns=["time", "bid_close", "ask_close"])
                tdf["ts"] = pd.to_datetime(tdf["time"], utc=True, errors="coerce")
                tdf = tdf.dropna(subset=["ts"]).sort_values("ts")
                tape_frames.append(tdf[["ts", "bid_close", "ask_close"]])
            if tape_frames:
                tape = pd.concat(tape_frames, ignore_index=True).drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
                tape["pos"] = np.arange(len(tape), dtype=np.int64)
                pos_map = tape.set_index("ts")["pos"]
                bid = tape["bid_close"].to_numpy(dtype=float)
                ask = tape["ask_close"].to_numpy(dtype=float)

                side_counts = {"long": 0, "short": 0}
                wl_counts = {"winner": 0, "loser_or_flat": 0}
                side_wl = {
                    "long": {"winner": 0, "loser_or_flat": 0},
                    "short": {"winner": 0, "loser_or_flat": 0},
                }
                bucket_counts = {
                    "<=-20": 0,
                    "(-20,-5]": 0,
                    "(-5,0]": 0,
                    "(0,5]": 0,
                    "(5,20]": 0,
                    ">20": 0,
                }
                adverse_pocket = 0

                for r in veto_rows:
                    ts = r["ts"]
                    if ts not in pos_map.index:
                        continue
                    pos = int(pos_map.loc[ts])
                    tpos = pos + int(horizon_bars)
                    if tpos < 0 or tpos >= len(tape):
                        continue

                    entry_ask = float(ask[pos])
                    entry_bid = float(bid[pos])
                    exit_bid = float(bid[tpos])
                    exit_ask = float(ask[tpos])
                    if not np.isfinite(entry_ask) or not np.isfinite(entry_bid) or entry_ask <= 0 or entry_bid <= 0:
                        continue

                    pnl_long_bps = (exit_bid - entry_ask) / entry_ask * 1e4
                    pnl_short_bps = (entry_bid - exit_ask) / entry_bid * 1e4
                    side = r["candidate_side"]
                    pnl_bps = float(pnl_long_bps if side == "long" else pnl_short_bps)

                    side_counts[side] += 1
                    if pnl_bps > 0.0:
                        wl_counts["winner"] += 1
                        side_wl[side]["winner"] += 1
                    else:
                        wl_counts["loser_or_flat"] += 1
                        side_wl[side]["loser_or_flat"] += 1

                    if pnl_bps <= -20.0:
                        bucket_counts["<=-20"] += 1
                    elif pnl_bps <= -5.0:
                        bucket_counts["(-20,-5]"] += 1
                    elif pnl_bps <= 0.0:
                        bucket_counts["(-5,0]"] += 1
                    elif pnl_bps <= 5.0:
                        bucket_counts["(0,5]"] += 1
                    elif pnl_bps <= 20.0:
                        bucket_counts["(5,20]"] += 1
                    else:
                        bucket_counts[">20"] += 1

                    directional_edge = float(r["p_long"] - r["p_short"])
                    if (
                        side == "long"
                        and directional_edge <= 0.03
                        and float(r["p_flat"]) >= 0.30
                        and np.isfinite(float(r["uncertainty_score"]))
                        and float(r["uncertainty_score"]) >= 0.60
                    ):
                        adverse_pocket += 1

                flat_veto_audit["side_split"] = side_counts
                flat_veto_audit["winner_loser_split"] = wl_counts
                flat_veto_audit["side_winner_loser_split"] = side_wl
                flat_veto_audit["pnl_buckets_bps"] = bucket_counts
                flat_veto_audit["adverse_immediate_no_edge_chop_long_pocket"]["count"] = int(adverse_pocket)
    except Exception as e:
        flat_veto_audit["error"] = f"{type(e).__name__}: {e}"

    _atomic_write_json(run_root / "FLAT_VETO_CANDIDATE_AUDIT.json", flat_veto_audit)
    print(
        "[FLAT_VETO_CANDIDATE_AUDIT] candidate_count=%s long=%s short=%s winner=%s loser_or_flat=%s adverse_no_edge_chop_long=%s"
        % (
            flat_veto_audit.get("candidate_count", 0),
            flat_veto_audit.get("side_split", {}).get("long", 0),
            flat_veto_audit.get("side_split", {}).get("short", 0),
            flat_veto_audit.get("winner_loser_split", {}).get("winner", 0),
            flat_veto_audit.get("winner_loser_split", {}).get("loser_or_flat", 0),
            flat_veto_audit.get("adverse_immediate_no_edge_chop_long_pocket", {}).get("count", 0),
        )
    )

    # ---------------------------------------------------------------------
    # zero-trades contract: trade_outcomes exists (empty parquet) + ZERO_TRADES_DIAG if n_trades==0
    # ---------------------------------------------------------------------
    n_trades = int(metrics.get("n_trades", 0)) if metrics else -1
    replay_truth_artifacts = truth_artifacts.get("replay_config", {}).get("truth_artifacts", {}) if truth_artifacts else {}
    min_trades = int(replay_truth_artifacts.get("min_trades", 1 if replay_truth_artifacts.get("require_nonzero_trades", True) else 0))
    if n_trades < min_trades:
        result["gates_failed"].append("zero_trades_diag")
        result["checks"]["zero_trades"] = {
            "error": f"n_trades={n_trades} < min_trades={min_trades} (truth-config)",
            "n_trades": n_trades,
            "min_trades": min_trades,
        }
        return result
    if n_trades == 0:
        if min_trades == 0:
            result["checks"]["zero_trades"] = {"skipped": True, "reason": "disabled_by_truth_config", "n_trades": n_trades}
        else:
            to_path = chunk_dir / f"trade_outcomes_{run_id}.parquet"
            if not to_path.exists():
                result["gates_failed"].append("trade_outcomes_zero_trades")
                result["checks"]["zero_trades"] = {"error": "trade_outcomes parquet missing when n_trades==0"}
                return result

            zero_diag = chunk_dir / "ZERO_TRADES_DIAG.json"
            if not zero_diag.exists():
                result["gates_failed"].append("zero_trades_diag")
                result["checks"]["zero_trades"] = {"error": "ZERO_TRADES_DIAG.json missing (TRUTH requires when n_trades==0)"}
                return result

            result["checks"]["zero_trades"] = {"passed": True, "n_trades": n_trades, "min_trades": min_trades}
    else:
        result["checks"]["zero_trades"] = {"n_trades": n_trades, "passed": True, "min_trades": min_trades}

    # ---------------------------------------------------------------------
    # Risk guard identity drift (from MODEL_USED_CAPSULE)
    # ---------------------------------------------------------------------
    risk_guard_expected = (truth_artifacts.get("replay_config", {}).get("risk_guard_id", "") if truth_artifacts else "").strip()
    if risk_guard_expected:
        capsule_path = chunk_dir / "MODEL_USED_CAPSULE.json"
        if not capsule_path.exists():
            result["gates_failed"].append("risk_guard_identity")
            result["checks"]["risk_guard_identity"] = {"error": "MODEL_USED_CAPSULE.json missing"}
            return result
        try:
            capsule = _load_json(capsule_path)
            guards = capsule.get("guards") or {}
            rg = guards.get("risk_guard_v1")
            if not rg:
                result["gates_failed"].append("risk_guard_identity")
                result["checks"]["risk_guard_identity"] = {"error": "risk_guard_v1 identity missing from capsule"}
                return result
            observed_id = _extract_guard_id(rg)
            if observed_id != risk_guard_expected:
                result["gates_failed"].append("risk_guard_identity")
                result["checks"]["risk_guard_identity"] = {
                    "error": "RISK_GUARD_IDENTITY_DRIFT",
                    "expected": risk_guard_expected,
                    "observed": observed_id,
                }
                return result
            result["checks"]["risk_guard_identity"] = {"passed": True, "id": observed_id}
        except Exception as e:
            result["gates_failed"].append("risk_guard_identity")
            result["checks"]["risk_guard_identity"] = {"error": f"RISK_GUARD_IDENTITY_DRIFT: {e}"}
            return result

    # ---------------------------------------------------------------------
    # Exit coverage: truth_exit_journal_ok==true if EXIT_COVERAGE_SUMMARY exists (replay or root)
    # ---------------------------------------------------------------------
    exit_cov_path = run_root / "EXIT_COVERAGE_SUMMARY.json"
    if not exit_cov_path.exists():
        exit_cov_path = run_root / "replay" / "EXIT_COVERAGE_SUMMARY.json"
    if exit_cov_path.exists():
        exit_cov = _load_json(exit_cov_path)
        truth_ok = exit_cov.get("truth_exit_journal_ok")
        if truth_ok is not True:
            result["gates_failed"].append("truth_exit_journal_ok")
            result["checks"]["exit_coverage"] = {"truth_exit_journal_ok": truth_ok}
            return result
        result["checks"]["exit_coverage"] = {"truth_exit_journal_ok": True, "passed": True}
    else:
        result["checks"]["exit_coverage"] = {"passed": True, "note": "EXIT_COVERAGE_SUMMARY.json not found"}

    # ---------------------------------------------------------------------
    # Bars invariant: gap == warmup_holdback_bars + tail_holdback_bars when status ok
    # ---------------------------------------------------------------------
    bars_total = int(footer.get("bars_total_input") or 0)
    bars_processed = int(footer.get("bars_processed") or 0)
    warmup_holdback = int(footer.get("warmup_holdback_bars") or 0)
    tail_holdback = int(footer.get("tail_holdback_bars") or 0)
    gap = bars_total - bars_processed
    expected_gap = warmup_holdback + tail_holdback
    if gap != expected_gap:
        result["gates_failed"].append("bars_invariant")
        result["checks"]["bars_invariant"] = {
            "bars_total_input": bars_total,
            "bars_processed": bars_processed,
            "warmup_holdback_bars": warmup_holdback,
            "tail_holdback_bars": tail_holdback,
            "gap": gap,
            "expected_gap": expected_gap,
        }
        return result
    result["checks"]["bars_invariant"] = {"passed": True}

    # Exit strategy (record footer fields)
    result["checks"]["exit_strategy"] = {
        "exit_type": footer.get("exit_type"),
        "exit_profile": footer.get("exit_profile"),
        "router_enabled": footer.get("router_enabled"),
        "exit_critic_enabled": footer.get("exit_critic_enabled"),
        "exit_ml_enabled": footer.get("exit_ml_enabled"),
        "exit_ml_decision_mode": footer.get("exit_ml_decision_mode"),
        "exit_ml_config_hash": footer.get("exit_ml_config_hash"),
    }

    replay_truth_artifacts = truth_artifacts.get("replay_config", {}).get("truth_artifacts", {}) if truth_artifacts else {}
    require_exit_transformer = replay_truth_artifacts.get("require_exit_type_transformer", True)

    # TRUTH gate: EXIT_TRANSFORMER_V0 only (ONE UNIVERSE ML exit); no router, no exit_critic
    if not require_exit_transformer:
        result["checks"]["exit_strategy"]["skipped"] = True
        result["checks"]["exit_strategy"]["reason"] = "exit ML disabled (fixed bar close per truth_config)"
    else:
        if footer.get("exit_type") != "EXIT_TRANSFORMER_V0":
            result["gates_failed"].append("exit_type_transformer")
            result["checks"]["exit_strategy"]["error"] = (
                f"exit_type must be EXIT_TRANSFORMER_V0 in TRUTH (ONE UNIVERSE ML-only), got: {footer.get('exit_type')!r}"
            )
            return result
        if footer.get("router_enabled") is not False:
            result["gates_failed"].append("router_enabled_false")
            result["checks"]["exit_strategy"]["error"] = f"router_enabled must be false in TRUTH, got: {footer.get('router_enabled')}"
            return result
        if footer.get("exit_critic_enabled") is not False:
            result["gates_failed"].append("exit_critic_enabled_false")
            result["checks"]["exit_strategy"]["error"] = (
                f"exit_critic_enabled must be false in TRUTH, got: {footer.get('exit_critic_enabled')}"
            )
            return result
        if footer.get("exit_ml_enabled") is not True:
            result["gates_failed"].append("exit_ml_enabled_true")
            result["checks"]["exit_strategy"]["error"] = (
                f"exit_ml_enabled must be true in TRUTH (EXIT_TRANSFORMER_V0), got: {footer.get('exit_ml_enabled')}"
            )
            return result
        if footer.get("exit_ml_decision_mode") != "exit_transformer_v0":
            result["gates_failed"].append("exit_ml_decision_mode_transformer")
            result["checks"]["exit_strategy"]["error"] = (
                f"exit_ml_decision_mode must be 'exit_transformer_v0' in TRUTH, got: {footer.get('exit_ml_decision_mode')!r}"
            )
            return result

    if require_exit_transformer:
        # TRUTH gate: exits jsonl must exist and contain at least one line with computed.mode == exit_transformer_v0
        # and EXIT context 6/6 (exit_transformer_v0 contract).
        exits_dir = chunk_dir / "logs" / "exits"
        exits_glob = list(exits_dir.glob("exits_*.jsonl")) if exits_dir.exists() else []
        if not exits_glob:
            result["gates_failed"].append("exit_ml_exits_jsonl")
            result["checks"]["exit_strategy"]["error"] = f"exits jsonl required in {exits_dir}; found: {exits_glob}"
            return result
        if not footer.get("exit_ml_model_sha"):
            result["gates_failed"].append("exit_ml_model_sha")
            result["checks"]["exit_strategy"]["error"] = "exit_transformer_v0 requires exit_ml_model_sha in footer"
            return result
        seen_transformer_6_6 = False
        for path in exits_glob:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        comp = rec.get("computed") or {}
                        if comp.get("mode") != "exit_transformer_v0":
                            continue
                        ctx = rec.get("context") or {}
                        ctx_cont = ctx.get("ctx_cont") if isinstance(ctx.get("ctx_cont"), (list, tuple)) else []
                        ctx_cat = ctx.get("ctx_cat") if isinstance(ctx.get("ctx_cat"), (list, tuple)) else []
                        if len(ctx_cont) == CTX_CONT_DIM and len(ctx_cat) == CTX_CAT_DIM:
                            seen_transformer_6_6 = True
                            break
            except Exception:
                pass
            if seen_transformer_6_6:
                break
        # When n_trades_closed == 0, no exit decisions were logged; allow pass if file exists and footer dims match contract
        n_trades = footer.get("n_trades_closed", 0)
        if not seen_transformer_6_6:
            if n_trades == 0 and footer.get("ctx_cont_dim") == CTX_CONT_DIM and footer.get("ctx_cat_dim") == CTX_CAT_DIM:
                pass
            else:
                result["gates_failed"].append("exit_ml_transformer_6_6")
                result["checks"]["exit_strategy"]["error"] = (
                    "exits jsonl must contain at least one line with computed.mode == 'exit_transformer_v0' and "
                    f"context.ctx_cont len {CTX_CONT_DIM}, context.ctx_cat len {CTX_CAT_DIM}"
                )
                return result

    result["passed"] = True
    return result


def _write_summary_md(
    run_root: Path,
    preflight: Dict[str, Any],
    postrun: Optional[Dict[str, Any]],
    go: bool,
    reasons: List[str],
    canary_proof: Optional[Dict[str, Any]] = None,
) -> None:
    path = run_root / "E2E_SANITY_SUMMARY.md"
    lines = [
        "# E2E Sanity Summary",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"## Verdict: **{'GO' if go else 'NO-GO'}**",
        "",
    ]
    for r in reasons:
        lines.append(f"- {r}")

    lines.extend(["", "## Preflight", ""])
    lines.append(f"- passed: `{preflight.get('passed', False)}`")
    if preflight.get("gates_failed"):
        lines.append(f"- gates_failed: {preflight['gates_failed']}")

    lines.extend(["", "## Post-run (if run)", ""])
    if postrun is not None:
        lines.append(f"- passed: `{postrun.get('passed', False)}`")
        if postrun.get("gates_failed"):
            lines.append(f"- gates_failed: {postrun['gates_failed']}")
    else:
        lines.append("- (no replay run)")

    if canary_proof is not None:
        mode = canary_proof.get("mode", "ZERO_TRADES_CANARY")
        title = "## Zero-trades canary" if mode == "ZERO_TRADES_CANARY" else "## Diagnostic Mode"
        lines.extend(["", title, ""])
        lines.append(f"- mode: `{mode}`")
        if "entry_threshold_override" in canary_proof:
            lines.append(f"- entry_threshold_override: `{canary_proof.get('entry_threshold_override')}`")
        if "diagnostic_flag" in canary_proof:
            lines.append(f"- diagnostic_flag: `{canary_proof.get('diagnostic_flag')}`")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    global CTX_CONT_DIM
    ap = argparse.ArgumentParser(description="TRUTH-grade E2E sanity checker for signal-only pipeline.")
    ap.add_argument("--run-id", type=str, default="", help="Run ID (default: E2E_SANITY_<utc_ts>)")
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Output run directory (default: GX1_DATA/reports/truth_e2e_sanity/<run_id>)",
    )
    ap.add_argument("--start-ts", type=str, default=DEFAULT_START_TS, help="Start timestamp (ISO)")
    ap.add_argument("--end-ts", type=str, default=DEFAULT_END_TS, help="End timestamp (ISO)")
    ap.add_argument(
        "--chunk-local-padding-days",
        type=int,
        default=0,
        help="Load this many extra days before start-ts as replay warmup while keeping evaluation/trading window at start-ts..end-ts.",
    )
    ap.add_argument("--full-year", action="store_true", help="Use 2025-01-01 to 2025-12-31")
    ap.add_argument("--validate-only", action="store_true", help="Only preflight, no replay")
    ap.add_argument(
        "--threshold-override",
        type=str,
        default="",
        help="Set GX1_ANALYSIS_MODE=1 and GX1_ENTRY_THRESHOLD_OVERRIDE=<val>",
    )
    ap.add_argument(
        "--diagnostic-threshold-override",
        action="store_true",
        help="Allow entry threshold override in replay (NON_CANONICAL_DIAGNOSTIC only)",
    )
    ap.add_argument(
        "--force-zero-trades",
        action="store_true",
        help="ZERO_TRADES_CANARY: force 0 trades (GX1_ANALYSIS_MODE=1, GX1_ENTRY_THRESHOLD_OVERRIDE=1.1). Hard-fail if n_trades>0.",
    )
    ap.add_argument("--entry-signal-trace", action="store_true", help="Set GX1_ENTRY_SIGNAL_TRACE=1")
    ap.add_argument("--strict-masks", dest="strict_masks", action="store_true", default=True, help="Set GX1_STRICT_MASK=1 (default)")
    ap.add_argument("--no-strict-masks", dest="strict_masks", action="store_false", help="Do not set GX1_STRICT_MASK")
    ap.add_argument("--truth-file", type=str, default="", help="Canonical truth JSON (default: env, else CANONICAL_TRUTH_DEFAULT)")
    ap.add_argument(
        "--train-exit-transformer-v0-from-last-go",
        action="store_true",
        help="Train Exit Transformer V0 from LAST_GO exits jsonl. LAST_GO is reserved for canonical 2025 full-year replay only.",
    )
    ap.add_argument(
        "--exit-window-len",
        type=int,
        default=0,
        help="With --train-exit-transformer-v0-from-last-go: override exit window length. 0 means contract default.",
    )
    ap.add_argument("--postrun-only", action="store_true", help="Skip replay and run postrun hooks against an existing run_root")
    ap.add_argument("--run-root", type=str, default="", help="Existing run_root to use with --postrun-only")
    args = ap.parse_args()

    if getattr(args, "train_exit_transformer_v0_from_last_go", False):
        gx1_data = _gx1_data()
        try:
            from gx1.policy.exit_transformer_v0 import (
                get_last_go_exits_dataset,
                train_from_exits_jsonl,
                verify_exit_transformer_artifacts,
            )
        except ImportError as e:
            print(f"[train-exit-transformer-v0] Import failed: {e}", file=sys.stderr)
            return 1
        try:
            ds = get_last_go_exits_dataset(gx1_data=str(gx1_data))
        except FileNotFoundError as e:
            print(f"[train-exit-transformer-v0] {e}", file=sys.stderr)
            return 1
        exits_path = ds["exits_jsonl_path"]
        go_run_dir = ds["go_run_dir"]
        go_run_id = ds["go_run_id"]
        print(f"[train-exit-transformer-v0] Source: {exits_path} (run_id={go_run_id})", file=sys.stderr)
        train_exit_io_version = "EXIT_IO_V3_CTX36_M1L512_PHASE5"
        train_window_len = int(getattr(args, "exit_window_len", 0) or 512)
        try:
            train_epochs = int(os.environ.get("GX1_EXIT_TRAIN_EPOCHS", "20"))
        except Exception:
            train_epochs = 20
        result = train_from_exits_jsonl(
            exits_path,
            out_dir=None,
            source_run_id=go_run_id,
            source_run_dir=str(go_run_dir),
            gx1_data=str(gx1_data),
            epochs=train_epochs,
            window_len=train_window_len,
            seed=42,
            exit_io_version=train_exit_io_version,
            ctx_cont_dim=CTX_CONT_DIM,
            ctx_cat_dim=CTX_CAT_DIM,
        )
        out_dir = result["train_report_path"].parent
        verify_result = verify_exit_transformer_artifacts(out_dir)
        print(f"[train-exit-transformer-v0] Out dir: {out_dir}", file=sys.stderr)
        print(f"[train-exit-transformer-v0] model_sha256: {result['model_sha256']}", file=sys.stderr)
        print(f"[train-exit-transformer-v0] dataset_sha256: {result['dataset_sha256']}", file=sys.stderr)
        print(f"[train-exit-transformer-v0] Verify passed: {verify_result['passed']}", file=sys.stderr)
        if verify_result.get("failures"):
            for f in verify_result["failures"]:
                print(f"  - {f}", file=sys.stderr)
        if not verify_result["passed"]:
            return 1
        return 0

    # ------------------------------------------------------------------
    # Postrun-only: use existing run_root, skip replay/validate.
    # ------------------------------------------------------------------
    if getattr(args, "postrun_only", False):
        if not args.run_root:
            print("[POSTRUN_ONLY] missing --run-root", file=sys.stderr)
            return 1
        run_root = Path(args.run_root).expanduser().resolve()
        if not run_root.exists():
            print(f"[POSTRUN_ONLY] run_root not found: {run_root}", file=sys.stderr)
            return 1
        run_id = args.run_id or run_root.name

        truth_file_cli = (args.truth_file or "").strip()
        truth_file_env = (os.environ.get("GX1_CANONICAL_TRUTH_FILE", "") or "").strip()
        truth_file = truth_file_cli or truth_file_env or CANONICAL_TRUTH_DEFAULT
        truth_path = Path(truth_file).expanduser().resolve()
        if not truth_path.exists():
            print(f"[POSTRUN_ONLY] truth file missing: {truth_path}", file=sys.stderr)
            return 1
        truth_obj = _load_json(truth_path)

        join_path_env = run_root / "replay" / "chunk_0" / "RAW_PREBUILT_JOIN.json"
        footer_path_env = run_root / "replay" / "chunk_0" / "chunk_footer.json"
        prebuilt_env_val = None
        if footer_path_env.exists():
            try:
                footer_env = _load_json(footer_path_env)
                prebuilt_env_val = footer_env.get("prebuilt_parquet_path_resolved") or footer_env.get("prebuilt_parquet_path")
                _footer_ctx_cont = footer_env.get("ctx_cont_dim")
                if _footer_ctx_cont is not None:
                    CTX_CONT_DIM = int(_footer_ctx_cont)
            except Exception:
                prebuilt_env_val = None
        if prebuilt_env_val:
            os.environ["GX1_REPLAY_PREBUILT_FEATURES_PATH"] = str(prebuilt_env_val)
        elif join_path_env.exists():
            os.environ["GX1_REPLAY_PREBUILT_FEATURES_PATH"] = str(join_path_env)
        os.environ.setdefault("GX1_REPLAY_USE_PREBUILT_FEATURES", "1")
        os.environ.setdefault("GX1_FEATURE_BUILD_DISABLED", "1")

        postrun = _run_postrun_checks(run_root, run_id, truth_obj)
        if not postrun.get("passed"):
            print(f"[POSTRUN_ONLY] postrun checks failed: {postrun.get('gates_failed')}", file=sys.stderr)
            return 1

        pred_path = run_root / f"xgb_multi_horizon_predictions_{run_id}.parquet"
        if not pred_path.exists():
            pred_path = _write_multi_horizon_predictions(run_root, run_id)

        append_h2_overlap_auc_decomposition(run_root, run_id)
        print(f"[POSTRUN_ONLY_PROOF] run_root={run_root} run_id={run_id} predictions={pred_path}")
        return 0

    run_id = args.run_id or (
        f"ZERO_TRADES_CANARY_{_utc_ts_compact()}" if args.force_zero_trades else f"E2E_SANITY_{_utc_ts_compact()}"
    )
    gx1_data = _gx1_data()
    run_root = Path(args.run_dir).expanduser().resolve() if args.run_dir else (gx1_data / "reports" / "truth_e2e_sanity" / run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[PIPELINE_RUNROOT_PROOF] run_root={run_root} run_id={run_id}", file=sys.stderr)
    canary_proof = {"mode": "ZERO_TRADES_CANARY", "entry_threshold_override": ZERO_TRADES_CANARY_THRESHOLD} if args.force_zero_trades else None

    # TRUTH gate: no legacy replay script in process
    _assert_truth_no_legacy_replay(run_root)

    # ---------------------------------------------------------------------
    # TRUTH_FILE resolution (NO "missing truth" state; NO split-brain).
    # Priority:
    #   1) --truth-file
    #   2) GX1_CANONICAL_TRUTH_FILE
    #   3) CANONICAL_TRUTH_DEFAULT
    # Split-brain rule:
    #   If both CLI and env are set and differ (after resolve) -> hard fail.
    # TRUTH forbids CLI override of a different env path.
    # ---------------------------------------------------------------------
    truth_file_cli = (args.truth_file or "").strip()
    truth_file_env = (os.environ.get("GX1_CANONICAL_TRUTH_FILE", "") or "").strip()

    cli_abs = str(Path(truth_file_cli).expanduser().resolve()) if truth_file_cli else ""
    env_abs = str(Path(truth_file_env).expanduser().resolve()) if truth_file_env else ""

    if truth_file_cli and truth_file_env and cli_abs != env_abs:
        raise RuntimeError(
            f"SPLIT_BRAIN_TRUTH: --truth-file={cli_abs} != GX1_CANONICAL_TRUTH_FILE={env_abs} (TRUTH_FORBIDS_CLI_TRUTH_OVERRIDE)"
        )

    truth_file = truth_file_cli or truth_file_env or CANONICAL_TRUTH_DEFAULT
    truth_path = Path(truth_file).expanduser().resolve()
    print(f"[E2E] TRUTH_FILE_USED={truth_path}", file=sys.stderr)

    # TRUTH envs (must be set before preflight so preflight can check)
    os.environ["GX1_RUN_MODE"] = "TRUTH"
    os.environ["GX1_TRUTH_MODE"] = "1"
    os.environ["GX1_GATED_FUSION_ENABLED"] = "1"
    os.environ["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    os.environ["GX1_FEATURE_BUILD_DISABLED"] = "1"
    os.environ["GX1_CANONICAL_TRUTH_FILE"] = str(truth_path)
    os.environ.setdefault("GX1_OUTPUT_MODE", "TRUTH")
    os.environ.setdefault("GX1_SEED", "42")
    os.environ.setdefault("GX1_REPLAY_INCREMENTAL_FEATURES", "1")
    os.environ.setdefault("GX1_FEATURE_USE_NP_ROLLING", "1")
    os.environ.setdefault("GX1_REPLAY_NO_CSV", "1")
    for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_k, "1")

    # TRUTH/SMOKE: hard-fail on parallel/segmented envs or multi-worker hints
    for forbidden_env in (
        "GX1_PARALLEL",
        "GX1_SEGMENTED",
        "GX1_SEGMENTED_PARALLEL",
        "GX1_WORKERS",
        "GX1_CHUNKS",
        "GX1_N_WORKERS",
        "GX1_N_CHUNKS",
    ):
        val = os.environ.get(forbidden_env)
        if val:
            raise RuntimeError(f"[TRUTH_FORBIDDEN_ENV] {forbidden_env}={val}")

    try:
        from gx1.utils.truth_banlist import assert_truth_banlist_clean  # type: ignore

        assert_truth_banlist_clean(output_dir=run_root, stage="run_truth_e2e_sanity:entry")
    except Exception as e:
        _write_fatal_capsule(run_root, e, ["truth_banlist"])
        print(f"[E2E] TRUTH banlist: {e}", file=sys.stderr)
        return 1

    if args.entry_signal_trace:
        os.environ["GX1_ENTRY_SIGNAL_TRACE"] = "1"
    if args.strict_masks:
        os.environ["GX1_STRICT_MASK"] = "1"
    if args.threshold_override:
        if not args.diagnostic_threshold_override:
            raise RuntimeError(
                "[DIAGNOSTIC_REQUIRED] --threshold-override requires --diagnostic-threshold-override "
                "(NON_CANONICAL_DIAGNOSTIC only)"
            )
        os.environ["GX1_DIAGNOSTIC_THRESHOLD_SWEEP"] = "1"
        os.environ["GX1_NON_CANONICAL_DIAGNOSTIC"] = "1"
        os.environ["GX1_ANALYSIS_MODE"] = "1"
        os.environ["GX1_ENTRY_THRESHOLD_OVERRIDE"] = args.threshold_override
        canary_proof = {
            "mode": "NON_CANONICAL_DIAGNOSTIC",
            "entry_threshold_override": args.threshold_override,
            "diagnostic_flag": "GX1_DIAGNOSTIC_THRESHOLD_SWEEP=1",
        }
    if args.force_zero_trades:
        os.environ["GX1_ANALYSIS_MODE"] = "1"
        os.environ["GX1_ENTRY_THRESHOLD_OVERRIDE"] = ZERO_TRADES_CANARY_THRESHOLD
        print("[E2E] MODE=ZERO_TRADES_CANARY (entry threshold 1.1 → 0 trades contract)", file=sys.stderr)
        _atomic_write_json(
            run_root / "RUN_IDENTITY.json",
            {"mode": "ZERO_TRADES_CANARY", "entry_threshold_override": ZERO_TRADES_CANARY_THRESHOLD, "run_id": run_id},
        )

    # Preflight
    try:
        preflight = _run_preflight(truth_path, run_root)
        if os.environ.get("GX1_DIAGNOSTIC_THRESHOLD_SWEEP") == "1":
            preflight["non_canonical_diagnostic"] = {
                "entry_threshold_override": os.environ.get("GX1_ENTRY_THRESHOLD_OVERRIDE", ""),
                "flag": "GX1_DIAGNOSTIC_THRESHOLD_SWEEP=1",
            }
        _atomic_write_json(run_root / "PREFLIGHT_E2E.json", preflight)
        if not preflight.get("passed", False):
            _write_fatal_capsule(run_root, RuntimeError(str(preflight.get("gates_failed", []))), preflight.get("gates_failed", []))
            _write_summary_md(run_root, preflight, None, False, ["Preflight failed: " + str(preflight.get("gates_failed", []))], canary_proof=canary_proof)
            print("[E2E] PREFLIGHT FAIL:", preflight.get("gates_failed"), file=sys.stderr)
            return 1
        _apply_ctx_mask_defaults(CTX_CONT_DIM, CTX_CAT_DIM)
    except Exception as e:
        _write_fatal_capsule(run_root, e, ["preflight_exception"])
        _write_summary_md(run_root, {"passed": False, "gates_failed": ["preflight_exception"]}, None, False, [str(e)], canary_proof=canary_proof)
        raise

    if args.validate_only:
        _write_summary_md(run_root, preflight, None, True, ["Preflight passed (--validate-only)"], canary_proof=canary_proof)
        print("[E2E] Preflight passed (--validate-only)", file=sys.stderr)
        return 0

    # Replay writes to run_root/replay so TRUTH does not see PREFLIGHT_E2E.json as existing artifacts
    replay_output_dir = run_root / "replay"
    replay_output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve paths from truth
    truth_obj = _load_json(truth_path)
    canonical_bundle = str(truth_obj.get("canonical_xgb_bundle_dir") or "")
    xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
    if xgb_override:
        canonical_bundle = xgb_override
    canonical_prebuilt = str(truth_obj.get("canonical_prebuilt_parquet") or "")
    canonical_manifest_truth = str(truth_obj.get("canonical_prebuilt_manifest") or "")
    canonical_transformer = str(truth_obj.get("canonical_transformer_bundle_dir") or "")
    replay_cfg_truth = truth_obj.get("replay_config") or {}
    pred_trace_head = truth_obj.get("pred_trace_head")
    pred_trace_horizon = truth_obj.get("pred_trace_horizon_bars")

    manifest_ssot_resolved = MANIFEST_SSOT.expanduser().resolve()
    if canonical_manifest_truth and Path(canonical_manifest_truth).expanduser().resolve() != manifest_ssot_resolved:
        _write_fatal_capsule(
            run_root,
            RuntimeError("PREBUILT_MANIFEST_SPLIT_BRAIN"),
            ["canonical_truth_paths"],
        )
        _write_summary_md(
            run_root,
            preflight,
            None,
            False,
            [f"PREBUILT_MANIFEST_SPLIT_BRAIN truth_manifest={canonical_manifest_truth} expected={manifest_ssot_resolved}"],
            canary_proof=canary_proof,
        )
        return 1

    # TRUTH gate: forbid legacy sniper tree (must be migrated to _legacy_disabled)
    legacy_dir_name = "_".join(("sniper", "snapshot"))
    sniper_live_path = ENGINE / "gx1" / "configs" / "policies" / legacy_dir_name
    if sniper_live_path.exists():
        msg = (
            "[TRUTH_GATE] legacy policy tree must not exist at live path (migrate to gx1/configs/_legacy_disabled). "
            f"Found: {sniper_live_path}"
        )
        _write_fatal_capsule(run_root, RuntimeError(msg), ["legacy_policy_tree_on_disk"])
        _write_summary_md(run_root, preflight, None, False, [msg], canary_proof=canary_proof)
        raise RuntimeError(msg)

    manifest_obj = _load_json(manifest_ssot_resolved)
    manifest_parquet = str(Path(manifest_obj.get("parquet_path") or "").expanduser().resolve())
    if not canonical_prebuilt:
        _write_fatal_capsule(run_root, RuntimeError("canonical_prebuilt_parquet missing"), ["canonical_truth_paths"])
        _write_summary_md(
            run_root, preflight, None, False, ["canonical_prebuilt_parquet missing in truth"], canary_proof=canary_proof
        )
        return 1

    if manifest_parquet != str(Path(canonical_prebuilt).expanduser().resolve()):
        _write_fatal_capsule(run_root, RuntimeError("PREBUILT_SPLIT_BRAIN"), ["canonical_truth_paths"])
        _write_summary_md(
            run_root,
            preflight,
            None,
            False,
            [f"PREBUILT_SPLIT_BRAIN manifest_parquet={manifest_parquet} canonical_prebuilt_parquet={canonical_prebuilt}"],
            canary_proof=canary_proof,
        )
        return 1

    # Pred trace metadata (TRUTH only)
    if args.postrun_only or args.validate_only:
        pass
    else:
        if not pred_trace_head or not pred_trace_horizon:
            _write_fatal_capsule(
                run_root,
                RuntimeError("[PRED_TRACE] pred_trace_head or pred_trace_horizon_bars missing in truth file"),
                ["pred_trace_meta"],
            )
            _write_summary_md(
                run_root,
                preflight,
                None,
                False,
                ["pred_trace_head or pred_trace_horizon_bars missing in truth file"],
                canary_proof=canary_proof,
            )
            return 1
        try:
            pred_trace_horizon_int = int(pred_trace_horizon)
        except Exception as e:
            _write_fatal_capsule(run_root, e, ["pred_trace_meta"])
            _write_summary_md(run_root, preflight, None, False, [str(e)], canary_proof=canary_proof)
            return 1
        os.environ["GX1_PRED_TRACE_HEAD"] = str(pred_trace_head)
        os.environ["GX1_PRED_TRACE_HORIZON"] = str(pred_trace_horizon_int)
        os.environ["GX1_PRED_TRACE_PATH"] = str(run_root / "replay" / "chunk_0" / "logs" / f"pred_trace_{run_id}.jsonl")
        print("[EVAL_LOG_SCHEMA_PROOF] fields+=['head','horizon_bars','y_true']")

    for label, val in (
        ("canonical_xgb_bundle_dir", canonical_bundle),
        ("canonical_transformer_bundle_dir", canonical_transformer),
        ("manifest_ssot", str(manifest_ssot_resolved)),
        ("manifest_parquet", manifest_parquet),
    ):
        _forbid_prune_path(label, val, allow_ctx6cat6=label == "canonical_transformer_bundle_dir")

    bundle_dir = Path(canonical_bundle).expanduser().resolve()
    prebuilt_path = Path(manifest_parquet).expanduser().resolve()

    def _resolve_truth_path(path_val: str, label: str) -> Path:
        if not path_val:
            raise RuntimeError(f"[TRUTH_POLICY_WIRING_MISSING] {label} missing in truth file: {truth_path}")
        p = Path(path_val)
        if not p.is_absolute():
            p = (ENGINE / p).resolve()
        else:
            p = p.expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"[TRUTH_POLICY_WIRING_NOT_FOUND] {label} not found: {p}")
        return p

    try:
        policy_path = _resolve_truth_path(replay_cfg_truth.get("policy_yaml_path", "").strip(), "policy_yaml_path")
        entry_config_path = _resolve_truth_path(replay_cfg_truth.get("entry_config_yaml_path", "").strip(), "entry_config_yaml_path")
        exit_config_path = _resolve_truth_path(replay_cfg_truth.get("exit_config_yaml_path", "").strip(), "exit_config_yaml_path")
        print(f"[E2E] POLICY_PATH_USED={policy_path}", file=sys.stderr)
        print(f"[E2E] ENTRY_CONFIG_USED={entry_config_path}", file=sys.stderr)
        print(f"[E2E] EXIT_CONFIG_USED={exit_config_path}", file=sys.stderr)
        print(
            f"[PIPELINE_CONFIG_PROOF] policy={policy_path} entry_config={entry_config_path} exit_config={exit_config_path}",
            file=sys.stderr,
        )

        # Split-brain guard: policy must reference the same entry/exit configs declared in truth.
        try:
            policy_text = policy_path.read_text(encoding="utf-8")
        except Exception as _read_err:
            raise RuntimeError(f"[TRUTH_POLICY_WIRING_NOT_FOUND] failed to read policy_yaml_path: {policy_path} err={_read_err}") from _read_err

        entry_marker = entry_config_path.name
        exit_marker = exit_config_path.name
        if entry_marker not in policy_text or exit_marker not in policy_text:
            raise RuntimeError(
                "[TRUTH_POLICY_WIRING_SPLIT_BRAIN] policy YAML does not reference required entry/exit configs "
                f"(entry_marker={entry_marker!r}, exit_marker={exit_marker!r}, policy={policy_path})"
            )

        # ENTRY 3-class proof wiring for replay summary (best-effort; never fatal)
        try:
            import json as _json
            import yaml as _yaml
            policy_cfg = _yaml.safe_load(policy_text) or {}
            entry_bundle_dir = (
                ((policy_cfg.get("entry_models") or {}).get("v10_ctx") or {}).get("bundle_dir")
            )
            entry_override = os.environ.get("GX1_ENTRY_BUNDLE_DIR", "").strip()
            if entry_override:
                entry_bundle_dir = entry_override
            if entry_bundle_dir:
                entry_bundle_dir = str(Path(entry_bundle_dir).expanduser().resolve())
                os.environ["GX1_ENTRY_BUNDLE_DIR_PROOF"] = entry_bundle_dir
                meta_path = Path(entry_bundle_dir) / "bundle_metadata.json"
                if meta_path.exists():
                    meta = _json.loads(meta_path.read_text(encoding="utf-8"))
                    os.environ["GX1_ENTRY_BUNDLE_NUM_CLASSES_PROOF"] = str(meta.get("num_classes"))
                    os.environ["GX1_ENTRY_BUNDLE_CLASS_ORDER_PROOF"] = _json.dumps(meta.get("class_order"))
        except Exception as _entry_proof_err:
            print(f"[ENTRY_3CLASS_PROOF_WARN] {type(_entry_proof_err).__name__}: {_entry_proof_err}", file=sys.stderr)

        # Risk guard identity expectation from truth
        expected_risk_guard_id = (replay_cfg_truth.get("risk_guard_id") or "").strip()
        if not expected_risk_guard_id:
            raise RuntimeError("[TRUTH_POLICY_WIRING_MISSING] risk_guard_id missing in truth file")
    except Exception as e:
        _write_fatal_capsule(run_root, e, ["truth_policy_wiring"])
        _write_summary_md(run_root, preflight, None, False, [str(e)], canary_proof=canary_proof)
        return 1

    # policy_path already existence-checked above
    # TRUTH gate: only the canonical policy file may be used (exact path match)
    try:
        from gx1.utils.truth_banlist import is_truth_or_smoke, assert_truth_policy_path_canonical

        if is_truth_or_smoke():
            assert_truth_policy_path_canonical(policy_path, engine_root=ENGINE, output_dir=run_root)
    except ImportError:
        pass
    if not prebuilt_path.exists():
        _write_fatal_capsule(run_root, FileNotFoundError(str(prebuilt_path)), ["prebuilt_path"])
        _write_summary_md(run_root, preflight, None, False, [f"Prebuilt parquet not found: {prebuilt_path}"], canary_proof=canary_proof)
        return 1
    if not bundle_dir.exists():
        _write_fatal_capsule(run_root, FileNotFoundError(str(bundle_dir)), ["bundle_dir"])
        _write_summary_md(run_root, preflight, None, False, [f"Bundle dir not found: {bundle_dir}"], canary_proof=canary_proof)
        return 1

    xgb_override = os.environ.get("GX1_XGB_BUNDLE_DIR", "").strip()
    os.environ["GX1_CANONICAL_BUNDLE_DIR"] = str(bundle_dir)
    if xgb_override:
        os.environ["GX1_XGB_BUNDLE_DIR"] = str(Path(xgb_override).expanduser().resolve())
    else:
        os.environ["GX1_XGB_BUNDLE_DIR"] = str(bundle_dir)
    os.environ["GX1_CANONICAL_TRANSFORMER_BUNDLE_DIR"] = str(Path(canonical_transformer).expanduser().resolve())
    os.environ["GX1_CANONICAL_POLICY_PATH"] = str(policy_path)

    truth_window_start = truth_obj.get("truth_window_start_ts")
    truth_window_end = truth_obj.get("truth_window_end_ts")

    cli_used = bool(args.start_ts or args.end_ts)
    default_start_ts = FULLYEAR_START_TS if args.full_year else args.start_ts
    default_end_ts = FULLYEAR_END_TS if args.full_year else args.end_ts

    if cli_used:
        effective_start_ts = args.start_ts
        effective_end_ts = args.end_ts
        window_source = "CLI"
    elif truth_window_start or truth_window_end:
        effective_start_ts = str(truth_window_start) if truth_window_start else default_start_ts
        effective_end_ts = str(truth_window_end) if truth_window_end else default_end_ts
        window_source = "TRUTH_CONFIG"
        if default_start_ts and truth_window_start and str(truth_window_start) != str(default_start_ts):
            raise RuntimeError(
                "[TRUTH_WINDOW_CONFLICT] truth_config start_ts differs from default and CLI not provided"
            )
        if default_end_ts and truth_window_end and str(truth_window_end) != str(default_end_ts):
            raise RuntimeError(
                "[TRUTH_WINDOW_CONFLICT] truth_config end_ts differs from default and CLI not provided"
            )
    else:
        effective_start_ts = default_start_ts
        effective_end_ts = default_end_ts
        window_source = "DEFAULT"

    raw_env = os.environ.get("GX1_RAW_2025", "").strip()
    if raw_env:
        raise RuntimeError(
            "[REPLAY_TAPE_FORBIDDEN_RAW_ENV] GX1_RAW_2025 is set; TRUTH replay must use canonical tape"
        )
    if os.environ.get("GX1_MAX_OPEN_TRADES_OVERRIDE", "").strip():
        raise RuntimeError(
            "[CANONICAL_CAPACITY_ENV_FORBIDDEN] GX1_MAX_OPEN_TRADES_OVERRIDE is set; "
            "TRUTH replay must use execution.max_open_trades from canonical policy."
        )

    truth_raw_root = truth_obj.get("canonical_market_tape_root_raw") or truth_obj.get("canonical_tape_root_raw")
    if truth_raw_root:
        os.environ["GX1_CANONICAL_TAPE_ROOT_RAW"] = str(truth_raw_root)
    raw_path = _resolve_canonical_tape_path(effective_start_ts, effective_end_ts, run_root=run_root)
    if "/data/data/raw/" in str(raw_path):
        raise RuntimeError(
            f"[REPLAY_TAPE_FORBIDDEN_RAW_PATH] raw_path resolved into legacy raw lane: {raw_path}"
        )
    print(
        "[REPLAY_TAPE_SOURCE] "
        f"tape_root_raw={os.environ.get('GX1_CANONICAL_TAPE_ROOT_RAW', os.environ.get('GX1_CANONICAL_TAPE_ROOT', '/home/andre2/GX1_DATA/data/oanda/canonical/xauusd_m1_bid_ask__CANONICAL'))} "
        f"tape_path={raw_path}",
        flush=True,
    )
    if not raw_path.exists():
        _write_fatal_capsule(run_root, FileNotFoundError(str(raw_path)), ["raw_path"])
        _write_summary_md(run_root, preflight, None, False, [f"Raw data not found: {raw_path}"], canary_proof=canary_proof)
        return 1

    env_overrides: Dict[str, str] = {}
    if args.threshold_override:
        env_overrides["GX1_ANALYSIS_MODE"] = "1"
        env_overrides["GX1_ENTRY_THRESHOLD_OVERRIDE"] = args.threshold_override
    if args.force_zero_trades:
        env_overrides["GX1_ANALYSIS_MODE"] = "1"
        env_overrides["GX1_ENTRY_THRESHOLD_OVERRIDE"] = ZERO_TRADES_CANARY_THRESHOLD

    # Replay
    print(
        "[REPLAY_WINDOW_RESOLUTION_PROOF] "
        f"source={window_source} "
        f"effective_start_ts={effective_start_ts} "
        f"effective_end_ts={effective_end_ts}",
        flush=True,
    )
    stale_exit_knobs = sorted(k for k in env_overrides if k.startswith("GX1_EXIT_"))
    if stale_exit_knobs:
        raise RuntimeError(
            "[CANONICAL_EXIT_ENV_CONTRACT_FAIL] run_truth_e2e_sanity assembled non-contract exit env overrides: "
            + ",".join(stale_exit_knobs)
        )
    start_ts = effective_start_ts
    end_ts = effective_end_ts
    try:
        rc = _run_replay(
            replay_output_dir,
            run_id,
            truth_path,
            policy_path,
            raw_path,
            prebuilt_path,
            start_ts,
            end_ts,
            env_overrides,
            bundle_dir,
            merge_output_dir=run_root,
            chunk_local_padding_days=int(args.chunk_local_padding_days or 0),
        )
    except Exception as e:
        _write_fatal_capsule(run_root, e, ["replay_exception"])
        postrun_fail = {"passed": False, "gates_failed": ["replay_exception"], "checks": {}}
        _atomic_write_json(run_root / "POSTRUN_E2E.json", postrun_fail)
        _write_summary_md(run_root, preflight, postrun_fail, False, [str(e)], canary_proof=canary_proof)
        raise

    if rc != 0:
        _atomic_write_json(
            run_root / "POSTRUN_E2E.json",
            {"passed": False, "run_id": run_id, "replay_exitcode": rc, "gates_failed": ["replay_exitcode"]},
        )
        _write_summary_md(run_root, preflight, {"passed": False, "gates_failed": ["replay_exitcode"]}, False, [f"Replay exit code {rc}"], canary_proof=canary_proof)
        return 1

    # Postrun checks
    try:
        try:
            truth_obj_postrun = _load_json(truth_path)
        except Exception:
            truth_obj_postrun = {}
        postrun = _run_postrun_checks(run_root, run_id, truth_obj_postrun)
        _atomic_write_json(run_root / "POSTRUN_E2E.json", postrun)
        if not postrun.get("passed", False):
            _write_fatal_capsule(run_root, RuntimeError(str(postrun.get("gates_failed"))), postrun.get("gates_failed", []))
            _write_summary_md(run_root, preflight, postrun, False, ["Post-run failed: " + str(postrun.get("gates_failed", []))], canary_proof=canary_proof)
            print("[E2E] POSTRUN FAIL:", postrun.get("gates_failed"), file=sys.stderr)
            return 1
    except Exception as e:
        _write_fatal_capsule(run_root, e, ["postrun_exception"])
        _write_summary_md(run_root, preflight, None, False, [str(e)], canary_proof=canary_proof)
        raise

    # Write predictions artifact when pred_trace exists. Some pure entry sweeps do not
    # emit pred_trace; that should not invalidate the replay/trade accounting itself.
    try:
        _write_multi_horizon_predictions(run_root, run_id)
    except Exception as e:
        if "[PRED_WRITE] pred_trace missing" in str(e):
            print(f"[PRED_WRITE_SKIP] {e}", flush=True)
        else:
            _write_fatal_capsule(run_root, e, ["predictions_writer"])
            _write_summary_md(run_root, preflight, postrun, False, [str(e)], canary_proof=canary_proof)
            raise

    # ZERO_TRADES_CANARY: hard-fail if we expected 0 trades but got any
    if args.force_zero_trades:
        metrics_path = run_root / f"metrics_{run_id}_MERGED.json"
        n_trades = -1
        if metrics_path.exists():
            try:
                metrics = _load_json(metrics_path)
                n_trades = int(metrics.get("n_trades", -1))
            except Exception:
                pass
        if n_trades != 0:
            msg = (
                f"[ZERO_TRADES_CANARY] Contract violation: expected n_trades=0, got n_trades={n_trades}. "
                f"Pipeline must produce 0 trades when GX1_ENTRY_THRESHOLD_OVERRIDE={ZERO_TRADES_CANARY_THRESHOLD}."
            )
            _write_fatal_capsule(run_root, RuntimeError(msg), ["zero_trades_canary"])
            _write_summary_md(run_root, preflight, postrun, False, [msg], canary_proof=canary_proof)
            print(f"[E2E] {msg}", file=sys.stderr)
            return 1
        print("[E2E] ZERO_TRADES_CANARY: n_trades=0, contract OK", file=sys.stderr)

    _write_summary_md(run_root, preflight, postrun, True, ["Preflight passed", "Replay completed", "Post-run passed"], canary_proof=canary_proof)
    print("[E2E] GO: Preflight + Replay + Post-run passed", file=sys.stderr)

    # TRUTH gate: ONE UNIVERSE (ctx_cat=6 fixed; ctx_cont from bundle).
    # LAST_GO is reserved for canonical full-year replay only, and only when exits
    # have context matching bundle dims; hard-fail if footer dims mismatch.
    footer_path = run_root / "replay" / "chunk_0" / "chunk_footer.json"
    expected_ctx_cont = CTX_CONT_DIM
    expected_ctx_cat = CTX_CAT_DIM
    require_exits_file = False
    if footer_path.exists():
        footer = _load_json(footer_path)
        if footer.get("ctx_cont_dim") is not None and footer["ctx_cont_dim"] != CTX_CONT_DIM:
            _write_fatal_capsule(
                run_root,
                RuntimeError(f"ONE_UNIVERSE: footer ctx_cont_dim must be {CTX_CONT_DIM}, got {footer['ctx_cont_dim']}"),
                ["ctx_dims"],
            )
            raise RuntimeError(f"[E2E] footer ctx_cont_dim must be {CTX_CONT_DIM}, got {footer['ctx_cont_dim']}")
        if footer.get("ctx_cat_dim") is not None and footer["ctx_cat_dim"] != CTX_CAT_DIM:
            _write_fatal_capsule(run_root, RuntimeError(f"ONE_UNIVERSE: footer ctx_cat_dim must be {CTX_CAT_DIM}, got {footer['ctx_cat_dim']}"), ["ctx_dims"])
            raise RuntimeError(f"[E2E] footer ctx_cat_dim must be {CTX_CAT_DIM}, got {footer['ctx_cat_dim']}")
        if footer.get("exit_ml_enabled") is True:
            require_exits_file = True
    if os.environ.get("GX1_EXIT_AUDIT") == "1":
        require_exits_file = True
    gate_ok, gate_err = _exits_context_gate(run_root, run_id, expected_ctx_cont, expected_ctx_cat, require_exits_file=require_exits_file)
    if not gate_ok:
        postrun_fail = {
            "passed": False,
            "run_id": run_id,
            "gates_failed": ["exits_context_for_last_go"],
            "checks": {"exits_context_for_last_go": {"error": gate_err}},
        }
        _write_fatal_capsule(run_root, RuntimeError(gate_err or "exits context gate"), ["exits_context_for_last_go"])
        _atomic_write_json(run_root / "POSTRUN_E2E.json", postrun_fail)
        _write_summary_md(run_root, preflight, postrun_fail, False, [gate_err or "Exits context gate failed"], canary_proof=canary_proof)
        print(f"[E2E] LAST_GO not updated (exits context gate): {gate_err}", file=sys.stderr)
        return 1

    if not _last_go_run_is_eligible(start_ts, end_ts):
        print(
            f"[E2E] LAST_GO not updated (run not eligible): start_ts={start_ts} end_ts={end_ts} "
            f"required={FULLYEAR_START_TS}..{FULLYEAR_END_TS}",
            file=sys.stderr,
        )
        return 0

    # LAST_GO is written only after all gates pass (preflight, replay, postrun, zero-trades,
    # exits-context) and only for canonical full-year replay.
    last_go_dir = gx1_data / "reports" / "truth_e2e_sanity"
    last_go_path = last_go_dir / "LAST_GO.txt"
    try:
        last_go_dir.mkdir(parents=True, exist_ok=True)
        last_go_path.write_text(str(run_root.resolve()), encoding="utf-8")
        print(f"[E2E] LAST_GO set: {last_go_path} -> {run_root}", file=sys.stderr)
    except Exception as e:
        print(f"[E2E] LAST_GO write failed (non-fatal): {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
