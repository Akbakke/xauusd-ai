#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FULLYEAR 2025 TRUTH/PREBUILT proof runner for GX1 (XGB → Transformer + exits).

This script is intentionally minimal and replay-safe:
- Forces TRUTH/PREBUILT env gates (fail-fast on any contract violation)
- Runs gated-parallel replay over 2025 using canonical bundle + policy + prebuilt parquet
- Produces a proof pack with:
  - TRUTH_FULLYEAR_2025_SUMMARY.json
  - TRUTH_FULLYEAR_2025_SUMMARY.md
  - (references to RUN_COMPLETED.json, EXIT_COVERAGE_SUMMARY.json, trade journals, telemetry)

No trading semantics are changed; this is a runner + post-run audit only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ProofPaths:
    run_root: Path
    run_identity: Path
    run_completed: Path
    exit_coverage_summary: Path
    entry_features_telemetry: Path
    trade_index_csv: Path
    trades_dir: Path
    prebuilt_manifest: Path
    prebuilt_schema_manifest: Path


def _utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"[FULLYEAR_PROOF] Missing required env var: {name}")
    return val


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    """
    Atomic JSON write (same-dir tmp file + os.replace).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _write_text_atomic(path: Path, text: str) -> None:
    """
    Atomic text write (same-dir tmp file + os.replace).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _truth_prebuilt_env_apply() -> None:
    # Hard-set TRUTH/PREBUILT env gates (SSoT)
    os.environ["GX1_RUN_MODE"] = "TRUTH"
    os.environ["GX1_TRUTH_MODE"] = "1"
    os.environ["GX1_GATED_FUSION_ENABLED"] = "1"
    os.environ["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    os.environ["GX1_FEATURE_BUILD_DISABLED"] = "1"
    # Replay-only IO opts (semantics-neutral)
    os.environ.setdefault("GX1_JOURNAL_BUFFERING", "1")
    os.environ.setdefault("GX1_REPLAY_JSON_DEFER", "1")
    # Keep replay tags fastskip OFF unless user explicitly enables it (audit clarity)
    os.environ.setdefault("GX1_REPLAY_TAGS_FASTSKIP", "0")
    # Determinism helpers
    os.environ.setdefault("GX1_SEED", "42")
    # Keep core watchdog reasonable; runner adds a progress-aware heartbeat contract.
    os.environ.setdefault("GX1_WATCHDOG_STALL_TIMEOUT_SEC", "1800")


def _resolve_default_paths() -> Dict[str, str]:
    gx1_data = _require_env("GX1_DATA")
    defaults = {
        "raw_2025": str(Path(gx1_data) / "data" / "data" / "raw" / "xauusd_m5_2025_bid_ask.parquet"),
        # Signal-only truth: prebuilt/bundle must be explicit; no defaults.
        "prebuilt_2025": "",
        "prebuilt_2025_manifest": "",
        "prebuilt_2025_schema_manifest": "",
    }
    return defaults


def _build_paths(run_root: Path) -> ProofPaths:
    w2 = run_root  # this script is fullyear; run_root is already the run dir
    return ProofPaths(
        run_root=run_root,
        run_identity=run_root / "RUN_IDENTITY.json",
        run_completed=run_root / "RUN_COMPLETED.json",
        exit_coverage_summary=run_root / "EXIT_COVERAGE_SUMMARY.json",
        entry_features_telemetry=run_root / "chunk_0" / "ENTRY_FEATURES_TELEMETRY.json",
        trade_index_csv=run_root / "chunk_0" / "trade_journal" / "trade_journal_index.csv",
        trades_dir=run_root / "chunk_0" / "trade_journal" / "trades",
        prebuilt_manifest=run_root / "PREBUILT_MANIFEST.json",
        prebuilt_schema_manifest=run_root / "PREBUILT_SCHEMA_MANIFEST.json",
    )


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return _load_json(path)
    except Exception:
        return None


def _detect_run_status(run_dir: Path) -> str:
    """
    Deterministic status detection from run_dir markers.
    """
    if (run_dir / "RUN_COMPLETED.json").exists():
        return "COMPLETED"
    if (run_dir / "RUN_FAILED.json").exists():
        return "FAILED"
    if (run_dir / "PROGRESS_HEARTBEAT.json").exists():
        return "RUNNING"
    return "UNKNOWN"


def _count_trade_index(index_csv: Path) -> Dict[str, int]:
    """
    Stream-count trade_journal_index.csv rows.
    """
    out = {
        "rows_total": 0,
        "entry_events_total": 0,
        "close_events_total": 0,
        "unique_trade_uid_total": 0,
        "unique_trade_id_total": 0,
    }
    if not index_csv.exists():
        return out
    uids = set()
    tids = set()
    with index_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            out["rows_total"] += 1
            et = row.get("event_type")
            # TradeLifecycleV1: OPEN/CLOSE are POSITION trades (legacy ENTRY/EXIT supported)
            if et in ("ENTRY", "OPEN"):
                out["entry_events_total"] += 1
            elif et in ("EXIT", "CLOSE"):
                out["close_events_total"] += 1
            uid = row.get("trade_uid")
            if uid:
                uids.add(uid)
            tid = row.get("trade_id")
            if tid:
                tids.add(tid)
    out["unique_trade_uid_total"] = len(uids)
    out["unique_trade_id_total"] = len(tids)
    return out


def _count_trade_json_snapshots(trades_dir: Path) -> Dict[str, Any]:
    """
    Count per-trade snapshot files and whether they contain an exit_summary.
    """
    out: Dict[str, Any] = {
        "exists": trades_dir.exists(),
        "n_files": 0,
        "closed_files_has_exit_summary": 0,
        "open_files_no_exit_summary": 0,
    }
    if not trades_dir.exists():
        return out
    import json as _json

    for p in trades_dir.glob("*.json"):
        out["n_files"] += 1
        try:
            obj = _json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if obj.get("exit_summary") is None:
            out["open_files_no_exit_summary"] += 1
        else:
            out["closed_files_has_exit_summary"] += 1
    return out


def _session_counts_from_raw_signals(raw_signals_parquet: Path, max_rows: Optional[int] = None) -> Dict[str, int]:
    """
    Lightweight per-session counts using only the 'session' column.
    """
    if not raw_signals_parquet.exists():
        return {}
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(raw_signals_parquet)
    n = pf.metadata.num_rows
    to_read = n if max_rows is None else min(n, int(max_rows))
    table = pf.read(columns=["session"])
    if to_read != n:
        table = table.slice(n - to_read, to_read)
    sessions = table.column("session").to_pylist()
    counts: Dict[str, int] = {}
    for s in sessions:
        k = str(s).upper() if s is not None else "UNKNOWN"
        counts[k] = counts.get(k, 0) + 1
    return counts


def _feature_sanity_sample(
    prebuilt_parquet: Path,
    last_bar_ts_iso: Optional[str],
    sample_rows: int = 10000,
    nan_fail_threshold: float = 0.01,
    inf_fail_threshold: float = 0.01,
) -> Dict[str, Any]:
    """
    Sample-based feature sanity (interim): computes nan/inf/constant/unique_ratio on a tail sample.
    Avoids full-run scans.
    """
    if not prebuilt_parquet.exists():
        return {"available": False, "reason": "prebuilt_parquet_missing"}

    df = pd.read_parquet(prebuilt_parquet)
    if df.empty:
        return {"available": False, "reason": "prebuilt_parquet_empty"}

    # Prefer a tail window around the latest known bar time if possible.
    if last_bar_ts_iso:
        try:
            ts = pd.Timestamp(last_bar_ts_iso)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            start = ts - pd.Timedelta(days=2)
            sample = df.loc[(df.index >= start) & (df.index <= ts)]
            if len(sample) == 0:
                sample = df.tail(int(sample_rows))
        except Exception:
            sample = df.tail(int(sample_rows))
    else:
        sample = df.tail(int(sample_rows))

    cols = list(sample.columns)
    forbidden = sorted([c for c in cols if c.startswith("prob_")])
    n_rows = int(len(sample))
    per: Dict[str, Any] = {}
    top_nan: List[Tuple[str, float]] = []
    constant: List[str] = []
    near_constant: List[str] = []
    nan_fail: List[str] = []
    inf_fail: List[str] = []

    for c in cols:
        arr = pd.to_numeric(sample[c], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        nan_rate = float(nan_mask.mean()) if n_rows else 0.0
        inf_rate = float(inf_mask.mean()) if n_rows else 0.0
        finite = arr[~nan_mask & ~inf_mask]
        if finite.size == 0:
            std = np.nan
            n_unique = 0
        else:
            std = float(np.std(finite))
            n_unique = int(pd.Series(finite).nunique(dropna=True))
        unique_ratio = float(n_unique / n_rows) if n_rows else 0.0
        is_constant = bool(n_unique <= 1 or (std == 0.0 and np.isfinite(std)))
        per[c] = {
            "nan_rate": nan_rate,
            "inf_rate": inf_rate,
            "unique_ratio": unique_ratio,
            "is_constant": is_constant,
        }
        top_nan.append((c, nan_rate))
        if is_constant:
            constant.append(c)
        if unique_ratio > 0.0 and unique_ratio < 0.001:
            near_constant.append(c)
        if nan_rate > nan_fail_threshold:
            nan_fail.append(c)
        if inf_rate > inf_fail_threshold:
            inf_fail.append(c)

    top_nan.sort(key=lambda x: x[1], reverse=True)

    return {
        "available": True,
        "sample_rows": n_rows,
        "sample_index_min": str(sample.index.min()) if n_rows else None,
        "sample_index_max": str(sample.index.max()) if n_rows else None,
        "forbidden_features_found": forbidden,
        "nan_fail_threshold": float(nan_fail_threshold),
        "inf_fail_threshold": float(inf_fail_threshold),
        "nan_features_over_threshold": sorted(nan_fail),
        "inf_features_over_threshold": sorted(inf_fail),
        "degenerate_constant_count": int(len(constant)),
        "degenerate_near_constant_count": int(len(near_constant)),
        "top10_nan_rate": [{"feature": c, "nan_rate": r} for c, r in top_nan[:10]],
        "per_feature": per,
    }


def _interim_verdict(interim: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    fails: List[str] = []
    warns: List[str] = []

    inv = interim.get("invariants") or {}
    if inv.get("prebuilt_used") is not True:
        fails.append("prebuilt_used != true (or missing)")
    fbc = inv.get("feature_build_call_count")
    if fbc is None:
        warns.append("feature_build_call_count missing (interim)")
    elif int(fbc) != 0:
        fails.append("feature_build_call_count != 0")

    prog = interim.get("progress") or {}
    last_bar_ts = prog.get("last_bar_ts")
    if last_bar_ts is None:
        warns.append("last_bar_ts missing (no heartbeat)")
    else:
        try:
            if pd.Timestamp(last_bar_ts) < pd.Timestamp("2025-06-01T00:00:00+00:00"):
                fails.append("last_bar_ts < 2025-06-01 (did not reach summer)")
        except Exception:
            warns.append("last_bar_ts parse failed")

    trades = interim.get("trades") or {}
    if int(trades.get("unique_trade_uid_total") or 0) <= 0:
        warns.append("unique_trade_uid_total == 0 (interim)")

    funnel = interim.get("model_funnel") or {}
    if int(funnel.get("forward_calls_total") or 0) <= 0:
        fails.append("forward_calls_total == 0")

    ident = interim.get("identity_checks") or {}
    if ident.get("bundle_lock_ok") is False:
        fails.append("bundle MASTER_MODEL_LOCK sha check failed")
    if ident.get("prebuilt_manifest_ok") is False:
        fails.append("prebuilt manifest sha check failed")

    feat = interim.get("feature_sanity_sample") or {}
    if feat.get("forbidden_features_found"):
        fails.append("forbidden features found in sample")
    if feat.get("nan_features_over_threshold"):
        fails.append("NaN rate > 1% in sample for some feature(s)")
    if feat.get("inf_features_over_threshold"):
        fails.append("Inf rate > 1% in sample for some feature(s)")

    # Exceptions
    exc = interim.get("exceptions") or {}
    if exc.get("exceptions_count") is None:
        warns.append("exceptions_count unknown (interim)")
    elif int(exc.get("exceptions_count") or 0) > 0:
        fails.append("exceptions_count > 0")

    verdict = "INTERIM_GO" if not fails else "INTERIM_NO_GO"
    return verdict, fails, warns

def _tail_last_jsonl_record(path: Path, max_bytes: int = 256 * 1024) -> Optional[Dict[str, Any]]:
    """
    Tail-read the last JSONL record without reading the full file.
    """
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_size = min(size, max_bytes)
            f.seek(-read_size, os.SEEK_END)
            data = f.read(read_size)
        lines = data.splitlines()
        if not lines:
            return None
        last = lines[-1].decode("utf-8", errors="replace").strip()
        if not last:
            return None
        return json.loads(last)
    except Exception:
        return None


@dataclass
class _EvalLogProgress:
    lines_counted: int = 0
    file_offset: int = 0


def _eval_log_progress_update(progress: _EvalLogProgress, eval_log_path: Path) -> _EvalLogProgress:
    """
    Incrementally count newlines appended to eval_log JSONL file.
    This yields a stable, low-overhead proxy for eval_called / bars_processed.
    """
    if not eval_log_path.exists():
        return progress
    try:
        with open(eval_log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if progress.file_offset > end:
                # File truncated/rotated; reset.
                progress.file_offset = 0
                progress.lines_counted = 0
            f.seek(progress.file_offset, os.SEEK_SET)
            data = f.read(end - progress.file_offset)
            progress.file_offset = end
            if data:
                progress.lines_counted += int(data.count(b"\n"))
        return progress
    except Exception:
        return progress


def _progress_heartbeat_snapshot(
    run_id: str,
    output_dir: Path,
    chunk_id: str,
    eval_log_path: Optional[Path],
    eval_progress: Optional[_EvalLogProgress],
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    last_rec = _tail_last_jsonl_record(eval_log_path) if eval_log_path else None
    last_bar_ts = None
    if last_rec:
        last_bar_ts = last_rec.get("ts_utc") or last_rec.get("timestamp") or last_rec.get("ts")

    # replay_eval_gated_parallel.py (TRUTH master) writes MASTER_HEARTBEAT.json every ~10s.
    # Use it as the authoritative liveness/progress signal for multi-worker runs.
    master_hb = _safe_read_json(output_dir / "MASTER_HEARTBEAT.json") or {}
    utc_now = datetime.now(timezone.utc).isoformat()

    bars_processed_total = None
    if eval_progress is not None:
        bars_processed_total = int(eval_progress.lines_counted)

    return {
        "run_id": run_id,
        "utc_ts_written": utc_now,
        "output_dir": str(output_dir),
        "chunk_id": str(chunk_id),
        "stage": "MASTER" if master_hb else None,
        "master_heartbeat": master_hb or None,
        "bars_processed_total": bars_processed_total,
        "last_bar_ts": last_bar_ts,
        "eval_called": bars_processed_total,
        "predict_entered": None,
        "forward_calls": None,
        "exceptions": None,
        "notes": notes,
    }


def _decision_signature_from_index_csv(index_csv: Path) -> Tuple[str, int]:
    rows: List[Dict[str, Any]] = []
    with index_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    "event_type": row.get("event_type"),
                    "entry_ts_utc": row.get("entry_ts_utc"),
                    "entry_price": row.get("entry_price"),
                    "side": row.get("side"),
                    "score": row.get("score"),
                    "exit_ts_utc": row.get("exit_ts_utc"),
                    "exit_price": row.get("exit_price"),
                    "exit_reason": row.get("exit_reason"),
                    "pnl_bps": row.get("pnl_bps"),
                    "status": row.get("status"),
                }
            )
    payload = json.dumps(rows, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(rows)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_integrity_and_stats(
    prebuilt_parquet: Path,
    schema_manifest: Dict[str, Any],
    nan_fail_threshold: float = 0.001,
    inf_fail_threshold: float = 0.001,
) -> Dict[str, Any]:
    """
    Feature integrity + sanity stats for prebuilt flat schema parquet.

    - Ensures schema exact match against manifest allowlist
    - Computes per-feature:
      nan_rate, inf_rate, min/max/mean/std, unique_ratio, is_constant
    """
    required_all = list(schema_manifest.get("required_all_features") or [])
    required_snap = list(schema_manifest.get("required_snap_features") or [])
    required_seq = list(schema_manifest.get("required_seq_features") or [])

    df = pd.read_parquet(prebuilt_parquet)
    cols = list(df.columns)
    n_rows = int(len(df))

    # Signal-only truth: contract match is enforced by MASTER_MODEL_LOCK ordered features (order-sensitive).
    bundle_dir = Path(str(os.getenv("GX1_CANONICAL_BUNDLE_DIR") or "")).expanduser().resolve()
    if not bundle_dir.exists():
        raise RuntimeError("[PREBUILT_FAIL] TRUTH_NO_FALLBACK: GX1_CANONICAL_BUNDLE_DIR missing/invalid for lock-based integrity")
    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError(f"[PREBUILT_FAIL] MASTER_MODEL_LOCK.json missing: {lock_path}")
    lock = _load_json(lock_path)
    expected = list(lock.get("ordered_features") or [])
    if not expected:
        raise RuntimeError("[PREBUILT_FAIL] MASTER_MODEL_LOCK missing ordered_features")
    allowed_metadata = {"ts", "timestamp", "time", "__index_level_0__"}
    actual_features = [c for c in cols if str(c) not in allowed_metadata]
    missing_core = [f for f in expected if f not in set(actual_features)]
    extra_core = sorted(set(actual_features) - set(expected))
    if missing_core or extra_core or actual_features != expected:
        raise RuntimeError(
            "[PREBUILT_FAIL] CONTRACT_MISMATCH: prebuilt parquet columns do not match MASTER_MODEL_LOCK. "
            f"(missing={missing_core[:50]}, extra={extra_core[:50]}, order_ok={actual_features == expected}). "
            f"Prebuilt file: {prebuilt_parquet}"
        )

    # Exact schema match (order-independent)
    missing = sorted(set(required_all) - set(cols))
    extra = sorted(set(cols) - set(required_all))
    schema_exact = (len(missing) == 0) and (len(extra) == 0) and (len(cols) == len(required_all))

    forbidden = sorted([c for c in cols if c.startswith("prob_") or c.startswith("p_") and c.startswith("p_long") is False])
    # keep forbidden check strict for prob_*; do not blanket-ban "p_" because many features might start with p
    forbidden = sorted([c for c in cols if c.startswith("prob_")])

    stats: Dict[str, Any] = {}
    degenerate_constant: List[str] = []
    degenerate_near_constant: List[str] = []
    worst_nan: List[Tuple[str, float]] = []

    for c in cols:
        s = df[c]
        # ensure numeric
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        nan_rate = float(nan_mask.mean()) if n_rows else 0.0
        inf_rate = float(inf_mask.mean()) if n_rows else 0.0
        finite = arr[~nan_mask & ~inf_mask]
        if finite.size == 0:
            vmin = vmax = mean = std = np.nan
            n_unique = 0
        else:
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            mean = float(np.mean(finite))
            std = float(np.std(finite))
            n_unique = int(pd.Series(finite).nunique(dropna=True))
        unique_ratio = float(n_unique / n_rows) if n_rows else 0.0
        is_constant = bool(n_unique <= 1 or (std == 0.0 and np.isfinite(std)))

        stats[c] = {
            "nan_rate": nan_rate,
            "inf_rate": inf_rate,
            "min": vmin,
            "max": vmax,
            "mean": mean,
            "std": std,
            "n_unique": n_unique,
            "unique_ratio": unique_ratio,
            "is_constant": is_constant,
        }
        worst_nan.append((c, nan_rate))
        if is_constant:
            degenerate_constant.append(c)
        if unique_ratio > 0.0 and unique_ratio < 0.001:
            degenerate_near_constant.append(c)

    worst_nan.sort(key=lambda x: x[1], reverse=True)

    # Fail checks
    nan_fail = [c for c, st in stats.items() if float(st["nan_rate"]) > nan_fail_threshold]
    inf_fail = [c for c, st in stats.items() if float(st["inf_rate"]) > inf_fail_threshold]

    return {
        "n_rows": n_rows,
        "n_features": int(len(cols)),
        "schema_exact_match": bool(schema_exact),
        "schema_missing": missing,
        "schema_extra": extra,
        "required_all_features_count": int(len(required_all)),
        "required_seq_features_count": int(len(required_seq)),
        "required_snap_features_count": int(len(required_snap)),
        "forbidden_features_found": forbidden,
        "nan_fail_threshold": float(nan_fail_threshold),
        "inf_fail_threshold": float(inf_fail_threshold),
        "nan_features_over_threshold": nan_fail,
        "inf_features_over_threshold": inf_fail,
        "degenerate_constant": sorted(degenerate_constant),
        "degenerate_near_constant": sorted(degenerate_near_constant),
        "top10_nan_rate": [{"feature": c, "nan_rate": r} for c, r in worst_nan[:10]],
        "per_feature": stats,
    }


def _summarize_trades_by_session(trades_dir: Path) -> Dict[str, Any]:
    sessions = {"ASIA": 0, "EU": 0, "OVERLAP": 0, "US": 0, "UNKNOWN": 0}
    pnl_bps: List[float] = []
    n_files = 0
    for p in sorted(trades_dir.glob("*.json")):
        n_files += 1
        obj = json.loads(p.read_text(encoding="utf-8"))
        es = obj.get("entry_snapshot") or {}
        sess = str(es.get("session") or "UNKNOWN").upper()
        if sess not in sessions:
            sessions["UNKNOWN"] += 1
        else:
            sessions[sess] += 1
        xs = obj.get("exit_summary") or {}
        if xs.get("realized_pnl_bps") is not None:
            try:
                pnl_bps.append(float(xs["realized_pnl_bps"]))
            except Exception:
                pass
    pnl_arr = np.asarray(pnl_bps, dtype=np.float64) if pnl_bps else np.asarray([], dtype=np.float64)
    return {
        "n_trade_json_files": int(n_files),
        "trades_by_session": sessions,
        "pnl_bps": {
            "n": int(pnl_arr.size),
            "mean": float(np.mean(pnl_arr)) if pnl_arr.size else 0.0,
            "p50": float(np.percentile(pnl_arr, 50)) if pnl_arr.size else 0.0,
            "p90": float(np.percentile(pnl_arr, 90)) if pnl_arr.size else 0.0,
            "p99": float(np.percentile(pnl_arr, 99)) if pnl_arr.size else 0.0,
            "min": float(np.min(pnl_arr)) if pnl_arr.size else 0.0,
            "max": float(np.max(pnl_arr)) if pnl_arr.size else 0.0,
        },
    }


def _load_session_funnel(entry_features_telemetry_path: Path) -> Dict[str, Any]:
    telem = _load_json(entry_features_telemetry_path)
    per_sess = telem.get("per_session_funnel_ledger") or {}
    post_soft = telem.get("post_soft_funnel") or {}
    model_forward_by_session = (
        (post_soft.get("model_forward_calls_by_session") or {}).get("v10_hybrid") or {}
    )
    return {
        "per_session_funnel_ledger": per_sess,
        "model_forward_calls_by_session": model_forward_by_session,
        "run_level_funnel_ledger": telem.get("run_level_funnel_ledger") or {},
        "entry_routing_aggregate": telem.get("entry_routing_aggregate") or {},
    }


def _bundle_identity_proof(bundle_dir: Path) -> Dict[str, Any]:
    """
    Bundle identity gate for TRUTH proof pack.
    Verifies the on-disk SHA256 of locked artifacts against MASTER_MODEL_LOCK.json.
    """
    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if not lock_path.exists():
        return {"lock_found": False}
    lock = _load_json(lock_path)
    model_rel = lock.get("model_path_relative")
    meta_rel = lock.get("meta_path_relative")
    expected_model_sha = lock.get("model_sha256")
    expected_meta_sha = lock.get("meta_sha256")
    model_path = (bundle_dir / str(model_rel)).resolve() if model_rel else None
    meta_path = (bundle_dir / str(meta_rel)).resolve() if meta_rel else None

    out: Dict[str, Any] = {
        "lock_found": True,
        "lock_file_sha256": _sha256_file(lock_path),
        "lock_version": lock.get("version"),
        "expected_model_sha256": expected_model_sha,
        "expected_meta_sha256": expected_meta_sha,
        "model_path": str(model_path) if model_path else None,
        "meta_path": str(meta_path) if meta_path else None,
        "model_sha256": None,
        "meta_sha256": None,
        "model_sha256_matches_lock": None,
        "meta_sha256_matches_lock": None,
        "lock_sessions": lock.get("sessions"),
        "lock_invariants": lock.get("invariants"),
    }
    if model_path and model_path.exists():
        out["model_sha256"] = _sha256_file(model_path)
        out["model_sha256_matches_lock"] = bool(out["model_sha256"] == expected_model_sha)
    if meta_path and meta_path.exists():
        out["meta_sha256"] = _sha256_file(meta_path)
        out["meta_sha256_matches_lock"] = bool(out["meta_sha256"] == expected_meta_sha)
    return out


def _go_no_go(summary: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
    fails: List[str] = []
    warns: List[str] = []

    inv = summary.get("invariants") or {}
    if not inv.get("run_completed", False):
        fails.append("RUN_COMPLETED missing/false")
    if not inv.get("prebuilt_used", False):
        fails.append("prebuilt_used != true")
    if int(inv.get("feature_build_call_count", -1)) != 0:
        fails.append("feature_build_call_count != 0")
    if not inv.get("truth_exit_journal_ok", False):
        fails.append("truth_exit_journal_ok != true")
    if not inv.get("session_ledger_present", False):
        fails.append("missing session ledger (per_session_funnel_ledger absent/empty)")

    b = summary.get("bundle_identity") or {}
    if not b.get("lock_found", False):
        fails.append("MASTER_MODEL_LOCK.json missing in bundle (lock_found=false)")
    if b.get("model_sha256_matches_lock") is False:
        fails.append("xgb model sha256 does not match MASTER_MODEL_LOCK.json")
    if b.get("meta_sha256_matches_lock") is False:
        fails.append("xgb meta sha256 does not match MASTER_MODEL_LOCK.json")

    idn = summary.get("identity") or {}
    if idn.get("prebuilt_sha256_matches_manifest") is False:
        fails.append("prebuilt parquet sha256 does not match PREBUILT_MANIFEST.json")
    if idn.get("raw_sha256_matches_manifest") is False:
        fails.append("raw parquet sha256 does not match PREBUILT_MANIFEST.json")

    funnel = summary.get("session_funnel") or {}
    mf = funnel.get("model_forward_calls_by_session") or {}
    nonzero_sessions = [s for s, v in mf.items() if int(v) > 0]
    if int((summary.get("model_funnel") or {}).get("transformer_forward_calls_total", 0)) <= 0:
        fails.append("transformer_forward_calls_total == 0")
    if len(nonzero_sessions) < 2:
        fails.append("forward_calls nonzero sessions < 2")

    feat = summary.get("feature_integrity") or {}
    if not feat.get("schema_exact_match", False):
        fails.append("feature schema mismatch")
    if feat.get("forbidden_features_found"):
        fails.append(f"forbidden features found: {feat.get('forbidden_features_found')}")
    if feat.get("nan_features_over_threshold"):
        fails.append("NaN rate over threshold for some feature(s)")
    if feat.get("inf_features_over_threshold"):
        fails.append("Inf rate over threshold for some feature(s)")
    deg_const = feat.get("degenerate_constant") or []
    if len(deg_const) > 3:
        warns.append(f"degenerate constant features > 3 (count={len(deg_const)})")

    verdict = "GO" if not fails else "NO-GO"
    return verdict, fails, warns


def _run_replay(
    output_dir: Path,
    policy_path: Path,
    raw_path: Path,
    prebuilt_path: Path,
    bundle_dir: Path,
    workers: int,
    chunks: int,
    start_ts: str,
    end_ts: str,
) -> None:
    cmd = [
        sys.executable,
        "gx1/scripts/replay_eval_gated_parallel.py",
        "--policy",
        str(policy_path),
        "--data",
        str(raw_path),
        "--prebuilt-parquet",
        str(prebuilt_path),
        "--bundle-dir",
        str(bundle_dir),
        "--output-dir",
        str(output_dir),
        "--workers",
        str(int(workers)),
        "--chunks",
        str(int(chunks)),
        "--start-ts",
        str(start_ts),
        "--end-ts",
        str(end_ts),
    ]
    subprocess.run(cmd, check=True)


def _run_replay_with_progress(
    output_dir: Path,
    policy_path: Path,
    raw_path: Path,
    prebuilt_path: Path,
    bundle_dir: Path,
    workers: int,
    chunks: int,
    start_ts: str,
    end_ts: str,
    heartbeat_interval_sec: float,
    stall_timeout_sec: float,
) -> int:
    """
    Run replay subprocess while periodically writing PROGRESS_HEARTBEAT.json.

    Progress source (low-risk): parse existing artifacts from disk:
    - eval_log_*.jsonl tail + incremental newline count for bars_processed_total proxy

    Returns subprocess exit code.
    """
    cmd = [
        sys.executable,
        "gx1/scripts/replay_eval_gated_parallel.py",
        "--policy",
        str(policy_path),
        "--data",
        str(raw_path),
        "--prebuilt-parquet",
        str(prebuilt_path),
        "--bundle-dir",
        str(bundle_dir),
        "--output-dir",
        str(output_dir),
        "--workers",
        str(int(workers)),
        "--chunks",
        str(int(chunks)),
        "--start-ts",
        str(start_ts),
        "--end-ts",
        str(end_ts),
    ]

    # NOTE: replay_eval_gated_parallel.py watchdog progress heuristics do not currently observe
    # long single-chunk forward progress (they key off a small set of chunk artifacts).
    # To avoid false STALL_FATAL inside the replay core, keep its stall timeout high enough
    # and use THIS runner's progress-aware stall detection (eval_log advancement) as the
    # authoritative liveness signal.
    env = os.environ.copy()
    core_watchdog_timeout = max(7200, int(float(stall_timeout_sec) * 4.0)) if stall_timeout_sec else 7200
    env["GX1_WATCHDOG_STALL_TIMEOUT_SEC"] = str(int(core_watchdog_timeout))

    proc = subprocess.Popen(cmd, env=env)
    last_write = 0.0
    last_progress_ts = time.time()
    last_bars: Optional[int] = None
    eval_progress = _EvalLogProgress()
    last_master_mtime: Optional[float] = None
    last_master_seen_ts = time.time()
    last_completed: Optional[int] = None
    last_completed_change_ts = time.time()

    while True:
        rc = proc.poll()

        now = time.time()

        # Liveness/progress signal for parallel runs: MASTER_HEARTBEAT.json (updated ~10s by master).
        master_hb_path = output_dir / "MASTER_HEARTBEAT.json"
        master_hb = None
        try:
            if master_hb_path.exists():
                st = master_hb_path.stat()
                if last_master_mtime is None or st.st_mtime > last_master_mtime:
                    last_master_mtime = float(st.st_mtime)
                    last_master_seen_ts = now
                master_hb = _safe_read_json(master_hb_path) or {}
                completed = master_hb.get("n_completed")
                if isinstance(completed, int):
                    if last_completed is None:
                        last_completed = completed
                        last_completed_change_ts = now
                    elif completed != last_completed:
                        last_completed = completed
                        last_completed_change_ts = now
        except Exception:
            master_hb = None

        # Refresh progress timestamp if master heartbeat is alive.
        if master_hb_path.exists() and last_master_seen_ts and (now - last_master_seen_ts) < 15.0:
            last_progress_ts = now

        # Fast stall detection (≤ stall_timeout_sec): scheduler stuck (no workers running, not fully submitted).
        try:
            if stall_timeout_sec and master_hb and rc is None:
                n_running = int(master_hb.get("n_running_workers") or 0)
                n_submitted = int(master_hb.get("n_submitted") or 0)
                total_chunks = int(master_hb.get("total_chunks") or 0)
                if total_chunks > 0 and n_submitted < total_chunks and n_running == 0:
                    if (now - last_completed_change_ts) > float(stall_timeout_sec):
                        stall = {
                            "run_id": str(output_dir.name),
                            "status": "STALL_FATAL",
                            "now_utc": datetime.now(timezone.utc).isoformat(),
                            "stall_timeout_seconds": float(stall_timeout_sec),
                            "seconds_since_completed_change": float(now - last_completed_change_ts),
                            "master_heartbeat": master_hb,
                            "notes": "No running workers and no completed-count change within stall window.",
                        }
                        if output_dir.exists():
                            _write_json_atomic(output_dir / "RUN_STALL_FATAL.json", stall)
                        proc.terminate()
                        try:
                            proc.wait(timeout=30)
                        except Exception:
                            proc.kill()
                        return 1
        except Exception:
            pass

        do_write = (now - last_write) >= float(heartbeat_interval_sec)
        if do_write:
            eval_log_path = None
            logs_dir = output_dir / "chunk_0" / "logs"
            if logs_dir.exists():
                # pick the most recent eval_log_*.jsonl
                candidates = sorted(logs_dir.glob("eval_log_*.jsonl"))
                if candidates:
                    eval_log_path = candidates[-1]
                    eval_progress = _eval_log_progress_update(eval_progress, eval_log_path)

            bars = int(eval_progress.lines_counted)
            if last_bars is None or bars > last_bars:
                last_progress_ts = now
                last_bars = bars

            snapshot = _progress_heartbeat_snapshot(
                run_id=str(output_dir.name),
                output_dir=output_dir,
                chunk_id="0",
                eval_log_path=eval_log_path,
                eval_progress=eval_progress,
                notes=None,
            )
            # Only write if output_dir exists (core creates it). If it doesn't yet, skip.
            if output_dir.exists():
                _write_json_atomic(output_dir / "PROGRESS_HEARTBEAT.json", snapshot)
            last_write = now

        # Progress-aware stall detection: only consider STALL if bars don't advance.
        if stall_timeout_sec and (now - last_progress_ts) > float(stall_timeout_sec) and rc is None:
            stall = {
                "run_id": str(output_dir.name),
                "status": "STALL_FATAL",
                "now_utc": datetime.now(timezone.utc).isoformat(),
                "stall_timeout_seconds": float(stall_timeout_sec),
                "seconds_since_progress": float(now - last_progress_ts),
                "bars_processed_total": int(eval_progress.lines_counted),
                "master_heartbeat": master_hb,
                "notes": "No progress detected within stall window (MASTER_HEARTBEAT + eval_log).",
            }
            if output_dir.exists():
                _write_json_atomic(output_dir / "RUN_STALL_FATAL.json", stall)
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()
            return 1

        if rc is not None:
            # Final heartbeat write.
            try:
                eval_log_path = None
                logs_dir = output_dir / "chunk_0" / "logs"
                if logs_dir.exists():
                    candidates = sorted(logs_dir.glob("eval_log_*.jsonl"))
                    if candidates:
                        eval_log_path = candidates[-1]
                        eval_progress = _eval_log_progress_update(eval_progress, eval_log_path)
                snapshot = _progress_heartbeat_snapshot(
                    run_id=str(output_dir.name),
                    output_dir=output_dir,
                    chunk_id="0",
                    eval_log_path=eval_log_path,
                    eval_progress=eval_progress,
                    notes="subprocess_exit",
                )
                if output_dir.exists():
                    _write_json_atomic(output_dir / "PROGRESS_HEARTBEAT.json", snapshot)
            except Exception:
                pass
            return int(rc)

        time.sleep(1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=1)
    ap.add_argument("--run_id", type=str, default=f"FULLYEAR_2025_PROOF_{_utc_ts_compact()}")
    ap.add_argument("--run_dir", type=str, default="")
    ap.add_argument("--start_ts", type=str, default="2025-01-01T00:00:00+00:00")
    ap.add_argument("--end_ts", type=str, default="2025-12-31T23:59:59+00:00")
    ap.add_argument("--raw_2025", type=str, default="")
    ap.add_argument("--prebuilt-parquet", type=str, default="")
    ap.add_argument("--skip_run", action="store_true")
    ap.add_argument("--mark_aborted", action="store_true")
    ap.add_argument("--heartbeat_interval_sec", type=float, default=60.0)
    ap.add_argument("--stall_timeout_sec", type=float, default=900.0)
    args = ap.parse_args()

    _truth_prebuilt_env_apply()

    # TRUTH BANLIST: fail-fast on forbidden fallback envs / legacy module imports (best-effort).
    from gx1.utils.truth_banlist import assert_truth_banlist_clean
    assert_truth_banlist_clean(output_dir=Path(args.run_dir) if args.run_dir else None, stage="run_fullyear_2025_truth_proof:entry")

    gx1_data = Path(_require_env("GX1_DATA"))
    defaults = _resolve_default_paths()
    raw_2025 = Path(args.raw_2025 or defaults["raw_2025"])
    # Signal-only truth: require explicit canonical bundle + explicit prebuilt parquet.
    canonical_bundle_dir = os.getenv("GX1_CANONICAL_BUNDLE_DIR") or ""
    if not canonical_bundle_dir:
        raise RuntimeError("[FULLYEAR_PROOF] TRUTH_NO_FALLBACK: GX1_CANONICAL_BUNDLE_DIR is required (absolute)")
    bundle_dir = Path(canonical_bundle_dir).expanduser().resolve()
    if not bundle_dir.is_absolute() or not bundle_dir.exists():
        raise RuntimeError(f"[FULLYEAR_PROOF] Invalid GX1_CANONICAL_BUNDLE_DIR: {bundle_dir}")
    lock_path = bundle_dir / "MASTER_MODEL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError(f"[FULLYEAR_PROOF] MASTER_MODEL_LOCK.json missing: {lock_path}")
    lock_obj = _load_json(lock_path)
    expected_features = list(lock_obj.get("ordered_features") or [])
    if not expected_features:
        raise RuntimeError("[FULLYEAR_PROOF] MASTER_MODEL_LOCK missing ordered_features")

    prebuilt_2025 = Path(str(args.prebuilt_parquet or "")).expanduser().resolve()
    if not str(args.prebuilt_parquet or "").strip():
        raise RuntimeError("[FULLYEAR_PROOF] TRUTH_NO_FALLBACK: --prebuilt-parquet is required (absolute)")
    if not prebuilt_2025.is_absolute() or not prebuilt_2025.exists():
        raise RuntimeError(f"[FULLYEAR_PROOF] --prebuilt-parquet invalid/missing: {prebuilt_2025}")
    prebuilt_manifest = prebuilt_2025.with_suffix(".manifest.json")
    prebuilt_schema_manifest = prebuilt_2025.with_suffix(".schema_manifest.json")
    if not prebuilt_schema_manifest.exists():
        raise RuntimeError(f"[FULLYEAR_PROOF] Prebuilt schema manifest missing: {prebuilt_schema_manifest}")

    # Hard gate: schema manifest required_all_features must match lock ordered_features (order-sensitive).
    schema_obj = _load_json(prebuilt_schema_manifest)
    required_all = list(schema_obj.get("required_all_features") or [])
    if required_all != expected_features:
        raise RuntimeError(
            "[FULLYEAR_PROOF] PREBUILT_SCHEMA_MISMATCH: schema_manifest.required_all_features must equal MASTER_MODEL_LOCK.ordered_features "
            "(order-sensitive)."
        )

    # Hard gate: XGB meta feature list must match lock ordered_features (order-sensitive).
    meta_rel = str(lock_obj.get("meta_path_relative") or "xgb_universal_multihead_v2_meta.json")
    meta_path = bundle_dir / meta_rel
    if not meta_path.exists():
        raise RuntimeError(f"[FULLYEAR_PROOF] XGB meta missing: {meta_path}")
    meta_obj = _load_json(meta_path)
    meta_features = list(meta_obj.get("feature_names_ordered") or meta_obj.get("ordered_features") or [])
    if meta_features != expected_features:
        raise RuntimeError(
            "[FULLYEAR_PROOF] XGB_META_MISMATCH: xgb meta ordered features must equal MASTER_MODEL_LOCK.ordered_features "
            "(order-sensitive)."
        )

    policy_path = Path(_require_env("GX1_CANONICAL_POLICY_PATH"))

    base_root = gx1_data / "reports" / "truth_fullyear_2025"
    base_root.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.run_dir) if args.run_dir else (base_root / str(args.run_id))

    if args.skip_run:
        if not out_root.exists():
            raise RuntimeError(f"[FULLYEAR_PROOF] --skip_run requested but run dir does not exist: {out_root}")
    else:
        if out_root.exists() and any(out_root.iterdir()):
            out_root = base_root / f"{args.run_id}_{_utc_ts_compact()}"

    rc = 0
    summary_md_path: Optional[Path] = None
    summary_json_path: Optional[Path] = None
    tb_path: Optional[Path] = None
    err_type: Optional[str] = None
    err_msg: Optional[str] = None

    try:
        if not args.skip_run:
            rc = _run_replay_with_progress(
                output_dir=out_root,
                policy_path=policy_path,
                raw_path=raw_2025,
                prebuilt_path=prebuilt_2025,
                bundle_dir=bundle_dir,
                workers=int(args.workers),
                chunks=int(args.chunks),
                start_ts=str(args.start_ts),
                end_ts=str(args.end_ts),
                heartbeat_interval_sec=float(args.heartbeat_interval_sec),
                stall_timeout_sec=float(args.stall_timeout_sec),
            )
            if rc != 0:
                raise RuntimeError(f"[FULLYEAR_PROOF] Replay subprocess failed (exit_code={rc})")

        run_status = _detect_run_status(out_root)
        paths = _build_paths(out_root)
        if run_status == "COMPLETED" and not paths.run_completed.exists():
            raise RuntimeError(f"[FULLYEAR_PROOF] Missing RUN_COMPLETED.json: {paths.run_completed}")

        prebuilt_manifest_obj = _load_json(prebuilt_manifest) if prebuilt_manifest.exists() else {}
        prebuilt_schema_manifest_obj = _load_json(prebuilt_schema_manifest)

        # Persist proof pack inputs into run-root (after replay, to avoid OUTPUT_DIR_INIT TRUTH violation).
        _write_text(out_root / "PROOF_CMD.txt", " ".join(sys.argv))
        _write_json(out_root / "PREBUILT_MANIFEST.json", prebuilt_manifest_obj)
        _write_json(out_root / "PREBUILT_SCHEMA_MANIFEST.json", prebuilt_schema_manifest_obj)

        run_completed = _load_json(paths.run_completed) if paths.run_completed.exists() else {}
        exit_cov = _load_json(paths.exit_coverage_summary) if paths.exit_coverage_summary.exists() else {}
        chunk_footer_path = out_root / "chunk_0" / "chunk_footer.json"
        chunk_footer = _load_json(chunk_footer_path) if chunk_footer_path.exists() else {}
        entry_telem = _load_session_funnel(paths.entry_features_telemetry) if paths.entry_features_telemetry.exists() else {}

        prebuilt_used_flag = None
        if chunk_footer:
            prebuilt_used_flag = bool(chunk_footer.get("prebuilt_used", False))
        else:
            ri = _safe_read_json(out_root / "RUN_IDENTITY.json") or {}
            if "prebuilt_used" in ri:
                prebuilt_used_flag = bool(ri.get("prebuilt_used"))

        feature_build_call_count = None
        if chunk_footer:
            try:
                feature_build_call_count = int(chunk_footer.get("feature_build_call_count", -1))
            except Exception:
                feature_build_call_count = None

        invariants = {
            "run_completed": bool(run_completed.get("status") == "COMPLETED") if run_completed else False,
            "prebuilt_used": prebuilt_used_flag,
            "feature_build_call_count": feature_build_call_count,
            "truth_exit_journal_ok": exit_cov.get("truth_exit_journal_ok") if "truth_exit_journal_ok" in exit_cov else None,
            "exceptions_count": int((entry_telem.get("run_level_funnel_ledger") or {}).get("exceptions_count", 0)) if entry_telem else None,
            "session_ledger_present": bool((entry_telem.get("per_session_funnel_ledger") or {}) != {}) if entry_telem else False,
        }

        run_level = (entry_telem.get("run_level_funnel_ledger") or {}) if entry_telem else {}
        model_funnel = {
            "predict_entered_total": int(run_level.get("predict_entered_count", 0)) if run_level else None,
            "transformer_forward_calls_total": int(run_level.get("transformer_forward_calls", 0)) if run_level else None,
        }

        # Avoid hashing the (potentially huge) raw parquet in interim mode.
        prebuilt_sha = _sha256_file(prebuilt_2025)
        raw_sha = None
        prebuilt_sha_expected = str(prebuilt_manifest_obj.get("features_file_sha256") or "")
        raw_sha_expected = str(prebuilt_manifest_obj.get("input_file_sha256") or "")

        requested_run_id = str(args.run_id) if not args.run_dir else None
        bundle_identity = _bundle_identity_proof(bundle_dir)

        md_lines: List[str] = []
        md_lines.append("## TRUTH FULLYEAR 2025 Proof Pack")
        md_lines.append("")
        md_lines.append(f"- **run_id**: `{out_root.name}`")
        md_lines.append(f"- **requested_run_id**: `{requested_run_id}`")
        md_lines.append(f"- **status_detected**: `{run_status}`")
        md_lines.append(f"- **run_root**: `{out_root}`")
        md_lines.append("")
        md_lines.append("## Identity (SSoT)")
        md_lines.append(f"- **bundle_dir**: `{bundle_dir}`")
        md_lines.append(f"- **policy_path**: `{policy_path}`")
        md_lines.append(f"- **prebuilt_sha256**: `{prebuilt_sha}`")
        md_lines.append(f"- **prebuilt_sha256_expected(manifest)**: `{prebuilt_sha_expected}`")
        md_lines.append(f"- **prebuilt_sha256_matches_manifest**: `{bool(prebuilt_sha == prebuilt_sha_expected)}`")
        md_lines.append(f"- **raw_sha256_expected(manifest)**: `{raw_sha_expected}`")
        md_lines.append(f"- **raw_sha256_checked**: `False`")
        md_lines.append(f"- **master_model_lock_sha256**: `{bundle_identity.get('lock_file_sha256')}`")
        md_lines.append(f"- **xgb_model_sha256_matches_lock**: `{bundle_identity.get('model_sha256_matches_lock')}`")
        md_lines.append(f"- **xgb_meta_sha256_matches_lock**: `{bundle_identity.get('meta_sha256_matches_lock')}`")
        md_lines.append("")
        md_lines.append("## Hard gates")
        for k, v in invariants.items():
            md_lines.append(f"- **{k}**: `{v}`")
        md_lines.append("")

        if run_status == "COMPLETED":
            feat = _feature_integrity_and_stats(prebuilt_2025, prebuilt_schema_manifest_obj)
            trade_sig, trade_sig_rows = _decision_signature_from_index_csv(paths.trade_index_csv)
            trades = _summarize_trades_by_session(paths.trades_dir)

            summary = {
                "run_id": str(out_root.name),
                "requested_run_id": requested_run_id,
                "effective_run_dir": str(out_root),
                "paths": {
                    "run_root": str(out_root),
                    "run_completed": str(paths.run_completed),
                    "exit_coverage_summary": str(paths.exit_coverage_summary),
                    "entry_features_telemetry": str(paths.entry_features_telemetry),
                    "trade_index_csv": str(paths.trade_index_csv),
                    "trades_dir": str(paths.trades_dir),
                },
                "identity": {
                    "bundle_dir": str(bundle_dir),
                    "policy_path": str(policy_path),
                    "prebuilt_parquet": str(prebuilt_2025),
                    "raw_parquet": str(raw_2025),
                    "prebuilt_sha256": prebuilt_sha,
                    "raw_sha256": raw_sha,
                    "prebuilt_sha256_expected": prebuilt_sha_expected,
                    "raw_sha256_expected": raw_sha_expected,
                    "prebuilt_sha256_matches_manifest": bool(prebuilt_sha == prebuilt_sha_expected),
                    "raw_sha256_matches_manifest": None,
                },
                "bundle_identity": bundle_identity,
                "invariants": invariants,
                "perf": run_completed.get("perf_counters") or {},
                "stage_timings_s": run_completed.get("stage_timings_s") or {},
                "session_funnel": entry_telem,
                "model_funnel": model_funnel,
                "trade_journal": {
                    "decision_signature_sha256": trade_sig,
                    "decision_signature_rows": int(trade_sig_rows),
                    **trades,
                },
                "feature_integrity": feat,
            }

            verdict, fails, warns = _go_no_go(summary)
            summary["verdict"] = {"verdict": verdict, "fails": fails, "warns": warns}
            md_lines.insert(4, f"- **verdict**: **{verdict}**")

            summary_json_path = out_root / "TRUTH_FULLYEAR_2025_SUMMARY.json"
            _write_json_atomic(summary_json_path, summary)

            if fails:
                md_lines.append("## NO-GO reasons")
                for r in fails:
                    md_lines.append(f"- **FAIL**: {r}")
                md_lines.append("")
            if warns:
                md_lines.append("## Warnings")
                for r in warns:
                    md_lines.append(f"- **WARN**: {r}")
                md_lines.append("")
        else:
            progress_obj = _safe_read_json(out_root / "PROGRESS_HEARTBEAT.json") or {}
            trade_counts = _count_trade_index(paths.trade_index_csv)

            join_obj = _safe_read_json(out_root / "chunk_0" / "RAW_PREBUILT_JOIN.json") or {}
            prebuilt_path_used = join_obj.get("prebuilt_path") or str(prebuilt_2025)

            raw_signals_path = out_root / "chunk_0" / f"raw_signals_{out_root.name}.parquet"
            if not raw_signals_path.exists():
                cand = sorted((out_root / "chunk_0").glob("raw_signals_*.parquet"))
                if cand:
                    raw_signals_path = cand[-1]
            per_session_forward = _session_counts_from_raw_signals(raw_signals_path) if raw_signals_path.exists() else {}
            forward_calls_total = int(sum(per_session_forward.values())) if per_session_forward else 0

            sample_feat = _feature_sanity_sample(Path(str(prebuilt_path_used)), progress_obj.get("last_bar_ts"))

            interim = {
                "run_id": str(out_root.name),
                "status_detected": run_status,
                "progress": {
                    "utc_ts_written": progress_obj.get("utc_ts_written"),
                    "bars_processed_total": progress_obj.get("bars_processed_total"),
                    "last_bar_ts": progress_obj.get("last_bar_ts"),
                    "chunk_id": progress_obj.get("chunk_id"),
                },
                "identity_gates": {
                    "run_identity_present": (out_root / "RUN_IDENTITY.json").exists(),
                    "pre_fork_freeze_present": (out_root / "PRE_FORK_FREEZE.json").exists(),
                    "raw_prebuilt_join_present": (out_root / "chunk_0" / "RAW_PREBUILT_JOIN.json").exists(),
                    "worker_boot_present": (out_root / "chunk_0" / "WORKER_BOOT.json").exists(),
                },
                "identity_checks": {
                    "bundle_lock_ok": bool(bundle_identity.get("model_sha256_matches_lock") and bundle_identity.get("meta_sha256_matches_lock")),
                    "prebuilt_sha256": prebuilt_sha,
                    "prebuilt_sha256_expected(manifest)": prebuilt_sha_expected,
                    "prebuilt_manifest_ok": bool(prebuilt_sha == prebuilt_sha_expected),
                    "prebuilt_path_used": str(prebuilt_path_used),
                },
                "invariants": invariants,
                "trades": {
                    "trade_index_rows_total": int(trade_counts.get("rows_total", 0)),
                    "entry_events_total": int(trade_counts.get("entry_events_total", 0)),
                    "close_events_total": int(trade_counts.get("close_events_total", 0)),
                    "unique_trade_uid_total": int(trade_counts.get("unique_trade_uid_total", 0)),
                    "unique_trade_id_total": int(trade_counts.get("unique_trade_id_total", 0)),
                    "entry_events_per_session_proxy": per_session_forward,
                },
                "model_funnel": {
                    "forward_calls_total": forward_calls_total,
                    "forward_calls_per_session_proxy": per_session_forward,
                },
                "truth_exit_journal_ok": invariants.get("truth_exit_journal_ok"),
                "feature_sanity_sample": sample_feat,
            }
            interim["trade_json_snapshots"] = _count_trade_json_snapshots(out_root / "chunk_0" / "trade_journal" / "trades")

            iv, ifails, iwarns = _interim_verdict(interim)
            interim["verdict"] = {"verdict": iv, "fails": ifails, "warns": iwarns}
            md_lines.insert(4, f"- **verdict**: **{iv}**")

            summary_json_path = out_root / "TRUTH_FULLYEAR_2025_INTERIM_SUMMARY.json"
            _write_json_atomic(summary_json_path, interim)

            md_lines.append("## Ops / progress (heartbeat)")
            md_lines.append(f"- **bars_processed_total**: `{progress_obj.get('bars_processed_total')}`")
            md_lines.append(f"- **last_bar_ts**: `{progress_obj.get('last_bar_ts')}`")
            md_lines.append(f"- **utc_ts_written**: `{progress_obj.get('utc_ts_written')}`")
            md_lines.append("")
            md_lines.append("## Trades / funnel (interim)")
            md_lines.append(f"- **entry_events_total**: `{int(trade_counts.get('entry_events_total', 0))}`")
            md_lines.append(f"- **close_events_total**: `{int(trade_counts.get('close_events_total', 0))}`")
            md_lines.append(f"- **unique_trade_uid_total**: `{int(trade_counts.get('unique_trade_uid_total', 0))}`")
            md_lines.append(f"- **forward_calls_total(proxy)**: `{forward_calls_total}`")
            md_lines.append("")
            md_lines.append("## Forward calls per session (proxy: raw_signals.session)")
            for s in ["ASIA", "EU", "OVERLAP", "US", "UNKNOWN"]:
                md_lines.append(f"- **{s}**: `{int(per_session_forward.get(s, 0) or 0)}`")
            md_lines.append("")
            md_lines.append("## Trade JSON snapshots (trades/*.json)")
            snaps = interim.get("trade_json_snapshots") or {}
            md_lines.append(f"- **n_files**: `{snaps.get('n_files')}`")
            md_lines.append(f"- **closed_files_has_exit_summary**: `{snaps.get('closed_files_has_exit_summary')}`")
            md_lines.append(f"- **open_files_no_exit_summary**: `{snaps.get('open_files_no_exit_summary')}`")
            md_lines.append("")
            md_lines.append("## Feature sanity (sample-based)")
            md_lines.append(f"- **forbidden_features_found**: `{sample_feat.get('forbidden_features_found')}`")
            md_lines.append(f"- **nan_features_over_threshold(>1%)**: `{len(sample_feat.get('nan_features_over_threshold') or [])}`")
            md_lines.append(f"- **inf_features_over_threshold(>1%)**: `{len(sample_feat.get('inf_features_over_threshold') or [])}`")
            md_lines.append("")
            if ifails:
                md_lines.append("## INTERIM NO-GO reasons")
                for r in ifails:
                    md_lines.append(f"- **FAIL**: {r}")
                md_lines.append("")
            if iwarns:
                md_lines.append("## Warnings")
                for r in iwarns:
                    md_lines.append(f"- **WARN**: {r}")
                md_lines.append("")

        summary_md_path = out_root / ("TRUTH_FULLYEAR_2025_SUMMARY.md" if run_status == "COMPLETED" else "TRUTH_FULLYEAR_2025_INTERIM_SUMMARY.md")
        _write_text_atomic(summary_md_path, "\n".join(md_lines) + "\n")
        print(f"[FULLYEAR_PROOF] OK: wrote {summary_md_path}")

    except Exception as e:
        rc = rc or 1
        err_type = type(e).__name__
        err_msg = str(e)
        if not out_root.exists():
            out_root.mkdir(parents=True, exist_ok=True)
        tb_path = out_root / "RUN_FAILED_TRACEBACK.txt"
        _write_text(tb_path, "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        raise

    finally:
        try:
            if not out_root.exists():
                out_root.mkdir(parents=True, exist_ok=True)

            detected = _detect_run_status(out_root)
            if args.mark_aborted:
                status = "ABORTED"
            else:
                status = detected
                if status == "UNKNOWN" and rc != 0:
                    status = "FAILED"
            ex = _safe_read_json(out_root / "EXIT_COVERAGE_SUMMARY.json") or {}
            truth_exit_ok = bool(ex.get("truth_exit_journal_ok")) if "truth_exit_journal_ok" in ex else None

            cf = _safe_read_json(out_root / "chunk_0" / "chunk_footer.json") or {}
            mes = cf.get("model_entry_summary") or {}
            forward_calls_total = int(mes.get("forward_calls") or 0) if mes else None
            trades_total = int(cf.get("killchain_n_trade_created") or 0) if cf else None

            final_status = {
                "run_id": str(out_root.name),
                "status": status,
                "utc_ts_finished": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(out_root),
                "summary_md_path": str(summary_md_path) if summary_md_path else None,
                "summary_json_path": str(summary_json_path) if summary_json_path else None,
                "exit_code": int(rc),
                "failure": {
                    "error_type": err_type,
                    "message": err_msg,
                    "traceback_path": str(tb_path) if tb_path else None,
                }
                if status in ["FAILED", "ABORTED", "UNKNOWN"]
                else None,
                "counters_snapshot": {
                    "trades_total": trades_total,
                    "forward_calls_total": forward_calls_total,
                    "truth_exit_journal_ok": truth_exit_ok,
                },
            }
            final_path = out_root / "FINAL_STATUS.json"
            if (not final_path.exists()) or args.mark_aborted:
                _write_json_atomic(final_path, final_status)
        except Exception:
            pass


if __name__ == "__main__":
    main()

