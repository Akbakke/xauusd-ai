#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from gx1.exits.contracts.exit_io_v1_ctx36_features import EXIT_IO_V1_CTX36_FEATURE_COUNT
from gx1.exits.contracts.exit_io_v3_ctx36_m1l512_phase5 import (
    EXIT_IO_V3_CTX36_M1L512_PHASE5_EXTRA_FEATURES,
    EXIT_IO_V3_CTX36_M1L512_PHASE5_IO_VERSION,
    compute_m5_phase_onehot,
)
from gx1.exits.contracts.registry import get_exit_io_contract
from gx1.policy import exit_transformer_v0 as exit_v0

if not exit_v0.TORCH_AVAILABLE:
    raise RuntimeError("PyTorch required")

import torch
import torch.nn as nn


log = logging.getLogger("train_exit_transformer_v0_sharded")


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _make_deterministic(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    np.random.seed(seed)
    random.seed(seed)


def _is_wsl() -> bool:
    rel = platform.release().lower()
    return "microsoft" in rel or "wsl" in rel


def _normalize_ts(ts_val: pd.Timestamp) -> pd.Timestamp:
    if ts_val is pd.NaT:
        return pd.NaT
    try:
        if ts_val.tzinfo is not None:
            return ts_val.tz_convert("UTC").tz_localize(None)
    except Exception:
        pass
    try:
        return ts_val.tz_localize(None)
    except Exception:
        return ts_val


def _parse_ts(rec: Dict[str, Any]) -> pd.Timestamp:
    ts_raw = rec.get("ts") or rec.get("event_ts") or rec.get("time")
    try:
        return _normalize_ts(pd.Timestamp(ts_raw)) if ts_raw else pd.NaT
    except Exception:
        return pd.NaT


def _record_trade_key(rec: Dict[str, Any]) -> str:
    return str(rec.get("trade_uid") or rec.get("trade_id") or "").strip()


def _record_io_features(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    io = rec.get("io_features")
    if io is None:
        io = (rec.get("io") or {}).get("io_features")
    if io is None:
        return None
    arr = np.asarray(io, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _augment_io_features_for_contract(
    rec: Dict[str, Any],
    arr: np.ndarray,
    *,
    exit_io_version: str,
    feature_count: int,
) -> np.ndarray:
    if int(arr.shape[-1]) == int(feature_count):
        return arr
    target_dim = int(EXIT_IO_V1_CTX36_FEATURE_COUNT + len(EXIT_IO_V3_CTX36_M1L512_PHASE5_EXTRA_FEATURES))
    if (
        str(exit_io_version) == EXIT_IO_V3_CTX36_M1L512_PHASE5_IO_VERSION
        and int(arr.shape[-1]) == int(EXIT_IO_V1_CTX36_FEATURE_COUNT)
        and int(feature_count) == target_dim
    ):
        phase_onehot = np.asarray(compute_m5_phase_onehot(_parse_ts(rec)), dtype=np.float32)
        phase_block = np.repeat(phase_onehot.reshape(1, -1), int(arr.shape[0]), axis=0)
        return np.concatenate([arr.astype(np.float32, copy=False), phase_block], axis=1)
    raise RuntimeError(
        f"[EXIT_SHARD_DIM_FAIL] expected_input_dim={feature_count} got_input_dim={int(arr.shape[-1])}"
    )


def _sample_split(ts_val: pd.Timestamp, train_cutoff: pd.Timestamp, val_cutoff: pd.Timestamp) -> str:
    if ts_val is pd.NaT:
        return "train"
    if ts_val < train_cutoff:
        return "train"
    if ts_val < val_cutoff:
        return "val"
    return "test"


def _hash_bucket(key: str, num_buckets: int) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(num_buckets)


def _hash_unit_interval(key: str) -> float:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    raw = int.from_bytes(digest, "little")
    return float(raw) / float((1 << 64) - 1)


def _sample_split_trade_hash(key: str, train_ratio: float, val_ratio: float) -> str:
    r = _hash_unit_interval(key)
    if r < float(train_ratio):
        return "train"
    if r < float(train_ratio + val_ratio):
        return "val"
    return "test"


def _record_main_label(rec: Dict[str, Any]) -> float:
    try:
        return float(rec.get("should_exit", 0.0) or 0.0)
    except Exception:
        return 0.0


def _read_focus_negative_backfill_cfg() -> Tuple[float, int]:
    try:
        ratio = float(os.environ.get("GX1_EXIT_FOCUS_MIN_MAIN_NEG_RATIO", "0.05"))
    except Exception:
        ratio = 0.05
    try:
        abs_min = int(os.environ.get("GX1_EXIT_FOCUS_MIN_MAIN_NEG_ABS", "64"))
    except Exception:
        abs_min = 64
    ratio = min(max(float(ratio), 0.0), 0.50)
    abs_min = max(1, int(abs_min))
    return ratio, abs_min


def _focus_negative_priority(rec: Dict[str, Any], *, window_len: int) -> Optional[Tuple[float, float, float, float, float]]:
    arr = _record_io_features(rec)
    if arr is None or int(arr.shape[0]) < int(window_len):
        return None
    row = np.asarray(arr[-1], dtype=np.float32)
    try:
        mfe_bps = float(row[exit_v0.exit_feature_index_v1_ctx36("mfe_bps")])
        dd_bps = float(row[exit_v0.exit_feature_index_v1_ctx36("dd_from_mfe_bps")])
        giveback_ratio = float(row[exit_v0.exit_feature_index_v1_ctx36("giveback_ratio")])
        pnl_bps_now = float(row[exit_v0.exit_feature_index_v1_ctx36("pnl_bps_now")])
        bars_held = float(row[exit_v0.exit_feature_index_v1_ctx36("bars_held")])
    except Exception:
        return (0.0, -1.0, 0.0, 0.0, float(arr.shape[0]))
    if not all(math.isfinite(v) for v in (mfe_bps, dd_bps, giveback_ratio, pnl_bps_now, bars_held)):
        return (0.0, -1.0, 0.0, 0.0, float(arr.shape[0]))
    try:
        focus_mfe_min = max(1.0, float(os.environ.get("GX1_EXIT_FOCUS_MFE_MIN", "8.0")))
    except Exception:
        focus_mfe_min = 8.0
    try:
        focus_dd_min = max(1.0, float(os.environ.get("GX1_EXIT_FOCUS_DD_MIN", "8.0")))
    except Exception:
        focus_dd_min = 8.0
    try:
        focus_gb_min = max(1e-6, float(os.environ.get("GX1_EXIT_FOCUS_GB_MIN", "0.20")))
    except Exception:
        focus_gb_min = 0.20
    try:
        focus_abs_pnl_min = max(1.0, float(os.environ.get("GX1_EXIT_FOCUS_ABS_PNL_MIN", "4.0")))
    except Exception:
        focus_abs_pnl_min = 4.0
    decision_score = max(
        mfe_bps / focus_mfe_min,
        dd_bps / focus_dd_min,
        giveback_ratio / focus_gb_min,
        abs(pnl_bps_now) / focus_abs_pnl_min,
    )
    ready = 1.0 if bars_held >= 2.0 else 0.0
    return (ready, decision_score, abs(pnl_bps_now), dd_bps, bars_held)


def _ensure_split_main_negative_coverage(
    *,
    split: str,
    raw_records: List[Dict[str, Any]],
    focused_records: List[Dict[str, Any]],
    window_len: int,
) -> List[Dict[str, Any]]:
    neg_ratio, neg_abs_min = _read_focus_negative_backfill_cfg()
    focused_main_pos = sum(1 for rec in focused_records if _record_main_label(rec) > 0.5)
    focused_main_neg = max(0, len(focused_records) - focused_main_pos)
    target_neg = max(int(math.ceil(float(focused_main_pos) * float(neg_ratio))), int(neg_abs_min))
    if focused_main_neg >= target_neg:
        return focused_records

    selected = {id(rec) for rec in focused_records}
    candidates: List[Tuple[Tuple[float, float, float, float, float], Dict[str, Any]]] = []
    for rec in raw_records:
        if id(rec) in selected:
            continue
        if _record_main_label(rec) > 0.5:
            continue
        priority = _focus_negative_priority(rec, window_len=window_len)
        if priority is None:
            continue
        candidates.append((priority, rec))
    if not candidates:
        log.warning(
            "[EXIT_SHARD_FOCUS_BACKFILL_EMPTY] split=%s focused_records=%d focused_main_pos=%d focused_main_neg=%d target_neg=%d raw_records=%d",
            split,
            len(focused_records),
            focused_main_pos,
            focused_main_neg,
            target_neg,
            len(raw_records),
        )
        return focused_records

    candidates.sort(key=lambda item: item[0], reverse=True)
    need = max(0, target_neg - focused_main_neg)
    added = [rec for _, rec in candidates[:need]]
    if added:
        log.info(
            "[EXIT_SHARD_FOCUS_BACKFILL] split=%s added_main_neg=%d target_neg=%d focused_main_pos=%d focused_main_neg_before=%d candidate_neg=%d",
            split,
            len(added),
            target_neg,
            focused_main_pos,
            focused_main_neg,
            len(candidates),
        )
    return focused_records + added


def _bucketize_dataset(input_path: Path, bucket_dir: Path, num_buckets: int) -> Dict[str, Any]:
    bucket_dir.mkdir(parents=True, exist_ok=True)
    meta_path = bucket_dir / "bucket_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())

    handles = [open(bucket_dir / f"bucket_{idx:04d}.jsonl", "w", encoding="utf-8") for idx in range(num_buckets)]
    total_rows = 0
    min_ts = pd.NaT
    max_ts = pd.NaT
    started = time.time()
    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = _record_trade_key(rec)
                bucket_idx = _hash_bucket(tid or f"line_{line_no}", num_buckets)
                handles[bucket_idx].write(line)
                handles[bucket_idx].write("\n")
                total_rows += 1
                ts_val = _parse_ts(rec)
                if ts_val is not pd.NaT:
                    if min_ts is pd.NaT or ts_val < min_ts:
                        min_ts = ts_val
                    if max_ts is pd.NaT or ts_val > max_ts:
                        max_ts = ts_val
                if (line_no % 50000) == 0:
                    log.info(
                        "[EXIT_SHARD_BUCKETIZE] rows=%d elapsed_sec=%.1f min_ts=%s max_ts=%s",
                        total_rows,
                        time.time() - started,
                        str(min_ts),
                        str(max_ts),
                    )
    finally:
        for fh in handles:
            fh.close()

    meta = {
        "input_path": str(input_path),
        "num_buckets": int(num_buckets),
        "total_rows": int(total_rows),
        "min_ts": None if min_ts is pd.NaT else str(min_ts),
        "max_ts": None if max_ts is pd.NaT else str(max_ts),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _file_sha_or_na(path_str: Optional[str]) -> str:
    if not path_str:
        return "NA"
    path = Path(path_str)
    if not path.exists():
        return "MISSING"
    return _sha256_file(path)


def _dataset_cache_identity(input_path: Path) -> Dict[str, Any]:
    st = input_path.stat()
    return {
        "path": str(input_path.resolve()),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _current_shard_cache_contract(
    *,
    input_path: Path,
    exit_io_version: str,
    window_len: int,
    num_buckets: int,
    stride: int,
) -> Dict[str, Any]:
    env_keys = [
        "GX1_EXIT_SPLIT_MODE",
        "GX1_EXIT_SPLIT_TRAIN_RATIO",
        "GX1_EXIT_SPLIT_VAL_RATIO",
        "GX1_EXIT_LABEL_VARIANT",
        "GX1_EXIT_LABEL_SWEEP_PROFILE",
        "GX1_EXIT_MAIN_LABEL_PROFILE",
        "GX1_EXIT_FOCUS_MFE_MIN",
        "GX1_EXIT_FOCUS_DD_MIN",
        "GX1_EXIT_FOCUS_GB_MIN",
        "GX1_EXIT_FOCUS_ABS_PNL_MIN",
        "GX1_EXIT_FOCUS_BG_STRIDE",
        "GX1_EXIT_FOCUS_MIN_MAIN_NEG_RATIO",
        "GX1_EXIT_FOCUS_MIN_MAIN_NEG_ABS",
    ]
    env_surface = {key: str(os.environ.get(key, "")) for key in env_keys}
    contract = {
        "dataset": _dataset_cache_identity(input_path),
        "exit_io_version": str(exit_io_version),
        "window_len": int(window_len),
        "num_buckets": int(num_buckets),
        "stride": int(stride),
        "env_surface": env_surface,
        "source_hashes": {
            "train_exit_transformer_v0_sharded.py": _file_sha_or_na(__file__),
            "exit_transformer_v0.py": _file_sha_or_na(getattr(exit_v0, "__file__", None)),
        },
    }
    contract_json = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    contract["cache_contract_hash"] = hashlib.sha256(contract_json.encode("utf-8")).hexdigest()
    return contract


def _build_shards(
    *,
    input_path: Path,
    work_dir: Path,
    exit_io_version: str,
    window_len: int,
    num_buckets: int,
    stride: int,
) -> Dict[str, Any]:
    shard_meta_path = work_dir / "shard_meta.json"
    current_cache_contract = _current_shard_cache_contract(
        input_path=input_path,
        exit_io_version=exit_io_version,
        window_len=window_len,
        num_buckets=num_buckets,
        stride=stride,
    )
    if shard_meta_path.exists():
        cached_meta = json.loads(shard_meta_path.read_text())
        cached_hash = str(cached_meta.get("cache_contract_hash") or "")
        current_hash = str(current_cache_contract["cache_contract_hash"])
        cached_shards_dir = Path(str(cached_meta.get("shards_dir") or ""))
        if cached_hash == current_hash and cached_shards_dir.exists():
            log.info(
                "[EXIT_SHARD_CACHE_HIT] work_dir=%s cache_contract_hash=%s shards_dir=%s",
                str(work_dir),
                current_hash,
                str(cached_shards_dir),
            )
            return cached_meta
        log.info(
            "[EXIT_SHARD_CACHE_MISS] work_dir=%s cached_hash=%s current_hash=%s cached_shards_dir_exists=%d",
            str(work_dir),
            cached_hash or "NA",
            current_hash,
            int(cached_shards_dir.exists()),
        )

    bucket_dir = work_dir / "trade_buckets"
    shards_dir = work_dir / "npy_shards"
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    bucket_meta = _bucketize_dataset(input_path, bucket_dir, num_buckets)
    min_ts = _normalize_ts(pd.Timestamp(bucket_meta["min_ts"]))
    max_ts = _normalize_ts(pd.Timestamp(bucket_meta["max_ts"]))
    if min_ts is pd.NaT or max_ts is pd.NaT or max_ts <= min_ts:
        raise RuntimeError("[EXIT_SHARD_SPLIT_FAIL] invalid min/max ts")
    duration_ns = int(max_ts.value - min_ts.value)
    train_cutoff = pd.Timestamp(min_ts.value + int(duration_ns * 0.85))
    val_cutoff = pd.Timestamp(min_ts.value + int(duration_ns * 0.95))
    split_mode = str(os.environ.get("GX1_EXIT_SPLIT_MODE", "TIME_RECORD")).strip().upper() or "TIME_RECORD"
    try:
        train_ratio = float(os.environ.get("GX1_EXIT_SPLIT_TRAIN_RATIO", "0.85"))
    except Exception:
        train_ratio = 0.85
    try:
        val_ratio = float(os.environ.get("GX1_EXIT_SPLIT_VAL_RATIO", "0.10"))
    except Exception:
        val_ratio = 0.10
    train_ratio = min(max(float(train_ratio), 0.50), 0.98)
    val_ratio = min(max(float(val_ratio), 0.01), 0.30)
    if (train_ratio + val_ratio) >= 0.995:
        val_ratio = max(0.01, 0.995 - train_ratio)

    contract = get_exit_io_contract(exit_io_version)
    feature_count = int(contract["feature_count"])

    split_stats: Dict[str, Dict[str, Any]] = {
        "train": {"rows": 0, "main_pos": 0, "profit_pos": 0, "profit_mask": 0, "shards": 0},
        "val": {"rows": 0, "main_pos": 0, "profit_pos": 0, "profit_mask": 0, "shards": 0},
        "test": {"rows": 0, "main_pos": 0, "profit_pos": 0, "profit_mask": 0, "shards": 0},
    }
    bucket_reports: List[Dict[str, Any]] = []
    worker_count = max(1, min(int(os.environ.get("GX1_EXIT_SHARD_WORKERS", "8")), os.cpu_count() or 1))
    bucket_args: List[Tuple[int, Path, Path, pd.Timestamp, pd.Timestamp, int, int, int, str, str, float, float]] = []
    for bucket_idx in range(num_buckets):
        bucket_path = bucket_dir / f"bucket_{bucket_idx:04d}.jsonl"
        if bucket_path.exists() and bucket_path.stat().st_size > 0:
            bucket_args.append(
                (
                    bucket_idx,
                    bucket_path,
                    shards_dir,
                    train_cutoff,
                    val_cutoff,
                    feature_count,
                    window_len,
                    stride,
                    exit_io_version,
                    split_mode,
                    train_ratio,
                    val_ratio,
                )
            )

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as ex:
        futs = [ex.submit(_process_bucket_file, *args) for args in bucket_args]
        for fut in concurrent.futures.as_completed(futs):
            bucket_report = fut.result()
            bucket_reports.append(bucket_report)
            for split, stats in bucket_report.get("splits", {}).items():
                split_stats[split]["rows"] += int(stats.get("rows", 0))
                split_stats[split]["main_pos"] += int(stats.get("main_pos", 0))
                split_stats[split]["profit_pos"] += int(stats.get("profit_pos", 0))
                split_stats[split]["profit_mask"] += int(stats.get("profit_mask", 0))
                split_stats[split]["shards"] += int(stats.get("shards", 0))
            log.info(
                "[EXIT_SHARD_BUILD] bucket=%d input_records=%d focused_records=%d splits=%s",
                int(bucket_report.get("bucket", -1)),
                int(bucket_report.get("input_records", 0)),
                int(bucket_report.get("focused_records", 0)),
                bucket_report.get("splits", {}),
            )

    shard_meta = {
        "input_path": str(input_path),
        "exit_io_version": str(exit_io_version),
        "window_len": int(window_len),
        "feature_count": int(feature_count),
        "num_buckets": int(num_buckets),
        "stride": int(stride),
        "train_cutoff": str(train_cutoff),
        "val_cutoff": str(val_cutoff),
        "split_mode": str(split_mode),
        "split_train_ratio": float(train_ratio),
        "split_val_ratio": float(val_ratio),
        "split_stats": split_stats,
        "bucket_reports": bucket_reports,
        "shards_dir": str(shards_dir),
        "cache_contract": current_cache_contract,
        "cache_contract_hash": str(current_cache_contract["cache_contract_hash"]),
    }
    shard_meta_path.write_text(json.dumps(shard_meta, indent=2), encoding="utf-8")
    return shard_meta


def _process_bucket_file(
    bucket_idx: int,
    bucket_path: Path,
    shards_dir: Path,
    train_cutoff: pd.Timestamp,
    val_cutoff: pd.Timestamp,
    feature_count: int,
    window_len: int,
    stride: int,
    exit_io_version: str,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, Any]:
    with bucket_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        return {"bucket": int(bucket_idx), "input_records": 0, "focused_records": 0, "splits": {}}

    exit_v0._attach_labels_to_exit_records(records)
    split_payloads: Dict[str, Dict[str, List[Any]]] = {
        "train": {"X": [], "y_main": [], "y_profit": [], "profit_mask": [], "w": [], "family_y": [], "profit_bucket": []},
        "val": {"X": [], "y_main": [], "y_profit": [], "profit_mask": [], "w": [], "family_y": [], "profit_bucket": []},
        "test": {"X": [], "y_main": [], "y_profit": [], "profit_mask": [], "w": [], "family_y": [], "profit_bucket": []},
    }
    trade_split_cache: Dict[str, str] = {}
    split_records: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    def _resolve_split(rec: Dict[str, Any]) -> str:
        if str(split_mode).upper() == "TRADE_HASH":
            tid = _record_trade_key(rec) or f"bucket_{bucket_idx}:{rec.get('ts') or rec.get('event_ts') or rec.get('time') or 'NA'}"
            split = trade_split_cache.get(tid)
            if split is None:
                split = _sample_split_trade_hash(tid, float(train_ratio), float(val_ratio))
                trade_split_cache[tid] = split
            return split
        return _sample_split(_parse_ts(rec), train_cutoff, val_cutoff)

    for rec in records:
        split_records[_resolve_split(rec)].append(rec)

    focused_total = 0
    for split, raw_split_records in split_records.items():
        focused_records = exit_v0._focus_exit_training_records(raw_split_records)
        focused_records = _ensure_split_main_negative_coverage(
            split=split,
            raw_records=raw_split_records,
            focused_records=focused_records,
            window_len=window_len,
        )
        focused_total += len(focused_records)
        split_main_pos_records = sum(1 for rec in focused_records if _record_main_label(rec) > 0.5)
        log.info(
            "[EXIT_SHARD_FOCUS_SPLIT] bucket=%d split=%s raw_records=%d focused_records=%d main_pos_records=%d main_neg_records=%d",
            int(bucket_idx),
            split,
            len(raw_split_records),
            len(focused_records),
            split_main_pos_records,
            max(0, len(focused_records) - split_main_pos_records),
        )
        for rec in focused_records:
            arr = _record_io_features(rec)
            if arr is None:
                continue
            arr = _augment_io_features_for_contract(
                rec,
                arr,
                exit_io_version=exit_io_version,
                feature_count=feature_count,
            )
            if arr.shape[0] < window_len:
                continue
            y_main = float(rec.get("should_exit", 0.0) or 0.0)
            y_profit = float(rec.get("profit_protect_should_exit", 0.0) or 0.0)
            profit_mask = float(rec.get("profit_protect_train_mask", 0.0) or 0.0)
            weight = float(rec.get("sample_weight", 1.0) or 1.0)
            aux_fam_name = str(rec.get("exit_aux_family") or "NEGATIVE").strip().upper()
            fam_id = exit_v0.EXIT_AUX_FAMILY_TO_ID.get(aux_fam_name, exit_v0.EXIT_AUX_FAMILY_IGNORE_INDEX)
            profit_bucket = str(rec.get("profit_head_bucket") or "OTHER").strip().upper()
            for start in range(0, arr.shape[0] - window_len + 1, stride):
                split_payloads[split]["X"].append(arr[start : start + window_len])
                split_payloads[split]["y_main"].append(y_main)
                split_payloads[split]["y_profit"].append(y_profit)
                split_payloads[split]["profit_mask"].append(profit_mask)
                split_payloads[split]["w"].append(weight)
                split_payloads[split]["family_y"].append(fam_id)
                split_payloads[split]["profit_bucket"].append(profit_bucket)

    bucket_report: Dict[str, Any] = {"bucket": int(bucket_idx), "input_records": len(records), "focused_records": int(focused_total), "splits": {}}
    for split, payload in split_payloads.items():
        n = len(payload["X"])
        if n == 0:
            continue
        shard_prefix = f"{split}_{bucket_idx:04d}"
        np.save(shards_dir / f"{shard_prefix}_X.npy", np.stack(payload["X"]).astype(np.float32))
        np.save(shards_dir / f"{shard_prefix}_y_main.npy", np.asarray(payload["y_main"], dtype=np.float32))
        np.save(shards_dir / f"{shard_prefix}_y_profit.npy", np.asarray(payload["y_profit"], dtype=np.float32))
        np.save(shards_dir / f"{shard_prefix}_profit_mask.npy", np.asarray(payload["profit_mask"], dtype=np.float32))
        np.save(shards_dir / f"{shard_prefix}_w.npy", np.asarray(payload["w"], dtype=np.float32))
        np.save(shards_dir / f"{shard_prefix}_family_y.npy", np.asarray(payload["family_y"], dtype=np.int64))
        (shards_dir / f"{shard_prefix}_profit_bucket.json").write_text(json.dumps(payload["profit_bucket"]), encoding="utf-8")
        bucket_report["splits"][split] = {
            "rows": int(n),
            "main_pos": int(np.sum(np.asarray(payload["y_main"], dtype=np.float32) > 0.5)),
            "profit_pos": int(np.sum(np.asarray(payload["y_profit"], dtype=np.float32) > 0.5)),
            "profit_mask": int(np.sum(np.asarray(payload["profit_mask"], dtype=np.float32) > 0.5)),
            "shards": 1,
        }
    return bucket_report


def _iter_split_shards(shards_dir: Path, split: str) -> Iterable[Dict[str, Any]]:
    for x_path in sorted(shards_dir.glob(f"{split}_*_X.npy")):
        stem = x_path.name[:-6]
        yield {
            "X": np.load(x_path, mmap_mode="r"),
            "y_main": np.load(shards_dir / f"{stem}_y_main.npy", mmap_mode="r"),
            "y_profit": np.load(shards_dir / f"{stem}_y_profit.npy", mmap_mode="r"),
            "profit_mask": np.load(shards_dir / f"{stem}_profit_mask.npy", mmap_mode="r"),
            "w": np.load(shards_dir / f"{stem}_w.npy", mmap_mode="r"),
            "family_y": np.load(shards_dir / f"{stem}_family_y.npy", mmap_mode="r"),
            "profit_bucket_path": shards_dir / f"{stem}_profit_bucket.json",
            "stem": stem,
        }


def _compute_train_class_weights(shards_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    main_pos = 0.0
    main_total = 0.0
    profit_pos = 0.0
    profit_total = 0.0
    for shard in _iter_split_shards(shards_dir, "train"):
        y_main = np.asarray(shard["y_main"], dtype=np.float32)
        y_profit = np.asarray(shard["y_profit"], dtype=np.float32)
        profit_mask = np.asarray(shard["profit_mask"], dtype=np.float32)
        main_pos += float(np.sum(y_main > 0.5))
        main_total += float(y_main.size)
        keep = profit_mask > 0.5
        if keep.any():
            profit_pos += float(np.sum(y_profit[keep] > 0.5))
            profit_total += float(np.sum(keep))
    main_neg = max(0.0, main_total - main_pos)
    profit_neg = max(0.0, profit_total - profit_pos)
    main_pos_weight = torch.tensor(main_neg / max(1.0, main_pos), dtype=torch.float32)
    profit_pos_weight = torch.tensor(profit_neg / max(1.0, profit_pos), dtype=torch.float32)
    return main_pos_weight, profit_pos_weight


def _validate_split_label_health(split_stats: Dict[str, Dict[str, Any]]) -> None:
    failures: List[str] = []
    for split in ("train", "val"):
        stats = split_stats.get(split, {})
        rows = int(stats.get("rows", 0))
        main_pos = int(stats.get("main_pos", 0))
        if rows <= 0:
            failures.append(f"{split}:rows=0")
            continue
        if main_pos <= 0:
            failures.append(f"{split}:main_has_no_positives")
        if main_pos >= rows:
            failures.append(f"{split}:main_has_no_negatives")
    if failures:
        raise RuntimeError(
            "[EXIT_SHARD_LABEL_DEGENERATE] "
            + " ".join(failures)
            + " split_stats="
            + json.dumps(split_stats, sort_keys=True)
        )


def _assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"[EXIT_TRAIN_NONFINITE_TENSOR] name={name}")


def _assert_finite_model_params(model: nn.Module, *, context: str) -> None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param.detach()).all():
            raise RuntimeError(f"[EXIT_TRAIN_NONFINITE_PARAM] context={context} param={name}")


def _batch_slices(n: int, batch_size: int, shuffle: bool, seed: int) -> Iterable[np.ndarray]:
    idx = np.arange(n, dtype=np.int64)
    if shuffle and n > 1:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def _should_exit_selection_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, precision_floor: float,
) -> Tuple[float, float, float, float]:
    """V3-4: consumption-matched should_exit metrics (import-free, no sklearn).

    Returns (precision, recall, f1) at the decision `threshold`, plus recall-at-precision-floor
    (max recall over a score sweep where precision >= floor; 0.0 if floor unmet or floor<=0).
    The live consumer thresholds the should_exit sigmoid, so these match deployment far better
    than the diluted 3-head focal-BCE loss the cement currently selects on.
    """
    yt = (np.asarray(y_true).reshape(-1) > 0.5).astype(np.int64)
    yp = np.asarray(y_prob).reshape(-1)
    pred = (yp >= float(threshold)).astype(np.int64)
    tp = int(((pred == 1) & (yt == 1)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    rec_at_floor = 0.0
    if precision_floor > 0.0 and yp.size > 0 and int(yt.sum()) > 0:
        order = np.argsort(-yp, kind="mergesort")          # high score first, stable
        ys = yt[order]
        tps = np.cumsum(ys)
        fps = np.cumsum(1 - ys)
        precs = tps / np.maximum(1, tps + fps)
        recs = tps / float(max(1, int(yt.sum())))
        ok = precs >= float(precision_floor)
        rec_at_floor = float(recs[ok].max()) if bool(ok.any()) else 0.0
    return float(prec), float(rec), float(f1), float(rec_at_floor)


def _evaluate_epoch(
    *,
    model: exit_v0.ExitTransformerV0,
    device: torch.device,
    shards_dir: Path,
    split: str,
    batch_size: int,
    loss_fn: nn.Module,
    profit_loss_fn: nn.Module,
    family_loss_fn: nn.Module,
    family_aux_loss_weight: float,
    profit_protect_head_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_main = 0.0
    total_profit = 0.0
    total_aux = 0.0
    total_count = 0
    # V3-4 (2026-06-03 cross-model scan): accumulate should_exit prob/label so we can report
    # CONSUMPTION-MATCHED metrics (F1 / recall-at-precision-floor at the decision threshold the
    # live >0.95 fail-safe actually uses), not just the diluted 3-head loss. Diagnostics by
    # default; only drives checkpoint selection when GX1_EXIT_CHECKPOINT_SELECTION_MODE != loss.
    _se_true: List[np.ndarray] = []
    _se_prob: List[np.ndarray] = []
    with torch.no_grad():
        for shard in _iter_split_shards(shards_dir, split):
            X = shard["X"]
            y_main = shard["y_main"]
            y_profit = shard["y_profit"]
            profit_mask = shard["profit_mask"]
            w = shard["w"]
            family_y = shard["family_y"]
            n = int(X.shape[0])
            for idx in _batch_slices(n, batch_size, shuffle=False, seed=0):
                bx = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(device)
                by = torch.from_numpy(np.asarray(y_main[idx], dtype=np.float32)).to(device)
                bp = torch.from_numpy(np.asarray(y_profit[idx], dtype=np.float32)).to(device)
                bpm = torch.from_numpy(np.asarray(profit_mask[idx], dtype=np.float32)).to(device)
                bw = torch.from_numpy(np.asarray(w[idx], dtype=np.float32)).to(device)
                bf = torch.from_numpy(np.asarray(family_y[idx], dtype=np.int64)).to(device)
                logits = model.forward_logits(bx)
                profit_logits = model.forward_profit_protect_logits(bx)
                family_logits = model.forward_family_logits(bx)
                _assert_finite_tensor(f"{split}.logits", logits)
                _assert_finite_tensor(f"{split}.profit_logits", profit_logits)
                _assert_finite_tensor(f"{split}.family_logits", family_logits)
                main_loss = (loss_fn(logits, by) * bw).mean()
                mask = (bpm > 0.5).float()
                if float((bw * mask).sum().detach().cpu().item()) > 0.0:
                    profit_raw = profit_loss_fn(profit_logits, bp)
                    profit_loss = ((profit_raw * bw) * mask).sum() / (((bw * mask).sum()) + 1e-8)
                else:
                    profit_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
                family_raw = family_loss_fn(family_logits, bf)
                family_mask = (bf != exit_v0.EXIT_AUX_FAMILY_IGNORE_INDEX).float()
                if float((bw * family_mask).sum().detach().cpu().item()) > 0.0:
                    aux_loss = ((family_raw * bw) * family_mask).sum() / (((bw * family_mask).sum()) + 1e-8)
                else:
                    aux_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
                _assert_finite_tensor(f"{split}.main_loss", main_loss)
                _assert_finite_tensor(f"{split}.profit_loss", profit_loss)
                _assert_finite_tensor(f"{split}.aux_loss", aux_loss)
                bs = int(by.shape[0])
                total_count += bs
                total_main += float(main_loss.detach().cpu().item()) * bs
                total_profit += float(profit_loss.detach().cpu().item()) * bs
                total_aux += float(aux_loss.detach().cpu().item()) * bs
                _se_prob.append(torch.sigmoid(logits).detach().float().cpu().numpy().reshape(-1))
                _se_true.append(by.detach().float().cpu().numpy().reshape(-1))
    main_mean = total_main / max(1, total_count)
    profit_mean = total_profit / max(1, total_count)
    aux_mean = total_aux / max(1, total_count)
    total = main_mean + (profit_protect_head_loss_weight * profit_mean) + (family_aux_loss_weight * aux_mean)
    out = {
        "main_loss": float(main_mean),
        "profit_loss": float(profit_mean),
        "aux_loss": float(aux_mean),
        "total_loss": float(total),
        "n_samples": int(total_count),
    }
    # should_exit consumption-matched metrics (read threshold/floor from env; import-free).
    if _se_prob:
        _thr = float(os.environ.get("GX1_EXIT_CONSUMPTION_THRESHOLD", "0.5"))
        _pfloor = float(os.environ.get("GX1_EXIT_RECALL_PRECISION_FLOOR", "0.0"))
        _prec, _rec, _f1, _rec_at_floor = _should_exit_selection_metrics(
            np.concatenate(_se_true), np.concatenate(_se_prob), _thr, _pfloor,
        )
        out["should_exit_precision"] = _prec
        out["should_exit_recall"] = _rec
        out["should_exit_f1"] = _f1
        out["should_exit_recall_at_pfloor"] = _rec_at_floor
    return out


def _sample_audit_arrays(shards_dir: Path, split: str, limit: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    yp: List[np.ndarray] = []
    buckets: List[str] = []
    taken = 0
    for shard in _iter_split_shards(shards_dir, split):
        if taken >= limit:
            break
        X = shard["X"]
        y = shard["y_main"]
        y_profit = shard["y_profit"]
        n = min(int(X.shape[0]), max(0, limit - taken))
        if n <= 0:
            continue
        xs.append(np.asarray(X[:n], dtype=np.float32))
        ys.append(np.asarray(y[:n], dtype=np.float32))
        yp.append(np.asarray(y_profit[:n], dtype=np.float32))
        with open(shard["profit_bucket_path"], "r", encoding="utf-8") as f:
            names = json.load(f)
        buckets.extend(list(names[:n]))
        taken += n
    if not xs:
        return (
            np.zeros((0, 1, 1), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            [],
        )
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(yp, axis=0), buckets


def train_sharded(
    *,
    dataset_path: Path,
    work_dir: Path,
    out_dir: Path,
    exit_io_version: str,
    window_len: int,
    num_buckets: int,
    stride: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    shard_meta = _build_shards(
        input_path=dataset_path,
        work_dir=work_dir,
        exit_io_version=exit_io_version,
        window_len=window_len,
        num_buckets=num_buckets,
        stride=stride,
    )
    shards_dir = Path(shard_meta["shards_dir"])
    split_stats = shard_meta["split_stats"]
    if int(split_stats["train"]["rows"]) <= 0 or int(split_stats["val"]["rows"]) <= 0:
        raise RuntimeError(f"[EXIT_SHARD_TRAIN_FAIL] invalid split sizes: {split_stats}")
    _validate_split_label_health(split_stats)

    _make_deterministic(seed)
    device_pref = os.environ.get("GX1_EXIT_TRAIN_DEVICE", "").strip().lower()
    if device_pref in ("cuda", "cuda:0", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("[EXIT_TRAIN_DEVICE] CUDA requested but not available")
        device = torch.device("cuda")
    elif device_pref in ("cpu", ""):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_pref)

    contract = get_exit_io_contract(exit_io_version)
    model = exit_v0.ExitTransformerV0(
        input_dim=int(contract["feature_count"]),
        window_len=int(window_len),
        d_model=int(d_model),
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        family_head_dim=len(exit_v0.EXIT_AUX_FAMILY_NAMES),
        profit_protect_head_dim=1,
    ).to(device)
    log.info(
        "[EXIT_SHARD_TRAIN_DEVICE] device=%s input_dim=%d window_len=%d feature_hash=%s",
        str(device),
        int(contract["feature_count"]),
        int(window_len),
        str(contract["feature_hash"]),
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    main_pos_weight, profit_pos_weight = _compute_train_class_weights(shards_dir)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=main_pos_weight.to(device), reduction="none")
    profit_loss_fn = nn.BCEWithLogitsLoss(pos_weight=profit_pos_weight.to(device), reduction="none")
    family_loss_fn = nn.CrossEntropyLoss(ignore_index=exit_v0.EXIT_AUX_FAMILY_IGNORE_INDEX, reduction="none")
    try:
        family_aux_loss_weight = float(
            os.environ.get("GX1_EXIT_FAMILY_AUX_LOSS_WEIGHT", str(exit_v0.DEFAULT_FAMILY_AUX_LOSS_WEIGHT))
        )
    except Exception:
        family_aux_loss_weight = float(exit_v0.DEFAULT_FAMILY_AUX_LOSS_WEIGHT)
    try:
        profit_protect_head_loss_weight = float(
            os.environ.get("GX1_EXIT_PROFIT_PROTECT_HEAD_LOSS_WEIGHT", str(exit_v0.DEFAULT_PROFIT_PROTECT_HEAD_LOSS_WEIGHT))
        )
    except Exception:
        profit_protect_head_loss_weight = float(exit_v0.DEFAULT_PROFIT_PROTECT_HEAD_LOSS_WEIGHT)
    early_stop = os.environ.get("GX1_EXIT_TRAIN_EARLY_STOP", "0").strip() == "1"
    try:
        early_stop_patience = int(os.environ.get("GX1_EXIT_TRAIN_EARLY_STOP_PATIENCE", "5"))
    except Exception:
        early_stop_patience = 5
    try:
        early_stop_min_delta = float(os.environ.get("GX1_EXIT_TRAIN_EARLY_STOP_MIN_DELTA", "0.0"))
    except Exception:
        early_stop_min_delta = 0.0
    # V3-4: which scalar drives "best checkpoint" + early-stop. Default "loss" = diluted
    # 3-head total_loss = EXACT cement behavior (bit-parity). "f1" / "recall_at_pfloor" select
    # on the consumption-matched should_exit metric the live >0.95 fail-safe actually reads
    # (higher=better, so we minimise its negative through the unchanged `<` logic below).
    _ckpt_sel_mode = os.environ.get("GX1_EXIT_CHECKPOINT_SELECTION_MODE", "loss").strip().lower()
    if _ckpt_sel_mode not in ("loss", "f1", "recall_at_pfloor"):
        raise ValueError(
            f"GX1_EXIT_CHECKPOINT_SELECTION_MODE='{_ckpt_sel_mode}' invalid; "
            f"use loss|f1|recall_at_pfloor"
        )

    pin_memory = False if _is_wsl() else bool(device.type == "cuda")
    log.info(
        "[EXIT_SHARD_TRAIN_CONFIG] epochs=%d batch_size=%d lr=%.6f early_stop=%d patience=%d min_delta=%.6f pin_memory=%d train_rows=%d val_rows=%d test_rows=%d",
        int(epochs),
        int(batch_size),
        float(lr),
        int(early_stop),
        int(early_stop_patience),
        float(early_stop_min_delta),
        int(pin_memory),
        int(split_stats["train"]["rows"]),
        int(split_stats["val"]["rows"]),
        int(split_stats["test"]["rows"]),
    )

    best_state = None
    best_vloss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    train_main_loss_last = math.nan
    train_profit_loss_last = math.nan
    train_aux_loss_last = math.nan
    val_main_loss_last = math.nan
    val_profit_loss_last = math.nan
    val_aux_loss_last = math.nan
    stopped_early = False

    for epoch in range(epochs):
        model.train()
        epoch_main_loss_sum = 0.0
        epoch_profit_loss_sum = 0.0
        epoch_aux_loss_sum = 0.0
        epoch_sample_count = 0
        for shard_idx, shard in enumerate(_iter_split_shards(shards_dir, "train")):
            X = shard["X"]
            y_main = shard["y_main"]
            y_profit = shard["y_profit"]
            profit_mask = shard["profit_mask"]
            w = shard["w"]
            family_y = shard["family_y"]
            n = int(X.shape[0])
            for batch_no, idx in enumerate(_batch_slices(n, batch_size, shuffle=True, seed=seed + epoch + shard_idx), start=1):
                bx = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(device, non_blocking=pin_memory)
                by = torch.from_numpy(np.asarray(y_main[idx], dtype=np.float32)).to(device, non_blocking=pin_memory)
                bp = torch.from_numpy(np.asarray(y_profit[idx], dtype=np.float32)).to(device, non_blocking=pin_memory)
                bpm = torch.from_numpy(np.asarray(profit_mask[idx], dtype=np.float32)).to(device, non_blocking=pin_memory)
                bw = torch.from_numpy(np.asarray(w[idx], dtype=np.float32)).to(device, non_blocking=pin_memory)
                bf = torch.from_numpy(np.asarray(family_y[idx], dtype=np.int64)).to(device, non_blocking=pin_memory)
                opt.zero_grad(set_to_none=True)
                logits = model.forward_logits(bx)
                profit_logits = model.forward_profit_protect_logits(bx)
                family_logits = model.forward_family_logits(bx)
                _assert_finite_tensor("train.logits", logits)
                _assert_finite_tensor("train.profit_logits", profit_logits)
                _assert_finite_tensor("train.family_logits", family_logits)
                main_loss = (loss_fn(logits, by) * bw).mean()
                profit_mask_batch = (bpm > 0.5).float()
                if float((bw * profit_mask_batch).sum().detach().cpu().item()) > 0.0:
                    profit_raw = profit_loss_fn(profit_logits, bp)
                    profit_loss = ((profit_raw * bw) * profit_mask_batch).sum() / (((bw * profit_mask_batch).sum()) + 1e-8)
                else:
                    profit_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
                family_raw = family_loss_fn(family_logits, bf)
                family_mask = (bf != exit_v0.EXIT_AUX_FAMILY_IGNORE_INDEX).float()
                if float((bw * family_mask).sum().detach().cpu().item()) > 0.0:
                    aux_loss = ((family_raw * bw) * family_mask).sum() / (((bw * family_mask).sum()) + 1e-8)
                else:
                    aux_loss = torch.zeros((), dtype=main_loss.dtype, device=main_loss.device)
                _assert_finite_tensor("train.main_loss", main_loss)
                _assert_finite_tensor("train.profit_loss", profit_loss)
                _assert_finite_tensor("train.aux_loss", aux_loss)
                loss = (
                    main_loss
                    + (float(profit_protect_head_loss_weight) * profit_loss)
                    + (float(family_aux_loss_weight) * aux_loss)
                )
                _assert_finite_tensor("train.total_loss", loss)
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        raise RuntimeError(f"[EXIT_TRAIN_NONFINITE_GRAD] param={name}")
                opt.step()
                _assert_finite_model_params(model, context="post_step")
                bs = int(by.shape[0])
                epoch_sample_count += bs
                epoch_main_loss_sum += float(main_loss.detach().cpu().item()) * bs
                epoch_profit_loss_sum += float(profit_loss.detach().cpu().item()) * bs
                epoch_aux_loss_sum += float(aux_loss.detach().cpu().item()) * bs

        train_main_loss_last = epoch_main_loss_sum / max(1, epoch_sample_count)
        train_profit_loss_last = epoch_profit_loss_sum / max(1, epoch_sample_count)
        train_aux_loss_last = epoch_aux_loss_sum / max(1, epoch_sample_count)
        val_metrics = _evaluate_epoch(
            model=model,
            device=device,
            shards_dir=shards_dir,
            split="val",
            batch_size=batch_size,
            loss_fn=loss_fn,
            profit_loss_fn=profit_loss_fn,
            family_loss_fn=family_loss_fn,
            family_aux_loss_weight=family_aux_loss_weight,
            profit_protect_head_loss_weight=profit_protect_head_loss_weight,
        )
        val_main_loss_last = float(val_metrics["main_loss"])
        val_profit_loss_last = float(val_metrics["profit_loss"])
        val_aux_loss_last = float(val_metrics["aux_loss"])
        # V3-4: selection scalar. "loss" -> total_loss (cement bit-parity); the metric modes
        # MINIMISE the negative of a higher-is-better should_exit metric so the `<` logic and
        # min_delta semantics below are reused unchanged.
        if _ckpt_sel_mode == "f1":
            vloss_epoch = -float(val_metrics.get("should_exit_f1", 0.0))
        elif _ckpt_sel_mode == "recall_at_pfloor":
            vloss_epoch = -float(val_metrics.get("should_exit_recall_at_pfloor", 0.0))
        else:
            vloss_epoch = float(val_metrics["total_loss"])
        log.info(
            "[EXIT_SHARD_EPOCH] epoch=%d/%d train_main=%.6f train_profit=%.6f train_aux=%.6f "
            "val_main=%.6f val_profit=%.6f val_aux=%.6f val_total=%.6f "
            "se_prec=%.4f se_rec=%.4f se_f1=%.4f se_rec@pfloor=%.4f sel_mode=%s sel_val=%.6f",
            epoch + 1,
            epochs,
            train_main_loss_last,
            train_profit_loss_last,
            train_aux_loss_last,
            val_main_loss_last,
            val_profit_loss_last,
            val_aux_loss_last,
            float(val_metrics["total_loss"]),
            float(val_metrics.get("should_exit_precision", float("nan"))),
            float(val_metrics.get("should_exit_recall", float("nan"))),
            float(val_metrics.get("should_exit_f1", float("nan"))),
            float(val_metrics.get("should_exit_recall_at_pfloor", float("nan"))),
            _ckpt_sel_mode,
            vloss_epoch,
        )
        if early_stop:
            if vloss_epoch + early_stop_min_delta < best_vloss:
                best_vloss = vloss_epoch
                best_epoch = epoch + 1
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    log.info(
                        "[EXIT_SHARD_EARLY_STOP] epoch=%d best_epoch=%d best_vloss=%.6f patience=%d",
                        epoch + 1,
                        best_epoch,
                        best_vloss,
                        early_stop_patience,
                    )
                    stopped_early = True
                    break
        else:
            best_vloss = vloss_epoch
            best_epoch = epoch + 1

    if early_stop and best_state is not None:
        model.load_state_dict(best_state)
    _assert_finite_model_params(model, context="pre_test_eval")

    test_metrics = _evaluate_epoch(
        model=model,
        device=device,
        shards_dir=shards_dir,
        split="test",
        batch_size=batch_size,
        loss_fn=loss_fn,
        profit_loss_fn=profit_loss_fn,
        family_loss_fn=family_loss_fn,
        family_aux_loss_weight=family_aux_loss_weight,
        profit_protect_head_loss_weight=profit_protect_head_loss_weight,
    )
    _assert_finite_model_params(model, context="pre_save")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path, cfg_path, sha = exit_v0.save_exit_transformer_artifacts(
        model,
        out_dir,
        int(window_len),
        int(d_model),
        int(n_layers),
        feature_names_hash=str(contract["feature_hash"]),
        input_dim=int(contract["feature_count"]),
        exit_io_version=str(exit_io_version),
    )
    audit_train_X, audit_train_y, audit_train_profit_y, audit_train_buckets = _sample_audit_arrays(shards_dir, "train", limit=4096)
    audit_val_X, audit_val_y, audit_val_profit_y, audit_val_buckets = _sample_audit_arrays(shards_dir, "val", limit=4096)
    audit = exit_v0._audit_score_compression(
        model,
        audit_train_X,
        audit_train_y,
        audit_train_profit_y,
        np.ones_like(audit_train_profit_y, dtype=np.float32),
        audit_train_buckets,
        audit_val_X,
        audit_val_y,
        audit_val_profit_y,
        np.ones_like(audit_val_profit_y, dtype=np.float32),
        audit_val_buckets,
    )
    (out_dir / "SCORE_COMPRESSION_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    verify_result = exit_v0.verify_exit_transformer_artifacts(out_dir)
    (out_dir / "ARTIFACT_VERIFY.json").write_text(json.dumps(verify_result, indent=2), encoding="utf-8")

    report = {
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "dataset_kind": "sharded_streaming",
        "shard_meta_path": str(work_dir / "shard_meta.json"),
        "exit_ml_io_version": str(exit_io_version),
        "window_len": int(window_len),
        "feature_names_hash": str(contract["feature_hash"]),
        "n_samples": int(sum(int(v["rows"]) for v in split_stats.values())),
        "split_stats": split_stats,
        "epochs_requested": int(epochs),
        "epochs_completed": int(best_epoch if stopped_early else max(best_epoch, epochs)),
        "early_stopping_enabled": bool(early_stop),
        "early_stopped": bool(stopped_early),
        "early_stop_patience": int(early_stop_patience),
        "best_epoch": int(best_epoch),
        "val_loss": float(best_vloss),
        # V3-4: in "loss" mode best_vloss IS the diluted total_loss (cement-identical); in metric
        # modes it's the NEGATED should_exit metric the selection minimised — this field disambiguates.
        "checkpoint_selection_mode": str(_ckpt_sel_mode),
        "train_main_loss_last": float(train_main_loss_last),
        "train_profit_loss_last": float(train_profit_loss_last),
        "train_aux_loss_last": float(train_aux_loss_last),
        "val_main_loss_last": float(val_main_loss_last),
        "val_profit_loss_last": float(val_profit_loss_last),
        "val_aux_loss_last": float(val_aux_loss_last),
        "test_metrics": test_metrics,
        "score_compression_audit_sampled": audit,
        "model_path": str(model_path),
        "config_path": str(cfg_path),
        "model_sha256": str(sha),
        "artifact_verify": verify_result,
    }
    (out_dir / "TRAIN_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exit-io-version", default="EXIT_IO_V3_CTX36_M1L512_PHASE5")
    ap.add_argument("--window-len", type=int, default=512)
    ap.add_argument("--num-buckets", type=int, default=256)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--d-model", type=int, default=exit_v0.DEFAULT_D_MODEL)
    ap.add_argument("--n-layers", type=int, default=exit_v0.DEFAULT_N_LAYERS)
    ap.add_argument("--n-heads", type=int, default=exit_v0.DEFAULT_N_HEADS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    report = train_sharded(
        dataset_path=Path(args.dataset),
        work_dir=Path(args.work_dir),
        out_dir=Path(args.out_dir),
        exit_io_version=str(args.exit_io_version),
        window_len=int(args.window_len),
        num_buckets=int(args.num_buckets),
        stride=int(args.stride),
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        seed=int(args.seed),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
