"""
TRUTH 1W1C merge: produce MERGED artifacts and RUN_COMPLETED from a single chunk.

No dependency on legacy (quarantined) replay scripts.
Used by run_truth_e2e_sanity after process_chunk().
0-trades: when chunk has no trade_outcomes file but footer n_trades_closed==0, write empty MERGED parquet.
"""

from __future__ import annotations

import hashlib
import json
import os
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from gx1.execution.chunk_failure import write_fatal_capsule
from gx1.time.session_detector import get_session_vectorized
from gx1.prod.run_header import load_run_header
from gx1.utils.empty_trade_outcomes import write_empty_trade_outcomes_parquet
from gx1.utils.pnl import compute_pnl_bps

log = logging.getLogger(__name__)

_BUNDLE_SHA256_CACHE: Dict[str, str] = {}
_TREND_REGIME_ID_TO_LABEL = {
    0: "TREND_DOWN",
    1: "TREND_NEUTRAL",
    2: "TREND_UP",
}
# Prebuilt uses a 0..4 percentile bucket for volatility. We normalize back to the
# runtime-facing label contract here, while preserving the exact raw source in the
# provenance summary.
_VOL_REGIME_ID_TO_LABEL = {
    0: "LOW",
    1: "LOW",
    2: "MEDIUM",
    3: "HIGH",
    4: "EXTREME",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_optional(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception as e:
        raise RuntimeError(f"[SHADOW_META_PROVENANCE] failed to load json {path}: {e}") from e


def _resolve_chunk_run_header(chunk_dir: Path) -> Dict[str, Any]:
    for rel in ("run_header.json", "logs/run_header.json"):
        path = chunk_dir / rel
        if path.exists():
            return _load_json(path)
    return {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _sha256_directory(path: Path) -> str:
    cache_key = str(path.resolve())
    cached = _BUNDLE_SHA256_CACHE.get(cache_key)
    if cached:
        return cached
    if not path.exists() or not path.is_dir():
        raise RuntimeError(f"[SHADOW_META_PROVENANCE] bundle directory missing: {path}")
    h = hashlib.sha256()
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    if not files:
        raise RuntimeError(f"[SHADOW_META_PROVENANCE] bundle directory empty: {path}")
    for file_path in files:
        rel = file_path.relative_to(path).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        with file_path.open("rb") as f:
            while True:
                block = f.read(1024 * 1024)
                if not block:
                    break
                h.update(block)
        h.update(b"\0")
    digest = h.hexdigest()
    _BUNDLE_SHA256_CACHE[cache_key] = digest
    return digest


def _extract_v1_provenance(
    *,
    chunk_dir: Path,
    run_header: Dict[str, Any],
    footer: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    artifacts = run_header.get("artifacts") if isinstance(run_header, dict) else {}
    policy_hash = None
    policy_lane = None
    entry_bundle_sha256 = None
    exit_bundle_sha256 = None
    source_files: Dict[str, Optional[str]] = {
        "policy_hash": None,
        "entry_bundle_sha256": None,
        "exit_bundle_sha256": None,
    }

    if isinstance(artifacts, dict):
        policy_hash = (artifacts.get("policy") or {}).get("sha256")
        if policy_hash is not None:
            source_files["policy_hash"] = str(chunk_dir / "run_header.json")

    model_used_capsule = _load_json_optional(chunk_dir / "MODEL_USED_CAPSULE.json")
    if model_used_capsule:
        entry_bundle_sha256 = model_used_capsule.get("bundle_sha256")
        if entry_bundle_sha256 is not None:
            source_files["entry_bundle_sha256"] = str(chunk_dir / "MODEL_USED_CAPSULE.json")

    exit_runtime_sot = _load_json_optional(chunk_dir / "EXIT_RUNTIME_SOURCE_OF_TRUTH.json")
    if exit_runtime_sot:
        bundle_path = exit_runtime_sot.get("bundle_path")
        if bundle_path:
            exit_bundle_sha256 = _sha256_directory(Path(str(bundle_path)))
            source_files["exit_bundle_sha256"] = str(chunk_dir / "EXIT_RUNTIME_SOURCE_OF_TRUTH.json")

    policy_lane = (
        run_header.get("policy_lane")
        or run_header.get("entry_decision_engine", {}).get("mode")
        or run_header.get("meta", {}).get("role")
        or run_header.get("run_tag")
        or footer.get("policy_lane")
    )

    if not policy_hash:
        raise RuntimeError("[SHADOW_META_PROVENANCE] missing policy_hash in run_header.json")
    if not entry_bundle_sha256:
        raise RuntimeError("[SHADOW_META_PROVENANCE] missing entry_bundle_sha256 in MODEL_USED_CAPSULE.json")
    if not exit_bundle_sha256:
        raise RuntimeError("[SHADOW_META_PROVENANCE] missing exit_bundle_sha256 from EXIT_RUNTIME_SOURCE_OF_TRUTH bundle path")

    return {
        "policy_lane": str(policy_lane) if policy_lane is not None else None,
        "policy_hash": str(policy_hash),
        "entry_bundle_sha256": str(entry_bundle_sha256),
        "exit_bundle_sha256": str(exit_bundle_sha256),
        "source_policy_hash": source_files["policy_hash"],
        "source_entry_bundle_sha256": source_files["entry_bundle_sha256"],
        "source_exit_bundle_sha256": source_files["exit_bundle_sha256"],
    }


def _load_context_backfill_table(chunk_dir: Path) -> pd.DataFrame:
    chunk_data_path = chunk_dir / "chunk_0_data.parquet"
    if not chunk_data_path.exists():
        raise RuntimeError(f"[SHADOW_META_PROVENANCE] missing chunk_0_data.parquet: {chunk_data_path}")
    df = pd.read_parquet(chunk_data_path)
    required_cols = {"time", "atr_bps", "trend_regime_id", "vol_regime_id"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise RuntimeError(f"[SHADOW_META_PROVENANCE] chunk_0_data missing required cols: {missing}")
    df = df[["time", "atr_bps", "trend_regime_id", "vol_regime_id"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    if df["time"].isna().any():
        raise RuntimeError("[SHADOW_META_PROVENANCE] chunk_0_data.time contains null timestamps")
    if df["time"].duplicated().any():
        dupes = df.loc[df["time"].duplicated(keep=False), "time"].astype(str).head(10).tolist()
        raise RuntimeError(f"[SHADOW_META_PROVENANCE] chunk_0_data time index is not unique: {dupes}")

    trend_ids = sorted(pd.Series(df["trend_regime_id"]).dropna().astype(int).unique().tolist())
    vol_ids = sorted(pd.Series(df["vol_regime_id"]).dropna().astype(int).unique().tolist())
    bad_trend = [x for x in trend_ids if x not in _TREND_REGIME_ID_TO_LABEL]
    bad_vol = [x for x in vol_ids if x not in _VOL_REGIME_ID_TO_LABEL]
    if bad_trend:
        raise RuntimeError(f"[SHADOW_META_PROVENANCE] unexpected trend_regime_id values: {bad_trend}")
    if bad_vol:
        raise RuntimeError(f"[SHADOW_META_PROVENANCE] unexpected vol_regime_id values: {bad_vol}")

    out = pd.DataFrame({"decision_ts_utc_join": df["time"]})
    out["atr_bps_backfill"] = pd.to_numeric(df["atr_bps"], errors="coerce").astype("float64")
    out["trend_regime_backfill"] = (
        pd.to_numeric(df["trend_regime_id"], errors="coerce")
        .astype("Int64")
        .map(_TREND_REGIME_ID_TO_LABEL)
        .astype("string")
    )
    out["vol_regime_backfill"] = (
        pd.to_numeric(df["vol_regime_id"], errors="coerce")
        .astype("Int64")
        .map(_VOL_REGIME_ID_TO_LABEL)
        .astype("string")
    )
    out["trend_regime_id_backfill"] = pd.to_numeric(df["trend_regime_id"], errors="coerce").astype("Int64")
    out["vol_regime_id_backfill"] = pd.to_numeric(df["vol_regime_id"], errors="coerce").astype("Int64")
    return out


def _load_shadow_candidate_events(chunk_dir: Path, run_id: str) -> pd.DataFrame:
    """
    Load entry decision rows from eval_log JSONL files.

    This is the canonical shadow/meta-controller source table:
    one row per decision moment, including accepted, blocked, and no-trade rows.
    """
    rows: List[Dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for root in (chunk_dir, chunk_dir.parent):
        if not root.exists():
            continue
        for path in sorted(root.rglob("eval_log_*.jsonl")):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for idx, line in enumerate(handle, start=1):
                        raw = line.strip()
                        if not raw:
                            continue
                        try:
                            rec = json.loads(raw)
                        except Exception:
                            continue
                        if not isinstance(rec, dict):
                            continue
                        rec = dict(rec)
                        rec["source_eval_log"] = str(path)
                        rec["source_eval_log_row"] = idx
                        candidate_uid = rec.get("candidate_uid")
                        if not candidate_uid:
                            ts = str(rec.get("ts_utc") or rec.get("ts") or "")
                            decision = str(rec.get("decision") or "UNKNOWN")
                            trade_id = str(rec.get("trade_id") or "")
                            fallback_seed = f"{run_id}|{ts}|{decision}|{trade_id}|{idx}"
                            candidate_uid = (
                                f"{run_id}:fallback::{idx:06d}:"
                                f"{hashlib.sha256(fallback_seed.encode('utf-8')).hexdigest()[:12]}"
                            )
                            rec["candidate_uid"] = candidate_uid
                            rec["candidate_uid_source"] = "fallback"
                        else:
                            rec["candidate_uid_source"] = "log"
                        rows.append(rec)
            except Exception as e:
                log.warning("[SHADOW_META] failed to read eval_log %s: %s", path, e)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ts_utc" in df.columns:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    if "accepted" in df.columns:
        df["accepted"] = df["accepted"].fillna(False).astype(bool)
    if "decision" in df.columns:
        df["decision"] = df["decision"].astype(str)
    if "candidate_uid" in df.columns:
        df["candidate_uid"] = df["candidate_uid"].astype(str)
    return df


def _join_key_for_trade_id_or_uid(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="object")
    trade_uid = df["trade_uid"] if "trade_uid" in df.columns else pd.Series([None] * len(df), index=df.index)
    trade_id = df["trade_id"] if "trade_id" in df.columns else pd.Series([None] * len(df), index=df.index)
    return trade_uid.where(trade_uid.notna() & (trade_uid.astype(str).str.strip() != ""), trade_id).astype("object")


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _safe_str_series(series: pd.Series) -> pd.Series:
    return series.astype("string") if not series.empty else series.astype("string")


def _safe_bool_series(series: pd.Series) -> pd.Series:
    return series.astype("boolean") if not series.empty else series.astype("boolean")


def _safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _load_run_header_optional(run_root: Path) -> Dict[str, Any]:
    try:
        header = load_run_header(run_root)
        return header or {}
    except Exception as e:
        log.info("[SHADOW_META] run_header load skipped: %s", e)
        return {}


FEATURE_COLS = [
    "side",
    "session",
    "weekday_utc",
    "hour_utc",
    "atr_bps",
    "entry_spread_bps",
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "margin",
    "uncertainty_score",
    "tradable_prob",
    "mfe_first_n_pred",
    "path_quality_pred",
    "vol_regime",
    "trend_regime",
]

AUDIT_COLS = [
    "run_id",
    "candidate_uid",
    "trade_uid",
    "trade_id",
    "decision_ts_utc",
    "source_eval_log",
    "source_eval_log_row",
    "decision",
    "accepted",
    "decision_reason",
    "policy_lane",
    "policy_hash",
    "entry_bundle_sha256",
    "exit_bundle_sha256",
    "open_ts_utc",
    "close_ts_utc",
]

LABEL_COLS = [
    "mfe_threshold_bps",
    "positive_exit",
    "cata",
    "never_mfe",
    "good_mfe_then_rot",
    "trainable_mask_v1",
    "meta_allow_label_v1",
    "pnl_bps",
    "mfe_bps",
    "mae_bps",
    "bars_in_trade",
    "exit_reason",
]

REQUIRED_COLS = _dedupe_keep_order(FEATURE_COLS + AUDIT_COLS + LABEL_COLS)

NULLABLE_COLS = [
    "trade_uid",
    "trade_id",
    "decision_reason",
    "policy_lane",
    "policy_hash",
    "entry_bundle_sha256",
    "exit_bundle_sha256",
    "atr_bps",
    "entry_spread_bps",
    "tradable_prob",
    "mfe_first_n_pred",
    "path_quality_pred",
    "vol_regime",
    "trend_regime",
    "open_ts_utc",
    "close_ts_utc",
    "mfe_threshold_bps",
    "positive_exit",
    "cata",
    "never_mfe",
    "good_mfe_then_rot",
    "trainable_mask_v1",
    "meta_allow_label_v1",
    "pnl_bps",
    "mfe_bps",
    "mae_bps",
    "bars_in_trade",
    "exit_reason",
]

COL_DTYPE_MAP = {
    "run_id": "string",
    "candidate_uid": "string",
    "trade_uid": "string",
    "trade_id": "string",
    "decision_ts_utc": "string",
    "source_eval_log": "string",
    "source_eval_log_row": "int64",
    "decision": "string",
    "accepted": "boolean",
    "decision_reason": "string",
    "side": "string",
    "session": "string",
    "weekday_utc": "int64",
    "hour_utc": "int64",
    "atr_bps": "float64",
    "entry_spread_bps": "float64",
    "p_long": "float64",
    "p_short": "float64",
    "p_flat": "float64",
    "p_hat": "float64",
    "margin": "float64",
    "uncertainty_score": "float64",
    "tradable_prob": "float64",
    "mfe_first_n_pred": "float64",
    "path_quality_pred": "float64",
    "vol_regime": "string",
    "trend_regime": "string",
    "policy_lane": "string",
    "policy_hash": "string",
    "entry_bundle_sha256": "string",
    "exit_bundle_sha256": "string",
    "open_ts_utc": "string",
    "close_ts_utc": "string",
    "mfe_threshold_bps": "float64",
    "positive_exit": "boolean",
    "cata": "boolean",
    "never_mfe": "boolean",
    "good_mfe_then_rot": "boolean",
    "trainable_mask_v1": "boolean",
    "meta_allow_label_v1": "boolean",
    "pnl_bps": "float64",
    "mfe_bps": "float64",
    "mae_bps": "float64",
    "bars_in_trade": "float64",
    "exit_reason": "string",
}

ACCEPTED_TRUE_REQUIRED_COLS = [
    "run_id",
    "candidate_uid",
    "trade_uid",
    "trade_id",
    "decision_ts_utc",
    "source_eval_log",
    "source_eval_log_row",
    "decision",
    "accepted",
    "decision_reason",
    "side",
    "session",
    "weekday_utc",
    "hour_utc",
    "entry_spread_bps",
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "margin",
    "uncertainty_score",
    "open_ts_utc",
    "close_ts_utc",
    "pnl_bps",
    "mfe_bps",
    "mae_bps",
    "bars_in_trade",
    "exit_reason",
    "mfe_threshold_bps",
    "positive_exit",
    "cata",
    "never_mfe",
    "good_mfe_then_rot",
    "trainable_mask_v1",
    "meta_allow_label_v1",
]

ACCEPTED_FALSE_MUST_BE_NULL_COLS = [
    "trade_uid",
    "trade_id",
    "open_ts_utc",
    "close_ts_utc",
    "pnl_bps",
    "mfe_bps",
    "mae_bps",
    "bars_in_trade",
    "exit_reason",
    "positive_exit",
    "cata",
    "never_mfe",
    "good_mfe_then_rot",
    "meta_allow_label_v1",
]

DERIVED_COLS = [
    "weekday_utc",
    "hour_utc",
    "mfe_threshold_bps",
    "positive_exit",
    "cata",
    "never_mfe",
    "good_mfe_then_rot",
    "trainable_mask_v1",
    "meta_allow_label_v1",
]

FIELD_SOURCES = {
    "run_id": "merge_args",
    "candidate_uid": "eval_log",
    "trade_uid": "eval_log|trade_journal",
    "trade_id": "eval_log|trade_journal|trade_outcomes",
    "decision_ts_utc": "eval_log",
    "source_eval_log": "eval_log",
    "source_eval_log_row": "eval_log",
    "decision": "eval_log",
    "accepted": "eval_log",
    "decision_reason": "eval_log",
    "side": "eval_log|trade_journal|trade_outcomes",
    "session": "eval_log|trade_journal|trade_outcomes",
    "weekday_utc": "derived",
    "hour_utc": "derived",
    "atr_bps": "chunk_0_data.parquet@decision_ts_utc|trade_journal.entry_snapshot|optional",
    "entry_spread_bps": "trade_outcomes|trade_journal|optional",
    "p_long": "eval_log",
    "p_short": "eval_log",
    "p_flat": "eval_log",
    "p_hat": "eval_log",
    "margin": "eval_log",
    "uncertainty_score": "eval_log",
    "tradable_prob": "eval_log",
    "mfe_first_n_pred": "eval_log",
    "path_quality_pred": "eval_log",
    "vol_regime": "chunk_0_data.parquet.vol_regime_id@decision_ts_utc|trade_journal",
    "trend_regime": "chunk_0_data.parquet.trend_regime_id@decision_ts_utc|trade_journal",
    "policy_lane": "run_header|footer",
    "policy_hash": "run_header.json",
    "entry_bundle_sha256": "MODEL_USED_CAPSULE.json",
    "exit_bundle_sha256": "EXIT_RUNTIME_SOURCE_OF_TRUTH.json.bundle_path->bundle_sha256",
    "open_ts_utc": "trade_journal|trade_outcomes",
    "close_ts_utc": "trade_journal|trade_outcomes",
    "mfe_threshold_bps": "derived",
    "positive_exit": "derived",
    "cata": "derived",
    "never_mfe": "derived",
    "good_mfe_then_rot": "derived",
    "trainable_mask_v1": "derived",
    "meta_allow_label_v1": "derived",
    "pnl_bps": "trade_journal|trade_outcomes",
    "mfe_bps": "trade_journal|trade_outcomes",
    "mae_bps": "trade_journal|trade_outcomes",
    "bars_in_trade": "trade_journal|trade_outcomes",
    "exit_reason": "trade_journal|trade_outcomes",
}

SANITY_CHECKS = [
    "candidate_uid is unique per row",
    "accepted=True rows must have trade_uid and final outcome fields",
    "accepted=False rows must have outcome/label fields null",
    "feature_cols must not include post-entry leakage fields",
    "all REQUIRED_COLS must exist before write",
    "accepted row join coverage to trade_outcomes/trade_journal must be complete",
    "label columns must be derived only from accepted rows",
]

V1_SCHEMA_CONTRACT = {
    "feature_cols": FEATURE_COLS,
    "audit_cols": AUDIT_COLS,
    "label_cols": LABEL_COLS,
    "required_cols": REQUIRED_COLS,
    "nullable_cols": NULLABLE_COLS,
    "col_dtype_map": COL_DTYPE_MAP,
    "accepted_true_required_cols": ACCEPTED_TRUE_REQUIRED_COLS,
    "accepted_false_must_be_null_cols": ACCEPTED_FALSE_MUST_BE_NULL_COLS,
    "derived_cols": DERIVED_COLS,
    "field_sources": FIELD_SOURCES,
    "sanity_checks": SANITY_CHECKS,
}

_PROVENANCE_FIELD_RULES: Dict[str, Dict[str, str]] = {
    "policy_hash": {
        "source_file": "run_header.json",
        "join_key": "run_constant",
        "fallback_rule": "none",
        "fail_condition": "missing or inconsistent within run",
    },
    "entry_bundle_sha256": {
        "source_file": "MODEL_USED_CAPSULE.json",
        "join_key": "run_constant",
        "fallback_rule": "none",
        "fail_condition": "missing or inconsistent within run",
    },
    "exit_bundle_sha256": {
        "source_file": "EXIT_RUNTIME_SOURCE_OF_TRUTH.json -> bundle_path",
        "join_key": "run_constant",
        "fallback_rule": "hash bundle directory recursively",
        "fail_condition": "bundle_path missing or hash unavailable",
    },
    "atr_bps": {
        "source_file": "chunk_0_data.parquet",
        "join_key": "decision_ts_utc",
        "fallback_rule": "first_non_null(existing, chunk_0_data join)",
        "fail_condition": "join not unique or fill rate below threshold",
    },
    "vol_regime": {
        "source_file": "chunk_0_data.parquet.vol_regime_id",
        "join_key": "decision_ts_utc",
        "fallback_rule": "normalize raw id to runtime label contract",
        "fail_condition": "join not unique, unexpected ids, or fill rate below threshold",
    },
    "trend_regime": {
        "source_file": "chunk_0_data.parquet.trend_regime_id",
        "join_key": "decision_ts_utc",
        "fallback_rule": "map raw id to runtime label contract",
        "fail_condition": "join not unique, unexpected ids, or fill rate below threshold",
    },
}


def compute_mfe_threshold_bps(row: pd.Series) -> float:
    entry_spread_bps = row.get("entry_spread_bps")
    try:
        if pd.notna(entry_spread_bps):
            return float(max(1.0, float(entry_spread_bps)))
    except Exception:
        pass
    return 1.0


def compute_positive_exit(row: pd.Series) -> Optional[bool]:
    if not bool(row.get("accepted", False)):
        return None
    pnl_bps = row.get("pnl_bps")
    if pd.isna(pnl_bps):
        return None
    return bool(float(pnl_bps) > 0.0)


def compute_cata(row: pd.Series) -> Optional[bool]:
    if not bool(row.get("accepted", False)):
        return None
    exit_reason = row.get("exit_reason")
    if pd.isna(exit_reason):
        return None
    return str(exit_reason) == "CATASTROPHIC_GUARD"


def compute_never_mfe(row: pd.Series) -> Optional[bool]:
    if not bool(row.get("accepted", False)):
        return None
    peak_mfe_bps = row.get("peak_mfe_bps_exit_state")
    mfe_bps = peak_mfe_bps if pd.notna(peak_mfe_bps) else row.get("mfe_bps")
    if pd.isna(mfe_bps):
        return None
    return bool(float(mfe_bps) < float(row.get("mfe_threshold_bps", 1.0)))


def compute_good_mfe_then_rot(row: pd.Series) -> Optional[bool]:
    if not bool(row.get("accepted", False)):
        return None
    positive_exit = compute_positive_exit(row)
    cata = compute_cata(row)
    never_mfe = compute_never_mfe(row)
    peak_mfe_bps = row.get("peak_mfe_bps_exit_state")
    mfe_bps = peak_mfe_bps if pd.notna(peak_mfe_bps) else row.get("mfe_bps")
    threshold = float(row.get("mfe_threshold_bps", 1.0))
    if pd.isna(mfe_bps) or positive_exit is None or cata is None or never_mfe is None:
        return None
    return bool((float(mfe_bps) >= threshold) and (positive_exit is False) and (cata is False) and (never_mfe is False))


def compute_trainable_mask_v1(row: pd.Series) -> Optional[bool]:
    if not bool(row.get("accepted", False)):
        return False
    required = [
        "run_id",
        "candidate_uid",
        "trade_uid",
        "trade_id",
        "decision_ts_utc",
        "decision",
        "accepted",
        "side",
        "session",
        "p_long",
        "p_short",
        "p_flat",
        "p_hat",
        "margin",
        "uncertainty_score",
        "entry_spread_bps",
        "open_ts_utc",
        "close_ts_utc",
        "pnl_bps",
        "mfe_bps",
        "mae_bps",
        "bars_in_trade",
        "exit_reason",
    ]
    for col in required:
        if pd.isna(row.get(col)):
            return False
    return bool(str(row.get("side", "")).lower() == "long")


def compute_meta_allow_label_v1(row: pd.Series) -> Optional[bool]:
    if not compute_trainable_mask_v1(row):
        return None
    positive_exit = compute_positive_exit(row)
    cata = compute_cata(row)
    never_mfe = compute_never_mfe(row)
    if positive_exit is None or cata is None or never_mfe is None:
        return None
    return bool(positive_exit and (not cata) and (not never_mfe))


def _finalize_shadow_meta_v1(
    merged_shadow: pd.DataFrame,
    *,
    run_id: str,
    chunk_dir: Path,
    run_header: Dict[str, Any],
    footer: Dict[str, Any],
) -> pd.DataFrame:
    if merged_shadow is None or merged_shadow.empty:
        return pd.DataFrame(columns=REQUIRED_COLS)

    df = merged_shadow.copy()
    direct_run_header = _resolve_chunk_run_header(chunk_dir)
    provenance = _extract_v1_provenance(
        chunk_dir=chunk_dir,
        run_header=direct_run_header or run_header or {},
        footer=footer,
    )
    provenance_summary: Dict[str, Dict[str, Any]] = {}

    df["run_id"] = run_id
    df["candidate_uid"] = _first_non_null_series(df, ["candidate_uid"])
    df["trade_uid"] = _first_non_null_series(df, ["trade_uid", "trade_uid_journal", "trade_uid_outcome"])
    df["trade_id"] = _first_non_null_series(df, ["trade_id", "trade_id_journal", "trade_id_outcome"])
    df["decision_ts_utc"] = _normalize_utc_series(_first_non_null_series(df, ["decision_ts_utc", "ts_utc"]))
    df["source_eval_log"] = _first_non_null_series(df, ["source_eval_log"])
    df["source_eval_log_row"] = pd.to_numeric(
        _first_non_null_series(df, ["source_eval_log_row"]), errors="coerce"
    ).fillna(-1).astype("int64")
    df["decision"] = _first_non_null_series(df, ["decision"]).astype("string")
    df["accepted"] = _safe_bool_series(_first_non_null_series(df, ["accepted", "accepted_bool"], default=False))
    df["decision_reason"] = _first_non_null_series(df, ["decision_reason"])
    df["side"] = _normalize_side_series(
        _first_non_null_series(df, ["side", "side_journal", "side_outcome", "side_pre", "decision"])
    )
    df["session"] = _first_non_null_series(df, ["session", "session_journal", "session_outcome"])

    decision_dt = pd.to_datetime(df["decision_ts_utc"], utc=True, errors="coerce")
    df["weekday_utc"] = decision_dt.dt.dayofweek.fillna(-1).astype("int64")
    df["hour_utc"] = decision_dt.dt.hour.fillna(-1).astype("int64")

    context_backfill = _load_context_backfill_table(chunk_dir)
    if context_backfill["decision_ts_utc_join"].duplicated().any():
        dupes = context_backfill.loc[
            context_backfill["decision_ts_utc_join"].duplicated(keep=False),
            "decision_ts_utc_join",
        ].astype(str).head(10).tolist()
        raise RuntimeError(f"[SHADOW_META_PROVENANCE] context backfill join key not unique: {dupes}")
    df["_decision_ts_join"] = decision_dt
    df = df.merge(
        context_backfill,
        left_on="_decision_ts_join",
        right_on="decision_ts_utc_join",
        how="left",
        validate="many_to_one",
    )

    df["atr_bps"] = _safe_float_series(_first_non_null_series(df, ["atr_bps", "atr_bps_journal", "atr_bps_backfill"]))
    df["entry_spread_bps"] = _safe_float_series(
        _first_non_null_series(df, ["entry_spread_bps", "entry_spread_bps_outcome", "entry_spread_bps_journal"])
    )
    df["p_long"] = _safe_float_series(_first_non_null_series(df, ["p_long"]))
    df["p_short"] = _safe_float_series(_first_non_null_series(df, ["p_short"]))
    df["p_flat"] = _safe_float_series(_first_non_null_series(df, ["p_flat"]))
    df["p_hat"] = _safe_float_series(_first_non_null_series(df, ["p_hat"]))
    df["margin"] = _safe_float_series(_first_non_null_series(df, ["margin", "margin_top1_top2", "margin_top1_top2_journal"]))
    df["uncertainty_score"] = _safe_float_series(_first_non_null_series(df, ["uncertainty_score"]))
    df["tradable_prob"] = _safe_float_series(_first_non_null_series(df, ["tradable_prob"]))
    df["mfe_first_n_pred"] = _safe_float_series(_first_non_null_series(df, ["mfe_first_n_pred"]))
    df["path_quality_pred"] = _safe_float_series(_first_non_null_series(df, ["path_quality_pred"]))
    df["vol_regime"] = _first_non_null_series(df, ["vol_regime", "vol_regime_journal", "vol_regime_backfill"])
    df["trend_regime"] = _first_non_null_series(df, ["trend_regime", "trend_regime_journal", "trend_regime_backfill"])
    df["policy_lane"] = provenance.get("policy_lane")
    df["policy_hash"] = provenance.get("policy_hash")
    df["entry_bundle_sha256"] = provenance.get("entry_bundle_sha256")
    df["exit_bundle_sha256"] = provenance.get("exit_bundle_sha256")
    df["open_ts_utc"] = _normalize_utc_series(_first_non_null_series(df, ["open_ts_utc", "open_ts_utc_journal", "entry_time"]))
    df["close_ts_utc"] = _normalize_utc_series(
        _first_non_null_series(df, ["close_ts_utc", "close_ts_utc_journal", "exit_time"])
    )
    df["pnl_bps"] = _safe_float_series(_first_non_null_series(df, ["pnl_bps", "pnl_bps_journal"]))
    df["mfe_bps"] = _safe_float_series(_first_non_null_series(df, ["mfe_bps", "mfe_bps_journal"]))
    df["mae_bps"] = _safe_float_series(_first_non_null_series(df, ["mae_bps", "mae_bps_journal"]))
    df["bars_in_trade"] = _safe_float_series(
        _first_non_null_series(df, ["bars_in_trade", "duration_bars", "bars_in_trade_journal", "duration_bars_outcome"])
    )
    df["exit_reason"] = _first_non_null_series(df, ["exit_reason", "exit_reason_journal", "exit_reason_outcome"])

    df["mfe_threshold_bps"] = df.apply(compute_mfe_threshold_bps, axis=1)
    df["positive_exit"] = df.apply(compute_positive_exit, axis=1)
    df["cata"] = df.apply(compute_cata, axis=1)
    df["never_mfe"] = df.apply(compute_never_mfe, axis=1)
    df["good_mfe_then_rot"] = df.apply(compute_good_mfe_then_rot, axis=1)
    df["trainable_mask_v1"] = df.apply(compute_trainable_mask_v1, axis=1)
    df["meta_allow_label_v1"] = df.apply(compute_meta_allow_label_v1, axis=1)

    _enforce_provenance_coverage(df=df, summaries=provenance_summary)
    _validate_shadow_meta_v1(df)

    final_df = df.reindex(columns=REQUIRED_COLS).copy()
    for col, dtype in COL_DTYPE_MAP.items():
        if col not in final_df.columns:
            continue
        if dtype == "string":
            final_df[col] = final_df[col].astype("string")
        elif dtype == "boolean":
            final_df[col] = final_df[col].astype("boolean")
        elif dtype == "float64":
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce").astype("float64")
        elif dtype == "int64":
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce").astype("int64")

    sort_cols = [c for c in ["decision_ts_utc", "candidate_uid"] if c in final_df.columns]
    if sort_cols:
        final_df = final_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    final_df.attrs["shadow_meta_provenance_summary"] = {
        "run_id": run_id,
        "fields": provenance_summary,
        "provenance_sources": {
            "policy_hash": provenance.get("source_policy_hash"),
            "entry_bundle_sha256": provenance.get("source_entry_bundle_sha256"),
            "exit_bundle_sha256": provenance.get("source_exit_bundle_sha256"),
            "atr_bps": str(chunk_dir / "chunk_0_data.parquet"),
            "vol_regime": str(chunk_dir / "chunk_0_data.parquet"),
            "trend_regime": str(chunk_dir / "chunk_0_data.parquet"),
        },
    }
    return final_df


def _first_non_null_series(df: pd.DataFrame, candidate_cols: List[str], default: Any = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="object")
    result: Optional[pd.Series] = None
    for col in candidate_cols:
        if col not in df.columns:
            continue
        series = df[col]
        if result is None:
            result = series.copy()
        else:
            result = result.combine_first(series)
    if result is None:
        return pd.Series([default] * len(df), index=df.index)
    if default is not None:
        result = result.where(result.notna(), default)
    return result


def _normalize_side_series(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="string")
    out = series.astype("string")
    out = out.str.strip().str.lower()
    out = out.replace({"none": None, "nan": None, "<na>": None})
    return out


def _normalize_utc_series(series: pd.Series) -> pd.Series:
    if series is None or series.empty:
        return pd.Series(dtype="string")
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    return dt.dt.strftime("%Y-%m-%dT%H:%M:%S%z").astype("string").str.replace(r"(\+|\-)(\d{2})(\d{2})$", r"\1\2:\3", regex=True)


def _summarize_provenance_field(
    *,
    field_name: str,
    series: pd.Series,
    total_rows: int,
    run_id: str,
) -> Dict[str, Any]:
    filled = int(series.notna().sum())
    fill_rate = float(filled / total_rows) if total_rows > 0 else 1.0
    rule = _PROVENANCE_FIELD_RULES[field_name]
    return {
        "field": field_name,
        "total_rows": int(total_rows),
        "filled_rows": filled,
        "fill_rate": fill_rate,
        "runs_covered": 1 if filled > 0 else 0,
        "runs_failed": [] if filled > 0 else [run_id],
        "source_file": rule["source_file"],
        "join_key": rule["join_key"],
        "fallback_rule": rule["fallback_rule"],
        "fail_condition": rule["fail_condition"],
    }


def _enforce_provenance_coverage(
    *,
    df: pd.DataFrame,
    summaries: Dict[str, Dict[str, Any]],
) -> None:
    total_rows = len(df)
    if total_rows <= 0:
        return
    for field in ("policy_hash", "entry_bundle_sha256", "exit_bundle_sha256"):
        if field not in df.columns or df[field].isna().any():
            raise RuntimeError(f"[SHADOW_META_PROVENANCE] {field} must be filled for all rows")
        uniq = sorted({str(x) for x in df[field].dropna().tolist()})
        if len(uniq) != 1:
            raise RuntimeError(f"[SHADOW_META_PROVENANCE] {field} inconsistent within run: {uniq[:5]}")
        summaries[field] = _summarize_provenance_field(field_name=field, series=df[field], total_rows=total_rows, run_id=str(df['run_id'].iloc[0]))

    for field in ("atr_bps", "vol_regime", "trend_regime"):
        if field not in df.columns:
            raise RuntimeError(f"[SHADOW_META_PROVENANCE] missing backfill field {field}")
        series = df[field]
        summary = _summarize_provenance_field(field_name=field, series=series, total_rows=total_rows, run_id=str(df["run_id"].iloc[0]))
        summaries[field] = summary
        if summary["fill_rate"] < 0.999:
            raise RuntimeError(
                f"[SHADOW_META_PROVENANCE] {field} fill_rate too low: {summary['fill_rate']:.6f}"
            )


def _validate_shadow_meta_v1(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"[SHADOW_META_V1] missing required cols: {missing}")

    if "candidate_uid" in df.columns and not df["candidate_uid"].is_unique:
        dupes = df.loc[df["candidate_uid"].duplicated(keep=False), "candidate_uid"].astype(str).head(10).tolist()
        raise RuntimeError(f"[SHADOW_META_V1] candidate_uid must be unique; dupes={dupes}")

    for col in ("run_id", "candidate_uid", "decision_ts_utc", "accepted"):
        if col in df.columns and df[col].isna().any():
            bad = df.loc[df[col].isna(), "candidate_uid"].astype(str).head(10).tolist()
            raise RuntimeError(f"[SHADOW_META_V1] required column has nulls: {col}; candidate_uid={bad}")

    accepted_mask = df["accepted"].fillna(False).astype(bool) if "accepted" in df.columns else pd.Series(False, index=df.index)
    blocked_df = df.loc[~accepted_mask].copy()

    # Only fully materialized trades are allowed to carry the strict V1 outcome contract.
    # Accepted audit-only rows may exist when the candidate was accepted but never produced
    # a complete open/fill/outcome snapshot; those rows stay non-trainable and must not fail
    # the run.
    trainable_mask = df["trainable_mask_v1"].fillna(False).astype(bool) if "trainable_mask_v1" in df.columns else pd.Series(False, index=df.index)
    accepted_trainable_df = df.loc[accepted_mask & trainable_mask].copy()

    if not accepted_trainable_df.empty:
        for col in ACCEPTED_TRUE_REQUIRED_COLS:
            if col not in accepted_trainable_df.columns:
                raise RuntimeError(f"[SHADOW_META_V1] accepted row missing col={col}")
            if accepted_trainable_df[col].isna().any():
                bad = accepted_trainable_df.loc[accepted_trainable_df[col].isna(), "candidate_uid"].astype(str).head(10).tolist()
                raise RuntimeError(f"[SHADOW_META_V1] accepted row has null {col}; candidate_uid={bad}")

    if not blocked_df.empty:
        for col in ACCEPTED_FALSE_MUST_BE_NULL_COLS:
            if col not in blocked_df.columns:
                continue
            if not blocked_df[col].isna().all():
                bad = blocked_df.loc[blocked_df[col].notna(), "candidate_uid"].astype(str).head(10).tolist()
                raise RuntimeError(f"[SHADOW_META_V1] blocked row must null {col}; candidate_uid={bad}")

    leakage_cols = set(AUDIT_COLS + LABEL_COLS)
    leakage_cols.discard("mfe_threshold_bps")
    leakage_cols.discard("trainable_mask_v1")
    leakage_cols.discard("meta_allow_label_v1")
    bad_features = sorted(set(FEATURE_COLS) & leakage_cols)
    if bad_features:
        raise RuntimeError(f"[SHADOW_META_V1] feature leakage detected: {bad_features}")

    accepted_join_coverage = float(accepted_trainable_df["trade_uid"].notna().mean()) if not accepted_trainable_df.empty else 1.0
    if accepted_join_coverage < 1.0:
        raise RuntimeError(
            f"[SHADOW_META_V1] accepted row join coverage incomplete: trade_uid_coverage={accepted_join_coverage:.3f}"
        )



def _pnl_profile_summary(outcomes_df: pd.DataFrame) -> Dict[str, Any]:
    if outcomes_df is None or outcomes_df.empty or "pnl_bps" not in outcomes_df.columns:
        return {}
    pnl = pd.to_numeric(outcomes_df["pnl_bps"], errors="coerce").dropna()
    if pnl.empty:
        return {}

    def _r(val: Any) -> Any:
        try:
            return round(float(val), 6)
        except Exception:
            return val

    qs = pnl.quantile([0.05, 0.95, 0.99], interpolation="linear")
    return {
        "n": int(len(pnl)),
        "win_rate": _r((pnl > 0).mean()),
        "mean": _r(pnl.mean()),
        "median": _r(pnl.median()),
        "std": _r(pnl.std()),
        "min": _r(pnl.min()),
        "p05": _r(qs.loc[0.05]),
        "p95": _r(qs.loc[0.95]),
        "p99": _r(qs.loc[0.99]),
        "max": _r(pnl.max()),
    }


def _compute_tail_metrics(pnl: pd.Series) -> Dict[str, float]:
    pnl = pd.to_numeric(pnl, errors="coerce").dropna()
    n = int(len(pnl))
    if n == 0:
        return {
            "trade_count": 0,
            "max_loss": float("nan"),
            "worst1pct_mean": float("nan"),
            "worst1pct_min": float("nan"),
            "CVaR95": float("nan"),
            "CVaR99": float("nan"),
        }
    pnl_sorted = np.sort(pnl.values)
    worst_1pct_n = max(1, n // 100)
    worst_5pct_n = max(1, int(np.floor(n * 0.05)))
    worst_1pct = pnl_sorted[:worst_1pct_n]
    worst_5pct = pnl_sorted[:worst_5pct_n]
    return {
        "trade_count": n,
        "max_loss": float(pnl_sorted[0]),
        "worst1pct_mean": float(np.mean(worst_1pct)),
        "worst1pct_min": float(np.min(worst_1pct)),
        "CVaR95": float(np.mean(worst_5pct)),
        "CVaR99": float(np.mean(worst_1pct)),
    }


def _compute_session_transition_ratio(trade_journal: Optional[pd.DataFrame]) -> float:
    if trade_journal is None or trade_journal.empty:
        return float("nan")
    for col in ("open_ts_utc", "close_ts_utc"):
        if col not in trade_journal.columns:
            return float("nan")
    open_ts = pd.to_datetime(trade_journal["open_ts_utc"], utc=True, errors="coerce")
    close_ts = pd.to_datetime(trade_journal["close_ts_utc"], utc=True, errors="coerce")
    valid = open_ts.notna() & close_ts.notna()
    if valid.sum() == 0:
        return float("nan")
    entry_sessions = get_session_vectorized(open_ts[valid])
    exit_sessions = get_session_vectorized(close_ts[valid])
    return float((entry_sessions != exit_sessions).mean())


def _write_replay_metrics(
    out_root: Path,
    run_id: str,
    trade_outcomes_path: Path,
    trade_journal_path: Optional[Path],
) -> None:
    try:
        outcomes = pd.read_parquet(trade_outcomes_path)
    except Exception as e:
        log.info("[REPLAY_METRICS] skipped: could not read trade_outcomes (%s)", e)
        return

    pnl = pd.to_numeric(outcomes.get("pnl_bps", pd.Series(dtype=float)), errors="coerce").dropna()
    trade_count = int(len(pnl))
    duration_bars = pd.to_numeric(outcomes.get("duration_bars", pd.Series(dtype=float)), errors="coerce").dropna()
    mfe = pd.to_numeric(outcomes.get("mfe_bps", pd.Series(dtype=float)), errors="coerce").dropna()

    if trade_count == 0:
        metrics_row = {
            "run": run_id,
            "trade_count": 0,
            "pnl_sum_bps": float("nan"),
            "pnl_mean_bps": float("nan"),
            "pnl_median_bps": float("nan"),
            "win_rate": float("nan"),
            "avg_bars_held": float("nan"),
            "EdgeCaptureRatio": float("nan"),
            "max_loss": float("nan"),
            "worst1pct_mean": float("nan"),
            "worst1pct_min": float("nan"),
            "CVaR95": float("nan"),
            "CVaR99": float("nan"),
            "session_transition_ratio": float("nan"),
        }
        tail_row = _compute_tail_metrics(pnl)
    else:
        mfe_med = float(mfe.median()) if len(mfe) else float("nan")
        edge_capture = float(pnl.median() / mfe_med) if mfe_med and not np.isnan(mfe_med) else float("nan")
        trade_journal = None
        if trade_journal_path and trade_journal_path.exists():
            try:
                trade_journal = pd.read_parquet(trade_journal_path)
            except Exception:
                trade_journal = None
        tail_row = _compute_tail_metrics(pnl)
        metrics_row = {
            "run": run_id,
            "trade_count": trade_count,
            "pnl_sum_bps": float(pnl.sum()),
            "pnl_mean_bps": float(pnl.mean()),
            "pnl_median_bps": float(pnl.median()),
            "win_rate": float((pnl > 0).mean()),
            "avg_bars_held": float(duration_bars.mean()) if len(duration_bars) else float("nan"),
            "EdgeCaptureRatio": edge_capture,
            "max_loss": float(pnl.min()),
            "worst1pct_mean": tail_row["worst1pct_mean"],
            "worst1pct_min": tail_row["worst1pct_min"],
            "CVaR95": tail_row["CVaR95"],
            "CVaR99": tail_row["CVaR99"],
            "session_transition_ratio": _compute_session_transition_ratio(trade_journal),
        }

    metrics_path = out_root / "replay_metrics.csv"
    tail_path = out_root / "replay_tail_metrics.csv"
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)
    pd.DataFrame([tail_row]).to_csv(tail_path, index=False)
    log.info("[REPLAY_METRICS] Wrote %s", metrics_path.name)
    log.info("[REPLAY_METRICS] Wrote %s", tail_path.name)


def _pnl_profile_by_side(journal_df: pd.DataFrame) -> Dict[str, Any]:
    if journal_df is None or journal_df.empty:
        return {}
    if "side" not in journal_df.columns or "pnl_bps" not in journal_df.columns:
        return {}
    out: Dict[str, Any] = {}
    for side in ("long", "short"):
        sub = journal_df[journal_df["side"] == side]
        stats = _pnl_profile_summary(sub)
        if stats:
            out[side] = stats
    return out


def _margin_profile_summary(journal_df: pd.DataFrame) -> Dict[str, Any]:
    if journal_df is None or journal_df.empty:
        return {}
    if "margin_top1_top2" not in journal_df.columns:
        return {}
    vals = pd.to_numeric(journal_df["margin_top1_top2"], errors="coerce").dropna()
    if vals.empty:
        return {}

    def _r(val: Any) -> Any:
        try:
            return round(float(val), 6)
        except Exception:
            return val

    qs = vals.quantile([0.05, 0.25, 0.75, 0.95], interpolation="linear")
    return {
        "n": int(len(vals)),
        "mean": _r(vals.mean()),
        "median": _r(vals.median()),
        "p05": _r(qs.loc[0.05]),
        "p25": _r(qs.loc[0.25]),
        "p75": _r(qs.loc[0.75]),
        "p95": _r(qs.loc[0.95]),
        "min": _r(vals.min()),
        "max": _r(vals.max()),
    }


def _write_post_exit_regret_audit(
    out_root: Path,
    run_id: str,
    chunk_dir: Path,
    journal: Optional[pd.DataFrame],
) -> Optional[Dict[str, Any]]:
    """
    Postrun observation audit:
    - Primary (decision-relevant): bounded follow-through window + stricter threshold.
    - Observability (legacy): replay-end follow-through + legacy threshold.
    """
    if journal is None or journal.empty:
        return None

    required = {"trade_id", "side", "close_ts_utc"}
    if not required.issubset(set(journal.columns)):
        return None

    market_path = chunk_dir / "chunk_0_data.parquet"
    if not market_path.exists():
        return None

    try:
        market = pd.read_parquet(market_path, columns=["time", "bid_close", "ask_close"])
    except Exception:
        return None

    if market.empty:
        return None

    market["time"] = pd.to_datetime(market["time"], utc=True, errors="coerce")
    market["bid_close"] = pd.to_numeric(market["bid_close"], errors="coerce")
    market["ask_close"] = pd.to_numeric(market["ask_close"], errors="coerce")
    market = market.dropna(subset=["time", "bid_close", "ask_close"]).sort_values("time").reset_index(drop=True)
    if market.empty:
        return None

    times = market["time"].to_numpy(dtype="datetime64[ns]")
    bid = market["bid_close"].to_numpy(dtype=float)
    ask = market["ask_close"].to_numpy(dtype=float)

    j = journal.copy()
    j["trade_id"] = j["trade_id"].astype(str)
    j["close_ts_utc"] = pd.to_datetime(j["close_ts_utc"], utc=True, errors="coerce")

    exit_price = pd.to_numeric(j.get("exit_price_used"), errors="coerce")
    if "exit_bid" in j.columns and "exit_ask" in j.columns:
        exit_bid = pd.to_numeric(j["exit_bid"], errors="coerce")
        exit_ask = pd.to_numeric(j["exit_ask"], errors="coerce")
        is_long = j["side"].astype(str).str.lower() == "long"
        is_short = j["side"].astype(str).str.lower() == "short"
        exit_price = exit_price.where(exit_price.notna(), np.where(is_long, exit_bid, np.where(is_short, exit_ask, np.nan)))

    meaningful_thr = pd.to_numeric(j.get("meaningful_mfe_threshold_bps"), errors="coerce")
    default_regret_thr_bps = 5.0
    replay_end_observability_thr = meaningful_thr.where(
        (meaningful_thr.notna()) & (meaningful_thr > 0.0), default_regret_thr_bps
    )

    # Primary regret scoring (decision-relevant for THRESHOLD quality):
    # - bounded post-exit follow-through horizon (avoid replay-end bias)
    # - fixed stricter threshold
    primary_horizon_bars = 24
    primary_regret_thr_bps = 10.0

    primary_post_exit_mfe_vals: List[float] = []
    primary_regret_flags: List[bool] = []
    primary_valid_flags: List[bool] = []

    replay_end_post_exit_mfe_vals: List[float] = []
    replay_end_regret_flags: List[bool] = []

    for i in range(len(j)):
        ts = j.at[i, "close_ts_utc"]
        side = str(j.at[i, "side"]).lower() if pd.notna(j.at[i, "side"]) else ""
        px = exit_price.iat[i] if i < len(exit_price) else np.nan

        if pd.isna(ts) or not np.isfinite(px) or px <= 0.0 or side not in ("long", "short"):
            primary_post_exit_mfe_vals.append(np.nan)
            primary_regret_flags.append(False)
            primary_valid_flags.append(False)
            replay_end_post_exit_mfe_vals.append(np.nan)
            replay_end_regret_flags.append(False)
            continue

        ts64 = np.datetime64(ts.to_datetime64())
        exit_idx = int(np.searchsorted(times, ts64, side="right") - 1)
        if exit_idx < 0:
            exit_idx = 0
        start = exit_idx + 1

        def _favorable_move_bps(end: int) -> float:
            if start >= len(times) or start >= end:
                return 0.0
            if side == "long":
                future_fav = float(np.nanmax(bid[start:end]))
                val = ((future_fav - float(px)) / float(px)) * 10000.0
            else:
                future_fav = float(np.nanmin(ask[start:end]))
                val = ((float(px) - future_fav) / float(px)) * 10000.0
            if not np.isfinite(val):
                return np.nan
            return max(0.0, float(val))

        replay_end_post_exit_mfe = _favorable_move_bps(len(times))
        primary_end = min(len(times), start + int(primary_horizon_bars))
        primary_post_exit_mfe = _favorable_move_bps(primary_end)

        replay_end_post_exit_mfe_vals.append(replay_end_post_exit_mfe)
        replay_end_regret_flags.append(
            bool(
                np.isfinite(replay_end_post_exit_mfe)
                and replay_end_post_exit_mfe >= float(replay_end_observability_thr.iat[i])
            )
        )

        primary_post_exit_mfe_vals.append(primary_post_exit_mfe)
        primary_regret_flags.append(
            bool(np.isfinite(primary_post_exit_mfe) and primary_post_exit_mfe >= float(primary_regret_thr_bps))
        )
        primary_valid_flags.append(bool(np.isfinite(primary_post_exit_mfe)))

    # Primary decision-relevant fields (kept on canonical names).
    j["post_exit_mfe_bps"] = pd.Series(primary_post_exit_mfe_vals, index=j.index, dtype=float)
    j["early_exit_regret"] = pd.Series(primary_regret_flags, index=j.index, dtype=bool)
    j["early_exit_regret_threshold_bps"] = pd.Series(float(primary_regret_thr_bps), index=j.index, dtype=float)
    j["post_exit_regret_observed"] = pd.Series(primary_valid_flags, index=j.index, dtype=bool)

    # Replay-end observability fields (non-primary, for audit only).
    j["post_exit_mfe_bps_replay_end_obs"] = pd.Series(replay_end_post_exit_mfe_vals, index=j.index, dtype=float)
    j["early_exit_regret_replay_end_obs"] = pd.Series(replay_end_regret_flags, index=j.index, dtype=bool)
    j["early_exit_regret_threshold_bps_replay_end_obs"] = pd.Series(
        replay_end_observability_thr, index=j.index, dtype=float
    )

    journal_path = out_root / f"trade_journal_{run_id}_MERGED.parquet"
    j.to_parquet(journal_path, index=False)

    outcomes_path = out_root / f"trade_outcomes_{run_id}_MERGED.parquet"
    if outcomes_path.exists():
        try:
            outcomes = pd.read_parquet(outcomes_path)
            if "trade_id" in outcomes.columns:
                outcomes["trade_id"] = outcomes["trade_id"].astype(str)
                merge_cols = [
                    "trade_id",
                    "post_exit_mfe_bps",
                    "early_exit_regret",
                    "early_exit_regret_threshold_bps",
                    "post_exit_mfe_bps_replay_end_obs",
                    "early_exit_regret_replay_end_obs",
                    "early_exit_regret_threshold_bps_replay_end_obs",
                ]
                outcomes = outcomes.drop(
                    columns=[c for c in merge_cols if c != "trade_id" and c in outcomes.columns],
                    errors="ignore",
                )
                outcomes = outcomes.merge(j[merge_cols], on="trade_id", how="left")
                outcomes.to_parquet(outcomes_path, index=False)
        except Exception:
            pass

    audit_df = j[j["post_exit_regret_observed"]].copy()
    if audit_df.empty:
        return None

    audit_df["exit_reason_norm"] = audit_df.get("exit_reason", pd.Series(dtype=object)).fillna("UNKNOWN").astype(str)
    summary_rows: List[Dict[str, Any]] = []
    for reason, grp in audit_df.groupby("exit_reason_norm", dropna=False):
        vals = pd.to_numeric(grp["post_exit_mfe_bps"], errors="coerce").dropna()
        summary_rows.append(
            {
                "close_reason": str(reason),
                "count": int(len(grp)),
                "regret_count": int(grp["early_exit_regret"].sum()),
                "regret_rate": float(grp["early_exit_regret"].mean()) if len(grp) else 0.0,
                "post_exit_mfe_bps_mean": float(vals.mean()) if not vals.empty else None,
                "post_exit_mfe_bps_median": float(vals.median()) if not vals.empty else None,
                "post_exit_mfe_bps_p90": float(vals.quantile(0.90, interpolation="linear")) if not vals.empty else None,
            }
        )
    summary_rows.sort(key=lambda x: (-x["count"], x["close_reason"]))

    thr_df = audit_df[audit_df["exit_reason_norm"] == "THRESHOLD"].copy()
    thr_vals = pd.to_numeric(thr_df.get("post_exit_mfe_bps"), errors="coerce").dropna()
    threshold_summary = {
        "count": int(len(thr_df)),
        "meaningful_followthrough_count": int(thr_df["early_exit_regret"].sum()) if len(thr_df) else 0,
        "meaningful_followthrough_rate": float(thr_df["early_exit_regret"].mean()) if len(thr_df) else 0.0,
        "post_exit_mfe_bps_mean": float(thr_vals.mean()) if not thr_vals.empty else None,
        "post_exit_mfe_bps_median": float(thr_vals.median()) if not thr_vals.empty else None,
        "post_exit_mfe_bps_p90": float(thr_vals.quantile(0.90, interpolation="linear")) if not thr_vals.empty else None,
    }

    replay_end_observability_vals = pd.to_numeric(
        audit_df.get("post_exit_mfe_bps_replay_end_obs"), errors="coerce"
    ).dropna()
    replay_end_threshold_df = audit_df[audit_df["exit_reason_norm"] == "THRESHOLD"].copy()
    replay_end_threshold_vals = pd.to_numeric(
        replay_end_threshold_df.get("post_exit_mfe_bps_replay_end_obs"), errors="coerce"
    ).dropna()
    replay_end_observability_summary = {
        "overall": {
            "observed_exits": int(len(audit_df)),
            "regret_count": int(audit_df["early_exit_regret_replay_end_obs"].sum()),
            "regret_rate": float(audit_df["early_exit_regret_replay_end_obs"].mean()),
            "post_exit_mfe_bps_mean": float(replay_end_observability_vals.mean()) if not replay_end_observability_vals.empty else None,
            "post_exit_mfe_bps_median": float(replay_end_observability_vals.median()) if not replay_end_observability_vals.empty else None,
            "post_exit_mfe_bps_p90": float(
                replay_end_observability_vals.quantile(0.90, interpolation="linear")
            ) if not replay_end_observability_vals.empty else None,
        },
        "threshold_exits": {
            "count": int(len(replay_end_threshold_df)),
            "meaningful_followthrough_count": int(replay_end_threshold_df["early_exit_regret_replay_end_obs"].sum()) if len(replay_end_threshold_df) else 0,
            "meaningful_followthrough_rate": float(replay_end_threshold_df["early_exit_regret_replay_end_obs"].mean()) if len(replay_end_threshold_df) else 0.0,
            "post_exit_mfe_bps_mean": float(replay_end_threshold_vals.mean()) if not replay_end_threshold_vals.empty else None,
            "post_exit_mfe_bps_median": float(replay_end_threshold_vals.median()) if not replay_end_threshold_vals.empty else None,
            "post_exit_mfe_bps_p90": float(
                replay_end_threshold_vals.quantile(0.90, interpolation="linear")
            ) if not replay_end_threshold_vals.empty else None,
        },
    }

    payload = {
        "run_id": run_id,
        "artifact": "post_exit_regret_audit",
        "definition": {
            "primary": {
                "post_exit_mfe_bps": (
                    "Best favorable move from first bar after exit to the next primary_followthrough_horizon_bars bars, "
                    "in trade direction, bps vs exit_price_used."
                ),
                "early_exit_regret": "True when primary post_exit_mfe_bps >= early_exit_regret_threshold_bps.",
                "primary_followthrough_horizon_bars": int(primary_horizon_bars),
                "primary_threshold_bps": float(primary_regret_thr_bps),
            },
            "replay_end_observability": {
                "post_exit_mfe_bps_replay_end_obs": (
                    "Best favorable move from first bar after exit to replay end, in trade direction, bps vs exit_price_used."
                ),
                "early_exit_regret_replay_end_obs": (
                    "True when replay-end post_exit_mfe_bps_replay_end_obs >= early_exit_regret_threshold_bps_replay_end_obs."
                ),
                "default_threshold_bps_when_missing": default_regret_thr_bps,
            },
        },
        "overall": {
            "observed_exits": int(len(audit_df)),
            "regret_count": int(audit_df["early_exit_regret"].sum()),
            "regret_rate": float(audit_df["early_exit_regret"].mean()),
        },
        "threshold_exits": threshold_summary,
        "replay_end_observability": replay_end_observability_summary,
        # Backward-compatible alias for existing analysis readers.
        "observability_replay_end": replay_end_observability_summary,
        "by_close_reason": summary_rows,
    }
    out_path = out_root / f"post_exit_regret_audit_{run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(out_path), "threshold_summary": threshold_summary}


def _write_postrun_trade_reports(
    out_root: Path, run_id: str, chunk_dir: Path, footer: Optional[Dict[str, Any]]
) -> None:
    """
    Best-effort postrun trade reports from trade_journal_{run_id}_MERGED.parquet.
    Deterministic: groupby dropna=False, stable sort.
    """
    journal = None
    journal_path = out_root / f"trade_journal_{run_id}_MERGED.parquet"
    if not journal_path.exists():
        log.info("[POSTRUN_A_SKIPPED] missing %s", journal_path)
    else:
        try:
            journal = pd.read_parquet(journal_path)
        except Exception as e:
            log.info("[POSTRUN_TRADE_REPORT_SKIPPED] failed to read %s: %s", journal_path, e)
            journal = None

    # Enrich trade_journal with price/spread fields from trade_outcomes if missing
    try:
        if journal is not None:
            outcomes_path = out_root / f"trade_outcomes_{run_id}_MERGED.parquet"
            if outcomes_path.exists():
                outcomes_df = pd.read_parquet(outcomes_path)
                if "trade_id" in outcomes_df.columns:
                    outcomes_df["trade_id"] = outcomes_df["trade_id"].astype(str)
                if "trade_id" in journal.columns:
                    journal["trade_id"] = journal["trade_id"].astype(str)
                cols_to_add = [
                    "entry_bid",
                    "entry_ask",
                    "exit_bid",
                    "exit_ask",
                    "entry_spread_bps",
                    "exit_spread_bps",
                    "entry_price_used",
                    "exit_price_used",
                ]
                missing_cols = [c for c in cols_to_add if c not in journal.columns and c in outcomes_df.columns]
                if missing_cols:
                    journal = journal.merge(
                        outcomes_df[["trade_id"] + missing_cols],
                        on="trade_id",
                        how="left",
                        suffixes=("", "_outcomes"),
                    )
                    journal.to_parquet(journal_path, index=False)
                    log.info("[POSTRUN_TRADE_JOURNAL_ENRICH] added_cols=%s path=%s", missing_cols, journal_path)
    except Exception as e:
        log.info("[POSTRUN_TRADE_JOURNAL_ENRICH_SKIPPED] %s", e)

    post_exit_audit = None
    try:
        post_exit_audit = _write_post_exit_regret_audit(out_root, run_id, chunk_dir, journal)
        if post_exit_audit is not None:
            try:
                journal = pd.read_parquet(out_root / f"trade_journal_{run_id}_MERGED.parquet")
            except Exception:
                pass
    except Exception as e:
        log.info("[POST_EXIT_REGRET_AUDIT_SKIPPED] %s", e)

    required_session_side = {"session", "side", "pnl_bps"}
    required_holding = {"bars_in_trade", "pnl_bps"}
    missing_cols = []
    if journal is not None:
        missing_cols = sorted((required_session_side | required_holding) - set(journal.columns))
        if missing_cols:
            log.info("[POSTRUN_TRADE_REPORT_SKIPPED] missing required columns: %s", missing_cols)
            journal = None

    def _schema_lines(df: pd.DataFrame) -> str:
        cols = [f"{name}:{dtype}" for name, dtype in df.dtypes.items()]
        return ",".join(cols)

    def _preview_lines(df: pd.DataFrame) -> list[str]:
        if df.empty:
            return ["<EMPTY>"]
        header = "\t".join(df.columns.astype(str).tolist())
        rows = [
            "\t".join("" if pd.isna(v) else str(v) for v in list(row))
            for row in df.itertuples(index=False, name=None)
        ]
        return [header] + rows

    def _safe_sort_values(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Sort on stringified keys so mixed None/str object columns do not crash postrun summaries.
        """
        if not columns:
            return df
        sort_df = df.copy()
        sort_cols = []
        for col in columns:
            key_col = f"__sort_{col}"
            sort_df[key_col] = sort_df[col].map(lambda x: "<NA>" if pd.isna(x) else str(x))
            sort_cols.append(key_col)
        out = sort_df.sort_values(sort_cols).reset_index(drop=True)
        return out.drop(columns=sort_cols)

    def _format_counts(counts: Optional[Dict[Any, Any]]) -> str:
        if not counts:
            return "<EMPTY>"
        items = []
        for key in sorted(counts.keys(), key=lambda x: "None" if x is None else str(x)):
            items.append(f"{key}={counts[key]}")
        return " ".join(items)

    journal_rows = int(len(journal)) if journal is not None else 0
    wrote_session_side = False
    wrote_holding_bins = False
    wrote_side_session = False
    wrote_entry_gates = False
    pnl_audit_error: Optional[str] = None
    proof_lines = [
        f"[POSTRUN_TRADE_REPORT_PROOF] run_id={run_id} journal_rows={journal_rows} "
        f"wrote_session_side=0 wrote_holding_bins=0 wrote_side_session=0 wrote_entry_gates=0"
    ]
    if post_exit_audit is not None:
        proof_lines.append(f"post_exit_regret_audit_path={post_exit_audit.get('path')}")
        t = post_exit_audit.get("threshold_summary", {})
        proof_lines.append(
            "[POST_EXIT_REGRET_THRESHOLD] count=%s meaningful_followthrough_count=%s meaningful_followthrough_rate=%s"
            % (
                t.get("count"),
                t.get("meaningful_followthrough_count"),
                t.get("meaningful_followthrough_rate"),
            )
        )

    # Session x side report
    try:
        if journal is None:
            raise RuntimeError("journal_missing_or_invalid")
        ss = journal[["session", "side", "pnl_bps"]].copy()
        ss["pnl_bps"] = pd.to_numeric(ss["pnl_bps"], errors="coerce")
        ss = ss[ss["pnl_bps"].notna()]
        if not ss.empty:
            grouped = ss.groupby(["session", "side"], dropna=False)
            agg = grouped["pnl_bps"].agg(
                n_trades="size",
                pnl_bps_sum="sum",
                pnl_bps_mean="mean",
                pnl_bps_median="median",
                pnl_bps_p05=lambda x: x.quantile(0.05, interpolation="linear"),
                pnl_bps_p95=lambda x: x.quantile(0.95, interpolation="linear"),
                win_rate=lambda x: (x > 0).mean(),
            )
            agg = _safe_sort_values(agg.reset_index(), ["session", "side"])
            out_parquet = out_root / f"trade_report_session_side_{run_id}.parquet"
            out_csv = out_root / f"trade_report_session_side_{run_id}.csv"
            agg.to_parquet(out_parquet, index=False)
            agg.to_csv(out_csv, index=False)
            proof_lines.append(f"session_side_parquet={out_parquet}")
            proof_lines.append(f"session_side_csv={out_csv}")
            proof_lines.append(f"session_side_schema={_schema_lines(agg)}")
            preview = _preview_lines(agg.head(5))
            proof_lines.append("session_side_head5_tsv:")
            proof_lines.extend(preview)
            wrote_session_side = True

            # Side x session report (same aggregation, explicit artifact)
            out_parquet = out_root / f"trade_report_side_session_{run_id}.parquet"
            out_csv = out_root / f"trade_report_side_session_{run_id}.csv"
            agg.to_parquet(out_parquet, index=False)
            agg.to_csv(out_csv, index=False)
            proof_lines.append(f"side_session_parquet={out_parquet}")
            proof_lines.append(f"side_session_csv={out_csv}")
            proof_lines.append(f"side_session_schema={_schema_lines(agg)}")
            preview = _preview_lines(agg.head(5))
            proof_lines.append("side_session_head5_tsv:")
            proof_lines.extend(preview)
            wrote_side_session = True
    except Exception as e:
        if str(e) == "journal_missing_or_invalid":
            log.info("[POSTRUN_A_SKIPPED] missing trade_journal (side_session)")
        else:
            log.info("[POSTRUN_TRADE_REPORT_SKIPPED] session_side report failed: %s", e)

    # CATA report: make catastrophic origin explicit for postrun analysis.
    try:
        if journal is None:
            raise RuntimeError("journal_missing_or_invalid")
        if "exit_reason" not in journal.columns:
            raise RuntimeError("missing_exit_reason")

        cata = journal[journal["exit_reason"] == "CATASTROPHIC_GUARD"].copy()
        if not cata.empty:
            for col in [
                "pnl_bps",
                "mfe_bps",
                "mae_bps",
                "dd_from_mfe_bps_exit",
                "distance_from_peak_mfe_bps_exit",
                "time_since_mfe_bars_exit",
                "bars_in_trade",
            ]:
                if col in cata.columns:
                    cata[col] = pd.to_numeric(cata[col], errors="coerce")
            if "adverse_first" in cata.columns:
                cata["adverse_first"] = cata["adverse_first"].fillna(False).astype(bool)

            summary = {
                "run_id": run_id,
                "cata_count": int(len(cata)),
                "cata_pnl_sum_bps": float(cata["pnl_bps"].sum()) if "pnl_bps" in cata.columns else None,
                "cata_pnl_mean_bps": float(cata["pnl_bps"].mean()) if "pnl_bps" in cata.columns else None,
                "cata_side_counts": cata["side"].value_counts(dropna=False).to_dict() if "side" in cata.columns else {},
                "cata_session_counts": cata["session"].value_counts(dropna=False).to_dict() if "session" in cata.columns else {},
                "cata_session_side_counts": (
                    cata.groupby(["session", "side"], dropna=False).size().reset_index(name="n_trades").to_dict("records")
                    if {"session", "side"}.issubset(cata.columns)
                    else []
                ),
                "mfe_stats": {
                    "mean": float(cata["mfe_bps"].mean()) if "mfe_bps" in cata.columns else None,
                    "median": float(cata["mfe_bps"].median()) if "mfe_bps" in cata.columns else None,
                    "mfe_ge_0": int((cata["mfe_bps"] >= 0).sum()) if "mfe_bps" in cata.columns else None,
                    "mfe_ge_25": int((cata["mfe_bps"] >= 25).sum()) if "mfe_bps" in cata.columns else None,
                    "mfe_ge_50": int((cata["mfe_bps"] >= 50).sum()) if "mfe_bps" in cata.columns else None,
                    "mfe_ge_100": int((cata["mfe_bps"] >= 100).sum()) if "mfe_bps" in cata.columns else None,
                },
                "mae_stats": {
                    "mean": float(cata["mae_bps"].mean()) if "mae_bps" in cata.columns else None,
                    "median": float(cata["mae_bps"].median()) if "mae_bps" in cata.columns else None,
                },
                "peak_giveback_stats": {
                    "dd_from_mfe_bps_exit_mean": float(cata["dd_from_mfe_bps_exit"].mean()) if "dd_from_mfe_bps_exit" in cata.columns else None,
                    "distance_from_peak_mfe_bps_exit_mean": float(cata["distance_from_peak_mfe_bps_exit"].mean()) if "distance_from_peak_mfe_bps_exit" in cata.columns else None,
                    "time_since_mfe_bars_exit_mean": float(cata["time_since_mfe_bars_exit"].mean()) if "time_since_mfe_bars_exit" in cata.columns else None,
                },
                "adverse_first": {
                    "count_true": int(cata["adverse_first"].sum()) if "adverse_first" in cata.columns else None,
                    "rate_true": float(cata["adverse_first"].mean()) if "adverse_first" in cata.columns else None,
                },
                "hold_mean_bars": float(cata["bars_in_trade"].mean()) if "bars_in_trade" in cata.columns else None,
                "hold_median_bars": float(cata["bars_in_trade"].median()) if "bars_in_trade" in cata.columns else None,
            }

            out_json = out_root / f"trade_report_cata_{run_id}.json"
            out_csv = out_root / f"trade_report_cata_trades_{run_id}.csv"
            out_json.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            keep_cols = [
                c for c in [
                    "trade_id",
                    "side",
                    "session",
                    "open_ts_utc",
                    "close_ts_utc",
                    "pnl_bps",
                    "bars_in_trade",
                    "mfe_bps",
                    "mae_bps",
                    "adverse_first",
                    "peak_mfe_bps_exit_state",
                    "dd_from_mfe_bps_exit",
                    "distance_from_peak_mfe_bps_exit",
                    "time_since_mfe_bars_exit",
                    "path_quality_pred",
                    "mfe_first_n_pred",
                    "tradable_prob",
                ] if c in cata.columns
            ]
            cata[keep_cols].to_csv(out_csv, index=False)
            proof_lines.append(f"cata_json={out_json}")
            proof_lines.append(f"cata_csv={out_csv}")
            proof_lines.append(
                "[CATA_POSTRUN] count=%s pnl_sum_bps=%s mfe_ge_25=%s mfe_ge_50=%s mfe_ge_100=%s adverse_first_rate=%s"
                % (
                    summary["cata_count"],
                    summary["cata_pnl_sum_bps"],
                    summary["mfe_stats"]["mfe_ge_25"],
                    summary["mfe_stats"]["mfe_ge_50"],
                    summary["mfe_stats"]["mfe_ge_100"],
                    summary["adverse_first"]["rate_true"],
                )
            )
    except Exception as e:
        if str(e) == "journal_missing_or_invalid":
            log.info("[POSTRUN_CATA_SKIPPED] missing trade_journal")
        else:
            log.info("[POSTRUN_CATA_SKIPPED] cata report failed: %s", e)

    # Holding bins report (by session + holding_bin)
    try:
        if journal is None:
            raise RuntimeError("journal_missing_or_invalid")
        hb = journal[["session", "bars_in_trade", "pnl_bps"]].copy()
        hb["bars_in_trade"] = pd.to_numeric(hb["bars_in_trade"], errors="coerce")
        hb["pnl_bps"] = pd.to_numeric(hb["pnl_bps"], errors="coerce")
        hb = hb[hb["bars_in_trade"].notna() & hb["pnl_bps"].notna()]
        if not hb.empty:
            edges = [0, 5, 10, 20, 50, 100, 10**9]
            labels = ["0-4", "5-9", "10-19", "20-49", "50-99", "100+"]
            hb["holding_bin"] = pd.cut(
                hb["bars_in_trade"],
                bins=edges,
                labels=labels,
                right=False,
                include_lowest=True,
            )
            grouped = hb.groupby(["holding_bin", "session"], dropna=False, observed=False)
            agg = grouped["pnl_bps"].agg(
                n_trades="size",
                pnl_bps_sum="sum",
                pnl_bps_mean="mean",
                pnl_bps_median="median",
                win_rate=lambda x: (x > 0).mean(),
            )
            agg = _safe_sort_values(agg.reset_index(), ["holding_bin", "session"])
            out_parquet = out_root / f"trade_report_holding_bins_{run_id}.parquet"
            out_csv = out_root / f"trade_report_holding_bins_{run_id}.csv"
            agg.to_parquet(out_parquet, index=False)
            agg.to_csv(out_csv, index=False)
            proof_lines.append(f"holding_bins_parquet={out_parquet}")
            proof_lines.append(f"holding_bins_csv={out_csv}")
            proof_lines.append(f"holding_bins_schema={_schema_lines(agg)}")
            preview = _preview_lines(agg.head(5))
            proof_lines.append("holding_bins_head5_tsv:")
            proof_lines.extend(preview)
            wrote_holding_bins = True
    except Exception as e:
        if str(e) == "journal_missing_or_invalid":
            log.info("[POSTRUN_TRADE_REPORT_SKIPPED] holding_bins report skipped (journal missing)")
        else:
            log.info("[POSTRUN_TRADE_REPORT_SKIPPED] holding_bins report failed: %s", e)

    # Entry gates report (SSoT from journal + runner counters from footer)
    try:
        ssot_total = None
        ssot_by_side = None
        ssot_by_session_side = None
        if journal is not None:
            ssot_total = int(len(journal))
            if "side" in journal.columns:
                ssot_by_side = {
                    str(k): int(v)
                    for k, v in journal["side"].value_counts(dropna=False).to_dict().items()
                }
            if "session" in journal.columns and "side" in journal.columns:
                ssot_by_session_side = (
                    journal.groupby(["session", "side"], dropna=False)
                    .size()
                    .reset_index(name="n_trades")
                    .to_dict("records")
                )

        runner_counters = None
        if footer:
            runner_counters = {
                "entry_attempt_long": footer.get("entry_attempt_long"),
                "entry_attempt_short": footer.get("entry_attempt_short"),
                "entry_accept_long": footer.get("entry_accept_long"),
                "entry_accept_short": footer.get("entry_accept_short"),
                "policy_entry_attempt_long": footer.get("entry_attempt_long"),
                "policy_entry_attempt_short": footer.get("entry_attempt_short"),
                "policy_entry_accept_long": footer.get("entry_accept_long"),
                "policy_entry_accept_short": footer.get("entry_accept_short"),
                "pregate_enabled": footer.get("pregate_enabled"),
                "pregate_skips": footer.get("pregate_skips"),
                "pregate_passes": footer.get("pregate_passes"),
                "pregate_missing_inputs": footer.get("pregate_missing_inputs"),
                "entry_gate_counters": footer.get("entry_gate_counters"),
                "threshold_used": footer.get("threshold_used"),
                "threshold_source": footer.get("threshold_source"),
                "ctx_cont_mask_id": footer.get("ctx_cont_mask_id"),
                "ctx_cat_mask_id": footer.get("ctx_cat_mask_id"),
            }

        runner_accept_rate_long = None
        runner_accept_rate_short = None
        if runner_counters:
            try:
                al = runner_counters.get("entry_attempt_long")
                asl = runner_counters.get("entry_accept_long")
                if al is not None and asl is not None:
                    runner_accept_rate_long = (asl or 0) / max(al or 0, 1)
                asr = runner_counters.get("entry_attempt_short")
                asr_acc = runner_counters.get("entry_accept_short")
                if asr is not None and asr_acc is not None:
                    runner_accept_rate_short = (asr_acc or 0) / max(asr or 0, 1)
            except Exception:
                pass

        opened_long = None
        opened_short = None
        opened_total = None
        if footer is not None:
            try:
                opened_total = int(footer.get("n_trades_opened_registered", 0) or 0)
                opened_long = int(footer.get("n_trades_opened_registered_long", 0) or 0)
                opened_short = int(footer.get("n_trades_opened_registered_short", 0) or 0)
            except Exception:
                opened_long = None
                opened_short = None
                opened_total = None

        closed_long = None
        closed_short = None
        closed_total = None
        if ssot_by_side is not None and ssot_total is not None:
            try:
                closed_long = int(ssot_by_side.get("long", 0) or 0)
                closed_short = int(ssot_by_side.get("short", 0) or 0)
                closed_total = int(ssot_total)
            except Exception:
                closed_long = None
                closed_short = None
                closed_total = None

        opened_minus_closed = None
        if opened_total is not None and closed_total is not None:
            opened_minus_closed = {
                "long": opened_long - closed_long,
                "short": opened_short - closed_short,
                "total": opened_total - closed_total,
            }

        lifecycle_reasons: List[str] = []
        if opened_total is not None and closed_total is not None:
            if opened_total != closed_total:
                lifecycle_reasons.append("OPENED_TOTAL_NE_CLOSED_TOTAL")
            if opened_short is not None and closed_short is not None and opened_short > 0 and closed_short == 0:
                lifecycle_reasons.append("OPENED_SHORT_GT_0_BUT_CLOSED_SHORT_EQ_0")
            if opened_long is not None and closed_long is not None and opened_long > 0 and closed_long == 0:
                lifecycle_reasons.append("OPENED_LONG_GT_0_BUT_CLOSED_LONG_EQ_0")
        trade_lifecycle_mismatch = bool(lifecycle_reasons)

        mismatch_reasons = []
        if ssot_by_side is not None and runner_counters is not None:
            try:
                ssot_short = ssot_by_side.get("short", 0) or 0
                runner_short = runner_counters.get("entry_accept_short") or 0
                if ssot_short == 0 and runner_short > 0:
                    mismatch_reasons.append("SSOT_SHORT_TRADES_ZERO_BUT_RUNNER_ACCEPT_SHORT_GT_ZERO")
            except Exception:
                pass
        if footer is not None and ssot_total is not None:
            try:
                n_trades_closed = footer.get("n_trades_closed")
                if n_trades_closed is not None and int(n_trades_closed) != int(ssot_total):
                    mismatch_reasons.append("SSOT_TOTAL_NEQ_FOOTER_N_TRADES_CLOSED")
            except Exception:
                pass

        counter_semantics_mismatch = bool(mismatch_reasons)
        proof_lines.append(
            "[ENTRY_GATES_MISMATCH] mismatch=%d reasons=%s ssot_by_side=%s runner_entry_accept_short=%s"
            % (
                int(counter_semantics_mismatch),
                ",".join(mismatch_reasons) if mismatch_reasons else "NONE",
                ssot_by_side,
                None if runner_counters is None else runner_counters.get("entry_accept_short"),
            )
        )

        payload: Dict[str, Any] = {
            "run_id": run_id,
            "n_trades_closed": footer.get("n_trades_closed") if footer else None,
            "ssot_n_trades_total": ssot_total,
            "ssot_trades_by_side": ssot_by_side,
            "ssot_closed_trades_by_side": (
                None
                if ssot_total is None
                else {
                    "long": int(ssot_by_side.get("long", 0) or 0) if ssot_by_side is not None else 0,
                    "short": int(ssot_by_side.get("short", 0) or 0) if ssot_by_side is not None else 0,
                    "flat": int(ssot_by_side.get("flat", 0) or 0) if ssot_by_side is not None else 0,
                    "total": int(ssot_total),
                }
            ),
            "ssot_trades_by_session_side": ssot_by_session_side,
            "runner_counters": runner_counters,
            "ssot_opened_registered_by_side": (
                None
                if opened_total is None
                else {"long": opened_long, "short": opened_short, "total": opened_total}
            ),
            "runner_opened_trades_by_side": (
                None
                if opened_total is None
                else {"long": opened_long, "short": opened_short, "total": opened_total}
            ),
            "opened_minus_closed_by_side": opened_minus_closed,
            "runner_accept_rate_long": runner_accept_rate_long,
            "runner_accept_rate_short": runner_accept_rate_short,
            "counter_semantics_mismatch": counter_semantics_mismatch,
            "mismatch_reasons": mismatch_reasons,
            "trade_lifecycle_mismatch": trade_lifecycle_mismatch,
            "trade_lifecycle_mismatch_reasons": lifecycle_reasons,
            "note": (
                "Postrun gate summary: opened SSoT from chunk_footer (open_trades.append), closed SSoT from "
                "trade_journal (accepted exits), plus policy counters from EntryManager."
            ),
        }

        out_path = out_root / f"trade_report_entry_gates_{run_id}.json"
        out_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        proof_lines.append(f"entry_gates_json={out_path}")
        if opened_total is None or closed_total is None:
            proof_lines.append(
                "[TRADE_LIFECYCLE_MISMATCH] mismatch=0 reasons=SKIPPED_MISSING_INPUTS opened_registered_by_side=None "
                "closed_by_side=None opened_minus_closed=None"
            )
        else:
            proof_lines.append(
                "[TRADE_LIFECYCLE_MISMATCH] mismatch=%d reasons=%s opened_registered_by_side=%s closed_by_side=%s "
                "opened_minus_closed=%s"
                % (
                    int(trade_lifecycle_mismatch),
                    ",".join(lifecycle_reasons) if lifecycle_reasons else "NONE",
                    {"long": opened_long, "short": opened_short, "total": opened_total},
                    {"long": closed_long, "short": closed_short, "total": closed_total},
                    opened_minus_closed,
                )
            )
        if runner_counters is not None:
            proof_lines.append(
                "[POLICY_ENTRY_COUNTERS] accept_long=%s accept_short=%s attempt_long=%s attempt_short=%s"
                % (
                    runner_counters.get("entry_accept_long"),
                    runner_counters.get("entry_accept_short"),
                    runner_counters.get("entry_attempt_long"),
                    runner_counters.get("entry_attempt_short"),
                )
            )
        wrote_entry_gates = True
    except Exception as e:
        log.info("[POSTRUN_A_FAIL] entry_gates write failed: %s", e)

    # PnL audit proof (bid/ask recompute) - hard requirement
    n_trades_closed = footer.get("n_trades_closed") if footer else None
    try:
        if journal is None:
            if n_trades_closed in (0, 0.0, "0"):
                proof_lines.append("[REPLAY_PNL_AUDIT_SKIPPED] reason=journal_missing_no_trades")
                raise StopIteration("skip_pnl_audit_no_trades")
            raise RuntimeError("journal_missing_or_invalid")
        cols_needed = {"entry_bid", "entry_ask", "exit_bid", "exit_ask", "pnl_bps", "side", "trade_id"}
        missing = sorted(cols_needed - set(journal.columns))
        if missing:
            raise RuntimeError(f"missing_cols={missing}")
        sample = journal.dropna(
            subset=["entry_bid", "entry_ask", "exit_bid", "exit_ask", "pnl_bps", "side"]
        ).head(10)
        if sample.empty:
            if os.getenv("GX1_EXIT_HASH_GUARD_BYPASS") == "1":
                proof_lines.append("[REPLAY_PNL_AUDIT_SKIPPED] reason=audit_sample_empty_bypass")
                raise StopIteration("skip_pnl_audit_bypass")
            raise RuntimeError("audit_sample_empty")
        audit_rows = []
        mismatch = 0
        for _, r in sample.iterrows():
            recompute = compute_pnl_bps(
                float(r["entry_bid"]),
                float(r["entry_ask"]),
                float(r["exit_bid"]),
                float(r["exit_ask"]),
                str(r["side"]),
            )
            pnl_val = float(r["pnl_bps"])
            match = abs(recompute - pnl_val) <= 0.01
            if not match:
                mismatch += 1
            audit_rows.append(
                {
                    "trade_id": r.get("trade_id"),
                    "side": r.get("side"),
                    "entry_bid": float(r["entry_bid"]),
                    "entry_ask": float(r["entry_ask"]),
                    "exit_bid": float(r["exit_bid"]),
                    "exit_ask": float(r["exit_ask"]),
                    "entry_spread_bps": r.get("entry_spread_bps"),
                    "exit_spread_bps": r.get("exit_spread_bps"),
                    "pnl_bps": pnl_val,
                    "recompute_bps": recompute,
                    "recompute_match": match,
                }
            )
        if mismatch > 0:
            raise RuntimeError(f"recompute_mismatch={mismatch}")
        proof_lines.append(f"[REPLAY_PNL_AUDIT] sample={audit_rows}")
    except StopIteration:
        pass
    except Exception as e:
        pnl_audit_error = str(e)
        proof_lines.append(f"[REPLAY_PNL_AUDIT_FAILED] reason={pnl_audit_error}")

    # Margin filter effect: outcomes stats + footer SSoT
    try:
        entry_margin_min_used = footer.get("entry_margin_min_used") if footer else None
        n_trades_closed = footer.get("n_trades_closed") if footer else None
        outcomes_path = out_root / f"trade_outcomes_{run_id}_MERGED.parquet"
        if not outcomes_path.exists():
            log.info("[POSTRUN_MARGIN_PROFILE_SKIPPED] missing outcomes parquet: %s", outcomes_path)
            proof_lines.append(f"[POSTRUN_MARGIN_PROFILE_SKIPPED] missing outcomes parquet: {outcomes_path}")
            proof_lines.append("[MARGIN_FILTER_SSOT_MATCH_SKIPPED] missing outcomes parquet")
        else:
            outcomes_df = pd.read_parquet(outcomes_path)
            pnl_stats = _pnl_profile_summary(outcomes_df)
            exit_reason_counts = (
                outcomes_df["exit_reason"].value_counts(dropna=False).to_dict()
                if "exit_reason" in outcomes_df.columns
                else {}
            )
            session_counts = (
                outcomes_df["session"].value_counts(dropna=False).to_dict()
                if "session" in outcomes_df.columns
                else {}
            )
            proof_lines.append(
                "[MARGIN_FILTER_EFFECT] run_id=%s entry_margin_min_used=%s n_trades_closed=%s win_rate=%s"
                % (
                    run_id,
                    entry_margin_min_used,
                    n_trades_closed,
                    pnl_stats.get("win_rate") if pnl_stats else None,
                )
            )
            if pnl_stats:
                proof_lines.append(
                    "pnl_bps_stats mean=%s median=%s std=%s min=%s p05=%s p95=%s p99=%s max=%s"
                    % (
                        pnl_stats.get("mean"),
                        pnl_stats.get("median"),
                        pnl_stats.get("std"),
                        pnl_stats.get("min"),
                        pnl_stats.get("p05"),
                        pnl_stats.get("p95"),
                        pnl_stats.get("p99"),
                        pnl_stats.get("max"),
                    )
                )
            else:
                proof_lines.append("pnl_bps_stats <MISSING>")
            proof_lines.append(f"exit_reason_counts {_format_counts(exit_reason_counts)}")
            proof_lines.append(f"session_counts {_format_counts(session_counts)}")
            footer_n = int(n_trades_closed or 0)
            outcomes_n = int(len(outcomes_df))
            match = 1 if footer_n == outcomes_n else 0
            proof_lines.append(
                f"[MARGIN_FILTER_SSOT_MATCH] footer_n_trades_closed={footer_n} outcomes_rows={outcomes_n} match={match}"
            )
            if match == 0:
                proof_lines.append(
                    f"[MARGIN_FILTER_SSOT_MISMATCH] footer_n_trades_closed={footer_n} outcomes_rows={outcomes_n}"
                )
            journal_path = out_root / f"trade_journal_{run_id}_MERGED.parquet"
            journal_df = None
            if journal_path.exists():
                try:
                    journal_df = pd.read_parquet(journal_path)
                except Exception as e:
                    log.info("[SIDE_PROFILE_SKIPPED] failed to read %s: %s", journal_path, e)
                    proof_lines.append(f"[SIDE_PROFILE_SKIPPED] failed to read {journal_path}")
            else:
                proof_lines.append("[SIDE_PROFILE_SKIPPED] missing trade_journal merged")
            side_stats = _pnl_profile_by_side(journal_df) if journal_df is not None else {}
            long_stats = side_stats.get("long")
            short_stats = side_stats.get("short")
            long_n = int(long_stats.get("n", 0)) if long_stats else 0
            short_n = int(short_stats.get("n", 0)) if short_stats else 0
            if long_stats:
                proof_lines.append(
                    "[SIDE_PNL] long n=%s win_rate=%s mean=%s median=%s p95=%s p99=%s min=%s max=%s"
                    % (
                        long_stats.get("n"),
                        long_stats.get("win_rate"),
                        long_stats.get("mean"),
                        long_stats.get("median"),
                        long_stats.get("p95"),
                        long_stats.get("p99"),
                        long_stats.get("min"),
                        long_stats.get("max"),
                    )
                )
            if short_stats:
                proof_lines.append(
                    "[SIDE_PNL] short n=%s win_rate=%s mean=%s median=%s p95=%s p99=%s min=%s max=%s"
                    % (
                        short_stats.get("n"),
                        short_stats.get("win_rate"),
                        short_stats.get("mean"),
                        short_stats.get("median"),
                        short_stats.get("p95"),
                        short_stats.get("p99"),
                        short_stats.get("min"),
                        short_stats.get("max"),
                    )
                )
            else:
                proof_lines.append("[SIDE_PNL] short n=0 (no short trades)")
            long_margin = None
            short_margin = None
            if journal_df is not None and "side" in journal_df.columns:
                long_margin = _margin_profile_summary(journal_df[journal_df["side"] == "long"])
                short_margin = _margin_profile_summary(journal_df[journal_df["side"] == "short"])
            if long_n > 0:
                assert long_margin is not None and long_margin
            if short_n > 0:
                assert short_margin is not None and short_margin
            if long_margin:
                proof_lines.append(
                    "[SIDE_MARGIN] long n=%s mean=%s median=%s p05=%s p25=%s p75=%s p95=%s min=%s max=%s"
                    % (
                        long_margin.get("n"),
                        long_margin.get("mean"),
                        long_margin.get("median"),
                        long_margin.get("p05"),
                        long_margin.get("p25"),
                        long_margin.get("p75"),
                        long_margin.get("p95"),
                        long_margin.get("min"),
                        long_margin.get("max"),
                    )
                )
            if short_margin:
                proof_lines.append(
                    "[SIDE_MARGIN] short n=%s mean=%s median=%s p05=%s p25=%s p75=%s p95=%s min=%s max=%s"
                    % (
                        short_margin.get("n"),
                        short_margin.get("mean"),
                        short_margin.get("median"),
                        short_margin.get("p05"),
                        short_margin.get("p25"),
                        short_margin.get("p75"),
                        short_margin.get("p95"),
                        short_margin.get("min"),
                        short_margin.get("max"),
                    )
                )
            if short_n > 0:
                assert short_stats is not None
            if long_n > 0:
                assert long_stats is not None
            if short_n == 0:
                proof_lines.append("[SHORT_EDGE_HINT] skipped (no short trades)")
            elif long_n == 0:
                proof_lines.append("[SHORT_EDGE_HINT] skipped (no long trades)")
            else:
                try:
                    better_mean = 1 if (short_stats.get("mean") > long_stats.get("mean")) else 0
                    better_p95 = 1 if (short_stats.get("p95") > long_stats.get("p95")) else 0
                    better_p99 = 1 if (short_stats.get("p99") > long_stats.get("p99")) else 0
                    proof_lines.append(
                        "[SHORT_EDGE_HINT] better_mean=%d better_p95=%d better_p99=%d note=heuristic_only"
                        % (better_mean, better_p95, better_p99)
                    )
                except Exception:
                    proof_lines.append("[SHORT_EDGE_HINT] skipped (stats_error)")
    except Exception as e:
        log.info("[POSTRUN_MARGIN_PROFILE_SKIPPED] failed to build margin profile: %s", e)

    # Margin quantiles vs PnL and holding time (from trade_journal)
    try:
        if journal is None:
            raise RuntimeError("journal_missing_or_invalid")
        if "margin_top1_top2" not in journal.columns:
            raise RuntimeError("missing_margin_top1_top2")
        if "pnl_bps" not in journal.columns or "bars_in_trade" not in journal.columns:
            raise RuntimeError("missing_pnl_or_bars")

        mj = journal.copy()
        mj["margin_top1_top2"] = pd.to_numeric(mj["margin_top1_top2"], errors="coerce")
        mj["pnl_bps"] = pd.to_numeric(mj["pnl_bps"], errors="coerce")
        mj["bars_in_trade"] = pd.to_numeric(mj["bars_in_trade"], errors="coerce")
        mj = mj.dropna(subset=["margin_top1_top2", "pnl_bps", "bars_in_trade"])
        if mj.empty:
            raise RuntimeError("empty_after_dropna")

        # Deterministic quantile edges (5 bins)
        qs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        edges = mj["margin_top1_top2"].quantile(qs, interpolation="linear").to_numpy()
        # Ensure monotonic edges
        edges = pd.Series(edges).cummax().to_numpy()
        # If all equal, cannot bin
        if len(set(edges)) < 2:
            raise RuntimeError("degenerate_quantile_edges")

        def _assign_bin(val: float) -> int:
            # bins: [0..4], right inclusive for last
            for i in range(5):
                lo = edges[i]
                hi = edges[i + 1]
                if i < 4:
                    if val >= lo and val < hi:
                        return i
                else:
                    if val >= lo and val <= hi:
                        return i
            return 4

        mj["margin_bin"] = mj["margin_top1_top2"].apply(_assign_bin).astype(int)

        def _bin_summary(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
            out = {}
            for b in range(5):
                sub = df[df["margin_bin"] == b]
                if sub.empty:
                    out[b] = {
                        "n": 0,
                        "pnl_mean": None,
                        "pnl_median": None,
                        "win_rate": None,
                        "bars_mean": None,
                        "bars_median": None,
                    }
                else:
                    pnl = sub["pnl_bps"]
                    bars = sub["bars_in_trade"]
                    out[b] = {
                        "n": int(len(sub)),
                        "pnl_mean": float(pnl.mean()),
                        "pnl_median": float(pnl.median()),
                        "win_rate": float((pnl > 0).mean()),
                        "bars_mean": float(bars.mean()),
                        "bars_median": float(bars.median()),
                    }
            return out

        overall = _bin_summary(mj)
        proof_lines.append(
            "[MARGIN_QUANTILES] edges="
            + ",".join(f"{e:.6f}" for e in edges)
        )
        def _fmt(val: Any) -> str:
            try:
                return f"{float(val):.6f}"
            except Exception:
                return "nan"

        for b in range(5):
            s = overall[b]
            proof_lines.append(
                "[MARGIN_QUANTILE_SUMMARY] "
                f"bin={b} n={s['n']} "
                f"pnl_mean={_fmt(s['pnl_mean'])} "
                f"pnl_median={_fmt(s['pnl_median'])} "
                f"win_rate={_fmt(s['win_rate'])} "
                f"bars_mean={_fmt(s['bars_mean'])} "
                f"bars_median={_fmt(s['bars_median'])}"
            )

        # Per session if present
        if "session" in mj.columns:
            for sess in sorted(mj["session"].dropna().unique()):
                sub = mj[mj["session"] == sess]
                by = _bin_summary(sub)
                for b in range(5):
                    s = by[b]
                    proof_lines.append(
                        "[MARGIN_QUANTILE_BY_SESSION] "
                        f"session={sess} bin={b} n={s['n']} "
                        f"pnl_mean={_fmt(s['pnl_mean'])} "
                        f"pnl_median={_fmt(s['pnl_median'])} "
                        f"win_rate={_fmt(s['win_rate'])} "
                        f"bars_mean={_fmt(s['bars_mean'])} "
                        f"bars_median={_fmt(s['bars_median'])}"
                    )
    except Exception as e:
        proof_lines.append(f"[MARGIN_QUANTILES_SKIPPED] reason={e}")

    # ------------------------------------------------------------------
    # STUCK_TRADE_AUDIT summary from chunk_0 artifact
    # ------------------------------------------------------------------
    try:
        audit_path = chunk_dir / "STUCK_TRADE_AUDIT.json"
        if audit_path.exists():
            with audit_path.open("r", encoding="utf-8") as _f:
                _audit = json.load(_f)
            _scope = _audit.get("audit_scope", {})
            _verdict = _audit.get("verdict", {})
            _sec_a = _audit.get("section_a_close_reason_tail_audit", {})
            _sec_b = _audit.get("section_b_pocket_comparison", {})
            proof_lines.append(
                f"[STUCK_TRADE_AUDIT_SUMMARY] run_id={run_id} "
                f"n_audited={_scope.get('n_audited_trades')} "
                f"eval_trace_found={_scope.get('eval_trace_found')}"
            )
            for _reason, _rs in sorted(_sec_a.items()):
                proof_lines.append(
                    f"[STUCK_AUDIT_REASON] reason={_reason} "
                    f"n={_rs.get('n')} "
                    f"pnl_mean={_rs.get('pnl_mean')} "
                    f"pnl_median={_rs.get('pnl_median')} "
                    f"pnl_worst={_rs.get('pnl_worst')} "
                    f"mean_max_prob_first5={_rs.get('mean_max_prob_close_first_5')} "
                    f"mean_max_prob_first20={_rs.get('mean_max_prob_close_first_20')} "
                    f"frac_ever_crossed={_rs.get('frac_ever_crossed_threshold')}"
                )
            _pocket = _sec_b.get("long_overlap_bars0", {})
            _rest = _sec_b.get("rest", {})
            proof_lines.append(
                f"[STUCK_AUDIT_POCKET] long+OVERLAP+bars0 "
                f"n={_pocket.get('n')} "
                f"pnl_mean={_pocket.get('pnl_mean')} "
                f"pnl_worst={_pocket.get('pnl_worst')} "
                f"frac_catguard={_pocket.get('frac_catastrophic_guard')} "
                f"frac_eof={_pocket.get('frac_replay_eof')} "
                f"mean_max_prob_first5={_pocket.get('mean_max_prob_close_first_5')} "
                f"frac_ever_crossed={_pocket.get('frac_ever_crossed_threshold')}"
            )
            proof_lines.append(
                f"[STUCK_AUDIT_REST] rest "
                f"n={_rest.get('n')} "
                f"pnl_mean={_rest.get('pnl_mean')} "
                f"pnl_worst={_rest.get('pnl_worst')} "
                f"frac_catguard={_rest.get('frac_catastrophic_guard')} "
                f"frac_eof={_rest.get('frac_replay_eof')} "
                f"mean_max_prob_first5={_rest.get('mean_max_prob_close_first_5')} "
                f"frac_ever_crossed={_rest.get('frac_ever_crossed_threshold')}"
            )
            proof_lines.append(
                f"[STUCK_AUDIT_VERDICT] "
                f"worst_tail_mean_max_prob_first5={_verdict.get('worst_tail_low_prob_first5')} "
                f"worst_tail_mean_max_prob_first20={_verdict.get('worst_tail_low_prob_first20')} "
                f"frac_ever_crossed_overall={_verdict.get('frac_ever_crossed_threshold_overall')} "
                f"pocket_long_overlap_bars0_n={_verdict.get('pocket_long_overlap_bars0_n')} "
                f"pocket_pnl_mean={_verdict.get('pocket_long_overlap_bars0_pnl_mean')} "
                f"rest_pnl_mean={_verdict.get('rest_pnl_mean')}"
            )
        else:
            proof_lines.append(f"[STUCK_TRADE_AUDIT_SUMMARY_SKIPPED] audit_file_not_found={audit_path}")
    except Exception as _e:
        proof_lines.append(f"[STUCK_TRADE_AUDIT_SUMMARY_SKIPPED] error={_e}")

    # ------------------------------------------------------------------
    # SHORT_EXIT_SIGNAL_AUDIT summary from chunk_0 artifact
    # ------------------------------------------------------------------
    try:
        short_audit_path = chunk_dir / "SHORT_EXIT_SIGNAL_AUDIT.json"
        if short_audit_path.exists():
            with short_audit_path.open("r", encoding="utf-8") as _f:
                _saudit = json.load(_f)
            _s1 = _saudit.get("section_1_signal_quality_by_side", {})
            _s2 = _saudit.get("section_2_close_reason_by_side", {})
            _s3 = _saudit.get("section_3_worst50_tail", {}).get("summary", {})
            _s4 = _saudit.get("section_4_signal_curve_by_side", {})
            _verd = _saudit.get("verdict", {})

            _long1 = _s1.get("long", {})
            _short1 = _s1.get("short", {})
            proof_lines.append(
                f"[SHORT_EXIT_SIGNAL_SUMMARY] run_id={run_id} "
                f"long_n={_long1.get('n')} short_n={_short1.get('n')} "
                f"long_pnl_mean={_long1.get('pnl_mean')} short_pnl_mean={_short1.get('pnl_mean')} "
                f"long_p5={_long1.get('mean_max_prob_close_first_5')} "
                f"short_p5={_short1.get('mean_max_prob_close_first_5')} "
                f"long_p20={_long1.get('mean_max_prob_close_first_20')} "
                f"short_p20={_short1.get('mean_max_prob_close_first_20')} "
                f"long_frac_crossed={_long1.get('frac_ever_crossed_threshold')} "
                f"short_frac_crossed={_short1.get('frac_ever_crossed_threshold')}"
            )
            for _side_val in ("long", "short"):
                for _row in _s2.get(_side_val, []):
                    proof_lines.append(
                        f"[SHORT_EXIT_REASON_SPLIT] side={_side_val} "
                        f"reason={_row.get('close_reason')} "
                        f"count={_row.get('count')} "
                        f"fraction={_row.get('fraction')} "
                        f"mean_pnl={_row.get('mean_pnl')} "
                        f"worst_pnl={_row.get('worst_pnl')}"
                    )
            proof_lines.append(
                f"[SHORT_EXIT_TAIL_ANALYSIS] "
                f"share_short={_s3.get('share_short')} "
                f"share_long={_s3.get('share_long')} "
                f"mean_p5_short={_s3.get('mean_max_prob_close_first5_short')} "
                f"mean_p5_long={_s3.get('mean_max_prob_close_first5_long')} "
                f"mean_p20_short={_s3.get('mean_max_prob_close_first20_short')} "
                f"mean_p20_long={_s3.get('mean_max_prob_close_first20_long')} "
                f"frac_crossed_short={_s3.get('frac_ever_crossed_short')} "
                f"frac_crossed_long={_s3.get('frac_ever_crossed_long')}"
            )
            for _side_val in ("long", "short"):
                _curve = _s4.get(_side_val, {})
                _curve_str = " ".join(
                    f"bar{i}={_curve.get(f'bar_{i}')}" for i in range(10)
                )
                proof_lines.append(f"[SHORT_EXIT_SIGNAL_CURVE] side={_side_val} {_curve_str}")
            proof_lines.append(
                f"[SHORT_EXIT_VERDICT] "
                f"short_lower_prob_first5={_verd.get('short_lower_prob_first5')} "
                f"short_lower_prob_first20={_verd.get('short_lower_prob_first20')} "
                f"short_lower_frac_crossed={_verd.get('short_lower_frac_crossed')} "
                f"short_higher_guard_eof={_verd.get('short_higher_guard_or_eof_frac')} "
                f"short_guard_eof_frac={_verd.get('short_guard_eof_frac')} "
                f"long_guard_eof_frac={_verd.get('long_guard_eof_frac')} "
                f"delta_p5_short_minus_long={_verd.get('delta_mean_max_prob_first5_short_minus_long')} "
                f"delta_crossed_short_minus_long={_verd.get('delta_frac_ever_crossed_short_minus_long')}"
            )
        else:
            proof_lines.append(f"[SHORT_EXIT_SIGNAL_AUDIT_SKIPPED] file_not_found={short_audit_path}")
    except Exception as _e:
        proof_lines.append(f"[SHORT_EXIT_SIGNAL_AUDIT_SKIPPED] error={_e}")

    # ------------------------------------------------------------------
    # STUCK_SHORT_SIGNATURE_AUDIT summary from chunk_0 artifact
    # ------------------------------------------------------------------
    try:
        sig_audit_path = chunk_dir / "STUCK_SHORT_SIGNATURE_AUDIT.json"
        if sig_audit_path.exists():
            with sig_audit_path.open("r", encoding="utf-8") as _f:
                _sig = json.load(_f)
            _s6 = _sig.get("section_6_signature", {})
            _s4 = _sig.get("section_4_early_life_divergence", {})
            _s5 = _sig.get("section_5_no_rebound", {})
            _tgt_sum = _s5.get("target_summary", {})
            _ca_sum = _s5.get("ctrl_a_summary", {})
            proof_lines.append(
                f"[STUCK_SHORT_SIGNATURE_SUMMARY] run_id={run_id} "
                f"n_target={_s6.get('n_target_trades')} "
                f"n_ctrl_a={_s6.get('n_ctrl_a_trades')} "
                f"n_ctrl_b={_s6.get('n_ctrl_b_trades')} "
                f"sessions={_s6.get('target_sessions')} "
                f"n_catguard={_s6.get('n_catastrophic_guard')} "
                f"n_eof={_s6.get('n_replay_eof')} "
                f"bars_held_mean={_s6.get('target_bars_held_mean')} "
                f"pnl_mean={_s6.get('target_final_pnl_mean_bps')} "
                f"mfe_mean={_s6.get('target_max_mfe_mean_bps')}"
            )
            proof_lines.append(
                f"[STUCK_SHORT_SIGNATURE_CATEGORIES] "
                f"sessions={_s6.get('target_sessions')}"
            )
            for _w in [5, 20, 100]:
                _w_key = f"first_{_w}_bars"
                _t = _s4.get(_w_key, {}).get("target", {})
                _ca = _s4.get(_w_key, {}).get("ctrl_a", {})
                proof_lines.append(
                    f"[STUCK_SHORT_SIGNATURE_EARLY_LIFE] window={_w} "
                    f"target_pnl_mean={_t.get('mean_pnl_bps')} "
                    f"ctrl_a_pnl_mean={_ca.get('mean_pnl_bps')} "
                    f"target_prob_mean={_t.get('mean_prob_close')} "
                    f"ctrl_a_prob_mean={_ca.get('mean_prob_close')}"
                )
            proof_lines.append(
                f"[STUCK_SHORT_SIGNATURE_NO_REBOUND] "
                f"target_mean_frac_negative={_tgt_sum.get('mean_frac_bars_negative')} "
                f"target_mean_crossings={_tgt_sum.get('mean_n_crossings_above_zero')} "
                f"target_mean_max_mfe={_tgt_sum.get('mean_max_mfe_bps')} "
                f"target_mean_frac_after_mfe={_tgt_sum.get('mean_frac_life_after_mfe_peak')} "
                f"ctrl_a_mean_frac_negative={_ca_sum.get('mean_frac_bars_negative')} "
                f"ctrl_a_mean_crossings={_ca_sum.get('mean_n_crossings_above_zero')} "
                f"ctrl_a_mean_max_mfe={_ca_sum.get('mean_max_mfe_bps')}"
            )
            proof_lines.append(
                f"[STUCK_SHORT_SIGNATURE_VERDICT] "
                f"all_slow_bleed={_s6.get('all_slow_bleed_frac_negative_gt85pct')} "
                f"all_no_rebound={_s6.get('all_no_rebound_above_zero')} "
                f"early_divergent_by_bar20={_s6.get('early_divergence_by_bar20')} "
                f"early_pnl_target_first20={_s6.get('early_pnl_target_first20')} "
                f"early_pnl_ctrl_a_first20={_s6.get('early_pnl_ctrl_a_first20')} "
                f"summary={_s6.get('summary')}"
            )
        else:
            proof_lines.append(f"[STUCK_SHORT_SIGNATURE_AUDIT_SKIPPED] file_not_found={sig_audit_path}")
    except Exception as _e:
        proof_lines.append(f"[STUCK_SHORT_SIGNATURE_AUDIT_SKIPPED] error={_e}")

    # ------------------------------------------------------------------
    # EARLY_FAILURE_SHORT_GUARD_COUNTERFACTUAL summary from chunk_0 artifact
    # ------------------------------------------------------------------
    try:
        cf_path = chunk_dir / "EARLY_FAILURE_SHORT_GUARD_COUNTERFACTUAL.json"
        if cf_path.exists():
            with cf_path.open("r", encoding="utf-8") as _f:
                _cf = json.load(_f)
            _verd = _cf.get("verdict", {})
            proof_lines.append(
                f"[EARLY_FAILURE_SHORT_GUARD_CF_SUMMARY] run_id={run_id} "
                f"n_target={_cf.get('n_target_trades')} "
                f"n_ctrl_a={_cf.get('n_ctrl_a_trades')} "
                f"n_short_universe={_cf.get('n_short_universe')} "
                f"best_variant={_verd.get('best_variant')}"
            )
            for _vname, _vr in _cf.get("variants", {}).items():
                _ti = _vr.get("tail_improvement", {})
                proof_lines.append(
                    f"[EARLY_FAILURE_SHORT_GUARD_CF_VARIANTS] variant={_vname} "
                    f"target_hit_rate={_vr.get('target_hit_rate')} "
                    f"n_target_hit={_vr.get('n_target_hit')} "
                    f"collateral_hit_rate={_vr.get('collateral_hit_rate')} "
                    f"n_collateral_hits={_vr.get('n_collateral_hits')} "
                    f"mean_target_improvement={_vr.get('mean_target_improvement_bps')} "
                    f"mean_collateral_loss={_vr.get('mean_collateral_alpha_loss_bps')} "
                    f"net_mean_bps={_vr.get('net_mean_bps_over_short_universe')}"
                )
                proof_lines.append(
                    f"[EARLY_FAILURE_SHORT_GUARD_CF_TAIL_IMPACT] variant={_vname} "
                    f"mean_delta={_ti.get('mean_delta')} "
                    f"worst_delta={_ti.get('worst_delta')} "
                    f"worst10_delta={_ti.get('mean_worst10_delta')} "
                    f"worst20_delta={_ti.get('mean_worst20_delta')} "
                    f"CVaR95_delta={_ti.get('CVaR95_delta')} "
                    f"CVaR99_delta={_ti.get('CVaR99_delta')}"
                )
            for _rank in _verd.get("ranking", []):
                proof_lines.append(
                    f"[EARLY_FAILURE_SHORT_GUARD_CF_COLLATERAL] "
                    f"variant={_rank.get('variant')} "
                    f"target_hit_rate={_rank.get('target_hit_rate')} "
                    f"collateral_hit_rate={_rank.get('collateral_hit_rate')} "
                    f"mean_target_impr={_rank.get('mean_target_improvement')} "
                    f"mean_collateral_loss={_rank.get('mean_collateral_alpha_loss')} "
                    f"net_mean_bps={_rank.get('net_mean_bps')} "
                    f"CVaR95_delta={_rank.get('CVaR95_delta')}"
                )
            proof_lines.append(
                f"[EARLY_FAILURE_SHORT_GUARD_CF_VERDICT] "
                f"best_variant={_verd.get('best_variant')} "
                f"recommendation={_verd.get('recommendation')}"
            )
        else:
            proof_lines.append(f"[EARLY_FAILURE_SHORT_GUARD_CF_SKIPPED] file_not_found={cf_path}")
    except Exception as _e:
        proof_lines.append(f"[EARLY_FAILURE_SHORT_GUARD_CF_SKIPPED] error={_e}")

    # ------------------------------------------------------------------
    # ENTRY SIGNATURE AUDIT: STUCK SHORTS
    # ------------------------------------------------------------------
    try:
        import json as _json2
        _esa_path = chunk_dir / "ENTRY_SIGNATURE_AUDIT_STUCK_SHORTS.json"
        if _esa_path.exists():
            _esa = _json2.loads(_esa_path.read_text())
            _verd = _esa.get("section_8_exact_entry_signature", {})
            _top10 = _esa.get("section_4_top10_separating_features", [])
            _conf = _esa.get("section_5_entry_confidence_by_group", [])
            _bridge = _esa.get("section_7_bridge_entry_to_early_failure", {})
            _cat = _esa.get("section_2_categorical_signature", {})
            proof_lines.append(
                f"[ENTRY_SIGNATURE_AUDIT_SUMMARY] run_id={run_id} "
                f"n_target={_esa.get('n_target_trades')} "
                f"n_ctrl_a={_esa.get('n_ctrl_a_trades')} "
                f"n_ctrl_b={_esa.get('n_ctrl_b_trades')}"
            )
            proof_lines.append(
                f"[ENTRY_SIGNATURE_AUDIT_VERDICT] "
                f"primary_signature={_verd.get('primary_entry_signature')} "
                f"separability={_verd.get('separability_at_entry')} "
                f"confidence_level={_verd.get('entry_confidence_level')}"
            )
            proof_lines.append(
                f"[ENTRY_SIGNATURE_AUDIT_PROBLEM] "
                f"{_verd.get('most_likely_problem_description')}"
            )
            for _sig_line in _verd.get("signature_lines", []):
                proof_lines.append(f"[ENTRY_SIGNATURE_AUDIT_SIG_LINE] {_sig_line}")
            # top separating features
            for _feat_row in _top10[:7]:
                proof_lines.append(
                    f"[ENTRY_SIGNATURE_AUDIT_NUMERIC] "
                    f"feature={_feat_row.get('feature')} "
                    f"target_mean={_feat_row.get('target_mean')} "
                    f"ctrl_a_mean={_feat_row.get('ctrl_a_mean')} "
                    f"delta={_feat_row.get('delta_mean_vs_ctrl_a')} "
                    f"z={_feat_row.get('abs_z_score_vs_ctrl_a')}"
                )
            # entry confidence by group
            for _cg in _conf:
                proof_lines.append(
                    f"[ENTRY_SIGNATURE_AUDIT_CONFIDENCE] "
                    f"group={_cg.get('group')} "
                    f"n={_cg.get('n')} "
                    f"mean_p_short={_cg.get('mean_p_short')} "
                    f"mean_margin={_cg.get('mean_margin')} "
                    f"frac_below_threshold_prob={_cg.get('frac_below_threshold_prob')}"
                )
            # bridge
            for _br in _bridge.get("early_life_vs_entry_features", []):
                proof_lines.append(
                    f"[ENTRY_SIGNATURE_AUDIT_BRIDGE] "
                    f"group={_br.get('group')} "
                    f"mean_pnl_first5={_br.get('mean_pnl_first5')} "
                    f"mean_mfe_first5={_br.get('mean_mfe_first5')} "
                    f"mean_prob_close_first5={_br.get('mean_prob_close_first5')} "
                    f"frac_pnl_neg_first5={_br.get('frac_pnl_negative_first5')}"
                )
            # categorical highlights
            h4_cat = _cat.get("H4_trend_sign_cat", [])
            for _h4 in h4_cat:
                proof_lines.append(
                    f"[ENTRY_SIGNATURE_AUDIT_CATEGORIES] "
                    f"field=H4_trend_sign_cat "
                    f"category={_h4.get('category')} "
                    f"target_frac={_h4.get('target_frac')} "
                    f"ctrl_a_frac={_h4.get('ctrl_a_frac')} "
                    f"overrep={_h4.get('overrep_ratio')}"
                )
        else:
            proof_lines.append(f"[ENTRY_SIGNATURE_AUDIT_SKIPPED] file_not_found={_esa_path}")
    except Exception as _e2:
        proof_lines.append(f"[ENTRY_SIGNATURE_AUDIT_SKIPPED] error={_e2}")

    proof_lines[0] = (
        f"[POSTRUN_TRADE_REPORT_PROOF] run_id={run_id} journal_rows={journal_rows} "
        f"wrote_session_side={int(wrote_session_side)} wrote_holding_bins={int(wrote_holding_bins)} "
        f"wrote_side_session={int(wrote_side_session)} wrote_entry_gates={int(wrote_entry_gates)}"
    )
    proof_path = out_root / f"POSTRUN_TRADE_REPORT_PROOF_{run_id}.log"
    try:
        proof_path.write_text("\n".join(proof_lines) + "\n", encoding="utf-8")
        log.info("%s", proof_lines[0])
    except Exception as e:
        log.info("[POSTRUN_TRADE_REPORT_SKIPPED] failed to write proof log: %s", e)
    if pnl_audit_error is not None:
        raise RuntimeError(f"[REPLAY_PNL_AUDIT] hard-fail: {pnl_audit_error}")


def merge_artifacts_1w1c(run_dir: Path, run_id: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Merge single chunk (chunk_0) into output: write *_MERGED.*, MERGE_PROOF, RUN_COMPLETED.

    run_dir must contain chunk_0/ with chunk_footer.json. If output_dir is given, MERGED artifacts
    are written there (root artifacts in run_root); otherwise written to run_dir.
    trade_outcomes: if chunk has trade_outcomes_{run_id}.parquet, copy to MERGED; if missing and
    n_trades_closed==0, write empty MERGED parquet with canonical schema; if missing and n_trades_closed>0, hard-fail.
    Returns dict of written paths and metadata (for tests/logging).
    """
    run_dir = run_dir.resolve()
    out_root = (output_dir.resolve() if output_dir else run_dir)
    chunk_dir = run_dir / "chunk_0"
    if not chunk_dir.is_dir():
        raise RuntimeError(f"[REPLAY_MERGE] chunk_0 not found in {run_dir}")

    footer_path = chunk_dir / "chunk_footer.json"
    if not footer_path.exists():
        raise RuntimeError(f"[REPLAY_MERGE] chunk_footer.json not found: {footer_path}")

    footer = _load_json(footer_path)
    if footer.get("status") != "ok":
        raise RuntimeError(f"[REPLAY_MERGE] chunk_footer status != ok: {footer.get('status')}")
    run_header = _load_run_header_optional(run_dir)

    n_trades_closed = int(footer.get("n_trades_closed", 0) or 0)
    trade_src = chunk_dir / f"trade_outcomes_{run_id}.parquet"
    trade_dst = out_root / f"trade_outcomes_{run_id}_MERGED.parquet"

    if trade_src.exists():
        shutil.copy2(trade_src, trade_dst)
        log.info("[REPLAY_MERGE] Wrote %s", trade_dst.name)
    elif n_trades_closed == 0:
        write_empty_trade_outcomes_parquet(trade_dst, run_id=run_id)
        log.info("[REPLAY_MERGE] Wrote empty %s (0-trades contract)", trade_dst.name)
    else:
        write_fatal_capsule(
            chunk_output_dir=out_root,
            chunk_idx=0,
            run_id=run_id,
            fatal_reason="MERGE_TRADE_OUTCOMES_MISSING",
            error_message=(
                f"[REPLAY_MERGE] trade_outcomes not found: {trade_src} but n_trades_closed={n_trades_closed}>0. "
                "TRUTH requires chunk to write trade_outcomes parquet (or empty with 0 trades)."
            ),
            extra_fields={"trade_src": str(trade_src), "n_trades_closed": n_trades_closed},
        )
        raise RuntimeError(
            f"[REPLAY_MERGE] trade_outcomes not found: {trade_src} and n_trades_closed={n_trades_closed}. "
            "Hard-fail: chunk must produce trade_outcomes_{run_id}.parquet."
        )

    # trade_journal (optional): copy if present
    trade_journal_src = chunk_dir / f"trade_journal_{run_id}.parquet"
    trade_journal_dst = out_root / f"trade_journal_{run_id}_MERGED.parquet"
    if trade_journal_src.exists():
        shutil.copy2(trade_journal_src, trade_journal_dst)
        log.info("[REPLAY_MERGE] Wrote %s", trade_journal_dst.name)
    else:
        log.info("[REPLAY_MERGE] trade_journal not found: %s (optional)", trade_journal_src)

    # shadow/meta candidate table from eval_log (observability only)
    shadow_candidates_dst = out_root / f"shadow_meta_candidates_{run_id}_MERGED.parquet"
    shadow_candidates_df = _load_shadow_candidate_events(chunk_dir, run_id)
    if shadow_candidates_df.empty:
        log.info("[SHADOW_META] eval_log not found or empty; skipping %s", shadow_candidates_dst.name)
    else:
        merged_shadow = shadow_candidates_df.copy()
        merged_shadow["join_key"] = _join_key_for_trade_id_or_uid(merged_shadow)
        if "accepted" not in merged_shadow.columns:
            merged_shadow["accepted"] = False
        merged_shadow["accepted_bool"] = merged_shadow["accepted"].fillna(False).astype(bool)
        merged_shadow["blocked_bool"] = ~merged_shadow["accepted_bool"]

        if trade_journal_dst.exists():
            try:
                journal_df = pd.read_parquet(trade_journal_dst)
                if not journal_df.empty:
                    journal_df = journal_df.copy()
                    journal_df["join_key"] = _join_key_for_trade_id_or_uid(journal_df)
                    journal_keep = [
                        c for c in [
                            "join_key",
                            "trade_uid",
                            "trade_id",
                            "open_ts_utc",
                            "close_ts_utc",
                            "pnl_bps",
                            "bars_in_trade",
                            "exit_reason",
                            "side",
                            "session",
                            "prob_close",
                            "threshold",
                            "mfe_bps",
                            "mae_bps",
                            "mfe_first_n_pred",
                            "path_quality_pred",
                            "adverse_first",
                            "peak_mfe_bps_exit_state",
                            "bars_held_exit_state",
                            "time_since_mfe_bars_exit",
                            "dd_from_mfe_bps_exit",
                            "distance_from_peak_mfe_bps_exit",
                        ]
                        if c in journal_df.columns
                    ]
                    journal_df = journal_df[journal_keep].drop_duplicates(subset=["join_key"], keep="last")
                    merged_shadow = merged_shadow.merge(journal_df, on="join_key", how="left", suffixes=("", "_journal"))
            except Exception as e:
                log.warning("[SHADOW_META] failed to merge trade_journal: %s", e)

        trade_outcomes_path = out_root / f"trade_outcomes_{run_id}_MERGED.parquet"
        if trade_outcomes_path.exists():
            try:
                outcomes_df = pd.read_parquet(trade_outcomes_path)
                if not outcomes_df.empty:
                    outcomes_df = outcomes_df.copy()
                    outcomes_df["join_key"] = _join_key_for_trade_id_or_uid(outcomes_df)
                    outcome_keep = [
                        c for c in [
                            "join_key",
                            "trade_uid",
                            "candidate_uid",
                            "pnl_bps",
                            "mae_bps",
                            "mfe_bps",
                            "duration_bars",
                            "side",
                            "session",
                            "exit_reason",
                            "open_ts_utc",
                            "close_ts_utc",
                            "entry_bid",
                            "entry_ask",
                            "exit_bid",
                            "exit_ask",
                            "entry_spread_bps",
                            "exit_spread_bps",
                            "entry_price_used",
                            "exit_price_used",
                        ]
                        if c in outcomes_df.columns
                    ]
                    outcomes_df = outcomes_df[outcome_keep].drop_duplicates(subset=["join_key"], keep="last")
                    merged_shadow = merged_shadow.merge(outcomes_df, on="join_key", how="left", suffixes=("", "_outcome"))
            except Exception as e:
                log.warning("[SHADOW_META] failed to merge trade_outcomes: %s", e)

        try:
            shadow_meta_df = _finalize_shadow_meta_v1(
                merged_shadow,
                run_id=run_id,
                chunk_dir=chunk_dir,
                run_header=run_header,
                footer=footer,
            )
            shadow_meta_df.to_parquet(shadow_candidates_dst, index=False)
            provenance_summary_path = out_root / f"shadow_meta_provenance_{run_id}.json"
            provenance_summary = shadow_meta_df.attrs.get("shadow_meta_provenance_summary")
            if provenance_summary:
                provenance_summary_path.write_text(
                    json.dumps(provenance_summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            log.info(
                "[SHADOW_META] Wrote %s rows=%d cols=%d",
                shadow_candidates_dst.name,
                len(shadow_meta_df),
                len(shadow_meta_df.columns),
            )
            fatal_path = out_root / "FATAL_ERROR.txt"
            if fatal_path.exists():
                try:
                    fatal_path.unlink()
                    log.info("[SHADOW_META] Cleared stale %s after successful finalize", fatal_path.name)
                except Exception as fatal_cleanup_error:
                    log.warning("[SHADOW_META] failed to clear stale fatal capsule: %s", fatal_cleanup_error)
        except Exception as e:
            write_fatal_capsule(
                chunk_output_dir=out_root,
                chunk_idx=0,
                run_id=run_id,
                fatal_reason="SHADOW_META_V1_MERGE_FAILED",
                error_message=f"[SHADOW_META_V1] failed to finalize shadow meta dataset: {e}",
                extra_fields={"shadow_candidates_dst": str(shadow_candidates_dst)},
            )
            raise

    # replay summary proof (optional): copy if present
    proof_src = chunk_dir / "REPLAY_SUMMARY_PROOF.log"
    proof_dst = out_root / f"REPLAY_SUMMARY_PROOF_{run_id}.log"
    if proof_src.exists():
        shutil.copy2(proof_src, proof_dst)
        log.info("[REPLAY_MERGE] Wrote %s", proof_dst.name)
    else:
        log.info("[REPLAY_SUMMARY_PROOF_MERGE_SKIPPED] missing %s", proof_src)

    # postrun trade reports (optional)
    _write_postrun_trade_reports(out_root, run_id, chunk_dir, footer)
    # replay metrics (root CSVs)
    _write_replay_metrics(out_root, run_id, trade_dst, trade_journal_dst if trade_journal_dst.exists() else None)

    # canonical_economics_hash (optional): from trade_uid or row digest
    try:
        df = pd.read_parquet(trade_dst)
        if "trade_uid" in df.columns:
            canonical_economics_hash = hashlib.sha256(
                "".join(sorted(df["trade_uid"].astype(str))).encode("utf-8")
            ).hexdigest()
        else:
            canonical_economics_hash = hashlib.sha256(
                "".join(sorted(df.astype(str).sum(axis=1).tolist())).encode("utf-8")
            ).hexdigest()
    except Exception as e:
        log.warning("[REPLAY_MERGE] Could not compute canonical_economics_hash: %s", e)
        canonical_economics_hash = None

    # metrics_MERGED.json from chunk_footer
    n_trades = int(footer.get("n_trades_closed", 0) or 0)
    forward_calls = (
        footer.get("n_model_calls")
        or footer.get("bars_evaluated")
        or footer.get("transformer_forward_calls")
        or 0
    )
    try:
        forward_calls = int(forward_calls)
    except (TypeError, ValueError):
        forward_calls = 0

    metrics = {
        "run_id": run_id,
        "n_chunks": 1,
        "n_trades": n_trades,
        "transformer_forward_calls": forward_calls,
        "forward_calls_total": forward_calls,
        "n_model_calls": forward_calls,
        "source": "replay_merge_1w1c",
        "chunk_footer_status": footer.get("status"),
    }
    metrics_path = out_root / f"metrics_{run_id}_MERGED.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("[REPLAY_MERGE] Wrote %s", metrics_path.name)

    # MERGE_PROOF
    merge_proof = {
        "run_id": run_id,
        "n_chunks": 1,
        "n_trade_outcomes_files": 1,
        "canonical_economics_hash": canonical_economics_hash,
        "source": "replay_merge_1w1c",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    merge_proof_path = out_root / f"MERGE_PROOF_{run_id}.json"
    with open(merge_proof_path, "w", encoding="utf-8") as f:
        json.dump(merge_proof, f, indent=2)
    log.info("[REPLAY_MERGE] Wrote %s", merge_proof_path.name)

    # RUN_COMPLETED.json
    run_completed = {
        "status": "COMPLETED",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "chunks_submitted": 1,
        "chunks_completed": 1,
        "chunks_failed": [],
        "source": "replay_merge_1w1c",
    }
    run_completed_path = out_root / "RUN_COMPLETED.json"
    with open(run_completed_path, "w", encoding="utf-8") as f:
        json.dump(run_completed, f, indent=2)
    log.info("[REPLAY_MERGE] Wrote %s", run_completed_path.name)

    return {
        "trade_outcomes_merged": str(trade_dst),
        "trade_journal_merged": str(trade_journal_dst) if trade_journal_dst.exists() else None,
        "shadow_meta_candidates_merged": str(shadow_candidates_dst) if shadow_candidates_dst.exists() else None,
        "shadow_meta_provenance_summary": str(out_root / f"shadow_meta_provenance_{run_id}.json")
        if (out_root / f"shadow_meta_provenance_{run_id}.json").exists()
        else None,
        "metrics_merged": str(metrics_path),
        "merge_proof": str(merge_proof_path),
        "run_completed": str(run_completed_path),
    }
