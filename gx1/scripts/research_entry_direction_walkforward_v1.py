#!/usr/bin/env python3
"""Chronological walk-forward ceiling measurement for Entry direction.

Research instrument, no authority.  It answers one question the candidate
pipeline cannot answer on its 22-D1-bar VAL month: on expanding chronological
folds INSIDE the declared TRAIN window, does a cheap learner fitted on the
model-native decision surface select a subset of rows whose realized
gross-spread-inclusive research PnL beats the coin-flip null, per target
horizon, per feature arm, per target scaling and per seed?

Everything statistical is delegated to the pre-registered selective-edge
owner (``evaluate_entry_candidate_selective_edge_v1``): fixed coverage grid,
Newey-West HAC standard error of the paired advantage, exact coin-flip
expectation and circular-shift null.  This module only supplies the
per-fold prediction frames in that owner's schema and adds descriptive
slices (side counts, hit rate, up/down-month split) plus the stricter
``strict_pass`` that additionally requires ``mean_pnl_bps > 0``.

Learners: ridge (alpha chosen on an inner chronological split) and
scikit-learn's HistGradientBoostingRegressor (iterations chosen on the same
inner split from staged predictions; early stopping disabled so the library
never draws its own random validation split).  Both fit one regressor per
side (LONG, SHORT) and decide by unique argmax over (LONG, SHORT, FLAT=0),
i.e. the same decision semantics as ``entry_fitted_q_v1``.

Targets: the dataset's own knee-horizon research outcomes
(``y_*_final_pnl_at_direction_horizon_bps``) and executable fixed-horizon
close-to-close returns on the native M5 tape at explicit horizons.  The
tape-based targets use bar-close fills (ask at entry / bid at exit for LONG
and the mirror for SHORT); they are a research convention labelled as such,
not the M1 next-open fill of the dataset owner.

Scalings: ``raw`` fits bps directly; ``atr`` fits bps divided by the
decision-row ``ctx_cont.atr_bps`` and multiplies predictions back before the
argmax, so the decision stays in bps while the fit is not dominated by the
volatility tail.

Why a new module (rule 21): ``evaluate_entry_candidate_selective_edge_v1``
owns the statistics but its ``run`` is bound to a trained bundle's
prediction evidence and stage lineage, and
``materialize_entry_offline_challenger_v1`` reviews finished rolling-OOS
events and cannot fit anything.  A TRAIN-internal per-fold learner cannot
live in either without mixing contracts, so this bounded research
authority imports the evaluator's functions instead of duplicating them.

The tape reader filters before materializing rows, at TRAIN end (or VAL end
for an explicit development-VAL stage). Parquet may decode a mixed row group
internally; no beyond-boundary rows are returned to research calculations.
Fit/holdout outcome windows must remain within their chronological boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import subprocess
import time as _time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.features.htf_features import (
    MODEL_NATIVE_MTF_SCALAR_PER_BAR_EXACT_ALIASES_V4,
    MULTI_TF_SHIFT,
)
from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_TIMEFRAMES
from gx1.features.entry_specialist_feature_groups_v1 import (
    MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4,
    classify_entry_specialist_feature,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SELECTION_MODE,
    MODEL_DIRECTION_SHORT_INDEX,
)
from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
    MIN_PREREGISTERED_TRADE_ROWS,
    EVALUATION_COVERAGES,
    RESEARCH_LONG_OUTCOME_COLUMN,
    RESEARCH_SHORT_OUTCOME_COLUMN,
    build_metric_rows,
)

SCHEMA_VERSION = "entry_direction_walkforward_research_v1"
AUTHORITY = {
    "research_only": True,
    "candidate": False,
    "promotion": False,
    "test": False,
    "paper": False,
    "live": False,
}
KNEE_TARGET_NAME = "knee_m1_final_pnl"
M5_BAR = pd.Timedelta(minutes=5)
BPS = 1e4
ATR_SCALE_FIELD = "atr_bps"
LEARNERS = ("ridge", "hgb")
FEATURE_ARMS = ("snapshot", "snapshot_mtf", "snapshot_patterns", "snapshot_mtf_patterns", "snapshot_cross", "snapshot_mtf_cross")
ABLATION_MODES = ("none", "families", "lanes", "cross", "all")
# Cross-asset research block (operator decision 2026-09-23; research measurement only, Entry contract unchanged).
# Daily macro: log levels from the recovered research table (dxy, tnx, vix, realyld = log(TIP/IEF)); causal
# transforms declared here; aligned with a conservative one-calendar-day lag (a decision on day D uses day D-1's row).
CROSS_DAILY_INSTRUMENTS = ("dxy", "tnx", "vix", "realyld")
CROSS_DAILY_CHANGE_DAYS = (1, 5, 20)
CROSS_DAILY_Z_WINDOW_DAYS = 60
CROSS_DAILY_Z_MIN_DAYS = 20
CROSS_DAILY_LAG = pd.Timedelta(days=1)
CROSS_DAILY_LEVEL_INSTRUMENTS = ("vix",)  # the only level used as a (bounded) regime input; trending levels are excluded
# H1 FX bars (USD_JPY): last closed bar under the owner's MTF cutoff rule; log returns and realized vol in bars.
CROSS_H1_RETURN_BARS = (1, 4, 24, 120)
CROSS_H1_VOL_BARS = 24
CROSS_H1_MAX_STALENESS = pd.Timedelta(hours=72)
OOD_DISTANCE_GROUPS = ("mtf_lane:all", "patterns:all")
OOD_DISTANCE_COLUMNS = {"all": "ood_abs_z_mean", "mtf_lane:all": "ood_abs_z_mean_mtf", "patterns:all": "ood_abs_z_mean_patterns"}
TARGET_SCALINGS = ("raw", "atr")
DECISION_RULES = ("argmax_flat", "contrast_always_trade")
RIDGE_ALPHA_GRID = tuple(float(v) for v in np.logspace(-2.0, 4.0, 13))
# scikit-learn 1.7 HistGradientBoostingRegressor library defaults (origin: the pinned library, not a tuned choice).
HGB_LIBRARY_DEFAULT_LEARNING_RATE = 0.1
HGB_LIBRARY_DEFAULT_MIN_SAMPLES_LEAF = 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def feature_fit_lineage(binding: dict[str, Any]) -> list[dict[str, str]]:
    """Read learned feature fit bounds; last-closed joins alone do not prove chronology."""
    try:
        registry = binding["v29_registry_constants"]
        volatility_binding = binding["volatility_squeeze_artifact_set"]
        volatility_path = Path(volatility_binding["manifest_path"])
        if _sha256_file(volatility_path) != volatility_binding["manifest_file_sha256"]:
            raise RuntimeError("WALKFORWARD_FEATURE_FIT_MANIFEST_HASH_MISMATCH")
        volatility = json.loads(volatility_path.read_text(encoding="utf-8"))
        sources = (
            ("registry", registry, str(registry["contract_sha256"])),
            ("volatility_squeeze", volatility["common_train_lineage"],
             volatility_binding["manifest_file_sha256"]),
        )
        result = []
        for name, source, digest in sources:
            start = pd.Timestamp(source["declared_train_window_start"])
            end = pd.Timestamp(source["declared_train_window_end"])
            if pd.isna(start) or pd.isna(end) or start.tzinfo is None or end.tzinfo is None or start >= end:
                raise ValueError("invalid feature-fit interval")
            result.append({"owner": name, "fit_start": start.isoformat(),
                           "fit_end_exclusive": end.isoformat(), "binding_sha256": digest})
        return result
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError("WALKFORWARD_FEATURE_FIT_LINEAGE_MISSING_OR_INVALID") from exc


def require_feature_fit_before(
    lineage: list[dict[str, str]], cutoff: pd.Timestamp, *, context: str
) -> None:
    """Reject preprocessing learned from the period being evaluated, including inner selection."""
    cutoff = pd.Timestamp(cutoff)
    if not lineage or pd.isna(cutoff) or cutoff.tzinfo is None:
        raise RuntimeError("WALKFORWARD_FEATURE_FIT_CUTOFF_INVALID")
    for item in lineage:
        end = pd.Timestamp(item["fit_end_exclusive"])
        if pd.isna(end) or end.tzinfo is None:
            raise RuntimeError("WALKFORWARD_FEATURE_FIT_LINEAGE_MISSING_OR_INVALID")
        # Fit bounds are half-open; ending exactly at the evaluation start is allowed.
        if end > cutoff:
            raise RuntimeError(
                f"WALKFORWARD_FEATURE_FIT_AFTER_EVALUATION_START: {context} "
                f"owner={item['owner']} fitted_until={end.isoformat()} "
                f"evaluation_start={cutoff.isoformat()}"
            )

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _list_column_to_matrix(table: pa.Table, name: str, width: int, dtype: Any) -> np.ndarray:
    column = table.column(name).combine_chunks()
    if not pa.types.is_list(column.type):
        raise RuntimeError(f"WALKFORWARD_COLUMN_NOT_LIST: {name}")
    lengths = column.value_lengths().to_numpy(zero_copy_only=False)
    if column.null_count or not np.all(lengths == width):
        raise RuntimeError(f"WALKFORWARD_LIST_WIDTH_INVALID: {name} expected {width}")
    flat = column.flatten().to_numpy(zero_copy_only=False)
    matrix = np.asarray(flat, dtype=dtype).reshape(len(column), width)
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"WALKFORWARD_NONFINITE_INPUT: {name}")
    return matrix


@dataclass(frozen=True)
class Dataset:
    time: pd.DatetimeIndex
    snap: np.ndarray
    ctx_cont: np.ndarray
    ctx_cat: np.ndarray
    ctx_cont_names: tuple[str, ...]
    signal_names: tuple[str, ...]
    knee_long: np.ndarray
    knee_short: np.ndarray
    knee_horizon_bars: int
    manifest_sha256: str
    parquet_path: str


def load_dataset(dataset_dir: Path, *, split: str = "train") -> Dataset:
    if split not in ("train", "val"):
        raise RuntimeError("WALKFORWARD_SPLIT_INVALID")
    parquet = dataset_dir / f"entry_dataset__ENTRY_FITTED_Q_{split}.parquet"
    manifest_path = dataset_dir / f"entry_dataset__ENTRY_FITTED_Q_{split}.manifest.json"
    if not parquet.is_file() or not manifest_path.is_file():
        raise RuntimeError("WALKFORWARD_DATASET_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest["feature_contract"]
    ctx_cont_names = tuple(contract["ctx_cont_names"])
    if len(ctx_cont_names) != MODEL_NATIVE_CTX_CONT_DIM:
        raise RuntimeError("WALKFORWARD_CTX_CONT_DIM_MISMATCH")
    if tuple(contract["ctx_cat_names"]) != MODEL_NATIVE_CTX_CAT_FIELDS:
        raise RuntimeError("WALKFORWARD_CTX_CAT_FIELDS_MISMATCH")
    if len(contract["signal_bridge_fields"]) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError("WALKFORWARD_SIGNAL_DIM_MISMATCH")
    columns = [
        "time", "snap", "ctx_cont", "ctx_cat", "label_horizon_bars",
        RESEARCH_LONG_OUTCOME_COLUMN, RESEARCH_SHORT_OUTCOME_COLUMN,
    ]
    table = pq.read_table(str(parquet), columns=columns)
    time_index = pd.DatetimeIndex(pd.to_datetime(table.column("time").to_pandas(), utc=True))
    if not time_index.is_monotonic_increasing or time_index.has_duplicates:
        raise RuntimeError("WALKFORWARD_DATASET_TIME_ORDER_INVALID")
    horizons = np.unique(table.column("label_horizon_bars").to_numpy())
    if len(horizons) != 1:
        raise RuntimeError(f"WALKFORWARD_KNEE_HORIZON_NOT_UNIQUE: {horizons.tolist()}")
    snap = _list_column_to_matrix(table, "snap", MODEL_NATIVE_SIGNAL_DIM, np.float32)
    ctx_cont = _list_column_to_matrix(table, "ctx_cont", MODEL_NATIVE_CTX_CONT_DIM, np.float32)
    ctx_cat = _list_column_to_matrix(table, "ctx_cat", len(MODEL_NATIVE_CTX_CAT_FIELDS), np.int64)
    knee_long = table.column(RESEARCH_LONG_OUTCOME_COLUMN).to_numpy(zero_copy_only=False).astype(np.float64)
    knee_short = table.column(RESEARCH_SHORT_OUTCOME_COLUMN).to_numpy(zero_copy_only=False).astype(np.float64)
    if not np.isfinite(knee_long).all() or not np.isfinite(knee_short).all():
        raise RuntimeError("WALKFORWARD_KNEE_OUTCOME_NONFINITE")
    return Dataset(
        time=time_index, snap=snap, ctx_cont=ctx_cont, ctx_cat=ctx_cat,
        ctx_cont_names=ctx_cont_names, signal_names=tuple(contract["signal_bridge_fields"]), knee_long=knee_long, knee_short=knee_short,
        knee_horizon_bars=int(horizons[0]), manifest_sha256=_sha256_file(manifest_path),
        parquet_path=str(parquet),
    )


@dataclass(frozen=True)
class Tape:
    time: pd.DatetimeIndex
    mid: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    manifest_sha256: str
    root: str


def load_tape(native_m5_root: Path, *, truncate_before: pd.Timestamp) -> Tape:
    manifest = native_m5_root / "MANIFEST.json"
    if not manifest.is_file():
        raise RuntimeError("WALKFORWARD_TAPE_MANIFEST_MISSING")
    truncate_before = pd.Timestamp(truncate_before)
    if truncate_before.tzinfo is None:
        raise RuntimeError("WALKFORWARD_TAPE_BOUNDARY_NOT_UTC_AWARE")
    truncate_before = truncate_before.tz_convert("UTC")
    files = sorted(
        p for p in native_m5_root.glob("year=*/*.parquet")
        if int(p.parent.name.split("=", 1)[1]) <= truncate_before.year
    )
    if not files:
        raise RuntimeError("WALKFORWARD_TAPE_EMPTY")
    frames = [
        pq.read_table(
            str(f), columns=["time", "close", "bid_close", "ask_close"],
            filters=[("time", "<", truncate_before.to_pydatetime())],
        ).to_pandas()
        for f in files
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time", kind="mergesort").reset_index(drop=True)
    if frame.empty or not (frame["time"] < truncate_before).all():
        raise RuntimeError("WALKFORWARD_TAPE_READ_BOUNDARY_INVALID")
    time_index = pd.DatetimeIndex(frame["time"])
    if time_index.has_duplicates:
        raise RuntimeError("WALKFORWARD_TAPE_DUPLICATE_TIME")
    mid = frame["close"].to_numpy(np.float64)
    bid = frame["bid_close"].to_numpy(np.float64)
    ask = frame["ask_close"].to_numpy(np.float64)
    if not (np.isfinite(mid).all() and np.isfinite(bid).all() and np.isfinite(ask).all()):
        raise RuntimeError("WALKFORWARD_TAPE_NONFINITE")
    if np.any(ask < bid) or np.any(bid <= 0):
        raise RuntimeError("WALKFORWARD_TAPE_QUOTE_GEOMETRY_INVALID")
    return Tape(time=time_index, mid=mid, bid=bid, ask=ask, manifest_sha256=_sha256_file(manifest), root=str(native_m5_root))


def tape_positions(dataset_time: pd.DatetimeIndex, tape: Tape) -> np.ndarray:
    positions = tape.time.searchsorted(dataset_time, side="left")
    if np.any(positions >= len(tape.time)) or not np.array_equal(tape.time.values[positions], dataset_time.values):
        raise RuntimeError("WALKFORWARD_DATASET_TIME_NOT_ON_TAPE")
    return positions.astype(np.int64)


def executable_horizon_targets(tape: Tape, positions: np.ndarray, horizon_bars: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executable close-fill research returns per side; NaN where the window leaves the tape."""
    n = len(tape.time)
    future = positions + int(horizon_bars)
    valid = future < n
    long_bps = np.full(len(positions), np.nan)
    short_bps = np.full(len(positions), np.nan)
    f = future[valid]
    p = positions[valid]
    long_bps[valid] = (tape.bid[f] / tape.ask[p] - 1.0) * BPS
    short_bps[valid] = (1.0 - tape.ask[f] / tape.bid[p]) * BPS
    return long_bps, short_bps, valid


def _decision_cutoff_ns(dataset_time: pd.DatetimeIndex, timeframe: str) -> np.ndarray:
    """Cutoff = decision-bar start + 5 min - TF duration (multi_tf_last_closed_label)."""
    cutoff = dataset_time + M5_BAR - MULTI_TF_SHIFT[timeframe]
    return cutoff.asi8


def load_mtf_last_closed(
    cache_dir: Path, dataset_time: pd.DatetimeIndex, ctx_cont: np.ndarray, ctx_cont_names: tuple[str, ...]
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = tuple(manifest["feature_names"])
    shift_contract = manifest["shift_contract"]
    blocks: list[np.ndarray] = []
    names: list[str] = []
    alias_report: dict[str, Any] = {}
    name_to_ctx = {name: index for index, name in enumerate(ctx_cont_names)}
    for tf in ENTRY_MTF_CONTEXT_TIMEFRAMES:
        if pd.Timedelta(shift_contract[tf]) != MULTI_TF_SHIFT[tf]:
            raise RuntimeError(f"WALKFORWARD_MTF_SHIFT_CONTRACT_MISMATCH: {tf}")
        ts = np.load(cache_dir / f"{tf}_ts.npy", mmap_mode="r")
        feats = np.load(cache_dir / f"{tf}_feats.npy", mmap_mode="r")
        if feats.shape[1] != len(feature_names) or feats.shape[0] != len(ts):
            raise RuntimeError(f"WALKFORWARD_MTF_CACHE_SHAPE_INVALID: {tf}")
        cutoff = _decision_cutoff_ns(dataset_time, tf)
        index = np.searchsorted(np.asarray(ts), cutoff, side="right") - 1
        if np.any(index < 0):
            raise RuntimeError(f"WALKFORWARD_MTF_NO_CLOSED_BAR: {tf}")
        block = np.asarray(feats[index], dtype=np.float32)
        if not np.isfinite(block).all():
            raise RuntimeError(f"WALKFORWARD_MTF_NONFINITE: {tf}")
        # Prove the join against the owner's exact scalar aliases (rule 2h).
        for ctx_name, lane_name in MODEL_NATIVE_MTF_SCALAR_PER_BAR_EXACT_ALIASES_V4[tf]:
            if ctx_name not in name_to_ctx or lane_name not in feature_names:
                raise RuntimeError(f"WALKFORWARD_MTF_ALIAS_MISSING: {tf} {ctx_name}/{lane_name}")
            lane_values = block[:, feature_names.index(lane_name)]
            ctx_values = ctx_cont[:, name_to_ctx[ctx_name]]
            mismatches = int(np.count_nonzero(lane_values != ctx_values))
            alias_report[f"{tf}:{ctx_name}=={lane_name}"] = {"rows": int(len(ctx_values)), "mismatch_rows": mismatches}
            if mismatches:
                raise RuntimeError(
                    f"WALKFORWARD_MTF_JOIN_NOT_EXACT: {tf} {ctx_name} vs {lane_name} mismatches={mismatches}"
                )
        blocks.append(block)
        names.extend(f"{tf}:{name}" for name in feature_names)
    matrix = np.concatenate(blocks, axis=1)
    return matrix, tuple(names), {"aliases": alias_report, "cache_manifest_sha256": _sha256_file(manifest_path)}


def load_pattern_primitives(parquet_path: Path, dataset_time: pd.DatetimeIndex) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    """Align the research pattern primitives (``research_entry_pattern_primitives_v1``) to the decision rows by exact time."""
    frame = pd.read_parquet(parquet_path)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    index = pd.DatetimeIndex(frame["time"])
    positions = index.searchsorted(dataset_time, side="left")
    if np.any(positions >= len(index)) or not np.array_equal(index.values[positions], dataset_time.values):
        raise RuntimeError("WALKFORWARD_PATTERN_ROWS_MISSING_FOR_DECISION_TIMES")
    columns = tuple(c for c in frame.columns if c != "time")
    matrix = frame[list(columns)].to_numpy(np.float32)[positions]
    if not np.isfinite(matrix).all():
        raise RuntimeError("WALKFORWARD_PATTERN_NONFINITE")
    manifest = parquet_path.parent / "manifest.json"
    return matrix, tuple(f"pattern:{c}" for c in columns), {"parquet": str(parquet_path), "sha256": _sha256_file(parquet_path), "manifest_sha256": _sha256_file(manifest) if manifest.is_file() else None, "columns": len(columns)}


def load_cross_asset_block(
    daily_parquet: Path | None, h1_parquet: Path | None, dataset_time: pd.DatetimeIndex
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, dict[str, Any]]:
    """Cross-asset research columns aligned to the decision rows, with a per-row validity mask.

    Invalid rows (before warmup, after a series ends, stale H1 bar) are zero-filled and must be excluded through
    the mask by the caller; they never enter a fit or a holdout.
    """
    if daily_parquet is None and h1_parquet is None:
        raise RuntimeError("WALKFORWARD_CROSS_ARM_REQUIRES_INPUT")
    blocks: list[np.ndarray] = []
    names: list[str] = []
    valid = np.ones(len(dataset_time), dtype=bool)
    report: dict[str, Any] = {}
    if daily_parquet is not None:
        daily = pd.read_parquet(daily_parquet)
        index = pd.DatetimeIndex(daily.index)
        if index.tz is None:
            index = index.tz_localize("UTC")
        if not index.is_monotonic_increasing or index.has_duplicates:
            raise RuntimeError("WALKFORWARD_CROSS_DAILY_INDEX_INVALID")
        feats: dict[str, pd.Series] = {}
        for inst in CROSS_DAILY_INSTRUMENTS:
            column = f"{inst}_lvl"
            if column not in daily.columns:
                raise RuntimeError(f"WALKFORWARD_CROSS_DAILY_COLUMN_MISSING: {column}")
            level = pd.Series(daily[column].to_numpy(np.float64), index=index)
            if inst in CROSS_DAILY_LEVEL_INSTRUMENTS:
                feats[f"cross:daily:{inst}_lvl"] = level
            for days in CROSS_DAILY_CHANGE_DAYS:
                feats[f"cross:daily:{inst}_chg{days}d"] = level - level.shift(days)
            rolling = level.rolling(CROSS_DAILY_Z_WINDOW_DAYS, min_periods=CROSS_DAILY_Z_MIN_DAYS)
            feats[f"cross:daily:{inst}_z{CROSS_DAILY_Z_WINDOW_DAYS}d"] = (level - rolling.mean()) / rolling.std()
        daily_matrix = np.column_stack([v.to_numpy(np.float64) for v in feats.values()])
        target_day = dataset_time.floor("D") - CROSS_DAILY_LAG
        pos = index.searchsorted(target_day, side="left")
        pos_clipped = np.minimum(pos, len(index) - 1)
        daily_valid = (pos < len(index)) & (index.values[pos_clipped] == target_day.values)
        rows = daily_matrix[pos_clipped]
        daily_valid &= np.isfinite(rows).all(axis=1)
        rows[~daily_valid] = 0.0
        blocks.append(rows.astype(np.float32))
        names.extend(feats.keys())
        valid &= daily_valid
        report["daily"] = {
            "parquet": str(daily_parquet), "sha256": _sha256_file(daily_parquet), "rows": int(len(index)),
            "span": [index[0].isoformat(), index[-1].isoformat()], "columns": len(feats), "lag": str(CROSS_DAILY_LAG),
            "valid_decision_rows": int(daily_valid.sum()), "invalid_decision_rows": int((~daily_valid).sum()),
            "provenance": "recovered research table without manifest; builder script only (see report notes)",
        }
    if h1_parquet is not None:
        bars = pd.read_parquet(h1_parquet)
        bars["time"] = pd.to_datetime(bars["time"], utc=True)
        bars = bars.sort_values("time", kind="mergesort").reset_index(drop=True)
        labels = pd.DatetimeIndex(bars["time"])
        if labels.has_duplicates:
            raise RuntimeError("WALKFORWARD_CROSS_H1_DUPLICATE_TIME")
        log_close = np.log(bars["close"].to_numpy(np.float64))
        h1_feats: dict[str, np.ndarray] = {}
        for n_bars in CROSS_H1_RETURN_BARS:
            h1_feats[f"cross:usdjpy_h1:ret{n_bars}"] = pd.Series(log_close).diff(n_bars).to_numpy()
        h1_feats[f"cross:usdjpy_h1:rv{CROSS_H1_VOL_BARS}"] = pd.Series(log_close).diff().rolling(CROSS_H1_VOL_BARS).std().to_numpy()
        h1_feats["cross:usdjpy_h1:spread_rel"] = ((bars["ask_close"] - bars["bid_close"]) / bars["close"]).to_numpy(np.float64)
        h1_matrix = np.column_stack(list(h1_feats.values()))
        cutoff = dataset_time + M5_BAR - MULTI_TF_SHIFT["H1"]
        pos = labels.searchsorted(cutoff, side="right") - 1
        pos_clipped = np.maximum(pos, 0)
        staleness = cutoff - labels[pos_clipped]
        h1_valid = (pos >= 0) & (staleness <= CROSS_H1_MAX_STALENESS)
        rows = h1_matrix[pos_clipped]
        h1_valid &= np.isfinite(rows).all(axis=1)
        rows[~h1_valid] = 0.0
        blocks.append(rows.astype(np.float32))
        names.extend(h1_feats.keys())
        valid &= h1_valid
        report["usdjpy_h1"] = {
            "parquet": str(h1_parquet), "sha256": _sha256_file(h1_parquet), "rows": int(len(labels)),
            "span": [labels[0].isoformat(), labels[-1].isoformat()], "columns": len(h1_feats),
            "cutoff_rule": "decision_time + M5 - H1 (owner MULTI_TF_SHIFT)", "max_staleness": str(CROSS_H1_MAX_STALENESS),
            "valid_decision_rows": int(h1_valid.sum()), "invalid_decision_rows": int((~h1_valid).sum()),
            "provenance": "recovered research bars without manifest (OANDA REST fetch, 2026-06-09 spike)",
        }
    report["valid_decision_rows"] = int(valid.sum())
    report["invalid_decision_rows"] = int((~valid).sum())
    return np.concatenate(blocks, axis=1), tuple(names), valid, report


def one_hot_ctx_cat(ctx_cat: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    columns: list[np.ndarray] = []
    names: list[str] = []
    for field_index, field in enumerate(MODEL_NATIVE_CTX_CAT_FIELDS):
        domain = MODEL_NATIVE_CTX_CAT_DOMAINS[field]
        values = ctx_cat[:, field_index]
        if not np.isin(values, domain).all():
            raise RuntimeError(f"WALKFORWARD_CTX_CAT_OUT_OF_DOMAIN: {field}")
        for level in domain:
            columns.append((values == level).astype(np.float32))
            names.append(f"{field}=={level}")
    return np.column_stack(columns), tuple(names)


@dataclass(frozen=True)
class Fold:
    index: int
    fit_start: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp


def build_folds(boundaries: list[pd.Timestamp], train_start: pd.Timestamp) -> list[Fold]:
    if len(boundaries) < 2 or any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise RuntimeError("WALKFORWARD_FOLD_BOUNDARIES_INVALID")
    if boundaries[0] <= train_start:
        raise RuntimeError("WALKFORWARD_FIRST_BOUNDARY_NOT_AFTER_TRAIN_START")
    return [
        Fold(index=k, fit_start=train_start, holdout_start=boundaries[k], holdout_end=boundaries[k + 1])
        for k in range(len(boundaries) - 1)
    ]


def fold_masks(
    dataset_time: pd.DatetimeIndex, positions: np.ndarray, fold: Fold, *, purge_bars: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit rows whose outcome windows end strictly before the first holdout row; holdout rows contained in the fold."""
    holdout = np.asarray((dataset_time >= fold.holdout_start) & (dataset_time < fold.holdout_end), dtype=bool)
    if not holdout.any():
        raise RuntimeError(f"WALKFORWARD_FOLD_EMPTY_HOLDOUT: {fold.index}")
    first_holdout_position = int(positions[holdout].min())
    fit = np.asarray(dataset_time >= fold.fit_start, dtype=bool) & np.asarray(dataset_time < fold.holdout_start, dtype=bool)
    fit &= (positions + int(purge_bars)) < first_holdout_position
    if fit.sum() < 1000:
        raise RuntimeError(f"WALKFORWARD_FOLD_FIT_TOO_SMALL: {fold.index}")
    return fit, holdout


def final_holdout_fit_mask(train_positions: np.ndarray, *, first_holdout_position: int, purge_bars: int) -> np.ndarray:
    """Fit rows for the confirmation stage: every TRAIN row whose outcome window ends before the first VAL row."""
    fit = (train_positions + int(purge_bars)) < int(first_holdout_position)
    if fit.sum() < 1000:
        raise RuntimeError("WALKFORWARD_FINAL_HOLDOUT_FIT_TOO_SMALL")
    return fit


def _inner_split(
    n_fit: int, inner_fraction: float, *, fit_positions: np.ndarray, purge_bars: int
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(fit_positions, dtype=np.int64)
    if positions.shape != (n_fit,) or np.any(np.diff(positions) <= 0) or purge_bars < 0:
        raise RuntimeError("WALKFORWARD_INNER_POSITIONS_INVALID")
    if not 0 < inner_fraction < 1:
        raise RuntimeError("WALKFORWARD_INNER_FRACTION_INVALID")
    cut = int(math.floor(n_fit * (1.0 - inner_fraction)))
    if cut < 100 or n_fit - cut < 100:
        raise RuntimeError("WALKFORWARD_INNER_SPLIT_TOO_SMALL")
    inner_fit = np.zeros(n_fit, dtype=bool)
    inner_fit[:cut] = True
    inner_val = ~inner_fit
    inner_fit &= positions + int(purge_bars) < positions[cut]
    if int(inner_fit.sum()) < 100:
        raise RuntimeError("WALKFORWARD_INNER_PURGED_FIT_TOO_SMALL")
    return inner_fit, inner_val


GRAM_CHUNK_ROWS = 16384


def _chunked_gram(matrix: np.ndarray, rows: np.ndarray | None = None, *, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Float64 X^T X accumulated over row blocks of a float32 matrix (no full float64 copy)."""
    width = int(matrix.shape[1])
    gram = np.zeros((width, width), dtype=np.float64)
    index = np.arange(matrix.shape[0]) if rows is None else rows
    for start in range(0, len(index), GRAM_CHUNK_ROWS):
        block = matrix[index[start:start + GRAM_CHUNK_ROWS]].astype(np.float64, copy=False)
        block = (block - mean) / scale
        gram += block.T @ block
    return gram


def _chunked_xty(matrix: np.ndarray, y: np.ndarray, rows: np.ndarray | None = None, *, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    index = np.arange(matrix.shape[0]) if rows is None else rows
    out = np.zeros(int(matrix.shape[1]), dtype=np.float64)
    for start in range(0, len(index), GRAM_CHUNK_ROWS):
        sel = index[start:start + GRAM_CHUNK_ROWS]
        block = (matrix[sel].astype(np.float64, copy=False) - mean) / scale
        out += block.T @ y[sel]
    return out


def _chunked_matvec(matrix: np.ndarray, beta: np.ndarray, *, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = np.empty(int(matrix.shape[0]), dtype=np.float64)
    for start in range(0, matrix.shape[0], GRAM_CHUNK_ROWS):
        block = (matrix[start:start + GRAM_CHUNK_ROWS].astype(np.float64, copy=False) - mean) / scale
        out[start:start + GRAM_CHUNK_ROWS] = block @ beta
    return out


def _design_scale(matrix: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros(matrix.shape[1], dtype=np.float64)
    for start in range(0, len(rows), GRAM_CHUNK_ROWS):
        mean += matrix[rows[start:start + GRAM_CHUNK_ROWS]].sum(axis=0, dtype=np.float64)
    mean /= len(rows)
    variance = np.zeros_like(mean)
    for start in range(0, len(rows), GRAM_CHUNK_ROWS):
        block = matrix[rows[start:start + GRAM_CHUNK_ROWS]].astype(np.float64) - mean
        variance += np.sum(block * block, axis=0)
    scale = np.sqrt(variance / len(rows))
    scale[scale == 0] = 1.0
    return mean, scale


def standardized_abs_z_mean(
    X_fit: np.ndarray, X_hold: np.ndarray, groups: dict[str, np.ndarray], *, fit_valid: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Per-holdout-row mean |z| under the fit rows' column mean/std: a label-free distance from the fit distribution.

    ``all`` covers every column; each entry of ``groups`` adds the same statistic over that column subset.
    Two exact chunked passes (mean, then centred squares) keep the producer cap for the wide arms.
    """
    n_fit, p = X_fit.shape
    if n_fit == 0 or X_hold.shape[1] != p or (fit_valid is not None and len(fit_valid) != n_fit):
        raise RuntimeError("WALKFORWARD_OOD_DISTANCE_INPUT_INVALID")
    n_used = n_fit if fit_valid is None else int(np.asarray(fit_valid, dtype=bool).sum())
    if n_used == 0:
        raise RuntimeError("WALKFORWARD_OOD_DISTANCE_INPUT_INVALID")

    def _chunks():
        for start in range(0, n_fit, GRAM_CHUNK_ROWS):
            block = X_fit[start : start + GRAM_CHUNK_ROWS]
            yield block if fit_valid is None else block[np.asarray(fit_valid[start : start + GRAM_CHUNK_ROWS], dtype=bool)]

    total = np.zeros(p, dtype=np.float64)
    for block in _chunks():
        total += block.sum(axis=0, dtype=np.float64)
    mean = total / n_used
    squares = np.zeros(p, dtype=np.float64)
    for block in _chunks():
        centred = block.astype(np.float64) - mean
        squares += np.einsum("ij,ij->j", centred, centred)
    std = np.sqrt(squares / n_used)
    std[std == 0] = 1.0
    out = {name: np.empty(len(X_hold), dtype=np.float32) for name in ("all", *groups)}
    for start in range(0, len(X_hold), GRAM_CHUNK_ROWS):
        z = np.abs((X_hold[start : start + GRAM_CHUNK_ROWS].astype(np.float64) - mean) / std)
        out["all"][start : start + len(z)] = z.mean(axis=1)
        for name, cols in groups.items():
            out[name][start : start + len(z)] = z[:, cols].mean(axis=1)
    return out


class RidgeGram:
    """Standardized design + Gram matrices for one (feature arm, fold), shared by every target/side/alpha.

    Alpha selection uses only purged inner-training rows for X/y centering
    and X scaling. The selected alpha is refitted with full-fold statistics.
    All float64 work is chunked over row blocks so the producer cap holds for
    the 1,076-wide MTF arm (the un-chunked version was cgroup-killed at 10 GiB
    on 2026-09-23).
    """

    def __init__(self, X_fit: np.ndarray, X_pred: np.ndarray, *, inner_fraction: float, fit_positions: np.ndarray, purge_bars: int) -> None:
        self.X_fit = X_fit
        self.X_pred = X_pred
        self.inner_fit, self.inner_val = _inner_split(
            len(X_fit), inner_fraction, fit_positions=fit_positions, purge_bars=purge_bars
        )
        self.inner_fit_rows = np.flatnonzero(self.inner_fit)
        self.inner_val_rows = np.flatnonzero(self.inner_val)
        self.mean, self.scale = _design_scale(X_fit, np.arange(len(X_fit)))
        self.inner_mean, self.inner_scale = _design_scale(X_fit, self.inner_fit_rows)
        self.gram_full = _chunked_gram(X_fit, mean=self.mean, scale=self.scale)
        self.gram_inner = _chunked_gram(X_fit, self.inner_fit_rows, mean=self.inner_mean, scale=self.inner_scale)
        self.n_features = int(X_fit.shape[1])
        self.mask_signature: tuple[bytes, bytes] | None = None

    def fit_predict(self, y_fit: np.ndarray, *, keep: np.ndarray | None = None, alpha: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Ridge on all columns or on the ``keep`` subset (sub-Gram: an ablation costs one solve, not one Gram).

        ``alpha`` fixed skips the inner grid search (used by the cardinality-matched null draws, which reuse the
        full fit's selected alpha).  Dropped columns get a zero coefficient, so predictions need no column copies.
        """
        y = np.asarray(y_fit, dtype=np.float64)
        cols = np.arange(self.n_features) if keep is None else np.asarray(keep, dtype=np.int64)
        eye = np.eye(len(cols))
        gram_full = self.gram_full[np.ix_(cols, cols)]
        full_mean = float(np.mean(y))
        xty_full = _chunked_xty(self.X_fit, y - full_mean, mean=self.mean, scale=self.scale)[cols]
        if alpha is None:
            gram_inner = self.gram_inner[np.ix_(cols, cols)]
            inner_mean = float(np.mean(y[self.inner_fit_rows]))
            xty_inner = _chunked_xty(self.X_fit, y - inner_mean, self.inner_fit_rows, mean=self.inner_mean, scale=self.inner_scale)[cols]
            y_val = y[self.inner_val_rows]
            X_val = self.X_fit[self.inner_val_rows]
            best_alpha, best_mse = None, math.inf
            for candidate in RIDGE_ALPHA_GRID:
                beta_val = np.zeros(self.n_features)
                beta_val[cols] = np.linalg.solve(gram_inner + candidate * eye, xty_inner)
                pred_val = _chunked_matvec(X_val, beta_val, mean=self.inner_mean, scale=self.inner_scale) + inner_mean
                mse = float(np.mean((pred_val - y_val) ** 2))
                if mse < best_mse:
                    best_alpha, best_mse = candidate, mse
        else:
            best_alpha, best_mse = float(alpha), None
        beta = np.zeros(self.n_features)
        beta[cols] = np.linalg.solve(gram_full + best_alpha * eye, xty_full)
        pred = _chunked_matvec(self.X_pred, beta, mean=self.mean, scale=self.scale) + full_mean
        return pred, {"alpha": best_alpha, "inner_val_mse": best_mse, "columns": int(len(cols)),
                      "fit_rows": len(y), "inner_fit_rows": len(self.inner_fit_rows),
                      "inner_val_rows": len(self.inner_val_rows),
                      "inner_purged_rows": int((~(self.inner_fit | self.inner_val)).sum())}


def fit_hgb(
    X_fit: np.ndarray, y_fit: np.ndarray, X_pred: np.ndarray, *, inner_fraction: float, max_iter: int, seed: int,
    fit_positions: np.ndarray, purge_bars: int,
    learning_rate: float = HGB_LIBRARY_DEFAULT_LEARNING_RATE, min_samples_leaf: int = HGB_LIBRARY_DEFAULT_MIN_SAMPLES_LEAF,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    inner_fit, inner_val = _inner_split(len(y_fit), inner_fraction, fit_positions=fit_positions, purge_bars=purge_bars)
    model = HistGradientBoostingRegressor(
        max_iter=int(max_iter), early_stopping=False, random_state=int(seed),
        learning_rate=float(learning_rate), min_samples_leaf=int(min_samples_leaf),
    )
    model.fit(X_fit[inner_fit], y_fit[inner_fit])
    best_iter, best_mse = None, math.inf
    for iteration, staged in enumerate(model.staged_predict(X_fit[inner_val]), start=1):
        mse = float(np.mean((staged - y_fit[inner_val]) ** 2))
        if mse < best_mse:
            best_iter, best_mse = iteration, mse
    if best_iter is None:
        raise RuntimeError("WALKFORWARD_HGB_STAGED_PREDICT_FAILED")
    model = HistGradientBoostingRegressor(
        max_iter=int(best_iter), early_stopping=False, random_state=int(seed),
        learning_rate=float(learning_rate), min_samples_leaf=int(min_samples_leaf),
    )
    model.fit(X_fit, y_fit)
    pred = model.predict(X_pred)
    return np.asarray(pred, dtype=np.float64), {
        "fit_rows": len(y_fit), "inner_fit_rows": int(inner_fit.sum()),
        "inner_val_rows": int(inner_val.sum()), "inner_purged_rows": int((~(inner_fit | inner_val)).sum()),
        "best_iter": best_iter, "inner_val_mse": best_mse, "max_iter": int(max_iter),
        "learning_rate": float(learning_rate), "min_samples_leaf": int(min_samples_leaf),
    }


def decisions(long_pred: np.ndarray, short_pred: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contrast = long_pred - short_pred
    if rule == "argmax_flat":
        stacked = np.column_stack([long_pred, short_pred, np.zeros_like(long_pred)])
        side = np.argmax(stacked, axis=1).astype(np.int64)
        score = stacked[np.arange(len(side)), side]
        # exact ties between LONG and SHORT fail closed in the owner; here they abstain.
        tie = (long_pred == short_pred) & (side != MODEL_DIRECTION_FLAT_INDEX)
        side[tie] = MODEL_DIRECTION_FLAT_INDEX
        score[tie] = 0.0
        return side, score, contrast
    if rule == "contrast_always_trade":
        side = np.where(contrast > 0, MODEL_DIRECTION_LONG_INDEX, MODEL_DIRECTION_SHORT_INDEX).astype(np.int64)
        side[contrast == 0] = MODEL_DIRECTION_FLAT_INDEX
        return side, np.abs(contrast), contrast
    raise RuntimeError(f"WALKFORWARD_DECISION_RULE_INVALID: {rule}")


def selected_mean_pnl_by_coverage(
    side: np.ndarray, score: np.ndarray, long_all: np.ndarray, short_all: np.ndarray, top_fracs: tuple[float, ...]
) -> dict[float, float | None]:
    """Mean traded PnL per coverage with the evaluator's selection: score descending, ties by time (row order), FLAT dropped, None below the evaluator's minimum trade count."""
    n = len(side)
    ranked = np.lexsort((np.arange(n), -np.asarray(score, dtype=np.float64)))
    out: dict[float, float | None] = {}
    for top_frac in top_fracs:
        n_budget = max(1, int(math.ceil(n * float(top_frac))))
        sel = ranked[:n_budget]
        traded = side[sel] != MODEL_DIRECTION_FLAT_INDEX
        sel = sel[traded]
        if len(sel) < MIN_PREREGISTERED_TRADE_ROWS:  # the evaluator reports no mean below its preregistered minimum
            out[float(top_frac)] = None
            continue
        pnl = np.where(side[sel] == MODEL_DIRECTION_LONG_INDEX, long_all[sel], short_all[sel])
        out[float(top_frac)] = float(np.mean(pnl))
    return out


def feature_column_groups(column_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Owner-mapped ablation groups over an arm's ordered columns.

    Snapshot/context columns map to their specialist family through the
    routing owner ``classify_entry_specialist_feature``; MTF columns
    (``TF:field``) map to a lane group and, through
    ``MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4``, to an MTF family group; pattern
    columns (``pattern:TF:field``) form one group per timeframe; one-hot
    context categories form ``ctx_cat``.
    """
    mtf_family_by_field: dict[str, str] = {}
    for family, fields in MULTI_TF_SPECIALIST_FEATURE_GROUPS_V4.items():
        for field in fields:
            mtf_family_by_field[str(field)] = str(family)
    groups: dict[str, list[int]] = {}

    def put(group: str, index: int) -> None:
        groups.setdefault(group, []).append(index)

    for index, name in enumerate(column_names):
        if name.startswith("pattern:"):
            put(f"patterns:{name.split(':')[1]}", index)
        elif name.startswith("cross:"):
            put(f"cross:{name.split(':')[1]}", index)
        elif ":" in name and name.split(":")[0] in MULTI_TF_SHIFT:
            tf, field = name.split(":", 1)
            put(f"mtf_lane:{tf}", index)
            put(f"mtf_family:{mtf_family_by_field.get(field, 'unmapped')}", index)
        elif name.startswith("session_id=="):
            put("ctx_cat", index)
        else:
            put(f"family:{classify_entry_specialist_feature(name)}", index)
    # union groups: "everything the pattern module adds" and "every MTF lane" (drop = the plain baseline arm on the same rows)
    for prefix, union_name in (("patterns:", "patterns:all"), ("mtf_lane:", "mtf_lane:all"), ("cross:", "cross:all")):
        members = sorted({i for g, idx in groups.items() if g.startswith(prefix) for i in idx})
        if members:
            groups[union_name] = members
    return {group: np.asarray(sorted(indices), dtype=np.int64) for group, indices in groups.items()}


def month_drift_sign(tape: Tape) -> pd.Series:
    """Realized sign of each calendar month's mid close-to-close drift on the tape (descriptive slice only)."""
    frame = pd.DataFrame({"time": tape.time.tz_convert(None), "mid": tape.mid})
    month = frame["time"].dt.to_period("M")
    first = frame.groupby(month)["mid"].first()
    last = frame.groupby(month)["mid"].last()
    return np.sign(last - first)


def evaluate_frame(
    frame: pd.DataFrame, *, fold: Fold, month_sign: pd.Series, meta: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = build_metric_rows(frame, top_fracs=list(EVALUATION_COVERAGES))
    ordered = frame.sort_values("time", kind="mergesort").reset_index(drop=True)
    ranked = ordered.sort_values(["selection_score", "time"], ascending=[False, True], kind="mergesort")
    long_all = ordered[RESEARCH_LONG_OUTCOME_COLUMN].to_numpy(np.float64)
    short_all = ordered[RESEARCH_SHORT_OUTCOME_COLUMN].to_numpy(np.float64)
    months = ordered["time"].dt.tz_convert(None).dt.to_period("M")
    up_month = months.map(lambda m: month_sign.get(m, 0.0) > 0).to_numpy()
    down_month = months.map(lambda m: month_sign.get(m, 0.0) < 0).to_numpy()
    out: list[dict[str, Any]] = []
    for row in rows:
        n_budget = max(1, int(math.ceil(len(ordered) * float(row["top_frac"]))))
        selected_index = ranked.head(n_budget).index.to_numpy()
        side = ordered.loc[selected_index, "pred_direction"].to_numpy(np.int64)
        traded = side != MODEL_DIRECTION_FLAT_INDEX
        sel = selected_index[traded]
        s = side[traded]
        pnl = np.where(s == MODEL_DIRECTION_LONG_INDEX, long_all[sel], short_all[sel])
        better_side_realized = np.where(long_all[sel] >= short_all[sel], MODEL_DIRECTION_LONG_INDEX, MODEL_DIRECTION_SHORT_INDEX)
        extra: dict[str, Any] = {
            "fold": fold.index,
            "fit_start": fold.fit_start.isoformat(),
            "holdout_start": fold.holdout_start.isoformat(),
            "holdout_end": fold.holdout_end.isoformat(),
            "holdout_rows": int(len(ordered)),
            **meta,
        }
        if len(sel):
            long_mask = s == MODEL_DIRECTION_LONG_INDEX
            best_constant = float(max(np.mean(long_all[sel]), np.mean(short_all[sel])))
            extra.update(
                {
                    "best_constant_side_mean_pnl_bps": best_constant,
                    "excess_over_best_constant_bps": float(np.mean(pnl)) - best_constant,
                    "beats_best_constant": bool(float(np.mean(pnl)) > best_constant),
                    "hit_rate_better_side": float(np.mean(s == better_side_realized)),
                    "p_long_chosen": float(np.mean(long_mask)),
                    "long_rows": int(long_mask.sum()),
                    "short_rows": int((~long_mask).sum()),
                    "long_mean_pnl_bps": float(np.mean(pnl[long_mask])) if long_mask.any() else None,
                    "short_mean_pnl_bps": float(np.mean(pnl[~long_mask])) if (~long_mask).any() else None,
                    "up_month_rows": int(up_month[sel].sum()),
                    "up_month_mean_pnl_bps": float(np.mean(pnl[up_month[sel]])) if up_month[sel].any() else None,
                    "down_month_rows": int(down_month[sel].sum()),
                    "down_month_mean_pnl_bps": float(np.mean(pnl[down_month[sel]])) if down_month[sel].any() else None,
                    "flat_share_of_holdout": float(np.mean(ordered["pred_direction"].to_numpy() == MODEL_DIRECTION_FLAT_INDEX)),
                }
            )
        mean_pnl = row.get("mean_pnl_bps")
        extra["strict_pass"] = bool(row.get("primary_pass") and mean_pnl is not None and mean_pnl > 0.0)
        out.append({**row, **extra})
    return out


def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[2]), check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"unserializable: {type(value)!r}")


def _array_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    for start in range(0, len(array), GRAM_CHUNK_ROWS):
        digest.update(np.ascontiguousarray(array[start:start + GRAM_CHUNK_ROWS]).tobytes())
    return digest.hexdigest()


def _check_run_binding(out_dir: Path, spec: dict[str, Any], inputs: dict[str, Any] | None = None) -> str | None:
    path = out_dir / "RUN_BINDING.json"
    binding = {"spec": spec, "inputs": inputs}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("spec") != spec or (inputs is not None and previous.get("inputs") != inputs):
            raise RuntimeError("WALKFORWARD_RESUME_BINDING_MISMATCH")
    elif any((out_dir / "per_config").glob("*.json")):
        raise RuntimeError("WALKFORWARD_RESUME_UNBOUND_CACHE")
    elif inputs is not None:
        path.write_text(json.dumps(binding, sort_keys=True, indent=2), encoding="utf-8")
    if inputs is None:
        return None
    return hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = _time.monotonic()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.resume:
        raise RuntimeError("WALKFORWARD_OUT_DIR_EXISTS")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_config_dir = out_dir / "per_config"
    per_config_dir.mkdir(exist_ok=True)
    run_spec = {
        "config": {key: value for key, value in vars(args).items() if key not in ("resume", "out_dir")},
        "instrument_source_sha256": _sha256_file(Path(__file__)),
        "evaluator_source_sha256": _sha256_file(Path(inspect.getfile(build_metric_rows))),
        "git_head": _git_head(),
    }
    _check_run_binding(out_dir, run_spec)

    dataset_dir = Path(args.dataset_dir)
    train_manifest = json.loads((dataset_dir / "entry_dataset__ENTRY_FITTED_Q_train.manifest.json").read_text(encoding="utf-8"))
    # Snapshot fields also include fitted registry/volatility outputs. Check before
    # materializing price/features, even when no extra MTF arm was requested.
    feature_fits = feature_fit_lineage(train_manifest.get("extra", {}).get("multi_tf_cache_binding", {}))
    if any(arm in args.feature_arms for arm in ("snapshot_mtf", "snapshot_mtf_patterns", "snapshot_mtf_cross")):
        if not args.multi_tf_cache_dir:
            raise RuntimeError("WALKFORWARD_MTF_ARM_REQUIRES_CACHE_DIR")
        cache_manifest = json.loads((Path(args.multi_tf_cache_dir) / "manifest.json").read_text(encoding="utf-8"))
        feature_fits.extend(feature_fit_lineage(cache_manifest))
    require_feature_fit_before(feature_fits, pd.Timestamp(args.fold_boundaries[0]), context="first_outer_fold")
    dataset = load_dataset(dataset_dir)
    train_start = pd.Timestamp(train_manifest["splits"]["train"]["start"])
    train_end_exclusive = pd.Timestamp(train_manifest["splits"]["val"]["start"])
    val_end_exclusive = pd.Timestamp(train_manifest["splits"]["val"]["end"])
    final_holdout = str(args.final_holdout)
    if final_holdout not in ("none", "val"):
        raise RuntimeError("WALKFORWARD_FINAL_HOLDOUT_INVALID")
    # Push the allowed split boundary into the reader before rows are materialized.
    tape = load_tape(Path(args.native_m5_root), truncate_before=(val_end_exclusive if final_holdout == "val" else train_end_exclusive))
    positions = tape_positions(dataset.time, tape)
    val_dataset: Dataset | None = None
    val_positions: np.ndarray | None = None
    if final_holdout == "val":
        val_dataset = load_dataset(dataset_dir, split="val")
        if val_dataset.knee_horizon_bars != dataset.knee_horizon_bars:
            raise RuntimeError("WALKFORWARD_VAL_KNEE_HORIZON_MISMATCH")
        val_positions = tape_positions(val_dataset.time, tape)
        if int(val_positions.min()) <= int(positions.max()):
            raise RuntimeError("WALKFORWARD_VAL_NOT_AFTER_TRAIN")
    boundaries = [pd.Timestamp(v) for v in args.fold_boundaries]
    if boundaries[-1] > train_end_exclusive:
        raise RuntimeError("WALKFORWARD_FOLD_BOUNDARY_BEYOND_TRAIN_END")
    folds = build_folds(boundaries, train_start)
    horizons = [int(h) for h in args.horizons]
    if any(h <= 0 for h in horizons):
        raise RuntimeError("WALKFORWARD_HORIZON_INVALID")
    purge_bars = max([dataset.knee_horizon_bars, *horizons]) + 1

    ctx_cat_matrix, ctx_cat_names = one_hot_ctx_cat(dataset.ctx_cat)
    snapshot = np.concatenate([dataset.snap, dataset.ctx_cont, ctx_cat_matrix], axis=1)
    snapshot_names = tuple(dataset.signal_names) + tuple(f"ctx_cont.{n}" for n in dataset.ctx_cont_names) + tuple(ctx_cat_names)
    arms: dict[str, np.ndarray] = {"snapshot": snapshot}
    arm_names: dict[str, tuple[str, ...]] = {"snapshot": snapshot_names}
    mtf_report: dict[str, Any] = {}
    needs_mtf = any(arm in args.feature_arms for arm in ("snapshot_mtf", "snapshot_mtf_patterns", "snapshot_mtf_cross"))
    needs_cross = any(arm in args.feature_arms for arm in ("snapshot_cross", "snapshot_mtf_cross"))
    needs_patterns = any(arm in args.feature_arms for arm in ("snapshot_patterns", "snapshot_mtf_patterns"))
    mtf_matrix = mtf_names = None
    if needs_mtf:
        if not args.multi_tf_cache_dir:
            raise RuntimeError("WALKFORWARD_MTF_ARM_REQUIRES_CACHE_DIR")
        mtf_matrix, mtf_names, mtf_report = load_mtf_last_closed(Path(args.multi_tf_cache_dir), dataset.time, dataset.ctx_cont, dataset.ctx_cont_names)
    pattern_matrix = pattern_names = None
    pattern_report: dict[str, Any] = {}
    if needs_patterns:
        if not args.pattern_primitives_parquet:
            raise RuntimeError("WALKFORWARD_PATTERN_ARM_REQUIRES_PARQUET")
        pattern_matrix, pattern_names, pattern_report = load_pattern_primitives(Path(args.pattern_primitives_parquet), dataset.time)
    cross_matrix = cross_names = None
    cross_report: dict[str, Any] = {}
    arm_valid: dict[str, np.ndarray] = {}
    val_arm_valid: dict[str, np.ndarray] = {}
    if needs_cross:
        daily_path = Path(args.cross_asset_daily_parquet) if args.cross_asset_daily_parquet else None
        h1_path = Path(args.cross_asset_h1_parquet) if args.cross_asset_h1_parquet else None
        cross_matrix, cross_names, cross_valid, cross_report = load_cross_asset_block(daily_path, h1_path, dataset.time)
    if "snapshot_mtf" in args.feature_arms:
        arms["snapshot_mtf"] = np.concatenate([snapshot, mtf_matrix], axis=1)
        arm_names["snapshot_mtf"] = snapshot_names + tuple(mtf_names)
    if "snapshot_cross" in args.feature_arms:
        arms["snapshot_cross"] = np.concatenate([snapshot, cross_matrix], axis=1)
        arm_names["snapshot_cross"] = snapshot_names + tuple(cross_names)
        arm_valid["snapshot_cross"] = cross_valid
    if "snapshot_mtf_cross" in args.feature_arms:
        arms["snapshot_mtf_cross"] = np.concatenate([snapshot, mtf_matrix, cross_matrix], axis=1)
        arm_names["snapshot_mtf_cross"] = snapshot_names + tuple(mtf_names) + tuple(cross_names)
        arm_valid["snapshot_mtf_cross"] = cross_valid
    if "snapshot_patterns" in args.feature_arms:
        arms["snapshot_patterns"] = np.concatenate([snapshot, pattern_matrix], axis=1)
        arm_names["snapshot_patterns"] = snapshot_names + tuple(pattern_names)
    if "snapshot_mtf_patterns" in args.feature_arms:
        arms["snapshot_mtf_patterns"] = np.concatenate([snapshot, mtf_matrix, pattern_matrix], axis=1)
        arm_names["snapshot_mtf_patterns"] = snapshot_names + tuple(mtf_names) + tuple(pattern_names)
    del mtf_matrix, pattern_matrix, cross_matrix
    column_groups_by_arm = {arm: feature_column_groups(names) for arm, names in arm_names.items()}
    ablation_groups: dict[str, dict[str, np.ndarray]] = {}
    if args.ablation != "none":
        for arm, groups in column_groups_by_arm.items():
            if args.ablation == "families":
                groups = {g: idx for g, idx in groups.items() if g.startswith("family:") or g.startswith("mtf_family:") or g.startswith("patterns:") or g == "mtf_lane:all"}
            elif args.ablation == "lanes":
                groups = {g: idx for g, idx in groups.items() if g.startswith("mtf_lane:") or g.startswith("patterns:") or g == "ctx_cat"}
            elif args.ablation == "cross":
                groups = {g: idx for g, idx in groups.items() if g.startswith("cross:")}
            ablation_groups[arm] = groups
    atr_index = dataset.ctx_cont_names.index(ATR_SCALE_FIELD)
    atr = dataset.ctx_cont[:, atr_index].astype(np.float64)
    if np.any(atr <= 0):
        raise RuntimeError("WALKFORWARD_ATR_SCALE_NONPOSITIVE")

    targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {
        KNEE_TARGET_NAME: (dataset.knee_long, dataset.knee_short, np.ones(len(dataset.time), dtype=bool), dataset.knee_horizon_bars)
    }
    for h in horizons:
        long_bps, short_bps, valid = executable_horizon_targets(tape, positions, h)
        targets[f"exec_close_h{h}"] = (long_bps, short_bps, valid, h)
    if args.targets:
        unknown = sorted(set(args.targets) - set(targets))
        if unknown:
            raise RuntimeError(f"WALKFORWARD_TARGET_FILTER_UNKNOWN: {unknown}")
        targets = {name: targets[name] for name in targets if name in set(args.targets)}
    val_targets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    val_arms: dict[str, np.ndarray] = {}
    if val_dataset is not None and val_positions is not None:
        val_targets[KNEE_TARGET_NAME] = (val_dataset.knee_long, val_dataset.knee_short, np.ones(len(val_dataset.time), dtype=bool), val_dataset.knee_horizon_bars)
        for h in horizons:
            long_bps, short_bps, valid = executable_horizon_targets(tape, val_positions, h)
            val_targets[f"exec_close_h{h}"] = (long_bps, short_bps, valid, h)
        val_cat, _ = one_hot_ctx_cat(val_dataset.ctx_cat)
        val_snapshot = np.concatenate([val_dataset.snap, val_dataset.ctx_cont, val_cat], axis=1)
        val_arms["snapshot"] = val_snapshot
        val_mtf = val_pat = None
        if needs_mtf:
            val_mtf, _, _ = load_mtf_last_closed(Path(args.multi_tf_cache_dir), val_dataset.time, val_dataset.ctx_cont, val_dataset.ctx_cont_names)
        if needs_patterns:
            val_pat, _, _ = load_pattern_primitives(Path(args.pattern_primitives_parquet), val_dataset.time)
        if "snapshot_mtf" in args.feature_arms:
            val_arms["snapshot_mtf"] = np.concatenate([val_snapshot, val_mtf], axis=1)
        if "snapshot_patterns" in args.feature_arms:
            val_arms["snapshot_patterns"] = np.concatenate([val_snapshot, val_pat], axis=1)
        if "snapshot_mtf_patterns" in args.feature_arms:
            val_arms["snapshot_mtf_patterns"] = np.concatenate([val_snapshot, val_mtf, val_pat], axis=1)
        val_cross = None
        if needs_cross:
            val_cross, _, val_cross_valid, _ = load_cross_asset_block(daily_path, h1_path, val_dataset.time)
            val_arm_valid = {arm: val_cross_valid for arm in ("snapshot_cross", "snapshot_mtf_cross") if arm in args.feature_arms}
        if "snapshot_cross" in args.feature_arms:
            val_arms["snapshot_cross"] = np.concatenate([val_snapshot, val_cross], axis=1)
        if "snapshot_mtf_cross" in args.feature_arms:
            val_arms["snapshot_mtf_cross"] = np.concatenate([val_snapshot, val_mtf, val_cross], axis=1)
        del val_mtf, val_pat, val_cross
    month_sign = month_drift_sign(tape)
    predictions_dir = out_dir / "predictions"
    if args.persist_predictions:
        predictions_dir.mkdir(exist_ok=True)

    # Bind the exact already-loaded arrays used by the run, without hashing sealed tape rows.
    bound_arrays = {"time": dataset.time.asi8, "positions": positions, "atr": atr,
                    "tape_time": tape.time.asi8, "tape_mid": tape.mid, "tape_bid": tape.bid, "tape_ask": tape.ask}
    for label, matrices in (("features", arms), ("val_features", val_arms), ("valid", arm_valid), ("val_valid", val_arm_valid)):
        bound_arrays.update({f"{label}:{name}": values for name, values in matrices.items()})
    for label, outcomes in (("target", targets), ("val_target", val_targets)):
        for name, (long_values, short_values, valid, _) in outcomes.items():
            bound_arrays.update({f"{label}:{name}:long": long_values, f"{label}:{name}:short": short_values,
                                 f"{label}:{name}:valid": valid})
    if val_dataset is not None:
        bound_arrays["val_time"] = val_dataset.time.asi8
        bound_arrays["val_atr"] = val_dataset.ctx_cont[:, atr_index]
    input_binding = {
        "arrays": {name: _array_sha256(values) for name, values in bound_arrays.items()},
        "feature_names": {name: list(names) for name, names in arm_names.items()},
        "feature_fit_lineage": feature_fits,
        "train_manifest_sha256": dataset.manifest_sha256, "tape_manifest_sha256": tape.manifest_sha256,
        "val_manifest_sha256": val_dataset.manifest_sha256 if val_dataset is not None else None,
    }
    run_binding_sha256 = _check_run_binding(out_dir, run_spec, input_binding)

    all_rows: list[dict[str, Any]] = []
    fit_reports: list[dict[str, Any]] = []

    def _config_keys(arm: str, fold: Fold) -> list[tuple[str, str, str, int]]:
        keys: list[tuple[str, str, str, int]] = []
        for target_name in targets:
            for scaling in args.target_scalings:
                for learner in args.learners:
                    seeds = [int(v) for v in args.seeds] if learner == "hgb" else [0]
                    for seed in seeds:
                        keys.append((target_name, scaling, learner, seed))
        return keys

    stages: list[tuple[Fold, str]] = [(fold, "fold") for fold in folds]
    if val_dataset is not None:
        stages.append((Fold(index=len(folds), fit_start=train_start, holdout_start=train_end_exclusive, holdout_end=val_end_exclusive), "final_val"))
    for arm in args.feature_arms:
        X = arms[arm]
        for fold, stage_kind in stages:
            pending = [
                cfg for cfg in _config_keys(arm, fold)
                if not (per_config_dir / f"{cfg[0]}__{arm}__{cfg[1]}__{cfg[2]}__seed{cfg[3]}__fold{fold.index}.json").is_file()
            ]
            for cfg in _config_keys(arm, fold):
                if cfg in pending:
                    continue
                cache = per_config_dir / f"{cfg[0]}__{arm}__{cfg[1]}__{cfg[2]}__seed{cfg[3]}__fold{fold.index}.json"
                payload = json.loads(cache.read_text(encoding="utf-8"))
                if payload.get("run_binding_sha256") != run_binding_sha256:
                    raise RuntimeError(f"WALKFORWARD_RESUME_CACHE_MISMATCH: {cache.name}")
                all_rows.extend(payload["rows"])
                fit_reports.append(payload["fit"])
            if not pending:
                continue
            if stage_kind == "final_val":
                assert val_dataset is not None and val_positions is not None
                base_fit_mask = final_holdout_fit_mask(positions, first_holdout_position=int(val_positions.min()), purge_bars=purge_bars)
                base_holdout_mask = np.ones(len(val_dataset.time), dtype=bool)
                X_hold_all = val_arms[arm]
                hold_time = val_dataset.time
                hold_atr = val_dataset.ctx_cont[:, atr_index].astype(np.float64)
                if np.any(hold_atr <= 0):
                    raise RuntimeError("WALKFORWARD_VAL_ATR_SCALE_NONPOSITIVE")
            else:
                base_fit_mask, base_holdout_mask = fold_masks(dataset.time, positions, fold, purge_bars=purge_bars)
                X_hold_all = X
                hold_time = dataset.time
                hold_atr = atr
            ood_distance: dict[str, np.ndarray] | None = None
            if args.persist_predictions and pending:
                # label-free distance of every holdout row from the stage's fit-period distribution (all rows before
                # the holdout start, a chronological prefix; the horizon purge is irrelevant to column statistics)
                fit_region = np.asarray(dataset.time < fold.holdout_start, dtype=bool)
                n_fit_region = int(fit_region.sum())
                if not (fit_region[:n_fit_region].all() and not fit_region[n_fit_region:].any()):
                    raise RuntimeError("WALKFORWARD_FIT_REGION_NOT_PREFIX")
                distance_groups = {g: idx for g, idx in column_groups_by_arm[arm].items() if g in OOD_DISTANCE_GROUPS}
                fit_valid = arm_valid[arm][:n_fit_region] if arm in arm_valid else None
                ood_distance = standardized_abs_z_mean(X[:n_fit_region], X_hold_all, distance_groups, fit_valid=fit_valid)
            ridge_gram: RidgeGram | None = None
            for target_name, scaling, learner, seed in pending:
                long_y, short_y, valid_y, horizon = targets[target_name]
                if stage_kind == "final_val":
                    hold_long, hold_short, hold_valid, _ = val_targets[target_name]
                else:
                    hold_long, hold_short, hold_valid = long_y, short_y, valid_y
                key = f"{target_name}__{arm}__{scaling}__{learner}__seed{seed}__fold{fold.index}"
                cache = per_config_dir / f"{key}.json"
                fit_mask = base_fit_mask & valid_y
                holdout_mask = base_holdout_mask & hold_valid
                if arm in arm_valid:
                    fit_mask &= arm_valid[arm]
                    holdout_mask &= (val_arm_valid[arm] if stage_kind == "final_val" else arm_valid[arm])
                if stage_kind == "fold":
                    # A fold holdout may never label a row with prices beyond its own end: the next
                    # fold's first TRAIN row, or (for the last fold) the first tape row at/after the
                    # TRAIN split end when the tape extends into VAL for the confirmation stage.
                    if fold.holdout_end < train_end_exclusive:
                        next_first = int(positions[np.asarray(dataset.time >= fold.holdout_end, dtype=bool)].min())
                    else:
                        next_first = int(tape.time.searchsorted(train_end_exclusive, side="left"))
                    holdout_mask &= (positions + int(horizon)) < next_first
                # The inner model-selection population needs its own feature-fit bound.
                _, inner_val = _inner_split(
                    int(fit_mask.sum()), args.inner_fraction,
                    fit_positions=positions[fit_mask], purge_bars=purge_bars,
                )
                require_feature_fit_before(
                    feature_fits, dataset.time[fit_mask][inner_val][0],
                    context=f"{key}:inner_selection",
                )
                y_scale_fit = atr[fit_mask] if scaling == "atr" else np.ones(int(fit_mask.sum()))
                y_scale_hold = hold_atr[holdout_mask] if scaling == "atr" else np.ones(int(holdout_mask.sum()))
                preds: dict[str, np.ndarray] = {}
                fit_info: dict[str, Any] = {}
                for side_name, y_all in (("long", long_y), ("short", short_y)):
                    y_fit = y_all[fit_mask] / y_scale_fit
                    if learner == "ridge":
                        # The Gram is shared across targets whenever the fit/holdout row sets coincide.
                        if ridge_gram is None or ridge_gram.X_fit.shape[0] != int(fit_mask.sum()) or ridge_gram.X_pred.shape[0] != int(holdout_mask.sum()) or ridge_gram.mask_signature != (fit_mask.tobytes(), holdout_mask.tobytes()):
                            ridge_gram = RidgeGram(X[fit_mask], X_hold_all[holdout_mask], inner_fraction=args.inner_fraction,
                                                   fit_positions=positions[fit_mask], purge_bars=purge_bars)
                            ridge_gram.mask_signature = (fit_mask.tobytes(), holdout_mask.tobytes())
                        pred, info = ridge_gram.fit_predict(y_fit)
                    else:
                        pred, info = fit_hgb(
                            X[fit_mask], y_fit, X_hold_all[holdout_mask], inner_fraction=args.inner_fraction, max_iter=args.max_hgb_iter, seed=seed,
                            learning_rate=args.hgb_learning_rate, min_samples_leaf=args.hgb_min_samples_leaf,
                            fit_positions=positions[fit_mask], purge_bars=purge_bars,
                        )
                    preds[side_name] = pred * y_scale_hold
                    fit_info[side_name] = info
                rows_for_key: list[dict[str, Any]] = []
                if args.persist_predictions:
                    persisted = {
                        "time": hold_time[holdout_mask],
                        "pred_long_bps": preds["long"],
                        "pred_short_bps": preds["short"],
                        RESEARCH_LONG_OUTCOME_COLUMN: hold_long[holdout_mask],
                        RESEARCH_SHORT_OUTCOME_COLUMN: hold_short[holdout_mask],
                    }
                    if ood_distance is not None:
                        for group, values in ood_distance.items():
                            persisted[OOD_DISTANCE_COLUMNS[group]] = values[holdout_mask]
                    pd.DataFrame(persisted).to_parquet(predictions_dir / f"{key}.parquet", index=False)
                for rule in args.decision_rules:
                    side, score, contrast = decisions(preds["long"], preds["short"], rule)
                    frame = pd.DataFrame(
                        {
                            "split": f"fold{fold.index}" if stage_kind == "fold" else "final_val",
                            "model": f"{key}__{rule}",
                            "time": hold_time[holdout_mask],
                            "pred_direction": side,
                            "selection_score": score,
                            "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
                            "edge_score": contrast,
                            RESEARCH_LONG_OUTCOME_COLUMN: hold_long[holdout_mask],
                            RESEARCH_SHORT_OUTCOME_COLUMN: hold_short[holdout_mask],
                        }
                    )
                    meta = {
                        "target": target_name, "horizon_bars": int(horizon), "feature_arm": arm,
                        "target_scaling": scaling, "learner": learner, "seed": int(seed), "decision_rule": rule,
                        "fit_rows": int(fit_mask.sum()), "stage": stage_kind,
                        "outcome_definition": (
                            "dataset knee-horizon M1 final PnL (research gross spread-inclusive)"
                            if target_name == KNEE_TARGET_NAME
                            else "native M5 tape executable close-fill return (research convention)"
                        ),
                    }
                    rows_for_key.extend(evaluate_frame(frame, fold=fold, month_sign=month_sign, meta=meta))
                if learner == "ridge" and ablation_groups.get(arm) and ridge_gram is not None:
                    full_by_rule = {r: {(row["top_frac"]): row for row in rows_for_key if row["decision_rule"] == r} for r in args.decision_rules}
                    all_cols = np.arange(X.shape[1])
                    null_draws = int(args.ablation_null_draws)
                    null_rng = np.random.default_rng(int(seed) * 100003 + int(fold.index))
                    hold_long_rows = hold_long[holdout_mask]
                    hold_short_rows = hold_short[holdout_mask]
                    for group_name, drop in ablation_groups[arm].items():
                        keep = np.setdiff1d(all_cols, drop)
                        abl_preds: dict[str, np.ndarray] = {}
                        for side_name, y_all in (("long", long_y), ("short", short_y)):
                            pred, _ = ridge_gram.fit_predict(y_all[fit_mask] / y_scale_fit, keep=keep)
                            abl_preds[side_name] = pred * y_scale_hold
                        group_rows: list[dict[str, Any]] = []
                        for rule in args.decision_rules:
                            side, score, contrast = decisions(abl_preds["long"], abl_preds["short"], rule)
                            frame = pd.DataFrame(
                                {
                                    "split": f"fold{fold.index}" if stage_kind == "fold" else "final_val",
                                    "model": f"{key}__{rule}__ablate_{group_name}",
                                    "time": hold_time[holdout_mask],
                                    "pred_direction": side,
                                    "selection_score": score,
                                    "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
                                    "edge_score": contrast,
                                    RESEARCH_LONG_OUTCOME_COLUMN: hold_long_rows,
                                    RESEARCH_SHORT_OUTCOME_COLUMN: hold_short_rows,
                                }
                            )
                            meta = {
                                "target": target_name, "horizon_bars": int(horizon), "feature_arm": arm,
                                "target_scaling": scaling, "learner": learner, "seed": int(seed), "decision_rule": rule,
                                "fit_rows": int(fit_mask.sum()), "stage": stage_kind, "ablation_group": group_name,
                                "ablated_columns": int(len(drop)), "outcome_definition": "ablation of one owner-mapped group; same rows and nulls as the full fit",
                            }
                            for row in evaluate_frame(frame, fold=fold, month_sign=month_sign, meta=meta):
                                full = full_by_rule[rule].get(row["top_frac"])
                                if full is not None and full.get("mean_pnl_bps") is not None and row.get("mean_pnl_bps") is not None:
                                    row["delta_mean_pnl_bps_vs_full"] = float(row["mean_pnl_bps"] - full["mean_pnl_bps"])
                                group_rows.append(row)
                        if null_draws > 0 and 0 < len(drop) < X.shape[1]:
                            # cardinality-matched null (fidelity review 2026-09-21, F-6): drop the same NUMBER of
                            # columns drawn at random, refit at the full fit's selected alpha, same selection rule
                            null_deltas: dict[str, dict[float, list[float]]] = {r: {float(tf): [] for tf in EVALUATION_COVERAGES} for r in args.decision_rules}
                            for _ in range(null_draws):
                                random_keep = np.setdiff1d(all_cols, null_rng.choice(X.shape[1], size=len(drop), replace=False))
                                null_preds: dict[str, np.ndarray] = {}
                                for side_name, y_all in (("long", long_y), ("short", short_y)):
                                    pred, _ = ridge_gram.fit_predict(y_all[fit_mask] / y_scale_fit, keep=random_keep, alpha=float(fit_info[side_name]["alpha"]))
                                    null_preds[side_name] = pred * y_scale_hold
                                for rule in args.decision_rules:
                                    side, score, _ = decisions(null_preds["long"], null_preds["short"], rule)
                                    means = selected_mean_pnl_by_coverage(side, score, hold_long_rows, hold_short_rows, EVALUATION_COVERAGES)
                                    for top_frac, mean_pnl in means.items():
                                        full = full_by_rule[rule].get(top_frac)
                                        if mean_pnl is not None and full is not None and full.get("mean_pnl_bps") is not None:
                                            null_deltas[rule][top_frac].append(mean_pnl - float(full["mean_pnl_bps"]))
                            for row in group_rows:
                                deltas = null_deltas[row["decision_rule"]].get(float(row["top_frac"]), [])
                                if len(deltas) == null_draws and "delta_mean_pnl_bps_vs_full" in row:
                                    p05, p95 = float(np.percentile(deltas, 5.0)), float(np.percentile(deltas, 95.0))
                                    row["ablation_null_draws"] = null_draws
                                    row["ablation_null_delta_mean_bps"] = float(np.mean(deltas))
                                    row["ablation_null_delta_p05_bps"] = p05
                                    row["ablation_null_delta_p95_bps"] = p95
                                    row["ablation_delta_below_null_p05"] = bool(row["delta_mean_pnl_bps_vs_full"] < p05)
                                    row["ablation_delta_above_null_p95"] = bool(row["delta_mean_pnl_bps_vs_full"] > p95)
                        rows_for_key.extend(group_rows)
                fit_record = {"key": key, **fit_info, "fit_rows": int(fit_mask.sum()), "holdout_rows": int(holdout_mask.sum())}
                cache.write_text(json.dumps({"run_binding_sha256": run_binding_sha256, "rows": rows_for_key, "fit": fit_record}, default=_json_default), encoding="utf-8")
                all_rows.extend(rows_for_key)
                fit_reports.append(fit_record)
                print(f"[walkforward] done {key} ({_time.monotonic() - started:.0f}s)", flush=True)
            del ridge_gram

    metrics = pd.DataFrame(all_rows)
    if "ablation_group" not in metrics.columns:
        metrics = metrics.assign(ablation_group="none")
    metrics = metrics.assign(ablation_group=metrics["ablation_group"].fillna("none"))
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = summarize(metrics)
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_binding_sha256": run_binding_sha256,
        "authority": AUTHORITY,
        "created_utc": _utc_now(),
        "instrument_source_sha256": _sha256_file(Path(__file__)),
        "git_head": _git_head(),
        "inputs": {
            "dataset_dir": str(dataset_dir),
            "dataset_train_parquet": dataset.parquet_path,
            "dataset_train_manifest_sha256": dataset.manifest_sha256,
            "native_m5_root": tape.root,
            "native_m5_manifest_sha256": tape.manifest_sha256,
            "multi_tf_cache": mtf_report,
            "train_start": train_start.isoformat(),
            "train_end_exclusive": train_end_exclusive.isoformat(),
            "tape_truncated_before": (val_end_exclusive if final_holdout == "val" else train_end_exclusive).isoformat(),
        },
        "config": {
            "fold_boundaries": [b.isoformat() for b in boundaries],
            "purge_bars": purge_bars,
            "horizons": horizons,
            "knee_horizon_bars": dataset.knee_horizon_bars,
            "feature_arms": list(args.feature_arms),
            "target_scalings": list(args.target_scalings),
            "learners": list(args.learners),
            "seeds": [int(s) for s in args.seeds],
            "decision_rules": list(args.decision_rules),
            "inner_fraction": args.inner_fraction,
            "max_hgb_iter": args.max_hgb_iter,
            "hgb_learning_rate": float(args.hgb_learning_rate),
            "hgb_min_samples_leaf": int(args.hgb_min_samples_leaf),
            "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
            "coverage_grid": list(EVALUATION_COVERAGES),
            "feature_counts": {arm: int(matrix.shape[1]) for arm, matrix in arms.items()},
            "ctx_cat_one_hot": list(ctx_cat_names),
            "targets": list(targets),
            "final_holdout": final_holdout,
            "final_holdout_rows": int(len(val_dataset.time)) if val_dataset is not None else 0,
            "persist_predictions": bool(args.persist_predictions),
            "ablation": str(args.ablation),
            "ablation_null_draws": int(args.ablation_null_draws),
            "ablation_groups": {arm: {g: int(len(idx)) for g, idx in groups.items()} for arm, groups in ablation_groups.items()},
            "pattern_primitives": pattern_report,
            "cross_asset": cross_report,
            "arm_valid_rows": {arm: int(v.sum()) for arm, v in arm_valid.items()},
            "val_arm_valid_rows": {arm: int(v.sum()) for arm, v in val_arm_valid.items()},
        },
        "fits": fit_reports,
        "summary": summary,
        "elapsed_seconds": _time.monotonic() - started,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_markdown(report, metrics), encoding="utf-8")
    return report


def summarize(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    if metrics.empty:
        return []
    group_keys = ["stage", "target", "horizon_bars", "feature_arm", "target_scaling", "learner", "decision_rule", "ablation_group", "top_frac"]
    out: list[dict[str, Any]] = []
    for keys, g in metrics.groupby(group_keys, sort=True):
        valid = g[g["mean_pnl_bps"].notna()]
        out.append(
            {
                **dict(zip(group_keys, keys)),
                "fold_seed_rows": int(len(g)),
                "mean_pnl_bps_avg": float(valid["mean_pnl_bps"].mean()) if len(valid) else None,
                "mean_pnl_bps_min": float(valid["mean_pnl_bps"].min()) if len(valid) else None,
                "excess_over_coin_avg": float(valid["mean_advantage_over_coin_bps"].mean()) if len(valid) else None,
                "primary_pass_count": int(g["primary_pass"].fillna(False).astype(bool).sum()),
                "strict_pass_count": int(g["strict_pass"].fillna(False).astype(bool).sum()),
                "hit_rate_avg": float(valid["hit_rate_better_side"].mean()) if "hit_rate_better_side" in valid and len(valid) else None,
                "p_long_chosen_avg": float(valid["p_long_chosen"].mean()) if "p_long_chosen" in valid and len(valid) else None,
                "n_avg": float(valid["n"].mean()) if len(valid) else None,
            }
        )
    return out


def render_markdown(report: dict[str, Any], metrics: pd.DataFrame) -> str:
    lines = [
        "# Entry direction walk-forward research (no authority)",
        "",
        f"Created {report['created_utc']}. Folds: {', '.join(report['config']['fold_boundaries'])}. "
        f"Purge {report['config']['purge_bars']} bars. Coverage grid {report['config']['coverage_grid']}.",
        "",
        "strict_pass = preregistered primary_pass (excess over coin flip > 2 HAC SE AND mean > circular-shift p95) AND mean_pnl_bps > 0.",
        "",
        "| stage | target | h | arm | scaling | learner | rule | ablation | cov | folds×seeds | mean bps | min bps | excess | primary | strict | hit | p_long | n |",
        "|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report["summary"]:
        def f(v: Any, nd: int = 2) -> str:
            return "" if v is None else f"{v:.{nd}f}"
        lines.append(
            f"| {s['stage']} | {s['target']} | {s['horizon_bars']} | {s['feature_arm']} | {s['target_scaling']} | {s['learner']} | {s['decision_rule']} | {s['ablation_group']} | "
            f"{s['top_frac']:.2f} | {s['fold_seed_rows']} | {f(s['mean_pnl_bps_avg'])} | {f(s['mean_pnl_bps_min'])} | {f(s['excess_over_coin_avg'])} | "
            f"{s['primary_pass_count']} | {s['strict_pass_count']} | {f(s['hit_rate_avg'],3)} | {f(s['p_long_chosen_avg'],3)} | {f(s['n_avg'],0)} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--native-m5-root", required=True)
    parser.add_argument("--multi-tf-cache-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fold-boundaries", nargs="+", required=True, help="UTC timestamps; consecutive pairs form holdouts")
    parser.add_argument("--horizons", nargs="+", required=True, help="M5 bar horizons for executable close-fill targets")
    parser.add_argument("--feature-arms", nargs="+", default=list(FEATURE_ARMS), choices=list(FEATURE_ARMS))
    parser.add_argument("--target-scalings", nargs="+", default=list(TARGET_SCALINGS), choices=list(TARGET_SCALINGS))
    parser.add_argument("--learners", nargs="+", default=list(LEARNERS), choices=list(LEARNERS))
    parser.add_argument("--decision-rules", nargs="+", default=list(DECISION_RULES), choices=list(DECISION_RULES))
    parser.add_argument("--seeds", nargs="+", default=["0"], help="HGB random states")
    parser.add_argument("--inner-fraction", type=float, required=True, help="tail fraction of each fit fold used to choose ridge alpha / HGB iterations")
    parser.add_argument("--max-hgb-iter", type=int, required=True)
    parser.add_argument("--hgb-learning-rate", type=float, default=HGB_LIBRARY_DEFAULT_LEARNING_RATE, help="explicit research input; default is the pinned library default")
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=HGB_LIBRARY_DEFAULT_MIN_SAMPLES_LEAF, help="explicit research input; default is the pinned library default")
    parser.add_argument("--resume", action="store_true", help="reuse per-config results already written to out-dir")
    parser.add_argument("--targets", nargs="*", default=None, help="restrict to these target names (default: knee + every --horizons target)")
    parser.add_argument("--final-holdout", choices=["none", "val"], default="none", help="val: add a confirmation stage fitted on all TRAIN (purged) and evaluated on the VAL split")
    parser.add_argument("--persist-predictions", action="store_true", help="write per-row holdout predictions per config under out-dir/predictions")
    parser.add_argument("--pattern-primitives-parquet", default=None, help="research pattern primitives aligned by time (enables the *_patterns arms)")
    parser.add_argument("--cross-asset-daily-parquet", default=None, help="recovered daily macro log-level table ({dxy,tnx,vix,realyld}_lvl, UTC date index) for the *_cross arms")
    parser.add_argument("--cross-asset-h1-parquet", default=None, help="recovered USD_JPY H1 bars (time, close, bid_close, ask_close) for the *_cross arms")
    parser.add_argument("--ablation", choices=list(ABLATION_MODES), default="none", help="ridge-only owner-mapped group ablations: families, lanes or all")
    parser.add_argument("--ablation-null-draws", type=int, default=0, help="cardinality-matched random-column null draws per ablated group (0 = none); explicit research input")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    print(json.dumps({"out_dir": args.out_dir, "elapsed_seconds": report["elapsed_seconds"], "configs": len(report["fits"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
