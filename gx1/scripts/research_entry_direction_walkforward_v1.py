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

No TEST bytes are read: the tape is truncated at the dataset TRAIN split end
before any statistic is computed, and every fit/holdout row whose outcome
window crosses its fold boundary is dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SELECTION_MODE,
    MODEL_DIRECTION_SHORT_INDEX,
)
from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
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
FEATURE_ARMS = ("snapshot", "snapshot_mtf")
TARGET_SCALINGS = ("raw", "atr")
DECISION_RULES = ("argmax_flat", "contrast_always_trade")
RIDGE_ALPHA_GRID = tuple(float(v) for v in np.logspace(-2.0, 4.0, 13))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    knee_long: np.ndarray
    knee_short: np.ndarray
    knee_horizon_bars: int
    manifest_sha256: str
    parquet_path: str


def load_dataset(dataset_dir: Path) -> Dataset:
    parquet = dataset_dir / "entry_dataset__ENTRY_FITTED_Q_train.parquet"
    manifest_path = dataset_dir / "entry_dataset__ENTRY_FITTED_Q_train.manifest.json"
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
        ctx_cont_names=ctx_cont_names, knee_long=knee_long, knee_short=knee_short,
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
    files = sorted(native_m5_root.glob("year=*/*.parquet"))
    if not files:
        raise RuntimeError("WALKFORWARD_TAPE_EMPTY")
    frames = [
        pq.read_table(str(f), columns=["time", "close", "bid_close", "ask_close"]).to_pandas()
        for f in files
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time", kind="mergesort").reset_index(drop=True)
    frame = frame[frame["time"] < truncate_before].reset_index(drop=True)
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


def _inner_split(n_fit: int, inner_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    cut = int(math.floor(n_fit * (1.0 - inner_fraction)))
    if cut < 100 or n_fit - cut < 100:
        raise RuntimeError("WALKFORWARD_INNER_SPLIT_TOO_SMALL")
    inner_fit = np.zeros(n_fit, dtype=bool)
    inner_fit[:cut] = True
    return inner_fit, ~inner_fit


GRAM_CHUNK_ROWS = 16384


def _chunked_gram(matrix: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """Float64 X^T X accumulated over row blocks of a float32 matrix (no full float64 copy)."""
    width = int(matrix.shape[1])
    gram = np.zeros((width, width), dtype=np.float64)
    index = np.arange(matrix.shape[0]) if rows is None else rows
    for start in range(0, len(index), GRAM_CHUNK_ROWS):
        block = matrix[index[start:start + GRAM_CHUNK_ROWS]].astype(np.float64, copy=False)
        gram += block.T @ block
    return gram


def _chunked_xty(matrix: np.ndarray, y: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    index = np.arange(matrix.shape[0]) if rows is None else rows
    out = np.zeros(int(matrix.shape[1]), dtype=np.float64)
    for start in range(0, len(index), GRAM_CHUNK_ROWS):
        sel = index[start:start + GRAM_CHUNK_ROWS]
        out += matrix[sel].astype(np.float64, copy=False).T @ y[sel]
    return out


def _chunked_matvec(matrix: np.ndarray, beta: np.ndarray) -> np.ndarray:
    out = np.empty(int(matrix.shape[0]), dtype=np.float64)
    for start in range(0, matrix.shape[0], GRAM_CHUNK_ROWS):
        out[start:start + GRAM_CHUNK_ROWS] = matrix[start:start + GRAM_CHUNK_ROWS].astype(np.float64, copy=False) @ beta
    return out


class RidgeGram:
    """Standardized design + Gram matrices for one (feature arm, fold), shared by every target/side/alpha.

    Standardization statistics come from the complete fit fold (label-free).
    The alpha grid is scored on the inner chronological split with the inner
    Gram; the selected alpha is refitted on the full fit fold.  Intercepts are
    handled by centering y (features are already zero-mean on the fit fold).
    All float64 work is chunked over row blocks so the producer cap holds for
    the 1,076-wide MTF arm (the un-chunked version was cgroup-killed at 10 GiB
    on 2026-09-23).
    """

    def __init__(self, X_fit: np.ndarray, X_pred: np.ndarray, *, inner_fraction: float) -> None:
        mean = X_fit.mean(axis=0, dtype=np.float32)
        scale = X_fit.std(axis=0, dtype=np.float32)
        scale[scale == 0] = 1.0
        self.Xs_fit = ((X_fit - mean) / scale).astype(np.float32, copy=False)
        self.Xs_pred = ((X_pred - mean) / scale).astype(np.float32, copy=False)
        self.inner_fit, self.inner_val = _inner_split(len(X_fit), inner_fraction)
        self.inner_fit_rows = np.flatnonzero(self.inner_fit)
        self.inner_val_rows = np.flatnonzero(self.inner_val)
        self.gram_full = _chunked_gram(self.Xs_fit)
        self.gram_inner = _chunked_gram(self.Xs_fit, self.inner_fit_rows)
        self.n_features = int(X_fit.shape[1])
        self.mask_signature: tuple[bytes, bytes] | None = None

    def fit_predict(self, y_fit: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        y = np.asarray(y_fit, dtype=np.float64)
        eye = np.eye(self.n_features)
        inner_mean = float(np.mean(y[self.inner_fit_rows]))
        xty_inner = _chunked_xty(self.Xs_fit, y - inner_mean, self.inner_fit_rows)
        y_val = y[self.inner_val_rows]
        best_alpha, best_mse = None, math.inf
        for alpha in RIDGE_ALPHA_GRID:
            beta = np.linalg.solve(self.gram_inner + alpha * eye, xty_inner)
            pred_val = _chunked_matvec(self.Xs_fit[self.inner_val_rows], beta) + inner_mean
            mse = float(np.mean((pred_val - y_val) ** 2))
            if mse < best_mse:
                best_alpha, best_mse = alpha, mse
        full_mean = float(np.mean(y))
        xty_full = _chunked_xty(self.Xs_fit, y - full_mean)
        beta = np.linalg.solve(self.gram_full + best_alpha * eye, xty_full)
        pred = _chunked_matvec(self.Xs_pred, beta) + full_mean
        return pred, {"alpha": best_alpha, "inner_val_mse": best_mse}


def fit_hgb(
    X_fit: np.ndarray, y_fit: np.ndarray, X_pred: np.ndarray, *, inner_fraction: float, max_iter: int, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    inner_fit, inner_val = _inner_split(len(y_fit), inner_fraction)
    model = HistGradientBoostingRegressor(max_iter=int(max_iter), early_stopping=False, random_state=int(seed))
    model.fit(X_fit[inner_fit], y_fit[inner_fit])
    best_iter, best_mse = None, math.inf
    for iteration, staged in enumerate(model.staged_predict(X_fit[inner_val]), start=1):
        mse = float(np.mean((staged - y_fit[inner_val]) ** 2))
        if mse < best_mse:
            best_iter, best_mse = iteration, mse
    pred = None
    for iteration, staged in enumerate(model.staged_predict(X_pred), start=1):
        if iteration == best_iter:
            pred = staged
            break
    if pred is None:
        raise RuntimeError("WALKFORWARD_HGB_STAGED_PREDICT_FAILED")
    return np.asarray(pred, dtype=np.float64), {"best_iter": best_iter, "inner_val_mse": best_mse, "max_iter": int(max_iter)}


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
            extra.update(
                {
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = _time.monotonic()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.resume:
        raise RuntimeError("WALKFORWARD_OUT_DIR_EXISTS")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_config_dir = out_dir / "per_config"
    per_config_dir.mkdir(exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    dataset = load_dataset(dataset_dir)
    train_manifest = json.loads((dataset_dir / "entry_dataset__ENTRY_FITTED_Q_train.manifest.json").read_text(encoding="utf-8"))
    train_start = pd.Timestamp(train_manifest["splits"]["train"]["start"])
    train_end_exclusive = pd.Timestamp(train_manifest["splits"]["val"]["start"])
    tape = load_tape(Path(args.native_m5_root), truncate_before=train_end_exclusive)
    positions = tape_positions(dataset.time, tape)
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
    arms: dict[str, np.ndarray] = {"snapshot": snapshot}
    mtf_report: dict[str, Any] = {}
    if "snapshot_mtf" in args.feature_arms:
        if not args.multi_tf_cache_dir:
            raise RuntimeError("WALKFORWARD_MTF_ARM_REQUIRES_CACHE_DIR")
        mtf_matrix, mtf_names, mtf_report = load_mtf_last_closed(Path(args.multi_tf_cache_dir), dataset.time, dataset.ctx_cont, dataset.ctx_cont_names)
        arms["snapshot_mtf"] = np.concatenate([snapshot, mtf_matrix], axis=1)
        del mtf_matrix
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
    month_sign = month_drift_sign(tape)

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

    for arm in args.feature_arms:
        X = arms[arm]
        for fold in folds:
            pending = [
                cfg for cfg in _config_keys(arm, fold)
                if not (per_config_dir / f"{cfg[0]}__{arm}__{cfg[1]}__{cfg[2]}__seed{cfg[3]}__fold{fold.index}.json").is_file()
            ]
            for cfg in _config_keys(arm, fold):
                if cfg in pending:
                    continue
                cache = per_config_dir / f"{cfg[0]}__{arm}__{cfg[1]}__{cfg[2]}__seed{cfg[3]}__fold{fold.index}.json"
                payload = json.loads(cache.read_text(encoding="utf-8"))
                all_rows.extend(payload["rows"])
                fit_reports.append(payload["fit"])
            if not pending:
                continue
            base_fit_mask, base_holdout_mask = fold_masks(dataset.time, positions, fold, purge_bars=purge_bars)
            ridge_gram: RidgeGram | None = None
            for target_name, scaling, learner, seed in pending:
                long_y, short_y, valid_y, horizon = targets[target_name]
                key = f"{target_name}__{arm}__{scaling}__{learner}__seed{seed}__fold{fold.index}"
                cache = per_config_dir / f"{key}.json"
                fit_mask = base_fit_mask & valid_y
                holdout_mask = base_holdout_mask & valid_y
                if fold.holdout_end < train_end_exclusive:
                    next_first = int(positions[np.asarray(dataset.time >= fold.holdout_end, dtype=bool)].min())
                    holdout_mask &= (positions + int(horizon)) < next_first
                y_scale_fit = atr[fit_mask] if scaling == "atr" else np.ones(int(fit_mask.sum()))
                y_scale_hold = atr[holdout_mask] if scaling == "atr" else np.ones(int(holdout_mask.sum()))
                preds: dict[str, np.ndarray] = {}
                fit_info: dict[str, Any] = {}
                for side_name, y_all in (("long", long_y), ("short", short_y)):
                    y_fit = y_all[fit_mask] / y_scale_fit
                    if learner == "ridge":
                        # The Gram is shared across targets whenever the fit/holdout row sets coincide.
                        if ridge_gram is None or ridge_gram.Xs_fit.shape[0] != int(fit_mask.sum()) or ridge_gram.Xs_pred.shape[0] != int(holdout_mask.sum()) or ridge_gram.mask_signature != (fit_mask.tobytes(), holdout_mask.tobytes()):
                            ridge_gram = RidgeGram(X[fit_mask], X[holdout_mask], inner_fraction=args.inner_fraction)
                            ridge_gram.mask_signature = (fit_mask.tobytes(), holdout_mask.tobytes())
                        pred, info = ridge_gram.fit_predict(y_fit)
                    else:
                        pred, info = fit_hgb(X[fit_mask], y_fit, X[holdout_mask], inner_fraction=args.inner_fraction, max_iter=args.max_hgb_iter, seed=seed)
                    preds[side_name] = pred * y_scale_hold
                    fit_info[side_name] = info
                rows_for_key: list[dict[str, Any]] = []
                for rule in args.decision_rules:
                    side, score, contrast = decisions(preds["long"], preds["short"], rule)
                    frame = pd.DataFrame(
                        {
                            "split": f"fold{fold.index}",
                            "model": f"{key}__{rule}",
                            "time": dataset.time[holdout_mask],
                            "pred_direction": side,
                            "selection_score": score,
                            "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
                            "edge_score": contrast,
                            RESEARCH_LONG_OUTCOME_COLUMN: long_y[holdout_mask],
                            RESEARCH_SHORT_OUTCOME_COLUMN: short_y[holdout_mask],
                        }
                    )
                    meta = {
                        "target": target_name, "horizon_bars": int(horizon), "feature_arm": arm,
                        "target_scaling": scaling, "learner": learner, "seed": int(seed), "decision_rule": rule,
                        "fit_rows": int(fit_mask.sum()),
                        "outcome_definition": (
                            "dataset knee-horizon M1 final PnL (research gross spread-inclusive)"
                            if target_name == KNEE_TARGET_NAME
                            else "native M5 tape executable close-fill return (research convention)"
                        ),
                    }
                    rows_for_key.extend(evaluate_frame(frame, fold=fold, month_sign=month_sign, meta=meta))
                fit_record = {"key": key, **fit_info, "fit_rows": int(fit_mask.sum()), "holdout_rows": int(holdout_mask.sum())}
                cache.write_text(json.dumps({"rows": rows_for_key, "fit": fit_record}, default=_json_default), encoding="utf-8")
                all_rows.extend(rows_for_key)
                fit_reports.append(fit_record)
                print(f"[walkforward] done {key} ({_time.monotonic() - started:.0f}s)", flush=True)
            del ridge_gram

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = summarize(metrics)
    report = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "created_utc": _utc_now(),
        "inputs": {
            "dataset_dir": str(dataset_dir),
            "dataset_train_parquet": dataset.parquet_path,
            "dataset_train_manifest_sha256": dataset.manifest_sha256,
            "native_m5_root": tape.root,
            "native_m5_manifest_sha256": tape.manifest_sha256,
            "multi_tf_cache": mtf_report,
            "train_start": train_start.isoformat(),
            "train_end_exclusive": train_end_exclusive.isoformat(),
            "tape_truncated_before": train_end_exclusive.isoformat(),
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
            "ridge_alpha_grid": list(RIDGE_ALPHA_GRID),
            "coverage_grid": list(EVALUATION_COVERAGES),
            "feature_counts": {arm: int(matrix.shape[1]) for arm, matrix in arms.items()},
            "ctx_cat_one_hot": list(ctx_cat_names),
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
    group_keys = ["target", "horizon_bars", "feature_arm", "target_scaling", "learner", "decision_rule", "top_frac"]
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
        "| target | h | arm | scaling | learner | rule | cov | folds×seeds | mean bps | min bps | excess | primary | strict | hit | p_long | n |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report["summary"]:
        def f(v: Any, nd: int = 2) -> str:
            return "" if v is None else f"{v:.{nd}f}"
        lines.append(
            f"| {s['target']} | {s['horizon_bars']} | {s['feature_arm']} | {s['target_scaling']} | {s['learner']} | {s['decision_rule']} | "
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
    parser.add_argument("--resume", action="store_true", help="reuse per-config results already written to out-dir")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    print(json.dumps({"out_dir": args.out_dir, "elapsed_seconds": report["elapsed_seconds"], "configs": len(report["fits"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
