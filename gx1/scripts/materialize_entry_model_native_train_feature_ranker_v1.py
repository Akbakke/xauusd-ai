#!/usr/bin/env python3
"""Materialize one immutable TRAIN-only feature ranking for the seq513 manifest.

This is the producer side of `entry_model_native_train_feature_ranking_v1`,
consumed exclusively by
`gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1`.
The 305 mandatory causal-layer fields are code-owned and never ranked here;
this producer scores only the CANDIDATE pool competing for the 174 remainder
positions.

Determinism contract (declared in the emitted JSON and enforced downstream):
- fit scope is TRAIN only: every feature value and every target value comes
  from rows at or before --train-end; no validation/test rows are read;
- the target is the forward 24-bar mid-close log return (bps), computed only
  where the full horizon stays inside the TRAIN window;
- score = |Spearman rank correlation| between candidate and target over rows
  where both are finite; candidates with fewer than MIN_SUPPORT_FRACTION
  finite rows score exactly 0.0;
- ordering is score descending with feature-name-ascending tie-break;
- candidate values are computed by the dataset builder's own
  `_build_inline_seq_structure_extension` (one truth, no duplicated layer
  math). The ranker feeds it the ctx_cont columns present in the source
  parquet; the dataset builder later computes the full 142-ctx surface, so
  ranking inputs are a deterministic subset, never an alternate math path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
)
from gx1.contracts.signal_bridge_v3 import (
    ORDERED_CTX_CAT_NAMES_V3,
    ORDERED_CTX_CONT_NAMES_V3,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    classify_entry_specialist_feature,
)
from gx1.features.volume_features import add_volume_features
from gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1 import (
    TRAIN_FEATURE_CAUSALITY_CONTRACT,
    TRAIN_FEATURE_RANKING_ORDER,
    TRAIN_FEATURE_RANKING_PRODUCER,
    TRAIN_FEATURE_RANKING_PRODUCER_VERSION,
    TRAIN_FEATURE_RANKING_SCHEMA_VERSION,
    _is_forbidden_leak_name,
)
from gx1_guards.gates import require_retrain_vedtak


RANKING_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING"
TARGET_HORIZON_BARS = 24
MIN_SUPPORT_FRACTION = 0.10
SCORE_DECIMALS = 12


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc_arg(value: str, *, field: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise RuntimeError(f"FEATURE_RANKER_{field.upper()}_NOT_UTC: {value!r}")
    return ts.tz_convert("UTC")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_target(times: np.ndarray, values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"entry_model_native_train_feature_ranker_target_v1")
    digest.update(np.ascontiguousarray(times.astype("datetime64[ns]").view(np.int64)).tobytes())
    digest.update(np.ascontiguousarray(values.astype(np.float64)).tobytes())
    return digest.hexdigest()


def _candidate_universe(source_ctx_cont: Sequence[str]) -> List[str]:
    """Deterministic candidate pool: layer extras + classifiable ctx pass-throughs."""
    from gx1.features import entry_model_native_feature_layers_v1 as _fl
    import gx1.features.entry_chart_geometry_v1 as _cg
    import gx1.features.entry_momentum_flow_v1 as _mf
    import gx1.features.entry_mtf_confluence_v1 as _mtf
    import gx1.features.entry_session_regime_interactions_v1 as _sr
    import gx1.features.entry_smc_liquidity_quality_v1 as _smc
    import gx1.features.entry_structure_swing_derivations_v1 as _ssw
    import gx1.features.entry_support_resistance_memory_v1 as _srm
    import gx1.features.entry_trend_ema_v1 as _te
    import gx1.features.entry_vol_compression_v1 as _vc

    union: set[str] = set()
    for module in (_fl, _cg, _mf, _mtf, _sr, _smc, _ssw, _srm, _te, _vc):
        for attr in dir(module):
            if attr.isupper() and attr.endswith("_FEATURE_NAMES"):
                values = getattr(module, attr)
                if (
                    isinstance(values, (list, tuple))
                    and values
                    and all(isinstance(v, str) for v in values)
                ):
                    union.update(values)
    union.update(f"ctx_cont.{name}" for name in source_ctx_cont)

    mandatory = set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)
    base = set(MODEL_NATIVE_BASE_FIELDS)
    forbidden = set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
    allowed = set(MODEL_NATIVE_TRAINING_SPECIALISTS)
    return sorted(
        name
        for name in union
        if name not in mandatory
        and name not in base
        and name not in forbidden
        and not _is_forbidden_leak_name(name)
        and classify_entry_specialist_feature(name) in allowed
    )


ATTACH_WORKERS = 12
ATTACH_SPOT_CHECK_ROWS = 40


def _attach_chunk_worker(args: tuple) -> tuple:
    """Row-loop worker over [lo, hi) against the SHARED full-series context.

    _ATTACH_SHARED (ctx, ts_index, extract) is inherited copy-on-write from the
    fork; every worker indexes the SAME precomputed full-series arrays (incl.
    the trailing-1yr percentile arrays), so chunk values are exact by
    construction — no overlap needed.
    """
    chunk_index, lo, hi = args
    ctx, ts_index, extract = _ATTACH_SHARED
    from gx1.scripts.augment_forward_outcome_v2 import compute_attach_rows

    cols = compute_attach_rows(ctx, ts_index, lo, hi, extract=extract)
    return chunk_index, lo, hi, {k: v[lo:hi] for k, v in cols.items()}


_ATTACH_SHARED: tuple = ()


def _attach_group_a_parallel(
    frame: pd.DataFrame,
    mtf: dict,
    *,
    workers: int = ATTACH_WORKERS,
) -> pd.DataFrame:
    """Parallel attach: ONE full-series context, row loop fanned over workers.

    The context (build_attach_context) is built once in the parent from the
    complete frame, so long-memory arrays are identical for every worker.
    After the merge, ATTACH_SPOT_CHECK_ROWS deterministic rows are recomputed
    serially in the parent and asserted bit-exact against the merged result.
    """
    import multiprocessing as mp

    from gx1.scripts.augment_forward_outcome_v2 import (
        build_attach_context,
        compute_attach_rows,
        finalize_attach_columns,
    )

    global _ATTACH_SHARED
    ctx, ts_index, extract, dip_from_aug = build_attach_context(
        frame, multi_tf=mtf, journal_label="train_feature_ranker"
    )
    n = len(frame)
    bounds = np.linspace(0, n, workers + 1, dtype=int)
    tasks = [
        (k, int(bounds[k]), int(bounds[k + 1])) for k in range(workers)
    ]
    _ATTACH_SHARED = (ctx, ts_index, extract)
    try:
        mp_ctx = mp.get_context("fork")
        with mp_ctx.Pool(processes=workers) as pool:
            results = pool.map(_attach_chunk_worker, tasks)
    finally:
        _ATTACH_SHARED = ()
    results.sort(key=lambda item: item[0])

    cols = {k: np.full(n, np.nan, dtype=np.float32) for k in extract}
    for _, lo, hi, chunk_cols in results:
        for k, values in chunk_cols.items():
            cols[k][lo:hi] = values

    # Deterministic serial spot-check: evenly spaced finite rows, bit-exact.
    finite = np.flatnonzero(np.isfinite(cols[extract[0]]))
    if len(finite) == 0:
        raise RuntimeError("FEATURE_RANKER_PARALLEL_ATTACH_ALL_NAN")
    picks = finite[
        np.linspace(0, len(finite) - 1, min(ATTACH_SPOT_CHECK_ROWS, len(finite)))
        .astype(int)
    ]
    for i in picks:
        serial = compute_attach_rows(
            ctx, ts_index, int(i), int(i) + 1, extract=extract
        )
        for k in extract:
            a = np.float32(serial[k][int(i)])
            b = cols[k][int(i)]
            if not (a == b or (np.isnan(a) and np.isnan(b))):
                raise RuntimeError(
                    "FEATURE_RANKER_PARALLEL_ATTACH_SPOT_CHECK_MISMATCH: "
                    f"row={int(i)} col={k} serial={a} parallel={b}"
                )

    return finalize_attach_columns(
        frame, cols, smc_col="smc_swing_state", dip_from_aug=dip_from_aug
    )


def _load_train_frame(
    source_parquet: Path,
    *,
    history_start: pd.Timestamp,
    train_end: pd.Timestamp,
    mtf_cache_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    frame = pd.read_parquet(source_parquet)
    if "time" not in frame.columns:
        raise RuntimeError("FEATURE_RANKER_SOURCE_TIME_COLUMN_MISSING")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.loc[
        (frame["time"] >= history_start) & (frame["time"] <= train_end)
    ].sort_values("time").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("FEATURE_RANKER_TRAIN_WINDOW_EMPTY")
    frame = add_volume_features(frame)

    # GROUP_A + DIP_STRUCT ctx columns are not carried by the source parquet;
    # recompute them through the SAME one-truth augmenter the dataset builder
    # uses (identical values by construction), then trim the causal warmup.
    from gx1.contracts.signal_bridge_v3 import (
        ORDERED_CTX_CONT_DIP_STRUCT,
        ORDERED_CTX_CONT_GROUP_A_PARITY,
    )
    from gx1.features.htf_features import load_multi_tf_v2_cache
    from gx1.scripts.augment_forward_outcome_v2 import (
        trim_causal_context_warmup_prefix,
    )

    if not mtf_cache_dir.is_dir():
        raise RuntimeError(f"FEATURE_RANKER_MTF_CACHE_MISSING: {mtf_cache_dir}")
    group_a_required = list(ORDERED_CTX_CONT_GROUP_A_PARITY) + list(
        ORDERED_CTX_CONT_DIP_STRUCT
    )
    frame = frame.drop(
        columns=[name for name in group_a_required if name in frame.columns]
    )
    # The one-truth augmenter computes augment_candidate per row (~85 ms/row =
    # ~9 h serial over the full range). It is strictly causal, so we chunk it
    # across workers with >= warmup overlap and verify chunk boundaries
    # bit-exactly (see _attach_group_a_parallel).
    frame = _attach_group_a_parallel(
        frame,
        load_multi_tf_v2_cache(mtf_cache_dir),
    )
    frame = trim_causal_context_warmup_prefix(frame, group_a_required).reset_index(
        drop=True
    )

    # ENTRY_SMART_DERIVED ctx family (smc pressures, S/R proximities, session
    # flags) — same one-truth call the dataset builder makes after the
    # GROUP_A attach.
    from gx1.features.entry_smart_context import add_entry_smart_context_features

    add_entry_smart_context_features(frame)
    # Session flag exactly as the dataset builder derives it
    # (build_entry_v10_ctx_training_dataset_v3.py:1799-1800).
    if "is_ASIA" not in frame.columns:
        frame["is_ASIA"] = (frame["session_id"].astype(int) == 0).astype(np.int8)
    missing_ctx = [n for n in ORDERED_CTX_CONT_NAMES_V3 if n not in frame.columns]
    if missing_ctx:
        raise RuntimeError(
            f"FEATURE_RANKER_CTX_INCOMPLETE_AFTER_DERIVATION: {missing_ctx[:10]} "
            f"total={len(missing_ctx)}"
        )

    missing_base = [f for f in MODEL_NATIVE_BASE_FIELDS if f not in frame.columns]
    if missing_base:
        raise RuntimeError(f"FEATURE_RANKER_BASE_FIELDS_MISSING: {missing_base}")
    source_ctx_cont = [n for n in ORDERED_CTX_CONT_NAMES_V3 if n in frame.columns]
    missing_cat = [n for n in ORDERED_CTX_CAT_NAMES_V3 if n not in frame.columns]
    if missing_cat:
        raise RuntimeError(f"FEATURE_RANKER_CTX_CAT_MISSING: {missing_cat}")
    return frame, source_ctx_cont


def _compute_candidate_matrix(
    frame: pd.DataFrame,
    *,
    source_parquet: Path,
    candidates: Sequence[str],
    source_ctx_cont: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Compute candidate values via the dataset builder's one-truth extension."""
    from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
        _build_inline_seq_structure_extension,
    )

    matrix, names, _meta = _build_inline_seq_structure_extension(
        frame,
        requested_features=list(candidates),
        ctx_cont_names=list(source_ctx_cont),
        ctx_cat_names=list(ORDERED_CTX_CAT_NAMES_V3),
        source_parquet=source_parquet,
        source_contract_label="train_feature_ranker_v1",
    )
    matrix = np.asarray(matrix, dtype=np.float32)
    index_by_name = {name: column for column, name in enumerate(names)}
    missing = [name for name in candidates if name not in index_by_name]
    if missing:
        raise RuntimeError(
            f"FEATURE_RANKER_EXTENSION_MISSING_CANDIDATES: {missing[:10]} "
            f"total={len(missing)}"
        )
    ordered = matrix[:, [index_by_name[name] for name in candidates]]
    return ordered, list(candidates)


def _forward_return_target(
    frame: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward TARGET_HORIZON_BARS mid-close log return in bps, TRAIN-capped."""
    bid = frame["bid_close"].astype(np.float64).to_numpy()
    ask = frame["ask_close"].astype(np.float64).to_numpy()
    mid = (bid + ask) / 2.0
    if not np.isfinite(mid).all() or (mid <= 0).any():
        raise RuntimeError("FEATURE_RANKER_MID_CLOSE_INVALID")
    future = np.roll(mid, -TARGET_HORIZON_BARS)
    target = (np.log(future) - np.log(mid)) * 1e4
    target[-TARGET_HORIZON_BARS:] = np.nan
    times = frame["time"].to_numpy()
    in_fit = (frame["time"] >= train_start) & (frame["time"] <= train_end)
    target[~in_fit.to_numpy()] = np.nan
    return times, target


def _spearman_scores(
    matrix: np.ndarray,
    names: Sequence[str],
    target: np.ndarray,
) -> Dict[str, float]:
    valid_target = np.isfinite(target)
    n_rows = int(valid_target.sum())
    if n_rows < 1000:
        raise RuntimeError(f"FEATURE_RANKER_TARGET_SUPPORT_TOO_SMALL: {n_rows}")
    target_series = pd.Series(target)
    scores: Dict[str, float] = {}
    for column, name in enumerate(names):
        values = matrix[:, column].astype(np.float64)
        both = valid_target & np.isfinite(values)
        support = int(both.sum())
        if support < MIN_SUPPORT_FRACTION * n_rows:
            scores[name] = 0.0
            continue
        feature_rank = pd.Series(values[both]).rank(method="average")
        target_rank = target_series[both].rank(method="average")
        with np.errstate(invalid="ignore"):
            rho = float(np.corrcoef(feature_rank, target_rank)[0, 1])
        scores[name] = 0.0 if not math.isfinite(rho) else round(abs(rho), SCORE_DECIMALS)
    return scores


def emit_ranking(
    *,
    out_dir: Path,
    vedtak: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    source_time_max: pd.Timestamp,
    target_time_max: pd.Timestamp,
    source_sha256: str,
    target_sha256: str,
    scores: Dict[str, float],
    created: datetime | None = None,
) -> Path:
    created = created or _utc_now()
    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    out_path = out_dir / f"{RANKING_EVENT_PREFIX}_{stamp}.json"
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    payload: Dict[str, Any] = {
        "schema_version": TRAIN_FEATURE_RANKING_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "explicit_vedtak_id": vedtak,
        "producer": TRAIN_FEATURE_RANKING_PRODUCER,
        "producer_version": TRAIN_FEATURE_RANKING_PRODUCER_VERSION,
        "fit_scope": "train_only",
        "train_start_utc": train_start.isoformat(),
        "train_end_utc": train_end.isoformat(),
        "source_time_max_utc": source_time_max.isoformat(),
        "target_time_max_utc": target_time_max.isoformat(),
        "source_sha256": source_sha256,
        "target_sha256": target_sha256,
        "ranking_order": dict(TRAIN_FEATURE_RANKING_ORDER),
        "causality_contract": dict(TRAIN_FEATURE_CAUSALITY_CONTRACT),
        "ranked_features": [
            {"rank": index, "name": name, "score": float(score)}
            for index, (name, score) in enumerate(ranked, start=1)
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(out_path, flags, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vedtak", required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--mtf-cache-dir", type=Path, required=True)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    vedtak = require_retrain_vedtak(args.vedtak)
    history_start = _parse_utc_arg(args.history_start, field="history_start")
    train_start = _parse_utc_arg(args.train_start, field="train_start")
    train_end = _parse_utc_arg(args.train_end, field="train_end")
    if not history_start <= train_start < train_end:
        raise RuntimeError("FEATURE_RANKER_WINDOW_ORDER_INVALID")

    source_parquet = args.source_parquet.expanduser().resolve()
    if not source_parquet.is_file():
        raise RuntimeError(f"FEATURE_RANKER_SOURCE_MISSING: {source_parquet}")
    source_sha256 = _sha256_file(source_parquet)

    # Checkpoint: attach + extension cost hours; a trivial late failure must
    # never force recomputation. Bound to source sha + exact window key.
    out_dir = args.out_dir.expanduser().resolve()
    checkpoint_path = out_dir / "_ranker_checkpoint.npz"
    window_key = (
        f"{history_start.isoformat()}|{train_start.isoformat()}|{train_end.isoformat()}"
    )
    matrix = None
    if checkpoint_path.exists():
        ck = np.load(checkpoint_path, allow_pickle=False)
        if (
            str(ck["source_sha256"]) == source_sha256
            and str(ck["window_key"]) == window_key
        ):
            matrix = ck["matrix"]
            names = [str(n) for n in ck["names"]]
            times = ck["times_ns"].view("datetime64[ns]")
            target = ck["target"]
            train_rows = int(ck["train_rows"])
            source_time_max = pd.Timestamp(int(ck["source_time_max_ns"]), tz="UTC")
            print(f"[CHECKPOINT] gjenbrukt {checkpoint_path}", flush=True)
        else:
            print("[CHECKPOINT] stale (sha/vindu-avvik) — beregner på nytt", flush=True)

    if matrix is None:
        frame, source_ctx_cont = _load_train_frame(
            source_parquet,
            history_start=history_start,
            train_end=train_end,
            mtf_cache_dir=args.mtf_cache_dir.expanduser().resolve(),
        )
        candidates = _candidate_universe(source_ctx_cont)
        if len(candidates) < 174:
            raise RuntimeError(
                f"FEATURE_RANKER_CANDIDATE_POOL_TOO_SMALL: {len(candidates)} < 174"
            )
        matrix, names = _compute_candidate_matrix(
            frame,
            source_parquet=source_parquet,
            candidates=candidates,
            source_ctx_cont=source_ctx_cont,
        )
        times, target = _forward_return_target(
            frame, train_start=train_start, train_end=train_end
        )
        train_rows = int(len(frame))
        _stm = pd.Timestamp(frame["time"].max())
        source_time_max = (
            _stm.tz_convert("UTC") if _stm.tzinfo else _stm.tz_localize("UTC")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            checkpoint_path,
            matrix=np.asarray(matrix, dtype=np.float32),
            names=np.array(list(names)),
            times_ns=pd.DatetimeIndex(times).asi8,
            target=np.asarray(target, dtype=np.float64),
            train_rows=np.int64(train_rows),
            source_time_max_ns=np.int64(source_time_max.value),
            source_sha256=np.array(source_sha256),
            window_key=np.array(window_key),
        )
        print(f"[CHECKPOINT] skrevet {checkpoint_path}", flush=True)

    finite_target = np.isfinite(target)
    target_sha256 = _sha256_target(
        np.asarray(times)[finite_target], np.asarray(target)[finite_target]
    )
    if target_sha256 == source_sha256:
        raise RuntimeError("FEATURE_RANKER_HASH_COLLISION")
    scores = _spearman_scores(np.asarray(matrix, dtype=np.float32), names, target)

    _tmax = pd.Timestamp(np.asarray(times)[finite_target].max())
    target_time_max = (
        _tmax.tz_convert("UTC") if _tmax.tzinfo else _tmax.tz_localize("UTC")
    )

    out_path = emit_ranking(
        out_dir=args.out_dir.expanduser().resolve(),
        vedtak=vedtak,
        train_start=train_start,
        train_end=train_end,
        source_time_max=source_time_max,
        target_time_max=target_time_max,
        source_sha256=source_sha256,
        target_sha256=target_sha256,
        scores=scores,
    )
    nonzero = sum(1 for s in scores.values() if s > 0.0)
    print(
        json.dumps(
            {
                "event": "ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING_WRITTEN",
                "out_path": str(out_path),
                "candidates": len(scores),
                "nonzero_scores": nonzero,
                "train_rows": train_rows,
                "target_rows": int(finite_target.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
