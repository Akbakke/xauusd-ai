#!/usr/bin/env python3
"""Materialize one immutable TRAIN-only feature ranking for the seq513 manifest.

This is the producer side of `entry_model_native_train_feature_ranking_v1`,
consumed exclusively by
`gx1.scripts.materialize_entry_model_native_seq513_signal_manifest_v1`.
The mandatory causal-layer fields are code-owned and never ranked here; this
producer scores only the CANDIDATE pool competing for the ranked remainder
positions.

Determinism contract (declared in the emitted JSON and enforced downstream):
- fit scope is TRAIN only: every feature value and every target value comes
  from rows at or before --train-end; no validation/test rows are read;
- the target is the exact spread-aware LONG-minus-SHORT direction-utility
  margin used by the model-native labels: final PnL at 24 bars plus the
  contracted first-10-bar MFE/MAE/path utility, computed only where both
  horizons stay inside the TRAIN window;
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
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from gx1.features.htf_features import MULTI_TF_TIMEFRAMES
from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
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
    TRAIN_FEATURE_RANKING_TARGET_CONTRACT,
    _is_forbidden_leak_name,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
    TrainRankReferenceV2,
    apply_train_rank_reference_v2,
    parse_utc,
    validate_train_rank_reference_lineage_v2,
)
from gx1.time.session_detector import ASIA_SESSION_ID
from gx1.scripts.audit_seq513_source_cascade_v1 import (
    validate_seq513_source_cascade_proof,
)


RANKING_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_TRAIN_FEATURE_RANKING"
DIRECTION_HORIZON_BARS = int(
    TRAIN_FEATURE_RANKING_TARGET_CONTRACT["direction_horizon_bars"]
)
PATH_HORIZON_BARS = int(
    TRAIN_FEATURE_RANKING_TARGET_CONTRACT["path_horizon_bars"]
)
UTILITY_MFE_WEIGHT = float(
    TRAIN_FEATURE_RANKING_TARGET_CONTRACT["mfe_weight"]
)
UTILITY_MAE_WEIGHT = float(
    TRAIN_FEATURE_RANKING_TARGET_CONTRACT["mae_weight"]
)
UTILITY_PATH_WEIGHT = float(
    TRAIN_FEATURE_RANKING_TARGET_CONTRACT["path_weight"]
)
MIN_SUPPORT_FRACTION = 0.10
SCORE_DECIMALS = 12
RANKER_CHECKPOINT_SCHEMA_VERSION = "entry_model_native_train_feature_ranker_checkpoint_v6"
_RANKING_OUTPUT_RE = re.compile(
    rf"^{RANKING_EVENT_PREFIX}_(\d{{8}}T\d{{6}}(?:\d{{6}})?Z)\.json$"
)


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


def _mtf_cache_sha256(cache_dir: Path) -> str:
    """Bind the exact manifest-named M5/M15/H1/H4/D1 cache bytes."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"FEATURE_RANKER_MTF_CACHE_MANIFEST_MISSING: {manifest_path}")
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("FEATURE_RANKER_MTF_CACHE_MANIFEST_INVALID") from exc
    tfs = manifest.get("tfs") if isinstance(manifest, dict) else None
    if not isinstance(tfs, dict) or set(tfs) != set(MULTI_TF_TIMEFRAMES):
        raise RuntimeError("FEATURE_RANKER_MTF_CACHE_MANIFEST_TFS_INVALID")
    digest = hashlib.sha256()
    digest.update(b"entry_model_native_ranker_mtf_cache_v1\0")
    digest.update(raw)
    for tf_name in MULTI_TF_TIMEFRAMES:
        info = tfs[tf_name]
        if not isinstance(info, dict):
            raise RuntimeError(f"FEATURE_RANKER_MTF_CACHE_ENTRY_INVALID: {tf_name}")
        for field in ("feats_npy", "ts_npy"):
            filename = str(info.get(field) or "")
            if Path(filename).name != filename:
                raise RuntimeError(
                    f"FEATURE_RANKER_MTF_CACHE_FILENAME_INVALID: {tf_name}.{field}"
                )
            path = cache_dir / filename
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"FEATURE_RANKER_MTF_CACHE_FILE_MISSING: {path}")
            digest.update(tf_name.encode("ascii") + b"\0" + field.encode("ascii") + b"\0")
            digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _ranker_checkpoint_key(
    *,
    run_id: str,
    source_sha256: str,
    mtf_cache_sha256: str,
    source_cascade_sha256: str,
    rank_reference_sha256: str,
    rank_reference_sidecar_sha256: str,
    history_start: pd.Timestamp,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> str:
    payload = {
        "schema_version": RANKER_CHECKPOINT_SCHEMA_VERSION,
        "entry_run_id": run_id,
        "source_sha256": source_sha256,
        "mtf_cache_sha256": mtf_cache_sha256,
        "source_cascade_sha256": source_cascade_sha256,
        "rank_reference_sha256": rank_reference_sha256,
        "rank_reference_sidecar_sha256": rank_reference_sidecar_sha256,
        "history_start_utc": history_start.isoformat(),
        "train_start_utc": train_start.isoformat(),
        "train_end_utc": train_end.isoformat(),
        "producer_version": TRAIN_FEATURE_RANKING_PRODUCER_VERSION,
        "target_contract": TRAIN_FEATURE_RANKING_TARGET_CONTRACT,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_ranker_checkpoint(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"FEATURE_RANKER_CHECKPOINT_ALREADY_EXISTS: {path}")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"FEATURE_RANKER_CHECKPOINT_PARTIAL_EXISTS: {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _sha256_target(times: np.ndarray, values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"entry_model_native_train_feature_ranker_target_v2")
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


ATTACH_WORKERS = 8


def _load_ranker_common_history_m5(
    *,
    train_end: pd.Timestamp,
    source_parquet: Path,
    source_cascade: dict[str, Any],
) -> pd.DataFrame:
    """Load the exact canonical M5 history used by the dataset Group-A path."""
    if (
        not isinstance(source_cascade, dict)
        or str(source_cascade.get("schema_version"))
        != "seq513_source_cascade_pair_proof_v1"
    ):
        raise RuntimeError("FEATURE_RANKER_CURRENT_SOURCE_CASCADE_REQUIRED")
    current = pd.read_parquet(
        Path(source_parquet).expanduser().resolve(),
        columns=["time", "open", "high", "low", "close"],
    )
    current["time"] = pd.to_datetime(current["time"], utc=True, errors="raise")
    current = current.loc[
        (current["time"] >= pd.Timestamp("2020-01-01T00:00:00Z"))
        & (current["time"] <= train_end)
    ]
    out = current.set_index("time")[["open", "high", "low", "close"]].sort_index()
    if (
        out.empty
        or not out.index.is_unique
        or not out.index.is_monotonic_increasing
        or out.index.max() > pd.Timestamp(train_end)
    ):
        raise RuntimeError("FEATURE_RANKER_CURRENT_SOURCE_HISTORY_INVALID")
    return out


def _attach_ranker_group_a_with_common_history(
    frame: pd.DataFrame,
    *,
    multi_tf: dict,
    context_m5: pd.DataFrame,
    checkpoint_dir: Path | None = None,
    checkpoint_key: str | None = None,
    workers: int = ATTACH_WORKERS,
) -> pd.DataFrame:
    """Run the shared Group-A owner with mandatory dataset-equivalent history."""
    if not isinstance(context_m5, pd.DataFrame) or context_m5.empty:
        raise RuntimeError("FEATURE_RANKER_COMMON_HISTORY_REQUIRED")
    from gx1.scripts.augment_forward_outcome_v2 import (
        attach_group_a_dip_struct_ctx_columns_parallel,
    )

    return attach_group_a_dip_struct_ctx_columns_parallel(
        frame,
        multi_tf=multi_tf,
        journal_label="train_feature_ranker",
        workers=workers,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
        context_m5=context_m5,
    )


def _load_train_frame(
    source_parquet: Path,
    *,
    history_start: pd.Timestamp,
    train_end: pd.Timestamp,
    mtf_cache_dir: Path,
    checkpoint_dir: Path,
    checkpoint_key: str,
    rank_reference: TrainRankReferenceV2,
    context_m5: pd.DataFrame,
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
    # Apply the same immutable TRAIN-fit ECDF/ATR state before any downstream
    # regime, GROUP_A, smart-context, or candidate construction.  Ranking on
    # source-provided buckets while dataset/live overwrite them is forbidden.
    frame = apply_train_rank_reference_v2(frame, rank_reference)
    frame = add_volume_features(frame)

    # GROUP_A + DIP_STRUCT ctx columns are not carried by the source parquet;
    # recompute them through the SAME one-truth augmenter the dataset builder
    # uses (identical values by construction), then trim the causal warmup.
    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS,
        MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS,
    )
    from gx1.features.htf_features import load_multi_tf_cache
    from gx1.scripts.augment_forward_outcome_v2 import (
        trim_causal_context_warmup_prefix,
    )

    if not mtf_cache_dir.is_dir():
        raise RuntimeError(f"FEATURE_RANKER_MTF_CACHE_MISSING: {mtf_cache_dir}")
    group_a_required = list(MODEL_NATIVE_CTX_CONT_GROUP_A_FIELDS) + list(
        MODEL_NATIVE_CTX_CONT_DIP_STRUCT_FIELDS
    )
    frame = frame.drop(
        columns=[name for name in group_a_required if name in frame.columns]
    )
    # The decision frame starts at feature_history_start, but history-sensitive
    # fields must use the same earlier canonical M5 prefix as the dataset.
    frame = _attach_ranker_group_a_with_common_history(
        frame,
        multi_tf=load_multi_tf_cache(mtf_cache_dir),
        workers=ATTACH_WORKERS,
        checkpoint_dir=checkpoint_dir,
        checkpoint_key=checkpoint_key,
        context_m5=context_m5,
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
        frame["is_ASIA"] = (
            frame["session_id"].astype(int) == ASIA_SESSION_ID
        ).astype(np.int8)
    missing_ctx = [n for n in MODEL_NATIVE_CTX_CONT_FIELDS if n not in frame.columns]
    if missing_ctx:
        raise RuntimeError(
            f"FEATURE_RANKER_CTX_INCOMPLETE_AFTER_DERIVATION: {missing_ctx[:10]} "
            f"total={len(missing_ctx)}"
        )

    missing_base = [f for f in MODEL_NATIVE_BASE_FIELDS if f not in frame.columns]
    if missing_base:
        raise RuntimeError(f"FEATURE_RANKER_BASE_FIELDS_MISSING: {missing_base}")
    source_ctx_cont = [n for n in MODEL_NATIVE_CTX_CONT_FIELDS if n in frame.columns]
    missing_cat = [n for n in MODEL_NATIVE_CTX_CAT_FIELDS if n not in frame.columns]
    if missing_cat:
        raise RuntimeError(f"FEATURE_RANKER_CTX_CAT_MISSING: {missing_cat}")
    return frame, source_ctx_cont


def _compute_candidate_matrix(
    frame: pd.DataFrame,
    *,
    candidates: Sequence[str],
    source_ctx_cont: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Compute candidate values via the dataset builder's one-truth extension."""
    from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
        _build_inline_seq_structure_extension,
    )

    inline_source_columns = ["time", "close", "atr", "open", "high", "low"]
    missing_inline = [name for name in inline_source_columns if name not in frame.columns]
    if missing_inline:
        raise RuntimeError(
            f"FEATURE_RANKER_INLINE_CAUSAL_SOURCE_MISSING: {missing_inline}"
        )
    with tempfile.NamedTemporaryFile(
        suffix="_train_feature_ranker_common_history.parquet", delete=False
    ) as temporary:
        inline_source_path = Path(temporary.name)
    try:
        frame[inline_source_columns].to_parquet(inline_source_path, index=False)
        matrix, names, _meta = _build_inline_seq_structure_extension(
            frame,
            requested_features=list(candidates),
            ctx_cont_names=list(source_ctx_cont),
            ctx_cat_names=list(MODEL_NATIVE_CTX_CAT_FIELDS),
            source_parquet=inline_source_path,
            source_contract_label="train_feature_ranker_common_causal_history_v2",
        )
    finally:
        inline_source_path.unlink(missing_ok=True)
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


def _direction_utility_margin_target(
    frame: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact spread-aware direction-utility margin, capped to TRAIN.

    This mirrors the dataset label contract.  It is target engineering for
    offline ranking only and is never a live direction rule.
    """
    bid = frame["bid_close"].astype(np.float64).to_numpy()
    ask = frame["ask_close"].astype(np.float64).to_numpy()
    if (
        not np.isfinite(bid).all()
        or not np.isfinite(ask).all()
        or (bid <= 0.0).any()
        or (ask <= 0.0).any()
        or (ask < bid).any()
    ):
        raise RuntimeError("FEATURE_RANKER_BID_ASK_INVALID")
    if len(frame) <= DIRECTION_HORIZON_BARS:
        raise RuntimeError("FEATURE_RANKER_DIRECTION_HORIZON_UNAVAILABLE")

    target = np.full(len(frame), np.nan, dtype=np.float64)
    usable = len(frame) - DIRECTION_HORIZON_BARS
    entry_bid = bid[:usable]
    entry_ask = ask[:usable]
    exit_bid = bid[DIRECTION_HORIZON_BARS:]
    exit_ask = ask[DIRECTION_HORIZON_BARS:]
    pnl_long = (exit_bid - entry_ask) / entry_ask * 1e4
    pnl_short = (entry_bid - exit_ask) / entry_bid * 1e4

    mfe_long = np.empty(usable, dtype=np.float64)
    mae_long = np.empty(usable, dtype=np.float64)
    mfe_short = np.empty(usable, dtype=np.float64)
    mae_short = np.empty(usable, dtype=np.float64)
    for offset in range(usable):
        window_bid = bid[offset : offset + PATH_HORIZON_BARS + 1]
        window_ask = ask[offset : offset + PATH_HORIZON_BARS + 1]
        mfe_long[offset] = (
            (float(np.max(window_bid)) - entry_ask[offset])
            / entry_ask[offset]
            * 1e4
        )
        mae_long[offset] = (
            (entry_ask[offset] - float(np.min(window_bid)))
            / entry_ask[offset]
            * 1e4
        )
        mfe_short[offset] = (
            (entry_bid[offset] - float(np.min(window_ask)))
            / entry_bid[offset]
            * 1e4
        )
        mae_short[offset] = (
            (float(np.max(window_ask)) - entry_bid[offset])
            / entry_bid[offset]
            * 1e4
        )

    long_utility = (
        pnl_long
        + UTILITY_MFE_WEIGHT * mfe_long
        - UTILITY_MAE_WEIGHT * mae_long
        + UTILITY_PATH_WEIGHT * (mfe_long - mae_long)
    )
    short_utility = (
        pnl_short
        + UTILITY_MFE_WEIGHT * mfe_short
        - UTILITY_MAE_WEIGHT * mae_short
        + UTILITY_PATH_WEIGHT * (mfe_short - mae_short)
    )
    target[:usable] = long_utility - short_utility
    times = frame["time"].to_numpy()
    in_fit = (frame["time"] >= train_start) & (frame["time"] <= train_end)
    target[~in_fit.to_numpy()] = np.nan
    full_horizon = np.arange(len(frame)) < usable
    if not np.isfinite(target[in_fit.to_numpy() & full_horizon]).all():
        raise RuntimeError("FEATURE_RANKER_DIRECTION_UTILITY_NONFINITE")
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
    out_path: Path,
    run_id: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    source_time_max: pd.Timestamp,
    target_time_max: pd.Timestamp,
    source_sha256: str,
    target_sha256: str,
    rank_reference: TrainRankReferenceV2,
    source_cascade: dict[str, Any],
    scores: Dict[str, float],
    created: datetime | None = None,
) -> Path:
    out_path = out_path.expanduser().resolve()
    match = _RANKING_OUTPUT_RE.fullmatch(out_path.name)
    if match is None or out_path.is_symlink():
        raise RuntimeError(f"FEATURE_RANKER_OUTPUT_IDENTITY_INVALID: {out_path}")
    stamp = match.group(1)
    stamp_format = "%Y%m%dT%H%M%SZ" if len(stamp) == 16 else "%Y%m%dT%H%M%S%fZ"
    filename_created = datetime.strptime(stamp, stamp_format).replace(tzinfo=timezone.utc)
    if created is not None and created != filename_created:
        raise RuntimeError(
            "FEATURE_RANKER_OUTPUT_TIMESTAMP_MISMATCH: "
            f"filename={filename_created.isoformat()} created={created.isoformat()}"
        )
    created = filename_created
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    payload: Dict[str, Any] = {
        "schema_version": TRAIN_FEATURE_RANKING_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "entry_run_id": run_id,
        "producer": TRAIN_FEATURE_RANKING_PRODUCER,
        "producer_version": TRAIN_FEATURE_RANKING_PRODUCER_VERSION,
        "fit_scope": "train_only",
        "train_start_utc": train_start.isoformat(),
        "train_end_utc": train_end.isoformat(),
        "source_time_max_utc": source_time_max.isoformat(),
        "target_time_max_utc": target_time_max.isoformat(),
        "source_sha256": source_sha256,
        "target_sha256": target_sha256,
        "target_contract": dict(TRAIN_FEATURE_RANKING_TARGET_CONTRACT),
        "source_cascade": dict(source_cascade),
        "rank_reference": {
            "path": str(rank_reference.path),
            "sha256": rank_reference.sha256,
            "sidecar_path": str(
                rank_reference.path.with_suffix(rank_reference.path.suffix + ".json")
            ),
            "sidecar_sha256": rank_reference.sidecar_sha256,
            "schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
            "entry_run_id": str(rank_reference.sidecar["entry_run_id"]),
            "source_parquet": str(rank_reference.sidecar["source_parquet"]),
            "source_parquet_sha256": str(
                rank_reference.sidecar["source_parquet_sha256"]
            ),
            "history_start_utc": parse_utc(
                rank_reference.sidecar["history_start_utc"],
                field="history_start_utc",
            ).isoformat(),
            "fit_start_utc": rank_reference.fit_start_utc.isoformat(),
            "fit_end_utc": rank_reference.fit_end_utc.isoformat(),
            "fit_row_count": rank_reference.fit_row_count,
            "fit_scope": "train_only",
            "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
        },
        "ranking_order": dict(TRAIN_FEATURE_RANKING_ORDER),
        "causality_contract": dict(TRAIN_FEATURE_CAUSALITY_CONTRACT),
        "ranked_features": [
            {"rank": index, "name": name, "score": float(score)}
            for index, (name, score) in enumerate(ranked, start=1)
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--canonical-v2-parquet", type=Path, required=True)
    parser.add_argument("--mtf-cache-dir", type=Path, required=True)
    parser.add_argument("--source-cascade-proof", type=Path, required=True)
    parser.add_argument("--expected-source-time-max", required=True)
    parser.add_argument("--rank-reference-npz", type=Path, required=True)
    parser.add_argument("--history-start", required=True)
    parser.add_argument("--train-start", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run_id = require_entry_run_id(args.run_id)
    history_start = _parse_utc_arg(args.history_start, field="history_start")
    train_start = _parse_utc_arg(args.train_start, field="train_start")
    train_end = _parse_utc_arg(args.train_end, field="train_end")
    if not history_start <= train_start < train_end:
        raise RuntimeError("FEATURE_RANKER_WINDOW_ORDER_INVALID")

    source_parquet = args.source_parquet.expanduser().resolve()
    if not source_parquet.is_file():
        raise RuntimeError(f"FEATURE_RANKER_SOURCE_MISSING: {source_parquet}")
    source_sha256 = _sha256_file(source_parquet)
    rank_reference = validate_train_rank_reference_lineage_v2(
        args.rank_reference_npz,
        expected_run_id=run_id,
        expected_source_parquet=source_parquet,
        expected_source_sha256=source_sha256,
        expected_history_start_utc=history_start,
        expected_fit_start_utc=train_start,
        expected_fit_end_utc=train_end,
    )
    mtf_cache_dir = args.mtf_cache_dir.expanduser().resolve()
    if not mtf_cache_dir.is_dir() or mtf_cache_dir.is_symlink():
        raise RuntimeError(f"FEATURE_RANKER_MTF_CACHE_MISSING: {mtf_cache_dir}")
    mtf_cache_sha256 = _mtf_cache_sha256(mtf_cache_dir)
    source_cascade = validate_seq513_source_cascade_proof(
        args.source_cascade_proof,
        expected_run_id=run_id,
        expected_source_parquet=source_parquet,
        expected_canonical_v2_parquet=args.canonical_v2_parquet,
        expected_mtf_cache_dir=mtf_cache_dir,
        expected_history_start_utc=history_start,
        expected_time_max_utc=args.expected_source_time_max,
    )
    checkpoint_key = _ranker_checkpoint_key(
        run_id=run_id,
        source_sha256=source_sha256,
        mtf_cache_sha256=mtf_cache_sha256,
        source_cascade_sha256=str(source_cascade["sha256"]),
        rank_reference_sha256=rank_reference.sha256,
        rank_reference_sidecar_sha256=rank_reference.sidecar_sha256,
        history_start=history_start,
        train_start=train_start,
        train_end=train_end,
    )

    # Checkpoint: attach + extension cost hours; a trivial late failure must
    # never force recomputation. Bound to run/source/cache and exact window.
    out_path = args.out.expanduser().resolve()
    if not _RANKING_OUTPUT_RE.fullmatch(out_path.name):
        raise RuntimeError(f"FEATURE_RANKER_OUTPUT_IDENTITY_INVALID: {out_path}")
    if out_path.exists() or out_path.is_symlink():
        raise RuntimeError(f"FEATURE_RANKER_OUTPUT_ALREADY_EXISTS: {out_path}")
    out_dir = out_path.parent
    checkpoint_path = out_dir / "_ranker_checkpoint.npz"
    group_a_checkpoint_dir = out_dir / "_ranker_group_a_checkpoint"
    if out_dir.exists():
        interrupted = sorted(
            path
            for path in out_dir.iterdir()
            if path.name.startswith(f".{checkpoint_path.name}.partial-")
        )
        if interrupted:
            raise RuntimeError(
                f"FEATURE_RANKER_CHECKPOINT_INTERRUPTED_REQUIRES_FRESH_EVENT: {interrupted}"
            )
    matrix = None
    if checkpoint_path.exists():
        try:
            with np.load(checkpoint_path, allow_pickle=False) as ck:
                expected_keys = {
                    "schema_version",
                    "checkpoint_key",
                    "matrix",
                    "names",
                    "times_ns",
                    "target",
                    "train_rows",
                    "source_time_max_ns",
                    "source_sha256",
                    "mtf_cache_sha256",
                    "source_cascade_sha256",
                    "rank_reference_sha256",
                    "rank_reference_sidecar_sha256",
                }
                if set(ck.files) != expected_keys:
                    raise RuntimeError("key set mismatch")
                if (
                    str(ck["schema_version"]) != RANKER_CHECKPOINT_SCHEMA_VERSION
                    or str(ck["checkpoint_key"]) != checkpoint_key
                    or str(ck["source_sha256"]) != source_sha256
                    or str(ck["mtf_cache_sha256"]) != mtf_cache_sha256
                    or str(ck["source_cascade_sha256"])
                    != source_cascade["sha256"]
                    or str(ck["rank_reference_sha256"])
                    != rank_reference.sha256
                    or str(ck["rank_reference_sidecar_sha256"])
                    != rank_reference.sidecar_sha256
                ):
                    raise RuntimeError("identity mismatch")
                matrix = ck["matrix"]
                names = [str(n) for n in ck["names"]]
                times = ck["times_ns"].view("datetime64[ns]")
                target = ck["target"]
                train_rows = int(ck["train_rows"])
                source_time_max = pd.Timestamp(int(ck["source_time_max_ns"]), tz="UTC")
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            raise RuntimeError(
                f"FEATURE_RANKER_CHECKPOINT_INVALID: {checkpoint_path}: {exc}"
            ) from exc
        print(f"[CHECKPOINT] gjenbrukt {checkpoint_path}", flush=True)

    if matrix is None:
        common_history_m5 = _load_ranker_common_history_m5(
            train_end=train_end,
            source_parquet=source_parquet,
            source_cascade=source_cascade,
        )
        frame, source_ctx_cont = _load_train_frame(
            source_parquet,
            history_start=history_start,
            train_end=train_end,
            mtf_cache_dir=mtf_cache_dir,
            checkpoint_dir=group_a_checkpoint_dir,
            checkpoint_key=checkpoint_key,
            rank_reference=rank_reference,
            context_m5=common_history_m5,
        )
        candidates = _candidate_universe(source_ctx_cont)
        if len(candidates) < MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT:
            raise RuntimeError(
                "FEATURE_RANKER_CANDIDATE_POOL_TOO_SMALL: "
                f"{len(candidates)} < {MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT}"
            )
        matrix, names = _compute_candidate_matrix(
            frame,
            candidates=candidates,
            source_ctx_cont=source_ctx_cont,
        )
        times, target = _direction_utility_margin_target(
            frame, train_start=train_start, train_end=train_end
        )
        train_rows = int(len(frame))
        _stm = pd.Timestamp(frame["time"].max())
        source_time_max = (
            _stm.tz_convert("UTC") if _stm.tzinfo else _stm.tz_localize("UTC")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_ranker_checkpoint(
            checkpoint_path,
            schema_version=np.array(RANKER_CHECKPOINT_SCHEMA_VERSION),
            checkpoint_key=np.array(checkpoint_key),
            matrix=np.asarray(matrix, dtype=np.float32),
            names=np.array(list(names)),
            times_ns=pd.DatetimeIndex(times).asi8,
            target=np.asarray(target, dtype=np.float64),
            train_rows=np.int64(train_rows),
            source_time_max_ns=np.int64(source_time_max.value),
            source_sha256=np.array(source_sha256),
            mtf_cache_sha256=np.array(mtf_cache_sha256),
            source_cascade_sha256=np.array(source_cascade["sha256"]),
            rank_reference_sha256=np.array(rank_reference.sha256),
            rank_reference_sidecar_sha256=np.array(
                rank_reference.sidecar_sha256
            ),
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
        out_path=out_path,
        run_id=run_id,
        train_start=train_start,
        train_end=train_end,
        source_time_max=source_time_max,
        target_time_max=target_time_max,
        source_sha256=source_sha256,
        target_sha256=target_sha256,
        rank_reference=rank_reference,
        source_cascade=source_cascade,
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
