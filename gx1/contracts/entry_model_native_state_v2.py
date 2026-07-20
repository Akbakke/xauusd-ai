"""Immutable TRAIN-only normalization and common-history state contract.

The model-native Entry path may fit percentile/rank transforms on the declared
TRAIN interval only.  Validation and test rows are never stored in the rank
artifact and never participate in its distributions.  Every offline split and
the serving adapter start causal feature construction at one immutable history
anchor; split-local feature resets are forbidden.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from gx1.features.model_native_market_context_v1 import derive_observed_spread_bps


MODEL_NATIVE_STATE_SCHEMA_VERSION = "model_native_state_contract_v4"
MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION = "model_native_train_rank_reference_v4"
MODEL_NATIVE_RANK_TRANSFORM = "train_fit_right_ecdf_quintile_v1"
MODEL_NATIVE_HISTORY_MODE = "common_causal_history_no_split_reset_v1"

TRAIN_RANK_NPZ_KEYS = frozenset(
    {
        "schema_version",
        "fit_start_ns",
        "fit_end_ns",
        "fit_row_count",
        "entry_run_id",
        "atr_bps_sorted",
        "spread_bps_sorted",
    }
)
FORBIDDEN_ROW_LEVEL_RANK_KEYS = frozenset(
    {"time_ns", "vol_regime_id", "atr_bucket", "spread_bucket", "atr_pinned"}
)
STALE_STATE_CONTRACT_FIELDS = frozenset(
    {"frame_anchor_utc", "model_range_start_utc", "rank_reference_end_utc", "time_split_reference_split"}
)
STALE_ARTIFACT_MARKERS = ("julyext", "smart_candidate_20260630", "utilityrepair", "20260710")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(value: Any, *, field: str) -> pd.Timestamp:
    try:
        ts = pd.to_datetime(value, utc=True, errors="raise")
    except Exception as exc:
        raise RuntimeError(f"MODEL_NATIVE_STATE_TIMESTAMP_INVALID: {field}={value!r}") from exc
    if pd.isna(ts):
        raise RuntimeError(f"MODEL_NATIVE_STATE_TIMESTAMP_INVALID: {field}={value!r}")
    return pd.Timestamp(ts)


def compute_causal_market_rank_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive exact past-only ATR/spread inputs without source substitutions."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError("MODEL_NATIVE_RANK_SOURCE_EMPTY")
    required = ("high", "low", "close", "bid_close", "ask_close")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"MODEL_NATIVE_RANK_SOURCE_FIELDS_MISSING: {missing}")

    numeric: dict[str, np.ndarray] = {}
    for name in required:
        try:
            values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(f"MODEL_NATIVE_RANK_SOURCE_COLUMN_INVALID: {name}") from exc
        if not np.isfinite(values).all():
            raise RuntimeError(f"MODEL_NATIVE_RANK_SOURCE_NONFINITE: {name}")
        numeric[name] = values

    high = numeric["high"]
    low = numeric["low"]
    close = numeric["close"]
    invalid_ohlc = (
        (high <= 0.0)
        | (low <= 0.0)
        | (close <= 0.0)
        | (high < low)
        | (high < close)
        | (low > close)
    )
    if invalid_ohlc.any():
        raise RuntimeError(
            "MODEL_NATIVE_RANK_SOURCE_OHLC_INVALID: "
            f"count={int(np.count_nonzero(invalid_ohlc))}"
        )

    high_s = pd.Series(high, dtype=np.float64)
    low_s = pd.Series(low, dtype=np.float64)
    close_s = pd.Series(close, dtype=np.float64)
    tr = pd.concat(
        [
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s - close_s.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().to_numpy(dtype=np.float64)
    mid_hl = (high + low) * 0.5
    atr_bps = atr / mid_hl * 1e4

    # Never accept an already-materialized spread_bps column here.  Train and
    # serve both derive from the exact observed bid/ask closes.
    spread_frame = frame[["bid_close", "ask_close"]].copy()
    spread_bps = np.asarray(derive_observed_spread_bps(spread_frame), dtype=np.float64)
    if not np.isfinite(atr).all() or not np.isfinite(atr_bps).all() or not np.isfinite(spread_bps).all():
        raise RuntimeError("MODEL_NATIVE_RANK_INPUT_NONFINITE")
    if np.any(atr <= 0.0) or np.any(atr_bps <= 0.0) or np.any(spread_bps < 0.0):
        raise RuntimeError("MODEL_NATIVE_RANK_INPUT_OUT_OF_RANGE")
    return pd.DataFrame(
        {"atr": atr, "atr_bps": atr_bps, "spread_bps": spread_bps},
        index=frame.index,
    )


def bucket_against_train_reference(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(sorted_reference, dtype=np.float64)
    if values.ndim != 1 or reference.ndim != 1 or reference.size == 0:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_ARRAY_SHAPE_INVALID")
    if not np.isfinite(values).all() or not np.isfinite(reference).all():
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_ARRAY_NONFINITE")
    if reference.size > 1 and np.any(reference[1:] < reference[:-1]):
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_REFERENCE_NOT_SORTED")
    percentile = np.searchsorted(reference, values, side="right") / float(reference.size)
    return np.clip(percentile * 5.0, 0.0, 4.99).astype(np.int64)


@dataclass(frozen=True)
class TrainRankReferenceV2:
    path: Path
    sha256: str
    sidecar: dict[str, Any]
    fit_start_utc: pd.Timestamp
    fit_end_utc: pd.Timestamp
    fit_row_count: int
    atr_bps_sorted: np.ndarray
    spread_bps_sorted: np.ndarray


def load_train_rank_reference_v2(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> TrainRankReferenceV2:
    rank_path = Path(path).expanduser().resolve()
    if not rank_path.is_file():
        raise RuntimeError(f"MODEL_NATIVE_TRAIN_RANK_REFERENCE_MISSING: {rank_path}")
    actual_sha = sha256_file(rank_path)
    if expected_sha256 is not None and actual_sha != str(expected_sha256).strip().lower():
        raise RuntimeError(
            "MODEL_NATIVE_TRAIN_RANK_REFERENCE_SHA_MISMATCH: "
            f"expected={expected_sha256!r} actual={actual_sha}"
        )
    sidecar_path = rank_path.with_suffix(rank_path.suffix + ".json")
    if not sidecar_path.is_file():
        raise RuntimeError(f"MODEL_NATIVE_TRAIN_RANK_SIDECAR_MISSING: {sidecar_path}")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"MODEL_NATIVE_TRAIN_RANK_SIDECAR_INVALID: {sidecar_path}") from exc
    if not isinstance(sidecar, dict):
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_SIDECAR_ROOT_INVALID")
    if sidecar.get("schema_version") != MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION:
        raise RuntimeError(
            "MODEL_NATIVE_TRAIN_RANK_SCHEMA_INVALID: "
            f"got={sidecar.get('schema_version')!r}"
        )
    if sidecar.get("fit_scope") != "train_only":
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_FIT_SCOPE_INVALID")
    if sidecar.get("row_level_state_present") is not False:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_ROW_LEVEL_STATE_FORBIDDEN")
    if sidecar.get("rank_transform") != MODEL_NATIVE_RANK_TRANSFORM:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_TRANSFORM_INVALID")
    entry_run_id = str(sidecar.get("entry_run_id") or "").strip()
    if not entry_run_id:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_EXPLICIT_RUN_ID_MISSING")
    declared_path = Path(str(sidecar.get("out_npz") or "")).expanduser().resolve()
    if declared_path != rank_path:
        raise RuntimeError(
            f"MODEL_NATIVE_TRAIN_RANK_PATH_MISMATCH: sidecar={declared_path} actual={rank_path}"
        )
    if str(sidecar.get("out_npz_sha256") or "").strip().lower() != actual_sha:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_SIDECAR_SHA_MISMATCH")

    fit_start = parse_utc(sidecar.get("fit_start_utc"), field="fit_start_utc")
    fit_end = parse_utc(sidecar.get("fit_end_utc"), field="fit_end_utc")
    if fit_end < fit_start:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_FIT_WINDOW_INVALID")
    fit_row_count = int(sidecar.get("fit_row_count") or 0)
    if fit_row_count <= 0:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_ROW_COUNT_INVALID")

    try:
        with np.load(rank_path, allow_pickle=False) as payload:
            keys = frozenset(payload.files)
            if keys != TRAIN_RANK_NPZ_KEYS:
                forbidden = sorted(keys & FORBIDDEN_ROW_LEVEL_RANK_KEYS)
                raise RuntimeError(
                    "MODEL_NATIVE_TRAIN_RANK_KEYS_INVALID: "
                    f"missing={sorted(TRAIN_RANK_NPZ_KEYS - keys)} "
                    f"extra={sorted(keys - TRAIN_RANK_NPZ_KEYS)} forbidden={forbidden}"
                )
            schema_value = np.asarray(payload["schema_version"])
            if schema_value.shape != (1,) or str(schema_value[0]) != MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION:
                raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_NPZ_SCHEMA_INVALID")
            npz_start = int(np.asarray(payload["fit_start_ns"]).reshape(-1)[0])
            npz_end = int(np.asarray(payload["fit_end_ns"]).reshape(-1)[0])
            npz_rows = int(np.asarray(payload["fit_row_count"]).reshape(-1)[0])
            npz_run_id = np.asarray(payload["entry_run_id"])
            atr_sorted = np.asarray(payload["atr_bps_sorted"], dtype=np.float64).copy()
            spread_sorted = np.asarray(payload["spread_bps_sorted"], dtype=np.float64).copy()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"MODEL_NATIVE_TRAIN_RANK_NPZ_INVALID: {rank_path}") from exc
    if npz_start != fit_start.value or npz_end != fit_end.value or npz_rows != fit_row_count:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_NPZ_SIDECAR_METADATA_MISMATCH")
    if npz_run_id.shape != (1,) or str(npz_run_id[0]).strip() != entry_run_id:
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_NPZ_RUN_ID_MISMATCH")
    if atr_sorted.shape != (fit_row_count,) or spread_sorted.shape != (fit_row_count,):
        raise RuntimeError("MODEL_NATIVE_TRAIN_RANK_DISTRIBUTION_LENGTH_INVALID")
    # Reuse the bucket validator for finite/sorted proof without mutating data.
    bucket_against_train_reference(np.asarray([atr_sorted[0]]), atr_sorted)
    bucket_against_train_reference(np.asarray([spread_sorted[0]]), spread_sorted)
    return TrainRankReferenceV2(
        path=rank_path,
        sha256=actual_sha,
        sidecar=sidecar,
        fit_start_utc=fit_start,
        fit_end_utc=fit_end,
        fit_row_count=fit_row_count,
        atr_bps_sorted=atr_sorted,
        spread_bps_sorted=spread_sorted,
    )


def apply_train_rank_reference_v2(
    frame: pd.DataFrame,
    reference: TrainRankReferenceV2,
) -> pd.DataFrame:
    derived = compute_causal_market_rank_inputs(frame)
    out = frame.copy()
    for name in ("atr", "atr_bps", "spread_bps"):
        out[name] = derived[name].to_numpy(dtype=np.float64)
    vol = bucket_against_train_reference(
        derived["atr_bps"].to_numpy(dtype=np.float64), reference.atr_bps_sorted
    )
    spread = bucket_against_train_reference(
        derived["spread_bps"].to_numpy(dtype=np.float64), reference.spread_bps_sorted
    )
    out["vol_regime_id"] = vol
    out["atr_bucket"] = vol
    out["spread_bucket"] = spread
    return out


def validate_state_contract_metadata_v2(
    raw: Mapping[str, Any],
    *,
    require_artifact: bool = True,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise RuntimeError("MODEL_NATIVE_STATE_CONTRACT_MISSING")
    data = dict(raw)
    if data.get("schema_version") != MODEL_NATIVE_STATE_SCHEMA_VERSION:
        raise RuntimeError(
            "MODEL_NATIVE_STATE_SCHEMA_INVALID: "
            f"got={data.get('schema_version')!r} expected={MODEL_NATIVE_STATE_SCHEMA_VERSION!r}"
        )
    stale = sorted(STALE_STATE_CONTRACT_FIELDS & set(data))
    if stale:
        raise RuntimeError(f"MODEL_NATIVE_STATE_STALE_FIELDS_FORBIDDEN: {stale}")
    required = {
        "feature_history_start_utc",
        "rank_fit_start_utc",
        "rank_fit_end_utc",
        "rank_reference_npz",
        "rank_reference_npz_sha256",
        "rank_reference_schema_version",
        "normalization_fit_scope",
        "rank_transform",
        "feature_history_mode",
        "split_reset_allowed",
        "post_fit_rows_in_rank_reference",
        "runtime_rule_free",
        "entry_run_id",
    }
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"MODEL_NATIVE_STATE_FIELDS_MISSING: {missing}")
    if data["rank_reference_schema_version"] != MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION:
        raise RuntimeError("MODEL_NATIVE_STATE_RANK_SCHEMA_INVALID")
    if data["normalization_fit_scope"] != "train_only":
        raise RuntimeError("MODEL_NATIVE_STATE_FIT_SCOPE_INVALID")
    if data["rank_transform"] != MODEL_NATIVE_RANK_TRANSFORM:
        raise RuntimeError("MODEL_NATIVE_STATE_RANK_TRANSFORM_INVALID")
    if data["feature_history_mode"] != MODEL_NATIVE_HISTORY_MODE:
        raise RuntimeError("MODEL_NATIVE_STATE_HISTORY_MODE_INVALID")
    if data["split_reset_allowed"] is not False:
        raise RuntimeError("MODEL_NATIVE_STATE_SPLIT_RESET_FORBIDDEN")
    if data["post_fit_rows_in_rank_reference"] is not False:
        raise RuntimeError("MODEL_NATIVE_STATE_POST_FIT_ROWS_FORBIDDEN")
    if data["runtime_rule_free"] is not True:
        raise RuntimeError("MODEL_NATIVE_STATE_RUNTIME_RULE_FREE_REQUIRED")
    entry_run_id = str(data["entry_run_id"] or "").strip()
    if not entry_run_id:
        raise RuntimeError("MODEL_NATIVE_STATE_EXPLICIT_RUN_ID_MISSING")

    history_start = parse_utc(data["feature_history_start_utc"], field="feature_history_start_utc")
    fit_start = parse_utc(data["rank_fit_start_utc"], field="rank_fit_start_utc")
    fit_end = parse_utc(data["rank_fit_end_utc"], field="rank_fit_end_utc")
    if not history_start <= fit_start <= fit_end:
        raise RuntimeError("MODEL_NATIVE_STATE_TIME_ORDER_INVALID")
    sha = str(data["rank_reference_npz_sha256"]).strip().lower()
    if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise RuntimeError("MODEL_NATIVE_STATE_RANK_SHA_INVALID")
    rank_path = Path(str(data["rank_reference_npz"] or "")).expanduser()
    low = str(rank_path).lower()
    for marker in STALE_ARTIFACT_MARKERS:
        if marker in low:
            raise RuntimeError(f"MODEL_NATIVE_STATE_STALE_RANK_ARTIFACT: marker={marker!r}")
    if require_artifact:
        reference = load_train_rank_reference_v2(rank_path, expected_sha256=sha)
        if reference.fit_start_utc != fit_start or reference.fit_end_utc != fit_end:
            raise RuntimeError("MODEL_NATIVE_STATE_RANK_FIT_WINDOW_MISMATCH")
        if str(reference.sidecar.get("entry_run_id") or "").strip() != entry_run_id:
            raise RuntimeError("MODEL_NATIVE_STATE_RANK_RUN_ID_MISMATCH")
    return data
