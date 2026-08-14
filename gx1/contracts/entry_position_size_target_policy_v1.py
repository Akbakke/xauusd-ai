"""Immutable TRAIN-only empirical target policy for Entry position sizing.

Sizing supervision is defined only when the immutable Entry direction policy
selects one tradable side.  The target is the exact empirical CDF rank of that
side's realized ``MFE - MAE`` path on TRAIN.  VAL and TEST only apply the
frozen sorted TRAIN population; they cannot fit, extend or recenter it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_direction_target_policy_v1 import (
    entry_direction_targets_from_policy,
    require_entry_direction_target_policy,
)
from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_DECISION_BAR_SECONDS
from gx1.io.price_glitch_guard import assert_no_price_scale_glitch


ENTRY_POSITION_SIZE_TARGET_POLICY_SCHEMA_VERSION = (
    "gx1_entry_position_size_target_policy_v1"
)
ENTRY_POSITION_SIZE_TARGET_POLICY_FIT_METHOD = (
    "train_tradable_selected_side_path_quality_exact_ecdf_v1"
)
ENTRY_POSITION_SIZE_TARGET_MAPPING_METHOD = (
    "right_continuous_exact_train_ecdf_searchsorted_right_v1"
)
ENTRY_POSITION_SIZE_TARGET_EVIDENCE_FORMULA = (
    "selected_side_mfe_first_n_bps_minus_mae_first_n_bps"
)
ENTRY_POSITION_SIZE_TARGET_MASK_COLUMN = "y_position_size_mask"
ENTRY_POSITION_SIZE_TARGET_STORAGE_SENTINEL = 0.0

_POLICY_KEYS = {
    "schema_version",
    "decision",
    "fit_split",
    "fit_scope",
    "fit_method",
    "mapping_method",
    "evidence_formula",
    "target_domain",
    "target_population",
    "target_mask_column",
    "nontradable_target_defined",
    "nontradable_storage_sentinel",
    "unmasked_training_allowed",
    "m5_row_clock",
    "source_parquet_sha256",
    "tape_provenance_sha256",
    "entry_direction_target_policy_sha256",
    "train_start_utc",
    "train_end_utc",
    "path_horizon_bars",
    "fit_candidate_rows",
    "fit_population_rows",
    "fit_long_rows",
    "fit_short_rows",
    "selected_row_indices_sha256",
    "selected_side_stream_sha256",
    "selected_path_evidence_stream_sha256",
    "ecdf_artifact_format",
    "ecdf_artifact_path",
    "ecdf_artifact_sha256",
    "ecdf_artifact_array_sha256",
    "ecdf_artifact_dtype",
    "ecdf_artifact_rows",
    "future_outcomes_used_as_model_inputs",
    "val_test_rows_used_for_fit",
    "live_direction_authority",
    "policy_sha256",
}


def canonical_entry_position_size_target_policy_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f8").tobytes()
    ).hexdigest()


def _npy_artifact_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(
        buffer,
        np.ascontiguousarray(values, dtype="<f8"),
        allow_pickle=False,
    )
    return buffer.getvalue()


def _write_exact_ecdf_artifact(
    path: Path,
    *,
    sorted_evidence: np.ndarray,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix != ".npy" or not resolved.parent.is_dir():
        raise RuntimeError(
            "ENTRY_POSITION_SIZE_TARGET_POLICY_ECDF_ARTIFACT_PATH_INVALID"
        )
    artifact_bytes = _npy_artifact_bytes(sorted_evidence)
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if resolved.exists() or resolved.is_symlink():
        if (
            resolved.is_symlink()
            or not resolved.is_file()
            or _sha256_file(resolved) != artifact_sha256
        ):
            raise RuntimeError(
                "ENTRY_POSITION_SIZE_TARGET_POLICY_ECDF_ARTIFACT_COLLISION"
            )
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(resolved, flags, 0o644)
        try:
            view = memoryview(artifact_bytes)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError(f"short ECDF artifact write: {resolved}")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {
        "ecdf_artifact_format": "numpy_npy_v1_exact_float64",
        "ecdf_artifact_path": str(resolved),
        "ecdf_artifact_sha256": artifact_sha256,
        "ecdf_artifact_array_sha256": _array_sha256(sorted_evidence),
        "ecdf_artifact_dtype": "float64_le",
        "ecdf_artifact_rows": int(len(sorted_evidence)),
    }


def _load_exact_ecdf_artifact(policy: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(policy["ecdf_artifact_path"])).expanduser()
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != policy["ecdf_artifact_sha256"]
    ):
        raise RuntimeError(
            "ENTRY_POSITION_SIZE_TARGET_POLICY_ECDF_ARTIFACT_INVALID"
        )
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "ENTRY_POSITION_SIZE_TARGET_POLICY_ECDF_ARTIFACT_INVALID"
        ) from exc
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 1
        or values.dtype != np.dtype("<f8")
        or len(values) != policy["ecdf_artifact_rows"]
        or len(values) != policy["fit_population_rows"]
        or not np.isfinite(values).all()
        or np.unique(values).size < 2
        or np.any(np.diff(values) < 0.0)
        or _array_sha256(values) != policy["ecdf_artifact_array_sha256"]
    ):
        raise RuntimeError(
            "ENTRY_POSITION_SIZE_TARGET_POLICY_ECDF_ARTIFACT_INVALID"
        )
    return values


def _utc(value: Any, *, field: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError(
            f"ENTRY_POSITION_SIZE_TARGET_POLICY_{field}_INVALID"
        ) from exc
    if (
        pd.isna(parsed)
        or parsed.tz is None
        or parsed.utcoffset() != pd.Timedelta(0)
    ):
        raise RuntimeError(f"ENTRY_POSITION_SIZE_TARGET_POLICY_{field}_INVALID")
    return parsed.as_unit("ns")


def _selected_side_path_evidence(
    *,
    bid: np.ndarray,
    ask: np.ndarray,
    rows: np.ndarray,
    horizon_bars: int,
    direction_policy: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entry_bid = bid[rows]
    entry_ask = ask[rows]
    future_bid = bid[rows + horizon_bars]
    future_ask = ask[rows + horizon_bars]
    long_pnl = (future_bid / entry_ask - 1.0) * 10_000.0
    short_pnl = (entry_bid / future_ask - 1.0) * 10_000.0
    targets = entry_direction_targets_from_policy(
        policy=direction_policy,
        long_executable_pnl_bps=long_pnl,
        short_executable_pnl_bps=short_pnl,
    )
    selected_side = np.asarray(targets["side"], dtype=np.int8)
    trade = np.asarray(targets["trade"], dtype=bool)

    max_bid = entry_bid.copy()
    min_bid = entry_bid.copy()
    max_ask = entry_ask.copy()
    min_ask = entry_ask.copy()
    for step in range(1, horizon_bars + 1):
        step_bid = bid[rows + step]
        step_ask = ask[rows + step]
        np.maximum(max_bid, step_bid, out=max_bid)
        np.minimum(min_bid, step_bid, out=min_bid)
        np.maximum(max_ask, step_ask, out=max_ask)
        np.minimum(min_ask, step_ask, out=min_ask)

    mfe_long = (max_bid - entry_ask) / entry_ask * 10_000.0
    mae_long = (entry_ask - min_bid) / entry_ask * 10_000.0
    mfe_short = (entry_bid - min_ask) / entry_bid * 10_000.0
    mae_short = (max_ask - entry_bid) / entry_bid * 10_000.0
    selected_mfe = np.where(selected_side == 0, mfe_long, mfe_short)
    selected_mae = np.where(selected_side == 0, mae_long, mae_short)
    evidence = (selected_mfe - selected_mae).astype(np.float64, copy=False)
    if (
        not np.isfinite(evidence[trade]).all()
        or np.any(selected_mae[trade] < 0.0)
        or np.any(~np.isin(selected_side, (-1, 0, 1)))
        or not np.array_equal(trade, selected_side >= 0)
    ):
        raise RuntimeError(
            "ENTRY_POSITION_SIZE_TARGET_POLICY_PATH_EVIDENCE_INVALID"
        )
    return selected_side, trade, evidence


def fit_entry_position_size_target_policy(
    *,
    closed_m5: pd.DataFrame,
    entry_direction_target_policy: Mapping[str, Any],
    source_parquet_sha256: str,
    tape_provenance_sha256: str,
    ecdf_artifact_path: Path,
) -> dict[str, Any]:
    """Fit one exact selected-side path ECDF on native TRAIN M5 only."""

    direction_policy = require_entry_direction_target_policy(
        entry_direction_target_policy,
        expected_source_parquet_sha256=source_parquet_sha256,
        expected_tape_provenance_sha256=tape_provenance_sha256,
    )
    if not isinstance(closed_m5, pd.DataFrame) or not {
        "time",
        "bid_close",
        "ask_close",
    }.issubset(closed_m5.columns):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_M5_FRAME_INVALID")
    assert_no_price_scale_glitch(
        closed_m5,
        context="ENTRY_POSITION_SIZE_TARGET_POLICY_TRAIN_SOURCE",
    )
    times = pd.DatetimeIndex(
        pd.to_datetime(closed_m5["time"], utc=True, errors="coerce")
    ).as_unit("ns")
    if (
        len(times) == 0
        or times.hasnans
        or not times.is_unique
        or not times.is_monotonic_increasing
        or not times.floor(f"{ENTRY_DECISION_BAR_SECONDS}s").equals(times)
    ):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_M5_TIME_INVALID")
    bid = pd.to_numeric(closed_m5["bid_close"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    ask = pd.to_numeric(closed_m5["ask_close"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(bid).all()
        or not np.isfinite(ask).all()
        or np.any(bid <= 0.0)
        or np.any(ask <= bid)
    ):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_QUOTES_INVALID")

    first = int(direction_policy["fit_first_m5_row"])
    last = int(direction_policy["fit_last_m5_row"])
    horizon = int(direction_policy["path_quality_horizon_bars"])
    if first < 0 or last < first or last + horizon >= len(times):
        raise RuntimeError(
            "ENTRY_POSITION_SIZE_TARGET_POLICY_DIRECTION_POPULATION_INVALID"
        )
    rows = np.arange(first, last + 1, dtype=np.int64)
    selected_side, trade, evidence = _selected_side_path_evidence(
        bid=bid,
        ask=ask,
        rows=rows,
        horizon_bars=horizon,
        direction_policy=direction_policy,
    )
    selected_rows = rows[trade]
    selected_sides = selected_side[trade]
    selected_evidence = evidence[trade]
    if (
        len(selected_evidence) < 2
        or not np.isfinite(selected_evidence).all()
        or np.unique(selected_evidence).size < 2
        or not bool(np.any(selected_sides == 0))
        or not bool(np.any(selected_sides == 1))
    ):
        raise RuntimeError(
            "ENTRY_POSITION_SIZE_TARGET_POLICY_TRADABLE_POPULATION_INVALID"
        )
    sorted_evidence = np.sort(selected_evidence, kind="mergesort")

    row_stream = hashlib.sha256(
        np.ascontiguousarray(selected_rows, dtype="<i8").tobytes()
    ).hexdigest()
    side_stream = hashlib.sha256(
        np.ascontiguousarray(selected_sides, dtype="i1").tobytes()
    ).hexdigest()
    evidence_stream = hashlib.sha256(
        np.ascontiguousarray(selected_evidence, dtype="<f8").tobytes()
    ).hexdigest()
    artifact_binding = _write_exact_ecdf_artifact(
        ecdf_artifact_path,
        sorted_evidence=sorted_evidence,
    )
    policy: dict[str, Any] = {
        "schema_version": ENTRY_POSITION_SIZE_TARGET_POLICY_SCHEMA_VERSION,
        "decision": "PASS",
        "fit_split": "train",
        "fit_scope": "TRAIN_ONLY",
        "fit_method": ENTRY_POSITION_SIZE_TARGET_POLICY_FIT_METHOD,
        "mapping_method": ENTRY_POSITION_SIZE_TARGET_MAPPING_METHOD,
        "evidence_formula": ENTRY_POSITION_SIZE_TARGET_EVIDENCE_FORMULA,
        "target_domain": [0.0, 1.0],
        "target_population": "tradable_selected_side_rows_only",
        "target_mask_column": ENTRY_POSITION_SIZE_TARGET_MASK_COLUMN,
        "nontradable_target_defined": False,
        "nontradable_storage_sentinel": (
            ENTRY_POSITION_SIZE_TARGET_STORAGE_SENTINEL
        ),
        "unmasked_training_allowed": False,
        "m5_row_clock": "consecutive_authoritative_closed_m5_source_rows",
        "source_parquet_sha256": source_parquet_sha256,
        "tape_provenance_sha256": tape_provenance_sha256,
        "entry_direction_target_policy_sha256": direction_policy["policy_sha256"],
        "train_start_utc": direction_policy["train_start_utc"],
        "train_end_utc": direction_policy["train_end_utc"],
        "path_horizon_bars": horizon,
        "fit_candidate_rows": int(len(rows)),
        "fit_population_rows": int(len(selected_evidence)),
        "fit_long_rows": int(np.count_nonzero(selected_sides == 0)),
        "fit_short_rows": int(np.count_nonzero(selected_sides == 1)),
        "selected_row_indices_sha256": row_stream,
        "selected_side_stream_sha256": side_stream,
        "selected_path_evidence_stream_sha256": evidence_stream,
        **artifact_binding,
        "future_outcomes_used_as_model_inputs": False,
        "val_test_rows_used_for_fit": 0,
        "live_direction_authority": False,
    }
    policy["policy_sha256"] = canonical_entry_position_size_target_policy_sha256(
        policy
    )
    return require_entry_position_size_target_policy(
        policy,
        expected_source_parquet_sha256=source_parquet_sha256,
        expected_tape_provenance_sha256=tape_provenance_sha256,
        expected_direction_policy_sha256=direction_policy["policy_sha256"],
    )


def require_entry_position_size_target_policy(
    value: Any,
    *,
    expected_source_parquet_sha256: str | None = None,
    expected_tape_provenance_sha256: str | None = None,
    expected_direction_policy_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _POLICY_KEYS:
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_SCHEMA_INVALID")
    policy = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    declared_hash = policy.pop("policy_sha256", None)
    if (
        not _is_sha256(declared_hash)
        or declared_hash
        != canonical_entry_position_size_target_policy_sha256(policy)
    ):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_HASH_INVALID")
    policy["policy_sha256"] = declared_hash
    if (
        policy["schema_version"]
        != ENTRY_POSITION_SIZE_TARGET_POLICY_SCHEMA_VERSION
        or policy["decision"] != "PASS"
        or policy["fit_split"] != "train"
        or policy["fit_scope"] != "TRAIN_ONLY"
        or policy["fit_method"] != ENTRY_POSITION_SIZE_TARGET_POLICY_FIT_METHOD
        or policy["mapping_method"] != ENTRY_POSITION_SIZE_TARGET_MAPPING_METHOD
        or policy["evidence_formula"]
        != ENTRY_POSITION_SIZE_TARGET_EVIDENCE_FORMULA
        or policy["target_domain"] != [0.0, 1.0]
        or policy["target_population"] != "tradable_selected_side_rows_only"
        or policy["target_mask_column"]
        != ENTRY_POSITION_SIZE_TARGET_MASK_COLUMN
        or policy["nontradable_target_defined"] is not False
        or policy["nontradable_storage_sentinel"]
        != ENTRY_POSITION_SIZE_TARGET_STORAGE_SENTINEL
        or policy["unmasked_training_allowed"] is not False
        or policy["m5_row_clock"]
        != "consecutive_authoritative_closed_m5_source_rows"
        or policy["future_outcomes_used_as_model_inputs"] is not False
        or policy["val_test_rows_used_for_fit"] != 0
        or policy["live_direction_authority"] is not False
    ):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_CONTRACT_INVALID")
    for field, expected in (
        ("source_parquet_sha256", expected_source_parquet_sha256),
        ("tape_provenance_sha256", expected_tape_provenance_sha256),
        ("entry_direction_target_policy_sha256", expected_direction_policy_sha256),
    ):
        if not _is_sha256(policy[field]):
            raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_LINEAGE_INVALID")
        if expected is not None and policy[field] != expected:
            raise RuntimeError(
                f"ENTRY_POSITION_SIZE_TARGET_POLICY_{field.upper()}_MISMATCH"
            )
    start = _utc(policy["train_start_utc"], field="TRAIN_START")
    end = _utc(policy["train_end_utc"], field="TRAIN_END")
    if end <= start:
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_TRAIN_RANGE_INVALID")
    integer_fields = (
        "path_horizon_bars",
        "fit_candidate_rows",
        "fit_population_rows",
        "fit_long_rows",
        "fit_short_rows",
    )
    if any(
        isinstance(policy[field], bool)
        or not isinstance(policy[field], int)
        or int(policy[field]) <= 0
        for field in integer_fields
    ):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_COUNTS_INVALID")
    if (
        policy["fit_population_rows"] > policy["fit_candidate_rows"]
        or policy["fit_long_rows"] + policy["fit_short_rows"]
        != policy["fit_population_rows"]
        or any(
            not _is_sha256(policy[field])
            for field in (
                "selected_row_indices_sha256",
                "selected_side_stream_sha256",
                "selected_path_evidence_stream_sha256",
            )
        )
    ):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_POPULATION_INVALID")
    if (
        policy["ecdf_artifact_format"] != "numpy_npy_v1_exact_float64"
        or policy["ecdf_artifact_dtype"] != "float64_le"
        or not isinstance(policy["ecdf_artifact_path"], str)
        or not policy["ecdf_artifact_path"]
        or not _is_sha256(policy["ecdf_artifact_sha256"])
        or not _is_sha256(policy["ecdf_artifact_array_sha256"])
        or type(policy["ecdf_artifact_rows"]) is not int
        or policy["ecdf_artifact_rows"] != policy["fit_population_rows"]
    ):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_ECDF_INVALID")
    _load_exact_ecdf_artifact(policy)
    return policy


def entry_position_size_targets_from_policy(
    *,
    policy: Mapping[str, Any],
    mfe_first_n_bps: Any,
    mae_first_n_bps: Any,
    selected_side: Any,
    trade_mask: Any,
) -> dict[str, np.ndarray]:
    """Apply the frozen TRAIN ECDF; no fit or direction decision occurs here."""

    normalized = require_entry_position_size_target_policy(policy)
    mfe = np.asarray(mfe_first_n_bps, dtype=np.float64)
    mae = np.asarray(mae_first_n_bps, dtype=np.float64)
    side = np.asarray(selected_side)
    mask = np.asarray(trade_mask)
    if not (mfe.shape == mae.shape == side.shape == mask.shape) or mfe.ndim != 1:
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_INPUT_SHAPE_INVALID")
    if (
        not np.isfinite(mfe).all()
        or not np.isfinite(mae).all()
        or np.any(mae < 0.0)
        or not np.isfinite(mask.astype(np.float64)).all()
        or np.any(~np.isin(mask, (0, 1, False, True)))
        or np.any(~np.isin(side, (-1, 0, 1)))
    ):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_INPUT_INVALID")
    active = mask.astype(bool)
    if not np.array_equal(active, side >= 0):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_SIDE_MASK_MISMATCH")
    evidence = (mfe - mae).astype(np.float64, copy=False)
    knots = _load_exact_ecdf_artifact(normalized)
    target = np.full(
        mfe.shape,
        float(normalized["nontradable_storage_sentinel"]),
        dtype=np.float64,
    )
    target[active] = (
        np.searchsorted(knots, evidence[active], side="right") / float(len(knots))
    )
    if (
        not np.isfinite(target).all()
        or np.any(target < 0.0)
        or np.any(target > 1.0)
    ):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_OUTPUT_INVALID")
    return {
        "target": target.astype(np.float32),
        "mask": active.astype(np.float32),
        "path_evidence_bps": evidence.astype(np.float32),
    }


def entry_position_size_target_policy_contract(
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = require_entry_position_size_target_policy(policy)
    return {
        "entry_position_size_target_policy": normalized,
        "entry_position_size_target_policy_sha256": normalized["policy_sha256"],
        "position_size_target_source": "train_fitted_selected_side_path_ecdf",
        "position_size_target_mask": normalized["target_mask_column"],
        "position_size_target_unmasked_training_allowed": False,
        "position_size_target_live_direction_authority": False,
    }


def require_entry_position_size_target_manifest_binding(
    extra: Any,
    *,
    expected_source_parquet_sha256: str | None = None,
    expected_tape_provenance_sha256: str | None = None,
    expected_direction_policy_sha256: str | None = None,
    expected_train_start: Any | None = None,
    expected_train_end: Any | None = None,
) -> dict[str, Any]:
    """Validate the complete frozen sizing-policy projection in a manifest."""

    if not isinstance(extra, Mapping):
        raise RuntimeError("ENTRY_POSITION_SIZE_TARGET_POLICY_MANIFEST_EXTRA_INVALID")
    policy = require_entry_position_size_target_policy(
        extra.get("entry_position_size_target_policy"),
        expected_source_parquet_sha256=expected_source_parquet_sha256,
        expected_tape_provenance_sha256=expected_tape_provenance_sha256,
        expected_direction_policy_sha256=expected_direction_policy_sha256,
    )
    expected_projection = entry_position_size_target_policy_contract(policy)
    for key, expected in expected_projection.items():
        if extra.get(key) != expected:
            raise RuntimeError(
                "ENTRY_POSITION_SIZE_TARGET_POLICY_MANIFEST_PROJECTION_MISMATCH: "
                f"key={key}"
            )
    if expected_train_start is not None and _utc(
        expected_train_start, field="EXPECTED_TRAIN_START"
    ) != _utc(policy["train_start_utc"], field="TRAIN_START"):
        raise RuntimeError(
            "ENTRY_POSITION_SIZE_TARGET_POLICY_MANIFEST_TRAIN_START_MISMATCH"
        )
    if expected_train_end is not None and _utc(
        expected_train_end, field="EXPECTED_TRAIN_END"
    ) != _utc(policy["train_end_utc"], field="TRAIN_END"):
        raise RuntimeError(
            "ENTRY_POSITION_SIZE_TARGET_POLICY_MANIFEST_TRAIN_END_MISMATCH"
        )
    return policy


__all__ = [
    "ENTRY_POSITION_SIZE_TARGET_EVIDENCE_FORMULA",
    "ENTRY_POSITION_SIZE_TARGET_MAPPING_METHOD",
    "ENTRY_POSITION_SIZE_TARGET_MASK_COLUMN",
    "ENTRY_POSITION_SIZE_TARGET_POLICY_FIT_METHOD",
    "ENTRY_POSITION_SIZE_TARGET_POLICY_SCHEMA_VERSION",
    "ENTRY_POSITION_SIZE_TARGET_STORAGE_SENTINEL",
    "canonical_entry_position_size_target_policy_sha256",
    "entry_position_size_target_policy_contract",
    "entry_position_size_targets_from_policy",
    "fit_entry_position_size_target_policy",
    "require_entry_position_size_target_manifest_binding",
    "require_entry_position_size_target_policy",
]
