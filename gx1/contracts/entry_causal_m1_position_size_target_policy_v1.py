"""Exact M1-path TRAIN ECDF for Entry position-size supervision.

The sizing auxiliary is meaningful only for the frozen, M1-fill direction
policy's selected side.  Its evidence is that side's exact M1 MFE minus MAE;
there is no M5-close fallback and no VAL/TEST population in the ECDF.
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

from gx1.contracts.entry_causal_m1_target_policy_v1 import (
    causal_m1_direction_targets_from_policy,
    materialize_causal_m1_auxiliary_outcomes,
    require_causal_m1_target_policy,
)


ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_SCHEMA_VERSION = "gx1_entry_causal_m1_position_size_target_policy_v1"
ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_FIT_METHOD = "train_exact_m1_selected_side_path_quality_exact_ecdf_v1"
ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_MAPPING_METHOD = "right_continuous_exact_train_ecdf_searchsorted_right_v1"
ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_EVIDENCE_FORMULA = "exact_m1_selected_side_mfe_bps_minus_mae_bps"
ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_MASK_COLUMN = "y_position_size_mask"
ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_STORAGE_SENTINEL = 0.0

_POLICY_KEYS = {
    "schema_version", "decision", "fit_split", "fit_scope", "fit_method",
    "mapping_method", "evidence_formula", "target_domain", "target_population",
    "target_mask_column", "nontradable_target_defined", "nontradable_storage_sentinel",
    "unmasked_training_allowed", "m5_decision_clock", "source_parquet_sha256",
    "tape_provenance_sha256", "m1_source_sha256", "entry_causal_m1_target_policy_sha256",
    "train_start_utc", "train_end_utc", "path_horizon_bars", "fit_candidate_rows",
    "fit_population_rows", "fit_long_rows", "fit_short_rows",
    "selected_row_indices_sha256", "selected_side_stream_sha256",
    "selected_path_evidence_stream_sha256", "ecdf_artifact_format",
    "ecdf_artifact_path", "ecdf_artifact_sha256", "ecdf_artifact_array_sha256",
    "ecdf_artifact_dtype", "ecdf_artifact_rows", "future_outcomes_used_as_model_inputs",
    "val_test_rows_used_for_fit", "live_direction_authority", "policy_sha256",
}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


def _utc(value: Any, *, field: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError(f"ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_{field}_INVALID") from exc
    if pd.isna(parsed) or parsed.tz is None or parsed.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(f"ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_{field}_INVALID")
    return parsed.as_unit("ns")


def canonical_causal_m1_position_size_target_policy_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _npy_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(values, dtype="<f8"), allow_pickle=False)
    return buffer.getvalue()


def _write_ecdf(path: Path, values: np.ndarray) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.suffix != ".npy" or not resolved.parent.is_dir():
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_ECDF_PATH_INVALID")
    encoded = _npy_bytes(values)
    file_hash = hashlib.sha256(encoded).hexdigest()
    if resolved.exists() or resolved.is_symlink():
        if resolved.is_symlink() or not resolved.is_file() or _sha256_file(resolved) != file_hash:
            raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_ECDF_COLLISION")
    else:
        fd = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError(f"short ECDF write: {resolved}")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
    return {
        "ecdf_artifact_format": "numpy_npy_v1_exact_float64",
        "ecdf_artifact_path": str(resolved),
        "ecdf_artifact_sha256": file_hash,
        "ecdf_artifact_array_sha256": _array_sha256(values),
        "ecdf_artifact_dtype": "float64_le",
        "ecdf_artifact_rows": int(len(values)),
    }


def _load_ecdf(policy: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(policy["ecdf_artifact_path"])).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file() or _sha256_file(path) != policy["ecdf_artifact_sha256"]:
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_ECDF_INVALID")
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_ECDF_INVALID") from exc
    if (
        not isinstance(values, np.ndarray) or values.ndim != 1 or values.dtype != np.dtype("<f8")
        or len(values) != policy["ecdf_artifact_rows"] or len(values) != policy["fit_population_rows"]
        or not np.isfinite(values).all() or np.unique(values).size < 2
        or np.any(np.diff(values) < 0.0) or _array_sha256(values) != policy["ecdf_artifact_array_sha256"]
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_ECDF_INVALID")
    return values


def fit_causal_m1_position_size_target_policy(
    *,
    closed_m5: pd.DataFrame,
    closed_m1: pd.DataFrame,
    entry_causal_m1_target_policy: Mapping[str, Any],
    source_parquet_sha256: str,
    tape_provenance_sha256: str,
    m1_source_sha256: str,
    ecdf_artifact_path: Path,
) -> dict[str, Any]:
    """Fit the one frozen ECDF from selected-side exact M1 TRAIN paths."""

    direction = require_causal_m1_target_policy(
        entry_causal_m1_target_policy,
        expected_source_parquet_sha256=source_parquet_sha256,
        expected_tape_provenance_sha256=tape_provenance_sha256,
        expected_m1_source_sha256=m1_source_sha256,
    )
    if not isinstance(closed_m5, pd.DataFrame) or "time" not in closed_m5.columns:
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_M5_FRAME_INVALID")
    outcomes = materialize_causal_m1_auxiliary_outcomes(
        policy=direction, m5_decision_times=closed_m5["time"], closed_m1=closed_m1
    )
    start = _utc(direction["train_start_utc"], field="TRAIN_START")
    end = _utc(direction["train_end_utc"], field="TRAIN_END")
    valid = outcomes["outcome_valid"].to_numpy(dtype=bool)
    in_train = (outcomes["entry_decision_at"] >= start).to_numpy() & (
        outcomes["exit_decision_at"].notna() & (outcomes["exit_decision_at"] <= end)
    ).to_numpy()
    candidate = valid & in_train
    rows = np.flatnonzero(candidate)
    if len(rows) < 2:
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_TRAIN_POPULATION_EMPTY")
    long_pnl = outcomes["v11_pnl_long_at_dir_horizon_bps"].to_numpy(dtype=np.float64)[rows]
    short_pnl = outcomes["v11_pnl_short_at_dir_horizon_bps"].to_numpy(dtype=np.float64)[rows]
    direction_target = causal_m1_direction_targets_from_policy(
        policy=direction, long_executable_pnl_bps=long_pnl, short_executable_pnl_bps=short_pnl
    )
    side = direction_target["side"]
    trade = direction_target["trade"]
    long_mfe = outcomes["mfe_long_first_n_bps"].to_numpy(dtype=np.float64)[rows]
    long_mae = outcomes["mae_long_first_n_bps"].to_numpy(dtype=np.float64)[rows]
    short_mfe = outcomes["mfe_short_first_n_bps"].to_numpy(dtype=np.float64)[rows]
    short_mae = outcomes["mae_short_first_n_bps"].to_numpy(dtype=np.float64)[rows]
    selected_mfe = np.where(side == 0, long_mfe, short_mfe)
    selected_mae = np.where(side == 0, long_mae, short_mae)
    evidence = selected_mfe - selected_mae
    selected_rows = rows[trade]
    selected_side = side[trade]
    selected_evidence = evidence[trade]
    if (
        len(selected_evidence) < 2 or not np.isfinite(selected_evidence).all()
        or np.unique(selected_evidence).size < 2 or not bool(np.any(selected_side == 0))
        or not bool(np.any(selected_side == 1))
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_TRADABLE_POPULATION_INVALID")
    sorted_evidence = np.sort(selected_evidence, kind="mergesort")
    artifact = _write_ecdf(ecdf_artifact_path, sorted_evidence)
    policy: dict[str, Any] = {
        "schema_version": ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_SCHEMA_VERSION,
        "decision": "PASS", "fit_split": "train", "fit_scope": "TRAIN_ONLY",
        "fit_method": ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_FIT_METHOD,
        "mapping_method": ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_MAPPING_METHOD,
        "evidence_formula": ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_EVIDENCE_FORMULA,
        "target_domain": [0.0, 1.0], "target_population": "tradable_selected_side_rows_only",
        "target_mask_column": ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_MASK_COLUMN,
        "nontradable_target_defined": False,
        "nontradable_storage_sentinel": ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_STORAGE_SENTINEL,
        "unmasked_training_allowed": False,
        "m5_decision_clock": "authoritative_closed_m5_bar_then_exact_m1_fill",
        "source_parquet_sha256": source_parquet_sha256,
        "tape_provenance_sha256": tape_provenance_sha256, "m1_source_sha256": m1_source_sha256,
        "entry_causal_m1_target_policy_sha256": direction["policy_sha256"],
        "train_start_utc": direction["train_start_utc"], "train_end_utc": direction["train_end_utc"],
        "path_horizon_bars": int(direction["path_quality_horizon_bars"]),
        "fit_candidate_rows": int(len(rows)), "fit_population_rows": int(len(selected_evidence)),
        "fit_long_rows": int(np.count_nonzero(selected_side == 0)),
        "fit_short_rows": int(np.count_nonzero(selected_side == 1)),
        "selected_row_indices_sha256": hashlib.sha256(np.ascontiguousarray(selected_rows, dtype="<i8").tobytes()).hexdigest(),
        "selected_side_stream_sha256": hashlib.sha256(np.ascontiguousarray(selected_side, dtype="i1").tobytes()).hexdigest(),
        "selected_path_evidence_stream_sha256": hashlib.sha256(np.ascontiguousarray(selected_evidence, dtype="<f8").tobytes()).hexdigest(),
        **artifact,
        "future_outcomes_used_as_model_inputs": False,
        "val_test_rows_used_for_fit": 0, "live_direction_authority": False,
    }
    policy["policy_sha256"] = canonical_causal_m1_position_size_target_policy_sha256(policy)
    return require_causal_m1_position_size_target_policy(
        policy, expected_source_parquet_sha256=source_parquet_sha256,
        expected_tape_provenance_sha256=tape_provenance_sha256,
        expected_m1_source_sha256=m1_source_sha256,
        expected_direction_policy_sha256=direction["policy_sha256"],
    )


def require_causal_m1_position_size_target_policy(
    value: Any,
    *,
    expected_source_parquet_sha256: str | None = None,
    expected_tape_provenance_sha256: str | None = None,
    expected_m1_source_sha256: str | None = None,
    expected_direction_policy_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _POLICY_KEYS:
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_SCHEMA_INVALID")
    policy = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    declared = policy.pop("policy_sha256", None)
    if not _is_sha256(declared) or declared != canonical_causal_m1_position_size_target_policy_sha256(policy):
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_HASH_INVALID")
    policy["policy_sha256"] = declared
    if (
        policy["schema_version"] != ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_SCHEMA_VERSION
        or policy["decision"] != "PASS" or policy["fit_split"] != "train" or policy["fit_scope"] != "TRAIN_ONLY"
        or policy["fit_method"] != ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_FIT_METHOD
        or policy["mapping_method"] != ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_MAPPING_METHOD
        or policy["evidence_formula"] != ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_EVIDENCE_FORMULA
        or policy["target_domain"] != [0.0, 1.0] or policy["target_population"] != "tradable_selected_side_rows_only"
        or policy["target_mask_column"] != ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_MASK_COLUMN
        or policy["nontradable_target_defined"] is not False
        or policy["nontradable_storage_sentinel"] != ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_STORAGE_SENTINEL
        or policy["unmasked_training_allowed"] is not False
        or policy["m5_decision_clock"] != "authoritative_closed_m5_bar_then_exact_m1_fill"
        or policy["future_outcomes_used_as_model_inputs"] is not False
        or policy["val_test_rows_used_for_fit"] != 0 or policy["live_direction_authority"] is not False
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_CONTRACT_INVALID")
    for field, expected in (
        ("source_parquet_sha256", expected_source_parquet_sha256),
        ("tape_provenance_sha256", expected_tape_provenance_sha256),
        ("m1_source_sha256", expected_m1_source_sha256),
        ("entry_causal_m1_target_policy_sha256", expected_direction_policy_sha256),
    ):
        if not _is_sha256(policy[field]):
            raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_LINEAGE_INVALID")
        if expected is not None and policy[field] != expected:
            raise RuntimeError(f"ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_{field.upper()}_MISMATCH")
    start = _utc(policy["train_start_utc"], field="TRAIN_START")
    end = _utc(policy["train_end_utc"], field="TRAIN_END")
    counts = ("path_horizon_bars", "fit_candidate_rows", "fit_population_rows", "fit_long_rows", "fit_short_rows")
    if (
        end <= start or any(isinstance(policy[field], bool) or not isinstance(policy[field], int) or policy[field] <= 0 for field in counts)
        or policy["fit_population_rows"] > policy["fit_candidate_rows"]
        or policy["fit_long_rows"] + policy["fit_short_rows"] != policy["fit_population_rows"]
        or any(not _is_sha256(policy[field]) for field in ("selected_row_indices_sha256", "selected_side_stream_sha256", "selected_path_evidence_stream_sha256"))
        or policy["ecdf_artifact_format"] != "numpy_npy_v1_exact_float64" or policy["ecdf_artifact_dtype"] != "float64_le"
        or not isinstance(policy["ecdf_artifact_path"], str) or not policy["ecdf_artifact_path"]
        or not _is_sha256(policy["ecdf_artifact_sha256"]) or not _is_sha256(policy["ecdf_artifact_array_sha256"])
        or type(policy["ecdf_artifact_rows"]) is not int or policy["ecdf_artifact_rows"] != policy["fit_population_rows"]
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_POPULATION_INVALID")
    _load_ecdf(policy)
    return policy


def causal_m1_position_size_targets_from_policy(
    *,
    policy: Mapping[str, Any],
    mfe_first_n_bps: Any,
    mae_first_n_bps: Any,
    selected_side: Any,
    trade_mask: Any,
) -> dict[str, np.ndarray]:
    """Apply only the frozen M1-path ECDF, never fit/recenter it."""

    normalized = require_causal_m1_position_size_target_policy(policy)
    mfe = np.asarray(mfe_first_n_bps, dtype=np.float64)
    mae = np.asarray(mae_first_n_bps, dtype=np.float64)
    side = np.asarray(selected_side)
    trade = np.asarray(trade_mask).astype(bool)
    if not (mfe.shape == mae.shape == side.shape == trade.shape) or mfe.ndim != 1:
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_INPUT_SHAPE_INVALID")
    if (
        not np.isfinite(mfe).all() or not np.isfinite(mae).all() or np.any(mae < 0.0)
        or np.any(~np.isin(side, (-1, 0, 1))) or not np.array_equal(trade, side >= 0)
    ):
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_INPUT_INVALID")
    evidence = mfe - mae
    target = np.full(len(mfe), float(normalized["nontradable_storage_sentinel"]), dtype=np.float64)
    knots = _load_ecdf(normalized)
    target[trade] = np.searchsorted(knots, evidence[trade], side="right") / float(len(knots))
    if not np.isfinite(target).all() or np.any(target < 0.0) or np.any(target > 1.0):
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_OUTPUT_INVALID")
    return {"target": target.astype(np.float32), "mask": trade.astype(np.float32), "path_evidence_bps": evidence.astype(np.float32)}


def causal_m1_position_size_target_policy_contract(
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = require_causal_m1_position_size_target_policy(policy)
    return {
        "entry_causal_m1_position_size_target_policy": normalized,
        "entry_causal_m1_position_size_target_policy_sha256": normalized["policy_sha256"],
        "position_size_target_source": "train_fitted_exact_m1_selected_side_path_ecdf",
        "position_size_target_mask": normalized["target_mask_column"],
        "position_size_target_unmasked_training_allowed": False,
        "position_size_target_live_direction_authority": False,
    }


def require_causal_m1_position_size_target_manifest_binding(
    extra: Any,
    *,
    expected_source_parquet_sha256: str | None = None,
    expected_tape_provenance_sha256: str | None = None,
    expected_m1_source_sha256: str | None = None,
    expected_direction_policy_sha256: str | None = None,
    expected_train_start: Any | None = None,
    expected_train_end: Any | None = None,
) -> dict[str, Any]:
    """Validate the complete frozen causal-M1 sizing projection in a manifest."""

    if not isinstance(extra, Mapping):
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_MANIFEST_EXTRA_INVALID")
    policy = require_causal_m1_position_size_target_policy(
        extra.get("entry_causal_m1_position_size_target_policy"),
        expected_source_parquet_sha256=expected_source_parquet_sha256,
        expected_tape_provenance_sha256=expected_tape_provenance_sha256,
        expected_m1_source_sha256=expected_m1_source_sha256,
        expected_direction_policy_sha256=expected_direction_policy_sha256,
    )
    expected_projection = causal_m1_position_size_target_policy_contract(policy)
    for key, expected in expected_projection.items():
        if extra.get(key) != expected:
            raise RuntimeError(
                "ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_MANIFEST_PROJECTION_MISMATCH: "
                f"key={key}"
            )
    if expected_train_start is not None and _utc(
        expected_train_start, field="EXPECTED_TRAIN_START"
    ) != _utc(policy["train_start_utc"], field="TRAIN_START"):
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_MANIFEST_TRAIN_START_MISMATCH")
    if expected_train_end is not None and _utc(
        expected_train_end, field="EXPECTED_TRAIN_END"
    ) != _utc(policy["train_end_utc"], field="TRAIN_END"):
        raise RuntimeError("ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_MANIFEST_TRAIN_END_MISMATCH")
    return policy
