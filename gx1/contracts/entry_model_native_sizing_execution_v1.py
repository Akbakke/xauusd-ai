"""Immutable joint Entry-sizing plus same-candidate Exit replay evidence.

This contract does not simulate or select direction.  It admits only a full
candidate TEST row set whose non-FLAT rows were stepped through the exact
candidate bundle's unified Exit head and whose learned sizing metrics are
independently recomputed from the resulting executable bid/ask exits. Missing
rows, horizon caps, failed Exit traces, candidate-byte drift, or a passive
sizing head fail closed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    MODEL_NATIVE_SIZING_OOS_ROW_COLUMNS,
    MODEL_NATIVE_SIZING_RUNTIME_CONSTRAINT_KEYS,
    ModelNativeSizingContractError,
    calibrated_sizing_transform,
    load_bound_sizing_calibration,
    load_bound_sizing_oos_proof,
    recompute_sizing_oos_evidence,
    require_immutable_json_binding,
    sha256_file,
    sizing_risk_policy_metadata,
)


MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_SCHEMA_VERSION = (
    "entry_model_native_joint_exit_sizing_proof_v9"
)
MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_EVENT_PREFIX = (
    "ENTRY_MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF"
)
MODEL_NATIVE_JOINT_EXIT_SIZING_REPLAY_CONTRACT = (
    "full_candidate_test_exact_unified_model_exit_head_to_exit_now_v9"
)
CANONICAL_UNIFIED_REPLAY_PRODUCER_SCHEMA_VERSION = (
    "canonical_unified_candidate_full_test_replay_producer_v3"
)
CANONICAL_UNIFIED_REPLAY_PRODUCER_CONTRACT = (
    "same_candidate_bundle_unified_exit_full_test_owned_rows_v3"
)
UNIFIED_CANDIDATE_BUNDLE_AUTHORITY_SCHEMA_VERSION = (
    "gx1_unified_candidate_bundle_authority_v1"
)
MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE = (
    "canonical_oos_reference"
)
MODEL_NATIVE_JOINT_EXIT_SIZING_MIN_TRADES = 128
MODEL_NATIVE_JOINT_EXIT_SIZING_MIN_TRADES_PER_SIDE = 32
# The admitted OOS rows use one canonical account scenario per independent
# decision. They do not replay shared equity, margin, exposure or drawdown
# across overlapping trades. Until an exact shared-portfolio producer exists,
# only one simultaneous exposure is provable.
MODEL_NATIVE_JOINT_EXIT_MAX_LIVE_TRADES = 1
_CANDIDATE_BUNDLE_AUTHORITY_KEYS = frozenset(
    {
        "schema_version",
        "bundle_dir",
        "bundle_commit_manifest_path",
        "bundle_commit_manifest_sha256",
        "bundle_commit_sha256",
        "bundle_commit_manifest",
        "evaluation_bundle",
    }
)
_CANONICAL_UNIFIED_REPLAY_PRODUCER_KEYS = frozenset(
    {
        "schema_version",
        "producer_contract",
        "decision",
        "failures",
        "source_tape",
        "prebuilt_pair",
        "train_rank_reference",
        "runtime_predictions",
        "prediction_report_artifact",
        "prediction_provenance",
        "canonical_oos_rows",
        "candidate_bundle_authority",
        "replay_rows",
        "exit_trace_rows",
        "producer_source_files",
        "producer_source_inventory_sha256",
        "rows",
        "trade_rows",
        "trace_rows",
        "first_utc",
        "last_utc",
    }
)
_CANONICAL_UNIFIED_REPLAY_ADDITIONAL_SOURCE_FILES = frozenset(
    {
        "gx1/contracts/entry_model_native_input_normalization_v1.py",
        "gx1/contracts/entry_model_native_learned_component_movement_v1.py",
        "gx1/contracts/entry_model_native_sizing_authority_v1.py",
        "gx1/contracts/entry_model_native_sizing_calibration_v1.py",
        "gx1/contracts/entry_model_native_sizing_execution_v1.py",
        "gx1/contracts/entry_model_native_state_v2.py",
        "gx1/contracts/entry_model_native_tf_input_scale_v1.py",
        "gx1/contracts/entry_model_native_training_objective_v1.py",
        "gx1/contracts/xau_tape_provenance_v1.py",
        "gx1/execution/oanda_client.py",
        "gx1/execution/v12_model_native_state_live.py",
        "gx1/execution/v12_pipeline.py",
        "gx1/execution/v12_smart_entry_live.py",
        "gx1/execution/v12_trade_state.py",
        "gx1/features/tf_agreement_score.py",
        "gx1/models/entry_v10/entry_v10_bundle.py",
        "gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py",
        "gx1/scripts/__init__.py",
        "gx1/scripts/audit_seq513_source_cascade_v1.py",
        "gx1/scripts/build_entry_v10_ctx_training_dataset_v3.py",
        "gx1/scripts/finalize_entry_model_native_sizing_v1.py",
        "gx1/scripts/materialize_cv3_modelrange_v1.py",
        "gx1/scripts/materialize_entry_model_native_seq513_signal_manifest_v1.py",
        "gx1/utils/env_loader.py",
        "gx1/utils/granularity.py",
    }
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def candidate_bundle_authority(
    *,
    bundle_dir: Path,
    evaluation_bundle: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    """Bind the exact pre-activation bundle used by both Entry and Exit."""

    from gx1.contracts.entry_model_native_bundle_commit_v1 import (
        MANIFEST_NAME,
        require_bundle_commit_manifest,
    )

    raw_dir = Path(bundle_dir).expanduser()
    if (
        not raw_dir.is_absolute()
        or raw_dir.is_symlink()
        or not raw_dir.is_dir()
    ):
        _fail(context, "candidate bundle directory is invalid")
    root = raw_dir.resolve()
    evaluation = dict(evaluation_bundle)
    if str(evaluation.get("bundle_dir") or "") != str(root):
        _fail(context, "candidate bundle differs from evaluation bundle")
    manifest = require_bundle_commit_manifest(root)
    expected_core = {
        "bundle_metadata.json": evaluation.get("bundle_metadata_sha256"),
        "MASTER_TRANSFORMER_LOCK.json": evaluation.get(
            "master_transformer_lock_sha256"
        ),
        "model_state_dict.pt": evaluation.get("model_state_dict_sha256"),
    }
    for name, expected_sha in expected_core.items():
        binding = manifest["artifacts"].get(name)
        if (
            not isinstance(binding, Mapping)
            or binding.get("sha256") != expected_sha
        ):
            _fail(context, f"candidate commit differs from evaluation {name}")
    manifest_path = root / MANIFEST_NAME
    return {
        "schema_version": UNIFIED_CANDIDATE_BUNDLE_AUTHORITY_SCHEMA_VERSION,
        "bundle_dir": str(root),
        "bundle_commit_manifest_path": str(manifest_path),
        "bundle_commit_manifest_sha256": sha256_file(manifest_path),
        "bundle_commit_sha256": manifest["commit_sha256"],
        "bundle_commit_manifest": manifest,
        "evaluation_bundle": evaluation,
    }


def require_candidate_bundle_authority(
    value: object,
    *,
    evaluation_bundle: Mapping[str, Any],
    context: str,
    verify_files: bool,
) -> dict[str, Any]:
    """Validate and optionally re-hash one candidate bundle authority."""

    observed = _exact_keys(
        value,
        _CANDIDATE_BUNDLE_AUTHORITY_KEYS,
        context=context,
    )
    if (
        observed["schema_version"]
        != UNIFIED_CANDIDATE_BUNDLE_AUTHORITY_SCHEMA_VERSION
        or observed["evaluation_bundle"] != dict(evaluation_bundle)
    ):
        _fail(context, "candidate bundle authority lineage mismatch")
    bundle_dir = Path(str(observed["bundle_dir"] or "")).expanduser()
    manifest_path = Path(
        str(observed["bundle_commit_manifest_path"] or "")
    ).expanduser()
    if (
        not bundle_dir.is_absolute()
        or not manifest_path.is_absolute()
        or manifest_path != bundle_dir / "ENTRY_MODEL_NATIVE_BUNDLE_COMMIT.json"
    ):
        _fail(context, "candidate bundle authority paths are invalid")
    for field_name in (
        "bundle_commit_manifest_sha256",
        "bundle_commit_sha256",
    ):
        _sha(observed[field_name], context=f"{context}.{field_name}")
    if verify_files:
        expected = candidate_bundle_authority(
            bundle_dir=bundle_dir,
            evaluation_bundle=evaluation_bundle,
            context=context,
        )
        if observed != expected:
            _fail(context, "candidate bundle bytes differ")
    return dict(observed)
MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS = frozenset(
    {
        "entry_fill_time",
        "exit_replay_status",
        "model_exit_decision_bar_time",
        "model_exit_fill_time",
        "model_exit_fill_bid",
        "model_exit_fill_ask",
        "exit_reason",
        "exit_steps",
        "exit_trace_sha256",
        "candidate_bundle_sha256",
    }
)
MODEL_NATIVE_JOINT_EXIT_SIZING_ROW_COLUMNS = frozenset(
    MODEL_NATIVE_SIZING_OOS_ROW_COLUMNS
    | MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS
)
MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS = frozenset(
    {
        "reference_row_id",
        "entry_fill_time",
        "step",
        "closed_bar_time",
        "model_exit_fill_time",
        "bar_committed",
        "action_id",
        "action",
        "decision_source",
        "state_bid",
        "state_ask",
        "state_pnl_bps",
        "exit_hold_logit",
        "exit_now_logit",
        "exit_hold_prob",
        "exit_now_prob",
        "candidate_bundle_sha256",
        "entry_snapshot_sha256",
        "exit_path_envelope_sha256",
        "output_evidence_sha256",
        "closed_m1_source_path",
        "closed_m1_source_sha256",
    }
)
MODEL_NATIVE_SIZING_RUNTIME_PARITY_SCHEMA_VERSION = (
    "entry_model_native_sizing_runtime_parity_v1"
)
MODEL_NATIVE_SIZING_RUNTIME_PARITY_EVENT_PREFIX = (
    "ENTRY_MODEL_NATIVE_SIZING_RUNTIME_PARITY"
)
MODEL_NATIVE_SIZING_RUNTIME_PARITY_CONTRACT = (
    "broker_live_shadow_exact_learned_sizing_transform_v1"
)
MODEL_NATIVE_SIZING_RUNTIME_PARITY_MIN_ROWS = 32
MODEL_NATIVE_SIZING_RUNTIME_PARITY_MIN_ROWS_PER_CLASS = 8
MODEL_NATIVE_SIZING_RUNTIME_PARITY_MAX_AGE_SECONDS = 86_400
MODEL_NATIVE_SIZING_RUNTIME_PARITY_MAX_EVENT_LAG_SECONDS = 300
_RUNTIME_PARITY_FLOAT_FIELDS = (
    "calibrated_size_fraction",
    "applied_size_multiplier",
    "reference_pre_round_units",
    "pre_round_units",
)
_RUNTIME_PARITY_INT_FIELDS = (
    "capacity_units",
    "units",
)
MODEL_NATIVE_SIZING_RUNTIME_PARITY_COLUMNS = frozenset(
    {
        "time",
        "position_size_logit",
        "model_direction_index",
        "direction_after_sizing",
        *MODEL_NATIVE_SIZING_RUNTIME_CONSTRAINT_KEYS,
        *_RUNTIME_PARITY_FLOAT_FIELDS,
        *_RUNTIME_PARITY_INT_FIELDS,
        "authorized_order",
        "no_order_reason",
        "runtime_bundle_metadata_sha256",
        "runtime_model_state_dict_sha256",
        "runtime_adoption_sha256",
        "order_submitted",
    }
)
_RUNTIME_PARITY_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "parity_contract",
        "adoption_artifact",
        "bundle_identity",
        "observations",
        "coverage",
    }
)
_RUNTIME_PARITY_COVERAGE_KEYS = frozenset(
    {
        "rows",
        "long_rows",
        "short_rows",
        "flat_rows",
        "first_utc",
        "last_utc",
        "utc_ns_sha256",
        "transaction_id_sequence_sha256",
        "distinct_transaction_ids",
        "max_float_abs_error",
        "integer_mismatch_count",
        "boolean_mismatch_count",
        "direction_mismatch_count",
        "order_submission_count",
    }
)
_RECOMPUTED_SECTION_NAMES = (
    "full_test_coverage",
    "position_size_head_liveness",
    "monotonicity",
    "exposure_bounds",
    "drawdown_bounds",
    "paired_oos_utility",
    "account_capacity_grid",
    "direction_invariance",
)
_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "replay_contract",
        "risk_policy",
        "calibration_artifact",
        "oos_proof_artifact",
        "evaluation_bundle",
        "test_prediction_provenance",
        "candidate_bundle_authority",
        "canonical_unified_replay_producer",
        "replay_rows",
        "exit_trace_rows",
        "exit_replay_coverage",
        *_RECOMPUTED_SECTION_NAMES,
    }
)
_EXIT_COVERAGE_KEYS = frozenset(
    {
        "rows",
        "trade_rows",
        "long_rows",
        "short_rows",
        "flat_rows",
        "failed_rows",
        "first_utc",
        "last_utc",
        "utc_ns_sha256",
        "exit_trace_sequence_sha256",
    }
)


class ModelNativeSizingExecutionContractError(RuntimeError):
    """Joint unified-Exit sizing evidence is absent, stale, or malformed."""


def _fail(context: str, detail: str) -> None:
    raise ModelNativeSizingExecutionContractError(
        f"[{context}_INVALID] {detail}"
    )


def _exact_keys(
    value: Mapping[str, Any] | Any,
    expected: frozenset[str],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(context, "expected an object")
    observed = dict(value)
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if missing or unexpected:
        _fail(context, f"exact keys mismatch: missing={missing} unexpected={unexpected}")
    return observed


def _utc(value: Any, *, context: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise ModelNativeSizingExecutionContractError(
            f"[{context}_INVALID] invalid UTC timestamp"
        ) from exc
    if parsed.tz is None or str(parsed.tz) != "UTC":
        _fail(context, "timestamp must be timezone-aware UTC")
    return parsed


def _sha(value: Any, *, context: str) -> str:
    parsed = str(value or "").strip().lower()
    if len(parsed) != 64 or any(ch not in "0123456789abcdef" for ch in parsed):
        _fail(context, "not an exact SHA-256")
    return parsed


def _json_file(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelNativeSizingExecutionContractError(
            f"[{context}_INVALID] unreadable JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        _fail(context, "JSON root is not an object")
    return value


def _read_bound_regular_file_bytes(
    binding: Mapping[str, Any],
    *,
    path_key: str,
    context: str,
) -> bytes:
    """Read and hash one exact regular-file inode without path reopen races."""

    path = Path(str(binding.get(path_key) or "")).expanduser()
    expected_sha = _sha(binding.get("sha256"), context=f"{context}.sha256")
    if not path.is_absolute():
        _fail(context, "bound path must be absolute")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ModelNativeSizingExecutionContractError(
            f"[{context}_INVALID] bound file cannot be opened exactly: {path}"
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            _fail(context, f"bound path is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        _fail(context, f"bound file changed while being read: {path}")
    try:
        path_state = os.lstat(path)
    except OSError as exc:
        raise ModelNativeSizingExecutionContractError(
            f"[{context}_INVALID] bound path disappeared after read: {path}"
        ) from exc
    if (
        stat.S_ISLNK(path_state.st_mode)
        or not stat.S_ISREG(path_state.st_mode)
        or path_state.st_dev != after.st_dev
        or path_state.st_ino != after.st_ino
    ):
        _fail(context, f"bound path identity changed while being read: {path}")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        _fail(context, f"bound file size changed while being read: {path}")
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        _fail(context, f"bound file hash mismatch: {path}")
    return payload


def _read_bound_json_exact(
    binding: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    payload = _read_bound_regular_file_bytes(
        binding,
        path_key="json_path",
        context=context,
    )
    try:
        value = json.loads(payload)
    except Exception as exc:
        raise ModelNativeSizingExecutionContractError(
            f"[{context}_INVALID] bound JSON is unreadable"
        ) from exc
    if not isinstance(value, dict):
        _fail(context, "bound JSON root is not an object")
    return value


def read_bound_parquet_exact(
    binding: Mapping[str, Any],
    *,
    context: str,
) -> pd.DataFrame:
    """Parse parquet from the exact bytes that satisfy its SHA-256 binding."""

    payload = _read_bound_regular_file_bytes(
        binding,
        path_key="path",
        context=context,
    )
    try:
        return pd.read_parquet(io.BytesIO(payload))
    except Exception as exc:
        raise ModelNativeSizingExecutionContractError(
            f"[{context}_INVALID] bound parquet is unreadable"
        ) from exc


def _source_binding(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
    verify_file: bool,
) -> dict[str, str]:
    observed = _exact_keys(value, frozenset({"path", "sha256"}), context=context)
    path = Path(str(observed["path"] or "")).expanduser()
    if not path.is_absolute():
        _fail(context, "path must be absolute")
    path = path.resolve()
    expected_sha = _sha(observed["sha256"], context=f"{context}.sha256")
    if verify_file:
        if not path.is_file():
            _fail(context, f"bound file is missing: {path}")
        if sha256_file(path) != expected_sha:
            _fail(context, f"bound file hash mismatch: {path}")
    return {"path": str(path), "sha256": expected_sha}


def _strict_directions(frame: pd.DataFrame, *, context: str) -> np.ndarray:
    numeric = pd.to_numeric(frame["model_direction_index"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(numeric).all()
        or not np.array_equal(numeric, numeric.astype(np.int64))
        or not np.isin(numeric.astype(np.int64), [0, 1, 2]).all()
    ):
        _fail(context, "model_direction_index must be exact LONG/SHORT/FLAT integers")
    return numeric.astype(np.int64)


def joint_exit_trace_sha256(frame: pd.DataFrame, *, context: str) -> str:
    """Hash one ordered per-trade Exit trace using strict JSON scalar values."""

    if set(frame.columns) != set(MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS):
        _fail(context, "Exit trace columns mismatch")
    records: list[dict[str, Any]] = []
    for _, row in frame.loc[:, sorted(MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS)].iterrows():
        records.append(
            {
                "action_id": int(row["action_id"]),
                "action": str(row["action"]),
                "bar_committed": bool(row["bar_committed"]),
                "candidate_bundle_sha256": str(
                    row["candidate_bundle_sha256"]
                ).lower(),
                "closed_bar_time": (
                    None
                    if pd.isna(row["closed_bar_time"])
                    else pd.Timestamp(row["closed_bar_time"]).isoformat()
                ),
                "closed_m1_source_path": str(row["closed_m1_source_path"]),
                "closed_m1_source_sha256": str(
                    row["closed_m1_source_sha256"]
                ).lower(),
                "decision_source": str(row["decision_source"]),
                "entry_fill_time": pd.Timestamp(
                    row["entry_fill_time"]
                ).isoformat(),
                "entry_snapshot_sha256": str(
                    row["entry_snapshot_sha256"]
                ).lower(),
                "exit_hold_logit": float(row["exit_hold_logit"]),
                "exit_hold_prob": float(row["exit_hold_prob"]),
                "exit_now_logit": float(row["exit_now_logit"]),
                "exit_now_prob": float(row["exit_now_prob"]),
                "exit_path_envelope_sha256": str(
                    row["exit_path_envelope_sha256"]
                ).lower(),
                "model_exit_fill_time": pd.Timestamp(
                    row["model_exit_fill_time"]
                ).isoformat(),
                "output_evidence_sha256": str(
                    row["output_evidence_sha256"]
                ).lower(),
                "reference_row_id": str(row["reference_row_id"]),
                "state_ask": float(row["state_ask"]),
                "state_bid": float(row["state_bid"]),
                "state_pnl_bps": float(row["state_pnl_bps"]),
                "step": int(row["step"]),
            }
        )
    return hashlib.sha256(
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def recompute_joint_exit_replay_coverage(
    frame: pd.DataFrame,
    *,
    exit_trace_rows: pd.DataFrame,
    candidate_bundle_sha256: str,
    context: str,
) -> dict[str, Any]:
    """Recompute strict full-TEST unified Exit coverage from exact model rows."""

    if set(frame.columns) != set(MODEL_NATIVE_JOINT_EXIT_SIZING_ROW_COLUMNS):
        _fail(
            context,
            "replay row columns mismatch: "
            f"missing={sorted(MODEL_NATIVE_JOINT_EXIT_SIZING_ROW_COLUMNS - set(frame.columns))} "
            f"unexpected={sorted(set(frame.columns) - MODEL_NATIVE_JOINT_EXIT_SIZING_ROW_COLUMNS)}",
        )
    times = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        _fail(context, "replay times must be unique, finite, monotonic UTC")
    if len(frame) < 256:
        _fail(context, "full TEST joint replay requires at least 256 rows")
    directions = _strict_directions(frame, context=context)
    trade_mask = np.isin(directions, [0, 1])
    long_rows = int(np.count_nonzero(directions == 0))
    short_rows = int(np.count_nonzero(directions == 1))
    flat_mask = directions == 2
    trade_rows = int(np.count_nonzero(trade_mask))
    if trade_rows < MODEL_NATIVE_JOINT_EXIT_SIZING_MIN_TRADES:
        _fail(context, "insufficient non-FLAT unified-Exit replay support")
    if min(long_rows, short_rows) < MODEL_NATIVE_JOINT_EXIT_SIZING_MIN_TRADES_PER_SIDE:
        _fail(context, "insufficient LONG/SHORT unified-Exit replay support")
    if set(frame.loc[trade_mask, "exit_replay_status"].astype(str)) != {"EXIT_NOW"}:
        _fail(context, "every non-FLAT row must reach unified Exit EXIT_NOW")
    if set(frame.loc[flat_mask, "exit_replay_status"].astype(str)) - {"FLAT_NO_ORDER"}:
        _fail(context, "FLAT rows must remain explicit no-order rows")
    steps = pd.to_numeric(frame["exit_steps"], errors="coerce").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(steps).all()
        or not np.array_equal(steps, steps.astype(np.int64))
        or np.any(steps[trade_mask] <= 0)
        or np.any(steps[flat_mask] != 0)
    ):
        _fail(context, "exit_steps must be positive for trades and zero for FLAT")
    entry_fill_times = pd.to_datetime(
        frame["entry_fill_time"],
        utc=True,
        errors="coerce",
    )
    if entry_fill_times.isna().any() or not bool(
        (
            entry_fill_times
            == times + pd.Timedelta(minutes=5)
        ).all()
    ):
        _fail(
            context,
            "entry_fill_time must be exactly decision time + 5m",
        )
    model_fill_times = pd.to_datetime(
        frame["model_exit_fill_time"],
        utc=True,
        errors="coerce",
    )
    if model_fill_times[trade_mask].isna().any() or np.any(
        model_fill_times[trade_mask].to_numpy()
        <= entry_fill_times[trade_mask].to_numpy()
    ):
        _fail(
            context,
            "trade model_exit_fill_time must be finite UTC after Entry fill",
        )
    decision_bar_times = pd.to_datetime(
        frame["model_exit_decision_bar_time"],
        utc=True,
        errors="coerce",
    )
    claimed_decision_bars = trade_mask & decision_bar_times.notna().to_numpy()
    if np.any(
        decision_bar_times[claimed_decision_bars].to_numpy()
        >= model_fill_times[claimed_decision_bars].to_numpy()
    ):
        _fail(context, "trade model Exit decision-bar time is invalid")
    if (
        model_fill_times[flat_mask].notna().any()
        or decision_bar_times[flat_mask].notna().any()
    ):
        _fail(context, "FLAT rows cannot claim model Exit times")
    if frame.loc[trade_mask, "exit_reason"].isna().any() or not frame.loc[
        trade_mask, "exit_reason"
    ].astype(str).str.strip().all():
        _fail(context, "trade rows require a non-empty model Exit reason")
    if set(frame.loc[flat_mask, "exit_reason"].astype(str)) - {"MODEL_FLAT"}:
        _fail(context, "FLAT rows require exact MODEL_FLAT reason")
    trace_hashes = frame["exit_trace_sha256"].astype(str).str.lower().tolist()
    for index, value in enumerate(trace_hashes):
        _sha(value, context=f"{context}.exit_trace_sha256[{index}]")
    expected_candidate_bundle_sha = _sha(
        candidate_bundle_sha256,
        context=f"{context}.candidate_bundle_sha",
    )
    if set(frame["candidate_bundle_sha256"].astype(str).str.lower()) != {
        expected_candidate_bundle_sha
    }:
        _fail(context, "row candidate bundle differs from proof authority")
    if set(exit_trace_rows.columns) != set(MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS):
        _fail(context, "Exit trace row columns mismatch")
    if exit_trace_rows.empty:
        _fail(context, "unified Exit trace artifact is empty")
    trace_bundle = set(
        exit_trace_rows["candidate_bundle_sha256"].astype(str).str.lower()
    )
    if trace_bundle != {expected_candidate_bundle_sha}:
        _fail(context, "Exit trace candidate bundle differs from proof authority")
    replay_ids = frame["reference_row_id"].astype(str)
    if replay_ids.duplicated().any():
        _fail(context, "replay reference_row_id must be unique")
    trade_ids = set(replay_ids[trade_mask])
    flat_ids = set(replay_ids[flat_mask])
    trace_ids = set(exit_trace_rows["reference_row_id"].astype(str))
    if trace_ids != trade_ids or trace_ids.intersection(flat_ids):
        _fail(context, "Exit traces must cover exactly every non-FLAT replay row")
    replay_by_id = frame.assign(reference_row_id=replay_ids).set_index(
        "reference_row_id", drop=False
    )
    normalized_trace_rows = exit_trace_rows.assign(
        reference_row_id=exit_trace_rows["reference_row_id"].astype(str)
    )
    for reference_row_id, trace in normalized_trace_rows.groupby(
        "reference_row_id", sort=False
    ):
        replay_row = replay_by_id.loc[reference_row_id]
        trace = trace.sort_values("step", kind="mergesort").reset_index(drop=True)
        steps_observed = pd.to_numeric(trace["step"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        expected_steps = np.arange(1, int(replay_row["exit_steps"]) + 1)
        if not np.array_equal(steps_observed, expected_steps):
            _fail(context, f"{reference_row_id} Exit trace steps are not contiguous")
        decision_time = pd.Timestamp(replay_row["time"])
        entry_time = pd.Timestamp(replay_row["entry_fill_time"])
        if (
            pd.isna(entry_time)
            or entry_time != decision_time + pd.Timedelta(minutes=5)
        ):
            _fail(
                context,
                "joint Exit entry_fill_time must be exactly decision time + 5m",
            )
        trace_entry_times = pd.to_datetime(
            trace["entry_fill_time"],
            utc=True,
            errors="coerce",
        )
        model_fill_times = pd.to_datetime(
            trace["model_exit_fill_time"],
            utc=True,
            errors="coerce",
        )
        closed_bar_times = pd.to_datetime(
            trace["closed_bar_time"],
            utc=True,
            errors="coerce",
        )
        expected_closed_bar_times = pd.date_range(
            start=entry_time,
            periods=len(trace),
            freq="min",
        )
        expected_model_fill_times = expected_closed_bar_times + pd.Timedelta(
            minutes=1
        )
        committed = trace["bar_committed"].map(
            lambda value: value
            if isinstance(value, (bool, np.bool_))
            else None
        )
        if (
            trace_entry_times.isna().any()
            or set(trace_entry_times) != {entry_time}
            or closed_bar_times.isna().any()
            or model_fill_times.isna().any()
            or not np.array_equal(
                closed_bar_times.astype("int64").to_numpy(),
                expected_closed_bar_times.astype("int64").to_numpy(),
            )
            or not np.array_equal(
                model_fill_times.astype("int64").to_numpy(),
                expected_model_fill_times.astype("int64").to_numpy(),
            )
            or committed.isna().any()
            or not committed.astype(bool).all()
        ):
            _fail(context, f"{reference_row_id} Exit trace time binding is invalid")
        actions = pd.to_numeric(trace["action_id"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        action_names = trace["action"].astype(str).to_numpy()
        if (
            not np.array_equal(actions, actions.astype(np.int64))
            or np.any(actions[:-1] != 0)
            or int(actions[-1]) != 1
            or np.any(action_names[:-1] != "HOLD")
            or action_names[-1] != "EXIT_NOW"
        ):
            _fail(context, f"{reference_row_id} must HOLD then finish EXIT_NOW")
        if (
            set(trace["decision_source"].astype(str)) != {"unified_model"}
            or str(replay_row["exit_reason"]) != "UNIFIED_MODEL_ARGMAX"
        ):
            _fail(context, f"{reference_row_id} Exit reason differs from trace")
        state_bid = pd.to_numeric(
            trace["state_bid"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        state_ask = pd.to_numeric(
            trace["state_ask"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        state_pnl = pd.to_numeric(
            trace["state_pnl_bps"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        if (
            not np.isfinite(state_bid).all()
            or not np.isfinite(state_ask).all()
            or not np.isfinite(state_pnl).all()
            or np.any(state_bid <= 0.0)
            or np.any(state_ask < state_bid)
        ):
            _fail(context, f"{reference_row_id} Exit trace prices are invalid")
        logits = trace.loc[
            :, ["exit_hold_logit", "exit_now_logit"]
        ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        probs = trace.loc[
            :, ["exit_hold_prob", "exit_now_prob"]
        ].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        expected_probs = np.exp(shifted)
        expected_probs /= expected_probs.sum(axis=1, keepdims=True)
        if (
            not np.isfinite(logits).all()
            or not np.isfinite(probs).all()
            or np.any(logits[:, 0] == logits[:, 1])
            or not np.allclose(probs, expected_probs, rtol=0.0, atol=1e-6)
            or not np.array_equal(np.argmax(logits, axis=1), actions.astype(int))
        ):
            _fail(context, f"{reference_row_id} Exit model outputs are invalid")
        hash_fields = (
            "candidate_bundle_sha256",
            "entry_snapshot_sha256",
            "exit_path_envelope_sha256",
            "output_evidence_sha256",
            "closed_m1_source_sha256",
        )
        for field_name in hash_fields:
            for index, value in enumerate(trace[field_name].astype(str)):
                _sha(
                    value,
                    context=f"{context}.{reference_row_id}.{field_name}[{index}]",
                )
        if set(trace["candidate_bundle_sha256"].astype(str).str.lower()) != {
            expected_candidate_bundle_sha
        }:
            _fail(context, f"{reference_row_id} candidate bundle changed in trace")
        source_paths = trace["closed_m1_source_path"].astype(str)
        if (
            not source_paths.str.strip().all()
            or any(not Path(value).is_absolute() for value in source_paths)
            or len(set(source_paths)) != 1
            or len(
                set(trace["closed_m1_source_sha256"].astype(str).str.lower())
            )
            != 1
        ):
            _fail(context, f"{reference_row_id} source binding is invalid")
        for index, row in trace.iterrows():
            output = {
                "exit_action_logits": [
                    float(row["exit_hold_logit"]),
                    float(row["exit_now_logit"]),
                ],
                "exit_action_probs": [
                    float(row["exit_hold_prob"]),
                    float(row["exit_now_prob"]),
                ],
                "exit_action_index": int(row["action_id"]),
                "action": str(row["action"]),
                "decision_source": str(row["decision_source"]),
                "bundle_sha256": str(row["candidate_bundle_sha256"]).lower(),
                "entry_snapshot_sha256": str(
                    row["entry_snapshot_sha256"]
                ).lower(),
                "exit_path_envelope_sha256": str(
                    row["exit_path_envelope_sha256"]
                ).lower(),
            }
            if _canonical_sha256(output) != str(
                row["output_evidence_sha256"]
            ).lower():
                _fail(
                    context,
                    f"{reference_row_id} output evidence hash mismatch at {index}",
                )
        direction = int(replay_row["model_direction_index"])
        expected_state_pnl = (
            (state_bid - float(replay_row["entry_ask"]))
            / float(replay_row["entry_ask"])
            * 10_000.0
            if direction == 0
            else (float(replay_row["entry_bid"]) - state_ask)
            / float(replay_row["entry_bid"])
            * 10_000.0
        )
        if not np.allclose(
            state_pnl,
            expected_state_pnl,
            rtol=0.0,
            atol=1e-9,
        ):
            _fail(
                context,
                f"{reference_row_id} state PnL differs from closed state prices",
            )
        replay_decision_bar_time = pd.to_datetime(
            replay_row["model_exit_decision_bar_time"],
            utc=True,
            errors="coerce",
        )
        if (
            model_fill_times.iloc[-1]
            != pd.Timestamp(replay_row["model_exit_fill_time"])
            or float(state_bid[-1])
            != float(replay_row["model_exit_fill_bid"])
            or float(state_ask[-1])
            != float(replay_row["model_exit_fill_ask"])
            or replay_decision_bar_time != closed_bar_times.iloc[-1]
        ):
            _fail(
                context,
                f"{reference_row_id} model Exit fill/decision binding is invalid",
            )
        if joint_exit_trace_sha256(
            trace, context=f"{context}.{reference_row_id}.trace_hash"
        ) != str(replay_row["exit_trace_sha256"]).lower():
            _fail(context, f"{reference_row_id} Exit trace hash mismatch")
    flat_trace_sha = hashlib.sha256(b"FLAT_NO_ORDER").hexdigest()
    if set(frame.loc[flat_mask, "exit_trace_sha256"].astype(str).str.lower()) - {
        flat_trace_sha
    }:
        _fail(context, "FLAT rows require the exact no-order trace hash")
    if (
        pd.to_numeric(
            frame.loc[flat_mask, "model_exit_fill_bid"],
            errors="coerce",
        ).notna().any()
        or pd.to_numeric(
            frame.loc[flat_mask, "model_exit_fill_ask"],
            errors="coerce",
        ).notna().any()
    ):
        _fail(context, "FLAT rows cannot claim model Exit fill prices")
    utc_ns = times.astype("int64").to_numpy(dtype=np.int64)
    trace_sequence_sha = hashlib.sha256(
        json.dumps(trace_hashes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "rows": int(len(frame)),
        "trade_rows": trade_rows,
        "long_rows": long_rows,
        "short_rows": short_rows,
        "flat_rows": int(np.count_nonzero(flat_mask)),
        "failed_rows": 0,
        "first_utc": times.iloc[0].isoformat(),
        "last_utc": times.iloc[-1].isoformat(),
        "utc_ns_sha256": hashlib.sha256(utc_ns.tobytes()).hexdigest(),
        "exit_trace_sequence_sha256": trace_sequence_sha,
    }


def require_joint_replay_extends_canonical_oos_rows(
    *,
    canonical_oos_rows: pd.DataFrame,
    replay_rows: pd.DataFrame,
    context: str,
) -> None:
    """Require replay rows to be exact canonical OOS rows plus Exit fields."""

    expected_columns = (
        MODEL_NATIVE_SIZING_OOS_ROW_COLUMNS
        | MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS
    )
    if set(canonical_oos_rows.columns) != set(
        MODEL_NATIVE_SIZING_OOS_ROW_COLUMNS
    ):
        _fail(context, "canonical OOS row columns mismatch")
    if set(replay_rows.columns) != set(expected_columns):
        _fail(context, "joint replay row columns mismatch")
    ordered = sorted(MODEL_NATIVE_SIZING_OOS_ROW_COLUMNS)
    canonical = canonical_oos_rows.loc[:, ordered].reset_index(drop=True)
    observed = replay_rows.loc[:, ordered].reset_index(drop=True)
    if not canonical.equals(observed):
        _fail(
            context,
            "joint replay rows differ from the exact canonical OOS TEST rows",
        )


def require_joint_exit_portfolio_capacity(
    proof: Mapping[str, Any],
    *,
    max_trades: int,
    context: str,
) -> dict[str, Any]:
    """Prove the only admitted single-exposure cap over full TEST decisions."""

    if (
        proof.get("schema_version")
        != MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_SCHEMA_VERSION
    ):
        _fail(context, "joint Exit proof schema mismatch")
    if (
        isinstance(max_trades, bool)
        or not isinstance(max_trades, int)
        or not 1 <= max_trades <= MODEL_NATIVE_JOINT_EXIT_MAX_LIVE_TRADES
    ):
        _fail(
            context,
            "max_trades is outside the portfolio replay contract "
            f"1..{MODEL_NATIVE_JOINT_EXIT_MAX_LIVE_TRADES}",
        )
    replay_binding = _source_binding(
        proof.get("replay_rows"),
        context=f"{context}.replay_rows",
        verify_file=True,
    )
    frame = read_bound_parquet_exact(
        replay_binding,
        context=f"{context}.replay_rows_exact",
    )
    if set(frame.columns) != set(MODEL_NATIVE_JOINT_EXIT_SIZING_ROW_COLUMNS):
        _fail(context, "joint Exit portfolio replay columns mismatch")
    decisions = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    times = pd.to_datetime(frame["entry_fill_time"], utc=True, errors="coerce")
    if (
        decisions.isna().any()
        or times.isna().any()
        or not bool((times == decisions + pd.Timedelta(minutes=5)).all())
    ):
        _fail(context, "portfolio replay entry_fill_time is not exact T+5")
    exits = pd.to_datetime(
        frame["model_exit_fill_time"],
        utc=True,
        errors="coerce",
    )
    directions = _strict_directions(frame, context=f"{context}.directions")
    authorized = frame["authorized_order"].to_numpy(dtype=bool)
    active_exits: list[pd.Timestamp] = []
    admitted: list[int] = []
    blocked = 0
    peak = 0
    for index, entry_time in enumerate(times):
        active_exits = [exit_time for exit_time in active_exits if exit_time > entry_time]
        if directions[index] == 2 or not authorized[index]:
            continue
        exit_time = exits.iloc[index]
        if pd.isna(exit_time) or exit_time <= entry_time:
            _fail(context, "portfolio trade lacks a valid unified-Exit time")
        if len(active_exits) >= max_trades:
            blocked += 1
            continue
        active_exits.append(exit_time)
        admitted.append(index)
        peak = max(peak, len(active_exits))
    if len(admitted) < MODEL_NATIVE_JOINT_EXIT_SIZING_MIN_TRADES:
        _fail(context, "portfolio cap leaves insufficient admitted TEST trades")
    admitted_directions = directions[np.asarray(admitted, dtype=np.int64)]
    long_count = int(np.count_nonzero(admitted_directions == 0))
    short_count = int(np.count_nonzero(admitted_directions == 1))
    if (
        min(long_count, short_count)
        < MODEL_NATIVE_JOINT_EXIT_SIZING_MIN_TRADES_PER_SIDE
    ):
        _fail(context, "portfolio cap leaves insufficient LONG/SHORT support")
    selected = frame.iloc[admitted]
    selected_directions = admitted_directions
    entry_bid = pd.to_numeric(selected["entry_bid"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    entry_ask = pd.to_numeric(selected["entry_ask"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    exit_bid = pd.to_numeric(
        selected["model_exit_fill_bid"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    exit_ask = pd.to_numeric(
        selected["model_exit_fill_ask"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    pnl = np.where(
        selected_directions == 0,
        (exit_bid - entry_ask) / entry_ask * 10_000.0,
        (entry_bid - exit_ask) / entry_bid * 10_000.0,
    )
    if not np.isfinite(pnl).all():
        _fail(context, "portfolio replay produced non-finite PnL")
    mean_total = float(np.mean(pnl))
    mean_long = float(np.mean(pnl[selected_directions == 0]))
    mean_short = float(np.mean(pnl[selected_directions == 1]))
    if min(mean_total, mean_long, mean_short) <= 0.0:
        _fail(context, "portfolio-cap TEST utility is not positive on both sides")
    admitted_ids = selected["reference_row_id"].astype(str).tolist()
    return {
        "contract": "full_test_single_exposure_unified_exit_capacity_v3",
        "max_trades": max_trades,
        "eligible_trade_rows": int(
            np.count_nonzero((directions != 2) & authorized)
        ),
        "admitted_trade_rows": len(admitted),
        "capacity_blocked_rows": blocked,
        "admitted_long_rows": long_count,
        "admitted_short_rows": short_count,
        "peak_concurrent_trades": peak,
        "mean_realized_pnl_bps": mean_total,
        "mean_long_realized_pnl_bps": mean_long,
        "mean_short_realized_pnl_bps": mean_short,
        "admitted_reference_row_ids_sha256": hashlib.sha256(
            json.dumps(admitted_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def canonical_unified_replay_source_code_files() -> frozenset[str]:
    """Return every local source byte reachable by the canonical producer."""

    repo_root = Path(__file__).resolve().parents[2]
    pending = list(_CANONICAL_UNIFIED_REPLAY_ADDITIONAL_SOURCE_FILES)
    observed: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source_path = repo_root / relative
        if not source_path.is_file():
            continue
        observed.add(relative)
        module_name = relative.removesuffix(".py").replace("/", ".")
        is_package = module_name.endswith(".__init__")
        if is_package:
            module_name = module_name.removesuffix(".__init__")
        package = module_name if is_package else module_name.rpartition(".")[0]
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    relative_name = "." * node.level + (node.module or "")
                    try:
                        imported_modules.append(
                            importlib.util.resolve_name(relative_name, package)
                        )
                    except (ImportError, ValueError):
                        continue
                elif node.module:
                    imported_modules.append(node.module)
        for imported in imported_modules:
            if not imported.startswith(("gx1", "gx1_guards")):
                continue
            module_path = Path(*imported.split("."))
            candidate = module_path.with_suffix(".py")
            package_init = module_path / "__init__.py"
            if (repo_root / candidate).is_file():
                pending.append(candidate.as_posix())
            elif (repo_root / package_init).is_file():
                pending.append(package_init.as_posix())
    return frozenset(observed)


def build_canonical_unified_replay_source_inventory(
    repo_root: Path,
) -> list[dict[str, str]]:
    root = Path(repo_root).expanduser()
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve() != root
    ):
        _fail("CANONICAL_UNIFIED_EXIT_SOURCE", "repository root is invalid")
    inventory: list[dict[str, str]] = []
    for relative in sorted(canonical_unified_replay_source_code_files()):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            _fail(
                "CANONICAL_UNIFIED_EXIT_SOURCE",
                f"source file is missing/noncanonical: {relative}",
            )
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode):
            _fail(
                "CANONICAL_UNIFIED_EXIT_SOURCE",
                f"source path is not a regular file: {relative}",
            )
        digest = sha256_file(path)
        after = os.lstat(path)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _fail(
                "CANONICAL_UNIFIED_EXIT_SOURCE",
                f"source file changed while hashing: {relative}",
            )
        inventory.append(
            {
                "relative_path": relative,
                "path": str(path.resolve()),
                "sha256": digest,
            }
        )
    return inventory


def _require_canonical_frozen_pair(
    value: object,
    *,
    context: str,
) -> dict[str, Any]:
    envelope = _exact_keys(
        value,
        frozenset({"generation_root", "identity"}),
        context=context,
    )
    generation_root = Path(str(envelope["generation_root"] or "")).expanduser()
    if (
        not generation_root.is_absolute()
        or generation_root.is_symlink()
        or not generation_root.is_dir()
    ):
        _fail(context, "prebuilt generation root is invalid")
    identity = _exact_keys(
        envelope["identity"],
        frozenset(
            {
                "manifest_path",
                "manifest_sha256",
                "pair_generation_id",
                "canonical_v3",
                "base28",
                "refresh_enabled",
            }
        ),
        context=f"{context}.identity",
    )
    if identity["refresh_enabled"] is not False:
        _fail(context, "prebuilt pair must have refresh disabled")
    from gx1.execution.v12_state_from_prebuilt import (
        read_prebuilt_pair_manifest,
        verify_prebuilt_pair,
    )

    binding = read_prebuilt_pair_manifest(
        Path(str(identity["manifest_path"] or "")),
        generation_root=generation_root,
    )
    verify_prebuilt_pair(binding)
    expected = {
        "manifest_path": str(binding.manifest_path),
        "manifest_sha256": binding.manifest_sha256,
        "pair_generation_id": binding.pair_generation_id,
        "canonical_v3": {
            "path": str(binding.canonical_v3.parquet_path),
            "sha256": binding.canonical_v3.parquet_sha256,
            "rows": binding.canonical_v3.rows,
            "cols_total": binding.canonical_v3.cols_total,
        },
        "base28": {
            "path": str(binding.base28.parquet_path),
            "sha256": binding.base28.parquet_sha256,
            "rows": binding.base28.rows,
            "cols_total": binding.base28.cols_total,
        },
        "refresh_enabled": False,
    }
    if identity != expected:
        _fail(context, "frozen prebuilt pair identity changed")
    return {
        "generation_root": str(generation_root.resolve()),
        "identity": expected,
    }


def require_canonical_unified_replay_producer_evidence(
    value: object,
    *,
    proof: Mapping[str, Any],
    context: str,
    verify_source_files: bool,
) -> dict[str, Any]:
    """Revalidate the producer-owned inputs behind one full-TEST replay."""

    evidence = _exact_keys(
        value,
        _CANONICAL_UNIFIED_REPLAY_PRODUCER_KEYS,
        context=context,
    )
    if (
        evidence["schema_version"]
        != CANONICAL_UNIFIED_REPLAY_PRODUCER_SCHEMA_VERSION
        or evidence["producer_contract"]
        != CANONICAL_UNIFIED_REPLAY_PRODUCER_CONTRACT
        or evidence["decision"] != "PASS"
        or evidence["failures"] != []
    ):
        _fail(context, "canonical producer evidence is not exact PASS")
    canonical_oos = _source_binding(
        evidence["canonical_oos_rows"],
        context=f"{context}.canonical_oos_rows",
        verify_file=verify_source_files,
    )
    oos_proof_binding = require_immutable_json_binding(
        proof["oos_proof_artifact"],
        event_prefix="ENTRY_MODEL_NATIVE_SIZING_OOS_PROOF",
        context=f"{context}.oos_proof_binding",
        verify_file=True,
    )
    oos_proof = _read_bound_json_exact(
        oos_proof_binding,
        context=f"{context}.oos_proof",
    )
    expected_oos = _source_binding(
        oos_proof["source_bindings"]["oos_rows"],
        context=f"{context}.proof_oos_rows",
        verify_file=verify_source_files,
    )
    if canonical_oos != expected_oos:
        _fail(context, "canonical OOS binding differs from proof")
    require_candidate_bundle_authority(
        evidence["candidate_bundle_authority"],
        evaluation_bundle=proof["evaluation_bundle"],
        context=f"{context}.candidate_bundle_authority",
        verify_files=verify_source_files,
    )
    if evidence["candidate_bundle_authority"] != proof[
        "candidate_bundle_authority"
    ]:
        _fail(context, "candidate bundle authority differs from producer evidence")
    producer_replay = _source_binding(
        evidence["replay_rows"],
        context=f"{context}.replay_rows",
        verify_file=verify_source_files,
    )
    proof_replay = _source_binding(
        proof["replay_rows"],
        context=f"{context}.proof_replay_rows",
        verify_file=verify_source_files,
    )
    producer_trace = _source_binding(
        evidence["exit_trace_rows"],
        context=f"{context}.exit_trace_rows",
        verify_file=verify_source_files,
    )
    proof_trace = _source_binding(
        proof["exit_trace_rows"],
        context=f"{context}.proof_exit_trace_rows",
        verify_file=verify_source_files,
    )
    if producer_replay != proof_replay or producer_trace != proof_trace:
        _fail(context, "producer output bindings differ from joint proof")
    source_tape = _source_binding(
        evidence["source_tape"],
        context=f"{context}.source_tape",
        verify_file=verify_source_files,
    )
    oos_source_binding = require_immutable_json_binding(
        oos_proof["oos_source_artifact"],
        event_prefix="ENTRY_MODEL_NATIVE_SIZING_OOS_SOURCE",
        context=f"{context}.oos_source_binding",
        verify_file=True,
    )
    oos_source = _read_bound_json_exact(
        oos_source_binding,
        context=f"{context}.oos_source",
    )
    expected_tape = _source_binding(
        {
            "path": oos_source["source_tape"]["path"],
            "sha256": oos_source["source_tape"]["sha256"],
        },
        context=f"{context}.proof_source_tape",
        verify_file=verify_source_files,
    )
    if source_tape != expected_tape:
        _fail(context, "SourceTape binding differs from canonical OOS source")
    runtime_predictions = _source_binding(
        evidence["runtime_predictions"],
        context=f"{context}.runtime_predictions",
        verify_file=verify_source_files,
    )
    report_binding = require_immutable_json_binding(
        evidence["prediction_report_artifact"],
        event_prefix="ENTRY_CANDIDATE_SELECTIVE_EDGE",
        context=f"{context}.prediction_report",
        verify_file=verify_source_files,
    )
    provenance = proof["test_prediction_provenance"]
    if (
        evidence["prediction_provenance"] != provenance
        or report_binding != provenance["prediction_report_artifact"]
    ):
        _fail(context, "prediction provenance differs from OOS proof")
    if verify_source_files:
        from gx1.execution.model_native_entry_replay_v1 import SourceTape
        from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
            resolve_and_validate_prediction_evidence,
        )

        tape = SourceTape.load(Path(source_tape["path"]))
        if tape.source_sha256 != source_tape["sha256"]:
            _fail(context, "SourceTape bytes differ from producer evidence")
        report = _read_bound_json_exact(
            report_binding,
            context=f"{context}.prediction_report_payload",
        )
        authoritative, _report, declaration = (
            resolve_and_validate_prediction_evidence(
                Path(runtime_predictions["path"]),
                prediction_report_path=Path(report_binding["json_path"]),
                bundle_dir=Path(str(provenance["bundle_dir"])),
                dataset_dir=Path(str(provenance["dataset_dir"])),
                expected_model="candidate",
                require_runtime_head_evidence=True,
            )
        )
        if (
            str(authoritative.resolve()) != runtime_predictions["path"]
            or sha256_file(authoritative) != runtime_predictions["sha256"]
            or report.get("prediction_evidence") != declaration
            or list(declaration.get("models") or []) != ["candidate"]
            or list(declaration.get("splits") or []) != ["test"]
        ):
            _fail(context, "runtime prediction evidence changed")
    frozen_pair = _require_canonical_frozen_pair(
        evidence["prebuilt_pair"],
        context=f"{context}.prebuilt_pair",
    )
    if evidence["prebuilt_pair"] != frozen_pair:
        _fail(context, "prebuilt pair evidence is noncanonical")
    from gx1.contracts.entry_model_native_state_v2 import (
        require_train_rank_reference_identity_v2,
    )

    try:
        require_train_rank_reference_identity_v2(
            evidence["train_rank_reference"],
            context=f"{context}.train_rank_reference",
            verify_artifact=verify_source_files,
        )
    except RuntimeError as exc:
        _fail(context, f"train rank reference identity invalid: {exc}")
    raw_sources = evidence["producer_source_files"]
    if not isinstance(raw_sources, list):
        _fail(context, "producer source inventory is invalid")
    expected_sources = build_canonical_unified_replay_source_inventory(
        Path(__file__).resolve().parents[2]
    )
    if (
        raw_sources != expected_sources
        or evidence["producer_source_inventory_sha256"]
        != _canonical_sha256(expected_sources)
    ):
        _fail(context, "producer source bytes differ")
    coverage = proof["exit_replay_coverage"]
    for key in ("rows", "trade_rows"):
        value_for_key = evidence[key]
        if (
            isinstance(value_for_key, bool)
            or not isinstance(value_for_key, int)
            or value_for_key != int(coverage[key])
        ):
            _fail(context, f"{key} differs from proof coverage")
    trace_rows = evidence["trace_rows"]
    if (
        isinstance(trace_rows, bool)
        or not isinstance(trace_rows, int)
        or trace_rows < int(coverage["trade_rows"])
        or evidence["first_utc"] != coverage["first_utc"]
        or evidence["last_utc"] != coverage["last_utc"]
    ):
        _fail(context, "producer row/time summary differs from proof")
    if verify_source_files:
        replay_frame = read_bound_parquet_exact(
            producer_replay,
            context=f"{context}.producer_replay_exact",
        )
        trace_frame = read_bound_parquet_exact(
            producer_trace,
            context=f"{context}.producer_trace_exact",
        )
        if "model_direction_index" not in replay_frame.columns:
            _fail(context, "producer replay lacks model direction column")
        directions = _strict_directions(
            replay_frame,
            context=f"{context}.producer_replay_directions",
        )
        if (
            len(replay_frame) != evidence["rows"]
            or int(np.count_nonzero(directions != 2)) != evidence["trade_rows"]
            or len(trace_frame) != trace_rows
        ):
            _fail(context, "producer output row summary differs from bound files")
        if (
            set(trace_frame["closed_m1_source_path"].astype(str))
            != {source_tape["path"]}
            or set(
                trace_frame["closed_m1_source_sha256"].astype(str).str.lower()
            )
            != {source_tape["sha256"]}
        ):
            _fail(context, "Exit trace source rows differ from bound SourceTape")
    return dict(evidence)


def require_canonical_unified_replay_launch_authority(
    proof: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Require producer-owned full-TEST rows from the exact candidate bundle."""

    evidence = proof.get("canonical_unified_replay_producer")
    if evidence is None:
        _fail(
            context,
            "canonical unified replay producer is absent; caller-supplied "
            "replay/trace rows have zero launch authority",
        )
    require_canonical_unified_replay_producer_evidence(
        evidence,
        proof=proof,
        context=f"{context}.canonical_producer",
        verify_source_files=True,
    )


def load_bound_joint_exit_sizing_proof(
    binding: Mapping[str, Any] | Any,
    *,
    context: str,
    verify_source_files: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load and independently recompute one newest immutable joint proof."""

    try:
        canonical_binding = require_immutable_json_binding(
            binding,
            event_prefix=MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_EVENT_PREFIX,
            context=f"{context}.binding",
            verify_file=True,
        )
        path = Path(canonical_binding["json_path"])
        observed = _exact_keys(
            _read_bound_json_exact(canonical_binding, context=f"{context}.event"),
            _PROOF_KEYS,
            context=context,
        )
        if observed["schema_version"] != MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_SCHEMA_VERSION:
            _fail(context, "schema_version mismatch")
        created = _utc(observed["created_utc"], context=f"{context}.created_utc")
        if Path(str(observed["json_path"] or "")).expanduser().resolve() != path:
            _fail(context, "json_path self-reference mismatch")
        if observed["decision"] != "PASS" or observed["failures"] != []:
            _fail(context, "joint proof must be zero-failure PASS")
        if observed["replay_contract"] != MODEL_NATIVE_JOINT_EXIT_SIZING_REPLAY_CONTRACT:
            _fail(context, "replay contract mismatch")
        if observed["risk_policy"] != sizing_risk_policy_metadata():
            _fail(context, "risk policy mismatch")
        calibration, calibration_binding = load_bound_sizing_calibration(
            observed["calibration_artifact"],
            context=f"{context}.calibration",
            verify_lineage_files=verify_source_files,
        )
        if calibration_binding != observed["calibration_artifact"]:
            _fail(context, "calibration binding canonicalization mismatch")
        proof, proof_binding = load_bound_sizing_oos_proof(
            observed["oos_proof_artifact"],
            calibration=calibration,
            calibration_artifact_sha256=calibration_binding["sha256"],
            context=f"{context}.oos_proof",
            verify_source_files=verify_source_files,
        )
        if proof_binding != observed["oos_proof_artifact"]:
            _fail(context, "OOS proof binding canonicalization mismatch")
        if _utc(proof["created_utc"], context=f"{context}.oos_created") >= created:
            _fail(context, "joint proof must be newer than OOS proof")
        if (
            observed["evaluation_bundle"] != proof["evaluation_bundle"]
            or observed["test_prediction_provenance"]
            != proof["test_prediction_provenance"]
        ):
            _fail(context, "joint proof lineage differs from canonical OOS proof")
        bundle_authority = require_candidate_bundle_authority(
            observed["candidate_bundle_authority"],
            evaluation_bundle=proof["evaluation_bundle"],
            context=f"{context}.candidate_bundle_authority",
            verify_files=verify_source_files,
        )
        replay_binding = _source_binding(
            observed["replay_rows"],
            context=f"{context}.replay_rows",
            verify_file=verify_source_files,
        )
        trace_binding = _source_binding(
            observed["exit_trace_rows"],
            context=f"{context}.exit_trace_rows",
            verify_file=verify_source_files,
        )
        replay_rows = read_bound_parquet_exact(
            replay_binding,
            context=f"{context}.replay_rows_exact",
        )
        exit_trace_rows = read_bound_parquet_exact(
            trace_binding,
            context=f"{context}.exit_trace_rows_exact",
        )
        canonical_oos_binding = _source_binding(
            proof["source_bindings"]["oos_rows"],
            context=f"{context}.canonical_oos_rows",
            verify_file=verify_source_files,
        )
        canonical_oos_rows = read_bound_parquet_exact(
            canonical_oos_binding,
            context=f"{context}.canonical_oos_rows_exact",
        )
        require_joint_replay_extends_canonical_oos_rows(
            canonical_oos_rows=canonical_oos_rows,
            replay_rows=replay_rows,
            context=f"{context}.canonical_oos_identity",
        )
        coverage = recompute_joint_exit_replay_coverage(
            replay_rows,
            exit_trace_rows=exit_trace_rows,
            candidate_bundle_sha256=bundle_authority[
                "bundle_commit_sha256"
            ],
            context=f"{context}.coverage",
        )
        if _exact_keys(
            observed["exit_replay_coverage"],
            _EXIT_COVERAGE_KEYS,
            context=f"{context}.exit_replay_coverage",
        ) != coverage:
            _fail(context, "reported Exit replay coverage differs from rows")
        recomputed = recompute_sizing_oos_evidence(
            calibration=calibration,
            source_bindings={"oos_rows": replay_binding},
            evaluation_bundle=proof["evaluation_bundle"],
            context=f"{context}.sizing_recompute",
            fact_provenance_mode=MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE,
            extra_row_columns=MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS,
            outcome_price_mode="model_exit_fill",
        )
        mismatched = [
            name for name in _RECOMPUTED_SECTION_NAMES
            if observed[name] != recomputed[name]
        ]
        if mismatched:
            _fail(context, f"reported sizing evidence differs from rows: {mismatched}")
        for name in _RECOMPUTED_SECTION_NAMES[1:]:
            section = observed[name]
            if not isinstance(section, Mapping) or section.get("decision") != "PASS":
                _fail(context, f"{name} must be row-recomputed PASS")
        producer_evidence = observed["canonical_unified_replay_producer"]
        if producer_evidence is not None:
            require_canonical_unified_replay_producer_evidence(
                producer_evidence,
                proof=observed,
                context=f"{context}.canonical_producer",
                verify_source_files=verify_source_files,
            )
        return observed, canonical_binding
    except ModelNativeSizingContractError as exc:
        raise ModelNativeSizingExecutionContractError(str(exc)) from exc


def recompute_runtime_sizing_parity_coverage(
    frame: pd.DataFrame,
    *,
    calibration: Mapping[str, Any],
    adoption: Mapping[str, Any],
    adoption_sha256: str,
    event_created_utc: Any,
    context: str,
) -> dict[str, Any]:
    """Recompute shadow runtime sizing outputs from exact live broker facts."""

    if set(frame.columns) != set(MODEL_NATIVE_SIZING_RUNTIME_PARITY_COLUMNS):
        _fail(
            context,
            "runtime parity columns mismatch: "
            f"missing={sorted(MODEL_NATIVE_SIZING_RUNTIME_PARITY_COLUMNS - set(frame.columns))} "
            f"unexpected={sorted(set(frame.columns) - MODEL_NATIVE_SIZING_RUNTIME_PARITY_COLUMNS)}",
        )
    if len(frame) < MODEL_NATIVE_SIZING_RUNTIME_PARITY_MIN_ROWS:
        _fail(context, "runtime parity has insufficient broker-live observations")
    times = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if times.isna().any() or times.duplicated().any() or not times.is_monotonic_increasing:
        _fail(context, "runtime parity times must be unique monotonic UTC")
    event_created = _utc(event_created_utc, context=f"{context}.event_created_utc")
    if times.iloc[-1] > event_created or (
        event_created - times.iloc[-1]
    ).total_seconds() > MODEL_NATIVE_SIZING_RUNTIME_PARITY_MAX_EVENT_LAG_SECONDS:
        _fail(context, "runtime parity observations are stale relative to the event")
    if (
        times.iloc[-1] - times.iloc[0]
    ).total_seconds() > MODEL_NATIVE_SIZING_RUNTIME_PARITY_MAX_AGE_SECONDS:
        _fail(context, "runtime parity observation window exceeds the maximum age")
    directions = _strict_directions(frame, context=context)
    class_counts = [int(np.count_nonzero(directions == index)) for index in range(3)]
    if min(class_counts) < MODEL_NATIVE_SIZING_RUNTIME_PARITY_MIN_ROWS_PER_CLASS:
        _fail(context, "runtime parity lacks LONG/SHORT/FLAT support")
    logits = pd.to_numeric(frame["position_size_logit"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(logits).all() or float(np.std(logits)) <= 1e-8:
        _fail(context, "runtime parity position_size_logit is not live")
    expected_adoption_sha = _sha(
        adoption_sha256, context=f"{context}.adoption_sha256"
    )
    lineage_sets = {
        "runtime_bundle_metadata_sha256": adoption["bundle_metadata_sha256"],
        "runtime_model_state_dict_sha256": adoption["model_state_dict_sha256"],
        "runtime_adoption_sha256": expected_adoption_sha,
    }
    for field, expected in lineage_sets.items():
        if set(frame[field].astype(str).str.lower()) != {str(expected).lower()}:
            _fail(context, f"{field} differs from adopted runtime identity")
    float_max_error = 0.0
    integer_mismatches = 0
    boolean_mismatches = 0
    direction_mismatches = 0
    transaction_ids: list[str] = []
    for position, (_, row) in enumerate(frame.iterrows()):
        constraints = {
            key: row[key] for key in MODEL_NATIVE_SIZING_RUNTIME_CONSTRAINT_KEYS
        }
        if str(constraints["fact_provenance_mode"]) != "broker_live":
            _fail(context, f"row {position} does not use broker_live facts")
        row_time = pd.Timestamp(times.iloc[position]).isoformat()
        if pd.Timestamp(constraints["sizing_decision_utc"]).isoformat() != row_time:
            _fail(
                context,
                f"row {position} decision time differs from observation time",
            )
        transformed = calibrated_sizing_transform(
            calibration=calibration,
            position_size_logit=logits[position],
            model_direction_index=int(directions[position]),
            runtime_constraints=constraints,
            context=f"{context}.row[{position}]",
        )
        for field in _RUNTIME_PARITY_FLOAT_FIELDS:
            observed_value = float(row[field])
            expected_value = float(transformed[field])
            if not np.isfinite(observed_value):
                _fail(context, f"row {position} {field} is non-finite")
            float_max_error = max(float_max_error, abs(observed_value - expected_value))
        for field in _RUNTIME_PARITY_INT_FIELDS:
            observed_value = row[field]
            if (
                isinstance(observed_value, bool)
                or not float(observed_value).is_integer()
                or int(observed_value) != int(transformed[field])
            ):
                integer_mismatches += 1
        if not isinstance(row["authorized_order"], (bool, np.bool_)):
            _fail(
                context,
                f"row {position} authorized_order is not an exact boolean",
            )
        observed_authorized = bool(row["authorized_order"])
        if observed_authorized != bool(transformed["authorized_order"]):
            boolean_mismatches += 1
        observed_reason = None if pd.isna(row["no_order_reason"]) else str(
            row["no_order_reason"]
        )
        if observed_reason != transformed["no_order_reason"]:
            boolean_mismatches += 1
        direction_after = row["direction_after_sizing"]
        if (
            isinstance(direction_after, (bool, np.bool_))
            or not float(direction_after).is_integer()
        ):
            _fail(
                context,
                f"row {position} direction_after_sizing is not an integer",
            )
        if int(direction_after) != int(directions[position]):
            direction_mismatches += 1
        transaction_ids.append(str(constraints["account_last_transaction_id"]))
    if not frame["order_submitted"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        _fail(context, "order_submitted must contain exact booleans")
    order_submitted = frame["order_submitted"].to_numpy(dtype=bool)
    order_submission_count = int(np.count_nonzero(order_submitted))
    if float_max_error > 1e-12:
        _fail(context, f"runtime sizing float parity error={float_max_error}")
    if integer_mismatches or boolean_mismatches or direction_mismatches:
        _fail(
            context,
            "runtime sizing parity mismatch: "
            f"integer={integer_mismatches} boolean={boolean_mismatches} "
            f"direction={direction_mismatches}",
        )
    if order_submission_count:
        _fail(context, "runtime parity must be shadow-only and submit no order")
    distinct_transaction_ids = len(set(transaction_ids))
    if distinct_transaction_ids < 2:
        _fail(context, "runtime parity requires at least two broker snapshots")
    utc_ns = times.astype("int64").to_numpy(dtype=np.int64)
    return {
        "rows": int(len(frame)),
        "long_rows": class_counts[0],
        "short_rows": class_counts[1],
        "flat_rows": class_counts[2],
        "first_utc": times.iloc[0].isoformat(),
        "last_utc": times.iloc[-1].isoformat(),
        "utc_ns_sha256": hashlib.sha256(utc_ns.tobytes()).hexdigest(),
        "transaction_id_sequence_sha256": hashlib.sha256(
            json.dumps(transaction_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "distinct_transaction_ids": distinct_transaction_ids,
        "max_float_abs_error": float_max_error,
        "integer_mismatch_count": integer_mismatches,
        "boolean_mismatch_count": boolean_mismatches,
        "direction_mismatch_count": direction_mismatches,
        "order_submission_count": order_submission_count,
    }


def load_bound_runtime_sizing_parity(
    binding: Mapping[str, Any] | Any,
    *,
    adoption: Mapping[str, Any],
    calibration: Mapping[str, Any],
    adoption_artifact: Mapping[str, Any],
    context: str,
    verify_source_files: bool,
    now_utc: Any | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load one fresh post-adoption broker-live shadow parity event."""

    try:
        canonical_binding = require_immutable_json_binding(
            binding,
            event_prefix=MODEL_NATIVE_SIZING_RUNTIME_PARITY_EVENT_PREFIX,
            context=f"{context}.binding",
            verify_file=True,
        )
        path = Path(canonical_binding["json_path"])
        observed = _exact_keys(
            _read_bound_json_exact(canonical_binding, context=f"{context}.event"),
            _RUNTIME_PARITY_EVENT_KEYS,
            context=context,
        )
        if observed["schema_version"] != MODEL_NATIVE_SIZING_RUNTIME_PARITY_SCHEMA_VERSION:
            _fail(context, "runtime parity schema_version mismatch")
        created = _utc(observed["created_utc"], context=f"{context}.created_utc")
        if Path(str(observed["json_path"] or "")).expanduser().resolve() != path:
            _fail(context, "runtime parity json_path self-reference mismatch")
        if observed["decision"] != "PASS" or observed["failures"] != []:
            _fail(context, "runtime parity must be zero-failure PASS")
        if observed["parity_contract"] != MODEL_NATIVE_SIZING_RUNTIME_PARITY_CONTRACT:
            _fail(context, "runtime parity contract mismatch")
        canonical_adoption = require_immutable_json_binding(
            adoption_artifact,
            event_prefix="ENTRY_MODEL_NATIVE_SIZING_ADOPTION",
            context=f"{context}.adoption.binding",
            verify_file=True,
        )
        if observed["adoption_artifact"] != canonical_adoption:
            _fail(context, "runtime parity adoption binding mismatch")
        adoption_created = _utc(
            adoption["created_utc"], context=f"{context}.adoption.created_utc"
        )
        if created <= adoption_created:
            _fail(context, "runtime parity must be strictly post-adoption")
        now = _utc(
            pd.Timestamp.now(tz="UTC") if now_utc is None else now_utc,
            context=f"{context}.now_utc",
        )
        age = (now - created).total_seconds()
        if age < 0.0 or age > MODEL_NATIVE_SIZING_RUNTIME_PARITY_MAX_AGE_SECONDS:
            _fail(context, f"runtime parity event age_seconds={age} is invalid")
        expected_bundle_identity = {
            key: adoption[key]
            for key in (
                "bundle_dir",
                "bundle_metadata_path",
                "bundle_metadata_sha256",
                "master_transformer_lock_path",
                "master_transformer_lock_sha256",
                "model_state_dict_path",
                "model_state_dict_sha256",
            )
        }
        if observed["bundle_identity"] != expected_bundle_identity:
            _fail(context, "runtime parity bundle identity mismatch")
        observations = _source_binding(
            observed["observations"],
            context=f"{context}.observations",
            verify_file=verify_source_files,
        )
        frame = read_bound_parquet_exact(
            observations,
            context=f"{context}.observations_exact",
        )
        coverage = recompute_runtime_sizing_parity_coverage(
            frame,
            calibration=calibration,
            adoption=adoption,
            adoption_sha256=canonical_adoption["sha256"],
            event_created_utc=created,
            context=f"{context}.coverage",
        )
        if _exact_keys(
            observed["coverage"],
            _RUNTIME_PARITY_COVERAGE_KEYS,
            context=f"{context}.reported_coverage",
        ) != coverage:
            _fail(context, "reported runtime parity coverage differs from observations")
        return observed, canonical_binding
    except ModelNativeSizingContractError as exc:
        raise ModelNativeSizingExecutionContractError(str(exc)) from exc


__all__ = [
    "CANONICAL_UNIFIED_REPLAY_PRODUCER_CONTRACT",
    "CANONICAL_UNIFIED_REPLAY_PRODUCER_SCHEMA_VERSION",
    "MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE",
    "MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS",
    "MODEL_NATIVE_JOINT_EXIT_SIZING_MIN_TRADES",
    "MODEL_NATIVE_JOINT_EXIT_SIZING_MIN_TRADES_PER_SIDE",
    "MODEL_NATIVE_JOINT_EXIT_MAX_LIVE_TRADES",
    "MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_EVENT_PREFIX",
    "MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_SCHEMA_VERSION",
    "MODEL_NATIVE_JOINT_EXIT_SIZING_REPLAY_CONTRACT",
    "MODEL_NATIVE_JOINT_EXIT_SIZING_ROW_COLUMNS",
    "MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS",
    "MODEL_NATIVE_SIZING_RUNTIME_PARITY_COLUMNS",
    "MODEL_NATIVE_SIZING_RUNTIME_PARITY_CONTRACT",
    "MODEL_NATIVE_SIZING_RUNTIME_PARITY_EVENT_PREFIX",
    "MODEL_NATIVE_SIZING_RUNTIME_PARITY_MAX_AGE_SECONDS",
    "MODEL_NATIVE_SIZING_RUNTIME_PARITY_MIN_ROWS",
    "MODEL_NATIVE_SIZING_RUNTIME_PARITY_SCHEMA_VERSION",
    "ModelNativeSizingExecutionContractError",
    "UNIFIED_CANDIDATE_BUNDLE_AUTHORITY_SCHEMA_VERSION",
    "build_canonical_unified_replay_source_inventory",
    "candidate_bundle_authority",
    "canonical_unified_replay_source_code_files",
    "joint_exit_trace_sha256",
    "load_bound_joint_exit_sizing_proof",
    "load_bound_runtime_sizing_parity",
    "recompute_joint_exit_replay_coverage",
    "require_candidate_bundle_authority",
    "require_canonical_unified_replay_launch_authority",
    "require_canonical_unified_replay_producer_evidence",
    "require_joint_exit_portfolio_capacity",
    "require_joint_replay_extends_canonical_oos_rows",
    "recompute_runtime_sizing_parity_coverage",
]
