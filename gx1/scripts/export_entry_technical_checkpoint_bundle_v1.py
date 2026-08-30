"""Export an attended technical checkpoint as a strict, non-promotable bundle.

This is deliberately a CPU-only, TRAIN/VAL-only preflight tool.  It does not
train, select a candidate, read TEST, paper-trade, or make a serving bundle.
It exists because a technical full-VAL run consumes the online/target pair in
an attended session, whereas a normal bundle contains only the online state.

The exporter binds the online state to that session, rebuilds every
state-derived metadata contract, strict-loads a fresh process-equivalent
bundle, and proves output parity on the exact eight rows preserved by the
completed technical VAL report.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    CORE_ARTIFACTS,
    publish_bundle_directory_noreplace,
    write_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
    joint_task_weighting_metadata,
)
from gx1.contracts.entry_model_native_tf_input_scale_v1 import (
    NEUTRAL_EFFECTIVE_INIT,
    TF_NAMES,
    build_tf_input_scale_contract,
)
from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    PrefreezeTestSealLineageError,
    require_pretest_or_prefreeze_test_guard_lineage,
)
from gx1.contracts.unified_exit_lifecycle_v1 import UnifiedExitLifecycleCorpus
from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset
from gx1.scripts.validate_entry_model_native_technical_checkpoint_v1 import (
    TEST_BOUNDARY_UTC,
    TechnicalValidationError,
    _attended_checkpoint_pair,
    _model_state_sha256,
    _reference_predictions,
    _sha256,
)


SCHEMA_VERSION = "gx1_technical_checkpoint_bundle_parity_v1"
_REFERENCE_NUMERIC_ATOL = 2.0e-5
_REFERENCE_NUMERIC_RTOL = 2.0e-5


class TechnicalCheckpointExportError(RuntimeError):
    """A non-promotable technical checkpoint could not be proved exactly."""


def _regular_absolute(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or any(parent.is_symlink() for parent in candidate.parents)
        or not candidate.is_file()
    ):
        raise TechnicalCheckpointExportError(f"[{label}_PATH_INVALID]")
    return candidate


def _directory_absolute(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or any(parent.is_symlink() for parent in candidate.parents)
        or not candidate.is_dir()
    ):
        raise TechnicalCheckpointExportError(f"[{label}_PATH_INVALID]")
    return candidate


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_absolute(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise TechnicalCheckpointExportError(f"[{label}_JSON_INVALID]") from exc
    if not isinstance(value, dict):
        raise TechnicalCheckpointExportError(f"[{label}_JSON_INVALID]")
    return value


def _atomic_json_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_REPORT_PATH_INVALID]")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _require_val_only_clock(val_parquet: Path) -> dict[str, Any]:
    try:
        times = pd.to_datetime(pd.read_parquet(val_parquet, columns=["time"])["time"], utc=True)
    except Exception as exc:
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_VAL_CLOCK_READ_FAILED]") from exc
    if (
        len(times) == 0
        or bool((times >= TEST_BOUNDARY_UTC).any())
        or bool(times.duplicated().any())
        or not bool(times.is_monotonic_increasing)
    ):
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_VAL_CLOCK_INVALID]")
    return {
        "rows": int(len(times)),
        "start_utc": str(times.iloc[0]),
        "end_utc_inclusive": str(times.iloc[-1]),
        "test_accessed": False,
    }


def _require_pretest_guard(
    *,
    guard_json: Path,
    guard_sha256: str,
    dataset_run_id: str,
    dataset_dir: Path,
) -> None:
    try:
        observed = require_pretest_or_prefreeze_test_guard_lineage(
            guard_json,
            guard_sha256,
            expected_dataset_run_id=dataset_run_id,
            expected_dataset_dir=dataset_dir,
        )
    except (PrefreezeTestSealLineageError, OSError, ValueError) as exc:
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_TEST_GUARD_INVALID]") from exc
    if (
        observed.get("test_accessed") is True
        or observed.get("access_proof", {}).get("test_dataset_bytes_read") is True
    ):
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_TEST_GUARD_OPEN]")


def _compare_reference_frames(
    *,
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    exact_numeric: bool,
    context: str,
) -> dict[str, Any]:
    """Prove rows, columns and every output element without silent coercion."""

    if list(expected.columns) != list(actual.columns) or len(expected) != len(actual):
        raise TechnicalCheckpointExportError(f"[{context}_REFERENCE_SCHEMA_MISMATCH]")
    max_abs_difference = 0.0
    numeric_columns = 0
    for column in expected.columns:
        left = expected[column]
        right = actual[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            lhs = left.to_numpy()
            rhs = right.to_numpy()
            if lhs.dtype.kind in "iu" and rhs.dtype.kind in "iu":
                if not np.array_equal(lhs, rhs):
                    raise TechnicalCheckpointExportError(f"[{context}_REFERENCE_INTEGER_MISMATCH] {column}")
                continue
            lhs_float = np.asarray(lhs, dtype=np.float64)
            rhs_float = np.asarray(rhs, dtype=np.float64)
            if not np.isfinite(lhs_float).all() or not np.isfinite(rhs_float).all():
                raise TechnicalCheckpointExportError(f"[{context}_REFERENCE_NONFINITE] {column}")
            difference = float(np.max(np.abs(lhs_float - rhs_float)))
            max_abs_difference = max(max_abs_difference, difference)
            numeric_columns += 1
            matched = (
                np.array_equal(lhs_float, rhs_float)
                if exact_numeric
                else np.allclose(
                    lhs_float,
                    rhs_float,
                    rtol=_REFERENCE_NUMERIC_RTOL,
                    atol=_REFERENCE_NUMERIC_ATOL,
                )
            )
            if not bool(matched):
                raise TechnicalCheckpointExportError(f"[{context}_REFERENCE_NUMERIC_MISMATCH] {column}")
        elif not left.equals(right):
            raise TechnicalCheckpointExportError(f"[{context}_REFERENCE_VALUE_MISMATCH] {column}")
    return {
        "rows": int(len(expected)),
        "columns": int(len(expected.columns)),
        "numeric_columns": int(numeric_columns),
        "max_abs_difference": float(max_abs_difference),
        "exact_numeric": bool(exact_numeric),
    }


def _bind_historical_reference_schema(
    *,
    historical: pd.DataFrame,
    current: pd.DataFrame,
) -> dict[str, Any]:
    """Bind a CUDA historical probe without pretending CPU kernels are bitwise.

    The completed technical VAL report already binds its CUDA predictions to
    the exact semantic online-state hash.  CPU/CUDA transformer arithmetic is
    not a valid bitwise-parity oracle, so this verifies the deterministic row
    identity/schema here and reserves exact numeric parity for two clean CPU
    reconstructions of the same state below.
    """

    if list(historical.columns) != list(current.columns) or len(historical) != len(current):
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_HISTORICAL_REFERENCE_SCHEMA_MISMATCH]")
    for column in ("time", "entry_row_index"):
        if column not in historical or not historical[column].equals(current[column]):
            raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_HISTORICAL_REFERENCE_ROW_MISMATCH]")
    for frame in (historical, current):
        numeric = frame.select_dtypes(include=[np.number])
        if not bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all()):
            raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_HISTORICAL_REFERENCE_NONFINITE]")
    return {
        "rows": int(len(historical)),
        "columns": int(len(historical.columns)),
        "binding": "same_hash_bound_online_checkpoint__same_deterministic_val_rows__finite_outputs",
        "numeric_cross_device_comparison": "not_claimed__clean_cpu_to_clean_cpu_parity_is_exact",
    }


def _state_bound_metadata(
    *,
    source_metadata: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    model_state: Mapping[str, Any],
    session_contract_sha256: str,
    session_model_state_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy static architecture evidence, then regenerate state-bound fields."""

    metadata = copy.deepcopy(dict(source_metadata))
    lock = copy.deepcopy(dict(source_lock))
    selected_log_variances: dict[str, float] = {}
    for name in JOINT_TASK_NAMES:
        value = model_state.get(f"task_log_variances.{name}")
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_JOINT_TASK_STATE_INVALID]")
        scalar = float(value.item())
        if not math.isfinite(scalar) or scalar == 0.0:
            raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_JOINT_TASK_STATE_INVALID]")
        selected_log_variances[name] = scalar
    weighting = joint_task_weighting_metadata(
        selected_log_variances,
        supervision_observed={name: True for name in JOINT_TASK_NAMES},
        gradient_observed={name: True for name in JOINT_TASK_NAMES},
    )
    learned_raw: dict[str, float] = {}
    for timeframe in TF_NAMES:
        value = model_state.get(f"tf_input_scale_{timeframe}")
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_TF_SCALE_STATE_INVALID]")
        learned_raw[timeframe] = float(value.item())
    tf_input_scale = build_tf_input_scale_contract(
        init_effective={timeframe: NEUTRAL_EFFECTIVE_INIT for timeframe in TF_NAMES},
        learned_raw=learned_raw,
    )
    exported_at = datetime.now(timezone.utc).isoformat()
    technical_provenance = {
        "schema_version": SCHEMA_VERSION,
        "authority": {
            "technical_preflight": True,
            "candidate": False,
            "test": False,
            "promotion": False,
            "paper": False,
            "live": False,
        },
        "source": "hash_bound_attended_technical_checkpoint_online_state",
        "session_contract_sha256": session_contract_sha256,
        "online_model_state_semantic_sha256": session_model_state_sha256,
        "exported_at_utc": exported_at,
    }
    for payload in (metadata, lock):
        payload["execution_tier"] = "attended_only"
        lineage = payload.get("run_lineage")
        if not isinstance(lineage, Mapping):
            raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_RUN_LINEAGE_INVALID]")
        payload["run_lineage"] = dict(lineage)
        payload["run_lineage"]["execution_tier"] = "attended_only"
        payload["model_native_joint_task_weighting"] = copy.deepcopy(weighting)
        payload["tf_input_scale"] = copy.deepcopy(tf_input_scale)
        payload["technical_checkpoint_export"] = copy.deepcopy(technical_provenance)
        payload["created_at_utc"] = exported_at
    return metadata, lock


def _write_bundle_staging(
    *,
    out_bundle_dir: Path,
    model_state: Mapping[str, Any],
    metadata: dict[str, Any],
    lock: dict[str, Any],
) -> Path:
    if out_bundle_dir.exists() or out_bundle_dir.is_symlink() or not out_bundle_dir.is_absolute():
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_OUTPUT_BUNDLE_INVALID]")
    if not out_bundle_dir.parent.is_dir() or out_bundle_dir.parent.is_symlink():
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_OUTPUT_PARENT_INVALID]")
    staging = Path(tempfile.mkdtemp(prefix=f".{out_bundle_dir.name}.staging.", dir=out_bundle_dir.parent))
    try:
        model_path = staging / "model_state_dict.pt"
        torch.save(dict(model_state), model_path)
        state_hash = _sha256(model_path)
        metadata["state_dict_sha256"] = state_hash
        lock["model_sha256"] = state_hash
        (staging / "bundle_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        (staging / "MASTER_TRANSFORMER_LOCK.json").write_text(json.dumps(lock, indent=2, allow_nan=False), encoding="utf-8")
        for name in CORE_ARTIFACTS:
            with (staging / name).open("rb") as handle:
                os.fsync(handle.fileno())
        write_bundle_commit_manifest(
            bundle_dir=staging,
            artifact_names=CORE_ARTIFACTS,
            bundle_kind="trained",
            created_at_utc=str(metadata["created_at_utc"]),
        )
        return staging
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run(
    *,
    source_bundle_dir: Path,
    checkpoint_session_dir: Path,
    train_parquet: Path,
    val_parquet: Path,
    m5_prebuilt: Path,
    multi_tf_cache_dir: Path,
    lifecycle_manifest: Path,
    val_sequence_source_audit: Path,
    test_guard_json: Path,
    test_guard_sha256: str,
    full_val_report: Path,
    out_bundle_dir: Path,
    out_report: Path,
) -> dict[str, Any]:
    source_bundle_dir = _directory_absolute(source_bundle_dir, label="TECHNICAL_EXPORT_SOURCE_BUNDLE")
    checkpoint_session_dir = _directory_absolute(checkpoint_session_dir, label="TECHNICAL_EXPORT_SESSION")
    train_parquet = _regular_absolute(train_parquet, label="TECHNICAL_EXPORT_TRAIN")
    val_parquet = _regular_absolute(val_parquet, label="TECHNICAL_EXPORT_VAL")
    m5_prebuilt = _regular_absolute(m5_prebuilt, label="TECHNICAL_EXPORT_M5")
    lifecycle_manifest = _regular_absolute(lifecycle_manifest, label="TECHNICAL_EXPORT_LIFECYCLE")
    val_sequence_source_audit = _regular_absolute(val_sequence_source_audit, label="TECHNICAL_EXPORT_VAL_SEQUENCE")
    test_guard_json = _regular_absolute(test_guard_json, label="TECHNICAL_EXPORT_TEST_GUARD")
    full_val_report = _regular_absolute(full_val_report, label="TECHNICAL_EXPORT_FULL_VAL_REPORT")
    cache_dir = _directory_absolute(multi_tf_cache_dir, label="TECHNICAL_EXPORT_MTF_CACHE")
    _regular_absolute(cache_dir / "manifest.json", label="TECHNICAL_EXPORT_MTF_MANIFEST")
    val_clock = _require_val_only_clock(val_parquet)
    source_meta = _read_json(source_bundle_dir / "bundle_metadata.json", label="TECHNICAL_EXPORT_SOURCE_METADATA")
    source_lock = _read_json(source_bundle_dir / "MASTER_TRANSFORMER_LOCK.json", label="TECHNICAL_EXPORT_SOURCE_LOCK")
    session_preview = _read_json(checkpoint_session_dir / "ATTENDED_RESEARCH_SESSION_CONTRACT.json", label="TECHNICAL_EXPORT_SESSION_CONTRACT")
    dataset_run_id = session_preview.get("dataset_run_id")
    if not isinstance(dataset_run_id, str) or not dataset_run_id:
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_DATASET_RUN_ID_INVALID]")
    _require_pretest_guard(
        guard_json=test_guard_json,
        guard_sha256=test_guard_sha256,
        dataset_run_id=dataset_run_id,
        dataset_dir=val_parquet.parent,
    )
    full_val = _read_json(full_val_report, label="TECHNICAL_EXPORT_FULL_VAL_REPORT")
    if full_val.get("decision") != "PASS" or full_val.get("test_accessed") is not False:
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_FULL_VAL_REPORT_INVALID]")
    reference_spec = full_val.get("reference_predictions")
    if not isinstance(reference_spec, Mapping):
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_FULL_VAL_REFERENCE_INVALID]")
    saved_reference_path = _regular_absolute(Path(str(reference_spec.get("path") or "")), label="TECHNICAL_EXPORT_SAVED_REFERENCE")
    if _sha256(saved_reference_path) != str(reference_spec.get("sha256") or ""):
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_SAVED_REFERENCE_HASH_INVALID]")
    bundle = load_entry_v10_ctx_bundle(bundle_dir=source_bundle_dir, device="cpu")
    contract, model_state, target_state = _attended_checkpoint_pair(
        session_dir=checkpoint_session_dir,
        bundle_metadata=bundle.metadata,
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        m5_prebuilt=m5_prebuilt,
        lifecycle_manifest=lifecycle_manifest,
    )
    if str(contract.get("source_commit") or "") != str(bundle.metadata.get("git_commit") or ""):
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_SOURCE_COMMIT_MISMATCH]")
    checkpoint = full_val.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("online_model_state_sha256") != _model_state_sha256(model_state):
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_FULL_VAL_CHECKPOINT_MISMATCH]")
    model = bundle.transformer_model
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_MODEL_STATE_INVALID]") from exc
    model.eval()
    os.environ["GX1_V10_MULTI_TF_V4_CACHE_DIR"] = str(cache_dir)
    mtf = bundle.metadata.get("multi_tf")
    if not isinstance(mtf, Mapping):
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_MTF_METADATA_INVALID]")
    per_tf = {tf: int(mtf[f"{tf.lower()}_seq_len"]) for tf in ("M5", "M15", "H1", "H4", "D1")}
    corpus = UnifiedExitLifecycleCorpus(
        root_manifest_path=lifecycle_manifest,
        entry_parquets={"val": val_parquet},
        dataset_run_id=dataset_run_id,
        splits=("val",),
    )
    val_dataset = EntryV10CtxDataset(
        val_parquet,
        seq_len=int(bundle.metadata["seq_len"]),
        m5_prebuilt_path=m5_prebuilt,
        per_tf_seq_lens=per_tf,
        multi_tf_closed_bar=True,
        sequence_source_audit_json=val_sequence_source_audit,
    )
    val_dataset.bind_unified_exit_lifecycle(corpus.splits["val"])
    if len(val_dataset) != int(val_clock["rows"]):
        raise TechnicalCheckpointExportError("[TECHNICAL_EXPORT_VAL_ROWS_MISMATCH]")
    session_contract_sha256 = _sha256(checkpoint_session_dir / "ATTENDED_RESEARCH_SESSION_CONTRACT.json")
    semantic_sha = _model_state_sha256(model_state)
    metadata, lock = _state_bound_metadata(
        source_metadata=source_meta,
        source_lock=source_lock,
        model_state=model_state,
        session_contract_sha256=session_contract_sha256,
        session_model_state_sha256=semantic_sha,
    )
    staging = _write_bundle_staging(
        out_bundle_dir=out_bundle_dir,
        model_state=model_state,
        metadata=metadata,
        lock=lock,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="gx1-tech-bundle-parity-") as temporary:
            temporary_path = Path(temporary)
            checkpoint_reference_path = temporary_path / "checkpoint_reference.parquet"
            exported_reference_path = temporary_path / "exported_reference.parquet"
            _reference_predictions(model=model, dataset=val_dataset, device=torch.device("cpu"), out_path=checkpoint_reference_path)
            clean_bundle = load_entry_v10_ctx_bundle(bundle_dir=staging, device="cpu")
            _reference_predictions(model=clean_bundle.transformer_model, dataset=val_dataset, device=torch.device("cpu"), out_path=exported_reference_path)
            stored_reference = pd.read_parquet(saved_reference_path)
            checkpoint_reference = pd.read_parquet(checkpoint_reference_path)
            exported_reference = pd.read_parquet(exported_reference_path)
            historical_binding = _bind_historical_reference_schema(
                historical=stored_reference,
                current=checkpoint_reference,
            )
            clean_parity = _compare_reference_frames(
                expected=checkpoint_reference,
                actual=exported_reference,
                exact_numeric=True,
                context="TECHNICAL_EXPORT_CLEAN_BUNDLE",
            )
        publish_bundle_directory_noreplace(staging, out_bundle_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS_TECHNICAL_ONLY_NOT_CANDIDATE",
        "test_accessed": False,
        "authority": {
            "technical_preflight": True,
            "candidate": False,
            "test": False,
            "promotion": False,
            "paper": False,
            "live": False,
        },
        "source": {
            "bundle_dir": str(source_bundle_dir),
            "session_dir": str(checkpoint_session_dir),
            "session_contract_sha256": session_contract_sha256,
            "online_model_state_semantic_sha256": semantic_sha,
            "target_model_state_semantic_sha256": _model_state_sha256(target_state),
            "full_val_report": {"path": str(full_val_report), "sha256": _sha256(full_val_report)},
            "saved_reference": {"path": str(saved_reference_path), "sha256": _sha256(saved_reference_path)},
        },
        "inputs": {
            "train_parquet": {"path": str(train_parquet), "sha256": _sha256(train_parquet)},
            "val_parquet": {"path": str(val_parquet), "sha256": _sha256(val_parquet)},
            "m5_prebuilt": {"path": str(m5_prebuilt), "sha256": _sha256(m5_prebuilt)},
            "multi_tf_cache_manifest": {"path": str(cache_dir / "manifest.json"), "sha256": _sha256(cache_dir / "manifest.json")},
            "lifecycle_manifest": {"path": str(lifecycle_manifest), "sha256": _sha256(lifecycle_manifest)},
            "val_sequence_source_audit": {"path": str(val_sequence_source_audit), "sha256": _sha256(val_sequence_source_audit)},
            "test_guard": {"path": str(test_guard_json), "sha256": str(test_guard_sha256)},
        },
        "val_clock": val_clock,
        "exported_bundle": {
            "path": str(out_bundle_dir),
            "state_dict_sha256": _sha256(out_bundle_dir / "model_state_dict.pt"),
            "bundle_commit_sha256": _read_json(out_bundle_dir / "ENTRY_MODEL_NATIVE_BUNDLE_COMMIT.json", label="TECHNICAL_EXPORT_COMMIT")["commit_sha256"],
        },
        "prediction_parity": {
            "historical_technical_val_checkpoint_binding": historical_binding,
            "current_checkpoint_cpu_to_clean_exported_bundle_cpu": clean_parity,
        },
    }
    _atomic_json_new(out_report, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-session-dir", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument("--m5-prebuilt", type=Path, required=True)
    parser.add_argument("--multi-tf-cache-dir", type=Path, required=True)
    parser.add_argument("--lifecycle-manifest", type=Path, required=True)
    parser.add_argument("--val-sequence-source-audit", type=Path, required=True)
    parser.add_argument("--test-guard-json", type=Path, required=True)
    parser.add_argument("--test-guard-sha256", required=True)
    parser.add_argument("--full-val-report", type=Path, required=True)
    parser.add_argument("--out-bundle-dir", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(
        source_bundle_dir=args.source_bundle_dir,
        checkpoint_session_dir=args.checkpoint_session_dir,
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        m5_prebuilt=args.m5_prebuilt,
        multi_tf_cache_dir=args.multi_tf_cache_dir,
        lifecycle_manifest=args.lifecycle_manifest,
        val_sequence_source_audit=args.val_sequence_source_audit,
        test_guard_json=args.test_guard_json,
        test_guard_sha256=str(args.test_guard_sha256),
        full_val_report=args.full_val_report,
        out_bundle_dir=args.out_bundle_dir,
        out_report=args.out_report,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
