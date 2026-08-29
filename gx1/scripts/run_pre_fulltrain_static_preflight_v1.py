#!/usr/bin/env python3
"""Fail-closed static preflight for the Entry external-training candidate.

This command deliberately accepts only TRAIN and VAL dataset paths.  It never
discovers split files by glob and it treats a TEST-like path, or any timestamp
at/after the TEST boundary, as a hard failure.  The output is technical
evidence only: it does not train, tune, evaluate, or select a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch

from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_model_native_joint_task_weighting_v1 import (
    JOINT_TASK_NAMES,
    require_joint_task_weighting_metadata,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    model_native_mandatory_full_stack_metadata,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_TRAINING_SPECIALISTS,
)


SCHEMA_VERSION = "gx1_pre_fulltrain_static_preflight_v1"
TEST_BOUNDARY_UTC = "2026-07-01T00:00:00+00:00"
TRAIN_START_UTC = "2021-06-01T00:00:00+00:00"
TRAIN_END_UTC = "2025-06-01T00:00:00+00:00"
VAL_START_UTC = TRAIN_END_UTC
VAL_END_UTC = TEST_BOUNDARY_UTC


class PreflightError(RuntimeError):
    """A preflight control failed before any model-related action occurred."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_iso(value: Any) -> str:
    result = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PreflightError("[PREFLIGHT_TIMEZONE_REQUIRED]")
    return parsed.astimezone(timezone.utc)


def _reject_test_like_path(path: Path, *, label: str) -> None:
    resolved = path.resolve()
    lowered = resolved.name.lower()
    # This is intentionally conservative: a preflight caller must use the
    # explicit train/val split artefacts rather than a generic or TEST path.
    if (
        "_test" in lowered
        or "-test" in lowered
        or any(part.lower() == "test" for part in resolved.parts)
    ):
        raise PreflightError(f"[PREFLIGHT_TEST_PATH_REJECTED] label={label}")


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _reject_test_like_path(path, label=label)
    if not path.is_file() or path.is_symlink():
        raise PreflightError(f"[PREFLIGHT_ARTIFACT_INVALID] label={label}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreflightError(f"[PREFLIGHT_ARTIFACT_JSON_INVALID] label={label}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"[PREFLIGHT_ARTIFACT_JSON_INVALID] label={label}")
    return value


def _artifact_binding(path: Path, *, label: str) -> dict[str, str]:
    _reject_test_like_path(path, label=label)
    if not path.is_file() or path.is_symlink():
        raise PreflightError(f"[PREFLIGHT_ARTIFACT_INVALID] label={label}")
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def scan_allowed_split(
    path: Path,
    *,
    label: str,
    nominal_start_utc: str,
    nominal_end_utc: str,
    test_boundary_utc: str = TEST_BOUNDARY_UTC,
) -> dict[str, Any]:
    """Read only an explicit TRAIN/VAL parquet and prove its time boundary."""

    _reject_test_like_path(path, label=label)
    if label not in {"train", "val"}:
        raise PreflightError("[PREFLIGHT_SPLIT_LABEL_INVALID]")
    if not path.is_file() or path.is_symlink():
        raise PreflightError(f"[PREFLIGHT_SPLIT_PATH_INVALID] label={label}")
    pf = pq.ParquetFile(path)
    names = tuple(pf.schema_arrow.names)
    required = {"time", "label_horizon_bars"}
    missing = sorted(required - set(names))
    if missing:
        raise PreflightError(
            f"[PREFLIGHT_SPLIT_COLUMNS_MISSING] label={label} missing={missing}"
        )

    nominal_start = _parse_utc(nominal_start_utc)
    nominal_end = _parse_utc(nominal_end_utc)
    test_boundary = _parse_utc(test_boundary_utc)
    previous: datetime | None = None
    first: datetime | None = None
    last: datetime | None = None
    rows = duplicates = non_monotonic = forbidden_rows = horizon_violations = 0
    max_horizon = -1
    label_columns = sorted(name for name in names if name.startswith("y_"))

    for batch in pf.iter_batches(columns=["time", "label_horizon_bars"], batch_size=8192):
        times = batch.column(0).to_pylist()
        horizons = batch.column(1).to_numpy(zero_copy_only=False)
        for raw_time, raw_horizon in zip(times, horizons, strict=True):
            if raw_time is None:
                raise PreflightError(f"[PREFLIGHT_TIMESTAMP_NULL] label={label}")
            current = _parse_utc(_utc_iso(raw_time))
            if current >= test_boundary:
                forbidden_rows += 1
            if previous is not None:
                if current == previous:
                    duplicates += 1
                elif current < previous:
                    non_monotonic += 1
            previous = current
            first = current if first is None else first
            last = current
            try:
                horizon = int(raw_horizon)
            except (TypeError, ValueError) as exc:
                raise PreflightError(f"[PREFLIGHT_HORIZON_INVALID] label={label}") from exc
            max_horizon = max(max_horizon, horizon)
            if horizon < 0 or horizon > MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS:
                horizon_violations += 1
            rows += 1

    if rows == 0 or first is None or last is None:
        raise PreflightError(f"[PREFLIGHT_SPLIT_EMPTY] label={label}")
    if first < nominal_start or last >= nominal_end:
        raise PreflightError(
            f"[PREFLIGHT_SPLIT_BOUNDARY_INVALID] label={label} "
            f"first={_utc_iso(first)} last={_utc_iso(last)}"
        )
    if duplicates or non_monotonic or forbidden_rows or horizon_violations:
        raise PreflightError(
            f"[PREFLIGHT_SPLIT_INTEGRITY_INVALID] label={label} duplicates={duplicates} "
            f"non_monotonic={non_monotonic} test_rows={forbidden_rows} "
            f"horizon_violations={horizon_violations}"
        )
    return {
        "label": label,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "rows": rows,
        "row_groups": pf.metadata.num_row_groups,
        "effective_start_utc": _utc_iso(first),
        "effective_end_utc_inclusive": _utc_iso(last),
        "nominal_start_utc": nominal_start_utc,
        "nominal_end_utc_exclusive": nominal_end_utc,
        "time_monotonic": True,
        "duplicate_timestamp_count": 0,
        "timestamps_at_or_after_test_boundary": 0,
        "max_label_horizon_bars_observed": max_horizon,
        "label_horizon_violations": 0,
        "label_columns": label_columns,
        "schema": str(pf.schema_arrow),
    }


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _environment_metadata() -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "cuda_available": cuda_available,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "operating_system": platform.platform(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "git_dirty_paths": (
            _git_value("status", "--porcelain=v1", "--untracked-files=all") or ""
        ).splitlines(),
    }


def inspect_mtf_cache_test_boundary(cache_manifest: Path) -> dict[str, Any]:
    """Inspect only cache metadata; never map, hash, or read cache arrays."""

    manifest = _read_json(cache_manifest, label="multi_tf_cache_manifest")
    frames = manifest.get("tfs")
    expected_frames = {"M5", "M15", "H1", "H4", "D1"}
    if not isinstance(frames, Mapping) or set(frames) != expected_frames:
        raise PreflightError("[PREFLIGHT_MTF_CACHE_TIMEFRAME_MANIFEST_INVALID]")
    boundary_ns = int(_parse_utc(TEST_BOUNDARY_UTC).timestamp() * 1_000_000_000)
    rows: dict[str, dict[str, Any]] = {}
    exposed: list[str] = []
    for timeframe in sorted(expected_frames):
        row = frames[timeframe]
        if not isinstance(row, Mapping):
            raise PreflightError("[PREFLIGHT_MTF_CACHE_TIMEFRAME_MANIFEST_INVALID]")
        last_ns = row.get("last_ts_ns")
        if isinstance(last_ns, bool) or not isinstance(last_ns, int):
            raise PreflightError("[PREFLIGHT_MTF_CACHE_TIMEFRAME_MANIFEST_INVALID]")
        rows[timeframe] = {
            "last_timestamp_utc": _utc_iso(datetime.fromtimestamp(last_ns / 1e9, tz=timezone.utc)),
            "strictly_before_test_boundary": last_ns < boundary_ns,
        }
        if last_ns >= boundary_ns:
            exposed.append(timeframe)
    return {
        "manifest": _artifact_binding(cache_manifest, label="multi_tf_cache_manifest"),
        "array_bytes_read": 0,
        "timeframes": rows,
        "test_boundary_utc": TEST_BOUNDARY_UTC,
        "safe_for_strict_preflight": not exposed,
        "test_exposed_timeframes": exposed,
        "required_remediation": (
            None
            if not exposed
            else "provide a separately immutable M5/M15/H1/H4/D1 cache whose declared last timestamps are strictly before the TEST boundary; do not slice or hash the current full-history cache during preflight"
        ),
    }


def inspect_dataset_mtf_cache_binding(
    dataset_manifest: Path,
    *,
    expected_manifest_sha256: str,
    expected_cache_identity_sha256: str,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Prove that a split was emitted from the exact inspected MTF cache.

    A clean cache cannot make an already-emitted dataset clean by itself.  The
    dataset's immutable provenance must bind to the manifest, content identity
    and M5 source of the cache passed to this preflight.  This reads only the
    explicit TRAIN/VAL manifest -- never cache arrays or TEST data.
    """

    manifest = _read_json(dataset_manifest, label="dataset_manifest")
    extra = manifest.get("extra")
    binding = extra.get("multi_tf_cache_binding") if isinstance(extra, Mapping) else None
    expected = {
        "manifest_sha256": expected_manifest_sha256,
        "cache_identity_sha256": expected_cache_identity_sha256,
        "m5_prebuilt_source_sha256": expected_source_sha256,
    }
    observed = {
        key: binding.get(key) if isinstance(binding, Mapping) else None
        for key in expected
    }
    mismatches = [key for key, value in expected.items() if observed[key] != value]
    return {
        "dataset_manifest": _artifact_binding(dataset_manifest, label="dataset_manifest"),
        "expected": expected,
        "observed": observed,
        "matches_inspected_cache": not mismatches,
        "mismatched_fields": mismatches,
        "array_bytes_read": 0,
        "test_accessed": False,
    }


def build_static_preflight(
    *,
    train_parquet: Path,
    val_parquet: Path,
    train_manifest: Path,
    val_manifest: Path,
    feature_audit: Path,
    target_audit: Path,
    liveness_audit: Path,
    specialist_audit: Path,
    execution_audit: Path,
    bundle_metadata: Path,
    multi_tf_cache_manifest: Path,
) -> dict[str, Any]:
    """Perform every no-model, no-TEST control needed before full VAL."""

    train = scan_allowed_split(
        train_parquet,
        label="train",
        nominal_start_utc=TRAIN_START_UTC,
        nominal_end_utc=TRAIN_END_UTC,
    )
    val = scan_allowed_split(
        val_parquet,
        label="val",
        nominal_start_utc=VAL_START_UTC,
        nominal_end_utc=VAL_END_UTC,
    )
    if train["effective_end_utc_inclusive"] >= val["effective_start_utc"]:
        raise PreflightError("[PREFLIGHT_TRAIN_VAL_OVERLAP]")

    source_audits = {
        "feature": _read_json(feature_audit, label="feature_audit"),
        "target": _read_json(target_audit, label="target_audit"),
        "liveness": _read_json(liveness_audit, label="liveness_audit"),
        "specialist": _read_json(specialist_audit, label="specialist_audit"),
        "execution": _read_json(execution_audit, label="execution_audit"),
    }
    failed = sorted(
        name for name, payload in source_audits.items() if payload.get("decision") != "PASS"
    )
    if failed:
        raise PreflightError(f"[PREFLIGHT_SOURCE_AUDIT_NOT_PASS] audits={failed}")
    bundle = _read_json(bundle_metadata, label="bundle_metadata")
    mtf_cache = inspect_mtf_cache_test_boundary(multi_tf_cache_manifest)
    cache_payload = _read_json(multi_tf_cache_manifest, label="multi_tf_cache_manifest")
    cache_identity = cache_payload.get("cache_identity_sha256")
    cache_source_sha256 = cache_payload.get("m5_prebuilt_source_sha256")
    if not isinstance(cache_identity, str) or not isinstance(cache_source_sha256, str):
        raise PreflightError("[PREFLIGHT_MTF_CACHE_BINDING_METADATA_INVALID]")
    dataset_cache_bindings = {
        "train": inspect_dataset_mtf_cache_binding(
            train_manifest,
            expected_manifest_sha256=mtf_cache["manifest"]["sha256"],
            expected_cache_identity_sha256=cache_identity,
            expected_source_sha256=cache_source_sha256,
        ),
        "val": inspect_dataset_mtf_cache_binding(
            val_manifest,
            expected_manifest_sha256=mtf_cache["manifest"]["sha256"],
            expected_cache_identity_sha256=cache_identity,
            expected_source_sha256=cache_source_sha256,
        ),
    }
    datasets_match_mtf_cache = all(
        item["matches_inspected_cache"] for item in dataset_cache_bindings.values()
    )
    normalization = bundle.get("input_normalization")
    if not isinstance(normalization, Mapping) or normalization.get("fit_scope") != "train_only":
        raise PreflightError("[PREFLIGHT_NORMALIZATION_SCOPE_INVALID]")
    fit_proof = bundle.get("input_normalization_fit_population_proof")
    if not isinstance(fit_proof, Mapping):
        raise PreflightError("[PREFLIGHT_NORMALIZATION_PROOF_MISSING]")
    lineage = normalization.get("lineage")
    if not isinstance(lineage, Mapping) or int(lineage.get("val_fit_row_count", -1)) != 0 or int(
        lineage.get("test_fit_row_count", -1)
    ) != 0:
        raise PreflightError("[PREFLIGHT_NORMALIZATION_LEAKAGE]")

    local_feature_layers = model_native_mandatory_full_stack_metadata()
    # The local physical feature registry intentionally has ten implementation
    # layers.  They route into these eight semantic specialist families; the
    # MTF V4 matrix applies the same eight owners across the five clocks.
    if len(MODEL_NATIVE_TRAINING_SPECIALISTS) != 8:
        raise PreflightError("[PREFLIGHT_EIGHT_FAMILY_CONTRACT_INVALID]")
    mtf = bundle.get("multi_tf")
    expected_tf_tokens = [
        f"{timeframe}:{family}"
        for timeframe in ("m15", "h1", "h4", "d1")
        for family in MODEL_NATIVE_TRAINING_SPECIALISTS
    ]
    if (
        not isinstance(mtf, Mapping)
        or mtf.get("matrix_contract") != "HTF_V4_EIGHT_FAMILY_CAUSAL_MATRIX_V20"
        or mtf.get("closed_bar_target_availability") is not True
        or mtf.get("entry_family_tf_token_order") != expected_tf_tokens
        or int(mtf.get("entry_family_tf_gate_width", 0)) != 32
    ):
        raise PreflightError("[PREFLIGHT_MTF_EIGHT_FAMILY_CONTRACT_INVALID]")
    if len(JOINT_TASK_NAMES) != 10:
        raise PreflightError("[PREFLIGHT_TEN_TASK_CONTRACT_INVALID]")
    observed_task_weights = bundle.get("model_native_joint_task_weighting")
    if not isinstance(observed_task_weights, Mapping):
        raise PreflightError("[PREFLIGHT_TASK_WEIGHTING_MISSING]")
    try:
        task_weights = require_joint_task_weighting_metadata(
            observed_task_weights, context="PREFLIGHT"
        )
    except RuntimeError as exc:
        raise PreflightError("[PREFLIGHT_TASK_WEIGHTING_INVALID]") from exc
    aux_contract = model_native_aux_target_contract_metadata()
    if int(aux_contract["max_future_horizon_bars"]) != MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS:
        raise PreflightError("[PREFLIGHT_AUX_HORIZON_CONTRACT_INVALID]")

    bindings = {
        "train_parquet": _artifact_binding(train_parquet, label="train_parquet"),
        "val_parquet": _artifact_binding(val_parquet, label="val_parquet"),
        "train_manifest": _artifact_binding(train_manifest, label="train_manifest"),
        "val_manifest": _artifact_binding(val_manifest, label="val_manifest"),
        "feature_audit": _artifact_binding(feature_audit, label="feature_audit"),
        "target_audit": _artifact_binding(target_audit, label="target_audit"),
        "liveness_audit": _artifact_binding(liveness_audit, label="liveness_audit"),
        "specialist_audit": _artifact_binding(specialist_audit, label="specialist_audit"),
        "execution_audit": _artifact_binding(execution_audit, label="execution_audit"),
        "bundle_metadata": _artifact_binding(bundle_metadata, label="bundle_metadata"),
        "multi_tf_cache_manifest": mtf_cache["manifest"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "PASS"
            if mtf_cache["safe_for_strict_preflight"] and datasets_match_mtf_cache
            else "NO_GO"
        ),
        "test_accessed": False,
        "test_accessed_confirmation": "NO",
        "environment": _environment_metadata(),
        "artifact_bindings": bindings,
        "data_split_audit": {
            "test_boundary_utc": TEST_BOUNDARY_UTC,
            "train": train,
            "val": val,
            "max_forward_horizon_bars": MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
            "purge_embargo": {
                "required_future_label_purge_bars": MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
                "effective_train_tail_excluded_by_dataset": (
                    TRAIN_END_UTC + " is exclusive; emitted TRAIN ends before it"
                ),
                "val_test_guard": "timestamp must be strictly earlier than TEST boundary",
            },
        },
        "feature_audit": {
            "semantic_eight_families": list(MODEL_NATIVE_TRAINING_SPECIALISTS),
            "local_physical_feature_layers": local_feature_layers,
            "mtf_v4": {
                "matrix_contract": mtf["matrix_contract"],
                "closed_bar_target_availability": mtf["closed_bar_target_availability"],
                "entry_family_tf_token_order": mtf["entry_family_tf_token_order"],
            },
            "mtf_cache_test_boundary": mtf_cache,
            "dataset_mtf_cache_binding": dataset_cache_bindings,
            "five_timeframes": ["M5", "M15", "H1", "H4", "D1"],
            "source_audits": {
                name: {"decision": payload["decision"], "sha256": bindings[f"{name}_audit"]["sha256"]}
                for name, payload in source_audits.items()
                if f"{name}_audit" in bindings
            },
        },
        "tasks": {"names": list(JOINT_TASK_NAMES), "weighting": task_weights},
        "label_contract": aux_contract,
        "normalization": {
            "fit_scope": normalization["fit_scope"],
            "fit_population_proof": fit_proof,
            "lineage": dict(lineage),
        },
        "bundle_candidate": {
            "path": str(bundle_metadata.resolve()),
            "sha256": bindings["bundle_metadata"]["sha256"],
            "git_commit": bundle.get("git_commit"),
            "checkpoint_monitor": bundle.get("ckpt_monitor"),
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--liveness-audit", type=Path, required=True)
    parser.add_argument("--specialist-audit", type=Path, required=True)
    parser.add_argument("--execution-audit", type=Path, required=True)
    parser.add_argument("--bundle-metadata", type=Path, required=True)
    parser.add_argument("--multi-tf-cache-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_static_preflight(
            train_parquet=args.train_parquet,
            val_parquet=args.val_parquet,
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            feature_audit=args.feature_audit,
            target_audit=args.target_audit,
            liveness_audit=args.liveness_audit,
            specialist_audit=args.specialist_audit,
            execution_audit=args.execution_audit,
            bundle_metadata=args.bundle_metadata,
            multi_tf_cache_manifest=args.multi_tf_cache_manifest,
        )
    except PreflightError as exc:
        print(f"FATAL: static preflight rejected: {exc}", file=sys.stderr)
        return 2
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    _write_json(output_dir / "preflight_manifest.json", report)
    _write_json(output_dir / "data_split_audit.json", report["data_split_audit"])
    _write_json(output_dir / "feature_audit.json", report["feature_audit"])
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
