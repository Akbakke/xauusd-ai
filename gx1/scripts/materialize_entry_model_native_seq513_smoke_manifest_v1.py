#!/usr/bin/env python3
"""Materialize a report-only model-native seq513 smoke dataset manifest event.

This script verifies an already-built model-native seq513 smoke dataset and writes a
manifest/report under the report directory only. It does not rebuild data, copy
parquets, start training, run replay, distill IQL, or touch shadow/live paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    model_native_signal_contract_failures,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    model_native_readiness_contract_metadata,
)
from gx1.contracts.entry_model_native_train_recipe_v1 import (
    DIRECTION_CONTEXT_SLICE_CONTRACT,
    DIRECTION_DIAGNOSTIC_ENV_TEMPLATE,
    DIRECTION_DIAGNOSTIC_RECIPE_CONTRACT,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    TRAIN_WRAPPER_RELATIVE_PATH,
)
from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    READY_DECISION as POST_REBUILD_READY_DECISION,
    REQUIRED_PROOF_CHECKS as REQUIRED_POST_REBUILD_ORCHESTRATION_CHECKS,
    SCHEMA_VERSION as POST_REBUILD_SCHEMA_VERSION,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.models.entry_v10.direction_decision_contract import (
    model_direction_decision_contract_metadata,
)
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    ENTRY_FITTED_Q_DATASET_STEM_SUFFIX,
)


SPLITS = ("train", "val")
SCHEMA_VERSION = "entry_model_native_seq513_smoke_dataset_v3"
SPLIT_SCHEMA_VERSION = MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION
REPORT_SCHEMA_VERSION = "entry_model_native_seq513_smoke_manifest_v3"
MANIFEST_VARIANT = MODEL_NATIVE_CONTRACT_MODE
EXPECTED_SEQ_SNAP_WIDTH = MODEL_NATIVE_SIGNAL_DIM
DEFAULT_STEM = f"v10_model_native_seq513_smoke{ENTRY_FITTED_Q_DATASET_STEM_SUFFIX}"
SMART_SPECIALIST_CONTRACT_MODE = MODEL_NATIVE_CONTRACT_MODE
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST"
_TIMESTAMPED_JSON_RE = re.compile(
    r"^.+_\d{8}T\d{6}(?:\d{6})?Z\.json$"
)
RAM_CAP_RUNNER = "scripts/gx1_capped_run.sh"
DEFAULT_MEMORY_CAP = "10G"
DEFAULT_SWAP_CAP = "512M"
CANONICAL_DIRECTION_DECISION_CONTRACT = model_direction_decision_contract_metadata()
SIDE_EFFECTS_STARTED = {
    "dataset_rebuild": False,
    "training": False,
    "replay": False,
    "iql_distillation": False,
    "shadow": False,
    "live": False,
}
REQUIRED_POST_REBUILD_SIDE_EFFECT_KEYS = tuple(SIDE_EFFECTS_STARTED)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _artifact_meta(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() else None,
        "sha256": _sha256_file(path),
    }


def _require_timestamped_evidence_path(path: Path, *, label: str) -> None:
    if path.name.endswith("_latest.json") or not _TIMESTAMPED_JSON_RE.fullmatch(
        path.name
    ):
        raise RuntimeError(
            f"{label} must be an explicit timestamped JSON evidence event, got {path}"
        )


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details if details is not None else {}}


def _direction_diagnostic_recipe_ok(
    contract: dict[str, Any],
) -> bool:
    recipe = contract.get("direction_diagnostic_recipe_contract")
    env_template = contract.get("direction_diagnostic_env_template")
    if not isinstance(recipe, dict) or not isinstance(env_template, dict):
        return False
    if recipe != DIRECTION_DIAGNOSTIC_RECIPE_CONTRACT:
        return False
    return all(
        env_template.get(key) == value
        for key, value in DIRECTION_DIAGNOSTIC_ENV_TEMPLATE.items()
    )


def _direction_context_slice_ok(contract: dict[str, Any]) -> bool:
    return (
        contract.get("requires_direction_context_slice_contract") is True
        and contract.get("direction_context_slice_contract") == DIRECTION_CONTEXT_SLICE_CONTRACT
    )


def _canonical_direction_decision_ok(contract: dict[str, Any]) -> bool:
    return (
        contract.get("requires_canonical_direction_decision_contract") is True
        and contract.get("canonical_direction_decision_contract")
        == CANONICAL_DIRECTION_DECISION_CONTRACT
    )


def _entry_run_id_ok(run_id: str) -> bool:
    value = str(run_id or "").strip()
    placeholders = {
        "",
        "<id>",
        "<run_id-id>",
        "<MODEL_NATIVE_SEQ513_SMOKE_RUN_ID_ID>",
        "TODO",
        "TBD",
    }
    return value not in placeholders and "<" not in value and ">" not in value


def _resolve_manifest_output_path(manifest: dict[str, Any]) -> Path | None:
    raw = str(manifest.get("output_data_path") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        return path.resolve() if path.is_absolute() else None
    return None


def _stack_list_column(values: Any, dtype: np.dtype) -> np.ndarray:
    items = list(values)
    if not items:
        return np.asarray([], dtype=dtype)
    try:
        return np.stack(items).astype(dtype, copy=False)
    except ValueError:
        return np.stack([np.stack(item) for item in items]).astype(dtype, copy=False)


def _sample_seq_snap_shapes(path: Path, *, sample_rows: int, batch_size: int) -> dict[str, Any]:
    if not path.exists():
        return {"ok": False, "errors": ["missing parquet"], "rows": 0}
    errors: list[str] = []
    try:
        pf = pq.ParquetFile(path)
    except Exception as exc:
        return {"ok": False, "errors": [f"open parquet failed: {type(exc).__name__}: {exc}"], "rows": 0}
    schema_names = set(pf.schema_arrow.names)
    missing = [name for name in ("seq", "snap") if name not in schema_names]
    if missing:
        return {
            "ok": False,
            "errors": [f"missing columns: {missing}"],
            "rows": int(pf.metadata.num_rows or 0),
            "schema_columns": sorted(schema_names),
        }
    total_rows = int(pf.metadata.num_rows or 0)
    if total_rows <= 0:
        return {"ok": False, "errors": ["empty parquet"], "rows": 0, "schema_columns": sorted(schema_names)}

    scan_limit = min(total_rows, int(sample_rows))
    scanned = 0
    shape_examples: dict[str, Any] = {}
    nonfinite = {"seq": 0, "snap": 0}
    for batch in pf.iter_batches(batch_size=int(batch_size), columns=["seq", "snap"]):
        if scanned >= scan_limit:
            break
        pdf = batch.to_pandas()
        remaining = scan_limit - scanned
        if len(pdf) > remaining:
            pdf = pdf.iloc[:remaining].copy()
        if pdf.empty:
            continue
        try:
            seq = _stack_list_column(pdf["seq"], np.float32)
            snap = _stack_list_column(pdf["snap"], np.float32)
        except Exception as exc:
            errors.append(f"seq/snap stack failed: {type(exc).__name__}: {exc}")
            scanned += int(len(pdf))
            continue
        shape_examples = {"seq": list(seq.shape), "snap": list(snap.shape)}
        if seq.ndim != 3 or int(seq.shape[-1]) != EXPECTED_SEQ_SNAP_WIDTH:
            errors.append(
                f"seq width mismatch got={list(seq.shape)} expected_last_dim={EXPECTED_SEQ_SNAP_WIDTH}"
            )
        if snap.ndim != 2 or int(snap.shape[-1]) != EXPECTED_SEQ_SNAP_WIDTH:
            errors.append(
                f"snap width mismatch got={list(snap.shape)} expected_last_dim={EXPECTED_SEQ_SNAP_WIDTH}"
            )
        if not errors:
            nonfinite["seq"] += int((~np.isfinite(seq)).sum())
            nonfinite["snap"] += int((~np.isfinite(snap)).sum())
        scanned += int(len(pdf))

    if scanned <= 0:
        errors.append("no seq/snap rows scanned")
    if any(count != 0 for count in nonfinite.values()):
        errors.append(f"nonfinite seq/snap values: {nonfinite}")
    return {
        "ok": bool(scanned > 0 and not errors),
        "rows": total_rows,
        "scanned_rows": int(scanned),
        "sample_rows": int(sample_rows),
        "shape_examples": shape_examples,
        "nonfinite_counts": nonfinite,
        "errors": errors,
    }


def _split_summary(
    dataset_dir: Path,
    split: str,
    *,
    parquet_value: str,
    parquet_sha256: str,
    manifest_value: str,
    manifest_sha256: str,
    sample_rows: int,
    batch_size: int,
) -> dict[str, Any]:
    parquet_input = Path(parquet_value).expanduser()
    manifest_input = Path(manifest_value).expanduser()
    parquet_path = parquet_input.resolve()
    manifest_path = manifest_input.resolve()
    paths_exact = bool(
        parquet_input.is_absolute()
        and manifest_input.is_absolute()
        and parquet_input == parquet_path
        and manifest_input == manifest_path
        and not parquet_input.is_symlink()
        and not manifest_input.is_symlink()
        and parquet_path.parent == dataset_dir
        and manifest_path.parent == dataset_dir
        and not any("latest" in part.lower() for part in parquet_path.parts)
        and not any("latest" in part.lower() for part in manifest_path.parts)
    )
    manifest = _read_json_or_empty(manifest_path)
    output_path = _resolve_manifest_output_path(manifest)
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    signal_bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    fields = [str(x) for x in signal_bridge.get("fields", []) if str(x).strip()]
    shape_probe = (
        _sample_seq_snap_shapes(parquet_path, sample_rows=sample_rows, batch_size=batch_size)
        if paths_exact and parquet_path.is_file()
        else {"ok": False, "errors": ["missing output_data_path"], "rows": 0}
    )
    observed_manifest_sha = _sha256_file(manifest_path)
    observed_parquet_sha = _sha256_file(parquet_path)
    hashes_exact = bool(
        _is_sha256(manifest_sha256)
        and _is_sha256(parquet_sha256)
        and observed_manifest_sha == str(manifest_sha256).lower()
        and observed_parquet_sha == str(parquet_sha256).lower()
    )
    return {
        "split": split,
        "parquet_path": str(parquet_path),
        "manifest_path": str(manifest_path),
        "explicit_paths_exact": paths_exact,
        "parquet_exists": bool(parquet_path.is_file()),
        "manifest_exists": bool(manifest_path.is_file()),
        "output_data_path": str(output_path) if output_path is not None else "",
        "output_data_exists": bool(output_path is not None and output_path.exists()),
        "manifest_output_matches_split_parquet": bool(
            output_path is not None
            and output_path == parquet_path
            and Path(str(manifest.get("output_data_path"))).expanduser() == parquet_path
        ),
        "rows": int(shape_probe.get("rows") or 0),
        "manifest_sha256": observed_manifest_sha,
        "parquet_sha256": observed_parquet_sha,
        "expected_manifest_sha256": str(manifest_sha256).lower(),
        "expected_parquet_sha256": str(parquet_sha256).lower(),
        "hashes_exact": hashes_exact,
        "manifest_variant": str(manifest.get("manifest_variant") or ""),
        "expected_seq_snap_width": int(manifest.get("expected_seq_snap_width") or 0),
        "schema_version": str(manifest.get("schema_version") or ""),
        "seq_input_dim": int(signal_bridge.get("seq_input_dim") or 0),
        "snap_input_dim": int(signal_bridge.get("snap_input_dim") or 0),
        "field_count": int(len(fields)),
        "fields": fields,
        "contract_mode": str(extra.get("contract_mode") or ""),
        "direction_logit_mode": str(extra.get("direction_logit_mode") or ""),
        "model_native_signal_contract": extra.get("model_native_signal_contract"),
        "shape_probe": shape_probe,
    }


def _future_command_contracts(
    *,
    dataset_dir: Path,
    splits: dict[str, dict[str, Any]],
    post_rebuild_readiness_json: Path,
    specialist_audit_json: Path,
    run_id: str,
    memory_cap: str,
    swap_cap: str,
) -> dict[str, Any]:
    wrapper_argv = [
        "scripts/entry_next_edge_control.sh",
        "model-native-smoke-train",
        "--run-id",
        run_id,
        "--dataset-dir",
        str(dataset_dir),
        "--train-manifest-json",
        splits["train"]["manifest_path"],
        "--val-manifest-json",
        splits["val"]["manifest_path"],
        "--train-parquet",
        splits["train"]["parquet_path"],
        "--val-parquet",
        splits["val"]["parquet_path"],
        "--unified-exit-lifecycle-manifest-json",
        "<IMMUTABLE_UNIFIED_EXIT_LIFECYCLE_MANIFEST_JSON>",
        "--m5-prebuilt-path",
        "<IMMUTABLE_TIMESTAMPED_M5_PREBUILT_PATH>",
        "--multi-tf-cache-manifest-json",
        "<IMMUTABLE_MULTI_TF_CACHE_MANIFEST_JSON>",
        "--post-rebuild-readiness-json",
        str(post_rebuild_readiness_json),
        "--full-input-liveness-audit-json",
        "<IMMUTABLE_TIMESTAMPED_FULL_INPUT_LIVENESS_AUDIT_JSON>",
        "--feature-audit-json",
        "<IMMUTABLE_TIMESTAMPED_FEATURE_AUDIT_JSON>",
        "--target-audit-json",
        "<IMMUTABLE_TIMESTAMPED_TARGET_AUDIT_JSON>",
        "--specialist-audit-json",
        str(specialist_audit_json),
        "--pretrain-audit-json",
        "<IMMUTABLE_TIMESTAMPED_PRETRAIN_AUDIT_JSON>",
        "--recipe-audit-json",
        "<IMMUTABLE_TIMESTAMPED_RECIPE_AUDIT_JSON>",
        "--smoke-manifest-json",
        "<THIS_IMMUTABLE_TIMESTAMPED_SMOKE_MANIFEST_JSON>",
        "--smoke-readiness-json",
        "<IMMUTABLE_TIMESTAMPED_SMOKE_READINESS_JSON>",
        "--trainability-readiness-json",
        "<IMMUTABLE_TIMESTAMPED_TRAINABILITY_READINESS_JSON>",
        "--out-bundle-dir",
        "<FRESH_ABSOLUTE_SMOKE_BUNDLE_DIR>",
        "--gx1-data-root",
        "<ABSOLUTE_CANONICAL_GX1_DATA_ROOT>",
        "--device",
        "cuda",
        "--seed",
        "1337",
        "--epochs",
        "1",
        "--batch-size",
        "64",
        "--learning-rate",
        "0.0003",
        "--early-stop-patience",
        "1",
        "--early-stop-min-delta",
        "0.0",
        "--grad-clip-norm",
        "1.0",
        "--weight-decay",
        "0.00001",
        "--dropout",
        "0.05",
        "--multi-tf-scale",
        "0.5",
        "--num-workers",
        "0",
        "--multi-tf-num-layers",
        "2",
        "--specialist-num-layers",
        "1",
        "--grad-accum-steps",
        "1",
        # Per-timeframe lookback. Each band is owned by the coarsest timeframe
        # that covers it, so no branch spends bars on a span a cheaper one
        # already sees: M5 the last hour and a half, M15 out to two thirds of a
        # day, H1 four days, H4 sixteen days and D1 a full trading year.
        "--per-tf-seq-len-m5",
        "16",
        "--per-tf-seq-len-m15",
        "64",
        "--per-tf-seq-len-h1",
        "96",
        "--per-tf-seq-len-h4",
        "96",
        "--per-tf-seq-len-d1",
        "252",
        "--specialist-fusion-scale",
        "0.25",
        "--cross-family-fusion-scale",
        "0.25",
        "--subsample-rows",
        "10000",
        "--memory-cap",
        memory_cap,
        "--swap-cap",
        swap_cap,
        "<EXACTLY_ONE_OF_DRY_RUN_OR_EXECUTE>",
    ]
    return {
        "smart_smoke_manifest": {
            "mode": "report_only_manifest_materialization",
            "run_lineage_required": True,
            "entry_run_id": run_id,
            "mutates_only_report_dir": True,
            "starts_training": False,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
        },
        "smart_smoke_train": {
            "mode": "future_exact_wrapper_contract_not_executed",
            "implemented_in_control_surface": True,
            "profile": "smoke",
            "control_route": "model-native-smoke-train",
            "wrapper_path": TRAIN_WRAPPER_RELATIVE_PATH,
            "execution_allowed_now": False,
            "argv_template": wrapper_argv,
            "wrapper_argv_template": wrapper_argv,
            "run_lineage_required": True,
            "entry_run_id": run_id,
            "requires_clean_git": True,
            "requires_ram_cap": True,
            "ram_cap_runner": RAM_CAP_RUNNER,
            "memory_cap": memory_cap,
            "swap_cap": swap_cap,
            "num_workers": 0,
            "starts_trainer": True,
            "requires_direction_diagnostic_recipe_contract": True,
            "direction_diagnostic_recipe_contract": dict(
                DIRECTION_DIAGNOSTIC_RECIPE_CONTRACT
            ),
            "direction_diagnostic_env_template": dict(
                DIRECTION_DIAGNOSTIC_ENV_TEMPLATE
            ),
            "requires_direction_context_slice_contract": True,
            "direction_context_slice_contract": dict(DIRECTION_CONTEXT_SLICE_CONTRACT),
            "requires_canonical_direction_decision_contract": True,
            "canonical_direction_decision_contract": dict(
                CANONICAL_DIRECTION_DECISION_CONTRACT
            ),
            "started_by_this_report": False,
            "starts_training_if_executed": True,
            "starts_replay": False,
            "starts_iql_distillation": False,
            "touches_shadow_or_live": False,
            "expected_seq_snap_width": EXPECTED_SEQ_SNAP_WIDTH,
            "manifest_variant": MANIFEST_VARIANT,
            "specialist_contract_mode": SMART_SPECIALIST_CONTRACT_MODE,
        },
    }


def _build_smoke_manifest(
    *,
    dataset_dir: Path,
    run_id: str,
    splits: dict[str, dict[str, Any]],
    future_command_contracts: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "report_only": True,
        "manifest_variant": MANIFEST_VARIANT,
        "expected_seq_snap_width": EXPECTED_SEQ_SNAP_WIDTH,
        "entry_run_id": run_id,
        "out_dir": str(dataset_dir),
        "dataset_dir": str(dataset_dir),
        "stem": DEFAULT_STEM,
        "splits": {
            split: {
                "rows": int(row["rows"]),
                "out_parquet": row["output_data_path"],
                "out_manifest": row["manifest_path"],
                "out_parquet_sha256": row["parquet_sha256"],
                "out_manifest_sha256": row["manifest_sha256"],
                "split_manifest_schema_version": row["schema_version"],
                "seq_input_dim": int(row["seq_input_dim"]),
                "snap_input_dim": int(row["snap_input_dim"]),
                "field_count": int(row["field_count"]),
            }
            for split, row in splits.items()
        },
        "future_command_contracts": future_command_contracts,
        "side_effects_started": dict(SIDE_EFFECTS_STARTED),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    post_rebuild_readiness_json = Path(args.post_rebuild_readiness_json).expanduser().resolve()
    specialist_audit_json = Path(args.smart_specialist_audit_json).expanduser().resolve()
    _require_timestamped_evidence_path(
        post_rebuild_readiness_json,
        label="post-rebuild readiness",
    )
    _require_timestamped_evidence_path(
        specialist_audit_json,
        label="specialist audit",
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    run_id = str(args.run_id or "").strip()
    memory_cap = str(args.memory_cap or DEFAULT_MEMORY_CAP)
    swap_cap = str(args.swap_cap or DEFAULT_SWAP_CAP)
    sample_rows = int(args.sample_rows)
    batch_size = int(args.batch_size)

    post_rebuild_readiness = _read_json_or_empty(post_rebuild_readiness_json)
    post_rebuild_refresh_contract = (
        post_rebuild_readiness.get("post_rebuild_refresh_command_contract")
        if isinstance(post_rebuild_readiness.get("post_rebuild_refresh_command_contract"), dict)
        else {}
    )
    raw_dataset_dir = str(args.smart_smoke_dataset_dir or "").strip()
    contract_dataset_dir = str(post_rebuild_refresh_contract.get("smoke_dataset_dir") or "").strip()
    dataset_dir_source = "argument"
    dataset_dir_missing = not bool(raw_dataset_dir)
    dataset_dir = Path(raw_dataset_dir).expanduser().resolve()
    splits = {
        split: _split_summary(
            dataset_dir,
            split,
            parquet_value=str(getattr(args, f"{split}_parquet")),
            parquet_sha256=str(getattr(args, f"{split}_parquet_sha256")),
            manifest_value=str(getattr(args, f"{split}_manifest_json")),
            manifest_sha256=str(getattr(args, f"{split}_manifest_sha256")),
            sample_rows=sample_rows,
            batch_size=batch_size,
        )
        for split in SPLITS
    }
    split_paths = [
        row[key]
        for row in splits.values()
        for key in ("parquet_path", "manifest_path")
    ]
    post_rebuild_side_effects = (
        post_rebuild_readiness.get("side_effects_started")
        if isinstance(post_rebuild_readiness.get("side_effects_started"), dict)
        else {}
    )
    post_rebuild_side_effects_closed = (
        all(key in post_rebuild_side_effects for key in REQUIRED_POST_REBUILD_SIDE_EFFECT_KEYS)
        and all(post_rebuild_side_effects.get(key) is False for key in REQUIRED_POST_REBUILD_SIDE_EFFECT_KEYS)
    )
    post_rebuild_checks = {
        str(row.get("name") or ""): row
        for row in post_rebuild_readiness.get("checks", [])
        if isinstance(row, dict)
    }
    missing_post_rebuild_orchestration_checks = [
        name for name in REQUIRED_POST_REBUILD_ORCHESTRATION_CHECKS if name not in post_rebuild_checks
    ]
    failed_post_rebuild_orchestration_checks = [
        name for name in REQUIRED_POST_REBUILD_ORCHESTRATION_CHECKS
        if name in post_rebuild_checks and not bool(post_rebuild_checks[name].get("ok"))
    ]
    future_command_contracts = _future_command_contracts(
        dataset_dir=dataset_dir,
        splits=splits,
        post_rebuild_readiness_json=post_rebuild_readiness_json,
        specialist_audit_json=specialist_audit_json,
        run_id=run_id,
        memory_cap=memory_cap,
        swap_cap=swap_cap,
    )
    checks: list[dict[str, Any]] = [
        _check(
            "explicit model-native seq513 smoke run_id id is provided",
            _entry_run_id_ok(run_id),
            {"run_id": run_id},
        ),
        _check(
            "smart smoke dataset directory is explicit or pinned by post-rebuild readiness",
            not dataset_dir_missing,
            {
                "dataset_dir_source": dataset_dir_source,
                "argument_smart_smoke_dataset_dir": raw_dataset_dir,
                "post_rebuild_contract_smart_smoke_dataset_dir": contract_dataset_dir,
                "resolved_smart_smoke_dataset_dir": str(dataset_dir),
            },
        ),
        _check(
            "smart post-rebuild readiness report exists",
            post_rebuild_readiness_json.exists(),
            _artifact_meta(post_rebuild_readiness_json),
        ),
        _check(
            "smart post-rebuild readiness decision is ready",
            post_rebuild_readiness.get("schema_version")
            == POST_REBUILD_SCHEMA_VERSION
            and post_rebuild_readiness.get("decision")
            == POST_REBUILD_READY_DECISION,
            {
                "schema_version": post_rebuild_readiness.get("schema_version"),
                "expected_schema_version": POST_REBUILD_SCHEMA_VERSION,
                "decision": post_rebuild_readiness.get("decision"),
                "expected": POST_REBUILD_READY_DECISION,
            },
        ),
        _check(
            "smart post-rebuild readiness proves orchestration provenance",
            not missing_post_rebuild_orchestration_checks
            and not failed_post_rebuild_orchestration_checks,
            {
                "required_checks": list(REQUIRED_POST_REBUILD_ORCHESTRATION_CHECKS),
                "missing_checks": missing_post_rebuild_orchestration_checks,
                "failed_checks": failed_post_rebuild_orchestration_checks,
            },
        ),
        _check(
            "smart post-rebuild refresh contract points at this smoke dataset",
            str(Path(str(post_rebuild_refresh_contract.get("smoke_dataset_dir") or "")).expanduser().resolve())
            == str(dataset_dir),
            {
                "reported_smoke_dataset_dir": post_rebuild_refresh_contract.get("smoke_dataset_dir"),
                "actual_smart_smoke_dataset_dir": str(dataset_dir),
            },
        ),
        _check(
            "smart post-rebuild refresh contract starts no trainer replay iql shadow live",
            post_rebuild_refresh_contract.get("all_commands_avoid_training_replay_iql_shadow_live") is True
            and post_rebuild_side_effects_closed,
            {
                "all_commands_avoid_training_replay_iql_shadow_live": post_rebuild_refresh_contract.get(
                    "all_commands_avoid_training_replay_iql_shadow_live"
                ),
                "required_side_effect_keys": list(REQUIRED_POST_REBUILD_SIDE_EFFECT_KEYS),
                "side_effects_started": post_rebuild_side_effects,
            },
        ),
        _check("smart smoke dataset directory exists", dataset_dir.exists(), {"dataset_dir": str(dataset_dir)}),
        _check(
            "train val split paths are explicit canonical and distinct",
            all(row["explicit_paths_exact"] for row in splits.values())
            and len(set(split_paths)) == 4,
            splits,
        ),
        _check(
            "exact train val split artifacts exist",
            all(
                row["parquet_exists"] and row["manifest_exists"]
                for row in splits.values()
            ),
            splits,
        ),
        _check(
            "split manifest output_data_path exists and matches split parquet",
            all(row["output_data_exists"] and row["manifest_output_matches_split_parquet"] for row in splits.values()),
            splits,
        ),
        _check(
            "caller-bound split hashes match train val bytes",
            all(row["hashes_exact"] for row in splits.values()),
            splits,
        ),
        _check(
            "split manifests use model-native seq513 split schema",
            all(row["schema_version"] == SPLIT_SCHEMA_VERSION for row in splits.values()),
            {split: row["schema_version"] for split, row in splits.items()},
        ),
        _check(
            "split manifests pin model-native seq513 candidate",
            all(row["manifest_variant"] == MANIFEST_VARIANT for row in splits.values()),
            {split: row["manifest_variant"] for split, row in splits.items()},
        ),
        _check(
            "split manifests pin the owner-declared seq/snap width",
            all(row["expected_seq_snap_width"] == EXPECTED_SEQ_SNAP_WIDTH for row in splits.values()),
            {split: row["expected_seq_snap_width"] for split, row in splits.items()},
        ),
        _check(
            "split signal seq and snap dims match the owner contract",
            all(row["seq_input_dim"] == EXPECTED_SEQ_SNAP_WIDTH and row["snap_input_dim"] == EXPECTED_SEQ_SNAP_WIDTH for row in splits.values()),
            splits,
        ),
        _check(
            "split signal field counts match the owner contract",
            all(row["field_count"] == EXPECTED_SEQ_SNAP_WIDTH for row in splits.values()),
            {split: row["field_count"] for split, row in splits.items()},
        ),
        _check(
            "split parquet seq and snap samples match the owner width",
            all(bool(row["shape_probe"].get("ok")) for row in splits.values()),
            {split: row["shape_probe"] for split, row in splits.items()},
        ),
        _check(
            "split manifests carry exact model-native signal contract",
            all(
                row["contract_mode"] == MODEL_NATIVE_CONTRACT_MODE
                and row["direction_logit_mode"] == MODEL_NATIVE_DIRECTION_LOGIT_MODE
                and not model_native_signal_contract_failures(
                    row["model_native_signal_contract"]
                    if isinstance(row["model_native_signal_contract"], dict)
                    else {}
                )
                and isinstance(row["model_native_signal_contract"], dict)
                and row["fields"]
                == row["model_native_signal_contract"].get("fields")
                and not (set(row["fields"]) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS))
                for row in splits.values()
            ),
            splits,
        ),
        _check("side effects remain closed", all(value is False for value in SIDE_EFFECTS_STARTED.values()), SIDE_EFFECTS_STARTED),
        _check(
            "future train contract uses only the exact wrapper and four explicit artifacts",
            future_command_contracts["smart_smoke_train"]["requires_ram_cap"] is True
            and future_command_contracts["smart_smoke_train"]["ram_cap_runner"] == RAM_CAP_RUNNER
            and future_command_contracts["smart_smoke_train"]["num_workers"] == 0
            and future_command_contracts["smart_smoke_train"]["profile"] == "smoke"
            and future_command_contracts["smart_smoke_train"]["control_route"]
            == "model-native-smoke-train"
            and future_command_contracts["smart_smoke_train"]["wrapper_path"]
            == TRAIN_WRAPPER_RELATIVE_PATH
            and future_command_contracts["smart_smoke_train"]["wrapper_argv_template"]
            == future_command_contracts["smart_smoke_train"]["argv_template"]
            and all(
                flag in future_command_contracts["smart_smoke_train"]["argv_template"]
                for flag in (
                    "--train-manifest-json",
                    "--val-manifest-json",
                    "--train-parquet",
                    "--val-parquet",
                    "--unified-exit-lifecycle-manifest-json",
                    "--m5-prebuilt-path",
                    "--multi-tf-cache-manifest-json",
                    "--post-rebuild-readiness-json",
                    "--dropout",
                    "--num-workers",
                    "--multi-tf-num-layers",
                    "--specialist-num-layers",
                    "--grad-accum-steps",
                    "--per-tf-seq-len-m5",
                    "--per-tf-seq-len-m15",
                    "--per-tf-seq-len-h1",
                    "--per-tf-seq-len-h4",
                    "--per-tf-seq-len-d1",
                    "--cross-family-fusion-scale",
                )
            )
            and future_command_contracts["smart_smoke_train"]["argv_template"][
                future_command_contracts["smart_smoke_train"]["argv_template"].index(
                    "--num-workers"
                )
                + 1
            ]
            == "0"
            and "gx1.models.entry_v10.entry_v10_ctx_train_v3"
            not in " ".join(future_command_contracts["smart_smoke_train"]["argv_template"])
            and "--dataset_dir"
            not in future_command_contracts["smart_smoke_train"]["argv_template"]
            and "--dataset_train_parquet"
            not in future_command_contracts["smart_smoke_train"]["argv_template"],
            future_command_contracts["smart_smoke_train"],
        ),
        _check(
            "future train contract declares diagnostic and learned-task recipe",
            _direction_diagnostic_recipe_ok(
                future_command_contracts["smart_smoke_train"]
            ),
            future_command_contracts["smart_smoke_train"],
        ),
        _check(
            "future train contract declares direction context slice audit",
            _direction_context_slice_ok(future_command_contracts["smart_smoke_train"]),
            future_command_contracts["smart_smoke_train"],
        ),
        _check(
            "future train contract declares canonical derived direction pair",
            _canonical_direction_decision_ok(future_command_contracts["smart_smoke_train"]),
            future_command_contracts["smart_smoke_train"],
        ),
    ]
    failures = [check for check in checks if not check["ok"]]
    ready = not failures
    decision = (
        "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_REVIEW"
        if ready
        else "BLOCKED_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_READINESS"
    )
    smoke_manifest = (
        _build_smoke_manifest(
            dataset_dir=dataset_dir,
            run_id=run_id,
            splits=splits,
            future_command_contracts=future_command_contracts,
        )
        if ready
        else {}
    )
    created_utc = datetime.now(timezone.utc).isoformat()
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": created_utc,
        "decision": decision,
        "model_native_readiness_contract": (
            model_native_readiness_contract_metadata()
        ),
        "report_only": True,
        "manifest_variant": MANIFEST_VARIANT,
        "expected_seq_snap_width": EXPECTED_SEQ_SNAP_WIDTH,
        "entry_run_id": run_id if _entry_run_id_ok(run_id) else None,
        "smart_smoke_dataset_dir": str(dataset_dir),
        "smart_smoke_dataset_dir_source": dataset_dir_source,
        "out_dir": str(out_dir),
        "manifest_embedded": bool(ready),
        "manifest_sha256": _sha256_json(smoke_manifest) if ready else None,
        "smoke_manifest": smoke_manifest,
        "post_rebuild_readiness": _artifact_meta(post_rebuild_readiness_json),
        "specialist_audit": _artifact_meta(specialist_audit_json),
        "split_artifacts": splits,
        "future_command_contracts": future_command_contracts,
        "checks": checks,
        "failures": failures,
        "blockers": [row["name"] for row in failures],
        "training_allowed": False,
        "replay_allowed": False,
        "iql_allowed": False,
        "shadow_live_allowed": False,
        "control_surface_mutated": False,
        "mutations_outside_report_dir": False,
        "side_effects_started": dict(SIDE_EFFECTS_STARTED),
        "next_required_gate": (
            "bind the materialized smoke manifest into the evidence-gated training control path"
            if ready
            else "repair missing or invalid model-native seq513 smoke split artifacts before any train/replay/IQL/shadow/live path"
        ),
    }
    evidence_binding = {
        "post_rebuild_readiness": report["post_rebuild_readiness"],
        "specialist_audit": report["specialist_audit"],
        "split_artifacts": report["split_artifacts"],
    }
    report["evidence_binding_sha256"] = _sha256_json(evidence_binding)
    _, report = write_immutable_json_event(out_dir, EVENT_PREFIX, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smart-smoke-dataset-dir", required=True)
    ap.add_argument("--post-rebuild-readiness-json", required=True)
    ap.add_argument("--smart-specialist-audit-json", required=True)
    for split in SPLITS:
        ap.add_argument(f"--{split}-parquet", required=True)
        ap.add_argument(f"--{split}-parquet-sha256", required=True)
        ap.add_argument(f"--{split}-manifest-json", required=True)
        ap.add_argument(f"--{split}-manifest-sha256", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--run-id", "--run-id", dest="run_id", required=True)
    ap.add_argument("--memory-cap", default=DEFAULT_MEMORY_CAP)
    ap.add_argument("--swap-cap", default=DEFAULT_SWAP_CAP)
    ap.add_argument("--sample-rows", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
