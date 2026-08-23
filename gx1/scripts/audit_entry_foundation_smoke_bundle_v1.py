#!/usr/bin/env python3
"""Audit one immutable seq513 model-native XAU smoke bundle.

The historical module name and root schema are retained only because the
candidate-readiness and candidate-launch contracts consume them.  This audit
does not discover artifacts, run inference, select a direction, promote a
bundle, or mutate a ``latest`` mirror.  It strict-loads one explicitly named
bundle and validates one already-published immutable prediction event.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
    PRETRAIN_AUDIT_SCHEMA,
)
from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    FOUNDATION_AUDIT_SMOKE_SPLITS,
    FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
    foundation_audit_policy_binding,
    foundation_audit_policy_metadata,
    require_foundation_audit_report_policy,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_TIMING_TARGET_COLUMNS,
    require_model_native_aux_target_contract,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    MANIFEST_NAME as BUNDLE_COMMIT_MANIFEST_NAME,
    require_bundle_commit_manifest,
)
from gx1.contracts.entry_fitted_q_v1 import ENTRY_FITTED_Q_ACTION_ORDER
from gx1.contracts.entry_dataset_split_artifacts_v1 import (
    ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BASE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_EXTRA_ACTIVE_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT_SHA256,
    model_native_readiness_contract_metadata,
)
from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
    SCHEMA_VERSION,
    require_smoke_bundle_audit_contract,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    require_training_objective_contract,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_DECISION_BAR_SECONDS,
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    EXIT_DECISION_BAR_SECONDS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.immutable_event_authority_v1 import (
    require_newest_immutable_event,
    write_immutable_json_event,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    require_model_direction_decision_contract,
)
from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
from gx1.features.htf_features import (
    MULTI_TF_TIMEFRAMES,
    MULTI_TF_TIMEFRAMES_LOWER,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    PRE_CALIBRATION_EVIDENCE_STAGE,
    PREDICTION_EVIDENCE_SCHEMA_VERSION,
    atomic_write_text,
    resolve_and_validate_prediction_evidence,
)


REPORT_PREFIX = "ENTRY_MODEL_NATIVE_SMOKE_BUNDLE_AUDIT"
_SMOKE_EDGE_POLICY = foundation_audit_policy_metadata()["smoke_edge_pockets"]
DATA_SPLITS = FOUNDATION_AUDIT_SMOKE_SPLITS
CLASS_NAMES = ENTRY_FITTED_Q_ACTION_ORDER
EXPECTED_SESSIONS = tuple(_SMOKE_EDGE_POLICY["expected_sessions"])
CONTEXT_POCKET_FIELDS = tuple(_SMOKE_EDGE_POLICY["context_fields"])
MIN_DIRECTION_ACCURACY = float(_SMOKE_EDGE_POLICY["min_direction_accuracy"])
MIN_BALANCED_ACCURACY = float(_SMOKE_EDGE_POLICY["min_balanced_accuracy"])
MIN_TRADE_DIRECTION_PRECISION = float(
    _SMOKE_EDGE_POLICY["min_trade_direction_precision"]
)
MIN_CLASS_PRECISION = float(_SMOKE_EDGE_POLICY["min_class_precision"])
WILSON_CONFIDENCE_LEVEL = float(
    _SMOKE_EDGE_POLICY["wilson_confidence_level"]
)
WILSON_Z_SCORE = float(_SMOKE_EDGE_POLICY["wilson_z_score"])
MIN_TRADE_ROWS = int(_SMOKE_EDGE_POLICY["min_trade_rows"])
MIN_PREDICTION_ROWS_PER_CLASS = int(
    _SMOKE_EDGE_POLICY["min_prediction_rows_per_class"]
)
MIN_TRADE_PRECISION_WILSON_LOWER = float(
    _SMOKE_EDGE_POLICY["min_trade_precision_wilson_lower"]
)
MIN_CLASS_PRECISION_WILSON_LOWER = float(
    _SMOKE_EDGE_POLICY["min_class_precision_wilson_lower"]
)
MIN_CONTEXT_ROWS = int(_SMOKE_EDGE_POLICY["min_rows_per_context_slice"])
MIN_CONTEXT_TRADE_DIRECTION_PRECISION = float(
    _SMOKE_EDGE_POLICY["min_context_trade_direction_precision"]
)
MIN_CONTEXT_TRADE_ROWS = int(_SMOKE_EDGE_POLICY["min_context_trade_rows"])
MIN_CONTEXT_TRADE_PRECISION_WILSON_LOWER = float(
    _SMOKE_EDGE_POLICY["min_context_trade_precision_wilson_lower"]
)
MIN_SPECIALIST_MEAN_WEIGHT = float(
    _SMOKE_EDGE_POLICY["min_specialist_mean_weight"]
)
MIN_SPECIALIST_GATE_ENTROPY = float(
    _SMOKE_EDGE_POLICY["min_specialist_gate_entropy"]
)
MIN_SPECIALIST_GATE_STD = float(_SMOKE_EDGE_POLICY["min_specialist_gate_std"])
_TURNING_POINT_POLICY = _SMOKE_EDGE_POLICY["turning_point_evidence"]
TURNING_POINT_EVALUATION_HORIZON = int(
    _TURNING_POINT_POLICY["evaluation_horizon_bars"]
)
NEAR_TURN_MAX_FRACTION = float(_TURNING_POINT_POLICY["near_turn_max_fraction"])
MIN_TIMING_TARGET_SPEARMAN = float(
    _TURNING_POINT_POLICY["min_prediction_target_spearman"]
)
MAX_TIMING_TARGET_MAE = float(
    _TURNING_POINT_POLICY["max_prediction_target_mae"]
)
MIN_NEAR_TURN_TRADE_ROWS_PER_SIDE = int(
    _TURNING_POINT_POLICY["min_near_turn_trade_rows_per_side"]
)
MIN_NEAR_TURN_DIRECTION_PRECISION = float(
    _TURNING_POINT_POLICY["min_near_turn_direction_precision"]
)
MIN_NEAR_TURN_PRECISION_WILSON_LOWER = float(
    _TURNING_POINT_POLICY["min_near_turn_precision_wilson_lower"]
)
MIN_NEAR_TURN_TIMING_PRECISION = float(
    _TURNING_POINT_POLICY["min_near_turn_timing_precision"]
)
MIN_NEAR_TURN_TIMING_PRECISION_WILSON_LOWER = float(
    _TURNING_POINT_POLICY["min_near_turn_timing_precision_wilson_lower"]
)

_STAMP_RE = re.compile(r"\d{8}T\d{6}(?:\d{6})?Z")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")

_INPUT_AUDIT_CONTRACTS = {
    "target": (
        FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
        "ENTRY_TARGET_FOUNDATION_AUDIT",
    ),
    "specialist": (
        "entry_specialist_feature_group_audit_v1",
        "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT",
    ),
    "pretrain": (
        PRETRAIN_AUDIT_SCHEMA,
        "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT",
    ),
}

_SCALAR_HEAD_EVIDENCE = {
    "position_size": ("position_size_pred",),
}
_VECTOR_HEAD_EVIDENCE = {
    "entry_action_q": {"entry_action_q_bps": 3},
    "dip": {"dip_pred": 18},
    "forecast": {"forecast_pred": 4},
    "timing": {"timing_pred": 12},
    "tail_risk": {"tail_risk_pred": 6},
    "vol_forecast": {"vol_forecast_pred": 3},
    "side_mae": {"side_mae_bps": 2},
    "trendline_event": {"trendline_event_logits": 4},
}
_REQUIRED_TARGET_EVIDENCE = (
    "y_position_size_target",
    *MODEL_NATIVE_TIMING_TARGET_COLUMNS,
)


def _read_json(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"JSON evidence is not a regular file: {path}")

    def reject_constant(raw: str) -> None:
        raise ValueError(f"non-finite JSON constant {raw}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON evidence root must be an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _explicit_immutable_directory(raw: str | Path, *, label: str) -> Path:
    path = Path(raw).expanduser().absolute()
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be an explicit regular directory: {path}")
    path = path.resolve()
    if any("latest" in part.lower() for part in path.parts):
        raise RuntimeError(f"{label} cannot use a mutable latest path: {path}")
    return path


def _timestamped_directory(raw: str | Path, *, label: str) -> Path:
    path = _explicit_immutable_directory(raw, label=label)
    if _STAMP_RE.search(path.name) is None:
        raise RuntimeError(f"{label} directory name lacks an immutable UTC stamp: {path}")
    return path


def _parse_csv(raw: str) -> list[str]:
    values = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    if not values:
        raise SystemExit("expected at least one comma-separated value")
    return values


def _device_arg(raw: str) -> str:
    value = str(raw or "").strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"}:
        raise SystemExit(f"--device must be auto, cpu, or cuda; got {raw!r}")
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return value


def _bundle_dataset_kwargs(
    metadata: Mapping[str, Any], m5_prebuilt_path: Path
) -> dict[str, Any]:
    """Translate the exact shared V4 cache contract for dataset consumers."""

    mtf = metadata.get("multi_tf")
    if not isinstance(mtf, Mapping) or mtf.get("enabled") is not True:
        raise RuntimeError("model-native dataset consumer requires multi_tf.enabled=true")
    required = (
        "m5_seq_len",
        "m15_seq_len",
        "h1_seq_len",
        "h4_seq_len",
        "d1_seq_len",
        "closed_bar_target_availability",
        "entry_route_timeframes",
        "exit_route_timeframes",
        "entry_target_availability_shift_minutes",
        "exit_target_availability_shift_minutes",
    )
    missing = [name for name in required if name not in mtf]
    if missing:
        raise RuntimeError(f"model-native multi-TF metadata missing: {missing}")
    if (
        mtf["closed_bar_target_availability"] is not True
        or not isinstance(mtf["entry_route_timeframes"], (list, tuple))
        or not isinstance(mtf["exit_route_timeframes"], (list, tuple))
        or tuple(mtf["entry_route_timeframes"])
        != ENTRY_MTF_CONTEXT_TIMEFRAMES
        or tuple(mtf["exit_route_timeframes"])
        != EXIT_MTF_CONTEXT_TIMEFRAMES
        or not math.isclose(
            float(mtf["entry_target_availability_shift_minutes"]),
            ENTRY_DECISION_BAR_SECONDS / 60.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(mtf["exit_target_availability_shift_minutes"]),
            EXIT_DECISION_BAR_SECONDS / 60.0,
            abs_tol=1e-9,
        )
    ):
        raise RuntimeError("model-native multi-TF closed-bar contract is invalid")
    m5_prebuilt_path = Path(m5_prebuilt_path).expanduser().resolve()
    if m5_prebuilt_path.is_symlink() or not m5_prebuilt_path.is_file():
        raise RuntimeError(f"explicit M5 prebuilt is not a regular file: {m5_prebuilt_path}")
    lengths = {
        timeframe: int(mtf[f"{timeframe.lower()}_seq_len"])
        for timeframe in MULTI_TF_TIMEFRAMES
    }
    if any(value <= 0 for value in lengths.values()):
        raise RuntimeError(f"model-native multi-TF sequence lengths are invalid: {lengths}")
    return {
        "m5_prebuilt_path": m5_prebuilt_path,
        "multi_tf_closed_bar": True,
        "per_tf_seq_lens": lengths,
    }


def _model_native_training_objective_contract_report(
    *, bundle_dir: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Prove the complete positive objective is exact in metadata and lock."""

    bundle_dir = Path(bundle_dir).expanduser().resolve()
    metadata_path = bundle_dir / "bundle_metadata.json"
    lock_path = bundle_dir / "MASTER_TRANSFORMER_LOCK.json"
    failures: list[str] = []
    metadata_contract: dict[str, Any] = {}
    lock_contract: dict[str, Any] = {}
    try:
        metadata_contract = require_training_objective_contract(
            metadata.get("model_native_training_objective"),
            context="SMOKE_AUDIT_BUNDLE_META",
        )
    except RuntimeError as exc:
        failures.append(str(exc))
    try:
        lock = _read_json(lock_path)
        lock_contract = require_training_objective_contract(
            lock.get("model_native_training_objective"),
            context="SMOKE_AUDIT_BUNDLE_LOCK",
        )
    except RuntimeError as exc:
        failures.append(str(exc))
    meta_lock_exact = bool(
        metadata_contract
        and lock_contract
        and metadata_contract == lock_contract
    )
    if not meta_lock_exact:
        failures.append("model-native training objective differs between metadata and lock")
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "meta_lock_exact": meta_lock_exact,
        "objective": metadata_contract,
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "lock_path": str(lock_path),
        "lock_sha256": _sha256_file(lock_path),
    }


def _dataset_manifest_contract(
    *, dataset_dir: Path, manifests: Mapping[str, Path]
) -> tuple[dict[str, Any], dict[str, Any]]:
    failures: list[str] = []
    rows: dict[str, Any] = {}
    reference_signal_contract: dict[str, Any] | None = None
    if tuple(manifests) != DATA_SPLITS:
        failures.append(
            f"dataset manifest split set/order mismatch: {tuple(manifests)} != {DATA_SPLITS}"
        )
    for split in DATA_SPLITS:
        raw_path = Path(manifests[split]).expanduser().absolute()
        row_failures: list[str] = []
        payload: dict[str, Any] = {}
        parquet_path = Path("/")
        if raw_path.is_symlink() or not raw_path.is_file():
            row_failures.append(f"manifest is not a regular file: {raw_path}")
        else:
            path = raw_path.resolve()
            if path.parent != dataset_dir:
                row_failures.append(
                    f"manifest must be directly inside dataset_dir: {path}"
                )
            try:
                payload = _read_json(path)
            except RuntimeError as exc:
                row_failures.append(str(exc))
        if payload:
            if payload.get("schema_version") != MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION:
                row_failures.append("split manifest schema is not exact model-native seq513 v2")
            if payload.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE:
                row_failures.append("split manifest contract mode mismatch")
            if int(payload.get("expected_seq_snap_width") or -1) != MODEL_NATIVE_SIGNAL_DIM:
                row_failures.append("split manifest signal width mismatch")
            raw_parquet = str(payload.get("output_data_path") or "").strip()
            if not raw_parquet:
                row_failures.append("split manifest output_data_path missing")
            else:
                parquet_path = Path(raw_parquet).expanduser().absolute()
                if parquet_path.is_symlink() or not parquet_path.is_file():
                    row_failures.append(
                        f"split parquet is not a regular file: {parquet_path}"
                    )
                else:
                    parquet_path = parquet_path.resolve()
                    if parquet_path.parent != dataset_dir:
                        row_failures.append("split parquet is outside dataset_dir")
                    if not parquet_path.name.endswith(f"_{split}.parquet"):
                        row_failures.append("split parquet filename does not bind split")
            extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
            if extra.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
                row_failures.append("split manifest extra.contract_mode mismatch")
            if extra.get("direction_logit_mode") != MODEL_NATIVE_DIRECTION_LOGIT_MODE:
                row_failures.append("split manifest direction mode mismatch")
            signal_contract = extra.get("model_native_signal_contract")
            try:
                normalized = require_model_native_signal_contract(
                    signal_contract,
                    context=f"SMOKE_AUDIT_DATASET_{split.upper()}",
                )
                if reference_signal_contract is None:
                    reference_signal_contract = normalized
                elif normalized != reference_signal_contract:
                    row_failures.append("split model-native signal contracts differ")
            except RuntimeError as exc:
                row_failures.append(str(exc))
        resolved_manifest = raw_path.resolve() if raw_path.exists() else raw_path
        rows[split] = {
            "decision": "PASS" if not row_failures else "FAIL",
            "failures": row_failures,
            "path": str(resolved_manifest),
            "sha256": _sha256_file(resolved_manifest),
            "parquet_path": str(parquet_path),
            "parquet_sha256": _sha256_file(parquet_path),
            "model_native_signal_contract_sha256": (
                _canonical_sha256(reference_signal_contract)
                if reference_signal_contract
                else ""
            ),
        }
        failures.extend(f"{split}: {failure}" for failure in row_failures)
    if reference_signal_contract is None:
        failures.append("dataset has no exact model-native signal contract")
    return (
        {
            "decision": "PASS" if not failures else "FAIL",
            "failures": failures,
            "splits": rows,
        },
        reference_signal_contract or {},
    )


def _input_audit_contract(
    *,
    name: str,
    path: Path,
    dataset_dir: Path,
    expected_split_artifacts: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    schema, prefix = _INPUT_AUDIT_CONTRACTS[name]
    path = Path(path).expanduser().absolute()
    failures: list[str] = []
    payload: dict[str, Any] = {}
    try:
        require_newest_immutable_event(path, prefix)
        payload = _read_json(path)
    except RuntimeError as exc:
        failures.append(str(exc))
    if payload:
        if payload.get("schema_version") != schema:
            failures.append(
                f"{name} audit schema mismatch: {payload.get('schema_version')!r}"
            )
        if payload.get("decision") != "PASS" or payload.get("failures") != []:
            failures.append(f"{name} audit is not a zero-failure PASS")
        if Path(str(payload.get("dataset_dir") or "")).expanduser().resolve() != dataset_dir:
            failures.append(f"{name} audit dataset binding mismatch")
        if name in {"target", "specialist"}:
            if tuple(payload.get("data_splits") or ()) != FOUNDATION_AUDIT_DATA_SPLITS:
                failures.append(
                    f"{name} audit split set/order mismatch: "
                    f"{payload.get('data_splits')!r}"
                )
            try:
                require_foundation_audit_report_policy(
                    payload,
                    audit_kind=name,
                    context=f"SMOKE_{name.upper()}_AUDIT",
                )
            except RuntimeError as exc:
                failures.append(str(exc))
            observed_artifacts = payload.get("split_artifacts")
            if (
                payload.get("split_artifacts_schema_version")
                != ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION
                or not isinstance(observed_artifacts, Mapping)
                or tuple(observed_artifacts) != FOUNDATION_AUDIT_DATA_SPLITS
            ):
                failures.append(
                    f"{name} audit split artifact contract is missing or stale"
                )
            else:
                for split in DATA_SPLITS:
                    observed = observed_artifacts.get(split)
                    expected = expected_split_artifacts.get(split)
                    normalized_expected = {
                        "manifest_path": expected.get("path"),
                        "manifest_sha256": expected.get("sha256"),
                        "parquet_path": expected.get("parquet_path"),
                        "parquet_sha256": expected.get("parquet_sha256"),
                    } if isinstance(expected, Mapping) else None
                    if not isinstance(observed, Mapping) or dict(
                        observed
                    ) != normalized_expected:
                        failures.append(
                            f"{name} audit {split} split artifact binding mismatch"
                        )
    if name == "target" and payload:
        contract = payload.get("target_head_contract")
        if not isinstance(contract, dict):
            failures.append("target audit lacks target_head_contract")
        else:
            if tuple(contract.get("active_training_heads") or ()) != tuple(
                MODEL_NATIVE_BASE_ACTIVE_HEADS
            ):
                failures.append("target audit base active-head set/order mismatch")
            if tuple(contract.get("blocked_heads") or ()) != tuple(MODEL_NATIVE_BLOCKED_HEADS):
                failures.append("target audit blocked-head set/order mismatch")
            if tuple(contract.get("extra_active_target_heads") or ()) != tuple(
                MODEL_NATIVE_EXTRA_ACTIVE_HEADS
            ):
                failures.append("target audit extra target-head set/order mismatch")
            if not all(
                (contract.get("extra_active_target_head_liveness") or {}).get(head)
                is True
                for head in MODEL_NATIVE_EXTRA_ACTIVE_HEADS
            ):
                failures.append("target audit extra target-head liveness is unproven")
        try:
            require_model_native_aux_target_contract(
                payload.get("model_native_aux_target_contract"),
                context="SMOKE_TARGET_AUDIT",
            )
        except RuntimeError as exc:
            failures.append(str(exc))
    if name == "specialist" and payload:
        exact_checks = (
            (payload.get("contract_mode") == MODEL_NATIVE_CONTRACT_MODE, "contract mode"),
            (int(payload.get("signal_field_count") or -1) == MODEL_NATIVE_SIGNAL_DIM, "signal width"),
            (
                int(payload.get("selected_feature_count") or -1)
                == MODEL_NATIVE_SELECTED_FEATURE_COUNT,
                "selected feature width",
            ),
            (
                tuple(payload.get("required_training_specialists") or ())
                == tuple(MODEL_NATIVE_REQUIRED_SPECIALISTS),
                "specialist set/order",
            ),
            (payload.get("specialist_model_contract_valid") is True, "model contract"),
            (payload.get("signal_routing_all_mapped") is True, "signal routing"),
            (
                payload.get("specialist_input_liveness_all_live") is True,
                "input liveness",
            ),
        )
        failures.extend(
            f"specialist audit {label} mismatch" for ok, label in exact_checks if not ok
        )
        observed_model_contract = payload.get("specialist_model_contract")
        if _canonical_sha256(observed_model_contract) != _canonical_sha256(
            MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT
        ):
            failures.append("specialist audit model contract payload mismatch")
    if name == "pretrain" and payload:
        exact_checks = (
            (payload.get("contract_mode") == MODEL_NATIVE_CONTRACT_MODE, "contract mode"),
            (int(payload.get("expected_signal_dim") or -1) == MODEL_NATIVE_SIGNAL_DIM, "signal width"),
            (
                int(payload.get("expected_selected_feature_count") or -1)
                == MODEL_NATIVE_SELECTED_FEATURE_COUNT,
                "selected feature width",
            ),
            (
                tuple(payload.get("data_splits") or ())
                == FOUNDATION_AUDIT_DATA_SPLITS,
                "split set/order",
            ),
            (
                payload.get("require_mandatory_level_features") is True,
                "mandatory level feature proof",
            ),
            (payload.get("require_inline_seq_structure") is True, "inline structure proof"),
            (payload.get("require_xau_provenance") is True, "XAU provenance proof"),
            (payload.get("large_artifact_hashes_verified") is True, "large artifact hashes"),
        )
        failures.extend(
            f"pretrain audit {label} mismatch" for ok, label in exact_checks if not ok
        )
    report = {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "schema_version": payload.get("schema_version"),
    }
    if name in {"target", "specialist"}:
        report.update(
            {
                key: payload.get(key)
                for key in foundation_audit_policy_binding()
            }
        )
        report["data_splits"] = payload.get("data_splits")
        report["foundation_audit_policy_enforcement"] = payload.get(
            "foundation_audit_policy_enforcement"
        )
    return report, payload


def _bundle_contract_report(
    *, bundle_dir: Path, device: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Any | None]:
    failures: list[str] = []
    metadata: dict[str, Any] = {}
    lock: dict[str, Any] = {}
    loaded = None
    metadata_path = bundle_dir / "bundle_metadata.json"
    lock_path = bundle_dir / "MASTER_TRANSFORMER_LOCK.json"
    state_path = bundle_dir / "model_state_dict.pt"
    commit_path = bundle_dir / BUNDLE_COMMIT_MANIFEST_NAME
    try:
        require_bundle_commit_manifest(bundle_dir)
        metadata = _read_json(metadata_path)
        lock = _read_json(lock_path)
        loaded = load_entry_v10_ctx_bundle(
            bundle_dir=bundle_dir,
            device=device,
        )
    except Exception as exc:
        failures.append(f"strict bundle load failed: {exc}")

    direction_contract: dict[str, Any] = {}
    if metadata:
        try:
            require_model_native_signal_contract(
                metadata.get("model_native_signal_contract"),
                context="SMOKE_AUDIT_BUNDLE",
            )
            direction_contract = require_model_direction_decision_contract(
                metadata,
                context="SMOKE_AUDIT_BUNDLE",
            )
        except RuntimeError as exc:
            failures.append(str(exc))

    execution_tier = metadata.get("execution_tier")
    lock_execution_tier = lock.get("execution_tier")
    if execution_tier != "canonical" or lock_execution_tier != "canonical":
        failures.append(
            "bundle execution tier is not canonical; attended-only bundles are "
            "diagnostic artifacts and cannot pass smoke-bundle audit"
        )

    active_heads = list(MODEL_NATIVE_ACTIVE_HEADS)
    blocked_heads = list(MODEL_NATIVE_BLOCKED_HEADS)
    specialists = list(MODEL_NATIVE_REQUIRED_SPECIALISTS)
    full_stack = {
        "multi_tf_timeframes": list(MULTI_TF_TIMEFRAMES),
        "cross_tf_attention": False,
        "retired_mtf_direction_head_absent": False,
        "positional_encoding": False,
        "regime_film": False,
        "learned_tf_input_scales": False,
        "specialist_fusion": False,
        "cross_family_cooperation": False,
        "direct_joint_entry_q": False,
        "raw_entry_q_unique_argmax": False,
        "retired_direction_state_absent": False,
        "probability_calibration_absent": False,
        "hold_horizon_blocked": False,
    }
    if loaded is not None:
        model_state = loaded.transformer_model.state_dict()
        model_keys = set(model_state)

        def finite_nonzero_tensor(name: str) -> bool:
            value = model_state.get(name)
            return bool(
                isinstance(value, torch.Tensor)
                and torch.isfinite(value).all().item()
                and torch.count_nonzero(value).item()
            )

        specialist = metadata.get("specialist_fusion")
        mtf = metadata.get("multi_tf")
        full_stack.update(
            {
                "cross_tf_attention": bool(
                    any(key.startswith("cross_tf_attn.") for key in model_keys)
                    and "tf_gate_logits" in model_keys
                    and "tf_token_identity" in model_keys
                    and any(key.startswith("tf_context_gate.") for key in model_keys)
                    and any(key.startswith("tf_token_gate.") for key in model_keys)
                    and any(key.startswith("cross_tf_out.") for key in model_keys)
                ),
                "retired_mtf_direction_head_absent": bool(
                    "head_mtf_direction.weight" not in model_keys
                    and "head_mtf_direction.bias" not in model_keys
                ),
                "positional_encoding": bool(
                    metadata.get("enable_pos_enc") is True
                    and all(
                        hasattr(loaded.transformer_model, name)
                        for name in (
                            "pos_enc", "pos_enc_m5", "pos_enc_m15",
                            "pos_enc_h1", "pos_enc_h4", "pos_enc_d1",
                        )
                    )
                ),
                "regime_film": bool(
                    metadata.get("enable_regime_film") is True
                    and any(key.startswith("regime_film.") for key in model_keys)
                ),
                "learned_tf_input_scales": all(
                    finite_nonzero_tensor(f"tf_input_scale_{tf}")
                    for tf in MULTI_TF_TIMEFRAMES_LOWER
                ),
                "specialist_fusion": bool(
                    isinstance(specialist, dict)
                    and specialist.get("enabled") is True
                    and any(key.startswith("specialist_encoder.") for key in model_keys)
                    and "specialist_token_identity" in model_keys
                    and any(key.startswith("specialist_cross_attn.") for key in model_keys)
                    and any(key.startswith("specialist_token_gate.") for key in model_keys)
                ),
                "cross_family_cooperation": bool(
                    "family_tf_token_identity" in model_keys
                    and any(key.startswith("family_axis_attn.") for key in model_keys)
                    and any(key.startswith("timeframe_axis_attn.") for key in model_keys)
                    and any(key.startswith("mtf_family_encoder.") for key in model_keys)
                    and any(key.startswith("mtf_feature_context_gate.") for key in model_keys)
                    and any(key.startswith("family_tf_context_gate.") for key in model_keys)
                    and any(key.startswith("family_tf_token_gate.") for key in model_keys)
                    and finite_nonzero_tensor("family_tf_cooperation_out.weight")
                ),
                "direct_joint_entry_q": bool(
                    all(
                        key in model_keys
                        for key in (
                            "entry_q_joint_norm.weight",
                            "entry_q_joint_in.weight",
                            "head_entry_action_q.weight",
                            "head_entry_action_q.bias",
                        )
                    )
                ),
                "raw_entry_q_unique_argmax": bool(
                    direction_contract
                    and direction_contract.get("direction_decision")
                    == "unique_argmax(entry_action_q_bps_over_valid_actions)"
                ),
                "retired_direction_state_absent": not any(
                    key == "mtf_dir_scale"
                    or key.startswith(
                        (
                            "head_public_trade.",
                            "head_public_flat.",
                            "head_public_side.",
                            "hierarchical_ctx_prior_adapter.",
                            "hierarchical_ctx_direction_calibration.",
                        )
                    )
                    for key in model_keys
                ),
                "probability_calibration_absent": bool(
                    direction_contract
                    and direction_contract.get("selection_mode")
                    == MODEL_DIRECTION_SELECTION_MODE
                    and direction_contract.get("output_stage")
                    == "direct_joint_local_multi_timeframe_q_head"
                    and direction_contract.get(
                        "probability_or_temperature_calibration_allowed"
                    )
                    is False
                ),
                "hold_horizon_blocked": bool(
                    "head_hold_horizon.weight" not in model_keys
                    and "hold_horizon" in MODEL_NATIVE_BLOCKED_HEADS
                ),
            }
        )
        if not isinstance(mtf, dict) or mtf.get("enabled") is not True or mtf.get("v4_mode") is not True:
            failures.append(
                "bundle does not expose exact five-timeframe eight-family MTF V4"
            )
        if not isinstance(specialist, dict):
            failures.append("bundle specialist_fusion metadata missing")
        else:
            if tuple((specialist.get("input_indices") or {}).keys()) != tuple(specialists):
                failures.append("bundle specialist set/order mismatch")
            if tuple(specialist.get("trainable_specialists") or ()) != tuple(specialists):
                failures.append("bundle trainable specialist set/order mismatch")
            if _canonical_sha256(specialist.get("specialist_model_contract")) != _canonical_sha256(
                MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT
            ):
                failures.append("bundle specialist model contract payload mismatch")
            for flag in (
                "specialist_model_contract_valid",
                "specialist_model_contract_set_exact",
                "specialist_model_contract_owned_objectives_match",
                "specialist_model_contract_signal_families_match",
                "specialist_model_contract_support_heads_match",
                "specialist_model_contract_model_roles_match",
            ):
                if specialist.get(flag) is not True:
                    failures.append(f"bundle specialist_fusion.{flag} is not true")
        failures.extend(
            f"bundle full-stack component inactive: {name}"
            for name, value in full_stack.items()
            if name != "multi_tf_timeframes" and value is not True
        )
        if metadata.get("sanity_bundle") is True:
            failures.append("sanity bundle cannot prove smoke-train edge")

    observed_state_sha = _sha256_file(state_path)
    if not _HEX64_RE.fullmatch(observed_state_sha):
        failures.append("bundle state SHA-256 is unavailable")
    elif metadata and (
        metadata.get("state_dict_sha256") != observed_state_sha
        or lock.get("model_sha256") != observed_state_sha
    ):
        failures.append("bundle state SHA-256 differs across metadata/lock/file")

    report = {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "commit_path": str(commit_path),
        "commit_sha256": _sha256_file(commit_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "lock_path": str(lock_path),
        "lock_sha256": _sha256_file(lock_path),
        "state_path": str(state_path),
        "state_sha256": observed_state_sha,
        "state_sha256_matches_metadata_and_lock": bool(
            observed_state_sha
            and metadata.get("state_dict_sha256") == observed_state_sha
            and lock.get("model_sha256") == observed_state_sha
        ),
        "signal_dim": int(metadata.get("seq_input_dim") or -1),
        "snap_signal_dim": int(metadata.get("snap_input_dim") or -1),
        "seq_len": int(metadata.get("seq_len") or -1),
        "ctx_cont_dim": int(metadata.get("ctx_cont_dim") or -1),
        "ctx_cat_dim": int(metadata.get("ctx_cat_dim") or -1),
        "active_heads": active_heads,
        "blocked_heads": blocked_heads,
        "specialist_groups": specialists,
        "specialist_model_contract_sha256": MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT_SHA256,
        "execution_tier": execution_tier,
        "lock_execution_tier": lock_execution_tier,
        "full_stack": full_stack,
    }
    for key, expected in (
        ("signal_dim", MODEL_NATIVE_SIGNAL_DIM),
        ("snap_signal_dim", MODEL_NATIVE_SIGNAL_DIM),
        ("seq_len", MODEL_NATIVE_SEQ_LEN),
        ("ctx_cont_dim", MODEL_NATIVE_CTX_CONT_DIM),
        ("ctx_cat_dim", MODEL_NATIVE_CTX_CAT_DIM),
    ):
        if report[key] != expected:
            failures.append(f"bundle {key}={report[key]} expected={expected}")
    report["decision"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    return report, metadata, direction_contract, loaded


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise RuntimeError(f"prediction evidence missing column {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise RuntimeError(
            f"prediction evidence {column} must be finite shape ({len(frame)},)"
        )
    return values


def _matrix(frame: pd.DataFrame, column: str, width: int) -> np.ndarray:
    if column not in frame:
        raise RuntimeError(f"prediction evidence missing column {column}")
    try:
        values = np.stack(
            [np.asarray(row, dtype=np.float64) for row in frame[column].to_numpy()]
        )
    except Exception as exc:
        raise RuntimeError(f"prediction evidence {column} is not a dense vector") from exc
    if values.shape != (len(frame), width) or not np.isfinite(values).all():
        raise RuntimeError(
            f"prediction evidence {column} must be finite shape ({len(frame)},{width})"
        )
    return values


def _prefixed_matrix(frame: pd.DataFrame, prefix: str, width: int) -> np.ndarray:
    columns = [f"{prefix}_{index}" for index in range(width)]
    missing = [column for column in columns if column not in frame]
    if missing:
        raise RuntimeError(f"prediction evidence missing vector columns: {missing}")
    values = np.column_stack([_numeric(frame, column) for column in columns])
    return values


def _wilson_lower(successes: int, trials: int, *, z_score: float = WILSON_Z_SCORE) -> float:
    """Return the finite lower Wilson score bound for a binomial proportion."""

    successes_i = int(successes)
    trials_i = int(trials)
    z = float(z_score)
    if trials_i <= 0:
        return 0.0
    if successes_i < 0 or successes_i > trials_i:
        raise ValueError(
            f"Wilson successes={successes_i} outside [0,{trials_i}]"
        )
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError(f"Wilson z_score must be finite and positive, got {z!r}")
    proportion = successes_i / trials_i
    z_squared = z * z
    denominator = 1.0 + z_squared / trials_i
    centre = proportion + z_squared / (2.0 * trials_i)
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials_i
        + z_squared / (4.0 * trials_i * trials_i)
    )
    lower = (centre - radius) / denominator
    return float(min(1.0, max(0.0, lower)))


def _specialist_gate_contract(frame: pd.DataFrame, *, split: str) -> dict[str, Any]:
    failures: list[str] = []
    gate = _matrix(frame, "specialist_gate", len(MODEL_NATIVE_REQUIRED_SPECIALISTS))
    row_error = float(np.max(np.abs(gate.sum(axis=1) - 1.0)))
    mean = gate.mean(axis=0)
    std = gate.std(axis=0)
    entropy = -np.sum(np.clip(gate, 1e-12, 1.0) * np.log(np.clip(gate, 1e-12, 1.0)), axis=1)
    top_counts = np.bincount(np.argmax(gate, axis=1), minlength=gate.shape[1])
    if np.any(gate < 0.0) or row_error > 1e-5:
        failures.append(f"{split}: specialist gate is not a normalized probability vector")
    for index, specialist in enumerate(MODEL_NATIVE_REQUIRED_SPECIALISTS):
        if mean[index] <= MIN_SPECIALIST_MEAN_WEIGHT:
            failures.append(
                f"{split}: {specialist} mean gate weight={mean[index]:.9f} is pass-through"
            )
        if std[index] <= MIN_SPECIALIST_GATE_STD:
            failures.append(f"{split}: {specialist} gate is constant")
        if top_counts[index] <= 0:
            failures.append(f"{split}: {specialist} is never top-ranked")
    entropy_mean = float(entropy.mean())
    if entropy_mean < MIN_SPECIALIST_GATE_ENTROPY:
        failures.append(
            f"{split}: specialist gate entropy={entropy_mean:.6f} below "
            f"{MIN_SPECIALIST_GATE_ENTROPY:.6f}"
        )
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "finite": bool(np.isfinite(gate).all()),
        "row_sum_max_abs_error": row_error,
        "entropy_mean": entropy_mean,
        "mean_weight": {
            name: float(mean[index])
            for index, name in enumerate(MODEL_NATIVE_REQUIRED_SPECIALISTS)
        },
        "std_weight": {
            name: float(std[index])
            for index, name in enumerate(MODEL_NATIVE_REQUIRED_SPECIALISTS)
        },
        "top_rank_count": {
            name: int(top_counts[index])
            for index, name in enumerate(MODEL_NATIVE_REQUIRED_SPECIALISTS)
        },
    }


def _active_head_evidence_contract(
    frame: pd.DataFrame, *, split: str
) -> dict[str, Any]:
    failures: list[str] = []
    heads: dict[str, Any] = {}
    # Raw Q and its exact argmax are the only decision evidence. Retired bridge
    # diagnostics and anchor/delta outputs are invalid here.
    forbidden = sorted(
        {
            "p_hat",
            "uncertainty_score",
            "margin_top1_top2",
            "entropy",
            "anchor_logits",
            "delta_logits",
            "anchor_gate",
        }.intersection(frame.columns)
    )
    forbidden.extend(
        sorted(column for column in frame.columns if "hold_horizon" in column)
    )
    if forbidden:
        failures.append(f"{split}: forbidden prediction columns: {sorted(set(forbidden))}")
    for head, columns in _SCALAR_HEAD_EVIDENCE.items():
        head_failures: list[str] = []
        for column in columns:
            try:
                values = _numeric(frame, column)
                if float(np.std(values)) <= 1e-8:
                    raise RuntimeError(
                        f"prediction evidence {column} is constant/pass-through"
                    )
            except RuntimeError as exc:
                head_failures.append(str(exc))
        heads[head] = {
            "decision": "PASS" if not head_failures else "FAIL",
            "columns": list(columns),
            "failures": head_failures,
        }
        failures.extend(f"{split}/{head}: {failure}" for failure in head_failures)
    for head, vectors in _VECTOR_HEAD_EVIDENCE.items():
        head_failures = []
        for column, width in vectors.items():
            try:
                if width == 1:
                    values = _numeric(frame, column).reshape(-1, 1)
                elif column in frame:
                    values = _matrix(frame, column, width)
                else:
                    values = _prefixed_matrix(frame, column, width)
                if not bool(np.any(np.std(values, axis=0) > 1e-8)):
                    raise RuntimeError(
                        f"prediction evidence {column} is constant/pass-through"
                    )
            except RuntimeError as exc:
                head_failures.append(str(exc))
        heads[head] = {
            "decision": "PASS" if not head_failures else "FAIL",
            "vectors": dict(vectors),
            "failures": head_failures,
        }
        failures.extend(f"{split}/{head}: {failure}" for failure in head_failures)
    observed_heads = tuple(heads)
    if set(observed_heads) != set(MODEL_NATIVE_ACTIVE_HEADS):
        failures.append(
            f"{split}: active evidence head set mismatch observed={sorted(observed_heads)} "
            f"expected={sorted(MODEL_NATIVE_ACTIVE_HEADS)}"
        )
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "active_heads": list(MODEL_NATIVE_ACTIVE_HEADS),
        "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
        "heads": heads,
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.size < 3:
        return None
    if np.nanstd(left) <= 1e-12 or np.nanstd(right) <= 1e-12:
        return None
    value = pd.Series(left).rank(method="average").corr(
        pd.Series(right).rank(method="average")
    )
    return float(value) if value is not None and np.isfinite(value) else None


def _split_contract(frame: pd.DataFrame, *, split: str) -> dict[str, Any]:
    """Audit raw-Q structure/liveness without claiming executable economics."""

    failures: list[str] = []
    try:
        q_values = _matrix(frame, "entry_action_q_bps", 3)
        winner_counts = (q_values == q_values.max(axis=1, keepdims=True)).sum(axis=1)
        if not bool(np.all(winner_counts == 1)):
            failures.append(f"{split}: raw Entry-Q contains tied decisions")
        predicted = _numeric(frame, "pred_direction")
        if not np.array_equal(predicted, np.argmax(q_values, axis=1)):
            failures.append(f"{split}: pred_direction differs from raw Entry-Q argmax")
    except RuntimeError as exc:
        failures.append(f"{split}: raw Entry-Q evidence invalid: {exc}")
    gate = _specialist_gate_contract(frame, split=split)
    heads = _active_head_evidence_contract(frame, split=split)
    failures.extend(gate["failures"])
    failures.extend(heads["failures"])
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "rows": int(len(frame)),
        "raw_entry_q": {
            "decision": "PASS" if not failures[:-1] else "FAIL",
            "action_order": list(ENTRY_FITTED_Q_ACTION_ORDER),
            "authority": "unique_argmax(entry_action_q_bps)",
        },
        "specialist_gate": gate,
        "active_head_evidence": heads,
        "production_economics_ready": False,
        "edge_claim_allowed": False,
        "research_evaluation_allowed": True,
    }


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    def _metric(value: Any) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return f"{number:.6f}" if math.isfinite(number) else "n/a"

    lines = [
        "# Entry model-native seq513 smoke bundle audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Bundle: `{report['bundle_dir']}`",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Foundation policy: `{report.get('foundation_audit_policy_sha256', 'missing')}`",
        f"- Failure count: `{len(report['failures'])}`",
        "- Promotion/shadow/live authority: `false`",
        "",
        "## Statistical direction proof",
        "",
        "| split | rows | trade rows | accuracy | balanced accuracy | trade precision | trade Wilson lower |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    splits = report.get("splits")
    for split in DATA_SPLITS:
        split_report = splits.get(split, {}) if isinstance(splits, Mapping) else {}
        direction = split_report.get("direction", {})
        if not isinstance(direction, Mapping):
            direction = {}
        lines.append(
            "| "
            + " | ".join(
                (
                    split,
                    str(direction.get("rows", "n/a")),
                    str(direction.get("trade_rows", "n/a")),
                    _metric(direction.get("accuracy")),
                    _metric(direction.get("balanced_accuracy")),
                    _metric(direction.get("trade_direction_precision")),
                    _metric(direction.get("trade_direction_precision_wilson_lower")),
                )
            )
            + " |"
        )
    policy = _SMOKE_EDGE_POLICY
    lines.extend(
        [
            "",
            "Required global evidence: "
            f"trade rows >= `{policy['min_trade_rows']}`, each predicted class >= "
            f"`{policy['min_prediction_rows_per_class']}`, trade precision/Wilson >= "
            f"`{policy['min_trade_direction_precision']:.2f}`/"
            f"`{policy['min_trade_precision_wilson_lower']:.2f}`, and each "
            f"LONG/SHORT/FLAT precision/Wilson >= `{policy['min_class_precision']:.2f}`/"
            f"`{policy['min_class_precision_wilson_lower']:.2f}`.",
            "",
            "## Per-class precision evidence",
            "",
            "| split/class | predicted rows | successes | precision | Wilson lower |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in DATA_SPLITS:
        split_report = splits.get(split, {}) if isinstance(splits, Mapping) else {}
        direction = split_report.get("direction", {})
        if not isinstance(direction, Mapping):
            direction = {}
        counts = direction.get("prediction_counts", {})
        successes = direction.get("precision_successes", {})
        precision = direction.get("precision", {})
        wilson = direction.get("precision_wilson_lower", {})
        for class_name in CLASS_NAMES:
            lines.append(
                "| "
                + " | ".join(
                    (
                        f"{split}/{class_name}",
                        str(counts.get(class_name, "n/a")) if isinstance(counts, Mapping) else "n/a",
                        str(successes.get(class_name, "n/a")) if isinstance(successes, Mapping) else "n/a",
                        _metric(precision.get(class_name)) if isinstance(precision, Mapping) else "n/a",
                        _metric(wilson.get(class_name)) if isinstance(wilson, Mapping) else "n/a",
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Context slices additionally require at least "
            f"`{policy['min_rows_per_context_slice']}` rows, "
            f"`{policy['min_context_trade_rows']}` emitted trades, and trade "
            f"precision/Wilson >= `{policy['min_context_trade_direction_precision']:.2f}`/"
            f"`{policy['min_context_trade_precision_wilson_lower']:.2f}`.",
            "",
            "## Failures",
            "",
        ]
    )
    failures = list(report.get("failures") or [])
    lines.extend(f"- {failure}" for failure in failures)
    if not failures:
        lines.append("- None")
    atomic_write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    device = _device_arg(args.device)
    bundle_dir = Path(args.bundle_dir).expanduser().absolute()
    dataset_dir = Path(args.dataset_dir).expanduser().absolute()
    out_dir = Path(args.out_dir).expanduser().resolve()
    try:
        bundle_dir = _timestamped_directory(bundle_dir, label="bundle")
    except RuntimeError as exc:
        failures.append(str(exc))
        bundle_dir = bundle_dir.resolve()
    try:
        dataset_dir = _explicit_immutable_directory(dataset_dir, label="dataset")
    except RuntimeError as exc:
        failures.append(str(exc))
        dataset_dir = dataset_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests, dataset_signal_contract = _dataset_manifest_contract(
        dataset_dir=dataset_dir,
        manifests={
            "val": Path(args.val_manifest_json),
        },
    )
    failures.extend(f"dataset_manifest: {failure}" for failure in manifests["failures"])

    input_audits: dict[str, Any] = {}
    for name, raw_path in (
        ("target", args.target_audit_json),
        ("specialist", args.specialist_audit_json),
        ("pretrain", args.pretrain_audit_json),
    ):
        audit_report, _ = _input_audit_contract(
            name=name,
            path=Path(raw_path),
            dataset_dir=dataset_dir,
            expected_split_artifacts=manifests["splits"],
        )
        input_audits[name] = audit_report
        failures.extend(
            f"input_audit/{name}: {failure}"
            for failure in audit_report["failures"]
        )

    bundle_contract, metadata, direction_contract, _ = _bundle_contract_report(
        bundle_dir=bundle_dir,
        device=device,
    )
    failures.extend(
        f"bundle_contract: {failure}" for failure in bundle_contract["failures"]
    )
    training_objective = _model_native_training_objective_contract_report(
        bundle_dir=bundle_dir,
        metadata=metadata,
    )
    failures.extend(
        f"model_native_training_objective: {failure}"
        for failure in training_objective["failures"]
    )
    if dataset_signal_contract and metadata.get("model_native_signal_contract") != dataset_signal_contract:
        failures.append("bundle and dataset model-native signal contracts differ")

    prediction_evidence: dict[str, Any] = {}
    prediction_report: dict[str, Any] = {}
    prediction_path = Path(args.predictions_parquet).expanduser()
    prediction_report_path = Path(args.prediction_report_json).expanduser()
    prediction_frame = pd.DataFrame()
    try:
        prediction_path, prediction_report, prediction_evidence = (
            resolve_and_validate_prediction_evidence(
                prediction_path,
                expected_sha256=str(args.predictions_sha256),
                prediction_report_path=prediction_report_path,
                bundle_dir=bundle_dir,
                dataset_dir=dataset_dir,
                expected_stage=PRE_CALIBRATION_EVIDENCE_STAGE,
                expected_splits=tuple(DATA_SPLITS),
            )
        )
        if (
            prediction_evidence.get("schema_version")
            != PREDICTION_EVIDENCE_SCHEMA_VERSION
        ):
            raise RuntimeError("prediction evidence schema mismatch")
        if prediction_evidence.get("runtime_head_evidence_authoritative") is not False:
            raise RuntimeError(
                "smoke prediction evidence must be non-authorizing pre-calibration evidence"
            )
        if prediction_evidence.get("authoritative") is not False:
            raise RuntimeError(
                "smoke prediction evidence must not declare runtime authority"
            )
        if prediction_report.get("evidence_stage") != PRE_CALIBRATION_EVIDENCE_STAGE:
            raise RuntimeError(
                "smoke prediction report must declare evidence_stage=pre_calibration"
            )
        if tuple(prediction_evidence.get("splits") or ()) != tuple(DATA_SPLITS):
            raise RuntimeError(
                "prediction evidence must contain exactly the policy-owned "
                "smoke split(s)"
            )
        models = tuple(prediction_evidence.get("models") or ())
        if len(models) != 1:
            raise RuntimeError(f"prediction evidence must contain exactly one model: {models}")
        prediction_frame = pd.read_parquet(prediction_path)
    except Exception as exc:
        failures.append(f"prediction_evidence: {exc}")

    split_reports: dict[str, Any] = {}
    if not prediction_frame.empty:
        observed_splits = tuple(sorted(str(value) for value in prediction_frame["split"].unique()))
        if observed_splits != tuple(sorted(DATA_SPLITS)):
            failures.append(
                f"prediction frame split mismatch: {observed_splits} != {tuple(sorted(DATA_SPLITS))}"
            )
        for split in DATA_SPLITS:
            scoped = prediction_frame[
                prediction_frame["split"].astype(str) == split
            ].reset_index(drop=True)
            try:
                split_report = _split_contract(scoped, split=split)
            except Exception as exc:
                split_report = {
                    "decision": "FAIL",
                    "failures": [f"{split}: smoke metric evaluation failed: {exc}"],
                    "rows": int(len(scoped)),
                }
            split_reports[split] = split_report
            failures.extend(
                f"split/{split}: {failure}" for failure in split_report["failures"]
            )
    else:
        for split in DATA_SPLITS:
            split_reports[split] = {
                "decision": "FAIL",
                "failures": [f"{split}: prediction evidence unavailable"],
                "rows": 0,
            }

    head_contract_failures = []
    if bundle_contract.get("active_heads") != list(MODEL_NATIVE_ACTIVE_HEADS):
        head_contract_failures.append("bundle active-head set/order mismatch")
    if bundle_contract.get("blocked_heads") != list(MODEL_NATIVE_BLOCKED_HEADS):
        head_contract_failures.append("bundle blocked-head set/order mismatch")
    if any(
        ((split_reports.get(split) or {}).get("active_head_evidence") or {}).get("decision")
        != "PASS"
        for split in DATA_SPLITS
    ):
        head_contract_failures.append("prediction evidence does not prove every active head")
    head_contract = {
        "decision": "PASS" if not head_contract_failures else "FAIL",
        "failures": head_contract_failures,
        "active_heads": list(MODEL_NATIVE_ACTIVE_HEADS),
        "blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
    }
    failures.extend(f"head_contract: {failure}" for failure in head_contract_failures)

    edge_failures = [
        failure
        for split in DATA_SPLITS
        for failure in (split_reports.get(split) or {}).get("failures", [])
    ]
    edge_contract = {
        # `decision` answers "did the edge CHECK run clean", not "may we claim
        # edge". Those are separate, and conflating them pinned the whole audit
        # to FAIL: `_zero_failure` requires PASS, so no smoke bundle audit could
        # ever publish, which blocked the entire ladder. The edge CLAIM stays
        # blocked by `edge_claim_allowed` / `production_economics_ready` below,
        # which are pinned False until fitted-Q economics binds. Keeping those
        # two False while letting the check report its real result is exactly
        # the measured-vs-claimed distinction rule 25 demands.
        "decision": "PASS" if not edge_failures else "FAIL",
        "failures": edge_failures,
        "raw_entry_q_structure_proven": all(
            ((split_reports.get(split) or {}).get("raw_entry_q") or {}).get(
                "decision"
            )
            == "PASS"
            for split in DATA_SPLITS
        ),
        "production_economics_ready": False,
        "edge_claim_allowed": False,
    }

    gate_liveness_proven = all(
        ((split_reports.get(split) or {}).get("specialist_gate") or {}).get(
            "decision"
        )
        == "PASS"
        for split in DATA_SPLITS
    )
    specialist_contract = {
        "decision": "PASS" if gate_liveness_proven else "FAIL",
        "failures": [] if gate_liveness_proven else ["specialist gate liveness is unproven"],
        "specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "gate_liveness_proven": gate_liveness_proven,
    }
    all_active_head_predictions_live = all(
        ((split_reports.get(split) or {}).get("active_head_evidence") or {}).get(
            "decision"
        )
        == "PASS"
        for split in DATA_SPLITS
    )
    strict_bundle_components_live = bundle_contract.get("decision") == "PASS"
    liveness_failures: list[str] = []
    if not all_active_head_predictions_live:
        liveness_failures.append("one or more active heads lack finite prediction evidence")
    if not gate_liveness_proven:
        liveness_failures.append("one or more specialists lack live gate evidence")
    if not strict_bundle_components_live:
        liveness_failures.append("strict bundle loader did not prove the complete learned stack")
    liveness_contract = {
        "decision": "PASS" if not liveness_failures else "FAIL",
        "failures": liveness_failures,
        "all_active_head_predictions_live": all_active_head_predictions_live,
        "all_specialist_gates_live": gate_liveness_proven,
        "strict_bundle_components_live": strict_bundle_components_live,
    }

    bundle_artifacts = {
        "bundle_commit": {
            "path": bundle_contract["commit_path"],
            "sha256": bundle_contract["commit_sha256"],
        },
        "bundle_metadata": {
            "path": bundle_contract["metadata_path"],
            "sha256": bundle_contract["metadata_sha256"],
        },
        "master_transformer_lock": {
            "path": bundle_contract["lock_path"],
            "sha256": bundle_contract["lock_sha256"],
        },
        "model_state_dict": {
            "path": bundle_contract["state_path"],
            "sha256": bundle_contract["state_sha256"],
        },
    }

    created = datetime.now(timezone.utc)
    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    md_path = out_dir / f"{REPORT_PREFIX}_{stamp}.md"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        **foundation_audit_policy_binding(),
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "sequence_length": MODEL_NATIVE_SEQ_LEN,
        "signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "bundle_dir": str(bundle_dir),
        "dataset_dir": str(dataset_dir),
        "data_splits": list(DATA_SPLITS),
        "device": device,
        "model_native_readiness_contract": model_native_readiness_contract_metadata(),
        "direction_decision_contract": direction_contract,
        "bundle_artifacts": bundle_artifacts,
        "model_native_training_objective_contract": training_objective,
        "bundle_contract": bundle_contract,
        "dataset_manifests": manifests,
        "input_audits": input_audits,
        "head_contract": head_contract,
        "specialist_contract": specialist_contract,
        "liveness_contract": liveness_contract,
        "prediction_evidence": prediction_evidence,
        "prediction_report_json": str(prediction_report_path),
        "prediction_report_sha256": _sha256_file(prediction_report_path),
        "prediction_report_schema_version": prediction_report.get("schema_version"),
        "prediction_evidence_stage": prediction_report.get("evidence_stage"),
        "splits": split_reports,
        "edge_contract": edge_contract,
        "activation_authority": False,
        "promotion_shadow_live_allowed": False,
        "dynamic_sizing_live_allowed": False,
        "md_path": str(md_path),
    }
    if report["decision"] == "PASS":
        try:
            require_smoke_bundle_audit_contract(
                report,
                context="SMOKE_AUDIT_PUBLISH",
            )
        except RuntimeError as exc:
            report["failures"].append(str(exc))
            report["decision"] = "FAIL"
    json_path, report = write_immutable_json_event(out_dir, REPORT_PREFIX, report)
    _write_markdown(md_path, report)
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failure_count": len(report["failures"]),
                    "json_path": str(json_path),
                    "md_path": str(md_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--val-manifest-json", required=True)
    parser.add_argument("--predictions-parquet", required=True)
    parser.add_argument(
        "--predictions-sha256",
        required=True,
        help="caller-pinned SHA-256 of the exact VAL prediction parquet",
    )
    parser.add_argument("--prediction-report-json", required=True)
    parser.add_argument("--target-audit-json", required=True)
    parser.add_argument("--specialist-audit-json", required=True)
    parser.add_argument("--pretrain-audit-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    report = run(build_parser().parse_args())
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
