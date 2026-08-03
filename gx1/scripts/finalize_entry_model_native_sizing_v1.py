"""Canonical producer/finalizer for the learned Entry sizing evidence chain.

No JSON in this chain is intended for hand editing.  The public stages are:

1. capture broker instrument evidence;
2. fit TRAIN/VAL calibration and publish its immutable event;
3. bind that calibration into a fresh bundle clone;
4. materialize canonical TEST/OOS sizing rows and publish a row-recomputed
   diagnostic proof;
5. finalize a full-TEST same-candidate unified-Exit proof from exact row traces;
6. adopt learned sizing only after that exact joint proof is bound;
7. finalize post-adoption broker-runtime shadow parity from exact observations.

Any red/malformed newer event keeps launch fail-closed.  This module never edits
the launch-state selector and never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from functools import wraps
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    MODEL_NATIVE_SIZING_ADOPTION_SCHEMA_VERSION,
    MODEL_NATIVE_SIZING_MODE_LEARNED,
    model_native_sizing_bundle_calibration_metadata,
    require_model_native_sizing_adoption_artifact,
    require_model_native_sizing_bundle_calibration,
)
from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    MODEL_NATIVE_SIZING_CALIBRATION_SCHEMA_VERSION,
    MODEL_NATIVE_SIZING_FIT_SPLITS,
    MODEL_NATIVE_SIZING_FIT_SCOPE,
    MODEL_NATIVE_SIZING_HOLDOUT_SPLIT,
    MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION,
    MODEL_NATIVE_SIZING_MAX_GROSS_XAU_UNITS,
    MODEL_NATIVE_SIZING_MIN_FIT_ROWS_PER_SPLIT,
    MODEL_NATIVE_SIZING_OOS_PROOF_SCHEMA_VERSION,
    MODEL_NATIVE_SIZING_OOS_SOURCE_SCHEMA_VERSION,
    MODEL_NATIVE_SIZING_OOS_SCOPE,
    MODEL_NATIVE_SIZING_TRANSFORM_VERSION,
    MODEL_NATIVE_SIZING_INSTRUMENT_EVIDENCE_SCHEMA_VERSION,
    derive_canonical_sizing_oos_rows,
    fit_monotone_sizing_parameters,
    load_bound_sizing_calibration,
    load_bound_sizing_oos_proof,
    load_bound_sizing_oos_source,
    recompute_sizing_oos_evidence,
    require_immutable_json_binding,
    require_sizing_evaluation_bundle,
    require_sizing_instrument_evidence_artifact,
    require_sizing_prediction_provenance,
    sha256_file,
    sizing_risk_policy_metadata,
    sizing_oos_reference_account_policy_metadata,
    sizing_fit_contract_metadata,
)
from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    CANONICAL_UNIFIED_REPLAY_PRODUCER_CONTRACT,
    CANONICAL_UNIFIED_REPLAY_PRODUCER_SCHEMA_VERSION,
    MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE,
    MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS,
    MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_EVENT_PREFIX,
    MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_SCHEMA_VERSION,
    MODEL_NATIVE_JOINT_EXIT_SIZING_REPLAY_CONTRACT,
    MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS,
    MODEL_NATIVE_SIZING_RUNTIME_PARITY_EVENT_PREFIX,
    MODEL_NATIVE_SIZING_RUNTIME_PARITY_CONTRACT,
    MODEL_NATIVE_SIZING_RUNTIME_PARITY_SCHEMA_VERSION,
    build_canonical_unified_replay_source_inventory,
    candidate_bundle_authority,
    joint_exit_trace_sha256,
    load_bound_joint_exit_sizing_proof,
    load_bound_runtime_sizing_parity,
    read_bound_parquet_exact,
    recompute_joint_exit_replay_coverage,
    require_joint_replay_extends_canonical_oos_rows,
    recompute_runtime_sizing_parity_coverage,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_parquet_immutable,
    resolve_and_validate_prediction_evidence,
)
from gx1.contracts.immutable_event_authority_v1 import (
    write_immutable_json_event,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    CORE_ARTIFACTS as BUNDLE_COMMIT_CORE_ARTIFACTS,
    MANIFEST_NAME as BUNDLE_COMMIT_MANIFEST_NAME,
    publish_bundle_directory_noreplace,
    require_bundle_commit_manifest,
    write_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_calibration_v1 import (
    CALIBRATION_EVENT_PREFIX as ENTRY_CALIBRATION_EVENT_PREFIX,
    IMMUTABLE_CALIBRATION_EVENT_SCHEMA_VERSION,
    require_model_native_calibration_metadata,
)


INSTRUMENT_EVIDENCE_SCHEMA_VERSION = (
    MODEL_NATIVE_SIZING_INSTRUMENT_EVIDENCE_SCHEMA_VERSION
)
INSTRUMENT_EVIDENCE_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_INSTRUMENT_EVIDENCE"
CALIBRATION_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_CALIBRATION"
OOS_SOURCE_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_OOS_SOURCE"
PROOF_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_OOS_PROOF"
ADOPTION_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_ADOPTION"
JOINT_EXIT_PROOF_PREFIX = MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_EVENT_PREFIX


RUNTIME_PARITY_PREFIX = MODEL_NATIVE_SIZING_RUNTIME_PARITY_EVENT_PREFIX
MIN_FIT_ROWS_PER_SPLIT = MODEL_NATIVE_SIZING_MIN_FIT_ROWS_PER_SPLIT
_AUTHORITY_STAGE_DIRS = {
    "instrument": "instrument",
    "calibration": "calibration",
    "oos": "oos",
    "proof": "proof",
    "joint_replay": "joint_replay",
    "adoption": "adoption",
    "runtime_parity": "runtime_parity",
}
_INSTRUMENT_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "instrument",
        "account_currency",
        "quote_currency",
        "trade_units_precision",
        "minimum_trade_size",
        "broker_maximum_order_units",
        "margin_rate",
        "account_observed_utc",
        "instrument_observed_utc",
        "account_last_transaction_id",
        "instrument_last_transaction_id",
    }
)


class SizingFinalizationError(RuntimeError):
    """A canonical sizing producer stage could not establish exact evidence."""


class _TerminalSizingEventPublished(SizingFinalizationError):
    """A terminal FAIL already exists; the wrapper must not publish a duplicate."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: Any, *, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError) as exc:
            raise SizingFinalizationError(f"{label} is not an exact UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise SizingFinalizationError(f"{label} must be timezone-aware UTC")
    return parsed


def _strictly_after_now(*predecessors: Any) -> datetime:
    """Return a UTC instant strictly after every predecessor despite clock steps."""

    created = _now()
    for index, value in enumerate(predecessors):
        if value is None:
            continue
        predecessor = _as_utc(value, label=f"predecessor[{index}]")
        if created <= predecessor:
            created = predecessor + timedelta(microseconds=1)
    return created


def _event_time_from_filename(path: Path, *, prefix: str) -> datetime | None:
    marker = f"{prefix}_"
    if not path.name.startswith(marker) or not path.name.endswith(".json"):
        return None
    stamp = path.name[len(marker) : -len(".json")]
    for fmt in ("%Y%m%dT%H%M%S%fZ", "%Y%m%dT%H%M%SZ"):
        try:
            return datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _monotonic_stage_created_utc(
    output_dir: Path,
    prefix: str,
    *predecessors: Any,
) -> datetime:
    """Choose an event time newer than its chain and any prior stage attempt.

    Immutable authority ordering cannot depend on wall-clock monotonicity: NTP or
    VM clock corrections may move ``datetime.now`` backwards between stages.
    Filename timestamps remain available even when the newest payload is a
    deliberately malformed terminal event, so they are included in the floor.
    """

    floors = list(predecessors)
    if output_dir.exists():
        floors.extend(
            event_time
            for path in output_dir.glob(f"{prefix}_*.json")
            if (event_time := _event_time_from_filename(path, prefix=prefix))
            is not None
        )
    return _strictly_after_now(*floors)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _binding(path: Path) -> dict[str, str]:
    return {"json_path": str(path.resolve()), "sha256": _sha(path)}


def _source_binding(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SizingFinalizationError(f"source file missing: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_immutable_parquet_binding(
    path: Path,
    *,
    context: str,
) -> tuple[Path, dict[str, str]]:
    """Reject mutable aliases before resolving and bind one exact parquet."""

    candidate = path.expanduser()
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    if (
        candidate.suffix != ".parquet"
        or "latest" in candidate.name.lower()
        or any(component.is_symlink() for component in (absolute, *absolute.parents))
    ):
        raise SizingFinalizationError(
            f"{context} must be a canonical immutable parquet"
        )
    canonical = candidate.resolve()
    return canonical, _source_binding(canonical)


def _stage_dir(authority_root: Path, stage: str) -> Path:
    """Return the single code-owned event-family directory for one authority root."""

    if stage not in _AUTHORITY_STAGE_DIRS:
        raise SizingFinalizationError(f"unknown authority stage: {stage}")
    root = authority_root.expanduser().resolve()
    if root.name in set(_AUTHORITY_STAGE_DIRS.values()):
        raise SizingFinalizationError(
            "--authority-root must name the family root, not a stage directory"
        )
    path = root / _AUTHORITY_STAGE_DIRS[stage]
    path.mkdir(parents=True, exist_ok=True)
    return path


def _terminal_event_attempt(stage: str, prefix: str):
    """Make every failed refresh newer than any prior PASS in that family."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                if isinstance(exc, _TerminalSizingEventPublished):
                    raise
                authority_root = kwargs.get("authority_root")
                if authority_root is not None:
                    try:
                        inputs = {
                            key: str(value.expanduser().resolve())
                            if isinstance(value, Path)
                            else str(value)
                            for key, value in kwargs.items()
                            if key not in {"client"} and value is not None
                        }
                        failure_dir = _stage_dir(Path(authority_root), stage)
                        failure_created = _monotonic_stage_created_utc(
                            failure_dir, prefix
                        )
                        failure_payload = {
                            "schema_version": (
                                "entry_model_native_sizing_terminal_failure_v1"
                            ),
                            "created_utc": failure_created.isoformat(),
                            "decision": "FAIL",
                            "failures": [f"{type(exc).__name__}: {exc}"],
                            "attempted_stage": stage,
                            "inputs": inputs,
                        }
                        write_immutable_json_event(
                            failure_dir,
                            prefix,
                            failure_payload,
                        )
                    except Exception as publication_exc:
                        exc.add_note(
                            "terminal FAIL publication also failed: "
                            f"{publication_exc}"
                        )
                raise

        return wrapped

    return decorate


def _require_stage_path(path: Path, authority_root: Path, stage: str) -> None:
    expected = _stage_dir(authority_root, stage)
    if path.expanduser().resolve().parent != expected:
        raise SizingFinalizationError(
            f"{stage} artifact is outside the bound authority family: {path}"
        )


def _dataset_split_bindings(
    prediction_report: dict[str, Any],
    dataset_dir: Path,
    expected_splits: tuple[str, ...],
) -> dict[str, dict[str, str]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SizingFinalizationError(f"dataset_dir missing: {dataset_dir}")
    contract = prediction_report.get("dataset_signal_contract")
    report_splits = contract.get("splits") if isinstance(contract, dict) else None
    if not isinstance(report_splits, dict) or set(report_splits) != set(
        expected_splits
    ):
        raise SizingFinalizationError(
            "prediction report dataset split bindings are not exact"
        )
    rows: dict[str, dict[str, str]] = {}
    for split in expected_splits:
        report_row = report_splits[split]
        required = {
            "manifest_path",
            "manifest_sha256",
            "parquet_path",
            "parquet_sha256",
        }
        if not isinstance(report_row, dict) or not required.issubset(report_row):
            raise SizingFinalizationError(
                f"prediction report {split} dataset artifact binding is incomplete"
            )
        rows[split] = {
            key: str(report_row[key])
            for key in sorted(required)
        }
    return rows


def _prediction_provenance(
    *,
    predictions_path: Path,
    prediction_report_path: Path,
    bundle_dir: Path,
    dataset_dir: Path,
    expected_splits: tuple[str, ...],
    require_runtime_head_evidence: bool,
    context: str,
) -> tuple[Path, dict[str, Any]]:
    predictions_path = predictions_path.expanduser().resolve()
    prediction_report_path = prediction_report_path.expanduser().resolve()
    bundle_dir = bundle_dir.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()
    try:
        authoritative, report, evidence = resolve_and_validate_prediction_evidence(
            predictions_path,
            prediction_report_path=prediction_report_path,
            bundle_dir=bundle_dir,
            dataset_dir=dataset_dir,
            expected_model="candidate",
            require_runtime_head_evidence=require_runtime_head_evidence,
        )
    except Exception as exc:
        raise SizingFinalizationError(
            f"{context} canonical prediction evidence invalid: {exc}"
        ) from exc
    if sorted(str(value) for value in evidence.get("splits") or []) != sorted(
        expected_splits
    ) or list(evidence.get("models") or []) != ["candidate"]:
        raise SizingFinalizationError(
            f"{context} requires exact splits={expected_splits} and candidate only"
        )
    provenance = {
        "prediction_report_artifact": _binding(prediction_report_path),
        "bundle_dir": str(bundle_dir),
        "dataset_dir": str(dataset_dir),
        "dataset_split_bindings": _dataset_split_bindings(
            report,
            dataset_dir,
            expected_splits,
        ),
    }
    require_sizing_prediction_provenance(
        provenance,
        predictions_binding=_source_binding(authoritative),
        expected_splits=expected_splits,
        require_runtime_head_evidence=require_runtime_head_evidence,
        context=context,
        verify_files=True,
    )
    return authoritative, provenance


def _evaluation_bundle(
    bundle_dir: Path, calibration: dict[str, Any], *, context: str
) -> dict[str, str]:
    bundle_dir = bundle_dir.expanduser().resolve()
    metadata = bundle_dir / "bundle_metadata.json"
    lock = bundle_dir / "MASTER_TRANSFORMER_LOCK.json"
    state = bundle_dir / "model_state_dict.pt"
    for path in (metadata, lock, state):
        if not path.is_file():
            raise SizingFinalizationError(f"evaluation bundle file missing: {path}")
    value = {
        "bundle_dir": str(bundle_dir),
        "bundle_metadata_path": str(metadata),
        "bundle_metadata_sha256": _sha(metadata),
        "master_transformer_lock_path": str(lock),
        "master_transformer_lock_sha256": _sha(lock),
        "model_state_dict_path": str(state),
        "model_state_dict_sha256": _sha(state),
    }
    require_sizing_evaluation_bundle(
        value,
        calibration=calibration,
        context=context,
        verify_files=True,
    )
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SizingFinalizationError(f"{label} unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SizingFinalizationError(f"{label} root is not an object: {path}")
    return payload


def _read_table(path: Path) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise SizingFinalizationError(f"table must be CSV/parquet: {path}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(tmp_raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@_terminal_event_attempt("instrument", INSTRUMENT_EVIDENCE_PREFIX)
def capture_oanda_instrument_evidence(
    *, authority_root: Path, client: Any | None = None
) -> tuple[Path, dict[str, Any]]:
    """Capture account-specific OANDA XAU constraints without a hand JSON step."""

    if client is None:
        from gx1.execution.oanda_client import OandaClient

        client = OandaClient.from_env()
    account_payload = client.get_account_summary()
    account_observed = _now()
    instrument_payload = client.get_account_instruments(["XAU_USD"])
    instrument_observed = _strictly_after_now(account_observed)
    account = account_payload.get("account") if isinstance(account_payload, dict) else None
    rows = (
        instrument_payload.get("instruments")
        if isinstance(instrument_payload, dict)
        else None
    )
    matches = [
        row
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and row.get("name") == "XAU_USD"
    ]
    if not isinstance(account, dict) or len(matches) != 1:
        raise SizingFinalizationError("broker account/instrument evidence is incomplete")
    instrument = matches[0]
    try:
        precision = int(instrument["tradeUnitsPrecision"])
        minimum = int(str(instrument["minimumTradeSize"]))
        broker_maximum = int(str(instrument["maximumOrderUnits"]))
        margin_rate = float(instrument["marginRate"])
        account_currency = str(account["currency"])
        account_tx = str(account_payload["lastTransactionID"])
        instrument_tx = str(instrument_payload["lastTransactionID"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SizingFinalizationError("broker instrument fields are not exact") from exc
    if (
        precision != 0
        or minimum != 1
        or broker_maximum < MODEL_NATIVE_SIZING_MAX_GROSS_XAU_UNITS
        or not math.isfinite(margin_rate)
        or not 0.0 < margin_rate <= 1.0
        or account_currency != "USD"
        or not account_tx
        or not instrument_tx
    ):
        raise SizingFinalizationError("broker XAU constraints fail immutable policy")
    output_dir = _stage_dir(authority_root, "instrument")
    created = _monotonic_stage_created_utc(
        output_dir,
        INSTRUMENT_EVIDENCE_PREFIX,
        account_observed,
        instrument_observed,
    )
    payload = {
        "schema_version": INSTRUMENT_EVIDENCE_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": "PASS",
        "failures": [],
        "instrument": "XAU_USD",
        "account_currency": account_currency,
        "quote_currency": "USD",
        "trade_units_precision": precision,
        "minimum_trade_size": minimum,
        "broker_maximum_order_units": broker_maximum,
        "margin_rate": margin_rate,
        "account_observed_utc": account_observed.isoformat(),
        "instrument_observed_utc": instrument_observed.isoformat(),
        "account_last_transaction_id": account_tx,
        "instrument_last_transaction_id": instrument_tx,
    }
    return write_immutable_json_event(
        output_dir,
        INSTRUMENT_EVIDENCE_PREFIX,
        payload,
    )


def require_instrument_evidence(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        payload = require_sizing_instrument_evidence_artifact(
            path,
            _sha(path),
            context="SIZING_FINALIZER_INSTRUMENT",
            verify_file=True,
        )
    except Exception as exc:
        raise SizingFinalizationError(str(exc)) from exc
    if payload is None:
        raise SizingFinalizationError("instrument evidence unexpectedly unavailable")
    return payload


@_terminal_event_attempt("calibration", CALIBRATION_PREFIX)
def fit_train_val_sizing_calibration(
    *,
    predictions_path: Path,
    prediction_report_path: Path,
    bundle_dir: Path,
    dataset_dir: Path,
    dataset_manifest_path: Path,
    instrument_evidence_path: Path,
    authority_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Fit only TRAIN/VAL rows and publish an immutable calibration event."""

    predictions_path, fit_provenance = _prediction_provenance(
        predictions_path=predictions_path,
        prediction_report_path=prediction_report_path,
        bundle_dir=bundle_dir,
        dataset_dir=dataset_dir,
        expected_splits=MODEL_NATIVE_SIZING_FIT_SPLITS,
        require_runtime_head_evidence=True,
        context="SIZING_FINALIZER_FIT_PREDICTIONS",
    )
    frame = _read_table(predictions_path)
    required = {
        "time",
        "split",
        "model",
        "position_size_logit",
        "y_position_size_target",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SizingFinalizationError(f"fit predictions missing columns: {missing}")
    if set(frame["split"].astype(str)) != set(MODEL_NATIVE_SIZING_FIT_SPLITS) or set(
        frame["model"].astype(str)
    ) != {"candidate"}:
        raise SizingFinalizationError(
            "fit prediction source must be exact candidate TRAIN+VAL only"
        )
    selected = frame.loc[
        frame["split"].astype(str).isin(MODEL_NATIVE_SIZING_FIT_SPLITS),
        sorted(required),
    ].copy()
    selected_times = pd.to_datetime(selected["time"], utc=True, errors="coerce")
    if (
        selected_times.isna().any()
        or selected_times.duplicated().any()
        or not selected_times.is_monotonic_increasing
    ):
        raise SizingFinalizationError(
            "fit prediction rows must have unique increasing UTC time"
        )
    counts = selected.groupby("split", observed=True).size().to_dict()
    if any(int(counts.get(split, 0)) < MIN_FIT_ROWS_PER_SPLIT for split in ("train", "val")):
        raise SizingFinalizationError(
            f"TRAIN/VAL calibration support below {MIN_FIT_ROWS_PER_SPLIT}: {counts}"
        )
    logits = pd.to_numeric(selected["position_size_logit"], errors="coerce").to_numpy(float)
    raw_targets = pd.to_numeric(
        selected["y_position_size_target"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(raw_targets).all() or np.any((raw_targets < 0.0) | (raw_targets > 1.0)):
        raise SizingFinalizationError("y_position_size_target must be finite in [0,1]")
    parameters = fit_monotone_sizing_parameters(
        logits,
        raw_targets * MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION,
    )
    authority_root = authority_root.expanduser().resolve()
    _require_stage_path(instrument_evidence_path, authority_root, "instrument")
    instrument = require_instrument_evidence(instrument_evidence_path)
    output_dir = _stage_dir(authority_root, "calibration")
    created = _monotonic_stage_created_utc(
        output_dir,
        CALIBRATION_PREFIX,
        instrument["created_utc"],
    )
    dataset_manifest_path = dataset_manifest_path.expanduser().resolve()
    expected_train_manifest = Path(
        fit_provenance["dataset_split_bindings"]["train"]["manifest_path"]
    )
    if dataset_manifest_path != expected_train_manifest:
        raise SizingFinalizationError(
            "dataset_manifest must be exact TRAIN manifest from prediction evidence"
        )
    model_checkpoint_path = bundle_dir.expanduser().resolve() / "model_state_dict.pt"
    lineage_paths = {
        "dataset_manifest": dataset_manifest_path.expanduser().resolve(),
        "fit_predictions": predictions_path,
        "model_checkpoint": model_checkpoint_path,
        "instrument_evidence": instrument_evidence_path.expanduser().resolve(),
    }
    for label, path in lineage_paths.items():
        if not path.is_file():
            raise SizingFinalizationError(f"{label} lineage file missing: {path}")
    payload = {
        "schema_version": MODEL_NATIVE_SIZING_CALIBRATION_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "fit_scope": MODEL_NATIVE_SIZING_FIT_SCOPE,
        "fit_splits": ["train", "val"],
        "holdout_split": MODEL_NATIVE_SIZING_HOLDOUT_SPLIT,
        "source_head": "position_size_logit",
        "transform_version": MODEL_NATIVE_SIZING_TRANSFORM_VERSION,
        "monotonic_direction": "non_decreasing",
        "instrument_constraints": {
            "instrument": "XAU_USD",
            "account_currency": "USD",
            "quote_currency": "USD",
            "unit_step": 1,
            "minimum_order_units": int(instrument["minimum_trade_size"]),
            "maximum_gross_xau_units": MODEL_NATIVE_SIZING_MAX_GROSS_XAU_UNITS,
            "margin_rate": float(instrument["margin_rate"]),
        },
        "parameters": parameters,
        "fit_contract": sizing_fit_contract_metadata(),
        "fit_prediction_provenance": fit_provenance,
        "lineage": {
            key: value
            for stem, path in lineage_paths.items()
            for key, value in (
                (f"{stem}_path", str(path)),
                (f"{stem}_sha256", _sha(path)),
            )
        },
    }
    event_path, event = write_immutable_json_event(
        output_dir, CALIBRATION_PREFIX, payload
    )
    load_bound_sizing_calibration(
        _binding(event_path),
        context="SIZING_FINALIZER_CALIBRATION",
        verify_lineage_files=True,
    )
    return event_path, event


def bind_bundle_sizing_calibration(
    *, source_bundle_dir: Path, output_bundle_dir: Path, calibration_path: Path
) -> dict[str, Any]:
    """Clone an immutable source bundle into a fresh sizing candidate bundle."""

    source_bundle_input = source_bundle_dir.expanduser()
    if source_bundle_input.is_symlink():
        raise SizingFinalizationError("source bundle root must not be a symlink")
    source_bundle_dir = source_bundle_input.resolve()
    output_bundle_dir = output_bundle_dir.expanduser().resolve()
    if not source_bundle_dir.is_dir():
        raise SizingFinalizationError(f"source bundle missing: {source_bundle_dir}")
    if output_bundle_dir.exists():
        raise SizingFinalizationError(
            f"output bundle must not already exist: {output_bundle_dir}"
        )
    if output_bundle_dir == source_bundle_dir or source_bundle_dir in output_bundle_dir.parents:
        raise SizingFinalizationError("output bundle cannot be source or nested inside source")
    calibration_path = calibration_path.expanduser().resolve()
    calibration_binding = _binding(calibration_path)
    calibration, _ = load_bound_sizing_calibration(
        calibration_binding,
        context="SIZING_FINALIZER_BUNDLE_BIND",
        verify_lineage_files=True,
    )
    if Path(calibration["fit_prediction_provenance"]["bundle_dir"]).resolve() != source_bundle_dir:
        raise SizingFinalizationError(
            "source bundle differs from canonical TRAIN/VAL fit provenance"
        )
    source_state = source_bundle_dir / "model_state_dict.pt"
    source_metadata = source_bundle_dir / "bundle_metadata.json"
    source_lock = source_bundle_dir / "MASTER_TRANSFORMER_LOCK.json"
    try:
        source_commit = require_bundle_commit_manifest(source_bundle_dir)
    except RuntimeError as exc:
        raise SizingFinalizationError(str(exc)) from exc
    source_inventory = {
        *source_commit["artifact_names"],
        BUNDLE_COMMIT_MANIFEST_NAME,
    }
    inherited_artifacts = sorted(
        set(source_commit["artifact_names"]) - set(BUNDLE_COMMIT_CORE_ARTIFACTS)
    )
    output_inventory = {
        *BUNDLE_COMMIT_CORE_ARTIFACTS,
        *inherited_artifacts,
        BUNDLE_COMMIT_MANIFEST_NAME,
    }
    source_entries = list(source_bundle_dir.iterdir())
    observed_inventory = {path.name for path in source_entries}
    if observed_inventory != source_inventory:
        raise SizingFinalizationError(
            "source bundle inventory must be exact code-owned files; "
            f"expected={sorted(source_inventory)} observed={sorted(observed_inventory)}"
        )
    if any(path.is_symlink() or not path.is_file() for path in source_entries):
        raise SizingFinalizationError(
            "source bundle inventory must contain regular non-symlink files only"
        )
    checkpoint_sha = calibration["lineage"]["model_checkpoint_sha256"]
    if (
        Path(calibration["lineage"]["model_checkpoint_path"]).resolve() != source_state
        or not source_state.is_file()
        or _sha(source_state) != checkpoint_sha
    ):
        raise SizingFinalizationError("source bundle checkpoint differs from calibration")
    for path in (source_metadata, source_lock):
        if not path.is_file():
            raise SizingFinalizationError(f"source bundle file missing: {path}")
    source_metadata_payload = _read_json(source_metadata, label="source bundle metadata")
    source_lock_payload = _read_json(source_lock, label="source transformer lock")
    for head in ("direction", "path"):
        key = f"{head}_calibration"
        try:
            require_model_native_calibration_metadata(
                source_metadata_payload.get(key),
                head=head,
                context=f"SIZING_SOURCE_{head.upper()}_CALIBRATION",
            )
        except RuntimeError as exc:
            raise SizingFinalizationError(str(exc)) from exc
    calibration_events = [
        name
        for name in inherited_artifacts
        if name.startswith(ENTRY_CALIBRATION_EVENT_PREFIX) and name.endswith(".json")
    ]
    if source_commit["bundle_kind"] != "calibrated" or len(calibration_events) < 2:
        raise SizingFinalizationError(
            "source must be the canonical direction+path calibrated bundle "
            "with both committed calibration events"
        )
    calibrated_heads: set[str] = set()
    for event_name in calibration_events:
        event = _read_json(
            source_bundle_dir / event_name,
            label=f"source calibration event {event_name}",
        )
        head = event.get("head")
        if (
            event.get("schema_version")
            != IMMUTABLE_CALIBRATION_EVENT_SCHEMA_VERSION
            or event.get("decision") != "PASS"
            or head not in {"direction", "path"}
            or head in calibrated_heads
            or event.get("calibration")
            != source_metadata_payload.get(f"{head}_calibration")
        ):
            raise SizingFinalizationError(
                f"source calibration event is not canonical: {event_name}"
            )
        calibrated_heads.add(str(head))
    if calibrated_heads != {"direction", "path"}:
        raise SizingFinalizationError(
            "source calibration events do not prove both direction and path"
        )
    if str(source_metadata_payload.get("state_dict_sha256") or "").lower() != checkpoint_sha:
        raise SizingFinalizationError("source metadata state_dict_sha256 mismatch")
    if (
        str(source_lock_payload.get("model_sha256") or "").lower() != checkpoint_sha
        or source_lock_payload.get("model_path_relative") != "model_state_dict.pt"
    ):
        raise SizingFinalizationError("source transformer lock checkpoint mismatch")
    if "model_native_sizing_calibration" in source_metadata_payload or (
        "model_native_sizing_calibration" in source_lock_payload
    ):
        raise SizingFinalizationError(
            "source bundle already contains sizing calibration; use pristine source"
        )
    declaration = model_native_sizing_bundle_calibration_metadata(
        calibration_artifact=calibration_binding
    )
    output_bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_bundle_dir.name}.staging.", dir=str(output_bundle_dir.parent)
        )
    )
    try:
        for name in sorted({*BUNDLE_COMMIT_CORE_ARTIFACTS, *inherited_artifacts}):
            shutil.copy2(source_bundle_dir / name, staging / name)
        paths = (
            staging / "bundle_metadata.json",
            staging / "MASTER_TRANSFORMER_LOCK.json",
        )
        payloads: list[dict[str, Any]] = []
        for path in paths:
            payload = _read_json(path, label=path.name)
            if "model_native_sizing_authority" in payload:
                raise SizingFinalizationError(
                    f"post-bundle sizing authority must not be embedded: {path}"
                )
            payload["model_native_sizing_calibration"] = declaration
            payloads.append(payload)
        for path, payload in zip(paths, payloads, strict=True):
            _atomic_json(path, payload)
        write_bundle_commit_manifest(
            bundle_dir=staging.resolve(),
            artifact_names=tuple(
                sorted({*BUNDLE_COMMIT_CORE_ARTIFACTS, *inherited_artifacts})
            ),
            bundle_kind="sizing_finalized",
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        try:
            publish_bundle_directory_noreplace(staging, output_bundle_dir)
        except RuntimeError as exc:
            raise SizingFinalizationError(str(exc)) from exc
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    paths = (
        output_bundle_dir / "bundle_metadata.json",
        output_bundle_dir / "MASTER_TRANSFORMER_LOCK.json",
    )
    if {path.name for path in output_bundle_dir.iterdir()} != output_inventory:
        raise SizingFinalizationError("finalized bundle inventory parity failed")
    require_bundle_commit_manifest(output_bundle_dir)
    metadata = _read_json(paths[0], label="finalized bundle metadata")
    lock = _read_json(paths[1], label="finalized transformer lock")
    for label, payload in (("metadata", metadata), ("lock", lock)):
        observed = require_model_native_sizing_bundle_calibration(
            payload.get("model_native_sizing_calibration"),
            context=f"SIZING_FINALIZER_{label.upper()}",
        )
        if observed != declaration:
            raise SizingFinalizationError(f"{label} finalization parity failed")
    return {
        "source_bundle_dir": str(source_bundle_dir),
        "bundle_dir": str(output_bundle_dir),
        "bundle_metadata_sha256": _sha(paths[0]),
        "master_transformer_lock_sha256": _sha(paths[1]),
        "model_native_sizing_calibration": declaration,
        "next_required_stage": "final_bundle_audits_then_full_test_oos_sizing_proof",
    }


@_terminal_event_attempt("oos", OOS_SOURCE_PREFIX)
def materialize_test_sizing_oos_source(
    *,
    calibration_path: Path,
    test_predictions_path: Path,
    test_prediction_report_path: Path,
    bundle_dir: Path,
    dataset_dir: Path,
    source_tape_path: Path,
    model_head_serve_parity_path: Path,
    authority_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Publish the sole canonical full-TEST sizing row source."""

    authority_root = authority_root.expanduser().resolve()
    calibration_path = calibration_path.expanduser().resolve()
    _require_stage_path(calibration_path, authority_root, "calibration")
    calibration_binding = _binding(calibration_path)
    calibration, _ = load_bound_sizing_calibration(
        calibration_binding,
        context="SIZING_FINALIZER_OOS_CALIBRATION",
        verify_lineage_files=True,
    )
    predictions_path, provenance = _prediction_provenance(
        predictions_path=test_predictions_path,
        prediction_report_path=test_prediction_report_path,
        bundle_dir=bundle_dir,
        dataset_dir=dataset_dir,
        expected_splits=("test",),
        require_runtime_head_evidence=True,
        context="SIZING_FINALIZER_OOS_PREDICTIONS",
    )
    prediction_binding = _source_binding(predictions_path)
    evaluation = _evaluation_bundle(
        bundle_dir, calibration, context="SIZING_FINALIZER_OOS_BUNDLE"
    )
    source_tape_path = source_tape_path.expanduser().resolve()
    test_manifest = provenance["dataset_split_bindings"]["test"]
    source_tape = {
        "path": str(source_tape_path),
        "sha256": _sha(source_tape_path),
        "dataset_test_manifest_path": test_manifest["manifest_path"],
        "dataset_test_manifest_sha256": test_manifest["manifest_sha256"],
    }
    model_head_serve_parity = _binding(
        model_head_serve_parity_path.expanduser().resolve()
    )
    rows = derive_canonical_sizing_oos_rows(
        calibration=calibration,
        test_predictions_binding=prediction_binding,
        test_prediction_provenance=provenance,
        evaluation_bundle=evaluation,
        source_tape=source_tape,
        model_head_serve_parity_artifact=model_head_serve_parity,
        context="SIZING_FINALIZER_OOS_DERIVE",
    )
    output_dir = _stage_dir(authority_root, "oos")
    created = _monotonic_stage_created_utc(
        output_dir,
        OOS_SOURCE_PREFIX,
        calibration["created_utc"],
    )
    stamp = created.strftime("%Y%m%dT%H%M%S%fZ")
    rows_path = output_dir / f"entry_model_native_sizing_oos_rows_{stamp}.parquet"
    atomic_write_parquet_immutable(rows, rows_path)
    source_bindings = {"oos_rows": _source_binding(rows_path)}
    payload = {
        "schema_version": MODEL_NATIVE_SIZING_OOS_SOURCE_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": "PASS",
        "failures": [],
        "calibration_artifact_sha256": calibration_binding["sha256"],
        "test_predictions": prediction_binding,
        "test_prediction_provenance": provenance,
        "evaluation_bundle": evaluation,
        "source_tape": source_tape,
        "model_head_serve_parity_artifact": model_head_serve_parity,
        "reference_account_policy": sizing_oos_reference_account_policy_metadata(),
        "source_bindings": source_bindings,
    }
    event_path, event = write_immutable_json_event(
        output_dir, OOS_SOURCE_PREFIX, payload
    )
    load_bound_sizing_oos_source(
        _binding(event_path),
        calibration=calibration,
        calibration_artifact_sha256=calibration_binding["sha256"],
        context="SIZING_FINALIZER_OOS_SOURCE",
        verify_source_files=True,
    )
    return event_path, event


@_terminal_event_attempt("proof", PROOF_PREFIX)
def finalize_test_sizing_proof(
    *,
    calibration_path: Path,
    oos_source_path: Path,
    authority_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Recompute exact row-level TEST evidence and publish PASS or terminal FAIL."""

    calibration_path = calibration_path.expanduser().resolve()
    authority_root = authority_root.expanduser().resolve()
    _require_stage_path(calibration_path, authority_root, "calibration")
    _require_stage_path(oos_source_path, authority_root, "oos")
    calibration_binding = _binding(calibration_path)
    calibration, _ = load_bound_sizing_calibration(
        calibration_binding,
        context="SIZING_FINALIZER_PROOF_CALIBRATION",
        verify_lineage_files=True,
    )
    oos_source_path = oos_source_path.expanduser().resolve()
    oos_source_binding = _binding(oos_source_path)
    source_event, _ = load_bound_sizing_oos_source(
        oos_source_binding,
        calibration=calibration,
        calibration_artifact_sha256=calibration_binding["sha256"],
        context="SIZING_FINALIZER_TEST_SOURCE",
        verify_source_files=True,
    )
    sources = source_event["source_bindings"]
    sections = (
        "full_test_coverage",
        "position_size_head_liveness",
        "monotonicity",
        "exposure_bounds",
        "drawdown_bounds",
        "paired_oos_utility",
        "account_capacity_grid",
        "direction_invariance",
    )
    failures: list[str] = []
    try:
        recomputed = recompute_sizing_oos_evidence(
            calibration=calibration,
            source_bindings=sources,
            evaluation_bundle=source_event["evaluation_bundle"],
            context="SIZING_FINALIZER_TEST_RECOMPUTE",
        )
        failures.extend(
            name
            for name in sections[1:]
            if recomputed[name].get("decision") != "PASS"
        )
    except Exception as exc:
        failures.append(f"row_recompute_failed:{type(exc).__name__}:{exc}")
        recomputed = {
            "full_test_coverage": {"rows": 0, "error": str(exc)},
            **{
                name: {"decision": "FAIL", "error": str(exc)}
                for name in sections[1:]
            },
        }
    output_dir = _stage_dir(authority_root, "proof")
    created = _monotonic_stage_created_utc(
        output_dir,
        PROOF_PREFIX,
        source_event["created_utc"],
    )
    payload = {
        "schema_version": MODEL_NATIVE_SIZING_OOS_PROOF_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "evaluation_scope": MODEL_NATIVE_SIZING_OOS_SCOPE,
        "evaluation_split": MODEL_NATIVE_SIZING_HOLDOUT_SPLIT,
        "calibration_artifact_sha256": calibration_binding["sha256"],
        "risk_policy": sizing_risk_policy_metadata(),
        "source_bindings": sources,
        "test_prediction_provenance": source_event[
            "test_prediction_provenance"
        ],
        "evaluation_bundle": source_event["evaluation_bundle"],
        "oos_source_artifact": oos_source_binding,
        **recomputed,
    }
    event_path, event = write_immutable_json_event(
        output_dir, PROOF_PREFIX, payload
    )
    if failures:
        raise _TerminalSizingEventPublished(
            f"terminal sizing proof FAIL published at {event_path}: {failures[:5]}"
        )
    load_bound_sizing_oos_proof(
        _binding(event_path),
        calibration=calibration,
        calibration_artifact_sha256=calibration_binding["sha256"],
        context="SIZING_FINALIZER_TEST_PROOF",
        verify_source_files=True,
    )
    return event_path, event


def _finalize_joint_exit_sizing_proof(
    *,
    calibration_path: Path,
    proof_path: Path,
    replay_rows_path: Path,
    exit_trace_rows_path: Path,
    authority_root: Path,
    canonical_producer_evidence: dict[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    """Publish row-recomputed sizing evidence from the exact candidate bundle.

    This stage does not accept a caller-supplied PASS or summary.  Every metric
    is recomputed from immutable full-TEST replay and per-M1 Exit trace
    parquets. The OOS proof's immutable candidate bundle binds both Entry and
    the unified HOLD/EXIT_NOW head before any activation is possible.
    """

    calibration, calibration_binding = load_bound_sizing_calibration(
        _binding(calibration_path),
        context="SIZING_JOINT_EXIT_CALIBRATION",
        verify_lineage_files=True,
    )
    proof, proof_binding = load_bound_sizing_oos_proof(
        _binding(proof_path),
        calibration=calibration,
        calibration_artifact_sha256=calibration_binding["sha256"],
        context="SIZING_JOINT_EXIT_OOS_PROOF",
        verify_source_files=True,
    )
    _require_stage_path(
        Path(calibration_binding["json_path"]), authority_root, "calibration"
    )
    _require_stage_path(Path(proof_binding["json_path"]), authority_root, "proof")

    replay_rows_path, replay_binding = _canonical_immutable_parquet_binding(
        replay_rows_path,
        context="joint unified-Exit replay rows",
    )
    exit_trace_rows_path, exit_trace_binding = (
        _canonical_immutable_parquet_binding(
            exit_trace_rows_path,
            context="joint unified-Exit trace rows",
        )
    )
    bundle_authority = candidate_bundle_authority(
        bundle_dir=Path(str(proof["evaluation_bundle"]["bundle_dir"])),
        evaluation_bundle=proof["evaluation_bundle"],
        context="SIZING_JOINT_EXIT_CANDIDATE_BUNDLE",
    )

    replay_rows = read_bound_parquet_exact(
        replay_binding,
        context="SIZING_JOINT_EXIT_REPLAY_ROWS_EXACT",
    )
    exit_trace_rows = read_bound_parquet_exact(
        exit_trace_binding,
        context="SIZING_JOINT_EXIT_TRACE_ROWS_EXACT",
    )
    canonical_oos_rows = read_bound_parquet_exact(
        proof["source_bindings"]["oos_rows"],
        context="SIZING_JOINT_EXIT_CANONICAL_OOS_ROWS_EXACT",
    )
    require_joint_replay_extends_canonical_oos_rows(
        canonical_oos_rows=canonical_oos_rows,
        replay_rows=replay_rows,
        context="SIZING_JOINT_EXIT_CANONICAL_OOS_IDENTITY",
    )
    coverage = recompute_joint_exit_replay_coverage(
        replay_rows,
        exit_trace_rows=exit_trace_rows,
        candidate_bundle_sha256=bundle_authority["bundle_commit_sha256"],
        context="SIZING_JOINT_EXIT_COVERAGE",
    )
    recomputed = recompute_sizing_oos_evidence(
        calibration=calibration,
        source_bindings={"oos_rows": replay_binding},
        evaluation_bundle=proof["evaluation_bundle"],
        context="SIZING_JOINT_EXIT_RECOMPUTE",
        fact_provenance_mode=MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE,
        extra_row_columns=MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS,
        outcome_price_mode="model_exit_fill",
    )
    for name, section in recomputed.items():
        if name != "full_test_coverage" and (
            not isinstance(section, dict) or section.get("decision") != "PASS"
        ):
            raise SizingFinalizationError(
                f"joint unified-Exit sizing section {name} is not PASS"
            )

    output_dir = _stage_dir(authority_root, "joint_replay")
    created = _monotonic_stage_created_utc(
        output_dir,
        JOINT_EXIT_PROOF_PREFIX,
        proof["created_utc"],
    )
    payload = {
        "schema_version": MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": "PASS",
        "failures": [],
        "replay_contract": MODEL_NATIVE_JOINT_EXIT_SIZING_REPLAY_CONTRACT,
        "risk_policy": sizing_risk_policy_metadata(),
        "calibration_artifact": calibration_binding,
        "oos_proof_artifact": proof_binding,
        "evaluation_bundle": proof["evaluation_bundle"],
        "test_prediction_provenance": proof["test_prediction_provenance"],
        "candidate_bundle_authority": bundle_authority,
        "canonical_unified_replay_producer": canonical_producer_evidence,
        "replay_rows": replay_binding,
        "exit_trace_rows": exit_trace_binding,
        "exit_replay_coverage": coverage,
        **recomputed,
    }
    event_path, event = write_immutable_json_event(
        output_dir,
        JOINT_EXIT_PROOF_PREFIX,
        payload,
    )
    load_bound_joint_exit_sizing_proof(
        _binding(event_path),
        context="SIZING_JOINT_EXIT_SELF_VALIDATION",
        verify_source_files=True,
    )
    return event_path, event


@_terminal_event_attempt("joint_replay", JOINT_EXIT_PROOF_PREFIX)
def finalize_joint_exit_sizing_proof(
    *,
    calibration_path: Path,
    proof_path: Path,
    replay_rows_path: Path,
    exit_trace_rows_path: Path,
    authority_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Validate caller rows as diagnostic-only joint Exit evidence.

    This retained compatibility route can never populate canonical producer
    evidence and therefore has zero launch authority.
    """

    return _finalize_joint_exit_sizing_proof(
        calibration_path=calibration_path,
        proof_path=proof_path,
        replay_rows_path=replay_rows_path,
        exit_trace_rows_path=exit_trace_rows_path,
        authority_root=authority_root,
        canonical_producer_evidence=None,
    )


@_terminal_event_attempt("joint_replay", JOINT_EXIT_PROOF_PREFIX)
def produce_canonical_unified_joint_sizing_proof(
    *,
    calibration_path: Path,
    proof_path: Path,
    source_tape_path: Path,
    prebuilt_pair_manifest_path: Path,
    prebuilt_generation_root: Path,
    train_rank_reference_npz: Path,
    train_rank_reference_sha256: str,
    authority_root: Path,
    device: str = "cpu",
) -> tuple[Path, dict[str, Any]]:
    """Produce exact full-TEST Entry/Exit replay from one candidate bundle.

    Direction is replayed from the persisted model head envelope. Every
    non-FLAT row then advances the production TradeState one complete M1 bar at
    a time and calls the same bundle's ``forward_exit_action`` path until the
    model itself emits EXIT_NOW. Missing bars, 512-bar non-exits, byte drift,
    direction mismatch, or a noncanonical lineage artifact fail closed.
    """

    from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
        decode_model_native_runtime_head_evidence,
    )
    from gx1.contracts.entry_model_native_state_v2 import (
        load_train_rank_reference_v2,
        train_rank_reference_identity_v2,
    )
    from gx1.execution.model_native_entry_replay_v1 import SourceTape
    from gx1.execution.v12_smart_entry_live import SmartEntryLiveInference
    from gx1.execution.v12_state_from_prebuilt import (
        read_prebuilt_pair_manifest,
        verify_prebuilt_pair,
    )
    from gx1.execution.v12_trade_state import (
        TradeState,
        first_full_closed_m1_bar_ts,
    )
    from gx1.models.entry_v10.direction_decision_contract import (
        UNIFIED_EXIT_MAX_PATH_BARS,
    )

    for label, path in (
        ("calibration", calibration_path),
        ("OOS proof", proof_path),
    ):
        candidate = Path(path).expanduser()
        if candidate.is_symlink() or not candidate.is_file():
            raise SizingFinalizationError(
                f"source file missing for {label}: {candidate}"
            )
    calibration, calibration_binding = load_bound_sizing_calibration(
        _binding(calibration_path),
        context="UNIFIED_REPLAY_CALIBRATION",
        verify_lineage_files=True,
    )
    proof, proof_binding = load_bound_sizing_oos_proof(
        _binding(proof_path),
        calibration=calibration,
        calibration_artifact_sha256=calibration_binding["sha256"],
        context="UNIFIED_REPLAY_OOS_PROOF",
        verify_source_files=True,
    )
    _require_stage_path(
        Path(calibration_binding["json_path"]),
        authority_root,
        "calibration",
    )
    _require_stage_path(
        Path(proof_binding["json_path"]),
        authority_root,
        "proof",
    )
    oos_source, _oos_source_binding = load_bound_sizing_oos_source(
        proof["oos_source_artifact"],
        calibration=calibration,
        calibration_artifact_sha256=calibration_binding["sha256"],
        context="UNIFIED_REPLAY_OOS_SOURCE",
        verify_source_files=True,
    )
    tape = SourceTape.load(source_tape_path)
    expected_tape = oos_source["source_tape"]
    if (
        str(tape.source_path) != str(expected_tape["path"])
        or tape.source_sha256 != expected_tape["sha256"]
    ):
        raise SizingFinalizationError(
            "SourceTape differs from the immutable OOS source"
        )

    provenance = proof["test_prediction_provenance"]
    report_binding = provenance["prediction_report_artifact"]
    runtime_predictions_path, prediction_report, declaration = (
        resolve_and_validate_prediction_evidence(
            Path(oos_source["test_predictions"]["path"]),
            prediction_report_path=Path(report_binding["json_path"]),
            bundle_dir=Path(str(provenance["bundle_dir"])),
            dataset_dir=Path(str(provenance["dataset_dir"])),
            expected_model="candidate",
            require_runtime_head_evidence=True,
        )
    )
    if (
        sha256_file(runtime_predictions_path)
        != oos_source["test_predictions"]["sha256"]
        or prediction_report.get("prediction_evidence") != declaration
        or list(declaration.get("models") or []) != ["candidate"]
        or list(declaration.get("splits") or []) != ["test"]
    ):
        raise SizingFinalizationError(
            "runtime prediction evidence differs from the OOS proof"
        )

    serve_parity_binding = require_immutable_json_binding(
        oos_source["model_head_serve_parity_artifact"],
        event_prefix="MODEL_NATIVE_SERVE_PARITY",
        context="UNIFIED_REPLAY_SERVE_PARITY",
        verify_file=True,
    )
    try:
        serve_parity = json.loads(
            Path(serve_parity_binding["json_path"]).read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise SizingFinalizationError(
            "model-head TRAIN==SERVE parity event is unreadable"
        ) from exc
    if (
        serve_parity.get("decision") != "PASS"
        or serve_parity.get("failures") != []
        or serve_parity.get("bundle_dir") != provenance["bundle_dir"]
        or serve_parity.get("dataset_dir") != provenance["dataset_dir"]
    ):
        raise SizingFinalizationError(
            "model-head TRAIN==SERVE parity does not bind the OOS candidate"
        )
    operating_point = serve_parity.get("operating_point")
    adapter = SmartEntryLiveInference.load_candidate_for_parity(
        bundle_dir=Path(str(provenance["bundle_dir"])),
        operating_point=operating_point,
        device=device,
    )
    bundle_authority = candidate_bundle_authority(
        bundle_dir=Path(str(provenance["bundle_dir"])),
        evaluation_bundle=proof["evaluation_bundle"],
        context="UNIFIED_REPLAY_CANDIDATE_BUNDLE",
    )
    if adapter._bundle_sha256 != bundle_authority["bundle_commit_sha256"]:
        raise SizingFinalizationError(
            "live adapter loaded different candidate bundle bytes"
        )

    pair_root = Path(prebuilt_generation_root).expanduser().resolve()
    pair_binding = read_prebuilt_pair_manifest(
        Path(prebuilt_pair_manifest_path).expanduser().resolve(),
        generation_root=pair_root,
    )
    verify_prebuilt_pair(pair_binding)
    prebuilt_pair = {
        "generation_root": str(pair_root),
        "identity": {
            "manifest_path": str(pair_binding.manifest_path),
            "manifest_sha256": pair_binding.manifest_sha256,
            "pair_generation_id": pair_binding.pair_generation_id,
            "canonical_v3": {
                "path": str(pair_binding.canonical_v3.parquet_path),
                "sha256": pair_binding.canonical_v3.parquet_sha256,
                "rows": pair_binding.canonical_v3.rows,
                "cols_total": pair_binding.canonical_v3.cols_total,
            },
            "base28": {
                "path": str(pair_binding.base28.parquet_path),
                "sha256": pair_binding.base28.parquet_sha256,
                "rows": pair_binding.base28.rows,
                "cols_total": pair_binding.base28.cols_total,
            },
            "refresh_enabled": False,
        },
    }
    rank_reference = load_train_rank_reference_v2(
        train_rank_reference_npz,
        expected_sha256=train_rank_reference_sha256,
    )
    train_rank_reference = train_rank_reference_identity_v2(rank_reference)
    state_contract = adapter._meta["model_native_state_contract"]
    if (
        str(Path(state_contract["rank_reference_npz"]).resolve())
        != train_rank_reference["path"]
        or state_contract["rank_reference_npz_sha256"]
        != train_rank_reference["sha256"]
        or state_contract["rank_reference_sidecar_sha256"]
        != train_rank_reference["sidecar_sha256"]
        or pd.Timestamp(state_contract["rank_fit_start_utc"])
        != pd.Timestamp(train_rank_reference["fit_start_utc"])
        or pd.Timestamp(state_contract["rank_fit_end_utc"])
        != pd.Timestamp(train_rank_reference["fit_end_utc"])
    ):
        raise SizingFinalizationError(
            "train-rank reference differs from candidate bundle state contract"
        )
    lifecycle = adapter._meta["unified_exit_training_evidence"]["lifecycle"]
    lifecycle_m1 = lifecycle["m1_authority"]
    if (
        lifecycle_m1["pair_manifest_path"]
        != prebuilt_pair["identity"]["manifest_path"]
        or lifecycle_m1["pair_manifest_sha256"]
        != prebuilt_pair["identity"]["manifest_sha256"]
        or lifecycle_m1["pair_generation_root"]
        != prebuilt_pair["generation_root"]
        or lifecycle_m1["pair_generation_id"]
        != prebuilt_pair["identity"]["pair_generation_id"]
        or lifecycle_m1["m1_source_path"] != str(tape.source_path)
        or lifecycle_m1["m1_source_sha256"] != tape.source_sha256
    ):
        raise SizingFinalizationError(
            "candidate lifecycle M1/pair authority differs from replay lineage"
        )
    m1_binding = adapter._meta.get("m1_feature_surface_binding")
    if not isinstance(m1_binding, dict):
        raise SizingFinalizationError(
            "candidate bundle lacks exact M1 feature-surface binding"
        )
    if (
        m1_binding.get("pair_generation_id")
        != pair_binding.pair_generation_id
    ):
        raise SizingFinalizationError(
            "candidate M1 feature surface differs from replay pair"
        )
    adapter.bind_admitted_m1_feature_surface(
        parquet_path=Path(str(m1_binding["parquet_path"])),
        manifest_path=Path(str(m1_binding["manifest_path"])),
        dataset_run_id=str(m1_binding["dataset_run_id"]),
        pair_generation_id=str(m1_binding["pair_generation_id"]),
        parquet_sha256=str(m1_binding["parquet_sha256"]),
        manifest_sha256=str(m1_binding["manifest_sha256"]),
        feature_field_order_sha256=str(
            m1_binding["feature_field_order_sha256"]
        ),
    )
    replay_pair_snapshot = SimpleNamespace(
        pair_generation_id=pair_binding.pair_generation_id
    )

    canonical_oos_rows = read_bound_parquet_exact(
        proof["source_bindings"]["oos_rows"],
        context="UNIFIED_REPLAY_CANONICAL_OOS_ROWS",
    )
    prediction_rows = pd.read_parquet(
        runtime_predictions_path,
        columns=[
            "time",
            "split",
            "model",
            "pred_direction",
            "position_size_logit",
            "runtime_head_evidence_json",
            "runtime_head_evidence_sha256",
        ],
    )
    prediction_rows["time"] = pd.to_datetime(
        prediction_rows["time"],
        utc=True,
        errors="coerce",
    )
    prediction_rows = prediction_rows.loc[
        (prediction_rows["split"].astype(str) == "test")
        & (prediction_rows["model"].astype(str) == "candidate")
    ].reset_index(drop=True)
    oos_times = pd.to_datetime(
        canonical_oos_rows["time"],
        utc=True,
        errors="coerce",
    )
    if (
        prediction_rows["time"].isna().any()
        or oos_times.isna().any()
        or len(prediction_rows) != len(canonical_oos_rows)
        or not prediction_rows["time"].reset_index(drop=True).equals(
            oos_times.reset_index(drop=True)
        )
    ):
        raise SizingFinalizationError(
            "runtime heads do not exactly cover canonical OOS TEST rows"
        )

    replay_rows = canonical_oos_rows.copy()
    for column in MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS:
        replay_rows[column] = None
    flat_trace_sha = hashlib.sha256(b"FLAT_NO_ORDER").hexdigest()
    trace_records: list[dict[str, Any]] = []
    for row_index, (oos_row, prediction_row) in enumerate(
        zip(
            canonical_oos_rows.to_dict("records"),
            prediction_rows.to_dict("records"),
            strict=True,
        )
    ):
        head = decode_model_native_runtime_head_evidence(
            prediction_row["runtime_head_evidence_json"],
            prediction_row["runtime_head_evidence_sha256"],
            context=f"UNIFIED_REPLAY_HEAD_{row_index}",
        )
        direction = adapter.decide_direction(head)
        expected_direction = int(oos_row["model_direction_index"])
        if (
            int(direction["model_direction_index"]) != expected_direction
            or int(prediction_row["pred_direction"]) != expected_direction
            or float(prediction_row["position_size_logit"])
            != float(oos_row["position_size_logit"])
        ):
            raise SizingFinalizationError(
                f"runtime Entry head differs from OOS row {row_index}"
            )
        decision_time = pd.Timestamp(oos_row["time"])
        if decision_time.tzinfo is None:
            decision_time = decision_time.tz_localize("UTC")
        else:
            decision_time = decision_time.tz_convert("UTC")
        entry_fill_time = decision_time + pd.Timedelta(minutes=5)
        replay_rows.at[row_index, "entry_fill_time"] = (
            entry_fill_time.isoformat()
        )
        replay_rows.at[row_index, "candidate_bundle_sha256"] = (
            bundle_authority["bundle_commit_sha256"]
        )
        if expected_direction == 2:
            replay_rows.at[row_index, "exit_replay_status"] = "FLAT_NO_ORDER"
            replay_rows.at[row_index, "exit_reason"] = "MODEL_FLAT"
            replay_rows.at[row_index, "exit_steps"] = 0
            replay_rows.at[row_index, "exit_trace_sha256"] = flat_trace_sha
            continue
        quote = tape.get_open_quote(entry_fill_time)
        if (
            float(quote["bid"]) != float(oos_row["entry_bid"])
            or float(quote["ask"]) != float(oos_row["entry_ask"])
        ):
            raise SizingFinalizationError(
                f"Entry fill quote differs from OOS row {row_index}"
            )
        side = "long" if expected_direction == 0 else "short"
        state = TradeState.open_unit_normalized_research(
            entry_ts=entry_fill_time,
            side=side,
            entry_bid=float(quote["bid"]),
            entry_ask=float(quote["ask"]),
            v10_snapshot=head,
            trade_id=str(oos_row["reference_row_id"]),
            normalization_contract="unit_normalized_direction_exit_research_v1",
        )
        first_bar_time = first_full_closed_m1_bar_ts(entry_fill_time)
        first_bar_position = int(tape.index.searchsorted(first_bar_time, side="left"))
        if (
            first_bar_position >= len(tape.index)
            or pd.Timestamp(tape.index[first_bar_position]) != first_bar_time
        ):
            raise SizingFinalizationError(
                f"source tape lacks first closed Entry M1 row {row_index}"
            )
        row_trace: list[dict[str, Any]] = []
        for step in range(1, UNIFIED_EXIT_MAX_PATH_BARS + 1):
            tape_position = first_bar_position + step - 1
            if tape_position >= len(tape.index):
                raise SizingFinalizationError(
                    f"source tape lacks Exit M1 tail for row {row_index}"
                )
            closed_bar_time = pd.Timestamp(tape.index[tape_position])
            closed_bar = tape.get_closed_m1_bar(closed_bar_time)
            staged = state.clone_for_exit_decision()
            staged.update_bar(**closed_bar)
            envelope = staged.build_closed_m1_path_evidence()
            exit_feature_surface = adapter.build_exit_feature_surface(
                decision_time=closed_bar_time,
                prebuilt_snapshot=replay_pair_snapshot,
            )
            exit_decision = adapter.decide_exit(
                entry_snapshot=head,
                exit_path_envelope=envelope,
                exit_feature_surface=exit_feature_surface,
                entry_bid=state.entry_bid,
                entry_ask=state.entry_ask,
                side=state.side,
            )
            staged.bind_unified_exit_decision(
                exit_decision,
                expected_bundle_sha256=bundle_authority[
                    "bundle_commit_sha256"
                ],
            )
            state.commit_complete_exit_bar(staged)
            model_fill_time = closed_bar_time + pd.Timedelta(minutes=1)
            row_trace.append(
                {
                    "reference_row_id": str(oos_row["reference_row_id"]),
                    "entry_fill_time": entry_fill_time,
                    "step": step,
                    "closed_bar_time": closed_bar_time,
                    "model_exit_fill_time": model_fill_time,
                    "bar_committed": True,
                    "action_id": int(exit_decision["exit_action_index"]),
                    "action": str(exit_decision["action"]),
                    "decision_source": str(
                        exit_decision["decision_source"]
                    ),
                    "state_bid": float(state.current_bid),
                    "state_ask": float(state.current_ask),
                    "state_pnl_bps": float(state.current_pnl_bps),
                    "exit_hold_logit": float(
                        exit_decision["exit_action_logits"][0]
                    ),
                    "exit_now_logit": float(
                        exit_decision["exit_action_logits"][1]
                    ),
                    "exit_hold_prob": float(
                        exit_decision["exit_action_probs"][0]
                    ),
                    "exit_now_prob": float(
                        exit_decision["exit_action_probs"][1]
                    ),
                    "candidate_bundle_sha256": str(
                        exit_decision["bundle_sha256"]
                    ),
                    "entry_snapshot_sha256": str(
                        exit_decision["entry_snapshot_sha256"]
                    ),
                    "exit_path_envelope_sha256": str(
                        exit_decision["exit_path_envelope_sha256"]
                    ),
                    "output_evidence_sha256": str(
                        exit_decision["output_evidence_sha256"]
                    ),
                    "closed_m1_source_path": str(closed_bar["source_path"]),
                    "closed_m1_source_sha256": str(
                        closed_bar["source_sha256"]
                    ),
                }
            )
            if exit_decision["action"] == "EXIT_NOW":
                break
        else:
            raise SizingFinalizationError(
                f"unified Exit never emitted EXIT_NOW within "
                f"{UNIFIED_EXIT_MAX_PATH_BARS} bars for OOS row {row_index}"
            )
        trace_frame = pd.DataFrame(
            row_trace,
            columns=sorted(MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS),
        )
        trace_records.extend(row_trace)
        replay_rows.at[row_index, "exit_replay_status"] = "EXIT_NOW"
        replay_rows.at[row_index, "model_exit_decision_bar_time"] = (
            closed_bar_time.isoformat()
        )
        replay_rows.at[row_index, "model_exit_fill_time"] = (
            model_fill_time.isoformat()
        )
        replay_rows.at[row_index, "model_exit_fill_bid"] = float(
            state.current_bid
        )
        replay_rows.at[row_index, "model_exit_fill_ask"] = float(
            state.current_ask
        )
        replay_rows.at[row_index, "exit_reason"] = "UNIFIED_MODEL_ARGMAX"
        replay_rows.at[row_index, "exit_steps"] = len(row_trace)
        replay_rows.at[row_index, "exit_trace_sha256"] = (
            joint_exit_trace_sha256(
                trace_frame,
                context=f"UNIFIED_REPLAY_TRACE_{row_index}",
            )
        )

    exit_trace_rows = pd.DataFrame(
        trace_records,
        columns=sorted(MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS),
    )
    output_dir = _stage_dir(authority_root, "joint_replay")
    output_time = _monotonic_stage_created_utc(
        output_dir,
        JOINT_EXIT_PROOF_PREFIX,
        proof["created_utc"],
    )
    stamp = output_time.strftime("%Y%m%dT%H%M%S%fZ")
    replay_rows_path = (
        output_dir / f"unified_candidate_replay_rows_{stamp}.parquet"
    )
    exit_trace_rows_path = (
        output_dir / f"unified_candidate_exit_trace_rows_{stamp}.parquet"
    )
    atomic_write_parquet_immutable(replay_rows, replay_rows_path)
    atomic_write_parquet_immutable(exit_trace_rows, exit_trace_rows_path)
    replay_binding = _source_binding(replay_rows_path)
    trace_binding = _source_binding(exit_trace_rows_path)
    producer_sources = build_canonical_unified_replay_source_inventory(
        Path(__file__).resolve().parents[2]
    )
    producer_evidence = {
        "schema_version": CANONICAL_UNIFIED_REPLAY_PRODUCER_SCHEMA_VERSION,
        "producer_contract": CANONICAL_UNIFIED_REPLAY_PRODUCER_CONTRACT,
        "decision": "PASS",
        "failures": [],
        "source_tape": {
            "path": str(tape.source_path),
            "sha256": tape.source_sha256,
        },
        "prebuilt_pair": prebuilt_pair,
        "train_rank_reference": train_rank_reference,
        "runtime_predictions": _source_binding(runtime_predictions_path),
        "prediction_report_artifact": report_binding,
        "prediction_provenance": provenance,
        "canonical_oos_rows": proof["source_bindings"]["oos_rows"],
        "candidate_bundle_authority": bundle_authority,
        "replay_rows": replay_binding,
        "exit_trace_rows": trace_binding,
        "producer_source_files": producer_sources,
        "producer_source_inventory_sha256": _canonical_json_sha256(
            producer_sources
        ),
        "rows": int(len(replay_rows)),
        "trade_rows": int(
            np.count_nonzero(
                pd.to_numeric(
                    replay_rows["model_direction_index"],
                    errors="coerce",
                ).to_numpy(dtype=np.float64)
                != 2
            )
        ),
        "trace_rows": int(len(exit_trace_rows)),
        "first_utc": oos_times.iloc[0].isoformat(),
        "last_utc": oos_times.iloc[-1].isoformat(),
    }
    return _finalize_joint_exit_sizing_proof(
        calibration_path=calibration_path,
        proof_path=proof_path,
        replay_rows_path=replay_rows_path,
        exit_trace_rows_path=exit_trace_rows_path,
        authority_root=authority_root,
        canonical_producer_evidence=producer_evidence,
    )


@_terminal_event_attempt("adoption", ADOPTION_PREFIX)
def adopt_learned_sizing(
    *,
    bundle_dir: Path,
    calibration_path: Path,
    proof_path: Path,
    joint_exit_proof_path: Path,
    authority_root: Path,
    entry_run_id: str,
) -> tuple[Path, dict[str, Any]]:
    """Adopt learned sizing after exact OOS and joint unified-Exit proofs.

    This does not authorize paper/live capital.  The launch artifact guard still
    requires a separate, newer post-adoption broker runtime-parity event.
    """

    from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id

    entry_run_id = require_entry_run_id(entry_run_id)
    if not joint_exit_proof_path.expanduser().resolve().is_file():
        raise SizingFinalizationError(
            "joint unified-Exit sizing proof is required before adoption"
        )
    bundle_dir = bundle_dir.expanduser().resolve()
    metadata_path = bundle_dir / "bundle_metadata.json"
    lock_path = bundle_dir / "MASTER_TRANSFORMER_LOCK.json"
    state_path = bundle_dir / "model_state_dict.pt"
    for path in (metadata_path, lock_path, state_path):
        if not path.is_file() or path.is_symlink():
            raise SizingFinalizationError(f"adopted bundle artifact missing: {path}")
    calibration, calibration_binding = load_bound_sizing_calibration(
        _binding(calibration_path),
        context="SIZING_ADOPTION_CALIBRATION",
        verify_lineage_files=True,
    )
    proof, proof_binding = load_bound_sizing_oos_proof(
        _binding(proof_path),
        calibration=calibration,
        calibration_artifact_sha256=calibration_binding["sha256"],
        context="SIZING_ADOPTION_OOS_PROOF",
        verify_source_files=True,
    )
    joint_proof, joint_binding = load_bound_joint_exit_sizing_proof(
        _binding(joint_exit_proof_path),
        context="SIZING_ADOPTION_JOINT_EXIT_PROOF",
        verify_source_files=True,
    )
    if (
        joint_proof["calibration_artifact"] != calibration_binding
        or joint_proof["oos_proof_artifact"] != proof_binding
    ):
        raise SizingFinalizationError(
            "joint unified-Exit proof differs from the adopted sizing chain"
        )
    _require_stage_path(
        Path(calibration_binding["json_path"]), authority_root, "calibration"
    )
    _require_stage_path(Path(proof_binding["json_path"]), authority_root, "proof")
    _require_stage_path(
        Path(joint_binding["json_path"]), authority_root, "joint_replay"
    )
    evaluation_bundle = proof["evaluation_bundle"]
    expected_bundle = {
        "bundle_dir": str(bundle_dir),
        "bundle_metadata_path": str(metadata_path),
        "bundle_metadata_sha256": _sha(metadata_path),
        "master_transformer_lock_path": str(lock_path),
        "master_transformer_lock_sha256": _sha(lock_path),
        "model_state_dict_path": str(state_path),
        "model_state_dict_sha256": _sha(state_path),
    }
    if evaluation_bundle != expected_bundle:
        raise SizingFinalizationError(
            "OOS/joint proof bundle differs from proposed adoption bundle"
        )
    expected_declaration = model_native_sizing_bundle_calibration_metadata(
        calibration_artifact=calibration_binding
    )
    for label, path in (("metadata", metadata_path), ("lock", lock_path)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if require_model_native_sizing_bundle_calibration(
            payload.get("model_native_sizing_calibration"),
            context=f"SIZING_ADOPTION_{label.upper()}",
        ) != expected_declaration:
            raise SizingFinalizationError(
                f"bundle {label} sizing calibration differs from adoption"
            )
    output_dir = _stage_dir(authority_root, "adoption")
    created = _monotonic_stage_created_utc(
        output_dir,
        ADOPTION_PREFIX,
        joint_proof["created_utc"],
    )
    payload = {
        "schema_version": MODEL_NATIVE_SIZING_ADOPTION_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": "PASS",
        "failures": [],
        "adoption_mode": MODEL_NATIVE_SIZING_MODE_LEARNED,
        "authority_root": str(authority_root.expanduser().resolve()),
        **expected_bundle,
        "calibration_artifact": calibration_binding,
        "oos_proof_artifact": proof_binding,
        "joint_exit_sizing_proof_artifact": joint_binding,
        "risk_policy": sizing_risk_policy_metadata(),
        "runtime_constraint_authority": (
            "exact_broker_account_instrument_exposure_facts_with_transaction_ids"
        ),
        "direction_authority": "none",
        "fixed_1x_fallback_allowed": False,
        "entry_run_id": entry_run_id,
    }
    event_path, event = write_immutable_json_event(
        output_dir,
        ADOPTION_PREFIX,
        payload,
    )
    require_model_native_sizing_adoption_artifact(
        event,
        context="SIZING_ADOPTION_SELF_VALIDATION",
    )
    return event_path, event


@_terminal_event_attempt("runtime_parity", RUNTIME_PARITY_PREFIX)
def finalize_runtime_sizing_parity(
    *,
    adoption_path: Path,
    observations_path: Path,
    authority_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Publish fresh broker-live shadow parity after learned sizing adoption."""

    adoption_path = adoption_path.expanduser().resolve()
    adoption_binding = _binding(adoption_path)
    try:
        adoption = json.loads(adoption_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SizingFinalizationError("sizing adoption is unreadable") from exc
    require_model_native_sizing_adoption_artifact(
        adoption,
        context="SIZING_RUNTIME_PARITY_ADOPTION",
    )
    if Path(adoption["json_path"]).resolve() != adoption_path:
        raise SizingFinalizationError("sizing adoption self-reference mismatch")
    _require_stage_path(adoption_path, authority_root, "adoption")
    calibration, calibration_binding = load_bound_sizing_calibration(
        adoption["calibration_artifact"],
        context="SIZING_RUNTIME_PARITY_CALIBRATION",
        verify_lineage_files=True,
    )
    if calibration_binding != adoption["calibration_artifact"]:
        raise SizingFinalizationError("runtime parity calibration binding mismatch")
    observations_path, observations_binding = _canonical_immutable_parquet_binding(
        observations_path,
        context="runtime sizing observations",
    )
    observations = read_bound_parquet_exact(
        observations_binding,
        context="SIZING_RUNTIME_PARITY_OBSERVATIONS_EXACT",
    )
    if "time" not in observations:
        raise SizingFinalizationError("runtime sizing observations lack time")
    observation_times = pd.to_datetime(
        observations["time"], utc=True, errors="coerce"
    )
    if len(observations) == 0 or observation_times.isna().any():
        raise SizingFinalizationError("runtime sizing observations lack exact UTC rows")
    output_dir = _stage_dir(authority_root, "runtime_parity")
    created = _monotonic_stage_created_utc(
        output_dir,
        RUNTIME_PARITY_PREFIX,
        adoption["created_utc"],
        observation_times.max(),
    )
    coverage = recompute_runtime_sizing_parity_coverage(
        observations,
        calibration=calibration,
        adoption=adoption,
        adoption_sha256=adoption_binding["sha256"],
        event_created_utc=created,
        context="SIZING_RUNTIME_PARITY_RECOMPUTE",
    )
    bundle_identity = {
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
    payload = {
        "schema_version": MODEL_NATIVE_SIZING_RUNTIME_PARITY_SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": "PASS",
        "failures": [],
        "parity_contract": MODEL_NATIVE_SIZING_RUNTIME_PARITY_CONTRACT,
        "adoption_artifact": adoption_binding,
        "bundle_identity": bundle_identity,
        "observations": observations_binding,
        "coverage": coverage,
    }
    event_path, event = write_immutable_json_event(
        output_dir,
        RUNTIME_PARITY_PREFIX,
        payload,
    )
    load_bound_runtime_sizing_parity(
        _binding(event_path),
        adoption=adoption,
        calibration=calibration,
        adoption_artifact=adoption_binding,
        context="SIZING_RUNTIME_PARITY_SELF_VALIDATION",
        verify_source_files=True,
        now_utc=created,
    )
    return event_path, event


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture-instrument")
    capture.add_argument("--authority-root", type=Path, required=True)
    fit = sub.add_parser("fit-calibration")
    fit.add_argument("--predictions", type=Path, required=True)
    fit.add_argument("--prediction-report", type=Path, required=True)
    fit.add_argument("--bundle-dir", type=Path, required=True)
    fit.add_argument("--dataset-dir", type=Path, required=True)
    fit.add_argument("--dataset-manifest", type=Path, required=True)
    fit.add_argument("--instrument-evidence", type=Path, required=True)
    fit.add_argument("--authority-root", type=Path, required=True)
    bind = sub.add_parser("bind-bundle")
    bind.add_argument("--source-bundle-dir", type=Path, required=True)
    bind.add_argument("--out-bundle-dir", type=Path, required=True)
    bind.add_argument("--calibration", type=Path, required=True)
    oos = sub.add_parser("materialize-test-oos")
    oos.add_argument("--calibration", type=Path, required=True)
    oos.add_argument("--test-predictions", type=Path, required=True)
    oos.add_argument("--test-prediction-report", type=Path, required=True)
    oos.add_argument("--bundle-dir", type=Path, required=True)
    oos.add_argument("--dataset-dir", type=Path, required=True)
    oos.add_argument("--source-tape", type=Path, required=True)
    oos.add_argument("--model-head-serve-parity", type=Path, required=True)
    oos.add_argument("--authority-root", type=Path, required=True)
    proof = sub.add_parser("finalize-test-proof")
    proof.add_argument("--calibration", type=Path, required=True)
    proof.add_argument("--oos-source", type=Path, required=True)
    proof.add_argument("--authority-root", type=Path, required=True)
    joint = sub.add_parser("finalize-joint-exit-proof")
    joint.add_argument("--calibration", type=Path, required=True)
    joint.add_argument("--proof", type=Path, required=True)
    joint.add_argument("--replay-rows", type=Path, required=True)
    joint.add_argument("--exit-trace-rows", type=Path, required=True)
    joint.add_argument("--authority-root", type=Path, required=True)
    unified = sub.add_parser("produce-unified-joint-exit-proof")
    unified.add_argument("--calibration", type=Path, required=True)
    unified.add_argument("--proof", type=Path, required=True)
    unified.add_argument("--source-tape", type=Path, required=True)
    unified.add_argument("--prebuilt-pair-manifest", type=Path, required=True)
    unified.add_argument("--prebuilt-generation-root", type=Path, required=True)
    unified.add_argument("--train-rank-reference-npz", type=Path, required=True)
    unified.add_argument("--train-rank-reference-sha256", required=True)
    unified.add_argument("--authority-root", type=Path, required=True)
    unified.add_argument("--device", default="cpu")
    adopt = sub.add_parser("adopt")
    adopt.add_argument("--bundle-dir", type=Path, required=True)
    adopt.add_argument("--calibration", type=Path, required=True)
    adopt.add_argument("--proof", type=Path, required=True)
    adopt.add_argument("--joint-exit-proof", type=Path, required=True)
    adopt.add_argument("--authority-root", type=Path, required=True)
    adopt.add_argument("--run-id", required=True)
    runtime = sub.add_parser("finalize-runtime-parity")
    runtime.add_argument("--adoption", type=Path, required=True)
    runtime.add_argument("--observations", type=Path, required=True)
    runtime.add_argument("--authority-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "capture-instrument":
        path, _ = capture_oanda_instrument_evidence(
            authority_root=args.authority_root
        )
        result: Any = _binding(path)
    elif args.command == "fit-calibration":
        path, _ = fit_train_val_sizing_calibration(
            predictions_path=args.predictions,
            prediction_report_path=args.prediction_report,
            bundle_dir=args.bundle_dir,
            dataset_dir=args.dataset_dir,
            dataset_manifest_path=args.dataset_manifest,
            instrument_evidence_path=args.instrument_evidence,
            authority_root=args.authority_root,
        )
        result = _binding(path)
    elif args.command == "bind-bundle":
        result = bind_bundle_sizing_calibration(
            source_bundle_dir=args.source_bundle_dir,
            output_bundle_dir=args.out_bundle_dir,
            calibration_path=args.calibration,
        )
    elif args.command == "materialize-test-oos":
        path, _ = materialize_test_sizing_oos_source(
            calibration_path=args.calibration,
            test_predictions_path=args.test_predictions,
            test_prediction_report_path=args.test_prediction_report,
            bundle_dir=args.bundle_dir,
            dataset_dir=args.dataset_dir,
            source_tape_path=args.source_tape,
            model_head_serve_parity_path=args.model_head_serve_parity,
            authority_root=args.authority_root,
        )
        result = _binding(path)
    elif args.command == "finalize-test-proof":
        path, _ = finalize_test_sizing_proof(
            calibration_path=args.calibration,
            oos_source_path=args.oos_source,
            authority_root=args.authority_root,
        )
        result = _binding(path)
    elif args.command == "finalize-joint-exit-proof":
        path, _ = finalize_joint_exit_sizing_proof(
            calibration_path=args.calibration,
            proof_path=args.proof,
            replay_rows_path=args.replay_rows,
            exit_trace_rows_path=args.exit_trace_rows,
            authority_root=args.authority_root,
        )
        result = _binding(path)
    elif args.command == "produce-unified-joint-exit-proof":
        path, _ = produce_canonical_unified_joint_sizing_proof(
            calibration_path=args.calibration,
            proof_path=args.proof,
            source_tape_path=args.source_tape,
            prebuilt_pair_manifest_path=args.prebuilt_pair_manifest,
            prebuilt_generation_root=args.prebuilt_generation_root,
            train_rank_reference_npz=args.train_rank_reference_npz,
            train_rank_reference_sha256=(
                args.train_rank_reference_sha256
            ),
            authority_root=args.authority_root,
            device=args.device,
        )
        result = _binding(path)
    elif args.command == "adopt":
        path, _ = adopt_learned_sizing(
            bundle_dir=args.bundle_dir,
            calibration_path=args.calibration,
            proof_path=args.proof,
            joint_exit_proof_path=args.joint_exit_proof,
            authority_root=args.authority_root,
            entry_run_id=args.run_id,
        )
        result = _binding(path)
    else:
        path, _ = finalize_runtime_sizing_parity(
            adoption_path=args.adoption,
            observations_path=args.observations,
            authority_root=args.authority_root,
        )
        result = _binding(path)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
