"""Canonical producer/finalizer for the learned Entry sizing evidence chain.

No JSON in this chain is intended for hand editing.  The public stages are:

1. capture broker instrument evidence;
2. fit TRAIN/VAL calibration and publish its immutable event;
3. bind that calibration into a fresh bundle clone;
4. materialize canonical TEST/OOS sizing rows and publish a row-recomputed
   diagnostic proof;
5. publish a terminal capital-adoption FAIL until joint active Exit execution
   proof and post-adoption broker-runtime sizing parity have real producers.

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
from typing import Any, NoReturn

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    MODEL_NATIVE_SIZING_CAPITAL_ADOPTION_BLOCKERS,
    model_native_sizing_bundle_calibration_metadata,
    require_model_native_sizing_bundle_calibration,
)
from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    MODEL_NATIVE_SIZING_CALIBRATION_SCHEMA_VERSION,
    MODEL_NATIVE_SIZING_FIT_SPLITS,
    MODEL_NATIVE_SIZING_FIT_SCOPE,
    MODEL_NATIVE_SIZING_HEAD_VARIATION_EPSILON,
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
    require_sizing_evaluation_bundle,
    require_sizing_instrument_evidence_artifact,
    require_sizing_prediction_provenance,
    sha256_file,
    sizing_risk_policy_metadata,
    sizing_oos_reference_account_policy_metadata,
    sizing_fit_contract_metadata,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_parquet_immutable,
    resolve_and_validate_prediction_evidence,
)
from gx1.contracts.immutable_event_authority_v1 import (
    require_newest_immutable_event,
    write_immutable_json_event,
)


INSTRUMENT_EVIDENCE_SCHEMA_VERSION = (
    MODEL_NATIVE_SIZING_INSTRUMENT_EVIDENCE_SCHEMA_VERSION
)
INSTRUMENT_EVIDENCE_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_INSTRUMENT_EVIDENCE"
CALIBRATION_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_CALIBRATION"
OOS_SOURCE_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_OOS_SOURCE"
PROOF_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_OOS_PROOF"
ADOPTION_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_ADOPTION"
MIN_FIT_ROWS_PER_SPLIT = MODEL_NATIVE_SIZING_MIN_FIT_ROWS_PER_SPLIT
_AUTHORITY_STAGE_DIRS = {
    "instrument": "instrument",
    "calibration": "calibration",
    "oos": "oos",
    "proof": "proof",
    "adoption": "adoption",
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
    dataset_dir: Path, expected_splits: tuple[str, ...]
) -> dict[str, dict[str, str]]:
    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SizingFinalizationError(f"dataset_dir missing: {dataset_dir}")
    rows: dict[str, dict[str, str]] = {}
    for split in expected_splits:
        manifests = sorted(dataset_dir.glob(f"*_{split}.manifest.json"))
        parquets = sorted(dataset_dir.glob(f"*_{split}.parquet"))
        if len(manifests) != 1 or len(parquets) != 1:
            raise SizingFinalizationError(
                f"dataset {split} requires exactly one manifest/parquet: "
                f"manifests={manifests} parquets={parquets}"
            )
        rows[split] = {
            "manifest_path": str(manifests[0].resolve()),
            "manifest_sha256": _sha(manifests[0]),
            "parquet_path": str(parquets[0].resolve()),
            "parquet_sha256": _sha(parquets[0]),
        }
    return rows


def _prediction_provenance(
    *,
    predictions_path: Path,
    prediction_report_path: Path,
    bundle_dir: Path,
    dataset_dir: Path,
    expected_splits: tuple[str, ...],
    context: str,
) -> tuple[Path, dict[str, Any]]:
    predictions_path = predictions_path.expanduser().resolve()
    prediction_report_path = prediction_report_path.expanduser().resolve()
    bundle_dir = bundle_dir.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()
    try:
        authoritative, _report, evidence = resolve_and_validate_prediction_evidence(
            predictions_path,
            prediction_report_path=prediction_report_path,
            bundle_dir=bundle_dir,
            dataset_dir=dataset_dir,
            expected_model="candidate",
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
            dataset_dir, expected_splits
        ),
    }
    require_sizing_prediction_provenance(
        provenance,
        predictions_binding=_source_binding(authoritative),
        expected_splits=expected_splits,
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
    required_inventory = {
        "bundle_metadata.json",
        "MASTER_TRANSFORMER_LOCK.json",
        "model_state_dict.pt",
    }
    source_entries = list(source_bundle_dir.iterdir())
    observed_inventory = {path.name for path in source_entries}
    if observed_inventory != required_inventory:
        raise SizingFinalizationError(
            "source bundle inventory must be exact code-owned files; "
            f"expected={sorted(required_inventory)} observed={sorted(observed_inventory)}"
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
        for name in sorted(required_inventory):
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
        os.rename(staging, output_bundle_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    paths = (
        output_bundle_dir / "bundle_metadata.json",
        output_bundle_dir / "MASTER_TRANSFORMER_LOCK.json",
    )
    if {path.name for path in output_bundle_dir.iterdir()} != required_inventory:
        raise SizingFinalizationError("finalized bundle inventory parity failed")
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
        raise SizingFinalizationError(
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


@_terminal_event_attempt("adoption", ADOPTION_PREFIX)
def adopt_learned_sizing(
    *,
    bundle_dir: Path,
    calibration_path: Path,
    proof_path: Path,
    authority_root: Path,
    accepted_via_vedtak: str,
) -> NoReturn:
    """Publish a terminal FAIL until joint Exit proof and runtime parity exist."""

    del bundle_dir, calibration_path, proof_path, accepted_via_vedtak
    raise SizingFinalizationError(
        "capital sizing adoption is structurally BLOCKED: "
        + " | ".join(MODEL_NATIVE_SIZING_CAPITAL_ADOPTION_BLOCKERS)
    )


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
    adopt = sub.add_parser("adopt")
    adopt.add_argument("--bundle-dir", type=Path, required=True)
    adopt.add_argument("--calibration", type=Path, required=True)
    adopt.add_argument("--proof", type=Path, required=True)
    adopt.add_argument("--authority-root", type=Path, required=True)
    adopt.add_argument("--vedtak", required=True)
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
    else:
        adopt_learned_sizing(
            bundle_dir=args.bundle_dir,
            calibration_path=args.calibration,
            proof_path=args.proof,
            authority_root=args.authority_root,
            accepted_via_vedtak=args.vedtak,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
