"""Immutable, fail-closed lineage for Entry candidate prediction evidence.

Authorizing consumers must name both the timestamped parquet and its matching
timestamped JSON event explicitly. Fixed-name mirrors are not produced or
resolved by this contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.immutable_event_authority_v1 import (
    require_newest_immutable_event,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_TIMING_OUTPUT_DIM,
    MODEL_NATIVE_TIMING_TARGET_COLUMNS,
)
from gx1.contracts.entry_model_native_offline_rl_v1 import (
    ACTION_VALUE_DIM,
    ACTION_VALUE_TARGET_COLUMNS,
    EXPECTILE_VALUE_DIM,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    require_model_direction_decision_contract,
)


PREDICTION_EVIDENCE_SCHEMA_VERSION = (
    "entry_candidate_model_direction_prediction_evidence_v2"
)
AUTHORITATIVE_PREDICTIONS_PREFIX = "selective_edge_predictions_"
REPORT_PREFIX = "ENTRY_CANDIDATE_SELECTIVE_EDGE_"
_EVENT_STAMP_RE = re.compile(r"\d{8}T\d{12}Z")

REQUIRED_MODEL_DIRECTION_COLUMNS = (
    "split",
    "model",
    "time",
    "y_direction",
    "pred_direction",
    "p_long",
    "p_short",
    "p_flat",
    "selection_score_mode",
    "public_trade_probability",
    "public_flat_probability",
    "public_trade_flat_margin",
    "public_trade_flat_hard_decision",
    "direction_logits",
    "public_trade_flat_decision_logits",
    "timing_pred",
    *MODEL_NATIVE_TIMING_TARGET_COLUMNS,
    "action_value",
    "expectile_value",
    "action_advantage",
    *ACTION_VALUE_TARGET_COLUMNS,
)


def sha256_file(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def parquet_schema_descriptor(path: Path) -> list[dict[str, Any]]:
    schema = pq.ParquetFile(path).schema_arrow
    return [
        {
            "name": str(field.name),
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def parquet_schema_sha256(descriptor: list[dict[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json(descriptor).encode("utf-8")).hexdigest()


def _unique_strings(path: Path, column: str) -> list[str]:
    values = pq.read_table(path, columns=[column]).column(column).to_pylist()
    return sorted({str(value) for value in values if value is not None})


def validate_model_direction_parquet_semantics(path: Path) -> None:
    """Prove that persisted classes/public pair equal the final model logits."""

    frame = pd.read_parquet(path, columns=list(REQUIRED_MODEL_DIRECTION_COLUMNS))
    if frame.empty:
        raise RuntimeError("prediction evidence is empty")

    def matrix(name: str, width: int) -> np.ndarray:
        try:
            out = np.stack(
                [np.asarray(value, dtype=np.float64) for value in frame[name].to_numpy()]
            )
        except Exception as exc:
            raise RuntimeError(f"prediction evidence {name} is not a dense vector") from exc
        if out.shape != (len(frame), width) or not np.isfinite(out).all():
            raise RuntimeError(
                f"prediction evidence {name} must be finite shape ({len(frame)},{width})"
            )
        return out

    def numeric(name: str) -> np.ndarray:
        out = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(out).all():
            raise RuntimeError(f"prediction evidence {name} contains non-finite values")
        return out

    direction_logits = matrix("direction_logits", 3)
    public_logits = matrix("public_trade_flat_decision_logits", 2)
    timing_pred = matrix("timing_pred", MODEL_NATIVE_TIMING_OUTPUT_DIM)
    action_value = matrix("action_value", ACTION_VALUE_DIM)
    expectile_value = matrix("expectile_value", EXPECTILE_VALUE_DIM)
    action_advantage = matrix("action_advantage", ACTION_VALUE_DIM)
    expected_advantage = (
        action_value.reshape(len(frame), 3, EXPECTILE_VALUE_DIM)
        - expectile_value[:, None, :]
    ).reshape(len(frame), ACTION_VALUE_DIM)
    if not np.allclose(
        action_advantage,
        expected_advantage,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("prediction evidence action_advantage does not equal Q-V")
    if np.any(timing_pred < 0.0) or np.any(timing_pred > 1.0):
        raise RuntimeError("prediction evidence timing_pred is outside [0,1]")
    for target_column in MODEL_NATIVE_TIMING_TARGET_COLUMNS:
        target = numeric(target_column)
        if np.any(target < 0.0) or np.any(target > 1.0):
            raise RuntimeError(
                f"prediction evidence {target_column} is outside [0,1]"
            )
    for target_column in ACTION_VALUE_TARGET_COLUMNS:
        numeric(target_column)
    expected_public_logits = np.column_stack(
        [np.maximum(direction_logits[:, 0], direction_logits[:, 1]), direction_logits[:, 2]]
    )
    if not np.allclose(public_logits, expected_public_logits, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            "prediction evidence public trade/FLAT logits mismatch canonical formula"
        )

    def softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    direction_probs = softmax(direction_logits)
    persisted_direction_probs = np.column_stack(
        [numeric("p_long"), numeric("p_short"), numeric("p_flat")]
    )
    if not np.allclose(
        direction_probs, persisted_direction_probs, rtol=1e-5, atol=1e-6
    ):
        raise RuntimeError("prediction evidence direction probabilities mismatch logits")
    pred_direction = numeric("pred_direction")
    if not np.array_equal(pred_direction, np.rint(pred_direction)) or not np.array_equal(
        pred_direction.astype(np.int64), np.argmax(direction_logits, axis=1)
    ):
        raise RuntimeError("prediction evidence pred_direction mismatch final logits argmax")

    public_probs = softmax(public_logits)
    persisted_public_probs = np.column_stack(
        [numeric("public_trade_probability"), numeric("public_flat_probability")]
    )
    if not np.allclose(public_probs, persisted_public_probs, rtol=1e-5, atol=1e-6):
        raise RuntimeError("prediction evidence public probabilities mismatch public logits")
    public_margin = numeric("public_trade_flat_margin")
    if not np.allclose(
        public_margin, public_logits[:, 0] - public_logits[:, 1], rtol=1e-5, atol=1e-6
    ):
        raise RuntimeError("prediction evidence public margin mismatch public logits")
    public_hard = numeric("public_trade_flat_hard_decision")
    if not np.array_equal(public_hard, np.rint(public_hard)) or not np.array_equal(
        public_hard.astype(np.int64), np.argmax(public_logits, axis=1)
    ):
        raise RuntimeError("prediction evidence public hard decision mismatch public logits")
    expected_public_hard = np.where(pred_direction.astype(np.int64) == 2, 1, 0)
    if not np.array_equal(public_hard.astype(np.int64), expected_public_hard):
        raise RuntimeError("prediction evidence public hard decision mismatches LONG/SHORT/FLAT")
    modes = sorted({str(value) for value in frame["selection_score_mode"].to_numpy()})
    if modes != [MODEL_DIRECTION_SELECTION_MODE]:
        raise RuntimeError(
            "prediction evidence direction mode mismatch: "
            f"observed={modes} required={[MODEL_DIRECTION_SELECTION_MODE]}"
        )


def atomic_write_parquet_immutable(frame: pd.DataFrame, path: Path) -> None:
    """Publish a new parquet atomically without ever replacing an old event."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"immutable prediction evidence already exists: {path}")
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        frame.to_parquet(tmp, index=False)
        # link() is an atomic no-replace publication on the same filesystem.
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"immutable report already exists: {path}")
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build_prediction_evidence_declaration(
    *,
    predictions_path: Path,
    bundle_dir: Path,
    bundle_metadata: Mapping[str, Any],
    requested_splits: list[str],
) -> dict[str, Any]:
    """Inspect the published parquet and return its exact lineage declaration."""

    path = predictions_path.expanduser().resolve()
    bundle_dir = bundle_dir.expanduser().resolve()
    metadata_path = bundle_dir / "bundle_metadata.json"
    if not path.is_file():
        raise RuntimeError(f"authoritative prediction evidence missing: {path}")
    if not metadata_path.is_file():
        raise RuntimeError(f"bundle metadata missing for prediction evidence: {metadata_path}")
    direction_contract = require_model_direction_decision_contract(
        bundle_metadata,
        context=f"prediction evidence bundle {bundle_dir}",
    )
    state_sha = str(bundle_metadata.get("state_dict_sha256") or "").strip().lower()
    if len(state_sha) != 64 or any(ch not in "0123456789abcdef" for ch in state_sha):
        raise RuntimeError("bundle state_dict_sha256 is missing or not an exact SHA-256")

    parquet = pq.ParquetFile(path)
    columns = list(parquet.schema_arrow.names)
    missing = sorted(set(REQUIRED_MODEL_DIRECTION_COLUMNS) - set(columns))
    if missing:
        raise RuntimeError(f"prediction evidence missing required columns: {missing}")
    if "selection_score_threshold" in columns:
        raise RuntimeError("prediction evidence contains forbidden selection_score_threshold")
    validate_model_direction_parquet_semantics(path)
    splits = _unique_strings(path, "split")
    expected_splits = sorted({str(value) for value in requested_splits})
    if splits != expected_splits:
        raise RuntimeError(
            f"prediction evidence split mismatch: observed={splits} expected={expected_splits}"
        )
    modes = _unique_strings(path, "selection_score_mode")
    if modes != [MODEL_DIRECTION_SELECTION_MODE]:
        raise RuntimeError(
            "prediction evidence direction mode mismatch: "
            f"observed={modes} required={[MODEL_DIRECTION_SELECTION_MODE]}"
        )
    descriptor = parquet_schema_descriptor(path)
    return {
        "schema_version": PREDICTION_EVIDENCE_SCHEMA_VERSION,
        "authoritative": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(parquet.metadata.num_rows),
        "splits": splits,
        "models": _unique_strings(path, "model"),
        "columns": columns,
        "required_columns": list(REQUIRED_MODEL_DIRECTION_COLUMNS),
        "parquet_schema": descriptor,
        "parquet_schema_sha256": parquet_schema_sha256(descriptor),
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "direction_decision_contract": direction_contract,
        "bundle_metadata_path": str(metadata_path.resolve()),
        "bundle_metadata_sha256": sha256_file(metadata_path),
        "model_state_dict_sha256": state_sha,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"could not read prediction evidence report {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RuntimeError(f"prediction evidence report must be a JSON object: {path}")
    return raw


def _timestamped_report_for_predictions(path: Path) -> Path:
    name = path.name
    if not name.startswith(AUTHORITATIVE_PREDICTIONS_PREFIX) or not name.endswith(".parquet"):
        raise RuntimeError(f"not a timestamped authoritative predictions path: {path}")
    stamp = name[len(AUTHORITATIVE_PREDICTIONS_PREFIX) : -len(".parquet")]
    if _EVENT_STAMP_RE.fullmatch(stamp) is None:
        raise RuntimeError(
            "prediction evidence path lacks an exact microsecond UTC event stamp: "
            f"{path}"
        )
    return path.parent / f"{REPORT_PREFIX}{stamp}.json"


def resolve_and_validate_prediction_evidence(
    requested_path: Path,
    *,
    prediction_report_path: Path,
    bundle_dir: Path | None,
    dataset_dir: Path,
    expected_split: str | None = None,
    expected_model: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Re-hash one explicitly named, newest immutable prediction event.

    Returns ``(authoritative_path, timestamped_report, evidence_declaration)``.
    Every mismatch raises before a consumer can read predictions.  This is an
    authorizing contract: mutable mirrors and older timestamped PASS events are
    deliberately rejected even when their content still hashes correctly.
    """

    requested = requested_path.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()
    authoritative = requested
    if authoritative.is_symlink():
        raise RuntimeError(
            f"authoritative prediction evidence cannot be a symlink: {authoritative}"
        )
    if prediction_report_path is None:
        raise RuntimeError("explicit timestamped prediction_report_path is required")
    report_path = prediction_report_path.expanduser().resolve()
    expected_report_path = _timestamped_report_for_predictions(authoritative).resolve()
    if report_path != expected_report_path:
        raise RuntimeError(
            "explicit prediction report does not match the timestamped prediction "
            f"event: predictions={authoritative} report={report_path} "
            f"expected={expected_report_path}"
        )
    require_newest_immutable_event(report_path, REPORT_PREFIX.rstrip("_"))
    report = _read_json(report_path)

    if Path(str(report.get("json_path") or "")).expanduser().resolve() != report_path:
        raise RuntimeError("timestamped prediction report json_path mismatch")
    if report.get("schema_version") != "entry_candidate_selective_edge_v1":
        raise RuntimeError("timestamped prediction report schema_version mismatch")
    if str(report.get("decision") or "") != "PASS" or report.get("failures"):
        raise RuntimeError("timestamped prediction report is not a zero-failure PASS event")
    evidence = report.get("prediction_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("timestamped prediction report lacks prediction_evidence")
    if evidence.get("schema_version") != PREDICTION_EVIDENCE_SCHEMA_VERSION:
        raise RuntimeError("prediction evidence schema_version mismatch")
    if evidence.get("authoritative") is not True:
        raise RuntimeError("prediction evidence is not declared authoritative")
    if Path(str(evidence.get("path") or "")).expanduser().resolve() != authoritative:
        raise RuntimeError("prediction evidence path mismatch")
    if Path(str(report.get("predictions_path") or "")).expanduser().resolve() != authoritative:
        raise RuntimeError("timestamped report predictions_path mismatch")
    if not authoritative.is_file():
        raise RuntimeError(f"authoritative prediction evidence missing: {authoritative}")

    declared_bundle_dir = Path(str(report.get("bundle_dir") or "")).expanduser().resolve()
    if bundle_dir is None:
        bundle_dir = declared_bundle_dir
    else:
        bundle_dir = bundle_dir.expanduser().resolve()
    if declared_bundle_dir != bundle_dir:
        raise RuntimeError("prediction evidence bundle_dir mismatch")
    if Path(str(report.get("dataset_dir") or "")).expanduser().resolve() != dataset_dir:
        raise RuntimeError("prediction evidence dataset_dir mismatch")
    if str(report.get("selection_score_mode") or "") != MODEL_DIRECTION_SELECTION_MODE:
        raise RuntimeError("prediction report does not use model_direction_argmax")
    if "selection_score_threshold" in report:
        raise RuntimeError("prediction report contains retired selection_score_threshold")

    metadata_path = bundle_dir / "bundle_metadata.json"
    metadata = _read_json(metadata_path)
    bundle_contract = require_model_direction_decision_contract(
        metadata,
        context=f"prediction evidence consumer bundle {bundle_dir}",
    )
    report_contract = require_model_direction_decision_contract(
        {"direction_decision_contract": report.get("direction_decision_contract")},
        context="prediction evidence timestamped report",
    )
    evidence_contract = require_model_direction_decision_contract(
        {"direction_decision_contract": evidence.get("direction_decision_contract")},
        context="prediction evidence declaration",
    )
    if report_contract != bundle_contract or evidence_contract != bundle_contract:
        raise RuntimeError("prediction evidence direction decision contract mismatch")

    expected_metadata_path = Path(
        str(evidence.get("bundle_metadata_path") or "")
    ).expanduser().resolve()
    if expected_metadata_path != metadata_path.resolve():
        raise RuntimeError("prediction evidence bundle metadata path mismatch")
    if str(evidence.get("bundle_metadata_sha256") or "").lower() != sha256_file(metadata_path):
        raise RuntimeError("prediction evidence bundle metadata SHA-256 mismatch")
    if str(report.get("bundle_metadata_sha256") or "").lower() != str(
        evidence.get("bundle_metadata_sha256") or ""
    ).lower():
        raise RuntimeError("timestamped report bundle metadata SHA-256 mismatch")
    state_sha = str(metadata.get("state_dict_sha256") or "").strip().lower()
    if str(evidence.get("model_state_dict_sha256") or "").lower() != state_sha:
        raise RuntimeError("prediction evidence model state SHA-256 mismatch")
    if str(report.get("model_state_dict_sha256") or "").lower() != state_sha:
        raise RuntimeError("timestamped report model state SHA-256 mismatch")

    observed_sha = sha256_file(authoritative)
    if str(evidence.get("sha256") or "").lower() != observed_sha:
        raise RuntimeError("prediction evidence parquet SHA-256 mismatch")
    parquet = pq.ParquetFile(authoritative)
    if int(evidence.get("rows") or -1) != int(parquet.metadata.num_rows):
        raise RuntimeError("prediction evidence row-count mismatch")
    observed_columns = list(parquet.schema_arrow.names)
    if list(evidence.get("columns") or []) != observed_columns:
        raise RuntimeError("prediction evidence column schema mismatch")
    required = list(evidence.get("required_columns") or [])
    if required != list(REQUIRED_MODEL_DIRECTION_COLUMNS):
        raise RuntimeError("prediction evidence required-column contract mismatch")
    if not set(required).issubset(observed_columns):
        raise RuntimeError("prediction evidence required columns are absent")
    if "selection_score_threshold" in observed_columns:
        raise RuntimeError("prediction evidence contains forbidden selection_score_threshold")
    validate_model_direction_parquet_semantics(authoritative)
    descriptor = parquet_schema_descriptor(authoritative)
    if list(evidence.get("parquet_schema") or []) != descriptor:
        raise RuntimeError("prediction evidence physical parquet schema mismatch")
    if str(evidence.get("parquet_schema_sha256") or "").lower() != parquet_schema_sha256(
        descriptor
    ):
        raise RuntimeError("prediction evidence parquet schema SHA-256 mismatch")

    observed_splits = _unique_strings(authoritative, "split")
    observed_models = _unique_strings(authoritative, "model")
    observed_modes = _unique_strings(authoritative, "selection_score_mode")
    if list(evidence.get("splits") or []) != observed_splits:
        raise RuntimeError("prediction evidence split declaration mismatch")
    if list(evidence.get("models") or []) != observed_models:
        raise RuntimeError("prediction evidence model declaration mismatch")
    if sorted(str(value) for value in report.get("splits") or []) != observed_splits:
        raise RuntimeError("timestamped report split declaration mismatch")
    if sorted(str(value) for value in report.get("models") or []) != observed_models:
        raise RuntimeError("timestamped report model declaration mismatch")
    if observed_modes != [MODEL_DIRECTION_SELECTION_MODE]:
        raise RuntimeError("prediction evidence contains a non-model direction mode")
    if evidence.get("selection_score_mode") != MODEL_DIRECTION_SELECTION_MODE:
        raise RuntimeError("prediction evidence declared direction mode mismatch")
    if "selection_score_threshold" in evidence:
        raise RuntimeError("prediction evidence contains retired selection_score_threshold")
    if expected_split is not None and str(expected_split) not in observed_splits:
        raise RuntimeError(f"prediction evidence lacks requested split {expected_split!r}")
    if expected_model is not None and str(expected_model) not in observed_models:
        raise RuntimeError(f"prediction evidence lacks requested model {expected_model!r}")
    return authoritative, report, evidence
