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
    MODEL_DIRECTION_FLAT_INDEX,
    MODEL_DIRECTION_LONG_INDEX,
    MODEL_DIRECTION_SELECTION_MODE,
    MODEL_DIRECTION_SHORT_INDEX,
    PUBLIC_FLAT_INDEX,
    PUBLIC_TRADE_INDEX,
    require_model_direction_decision_contract,
)


PREDICTION_EVIDENCE_SCHEMA_VERSION = (
    "entry_candidate_model_direction_prediction_evidence_v3"
)
RUNTIME_PREDICTION_EVIDENCE_SCHEMA_VERSION = (
    "entry_candidate_model_direction_prediction_evidence_v7"
)
RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION = (
    "entry_model_native_runtime_head_evidence_v5"
)
AUTHORITATIVE_PREDICTIONS_PREFIX = "selective_edge_predictions_"
REPORT_PREFIX = "ENTRY_CANDIDATE_SELECTIVE_EDGE_"
_EVENT_STAMP_RE = re.compile(r"\d{8}T\d{12}Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

PRE_CALIBRATION_EVIDENCE_STAGE = "pre_calibration"
RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE = "runtime_authoritative"
PREDICTION_EVIDENCE_STAGE_SPLITS = {
    PRE_CALIBRATION_EVIDENCE_STAGE: ("val",),
    RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE: ("test",),
}

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
RUNTIME_HEAD_PREDICTION_COLUMNS = (
    "runtime_head_evidence_schema_version",
    "runtime_head_evidence_json",
    "runtime_head_evidence_sha256",
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


def _exact_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{context} must be an exact lowercase SHA-256")
    normalized = value.strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise RuntimeError(f"{context} must be an exact lowercase SHA-256")
    return normalized


def _explicit_regular_file(path: Path, *, context: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError(f"{context} must be an explicit absolute path")
    if candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError(f"{context} must be a regular non-symlink file: {candidate}")
    return candidate.resolve(strict=True)


def _explicit_directory(path: Path, *, context: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError(f"{context} must be an explicit absolute path")
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(f"{context} must be a non-symlink directory: {candidate}")
    return candidate.resolve(strict=True)


def _exact_stage_splits(
    evidence_stage: Any,
    splits: Any,
    *,
    context: str,
) -> tuple[tuple[str, ...], bool]:
    if not isinstance(evidence_stage, str):
        raise RuntimeError(f"{context} evidence_stage is missing or invalid")
    stage = evidence_stage.strip()
    if stage != evidence_stage:
        raise RuntimeError(f"{context} evidence_stage must be exact")
    expected = PREDICTION_EVIDENCE_STAGE_SPLITS.get(stage)
    if expected is None:
        raise RuntimeError(
            f"{context} evidence_stage must be one of "
            f"{tuple(PREDICTION_EVIDENCE_STAGE_SPLITS)}"
        )
    if not isinstance(splits, (list, tuple)) or any(
        not isinstance(split, str) for split in splits
    ):
        raise RuntimeError(f"{context} expected_splits must be an exact list/tuple")
    observed = tuple(splits)
    if observed != expected:
        raise RuntimeError(
            f"{context} stage/split mismatch: stage={stage!r} "
            f"observed={observed!r} expected={expected!r}"
        )
    return expected, stage == RUNTIME_AUTHORITATIVE_EVIDENCE_STAGE


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


def validate_model_direction_parquet_semantics(
    path: Path,
    *,
    bundle_metadata: Mapping[str, Any] | None = None,
) -> None:
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
        [
            np.maximum(
                direction_logits[:, MODEL_DIRECTION_LONG_INDEX],
                direction_logits[:, MODEL_DIRECTION_SHORT_INDEX],
            ),
            direction_logits[:, MODEL_DIRECTION_FLAT_INDEX],
        ]
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
    winner_counts = np.count_nonzero(
        direction_logits == np.max(direction_logits, axis=1, keepdims=True),
        axis=1,
    )
    tied_rows = int(np.count_nonzero(winner_counts != 1))
    if tied_rows:
        raise RuntimeError(
            "prediction evidence direction logits have no unique top class; "
            f"rows={tied_rows}"
        )
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
        public_margin,
        public_logits[:, PUBLIC_TRADE_INDEX]
        - public_logits[:, PUBLIC_FLAT_INDEX],
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("prediction evidence public margin mismatch public logits")
    public_hard = numeric("public_trade_flat_hard_decision")
    if not np.array_equal(public_hard, np.rint(public_hard)) or not np.array_equal(
        public_hard.astype(np.int64), np.argmax(public_logits, axis=1)
    ):
        raise RuntimeError("prediction evidence public hard decision mismatch public logits")
    expected_public_hard = np.where(
        pred_direction.astype(np.int64) == MODEL_DIRECTION_FLAT_INDEX,
        PUBLIC_FLAT_INDEX,
        PUBLIC_TRADE_INDEX,
    )
    if not np.array_equal(public_hard.astype(np.int64), expected_public_hard):
        raise RuntimeError("prediction evidence public hard decision mismatches LONG/SHORT/FLAT")
    modes = sorted({str(value) for value in frame["selection_score_mode"].to_numpy()})
    if modes != [MODEL_DIRECTION_SELECTION_MODE]:
        raise RuntimeError(
            "prediction evidence direction mode mismatch: "
            f"observed={modes} required={[MODEL_DIRECTION_SELECTION_MODE]}"
        )

def validate_runtime_head_parquet_semantics(
    path: Path,
    *,
    bundle_metadata: Mapping[str, Any],
) -> None:
    # Local import avoids the sizing-calibration -> prediction-evidence ->
    # runtime-evidence module cycle during contract initialization.
    from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
        MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_REQUIRED_FIELDS,
        MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION,
        decode_model_native_runtime_head_evidence,
        project_model_native_path_calibration,
    )

    if (
        MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
        != RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION
    ):
        raise RuntimeError("prediction/runtime head schema constants diverged")
    parquet_columns = set(pq.ParquetFile(path).schema_arrow.names)
    duplicated_head_columns = sorted(
        parquet_columns.intersection(
            MODEL_NATIVE_RUNTIME_HEAD_EVIDENCE_REQUIRED_FIELDS
        )
        - {"runtime_head_evidence_schema_version"}
    )
    frame = pd.read_parquet(
        path,
        columns=list(
            dict.fromkeys(
                [
                    "time",
                    "pred_direction",
                    "direction_logits",
                        *RUNTIME_HEAD_PREDICTION_COLUMNS,
                        *duplicated_head_columns,
                    ]
            )
        ),
    )
    runtime_schema = sorted(
        {
            str(value)
            for value in frame["runtime_head_evidence_schema_version"].to_numpy()
        }
    )
    if runtime_schema != [RUNTIME_HEAD_EVIDENCE_SCHEMA_VERSION]:
        raise RuntimeError(
            "prediction runtime-head schema mismatch: "
            f"observed={runtime_schema}"
        )
    direction_calibration = bundle_metadata.get("direction_calibration")
    path_calibration = bundle_metadata.get("path_calibration")
    if (
        not isinstance(direction_calibration, Mapping)
        or direction_calibration.get("enabled") is not True
        or not isinstance(path_calibration, Mapping)
        or path_calibration.get("enabled") is not True
    ):
        raise RuntimeError(
            "prediction bundle lacks enabled direction/path calibration "
            "for runtime-head evidence"
        )
    path_calibration_inference = project_model_native_path_calibration(
        path_calibration,
        context="PREDICTION_BUNDLE_PATH_CALIBRATION",
    )

    def exact_python(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [exact_python(item) for item in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [exact_python(item) for item in value]
        if isinstance(value, Mapping):
            return {
                str(key): exact_python(item)
                for key, item in value.items()
            }
        return value

    for index, row in frame.iterrows():
        head = decode_model_native_runtime_head_evidence(
            row["runtime_head_evidence_json"],
            row["runtime_head_evidence_sha256"],
            context=f"PREDICTION_RUNTIME_HEAD_ROW_{index}",
        )
        head_time = pd.Timestamp(head["decision_ts"])
        row_time = pd.Timestamp(row["time"])
        if head_time.tzinfo is None:
            head_time = head_time.tz_localize("UTC")
        else:
            head_time = head_time.tz_convert("UTC")
        if row_time.tzinfo is None:
            row_time = row_time.tz_localize("UTC")
        else:
            row_time = row_time.tz_convert("UTC")
        if head_time != row_time:
            raise RuntimeError(
                f"prediction runtime-head decision_ts mismatch at row {index}"
            )
        if int(head["model_direction_index"]) != int(row["pred_direction"]):
            raise RuntimeError(
                f"prediction runtime-head direction mismatch at row {index}"
            )
        if not np.allclose(
            np.asarray(head["direction_logits"], dtype=np.float64),
            np.asarray(row["direction_logits"], dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        ):
            raise RuntimeError(
                f"prediction runtime-head logits mismatch at row {index}"
            )
        for field in duplicated_head_columns:
            if exact_python(row[field]) != exact_python(head[field]):
                raise RuntimeError(
                    "prediction runtime-head duplicated field mismatch at "
                    f"row {index}: {field}"
                )
        if (
            head["calibration_version"]
            != direction_calibration.get("version")
            or head["direction_calibration_enabled"] is not True
            or float(head["direction_calibration_temperature"])
            != float(direction_calibration.get("temperature"))
            or head["path_calibration"] != path_calibration_inference
        ):
            raise RuntimeError(
                f"prediction runtime-head calibration mismatch at row {index}"
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
    evidence_stage: str,
    requested_splits: list[str],
) -> dict[str, Any]:
    """Inspect the published parquet and return its exact lineage declaration."""

    expected_splits, runtime_authoritative = _exact_stage_splits(
        evidence_stage,
        requested_splits,
        context="prediction evidence declaration",
    )
    path = _explicit_regular_file(
        predictions_path,
        context="prediction evidence parquet",
    )
    bundle_dir = _explicit_directory(
        bundle_dir,
        context="prediction evidence bundle_dir",
    )
    metadata_path = _explicit_regular_file(
        bundle_dir / "bundle_metadata.json",
        context="prediction evidence bundle metadata",
    )
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
    runtime_columns = set(RUNTIME_HEAD_PREDICTION_COLUMNS)
    observed_runtime_columns = runtime_columns.intersection(columns)
    expected_runtime_columns = runtime_columns if runtime_authoritative else set()
    if observed_runtime_columns != expected_runtime_columns:
        raise RuntimeError(
            "prediction evidence runtime-head envelope does not match its stage: "
            f"stage={evidence_stage!r} "
            f"observed={sorted(observed_runtime_columns)} "
            f"expected={sorted(expected_runtime_columns)}"
        )
    validate_model_direction_parquet_semantics(path)
    if runtime_authoritative:
        validate_runtime_head_parquet_semantics(
            path,
            bundle_metadata=bundle_metadata,
        )
    splits = _unique_strings(path, "split")
    if splits != list(expected_splits):
        raise RuntimeError(
            "prediction evidence split mismatch: "
            f"observed={splits} expected={list(expected_splits)}"
        )
    modes = _unique_strings(path, "selection_score_mode")
    if modes != [MODEL_DIRECTION_SELECTION_MODE]:
        raise RuntimeError(
            "prediction evidence direction mode mismatch: "
            f"observed={modes} required={[MODEL_DIRECTION_SELECTION_MODE]}"
        )
    descriptor = parquet_schema_descriptor(path)
    return {
        "schema_version": (
            RUNTIME_PREDICTION_EVIDENCE_SCHEMA_VERSION
            if runtime_authoritative
            else PREDICTION_EVIDENCE_SCHEMA_VERSION
        ),
        "evidence_stage": evidence_stage,
        "authoritative": runtime_authoritative,
        "runtime_head_evidence_authoritative": runtime_authoritative,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(parquet.metadata.num_rows),
        "splits": splits,
        "models": _unique_strings(path, "model"),
        "columns": columns,
        "required_columns": [
            *REQUIRED_MODEL_DIRECTION_COLUMNS,
            *(RUNTIME_HEAD_PREDICTION_COLUMNS if runtime_authoritative else ()),
        ],
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
    expected_sha256: str,
    prediction_report_path: Path,
    bundle_dir: Path,
    dataset_dir: Path,
    expected_stage: str,
    expected_splits: tuple[str, ...],
    expected_model: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Re-hash one explicitly pinned immutable prediction event.

    Returns ``(authoritative_path, timestamped_report, evidence_declaration)``.
    Every mismatch raises before a consumer can read predictions. Resolution
    uses only the caller's explicit regular path, expected SHA-256, exact stage
    and split tuple, and explicit report/bundle/dataset parent lineage. It never
    scans for a newest sibling or follows a mutable alias.
    """

    if not isinstance(expected_splits, tuple):
        raise RuntimeError(
            "prediction evidence resolver expected_splits must be an exact tuple"
        )
    stage_splits, runtime_authoritative = _exact_stage_splits(
        expected_stage,
        expected_splits,
        context="prediction evidence resolver",
    )
    pinned_sha256 = _exact_sha256(
        expected_sha256,
        context="prediction evidence expected_sha256",
    )
    # A mutable locator is rejected before any filesystem access: naming the
    # evidence wrongly must never reach I/O, hashing or lineage resolution.
    _timestamped_report_for_predictions(Path(requested_path))
    authoritative = _explicit_regular_file(
        requested_path,
        context="authoritative prediction evidence",
    )
    report_path = _explicit_regular_file(
        prediction_report_path,
        context="prediction evidence parent report",
    )
    bundle_dir = _explicit_directory(
        bundle_dir,
        context="prediction evidence bundle_dir",
    )
    dataset_dir = _explicit_directory(
        dataset_dir,
        context="prediction evidence dataset_dir",
    )
    expected_report_path = _timestamped_report_for_predictions(authoritative).resolve()
    if report_path != expected_report_path:
        raise RuntimeError(
            "explicit prediction report does not match the timestamped prediction "
            f"event: predictions={authoritative} report={report_path} "
            f"expected={expected_report_path}"
        )
    observed_sha = sha256_file(authoritative)
    if observed_sha != pinned_sha256:
        raise RuntimeError(
            "prediction evidence does not match caller-pinned expected SHA-256"
        )
    report = _read_json(report_path)

    if Path(str(report.get("json_path") or "")).expanduser().resolve() != report_path:
        raise RuntimeError("timestamped prediction report json_path mismatch")
    if report.get("schema_version") != "entry_candidate_selective_edge_v1":
        raise RuntimeError("timestamped prediction report schema_version mismatch")
    if str(report.get("decision") or "") != "PASS" or report.get("failures"):
        raise RuntimeError("timestamped prediction report is not a zero-failure PASS event")
    if report.get("evidence_stage") != expected_stage:
        raise RuntimeError("timestamped prediction report evidence_stage mismatch")
    evidence = report.get("prediction_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("timestamped prediction report lacks prediction_evidence")
    evidence_schema = evidence.get("schema_version")
    expected_evidence_schema = (
        RUNTIME_PREDICTION_EVIDENCE_SCHEMA_VERSION
        if runtime_authoritative
        else PREDICTION_EVIDENCE_SCHEMA_VERSION
    )
    if evidence_schema != expected_evidence_schema:
        raise RuntimeError("prediction evidence schema_version mismatch")
    if evidence.get("evidence_stage") != expected_stage:
        raise RuntimeError("prediction evidence evidence_stage mismatch")
    if evidence.get("runtime_head_evidence_authoritative") is not runtime_authoritative:
        raise RuntimeError(
            "prediction runtime-head authority declaration mismatch"
        )
    if evidence.get("authoritative") is not runtime_authoritative:
        raise RuntimeError("prediction evidence authority does not match its stage")
    if Path(str(evidence.get("path") or "")).expanduser().resolve() != authoritative:
        raise RuntimeError("prediction evidence path mismatch")
    if Path(str(report.get("predictions_path") or "")).expanduser().resolve() != authoritative:
        raise RuntimeError("timestamped report predictions_path mismatch")
    declared_bundle_dir = Path(str(report.get("bundle_dir") or "")).expanduser().resolve()
    if declared_bundle_dir != bundle_dir:
        raise RuntimeError("prediction evidence bundle_dir mismatch")
    if Path(str(report.get("dataset_dir") or "")).expanduser().resolve() != dataset_dir:
        raise RuntimeError("prediction evidence dataset_dir mismatch")
    if str(report.get("selection_score_mode") or "") != MODEL_DIRECTION_SELECTION_MODE:
        raise RuntimeError("prediction report does not use model_direction_argmax")
    if "selection_score_threshold" in report:
        raise RuntimeError("prediction report contains retired selection_score_threshold")

    metadata_path = _explicit_regular_file(
        bundle_dir / "bundle_metadata.json",
        context="prediction evidence bundle metadata",
    )
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

    if str(evidence.get("sha256") or "").lower() != pinned_sha256:
        raise RuntimeError("prediction evidence parquet SHA-256 mismatch")
    parquet = pq.ParquetFile(authoritative)
    if int(evidence.get("rows") or -1) != int(parquet.metadata.num_rows):
        raise RuntimeError("prediction evidence row-count mismatch")
    observed_columns = list(parquet.schema_arrow.names)
    if list(evidence.get("columns") or []) != observed_columns:
        raise RuntimeError("prediction evidence column schema mismatch")
    required = list(evidence.get("required_columns") or [])
    expected_required = [
        *REQUIRED_MODEL_DIRECTION_COLUMNS,
        *(RUNTIME_HEAD_PREDICTION_COLUMNS if runtime_authoritative else ()),
    ]
    if required != expected_required:
        raise RuntimeError("prediction evidence required-column contract mismatch")
    if not set(required).issubset(observed_columns):
        raise RuntimeError("prediction evidence required columns are absent")
    if "selection_score_threshold" in observed_columns:
        raise RuntimeError("prediction evidence contains forbidden selection_score_threshold")
    validate_model_direction_parquet_semantics(authoritative)
    if runtime_authoritative:
        validate_runtime_head_parquet_semantics(
            authoritative,
            bundle_metadata=metadata,
        )
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
    exact_splits = list(stage_splits)
    if observed_splits != exact_splits:
        raise RuntimeError(
            "prediction evidence physical split set does not match exact stage splits"
        )
    if list(evidence.get("splits") or []) != exact_splits:
        raise RuntimeError("prediction evidence split declaration mismatch")
    if list(evidence.get("models") or []) != observed_models:
        raise RuntimeError("prediction evidence model declaration mismatch")
    if list(report.get("splits") or []) != exact_splits:
        raise RuntimeError("timestamped report split declaration mismatch")
    if sorted(str(value) for value in report.get("models") or []) != observed_models:
        raise RuntimeError("timestamped report model declaration mismatch")
    if observed_modes != [MODEL_DIRECTION_SELECTION_MODE]:
        raise RuntimeError("prediction evidence contains a non-model direction mode")
    if evidence.get("selection_score_mode") != MODEL_DIRECTION_SELECTION_MODE:
        raise RuntimeError("prediction evidence declared direction mode mismatch")
    if "selection_score_threshold" in evidence:
        raise RuntimeError("prediction evidence contains retired selection_score_threshold")
    if expected_model is not None and observed_models != [str(expected_model)]:
        raise RuntimeError(
            "prediction evidence model set is not exact: "
            f"observed={observed_models} expected={[str(expected_model)]}"
        )
    return authoritative, report, evidence
