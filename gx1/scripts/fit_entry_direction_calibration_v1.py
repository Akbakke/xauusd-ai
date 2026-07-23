"""Produce an immutable calibrated model-native Entry bundle.

Calibration is fitted only from an explicitly named, immutable prediction
event on one held-out ``val`` or ``calibration`` split.  The source bundle is
never modified.  Execution creates a new timestamped sibling bundle, keeps
the model state and lock byte-identical, and records a hash-bound calibration
event inside the derived bundle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from gx1.contracts.immutable_event_authority_v1 import require_newest_immutable_event
from gx1.models.entry_v10.entry_v10_bundle import (
    _require_exact_model_native_bundle_metadata,
    _require_model_native_learned_component_liveness,
    _require_model_native_state_head_contract,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    CORE_ARTIFACTS as BUNDLE_COMMIT_CORE_ARTIFACTS,
    publish_bundle_directory_noreplace,
    require_bundle_commit_manifest,
    write_bundle_commit_manifest,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    resolve_and_validate_prediction_evidence,
    sha256_file,
)


SCHEMA_VERSION = "entry_model_native_immutable_calibration_v2"
DIRECTION_CALIBRATION_VERSION = "entry_model_native_direction_calibration_v1"
PATH_CALIBRATION_VERSION = "entry_model_native_path_calibration_v1"
CLASS_COLUMNS = ("p_long", "p_short", "p_flat")
CLASS_ORDER = ("LONG", "SHORT", "FLAT")
HELD_OUT_SPLITS = ("val", "calibration")
BUNDLE_REQUIRED_FILES = (
    "MASTER_TRANSFORMER_LOCK.json",
    "bundle_metadata.json",
    "model_state_dict.pt",
)
CALIBRATION_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_CALIBRATION_"
_TIMESTAMPED_DIR_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*_(?P<stamp>\d{8}T\d{12}Z)"
)
_MUTABLE_TOKENS = frozenset({"active", "current", "latest", "mutable"})
_HEX64_RE = re.compile(r"[0-9a-f]{64}")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _direction_odds_cap(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 1.0:
        raise argparse.ArgumentTypeError("must be finite and >= 1.0")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle-dir", required=True)
    parser.add_argument("--out-bundle-dir", required=True)
    parser.add_argument(
        "--predictions-parquet",
        required=True,
        help="exact timestamped selective_edge_predictions event",
    )
    parser.add_argument(
        "--prediction-report-json",
        required=True,
        help="matching timestamped ENTRY_CANDIDATE_SELECTIVE_EDGE event",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--heads", required=True, choices=("direction", "path"))
    parser.add_argument("--fit-split", required=True, choices=HELD_OUT_SPLITS)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--min-fit-rows", required=True, type=_positive_int)
    parser.add_argument(
        "--direction-odds-cap",
        type=_direction_odds_cap,
        help="required for direction; forbidden for path",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"could not read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_fsync(path: Path, value: Mapping[str, Any]) -> None:
    # Mapping order is contractual for specialist input indices.  Do not sort
    # bundle metadata while serializing the derived bundle.
    encoded = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _mutable_alias_tokens(path: Path) -> list[str]:
    tokens = {part.lower() for part in path.parent.parts}
    tokens.update(token for token in re.split(r"[._-]+", path.name.lower()) if token)
    return sorted(tokens & _MUTABLE_TOKENS)


def _copy_file_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())


def _timestamped_bundle_dir(raw: str, *, label: str, must_exist: bool) -> tuple[Path, str]:
    supplied = Path(raw).expanduser()
    if not supplied.is_absolute():
        raise RuntimeError(f"{label} must be an absolute canonical path")
    if supplied.is_symlink():
        raise RuntimeError(f"{label} cannot be a symlink: {supplied}")
    path = supplied.resolve(strict=False)
    if path != supplied:
        raise RuntimeError(f"{label} is not canonical: supplied={supplied} resolved={path}")
    match = _TIMESTAMPED_DIR_RE.fullmatch(path.name)
    if match is None:
        raise RuntimeError(
            f"{label} basename must end in an exact microsecond UTC stamp: {path.name!r}"
        )
    mutable = _mutable_alias_tokens(path)
    if mutable:
        raise RuntimeError(f"{label} contains mutable alias tokens: {mutable}")
    stamp = match.group("stamp")
    try:
        parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise RuntimeError(f"{label} has an invalid UTC stamp: {stamp}") from exc
    if parsed.strftime("%Y%m%dT%H%M%S%fZ") != stamp:
        raise RuntimeError(f"{label} has a non-canonical UTC stamp: {stamp}")
    if must_exist:
        if not path.is_dir():
            raise RuntimeError(f"{label} does not exist as a directory: {path}")
    elif path.exists():
        raise RuntimeError(f"immutable output bundle already exists: {path}")
    return path, stamp


def _require_plain_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise RuntimeError(f"{label} cannot be a symlink: {path}")
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")


def _load_state_dict(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"could not load model state {path}: {exc}") from exc
    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(f"model state must be a non-empty mapping: {path}")
    return value


def _bundle_artifact_hashes(bundle_dir: Path) -> dict[str, str]:
    names = list(BUNDLE_REQUIRED_FILES)
    hashes: dict[str, str] = {}
    for name in names:
        path = bundle_dir / name
        _require_plain_file(path, label=f"bundle artifact {name}")
        digest = sha256_file(path)
        if _HEX64_RE.fullmatch(digest) is None:
            raise RuntimeError(f"could not hash bundle artifact: {path}")
        hashes[name] = digest
    return hashes


def _validate_source_bundle(
    source_bundle_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    require_bundle_commit_manifest(source_bundle_dir)
    required = {name: source_bundle_dir / name for name in BUNDLE_REQUIRED_FILES}
    for name, path in required.items():
        _require_plain_file(path, label=name)
    metadata = _read_json(required["bundle_metadata.json"], label="bundle metadata")
    lock = _read_json(required["MASTER_TRANSFORMER_LOCK.json"], label="bundle lock")
    _require_exact_model_native_bundle_metadata(metadata, lock)

    state_path = required["model_state_dict.pt"]
    observed_state_sha = sha256_file(state_path)
    if _HEX64_RE.fullmatch(observed_state_sha) is None:
        raise RuntimeError("model state SHA-256 is not canonical")
    if lock.get("model_path_relative") != "model_state_dict.pt":
        raise RuntimeError("bundle lock model_path_relative is not exact")
    if str(lock.get("model_sha256") or "").lower() != observed_state_sha:
        raise RuntimeError("bundle lock model SHA-256 mismatch")
    if str(metadata.get("state_dict_sha256") or "").lower() != observed_state_sha:
        raise RuntimeError("bundle metadata model SHA-256 mismatch")

    state_dict = _load_state_dict(state_path)
    _require_model_native_state_head_contract(metadata, state_dict)
    _require_model_native_learned_component_liveness(state_dict)
    artifact_hashes = _bundle_artifact_hashes(source_bundle_dir)
    return metadata, lock, artifact_hashes


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(f"prediction column {name} is not numeric") from exc
    if values.ndim != 1 or len(values) != len(frame) or not np.isfinite(values).all():
        raise RuntimeError(f"prediction column {name} is non-finite or malformed")
    return values


def _scoped_frame(
    predictions_path: Path,
    *,
    columns: Sequence[str],
    fit_split: str,
    model: str,
    min_fit_rows: int,
) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(predictions_path, columns=list(columns))
    except Exception as exc:
        raise RuntimeError(f"could not read required calibration columns: {exc}") from exc
    scoped = frame[
        frame["split"].astype(str).eq(fit_split)
        & frame["model"].astype(str).eq(model)
    ].copy()
    if len(scoped) < min_fit_rows:
        raise RuntimeError(
            f"only {len(scoped)} calibration rows for split={fit_split!r} "
            f"model={model!r}; require >= {min_fit_rows}"
        )
    return scoped


def _direction_nll(params: np.ndarray, logp: np.ndarray, labels: np.ndarray) -> float:
    if params.shape != (3,) or not np.isfinite(params).all():
        return float("inf")
    temperature = float(np.exp(params[0]))
    if not math.isfinite(temperature) or temperature <= 0.0:
        return float("inf")
    bias = np.asarray([params[1], params[2], -params[1] - params[2]], dtype=np.float64)
    logits = logp / temperature + bias
    value = np.mean(logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels])
    return float(value) if math.isfinite(float(value)) else float("inf")


def _fit_direction(
    frame: pd.DataFrame,
    *,
    odds_cap: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    probabilities = np.column_stack([_numeric_column(frame, name) for name in CLASS_COLUMNS])
    if probabilities.shape != (len(frame), 3):
        raise RuntimeError("direction probabilities must have exact shape (rows, 3)")
    if np.any(probabilities <= 0.0) or np.any(probabilities > 1.0):
        raise RuntimeError("direction probabilities must be strictly positive and <= 1")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise RuntimeError("direction probability rows do not sum to one")

    raw_labels = _numeric_column(frame, "y_direction")
    if not np.array_equal(raw_labels, np.rint(raw_labels)):
        raise RuntimeError("direction labels must be exact integers")
    labels = raw_labels.astype(np.int64)
    if np.any((labels < 0) | (labels > 2)):
        raise RuntimeError("direction labels must use only LONG=0, SHORT=1, FLAT=2")
    observed_classes = sorted(int(value) for value in np.unique(labels))
    if observed_classes != [0, 1, 2]:
        raise RuntimeError(f"direction calibration is missing classes: observed={observed_classes}")

    logp = np.log(probabilities)
    initial = np.zeros(3, dtype=np.float64)
    nll_before = _direction_nll(initial, logp, labels)
    result = minimize(
        _direction_nll,
        initial,
        args=(logp, labels),
        method="L-BFGS-B",
        bounds=((-5.0, 5.0), (-10.0, 10.0), (-10.0, 10.0)),
        options={"maxiter": 4000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if result.success is not True:
        raise RuntimeError(f"direction calibration optimizer failed: {result.message}")
    fitted = np.asarray(result.x, dtype=np.float64)
    if fitted.shape != (3,) or not np.isfinite(fitted).all():
        raise RuntimeError("direction calibration optimizer returned malformed parameters")
    nll_after = _direction_nll(fitted, logp, labels)
    if not (math.isfinite(nll_before) and math.isfinite(nll_after) and nll_after < nll_before):
        raise RuntimeError(
            f"direction calibration lacks strict NLL improvement: {nll_before} -> {nll_after}"
        )
    temperature = float(np.exp(fitted[0]))
    bias = [float(fitted[1]), float(fitted[2]), float(-fitted[1] - fitted[2])]
    equal_logits = np.asarray(bias, dtype=np.float64)
    equal_prob = np.exp(equal_logits - logsumexp(equal_logits))
    long_short_odds = float(
        max(equal_prob[0] / equal_prob[1], equal_prob[1] / equal_prob[0])
    )
    if not math.isfinite(long_short_odds) or long_short_odds > odds_cap:
        raise RuntimeError(
            "direction calibration exceeds the explicit equal-logit LONG/SHORT "
            f"odds cap: observed={long_short_odds} cap={odds_cap}"
        )
    calibration = {
        "enabled": True,
        "version": DIRECTION_CALIBRATION_VERSION,
        "temperature": temperature,
        "bias": bias,
        "class_order": list(CLASS_ORDER),
    }
    metrics = {
        "fitted_rows": int(len(frame)),
        "observed_classes": observed_classes,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "nll_improvement": nll_before - nll_after,
        "equal_logit_long_probability": float(equal_prob[0]),
        "equal_logit_short_probability": float(equal_prob[1]),
        "equal_logit_flat_probability": float(equal_prob[2]),
        "equal_logit_long_short_odds": long_short_odds,
        "direction_odds_cap": odds_cap,
        "optimizer": {
            "method": "L-BFGS-B",
            "success": True,
            "iterations": int(result.nit),
        },
    }
    return calibration, metrics


def _binary_bce(params: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> float:
    if params.shape != (2,) or not np.isfinite(params).all():
        return float("inf")
    temperature = float(np.exp(params[0]))
    if not math.isfinite(temperature) or temperature <= 0.0:
        return float("inf")
    calibrated = logits / temperature + float(params[1])
    value = np.mean(np.logaddexp(0.0, calibrated) - labels * calibrated)
    return float(value) if math.isfinite(float(value)) else float("inf")


def _fit_path(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    path_pred = _numeric_column(frame, "path_quality_pred")
    path_target = _numeric_column(frame, "path_quality_bps")
    bad_probability = _numeric_column(frame, "bad_path_prob")
    raw_bad_labels = _numeric_column(frame, "y_bad_path")
    if np.any(bad_probability <= 0.0) or np.any(bad_probability >= 1.0):
        raise RuntimeError("bad-path probabilities must be strictly between zero and one")
    if not np.array_equal(raw_bad_labels, np.rint(raw_bad_labels)):
        raise RuntimeError("bad-path labels must be exact integers")
    bad_labels = raw_bad_labels.astype(np.int64)
    observed_bad_classes = sorted(int(value) for value in np.unique(bad_labels))
    if observed_bad_classes != [0, 1]:
        raise RuntimeError(
            f"path calibration is missing bad-path classes: observed={observed_bad_classes}"
        )

    design = np.column_stack([path_pred, np.ones_like(path_pred)])
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, path_target, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(f"path-quality affine fit failed: {exc}") from exc
    if rank != 2 or coefficients.shape != (2,) or not np.isfinite(coefficients).all():
        raise RuntimeError("path-quality affine fit is rank-deficient or non-finite")
    path_scale, path_shift = (float(value) for value in coefficients)
    if path_scale <= 0.0:
        raise RuntimeError(f"path-quality affine scale must be positive, got {path_scale}")
    path_calibrated = path_scale * path_pred + path_shift
    mse_before = float(np.mean(np.square(path_pred - path_target)))
    mse_after = float(np.mean(np.square(path_calibrated - path_target)))
    if not (math.isfinite(mse_before) and math.isfinite(mse_after) and mse_after < mse_before):
        raise RuntimeError(
            f"path-quality calibration lacks strict MSE improvement: {mse_before} -> {mse_after}"
        )
    corr = float(np.corrcoef(path_pred, path_target)[0, 1])
    if not math.isfinite(corr) or corr <= 0.0:
        raise RuntimeError(f"path-quality raw correlation must be positive, got {corr}")

    bad_logits = np.log(bad_probability) - np.log1p(-bad_probability)
    initial = np.zeros(2, dtype=np.float64)
    bce_before = _binary_bce(initial, bad_logits, bad_labels)
    result = minimize(
        _binary_bce,
        initial,
        args=(bad_logits, bad_labels),
        method="L-BFGS-B",
        bounds=((-5.0, 5.0), (-10.0, 10.0)),
        options={"maxiter": 4000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if result.success is not True:
        raise RuntimeError(f"bad-path calibration optimizer failed: {result.message}")
    fitted = np.asarray(result.x, dtype=np.float64)
    if fitted.shape != (2,) or not np.isfinite(fitted).all():
        raise RuntimeError("bad-path calibration optimizer returned malformed parameters")
    bce_after = _binary_bce(fitted, bad_logits, bad_labels)
    if not (math.isfinite(bce_before) and math.isfinite(bce_after) and bce_after < bce_before):
        raise RuntimeError(
            f"bad-path calibration lacks strict BCE improvement: {bce_before} -> {bce_after}"
        )
    bad_temperature = float(np.exp(fitted[0]))
    bad_bias = float(fitted[1])
    calibration = {
        "enabled": True,
        "version": PATH_CALIBRATION_VERSION,
        "path_quality_scale": path_scale,
        "path_quality_shift": path_shift,
        "bad_path_temperature": bad_temperature,
        "bad_path_bias": bad_bias,
    }
    metrics = {
        "fitted_rows": int(len(frame)),
        "observed_bad_path_classes": observed_bad_classes,
        "path_quality_mse_before": mse_before,
        "path_quality_mse_after": mse_after,
        "path_quality_mse_improvement": mse_before - mse_after,
        "path_quality_raw_correlation": corr,
        "bad_path_bce_before": bce_before,
        "bad_path_bce_after": bce_after,
        "bad_path_bce_improvement": bce_before - bce_after,
        "optimizer": {
            "method": "L-BFGS-B",
            "success": True,
            "iterations": int(result.nit),
        },
    }
    return calibration, metrics


def _copy_bundle_to_stage(
    *,
    source_bundle_dir: Path,
    stage_dir: Path,
) -> None:
    for name in ("MASTER_TRANSFORMER_LOCK.json", "model_state_dict.pt"):
        _copy_file_fsync(source_bundle_dir / name, stage_dir / name)


def _publish_bundle(
    *,
    source_bundle_dir: Path,
    out_bundle_dir: Path,
    out_stamp: str,
    source_metadata: Mapping[str, Any],
    source_lock: Mapping[str, Any],
    source_hashes: Mapping[str, str],
    calibration_key: str,
    calibration: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    parent = out_bundle_dir.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError(f"output bundle parent must be an existing real directory: {parent}")
    if out_bundle_dir.exists():
        raise RuntimeError(f"immutable output bundle already exists: {out_bundle_dir}")
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{out_bundle_dir.name}.", dir=parent))
    published = False
    try:
        if _bundle_artifact_hashes(source_bundle_dir) != dict(source_hashes):
            raise RuntimeError("source bundle changed after validation")
        if sha256_file(Path(str(provenance["predictions_path"]))) != provenance[
            "predictions_sha256"
        ]:
            raise RuntimeError("prediction parquet changed after validation")
        if sha256_file(Path(str(provenance["prediction_report_path"]))) != provenance[
            "prediction_report_sha256"
        ]:
            raise RuntimeError("prediction report changed after validation")
        _copy_bundle_to_stage(
            source_bundle_dir=source_bundle_dir,
            stage_dir=stage_dir,
        )
        output_metadata = copy.deepcopy(dict(source_metadata))
        output_metadata[calibration_key] = dict(calibration)
        if output_metadata.get("model_native_training_objective") != source_lock.get(
            "model_native_training_objective"
        ):
            raise RuntimeError(
                "model-native training objective changed during calibration derivation"
            )
        metadata_path = stage_dir / "bundle_metadata.json"
        _write_json_fsync(metadata_path, output_metadata)

        staged_lock = _read_json(stage_dir / "MASTER_TRANSFORMER_LOCK.json", label="staged lock")
        staged_metadata = _read_json(metadata_path, label="staged bundle metadata")
        if staged_lock != dict(source_lock):
            raise RuntimeError("staged lock content differs from source lock")
        if staged_metadata != output_metadata:
            raise RuntimeError("staged metadata content differs from derived metadata")
        _require_exact_model_native_bundle_metadata(staged_metadata, staged_lock)
        if sha256_file(stage_dir / "model_state_dict.pt") != source_hashes["model_state_dict.pt"]:
            raise RuntimeError("staged model state differs from source model state")
        if sha256_file(stage_dir / "MASTER_TRANSFORMER_LOCK.json") != source_hashes[
            "MASTER_TRANSFORMER_LOCK.json"
        ]:
            raise RuntimeError("staged lock bytes differ from source lock")

        output_hashes = _bundle_artifact_hashes(stage_dir)
        unchanged_names = {
            "MASTER_TRANSFORMER_LOCK.json",
            "model_state_dict.pt",
        }
        changed_copies = sorted(
            name
            for name in unchanged_names
            if output_hashes.get(name) != source_hashes.get(name)
        )
        if changed_copies:
            raise RuntimeError(f"derived bundle copied artifacts changed: {changed_copies}")
        event_name = f"{CALIBRATION_EVENT_PREFIX}{out_stamp}.json"
        final_event_path = out_bundle_dir / event_name
        event = {
            "schema_version": SCHEMA_VERSION,
            "decision": "PASS",
            "failures": [],
            "created_utc": provenance["fitted_at_utc"],
            "json_path": str(final_event_path),
            "head": provenance["head"],
            "model": provenance["model"],
            "fit_split": provenance["fit_split"],
            "min_fit_rows": provenance["min_fit_rows"],
            "run_id": provenance["run_id"],
            "calibration": dict(calibration),
            "metrics": dict(metrics),
            "source_bundle": {
                "path": str(source_bundle_dir),
                "artifact_sha256": dict(source_hashes),
                "artifact_set_sha256": _canonical_json_sha256(source_hashes),
            },
            "output_bundle": {
                "path": str(out_bundle_dir),
                "artifact_sha256": output_hashes,
                "artifact_set_sha256": _canonical_json_sha256(output_hashes),
                "lock_and_state_unchanged": (
                    output_hashes["MASTER_TRANSFORMER_LOCK.json"]
                    == source_hashes["MASTER_TRANSFORMER_LOCK.json"]
                    and output_hashes["model_state_dict.pt"]
                    == source_hashes["model_state_dict.pt"]
                ),
                "training_objective_unchanged": (
                    output_metadata["model_native_training_objective"]
                    == source_metadata["model_native_training_objective"]
                    == source_lock["model_native_training_objective"]
                ),
            },
            "prediction_evidence": dict(provenance["prediction_evidence"]),
            "predictions": {
                "path": provenance["predictions_path"],
                "sha256": provenance["predictions_sha256"],
            },
            "prediction_report": {
                "path": provenance["prediction_report_path"],
                "sha256": provenance["prediction_report_sha256"],
            },
            "dataset_dir": provenance["dataset_dir"],
        }
        if event["output_bundle"]["lock_and_state_unchanged"] is not True:
            raise RuntimeError("derived bundle lock/state identity proof failed")
        if event["output_bundle"]["training_objective_unchanged"] is not True:
            raise RuntimeError("derived bundle training-objective identity proof failed")
        _write_json_fsync(stage_dir / event_name, event)
        write_bundle_commit_manifest(
            bundle_dir=stage_dir,
            artifact_names=(*BUNDLE_COMMIT_CORE_ARTIFACTS, event_name),
            bundle_kind="calibrated",
            created_at_utc=str(provenance["fitted_at_utc"]),
        )
        stage_fd = os.open(stage_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        publish_bundle_directory_noreplace(stage_dir, out_bundle_dir)
        published = True
        return final_event_path, event
    finally:
        if not published:
            shutil.rmtree(stage_dir, ignore_errors=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_bundle_dir, source_stamp = _timestamped_bundle_dir(
        args.source_bundle_dir,
        label="source bundle",
        must_exist=True,
    )
    out_bundle_dir, out_stamp = _timestamped_bundle_dir(
        args.out_bundle_dir,
        label="output bundle",
        must_exist=False,
    )
    if source_bundle_dir == out_bundle_dir:
        raise RuntimeError("source and output bundle paths must differ")
    if out_stamp <= source_stamp:
        raise RuntimeError("output bundle stamp must be later than source bundle stamp")
    if args.heads == "direction" and args.direction_odds_cap is None:
        raise RuntimeError("--direction-odds-cap is required for direction calibration")
    if args.heads == "path" and args.direction_odds_cap is not None:
        raise RuntimeError("--direction-odds-cap is forbidden for path calibration")
    if not str(args.model).strip():
        raise RuntimeError("--model cannot be blank")
    from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id

    args.run_id = require_entry_run_id(args.run_id)

    dataset_supplied = Path(args.dataset_dir).expanduser()
    if not dataset_supplied.is_absolute() or dataset_supplied.is_symlink():
        raise RuntimeError("dataset dir must be an absolute, non-symlink path")
    dataset_dir = dataset_supplied.resolve(strict=False)
    if dataset_dir != dataset_supplied or not dataset_dir.is_dir():
        raise RuntimeError(f"dataset dir is not canonical or missing: {dataset_supplied}")
    if mutable := _mutable_alias_tokens(dataset_dir):
        raise RuntimeError(f"dataset dir contains mutable alias tokens: {mutable}")

    source_metadata, source_lock, source_hashes = (
        _validate_source_bundle(source_bundle_dir)
    )
    calibration_key = f"{args.heads}_calibration"
    if calibration_key in source_metadata:
        raise RuntimeError(
            f"source bundle already carries selected {calibration_key}; re-fit is forbidden"
        )

    requested_predictions = Path(args.predictions_parquet).expanduser()
    requested_report = Path(args.prediction_report_json).expanduser()
    if not requested_predictions.is_absolute() or not requested_report.is_absolute():
        raise RuntimeError("prediction parquet and report paths must be absolute")
    for supplied, label in (
        (requested_predictions, "prediction parquet"),
        (requested_report, "prediction report"),
    ):
        if supplied.is_symlink() or supplied.resolve(strict=False) != supplied:
            raise RuntimeError(f"{label} must be a canonical, non-symlink path: {supplied}")
        if not supplied.is_file():
            raise RuntimeError(f"{label} is missing: {supplied}")
        if mutable := _mutable_alias_tokens(supplied):
            raise RuntimeError(f"{label} contains mutable alias tokens: {mutable}")
    predictions_path, _report, prediction_evidence = resolve_and_validate_prediction_evidence(
        requested_predictions,
        prediction_report_path=requested_report,
        bundle_dir=source_bundle_dir,
        dataset_dir=dataset_dir,
        expected_split=args.fit_split,
        expected_model=args.model,
    )
    prediction_report_path = requested_report.resolve()
    prediction_stamp_match = re.fullmatch(
        r"selective_edge_predictions_(\d{8}T\d{12}Z)\.parquet",
        predictions_path.name,
    )
    if prediction_stamp_match is None:
        raise RuntimeError("prediction parquet lacks the exact immutable event identity")
    if out_stamp <= prediction_stamp_match.group(1):
        raise RuntimeError("output bundle stamp must be later than prediction evidence stamp")
    validated_prediction_sha = str(prediction_evidence.get("sha256") or "").lower()
    if _HEX64_RE.fullmatch(validated_prediction_sha) is None or sha256_file(
        predictions_path
    ) != validated_prediction_sha:
        raise RuntimeError("prediction parquet hash changed after evidence validation")
    validated_report_sha = sha256_file(prediction_report_path)
    if _HEX64_RE.fullmatch(validated_report_sha) is None:
        raise RuntimeError("prediction report SHA-256 is not canonical")
    if args.heads == "direction":
        columns = ("split", "model", "y_direction", *CLASS_COLUMNS)
    else:
        columns = (
            "split",
            "model",
            "path_quality_pred",
            "bad_path_prob",
            "path_quality_bps",
            "y_bad_path",
        )
    frame = _scoped_frame(
        predictions_path,
        columns=columns,
        fit_split=args.fit_split,
        model=args.model,
        min_fit_rows=args.min_fit_rows,
    )
    if args.heads == "direction":
        calibration, metrics = _fit_direction(
            frame,
            odds_cap=float(args.direction_odds_cap),
        )
    else:
        calibration, metrics = _fit_path(frame)

    if sha256_file(predictions_path) != validated_prediction_sha:
        raise RuntimeError("prediction parquet changed while calibration was fitting")
    if sha256_file(prediction_report_path) != validated_report_sha:
        raise RuntimeError("prediction report changed while calibration was fitting")

    fitted_at = datetime.strptime(out_stamp, "%Y%m%dT%H%M%S%fZ").replace(
        tzinfo=timezone.utc
    ).isoformat()
    provenance = {
        "head": args.heads,
        "model": str(args.model),
        "fit_split": str(args.fit_split),
        "min_fit_rows": int(args.min_fit_rows),
        "run_id": str(args.run_id),
        "fitted_at_utc": fitted_at,
        "dataset_dir": str(dataset_dir),
        "predictions_path": str(predictions_path),
        "predictions_sha256": validated_prediction_sha,
        "prediction_report_path": str(prediction_report_path),
        "prediction_report_sha256": validated_report_sha,
        "prediction_evidence": prediction_evidence,
    }
    calibration.update(
        {
            "fitted_at_utc": fitted_at,
            "fitted_on_split": str(args.fit_split),
            "fitted_rows": int(metrics["fitted_rows"]),
            "model": str(args.model),
            "min_fit_rows": int(args.min_fit_rows),
            "run_id": str(args.run_id),
            "source_bundle_dir": str(source_bundle_dir),
            "source_bundle_metadata_sha256": source_hashes["bundle_metadata.json"],
            "predictions_path": str(predictions_path),
            "predictions_sha256": provenance["predictions_sha256"],
            "prediction_report_path": str(prediction_report_path),
            "prediction_report_sha256": provenance["prediction_report_sha256"],
        }
    )
    preview = {
        "decision": "DRY_RUN_PASS" if args.dry_run else "PASS",
        "source_bundle_dir": str(source_bundle_dir),
        "out_bundle_dir": str(out_bundle_dir),
        "head": args.heads,
        "calibration": calibration,
        "metrics": metrics,
        "source_artifact_sha256": source_hashes,
        "predictions_sha256": provenance["predictions_sha256"],
        "prediction_report_sha256": provenance["prediction_report_sha256"],
    }
    if args.dry_run:
        return preview
    event_path, event = _publish_bundle(
        source_bundle_dir=source_bundle_dir,
        out_bundle_dir=out_bundle_dir,
        out_stamp=out_stamp,
        source_metadata=source_metadata,
        source_lock=source_lock,
        source_hashes=source_hashes,
        calibration_key=calibration_key,
        calibration=calibration,
        metrics=metrics,
        provenance=provenance,
    )
    require_newest_immutable_event(
        event_path,
        CALIBRATION_EVENT_PREFIX.rstrip("_"),
    )
    return {
        **preview,
        "calibration_evidence_json": str(event_path),
        "calibration_evidence_sha256": sha256_file(event_path),
        "output_artifact_sha256": event["output_bundle"]["artifact_sha256"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
