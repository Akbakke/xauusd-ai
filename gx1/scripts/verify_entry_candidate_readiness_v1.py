#!/usr/bin/env python3
"""Fail-closed candidate readiness from one exact immutable seq513 smoke proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_SMOKE_SPLITS,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_smoke_bundle_audit_v1 import (
    require_smoke_bundle_audit_contract,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    MODEL_NATIVE_RECIPE_ENV_KEYS,
    RECIPE_AUDIT_SCHEMA,
    TRAIN_WRAPPER_RELATIVE_PATH,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    REQUIRED_POSITIVE_LOSS_WEIGHTS,
    SCHEMA_VERSION as TRAINING_OBJECTIVE_SCHEMA,
    require_training_objective_contract,
)
from gx1.contracts.immutable_event_authority_v1 import (
    require_newest_immutable_event,
    write_immutable_json_event,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    PRE_CALIBRATION_EVIDENCE_STAGE,
    resolve_and_validate_prediction_evidence,
    sha256_file,
)


SCHEMA_VERSION = "entry_candidate_readiness_model_native_v1"
READY_DECISION = "READY_FOR_CANDIDATE_TRAINING"
BLOCKED_DECISION = "NOT_READY_FOR_CANDIDATE_TRAINING"
EVENT_PREFIX = "ENTRY_CANDIDATE_READINESS"
TRAINABILITY_SCHEMA = "entry_model_native_seq513_trainability_readiness_v1"
TRAINABILITY_DECISION = "READY_FOR_MODEL_NATIVE_SEQ513_TRAINABILITY_REVIEW"
SPECIALIST_SCHEMA = "entry_specialist_feature_group_audit_v1"
REQUIRED_MIN_GATE_ENTROPY = 0.05
_EVENT_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<stamp>\d{8}T\d{6}(?:\d{6})?Z)\.json$"
)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"could not read immutable JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"immutable JSON root must be an object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {
        "name": str(name),
        "ok": bool(ok),
        "details": details if details is not None else {},
    }


def _immutable_event(path: Path, *, label: str) -> dict[str, Any]:
    path = path.expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be an absolute regular immutable event: {path}")
    resolved = path.resolve(strict=True)
    if str(resolved) != str(path):
        raise RuntimeError(f"{label} path must already be canonical: {path}")
    match = _EVENT_RE.fullmatch(path.name)
    if match is None or "latest" in path.name.lower():
        raise RuntimeError(f"{label} lacks an exact immutable UTC event stamp: {path}")
    require_newest_immutable_event(path, match.group("prefix"))
    return _read_json(path)


def _artifact_binding(path: Path) -> dict[str, str]:
    raw = path.expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise RuntimeError(f"artifact is not a regular file: {raw}")
    resolved = raw.resolve(strict=True)
    digest = _sha256_file(resolved)
    if _SHA_RE.fullmatch(digest) is None:
        raise RuntimeError(f"artifact has no exact SHA-256: {resolved}")
    return {"path": str(resolved), "sha256": digest}


def _zero_failure(payload: dict[str, Any], *, schema: str, decision: str, label: str) -> None:
    if payload.get("schema_version") != schema:
        raise RuntimeError(f"{label} schema mismatch")
    if payload.get("decision") != decision or payload.get("failures") != []:
        raise RuntimeError(f"{label} is not an explicit zero-failure {decision} event")


def _validate_bound_input_audits(
    smoke_contract: dict[str, Any],
    *,
    specialist_path: Path,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for name, declared in smoke_contract["input_audits"].items():
        path = Path(declared["path"])
        payload = _read_json(path)
        observed = _artifact_binding(path)
        if observed != {"path": declared["path"], "sha256": declared["sha256"]}:
            raise RuntimeError(f"smoke audit {name} input binding changed")
        if (
            payload.get("schema_version") != declared["schema_version"]
            or payload.get("decision") != "PASS"
            or payload.get("failures") != []
        ):
            raise RuntimeError(f"smoke audit {name} input is not an exact zero-failure PASS")
        results[name] = payload
    if Path(smoke_contract["input_audits"]["specialist"]["path"]) != specialist_path:
        raise RuntimeError("explicit specialist audit does not match smoke audit binding")
    return results


def _bundle_file_check(
    smoke_contract: dict[str, Any],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    try:
        observed_bindings = {
            name: _artifact_binding(Path(binding["path"]))
            for name, binding in smoke_contract["bundle_artifacts"].items()
        }
        if observed_bindings != smoke_contract["bundle_artifacts"]:
            raise RuntimeError("bundle artifact hashes changed after smoke audit")
        metadata = _read_json(Path(observed_bindings["bundle_metadata"]["path"]))
        lock = _read_json(Path(observed_bindings["master_transformer_lock"]["path"]))
        if metadata.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
            raise RuntimeError("bundle metadata contract_mode mismatch")
        if lock.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
            raise RuntimeError("bundle lock contract_mode mismatch")
        if (
            int(metadata.get("seq_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM
            or int(metadata.get("snap_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM
            or int(metadata.get("seq_len") or -1) != MODEL_NATIVE_SEQ_LEN
        ):
            raise RuntimeError("bundle metadata sequence dimensions mismatch")
        metadata_signal = metadata.get("model_native_signal_contract")
        lock_signal = lock.get("model_native_signal_contract")
        require_model_native_signal_contract(
            metadata_signal,
            context="CANDIDATE_READINESS_BUNDLE_META",
        )
        require_model_native_signal_contract(
            lock_signal,
            context="CANDIDATE_READINESS_BUNDLE_LOCK",
        )
        if metadata_signal != lock_signal:
            raise RuntimeError("bundle signal contract differs between metadata and lock")
        metadata_objective = require_training_objective_contract(
            metadata.get("model_native_training_objective"),
            context="CANDIDATE_READINESS_BUNDLE_META",
        )
        lock_objective = require_training_objective_contract(
            lock.get("model_native_training_objective"),
            context="CANDIDATE_READINESS_BUNDLE_LOCK",
        )
        expected_objective = smoke_contract["model_native_training_objective"]
        if not (
            metadata_objective == lock_objective == expected_objective
        ):
            raise RuntimeError("bundle training objective proof is split-brain")
        details = {
            "bundle_artifacts": observed_bindings,
            "ordered_fields_sha256": metadata_signal["ordered_fields_sha256"],
            "training_objective_meta_lock_exact": True,
        }
        return _check("bundle files rehash and preserve exact seq513 meta/lock", True, details)
    except Exception as exc:
        details["error"] = str(exc)
        return _check("bundle files rehash and preserve exact seq513 meta/lock", False, details)


def _prediction_evidence_check(
    smoke_contract: dict[str, Any],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    try:
        report_path = Path(smoke_contract["prediction_report_json"])
        if _sha256_file(report_path) != smoke_contract["prediction_report_sha256"]:
            raise RuntimeError("prediction report SHA-256 mismatch")
        evidence = smoke_contract["prediction_evidence"]
        models = evidence.get("models")
        if (
            not isinstance(models, list)
            or len(models) != 1
            or not isinstance(models[0], str)
            or not models[0]
        ):
            raise RuntimeError(
                "prediction evidence must pin exactly one non-empty model name"
            )
        predictions, report, observed = resolve_and_validate_prediction_evidence(
            Path(str(evidence.get("path") or "")),
            expected_sha256=str(evidence.get("sha256") or ""),
            prediction_report_path=report_path,
            bundle_dir=Path(smoke_contract["bundle_dir"]),
            dataset_dir=Path(smoke_contract["dataset_dir"]),
            expected_stage=PRE_CALIBRATION_EVIDENCE_STAGE,
            expected_splits=tuple(FOUNDATION_AUDIT_SMOKE_SPLITS),
            expected_model=models[0],
        )
        if observed != evidence:
            raise RuntimeError("smoke audit prediction declaration is not exact")
        expected_splits = list(FOUNDATION_AUDIT_SMOKE_SPLITS)
        if list(observed.get("splits") or ()) != expected_splits:
            raise RuntimeError(
                "prediction evidence must cover exactly the policy-owned smoke "
                f"splits: {expected_splits}"
            )
        details = {
            "predictions": str(predictions),
            "prediction_report": str(report_path),
            "rows": observed.get("rows"),
            "splits": observed.get("splits"),
            "report_decision": report.get("decision"),
        }
        return _check("immutable prediction evidence rehashes and is model-native", True, details)
    except Exception as exc:
        details["error"] = str(exc)
        return _check("immutable prediction evidence rehashes and is model-native", False, details)


def _trainability_contract_check(payload: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    try:
        _zero_failure(
            payload,
            schema=TRAINABILITY_SCHEMA,
            decision=TRAINABILITY_DECISION,
            label="trainability readiness",
        )
        if payload.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE:
            raise RuntimeError("trainability contract mode mismatch")
        if int(payload.get("expected_signal_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM:
            raise RuntimeError("trainability signal dimension mismatch")
        if tuple(payload.get("required_training_specialists") or ()) != tuple(
            MODEL_NATIVE_REQUIRED_SPECIALISTS
        ):
            raise RuntimeError("trainability specialist set mismatch")
        future = payload.get("future_train_contract")
        if not isinstance(future, dict):
            raise RuntimeError("trainability future train contract missing")
        if (
            future.get("profile") != "smoke"
            or future.get("control_route") != "model-native-smoke-train"
            or future.get("wrapper_path") != TRAIN_WRAPPER_RELATIVE_PATH
            or future.get("recipe_audit_schema") != RECIPE_AUDIT_SCHEMA
            or future.get("training_objective_schema") != TRAINING_OBJECTIVE_SCHEMA
            or set(future.get("recipe_env_keys") or ())
            != set(MODEL_NATIVE_RECIPE_ENV_KEYS)
            or set(future.get("required_positive_loss_weights") or ())
            != set(REQUIRED_POSITIVE_LOSS_WEIGHTS)
        ):
            raise RuntimeError("trainability exact wrapper/objective contract mismatch")
        details = {
            "control_route": future["control_route"],
            "recipe_audit_schema": future["recipe_audit_schema"],
            "training_objective_schema": future["training_objective_schema"],
        }
        return _check("trainability proves exact wrapper and positive objective", True, details)
    except Exception as exc:
        details["error"] = str(exc)
        return _check("trainability proves exact wrapper and positive objective", False, details)


def _specialist_contract_check(payload: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    try:
        _zero_failure(
            payload,
            schema=SPECIALIST_SCHEMA,
            decision="PASS",
            label="specialist audit",
        )
        if payload.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
            raise RuntimeError("specialist audit contract mode mismatch")
        if int(payload.get("signal_field_count") or -1) != MODEL_NATIVE_SIGNAL_DIM:
            raise RuntimeError("specialist audit signal dimension mismatch")
        if int(payload.get("selected_feature_count") or -1) != MODEL_NATIVE_SELECTED_FEATURE_COUNT:
            raise RuntimeError("specialist audit selected feature count mismatch")
        if tuple(payload.get("required_training_specialists") or ()) != tuple(
            MODEL_NATIVE_REQUIRED_SPECIALISTS
        ):
            raise RuntimeError("specialist audit required set mismatch")
        for key in (
            "specialist_model_contract_valid",
            "signal_routing_all_mapped",
            "specialist_input_liveness_all_live",
        ):
            if payload.get(key) is not True:
                raise RuntimeError(f"specialist audit {key} is not proven")
        details = {
            "required_training_specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
            "signal_field_count": MODEL_NATIVE_SIGNAL_DIM,
        }
        return _check("specialist audit proves exact eight live encoders", True, details)
    except Exception as exc:
        details["error"] = str(exc)
        return _check("specialist audit proves exact eight live encoders", False, details)


def _bundle_specialist_model_contract_passes(report: dict[str, Any]) -> bool:
    """Compatibility helper used by replay readiness against the compact audit."""

    try:
        contract = require_smoke_bundle_audit_contract(
            report,
            context="REPLAY_READINESS_BUNDLE_AUDIT",
        )
    except RuntimeError:
        return False
    specialist = contract["specialist_contract"]
    return bool(
        specialist["decision"] == "PASS"
        and specialist["failures"] == []
        and tuple(specialist["specialists"])
        == tuple(MODEL_NATIVE_REQUIRED_SPECIALISTS)
        and specialist["gate_liveness_proven"] is True
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if str(args.edge_test_scope) != "strict":
        raise RuntimeError("candidate readiness edge_test_scope must be exactly strict")
    if int(args.min_active_specialists) != len(MODEL_NATIVE_REQUIRED_SPECIALISTS):
        raise RuntimeError("candidate readiness requires exactly eight active specialists")

    smoke_path = Path(args.smoke_bundle_audit_json).expanduser()
    specialist_path = Path(args.specialist_audit_json).expanduser()
    trainability_path = Path(args.trainability_readiness_json).expanduser()
    expected_dataset = Path(args.expected_smoke_dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    smoke = _immutable_event(smoke_path, label="smoke bundle audit")
    specialist = _immutable_event(specialist_path, label="specialist audit")
    trainability = _immutable_event(trainability_path, label="trainability readiness")

    checks: list[dict[str, Any]] = []
    normalized_smoke: dict[str, Any] = {}
    try:
        normalized_smoke = require_smoke_bundle_audit_contract(
            smoke,
            context="CANDIDATE_READINESS",
        )
        if Path(normalized_smoke["dataset_dir"]).resolve() != expected_dataset:
            raise RuntimeError("smoke audit dataset_dir does not match explicit expected dataset")
        checks.append(_check("smoke audit satisfies exact compact seq513 contract", True))
    except Exception as exc:
        checks.append(
            _check(
                "smoke audit satisfies exact compact seq513 contract",
                False,
                {"error": str(exc)},
            )
        )

    if normalized_smoke:
        try:
            _validate_bound_input_audits(
                normalized_smoke,
                specialist_path=specialist_path,
            )
            checks.append(_check("smoke input audits rehash and remain zero-failure", True))
        except Exception as exc:
            checks.append(
                _check(
                    "smoke input audits rehash and remain zero-failure",
                    False,
                    {"error": str(exc)},
                )
            )
        checks.append(_bundle_file_check(normalized_smoke))
        checks.append(_prediction_evidence_check(normalized_smoke))
    else:
        checks.extend(
            [
                _check("smoke input audits rehash and remain zero-failure", False),
                _check("bundle files rehash and preserve exact seq513 meta/lock", False),
                _check("immutable prediction evidence rehashes and is model-native", False),
            ]
        )
    checks.append(_trainability_contract_check(trainability))
    checks.append(_specialist_contract_check(specialist))

    failures = [
        {"check": check["name"], "details": check.get("details") or {}}
        for check in checks
        if not check["ok"]
    ]
    ready = not failures
    created = datetime.now(timezone.utc)
    input_bindings = {
        "smoke_bundle_audit": _artifact_binding(smoke_path),
        "specialist_audit": _artifact_binding(specialist_path),
        "trainability_readiness": _artifact_binding(trainability_path),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created.isoformat(),
        "decision": READY_DECISION if ready else BLOCKED_DECISION,
        "failures": failures,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "sequence_length": MODEL_NATIVE_SEQ_LEN,
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "expected_smoke_dataset_dir": str(expected_dataset),
        "required_specialist_groups": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "candidate_training_allowed": ready,
        "promotion_shadow_live_allowed": False,
        "activation_authority": False,
        "checks": checks,
        "input_bindings": input_bindings,
        "input_bindings_sha256": _canonical_sha256(input_bindings),
    }
    _, report = write_immutable_json_event(out_dir, EVENT_PREFIX, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-bundle-audit-json", required=True)
    parser.add_argument("--specialist-audit-json", required=True)
    parser.add_argument("--trainability-readiness-json", required=True)
    parser.add_argument("--expected-smoke-dataset-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--edge-test-scope", choices=("strict",), default="strict")
    parser.add_argument(
        "--min-active-specialists",
        type=int,
        default=len(MODEL_NATIVE_REQUIRED_SPECIALISTS),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
