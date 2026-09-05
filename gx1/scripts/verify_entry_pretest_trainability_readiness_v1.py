#!/usr/bin/env python3
"""Bind the direct V9 pre-TEST lane before candidate readiness.

This is deliberately independent of the retired seq513 rebuild-chain
readiness reports.  A V9 pre-TEST dataset owns an unopened-TEST guard and
directly bound TRAIN/VAL artifacts; manufacturing old chain events for it
would weaken, rather than prove, its lineage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    PretestTechnicalRecipeError,
    require_pretest_technical_recipe_metadata,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    LaunchContractError,
    require_training_recipe_source_provenance,
)
from gx1.contracts.immutable_event_authority_v1 import (
    require_newest_immutable_event,
    write_immutable_json_event,
)
from gx1.contracts.entry_pretest_candidate_launch_gate_v1 import artifact_binding


SCHEMA_VERSION = "entry_pretest_trainability_readiness_v1"
EVENT_PREFIX = "ENTRY_PRETEST_TRAINABILITY_READINESS"
READY_DECISION = "READY_FOR_PRETEST_CANDIDATE_TRAINABILITY_REVIEW"
BLOCKED_DECISION = "BLOCKED_PRETEST_CANDIDATE_TRAINABILITY_READINESS"
PRETRAIN_SCHEMA = "xau_direction_repair_pretrain_audit_v6"
_SHA_LENGTH = 64


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be an absolute regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} root must be an object")
    return payload


def _check(name: str, ok: bool, details: Any = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _recipe(
    path: Path,
    sha256: str,
    *,
    profile: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    binding = artifact_binding(path)
    if binding["sha256"] != sha256:
        raise RuntimeError(f"{label} recipe SHA-256 mismatch")
    try:
        recipe = require_pretest_technical_recipe_metadata(
            _read_json(path, label=f"{label} recipe"), expected_profile=profile
        )
    except PretestTechnicalRecipeError as exc:
        raise RuntimeError(f"{label} recipe contract rejected: {exc}") from exc
    return recipe, binding


def _rehash_recipe_artifacts(recipe: Mapping[str, Any]) -> None:
    artifacts = recipe.get("artifact_bindings")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("recipe artifact bindings are missing")
    for name, declared in artifacts.items():
        if not isinstance(declared, Mapping):
            raise RuntimeError(f"recipe artifact binding is malformed: {name}")
        path = Path(str(declared.get("path") or ""))
        if "test" in path.name.lower():
            raise RuntimeError(f"physical TEST-like recipe artifact is forbidden: {name}")
        observed = artifact_binding(path)
        if observed != {"path": str(path), "sha256": declared.get("sha256")}:
            raise RuntimeError(f"recipe artifact binding changed: {name}")


def _pretrain(
    path: Path,
    *,
    dataset_dir: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = _read_json(path, label="pretrain audit")
    try:
        require_newest_immutable_event(path, "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT")
    except RuntimeError as exc:
        raise RuntimeError("pretrain audit is not the current immutable event") from exc
    if (
        payload.get("schema_version") != PRETRAIN_SCHEMA
        or payload.get("decision") != "PASS"
        or payload.get("failures") != []
        or payload.get("dataset_dir") != dataset_dir
        or tuple(payload.get("data_splits") or ()) != ("train", "val")
        or payload.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE
        or int(payload.get("expected_signal_dim") or 0) != MODEL_NATIVE_SIGNAL_DIM
        or payload.get("large_artifact_hashes_verified") is not True
        or payload.get("require_mandatory_level_features") is not True
        or payload.get("require_inline_seq_structure") is not True
        or payload.get("require_xau_provenance") is not True
    ):
        raise RuntimeError("pretrain audit is not an exact five-year zero-failure PASS")
    return payload, artifact_binding(path)


def _recipe_compatibility(
    candidate: Mapping[str, Any], smoke: Mapping[str, Any]
) -> None:
    for key in ("dataset_run_id", "dataset_dir", "test_guard_lineage", "artifact_bindings"):
        if candidate.get(key) != smoke.get(key):
            raise RuntimeError(f"candidate and smoke recipes differ at {key}")
    if candidate.get("run_id") == smoke.get("run_id"):
        raise RuntimeError("candidate and smoke recipes must use distinct run IDs")
    if candidate.get("out_bundle_dir") == smoke.get("out_bundle_dir"):
        raise RuntimeError("candidate and smoke recipes must use distinct output bundles")


def require_pretest_trainability_readiness(
    path: Path, sha256: str, *, selected_recipe: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate the small direct evidence chain for executable handover.

    This is not a launch grant or a substitute for the trainer's parquet/source
    checks. Never manufacture legacy three-split reports for a pre-TEST dataset.
    """
    if artifact_binding(path) != {"path": str(path), "sha256": sha256}:
        raise RuntimeError("pretest handover readiness hash mismatch")
    require_newest_immutable_event(path, EVENT_PREFIX)
    report = _read_json(path, label="pretest handover readiness")
    bindings = report.get("input_bindings")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or report.get("decision") != READY_DECISION
        or report.get("failures") != []
        or report.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE
        or report.get("sequence_length") != MODEL_NATIVE_SEQ_LEN
        or report.get("expected_signal_dim") != MODEL_NATIVE_SIGNAL_DIM
        or report.get("required_training_specialists") != list(MODEL_NATIVE_REQUIRED_SPECIALISTS)
        or not isinstance(report.get("checks"), list) or not report["checks"]
        or any(not isinstance(row, Mapping) or row.get("ok") is not True
               for row in report["checks"])
        or any(report.get(key) is not False for key in (
            "candidate_training_allowed", "activation_authority",
            "promotion_shadow_live_allowed",
        ))
        or not isinstance(bindings, Mapping)
        or set(bindings) != {"candidate_recipe", "smoke_recipe", "pretrain_audit"}
        or report.get("input_bindings_sha256") != hashlib.sha256(
            json.dumps(bindings, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
    ):
        raise RuntimeError("pretest handover readiness contract invalid")
    for name, binding in bindings.items():
        if (
            not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}
            or artifact_binding(Path(str(binding["path"]))) != dict(binding)
        ):
            raise RuntimeError(f"pretest handover nested binding changed: {name}")
    candidate, _ = _recipe(Path(bindings["candidate_recipe"]["path"]),
                           bindings["candidate_recipe"]["sha256"],
                           profile="candidate", label="candidate")
    smoke, _ = _recipe(Path(bindings["smoke_recipe"]["path"]),
                       bindings["smoke_recipe"]["sha256"],
                       profile="smoke", label="smoke")
    _recipe_compatibility(candidate, smoke)
    profile = selected_recipe.get("profile")
    expected_recipe = smoke if profile == "smoke" else candidate
    if profile not in {"smoke", "candidate"} or dict(selected_recipe) != expected_recipe:
        raise RuntimeError("pretest handover selected recipe mismatch")
    if (
        report.get("dataset_dir") != candidate["dataset_dir"]
        or report.get("dataset_run_id") != candidate["dataset_run_id"]
        or report.get("candidate_run_id") != candidate["run_id"]
        or report.get("smoke_run_id") != smoke["run_id"]
    ):
        raise RuntimeError("pretest handover dataset/run identity mismatch")
    _pretrain(Path(bindings["pretrain_audit"]["path"]),
              dataset_dir=str(candidate["dataset_dir"]))
    for name, declared in expected_recipe["artifact_bindings"].items():
        artifact_path = Path(declared["path"])
        if artifact_path.suffix == ".json" and artifact_binding(artifact_path) != declared:
            raise RuntimeError(f"pretest handover audit/manifest changed: {name}")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = Path(args.candidate_recipe_json).expanduser().resolve(strict=True)
    smoke_path = Path(args.smoke_recipe_json).expanduser().resolve(strict=True)
    pretrain_path = Path(args.pretrain_audit_json).expanduser().resolve(strict=True)
    repo = Path(args.repo_dir).expanduser().resolve(strict=True)
    out_dir = Path(args.out_dir).expanduser().resolve()

    candidate, candidate_binding = _recipe(
        candidate_path,
        str(args.candidate_recipe_sha256),
        profile="candidate",
        label="candidate",
    )
    smoke, smoke_binding = _recipe(
        smoke_path,
        str(args.smoke_recipe_sha256),
        profile="smoke",
        label="smoke",
    )
    if len(str(args.candidate_recipe_sha256)) != _SHA_LENGTH or len(str(args.smoke_recipe_sha256)) != _SHA_LENGTH:
        raise RuntimeError("recipe SHA-256 length is invalid")

    checks: list[dict[str, Any]] = []
    try:
        _recipe_compatibility(candidate, smoke)
        checks.append(_check("candidate and smoke recipes bind one exact pre-TEST dataset", True))
    except RuntimeError as exc:
        checks.append(_check("candidate and smoke recipes bind one exact pre-TEST dataset", False, {"error": str(exc)}))
    try:
        _rehash_recipe_artifacts(candidate)
        _rehash_recipe_artifacts(smoke)
        checks.append(_check("all direct TRAIN/VAL recipe artifacts rehash exactly", True))
    except RuntimeError as exc:
        checks.append(_check("all direct TRAIN/VAL recipe artifacts rehash exactly", False, {"error": str(exc)}))
    try:
        _pretrain_payload, pretrain_binding = _pretrain(
            pretrain_path, dataset_dir=str(candidate["dataset_dir"])
        )
        checks.append(_check("direct pretrain audit is a five-year zero-failure PASS", True))
    except RuntimeError as exc:
        pretrain_binding = artifact_binding(pretrain_path)
        checks.append(_check("direct pretrain audit is a five-year zero-failure PASS", False, {"error": str(exc)}))
    # Dataset compatibility does not imply identical recipe source closures.
    # Revalidate both independently without lifting a runtime review hold.
    for profile, recipe, path, binding in (
        ("candidate", candidate, candidate_path, candidate_binding),
        ("smoke", smoke, smoke_path, smoke_binding),
    ):
        check_name = f"{profile} recipe source closure is current and worktree-clean"
        try:
            require_training_recipe_source_provenance(
                recipe_audit_path=path,
                recipe_audit_sha256=binding["sha256"],
                repo=repo,
                profile=profile,
                run_id=str(recipe["run_id"]),
                dataset_run_id=str(recipe["dataset_run_id"]),
                dataset_dir=Path(str(recipe["dataset_dir"])),
                out_bundle_dir=Path(str(recipe["out_bundle_dir"])),
            )
            checks.append(_check(check_name, True))
        except (LaunchContractError, OSError, RuntimeError, ValueError) as exc:
            checks.append(_check(check_name, False, {"error": str(exc)}))

    failures = [
        {"check": row["name"], "details": row["details"]}
        for row in checks
        if not row["ok"]
    ]
    ready = not failures
    inputs = {
        "candidate_recipe": candidate_binding,
        "smoke_recipe": smoke_binding,
        "pretrain_audit": pretrain_binding,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": READY_DECISION if ready else BLOCKED_DECISION,
        "failures": failures,
        "candidate_training_allowed": False,
        "activation_authority": False,
        "promotion_shadow_live_allowed": False,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "sequence_length": MODEL_NATIVE_SEQ_LEN,
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "required_training_specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
        "dataset_run_id": candidate["dataset_run_id"],
        "dataset_dir": candidate["dataset_dir"],
        "candidate_run_id": candidate["run_id"],
        "smoke_run_id": smoke["run_id"],
        "input_bindings": inputs,
        "input_bindings_sha256": hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
        "checks": checks,
    }
    _path, report = write_immutable_json_event(out_dir, EVENT_PREFIX, report)
    if failures:
        raise SystemExit(1)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-recipe-json", required=True)
    parser.add_argument("--candidate-recipe-sha256", required=True)
    parser.add_argument("--smoke-recipe-json", required=True)
    parser.add_argument("--smoke-recipe-sha256", required=True)
    parser.add_argument("--pretrain-audit-json", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
