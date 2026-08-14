#!/usr/bin/env python3
"""Bind one completed seq513 rebuild into pre-freeze smoke readiness.

This report-only boundary physically revalidates TRAIN/VAL, the complete
liveness artifact, the pretrain target audit and the on-disk XAU tape lineage.
It never opens, parses, stats or hashes TEST dataset/manifest paths.  TEST may
only cross this boundary as metadata in an exact seal already hash-bound by the
declared rebuild completion authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_dataset_split_artifacts_v1 import (
    ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION,
    require_dataset_split_artifacts,
)
from gx1.contracts.entry_full_input_liveness_v1 import (
    EXPECTED_FIELD_COUNTS,
    SCHEMA_VERSION as LIVENESS_SCHEMA_VERSION,
    validate_full_input_liveness_artifact,
)
from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    DATASET_REBUILD_TERMINAL_DECISION,
    DATASET_REBUILD_TERMINAL_EVENT_PREFIX,
    DATASET_REBUILD_TERMINAL_SCHEMA_VERSION,
    EVENT_PREFIX,
    PREFREEZE_SPLIT_ARTIFACTS_SCHEMA_VERSION,
    PREFREEZE_SPLITS,
    READY_DECISION,
    REQUIRED_PROOF_CHECKS,
    SCHEMA_VERSION,
    SIDE_EFFECT_KEYS,
    TEST_SEAL_ACCESS_POLICY,
    TEST_SEAL_DECISION,
    TEST_SEAL_EVENT_PREFIX,
    TEST_SEAL_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    require_model_native_signal_contract,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_SOURCE_SCHEMA,
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    XAU_INSTRUMENT,
    validate_xau_tape_provenance_v1,
)


SPLITS = PREFREEZE_SPLITS
SPLIT_MANIFEST_SCHEMA = MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION
PREFLIGHT_SCHEMA = "entry_model_native_seq513_rebuild_preflight_v13"
PREFLIGHT_DECISION = "READY_FOR_MODEL_NATIVE_SEQ513_REBUILD"
PRETRAIN_SCHEMA = "xau_direction_repair_pretrain_audit_v4"
CHAIN_SCHEMA = "seq513_rebuild_chain_status_v7"
DIRECT_BUILD_PROOF_FILENAME = "DATASET_BUILD_PROOF.json"
FORBIDDEN_TEST_INPUT_FIELDS = (
    "test_manifest_json",
    "test_manifest_sha256",
    "test_parquet",
    "test_parquet_sha256",
)
_SHA256_HEX = frozenset("0123456789abcdef")
_TEST_SEAL_AUTHORITY_KEYS = frozenset({"path", "sha256"})
_TEST_SEAL_ARTIFACT_KEYS = frozenset({"path", "sha256"})
_TEST_SEAL_MANIFEST_CONTRACT_KEYS = frozenset(
    {"schema_version", "manifest_variant", "entry_run_id", "instrument"}
)
_TEST_SEAL_KEYS = frozenset(
    {
        "schema_version",
        "decision",
        "created_utc",
        "seal_path",
        "entry_run_id",
        "dataset_dir",
        "split",
        "access_policy",
        "disclosure_count",
        "manifest",
        "parquet",
        "manifest_contract",
        "rows",
        "pair_lineage",
        "pair_lineage_sha256",
        "source_lineage",
        "source_lineage_sha256",
        "rebuild_terminal",
        "content_binding_sha256",
    }
)
_REBUILD_TERMINAL_AUTHORITY_KEYS = frozenset({"path", "sha256"})
_REBUILD_TERMINAL_BINDING_KEYS = frozenset(
    {
        "path",
        "sha256",
        "schema_version",
        "decision",
        "content_binding_sha256",
    }
)
_REBUILD_TERMINAL_KEYS = frozenset(
    {
        "schema_version",
        "decision",
        "created_utc",
        "entry_run_id",
        "dataset_dir",
        "dataset_stem",
        "terminal_event_path",
        "split_artifacts",
        "pair_lineage",
        "pair_lineage_sha256",
        "source_lineage",
        "source_lineage_sha256",
        "content_binding_sha256",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    observed = str(value or "").strip().lower()
    if len(observed) != 64 or any(
        character not in _SHA256_HEX for character in observed
    ):
        raise RuntimeError(f"{label}: expected exact SHA-256")
    return observed


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_test_inputs(args: argparse.Namespace) -> None:
    supplied = [
        field
        for field in FORBIDDEN_TEST_INPUT_FIELDS
        if getattr(args, field, None) not in (None, "")
    ]
    if supplied:
        raise RuntimeError(
            "pre-freeze post-rebuild readiness forbids caller-supplied TEST "
            f"artifacts: {supplied}"
        )


def _json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label}: missing regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{label}: invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}: JSON root must be an object: {path}")
    return value


def _path(raw: str, *, label: str, directory: bool = False) -> Path:
    path = Path(str(raw or "")).expanduser()
    valid = path.is_dir() if directory else path.is_file()
    if (
        not path.is_absolute()
        or path.resolve() != path
        or path.is_symlink()
        or not valid
        or any("latest" in part.lower() for part in path.parts)
    ):
        raise RuntimeError(f"{label}: invalid immutable path: {path}")
    return path


def _expected_opaque_test_paths(
    *,
    dataset_dir: Path,
    prefreeze_artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, Path]:
    train_manifest = Path(str(prefreeze_artifacts["train"]["manifest_path"]))
    train_parquet = Path(str(prefreeze_artifacts["train"]["parquet_path"]))
    manifest_suffix = "_train.manifest.json"
    parquet_suffix = "_train.parquet"
    if (
        train_manifest.parent != dataset_dir
        or train_parquet.parent != dataset_dir
        or not train_manifest.name.endswith(manifest_suffix)
        or not train_parquet.name.endswith(parquet_suffix)
    ):
        raise RuntimeError("TRAIN split cannot establish one exact dataset stem")
    manifest_stem = train_manifest.name[: -len(manifest_suffix)]
    parquet_stem = train_parquet.name[: -len(parquet_suffix)]
    if not manifest_stem or manifest_stem != parquet_stem:
        raise RuntimeError("TRAIN manifest/parquet stems differ")
    for split in PREFREEZE_SPLITS:
        expected_manifest = dataset_dir / f"{manifest_stem}_{split}.manifest.json"
        expected_parquet = dataset_dir / f"{manifest_stem}_{split}.parquet"
        if (
            Path(str(prefreeze_artifacts[split]["manifest_path"]))
            != expected_manifest
            or Path(str(prefreeze_artifacts[split]["parquet_path"]))
            != expected_parquet
        ):
            raise RuntimeError(f"{split}: split stem differs from TRAIN")
    return (
        dataset_dir / f"{manifest_stem}_test.manifest.json",
        dataset_dir / f"{manifest_stem}_test.parquet",
    )


def _opaque_test_artifact(
    raw: Any,
    *,
    expected_path: Path,
    label: str,
) -> dict[str, str]:
    if not isinstance(raw, Mapping) or frozenset(raw) != _TEST_SEAL_ARTIFACT_KEYS:
        raise RuntimeError(f"{label}: seal artifact keys are not exact")
    # Deliberately lexical only: no resolve/stat/open operation may touch TEST.
    observed_path = Path(str(raw.get("path") or ""))
    if (
        not observed_path.is_absolute()
        or observed_path != expected_path
        or any("latest" in part.lower() for part in observed_path.parts)
    ):
        raise RuntimeError(f"{label}: opaque TEST path binding mismatch")
    return {
        "path": str(observed_path),
        "sha256": _sha256(raw.get("sha256"), label=f"{label}.sha256"),
    }


def _completion_bound_test_seal(
    *,
    completion_payload: Mapping[str, Any],
    supplied_seal_path: str,
    supplied_seal_sha256: str,
    event_root: Path,
    dataset_dir: Path,
    expected_run_id: str,
    prefreeze_artifacts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    authority = completion_payload.get("prefreeze_test_seal")
    if (
        not isinstance(authority, Mapping)
        or frozenset(authority) != _TEST_SEAL_AUTHORITY_KEYS
    ):
        raise RuntimeError(
            "rebuild completion lacks one exact content-bound pre-freeze TEST seal"
        )
    raw_seal_path = Path(str(authority.get("path") or ""))
    requested_seal_path = Path(str(supplied_seal_path or ""))
    requested_seal_sha = _sha256(
        supplied_seal_sha256,
        label="supplied TEST seal sha256",
    )
    if (
        not raw_seal_path.is_absolute()
        or requested_seal_path != raw_seal_path
        or event_root not in raw_seal_path.parents
        or dataset_dir == raw_seal_path.parent
        or dataset_dir in raw_seal_path.parents
        or not raw_seal_path.name.startswith(f"{TEST_SEAL_EVENT_PREFIX}_")
        or not raw_seal_path.name.endswith(".json")
        or any("latest" in part.lower() for part in raw_seal_path.parts)
    ):
        raise RuntimeError("TEST seal is not an event-owned non-dataset artifact")
    seal_path = _path(str(raw_seal_path), label="TEST seal")
    authority_sha = _sha256(
        authority.get("sha256"),
        label="TEST seal authority sha256",
    )
    if authority_sha != requested_seal_sha or _sha256_file(seal_path) != authority_sha:
        raise RuntimeError("TEST seal differs from rebuild completion binding")
    seal = _json(seal_path, label="TEST seal")
    if frozenset(seal) != _TEST_SEAL_KEYS:
        raise RuntimeError("TEST seal keys are not exact")
    if (
        seal.get("schema_version") != TEST_SEAL_SCHEMA_VERSION
        or seal.get("decision") != TEST_SEAL_DECISION
        or seal.get("seal_path") != str(seal_path)
        or seal.get("entry_run_id") != expected_run_id
        or seal.get("dataset_dir") != str(dataset_dir)
        or seal.get("split") != "test"
        or seal.get("access_policy") != TEST_SEAL_ACCESS_POLICY
        or seal.get("disclosure_count") != 0
    ):
        raise RuntimeError("TEST seal identity/access contract mismatch")

    expected_manifest_path, expected_parquet_path = _expected_opaque_test_paths(
        dataset_dir=dataset_dir,
        prefreeze_artifacts=prefreeze_artifacts,
    )
    manifest = _opaque_test_artifact(
        seal.get("manifest"),
        expected_path=expected_manifest_path,
        label="TEST manifest seal",
    )
    parquet = _opaque_test_artifact(
        seal.get("parquet"),
        expected_path=expected_parquet_path,
        label="TEST parquet seal",
    )
    manifest_contract = seal.get("manifest_contract")
    if (
        not isinstance(manifest_contract, Mapping)
        or frozenset(manifest_contract) != _TEST_SEAL_MANIFEST_CONTRACT_KEYS
        or manifest_contract.get("schema_version") != SPLIT_MANIFEST_SCHEMA
        or manifest_contract.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE
        or manifest_contract.get("entry_run_id") != expected_run_id
        or manifest_contract.get("instrument") != XAU_INSTRUMENT
    ):
        raise RuntimeError("TEST seal manifest contract mismatch")
    rows = seal.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise RuntimeError("TEST seal row count must be a positive integer")
    bound_content = {
        key: seal[key]
        for key in sorted(_TEST_SEAL_KEYS - {"content_binding_sha256"})
    }
    content_binding_sha256 = _sha256(
        seal.get("content_binding_sha256"),
        label="TEST seal content binding sha256",
    )
    if _canonical_json_sha256(bound_content) != content_binding_sha256:
        raise RuntimeError("TEST seal canonical content binding mismatch")

    pair_lineage = seal.get("pair_lineage")
    source_lineage = seal.get("source_lineage")
    if (
        not isinstance(pair_lineage, Mapping)
        or not isinstance(source_lineage, Mapping)
        or _canonical_json_sha256(pair_lineage)
        != _sha256(
            seal.get("pair_lineage_sha256"),
            label="TEST seal pair lineage sha256",
        )
        or _canonical_json_sha256(source_lineage)
        != _sha256(
            seal.get("source_lineage_sha256"),
            label="TEST seal source lineage sha256",
        )
    ):
        raise RuntimeError("TEST seal pair/source lineage binding mismatch")

    rebuild_binding = seal.get("rebuild_terminal")
    completion_rebuild = completion_payload.get("dataset_rebuild_terminal")
    if (
        not isinstance(rebuild_binding, Mapping)
        or frozenset(rebuild_binding) != _REBUILD_TERMINAL_BINDING_KEYS
        or not isinstance(completion_rebuild, Mapping)
        or frozenset(completion_rebuild) != _REBUILD_TERMINAL_AUTHORITY_KEYS
        or completion_rebuild.get("path") != rebuild_binding.get("path")
        or completion_rebuild.get("sha256") != rebuild_binding.get("sha256")
    ):
        raise RuntimeError("TEST seal rebuild-terminal authority mismatch")
    raw_rebuild_path = Path(str(rebuild_binding.get("path") or ""))
    if (
        not raw_rebuild_path.is_absolute()
        or event_root not in raw_rebuild_path.parents
        or dataset_dir == raw_rebuild_path.parent
        or dataset_dir in raw_rebuild_path.parents
        or not raw_rebuild_path.name.startswith(
            f"{DATASET_REBUILD_TERMINAL_EVENT_PREFIX}_"
        )
        or not raw_rebuild_path.name.endswith(".json")
        or any("latest" in part.lower() for part in raw_rebuild_path.parts)
    ):
        raise RuntimeError("dataset rebuild terminal is not event-owned")
    rebuild_path = _path(str(raw_rebuild_path), label="dataset rebuild terminal")
    rebuild_sha = _sha256(
        rebuild_binding.get("sha256"),
        label="dataset rebuild terminal sha256",
    )
    if _sha256_file(rebuild_path) != rebuild_sha:
        raise RuntimeError("dataset rebuild terminal hash mismatch")
    rebuild = _json(rebuild_path, label="dataset rebuild terminal")
    if frozenset(rebuild) != _REBUILD_TERMINAL_KEYS:
        raise RuntimeError("dataset rebuild terminal keys are not exact")
    rebuild_content = dict(rebuild)
    rebuild_content_sha = _sha256(
        rebuild_content.pop("content_binding_sha256", None),
        label="dataset rebuild terminal content binding sha256",
    )
    if (
        rebuild.get("schema_version") != DATASET_REBUILD_TERMINAL_SCHEMA_VERSION
        or rebuild.get("decision") != DATASET_REBUILD_TERMINAL_DECISION
        or rebuild.get("entry_run_id") != expected_run_id
        or rebuild.get("dataset_dir") != str(dataset_dir)
        or expected_manifest_path.name
        != f"{rebuild.get('dataset_stem')}_test.manifest.json"
        or expected_parquet_path.name
        != f"{rebuild.get('dataset_stem')}_test.parquet"
        or rebuild.get("terminal_event_path") != str(rebuild_path)
        or rebuild_binding.get("schema_version")
        != DATASET_REBUILD_TERMINAL_SCHEMA_VERSION
        or rebuild_binding.get("decision") != DATASET_REBUILD_TERMINAL_DECISION
        or rebuild_binding.get("content_binding_sha256") != rebuild_content_sha
        or _canonical_json_sha256(rebuild_content) != rebuild_content_sha
        or rebuild.get("pair_lineage") != pair_lineage
        or rebuild.get("source_lineage") != source_lineage
        or rebuild.get("pair_lineage_sha256")
        != seal.get("pair_lineage_sha256")
        or rebuild.get("source_lineage_sha256")
        != seal.get("source_lineage_sha256")
    ):
        raise RuntimeError("dataset rebuild terminal content/lineage mismatch")
    rebuild_splits = rebuild.get("split_artifacts")
    if not isinstance(rebuild_splits, Mapping) or set(rebuild_splits) != {
        "train",
        "val",
        "test",
    }:
        raise RuntimeError("dataset rebuild terminal split set is not exact")
    for split in PREFREEZE_SPLITS:
        expected = prefreeze_artifacts[split]
        observed = rebuild_splits.get(split)
        if not isinstance(observed, Mapping) or any(
            observed.get(key) != expected.get(key)
            for key in (
                "manifest_path",
                "manifest_sha256",
                "parquet_path",
                "parquet_sha256",
                "schema_version",
                "manifest_variant",
                "rows",
                "entry_run_id",
                "instrument",
            )
        ):
            raise RuntimeError(f"dataset rebuild terminal {split} binding mismatch")
    rebuilt_test = rebuild_splits.get("test")
    if not isinstance(rebuilt_test, Mapping) or any(
        rebuilt_test.get(key) != expected
        for key, expected in {
            "manifest_path": manifest["path"],
            "manifest_sha256": manifest["sha256"],
            "parquet_path": parquet["path"],
            "parquet_sha256": parquet["sha256"],
            "schema_version": manifest_contract["schema_version"],
            "manifest_variant": manifest_contract["manifest_variant"],
            "rows": rows,
            "entry_run_id": expected_run_id,
            "instrument": manifest_contract["instrument"],
        }.items()
    ):
        raise RuntimeError("dataset rebuild terminal TEST binding mismatch")

    test_artifact = {
        "manifest_path": manifest["path"],
        "manifest_sha256": manifest["sha256"],
        "parquet_path": parquet["path"],
        "parquet_sha256": parquet["sha256"],
        "schema_version": manifest_contract["schema_version"],
        "manifest_variant": manifest_contract["manifest_variant"],
        "rows": rows,
        "entry_run_id": expected_run_id,
        "instrument": manifest_contract["instrument"],
        "access_mode": TEST_SEAL_ACCESS_POLICY,
        "seal_path": str(seal_path),
        "seal_sha256": authority_sha,
    }
    details = {
        "schema_version": TEST_SEAL_SCHEMA_VERSION,
        "decision": TEST_SEAL_DECISION,
        "access_policy": TEST_SEAL_ACCESS_POLICY,
        "authority": {"path": str(seal_path), "sha256": authority_sha},
        "content_binding_sha256": content_binding_sha256,
        "rebuild_terminal": {
            "path": str(rebuild_path),
            "sha256": rebuild_sha,
            "schema_version": DATASET_REBUILD_TERMINAL_SCHEMA_VERSION,
            "content_binding_sha256": rebuild_content_sha,
        },
        "pair_lineage_sha256": seal["pair_lineage_sha256"],
        "source_lineage_sha256": seal["source_lineage_sha256"],
        "disclosure_count": 0,
        "test_dataset_bytes_read": False,
        "test_manifest_bytes_read": False,
        "test_paths_resolved_or_statted": False,
    }
    return test_artifact, details


def _artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "decision": payload.get("decision"),
    }


def _check(name: str, ok: bool, details: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details}


def _git_identity(repo_dir: Path) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if head.returncode != 0 or status.returncode != 0:
        raise RuntimeError("cannot establish post-rebuild producer Git identity")
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    if dirty:
        raise RuntimeError(
            f"post-rebuild readiness requires a clean worktree: {dirty[:20]}"
        )
    observed_head = head.stdout.strip().lower()
    if len(observed_head) != 40:
        raise RuntimeError("invalid post-rebuild producer Git head")
    return {"repo_dir": str(repo_dir), "head": observed_head, "status_short": []}


def _manifest_contract(
    *,
    split: str,
    binding: dict[str, str],
    expected_run_id: str,
    expected_xau_provenance: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    manifest_path = Path(binding["manifest_path"])
    manifest = _json(manifest_path, label=f"{split} manifest")
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    feature = (
        manifest.get("feature_contract")
        if isinstance(manifest.get("feature_contract"), dict)
        else {}
    )
    signal_contract = (
        extra.get("model_native_signal_contract")
        if isinstance(extra.get("model_native_signal_contract"), dict)
        else {}
    )
    failures: list[str] = []
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
        failures.append(f"{split}: split manifest schema mismatch")
    if manifest.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE:
        failures.append(f"{split}: contract mode mismatch")
    if manifest.get("output_data_path") != binding["parquet_path"]:
        failures.append(f"{split}: parquet self-path mismatch")
    if manifest.get("expected_seq_snap_width") != MODEL_NATIVE_SIGNAL_DIM:
        failures.append(f"{split}: expected signal width mismatch")
    signal_fields = feature.get("signal_bridge_fields")
    if (
        not isinstance(signal_fields, list)
        or len(signal_fields) != MODEL_NATIVE_SIGNAL_DIM
    ):
        failures.append(f"{split}: signal dimension mismatch")
    try:
        require_model_native_signal_contract(
            signal_contract,
            context=f"post-rebuild {split} split manifest",
        )
    except Exception as exc:
        failures.append(f"{split}: model-native signal contract invalid: {exc}")
    if signal_fields != signal_contract.get("fields"):
        failures.append(f"{split}: feature/signal field order mismatch")
    if feature.get("ctx_cont_dim") != EXPECTED_FIELD_COUNTS["ctx_cont"]:
        failures.append(f"{split}: continuous context dimension mismatch")
    if feature.get("ctx_cat_dim") != EXPECTED_FIELD_COUNTS["ctx_cat"]:
        failures.append(f"{split}: categorical context dimension mismatch")
    if extra.get("entry_run_id") != expected_run_id:
        failures.append(f"{split}: run id mismatch")
    if extra.get("xau_tape_provenance") != expected_xau_provenance:
        failures.append(f"{split}: XAU tape provenance mismatch")
    return {
        **binding,
        "schema_version": manifest.get("schema_version"),
        "manifest_variant": manifest.get("manifest_variant"),
        "rows": int(extra.get("rows") or 0),
        "entry_run_id": extra.get("entry_run_id"),
        "instrument": (extra.get("xau_tape_provenance") or {}).get("instrument"),
    }, failures


def run(args: argparse.Namespace) -> dict[str, Any]:
    _reject_test_inputs(args)
    run_id = str(args.run_id or "").strip()
    if not run_id:
        raise RuntimeError("entry run id is required")
    event_root = _path(args.event_root, label="event root", directory=True)
    repo_dir = _path(args.repo_dir, label="repository", directory=True)
    producer_git = _git_identity(repo_dir)
    dataset_dir = _path(args.dataset_dir, label="dataset dir", directory=True)
    smoke_dataset_dir = _path(
        args.smoke_dataset_dir,
        label="smoke dataset dir",
        directory=True,
    )
    if dataset_dir != smoke_dataset_dir:
        raise RuntimeError(
            "separate smoke dataset is forbidden; smoke must consume the exact rebuild splits"
        )
    if dataset_dir.parent != event_root:
        raise RuntimeError("dataset directory is not owned by the declared event root")

    terminal_path = _path(args.chain_terminal_json, label="chain terminal")
    preflight_path = _path(args.rebuild_preflight_json, label="rebuild preflight")
    liveness_path = _path(args.full_input_liveness_json, label="full-input liveness")
    pretrain_path = _path(args.pretrain_audit_json, label="pretrain audit")
    for label, path in (
        ("chain terminal", terminal_path),
        ("rebuild preflight", preflight_path),
        ("full-input liveness", liveness_path),
        ("pretrain audit", pretrain_path),
    ):
        if event_root not in path.parents:
            raise RuntimeError(f"{label} is outside the declared event root: {path}")

    terminal = _json(terminal_path, label="chain terminal")
    preflight = _json(preflight_path, label="rebuild preflight")
    liveness = _json(liveness_path, label="full-input liveness")
    pretrain = _json(pretrain_path, label="pretrain audit")

    supplied_bindings = {
        split: {
            "manifest_path": str(
                _path(getattr(args, f"{split}_manifest_json"), label=f"{split} manifest")
            ),
            "manifest_sha256": str(getattr(args, f"{split}_manifest_sha256")),
            "parquet_sha256": str(getattr(args, f"{split}_parquet_sha256")),
        }
        for split in SPLITS
    }
    resolved_splits = require_dataset_split_artifacts(
        dataset_dir,
        supplied_bindings,
        expected_splits=SPLITS,
        context="MODEL_NATIVE_POST_REBUILD_SPLITS",
    )
    for split in SPLITS:
        supplied_parquet = _path(
            getattr(args, f"{split}_parquet"),
            label=f"{split} parquet",
        )
        if str(supplied_parquet) != resolved_splits[split]["parquet_path"]:
            raise RuntimeError(f"{split}: explicit parquet differs from manifest binding")

    first_manifest = _json(
        Path(resolved_splits["train"]["manifest_path"]),
        label="train manifest",
    )
    first_extra = (
        first_manifest.get("extra")
        if isinstance(first_manifest.get("extra"), dict)
        else {}
    )
    declared_xau = first_extra.get("xau_tape_provenance")
    if not isinstance(declared_xau, dict):
        raise RuntimeError("train manifest lacks XAU tape provenance")
    xau_provenance = validate_xau_tape_provenance_v1(
        declared_xau.get("tape_root"),
        expected_run_id=run_id,
        require_current=True,
    )
    if declared_xau != xau_provenance:
        raise RuntimeError("train manifest XAU binding differs from disk revalidation")

    split_artifacts: dict[str, dict[str, Any]] = {}
    split_failures: list[str] = []
    for split in SPLITS:
        split_artifacts[split], failures = _manifest_contract(
            split=split,
            binding=resolved_splits[split],
            expected_run_id=run_id,
            expected_xau_provenance=xau_provenance,
        )
        split_failures.extend(failures)

    try:
        test_artifact, test_seal_details = _completion_bound_test_seal(
            completion_payload=terminal,
            supplied_seal_path=args.test_seal_json,
            supplied_seal_sha256=args.test_seal_sha256,
            event_root=event_root,
            dataset_dir=dataset_dir,
            expected_run_id=run_id,
            prefreeze_artifacts=split_artifacts,
        )
    except RuntimeError as exc:
        test_artifact = None
        test_seal_ok = False
        test_seal_details = {
            "schema_version": TEST_SEAL_SCHEMA_VERSION,
            "decision": "REJECTED",
            "access_policy": TEST_SEAL_ACCESS_POLICY,
            "error": str(exc),
            "disclosure_count": 0,
            "test_dataset_bytes_read": False,
            "test_manifest_bytes_read": False,
            "test_paths_resolved_or_statted": False,
        }
    else:
        test_seal_ok = True

    liveness_validation = validate_full_input_liveness_artifact(
        liveness_path,
        expected_sha256=_sha256_file(liveness_path),
        expected_dataset_dir=dataset_dir,
        expected_contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        expected_manifest_bindings={
            split: {
                "path": split_artifacts[split]["manifest_path"],
                "sha256": split_artifacts[split]["manifest_sha256"],
            }
            for split in SPLITS
        },
    )

    standalone_build_proof_supplied = (
        terminal_path.name == DIRECT_BUILD_PROOF_FILENAME
    )
    chain_terminal_ok = (
        not standalone_build_proof_supplied
        and terminal.get("schema_version") == CHAIN_SCHEMA
        and terminal.get("state") == "GREEN"
        and terminal.get("step") == "chain-complete"
        and terminal.get("exit_code") == 0
        and terminal.get("entry_run_id") == run_id
        and Path(str(terminal.get("event_root") or "")).resolve() == event_root
        and Path(str(terminal.get("terminal_event_path") or "")).resolve()
        == terminal_path
    )
    terminal_ok = chain_terminal_ok
    completion_mode = "seq513_rebuild_chain_v7" if chain_terminal_ok else "invalid"
    terminal_preflight = (
        terminal.get("preflight")
        if isinstance(terminal.get("preflight"), dict)
        else {}
    )
    preflight_common_ok = (
        preflight.get("schema_version") == PREFLIGHT_SCHEMA
        and preflight.get("decision") == PREFLIGHT_DECISION
        and preflight.get("entry_run_id") == run_id
        and preflight.get("training_allowed") is False
    )
    preflight_ok = preflight_common_ok and (
        terminal_preflight.get("json_path") == str(preflight_path)
        and terminal_preflight.get("sha256") == _sha256_file(preflight_path)
    )
    liveness_ok = (
        liveness.get("schema_version") == LIVENESS_SCHEMA_VERSION
        and liveness.get("decision") == "PASS"
        and not liveness.get("failures")
        and bool(liveness_validation.get("ok"))
    )
    pretrain_tape = (
        pretrain.get("tape_provenance")
        if isinstance(pretrain.get("tape_provenance"), dict)
        else {}
    )
    pretrain_ok = (
        pretrain.get("schema_version") == PRETRAIN_SCHEMA
        and pretrain.get("decision") == "PASS"
        and not pretrain.get("failures")
        and pretrain.get("require_xau_provenance") is True
        and tuple(pretrain.get("data_splits") or ()) == PREFREEZE_SPLITS
        and set(pretrain_tape) == set(PREFREEZE_SPLITS)
        and all(
            pretrain_tape.get(split) == xau_provenance
            for split in PREFREEZE_SPLITS
        )
    )
    provenance_schema = xau_provenance.get("schema_version")
    tape_identity_ok = (
        provenance_schema in {
        CANONICAL_NATIVE_SOURCE_SCHEMA,
        CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
        }
        and xau_provenance.get("instrument") == XAU_INSTRUMENT
    )
    splits_ok = (
        not split_failures
        and tape_identity_ok
        and all(
            split_artifacts[split]["rows"] > 0
            for split in PREFREEZE_SPLITS
        )
    )

    checks = [
        _check(
            REQUIRED_PROOF_CHECKS[0],
            terminal_ok,
            {
                "completion_mode": completion_mode,
                "chain_terminal": chain_terminal_ok,
                "standalone_build_proof_allowed": False,
                "standalone_build_proof_supplied": (
                    standalone_build_proof_supplied
                ),
                "artifact": _artifact(terminal_path, terminal),
            },
        ),
        _check(REQUIRED_PROOF_CHECKS[1], preflight_ok, _artifact(preflight_path, preflight)),
        _check(REQUIRED_PROOF_CHECKS[2], liveness_ok, liveness_validation),
        _check(REQUIRED_PROOF_CHECKS[3], pretrain_ok, _artifact(pretrain_path, pretrain)),
        _check(
            REQUIRED_PROOF_CHECKS[4],
            splits_ok,
            {"failures": split_failures, "splits": split_artifacts},
        ),
        _check(
            REQUIRED_PROOF_CHECKS[5],
            test_seal_ok,
            test_seal_details,
        ),
    ]
    failures = [row for row in checks if not row["ok"]]
    side_effects = {key: False for key in SIDE_EFFECT_KEYS}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": READY_DECISION if not failures else "BLOCKED_MODEL_NATIVE_SEQ513_POST_REBUILD",
        "entry_run_id": run_id,
        "event_root": str(event_root),
        "producer_git": producer_git,
        "source_git_head": terminal.get("git_head") or first_manifest.get("git_commit"),
        "dataset_dir": str(dataset_dir),
        "smoke_dataset_dir": str(smoke_dataset_dir),
        "report_only": True,
        "full_input_liveness_contract": {
            **_artifact(liveness_path, liveness),
            "field_order_sha256": liveness.get("field_order_sha256"),
            "field_counts": liveness.get("expected_field_counts"),
            "atr_ood_status": (liveness.get("atr_ood_drift") or {}).get("status"),
        },
        "pretrain_audit": _artifact(pretrain_path, pretrain),
        "chain_terminal": _artifact(terminal_path, terminal),
        "rebuild_completion_mode": completion_mode,
        "rebuild_preflight": _artifact(preflight_path, preflight),
        "split_artifacts_schema_version": PREFREEZE_SPLIT_ARTIFACTS_SCHEMA_VERSION,
        "prefreeze_physical_split_contract_schema_version": (
            ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION
        ),
        "split_artifacts": {
            **split_artifacts,
            **({"test": test_artifact} if test_artifact is not None else {}),
        },
        "test_isolation": test_seal_details,
        "xau_tape_provenance": xau_provenance,
        "post_rebuild_refresh_command_contract": {
            "smoke_dataset_dir": str(smoke_dataset_dir),
            "all_commands_avoid_training_replay_iql_shadow_live": True,
        },
        "checks": checks,
        "failures": failures,
        "side_effects_started": side_effects,
        "training_allowed": False,
        "replay_allowed": False,
        "shadow_live_allowed": False,
    }
    out_dir = Path(args.out_dir).expanduser().resolve()
    path, payload = write_immutable_json_event(out_dir, EVENT_PREFIX, payload)
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": payload["decision"],
                    "entry_run_id": run_id,
                    "split_rows": {
                        split: row["rows"]
                        for split, row in payload["split_artifacts"].items()
                    },
                    "json_path": str(path),
                    "failures": failures,
                },
                indent=2,
                sort_keys=True,
            )
        )
    if failures:
        raise SystemExit(2)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--event-root", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--chain-terminal-json", required=True)
    parser.add_argument("--test-seal-json", required=True)
    parser.add_argument("--test-seal-sha256", required=True)
    parser.add_argument("--rebuild-preflight-json", required=True)
    parser.add_argument("--full-input-liveness-json", required=True)
    parser.add_argument("--pretrain-audit-json", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--smoke-dataset-dir", required=True)
    for split in SPLITS:
        parser.add_argument(f"--{split}-manifest-json", required=True)
        parser.add_argument(f"--{split}-manifest-sha256", required=True)
        parser.add_argument(f"--{split}-parquet", required=True)
        parser.add_argument(f"--{split}-parquet-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
