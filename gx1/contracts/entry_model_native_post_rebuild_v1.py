"""Exact pre-freeze post-rebuild admission for model-native seq513 data.

Only the exact immutable rebuild-chain terminal is completion authority; a
standalone build proof is not an alternate admission route.  TRAIN and VAL are
physically revalidated.  TEST remains opaque and may only be represented by an
exact terminal-bound seal whose own bytes do not contain dataset content.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.contracts.xau_tape_provenance_v1 import XAU_INSTRUMENT


SCHEMA_VERSION = "entry_model_native_seq513_post_rebuild_readiness_v2"
READY_DECISION = "READY_FOR_MODEL_NATIVE_SEQ513_POST_REBUILD_REVIEW"
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_POST_REBUILD_READINESS"

PREFREEZE_SPLITS = ("train", "val")
TEST_SEAL_SCHEMA_VERSION = "entry_model_native_seq513_untouched_test_seal_v2"
TEST_SEAL_DECISION = "SEALED_UNTOUCHED_TEST"
TEST_SEAL_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_UNTOUCHED_TEST_SEAL"
TEST_SEAL_ACCESS_POLICY = (
    "pre_freeze_metadata_only_no_test_dataset_or_manifest_byte_access"
)
DATASET_REBUILD_TERMINAL_SCHEMA_VERSION = (
    "entry_model_native_seq513_dataset_rebuild_terminal_v1"
)
DATASET_REBUILD_TERMINAL_DECISION = "COMPLETED_MODEL_NATIVE_SEQ513_DATASET_REBUILD"
DATASET_REBUILD_TERMINAL_EVENT_PREFIX = (
    "ENTRY_MODEL_NATIVE_SEQ513_DATASET_REBUILD_TERMINAL"
)
PREFREEZE_SPLIT_ARTIFACTS_SCHEMA_VERSION = (
    "entry_model_native_prefreeze_split_artifacts_v2"
)
PREFREEZE_TEST_SEAL_LINEAGE_SCHEMA_VERSION = (
    "entry_model_native_prefreeze_test_seal_lineage_v1"
)
PREFREEZE_TEST_SEAL_VERIFICATION_MODE = (
    "seal_event_bytes_only_no_test_artifact_or_metrics_access"
)
# A pre-TEST dataset deliberately has no physical TEST parquet or manifest.
# It must therefore never be forced to invent a legacy TEST seal simply to
# start an otherwise TRAIN/VAL-only technical run.  This guard is a separate,
# intentionally narrow control-plane contract: it binds only already-approved
# TRAIN/VAL metadata and proves that TEST was neither materialized nor read.
# The legacy physical seal remains mandatory for every TEST evaluation path.
PRETEST_TEST_GUARD_SCHEMA_VERSION = "entry_model_native_pretest_test_guard_v1"
PRETEST_TEST_GUARD_DECISION = "SEALED_PRETEST_TEST_UNMATERIALIZED"
PRETEST_TEST_GUARD_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_PRETEST_TEST_GUARD"
PRETEST_TEST_GUARD_ACCESS_POLICY = (
    "pretest_guard_no_test_artifact_or_metrics_access"
)
PRETEST_TEST_GUARD_LINEAGE_SCHEMA_VERSION = (
    "entry_model_native_pretest_test_guard_lineage_v1"
)
PRETEST_TEST_GUARD_VERIFICATION_MODE = (
    "guard_event_bytes_only_no_test_artifact_or_metrics_access"
)

_SHA256_HEX = frozenset("0123456789abcdef")
_SEAL_KEYS = frozenset(
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
_ARTIFACT_KEYS = frozenset({"path", "sha256"})
_MANIFEST_CONTRACT_KEYS = frozenset(
    {"schema_version", "manifest_variant", "entry_run_id", "instrument"}
)
_REBUILD_TERMINAL_KEYS = frozenset(
    {
        "path",
        "sha256",
        "schema_version",
        "decision",
        "content_binding_sha256",
    }
)
_LINEAGE_KEYS = frozenset(
    {
        "schema_version",
        "verification_mode",
        "seal_event",
        "dataset_run_id",
        "dataset_dir",
        "split",
        "access_policy",
        "disclosure_count",
        "test_manifest",
        "test_parquet",
        "rows",
        "pair_lineage_sha256",
        "source_lineage_sha256",
        "rebuild_terminal",
        "access_proof",
    }
)
_SEAL_EVENT_LINEAGE_KEYS = frozenset(
    {"path", "sha256", "schema_version", "decision", "created_utc", "content_binding_sha256"}
)
_TEST_MANIFEST_LINEAGE_KEYS = frozenset(
    {
        "path",
        "sha256",
        "schema_version",
        "manifest_variant",
        "entry_run_id",
        "instrument",
    }
)
_ACCESS_PROOF_KEYS = frozenset(
    {
        "seal_event_bytes_hash_validated",
        "test_dataset_bytes_read",
        "test_manifest_bytes_read",
        "test_metrics_read",
        "test_paths_resolved_or_statted",
    }
)
_PRETEST_GUARD_KEYS = frozenset(
    {
        "schema_version",
        "decision",
        "created_utc",
        "guard_path",
        "entry_run_id",
        "dataset_dir",
        "split",
        "access_policy",
        "disclosure_count",
        "test_boundary_utc",
        "test_accessed",
        "test_materialized",
        "train_manifest",
        "train_parquet",
        "val_manifest",
        "val_parquet",
        "dataset_build_proof",
        "full_input_liveness",
        "content_binding_sha256",
    }
)
_PRETEST_GUARD_EVENT_LINEAGE_KEYS = frozenset(
    {"path", "sha256", "schema_version", "decision", "created_utc", "content_binding_sha256"}
)
_PRETEST_GUARD_LINEAGE_KEYS = frozenset(
    {
        "schema_version",
        "verification_mode",
        "guard_event",
        "dataset_run_id",
        "dataset_dir",
        "split",
        "access_policy",
        "disclosure_count",
        "test_boundary_utc",
        "train_manifest",
        "train_parquet",
        "val_manifest",
        "val_parquet",
        "dataset_build_proof",
        "full_input_liveness",
        "access_proof",
    }
)
_PRETEST_GUARD_ACCESS_PROOF_KEYS = frozenset(
    {
        "guard_event_bytes_hash_validated",
        "test_dataset_bytes_read",
        "test_manifest_bytes_read",
        "test_metrics_read",
        "test_paths_resolved_or_statted",
    }
)


class PrefreezeTestSealLineageError(RuntimeError):
    """The immutable TEST seal or its metadata-only lineage is invalid."""


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: Any, *, label: str) -> str:
    observed = value if isinstance(value, str) else ""
    if len(observed) != 64 or any(char not in _SHA256_HEX for char in observed):
        raise PrefreezeTestSealLineageError(f"{label}: expected exact lowercase SHA-256")
    return observed


def _absolute_immutable_path(value: Any, *, label: str) -> Path:
    raw = str(value) if isinstance(value, (str, Path)) else ""
    path = Path(raw)
    if (
        not raw
        or not path.is_absolute()
        or str(path) != raw
        or any(part in {".", ".."} for part in path.parts)
        or any("latest" in part.lower() for part in path.parts)
    ):
        raise PrefreezeTestSealLineageError(
            f"{label}: expected one canonical absolute immutable path"
        )
    return path


def _mapping(value: Any, *, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or frozenset(value) != keys:
        raise PrefreezeTestSealLineageError(f"{label}: keys are not exact")
    return value


def _parse_exact_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise PrefreezeTestSealLineageError(
                    f"{label}: duplicate JSON key {key!r}"
                )
            out[key] = value
        return out

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    except PrefreezeTestSealLineageError:
        raise
    except Exception as exc:
        raise PrefreezeTestSealLineageError(f"{label}: invalid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise PrefreezeTestSealLineageError(f"{label}: root must be an object")
    return payload


def require_prefreeze_test_seal_lineage_metadata(
    value: Mapping[str, Any],
    *,
    expected_dataset_run_id: str | None = None,
    expected_dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate bundle-safe TEST lineage without touching any external path."""

    lineage = _mapping(value, keys=_LINEAGE_KEYS, label="TEST seal lineage")
    if lineage.get("schema_version") != PREFREEZE_TEST_SEAL_LINEAGE_SCHEMA_VERSION:
        raise PrefreezeTestSealLineageError("TEST seal lineage schema mismatch")
    if lineage.get("verification_mode") != PREFREEZE_TEST_SEAL_VERIFICATION_MODE:
        raise PrefreezeTestSealLineageError("TEST seal lineage verification mode mismatch")

    dataset_run_id = require_entry_run_id(lineage.get("dataset_run_id"))
    if expected_dataset_run_id is not None:
        expected_run_id = require_entry_run_id(expected_dataset_run_id)
        if dataset_run_id != expected_run_id:
            raise PrefreezeTestSealLineageError("TEST seal dataset run lineage mismatch")
    dataset_dir = _absolute_immutable_path(
        lineage.get("dataset_dir"), label="TEST seal dataset_dir"
    )
    if expected_dataset_dir is not None and str(dataset_dir) != str(expected_dataset_dir):
        raise PrefreezeTestSealLineageError("TEST seal dataset directory mismatch")
    if (
        lineage.get("split") != "test"
        or lineage.get("access_policy") != TEST_SEAL_ACCESS_POLICY
        or lineage.get("disclosure_count") != 0
    ):
        raise PrefreezeTestSealLineageError("TEST seal isolation contract mismatch")

    seal_event = _mapping(
        lineage.get("seal_event"),
        keys=_SEAL_EVENT_LINEAGE_KEYS,
        label="TEST seal event lineage",
    )
    seal_path = _absolute_immutable_path(
        seal_event.get("path"), label="TEST seal event path"
    )
    if (
        not seal_path.name.startswith(f"{TEST_SEAL_EVENT_PREFIX}_")
        or not seal_path.name.endswith(".json")
        or seal_path.parent == dataset_dir
        or dataset_dir in seal_path.parents
        or seal_event.get("schema_version") != TEST_SEAL_SCHEMA_VERSION
        or seal_event.get("decision") != TEST_SEAL_DECISION
        or not isinstance(seal_event.get("created_utc"), str)
        or not str(seal_event["created_utc"]).strip()
    ):
        raise PrefreezeTestSealLineageError("TEST seal event identity mismatch")
    _sha256(seal_event.get("sha256"), label="TEST seal event sha256")
    _sha256(
        seal_event.get("content_binding_sha256"),
        label="TEST seal event content binding sha256",
    )

    manifest = _mapping(
        lineage.get("test_manifest"),
        keys=_TEST_MANIFEST_LINEAGE_KEYS,
        label="sealed TEST manifest lineage",
    )
    manifest_path = _absolute_immutable_path(
        manifest.get("path"), label="sealed TEST manifest path"
    )
    parquet = _mapping(
        lineage.get("test_parquet"),
        keys=_ARTIFACT_KEYS,
        label="sealed TEST parquet lineage",
    )
    parquet_path = _absolute_immutable_path(
        parquet.get("path"), label="sealed TEST parquet path"
    )
    if (
        manifest_path.parent != dataset_dir
        or parquet_path.parent != dataset_dir
        or not manifest_path.name.endswith("_test.manifest.json")
        or not parquet_path.name.endswith("_test.parquet")
        or manifest_path.name.removesuffix(".manifest.json")
        != parquet_path.name.removesuffix(".parquet")
        or manifest.get("schema_version") != MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION
        or manifest.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE
        or manifest.get("entry_run_id") != dataset_run_id
        or manifest.get("instrument") != XAU_INSTRUMENT
    ):
        raise PrefreezeTestSealLineageError("sealed TEST artifact contract mismatch")
    _sha256(manifest.get("sha256"), label="sealed TEST manifest sha256")
    _sha256(parquet.get("sha256"), label="sealed TEST parquet sha256")
    rows = lineage.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise PrefreezeTestSealLineageError("sealed TEST rows must be a positive integer")
    _sha256(lineage.get("pair_lineage_sha256"), label="TEST pair lineage sha256")
    _sha256(lineage.get("source_lineage_sha256"), label="TEST source lineage sha256")

    rebuild = _mapping(
        lineage.get("rebuild_terminal"),
        keys=_REBUILD_TERMINAL_KEYS,
        label="TEST seal rebuild terminal lineage",
    )
    rebuild_path = _absolute_immutable_path(
        rebuild.get("path"), label="TEST seal rebuild terminal path"
    )
    if (
        not rebuild_path.name.startswith(f"{DATASET_REBUILD_TERMINAL_EVENT_PREFIX}_")
        or not rebuild_path.name.endswith(".json")
        or rebuild_path.parent == dataset_dir
        or dataset_dir in rebuild_path.parents
        or rebuild.get("schema_version") != DATASET_REBUILD_TERMINAL_SCHEMA_VERSION
        or rebuild.get("decision") != DATASET_REBUILD_TERMINAL_DECISION
    ):
        raise PrefreezeTestSealLineageError("TEST seal rebuild terminal identity mismatch")
    _sha256(rebuild.get("sha256"), label="TEST seal rebuild terminal sha256")
    _sha256(
        rebuild.get("content_binding_sha256"),
        label="TEST seal rebuild terminal content binding sha256",
    )

    access = _mapping(
        lineage.get("access_proof"),
        keys=_ACCESS_PROOF_KEYS,
        label="TEST seal access proof",
    )
    if access != {
        "seal_event_bytes_hash_validated": True,
        "test_dataset_bytes_read": False,
        "test_manifest_bytes_read": False,
        "test_metrics_read": False,
        "test_paths_resolved_or_statted": False,
    }:
        raise PrefreezeTestSealLineageError("TEST seal access proof is not exact")

    return json.loads(json.dumps(lineage, sort_keys=True, allow_nan=False))


def require_prefreeze_test_seal_lineage(
    seal_json: str | Path,
    seal_sha256: str,
    *,
    expected_dataset_run_id: str,
    expected_dataset_dir: str | Path,
) -> dict[str, Any]:
    """Hash/parse only the seal event; TEST artifacts and metrics stay opaque."""

    supplied_path = _absolute_immutable_path(seal_json, label="TEST seal event path")
    if supplied_path.is_symlink() or not supplied_path.is_file():
        raise PrefreezeTestSealLineageError("TEST seal event must be a regular file")
    resolved_path = supplied_path.resolve(strict=True)
    if resolved_path != supplied_path:
        raise PrefreezeTestSealLineageError("TEST seal event path must already be canonical")
    expected_sha = _sha256(seal_sha256, label="TEST seal event sha256")
    raw = resolved_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise PrefreezeTestSealLineageError("TEST seal event sha256 mismatch")
    seal = _parse_exact_json_object(raw, label="TEST seal event")
    _mapping(seal, keys=_SEAL_KEYS, label="TEST seal event")

    dataset_run_id = require_entry_run_id(expected_dataset_run_id)
    dataset_dir = _absolute_immutable_path(
        expected_dataset_dir, label="expected TEST dataset_dir"
    )
    if (
        seal.get("schema_version") != TEST_SEAL_SCHEMA_VERSION
        or seal.get("decision") != TEST_SEAL_DECISION
        or seal.get("seal_path") != str(resolved_path)
        or seal.get("entry_run_id") != dataset_run_id
        or seal.get("dataset_dir") != str(dataset_dir)
        or seal.get("split") != "test"
        or seal.get("access_policy") != TEST_SEAL_ACCESS_POLICY
        or seal.get("disclosure_count") != 0
        or not isinstance(seal.get("created_utc"), str)
        or not str(seal["created_utc"]).strip()
    ):
        raise PrefreezeTestSealLineageError("TEST seal event identity/access mismatch")

    manifest = _mapping(
        seal.get("manifest"), keys=_ARTIFACT_KEYS, label="TEST seal manifest"
    )
    parquet = _mapping(
        seal.get("parquet"), keys=_ARTIFACT_KEYS, label="TEST seal parquet"
    )
    manifest_contract = _mapping(
        seal.get("manifest_contract"),
        keys=_MANIFEST_CONTRACT_KEYS,
        label="TEST seal manifest contract",
    )
    if not isinstance(seal.get("pair_lineage"), Mapping) or not isinstance(
        seal.get("source_lineage"), Mapping
    ):
        raise PrefreezeTestSealLineageError("TEST seal source lineage objects are missing")
    pair_sha = _sha256(
        seal.get("pair_lineage_sha256"), label="TEST seal pair lineage sha256"
    )
    source_sha = _sha256(
        seal.get("source_lineage_sha256"), label="TEST seal source lineage sha256"
    )
    if (
        _canonical_json_sha256(seal["pair_lineage"]) != pair_sha
        or _canonical_json_sha256(seal["source_lineage"]) != source_sha
    ):
        raise PrefreezeTestSealLineageError("TEST seal source lineage hash mismatch")
    content = {key: seal[key] for key in sorted(_SEAL_KEYS - {"content_binding_sha256"})}
    content_sha = _sha256(
        seal.get("content_binding_sha256"), label="TEST seal content binding sha256"
    )
    if _canonical_json_sha256(content) != content_sha:
        raise PrefreezeTestSealLineageError("TEST seal canonical content binding mismatch")
    rebuild = _mapping(
        seal.get("rebuild_terminal"),
        keys=_REBUILD_TERMINAL_KEYS,
        label="TEST seal rebuild terminal",
    )

    lineage = {
        "schema_version": PREFREEZE_TEST_SEAL_LINEAGE_SCHEMA_VERSION,
        "verification_mode": PREFREEZE_TEST_SEAL_VERIFICATION_MODE,
        "seal_event": {
            "path": str(resolved_path),
            "sha256": expected_sha,
            "schema_version": seal["schema_version"],
            "decision": seal["decision"],
            "created_utc": seal["created_utc"],
            "content_binding_sha256": content_sha,
        },
        "dataset_run_id": dataset_run_id,
        "dataset_dir": str(dataset_dir),
        "split": "test",
        "access_policy": TEST_SEAL_ACCESS_POLICY,
        "disclosure_count": 0,
        "test_manifest": {
            "path": manifest.get("path"),
            "sha256": manifest.get("sha256"),
            "schema_version": manifest_contract.get("schema_version"),
            "manifest_variant": manifest_contract.get("manifest_variant"),
            "entry_run_id": manifest_contract.get("entry_run_id"),
            "instrument": manifest_contract.get("instrument"),
        },
        "test_parquet": {
            "path": parquet.get("path"),
            "sha256": parquet.get("sha256"),
        },
        "rows": seal.get("rows"),
        "pair_lineage_sha256": pair_sha,
        "source_lineage_sha256": source_sha,
        "rebuild_terminal": dict(rebuild),
        "access_proof": {
            "seal_event_bytes_hash_validated": True,
            "test_dataset_bytes_read": False,
            "test_manifest_bytes_read": False,
            "test_metrics_read": False,
            "test_paths_resolved_or_statted": False,
        },
    }
    return require_prefreeze_test_seal_lineage_metadata(
        lineage,
        expected_dataset_run_id=dataset_run_id,
        expected_dataset_dir=dataset_dir,
    )


def _require_pretest_guard_artifact(
    value: Any,
    *,
    dataset_dir: Path,
    split: str,
    manifest: bool,
    label: str,
) -> dict[str, Any]:
    artifact = _mapping(value, keys=_ARTIFACT_KEYS, label=label)
    path = _absolute_immutable_path(artifact.get("path"), label=f"{label} path")
    expected_suffix = f"_{split}.manifest.json" if manifest else f"_{split}.parquet"
    if path.parent != dataset_dir or not path.name.endswith(expected_suffix):
        raise PrefreezeTestSealLineageError(f"{label}: path is outside its sealed split")
    return {"path": str(path), "sha256": _sha256(artifact.get("sha256"), label=f"{label} sha256")}


def require_pretest_test_guard_lineage_metadata(
    value: Mapping[str, Any],
    *,
    expected_dataset_run_id: str | None = None,
    expected_dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate an unopened-TEST lineage without touching TEST or any artifact.

    This is deliberately metadata-only.  The event holds only TRAIN/VAL and
    build-proof hashes; it contains no TEST filename, hash, row count or
    statistics.  A caller that needs a physical TEST evaluation must use
    :func:`require_prefreeze_test_seal_lineage_metadata` instead.
    """

    lineage = _mapping(value, keys=_PRETEST_GUARD_LINEAGE_KEYS, label="pre-TEST guard lineage")
    if (
        lineage.get("schema_version") != PRETEST_TEST_GUARD_LINEAGE_SCHEMA_VERSION
        or lineage.get("verification_mode") != PRETEST_TEST_GUARD_VERIFICATION_MODE
    ):
        raise PrefreezeTestSealLineageError("pre-TEST guard lineage schema mismatch")
    dataset_run_id = require_entry_run_id(lineage.get("dataset_run_id"))
    if expected_dataset_run_id is not None and dataset_run_id != require_entry_run_id(expected_dataset_run_id):
        raise PrefreezeTestSealLineageError("pre-TEST guard dataset run lineage mismatch")
    dataset_dir = _absolute_immutable_path(lineage.get("dataset_dir"), label="pre-TEST guard dataset_dir")
    if expected_dataset_dir is not None and str(dataset_dir) != str(expected_dataset_dir):
        raise PrefreezeTestSealLineageError("pre-TEST guard dataset directory mismatch")
    if (
        lineage.get("split") != "test"
        or lineage.get("access_policy") != PRETEST_TEST_GUARD_ACCESS_POLICY
        or lineage.get("disclosure_count") != 0
        or not isinstance(lineage.get("test_boundary_utc"), str)
        or not str(lineage["test_boundary_utc"]).strip()
    ):
        raise PrefreezeTestSealLineageError("pre-TEST guard isolation contract mismatch")

    event = _mapping(
        lineage.get("guard_event"),
        keys=_PRETEST_GUARD_EVENT_LINEAGE_KEYS,
        label="pre-TEST guard event lineage",
    )
    event_path = _absolute_immutable_path(event.get("path"), label="pre-TEST guard event path")
    if (
        not event_path.name.startswith(f"{PRETEST_TEST_GUARD_EVENT_PREFIX}_")
        or not event_path.name.endswith(".json")
        or event_path.parent == dataset_dir
        or dataset_dir in event_path.parents
        or event.get("schema_version") != PRETEST_TEST_GUARD_SCHEMA_VERSION
        or event.get("decision") != PRETEST_TEST_GUARD_DECISION
        or not isinstance(event.get("created_utc"), str)
        or not str(event["created_utc"]).strip()
    ):
        raise PrefreezeTestSealLineageError("pre-TEST guard event identity mismatch")
    normalized_event = {
        "path": str(event_path),
        "sha256": _sha256(event.get("sha256"), label="pre-TEST guard event sha256"),
        "schema_version": event["schema_version"],
        "decision": event["decision"],
        "created_utc": event["created_utc"],
        "content_binding_sha256": _sha256(
            event.get("content_binding_sha256"),
            label="pre-TEST guard event content binding sha256",
        ),
    }
    train_manifest = _require_pretest_guard_artifact(
        lineage.get("train_manifest"), dataset_dir=dataset_dir, split="train", manifest=True,
        label="pre-TEST TRAIN manifest",
    )
    train_parquet = _require_pretest_guard_artifact(
        lineage.get("train_parquet"), dataset_dir=dataset_dir, split="train", manifest=False,
        label="pre-TEST TRAIN parquet",
    )
    val_manifest = _require_pretest_guard_artifact(
        lineage.get("val_manifest"), dataset_dir=dataset_dir, split="val", manifest=True,
        label="pre-TEST VAL manifest",
    )
    val_parquet = _require_pretest_guard_artifact(
        lineage.get("val_parquet"), dataset_dir=dataset_dir, split="val", manifest=False,
        label="pre-TEST VAL parquet",
    )
    proof = _mapping(
        lineage.get("dataset_build_proof"), keys=_ARTIFACT_KEYS, label="pre-TEST dataset build proof"
    )
    proof_path = _absolute_immutable_path(proof.get("path"), label="pre-TEST dataset build proof path")
    if proof_path.parent != dataset_dir or proof_path.name != "DATASET_BUILD_PROOF.json":
        raise PrefreezeTestSealLineageError("pre-TEST dataset build proof path mismatch")
    dataset_build_proof = {
        "path": str(proof_path),
        "sha256": _sha256(proof.get("sha256"), label="pre-TEST dataset build proof sha256"),
    }
    liveness = _mapping(
        lineage.get("full_input_liveness"),
        keys=_ARTIFACT_KEYS,
        label="pre-TEST full-input liveness proof",
    )
    liveness_path = _absolute_immutable_path(
        liveness.get("path"), label="pre-TEST full-input liveness proof path"
    )
    if (
        liveness_path.parent != dataset_dir
        or not liveness_path.name.startswith("ENTRY_FULL_INPUT_LIVENESS_")
        or not liveness_path.name.endswith(".json")
    ):
        raise PrefreezeTestSealLineageError("pre-TEST full-input liveness proof path mismatch")
    full_input_liveness = {
        "path": str(liveness_path),
        "sha256": _sha256(
            liveness.get("sha256"), label="pre-TEST full-input liveness proof sha256"
        ),
    }
    access = _mapping(
        lineage.get("access_proof"),
        keys=_PRETEST_GUARD_ACCESS_PROOF_KEYS,
        label="pre-TEST guard access proof",
    )
    if access != {
        "guard_event_bytes_hash_validated": True,
        "test_dataset_bytes_read": False,
        "test_manifest_bytes_read": False,
        "test_metrics_read": False,
        "test_paths_resolved_or_statted": False,
    }:
        raise PrefreezeTestSealLineageError("pre-TEST guard access proof is not exact")
    return {
        "schema_version": PRETEST_TEST_GUARD_LINEAGE_SCHEMA_VERSION,
        "verification_mode": PRETEST_TEST_GUARD_VERIFICATION_MODE,
        "guard_event": normalized_event,
        "dataset_run_id": dataset_run_id,
        "dataset_dir": str(dataset_dir),
        "split": "test",
        "access_policy": PRETEST_TEST_GUARD_ACCESS_POLICY,
        "disclosure_count": 0,
        "test_boundary_utc": lineage["test_boundary_utc"],
        "train_manifest": train_manifest,
        "train_parquet": train_parquet,
        "val_manifest": val_manifest,
        "val_parquet": val_parquet,
        "dataset_build_proof": dataset_build_proof,
        "full_input_liveness": full_input_liveness,
        "access_proof": dict(access),
    }


def require_pretest_test_guard_lineage(
    guard_json: str | Path,
    guard_sha256: str,
    *,
    expected_dataset_run_id: str,
    expected_dataset_dir: str | Path,
) -> dict[str, Any]:
    """Hash/parse a pre-TEST guard event; no TEST path is ever present or read."""

    supplied_path = _absolute_immutable_path(guard_json, label="pre-TEST guard event path")
    if supplied_path.is_symlink() or not supplied_path.is_file():
        raise PrefreezeTestSealLineageError("pre-TEST guard event must be a regular file")
    resolved_path = supplied_path.resolve(strict=True)
    if resolved_path != supplied_path:
        raise PrefreezeTestSealLineageError("pre-TEST guard event path must already be canonical")
    expected_sha = _sha256(guard_sha256, label="pre-TEST guard event sha256")
    raw = resolved_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise PrefreezeTestSealLineageError("pre-TEST guard event sha256 mismatch")
    guard = _parse_exact_json_object(raw, label="pre-TEST guard event")
    _mapping(guard, keys=_PRETEST_GUARD_KEYS, label="pre-TEST guard event")
    dataset_run_id = require_entry_run_id(expected_dataset_run_id)
    dataset_dir = _absolute_immutable_path(expected_dataset_dir, label="expected pre-TEST dataset_dir")
    if (
        guard.get("schema_version") != PRETEST_TEST_GUARD_SCHEMA_VERSION
        or guard.get("decision") != PRETEST_TEST_GUARD_DECISION
        or guard.get("guard_path") != str(resolved_path)
        or guard.get("entry_run_id") != dataset_run_id
        or guard.get("dataset_dir") != str(dataset_dir)
        or guard.get("split") != "test"
        or guard.get("access_policy") != PRETEST_TEST_GUARD_ACCESS_POLICY
        or guard.get("disclosure_count") != 0
        or guard.get("test_accessed") is not False
        or guard.get("test_materialized") is not False
        or not isinstance(guard.get("test_boundary_utc"), str)
        or not str(guard["test_boundary_utc"]).strip()
        or not isinstance(guard.get("created_utc"), str)
        or not str(guard["created_utc"]).strip()
    ):
        raise PrefreezeTestSealLineageError("pre-TEST guard event identity/access mismatch")
    content = {key: guard[key] for key in sorted(_PRETEST_GUARD_KEYS - {"content_binding_sha256"})}
    content_sha = _sha256(
        guard.get("content_binding_sha256"), label="pre-TEST guard content binding sha256"
    )
    if _canonical_json_sha256(content) != content_sha:
        raise PrefreezeTestSealLineageError("pre-TEST guard canonical content binding mismatch")
    lineage = {
        "schema_version": PRETEST_TEST_GUARD_LINEAGE_SCHEMA_VERSION,
        "verification_mode": PRETEST_TEST_GUARD_VERIFICATION_MODE,
        "guard_event": {
            "path": str(resolved_path),
            "sha256": expected_sha,
            "schema_version": guard["schema_version"],
            "decision": guard["decision"],
            "created_utc": guard["created_utc"],
            "content_binding_sha256": content_sha,
        },
        "dataset_run_id": dataset_run_id,
        "dataset_dir": str(dataset_dir),
        "split": "test",
        "access_policy": PRETEST_TEST_GUARD_ACCESS_POLICY,
        "disclosure_count": 0,
        "test_boundary_utc": guard["test_boundary_utc"],
        "train_manifest": guard["train_manifest"],
        "train_parquet": guard["train_parquet"],
        "val_manifest": guard["val_manifest"],
        "val_parquet": guard["val_parquet"],
        "dataset_build_proof": guard["dataset_build_proof"],
        "full_input_liveness": guard["full_input_liveness"],
        "access_proof": {
            "guard_event_bytes_hash_validated": True,
            "test_dataset_bytes_read": False,
            "test_manifest_bytes_read": False,
            "test_metrics_read": False,
            "test_paths_resolved_or_statted": False,
        },
    }
    return require_pretest_test_guard_lineage_metadata(
        lineage,
        expected_dataset_run_id=dataset_run_id,
        expected_dataset_dir=dataset_dir,
    )


def require_pretest_or_prefreeze_test_guard_lineage_metadata(
    value: Mapping[str, Any],
    *,
    expected_dataset_run_id: str | None = None,
    expected_dataset_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Accept either legacy physical TEST sealing or strict pre-TEST guarding.

    Consumers that evaluate TEST must call the legacy physical-seal validator
    directly.  TRAIN/VAL-only control-plane consumers may use this union.
    """

    if not isinstance(value, Mapping):
        raise PrefreezeTestSealLineageError("TEST guard lineage must be a mapping")
    schema = value.get("schema_version")
    if schema == PREFREEZE_TEST_SEAL_LINEAGE_SCHEMA_VERSION:
        return require_prefreeze_test_seal_lineage_metadata(
            value,
            expected_dataset_run_id=expected_dataset_run_id,
            expected_dataset_dir=expected_dataset_dir,
        )
    if schema == PRETEST_TEST_GUARD_LINEAGE_SCHEMA_VERSION:
        return require_pretest_test_guard_lineage_metadata(
            value,
            expected_dataset_run_id=expected_dataset_run_id,
            expected_dataset_dir=expected_dataset_dir,
        )
    raise PrefreezeTestSealLineageError("unrecognized TEST guard lineage schema")


def require_pretest_or_prefreeze_test_guard_lineage(
    event_json: str | Path,
    event_sha256: str,
    *,
    expected_dataset_run_id: str,
    expected_dataset_dir: str | Path,
) -> dict[str, Any]:
    """Read only the supplied control-plane event and dispatch by its schema."""

    supplied_path = _absolute_immutable_path(event_json, label="TEST guard event path")
    if supplied_path.is_symlink() or not supplied_path.is_file():
        raise PrefreezeTestSealLineageError("TEST guard event must be a regular file")
    raw = supplied_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha256(event_sha256, label="TEST guard event sha256"):
        raise PrefreezeTestSealLineageError("TEST guard event sha256 mismatch")
    event = _parse_exact_json_object(raw, label="TEST guard event")
    if event.get("schema_version") == TEST_SEAL_SCHEMA_VERSION:
        return require_prefreeze_test_seal_lineage(
            supplied_path,
            event_sha256,
            expected_dataset_run_id=expected_dataset_run_id,
            expected_dataset_dir=expected_dataset_dir,
        )
    if event.get("schema_version") == PRETEST_TEST_GUARD_SCHEMA_VERSION:
        return require_pretest_test_guard_lineage(
            supplied_path,
            event_sha256,
            expected_dataset_run_id=expected_dataset_run_id,
            expected_dataset_dir=expected_dataset_dir,
        )
    raise PrefreezeTestSealLineageError("unrecognized TEST guard event schema")

REQUIRED_PROOF_CHECKS = (
    "rebuild chain terminal is exact green",
    "rebuild preflight is exact ready",
    "full-input liveness is exact pass",
    "pretrain audit is exact pass",
    "TRAIN/VAL split identities are exact and XAU-bound",
    "TEST is completion-bound but remains byte-opaque",
)

SIDE_EFFECT_KEYS = (
    "dataset_rebuild",
    "training",
    "replay",
    "iql_distillation",
    "shadow",
    "live",
)
