"""Materialize one strict no-TEST-access guard for a sealed TRAIN/VAL dataset.

The event contains only TRAIN/VAL artifacts and an already-passed full-input
liveness report.  It never accepts, resolves, stats, hashes, reads or names a
TEST artifact.  Physical TEST evaluation remains governed by the legacy
untouched-TEST seal contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    PRETEST_TEST_GUARD_ACCESS_POLICY,
    PRETEST_TEST_GUARD_DECISION,
    PRETEST_TEST_GUARD_EVENT_PREFIX,
    PRETEST_TEST_GUARD_SCHEMA_VERSION,
    require_pretest_test_guard_lineage,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"PRETEST_GUARD_{label}_MISSING")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"PRETEST_GUARD_{label}_INVALID") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"PRETEST_GUARD_{label}_INVALID")
    return payload


def _regular_inside(path: str | Path, *, dataset_dir: Path, label: str) -> Path:
    candidate = Path(path)
    if (
        not candidate.is_absolute()
        or candidate != candidate.resolve(strict=False)
        or candidate.is_symlink()
        or not candidate.is_file()
        or candidate.parent != dataset_dir
        or "test" in candidate.name.lower()
    ):
        raise RuntimeError(f"PRETEST_GUARD_{label}_PATH_INVALID")
    return candidate


def _require_sha256(value: object, *, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise RuntimeError(f"PRETEST_GUARD_{label}_SHA256_INVALID")
    return text


def _split_extra(
    manifest: Mapping[str, Any],
    *,
    split: str,
    dataset_dir: Path,
    parquet: Path,
) -> tuple[str, str]:
    extra = manifest.get("extra")
    if not isinstance(extra, Mapping):
        raise RuntimeError("PRETEST_GUARD_SPLIT_EXTRA_INVALID")
    run_id = require_entry_run_id(extra.get("entry_run_id"))
    test_guard = extra.get("pretest_test_guard")
    if (
        extra.get("pretest_only") is not True
        or not isinstance(test_guard, Mapping)
        or test_guard.get("test_accessed") is not False
        or not isinstance(test_guard.get("test_boundary_utc"), str)
        or not str(test_guard["test_boundary_utc"]).strip()
        or manifest.get("output_data_path") != str(parquet)
        or not str(parquet.name).endswith(f"_{split}.parquet")
    ):
        raise RuntimeError("PRETEST_GUARD_SPLIT_MANIFEST_CONTRACT_INVALID")
    return run_id, str(test_guard["test_boundary_utc"])


def _validate_liveness(
    report: Mapping[str, Any],
    *,
    dataset_dir: Path,
    artifacts: Mapping[str, tuple[Path, str]],
) -> None:
    if report.get("decision") != "PASS" or report.get("dataset_dir") != str(dataset_dir):
        raise RuntimeError("PRETEST_GUARD_FULL_INPUT_LIVENESS_NOT_PASS")
    bindings = report.get("input_bindings")
    if not isinstance(bindings, Mapping):
        raise RuntimeError("PRETEST_GUARD_FULL_INPUT_LIVENESS_BINDINGS_INVALID")
    manifests = bindings.get("split_manifests")
    fullscan = bindings.get("fullscan_proof")
    if not isinstance(manifests, Mapping) or not isinstance(fullscan, Mapping):
        raise RuntimeError("PRETEST_GUARD_FULL_INPUT_LIVENESS_BINDINGS_INVALID")
    if set(manifests) != {"train", "val"} or set(fullscan) != {"train", "val"}:
        raise RuntimeError("PRETEST_GUARD_FULL_INPUT_LIVENESS_SPLITS_INVALID")
    for split in ("train", "val"):
        manifest, manifest_sha = artifacts[f"{split}_manifest"]
        parquet, parquet_sha = artifacts[f"{split}_parquet"]
        manifest_binding = manifests[split]
        scan_binding = fullscan[split]
        if (
            not isinstance(manifest_binding, Mapping)
            or not isinstance(scan_binding, Mapping)
            or manifest_binding.get("path") != str(manifest)
            or manifest_binding.get("sha256") != manifest_sha
            or manifest_binding.get("observed_sha256") != manifest_sha
            or scan_binding.get("parquet_path") != str(parquet)
            or scan_binding.get("parquet_sha256") != parquet_sha
            or scan_binding.get("fullscan") is not True
            or scan_binding.get("scan_complete") is not True
            or int(scan_binding.get("scanned_rows") or 0) <= 0
            or int(scan_binding.get("total_rows") or 0)
            != int(scan_binding.get("scanned_rows") or 0)
        ):
            raise RuntimeError("PRETEST_GUARD_FULL_INPUT_LIVENESS_BINDING_MISMATCH")


def materialize_pretest_test_guard(
    *,
    dataset_dir: Path,
    train_manifest: Path,
    train_parquet: Path,
    train_parquet_sha256: str,
    val_manifest: Path,
    val_parquet: Path,
    val_parquet_sha256: str,
    dataset_build_proof: Path,
    full_input_liveness: Path,
    out_json: Path,
    created_utc: str,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve(strict=True)
    if dataset_dir.is_symlink() or not dataset_dir.is_dir():
        raise RuntimeError("PRETEST_GUARD_DATASET_DIR_INVALID")
    train_manifest = _regular_inside(train_manifest, dataset_dir=dataset_dir, label="TRAIN_MANIFEST")
    train_parquet = _regular_inside(train_parquet, dataset_dir=dataset_dir, label="TRAIN_PARQUET")
    val_manifest = _regular_inside(val_manifest, dataset_dir=dataset_dir, label="VAL_MANIFEST")
    val_parquet = _regular_inside(val_parquet, dataset_dir=dataset_dir, label="VAL_PARQUET")
    dataset_build_proof = _regular_inside(
        dataset_build_proof, dataset_dir=dataset_dir, label="DATASET_BUILD_PROOF"
    )
    if dataset_build_proof.name != "DATASET_BUILD_PROOF.json":
        raise RuntimeError("PRETEST_GUARD_DATASET_BUILD_PROOF_PATH_INVALID")
    full_input_liveness = _regular_inside(
        full_input_liveness, dataset_dir=dataset_dir, label="FULL_INPUT_LIVENESS"
    )
    if not full_input_liveness.name.startswith("ENTRY_FULL_INPUT_LIVENESS_"):
        raise RuntimeError("PRETEST_GUARD_FULL_INPUT_LIVENESS_PATH_INVALID")
    for path in (train_manifest, val_manifest, dataset_build_proof, full_input_liveness):
        if path.stat().st_size <= 0:
            raise RuntimeError("PRETEST_GUARD_CONTROL_PLANE_ARTIFACT_EMPTY")
    if (
        not out_json.is_absolute()
        or out_json.exists()
        or out_json.is_symlink()
        or not out_json.parent.is_dir()
        or out_json.parent == dataset_dir
        or dataset_dir in out_json.parents
        or not out_json.name.startswith(f"{PRETEST_TEST_GUARD_EVENT_PREFIX}_")
        or not out_json.name.endswith(".json")
    ):
        raise RuntimeError("PRETEST_GUARD_OUTPUT_PATH_INVALID")
    train_payload = _read_json(train_manifest, label="TRAIN_MANIFEST")
    val_payload = _read_json(val_manifest, label="VAL_MANIFEST")
    proof_payload = _read_json(dataset_build_proof, label="DATASET_BUILD_PROOF")
    train_run_id, train_boundary = _split_extra(
        train_payload, split="train", dataset_dir=dataset_dir, parquet=train_parquet
    )
    val_run_id, val_boundary = _split_extra(
        val_payload, split="val", dataset_dir=dataset_dir, parquet=val_parquet
    )
    proof_guard = proof_payload.get("pretest_test_guard")
    proof_run_id = require_entry_run_id(proof_payload.get("entry_run_id"))
    if (
        proof_payload.get("pretest_only") is not True
        or not isinstance(proof_guard, Mapping)
        or proof_guard.get("test_accessed") is not False
        or proof_guard.get("test_boundary_utc") != train_boundary
        or train_run_id != val_run_id != ""
        or train_run_id != proof_run_id
        or train_boundary != val_boundary
    ):
        raise RuntimeError("PRETEST_GUARD_DATASET_PROOF_CONTRACT_INVALID")
    artifacts = {
        "train_manifest": (train_manifest, _sha256_file(train_manifest)),
        "train_parquet": (train_parquet, _require_sha256(train_parquet_sha256, label="TRAIN_PARQUET")),
        "val_manifest": (val_manifest, _sha256_file(val_manifest)),
        "val_parquet": (val_parquet, _require_sha256(val_parquet_sha256, label="VAL_PARQUET")),
    }
    report = _read_json(full_input_liveness, label="FULL_INPUT_LIVENESS")
    _validate_liveness(report, dataset_dir=dataset_dir, artifacts=artifacts)
    payload: dict[str, Any] = {
        "schema_version": PRETEST_TEST_GUARD_SCHEMA_VERSION,
        "decision": PRETEST_TEST_GUARD_DECISION,
        "created_utc": created_utc,
        "guard_path": str(out_json),
        "entry_run_id": train_run_id,
        "dataset_dir": str(dataset_dir),
        "split": "test",
        "access_policy": PRETEST_TEST_GUARD_ACCESS_POLICY,
        "disclosure_count": 0,
        "test_boundary_utc": train_boundary,
        "test_accessed": False,
        "test_materialized": False,
        "train_manifest": {"path": str(train_manifest), "sha256": artifacts["train_manifest"][1]},
        "train_parquet": {"path": str(train_parquet), "sha256": artifacts["train_parquet"][1]},
        "val_manifest": {"path": str(val_manifest), "sha256": artifacts["val_manifest"][1]},
        "val_parquet": {"path": str(val_parquet), "sha256": artifacts["val_parquet"][1]},
        "dataset_build_proof": {
            "path": str(dataset_build_proof),
            "sha256": _sha256_file(dataset_build_proof),
        },
        "full_input_liveness": {
            "path": str(full_input_liveness),
            "sha256": _sha256_file(full_input_liveness),
        },
    }
    payload["content_binding_sha256"] = _canonical_json_sha256(payload)
    raw = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(out_json, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise RuntimeError("PRETEST_GUARD_OUTPUT_EXISTS") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
    event_sha256 = _sha256_file(out_json)
    lineage = require_pretest_test_guard_lineage(
        out_json,
        event_sha256,
        expected_dataset_run_id=train_run_id,
        expected_dataset_dir=dataset_dir,
    )
    return {"path": str(out_json), "sha256": event_sha256, "lineage": lineage}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--train-manifest-json", required=True)
    parser.add_argument("--train-parquet", required=True)
    parser.add_argument("--train-parquet-sha256", required=True)
    parser.add_argument("--val-manifest-json", required=True)
    parser.add_argument("--val-parquet", required=True)
    parser.add_argument("--val-parquet-sha256", required=True)
    parser.add_argument("--dataset-build-proof-json", required=True)
    parser.add_argument("--full-input-liveness-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    result = materialize_pretest_test_guard(
        dataset_dir=Path(args.dataset_dir),
        train_manifest=Path(args.train_manifest_json),
        train_parquet=Path(args.train_parquet),
        train_parquet_sha256=args.train_parquet_sha256,
        val_manifest=Path(args.val_manifest_json),
        val_parquet=Path(args.val_parquet),
        val_parquet_sha256=args.val_parquet_sha256,
        dataset_build_proof=Path(args.dataset_build_proof_json),
        full_input_liveness=Path(args.full_input_liveness_json),
        out_json=Path(args.out_json),
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
